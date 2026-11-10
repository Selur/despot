"""VapourSynth tests of the DeSpot plugin.

Usage: python test_despot.py [path/to/DeSpot.(dll|so|dylib) [avs_raw_dir]]
Without a path the autoloaded plugin (e.g. an installed wheel) is tested.

- every parameter case with every available asm_opt gives the same output as asm_opt=1 (C)
- 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0, 4:1:0 and Gray, 8 to 16 bit, also with sizes that are no multiple of 8 or 16
- random frame request order gives the same output as sequential order
- an 8 bit clip shifted to 16 bit gives the same spot/motion map (show=2) and outfile as the 8 bit clip
- with avs_raw_dir (new<i>.raw written by avs_compare.py for the same DLL and gen_clip defaults),
  the VapourSynth output must equal the AviSynth+ output
Exit code 1 on any failure.
"""
import os
import platform
import random
import sys
import tempfile

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_clip import make_frames  # noqa: E402
from avs_compare import CASES  # noqa: E402

core = vs.core
HIGH = {("420", 10): vs.YUV420P10, ("422", 12): vs.YUV422P12, ("444", 16): vs.YUV444P16, ("gray", 16): vs.GRAY16,
        ("420", 16): vs.YUV420P16, ("411", 9): None}
FORMATS = {"420": vs.YUV420P8, "422": vs.YUV422P8, "444": vs.YUV444P8, "411": vs.YUV411P8, "440": vs.YUV440P8,
           "410": vs.YUV410P8, "gray": vs.GRAY8}


def to_clip(frames, fmt, shift=0, low_bits=False):
    """8 bit test frames as a clip of format fmt: samples << shift, plus random low bits if low_bits."""
    h, w = frames[0][0].shape
    blank = core.std.BlankClip(format=fmt, width=w, height=h, length=len(frames), fpsnum=25, fpsden=1)
    if shift:
        rng = np.random.default_rng(99)
        frames = [[(p.astype(np.uint16) << shift) | (rng.integers(0, 1 << shift, p.shape, dtype=np.uint16)
                                                     if low_bits else 0) for p in planes] for planes in frames]

    def sel(n, f):
        out = f.copy()
        for p, arr in enumerate(frames[n]):
            np.asarray(out[p])[:] = arr
        return out

    return core.std.ModifyFrame(blank, blank, sel)


def mask_clip(src, bits=8):
    w, h = src.width, src.height
    y = np.full((h, w), (1 << bits) - 1, np.uint8 if bits == 8 else np.uint16)
    y[100:h - 100, 100:w - 100] = 0
    fmt = vs.GRAY8 if bits == 8 else vs.GRAY16
    m = core.std.BlankClip(format=fmt, width=w, height=h, length=src.num_frames)
    return core.std.ModifyFrame(m, m, lambda n, f: _fill(f, y))


def _fill(f, y):
    out = f.copy()
    np.asarray(out[0])[:] = y
    return out


def parse(args, src, mask_bits=8):
    kw = {}
    for part in filter(None, (a.strip() for a in args.split(","))):
        k, v = (t.strip() for t in part.split("="))
        if v in ("true", "false"):
            kw[k] = int(v == "true")
        elif v == "mask":
            kw[k] = mask_clip(src, mask_bits)
        else:
            kw[k] = int(v)
    return kw


def frame_bytes(f):
    return b"".join(np.asarray(f[p]).tobytes() for p in range(f.format.num_planes))


def render(clip, order=None):
    order = order if order is not None else range(clip.num_frames)
    out = {}
    for n in order:
        out[n] = frame_bytes(clip.get_frame(n))
    return [out[n] for n in range(clip.num_frames)]


def main():
    avs_dir = sys.argv[2] if len(sys.argv) > 2 else None
    if len(sys.argv) > 1:
        core.std.LoadPlugin(os.path.abspath(sys.argv[1]))
    if not hasattr(core, "despot"):
        sys.exit("DeSpot plugin not found")
    x86 = platform.machine().lower() in ("amd64", "x86_64", "i386", "i686", "x86")
    opts = [0, 2, 3] if x86 else [0]
    failures = 0

    def fail(msg):
        nonlocal failures
        failures += 1
        print("FAIL", msg)

    # errors for invalid input
    src = to_clip(make_frames(64, 48, 5, "420", 3), vs.YUV420P8)
    for bad in ({"p1": 256}, {"show": 3}, {"asm_opt": 9}):
        try:
            core.despot.DeSpot(src, **bad).get_frame(2)
            fail(f"no error for {bad}")
        except vs.Error:
            pass
    try:
        core.despot.DeSpot(core.std.BlankClip(format=vs.YUV420PS)).get_frame(0)
        fail("no error for float input")
    except vs.Error:
        pass

    setups = [("420", 320, 240), ("422", 320, 240), ("444", 317, 203), ("gray", 333, 121), ("420", 322, 182),
              ("411", 324, 120), ("440", 160, 122), ("410", 196, 100)]
    for ss, w, h in setups:
        src = to_clip(make_frames(w, h, 12 if (w, h) != (320, 240) else 30, ss, 7 if (w, h) != (320, 240) else 1),
                      FORMATS[ss])
        cases = CASES if (ss, w, h) == ("420", 320, 240) else CASES[::3]
        for i, args in enumerate(cases):
            kw = parse(args, src)
            ref = render(core.despot.DeSpot(src, asm_opt=1, **kw))
            for o in opts:
                try:
                    got = render(core.despot.DeSpot(src, asm_opt=o, **kw))
                except vs.Error as e:
                    if o == 3 and "not supported" in str(e):
                        continue
                    raise
                if got != ref:
                    fail(f"{ss} {w}x{h} [{args}] asm_opt={o} differs from C")
            if i % 5 == 0:
                order = list(range(src.num_frames))
                random.Random(i).shuffle(order)
                if render(core.despot.DeSpot(src, **kw), order) != ref:
                    fail(f"{ss} {w}x{h} [{args}] random frame order differs")
            if avs_dir and (ss, w, h) == ("420", 320, 240):
                raw = os.path.join(avs_dir, f"new{CASES.index(args)}.raw")
                if os.path.exists(raw) and open(raw, "rb").read() != b"".join(ref):
                    fail(f"[{args}] VapourSynth output differs from AviSynth+")
        print(f"{ss} {w}x{h}: {len(cases)} cases checked")

    # 9..16 bit: every instruction set against C (16 bit extmask on 16 bit clips)
    for (ss, bits), fmt in HIGH.items():
        if fmt is None:
            fmt = core.query_video_format(vs.YUV, vs.INTEGER, bits, 2, 0).id
        w, h = (320, 240) if ss == "420" else (324, 120) if ss == "411" else (317, 203) if ss == "444" else (256, 130)
        src = to_clip(make_frames(w, h, 12, ss, 11), fmt, bits - 8, low_bits=True)
        cases = CASES[::2] if (ss, bits) == ("420", 10) else CASES[::4]
        for i, args in enumerate(cases):
            kw = parse(args, src, 16 if bits == 16 else 8)
            ref = render(core.despot.DeSpot(src, asm_opt=1, **kw))
            for o in opts:
                try:
                    got = render(core.despot.DeSpot(src, asm_opt=o, **kw))
                except vs.Error as e:
                    if o == 3 and "not supported" in str(e):
                        continue
                    raise
                if got != ref:
                    fail(f"{ss} {bits} bit [{args}] asm_opt={o} differs from C")
            if i % 6 == 0:
                order = list(range(src.num_frames))
                random.Random(i).shuffle(order)
                if render(core.despot.DeSpot(src, **kw), order) != ref:
                    fail(f"{ss} {bits} bit [{args}] random frame order differs")
        print(f"{ss} {bits} bit {w}x{h}: {len(cases)} cases checked")

    # 8 bit content in 16 bit: the same detection as in 8 bit
    frames = make_frames(320, 240, 30, "420", 1)
    src8 = to_clip(frames, vs.YUV420P8)
    src16 = to_clip(frames, vs.YUV420P16, 8)
    maxdiff = 0
    for args in CASES:
        if "show" in args:
            continue
        kw8, kw16 = parse(args, src8), parse(args, src16, 16)
        m8 = core.despot.DeSpot(src8, show=2, **kw8)
        m16 = core.despot.DeSpot(src16, show=2, **kw16)
        for n in range(src8.num_frames):
            a = np.asarray(m8.get_frame(n)[0])
            b = np.asarray(m16.get_frame(n)[0]) >> 8
            if not np.array_equal(a, b):
                fail(f"[{args}] 16 bit spot/motion map differs from 8 bit in frame {n}")
                break
        o8 = core.despot.DeSpot(src8, **kw8)
        o16 = core.despot.DeSpot(src16, **kw16)
        for n in range(1, src8.num_frames - 1):
            a = np.asarray(o8.get_frame(n)[0]).astype(np.int32)
            b = np.asarray(o16.get_frame(n)[0]).astype(np.int32)
            maxdiff = max(maxdiff, int(np.abs(a * 256 - b).max()))
    print(f"16 bit maps identical to 8 bit; largest output difference {maxdiff / 256:.2f} (8 bit steps)")
    if maxdiff > 4 * 256:  # 8 bit truncates in every average, blur pass and luma fit
        fail(f"16 bit output differs from 8 bit by {maxdiff / 256:.2f} 8 bit steps")
    with tempfile.TemporaryDirectory() as tmp:
        f8, f16 = os.path.join(tmp, "a8.ass"), os.path.join(tmp, "a16.ass")
        render(core.despot.DeSpot(src8, outfile=f8))
        render(core.despot.DeSpot(src16, outfile=f16))
        core.clear_cache() if hasattr(core, "clear_cache") else None
        import gc
        gc.collect()
        if sorted(open(f8).read().splitlines()) != sorted(open(f16).read().splitlines()):
            fail("16 bit outfile differs from 8 bit")

    # threads: parallel rendering equals sequential rendering
    src = to_clip(make_frames(320, 240, 30, "420", 1), vs.YUV420P8)
    clip = core.despot.DeSpot(src, motpn=0, tsmooth=3)
    seq = render(clip)
    par = [None] * clip.num_frames
    for n, f in enumerate(clip.frames(prefetch=8)):
        par[n] = frame_bytes(f)
    if par != seq:
        fail("parallel rendering differs")

    print("all tests passed" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
