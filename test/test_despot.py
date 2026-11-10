"""VapourSynth tests of the DeSpot plugin.

Usage: python test_despot.py [path/to/DeSpot.(dll|so|dylib) [avs_raw_dir]]
Without a path the autoloaded plugin (e.g. an installed wheel) is tested.

- every parameter case with every available asm_opt gives the same output as asm_opt=1 (C)
- 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0, 4:1:0 and Gray, also with sizes that are no multiple of 8 or 16
- random frame request order gives the same output as sequential order
- with avs_raw_dir (new<i>.raw written by avs_compare.py for the same DLL and gen_clip defaults),
  the VapourSynth output must equal the AviSynth+ output
Exit code 1 on any failure.
"""
import os
import platform
import random
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_clip import make_frames  # noqa: E402
from avs_compare import CASES  # noqa: E402

core = vs.core
FORMATS = {"420": vs.YUV420P8, "422": vs.YUV422P8, "444": vs.YUV444P8, "411": vs.YUV411P8, "440": vs.YUV440P8,
           "410": vs.YUV410P8, "gray": vs.GRAY8}


def to_clip(frames, fmt):
    h, w = frames[0][0].shape
    blank = core.std.BlankClip(format=fmt, width=w, height=h, length=len(frames), fpsnum=25, fpsden=1)

    def sel(n, f):
        out = f.copy()
        for p, arr in enumerate(frames[n]):
            np.asarray(out[p])[:] = arr
        return out

    return core.std.ModifyFrame(blank, blank, sel)


def mask_clip(src):
    w, h = src.width, src.height
    y = np.full((h, w), 255, np.uint8)
    y[100:h - 100, 100:w - 100] = 0
    m = core.std.BlankClip(format=vs.GRAY8, width=w, height=h, length=src.num_frames)
    return core.std.ModifyFrame(m, m, lambda n, f: _fill(f, y))


def _fill(f, y):
    out = f.copy()
    np.asarray(out[0])[:] = y
    return out


def parse(args, src):
    kw = {}
    for part in filter(None, (a.strip() for a in args.split(","))):
        k, v = (t.strip() for t in part.split("="))
        if v in ("true", "false"):
            kw[k] = int(v == "true")
        elif v == "mask":
            kw[k] = mask_clip(src)
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
        core.despot.DeSpot(core.std.BlankClip(format=vs.YUV420P16)).get_frame(0)
        fail("no error for 16 bit input")
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
