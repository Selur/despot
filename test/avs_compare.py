"""Compare two DeSpot AviSynth plugin DLLs frame by frame over a parameter matrix.

Usage: python avs_compare.py ref.dll new.dll clip [workdir]
clip: a .y4m file or any video file ffms2 opens (its index is written to workdir).
Environment: DESPOT_PRE filters for the source (e.g. "Trim(1000, -200).ConvertToYV12()"),
DESPOT_CASES comma separated case indices, DESPOT_EXTRA extra arguments for new.dll.
Needs a portable AviSynth+ (AVS_DIR, default Hybrid's) with avs2yuv64.exe and ffms2.dll.
Prints, for every parameter set, whether the outputs are identical and if not how many
frames / pixels differ.
"""
import os
import subprocess
import sys

import numpy as np

AVS_DIR = os.environ.get("AVS_DIR", r"F:\Hybrid\64bit\Avisynth")
AVS2YUV = os.path.join(AVS_DIR, "avs2yuv64.exe")
FFMS = os.path.join(AVS_DIR, "avisynthPlugins", "ffms2.dll")

CASES = [
    "",
    "show=1",
    "show=2",
    "show=2, show_chroma=true",
    "seg=0",
    "seg=1",
    "seg=0, show=1",
    "motpn=false",
    "motpn=false, seg=0",
    "motpn=false, seg=1",
    "motpn=false, seg=0, show=2",
    "median=true",
    "median=true, show=2",
    "ranked=false",
    "ranked=false, sign=1",
    "ranked=false, sign=-1",
    "ranked=false, sign=2",
    "ranked=false, sign=-2",
    "sign=1", "sign=-1", "sign=2", "sign=-2",
    "fitluma=false",
    "fitluma=true",
    "blur=0", "blur=2", "blur=3", "blur=4",
    "seg=0, blur=2", "seg=0, blur=3", "seg=0, blur=4",
    "tsmooth=5",
    "seg=0, tsmooth=5",
    "dilate=0", "dilate=3",
    "seg=0, dilate=3",
    "color=true",
    "interlaced=true",
    "interlaced=true, seg=0",
    "interlaced=true, motpn=false",
    "interlaced=true, color=true",
    "maxpts=10, minpts=2",
    "p1percent=0",
    "mscene=5",
    "merode=0",
    "mwidth=0, mheight=0",
    "pwidth=6, pheight=5",
    "p1=35, p2=14, pwidth=70, pheight=70, mthres=25, mwidth=20, mheight=15, merode=33, ranked=false, p1percent=0, dilate=0, fitluma=false, blur=0, motpn=false, seg=0",
    "p1=30, p2=12, mthres=20, dilate=2, fitluma=true, blur=2, seg=2",
    "extmask=mask",
]


NEW_EXTRA = os.environ.get("DESPOT_EXTRA", "")  # appended to the arguments of the second DLL
PRE = os.environ.get("DESPOT_PRE", "")  # filters applied to the source, e.g. "Trim(1000, -200).ConvertToYV12()"
SELECT = os.environ.get("DESPOT_CASES", "")  # comma separated case indices, default all
WORK = None


def source_lines(clip):
    if clip.lower().endswith(".y4m"):
        lines = f'src = FFVideoSource("{clip}", cache=false)\n'
    else:  # any video file; the index goes to the work directory, not next to the clip
        index = os.path.join(WORK, os.path.basename(clip) + ".ffindex")
        lines = f'src = FFVideoSource("{clip}", cachefile="{index}")\n'
    if PRE:
        lines += f"src = src.{PRE}\n"
    return lines


def probe(clip):
    """Width, height and frame size in bytes of the (8 bit) source after DESPOT_PRE."""
    script = os.path.join(WORK, "probe.avs")
    with open(script, "w") as fh:
        fh.write(f'LoadPlugin("{FFMS}")\n' + source_lines(clip) + "src\n")
    out = os.path.join(WORK, "probe.y4m")
    subprocess.run([AVS2YUV, "-nstdr", "-frames", "1", script, "-o", out], check=True, capture_output=True)
    with open(out, "rb") as fh:
        hdr = fh.readline().decode().split()
    w = int([t for t in hdr if t.startswith("W")][0][1:])
    h = int([t for t in hdr if t.startswith("H")][0][1:])
    cs = ([t for t in hdr if t.startswith("C")] or ["C420"])[0]
    sizes = {"C420": w * h * 3 // 2, "C422": w * h * 2, "C444": w * h * 3, "Cmono": w * h, "C411": w * h * 3 // 2}
    return w, h, next(v for k, v in sizes.items() if cs.startswith(k))


def run(dll, clip, args, out, extra=""):
    script = out + ".avs"
    with open(script, "w") as fh:
        fh.write(f'LoadPlugin("{FFMS}")\nLoadPlugin("{dll}")\n')
        fh.write(source_lines(clip))
        fh.write('mask = src.BlankClip(color_yuv=$000000).Letterbox(100, 100, 100, 100, color_yuv=$FF8080)\n')
        sep = ", " if args else ""
        fh.write(f"DeSpot(src{sep}{args})\n")
    r = subprocess.run([AVS2YUV, "-nstdr", "-raw", script, "-o", out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None, (r.stderr or r.stdout).strip()
    return np.fromfile(out, np.uint8), None


def main():
    global WORK
    ref_dll, new_dll, clip = (os.path.abspath(a) for a in sys.argv[1:4])
    work = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else os.path.join(os.path.dirname(clip), "avs_work")
    os.makedirs(work, exist_ok=True)
    WORK = work
    w, h, fsize = probe(clip)
    selected = [int(i) for i in SELECT.split(",")] if SELECT else range(len(CASES))
    bad = 0
    for i in selected:
        args = CASES[i]
        a, ea = run(ref_dll, clip, args, os.path.join(work, f"ref{i}.raw"))
        b, eb = run(new_dll, clip, args, os.path.join(work, f"new{i}.raw"), NEW_EXTRA)
        if a is None or b is None:
            print(f"[{i:2}] {args!r}: ERROR ref={ea} new={eb}")
            bad += 1
            continue
        if a.size != b.size:
            print(f"[{i:2}] {args!r}: SIZE {a.size} vs {b.size}")
            bad += 1
            continue
        if np.array_equal(a, b):
            print(f"[{i:2}] {args!r}: identical")
            continue
        bad += 1
        fa = a.reshape(-1, fsize)
        fb = b.reshape(-1, fsize)
        diff_frames = [n for n in range(fa.shape[0]) if not np.array_equal(fa[n], fb[n])]
        npx = int(np.count_nonzero(fa != fb))
        ya, yb = fa[:, : w * h].astype(np.int16), fb[:, : w * h].astype(np.int16)
        luma = int(np.count_nonzero(ya != yb))
        print(f"[{i:2}] {args!r}: DIFF frames={len(diff_frames)}/{fa.shape[0]} pixels={npx} (luma {luma}, "
              f"{100.0 * luma / ya.size:.4f}%, max {int(np.abs(ya - yb).max())})")
    print("all identical" if bad == 0 else f"{bad} case(s) differ")


if __name__ == "__main__":
    main()
