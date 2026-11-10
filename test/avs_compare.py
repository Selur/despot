"""Compare two DeSpot AviSynth plugin DLLs frame by frame over a parameter matrix.

Usage: python avs_compare.py ref.dll new.dll clip.y4m [workdir]
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


def run(dll, clip, args, out, extra=""):
    script = out + ".avs"
    with open(script, "w") as fh:
        fh.write(f'LoadPlugin("{FFMS}")\nLoadPlugin("{dll}")\n')
        fh.write(f'src = FFVideoSource("{clip}", cache=false)\n')
        fh.write('mask = src.BlankClip(color_yuv=$000000).Letterbox(100, 100, 100, 100, color_yuv=$FF8080)\n')
        sep = ", " if args else ""
        fh.write(f"DeSpot(src{sep}{args})\n")
    r = subprocess.run([AVS2YUV, "-nstdr", "-raw", script, "-o", out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None, (r.stderr or r.stdout).strip()
    return np.fromfile(out, np.uint8), None


def main():
    ref_dll, new_dll, clip = (os.path.abspath(a) for a in sys.argv[1:4])
    work = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(clip), "avs_work")
    os.makedirs(work, exist_ok=True)
    with open(clip, "rb") as fh:
        hdr = fh.readline().decode().split()
    w = int([t for t in hdr if t.startswith("W")][0][1:])
    h = int([t for t in hdr if t.startswith("H")][0][1:])
    cs = [t for t in hdr if t.startswith("C")]
    cs = cs[0] if cs else "C420"
    fsize = {"C420jpeg": w * h * 3 // 2, "C420": w * h * 3 // 2, "C422": w * h * 2, "C444": w * h * 3, "Cmono": w * h}[cs]
    bad = 0
    for i, args in enumerate(CASES):
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
        luma = int(np.count_nonzero(fa[:, : w * h] != fb[:, : w * h]))
        print(f"[{i:2}] {args!r}: DIFF frames={diff_frames[:10]}{'...' if len(diff_frames) > 10 else ''} "
              f"pixels={npx} (luma {luma})")
    print("all identical" if bad == 0 else f"{bad} case(s) differ")


if __name__ == "__main__":
    main()
