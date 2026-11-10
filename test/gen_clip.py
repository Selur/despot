"""Synthetic test clip for DeSpot: moving shapes, film grain and random dark/bright spots.

Usage: python gen_clip.py out.y4m [width height frames subsampling seed]
subsampling: 420, 422, 444, 411, 440, 410 or gray (y4m: not 440/410).  Deterministic for a given seed.
"""
import sys
import numpy as np


def make_frames(width=320, height=240, frames=30, subsampling="420", seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 60 + 100 * (xx / width) + 40 * np.sin(yy / 17.0)
    sx, sy = {"420": (2, 2), "422": (2, 1), "444": (1, 1), "411": (4, 1), "440": (1, 2), "410": (4, 4),
              "gray": (1, 1)}[subsampling]
    cw, ch = width // sx, height // sy
    cyy, cxx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    out = []
    for f in range(frames):
        y = base.copy()
        # moving bright box and dark disc (real motion)
        bx = int(20 + 5 * f) % (width - 40)
        y[40:80, bx:bx + 30] = 200
        cx, cy = width * 0.6 + 3 * f, height * 0.6
        y[(xx - cx) ** 2 + (yy - cy) ** 2 < 15 ** 2] = 30
        # a static textured area
        y[150:200, 20:80] += 20 * ((xx[150:200, 20:80] // 4 + yy[150:200, 20:80] // 4) % 2)
        # temporal grain
        y += rng.normal(0, 2.0, y.shape)
        # spots: present in this frame only
        for _ in range(rng.integers(5, 25)):
            w = int(rng.integers(1, 8))
            h = int(rng.integers(1, 6))
            px = int(rng.integers(0, width - w))
            py = int(rng.integers(0, height - h))
            v = 10 if rng.random() < 0.5 else 245
            y[py:py + h, px:px + w] = v
        # spots touching borders (top/bottom rows, left/right columns)
        if f % 3 == 0:
            y[height - 2:height, int(rng.integers(0, width - 4)):][:, :3] = 250
            y[0:2, int(rng.integers(0, width - 4)):][:, :3] = 5
            y[int(rng.integers(0, height - 4)):][:3, width - 2:] = 250
            y[int(rng.integers(0, height - 4)):][:3, 0:2] = 5
        y8 = np.clip(np.rint(y), 0, 255).astype(np.uint8)
        planes = [y8]
        if subsampling != "gray":
            u = np.clip(128 + 30 * np.sin(cxx / 11.0 + f * 0.1) + rng.normal(0, 1, (ch, cw)), 0, 255)
            v = np.clip(128 + 30 * np.cos(cyy / 9.0) + rng.normal(0, 1, (ch, cw)), 0, 255)
            planes += [np.rint(u).astype(np.uint8), np.rint(v).astype(np.uint8)]
        out.append(planes)
    return out


def write_y4m(path, frames, width, height, subsampling):
    cs = {"420": "C420jpeg", "422": "C422", "444": "C444", "411": "C411", "gray": "Cmono"}[subsampling]
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 {cs}\n".encode())
        for planes in frames:
            fh.write(b"FRAME\n")
            for p in planes:
                fh.write(p.tobytes())


if __name__ == "__main__":
    path = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 320
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 240
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    ss = sys.argv[5] if len(sys.argv) > 5 else "420"
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    write_y4m(path, make_frames(w, h, n, ss, seed), w, h, ss)
