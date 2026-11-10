# DeSpot for AviSynth+ and VapourSynth

DeSpot removes temporal noise in the form of spots, dots and short streaks (dust, dirt) from video,
mainly from film transfers. A pixel is a spot candidate if it lies outside the value range of the
previous and the next frame; candidates are merged into spots, checked for size and motion, and
replaced from the neighbour frames.

This is a port of DeSpot 3.6.3 (Copyright (C) 2003-2009 Alexander G. Balakhnin aka Fizick, outfile and
mc additions by Firesledge, based on the Conditional Temporal Median Filter by Kevin Atkinson) to:

- AviSynth+: function `DeSpot`
- VapourSynth (API 4): function `despot.DeSpot`

One binary contains both interfaces. Windows x64, Linux x86_64 and macOS arm64 builds are made by
GitHub Actions.

The original documentation is in [doc/despot.html](doc/despot.html). Its algorithm description still
applies; this file lists the parameters with their real defaults and what changed in the port.

## Formats

8 bit only:

- YUV with any chroma subsampling up to 4 in each direction: 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0, 4:1:0
  (AviSynth+: YV12, YV16, YV24, YV411, and the planar YUV formats of AviSynth+)
- Greyscale (Y8 / GRAY8)
- AviSynth+ YUVA: the alpha plane is passed through

Spot detection works on luma; chroma is only touched by `color=true` (cleaning) and `show=1` (marking).
High bit depth and RGB are not supported (YUY2: use `ConvertToYV16()`).

## Usage

```
# AviSynth+
DeSpot(p1=30, p2=12, mthres=20, dilate=2, fitluma=true, blur=2, seg=2)
```

```python
# VapourSynth (booleans are 0/1)
clip = core.despot.DeSpot(clip, p1=30, p2=12, mthres=20, dilate=2, fitluma=1, blur=2, seg=2)
```

The first and the last frame are returned unchanged (no temporal neighbours).

## Parameters

| Name | Default | Range | Meaning |
|---|---|---|---|
| `p1` | 24 | 0..255 | a pixel must differ from its temporal neighbours by more than p1 to be a spot |
| `p2` | 12 | 0..255 | pixels differing by more than p2 belong to the same spot |
| `pwidth` | 16 | 0..32000 | maximum spot width |
| `pheight` | 5 | 0..32000 | maximum spot height (in field lines if `interlaced`) |
| `ranked` | true | | ranked difference detector over 3 neighbours in the previous and next frame (false: 1 pixel each) |
| `sign` | 0 | -2..2 | 0 any spots, 1 only dark, -1 only light, 2 dark spots and any outliers, -2 light spots and any outliers |
| `maxpts` | 0 | 0..10000000 | maximum number of pixels per spot (0: no limit) |
| `minpts` | 0 | 0..10000000 | minimum number of pixels per spot (0: no limit) |
| `p1percent` | 10 | 0..100 | minimum percentage of pixels above p1 per spot |
| `dilate` | 1 | 0..1000 | dilation of removed spots, in pixels |
| `fitluma` | false | | luma correction at removed spots |
| `blur` | 1 | 0..4 | blur length at the borders of removed spots |
| `tsmooth` | 0 | 0..255 | temporal smoothing of static areas (0: off, try 4..8) |
| `motpn` | true | | motion from previous to next frame (false: previous to current and current to next) |
| `seg` | 2 | 0..2 | 0 remove spot pixels without motion, 1 remove line segments without motion, 2 remove whole spots without motion (safest) |
| `color` | false | | clean chroma at removed spots (mean of previous, current and next frame) |
| `mscene` | 40 | 0..100 | percentage of moving pixels treated as a scene change (no spot removal) |
| `extmask` | none | clip | external mask, luma > 127 is protected like motion; same size, 8 bit |
| `mthres` | 16 | 0..255 | a pixel must change by more than mthres to be moving |
| `mwidth` | 7 | 0..32000 | width of the motion map denoise block |
| `mheight` | 5 | 0..32000 | height of the motion map denoise block |
| `merode` | 33 | 0..100 | percentage of moving pixels in the block for the erode stage |
| `interlaced` | false | | process fields separately |
| `median` | false | | simple temporal median of static areas instead of spot removal |
| `show` | 0 | 0..2 | 0 filter, 1 mark spots with `mark_v` (and colour), 2 map: 255 removed spot, 159 spot kept because of motion, 95 motion |
| `mark_v` | 255 | 0..255 | luma of the marks for `show=1`; its value modulo 3 selects pink, green or grey |
| `show_chroma` | false | | `show=2`: keep the chroma of the clip instead of grey |
| `outfile` | none | string | write the detected spots as an ASS subtitle file (only `motpn=true` with `seg>0`) |
| `mc` | false | | input is `Interleave(forward, original, backward)`; only every middle frame is reported in `outfile` |
| `spotmax1` | 12 | 1..10000000 | `outfile`: above this many spots, a frame with tightly packed spots is skipped |
| `spotmax2` | 20 | 1..10000000 | `outfile`: above this many spots, the frame is skipped |
| `planar` | false | | AviSynth+ only, for script compatibility; true (planar YUY2 hack) is an error |
| `asm_opt` | 0 | 0..3 | 0 auto, 1 C, 2 SSE2, 3 AVX2 (x86); 0 NEON, 1 C (arm64) |

The defaults of `pwidth` (16) and `fitluma` (false) are those of the 3.6.3 binary; its documentation
listed 6 and true.

## Differences from DeSpot 3.6.3

The output is bit-identical to the original 3.6.3 binary (checked on test clips over 50 parameter
combinations, in all instruction sets), except where these defects were fixed:

- `blur=4` used the pixel at x-11 instead of x-1 in one tap of its 9-tap filter (and read outside the
  frame near the left border).
- `interlaced=true`: the two field accumulators of the motion map denoise and of `dilate` overlapped,
  so the fields were mixed.
- The spot border blur read up to 3 pixels outside the row next to the left and right frame border;
  spots there are now clamped so that the blur stays inside.
- Spots reaching the right frame border kept a one pixel wide bounding box for the size check.
- The segment list sentinel was uninitialised, so spots touching the bottom rows could be linked with
  garbage (undefined behaviour, possible crash).
- `outfile`: the cluster test of `spotmax1` never worked (its standard deviation was always NaN).
- Large `mwidth`/`mheight`/`dilate` values overflowed 8 bit counters and wrote outside a fixed padding.
- Frames whose width is no multiple of 8 read pixels beyond the row end.

Also new: the port is multithreaded (AviSynth+ `MT_NICE_FILTER`, VapourSynth `fmParallel`; with
`outfile` the filter runs serialized). The original cached maps of neighbour frames and was only
correct for linear access; the port computes everything a frame needs from frames n-2..n+2, so random
access and any number of threads give the same output as linear access. Frame properties are kept.
With SSE2/AVX2 the port is 2 to 2.8 times faster than the original on one thread (1080p, depending on the mode).

## Building

```
meson setup build && ninja -C build              # Linux, macOS
meson setup build --vsenv -Db_vscrt=mt && ninja -C build   # Windows, MSVC
pip install .                                    # wheel into site-packages/vapoursynth/plugins
```

`build.bat` builds `build\DeSpot.dll` and `build\kernel_test.exe` with MSVC directly; there is also a
Visual Studio project in `msvc/`.

## Tests

- `meson test -C build` (or `build\kernel_test.exe`): every SIMD kernel against the C kernels on random data.
- `python test/test_despot.py [plugin]`: VapourSynth tests: every parameter case in every instruction
  set against C, all formats, odd frame sizes, random frame order, parallel rendering.
- `python test/avs_compare.py ref.dll new.dll clip.y4m`: compares two AviSynth+ builds over the
  parameter cases (`test/gen_clip.py` writes a synthetic clip with spots). `build.bat compat` builds the
  port with the 3.6.3 defects restored (`DESPOT_COMPAT_363`), which must be identical to the original.

## License

GNU GPL version 2, see [LICENSE](LICENSE). The vendored AviSynth+ and VapourSynth headers in `include/`
have their own licenses.
