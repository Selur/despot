# DeSpot for AviSynth+ and VapourSynth

DeSpot removes temporal noise that shows up as spots, dots and short streaks: dust and dirt on film,
dropouts, small scratches. It is mainly used to restore telecined 8 mm and other film transfers.

A pixel is a spot candidate when it lies outside the range of the same area in the previous and the
next frame. Candidates are grown into spots, spots that are too large or touch moving areas are kept,
and the rest is replaced from the neighbour frames.

This is a port of DeSpot 3.6.3 to

- **AviSynth+**: function `DeSpot`, same name and parameters as before
- **VapourSynth** (API 4): function `despot.DeSpot`

One binary contains both interfaces. Builds for Windows x64, Linux x86_64 and macOS arm64 are made by
GitHub Actions. The output is bit-identical to DeSpot 3.6.3 except where defects were fixed (see
[Differences from DeSpot 3.6.3](#differences-from-despot-363)). The port is multithreaded and uses
SSE2/AVX2 (x86) or NEON (arm64).

## Contents

- [Formats](#formats)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Tuning](#tuning)
- [How it works](#how-it-works)
- [Examples](#examples)
- [Differences from DeSpot 3.6.3](#differences-from-despot-363)
- [History](#history)
- [Building](#building) and [Tests](#tests)
- [Credits and license](#credits-and-license)

## Formats

8 to 16 bit integer samples:

- YUV with any chroma subsampling up to 4 in each direction: 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0, 4:1:0
  (AviSynth+: YV12, YV16, YV24, YV411 and the other planar YUV formats, also with 10, 12, 14, 16 bit)
- greyscale (Y8 / GRAY8 up to Y16 / GRAY16)
- AviSynth+ YUVA: the alpha plane is passed through unchanged

Spots are detected and removed on luma. Chroma is only changed by `color=true` (cleaning) and
`show=1` (marking). RGB, 32 bit float and interleaved YUY2 are not supported; use for example
`ConvertToYV16()` for YUY2 or `ConvertBits(16)` for float sources.

### High bit depth

All thresholds and levels keep their 8 bit meaning, whatever the bit depth: `p1`, `p2`, `mthres`,
`tsmooth`, `mark_v` and the `show=2` levels are scaled by 2^(bits-8) internally (`mthres=16` is 64 for
10 bit). Spot and motion detection use the same 8 bit decisions: a clip converted from 8 bit by a shift
(`ConvertBits(16)` in AviSynth+) gives exactly the same spots, motion map and `outfile` as the 8 bit
clip. Replaced pixels are computed at full precision, so the output then differs from the 8 bit
result only by the rounding of 8 bit (on average half an 8 bit step at the replaced pixels).

`extmask` may have any bit depth from 8 to 16, independent of the clip; values above half of its range
(above 127 in 8 bit) are protected.

The first and the last frame of a clip are returned unchanged, because they lack a temporal neighbour.

## Installation

- **AviSynth+**: put `DeSpot.dll` (`libDeSpot.so` / `libDeSpot.dylib`) into the plugin directory or load it
  with `LoadPlugin()`. This replaces older `DeSpot.dll` versions; do not keep both.
- **VapourSynth**: `pip install` the wheel from the releases page (it installs into
  `site-packages/vapoursynth/plugins`), copy the library into a VapourSynth plugin directory, or use
  `core.std.LoadPlugin(path)`.

## Usage

```
# AviSynth+
LWLibavVideoSource("film.mkv")
DeSpot(p1=30, p2=12, mthres=20, dilate=2, fitluma=true, blur=2)
```

```python
# VapourSynth: booleans are passed as 0/1
import vapoursynth as vs
core = vs.core
clip = core.lsmas.LWLibavSource("film.mkv")
clip = core.despot.DeSpot(clip, p1=30, p2=12, mthres=20, dilate=2, fitluma=1, blur=2)
clip.set_output()
```

To see what the filter detects, use `show=1` (spots painted with `mark_v`, motion brightened) or
`show=2` (a map of spots and motion) while tuning.

## Parameters

All parameters are optional and named. The defaults are those of the DeSpot 3.6.3 binary.

### Spot detection

| Name | Default | Range | Meaning |
|---|---|---|---|
| `p1` | 24 | 0..255 | A pixel must differ from its temporal neighbours by more than `p1` to start a spot. |
| `p2` | 12 | 0..255 | Pixels next to a spot that differ by more than `p2` belong to the same spot. |
| `pwidth` | 16 | 0..32000 | Maximum width of a spot. Larger spots are kept (probably not dirt). |
| `pheight` | 5 | 0..32000 | Maximum height of a spot; with `interlaced=true` counted in field lines. |
| `ranked` | true | | Ranked detector: compare with the minimum/maximum of 3 pixels (x-1, x, x+1) in the previous and in the next frame. `false`: compare with one pixel in each frame only. The ranked detector is more stable on noisy video. |
| `sign` | 0 | -2..2 | `0` dark and light spots, `1` only dark spots and outliers, `-1` only light spots and outliers, `2` dark spots with outliers of either sign, `-2` light spots with outliers of either sign. |
| `maxpts` | 0 | 0..10000000 | Maximum number of pixels per spot (0: no limit). |
| `minpts` | 0 | 0..10000000 | Minimum number of pixels per spot (0: no limit). |
| `p1percent` | 10 | 0..100 | Minimum percentage of pixels above `p1` in a spot, so that weak spots made mostly of `p2` outliers are kept. |

### Motion detection

| Name | Default | Range | Meaning |
|---|---|---|---|
| `mthres` | 16 | 0..255 | A pixel that changes by more than `mthres` between the compared frames is moving. Should be larger than `p2`. |
| `mwidth` | 7 | 0..32000 | Width of the block for the motion map denoise (erode and dilate). |
| `mheight` | 5 | 0..32000 | Height of that block. |
| `merode` | 33 | 0..100 | Percentage of moving pixels a block needs in the erode stage; smaller motion areas are dropped. |
| `motpn` | true | | `true`: motion from the previous to the next frame. `false`: from previous to current and from current to next frame (the method before 3.0). |
| `mscene` | 40 | 0..100 | If more than this percentage of the frame moves, the frame is treated as a scene change and nothing is removed. |
| `extmask` | none | clip | External mask, values above 127 (8 bit scale) are protected like motion. Same frame size, 8..16 bit; only its first plane is used. |
| `interlaced` | false | | Process the two fields separately. |

### Spot removal

| Name | Default | Range | Meaning |
|---|---|---|---|
| `seg` | 2 | 0..2 | `0`: remove every spot pixel that does not overlap motion (strongest, the method before 3.0). `1`: remove every line segment of a spot that does not touch motion. `2`: remove only whole spots that do not touch motion (safest). |
| `dilate` | 1 | 0..1000 | Grow removed spots by this many pixels, to cover soft spot edges. |
| `fitluma` | false | | Correct the luma of the replaced pixels to their surroundings (per line segment with `seg`>0). Use it with suitably dilated spots. |
| `blur` | 1 | 0..4 | Length of the blur across the borders of removed spots. |
| `tsmooth` | 0 | 0..255 | Temporal smoothing of static areas (no spots, no motion): roughly the luma variance over 3 frames up to which pixels are smoothed. 0: off; try 4..8. |
| `color` | false | | Replace chroma at removed spots with the mean of previous, current and next frame. |
| `median` | false | | Simple median mode instead of spot removal: a temporal median of all static pixels. Only `mthres`, `mwidth`, `mheight`, `merode`, `interlaced`, `show` and `show_chroma` apply then. |

### Display and spot file

| Name | Default | Range | Meaning |
|---|---|---|---|
| `show` | 0 | 0..2 | `0`: filter. `1`: paint spots with `mark_v` instead of removing them, brighten moving areas, colour spots in chroma. `2`: map in luma: 255 spot that is removed, 159 spot kept because of motion, 95 motion, 0 rest. |
| `mark_v` | 255 | 0..255 | Luma of the marks for `show=1`; its value modulo 3 selects pink, green or grey chroma. |
| `show_chroma` | false | | `show=2`: keep the chroma of the clip instead of grey. |
| `outfile` | none | string | Write the detected spots as an ASS subtitle file: magenta rectangles 2 pixels larger than each spot. In an ASS editor (e.g. Aegisub) the `Outline` style can be switched to `Mask` to get filled white rectangles. Only with `motpn=true` and `seg`>0. Frames are written in frame order, whatever order and number of threads they are rendered with (frames after a gap of unrendered frames when the filter is closed). The file is overwritten every time the script is opened. |
| `mc` | false | | The input is `Interleave(forward, original, backward)` from motion compensation; only every middle frame is written to `outfile`, with the timing of the original clip. |
| `spotmax1` | 12 | 1..10000000 | `outfile`: with more spots than this, a frame whose spots are packed tightly is treated as a false detection and left out of the file. |
| `spotmax2` | 20 | 1..10000000 | `outfile`: with more spots than this, the frame is left out of the file. The spots are still removed in the video. |

### Other

| Name | Default | Range | Meaning |
|---|---|---|---|
| `asm_opt` | 0 | 0..3 | Instruction set: `0` best available, `1` C, `2` SSE2, `3` AVX2 (x86); `0` NEON, `1` C (arm64). All give identical output. |
| `planar` | false | | AviSynth+ only, kept for old scripts: `true` (the planar YUY2 hack of AviSynth 2.5) is an error. |

The documentation of 3.6.3 and the AviSynth wiki list `pwidth=6` and `fitluma=true` as defaults, but the
binaries have always used 16 and false. This port keeps the real values, so existing scripts give the
same result.

## Tuning

There are no good defaults; the parameters must fit the material. `show=1` and `show=2` help a lot.

1. Set `interlaced=true` for interlaced video.
2. Set `pwidth` and `pheight` a little larger than the specks you want to remove. With `interlaced`,
   `pheight` is per field, so it effectively doubles.
3. Set `p1`, `p2` and `mthres`, usually `p1 > mthres > p2`. Values that are too low turn small
   details into spot candidates; real specks then merge with them and grow beyond the size limit.
4. `mwidth` and `mheight` set how far motion protects its surroundings. Increase them against false
   detections in fast motion, then increase `merode` to ignore tiny motion areas again.
5. If almost all spots are dark (or light), set `sign` accordingly.
6. Use `maxpts` to protect larger objects in addition to `pwidth`/`pheight`, and `p1percent` to keep
   weak, low contrast spots.
7. Increase `dilate` to also remove the soft edges of spots; `fitluma` adapts the brightness of the
   replaced area and `blur` hides the borders.
8. For noisy static areas try `tsmooth` 4..8. Enable `color` for coloured spots.
9. Lower `mscene` if scene changes produce artefacts.
10. `extmask` adds your own protection mask, for example an inverted SAD mask from MVTools (see
    [Examples](#examples)).

The best results come from motion compensated neighbour frames (global motion with DePan, local
motion with MVTools): moving objects then line up with the current frame, there are fewer false
detections, and more spots can be removed.

## How it works

Spot removal (`median=false`):

1. **Candidates.** For every pixel the minimum and maximum of the previous and next frame are taken
   (`ranked=true`: over x-1..x+1 of both frames). If the pixel lies below the minimum or above the
   maximum (only one side with `sign`), the distance is its outlier value.
2. **Spots.** Neighbouring pixels with outlier values above `p2` form horizontal line segments;
   segments that touch on adjacent lines form spots. A spot needs at least one pixel above `p1`.
3. **Size check.** Spots larger than `pwidth` x `pheight`, outside `minpts`..`maxpts`, or with fewer
   than `p1percent` percent pixels above `p1` are kept. With `seg=0` the remaining spot pixels are
   dilated by `dilate`.
4. **Motion.** Pixels that change by more than `mthres` (previous to next frame, or with `motpn=false`
   previous to current and current to next frame) are moving. With `motpn=false`, moving pixels that
   are also spot candidates count less.
5. **Motion map denoise.** An erosion removes small motion areas (fewer than `merode` percent moving
   pixels in an `mwidth` x `mheight` block), a dilation with the same block grows the rest.
6. **Scene change, external mask.** Above `mscene` percent motion the whole frame counts as moving.
   `extmask` is added to the motion map.
7. **Removal.** Depending on `seg`, spot pixels, segments or whole spots that touch motion are kept; the
   rest is replaced by the average of the previous and next frame, optionally luma corrected
   (`fitluma`), dilated (`dilate`) and blurred at the borders (`blur`).
8. **Optional:** temporal smoothing of static areas (`tsmooth`) and chroma cleaning (`color`).

Simple median mode (`median=true`): motion is detected as above (previous to current and current to
next frame) and denoised; every static pixel is replaced by the median of previous, current and next
frame.

## Examples

### Global motion compensation with DePan (AviSynth+)

```
LWLibavVideoSource("film.mkv")
src = last
data = DePanEstimate(src, trust=3)
DePanInterleave(src, data=data)          # compensated previous, source, compensated next
DeSpot(p1=30, p2=15, pwidth=800, pheight=600, mthres=20, motpn=true, dilate=1, seg=1)
SelectEvery(3, 1)
```

With compensated neighbours very large `pwidth`/`pheight` values become usable, because real objects
no longer differ from their neighbours.

### Local motion compensation and SAD mask with MVTools (AviSynth+)

```
LWLibavVideoSource("film.mkv")
src = last
pre = src.RemoveGrain(4)                # median prefiltered copy for the motion search only (RgTools)
sup = src.MSuper()
psup = pre.MSuper()

vf = psup.MAnalyse(isb=false)
vb = psup.MAnalyse(isb=true)
fwd = src.MFlow(sup, vf, thscd1=400)    # previous frame moved onto the current one
bwd = src.MFlow(sup, vb, thscd1=400)    # next frame moved onto the current one

# 255 where at least one motion estimate is good: these areas are protected
okf = src.MMask(vf, ml=100, kind=1, gamma=1, thscd1=400).mt_binarize(20, upper=true)
okb = src.MMask(vb, ml=100, kind=1, gamma=1, thscd1=400).mt_binarize(20, upper=true)
ok = mt_logic(okf, okb, "or").mt_expand()

Interleave(fwd, src, bwd)
DeSpot(p1=30, p2=12, pwidth=800, pheight=600, mthres=20, merode=33, seg=0, color=true,
\      motpn=true, extmask=Interleave(ok, ok, ok))
SelectEvery(3, 1)
```

### The same with VapourSynth

```python
import vapoursynth as vs
core = vs.core

src = core.lsmas.LWLibavSource("film.mkv")
sup = core.mv.Super(src)
psup = core.mv.Super(core.std.Median(src))   # prefiltered copy for the motion search only

vf = core.mv.Analyse(psup, isb=False)
vb = core.mv.Analyse(psup, isb=True)
fwd = core.mv.Flow(src, sup, vf, thscd1=400)
bwd = core.mv.Flow(src, sup, vb, thscd1=400)

def good(vectors):  # 255 where the motion estimate is good
    sad = core.mv.Mask(src, vectors, ml=100, kind=1, gamma=1, thscd1=400)
    return core.std.Expr(core.std.ShufflePlanes(sad, 0, vs.GRAY), "x 20 > 0 255 ?")

ok = core.std.Maximum(core.std.Expr([good(vf), good(vb)], "x y max"))

clip = core.std.Interleave([fwd, src, bwd])
clip = core.despot.DeSpot(clip, p1=30, p2=12, pwidth=800, pheight=600, mthres=20, merode=33, seg=0,
                          color=1, motpn=1, extmask=core.std.Interleave([ok, ok, ok]))
clip = core.std.SelectEvery(clip, 3, 1)
clip.set_output()
```

Use `motpn=true` with interleaved input: then frame 3k+1 is only compared with its compensated
neighbours 3k and 3k+2. Add `mc=true` when writing an `outfile`, so that only the real frames are
listed, with correct times.

### Spot file for manual editing

```
DeSpot(p1=30, p2=12, outfile="spots.ass")
```

Open the video and `spots.ass` in Aegisub, delete false detections or add rectangles, switch the style
to `Mask`, and use the rendered subtitles (for example with `assrender`) as a mask for further
processing. Copy the file before editing: it is overwritten when the script is loaded again.

## Differences from DeSpot 3.6.3

The output is bit-identical to the DeSpot 3.6.3 binary (checked on test clips over 50 parameter
combinations, in all instruction sets, including `outfile` and `mc`), except where these defects were
fixed:

- `blur=4` used the pixel at x-11 instead of x-1 in one tap of its 9-tap filter (and read outside the
  frame near the left border).
- `interlaced=true`: the two field accumulators of the motion map denoise and of `dilate` overlapped,
  so the fields were mixed.
- The spot border blur read up to 3 pixels outside the row next to the left and right frame border;
  spots there are now clamped so that the blur stays inside.
- Spots reaching the right frame border kept a one pixel wide bounding box for the size check.
- The end marker of the spot list was uninitialised, so spots touching the bottom rows could be linked
  with garbage (undefined behaviour, possible crash).
- `outfile`: the cluster test of `spotmax1` never worked (its standard deviation was always NaN).
- Large `mwidth`/`mheight`/`dilate` values overflowed 8 bit counters and wrote outside a fixed padding.
- Frames whose width is no multiple of 8 read pixels beyond the row end.

New in the port:

- VapourSynth interface, Linux and macOS builds.
- Multithreading (AviSynth+ `MT_NICE_FILTER`, VapourSynth `fmParallel`, also with `outfile`). 3.6.3 cached maps of neighbour frames and was only correct for linear access. The port
  computes everything a frame needs from frames n-2..n+2, so random access, seeking and any number of
  threads give the same output as linear access. Maps that neighbouring output frames share (`motpn=false`
  and median mode) are computed once and kept in a thread-safe cache.
- SSE2/AVX2/NEON code (`asm_opt`): with AVX2 2.8 to 4.4 times faster than 3.6.3 on one thread (1080p,
  including decoding, depending on the mode), plus the gain from multithreading.
- Formats other than YV12 and 9 to 16 bit samples (see [Formats](#formats)); frame properties are kept.

## History

| Version | Date | Changes |
|---|---|---|
| 3.7.0 | unreleased | This port: AviSynth+ and VapourSynth, multithreading, SIMD, more formats, 9..16 bit, bug fixes. |
| 3.6.3.1 | 2020-08-07 | Groucho2004: fixed a crash with AviSynth+ 3.6.1. |
| 3.6.3.0 | 2020-06-03 | Groucho2004: AviSynth 2.6 API, x64 build. |
| 3.6.3 | 2013-10-08 | cretindesalpes (Firesledge): `mc`, `spotmax1`, `spotmax2`. |
| 3.6.2 | 2011-08-19 | cretindesalpes (Firesledge): `outfile`. |
| 3.6.1 | 2010-06-11 | Last release by Fizick; earlier changes are listed in [doc/despot.html](doc/despot.html). |

[doc/despot.html](doc/despot.html) is the original documentation of 3.6.3. Links: the
[AviSynth wiki page](http://avisynth.nl/index.php/DeSpot) and the DeSpot thread in the Doom9 forum.

## Building

```
meson setup build && ninja -C build                        # Linux, macOS
meson setup build --vsenv -Db_vscrt=mt && ninja -C build   # Windows, MSVC
pip install .                                              # wheel for site-packages/vapoursynth/plugins
```

`build.bat` builds `build\DeSpot.dll`, `build\kernel_test.exe` and `build\bench.exe` (speed of the engine
and its stages on synthetic 1080p frames: `bench.exe [asm_opt]`) with MSVC directly; there is also a
Visual Studio project in `msvc/`. The AviSynth+ and VapourSynth headers are included in `include/`.

## Tests

- `meson test -C build` (or `build\kernel_test.exe`): every SIMD kernel against the C kernels on random data.
- `python test/test_despot.py [plugin]`: VapourSynth tests: every parameter case in every instruction
  set against C, all formats and bit depths, odd frame sizes, random frame order, parallel rendering, and
  that 8 bit content in 16 bit gives the same spots as in 8 bit. Without a path
  the installed plugin is tested.
- `python test/avs_compare.py ref.dll new.dll clip [workdir]`: compares two AviSynth+ builds over the
  parameter cases, on a `.y4m` file or any video file ffms2 opens (`DESPOT_PRE="Trim(1000, -200)"` selects a
  section, `DESPOT_CASES` a subset of the cases); `test/gen_clip.py` writes a synthetic clip with spots. `build.bat compat` builds the
  port with the 3.6.3 defects restored (`DESPOT_COMPAT_363`); that build must be identical to 3.6.3.

## Credits and license

- Kevin Atkinson: Conditional Temporal Median Filter 0.93 (2003), the base of DeSpot
- Alexander G. Balakhnin (Fizick): DeSpot 1.0 to 3.6.1 (2003-2010)
- cretindesalpes (Firesledge): `outfile`, `mc`, `spotmax1`, `spotmax2` (3.6.2, 3.6.3)
- Groucho2004: AviSynth 2.6 API and x64 builds (3.6.3.0, 3.6.3.1)

GNU General Public License version 2, see [LICENSE](LICENSE). The vendored AviSynth+ and VapourSynth
headers in `include/` have their own licenses.
