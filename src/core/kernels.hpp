// SPDX-License-Identifier: GPL-2.0-only
// Row kernels of DeSpot. One table per instruction set; every table produces bit-identical
// results to the C table (kernels_c.cpp).
#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DESPOT_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DESPOT_NEON 1
#endif

namespace despot {

struct Kernels {
    // Spot candidates of one row: o[x] = difference of c[x] to the range of p/n (0 if inside).
    // ranked: range over x-1..x+1 of both p and n, o[0] = o[w-1] = 0. sign: see DeSpot "sign".
    void (*outliers)(const uint8_t *p, const uint8_t *n, const uint8_t *c, uint8_t *o, int w,
                     int sign, bool ranked, uint8_t p1);
    // m[x] = |p[x] - c[x]| > thr ? 3 : 0
    void (*motion)(const uint8_t *p, const uint8_t *c, uint8_t *m, int w, uint8_t thr);
    // r[x] += add[x] - sub[x]; either row may be null.
    void (*colsum)(int32_t *r, const uint8_t *sub, const uint8_t *add, int w);
    // S[x] = r[x-rad] + ... + r[x+rad] (r readable from -rad-1 to w+rad);
    // out[x] = ((!gate || gate[x]) && S[x] > thr) ? val : 0. tmp holds w + 2 * rad + 2 values.
    void (*window)(const int32_t *r, int w, int rad, const uint8_t *gate, int32_t thr, uint8_t val,
                   uint8_t *out, int32_t *tmp);
    // d[x] = s[x] ? val : 0
    void (*binarize)(const uint8_t *s, uint8_t *d, int w, uint8_t val);
    // Sum of c[0..w) and of p[0..w) + n[0..w).
    void (*luma_sums)(const uint8_t *c, const uint8_t *p, const uint8_t *n, int w, int64_t *sc, int64_t *spn);
    // o[x] = (noise && !motion) ? clamp(((p + n) >> 1) + lumad) : c
    void (*remove)(const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *noise,
                   const uint8_t *motion, uint8_t *o, int w, int lumad);
    // show=1 marking
    void (*mark)(const uint8_t *c, const uint8_t *noise, const uint8_t *motion, uint8_t *o, int w, uint8_t mark_v);
    // show=2 map; noise may be null
    void (*map)(const uint8_t *noise, const uint8_t *motion, uint8_t *o, int w);
    // o[x] = (a[x] || b[x]) ? 255 : 0
    void (*merge)(const uint8_t *a, const uint8_t *b, uint8_t *o, int w);
    // m[x] = (m[x] && (pn[x] || cn[x])) ? 1 : m[x]
    void (*noise_to_one)(const uint8_t *pn, const uint8_t *cn, uint8_t *m, int w);
    // temporal smoothing of pixels without noise and motion
    void (*tsmooth)(const uint8_t *p, const uint8_t *n, uint8_t *o, const uint8_t *noise,
                    const uint8_t *motion, int w, int limit);
    // o[x] = motion ? c : median(p, c, n)
    void (*median)(const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *motion, uint8_t *o, int w);
    // number of non-zero bytes
    int64_t (*count_nonzero)(const uint8_t *s, int w);
    // m[x] = ext[x] > 127 ? 255 : m[x]
    void (*ext_mask)(const uint8_t *ext, uint8_t *m, int w);
    // first index >= x with l[i] > thr, or w
    int (*next_above)(const uint8_t *l, int x, int w, uint8_t thr);
    // out[x] = rows[0][x] | rows[1][x] | ... (n rows)
    void (*or_rows)(const uint8_t *const *rows, int n, uint8_t *out, int w);
    // out[x] = (s[x-rad] | ... | s[x+rad]) ? val : 0; s readable from -rad to w+rad-1
    void (*or_window)(const uint8_t *s, uint8_t *out, int w, int rad, uint8_t val);

    // 9..16 bit samples (shift = bits - 8). The noise and motion maps stay 8 bit.
    // outliers: o[x] = min(difference >> shift, 255), then as the 8 bit kernel (p1 in 8 bit units)
    void (*outliers16)(const uint16_t *p, const uint16_t *n, const uint16_t *c, uint8_t *o, int w, int sign,
                       bool ranked, uint8_t p1, int shift);
    // m[x] = |p[x] - c[x]| > thr ? 3 : 0 (thr in sample units)
    void (*motion16)(const uint16_t *p, const uint16_t *c, uint8_t *m, int w, uint16_t thr);
    void (*luma_sums16)(const uint16_t *c, const uint16_t *p, const uint16_t *n, int w, int64_t *sc, int64_t *spn);
    // o[x] = (noise && !motion) ? clamp(((p + n) >> 1) + lumad, 0, maxval) : c
    void (*remove16)(const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *noise,
                     const uint8_t *motion, uint16_t *o, int w, int lumad, uint16_t maxval);
    // show=1 marking: mark_v and 126 are scaled by << shift
    void (*mark16)(const uint16_t *c, const uint8_t *noise, const uint8_t *motion, uint16_t *o, int w,
                   uint16_t mark_v, int shift);
    // show=2 map: 255, 159, 95 scaled by << shift; noise may be null
    void (*map16)(const uint8_t *noise, const uint8_t *motion, uint16_t *o, int w, int shift);
    void (*tsmooth16)(const uint16_t *p, const uint16_t *n, uint16_t *o, const uint8_t *noise,
                      const uint8_t *motion, int w, int limit);
    void (*median16)(const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *motion, uint16_t *o,
                     int w);
    // m[x] = (ext[x] >> shift) > 127 ? 255 : m[x]
    void (*ext_mask16)(const uint16_t *ext, uint8_t *m, int w, int shift);
};

const Kernels &kernelsC();
#if defined(DESPOT_X86)
const Kernels &kernelsSSE2();
const Kernels &kernelsAVX2();
#elif defined(DESPOT_NEON)
const Kernels &kernelsNEON();
#endif

// opt: 0 auto, 1 C, 2 SSE2, 3 AVX2 (x86); 0 auto (NEON), 1 C (arm64).
bool simdAvailable(int opt);
const Kernels &kernelsFor(int opt);

} // namespace despot
