// SPDX-License-Identifier: GPL-2.0-only
/*
    DeSpot - Conditional Temporal Despotting Filter: filter functions

    Copyright (C)2003-2008 Alexander G. Balakhnin aka Fizick.
    Based on Conditional Temporal Median Filter 0.93 (C) 2003 Kevin Atkinson.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 2 as
    published by the Free Software Foundation.

    DESPOT_COMPAT_363 (testing only) restores the defects of DeSpot 3.6.3 that this port fixes,
    so the output can be compared bit for bit with the original binary:
    - 9-tap blur (blur=4) used pixel x-11 instead of x-1 (out-of-bounds read)
    - the two field accumulators of the interlaced motion/noise denoise overlapped
    - segments that reach the right frame border kept a one pixel wide bounding box
    - the spot blur read up to 3 pixels outside the row near the left/right border
    - the spot cluster test of outfile divided neither sum by the count (always NaN)
*/
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "despot.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4244) // blur results are byte averages
#endif

#if DESPOT_COMPAT_363
#define M11 11
#else
#define M11 1
#endif

namespace despot {

// Two sentinels behind the last segment; their ly never matches a real line.
void finish_segments(Segments &segs) {
    segs.end[0].init(SHRT_MAX, SHRT_MIN / 2);
    segs.end[0].what = B_NOTHING;
    segs.end[1].init(SHRT_MAX - 1, SHRT_MIN / 2);
    segs.end[1].what = B_NOTHING;
}

namespace {

void combine_xs(const uint8_t *noise, Segments &segs, const Parms &p) {
    const uint8_t *l = noise;
    const uint8_t p1 = p.p1, p2 = p.p2;
    const int width = p.width;
    for (int y = 0; y < p.height; ++y) {
        int x = 0;
        for (;;) {
            x = p.k->next_above(l, x, width, p2);
            if (x >= width) break;
            Segment *c = segs.end++; // new left edge found
            c->init(static_cast<short>(y), static_cast<short>(x));
            for (; x < width && l[x] > p2; ++x) {
                c->nump2 += 1;
                c->sumnoise += l[x];
                if (l[x] > p1) {
                    c->p1yes = true;
                    c->nump1 += 1;
                }
            }
            c->lx2 = static_cast<short>(x - 1);
#if DESPOT_COMPAT_363
            if (x < width) c->data.b.x2 = static_cast<short>(x - 1);
#else
            c->data.b.x2 = static_cast<short>(x - 1);
#endif
        }
        l += p.pitch;
    }
    finish_segments(segs);
}

void combine_ys(Segments &segs, const Parms &p) {
    if (segs.data == segs.end) return;
    Segment *c = segs.data; // first segment
    Segment *n = c + 1;
    while (n->ly == (n - 1)->ly) ++n; // search next segment with ly not equal to ly of c
    Segment *n0;
    bool c_has_next, n_has_next;

    while (n < segs.end) {
        n0 = n;
        if (n->ly < c->ly + p.y_next) { // search next segment with ly > ly of c
            ++n;
            while (n->ly == (n - 1)->ly) ++n;
        }

        if (c->ly + p.y_next == n->ly) { // we found next segment in neighbor line
            do {
                if ((c->lx1 <= n->lx1 && c->lx2 >= n->lx1 - 1) ||
                    (c->lx1 > n->lx1 && n->lx2 >= c->lx1 - 1)) { // overlap or neighbor
                    Segment *a = c;
                    while (a->what & B_LINK) a = a->data.nowis;
                    Segment *b = n;
                    while (b->what & B_LINK) b = b->data.nowis;
                    if (a > b) std::swap(a, b); // this is important
                    a->p1yes = a->p1yes || b->p1yes;
                    a->data.b.x1 = std::min(a->data.b.x1, b->data.b.x1);
                    a->data.b.x2 = std::max(a->data.b.x2, b->data.b.x2);
                    a->data.b.y2 = n->ly;

                    if (a != b) {
                        a->nump1 += b->nump1;
                        a->nump2 += b->nump2;
                        a->sumnoise += b->sumnoise;
                        a->what = a->what | B_MASTER;
                        b->what = b->what | B_LINK;
                        b->data.nowis = a; // "b" now is included in "a" as LINK
                    }
                }

                c_has_next = c->ly == (c + 1)->ly;
                n_has_next = n->ly == (n + 1)->ly;
                if (c->lx2 <= n->lx2 && c_has_next) ++c;
                else if (n_has_next) ++n;
                else if (c_has_next) ++c;
                else break;
            } while (true);

            ++c;
            ++n;
        } else {
            if (c->ly < n0->ly && n0->ly < n->ly) {
                c = n0;
            } else {
                c = n;
                ++n;
                while (n->ly == (n - 1)->ly) ++n;
            }
        }
    }
}

void to_size(Segments &segs, const Parms &p) {
    const int PH = p.pheight * p.y_next;
    for (Segment *cur = segs.data; cur != segs.end; ++cur) {
        if (!(cur->what & B_LINK)) { // super-master or single segment
            Segment::Data &b = cur->data;
            const int w = b.b.x2 - b.b.x1 + 1;
            const int h = b.b.y2 - b.b.y1 + 1;
            b.s.is_noise = cur->p1yes && w <= p.pwidth && h <= PH;
            if (p.maxpts > 0 && b.s.is_noise) b.s.is_noise = cur->nump2 <= p.maxpts;
            if (p.minpts > 0 && b.s.is_noise) b.s.is_noise = cur->nump2 >= p.minpts;
            if (p.p1percent > 0 && b.s.is_noise) b.s.is_noise = (100 * cur->nump1) / cur->nump2 >= p.p1percent;
            b.s.width = static_cast<short>(w);
            b.s.height = static_cast<short>(h);
        } else { // B_LINK: same decision as the master (processed before, it has a lower address)
            cur->data.s.is_noise = cur->data.nowis->data.s.is_noise;
        }
    }
}

void reject_chain(Segment *c, const Parms &p) {
    c->data.s.is_noise = false;
    if (p.seg == 2) // 2D spot mode - reject all masters
        for (Segment *m = c->data.nowis; m; m = m->data.nowis) m->data.s.is_noise = false;
}

// Sample kernels for 8 bit (original kernels) and 9..16 bit samples.
inline void k_outliers(const Parms &P, const uint8_t *p, const uint8_t *n, const uint8_t *c, uint8_t *o) {
    P.k->outliers(p, n, c, o, P.width, P.sign, P.ranked, P.p1);
}
inline void k_outliers(const Parms &P, const uint16_t *p, const uint16_t *n, const uint16_t *c, uint8_t *o) {
    P.k->outliers16(p, n, c, o, P.width, P.sign, P.ranked, P.p1, P.shift);
}
inline void k_motion(const Parms &P, const uint8_t *p, const uint8_t *c, uint8_t *m) {
    P.k->motion(p, c, m, P.width, P.mthres);
}
inline void k_motion(const Parms &P, const uint16_t *p, const uint16_t *c, uint8_t *m) {
    P.k->motion16(p, c, m, P.width, static_cast<uint16_t>(P.mthres << P.shift));
}
inline void k_ext_mask(const Parms &P, const uint8_t *e, uint8_t *m, int) { P.k->ext_mask(e, m, P.width); }
inline void k_ext_mask(const Parms &P, const uint16_t *e, uint8_t *m, int shift) {
    P.k->ext_mask16(e, m, P.width, shift);
}
inline void k_median(const Parms &P, const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *m,
                     uint8_t *o) {
    P.k->median(p, n, c, m, o, P.width);
}
inline void k_median(const Parms &P, const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *m,
                     uint16_t *o) {
    P.k->median16(p, n, c, m, o, P.width);
}
inline void k_luma_sums(const Parms &P, const uint8_t *c, const uint8_t *p, const uint8_t *n, int64_t *a, int64_t *b) {
    P.k->luma_sums(c, p, n, P.width, a, b);
}
inline void k_luma_sums(const Parms &P, const uint16_t *c, const uint16_t *p, const uint16_t *n, int64_t *a,
                        int64_t *b) {
    P.k->luma_sums16(c, p, n, P.width, a, b);
}
inline void k_remove(const Parms &P, const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *noise,
                     const uint8_t *motion, uint8_t *o, int lumad) {
    P.k->remove(p, n, c, noise, motion, o, P.width, lumad);
}
inline void k_remove(const Parms &P, const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *noise,
                     const uint8_t *motion, uint16_t *o, int lumad) {
    P.k->remove16(p, n, c, noise, motion, o, P.width, lumad, static_cast<uint16_t>(P.maxval));
}
inline void k_tsmooth(const Parms &P, const uint8_t *p, const uint8_t *n, uint8_t *o, const uint8_t *noise,
                      const uint8_t *motion) {
    P.k->tsmooth(p, n, o, noise, motion, P.width, P.tsmooth);
}
inline void k_tsmooth(const Parms &P, const uint16_t *p, const uint16_t *n, uint16_t *o, const uint8_t *noise,
                      const uint8_t *motion) {
    P.k->tsmooth16(p, n, o, noise, motion, P.width, P.tsmooth << P.shift);
}
inline void k_mark(const Parms &P, const uint8_t *c, const uint8_t *noise, const uint8_t *motion, uint8_t *o) {
    P.k->mark(c, noise, motion, o, P.width, P.mark_v);
}
inline void k_mark(const Parms &P, const uint16_t *c, const uint8_t *noise, const uint8_t *motion, uint16_t *o) {
    P.k->mark16(c, noise, motion, o, P.width, static_cast<uint16_t>(P.mark_v << P.shift), P.shift);
}
inline void k_map(const Parms &P, const uint8_t *noise, const uint8_t *motion, uint8_t *o) {
    P.k->map(noise, motion, o, P.width);
}
inline void k_map(const Parms &P, const uint8_t *noise, const uint8_t *motion, uint16_t *o) {
    P.k->map16(noise, motion, o, P.width, P.shift);
}

// Per-thread temporary buffers of the box filters, kept between calls.
struct Scratch {
    std::vector<int32_t> rows, tmp;
    std::vector<uint8_t> map;
    std::vector<uint8_t> orrow;
};

Scratch &scratch() {
    thread_local Scratch s;
    return s;
}

bool any_motion(const uint8_t *row, const Segment *c, const Parms &p) {
    const uint8_t *b = row + std::max(c->lx1 - 1, 0);
    const uint8_t *end = row + std::min(c->lx2 + 2, p.width);
    for (; b < end; ++b)
        if (*b) return true;
    return false;
}

// Sliding box (2 * rad_y + 1 lines of the same field, 2 * rad_x + 1 columns) over a 0/1 or 0/3 map.
// erode: dst = src && sum > thr ? 1 : 0; otherwise dst = sum ? val : 0.
void box_filter(const uint8_t *src, uint8_t *dst, int rad_x, int rad_y, const uint8_t *gate_map, int32_t thr,
                uint8_t val, const Parms &p) {
    const int w = p.width, h = p.height, Y = p.y_next;
    const ptrdiff_t pitch = p.pitch;
#if !DESPOT_COMPAT_363
    if (!gate_map && thr == 0 && rad_x <= 8 && rad_y <= 8) {
        // Dilate of a map without negative values: "sum of the box > 0" is "any pixel of the box set",
        // which is cheaper as an OR of the 2 * rad_y + 1 lines (of the same field) and of 2 * rad_x + 1 columns.
        std::vector<uint8_t> &orrow = scratch().orrow;
        orrow.assign(static_cast<size_t>(w) + 2 * rad_x, 0); // zero padding left and right
        const uint8_t *lines[17];
        for (int y = 0; y < h; ++y) {
            int n = 0;
            for (int k = -rad_y; k <= rad_y; ++k) {
                const int yy = y + k * Y;
                if (yy >= 0 && yy < h) lines[n++] = src + yy * pitch;
            }
            p.k->or_rows(lines, n, orrow.data() + rad_x, w);
            p.k->or_window(orrow.data() + rad_x, dst + y * pitch, w, rad_x, val);
        }
        return;
    }
#endif
    const int pad = rad_x + 1;
    const size_t rowlen = static_cast<size_t>(w) + 2 * pad + 1;
    std::vector<int32_t> &rows = scratch().rows, &tmp = scratch().tmp;
    rows.assign(rowlen * Y, 0);
    tmp.resize(static_cast<size_t>(w) + 2 * rad_x + 2);
    for (int y = -rad_y * Y; y < h; ++y) {
#if DESPOT_COMPAT_363
        int32_t *r = rows.data() + (y & (Y - 1)) + pad;
#else
        int32_t *r = rows.data() + (y & (Y - 1)) * rowlen + pad;
#endif
        const int y1 = y - rad_y * Y - Y;
        const int y2 = y + rad_y * Y;
        p.k->colsum(r, y1 >= 0 ? src + y1 * pitch : nullptr, y2 < h ? src + y2 * pitch : nullptr, w);
        if (y >= 0)
            p.k->window(r, w, rad_x, gate_map ? gate_map + y * pitch : nullptr, thr, val, dst + y * pitch,
                        tmp.data());
    }
}

void binarize_map(const uint8_t *src, uint8_t *dst, uint8_t val, const Parms &p) {
    for (int y = 0; y < p.height; ++y) p.k->binarize(src + y * p.pitch, dst + y * p.pitch, p.width, val);
}

// 1D blur along the row near spots (seg=0 mode)
template <typename T>
void lineblur(const uint8_t *c_noise, const uint8_t *c_motion, T *o, int width, const Parms &parms) {
    int b[10];
    const int xleft = 2 * parms.blur;
    const int xright = width - 2 * parms.blur;
    // only pixels with noise do anything: jump from one to the next
    auto next = [&](int x, int end) { return parms.k->next_above(c_noise, x, end, 0); };

    switch (parms.blur) {
    case 0: break;
    case 1:
        for (int x = next(xleft, xright); x < xright; x = next(x + 1, xright)) {
            if (c_noise[x] && !c_motion[x]) {
                if (!c_noise[x - 1]) { // spot begins at x
                    b[0] = (o[x - 2] + 2 * o[x - 1] + o[x]) / 4;
                    b[1] = (o[x - 1] + 2 * o[x] + o[x + 1]) / 4;
                    o[x - 1] = b[0];
                    o[x] = b[1];
                } else if (!c_noise[x + 1]) { // spot ends at x
                    b[0] = (o[x - 1] + 2 * o[x] + o[x + 1]) / 4;
                    b[1] = (o[x] + 2 * o[x + 1] + o[x + 2]) / 4;
                    o[x] = b[0];
                    o[x + 1] = b[1];
                }
            }
        }
        break;
    case 2:
        for (int x = next(xleft, xright); x < xright; x = next(x + 1, xright)) {
            if (c_noise[x] && !c_motion[x]) {
                if (!c_noise[x - 1]) {
                    b[3] = (o[x - 4] + 2 * o[x - 3] + 3 * o[x - 2] + 2 * o[x - 1] + o[x]) / 9;
                    b[4] = (o[x - 3] + 2 * o[x - 2] + 3 * o[x - 1] + 2 * o[x] + o[x + 1]) / 9;
                    b[5] = (o[x - 2] + 2 * o[x - 1] + 3 * o[x] + 2 * o[x + 1] + o[x + 2]) / 9;
                    b[6] = (o[x - 1] + 2 * o[x] + 3 * o[x + 1] + 2 * o[x + 2] + o[x + 3]) / 9;
                    o[x - 2] = b[3];
                    o[x - 1] = b[4];
                    o[x] = b[5];
                    o[x + 1] = b[6];
                } else if (!c_noise[x + 1]) {
                    b[4] = (o[x - 3] + 2 * o[x - 2] + 3 * o[x - 1] + 2 * o[x] + o[x + 1]) / 9;
                    b[5] = (o[x - 2] + 2 * o[x - 1] + 3 * o[x] + 2 * o[x + 1] + o[x + 2]) / 9;
                    b[6] = (o[x - 1] + 2 * o[x] + 3 * o[x + 1] + 2 * o[x + 2] + o[x + 3]) / 9;
                    b[7] = (o[x] + 2 * o[x + 1] + 3 * o[x + 2] + 2 * o[x + 3] + o[x + 4]) / 9;
                    o[x - 1] = b[4];
                    o[x] = b[5];
                    o[x + 1] = b[6];
                    o[x + 2] = b[7];
                }
            }
        }
        break;
    case 3:
        for (int x = next(xleft, xright); x < xright; x = next(x + 1, xright)) {
            if (c_noise[x] && !c_motion[x]) {
                if (!c_noise[x - 1]) {
                    b[2] = (o[x - 6] + 2 * o[x - 5] + 3 * o[x - 4] + 4 * o[x - 3] + 3 * o[x - 2] + 2 * o[x - 1] + o[x]) / 16;
                    b[3] = (o[x - 5] + 2 * o[x - 4] + 3 * o[x - 3] + 4 * o[x - 2] + 3 * o[x - 1] + 2 * o[x] + o[x + 1]) / 16;
                    b[4] = (o[x - 4] + 2 * o[x - 3] + 3 * o[x - 2] + 4 * o[x - 1] + 3 * o[x] + 2 * o[x + 1] + o[x + 2]) / 16;
                    b[5] = (o[x - 3] + 2 * o[x - 2] + 3 * o[x - 1] + 4 * o[x] + 3 * o[x + 1] + 2 * o[x + 2] + o[x + 3]) / 16;
                    b[6] = (o[x - 2] + 2 * o[x - 1] + 3 * o[x] + 4 * o[x + 1] + 3 * o[x + 2] + 2 * o[x + 3] + o[x + 4]) / 16;
                    b[7] = (o[x - 1] + 2 * o[x] + 3 * o[x + 1] + 4 * o[x + 2] + 3 * o[x + 3] + 2 * o[x + 4] + o[x + 5]) / 16;
                    o[x - 3] = b[2];
                    o[x - 2] = b[3];
                    o[x - 1] = b[4];
                    o[x] = b[5];
                    o[x + 1] = b[6];
                    o[x + 2] = b[7];
                } else if (!c_noise[x + 1]) {
                    b[3] = (o[x - 5] + 2 * o[x - 4] + 3 * o[x - 3] + 4 * o[x - 2] + 3 * o[x - 1] + 2 * o[x] + o[x + 1]) / 16;
                    b[4] = (o[x - 4] + 2 * o[x - 3] + 3 * o[x - 2] + 4 * o[x - 1] + 3 * o[x] + 2 * o[x + 1] + o[x + 2]) / 16;
                    b[5] = (o[x - 3] + 2 * o[x - 2] + 3 * o[x - 1] + 4 * o[x] + 3 * o[x + 1] + 2 * o[x + 2] + o[x + 3]) / 16;
                    b[6] = (o[x - 2] + 2 * o[x - 1] + 3 * o[x] + 4 * o[x + 1] + 3 * o[x + 2] + 2 * o[x + 3] + o[x + 4]) / 16;
                    b[7] = (o[x - 1] + 2 * o[x] + 3 * o[x + 1] + 4 * o[x + 2] + 3 * o[x + 3] + 2 * o[x + 4] + o[x + 5]) / 16;
                    b[8] = (o[x] + 2 * o[x + 1] + 3 * o[x + 2] + 4 * o[x + 3] + 3 * o[x + 4] + 2 * o[x + 5] + o[x + 6]) / 16;
                    o[x - 2] = b[3];
                    o[x - 1] = b[4];
                    o[x] = b[5];
                    o[x + 1] = b[6];
                    o[x + 2] = b[7];
                    o[x + 3] = b[8];
                }
            }
        }
        break;
    case 4:
    default:
        for (int x = next(8, width - 8); x < width - 8; x = next(x + 1, width - 8)) {
            if (c_noise[x] && !c_motion[x]) {
                if (!c_noise[x - 1]) {
                    b[1] = (o[x - 8] + 2 * o[x - 7] + 3 * o[x - 6] + 4 * o[x - 5] + 5 * o[x - 4] + 4 * o[x - 3] + 3 * o[x - 2] + 2 * o[x - 1] + o[x]) / 25;
                    b[2] = (o[x - 7] + 2 * o[x - 6] + 3 * o[x - 5] + 4 * o[x - 4] + 5 * o[x - 3] + 4 * o[x - 2] + 3 * o[x - 1] + 2 * o[x] + o[x + 1]) / 25;
                    b[3] = (o[x - 6] + 2 * o[x - 5] + 3 * o[x - 4] + 4 * o[x - 3] + 5 * o[x - 2] + 4 * o[x - M11] + 3 * o[x] + 2 * o[x + 1] + o[x + 2]) / 25;
                    b[4] = (o[x - 5] + 2 * o[x - 4] + 3 * o[x - 3] + 4 * o[x - 2] + 5 * o[x - 1] + 4 * o[x] + 3 * o[x + 1] + 2 * o[x + 2] + o[x + 3]) / 25;
                    b[5] = (o[x - 4] + 2 * o[x - 3] + 3 * o[x - 2] + 4 * o[x - 1] + 5 * o[x] + 4 * o[x + 1] + 3 * o[x + 2] + 2 * o[x + 3] + o[x + 4]) / 25;
                    b[6] = (o[x - 3] + 2 * o[x - 2] + 3 * o[x - 1] + 4 * o[x] + 5 * o[x + 1] + 4 * o[x + 2] + 3 * o[x + 3] + 2 * o[x + 4] + o[x + 5]) / 25;
                    b[7] = (o[x - 2] + 2 * o[x - 1] + 3 * o[x] + 4 * o[x + 1] + 5 * o[x + 2] + 4 * o[x + 3] + 3 * o[x + 4] + 2 * o[x + 5] + o[x + 6]) / 25;
                    b[8] = (o[x - 1] + 2 * o[x] + 3 * o[x + 1] + 4 * o[x + 2] + 5 * o[x + 3] + 4 * o[x + 4] + 3 * o[x + 5] + 2 * o[x + 6] + o[x + 7]) / 25;
                    o[x - 4] = b[1];
                    o[x - 3] = b[2];
                    o[x - 2] = b[3];
                    o[x - 1] = b[4];
                    o[x] = b[5];
                    o[x + 1] = b[6];
                    o[x + 2] = b[7];
                    o[x + 3] = b[8];
                } else if (!c_noise[x + 1]) {
                    b[2] = (o[x - 7] + 2 * o[x - 6] + 3 * o[x - 5] + 4 * o[x - 4] + 5 * o[x - 3] + 4 * o[x - 2] + 3 * o[x - 1] + 2 * o[x] + o[x + 1]) / 25;
                    b[3] = (o[x - 6] + 2 * o[x - 5] + 3 * o[x - 4] + 4 * o[x - 3] + 5 * o[x - 2] + 4 * o[x - M11] + 3 * o[x] + 2 * o[x + 1] + o[x + 2]) / 25;
                    b[4] = (o[x - 5] + 2 * o[x - 4] + 3 * o[x - 3] + 4 * o[x - 2] + 5 * o[x - 1] + 4 * o[x] + 3 * o[x + 1] + 2 * o[x + 2] + o[x + 3]) / 25;
                    b[5] = (o[x - 4] + 2 * o[x - 3] + 3 * o[x - 2] + 4 * o[x - 1] + 5 * o[x] + 4 * o[x + 1] + 3 * o[x + 2] + 2 * o[x + 3] + o[x + 4]) / 25;
                    b[6] = (o[x - 3] + 2 * o[x - 2] + 3 * o[x - 1] + 4 * o[x] + 5 * o[x + 1] + 4 * o[x + 2] + 3 * o[x + 3] + 2 * o[x + 4] + o[x + 5]) / 25;
                    b[7] = (o[x - 2] + 2 * o[x - 1] + 3 * o[x] + 4 * o[x + 1] + 5 * o[x + 2] + 4 * o[x + 3] + 3 * o[x + 4] + 2 * o[x + 5] + o[x + 6]) / 25;
                    b[8] = (o[x - 1] + 2 * o[x] + 3 * o[x + 1] + 4 * o[x + 2] + 5 * o[x + 3] + 4 * o[x + 4] + 3 * o[x + 5] + 2 * o[x + 6] + o[x + 7]) / 25;
                    b[9] = (o[x] + 2 * o[x + 1] + 3 * o[x + 2] + 4 * o[x + 3] + 5 * o[x + 4] + 4 * o[x + 5] + 3 * o[x + 6] + 2 * o[x + 7] + o[x + 8]) / 25;
                    o[x - 3] = b[2];
                    o[x - 2] = b[3];
                    o[x - 1] = b[4];
                    o[x] = b[5];
                    o[x + 1] = b[6];
                    o[x + 2] = b[7];
                    o[x + 3] = b[8];
                    o[x + 4] = b[9];
                }
            }
        }
    }
}

// 1D triangle blur at both ends of a removed segment
template <typename T>
void segment_blur(T *o, int x1, int x2, int blur) {
    int b[16];
    switch (blur) {
    case 0: break;
    case 1:
        b[0] = (o[x1 - 2] + 2 * o[x1 - 1] + o[x1]) / 4;
        b[1] = (o[x1 - 1] + 2 * o[x1] + o[x1 + 1]) / 4;
        o[x1 - 1] = b[0];
        o[x1] = b[1];
        b[2] = (o[x2 - 1] + 2 * o[x2] + o[x2 + 1]) / 4;
        b[3] = (o[x2] + 2 * o[x2 + 1] + o[x2 + 2]) / 4;
        o[x2] = b[2];
        o[x2 + 1] = b[3];
        break;
    case 2:
        b[0] = (o[x1 - 4] + 2 * o[x1 - 3] + 3 * o[x1 - 2] + 2 * o[x1 - 1] + o[x1]) / 9;
        b[1] = (o[x1 - 3] + 2 * o[x1 - 2] + 3 * o[x1 - 1] + 2 * o[x1] + o[x1 + 1]) / 9;
        b[2] = (o[x1 - 2] + 2 * o[x1 - 1] + 3 * o[x1] + 2 * o[x1 + 1] + o[x1 + 2]) / 9;
        b[3] = (o[x1 - 1] + 2 * o[x1] + 3 * o[x1 + 1] + 2 * o[x1 + 2] + o[x1 + 3]) / 9;
        o[x1 - 2] = b[0];
        o[x1 - 1] = b[1];
        o[x1] = b[2];
        o[x1 + 1] = b[3];
        b[4] = (o[x2 - 3] + 2 * o[x2 - 2] + 3 * o[x2 - 1] + 2 * o[x2] + o[x2 + 1]) / 9;
        b[5] = (o[x2 - 2] + 2 * o[x2 - 1] + 3 * o[x2] + 2 * o[x2 + 1] + o[x2 + 2]) / 9;
        b[6] = (o[x2 - 1] + 2 * o[x2] + 3 * o[x2 + 1] + 2 * o[x2 + 2] + o[x2 + 3]) / 9;
        b[7] = (o[x2] + 2 * o[x2 + 1] + 3 * o[x2 + 2] + 2 * o[x2 + 3] + o[x2 + 4]) / 9;
        o[x2 - 1] = b[4];
        o[x2] = b[5];
        o[x2 + 1] = b[6];
        o[x2 + 2] = b[7];
        break;
    case 3:
        b[0] = (o[x1 - 6] + 2 * o[x1 - 5] + 3 * o[x1 - 4] + 4 * o[x1 - 3] + 3 * o[x1 - 2] + 2 * o[x1 - 1] + o[x1]) / 16;
        b[1] = (o[x1 - 5] + 2 * o[x1 - 4] + 3 * o[x1 - 3] + 4 * o[x1 - 2] + 3 * o[x1 - 1] + 2 * o[x1] + o[x1 + 1]) / 16;
        b[2] = (o[x1 - 4] + 2 * o[x1 - 3] + 3 * o[x1 - 2] + 4 * o[x1 - 1] + 3 * o[x1] + 2 * o[x1 + 1] + o[x1 + 2]) / 16;
        b[3] = (o[x1 - 3] + 2 * o[x1 - 2] + 3 * o[x1 - 1] + 4 * o[x1] + 3 * o[x1 + 1] + 2 * o[x1 + 2] + o[x1 + 3]) / 16;
        b[4] = (o[x1 - 2] + 2 * o[x1 - 1] + 3 * o[x1] + 4 * o[x1 + 1] + 3 * o[x1 + 2] + 2 * o[x1 + 3] + o[x1 + 4]) / 16;
        b[5] = (o[x1 - 1] + 2 * o[x1] + 3 * o[x1 + 1] + 4 * o[x1 + 2] + 3 * o[x1 + 3] + 2 * o[x1 + 4] + o[x1 + 5]) / 16;
        o[x1 - 3] = b[0];
        o[x1 - 2] = b[1];
        o[x1 - 1] = b[2];
        o[x1] = b[3];
        o[x1 + 1] = b[4];
        o[x1 + 2] = b[5];
        b[6] = (o[x2 - 5] + 2 * o[x2 - 4] + 3 * o[x2 - 3] + 4 * o[x2 - 2] + 3 * o[x2 - 1] + 2 * o[x2] + o[x2 + 1]) / 16;
        b[7] = (o[x2 - 4] + 2 * o[x2 - 3] + 3 * o[x2 - 2] + 4 * o[x2 - 1] + 3 * o[x2] + 2 * o[x2 + 1] + o[x2 + 2]) / 16;
        b[8] = (o[x2 - 3] + 2 * o[x2 - 2] + 3 * o[x2 - 1] + 4 * o[x2] + 3 * o[x2 + 1] + 2 * o[x2 + 2] + o[x2 + 3]) / 16;
        b[9] = (o[x2 - 2] + 2 * o[x2 - 1] + 3 * o[x2] + 4 * o[x2 + 1] + 3 * o[x2 + 2] + 2 * o[x2 + 3] + o[x2 + 4]) / 16;
        b[10] = (o[x2 - 1] + 2 * o[x2] + 3 * o[x2 + 1] + 4 * o[x2 + 2] + 3 * o[x2 + 3] + 2 * o[x2 + 4] + o[x2 + 5]) / 16;
        b[11] = (o[x2] + 2 * o[x2 + 1] + 3 * o[x2 + 2] + 4 * o[x2 + 3] + 3 * o[x2 + 4] + 2 * o[x2 + 5] + o[x2 + 6]) / 16;
        o[x2 - 2] = b[6];
        o[x2 - 1] = b[7];
        o[x2] = b[8];
        o[x2 + 1] = b[9];
        o[x2 + 2] = b[10];
        o[x2 + 3] = b[11];
        break;
    case 4:
    default:
        b[0] = (o[x1 - 8] + 2 * o[x1 - 7] + 3 * o[x1 - 6] + 4 * o[x1 - 5] + 5 * o[x1 - 4] + 4 * o[x1 - 3] + 3 * o[x1 - 2] + 2 * o[x1 - 1] + o[x1]) / 25;
        b[1] = (o[x1 - 7] + 2 * o[x1 - 6] + 3 * o[x1 - 5] + 4 * o[x1 - 4] + 5 * o[x1 - 3] + 4 * o[x1 - 2] + 3 * o[x1 - 1] + 2 * o[x1] + o[x1 + 1]) / 25;
        b[2] = (o[x1 - 6] + 2 * o[x1 - 5] + 3 * o[x1 - 4] + 4 * o[x1 - 3] + 5 * o[x1 - 2] + 4 * o[x1 - M11] + 3 * o[x1] + 2 * o[x1 + 1] + o[x1 + 2]) / 25;
        b[3] = (o[x1 - 5] + 2 * o[x1 - 4] + 3 * o[x1 - 3] + 4 * o[x1 - 2] + 5 * o[x1 - 1] + 4 * o[x1] + 3 * o[x1 + 1] + 2 * o[x1 + 2] + o[x1 + 3]) / 25;
        b[4] = (o[x1 - 4] + 2 * o[x1 - 3] + 3 * o[x1 - 2] + 4 * o[x1 - 1] + 5 * o[x1] + 4 * o[x1 + 1] + 3 * o[x1 + 2] + 2 * o[x1 + 3] + o[x1 + 4]) / 25;
        b[5] = (o[x1 - 3] + 2 * o[x1 - 2] + 3 * o[x1 - 1] + 4 * o[x1] + 5 * o[x1 + 1] + 4 * o[x1 + 2] + 3 * o[x1 + 3] + 2 * o[x1 + 4] + o[x1 + 5]) / 25;
        b[6] = (o[x1 - 2] + 2 * o[x1 - 1] + 3 * o[x1] + 4 * o[x1 + 1] + 5 * o[x1 + 2] + 4 * o[x1 + 3] + 3 * o[x1 + 4] + 2 * o[x1 + 5] + o[x1 + 6]) / 25;
        b[7] = (o[x1 - 1] + 2 * o[x1] + 3 * o[x1 + 1] + 4 * o[x1 + 2] + 5 * o[x1 + 3] + 4 * o[x1 + 4] + 3 * o[x1 + 5] + 2 * o[x1 + 6] + o[x1 + 7]) / 25;
        o[x1 - 4] = b[0];
        o[x1 - 3] = b[1];
        o[x1 - 2] = b[2];
        o[x1 - 1] = b[3];
        o[x1] = b[4];
        o[x1 + 1] = b[5];
        o[x1 + 2] = b[6];
        o[x1 + 3] = b[7];
        b[8] = (o[x2 - 7] + 2 * o[x2 - 6] + 3 * o[x2 - 5] + 4 * o[x2 - 4] + 5 * o[x2 - 3] + 4 * o[x2 - 2] + 3 * o[x2 - 1] + 2 * o[x2] + o[x2 + 1]) / 25;
        b[9] = (o[x2 - 6] + 2 * o[x2 - 5] + 3 * o[x2 - 4] + 4 * o[x2 - 3] + 5 * o[x2 - 2] + 4 * o[x2 - M11] + 3 * o[x2] + 2 * o[x2 + 1] + o[x2 + 2]) / 25;
        b[10] = (o[x2 - 5] + 2 * o[x2 - 4] + 3 * o[x2 - 3] + 4 * o[x2 - 2] + 5 * o[x2 - 1] + 4 * o[x2] + 3 * o[x2 + 1] + 2 * o[x2 + 2] + o[x2 + 3]) / 25;
        b[11] = (o[x2 - 4] + 2 * o[x2 - 3] + 3 * o[x2 - 2] + 4 * o[x2 - 1] + 5 * o[x2] + 4 * o[x2 + 1] + 3 * o[x2 + 2] + 2 * o[x2 + 3] + o[x2 + 4]) / 25;
        b[12] = (o[x2 - 3] + 2 * o[x2 - 2] + 3 * o[x2 - 1] + 4 * o[x2] + 5 * o[x2 + 1] + 4 * o[x2 + 2] + 3 * o[x2 + 3] + 2 * o[x2 + 4] + o[x2 + 5]) / 25;
        b[13] = (o[x2 - 2] + 2 * o[x2 - 1] + 3 * o[x2] + 4 * o[x2 + 1] + 5 * o[x2 + 2] + 4 * o[x2 + 3] + 3 * o[x2 + 4] + 2 * o[x2 + 5] + o[x2 + 6]) / 25;
        b[14] = (o[x2 - 1] + 2 * o[x2] + 3 * o[x2 + 1] + 4 * o[x2 + 2] + 5 * o[x2 + 3] + 4 * o[x2 + 4] + 3 * o[x2 + 5] + 2 * o[x2 + 6] + o[x2 + 7]) / 25;
        b[15] = (o[x2] + 2 * o[x2 + 1] + 3 * o[x2 + 2] + 4 * o[x2 + 3] + 5 * o[x2 + 4] + 4 * o[x2 + 5] + 3 * o[x2 + 6] + 2 * o[x2 + 7] + o[x2 + 8]) / 25;
        o[x2 - 3] = b[8];
        o[x2 - 2] = b[9];
        o[x2 - 1] = b[10];
        o[x2] = b[11];
        o[x2 + 1] = b[12];
        o[x2 + 2] = b[13];
        o[x2 + 3] = b[14];
        o[x2 + 4] = b[15];
    }
}

void print_timestamp(char *buf, size_t len, double t) {
    const int all = int(std::floor(t * 100 + 0.5));
    const int cs = all % 100;
    const int s = (all / 100) % 60;
    const int m = (all / 6000) % 60;
    const int h = all / 360000;
    std::snprintf(buf, len, "%01d:%02d:%02d.%02d", h, m, s, cs);
}

// Luma noise map rows covered by chroma row j (1 << ssh rows; interlaced: rows of the same field).
int chroma_rows(int j, int ssh, int y_next, int height, int rows[4]) {
    const int k = 1 << ssh;
    for (int i = 0; i < k; ++i) {
        int r;
        if (ssh == 0 || y_next == 1)
            r = (j << ssh) + i;
        else // interlaced: chroma row j belongs to field j & 1
            r = 2 * (((j >> 1) << ssh) + i) + (j & 1);
        rows[i] = std::min(r, height - 1);
    }
    return k;
}

// Does the luma block of chroma pixel i touch the noise map? strict: only values of 255 count.
inline bool chroma_spot(const uint8_t *const *rows, int nrows, int i, int ssw, int width, bool strict) {
    const int x0 = i << ssw, x1 = std::min((i + 1) << ssw, width);
    for (int r = 0; r < nrows; ++r)
        for (int x = x0; x < x1; ++x)
            if (strict ? rows[r][x] == 255 : rows[r][x] != 0) return true;
    return false;
}

} // namespace

// ------------------------------------------------------------------------------------------------

template <typename T>
void find_outliers(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const T *n, ptrdiff_t npitch,
                   uint8_t *o, const Parms &parms) {
    // o is the difference of c to the range of prev and next (ranked: of 3 neighbours each)
    for (int y = 0; y < parms.height; ++y) { // the kernels write every column of the frame width
        k_outliers(parms, p, n, c, o);
        p += ppitch;
        c += cpitch;
        n += npitch;
        o += parms.pitch;
    }
}

void find_sizes(uint8_t *noise, Segments &segs, const Parms &parms) {
    segs.clear();
    combine_xs(noise, segs, parms);
    combine_ys(segs, parms);
    to_size(segs, parms);
}

void mark_noise(Segments &segs, uint8_t *l, const Parms &p) {
    std::memset(l, 0, p.size);
    const Segment *c = segs.data;
    for (int y = 0; y < p.height; ++y) {
        for (; c->ly == y; ++c)
            if (c->data.s.is_noise) std::memset(l + c->lx1, 0xFF, c->lx2 - c->lx1 + 1);
        l += p.pitch;
    }
}

void reject_on_motion(Segments &segs, const uint8_t *motion, const Parms &p) {
    // reverse order is faster because most segments have a master segment
    Segment *c = segs.end - 1;
    const uint8_t *mot = motion + (p.height - 1) * static_cast<ptrdiff_t>(p.pitch);
    const ptrdiff_t step = static_cast<ptrdiff_t>(p.pitch) * p.y_next;

    // First pass: reject segments over or near motion pixels and all their masters
    for (int y = p.height - 1; y >= 0; y--) {
        for (; c >= segs.data && c->ly == y; c--) {
            if (!c->data.s.is_noise) continue;
            if (any_motion(mot, c, p)) reject_chain(c, p);
            if (y >= p.y_next && any_motion(mot - step, c, p)) reject_chain(c, p);
            if (y < p.height - p.y_next && any_motion(mot + step, c, p)) reject_chain(c, p);
        }
        mot -= p.pitch;
    }

    if (p.seg == 2) // second pass: reject all slaves of rejected masters
        for (c = segs.data; c != segs.end; ++c)
            if (c->data.nowis) c->data.s.is_noise = c->data.nowis->data.s.is_noise;
}

template <typename T>
void find_motion(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, uint8_t *motion,
                 const Parms &parms) {
    for (int y = 0; y < parms.height; ++y) {
        k_motion(parms, p, c, motion);
        p += ppitch;
        c += cpitch;
        motion += parms.pitch;
    }
}

void motion_denoise(uint8_t *moving, const Parms &p) {
    // Based on the motion map denoise of Donald Graft's Smart Deinterlacer, rewritten by
    // K. Atkinson to O(w*h). Input: 3 = motion, 1 = motion and noise, 0 = none.
    // Erode removes small motion objects, dilate enlarges the remaining ones. Output: 0 or 255.
    const int MHEIGHT_2 = p.mheight / 2;
    const int MWIDTH_2 = p.mwidth / 2;
    const int MPx3 = static_cast<int>((int64_t(p.merode) * 3 * (p.mwidth + 1) * (p.mheight + 1)) / 100);
    if (MHEIGHT_2 == 0 && MWIDTH_2 == 0) return;

    std::vector<uint8_t> &fmoving = scratch().map; // fully written before it is read
    fmoving.resize(p.size);
    if (MPx3 > 0)
        box_filter(moving, fmoving.data(), MWIDTH_2, MHEIGHT_2, moving, MPx3, 1, p); // erode
    else
        binarize_map(moving, fmoving.data(), 1, p);
    box_filter(fmoving.data(), moving, MWIDTH_2, MHEIGHT_2, nullptr, 0, 255, p); // dilate
}

void noise_dilate(uint8_t *noise, const Parms &p) {
    // dilate the noise map with range Parms.dilate (dilate stage of motion_denoise)
    if (p.dilate <= 0) return;
    std::vector<uint8_t> &fmoving = scratch().map; // fully written before it is read
    fmoving.resize(p.size);
    binarize_map(noise, fmoving.data(), 1, p);
    box_filter(fmoving.data(), noise, p.dilate, p.dilate, nullptr, 0, 255, p);
}

void motion_scene(uint8_t *motion, const Parms &parms) {
    // whole frame is motion at a scene change (motion pixels percent > mscene)
    const int width_mod8 = (parms.width / 8) * 8;
    if (width_mod8 == 0) return;
    int64_t count = 0;
    for (int y = 0; y < parms.height; ++y) count += parms.k->count_nonzero(motion + y * parms.pitch, width_mod8);
    const int64_t mpercent = (count * 100) / (int64_t(parms.height) * width_mod8);
    if (mpercent > parms.mscene)
        for (int y = 0; y < parms.height; y++) std::memset(motion + y * parms.pitch, 255, parms.width);
}

void motion_merge(const uint8_t *c_motion, const uint8_t *n_motion, uint8_t *m_motion, const Parms &parms) {
    for (int y = 0; y < parms.height; ++y) {
        const ptrdiff_t off = y * static_cast<ptrdiff_t>(parms.pitch);
        parms.k->merge(c_motion + off, n_motion + off, m_motion + off, parms.width);
    }
}

void noise_to_one(const uint8_t *p_noise, const uint8_t *c_noise, uint8_t *c_motion, const Parms &parms) {
    // motion and previous or current noise: 1, else no change
    for (int y = 0; y < parms.height; ++y) {
        const ptrdiff_t off = y * static_cast<ptrdiff_t>(parms.pitch);
        parms.k->noise_to_one(p_noise + off, c_noise + off, c_motion + off, parms.width);
    }
}

template <typename E>
void add_external_mask(const E *ext, ptrdiff_t ext_pitch, int ext_shift, uint8_t *motion, const Parms &parms) {
    // result = ext > 127 || motion
    for (int y = 0; y < parms.height; y++) {
        k_ext_mask(parms, ext, motion, ext_shift);
        motion += parms.pitch;
        ext += ext_pitch;
    }
}

template <typename T>
void cond_median(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const uint8_t *, const uint8_t *c_motion,
                 const T *n, ptrdiff_t npitch, T *o, ptrdiff_t opitch, const Parms &parms) {
    for (int y = 0; y < parms.height; ++y) {
        k_median(parms, p, n, c, c_motion, o);
        p += ppitch;
        c += cpitch;
        n += npitch;
        o += opitch;
        c_motion += parms.pitch;
    }
}

template <typename T>
void remove_outliers(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const uint8_t *c_noise,
                     const uint8_t *c_motion, const T *n, ptrdiff_t npitch, T *o, ptrdiff_t opitch,
                     const Parms &parms) {
    const int width = parms.width;
    for (int y = 0; y < parms.height; ++y) {
        int lumad = 0;
        if (parms.fitluma) { // line luma correction
            int64_t sc, spn;
            k_luma_sums(parms, c, p, n, &sc, &spn);
            const int lumac = static_cast<int>(sc / width);
            lumad = lumac - static_cast<int>(spn / (int64_t(width) * 2));
        }
        k_remove(parms, p, n, c, c_noise, c_motion, o, lumad);
        if (parms.blur) lineblur(c_noise, c_motion, o, width, parms); // blur line near spot segments
        if (parms.tsmooth) k_tsmooth(parms, p, n, o, c_noise, c_motion); // after blur
        p += ppitch;
        c += cpitch;
        n += npitch;
        o += opitch;
        c_noise += parms.pitch;
        c_motion += parms.pitch;
    }
}

template <typename T>
void mark_outliers(const T *, ptrdiff_t, const T *c, ptrdiff_t cpitch, const uint8_t *c_noise, const uint8_t *c_motion,
                   const T *, ptrdiff_t, T *o, ptrdiff_t opitch, const Parms &parms) {
    for (int y = 0; y < parms.height; ++y) {
        k_mark(parms, c, c_noise, c_motion, o);
        c += cpitch;
        o += opitch;
        c_noise += parms.pitch;
        c_motion += parms.pitch;
    }
}

template <typename T>
void map_outliers(const T *, ptrdiff_t, const T *, ptrdiff_t, const uint8_t *c_noise, const uint8_t *c_motion,
                  const T *, ptrdiff_t, T *o, ptrdiff_t opitch, const Parms &parms) {
    for (int y = 0; y < parms.height; ++y) {
        k_map(parms, c_noise, c_motion, o);
        o += opitch;
        if (c_noise) c_noise += parms.pitch;
        c_motion += parms.pitch;
    }
}

template <typename T>
void temporal_smooth(Segments &segs, const T *p, ptrdiff_t ppitch, const T *n, ptrdiff_t npitch, T *o,
                     ptrdiff_t opitch, uint8_t *c_noise, const uint8_t *c_motion, const Parms &parms) {
    const int width = parms.width;
    const Segment *seg = segs.data;
    for (int y = 0; y < parms.height; ++y) {
        for (; seg->ly == y; ++seg) {
            const int x1 = std::max<int>(seg->lx1, 0);
            const int x2 = std::min<int>(seg->lx2, width - 1);
            for (int x = x1; x <= x2; ++x) c_noise[x] = 255;
        }
        k_tsmooth(parms, p, n, o, c_noise, c_motion);
        p += ppitch;
        n += npitch;
        o += opitch;
        c_noise += parms.pitch;
        c_motion += parms.pitch;
    }
}

template <typename T>
void remove_segments(Segments &segs, const T *p, ptrdiff_t Ppitch, const T *n, ptrdiff_t Npitch, T *o,
                     ptrdiff_t Opitch, uint8_t *c_noise, const Parms &parms) {
    const int width = parms.width;
    const int blur = parms.blur;
    const bool fitluma = parms.fitluma;
    const int dilateX = parms.dilate;
    const int dilateY = parms.dilate / parms.y_next;
    const Segment *seg = segs.data;

    const ptrdiff_t opitch = Opitch * parms.y_next;
    const ptrdiff_t ppitch = Ppitch * parms.y_next;
    const ptrdiff_t npitch = Npitch * parms.y_next;
    const ptrdiff_t cnpitch = static_cast<ptrdiff_t>(parms.pitch) * parms.y_next; // noise

#if DESPOT_COMPAT_363
    const int left = dilateX + 1 + blur;
    const int right = width - dilateX - 2 - blur;
#else
    // the blur reads 2 * blur pixels beyond the (dilated) segment ends
    const int left = dilateX + std::max(1, 2 * blur);
    const int right = width - 1 - dilateX - std::max(1, 2 * blur);
#endif

    std::memset(c_noise, 0, parms.size);

    o += opitch * dilateY; // skip first dilateY lines
    p += ppitch * dilateY;
    n += npitch * dilateY;
    c_noise += cnpitch * dilateY;

    const int maxval = parms.maxval;
    auto fill = [&](T *orow, const T *prow, const T *nrow, uint8_t *noiserow, int x1, int x2, int lumaAdd) {
        for (int x = x1; x <= x2; ++x) {
            orow[x] = static_cast<T>(std::min(std::max((prow[x] + nrow[x]) / 2 + lumaAdd, 0), maxval));
            noiserow[x] = 255;
        }
        if (blur) segment_blur(orow, x1, x2, blur);
    };

    for (int y = dilateY * parms.y_next; y < parms.height - dilateY * parms.y_next; ++y) {
        for (; seg->ly < y; ++seg) { // skip segments in skipped dilateY lines
        }
        for (; seg->ly == y; ++seg) {
            if (!seg->data.s.is_noise) continue;
            const int x1 = std::max<int>(seg->lx1, left) - dilateX;
            const int x2 = std::min<int>(seg->lx2, right) + dilateX;
#if !DESPOT_COMPAT_363
            if (x1 > x2) continue; // segment entirely in the border zone
#endif
            const int dilateYC = dilateY;

            // local (segment) luma correction to fit luma to neighbour pixels
            int lumaAdd = 0;
            if (fitluma) {
                if (y >= parms.y_next && y < parms.height - parms.y_next) { // middle lines
                    lumaAdd = o[x1 - 1] + o[x2 + 1] - (p[x1 - 1] + p[x2 + 1] + n[x1 - 1] + n[x2 + 1]) +
                              (o[x1 - 1 + opitch] + o[x2 + 1 + opitch] + o[x1 - 1 - opitch] + o[x2 + 1 - opitch]) / 2;
                    lumaAdd /= 4;
                } else {
                    lumaAdd = o[x1 - 1] + o[x2 + 1] - (p[x1 - 1] + p[x2 + 1] + n[x1 - 1] + n[x2 + 1]) / 2;
                    lumaAdd /= 2;
                }
            }

            fill(o, p, n, c_noise, x1, x2, lumaAdd);

            if (dilateYC > 0) {
                if (!(seg->what & B_LINK)) // the top master (or single): dilate to top
                    for (int dy = 1; dy <= dilateYC; dy++)
                        fill(o - opitch * dy, p - ppitch * dy, n - npitch * dy, c_noise - cnpitch * dy, x1, x2, lumaAdd);
                if (!(seg->what & B_MASTER)) // the bottom link (or single): dilate to bottom
                    for (int dy = 1; dy <= dilateYC; dy++)
                        fill(o + opitch * dy, p + ppitch * dy, n + npitch * dy, c_noise + cnpitch * dy, x1, x2, lumaAdd);
            }
        }
        p += Ppitch;
        n += Npitch;
        o += Opitch;
        c_noise += parms.pitch;
    }
}

template <typename T>
void clean_color_plane(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const T *n, ptrdiff_t npitch,
                       int widthUV, int heightUV, int ssw, int ssh, const uint8_t *c_noise, T *o, ptrdiff_t opitch,
                       const Parms &parms) {
    // chroma at removed spots: mean of previous, current and next frame
    constexpr int mult = (1 << 10) / 3;
    for (int j = 0; j < heightUV; j++) {
        int r[4];
        const int nr = chroma_rows(j, ssh, parms.y_next, parms.height, r);
        const uint8_t *rows[4];
        for (int k = 0; k < nr; ++k) rows[k] = c_noise + r[k] * static_cast<ptrdiff_t>(parms.pitch);
        for (int i = 0; i < widthUV; i++)
            o[i] = chroma_spot(rows, nr, i, ssw, parms.width, false)
                       ? static_cast<T>(((p[i] + n[i] + c[i]) * mult) >> 10)
                       : c[i];
        p += ppitch;
        c += cpitch;
        n += npitch;
        o += opitch;
    }
}

template <typename T>
void mark_color_plane(const uint8_t *c_noise, T *o, ptrdiff_t opitch, int widthUV, int heightUV, int ssw, int ssh,
                      const Parms &parms) {
    // mark spots by colour (pink, green or grey depending on mark_v)
    const int mark_v = parms.mark_v;
    const T mark_color = static_cast<T>((((mark_v + 2) % 3) * mark_v / 2) << parms.shift);
    for (int j = 0; j < heightUV; j++) {
        int r[4];
        const int nr = chroma_rows(j, ssh, parms.y_next, parms.height, r);
        const uint8_t *rows[4];
        for (int k = 0; k < nr; ++k) rows[k] = c_noise + r[k] * static_cast<ptrdiff_t>(parms.pitch);
        for (int i = 0; i < widthUV; i++)
            if (chroma_spot(rows, nr, i, ssw, parms.width, true)) o[i] = mark_color;
        o += opitch;
    }
}

void print_segments(const Segments &segs, FILE *f_ptr, int fn, double frate, int w, int h, int spotmax1,
                    int spotmax2) {
    // First check whether this is a cluster of false positives
    int count = 0;
    double sum1[2] = {0, 0};
    double sum2[2] = {0, 0};
    for (const Segment *cur = segs.data; cur != segs.end; ++cur) {
        if ((cur->what & B_LINK) == 0) {
            const Segment::Data &data = cur->data;
            const double x = (data.b.x1 + data.b.x2) * 0.5;
            const double y = (data.b.y1 + data.b.y2) * 0.5;
            sum1[0] += x;
            sum1[1] += y;
            sum2[0] += x * x;
            sum2[1] += y * y;
            ++count;
        }
    }
#if DESPOT_COMPAT_363
    const double std_dev_x = std::sqrt(sum2[0] - sum1[0] * sum1[0]) / w;
    const double std_dev_y = std::sqrt(sum2[1] - sum1[1] * sum1[1]) / h;
#else
    if (count > 0)
        for (int i = 0; i < 2; ++i) {
            sum1[i] /= count;
            sum2[i] /= count;
        }
    const double std_dev_x = std::sqrt(std::max(sum2[0] - sum1[0] * sum1[0], 0.0)) / w;
    const double std_dev_y = std::sqrt(std::max(sum2[1] - sum1[1] * sum1[1], 0.0)) / h;
#endif
    const double std_dev_diag = std::sqrt((std_dev_x * std_dev_x + std_dev_y * std_dev_y) * 0.5);
    const double thr = 0.5 / std::sqrt(12.0); // standard deviation of a uniform distribution, halved
    if ((count > spotmax1 && std_dev_diag < thr) || count > spotmax2) return;

    const double tb = std::max((fn - 0.5) / frate, 0.0);
    const double te = (fn + 0.5) / frate;
    char tb_0[64];
    char te_0[64];
    print_timestamp(tb_0, sizeof(tb_0), tb);
    print_timestamp(te_0, sizeof(te_0), te);

    for (const Segment *cur = segs.data; cur != segs.end; ++cur) {
        if ((cur->what & B_LINK) == 0) {
            const Segment::Data &data = cur->data;
            if (data.s.is_noise) {
                const int margin = 2;
                const int left = std::min(std::max(data.b.x1 - margin, 0), w);
                const int right = std::min(std::max(data.b.x2 + margin, 0), w);
                const int top = std::min(std::max(data.b.y1 - margin, 0), h);
                const int bottom = std::min(std::max(data.b.y2 + margin, 0), h);
                const int cx = (data.b.x1 + data.b.x2 + 1) >> 1;
                const int cy = (data.b.y1 + data.b.y2 + 1) >> 1;
                std::fprintf(f_ptr,
                             "Dialogue: 0,%s,%s,Mask,,0,0,0,,"
                             "{\\pos(%d,%d)\\fscx100\\fscy100\\p1}m %d %d l %d %d %d %d %d %d{\\p0}\n",
                             tb_0, te_0, cx, cy, left - cx, top - cy, right - cx, top - cy, right - cx, bottom - cy,
                             left - cx, bottom - cy);
            }
        }
    }
}

#define DESPOT_INSTANTIATE(T)                                                                                     \
    template void find_outliers<T>(const T *, ptrdiff_t, const T *, ptrdiff_t, const T *, ptrdiff_t, uint8_t *,       \
                                   const Parms &);                                                                    \
    template void find_motion<T>(const T *, ptrdiff_t, const T *, ptrdiff_t, uint8_t *, const Parms &);              \
    template void add_external_mask<T>(const T *, ptrdiff_t, int, uint8_t *, const Parms &);                          \
    template void cond_median<T>(EXEC_ARGS(T));                                                                       \
    template void remove_outliers<T>(EXEC_ARGS(T));                                                                   \
    template void mark_outliers<T>(EXEC_ARGS(T));                                                                     \
    template void map_outliers<T>(EXEC_ARGS(T));                                                                      \
    template void remove_segments<T>(Segments &, const T *, ptrdiff_t, const T *, ptrdiff_t, T *, ptrdiff_t,          \
                                     uint8_t *, const Parms &);                                                       \
    template void temporal_smooth<T>(Segments &, const T *, ptrdiff_t, const T *, ptrdiff_t, T *, ptrdiff_t,          \
                                     uint8_t *, const uint8_t *, const Parms &);                                      \
    template void clean_color_plane<T>(const T *, ptrdiff_t, const T *, ptrdiff_t, const T *, ptrdiff_t, int, int,    \
                                       int, int, const uint8_t *, T *, ptrdiff_t, const Parms &);                     \
    template void mark_color_plane<T>(const uint8_t *, T *, ptrdiff_t, int, int, int, int, const Parms &);

DESPOT_INSTANTIATE(uint8_t)
DESPOT_INSTANTIATE(uint16_t)

} // namespace despot
