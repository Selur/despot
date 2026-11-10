// SPDX-License-Identifier: GPL-2.0-only
// Scalar 9..16 bit kernels over a column range [x0, x1). Used by the C table for whole rows and by
// the SIMD tables for the columns after the last full vector, so both are identical by construction.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace despot {
namespace ref16 {

inline int outlier(int c, int mn, int mx, int sign, int p1, int shift) {
    const int below = mn > c ? std::min((mn - c) >> shift, 255) : 0;
    const int above = c > mx ? std::min((c - mx) >> shift, 255) : 0;
    switch (sign) {
    case 1: return below;
    case -1: return above;
    case 2: return below ? below : std::min(above, p1);
    case -2: return above ? above : std::min(below, p1);
    default: return below ? below : above;
    }
}

// ranked: x0 >= 1 and x1 <= w - 1 (the caller writes o[0] = o[w-1] = 0)
inline void outliers(const uint16_t *p, const uint16_t *n, const uint16_t *c, uint8_t *o, int x0, int x1, int sign,
                     bool ranked, uint8_t p1, int shift) {
    for (int x = x0; x < x1; ++x) {
        int mn, mx;
        if (ranked) {
            mn = std::min({p[x - 1], p[x], p[x + 1], n[x - 1], n[x], n[x + 1]});
            mx = std::max({p[x - 1], p[x], p[x + 1], n[x - 1], n[x], n[x + 1]});
        } else {
            mn = std::min(p[x], n[x]);
            mx = std::max(p[x], n[x]);
        }
        o[x] = static_cast<uint8_t>(outlier(c[x], mn, mx, sign, p1, shift));
    }
}

inline void motion(const uint16_t *p, const uint16_t *c, uint8_t *m, int x0, int x1, uint16_t thr) {
    for (int x = x0; x < x1; ++x) m[x] = std::abs(p[x] - c[x]) > thr ? 3 : 0;
}

inline void remove(const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *noise,
                   const uint8_t *motion, uint16_t *o, int x0, int x1, int lumad, int maxval) {
    for (int x = x0; x < x1; ++x) {
        if (noise[x] && !motion[x])
            o[x] = static_cast<uint16_t>(std::min(std::max(((p[x] + n[x]) >> 1) + lumad, 0), maxval));
        else
            o[x] = c[x];
    }
}

inline void mark(const uint16_t *c, const uint8_t *noise, const uint8_t *motion, uint16_t *o, int x0, int x1,
                 uint16_t mark_v, int shift) {
    for (int x = x0; x < x1; ++x) {
        if (noise[x] && !motion[x])
            o[x] = mark_v;
        else if (!noise[x] && motion[x])
            o[x] = static_cast<uint16_t>((126 << shift) + c[x] / 2);
        else if (c[x] == mark_v)
            o[x] = static_cast<uint16_t>(c[x] >= 1 ? c[x] - 1 : c[x] + 1);
        else
            o[x] = c[x];
    }
}

inline void map(const uint8_t *noise, const uint8_t *motion, uint16_t *o, int x0, int x1, int shift) {
    for (int x = x0; x < x1; ++x) {
        int v;
        if (noise && noise[x])
            v = motion[x] ? 159 : 255;
        else
            v = motion[x] ? 95 : 0;
        o[x] = static_cast<uint16_t>(v << shift);
    }
}

inline void tsmooth(const uint16_t *p, const uint16_t *n, uint16_t *o, const uint8_t *noise, const uint8_t *motion,
                    int x0, int x1, int limit) {
    for (int x = x0; x < x1; ++x) {
        if (!noise[x] && !motion[x]) {
            const int m = (p[x] + o[x] + n[x]) / 3;
            int corr = m - o[x];
            if (corr > limit * 2) corr = 0;
            if (corr < -(limit * 2)) corr = 0;
            if (corr > limit) corr = limit * 2 - corr;
            if (corr < -limit) corr = -(limit * 2) - corr;
            o[x] = static_cast<uint16_t>(o[x] + corr);
        }
    }
}

inline void median(const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *motion, uint16_t *o,
                   int x0, int x1) {
    for (int x = x0; x < x1; ++x)
        o[x] = motion[x] ? c[x] : std::min(std::max(p[x], n[x]), std::max(std::min(p[x], n[x]), c[x]));
}

inline void ext_mask(const uint16_t *ext, uint8_t *m, int x0, int x1, int shift) {
    for (int x = x0; x < x1; ++x)
        if ((ext[x] >> shift) > 127) m[x] = 255;
}

inline void luma_sums(const uint16_t *c, const uint16_t *p, const uint16_t *n, int x0, int x1, int64_t &a,
                      int64_t &b) {
    for (int x = x0; x < x1; ++x) {
        a += c[x];
        b += p[x] + n[x];
    }
}

} // namespace ref16
} // namespace despot
