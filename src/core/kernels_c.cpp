// SPDX-License-Identifier: GPL-2.0-only
// Reference C kernels; the SIMD tables must match these bit for bit.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "kernels.hpp"

namespace despot {

namespace {

inline int outlier(int c, int mn, int mx, int sign, int p1) {
    switch (sign) {
    case 1: return mn > c ? mn - c : 0;
    case -1: return c > mx ? c - mx : 0;
    case 2: return c < mn ? mn - c : (c > mx ? std::min(c - mx, p1) : 0);
    case -2: return c > mx ? c - mx : (c < mn ? std::min(mn - c, p1) : 0);
    default: return mn > c ? mn - c : (c > mx ? c - mx : 0);
    }
}

void outliersC(const uint8_t *p, const uint8_t *n, const uint8_t *c, uint8_t *o, int w, int sign, bool ranked,
               uint8_t p1) {
    std::memset(o, 0, w);
    if (ranked) {
        for (int x = 1; x < w - 1; ++x) {
            const int mn = std::min({p[x - 1], p[x], p[x + 1], n[x - 1], n[x], n[x + 1]});
            const int mx = std::max({p[x - 1], p[x], p[x + 1], n[x - 1], n[x], n[x + 1]});
            o[x] = static_cast<uint8_t>(outlier(c[x], mn, mx, sign, p1));
        }
    } else {
        for (int x = 0; x < w; ++x)
            o[x] = static_cast<uint8_t>(outlier(c[x], std::min(p[x], n[x]), std::max(p[x], n[x]), sign, p1));
    }
}

void motionC(const uint8_t *p, const uint8_t *c, uint8_t *m, int w, uint8_t thr) {
    for (int x = 0; x < w; ++x)
        m[x] = std::abs(p[x] - c[x]) > thr ? 3 : 0;
}

void colsumC(int32_t *r, const uint8_t *sub, const uint8_t *add, int w) {
    if (sub)
        for (int x = 0; x < w; ++x) r[x] -= sub[x];
    if (add)
        for (int x = 0; x < w; ++x) r[x] += add[x];
}

void windowC(const int32_t *r, int w, int rad, const uint8_t *gate, int32_t thr, uint8_t val, uint8_t *out,
             int32_t *) {
    int32_t total = 0;
    for (int x = 0; x < rad; ++x) total += r[x];
    for (int x = 0; x < w; ++x) {
        total -= r[x - rad - 1];
        total += r[x + rad];
        out[x] = ((!gate || gate[x]) && total > thr) ? val : 0;
    }
}

void binarizeC(const uint8_t *s, uint8_t *d, int w, uint8_t val) {
    for (int x = 0; x < w; ++x) d[x] = s[x] ? val : 0;
}

void lumaSumsC(const uint8_t *c, const uint8_t *p, const uint8_t *n, int w, int64_t *sc, int64_t *spn) {
    int64_t a = 0, b = 0;
    for (int x = 0; x < w; ++x) {
        a += c[x];
        b += p[x] + n[x];
    }
    *sc = a;
    *spn = b;
}

void removeC(const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *noise, const uint8_t *motion,
             uint8_t *o, int w, int lumad) {
    for (int x = 0; x < w; ++x) {
        if (noise[x] && !motion[x])
            o[x] = static_cast<uint8_t>(std::min(std::max(((p[x] + n[x]) >> 1) + lumad, 0), 255));
        else
            o[x] = c[x];
    }
}

void markC(const uint8_t *c, const uint8_t *noise, const uint8_t *motion, uint8_t *o, int w, uint8_t mark_v) {
    for (int x = 0; x < w; ++x) {
        if (noise[x] && !motion[x])
            o[x] = mark_v;
        else if (!noise[x] && motion[x])
            o[x] = static_cast<uint8_t>(126 + c[x] / 2);
        else if (c[x] == mark_v)
            o[x] = c[x] >= 1 ? c[x] - 1 : c[x] + 1;
        else
            o[x] = c[x];
    }
}

void mapC(const uint8_t *noise, const uint8_t *motion, uint8_t *o, int w) {
    for (int x = 0; x < w; ++x) {
        if (noise && noise[x])
            o[x] = motion[x] ? 159 : 255;
        else
            o[x] = motion[x] ? 95 : 0;
    }
}

void mergeC(const uint8_t *a, const uint8_t *b, uint8_t *o, int w) {
    for (int x = 0; x < w; ++x) o[x] = (a[x] || b[x]) ? 255 : 0;
}

void noiseToOneC(const uint8_t *pn, const uint8_t *cn, uint8_t *m, int w) {
    for (int x = 0; x < w; ++x)
        if (m[x] && (pn[x] || cn[x])) m[x] = 1;
}

void tsmoothC(const uint8_t *p, const uint8_t *n, uint8_t *o, const uint8_t *noise, const uint8_t *motion, int w,
              int limit) {
    for (int x = 0; x < w; ++x) {
        if (!noise[x] && !motion[x]) {
            const int m = (p[x] + o[x] + n[x]) / 3;
            int corr = m - o[x];
            if (corr > limit * 2) corr = 0;
            if (corr < -(limit * 2)) corr = 0;
            if (corr > limit) corr = limit * 2 - corr;
            if (corr < -limit) corr = -(limit * 2) - corr;
            o[x] = static_cast<uint8_t>(o[x] + corr);
        }
    }
}

void medianC(const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *motion, uint8_t *o, int w) {
    for (int x = 0; x < w; ++x) {
        if (motion[x])
            o[x] = c[x];
        else
            o[x] = std::min(std::max(p[x], n[x]), std::max(std::min(p[x], n[x]), c[x]));
    }
}

int64_t countNonzeroC(const uint8_t *s, int w) {
    int64_t k = 0;
    for (int x = 0; x < w; ++x) k += s[x] != 0;
    return k;
}

void extMaskC(const uint8_t *ext, uint8_t *m, int w) {
    for (int x = 0; x < w; ++x)
        if (ext[x] > 127) m[x] = 255;
}

int nextAboveC(const uint8_t *l, int x, int w, uint8_t thr) {
    while (x < w && l[x] <= thr) ++x;
    return x;
}

} // namespace

const Kernels &kernelsC() {
    static const Kernels k = {outliersC, motionC,  colsumC,     windowC,  binarizeC,     lumaSumsC,
                              removeC,   markC,    mapC,        mergeC,   noiseToOneC,   tsmoothC,
                              medianC,   countNonzeroC, extMaskC, nextAboveC};
    return k;
}

} // namespace despot
