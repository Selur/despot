// SPDX-License-Identifier: GPL-2.0-only
// SIMD row kernels over a vector traits type V (unsigned 8 bit lanes, plus 16 bit helpers).
// Included by one source file per instruction set, inside its own anonymous namespace; every
// kernel produces exactly the result of the C kernel (kernels_c.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "kernels.hpp"
#include "kernels16_ref.hpp"

namespace {

using despot::Kernels;

inline int outlierS(int c, int mn, int mx, int sign, int p1) {
    switch (sign) {
    case 1: return mn > c ? mn - c : 0;
    case -1: return c > mx ? c - mx : 0;
    case 2: return c < mn ? mn - c : (c > mx ? std::min(c - mx, p1) : 0);
    case -2: return c > mx ? c - mx : (c < mn ? std::min(mn - c, p1) : 0);
    default: return mn > c ? mn - c : (c > mx ? c - mx : 0);
    }
}

template <class V>
inline typename V::T nzV(typename V::T a) { // 0xFF where a != 0
    return V::andnot(V::cmpeq(a, V::zero()), V::set1(0xFF));
}

template <class V>
inline typename V::T selV(typename V::T m, typename V::T a, typename V::T b) { // m ? a : b
    return V::or_(V::and_(m, a), V::andnot(m, b));
}

template <class V>
inline typename V::T outlierV(typename V::T c, typename V::T mn, typename V::T mx, int sign, typename V::T vp1) {
    const typename V::T below = V::subs(mn, c); // mn - c where c < mn
    const typename V::T above = V::subs(c, mx); // c - mx where c > mx
    switch (sign) {
    case 1: return below;
    case -1: return above;
    case 2: return V::or_(below, V::min(above, vp1));
    case -2: return V::or_(above, V::min(below, vp1));
    default: return V::or_(below, above); // at most one of them is non-zero
    }
}

template <class V>
void outliersT(const uint8_t *p, const uint8_t *n, const uint8_t *c, uint8_t *o, int w, int sign, bool ranked,
               uint8_t p1) {
    using T = typename V::T;
    const T vp1 = V::set1(p1);
    if (ranked) {
        if (w <= 0) return;
        o[0] = 0;
        o[w - 1] = 0;
        int x = 1;
        for (; x + V::N <= w - 1; x += V::N) {
            const T pl = V::load(p + x - 1), pc = V::load(p + x), pr = V::load(p + x + 1);
            const T nl = V::load(n + x - 1), nc = V::load(n + x), nr = V::load(n + x + 1);
            const T mn = V::min(V::min(V::min(pl, nl), V::min(pc, nc)), V::min(pr, nr));
            const T mx = V::max(V::max(V::max(pl, nl), V::max(pc, nc)), V::max(pr, nr));
            V::store(o + x, outlierV<V>(V::load(c + x), mn, mx, sign, vp1));
        }
        for (; x < w - 1; ++x) {
            const int mn = std::min({p[x - 1], p[x], p[x + 1], n[x - 1], n[x], n[x + 1]});
            const int mx = std::max({p[x - 1], p[x], p[x + 1], n[x - 1], n[x], n[x + 1]});
            o[x] = static_cast<uint8_t>(outlierS(c[x], mn, mx, sign, p1));
        }
    } else {
        int x = 0;
        for (; x + V::N <= w; x += V::N) {
            const T pv = V::load(p + x), nv = V::load(n + x);
            V::store(o + x, outlierV<V>(V::load(c + x), V::min(pv, nv), V::max(pv, nv), sign, vp1));
        }
        for (; x < w; ++x)
            o[x] = static_cast<uint8_t>(outlierS(c[x], std::min(p[x], n[x]), std::max(p[x], n[x]), sign, p1));
    }
}

template <class V>
void motionT(const uint8_t *p, const uint8_t *c, uint8_t *m, int w, uint8_t thr) {
    using T = typename V::T;
    const T vthr = V::set1(thr), three = V::set1(3);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T a = V::load(p + x), b = V::load(c + x);
        const T d = V::or_(V::subs(a, b), V::subs(b, a));
        V::store(m + x, V::and_(nzV<V>(V::subs(d, vthr)), three));
    }
    for (; x < w; ++x) m[x] = std::abs(p[x] - c[x]) > thr ? 3 : 0;
}

template <class V>
void binarizeT(const uint8_t *s, uint8_t *d, int w, uint8_t val) {
    using T = typename V::T;
    const T vv = V::set1(val);
    int x = 0;
    for (; x + V::N <= w; x += V::N) V::store(d + x, V::and_(nzV<V>(V::load(s + x)), vv));
    for (; x < w; ++x) d[x] = s[x] ? val : 0;
}

template <class V>
void lumaSumsT(const uint8_t *c, const uint8_t *p, const uint8_t *n, int w, int64_t *sc, int64_t *spn) {
    int64_t a = 0, b = 0;
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        a += V::hsum(V::load(c + x));
        b += V::hsum(V::load(p + x));
        b += V::hsum(V::load(n + x));
    }
    for (; x < w; ++x) {
        a += c[x];
        b += p[x] + n[x];
    }
    *sc = a;
    *spn = b;
}

template <class V>
void removeT(const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *noise, const uint8_t *motion,
             uint8_t *o, int w, int lumad) {
    using T = typename V::T;
    const T one = V::set1(1);
    const T vadd = V::set1(static_cast<uint8_t>(std::min(std::max(lumad, 0), 255)));
    const T vsub = V::set1(static_cast<uint8_t>(std::min(std::max(-lumad, 0), 255)));
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T a = V::load(p + x), b = V::load(n + x);
        const T avg = V::sub(V::avg(a, b), V::and_(V::xor_(a, b), one)); // (a + b) >> 1
        const T v = V::subs(V::adds(avg, vadd), vsub);
        const T m = V::andnot(V::cmpeq(V::load(noise + x), V::zero()), V::cmpeq(V::load(motion + x), V::zero()));
        V::store(o + x, selV<V>(m, v, V::load(c + x)));
    }
    for (; x < w; ++x) {
        if (noise[x] && !motion[x])
            o[x] = static_cast<uint8_t>(std::min(std::max(((p[x] + n[x]) >> 1) + lumad, 0), 255));
        else
            o[x] = c[x];
    }
}

template <class V>
void markT(const uint8_t *c, const uint8_t *noise, const uint8_t *motion, uint8_t *o, int w, uint8_t mark_v) {
    using T = typename V::T;
    const T mv = V::set1(mark_v);
    const T v3 = V::set1(mark_v >= 1 ? mark_v - 1 : 1);
    const T base = V::set1(126);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T cv = V::load(c + x);
        const T zn = V::cmpeq(V::load(noise + x), V::zero());
        const T zm = V::cmpeq(V::load(motion + x), V::zero());
        const T case1 = V::andnot(zn, zm); // noise, no motion
        const T case2 = V::andnot(zm, zn); // motion, no noise
        const T case3 = V::cmpeq(cv, mv);
        T r = selV<V>(case3, v3, cv);
        r = selV<V>(case2, V::add(base, V::shr1(cv)), r);
        r = selV<V>(case1, mv, r);
        V::store(o + x, r);
    }
    for (; x < w; ++x) {
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

template <class V>
void mapT(const uint8_t *noise, const uint8_t *motion, uint8_t *o, int w) {
    using T = typename V::T;
    const T v255 = V::set1(255), v159 = V::set1(159), v95 = V::set1(95);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T nm = nzV<V>(V::load(motion + x));
        const T nn = noise ? nzV<V>(V::load(noise + x)) : V::zero();
        V::store(o + x, selV<V>(nn, selV<V>(nm, v159, v255), V::and_(nm, v95)));
    }
    for (; x < w; ++x) {
        if (noise && noise[x])
            o[x] = motion[x] ? 159 : 255;
        else
            o[x] = motion[x] ? 95 : 0;
    }
}

template <class V>
void mergeT(const uint8_t *a, const uint8_t *b, uint8_t *o, int w) {
    int x = 0;
    for (; x + V::N <= w; x += V::N) V::store(o + x, nzV<V>(V::or_(V::load(a + x), V::load(b + x))));
    for (; x < w; ++x) o[x] = (a[x] || b[x]) ? 255 : 0;
}

template <class V>
void noiseToOneT(const uint8_t *pn, const uint8_t *cn, uint8_t *m, int w) {
    using T = typename V::T;
    const T one = V::set1(1);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T mv = V::load(m + x);
        const T sel = V::and_(nzV<V>(mv), nzV<V>(V::or_(V::load(pn + x), V::load(cn + x))));
        V::store(m + x, selV<V>(sel, one, mv));
    }
    for (; x < w; ++x)
        if (m[x] && (pn[x] || cn[x])) m[x] = 1;
}

template <class V>
typename V::W tsmoothHalf(typename V::W p, typename V::W o, typename V::W n, int limit) {
    using W = typename V::W;
    const W L = V::set16(static_cast<int16_t>(limit)), L2 = V::set16(static_cast<int16_t>(2 * limit));
    const W nL = V::set16(static_cast<int16_t>(-limit)), nL2 = V::set16(static_cast<int16_t>(-2 * limit));
    const W m = V::mulhi16(V::add16(V::add16(p, o), n), 21846); // (p + o + n) / 3 for sums up to 765
    W corr = V::sub16(m, o);
    corr = V::andnot16(V::or16(V::gt16(corr, L2), V::gt16(nL2, corr)), corr);
    const W a = V::gt16(corr, L), b = V::gt16(nL, corr);
    corr = V::sel16(a, V::sub16(L2, corr), corr);
    corr = V::sel16(b, V::sub16(nL2, corr), corr);
    return V::add16(o, corr);
}

template <class V>
void tsmoothT(const uint8_t *p, const uint8_t *n, uint8_t *o, const uint8_t *noise, const uint8_t *motion, int w,
              int limit) {
    using T = typename V::T;
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T pv = V::load(p + x), nv = V::load(n + x), ov = V::load(o + x);
        const T keep = V::and_(V::cmpeq(V::load(noise + x), V::zero()), V::cmpeq(V::load(motion + x), V::zero()));
        if (!V::any(keep)) continue;
        const auto lo = tsmoothHalf<V>(V::lo(pv), V::lo(ov), V::lo(nv), limit);
        const auto hi = tsmoothHalf<V>(V::hi(pv), V::hi(ov), V::hi(nv), limit);
        V::store(o + x, selV<V>(keep, V::pack(lo, hi), ov));
    }
    for (; x < w; ++x) {
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

template <class V>
void medianT(const uint8_t *p, const uint8_t *n, const uint8_t *c, const uint8_t *motion, uint8_t *o, int w) {
    using T = typename V::T;
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T pv = V::load(p + x), nv = V::load(n + x), cv = V::load(c + x);
        const T med = V::min(V::max(pv, nv), V::max(V::min(pv, nv), cv));
        V::store(o + x, selV<V>(nzV<V>(V::load(motion + x)), cv, med));
    }
    for (; x < w; ++x)
        o[x] = motion[x] ? c[x] : std::min(std::max(p[x], n[x]), std::max(std::min(p[x], n[x]), c[x]));
}

template <class V>
int64_t countNonzeroT(const uint8_t *s, int w) {
    using T = typename V::T;
    const T one = V::set1(1);
    int64_t k = 0;
    int x = 0;
    for (; x + V::N <= w; x += V::N) k += V::hsum(V::and_(nzV<V>(V::load(s + x)), one));
    for (; x < w; ++x) k += s[x] != 0;
    return k;
}

template <class V>
void extMaskT(const uint8_t *ext, uint8_t *m, int w) {
    using T = typename V::T;
    const T hb = V::set1(0x80);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const T e = V::cmpeq(V::and_(V::load(ext + x), hb), hb); // ext > 127
        V::store(m + x, V::or_(V::load(m + x), e));
    }
    for (; x < w; ++x)
        if (ext[x] > 127) m[x] = 255;
}

template <class V>
int nextAboveT(const uint8_t *l, int x, int w, uint8_t thr) {
    using T = typename V::T;
    const T vt = V::set1(thr);
    for (; x + V::N <= w; x += V::N)
        if (V::any(V::subs(V::load(l + x), vt))) break;
    while (x < w && l[x] <= thr) ++x;
    return x;
}

// ---- 9..16 bit samples: V::U holds V::N / 2 unsigned 16 bit lanes ----

template <class V>
inline typename V::U nz16(typename V::U a) { // 0xFFFF where a != 0
    return V::andnot16u(V::cmpeq16u(a, V::zero16()), V::set16u(0xFFFF));
}

template <class V>
inline typename V::U sel16V(typename V::U m, typename V::U a, typename V::U b) {
    return V::or16u(V::and16u(m, a), V::andnot16u(m, b));
}

template <class V>
inline typename V::U outlier16V(typename V::U c, typename V::U mn, typename V::U mx, int sign, typename V::U vp1,
                                int shift) {
    const typename V::U v255 = V::set16u(255);
    const typename V::U below = V::min16u(V::srl16(V::subs16u(mn, c), shift), v255);
    const typename V::U above = V::min16u(V::srl16(V::subs16u(c, mx), shift), v255);
    switch (sign) {
    case 1: return below;
    case -1: return above;
    case 2: return V::or16u(below, V::min16u(above, vp1));
    case -2: return V::or16u(above, V::min16u(below, vp1));
    default: return V::or16u(below, above); // at most one of them is non-zero
    }
}

template <class V>
void outliers16T(const uint16_t *p, const uint16_t *n, const uint16_t *c, uint8_t *o, int w, int sign, bool ranked,
                 uint8_t p1, int shift) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    const U vp1 = V::set16u(p1);
    if (ranked) {
        if (w <= 0) return;
        o[0] = 0;
        o[w - 1] = 0;
        int x = 1;
        for (; x + V::N <= w - 1; x += V::N) {
            U r[2];
            for (int h = 0; h < 2; ++h) {
                const int xi = x + h * H;
                const U pl = V::load16(p + xi - 1), pc = V::load16(p + xi), pr = V::load16(p + xi + 1);
                const U nl = V::load16(n + xi - 1), nc = V::load16(n + xi), nr = V::load16(n + xi + 1);
                const U mn = V::min16u(V::min16u(V::min16u(pl, nl), V::min16u(pc, nc)), V::min16u(pr, nr));
                const U mx = V::max16u(V::max16u(V::max16u(pl, nl), V::max16u(pc, nc)), V::max16u(pr, nr));
                r[h] = outlier16V<V>(V::load16(c + xi), mn, mx, sign, vp1, shift);
            }
            V::store(o + x, V::narrow(r[0], r[1]));
        }
        despot::ref16::outliers(p, n, c, o, x, w - 1, sign, true, p1, shift);
    } else {
        int x = 0;
        for (; x + V::N <= w; x += V::N) {
            U r[2];
            for (int h = 0; h < 2; ++h) {
                const int xi = x + h * H;
                const U pv = V::load16(p + xi), nv = V::load16(n + xi);
                r[h] = outlier16V<V>(V::load16(c + xi), V::min16u(pv, nv), V::max16u(pv, nv), sign, vp1, shift);
            }
            V::store(o + x, V::narrow(r[0], r[1]));
        }
        despot::ref16::outliers(p, n, c, o, x, w, sign, false, p1, shift);
    }
}

template <class V>
void motion16T(const uint16_t *p, const uint16_t *c, uint8_t *m, int w, uint16_t thr) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    const U vthr = V::set16u(thr), three = V::set16u(3);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        U r[2];
        for (int h = 0; h < 2; ++h) {
            const U a = V::load16(p + x + h * H), b = V::load16(c + x + h * H);
            const U d = V::or16u(V::subs16u(a, b), V::subs16u(b, a));
            r[h] = V::and16u(nz16<V>(V::subs16u(d, vthr)), three);
        }
        V::store(m + x, V::narrow(r[0], r[1]));
    }
    despot::ref16::motion(p, c, m, x, w, thr);
}

template <class V>
void remove16T(const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *noise, const uint8_t *motion,
               uint16_t *o, int w, int lumad, uint16_t maxval) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    const U one = V::set16u(1), vmax = V::set16u(maxval);
    const U vadd = V::set16u(std::min(std::max(lumad, 0), 65535));
    const U vsub = V::set16u(std::min(std::max(-lumad, 0), 65535));
    int x = 0;
    for (; x + H <= w; x += H) {
        const U a = V::load16(p + x), b = V::load16(n + x);
        const U avg = V::sub16u(V::avg16u(a, b), V::and16u(V::xor16u(a, b), one)); // (a + b) >> 1
        const U v = V::min16u(V::subs16u(V::adds16u(avg, vadd), vsub), vmax);
        const U m = V::andnot16u(V::nz8to16(motion + x), V::nz8to16(noise + x));
        V::store16(o + x, sel16V<V>(m, v, V::load16(c + x)));
    }
    despot::ref16::remove(p, n, c, noise, motion, o, x, w, lumad, maxval);
}

template <class V>
void mark16T(const uint16_t *c, const uint8_t *noise, const uint8_t *motion, uint16_t *o, int w, uint16_t mark_v,
             int shift) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    const U mv = V::set16u(mark_v);
    const U v3 = V::set16u(mark_v >= 1 ? mark_v - 1 : 1);
    const U base = V::set16u(126 << shift);
    int x = 0;
    for (; x + H <= w; x += H) {
        const U cv = V::load16(c + x);
        const U nn = V::nz8to16(noise + x), nm = V::nz8to16(motion + x);
        const U case1 = V::andnot16u(nm, nn); // noise, no motion
        const U case2 = V::andnot16u(nn, nm); // motion, no noise
        const U case3 = V::cmpeq16u(cv, mv);
        U r = sel16V<V>(case3, v3, cv);
        r = sel16V<V>(case2, V::add16u(base, V::srl16(cv, 1)), r);
        r = sel16V<V>(case1, mv, r);
        V::store16(o + x, r);
    }
    despot::ref16::mark(c, noise, motion, o, x, w, mark_v, shift);
}

template <class V>
void map16T(const uint8_t *noise, const uint8_t *motion, uint16_t *o, int w, int shift) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    const U v255 = V::set16u(255 << shift), v159 = V::set16u(159 << shift), v95 = V::set16u(95 << shift);
    int x = 0;
    for (; x + H <= w; x += H) {
        const U nm = V::nz8to16(motion + x);
        const U nn = noise ? V::nz8to16(noise + x) : V::zero16();
        V::store16(o + x, sel16V<V>(nn, sel16V<V>(nm, v159, v255), V::and16u(nm, v95)));
    }
    despot::ref16::map(noise, motion, o, x, w, shift);
}

template <class V>
void median16T(const uint16_t *p, const uint16_t *n, const uint16_t *c, const uint8_t *motion, uint16_t *o, int w) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    int x = 0;
    for (; x + H <= w; x += H) {
        const U pv = V::load16(p + x), nv = V::load16(n + x), cv = V::load16(c + x);
        const U med = V::min16u(V::max16u(pv, nv), V::max16u(V::min16u(pv, nv), cv));
        V::store16(o + x, sel16V<V>(V::nz8to16(motion + x), cv, med));
    }
    despot::ref16::median(p, n, c, motion, o, x, w);
}

template <class V>
void extMask16T(const uint16_t *ext, uint8_t *m, int w, int shift) {
    using U = typename V::U;
    constexpr int H = V::N / 2;
    const U thr = V::set16u((128 << shift) - 1), v255 = V::set16u(255);
    int x = 0;
    for (; x + V::N <= w; x += V::N) {
        const U e0 = V::and16u(nz16<V>(V::subs16u(V::load16(ext + x), thr)), v255);
        const U e1 = V::and16u(nz16<V>(V::subs16u(V::load16(ext + x + H), thr)), v255);
        V::store(m + x, V::or_(V::load(m + x), V::narrow(e0, e1)));
    }
    despot::ref16::ext_mask(ext, m, x, w, shift);
}

inline void lumaSums16S(const uint16_t *c, const uint16_t *p, const uint16_t *n, int w, int64_t *sc, int64_t *spn) {
    int64_t a = 0, b = 0;
    despot::ref16::luma_sums(c, p, n, 0, w, a, b);
    *sc = a;
    *spn = b;
}

inline void tsmooth16S(const uint16_t *p, const uint16_t *n, uint16_t *o, const uint8_t *noise, const uint8_t *motion,
                       int w, int limit) {
    despot::ref16::tsmooth(p, n, o, noise, motion, 0, w, limit);
}

template <class V>
const Kernels &makeKernels() {
    static const Kernels k = {outliersT<V>, motionT<V>,    V::colsum,     V::window,     binarizeT<V>,
                              lumaSumsT<V>, removeT<V>,    markT<V>,      mapT<V>,       mergeT<V>,
                              noiseToOneT<V>, tsmoothT<V>, medianT<V>,    countNonzeroT<V>, extMaskT<V>,
                              nextAboveT<V>,  outliers16T<V>, motion16T<V>, lumaSums16S, remove16T<V>,
                              mark16T<V>,     map16T<V>,   tsmooth16S,    median16T<V>, extMask16T<V>};
    return k;
}

// Scalar parts shared by the window kernels: running sum exactly as the C kernel.
inline int32_t windowInit(const int32_t *r, int rad) {
    int32_t total = 0;
    for (int x = 0; x < rad; ++x) total += r[x];
    return total;
}

inline void windowTail(const int32_t *r, int x, int w, int rad, const uint8_t *gate, int32_t thr, uint8_t val,
                       uint8_t *out, int32_t total) {
    for (; x < w; ++x) {
        total -= r[x - rad - 1];
        total += r[x + rad];
        out[x] = ((!gate || gate[x]) && total > thr) ? val : 0;
    }
}

} // namespace
