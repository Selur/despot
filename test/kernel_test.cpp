// SPDX-License-Identifier: GPL-2.0-only
// Every SIMD kernel table against the C kernels on random data; exit code 1 on any mismatch.
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.hpp"

using namespace despot;

namespace {

std::mt19937 rng(12345);
int failures = 0;

int rnd(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

// Random bytes, biased towards few distinct values so that equalities and zeros occur.
std::vector<uint8_t> bytes(int n, int mode) {
    std::vector<uint8_t> v(n + 64);
    for (auto &b : v) {
        switch (mode) {
        case 0: b = static_cast<uint8_t>(rnd(0, 255)); break;
        case 1: b = rnd(0, 3) ? 0 : static_cast<uint8_t>(rnd(1, 255)); break; // sparse map
        case 2: b = static_cast<uint8_t>(rnd(0, 1) * 255); break;
        default: b = static_cast<uint8_t>(rnd(120, 136)); break;
        }
    }
    return v;
}

void check(bool ok, const char *isa, const char *what, int w, const std::string &extra = {}) {
    if (!ok) {
        ++failures;
        if (failures < 50) std::printf("MISMATCH %s %s w=%d %s\n", isa, what, w, extra.c_str());
    }
}

// Random 9..16 bit samples: full range, near the maximum, or a narrow range (equal values).
std::vector<uint16_t> samples(int n, int bits, int mode) {
    const int maxv = (1 << bits) - 1;
    std::vector<uint16_t> v(n + 64);
    const int mid = rnd(0, maxv);
    for (auto &s : v) {
        switch (mode) {
        case 0: s = static_cast<uint16_t>(rnd(0, maxv)); break;
        case 1: s = static_cast<uint16_t>(rnd(std::max(0, maxv - 300), maxv)); break;
        default: s = static_cast<uint16_t>(std::min(maxv, std::max(0, mid + rnd(-600, 600)))); break;
        }
    }
    return v;
}

void test16(const Kernels &k, const char *isa) {
    const Kernels &c = kernelsC();
    for (int iter = 0; iter < 3000; ++iter) {
        const int bits = rnd(9, 16), shift = bits - 8, maxv = (1 << bits) - 1;
        const int w = iter < 200 ? iter % 100 + 1 : rnd(1, 700);
        const int mode = rnd(0, 2);
        auto p = samples(w, bits, mode), n = samples(w, bits, mode), cc = samples(w, bits, mode);
        auto noise = bytes(w, rnd(1, 2)), motion = bytes(w, rnd(1, 2));
        std::vector<uint8_t> o1(w + 64, 7), o2(w + 64, 7);
        std::vector<uint16_t> q1(w + 64, 7), q2(w + 64, 7);
        const std::string tag = " bits=" + std::to_string(bits);

        for (int sign = -2; sign <= 2; ++sign)
            for (int ranked = 0; ranked < 2; ++ranked) {
                const uint8_t p1 = static_cast<uint8_t>(rnd(0, 255));
                c.outliers16(p.data(), n.data(), cc.data(), o1.data(), w, sign, ranked != 0, p1, shift);
                k.outliers16(p.data(), n.data(), cc.data(), o2.data(), w, sign, ranked != 0, p1, shift);
                check(o1 == o2, isa, "outliers16", w, tag + " sign=" + std::to_string(sign));
            }

        const uint16_t thr = static_cast<uint16_t>(rnd(0, 255) << shift);
        c.motion16(p.data(), cc.data(), o1.data(), w, thr);
        k.motion16(p.data(), cc.data(), o2.data(), w, thr);
        check(o1 == o2, isa, "motion16", w, tag);

        int64_t a1, b1, a2, b2;
        c.luma_sums16(cc.data(), p.data(), n.data(), w, &a1, &b1);
        k.luma_sums16(cc.data(), p.data(), n.data(), w, &a2, &b2);
        check(a1 == a2 && b1 == b2, isa, "luma_sums16", w, tag);

        const int lumad = rnd(-maxv - 10, maxv + 10);
        c.remove16(p.data(), n.data(), cc.data(), noise.data(), motion.data(), q1.data(), w, lumad,
                   static_cast<uint16_t>(maxv));
        k.remove16(p.data(), n.data(), cc.data(), noise.data(), motion.data(), q2.data(), w, lumad,
                   static_cast<uint16_t>(maxv));
        check(q1 == q2, isa, "remove16", w, tag + " lumad=" + std::to_string(lumad));

        const uint16_t mv = static_cast<uint16_t>(rnd(0, 3) ? rnd(0, 255) << shift : 0);
        if (rnd(0, 1)) cc[rnd(0, w - 1)] = mv; // make c == mark_v occur
        c.mark16(cc.data(), noise.data(), motion.data(), q1.data(), w, mv, shift);
        k.mark16(cc.data(), noise.data(), motion.data(), q2.data(), w, mv, shift);
        check(q1 == q2, isa, "mark16", w, tag);

        c.map16(noise.data(), motion.data(), q1.data(), w, shift);
        k.map16(noise.data(), motion.data(), q2.data(), w, shift);
        check(q1 == q2, isa, "map16", w, tag);
        c.map16(nullptr, motion.data(), q1.data(), w, shift);
        k.map16(nullptr, motion.data(), q2.data(), w, shift);
        check(q1 == q2, isa, "map16(null)", w, tag);

        auto t1 = cc, t2 = cc;
        const int limit = rnd(1, 255) << shift;
        c.tsmooth16(p.data(), n.data(), t1.data(), noise.data(), motion.data(), w, limit);
        k.tsmooth16(p.data(), n.data(), t2.data(), noise.data(), motion.data(), w, limit);
        check(t1 == t2, isa, "tsmooth16", w, tag);

        c.median16(p.data(), n.data(), cc.data(), motion.data(), q1.data(), w);
        k.median16(p.data(), n.data(), cc.data(), motion.data(), q2.data(), w);
        check(q1 == q2, isa, "median16", w, tag);

        auto e1 = motion, e2 = motion;
        c.ext_mask16(p.data(), e1.data(), w, shift);
        k.ext_mask16(p.data(), e2.data(), w, shift);
        check(e1 == e2, isa, "ext_mask16", w, tag);
    }
}

void testTable(const Kernels &k, const char *isa) {
    const Kernels &c = kernelsC();
    for (int iter = 0; iter < 3000; ++iter) {
        const int w = iter < 200 ? iter % 100 + 1 : rnd(1, 700);
        const int mode = rnd(0, 3);
        auto p = bytes(w, mode), n = bytes(w, mode), cc = bytes(w, mode);
        auto noise = bytes(w, rnd(1, 2)), motion = bytes(w, rnd(1, 2));
        std::vector<uint8_t> o1(w + 64, 7), o2(w + 64, 7);

        for (int sign = -2; sign <= 2; ++sign)
            for (int ranked = 0; ranked < 2; ++ranked) {
                const uint8_t p1 = static_cast<uint8_t>(rnd(0, 255));
                c.outliers(p.data(), n.data(), cc.data(), o1.data(), w, sign, ranked != 0, p1);
                k.outliers(p.data(), n.data(), cc.data(), o2.data(), w, sign, ranked != 0, p1);
                check(o1 == o2, isa, "outliers", w, "sign=" + std::to_string(sign) + " ranked=" + std::to_string(ranked));
            }

        const uint8_t thr = static_cast<uint8_t>(rnd(0, 255));
        c.motion(p.data(), cc.data(), o1.data(), w, thr);
        k.motion(p.data(), cc.data(), o2.data(), w, thr);
        check(o1 == o2, isa, "motion", w);

        c.binarize(noise.data(), o1.data(), w, 1);
        k.binarize(noise.data(), o2.data(), w, 1);
        check(o1 == o2, isa, "binarize", w);

        int64_t a1, b1, a2, b2;
        c.luma_sums(cc.data(), p.data(), n.data(), w, &a1, &b1);
        k.luma_sums(cc.data(), p.data(), n.data(), w, &a2, &b2);
        check(a1 == a2 && b1 == b2, isa, "luma_sums", w);

        const int lumad = rnd(-260, 260);
        c.remove(p.data(), n.data(), cc.data(), noise.data(), motion.data(), o1.data(), w, lumad);
        k.remove(p.data(), n.data(), cc.data(), noise.data(), motion.data(), o2.data(), w, lumad);
        check(o1 == o2, isa, "remove", w, "lumad=" + std::to_string(lumad));

        const uint8_t mv = static_cast<uint8_t>(rnd(0, 3) ? rnd(0, 255) : 0);
        c.mark(cc.data(), noise.data(), motion.data(), o1.data(), w, mv);
        k.mark(cc.data(), noise.data(), motion.data(), o2.data(), w, mv);
        check(o1 == o2, isa, "mark", w);

        c.map(noise.data(), motion.data(), o1.data(), w);
        k.map(noise.data(), motion.data(), o2.data(), w);
        check(o1 == o2, isa, "map", w);
        c.map(nullptr, motion.data(), o1.data(), w);
        k.map(nullptr, motion.data(), o2.data(), w);
        check(o1 == o2, isa, "map(null)", w);

        c.merge(noise.data(), motion.data(), o1.data(), w);
        k.merge(noise.data(), motion.data(), o2.data(), w);
        check(o1 == o2, isa, "merge", w);

        auto m1 = bytes(w, rnd(0, 1));
        for (auto &b : m1) b = b ? (rnd(0, 1) ? 3 : 1) : 0;
        auto m2 = m1;
        c.noise_to_one(noise.data(), p.data(), m1.data(), w);
        k.noise_to_one(noise.data(), p.data(), m2.data(), w);
        check(m1 == m2, isa, "noise_to_one", w);

        const int limit = rnd(1, 255);
        auto t1 = cc, t2 = cc;
        c.tsmooth(p.data(), n.data(), t1.data(), noise.data(), motion.data(), w, limit);
        k.tsmooth(p.data(), n.data(), t2.data(), noise.data(), motion.data(), w, limit);
        check(t1 == t2, isa, "tsmooth", w, "limit=" + std::to_string(limit));

        c.median(p.data(), n.data(), cc.data(), motion.data(), o1.data(), w);
        k.median(p.data(), n.data(), cc.data(), motion.data(), o2.data(), w);
        check(o1 == o2, isa, "median", w);

        check(c.count_nonzero(noise.data(), w) == k.count_nonzero(noise.data(), w), isa, "count_nonzero", w);

        auto e1 = motion, e2 = motion;
        c.ext_mask(p.data(), e1.data(), w);
        k.ext_mask(p.data(), e2.data(), w);
        check(e1 == e2, isa, "ext_mask", w);

        for (int t = 0; t < 4; ++t) {
            const uint8_t th = static_cast<uint8_t>(rnd(0, 255));
            const int x0 = rnd(0, w);
            check(c.next_above(noise.data(), x0, w, th) == k.next_above(noise.data(), x0, w, th), isa, "next_above",
                  w);
        }

        // column sums and sliding window
        const int rad = rnd(0, 3) ? rnd(0, 6) : rnd(0, 80);
        const int pad = rad + 1;
        std::vector<int32_t> r1(w + 2 * pad + 8, 0);
        for (int x = 0; x < w; ++x) r1[pad + x] = rnd(0, 3) ? 0 : rnd(0, 400);
        if (rnd(0, 3) == 0) // non-zero padding (interlaced compat layout)
            for (auto &v : r1) v += rnd(0, 2);
        auto r2 = r1;
        const bool sub = rnd(0, 1), add = rnd(0, 1);
        c.colsum(r1.data() + pad, sub ? p.data() : nullptr, add ? n.data() : nullptr, w);
        k.colsum(r2.data() + pad, sub ? p.data() : nullptr, add ? n.data() : nullptr, w);
        check(r1 == r2, isa, "colsum", w);
        std::vector<int32_t> tmp(w + 2 * rad + 2);
        const bool gate = rnd(0, 1);
        const int32_t wthr = rnd(0, 2) ? 0 : rnd(-5, 600);
        const uint8_t val = rnd(0, 1) ? 255 : 1;
        c.window(r1.data() + pad, w, rad, gate ? noise.data() : nullptr, wthr, val, o1.data(), tmp.data());
        k.window(r1.data() + pad, w, rad, gate ? noise.data() : nullptr, wthr, val, o2.data(), tmp.data());
        check(o1 == o2, isa, "window", w, "rad=" + std::to_string(rad));
    }
    test16(k, isa);
}

} // namespace

int main() {
    int tested = 0;
#if defined(DESPOT_X86)
    testTable(kernelsSSE2(), "SSE2");
    ++tested;
    if (simdAvailable(3)) {
        testTable(kernelsAVX2(), "AVX2");
        ++tested;
    } else {
        std::printf("AVX2 not available, skipped\n");
    }
#elif defined(DESPOT_NEON)
    testTable(kernelsNEON(), "NEON");
    ++tested;
#endif
    if (failures) {
        std::printf("%d mismatches\n", failures);
        return 1;
    }
    std::printf("kernel_test: %d instruction set(s) identical to C\n", tested);
    return 0;
}
