// SPDX-License-Identifier: GPL-2.0-only
// AVX2 kernels (32 lanes). Only this file is compiled with AVX2 enabled; the dispatcher selects
// it at runtime.
#include "kernels.hpp"

#if defined(DESPOT_X86)
#include <immintrin.h>

#include "kernels_simd.hpp"

namespace {

struct VAvx2 {
    using T = __m256i;
    using W = __m256i;
    static constexpr int N = 32;
    static T zero() { return _mm256_setzero_si256(); }
    static T set1(int v) { return _mm256_set1_epi8(static_cast<char>(v)); }
    static T load(const uint8_t *p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)); }
    static void store(uint8_t *p, T v) { _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v); }
    static T min(T a, T b) { return _mm256_min_epu8(a, b); }
    static T max(T a, T b) { return _mm256_max_epu8(a, b); }
    static T subs(T a, T b) { return _mm256_subs_epu8(a, b); }
    static T adds(T a, T b) { return _mm256_adds_epu8(a, b); }
    static T add(T a, T b) { return _mm256_add_epi8(a, b); }
    static T sub(T a, T b) { return _mm256_sub_epi8(a, b); }
    static T avg(T a, T b) { return _mm256_avg_epu8(a, b); }
    static T and_(T a, T b) { return _mm256_and_si256(a, b); }
    static T or_(T a, T b) { return _mm256_or_si256(a, b); }
    static T xor_(T a, T b) { return _mm256_xor_si256(a, b); }
    static T andnot(T a, T b) { return _mm256_andnot_si256(a, b); }
    static T cmpeq(T a, T b) { return _mm256_cmpeq_epi8(a, b); }
    static T shr1(T a) { return _mm256_and_si256(_mm256_srli_epi16(a, 1), _mm256_set1_epi8(0x7F)); }
    static bool any(T a) { return _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, _mm256_setzero_si256())) != -1; }
    static uint32_t hsum(T a) {
        const __m256i s = _mm256_sad_epu8(a, _mm256_setzero_si256());
        const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        return static_cast<uint32_t>(_mm_cvtsi128_si32(t) + _mm_cvtsi128_si32(_mm_srli_si128(t, 8)));
    }
    // lo/hi/pack work per 128 bit lane; pack(lo(a), hi(a)) restores the byte order of a
    static W lo(T a) { return _mm256_unpacklo_epi8(a, _mm256_setzero_si256()); }
    static W hi(T a) { return _mm256_unpackhi_epi8(a, _mm256_setzero_si256()); }
    static T pack(W a, W b) { return _mm256_packus_epi16(a, b); }
    static W set16(int16_t v) { return _mm256_set1_epi16(v); }
    static W add16(W a, W b) { return _mm256_add_epi16(a, b); }
    static W sub16(W a, W b) { return _mm256_sub_epi16(a, b); }
    static W mulhi16(W a, int k) { return _mm256_mulhi_epu16(a, _mm256_set1_epi16(static_cast<short>(k))); }
    static W gt16(W a, W b) { return _mm256_cmpgt_epi16(a, b); }
    static W or16(W a, W b) { return _mm256_or_si256(a, b); }
    static W andnot16(W a, W b) { return _mm256_andnot_si256(a, b); }
    static W sel16(W m, W a, W b) { return _mm256_blendv_epi8(b, a, m); }

    using U = __m256i;
    static U zero16() { return _mm256_setzero_si256(); }
    static U set16u(int v) { return _mm256_set1_epi16(static_cast<short>(v)); }
    static U load16(const uint16_t *p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)); }
    static void store16(uint16_t *p, U v) { _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), v); }
    static U min16u(U a, U b) { return _mm256_min_epu16(a, b); }
    static U max16u(U a, U b) { return _mm256_max_epu16(a, b); }
    static U subs16u(U a, U b) { return _mm256_subs_epu16(a, b); }
    static U adds16u(U a, U b) { return _mm256_adds_epu16(a, b); }
    static U add16u(U a, U b) { return _mm256_add_epi16(a, b); }
    static U sub16u(U a, U b) { return _mm256_sub_epi16(a, b); }
    static U avg16u(U a, U b) { return _mm256_avg_epu16(a, b); }
    static U and16u(U a, U b) { return _mm256_and_si256(a, b); }
    static U or16u(U a, U b) { return _mm256_or_si256(a, b); }
    static U xor16u(U a, U b) { return _mm256_xor_si256(a, b); }
    static U andnot16u(U a, U b) { return _mm256_andnot_si256(a, b); }
    static U cmpeq16u(U a, U b) { return _mm256_cmpeq_epi16(a, b); }
    static U srl16(U a, int k) { return _mm256_srl_epi16(a, _mm_cvtsi32_si128(k)); }
    static U nz8to16(const uint8_t *p) { // 0xFFFF where p[i] != 0, 16 bytes
        const __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
        return _mm256_andnot_si256(_mm256_cmpeq_epi16(b, _mm256_setzero_si256()), _mm256_set1_epi16(-1));
    }
    static T narrow(U a, U b) { // lanes <= 255; packus works per 128 bit lane
        return _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
    }

    static __m256i u8to32(const uint8_t *p) {
        return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(p)));
    }

    static void colsum(int32_t *r, const uint8_t *sub, const uint8_t *add, int w) {
        int x = 0;
        for (; x + 32 <= w; x += 32) {
            for (int i = 0; i < 4; ++i) {
                const int xi = x + 8 * i;
                __m256i *pr = reinterpret_cast<__m256i *>(r + xi);
                __m256i v = _mm256_loadu_si256(pr);
                if (add) v = _mm256_add_epi32(v, u8to32(add + xi));
                if (sub) v = _mm256_sub_epi32(v, u8to32(sub + xi));
                _mm256_storeu_si256(pr, v);
            }
        }
        for (; x < w; ++x) r[x] += (add ? add[x] : 0) - (sub ? sub[x] : 0);
    }

    // inclusive prefix sum of 8 lanes plus carry
    static __m256i scan(__m256i v, __m256i carry) {
        v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
        v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));
        // add the last element of the low lane to the high lane
        const __m256i t = _mm256_permute2x128_si256(v, v, 0x08); // [0, low lane]
        v = _mm256_add_epi32(v, _mm256_shuffle_epi32(t, 0xFF));
        return _mm256_add_epi32(v, carry);
    }

    static void window(const int32_t *r, int w, int rad, const uint8_t *gate, int32_t thr, uint8_t val,
                       uint8_t *out, int32_t *) {
        int32_t total = windowInit(r, rad);
        __m256i carry = _mm256_set1_epi32(total);
        const __m256i vthr = _mm256_set1_epi32(thr);
        const __m256i vval = set1(val);
        const __m256i last = _mm256_set1_epi32(7);
        const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        int x = 0;
        for (; x + 32 <= w; x += 32) {
            __m256i m[4];
            for (int i = 0; i < 4; ++i) {
                const int xi = x + 8 * i;
                const __m256i d =
                    _mm256_sub_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(r + xi + rad)),
                                     _mm256_loadu_si256(reinterpret_cast<const __m256i *>(r + xi - rad - 1)));
                const __m256i s = scan(d, carry);
                carry = _mm256_permutevar8x32_epi32(s, last);
                m[i] = _mm256_cmpgt_epi32(s, vthr);
            }
            __m256i mask = _mm256_packs_epi16(_mm256_packs_epi32(m[0], m[1]), _mm256_packs_epi32(m[2], m[3]));
            mask = _mm256_permutevar8x32_epi32(mask, order);
            if (gate) mask = andnot(cmpeq(load(gate + x), zero()), mask);
            store(out + x, _mm256_and_si256(mask, vval));
        }
        total = _mm256_cvtsi256_si32(carry);
        windowTail(r, x, w, rad, gate, thr, val, out, total);
    }
};

} // namespace

namespace despot {
const Kernels &kernelsAVX2() { return makeKernels<VAvx2>(); }
} // namespace despot

#endif
