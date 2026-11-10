// SPDX-License-Identifier: GPL-2.0-only
// SSE2 kernels (16 lanes).
#include "kernels.hpp"

#if defined(DESPOT_X86)
#include <emmintrin.h>

#include "kernels_simd.hpp"

namespace {

struct VSse2 {
    using T = __m128i;
    using W = __m128i;
    static constexpr int N = 16;
    static T zero() { return _mm_setzero_si128(); }
    static T set1(int v) { return _mm_set1_epi8(static_cast<char>(v)); }
    static T load(const uint8_t *p) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(p)); }
    static void store(uint8_t *p, T v) { _mm_storeu_si128(reinterpret_cast<__m128i *>(p), v); }
    static T min(T a, T b) { return _mm_min_epu8(a, b); }
    static T max(T a, T b) { return _mm_max_epu8(a, b); }
    static T subs(T a, T b) { return _mm_subs_epu8(a, b); }
    static T adds(T a, T b) { return _mm_adds_epu8(a, b); }
    static T add(T a, T b) { return _mm_add_epi8(a, b); }
    static T sub(T a, T b) { return _mm_sub_epi8(a, b); }
    static T avg(T a, T b) { return _mm_avg_epu8(a, b); }
    static T and_(T a, T b) { return _mm_and_si128(a, b); }
    static T or_(T a, T b) { return _mm_or_si128(a, b); }
    static T xor_(T a, T b) { return _mm_xor_si128(a, b); }
    static T andnot(T a, T b) { return _mm_andnot_si128(a, b); }
    static T cmpeq(T a, T b) { return _mm_cmpeq_epi8(a, b); }
    static T shr1(T a) { return _mm_and_si128(_mm_srli_epi16(a, 1), _mm_set1_epi8(0x7F)); }
    static bool any(T a) { return _mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128())) != 0xFFFF; }
    static uint32_t hsum(T a) {
        const __m128i s = _mm_sad_epu8(a, _mm_setzero_si128());
        return static_cast<uint32_t>(_mm_cvtsi128_si32(s) + _mm_cvtsi128_si32(_mm_srli_si128(s, 8)));
    }
    static W lo(T a) { return _mm_unpacklo_epi8(a, _mm_setzero_si128()); }
    static W hi(T a) { return _mm_unpackhi_epi8(a, _mm_setzero_si128()); }
    static T pack(W a, W b) { return _mm_packus_epi16(a, b); }
    static W set16(int16_t v) { return _mm_set1_epi16(v); }
    static W add16(W a, W b) { return _mm_add_epi16(a, b); }
    static W sub16(W a, W b) { return _mm_sub_epi16(a, b); }
    static W mulhi16(W a, int k) { return _mm_mulhi_epu16(a, _mm_set1_epi16(static_cast<short>(k))); }
    static W gt16(W a, W b) { return _mm_cmpgt_epi16(a, b); }
    static W or16(W a, W b) { return _mm_or_si128(a, b); }
    static W andnot16(W a, W b) { return _mm_andnot_si128(a, b); }
    static W sel16(W m, W a, W b) { return _mm_or_si128(_mm_and_si128(m, a), _mm_andnot_si128(m, b)); }

    static void colsum(int32_t *r, const uint8_t *sub, const uint8_t *add, int w) {
        const __m128i z = _mm_setzero_si128();
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            __m128i a = sub ? load(sub + x) : z, b = add ? load(add + x) : z;
            // r += b - a, as four 32 bit vectors
            const __m128i alo = _mm_unpacklo_epi8(a, z), ahi = _mm_unpackhi_epi8(a, z);
            const __m128i blo = _mm_unpacklo_epi8(b, z), bhi = _mm_unpackhi_epi8(b, z);
            const __m128i dlo = _mm_sub_epi16(blo, alo), dhi = _mm_sub_epi16(bhi, ahi); // -255..255
            const __m128i d[4] = {_mm_srai_epi32(_mm_unpacklo_epi16(dlo, dlo), 16),
                                  _mm_srai_epi32(_mm_unpackhi_epi16(dlo, dlo), 16),
                                  _mm_srai_epi32(_mm_unpacklo_epi16(dhi, dhi), 16),
                                  _mm_srai_epi32(_mm_unpackhi_epi16(dhi, dhi), 16)};
            for (int i = 0; i < 4; ++i) {
                __m128i *pr = reinterpret_cast<__m128i *>(r + x + 4 * i);
                _mm_storeu_si128(pr, _mm_add_epi32(_mm_loadu_si128(pr), d[i]));
            }
        }
        for (; x < w; ++x) r[x] += (add ? add[x] : 0) - (sub ? sub[x] : 0);
    }

    // inclusive prefix sum of 4 lanes plus carry
    static __m128i scan(__m128i v, __m128i carry) {
        v = _mm_add_epi32(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi32(v, _mm_slli_si128(v, 8));
        return _mm_add_epi32(v, carry);
    }

    static void window(const int32_t *r, int w, int rad, const uint8_t *gate, int32_t thr, uint8_t val,
                       uint8_t *out, int32_t *) {
        int32_t total = windowInit(r, rad);
        __m128i carry = _mm_set1_epi32(total);
        const __m128i vthr = _mm_set1_epi32(thr);
        const __m128i vval = set1(val);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            __m128i m[4];
            for (int i = 0; i < 4; ++i) {
                const int xi = x + 4 * i;
                const __m128i d = _mm_sub_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(r + xi + rad)),
                                                _mm_loadu_si128(reinterpret_cast<const __m128i *>(r + xi - rad - 1)));
                const __m128i s = scan(d, carry);
                carry = _mm_shuffle_epi32(s, 0xFF);
                m[i] = _mm_cmpgt_epi32(s, vthr);
            }
            __m128i mask = _mm_packs_epi16(_mm_packs_epi32(m[0], m[1]), _mm_packs_epi32(m[2], m[3]));
            if (gate) mask = andnot(cmpeq(load(gate + x), zero()), mask);
            store(out + x, _mm_and_si128(mask, vval));
        }
        total = _mm_cvtsi128_si32(carry);
        windowTail(r, x, w, rad, gate, thr, val, out, total);
    }
};

} // namespace

namespace despot {
const Kernels &kernelsSSE2() { return makeKernels<VSse2>(); }
} // namespace despot

#endif
