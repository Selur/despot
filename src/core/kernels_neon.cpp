// SPDX-License-Identifier: GPL-2.0-only
// NEON kernels for arm64 (16 lanes).
#include "kernels.hpp"

#if defined(DESPOT_NEON)
#include <arm_neon.h>

#include "kernels_simd.hpp"

namespace {

struct VNeon {
    using T = uint8x16_t;
    using W = int16x8_t;
    static constexpr int N = 16;
    static T zero() { return vdupq_n_u8(0); }
    static T set1(int v) { return vdupq_n_u8(static_cast<uint8_t>(v)); }
    static T load(const uint8_t *p) { return vld1q_u8(p); }
    static void store(uint8_t *p, T v) { vst1q_u8(p, v); }
    static T min(T a, T b) { return vminq_u8(a, b); }
    static T max(T a, T b) { return vmaxq_u8(a, b); }
    static T subs(T a, T b) { return vqsubq_u8(a, b); }
    static T adds(T a, T b) { return vqaddq_u8(a, b); }
    static T add(T a, T b) { return vaddq_u8(a, b); }
    static T sub(T a, T b) { return vsubq_u8(a, b); }
    static T avg(T a, T b) { return vrhaddq_u8(a, b); }
    static T and_(T a, T b) { return vandq_u8(a, b); }
    static T or_(T a, T b) { return vorrq_u8(a, b); }
    static T xor_(T a, T b) { return veorq_u8(a, b); }
    static T andnot(T a, T b) { return vbicq_u8(b, a); } // ~a & b
    static T cmpeq(T a, T b) { return vceqq_u8(a, b); }
    static T shr1(T a) { return vshrq_n_u8(a, 1); }
    static bool any(T a) { return vmaxvq_u8(a) != 0; }
    static uint32_t hsum(T a) { return vaddlvq_u8(a); }
    static W lo(T a) { return vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a))); }
    static W hi(T a) { return vreinterpretq_s16_u16(vmovl_high_u8(a)); }
    static T pack(W a, W b) { return vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)); }
    static W set16(int16_t v) { return vdupq_n_s16(v); }
    static W add16(W a, W b) { return vaddq_s16(a, b); }
    static W sub16(W a, W b) { return vsubq_s16(a, b); }
    static W mulhi16(W a, int k) {
        const uint16x8_t u = vreinterpretq_u16_s16(a);
        const uint16_t kk = static_cast<uint16_t>(k);
        const uint32x4_t l = vmull_u16(vget_low_u16(u), vdup_n_u16(kk));
        const uint32x4_t h = vmull_high_u16(u, vdupq_n_u16(kk));
        return vreinterpretq_s16_u16(vcombine_u16(vshrn_n_u32(l, 16), vshrn_n_u32(h, 16)));
    }
    static W gt16(W a, W b) { return vreinterpretq_s16_u16(vcgtq_s16(a, b)); }
    static W or16(W a, W b) { return vorrq_s16(a, b); }
    static W andnot16(W a, W b) { return vbicq_s16(b, a); }
    static W sel16(W m, W a, W b) { return vbslq_s16(vreinterpretq_u16_s16(m), a, b); }

    static void colsum(int32_t *r, const uint8_t *sub, const uint8_t *add, int w) {
        const uint8x16_t z = vdupq_n_u8(0);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const uint8x16_t a = sub ? vld1q_u8(sub + x) : z, b = add ? vld1q_u8(add + x) : z;
            const int16x8_t dlo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(b), vget_low_u8(a)));
            const int16x8_t dhi = vreinterpretq_s16_u16(vsubl_high_u8(b, a));
            const int32x4_t d[4] = {vmovl_s16(vget_low_s16(dlo)), vmovl_high_s16(dlo), vmovl_s16(vget_low_s16(dhi)),
                                    vmovl_high_s16(dhi)};
            for (int i = 0; i < 4; ++i) vst1q_s32(r + x + 4 * i, vaddq_s32(vld1q_s32(r + x + 4 * i), d[i]));
        }
        for (; x < w; ++x) r[x] += (add ? add[x] : 0) - (sub ? sub[x] : 0);
    }

    static int32x4_t scan(int32x4_t v, int32x4_t carry) {
        const int32x4_t z = vdupq_n_s32(0);
        v = vaddq_s32(v, vextq_s32(z, v, 3));
        v = vaddq_s32(v, vextq_s32(z, v, 2));
        return vaddq_s32(v, carry);
    }

    static void window(const int32_t *r, int w, int rad, const uint8_t *gate, int32_t thr, uint8_t val,
                       uint8_t *out, int32_t *) {
        int32_t total = windowInit(r, rad);
        int32x4_t carry = vdupq_n_s32(total);
        const int32x4_t vthr = vdupq_n_s32(thr);
        const uint8x16_t vval = vdupq_n_u8(val);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            uint16x4_t m[4];
            for (int i = 0; i < 4; ++i) {
                const int xi = x + 4 * i;
                const int32x4_t d = vsubq_s32(vld1q_s32(r + xi + rad), vld1q_s32(r + xi - rad - 1));
                const int32x4_t s = scan(d, carry);
                carry = vdupq_laneq_s32(s, 3);
                m[i] = vmovn_u32(vcgtq_s32(s, vthr));
            }
            uint8x16_t mask = vcombine_u8(vmovn_u16(vcombine_u16(m[0], m[1])), vmovn_u16(vcombine_u16(m[2], m[3])));
            if (gate) mask = vbicq_u8(mask, vceqq_u8(vld1q_u8(gate + x), vdupq_n_u8(0)));
            vst1q_u8(out + x, vandq_u8(mask, vval));
        }
        total = vgetq_lane_s32(carry, 0);
        windowTail(r, x, w, rad, gate, thr, val, out, total);
    }
};

} // namespace

namespace despot {
const Kernels &kernelsNEON() { return makeKernels<VNeon>(); }
} // namespace despot

#endif
