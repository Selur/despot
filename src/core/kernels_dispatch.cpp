// SPDX-License-Identifier: GPL-2.0-only
// CPU detection and kernel selection for asm_opt.
#include "kernels.hpp"

#if defined(DESPOT_X86)
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace despot {

namespace {

#if defined(DESPOT_X86)
bool cpuHasAvx2() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const unsigned ecx = static_cast<unsigned>(r[2]);
#else
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid(1, a, b, c, d);
    const unsigned ecx = c;
#endif
    const bool osxsave = (ecx & (1u << 27)) != 0, avx = (ecx & (1u << 28)) != 0;
    if (!osxsave || !avx) return false;
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
    __cpuidex(r, 7, 0);
    const unsigned ebx = static_cast<unsigned>(r[1]);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
    __cpuid_count(7, 0, a, b, c, d);
    const unsigned ebx = b;
#endif
    if ((xcr0 & 6) != 6) return false; // the OS must save XMM and YMM registers
    return (ebx & (1u << 5)) != 0;
}
#endif

} // namespace

bool simdAvailable(int opt) {
    switch (opt) {
    case 0:
    case 1: return true;
#if defined(DESPOT_X86)
    case 2: return true; // part of x86-64
    case 3: return cpuHasAvx2();
#endif
    default: return false;
    }
}

const Kernels &kernelsFor(int opt) {
#if defined(DESPOT_X86)
    if (opt == 0) opt = cpuHasAvx2() ? 3 : 2;
    if (opt == 3) return kernelsAVX2();
    if (opt == 2) return kernelsSSE2();
#elif defined(DESPOT_NEON)
    if (opt == 0) return kernelsNEON();
#endif
    return kernelsC();
}

} // namespace despot
