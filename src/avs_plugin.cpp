// SPDX-License-Identifier: GPL-2.0-only
// AviSynth+ interface of DeSpot (name and arguments of DeSpot 3.6.3, plus asm_opt).
#include <memory>
#include <stdexcept>
#include <string>

#include "avisynth.h"
#include "core/despot.hpp"
#include "version.hpp"

using namespace despot;

namespace {

const int kPlanes[3] = {PLANAR_Y, PLANAR_U, PLANAR_V};

class DeSpotAvs : public GenericVideoFilter {
public:
    DeSpotAvs(PClip child, PClip extmask, const Params &p, IScriptEnvironment *env)
        : GenericVideoFilter(child), extmask_(extmask) {
        try {
            env->CheckVersion(8);
            has_v8_ = true;
        } catch (const AvisynthError &) {
            has_v8_ = false;
        }
        if (!vi.IsPlanar() || !(vi.IsYUV() || vi.IsYUVA()) || vi.BitsPerComponent() > 16)
            env->ThrowError("DeSpot: input must be 8..16 bit planar YUV(A) or Y");
        Format f;
        f.width = vi.width;
        f.height = vi.height;
        f.gray = vi.IsY();
        f.ssw = f.gray ? 0 : vi.GetPlaneWidthSubsampling(PLANAR_U);
        f.ssh = f.gray ? 0 : vi.GetPlaneHeightSubsampling(PLANAR_U);
        f.bits = vi.BitsPerComponent();
        if (extmask_) {
            const VideoInfo &evi = extmask_->GetVideoInfo();
            if (evi.width != vi.width || evi.height != vi.height || !evi.IsPlanar() || evi.BitsPerComponent() > 16)
                env->ThrowError("DeSpot: extmask must be an 8..16 bit planar clip of the same size");
            ext_frames_ = evi.num_frames;
            f.ext_bits = evi.BitsPerComponent();
        }
        try {
            engine_ = std::make_unique<Engine>(p, f, vi.num_frames, vi.fps_numerator, vi.fps_denominator);
        } catch (const std::exception &e) {
            env->ThrowError("DeSpot: %s", e.what());
        }
    }

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override {
        n = n < 0 ? 0 : (n >= vi.num_frames ? vi.num_frames - 1 : n);
        if (!engine_->processes(n)) return child->GetFrame(n, env); // no temporal neighbours
        PVideoFrame frames[5];
        FrameR fr[5];
        const FrameR *src[5] = {};
        for (int i = 0; i < 5; ++i) {
            const int k = n - 2 + i;
            if (k < 0 || k >= vi.num_frames || k < engine_->firstNeeded(n) || k > engine_->lastNeeded(n)) continue;
            frames[i] = child->GetFrame(k, env);
            for (int pl = 0; pl < (vi.IsY() ? 1 : 3); ++pl) {
                fr[i].plane[pl].ptr = frames[i]->GetReadPtr(kPlanes[pl]);
                fr[i].plane[pl].stride = frames[i]->GetPitch(kPlanes[pl]);
            }
            src[i] = &fr[i];
        }
        PVideoFrame ext;
        PlaneR extp;
        if (extmask_) {
            ext = extmask_->GetFrame(std::min(n, ext_frames_ - 1), env);
            extp.ptr = ext->GetReadPtr(PLANAR_Y);
            extp.stride = ext->GetPitch(PLANAR_Y);
        }
        PVideoFrame dst = has_v8_ ? env->NewVideoFrameP(vi, &frames[2]) : env->NewVideoFrame(vi);
        FrameW out;
        for (int pl = 0; pl < (vi.IsY() ? 1 : 3); ++pl) {
            out.plane[pl].ptr = dst->GetWritePtr(kPlanes[pl]);
            out.plane[pl].stride = dst->GetPitch(kPlanes[pl]);
        }
        try {
            engine_->process(n, src, extmask_ ? &extp : nullptr, out);
        } catch (const std::exception &e) {
            env->ThrowError("DeSpot: %s", e.what());
        }
        if (vi.IsYUVA()) // alpha is passed through
            env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), frames[2]->GetReadPtr(PLANAR_A),
                        frames[2]->GetPitch(PLANAR_A), frames[2]->GetRowSize(PLANAR_A), frames[2]->GetHeight(PLANAR_A));
        return dst;
    }

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
        (void)frame_range;
        // no state between frames; the engine writes the outfile in frame order itself
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        return 0;
    }

private:
    PClip extmask_;
    int ext_frames_ = 0;
    bool has_v8_ = false;
    std::unique_ptr<Engine> engine_;
};

AVSValue __cdecl Create(AVSValue args, void *, IScriptEnvironment *env) {
    Params p;
    int i = 1;
    auto opt_int = [&](int &v) {
        if (args[i].Defined()) v = args[i].AsInt();
        ++i;
    };
    auto opt_bool = [&](bool &v) {
        if (args[i].Defined()) v = args[i].AsBool();
        ++i;
    };
    opt_int(p.mthres);
    opt_int(p.mwidth);
    opt_int(p.mheight);
    opt_int(p.merode);
    opt_bool(p.interlaced);
    opt_bool(p.median);
    opt_int(p.p1);
    opt_int(p.p2);
    opt_int(p.pwidth);
    opt_int(p.pheight);
    opt_bool(p.ranked);
    opt_int(p.sign);
    opt_int(p.maxpts);
    opt_int(p.p1percent);
    opt_int(p.dilate);
    opt_bool(p.fitluma);
    opt_int(p.blur);
    opt_int(p.tsmooth);
    opt_int(p.show);
    opt_int(p.mark_v);
    opt_bool(p.show_chroma);
    opt_bool(p.motpn);
    opt_int(p.seg);
    opt_bool(p.color);
    opt_int(p.mscene);
    opt_int(p.minpts);
    PClip extmask = args[i].Defined() ? args[i].AsClip() : nullptr;
    ++i;
    if (args[i].AsBool(false)) env->ThrowError("DeSpot: planar YUY2 is not supported, use YV16");
    ++i;
    if (args[i].Defined()) p.outfile = args[i].AsString();
    ++i;
    opt_bool(p.mc);
    opt_int(p.spotmax1);
    opt_int(p.spotmax2);
    opt_int(p.asm_opt);
    const std::string err = validate(p);
    if (!err.empty()) env->ThrowError("DeSpot: %s", err.c_str());
    return new DeSpotAvs(args[0].AsClip(), extmask, p, env);
}

} // namespace

#ifdef _WIN32
#define DESPOT_AVS_EXPORT extern "C" __declspec(dllexport)
#else
#define DESPOT_AVS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

const AVS_Linkage *AVS_linkage = nullptr;

DESPOT_AVS_EXPORT const char *__stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("DeSpot",
                     "c[mthres]i[mwidth]i[mheight]i[merode]i[interlaced]b[median]b"
                     "[p1]i[p2]i[pwidth]i[pheight]i[ranked]b[sign]i[maxpts]i[p1percent]i[dilate]i[fitluma]b"
                     "[blur]i[tsmooth]i[show]i[mark_v]i[show_chroma]b[motpn]b[seg]i[color]b[mscene]i[minpts]i"
                     "[extmask]c[planar]b[outfile]s[mc]b[spotmax1]i[spotmax2]i[asm_opt]i",
                     Create, nullptr);
    return "DeSpot " DESPOT_VERSION;
}
