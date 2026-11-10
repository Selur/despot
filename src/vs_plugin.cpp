// SPDX-License-Identifier: GPL-2.0-only
// VapourSynth (API 4) interface of DeSpot: despot.DeSpot(clip, ...), same parameters as for AviSynth+.
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "VSHelper4.h"
#include "VapourSynth4.h"
#include "core/despot.hpp"
#include "version.hpp"

using namespace despot;

namespace {

struct DeSpotData {
    VSNode *node = nullptr;
    VSNode *extmask = nullptr;
    VSVideoInfo vi{};
    int ext_frames = 0;
    std::unique_ptr<Engine> engine;
};

const VSFrame *VS_CC getFrame(int n, int activationReason, void *instanceData, void **, VSFrameContext *ctx,
                              VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<DeSpotData *>(instanceData);
    const int nf = d->vi.numFrames;
    const bool proc = d->engine->processes(n);
    const int first = proc ? std::max(0, d->engine->firstNeeded(n)) : n;
    const int last = proc ? std::min(nf - 1, d->engine->lastNeeded(n)) : n;

    if (activationReason == arInitial) {
        for (int k = first; k <= last; ++k) vsapi->requestFrameFilter(k, d->node, ctx);
        if (proc && d->extmask) vsapi->requestFrameFilter(std::min(n, d->ext_frames - 1), d->extmask, ctx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady) return nullptr;

    if (!proc) return vsapi->getFrameFilter(n, d->node, ctx);

    const VSFrame *frames[5] = {};
    FrameR fr[5];
    const FrameR *src[5] = {};
    const int planes = d->vi.format.numPlanes;
    for (int i = 0; i < 5; ++i) {
        const int k = n - 2 + i;
        if (k < first || k > last) continue;
        frames[i] = vsapi->getFrameFilter(k, d->node, ctx);
        for (int pl = 0; pl < planes; ++pl) {
            fr[i].plane[pl].ptr = vsapi->getReadPtr(frames[i], pl);
            fr[i].plane[pl].stride = vsapi->getStride(frames[i], pl);
        }
        src[i] = &fr[i];
    }
    const VSFrame *ext = nullptr;
    PlaneR extp;
    if (d->extmask) {
        ext = vsapi->getFrameFilter(std::min(n, d->ext_frames - 1), d->extmask, ctx);
        extp.ptr = vsapi->getReadPtr(ext, 0);
        extp.stride = vsapi->getStride(ext, 0);
    }
    VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, frames[2], core);
    FrameW out;
    for (int pl = 0; pl < planes; ++pl) {
        out.plane[pl].ptr = vsapi->getWritePtr(dst, pl);
        out.plane[pl].stride = vsapi->getStride(dst, pl);
    }
    std::string error;
    try {
        d->engine->process(n, src, d->extmask ? &extp : nullptr, out);
    } catch (const std::exception &e) {
        error = e.what();
    }
    for (const VSFrame *f : frames) vsapi->freeFrame(f);
    vsapi->freeFrame(ext);
    if (!error.empty()) {
        vsapi->freeFrame(dst);
        vsapi->setFilterError(("DeSpot: " + error).c_str(), ctx);
        return nullptr;
    }
    return dst;
}

void VS_CC freeFilter(void *instanceData, VSCore *, const VSAPI *vsapi) {
    auto *d = static_cast<DeSpotData *>(instanceData);
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->extmask);
    delete d;
}

void VS_CC create(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) {
    auto d = std::make_unique<DeSpotData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = *vsapi->getVideoInfo(d->node);
    auto fail = [&](const std::string &msg) {
        vsapi->mapSetError(out, ("DeSpot: " + msg).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->extmask);
    };

    const VSVideoFormat &f = d->vi.format;
    if (!vsh::isConstantVideoFormat(&d->vi) || f.sampleType != stInteger || f.bitsPerSample > 16 ||
        (f.colorFamily != cfYUV && f.colorFamily != cfGray))
        return fail("clip must be 8..16 bit integer YUV or Gray with constant format and size");

    Params p;
    int err;
    auto get_int = [&](const char *name, int &v) {
        const int64_t x = vsapi->mapGetInt(in, name, 0, &err);
        if (!err) v = static_cast<int>(std::max<int64_t>(INT32_MIN, std::min<int64_t>(INT32_MAX, x)));
    };
    auto get_bool = [&](const char *name, bool &v) {
        const int64_t x = vsapi->mapGetInt(in, name, 0, &err);
        if (!err) v = x != 0;
    };
    get_int("mthres", p.mthres);
    get_int("mwidth", p.mwidth);
    get_int("mheight", p.mheight);
    get_int("merode", p.merode);
    get_bool("interlaced", p.interlaced);
    get_bool("median", p.median);
    get_int("p1", p.p1);
    get_int("p2", p.p2);
    get_int("pwidth", p.pwidth);
    get_int("pheight", p.pheight);
    get_bool("ranked", p.ranked);
    get_int("sign", p.sign);
    get_int("maxpts", p.maxpts);
    get_int("p1percent", p.p1percent);
    get_int("dilate", p.dilate);
    get_bool("fitluma", p.fitluma);
    get_int("blur", p.blur);
    get_int("tsmooth", p.tsmooth);
    get_int("show", p.show);
    get_int("mark_v", p.mark_v);
    get_bool("show_chroma", p.show_chroma);
    get_bool("motpn", p.motpn);
    get_int("seg", p.seg);
    get_bool("color", p.color);
    get_int("mscene", p.mscene);
    get_int("minpts", p.minpts);
    const char *outfile = vsapi->mapGetData(in, "outfile", 0, &err);
    if (!err && outfile) p.outfile = outfile;
    get_bool("mc", p.mc);
    get_int("spotmax1", p.spotmax1);
    get_int("spotmax2", p.spotmax2);
    get_int("asm_opt", p.asm_opt);

    d->extmask = vsapi->mapGetNode(in, "extmask", 0, &err);
    if (d->extmask) {
        const VSVideoInfo *evi = vsapi->getVideoInfo(d->extmask);
        if (evi->width != d->vi.width || evi->height != d->vi.height || evi->format.sampleType != stInteger ||
            evi->format.bitsPerSample > 16)
            return fail("extmask must be an 8..16 bit integer clip of the same size");
        d->ext_frames = evi->numFrames;
    }

    Format fmt;
    fmt.width = d->vi.width;
    fmt.height = d->vi.height;
    fmt.gray = f.colorFamily == cfGray;
    fmt.ssw = f.subSamplingW;
    fmt.ssh = f.subSamplingH;
    fmt.bits = f.bitsPerSample;
    fmt.ext_bits = d->extmask ? vsapi->getVideoInfo(d->extmask)->format.bitsPerSample : 8;
    try {
        d->engine = std::make_unique<Engine>(p, fmt, d->vi.numFrames, d->vi.fpsNum > 0 ? d->vi.fpsNum : 25,
                                             d->vi.fpsNum > 0 ? d->vi.fpsDen : 1);
    } catch (const std::exception &e) {
        return fail(e.what());
    }

    // the outfile is written as frames are produced: one frame at a time keeps it tidy
    const VSFilterMode mode = d->engine->usesOutfile() ? fmUnordered : fmParallel;
    VSFilterDependency deps[2] = {{d->node, rpGeneral}, {d->extmask, rpGeneral}};
    const VSVideoInfo vi = d->vi;
    const int num_deps = d->extmask ? 2 : 1;
    vsapi->createVideoFilter(out, "DeSpot", &vi, getFrame, freeFilter, mode, deps, num_deps, d.release(), core);
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("io.github.selur.despot", "despot", "DeSpot " DESPOT_VERSION,
                         VS_MAKE_VERSION(DESPOT_VERSION_MAJOR, DESPOT_VERSION_MINOR), VAPOURSYNTH_API_VERSION, 0,
                         plugin);
    vspapi->registerFunction("DeSpot",
                             "clip:vnode;mthres:int:opt;mwidth:int:opt;mheight:int:opt;merode:int:opt;"
                             "interlaced:int:opt;median:int:opt;p1:int:opt;p2:int:opt;pwidth:int:opt;"
                             "pheight:int:opt;ranked:int:opt;sign:int:opt;maxpts:int:opt;p1percent:int:opt;"
                             "dilate:int:opt;fitluma:int:opt;blur:int:opt;tsmooth:int:opt;show:int:opt;"
                             "mark_v:int:opt;show_chroma:int:opt;motpn:int:opt;seg:int:opt;color:int:opt;"
                             "mscene:int:opt;minpts:int:opt;extmask:vnode:opt;outfile:data:opt;mc:int:opt;"
                             "spotmax1:int:opt;spotmax2:int:opt;asm_opt:int:opt;",
                             "clip:vnode;", create, nullptr, plugin);
}
