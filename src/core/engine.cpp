// SPDX-License-Identifier: GPL-2.0-only
// DeSpot engine: one output frame from the source frames n-2..n+2. Every map a frame needs is
// recomputed for that frame, which gives exactly the result of the original sequential filter
// (it cached the maps of neighbour frames) for any request order and any number of threads.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "despot.hpp"

namespace despot {

namespace {

constexpr char ass_header[] = "[Script Info]\n"
                              "Title: DeSpot automatically generated file\n"
                              "ScriptType: v4.00+\n"
                              "WrapStyle: 0\n"
                              "PlayResX: %d\n"
                              "PlayResY: %d\n"
                              "ScaledBorderAndShadow: yes\n"
                              "Video Aspect Ratio: 0\n"
                              "Video Zoom: 8\n"
                              "Video Position: 0\n"
                              "\n"
                              "[V4+ Styles]\n"
                              "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                              "Style: Mask,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"
                              "Style: Outline,Arial,20,&HFFFFFFFF,&H000000FF,&H00FF00FF,&H00000000,0,0,0,0,100,100,0,0,1,%d,0,7,0,0,0,1\n"
                              "\n"
                              "[Events]\n"
                              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                              "Comment: 0,0:00:00.00,0:00:00.00,Mask,,0,0,0,,--- Templates for manual editing ---\n"
                              "Dialogue: 0,0:00:00.00,0:00:00.00,Mask,,0,0,0,,{\\pos(100,100)\\fscx100\\fscy100\\p1}m -4 -4 l 4 -4 4 4 -4 4{\\p0}\n"
                              "Dialogue: 0,0:00:00.00,0:00:00.00,Mask,,0,0,0,,{\\pos(100,100)\\fscx100\\fscy100\\p1}m -6 -6 l 6 -6 6 6 -6 6{\\p0}\n"
                              "Dialogue: 0,0:00:00.00,0:00:00.00,Mask,,0,0,0,,{\\pos(100,100)\\fscx100\\fscy100\\p1}m -8 -8 l 8 -8 8 8 -8 8{\\p0}\n"
                              "Comment: 0,0:00:00.00,0:00:00.00,Mask,,0,0,0,,--- Masks ---\n";

std::string range(const char *name, int v, int lo, int hi) {
    if (v >= lo && v <= hi) return {};
    return std::string(name) + " must be from " + std::to_string(lo) + " to " + std::to_string(hi);
}

// w in bytes
void copy_plane(const PlaneR &s, const PlaneW &d, int w, int h) {
    for (int y = 0; y < h; ++y) std::memcpy(d.ptr + y * d.stride, s.ptr + y * s.stride, w);
}

// Copies count segments plus the two sentinels; links (nowis) are moved into the destination array.
void copy_segments(const Segment *src, size_t count, Segment *dst) {
    std::memcpy(static_cast<void *>(dst), src, (count + 2) * sizeof(Segment));
    for (size_t i = 0; i < count; ++i)
        if (dst[i].data.nowis) dst[i].data.nowis = dst + (dst[i].data.nowis - src);
}

} // namespace

std::string validate(const Params &p) {
    std::string e;
    auto chk = [&](const char *name, int v, int lo, int hi) {
        if (e.empty()) e = range(name, v, lo, hi);
    };
    chk("mthres", p.mthres, 0, 255);
    chk("mwidth", p.mwidth, 0, 32000);
    chk("mheight", p.mheight, 0, 32000);
    chk("merode", p.merode, 0, 100);
    chk("p1", p.p1, 0, 255);
    chk("p2", p.p2, 0, 255);
    chk("pwidth", p.pwidth, 0, 32000);
    chk("pheight", p.pheight, 0, 32000);
    chk("sign", p.sign, -2, 2);
    chk("maxpts", p.maxpts, 0, 10000000);
    chk("p1percent", p.p1percent, 0, 100);
    chk("dilate", p.dilate, 0, 1000);
    chk("blur", p.blur, 0, 4);
    chk("tsmooth", p.tsmooth, 0, 255);
    chk("show", p.show, 0, 2);
    chk("mark_v", p.mark_v, 0, 255);
    chk("seg", p.seg, 0, 2);
    chk("mscene", p.mscene, 0, 100);
    chk("minpts", p.minpts, 0, 10000000);
    chk("spotmax1", p.spotmax1, 1, 10000000);
    chk("spotmax2", p.spotmax2, 1, 10000000);
    chk("asm_opt", p.asm_opt, 0, 3);
    if (e.empty() && !simdAvailable(p.asm_opt)) e = "asm_opt=" + std::to_string(p.asm_opt) + " is not supported by this CPU";
    return e;
}

struct Engine::Work {
    std::vector<uint8_t> noise[3]; // frames n-1, n, n+1
    std::vector<uint8_t> motion[3]; // n, n+1, merged
    std::unique_ptr<Segment[]> segs_mem[3];
    Segments segs[3];

    Work(size_t size, size_t max_segments) {
        for (int i = 0; i < 3; ++i) {
            noise[i].assign(size, 0);
            motion[i].assign(size, 0);
            segs_mem[i].reset(new Segment[max_segments + 2]);
            segs[i].data = segs_mem[i].get();
            segs[i].clear();
        }
    }
};

Engine::Engine(const Params &p, const Format &f, int nframes, int64_t fnum, int64_t fden)
    : params(p), format(f), num_frames(nframes), fps_num(fnum), fps_den(fden) {
    const std::string err = validate(p);
    if (!err.empty()) throw std::runtime_error(err);
    if (f.width < 1 || f.height < 1) throw std::runtime_error("clip must have a constant, non-zero frame size");
    if (f.width > 32000 || f.height > 32000) throw std::runtime_error("frame size must not exceed 32000");
    if (!f.gray && (f.ssw < 0 || f.ssw > 2 || f.ssh < 0 || f.ssh > 2))
        throw std::runtime_error("chroma subsampling must be at most 4 in each direction");
    if (f.bits < 8 || f.bits > 16) throw std::runtime_error("only integer samples with 8 to 16 bits are supported");
    if (f.ext_bits < 8 || f.ext_bits > 16) throw std::runtime_error("extmask must have 8 to 16 bit integer samples");

    parms.width = f.width;
    parms.height = f.height;
    parms.pitch = (f.width + 63) & ~63;
    parms.size = static_cast<size_t>(parms.pitch) * f.height;
    parms.p1 = static_cast<uint8_t>(p.p1);
    parms.p2 = static_cast<uint8_t>(p.p2);
    parms.pwidth = p.pwidth;
    parms.pheight = p.pheight;
    parms.y_next = p.interlaced ? 2 : 1;
    parms.mthres = static_cast<uint8_t>(p.mthres);
    parms.mwidth = p.mwidth;
    parms.mheight = p.mheight;
    parms.merode = p.merode;
    parms.median = p.median;
    parms.show = p.show;
    parms.mark_v = static_cast<uint8_t>(p.mark_v);
    parms.show_chroma = p.show_chroma;
    parms.ranked = p.ranked;
    parms.sign = p.sign;
    parms.maxpts = p.maxpts;
    parms.p1percent = p.p1percent;
    parms.dilate = p.dilate;
    parms.fitluma = p.fitluma;
    parms.blur = p.blur;
    parms.tsmooth = p.tsmooth;
    parms.motpn = p.motpn;
    parms.seg = p.seg;
    parms.color = p.color;
    parms.mscene = p.mscene;
    parms.minpts = p.minpts;
    parms.shift = f.bits - 8;
    parms.maxval = (1 << f.bits) - 1;
    parms.k = &kernelsFor(p.asm_opt);
    // at most one segment per two pixels of a row
    max_segments = static_cast<size_t>(f.height) * ((f.width + 1) / 2);
    // Map cache: a frame in flight needs 5 entries, consecutive frames share most of them. Keep room for
    // all host threads, but at most about 512 MB of maps.
    const size_t threads = std::max(1u, std::thread::hardware_concurrency());
    cache_capacity = std::max<size_t>(12, 3 * (threads + 2));
    cache_capacity = std::min(cache_capacity, std::max<size_t>(12, (size_t(512) << 20) / std::max<size_t>(parms.size, 1)));

    if (!p.outfile.empty()) {
        outfile = std::fopen(p.outfile.c_str(), "wb");
        if (!outfile) throw std::runtime_error("failed to open " + p.outfile + " for writing");
        const double diag = std::sqrt(double(f.width) * f.width + double(f.height) * f.height);
        const int thickness = int(std::floor(diag / 860 + 0.75));
        std::fprintf(outfile, ass_header, f.width, f.height, thickness);
        written.assign(num_frames > 0 ? num_frames : 0, false);
        while (next_spot_frame < num_frames && !reportsSpots(next_spot_frame)) ++next_spot_frame;
    }
}

Engine::~Engine() {
    if (outfile) {
        flushSpots(true); // frames after a gap (frames that were never requested), in frame order
        std::fclose(outfile);
    }
}

bool Engine::reportsSpots(int n) const {
    // in mc mode only the middle frame of every three is a real frame
    const int r = params.mc ? 3 : 1;
    return processes(n) && n == (n / r) * r + (r - 1) / 2;
}

void Engine::flushSpots(bool all) {
    bool wrote = false;
    while (!pending.empty()) {
        auto it = pending.begin();
        if (!all && it->first != next_spot_frame) break;
        std::fputs(it->second.c_str(), outfile);
        wrote = true;
        next_spot_frame = it->first + 1;
        while (next_spot_frame < num_frames && !reportsSpots(next_spot_frame)) ++next_spot_frame;
        pending.erase(it);
    }
    if (wrote) std::fflush(outfile);
}

std::unique_ptr<Engine::Work> Engine::acquire() {
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!pool.empty()) {
            std::unique_ptr<Work> w = std::move(pool.back());
            pool.pop_back();
            return w;
        }
    }
    return std::make_unique<Work>(parms.size, max_segments);
}

void Engine::release(std::unique_ptr<Work> w) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    pool.push_back(std::move(w));
}

std::shared_ptr<const Engine::MapEntry> Engine::cachedMap(MapKind kind, int frame,
                                                          const std::function<void(MapEntry &)> &compute) {
    const std::pair<int, int> key(static_cast<int>(kind), frame);
    std::shared_ptr<MapEntry> e;
    {
        std::unique_lock<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            e = it->second;
            e->last_use = ++cache_tick;
            cache_cv.wait(lock, [&] { return e->ready; });
            if (!e->failed) return e;
            cache.erase(key); // the computing thread failed: try again
        }
        e = std::make_shared<MapEntry>();
        e->last_use = ++cache_tick;
        cache[key] = e;
        // evict the least recently used finished entries that nobody holds
        while (cache.size() > cache_capacity) {
            auto victim = cache.end();
            for (auto i = cache.begin(); i != cache.end(); ++i)
                if (i->second->ready && i->second.use_count() == 1 &&
                    (victim == cache.end() || i->second->last_use < victim->second->last_use))
                    victim = i;
            if (victim == cache.end()) break;
            cache.erase(victim);
        }
    }
    try {
        compute(*e);
    } catch (...) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        e->failed = e->ready = true;
        cache_cv.notify_all();
        throw;
    }
    std::lock_guard<std::mutex> lock(cache_mutex);
    e->ready = true;
    cache_cv.notify_all();
    return e;
}

void Engine::writeSpots(int fn, const Segments &segs) {
    if (!reportsSpots(fn)) return;
    // in mc mode frame fn is frame fn / 3 of the source
    const int r = params.mc ? 3 : 1;
    const double fps = double(fps_num) / double(fps_den * r);
    std::string text;
    print_segments(segs, text, fn / r, fps, format.width, format.height, params.spotmax1, params.spotmax2);
    std::lock_guard<std::mutex> lock(outfile_mutex);
    if (written[fn]) return; // a frame requested twice
    written[fn] = true;
    pending.emplace(fn, std::move(text));
    flushSpots(false);
}

void Engine::process(int fn, const FrameR *const src[5], const PlaneR *ext, const FrameW &dst) {
    const FrameR *p = src[1], *c = src[2], *n = src[3];
    if (!p || !c || !n) throw std::runtime_error("internal error: missing neighbour frame");
    std::unique_ptr<Work> wk = acquire();
    if (format.bits == 8)
        processT<uint8_t>(fn, src, ext, dst, *wk);
    else
        processT<uint16_t>(fn, src, ext, dst, *wk);
    release(std::move(wk));
}

template <typename T>
void Engine::processT(int fn, const FrameR *const src[5], const PlaneR *ext, const FrameW &dst, Work &W) {
    const Parms &P = parms;
    const FrameR *pp = src[0], *p = src[1], *c = src[2], *n = src[3], *nn = src[4];
    constexpr ptrdiff_t B = sizeof(T);

    auto Y = [](const FrameR *f) { return reinterpret_cast<const T *>(f->plane[0].ptr); };
    auto S = [](const FrameR *f) { return f->plane[0].stride / static_cast<ptrdiff_t>(sizeof(T)); };
    T *o = reinterpret_cast<T *>(dst.plane[0].ptr);
    const ptrdiff_t os = dst.plane[0].stride / B;

    // spot candidates and segments of frame b (with neighbours a, c), or none if a neighbour is missing
    auto segments_of = [&](const FrameR *a, const FrameR *b, const FrameR *d, uint8_t *noise, Segments &segs) {
        if (a && d) {
            find_outliers(Y(a), S(a), Y(b), S(b), Y(d), S(d), noise, P);
            find_sizes(noise, segs, P);
        } else {
            segs.clear();
            finish_segments(segs);
        }
    };
    auto motion_pc = [&](const FrameR *a, const FrameR *b, uint8_t *m) { find_motion(Y(a), S(a), Y(b), S(b), m, P); };
    auto add_ext = [&](uint8_t *m) {
        if (!ext) return;
        if (format.ext_bits == 8)
            add_external_mask(ext->ptr, ext->stride, 0, m, P);
        else
            add_external_mask(reinterpret_cast<const uint16_t *>(ext->ptr), ext->stride / 2, format.ext_bits - 8, m, P);
    };

    uint8_t *c_noise = W.noise[1].data();
    uint8_t *c_motion = W.motion[0].data();
    uint8_t *m_motion = W.motion[2].data();
    Segments &c_segs = W.segs[1];

    Exec<T> *const exec_median[3] = {cond_median<T>, map_outliers<T>, map_outliers<T>};
    Exec<T> *const exec_pixel[3] = {remove_outliers<T>, mark_outliers<T>, map_outliers<T>};
    const uint8_t *noise_out = c_noise; // noise map used for the chroma planes
    const int row_bytes = P.width * static_cast<int>(B);

    if (params.median) { // simple median mode
        // motion of frame k against k-1, shared by the output frames k-1 and k
        auto median_motion = [&](const FrameR *a, const FrameR *b, int k) {
            return cachedMap(kMedianMotion, k, [&](MapEntry &e) {
                e.map.resize(P.size);
                motion_pc(a, b, e.map.data());
                motion_denoise(e.map.data(), P);
            });
        };
        const auto cm = median_motion(p, c, fn), nm = median_motion(c, n, fn + 1);
        motion_merge(cm->map.data(), nm->map.data(), m_motion, P);
        exec_median[P.show](Y(p), S(p), Y(c), S(c), nullptr, m_motion, Y(n), S(n), o, os, P);
        noise_out = nullptr;
    } else if (params.motpn) { // motion from previous to next frame
        motion_pc(p, n, c_motion);
        motion_denoise(c_motion, P);
        motion_scene(c_motion, P);
        add_ext(c_motion);
        segments_of(p, c, n, c_noise, c_segs);
        if (P.seg != 0) {
            reject_on_motion(c_segs, c_motion, P);
            copy_plane(c->plane[0], dst.plane[0], row_bytes, P.height);
            remove_segments(c_segs, Y(p), S(p), Y(n), S(n), o, os, c_noise, P);
            if (P.show == S_DENOISE) {
                if (P.tsmooth > 0) temporal_smooth(c_segs, Y(p), S(p), Y(n), S(n), o, os, c_noise, c_motion, P);
            } else if (P.show == S_MARK) {
                mark_outliers(Y(p), S(p), Y(c), S(c), c_noise, c_motion, Y(n), S(n), o, os, P);
            } else {
                map_outliers(Y(p), S(p), Y(c), S(c), c_noise, c_motion, Y(n), S(n), o, os, P);
            }
            if (outfile) writeSpots(fn, c_segs);
        } else {
            mark_noise(c_segs, c_noise, P);
            noise_dilate(c_noise, P);
            exec_pixel[P.show](Y(p), S(p), Y(c), S(c), c_noise, c_motion, Y(n), S(n), o, os, P);
        }
    } else { // motion from previous to current and from current to next frame
        // Noise of frames n-1, n, n+1 and motion of n (against n-1) and n+1 (against n) are shared with
        // the neighbouring output frames, so they come from the map cache.
        const FrameR *f3[3] = {p, c, n};
        const FrameR *before[3] = {pp, p, c};
        const FrameR *after[3] = {c, n, nn};
        std::shared_ptr<const MapEntry> noise[3], motion[2];
        for (int i = 0; i < 3; ++i) {
            noise[i] = cachedMap(kNoise, fn - 1 + i, [&](MapEntry &e) {
                Segments &tmp = W.segs[0];
                uint8_t *map = W.noise[0].data();
                segments_of(before[i], f3[i], after[i], map, tmp);
                mark_noise(tmp, map, P);
                if (P.seg == 0) noise_dilate(map, P);
                e.map.assign(map, map + P.size);
                const size_t count = static_cast<size_t>(tmp.end - tmp.data);
                e.segs.resize(count + 2);
                copy_segments(tmp.data, count, e.segs.data());
            });
        }
        for (int j = 0; j < 2; ++j) {
            motion[j] = cachedMap(kMotion, fn + j, [&](MapEntry &e) {
                e.map.resize(P.size);
                motion_pc(f3[j], f3[j + 1], e.map.data());
                noise_to_one(noise[j]->map.data(), noise[j + 1]->map.data(), e.map.data(), P);
                motion_denoise(e.map.data(), P);
                motion_scene(e.map.data(), P);
            });
        }
        const uint8_t *cm = motion[0]->map.data();
        motion_merge(cm, motion[1]->map.data(), m_motion, P);
        add_ext(m_motion);
        if (P.seg != 0) {
            // segments and noise map of frame n are changed below: work on copies
            const size_t count = noise[1]->segs.size() - 2;
            copy_segments(noise[1]->segs.data(), count, c_segs.data);
            c_segs.end = c_segs.data + count;
            reject_on_motion(c_segs, m_motion, P);
            copy_plane(c->plane[0], dst.plane[0], row_bytes, P.height);
            remove_segments(c_segs, Y(p), S(p), Y(n), S(n), o, os, c_noise, P);
            if (P.show == S_DENOISE) {
                if (P.tsmooth > 0) temporal_smooth(c_segs, Y(p), S(p), Y(n), S(n), o, os, c_noise, cm, P);
            } else if (P.show == S_MARK) {
                mark_outliers(Y(p), S(p), Y(c), S(c), c_noise, m_motion, Y(n), S(n), o, os, P);
            } else {
                map_outliers(Y(p), S(p), Y(c), S(c), c_noise, m_motion, Y(n), S(n), o, os, P);
            }
        } else {
            noise_out = noise[1]->map.data();
            exec_pixel[P.show](Y(p), S(p), Y(c), S(c), noise_out, m_motion, Y(n), S(n), o, os, P);
        }
    }

    // chroma planes
    if (!format.gray) {
        const int cw = format.width >> format.ssw;
        const int ch = format.height >> format.ssh;
        const T grey = static_cast<T>(128 << P.shift);
        for (int pl = 1; pl < 3; ++pl) {
            const PlaneW &d = dst.plane[pl];
            T *dp = reinterpret_cast<T *>(d.ptr);
            const ptrdiff_t ds = d.stride / B;
            auto C = [&](const FrameR *f) { return reinterpret_cast<const T *>(f->plane[pl].ptr); };
            auto CS = [&](const FrameR *f) { return f->plane[pl].stride / B; };
            if (P.show == S_MAP && !P.show_chroma) { // grey map
                for (int y = 0; y < ch; ++y) std::fill(dp + y * ds, dp + y * ds + cw, grey);
            } else if (P.show == S_MARK && !P.median) { // mark noise by colour in V
                copy_plane(c->plane[pl], d, cw * static_cast<int>(B), ch);
                if (pl == 2) mark_color_plane(noise_out, dp, ds, cw, ch, format.ssw, format.ssh, P);
            } else if (P.show == S_DENOISE && P.color && !P.median) { // clean colour at spots
                clean_color_plane(C(p), CS(p), C(c), CS(c), C(n), CS(n), cw, ch, format.ssw, format.ssh, noise_out,
                                  dp, ds, P);
            } else {
                copy_plane(c->plane[pl], d, cw * static_cast<int>(B), ch);
            }
        }
    }
}

} // namespace despot
