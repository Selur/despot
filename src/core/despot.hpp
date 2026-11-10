// SPDX-License-Identifier: GPL-2.0-only
/*
    DeSpot - Conditional Temporal Despotting Filter

    Copyright (C)2003-2008 Alexander G. Balakhnin aka Fizick.
    Based on Conditional Temporal Median Filter 0.93 (C) 2003 Kevin Atkinson.
    Outfile/mc additions (3.6.x) by Firesledge.
    AviSynth+/VapourSynth port: host-independent engine.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 2 as
    published by the Free Software Foundation.
*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kernels.hpp"

namespace despot {

constexpr int S_DENOISE = 0;
constexpr int S_MARK = 1;
constexpr int S_MAP = 2;

// User parameters; defaults as in DeSpot 3.6.3.
struct Params {
    int mthres = 16;
    int mwidth = 7;
    int mheight = 5;
    int merode = 33;
    bool interlaced = false;
    bool median = false;
    int p1 = 24;
    int p2 = 12;
    int pwidth = 16;
    int pheight = 5;
    bool ranked = true;
    int sign = 0;
    int maxpts = 0;
    int p1percent = 10;
    int dilate = 1;
    bool fitluma = false;
    int blur = 1;
    int tsmooth = 0;
    int show = 0;
    int mark_v = 255;
    bool show_chroma = false;
    bool motpn = true;
    int seg = 2;
    bool color = false;
    int mscene = 40;
    int minpts = 0;
    std::string outfile;
    bool mc = false;
    int spotmax1 = 12;
    int spotmax2 = 20;
    int asm_opt = 0; // instruction set, see kernelsFor()
};

// Checks every parameter; returns an error message or an empty string.
std::string validate(const Params &p);

// Derived, frame-size dependent values used by the filter functions.
struct Parms {
    int width = 0;
    int height = 0;
    int pitch = 0; // stride of the internal noise/motion maps
    size_t size = 0;
    uint8_t p1 = 24;
    uint8_t p2 = 12;
    int pwidth = 16;
    int pheight = 5;
    int y_next = 1;
    uint8_t mthres = 16;
    int mwidth = 7;
    int mheight = 5;
    int merode = 33;
    bool median = false;
    int show = 0;
    uint8_t mark_v = 255;
    bool show_chroma = false;
    bool ranked = true;
    int sign = 0;
    int maxpts = 0;
    int p1percent = 10;
    int dilate = 1;
    bool fitluma = false;
    int blur = 1;
    int tsmooth = 0;
    bool motpn = true;
    int seg = 2;
    bool color = false;
    int mscene = 40;
    int minpts = 0;
    int shift = 0;      // bit depth - 8
    int maxval = 255;   // largest sample value
    const Kernels *k = nullptr;
};

constexpr char B_NOTHING = 0;
constexpr char B_SEGMENT = 1; // single segment
constexpr char B_LINK = 2;    // has master
constexpr char B_MASTER = 4;  // has slave link

struct Segment {
    bool p1yes;
    char what;
    short ly;  // top
    short lx1; // left
    short lx2; // right
    long nump1;
    long nump2;
    long sumnoise;
    struct Data {
        struct BStruct {
            short x1, x2, y1, y2;
        } b;
        struct SStruct {
            short width, height;
            bool is_noise;
        } s;
        Segment *nowis;
    } data;

    void init(short y, short x) {
        p1yes = false;
        what = B_SEGMENT;
        ly = y;
        lx1 = lx2 = x;
        nump1 = nump2 = sumnoise = 0;
        data.b = {x, x, y, y};
        data.s = {0, 0, false};
        data.nowis = nullptr;
    }
};

// Segments of one frame. 'end' always has two writable sentinel slots behind it.
struct Segments {
    Segment *data = nullptr;
    Segment *end = nullptr;
    void clear() { end = data; }
};

// Writes the two sentinels behind segs.end (after segs.clear() or new segments).
void finish_segments(Segments &segs);

// Sample functions: T is uint8_t (8 bit) or uint16_t (9..16 bit); pitches are in samples.
template <typename T>
void find_outliers(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const T *n, ptrdiff_t npitch,
                   uint8_t *noise, const Parms &);
void find_sizes(uint8_t *noise, Segments &segs, const Parms &);
void mark_noise(Segments &segs, uint8_t *noise, const Parms &);
void reject_on_motion(Segments &segs, const uint8_t *motion, const Parms &);
template <typename T>
void find_motion(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, uint8_t *motion, const Parms &);
void motion_denoise(uint8_t *motion, const Parms &);
void noise_dilate(uint8_t *noise, const Parms &);
void motion_scene(uint8_t *motion, const Parms &);
void motion_merge(const uint8_t *c_motion, const uint8_t *n_motion, uint8_t *m_motion, const Parms &);
void noise_to_one(const uint8_t *p_noise, const uint8_t *c_noise, uint8_t *c_motion, const Parms &);
// ext_shift: bit depth of the mask - 8
template <typename E>
void add_external_mask(const E *ext, ptrdiff_t ext_pitch, int ext_shift, uint8_t *motion, const Parms &);

#define EXEC_ARGS(T)                                                                                                  \
    const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const uint8_t *c_noise, const uint8_t *c_motion,      \
        const T *n, ptrdiff_t npitch, T *o, ptrdiff_t opitch, const Parms &
template <typename T>
using Exec = void(EXEC_ARGS(T));
template <typename T> void cond_median(EXEC_ARGS(T));
template <typename T> void remove_outliers(EXEC_ARGS(T));
template <typename T> void mark_outliers(EXEC_ARGS(T));
template <typename T> void map_outliers(EXEC_ARGS(T));

template <typename T>
void remove_segments(Segments &segs, const T *p, ptrdiff_t ppitch, const T *n, ptrdiff_t npitch, T *o,
                     ptrdiff_t opitch, uint8_t *c_noise, const Parms &);
template <typename T>
void temporal_smooth(Segments &segs, const T *p, ptrdiff_t ppitch, const T *n, ptrdiff_t npitch, T *o,
                     ptrdiff_t opitch, uint8_t *c_noise, const uint8_t *c_motion, const Parms &);
// ssw/ssh: log2 chroma subsampling, 0..2 each (420: 1/1, 422: 1/0, 444: 0/0, 411: 2/0, 440: 0/1, 410: 2/2)
template <typename T>
void clean_color_plane(const T *p, ptrdiff_t ppitch, const T *c, ptrdiff_t cpitch, const T *n, ptrdiff_t npitch,
                       int widthUV, int heightUV, int ssw, int ssh, const uint8_t *c_noise, T *o, ptrdiff_t opitch,
                       const Parms &);
template <typename T>
void mark_color_plane(const uint8_t *c_noise, T *o, ptrdiff_t opitch, int widthUV, int heightUV, int ssw, int ssh,
                      const Parms &);
// Appends the outfile (ASS) lines of the spots of one frame to out.
void print_segments(const Segments &segs, std::string &out, int fn, double frate, int w, int h, int spotmax1,
                    int spotmax2);

// ---------------------------------------------------------------------------------------------
// Engine: everything for one output frame, computed from the source frames n-2..n+2 only, so
// the hosts may call process() from several threads at once.

struct PlaneR {
    const uint8_t *ptr = nullptr;
    ptrdiff_t stride = 0;
};
struct PlaneW {
    uint8_t *ptr = nullptr;
    ptrdiff_t stride = 0;
};
struct FrameR {
    PlaneR plane[3];
};
struct FrameW {
    PlaneW plane[3];
};

struct Format {
    int width = 0;
    int height = 0;
    int ssw = 1; // log2 horizontal chroma subsampling
    int ssh = 1; // log2 vertical chroma subsampling
    bool gray = false;
    int bits = 8;     // 8..16 bit integer samples
    int ext_bits = 8; // bit depth of extmask
};

class Engine {
public:
    // Throws std::runtime_error for bad parameters, formats or an unwritable outfile.
    Engine(const Params &params, const Format &format, int num_frames, int64_t fps_num, int64_t fps_den);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    // Source frames the output frame n needs (indices may be outside the clip; the host skips those).
    int firstNeeded(int n) const { return params.motpn || params.median ? n - 1 : n - 2; }
    int lastNeeded(int n) const { return params.motpn || params.median ? n + 1 : n + 2; }
    // False for the first and last frame: the output is the unchanged source frame.
    bool processes(int n) const { return n > 0 && n < num_frames - 1; }

    // src[i] is source frame n - 2 + i, or null when outside the clip (or not needed); strides in bytes.
    // ext: first plane of the extmask frame n, or null.
    void process(int n, const FrameR *const src[5], const PlaneR *ext, const FrameW &dst);

private:
    struct Work;
    std::unique_ptr<Work> acquire();
    void release(std::unique_ptr<Work> w);
    template <typename T>
    void processT(int n, const FrameR *const src[5], const PlaneR *ext, const FrameW &dst, Work &w);
    void writeSpots(int n, const Segments &segs);

    // Per-frame maps shared between output frames (motpn=false and median mode): computed once by the
    // first thread that needs them, other threads wait for that result. Entries are never modified.
    struct MapEntry {
        std::vector<uint8_t> map;    // noise map or motion map
        std::vector<Segment> segs;   // noise entries: the segments plus two sentinels
        bool ready = false;
        bool failed = false;
        uint64_t last_use = 0;
    };
    enum MapKind { kNoise = 0, kMotion = 1, kMedianMotion = 2 };
    std::shared_ptr<const MapEntry> cachedMap(MapKind kind, int frame, const std::function<void(MapEntry &)> &compute);


    Params params;
    Format format;
    Parms parms;
    int num_frames;
    int64_t fps_num, fps_den;
    size_t max_segments;

    std::mutex pool_mutex;
    std::vector<std::unique_ptr<Work>> pool;

    std::mutex cache_mutex;
    std::condition_variable cache_cv;
    std::map<std::pair<int, int>, std::shared_ptr<MapEntry>> cache;
    size_t cache_capacity = 16;
    uint64_t cache_tick = 0;

    FILE *outfile = nullptr;
    // Spot lines are produced in any order (threads, seeking) and written in frame order: the lines of
    // a frame wait in 'pending' until all earlier frames that report spots are written.
    bool reportsSpots(int n) const;
    void flushSpots(bool all);
    std::mutex outfile_mutex;
    std::vector<bool> written;
    std::map<int, std::string> pending;
    int next_spot_frame = 0;
};

} // namespace despot
