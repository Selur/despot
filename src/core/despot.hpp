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
#include <cstdio>
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

void find_outliers(const uint8_t *p, ptrdiff_t ppitch, const uint8_t *c, ptrdiff_t cpitch, const uint8_t *n,
                   ptrdiff_t npitch, uint8_t *noise, const Parms &);
void find_sizes(uint8_t *noise, Segments &segs, const Parms &);
void mark_noise(Segments &segs, uint8_t *noise, const Parms &);
void reject_on_motion(Segments &segs, const uint8_t *motion, const Parms &);
void find_motion(const uint8_t *p, ptrdiff_t ppitch, const uint8_t *c, ptrdiff_t cpitch, uint8_t *motion,
                 const Parms &);
void motion_denoise(uint8_t *motion, const Parms &);
void noise_dilate(uint8_t *noise, const Parms &);
void motion_scene(uint8_t *motion, const Parms &);
void motion_merge(const uint8_t *c_motion, const uint8_t *n_motion, uint8_t *m_motion, const Parms &);
void noise_to_one(const uint8_t *p_noise, const uint8_t *c_noise, uint8_t *c_motion, const Parms &);
void add_external_mask(const uint8_t *ext, ptrdiff_t ext_pitch, uint8_t *motion, const Parms &);

typedef void Exec(const uint8_t *p, ptrdiff_t ppitch, const uint8_t *c, ptrdiff_t cpitch, const uint8_t *c_noise,
                  const uint8_t *c_motion, const uint8_t *n, ptrdiff_t npitch, uint8_t *o, ptrdiff_t opitch,
                  const Parms &);
Exec cond_median;
Exec remove_outliers;
Exec mark_outliers;
Exec map_outliers;

void remove_segments(Segments &segs, const uint8_t *p, ptrdiff_t ppitch, const uint8_t *n, ptrdiff_t npitch,
                     uint8_t *o, ptrdiff_t opitch, uint8_t *c_noise, const Parms &);
void temporal_smooth(Segments &segs, const uint8_t *p, ptrdiff_t ppitch, const uint8_t *n, ptrdiff_t npitch,
                     uint8_t *o, ptrdiff_t opitch, uint8_t *c_noise, const uint8_t *c_motion, const Parms &);
// ssw/ssh: log2 chroma subsampling, 0..2 each (420: 1/1, 422: 1/0, 444: 0/0, 411: 2/0, 440: 0/1, 410: 2/2)
void clean_color_plane(const uint8_t *p, ptrdiff_t ppitch, const uint8_t *c, ptrdiff_t cpitch, const uint8_t *n,
                       ptrdiff_t npitch, int widthUV, int heightUV, int ssw, int ssh, const uint8_t *c_noise,
                       uint8_t *o, ptrdiff_t opitch, const Parms &);
void mark_color_plane(const uint8_t *c_noise, uint8_t *o, ptrdiff_t opitch, int widthUV, int heightUV, int ssw,
                      int ssh, const Parms &);
void print_segments(const Segments &segs, FILE *f, int fn, double frate, int w, int h, int spotmax1, int spotmax2);

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
    bool usesOutfile() const { return outfile != nullptr; }

    // src[i] is source frame n - 2 + i, or null when outside the clip (or not needed).
    // ext: luma of the extmask frame n, or null.
    void process(int n, const FrameR *const src[5], const PlaneR *ext, const FrameW &dst);

private:
    struct Work;
    std::unique_ptr<Work> acquire();
    void release(std::unique_ptr<Work> w);
    void writeSpots(int n, const Segments &segs);

    Params params;
    Format format;
    Parms parms;
    int num_frames;
    int64_t fps_num, fps_den;
    size_t max_segments;

    std::mutex pool_mutex;
    std::vector<std::unique_ptr<Work>> pool;

    FILE *outfile = nullptr;
    std::mutex outfile_mutex;
    std::vector<bool> written;
};

} // namespace despot
