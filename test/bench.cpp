// SPDX-License-Identifier: GPL-2.0-only
// Speed of the engine and of its stages on synthetic 1080p frames (no host, no decoding).
// Usage: bench [asm_opt] [frames]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/despot.hpp"

using namespace despot;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int W = 1920, H = 1080;

struct Clip {
    std::vector<std::vector<uint8_t>> y, u, v;
};

Clip makeClip(int frames) {
    std::mt19937 rng(7);
    Clip c;
    for (int f = 0; f < frames; ++f) {
        std::vector<uint8_t> y(W * H), u(W * H / 4, 128), v(W * H / 4, 128);
        for (int yy = 0; yy < H; ++yy)
            for (int x = 0; x < W; ++x) {
                int val = 60 + x * 100 / W + (yy % 37) + static_cast<int>(rng() % 5);
                if (x > 200 + f * 7 && x < 400 + f * 7 && yy > 300 && yy < 500) val = 200; // moving box
                y[yy * W + x] = static_cast<uint8_t>(val);
            }
        for (int s = 0; s < 300; ++s) { // spots
            const int sx = rng() % (W - 8), sy = rng() % (H - 6), sw = 1 + rng() % 7, sh = 1 + rng() % 5;
            const uint8_t val = (rng() & 1) ? 10 : 245;
            for (int j = 0; j < sh; ++j) std::memset(&y[(sy + j) * W + sx], val, sw);
        }
        c.y.push_back(std::move(y));
        c.u.push_back(std::move(u));
        c.v.push_back(std::move(v));
    }
    return c;
}

double runEngine(const Clip &clip, const Params &p) {
    Format fmt;
    fmt.width = W;
    fmt.height = H;
    const int nf = static_cast<int>(clip.y.size());
    Engine e(p, fmt, nf, 25, 1);
    std::vector<uint8_t> oy(W * H), ou(W * H / 4), ov(W * H / 4);
    FrameW out;
    out.plane[0] = {oy.data(), W};
    out.plane[1] = {ou.data(), W / 2};
    out.plane[2] = {ov.data(), W / 2};
    std::vector<FrameR> fr(nf);
    for (int i = 0; i < nf; ++i) {
        fr[i].plane[0] = {clip.y[i].data(), W};
        fr[i].plane[1] = {clip.u[i].data(), W / 2};
        fr[i].plane[2] = {clip.v[i].data(), W / 2};
    }
    const auto t0 = Clock::now();
    int done = 0;
    for (int n = 1; n < nf - 1; ++n) {
        const FrameR *src[5] = {};
        for (int i = 0; i < 5; ++i) {
            const int k = n - 2 + i;
            if (k >= 0 && k < nf && k >= e.firstNeeded(n) && k <= e.lastNeeded(n)) src[i] = &fr[k];
        }
        e.process(n, src, nullptr, out);
        ++done;
    }
    return done / std::chrono::duration<double>(Clock::now() - t0).count();
}

template <class F>
double ms(F f, int reps = 20) {
    const auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;
}

} // namespace

int main(int argc, char **argv) {
    const int opt = argc > 1 ? std::atoi(argv[1]) : 0;
    const int frames = argc > 2 ? std::atoi(argv[2]) : 40;
    const Clip clip = makeClip(frames);

    const char *modes[][2] = {{"defaults", ""}, {"motpn=false", "m"}, {"motpn=false seg=0", "m0"},
                              {"seg=0", "0"},  {"median", "d"}};
    for (auto &m : modes) {
        Params p;
        p.asm_opt = opt;
        const std::string k = m[1];
        if (k.find('m') != std::string::npos) p.motpn = false;
        if (k.find('0') != std::string::npos) p.seg = 0;
        if (k == "d") p.median = true;
        std::printf("%-20s %7.1f fps\n", m[0], runEngine(clip, p));
    }

    // stages on one frame
    Params p;
    p.asm_opt = opt;
    Parms P;
    P.width = W;
    P.height = H;
    P.pitch = (W + 63) & ~63;
    P.size = static_cast<size_t>(P.pitch) * H;
    P.k = &kernelsFor(opt);
    std::vector<uint8_t> noise(P.size), motion(P.size);
    std::vector<Segment> segmem(static_cast<size_t>(H) * (W + 1) / 2 + 2);
    Segments segs;
    segs.data = segmem.data();
    const uint8_t *a = clip.y[0].data(), *b = clip.y[1].data(), *c = clip.y[2].data();
    std::printf("\nstages (ms per call):\n");
    std::printf("find_outliers   %6.2f\n", ms([&] { find_outliers(a, W, b, W, c, W, noise.data(), P); }));
    std::printf("find_sizes      %6.2f\n", ms([&] {
                    find_outliers(a, W, b, W, c, W, noise.data(), P);
                    find_sizes(noise.data(), segs, P);
                }) - ms([&] { find_outliers(a, W, b, W, c, W, noise.data(), P); }));
    std::printf("mark_noise      %6.2f\n", ms([&] { mark_noise(segs, noise.data(), P); }));
    std::printf("noise_dilate    %6.2f\n", ms([&] { noise_dilate(noise.data(), P); }));
    std::printf("find_motion     %6.2f\n", ms([&] { find_motion(a, W, c, W, motion.data(), P); }));
    std::printf("motion_denoise  %6.2f (incl. find_motion)\n", ms([&] {
                    find_motion(a, W, c, W, motion.data(), P);
                    motion_denoise(motion.data(), P);
                }));
    std::printf("motion_scene    %6.2f\n", ms([&] { motion_scene(motion.data(), P); }));
    std::printf("reject_on_motion%6.2f\n", ms([&] { reject_on_motion(segs, motion.data(), P); }));
    return 0;
}
