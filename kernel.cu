/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>
*/

#ifndef __NVRTC__
#include "fitness.h"
#include "jit.h"
#include "ports/sim.cuh"
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#endif

#ifndef __NVRTC__
#define CUDA_CALL(call)                                                   \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e));                              \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#ifndef CU_CHECK
#define CU_CHECK(call)                                                 \
    do {                                                               \
        CUresult _r = (call);                                          \
        if (_r != CUDA_SUCCESS) {                                      \
            const char* str = nullptr;                                 \
            cuGetErrorString(_r, &str);                                \
            fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__, \
                    __LINE__, str ? str : "unknown");                  \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)
#endif
#endif

static constexpr int MAX_BRANCHES = 8;

__device__ __constant__ uint8_t BRANCH_BIAS[MAX_BRANCHES] = {
    0x0,       // 0: greedy.
    0x4,       // 1: thrust.
    0x1,       // 2: left.
    0x2,       // 3: right.
    0x5,       // 4: thrust+left.
    0x6,       // 5: thrust+right.
    0x8,       // 6: suppress fire.
    0x4 | 0x8, // 7: thrust, suppress fire.
};

__device__ __forceinline__ uint8_t applyBias(uint8_t greedy, int branch) {
    uint8_t bias = BRANCH_BIAS[branch];
    uint8_t out = greedy;
    out |= (bias & 0x7);
    if (bias & 0x8) {
        out &= ~0x8u;
    }
    return out & 0xf;
}

__device__ uint8_t clear(const Simulation& sim) {
    if (!sim.ship.canControl) {
        return 0;
    }

    const Ship& ship = sim.ship;
    int32_t best = -1;
    bool isSaucer = false;
    int64_t bestDist = INT64_MAX;
    for (int i = 0; i < 3; i++) {
        if (!sim.saucers[i].alive) {
            continue;
        }
        int32_t dx = shortDX(ship.x, sim.saucers[i].x);
        int32_t dy = shortDY(ship.y, sim.saucers[i].y);
        int32_t dist = (int32_t)__fsqrt_rn((float)((int64_t)dx * dx + (int64_t)dy * dy));
        int32_t speed = SHIP_BSPEED_Q88 >> 4;
        int32_t frames = (speed > 0) ? dist / speed : 999;
        uint8_t angle = (uint8_t)simAtan2(dy, dx);
        int32_t delta = ((int32_t)angle - (int32_t)ship.angle + 256) & 0xff;
        if (delta > 128) {
            delta = 256 - delta;
        }
        int64_t time = ((int64_t)delta + frames) >> 2;
        if (time < bestDist) {
            bestDist = time;
            best = i;
            isSaucer = true;
        }
    }
    for (int i = 0; i < ASTEROID_CAP; i++) {
        if (!sim.asteroids[i].alive) {
            continue;
        }
        int32_t dx = shortDX(ship.x, sim.asteroids[i].x);
        int32_t dy = shortDY(ship.y, sim.asteroids[i].y);
        int32_t dist = (int32_t)__fsqrt_rn((float)((int64_t)dx * dx + (int64_t)dy * dy));
        int32_t speed = SHIP_BSPEED_Q88 >> 4;
        int32_t frames = (speed > 0) ? dist / speed : 999;
        uint8_t angle = (uint8_t)simAtan2(dy, dx);
        int32_t delta = ((int32_t)angle - (int32_t)ship.angle + 256) & 0xff;
        if (delta > 128) {
            delta = 256 - delta;
        }
        int64_t time = (int64_t)delta + frames;
        if (time < bestDist) {
            bestDist = time;
            best = i;
            isSaucer = false;
        }
    }

    if (best < 0) {
        return 0;
    }

    int32_t tx, ty, tvx, tvy;
    if (isSaucer) {
        tx = sim.saucers[best].x;
        ty = sim.saucers[best].y;
        tvx = sim.saucers[best].vx;
        tvy = sim.saucers[best].vy;
    } else {
        tx = sim.asteroids[best].x;
        ty = sim.asteroids[best].y;
        tvx = sim.asteroids[best].vx;
        tvy = sim.asteroids[best].vy;
    }

    int32_t dx = shortDX(ship.x, tx);
    int32_t dy = shortDY(ship.y, ty);
    int32_t speed = SHIP_BSPEED_Q88 >> 4;
    int32_t lead = (speed > 0) ? min((int32_t)__fsqrt_rn((float)((int64_t)dx * dx + (int64_t)dy * dy)) / speed, 60) : 0;
    int32_t pdx = shortDX(ship.x, tx + (tvx >> 4) * lead);
    int32_t pdy = shortDY(ship.y, ty + (tvy >> 4) * lead);
    uint8_t angle = (uint8_t)simAtan2(pdy, pdx);
    int delta = ((int)angle - (int)ship.angle + 256) & 0xff;
    if (delta > 128) {
        delta = 256 - delta;
    }

    int8_t dir = 0;
    {
        int d = ((int)angle - (int)ship.angle + 256) & 0xff;
        if (d != 0) {
            dir = (d <= 128) ? 1 : -1;
        }
    }

    uint8_t inp = 0;
    if (dir == -1) {
        inp |= INPUT_LEFT;
    } else if (dir == 1) {
        inp |= INPUT_RIGHT;
    }

    if (delta <= 18 && ship.fireCooldown == 0 && sim.bulletCount < SHIP_BLIMIT) {
        inp |= INPUT_FIRE;
    }
    return inp & 0xf;
}

__device__ uint8_t farm(const Simulation& sim) {
    if (!sim.ship.canControl) {
        return 0;
    }
    const Ship& ship = sim.ship;
    int id = -1;
    int64_t best = INT64_MAX;
    int32_t speed = SHIP_BSPEED_Q88 >> 4;
    for (int i = 0; i < 3; i++) {
        if (!sim.saucers[i].alive) {
            continue;
        }
        int32_t dx = shortDX(ship.x, sim.saucers[i].x);
        int32_t dy = shortDY(ship.y, sim.saucers[i].y);
        int32_t dist = (int32_t)__fsqrt_rn((float)((int64_t)dx * dx + (int64_t)dy * dy));
        int32_t frames = (speed > 0) ? dist / speed : 999;
        uint8_t angle = (uint8_t)simAtan2(dy, dx);
        int32_t delta = ((int32_t)angle - (int32_t)ship.angle + 256) & 0xff;
        if (delta > 128) {
            delta = 256 - delta;
        }
        int64_t time = (int64_t)delta + frames;
        int32_t cull = (sim.saucers[i].vx > 0) ? (SAUCER_CULL_MAX_X_Q12_4 - sim.saucers[i].x) : (sim.saucers[i].x - SAUCER_CULL_MIN_X_Q12_4);
        if (cull < 2048) {
            time -= 64;
        }
        if (time < best) {
            best = time;
            id = i;
        }
    }

    uint8_t inp = 0;
    if (id >= 0) {
        int32_t tx = sim.saucers[id].x, ty = sim.saucers[id].y;
        int32_t tvx = sim.saucers[id].vx, tvy = sim.saucers[id].vy;
        int32_t dx = shortDX(ship.x, tx), dy = shortDY(ship.y, ty);
        int32_t dist = (int32_t)__fsqrt_rn((float)((int64_t)dx * dx + (int64_t)dy * dy));
        int32_t speed = SHIP_BSPEED_Q88 >> 4;
        int32_t lead = (speed > 0) ? min(dist / speed, 48) : 0;
        int32_t pdx = shortDX(ship.x, tx + (tvx >> 4) * lead);
        int32_t pdy = shortDY(ship.y, ty + (tvy >> 4) * lead);
        uint8_t angle = (uint8_t)simAtan2(pdy, pdx);
        int delta = ((int)angle - (int)ship.angle + 256) & 0xff;
        if (delta > 128) {
            delta = 256 - delta;
        }
        int raw = ((int)angle - (int)ship.angle + 256) & 0xff;
        int8_t dir = (raw == 0) ? 0 : ((raw <= 128) ? 1 : -1);
        if (dir == -1) {
            inp |= INPUT_LEFT;
        } else if (dir == 1) {
            inp |= INPUT_RIGHT;
        }

        if (delta <= 7 && ship.fireCooldown == 0 && sim.bulletCount < SHIP_BLIMIT) {
            inp |= INPUT_FIRE;
        }
    }
    return inp & 0xf;
}

__device__ __forceinline__ uint8_t decide(const Simulation& sim, int32_t wave) {
    if (wave > 0 && sim.wave >= wave && sim.astCount == 1) {
        return farm(sim);
    }
    return clear(sim);
}

__global__ void launchKernel(const Simulation* __restrict__ states, int32_t width, int32_t horizon,
    int32_t wave, int32_t branches, Simulation* outStates, float* outFitness, uint8_t* outNibbles, int32_t bytes) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * branches;
    if (id >= total) {
        return;
    }

    int parent = id / branches;
    int branch = id % branches;
    Simulation sim = states[parent];
    uint8_t* out = outNibbles + (int64_t)id * bytes;
    for (int i = 0; i < bytes; i++) {
        out[i] = 0;
    }

    if (sim.gameOver) {
        outStates[id] = sim;
        outFitness[id] = fitness(sim, wave);
        return;
    }

   for (int f = 0; f < horizon && !sim.gameOver; f++) {
        uint8_t greedy = decide(sim, wave);
        uint8_t inp = applyBias(greedy, branch);
        int b = f >> 1;
        if (f & 1) {
            out[b] |= (inp << 4);
        } else {
            out[b] = inp;
        }
        simStep(sim, inp);
    }

    outStates[id] = sim;
    outFitness[id] = fitness(sim, wave);
}

__global__ void initKernel(Simulation* out, uint32_t seed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        simInit(out[0], seed);
    }
}

#ifndef __NVRTC__
extern "C" void
runSearch(int device, uint32_t seed, uint32_t salt, int32_t width,
          int32_t horizon, int32_t frames, int32_t wave, int32_t branches, int32_t* outScore, uint8_t* outTape, int32_t* outFrames,
          const std::chrono::steady_clock::time_point& start, JitKernel* jit, bool trace, const Simulation* replay = nullptr, int32_t replayFrame = 0) {
    CUDA_CALL(cudaSetDevice(device));
    if (branches < 1 || branches > MAX_BRANCHES) {
        std::fprintf(stderr, "branch count must be 1..%d\n", MAX_BRANCHES);
        exit(EXIT_FAILURE);
    }

    const int size = width * branches;
    int bytes = (horizon + 1) >> 1;
    Simulation* beam;
    Simulation* expanded;
    float* fitness;
    uint8_t* nibbles;
    auto last = start;

    CUDA_CALL(cudaMalloc(&beam, width * sizeof(Simulation)));
    CUDA_CALL(cudaMalloc(&expanded, size * sizeof(Simulation)));
    CUDA_CALL(cudaMalloc(&fitness, size * sizeof(float)));
    CUDA_CALL(cudaMalloc(&nibbles, (int64_t)size * bytes));

    std::vector<float> hFitness(size);
    std::vector<Simulation> hExpanded(size);
    std::vector<uint8_t> hNibbles((int64_t)size * bytes);
    const int maxBytes = (frames + 1) >> 1;
    std::vector<std::vector<uint8_t>> tapes(width, std::vector<uint8_t>(maxBytes, 0));
    if (replay && replayFrame > 0) {
        int32_t bytes = (replayFrame + 1) >> 1;
        for (int k = 0; k < width; k++) {
            memcpy(tapes[k].data(), outTape, bytes);
        }
    }

    int32_t pos = replayFrame;
    if (replay) {
        CUDA_CALL(cudaMemcpy(beam, replay, sizeof(Simulation), cudaMemcpyHostToDevice));
    } else {
        initKernel<<<1, 1>>>(beam, seed);
        CUDA_CALL(cudaDeviceSynchronize());
    }

    for (int i = 1; i < width; i++) {
        CUDA_CALL(cudaMemcpy(beam + i, beam, sizeof(Simulation), cudaMemcpyDeviceToDevice));
    }

    int32_t roundFrames = replayFrame;
    std::vector<Simulation> newBeam(width);
    std::vector<Simulation> prevBeam(width);
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    cudaEvent_t evKernelStart, evKernelStop;
    CUDA_CALL(cudaEventCreate(&evKernelStart));
    CUDA_CALL(cudaEventCreate(&evKernelStop));
    double totalKernelMs = 0.0;
    int32_t kernelRounds = 0;

    while (roundFrames < frames) {
        CUDA_CALL(cudaMemcpy(prevBeam.data(), beam,
                             width * sizeof(Simulation),
                             cudaMemcpyDeviceToHost));
        int curFrames = min(horizon, frames - roundFrames);
        CUDA_CALL(cudaEventRecord(evKernelStart));
        if (jit) {
            void* args[] = {
                &beam, &width, &curFrames,
                &wave, &branches, &expanded,
                &fitness, &nibbles, &bytes};
            CU_CHECK(cuLaunchKernel(jit->beam, (unsigned int)blocks, 1, 1, (unsigned int)threads, 1, 1, 0, nullptr, args, nullptr));
            CU_CHECK(cuCtxSynchronize());
        } else {
            launchKernel<<<blocks, threads>>>(beam, width, curFrames, wave, branches, expanded, fitness, nibbles, bytes);
            CUDA_CALL(cudaDeviceSynchronize());
        }
        CUDA_CALL(cudaEventRecord(evKernelStop));
        CUDA_CALL(cudaEventSynchronize(evKernelStop));
        float roundKernelMs = 0.0f;
        CUDA_CALL(cudaEventElapsedTime(&roundKernelMs, evKernelStart, evKernelStop));
        totalKernelMs += roundKernelMs;
        kernelRounds++;

        CUDA_CALL(cudaMemcpy(hFitness.data(), fitness, size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hExpanded.data(), expanded, size * sizeof(Simulation), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hNibbles.data(), nibbles, (int64_t)size * bytes, cudaMemcpyDeviceToHost));

        uint32_t rng = salt ^ (uint32_t)(roundFrames * 0x9e3779b9u);
        for (int i = 0; i < size; i++) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            hFitness[i] += (float)(rng & 0xffff) * (1.0f / 65536.0f) * 500.0f;
        }

        std::vector<int> tops;
        prune(hFitness.data(), size, width, tops);

       for (int k = 0; k < width; k++) {
            newBeam[k] = hExpanded[tops[k]];
        }

        {
            std::vector<std::vector<uint8_t>> newTapes(width);
            for (int k = 0; k < width; k++) {
                int slot = tops[k] / branches;
                int child = tops[k];
                newTapes[k] = tapes[slot];
                const uint8_t* win = hNibbles.data() + (int64_t)child * bytes;
                for (int f = 0; f < curFrames; f++) {
                    int nibPos = pos + f;
                    int byteIdx = nibPos >> 1;
                    uint8_t nib = (f & 1) ? (win[f >> 1] >> 4) : (win[f >> 1] & 0xf);
                    if (nibPos & 1) {
                        newTapes[k][byteIdx] |= (nib << 4);
                    } else {
                        newTapes[k][byteIdx] = nib;
                    }
                }
            }
            tapes = std::move(newTapes);
            pos += curFrames;
        }

        CUDA_CALL(cudaMemcpy(beam, newBeam.data(), width * sizeof(Simulation), cudaMemcpyHostToDevice));
        roundFrames += curFrames;

        {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - last).count();
            if (trace || elapsed >= 10.0 || roundFrames >= frames) {
                int bestSlotNow = 0;
                for (int k = 1; k < width; k++) {
                    if (newBeam[k].score > newBeam[bestSlotNow].score) {
                        bestSlotNow = k;
                    }
                }
                const Simulation& b = newBeam[bestSlotNow];
                int64_t ts = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

                if (trace) {
                    auto hwrap = [](int32_t v, int32_t sz) -> int32_t {
                        v %= sz;
                        if (v < 0) {
                            v += sz;
                        }
                        if (v > sz / 2) {
                            v -= sz;
                        }
                        return v;
                    };
                    int32_t sx = b.ship.x >> 4;
                    int32_t sy = b.ship.y >> 4;
                    int32_t spd = (int32_t)(std::sqrt((float)((int64_t)b.ship.vx * b.ship.vx + (int64_t)b.ship.vy * b.ship.vy)) * 100.0f / 256.0f);
                    int32_t minBulletDist = 9999;
                    for (int i = 0; i < SAUCER_BLIMIT; i++) {
                        if (!b.saucerBullets[i].alive) {
                            continue;
                        }
                        int32_t ddx = hwrap(b.ship.x - b.saucerBullets[i].x, WORLD_WIDTH_Q12_4) >> 4;
                        int32_t ddy = hwrap(b.ship.y - b.saucerBullets[i].y, WORLD_HEIGHT_Q12_4) >> 4;
                        int32_t d = (int32_t)std::sqrt((float)((int64_t)ddx * ddx + (int64_t)ddy * ddy));
                        if (d < minBulletDist) {
                            minBulletDist = d;
                        }
                    }
                    int32_t minScDist = 9999, scSmall = 0;
                    for (int i = 0; i < 3; i++) {
                        if (!b.saucers[i].alive) {
                            continue;
                        }
                        int32_t ddx = hwrap(b.ship.x - b.saucers[i].x, WORLD_WIDTH_Q12_4) >> 4;
                        int32_t ddy = hwrap(b.ship.y - b.saucers[i].y, WORLD_HEIGHT_Q12_4) >> 4;
                        int32_t d = (int32_t)std::sqrt(
                            (float)((int64_t)ddx * ddx + (int64_t)ddy * ddy));
                        if (d < minScDist) {
                            minScDist = d;
                            scSmall = b.saucers[i].small ? 1 : 0;
                        }
                    }
                    std::printf("[BEAM] frame=%05d score=%07d lives=%2d "
                                "wave=%2d fit=%07.0f time=%lld seed=0x%08X"
                                "  sc=%d sb=%d st=%d tslk=%d"
                                " ship=(%d,%d) spd=%d hdg=%d ctrl=%d"
                                " bul=%d bdist=%d scdist=%d sc=%d ast=%d\n",
                                roundFrames, b.score, b.lives, b.wave,
                                hFitness[tops[0]], (long long)ts, seed, b.saucerCount, b.saucerBulletCount,
                                b.saucerSpawnTimer, b.timeSinceLastKill, sx, sy, spd, (int32_t)b.ship.angle,
                                b.ship.canControl ? 1 : 0, b.bulletCount, minBulletDist, minScDist, scSmall, b.astCount);
                } else {
                    std::printf("[BEAM] frame=%05d score=%07d lives=%2d "
                                "wave=%2d fit=%07.0f time=%lld seed=0x%08X\n",
                                roundFrames, b.score,
                                b.lives, b.wave, hFitness[tops[0]], (long long)ts, seed);
                }
                fflush(stdout);
                last = now;
            }
        }

        bool allDead = true;
        for (int k = 0; k < width && allDead; k++) {
            if (!newBeam[k].gameOver) {
                allDead = false;
            }
        }
        if (allDead) {
            break;
        }
    }

    int bestSlot = 0;
    for (int k = 1; k < width; k++) {
        if (newBeam[k].score > newBeam[bestSlot].score) {
            bestSlot = k;
        }
    }
    *outScore = newBeam[bestSlot].score;
    *outFrames = pos;
    memcpy(outTape, tapes[bestSlot].data(), (pos + 1) >> 1);

    CUDA_CALL(cudaFree(beam));
    CUDA_CALL(cudaFree(expanded));
    CUDA_CALL(cudaFree(fitness));
    CUDA_CALL(cudaFree(nibbles));

    cudaEventDestroy(evKernelStart);
    cudaEventDestroy(evKernelStop);

    double avgKernelMs   = kernelRounds > 0 ? totalKernelMs / kernelRounds : 0.0;
    double framesPerSec  = totalKernelMs > 0.0 ? (double)frames / (totalKernelMs / 1000.0) : 0.0;
    int64_t statesPerRound = (int64_t)width * branches;
    double stepsPerSec   = totalKernelMs > 0.0
                           ? (statesPerRound * (double)horizon * kernelRounds) / (totalKernelMs / 1000.0)
                           : 0.0;

    std::printf("[BEAM] frame=%05d score=%07d lives=%2d wave=%2d\n",
                pos, *outScore, newBeam[bestSlot].lives, newBeam[bestSlot].wave);
    std::printf("[ENGINE] kernel_total=%.1fms rounds=%d avg_round=%.2fms\n",
                totalKernelMs, kernelRounds, avgKernelMs);
    std::printf("[ENGINE] frames/s=%.1f  beam_steps/s=%.3eM  (width=%d branches=%d horizon=%d)\n",
                framesPerSec, stepsPerSec / 1e6, width, branches, horizon);
}
#endif
