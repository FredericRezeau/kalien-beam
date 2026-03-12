/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __constant__
#define __constant__
#endif
#endif

#include "fitness.h"
#include "jit.h"
#include "ports/sim.cuh"
#include "tape.h"

static inline float __fsqrt_rn(float x) { return sqrtf(x); }

extern "C" void runSearch(int device, uint32_t seed, uint32_t salt, int32_t width, int32_t horizon, int32_t frames, int32_t wave,
    int32_t branches, int32_t* outScore, uint8_t* outTape, int32_t* outFrames, const std::chrono::steady_clock::time_point& start,
    JitKernel* jit, bool trace);

static std::string getPath(const std::string& path, uint32_t salt, int32_t score) {
    std::string stem = path;
    if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".tape") {
        stem = stem.substr(0, stem.size() - 5);
    }
    std::ostringstream oss;
    oss << stem << "_" << salt << "_" << score << ".tape";
    return oss.str();
}

static bool writeTape(const std::string& path, uint32_t seed, int32_t score, const std::vector<uint8_t>& packed, int32_t frames) {
    Tape tape;
    for (int32_t i = 0; i < frames; i++) {
        int b = i >> 1;
        uint8_t n = (i & 1) ? (packed[b] >> 4) : (packed[b] & 0xf);
        tape.add(n);
    }
    return tape.write(path, seed, (uint32_t)score);
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --seed <hex> --out <file> [options]\n\n"
        "Required:\n"
        "  --seed         <hex|dec>  Kalien contract seed\n"
        "  --out          <path>     Output tape base path (suffix _<salt>_<score>.tape added)\n\n"
        "Options:\n"
        "  --fitness      <path>     CUDA source file (.cu) for custom fitness function (JIT NVRTC compiled).\n"
        "  --beam         <n>        Beam width (default: 16384)\n"
        "  --branches     <n>        Branches explored, 1..8 (default: 8)\n"
        "  --horizon      <n>        Lookahead depth in frames (default: 20)\n"
        "  --frames       <n>        Total simulation frames (default: 36000)\n"
        "  --salt         <hex|dec>  Salt value (default: 0)\n"
        "  --wave         <n>        Lurk mode activation threshold (default: 7; 0 = disable)\n"
        "  --iterations   <n>        Number of runs (default: 1)\n"
        "  --device       <n>        GPU device index (default: 0)\n", prog);
}

int main(int argc, char* argv[]) {
    std::printf(
        "KALIEN BEAM\n"
        "░▒▓▶ GPU-POWERED BEAM SEARCH\n"
        "v1.0.0\n"
        "\n");

    uint32_t seed = 0;
    bool hasSeed = false;
    int32_t width = 16384;
    int32_t branches = 8;
    int32_t horizon = 20;
    int32_t frames = 36000;
    int32_t wave = 7;
    uint32_t salt = 0;
    int32_t iterations = 1;
    int32_t device = 0;
    std::string outPath;
    std::string fitnessPath;
    bool trace = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = (uint32_t)std::stoul(argv[++i], nullptr, 0);
            hasSeed = true;
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            outPath = argv[++i];
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fitness") && i + 1 < argc) {
            fitnessPath = argv[++i];
        } else if (!strcmp(argv[i], "--beam") && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--branches") && i + 1 < argc) {
            branches = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--horizon") && i + 1 < argc) {
            horizon = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            frames = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--wave") && i + 1 < argc) {
            wave = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--salt") && i + 1 < argc) {
            salt = (uint32_t)std::stoul(argv[++i], nullptr, 0);
        } else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            trace = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!hasSeed || outPath.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::printf("[ENGINE] Warming up...\n");
    std::printf("[ENGINE] device=%d seed=0x%08X salt=0x%08X\n", device, seed, salt);
    std::printf("[ENGINE] Flight plan -> width=%d branches=%d horizon=%d frames=%d iterations=%d wave=%s\n",
                width, branches, horizon, frames, iterations, wave > 0 ? std::to_string(wave).c_str() : "disabled");
    int len = 0;
    cudaGetDeviceCount(&len);
    for (int d = 0; d < len; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::printf("[WEAPON:%d] %s compute=%d.%d SMs=%d mem=%zuMB memBW=%.0fGB/s clockMHz=%d\n",
            d, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.totalGlobalMem / (1024 * 1024), 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6, prop.clockRate / 1000);
    }
    std::fflush(stdout);

    JitKernel kernel;
    JitKernel* jit = nullptr;
    if (!fitnessPath.empty()) {
        std::string arch = getArch(device);
        std::printf("[MATRIX] Compiling from '%s' (arch=%s)...\n", fitnessPath.c_str(), arch.c_str());
        kernel = compile(fitnessPath.c_str(), "kernel.cu", "ports/sim.cuh", arch.c_str(), device);
        jit = &kernel;
    } else {
        std::printf("[MATRIX] Loading built-in heuristic.\n");
    }
    std::printf("[MATRIX] Online.\n");

    const int32_t bytes = (frames + 1) >> 1;
    std::vector<uint8_t> bestTape(bytes, 0);
    int32_t bestScore = -1;
    int32_t bestFrames = 0;
    uint32_t bestSalt = salt;
    uint32_t curSalt = salt;
    int32_t totalRuns = 0;
    auto start = std::chrono::steady_clock::now();
    for (int32_t iter = 0; iter < iterations; iter++) {
        totalRuns++;
        std::vector<uint8_t> tape(bytes, 0);
        int32_t outScore = 0, outFrames = 0;
        runSearch(device, seed, curSalt, width, horizon, frames, wave, branches, &outScore, tape.data(), &outFrames, start, jit, trace);
        if (outScore > bestScore) {
            bestScore = outScore;
            bestFrames = outFrames;
            bestTape = tape;
            bestSalt = curSalt;
            std::string path = getPath(outPath, bestSalt, bestScore);
            if (!writeTape(path, seed, bestScore, bestTape, bestFrames)) {
                std::fprintf(stderr, "Error: could not write %s\n", path.c_str());
                break;
            }
            std::printf("[ARTIFACT] Recovered -> seed=0x%08X salt=0x%08X score=%07d frames=%05d\n", seed, bestSalt, bestScore, bestFrames);
            std::printf("[ARTIFACT] Archived -> %s\n", path.c_str());
        }
        curSalt++;
    }

    std::printf("[ENGINE] iterations=%d best=%07d salt=0x%08X\n", iterations, bestScore, bestSalt);
    std::printf("[ENGINE] Shutdown.\n");

    if (jit) {
        cuModuleUnload(jit->module);
    }
    return 0;
}
