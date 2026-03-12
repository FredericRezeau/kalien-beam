/*
    This file is part of kalien-beam project.
    Licensed under the MIT License.
    Author: Fred Kyung-jin Rezeau <hello@kyungj.in>
*/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <nvrtc.h>
#include <sstream>
#include <stdexcept>
#include <string>

#define NVRTC_CHECK(call)                                              \
    do {                                                               \
        nvrtcResult _r = (call);                                       \
        if (_r != NVRTC_SUCCESS) {                                     \
            std::fprintf(stderr, "[COMPILER] NVRTC error at %s:%d: %s\n",   \
                         __FILE__, __LINE__, nvrtcGetErrorString(_r)); \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (0)

#ifndef CU_CHECK
#define CU_CHECK(call)                                                     \
    do {                                                                   \
        CUresult _r = (call);                                              \
        if (_r != CUDA_SUCCESS) {                                          \
            const char* str = nullptr;                                     \
            cuGetErrorString(_r, &str);                                    \
            std::fprintf(stderr, "[COMPILER] CUDA error at %s:%d: %s\n", \
                         __FILE__, __LINE__, str ? str : "unknown");       \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)
#endif

static std::string read(const char* path) {
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "[COMPILER] Error: cannot open '%s'\n", path);
        std::exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

struct JitKernel {
    CUmodule module;
    CUfunction beam;
};

static JitKernel compile(const char* fitnessPath, const char* kernelPath, const char* simPath, const char* arch, int device = 0) {
    std::string simSrc = read(simPath);
    std::string fitnessSrc = read(fitnessPath);
    std::string kernelSrc = read(kernelPath);
    for (const char* name : {"__global__ void launchKernel(", "__global__ void initKernel("}) {
        std::string needle(name);
        auto pos = kernelSrc.find(needle);
        if (pos != std::string::npos) {
            kernelSrc.insert(pos, "extern \"C\" ");
        }
    }

    std::string combined =
        "#define __JIT__ 1\n"
        "typedef signed char int8_t;\n"
        "typedef unsigned char uint8_t;\n"
        "typedef short int16_t;\n"
        "typedef unsigned short uint16_t;\n"
        "typedef int int32_t;\n"
        "typedef unsigned int uint32_t;\n"
        "typedef long long int64_t;\n"
        "typedef unsigned long long uint64_t;\n"
        "#define INT8_MIN (-128)\n"
        "#define INT8_MAX (127)\n"
        "#define INT32_MIN (-2147483647 - 1)\n"
        "#define INT32_MAX (2147483647)\n"
        "#define INT64_MIN (-9223372036854775807LL - 1)\n"
        "#define INT64_MAX (9223372036854775807LL)\n"
        "#define FLT_MAX (3.402823466e+38f)\n"
        "#define FLT_MIN (1.175494351e-38f)\n"
        "\n" + simSrc + "\n" + fitnessSrc + "\n" + kernelSrc;

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, combined.c_str(), "jit.cu", 0, nullptr, nullptr));
    std::string archFlag = std::string("--gpu-architecture=") + arch;
    std::string cudaInclude;
    {
        const char* cudaPath = std::getenv("CUDA_PATH");
        if (cudaPath) {
            cudaInclude = std::string(cudaPath) + "/include";
        } else {
            for (const char* candidate : {
                     "/usr/local/cuda/include",
                     "/usr/cuda/include",
                     "/opt/cuda/include",
            }) {
                std::ifstream probe(std::string(candidate) + "/cuda_runtime.h");
                if (probe) {
                    cudaInclude = candidate;
                    break;
                }
            }
        }
    }

    std::string include;
    if (!cudaInclude.empty()) {
        include = std::string("--include-path=") + cudaInclude;
    } else {
        std::fprintf(stderr, "[COMPILER] Warning: CUDA include dir not found. Set CUDA_PATH.\n");
    }

    std::vector<const char*> opts = {
        archFlag.c_str(), "--std=c++17", "-DNDEBUG", "-D__NVRTC__", "-diag-suppress=177", "-diag-suppress=1835",
    };
    if (!include.empty()) {
        opts.push_back(include.c_str());
    }

    nvrtcResult result = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (result != NVRTC_SUCCESS) {
        size_t size = 0;
        nvrtcGetProgramLogSize(prog, &size);
        if (size > 1) {
            std::vector<char> buffer(size);
            nvrtcGetProgramLog(prog, buffer.data());
            std::fprintf(stderr, "[COMPILER] log:\n%s\n", buffer.data());
        }
        std::fprintf(stderr, "[COMPILER] Compilation failed: %s\n", nvrtcGetErrorString(result));
        nvrtcDestroyProgram(&prog);
        std::exit(EXIT_FAILURE);
    }

    size_t size = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &size));
    std::vector<char> ptx(size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);
    cudaSetDevice(device);
    cudaFree(nullptr);
    CU_CHECK(cuInit(0));
    CUcontext context;
    CU_CHECK(cuCtxGetCurrent(&context));
    if (!context) {
        CUdevice cuDev;
        CU_CHECK(cuDeviceGet(&cuDev, device));
        CU_CHECK(cuDevicePrimaryCtxRetain(&context, cuDev));
        CU_CHECK(cuCtxSetCurrent(context));
    }

    JitKernel jit;
    CU_CHECK(cuModuleLoadData(&jit.module, ptx.data()));
    CU_CHECK(cuModuleGetFunction(&jit.beam, jit.module, "launchKernel"));
    std::fflush(stdout);
    return jit;
}

static std::string getArch(int device = 0) {
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, device));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    return "sm_" + std::to_string(major) + std::to_string(minor);
}