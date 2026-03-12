ifneq ($(OS),Windows_NT)
    TARGET   = kalien
    CXX     ?= g++
    NVCC     = nvcc -ccbin $(CXX)
    GPU_ARCH  = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    ARCH_FLAG = $(if $(GPU_ARCH),-arch=sm_$(GPU_ARCH),-arch=sm_75)

    CXXFLAGS  = -O3 -DNDEBUG -std=c++17 -Iutils -march=native -flto
    NVCCFLAGS = -O3 -DNDEBUG -std=c++17 -Iutils $(ARCH_FLAG) -Wno-deprecated-gpu-targets

    OBJS = kalien.o kernel.o
    LINKER = $(NVCC)

    .PHONY: all clean

    all: $(TARGET)

    $(TARGET): $(OBJS)
	    $(LINKER) $(NVCCFLAGS) -o $@ $(OBJS) -lnvrtc -lcuda

    kalien.o: kalien.cpp ports/sim.cuh tape.h fitness.h fitness.cuh jit.h
	    $(CXX) $(CXXFLAGS) -I"$(shell dirname $$(which nvcc))/../include" -c $< -o $@

    kernel.o: kernel.cu ports/sim.cuh fitness.h fitness.cuh jit.h
	    $(NVCC) $(NVCCFLAGS) -c $< -o $@

    clean:
	    rm -f $(TARGET) $(OBJS)

else
    TARGET   = kalien.exe
    CXX      = cl
    NVCC     = nvcc

    VS_PATH      = C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.38.33130
    WINSDK_INC   = C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0
    WINSDK_LIB   = C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0
    GPU_INCLUDE  = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include
    GPU_LIB      = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64

    CXXFLAGS  = /O2 /DNDEBUG /EHsc /std:c++17 /Iutils /I"$(VS_PATH)/include" /I"$(WINSDK_INC)/ucrt" /wd4819 /I"$(GPU_INCLUDE)"
    LDFLAGS   = /link /LIBPATH:"$(WINSDK_LIB)/um/x64" /LIBPATH:"$(WINSDK_LIB)/ucrt/x64" /LIBPATH:"$(VS_PATH)/lib/x64" /LIBPATH:"$(GPU_LIB)" cudart.lib nvrtc.lib cuda.lib
    NVCCFLAGS = -ccbin "cl" -I"$(GPU_INCLUDE)" -Xcompiler /wd4819

    OBJS = kalien.obj kernel.obj

    .PHONY: all clean

    all: $(TARGET)

    $(TARGET): $(OBJS)
	    $(CXX) $(OBJS) $(LDFLAGS) /OUT:$(TARGET)

    kalien.obj: kalien.cpp ports/sim.cuh tape.h fitness.h
	    $(CXX) $(CXXFLAGS) /c $< /Fokalien.obj

    kernel.obj: kernel.cu ports/sim.cuh
	    $(NVCC) $(NVCCFLAGS) -c $< -o kernel.obj

    clean:
	    del /Q $(TARGET) $(OBJS)

endif
