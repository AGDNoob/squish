cmake_minimum_required(VERSION 3.20)
project(squish VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Try to enable NASM for handwritten assembly
include(CheckLanguage)
check_language(ASM_NASM)
if(CMAKE_ASM_NASM_COMPILER)
    enable_language(ASM_NASM)
    set(HAVE_NASM TRUE)
    if(WIN32)
        set(CMAKE_ASM_NASM_FLAGS "-f win64")
    else()
        set(CMAKE_ASM_NASM_FLAGS "-f elf64")
    endif()
else()
    set(HAVE_NASM FALSE)
    message(STATUS "NASM not found, using C++ DCT fallback")
endif()

# Aggressive optimization flags
if(MSVC)
    # LEGACY HARDWARE FIX: Removed /arch:AVX2 - was crashing on pre-Haswell CPUs
    # AVX2 functions use __declspec(target) attributes instead for runtime dispatch
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "")
elseif(WIN32)
    # MinGW on Windows: static linking for portable builds
    # LEGACY HARDWARE FIX: Use x86-64-v2 (SSE4.2) baseline + explicit -mno-avx2
    # Even with function attributes, LTO can leak AVX2 into main() - disable globally
    # FAST-MATH FIX: Use safe subset instead of -ffast-math to keep NaN/Inf checks working
    # -ffast-math implies -ffinite-math-only which makes NaN checks UB (dangerous with zero-dimension images)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v2 -msse4.2 -mno-avx2 -funsafe-math-optimizations -fno-math-errno -fno-trapping-math -funroll-loops -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
else()
    # Linux: dynamic linking (glibc doesnt like static)
    # LEGACY HARDWARE FIX: Use x86-64-v2 baseline for portability
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=x86-64-v2 -msse4.2 -funsafe-math-optimizations -fno-math-errno -fno-trapping-math -funroll-loops -DNDEBUG")
endif()

# Find threads
find_package(Threads REQUIRED)

# Main executable
add_executable(squish
    src/main.cpp
    src/image_processor.cpp
    src/thread_pool.cpp
    src/cli.cpp
    lib/fpng.cpp
)

# Add handwritten ASM if NASM is available
if(HAVE_NASM)
    target_sources(squish PRIVATE lib/dct_avx2.asm)
    target_compile_definitions(squish PRIVATE HAVE_DCT_AVX2_ASM=1)
endif()

target_include_directories(squish PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/lib
)

# Enable SSE4.1 for fpng (CXX only)
if(MSVC)
    target_compile_definitions(squish PRIVATE FPNG_NO_SSE=0)
else()
    target_compile_options(squish PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-msse4.1>)
    target_compile_definitions(squish PRIVATE FPNG_NO_SSE=0)
endif()

target_link_libraries(squish PRIVATE Threads::Threads)

# DirectCompute for GPU DCT (Windows only)
if(WIN32)
    target_link_libraries(squish PRIVATE d3d11 d3dcompiler)
endif()

# Enable all warnings (CXX only)
if(MSVC)
    target_compile_options(squish PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(squish PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

# Install target
install(TARGETS squish RUNTIME DESTINATION bin)
