cmake_minimum_required(VERSION 3.20)
project(squish VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aggressive optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /GL /DNDEBUG /arch:AVX2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
elseif(WIN32)
    # MinGW on Windows: static linking for portable builds
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
else()
    # Linux: dynamic linking (glibc doesnt like static)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
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
