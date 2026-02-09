# squish

Bulk image optimizer in C++20. MIT license. By AGDNoob.

Does one thing: takes images, compresses them, writes them out. No GUI,
no config files, no bullshit. Runs on Windows and Linux.

## Why this exists

Every image optimizer I tried was either slow, required a subscription,
or came with 50 dependencies. This one is ~4000 lines of C++, compiles
to a single static binary, and processes 60+ images per second.

## Installation

### Windows

Download `squish-1.0.0-setup.exe` from Releases. Installs to Program Files,
adds itself to PATH. Uninstaller included.

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/AGDNoob/squish/main/install.sh | bash
```

Or build manually:

```bash
git clone https://github.com/AGDNoob/squish
cd squish
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build
```

Needs: g++ (GCC 13+ or Clang 16+), cmake, make. Ubuntu: `apt install build-essential cmake`

### Build from source (Windows)

Same as Linux but use MinGW or MSVC. Tested with GCC 15 via MSYS2.

## Usage

```bash
squish photo.jpg                 # -> ./optimized/photo.jpg
squish photos/                   # entire directory
squish photos/ -o out/           # custom output dir
squish photos/ -q 60             # jpeg quality 1-100 (default: 80)
squish photos/ -w 1920           # max width, keeps aspect ratio
squish photos/ --height 1080     # max height, keeps aspect ratio
squish photos/ -w 1920 -h 1080   # fit into box
squish photos/ --gpu             # GPU acceleration (Windows only)
squish -v photos/                # verbose output
```

## What happens under the hood

1. **Load**: `stb_image` decodes JPEG/PNG/BMP/TGA/GIF into raw RGB
2. **EXIF**: For JPEG, reads orientation tag and rotates pixels (phone photos)
3. **Resize**: `stb_image_resize2` with Mitchell filter if dimensions specified
4. **Encode**: Custom SIMD JPEG encoder or `fpng` for PNG
5. **Write**: Memory-mapped I/O for speed

Each image runs in its own thread. Thread pool uses all available cores.

### Skip logic

If a JPEG is already compressed to <10% of raw pixel size, we just copy it.
Re-encoding a JPEG only makes quality worse. Same for PNG at <50% threshold.

### JPEG encoder

The included `fast_jpeg.hpp` is a custom encoder. Uses:

- AVX2/SSE4 for DCT and color conversion
- Precomputed Huffman tables (no per-image optimization)
- Buffered output (8KB chunks)
- ~2x faster than stb_image_write, quality is identical

There's also `dct_avx2.asm` - a handwritten x86-64 assembly DCT kernel.
~280 lines of AVX2 intrinsics doing the 8x8 block transform. Written from
scratch because compiler-generated SIMD wasn't fast enough.

On Windows there's optional GPU acceleration via DirectCompute (D3D11).
Enable with `--gpu` flag. Uses GPU for DCT on images >= 1 megapixel.
CPU path is already fast, but GPU helps on very large batches.

### PNG encoder

Uses `fpng` by Rich Geldreich. SSE-accelerated deflate, produces valid PNG.
3-5x faster than libpng with identical output.

## Performance

Test: 11,343 mixed images, 4.8 GB total, 6-core CPU

| Metric     | Value          |
| ---------- | -------------- |
| Throughput | 52 MB/s        |
| Speed      | 123 images/sec |
| Time       | 92 seconds     |
| Threads    | 6 (auto)       |

Comparison (estimated):

- ImageMagick: 4-6 hours for same workload
- ffmpeg: 1.5-2 hours
- Most tools: single-threaded, no SIMD

Bottleneck is the compression math itself. I/O is memory-mapped, threading
is embarrassingly parallel, allocations are minimized. Not much left to optimize.

## Source layout

```text
src/
  main.cpp              - entry point, calls CLI::parse/run
  cli.cpp               - argument parsing, thread pool, progress output
  image_processor.cpp   - load/resize/save logic
  thread_pool.cpp       - work queue implementation

include/
  cli.hpp               - CLIConfig struct
  image_processor.hpp   - ImageProcessor class
  thread_pool.hpp       - ThreadPool class

lib/
  stb_image.h           - image decoder (Sean Barrett, public domain)
  stb_image_write.h     - fallback encoder
  stb_image_resize2.h   - resize with filters
  fpng.cpp/hpp          - fast PNG encoder (Rich Geldreich, MIT)
  fast_jpeg.hpp         - custom JPEG encoder
  dct_avx2.asm          - handwritten AVX2 DCT kernel (x86-64 asm)
  exif_orient.hpp       - EXIF orientation parser
  mmap_file.hpp         - memory-mapped file I/O
  gpu_dct.hpp           - DirectCompute DCT (Windows only)
```

Everything in `lib/` except fast_jpeg.hpp, dct_avx2.asm, exif_orient.hpp,
mmap_file.hpp, and gpu_dct.hpp is third-party. All included, no external dependencies.

## Limitations

Things this doesn't do:

- **WebP**: stb doesn't support it, adding libwebp means actual dependencies
- **EXIF preservation**: we read orientation, rotate pixels, strip the rest
- **Animated GIF**: first frame only
- **Lossless JPEG rotation**: we decode and re-encode, some quality loss
- **Content-aware quality**: fixed quality setting, no perceptual analysis

These could be added. PRs welcome if you need them.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

That's it. C++20 required. GCC 13+, Clang 16+, or MSVC 2022.

Debug build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Run tests:

```bash
./test.sh
```

Smoke test for basic functionality. Real test suite: 50,000+ images, works or it doesn't.

## License

MIT. Do whatever you want. Credit appreciated but not required.

Copyright (c) 2026 AGDNoob
