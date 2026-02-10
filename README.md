# squish

Bulk image optimizer in C++20. MIT license. By AGDNoob.

Does one thing: takes images, compresses them, writes them out. No GUI,
no config files, no bullshit. Runs on Windows and Linux.

## Why this exists

Every image optimizer I tried was either mass-produced enterprise garbage,
required a subscription for basic functionality, or dragged in 50 dependencies
just to resize a JPEG. Unacceptable.

This one is ~4400 lines of C++ (not counting third-party headers), compiles
to a single static binary, and pushes 77-95 MB/s through your disk. It does
the job correctly and then gets out of the way.

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

Needs: g++ (GCC 13+ or Clang 16+), cmake, make, nasm (for ASM DCT). Ubuntu: `apt install build-essential cmake nasm`

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
2. **EXIF**: Reads orientation tag, rotates pixels. Your phone photos come out right-side-up.
3. **Resize**: `stb_image_resize2` with Mitchell filter if dimensions specified
4. **Encode**: Custom SIMD JPEG encoder (AVX2 with scalar fallback) or `fpng` for PNG
5. **Write**: Memory-mapped I/O, atomic writes (`.tmp` + rename, no half-written files)

Each image runs in its own thread. Thread pool uses all available cores.
STB operations are mutex-protected because STB's global state is not thread-safe
(and no, "just don't call it from multiple threads" is not a real solution).

### Skip logic

If a JPEG is already compressed to <10% of raw pixel size, we just copy it.
Re-encoding a JPEG only makes quality worse. Same for PNG at <50% threshold.

### JPEG encoder

The included `fast_jpeg.hpp` is a custom encoder. Uses:

- AVX2/SSE4 for DCT and color conversion (runtime CPUID detection)
- Scalar fallback for CPUs without AVX2 — it'll run on your grandma's Pentium
- Precomputed Huffman tables (no per-image optimization)
- Buffered output (8KB chunks)
- `_mm_malloc` with NULL checks and graceful fallback (no silent corruption on OOM)
- ~2x faster than stb_image_write, quality is identical

There's also `dct_avx2.asm` — ~330 lines of handwritten x86-64 assembly doing
the 8x8 block DCT. Written from scratch because the compiler's auto-vectorizer
produced embarrassing code.

On Windows there's optional GPU acceleration via DirectCompute (D3D11).
Enable with `--gpu` flag. Uses GPU for DCT on images >= 1 megapixel.
CPU path is already fast, but GPU helps on very large batches.

### PNG encoder

Uses `fpng` by Rich Geldreich. SSE-accelerated deflate, produces valid PNG.
3-5x faster than libpng with identical output.

## Performance

Benchmarked with synthetic test images (640px to 5K), GCC 15, 6-core CPU:

**Compress only (Q80)**

| Workload    | Time   | Throughput | Speed      | Savings |
| ----------- | ------ | ---------- | ---------- | ------- |
| 6 images    | 877 ms | 77 MB/s    | 6.8 img/s  | 88.8%   |
| 70 images   | 8.7 s  | 77.7 MB/s  | 8.1 img/s  | 88.8%   |

**Compress + resize (Q80, max 1920px)**

| Workload    | Time   | Throughput | Speed      | Savings |
| ----------- | ------ | ---------- | ---------- | ------- |
| 6 images    | 708 ms | 95.4 MB/s  | 8.5 img/s  | 94.5%   |
| 70 images   | 7.4 s  | 91.6 MB/s  | 9.5 img/s  | 94.5%   |

Comparison (estimated):

- ImageMagick: 10-20x slower for same workload
- ffmpeg: 5-10x slower
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
  fast_resize.hpp       - SIMD image resize
  dct_avx2.asm          - handwritten AVX2 DCT kernel (x86-64 asm)
  exif_orient.hpp       - EXIF orientation parser
  mmap_file.hpp         - memory-mapped file I/O
  gpu_dct.hpp           - DirectCompute DCT (Windows only)
```

Everything in `lib/` except fast_jpeg.hpp, fast_resize.hpp, dct_avx2.asm, exif_orient.hpp,
mmap_file.hpp, and gpu_dct.hpp is third-party. All included, no external dependencies.

## Hardening

This thing is built to not break. If your image optimizer corrupts files or
crashes on bad input, you wrote it wrong.

- **Atomic writes**: Output goes to `.tmp` first, then `rename()`. Power loss
  mid-write won't leave you with half a JPEG.
- **AVX2 safety**: Binary runs on any x86-64 CPU. AVX2 code paths are gated
  behind CPUID checks and `__attribute__((target))`. No illegal instruction traps.
- **Symlink protection**: Directory traversal won't follow symlinks into `/etc`.
  500k file limit to prevent zip-bomb style directory attacks.
- **Path traversal**: Output paths are sanitized. No `../../` nonsense.
- **OOM handling**: Every allocation is checked. `_mm_malloc` failures fall back
  to scalar. `vector::resize` failures get caught and reported, not ignored.
- **Thread safety**: STB's global state is mutex-protected. Because apparently
  "not thread-safe" means most people just cross their fingers.
- **No `-ffast-math`**: Uses `-funsafe-math-optimizations -fno-math-errno
  -fno-trapping-math` instead. NaN propagation works. Your images don't turn green.

## Limitations

Things this doesn't do, and I'm not going to pretend otherwise:

- **WebP**: stb doesn't support it. Adding libwebp means real dependencies.
- **EXIF preservation**: We read orientation, rotate pixels, strip the rest.
  If you need EXIF, use something else.
- **Animated GIF**: First frame only. Animated image "optimization" is a different problem.
- **Lossless JPEG rotation**: We decode and re-encode. Some quality loss. That's how math works.
- **Content-aware quality**: Fixed quality setting. No perceptual analysis.
  You pick the number, you live with the number.

PRs welcome if you actually need these.

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

C++20, GCC 13+, Clang 16+, or MSVC 2022. If your compiler is older, upgrade it.

```bash
./test.sh
```

Smoke test. Verifies basic functionality. If it passes, ship it.

## License

MIT. Do whatever you want. Credit appreciated but not required.

Copyright (c) 2026 AGDNoob
