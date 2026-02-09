# squish 1.0.0

Fast bulk image optimizer. Compresses JPEG/PNG in parallel with custom SIMD encoder.

## Downloads

| Platform | Download                         |
|----------|----------------------------------|
| Windows  | `squish-1.0.0-setup.exe` (below) |
| Linux    | See install instructions         |

## Linux Install

```bash
curl -fsSL https://raw.githubusercontent.com/AGDNoob/squish/main/install.sh | bash
```

Requires: `cmake`, `g++`, `make` (script offers to install them)

## What's New

- Initial release
- Custom JPEG encoder with handwritten AVX2 DCT kernel
- Multi-threaded batch processing
- EXIF orientation auto-correction
- Supports JPEG, PNG input/output
- Windows installer with PATH integration
- Linux one-liner installer

## Usage

```bash
squish photos/ -o compressed/
squish *.jpg -q 80
squish photos/ -w 1920 -h 1080
```

## Build from Source

```bash
git clone https://github.com/AGDNoob/squish.git
cd squish
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

---

MIT License
