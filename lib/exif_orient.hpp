// exif_orient.hpp - exif orientation parser
// holt nur den orientation tag, rest ist mir egal
// handyfotos sind sonst immer gedreht, nervig
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace exif {

// orientation werte laut spec:
// 1 = normal
// 2 = horizontal gespiegelt
// 3 = 180 grad gedreht
// 4 = vertikal gespiegelt
// 5 = gespiegelt + 270 grad
// 6 = 90 grad rechts (handy hochkant)
// 7 = gespiegelt + 90 grad
// 8 = 270 grad (oder 90 links)

enum class Orientation : uint8_t {
    Normal = 1,
    FlipH = 2,
    Rotate180 = 3,
    FlipV = 4,
    Transpose = 5,   // FlipH + Rotate270
    Rotate90 = 6,
    Transverse = 7,  // FlipH + Rotate90
    Rotate270 = 8
};

// orientation aus memory buffer lesen (kein file io)
// gibt 1 (normal) zurück wenn kein exif gefunden
inline int read_jpeg_orientation_mem(const uint8_t* buf, size_t len) {
    if (len < 12) return 1;
    
    // jpeg soi marker checken
    if (buf[0] != 0xFF || buf[1] != 0xD8) return 1;
    
    // nur erste 64kb durchsuchen, exif is immer am anfang
    if (len > 65536) len = 65536;
    
    // APP1 (EXIF) marker finden
    size_t pos = 2;
    while (pos + 4 < len) {
        if (buf[pos] != 0xFF) {
            pos++;
            continue;
        }
        
        uint8_t marker = buf[pos + 1];
        
        // padding bytes skippen
        if (marker == 0xFF) {
            pos++;
            continue;
        }
        
        // SOS oder image data - stop
        if (marker == 0xDA || marker == 0xD9) break;
        
        // standalone markers
        if (marker == 0xD0 || marker == 0x01) {
            pos += 2;
            continue;
        }
        
        // segment länge
        uint16_t seg_len = (buf[pos + 2] << 8) | buf[pos + 3];
        
        // APP1 marker mit EXIF
        if (marker == 0xE1 && pos + 10 < len) {
            // exif check
            if (memcmp(buf + pos + 4, "Exif\0\0", 6) == 0) {
                // TIFF header parsen
                size_t tiff_start = pos + 10;
                if (tiff_start + 8 > len) return 1;
                
                // byte order
                bool big_endian = (buf[tiff_start] == 'M');
                
                auto read16 = [&](size_t offset) -> uint16_t {
                    if (tiff_start + offset + 2 > len) return 0;
                    const uint8_t* p = buf + tiff_start + offset;
                    return big_endian ? (p[0] << 8) | p[1] : p[0] | (p[1] << 8);
                };
                
                auto read32 = [&](size_t offset) -> uint32_t {
                    if (tiff_start + offset + 4 > len) return 0;
                    const uint8_t* p = buf + tiff_start + offset;
                    return big_endian ? 
                        (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3] :
                        p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
                };
                
                // IFD0 position
                uint32_t ifd_offset = read32(4);
                if (ifd_offset == 0 || tiff_start + ifd_offset + 2 > len) return 1;
                
                // einträge im IFD0
                uint16_t entry_count = read16(ifd_offset);
                
                // orientation tag suchen (0x0112)
                for (int i = 0; i < entry_count; i++) {
                    size_t entry_offset = ifd_offset + 2 + i * 12;
                    if (tiff_start + entry_offset + 12 > len) break;
                    
                    uint16_t tag = read16(entry_offset);
                    if (tag == 0x0112) {  // Orientation tag
                        uint16_t orientation = read16(entry_offset + 8);
                        if (orientation >= 1 && orientation <= 8) {
                            return orientation;
                        }
                        return 1;
                    }
                }
            }
        }
        
        pos += 2 + seg_len;
    }
    
    return 1;  // No orientation found
}

// orientation aus file lesen
// gibt 1 (normal) zurück wenn kein exif
inline int read_jpeg_orientation(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return 1;
    
    // Read first 64KB max - EXIF is always near the start
    uint8_t buf[65536];
    size_t len = fread(buf, 1, sizeof(buf), fp);
    fclose(fp);
    
    return read_jpeg_orientation_mem(buf, len);
}

// Apply orientation transform to pixel data
// Returns true if dimensions were swapped (90/270 rotation)
// For 3-channel RGB only
inline bool apply_orientation(
    std::vector<uint8_t>& pixels,
    int& width, int& height,
    int orientation,
    int channels = 3
) {
    if (orientation == 1 || orientation < 1 || orientation > 8) {
        return false;  // Normal or invalid - no transform needed
    }
    
    size_t pixel_size = channels;
    size_t row_size = width * pixel_size;
    std::vector<uint8_t> temp;
    
    auto get_pixel = [&](int x, int y) -> const uint8_t* {
        return pixels.data() + y * row_size + x * pixel_size;
    };
    
    auto set_pixel = [&](uint8_t* dst, const uint8_t* src) {
        for (int c = 0; c < channels; c++) dst[c] = src[c];
    };
    
    switch (orientation) {
        case 2: {  // Flip horizontal
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width / 2; x++) {
                    int x2 = width - 1 - x;
                    uint8_t* p1 = pixels.data() + y * row_size + x * pixel_size;
                    uint8_t* p2 = pixels.data() + y * row_size + x2 * pixel_size;
                    for (int c = 0; c < channels; c++) {
                        std::swap(p1[c], p2[c]);
                    }
                }
            }
            return false;
        }
        
        case 3: {  // Rotate 180
            size_t size = pixels.size();
            for (size_t i = 0; i < size / 2; i += pixel_size) {
                size_t j = size - pixel_size - i;
                for (int c = 0; c < channels; c++) {
                    std::swap(pixels[i + c], pixels[j + c]);
                }
            }
            return false;
        }
        
        case 4: {  // Flip vertical
            temp.resize(row_size);
            for (int y = 0; y < height / 2; y++) {
                int y2 = height - 1 - y;
                uint8_t* row1 = pixels.data() + y * row_size;
                uint8_t* row2 = pixels.data() + y2 * row_size;
                memcpy(temp.data(), row1, row_size);
                memcpy(row1, row2, row_size);
                memcpy(row2, temp.data(), row_size);
            }
            return false;
        }
        
        case 5: {  // Transpose (flip H + rotate 270)
            temp.resize(height * width * pixel_size);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    set_pixel(temp.data() + x * height * pixel_size + y * pixel_size,
                              get_pixel(x, y));
                }
            }
            pixels = std::move(temp);
            std::swap(width, height);
            return true;
        }
        
        case 6: {  // Rotate 90 CW
            temp.resize(height * width * pixel_size);
            int new_width = height;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int nx = height - 1 - y;
                    int ny = x;
                    set_pixel(temp.data() + ny * new_width * pixel_size + nx * pixel_size,
                              get_pixel(x, y));
                }
            }
            pixels = std::move(temp);
            std::swap(width, height);
            return true;
        }
        
        case 7: {  // Transverse (flip H + rotate 90)
            temp.resize(height * width * pixel_size);
            int new_width = height;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int nx = height - 1 - y;
                    int ny = width - 1 - x;
                    set_pixel(temp.data() + ny * new_width * pixel_size + nx * pixel_size,
                              get_pixel(x, y));
                }
            }
            pixels = std::move(temp);
            std::swap(width, height);
            return true;
        }
        
        case 8: {  // Rotate 270 CW (90 CCW)
            temp.resize(height * width * pixel_size);
            int new_width = height;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int nx = y;
                    int ny = width - 1 - x;
                    set_pixel(temp.data() + ny * new_width * pixel_size + nx * pixel_size,
                              get_pixel(x, y));
                }
            }
            pixels = std::move(temp);
            std::swap(width, height);
            return true;
        }
    }
    
    return false;
}

} // namespace exif
