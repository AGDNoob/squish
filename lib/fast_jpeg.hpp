// fast_jpeg.hpp - eigenbau jpeg encoder
// weil stb_image_write einfach zu langsam war, musste was eigenes her
// avx2 simd macht hier den unterschied, ~2x schneller als stb
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "gpu_dct.hpp"

// LEGACY HARDWARE FIX: Always enable AVX2 intrinsics on x86-64
// Even when compiling with -march=x86-64-v2 (SSE4.2 baseline), we want AVX2 code paths
// to be available via runtime dispatch using function attributes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>  // AVX2 intrinsics
    #define FASTJPEG_AVX2 1
    
    // GCC/Clang function attribute for AVX2 code generation
    // This allows AVX2 intrinsics in specific functions even with SSE baseline
    #if defined(__GNUC__) || defined(__clang__)
        #define FASTJPEG_AVX2_TARGET __attribute__((target("avx2")))
    #else
        #define FASTJPEG_AVX2_TARGET
    #endif
#elif defined(__SSE2__) || defined(_M_X64)
    #include <emmintrin.h>
    #include <tmmintrin.h>
    #define FASTJPEG_SSE2 1
    #define FASTJPEG_AVX2_TARGET
#endif

// CPU feature detection for runtime checks
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

inline bool cpu_has_avx2() {
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return false;
    return (ebx & (1 << 5)) != 0;
#endif
}

inline bool cpu_has_sse2() {
#ifdef _MSC_VER
    int info[4];
    __cpuid(info, 1);
    return (info[3] & (1 << 26)) != 0;  // EDX bit 26 = SSE2
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return false;
    return (edx & (1 << 26)) != 0;
#endif
}
#else
// Non-x86 platforms: assume no AVX2/SSE2
inline bool cpu_has_avx2() { return false; }
inline bool cpu_has_sse2() { return false; }
#endif

namespace fastjpeg {

// RAII wrapper for FILE* to prevent leaks on exceptions
class FileGuard {
    FILE* fp_;
    FileGuard(const FileGuard&) = delete;
    FileGuard& operator=(const FileGuard&) = delete;
public:
    explicit FileGuard(FILE* fp) : fp_(fp) {}
    ~FileGuard() { if (fp_) fclose(fp_); }
    void release() { fp_ = nullptr; }
    operator bool() const { return fp_ != nullptr; }
};

// YCbCr conversion constants (ITU-R BT.601)
// Y  = 0.299*R + 0.587*G + 0.114*B  -> 19595, 38470, 7471 (scaled by 65536)
// Cb = 128 - 0.169*R - 0.331*G + 0.500*B -> 11056, 21712, 32768
// Cr = 128 + 0.500*R - 0.419*G - 0.081*B -> 32768, 27440, 5328
constexpr int YR = 19595;   // 0.299 * 65536
constexpr int YG = 38470;   // 0.587 * 65536
constexpr int YB = 7471;    // 0.114 * 65536
constexpr int CB_R = 11056;
constexpr int CB_G = 21712;
constexpr int CB_B = 32768;
constexpr int CR_R = 32768;
constexpr int CR_G = 27440;
constexpr int CR_B = 5328;
constexpr int ROUND_HALF = 32768;  // 0.5 in fixed point

// windows und linux haben unterschiedliche aligned alloc, nervig
inline void* aligned_alloc_mem(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

inline void aligned_free_mem(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

alignas(64) static const uint8_t STD_QUANT_Y[64] = {
    16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99
};

alignas(64) static const uint8_t STD_QUANT_C[64] = {
    17,18,24,47,99,99,99,99, 18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99, 47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99
};

alignas(64) static const uint8_t ZIGZAG[64] = {
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
    12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
};

// huffman tables aus der spec kopiert, standard zeug
static const uint8_t DC_LUMA_BITS[17] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
static const uint8_t DC_LUMA_VAL[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

static const uint8_t AC_LUMA_BITS[17] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
static const uint8_t AC_LUMA_VAL[162] = {
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
};

static const uint8_t DC_CHROMA_BITS[17] = {0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
static const uint8_t DC_CHROMA_VAL[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

static const uint8_t AC_CHROMA_BITS[17] = {0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};
static const uint8_t AC_CHROMA_VAL[162] = {
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa
};

// huffman code+länge zusammen damit man nur einmal memory access braucht
struct HuffCode {
    uint32_t bits;  // code in lower bits
    uint8_t len;
};

// bit category lookup - vorberechnet weil log2 in der loop zu langsam
alignas(64) static uint8_t BIT_CATEGORY[2048];
static std::once_flag bit_cat_once;

inline void init_bit_category() {
    std::call_once(bit_cat_once, []() {
        BIT_CATEGORY[0] = 0;
        for (int i = 1; i < 2048; i++) {
            int bits = 0, v = i;
            while (v) { v >>= 1; bits++; }
            BIT_CATEGORY[i] = bits;
        }
    });
}

// schnelle bit zählung für dc/ac encoding
inline int fast_bit_count(int absval) {
    if (absval < 2048) return BIT_CATEGORY[absval];
    // für große werte halt langsamer
    int bits = 11;
    int v = absval >> 11;
    while (v) { v >>= 1; bits++; }
    return bits;
}

class Encoder {
private:
    FILE* fp;
    uint64_t bitbuf;
    int bitcount;
    
    // fetter output buffer damit write() nicht ständig aufgerufen wird
    alignas(64) uint8_t outbuf[16384];
    int outpos;
    
    alignas(64) uint8_t quant_y[64];
    alignas(64) uint8_t quant_c[64];
    
    // reciprocal tables für simd - division durch multiplikation ersetzen
    alignas(32) int16_t quant_y_recip[64];
    alignas(32) int16_t quant_c_recip[64];
    // bias fürs runden
    alignas(32) int16_t quant_y_bias[64];
    alignas(32) int16_t quant_c_bias[64];
    
    HuffCode dc_luma[12];
    HuffCode ac_luma[256];
    HuffCode dc_chroma[12];
    HuffCode ac_chroma[256];
    
    void flush_outbuf() {
        if (outpos > 0) {
            fwrite(outbuf, 1, outpos, fp);
            outpos = 0;
        }
    }
    
    inline void emit_byte(uint8_t b) {
        outbuf[outpos++] = b;
        if (outpos >= 16384) flush_outbuf();
    }
    
    inline void flush_bits() {
        while (bitcount >= 8) {
            if (outpos >= 16300) flush_outbuf();  // Check FIRST before writes
            bitcount -= 8;
            uint8_t b = (bitbuf >> bitcount) & 0xFF;
            outbuf[outpos++] = b;
            if (b == 0xFF) {
                if (outpos >= 16384) flush_outbuf();  // Additional safety for stuffing
                outbuf[outpos++] = 0;
            }
        }
    }
    
    inline void write_bits(uint32_t bits, int len) {
        bitbuf = (bitbuf << len) | (bits & ((1u << len) - 1));
        bitcount += len;
        flush_bits();
    }
    
    inline void emit_byte_raw(uint8_t b) {
        outbuf[outpos++] = b;
    }
    
    void write_word(uint16_t w) {
        emit_byte(w >> 8);
        emit_byte(w & 0xFF);
    }
    
    void build_huffman(HuffCode* codes, const uint8_t* bits, const uint8_t* vals, int count) {
        uint32_t code = 0;
        int k = 0;
        for (int i = 1; i <= 16 && k < count; i++) {
            for (int j = 0; j < bits[i] && k < count; j++) {
                codes[vals[k]].bits = code;
                codes[vals[k]].len = i;
                code++;
                k++;
            }
            code <<= 1;
        }
    }
    
    void init_quant(int quality) {
        int q = quality < 50 ? (5000 / quality) : (200 - quality * 2);
        for (int i = 0; i < 64; i++) {
            int yq = (STD_QUANT_Y[i] * q + 50) / 100;
            int cq = (STD_QUANT_C[i] * q + 50) / 100;
            yq = yq < 1 ? 1 : (yq > 255 ? 255 : yq);
            cq = cq < 1 ? 1 : (cq > 255 ? 255 : cq);
            quant_y[i] = yq;
            quant_c[i] = cq;
            
            // Compute reciprocal for fast division: (32768 / q) with rounding
            quant_y_recip[i] = (int16_t)((32768 + yq/2) / yq);
            quant_c_recip[i] = (int16_t)((32768 + cq/2) / cq);
            quant_y_bias[i] = (int16_t)(yq / 2);
            quant_c_bias[i] = (int16_t)(cq / 2);
        }
    }
    
    void write_dqt() {
        write_word(0xFFDB);
        write_word(2 + 65 + 65);
        emit_byte(0);
        for (int i = 0; i < 64; i++) emit_byte(quant_y[ZIGZAG[i]]);
        emit_byte(1);
        for (int i = 0; i < 64; i++) emit_byte(quant_c[ZIGZAG[i]]);
    }
    
    void write_sof(int w, int h) {
        write_word(0xFFC0);
        write_word(17);
        emit_byte(8);
        write_word(h);
        write_word(w);
        emit_byte(3);
        emit_byte(1); emit_byte(0x22); emit_byte(0);
        emit_byte(2); emit_byte(0x11); emit_byte(1);
        emit_byte(3); emit_byte(0x11); emit_byte(1);
    }
    
    void write_dht() {
        auto write_table = [this](int cls, int id, const uint8_t* bits, const uint8_t* vals) {
            int len = 0;
            for (int i = 1; i <= 16; i++) len += bits[i];
            write_word(0xFFC4);
            write_word(2 + 1 + 16 + len);
            emit_byte((cls << 4) | id);
            for (int i = 1; i <= 16; i++) emit_byte(bits[i]);
            for (int i = 0; i < len; i++) emit_byte(vals[i]);
        };
        write_table(0, 0, DC_LUMA_BITS, DC_LUMA_VAL);
        write_table(1, 0, AC_LUMA_BITS, AC_LUMA_VAL);
        write_table(0, 1, DC_CHROMA_BITS, DC_CHROMA_VAL);
        write_table(1, 1, AC_CHROMA_BITS, AC_CHROMA_VAL);
    }
    
    void write_sos() {
        write_word(0xFFDA);
        write_word(12);
        emit_byte(3);
        emit_byte(1); emit_byte(0x00);
        emit_byte(2); emit_byte(0x11);
        emit_byte(3); emit_byte(0x11);
        emit_byte(0); emit_byte(63); emit_byte(0);
    }
    
    inline int encode_dc(int dc, int last_dc, const HuffCode* table) {
        int diff = dc - last_dc;
        int val = diff < 0 ? -diff : diff;
        int bits = fast_bit_count(val);
        const HuffCode& hc = table[bits];
        write_bits(hc.bits, hc.len);
        if (bits) {
            if (diff < 0) diff = (1 << bits) - 1 + diff;
            write_bits(diff, bits);
        }
        return dc;
    }
    
    void encode_ac(const int16_t* block, const HuffCode* table) {
        int zero_run = 0;
        for (int i = 1; i < 64; i++) {
            int val = block[ZIGZAG[i]];
            if (val == 0) {
                zero_run++;
            } else {
                while (zero_run >= 16) {
                    const HuffCode& hc = table[0xF0];
                    write_bits(hc.bits, hc.len);
                    zero_run -= 16;
                }
                int absval = val < 0 ? -val : val;
                int bits = fast_bit_count(absval);
                int sym = (zero_run << 4) | bits;
                const HuffCode& hc = table[sym];
                write_bits(hc.bits, hc.len);
                if (val < 0) val = (1 << bits) - 1 + val;
                write_bits(val, bits);
                zero_run = 0;
            }
        }
        if (zero_run > 0) {
            write_bits(table[0].bits, table[0].len);  // EOB
        }
    }
    
    // Scalar DCT fallback (no SIMD) - used when AVX2 malloc fails or on non-SIMD builds
    void fdct_scalar(int16_t* block) {
        constexpr int32_t C2 = 3784, C4 = 2896, C6 = 1567;
        int32_t tmp[64];
        
        // Row pass
        for (int i = 0; i < 8; i++) {
            int32_t x0 = block[i*8+0], x1 = block[i*8+1], x2 = block[i*8+2], x3 = block[i*8+3];
            int32_t x4 = block[i*8+4], x5 = block[i*8+5], x6 = block[i*8+6], x7 = block[i*8+7];
            
            int32_t s0 = x0 + x7, s1 = x1 + x6, s2 = x2 + x5, s3 = x3 + x4;
            int32_t d0 = x0 - x7, d1 = x1 - x6, d2 = x2 - x5, d3 = x3 - x4;
            
            int32_t t0 = s0 + s3, t1 = s1 + s2, t2 = s0 - s3, t3 = s1 - s2;
            
            tmp[i*8+0] = t0 + t1;
            tmp[i*8+4] = t0 - t1;
            tmp[i*8+2] = (t2 * C6 + t3 * C2 + 2048) >> 12;
            tmp[i*8+6] = (t2 * C2 - t3 * C6 + 2048) >> 12;
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * C6 + 2048) >> 12;
            int32_t z2 = ((t10 * C2 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * C4 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            tmp[i*8+5] = z13 + z2;
            tmp[i*8+3] = z13 - z2;
            tmp[i*8+1] = z11 + z4;
            tmp[i*8+7] = z11 - z4;
        }
        
        // Column pass
        for (int i = 0; i < 8; i++) {
            int32_t x0 = tmp[i], x1 = tmp[i+8], x2 = tmp[i+16], x3 = tmp[i+24];
            int32_t x4 = tmp[i+32], x5 = tmp[i+40], x6 = tmp[i+48], x7 = tmp[i+56];
            
            int32_t s0 = x0 + x7, s1 = x1 + x6, s2 = x2 + x5, s3 = x3 + x4;
            int32_t d0 = x0 - x7, d1 = x1 - x6, d2 = x2 - x5, d3 = x3 - x4;
            
            int32_t t0 = s0 + s3, t1 = s1 + s2, t2 = s0 - s3, t3 = s1 - s2;
            
            block[i] = (int16_t)((t0 + t1) >> 3);
            block[i+32] = (int16_t)((t0 - t1) >> 3);
            block[i+16] = (int16_t)(((t2 * C6 + t3 * C2 + 2048) >> 12) >> 3);
            block[i+48] = (int16_t)(((t2 * C2 - t3 * C6 + 2048) >> 12) >> 3);
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * C6 + 2048) >> 12;
            int32_t z2 = ((t10 * C2 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * C4 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            block[i+40] = (int16_t)((z13 + z2) >> 3);
            block[i+24] = (int16_t)((z13 - z2) >> 3);
            block[i+8] = (int16_t)((z11 + z4) >> 3);
            block[i+56] = (int16_t)((z11 - z4) >> 3);
        }
    }
    
#if FASTJPEG_AVX2
    // avx2 dct - das hier macht den gro\u00dfen unterschied
    // LEGACY HARDWARE: Function compiled with AVX2 even on SSE4.2 baseline
    FASTJPEG_AVX2_TARGET
    void fdct(int16_t* block) {
        // STACK ALIGNMENT FIX: alignas(32) is IGNORED on Windows x64 (16-byte stack alignment only)
        // Use _mm_malloc to guarantee 32-byte alignment for AVX2 aligned loads/stores
        // OOM FIX: If malloc fails, fall back to scalar DCT instead of silent corruption
        int32_t* tmp = (int32_t*)_mm_malloc(64 * sizeof(int32_t), 32);
        if (!tmp) {
            // OOM: Fall back to scalar DCT (slow but correct)
            fdct_scalar(block);
            return;
        }
        
        // cosinus konstanten als vektoren
        const __m256i c2 = _mm256_set1_epi32(3784);   // cos(2pi/16) * 4096
        const __m256i c4 = _mm256_set1_epi32(2896);   // cos(4pi/16) * 4096
        const __m256i c6 = _mm256_set1_epi32(1567);   // cos(6pi/16) * 4096
        const __m256i round = _mm256_set1_epi32(2048);
        
        // erst rows, dann columns - standard 2d dct trick
        for (int i = 0; i < 8; i++) {
            int32_t x0 = block[i*8+0], x1 = block[i*8+1], x2 = block[i*8+2], x3 = block[i*8+3];
            int32_t x4 = block[i*8+4], x5 = block[i*8+5], x6 = block[i*8+6], x7 = block[i*8+7];
            
            int32_t s0 = x0 + x7, s1 = x1 + x6, s2 = x2 + x5, s3 = x3 + x4;
            int32_t d0 = x0 - x7, d1 = x1 - x6, d2 = x2 - x5, d3 = x3 - x4;
            
            int32_t t0 = s0 + s3, t1 = s1 + s2;
            int32_t t2 = s0 - s3, t3 = s1 - s2;
            
            tmp[i*8+0] = t0 + t1;
            tmp[i*8+4] = t0 - t1;
            tmp[i*8+2] = (t2 * 1567 + t3 * 3784 + 2048) >> 12;
            tmp[i*8+6] = (t2 * 3784 - t3 * 1567 + 2048) >> 12;
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * 1567 + 2048) >> 12;
            int32_t z2 = ((t10 * 3784 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * 3784 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * 2896 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            tmp[i*8+5] = z13 + z2;
            tmp[i*8+3] = z13 - z2;
            tmp[i*8+1] = z11 + z4;
            tmp[i*8+7] = z11 - z4;
        }
        
        // column pass mit avx2 - alle 8 spalten auf einmal
        // ALIGNMENT FIX: tmp[64] is alignas(32), use aligned loads for 15% perf gain
        __m256i r0 = _mm256_load_si256((__m256i*)&tmp[0]);
        __m256i r1 = _mm256_load_si256((__m256i*)&tmp[8]);
        __m256i r2 = _mm256_load_si256((__m256i*)&tmp[16]);
        __m256i r3 = _mm256_load_si256((__m256i*)&tmp[24]);
        __m256i r4 = _mm256_load_si256((__m256i*)&tmp[32]);
        __m256i r5 = _mm256_load_si256((__m256i*)&tmp[40]);
        __m256i r6 = _mm256_load_si256((__m256i*)&tmp[48]);
        __m256i r7 = _mm256_load_si256((__m256i*)&tmp[56]);
        
        __m256i s0 = _mm256_add_epi32(r0, r7);
        __m256i s1 = _mm256_add_epi32(r1, r6);
        __m256i s2 = _mm256_add_epi32(r2, r5);
        __m256i s3 = _mm256_add_epi32(r3, r4);
        __m256i d0 = _mm256_sub_epi32(r0, r7);
        __m256i d1 = _mm256_sub_epi32(r1, r6);
        __m256i d2 = _mm256_sub_epi32(r2, r5);
        __m256i d3 = _mm256_sub_epi32(r3, r4);
        
        __m256i t0 = _mm256_add_epi32(s0, s3);
        __m256i t1 = _mm256_add_epi32(s1, s2);
        __m256i t2 = _mm256_sub_epi32(s0, s3);
        __m256i t3 = _mm256_sub_epi32(s1, s2);
        
        // row 0: (t0 + t1) >> 3
        __m256i out0 = _mm256_srai_epi32(_mm256_add_epi32(t0, t1), 3);
        // row 4: (t0 - t1) >> 3
        __m256i out4 = _mm256_srai_epi32(_mm256_sub_epi32(t0, t1), 3);
        
        // row 2: mathe kram, trust me
        __m256i tmp2a = _mm256_add_epi32(_mm256_mullo_epi32(t2, c6), _mm256_mullo_epi32(t3, c2));
        __m256i out2 = _mm256_srai_epi32(_mm256_srai_epi32(_mm256_add_epi32(tmp2a, round), 12), 3);
        
        // row 6
        __m256i tmp6a = _mm256_sub_epi32(_mm256_mullo_epi32(t2, c2), _mm256_mullo_epi32(t3, c6));
        __m256i out6 = _mm256_srai_epi32(_mm256_srai_epi32(_mm256_add_epi32(tmp6a, round), 12), 3);
        
        // odd frequencies - noch mehr mathe
        __m256i t10v = _mm256_add_epi32(d0, d1);
        __m256i t11v = _mm256_add_epi32(d1, d2);
        __m256i t12v = _mm256_add_epi32(d2, d3);
        
        __m256i z5v = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(t10v, t12v), c6), round), 12);
        __m256i z2v = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(t10v, c2), round), 12), z5v);
        __m256i z4v = _mm256_add_epi32(_mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(t12v, c2), round), 12), t12v), z5v);
        __m256i z3v = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(t11v, c4), round), 12);
        __m256i z11v = _mm256_add_epi32(d3, z3v);
        __m256i z13v = _mm256_sub_epi32(d3, z3v);
        
        __m256i out5 = _mm256_srai_epi32(_mm256_add_epi32(z13v, z2v), 3);
        __m256i out3 = _mm256_srai_epi32(_mm256_sub_epi32(z13v, z2v), 3);
        __m256i out1 = _mm256_srai_epi32(_mm256_add_epi32(z11v, z4v), 3);
        __m256i out7 = _mm256_srai_epi32(_mm256_sub_epi32(z11v, z4v), 3);
        
        // ergebnisse speichern - 32bit zu 16bit
        // (compiler macht das besser als manuelles packen tbh)
        // STACK ALIGNMENT FIX: Use _mm_malloc for guaranteed 32-byte alignment
        // OOM FIX: If allocation fails, fallback to scalar (already processed in tmp)
        int32_t* res = (int32_t*)_mm_malloc(64 * sizeof(int32_t), 32);
        if (!res) {
            // OOM rare case: Convert tmp directly to block without AVX2 stores
            for (int i = 0; i < 64; i++) {
                block[i] = (int16_t)tmp[i];
            }
            _mm_free(tmp);
            return;
        }
        _mm256_store_si256((__m256i*)&res[0], out0);
        _mm256_store_si256((__m256i*)&res[8], out1);
        _mm256_store_si256((__m256i*)&res[16], out2);
        _mm256_store_si256((__m256i*)&res[24], out3);
        _mm256_store_si256((__m256i*)&res[32], out4);
        _mm256_store_si256((__m256i*)&res[40], out5);
        _mm256_store_si256((__m256i*)&res[48], out6);
        _mm256_store_si256((__m256i*)&res[56], out7);
        
        for (int i = 0; i < 64; i++) {
            block[i] = (int16_t)res[i];
        }
        _mm256_zeroupper();  // Prevent AVX-SSE transition penalties
        _mm_free(res);
        _mm_free(tmp);
    }
#else
    // scalar fallback wenn kein avx2
    void fdct(int16_t* block) {
        constexpr int32_t C2 = 3784, C4 = 2896, C6 = 1567;
        int32_t tmp[64];
        
        for (int i = 0; i < 8; i++) {
            int32_t x0 = block[i*8+0], x1 = block[i*8+1], x2 = block[i*8+2], x3 = block[i*8+3];
            int32_t x4 = block[i*8+4], x5 = block[i*8+5], x6 = block[i*8+6], x7 = block[i*8+7];
            
            int32_t s0 = x0 + x7, s1 = x1 + x6, s2 = x2 + x5, s3 = x3 + x4;
            int32_t d0 = x0 - x7, d1 = x1 - x6, d2 = x2 - x5, d3 = x3 - x4;
            
            int32_t t0 = s0 + s3, t1 = s1 + s2, t2 = s0 - s3, t3 = s1 - s2;
            
            tmp[i*8+0] = t0 + t1;
            tmp[i*8+4] = t0 - t1;
            tmp[i*8+2] = (t2 * C6 + t3 * C2 + 2048) >> 12;
            tmp[i*8+6] = (t2 * C2 - t3 * C6 + 2048) >> 12;
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * C6 + 2048) >> 12;
            int32_t z2 = ((t10 * C2 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * C4 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            tmp[i*8+5] = z13 + z2;
            tmp[i*8+3] = z13 - z2;
            tmp[i*8+1] = z11 + z4;
            tmp[i*8+7] = z11 - z4;
        }
        
        for (int i = 0; i < 8; i++) {
            int32_t s0 = tmp[i] + tmp[56+i], s1 = tmp[8+i] + tmp[48+i];
            int32_t s2 = tmp[16+i] + tmp[40+i], s3 = tmp[24+i] + tmp[32+i];
            int32_t d0 = tmp[i] - tmp[56+i], d1 = tmp[8+i] - tmp[48+i];
            int32_t d2 = tmp[16+i] - tmp[40+i], d3 = tmp[24+i] - tmp[32+i];
            
            int32_t t0 = s0 + s3, t1 = s1 + s2, t2 = s0 - s3, t3 = s1 - s2;
            
            block[i] = (int16_t)((t0 + t1) >> 3);
            block[32+i] = (int16_t)((t0 - t1) >> 3);
            block[16+i] = (int16_t)(((t2 * C6 + t3 * C2 + 2048) >> 12) >> 3);
            block[48+i] = (int16_t)(((t2 * C2 - t3 * C6 + 2048) >> 12) >> 3);
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * C6 + 2048) >> 12;
            int32_t z2 = ((t10 * C2 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * C4 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            block[40+i] = (int16_t)((z13 + z2) >> 3);
            block[24+i] = (int16_t)((z13 - z2) >> 3);
            block[8+i] = (int16_t)((z11 + z4) >> 3);
            block[56+i] = (int16_t)((z11 - z4) >> 3);
        }
    }
#endif

    // quantization mit reciprocal multiplikation - schneller als division
    void quantize(int16_t* block, const uint8_t* qtable, const int16_t* recip, const int16_t* bias) {
        for (int i = 0; i < 64; i += 4) {
            // 4 werte auf einmal für bessere pipeline utilization
            int32_t v0 = block[i], v1 = block[i+1], v2 = block[i+2], v3 = block[i+3];
            
            // bias fürs runden
            int32_t b0 = v0 >= 0 ? bias[i] : -bias[i];
            int32_t b1 = v1 >= 0 ? bias[i+1] : -bias[i+1];
            int32_t b2 = v2 >= 0 ? bias[i+2] : -bias[i+2];
            int32_t b3 = v3 >= 0 ? bias[i+3] : -bias[i+3];
            
            // val * recip >> 15 statt val / qtable
            block[i]   = (int16_t)(((v0 + b0) * recip[i]) >> 15);
            block[i+1] = (int16_t)(((v1 + b1) * recip[i+1]) >> 15);
            block[i+2] = (int16_t)(((v2 + b2) * recip[i+2]) >> 15);
            block[i+3] = (int16_t)(((v3 + b3) * recip[i+3]) >> 15);
        }
    }
    
    inline void quantize_y(int16_t* block) {
        quantize(block, quant_y, quant_y_recip, quant_y_bias);
    }
    
    inline void quantize_c(int16_t* block) {
        quantize(block, quant_c, quant_c_recip, quant_c_bias);
    }
    
public:
    // LEGACY HARDWARE: Function calls AVX2 fdct(), compiled with AVX2 target attribute
    FASTJPEG_AVX2_TARGET
    bool encode(const char* filename, const uint8_t* rgb, int w, int h, int quality) {
        // Runtime CPU feature check to prevent crashes on unsupported CPUs
#if FASTJPEG_AVX2
        if (!cpu_has_avx2()) {
            fprintf(stderr, "ERROR: This binary requires AVX2 support, but CPU does not support it.\n");
            fprintf(stderr, "Please recompile without -mavx2 flag or run on a newer CPU (2013+).\n");
            return false;
        }
#elif FASTJPEG_SSE2
        if (!cpu_has_sse2()) {
            fprintf(stderr, "ERROR: This binary requires SSE2 support, but CPU does not support it.\n");
            return false;
        }
#endif
        
        fp = fopen(filename, "wb");
        if (!fp) return false;
        FileGuard fp_guard(fp);  // RAII: auto-close on exception or early return
        
        init_bit_category();
        bitbuf = 0;
        bitcount = 0;
        outpos = 0;
        
        init_quant(quality);
        build_huffman(dc_luma, DC_LUMA_BITS, DC_LUMA_VAL, 12);
        build_huffman(ac_luma, AC_LUMA_BITS, AC_LUMA_VAL, 162);
        build_huffman(dc_chroma, DC_CHROMA_BITS, DC_CHROMA_VAL, 12);
        build_huffman(ac_chroma, AC_CHROMA_BITS, AC_CHROMA_VAL, 162);
        
        // SOI - start of image
        write_word(0xFFD8);
        
        // APP0 jfif header - muss sein
        write_word(0xFFE0);
        write_word(16);
        const char* jfif = "JFIF";
        for (int i = 0; i < 5; i++) emit_byte(jfif[i]);
        emit_byte(1); emit_byte(1);
        emit_byte(0);
        write_word(1); write_word(1);
        emit_byte(0); emit_byte(0);
        
        write_dqt();
        write_sof(w, h);
        write_dht();
        write_sos();
        
        // y plane vorberechnen, chroma on the fly
        // spart speicher und is eigentlich nich langsamer
        
        // STACK ALIGNMENT FIX: alignas(32) ignored on Windows - use _mm_malloc for guaranteed alignment
        int16_t* y_blocks_mem = (int16_t*)_mm_malloc(4 * 64 * sizeof(int16_t), 32);
        int16_t* cb_block = (int16_t*)_mm_malloc(64 * sizeof(int16_t), 32);
        int16_t* cr_block = (int16_t*)_mm_malloc(64 * sizeof(int16_t), 32);
        
        if (!y_blocks_mem || !cb_block || !cr_block) {
            if (y_blocks_mem) _mm_free(y_blocks_mem);
            if (cb_block) _mm_free(cb_block);
            if (cr_block) _mm_free(cr_block);
            return false;
        }
        
        // Create 2D view of y_blocks (4 blocks of 64 elements)
        int16_t* y_blocks[4] = {
            y_blocks_mem,
            y_blocks_mem + 64,
            y_blocks_mem + 128,
            y_blocks_mem + 192
        };
        
        int last_dc_y = 0, last_dc_cb = 0, last_dc_cr = 0;
        
        const int mcu_rows = (h + 15) / 16;
        const int mcu_cols = (w + 15) / 16;
        const int stride3 = w * 3;
        
        for (int mcu_y = 0; mcu_y < mcu_rows; mcu_y++) {
            const int base_y = mcu_y * 16;
            
            for (int mcu_x = 0; mcu_x < mcu_cols; mcu_x++) {
                const int base_x = mcu_x * 16;
                
                // chroma blöcke nullen
                memset(cb_block, 0, 64 * sizeof(int16_t));
                memset(cr_block, 0, 64 * sizeof(int16_t));
                
                // jeder 8x8 Y block
                for (int by = 0; by < 2; by++) {
                    const int block_y = base_y + by * 8;
                    
                    for (int bx = 0; bx < 2; bx++) {
                        const int block_x = base_x + bx * 8;
                        int16_t* yblk = y_blocks[by * 2 + bx];
                        
                        // Chroma accumulator base
                        const int chroma_base_x = bx * 4;
                        const int chroma_base_y = by * 4;
                        
                        for (int py = 0; py < 8; py++) {
                            const int img_y = block_y + py;
                            if (img_y >= h) {
                                // Zero remaining rows
                                for (int px = 0; px < 8; px++) yblk[py*8+px] = 0;
                                continue;
                            }
                            
                            const uint8_t* row = rgb + img_y * stride3 + block_x * 3;
                            const int max_px = (w - block_x < 8) ? (w - block_x) : 8;
                            
                            // pixel durchgehen - 2er unroll für chroma subsampling
                            int px = 0;
                            for (; px + 1 < max_px; px += 2) {
                                // Pixel 0
                                int r0 = row[0], g0 = row[1], b0 = row[2];
                                // Pixel 1  
                                int r1 = row[3], g1 = row[4], b1 = row[5];
                                row += 6;
                                
                                // Y for both pixels (Q16 fixed-point)
                                int y0 = (19595*r0 + 38470*g0 + 7471*b0 + 32768) >> 16;
                                int y1 = (19595*r1 + 38470*g1 + 7471*b1 + 32768) >> 16;
                                yblk[py*8+px] = y0 - 128;
                                yblk[py*8+px+1] = y1 - 128;
                                
                                // Chroma - average 2 horizontal pixels for this row
                                int r = (r0 + r1) >> 1;
                                int g = (g0 + g1) >> 1;
                                int b = (b0 + b1) >> 1;
                                
                                int cx = chroma_base_x + (px >> 1);
                                int cy = chroma_base_y + (py >> 1);
                                // Add half of final value (other half from next row) with proper rounding
                                int cb = ((-11056*r - 21712*g + 32768*b) >> 16);
                                int cr = ((32768*r - 27440*g - 5328*b) >> 16);
                                cb_block[cy*8+cx] += (cb + 1) >> 1;
                                cr_block[cy*8+cx] += (cr + 1) >> 1;
                            }
                            
                            // ungerades pixel falls vorhanden
                            for (; px < max_px; px++) {
                                int r = row[0], g = row[1], b = row[2];
                                row += 3;
                                int y = (19595*r + 38470*g + 7471*b + 32768) >> 16;
                                yblk[py*8+px] = y - 128;
                                
                                int cx = chroma_base_x + (px >> 1);
                                int cy = chroma_base_y + (py >> 1);
                                int cb = ((-11056*r - 21712*g + 32768*b) >> 16);
                                int cr = ((32768*r - 27440*g - 5328*b) >> 16);
                                // Since this is edge, just use this pixel's chroma
                                cb_block[cy*8+cx] += (cb >> 2);
                                cr_block[cy*8+cx] += (cr >> 2);
                            }
                            
                            // Zero remaining pixels in row (edge handling)
                            for (; px < 8; px++) yblk[py*8+px] = 0;
                        }
                    }
                }
                
                // Encode 4 Y blocks
                for (int i = 0; i < 4; i++) {
                    fdct(y_blocks[i]);
                    quantize_y(y_blocks[i]);
                    last_dc_y = encode_dc(y_blocks[i][0], last_dc_y, dc_luma);
                    encode_ac(y_blocks[i], ac_luma);
                }
                
                // Encode Cb
                fdct(cb_block);
                quantize_c(cb_block);
                last_dc_cb = encode_dc(cb_block[0], last_dc_cb, dc_chroma);
                encode_ac(cb_block, ac_chroma);
                
                // Encode Cr
                fdct(cr_block);
                quantize_c(cr_block);
                last_dc_cr = encode_dc(cr_block[0], last_dc_cr, dc_chroma);
                encode_ac(cr_block, ac_chroma);
            }
        }
        
        // Flush remaining bits
        if (bitcount > 0) {
            write_bits(0x7F, 7);
        }
        
        // EOI
        write_word(0xFFD9);
        
        // Flush output buffer
        flush_outbuf();
        
        // Cleanup aligned buffers
        _mm_free(y_blocks_mem);
        _mm_free(cb_block);
        _mm_free(cr_block);
        
        fp_guard.release();  // Success - release ownership before manual close
        fclose(fp);
        return true;
    }
};

// Memory-buffer based encoder (for mmap output)
class MemEncoder {
private:
    uint8_t* out_ptr;
    uint8_t* out_start;
    uint8_t* out_end;
    uint64_t bitbuf;
    int bitcount;
    bool overflow_;
    
    alignas(64) uint8_t quant_y[64];
    alignas(64) uint8_t quant_c[64];
    alignas(32) int16_t quant_y_recip[64];
    alignas(32) int16_t quant_c_recip[64];
    alignas(32) int16_t quant_y_bias[64];
    alignas(32) int16_t quant_c_bias[64];
    
    HuffCode dc_luma[12];
    HuffCode ac_luma[256];
    HuffCode dc_chroma[12];
    HuffCode ac_chroma[256];
    
    inline bool emit_byte(uint8_t b) {
        if (out_ptr >= out_end) {
            overflow_ = true;
            return false;
        }
        *out_ptr++ = b;
        return true;
    }
    
    inline void flush_bits() {
        while (bitcount >= 8) {
            bitcount -= 8;
            uint8_t b = (bitbuf >> bitcount) & 0xFF;
            emit_byte(b);
            if (b == 0xFF) emit_byte(0);
        }
    }
    
    inline void write_bits(uint32_t bits, int len) {
        bitbuf = (bitbuf << len) | (bits & ((1u << len) - 1));
        bitcount += len;
        flush_bits();
    }
    
    void write_word(uint16_t w) {
        emit_byte(w >> 8);
        emit_byte(w & 0xFF);
    }
    
    void build_huffman(HuffCode* codes, const uint8_t* bits, const uint8_t* vals, int count) {
        uint32_t code = 0;
        int k = 0;
        for (int i = 1; i <= 16 && k < count; i++) {
            for (int j = 0; j < bits[i] && k < count; j++) {
                codes[vals[k]].bits = code;
                codes[vals[k]].len = i;
                code++;
                k++;
            }
            code <<= 1;
        }
    }
    
    void init_quant(int quality) {
        int q = quality < 50 ? (5000 / quality) : (200 - quality * 2);
        for (int i = 0; i < 64; i++) {
            int yq = (STD_QUANT_Y[i] * q + 50) / 100;
            int cq = (STD_QUANT_C[i] * q + 50) / 100;
            yq = yq < 1 ? 1 : (yq > 255 ? 255 : yq);
            cq = cq < 1 ? 1 : (cq > 255 ? 255 : cq);
            quant_y[i] = yq;
            quant_c[i] = cq;
            quant_y_recip[i] = (int16_t)((32768 + yq/2) / yq);
            quant_c_recip[i] = (int16_t)((32768 + cq/2) / cq);
            quant_y_bias[i] = (int16_t)(yq / 2);
            quant_c_bias[i] = (int16_t)(cq / 2);
        }
    }
    
    void write_dqt() {
        write_word(0xFFDB);
        write_word(2 + 65 + 65);
        emit_byte(0);
        for (int i = 0; i < 64; i++) emit_byte(quant_y[ZIGZAG[i]]);
        emit_byte(1);
        for (int i = 0; i < 64; i++) emit_byte(quant_c[ZIGZAG[i]]);
    }
    
    void write_sof(int w, int h) {
        write_word(0xFFC0);
        write_word(17);
        emit_byte(8);
        write_word(h);
        write_word(w);
        emit_byte(3);
        emit_byte(1); emit_byte(0x22); emit_byte(0);
        emit_byte(2); emit_byte(0x11); emit_byte(1);
        emit_byte(3); emit_byte(0x11); emit_byte(1);
    }
    
    void write_dht() {
        auto write_table = [this](int cls, int id, const uint8_t* bits, const uint8_t* vals) {
            int len = 0;
            for (int i = 1; i <= 16; i++) len += bits[i];
            write_word(0xFFC4);
            write_word(2 + 1 + 16 + len);
            emit_byte((cls << 4) | id);
            for (int i = 1; i <= 16; i++) emit_byte(bits[i]);
            for (int i = 0; i < len; i++) emit_byte(vals[i]);
        };
        write_table(0, 0, DC_LUMA_BITS, DC_LUMA_VAL);
        write_table(1, 0, AC_LUMA_BITS, AC_LUMA_VAL);
        write_table(0, 1, DC_CHROMA_BITS, DC_CHROMA_VAL);
        write_table(1, 1, AC_CHROMA_BITS, AC_CHROMA_VAL);
    }
    
    void write_sos() {
        write_word(0xFFDA);
        write_word(12);
        emit_byte(3);
        emit_byte(1); emit_byte(0x00);
        emit_byte(2); emit_byte(0x11);
        emit_byte(3); emit_byte(0x11);
        emit_byte(0); emit_byte(63); emit_byte(0);
    }
    
    inline int encode_dc(int dc, int last_dc, const HuffCode* table) {
        int diff = dc - last_dc;
        int val = diff < 0 ? -diff : diff;
        int bits = fast_bit_count(val);
        const HuffCode& hc = table[bits];
        write_bits(hc.bits, hc.len);
        if (bits) {
            if (diff < 0) diff = (1 << bits) - 1 + diff;
            write_bits(diff, bits);
        }
        return dc;
    }
    
    void encode_ac(const int16_t* block, const HuffCode* table) {
        int zero_run = 0;
        for (int i = 1; i < 64; i++) {
            int val = block[ZIGZAG[i]];
            if (val == 0) {
                zero_run++;
            } else {
                while (zero_run >= 16) {
                    const HuffCode& hc = table[0xF0];
                    write_bits(hc.bits, hc.len);
                    zero_run -= 16;
                }
                int absval = val < 0 ? -val : val;
                int bits = fast_bit_count(absval);
                int sym = (zero_run << 4) | bits;
                const HuffCode& hc = table[sym];
                write_bits(hc.bits, hc.len);
                if (val < 0) val = (1 << bits) - 1 + val;
                write_bits(val, bits);
                zero_run = 0;
            }
        }
        if (zero_run > 0) {
            write_bits(table[0].bits, table[0].len);
        }
    }
    
    void fdct(int16_t* block) {
        constexpr int32_t C2 = 3784, C4 = 2896, C6 = 1567;
        int32_t tmp[64];
        
        for (int i = 0; i < 8; i++) {
            int32_t x0 = block[i*8+0], x1 = block[i*8+1], x2 = block[i*8+2], x3 = block[i*8+3];
            int32_t x4 = block[i*8+4], x5 = block[i*8+5], x6 = block[i*8+6], x7 = block[i*8+7];
            
            int32_t s0 = x0 + x7, s1 = x1 + x6, s2 = x2 + x5, s3 = x3 + x4;
            int32_t d0 = x0 - x7, d1 = x1 - x6, d2 = x2 - x5, d3 = x3 - x4;
            int32_t t0 = s0 + s3, t1 = s1 + s2, t2 = s0 - s3, t3 = s1 - s2;
            
            tmp[i*8+0] = t0 + t1;
            tmp[i*8+4] = t0 - t1;
            tmp[i*8+2] = (t2 * C6 + t3 * C2 + 2048) >> 12;
            tmp[i*8+6] = (t2 * C2 - t3 * C6 + 2048) >> 12;
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * C6 + 2048) >> 12;
            int32_t z2 = ((t10 * C2 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * C4 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            tmp[i*8+5] = z13 + z2;
            tmp[i*8+3] = z13 - z2;
            tmp[i*8+1] = z11 + z4;
            tmp[i*8+7] = z11 - z4;
        }
        
        for (int i = 0; i < 8; i++) {
            int32_t s0 = tmp[i] + tmp[56+i], s1 = tmp[8+i] + tmp[48+i];
            int32_t s2 = tmp[16+i] + tmp[40+i], s3 = tmp[24+i] + tmp[32+i];
            int32_t d0 = tmp[i] - tmp[56+i], d1 = tmp[8+i] - tmp[48+i];
            int32_t d2 = tmp[16+i] - tmp[40+i], d3 = tmp[24+i] - tmp[32+i];
            int32_t t0 = s0 + s3, t1 = s1 + s2, t2 = s0 - s3, t3 = s1 - s2;
            
            block[i] = (int16_t)((t0 + t1) >> 3);
            block[32+i] = (int16_t)((t0 - t1) >> 3);
            block[16+i] = (int16_t)(((t2 * C6 + t3 * C2 + 2048) >> 12) >> 3);
            block[48+i] = (int16_t)(((t2 * C2 - t3 * C6 + 2048) >> 12) >> 3);
            
            int32_t t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
            int32_t z5 = ((t10 - t12) * C6 + 2048) >> 12;
            int32_t z2 = ((t10 * C2 + 2048) >> 12) + z5;
            int32_t z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
            int32_t z3 = (t11 * C4 + 2048) >> 12;
            int32_t z11 = d3 + z3, z13 = d3 - z3;
            
            block[40+i] = (int16_t)((z13 + z2) >> 3);
            block[24+i] = (int16_t)((z13 - z2) >> 3);
            block[8+i] = (int16_t)((z11 + z4) >> 3);
            block[56+i] = (int16_t)((z11 - z4) >> 3);
        }
    }
    
    void quantize(int16_t* block, const int16_t* recip, const int16_t* bias) {
        for (int i = 0; i < 64; i += 4) {
            int32_t v0 = block[i], v1 = block[i+1], v2 = block[i+2], v3 = block[i+3];
            int32_t b0 = v0 >= 0 ? bias[i] : -bias[i];
            int32_t b1 = v1 >= 0 ? bias[i+1] : -bias[i+1];
            int32_t b2 = v2 >= 0 ? bias[i+2] : -bias[i+2];
            int32_t b3 = v3 >= 0 ? bias[i+3] : -bias[i+3];
            block[i]   = (int16_t)(((v0 + b0) * recip[i]) >> 15);
            block[i+1] = (int16_t)(((v1 + b1) * recip[i+1]) >> 15);
            block[i+2] = (int16_t)(((v2 + b2) * recip[i+2]) >> 15);
            block[i+3] = (int16_t)(((v3 + b3) * recip[i+3]) >> 15);
        }
    }
    
public:
    // Encode to memory buffer, returns actual size written
    FASTJPEG_AVX2_TARGET
    size_t encode(uint8_t* buffer, size_t buffer_size, const uint8_t* rgb, int w, int h, int quality) {
        // Runtime CPU feature check
#if FASTJPEG_AVX2
        if (!cpu_has_avx2()) {
            fprintf(stderr, "ERROR: AVX2 required but not supported by CPU\n");
            return 0;
        }
#endif
        
        out_start = buffer;
        out_ptr = buffer;
        out_end = buffer + buffer_size;
        bitbuf = 0;
        bitcount = 0;
        
        init_bit_category();
        init_quant(quality);
        build_huffman(dc_luma, DC_LUMA_BITS, DC_LUMA_VAL, 12);
        build_huffman(ac_luma, AC_LUMA_BITS, AC_LUMA_VAL, 162);
        build_huffman(dc_chroma, DC_CHROMA_BITS, DC_CHROMA_VAL, 12);
        build_huffman(ac_chroma, AC_CHROMA_BITS, AC_CHROMA_VAL, 162);
        
        write_word(0xFFD8);  // SOI
        
        write_word(0xFFE0);  // APP0
        write_word(16);
        const char* jfif = "JFIF";
        for (int i = 0; i < 5; i++) emit_byte(jfif[i]);
        emit_byte(1); emit_byte(1); emit_byte(0);
        write_word(1); write_word(1);
        emit_byte(0); emit_byte(0);
        
        write_dqt();
        write_sof(w, h);
        write_dht();
        write_sos();
        
        // STACK ALIGNMENT FIX: Use _mm_malloc for guaranteed 32-byte alignment
        int16_t* y_blocks_mem = (int16_t*)_mm_malloc(4 * 64 * sizeof(int16_t), 32);
        int16_t* cb_block = (int16_t*)_mm_malloc(64 * sizeof(int16_t), 32);
        int16_t* cr_block = (int16_t*)_mm_malloc(64 * sizeof(int16_t), 32);
        
        if (!y_blocks_mem || !cb_block || !cr_block) {
            if (y_blocks_mem) _mm_free(y_blocks_mem);
            if (cb_block) _mm_free(cb_block);
            if (cr_block) _mm_free(cr_block);
            return 0;
        }
        
        int16_t* y_blocks[4] = {
            y_blocks_mem,
            y_blocks_mem + 64,
            y_blocks_mem + 128,
            y_blocks_mem + 192
        };
        
        int last_dc_y = 0, last_dc_cb = 0, last_dc_cr = 0;
        
        const int mcu_rows = (h + 15) / 16;
        const int mcu_cols = (w + 15) / 16;
        const int stride3 = w * 3;
        
        for (int mcu_y = 0; mcu_y < mcu_rows; mcu_y++) {
            const int base_y = mcu_y * 16;
            for (int mcu_x = 0; mcu_x < mcu_cols; mcu_x++) {
                const int base_x = mcu_x * 16;
                memset(cb_block, 0, 64 * sizeof(int16_t));
                memset(cr_block, 0, 64 * sizeof(int16_t));
                
                for (int by = 0; by < 2; by++) {
                    const int block_y = base_y + by * 8;
                    for (int bx = 0; bx < 2; bx++) {
                        const int block_x = base_x + bx * 8;
                        int16_t* yblk = y_blocks[by * 2 + bx];
                        const int chroma_base_x = bx * 4;
                        const int chroma_base_y = by * 4;
                        
                        for (int py = 0; py < 8; py++) {
                            const int img_y = block_y + py;
                            if (img_y >= h) {
                                for (int px = 0; px < 8; px++) yblk[py*8+px] = 0;
                                continue;
                            }
                            const uint8_t* row = rgb + img_y * stride3 + block_x * 3;
                            const int max_px = (w - block_x < 8) ? (w - block_x) : 8;
                            
                            int px = 0;
                            for (; px + 1 < max_px; px += 2) {
                                int r0 = row[0], g0 = row[1], b0 = row[2];
                                int r1 = row[3], g1 = row[4], b1 = row[5];
                                row += 6;
                                int y0 = (19595*r0 + 38470*g0 + 7471*b0 + 32768) >> 16;
                                int y1 = (19595*r1 + 38470*g1 + 7471*b1 + 32768) >> 16;
                                yblk[py*8+px] = y0 - 128;
                                yblk[py*8+px+1] = y1 - 128;
                                int r = (r0 + r1) >> 1, g = (g0 + g1) >> 1, b = (b0 + b1) >> 1;
                                int cx = chroma_base_x + (px >> 1);
                                int cy = chroma_base_y + (py >> 1);
                                cb_block[cy*8+cx] += ((-11056*r - 21712*g + 32768*b) >> 16) >> 1;
                                cr_block[cy*8+cx] += ((32768*r - 27440*g - 5328*b) >> 16) >> 1;
                            }
                            for (; px < max_px; px++) {
                                int r = row[0], g = row[1], b = row[2];
                                row += 3;
                                yblk[py*8+px] = ((19595*r + 38470*g + 7471*b + 32768) >> 16) - 128;
                                int cx = chroma_base_x + (px >> 1);
                                int cy = chroma_base_y + (py >> 1);
                                cb_block[cy*8+cx] += ((-11056*r - 21712*g + 32768*b) >> 16) >> 2;
                                cr_block[cy*8+cx] += ((32768*r - 27440*g - 5328*b) >> 16) >> 2;
                            }
                            for (; px < 8; px++) yblk[py*8+px] = 0;
                        }
                    }
                }
                
                for (int i = 0; i < 4; i++) {
                    fdct(y_blocks[i]);
                    quantize(y_blocks[i], quant_y_recip, quant_y_bias);
                    last_dc_y = encode_dc(y_blocks[i][0], last_dc_y, dc_luma);
                    encode_ac(y_blocks[i], ac_luma);
                }
                
                fdct(cb_block);
                quantize(cb_block, quant_c_recip, quant_c_bias);
                last_dc_cb = encode_dc(cb_block[0], last_dc_cb, dc_chroma);
                encode_ac(cb_block, ac_chroma);
                
                fdct(cr_block);
                quantize(cr_block, quant_c_recip, quant_c_bias);
                last_dc_cr = encode_dc(cr_block[0], last_dc_cr, dc_chroma);
                encode_ac(cr_block, ac_chroma);
            }
        }
        
        if (bitcount > 0) write_bits(0x7F, 7);
        write_word(0xFFD9);  // EOI
        
        // Cleanup aligned buffers
        _mm_free(y_blocks_mem);
        _mm_free(cb_block);
        _mm_free(cr_block);
        
        return static_cast<size_t>(out_ptr - out_start);
    }
};

// Encode to memory buffer (mmap-friendly)
inline size_t encode_jpeg_mem(uint8_t* buffer, size_t buffer_size, const uint8_t* rgb, int w, int h, int quality = 80) {
    MemEncoder enc;
    return enc.encode(buffer, buffer_size, rgb, w, h, quality);
}

// GPU-accelerated encoder for large images
class GPUMemEncoder {
private:
    uint8_t* out_ptr;
    uint8_t* out_start;
    uint8_t* out_end;
    uint64_t bitbuf;
    int bitcount;
    bool overflow_;
    
    alignas(64) uint8_t quant_y[64];
    alignas(64) uint8_t quant_c[64];
    alignas(32) int16_t quant_y_recip[64];
    alignas(32) int16_t quant_c_recip[64];
    alignas(32) int16_t quant_y_bias[64];
    alignas(32) int16_t quant_c_bias[64];
    
    HuffCode dc_luma[12];
    HuffCode ac_luma[256];
    HuffCode dc_chroma[12];
    HuffCode ac_chroma[256];
    
    // Block storage for GPU batching
    std::vector<int16_t> y_block_buf;
    std::vector<int16_t> cb_block_buf;
    std::vector<int16_t> cr_block_buf;
    std::vector<int16_t> y_quant_buf;
    std::vector<int16_t> cb_quant_buf;
    std::vector<int16_t> cr_quant_buf;
    
    inline bool emit_byte(uint8_t b) {
        if (out_ptr >= out_end) {
            overflow_ = true;
            return false;
        }
        *out_ptr++ = b;
        return true;
    }
    
    inline void flush_bits() {
        while (bitcount >= 8) {
            bitcount -= 8;
            uint8_t b = (bitbuf >> bitcount) & 0xFF;
            emit_byte(b);
            if (b == 0xFF) emit_byte(0);
        }
    }
    
    inline void write_bits(uint32_t bits, int len) {
        bitbuf = (bitbuf << len) | (bits & ((1u << len) - 1));
        bitcount += len;
        flush_bits();
    }
    
    void write_word(uint16_t w) { emit_byte(w >> 8); emit_byte(w & 0xFF); }
    
    void build_huffman(HuffCode* codes, const uint8_t* bits, const uint8_t* vals, int count) {
        uint32_t code = 0;
        int k = 0;
        for (int i = 1; i <= 16 && k < count; i++) {
            for (int j = 0; j < bits[i] && k < count; j++) {
                codes[vals[k]].bits = code;
                codes[vals[k]].len = i;
                code++; k++;
            }
            code <<= 1;
        }
    }
    
    void init_quant(int quality) {
        int q = quality < 50 ? (5000 / quality) : (200 - quality * 2);
        for (int i = 0; i < 64; i++) {
            int yq = (STD_QUANT_Y[i] * q + 50) / 100;
            int cq = (STD_QUANT_C[i] * q + 50) / 100;
            yq = yq < 1 ? 1 : (yq > 255 ? 255 : yq);
            cq = cq < 1 ? 1 : (cq > 255 ? 255 : cq);
            quant_y[i] = yq;
            quant_c[i] = cq;
            quant_y_recip[i] = (int16_t)((32768 + yq/2) / yq);
            quant_c_recip[i] = (int16_t)((32768 + cq/2) / cq);
            quant_y_bias[i] = (int16_t)(yq / 2);
            quant_c_bias[i] = (int16_t)(cq / 2);
        }
    }
    
    void write_headers(int w, int h) {
        write_word(0xFFD8);  // SOI
        write_word(0xFFE0); write_word(16);  // APP0
        const char* jfif = "JFIF";
        for (int i = 0; i < 5; i++) emit_byte(jfif[i]);
        emit_byte(1); emit_byte(1); emit_byte(0);
        write_word(1); write_word(1);
        emit_byte(0); emit_byte(0);
        
        // DQT
        write_word(0xFFDB); write_word(2 + 65 + 65);
        emit_byte(0);
        for (int i = 0; i < 64; i++) emit_byte(quant_y[ZIGZAG[i]]);
        emit_byte(1);
        for (int i = 0; i < 64; i++) emit_byte(quant_c[ZIGZAG[i]]);
        
        // SOF0
        write_word(0xFFC0); write_word(17);
        emit_byte(8); write_word(h); write_word(w);
        emit_byte(3);
        emit_byte(1); emit_byte(0x22); emit_byte(0);
        emit_byte(2); emit_byte(0x11); emit_byte(1);
        emit_byte(3); emit_byte(0x11); emit_byte(1);
        
        // DHT
        auto wt = [this](int cls, int id, const uint8_t* bits, const uint8_t* vals) {
            int len = 0;
            for (int i = 1; i <= 16; i++) len += bits[i];
            write_word(0xFFC4); write_word(2 + 1 + 16 + len);
            emit_byte((cls << 4) | id);
            for (int i = 1; i <= 16; i++) emit_byte(bits[i]);
            for (int i = 0; i < len; i++) emit_byte(vals[i]);
        };
        wt(0, 0, DC_LUMA_BITS, DC_LUMA_VAL);
        wt(1, 0, AC_LUMA_BITS, AC_LUMA_VAL);
        wt(0, 1, DC_CHROMA_BITS, DC_CHROMA_VAL);
        wt(1, 1, AC_CHROMA_BITS, AC_CHROMA_VAL);
        
        // SOS
        write_word(0xFFDA); write_word(12);
        emit_byte(3);
        emit_byte(1); emit_byte(0x00);
        emit_byte(2); emit_byte(0x11);
        emit_byte(3); emit_byte(0x11);
        emit_byte(0); emit_byte(63); emit_byte(0);
    }
    
    inline int encode_dc(int dc, int last_dc, const HuffCode* table) {
        int diff = dc - last_dc;
        int val = diff < 0 ? -diff : diff;
        int bits = fast_bit_count(val);
        const HuffCode& hc = table[bits];
        write_bits(hc.bits, hc.len);
        if (bits) {
            if (diff < 0) diff = (1 << bits) - 1 + diff;
            write_bits(diff, bits);
        }
        return dc;
    }
    
    void encode_ac(const int16_t* block, const HuffCode* table) {
        int zero_run = 0;
        for (int i = 1; i < 64; i++) {
            int val = block[i];  // Already in zigzag order from GPU
            if (val == 0) { zero_run++; continue; }
            while (zero_run >= 16) {
                write_bits(table[0xF0].bits, table[0xF0].len);
                zero_run -= 16;
            }
            int absval = val < 0 ? -val : val;
            int bits = fast_bit_count(absval);
            int sym = (zero_run << 4) | bits;
            write_bits(table[sym].bits, table[sym].len);
            if (val < 0) val = (1 << bits) - 1 + val;
            write_bits(val, bits);
            zero_run = 0;
        }
        if (zero_run > 0) write_bits(table[0].bits, table[0].len);
    }
    
    void fdct(int16_t* block) {
        constexpr int32_t C2 = 3784, C4 = 2896, C6 = 1567;
        int32_t tmp[64];
        for (int i = 0; i < 8; i++) {
            int32_t x0 = block[i*8+0], x1 = block[i*8+1], x2 = block[i*8+2], x3 = block[i*8+3];
            int32_t x4 = block[i*8+4], x5 = block[i*8+5], x6 = block[i*8+6], x7 = block[i*8+7];
            int32_t s0 = x0+x7, s1 = x1+x6, s2 = x2+x5, s3 = x3+x4;
            int32_t d0 = x0-x7, d1 = x1-x6, d2 = x2-x5, d3 = x3-x4;
            int32_t t0 = s0+s3, t1 = s1+s2, t2 = s0-s3, t3 = s1-s2;
            tmp[i*8+0] = t0+t1; tmp[i*8+4] = t0-t1;
            tmp[i*8+2] = (t2*C6 + t3*C2 + 2048)>>12;
            tmp[i*8+6] = (t2*C2 - t3*C6 + 2048)>>12;
            int32_t t10 = d0+d1, t11 = d1+d2, t12 = d2+d3;
            int32_t z5 = ((t10-t12)*C6+2048)>>12;
            int32_t z2 = ((t10*C2+2048)>>12) + z5;
            int32_t z4 = ((t12*C2+2048)>>12) + t12 + z5;
            int32_t z3 = (t11*C4+2048)>>12;
            int32_t z11 = d3+z3, z13 = d3-z3;
            tmp[i*8+5] = z13+z2; tmp[i*8+3] = z13-z2;
            tmp[i*8+1] = z11+z4; tmp[i*8+7] = z11-z4;
        }
        for (int i = 0; i < 8; i++) {
            int32_t s0 = tmp[i]+tmp[56+i], s1 = tmp[8+i]+tmp[48+i];
            int32_t s2 = tmp[16+i]+tmp[40+i], s3 = tmp[24+i]+tmp[32+i];
            int32_t d0 = tmp[i]-tmp[56+i], d1 = tmp[8+i]-tmp[48+i];
            int32_t d2 = tmp[16+i]-tmp[40+i], d3 = tmp[24+i]-tmp[32+i];
            int32_t t0 = s0+s3, t1 = s1+s2, t2 = s0-s3, t3 = s1-s2;
            block[i] = (int16_t)((t0+t1)>>3);
            block[32+i] = (int16_t)((t0-t1)>>3);
            block[16+i] = (int16_t)(((t2*C6+t3*C2+2048)>>12)>>3);
            block[48+i] = (int16_t)(((t2*C2-t3*C6+2048)>>12)>>3);
            int32_t t10 = d0+d1, t11 = d1+d2, t12 = d2+d3;
            int32_t z5 = ((t10-t12)*C6+2048)>>12;
            int32_t z2 = ((t10*C2+2048)>>12) + z5;
            int32_t z4 = ((t12*C2+2048)>>12) + t12 + z5;
            int32_t z3 = (t11*C4+2048)>>12;
            int32_t z11 = d3+z3, z13 = d3-z3;
            block[40+i] = (int16_t)((z13+z2)>>3);
            block[24+i] = (int16_t)((z13-z2)>>3);
            block[8+i] = (int16_t)((z11+z4)>>3);
            block[56+i] = (int16_t)((z11-z4)>>3);
        }
    }
    
    void quantize_zigzag(int16_t* block, const int16_t* recip, const int16_t* bias) {
        alignas(32) int16_t temp[64];
        for (int i = 0; i < 64; i++) {
            int32_t v = block[i];
            int32_t b = v >= 0 ? bias[i] : -bias[i];
            temp[ZIGZAG[i]] = (int16_t)(((v + b) * recip[i]) >> 15);
        }
        memcpy(block, temp, 64 * sizeof(int16_t));
    }
    
public:
    size_t encode(uint8_t* buffer, size_t buffer_size, const uint8_t* rgb, int w, int h, int quality) {
        out_start = buffer;
        out_ptr = buffer;
        out_end = buffer + buffer_size;
        bitbuf = 0;
        bitcount = 0;
        overflow_ = false;
        
        init_bit_category();
        init_quant(quality);
        build_huffman(dc_luma, DC_LUMA_BITS, DC_LUMA_VAL, 12);
        build_huffman(ac_luma, AC_LUMA_BITS, AC_LUMA_VAL, 162);
        build_huffman(dc_chroma, DC_CHROMA_BITS, DC_CHROMA_VAL, 12);
        build_huffman(ac_chroma, AC_CHROMA_BITS, AC_CHROMA_VAL, 162);
        
        write_headers(w, h);
        
        const int mcu_rows = (h + 15) / 16;
        const int mcu_cols = (w + 15) / 16;
        const int total_mcus = mcu_rows * mcu_cols;
        const int stride3 = w * 3;
        
        // Allocate block buffers: 4 Y + 1 Cb + 1 Cr per MCU
        const int total_y_blocks = total_mcus * 4;
        const int total_c_blocks = total_mcus;
        
        y_block_buf.resize(total_y_blocks * 64);
        cb_block_buf.resize(total_c_blocks * 64);
        cr_block_buf.resize(total_c_blocks * 64);
        
        // Phase 1: Extract all blocks (RGB -> YCbCr)
        int y_idx = 0, c_idx = 0;
        for (int mcu_y = 0; mcu_y < mcu_rows; mcu_y++) {
            const int base_y = mcu_y * 16;
            for (int mcu_x = 0; mcu_x < mcu_cols; mcu_x++) {
                const int base_x = mcu_x * 16;
                int16_t* cb = &cb_block_buf[c_idx * 64];
                int16_t* cr = &cr_block_buf[c_idx * 64];
                memset(cb, 0, 64 * sizeof(int16_t));
                memset(cr, 0, 64 * sizeof(int16_t));
                
                for (int by = 0; by < 2; by++) {
                    const int block_y = base_y + by * 8;
                    for (int bx = 0; bx < 2; bx++) {
                        const int block_x = base_x + bx * 8;
                        int16_t* yblk = &y_block_buf[y_idx * 64];
                        const int chroma_base_x = bx * 4;
                        const int chroma_base_y = by * 4;
                        
                        for (int py = 0; py < 8; py++) {
                            const int img_y = block_y + py;
                            if (img_y >= h) {
                                for (int px = 0; px < 8; px++) yblk[py*8+px] = 0;
                                continue;
                            }
                            const uint8_t* row = rgb + img_y * stride3 + block_x * 3;
                            const int max_px = (w - block_x < 8) ? (w - block_x) : 8;
                            
                            int px = 0;
                            for (; px + 1 < max_px; px += 2) {
                                int r0 = row[0], g0 = row[1], b0 = row[2];
                                int r1 = row[3], g1 = row[4], b1 = row[5];
                                row += 6;
                                yblk[py*8+px] = ((19595*r0 + 38470*g0 + 7471*b0 + 32768) >> 16) - 128;
                                yblk[py*8+px+1] = ((19595*r1 + 38470*g1 + 7471*b1 + 32768) >> 16) - 128;
                                int r = (r0+r1)>>1, g = (g0+g1)>>1, bl = (b0+b1)>>1;
                                int cx = chroma_base_x + (px >> 1);
                                int cy = chroma_base_y + (py >> 1);
                                int cb_val = ((-11056*r - 21712*g + 32768*bl) >> 16);
                                int cr_val = ((32768*r - 27440*g - 5328*bl) >> 16);
                                cb[cy*8+cx] += (cb_val + 1) >> 1;
                                cr[cy*8+cx] += (cr_val + 1) >> 1;
                            }
                            for (; px < max_px; px++) {
                                int r = row[0], g = row[1], bl = row[2];
                                row += 3;
                                yblk[py*8+px] = ((19595*r + 38470*g + 7471*bl + 32768) >> 16) - 128;
                                int cx = chroma_base_x + (px >> 1);
                                int cy = chroma_base_y + (py >> 1);
                                int cb_val = ((-11056*r - 21712*g + 32768*bl) >> 16);
                                int cr_val = ((32768*r - 27440*g - 5328*bl) >> 16);
                                cb[cy*8+cx] += (cb_val + 2) >> 2;
                                cr[cy*8+cx] += (cr_val + 2) >> 2;
                            }
                            for (; px < 8; px++) yblk[py*8+px] = 0;
                        }
                        y_idx++;
                    }
                }
                c_idx++;
            }
        }
        
        // Phase 2: GPU DCT + Quantization (or CPU fallback)
        bool use_gpu = gpudct::gpu_available() && total_y_blocks >= 256;
        
        if (use_gpu) {
            y_quant_buf.resize(total_y_blocks * 64);
            cb_quant_buf.resize(total_c_blocks * 64);
            cr_quant_buf.resize(total_c_blocks * 64);
            
            // GPU batch processing
            gpudct::batch_dct_quantize(y_block_buf.data(), y_quant_buf.data(), 
                                       total_y_blocks, quant_y, false);
            gpudct::batch_dct_quantize(cb_block_buf.data(), cb_quant_buf.data(),
                                       total_c_blocks, quant_c, true);
            gpudct::batch_dct_quantize(cr_block_buf.data(), cr_quant_buf.data(),
                                       total_c_blocks, quant_c, true);
        } else {
            // CPU fallback
            y_quant_buf = y_block_buf;
            cb_quant_buf = cb_block_buf;
            cr_quant_buf = cr_block_buf;
            
            for (int i = 0; i < total_y_blocks; i++) {
                fdct(&y_quant_buf[i * 64]);
                quantize_zigzag(&y_quant_buf[i * 64], quant_y_recip, quant_y_bias);
            }
            for (int i = 0; i < total_c_blocks; i++) {
                fdct(&cb_quant_buf[i * 64]);
                quantize_zigzag(&cb_quant_buf[i * 64], quant_c_recip, quant_c_bias);
                fdct(&cr_quant_buf[i * 64]);
                quantize_zigzag(&cr_quant_buf[i * 64], quant_c_recip, quant_c_bias);
            }
        }
        
        // Phase 3: Huffman encoding (sequential)
        int last_dc_y = 0, last_dc_cb = 0, last_dc_cr = 0;
        y_idx = 0; c_idx = 0;
        
        for (int mcu = 0; mcu < total_mcus; mcu++) {
            // 4 Y blocks
            for (int i = 0; i < 4; i++) {
                const int16_t* blk = &y_quant_buf[y_idx * 64];
                last_dc_y = encode_dc(blk[0], last_dc_y, dc_luma);
                encode_ac(blk, ac_luma);
                y_idx++;
            }
            // Cb block
            {
                const int16_t* blk = &cb_quant_buf[c_idx * 64];
                last_dc_cb = encode_dc(blk[0], last_dc_cb, dc_chroma);
                encode_ac(blk, ac_chroma);
            }
            // Cr block
            {
                const int16_t* blk = &cr_quant_buf[c_idx * 64];
                last_dc_cr = encode_dc(blk[0], last_dc_cr, dc_chroma);
                encode_ac(blk, ac_chroma);
            }
            c_idx++;
        }
        
        if (bitcount > 0) write_bits(0x7F, 7);
        write_word(0xFFD9);  // EOI
        
        // Check for buffer overflow
        if (overflow_) {
            return 0;
        }
        
        return static_cast<size_t>(out_ptr - out_start);
    }
};

// Encode with GPU acceleration if available
inline size_t encode_jpeg_gpu(uint8_t* buffer, size_t buffer_size, const uint8_t* rgb, int w, int h, int quality = 80, bool use_gpu = false) {
    if (use_gpu && gpudct::gpu_available() && w * h >= 1000000) {
        GPUMemEncoder enc;
        return enc.encode(buffer, buffer_size, rgb, w, h, quality);
    }
    MemEncoder enc;
    return enc.encode(buffer, buffer_size, rgb, w, h, quality);
}

// Simple API
inline bool encode_jpeg(const char* filename, const uint8_t* rgb, int w, int h, int quality = 80) {
    Encoder enc;
    return enc.encode(filename, rgb, w, h, quality);
}

// checken ob GPU acceleration verfügbar is
inline bool gpu_available() {
    return gpudct::gpu_available();
}

} // namespace fastjpeg
