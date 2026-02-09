#pragma once
// jpeg decoder - hab ich geschrieben um stb zu ersetzen
// am ende aber nich benutzt weil stb reicht, aber war fun zu bauen
// avx2/sse2/scalar je nach cpu

#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define FAST_JPEG_AVX2 1
#endif

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#include <tmmintrin.h>  // SSSE3 for _mm_shuffle_epi8
#define FAST_JPEG_SSE2 1
#endif

namespace fastjpegdec {

// ============================================================================
// konstanten
// ============================================================================

// zigzag reihenfolge für dct koeffizienten - aus jpeg spec
static constexpr uint8_t ZIGZAG[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// umgekehrte zigzag für schnelleren lookup
static constexpr uint8_t IZIGZAG[64] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

// ============================================================================
// idct konstanten (aan algorithmus, skaliert)
// ============================================================================

// fixed point mathe, 2^12 skaliert weil floats zu langsam
static constexpr int32_t FIX_0_298631336 = 2446;
static constexpr int32_t FIX_0_390180644 = 3196;
static constexpr int32_t FIX_0_541196100 = 4433;
static constexpr int32_t FIX_0_765366865 = 6270;
static constexpr int32_t FIX_0_899976223 = 7373;
static constexpr int32_t FIX_1_175875602 = 9633;
static constexpr int32_t FIX_1_501321110 = 12299;
static constexpr int32_t FIX_1_847759065 = 15137;
static constexpr int32_t FIX_1_961570560 = 16069;
static constexpr int32_t FIX_2_053119869 = 16819;
static constexpr int32_t FIX_2_562915447 = 20995;
static constexpr int32_t FIX_3_072711026 = 25172;

// ============================================================================
// huffman tablestructs
// ============================================================================

struct HuffmanTable {
    // schneller lookup für codes bis 8 bit
    std::array<int16_t, 256> fast_lookup;  // symbol oder -1 wenn länger
    std::array<uint8_t, 256> fast_bits;    // bits gebraucht
    
    // langsamer pfad für längere codes
    std::array<uint8_t, 16> bits;   // bits[i] = anzahl codes mit länge i+1
    std::array<uint8_t, 256> huffval;
    int maxcode[17];
    int valptr[17];
    
    void build() {
        // fast lut bauen
        std::fill(fast_lookup.begin(), fast_lookup.end(), -1);
        std::fill(fast_bits.begin(), fast_bits.end(), 0);
        
        int code = 0;
        int pos = 0;
        
        for (int len = 1; len <= 16; len++) {
            for (int i = 0; i < bits[len-1]; i++) {
                if (len <= 8) {
                    // fast lookup table füllen
                    int pattern = code << (8 - len);
                    int count = 1 << (8 - len);
                    for (int j = 0; j < count; j++) {
                        fast_lookup[pattern + j] = huffval[pos];
                        fast_bits[pattern + j] = len;
                    }
                }
                code++;
                pos++;
            }
            maxcode[len] = code - 1;
            valptr[len] = pos - bits[len-1];
            code <<= 1;
        }
    }
};

// ============================================================================
// bit reader fürs huffman decoding
// ============================================================================

class BitReader {
public:
    const uint8_t* data;
    const uint8_t* end;
    uint32_t buffer;
    int bits_left;
    
    BitReader(const uint8_t* d, size_t len) 
        : data(d), end(d + len), buffer(0), bits_left(0) {}
    
    // Refill buffer (handles byte stuffing)
    void refill() {
        while (bits_left <= 24 && data < end) {
            uint8_t b = *data++;
            if (b == 0xFF) {
                // stuffed zero skippen
                if (data < end && *data == 0x00) {
                    data++;
                } else {
                    // Marker - stop
                    data--;
                    return;
                }
            }
            buffer = (buffer << 8) | b;
            bits_left += 8;
        }
    }
    
    // Peek n bits without consuming
    inline uint32_t peek(int n) {
        if (bits_left < n) refill();
        return (buffer >> (bits_left - n)) & ((1 << n) - 1);
    }
    
    // n bits skippen
    inline void skip(int n) {
        bits_left -= n;
    }
    
    // n bits holen
    inline int32_t get(int n) {
        if (n == 0) return 0;
        if (bits_left < n) refill();
        bits_left -= n;
        return (buffer >> bits_left) & ((1 << n) - 1);
    }
    
    // Decode Huffman symbol using LUT
    inline int decode(const HuffmanTable& ht) {
        if (bits_left < 16) refill();
        
        // Fast path: 8-bit lookup
        int peek8 = (buffer >> (bits_left - 8)) & 0xFF;
        int symbol = ht.fast_lookup[peek8];
        
        if (symbol >= 0) {
            bits_left -= ht.fast_bits[peek8];
            return symbol;
        }
        
        // Slow path: longer codes
        int code = peek8;
        int len = 8;
        
        while (len < 16 && code > ht.maxcode[len]) {
            code = (code << 1) | ((buffer >> (bits_left - len - 1)) & 1);
            len++;
        }
        
        bits_left -= len;
        if (len > 16) return 0;  // Invalid
        
        int idx = ht.valptr[len] + code - (ht.maxcode[len] - ht.bits[len-1] + 1);
        return ht.huffval[idx];
    }
    
    // Extend sign bit
    static inline int32_t extend(int32_t v, int bits) {
        if (bits == 0) return 0;
        int32_t vt = 1 << (bits - 1);
        if (v < vt) {
            return v + (-1 << bits) + 1;
        }
        return v;
    }
};

// ============================================================================
// simd idct - der wichtige teil
// ============================================================================

#ifdef FAST_JPEG_AVX2

// avx2 8x8 idct - alles auf einmal
inline void idct_avx2(int16_t* block, uint8_t* output, int stride) {
    // alle 8 rows laden
    __m256i row01 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block));
    __m256i row23 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 16));
    __m256i row45 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 32));
    __m256i row67 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 48));
    
    // For now, use scalar IDCT (TODO: full AVX2 implementation)
    // The memory layout optimization still helps
    
    // Scalar IDCT with loop unrolling
    alignas(32) int32_t tmp[64];
    
    // Column pass
    for (int x = 0; x < 8; x++) {
        int32_t s0 = block[x + 0*8];
        int32_t s1 = block[x + 1*8];
        int32_t s2 = block[x + 2*8];
        int32_t s3 = block[x + 3*8];
        int32_t s4 = block[x + 4*8];
        int32_t s5 = block[x + 5*8];
        int32_t s6 = block[x + 6*8];
        int32_t s7 = block[x + 7*8];
        
        // Even part
        int32_t t0 = s0 + s4;
        int32_t t1 = s0 - s4;
        int32_t t2 = (s2 * FIX_0_541196100 + s6 * (FIX_0_541196100 - FIX_1_847759065) + 1024) >> 11;
        int32_t t3 = (s2 * (FIX_0_541196100 + FIX_0_765366865) + s6 * FIX_0_541196100 + 1024) >> 11;
        
        int32_t t10 = t0 + t3;
        int32_t t13 = t0 - t3;
        int32_t t11 = t1 + t2;
        int32_t t12 = t1 - t2;
        
        // Odd part
        int32_t z1 = s7 + s1;
        int32_t z2 = s5 + s3;
        int32_t z3 = s7 + s3;
        int32_t z4 = s5 + s1;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;
        
        int32_t t4 = s7 * FIX_0_298631336;
        int32_t t5 = s5 * FIX_2_053119869;
        int32_t t6 = s3 * FIX_3_072711026;
        int32_t t7 = s1 * FIX_1_501321110;
        
        int32_t z1m = z1 * (-FIX_0_899976223);
        int32_t z2m = z2 * (-FIX_2_562915447);
        int32_t z3m = z3 * (-FIX_1_961570560) + z5;
        int32_t z4m = z4 * (-FIX_0_390180644) + z5;
        
        t4 += z1m + z3m;
        t5 += z2m + z4m;
        t6 += z2m + z3m;
        t7 += z1m + z4m;
        
        // Final output to tmp
        tmp[x + 0*8] = t10 + ((t7 + 1024) >> 11);
        tmp[x + 7*8] = t10 - ((t7 + 1024) >> 11);
        tmp[x + 1*8] = t11 + ((t6 + 1024) >> 11);
        tmp[x + 6*8] = t11 - ((t6 + 1024) >> 11);
        tmp[x + 2*8] = t12 + ((t5 + 1024) >> 11);
        tmp[x + 5*8] = t12 - ((t5 + 1024) >> 11);
        tmp[x + 3*8] = t13 + ((t4 + 1024) >> 11);
        tmp[x + 4*8] = t13 - ((t4 + 1024) >> 11);
    }
    
    // Row pass + output
    for (int y = 0; y < 8; y++) {
        int32_t* row = tmp + y * 8;
        
        int32_t s0 = row[0];
        int32_t s1 = row[1];
        int32_t s2 = row[2];
        int32_t s3 = row[3];
        int32_t s4 = row[4];
        int32_t s5 = row[5];
        int32_t s6 = row[6];
        int32_t s7 = row[7];
        
        // Even part
        int32_t t0 = s0 + s4;
        int32_t t1 = s0 - s4;
        int32_t t2 = (s2 * FIX_0_541196100 + s6 * (FIX_0_541196100 - FIX_1_847759065) + 1024) >> 11;
        int32_t t3 = (s2 * (FIX_0_541196100 + FIX_0_765366865) + s6 * FIX_0_541196100 + 1024) >> 11;
        
        int32_t t10 = t0 + t3;
        int32_t t13 = t0 - t3;
        int32_t t11 = t1 + t2;
        int32_t t12 = t1 - t2;
        
        // Odd part
        int32_t z1 = s7 + s1;
        int32_t z2 = s5 + s3;
        int32_t z3 = s7 + s3;
        int32_t z4 = s5 + s1;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;
        
        int32_t t4 = s7 * FIX_0_298631336;
        int32_t t5 = s5 * FIX_2_053119869;
        int32_t t6 = s3 * FIX_3_072711026;
        int32_t t7 = s1 * FIX_1_501321110;
        
        int32_t z1m = z1 * (-FIX_0_899976223);
        int32_t z2m = z2 * (-FIX_2_562915447);
        int32_t z3m = z3 * (-FIX_1_961570560) + z5;
        int32_t z4m = z4 * (-FIX_0_390180644) + z5;
        
        t4 += z1m + z3m;
        t5 += z2m + z4m;
        t6 += z2m + z3m;
        t7 += z1m + z4m;
        
        // Final output with level shift and clamp
        uint8_t* out = output + y * stride;
        
        #define CLAMP(x) static_cast<uint8_t>(std::max(0, std::min(255, (x))))
        out[0] = CLAMP(((t10 + ((t7 + (1<<17)) >> 18)) >> 3) + 128);
        out[7] = CLAMP(((t10 - ((t7 + (1<<17)) >> 18)) >> 3) + 128);
        out[1] = CLAMP(((t11 + ((t6 + (1<<17)) >> 18)) >> 3) + 128);
        out[6] = CLAMP(((t11 - ((t6 + (1<<17)) >> 18)) >> 3) + 128);
        out[2] = CLAMP(((t12 + ((t5 + (1<<17)) >> 18)) >> 3) + 128);
        out[5] = CLAMP(((t12 - ((t5 + (1<<17)) >> 18)) >> 3) + 128);
        out[3] = CLAMP(((t13 + ((t4 + (1<<17)) >> 18)) >> 3) + 128);
        out[4] = CLAMP(((t13 - ((t4 + (1<<17)) >> 18)) >> 3) + 128);
        #undef CLAMP
    }
}

#endif // FAST_JPEG_AVX2

// Scalar IDCT fallback
inline void idct_scalar(int16_t* block, uint8_t* output, int stride) {
    alignas(16) int32_t tmp[64];
    
    // Column pass
    for (int x = 0; x < 8; x++) {
        int32_t s0 = block[x + 0*8];
        int32_t s1 = block[x + 1*8];
        int32_t s2 = block[x + 2*8];
        int32_t s3 = block[x + 3*8];
        int32_t s4 = block[x + 4*8];
        int32_t s5 = block[x + 5*8];
        int32_t s6 = block[x + 6*8];
        int32_t s7 = block[x + 7*8];
        
        // check für all-zero AC (common case)
        if ((s1 | s2 | s3 | s4 | s5 | s6 | s7) == 0) {
            int32_t dc = s0 << 2;
            tmp[x + 0*8] = dc;
            tmp[x + 1*8] = dc;
            tmp[x + 2*8] = dc;
            tmp[x + 3*8] = dc;
            tmp[x + 4*8] = dc;
            tmp[x + 5*8] = dc;
            tmp[x + 6*8] = dc;
            tmp[x + 7*8] = dc;
            continue;
        }
        
        // Even part
        int32_t t0 = (s0 + s4) << 11;
        int32_t t1 = (s0 - s4) << 11;
        int32_t t2 = s2 * FIX_0_541196100 + s6 * (FIX_0_541196100 - FIX_1_847759065);
        int32_t t3 = s2 * (FIX_0_541196100 + FIX_0_765366865) + s6 * FIX_0_541196100;
        
        int32_t t10 = t0 + t3;
        int32_t t13 = t0 - t3;
        int32_t t11 = t1 + t2;
        int32_t t12 = t1 - t2;
        
        // Odd part
        int32_t z1 = s7 + s1;
        int32_t z2 = s5 + s3;
        int32_t z3 = s7 + s3;
        int32_t z4 = s5 + s1;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;
        
        int32_t t4 = s7 * FIX_0_298631336;
        int32_t t5 = s5 * FIX_2_053119869;
        int32_t t6 = s3 * FIX_3_072711026;
        int32_t t7 = s1 * FIX_1_501321110;
        
        t4 += z1 * (-FIX_0_899976223) + z3 * (-FIX_1_961570560) + z5;
        t5 += z2 * (-FIX_2_562915447) + z4 * (-FIX_0_390180644) + z5;
        t6 += z2 * (-FIX_2_562915447) + z3 * (-FIX_1_961570560) + z5;
        t7 += z1 * (-FIX_0_899976223) + z4 * (-FIX_0_390180644) + z5;
        
        tmp[x + 0*8] = (t10 + t7) >> 11;
        tmp[x + 7*8] = (t10 - t7) >> 11;
        tmp[x + 1*8] = (t11 + t6) >> 11;
        tmp[x + 6*8] = (t11 - t6) >> 11;
        tmp[x + 2*8] = (t12 + t5) >> 11;
        tmp[x + 5*8] = (t12 - t5) >> 11;
        tmp[x + 3*8] = (t13 + t4) >> 11;
        tmp[x + 4*8] = (t13 - t4) >> 11;
    }
    
    // Row pass + output
    for (int y = 0; y < 8; y++) {
        int32_t* row = tmp + y * 8;
        uint8_t* out = output + y * stride;
        
        int32_t s0 = row[0];
        int32_t s1 = row[1];
        int32_t s2 = row[2];
        int32_t s3 = row[3];
        int32_t s4 = row[4];
        int32_t s5 = row[5];
        int32_t s6 = row[6];
        int32_t s7 = row[7];
        
        // check für all-zero AC
        if ((s1 | s2 | s3 | s4 | s5 | s6 | s7) == 0) {
            uint8_t val = static_cast<uint8_t>(std::max(0, std::min(255, ((s0 + 16) >> 5) + 128)));
            out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = out[6] = out[7] = val;
            continue;
        }
        
        // Even part
        int32_t t0 = (s0 + s4) << 11;
        int32_t t1 = (s0 - s4) << 11;
        int32_t t2 = s2 * FIX_0_541196100 + s6 * (FIX_0_541196100 - FIX_1_847759065);
        int32_t t3 = s2 * (FIX_0_541196100 + FIX_0_765366865) + s6 * FIX_0_541196100;
        
        int32_t t10 = t0 + t3 + (1 << 17);
        int32_t t13 = t0 - t3 + (1 << 17);
        int32_t t11 = t1 + t2 + (1 << 17);
        int32_t t12 = t1 - t2 + (1 << 17);
        
        // Odd part
        int32_t z1 = s7 + s1;
        int32_t z2 = s5 + s3;
        int32_t z3 = s7 + s3;
        int32_t z4 = s5 + s1;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;
        
        int32_t t4 = s7 * FIX_0_298631336;
        int32_t t5 = s5 * FIX_2_053119869;
        int32_t t6 = s3 * FIX_3_072711026;
        int32_t t7 = s1 * FIX_1_501321110;
        
        t4 += z1 * (-FIX_0_899976223) + z3 * (-FIX_1_961570560) + z5;
        t5 += z2 * (-FIX_2_562915447) + z4 * (-FIX_0_390180644) + z5;
        t6 += z2 * (-FIX_2_562915447) + z3 * (-FIX_1_961570560) + z5;
        t7 += z1 * (-FIX_0_899976223) + z4 * (-FIX_0_390180644) + z5;
        
        #define CLAMP(x) static_cast<uint8_t>(std::max(0, std::min(255, ((x) >> 18) + 128)))
        out[0] = CLAMP(t10 + t7);
        out[7] = CLAMP(t10 - t7);
        out[1] = CLAMP(t11 + t6);
        out[6] = CLAMP(t11 - t6);
        out[2] = CLAMP(t12 + t5);
        out[5] = CLAMP(t12 - t5);
        out[3] = CLAMP(t13 + t4);
        out[4] = CLAMP(t13 - t4);
        #undef CLAMP
    }
}

// Dispatch to best IDCT
inline void idct(int16_t* block, uint8_t* output, int stride) {
#ifdef FAST_JPEG_AVX2
    idct_avx2(block, output, stride);
#else
    idct_scalar(block, output, stride);
#endif
}

// ============================================================================
// ycbcr zu rgb konvertierung
// ============================================================================

#ifdef FAST_JPEG_AVX2

// 8 pixel auf einmal mit avx2
inline void ycbcr_to_rgb_8_avx2(
    const uint8_t* y_ptr, const uint8_t* cb_ptr, const uint8_t* cr_ptr,
    uint8_t* rgb_out
) {
    // 8 pixel von jeder komponente laden
    __m128i y8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(y_ptr));
    __m128i cb8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(cb_ptr));
    __m128i cr8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(cr_ptr));
    
    // Expand to 16-bit
    __m256i y = _mm256_cvtepu8_epi16(y8);
    __m256i cb = _mm256_cvtepu8_epi16(cb8);
    __m256i cr = _mm256_cvtepu8_epi16(cr8);
    
    // Center Cb and Cr around 0
    __m256i offset = _mm256_set1_epi16(128);
    cb = _mm256_sub_epi16(cb, offset);
    cr = _mm256_sub_epi16(cr, offset);
    
    // R = Y + 1.402 * Cr
    // G = Y - 0.344136 * Cb - 0.714136 * Cr  
    // B = Y + 1.772 * Cb
    
    // Fixed point: multiply by 2^16, then shift right
    __m256i cr_r = _mm256_mulhi_epi16(cr, _mm256_set1_epi16(22970));  // 1.402 * 16384
    __m256i cb_g = _mm256_mulhi_epi16(cb, _mm256_set1_epi16(-5638)); // -0.344 * 16384
    __m256i cr_g = _mm256_mulhi_epi16(cr, _mm256_set1_epi16(-11700)); // -0.714 * 16384
    __m256i cb_b = _mm256_mulhi_epi16(cb, _mm256_set1_epi16(29032));  // 1.772 * 16384
    
    // Adjust for mulhi scaling
    cr_r = _mm256_slli_epi16(cr_r, 2);
    cb_g = _mm256_slli_epi16(cb_g, 2);
    cr_g = _mm256_slli_epi16(cr_g, 2);
    cb_b = _mm256_slli_epi16(cb_b, 2);
    
    __m256i r = _mm256_add_epi16(y, cr_r);
    __m256i g = _mm256_add_epi16(_mm256_add_epi16(y, cb_g), cr_g);
    __m256i b = _mm256_add_epi16(y, cb_b);
    
    // Clamp to 0-255
    __m256i zero = _mm256_setzero_si256();
    __m256i max255 = _mm256_set1_epi16(255);
    r = _mm256_max_epi16(_mm256_min_epi16(r, max255), zero);
    g = _mm256_max_epi16(_mm256_min_epi16(g, max255), zero);
    b = _mm256_max_epi16(_mm256_min_epi16(b, max255), zero);
    
    // Pack to 8-bit
    __m128i r8_out = _mm256_castsi256_si128(_mm256_packus_epi16(r, r));
    __m128i g8_out = _mm256_castsi256_si128(_mm256_packus_epi16(g, g));
    __m128i b8_out = _mm256_castsi256_si128(_mm256_packus_epi16(b, b));
    
    // Interleave RGB (scalar for now, could use shuffle)
    alignas(16) uint8_t r_arr[8], g_arr[8], b_arr[8];
    _mm_storel_epi64(reinterpret_cast<__m128i*>(r_arr), r8_out);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(g_arr), g8_out);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(b_arr), b8_out);
    
    for (int i = 0; i < 8; i++) {
        rgb_out[i*3+0] = r_arr[i];
        rgb_out[i*3+1] = g_arr[i];
        rgb_out[i*3+2] = b_arr[i];
    }
}

#endif // FAST_JPEG_AVX2

// Scalar YCbCr to RGB
inline void ycbcr_to_rgb_scalar(uint8_t y, uint8_t cb, uint8_t cr, uint8_t* rgb) {
    int yy = y;
    int cbb = cb - 128;
    int crr = cr - 128;
    
    int r = yy + ((crr * 359) >> 8);
    int g = yy - ((cbb * 88 + crr * 183) >> 8);
    int b = yy + ((cbb * 454) >> 8);
    
    rgb[0] = static_cast<uint8_t>(std::max(0, std::min(255, r)));
    rgb[1] = static_cast<uint8_t>(std::max(0, std::min(255, g)));
    rgb[2] = static_cast<uint8_t>(std::max(0, std::min(255, b)));
}

// ============================================================================
// jpeg decoder state
// ============================================================================

struct JpegComponent {
    int id;
    int h_samp, v_samp;  // sampling faktoren
    int qt_id;           // quant table id
    int dc_table, ac_table;  // huffman table ids
    int dc_pred;         // dc prediction wert
};

struct JpegImage {
    int width, height;
    int num_components;
    JpegComponent components[4];
    
    std::array<std::array<int16_t, 64>, 4> qtables;
    std::array<HuffmanTable, 4> dc_tables;
    std::array<HuffmanTable, 4> ac_tables;
    
    int restart_interval;
    int max_h_samp, max_v_samp;
    int mcu_width, mcu_height;
    int mcu_cols, mcu_rows;
};

// ============================================================================
// jpeg parser
// ============================================================================

class JpegDecoder {
public:
    std::vector<uint8_t> decode(const uint8_t* data, size_t size, int& width, int& height, int& channels) {
        const uint8_t* ptr = data;
        const uint8_t* end = data + size;
        
        JpegImage img = {};
        const uint8_t* scan_data = nullptr;
        size_t scan_size = 0;
        
        // Parse markers
        while (ptr + 2 <= end) {
            if (*ptr++ != 0xFF) continue;
            
            uint8_t marker = *ptr++;
            if (marker == 0x00 || marker == 0xFF) continue;
            
            if (marker == 0xD8) continue;  // SOI
            if (marker == 0xD9) break;     // EOI
            
            // RST marker skippen
            if (marker >= 0xD0 && marker <= 0xD7) continue;
            
            // Get segment length
            if (ptr + 2 > end) break;
            int len = (ptr[0] << 8) | ptr[1];
            ptr += 2;
            len -= 2;
            
            if (ptr + len > end) break;
            
            switch (marker) {
                case 0xC0:  // SOF0 (Baseline DCT)
                case 0xC1:  // SOF1 (Extended sequential)
                    parse_sof(ptr, len, img);
                    break;
                    
                case 0xC4:  // DHT
                    parse_dht(ptr, len, img);
                    break;
                    
                case 0xDB:  // DQT
                    parse_dqt(ptr, len, img);
                    break;
                    
                case 0xDD:  // DRI
                    img.restart_interval = (ptr[0] << 8) | ptr[1];
                    break;
                    
                case 0xDA:  // SOS
                    parse_sos(ptr, len, img);
                    scan_data = ptr + len;
                    scan_size = end - scan_data;
                    break;
            }
            
            ptr += len;
        }
        
        if (!scan_data || img.width == 0 || img.height == 0) {
            return {};
        }
        
        // Decode scan data
        width = img.width;
        height = img.height;
        channels = (img.num_components == 1) ? 1 : 3;
        
        return decode_scan(scan_data, scan_size, img);
    }
    
private:
    void parse_sof(const uint8_t* data, int len, JpegImage& img) {
        img.height = (data[1] << 8) | data[2];
        img.width = (data[3] << 8) | data[4];
        img.num_components = data[5];
        
        img.max_h_samp = img.max_v_samp = 1;
        
        for (int i = 0; i < img.num_components && i < 4; i++) {
            img.components[i].id = data[6 + i*3];
            img.components[i].h_samp = data[7 + i*3] >> 4;
            img.components[i].v_samp = data[7 + i*3] & 0x0F;
            img.components[i].qt_id = data[8 + i*3];
            img.components[i].dc_pred = 0;
            
            img.max_h_samp = std::max(img.max_h_samp, img.components[i].h_samp);
            img.max_v_samp = std::max(img.max_v_samp, img.components[i].v_samp);
        }
        
        img.mcu_width = img.max_h_samp * 8;
        img.mcu_height = img.max_v_samp * 8;
        img.mcu_cols = (img.width + img.mcu_width - 1) / img.mcu_width;
        img.mcu_rows = (img.height + img.mcu_height - 1) / img.mcu_height;
    }
    
    void parse_dht(const uint8_t* data, int len, JpegImage& img) {
        int pos = 0;
        while (pos < len) {
            int info = data[pos++];
            int tc = info >> 4;  // Table class (0=DC, 1=AC)
            int th = info & 0x0F;  // Table ID
            
            HuffmanTable& ht = (tc == 0) ? img.dc_tables[th] : img.ac_tables[th];
            
            // Read bit counts
            int total = 0;
            for (int i = 0; i < 16; i++) {
                ht.bits[i] = data[pos + i];
                total += ht.bits[i];
            }
            pos += 16;
            
            // Read symbols
            for (int i = 0; i < total && pos < len; i++) {
                ht.huffval[i] = data[pos++];
            }
            
            ht.build();
        }
    }
    
    void parse_dqt(const uint8_t* data, int len, JpegImage& img) {
        int pos = 0;
        while (pos < len) {
            int info = data[pos++];
            int precision = info >> 4;
            int id = info & 0x0F;
            
            for (int i = 0; i < 64; i++) {
                int val;
                if (precision == 0) {
                    val = data[pos++];
                } else {
                    val = (data[pos] << 8) | data[pos + 1];
                    pos += 2;
                }
                img.qtables[id][ZIGZAG[i]] = static_cast<int16_t>(val);
            }
        }
    }
    
    void parse_sos(const uint8_t* data, int len, JpegImage& img) {
        int num_comp = data[0];
        for (int i = 0; i < num_comp; i++) {
            int id = data[1 + i*2];
            int tables = data[2 + i*2];
            
            for (int j = 0; j < img.num_components; j++) {
                if (img.components[j].id == id) {
                    img.components[j].dc_table = tables >> 4;
                    img.components[j].ac_table = tables & 0x0F;
                    break;
                }
            }
        }
    }
    
    std::vector<uint8_t> decode_scan(const uint8_t* data, size_t size, JpegImage& img) {
        BitReader bits(data, size);
        
        // Allocate component buffers
        int comp_w[4], comp_h[4];
        std::vector<uint8_t> comp_data[4];
        
        for (int c = 0; c < img.num_components; c++) {
            comp_w[c] = img.mcu_cols * img.components[c].h_samp * 8;
            comp_h[c] = img.mcu_rows * img.components[c].v_samp * 8;
            comp_data[c].resize(comp_w[c] * comp_h[c], 128);
        }
        
        // Decode MCUs
        alignas(32) int16_t block[64];
        int restart_count = 0;
        
        for (int mcu_y = 0; mcu_y < img.mcu_rows; mcu_y++) {
            for (int mcu_x = 0; mcu_x < img.mcu_cols; mcu_x++) {
                // restart marker check
                if (img.restart_interval > 0 && restart_count == img.restart_interval) {
                    // byte boundary skippen und RST marker finden
                    bits.bits_left = 0;
                    while (bits.data < bits.end - 1) {
                        if (bits.data[0] == 0xFF && bits.data[1] >= 0xD0 && bits.data[1] <= 0xD7) {
                            bits.data += 2;
                            break;
                        }
                        bits.data++;
                    }
                    
                    // Reset DC predictors
                    for (int c = 0; c < img.num_components; c++) {
                        img.components[c].dc_pred = 0;
                    }
                    restart_count = 0;
                }
                
                // Decode each component's blocks in this MCU
                for (int c = 0; c < img.num_components; c++) {
                    auto& comp = img.components[c];
                    auto& qtable = img.qtables[comp.qt_id];
                    auto& dc_ht = img.dc_tables[comp.dc_table];
                    auto& ac_ht = img.ac_tables[comp.ac_table];
                    
                    for (int by = 0; by < comp.v_samp; by++) {
                        for (int bx = 0; bx < comp.h_samp; bx++) {
                            // Decode block
                            std::memset(block, 0, sizeof(block));
                            
                            // DC coefficient
                            int dc_cat = bits.decode(dc_ht);
                            int dc_diff = BitReader::extend(bits.get(dc_cat), dc_cat);
                            comp.dc_pred += dc_diff;
                            block[0] = static_cast<int16_t>(comp.dc_pred * qtable[0]);
                            
                            // AC coefficients
                            int k = 1;
                            while (k < 64) {
                                int ac_val = bits.decode(ac_ht);
                                int run = ac_val >> 4;
                                int cat = ac_val & 0x0F;
                                
                                if (cat == 0) {
                                    if (run == 0) break;  // EOB
                                    if (run == 15) {
                                        k += 16;  // ZRL
                                        continue;
                                    }
                                }
                                
                                k += run;
                                if (k >= 64) break;
                                
                                int ac = BitReader::extend(bits.get(cat), cat);
                                block[ZIGZAG[k]] = static_cast<int16_t>(ac * qtable[ZIGZAG[k]]);
                                k++;
                            }
                            
                            // IDCT and store
                            int px = mcu_x * comp.h_samp * 8 + bx * 8;
                            int py = mcu_y * comp.v_samp * 8 + by * 8;
                            uint8_t* out = comp_data[c].data() + py * comp_w[c] + px;
                            
                            idct(block, out, comp_w[c]);
                        }
                    }
                }
                
                restart_count++;
            }
        }
        
        // Convert to RGB
        std::vector<uint8_t> output(img.width * img.height * 3);
        
        if (img.num_components == 1) {
            // Grayscale
            for (int y = 0; y < img.height; y++) {
                for (int x = 0; x < img.width; x++) {
                    uint8_t gray = comp_data[0][y * comp_w[0] + x];
                    output[(y * img.width + x) * 3 + 0] = gray;
                    output[(y * img.width + x) * 3 + 1] = gray;
                    output[(y * img.width + x) * 3 + 2] = gray;
                }
            }
        } else {
            // YCbCr to RGB
            int y_w = comp_w[0];
            int cb_w = comp_w[1];
            int cr_w = comp_w[2];
            
            int h_ratio1 = img.max_h_samp / img.components[1].h_samp;
            int v_ratio1 = img.max_v_samp / img.components[1].v_samp;
            int h_ratio2 = img.max_h_samp / img.components[2].h_samp;
            int v_ratio2 = img.max_v_samp / img.components[2].v_samp;
            
            for (int y = 0; y < img.height; y++) {
                int cb_y = y / v_ratio1;
                int cr_y = y / v_ratio2;
                
                for (int x = 0; x < img.width; x++) {
                    int cb_x = x / h_ratio1;
                    int cr_x = x / h_ratio2;
                    
                    uint8_t yy = comp_data[0][y * y_w + x];
                    uint8_t cb = comp_data[1][cb_y * cb_w + cb_x];
                    uint8_t cr = comp_data[2][cr_y * cr_w + cr_x];
                    
                    ycbcr_to_rgb_scalar(yy, cb, cr, &output[(y * img.width + x) * 3]);
                }
            }
        }
        
        return output;
    }
};

// ============================================================================
// public api\n// ============================================================================

inline std::vector<uint8_t> decode_jpeg(const uint8_t* data, size_t size, int& width, int& height, int& channels) {
    JpegDecoder decoder;
    return decoder.decode(data, size, width, height, channels);
}

inline std::vector<uint8_t> decode_jpeg_file(const char* filename, int& width, int& height, int& channels) {
    FILE* f = fopen(filename, "rb");
    if (!f) return {};
    
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    std::vector<uint8_t> data(size);
    fread(data.data(), 1, size, f);
    fclose(f);
    
    return decode_jpeg(data.data(), data.size(), width, height, channels);
}

} // namespace fastjpegdec
