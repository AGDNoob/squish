#pragma once
// gpu dct mit directcompute - dachte das wäre schneller, is aber nicht wirklich lol
// hab ich trotzdem drin gelassen weils lustig war das zu bauen
// braucht keine extra dlls, directx is auf win10+ immer da

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#endif

#include <cstdint>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <cstring>

namespace gpudct {

// HLSL Compute Shader for batched DCT (embedded as string)
static const char* DCT_SHADER = R"(
// DCT constants (scaled by 4096 for fixed-point)
static const int C1 = 4017;  // cos(pi/16) * 4096
static const int C2 = 3784;  // cos(2*pi/16) * 4096
static const int C3 = 3406;  // cos(3*pi/16) * 4096
static const int C4 = 2896;  // cos(4*pi/16) * 4096 = sqrt(2)/2 * 4096
static const int C5 = 2276;  // cos(5*pi/16) * 4096
static const int C6 = 1567;  // cos(6*pi/16) * 4096
static const int C7 = 799;   // cos(7*pi/16) * 4096

// Quantization tables (will be set via constant buffer)
cbuffer QuantTables : register(b0) {
    int4 quant_y[16];   // 64 values packed into int4
    int4 quant_c[16];
    int block_count;
    int is_chroma;
    int2 padding;
};

// Input: RGB pixels as int16 blocks (already converted to YCbCr and shifted by -128)
// Output: Quantized DCT coefficients
StructuredBuffer<int> input_blocks : register(t0);
RWStructuredBuffer<int> output_blocks : register(u0);

// Zigzag order
static const int ZIGZAG[64] = {
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
};

groupshared int block[64];
groupshared int temp[64];

// 1D DCT on 8 values (AAN algorithm)
void dct_1d(inout int v0, inout int v1, inout int v2, inout int v3,
            inout int v4, inout int v5, inout int v6, inout int v7) {
    // Stage 1: butterflies
    int s0 = v0 + v7, d0 = v0 - v7;
    int s1 = v1 + v6, d1 = v1 - v6;
    int s2 = v2 + v5, d2 = v2 - v5;
    int s3 = v3 + v4, d3 = v3 - v4;
    
    // Stage 2
    int t0 = s0 + s3, t3 = s0 - s3;
    int t1 = s1 + s2, t2 = s1 - s2;
    
    // Even outputs
    v0 = t0 + t1;           // DC
    v4 = t0 - t1;
    v2 = (t2 * C6 + t3 * C2 + 2048) >> 12;
    v6 = (t3 * C6 - t2 * C2 + 2048) >> 12;
    
    // Odd outputs
    int t10 = d0 + d1, t11 = d1 + d2, t12 = d2 + d3;
    int z5 = ((t10 - t12) * C6 + 2048) >> 12;
    int z2 = ((t10 * C2 + 2048) >> 12) + z5;
    int z4 = ((t12 * C2 + 2048) >> 12) + t12 + z5;
    int z3 = (t11 * C4 + 2048) >> 12;
    int z11 = d3 + z3, z13 = d3 - z3;
    
    v5 = z13 + z2;
    v3 = z13 - z2;
    v1 = z11 + z4;
    v7 = z11 - z4;
}

[numthreads(64, 1, 1)]
void DCTKernel(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint li : SV_GroupIndex) {
    uint block_idx = gid.x;
    if (block_idx >= (uint)block_count) return;
    
    // block in shared memory laden
    block[li] = input_blocks[block_idx * 64 + li];
    GroupMemoryBarrierWithGroupSync();
    
    // Row DCT (each thread handles one coefficient, but we need 8 threads per row)
    if (li < 8) {
        int row = li;
        int v0 = block[row*8+0], v1 = block[row*8+1], v2 = block[row*8+2], v3 = block[row*8+3];
        int v4 = block[row*8+4], v5 = block[row*8+5], v6 = block[row*8+6], v7 = block[row*8+7];
        dct_1d(v0, v1, v2, v3, v4, v5, v6, v7);
        temp[row*8+0] = v0; temp[row*8+1] = v1; temp[row*8+2] = v2; temp[row*8+3] = v3;
        temp[row*8+4] = v4; temp[row*8+5] = v5; temp[row*8+6] = v6; temp[row*8+7] = v7;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Column DCT
    if (li < 8) {
        int col = li;
        int v0 = temp[col], v1 = temp[8+col], v2 = temp[16+col], v3 = temp[24+col];
        int v4 = temp[32+col], v5 = temp[40+col], v6 = temp[48+col], v7 = temp[56+col];
        dct_1d(v0, v1, v2, v3, v4, v5, v6, v7);
        // Scale down and store
        block[col] = v0 >> 3; block[8+col] = v1 >> 3; block[16+col] = v2 >> 3; block[24+col] = v3 >> 3;
        block[32+col] = v4 >> 3; block[40+col] = v5 >> 3; block[48+col] = v6 >> 3; block[56+col] = v7 >> 3;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Quantize and zigzag reorder (all 64 threads)
    int4 qt = is_chroma ? quant_c[li / 4] : quant_y[li / 4];
    int q = (li % 4 == 0) ? qt.x : (li % 4 == 1) ? qt.y : (li % 4 == 2) ? qt.z : qt.w;
    
    int val = block[ZIGZAG[li]];
    int sign = val < 0 ? -1 : 1;
    val = sign * ((abs(val) + q/2) / q);
    
    output_blocks[block_idx * 64 + li] = val;
}
)";

#ifdef _WIN32
using Microsoft::WRL::ComPtr;

class GPUContext {
private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<ID3D11ComputeShader> dct_shader_;
    ComPtr<ID3D11Buffer> const_buffer_;
    ComPtr<ID3D11Buffer> input_buffer_;
    ComPtr<ID3D11Buffer> output_buffer_;
    ComPtr<ID3D11Buffer> staging_buffer_;
    ComPtr<ID3D11ShaderResourceView> input_srv_;
    ComPtr<ID3D11UnorderedAccessView> output_uav_;
    
    size_t max_blocks_ = 0;
    bool initialized_ = false;
    
    struct alignas(16) ConstantBuffer {
        int32_t quant_y[64];
        int32_t quant_c[64];
        int32_t block_count;
        int32_t is_chroma;
        int32_t padding[2];
    };
    
public:
    bool init() {
        if (initialized_) return true;
        
        D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
        UINT flags = 0;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            levels, 1, D3D11_SDK_VERSION,
            &device_, nullptr, &context_
        );
        
        if (FAILED(hr)) {
            // Try WARP (software) fallback
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags,
                levels, 1, D3D11_SDK_VERSION,
                &device_, nullptr, &context_
            );
            if (FAILED(hr)) return false;
        }
        
        // Compile shader
        ComPtr<ID3DBlob> blob, errors;
        hr = D3DCompile(DCT_SHADER, strlen(DCT_SHADER), "DCT",
                       nullptr, nullptr, "DCTKernel", "cs_5_0",
                       D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &errors);
        if (FAILED(hr)) {
            if (errors) {
                // Shader compile error - fall back to CPU
            }
            return false;
        }
        
        hr = device_->CreateComputeShader(blob->GetBufferPointer(), 
                                          blob->GetBufferSize(), nullptr, &dct_shader_);
        if (FAILED(hr)) return false;
        
        // constant buffer bauen
        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = sizeof(ConstantBuffer);
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = device_->CreateBuffer(&cbd, nullptr, &const_buffer_);
        if (FAILED(hr)) return false;
        
        initialized_ = true;
        return true;
    }
    
    bool ensure_buffers(size_t num_blocks) {
        if (num_blocks <= max_blocks_) return true;
        
        // Round up to power of 2 for efficiency
        size_t new_size = 1;
        while (new_size < num_blocks) new_size *= 2;
        new_size = std::min(new_size, size_t(65536));  // Limit to 64K blocks
        
        size_t buffer_size = new_size * 64 * sizeof(int32_t);
        
        // Input buffer (structured)
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = (UINT)buffer_size;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        bd.StructureByteStride = sizeof(int32_t);
        
        if (FAILED(device_->CreateBuffer(&bd, nullptr, &input_buffer_))) return false;
        
        // Output buffer (UAV)
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, &output_buffer_))) return false;
        
        // Staging buffer for readback
        bd.Usage = D3D11_USAGE_STAGING;
        bd.BindFlags = 0;
        bd.MiscFlags = 0;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(device_->CreateBuffer(&bd, nullptr, &staging_buffer_))) return false;
        
        // views erstellen
        D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
        srvd.Format = DXGI_FORMAT_UNKNOWN;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvd.Buffer.NumElements = (UINT)(new_size * 64);
        if (FAILED(device_->CreateShaderResourceView(input_buffer_.Get(), &srvd, &input_srv_))) return false;
        
        D3D11_UNORDERED_ACCESS_VIEW_DESC uavd = {};
        uavd.Format = DXGI_FORMAT_UNKNOWN;
        uavd.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavd.Buffer.NumElements = (UINT)(new_size * 64);
        if (FAILED(device_->CreateUnorderedAccessView(output_buffer_.Get(), &uavd, &output_uav_))) return false;
        
        max_blocks_ = new_size;
        return true;
    }
    
    // mehrere 8x8 blöcke in einem GPU dispatch
    // input: int16_t[64] blocks (YCbCr - 128)
    // output: quantized DCT coeffs in zigzag order
    bool process_blocks(const int16_t* input, int16_t* output, size_t num_blocks,
                       const uint8_t* quant_table, bool is_chroma) {
        if (!initialized_ || num_blocks == 0) return false;
        if (!ensure_buffers(num_blocks)) return false;
        
        // constant buffer updaten
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (FAILED(context_->Map(const_buffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
            return false;
        
        ConstantBuffer* cb = static_cast<ConstantBuffer*>(mapped.pData);
        for (int i = 0; i < 64; i++) {
            cb->quant_y[i] = quant_table[i];
            cb->quant_c[i] = quant_table[i];  // Same table for this call
        }
        cb->block_count = (int32_t)num_blocks;
        cb->is_chroma = is_chroma ? 1 : 0;
        context_->Unmap(const_buffer_.Get(), 0);
        
        // Convert input to int32 and upload
        std::vector<int32_t> input_i32(num_blocks * 64);
        for (size_t i = 0; i < num_blocks * 64; i++) {
            input_i32[i] = input[i];
        }
        context_->UpdateSubresource(input_buffer_.Get(), 0, nullptr, 
                                    input_i32.data(), 0, 0);
        
        // Set pipeline
        context_->CSSetShader(dct_shader_.Get(), nullptr, 0);
        context_->CSSetConstantBuffers(0, 1, const_buffer_.GetAddressOf());
        context_->CSSetShaderResources(0, 1, input_srv_.GetAddressOf());
        context_->CSSetUnorderedAccessViews(0, 1, output_uav_.GetAddressOf(), nullptr);
        
        // Dispatch - one group per block
        context_->Dispatch((UINT)num_blocks, 1, 1);
        
        // Readback
        context_->CopyResource(staging_buffer_.Get(), output_buffer_.Get());
        
        if (FAILED(context_->Map(staging_buffer_.Get(), 0, D3D11_MAP_READ, 0, &mapped)))
            return false;
        
        const int32_t* result = static_cast<const int32_t*>(mapped.pData);
        for (size_t i = 0; i < num_blocks * 64; i++) {
            output[i] = (int16_t)result[i];
        }
        context_->Unmap(staging_buffer_.Get(), 0);
        
        return true;
    }
    
    bool is_available() const { return initialized_; }
};

// Global GPU context (lazy init, thread-safe)
inline GPUContext& get_gpu() {
    static std::once_flag init_flag;
    static GPUContext ctx;
    
    std::call_once(init_flag, []() {
        ctx.init();
    });
    return ctx;
}

inline bool gpu_available() {
    return get_gpu().is_available();
}

// Batch DCT + quantization on GPU
// Returns false if GPU unavailable (caller should use CPU fallback)
inline bool batch_dct_quantize(const int16_t* blocks, int16_t* output, 
                               size_t num_blocks, const uint8_t* quant_table,
                               bool is_chroma = false) {
    GPUContext& gpu = get_gpu();
    if (!gpu.is_available()) return false;
    return gpu.process_blocks(blocks, output, num_blocks, quant_table, is_chroma);
}

#else
// Non-Windows stub
inline bool gpu_available() { return false; }
inline bool batch_dct_quantize(const int16_t*, int16_t*, size_t, const uint8_t*, bool = false) {
    return false;
}
#endif

} // namespace gpudct
