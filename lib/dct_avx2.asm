; dct_avx2.asm - handwritten AVX2 DCT for JPEG encoding
; by AGDNoob, MIT license
; 
; 8x8 block DCT using SIMD. col pass -> transpose -> row pass -> store
; ~280 lines because compilers cant do this properly
;
; FUNCTION: void dct_avx2(int16_t* block);
; 
; ALIGNMENT REQUIREMENT: Input pointer MUST be 16-byte aligned minimum.
;   - Uses VMOVDQU (unaligned loads) for tolerance, but aligned is faster
;   - Caller should allocate with: alignas(16) or posix_memalign(16)
;   - Unaligned pointers will work but incur 10-15% performance penalty
;
; SAFETY: Function will NOT crash on misaligned pointers (uses VMOVDQU),
;   but aligned pointers are strongly recommended for performance.

bits 64
default rel

section .data align=32
c_0707: times 16 dw 23170
c_0541: times 16 dw 17734
c_0383: times 16 dw 12540

section .text

global dct_avx2
dct_avx2:
    sub rsp, 168
    vmovaps [rsp], xmm6
    vmovaps [rsp+16], xmm7
    vmovaps [rsp+32], xmm8
    vmovaps [rsp+48], xmm9
    vmovaps [rsp+64], xmm10
    vmovaps [rsp+80], xmm11
    vmovaps [rsp+96], xmm12
    vmovaps [rsp+112], xmm13
    vmovaps [rsp+128], xmm14
    vmovaps [rsp+144], xmm15
    
    mov rax, rcx
    
    ; Load as rows
    vmovdqu xmm0, [rax]
    vmovdqu xmm1, [rax+16]
    vmovdqu xmm2, [rax+32]
    vmovdqu xmm3, [rax+48]
    vmovdqu xmm4, [rax+64]
    vmovdqu xmm5, [rax+80]
    vmovdqu xmm6, [rax+96]
    vmovdqu xmm7, [rax+112]
    
    ; Transpose: rows -> columns in xmm8-15
    vpunpcklwd xmm8, xmm0, xmm1
    vpunpckhwd xmm9, xmm0, xmm1
    vpunpcklwd xmm10, xmm2, xmm3
    vpunpckhwd xmm11, xmm2, xmm3
    vpunpcklwd xmm12, xmm4, xmm5
    vpunpckhwd xmm13, xmm4, xmm5
    vpunpcklwd xmm14, xmm6, xmm7
    vpunpckhwd xmm15, xmm6, xmm7
    
    vpunpckldq xmm0, xmm8, xmm10
    vpunpckhdq xmm1, xmm8, xmm10
    vpunpckldq xmm2, xmm9, xmm11
    vpunpckhdq xmm3, xmm9, xmm11
    vpunpckldq xmm4, xmm12, xmm14
    vpunpckhdq xmm5, xmm12, xmm14
    vpunpckldq xmm6, xmm13, xmm15
    vpunpckhdq xmm7, xmm13, xmm15
    
    vpunpcklqdq xmm8, xmm0, xmm4
    vpunpckhqdq xmm9, xmm0, xmm4
    vpunpcklqdq xmm10, xmm1, xmm5
    vpunpckhqdq xmm11, xmm1, xmm5
    vpunpcklqdq xmm12, xmm2, xmm6
    vpunpckhqdq xmm13, xmm2, xmm6
    vpunpcklqdq xmm14, xmm3, xmm7
    vpunpckhqdq xmm15, xmm3, xmm7
    
    ; xmm8-15 = columns 0-7, each has 8 values from 8 rows
    
    ; ===== COLUMN DCT =====
    vpaddw xmm0, xmm8, xmm15
    vpsubw xmm7, xmm8, xmm15
    vpaddw xmm1, xmm9, xmm14
    vpsubw xmm6, xmm9, xmm14
    vpaddw xmm2, xmm10, xmm13
    vpsubw xmm5, xmm10, xmm13
    vpaddw xmm3, xmm11, xmm12
    vpsubw xmm4, xmm11, xmm12
    
    vpaddw xmm8, xmm0, xmm3
    vpsubw xmm9, xmm0, xmm3
    vpaddw xmm10, xmm1, xmm2
    vpsubw xmm11, xmm1, xmm2
    
    vpaddw xmm12, xmm8, xmm10
    vpsubw xmm13, xmm8, xmm10
    
    vpmulhrsw xmm8, xmm9, [c_0541]
    vpmulhrsw xmm0, xmm11, [c_0383]
    vpaddw xmm14, xmm8, xmm0
    
    vpmulhrsw xmm8, xmm9, [c_0383]
    vpmulhrsw xmm0, xmm11, [c_0541]
    vpsubw xmm15, xmm8, xmm0
    
    vpaddw xmm0, xmm7, xmm6
    vpaddw xmm1, xmm6, xmm5
    vpaddw xmm2, xmm5, xmm4
    
    vpsubw xmm3, xmm0, xmm2
    vpmulhrsw xmm3, xmm3, [c_0383]
    
    vpmulhrsw xmm8, xmm0, [c_0541]
    vpaddw xmm8, xmm8, xmm3
    
    vpmulhrsw xmm9, xmm2, [c_0383]
    vpaddw xmm9, xmm2, xmm9
    vpaddw xmm9, xmm9, xmm3
    
    vpmulhrsw xmm10, xmm1, [c_0707]
    
    vpaddw xmm11, xmm4, xmm10
    vpsubw xmm1, xmm4, xmm10
    
    vpaddw xmm0, xmm1, xmm8
    vpsubw xmm2, xmm1, xmm8
    vpaddw xmm3, xmm11, xmm9
    vpsubw xmm4, xmm11, xmm9
    
    ; Reorder: Y0=xmm12, Y1=xmm3, Y2=xmm14, Y3=xmm2, Y4=xmm13, Y5=xmm0, Y6=xmm15, Y7=xmm4
    vmovdqa xmm8, xmm12
    vmovdqa xmm9, xmm3
    vmovdqa xmm10, xmm14
    vmovdqa xmm11, xmm2
    vmovdqa xmm5, xmm13
    vmovdqa xmm12, xmm5
    vmovdqa xmm13, xmm0
    vmovdqa xmm6, xmm15
    vmovdqa xmm14, xmm6
    vmovdqa xmm15, xmm4
    
    ; ===== ROW DCT (same on transposed data = no extra transpose needed) =====
    ; xmm8-15 are now column results, need to transpose back for row DCT
    ; Then transpose result for final output
    
    ; Transpose columns back to rows
    vpunpcklwd xmm0, xmm8, xmm9
    vpunpckhwd xmm1, xmm8, xmm9
    vpunpcklwd xmm2, xmm10, xmm11
    vpunpckhwd xmm3, xmm10, xmm11
    vpunpcklwd xmm4, xmm12, xmm13
    vpunpckhwd xmm5, xmm12, xmm13
    vpunpcklwd xmm6, xmm14, xmm15
    vpunpckhwd xmm7, xmm14, xmm15
    
    vpunpckldq xmm8, xmm0, xmm2
    vpunpckhdq xmm9, xmm0, xmm2
    vpunpckldq xmm10, xmm1, xmm3
    vpunpckhdq xmm11, xmm1, xmm3
    vpunpckldq xmm12, xmm4, xmm6
    vpunpckhdq xmm13, xmm4, xmm6
    vpunpckldq xmm14, xmm5, xmm7
    vpunpckhdq xmm15, xmm5, xmm7
    
    vpunpcklqdq xmm0, xmm8, xmm12
    vpunpckhqdq xmm1, xmm8, xmm12
    vpunpcklqdq xmm2, xmm9, xmm13
    vpunpckhqdq xmm3, xmm9, xmm13
    vpunpcklqdq xmm4, xmm10, xmm14
    vpunpckhqdq xmm5, xmm10, xmm14
    vpunpcklqdq xmm6, xmm11, xmm15
    vpunpckhqdq xmm7, xmm11, xmm15
    
    ; xmm0-7 are rows after col DCT
    ; Transpose again for row DCT operating on columns
    vpunpcklwd xmm8, xmm0, xmm1
    vpunpckhwd xmm9, xmm0, xmm1
    vpunpcklwd xmm10, xmm2, xmm3
    vpunpckhwd xmm11, xmm2, xmm3
    vpunpcklwd xmm12, xmm4, xmm5
    vpunpckhwd xmm13, xmm4, xmm5
    vpunpcklwd xmm14, xmm6, xmm7
    vpunpckhwd xmm15, xmm6, xmm7
    
    vpunpckldq xmm0, xmm8, xmm10
    vpunpckhdq xmm1, xmm8, xmm10
    vpunpckldq xmm2, xmm9, xmm11
    vpunpckhdq xmm3, xmm9, xmm11
    vpunpckldq xmm4, xmm12, xmm14
    vpunpckhdq xmm5, xmm12, xmm14
    vpunpckldq xmm6, xmm13, xmm15
    vpunpckhdq xmm7, xmm13, xmm15
    
    vpunpcklqdq xmm8, xmm0, xmm4
    vpunpckhqdq xmm9, xmm0, xmm4
    vpunpcklqdq xmm10, xmm1, xmm5
    vpunpckhqdq xmm11, xmm1, xmm5
    vpunpcklqdq xmm12, xmm2, xmm6
    vpunpckhqdq xmm13, xmm2, xmm6
    vpunpcklqdq xmm14, xmm3, xmm7
    vpunpckhqdq xmm15, xmm3, xmm7
    
    ; Row DCT
    vpaddw xmm0, xmm8, xmm15
    vpsubw xmm7, xmm8, xmm15
    vpaddw xmm1, xmm9, xmm14
    vpsubw xmm6, xmm9, xmm14
    vpaddw xmm2, xmm10, xmm13
    vpsubw xmm5, xmm10, xmm13
    vpaddw xmm3, xmm11, xmm12
    vpsubw xmm4, xmm11, xmm12
    
    vpaddw xmm8, xmm0, xmm3
    vpsubw xmm9, xmm0, xmm3
    vpaddw xmm10, xmm1, xmm2
    vpsubw xmm11, xmm1, xmm2
    
    vpaddw xmm12, xmm8, xmm10
    vpsubw xmm13, xmm8, xmm10
    
    vpmulhrsw xmm8, xmm9, [c_0541]
    vpmulhrsw xmm0, xmm11, [c_0383]
    vpaddw xmm14, xmm8, xmm0
    
    vpmulhrsw xmm8, xmm9, [c_0383]
    vpmulhrsw xmm0, xmm11, [c_0541]
    vpsubw xmm15, xmm8, xmm0
    
    vpaddw xmm0, xmm7, xmm6
    vpaddw xmm1, xmm6, xmm5
    vpaddw xmm2, xmm5, xmm4
    
    vpsubw xmm3, xmm0, xmm2
    vpmulhrsw xmm3, xmm3, [c_0383]
    
    vpmulhrsw xmm8, xmm0, [c_0541]
    vpaddw xmm8, xmm8, xmm3
    
    vpmulhrsw xmm9, xmm2, [c_0383]
    vpaddw xmm9, xmm2, xmm9
    vpaddw xmm9, xmm9, xmm3
    
    vpmulhrsw xmm10, xmm1, [c_0707]
    
    vpaddw xmm11, xmm4, xmm10
    vpsubw xmm1, xmm4, xmm10
    
    vpaddw xmm0, xmm1, xmm8
    vpsubw xmm2, xmm1, xmm8
    vpaddw xmm3, xmm11, xmm9
    vpsubw xmm4, xmm11, xmm9
    
    ; Reorder
    vmovdqa xmm8, xmm12
    vmovdqa xmm9, xmm3
    vmovdqa xmm10, xmm14
    vmovdqa xmm11, xmm2
    vmovdqa xmm5, xmm13
    vmovdqa xmm12, xmm5
    vmovdqa xmm13, xmm0
    vmovdqa xmm6, xmm15
    vmovdqa xmm14, xmm6
    vmovdqa xmm15, xmm4
    
    ; Final transpose to row-major output
    vpunpcklwd xmm0, xmm8, xmm9
    vpunpckhwd xmm1, xmm8, xmm9
    vpunpcklwd xmm2, xmm10, xmm11
    vpunpckhwd xmm3, xmm10, xmm11
    vpunpcklwd xmm4, xmm12, xmm13
    vpunpckhwd xmm5, xmm12, xmm13
    vpunpcklwd xmm6, xmm14, xmm15
    vpunpckhwd xmm7, xmm14, xmm15
    
    vpunpckldq xmm8, xmm0, xmm2
    vpunpckhdq xmm9, xmm0, xmm2
    vpunpckldq xmm10, xmm1, xmm3
    vpunpckhdq xmm11, xmm1, xmm3
    vpunpckldq xmm12, xmm4, xmm6
    vpunpckhdq xmm13, xmm4, xmm6
    vpunpckldq xmm14, xmm5, xmm7
    vpunpckhdq xmm15, xmm5, xmm7
    
    vpunpcklqdq xmm0, xmm8, xmm12
    vpunpckhqdq xmm1, xmm8, xmm12
    vpunpcklqdq xmm2, xmm9, xmm13
    vpunpckhqdq xmm3, xmm9, xmm13
    vpunpcklqdq xmm4, xmm10, xmm14
    vpunpckhqdq xmm5, xmm10, xmm14
    vpunpcklqdq xmm6, xmm11, xmm15
    vpunpckhqdq xmm7, xmm11, xmm15
    
    ; Scale >> 3
    vpsraw xmm0, xmm0, 3
    vpsraw xmm1, xmm1, 3
    vpsraw xmm2, xmm2, 3
    vpsraw xmm3, xmm3, 3
    vpsraw xmm4, xmm4, 3
    vpsraw xmm5, xmm5, 3
    vpsraw xmm6, xmm6, 3
    vpsraw xmm7, xmm7, 3
    
    ; Store
    vmovdqu [rax], xmm0
    vmovdqu [rax+16], xmm1
    vmovdqu [rax+32], xmm2
    vmovdqu [rax+48], xmm3
    vmovdqu [rax+64], xmm4
    vmovdqu [rax+80], xmm5
    vmovdqu [rax+96], xmm6
    vmovdqu [rax+112], xmm7
    
    vmovaps xmm6, [rsp]
    vmovaps xmm7, [rsp+16]
    vmovaps xmm8, [rsp+32]
    vmovaps xmm9, [rsp+48]
    vmovaps xmm10, [rsp+64]
    vmovaps xmm11, [rsp+80]
    vmovaps xmm12, [rsp+96]
    vmovaps xmm13, [rsp+112]
    vmovaps xmm14, [rsp+128]
    vmovaps xmm15, [rsp+144]
    add rsp, 168
    
    vzeroupper
    ret
