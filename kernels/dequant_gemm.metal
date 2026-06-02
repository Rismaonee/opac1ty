// dequant_gemm.metal — Fused 2-bit dequant + GEMM for batched prefill
//
// During prefill (processing the prompt), the input is a matrix of shape
// (K × batch_size) rather than a single vector. This kernel computes the
// full matrix-matrix product with fused dequantization.
//
// Strategy: block-tiled GEMM where each threadgroup computes one tile
// of the output matrix. Codebook lookups happen inside the inner loop;
// codebook values are loaded into threadgroup memory once and shared
// across all threads in the group.
//
// Tiling parameters tuned for Apple GPU:
//   - BM=64, BN=64: output tile size
//   - BK=32: K-dimension tile (must be multiple of SUB_VECTOR_SIZE=8)
//   - Threadgroup memory: ~4KB for input tile + 256B for codebook cache

#include <metal_stdlib>
using namespace metal;

constant uint BM = 64;  // Output tile rows
constant uint BN = 64;  // Output tile columns
constant uint BK = 32;  // Inner loop tile size
constant uint CODEBOOK_SIZE = 4;
constant uint SUB_VECTOR_SIZE = 8;

/// Fused dequant + GEMM for batched prefill.
///
/// Computes: C[M×N] = dequant(A_weights) × B[K×N]
/// where A_weights are stored as 2-bit indices + codebooks.
///
/// This kernel reduces memory bandwidth by ~6× vs. loading fp16 weights,
/// since indices are 1/8 the size and the codebook is cached in threadgroup
/// memory for the duration of the tile computation.
kernel void dequant_gemm_2bit(
    // Quantized weight matrix A (M×K), stored compressed
    device const half*      cb              [[buffer(0)]],  // codebooks: [M][4][8]
    device const uchar*     indices         [[buffer(1)]],  // 2-bit indices: [M][K/8]
    // Input matrix B (K×N)
    device const half*      input           [[buffer(2)]],
    // Output matrix C (M×N)
    device half*            output          [[buffer(3)]],
    // Dimensions
    constant uint&          M               [[buffer(4)]],
    constant uint&          K               [[buffer(5)]],
    constant uint&          N               [[buffer(6)]],
    constant uint&          K_groups        [[buffer(7)]],  // K / 8
    // Threadgroup position in the grid
    uint2                   group_pos       [[threadgroup_position_in_grid]],
    uint2                   thread_pos      [[thread_position_in_threadgroup]]
) {
    // Tile of output matrix C this threadgroup computes
    uint row_start = group_pos.y * BM;
    uint col_start = group_pos.x * BN;

    // Threadgroup-shared memory for input tile and codebook cache
    threadgroup half input_tile[BK][BN];
    threadgroup half cb_cache[BM][CODEBOOK_SIZE][SUB_VECTOR_SIZE];

    // Accumulator for this thread's output elements (in registers)
    float acc[BM][BN];

    // Zero accumulators
    for (uint mi = 0; mi < BM; mi++) {
        for (uint ni = 0; ni < BN; ni++) {
            acc[mi][ni] = 0.0f;
        }
    }

    // Preload codebooks for this tile's rows into threadgroup memory
    // Each thread helps load a portion
    uint cb_rows_per_thread = (BM + 31) / 32;  // 32 threads per SIMD group
    for (uint r = 0; r < cb_rows_per_thread; r++) {
        uint row = row_start + thread_pos.x * cb_rows_per_thread + r;
        if (row < M) {
            uint cb_base = row * CODEBOOK_SIZE * SUB_VECTOR_SIZE;
            for (uint c = 0; c < CODEBOOK_SIZE; c++) {
                for (uint s = 0; s < SUB_VECTOR_SIZE; s++) {
                    cb_cache[thread_pos.x * cb_rows_per_thread + r][c][s] =
                        cb[cb_base + c * SUB_VECTOR_SIZE + s];
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over K-dimension tiles
    for (uint k_tile = 0; k_tile < K; k_tile += BK) {
        // Cooperative load of input tile into threadgroup memory
        uint k_remain = min(BK, K - k_tile);

        // Each thread loads a few elements of the input tile
        for (uint i = thread_pos.y; i < k_remain; i += 32) {
            for (uint j = thread_pos.x; j < BN; j += 32) {
                uint k_idx = k_tile + i;
                uint n_idx = col_start + j;
                if (k_idx < K && n_idx < N) {
                    input_tile[i][j] = input[k_idx * N + n_idx];
                } else {
                    input_tile[i][j] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial products for this K tile
        // Each thread computes its portion of the output tile
        for (uint mi = thread_pos.y; mi < BM; mi += 32) {
            uint m_idx = row_start + mi;
            if (m_idx >= M) continue;

            for (uint ki = 0; ki < k_remain; ki += SUB_VECTOR_SIZE) {
                uint kg = (k_tile + ki) / SUB_VECTOR_SIZE;

                // Load 2-bit index for this sub-vector
                uint byte_idx = m_idx * K_groups + (kg / 4);
                uint bit_shift = (kg % 4) * 2;
                uint cb_idx = (indices[byte_idx] >> bit_shift) & 0x3;

                // For each column in the output tile
                for (uint ni = thread_pos.x; ni < BN; ni += 32) {
                    half inp = input_tile[ki][ni];
                    // Dot product of 8-element sub-vector
                    acc[mi][ni] += float(inp) * float(cb_cache[mi][cb_idx][0]);
                }

                // Remaining 7 elements of the sub-vector
                for (uint s = 1; s < SUB_VECTOR_SIZE; s++) {
                    if (ki + s >= k_remain) break;
                    for (uint ni = thread_pos.x; ni < BN; ni += 32) {
                        half inp = input_tile[ki + s][ni];
                        acc[mi][ni] += float(inp) * float(cb_cache[mi][cb_idx][s]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back results
    for (uint mi = thread_pos.y; mi < BM; mi += 32) {
        uint m_idx = row_start + mi;
        if (m_idx >= M) continue;
        for (uint ni = thread_pos.x; ni < BN; ni += 32) {
            uint n_idx = col_start + ni;
            if (n_idx >= N) continue;
            output[m_idx * N + n_idx] = half(acc[mi][ni]);
        }
    }
}
