/**
 * metal_backend.m — Metal GPU backend for Opacc1ty inference.
 *
 * This is the performance-critical Objective-C implementation that:
 * 1. Compiles the Metal shaders at runtime (or loads precompiled .metallib)
 * 2. Allocates Metal buffers from mmap'd .bf2 weight data (zero-copy)
 * 3. Dispatches fused dequant+matmul kernels for each transformer layer
 *
 * The key optimization: weight data from the mmap'd .bf2 file is wrapped
 * in MTLBuffer objects with MTLStorageModeShared, so the GPU reads
 * quantized weights directly from the file mapping — no copies.
 *
 * Buffer layout for a quantized layer:
 *   codebooks_buf:  [n_groups][n_entries][sub_vector_size]  fp16
 *   indices_buf:    [n_groups][sub_vector_size]              uint8
 *   input_buf:      [in_features]                            fp16
 *   output_buf:     [out_features]                           fp16
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dlfcn.h>

#include "opacc1ty.h"

/* ================================================================
 * Metal context
 * ================================================================ */

typedef struct {
    id<MTLDevice>           device;
    id<MTLCommandQueue>     command_queue;
    id<MTLLibrary>          library;
    id<MTLComputePipelineState>  gemv_pipeline;
    id<MTLComputePipelineState>  gemv_tiled_pipeline;
    id<MTLComputePipelineState>  gemm_pipeline;

    /* Temporary buffers for intermediate activations */
    id<MTLBuffer>           input_buf;
    id<MTLBuffer>           output_buf;

    /* Pre-allocated KV cache */
    id<MTLBuffer>           k_cache;
    id<MTLBuffer>           v_cache;
    uint32_t                kv_seq_len;

    /* GPU capabilities */
    NSUInteger              max_threads_per_threadgroup;
    NSUInteger              threadgroup_memory_size;
} MetalContext;

/* Forward declarations */
static MetalContext *metal_init(BFEngine *engine);
static void metal_destroy(MetalContext *ctx);
static int metal_forward_layer(
    MetalContext *ctx,
    const void *codebooks,
    const void *indices,
    const float *input,
    float *output,
    uint32_t M,
    uint32_t K
);

/* ================================================================
 * Public API implementation
 * ================================================================ */

MetalContext *metal_backend_init(BFEngine *engine) {
    MetalContext *ctx = calloc(1, sizeof(MetalContext));
    if (!ctx) return NULL;

    /* Get default Metal device (Apple Silicon GPU) */
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "Opacc1ty: No Metal device found\n");
        free(ctx);
        return NULL;
    }

    NSLog(@"Opacc1ty Metal backend: %@", ctx->device.name);

    ctx->command_queue = [ctx->device newCommandQueue];
    ctx->max_threads_per_threadgroup =
        ctx->device.maxThreadsPerThreadgroup;
    ctx->threadgroup_memory_size =
        ctx->device.maxThreadgroupMemoryLength;

    /* Load pre-compiled Metallib */
    NSError *error = nil;

    /* Look for opacc1ty.metallib next to the executable */
    NSString *exePath = [[NSBundle mainBundle] executablePath];
    if (!exePath) {
        /* Not in a bundle — look in standard locations */
        exePath = [[[NSProcessInfo processInfo] arguments] objectAtIndex:0];
    }

    NSString *metallibDir = [[exePath stringByDeletingLastPathComponent]
                              stringByDeletingLastPathComponent];
    NSString *metallibPath = [metallibDir
        stringByAppendingPathComponent:@"kernels/opacc1ty.metallib"];

    /* Also try current directory */
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:metallibPath]) {
        metallibPath = @"kernels/opacc1ty.metallib";
    }

    ctx->library = [ctx->device newLibraryWithFile:metallibPath error:&error];
    if (!ctx->library) {
        NSLog(@"Opacc1ty: Failed to load metallib: %@", error);

        /* Attempt runtime compilation from source */
        NSString *kernelPath = @"kernels/dequant_gemv.metal";
        NSString *source = [NSString stringWithContentsOfFile:kernelPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;
            ctx->library = [ctx->device newLibraryWithSource:source
                                                     options:opts
                                                       error:&error];
        }

        if (!ctx->library) {
            NSLog(@"Opacc1ty: Cannot compile Metal shaders: %@", error);
            free(ctx);
            return NULL;
        }
    }

    /* Load compute pipelines */
    id<MTLFunction> gemv_func = [ctx->library
        newFunctionWithName:@"dequant_gemv_2bit"];
    id<MTLFunction> gemv_tiled_func = [ctx->library
        newFunctionWithName:@"dequant_gemv_2bit_tiled"];
    id<MTLFunction> gemm_func = [ctx->library
        newFunctionWithName:@"dequant_gemm_2bit"];

    if (gemv_func) {
        ctx->gemv_pipeline = [ctx->device
            newComputePipelineStateWithFunction:gemv_func error:&error];
    }

    if (gemv_tiled_func) {
        ctx->gemv_tiled_pipeline = [ctx->device
            newComputePipelineStateWithFunction:gemv_tiled_func error:&error];
    }

    if (gemm_func) {
        ctx->gemm_pipeline = [ctx->device
            newComputePipelineStateWithFunction:gemm_func error:&error];
    }

    NSLog(@"Opacc1ty: Loaded %d compute pipelines",
          (ctx->gemv_pipeline ? 1 : 0) +
          (ctx->gemm_pipeline ? 1 : 0));

    /* Allocate reusable buffers for activations */
    BFModelConfig *cfg = &engine->config;
    uint32_t hidden_size = cfg->hidden_size > 0 ? cfg->hidden_size : 4096;

    ctx->input_buf = [ctx->device
        newBufferWithLength:hidden_size * sizeof(uint16_t)
        options:MTLResourceStorageModeShared];
    ctx->output_buf = [ctx->device
        newBufferWithLength:hidden_size * sizeof(uint16_t)
        options:MTLResourceStorageModeShared];

    /* Allocate KV cache */
    uint32_t n_layers = cfg->num_hidden_layers > 0 ? cfg->num_hidden_layers : 32;
    uint32_t max_seq = cfg->max_position_embeddings > 0
                       ? cfg->max_position_embeddings : 4096;
    uint32_t n_kv_heads = cfg->num_kv_heads > 0 ? cfg->num_kv_heads : 32;
    uint32_t head_dim = cfg->head_dim > 0 ? cfg->head_dim : 128;

    NSUInteger kv_size = (NSUInteger)n_layers * max_seq
                         * n_kv_heads * head_dim * sizeof(uint16_t);

    ctx->k_cache = [ctx->device
        newBufferWithLength:kv_size
        options:MTLResourceStorageModeShared];
    ctx->v_cache = [ctx->device
        newBufferWithLength:kv_size
        options:MTLResourceStorageModeShared];

    engine->kv_cache_size = kv_size * 2;

    NSLog(@"Opacc1ty: KV cache allocated (%lu MB for %u layers, %u seq)",
          (unsigned long)(kv_size * 2 / (1024 * 1024)),
          n_layers, max_seq);

    return ctx;
}

void metal_backend_destroy(MetalContext *ctx) {
    if (!ctx) return;
    [ctx->input_buf release];
    [ctx->output_buf release];
    [ctx->k_cache release];
    [ctx->v_cache release];
    [ctx->gemv_pipeline release];
    [ctx->gemm_pipeline release];
    [ctx->library release];
    [ctx->command_queue release];
    [ctx->device release];
    free(ctx);
}

/**
 * Dispatch the fused dequant+GEMV kernel for one layer.
 *
 * This is the hot path for autoregressive decode.
 *
 * @param codebooks  Pointer to codebook values in shared memory.
 * @param indices    Pointer to packed 2-bit indices in shared memory.
 * @param input      Input activation vector (fp32 on host, uploaded to GPU).
 * @param output     Output activation vector (fp32 on host, downloaded from GPU).
 * @param M          Output dimension (rows).
 * @param K          Input dimension (columns).
 */
static int metal_forward_layer(
    MetalContext *ctx,
    const void *codebooks,
    const void *indices,
    const float *input,
    float *output,
    uint32_t M,
    uint32_t K
) {
    if (!ctx->gemv_pipeline) {
        /* CPU fallback */
        /* In production, this would run the dequant+GEMV on CPU */
        for (uint32_t i = 0; i < M; i++) {
            output[i] = 0.0f;
        }
        return -1;
    }

    uint32_t K_groups = K / 8;

    /* Upload input to GPU buffer */
    uint16_t *input_half = (uint16_t *)[ctx->input_buf contents];
    for (uint32_t i = 0; i < K; i++) {
        /* Convert fp32 → fp16 */
        uint32_t f32 = *(uint32_t *)&input[i];
        uint32_t sign = (f32 >> 16) & 0x8000;
        int32_t exp = ((f32 >> 23) & 0xff) - 127;
        uint32_t mant = (f32 >> 13) & 0x3ff;

        if (exp > 15) {
            input_half[i] = sign | 0x7c00; /* inf */
        } else if (exp < -14) {
            input_half[i] = sign; /* zero/subnormal */
        } else {
            input_half[i] = sign | ((exp + 15) << 10) | mant;
        }
    }

    /* Create Metal buffers wrapping the quantized weight data */
    /* These use MTLStorageModeShared so GPU can read directly from mmap'd data */
    id<MTLBuffer> cb_buf = [ctx->device
        newBufferWithBytesNoCopy:(void *)codebooks
        length:M * 4 * 8 * sizeof(uint16_t)
        options:MTLResourceStorageModeShared
        deallocator:nil];

    id<MTLBuffer> idx_buf = [ctx->device
        newBufferWithBytesNoCopy:(void *)indices
        length:M * K_groups
        options:MTLResourceStorageModeShared
        deallocator:nil];

    /* Create command buffer and encoder */
    id<MTLCommandBuffer> cmd_buf = [ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

    [encoder setComputePipelineState:ctx->gemv_pipeline];
    [encoder setBuffer:cb_buf offset:0 atIndex:0];
    [encoder setBuffer:idx_buf offset:0 atIndex:1];
    [encoder setBuffer:ctx->input_buf offset:0 atIndex:2];
    [encoder setBuffer:ctx->output_buf offset:0 atIndex:3];

    /* Encode scalar parameters */
    uint32_t params[] = {M, K, K_groups};
    [encoder setBytes:params length:sizeof(params) atIndex:4];
    [encoder setBytes:&params[0] length:4 atIndex:4];
    [encoder setBytes:&params[1] length:4 atIndex:5];
    [encoder setBytes:&params[2] length:4 atIndex:6];

    /* Calculate threadgroup and grid sizes */
    NSUInteger threadgroup_size = 256;
    if (threadgroup_size > ctx->max_threads_per_threadgroup) {
        threadgroup_size = ctx->max_threads_per_threadgroup;
    }

    MTLSize grid_size = MTLSizeMake(M, 1, 1);
    MTLSize group_size = MTLSizeMake(threadgroup_size, 1, 1);

    [encoder dispatchThreads:grid_size
       threadsPerThreadgroup:group_size];
    [encoder endEncoding];

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    /* Download results from GPU */
    uint16_t *output_half = (uint16_t *)[ctx->output_buf contents];
    for (uint32_t i = 0; i < M; i++) {
        /* Convert fp16 → fp32 */
        uint16_t h = output_half[i];
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = ((h >> 10) & 0x1f);
        uint32_t mant = (h & 0x3ff) << 13;

        if (exp == 0) {
            *(uint32_t *)&output[i] = sign;
        } else if (exp == 31) {
            *(uint32_t *)&output[i] = sign | 0x7f800000 | mant;
        } else {
            *(uint32_t *)&output[i] = sign | ((exp + 112) << 23) | mant;
        }
    }

    [cb_buf release];
    [idx_buf release];

    return 0;
}
