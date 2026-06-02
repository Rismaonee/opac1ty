/**
 * opacc1ty.h — C API for Opacc1ty 2-bit quantized inference on Apple Silicon.
 *
 * This header provides a minimal, high-performance C interface for loading
 * .bf2 quantized models and running inference via fused Metal dequant kernels.
 *
 * Design goals:
 *   - Zero-copy: weights loaded via mmap, shared with Metal via MTLBuffer
 *     with MTLStorageModeShared (no copies between CPU and GPU).
 *   - Minimal overhead: direct Metal dispatch without framework overhead.
 *   - Embeddable: single .dylib, easy to link into any project.
 *
 * Example:
 *     #include "opacc1ty.h"
 *
 *     BFEngine *engine = bf_engine_create("model.bf2");
 *     bf_engine_generate(engine, "Hello, world!", 128, stdout);
 *     bf_engine_destroy(engine);
 */

#ifndef OPACC1TY_H
#define OPACC1TY_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types */
typedef struct BFEngine         BFEngine;
typedef struct BFTokenizer      BFTokenizer;
typedef struct BFModelConfig    BFModelConfig;

/**
 * Model configuration inferred from the weight shapes.
 */
struct BFModelConfig {
    char        architecture[64];
    uint32_t    vocab_size;
    uint32_t    hidden_size;
    uint32_t    intermediate_size;
    uint32_t    num_hidden_layers;
    uint32_t    num_attention_heads;
    uint32_t    num_kv_heads;
    uint32_t    max_position_embeddings;
    float       rope_theta;
    uint32_t    head_dim;
};

/**
 * Sampling parameters for text generation.
 */
typedef struct {
    float       temperature;        /* 0.0 = greedy, 1.0 = default */
    float       top_p;              /* 0.0 = disabled */
    int32_t     top_k;              /* 0 = disabled */
    float       repetition_penalty;  /* 1.0 = no penalty */
    uint32_t    seed;               /* Random seed (0 = random) */
} BFSamplingParams;

/**
 * Default sampling parameters (temperature=1.0, no top-p/top-k).
 */
extern const BFSamplingParams BF_DEFAULT_SAMPLING;

/**
 * Create a Opacc1ty inference engine from a .bf2 file.
 *
 * Loads the quantized model weights via mmap, compiles the Metal compute
 * pipeline, and allocates KV cache buffers. Returns NULL on error.
 *
 * @param path  Path to a .bf2 file produced by `opacc1ty quantize`.
 * @return      Engine handle, or NULL on error. Free with bf_engine_destroy().
 */
BFEngine *bf_engine_create(const char *path);

/**
 * Destroy an inference engine and free all resources.
 */
void bf_engine_destroy(BFEngine *engine);

/**
 * Get the model configuration.
 */
const BFModelConfig *bf_engine_get_config(const BFEngine *engine);

/**
 * Get a human-readable error message for the last error on this engine.
 */
const char *bf_engine_get_error(const BFEngine *engine);

/**
 * Get the number of tokens in the engine's vocabulary.
 */
uint32_t bf_engine_vocab_size(const BFEngine *engine);

/**
 * Generate text from a prompt string.
 *
 * This is a convenience function that tokenizes, runs inference autoregressively,
 * and writes decoded tokens to the output stream.
 *
 * @param engine         The inference engine.
 * @param prompt         Input text prompt (UTF-8, null-terminated).
 * @param max_new_tokens Maximum number of tokens to generate.
 * @param params         Sampling parameters (NULL = defaults).
 * @param output         Stream to write generated text to (NULL = no output).
 * @return               Number of tokens generated, or -1 on error.
 */
int32_t bf_engine_generate(
    BFEngine         *engine,
    const char       *prompt,
    uint32_t          max_new_tokens,
    const BFSamplingParams *params,
    FILE             *output
);

/**
 * Run a single forward pass of the model.
 *
 * Low-level API: takes token IDs as input and returns logits for the next token.
 * Caller is responsible for tokenization and sampling.
 *
 * @param engine       The inference engine.
 * @param tokens       Input token IDs (shape: [seq_len]).
 * @param seq_len      Number of input tokens.
 * @param logits_out   Output logits (shape: [vocab_size]). Caller allocates.
 * @param vocab_size   Size of logits_out buffer.
 * @return             0 on success, -1 on error.
 */
int bf_engine_forward(
    BFEngine         *engine,
    const int32_t    *tokens,
    uint32_t          seq_len,
    float            *logits_out,
    uint32_t          vocab_size
);

/**
 * Prefill the KV cache with a prompt (no token generation).
 *
 * Processes the prompt tokens through the model and populates the KV cache
 * so that subsequent calls to bf_engine_forward_step() start from a warm cache.
 *
 * @param engine    The inference engine.
 * @param tokens    Input token IDs (shape: [seq_len]).
 * @param seq_len   Number of input tokens.
 * @return          0 on success, -1 on error.
 */
int bf_engine_prefill(
    BFEngine         *engine,
    const int32_t    *tokens,
    uint32_t          seq_len
);

/**
 * Run a single autoregressive decode step (one token).
 *
 * Uses the KV cache populated by bf_engine_prefill() or previous steps.
 * Only processes the last token; new KV entry is appended to the cache.
 *
 * @param engine       The inference engine.
 * @param token        The last generated token ID.
 * @param logits_out   Output logits (shape: [vocab_size]).
 * @param vocab_size   Size of logits_out buffer.
 * @return             0 on success, -1 on error.
 */
int bf_engine_forward_step(
    BFEngine         *engine,
    int32_t           token,
    float            *logits_out,
    uint32_t          vocab_size
);

/**
 * Reset the KV cache (start a new conversation).
 */
void bf_engine_reset_kv_cache(BFEngine *engine);

/**
 * Get the current KV cache memory usage in bytes.
 */
size_t bf_engine_kv_cache_size(const BFEngine *engine);

/**
 * Get engine statistics since creation or last reset.
 */
typedef struct {
    uint64_t    total_tokens_generated;
    double      total_time_seconds;
    double      tokens_per_second;
    size_t      peak_memory_bytes;
    uint32_t    n_kernel_dispatches;
} BFStats;

void bf_engine_get_stats(const BFEngine *engine, BFStats *stats);
void bf_engine_reset_stats(BFEngine *engine);

/**
 * Get the version string of the Opacc1ty runtime.
 */
const char *bf_version(void);

/**
 * Check if the current system has a compatible Apple Silicon GPU.
 *
 * @return  1 if Metal GPU is available, 0 otherwise.
 */
int bf_has_metal_gpu(void);

#ifdef __cplusplus
}
#endif

#endif /* OPACC1TY_H */
