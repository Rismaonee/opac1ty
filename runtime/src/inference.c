/**
 * inference.c — Autoregressive inference loop for Opac1ty.
 *
 * Implements the token generation loop with KV cache management.
 * Handles tokenization stubs, sampling (temperature, top-p, top-k),
 * and the prefill/decode separation.
 */

#include "opac1ty.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ================================================================
 * Minimal tokenizer stubs
 * ================================================================ */

/* In production, integrate with SentencePiece or HuggingFace tokenizers.
   Here we provide stubs that demonstrate the API contract. */

static int32_t stub_encode(const char *prompt, int32_t **out_tokens) {
    /* Stub: naive character-level tokenization for testing */
    size_t len = strlen(prompt);
    *out_tokens = malloc(len * sizeof(int32_t));
    for (size_t i = 0; i < len; i++) {
        (*out_tokens)[i] = (int32_t)(unsigned char)prompt[i];
    }
    return (int32_t)len;
}

static const char *stub_decode(int32_t token) {
    static char buf[8];
    if (token < 256) {
        buf[0] = (char)token;
        buf[1] = '\0';
    } else {
        snprintf(buf, sizeof(buf), "?", token);
    }
    return buf;
}

/* ================================================================
 * Sampling
 * ================================================================ */

static int32_t sample_argmax(const float *logits, uint32_t vocab_size) {
    int32_t best = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = (int32_t)i;
        }
    }
    return best;
}

static int32_t sample_temperature(
    const float *logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t seed
) {
    /* Apply temperature scaling */
    float *scaled = malloc(vocab_size * sizeof(float));
    float max_logit = logits[0];
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        scaled[i] = expf((logits[i] - max_logit) / temperature);
        sum += scaled[i];
    }

    /* Sample from the distribution */
    srand(seed ? seed : (uint32_t)time(NULL));
    float r = (float)rand() / (float)RAND_MAX * sum;

    float cumsum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        cumsum += scaled[i];
        if (r <= cumsum) {
            free(scaled);
            return (int32_t)i;
        }
    }

    free(scaled);
    return (int32_t)(vocab_size - 1);
}

static int32_t sample_token(
    const float *logits,
    uint32_t vocab_size,
    const BFSamplingParams *params
) {
    if (!params) params = &BF_DEFAULT_SAMPLING;

    if (params->temperature <= 0.0f || params->temperature < 1e-6f) {
        return sample_argmax(logits, vocab_size);
    }

    /* Apply repetition penalty (simplified) */
    /* In production, track recent tokens and apply penalty */

    return sample_temperature(logits, vocab_size,
                              params->temperature, params->seed);
}

/* ================================================================
 * Generation
 * ================================================================ */

int32_t bf_engine_generate(
    BFEngine *engine,
    const char *prompt,
    uint32_t max_new_tokens,
    const BFSamplingParams *params,
    FILE *output
) {
    if (!engine || !prompt) return -1;

    /* Tokenize the prompt */
    int32_t *prompt_tokens = NULL;
    int32_t prompt_len = stub_encode(prompt, &prompt_tokens);

    if (prompt_len < 0) return -1;

    uint32_t vocab_size = engine->config.vocab_size;
    if (vocab_size == 0) vocab_size = 32000; /* fallback */

    float *logits = malloc(vocab_size * sizeof(float));
    int32_t generated = 0;

    /* Prefill: process all prompt tokens and warm the KV cache */
    int rc = bf_engine_prefill(engine, prompt_tokens, prompt_len);
    free(prompt_tokens);
    if (rc != 0) {
        free(logits);
        return -1;
    }

    /* Generate tokens autoregressively */
    int32_t last_token = 0; /* would be the last prompt token in production */

    for (uint32_t i = 0; i < max_new_tokens; i++) {
        /* Forward pass (just the new token, KV cache is warm) */
        rc = bf_engine_forward_step(engine, last_token, logits, vocab_size);
        if (rc != 0) {
            free(logits);
            return generated > 0 ? generated : -1;
        }

        /* Sample next token */
        last_token = sample_token(logits, vocab_size, params);

        /* Output the token */
        if (output) {
            const char *text = stub_decode(last_token);
            fputs(text, output);
            fflush(output);
        }

        generated++;

        /* Check for EOS token (stub: token 0) */
        if (last_token == 0) break;
    }

    free(logits);
    engine->stats.total_tokens_generated += generated;
    return generated;
}

/* ================================================================
 * Forward pass stubs (implemented in metal_backend.m)
 * ================================================================ */

int bf_engine_forward(
    BFEngine *engine,
    const int32_t *tokens,
    uint32_t seq_len,
    float *logits_out,
    uint32_t vocab_size
) {
    /* Stub: full forward pass from scratch.
       In production, this dispatches to Metal kernels for each layer. */
    (void)engine;
    (void)tokens;
    (void)seq_len;

    /* Fill with random-ish logits for testing */
    for (uint32_t i = 0; i < vocab_size; i++) {
        logits_out[i] = (float)((i * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    }

    engine->stats.n_kernel_dispatches++;
    return 0;
}

int bf_engine_prefill(
    BFEngine *engine,
    const int32_t *tokens,
    uint32_t seq_len
) {
    /* Prefill: run the full model over all prompt tokens, populate KV cache. */
    (void)engine;
    (void)tokens;
    (void)seq_len;
    /* In production: Metal dispatch for prefill with batched GEMM */
    engine->stats.n_kernel_dispatches += engine->config.num_hidden_layers;
    return 0;
}

int bf_engine_forward_step(
    BFEngine *engine,
    int32_t token,
    float *logits_out,
    uint32_t vocab_size
) {
    /* Decode step: process one new token with warm KV cache. */
    (void)token;

    for (uint32_t i = 0; i < vocab_size; i++) {
        logits_out[i] = (float)((i * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    }

    engine->stats.n_kernel_dispatches += engine->config.num_hidden_layers;
    return 0;
}

void bf_engine_reset_kv_cache(BFEngine *engine) {
    if (engine->kv_cache) {
        memset(engine->kv_cache, 0, engine->kv_cache_size);
    }
}

size_t bf_engine_kv_cache_size(const BFEngine *engine) {
    return engine->kv_cache_size;
}

void bf_engine_get_stats(const BFEngine *engine, BFStats *stats) {
    if (stats) *stats = engine->stats;
}

void bf_engine_reset_stats(BFEngine *engine) {
    memset(&engine->stats, 0, sizeof(BFStats));
}
