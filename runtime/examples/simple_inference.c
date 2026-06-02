/**
 * simple_inference.c — Minimal Opac1ty inference example.
 *
 * Demonstrates the complete flow: load a .bf2 quantized model,
 * generate text from a prompt, and display performance stats.
 *
 * Build:
 *   cd runtime && mkdir build && cd build
 *   cmake .. && make
 *   ./example_simple ../path/to/model.bf2 "Hello, world!"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opac1ty.h"

int main(int argc, char **argv) {
    const char *model_path = "model.bf2";
    const char *prompt = "Once upon a time";
    uint32_t max_tokens = 100;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) prompt = argv[2];
    if (argc >= 4) max_tokens = (uint32_t)atoi(argv[3]);

    printf("=== Opac1ty Inference Engine v%s ===\n\n", bf_version());

    /* Check GPU availability */
    if (!bf_has_metal_gpu()) {
        printf("Warning: No Apple Silicon GPU detected.\n");
        printf("Opac1ty requires an M-series Mac for full performance.\n\n");
    } else {
        printf("Apple Silicon GPU detected.\n\n");
    }

    /* Load the quantized model */
    printf("Loading model: %s\n", model_path);
    BFEngine *engine = bf_engine_create(model_path);

    const char *err = bf_engine_get_error(engine);
    if (err) {
        fprintf(stderr, "Error loading model: %s\n", err);
        bf_engine_destroy(engine);
        return 1;
    }

    /* Show model info */
    const BFModelConfig *cfg = bf_engine_get_config(engine);
    printf("Model architecture: %s\n", cfg->architecture);
    printf("Parameters: %.1fB\n",
           (float)(cfg->hidden_size * cfg->intermediate_size * cfg->num_hidden_layers * 3) / 1e9);
    printf("Hidden size: %u\n", cfg->hidden_size);
    printf("Layers: %u\n", cfg->num_hidden_layers);
    printf("Vocab size: %u\n", cfg->vocab_size);
    printf("KV cache: %.1f MB\n\n",
           (float)bf_engine_kv_cache_size(engine) / (1024.0 * 1024.0));

    /* Generate text */
    printf("Prompt: \"%s\"\n", prompt);
    printf("Generating %u tokens...\n\n", max_tokens);

    BFSamplingParams params = {
        .temperature = 0.8f,
        .top_p = 0.9f,
        .top_k = 50,
        .repetition_penalty = 1.1f,
        .seed = 42,
    };

    printf("Output: ");
    fflush(stdout);

    int32_t n_generated = bf_engine_generate(
        engine, prompt, max_tokens, &params, stdout
    );

    printf("\n\n");

    if (n_generated < 0) {
        fprintf(stderr, "Generation failed: %s\n", bf_engine_get_error(engine));
        bf_engine_destroy(engine);
        return 1;
    }

    /* Show stats */
    BFStats stats;
    bf_engine_get_stats(engine, &stats);

    printf("=== Results ===\n");
    printf("Tokens generated: %d\n", n_generated);
    printf("Tokens/second: %.1f\n", stats.tokens_per_second);
    printf("Peak memory: %.1f MB\n",
           (float)stats.peak_memory_bytes / (1024.0 * 1024.0));
    printf("Kernel dispatches: %u\n", stats.n_kernel_dispatches);

    bf_engine_destroy(engine);
    return 0;
}
