/**
 * bf2_format.c — BF2 binary format parser for the C runtime.
 *
 * Parses .bf2 files directly into memory-mapped Metal buffers.
 * Handles the three layer types (quantized, sparse_outlier, fp16_pass-through)
 * and provides zero-copy access to codebooks, indices, and outlier data.
 *
 * File format (see opacc1ty/format/header.py for full spec):
 *   [Magic:4] [Version:4] [Flags:4] [ModelJSONLen:4]
 *   [ModelJSON:N] [QuantConfigJSONLen:4] [QuantConfigJSON:M]
 *   [LayerCount:4] [LayerHeaders...] [LayerData...] [Checksum:8]
 */

#include "opacc1ty.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

/* Magic bytes for BF2 file identification */
#define BF2_MAGIC       "BF2\x00"
#define BF2_VERSION     1
#define BF2_MAGIC_LEN   4

/* Layer header size */
#define LAYER_HEADER_SIZE 64

/* Layer types */
enum {
    LAYER_QUANTIZED = 0,
    LAYER_SPARSE_OUTLIER = 1,
    LAYER_FP16_PASSTHROUGH = 2,
};

/* Internal structures */

typedef struct {
    uint16_t    name_len;
    uint8_t     layer_type;
    uint32_t    out_features;
    uint32_t    in_features;
    uint32_t    n_groups;
    uint8_t     codebook_entries;
    uint8_t     sub_vector_size;
    uint16_t    n_outliers;
    uint64_t    cb_offset;       /* from layer header start */
    uint64_t    idx_offset;      /* from layer header start */
    uint64_t    out_val_offset;  /* from layer header start */
    uint64_t    out_idx_offset;  /* from layer header start */
    uint8_t     _reserved[14];
} __attribute__((packed)) LayerHeader;

typedef struct {
    char           *name;
    LayerHeader     header;
    uint8_t        *layer_data_start;  /* pointer into mmap'd file */
} LayerInfo;

struct BFEngine {
    /* File mapping */
    int             fd;
    uint8_t        *mapped_data;
    size_t          file_size;

    /* Model metadata */
    BFModelConfig   config;
    char           *model_json;
    char           *quant_json;

    /* Layer index */
    uint32_t        n_layers;
    LayerInfo      *layers;

    /* Metal runtime (opaque, defined in metal_backend.m) */
    void           *metal_ctx;

    /* KV cache */
    void           *kv_cache;
    size_t          kv_cache_size;

    /* Statistics */
    BFStats         stats;

    /* Error message */
    char            error[256];
};

/* JSON parsing helpers (minimal, no external dependency) */

static const char *json_get_string(const char *json, const char *key) {
    /* Simple key lookup: find "key": "value" */
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return NULL;

    pos = strchr(pos + strlen(search), '"');
    if (!pos) return NULL;
    pos++; /* skip opening quote */

    static char value[256];
    const char *end = strchr(pos, '"');
    if (!end) return NULL;

    size_t len = end - pos;
    if (len >= sizeof(value)) len = sizeof(value) - 1;
    memcpy(value, pos, len);
    value[len] = '\0';
    return value;
}

static double json_get_number(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return 0.0;

    pos = strchr(pos + strlen(search), ':');
    if (!pos) return 0.0;
    pos++;

    while (*pos == ' ' || *pos == '\t') pos++;
    return strtod(pos, NULL);
}

/* Forward declarations */
static int parse_model_config(BFEngine *e, const char *json, size_t len);
static int build_layer_index(BFEngine *e);

/* Public API implementation */

const BFSamplingParams BF_DEFAULT_SAMPLING = {
    .temperature = 1.0f,
    .top_p = 0.0f,
    .top_k = 0,
    .repetition_penalty = 1.0f,
    .seed = 0,
};

BFEngine *bf_engine_create(const char *path) {
    BFEngine *e = calloc(1, sizeof(BFEngine));
    if (!e) return NULL;

    /* Open and memory-map the BF2 file */
    e->fd = open(path, O_RDONLY);
    if (e->fd < 0) {
        snprintf(e->error, sizeof(e->error),
                 "Cannot open file: %s", path);
        return e; /* caller checks error */
    }

    struct stat st;
    fstat(e->fd, &st);
    e->file_size = st.st_size;

    e->mapped_data = mmap(NULL, e->file_size, PROT_READ,
                          MAP_PRIVATE, e->fd, 0);
    if (e->mapped_data == MAP_FAILED) {
        snprintf(e->error, sizeof(e->error),
                 "Cannot mmap file: %s", path);
        close(e->fd);
        return e;
    }

    /* Parse magic and version */
    if (memcmp(e->mapped_data, BF2_MAGIC, BF2_MAGIC_LEN) != 0) {
        snprintf(e->error, sizeof(e->error),
                 "Not a valid BF2 file (bad magic)");
        return e;
    }

    uint32_t version = *(uint32_t *)(e->mapped_data + 4);
    if (version != BF2_VERSION) {
        snprintf(e->error, sizeof(e->error),
                 "Unsupported BF2 version: %u (expected %u)",
                 version, BF2_VERSION);
        return e;
    }

    /* Parse model config JSON */
    uint32_t flags = *(uint32_t *)(e->mapped_data + 8);
    (void)flags; /* reserved for future use */

    size_t pos = 12; /* after magic + version + flags */

    /* Model JSON length appears twice for alignment */
    uint32_t model_json_len = *(uint32_t *)(e->mapped_data + pos);
    pos += 4;

    /* Re-read the actual length (it's stored once, but we read twice due
       to alignment in the Python writer — align to what the writer does) */
    model_json_len = *(uint32_t *)(e->mapped_data + pos);
    pos += 4;

    /* Copy model JSON */
    e->model_json = malloc(model_json_len + 1);
    memcpy(e->model_json, e->mapped_data + pos, model_json_len);
    e->model_json[model_json_len] = '\0';
    pos += model_json_len;

    /* Quant config JSON */
    uint32_t quant_json_len = *(uint32_t *)(e->mapped_data + pos);
    pos += 4;
    e->quant_json = malloc(quant_json_len + 1);
    memcpy(e->quant_json, e->mapped_data + pos, quant_json_len);
    e->quant_json[quant_json_len] = '\0';
    pos += quant_json_len;

    /* Parse model config from JSON */
    parse_model_config(e, e->model_json, model_json_len);

    /* Layer count */
    e->n_layers = *(uint32_t *)(e->mapped_data + pos);
    pos += 4;

    /* Build layer index */
    e->_data_start = pos;
    if (build_layer_index(e) != 0) {
        snprintf(e->error, sizeof(e->error),
                 "Failed to build layer index");
        return e;
    }

    /* Store data start for layer parsing */
    /* (hack: reuse pos tracking for layer offset calculation) */

    /* Initialize Metal backend if available */
    /* e->metal_ctx = metal_backend_init(e); */

    return e;
}

void bf_engine_destroy(BFEngine *engine) {
    if (!engine) return;

    if (engine->mapped_data) {
        munmap(engine->mapped_data, engine->file_size);
    }
    if (engine->fd >= 0) {
        close(engine->fd);
    }

    free(engine->model_json);
    free(engine->quant_json);

    if (engine->layers) {
        for (uint32_t i = 0; i < engine->n_layers; i++) {
            free(engine->layers[i].name);
        }
        free(engine->layers);
    }

    /* Free Metal context */
    /* metal_backend_destroy(engine->metal_ctx); */

    free(engine);
}

const BFModelConfig *bf_engine_get_config(const BFEngine *engine) {
    return &engine->config;
}

const char *bf_engine_get_error(const BFEngine *engine) {
    return engine->error[0] ? engine->error : NULL;
}

uint32_t bf_engine_vocab_size(const BFEngine *engine) {
    return engine->config.vocab_size;
}

const char *bf_version(void) {
    return "0.1.0";
}

int bf_has_metal_gpu(void) {
    /* Check if Metal is available by testing for the framework */
    void *handle = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                          RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return 1;
    }
    return 0;
}

/* Private helpers */

static int parse_model_config(BFEngine *e, const char *json, size_t len) {
    BFModelConfig *cfg = &e->config;

    const char *arch = json_get_string(json, "architecture");
    if (arch) {
        strncpy(cfg->architecture, arch, sizeof(cfg->architecture) - 1);
    } else {
        strcpy(cfg->architecture, "unknown");
    }

    cfg->vocab_size = (uint32_t)json_get_number(json, "vocab_size");
    cfg->hidden_size = (uint32_t)json_get_number(json, "hidden_size");
    cfg->intermediate_size = (uint32_t)json_get_number(json, "intermediate_size");
    cfg->num_hidden_layers = (uint32_t)json_get_number(json, "num_hidden_layers");
    cfg->num_attention_heads = (uint32_t)json_get_number(json, "num_attention_heads");
    cfg->num_kv_heads = (uint32_t)json_get_number(json, "num_kv_heads");
    cfg->max_position_embeddings = (uint32_t)json_get_number(json, "max_position_embeddings");
    cfg->rope_theta = (float)json_get_number(json, "rope_theta");

    if (cfg->num_kv_heads == 0) cfg->num_kv_heads = cfg->num_attention_heads;
    if (cfg->hidden_size > 0 && cfg->num_attention_heads > 0) {
        cfg->head_dim = cfg->hidden_size / cfg->num_attention_heads;
    }

    return 0;
}

static int build_layer_index(BFEngine *e) {
    e->layers = calloc(e->n_layers, sizeof(LayerInfo));
    if (!e->layers) return -1;

    size_t pos = e->_data_start;

    for (uint32_t i = 0; i < e->n_layers; i++) {
        LayerInfo *li = &e->layers[i];

        /* Read layer header */
        memcpy(&li->header, e->mapped_data + pos, LAYER_HEADER_SIZE);

        /* Read layer name */
        li->name = malloc(li->header.name_len + 1);
        memcpy(li->name,
               e->mapped_data + pos + LAYER_HEADER_SIZE,
               li->header.name_len);
        li->name[li->header.name_len] = '\0';

        /* Point to layer data (after header + name) */
        li->layer_data_start = e->mapped_data + pos;

        /* Compute next layer position */
        pos += LAYER_HEADER_SIZE + li->header.name_len;

        if (li->header.layer_type == LAYER_QUANTIZED) {
            pos += li->header.n_groups * li->header.codebook_entries
                   * li->header.sub_vector_size * 2;  /* codebooks (fp16) */
            pos += li->header.n_groups * li->header.sub_vector_size;  /* indices */
            if (li->header.n_outliers > 0) {
                pos += li->header.n_outliers * li->header.in_features * 2;
                pos += li->header.n_outliers * 4;
            }
        } else if (li->header.layer_type == LAYER_FP16_PASSTHROUGH) {
            pos += li->header.out_features * li->header.in_features * 2;
        }
    }

    return 0;
}
