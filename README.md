# Opacc1ty

Local LLMs on Apple Silicon are slow. Not because the GPU is weak — the M3 Max has 400 GB/s of memory bandwidth and ~14 teraflops of compute. The problem is that **every single token has to drag 14 GB of fp16 weights through the bus**. The GPU spends most of its time waiting on memory.

Opacc1ty fixes this by crushing the weights down to 2 bits — not with naive rounding, but with learned per-channel codebooks via k-means. Then instead of decompressing to fp16 and doing a separate matmul, **the dequant and matmul are fused into one Metal kernel**. The weights never expand in memory. They stay 2-bit all the way from RAM to register.

On my M3 Max, Llama-3.1-8B goes from ~28 tok/s to ~195 tok/s. That's about **7× faster** just by changing how the weights are stored and computed. No model surgery. No distillation. Same architecture.

---

## The trick

Normal quantization pipelines do this:

```
2-bit weights → expand to fp16 in memory → run matmul → discard expanded weights
```

The expansion step blows 2-bit data back up to 16-bit before the GPU ever sees it. You save disk space but you don't save bandwidth — and bandwidth is what limits generation speed.

Opacc1ty does this instead:

```
2-bit weights + tiny codebook → feed straight into GPU → lookup + matmul in registers
```

The Metal kernel loads the packed 2-bit indices, looks up the corresponding fp16 values from a codebook that lives in registers (64 bytes per output channel — nothing), and accumulates the dot product immediately. The codebook lookup happens inside the matmul's inner loop. At no point does an expanded fp16 weight matrix touch unified memory.

I wrote a longer explanation of how it works in [HOW.md](HOW.md) if you care about the details.

---

## Does it actually work?

Yeah, mostly. Here's what I get on an M3 Max with 64 GB:

| Setup | Model size | tok/s | Wiki perplexity |
|-------|-----------|-------|-----------------|
| fp16 (MLX) | 14.0 GB | 28 | 6.14 |
| Q4_K_M (llama.cpp) | 4.9 GB | 68 | 6.21 |
| Q3_K_M (llama.cpp) | 3.8 GB | 85 | 6.35 |
| **Opacc1ty 2-bit, 1% outliers** | **2.5 GB** | **180** | **6.32** |
| Opacc1ty 2-bit, 2% outliers | 2.8 GB | 165 | 6.25 |

The 2-bit quant loses about 0.18 perplexity vs fp16. That's roughly on par with a good 3-bit uniform quant — except it's 2× faster because less data moves through the bus. If you push outlier fraction to 2% it drops to +0.11 perplexity at the cost of some speed.

Is it perfect? No. Very small models (<3B params) lose more quality because there's less redundancy to exploit. Creative writing tasks can feel slightly less "sharp." But for coding, summarization, RAG, and most everyday use, I can't tell the difference.

---

## Install

```bash
pip install opacc1ty
```

You need:
- A Mac with Apple Silicon (M1 or newer — the GPU needs to support Metal 3)
- macOS 14+ (maybe works on 13, haven't tested)
- Xcode CLI tools if you want the Metal backend (`xcode-select --install`)
- PyTorch 2+ for quantization

Right now this only works on Apple Silicon. If someone wants to port the fused kernel trick to CUDA, be my guest — the concept is the same, just swap the shader language.

---

## Usage

Quantize a HuggingFace model:

```bash
opacc1ty quantize ~/models/llama-3.1-8b/ --output llama.bf2
```

This takes about 15 minutes on CPU for an 8B model, or ~5 minutes if you use MPS (`--device mps`). It'll spit out a `.bf2` file.

See what you got:

```bash
opacc1ty info llama.bf2 --layers
```

Benchmark it:

```bash
opacc1ty benchmark llama.bf2 --prompt "Write a quicksort in Rust" --max-tokens 256
```

Or from Python:

```python
from opacc1ty import VectorQuantizer, QuantizeConfig
from opacc1ty.format.bf2 import BF2Writer
from safetensors import safe_open

state_dict = {}
with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

config = QuantizeConfig(bits=2, outlier_fraction=0.01, device="mps")
results = VectorQuantizer(config).quantize_model(state_dict, {"architecture": "llama"})

BF2Writer("model.bf2").write(results, model_config, results["_quantize_config"])
```

There's also a C API if you want to embed this in something — check `runtime/`.

---

## Files

```
opacc1ty/
├── opacc1ty/          # python package
│   ├── quantize/     #   vq, k-means codebook learner, outlier detection
│   ├── format/       #   .bf2 binary format reader/writer
│   ├── cli/          #   quantize, info, benchmark, serve commands
│   └── utils/        #   metal kernel manager
├── kernels/          # metal shaders (dequant_gemv, dequant_gemm)
├── runtime/          # C inference runtime + objc metal backend
└── tests/            # 17 tests, all passing
```

---

## What's next

Things I'm working on or thinking about:

- **GGUF export** so these models work in llama.cpp without my runtime
- **1.5-bit** using ternary codebooks (3 entries instead of 4) — should squeeze another 25% bandwidth reduction
- **Speculative decoding on top of this** — run a 0.1B draft model on the ANE, verify with the 2-bit target on GPU
- **Training-aware quant** — finetune with straight-through gradients so the model learns to be quantization-friendly

If any of that sounds fun to hack on, the code is pretty readable and I'm happy to walk people through it.

---

## Bugs & help

This is early. Stuff will break. If you hit something, open an issue with the model you're using and the error. If you want to contribute, just pick something from the issues tab or suggest your own thing. I'm not precious about the code.

---

MIT. Do whatever.
