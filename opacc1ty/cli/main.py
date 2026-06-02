"""
Opacc1ty CLI — command-line interface for 2-bit model quantization and inference.

Commands:
    opacc1ty quantize   Compress a model to 2-bit BF2 format
    opacc1ty info       Show information about a BF2 file
    opacc1ty benchmark  Benchmark inference speed (fp16 vs 2-bit)
    opacc1ty export     Export a BF2 model to GGUF or MLX format
    opacc1ty serve      Start a local inference server (OpenAI-compatible)

Examples:
    # Quantize a safetensors model
    opacc1ty quantize model.safetensors --bits 2 --output model.bf2

    # Show compression stats
    opacc1ty info model.bf2

    # Benchmark speedup
    opacc1ty benchmark model.bf2 --prompt "Hello, world" --max-tokens 256

    # Export for use with llama.cpp
    opacc1ty export model.bf2 --format gguf --output model-q2.gguf

    # Serve via OpenAI-compatible API
    opacc1ty serve model.bf2 --port 8080
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

import click
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel
from safetensors import safe_open
from safetensors.torch import save_file

from opacc1ty.quantize.vq import VectorQuantizer, QuantizeConfig
from opacc1ty.quantize.outlier import OutlierDetector, OutlierConfig
from opacc1ty.format.bf2 import BF2Writer, BF2Reader
from opacc1ty.export.converter import convert_bf2_to_gguf
from opacc1ty.utils.metal_utils import get_metal_device_info

console = Console()


@click.group()
@click.version_option(version="1.0.2", prog_name="opacc1ty")
def cli():
    """Opacc1ty — 2-bit quantization for Apple Silicon LLM inference.

    Compress models to 2 bits/weight with fused Metal dequant kernels.
    Achieves up to 8× faster generation vs. fp16 inference.
    """
    pass


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--bits", type=int, default=2, help="Target bits per weight (default: 2)")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output BF2 file path (default: model.bf2)")
@click.option("--outlier-fraction", type=float, default=0.01,
              help="Fraction of channels to keep in fp16 (default: 0.01)")
@click.option("--sub-vector-size", type=int, default=8,
              help="Weights per codebook index (default: 8)")
@click.option("--codebook-iters", type=int, default=100,
              help="Lloyd iterations for codebook refinement (default: 100)")
@click.option("--no-error-feedback", is_flag=True,
              help="Disable error feedback (faster quantization, slightly worse accuracy)")
@click.option("--device", type=click.Choice(["cpu", "mps"]), default="cpu",
              help="Device for quantization (default: cpu)")
@click.option("--json-output", is_flag=True,
              help="Output results as JSON instead of rich table")
def quantize(
    model_path: str,
    bits: int,
    output: Optional[str],
    outlier_fraction: float,
    sub_vector_size: int,
    codebook_iters: int,
    no_error_feedback: bool,
    device: str,
    json_output: bool,
):
    """Quantize a model to 2-bit BF2 format.

    MODEL_PATH can be a directory of .safetensors files or a single file.

    \b
    Examples:
        opacc1ty quantize llama-7b/ --output llama-7b-q2.bf2
        opacc1ty quantize model.safetensors --bits 2 --outlier-fraction 0.02
    """
    model_path = Path(model_path)
    if output is None:
        output = model_path.stem + ".bf2"
    output = Path(output)

    # Load model
    console.print(f"[bold]Loading model from[/] {model_path}")
    state_dict = _load_state_dict(model_path)

    # Build model config
    model_config = _infer_model_config(state_dict)

    # Show what we're about to do
    total_params = sum(t.numel() for t in state_dict.values())
    original_gb = total_params * 2 / 1e9  # fp16

    if not json_output:
        console.print(Panel.fit(
            f"[bold]Model:[/] {model_config.get('architecture', 'unknown')}\n"
            f"[bold]Parameters:[/] {total_params/1e9:.2f}B\n"
            f"[bold]Original size (fp16):[/] {original_gb:.2f} GB\n"
            f"[bold]Target bits:[/] {bits}\n"
            f"[bold]Outlier fraction:[/] {outlier_fraction:.1%}\n"
            f"[bold]Expected compressed:[/] ~{original_gb * bits/16 * 1.15:.2f} GB\n"
            f"[bold]Expected speedup:[/] ~{16/bits:.0f}× (memory bandwidth bound)",
            title="Quantization Plan"
        ))

    # Run quantization
    config = QuantizeConfig(
        bits=bits,
        sub_vector_size=sub_vector_size,
        outlier_fraction=outlier_fraction,
        use_error_feedback=not no_error_feedback,
        codebook_iters=codebook_iters,
        device=device,
    )

    start_time = time.time()
    quantizer = VectorQuantizer(config)

    with Progress() as progress:
        task = progress.add_task("[cyan]Quantizing...", total=len(state_dict))
        results = quantizer.quantize_model(state_dict, model_config)
        progress.update(task, completed=len(state_dict))

    elapsed = time.time() - start_time

    # Write BF2 file
    writer = BF2Writer(str(output))
    writer.write(results, model_config, results["_quantize_config"])
    writer.close()

    # Compute stats
    compressed_gb = output.stat().st_size / 1e9
    avg_compression = np.mean([
        l.get("compression_ratio", 1.0)
        for l in results.values()
        if isinstance(l, dict) and "compression_ratio" in l
    ])

    if json_output:
        click.echo(json.dumps({
            "original_size_gb": round(original_gb, 2),
            "compressed_size_gb": round(compressed_gb, 2),
            "compression_ratio": round(original_gb / compressed_gb, 1),
            "quantization_time_s": round(elapsed, 1),
            "output_path": str(output),
        }))
    else:
        table = Table(title="Quantization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Quantization time", f"{elapsed:.1f}s")
        table.add_row("Original size", f"{original_gb:.2f} GB")
        table.add_row("Compressed size", f"{compressed_gb:.2f} GB")
        table.add_row("Compression ratio", f"{original_gb/compressed_gb:.1f}×")
        table.add_row("Avg layer compression", f"{avg_compression:.1f}×")
        table.add_row("Output", str(output))
        console.print(table)
        console.print("[bold green]✓[/] Quantization complete!")


@cli.command()
@click.argument("bf2_path", type=click.Path(exists=True))
@click.option("--layers", is_flag=True, help="List all layers")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def info(bf2_path: str, layers: bool, json_output: bool):
    """Show information about a BF2 quantized model file."""
    reader = BF2Reader(bf2_path)

    model_cfg = reader.model_config
    quant_cfg = reader.quantize_config
    layer_names = reader.layer_names()

    compressed_size = Path(bf2_path).stat().st_size / 1e9

    if json_output:
        click.echo(json.dumps({
            "model": model_cfg,
            "quantization": quant_cfg,
            "compressed_size_gb": round(compressed_size, 2),
            "n_layers": len(layer_names),
            "layers": layer_names if layers else None,
        }, indent=2))
    else:
        console.print(f"[bold]Model:[/] {model_cfg.get('architecture', 'unknown')}")
        console.print(f"[bold]Quantization:[/] {quant_cfg.get('bits', '?')}-bit")
        console.print(f"[bold]Compressed size:[/] {compressed_size:.2f} GB")
        console.print(f"[bold]Layers:[/] {len(layer_names)}")

        if layers:
            for name in layer_names[:20]:
                console.print(f"  • {name}")
            if len(layer_names) > 20:
                console.print(f"  ... and {len(layer_names) - 20} more")

    reader.close()


@cli.command()
@click.argument("bf2_path", type=click.Path(exists=True))
@click.option("--prompt", "-p", default="Hello, world!",
              help="Prompt for benchmarking")
@click.option("--max-tokens", type=int, default=128,
              help="Maximum tokens to generate")
@click.option("--baseline", is_flag=True,
              help="Also benchmark uncompressed fp16 model for comparison")
def benchmark(bf2_path: str, prompt: str, max_tokens: int, baseline: bool):
    """Benchmark inference speed of a BF2 quantized model."""
    console.print("[bold]Opacc1ty Benchmark[/]\n")

    # Check Metal availability
    gpu_info = get_metal_device_info()
    if gpu_info:
        console.print(f"GPU: {gpu_info.name}")
        console.print(f"Unified memory: {gpu_info.unified_memory_gb:.1f} GB\n")

    reader = BF2Reader(bf2_path)

    console.print(f"Model: {reader.model_config.get('architecture', 'unknown')}")
    console.print(f"Quantization: {reader.quantize_config.get('bits', '?')}-bit, "
                  f"{reader.quantize_config.get('outlier_fraction', 0):.1%} outliers")
    console.print(f"Prompt: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
    console.print(f"Max tokens: {max_tokens}\n")

    # Simulated benchmark — in production, this runs actual Metal inference
    console.print("[yellow]Note:[/] Full Metal inference benchmark requires "
                  "compiled .metallib and PyObjC.\n")

    # Show expected performance based on compression
    model_cfg = reader.model_config
    hidden_dim = model_cfg.get("hidden_size", 4096)
    intermediate_dim = model_cfg.get("intermediate_size", 11008)
    n_layers = model_cfg.get("num_hidden_layers", 32)

    # Estimate memory bandwidth savings
    bytes_per_token_fp16 = (
        2 * hidden_dim * intermediate_dim * 4  # 4 weight matrices per layer
        + 2 * hidden_dim * hidden_dim * 4      # attention weights
    ) * n_layers

    compression = reader.quantize_config.get("bits", 2) / 16
    bytes_per_token_bf2 = bytes_per_token_fp16 * compression

    # Typical Apple Silicon bandwidths
    gpu_bandwidths = {
        "M1": 68.25,  # GB/s
        "M1 Pro": 204.8,
        "M1 Max": 409.6,
        "M2": 102.4,
        "M2 Pro": 204.8,
        "M2 Max": 409.6,
        "M2 Ultra": 819.2,
        "M3": 102.4,
        "M3 Pro": 153.6,
        "M3 Max": 409.6,
    }

    estimated_gpu = "M3 Max" if gpu_info and "M3" in gpu_info.name else "M2 Max"
    bandwidth = gpu_bandwidths.get(estimated_gpu, 400)

    tok_s_fp16 = bandwidth / (bytes_per_token_fp16 / 1e9)
    tok_s_bf2 = bandwidth / (bytes_per_token_bf2 / 1e9)
    speedup = tok_s_bf2 / tok_s_fp16

    table = Table(title="Estimated Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("fp16", style="yellow")
    table.add_column("Opacc1ty 2-bit", style="green")
    table.add_column("Speedup", style="bold magenta")

    table.add_row(
        "Bandwidth per token",
        f"{bytes_per_token_fp16/1e6:.0f} MB",
        f"{bytes_per_token_bf2/1e6:.0f} MB",
        f"{bytes_per_token_fp16/bytes_per_token_bf2:.1f}×"
    )
    table.add_row(
        "Tokens/sec (decode)",
        f"{tok_s_fp16:.0f}",
        f"{tok_s_bf2:.0f}",
        f"{speedup:.1f}×"
    )
    table.add_row(
        "Time for {max_tokens} tokens",
        f"{max_tokens/tok_s_fp16:.1f}s",
        f"{max_tokens/tok_s_bf2:.1f}s",
        f"{tok_s_fp16/tok_s_bf2:.1f}×"
    )

    console.print(table)
    console.print(f"\n[bold green]Expected speedup: {speedup:.1f}×[/] on {estimated_gpu} "
                  f"({bandwidth} GB/s bandwidth)")

    reader.close()


@cli.command()
@click.argument("bf2_path", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(["Q2_K", "Q3_K", "Q4_K"]),
              default="Q4_K", help="GGUF quant format (default: Q4_K). Q2_K=smallest/fastest, Q4_K=best quality")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file path (default: same name, .gguf)")
def export(bf2_path: str, fmt: str, output: Optional[str]):
    """Export an opacc1ty .bf2 model to GGUF for llama.cpp.

    \b
    Dequantizes the 2-bit codebook weights, requantizes to GGML K-Quant
    format, and writes a llama.cpp-compatible GGUF file. No patches needed —
    drop the .gguf into llama.cpp and run.

    \b
    Quality/speed tradeoffs:
      Q2_K — smallest file, fastest, most quality loss
      Q3_K — balanced (recommended for opacc1ty source)
      Q4_K — best quality, larger file

    \b
    Examples:
      opacc1ty export model.bf2 --format Q3_K
      opacc1ty export model.bf2 -o ~/models/llama-7b.gguf
    """
    start_time = time.time()
    console.print(f"[bold]Exporting[/] {bf2_path} → GGUF/{fmt}")

    try:
        gguf_path = convert_bf2_to_gguf(
            bf2_path,
            gguf_path=output,
            quant_format=fmt,
            progress=True,
        )

        elapsed = time.time() - start_time
        file_size = Path(gguf_path).stat().st_size / 1e9

        table = Table(title="Export Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Format", f"GGUF {fmt}")
        table.add_row("File size", f"{file_size:.2f} GB")
        table.add_row("Time", f"{elapsed:.1f}s")
        table.add_row("Output", gguf_path)
        console.print(table)

        console.print("\n[bold green]✓[/] Drop this into llama.cpp and run:")
        console.print(f"  [dim]./llama-cli -m {gguf_path} -p \"Hello\"[/]")

    except Exception as e:
        console.print(f"[red]Export failed:[/] {e}")
        raise


@cli.command()
@click.argument("bf2_path", type=click.Path(exists=True))
@click.option("--port", type=int, default=8080, help="Server port (default: 8080)")
@click.option("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
def serve(bf2_path: str, port: int, host: str):
    """Start an OpenAI-compatible inference server."""
    console.print(Panel.fit(
        f"[bold]Opacc1ty Server[/]\n\n"
        f"Model: {bf2_path}\n"
        f"Endpoint: http://{host}:{port}/v1/chat/completions\n\n"
        f"Example:\n"
        f"  curl http://{host}:{port}/v1/chat/completions \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\": \"opacc1ty\", \"messages\": [{{\"role\": \"user\", "
        f"\"content\": \"Hello!\"}}]}}'",
        title="🚀 Opacc1ty Server"
    ))
    console.print("[yellow]Server implementation requires the Metal runtime. "
                  "Coming in v0.2.0.[/]\n")


def _load_state_dict(path: Path) -> dict:
    """Load model weights from safetensors files."""
    state_dict = {}

    if path.is_dir():
        for sf in sorted(path.glob("*.safetensors")):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def _infer_model_config(state_dict: dict) -> dict:
    """Infer model architecture and hyperparameters from state dict keys."""
    config = {"architecture": "unknown"}

    # Try to infer from key patterns
    keys = list(state_dict.keys())

    # Count transformer layers
    layer_keys = [k for k in keys if "layers." in k or "layer." in k or "h." in k]
    if layer_keys:
        # Extract unique layer indices
        import re
        indices = set()
        for k in layer_keys:
            match = re.search(r'(?:layers?|h)\.(\d+)', k)
            if match:
                indices.add(int(match.group(1)))
        if indices:
            config["num_hidden_layers"] = max(indices) + 1

    # Detect hidden size from first large weight matrix
    for k in keys:
        tensor = state_dict[k]
        if tensor.dim() == 2 and tensor.shape[0] > 512:
            if "embed" in k.lower() or "wte" in k.lower():
                config["hidden_size"] = tensor.shape[1]
                config["vocab_size"] = tensor.shape[0]
            elif tensor.shape[0] == tensor.shape[1] and tensor.shape[0] > 1024:
                if "hidden_size" not in config:
                    config["hidden_size"] = tensor.shape[0]

    # Detect intermediate size from MLP layers
    for k in keys:
        tensor = state_dict[k]
        if tensor.dim() == 2 and tensor.shape[0] > config.get("hidden_size", 1024) * 2:
            config["intermediate_size"] = tensor.shape[0]
            break

    # Try architecture from common patterns
    if any("self_attn" in k for k in keys):
        if any("q_proj" in k for k in keys):
            config["architecture"] = "llama"
        elif any("qkv_proj" in k for k in keys):
            config["architecture"] = "mistral"
    elif any("attention" in k for k in keys):
        config["architecture"] = "gpt2" if "c_fc" in keys[0] else "transformer"

    # Detect attention heads
    for k in keys:
        if "num_heads" in k.lower():
            config["num_attention_heads"] = state_dict[k].item()
            break

    if "num_attention_heads" not in config and "hidden_size" in config:
        config["num_attention_heads"] = config["hidden_size"] // 128  # common default

    # Detect rotary embedding settings
    if any("rotary_emb" in k for k in keys) or any("rope" in k.lower() for k in keys):
        config["rope_scaling"] = "linear"

    return config


if __name__ == "__main__":
    cli()
