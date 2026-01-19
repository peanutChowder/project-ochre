#!/usr/bin/env python3
"""
Inference Benchmark for World Models
-------------------------------------
Tests FPS on CPU, GPU, or MPS.

Supports:
- v6.x ConvGRU (WorldModelConvFiLM)
- v7.0 ConvTransformer (MinecraftConvTransformer)
"""

import torch
import time
import argparse
from model_transformer import MinecraftConvTransformer

try:
    from model_convGru import WorldModelConvFiLM  # legacy v6.x
except ModuleNotFoundError:
    WorldModelConvFiLM = None

def benchmark_inference(
    device: str = "cpu",
    hidden_dim: int = 512,
    n_layers: int = 6,
    temporal_context_len: int = 8,
    num_warmup: int = 10,
    num_iterations: int = 100,
    batch_size: int = 1,
):
    """
    Benchmark single-step inference speed.

    This simulates real-time inference where we:
    1. Have a hidden state from previous frame
    2. Get new tokens and action
    3. Predict next frame tokens
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {device.upper()}")
    print(f"Config: hidden_dim={hidden_dim}, n_layers={n_layers}, temporal_ctx={temporal_context_len}")
    print(f"{'='*60}")

    if WorldModelConvFiLM is None:
        raise RuntimeError("WorldModelConvFiLM not available (missing model_convGru.py). Use --model transformer.")

    # Initialize model
    model = WorldModelConvFiLM(
        codebook_size=2048,
        embed_dim=256,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        H=18, W=32,
        action_dim=15,
        temporal_context_len=temporal_context_len,
        use_residuals=True,
    ).to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Create dummy inputs
    Z_t = torch.randint(0, 2048, (batch_size, 18, 32), device=device)
    a_t = torch.randn(batch_size, 15, device=device)
    h = model.init_state(batch_size, device)

    # Pre-fill temporal buffer (simulating steady state)
    temporal_buffer = []
    for _ in range(temporal_context_len):
        temporal_buffer.append(torch.randn(batch_size, hidden_dim, 18, 32, device=device))

    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            logits, h_new = model.step(Z_t, a_t, h, temporal_buffer=temporal_buffer)
            # Update buffer
            temporal_buffer.append(h_new[-1])
            temporal_buffer.pop(0)
            h = h_new

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")

    # Reset state
    h = model.init_state(batch_size, device)
    temporal_buffer = [torch.randn(batch_size, hidden_dim, 18, 32, device=device)
                       for _ in range(temporal_context_len)]

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            logits, h_new = model.step(Z_t, a_t, h, temporal_buffer=temporal_buffer)
            temporal_buffer.append(h_new[-1])
            temporal_buffer.pop(0)
            h = h_new

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start_time

    # Results
    fps = num_iterations / elapsed
    ms_per_frame = (elapsed / num_iterations) * 1000

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  ms/frame: {ms_per_frame:.2f}")
    print(f"  Target (16 FPS): {'PASS' if fps >= 16 else 'FAIL'}")

    return fps, ms_per_frame


def benchmark_transformer(
    device: str = "cpu",
    hidden_dim: int = 384,
    num_layers: int = 4,
    temporal_context_len: int = 4,
    num_warmup: int = 10,
    num_iterations: int = 100,
    batch_size: int = 1,
):
    """
    Benchmark v7.0 ConvTransformer inference speed.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: ConvTransformer on {device.upper()}")
    print(f"Config: hidden_dim={hidden_dim}, num_layers={num_layers}, temporal_ctx={temporal_context_len}")
    print(f"{'='*60}")

    # Initialize model
    model = MinecraftConvTransformer(
        codebook_size=2048,
        embed_dim=256,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=6,
        H=18, W=32,
        action_dim=15,
        temporal_context_len=temporal_context_len,
        window_size=4,
    ).to(device)
    model.eval()

    # Count parameters
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total'] / 1e6:.2f}M")
    print(f"  - Embedding: {param_counts['embed'] / 1e6:.2f}M")
    print(f"  - Stem: {param_counts['stem'] / 1e6:.2f}M")
    print(f"  - Blocks: {param_counts['blocks'] / 1e6:.2f}M")
    print(f"  - Temporal: {param_counts['temporal_attn'] / 1e6:.2f}M")
    print(f"  - IDM: {param_counts['idm'] / 1e6:.2f}M")
    print(f"  - Output: {param_counts['output'] / 1e6:.2f}M")

    # Create dummy inputs
    Z_t = torch.randint(0, 2048, (batch_size, 18, 32), device=device)
    a_t = torch.randn(batch_size, 15, device=device)

    # Pre-fill temporal buffer
    temporal_buffer = []
    compressed_dim = hidden_dim // 2
    # Approximate compressed spatial size: 6x11 = 66 tokens (with stride 3 on 18x32)
    compressed_tokens = (18 // 3) * ((32 + 2) // 3)  # ~60-66 tokens
    for _ in range(temporal_context_len):
        temporal_buffer.append(
            torch.randn(batch_size, compressed_tokens, compressed_dim, device=device)
        )

    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            logits, new_state = model.step(Z_t, a_t, temporal_buffer)
            temporal_buffer.append(new_state)
            temporal_buffer.pop(0)

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")

    # Reset buffer
    temporal_buffer = [
        torch.randn(batch_size, compressed_tokens, compressed_dim, device=device)
        for _ in range(temporal_context_len)
    ]

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            logits, new_state = model.step(Z_t, a_t, temporal_buffer)
            temporal_buffer.append(new_state)
            temporal_buffer.pop(0)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start_time

    # Results
    fps = num_iterations / elapsed
    ms_per_frame = (elapsed / num_iterations) * 1000

    print(f"\nResults:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  FPS: {fps:.1f}")
    print(f"  ms/frame: {ms_per_frame:.2f}")
    print(f"  Target (16 FPS): {'PASS' if fps >= 16 else 'FAIL'}")

    return fps, ms_per_frame


def benchmark_comparison():
    """Compare v6.x ConvGRU vs v7.0 ConvTransformer configurations."""

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON")
    print(f"Device: {device}")
    print("="*60)

    configs = [
        {"name": "v6.0 ConvGRU (384-dim)", "model": "convgru", "hidden_dim": 384, "temporal_context_len": 0},
        {"name": "v6.1 ConvGRU (512-dim, t=8)", "model": "convgru", "hidden_dim": 512, "temporal_context_len": 8},
        {"name": "v7.0 Transformer (384-dim, L=4)", "model": "transformer", "hidden_dim": 384, "num_layers": 4, "temporal_context_len": 4},
        {"name": "v7.0 Transformer (384-dim, L=6)", "model": "transformer", "hidden_dim": 384, "num_layers": 6, "temporal_context_len": 4},
    ]

    results = []

    for cfg in configs:
        print(f"\n>>> Testing: {cfg['name']}")

        if cfg["model"] == "convgru":
            if cfg["temporal_context_len"] == 0:
                fps, ms = benchmark_no_temporal(device, cfg["hidden_dim"])
            else:
                fps, ms = benchmark_inference(
                    device=device,
                    hidden_dim=cfg["hidden_dim"],
                    temporal_context_len=cfg["temporal_context_len"],
                )
        else:  # transformer
            fps, ms = benchmark_transformer(
                device=device,
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg.get("num_layers", 4),
                temporal_context_len=cfg["temporal_context_len"],
            )

        results.append({"name": cfg["name"], "fps": fps, "ms": ms})

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Configuration':<40} {'FPS':>8} {'ms/frame':>10} {'Target':>8}")
    print("-"*68)
    for r in results:
        status = "PASS" if r["fps"] >= 16 else "FAIL"
        print(f"{r['name']:<40} {r['fps']:>8.1f} {r['ms']:>10.2f} {status:>8}")


def benchmark_no_temporal(device: str, hidden_dim: int):
    """Benchmark without temporal attention (for comparison)."""

    if WorldModelConvFiLM is None:
        raise RuntimeError("WorldModelConvFiLM not available (missing model_convGru.py). Use --model transformer.")

    model = WorldModelConvFiLM(
        codebook_size=2048,
        embed_dim=256,
        hidden_dim=hidden_dim,
        n_layers=6,
        H=18, W=32,
        action_dim=15,
        temporal_context_len=8,  # Still init but won't use
        use_residuals=True,
    ).to(device)
    model.eval()

    batch_size = 1
    Z_t = torch.randint(0, 2048, (batch_size, 18, 32), device=device)
    a_t = torch.randn(batch_size, 15, device=device)
    h = model.init_state(batch_size, device)

    num_iterations = 100

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            logits, h = model.step(Z_t, a_t, h, temporal_buffer=None)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            logits, h = model.step(Z_t, a_t, h, temporal_buffer=None)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start_time
    fps = num_iterations / elapsed
    ms = (elapsed / num_iterations) * 1000

    return fps, ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--compare", action="store_true",
                        help="Compare different configurations")
    parser.add_argument("--model", type=str, default="convgru",
                        choices=["convgru", "transformer"],
                        help="Model architecture to benchmark")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Hidden dimension (default: 512 for convgru, 384 for transformer)")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers (transformer only)")
    parser.add_argument("--temporal-ctx", type=int, default=None,
                        help="Temporal context length (default: 8 for convgru, 4 for transformer)")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    if args.compare:
        benchmark_comparison()
    elif args.model == "transformer":
        # Default transformer config
        hidden_dim = args.hidden_dim if args.hidden_dim else 384
        temporal_ctx = args.temporal_ctx if args.temporal_ctx else 4
        benchmark_transformer(
            device=device,
            hidden_dim=hidden_dim,
            num_layers=args.num_layers,
            temporal_context_len=temporal_ctx,
        )
    else:
        # Default: benchmark v6.1 ConvGRU config
        hidden_dim = args.hidden_dim if args.hidden_dim else 512
        temporal_ctx = args.temporal_ctx if args.temporal_ctx else 8
        benchmark_inference(
            device=device,
            hidden_dim=hidden_dim,
            temporal_context_len=temporal_ctx,
        )
