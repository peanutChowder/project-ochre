#!/usr/bin/env python3
"""
Debug Tool 3: Temporal Buffer Ablation Evaluation

Tests whether the temporal buffer is helping, ignored, or actively harmful
by running the same rollout under controlled buffer variants:

  - full:          normal primed buffer (baseline)
  - empty:         no buffer at all
  - last_only:     only the most recent buffer state
  - truncated_N:   keep only the N most recent states (N = 1, 2, 4)
  - shuffled:      full buffer with order randomized
  - noise_corrupt: full buffer with Gaussian noise added to all states
  - zero_oldest:   zero out the oldest half of buffer states
  - zero_newest:   zero out the newest half of buffer states

Per-frame metrics track how each ablation diverges from the full-buffer
baseline, revealing whether the buffer is useful, weak, or amplifying error.

Also includes a compact "buffer usage summary" that shows:
  - performance vs buffer length
  - sensitivity to dropping each slot
  - relative importance of newest vs oldest slots

Usage:
    python eval/eval_tools/debug/temporal_buffer_ablation_eval.py \
        --checkpoint ./checkpoints/ochre-v7.5.3-step240k.pt \
        --vqvae_ckpt ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --output_dir ./eval/runs/debug/buffer-ablation/v7.5.3-240k/
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_tools.shared import (
    ensure_dir,
    load_model_bundle,
    resolve_context_paths,
    save_json,
    select_device,
)
from diagnostics.visual_quality_eval import _edge_map

try:
    import lpips as _lpips_mod

    def _make_lpips(device: torch.device):
        return _lpips_mod.LPIPS(net="alex").to(device).eval()
except ImportError:
    _make_lpips = None


# ---------------------------------------------------------------------------
# Buffer manipulation helpers
# ---------------------------------------------------------------------------

def _make_buffer_variants(full_buffer: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """Create all buffer variants from the primed full buffer."""
    n = len(full_buffer)
    variants = {"full": list(full_buffer)}

    # Empty
    variants["empty"] = []

    # Last state only
    if n > 0:
        variants["last_only"] = [full_buffer[-1]]

    # Truncated: keep N most recent
    for k in [1, 2, 4]:
        if k < n:
            variants[f"truncated_{k}"] = list(full_buffer[-k:])

    # Shuffled order
    if n >= 2:
        perm = torch.randperm(n).tolist()
        variants["shuffled"] = [full_buffer[i] for i in perm]

    # Noise-corrupted: add Gaussian noise (std = 0.5 * per-state std)
    if n > 0:
        noisy = []
        for state in full_buffer:
            std = state.std().item()
            noise = torch.randn_like(state) * (0.5 * std)
            noisy.append(state + noise)
        variants["noise_corrupt"] = noisy

    # Zero oldest half
    if n >= 2:
        half = n // 2
        zeroed = []
        for i, state in enumerate(full_buffer):
            zeroed.append(torch.zeros_like(state) if i < half else state.clone())
        variants["zero_oldest"] = zeroed

    # Zero newest half
    if n >= 2:
        half = n // 2
        zeroed = []
        for i, state in enumerate(full_buffer):
            zeroed.append(torch.zeros_like(state) if i >= n - half else state.clone())
        variants["zero_newest"] = zeroed

    return variants


def _make_drop_one_variants(full_buffer: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """Create variants that drop exactly one slot, for per-slot sensitivity."""
    n = len(full_buffer)
    variants = {}
    for drop_idx in range(n):
        key = f"drop_slot_{drop_idx}"
        variants[key] = [s for i, s in enumerate(full_buffer) if i != drop_idx]
    return variants


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_with_buffer(
    model,
    vqvae,
    z_start: torch.Tensor,
    action_seq: torch.Tensor,
    gt_tokens: torch.Tensor,
    temporal_buffer: list[torch.Tensor],
    topk: int = 50,
    temperature: float = 1.0,
):
    """Run AR rollout with the given buffer, returning per-frame metrics and predictions."""
    buf = list(temporal_buffer)
    z_t = z_start
    num_steps = action_seq.shape[0]

    per_frame = {
        "token_accuracy": [],
        "argmax_unique": [],
        "mean_max_prob": [],
        "mean_entropy": [],
    }
    pred_tokens_list = []
    pred_rgb_list = []

    for t in range(num_steps):
        logits, new_state = model.step(z_t, action_seq[t:t + 1], buf)
        probs = F.softmax(logits / temperature, dim=1)

        greedy = logits.argmax(dim=1)
        per_frame["argmax_unique"].append(int(greedy.unique().numel()))
        per_frame["mean_max_prob"].append(float(probs.max(dim=1)[0].mean()))

        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        per_frame["mean_entropy"].append(float(entropy.mean()))

        gt_t = gt_tokens[t:t + 1]
        per_frame["token_accuracy"].append(float((greedy == gt_t).float().mean()))

        # Sample next tokens
        if topk > 0:
            topk_probs, topk_idx = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
            b, _, h, w = topk_probs.shape
            flat_probs = topk_probs.permute(0, 2, 3, 1).reshape(b * h * w, topk)
            flat_idx = topk_idx.permute(0, 2, 3, 1).reshape(b * h * w, topk)
            if flat_probs.device.type == "mps":
                chosen_rel = torch.multinomial(flat_probs.cpu(), 1).to(flat_probs.device)
            else:
                chosen_rel = torch.multinomial(flat_probs, 1)
            z_sampled = flat_idx[
                torch.arange(b * h * w, device=flat_idx.device), chosen_rel.flatten()
            ].reshape(b, h, w)
        else:
            z_sampled = greedy

        pred_tokens_list.append(z_sampled.detach().clone())
        pred_rgb_list.append(vqvae.decode_code(z_sampled))

        buf.append(new_state.detach())
        if len(buf) > getattr(model, "temporal_context_len", 8):
            buf.pop(0)

        z_t = z_sampled

    return {
        "per_frame": per_frame,
        "pred_tokens": torch.stack(pred_tokens_list, dim=0),
        "pred_rgb": torch.cat(pred_rgb_list, dim=0),
    }


# ---------------------------------------------------------------------------
# Divergence from baseline
# ---------------------------------------------------------------------------

def _compute_divergence_from_baseline(
    baseline: dict,
    ablated: dict,
    gt_rgb: torch.Tensor,
    lpips_fn,
) -> dict:
    """Compare ablated rollout to full-buffer baseline, frame-by-frame."""
    base_rgb = baseline["pred_rgb"]
    abl_rgb = ablated["pred_rgb"]
    base_tokens = baseline["pred_tokens"]
    abl_tokens = ablated["pred_tokens"]
    n = base_rgb.shape[0]

    div = {
        "token_agreement_vs_baseline": [],
        "psnr_vs_gt_baseline": [],
        "psnr_vs_gt_ablated": [],
        "lpips_vs_baseline": [],
        "edge_l1_vs_baseline": [],
    }

    for i in range(n):
        agree = float((base_tokens[i] == abl_tokens[i]).float().mean())
        div["token_agreement_vs_baseline"].append(agree)

        mse_base = (base_rgb[i:i+1] - gt_rgb[i:i+1]).square().mean().detach()
        mse_abl = (abl_rgb[i:i+1] - gt_rgb[i:i+1]).square().mean().detach()
        div["psnr_vs_gt_baseline"].append(float(10 * torch.log10(1.0 / (mse_base + 1e-8))))
        div["psnr_vs_gt_ablated"].append(float(10 * torch.log10(1.0 / (mse_abl + 1e-8))))

        if lpips_fn is not None:
            base_norm = base_rgb[i:i+1] * 2.0 - 1.0
            abl_norm = abl_rgb[i:i+1] * 2.0 - 1.0
            div["lpips_vs_baseline"].append(float(lpips_fn(base_norm, abl_norm).detach()))

        base_edge = _edge_map(base_rgb[i:i+1])
        abl_edge = _edge_map(abl_rgb[i:i+1])
        div["edge_l1_vs_baseline"].append(float((base_edge - abl_edge).abs().mean()))

    return div


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _variant_summary(per_frame: dict, divergence: dict | None, label: str) -> dict:
    """Compute compact summary for one variant."""
    summary = {}
    for key, vals in per_frame.items():
        arr = np.array(vals)
        summary[f"{label}_{key}_mean"] = float(arr.mean())
        summary[f"{label}_{key}_last"] = float(arr[-1])

    if divergence:
        agree = np.array(divergence["token_agreement_vs_baseline"])
        summary[f"{label}_agreement_vs_full_mean"] = float(agree.mean())
        summary[f"{label}_agreement_vs_full_first"] = float(agree[0])
        summary[f"{label}_agreement_vs_full_last"] = float(agree[-1])

        psnr_base = np.array(divergence["psnr_vs_gt_baseline"])
        psnr_abl = np.array(divergence["psnr_vs_gt_ablated"])
        psnr_delta = psnr_abl - psnr_base
        summary[f"{label}_psnr_delta_mean"] = float(psnr_delta.mean())
        summary[f"{label}_psnr_delta_last"] = float(psnr_delta[-1])

    return summary


def _buffer_usage_summary(variant_summaries: dict, drop_summaries: dict, full_buffer_len: int) -> dict:
    """Emit the compact buffer usage summary from the spec."""
    usage = {
        "full_buffer_len": full_buffer_len,
    }

    # Performance vs buffer length
    length_perf = {}
    length_map = {
        "empty": 0,
        "truncated_1": 1,
        "last_only": 1,
        "truncated_2": 2,
        "truncated_4": 4,
        "full": full_buffer_len,
    }
    for variant, length in length_map.items():
        key = f"{variant}_token_accuracy_mean"
        if key in variant_summaries:
            length_perf[str(length)] = variant_summaries[key]
    usage["token_accuracy_by_buffer_length"] = length_perf

    # Sensitivity to dropping each slot
    slot_sensitivity = {}
    full_acc = variant_summaries.get("full_token_accuracy_mean", 0)
    for slot_idx in range(full_buffer_len):
        drop_key = f"drop_slot_{slot_idx}_token_accuracy_mean"
        if drop_key in drop_summaries:
            delta = drop_summaries[drop_key] - full_acc
            slot_sensitivity[f"slot_{slot_idx}_acc_delta"] = float(delta)
    usage["per_slot_sensitivity"] = slot_sensitivity

    # Newest vs oldest importance
    oldest_key = "zero_oldest_agreement_vs_full_mean"
    newest_key = "zero_newest_agreement_vs_full_mean"
    if oldest_key in variant_summaries and newest_key in variant_summaries:
        oldest_agree = variant_summaries[oldest_key]
        newest_agree = variant_summaries[newest_key]
        usage["zero_oldest_agreement"] = oldest_agree
        usage["zero_newest_agreement"] = newest_agree
        # Higher agreement = less impact of zeroing → that half is less important
        if oldest_agree > newest_agree:
            usage["more_important_half"] = "newest"
        elif newest_agree > oldest_agree:
            usage["more_important_half"] = "oldest"
        else:
            usage["more_important_half"] = "equal"

    # Overall buffer importance
    empty_agree = variant_summaries.get("empty_agreement_vs_full_mean")
    if empty_agree is not None:
        usage["empty_vs_full_agreement"] = empty_agree
        if empty_agree > 0.9:
            usage["buffer_verdict"] = "ignored"
        elif empty_agree > 0.7:
            usage["buffer_verdict"] = "weakly_used"
        else:
            usage["buffer_verdict"] = "actively_used"

    # Shuffle sensitivity
    shuffled_agree = variant_summaries.get("shuffled_agreement_vs_full_mean")
    if shuffled_agree is not None:
        usage["shuffled_vs_full_agreement"] = shuffled_agree
        usage["order_sensitive"] = shuffled_agree < 0.85

    # Noise sensitivity
    noise_agree = variant_summaries.get("noise_corrupt_agreement_vs_full_mean")
    if noise_agree is not None:
        usage["noise_corrupt_vs_full_agreement"] = noise_agree
        usage["noise_sensitive"] = noise_agree < 0.7

    return usage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal buffer ablation evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--context_npz", nargs="+")
    parser.add_argument("--data_dir", default="./preprocessedv5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=8)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_drop_one", action="store_true",
                        help="Skip per-slot drop-one ablations (faster)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    model, vqvae, _ = load_model_bundle(args.checkpoint, args.vqvae_ckpt, device)
    lpips_fn = _make_lpips(device) if _make_lpips else None
    contexts = resolve_context_paths(args.context_npz, args.data_dir)
    out_dir = ensure_dir(args.output_dir)

    all_results = []

    for context_path in contexts:
        ctx_name = Path(context_path).stem
        print(f"\n{'='*60}")
        print(f"Context: {ctx_name}")
        print(f"{'='*60}")

        # Load context and prime full buffer
        data = np.load(context_path)
        tokens_all = torch.from_numpy(data["tokens"]).long().to(device)
        actions_all = torch.from_numpy(data["actions"]).float().to(device)

        end_idx = args.start_idx + args.rollout_steps + 1
        tokens = tokens_all[args.start_idx:end_idx]
        actions = actions_all[args.start_idx:args.start_idx + args.rollout_steps]
        gt_tokens = tokens[1:]
        gt_rgb = vqvae.decode_code(gt_tokens)
        z_start = tokens[0:1]

        # Prime temporal buffer
        prime_len = min(getattr(model, "temporal_context_len", 8), args.start_idx)
        temporal_buffer = []
        if prime_len > 0:
            from diagnostics.visual_quality_eval import prime_temporal_buffer
            hist_tokens = tokens_all[args.start_idx - prime_len:args.start_idx + 1]
            hist_actions = actions_all[args.start_idx - prime_len:args.start_idx]
            temporal_buffer = prime_temporal_buffer(model, hist_tokens, hist_actions)

        full_buffer_len = len(temporal_buffer)
        print(f"  Full buffer length: {full_buffer_len}")

        if full_buffer_len == 0:
            print("  WARNING: Buffer is empty, ablation is trivial. Skipping.")
            continue

        # Create buffer variants
        torch.manual_seed(args.seed)  # deterministic noise/shuffle
        variants = _make_buffer_variants(temporal_buffer)

        # Run baseline (full buffer) first
        print(f"  Running full (baseline) rollout...")
        torch.manual_seed(args.seed)
        baseline = _rollout_with_buffer(
            model, vqvae, z_start, actions, gt_tokens,
            variants["full"], topk=args.topk, temperature=args.temperature,
        )

        # Run each variant
        variant_results = {"full": {"per_frame": baseline["per_frame"], "divergence": None}}
        variant_summaries = _variant_summary(baseline["per_frame"], None, "full")

        for vname, vbuf in variants.items():
            if vname == "full":
                continue
            print(f"  Running {vname} rollout (buffer len={len(vbuf)})...")
            torch.manual_seed(args.seed)
            result = _rollout_with_buffer(
                model, vqvae, z_start, actions, gt_tokens,
                vbuf, topk=args.topk, temperature=args.temperature,
            )
            divergence = _compute_divergence_from_baseline(baseline, result, gt_rgb, lpips_fn)
            variant_results[vname] = {"per_frame": result["per_frame"], "divergence": divergence}
            variant_summaries.update(_variant_summary(result["per_frame"], divergence, vname))

        # Per-slot drop-one ablations
        drop_summaries = {}
        if not args.skip_drop_one and full_buffer_len >= 2:
            drop_variants = _make_drop_one_variants(temporal_buffer)
            for dname, dbuf in drop_variants.items():
                print(f"  Running {dname} rollout...")
                torch.manual_seed(args.seed)
                result = _rollout_with_buffer(
                    model, vqvae, z_start, actions, gt_tokens,
                    dbuf, topk=args.topk, temperature=args.temperature,
                )
                divergence = _compute_divergence_from_baseline(baseline, result, gt_rgb, lpips_fn)
                s = _variant_summary(result["per_frame"], divergence, dname)
                drop_summaries.update(s)

        # Buffer usage summary
        usage = _buffer_usage_summary(variant_summaries, drop_summaries, full_buffer_len)

        # Print key findings
        print(f"\n  Buffer verdict: {usage.get('buffer_verdict', 'unknown')}")
        print(f"  Empty vs full agreement: {usage.get('empty_vs_full_agreement', 'N/A'):.4f}"
              if isinstance(usage.get('empty_vs_full_agreement'), float) else
              f"  Empty vs full agreement: N/A")
        print(f"  Order sensitive: {usage.get('order_sensitive', 'N/A')}")
        print(f"  More important half: {usage.get('more_important_half', 'N/A')}")

        if usage.get("token_accuracy_by_buffer_length"):
            print(f"  Token accuracy by buffer length:")
            for length, acc in sorted(usage["token_accuracy_by_buffer_length"].items(), key=lambda x: int(x[0])):
                print(f"    len={length}: {acc:.4f}")

        if usage.get("per_slot_sensitivity"):
            print(f"  Per-slot sensitivity (acc delta when dropped):")
            for slot, delta in usage["per_slot_sensitivity"].items():
                print(f"    {slot}: {delta:+.4f}")

        # Per-context result
        ctx_result = {
            "context": ctx_name,
            "config": {
                "checkpoint": args.checkpoint,
                "rollout_steps": args.rollout_steps,
                "start_idx": args.start_idx,
                "topk": args.topk,
                "temperature": args.temperature,
                "full_buffer_len": full_buffer_len,
            },
            "buffer_usage_summary": usage,
            "variant_summaries": variant_summaries,
            "drop_summaries": drop_summaries,
            "per_frame": {
                vname: vdata["per_frame"]
                for vname, vdata in variant_results.items()
            },
            "divergence": {
                vname: vdata["divergence"]
                for vname, vdata in variant_results.items()
                if vdata["divergence"] is not None
            },
        }
        all_results.append(ctx_result)
        save_json(ctx_result, out_dir / f"{ctx_name}_buffer_ablation.json")

    # Aggregate across contexts
    if all_results:
        print(f"\n{'='*60}")
        print("Aggregate summary")
        print(f"{'='*60}")

        agg = {}
        # Aggregate buffer verdicts
        verdicts = [r["buffer_usage_summary"].get("buffer_verdict", "unknown") for r in all_results]
        agg["buffer_verdict_majority"] = max(set(verdicts), key=verdicts.count)
        print(f"  Buffer verdict (majority): {agg['buffer_verdict_majority']}")

        # Aggregate key metrics
        for key in ["empty_vs_full_agreement", "shuffled_vs_full_agreement",
                     "noise_corrupt_vs_full_agreement",
                     "zero_oldest_agreement", "zero_newest_agreement"]:
            vals = [r["buffer_usage_summary"].get(key) for r in all_results
                    if r["buffer_usage_summary"].get(key) is not None]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                print(f"  {key}: {np.mean(vals):.4f}")

        # More-important-half consensus
        halves = [r["buffer_usage_summary"].get("more_important_half", "unknown") for r in all_results]
        agg["more_important_half_majority"] = max(set(halves), key=halves.count)
        print(f"  More important half (majority): {agg['more_important_half_majority']}")

        save_json(
            {"aggregate": agg, "per_context": [r["buffer_usage_summary"] for r in all_results]},
            out_dir / "aggregate_summary.json",
        )

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
