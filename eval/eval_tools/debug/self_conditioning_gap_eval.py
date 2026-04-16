#!/usr/bin/env python3
"""
Debug Tool 1: Self-Conditioning Gap Evaluation

Compares three rollout modes on the same context + action sequence:
  - Teacher-forced (TF): every step uses GT tokens as input
  - Mixed: first half GT, second half AR
  - Pure autoregressive (AR): every step uses model's own predictions

Per-frame metrics track where and how quickly quality diverges between
TF and AR, isolating whether long-horizon failure is:
  - an immediate first-AR-step cliff
  - gradual compounding drift
  - a later phase transition

Usage:
    python eval/eval_tools/debug/self_conditioning_gap_eval.py \
        --checkpoint ./checkpoints/ochre-v7.5.3-step240k.pt \
        --vqvae_ckpt ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --output_dir ./eval/runs/debug/self-conditioning-gap/v7.5.3-240k/
"""
from __future__ import annotations

import argparse
import json
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
# Rollout helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout(
    model,
    vqvae,
    z_start: torch.Tensor,
    action_seq: torch.Tensor,
    gt_tokens: torch.Tensor,
    temporal_buffer: list[torch.Tensor],
    mode: str,
    topk: int = 50,
    temperature: float = 1.0,
    switch_step: int | None = None,
):
    """
    Run a rollout in one of three modes:
      'tf'    – always feed GT tokens as the next input
      'ar'    – always feed sampled tokens
      'mixed' – GT for steps < switch_step, AR after
    """
    buf = list(temporal_buffer)
    z_t = z_start  # (1, H, W)
    num_steps = action_seq.shape[0]
    if switch_step is None:
        switch_step = num_steps // 2

    per_frame: dict[str, list] = {
        "token_accuracy": [],
        "argmax_unique": [],
        "mean_max_prob": [],
        "mean_entropy": [],
        "input_unique": [],
        "logit_kl_vs_uniform": [],
    }

    pred_tokens_list = []
    pred_rgb_list = []

    for t in range(num_steps):
        per_frame["input_unique"].append(int(z_t.unique().numel()))

        logits, new_state = model.step(z_t, action_seq[t : t + 1], buf)
        probs = F.softmax(logits / temperature, dim=1)  # (B, C, H, W)

        # --- per-frame scalars ---
        greedy = logits.argmax(dim=1)
        per_frame["argmax_unique"].append(int(greedy.unique().numel()))
        per_frame["mean_max_prob"].append(float(probs.max(dim=1)[0].mean()))

        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        per_frame["mean_entropy"].append(float(entropy.mean()))

        # KL(pred || uniform) — how peaked is the distribution?
        C = probs.shape[1]
        uniform = torch.full_like(probs[:, :1, :, :], 1.0 / C).expand_as(probs)
        kl = (probs * (torch.log(probs + 1e-9) - torch.log(uniform))).sum(dim=1)
        per_frame["logit_kl_vs_uniform"].append(float(kl.mean()))

        gt_t = gt_tokens[t : t + 1]
        per_frame["token_accuracy"].append(float((greedy == gt_t).float().mean()))

        # --- sample next tokens ---
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
            sampled = flat_idx[torch.arange(b * h * w, device=flat_idx.device), chosen_rel.flatten()]
            z_sampled = sampled.reshape(b, h, w)
        else:
            z_sampled = greedy

        # Decide next input
        if mode == "tf":
            z_next = gt_t
        elif mode == "ar":
            z_next = z_sampled
        elif mode == "mixed":
            z_next = gt_t if t < switch_step else z_sampled
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pred_tokens_list.append(z_sampled.detach().clone())
        pred_rgb_list.append(vqvae.decode_code(z_sampled))

        buf.append(new_state.detach())
        if len(buf) > getattr(model, "temporal_context_len", 8):
            buf.pop(0)

        z_t = z_next

    return {
        "per_frame": per_frame,
        "pred_tokens": torch.stack(pred_tokens_list, dim=0),
        "pred_rgb": torch.cat(pred_rgb_list, dim=0),
    }


# ---------------------------------------------------------------------------
# Cross-mode divergence
# ---------------------------------------------------------------------------

def _compute_divergence(tf_result: dict, ar_result: dict, gt_rgb: torch.Tensor, lpips_fn) -> dict:
    """Compare TF vs AR rollouts frame-by-frame."""
    tf_rgb = tf_result["pred_rgb"]
    ar_rgb = ar_result["pred_rgb"]
    tf_tokens = tf_result["pred_tokens"]
    ar_tokens = ar_result["pred_tokens"]
    n = tf_rgb.shape[0]

    div = {
        "tf_ar_token_agreement": [],
        "tf_ar_lpips": [],
        "tf_gt_lpips": [],
        "ar_gt_lpips": [],
        "tf_gt_psnr": [],
        "ar_gt_psnr": [],
        "tf_ar_edge_l1": [],
    }

    for i in range(n):
        # Token agreement between TF and AR predictions
        agree = float((tf_tokens[i] == ar_tokens[i]).float().mean())
        div["tf_ar_token_agreement"].append(agree)

        # PSNR
        mse_tf = (tf_rgb[i:i+1] - gt_rgb[i:i+1]).square().mean().detach()
        mse_ar = (ar_rgb[i:i+1] - gt_rgb[i:i+1]).square().mean().detach()
        div["tf_gt_psnr"].append(float(10 * torch.log10(1.0 / (mse_tf + 1e-8))))
        div["ar_gt_psnr"].append(float(10 * torch.log10(1.0 / (mse_ar + 1e-8))))

        # LPIPS
        if lpips_fn is not None:
            tf_norm = tf_rgb[i:i+1] * 2.0 - 1.0
            ar_norm = ar_rgb[i:i+1] * 2.0 - 1.0
            gt_norm = gt_rgb[i:i+1] * 2.0 - 1.0
            div["tf_ar_lpips"].append(float(lpips_fn(tf_norm, ar_norm)))
            div["tf_gt_lpips"].append(float(lpips_fn(tf_norm, gt_norm)))
            div["ar_gt_lpips"].append(float(lpips_fn(ar_norm, gt_norm)))

        # Edge divergence
        tf_edge = _edge_map(tf_rgb[i:i+1])
        ar_edge = _edge_map(ar_rgb[i:i+1])
        div["tf_ar_edge_l1"].append(float((tf_edge - ar_edge).abs().mean()))

    return div


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _summarize(per_frame: dict, label: str) -> dict:
    """Compute summary stats from per-frame metrics."""
    summary = {}
    for key, vals in per_frame.items():
        arr = np.array(vals)
        summary[f"{label}_{key}_mean"] = float(arr.mean())
        summary[f"{label}_{key}_first"] = float(arr[0])
        summary[f"{label}_{key}_last"] = float(arr[-1])
        if len(arr) >= 2:
            summary[f"{label}_{key}_first_step_delta"] = float(arr[1] - arr[0])
        if len(arr) >= 6:
            summary[f"{label}_{key}_first5_mean"] = float(arr[:5].mean())
            summary[f"{label}_{key}_last5_mean"] = float(arr[-5:].mean())
    return summary


def _gap_summary(divergence: dict) -> dict:
    """Characterize the TF-AR gap shape."""
    summary = {}
    agree = np.array(divergence["tf_ar_token_agreement"])
    summary["tf_ar_agreement_frame0"] = float(agree[0])
    summary["tf_ar_agreement_frame5"] = float(agree[min(5, len(agree) - 1)])
    summary["tf_ar_agreement_last"] = float(agree[-1])
    summary["tf_ar_agreement_mean"] = float(agree.mean())

    if len(agree) >= 3:
        # Classify gap shape: cliff vs drift
        first_drop = agree[0] - agree[1]
        total_drop = agree[0] - agree[-1]
        if total_drop > 0.01:
            cliff_ratio = first_drop / total_drop
            summary["gap_cliff_ratio"] = float(cliff_ratio)
            summary["gap_shape"] = "cliff" if cliff_ratio > 0.5 else "drift"
        else:
            summary["gap_cliff_ratio"] = 0.0
            summary["gap_shape"] = "minimal"

    psnr_gap = np.array(divergence.get("tf_gt_psnr", [])) - np.array(divergence.get("ar_gt_psnr", []))
    if len(psnr_gap) > 0:
        summary["psnr_gap_mean"] = float(psnr_gap.mean())
        summary["psnr_gap_last"] = float(psnr_gap[-1])

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-conditioning gap evaluation")
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

        # Load context with buffer priming
        data = np.load(context_path)
        tokens_all = torch.from_numpy(data["tokens"]).long().to(device)
        actions_all = torch.from_numpy(data["actions"]).float().to(device)

        end_idx = args.start_idx + args.rollout_steps + 1
        tokens = tokens_all[args.start_idx : end_idx]
        actions = actions_all[args.start_idx : args.start_idx + args.rollout_steps]
        gt_tokens = tokens[1:]
        gt_rgb = vqvae.decode_code(gt_tokens)
        z_start = tokens[0:1]

        # Prime temporal buffer
        prime_len = min(getattr(model, "temporal_context_len", 8), args.start_idx)
        temporal_buffer = []
        if prime_len > 0:
            from diagnostics.visual_quality_eval import prime_temporal_buffer
            hist_tokens = tokens_all[args.start_idx - prime_len : args.start_idx + 1]
            hist_actions = actions_all[args.start_idx - prime_len : args.start_idx]
            temporal_buffer = prime_temporal_buffer(model, hist_tokens, hist_actions)

        # Run three modes
        modes = {}
        for mode_name in ["tf", "ar", "mixed"]:
            print(f"  Running {mode_name} rollout...")
            # Use same seed for sampling fairness
            torch.manual_seed(args.seed)
            modes[mode_name] = _rollout(
                model, vqvae, z_start, actions, gt_tokens,
                temporal_buffer, mode=mode_name,
                topk=args.topk, temperature=args.temperature,
            )

        # Compute TF-AR divergence
        print("  Computing divergence...")
        divergence = _compute_divergence(modes["tf"], modes["ar"], gt_rgb, lpips_fn)

        # Summaries
        summary = {}
        for mode_name in ["tf", "ar", "mixed"]:
            summary.update(_summarize(modes[mode_name]["per_frame"], mode_name))
        summary.update(_gap_summary(divergence))

        # Print key findings
        shape = summary.get("gap_shape", "unknown")
        agree_0 = summary.get("tf_ar_agreement_frame0", 0)
        agree_last = summary.get("tf_ar_agreement_last", 0)
        cliff = summary.get("gap_cliff_ratio", 0)
        print(f"\n  Gap shape: {shape}")
        print(f"  TF-AR token agreement: frame 0 = {agree_0:.3f}, last = {agree_last:.3f}")
        print(f"  Cliff ratio: {cliff:.3f}")
        if "psnr_gap_mean" in summary:
            print(f"  PSNR gap (TF - AR): mean = {summary['psnr_gap_mean']:.2f} dB, last = {summary['psnr_gap_last']:.2f} dB")

        # Per-context result
        ctx_result = {
            "context": ctx_name,
            "config": {
                "checkpoint": args.checkpoint,
                "rollout_steps": args.rollout_steps,
                "start_idx": args.start_idx,
                "topk": args.topk,
                "temperature": args.temperature,
                "prime_len": prime_len,
            },
            "summary": summary,
            "per_frame": {
                mode_name: modes[mode_name]["per_frame"]
                for mode_name in ["tf", "ar", "mixed"]
            },
            "divergence": divergence,
        }
        all_results.append(ctx_result)

        # Save per-context JSON
        save_json(ctx_result, out_dir / f"{ctx_name}_self_conditioning_gap.json")

    # Aggregate across contexts
    print(f"\n{'='*60}")
    print("Aggregate summary")
    print(f"{'='*60}")

    agg_keys = [
        "tf_ar_agreement_frame0", "tf_ar_agreement_last", "gap_cliff_ratio",
        "psnr_gap_mean", "ar_token_accuracy_first", "ar_token_accuracy_last",
        "ar_mean_entropy_first", "ar_mean_entropy_last",
        "ar_mean_max_prob_first", "ar_mean_max_prob_last",
    ]
    agg = {}
    for key in agg_keys:
        vals = [r["summary"].get(key) for r in all_results if key in r["summary"]]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            print(f"  {key}: {np.mean(vals):.4f}")

    shapes = [r["summary"].get("gap_shape", "unknown") for r in all_results]
    agg["gap_shape_majority"] = max(set(shapes), key=shapes.count)
    print(f"  gap shape (majority): {agg['gap_shape_majority']}")

    save_json({"aggregate": agg, "per_context": [r["summary"] for r in all_results]}, out_dir / "aggregate_summary.json")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
