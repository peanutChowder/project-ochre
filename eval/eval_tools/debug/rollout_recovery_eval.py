#!/usr/bin/env python3
"""
Debug Tool 4: Rollout Recovery Evaluation

Tests whether the model can recover after a controlled perturbation, or
whether small errors cause permanent drift.

For each perturbation type, runs two rollouts from the same starting state:
  - clean:     normal AR rollout (unperturbed baseline)
  - perturbed: identical rollout, but at step `inject_step` a perturbation
               is injected into the token frame or buffer state

Perturbation types:
  - token_patch_5pct:   replace 5% of tokens with random codebook entries
  - token_patch_10pct:  replace 10% of tokens with random codebook entries
  - wrong_frame:        replace the entire token frame with a random frame
  - buffer_corrupt:     add Gaussian noise to the most recent buffer state
  - buffer_zero_last:   zero out the most recent buffer state

Recovery is measured by tracking how the perturbed rollout re-converges
(or fails to) toward the clean rollout after the injection point.

Usage:
    python eval/eval_tools/debug/rollout_recovery_eval.py \
        --checkpoint ./checkpoints/ochre-v7.5.3-step240k.pt \
        --vqvae_ckpt ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --output_dir ./eval/runs/debug/rollout-recovery/v7.5.3-240k/
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
# Perturbation helpers
# ---------------------------------------------------------------------------

def _perturb_tokens(z: torch.Tensor, fraction: float, codebook_size: int) -> torch.Tensor:
    """Replace `fraction` of tokens with random codebook entries."""
    z_out = z.clone()
    mask = torch.rand_like(z.float()) < fraction
    random_tokens = torch.randint(0, codebook_size, z.shape, device=z.device)
    z_out[mask] = random_tokens[mask]
    return z_out


def _random_token_frame(z: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """Replace entire frame with random codebook entries."""
    return torch.randint(0, codebook_size, z.shape, device=z.device)


def _corrupt_buffer_last(buf: list[torch.Tensor], noise_scale: float = 1.0) -> list[torch.Tensor]:
    """Add Gaussian noise to the most recent buffer state."""
    if not buf:
        return buf
    out = list(buf)
    last = out[-1]
    std = last.std().item()
    out[-1] = last + torch.randn_like(last) * (noise_scale * std)
    return out


def _zero_buffer_last(buf: list[torch.Tensor]) -> list[torch.Tensor]:
    """Zero out the most recent buffer state."""
    if not buf:
        return buf
    out = list(buf)
    out[-1] = torch.zeros_like(out[-1])
    return out


# ---------------------------------------------------------------------------
# Paired rollout with injection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _rollout_pair(
    model,
    vqvae,
    z_start: torch.Tensor,
    action_seq: torch.Tensor,
    gt_tokens: torch.Tensor,
    temporal_buffer: list[torch.Tensor],
    perturbation: str,
    inject_step: int,
    codebook_size: int,
    topk: int = 50,
    temperature: float = 1.0,
    seed: int = 42,
):
    """
    Run clean and perturbed rollouts side by side.

    Both rollouts use the same random seed for sampling so that any divergence
    is caused by the perturbation, not sampling variance.
    """
    num_steps = action_seq.shape[0]

    # We run both rollouts step-by-step so they share state up to inject_step
    clean_buf = list(temporal_buffer)
    pert_buf = list(temporal_buffer)
    clean_z = z_start.clone()
    pert_z = z_start.clone()

    # Track whether perturbation has been injected
    injected = False

    clean_per_frame = {"token_accuracy": [], "argmax_unique": [], "mean_max_prob": []}
    pert_per_frame = {"token_accuracy": [], "argmax_unique": [], "mean_max_prob": []}
    divergence = {
        "token_agreement": [],
        "psnr_clean_vs_gt": [],
        "psnr_pert_vs_gt": [],
        "psnr_delta": [],
        "edge_l1": [],
    }

    clean_rgb_list = []
    pert_rgb_list = []

    for t in range(num_steps):
        # --- Clean rollout step ---
        torch.manual_seed(seed + t)
        clean_logits, clean_state = model.step(clean_z, action_seq[t:t+1], clean_buf)
        clean_probs = F.softmax(clean_logits / temperature, dim=1)
        clean_greedy = clean_logits.argmax(dim=1)
        clean_per_frame["argmax_unique"].append(int(clean_greedy.unique().numel()))
        clean_per_frame["mean_max_prob"].append(float(clean_probs.max(dim=1)[0].mean()))
        clean_per_frame["token_accuracy"].append(float((clean_greedy == gt_tokens[t:t+1]).float().mean()))

        clean_sampled = _sample(clean_probs, topk)

        # --- Perturbed rollout step ---
        torch.manual_seed(seed + t)
        pert_logits, pert_state = model.step(pert_z, action_seq[t:t+1], pert_buf)
        pert_probs = F.softmax(pert_logits / temperature, dim=1)
        pert_greedy = pert_logits.argmax(dim=1)
        pert_per_frame["argmax_unique"].append(int(pert_greedy.unique().numel()))
        pert_per_frame["mean_max_prob"].append(float(pert_probs.max(dim=1)[0].mean()))
        pert_per_frame["token_accuracy"].append(float((pert_greedy == gt_tokens[t:t+1]).float().mean()))

        pert_sampled = _sample(pert_probs, topk)

        # Decode for metrics
        clean_rgb = vqvae.decode_code(clean_sampled)
        pert_rgb = vqvae.decode_code(pert_sampled)
        clean_rgb_list.append(clean_rgb)
        pert_rgb_list.append(pert_rgb)

        # Divergence metrics
        agree = float((clean_sampled == pert_sampled).float().mean())
        divergence["token_agreement"].append(agree)

        gt_rgb_t = vqvae.decode_code(gt_tokens[t:t+1])
        mse_clean = (clean_rgb - gt_rgb_t).square().mean().detach()
        mse_pert = (pert_rgb - gt_rgb_t).square().mean().detach()
        psnr_clean = float(10 * torch.log10(1.0 / (mse_clean + 1e-8)))
        psnr_pert = float(10 * torch.log10(1.0 / (mse_pert + 1e-8)))
        divergence["psnr_clean_vs_gt"].append(psnr_clean)
        divergence["psnr_pert_vs_gt"].append(psnr_pert)
        divergence["psnr_delta"].append(psnr_pert - psnr_clean)

        clean_edge = _edge_map(clean_rgb)
        pert_edge = _edge_map(pert_rgb)
        divergence["edge_l1"].append(float((clean_edge - pert_edge).abs().mean()))

        # Update buffers
        clean_buf.append(clean_state.detach())
        if len(clean_buf) > getattr(model, "temporal_context_len", 8):
            clean_buf.pop(0)

        pert_buf.append(pert_state.detach())
        if len(pert_buf) > getattr(model, "temporal_context_len", 8):
            pert_buf.pop(0)

        # Advance z_t for both
        clean_z = clean_sampled

        # Apply perturbation at inject_step
        if t == inject_step and not injected:
            if perturbation == "token_patch_5pct":
                pert_z = _perturb_tokens(pert_sampled, 0.05, codebook_size)
            elif perturbation == "token_patch_10pct":
                pert_z = _perturb_tokens(pert_sampled, 0.10, codebook_size)
            elif perturbation == "wrong_frame":
                pert_z = _random_token_frame(pert_sampled, codebook_size)
            elif perturbation == "buffer_corrupt":
                pert_buf = _corrupt_buffer_last(pert_buf)
                pert_z = pert_sampled
            elif perturbation == "buffer_zero_last":
                pert_buf = _zero_buffer_last(pert_buf)
                pert_z = pert_sampled
            else:
                raise ValueError(f"Unknown perturbation: {perturbation}")
            injected = True
        else:
            pert_z = pert_sampled

    return {
        "clean_per_frame": clean_per_frame,
        "pert_per_frame": pert_per_frame,
        "divergence": divergence,
    }


def _sample(probs: torch.Tensor, topk: int) -> torch.Tensor:
    """Top-k sample from probability distribution."""
    if topk > 0:
        topk_probs, topk_idx = probs.topk(topk, dim=1)
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
        b, _, h, w = topk_probs.shape
        flat_probs = topk_probs.permute(0, 2, 3, 1).reshape(b * h * w, topk)
        flat_idx = topk_idx.permute(0, 2, 3, 1).reshape(b * h * w, topk)
        if flat_probs.device.type == "mps":
            chosen = torch.multinomial(flat_probs.cpu(), 1).to(flat_probs.device)
        else:
            chosen = torch.multinomial(flat_probs, 1)
        return flat_idx[
            torch.arange(b * h * w, device=flat_idx.device), chosen.flatten()
        ].reshape(b, h, w)
    else:
        return probs.argmax(dim=1)


# ---------------------------------------------------------------------------
# Recovery analysis
# ---------------------------------------------------------------------------

def _analyze_recovery(divergence: dict, inject_step: int, threshold: float = 0.8) -> dict:
    """Analyze whether and how quickly the perturbed rollout recovers."""
    agree = np.array(divergence["token_agreement"])
    psnr_delta = np.array(divergence["psnr_delta"])

    # Pre-injection agreement should be ~1.0 (identical rollouts)
    pre_agree = agree[:inject_step + 1]
    post_agree = agree[inject_step + 1:] if inject_step + 1 < len(agree) else np.array([])

    analysis = {
        "inject_step": inject_step,
        "pre_injection_agreement_mean": float(pre_agree.mean()) if len(pre_agree) > 0 else None,
    }

    if len(post_agree) == 0:
        analysis["recovery_possible"] = False
        analysis["note"] = "injection at last step, no post-injection frames"
        return analysis

    # Immediate impact
    analysis["first_post_agreement"] = float(post_agree[0])
    analysis["last_post_agreement"] = float(post_agree[-1])
    analysis["post_agreement_mean"] = float(post_agree.mean())

    # Recovery time: first frame where agreement returns above threshold
    recovered_frames = np.where(post_agree >= threshold)[0]
    if len(recovered_frames) > 0:
        analysis["recovery_time"] = int(recovered_frames[0]) + 1  # frames after injection
        analysis["recovered"] = True
    else:
        analysis["recovery_time"] = None
        analysis["recovered"] = False

    # Trajectory shape
    if len(post_agree) >= 3:
        first_half = post_agree[:len(post_agree)//2]
        second_half = post_agree[len(post_agree)//2:]
        if second_half.mean() > first_half.mean() + 0.05:
            analysis["trajectory"] = "recovering"
        elif second_half.mean() < first_half.mean() - 0.05:
            analysis["trajectory"] = "diverging"
        else:
            analysis["trajectory"] = "stable_drift"
    else:
        analysis["trajectory"] = "too_short"

    # PSNR impact
    post_psnr = psnr_delta[inject_step + 1:] if inject_step + 1 < len(psnr_delta) else np.array([])
    if len(post_psnr) > 0:
        analysis["psnr_delta_mean"] = float(post_psnr.mean())
        analysis["psnr_delta_last"] = float(post_psnr[-1])

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PERTURBATION_TYPES = [
    "token_patch_5pct",
    "token_patch_10pct",
    "wrong_frame",
    "buffer_corrupt",
    "buffer_zero_last",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout recovery evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--context_npz", nargs="+")
    parser.add_argument("--data_dir", default="./preprocessedv5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=8)
    parser.add_argument("--inject_step", type=int, default=3,
                        help="Step at which to inject perturbation (default: 3)")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery_threshold", type=float, default=0.8,
                        help="Token agreement threshold to consider 'recovered'")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    model, vqvae, model_kwargs = load_model_bundle(args.checkpoint, args.vqvae_ckpt, device)
    codebook_size = model_kwargs.get("codebook_size", 1024)
    contexts = resolve_context_paths(args.context_npz, args.data_dir)
    out_dir = ensure_dir(args.output_dir)

    all_results = []

    for context_path in contexts:
        ctx_name = Path(context_path).stem
        print(f"\n{'='*60}")
        print(f"Context: {ctx_name}")
        print(f"{'='*60}")

        # Load context and prime buffer
        data = np.load(context_path)
        tokens_all = torch.from_numpy(data["tokens"]).long().to(device)
        actions_all = torch.from_numpy(data["actions"]).float().to(device)

        end_idx = args.start_idx + args.rollout_steps + 1
        tokens = tokens_all[args.start_idx:end_idx]
        actions = actions_all[args.start_idx:args.start_idx + args.rollout_steps]
        gt_tokens = tokens[1:]
        z_start = tokens[0:1]

        prime_len = min(getattr(model, "temporal_context_len", 8), args.start_idx)
        temporal_buffer = []
        if prime_len > 0:
            from diagnostics.visual_quality_eval import prime_temporal_buffer
            hist_tokens = tokens_all[args.start_idx - prime_len:args.start_idx + 1]
            hist_actions = actions_all[args.start_idx - prime_len:args.start_idx]
            temporal_buffer = prime_temporal_buffer(model, hist_tokens, hist_actions)

        ctx_results = {"context": ctx_name, "perturbations": {}}

        for pert_type in PERTURBATION_TYPES:
            print(f"\n  Perturbation: {pert_type} @ step {args.inject_step}")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            result = _rollout_pair(
                model, vqvae, z_start, actions, gt_tokens,
                temporal_buffer, perturbation=pert_type,
                inject_step=args.inject_step, codebook_size=codebook_size,
                topk=args.topk, temperature=args.temperature, seed=args.seed,
            )

            recovery = _analyze_recovery(
                result["divergence"], args.inject_step,
                threshold=args.recovery_threshold,
            )

            # Print summary
            recovered = recovery.get("recovered", False)
            r_time = recovery.get("recovery_time")
            trajectory = recovery.get("trajectory", "unknown")
            first_post = recovery.get("first_post_agreement", 0)
            last_post = recovery.get("last_post_agreement", 0)

            status = f"RECOVERED in {r_time} frames" if recovered else "NOT RECOVERED"
            print(f"    {status} | trajectory: {trajectory}")
            print(f"    Post-injection agreement: first={first_post:.3f}, last={last_post:.3f}")
            if recovery.get("psnr_delta_mean") is not None:
                print(f"    PSNR delta (pert - clean): mean={recovery['psnr_delta_mean']:.2f} dB, "
                      f"last={recovery['psnr_delta_last']:.2f} dB")

            ctx_results["perturbations"][pert_type] = {
                "recovery": recovery,
                "divergence": result["divergence"],
                "clean_per_frame": result["clean_per_frame"],
                "pert_per_frame": result["pert_per_frame"],
            }

        # Summary for this context
        ctx_summary = {
            "context": ctx_name,
            "inject_step": args.inject_step,
            "config": {
                "checkpoint": args.checkpoint,
                "rollout_steps": args.rollout_steps,
                "start_idx": args.start_idx,
                "topk": args.topk,
                "temperature": args.temperature,
                "recovery_threshold": args.recovery_threshold,
            },
        }
        for pert_type in PERTURBATION_TYPES:
            ctx_summary[pert_type] = ctx_results["perturbations"][pert_type]["recovery"]

        ctx_results["summary"] = ctx_summary
        all_results.append(ctx_results)
        save_json(ctx_results, out_dir / f"{ctx_name}_rollout_recovery.json")

    # Aggregate
    if all_results:
        print(f"\n{'='*60}")
        print("Aggregate recovery summary")
        print(f"{'='*60}")

        agg = {}
        for pert_type in PERTURBATION_TYPES:
            recoveries = []
            trajectories = []
            post_agrees = []
            for r in all_results:
                rec = r["perturbations"][pert_type]["recovery"]
                recoveries.append(rec.get("recovered", False))
                trajectories.append(rec.get("trajectory", "unknown"))
                if rec.get("post_agreement_mean") is not None:
                    post_agrees.append(rec["post_agreement_mean"])

            recovery_rate = sum(recoveries) / len(recoveries) if recoveries else 0
            traj_majority = max(set(trajectories), key=trajectories.count) if trajectories else "unknown"
            mean_post_agree = float(np.mean(post_agrees)) if post_agrees else None

            agg[pert_type] = {
                "recovery_rate": recovery_rate,
                "trajectory_majority": traj_majority,
                "post_agreement_mean": mean_post_agree,
            }
            print(f"  {pert_type}:")
            print(f"    recovery rate: {recovery_rate:.0%}")
            print(f"    trajectory: {traj_majority}")
            if mean_post_agree is not None:
                print(f"    post-injection agreement: {mean_post_agree:.4f}")

        # Overall verdict
        any_recovered = any(agg[p]["recovery_rate"] > 0 for p in PERTURBATION_TYPES)
        all_stable_drift = all(
            agg[p]["trajectory_majority"] == "stable_drift"
            for p in PERTURBATION_TYPES
        )
        if not any_recovered and all_stable_drift:
            verdict = "no_recovery_stable_drift"
        elif not any_recovered:
            verdict = "no_recovery_diverging"
        elif all(agg[p]["recovery_rate"] >= 0.5 for p in PERTURBATION_TYPES[:2]):
            verdict = "robust_to_small_perturbations"
        else:
            verdict = "partial_recovery"

        agg["overall_verdict"] = verdict
        print(f"\n  Overall verdict: {verdict}")

        save_json(
            {"aggregate": agg, "per_context": [r["summary"] for r in all_results]},
            out_dir / "aggregate_summary.json",
        )

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
