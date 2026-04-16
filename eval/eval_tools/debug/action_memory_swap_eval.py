#!/usr/bin/env python3
"""
Debug Tool 2: Action–Memory Swap Evaluation

Tests whether the model's predictions are dominated by temporal buffer state
or by the current action, by running controlled swap experiments:

  Probe A — Same buffer, different actions:
    Using the GT-primed buffer and current latent, predict with several
    different actions. Measures how much the output changes when only
    the action changes.

  Probe B — Same action, different buffers:
    Prime buffers with different action regimes, then predict with a single
    fixed action. Measures how much the output changes when only history changes.

  Probe C — Action sensitivity over AR depth:
    Run a static AR rollout and at each step test how much the logits
    would change if a different action were applied. Tracks whether
    action sensitivity decays as AR error accumulates.

If Probe A shows small differences → actions are being ignored (memory-dominated).
If Probe B shows small differences → buffer is being ignored (action-dominated).

Usage:
    python eval/eval_tools/debug/action_memory_swap_eval.py \
        --checkpoint ./checkpoints/ochre-v7.5.3-step240k.pt \
        --vqvae_ckpt ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --output_dir ./eval/runs/debug/action-memory-swap/v7.5.3-240k/
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

from diagnostics.analyze_checkpoint import get_test_actions
from diagnostics.visual_quality_eval import prime_temporal_buffer
from eval.eval_tools.shared import (
    ensure_dir,
    load_model_bundle,
    resolve_context_paths,
    save_json,
    select_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logit_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """Compute divergence metrics between two logit tensors (B, C, H, W)."""
    probs_a = F.softmax(logits_a, dim=1)
    probs_b = F.softmax(logits_b, dim=1)

    # Token-level agreement (argmax)
    greedy_a = logits_a.argmax(dim=1)
    greedy_b = logits_b.argmax(dim=1)
    token_agreement = float((greedy_a == greedy_b).float().mean())

    # Mean L1 between probability distributions
    prob_l1 = float((probs_a - probs_b).abs().mean())

    # Jensen-Shannon divergence (symmetric KL)
    m = 0.5 * (probs_a + probs_b)
    kl_am = (probs_a * (torch.log(probs_a + 1e-9) - torch.log(m + 1e-9))).sum(dim=1).mean()
    kl_bm = (probs_b * (torch.log(probs_b + 1e-9) - torch.log(m + 1e-9))).sum(dim=1).mean()
    jsd = float(0.5 * kl_am + 0.5 * kl_bm)

    # Unique code difference
    unique_a = int(greedy_a.unique().numel())
    unique_b = int(greedy_b.unique().numel())

    return {
        "token_agreement": token_agreement,
        "prob_l1": prob_l1,
        "jsd": jsd,
        "unique_a": unique_a,
        "unique_b": unique_b,
    }


def _prime_buffer_with_action(
    model, z_start: torch.Tensor, action: torch.Tensor,
    num_prime_steps: int, temporal_buffer: list[torch.Tensor],
    topk: int = 50,
) -> list[torch.Tensor]:
    """Run a short AR rollout to build a buffer primed with a specific action."""
    buf = list(temporal_buffer)
    z_t = z_start
    for _ in range(num_prime_steps):
        logits, new_state = model.step(z_t, action, buf)
        probs = F.softmax(logits, dim=1)
        if topk > 0:
            topk_probs, topk_idx = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
            b, _, h, w = topk_probs.shape
            flat_p = topk_probs.permute(0, 2, 3, 1).reshape(b * h * w, topk)
            flat_i = topk_idx.permute(0, 2, 3, 1).reshape(b * h * w, topk)
            if flat_p.device.type == "mps":
                chosen = torch.multinomial(flat_p.cpu(), 1).to(flat_p.device)
            else:
                chosen = torch.multinomial(flat_p, 1)
            z_t = flat_i[torch.arange(b * h * w, device=flat_i.device), chosen.flatten()].reshape(b, h, w)
        else:
            z_t = logits.argmax(dim=1)
        buf.append(new_state.detach())
        if len(buf) > getattr(model, "temporal_context_len", 8):
            buf.pop(0)
    return buf, z_t


# ---------------------------------------------------------------------------
# Probe A: Same buffer, different actions
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_same_buffer_diff_action(
    model, z_t: torch.Tensor, temporal_buffer: list[torch.Tensor],
    test_actions: dict[str, torch.Tensor],
) -> dict:
    """Single-step probe: same input + buffer, vary the action."""
    logits_by_action = {}
    for name, action in test_actions.items():
        logits, _ = model.step(z_t, action, temporal_buffer)
        logits_by_action[name] = logits

    # Pairwise divergence against static baseline
    static_logits = logits_by_action["static"]
    results = {}
    for name, logits in logits_by_action.items():
        if name == "static":
            continue
        results[name] = _logit_divergence(static_logits, logits)

    # Summary: mean divergence across all non-static actions
    mean_agreement = np.mean([r["token_agreement"] for r in results.values()])
    mean_jsd = np.mean([r["jsd"] for r in results.values()])
    mean_prob_l1 = np.mean([r["prob_l1"] for r in results.values()])

    return {
        "per_action": results,
        "mean_token_agreement_vs_static": float(mean_agreement),
        "mean_jsd_vs_static": float(mean_jsd),
        "mean_prob_l1_vs_static": float(mean_prob_l1),
    }


# ---------------------------------------------------------------------------
# Probe B: Same action, different buffers
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_same_action_diff_buffer(
    model, z_start: torch.Tensor, test_actions: dict[str, torch.Tensor],
    base_buffer: list[torch.Tensor],
    prime_steps: int = 5, topk: int = 50,
    query_action_name: str = "static",
) -> dict:
    """
    Prime separate buffers with different action regimes, then query
    with the same action. Measures buffer influence.
    """
    prime_actions = ["static", "camera_right", "move_forward", "jump"]
    buffers = {}

    for act_name in prime_actions:
        torch.manual_seed(42)  # consistent sampling across primes
        buf, _ = _prime_buffer_with_action(
            model, z_start, test_actions[act_name],
            prime_steps, base_buffer, topk=topk,
        )
        buffers[act_name] = buf

    # Query each primed buffer with the same action AND the same z_t.
    # Using z_start (not z_after_prime) ensures we isolate buffer influence
    # from input-state differences caused by different priming rollouts.
    query_action = test_actions[query_action_name]
    logits_by_prime = {}
    for act_name in prime_actions:
        logits, _ = model.step(z_start, query_action, buffers[act_name])
        logits_by_prime[act_name] = logits

    # Pairwise divergence: static-primed buffer vs others
    static_logits = logits_by_prime["static"]
    results = {}
    for act_name, logits in logits_by_prime.items():
        if act_name == "static":
            continue
        results[f"{act_name}_primed"] = _logit_divergence(static_logits, logits)

    mean_agreement = np.mean([r["token_agreement"] for r in results.values()])
    mean_jsd = np.mean([r["jsd"] for r in results.values()])

    return {
        "query_action": query_action_name,
        "prime_steps": prime_steps,
        "per_buffer": results,
        "mean_token_agreement_vs_static_buffer": float(mean_agreement),
        "mean_jsd_vs_static_buffer": float(mean_jsd),
    }


# ---------------------------------------------------------------------------
# Probe C: Action sensitivity over AR depth
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_action_sensitivity_over_ar(
    model, z_start: torch.Tensor, test_actions: dict[str, torch.Tensor],
    base_buffer: list[torch.Tensor],
    ar_steps: int = 8, topk: int = 50,
) -> dict:
    """
    Run an AR rollout under 'static', then at each step test how much the
    logits would change if a different action were applied. Tracks whether
    action sensitivity decays as AR error accumulates.
    """
    buf = list(base_buffer)
    z_t = z_start
    static_action = test_actions["static"]
    probe_actions = {k: v for k, v in test_actions.items() if k != "static"}

    per_step = []

    for step in range(ar_steps):
        # Get static baseline logits
        logits_static, new_state = model.step(z_t, static_action, buf)

        # Probe each non-static action at this step
        step_results = {}
        for name, action in probe_actions.items():
            logits_alt, _ = model.step(z_t, action, buf)
            step_results[name] = _logit_divergence(logits_static, logits_alt)

        mean_jsd = float(np.mean([r["jsd"] for r in step_results.values()]))
        mean_agreement = float(np.mean([r["token_agreement"] for r in step_results.values()]))

        per_step.append({
            "step": step,
            "per_action": step_results,
            "mean_jsd_vs_static": mean_jsd,
            "mean_token_agreement_vs_static": mean_agreement,
        })

        # Advance AR rollout with static action
        probs = F.softmax(logits_static, dim=1)
        if topk > 0:
            topk_probs, topk_idx = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
            b, _, h, w = topk_probs.shape
            flat_p = topk_probs.permute(0, 2, 3, 1).reshape(b * h * w, topk)
            flat_i = topk_idx.permute(0, 2, 3, 1).reshape(b * h * w, topk)
            if flat_p.device.type == "mps":
                chosen = torch.multinomial(flat_p.cpu(), 1).to(flat_p.device)
            else:
                chosen = torch.multinomial(flat_p, 1)
            z_t = flat_i[torch.arange(b * h * w, device=flat_i.device), chosen.flatten()].reshape(b, h, w)
        else:
            z_t = logits_static.argmax(dim=1)

        buf.append(new_state.detach())
        if len(buf) > getattr(model, "temporal_context_len", 8):
            buf.pop(0)

    # Summarize trend
    jsd_curve = [s["mean_jsd_vs_static"] for s in per_step]
    agreement_curve = [s["mean_token_agreement_vs_static"] for s in per_step]

    summary = {
        "jsd_step0": jsd_curve[0],
        "jsd_last": jsd_curve[-1],
        "agreement_step0": agreement_curve[0],
        "agreement_last": agreement_curve[-1],
    }
    if len(jsd_curve) >= 2 and jsd_curve[0] > 1e-9:
        summary["jsd_decay_ratio"] = jsd_curve[-1] / jsd_curve[0]
        summary["action_sensitivity_trend"] = (
            "stable" if jsd_curve[-1] > 0.8 * jsd_curve[0]
            else "weakening" if jsd_curve[-1] > 0.4 * jsd_curve[0]
            else "collapsing"
        )

    return {
        "per_step": per_step,
        "jsd_curve": jsd_curve,
        "agreement_curve": agreement_curve,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Action–Memory swap evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--context_npz", nargs="+")
    parser.add_argument("--data_dir", default="./preprocessedv5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_idx", type=int, default=8)
    parser.add_argument("--prime_steps", type=int, default=5)
    parser.add_argument("--ar_steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    model, vqvae, _ = load_model_bundle(args.checkpoint, args.vqvae_ckpt, device)
    test_actions = get_test_actions(device)
    contexts = resolve_context_paths(args.context_npz, args.data_dir)
    out_dir = ensure_dir(args.output_dir)

    all_results = []

    for context_path in contexts:
        ctx_name = Path(context_path).stem
        print(f"\n{'='*60}")
        print(f"Context: {ctx_name}")
        print(f"{'='*60}")

        data = np.load(context_path)
        tokens_all = torch.from_numpy(data["tokens"]).long().to(device)
        actions_all = torch.from_numpy(data["actions"]).float().to(device)

        z_start = tokens_all[args.start_idx : args.start_idx + 1]

        # Prime temporal buffer from GT history
        prime_len = min(getattr(model, "temporal_context_len", 8), args.start_idx)
        if prime_len > 0:
            hist_tokens = tokens_all[args.start_idx - prime_len : args.start_idx + 1]
            hist_actions = actions_all[args.start_idx - prime_len : args.start_idx]
            base_buffer = prime_temporal_buffer(model, hist_tokens, hist_actions)
        else:
            base_buffer = []

        # --- Probe A ---
        print("  Probe A: same buffer, different actions...")
        torch.manual_seed(args.seed)
        probe_a = probe_same_buffer_diff_action(model, z_start, base_buffer, test_actions)
        print(f"    Mean token agreement vs static: {probe_a['mean_token_agreement_vs_static']:.3f}")
        print(f"    Mean JSD vs static: {probe_a['mean_jsd_vs_static']:.4f}")

        # --- Probe B ---
        print("  Probe B: same action, different buffers...")
        torch.manual_seed(args.seed)
        probe_b = probe_same_action_diff_buffer(
            model, z_start, test_actions, base_buffer,
            prime_steps=args.prime_steps, topk=args.topk,
        )
        print(f"    Mean token agreement vs static-buffer: {probe_b['mean_token_agreement_vs_static_buffer']:.3f}")
        print(f"    Mean JSD vs static-buffer: {probe_b['mean_jsd_vs_static_buffer']:.4f}")

        # --- Probe C ---
        print("  Probe C: action sensitivity over AR depth...")
        torch.manual_seed(args.seed)
        probe_c = probe_action_sensitivity_over_ar(
            model, z_start, test_actions, base_buffer,
            ar_steps=args.ar_steps, topk=args.topk,
        )
        trend = probe_c["summary"].get("action_sensitivity_trend", "unknown")
        decay = probe_c["summary"].get("jsd_decay_ratio", 0)
        print(f"    Action sensitivity trend: {trend}")
        print(f"    JSD decay ratio (last/first): {decay:.3f}")
        print(f"    JSD curve: {[f'{v:.4f}' for v in probe_c['jsd_curve']]}")

        # --- Diagnosis ---
        a_jsd = probe_a["mean_jsd_vs_static"]
        b_jsd = probe_b["mean_jsd_vs_static_buffer"]
        dominance_ratio = None

        if a_jsd < 1e-4 and b_jsd < 1e-4:
            diagnosis = "both_weak"
            diagnosis_detail = "Neither action nor buffer meaningfully affect output — model may be dominated by spatial context alone"
        else:
            if b_jsd > 1e-6:
                dominance_ratio = a_jsd / b_jsd
            else:
                dominance_ratio = float("inf")

            if dominance_ratio > 3.0:
                diagnosis = "action_dominated"
                diagnosis_detail = f"Actions move logits {dominance_ratio:.1f}x more than buffer history — buffer may be underused"
            elif dominance_ratio < 0.33:
                diagnosis = "memory_dominated"
                diagnosis_detail = f"Buffer history moves logits {1/dominance_ratio:.1f}x more than actions — action conditioning may be too weak"
            else:
                diagnosis = "balanced"
                diagnosis_detail = f"Action/buffer influence ratio {dominance_ratio:.1f}x — reasonably balanced"

        print(f"\n  Diagnosis: {diagnosis}")
        print(f"  {diagnosis_detail}")

        ctx_result = {
            "context": ctx_name,
            "config": {
                "checkpoint": args.checkpoint,
                "start_idx": args.start_idx,
                "prime_steps": args.prime_steps,
                "ar_steps": args.ar_steps,
                "topk": args.topk,
            },
            "probe_a_same_buffer_diff_action": probe_a,
            "probe_b_same_action_diff_buffer": probe_b,
            "probe_c_action_sensitivity_over_ar": {
                "jsd_curve": probe_c["jsd_curve"],
                "agreement_curve": probe_c["agreement_curve"],
                "summary": probe_c["summary"],
            },
            "diagnosis": diagnosis,
            "diagnosis_detail": diagnosis_detail,
            "dominance_ratio_action_over_buffer": float(dominance_ratio) if dominance_ratio is not None and dominance_ratio != float("inf") else None,
        }
        all_results.append(ctx_result)
        save_json(ctx_result, out_dir / f"{ctx_name}_action_memory_swap.json")

    # Aggregate
    print(f"\n{'='*60}")
    print("Aggregate")
    print(f"{'='*60}")

    diagnoses = [r["diagnosis"] for r in all_results]
    ratios = [r["dominance_ratio_action_over_buffer"] for r in all_results if r["dominance_ratio_action_over_buffer"] is not None]
    trends = [r["probe_c_action_sensitivity_over_ar"]["summary"].get("action_sensitivity_trend", "unknown") for r in all_results]

    agg = {
        "diagnosis_majority": max(set(diagnoses), key=diagnoses.count),
        "dominance_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "action_sensitivity_trend_majority": max(set(trends), key=trends.count),
        "per_context_diagnoses": diagnoses,
    }
    print(f"  Majority diagnosis: {agg['diagnosis_majority']}")
    if ratios:
        print(f"  Mean dominance ratio (action/buffer): {agg['dominance_ratio_mean']:.2f}")
    print(f"  Action sensitivity trend: {agg['action_sensitivity_trend_majority']}")

    save_json(agg, out_dir / "aggregate_summary.json")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
