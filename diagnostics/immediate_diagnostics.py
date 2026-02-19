"""
Immediate diagnostic tests to identify root cause of first-step collapse.

Four critical tests:
1. Analyze collapsed codebook - which 30 codes does the model use?
2. Compare TF vs AR logit distributions - is there a distribution shift?
3. Test soft sampling alternatives - does top-k/nucleus prevent collapse?
4. Verify Gumbel annealing - was the schedule applied correctly?

Usage:
    python immediate_diagnostics.py \
        --checkpoint ./checkpoints/ochre-v7.0.5-step95k.pt \
        --vqvae ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --device mps
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from analyze_checkpoint import load_checkpoint, get_test_actions


# ========================================
# Test 1: Collapsed Codebook Analysis
# ========================================

@torch.no_grad()
def test_1_analyze_collapsed_codebook(model, vqvae, z_start, test_actions, device):
    """
    Analyze which codes the model uses when collapsed.

    Returns:
        - Frequency distribution of codes used
        - Which codes are used (indices)
        - Semantic analysis of collapsed codes (if VQ-VAE embedding available)
    """
    print("\n" + "="*80)
    print("TEST 1: ANALYZE COLLAPSED CODEBOOK")
    print("="*80)

    results = {}

    # Run 20-step AR rollout for each action
    for action_name, action_vec in test_actions.items():
        temporal_buffer = []
        z_t = z_start
        all_codes = []

        for step in range(20):
            logits, new_state = model.step(z_t, action_vec, temporal_buffer)
            z_t = logits.argmax(dim=1)
            all_codes.extend(z_t[0].cpu().flatten().tolist())

            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > 8:
                temporal_buffer.pop(0)

        # Count code frequencies
        code_counter = Counter(all_codes)
        unique_codes = len(code_counter)

        # Get top 50 most frequent codes
        top_codes = code_counter.most_common(50)

        results[action_name] = {
            'unique_codes': unique_codes,
            'total_tokens': len(all_codes),
            'top_30_codes': [code for code, count in top_codes[:30]],
            'top_30_frequencies': [count for code, count in top_codes[:30]],
            'top_30_coverage': sum(count for _, count in top_codes[:30]) / len(all_codes),
        }

        print(f"\n{action_name}:")
        print(f"  - Unique codes used: {unique_codes} / 1024")
        print(f"  - Top 10 codes: {[code for code, _ in top_codes[:10]]}")
        print(f"  - Top 30 codes cover: {results[action_name]['top_30_coverage']:.1%} of tokens")

    # Analyze code clustering (are collapsed codes similar in embedding space?)
    print("\n" + "-"*80)
    print("CODEBOOK CLUSTERING ANALYSIS:")

    # Get VQ-VAE codebook embeddings
    try:
        codebook = vqvae.vq_vae.embedding.weight.data  # [1024, embed_dim]

        # Get collapsed codes from camera_right (representative)
        collapsed_codes = results['camera_right']['top_30_codes']

        # Compute pairwise distances between collapsed codes
        collapsed_embeddings = codebook[collapsed_codes]  # [30, embed_dim]
        distances = torch.cdist(collapsed_embeddings, collapsed_embeddings, p=2)  # [30, 30]

        avg_collapsed_distance = distances.triu(diagonal=1).sum() / (30 * 29 / 2)

        # Compare to average distance in full codebook (sample 30 random codes)
        random_codes = torch.randint(0, 1024, (30,))
        random_embeddings = codebook[random_codes]
        random_distances = torch.cdist(random_embeddings, random_embeddings, p=2)
        avg_random_distance = random_distances.triu(diagonal=1).sum() / (30 * 29 / 2)

        print(f"  - Avg distance between collapsed codes: {avg_collapsed_distance.item():.4f}")
        print(f"  - Avg distance between random 30 codes: {avg_random_distance.item():.4f}")
        print(f"  - Ratio (collapsed/random): {(avg_collapsed_distance / avg_random_distance).item():.4f}")

        if avg_collapsed_distance < 0.8 * avg_random_distance:
            print(f"  ⚠️  FINDING: Collapsed codes are CLUSTERED (closer than random)")
            print(f"      → Model converged to similar semantic region")
        else:
            print(f"  ✅ Collapsed codes are SCATTERED (similar to random)")
            print(f"      → Model uses diverse codes, just fewer of them")

        results['clustering'] = {
            'avg_collapsed_distance': float(avg_collapsed_distance.item()),
            'avg_random_distance': float(avg_random_distance.item()),
            'ratio': float((avg_collapsed_distance / avg_random_distance).item()),
        }

    except Exception as e:
        print(f"  ⚠️  Could not analyze clustering: {e}")

    return results


# ========================================
# Test 2: TF vs AR Logit Distribution
# ========================================

@torch.no_grad()
def test_2_tf_vs_ar_logit_distribution(model, z_start, action_vec, device):
    """
    Compare logit distributions for teacher-forced vs autoregressive predictions.

    Returns:
        - KL divergence between TF and AR logit distributions
        - Entropy of TF vs AR logits
        - Top-k overlap between TF and AR predictions
    """
    print("\n" + "="*80)
    print("TEST 2: TEACHER-FORCED VS AUTOREGRESSIVE LOGIT DISTRIBUTION")
    print("="*80)

    temporal_buffer = []

    # Step 1: Teacher-forced (using real seed codes)
    logits_tf, state_tf = model.step(z_start, action_vec, temporal_buffer)
    z_tf = logits_tf.argmax(dim=1)

    # Step 2: Autoregressive (using TF prediction as input)
    temporal_buffer_ar = [state_tf.detach()]
    logits_ar, _ = model.step(z_tf, action_vec, temporal_buffer_ar)

    # Flatten logits for analysis [B, codebook_size, H, W] -> [B*H*W, codebook_size]
    B, C, H, W = logits_tf.shape
    logits_tf_flat = logits_tf.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, codebook_size]
    logits_ar_flat = logits_ar.permute(0, 2, 3, 1).reshape(-1, C)

    # Convert to probabilities
    probs_tf = F.softmax(logits_tf_flat, dim=-1)
    probs_ar = F.softmax(logits_ar_flat, dim=-1)

    # Compute KL divergence (per token, then average)
    kl_divs = F.kl_div(
        F.log_softmax(logits_ar_flat, dim=-1),
        probs_tf,
        reduction='none'
    ).sum(dim=-1)  # [B*H*W]
    avg_kl = kl_divs.mean().item()

    # Compute entropy
    entropy_tf = -(probs_tf * torch.log(probs_tf + 1e-10)).sum(dim=-1).mean().item()
    entropy_ar = -(probs_ar * torch.log(probs_ar + 1e-10)).sum(dim=-1).mean().item()

    # Compute confidence (max probability)
    confidence_tf = probs_tf.max(dim=-1)[0].mean().item()
    confidence_ar = probs_ar.max(dim=-1)[0].mean().item()

    # Top-k overlap (how many of top-10 predictions are the same?)
    topk_tf = torch.topk(logits_tf_flat, k=10, dim=-1).indices  # [B*H*W, 10]
    topk_ar = torch.topk(logits_ar_flat, k=10, dim=-1).indices

    # Count overlaps
    overlaps = []
    for i in range(topk_tf.shape[0]):
        overlap = len(set(topk_tf[i].tolist()) & set(topk_ar[i].tolist()))
        overlaps.append(overlap)
    avg_overlap = np.mean(overlaps)

    print(f"\n  Teacher-Forced Logits:")
    print(f"    - Entropy: {entropy_tf:.4f}")
    print(f"    - Confidence (max prob): {confidence_tf:.4f}")

    print(f"\n  Autoregressive Logits:")
    print(f"    - Entropy: {entropy_ar:.4f}")
    print(f"    - Confidence (max prob): {confidence_ar:.4f}")

    print(f"\n  Distribution Shift Metrics:")
    print(f"    - KL divergence (AR || TF): {avg_kl:.4f}")
    print(f"    - Top-10 overlap: {avg_overlap:.1f} / 10")
    print(f"    - Entropy change (AR - TF): {entropy_ar - entropy_tf:+.4f}")

    if avg_kl > 0.5:
        print(f"\n  ⚠️  FINDING: Large KL divergence ({avg_kl:.4f}) → significant distribution shift")
        print(f"      → AR logits are very different from TF logits")
    else:
        print(f"\n  ✅ Moderate KL divergence ({avg_kl:.4f}) → distributions similar")

    if entropy_ar < 0.5 * entropy_tf:
        print(f"  ⚠️  FINDING: AR entropy ({entropy_ar:.4f}) << TF entropy ({entropy_tf:.4f})")
        print(f"      → AR predictions are much more confident/peaked")
    elif entropy_ar > 1.5 * entropy_tf:
        print(f"  ⚠️  FINDING: AR entropy ({entropy_ar:.4f}) >> TF entropy ({entropy_tf:.4f})")
        print(f"      → AR predictions are much more uncertain")

    return {
        'kl_divergence': float(avg_kl),
        'entropy_tf': float(entropy_tf),
        'entropy_ar': float(entropy_ar),
        'confidence_tf': float(confidence_tf),
        'confidence_ar': float(confidence_ar),
        'top10_overlap': float(avg_overlap),
    }


# ========================================
# Test 3: Soft Sampling Alternatives
# ========================================

@torch.no_grad()
def test_3_soft_sampling_alternatives(model, vqvae, z_start, action_vec, device):
    """
    Test if soft sampling (top-k, nucleus, temperature) prevents collapse.

    Returns:
        - Unique codes for argmax vs top-k vs nucleus sampling
        - Comparison of collapse rates
    """
    print("\n" + "="*80)
    print("TEST 3: SOFT SAMPLING ALTERNATIVES TO ARGMAX")
    print("="*80)

    results = {}

    sampling_strategies = {
        'argmax': lambda logits: logits.argmax(dim=1),
        'top_k_50': lambda logits: sample_top_k(logits, k=50),
        'top_k_100': lambda logits: sample_top_k(logits, k=100),
        'nucleus_0.9': lambda logits: sample_nucleus(logits, p=0.9),
        'temperature_0.8': lambda logits: sample_temperature(logits, temp=0.8),
    }

    for strategy_name, sample_fn in sampling_strategies.items():
        temporal_buffer = []
        z_t = z_start
        unique_codes_per_frame = []

        for step in range(20):
            logits, new_state = model.step(z_t, action_vec, temporal_buffer)
            z_t = sample_fn(logits)

            unique_count = z_t[0].unique().numel()
            unique_codes_per_frame.append(unique_count)

            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > 8:
                temporal_buffer.pop(0)

        frame_0_codes = unique_codes_per_frame[0]
        frame_1_codes = unique_codes_per_frame[1]
        frame_10_codes = unique_codes_per_frame[10]
        frame_19_codes = unique_codes_per_frame[19]

        first_step_drop = ((frame_0_codes - frame_1_codes) / frame_0_codes) * 100 if frame_0_codes > 0 else 0.0

        results[strategy_name] = {
            'frame_0_codes': frame_0_codes,
            'frame_1_codes': frame_1_codes,
            'frame_10_codes': frame_10_codes,
            'frame_19_codes': frame_19_codes,
            'first_step_drop_pct': float(first_step_drop),
            'all_frames': unique_codes_per_frame,
        }

        print(f"\n  {strategy_name}:")
        print(f"    - Frame 0: {frame_0_codes} codes")
        print(f"    - Frame 1: {frame_1_codes} codes ({first_step_drop:.1f}% drop)")
        print(f"    - Frame 10: {frame_10_codes} codes")
        print(f"    - Frame 19: {frame_19_codes} codes")

    # Compare strategies
    print("\n" + "-"*80)
    print("SAMPLING STRATEGY COMPARISON:")

    argmax_drop = results['argmax']['first_step_drop_pct']
    for strategy_name in ['top_k_50', 'nucleus_0.9', 'temperature_0.8']:
        drop = results[strategy_name]['first_step_drop_pct']
        improvement = argmax_drop - drop

        if improvement > 10:
            print(f"  ✅ {strategy_name}: {improvement:+.1f}% better than argmax (prevents collapse!)")
        elif improvement > 0:
            print(f"  ~ {strategy_name}: {improvement:+.1f}% better than argmax (minor improvement)")
        else:
            print(f"  ❌ {strategy_name}: {improvement:+.1f}% vs argmax (no improvement)")

    return results


def sample_top_k(logits, k=50):
    """Sample from top-k logits."""
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(logits_flat, k=k, dim=-1)

    # Create mask for top-k
    mask = torch.full_like(logits_flat, float('-inf'))
    mask.scatter_(1, topk_indices, topk_values)

    # Sample from masked distribution
    probs = F.softmax(mask, dim=-1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B*H*W]

    return samples.reshape(B, H, W)


def sample_nucleus(logits, p=0.9):
    """Sample from nucleus (top-p) distribution."""
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)

    # Sort probabilities
    probs = F.softmax(logits_flat, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Find cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask
    mask = torch.full_like(logits_flat, float('-inf'))
    for i in range(logits_flat.shape[0]):
        valid_indices = sorted_indices[i][~sorted_indices_to_remove[i]]
        mask[i, valid_indices] = logits_flat[i, valid_indices]

    # Sample
    probs = F.softmax(mask, dim=-1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return samples.reshape(B, H, W)


def sample_temperature(logits, temp=0.8):
    """Sample with temperature scaling."""
    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)

    # Apply temperature
    logits_scaled = logits_flat / temp

    # Sample
    probs = F.softmax(logits_scaled, dim=-1)
    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return samples.reshape(B, H, W)


# ========================================
# Test 4: Verify Gumbel Annealing
# ========================================

def test_4_verify_gumbel_annealing(checkpoint_path):
    """
    Check if Gumbel annealing was applied correctly during training.

    Returns:
        - Whether checkpoint has tau value stored
        - Expected vs actual tau at checkpoint step
    """
    print("\n" + "="*80)
    print("TEST 4: VERIFY GUMBEL ANNEALING WAS APPLIED")
    print("="*80)

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract global step
    global_step = ckpt.get('global_step', ckpt.get('step', None))

    if global_step is None:
        print("\n  ⚠️  Could not find global_step in checkpoint")
        return {'error': 'No global_step found'}

    print(f"\n  Checkpoint global_step: {global_step}")

    # Check for stored tau value
    tau_stored = ckpt.get('gumbel_tau', None)

    if tau_stored is not None:
        print(f"  Stored Gumbel tau: {tau_stored:.4f}")
    else:
        print(f"  ⚠️  No stored Gumbel tau in checkpoint")

    # Compute expected tau for v7.0.2 schedule (tau: 1.0 → 0.1 over 20k steps)
    # From changelog: v7.0.2 used tau annealing 1.0→0.1 over 20k steps
    decay_steps = 20000
    tau_start = 1.0
    tau_end = 0.1

    if global_step <= decay_steps:
        progress = global_step / decay_steps
        expected_tau = tau_start + (tau_end - tau_start) * progress
    else:
        expected_tau = tau_end

    print(f"\n  Expected tau (1.0→0.1 over 20k schedule):")
    print(f"    - At step {global_step}: {expected_tau:.4f}")

    if tau_stored is not None:
        diff = abs(tau_stored - expected_tau)
        if diff < 0.05:
            print(f"  ✅ Stored tau matches expected ({diff:.4f} difference)")
        else:
            print(f"  ⚠️  Stored tau differs from expected by {diff:.4f}")
            print(f"      → Annealing schedule may have been incorrect")
    else:
        print(f"  ⚠️  Cannot verify - no tau stored in checkpoint")

    # Check for other training state
    optimizer_state = ckpt.get('optimizer_state', None)
    if optimizer_state:
        print(f"\n  ✅ Optimizer state found (training state preserved)")
    else:
        print(f"\n  ⚠️  No optimizer state (checkpoint may be inference-only)")

    return {
        'global_step': global_step,
        'tau_stored': float(tau_stored) if tau_stored is not None else None,
        'tau_expected': float(expected_tau),
        'has_optimizer_state': optimizer_state is not None,
    }


# ========================================
# Main
# ========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--vqvae', required=True, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--device', default='mps', help='Device to use')
    parser.add_argument('--output', default='./diagnostics/immediate-tests/results.json', help='Output JSON path')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("\n" + "="*80)
    print("IMMEDIATE DIAGNOSTIC TESTS: FIRST-STEP COLLAPSE ROOT CAUSE")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"VQ-VAE: {args.vqvae}")
    print(f"Device: {args.device}\n")

    # Load checkpoint
    print("Loading checkpoint...")
    model, vqvae, model_config = load_checkpoint(args.checkpoint, args.vqvae, device)

    # Create test actions
    test_actions = get_test_actions(device)

    # Generate seed frame
    codebook_size = model_config['codebook_size']
    latent_h = model_config['H']
    latent_w = model_config['W']
    z_start = torch.randint(0, codebook_size, (1, latent_h, latent_w), device=device)

    # Use camera_right as representative action for tests 2-3
    action_vec = test_actions['camera_right']

    # Run all 4 tests
    all_results = {}

    # Test 1: Collapsed codebook analysis
    all_results['test_1_collapsed_codebook'] = test_1_analyze_collapsed_codebook(
        model, vqvae, z_start, test_actions, device
    )

    # Test 2: TF vs AR logit distribution
    all_results['test_2_tf_vs_ar_logits'] = test_2_tf_vs_ar_logit_distribution(
        model, z_start, action_vec, device
    )

    # Test 3: Soft sampling alternatives
    all_results['test_3_soft_sampling'] = test_3_soft_sampling_alternatives(
        model, vqvae, z_start, action_vec, device
    )

    # Test 4: Verify Gumbel annealing
    all_results['test_4_gumbel_annealing'] = test_4_verify_gumbel_annealing(
        args.checkpoint
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ All tests complete! Results saved to: {output_path}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
