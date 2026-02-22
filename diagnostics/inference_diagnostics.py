"""
Inference-Time Diagnostics: Understanding "Mush Prediction"

This script runs 6 diagnostic tools to understand why the model predicts blurry/mushy
outputs instead of sharp Minecraft scenes:

1. Logit Entropy/Confidence Tracking - Is the model uncertain?
2. Code Usage Distribution Analysis - Using generic/safe codes?
3. Top-k Sampling Consistency Test - Is top-k=50 adding noise?
4. Action Sensitivity Measurement - Do actions meaningfully change outputs?
5. Per-Frame Quality Degradation Tracking - When does degradation start?
6. VQ-VAE Decode Quality Check - Is decoder the bottleneck?

Usage:
    python diagnostics/inference_diagnostics.py \
        --checkpoint ./checkpoints/ochre-v7.0.5-step95k.pt \
        --vqvae_ckpt ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --context_npz ./contexts/starter_001.npz \
        --recency_decay 0.9 \
        --topk 50 \
        --temperature 1.0 \
        --output_dir ./diagnostics/inference-analysis/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_transformer import MinecraftConvTransformer
from vq_vae.vq_vae import VQVAE
from action_encoding import encode_action_v5_np
from diagnostics.analyze_checkpoint import load_checkpoint, get_test_actions


# ========================================
# Diagnostic 1: Logit Entropy/Confidence
# ========================================

@torch.no_grad()
def diagnostic_logit_entropy_over_time(model, z_start, action, num_steps=20, topk=50, temperature=1.0, recency_decay=1.0):
    """
    Track prediction confidence/uncertainty over AR rollout.

    Returns:
        - entropies: Per-frame average entropy (high = uncertain)
        - max_probs: Per-frame average max probability (high = confident)
        - input_unique_codes: Unique codes in the input token grid z_t
        - argmax_unique_codes: Unique codes in argmax(logits) BEFORE sampling
        - unique_codes: Unique codes in the selected next tokens (sampled/greedy)
    """
    entropies = []
    max_probs = []
    input_unique_codes = []
    argmax_unique_codes = []
    unique_codes = []
    temporal_buffer = []

    z_t = z_start

    for step in range(num_steps):
        # Track diversity of the conditioning input at this step
        input_unique_codes.append(z_t.unique().numel())

        # Get logits
        logits, new_state = model.step(z_t, action, temporal_buffer)

        # Compute probability distribution
        probs = F.softmax(logits / temperature, dim=1)  # (B, codebook_size, H, W)

        # Entropy per spatial position (high = uncertain, low = confident)
        entropy_per_position = -(probs * torch.log(probs + 1e-9)).sum(dim=1)  # (B, H, W)
        entropies.append(entropy_per_position.mean().item())

        # Max probability per position (high = confident, low = guessing)
        max_prob_per_position = probs.max(dim=1)[0]  # (B, H, W)
        max_probs.append(max_prob_per_position.mean().item())

        # Track diversity of greedy (argmax) prediction BEFORE any sampling
        z_argmax = logits.argmax(dim=1)
        argmax_unique_codes.append(z_argmax.unique().numel())

        # Sample next token
        if topk > 0:
            # Top-k sampling
            topk_probs, topk_indices = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

            # Sample from top-k
            B, _, H, W = topk_probs.shape
            topk_probs_flat = topk_probs.permute(0, 2, 3, 1).reshape(B*H*W, topk)
            sampled_indices = torch.multinomial(topk_probs_flat, 1).reshape(B, H, W)
            z_t = topk_indices.permute(0, 2, 3, 1).reshape(B*H*W, topk)[torch.arange(B*H*W), sampled_indices.flatten()].reshape(B, H, W)
        else:
            # Argmax (greedy)
            z_t = z_argmax

        # Count unique codes
        unique_codes.append(z_t.unique().numel())

        # Update temporal buffer
        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > 8:
            temporal_buffer.pop(0)

    return {
        'entropy': np.array(entropies),
        'max_prob': np.array(max_probs),
        'input_unique_codes': np.array(input_unique_codes),
        'argmax_unique_codes': np.array(argmax_unique_codes),
        'unique_codes': np.array(unique_codes),
    }


# ========================================
# Diagnostic 2: Code Distribution Analysis
# ========================================

@torch.no_grad()
def diagnostic_code_distribution(model, vqvae, z_start, action, num_steps=20, topk=50, temperature=1.0, recency_decay=1.0):
    """
    Analyze which codes are predicted and how they're distributed.

    Returns:
        - num_unique_codes: Total unique codes used across rollout
        - top_10_codes: List of (code_index, frequency) tuples
        - code_concentration: Fraction of tokens using most common code
        - code_histogram: Full histogram of code usage
    """
    temporal_buffer = []
    z_t = z_start
    all_codes = []

    for step in range(num_steps):
        logits, new_state = model.step(z_t, action, temporal_buffer)

        # Sample next token
        probs = F.softmax(logits / temperature, dim=1)

        if topk > 0:
            # Top-k sampling
            topk_probs, topk_indices = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

            B, _, H, W = topk_probs.shape
            topk_probs_flat = topk_probs.permute(0, 2, 3, 1).reshape(B*H*W, topk)
            sampled_indices = torch.multinomial(topk_probs_flat, 1).reshape(B, H, W)
            z_t = topk_indices.permute(0, 2, 3, 1).reshape(B*H*W, topk)[torch.arange(B*H*W), sampled_indices.flatten()].reshape(B, H, W)
        else:
            z_t = logits.argmax(dim=1)

        # Collect codes
        all_codes.extend(z_t[0].cpu().flatten().tolist())

        # Update buffer
        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > 8:
            temporal_buffer.pop(0)

    # Analyze distribution
    code_counter = Counter(all_codes)
    total_predictions = len(all_codes)

    # Get top codes
    top_codes = code_counter.most_common(10)

    return {
        'num_unique_codes': len(code_counter),
        'total_tokens': total_predictions,
        'top_10_codes': [(code, count/total_predictions) for code, count in top_codes],
        'code_concentration': max(code_counter.values()) / total_predictions,
        'code_histogram': dict(code_counter),
    }


# ========================================
# Diagnostic 3: Top-k Consistency Test
# ========================================

@torch.no_grad()
def diagnostic_topk_consistency(model, vqvae, z_start, action, num_samples=10, num_steps=10, topk=50, temperature=1.0, recency_decay=1.0):
    """
    Test if top-k sampling produces consistent outputs or random jitter.

    Returns:
        - code_overlap: Average fraction of codes matching between samples
        - mean_variance: Average per-pixel variance across samples
        - consistency_score: Overall consistency metric (higher = more consistent)
    """
    # Run model multiple times with same input, different random samples
    all_final_codes = []

    for sample_idx in range(num_samples):
        temporal_buffer = []
        z_t = z_start.clone()

        for step in range(num_steps):
            logits, new_state = model.step(z_t, action, temporal_buffer)

            # Sample with top-k
            probs = F.softmax(logits / temperature, dim=1)

            if topk > 0:
                topk_probs, topk_indices = probs.topk(topk, dim=1)
                topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

                B, _, H, W = topk_probs.shape
                topk_probs_flat = topk_probs.permute(0, 2, 3, 1).reshape(B*H*W, topk)
                sampled_indices = torch.multinomial(topk_probs_flat, 1).reshape(B, H, W)
                z_t = topk_indices.permute(0, 2, 3, 1).reshape(B*H*W, topk)[torch.arange(B*H*W), sampled_indices.flatten()].reshape(B, H, W)
            else:
                z_t = logits.argmax(dim=1)

            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > 8:
                temporal_buffer.pop(0)

        all_final_codes.append(z_t.cpu())

    # Measure consistency via code overlap
    # For each position, what fraction of samples agree?
    all_final_codes = torch.stack(all_final_codes, dim=0)  # (num_samples, B, H, W)

    # For each spatial position, compute mode (most common code)
    B, H, W = all_final_codes.shape[1:]
    code_overlap = 0.0

    for b in range(B):
        for h in range(H):
            for w in range(W):
                codes_at_position = all_final_codes[:, b, h, w]  # (num_samples,)
                mode_count = (codes_at_position == codes_at_position.mode()[0]).sum().item()
                code_overlap += mode_count / num_samples

    code_overlap /= (B * H * W)

    # Compute variance (how much do codes differ?)
    variance = all_final_codes.float().var(dim=0).mean().item()

    return {
        'code_overlap': code_overlap,  # 1.0 = all samples agree, 0.1 = totally random
        'code_variance': variance,  # Low = consistent, high = jittery
        'consistency_score': code_overlap,  # Higher = better
    }


# ========================================
# Diagnostic 4: Action Sensitivity
# ========================================

@torch.no_grad()
def diagnostic_action_sensitivity_inference(model, vqvae, z_start, num_steps=10, topk=50, temperature=1.0, recency_decay=1.0):
    """
    Measure how much predictions change with different actions.

    Returns:
        - static_unique_codes: Codes used with no action
        - camera_left_unique_codes: Codes used with camera left
        - camera_right_unique_codes: Codes used with camera right
        - camera_left_vs_static_overlap: Fraction of codes overlapping
        - camera_right_vs_static_overlap: Fraction of codes overlapping
        - left_vs_right_overlap: Fraction of codes overlapping
    """
    device = z_start.device

    # Define actions
    actions = {
        'static': torch.tensor(encode_action_v5_np(), device=device).unsqueeze(0),
        'camera_left': torch.tensor(encode_action_v5_np(yaw_raw=-0.5), device=device).unsqueeze(0),
        'camera_right': torch.tensor(encode_action_v5_np(yaw_raw=0.5), device=device).unsqueeze(0),
    }

    # Run rollouts for each action
    rollout_codes = {}
    for name, action in actions.items():
        temporal_buffer = []
        z_t = z_start.clone()
        final_codes = []

        for step in range(num_steps):
            logits, new_state = model.step(z_t, action, temporal_buffer)

            probs = F.softmax(logits / temperature, dim=1)

            if topk > 0:
                topk_probs, topk_indices = probs.topk(topk, dim=1)
                topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

                B, _, H, W = topk_probs.shape
                topk_probs_flat = topk_probs.permute(0, 2, 3, 1).reshape(B*H*W, topk)
                sampled_indices = torch.multinomial(topk_probs_flat, 1).reshape(B, H, W)
                z_t = topk_indices.permute(0, 2, 3, 1).reshape(B*H*W, topk)[torch.arange(B*H*W), sampled_indices.flatten()].reshape(B, H, W)
            else:
                z_t = logits.argmax(dim=1)

            final_codes.append(z_t.cpu())

            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > 8:
                temporal_buffer.pop(0)

        # Concatenate all codes from rollout
        final_codes = torch.cat(final_codes, dim=0)  # (num_steps, H, W)
        rollout_codes[name] = set(final_codes.flatten().tolist())

    # Compute overlaps
    static_codes = rollout_codes['static']
    left_codes = rollout_codes['camera_left']
    right_codes = rollout_codes['camera_right']

    left_vs_static_overlap = len(left_codes & static_codes) / len(left_codes | static_codes)
    right_vs_static_overlap = len(right_codes & static_codes) / len(right_codes | static_codes)
    left_vs_right_overlap = len(left_codes & right_codes) / len(left_codes | right_codes)

    return {
        'static_unique_codes': len(static_codes),
        'camera_left_unique_codes': len(left_codes),
        'camera_right_unique_codes': len(right_codes),
        'camera_left_vs_static_overlap': left_vs_static_overlap,
        'camera_right_vs_static_overlap': right_vs_static_overlap,
        'left_vs_right_overlap': left_vs_right_overlap,
        'action_effect_magnitude': 1.0 - left_vs_static_overlap,  # Higher = more effect
    }


# ========================================
# Diagnostic 5: Per-Frame Quality Degradation
# ========================================

@torch.no_grad()
def diagnostic_per_frame_quality(model, vqvae, z_start, action, num_steps=20, topk=50, temperature=1.0, recency_decay=1.0):
    """
    Track when and how quality degrades over AR rollout.

    Returns:
        - input_unique_codes: Array of unique codes in input z_t at each step
        - argmax_unique_codes: Array of unique codes in argmax(logits) BEFORE sampling
        - unique_codes: Array of unique codes in selected next tokens (sampled/greedy)
        - sharpness_per_frame: Array of sharpness (Laplacian variance) at each frame
        - degradation_onset_frame: Frame number where quality drops significantly
    """
    input_unique_codes_per_frame = []
    argmax_unique_codes_per_frame = []
    unique_codes_per_frame = []
    sharpness_per_frame = []
    temporal_buffer = []
    z_t = z_start

    # Laplacian kernel for sharpness
    laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32, device=z_start.device)

    for step in range(num_steps):
        input_unique_codes_per_frame.append(z_t.unique().numel())

        logits, new_state = model.step(z_t, action, temporal_buffer)

        probs = F.softmax(logits / temperature, dim=1)
        z_argmax = logits.argmax(dim=1)
        argmax_unique_codes_per_frame.append(z_argmax.unique().numel())

        if topk > 0:
            topk_probs, topk_indices = probs.topk(topk, dim=1)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)

            B, _, H, W = topk_probs.shape
            topk_probs_flat = topk_probs.permute(0, 2, 3, 1).reshape(B*H*W, topk)
            sampled_indices = torch.multinomial(topk_probs_flat, 1).reshape(B, H, W)
            z_t = topk_indices.permute(0, 2, 3, 1).reshape(B*H*W, topk)[torch.arange(B*H*W), sampled_indices.flatten()].reshape(B, H, W)
        else:
            z_t = z_argmax

        # Count unique codes
        unique_codes_per_frame.append(z_t.unique().numel())

        # Decode and measure sharpness
        with torch.no_grad():
            img_pred = vqvae.decode_code(z_t)  # (B, C, H_img, W_img)

            # Convert to grayscale
            gray = img_pred.mean(dim=1, keepdim=True)  # (B, 1, H_img, W_img)

            # Apply Laplacian
            laplacian = F.conv2d(gray, laplacian_kernel, padding=1)

            # Compute variance (higher = sharper)
            sharpness = laplacian.var().item()
            sharpness_per_frame.append(sharpness)

        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > 8:
            temporal_buffer.pop(0)

    # Find degradation onset (when sharpness drops by > 30%)
    sharpness_array = np.array(sharpness_per_frame)
    baseline_sharpness = sharpness_array[0]
    degradation_threshold = baseline_sharpness * 0.7  # 30% drop

    degradation_onset = num_steps  # Default: no degradation
    for i in range(1, num_steps):
        if sharpness_array[i] < degradation_threshold:
            degradation_onset = i
            break

    return {
        'input_unique_codes': np.array(input_unique_codes_per_frame),
        'argmax_unique_codes': np.array(argmax_unique_codes_per_frame),
        'unique_codes': np.array(unique_codes_per_frame),
        'sharpness': sharpness_array,
        'degradation_onset_frame': degradation_onset,
        'baseline_sharpness': baseline_sharpness,
        'final_sharpness': sharpness_array[-1],
        'sharpness_retention': sharpness_array[-1] / baseline_sharpness,
    }


# ========================================
# Diagnostic 6: VQ-VAE Decode Quality Check
# ========================================

@torch.no_grad()
def diagnostic_vqvae_decode_quality(vqvae, z_pred):
    """
    Check if VQ-VAE decoder is the bottleneck or if codes are bad.

    Returns:
        - pred_code_diversity: Number of unique codes
        - most_common_code: Index of most frequently used code
        - most_common_code_frequency: Fraction of tokens using most common code
    """
    # Decode predicted codes
    img_pred = vqvae.decode_code(z_pred)  # (B, C, H, W)

    # Count code diversity
    unique_codes = z_pred.unique()
    code_diversity = unique_codes.numel()

    # Find most common code
    code_counter = Counter(z_pred.cpu().flatten().tolist())
    most_common_code, most_common_count = code_counter.most_common(1)[0]
    total_tokens = z_pred.numel()

    return {
        'pred_code_diversity': code_diversity,
        'most_common_code': most_common_code,
        'most_common_code_frequency': most_common_count / total_tokens,
        'total_tokens': total_tokens,
    }


# ========================================
# Main Diagnostic Runner
# ========================================

def run_all_diagnostics(model, vqvae, z_start, test_actions, config):
    """
    Run all 6 diagnostic categories with specified configuration.

    Args:
        model: Loaded world model
        vqvae: Loaded VQ-VAE
        z_start: Initial latent codes (B, H, W)
        test_actions: Dict of action tensors
        config: Dict with topk, temperature, recency_decay

    Returns:
        Dict with all diagnostic results
    """
    results = {}

    topk = config['topk']
    temperature = config['temperature']
    recency_decay = config['recency_decay']
    entropy_steps = int(config.get("entropy_steps", 20))
    code_dist_steps = int(config.get("code_dist_steps", 20))
    consistency_samples = int(config.get("consistency_samples", 10))
    consistency_steps = int(config.get("consistency_steps", 10))
    action_sensitivity_steps = int(config.get("action_sensitivity_steps", 10))
    quality_steps = int(config.get("quality_steps", 20))

    print("\n" + "="*80)
    print("RUNNING INFERENCE DIAGNOSTICS")
    print(f"Config: topk={topk}, temperature={temperature}, recency_decay={recency_decay}")
    print("="*80)

    # Diagnostic 1: Logit Entropy/Confidence
    print("\n[1/6] Running logit entropy/confidence analysis...")
    for action_name, action in test_actions.items():
        results[f'entropy/{action_name}'] = diagnostic_logit_entropy_over_time(
            model, z_start, action, num_steps=entropy_steps, topk=topk, temperature=temperature, recency_decay=recency_decay
        )
        print(f"  {action_name}: avg_entropy={results[f'entropy/{action_name}']['entropy'].mean():.3f}, avg_max_prob={results[f'entropy/{action_name}']['max_prob'].mean():.3f}")

    # Diagnostic 2: Code Distribution
    print("\n[2/6] Running code distribution analysis...")
    for action_name, action in test_actions.items():
        results[f'code_dist/{action_name}'] = diagnostic_code_distribution(
            model, vqvae, z_start, action, num_steps=code_dist_steps, topk=topk, temperature=temperature, recency_decay=recency_decay
        )
        print(f"  {action_name}: unique_codes={results[f'code_dist/{action_name}']['num_unique_codes']}, concentration={results[f'code_dist/{action_name}']['code_concentration']:.3f}")

    # Diagnostic 3: Top-k Consistency
    print("\n[3/6] Running top-k consistency test...")
    results['consistency'] = diagnostic_topk_consistency(
        model, vqvae, z_start, test_actions['camera_right'], num_samples=consistency_samples, num_steps=consistency_steps, topk=topk, temperature=temperature, recency_decay=recency_decay
    )
    print(f"  consistency_score={results['consistency']['consistency_score']:.3f}, code_overlap={results['consistency']['code_overlap']:.3f}")

    # Diagnostic 4: Action Sensitivity
    print("\n[4/6] Running action sensitivity analysis...")
    results['action_sensitivity'] = diagnostic_action_sensitivity_inference(
        model, vqvae, z_start, num_steps=action_sensitivity_steps, topk=topk, temperature=temperature, recency_decay=recency_decay
    )
    print(f"  action_effect_magnitude={results['action_sensitivity']['action_effect_magnitude']:.3f}")

    # Diagnostic 5: Per-Frame Quality Degradation
    print("\n[5/6] Running per-frame quality degradation analysis...")
    for action_name, action in test_actions.items():
        results[f'per_frame_quality/{action_name}'] = diagnostic_per_frame_quality(
            model, vqvae, z_start, action, num_steps=quality_steps, topk=topk, temperature=temperature, recency_decay=recency_decay
        )
        print(f"  {action_name}: degradation_onset_frame={results[f'per_frame_quality/{action_name}']['degradation_onset_frame']}, sharpness_retention={results[f'per_frame_quality/{action_name}']['sharpness_retention']:.2%}")

    # Diagnostic 6: VQ-VAE Decode Quality
    print("\n[6/6] Running VQ-VAE decode quality check...")
    results['vqvae_quality'] = diagnostic_vqvae_decode_quality(vqvae, z_start)
    print(f"  code_diversity={results['vqvae_quality']['pred_code_diversity']}, most_common_freq={results['vqvae_quality']['most_common_code_frequency']:.3f}")

    return results


# ========================================
# Result Analysis & Interpretation
# ========================================

def analyze_results(results, config):
    """
    Interpret diagnostic results and provide actionable insights.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS SUMMARY")
    print("="*80)

    # Check for Scenario A: Model is Uncertain (High Entropy)
    avg_entropy = np.mean([results[f'entropy/{action}']['entropy'].mean() for action in ['static', 'camera_left', 'camera_right']])
    avg_max_prob = np.mean([results[f'entropy/{action}']['max_prob'].mean() for action in ['static', 'camera_left', 'camera_right']])

    print("\n1. PREDICTION CONFIDENCE:")
    print(f"   - Average entropy: {avg_entropy:.3f} (expect ~2.0 for sharp predictions)")
    print(f"   - Average max_prob: {avg_max_prob:.3f} (expect ~0.7-0.9 for confident predictions)")

    if avg_entropy > 5.0:
        print("   ⚠️  SCENARIO A: HIGH ENTROPY - Model is uncertain, spreading probability across many codes")
        print("       → Recommended fix: Add commitment loss, use temperature < 1.0")
    elif avg_max_prob < 0.3:
        print("   ⚠️  SCENARIO A: LOW CONFIDENCE - Model has no confident predictions")
        print("       → Recommended fix: Add commitment loss, increase Gumbel annealing duration")
    else:
        print("   ✅  Confidence levels reasonable")

    # Check for Scenario B: Top-k is Too Noisy
    consistency = results['consistency']['consistency_score']

    print("\n2. TOP-K CONSISTENCY:")
    print(f"   - Consistency score: {consistency:.3f} (1.0 = all samples agree, 0.1 = random)")

    if consistency < 0.5:
        print("   ⚠️  SCENARIO B: TOP-K TOO NOISY - High variance across samples")
        print(f"       → Recommended fix: Reduce k from {config['topk']} to 25, or use nucleus sampling (top-p)")
    else:
        print("   ✅  Top-k sampling produces consistent results")

    # Check for Scenario C: Using Generic/Safe Codes
    avg_unique_codes = np.mean([results[f'code_dist/{action}']['num_unique_codes'] for action in ['static', 'camera_left', 'camera_right']])
    avg_concentration = np.mean([results[f'code_dist/{action}']['code_concentration'] for action in ['static', 'camera_left', 'camera_right']])

    print("\n3. CODE USAGE DISTRIBUTION:")
    print(f"   - Average unique codes: {avg_unique_codes:.1f} / 1024")
    print(f"   - Average code concentration: {avg_concentration:.3f} (fraction using most common code)")

    if avg_unique_codes < 40:
        print("   ⚠️  SCENARIO C: USING NARROW SET OF CODES - May be generic/safe codes")
        print("       → Recommended fix: Add code diversity loss, anti-repetition loss")
    elif avg_concentration > 0.15:
        print("   ⚠️  SCENARIO C: HIGH CODE CONCENTRATION - One code dominates")
        print("       → Recommended fix: Penalize repeated code usage")
    else:
        print("   ✅  Code usage distribution reasonable")

    # Check for Scenario D: Actions Don't Change Predictions
    action_effect = results['action_sensitivity']['action_effect_magnitude']

    print("\n4. ACTION SENSITIVITY:")
    print(f"   - Action effect magnitude: {action_effect:.3f} (expect > 0.15 for meaningful effect)")

    if action_effect < 0.05:
        print("   ⚠️  SCENARIO D: ACTIONS BARELY CHANGE PREDICTIONS")
        print("       → Recommended fix: Separate optimizer for action pathway, consistency loss")
    else:
        print("   ✅  Actions have meaningful effect on predictions")

    # Check for degradation onset
    avg_degradation_onset = np.mean([results[f'per_frame_quality/{action}']['degradation_onset_frame'] for action in ['static', 'camera_left', 'camera_right']])
    avg_sharpness_retention = np.mean([results[f'per_frame_quality/{action}']['sharpness_retention'] for action in ['static', 'camera_left', 'camera_right']])

    print("\n5. QUALITY DEGRADATION:")
    print(f"   - Average degradation onset: frame {avg_degradation_onset:.1f}")
    print(f"   - Average sharpness retention: {avg_sharpness_retention:.2%}")

    if avg_degradation_onset < 5:
        print("   ⚠️  SEVERE DEGRADATION - Quality drops within 5 frames")
    elif avg_sharpness_retention < 0.5:
        print("   ⚠️  POOR SHARPNESS RETENTION - More than 50% loss over rollout")
    else:
        print("   ✅  Quality degradation acceptable")

    # VQ-VAE quality check
    vqvae_diversity = results['vqvae_quality']['pred_code_diversity']
    vqvae_concentration = results['vqvae_quality']['most_common_code_frequency']

    print("\n6. VQ-VAE DECODE QUALITY:")
    print(f"   - Initial code diversity: {vqvae_diversity}")
    print(f"   - Most common code frequency: {vqvae_concentration:.3f}")

    if vqvae_diversity > 60 and avg_unique_codes < 40:
        print("   ⚠️  SCENARIO E: VQ-VAE BOTTLENECK - Codes diverse initially but model doesn't maintain")
    else:
        print("   ✅  VQ-VAE decode quality acceptable")

    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS:")
    print("="*80)

    recommendations = []

    if avg_entropy > 5.0 or avg_max_prob < 0.3:
        recommendations.append("1. Add commitment loss to force confident predictions")
        recommendations.append("2. Use temperature < 1.0 (e.g., 0.8) during AR sampling")

    if consistency < 0.5:
        recommendations.append(f"3. Reduce top-k from {config['topk']} to 25 or use nucleus sampling")

    if avg_unique_codes < 40 or avg_concentration > 0.15:
        recommendations.append("4. Add code diversity loss (encourage full codebook usage)")

    if action_effect < 0.05:
        recommendations.append("5. Implement separate optimizer for action pathway")
        recommendations.append("6. Add multi-rollout consistency loss")

    if avg_degradation_onset < 5:
        recommendations.append("7. Investigate initial prediction quality (may need VQ-VAE fixes)")

    if not recommendations:
        recommendations.append("No critical issues detected - model behavior is reasonable")

    for rec in recommendations:
        print(f"   {rec}")

    print("\n")


# ========================================
# Main Entry Point
# ========================================

def main():
    parser = argparse.ArgumentParser(description="Inference-time diagnostics for mush prediction")
    parser.add_argument("--checkpoint", required=True, help="Path to world model checkpoint")
    parser.add_argument("--vqvae_ckpt", required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--context_npz", help="Optional preprocessed .npz for initial tokens")
    parser.add_argument("--topk", type=int, default=50, help="Top-k sampling cutoff (0 = greedy)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--recency_decay", type=float, default=1.0, help="Temporal attention recency decay")
    parser.add_argument("--output_dir", default="./diagnostics/inference-analysis", help="Output directory for results")
    parser.add_argument("--device", default="mps", help="Device to use (cuda/mps/cpu)")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load checkpoint
    model, vqvae, model_kwargs = load_checkpoint(args.checkpoint, args.vqvae_ckpt, device)

    # Update model with recency_decay if needed
    if hasattr(model, 'temporal_attn') and hasattr(model.temporal_attn, 'recency_decay'):
        model.temporal_attn.recency_decay = args.recency_decay
        print(f"Set recency_decay to {args.recency_decay}")

    # Get test actions
    test_actions = get_test_actions(device)

    # Get initial latent codes
    if args.context_npz:
        # Load from context file
        context_data = np.load(args.context_npz)
        z_start = torch.from_numpy(context_data['tokens'][0]).long().to(device).unsqueeze(0)
        print(f"Loaded initial codes from {args.context_npz}")
    else:
        # Use random codes
        H, W = model_kwargs['H'], model_kwargs['W']
        z_start = torch.randint(0, model_kwargs['codebook_size'], (1, H, W), device=device)
        print("Using random initial codes")

    # Configuration
    config = {
        'topk': args.topk,
        'temperature': args.temperature,
        'recency_decay': args.recency_decay,
    }

    # Run diagnostics
    results = run_all_diagnostics(model, vqvae, z_start, test_actions, config)

    # Analyze results
    analyze_results(results, config)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    results_serializable[key][k] = v.tolist()
                else:
                    results_serializable[key][k] = v
        else:
            results_serializable[key] = value

    # Save to JSON
    output_path = os.path.join(args.output_dir, f"inference_diagnostics_k{args.topk}_t{args.temperature}_rd{args.recency_decay}.json")
    with open(output_path, 'w') as f:
        json.dump({
            'config': config,
            'results': results_serializable,
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
