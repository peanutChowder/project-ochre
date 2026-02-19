"""
Enhanced Diagnostics for Checkpoint Analysis

This module provides comprehensive diagnostic functions to objectively analyze
world model checkpoints and answer critical questions about failure modes.

Three core questions addressed:
1. Why is there no response to WASD+jump? (action pathway health)
2. Why does performance degrade on medium rollouts? (stability analysis)
3. Why can't model bridge gap from single-frame to medium-rollout quality? (TF-AR gap)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import lpips


# Initialize LPIPS network (using AlexNet backend)
lpips_fn = lpips.LPIPS(net='alex')


@torch.no_grad()
def diagnostic_per_frame_ar_quality(
    model,
    vqvae,
    z_start: torch.Tensor,
    action_vec: torch.Tensor,
    num_steps: int = 20,
    z_gt_seq: Optional[torch.Tensor] = None,
    temporal_context_len: int = 8
) -> Dict[str, np.ndarray]:
    """
    Track quality degradation frame-by-frame during AR rollout.

    Answers Q2: At which frame does quality start degrading?

    Args:
        model: World model (MinecraftConvTransformer)
        vqvae: VQ-VAE for decoding frames
        z_start: Initial latent token grid [1, H, W]
        action_vec: Action to apply at each step [1, action_dim]
        num_steps: Number of AR rollout steps
        z_gt_seq: Optional ground truth sequence for LPIPS computation [num_steps, H, W]
        temporal_context_len: Max temporal buffer size

    Returns:
        Dict with arrays of length num_steps:
            - 'unique_codes': Per-frame unique token count
            - 'film_camera_gate': Per-frame camera gate L2 magnitude
            - 'film_movement_gate': Per-frame movement gate L2 magnitude
            - 'temporal_attn_entropy': Per-frame attention entropy
            - 'lpips': Per-frame LPIPS vs ground truth (if z_gt_seq provided)
    """
    device = z_start.device
    model.eval()
    vqvae.eval()
    global lpips_fn
    lpips_fn = lpips_fn.to(device)

    temporal_buffer = []
    z_t = z_start

    results = {
        'unique_codes': [],
        'film_camera_gate': [],
        'film_movement_gate': [],
        'temporal_attn_entropy': [],
        'lpips': [] if z_gt_seq is not None else None
    }

    for step in range(num_steps):
        # 1. Count unique codes
        unique_count = z_t[0].unique().numel()
        results['unique_codes'].append(unique_count)

        # 2. Compute FiLM gate magnitudes
        cam_l2_sum = 0.0
        mov_l2_sum = 0.0
        count = 0
        for block in model.blocks:
            # Attention pathway
            p_attn = block.adaln_attn.action_to_params(action_vec, split_paths=True)
            cam_l2_sum += p_attn["camera"]["gate_raw"].pow(2).mean().sqrt().item()
            mov_l2_sum += p_attn["movement"]["gate_raw"].pow(2).mean().sqrt().item()

            # FFN pathway
            p_ffn = block.adaln_ffn.action_to_params(action_vec, split_paths=True)
            cam_l2_sum += p_ffn["camera"]["gate_raw"].pow(2).mean().sqrt().item()
            mov_l2_sum += p_ffn["movement"]["gate_raw"].pow(2).mean().sqrt().item()

            count += 2  # attn + ffn

        results['film_camera_gate'].append(cam_l2_sum / count if count > 0 else 0.0)
        results['film_movement_gate'].append(mov_l2_sum / count if count > 0 else 0.0)

        # 3. Compute temporal attention entropy (if buffer non-empty)
        if len(temporal_buffer) > 0:
            # Build x (post-block embedding)
            x = model.embed(z_t).permute(0, 3, 1, 2)
            x = model.stem(x)
            x = x.flatten(2).transpose(1, 2)
            for block in model.blocks:
                x = block(x, action_vec, model.H, model.W)

            attn_stats = model.temporal_attn.attention_stats(x, temporal_buffer, num_query_tokens=64)
            entropy = attn_stats["attn_entropy"].item() if attn_stats is not None else 0.0
            results['temporal_attn_entropy'].append(entropy)
        else:
            results['temporal_attn_entropy'].append(0.0)

        # 4. Compute LPIPS if ground truth provided
        if z_gt_seq is not None and step < len(z_gt_seq):
            frame_pred = vqvae.decode_code(z_t)
            frame_gt = vqvae.decode_code(z_gt_seq[step:step+1])

            # Normalize to [-1, 1] for LPIPS
            frame_pred = frame_pred * 2.0 - 1.0
            frame_gt = frame_gt * 2.0 - 1.0

            lpips_val = lpips_fn(frame_pred, frame_gt).item()
            results['lpips'].append(lpips_val)

        # 5. Step forward
        logits, new_state = model.step(z_t, action_vec, temporal_buffer)
        z_t = logits.argmax(dim=1)

        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > temporal_context_len:
            temporal_buffer.pop(0)

    # Convert to numpy arrays
    return {k: np.array(v) if v is not None else None for k, v in results.items()}


@torch.no_grad()
def diagnostic_ablation_action_effect(
    model,
    vqvae,
    z_start: torch.Tensor,
    test_action: torch.Tensor,
    num_steps: int = 10,
    temporal_context_len: int = 8
) -> Dict[str, float]:
    """
    Compare rollouts with real action vs zero/random actions.

    Answers Q1: Do actions actually affect predictions, or are they ignored?

    Args:
        model: World model
        vqvae: VQ-VAE
        z_start: Initial latent [1, H, W]
        test_action: Action to test [1, action_dim]
        num_steps: Rollout length
        temporal_context_len: Buffer size

    Returns:
        - 'action_vs_zero_lpips_mean': Avg LPIPS diff vs zero action
        - 'action_vs_random_lpips_mean': Avg LPIPS diff vs random action
        - 'action_effect_magnitude': Overall action effect score (0 = ignored, 1 = strong effect)
    """
    device = z_start.device
    model.eval()
    vqvae.eval()
    global lpips_fn
    lpips_fn = lpips_fn.to(device)

    # Generate rollouts
    def rollout(action):
        temporal_buffer = []
        z_t = z_start
        frames = []
        for _ in range(num_steps):
            logits, new_state = model.step(z_t, action, temporal_buffer)
            z_t = logits.argmax(dim=1)
            frame = vqvae.decode_code(z_t)
            frames.append(frame)
            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > temporal_context_len:
                temporal_buffer.pop(0)
        return torch.cat(frames, dim=0)

    # Rollout A: Test action
    frames_action = rollout(test_action)

    # Rollout B: Zero action
    zero_action = torch.zeros_like(test_action)
    frames_zero = rollout(zero_action)

    # Rollout C: Random action
    random_action = torch.randn_like(test_action)
    frames_random = rollout(random_action)

    # Compute LPIPS diffs
    frames_action_norm = frames_action * 2.0 - 1.0
    frames_zero_norm = frames_zero * 2.0 - 1.0
    frames_random_norm = frames_random * 2.0 - 1.0

    lpips_vs_zero = []
    lpips_vs_random = []

    for i in range(num_steps):
        lpips_vs_zero.append(lpips_fn(
            frames_action_norm[i:i+1],
            frames_zero_norm[i:i+1]
        ).item())

        lpips_vs_random.append(lpips_fn(
            frames_action_norm[i:i+1],
            frames_random_norm[i:i+1]
        ).item())

    lpips_vs_zero_mean = float(np.mean(lpips_vs_zero))
    lpips_vs_random_mean = float(np.mean(lpips_vs_random))

    # Action effect magnitude: if close to zero_action, effect is weak
    # Normalize by typical LPIPS range (0-1)
    action_effect = lpips_vs_zero_mean

    return {
        'action_vs_zero_lpips_mean': lpips_vs_zero_mean,
        'action_vs_random_lpips_mean': lpips_vs_random_mean,
        'action_effect_magnitude': action_effect
    }


@torch.no_grad()
def diagnostic_ablation_buffer_quality(
    model,
    vqvae,
    z_start: torch.Tensor,
    action_vec: torch.Tensor,
    num_steps: int = 10,
    corrupt_p: float = 0.3,
    temporal_context_len: int = 8,
    codebook_size: int = 1024
) -> Dict[str, float]:
    """
    Compare rollouts with clean vs corrupted vs empty temporal buffer.

    Answers Q3: Does model rely on buffer quality or action signal?

    Args:
        model: World model
        vqvae: VQ-VAE
        z_start: Initial latent [1, H, W]
        action_vec: Action [1, action_dim]
        num_steps: Rollout length
        corrupt_p: Corruption probability for buffer
        temporal_context_len: Buffer size
        codebook_size: VQ-VAE codebook size for random corruption

    Returns:
        - 'clean_buffer_mean_unique_codes': Quality with clean buffer (measured by unique codes)
        - 'corrupted_buffer_mean_unique_codes': Quality with corrupted buffer
        - 'no_buffer_mean_unique_codes': Quality with empty buffer
        - 'buffer_dependence': How much quality drops without clean buffer (0-1 scale)
    """
    device = z_start.device
    model.eval()

    def rollout_clean():
        temporal_buffer = []
        z_t = z_start
        unique_codes = []
        for _ in range(num_steps):
            logits, new_state = model.step(z_t, action_vec, temporal_buffer)
            z_t = logits.argmax(dim=1)
            unique_codes.append(z_t[0].unique().numel())
            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > temporal_context_len:
                temporal_buffer.pop(0)
        return unique_codes

    def rollout_corrupted():
        temporal_buffer = []
        z_t = z_start
        unique_codes = []
        for _ in range(num_steps):
            # Corrupt buffer before stepping
            corrupted_buffer = []
            for state in temporal_buffer:
                # Assume state is compressed frame representation
                # Apply random corruption to simulate degraded memory
                if torch.rand(1).item() < corrupt_p:
                    noise = torch.randn_like(state) * 0.5
                    corrupted_buffer.append(state + noise)
                else:
                    corrupted_buffer.append(state)

            logits, new_state = model.step(z_t, action_vec, corrupted_buffer)
            z_t = logits.argmax(dim=1)
            unique_codes.append(z_t[0].unique().numel())
            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > temporal_context_len:
                temporal_buffer.pop(0)
        return unique_codes

    def rollout_no_buffer():
        z_t = z_start
        unique_codes = []
        for _ in range(num_steps):
            logits, _ = model.step(z_t, action_vec, [])  # Empty buffer
            z_t = logits.argmax(dim=1)
            unique_codes.append(z_t[0].unique().numel())
        return unique_codes

    clean_codes = rollout_clean()
    corrupted_codes = rollout_corrupted()
    no_buffer_codes = rollout_no_buffer()

    clean_mean = float(np.mean(clean_codes))
    corrupted_mean = float(np.mean(corrupted_codes))
    no_buffer_mean = float(np.mean(no_buffer_codes))

    # Buffer dependence: how much does quality drop without clean buffer?
    # Higher = more dependent on buffer quality (overfitting to clean memory)
    max_dependence = max(clean_mean - corrupted_mean, clean_mean - no_buffer_mean)
    buffer_dependence = max_dependence / (clean_mean + 1e-6)

    return {
        'clean_buffer_mean_unique_codes': clean_mean,
        'corrupted_buffer_mean_unique_codes': corrupted_mean,
        'no_buffer_mean_unique_codes': no_buffer_mean,
        'buffer_dependence': float(buffer_dependence)
    }


@torch.no_grad()
def diagnostic_film_evolution(
    model,
    z_start: torch.Tensor,
    action_vec: torch.Tensor,
    num_steps: int = 20,
    temporal_context_len: int = 8
) -> Dict[str, np.ndarray]:
    """
    Track FiLM gate magnitudes over AR rollout.

    Answers Q2: Does action conditioning weaken during rollout?

    Args:
        model: World model
        z_start: Initial latent [1, H, W]
        action_vec: Action [1, action_dim]
        num_steps: Rollout length
        temporal_context_len: Buffer size

    Returns:
        Arrays of length num_steps:
            - 'camera_gate_l2': Per-step camera pathway gate magnitude
            - 'movement_gate_l2': Per-step movement pathway gate magnitude
            - 'gate_ratio': camera/movement ratio per step
    """
    model.eval()

    temporal_buffer = []
    z_t = z_start

    camera_gates = []
    movement_gates = []

    for _ in range(num_steps):
        # Compute gate magnitudes
        cam_l2_sum = 0.0
        mov_l2_sum = 0.0
        count = 0

        for block in model.blocks:
            p_attn = block.adaln_attn.action_to_params(action_vec, split_paths=True)
            cam_l2_sum += p_attn["camera"]["gate_raw"].pow(2).mean().sqrt().item()
            mov_l2_sum += p_attn["movement"]["gate_raw"].pow(2).mean().sqrt().item()

            p_ffn = block.adaln_ffn.action_to_params(action_vec, split_paths=True)
            cam_l2_sum += p_ffn["camera"]["gate_raw"].pow(2).mean().sqrt().item()
            mov_l2_sum += p_ffn["movement"]["gate_raw"].pow(2).mean().sqrt().item()

            count += 2

        camera_gates.append(cam_l2_sum / count if count > 0 else 0.0)
        movement_gates.append(mov_l2_sum / count if count > 0 else 0.0)

        # Step forward
        logits, new_state = model.step(z_t, action_vec, temporal_buffer)
        z_t = logits.argmax(dim=1)

        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > temporal_context_len:
            temporal_buffer.pop(0)

    camera_gates = np.array(camera_gates)
    movement_gates = np.array(movement_gates)
    gate_ratio = camera_gates / (movement_gates + 1e-9)

    return {
        'camera_gate_l2': camera_gates,
        'movement_gate_l2': movement_gates,
        'gate_ratio': gate_ratio
    }


@torch.no_grad()
def diagnostic_temporal_attention_evolution(
    model,
    z_start: torch.Tensor,
    action_vec: torch.Tensor,
    num_steps: int = 20,
    temporal_context_len: int = 8
) -> Dict[str, np.ndarray]:
    """
    Track temporal attention distribution over AR rollout.

    Answers Q2: Does attention become sticky during rollout?

    Args:
        model: World model
        z_start: Initial latent [1, H, W]
        action_vec: Action [1, action_dim]
        num_steps: Rollout length
        temporal_context_len: Buffer size

    Returns:
        Arrays of length num_steps:
            - 'entropy': Attention entropy per step
            - 'frame_mass_last': Attention to most recent frame
            - 'frame_mass_first': Attention to oldest frame
    """
    model.eval()

    temporal_buffer = []
    z_t = z_start

    entropies = []
    mass_last = []
    mass_first = []

    for _ in range(num_steps):
        if len(temporal_buffer) > 0:
            # Build x
            x = model.embed(z_t).permute(0, 3, 1, 2)
            x = model.stem(x)
            x = x.flatten(2).transpose(1, 2)
            for block in model.blocks:
                x = block(x, action_vec, model.H, model.W)

            attn_stats = model.temporal_attn.attention_stats(x, temporal_buffer, num_query_tokens=64)

            if attn_stats is not None:
                entropies.append(attn_stats["attn_entropy"].item())

                per_frame = attn_stats.get("attn_mean_per_frame")
                if per_frame is not None:
                    mass_last.append(per_frame[-1].item())
                    mass_first.append(per_frame[0].item())
                else:
                    mass_last.append(0.0)
                    mass_first.append(0.0)
            else:
                entropies.append(0.0)
                mass_last.append(0.0)
                mass_first.append(0.0)
        else:
            entropies.append(0.0)
            mass_last.append(0.0)
            mass_first.append(0.0)

        # Step forward
        logits, new_state = model.step(z_t, action_vec, temporal_buffer)
        z_t = logits.argmax(dim=1)

        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > temporal_context_len:
            temporal_buffer.pop(0)

    return {
        'entropy': np.array(entropies),
        'frame_mass_last': np.array(mass_last),
        'frame_mass_first': np.array(mass_first)
    }


@torch.no_grad()
def diagnostic_tf_vs_ar_comparison(
    model,
    vqvae,
    Z_seq: torch.Tensor,
    A_seq: torch.Tensor,
    temporal_context_len: int = 8
) -> Dict[str, np.ndarray]:
    """
    Compare teacher-forced reconstruction vs AR rollout on same sequence.

    Answers Q3: How big is the TF-AR quality gap?

    Args:
        model: World model
        vqvae: VQ-VAE
        Z_seq: Ground truth token sequence [seq_len, H, W]
        A_seq: Action sequence [seq_len, action_dim]
        temporal_context_len: Buffer size

    Returns:
        Arrays of length seq_len:
            - 'tf_unique_codes': Per-frame unique codes (teacher-forced)
            - 'ar_unique_codes': Per-frame unique codes (autoregressive)
            - 'gap': ar - tf (quality gap in unique codes)
    """
    model.eval()
    vqvae.eval()

    seq_len = len(Z_seq)

    # Teacher-forced rollout
    tf_buffer = []
    tf_codes = []
    for t in range(seq_len):
        z_t = Z_seq[t:t+1]
        logits_tf, new_state = model.step(z_t, A_seq[t:t+1], tf_buffer)
        z_pred_tf = logits_tf.argmax(dim=1)
        tf_codes.append(z_pred_tf[0].unique().numel())

        tf_buffer.append(new_state.detach())
        if len(tf_buffer) > temporal_context_len:
            tf_buffer.pop(0)

    # Autoregressive rollout
    ar_buffer = []
    z_t = Z_seq[0:1]
    ar_codes = []
    for t in range(seq_len):
        logits_ar, new_state = model.step(z_t, A_seq[t:t+1], ar_buffer)
        z_t = logits_ar.argmax(dim=1)
        ar_codes.append(z_t[0].unique().numel())

        ar_buffer.append(new_state.detach())
        if len(ar_buffer) > temporal_context_len:
            ar_buffer.pop(0)

    tf_codes = np.array(tf_codes)
    ar_codes = np.array(ar_codes)
    gap = ar_codes - tf_codes

    return {
        'tf_unique_codes': tf_codes,
        'ar_unique_codes': ar_codes,
        'gap': gap
    }
