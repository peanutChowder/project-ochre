"""
Analyze v7.0.5 @ 95k Checkpoint with Comprehensive Diagnostics

This script loads a trained world model checkpoint and runs comprehensive
diagnostic analysis to objectively answer three critical questions:

1. Why is there no response to WASD+jump?
2. Why does performance degrade on medium rollouts with action inputs?
3. Why can't model bridge gap from single-frame to medium-rollout quality?

Usage:
    python analyze_checkpoint.py \
        --checkpoint ./checkpoints/ochre-v7.0.5-step95k.pt \
        --vqvae ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --output_dir ./diagnostics/v7.0.5-step95k
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path

from model_transformer import MinecraftConvTransformer
from vq_vae.vq_vae import VQVAE
from action_encoding import encode_action_v5_np
from diagnostics.diagnostics_enhanced import (
	diagnostic_per_frame_ar_quality,
	diagnostic_ablation_action_effect,
	diagnostic_ablation_buffer_quality,
	diagnostic_film_evolution,
	diagnostic_temporal_attention_evolution,
	diagnostic_tf_vs_ar_comparison,
)


def _infer_conv_transformer_kwargs(checkpoint, state_dict, LATENT_H=18, LATENT_W=32):
    """
    Infer model hyperparameters from checkpoint (adapted from live_inference.py).
    """
    config = checkpoint.get("config", {})

    if "hidden_dim" in config:
        hidden_dim = config["hidden_dim"]
        num_layers = config.get("num_layers", 4)
        num_heads = config.get("num_heads", 6)
        temporal_context_len = config.get("temporal_context_len", 8)
    else:
        # Infer from state dict shapes
        hidden_dim = state_dict["blocks.0.adaln_attn.norm.weight"].shape[0]
        num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".adaln_attn.norm.weight"))
        head_dim = state_dict.get("blocks.0.adaln_attn.attn.head_dim", 64)
        num_heads = hidden_dim // head_dim
        temporal_context_len = 8  # default

    # Infer codebook size
    codebook_size = state_dict["embed.weight"].shape[0]
    embed_dim = state_dict["embed.weight"].shape[1]

    return {
        "codebook_size": codebook_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "H": LATENT_H,
        "W": LATENT_W,
        "window_size": 4,
        "temporal_context_len": temporal_context_len,
    }


def load_vqvae(vqvae_path, device):
    """
    Load VQ-VAE checkpoint (adapted from train.py).
    """
    ckpt = torch.load(vqvae_path, map_location=device)

    # Get config
    config = ckpt.get("config", {})
    embedding_dim = config.get("embedding_dim", 384)
    codebook_size = config.get("codebook_size", 1024)

    # Create model
    vqvae = VQVAE(
        embedding_dim=embedding_dim,
        num_embeddings=codebook_size
    ).to(device)

    # Load weights based on checkpoint format
    if isinstance(ckpt, dict) and {"encoder", "decoder", "quantizer"}.issubset(ckpt.keys()):
        # New format with separate state dicts
        vqvae.encoder.load_state_dict(ckpt["encoder"])
        vqvae.decoder.load_state_dict(ckpt["decoder"])
        vqvae.vq_vae.load_state_dict(ckpt["quantizer"])
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        vqvae.load_state_dict(ckpt["model_state"], strict=False)
    else:
        vqvae.load_state_dict(ckpt, strict=False)

    vqvae.eval()
    return vqvae


def load_checkpoint(checkpoint_path, vqvae_path, device):
    """
    Load world model and VQ-VAE from checkpoints.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Infer model config
    model_kwargs = _infer_conv_transformer_kwargs(ckpt, state_dict)

    print(f"Model config: {model_kwargs}")

    # Create model
    model = MinecraftConvTransformer(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loading VQ-VAE from {vqvae_path}...")
    vqvae = load_vqvae(vqvae_path, device)

    print("Checkpoint loaded successfully!")
    return model, vqvae, model_kwargs


def get_test_actions(device):
    """
    Create test action vectors.
    """
    return {
        'static': torch.tensor(encode_action_v5_np(), device=device).unsqueeze(0),
        'camera_left': torch.tensor(encode_action_v5_np(yaw_raw=-0.5), device=device).unsqueeze(0),
        'camera_right': torch.tensor(encode_action_v5_np(yaw_raw=0.5), device=device).unsqueeze(0),
        'camera_up': torch.tensor(encode_action_v5_np(pitch_raw=0.5), device=device).unsqueeze(0),
        'camera_down': torch.tensor(encode_action_v5_np(pitch_raw=-0.5), device=device).unsqueeze(0),
        'move_forward': torch.tensor(encode_action_v5_np(w=1.0), device=device).unsqueeze(0),
        'jump': torch.tensor(encode_action_v5_np(jump=1.0), device=device).unsqueeze(0),
    }


def run_all_diagnostics(model, vqvae, z_start, test_actions, device, temporal_context_len=8):
    """
    Run all 6 diagnostic categories.
    """
    results = {}

    print("\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80 + "\n")

    # 1. Per-frame AR quality for each action
    print("[1/5] Running per-frame AR quality analysis...")
    for name, action in test_actions.items():
        print(f"  - {name}...")
        results[f'per_frame_quality/{name}'] = diagnostic_per_frame_ar_quality(
            model, vqvae, z_start, action, num_steps=20, temporal_context_len=temporal_context_len
        )

    # 2. Action effect ablations
    print("\n[2/5] Running action effect ablations...")
    for name, action in test_actions.items():
        if name != 'static':
            print(f"  - {name}...")
            results[f'ablation_action/{name}'] = diagnostic_ablation_action_effect(
                model, vqvae, z_start, action, num_steps=10, temporal_context_len=temporal_context_len
            )

    # 3. Buffer quality ablation
    print("\n[3/5] Running buffer quality ablation...")
    results['ablation_buffer'] = diagnostic_ablation_buffer_quality(
        model, vqvae, z_start, test_actions['camera_right'], num_steps=10, temporal_context_len=temporal_context_len
    )

    # 4. FiLM evolution
    print("\n[4/5] Running FiLM evolution analysis...")
    for name, action in test_actions.items():
        print(f"  - {name}...")
        results[f'film_evolution/{name}'] = diagnostic_film_evolution(
            model, z_start, action, num_steps=20, temporal_context_len=temporal_context_len
        )

    # 5. Temporal attention evolution
    print("\n[5/5] Running temporal attention evolution analysis...")
    for name, action in test_actions.items():
        print(f"  - {name}...")
        results[f'temporal_attn_evolution/{name}'] = diagnostic_temporal_attention_evolution(
            model, z_start, action, num_steps=20, temporal_context_len=temporal_context_len
        )

    print("\nDiagnostics complete!")
    return results


def analyze_results(results):
    """
    Interpret diagnostic results to answer 3 questions.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS REPORT")
    print("="*80 + "\n")

    findings = []

    # Q1: Why no WASD response?
    print("Q1: Why is there no response to WASD+jump?\n")

    wasd_effect = results.get('ablation_action/move_forward', {}).get('action_effect_magnitude', 0.0)
    camera_effect = results.get('ablation_action/camera_left', {}).get('action_effect_magnitude', 0.0)
    jump_effect = results.get('ablation_action/jump', {}).get('action_effect_magnitude', 0.0)

    print(f"  - WASD action effect magnitude: {wasd_effect:.4f}")
    print(f"  - Jump action effect magnitude: {jump_effect:.4f}")
    print(f"  - Camera action effect magnitude: {camera_effect:.4f}")
    print(f"  - Ratio (WASD/Camera): {wasd_effect/(camera_effect+1e-9):.4f}")

    if wasd_effect < 0.01:
        finding = "WASD effect near zero → actions are ignored (shortcut)"
        print(f"  ⚠️  FINDING: {finding}")
        findings.append(("Q1", finding))
    elif wasd_effect < 0.1 * camera_effect:
        finding = f"WASD effect {camera_effect/wasd_effect:.1f}x weaker than camera → pathway imbalance"
        print(f"  ⚠️  FINDING: {finding}")
        findings.append(("Q1", finding))

    # Check FiLM gates
    wasd_gate = results.get('film_evolution/move_forward', {}).get('movement_gate_l2', [0])[0]
    camera_gate = results.get('film_evolution/camera_left', {}).get('camera_gate_l2', [0])[0]
    print(f"\n  - WASD movement gate L2 (step 0): {wasd_gate:.4f}")
    print(f"  - Camera gate L2 (step 0): {camera_gate:.4f}")
    print(f"  - Ratio (Camera/Movement): {camera_gate/(wasd_gate+1e-9):.1f}x")

    if wasd_gate < 0.01:
        finding = "Movement gates near zero → pathway dead on arrival"
        print(f"  ⚠️  FINDING: {finding}")
        findings.append(("Q1", finding))
    elif camera_gate > 10 * wasd_gate:
        finding = f"Camera gates {camera_gate/wasd_gate:.1f}x stronger → severe pathway imbalance"
        print(f"  ⚠️  FINDING: {finding}")
        findings.append(("Q1", finding))

    print("\n" + "-"*80 + "\n")

    # Q2: Why degradation on medium rollouts?
    print("Q2: Why does performance degrade on medium rollouts?\n")

    quality_data = results.get('per_frame_quality/camera_right', {})
    unique_codes = quality_data.get('unique_codes', [])

    if len(unique_codes) >= 20:
        quality_frame_0 = unique_codes[0]
        quality_frame_10 = unique_codes[10]
        quality_frame_19 = unique_codes[19]

        print(f"  - Unique codes at frame 0: {quality_frame_0}")
        print(f"  - Unique codes at frame 10: {quality_frame_10}")
        print(f"  - Unique codes at frame 19: {quality_frame_19}")

        if quality_frame_10 < 0.8 * quality_frame_0:
            finding = f"Quality drops {100*(1-quality_frame_10/quality_frame_0):.1f}% by frame 10 → early degradation"
            print(f"  ⚠️  FINDING: {finding}")
            findings.append(("Q2", finding))

        if quality_frame_19 < 0.6 * quality_frame_0:
            finding = f"Quality drops {100*(1-quality_frame_19/quality_frame_0):.1f}% by frame 19 → severe long-term degradation"
            print(f"  ⚠️  FINDING: {finding}")
            findings.append(("Q2", finding))

    # Check attention entropy
    attn_data = results.get('temporal_attn_evolution/camera_right', {})
    entropy = attn_data.get('entropy', [])

    if len(entropy) >= 11 and entropy[0] > 0:
        entropy_0 = entropy[0]
        entropy_10 = entropy[10]

        print(f"\n  - Attention entropy at frame 0: {entropy_0:.3f}")
        print(f"  - Attention entropy at frame 10: {entropy_10:.3f}")

        if entropy_10 < 0.5 * entropy_0:
            finding = f"Attention entropy collapses {100*(1-entropy_10/entropy_0):.1f}% → model becomes short-sighted"
            print(f"  ⚠️  FINDING: {finding}")
            findings.append(("Q2", finding))

    # Check FiLM weakening
    film_data = results.get('film_evolution/camera_right', {})
    camera_gates = film_data.get('camera_gate_l2', [])

    if len(camera_gates) >= 11:
        film_0 = camera_gates[0]
        film_10 = camera_gates[10]

        print(f"\n  - FiLM gate strength at frame 0: {film_0:.4f}")
        print(f"  - FiLM gate strength at frame 10: {film_10:.4f}")

        if film_10 < 0.7 * film_0:
            finding = f"FiLM gates weaken {100*(1-film_10/film_0):.1f}% over rollout → action signal fades"
            print(f"  ⚠️  FINDING: {finding}")
            findings.append(("Q2", finding))

    print("\n" + "-"*80 + "\n")

    # Q3: Why TF-AR gap?
    print("Q3: Why can't model bridge gap from single-frame to medium-rollout quality?\n")

    buffer_data = results.get('ablation_buffer', {})
    buffer_clean = buffer_data.get('clean_buffer_mean_unique_codes', 0.0)
    buffer_corrupted = buffer_data.get('corrupted_buffer_mean_unique_codes', 0.0)
    buffer_dependence = buffer_data.get('buffer_dependence', 0.0)

    print(f"  - Quality with clean buffer: {buffer_clean:.2f} unique codes")
    print(f"  - Quality with corrupted buffer: {buffer_corrupted:.2f} unique codes")
    print(f"  - Buffer dependence: {buffer_dependence:.2%}")

    if buffer_dependence > 0.3:
        finding = f"Buffer dependence {buffer_dependence:.1%} → model overfits to buffer quality (cannot recover from errors)"
        print(f"  ⚠️  FINDING: {finding}")
        findings.append(("Q3", finding))

    print("\n" + "="*80 + "\n")

    return findings


def save_results(results, findings, output_dir):
    """
    Save results to JSON and generate summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value

    # Save metrics
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Save summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write("DIAGNOSTIC ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for question, finding in findings:
            f.write(f"[{question}] {finding}\n")

    print(f"\nResults saved to {output_dir}/")
    print(f"  - metrics.json: Full numerical results")
    print(f"  - summary.txt: Key findings summary")


def main():
    parser = argparse.ArgumentParser(description="Analyze world model checkpoint with comprehensive diagnostics")
    parser.add_argument('--checkpoint', required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument('--vqvae', required=True, help="Path to VQ-VAE checkpoint (.pt file)")
    parser.add_argument('--output_dir', default='./diagnostics/v7.0.5-step95k', help="Output directory for results")
    parser.add_argument('--device', default='mps', choices=['cpu', 'cuda', 'mps'], help="Device to run on")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Load checkpoint
    model, vqvae, model_config = load_checkpoint(args.checkpoint, args.vqvae, device)

    # Create test actions
    print("\nCreating test actions...")
    test_actions = get_test_actions(device)

    # Get initial frame (random latent codes as seed)
    print("Generating initial frame...")
    codebook_size = model_config['codebook_size']
    latent_h = model_config['H']
    latent_w = model_config['W']
    z_start = torch.randint(0, codebook_size, (1, latent_h, latent_w), device=device)

    # Run diagnostics
    results = run_all_diagnostics(
        model, vqvae, z_start, test_actions, device,
        temporal_context_len=model_config['temporal_context_len']
    )

    # Analyze results
    findings = analyze_results(results)

    # Save results
    save_results(results, findings, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
