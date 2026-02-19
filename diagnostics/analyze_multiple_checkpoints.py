"""
Run diagnostic analysis across multiple checkpoints to identify when first-step collapse started.

Usage:
    python analyze_multiple_checkpoints.py --vqvae ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt --device mps
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

# Import the existing analyze_checkpoint functions
from analyze_checkpoint import load_checkpoint, get_test_actions, run_all_diagnostics


def analyze_checkpoint_quick(checkpoint_path, vqvae_path, device):
    """
    Run quick diagnostic on one checkpoint focused on first-step collapse.

    Returns summary dict with key metrics.
    """
    try:
        print(f"\n{'='*80}")
        print(f"Analyzing: {Path(checkpoint_path).name}")
        print(f"{'='*80}")

        # Load checkpoint
        model, vqvae, model_config = load_checkpoint(checkpoint_path, vqvae_path, device)

        # Create test actions
        test_actions = get_test_actions(device)

        # Generate initial frame
        codebook_size = model_config['codebook_size']
        latent_h = model_config['H']
        latent_w = model_config['W']
        z_start = torch.randint(0, codebook_size, (1, latent_h, latent_w), device=device)

        # Run diagnostics
        results = run_all_diagnostics(
            model, vqvae, z_start, test_actions, device,
            temporal_context_len=model_config['temporal_context_len']
        )

        # Extract first-step collapse metrics
        camera_right_quality = results['per_frame_quality/camera_right']

        frame_0_codes = camera_right_quality['unique_codes'][0]
        frame_1_codes = camera_right_quality['unique_codes'][1]
        frame_10_codes = camera_right_quality['unique_codes'][10]
        frame_19_codes = camera_right_quality['unique_codes'][19]

        first_step_drop_pct = ((frame_0_codes - frame_1_codes) / frame_0_codes) * 100 if frame_0_codes > 0 else 0.0
        ten_step_drop_pct = ((frame_0_codes - frame_10_codes) / frame_0_codes) * 100 if frame_0_codes > 0 else 0.0

        # Extract action effect magnitudes
        wasd_effect = results['ablation_action/move_forward']['action_effect_magnitude']
        camera_effect = results['ablation_action/camera_left']['action_effect_magnitude']

        # Extract FiLM gate stats
        film_evolution = results['film_evolution/camera_right']
        camera_gate_start = film_evolution['camera_gate_l2'][0]
        movement_gate_start = film_evolution['movement_gate_l2'][0]

        # Extract temporal attention stats
        temporal_attn_evolution = results['temporal_attn_evolution/camera_right']
        entropy_start = temporal_attn_evolution['entropy'][0]
        entropy_end = temporal_attn_evolution['entropy'][19]

        summary = {
            'checkpoint': Path(checkpoint_path).name,
            'config': {
                'codebook_size': codebook_size,
                'embed_dim': model_config['embed_dim'],
                'hidden_dim': model_config['hidden_dim'],
                'num_layers': model_config['num_layers'],
                'temporal_context_len': model_config['temporal_context_len'],
            },
            'first_step_collapse': {
                'frame_0_codes': int(frame_0_codes),
                'frame_1_codes': int(frame_1_codes),
                'frame_10_codes': int(frame_10_codes),
                'frame_19_codes': int(frame_19_codes),
                'first_step_drop_pct': float(first_step_drop_pct),
                'ten_step_drop_pct': float(ten_step_drop_pct),
            },
            'action_conditioning': {
                'wasd_effect_magnitude': float(wasd_effect),
                'camera_effect_magnitude': float(camera_effect),
                'wasd_to_camera_ratio': float(wasd_effect / (camera_effect + 1e-9)),
            },
            'film_gates': {
                'camera_gate_l2_start': float(camera_gate_start),
                'movement_gate_l2_start': float(movement_gate_start),
                'camera_to_movement_ratio': float(camera_gate_start / (movement_gate_start + 1e-9)),
            },
            'temporal_attention': {
                'entropy_start': float(entropy_start),
                'entropy_end': float(entropy_end),
                'entropy_change_pct': float(((entropy_end - entropy_start) / entropy_start) * 100) if entropy_start > 0 else 0.0,
            },
        }

        return summary, True

    except Exception as e:
        print(f"❌ FAILED to analyze {Path(checkpoint_path).name}: {e}")
        import traceback
        traceback.print_exc()
        return {'checkpoint': Path(checkpoint_path).name, 'error': str(e)}, False


def print_comparison_table(summaries):
    """Print comparison table of key metrics across checkpoints."""
    print("\n" + "="*120)
    print("FIRST-STEP COLLAPSE COMPARISON ACROSS CHECKPOINTS")
    print("="*120 + "\n")

    # Header
    print(f"{'Checkpoint':<30} | {'Frame 0':<8} | {'Frame 1':<8} | {'1-Step Drop':<12} | {'10-Step Drop':<12} | {'WASD/Cam':<10}")
    print("-" * 120)

    for summary in summaries:
        if 'error' in summary:
            print(f"{summary['checkpoint']:<30} | ERROR: {summary['error']}")
            continue

        collapse = summary['first_step_collapse']
        action = summary['action_conditioning']

        checkpoint_name = summary['checkpoint']
        frame_0 = collapse['frame_0_codes']
        frame_1 = collapse['frame_1_codes']
        drop_1 = collapse['first_step_drop_pct']
        drop_10 = collapse['ten_step_drop_pct']
        wasd_cam_ratio = action['wasd_to_camera_ratio']

        print(f"{checkpoint_name:<30} | {frame_0:<8} | {frame_1:<8} | {drop_1:>10.1f}% | {drop_10:>10.1f}% | {wasd_cam_ratio:>9.2f}")

    print("\n" + "="*120)
    print("INTERPRETATION:")
    print("  - Frame 0: Unique codes in seed frame (should be ~400-500)")
    print("  - Frame 1: Unique codes after FIRST AR prediction")
    print("  - 1-Step Drop: % collapse in one autoregressive step")
    print("  - 10-Step Drop: % collapse by frame 10")
    print("  - WASD/Cam: Action effect ratio (>0.5 = balanced, <0.1 = WASD dead)")
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae', required=True, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--device', default='mps', help='Device to use')
    parser.add_argument('--checkpoints', nargs='+', help='Specific checkpoints to analyze (default: all v4.11, v5, v6, v7)')
    parser.add_argument('--output', default='./diagnostics/cross-checkpoint/checkpoint_comparison.json', help='Output JSON path')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Determine which checkpoints to analyze
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    else:
        # Default: analyze key checkpoints from each version series
        checkpoint_paths = [
            # v7.x series (Transformer)
            './checkpoints/ochre-v7.0.1-step55k.pt',
            './checkpoints/ochre-v7.0.2-step85k.pt',
            './checkpoints/ochre-v7.0.4-step110k.pt',
            './checkpoints/ochre-v7.0.5-step95k.pt',
            # v6.x series
            './checkpoints/ochre-v6.2-step100000.pt',
            './checkpoints/ochre-v6.1-step20000.pt',
            # v5.x series
            './checkpoints/ochre-v5.0-step30000.pt',
            # v4.x series (mature checkpoints)
            './checkpoints/ochre-v4.11.0-step70000.pt',
            './checkpoints/ochre-v4.10.1-step80000.pt',
        ]

    # Filter to existing checkpoints
    checkpoint_paths = [p for p in checkpoint_paths if os.path.exists(p)]

    print(f"\nAnalyzing {len(checkpoint_paths)} checkpoints...")
    print(f"VQ-VAE: {args.vqvae}")
    print(f"Device: {args.device}\n")

    # Run analysis on each checkpoint
    summaries = []
    successful = 0
    failed = 0

    for ckpt_path in checkpoint_paths:
        summary, success = analyze_checkpoint_quick(ckpt_path, args.vqvae, device)
        summaries.append(summary)
        if success:
            successful += 1
        else:
            failed += 1

    # Print comparison table
    print_comparison_table(summaries)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'summaries': summaries,
            'metadata': {
                'vqvae_path': args.vqvae,
                'device': args.device,
                'num_checkpoints': len(checkpoint_paths),
                'successful': successful,
                'failed': failed,
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 Successfully analyzed: {successful}/{len(checkpoint_paths)} checkpoints")

    if failed > 0:
        print(f"⚠️  Failed to analyze: {failed}/{len(checkpoint_paths)} checkpoints")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
