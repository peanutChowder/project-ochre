#!/usr/bin/env python3
"""
Measure LPIPS sensitivity to camera vs movement perturbations.

Quantifies LPIPS gradient sensitivity to test the loss insensitivity hypothesis.

Usage:
    python measure_lpips_sensitivity.py \
        --vqvae_ckpt vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt \
        --data_dir preprocessedv5/ \
        --n_samples 500 \
        --output_dir diagnostics/lpips_sensitivity
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import lpips
except ImportError:
    print("Error: lpips package not installed. Install with: pip install lpips")
    exit(1)

from vq_vae.vq_vae import VQVAE


@dataclass
class PerturbationResult:
    """Result from a single perturbation test."""
    perturbation_type: str  # "camera_yaw", "camera_pitch", "movement_translate"
    magnitude: float
    lpips_loss: float
    lpips_gradient_norm: float


def load_vqvae(vqvae_ckpt: str, device: torch.device) -> VQVAE:
    """Load VQVAE model from checkpoint."""
    ckpt = torch.load(vqvae_ckpt, map_location=device)

    config = ckpt.get("config", {})
    embedding_dim = config.get("embedding_dim", 384)
    codebook_size = config.get("codebook_size", 1024)
    beta = config.get("beta", 0.25)
    ema_decay = config.get("ema_decay", 0.99)

    vqvae = VQVAE(
        embedding_dim=embedding_dim,
        num_embeddings=codebook_size,
        commitment_cost=beta,
        decay=ema_decay,
    ).to(device)

    vqvae.encoder.load_state_dict(ckpt["encoder"])
    vqvae.decoder.load_state_dict(ckpt["decoder"])
    vqvae.vq_vae.load_state_dict(ckpt["quantizer"])

    vqvae.eval()
    return vqvae


def load_random_frames(data_dir: str,
                        vqvae: VQVAE,
                        n_samples: int,
                        device: torch.device) -> torch.Tensor:
    """
    Load and decode random frames from dataset.

    Returns: (n_samples, 3, H, W) RGB tensor in [0, 1]
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    sequences = manifest["sequences"]

    rgb_frames = []

    with torch.no_grad():
        for _ in tqdm(range(n_samples), desc="Loading frames"):
            # Random sequence
            seq = random.choice(sequences)
            file_path = os.path.join(data_dir, seq["file"])

            if not os.path.exists(file_path):
                continue

            # Load tokens
            with np.load(file_path, mmap_mode='r') as data:
                tokens = np.array(data["tokens"])

            # Random frame
            idx = random.randint(0, len(tokens) - 1)
            token_indices = torch.from_numpy(tokens[idx]).long().to(device)

            # Decode to RGB
            rgb = vqvae.decode_code(token_indices.unsqueeze(0))  # (1, 3, H, W)
            rgb_frames.append(rgb[0])

    return torch.stack(rgb_frames, dim=0)  # (n_samples, 3, H, W)


def apply_camera_perturbation(rgb_image: torch.Tensor,
                               perturb_type: str,
                               magnitude: float) -> torch.Tensor:
    """
    Apply camera-like perturbation (rotation approximation via shift).

    rgb_image: (B, 3, H, W) tensor in [0, 1]
    perturb_type: "yaw" or "pitch"
    magnitude: pixel shift amount

    Returns: perturbed image (B, 3, H, W)
    """
    B, C, H, W = rgb_image.shape

    if perturb_type == "yaw":
        # Horizontal shift (wrap around for circular panorama effect)
        shift_pixels = int(magnitude)
        perturbed = torch.roll(rgb_image, shifts=shift_pixels, dims=3)

    elif perturb_type == "pitch":
        # Vertical shift (no wrap, fill with black/gray for sky/ground)
        shift_pixels = int(magnitude)
        perturbed = torch.zeros_like(rgb_image)
        if shift_pixels > 0:
            perturbed[:, :, shift_pixels:, :] = rgb_image[:, :, :-shift_pixels, :]
            # Fill top with sky-like color (light blue)
            perturbed[:, :, :shift_pixels, :] = torch.tensor([0.5, 0.7, 0.9]).view(1, 3, 1, 1).to(rgb_image.device)
        else:
            shift_pixels = abs(shift_pixels)
            perturbed[:, :, :-shift_pixels, :] = rgb_image[:, :, shift_pixels:, :]
            # Fill bottom with ground-like color (brown/green)
            perturbed[:, :, -shift_pixels:, :] = torch.tensor([0.4, 0.5, 0.3]).view(1, 3, 1, 1).to(rgb_image.device)

    return perturbed


def apply_movement_perturbation(rgb_image: torch.Tensor,
                                 magnitude: float) -> torch.Tensor:
    """
    Apply movement-like perturbation (small translation).

    Movement in Minecraft: translates camera position but keeps textures similar.
    Simulate via small random shift in both directions.

    magnitude: pixel shift (e.g., 2-4 pixels for realistic movement)
    """
    B, C, H, W = rgb_image.shape

    # Random direction for translation
    dx = int(magnitude * np.random.uniform(-1, 1))
    dy = int(magnitude * np.random.uniform(-1, 1))

    perturbed = torch.zeros_like(rgb_image)

    # Apply shift with zero padding
    src_x_start = max(0, -dx)
    src_x_end = W + min(0, -dx)
    dst_x_start = max(0, dx)
    dst_x_end = W + min(0, dx)

    src_y_start = max(0, -dy)
    src_y_end = H + min(0, -dy)
    dst_y_start = max(0, dy)
    dst_y_end = H + min(0, dy)

    perturbed[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        rgb_image[:, :, src_y_start:src_y_end, src_x_start:src_x_end]

    return perturbed


def measure_lpips_sensitivity(rgb_original: torch.Tensor,
                               lpips_model: lpips.LPIPS,
                               perturb_type: str,
                               magnitude: float) -> PerturbationResult:
    """
    Measure LPIPS loss and gradient for a perturbation.

    Returns: PerturbationResult with loss and gradient norm
    """
    rgb_original = rgb_original.clone().requires_grad_(True)

    # Apply perturbation
    if perturb_type in ["yaw", "pitch"]:
        rgb_perturbed = apply_camera_perturbation(rgb_original, perturb_type, magnitude)
    else:  # "translate"
        rgb_perturbed = apply_movement_perturbation(rgb_original, magnitude)

    # Normalize to [-1, 1] for LPIPS
    rgb_orig_norm = rgb_original * 2.0 - 1.0
    rgb_pert_norm = rgb_perturbed * 2.0 - 1.0

    # Compute LPIPS loss
    lpips_loss = lpips_model(rgb_pert_norm, rgb_orig_norm.detach()).mean()

    # Compute gradient
    lpips_loss.backward()
    grad_norm = torch.norm(rgb_original.grad).item() if rgb_original.grad is not None else 0.0

    return PerturbationResult(
        perturbation_type=perturb_type,
        magnitude=magnitude,
        lpips_loss=lpips_loss.item(),
        lpips_gradient_norm=grad_norm
    )


def compute_summary_stats(results: Dict[str, List[PerturbationResult]]) -> Dict:
    """Compute summary statistics."""
    all_camera_lpips = []
    all_camera_grads = []
    all_movement_lpips = []
    all_movement_grads = []

    for key, result_list in results.items():
        for result in result_list:
            if 'camera' in key:
                all_camera_lpips.append(result.lpips_loss)
                all_camera_grads.append(result.lpips_gradient_norm)
            else:
                all_movement_lpips.append(result.lpips_loss)
                all_movement_grads.append(result.lpips_gradient_norm)

    mean_lpips_camera = np.mean(all_camera_lpips) if all_camera_lpips else 0
    mean_lpips_movement = np.mean(all_movement_lpips) if all_movement_lpips else 0
    mean_grad_camera = np.mean(all_camera_grads) if all_camera_grads else 0
    mean_grad_movement = np.mean(all_movement_grads) if all_movement_grads else 0

    lpips_ratio = mean_lpips_camera / mean_lpips_movement if mean_lpips_movement > 0 else 0
    gradient_ratio = mean_grad_camera / mean_grad_movement if mean_grad_movement > 0 else 0

    return {
        "mean_lpips_camera": float(mean_lpips_camera),
        "mean_lpips_movement": float(mean_lpips_movement),
        "lpips_ratio": float(lpips_ratio),
        "mean_grad_camera": float(mean_grad_camera),
        "mean_grad_movement": float(mean_grad_movement),
        "gradient_ratio": float(gradient_ratio)
    }


def save_results(results: Dict[str, List[PerturbationResult]], output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Flatten results for JSON
    per_perturbation = []
    for key, result_list in results.items():
        for result in result_list:
            per_perturbation.append(asdict(result))

    # Compute summary
    summary = compute_summary_stats(results)

    output = {
        "summary": summary,
        "per_perturbation": per_perturbation
    }

    output_path = os.path.join(output_dir, "lpips_sensitivity.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Mean LPIPS (camera): {summary['mean_lpips_camera']:.4f}")
    print(f"  Mean LPIPS (movement): {summary['mean_lpips_movement']:.4f}")
    print(f"  LPIPS ratio (camera/movement): {summary['lpips_ratio']:.2f}×")
    print(f"  Mean gradient norm (camera): {summary['mean_grad_camera']:.4f}")
    print(f"  Mean gradient norm (movement): {summary['mean_grad_movement']:.4f}")
    print(f"  Gradient ratio (camera/movement): {summary['gradient_ratio']:.2f}×")


def plot_results(results: Dict[str, List[PerturbationResult]], output_dir: str):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    camera_yaw_lpips = [r.lpips_loss for r in results["camera_yaw"]]
    camera_pitch_lpips = [r.lpips_loss for r in results["camera_pitch"]]
    movement_lpips = [r.lpips_loss for r in results["movement_translate"]]

    camera_yaw_grads = [r.lpips_gradient_norm for r in results["camera_yaw"]]
    camera_pitch_grads = [r.lpips_gradient_norm for r in results["camera_pitch"]]
    movement_grads = [r.lpips_gradient_norm for r in results["movement_translate"]]

    # 1. LPIPS loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    data_lpips = [camera_yaw_lpips, camera_pitch_lpips, movement_lpips]
    labels_lpips = ['Camera Yaw', 'Camera Pitch', 'Movement']
    colors = ['#FF6B6B', '#FFA07A', '#4ECDC4']

    bp = ax.boxplot(data_lpips, labels=labels_lpips, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('LPIPS Loss', fontsize=12)
    ax.set_title('LPIPS Sensitivity by Perturbation Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lpips_comparison.png'), dpi=150)
    plt.close()

    # 2. Gradient comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    data_grads = [camera_yaw_grads, camera_pitch_grads, movement_grads]

    bp = ax.boxplot(data_grads, labels=labels_lpips, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('LPIPS Gradient Magnitude by Perturbation Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_comparison.png'), dpi=150)
    plt.close()

    # 3. Magnitude curves (showing how sensitivity changes with perturbation size)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Group by magnitude
    magnitudes_yaw = sorted(set(r.magnitude for r in results["camera_yaw"]))
    magnitudes_pitch = sorted(set(r.magnitude for r in results["camera_pitch"]))
    magnitudes_move = sorted(set(r.magnitude for r in results["movement_translate"]))

    def get_mean_by_mag(result_list, magnitudes):
        lpips_means = []
        for mag in magnitudes:
            lpips_values = [r.lpips_loss for r in result_list if r.magnitude == mag]
            lpips_means.append(np.mean(lpips_values) if lpips_values else 0)
        return lpips_means

    yaw_lpips_curve = get_mean_by_mag(results["camera_yaw"], magnitudes_yaw)
    pitch_lpips_curve = get_mean_by_mag(results["camera_pitch"], magnitudes_pitch)
    move_lpips_curve = get_mean_by_mag(results["movement_translate"], magnitudes_move)

    ax1.plot(magnitudes_yaw, yaw_lpips_curve, marker='o', label='Camera Yaw', color='#FF6B6B')
    ax1.plot(magnitudes_pitch, pitch_lpips_curve, marker='s', label='Camera Pitch', color='#FFA07A')
    ax1.plot(magnitudes_move, move_lpips_curve, marker='^', label='Movement', color='#4ECDC4')
    ax1.set_xlabel('Perturbation Magnitude (pixels)', fontsize=12)
    ax1.set_ylabel('Mean LPIPS Loss', fontsize=12)
    ax1.set_title('LPIPS vs Perturbation Magnitude', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gradient curves
    def get_mean_grad_by_mag(result_list, magnitudes):
        grad_means = []
        for mag in magnitudes:
            grad_values = [r.lpips_gradient_norm for r in result_list if r.magnitude == mag]
            grad_means.append(np.mean(grad_values) if grad_values else 0)
        return grad_means

    yaw_grad_curve = get_mean_grad_by_mag(results["camera_yaw"], magnitudes_yaw)
    pitch_grad_curve = get_mean_grad_by_mag(results["camera_pitch"], magnitudes_pitch)
    move_grad_curve = get_mean_grad_by_mag(results["movement_translate"], magnitudes_move)

    ax2.plot(magnitudes_yaw, yaw_grad_curve, marker='o', label='Camera Yaw', color='#FF6B6B')
    ax2.plot(magnitudes_pitch, pitch_grad_curve, marker='s', label='Camera Pitch', color='#FFA07A')
    ax2.plot(magnitudes_move, move_grad_curve, marker='^', label='Movement', color='#4ECDC4')
    ax2.set_xlabel('Perturbation Magnitude (pixels)', fontsize=12)
    ax2.set_ylabel('Mean Gradient Norm', fontsize=12)
    ax2.set_title('Gradient vs Perturbation Magnitude', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_curves.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Measure LPIPS sensitivity to perturbations")
    parser.add_argument("--data_dir", type=str, default="preprocessedv5",
                        help="Path to preprocessed dataset")
    parser.add_argument("--vqvae_ckpt", type=str,
                        default="vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt",
                        help="Path to VQ-VAE checkpoint")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of random frames to test")
    parser.add_argument("--output_dir", type=str, default="diagnostics/lpips_sensitivity",
                        help="Where to save results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load models
    print("Loading VQ-VAE...")
    vqvae = load_vqvae(args.vqvae_ckpt, device)

    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    # Load random frames
    print(f"Loading {args.n_samples} random frames...")
    rgb_frames = load_random_frames(args.data_dir, vqvae, args.n_samples, device)
    print(f"Loaded frames: shape={rgb_frames.shape}")

    # Test perturbations with realistic magnitudes
    perturbation_configs = [
        ("yaw", 8),    # 8 pixels horizontal shift (strong camera rotation)
        ("yaw", 4),    # 4 pixels
        ("pitch", 4),  # 4 pixels vertical shift
        ("pitch", 2),  # 2 pixels
        ("translate", 4),  # 4 pixels translation (movement)
        ("translate", 2),  # 2 pixels
    ]

    results = {
        "camera_yaw": [],
        "camera_pitch": [],
        "movement_translate": []
    }

    print("\nTesting perturbations...")
    for rgb_frame in tqdm(rgb_frames, desc="Processing frames"):
        rgb_frame = rgb_frame.unsqueeze(0)  # (1, 3, H, W)

        for perturb_type, magnitude in perturbation_configs:
            result = measure_lpips_sensitivity(
                rgb_frame, lpips_model, perturb_type, magnitude
            )

            if perturb_type == "yaw":
                results["camera_yaw"].append(result)
            elif perturb_type == "pitch":
                results["camera_pitch"].append(result)
            else:
                results["movement_translate"].append(result)

    # Save results
    save_results(results, args.output_dir)

    # Generate plots
    plot_results(results, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
