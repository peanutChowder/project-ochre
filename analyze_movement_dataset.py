#!/usr/bin/env python3
"""
Dataset Action Analysis - Phase 1 (v4.10)
Analyzes action magnitude distribution in preprocessed Minecraft dataset.

Goal: Determine if dataset has sufficient high-action sequences to support
action conditioning training (Hypothesis 7 validation).

Actions format: [yaw, pitch, move_x, move_z, jump]
- yaw, pitch: camera movement (normalized to [-1, 1])
- move_x: strafe movement (-1=left/A, 0=none, +1=right/D)
- move_z: forward/back movement (+1=forward/W, 0=none, -1=backward/S)
- jump: binary flag (1=jump, 0=no jump)

Output:
- Action magnitude statistics (L2 norm across all dimensions)
- Per-dimension breakdown
- Threshold analysis (% frames with |action| > 0.05, 0.1, 0.2, 0.3)
- High-action sequence identification (top 20%)
- JSON cache for dataset rebalancing (Phase 4)
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def analyze_dataset(dataset_dir: str):
    """
    Phase 1 (v4.10): Analyze action magnitude distribution.

    Computes L2 norm of action vectors to determine if dataset has sufficient
    high-action sequences for action conditioning training.

    Args:
        dataset_dir: Path to preprocessedv4 directory

    Returns:
        dict: Action magnitude statistics and high-action sequence cache
    """
    dataset_path = Path(dataset_dir)
    npz_files = sorted(dataset_path.glob("*.npz"))

    print(f"Analyzing {len(npz_files)} files from {dataset_dir}...")
    print("Computing action magnitudes (L2 norm across all dimensions)...\n")

    # Accumulators
    all_magnitudes = []
    sequence_stats = []  # Cache for Phase 4 rebalancing

    # Per-dimension statistics
    yaw_values = []
    pitch_values = []
    move_x_values = []
    move_z_values = []
    jump_values = []

    total_frames = 0

    for npz_file in tqdm(npz_files, desc="Processing files"):
        try:
            data = np.load(npz_file)
            actions = data['actions']  # Shape: [T, 5] where 5 = [yaw, pitch, move_x, move_z, jump]

            n_frames = actions.shape[0]
            total_frames += n_frames

            # Compute per-frame action magnitude (L2 norm)
            per_frame_mag = np.linalg.norm(actions, axis=-1)  # Shape: [T]
            all_magnitudes.extend(per_frame_mag.tolist())

            # Store sequence-level statistics for Phase 4
            sequence_stats.append({
                "filename": npz_file.name,
                "num_frames": int(n_frames),
                "mean_magnitude": float(np.mean(per_frame_mag)),
                "max_magnitude": float(np.max(per_frame_mag)),
                "p90_magnitude": float(np.percentile(per_frame_mag, 90)),
            })

            # Per-dimension values (sample 10% to avoid memory issues)
            sample_size = max(1, n_frames // 10)
            sample_idx = np.random.choice(n_frames, size=min(sample_size, n_frames), replace=False)
            yaw_values.extend(actions[sample_idx, 0].tolist())
            pitch_values.extend(actions[sample_idx, 1].tolist())
            move_x_values.extend(actions[sample_idx, 2].tolist())
            move_z_values.extend(actions[sample_idx, 3].tolist())
            jump_values.extend(actions[sample_idx, 4].tolist())

        except Exception as e:
            print(f"Error processing {npz_file.name}: {e}")
            continue

    # Convert to numpy for statistics
    all_magnitudes = np.array(all_magnitudes)
    yaw_arr = np.array(yaw_values)
    pitch_arr = np.array(pitch_values)
    move_x_arr = np.array(move_x_values)
    move_z_arr = np.array(move_z_values)
    jump_arr = np.array(jump_values)

    # Compute statistics
    mag_mean = np.mean(all_magnitudes)
    mag_median = np.median(all_magnitudes)
    mag_std = np.std(all_magnitudes)
    mag_min = np.min(all_magnitudes)
    mag_max = np.max(all_magnitudes)

    percentiles = {
        "p25": np.percentile(all_magnitudes, 25),
        "p50": np.percentile(all_magnitudes, 50),
        "p75": np.percentile(all_magnitudes, 75),
        "p90": np.percentile(all_magnitudes, 90),
        "p95": np.percentile(all_magnitudes, 95),
        "p99": np.percentile(all_magnitudes, 99),
    }

    # Threshold analysis
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    threshold_percentages = {}
    for thresh in thresholds:
        count = np.sum(all_magnitudes > thresh)
        pct = (count / len(all_magnitudes)) * 100
        threshold_percentages[f">{thresh}"] = {"count": int(count), "percentage": pct}

    # Identify high-action sequences (top 20%)
    sequence_stats_sorted = sorted(sequence_stats, key=lambda x: x["mean_magnitude"], reverse=True)
    top_20_count = max(1, len(sequence_stats_sorted) // 5)
    high_action_sequences = sequence_stats_sorted[:top_20_count]

    # Print results
    print("\n" + "="*80)
    print("ACTION MAGNITUDE ANALYSIS RESULTS (Phase 1 - v4.10)")
    print("="*80)
    print(f"\nTotal frames analyzed: {total_frames:,}")
    print(f"Total sequences analyzed: {len(npz_files)}")

    print("\n" + "-"*80)
    print("ACTION MAGNITUDE STATISTICS (L2 norm)")
    print("-"*80)
    print(f"Mean:   {mag_mean:.4f}")
    print(f"Median: {mag_median:.4f}")
    print(f"Std:    {mag_std:.4f}")
    print(f"Min:    {mag_min:.4f}")
    print(f"Max:    {mag_max:.4f}")

    print("\n" + "-"*80)
    print("PERCENTILES")
    print("-"*80)
    for name, value in percentiles.items():
        print(f"{name}: {value:.4f}")

    print("\n" + "-"*80)
    print("THRESHOLD ANALYSIS")
    print("-"*80)
    for thresh_name, thresh_data in threshold_percentages.items():
        count = thresh_data["count"]
        pct = thresh_data["percentage"]
        print(f"|action| {thresh_name}: {count:,} frames ({pct:.2f}%)")

    print("\n" + "-"*80)
    print("PER-DIMENSION BREAKDOWN (sampled)")
    print("-"*80)
    print(f"\nYaw (camera horizontal):")
    print(f"  Mean: {np.mean(yaw_arr):.4f}, Std: {np.std(yaw_arr):.4f}")
    print(f"  Min: {np.min(yaw_arr):.4f}, Max: {np.max(yaw_arr):.4f}")

    print(f"\nPitch (camera vertical):")
    print(f"  Mean: {np.mean(pitch_arr):.4f}, Std: {np.std(pitch_arr):.4f}")
    print(f"  Min: {np.min(pitch_arr):.4f}, Max: {np.max(pitch_arr):.4f}")

    print(f"\nMove X (strafe, discrete):")
    print(f"  Mean: {np.mean(move_x_arr):.4f}, Std: {np.std(move_x_arr):.4f}")
    print(f"  Left (-1): {np.sum(move_x_arr < 0):,}, Right (+1): {np.sum(move_x_arr > 0):,}, None (0): {np.sum(move_x_arr == 0):,}")

    print(f"\nMove Z (forward/back, discrete):")
    print(f"  Mean: {np.mean(move_z_arr):.4f}, Std: {np.std(move_z_arr):.4f}")
    print(f"  Forward (+1): {np.sum(move_z_arr > 0):,}, Backward (-1): {np.sum(move_z_arr < 0):,}, None (0): {np.sum(move_z_arr == 0):,}")

    print(f"\nJump (binary):")
    print(f"  Mean: {np.mean(jump_arr):.4f}, Std: {np.std(jump_arr):.4f}")
    print(f"  Jump (>0.5): {np.sum(jump_arr > 0.5):,}, No jump: {np.sum(jump_arr <= 0.5):,}")

    print("\n" + "-"*80)
    print("HIGH-ACTION SEQUENCES (Top 20%)")
    print("-"*80)
    print(f"Total high-action sequences: {len(high_action_sequences)}")
    print(f"\nTop 10 sequences by mean action magnitude:")
    for i, seq in enumerate(high_action_sequences[:10], 1):
        print(f"  {i}. {seq['filename']}: mean={seq['mean_magnitude']:.4f}, max={seq['max_magnitude']:.4f}, p90={seq['p90_magnitude']:.4f}")

    # Decision criteria
    print("\n" + "-"*80)
    print("PHASE 1 DECISION CRITERIA")
    print("-"*80)
    pct_above_01 = threshold_percentages[">0.1"]["percentage"]
    if pct_above_01 > 40:
        recommendation = "GOOD dataset: Proceed to Phase 2 (multi-step ranking)"
    elif pct_above_01 > 20:
        recommendation = "MEDIUM dataset: Implement Phase 2 + Phase 4 (rebalancing)"
    else:
        recommendation = "LOW-ACTION dataset: Dataset quality is the issue, need aggressive rebalancing or data collection"

    print(f"\n% frames with |action| > 0.1: {pct_above_01:.2f}%")
    print(f"Recommendation: {recommendation}")

    # Save results to JSON
    results = {
        "version": "v4.10-phase1",
        "total_frames": int(total_frames),
        "total_sequences": len(npz_files),
        "magnitude_statistics": {
            "mean": float(mag_mean),
            "median": float(mag_median),
            "std": float(mag_std),
            "min": float(mag_min),
            "max": float(mag_max),
        },
        "percentiles": {k: float(v) for k, v in percentiles.items()},
        "threshold_analysis": threshold_percentages,
        "per_dimension_stats": {
            "yaw": {"mean": float(np.mean(yaw_arr)), "std": float(np.std(yaw_arr)), "min": float(np.min(yaw_arr)), "max": float(np.max(yaw_arr))},
            "pitch": {"mean": float(np.mean(pitch_arr)), "std": float(np.std(pitch_arr)), "min": float(np.min(pitch_arr)), "max": float(np.max(pitch_arr))},
            "move_x": {"mean": float(np.mean(move_x_arr)), "std": float(np.std(move_x_arr)), "min": float(np.min(move_x_arr)), "max": float(np.max(move_x_arr))},
            "move_z": {"mean": float(np.mean(move_z_arr)), "std": float(np.std(move_z_arr)), "min": float(np.min(move_z_arr)), "max": float(np.max(move_z_arr))},
            "jump": {"mean": float(np.mean(jump_arr)), "std": float(np.std(jump_arr)), "min": float(np.min(jump_arr)), "max": float(np.max(jump_arr))},
        },
        "high_action_sequences": high_action_sequences,
        "recommendation": recommendation,
    }

    # Save action magnitude cache (for Phase 4)
    output_path = Path(dataset_dir) / "action_analysis_phase1.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(all_magnitudes, sequence_stats, dataset_dir)

    print("="*80)
    return results

def generate_visualizations(all_magnitudes, sequence_stats, dataset_dir):
    """
    Generate histogram and distribution plots for action magnitudes.

    Args:
        all_magnitudes: numpy array of all frame action magnitudes
        sequence_stats: list of per-sequence statistics
        dataset_dir: directory to save plots
    """
    output_dir = Path(dataset_dir)

    # 1. Histogram of action magnitudes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Main histogram
    ax = axes[0, 0]
    ax.hist(all_magnitudes, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Action Magnitude (L2 norm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Action Magnitudes (All Frames)')
    ax.axvline(np.mean(all_magnitudes), color='red', linestyle='--', label=f'Mean: {np.mean(all_magnitudes):.4f}')
    ax.axvline(np.median(all_magnitudes), color='green', linestyle='--', label=f'Median: {np.median(all_magnitudes):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoomed histogram (magnitudes < 0.5)
    ax = axes[0, 1]
    magnitudes_filtered = all_magnitudes[all_magnitudes < 0.5]
    ax.hist(magnitudes_filtered, bins=100, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Action Magnitude (L2 norm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Action Magnitudes (|action| < 0.5, zoomed)')
    ax.axvline(np.mean(magnitudes_filtered), color='red', linestyle='--', label=f'Mean: {np.mean(magnitudes_filtered):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CDF (Cumulative Distribution Function)
    ax = axes[1, 0]
    sorted_mags = np.sort(all_magnitudes)
    cdf = np.arange(1, len(sorted_mags) + 1) / len(sorted_mags)
    ax.plot(sorted_mags, cdf, linewidth=2)
    ax.set_xlabel('Action Magnitude (L2 norm)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function (CDF)')
    ax.grid(True, alpha=0.3)
    # Mark key percentiles
    for pct, label in [(0.5, 'p50'), (0.75, 'p75'), (0.9, 'p90'), (0.95, 'p95')]:
        val = np.percentile(all_magnitudes, pct * 100)
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.text(val, pct, f'{label}={val:.3f}', fontsize=8, rotation=90, va='bottom')

    # Sequence-level mean magnitudes
    ax = axes[1, 1]
    sequence_means = [s['mean_magnitude'] for s in sequence_stats]
    ax.hist(sequence_means, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Mean Action Magnitude per Sequence')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sequence-Level Mean Magnitudes')
    ax.axvline(np.mean(sequence_means), color='red', linestyle='--', label=f'Mean: {np.mean(sequence_means):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "action_magnitude_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plots to: {plot_path}")

    # 2. Per-dimension breakdown
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Note: We need to recompute per-dimension histograms from the dataset
    # Since we only sampled 10%, we'll note this in the titles

    ax = axes[0, 0]
    ax.set_title('Placeholder: Per-Dimension Histograms')
    ax.text(0.5, 0.5, 'Run with full data\nfor per-dimension plots',
            ha='center', va='center', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plot_path_dims = output_dir / "action_per_dimension.png"
    plt.savefig(plot_path_dims, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-dimension plots to: {plot_path_dims}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 1 (v4.10): Analyze action magnitude distribution in preprocessed dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="preprocessedv4",
        help="Path to preprocessed dataset directory (default: preprocessedv4)"
    )
    args = parser.parse_args()

    analyze_dataset(args.dataset_dir)
