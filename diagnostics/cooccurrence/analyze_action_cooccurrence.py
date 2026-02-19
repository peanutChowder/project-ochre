#!/usr/bin/env python3
"""
Analyze action co-occurrence in preprocessedv5/ dataset.

Measures camera vs movement action co-occurrence to test the signal masking hypothesis.

Usage:
    python analyze_action_cooccurrence.py --data_dir preprocessedv5/ --output_dir diagnostics/cooccurrence
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class FrameActionType:
    """Classification of a single frame's action."""
    camera_only: bool
    movement_only: bool
    combined: bool
    static: bool


@dataclass
class SequenceStats:
    """Per-sequence action statistics."""
    file: str
    length: int
    camera_only: int
    movement_only: int
    combined: int
    static: int


@dataclass
class CooccurrenceStats:
    """Aggregated co-occurrence statistics."""
    total_frames: int
    camera_only_count: int
    movement_only_count: int
    combined_count: int
    static_count: int
    per_sequence_stats: List[Dict]
    action_correlation: List[List[float]]
    temporal_autocorr: Dict[str, List[float]]


def classify_action_frame(action_vec: np.ndarray) -> FrameActionType:
    """
    Classify a single 15D action vector.

    action_vec: [yaw(5), pitch(3), WASD(4), jump(1), sprint(1), sneak(1)]

    Returns: FrameActionType classification
    """
    # Yaw center bin: dim 2
    yaw_center = (action_vec[2] > 0.5)

    # Pitch center bin: dim 6 (pitch starts at dim 5, level is index 1)
    pitch_center = (action_vec[6] > 0.5)

    camera_neutral = yaw_center and pitch_center

    # Movement includes WASD + jump + sprint + sneak (dims 8-14)
    movement_active = np.any(action_vec[8:15] > 0.5)

    camera_active = not camera_neutral

    return FrameActionType(
        camera_only=(camera_active and not movement_active),
        movement_only=(camera_neutral and movement_active),
        combined=(camera_active and movement_active),
        static=(camera_neutral and not movement_active)
    )


def compute_temporal_autocorrelation(classifications: List[FrameActionType],
                                      max_lag: int) -> Dict[str, np.ndarray]:
    """
    Compute temporal autocorrelation for each action type.

    Returns: {action_type: [autocorr_lag0, autocorr_lag1, ...]}
    """
    # Convert to binary time series
    camera_only_series = np.array([c.camera_only for c in classifications], dtype=float)
    movement_only_series = np.array([c.movement_only for c in classifications], dtype=float)
    combined_series = np.array([c.combined for c in classifications], dtype=float)
    static_series = np.array([c.static for c in classifications], dtype=float)

    def autocorr(series, lag):
        n = len(series)
        if n <= lag:
            return np.nan
        if lag == 0:
            return 1.0  # Perfect correlation at lag 0
        mean = series.mean()
        c0 = np.sum((series - mean) ** 2)
        if c0 == 0:
            return 0.0
        ct = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
        return ct / c0

    results = {}
    for name, series in [
        ("camera_only", camera_only_series),
        ("movement_only", movement_only_series),
        ("combined", combined_series),
        ("static", static_series)
    ]:
        autocorr_values = [autocorr(series, lag) for lag in range(max_lag)]
        results[name] = np.array(autocorr_values)

    return results


def compute_action_correlation(all_actions: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between all 15 action dimensions.

    all_actions: (N, 15) array of all action vectors
    Returns: (15, 15) correlation matrix
    """
    # Handle columns with zero variance (constant values)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr_matrix = np.corrcoef(all_actions, rowvar=False)
        # Replace NaN with 0 (indicates no correlation due to constant column)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    return corr_matrix


def process_dataset(data_dir: str,
                     manifest_path: str,
                     sample_rate: float,
                     temporal_window: int) -> CooccurrenceStats:
    """
    Memory-efficient streaming processing of all .npz files.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    sequences = manifest["sequences"]

    # Sample files if requested
    if sample_rate < 1.0:
        n_sample = max(1, int(len(sequences) * sample_rate))
        sequences = random.sample(sequences, n_sample)

    print(f"Processing {len(sequences)} sequences...")

    total_frames = 0
    counts = {"camera_only": 0, "movement_only": 0, "combined": 0, "static": 0}
    per_sequence_stats = []

    # Collect actions for correlation (subsample if dataset is huge)
    all_actions = []
    MAX_ACTIONS_FOR_CORR = 1_000_000  # Limit memory usage

    # Collect classifications for temporal autocorrelation
    all_classifications = []
    MAX_FRAMES_FOR_TEMPORAL = 100_000

    for seq_meta in tqdm(sequences, desc="Processing sequences"):
        file_path = os.path.join(data_dir, seq_meta["file"])

        if not os.path.exists(file_path):
            continue

        # Use mmap_mode='r' for memory efficiency
        with np.load(file_path, mmap_mode='r') as data:
            actions = np.array(data["actions"])  # (T, 15)

        # Classify each frame
        classifications = [classify_action_frame(a) for a in actions]

        # Update counts
        seq_stats = {
            "file": seq_meta["file"],
            "length": len(actions),
            "camera_only": sum(c.camera_only for c in classifications),
            "movement_only": sum(c.movement_only for c in classifications),
            "combined": sum(c.combined for c in classifications),
            "static": sum(c.static for c in classifications),
        }

        for key in ["camera_only", "movement_only", "combined", "static"]:
            counts[key] += seq_stats[key]

        total_frames += len(actions)
        per_sequence_stats.append(seq_stats)

        # Subsample actions for correlation
        if len(all_actions) < MAX_ACTIONS_FOR_CORR:
            n_to_take = min(len(actions), MAX_ACTIONS_FOR_CORR - sum(len(a) for a in all_actions))
            if n_to_take > 0:
                indices = np.random.choice(len(actions), n_to_take, replace=False)
                all_actions.append(actions[indices])

        # Subsample for temporal autocorrelation
        if len(all_classifications) < MAX_FRAMES_FOR_TEMPORAL:
            n_to_take = min(len(classifications), MAX_FRAMES_FOR_TEMPORAL - len(all_classifications))
            all_classifications.extend(classifications[:n_to_take])

    # Compute correlation matrix
    print("Computing action correlation matrix...")
    all_actions_array = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 15))
    if len(all_actions_array) > 0:
        action_corr = compute_action_correlation(all_actions_array)
    else:
        action_corr = np.zeros((15, 15))

    # Compute temporal autocorrelation
    print("Computing temporal autocorrelation...")
    temporal_autocorr = compute_temporal_autocorrelation(all_classifications, temporal_window)

    return CooccurrenceStats(
        total_frames=total_frames,
        camera_only_count=counts["camera_only"],
        movement_only_count=counts["movement_only"],
        combined_count=counts["combined"],
        static_count=counts["static"],
        per_sequence_stats=per_sequence_stats,
        action_correlation=action_corr.tolist(),
        temporal_autocorr={k: v.tolist() for k, v in temporal_autocorr.items()}
    )


def save_results(stats: CooccurrenceStats, output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute percentages
    total = int(stats.total_frames)
    summary = {
        "total_frames": total,
        "camera_only_pct": float((stats.camera_only_count / total * 100) if total > 0 else 0),
        "movement_only_pct": float((stats.movement_only_count / total * 100) if total > 0 else 0),
        "combined_pct": float((stats.combined_count / total * 100) if total > 0 else 0),
        "static_pct": float((stats.static_count / total * 100) if total > 0 else 0),
    }

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    output = {
        "summary": summary,
        "correlation_matrix": stats.action_correlation,
        "temporal_autocorr": stats.temporal_autocorr,
        "per_sequence_stats": convert_to_native(stats.per_sequence_stats)
    }

    output_path = os.path.join(output_dir, "cooccurrence_stats.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Total frames: {total}")
    print(f"  Camera-only: {summary['camera_only_pct']:.1f}%")
    print(f"  Movement-only: {summary['movement_only_pct']:.1f}%")
    print(f"  Combined: {summary['combined_pct']:.1f}%")
    print(f"  Static: {summary['static_pct']:.1f}%")


def plot_results(stats: CooccurrenceStats, output_dir: str):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    total = stats.total_frames

    # 1. Distribution bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Camera-only', 'Movement-only', 'Combined', 'Static']
    counts = [
        stats.camera_only_count,
        stats.movement_only_count,
        stats.combined_count,
        stats.static_count
    ]
    percentages = [(c / total * 100) if total > 0 else 0 for c in counts]

    bars = ax.bar(categories, percentages, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3'])
    ax.set_ylabel('Percentage of Frames (%)', fontsize=12)
    ax.set_title('Action Co-occurrence Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cooccurrence_distribution.png'), dpi=150)
    plt.close()

    # 2. Correlation heatmap
    if len(stats.action_correlation) > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = np.array(stats.action_correlation)

        action_labels = [
            'Yaw-0', 'Yaw-1', 'Yaw-2', 'Yaw-3', 'Yaw-4',
            'Pitch-0', 'Pitch-1', 'Pitch-2',
            'W', 'A', 'S', 'D', 'Jump', 'Sprint', 'Sneak'
        ]

        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    xticklabels=action_labels, yticklabels=action_labels,
                    cbar_kws={'label': 'Correlation'}, ax=ax)
        ax.set_title('Action Dimension Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_correlation_heatmap.png'), dpi=150)
        plt.close()

    # 3. Temporal autocorrelation
    fig, ax = plt.subplots(figsize=(10, 6))
    for action_type, autocorr_values in stats.temporal_autocorr.items():
        lags = range(len(autocorr_values))
        ax.plot(lags, autocorr_values, marker='o', label=action_type.replace('_', ' ').title())

    ax.set_xlabel('Lag (frames)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Temporal Autocorrelation of Action Types', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_autocorr.png'), dpi=150)
    plt.close()

    # 4. Per-sequence scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    movement_pcts = []
    camera_pcts = []

    for seq in stats.per_sequence_stats:
        total_seq = seq['length']
        if total_seq > 0:
            movement_pct = (seq['movement_only'] + seq['combined']) / total_seq * 100
            camera_pct = (seq['camera_only'] + seq['combined']) / total_seq * 100
            movement_pcts.append(movement_pct)
            camera_pcts.append(camera_pct)

    ax.scatter(camera_pcts, movement_pcts, alpha=0.5, s=20)
    ax.set_xlabel('Camera Activity (% of frames)', fontsize=12)
    ax.set_ylabel('Movement Activity (% of frames)', fontsize=12)
    ax.set_title('Per-Sequence Action Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add diagonal line (equal camera/movement)
    ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Equal camera/movement')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_sequence_scatter.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze action co-occurrence in dataset")
    parser.add_argument("--data_dir", type=str, default="preprocessedv5",
                        help="Path to preprocessed dataset directory")
    parser.add_argument("--manifest", type=str, default="manifest.json",
                        help="Manifest filename within data_dir")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                        help="Fraction of files to sample (1.0 = all files)")
    parser.add_argument("--output_dir", type=str, default="diagnostics/cooccurrence",
                        help="Where to save results")
    parser.add_argument("--temporal_window", type=int, default=20,
                        help="Window size for temporal autocorrelation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = os.path.join(args.data_dir, args.manifest)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    # Process dataset
    stats = process_dataset(args.data_dir, manifest_path, args.sample_rate, args.temporal_window)

    # Save results
    save_results(stats, args.output_dir)

    # Generate plots
    plot_results(stats, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
