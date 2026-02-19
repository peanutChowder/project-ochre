#!/usr/bin/env python3
"""
Isolate movement-rich sequences from dataset.

Identifies sequences with high movement activity and low camera activity for
potential curriculum learning or dataset curation.

Usage:
    python isolate_movement_sequences.py \
        --data_dir preprocessedv5/ \
        --output_dir diagnostics/movement_sequences \
        --camera_static_threshold 0.80 \
        --movement_active_threshold 0.30
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class SequenceMetrics:
    """Metrics for a single sequence."""
    file: str
    length: int
    camera_static_pct: float
    movement_active_pct: float
    wasd_active_pct: float
    jump_active_pct: float
    movement_richness_score: float


@dataclass
class FilterResults:
    """Results from filtering operation."""
    total_sequences: int
    qualified_sequences: int
    total_frames: int
    qualified_frames: int
    metrics: List[SequenceMetrics]


def compute_sequence_metrics(actions: np.ndarray, filename: str) -> SequenceMetrics:
    """
    Compute movement metrics for a single sequence.

    actions: (T, 15) action array
    """
    T = len(actions)

    # Camera static: yaw center (dim 2) AND pitch center (dim 6)
    yaw_center = actions[:, 2] > 0.5
    pitch_center = actions[:, 6] > 0.5
    camera_static = yaw_center & pitch_center
    camera_static_pct = camera_static.mean() * 100

    # Movement active: any of WASD, jump, sprint, sneak (dims 8-14)
    movement_active = np.any(actions[:, 8:15] > 0.5, axis=1)
    movement_active_pct = movement_active.mean() * 100

    # WASD specifically
    wasd_active = np.any(actions[:, 8:12] > 0.5, axis=1)
    wasd_active_pct = wasd_active.mean() * 100

    # Jump specifically
    jump_active = actions[:, 12] > 0.5
    jump_active_pct = jump_active.mean() * 100

    # Movement richness score:
    # Higher score = more movement, less camera, with bonus for diversity
    movement_richness_score = (
        movement_active_pct * 1.0 +
        camera_static_pct * 0.5 +
        jump_active_pct * 0.3
    )

    return SequenceMetrics(
        file=filename,
        length=T,
        camera_static_pct=camera_static_pct,
        movement_active_pct=movement_active_pct,
        wasd_active_pct=wasd_active_pct,
        jump_active_pct=jump_active_pct,
        movement_richness_score=movement_richness_score
    )


def filter_sequences(data_dir: str,
                      manifest_path: str,
                      camera_static_threshold: float,
                      movement_active_threshold: float,
                      min_sequence_length: int) -> FilterResults:
    """
    Filter sequences based on movement criteria.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    sequences = manifest["sequences"]
    total_sequences = len(sequences)
    total_frames = sum(s["length"] for s in sequences)

    print(f"Processing {total_sequences} sequences...")

    qualified_metrics = []

    for seq_meta in tqdm(sequences, desc="Filtering sequences"):
        # Skip short sequences
        if seq_meta["length"] < min_sequence_length:
            continue

        file_path = os.path.join(data_dir, seq_meta["file"])

        if not os.path.exists(file_path):
            continue

        # Load actions
        with np.load(file_path, mmap_mode='r') as data:
            actions = np.array(data["actions"])

        # Compute metrics
        metrics = compute_sequence_metrics(actions, seq_meta["file"])

        # Apply thresholds
        if (metrics.camera_static_pct >= camera_static_threshold * 100 and
            metrics.movement_active_pct >= movement_active_threshold * 100):
            qualified_metrics.append(metrics)

    # Sort by movement richness score (descending)
    qualified_metrics.sort(key=lambda m: m.movement_richness_score, reverse=True)

    qualified_frames = sum(m.length for m in qualified_metrics)

    return FilterResults(
        total_sequences=total_sequences,
        qualified_sequences=len(qualified_metrics),
        total_frames=total_frames,
        qualified_frames=qualified_frames,
        metrics=qualified_metrics
    )


def save_results(results: FilterResults,
                  output_dir: str,
                  camera_static_threshold: float,
                  movement_active_threshold: float,
                  min_sequence_length: int):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Save filtered manifest
    manifest = {
        "sequences": [
            {
                "file": m.file,
                "length": m.length,
                "camera_static_pct": m.camera_static_pct,
                "movement_active_pct": m.movement_active_pct,
                "wasd_active_pct": m.wasd_active_pct,
                "jump_active_pct": m.jump_active_pct,
                "movement_richness_score": m.movement_richness_score
            }
            for m in results.metrics
        ]
    }

    manifest_path = os.path.join(output_dir, "movement_sequences_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save statistics
    stats = {
        "total_sequences": results.total_sequences,
        "qualified_sequences": results.qualified_sequences,
        "qualification_rate": (results.qualified_sequences / results.total_sequences * 100)
                              if results.total_sequences > 0 else 0,
        "total_frames": results.total_frames,
        "qualified_frames": results.qualified_frames,
        "frame_retention_rate": (results.qualified_frames / results.total_frames * 100)
                                if results.total_frames > 0 else 0,
        "thresholds": {
            "camera_static": camera_static_threshold,
            "movement_active": movement_active_threshold,
            "min_length": min_sequence_length
        }
    }

    stats_path = os.path.join(output_dir, "filter_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"\nStatistics:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Qualified sequences: {stats['qualified_sequences']} ({stats['qualification_rate']:.1f}%)")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Qualified frames: {stats['qualified_frames']} ({stats['frame_retention_rate']:.1f}%)")


def plot_results(results: FilterResults, output_dir: str):
    """Generate visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sequence scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    camera_pcts = [m.camera_static_pct for m in results.metrics]
    movement_pcts = [m.movement_active_pct for m in results.metrics]

    ax.scatter(camera_pcts, movement_pcts, alpha=0.6, s=30, c='#45B7D1')
    ax.set_xlabel('Camera Static (% of frames)', fontsize=12)
    ax.set_ylabel('Movement Active (% of frames)', fontsize=12)
    ax.set_title('Movement-Rich Sequences: Camera vs Movement Activity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add threshold lines
    ax.axvline(x=80, color='r', linestyle='--', alpha=0.5, label='Camera static threshold (80%)')
    ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Movement active threshold (30%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sequence_scatter.png'), dpi=150)
    plt.close()

    # 2. Movement richness histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    richness_scores = [m.movement_richness_score for m in results.metrics]

    ax.hist(richness_scores, bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Movement Richness Score', fontsize=12)
    ax.set_ylabel('Number of Sequences', fontsize=12)
    ax.set_title('Distribution of Movement Richness Scores', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean line
    mean_score = np.mean(richness_scores) if richness_scores else 0
    ax.axvline(x=mean_score, color='r', linestyle='--', linewidth=2,
               label=f'Mean: {mean_score:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'movement_richness_histogram.png'), dpi=150)
    plt.close()

    # 3. Qualified vs total comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Sequences', 'Frames']
    total_values = [results.total_sequences, results.total_frames]
    qualified_values = [results.qualified_sequences, results.qualified_frames]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, total_values, width, label='Total', color='#95E1D3')
    bars2 = ax.bar(x + width/2, qualified_values, width, label='Qualified', color='#45B7D1')

    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Movement-Rich Dataset Filtering Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (bar, total, qualified) in enumerate(zip(bars2, total_values, qualified_values)):
        height = bar.get_height()
        pct = (qualified / total * 100) if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2., height + total * 0.02,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qualified_vs_total.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def copy_qualified_sequences(results: FilterResults,
                               src_dir: str,
                               dst_dir: str):
    """
    Copy or symlink qualified sequence files to output directory.
    """
    os.makedirs(dst_dir, exist_ok=True)

    print(f"\nCopying {len(results.metrics)} qualified sequences to {dst_dir}...")

    for metrics in tqdm(results.metrics, desc="Copying files"):
        src_path = os.path.join(src_dir, metrics.file)
        dst_path = os.path.join(dst_dir, metrics.file)

        if not os.path.exists(src_path):
            continue

        # Use symlink for efficiency (or copy if on different filesystems)
        try:
            if not os.path.exists(dst_path):
                os.symlink(src_path, dst_path)
        except OSError:
            # Symlink failed (different filesystem), use copy instead
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    print(f"Sequences copied/linked to {dst_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Isolate movement-rich sequences")
    parser.add_argument("--data_dir", type=str, default="preprocessedv5",
                        help="Path to preprocessed dataset")
    parser.add_argument("--manifest", type=str, default="manifest.json",
                        help="Manifest filename")
    parser.add_argument("--output_dir", type=str, default="diagnostics/movement_sequences",
                        help="Where to save filtered manifest")
    parser.add_argument("--camera_static_threshold", type=float, default=0.80,
                        help="Min fraction of frames with camera in center bins")
    parser.add_argument("--movement_active_threshold", type=float, default=0.30,
                        help="Min fraction of frames with WASD active")
    parser.add_argument("--min_sequence_length", type=int, default=100,
                        help="Minimum sequence length to consider")
    parser.add_argument("--copy_files", action="store_true",
                        help="Copy/symlink qualified files to output directory")

    args = parser.parse_args()

    manifest_path = os.path.join(args.data_dir, args.manifest)

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    # Filter sequences
    results = filter_sequences(
        args.data_dir,
        manifest_path,
        args.camera_static_threshold,
        args.movement_active_threshold,
        args.min_sequence_length
    )

    # Save results
    save_results(
        results,
        args.output_dir,
        args.camera_static_threshold,
        args.movement_active_threshold,
        args.min_sequence_length
    )

    # Generate plots
    plot_results(results, args.output_dir)

    # Optionally copy files
    if args.copy_files:
        dst_dir = os.path.join(args.output_dir, "sequences")
        copy_qualified_sequences(results, args.data_dir, dst_dir)

    print("\nFiltering complete!")


if __name__ == "__main__":
    main()
