#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def score_band(value: float, good: float, fair: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= good:
            return "good"
        if value >= fair:
            return "fair"
        return "poor"
    if value <= good:
        return "good"
    if value <= fair:
        return "fair"
    return "poor"


def get_visual_metrics(summary: dict) -> tuple[dict, list[str]]:
    notes: list[str] = []
    if "pred" in summary:
        pred = summary["pred"]
        metrics = {
            "textures_discrete_value": pred.get("token_accuracy_mean"),
            "textures_discrete_label": "token_accuracy_mean",
            "edge_alignment": pred.get("edge_f1_mean"),
            "flicker_control": None,
        }
        notes.append(
            "Visual JSON is contact-sheet metadata; textures_discrete is approximated from token_accuracy_mean and flicker_control is unavailable without a stability-strip metadata file."
        )
        return metrics, notes
    metrics = {
        "textures_discrete_value": (
            summary["pred_unique_codes_mean"] / max(1e-8, summary["gt_unique_codes_mean"])
            if "pred_unique_codes_mean" in summary and "gt_unique_codes_mean" in summary
            else None
        ),
        "textures_discrete_label": "pred_unique_codes_mean / gt_unique_codes_mean",
        "edge_alignment": summary.get("edge_f1_mean"),
        "flicker_control": summary.get("edge_flicker_l1_mean"),
    }
    return metrics, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a compact agent-friendly rubric summary")
    parser.add_argument("--inference_json", required=True)
    parser.add_argument("--visual_json", required=True)
    parser.add_argument("--stability_json")
    parser.add_argument("--artifact_paths", nargs="*", default=[])
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    inf = json.load(open(args.inference_json))
    vis = json.load(open(args.visual_json))
    stability = json.load(open(args.stability_json)) if args.stability_json else None
    s_inf = inf["summary"]
    s_vis = vis["summary"]
    visual_metrics, metric_notes = get_visual_metrics(s_vis)
    stability_flicker = None
    if stability is not None:
        static_rows = [row for row in stability.get("rows", []) if row.get("action") == "static"]
        if static_rows:
            vals = static_rows[0].get("edge_flicker_proxy_per_frame", [])
            if vals:
                stability_flicker = float(sum(vals) / len(vals))

    computed = {
        "static_to_jump_post_au": score_band(
            s_inf["static_to_jump_post_au"], good=40.0, fair=30.0, higher_is_better=True
        ),
        "avg_max_prob": score_band(
            s_inf["avg_max_prob"], good=0.40, fair=0.50, higher_is_better=False
        ),
        "scene_persistence": score_band(s_inf["consistency_score"], good=0.30, fair=0.20, higher_is_better=True),
        "motion_coherence": score_band(s_inf["action_effect_magnitude"], good=0.28, fair=0.24, higher_is_better=True),
        "textures_discrete": (
            score_band(visual_metrics["textures_discrete_value"], good=0.13, fair=0.10, higher_is_better=True)
            if visual_metrics["textures_discrete_value"] is not None
            else "unknown"
        ),
        "edge_alignment": (
            score_band(visual_metrics["edge_alignment"], good=0.78, fair=0.72, higher_is_better=True)
            if visual_metrics["edge_alignment"] is not None
            else "unknown"
        ),
        "flicker_control": (
            score_band(stability_flicker, good=0.06, fair=0.09, higher_is_better=False)
            if stability_flicker is not None
            else score_band(visual_metrics["flicker_control"], good=0.12, fair=0.145, higher_is_better=False)
            if visual_metrics["flicker_control"] is not None
            else "unknown"
        ),
    }
    visual = {"block_boundaries_clear": "unrated", "overall_minecraftness": "unrated"}
    payload = {
        "schema_version": 1,
        "scalar_values": {
            "static_to_jump_post_au": s_inf["static_to_jump_post_au"],
            "avg_max_prob": s_inf["avg_max_prob"],
            "consistency_score": s_inf["consistency_score"],
            "action_effect_magnitude": s_inf["action_effect_magnitude"],
        },
        "computed": computed,
        "visual": visual,
        "source_files": {
            "inference_json": args.inference_json,
            "visual_json": args.visual_json,
            "stability_json": args.stability_json,
        },
        "artifact_paths": args.artifact_paths,
        "notes": [
            "Computed fields are threshold-based and should be recalibrated if the evaluation distribution changes.",
            "Visual fields are intentionally left unrated for agent/manual review based on generated artifacts.",
            *metric_notes,
            f"textures_discrete source: {visual_metrics['textures_discrete_label']}",
            "flicker_control uses stability-strip metadata when provided; otherwise it falls back to any flicker metric present in the visual JSON.",
        ],
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
