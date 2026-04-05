#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lpips

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_tools.shared import (
    _edge_map,
    compute_decoded_target_reference,
    compute_quality_summary,
    edge_to_pil,
    ensure_dir,
    fixed_frame_indices,
    load_context_window,
    load_model_bundle,
    make_tiled_image,
    resolve_context_paths,
    rollout_sequence,
    save_json,
    select_device,
    tensor_to_pil,
    token_agreement_to_pil,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-context contact sheets for agent evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--context_npz", nargs="+")
    parser.add_argument("--data_dir", default="./preprocessedv5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--edge_threshold", type=float, default=0.08)
    parser.add_argument("--include_oracle", action="store_true")
    parser.add_argument("--cold_start", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    model, vqvae, _ = load_model_bundle(args.checkpoint, args.vqvae_ckpt, device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    contexts = resolve_context_paths(args.context_npz, args.data_dir)
    out_dir = ensure_dir(args.output_dir)

    for context_path in contexts:
        ctx = load_context_window(
            model,
            vqvae,
            context_path,
            start_idx=args.start_idx,
            rollout_steps=args.rollout_steps,
            device=device,
            prime_buffer_enabled=not args.cold_start,
        )
        pred = rollout_sequence(
            model,
            vqvae,
            ctx["z_start"],
            ctx["actions"],
            topk=args.topk,
            temporal_buffer=ctx["temporal_buffer"],
            gt_tokens=ctx["gt_tokens"],
        )
        reference_decode_metrics = compute_decoded_target_reference(vqvae, ctx["gt_tokens"], ctx["gt_rgb"], lpips_fn)
        pred_metrics = compute_quality_summary(pred["pred_rgb"], ctx["gt_rgb"], lpips_fn, args.edge_threshold)

        oracle = None
        oracle_metrics = None
        if args.include_oracle:
            oracle = rollout_sequence(
                model,
                vqvae,
                ctx["z_start"],
                ctx["actions"],
                topk=args.topk,
                temporal_buffer=ctx["temporal_buffer"],
                oracle_tokens=ctx["gt_tokens"],
                gt_tokens=ctx["gt_tokens"],
            )
            oracle_metrics = compute_quality_summary(oracle["pred_rgb"], ctx["gt_rgb"], lpips_fn, args.edge_threshold)

        frame_ids = fixed_frame_indices(args.rollout_steps)
        col_labels = [f"f{idx}" for idx in frame_ids]
        rows = [
            ("Target Decode", [tensor_to_pil(ctx["gt_rgb"][i]) for i in frame_ids]),
            ("Prediction", [tensor_to_pil(pred["pred_rgb"][i]) for i in frame_ids]),
        ]
        if oracle is not None:
            rows.append(("Oracle", [tensor_to_pil(oracle["pred_rgb"][i]) for i in frame_ids]))
        rows.extend([
            ("Token Match", [token_agreement_to_pil(pred["pred_tokens"][i, 0], ctx["gt_tokens"][i]) for i in frame_ids]),
            ("Pred Edges", [edge_to_pil(_edge_map(pred["pred_rgb"][i:i + 1])[0]) for i in frame_ids]),
            ("Target Edges", [edge_to_pil(_edge_map(ctx["gt_rgb"][i:i + 1])[0]) for i in frame_ids]),
        ])

        title = f"{Path(args.checkpoint).name} :: {ctx['context_name']} :: decoded-target reference"
        sheet = make_tiled_image(rows, col_labels, title=title)
        png_path = out_dir / f"{Path(ctx['context_name']).stem}_contact_sheet.png"
        sheet.save(png_path)

        metadata = {
            "schema_version": 1,
            "type": "contact_sheet_eval",
            "config": {
                "checkpoint": args.checkpoint,
                "vqvae_ckpt": args.vqvae_ckpt,
                "context": context_path,
                "rollout_steps": args.rollout_steps,
                "start_idx": args.start_idx,
                "device": str(device),
                "topk": args.topk,
                "edge_threshold": args.edge_threshold,
                "include_oracle": args.include_oracle,
                "cold_start": args.cold_start,
            },
            "artifact_path": str(png_path),
            "frame_indices": frame_ids,
            "summary": {
                "pred": {
                    "lpips_mean": pred_metrics["lpips_mean"],
                    "psnr_mean": pred_metrics["psnr_mean"],
                    "edge_f1_mean": pred_metrics["edge_f1_mean"],
                    "token_accuracy_mean": float(sum(pred["token_accuracy_per_frame"]) / max(1, len(pred["token_accuracy_per_frame"]))),
                },
                "reference_decode": {
                    "reference_type": reference_decode_metrics["reference_type"],
                    "reference_note": reference_decode_metrics["reference_note"],
                },
                "oracle": {
                    "lpips_mean": oracle_metrics["lpips_mean"],
                    "psnr_mean": oracle_metrics["psnr_mean"],
                    "edge_f1_mean": oracle_metrics["edge_f1_mean"],
                } if oracle_metrics is not None else None,
            },
            "per_frame_metrics": {
                "pred": pred_metrics,
                "reference_decode": reference_decode_metrics,
                "oracle": oracle_metrics,
                "pred_argmax_unique_codes": pred["argmax_unique_codes"],
                "pred_mean_max_prob": pred["mean_max_prob"],
                "pred_token_accuracy_per_frame": pred["token_accuracy_per_frame"],
            },
            "notes": [
                "Target Decode is VQ-VAE decode(gt_tokens), not raw source RGB.",
                "These metrics measure distance to the decoded target-token reference, not direct source-frame fidelity.",
            ],
        }
        save_json(metadata, out_dir / f"{Path(ctx['context_name']).stem}_contact_sheet.metadata.json")
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
