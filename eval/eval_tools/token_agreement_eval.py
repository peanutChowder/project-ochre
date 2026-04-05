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
    compute_quality_summary,
    confidence_to_pil,
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
    parser = argparse.ArgumentParser(description="High-resolution token agreement inspection")
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
        run = rollout_sequence(
            model,
            vqvae,
            ctx["z_start"],
            ctx["actions"],
            topk=args.topk,
            temporal_buffer=ctx["temporal_buffer"],
            gt_tokens=ctx["gt_tokens"],
        )
        metrics = compute_quality_summary(run["pred_rgb"], ctx["gt_rgb"], lpips_fn, args.edge_threshold)
        frame_ids = fixed_frame_indices(args.rollout_steps)
        rows = [
            ("Prediction", [tensor_to_pil(run["pred_rgb"][i]) for i in frame_ids]),
            ("GT", [tensor_to_pil(ctx["gt_rgb"][i]) for i in frame_ids]),
            ("Token Match", [token_agreement_to_pil(run["pred_tokens"][i, 0], ctx["gt_tokens"][i]) for i in frame_ids]),
            ("Confidence", [confidence_to_pil(run["confidence_maps"][i]) for i in frame_ids]),
        ]
        labels = [f"f{i}" for i in frame_ids]
        title = f"{Path(args.checkpoint).name} :: {ctx['context_name']} :: token agreement"
        sheet = make_tiled_image(rows, labels, title=title)
        png_path = out_dir / f"{Path(ctx['context_name']).stem}_token_agreement.png"
        sheet.save(png_path)
        save_json({
            "schema_version": 1,
            "type": "token_agreement_eval",
            "config": {
                "checkpoint": args.checkpoint,
                "vqvae_ckpt": args.vqvae_ckpt,
                "context": context_path,
                "rollout_steps": args.rollout_steps,
                "start_idx": args.start_idx,
                "device": str(device),
                "topk": args.topk,
                "edge_threshold": args.edge_threshold,
                "cold_start": args.cold_start,
            },
            "artifact_path": str(png_path),
            "frame_indices": frame_ids,
            "summary": {
                "token_accuracy_mean": float(sum(run["token_accuracy_per_frame"]) / max(1, len(run["token_accuracy_per_frame"]))),
                "lpips_mean": metrics["lpips_mean"],
                "psnr_mean": metrics["psnr_mean"],
                "edge_f1_mean": metrics["edge_f1_mean"],
            },
            "per_frame_metrics": {
                "token_accuracy_per_frame": run["token_accuracy_per_frame"],
                "mean_max_prob": run["mean_max_prob"],
                "lpips_per_frame": metrics["lpips_per_frame"],
                "psnr_per_frame": metrics["psnr_per_frame"],
            },
        }, out_dir / f"{Path(ctx['context_name']).stem}_token_agreement.metadata.json")
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
