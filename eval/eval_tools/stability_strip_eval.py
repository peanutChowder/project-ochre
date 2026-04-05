#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_tools.shared import (
    action_name_map,
    edge_flicker_proxy,
    ensure_dir,
    load_context_window,
    load_model_bundle,
    make_tiled_image,
    resolve_context_paths,
    rollout_sequence,
    save_json,
    select_device,
    tensor_to_pil,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stability strips for fixed repeated-action scenarios")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--context_npz", nargs="+")
    parser.add_argument("--data_dir", default="./preprocessedv5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rollout_steps", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cold_start", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    model, vqvae, _ = load_model_bundle(args.checkpoint, args.vqvae_ckpt, device)
    contexts = resolve_context_paths(args.context_npz, args.data_dir)
    actions = action_name_map(device)
    scenarios = [
        ("static", actions["static"]),
        ("move_forward", actions["move_forward"]),
        ("camera_right", actions["camera_right"]),
    ]
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
        rows = []
        metadata_rows = []
        frame_labels = [f"f{i}" for i in range(args.rollout_steps)]
        for name, action in scenarios:
            action_seq = action.repeat(args.rollout_steps, 1)
            run = rollout_sequence(
                model,
                vqvae,
                ctx["z_start"],
                action_seq,
                topk=args.topk,
                temporal_buffer=ctx["temporal_buffer"],
            )
            rows.append((f"action:{name}", [tensor_to_pil(run["pred_rgb"][i]) for i in range(args.rollout_steps)]))
            metadata_rows.append({
                "action": name,
                "edge_flicker_proxy_per_frame": edge_flicker_proxy(run["pred_rgb"]),
                "argmax_unique_codes": run["argmax_unique_codes"],
                "mean_max_prob": run["mean_max_prob"],
            })

        title = f"{Path(args.checkpoint).name} :: {ctx['context_name']} :: stability"
        sheet = make_tiled_image(rows, frame_labels, title=title)
        png_path = out_dir / f"{Path(ctx['context_name']).stem}_stability_strip.png"
        sheet.save(png_path)
        save_json({
            "schema_version": 1,
            "type": "stability_strip_eval",
            "config": {
                "checkpoint": args.checkpoint,
                "vqvae_ckpt": args.vqvae_ckpt,
                "context": context_path,
                "rollout_steps": args.rollout_steps,
                "start_idx": args.start_idx,
                "device": str(device),
                "topk": args.topk,
                "cold_start": args.cold_start,
            },
            "artifact_path": str(png_path),
            "frame_indices": list(range(args.rollout_steps)),
            "rows": metadata_rows,
        }, out_dir / f"{Path(ctx['context_name']).stem}_stability_strip.metadata.json")
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
