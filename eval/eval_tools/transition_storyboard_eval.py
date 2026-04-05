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
    parser = argparse.ArgumentParser(description="Generate transition storyboards for canonical action switches")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--context_npz", nargs="+")
    parser.add_argument("--data_dir", default="./preprocessedv5")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--post_steps", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--cold_start", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    model, vqvae, _ = load_model_bundle(args.checkpoint, args.vqvae_ckpt, device)
    contexts = resolve_context_paths(args.context_npz, args.data_dir)
    actions = action_name_map(device)
    transitions = [
        ("static_to_jump", "static", "jump"),
        ("camera_right_to_move_forward", "camera_right", "move_forward"),
        ("camera_right_to_static", "camera_right", "static"),
    ]
    total_steps = args.warmup_steps + args.post_steps
    out_dir = ensure_dir(args.output_dir)

    for context_path in contexts:
        ctx = load_context_window(
            model,
            vqvae,
            context_path,
            start_idx=args.start_idx,
            rollout_steps=total_steps,
            device=device,
            prime_buffer_enabled=not args.cold_start,
        )
        for name, warmup_name, test_name in transitions:
            warm_action = actions[warmup_name]
            test_action = actions[test_name]
            action_seq = warm_action.repeat(total_steps, 1)
            action_seq[args.warmup_steps:] = test_action.repeat(args.post_steps, 1)
            run = rollout_sequence(
                model,
                vqvae,
                ctx["z_start"],
                action_seq,
                topk=args.topk,
                temporal_buffer=ctx["temporal_buffer"],
            )
            frame_imgs = [tensor_to_pil(run["pred_rgb"][i]) for i in range(total_steps)]
            pre_idxs = list(range(max(0, args.warmup_steps - 3), args.warmup_steps))
            post_idxs = list(range(args.warmup_steps, total_steps))
            idxs = pre_idxs + post_idxs
            rows = [("Prediction", [frame_imgs[i] for i in idxs])]
            labels = [f"pre{idx - args.warmup_steps}" if idx < args.warmup_steps else f"post{idx - args.warmup_steps + 1}" for idx in idxs]
            storyboard_post_au = run["argmax_unique_codes"][-1]
            title = f"{Path(args.checkpoint).name} :: {ctx['context_name']} :: {name} :: storyboard_post_au={storyboard_post_au}"
            sheet = make_tiled_image(rows, labels, title=title)
            png_path = out_dir / f"{Path(ctx['context_name']).stem}_{name}_storyboard.png"
            sheet.save(png_path)
            save_json({
                "schema_version": 1,
                "type": "transition_storyboard_eval",
                "config": {
                    "checkpoint": args.checkpoint,
                    "vqvae_ckpt": args.vqvae_ckpt,
                    "context": context_path,
                    "start_idx": args.start_idx,
                    "warmup_steps": args.warmup_steps,
                    "post_steps": args.post_steps,
                    "device": str(device),
                    "topk": args.topk,
                    "cold_start": args.cold_start,
                },
                "artifact_path": str(png_path),
                "transition": {
                    "name": name,
                    "warmup_action": warmup_name,
                    "test_action": test_name,
                    "transition_at": args.warmup_steps,
                    "storyboard_post_au": storyboard_post_au,
                    "protocol": {
                        "warmup_steps": args.warmup_steps,
                        "post_steps": args.post_steps,
                        "note": "storyboard_post_au is specific to this storyboard rollout protocol and is not directly comparable to inference_diagnostics transition metrics",
                    },
                    "argmax_unique_codes": run["argmax_unique_codes"],
                    "mean_max_prob": run["mean_max_prob"],
                },
                "frame_indices": idxs,
            }, out_dir / f"{Path(ctx['context_name']).stem}_{name}_storyboard.metadata.json")
            print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
