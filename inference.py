#!/usr/bin/env python3
"""
Interactive or scripted inference on Apple M-series Mac.
Generates a rollout video given a trained TemporalTransformer checkpoint.
"""

import argparse, torch, numpy as np, imageio
from PIL import Image
from diffusers import AutoencoderKL
from model_temporal import TemporalTransformer
from minerl_dataset import MineRLSequenceDataset

# ---------------------- Config ----------------------
SEQ_LEN   = 7
LATENT_C  = 4
LATENT_H  = LATENT_W = 32
FRAME_SIZE = 256

def parse_action_vector(action_str: str, device, dtype):
    values = [float(x) for x in action_str.split(",") if x.strip()]
    if len(values) != 7:
        raise ValueError(f"Expected 7 comma-separated values for action, got {len(values)}")
    return torch.tensor(values, device=device, dtype=dtype).view(1, 1, 7)

def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype  = torch.float32   # use float32 on MPS for stability
    print(f"Running on {device}")

    # --- Load models ---
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=dtype
    ).to(device)
    vae.eval()

    model = TemporalTransformer(
        latent_channels=LATENT_C,
        H=LATENT_H, W=LATENT_W
    ).to(device)
    model_data = torch.load(args.checkpoint, map_location=device)
    model_state_dict = model_data["model_state"]
    model.load_state_dict(model_state_dict)
    model.eval()

    if args.context_npz and args.data_dir:
        raise ValueError("Specify only one of --context_npz or --data_dir")
    if args.context_index is not None and not args.data_dir:
        raise ValueError("--context_index requires --data_dir")

    # --- Prepare initial context ---
    context_pixels = None
    context_actions = None

    if args.context_npz:
        data = np.load(args.context_npz, allow_pickle=False)
        video = torch.from_numpy(data["video"])  # [7, H, W, 3]
        actions_arr = torch.from_numpy(data["actions"])  # [6, 7]
        if video.shape[0] < SEQ_LEN:
            raise ValueError(f"Context file must contain at least {SEQ_LEN} frames")
        context_pixels = video[:SEQ_LEN-1].permute(0, 3, 1, 2).float() / 255.0
        context_actions = actions_arr[:SEQ_LEN-1].float()
    elif args.data_dir:
        dataset = MineRLSequenceDataset(args.data_dir)
        if args.context_index is not None:
            if not (0 <= args.context_index < len(dataset)):
                raise ValueError("--context_index out of range for provided --data_dir")
            idx = args.context_index
        else:
            idx = torch.randint(len(dataset), (1,), dtype=torch.long).item()
        frames, actions_arr = dataset[idx]
        context_pixels = frames[:SEQ_LEN-1]
        context_actions = actions_arr[:SEQ_LEN-1]
        print(f"Using context index {idx} from {args.data_dir}")
    else:
        if not args.start_image:
            raise ValueError("Provide --start_image when no dataset context is supplied")
        init_img = Image.open(args.start_image).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
        init = torch.from_numpy(np.array(init_img)).permute(2,0,1).unsqueeze(0).float() / 255.
        init = init.to(device, dtype=dtype)
        context_pixels = init.repeat(SEQ_LEN-1, 1, 1, 1)

    context_pixels = context_pixels.to(device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(context_pixels).latent_dist.sample()
        latents = latents.unsqueeze(0)  # [1,K,C,H,W]

    # --- Actions setup ---
    if context_actions is not None:
        actions = context_actions.unsqueeze(0).to(device, dtype=dtype)
        if args.action:
            base_action = parse_action_vector(args.action, device, dtype)
        else:
            base_action = actions[:, -1:].clone()
    else:
        default_action = args.action or "1,0,0,0,0,0,0"
        base_action = parse_action_vector(default_action, device, dtype)
        actions = base_action.repeat(1, SEQ_LEN-1, 1)

    # --- Rollout generation ---
    frames_out = []
    for step in range(args.steps):
        pred_latent = model(latents, actions)
        img = vae.decode(pred_latent).sample
        img = ((img.clamp(-1,1) + 1) / 2).cpu()
        img = (img[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        frames_out.append(img)

        # update context for next step
        latents = torch.cat([latents[:,1:], pred_latent.unsqueeze(1)], dim=1)
        actions = torch.cat([actions[:,1:], base_action], dim=1)

    # --- Save video ---
    imageio.mimsave(
        args.out,
        [frame.astype('uint8') for frame in frames_out],
        fps=args.fps,
        format='FFMPEG',
        codec='libx264'
    )
    print(f"Saved {len(frames_out)} frames to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--start_image", help="Seed image (jpg/png). Required if no context is provided")
    p.add_argument("--context_npz", help="Path to preprocessed sequence (.npz) to use as initial context")
    p.add_argument("--data_dir", help="Directory of preprocessed sequences to sample context from")
    p.add_argument("--context_index", type=int, help="Specific dataset index to use when sampling context")
    p.add_argument("--out", default="dream_rollout.mp4")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument(
        "--action",
        help="Comma-separated 7D action appended each step; defaults to last context action when available"
    )
    main(p.parse_args())
    
