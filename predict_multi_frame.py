#!/usr/bin/env python3
"""
predict_autoregressive_row.py
Show 6 ground-truth context frames followed by N predicted frames in a single row.
"""

import argparse, torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from minerl_dataset import MineRLSequenceDataset
from model_temporal import TemporalTransformer

SEQ_LEN  = 7
LATENT_C = 4
LATENT_H = LATENT_W = 32

def to_uint8(x):
    x = ((x.clamp(-1,1)+1)/2)[0]     # [-1,1] -> [0,1]
    return (x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype  = torch.float32

    # --- Load models ---
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=dtype
    ).to(device).eval()

    model = TemporalTransformer(
        latent_channels=LATENT_C, H=LATENT_H, W=LATENT_W
    ).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # --- Get one 7-frame sample (6 context + 1 target) ---
    ds  = MineRLSequenceDataset(args.data_dir)
    idx = torch.randint(len(ds), (1,)).item()
    frames, actions = ds[idx]          # [7,3,256,256], [6,7]
    frames  = frames.unsqueeze(0).to(device, dtype=dtype)
    actions = actions.unsqueeze(0).to(device, dtype=dtype)

    # Take first 6 frames as context
    context_imgs = [to_uint8(frames[:, i]) for i in range(6)]

    with torch.no_grad():
        ctx_lat = vae.encode(frames[:, :-1].reshape(-1,3,256,256)).latent_dist.sample()
        ctx_lat = ctx_lat.view(1, SEQ_LEN-1, LATENT_C, LATENT_H, LATENT_W)

    cur_lat, cur_act = ctx_lat.clone(), actions.clone()
    generated = []

    # --- Autoregressive prediction of N frames ---
    for step in range(args.n):
        with torch.no_grad():
            pred_lat = model(cur_lat, cur_act)
            pred_img = to_uint8(vae.decode(pred_lat).sample)
        generated.append(pred_img)

        # Shift context: drop first latent, add prediction
        cur_lat = torch.cat([cur_lat[:, 1:], pred_lat.unsqueeze(1)], dim=1)
        cur_act = torch.cat([cur_act[:, 1:], torch.zeros_like(cur_act[:, 0:1])], dim=1)

    # --- Combine into single row: 6 GT context + N predictions ---
    all_imgs = context_imgs + generated
    H, W = all_imgs[0].shape[0], all_imgs[0].shape[1]
    grid = np.zeros((H, W * len(all_imgs), 3), dtype=np.uint8)
    for i, img in enumerate(all_imgs):
        grid[:, i * W:(i + 1) * W] = img

    Image.fromarray(grid).save(args.out)
    print(f"Saved grid with {len(all_imgs)} columns to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--data_dir", required=True, help="Path to preprocessed dataset")
    p.add_argument("--out", default="prediction_row.png", help="Output PNG path")
    p.add_argument("--n", type=int, default=10, help="Number of future frames to generate")
    args = p.parse_args()
    main(args)