#!/usr/bin/env python3
"""
predict_one_step.py
Loads 6 consecutive frames + actions from your preprocessed dataset,
uses the trained TemporalTransformer to predict the 7th frame,
and saves a side-by-side comparison with the ground-truth frame.
"""

import argparse, torch
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL
from minerl_dataset import MineRLSequenceDataset   # reuse your dataset class
from model_temporal import TemporalTransformer     # your transformer model

# ---------------------- Config ----------------------
SEQ_LEN   = 7
LATENT_C  = 4
LATENT_H  = LATENT_W = 32
FRAME_SIZE = 256

def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype  = torch.float32

    # --- load models ---
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=dtype
    ).to(device).eval()

    model = TemporalTransformer(
        latent_channels=LATENT_C,
        H=LATENT_H, W=LATENT_W
    ).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # --- dataset sample ---
    ds = MineRLSequenceDataset(args.data_dir)
    # pick a random sequence
    idx = torch.randint(len(ds), (1,)).item()
    frames, actions = ds[idx]              # frames: [7,3,256,256]
    frames  = frames.unsqueeze(0).to(device, dtype=dtype)
    actions = actions.unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        # Encode first 6 frames to latents
        latents = vae.encode(frames[:, :-1].reshape(-1,3,256,256)).latent_dist.sample()
        latents = latents.view(1, SEQ_LEN-1, LATENT_C, LATENT_H, LATENT_W)

        # Encode ground-truth next latent
        target  = vae.encode(frames[:, -1]).latent_dist.sample()

        # Predict next latent
        pred = model(latents, actions)

        # Decode both to images
        recon_pred = vae.decode(pred).sample
        recon_true = vae.decode(target).sample

    # convert to uint8 images
    def to_uint8(x):
        x = ((x.clamp(-1,1) + 1)/2)[0]  # [0,1]
        return (x.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

    img_pred  = to_uint8(recon_pred)
    img_true  = to_uint8(recon_true)

    # save side-by-side for visual comparison
    side_by_side = np.concatenate([img_true, img_pred], axis=1)
    Image.fromarray(side_by_side).save(args.out)
    print(f"Saved ground-truth | predicted comparison to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data_dir",  required=True, help="Path to preprocessed dataset")
    p.add_argument("--out", default="one_step_comparison.png")
    main(p.parse_args())