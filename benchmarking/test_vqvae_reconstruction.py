#!/usr/bin/env python3
"""
Test VQVAE reconstruction quality by encoding and decoding frames from preprocessed data.
Outputs a video showing original vs reconstructed frames to assess motion fidelity.

Usage:
    python test_vqvae_reconstruction.py preprocessedv5/seed_1000_part_1000.npz
    python test_vqvae_reconstruction.py preprocessedv5/seed_1000_part_1000.npz --output reconstruction.mp4
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

# Import VQVAE from local module
from vq_vae.vq_vae import VQVAE


def load_vqvae(vqvae_ckpt: str, device: torch.device) -> VQVAE:
    """Load VQVAE model from checkpoint."""
    ckpt = torch.load(vqvae_ckpt, map_location=device)

    # Get config with defaults
    config = ckpt.get("config", {})
    embedding_dim = config.get("embedding_dim", 384)
    codebook_size = config.get("codebook_size", 1024)
    beta = config.get("beta", 0.25)
    ema_decay = config.get("ema_decay", 0.99)

    # Create model
    vqvae = VQVAE(
        embedding_dim=embedding_dim,
        num_embeddings=codebook_size,
        commitment_cost=beta,
        decay=ema_decay,
    ).to(device)

    # Load state dicts
    vqvae.encoder.load_state_dict(ckpt["encoder"])
    vqvae.decoder.load_state_dict(ckpt["decoder"])
    vqvae.vq_vae.load_state_dict(ckpt["quantizer"])

    vqvae.eval()
    return vqvae


def decode_tokens_to_frames(vqvae: VQVAE, tokens: np.ndarray, batch_size: int = 32, device: torch.device = None) -> np.ndarray:
    """
    Decode tokens back to RGB frames.

    Args:
        vqvae: VQVAE model
        tokens: (T, H, W) array of uint16 token indices
        batch_size: Number of frames to decode at once
        device: Torch device

    Returns:
        frames: (T, 72, 128, 3) array of RGB frames in [0, 255] uint8
    """
    if device is None:
        device = next(vqvae.parameters()).device

    T = tokens.shape[0]
    frames = []

    with torch.no_grad():
        for i in tqdm(range(0, T, batch_size), desc="Decoding frames"):
            batch_tokens = tokens[i:i + batch_size]

            # Convert to torch tensor and move to device
            batch_tokens_torch = torch.from_numpy(batch_tokens).long().to(device)

            # Decode: (B, H, W) -> (B, 3, 72, 128)
            batch_frames = vqvae.decode_code(batch_tokens_torch)

            # Convert to numpy: (B, 3, 72, 128) -> (B, 72, 128, 3)
            batch_frames = batch_frames.permute(0, 2, 3, 1).cpu().numpy()

            # Clamp to [0, 1] and convert to uint8
            batch_frames = np.clip(batch_frames, 0, 1)
            batch_frames = (batch_frames * 255).astype(np.uint8)

            frames.append(batch_frames)

    return np.concatenate(frames, axis=0)


def save_video(frames: np.ndarray, output_path: str, fps: int = 20):
    """
    Save frames as a video file.

    Args:
        frames: (T, H, W, 3) array of RGB frames
        output_path: Path to output video file
        fps: Frames per second
    """
    T, H, W, C = frames.shape

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    print(f"Writing video to {output_path}...")
    for i in tqdm(range(T), desc="Writing frames"):
        # OpenCV uses BGR, so convert from RGB
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved: {output_path}")
    print(f"  Resolution: {W}x{H}")
    print(f"  Frames: {T}")
    print(f"  Duration: {T/fps:.2f}s @ {fps} fps")


def main():
    parser = argparse.ArgumentParser(description="Test VQVAE reconstruction quality")
    parser.add_argument("npz_path", type=str, help="Path to .npz file from preprocessedv5/")
    parser.add_argument("--vqvae-ckpt", type=str,
                        default="vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt",
                        help="Path to VQVAE checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path (default: <npz_name>_reconstruction.mp4)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Output video FPS (default: 20)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for decoding (default: 32)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")

    args = parser.parse_args()

    # Validate inputs
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}", file=sys.stderr)
        sys.exit(1)

    vqvae_ckpt_path = Path(args.vqvae_ckpt)
    if not vqvae_ckpt_path.exists():
        print(f"Error: VQVAE checkpoint not found: {vqvae_ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Set up output path
    if args.output is None:
        output_path = npz_path.stem + "_reconstruction.mp4"
    else:
        output_path = args.output

    # Set up device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading data from: {npz_path}")

    # Load .npz file
    with np.load(npz_path) as data:
        tokens = data["tokens"]  # (T, 18, 32)
        actions = data["actions"] if "actions" in data else None

    print(f"Loaded tokens: shape={tokens.shape}, dtype={tokens.dtype}")
    if actions is not None:
        print(f"Loaded actions: shape={actions.shape}, dtype={actions.dtype}")

    # Limit frames if requested
    if args.max_frames is not None:
        tokens = tokens[:args.max_frames]
        print(f"Limited to {args.max_frames} frames")

    # Load VQVAE
    print(f"Loading VQVAE from: {vqvae_ckpt_path}")
    vqvae = load_vqvae(str(vqvae_ckpt_path), device)
    print(f"VQVAE loaded successfully")

    # Decode tokens to frames
    print(f"Decoding {tokens.shape[0]} frames...")
    frames = decode_tokens_to_frames(vqvae, tokens, batch_size=args.batch_size, device=device)
    print(f"Decoded frames: shape={frames.shape}, dtype={frames.dtype}")

    # Save video
    save_video(frames, output_path, fps=args.fps)

    print("\nDone! You can now view the reconstruction to assess motion quality.")
    print(f"Command: open {output_path}  # or use your video player")


if __name__ == "__main__":
    main()
