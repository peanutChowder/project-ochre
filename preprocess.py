#!/usr/bin/env python3
"""
Preprocess MineRL Navigate dataset.

For every subfolder inside --parent_dir that contains
  recording.mp4 + rendered.npz:
    * read frames from mp4
    * read action arrays from npz
    * downsample to target FPS
    * resize to target size
    * create sequences of length seq_len (K context + 1 target)
    * save each sequence as compressed npz in --out_dir
"""

import argparse, os, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio

def build_action_matrix(d):
    """Combine action$* arrays into a [T,7] float32 matrix."""
    camera = d["action$camera"]        # shape (T, 2) pitch, yaw
    return np.stack([
        d["action$forward"],
        d["action$left"],
        d["action$back"],
        d["action$right"],
        d["action$jump"],
        np.clip(camera[:,1] / 180.0, -1, 1),  # yaw
        np.clip(camera[:,0] / 180.0, -1, 1),  # pitch
    ], axis=1).astype(np.float32)

def process_trajectory(traj_dir: Path,
                       out_dir: Path,
                       fps_target: int = 8,
                       seq_len: int = 7,
                       size: int = 256) -> int:
    """
    Process a single trajectory directory.
    Returns number of sequences written.
    """
    # ---- Load action arrays ----
    d = np.load(traj_dir / "rendered.npz", allow_pickle=True)
    actions = build_action_matrix(d)

    # ---- Read frames from recording.mp4 ----
    frames = [np.asarray(f) for f in iio.imiter(traj_dir / "recording.mp4")]
    T = min(len(frames), len(actions))
    frames, actions = frames[:T], actions[:T]

    # ---- Downsample to target FPS (MineRL native ≈20) ----
    step = max(1, int(round(20 / fps_target)))
    frames = frames[::step]
    actions = actions[::step]
    T = len(frames)

    # ---- Resize frames ----
    resized = np.zeros((T, size, size, 3), dtype=np.uint8)
    for i, img in enumerate(frames):
        im = Image.fromarray(img)
        im = im.resize((size, size), Image.BICUBIC)
        resized[i] = np.asarray(im, dtype=np.uint8)

    # ---- Sliding window sequences ----
    K = seq_len - 1
    count = 0
    for i in range(T - seq_len + 1):
        vid = resized[i:i + seq_len]
        act = actions[i:i + K]
        out_name = f"{traj_dir.name}_{i:06d}.npz"
        np.savez_compressed(out_dir / out_name,
                            video=vid,
                            actions=act)
        count += 1
    return count

def main(parent_dir: str,
         out_dir: str,
         fps_target: int = 8,
         seq_len: int = 7,
         size: int = 256):
    parent = Path(parent_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trajs = [p for p in parent.iterdir()
             if p.is_dir() and (p / "rendered.npz").exists()
                             and (p / "recording.mp4").exists()]

    total = 0
    for traj in tqdm(trajs, desc="Processing trajectories"):
        total += process_trajectory(
            traj,
            out,
            fps_target=fps_target,
            seq_len=seq_len,
            size=size
        )
    print(f"\nProcessed {len(trajs)} trajectories")
    print(f"Saved {total} sequences to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent_dir", required=True,
                   help="Path to MineRLNavigate-v0 parent directory")
    p.add_argument("--out_dir", required=True,
                   help="Directory to save processed sequences")
    p.add_argument("--fps_target", type=int, default=8,
                   help="Target FPS after downsampling (default 8)")
    p.add_argument("--seq_len", type=int, default=7,
                   help="Frames per sequence (default 7)")
    p.add_argument("--size", type=int, default=256,
                   help="Output frame size (default 256)")
    args = p.parse_args()
    main(**vars(args))