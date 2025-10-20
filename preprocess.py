#!/usr/bin/env python3
"""
Preprocess MineRL Navigate dataset.

For every subfolder inside --parent_dir that contains
  recording.mp4 + rendered.npz:
    * read frames from mp4
    * read action arrays from npz
    * downsample to target FPS
    * resize to target size
    * save full trajectory as compressed npz in --out_dir

Note: Each MP4 file is processed as a full trajectory, not split into shorter sequences.
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
                       size: int = 64) -> int:
    """
    Process a single trajectory directory.
    Returns 1 if saved successfully.

    Note: During downsampling, movement action booleans are max-pooled within each frame group
    to preserve transient keypresses (e.g., "W" pressed between sampled frames).
    Camera deltas are averaged across the window to preserve smooth movement data.
    """
    # ---- Load action arrays ----
    d = np.load(traj_dir / "rendered.npz", allow_pickle=True)
    actions = build_action_matrix(d)

    # ---- Read frames from recording.mp4 ----
    frames = [np.asarray(f) for f in iio.imiter(traj_dir / "recording.mp4")]
    T = min(len(frames), len(actions))
    frames, actions = frames[:T], actions[:T]

    # ---- Skip first 1 second of footage ----
    skip_frames = int(round(20))  # MineRL native ~20 FPS
    frames = frames[skip_frames:]
    actions = actions[skip_frames:]
    T = len(frames)

    # ---- Downsample to target FPS (MineRL native ≈20) ----
    step = max(1, int(round(20 / fps_target)))

    # Downsample frames by taking every 'step' frame
    frames = frames[::step]

    # Downsample actions with movement booleans max-pooled and camera deltas averaged
    num_groups = (len(actions) + step - 1) // step
    downsampled_actions = []
    for i in range(num_groups):
        group = actions[i*step:(i+1)*step]
        if len(group) == 0:
            continue
        # Movement booleans: indices 0 to 4
        movement = np.max(group[:, 0:5], axis=0)
        # Camera deltas: indices 5 and 6
        camera = np.mean(group[:, 5:7], axis=0)
        combined = np.concatenate([movement, camera])
        downsampled_actions.append(combined)
    actions = np.array(downsampled_actions, dtype=np.float32)
    dropped = T - len(frames)
    print(f"Dropped {dropped} frames during downsampling for trajectory {traj_dir.name}")
    T = len(frames)

    # Ensure frames and actions are aligned before saving
    T = min(len(frames), len(actions))
    frames, actions = frames[:T], actions[:T]

    # ---- Resize frames ----
    resized = np.zeros((T, size, size, 3), dtype=np.uint8)
    for i, img in enumerate(frames):
        im = Image.fromarray(img)
        im = im.resize((size, size), Image.BICUBIC)
        resized[i] = np.asarray(im, dtype=np.uint8)

    # ---- Save full trajectory ----
    out_name = f"{traj_dir.name}.npz"
    np.savez_compressed(out_dir / out_name,
                        video=resized,
                        actions=actions)
    return 1

def main(parent_dir: str,
         out_dir: str,
         fps_target: int = 8,
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
    p.add_argument("--size", type=int, default=256,
                   help="Output frame size (default 256)")
    args = p.parse_args()
    main(**vars(args))