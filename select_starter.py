#!/usr/bin/env python3
"""
sample_starter_image.py
Randomly selects a MineRL subfolder, grabs a random frame from its recording.mp4,
and saves it as starter.png (or a path you specify).
"""

import argparse, random
from pathlib import Path
import cv2   # pip install opencv-python

def main(parent_dir, out_file):
    parent = Path(parent_dir)
    # all subfolders that contain recording.mp4
    candidates = [d for d in parent.iterdir() if (d / "recording.mp4").exists()]
    if not candidates:
        raise SystemExit("No valid subfolders with recording.mp4 found")

    # pick a random trajectory and open its video
    traj = random.choice(candidates)
    vid_path = traj / "recording.mp4"
    print(f"Sampling from: {vid_path}")

    cap = cv2.VideoCapture(str(vid_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_idx = random.randint(0, frame_count - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, rand_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Failed to read frame")

    # OpenCV loads BGR; convert to RGB for a nicer PNG
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(out_file), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved random starter frame to {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent_dir", required=True,
                   help="Path to MineRLNavigate-v0 parent folder")
    p.add_argument("--out", default="starter.png",
                   help="Output image path")
    args = p.parse_args()
    main(args.parent_dir, args.out)