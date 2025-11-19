import os
import cv2
import argparse
import random
from tqdm import tqdm

def extract_frames(input_dir, output_dir, frame_skip, max_gb=None):
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    # Shuffle video order so early/late files don't dominate when using a size cap.
    random.shuffle(video_files)

    os.makedirs(output_dir, exist_ok=True)

    total_videos = 0
    total_frames = 0
    total_bytes = 0
    max_bytes = None
    if max_gb is not None:
        max_bytes = max_gb * (1024 ** 3)

    for video_path in tqdm(video_files, desc="Processing videos"):
        parent_name = os.path.basename(os.path.dirname(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_prefix = f"{parent_name}_{video_name}"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}, skipping.")
            continue

        # We no longer skip an initial second; process from the first frame.
        skip_frames = 0
        current_frame = 0
        saved_frames = 0

        while current_frame < skip_frames:
            ret = cap.grab()
            if not ret:
                break
            current_frame += 1

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_skip > 1 and (frame_idx % frame_skip) != 0:
                frame_idx += 1
                continue

            # Save frame as PNG
            filename = f"{safe_prefix}_{saved_frames}.png"
            filepath = os.path.join(output_dir, filename)
            success = cv2.imwrite(filepath, frame)
            if success:
                saved_frames += 1
                if max_bytes is not None:
                    try:
                        total_bytes += os.path.getsize(filepath)
                    except OSError:
                        pass
                    if total_bytes >= max_bytes:
                        print(f"Reached max size budget ({max_gb} GB). Stopping.")
                        cap.release()
                        print(f"Processed {total_videos + 1} videos.")
                        print(f"Saved {total_frames + saved_frames} frames.")
                        return
            else:
                print(f"⚠️ Failed to save {filepath}")
            frame_idx += 1

        cap.release()
        total_videos += 1
        total_frames += saved_frames

    print(f"Processed {total_videos} videos.")
    print(f"Saved {total_frames} frames.")
    if max_bytes is not None:
        print(f"Approx total size: {total_bytes / (1024 ** 3):.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Preprocess GameFactory/MineRL videos into PNG frames for VQ-VAE training.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing videos.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save PNG frames.')
    parser.add_argument('--frame_skip', type=int, default=1, help='Keep every Nth frame (default: 1, keep all).')
    parser.add_argument('--max_gb', type=float, default=None, help='Approximate max size of saved images in GB (stop when reached).')

    args = parser.parse_args()

    extract_frames(args.input_dir, args.output_dir, args.frame_skip, max_gb=args.max_gb)

if __name__ == "__main__":
    main()
