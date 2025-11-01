import os
import cv2
import argparse
from tqdm import tqdm

def extract_frames(input_dir, output_dir, frame_skip):
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))

    os.makedirs(output_dir, exist_ok=True)

    total_videos = 0
    total_frames = 0

    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}, skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps != 20:
            # Not the expected fps, but continue anyway
            pass

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Skip first second (20 frames)
        skip_frames = 20
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
            filename = f"{video_name}_{saved_frames}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_frames += 1
            frame_idx += 1

        cap.release()
        total_videos += 1
        total_frames += saved_frames

    print(f"Processed {total_videos} videos.")
    print(f"Saved {total_frames} frames.")

def main():
    parser = argparse.ArgumentParser(description="Preprocess MineRL videos into PNG frames for VQ-VAE training.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing MineRL videos.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save PNG frames.')
    parser.add_argument('--frame_skip', type=int, default=1, help='Keep every Nth frame (default: 1, keep all).')

    args = parser.parse_args()

    extract_frames(args.input_dir, args.output_dir, args.frame_skip)

if __name__ == "__main__":
    main()
