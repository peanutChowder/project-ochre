import argparse
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image (e.g., hallway.jpg)")
    parser.add_argument("--out", default="walk.mp4")
    parser.add_argument("--frames", type=int, default=25)       # SVD-XT is trained for 25 frames
    parser.add_argument("--fps", type=int, default=7)           # 6–12 looks natural
    parser.add_argument("--motion", type=int, default=180)      # 0–255, higher ⇒ more motion
    parser.add_argument("--noise_aug", type=float, default=0.10)# 0.0–0.5, higher ⇒ more change
    parser.add_argument("--decode_chunk", type=int, default=2)  # smaller = lower peak memory
    args = parser.parse_args()

    # --- Apple-silicon device setup -----------------------------------------
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Apple GPU) not available. Update PyTorch to 2.0+ and run on macOS 12.3+.")
    device = "mps"
    dtype = torch.float16  # half precision works well on MPS

    # --- Load the SVD-XT pipeline ------------------------------------------
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=dtype,
        variant="fp16",
    ).to(device)

    # Reduce memory spikes on MPS
    pipe.enable_attention_slicing("max")
    pipe.unet.enable_forward_chunking()

    # --- Prepare input image -----------------------------------------------
    image = load_image(args.image).convert("RGB")
    image = image.resize((1024, 576))  # model expects 1024×576 conditioning

    # --- Generate video -----------------------------------------------------
    generator = torch.manual_seed(42)
    result = pipe(
        image,
        num_frames=args.frames,
        decode_chunk_size=args.decode_chunk,
        generator=generator,
        motion_bucket_id=args.motion,
        noise_aug_strength=args.noise_aug,
        fps=args.fps,
    )

    frames = result.frames[0]
    export_to_video(frames, args.out, fps=args.fps)
    print(f"Saved {len(frames)} frames @ {args.fps} fps → {args.out}")


if __name__ == "__main__":
    main()