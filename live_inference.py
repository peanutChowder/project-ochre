#!/usr/bin/env python3
"""
Live interactive inference for 1→1 temporal transformer (single-frame model).

Controls (default):
  - W/A/S/D: forward/left/back/right (binary 0/1)
  - Space: jump (binary 0/1)
  - Mouse move: yaw (dx) and pitch (dy) scaled to [-1, 1]
  - Arrow keys (fallback if --no-mouse): Left/Right -> yaw, Up/Down -> pitch
  - Esc or Q: quit
"""

import argparse, sys, time
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
print("Imported AutoencoderKL")
from model_temporal import TemporalTransformer
print("Imported TemporalTransformer")


# ---------------------- Config ----------------------
LATENT_C  = 4
LATENT_H  = LATENT_W = 32
FRAME_SIZE = 256  # model’s training size

def try_load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    data = torch.load(checkpoint_path, map_location=device)
    if isinstance(data, dict) and "model_state" in data:
        model.load_state_dict(data["model_state"])
    else:
        model.load_state_dict(data)

    print("Loaded model")

def build_initial_context_from_image(img_path: str, device, dtype) -> torch.Tensor:
    """Load an RGB image, resize, normalize to [0,1], return [1,3,H,W]."""
    init_img = Image.open(img_path).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
    arr = np.array(init_img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device=device, dtype=dtype)

def clamp01(x): return max(0.0, min(1.0, float(x)))
def clamp11(x): return max(-1.0, min(1.0, float(x)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--start_image", required=True, help="Seed image to bootstrap initial frame")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--no-mouse", action="store_true")
    p.add_argument("--mouse-sens", type=float, default=0.25)
    p.add_argument("--yaw-max-deg", type=float, default=60.0)
    p.add_argument("--pitch-max-deg", type=float, default=60.0)
    p.add_argument("--look-gain", type=float, default=1.0)
    p.add_argument("--autowalk", action="store_true")
    p.add_argument("--latent-noise", type=float, default=0.0)
    p.add_argument("--reencode-context", action="store_true")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32
    print(f"Live inference on {device}, target {args.fps} FPS")

    # --- Load models ---
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype).to(device)
    vae.eval()

    model = TemporalTransformer(latent_channels=LATENT_C, H=LATENT_H, W=LATENT_W).to(device)
    try_load_state_dict(model, args.checkpoint, device)
    model.eval()

    # --- Prepare initial frame ---
    init_img = build_initial_context_from_image(args.start_image, device, dtype)
    with torch.no_grad():
        curr_lat = vae.encode(init_img).latent_dist.sample()  # [1,4,32,32]

    # --- Setup display/input ---
    import pygame
    pygame.init()
    width, height = FRAME_SIZE * args.scale, FRAME_SIZE * args.scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Live Model Inference")
    clock = pygame.time.Clock()
    pygame.font.init()
    hud_font = pygame.font.SysFont("Arial", max(10, int(12 * args.scale * 0.6)))
    if not args.no_mouse:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

    running = True

    def get_keyboard_action():
        keys = pygame.key.get_pressed()
        fwd = 1.0 if (args.autowalk or keys[pygame.K_w]) else 0.0
        left = 1.0 if keys[pygame.K_a] else 0.0
        back = 1.0 if keys[pygame.K_s] else 0.0
        right = 1.0 if keys[pygame.K_d] else 0.0
        jump = 1.0 if keys[pygame.K_SPACE] else 0.0
        return fwd, left, back, right, jump

    def get_look_action():
        if args.no_mouse:
            keys = pygame.key.get_pressed()
            yaw = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.25
            pitch = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 0.25
            return clamp11(yaw), clamp11(pitch)
        else:
            dx, dy = pygame.mouse.get_rel()
            yaw_deg = np.clip(dx * args.mouse_sens, -args.yaw_max_deg, args.yaw_max_deg)
            pitch_deg = np.clip(-dy * args.mouse_sens, -args.pitch_max_deg, args.pitch_max_deg)
            yaw = clamp11((yaw_deg / 180.0) * args.look_gain)
            pitch = clamp11((pitch_deg / 180.0) * args.look_gain)
            return yaw, pitch

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q): running = False
                elif event.key == pygame.K_n: args.latent_noise = 0.25 if args.latent_noise == 0 else 0.0
                elif event.key == pygame.K_r: args.reencode_context = not args.reencode_context

        fwd, left, back, right, jump = get_keyboard_action()
        yaw, pitch = get_look_action()
        action_vec = torch.tensor([[[fwd, left, back, right, jump, yaw, pitch]]],
                                  device=device, dtype=dtype)

        with torch.no_grad():
            pred_lat = model(curr_lat.unsqueeze(1), action_vec)
            if args.latent_noise > 0:
                pred_lat += torch.randn_like(pred_lat) * args.latent_noise
            img = vae.decode(pred_lat).sample
            img = ((img.clamp(-1,1) + 1)/2).cpu()
            frame = (img[0].permute(1,2,0).numpy() * 255).astype(np.uint8)

        if args.reencode_context:
            with torch.no_grad():
                fr_t = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
                curr_lat = vae.encode(fr_t).latent_dist.sample()
        else:
            curr_lat = pred_lat.squeeze(1)

        surf = pygame.image.frombuffer(frame.tobytes(), (FRAME_SIZE, FRAME_SIZE), "RGB")
        if args.scale != 1:
            surf = pygame.transform.smoothscale(surf, (width, height))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()

if __name__ == "__main__":
    main()