#!/usr/bin/env python3
"""
Live interactive inference on Apple Silicon (MPS) or CPU.

Controls (default):
  - W/A/S/D: forward/left/back/right (binary 0/1)
  - Space: jump (binary 0/1)
  - Mouse move: yaw (dx) and pitch (dy) scaled to [-1, 1]
  - Arrow keys (fallback if --no-mouse): Left/Right -> yaw, Up/Down -> pitch
  - Esc or Q: quit

Action format matches preprocessing/build_action_matrix:
  [forward, left, back, right, jump, yaw, pitch] where yaw/pitch in [-1, 1].

Requires: torch, diffusers, numpy, pillow, pygame
"""

import argparse
import sys
import time
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL

from model_temporal import TemporalTransformer
from minerl_dataset import MineRLSequenceDataset


# ---------------------- Config ----------------------
SEQ_LEN   = 7          # K + 1, where K = 6 context frames
K         = SEQ_LEN-1
LATENT_C  = 4
LATENT_H  = LATENT_W = 32
FRAME_SIZE = 256       # model's training size


def try_load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """Load a checkpoint that may be either: full state_dict or dict with 'model_state'."""
    data = torch.load(checkpoint_path, map_location=device)
    if isinstance(data, dict) and "model_state" in data:
        model.load_state_dict(data["model_state"])
    else:
        model.load_state_dict(data)


def build_initial_context_from_image(img_path: str, device, dtype) -> torch.Tensor:
    """Load an RGB image, resize to FRAME_SIZE, normalize to [0,1], repeat K times -> [K,3,H,W]."""
    init_img = Image.open(img_path).convert("RGB").resize((FRAME_SIZE, FRAME_SIZE))
    arr = np.array(init_img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0).repeat(K, 1, 1, 1)  # [K,3,H,W]
    return t.to(device=device, dtype=dtype)


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def clamp11(x):
    return max(-1.0, min(1.0, float(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    # Context options
    p.add_argument("--start_image", help="Seed image (jpg/png) to bootstrap context if no dataset context is given")
    p.add_argument("--context_npz", help="Path to preprocessed sequence (.npz) to use as initial context")
    p.add_argument("--data_dir", help="Directory of preprocessed sequences to sample context from")
    p.add_argument("--context_index", type=int, help="Specific dataset index to use when sampling context")
    p.add_argument("--min-mean-action", type=float, default=0.02,
                   help="Require mean(|action|) over K context steps >= threshold when sampling context")
    p.add_argument("--context-sample-attempts", type=int, default=30,
                   help="When sampling from --data_dir without --context_index, try up to N samples to satisfy motion filter")
    p.add_argument("--fps", type=int, default=8, help="Target interactive FPS")
    p.add_argument("--scale", type=int, default=2, help="Window scale factor (display size = FRAME_SIZE*scale)")
    p.add_argument("--no-mouse", action="store_true", help="Disable mouse look; use arrow keys for yaw/pitch")
    p.add_argument("--mouse-sens", type=float, default=0.25, help="Degrees-per-pixel for yaw/pitch mapping")
    p.add_argument("--yaw-max-deg", type=float, default=60.0, help="Clamp yaw magnitude per frame in degrees")
    p.add_argument("--pitch-max-deg", type=float, default=60.0, help="Clamp pitch magnitude per frame in degrees")
    p.add_argument("--look-gain", type=float, default=1.0, help="Multiply yaw/pitch after normalization (deg/180 * gain)")
    p.add_argument("--autowalk", action="store_true", help="Hold forward=1.0 by default (useful to force motion)")
    p.add_argument("--latent-noise", type=float, default=0.0, help="Add Gaussian noise N(0, std) to predicted latent each step")
    p.add_argument("--reencode-context", action="store_true", help="Append context from VAE re-encode of displayed frame instead of raw predicted latent")
    p.add_argument("--oscillate-yaw", action="store_true", help="Override yaw with a sinusoid to probe action conditioning")
    p.add_argument("--osc-freq", type=float, default=0.5, help="Oscillation frequency in Hz for yaw override")
    p.add_argument("--osc-amp", type=float, default=1.0, help="Oscillation amplitude for yaw override (0..1)")
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

    # --- Prepare initial context (K frames) and actions window ---
    if args.context_npz and args.data_dir:
        print("Specify only one of --context_npz or --data_dir")
        sys.exit(2)
    if args.context_index is not None and not args.data_dir:
        print("--context_index requires --data_dir")
        sys.exit(2)

    context_pixels = None
    actions_win = None

    def context_metric(a: torch.Tensor) -> float:
        # a: [K,7] -> return mean absolute value over all entries
        return float(a.abs().mean().item())

    if args.context_npz:
        d = np.load(args.context_npz, allow_pickle=False)
        video = d["video"]  # [7,H,W,3] uint8
        acts = d["actions"]  # [6,7] float32
        if video.shape[0] < K:
            print(f"Context file must contain at least {K} frames")
            sys.exit(2)
        context_pixels = torch.from_numpy(video[:K]).permute(0, 3, 1, 2).float() / 255.0
        actions_win = torch.from_numpy(acts[:K]).float().unsqueeze(0)
        metric = context_metric(actions_win[0])
        print(f"Seeded context from {args.context_npz} | mean|action|={metric:.4f}")
    elif args.data_dir:
        ds = MineRLSequenceDataset(args.data_dir)
        if args.context_index is not None:
            if not (0 <= args.context_index < len(ds)):
                print("--context_index out of range for provided --data_dir")
                sys.exit(2)
            idx = args.context_index
            frames, acts = ds[idx]
            context_pixels = frames[:K]
            actions_win = acts[:K].unsqueeze(0)
            metric = context_metric(actions_win[0])
            print(f"Using context index {idx} from {args.data_dir} | mean|action|={metric:.4f}")
        else:
            picked = False
            metric = 0.0
            idx = -1
            for attempt in range(max(1, args.context_sample_attempts)):
                cand = torch.randint(len(ds), (1,), dtype=torch.long).item()
                frames, acts = ds[cand]
                m = context_metric(acts[:K])
                if m >= args.min_mean_action:
                    idx = cand
                    context_pixels = frames[:K]
                    actions_win = acts[:K].unsqueeze(0)
                    metric = m
                    picked = True
                    break
            if not picked:
                # fallback to last candidate
                if idx == -1:
                    idx = 0
                    frames, acts = ds[idx]
                    metric = context_metric(acts[:K])
                context_pixels = frames[:K]
                actions_win = acts[:K].unsqueeze(0)
            print(f"Sampled context idx {idx} from {args.data_dir} | mean|action|={metric:.4f} (threshold {args.min_mean_action})")
    else:
        if not args.start_image:
            print("Provide --start_image when no dataset context is supplied")
            sys.exit(2)
        context_pixels = build_initial_context_from_image(args.start_image, device, dtype)
        actions_win = torch.zeros((1, K, 7), dtype=dtype)

    context_pixels = context_pixels.to(device=device, dtype=dtype)
    actions_win = actions_win.to(device=device, dtype=dtype)

    with torch.no_grad():
        ctx_lat = vae.encode(context_pixels).latent_dist.sample()  # [K,4,32,32]
        latents_win = ctx_lat.unsqueeze(0)  # [1,K,C,H,W]

    # --- Setup display and input ---
    try:
        import pygame
    except Exception as e:
        print("pygame is required for the live demo. Install with: pip install pygame")
        print(f"Import error: {e}")
        sys.exit(1)

    pygame.init()
    width, height = FRAME_SIZE * args.scale, FRAME_SIZE * args.scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Live Model Inference")
    clock = pygame.time.Clock()
    pygame.font.init()
    hud_scale = max(0.6, args.scale * 0.6)
    hud_font = pygame.font.SysFont("Arial", max(10, int(12 * hud_scale)))

    if not args.no_mouse:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()  # reset relative motion

    running = True
    last_frame = None

    def get_keyboard_action() -> Tuple[float, float, float, float, float]:
        keys = pygame.key.get_pressed()
        fwd = 1.0 if (args.autowalk or keys[pygame.K_w]) else 0.0
        left = 1.0 if keys[pygame.K_a] else 0.0
        back = 1.0 if keys[pygame.K_s] else 0.0
        right = 1.0 if keys[pygame.K_d] else 0.0
        jump = 1.0 if keys[pygame.K_SPACE] else 0.0
        return fwd, left, back, right, jump

    def get_look_action() -> Tuple[float, float]:
        # Returns (yaw, pitch) in [-1, 1]
        if args.no_mouse:
            keys = pygame.key.get_pressed()
            yaw = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_LEFT] else 0.0)
            pitch = (1.0 if keys[pygame.K_UP] else 0.0) - (1.0 if keys[pygame.K_DOWN] else 0.0)
            # Simple rate; tune if desired
            yaw *= 0.25
            pitch *= 0.25
            return clamp11(yaw), clamp11(pitch)
        else:
            dx, dy = pygame.mouse.get_rel()  # pixels since last call
            # Convert pixels -> degrees then normalize by 180 to match preprocessing
            yaw_deg = float(np.clip(dx * args.mouse_sens, -args.yaw_max_deg, args.yaw_max_deg))
            pitch_deg = float(np.clip(-dy * args.mouse_sens, -args.pitch_max_deg, args.pitch_max_deg))
            yaw = clamp11((yaw_deg / 180.0) * args.look_gain)
            pitch = clamp11((pitch_deg / 180.0) * args.look_gain)
            return yaw, pitch

    def draw_hud(surface, fwd, left, back, right, jump, yaw, pitch, fps, latent_delta):
        # Semi-transparent background box
        pad = max(4, int(6 * hud_scale))
        ksz = max(12, int(16 * hud_scale))
        gap = max(2, int(3 * hud_scale))
        base_x, base_y = pad, pad
        # Positions relative to overlay
        pos_W = (base_x + ksz + gap, base_y)
        pos_A = (base_x, base_y + ksz + gap)
        pos_S = (base_x + ksz + gap, base_y + ksz + gap)
        pos_D = (base_x + 2 * (ksz + gap), base_y + ksz + gap)
        pos_SPC = (base_x, base_y + 2 * (ksz + gap))

        info = (
            f"yaw: {yaw:+.2f}  pitch: {pitch:+.2f}  fps: {fps:.1f}  dLat: {latent_delta:.4f} "
            f"gain: {args.look_gain:.2f}  autowalk: {args.autowalk}  noise: {args.latent_noise:.3f} "
            f"reenc: {args.reencode_context}  oscYaw: {args.oscillate_yaw}"
        )
        info_txt = hud_font.render(info, True, (230, 230, 230))

        sp_x, sp_y = pos_SPC
        sp_w, sp_h = ksz * 3 + gap * 2, ksz
        key_block_w = base_x + sp_w + pad
        text_block_w = base_x + info_txt.get_width() + pad
        box_w = max(140, int(160 * hud_scale), key_block_w, text_block_w)

        key_block_h = base_y + 2 * (ksz + gap) + ksz + pad
        text_y = sp_y + sp_h + gap
        text_block_h = text_y + info_txt.get_height() + pad
        box_h = max(85, int(95 * hud_scale), key_block_h, text_block_h)

        overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))

        def draw_key(pos, label, active):
            x, y = pos
            color = (70, 160, 255) if active else (180, 180, 180)
            border_radius = max(2, int(3 * hud_scale))
            pygame.draw.rect(overlay, color, (x, y, ksz, ksz), border_radius=border_radius)
            pygame.draw.rect(overlay, (20, 20, 20), (x, y, ksz, ksz), width=2, border_radius=border_radius)
            txt = hud_font.render(label, True, (0, 0, 0))
            tx = x + (ksz - txt.get_width()) // 2
            ty = y + (ksz - txt.get_height()) // 2
            overlay.blit(txt, (tx, ty))

        draw_key(pos_W, "W", fwd >= 0.5)
        draw_key(pos_A, "A", left >= 0.5)
        draw_key(pos_S, "S", back >= 0.5)
        draw_key(pos_D, "D", right >= 0.5)
        # Space bar as a wider box
        sp_color = (70, 160, 255) if jump >= 0.5 else (180, 180, 180)
        border_radius = max(2, int(3 * hud_scale))
        pygame.draw.rect(overlay, sp_color, (sp_x, sp_y, sp_w, sp_h), border_radius=border_radius)
        pygame.draw.rect(overlay, (20, 20, 20), (sp_x, sp_y, sp_w, sp_h), width=2, border_radius=border_radius)
        sp_txt = hud_font.render("SPACE", True, (0, 0, 0))
        overlay.blit(sp_txt, (sp_x + (sp_w - sp_txt.get_width()) // 2, sp_y + (sp_h - sp_txt.get_height()) // 2))

        # Yaw/Pitch and FPS text
        overlay.blit(info_txt, (base_x, text_y))

        surface.blit(overlay, (pad, pad))

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_f:
                        args.autowalk = not args.autowalk
                    elif event.key == pygame.K_g:
                        args.look_gain = min(args.look_gain * 1.25, 8.0)
                    elif event.key == pygame.K_h:
                        args.look_gain = max(args.look_gain / 1.25, 0.1)
                    elif event.key == pygame.K_n:
                        # toggle noise between 0 and a moderate value
                        args.latent_noise = 0.25 if args.latent_noise == 0 else 0.0
                    elif event.key == pygame.K_r:
                        args.reencode_context = not args.reencode_context
                    elif event.key == pygame.K_o:
                        args.oscillate_yaw = not args.oscillate_yaw

            # Build current action from input
            fwd, left, back, right, jump = get_keyboard_action()
            yaw, pitch = get_look_action()
            # Optional yaw override (sinusoid) to probe responsiveness regardless of mouse input
            if args.oscillate_yaw:
                t = pygame.time.get_ticks() / 1000.0
                yaw = clamp11(args.osc_amp * np.sin(2 * np.pi * args.osc_freq * t))
            action_vec = torch.tensor([[ [fwd, left, back, right, jump, yaw, pitch] ]],
                                      device=device, dtype=dtype)  # [1,1,7]

            # Shift actions window and append
            actions_win = torch.cat([actions_win[:, 1:], action_vec], dim=1)

            # Model step
            with torch.no_grad():
                pred_lat = model(latents_win, actions_win)
                if args.latent_noise > 0:
                    pred_lat = pred_lat + torch.randn_like(pred_lat) * args.latent_noise
                # Latent delta (L2 norm between last context and predicted)
                latent_delta = torch.linalg.vector_norm((pred_lat - latents_win[:, -1]).reshape(1, -1)).item() / 1000.0
                img = vae.decode(pred_lat).sample  # [-1,1]
                img = ((img.clamp(-1, 1) + 1) / 2).cpu()  # [0,1]
                frame = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Update latent window
            if args.reencode_context:
                # Re-encode the displayed frame to obtain a latent from VAE's manifold
                with torch.no_grad():
                    fr_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    enc = vae.encode(fr_t).latent_dist.sample()  # [1,4,32,32]
                latents_win = torch.cat([latents_win[:, 1:], enc.unsqueeze(1)], dim=1)
            else:
                latents_win = torch.cat([latents_win[:, 1:], pred_lat.unsqueeze(1)], dim=1)

            # Display
            surf = pygame.image.frombuffer(frame.tobytes(), (FRAME_SIZE, FRAME_SIZE), "RGB")
            if args.scale != 1:
                surf = pygame.transform.smoothscale(surf, (width, height))
            screen.blit(surf, (0, 0))
            # Draw HUD on top
            draw_hud(screen, fwd, left, back, right, jump, yaw, pitch, clock.get_fps(), latent_delta)
            pygame.display.flip()

            clock.tick(args.fps)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
