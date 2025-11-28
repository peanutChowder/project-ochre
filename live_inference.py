#!/usr/bin/env python3
"""
Live interactive inference for the Ochre world model (VQ‑VAE + WorldModelConvFiLM).

Controls:
  - W/A/S/D: movement (binary)
  - Space: jump (binary)
  - Mouse move: yaw (dx) and pitch (dy) → normalized to [-1, 1]
  - Arrow keys: look yaw/pitch 
  - Esc or Q: quit
"""

import argparse, os, time, numpy as np, torch, pygame
from vq_vae.vq_vae import VQVAE, IMAGE_HEIGHT, IMAGE_WIDTH
from model_convGru import WorldModelConvFiLM

# ---------------------- Constants ----------------------
FRAME_H = IMAGE_HEIGHT  # 72
FRAME_W = IMAGE_WIDTH   # 128
LATENT_H = 18
LATENT_W = 32
CTX_LEN = 6  # rolling token/action window
ACTION_DIM = 5 # [yaw, pitch, move_x, move_z, jump]


def load_vqvae(vqvae_ckpt: str, device: torch.device) -> VQVAE:
    ckpt = torch.load(vqvae_ckpt, map_location=device)
    vqvae = VQVAE(
        embedding_dim=ckpt["config"]["embedding_dim"],
        num_embeddings=ckpt["config"]["codebook_size"],
        commitment_cost=ckpt["config"]["beta"],
        decay=ckpt["config"]["ema_decay"],
    ).to(device)
    vqvae.encoder.load_state_dict(ckpt["encoder"])  # not used here, but kept for completeness
    vqvae.decoder.load_state_dict(ckpt["decoder"])
    vqvae.vq_vae.load_state_dict(ckpt["quantizer"])  # codebook
    vqvae.eval()
    return vqvae


def try_load_state_dict(model: torch.nn.Module, path: str, device: torch.device):
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "model_state" in data:
        model.load_state_dict(data["model_state"])
    else:
        model.load_state_dict(data)
    print(f"✅ Loaded world model from {path}")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="WorldModelConvFiLM checkpoint (.pt)")
    p.add_argument("--vqvae_ckpt", required=True, help="Trained VQ‑VAE checkpoint (.pt)")
    p.add_argument("--context_npz", help="Optional preprocessed .npz providing initial tokens/actions")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--no_mouse", action="store_true", help="Disable mouse look; use arrow keys instead")
    p.add_argument("--mouse_sens", type=float, default=60.0, help="Divisor for mouse dx/dy → [-1,1]")
    p.add_argument("--key_look_gain", type=float, default=0.5, help="Arrow key look gain added to yaw/pitch")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Live inference on {device} @ {args.fps} FPS")
    print(f"Resolution: {FRAME_W}x{FRAME_H} (Latent: {LATENT_W}x{LATENT_H})")

    # --- Load models ---
    vqvae = load_vqvae(args.vqvae_ckpt, device)
    # Instantiate with correct dimensions from training
    model = WorldModelConvFiLM(
        action_dim=ACTION_DIM, 
        H=LATENT_H, 
        W=LATENT_W
    ).to(device)
    try_load_state_dict(model, args.checkpoint, device)
    model.eval()

    # --- Initialize rolling context (tokens/actions) ---
    if args.context_npz and os.path.exists(args.context_npz):
        data = np.load(args.context_npz)
        tokens = torch.tensor(data["tokens"], dtype=torch.long, device=device)  # (T,H,W)
        actions = torch.tensor(data["actions"], dtype=torch.float32, device=device)  # (T,A)
        
        # Adjust loaded action dimension if necessary
        if actions.shape[1] < ACTION_DIM:
            pad_size = ACTION_DIM - actions.shape[1]
            actions = torch.cat([actions, torch.zeros(actions.shape[0], pad_size, device=device)], dim=1)
        elif actions.shape[1] > ACTION_DIM:
            actions = actions[:, :ACTION_DIM]

        if tokens.size(0) >= CTX_LEN:
            Z_seq = tokens[:CTX_LEN].unsqueeze(0)  # (1,K,H,W)
        else:
            pad = CTX_LEN - tokens.size(0)
            Z_seq = torch.cat([tokens, torch.randint(0, 2048, (pad, LATENT_H, LATENT_W), device=device)], dim=0).unsqueeze(0)
        if actions.size(0) >= CTX_LEN:
            A_seq = actions[:CTX_LEN].unsqueeze(0)  # (1,K,A)
        else:
            pad = CTX_LEN - actions.size(0)
            A_seq = torch.cat([actions, torch.zeros(pad, ACTION_DIM, device=device)], dim=0).unsqueeze(0)
        print(f"Seeded from {args.context_npz}")
    else:
        Z_seq = torch.randint(0, 2048, (1, CTX_LEN, LATENT_H, LATENT_W), device=device)
        A_seq = torch.zeros((1, CTX_LEN, ACTION_DIM), device=device)
        print("⚠️ No context provided — starting from random tokens")

    # --- Pygame setup ---
    pygame.init()
    width, height = FRAME_W * args.scale, FRAME_H * args.scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Project Ochre — Live Inference")
    clock = pygame.time.Clock()
    pygame.font.init()
    hud_font = pygame.font.SysFont("Arial", max(10, int(10 * args.scale / 5)))

    if not args.no_mouse:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()  # reset relative motion accumulator

    def get_action_vec():
        keys = pygame.key.get_pressed()
        move_x = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        move_z = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        jump   = float(keys[pygame.K_SPACE])
        
        # Arrow keys contribute to look regardless of mouse mode
        key_yaw = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        key_pitch = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])

        if args.no_mouse:
            # Arrow-only look
            yaw = clamp(key_yaw * args.key_look_gain, -1.0, 1.0)
            pitch = clamp(key_pitch * args.key_look_gain, -1.0, 1.0)
        else:
            # Combine mouse and arrow key look
            dx, dy = pygame.mouse.get_rel()
            yaw = clamp((dx / args.mouse_sens) + (key_yaw * args.key_look_gain), -1.0, 1.0)
            pitch = clamp((-dy / args.mouse_sens) + (key_pitch * args.key_look_gain), -1.0, 1.0)
            
        # Action vector: [yaw, pitch, move_x, move_z, jump]
        return [yaw, pitch, move_x, move_z, jump]

    print("🎮 W/A/S/D move • Space jump • Mouse/Arrows look • ESC/Q quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        act = get_action_vec()
        A_seq = torch.cat([A_seq[:, 1:], torch.tensor([[act]], device=device)], dim=1)  # (1,K,A)

        with torch.no_grad():
            logits = model(Z_seq, A_seq)               # (B,2048,H,W)
            pred_tokens = logits.argmax(dim=1)         # (B,H,W)
            frame = vqvae.decode_code(pred_tokens)[0]  # (3,H_img,W_img)
            img = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # update token window
        Z_seq = torch.cat([Z_seq[:, 1:], pred_tokens.unsqueeze(1)], dim=1)

        # draw
        surf = pygame.image.frombuffer(img.tobytes(), (FRAME_W, FRAME_H), "RGB")
        if args.scale != 1:
            surf = pygame.transform.smoothscale(surf, (width, height))
        screen.blit(surf, (0, 0))
        
        hud = f"Y:{act[0]:+.2f} P:{act[1]:+.2f} X:{act[2]:.0f} Z:{act[3]:.0f} J:{act[4]:.0f}"
        screen.blit(hud_font.render(hud, True, (255, 255, 255)), (10, 10))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print("✅ Exited live session")


if __name__ == "__main__":
    main()
