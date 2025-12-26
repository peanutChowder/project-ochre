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


def sample_tokens(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """
    Sample discrete VQ‑VAE token indices from logits.
    logits: (B, C, H, W)
    Returns: (B, H, W) long
    """
    B, C, H, W = logits.shape
    # Guard against tiny / negative temperatures.
    temperature = max(float(temperature), 1e-3)
    logits = logits / temperature

    # (B, C, H, W) -> (B, HW, C)
    logits_flat = logits.view(B, C, H * W).permute(0, 2, 1)

    if top_k is not None and top_k > 0 and top_k < C:
        k = int(top_k)
        topk_vals, topk_idx = torch.topk(logits_flat, k, dim=-1)  # (B, HW, k)
        probs = torch.softmax(topk_vals, dim=-1)
        sampled_rel = torch.multinomial(
            probs.view(-1, k), num_samples=1
        ).view(B, -1)  # (B, HW)
        chosen = topk_idx.gather(-1, sampled_rel.unsqueeze(-1)).squeeze(-1)
    else:
        probs = torch.softmax(logits_flat, dim=-1)  # (B, HW, C)
        sampled = torch.multinomial(
            probs.view(-1, C), num_samples=1
        ).view(B, -1)  # (B, HW)
        chosen = sampled

    return chosen.view(B, H, W).long()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="WorldModelConvFiLM checkpoint (.pt)")
    p.add_argument("--vqvae_ckpt", required=True, help="Trained VQ‑VAE checkpoint (.pt)")
    p.add_argument("--context_npz", help="Optional preprocessed .npz providing initial tokens/actions")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--key_look_gain", type=float, default=0.5, help="Arrow key look gain added to yaw/pitch")
    p.add_argument("--greedy", action="store_true", help="Use argmax decoding (no sampling)")
    p.add_argument("--temperature", type=float, default=1.05, help="Sampling temperature (>0); 1.0 ≈ unbiased")
    p.add_argument("--topk", type=int, default=32, help="Top‑k sampling cutoff (0 = disable)")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Live inference on {device} @ {args.fps} FPS")
    print(f"Resolution: {FRAME_W}x{FRAME_H} (Latent: {LATENT_W}x{LATENT_H})")

    # --- Load models ---
    vqvae = load_vqvae(args.vqvae_ckpt, device)
    # Instantiate with correct dimensions from training
    model = WorldModelConvFiLM(
        codebook_size=1024,
        action_dim=ACTION_DIM, 
        H=LATENT_H, 
        W=LATENT_W
    ).to(device)
    try_load_state_dict(model, args.checkpoint, device)
    model.eval()

    # --- Initialize state ---
    h_state = model.init_state(1, device=device)
    
    # Warmup if context provided
    if args.context_npz and os.path.exists(args.context_npz):
        print(f"Seeding from {args.context_npz}...")
        data = np.load(args.context_npz)
        # Load context tokens and actions
        # We need to run the model through these to build up h_state
        
        # Take up to last 60 frames or so to build state, don't need infinite history for warmup
        # But actually, we just need to run through them.
        tokens_ctx = torch.tensor(data["tokens"], dtype=torch.long, device=device)
        actions_ctx = torch.tensor(data["actions"], dtype=torch.float32, device=device)
        
        # Adjust dimensions
        if actions_ctx.shape[1] < ACTION_DIM:
             pad = torch.zeros(actions_ctx.shape[0], ACTION_DIM - actions_ctx.shape[1], device=device)
             actions_ctx = torch.cat([actions_ctx, pad], dim=1)
        
        # Run through context to prime h_state
        # We can use the efficient forward unroll since we just want the final state
        # Z_seq needs to be (1, K, H, W), A_seq (1, K, A)
        
        # Limit to reasonable warmup length (e.g. 30 frames) to avoid wait
        warmup_len = min(60, tokens_ctx.shape[0], actions_ctx.shape[0])
        start_idx = 0 # or -warmup_len
        
        z_warmup = tokens_ctx[start_idx : start_idx+warmup_len].unsqueeze(0)
        a_warmup = actions_ctx[start_idx : start_idx+warmup_len].unsqueeze(0)
        
        with torch.no_grad():
            # model.forward returns logits, but we need the hidden state.
            # model.forward unrolls but discards intermediate states unless we change it.
            # Actually, model.forward calls model.init_state internally.
            # We need to modify how we get the state out, or just loop step() manually.
            # Looping step manually is safer to ensure we have the final h_state.
            
            print(f"Priming hidden state with {warmup_len} frames...")
            for i in range(warmup_len):
                z_t = z_warmup[:, i] # (1, H, W)
                a_t = a_warmup[:, i] # (1, A)
                _, h_state = model.step(z_t, a_t, h_state)
        
        # Set current token to the last one from context
        current_z = z_warmup[:, -1]
        print("State primed.")

    else:
        print("⚠️ No context provided — starting from random state")
        current_z = torch.randint(0, 2048, (1, LATENT_H, LATENT_W), device=device)
        # h_state is already zeros

    # --- Pygame setup ---
    pygame.init()
    width, height = FRAME_W * args.scale, FRAME_H * args.scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Project Ochre — Live Inference (Stateful)")
    clock = pygame.time.Clock()
    pygame.font.init()
    hud_font = pygame.font.SysFont("Arial", max(10, int(10 * args.scale / 5)))
    button_font = pygame.font.SysFont("Arial", max(14, int(14 * args.scale / 5)), bold=True)

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
        key_pitch = float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])

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

    def draw_key_button(screen, x, y, w, h, label, is_pressed):
        """Draw a button-like visualization for a key."""
        # Colors
        bg_color = (100, 200, 100) if is_pressed else (60, 60, 60)
        border_color = (150, 255, 150) if is_pressed else (100, 100, 100)
        text_color = (255, 255, 255)

        # Draw button background
        pygame.draw.rect(screen, bg_color, (x, y, w, h))
        # Draw border
        pygame.draw.rect(screen, border_color, (x, y, w, h), 2)

        # Draw label
        text_surf = button_font.render(label, True, text_color)
        text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
        screen.blit(text_surf, text_rect)

    def draw_arrow_indicator(screen, x, y, direction, is_active):
        """Draw an arrow indicator on screen edge."""
        # direction: 'up', 'down', 'left', 'right'
        size = max(20, int(20 * args.scale / 5))
        color = (255, 200, 50) if is_active else (80, 80, 80)

        # Define arrow points based on direction
        if direction == 'up':
            points = [(x, y - size), (x - size//2, y), (x + size//2, y)]
        elif direction == 'down':
            points = [(x, y + size), (x - size//2, y), (x + size//2, y)]
        elif direction == 'left':
            points = [(x - size, y), (x, y - size//2), (x, y + size//2)]
        elif direction == 'right':
            points = [(x + size, y), (x, y - size//2), (x, y + size//2)]

        pygame.draw.polygon(screen, color, points)
        # Add outline
        pygame.draw.polygon(screen, (200, 200, 200) if is_active else (120, 120, 120), points, 2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        # 1. Get Action
        act = get_action_vec()
        current_a = torch.tensor([act], dtype=torch.float32, device=device) # (1, 5)

        # 2. Step Model
        with torch.no_grad():
            # z_t: (1, H, W), a_t: (1, A), h_prev: (...)
            logits, h_state = model.step(current_z, current_a, h_state)

            # 3. Choose next tokens
            if args.greedy:
                # Old behavior: pure argmax decoding
                pred_tokens = logits.argmax(dim=1)  # (1, H, W)
            else:
                # Default: stochastic sampling for more diverse scenes
                pred_tokens = sample_tokens(
                    logits, temperature=args.temperature, top_k=args.topk
                )  # (1, H, W)

            # 4. Decode
            frame = vqvae.decode_code(pred_tokens)[0]  # (3,H_img,W_img)
            img = (frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Update state for next loop
        current_z = pred_tokens

        # draw
        surf = pygame.image.frombuffer(img.tobytes(), (FRAME_W, FRAME_H), "RGB")
        if args.scale != 1:
            surf = pygame.transform.smoothscale(surf, (width, height))
        screen.blit(surf, (0, 0))

        # Draw HUD
        hud = f"Y:{act[0]:+.2f} P:{act[1]:+.2f} X:{act[2]:.0f} Z:{act[3]:.0f} J:{act[4]:.0f}"
        screen.blit(hud_font.render(hud, True, (255, 255, 255)), (10, 10))

        # --- Draw WASD+Jump key visualizations ---
        keys = pygame.key.get_pressed()
        button_size = max(30, int(30 * args.scale / 5))
        button_spacing = max(5, int(5 * args.scale / 5))

        # Position in bottom-left area
        base_x = 20
        base_y = height - button_size * 3 - button_spacing * 2 - 20

        # W key (top middle)
        draw_key_button(screen, base_x + button_size + button_spacing, base_y,
                       button_size, button_size, "W", keys[pygame.K_w])

        # A key (middle left)
        draw_key_button(screen, base_x, base_y + button_size + button_spacing,
                       button_size, button_size, "A", keys[pygame.K_a])

        # S key (middle middle)
        draw_key_button(screen, base_x + button_size + button_spacing, base_y + button_size + button_spacing,
                       button_size, button_size, "S", keys[pygame.K_s])

        # D key (middle right)
        draw_key_button(screen, base_x + (button_size + button_spacing) * 2, base_y + button_size + button_spacing,
                       button_size, button_size, "D", keys[pygame.K_d])

        # Space key (bottom, wide)
        space_width = button_size * 3 + button_spacing * 2
        draw_key_button(screen, base_x, base_y + (button_size + button_spacing) * 2,
                       space_width, button_size, "JUMP", keys[pygame.K_SPACE])

        # --- Draw camera direction arrows ---
        # Yaw (horizontal camera movement) -> left/right arrows
        # Pitch (vertical camera movement) -> up/down arrows
        yaw, pitch = act[0], act[1]
        threshold = 0.05  # Small threshold to avoid flickering on tiny movements

        # Left arrow (left edge, middle)
        draw_arrow_indicator(screen, 30, height // 2, 'left', yaw < -threshold)

        # Right arrow (right edge, middle)
        draw_arrow_indicator(screen, width - 30, height // 2, 'right', yaw > threshold)

        # Up arrow (top edge, middle) - negative pitch = look up
        draw_arrow_indicator(screen, width // 2, 30, 'up', pitch < -threshold)

        # Down arrow (bottom edge, middle) - positive pitch = look down
        draw_arrow_indicator(screen, width // 2, height - 30, 'down', pitch > threshold)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print("✅ Exited live session")


if __name__ == "__main__":
    main()
