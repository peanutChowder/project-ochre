#!/usr/bin/env python3
"""
Live interactive inference for the Ochre world model (VQ‑VAE + MinecraftConvTransformer).

Controls:
  - W/A/S/D: movement (binary)
  - Space: jump (binary)
  - Arrow keys: look yaw/pitch → normalized to [-1, 1]
  - Esc or Q: quit
"""

import argparse, os, time, numpy as np, torch, pygame
from vq_vae.vq_vae import VQVAE, IMAGE_HEIGHT, IMAGE_WIDTH
from model_transformer import MinecraftConvTransformer
from action_encoding import encode_action_v5_np

# ---------------------- Constants ----------------------
FRAME_H = IMAGE_HEIGHT  # 72
FRAME_W = IMAGE_WIDTH   # 128
LATENT_H = 18
LATENT_W = 32
CTX_LEN = 30  # rolling token/action window
ACTION_DIM = 15  # v5.0: 15D discrete encoding [yaw(5), pitch(3), W, A, S, D, jump, sprint, sneak]


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


def _infer_conv_transformer_kwargs(checkpoint: dict, state_dict: dict, latent_h: int, latent_w: int) -> dict:
    """
    Infer `MinecraftConvTransformer(...)` kwargs from checkpoint metadata + state_dict.
    Prefers checkpoint["config"] (saved by train.py), with state_dict fallbacks.
    """
    if "embed.weight" not in state_dict:
        raise KeyError("Checkpoint does not look like a MinecraftConvTransformer state_dict (missing embed.weight).")

    codebook_size, embed_dim = state_dict["embed.weight"].shape
    hidden_dim = int(state_dict.get("stem.0.weight", torch.empty((0,))).shape[0]) or None

    # Count transformer blocks.
    block_indices = set()
    for k in state_dict.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    num_layers = (max(block_indices) + 1) if block_indices else None

    # Infer window_size + num_heads from relative position bias table if needed.
    window_size = None
    num_heads = None
    for k, v in state_dict.items():
        if k.endswith("attn.relative_position_bias_table") and isinstance(v, torch.Tensor):
            num_heads = int(v.shape[1])
            n = int(v.shape[0])
            # n = (2w-1)^2  => w = (sqrt(n)+1)/2
            root = int(round(n ** 0.5))
            if root * root == n:
                window_size = int((root + 1) // 2)
            break

    # IDM span from dt embedding table.
    idm_max_span = 5
    if "idm.dt_embed.weight" in state_dict:
        idm_max_span = int(state_dict["idm.dt_embed.weight"].shape[0] - 1)

    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    # Prefer explicit config values saved by train.py.
    hidden_dim = int(cfg.get("hidden_dim", hidden_dim or 384))
    num_layers = int(cfg.get("num_layers", num_layers or 4))
    num_heads = int(cfg.get("num_heads", num_heads or 6))
    temporal_context_len = int(cfg.get("temporal_context_len", cfg.get("temporal_context", 4)))

    # window_size isn't currently saved in train.py checkpoint; infer or default.
    window_size = int(cfg.get("window_size", window_size or 4))

    return dict(
        codebook_size=int(codebook_size),
        embed_dim=int(embed_dim),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        H=latent_h,
        W=latent_w,
        action_dim=ACTION_DIM,
        temporal_context_len=temporal_context_len,
        window_size=window_size,
        idm_max_span=idm_max_span,
        use_checkpointing=False,
    )


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
    p.add_argument("--checkpoint", required=True, help="MinecraftConvTransformer checkpoint (.pt)")
    p.add_argument("--vqvae_ckpt", required=True, help="Trained VQ‑VAE checkpoint (.pt)")
    p.add_argument("--context_npz", help="Optional preprocessed .npz providing initial tokens/actions")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--warmup", type=int, default=20, help="Number of context frames to use for hidden state warmup (0 = no warmup)")
    p.add_argument("--key_look_gain", type=float, default=0.5, help="Arrow key look gain added to yaw/pitch")
    p.add_argument("--greedy", action="store_true", help="Use argmax decoding (no sampling)")
    p.add_argument("--temperature", type=float, default=1.05, help="Sampling temperature (>0); 1.0 ≈ unbiased")
    p.add_argument("--topk", type=int, default=32, help="Top‑k sampling cutoff (0 = disable)")
    p.add_argument("--recency_decay", type=float, default=1.0, help="Temporal attention recency decay (1.0=no decay, 0.9=prefer recent frames)")
    p.add_argument("--use_context_actions", action="store_true", help="Use GT actions from context file instead of keyboard input")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Live inference on {device} @ {args.fps} FPS")
    print(f"Resolution: {FRAME_W}x{FRAME_H} (Latent: {LATENT_W}x{LATENT_H})")

    # --- Load models ---
    vqvae = load_vqvae(args.vqvae_ckpt, device)

    # Load ConvTransformer checkpoint + infer constructor kwargs.
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state_dict)} (expected dict/state_dict)")
    model_kwargs = _infer_conv_transformer_kwargs(ckpt if isinstance(ckpt, dict) else {}, state_dict, LATENT_H, LATENT_W)

    # Add runtime recency_decay parameter
    model_kwargs["recency_decay"] = args.recency_decay

    # Sanity-check codebook size agreement with the VQ-VAE.
    vq_codebook = int(vqvae.vq_vae.embedding.shape[1]) if hasattr(vqvae, "vq_vae") and hasattr(vqvae.vq_vae, "embedding") else None
    if vq_codebook is not None and int(model_kwargs["codebook_size"]) != vq_codebook:
        raise RuntimeError(
            f"Checkpoint codebook_size={model_kwargs['codebook_size']} does not match VQ-VAE codebook_size={vq_codebook}. "
            "Use a matching world model checkpoint and VQ-VAE checkpoint."
        )

    model = MinecraftConvTransformer(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    print(
        "✅ Loaded world model from "
        f"{args.checkpoint} (embed_dim={model_kwargs['embed_dim']}, hidden_dim={model_kwargs['hidden_dim']}, "
        f"layers={model_kwargs['num_layers']}, temporal_ctx={model_kwargs['temporal_context_len']}, window={model_kwargs['window_size']})"
    )
    model.eval()

    # --- Initialize state ---
    temporal_buffer = []

    # Store context actions for --use_context_actions flag
    actions_ctx = None

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
        
        # Use warmup from args (default 20 to match training seq_len)
        warmup = min(args.warmup, tokens_ctx.shape[0], actions_ctx.shape[0]) if args.warmup > 0 else 0
        start_idx = 0 # or -warmup
        
        z_warmup = tokens_ctx[start_idx : start_idx+warmup].unsqueeze(0)
        a_warmup = actions_ctx[start_idx : start_idx+warmup].unsqueeze(0)
        
        if warmup > 0:
            with torch.no_grad():
                print(f"Priming temporal buffer with {warmup} frames...")
                for i in range(warmup):
                    z_t = z_warmup[:, i] # (1, H, W)
                    a_t = a_warmup[:, i] # (1, A)
                    _, new_state = model.step(z_t, a_t, temporal_buffer)
                    temporal_buffer.append(new_state.detach())
                    if len(temporal_buffer) > model.temporal_context_len:
                        temporal_buffer.pop(0)

            # Set current token to the last one from context
            current_z = z_warmup[:, -1]
            print("State primed.")
        else:
            # No warmup - use first frame from context
            current_z = tokens_ctx[0:1]
            print("⚠️ Warmup disabled (--warmup=0)")

    else:
        print("⚠️ No context provided — starting from random state")
        codebook_size = vq_codebook if vq_codebook is not None else int(model_kwargs["codebook_size"])
        current_z = torch.randint(0, codebook_size, (1, LATENT_H, LATENT_W), device=device)

    # Validate --use_context_actions flag
    if args.use_context_actions:
        if actions_ctx is None:
            print("❌ ERROR: --use_context_actions requires --context_npz")
            return
        print(f"✅ Using GT actions from context ({actions_ctx.shape[0]} frames available)")

    # --- Pygame setup ---
    pygame.init()
    width, height = FRAME_W * args.scale, FRAME_H * args.scale
    screen = pygame.display.set_mode((width, height))
    title = "Project Ochre — Live Inference (Stateful)"
    if args.use_context_actions:
        title += " [Context Actions]"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    pygame.font.init()
    hud_font = pygame.font.SysFont("Arial", max(10, int(10 * args.scale / 5)))
    button_font = pygame.font.SysFont("Arial", max(14, int(14 * args.scale / 5)), bold=True)

    def get_action_vec():
        """
        v5.0: Encode keyboard input as 15D discrete action vector.
        Format: [yaw(5), pitch(3), W, A, S, D, jump, sprint, sneak]
        """
        keys = pygame.key.get_pressed()
        # Arrow keys for camera look (continuous input)
        key_yaw = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        key_pitch = float(keys[pygame.K_DOWN]) - float(keys[pygame.K_UP])
        yaw_raw = clamp(key_yaw * args.key_look_gain, -1.0, 1.0)
        pitch_raw = clamp(key_pitch * args.key_look_gain, -1.0, 1.0)
        action = encode_action_v5_np(
            yaw_raw=yaw_raw,
            pitch_raw=pitch_raw,
            w=keys[pygame.K_w],
            a=keys[pygame.K_a],
            s=keys[pygame.K_s],
            d=keys[pygame.K_d],
            jump=keys[pygame.K_SPACE],
            sprint=(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]),
            sneak=(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]),
        )

        return action.tolist(), float(yaw_raw), float(pitch_raw)

    def decode_action_v5_for_hud(action_15):
        """
        Convert a 15D discrete action vector into approximate continuous values for HUD/UI only.
        """
        # Representative values aligned with action bin thresholds in action_encoding.py.
        yaw_vals = [-1.0, -0.3, 0.0, 0.3, 1.0]
        pitch_vals = [-1.0, 0.0, 1.0]
        yaw_idx = int(np.argmax(action_15[0:5]))
        pitch_idx = int(np.argmax(action_15[5:8]))
        yaw_raw = yaw_vals[yaw_idx]
        pitch_raw = pitch_vals[pitch_idx]
        return yaw_raw, pitch_raw

    if args.use_context_actions:
        print("🎬 Replaying GT actions from context (no user input)")
    else:
        print("🎮 W/A/S/D move • Space jump • Shift sprint • Ctrl sneak • Arrows look • ESC/Q quit")

    # Frame counter for context action indexing
    frame_idx = 0

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
        if args.use_context_actions:
            # Use GT actions from context file
            if frame_idx < actions_ctx.shape[0]:
                current_a = actions_ctx[frame_idx:frame_idx+1]  # (1, ACTION_DIM)
                action_vec = current_a[0].cpu().tolist()
                yaw, pitch = decode_action_v5_for_hud(action_vec)
            else:
                # Ran out of context actions, use zeros
                action_vec = [0.0] * ACTION_DIM
                yaw, pitch = 0.0, 0.0
                current_a = torch.tensor([action_vec], dtype=torch.float32, device=device)
        else:
            # Use keyboard input
            action_vec, yaw, pitch = get_action_vec()
            current_a = torch.tensor([action_vec], dtype=torch.float32, device=device)  # (1, 15)

        # 2. Step Model
        with torch.no_grad():
            logits, new_state = model.step(current_z, current_a, temporal_buffer)
            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > model.temporal_context_len:
                temporal_buffer.pop(0)

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

        # Increment frame counter for context action indexing
        frame_idx += 1

        # draw
        surf = pygame.image.frombuffer(img.tobytes(), (FRAME_W, FRAME_H), "RGB")
        if args.scale != 1:
            surf = pygame.transform.smoothscale(surf, (width, height))
        screen.blit(surf, (0, 0))

        # Draw HUD
        hud = f"Y:{yaw:+.2f} P:{pitch:+.2f}"
        screen.blit(hud_font.render(hud, True, (255, 255, 255)), (10, 10))

        # --- Draw WASD+Jump key visualizations ---
        # When using context actions, visualize the context actions instead of keyboard
        if args.use_context_actions:
            # v5.0 discrete layout: [yaw(5), pitch(3), W, A, S, D, jump, sprint, sneak]
            w_pressed = bool(action_vec[8] > 0.5)
            a_pressed = bool(action_vec[9] > 0.5)
            s_pressed = bool(action_vec[10] > 0.5)
            d_pressed = bool(action_vec[11] > 0.5)
            space_pressed = bool(action_vec[12] > 0.5)
        else:
            keys = pygame.key.get_pressed()
            w_pressed = keys[pygame.K_w]
            s_pressed = keys[pygame.K_s]
            a_pressed = keys[pygame.K_a]
            d_pressed = keys[pygame.K_d]
            space_pressed = keys[pygame.K_SPACE]

        button_size = max(30, int(30 * args.scale / 5))
        button_spacing = max(5, int(5 * args.scale / 5))

        # Position in bottom-left area
        base_x = 20
        base_y = height - button_size * 3 - button_spacing * 2 - 20

        # W key (top middle)
        draw_key_button(screen, base_x + button_size + button_spacing, base_y,
                       button_size, button_size, "W", w_pressed)

        # A key (middle left)
        draw_key_button(screen, base_x, base_y + button_size + button_spacing,
                       button_size, button_size, "A", a_pressed)

        # S key (middle middle)
        draw_key_button(screen, base_x + button_size + button_spacing, base_y + button_size + button_spacing,
                       button_size, button_size, "S", s_pressed)

        # D key (middle right)
        draw_key_button(screen, base_x + (button_size + button_spacing) * 2, base_y + button_size + button_spacing,
                       button_size, button_size, "D", d_pressed)

        # Space key (bottom, wide)
        space_width = button_size * 3 + button_spacing * 2
        draw_key_button(screen, base_x, base_y + (button_size + button_spacing) * 2,
                       space_width, button_size, "JUMP", space_pressed)

        # --- Draw camera direction arrows ---
        # Yaw (horizontal camera movement) -> left/right arrows
        # Pitch (vertical camera movement) -> up/down arrows
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
