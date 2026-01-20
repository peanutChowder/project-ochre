#!/usr/bin/env python3
"""
v7.0.1 ConvTransformer Training Script
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid

# Model imports
from vq_vae.vq_vae import VQVAE
from model_transformer import MinecraftConvTransformer
from action_encoding import encode_action_v5_np

import wandb
if 'WANDB_API_KEY' in os.environ:
    wandb.login(key=os.environ['WANDB_API_KEY'])

# ==========================================
# CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# --- PATHS ---
DATA_DIR = "../preprocessedv5_plains_clear"
VQVAE_PATH = "./checkpoints/vqvae_v2.1.6__epoch100.pt"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# --- MODEL CONFIG ---
HIDDEN_DIM = 384
NUM_LAYERS = 4
NUM_HEADS = 6
TEMPORAL_CONTEXT_LEN = 8
WINDOW_SIZE = 4

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 8            # Reduced to prevent OOM
MAX_STEPS = 300_000
LR = 1e-4                 # Higher than ConvGRU - transformers tolerate more
WARMUP_STEPS = 2000       # Longer warmup for transformer stability
MIN_LR = 1e-6

# --- LOSS WEIGHTS (v7.0.1 rebalance) ---
SEMANTIC_WEIGHT = 0.5     # Reduced - was dominating gradients over visual quality
LPIPS_WEIGHT = 5.0        # Increased - must be primary signal for good reconstructions
IDM_LOSS_WEIGHT = 0.5     # v4.11: 84x gradient boost for movement
AR_LOSS_WEIGHT = 2.5      # Keep same as v6.3

# Gumbel temperature (fixed, no schedule - simpler)
GUMBEL_TAU = 0.2  # Reduced from 0.5 - sharper soft embeddings for better reconstruction

# --- AR CURRICULUM (v6.3 config) ---
BASE_SEQ_LEN = 16
CURRICULUM_AR = True
AR_WARMUP_STEPS = 5000
AR_MIN_LEN = 10
AR_MAX_LEN = 20
AR_DIVERSITY_GATE_START = 5000
MIN_UNIQUE_CODES_FOR_AR_GROWTH = 30

# --- IDM CONFIG ---
MAX_IDM_SPAN = 5
ACTION_DIM = 15
ACTION_WEIGHTS = torch.tensor([
    1.0, 1.0, 1.0, 1.0, 1.0,    # Yaw
    1.0, 1.0, 1.0,               # Pitch
    10.0, 10.0, 10.0, 10.0,      # WASD
    5.0,                          # Jump
    2.0, 2.0                      # Sprint/Sneak
], device=DEVICE)

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v7.0.1-step10k"
MODEL_OUT_PREFIX = "ochre-v7.0.1"

LOG_STEPS = 10
IMAGE_LOG_STEPS = 1000
MILESTONE_SAVE_STEPS = 5000

# Action validation
ACTION_VALIDATION_STEPS = [1, 5, 10]
ACTION_VISUAL_ROLLOUT_LEN = 30


# ==========================================
# HELPERS
# ==========================================

class ARCurriculum:
    """v6.3 AR Curriculum with diversity gating."""
    def __init__(self):
        self.ar_len = AR_MIN_LEN
        self.tf_ema = None
        self.ar_ema = None
        self.alpha = 0.98
        self.brake_ratio_upper = 2.5
        self.brake_ratio_lower = 1.6

    def update(self, step, lpips_tf, lpips_ar, unique_codes=None):
        if step < AR_WARMUP_STEPS:
            return 0

        # Diversity gate: hold back AR growth if token diversity is low
        if (unique_codes is not None and
            step >= AR_DIVERSITY_GATE_START and
            unique_codes < MIN_UNIQUE_CODES_FOR_AR_GROWTH):
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 1)
            return self.ar_len

        if lpips_ar <= 0 or lpips_tf <= 0:
            return self.ar_len

        # EMA update
        if self.tf_ema is None:
            self.tf_ema = lpips_tf
            self.ar_ema = lpips_ar
        else:
            self.tf_ema = self.alpha * self.tf_ema + (1 - self.alpha) * lpips_tf
            self.ar_ema = self.alpha * self.ar_ema + (1 - self.alpha) * lpips_ar

        ratio = self.ar_ema / (self.tf_ema + 1e-6)

        if ratio > self.brake_ratio_upper + 0.05:
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 1)
        elif ratio < self.brake_ratio_lower - 0.05:
            self.ar_len = min(AR_MAX_LEN, self.ar_len + 1)

        return self.ar_len


class SemanticCodebookLoss(nn.Module):
    """Codebook-aware semantic loss with Gumbel-Softmax."""
    def __init__(self, codebook_tensor):
        super().__init__()
        self.register_buffer('codebook', codebook_tensor.clone().detach())

    def forward(self, logits, target_indices):
        probs = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=True, dim=-1)
        pred_vectors = torch.matmul(probs, self.codebook)
        target_vectors = F.embedding(target_indices, self.codebook)
        return F.mse_loss(pred_vectors, target_vectors)


class GTTokenDataset(Dataset):
    """Dataset for pre-tokenized sequences."""
    def __init__(self, manifest_path, root_dir, seq_len=16):
        with open(manifest_path, "r") as f:
            self.entries = json.load(f)["sequences"]
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.index_map = []
        for vid_idx, meta in enumerate(self.entries):
            L = meta["length"]
            if L > seq_len + 1:
                for i in range(L - (seq_len + 1)):
                    self.index_map.append((vid_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid_idx, start = self.index_map[idx]
        meta = self.entries[vid_idx]
        path = os.path.join(self.root_dir, meta["file"])

        try:
            with np.load(path, mmap_mode='r') as data:
                tokens = data["tokens"][start:start + self.seq_len + 1]
                actions = data["actions"][start:start + self.seq_len]
        except Exception:
            data = np.load(path)
            tokens = data["tokens"][start:start + self.seq_len + 1]
            actions = data["actions"][start:start + self.seq_len]

        Z_seq = tokens[:-1]
        Z_target = tokens[1:]

        return (
            torch.tensor(Z_seq.astype(np.int32), dtype=torch.long),
            torch.tensor(actions.astype(np.float32), dtype=torch.float32),
            torch.tensor(Z_target.astype(np.int32), dtype=torch.long),
        )


def log_images_to_wandb(vqvae, Z_target, logits, global_step):
    """Log reconstruction comparison to wandb."""
    if wandb is None:
        return
    with torch.no_grad():
        gt_indices = Z_target[:, -1]
        pred_indices = logits.argmax(dim=1)
        gt_rgb = vqvae.decode_code(gt_indices)[:4]
        pred_rgb = vqvae.decode_code(pred_indices)[:4]

        vis = torch.cat([gt_rgb, pred_rgb], dim=0)
        grid = make_grid(vis, nrow=4, normalize=False, value_range=(0, 1))
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/reconstruction": wandb.Image(
                grid_np, caption=f"Top: GT | Bot: Pred (Step {global_step})"
            )
        }, step=global_step)


def compute_rollout(model, vqvae, z_start, action_vec, num_steps, device):
    """Multi-step AR rollout with fixed action."""
    with torch.no_grad():
        temporal_buffer = []
        frames = []
        z_t = z_start

        for _ in range(num_steps):
            logits, new_state = model.step(z_t, action_vec, temporal_buffer)
            z_t = logits.argmax(dim=1)
            pred_rgb = vqvae.decode_code(z_t)[0]
            frames.append(pred_rgb.cpu())

            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > TEMPORAL_CONTEXT_LEN:
                temporal_buffer.pop(0)

        return frames


def validate_action_conditioning(model, vqvae, Z_seq, global_step):
    """Multi-step action validation."""
    if wandb is None:
        return {}

    with torch.no_grad():
        device = Z_seq.device
        z_start = Z_seq[0:1, 0]

        test_actions = {
            "static": torch.tensor(encode_action_v5_np(), device=device).unsqueeze(0),
            "camera_left": torch.tensor(encode_action_v5_np(yaw_raw=-0.5), device=device).unsqueeze(0),
            "camera_right": torch.tensor(encode_action_v5_np(yaw_raw=0.5), device=device).unsqueeze(0),
            "move_forward": torch.tensor(encode_action_v5_np(w=1.0), device=device).unsqueeze(0),
        }

        metrics = {}

        # Quick validation at different rollout lengths
        for num_steps in ACTION_VALIDATION_STEPS:
            rollouts = {name: compute_rollout(model, vqvae, z_start, action, num_steps, device)
                       for name, action in test_actions.items()}

            static_frame = rollouts["static"][-1].flatten()
            diff_cam_l = (static_frame - rollouts["camera_left"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_cam_r = (static_frame - rollouts["camera_right"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_move = (static_frame - rollouts["move_forward"][-1].flatten()).pow(2).mean().sqrt().item()

            suffix = f"_{num_steps}step" if num_steps > 1 else ""
            metrics[f"action_response/camera_left_diff{suffix}"] = diff_cam_l
            metrics[f"action_response/camera_right_diff{suffix}"] = diff_cam_r
            metrics[f"action_response/move_forward_diff{suffix}"] = diff_move
            metrics[f"action_response/average{suffix}"] = (diff_cam_l + diff_cam_r + diff_move) / 3

        # Visual rollout grid (30 frames)
        rollouts_30 = {name: compute_rollout(model, vqvae, z_start, action, ACTION_VISUAL_ROLLOUT_LEN, device)
                      for name, action in test_actions.items()}

        num_vis = 6
        vis_frames = []
        for name in ["static", "camera_left", "camera_right", "move_forward"]:
            frames = rollouts_30[name]
            indices = [int(i * (ACTION_VISUAL_ROLLOUT_LEN - 1) / (num_vis - 1)) for i in range(num_vis)]
            for idx in indices:
                vis_frames.append(torch.clamp(frames[idx], 0.0, 1.0))

        grid = make_grid(torch.stack(vis_frames), nrow=num_vis, normalize=False, padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/action_rollout_30step": wandb.Image(
                grid_np,
                caption=f"30-Step Rollouts | Rows: Static, Cam-L, Cam-R, Move-Fwd | Step {global_step}"
            )
        }, step=global_step)

        return metrics


# ==========================================
# SETUP
# ==========================================

# Check if this is being imported for testing
if __name__ != "__main__":
    # Skip setup when imported
    import sys
    sys.exit(0)

# Create checkpoints directory
os.makedirs("./checkpoints", exist_ok=True)

if wandb:
    wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
               config=dict(
                   batch_size=BATCH_SIZE, lr=LR, max_steps=MAX_STEPS,
                   hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                   semantic_weight=SEMANTIC_WEIGHT, lpips_weight=LPIPS_WEIGHT,
                   idm_weight=IDM_LOSS_WEIGHT, ar_loss_weight=AR_LOSS_WEIGHT,
               ))

# Load VQ-VAE
print(f"Loading VQ-VAE from {VQVAE_PATH}...")
vqvae_ckpt = torch.load(VQVAE_PATH, map_location=DEVICE)
conf = vqvae_ckpt.get("config", {})
vqvae_model = VQVAE(
    embedding_dim=conf.get("embedding_dim", 384),
    num_embeddings=conf.get("codebook_size", 1024)
).to(DEVICE)

if isinstance(vqvae_ckpt, dict) and {"encoder", "decoder", "quantizer"}.issubset(vqvae_ckpt.keys()):
    vqvae_model.encoder.load_state_dict(vqvae_ckpt["encoder"])
    vqvae_model.decoder.load_state_dict(vqvae_ckpt["decoder"])
    vqvae_model.vq_vae.load_state_dict(vqvae_ckpt["quantizer"])
elif isinstance(vqvae_ckpt, dict) and "model_state" in vqvae_ckpt:
    vqvae_model.load_state_dict(vqvae_ckpt["model_state"], strict=False)
else:
    vqvae_model.load_state_dict(vqvae_ckpt, strict=False)

vqvae_model.eval().requires_grad_(False)

codebook = vqvae_model.vq_vae.embedding.clone().detach().t()  # (K, D)
codebook_size = codebook.shape[0]
print(f"Codebook size: {codebook_size}")

# Initialize ConvTransformer
model = MinecraftConvTransformer(
    codebook_size=codebook_size,
    embed_dim=256,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    H=18, W=32,
    action_dim=ACTION_DIM,
    temporal_context_len=TEMPORAL_CONTEXT_LEN,
    window_size=WINDOW_SIZE,
    idm_max_span=MAX_IDM_SPAN,
    use_checkpointing=True,  # Enable gradient checkpointing to save VRAM
).to(DEVICE)

param_counts = model.count_parameters()
print(f"Model parameters: {param_counts['total']/1e6:.2f}M")

# Optimizer (single group - transformers don't need separate FiLM LR)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler = GradScaler(enabled=(DEVICE == "cuda"))

# Losses
semantic_criterion = SemanticCodebookLoss(codebook).to(DEVICE)
import lpips
lpips_criterion = lpips.LPIPS(net='alex').to(DEVICE).eval().requires_grad_(False)

# Curriculum
curriculum = ARCurriculum()

# ==========================================
# TRAINING LOOP
# ==========================================

global_step = 0
prev_seq_len = BASE_SEQ_LEN
dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=prev_seq_len)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)
loader_iter = iter(loader)

print(f"Training Start: v7.0 ConvTransformer")
print(f"  - Losses: Semantic({SEMANTIC_WEIGHT}) + LPIPS({LPIPS_WEIGHT}) + IDM({IDM_LOSS_WEIGHT})")
print(f"  - AR curriculum: {AR_MIN_LEN}-{AR_MAX_LEN}, diversity gate at {MIN_UNIQUE_CODES_FOR_AR_GROWTH}")

model.train()
prev_lpips_tf = 0.0
prev_lpips_ar = 0.0
prev_unique_codes = None

# Timing EMA for smooth metrics (alpha=0.1 gives ~10 step window)
timing_ema = {
    'data_load': 0.0,
    'forward': 0.0,
    'semantic': 0.0,
    'lpips': 0.0,
    'idm': 0.0,
    'backward': 0.0,
    'optimizer': 0.0,
    'diagnostics': 0.0,
    'total': 0.0,
}
timing_alpha = 0.1

while global_step < MAX_STEPS:
    t_step_start = time.perf_counter()
    t_last = t_step_start

    # 1. Update Curriculum
    ar_len = curriculum.update(global_step, prev_lpips_tf, prev_lpips_ar, prev_unique_codes)
    seq_len = max(BASE_SEQ_LEN, ar_len + 1)

    if seq_len != prev_seq_len:
        print(f"[Curriculum] Resizing: seq_len {prev_seq_len} -> {seq_len}")
        prev_seq_len = seq_len
        dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, persistent_workers=True)
        loader_iter = iter(loader)

    # 2. Get Batch
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)

    Z_seq, A_seq, Z_target = batch
    Z_seq, A_seq, Z_target = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target.to(DEVICE)
    B, K, H, W = Z_seq.shape

    t_now = time.perf_counter()
    t_data_load = (t_now - t_last) * 1000  # ms
    t_last = t_now

    # 3. Warmup LR
    if global_step <= WARMUP_STEPS:
        curr_lr = MIN_LR + (global_step / WARMUP_STEPS) * (LR - MIN_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = curr_lr

    optimizer.zero_grad()

    # 4. Forward Pass
    t_forward_start = time.perf_counter()
    t_sem_total = 0.0
    t_lpips_total = 0.0
    t_idm_total = 0.0

    with autocast(enabled=(DEVICE == "cuda")):
        loss_sem_list = []
        loss_lpips_tf_list = []
        loss_lpips_ar_list = []
        loss_idm_list = []

        ar_cutoff = K - ar_len
        temporal_buffer = []
        h_buffer = []  # For IDM
        logits_last = None

        for t in range(K):
            is_ar_step = (t >= ar_cutoff and t > 0)

            # Input: teacher forcing or AR
            if t == 0:
                z_in = Z_seq[:, 0]
            elif is_ar_step:
                with torch.no_grad():
                    z_in = logits_last.argmax(dim=1)
            else:
                z_in = Z_seq[:, t]

            # Model step (return post-temporal spatial features for IDM supervision)
            logits_t, new_state, x_spatial_t = model.step(
                z_in,
                A_seq[:, t],
                temporal_buffer,
                return_spatial_features=True,
            )
            logits_last = logits_t

            # Store for IDM (detach past states to prevent backprop through time)
            h_buffer.append(x_spatial_t)

            # Update temporal buffer
            temporal_buffer.append(new_state.detach())
            if len(temporal_buffer) > TEMPORAL_CONTEXT_LEN:
                temporal_buffer.pop(0)

            # --- Losses ---

            # A. Semantic Loss
            t_sem_start = time.perf_counter()
            logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
            target_flat = Z_target[:, t].reshape(-1)
            loss_sem = semantic_criterion(logits_flat, target_flat)
            loss_sem_list.append(loss_sem)
            t_sem_total += (time.perf_counter() - t_sem_start)

            # B. LPIPS Loss
            t_lpips_start = time.perf_counter()
            probs = F.gumbel_softmax(logits_flat, tau=GUMBEL_TAU, hard=True, dim=-1)
            probs = probs.reshape(B, H, W, -1)
            soft_emb = torch.matmul(probs, codebook).permute(0, 3, 1, 2)
            pred_rgb = vqvae_model.decoder(soft_emb)

            with torch.no_grad():
                tgt_rgb = vqvae_model.decode_code(Z_target[:, t])

            lpips_val = lpips_criterion(pred_rgb * 2 - 1, tgt_rgb * 2 - 1).mean()

            if is_ar_step:
                loss_lpips_ar_list.append(lpips_val)
            else:
                loss_lpips_tf_list.append(lpips_val)
            t_lpips_total += (time.perf_counter() - t_lpips_start)

            # C. IDM Loss
            if t >= 1:
                t_idm_start = time.perf_counter()
                k = random.randint(1, min(t, MAX_IDM_SPAN))
                h_start = h_buffer[-1-k]  # detached
                h_end = x_spatial_t       # attached (current step only)

                dt_tensor = torch.full((B,), k, device=DEVICE, dtype=torch.long)
                pred_action = model.idm(h_start, h_end, dt_tensor)

                # Average targets over span.
                # Feature at loop index t is a post-action state for A_seq[:, t] (predicting Z_{t+1}),
                # so the k actions "between" h_start (t-k) and h_end (t) are A[t-k+1 .. t].
                action_segment = A_seq[:, t-k:t]
                yaw_target = action_segment[:, :, 0:5].mean(dim=1)
                pitch_target = action_segment[:, :, 5:8].mean(dim=1)
                bin_target = action_segment[:, :, 8:15].mean(dim=1)

                ly = -(yaw_target * F.log_softmax(pred_action[:, 0:5], -1)).sum(-1).mean()
                lp = -(pitch_target * F.log_softmax(pred_action[:, 5:8], -1)).sum(-1).mean()
                lb = (F.binary_cross_entropy_with_logits(
                    pred_action[:, 8:15], bin_target, reduction='none'
                ) * ACTION_WEIGHTS[8:15]).mean()

                loss_idm_list.append(ly + lp + lb)
                t_idm_total += (time.perf_counter() - t_idm_start)

        # Aggregate losses
        loss_sem_total = torch.stack(loss_sem_list).mean()
        loss_lpips_tf = torch.stack(loss_lpips_tf_list).mean() if loss_lpips_tf_list else torch.tensor(0.0, device=DEVICE)
        loss_lpips_ar = torch.stack(loss_lpips_ar_list).mean() if loss_lpips_ar_list else torch.tensor(0.0, device=DEVICE)
        loss_idm_total = torch.stack(loss_idm_list).mean() if loss_idm_list else torch.tensor(0.0, device=DEVICE)

        loss_lpips_total = loss_lpips_tf + AR_LOSS_WEIGHT * loss_lpips_ar

        # Track for curriculum
        if loss_lpips_tf_list:
            prev_lpips_tf = loss_lpips_tf.item()
        if loss_lpips_ar_list:
            prev_lpips_ar = loss_lpips_ar.item()

        total_loss = (SEMANTIC_WEIGHT * loss_sem_total +
                     LPIPS_WEIGHT * loss_lpips_total +
                     IDM_LOSS_WEIGHT * loss_idm_total)

    t_forward = (time.perf_counter() - t_forward_start) * 1000  # ms

    # 5. Backward
    t_backward_start = time.perf_counter()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    t_backward = (time.perf_counter() - t_backward_start) * 1000  # ms

    t_optimizer_start = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    t_optimizer = (time.perf_counter() - t_optimizer_start) * 1000  # ms

    # 6. Diagnostics
    t_diagnostics_start = time.perf_counter()
    with torch.no_grad():
        probs_last = F.softmax(logits_last.float(), dim=1)
        entropy = -(probs_last * torch.log(probs_last + 1e-9)).sum(dim=1).mean()
        confidence = probs_last.max(dim=1)[0].mean()
        unique_codes = logits_last.argmax(dim=1).unique().numel()
        prev_unique_codes = int(unique_codes)
    t_diagnostics = (time.perf_counter() - t_diagnostics_start) * 1000  # ms

    step_time = time.perf_counter() - t_step_start

    # Update timing EMA
    timing_ema['data_load'] = timing_alpha * t_data_load + (1 - timing_alpha) * timing_ema['data_load']
    timing_ema['forward'] = timing_alpha * t_forward + (1 - timing_alpha) * timing_ema['forward']
    timing_ema['semantic'] = timing_alpha * (t_sem_total * 1000) + (1 - timing_alpha) * timing_ema['semantic']
    timing_ema['lpips'] = timing_alpha * (t_lpips_total * 1000) + (1 - timing_alpha) * timing_ema['lpips']
    timing_ema['idm'] = timing_alpha * (t_idm_total * 1000) + (1 - timing_alpha) * timing_ema['idm']
    timing_ema['backward'] = timing_alpha * t_backward + (1 - timing_alpha) * timing_ema['backward']
    timing_ema['optimizer'] = timing_alpha * t_optimizer + (1 - timing_alpha) * timing_ema['optimizer']
    timing_ema['diagnostics'] = timing_alpha * t_diagnostics + (1 - timing_alpha) * timing_ema['diagnostics']
    timing_ema['total'] = timing_alpha * (step_time * 1000) + (1 - timing_alpha) * timing_ema['total']

    # 7. Logging
    if global_step % LOG_STEPS == 0:
        throughput = 1 / step_time
        print(
            f"[Step {global_step}] Loss: {total_loss.item():.4f} | "
            f"Sem: {loss_sem_total.item():.4f} | LPIPS: {loss_lpips_total.item():.4f} | "
            f"IDM: {loss_idm_total.item():.4f} | "
            f"AR: {ar_len} | Unique: {unique_codes} | "
            f"{throughput:.2f} steps/s\n"
            f"  Timing: Data={timing_ema['data_load']:.0f}ms | Fwd={timing_ema['forward']:.0f}ms "
            f"(Sem={timing_ema['semantic']:.0f}ms, LPIPS={timing_ema['lpips']:.0f}ms, IDM={timing_ema['idm']:.0f}ms) | "
            f"Bwd={timing_ema['backward']:.0f}ms | Opt={timing_ema['optimizer']:.0f}ms"
        )

        if wandb:
            wandb.log({
                "train/loss": total_loss.item(),
                "train/loss_semantic": loss_sem_total.item(),
                "train/loss_lpips": loss_lpips_total.item(),
                "train/loss_lpips_tf": loss_lpips_tf.item(),
                "train/loss_lpips_ar": loss_lpips_ar.item(),
                "train/loss_idm": loss_idm_total.item(),
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]['lr'],
                "curriculum/seq_len": seq_len,
                "curriculum/ar_len": ar_len,
                "curriculum/lpips_ratio": prev_lpips_ar / (prev_lpips_tf + 1e-6),
                "diagnostics/entropy": float(entropy),
                "diagnostics/confidence": float(confidence),
                "diagnostics/unique_codes": unique_codes,
                "timing/step_ms": step_time * 1000,
                "timing/throughput": 1 / step_time,
                "timing/data_load_ms": timing_ema['data_load'],
                "timing/forward_ms": timing_ema['forward'],
                "timing/semantic_ms": timing_ema['semantic'],
                "timing/lpips_ms": timing_ema['lpips'],
                "timing/idm_ms": timing_ema['idm'],
                "timing/backward_ms": timing_ema['backward'],
                "timing/optimizer_ms": timing_ema['optimizer'],
                "timing/diagnostics_ms": timing_ema['diagnostics'],
            }, step=global_step)

    if global_step % IMAGE_LOG_STEPS == 0:
        log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)
        action_metrics = validate_action_conditioning(model, vqvae_model, Z_seq, global_step)
        if wandb and action_metrics:
            wandb.log(action_metrics, step=global_step)

    if global_step % MILESTONE_SAVE_STEPS == 0 and global_step > 0:
        # Remove old checkpoint if it exists
        old_checkpoint = f"./checkpoints/{MODEL_OUT_PREFIX}-step{global_step - MILESTONE_SAVE_STEPS}.pt"
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'global_step': global_step,
            'config': {
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'num_heads': NUM_HEADS,
                'temporal_context_len': TEMPORAL_CONTEXT_LEN,
            }
        }, f"./checkpoints/{MODEL_OUT_PREFIX}-step{global_step//1000}k.pt")
        print(f"Saved checkpoint: {MODEL_OUT_PREFIX}-step{global_step//1000}k.pt")

    global_step += 1

print("Training complete!")
