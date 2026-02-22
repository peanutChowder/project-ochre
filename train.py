#!/usr/bin/env python3

import os
import time
import json
import random
import math
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

# v7.0.5: Performance optimization - enable cuDNN autotuner
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    print("Enabled cuDNN benchmark mode for performance")

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

# --- LOSS WEIGHTS (v7.0.5 rebalance) ---
SEMANTIC_WEIGHT = 1.0     # v7.0.2: Restored - essential to prevent mode collapse
LPIPS_WEIGHT = 2.0        # v7.0.5: Increased to restore camera U/D (was strongest action in v4.x)
IDM_LOSS_WEIGHT = 0.5     # v4.11: 84x gradient boost for movement
AR_LOSS_WEIGHT = 2.5      # Keep same as v6.3

# --- v7.1.1 ANTI-COLLAPSE TRAINING ---
# Objective: prevent AR rollouts from collapsing to a tiny, overconfident code subset.
#
# Change 1: Stochastic AR feedback during training (sampling instead of argmax).
AR_FEEDBACK_MODE = "topk"   # "argmax" | "topk"
AR_SAMPLE_TOPK = 50
AR_SAMPLE_TEMP = 1.0
#
# Change 2: Differentiable marginal code-distribution regularizer (AR-only by default).
# v7.1.1: DIV_REG_WEIGHT increased 1e-4 → 1e-3. Inference diagnostics (k50_t1.0 and k50_t2.0)
# showed consistency_score=1.0 at both temperatures, confirming 1e-4 is too weak to overcome
# cross-entropy pressure. See detailed-changelogs/v7.1.1-CHANGES.md.
DIV_REG_WEIGHT = 1e-3
DIV_REG_START_STEP = 5_000
DIV_REG_APPLY_TO = "ar"     # "ar" | "tf+ar"
#
# Change 3 (optional): Differentiable anti-repetition prior (AR-only; default off).
# v7.1.1: Enabled at 1e-4. argmax_unique_codes has improved enough (~46-55 unique codes)
# that spatial tiling of the remaining dominant codes is now the next-order failure mode.
REP_REG_WEIGHT = 1e-4
REP_REG_APPLY_TO = "ar"     # "ar"

# Gumbel temperature schedule (v7.0.2: Annealing restored)
GUMBEL_START = 1.0
GUMBEL_END = 0.1
GUMBEL_DECAY_STEPS = 20000

# v7.0.5: Corruption recovery training (denoising objective).
# Train the model to predict the correct next frame even when its input tokens are slightly wrong.
# Synchronized corruption + AR: Both start at step 5k, corruption applies to TF and AR steps.
CORRUPT_TF_ONLY = False          # v7.0.5: Corrupt both TF and AR to train actual AR error recovery
CORRUPT_START_P = 0.00
# v7.1.0: Disable corruption by default to reduce confounds; re-enable only if objective metrics improve.
CORRUPT_END_P = 0.00
CORRUPT_RAMP_STEPS = 30000       # v7.0.5: Fast ramp, reaches max by step 30k (aligns with Gumbel)
CORRUPT_TOKEN_REPLACE_FRAC = 1.0 # fraction of corrupted tokens replaced with random indices (rest reserved for future modes)
CORRUPT_BLOCK_PROB = 0.15        # per-sample chance to corrupt a spatial block (in addition to per-token corruption)
CORRUPT_BLOCK_MIN_FRAC = 0.10    # min block side length as fraction of H/W
CORRUPT_BLOCK_MAX_FRAC = 0.35    # max block side length as fraction of H/W

# LPIPS decode should start "soft" (mixture of codebook vectors) to match the v7.0.2 intent:
# encourage early exploration + smoother gradients, then optionally become "hard" later.
LPIPS_SOFT_TAU_THRESHOLD = 0.3  # use soft embeddings when current_tau > threshold
LPIPS_SOFT_TF_ONLY = True       # keep AR steps hard to reduce drift/feedback early on

# --- AR CURRICULUM (v6.3 config) ---
BASE_SEQ_LEN = 16
CURRICULUM_AR = True
AR_WARMUP_STEPS = 5000
AR_MIN_LEN = 1            # v7.0.2: Critical fix: Allow short AR horizons if diversity is low
AR_MAX_LEN = 20
AR_DIVERSITY_GATE_START = 5000
MIN_UNIQUE_CODES_FOR_AR_GROWTH = 30

# v7.0.3: Run-local AR resize freeze.
# When resuming from a checkpoint, relying only on GLOBAL step means the AR curriculum can immediately start
# resizing (often increasing) based on fresh/empty EMA state. Freeze resizing for the first N steps of the
# CURRENT run to avoid sudden ar_len jumps after each resume.
RUN_LOCAL_AR_RESIZE_FREEZE_STEPS = 5000

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
RUN_NAME = "v7.1.1-step0k"
MODEL_OUT_PREFIX = "ochre-v7.1.1"

LOG_STEPS = 10
IMAGE_LOG_STEPS = 1000
MILESTONE_SAVE_STEPS = 5000

RESUME_CHECKPOINT_PATH = ""  # v7.0.5: Start fresh (no checkpoint resume)

# Action validation
ACTION_VALIDATION_STEPS = [1, 5, 10]
ACTION_VISUAL_ROLLOUT_LEN = 30

# --- DIAGNOSTIC LOGGING ---
# Goal: make failure modes legible (action pathways dead? temporal attn over-sticky? token motion not shifting?).
DIAGNOSTIC_LOG_STEPS = 1000
DIAG_ROLLOUT_LEN = 30
DIAG_MAX_SHIFT_X = 10
DIAG_MAX_SHIFT_Y = 6
DIAG_ATTENTION_QUERY_TOKENS = 64
DIAG_LOG_PER_BLOCK = False

# --- FIXED-CONTEXT EVAL SNAPSHOTS (v7.1.0; strongly recommended) ---
# Periodically write inference-style diagnostic JSONs on fixed contexts to prevent "it looked better once" drift.
EVAL_SNAPSHOT_STEPS = 5000
# Fixed contexts for reproducibility — use a small local set, not the full training dataset.
EVAL_SNAPSHOT_CONTEXT_DIR = DATA_DIR
EVAL_SNAPSHOT_NUM_CONTEXTS = 3
EVAL_SNAPSHOT_OUT_ROOT = "./diagnostics/runs/v7.1.1"
EVAL_SNAPSHOT_TOPK = 50
EVAL_SNAPSHOT_TEMP = 1.0
EVAL_SNAPSHOT_RECENCY_DECAY = 1.0
# Keep snapshots reasonably light to avoid stalling training.
EVAL_PROFILE = "light"  # "light" | "full"


# ==========================================
# HELPERS
# ==========================================

class GumbelScheduler:
    def __init__(self, start=1.0, end=0.1, decay_steps=20000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get_tau(self, step):
        if step >= self.decay_steps:
            return self.end
        return self.start - (self.start - self.end) * (step / self.decay_steps)


class CorruptionScheduler:
    """v7.0.3: Linearly ramp input token corruption probability."""
    def __init__(self, start_p: float, end_p: float, ramp_steps: int):
        self.start_p = float(start_p)
        self.end_p = float(end_p)
        self.ramp_steps = int(ramp_steps)

    def get_p(self, step: int) -> float:
        if self.ramp_steps <= 0:
            return self.end_p
        if step >= self.ramp_steps:
            return self.end_p
        return self.start_p + (self.end_p - self.start_p) * (step / self.ramp_steps)


def corrupt_tokens(
    z_in: torch.Tensor,
    *,
    codebook_size: int,
    p: float,
) -> torch.Tensor:
    """
    v7.0.3: Corrupt discrete token grid to create a denoising/recovery training signal.

    z_in: (B, H, W) long
    Returns: (B, H, W) long
    """
    if p <= 0:
        return z_in

    z = z_in
    B, H, W = z.shape
    device = z.device

    # 1) Per-token random replacement.
    mask = (torch.rand((B, H, W), device=device) < p)
    if mask.any():
        rand_idx = torch.randint(0, int(codebook_size), (B, H, W), device=device, dtype=z.dtype)
        # Reserved for future corruption modes; currently always replace.
        z = torch.where(mask, rand_idx, z)

    # 2) Spatial block corruption (simulates localized smear/ghost patches).
    if CORRUPT_BLOCK_PROB > 0:
        block_do = (torch.rand((B,), device=device) < CORRUPT_BLOCK_PROB)
        if block_do.any():
            z_out = z.clone()
            for b in torch.nonzero(block_do, as_tuple=False).flatten().tolist():
                bh = max(1, int(round(H * random.uniform(CORRUPT_BLOCK_MIN_FRAC, CORRUPT_BLOCK_MAX_FRAC))))
                bw = max(1, int(round(W * random.uniform(CORRUPT_BLOCK_MIN_FRAC, CORRUPT_BLOCK_MAX_FRAC))))
                y0 = random.randint(0, max(0, H - bh))
                x0 = random.randint(0, max(0, W - bw))
                block = torch.randint(0, int(codebook_size), (bh, bw), device=device, dtype=z.dtype)
                z_out[b, y0:y0 + bh, x0:x0 + bw] = block
            z = z_out

    return z


def sample_tokens(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """
    Sample discrete token indices from logits (for stochastic AR feedback).

    logits: (B, C, H, W)
    returns: (B, H, W) long
    """
    B, C, H, W = logits.shape
    temperature = max(float(temperature), 1e-3)
    logits = logits.float() / temperature

    logits_flat = logits.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

    if top_k is not None and 0 < int(top_k) < C:
        k = int(top_k)
        topk_vals, topk_idx = torch.topk(logits_flat, k, dim=-1)  # (B, HW, k)
        probs = torch.softmax(topk_vals, dim=-1)
        sampled_rel = torch.multinomial(probs.reshape(-1, k), num_samples=1).view(B, -1)  # (B, HW)
        chosen = topk_idx.gather(-1, sampled_rel.unsqueeze(-1)).squeeze(-1)
    else:
        probs = torch.softmax(logits_flat, dim=-1)  # (B, HW, C)
        chosen = torch.multinomial(probs.reshape(-1, C), num_samples=1).view(B, -1)  # (B, HW)

    return chosen.view(B, H, W).long()


def marginal_kl_to_uniform_from_logits(logits: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Differentiable anti-collapse regularizer: KL(p_bar || Uniform).

    logits: (B, C, H, W)
    """
    p = torch.softmax(logits.float(), dim=1)  # (B, C, H, W)
    p_bar = p.mean(dim=(0, 2, 3))            # (C,)
    p_bar = p_bar / (p_bar.sum() + eps)
    C = int(p_bar.numel())
    log_u = -math.log(C)
    return (p_bar * (torch.log(p_bar + eps) - log_u)).sum()


def expected_neighbor_agreement_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Differentiable anti-repetition prior: expected equality of adjacent codes.

    logits: (B, C, H, W)
    """
    p = torch.softmax(logits.float(), dim=1)
    eq_h = (p[:, :, :, :-1] * p[:, :, :, 1:]).sum(dim=1).mean()
    eq_v = (p[:, :, :-1, :] * p[:, :, 1:, :]).sum(dim=1).mean()
    return 0.5 * (eq_h + eq_v)


@torch.no_grad()
def logits_calibration_stats(logits: torch.Tensor, *, codebook_size: int, eps: float = 1e-9) -> dict:
    """
    Calibration + diversity stats from logits (no gradients).
    """
    probs = torch.softmax(logits.float(), dim=1)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1).mean()
    max_prob = probs.max(dim=1).values.mean()
    argmax = logits.argmax(dim=1)
    present = torch.bincount(argmax.reshape(-1), minlength=int(codebook_size)) > 0
    unique_codes = present.sum().float()
    return {"mean_entropy": entropy, "mean_max_prob": max_prob, "argmax_unique_codes": unique_codes}


def _select_fixed_eval_contexts() -> list[str]:
    if EVAL_SNAPSHOT_STEPS <= 0 or EVAL_SNAPSHOT_NUM_CONTEXTS <= 0:
        return []
    if not isinstance(EVAL_SNAPSHOT_CONTEXT_DIR, str) or not os.path.isdir(EVAL_SNAPSHOT_CONTEXT_DIR):
        return []
    import glob
    paths = sorted(glob.glob(os.path.join(EVAL_SNAPSHOT_CONTEXT_DIR, "*.npz")))
    return paths[: int(EVAL_SNAPSHOT_NUM_CONTEXTS)]


def _serialize_inference_diagnostics(results: dict) -> dict:
    """
    Convert numpy arrays to lists (matches diagnostics/inference_diagnostics.py JSON format).
    """
    import numpy as _np

    results_serializable: dict = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, _np.ndarray):
                    results_serializable[key][k] = v.tolist()
                else:
                    results_serializable[key][k] = v
        else:
            results_serializable[key] = value
    return results_serializable


@torch.no_grad()
def save_fixed_context_eval_snapshots(model, vqvae, context_paths: list[str], *, global_step: int, device: str) -> None:
    if not context_paths:
        return

    import io
    import contextlib
    from pathlib import Path
    from diagnostics.inference_diagnostics import run_all_diagnostics
    from diagnostics.analyze_checkpoint import get_test_actions

    out_dir = os.path.join(EVAL_SNAPSHOT_OUT_ROOT, str(global_step))
    os.makedirs(out_dir, exist_ok=True)

    # Configure diagnostics workload.
    config = {
        "topk": int(EVAL_SNAPSHOT_TOPK),
        "temperature": float(EVAL_SNAPSHOT_TEMP),
        "recency_decay": float(EVAL_SNAPSHOT_RECENCY_DECAY),
    }
    if EVAL_PROFILE == "light":
        config.update({
            "entropy_steps": 20,
            "code_dist_steps": 20,
            "consistency_samples": 3,
            "consistency_steps": 8,
            "action_sensitivity_steps": 8,
            "quality_steps": 20,
        })

    # Temporarily set recency_decay if supported.
    old_rd = None
    if hasattr(model, "temporal_attn") and hasattr(model.temporal_attn, "recency_decay"):
        old_rd = float(getattr(model.temporal_attn, "recency_decay"))
        model.temporal_attn.recency_decay = float(config["recency_decay"])

    test_actions = get_test_actions(torch.device(device))

    try:
        for ctx_path in context_paths:
            data = np.load(ctx_path)
            z0 = torch.from_numpy(data["tokens"][0]).long().to(device).unsqueeze(0)

            # Inference diagnostics are chatty; keep training logs clean.
            with contextlib.redirect_stdout(io.StringIO()):
                results = run_all_diagnostics(model, vqvae, z0, test_actions, config)

            payload = {"config": config, "results": _serialize_inference_diagnostics(results)}
            out_path = os.path.join(
                out_dir,
                f"{Path(ctx_path).stem}_inference_diagnostics_k{config['topk']}_t{config['temperature']}_rd{config['recency_decay']}.json",
            )
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
    finally:
        if old_rd is not None:
            model.temporal_attn.recency_decay = old_rd


class ARCurriculum:
    """v6.3 AR Curriculum with diversity gating."""
    def __init__(self):
        self.ar_len = AR_MIN_LEN
        self.tf_ema = None
        self.ar_ema = None
        self.alpha = 0.98
        self.brake_ratio_upper = 2.5
        self.brake_ratio_lower = 1.6
        # v7.0.5: Throttle curriculum changes to prevent file descriptor exhaustion
        self.last_resize_time = 0
        self.min_resize_interval = 10.0  # seconds

    def update(self, step, lpips_tf, lpips_ar, unique_codes=None):
        import time
        current_time = time.time()

        if step < AR_WARMUP_STEPS:
            return 0

        # Diversity gate: hold back AR growth if token diversity is low
        if (unique_codes is not None and
            step >= AR_DIVERSITY_GATE_START and
            unique_codes < MIN_UNIQUE_CODES_FOR_AR_GROWTH):
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 1)
            return self.ar_len

        # v7.0.5: Convert tensors to scalars only here (avoids sync every step in main loop)
        lpips_tf_val = lpips_tf.item() if torch.is_tensor(lpips_tf) else lpips_tf
        lpips_ar_val = lpips_ar.item() if torch.is_tensor(lpips_ar) else lpips_ar

        if lpips_ar_val <= 0 or lpips_tf_val <= 0:
            return self.ar_len

        # EMA update
        if self.tf_ema is None:
            self.tf_ema = lpips_tf_val
            self.ar_ema = lpips_ar_val
        else:
            self.tf_ema = self.alpha * self.tf_ema + (1 - self.alpha) * lpips_tf_val
            self.ar_ema = self.alpha * self.ar_ema + (1 - self.alpha) * lpips_ar_val

        ratio = self.ar_ema / (self.tf_ema + 1e-6)

        # v7.0.5: Throttle resize to prevent file descriptor exhaustion
        # Only allow ar_len changes if enough time has passed since last change
        time_since_last = current_time - self.last_resize_time
        old_ar_len = self.ar_len

        if ratio > self.brake_ratio_upper + 0.05:
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 1)
        elif ratio < self.brake_ratio_lower - 0.05 and time_since_last >= self.min_resize_interval:
            self.ar_len = min(AR_MAX_LEN, self.ar_len + 1)

        # Update last resize time if ar_len actually changed
        if self.ar_len != old_ar_len:
            self.last_resize_time = current_time

        return self.ar_len


class SemanticCodebookLoss(nn.Module):
    """Codebook-aware semantic loss with Gumbel-Softmax."""
    def __init__(self, codebook_tensor):
        super().__init__()
        self.register_buffer('codebook', codebook_tensor.clone().detach())

    def forward(self, logits, target_indices, tau):
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
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


@torch.no_grad()
def compute_rollout_tokens(model, z_start, action_vec, num_steps):
    """Multi-step AR rollout returning predicted token grids."""
    temporal_buffer = []
    tokens = []
    z_t = z_start
    for _ in range(num_steps):
        logits, new_state = model.step(z_t, action_vec, temporal_buffer)
        z_t = logits.argmax(dim=1)
        tokens.append(z_t.detach())
        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > TEMPORAL_CONTEXT_LEN:
            temporal_buffer.pop(0)
    return tokens


@torch.no_grad()
def best_shift_match(
    z_prev: torch.Tensor,
    z_curr: torch.Tensor,
    *,
    max_dx: int,
    max_dy: int,
) -> dict:
    """
    Estimate spatial shift between discrete token grids by maximizing exact-token match.

    z_prev, z_curr: (H, W) long
    Returns: dict(best_dx, best_dy, best_match)
    """
    assert z_prev.dim() == 2 and z_curr.dim() == 2
    H, W = z_prev.shape

    best = {"best_dx": 0, "best_dy": 0, "best_match": -1.0}

    # Use a sentinel that should never appear in tokens.
    sentinel = torch.iinfo(z_prev.dtype).min if not torch.is_floating_point(z_prev) else -1.0

    for dy in range(-max_dy, max_dy + 1):
        if dy == 0:
            y_shifted = z_prev
        elif dy > 0:
            pad = torch.full((dy, W), sentinel, device=z_prev.device, dtype=z_prev.dtype)
            y_shifted = torch.cat([pad, z_prev[:H - dy, :]], dim=0)
        else:
            pad = torch.full((-dy, W), sentinel, device=z_prev.device, dtype=z_prev.dtype)
            y_shifted = torch.cat([z_prev[-dy:, :], pad], dim=0)

        for dx in range(-max_dx, max_dx + 1):
            x_shifted = torch.roll(y_shifted, shifts=dx, dims=1)
            match = (x_shifted == z_curr).float().mean().item()
            if match > best["best_match"]:
                best = {"best_dx": int(dx), "best_dy": int(dy), "best_match": float(match)}

    return best


@torch.no_grad()
def diagnostic_action_path_stats(model, action_vec: torch.Tensor) -> dict:
    """
    Summarize per-block AdaLN camera vs movement pathway magnitudes for a given action.
    """
    stats = {}
    cam_l2 = {"attn": [], "ffn": []}
    mov_l2 = {"attn": [], "ffn": []}
    gate_mean = {"attn": [], "ffn": []}

    for i, block in enumerate(model.blocks):
        p_attn = block.adaln_attn.action_to_params(action_vec, split_paths=True)
        cam_l2["attn"].append(p_attn["camera"]["gate_raw"].pow(2).mean().sqrt().item())
        mov_l2["attn"].append(p_attn["movement"]["gate_raw"].pow(2).mean().sqrt().item())
        gate_mean["attn"].append(p_attn["combined"]["gate"].mean().item())

        p_ffn = block.adaln_ffn.action_to_params(action_vec, split_paths=True)
        cam_l2["ffn"].append(p_ffn["camera"]["gate_raw"].pow(2).mean().sqrt().item())
        mov_l2["ffn"].append(p_ffn["movement"]["gate_raw"].pow(2).mean().sqrt().item())
        gate_mean["ffn"].append(p_ffn["combined"]["gate"].mean().item())

        if DIAG_LOG_PER_BLOCK:
            stats[f"adaln/block{i:02d}/attn_cam_gate_raw_l2"] = cam_l2["attn"][-1]
            stats[f"adaln/block{i:02d}/attn_mov_gate_raw_l2"] = mov_l2["attn"][-1]
            stats[f"adaln/block{i:02d}/attn_gate_mean"] = gate_mean["attn"][-1]
            stats[f"adaln/block{i:02d}/ffn_cam_gate_raw_l2"] = cam_l2["ffn"][-1]
            stats[f"adaln/block{i:02d}/ffn_mov_gate_raw_l2"] = mov_l2["ffn"][-1]
            stats[f"adaln/block{i:02d}/ffn_gate_mean"] = gate_mean["ffn"][-1]

    for which in ("attn", "ffn"):
        stats[f"adaln/{which}_cam_gate_raw_l2_mean"] = float(np.mean(cam_l2[which])) if cam_l2[which] else 0.0
        stats[f"adaln/{which}_mov_gate_raw_l2_mean"] = float(np.mean(mov_l2[which])) if mov_l2[which] else 0.0
        stats[f"adaln/{which}_gate_mean_mean"] = float(np.mean(gate_mean[which])) if gate_mean[which] else 0.0
        stats[f"adaln/{which}_cam_vs_mov_gate_raw_l2_ratio"] = (
            stats[f"adaln/{which}_cam_gate_raw_l2_mean"] / (stats[f"adaln/{which}_mov_gate_raw_l2_mean"] + 1e-9)
        )

    return stats


@torch.no_grad()
def diagnostic_temporal_attention_stats(model, z_t: torch.Tensor, action_vec: torch.Tensor) -> dict:
    """
    Run a single forward to probe temporal attention distribution with a synthetic buffer.
    Uses the model's own compress_frame outputs so stats reflect real tokenization of memory.
    """
    # Build a small temporal buffer by running a few static steps.
    temporal_buffer = []
    z = z_t
    for _ in range(min(4, TEMPORAL_CONTEXT_LEN)):
        logits, new_state = model.step(z, action_vec, temporal_buffer)
        temporal_buffer.append(new_state.detach())
        if len(temporal_buffer) > TEMPORAL_CONTEXT_LEN:
            temporal_buffer.pop(0)
        z = logits.argmax(dim=1)

    # Recompute x (post-block) for the current z and query attention weights.
    x = model.embed(z).permute(0, 3, 1, 2)
    x = model.stem(x)
    x = x.flatten(2).transpose(1, 2)
    for block in model.blocks:
        x = block(x, action_vec, model.H, model.W)

    attn = model.temporal_attn.attention_stats(x, temporal_buffer, num_query_tokens=DIAG_ATTENTION_QUERY_TOKENS)
    if attn is None:
        return {}

    out = {
        "temporal_attn/entropy": float(attn["attn_entropy"].item()),
        "temporal_attn/src_len": float(attn["src_len"]),
        "temporal_attn/num_queries": float(attn["num_queries"]),
        "temporal_attn/attn_max": float(attn["attn_mean_per_src"].max().item()),
        "temporal_attn/attn_min": float(attn["attn_mean_per_src"].min().item()),
        "temporal_attn/frames_in_buffer": float(attn.get("frames_in_buffer", 0)),
        "temporal_attn/tokens_per_frame": float(attn.get("tokens_per_frame", 0)),
    }

    per_frame = attn.get("attn_mean_per_frame")
    if per_frame is not None:
        # Most recent frame is last in temporal_buffer; report how much mass goes to it vs oldest.
        out["temporal_attn/frame_mass_max"] = float(per_frame.max().item())
        out["temporal_attn/frame_mass_min"] = float(per_frame.min().item())
        out["temporal_attn/frame_mass_last"] = float(per_frame[-1].item())
        out["temporal_attn/frame_mass_first"] = float(per_frame[0].item())
        out["temporal_attn/frame_mass_last_over_first"] = float(per_frame[-1].item() / (per_frame[0].item() + 1e-9))

    return out


def diagnostic_action_grad_stats(model) -> dict:
    """
    Gradient-flow diagnostic: do movement/camera action pathways get gradient signal?
    Call after backward + unscale, before optimizer step.
    """
    def grad_l2(params):
        total = 0.0
        count = 0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += float(g.pow(2).sum().item())
            count += g.numel()
        return (total / max(count, 1)) ** 0.5

    cam_params = []
    mov_params = []
    for block in model.blocks:
        cam_params += list(block.adaln_attn.camera_mlp.parameters())
        mov_params += list(block.adaln_attn.movement_mlp.parameters())
        cam_params += list(block.adaln_ffn.camera_mlp.parameters())
        mov_params += list(block.adaln_ffn.movement_mlp.parameters())

    return {
        "grads/adaln_camera_l2": float(grad_l2(cam_params)),
        "grads/adaln_movement_l2": float(grad_l2(mov_params)),
        "grads/adaln_camera_over_movement": float(grad_l2(cam_params) / (grad_l2(mov_params) + 1e-12)),
    }


@torch.no_grad()
def diagnostic_action_logit_sensitivity(model, z_t: torch.Tensor, actions: dict) -> dict:
    """
    Forward sensitivity (token-space): how much do logits change under different actions?
    Uses empty temporal buffer to isolate action conditioning vs temporal memory.
    """
    device = z_t.device
    temporal_buffer = []
    logits_static, _ = model.step(z_t, actions["static"], temporal_buffer)
    logits_static = logits_static.float()

    out = {}
    for name, a in actions.items():
        if name == "static":
            continue
        logits, _ = model.step(z_t, a.to(device), temporal_buffer)
        d = (logits.float() - logits_static).abs().mean().item()
        out[f"logit_sensitivity/{name}_absmean"] = float(d)
    return out


@torch.no_grad()
def diagnostic_rollout_shift_stats(model, z_start: torch.Tensor, action_vec: torch.Tensor, *, name: str) -> dict:
    """
    Token-space rollout diagnostics: does the model produce consistent spatial shifts or just churn/ghost?
    """
    tokens = compute_rollout_tokens(model, z_start, action_vec, DIAG_ROLLOUT_LEN)
    if len(tokens) < 2:
        return {}

    dxs, dys, matches = [], [], []
    churns = []

    z_prev = tokens[0][0]
    for z_next_b in tokens[1:]:
        z_next = z_next_b[0]
        s = best_shift_match(z_prev, z_next, max_dx=DIAG_MAX_SHIFT_X, max_dy=DIAG_MAX_SHIFT_Y)
        dxs.append(s["best_dx"])
        dys.append(s["best_dy"])
        matches.append(s["best_match"])
        churns.append(float((z_prev != z_next).float().mean().item()))
        z_prev = z_next

    dxs_np = np.array(dxs, dtype=np.float32)
    dys_np = np.array(dys, dtype=np.float32)
    matches_np = np.array(matches, dtype=np.float32)
    churns_np = np.array(churns, dtype=np.float32)

    return {
        f"rollout/{name}/dx_mean": float(dxs_np.mean()),
        f"rollout/{name}/dx_std": float(dxs_np.std()),
        f"rollout/{name}/dy_mean": float(dys_np.mean()),
        f"rollout/{name}/dy_std": float(dys_np.std()),
        f"rollout/{name}/match_mean": float(matches_np.mean()),
        f"rollout/{name}/match_min": float(matches_np.min()),
        f"rollout/{name}/token_churn_mean": float(churns_np.mean()),
        f"rollout/{name}/token_churn_max": float(churns_np.max()),
    }


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
            "camera_up": torch.tensor(encode_action_v5_np(pitch_raw=0.5), device=device).unsqueeze(0),
            "camera_down": torch.tensor(encode_action_v5_np(pitch_raw=-0.5), device=device).unsqueeze(0),
            "move_forward": torch.tensor(encode_action_v5_np(w=1.0), device=device).unsqueeze(0),
            "jump": torch.tensor(encode_action_v5_np(jump=1.0), device=device).unsqueeze(0),
        }

        metrics = {}

        # Quick validation at different rollout lengths
        for num_steps in ACTION_VALIDATION_STEPS:
            rollouts = {name: compute_rollout(model, vqvae, z_start, action, num_steps, device)
                       for name, action in test_actions.items()}

            static_frame = rollouts["static"][-1].flatten()
            diff_cam_l = (static_frame - rollouts["camera_left"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_cam_r = (static_frame - rollouts["camera_right"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_cam_u = (static_frame - rollouts["camera_up"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_cam_d = (static_frame - rollouts["camera_down"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_move = (static_frame - rollouts["move_forward"][-1].flatten()).pow(2).mean().sqrt().item()
            diff_jump = (static_frame - rollouts["jump"][-1].flatten()).pow(2).mean().sqrt().item()

            suffix = f"_{num_steps}step" if num_steps > 1 else ""
            metrics[f"action_response/camera_left_diff{suffix}"] = diff_cam_l
            metrics[f"action_response/camera_right_diff{suffix}"] = diff_cam_r
            metrics[f"action_response/camera_up_diff{suffix}"] = diff_cam_u
            metrics[f"action_response/camera_down_diff{suffix}"] = diff_cam_d
            metrics[f"action_response/move_forward_diff{suffix}"] = diff_move
            metrics[f"action_response/jump_diff{suffix}"] = diff_jump
            metrics[f"action_response/average{suffix}"] = (diff_cam_l + diff_cam_r + diff_cam_u + diff_cam_d + diff_move + diff_jump) / 6

        # Visual rollout grid (30 frames)
        rollouts_30 = {name: compute_rollout(model, vqvae, z_start, action, ACTION_VISUAL_ROLLOUT_LEN, device)
                      for name, action in test_actions.items()}

        num_vis = 6
        vis_frames = []
        for name in ["static", "camera_left", "camera_right", "camera_up", "camera_down", "move_forward", "jump"]:
            frames = rollouts_30[name]
            indices = [int(i * (ACTION_VISUAL_ROLLOUT_LEN - 1) / (num_vis - 1)) for i in range(num_vis)]
            for idx in indices:
                vis_frames.append(torch.clamp(frames[idx], 0.0, 1.0))

        grid = make_grid(torch.stack(vis_frames), nrow=num_vis, normalize=False, padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/action_rollout_30step": wandb.Image(
                grid_np,
                caption=f"30-Step Rollouts | Rows: Static, Cam-L, Cam-R, Cam-U, Cam-D, Move-Fwd, Jump | Step {global_step}"
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

# v7.0.3: Optional resume from checkpoint (e.g. continue from v7.0.2 weights).
resume_checkpoint = None
if RESUME_CHECKPOINT_PATH:
    if not os.path.isfile(RESUME_CHECKPOINT_PATH):
        raise FileNotFoundError(f"RESUME_CHECKPOINT_PATH not found: {RESUME_CHECKPOINT_PATH}")
    print(f"Resuming from checkpoint: {RESUME_CHECKPOINT_PATH}")
    resume_checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE)

    # Model state
    resume_state = resume_checkpoint.get("model_state") if isinstance(resume_checkpoint, dict) else None
    if not isinstance(resume_state, dict):
        raise KeyError("Resume checkpoint missing 'model_state' dict.")
    model.load_state_dict(resume_state, strict=True)

    # Optimizer state (best-effort; allow continuing even if optimizer state is incompatible)
    opt_state = resume_checkpoint.get("optimizer_state") if isinstance(resume_checkpoint, dict) else None
    if isinstance(opt_state, dict):
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"⚠️ Warning: could not load optimizer state ({type(e).__name__}: {e}); continuing with fresh optimizer.")

# Losses
semantic_criterion = SemanticCodebookLoss(codebook).to(DEVICE)
import lpips
lpips_criterion = lpips.LPIPS(net='alex').to(DEVICE).eval().requires_grad_(False)

# Curriculum
curriculum = ARCurriculum()
tau_scheduler = GumbelScheduler(GUMBEL_START, GUMBEL_END, GUMBEL_DECAY_STEPS)
corrupt_scheduler = CorruptionScheduler(CORRUPT_START_P, CORRUPT_END_P, CORRUPT_RAMP_STEPS)

# v7.0.3: Restore curriculum state when available (new checkpoints will include this).
if isinstance(resume_checkpoint, dict):
    cur_state = resume_checkpoint.get("curriculum_state")
    if isinstance(cur_state, dict):
        curriculum.ar_len = int(cur_state.get("ar_len", curriculum.ar_len))
        curriculum.tf_ema = cur_state.get("tf_ema", curriculum.tf_ema)
        curriculum.ar_ema = cur_state.get("ar_ema", curriculum.ar_ema)

    # v7.0.5: Pre-warm EMA if missing to prevent AR shock on resume.
    # If EMA is None after restore and ar_len > 1, estimate based on v7.0.2 results.
    if curriculum.tf_ema is None and curriculum.ar_len > 1:
        curriculum.tf_ema = 0.11  # v7.0.2 typical TF LPIPS at ar_len=6
        curriculum.ar_ema = 0.19  # Estimated AR LPIPS for stable ar_len=6 (ratio ~1.7)
        print(f"⚠️ Warning: Pre-warmed curriculum EMA (tf={curriculum.tf_ema:.3f}, ar={curriculum.ar_ema:.3f}) based on ar_len={curriculum.ar_len}")

# ==========================================
# TRAINING LOOP
# ==========================================

global_step = int(resume_checkpoint.get("global_step", 0)) if isinstance(resume_checkpoint, dict) else 0
run_start_global_step = global_step
prev_seq_len = BASE_SEQ_LEN
dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=prev_seq_len)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)
loader_iter = iter(loader)

print(f"Training Start")
print(f"  - Losses: Semantic({SEMANTIC_WEIGHT}) + LPIPS({LPIPS_WEIGHT}) + IDM({IDM_LOSS_WEIGHT})")
print(f"  - AR curriculum: {AR_MIN_LEN}-{AR_MAX_LEN}, diversity gate at {MIN_UNIQUE_CODES_FOR_AR_GROWTH}")

model.train()
prev_lpips_tf = 0.0
prev_lpips_ar = 0.0
prev_unique_codes = None

fixed_eval_contexts = _select_fixed_eval_contexts()
if fixed_eval_contexts:
    print(f"  - Fixed-context eval snapshots: {len(fixed_eval_contexts)} contexts every {EVAL_SNAPSHOT_STEPS} steps")
elif EVAL_SNAPSHOT_STEPS > 0 and EVAL_SNAPSHOT_NUM_CONTEXTS > 0:
    print(
        "  - Fixed-context eval snapshots: ENABLED but no .npz contexts found in "
        f"{EVAL_SNAPSHOT_CONTEXT_DIR!r}; no snapshots will be written."
    )

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
    run_local_step = global_step - run_start_global_step
    if run_local_step < RUN_LOCAL_AR_RESIZE_FREEZE_STEPS:
        ar_len = curriculum.ar_len
    else:
        ar_len = curriculum.update(global_step, prev_lpips_tf, prev_lpips_ar, prev_unique_codes)
    seq_len = max(BASE_SEQ_LEN, ar_len + 1)
    current_tau = tau_scheduler.get_tau(global_step)
    current_corrupt_p = corrupt_scheduler.get_p(global_step)

    if seq_len != prev_seq_len:
        print(f"[Curriculum] Resizing: seq_len {prev_seq_len} -> {seq_len}")
        prev_seq_len = seq_len

        # v7.0.5: Clean up old dataloader to prevent file descriptor leaks
        if 'loader' in locals():
            del loader_iter
            del loader
            import gc
            gc.collect()  # Force cleanup of worker processes

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

    # v7.0.5: Performance optimization - set_to_none=True is faster than zeroing
    optimizer.zero_grad(set_to_none=True)

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
        logits_tf_last = None
        logits_ar_last = None

        # --- Per-segment objective stats (TF vs AR) ---
        tf_stats_sum = {"mean_entropy": torch.tensor(0.0, device=DEVICE),
                        "mean_max_prob": torch.tensor(0.0, device=DEVICE),
                        "argmax_unique_codes": torch.tensor(0.0, device=DEVICE)}
        ar_stats_sum = {"mean_entropy": torch.tensor(0.0, device=DEVICE),
                        "mean_max_prob": torch.tensor(0.0, device=DEVICE),
                        "argmax_unique_codes": torch.tensor(0.0, device=DEVICE)}
        tf_steps = 0
        ar_steps = 0

        # v7.0.5: Performance optimization - pre-decode all target RGB frames once
        # (15-20% speedup by avoiding K decoder calls per step)
        with torch.no_grad():
            tgt_rgb_all = torch.stack([vqvae_model.decode_code(Z_target[:, t]) for t in range(K)], dim=1)

        for t in range(K):
            is_ar_step = (t >= ar_cutoff and t > 0)

            # Input: teacher forcing or AR
            if t == 0:
                z_in = Z_seq[:, 0]
            elif is_ar_step:
                # v7.1.0: Stochastic AR feedback during training (match inference regime).
                assert logits_last is not None, "logits_last should be set from previous timestep"
                if AR_FEEDBACK_MODE == "argmax":
                    z_in = logits_last.argmax(dim=1).detach()
                elif AR_FEEDBACK_MODE == "topk":
                    with torch.no_grad():
                        z_in = sample_tokens(logits_last, temperature=AR_SAMPLE_TEMP, top_k=AR_SAMPLE_TOPK)
                else:
                    raise ValueError(f"Unknown AR_FEEDBACK_MODE={AR_FEEDBACK_MODE!r} (expected 'argmax' or 'topk').")
            else:
                z_in = Z_seq[:, t]

            # v7.0.3: corruption recovery training.
            if (not is_ar_step) or (not CORRUPT_TF_ONLY):
                z_in = corrupt_tokens(z_in, codebook_size=codebook_size, p=current_corrupt_p)

            # Model step (return post-temporal spatial features for IDM supervision)
            logits_t, new_state, x_spatial_t = model.step(
                z_in,
                A_seq[:, t],
                temporal_buffer,
                return_spatial_features=True,
            )
            logits_last = logits_t
            if is_ar_step:
                logits_ar_last = logits_t
            else:
                logits_tf_last = logits_t

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
            loss_sem = semantic_criterion(logits_flat, target_flat, current_tau)
            loss_sem_list.append(loss_sem)
            t_sem_total += (time.perf_counter() - t_sem_start)

            # B. LPIPS Loss
            t_lpips_start = time.perf_counter()
            lpips_use_soft = (current_tau > LPIPS_SOFT_TAU_THRESHOLD) and (not (LPIPS_SOFT_TF_ONLY and is_ar_step))
            probs = F.gumbel_softmax(logits_flat, tau=current_tau, hard=not lpips_use_soft, dim=-1)
            probs = probs.reshape(B, H, W, -1)
            soft_emb = torch.matmul(probs, codebook).permute(0, 3, 1, 2)
            pred_rgb = vqvae_model.decoder(soft_emb)

            # v7.0.5: Use pre-decoded target (already computed before loop)
            tgt_rgb = tgt_rgb_all[:, t]

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

            # v7.1.0 objective logging: TF vs AR logit calibration + diversity.
            with torch.no_grad():
                s = logits_calibration_stats(logits_t, codebook_size=codebook_size)
                if is_ar_step:
                    for k_, v_ in s.items():
                        ar_stats_sum[k_] = ar_stats_sum[k_] + v_
                    ar_steps += 1
                else:
                    for k_, v_ in s.items():
                        tf_stats_sum[k_] = tf_stats_sum[k_] + v_
                    tf_steps += 1

        # Aggregate losses
        loss_sem_total = torch.stack(loss_sem_list).mean()
        loss_lpips_tf = torch.stack(loss_lpips_tf_list).mean() if loss_lpips_tf_list else torch.tensor(0.0, device=DEVICE)
        loss_lpips_ar = torch.stack(loss_lpips_ar_list).mean() if loss_lpips_ar_list else torch.tensor(0.0, device=DEVICE)
        loss_idm_total = torch.stack(loss_idm_list).mean() if loss_idm_list else torch.tensor(0.0, device=DEVICE)

        loss_lpips_total = loss_lpips_tf + AR_LOSS_WEIGHT * loss_lpips_ar

        # v7.0.5: Track for curriculum as tensors (avoid GPU->CPU sync every step)
        # Only sync when curriculum.update() needs the values
        if loss_lpips_tf_list:
            prev_lpips_tf = loss_lpips_tf.detach()
        if loss_lpips_ar_list:
            prev_lpips_ar = loss_lpips_ar.detach()

        # v7.1.0: Differentiable anti-collapse regularizers (logits -> probs -> marginal stats).
        loss_div_reg = torch.tensor(0.0, device=DEVICE)
        loss_rep_reg = torch.tensor(0.0, device=DEVICE)

        # Diversity regularizer (KL to uniform) with warm start.
        if DIV_REG_WEIGHT > 0 and global_step >= DIV_REG_START_STEP:
            if DIV_REG_APPLY_TO == "ar" and logits_ar_last is not None:
                loss_div_reg = marginal_kl_to_uniform_from_logits(logits_ar_last)
            elif DIV_REG_APPLY_TO == "tf+ar" and (logits_tf_last is not None or logits_ar_last is not None):
                parts = []
                if logits_tf_last is not None:
                    parts.append(marginal_kl_to_uniform_from_logits(logits_tf_last))
                if logits_ar_last is not None:
                    parts.append(marginal_kl_to_uniform_from_logits(logits_ar_last))
                loss_div_reg = torch.stack(parts).mean() if parts else loss_div_reg
            else:
                raise ValueError(f"Unknown DIV_REG_APPLY_TO={DIV_REG_APPLY_TO!r} (expected 'ar' or 'tf+ar').")

        # Optional neighbor anti-repetition prior (AR-only by default).
        if REP_REG_WEIGHT > 0:
            if REP_REG_APPLY_TO == "ar" and logits_ar_last is not None:
                loss_rep_reg = expected_neighbor_agreement_from_logits(logits_ar_last)
            else:
                raise ValueError(f"Unknown REP_REG_APPLY_TO={REP_REG_APPLY_TO!r} (expected 'ar').")

        total_loss = (SEMANTIC_WEIGHT * loss_sem_total +
                     LPIPS_WEIGHT * loss_lpips_total +
                     IDM_LOSS_WEIGHT * loss_idm_total +
                     DIV_REG_WEIGHT * loss_div_reg +
                     REP_REG_WEIGHT * loss_rep_reg)

        # Finalize TF/AR stats (mean over timesteps in each segment).
        with torch.no_grad():
            tf_steps_safe = max(tf_steps, 1)
            ar_steps_safe = max(ar_steps, 1)
            tf_stats_mean = {k_: (v_ / tf_steps_safe) for k_, v_ in tf_stats_sum.items()}
            ar_stats_mean = {k_: (v_ / ar_steps_safe) for k_, v_ in ar_stats_sum.items()}

            # AR marginal distribution stats (p_bar) on the last AR logits (or fallback to last logits).
            pbar_entropy = torch.tensor(0.0, device=DEVICE)
            pbar_top1_mass = torch.tensor(0.0, device=DEVICE)
            if logits_ar_last is not None:
                probs_ar = torch.softmax(logits_ar_last.float(), dim=1)
                p_bar = probs_ar.mean(dim=(0, 2, 3))
                p_bar = p_bar / (p_bar.sum() + 1e-9)
                pbar_entropy = -(p_bar * torch.log(p_bar + 1e-9)).sum()
                pbar_top1_mass = p_bar.max()

            # Curriculum diversity gate should track AR regime when present.
            prev_unique_codes = int(ar_stats_mean["argmax_unique_codes"].item() if logits_ar_last is not None else tf_stats_mean["argmax_unique_codes"].item())

    t_forward = (time.perf_counter() - t_forward_start) * 1000  # ms

    # 5. Backward
    t_backward_start = time.perf_counter()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    t_backward = (time.perf_counter() - t_backward_start) * 1000  # ms

    diag_grad_log = None
    if wandb and (global_step % DIAGNOSTIC_LOG_STEPS == 0):
        diag_grad_log = diagnostic_action_grad_stats(model)

    t_optimizer_start = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    t_optimizer = (time.perf_counter() - t_optimizer_start) * 1000  # ms

    # 6. Diagnostics (already computed during forward; this is just CPU sync for logging)
    t_diagnostics_start = time.perf_counter()
    tf_entropy = float(tf_stats_mean["mean_entropy"].item())
    tf_max_prob = float(tf_stats_mean["mean_max_prob"].item())
    tf_unique = float(tf_stats_mean["argmax_unique_codes"].item())
    ar_entropy = float(ar_stats_mean["mean_entropy"].item())
    ar_max_prob = float(ar_stats_mean["mean_max_prob"].item())
    ar_unique = float(ar_stats_mean["argmax_unique_codes"].item())
    ar_pbar_entropy = float(pbar_entropy.item())
    ar_pbar_top1_mass = float(pbar_top1_mass.item())
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
            f"AR: {ar_len} | UniqueAR: {ar_unique:.0f} | "
            f"{throughput:.2f} steps/s\n"
            f"  Tau: {current_tau:.3f} | CorruptP: {current_corrupt_p:.3f} | "
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
                "train/loss_div_reg": loss_div_reg.item(),
                "train/loss_rep_reg": loss_rep_reg.item(),
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/gumbel_tau": current_tau,
                "train/corrupt_p": current_corrupt_p,
                "curriculum/seq_len": seq_len,
                "curriculum/ar_len": ar_len,
                "curriculum/lpips_ratio": prev_lpips_ar / (prev_lpips_tf + 1e-6),
                "diagnostics/tf/mean_entropy": tf_entropy,
                "diagnostics/ar/mean_entropy": ar_entropy,
                "diagnostics/tf/mean_max_prob": tf_max_prob,
                "diagnostics/ar/mean_max_prob": ar_max_prob,
                "diagnostics/tf/argmax_unique_codes": tf_unique,
                "diagnostics/ar/argmax_unique_codes": ar_unique,
                "diagnostics/ar/pbar_entropy": ar_pbar_entropy,
                "diagnostics/ar/pbar_top1_mass": ar_pbar_top1_mass,
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

    if wandb and (global_step % DIAGNOSTIC_LOG_STEPS == 0):
        with torch.no_grad():
            device = Z_seq.device
            z_start = Z_seq[0:1, 0]

            diag_actions = {
                "static": torch.tensor(encode_action_v5_np(), device=device).unsqueeze(0),
                "camera_left": torch.tensor(encode_action_v5_np(yaw_raw=-0.5), device=device).unsqueeze(0),
                "camera_right": torch.tensor(encode_action_v5_np(yaw_raw=0.5), device=device).unsqueeze(0),
                "camera_up": torch.tensor(encode_action_v5_np(pitch_raw=0.5), device=device).unsqueeze(0),
                "camera_down": torch.tensor(encode_action_v5_np(pitch_raw=-0.5), device=device).unsqueeze(0),
                "move_forward": torch.tensor(encode_action_v5_np(w=1.0), device=device).unsqueeze(0),
                "jump": torch.tensor(encode_action_v5_np(jump=1.0), device=device).unsqueeze(0),
            }

            diag_log = {}

            # 1) Action-path “is movement dead?” probes
            for name, a in diag_actions.items():
                s = diagnostic_action_path_stats(model, a)
                for k, v in s.items():
                    diag_log[f"{k}/{name}"] = v

            # 1b) Gradient-flow: are movement AdaLN params getting grads?
            if diag_grad_log:
                diag_log.update(diag_grad_log)

            # 1c) Token-space action sensitivity (logits differ at all?)
            diag_log.update(diagnostic_action_logit_sensitivity(model, z_start, diag_actions))

            # 2) Temporal attention stickiness probes (entropy / max weight)
            diag_log.update(diagnostic_temporal_attention_stats(model, z_start, diag_actions["static"]))
            diag_log.update(diagnostic_temporal_attention_stats(model, z_start, diag_actions["camera_right"]))

            # 3) Token-space rollout shift probes (does yaw behave like a shift? does it churn?)
            for name, a in diag_actions.items():
                if name in ("static", "camera_left", "camera_right", "camera_up", "camera_down", "move_forward", "jump"):
                    diag_log.update(diagnostic_rollout_shift_stats(model, z_start, a, name=name))

            wandb.log(diag_log, step=global_step)

    if fixed_eval_contexts and EVAL_SNAPSHOT_STEPS > 0 and (global_step % EVAL_SNAPSHOT_STEPS == 0) and global_step > 0:
        was_training = model.training
        model.eval()
        save_fixed_context_eval_snapshots(
            model,
            vqvae_model,
            fixed_eval_contexts,
            global_step=global_step,
            device=DEVICE,
        )
        if was_training:
            model.train()

    if global_step % MILESTONE_SAVE_STEPS == 0 and global_step > 0:
        # Remove old checkpoint if it exists
        old_checkpoint = f"./checkpoints/{MODEL_OUT_PREFIX}-step{global_step - MILESTONE_SAVE_STEPS}.pt"
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'global_step': global_step,
            # v7.0.3: Save curriculum state to enable accurate resume behavior.
            'curriculum_state': {
                'ar_len': int(curriculum.ar_len),
                'tf_ema': float(curriculum.tf_ema) if curriculum.tf_ema is not None else None,
                'ar_ema': float(curriculum.ar_ema) if curriculum.ar_ema is not None else None,
            },
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
