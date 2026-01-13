#!/usr/bin/env python3
import os, time, json, math, numpy as np, torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torchvision.utils import make_grid

# World Model imports
from vq_vae.vq_vae import VQVAE
from model_convGru import WorldModelConvFiLM
from action_encoding import encode_action_v5_np

import wandb
if 'WANDB_API_KEY' in os.environ:
    wandb.login(key=os.environ['WANDB_API_KEY'])

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# --- PATHS ---
DATA_DIR = "../preprocessedv5"
VQVAE_PATH = "./checkpoints/vqvae_v2.1.6__epoch100.pt" 
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 20         # v6.1: Reduced for larger model (512-dim + temporal attn)
MAX_STEPS = 300_000
LR = 3e-5
WARMUP_STEPS = 1000
MIN_LR = 1e-6
USE_CHECKPOINTING = False
TEMPORAL_CONTEXT_LEN = 8  # v6.1: Frames for temporal attention 

# --- LOSS WEIGHTS ---
SEMANTIC_WEIGHT = 1.0    
LPIPS_WEIGHT = 3.0       
LPIPS_FREQ = 1           # v6.0: Keep at 1 (model learns periodic artifacts otherwise)
GUMBEL_TAU_STEPS = 80_000  # v6.2: Slow down hardening to reduce early collapse
GUMBEL_TAU_MIN = 0.30      # v6.2: Higher floor to preserve token diversity early

AR_LOSS_WEIGHT = 2.5     

# --- CURRICULUM ---
BASE_SEQ_LEN = 16               
CURRICULUM_AR = True             
AR_WARMUP_STEPS = 5000          
AR_MIN_LEN = 10                 
AR_MAX_LEN = 20                 

# v6.2: Gate AR growth on token diversity to avoid pushing AR while in a collapsed regime.
AR_DIVERSITY_GATE_START = 5000
MIN_UNIQUE_CODES_FOR_AR_GROWTH = 30

# --- IDM CONFIG ---
IDM_LOSS_WEIGHT = 1.0   # Increased from 0.5 to compensate for lack of Ranking
MAX_IDM_SPAN = 5       
ACTION_DIM = 15

# v6.2: Teacher forcing token corruption to break the "identity map" shortcut.
TOKEN_CORRUPT_MAX_P = 0.05
TOKEN_CORRUPT_RAMP_STEPS = 50_000

# v6.2: Temporary entropy bonus (anti-collapse) applied only when LPIPS is computed.
ENTROPY_BONUS_WEIGHT = 0.01
ENTROPY_BONUS_STEPS = 50_000

# v6.2: Action-conditional LPIPS reweighting to address 87% signal masking.
# Diagnostic finding: 87% of frames have camera+movement co-occurring, camera gradients dominate.
# Solution: Boost LPIPS weight on movement-active frames to amplify weak movement signal.
LPIPS_MOVEMENT_BOOST = 4.0  # Boost LPIPS weight when any WASD/jump/sprint/sneak active
LPIPS_MOVEMENT_BOOST_RAMP_STEPS = 30_000  # Ramp from 1.0 → LPIPS_MOVEMENT_BOOST

# Per-dimension weights for 15D discrete action vector
ACTION_WEIGHTS = torch.tensor([
    1.0, 1.0, 1.0, 1.0, 1.0,    # Yaw
    1.0, 1.0, 1.0,               # Pitch
    10.0, 10.0, 10.0, 10.0,      # WASD
    5.0,                          # Jump
    2.0, 2.0                      # Sprint/Sneak
], device=DEVICE)

# --- OPTIMIZATION ---
FILM_LR_MULT = 10.0             # Lowered for smaller internal dim

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v6.2-step0"
MODEL_OUT_PREFIX = "ochre-v6.2"
RESUME_PATH = "" 

LOG_STEPS = 10
IMAGE_LOG_STEPS = 1000 
MILESTONE_SAVE_STEPS = 20000  

# v5.0-style timing / throughput tracking (EMA)
TIMING_EMA_ALPHA = 0.1

# v5.0-style action validation (runs on image logging steps)
ACTION_VALIDATION_STEPS = [1, 5, 10]
ACTION_VISUAL_ROLLOUT_LEN = 30

# ==========================================
# HELPERS
# ==========================================

def compute_gumbel_tau(step: int) -> float:
    frac = min(1.0, step / float(GUMBEL_TAU_STEPS))
    return max(GUMBEL_TAU_MIN, 1.0 - frac * (1.0 - GUMBEL_TAU_MIN))


def compute_token_corrupt_p(step: int) -> float:
    if TOKEN_CORRUPT_MAX_P <= 0:
        return 0.0
    frac = min(1.0, step / float(TOKEN_CORRUPT_RAMP_STEPS))
    return float(frac * TOKEN_CORRUPT_MAX_P)


def compute_lpips_movement_boost(step: int) -> float:
    """Ramp LPIPS movement boost from 1.0 to LPIPS_MOVEMENT_BOOST over LPIPS_MOVEMENT_BOOST_RAMP_STEPS."""
    if LPIPS_MOVEMENT_BOOST_RAMP_STEPS <= 0:
        return LPIPS_MOVEMENT_BOOST
    frac = min(1.0, step / float(LPIPS_MOVEMENT_BOOST_RAMP_STEPS))
    return 1.0 + frac * (LPIPS_MOVEMENT_BOOST - 1.0)

class ARCurriculum:
    """Manages AR Length and Adaptive Brake Logic"""
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

        # v6.2: Hold back AR growth while token diversity is low (collapse-like regime).
        if (
            unique_codes is not None
            and step >= AR_DIVERSITY_GATE_START
            and unique_codes < MIN_UNIQUE_CODES_FOR_AR_GROWTH
        ):
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 2)
            return self.ar_len

        # Only apply brake once AR LPIPS is available; otherwise keep minimum exposure.
        if lpips_ar <= 0 or lpips_tf <= 0:
            return self.ar_len
        
        # Initialize or Update EMA
        if self.tf_ema is None:
            self.tf_ema = lpips_tf
            self.ar_ema = lpips_ar
        else:
            self.tf_ema = self.alpha * self.tf_ema + (1 - self.alpha) * lpips_tf
            self.ar_ema = self.alpha * self.ar_ema + (1 - self.alpha) * lpips_ar
            
        # Brake Logic
        ratio = self.ar_ema / (self.tf_ema + 1e-6)

        prev_ar_len = self.ar_len
        if ratio > self.brake_ratio_upper + 0.05:
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 2)
            if self.ar_len != prev_ar_len:
                print(f"[Curriculum] Brake ENGAGED: Ratio {ratio:.2f} -> Reducing AR to {self.ar_len}")
        elif ratio < self.brake_ratio_lower - 0.05:
            self.ar_len = min(AR_MAX_LEN, self.ar_len + 2)
            if self.ar_len != prev_ar_len:
                print(f"[Curriculum] Brake RELEASED: Ratio {ratio:.2f} -> Increasing AR to {self.ar_len}")
            
        return self.ar_len

class SemanticCodebookLoss(nn.Module):
    def __init__(self, codebook_tensor):
        super().__init__()
        self.register_buffer('codebook', codebook_tensor.clone().detach())

    def forward(self, logits, target_indices, global_step=0):
        tau = max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9)
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        pred_vectors = torch.matmul(probs, self.codebook) 
        target_vectors = F.embedding(target_indices, self.codebook) 
        return F.mse_loss(pred_vectors, target_vectors)

def log_images_to_wandb(vqvae, Z_target, logits, global_step):
    if wandb is None: return
    with torch.no_grad():
        gt_indices = Z_target[:, -1]
        pred_indices = logits.argmax(dim=1)
        gt_rgb = vqvae.decode_code(gt_indices)[:4]
        pred_rgb = vqvae.decode_code(pred_indices)[:4]
        
        vis = torch.cat([gt_rgb, pred_rgb], dim=0) # Stack for grid
        grid = make_grid(vis, nrow=4, normalize=False, value_range=(0, 1))
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        wandb.log({"visuals/reconstruction": wandb.Image(grid_np, caption=f"Top: GT | Bot: Pred (Step {global_step})")}, step=global_step)

def compute_multistep_action_response(model, vqvae, z_start, action_vec, num_steps, device):
    """
    Perform multi-step AR rollout with a fixed action.
    Returns list of decoded RGB frames (length=num_steps).
    """
    with torch.no_grad():
        x_t = model._embed_tokens(z_start)
        h_state = model.init_state(1, device=device)
        temporal_buffer = [] if getattr(model, "temporal_attn", None) is not None else None

        frames = []
        for _ in range(num_steps):
            gammas, betas = model.film(action_vec)  # (L, 1, C, 1, 1)
            logits, h_state = model.step(None, None, h_state, x_t=x_t, gammas_t=gammas, betas_t=betas, temporal_buffer=temporal_buffer)
            pred_tokens = logits.argmax(dim=1)  # (1, H, W)
            pred_rgb = vqvae.decode_code(pred_tokens)[0]  # (3, IMG_H, IMG_W)
            frames.append(pred_rgb.cpu())

            if temporal_buffer is not None:
                temporal_buffer.append(model.temporal_attn.pool_state(h_state[-1].detach()))
                if len(temporal_buffer) > model.temporal_context_len:
                    temporal_buffer.pop(0)

            x_t = model._embed_tokens(pred_tokens)

        return frames

def validate_action_conditioning(model, vqvae, Z_seq, global_step):
    """
    Multi-step action conditioning validation to catch degradation over AR rollouts.
    Logs scalar action-response diffs and a 30-frame visualization grid.
    """
    if wandb is None:
        return {}

    with torch.no_grad():
        device = Z_seq.device
        z_start = Z_seq[0:1, 0]  # (1, H, W)

        test_actions = {
            "static": torch.tensor(encode_action_v5_np(), device=device).unsqueeze(0),
            "camera_left": torch.tensor(encode_action_v5_np(yaw_raw=-0.5), device=device).unsqueeze(0),
            "camera_right": torch.tensor(encode_action_v5_np(yaw_raw=0.5), device=device).unsqueeze(0),
            "move_forward": torch.tensor(encode_action_v5_np(w=1.0), device=device).unsqueeze(0),
        }

        rollout_predictions = {}
        for action_name, action_vec in test_actions.items():
            rollout_predictions[action_name] = {}
            for num_steps in ACTION_VALIDATION_STEPS:
                frames = compute_multistep_action_response(model, vqvae, z_start, action_vec, num_steps, device)
                rollout_predictions[action_name][num_steps] = frames

        metrics = {}
        for num_steps in ACTION_VALIDATION_STEPS:
            static_frame = rollout_predictions["static"][num_steps][-1].flatten()
            camera_l_frame = rollout_predictions["camera_left"][num_steps][-1].flatten()
            camera_r_frame = rollout_predictions["camera_right"][num_steps][-1].flatten()
            move_fwd_frame = rollout_predictions["move_forward"][num_steps][-1].flatten()

            diff_camera_l = (static_frame - camera_l_frame).pow(2).mean().sqrt().item()
            diff_camera_r = (static_frame - camera_r_frame).pow(2).mean().sqrt().item()
            diff_move_fwd = (static_frame - move_fwd_frame).pow(2).mean().sqrt().item()
            action_response = (diff_camera_l + diff_camera_r + diff_move_fwd) / 3

            step_suffix = f"_{num_steps}step" if num_steps > 1 else ""
            metrics[f"action_response/camera_left_diff{step_suffix}"] = diff_camera_l
            metrics[f"action_response/camera_right_diff{step_suffix}"] = diff_camera_r
            metrics[f"action_response/move_forward_diff{step_suffix}"] = diff_move_fwd
            metrics[f"action_response/average{step_suffix}"] = action_response

        rollout_30_predictions = {}
        for action_name, action_vec in test_actions.items():
            frames_30 = compute_multistep_action_response(model, vqvae, z_start, action_vec, ACTION_VISUAL_ROLLOUT_LEN, device)
            rollout_30_predictions[action_name] = frames_30

        num_vis_frames = 6
        vis_frames_30 = []
        for action_name in ["static", "camera_left", "camera_right", "move_forward"]:
            frames = rollout_30_predictions[action_name]
            indices = [int(i * (ACTION_VISUAL_ROLLOUT_LEN - 1) / (num_vis_frames - 1)) for i in range(num_vis_frames)]
            for idx in indices:
                vis_frames_30.append(torch.clamp(frames[idx], 0.0, 1.0))

        vis_tensor_30 = torch.stack(vis_frames_30, dim=0)
        grid_30 = make_grid(vis_tensor_30, nrow=num_vis_frames, normalize=False, value_range=(0, 1), padding=2)
        grid_np_30 = (grid_30.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/action_rollout_30step": wandb.Image(
                grid_np_30,
                caption=f"30-Step AR Rollouts (6 frames shown) | Rows: Static, Cam-L, Cam-R, Move-Fwd | Step {global_step}"
            )
        }, step=global_step)

        return metrics

class GTTokenDataset(Dataset):
    def __init__(self, manifest_path, root_dir, seq_len=6):
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
        # Optimized loading: use mmap always, copy only needed slice
        try:
            # Using mmap_mode='r' prevents loading entire file into RAM
            # We then slice it immediately which triggers the read of just that section
            with np.load(path, mmap_mode='r') as data:
                tokens = data["tokens"][start:start + self.seq_len + 1]
                actions = data["actions"][start:start + self.seq_len]
        except Exception:
            # Fallback for corrupted/standard files
            data = np.load(path)
            tokens = data["tokens"][start:start + self.seq_len + 1]
            actions = data["actions"][start:start + self.seq_len]

        Z_seq = tokens[:-1]
        Z_target = tokens[1:]
        
        return (
            torch.tensor(Z_seq.astype(np.int32), dtype=torch.long),
            torch.tensor(actions.astype(np.float32), dtype=torch.float32),
            torch.tensor(Z_target.astype(np.int32), dtype=torch.long),
            idx, vid_idx, start
        )

# ==========================================
# SETUP
# ==========================================

if wandb:
    wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
                config=dict(batch_size=BATCH_SIZE, lr=LR, max_steps=MAX_STEPS))

# 1. Load VQ-VAE
print(f"📥 Loading VQ-VAE from {VQVAE_PATH}...")
vqvae_ckpt = torch.load(VQVAE_PATH, map_location=DEVICE)
conf = vqvae_ckpt.get("config", {})
vqvae_model = VQVAE(
    embedding_dim=conf.get("embedding_dim", 384),
    num_embeddings=conf.get("codebook_size", 1024)
).to(DEVICE)

# Support both "component-wise" checkpoints (encoder/decoder/quantizer) and unified model_state.
if isinstance(vqvae_ckpt, dict) and {"encoder", "decoder", "quantizer"}.issubset(vqvae_ckpt.keys()):
    vqvae_model.encoder.load_state_dict(vqvae_ckpt["encoder"])
    vqvae_model.decoder.load_state_dict(vqvae_ckpt["decoder"])
    vqvae_model.vq_vae.load_state_dict(vqvae_ckpt["quantizer"])
elif isinstance(vqvae_ckpt, dict) and "model_state" in vqvae_ckpt:
    vqvae_model.load_state_dict(vqvae_ckpt["model_state"], strict=False)
else:
    vqvae_model.load_state_dict(vqvae_ckpt, strict=False)

vqvae_model.eval().requires_grad_(False)

# Codebook setup
codebook = vqvae_model.vq_vae.embedding.clone().detach().t() # (K, D)
codebook_size = codebook.shape[0]

# 2. Init Model (v6.1 Configuration)
model = WorldModelConvFiLM(
    codebook_size=codebook_size,
    embed_dim=256,
    hidden_dim=512,             # v6.1: Up from 384 for better capacity
    n_layers=6,
    action_dim=ACTION_DIM,
    idm_max_span=MAX_IDM_SPAN,
    temporal_context_len=TEMPORAL_CONTEXT_LEN,  # v6.1: For temporal attention
    enable_camera_warp=True,     # v6.2: Explicit spatial transport bias for yaw/pitch
    max_yaw_warp=2,
    max_pitch_warp=2,
    H=18, W=32,
    use_checkpointing=USE_CHECKPOINTING,
    use_residuals=True
).to(DEVICE)

# Optimizer
film_params = [p for n, p in model.named_parameters() if ("film" in n or "action" in n)]
dynamics_params = [p for n, p in model.named_parameters() if ("film" not in n and "action" not in n)]

optimizer = torch.optim.AdamW([
    {'params': dynamics_params, 'lr': LR},
    {'params': film_params, 'lr': LR * FILM_LR_MULT}
], lr=LR)
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

print(f"Training Start: v6.2 (camera warp + token corruption + anti-collapse), LPIPS freq={LPIPS_FREQ}")

model.train()
prev_lpips_tf = 0.0
prev_lpips_ar = 0.0
prev_unique_codes = None

timing_stats = {
    "step_total": None,
    "data_load": None,
    "forward_total": None,
    "embed_film": None,
    "model_step": None,
    "loss_semantic": None,
    "loss_lpips_total": None,
    "loss_lpips_call": None,
    "loss_idm": None,
    "backward_total": None,
    "optimizer_total": None,
}

while global_step < MAX_STEPS:
    t_step_start = time.perf_counter()

    # 1. Update Curriculum
    ar_len = curriculum.update(global_step, prev_lpips_tf, prev_lpips_ar, prev_unique_codes)
    seq_len = max(BASE_SEQ_LEN, ar_len + 1)
    
    # Reload loader if seq_len changes
    if seq_len != prev_seq_len:
        print(f"Resizing Loader: seq_len {prev_seq_len} -> {seq_len}")
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

    t_data_end = time.perf_counter()
        
    Z_seq, A_seq, Z_target, _, _, _ = batch
    Z_seq, A_seq, Z_target = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target.to(DEVICE)
    B, K, H, W = Z_seq.shape

    # 3. Warmup
    if global_step <= WARMUP_STEPS:
        curr_lr = MIN_LR + (global_step/WARMUP_STEPS) * (LR - MIN_LR)
        optimizer.param_groups[0]['lr'] = curr_lr
        optimizer.param_groups[1]['lr'] = curr_lr * FILM_LR_MULT

    optimizer.zero_grad()
    
    # 4. Forward Pass
    t_forward_start = time.perf_counter()
    t_embed_film_start = time.perf_counter()

    # v6.2: Corrupt teacher-forcing inputs (on-manifold) to weaken the "copy Z_t" shortcut.
    token_corrupt_p = compute_token_corrupt_p(global_step)
    if token_corrupt_p > 0:
        Z_seq_tf = Z_seq.clone()
        # Do not corrupt t=0 (bootstraps the hidden state).
        mask = torch.rand((B, K - 1, H, W), device=DEVICE) < token_corrupt_p
        rand_tokens = torch.randint(0, codebook_size, (B, K - 1, H, W), device=DEVICE, dtype=Z_seq_tf.dtype)
        Z_seq_tf[:, 1:] = torch.where(mask, rand_tokens, Z_seq_tf[:, 1:])
    else:
        Z_seq_tf = Z_seq

    with autocast(enabled=(DEVICE == "cuda")):
        # Pre-computes
        X_seq = model.compute_embeddings(Z_seq_tf)
        Gammas, Betas = model.compute_film(A_seq)
        h = model.init_state(B, device=DEVICE)

        t_embed_film_end = time.perf_counter()
        
        # Losses
        loss_sem_list = []
        loss_lpips_tf_list = []
        loss_lpips_ar_list = []
        loss_idm_list = []
        loss_entropy_list = []
        movement_active_counts = []  # v6.2: Track movement-active frames for diagnostics

        ar_cutoff = K - ar_len
        
        # v6.1: Store h states for IDM and temporal attention
        h_buffer = []  # For IDM (detached top-layer states)
        temporal_buffer = []  # For temporal attention

        logits_last = None
        model_step_time_total = 0.0
        loss_sem_time_total = 0.0
        lpips_time_total = 0.0
        lpips_call_count = 0
        loss_idm_time_total = 0.0
        entropy_bonus_weight = ENTROPY_BONUS_WEIGHT if global_step < ENTROPY_BONUS_STEPS else 0.0

        for t in range(K):
            # Teacher Forcing vs AR
            is_ar_step = (t >= ar_cutoff and t > 0)

            if t == 0:
                x_in = X_seq[:, 0]
            elif is_ar_step:
                # Detached AR for stability
                with torch.no_grad():
                    prev_tokens = logits_last.argmax(dim=1)
                    x_in = model.compute_embeddings(prev_tokens.unsqueeze(1))[:, 0]
            else:
                x_in = X_seq[:, t]

            # Step Model (v6.1: pass temporal_buffer for temporal attention)
            t_model_step_start = time.perf_counter()
            logits_t, h = model.step(
                None, A_seq[:, t], h,
                x_t=x_in,
                gammas_t=Gammas[:, :, t],
                betas_t=Betas[:, :, t],
                temporal_buffer=temporal_buffer
            )
            model_step_time_total += (time.perf_counter() - t_model_step_start)
            logits_last = logits_t

            # Store State for IDM (Detached)
            h_buffer.append(h[-1].detach())

            # Update temporal buffer for next step
            if model.temporal_attn is not None and TEMPORAL_CONTEXT_LEN > 0:
                temporal_buffer.append(model.temporal_attn.pool_state(h[-1].detach()))
                if len(temporal_buffer) > TEMPORAL_CONTEXT_LEN:
                    temporal_buffer.pop(0)
            
            # --- Loss Calculation ---
            
            # A. Semantic Loss
            t_sem_start = time.perf_counter()
            logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
            target_flat = Z_target[:, t].reshape(-1)
            loss_sem = semantic_criterion(logits_flat, target_flat, global_step)
            loss_sem_list.append(loss_sem)
            loss_sem_time_total += (time.perf_counter() - t_sem_start)
            
            # B. LPIPS Loss
            if t % LPIPS_FREQ == 0:
                t_lpips_start = time.perf_counter()
                tau = compute_gumbel_tau(global_step)
                probs = F.gumbel_softmax(logits_flat, tau=tau, hard=False, dim=-1)
                probs = probs.reshape(B, H, W, -1)

                if entropy_bonus_weight > 0:
                    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                    # Minimize negative entropy => encourage higher entropy early (anti-collapse).
                    loss_entropy_list.append(-entropy)

                soft_emb = torch.matmul(probs, codebook).permute(0, 3, 1, 2)
                pred_rgb = vqvae_model.decoder(soft_emb)

                with torch.no_grad():
                    tgt_rgb = vqvae_model.decode_code(Z_target[:, t])

                # v6.2: Action-conditional LPIPS reweighting
                # Detect movement-active frames (any WASD/jump/sprint/sneak active)
                movement_active = torch.any(A_seq[:, t, 8:15] > 0.5, dim=-1)  # (B,) bool
                movement_active_counts.append(movement_active.float().sum().item())  # Track for diagnostics
                # Boost LPIPS weight on movement frames (ramped over 30k steps)
                movement_boost = compute_lpips_movement_boost(global_step)
                lpips_weight = torch.where(movement_active,
                                          torch.tensor(movement_boost, device=DEVICE),
                                          torch.tensor(1.0, device=DEVICE))  # (B,)

                lpips_raw = lpips_criterion(pred_rgb * 2 - 1, tgt_rgb * 2 - 1)  # (B, 1, 1, 1)
                lpips_val = (lpips_raw.squeeze() * lpips_weight).mean()  # Weighted average

                # Track TF vs AR separately for curriculum
                if is_ar_step:
                    loss_lpips_ar_list.append(lpips_val)
                else:
                    loss_lpips_tf_list.append(lpips_val)
                lpips_time_total += (time.perf_counter() - t_lpips_start)
                lpips_call_count += 1

            # C. IDM Loss (Variable Span)
            if t >= 1:
                t_idm_start = time.perf_counter()
                # Random lookback k
                k = random.randint(1, min(t, MAX_IDM_SPAN))
                
                # Assert buffer health
                assert len(h_buffer) >= k + 1, "IDM Buffer Underflow"
                
                # Past state (detached) vs Current state (attached via h)
                # Note: h_buffer stores detached, but for current state 'h' we want gradient flow
                h_start = h_buffer[-1-k] 
                h_end = h[-1] # Gradients flow here
                
                dt_tensor = torch.full((B,), k, device=DEVICE, dtype=torch.long)
                pred_action = model.idm(h_start, h_end, dt_tensor)
                
                # Targets
                action_segment = A_seq[:, t-k+1:t+1]
                yaw_target = action_segment[:, :, 0:5].mean(dim=1)
                pitch_target = action_segment[:, :, 5:8].mean(dim=1)
                bin_target = action_segment[:, :, 8:15].mean(dim=1)
                
                # Discrete Losses
                ly = -(yaw_target * F.log_softmax(pred_action[:, 0:5], -1)).sum(-1).mean()
                lp = -(pitch_target * F.log_softmax(pred_action[:, 5:8], -1)).sum(-1).mean()
                lb = (F.binary_cross_entropy_with_logits(pred_action[:, 8:15], bin_target, reduction='none') * ACTION_WEIGHTS[8:15]).mean()
                
                loss_idm_list.append(ly + lp + lb)
                loss_idm_time_total += (time.perf_counter() - t_idm_start)

        # Aggregate
        loss_sem_total = torch.stack(loss_sem_list).mean()
        loss_lpips_tf = torch.stack(loss_lpips_tf_list).mean() if loss_lpips_tf_list else torch.tensor(0.0, device=DEVICE)
        loss_lpips_ar = torch.stack(loss_lpips_ar_list).mean() if loss_lpips_ar_list else torch.tensor(0.0, device=DEVICE)
        loss_idm_total = torch.stack(loss_idm_list).mean() if loss_idm_list else torch.tensor(0.0, device=DEVICE)
        loss_entropy_total = torch.stack(loss_entropy_list).mean() if loss_entropy_list else torch.tensor(0.0, device=DEVICE)

        # Combined LPIPS (TF + AR upweighted)
        loss_lpips_total = loss_lpips_tf + AR_LOSS_WEIGHT * loss_lpips_ar

        # Track metrics for Curriculum
        if loss_lpips_tf_list:
            prev_lpips_tf = loss_lpips_tf.item()
        if loss_lpips_ar_list:
            prev_lpips_ar = loss_lpips_ar.item()

        total_loss = (SEMANTIC_WEIGHT * loss_sem_total) + \
                     (LPIPS_WEIGHT * loss_lpips_total) + \
                     (IDM_LOSS_WEIGHT * loss_idm_total) + \
                     (entropy_bonus_weight * loss_entropy_total)

    t_forward_end = time.perf_counter()

    # 5. Backward
    t_backward_start = time.perf_counter()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    # v5.0-style grad norms (FiLM vs dynamics) for imbalance diagnosis
    grad_film = 0.0
    grad_dynamics = 0.0
    if global_step % LOG_STEPS == 0:
        film_params_with_grad = [p for n, p in model.named_parameters() if ("film" in n or "action" in n) and p.grad is not None]
        dynamics_params_with_grad = [p for n, p in model.named_parameters() if ("film" not in n and "action" not in n) and p.grad is not None]
        if film_params_with_grad:
            grad_film = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in film_params_with_grad]), 2).item()
        if dynamics_params_with_grad:
            grad_dynamics = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in dynamics_params_with_grad]), 2).item()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    t_backward_end = time.perf_counter()

    t_opt_end = time.perf_counter()

    # --- Timing EMA update ---
    data_time = t_data_end - t_step_start
    forward_time = t_forward_end - t_forward_start
    step_time = t_opt_end - t_step_start
    backward_time = t_backward_end - t_backward_start
    optimizer_time = t_opt_end - t_backward_end

    lpips_call_time = (lpips_time_total / lpips_call_count) if lpips_call_count > 0 else 0.0

    def _ema(prev, curr):
        return curr if prev is None else ((1 - TIMING_EMA_ALPHA) * prev + TIMING_EMA_ALPHA * curr)

    timing_stats["step_total"] = _ema(timing_stats["step_total"], step_time)
    timing_stats["data_load"] = _ema(timing_stats["data_load"], data_time)
    timing_stats["forward_total"] = _ema(timing_stats["forward_total"], forward_time)
    timing_stats["embed_film"] = _ema(timing_stats["embed_film"], t_embed_film_end - t_embed_film_start)
    timing_stats["model_step"] = _ema(timing_stats["model_step"], model_step_time_total)
    timing_stats["loss_semantic"] = _ema(timing_stats["loss_semantic"], loss_sem_time_total)
    timing_stats["loss_lpips_total"] = _ema(timing_stats["loss_lpips_total"], lpips_time_total)
    timing_stats["loss_lpips_call"] = _ema(timing_stats["loss_lpips_call"], lpips_call_time)
    timing_stats["loss_idm"] = _ema(timing_stats["loss_idm"], loss_idm_time_total)
    timing_stats["backward_total"] = _ema(timing_stats["backward_total"], backward_time)
    timing_stats["optimizer_total"] = _ema(timing_stats["optimizer_total"], optimizer_time)

    # 6. Log
    # Diagnostics that were present in v5.0 (token diversity + confidence)
    with torch.no_grad():
        probs_last = F.softmax(logits_last.float(), dim=1) if logits_last is not None else None
        if probs_last is not None:
            entropy = -(probs_last * torch.log(probs_last + 1e-9)).sum(dim=1).mean()
            confidence = probs_last.max(dim=1)[0].mean()
            unique_codes = logits_last.argmax(dim=1).unique().numel()
            top1_conf = probs_last.max(dim=1)[0]
            confidence_std = top1_conf.std()
            confidence_min = top1_conf.min()
        else:
            entropy = torch.tensor(0.0)
            confidence = torch.tensor(0.0)
            unique_codes = 0
            confidence_std = torch.tensor(0.0)
            confidence_min = torch.tensor(0.0)

        # Action diagnostics (FiLM magnitude proxy)
        gamma_magnitude = Gammas.abs().mean().item()
        beta_magnitude = Betas.abs().mean().item()
        action_sensitivity = (gamma_magnitude + beta_magnitude) / 2
        prev_unique_codes = int(unique_codes)

        # v6.2: Movement-active frame percentage (for LPIPS reweighting diagnostics)
        if len(movement_active_counts) > 0:
            total_frames_checked = len(movement_active_counts) * B
            total_movement_active = sum(movement_active_counts)
            movement_active_pct = (total_movement_active / total_frames_checked) * 100 if total_frames_checked > 0 else 0.0
        else:
            movement_active_pct = 0.0

    if global_step % LOG_STEPS == 0:
        throughput = 1.0 / timing_stats["step_total"] if timing_stats["step_total"] and timing_stats["step_total"] > 0 else 0.0
        lpips_call_ms = int((timing_stats["loss_lpips_call"] or 0.0) * 1000)
        print(
            f"[Step {global_step}] Loss: {total_loss.item():.4f} | "
            f"Sem: {loss_sem_total.item():.4f} | LPIPS: {loss_lpips_total.item():.4f} | IDM: {loss_idm_total.item():.4f} | "
            f"{throughput:.2f} steps/s | Total: {(timing_stats['step_total'] or 0.0)*1000:.1f}ms "
            f"(Data: {(timing_stats['data_load'] or 0.0)*1000:.0f}ms, "
            f"Emb+FiLM: {(timing_stats['embed_film'] or 0.0)*1000:.0f}ms, "
            f"Step: {(timing_stats['model_step'] or 0.0)*1000:.0f}ms, "
            f"Sem: {(timing_stats['loss_semantic'] or 0.0)*1000:.0f}ms, "
            f"LPIPS: {(timing_stats['loss_lpips_total'] or 0.0)*1000:.0f}ms"
            f"{f' ({lpips_call_ms}ms/call)' if lpips_call_ms > 0 else ''}, "
            f"IDM: {(timing_stats['loss_idm'] or 0.0)*1000:.0f}ms, "
            f"Bwd: {(timing_stats['backward_total'] or 0.0)*1000:.0f}ms, "
            f"Opt: {(timing_stats['optimizer_total'] or 0.0)*1000:.0f}ms)"
        )
        if wandb:
            wandb.log({
                "train/loss": total_loss.item(),
                "train/loss_texture": loss_sem_total.item(),
                "train/loss_lpips": loss_lpips_total.item(),
                "train/loss_lpips_tf": loss_lpips_tf.item(),
                "train/loss_lpips_ar": loss_lpips_ar.item(),
                "train/loss_idm": loss_idm_total.item(),
                "train/loss_entropy": loss_entropy_total.item(),
                "train/gumbel_tau": compute_gumbel_tau(global_step),
                "train/entropy_bonus_weight": float(ENTROPY_BONUS_WEIGHT if global_step < ENTROPY_BONUS_STEPS else 0.0),
                "train/token_corrupt_p": float(token_corrupt_p),
                "train/lpips_movement_boost": compute_lpips_movement_boost(global_step),
                "train/movement_active_pct": movement_active_pct,
                "curriculum/seq_len": seq_len,
                "curriculum/ar_len": ar_len,
                "curriculum/ar_cutoff": ar_cutoff,
                "curriculum/lpips_ratio": prev_lpips_ar / (prev_lpips_tf + 1e-6),
                "train/grad_norm": float(norm),
                "grad/film_norm": grad_film,
                "grad/dynamics_norm": grad_dynamics,
                "action_diagnostics/film_gamma_magnitude": gamma_magnitude,
                "action_diagnostics/film_beta_magnitude": beta_magnitude,
                "action_diagnostics/sensitivity": action_sensitivity,
                "action_diagnostics/action_magnitude": A_seq.abs().mean().item(),
                "diagnostics/entropy": float(entropy),
                "diagnostics/confidence": float(confidence),
                "diagnostics/unique_codes": unique_codes,
                "diagnostics/confidence_std": float(confidence_std),
                "diagnostics/confidence_min": float(confidence_min),
                "timing/step_total_ms": (timing_stats["step_total"] or 0.0) * 1000,
                "timing/loss_lpips_call_ms": (timing_stats["loss_lpips_call"] or 0.0) * 1000,
                "timing/throughput_steps_per_sec": throughput,
            }, step=global_step)
            
    if global_step % IMAGE_LOG_STEPS == 0:
        log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)
        action_metrics = validate_action_conditioning(model, vqvae_model, Z_seq, global_step)
        if wandb and action_metrics:
            wandb.log(action_metrics, step=global_step)

    if global_step % MILESTONE_SAVE_STEPS == 0:
        torch.save(model.state_dict(), f"./checkpoints/{MODEL_OUT_PREFIX}-step{global_step}.pt")

    global_step += 1
