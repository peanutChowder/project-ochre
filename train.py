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
GUMBEL_TAU_STEPS = 20000 

AR_LOSS_WEIGHT = 2.5     

# --- CURRICULUM ---
BASE_SEQ_LEN = 16               
CURRICULUM_AR = True             
AR_WARMUP_STEPS = 5000          
AR_MIN_LEN = 10                 
AR_MAX_LEN = 20                 

# --- IDM CONFIG ---
IDM_LOSS_WEIGHT = 1.0   # Increased from 0.5 to compensate for lack of Ranking
MAX_IDM_SPAN = 5       
ACTION_DIM = 15

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
RUN_NAME = "v6.1-step0"
MODEL_OUT_PREFIX = "ochre-v6.1"
RESUME_PATH = "" 

LOG_STEPS = 10
IMAGE_LOG_STEPS = 1000 
MILESTONE_SAVE_STEPS = 20000  

# ==========================================
# HELPERS
# ==========================================

class ARCurriculum:
    """Manages AR Length and Adaptive Brake Logic"""
    def __init__(self):
        self.ar_len = AR_MIN_LEN
        self.tf_ema = None
        self.ar_ema = None
        self.alpha = 0.98
        self.brake_ratio_upper = 2.5
        self.brake_ratio_lower = 1.6
        
    def update(self, step, lpips_tf, lpips_ar):
        if step < AR_WARMUP_STEPS:
            return 0

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
        
        if ratio > self.brake_ratio_upper + 0.05:
            self.ar_len = max(AR_MIN_LEN, self.ar_len - 2)
            print(f"[Curriculum] Brake ENGAGED: Ratio {ratio:.2f} -> Reducing AR to {self.ar_len}")
        elif ratio < self.brake_ratio_lower - 0.05:
            self.ar_len = min(AR_MAX_LEN, self.ar_len + 2)
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

print(f"Training Start: v6.1 (512-dim, SeparateFiLM, TemporalAttn), LPIPS freq={LPIPS_FREQ}")

model.train()
prev_lpips_tf = 0.0
prev_lpips_ar = 0.0

while global_step < MAX_STEPS:
    # 1. Update Curriculum
    ar_len = curriculum.update(global_step, prev_lpips_tf, prev_lpips_ar)
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
    with autocast(enabled=(DEVICE == "cuda")):
        # Pre-computes
        X_seq = model.compute_embeddings(Z_seq)
        Gammas, Betas = model.compute_film(A_seq)
        h = model.init_state(B, device=DEVICE)
        
        # Losses
        loss_sem_list = []
        loss_lpips_tf_list = []
        loss_lpips_ar_list = []
        loss_idm_list = []
        
        ar_cutoff = K - ar_len
        
        # v6.1: Store h states for IDM and temporal attention
        h_buffer = []  # For IDM (detached top-layer states)
        temporal_buffer = []  # For temporal attention

        logits_last = None

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
            logits_t, h = model.step(
                None, None, h,
                x_t=x_in,
                gammas_t=Gammas[:, :, t],
                betas_t=Betas[:, :, t],
                temporal_buffer=temporal_buffer
            )
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
            logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
            target_flat = Z_target[:, t].reshape(-1)
            loss_sem = semantic_criterion(logits_flat, target_flat, global_step)
            loss_sem_list.append(loss_sem)
            
            # B. LPIPS Loss
            if t % LPIPS_FREQ == 0:
                tau = max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9)
                probs = F.gumbel_softmax(logits_flat, tau=tau, hard=False, dim=-1)
                probs = probs.reshape(B, H, W, -1)

                soft_emb = torch.matmul(probs, codebook).permute(0, 3, 1, 2)
                pred_rgb = vqvae_model.decoder(soft_emb)

                with torch.no_grad():
                    tgt_rgb = vqvae_model.decode_code(Z_target[:, t])

                lpips_val = lpips_criterion(pred_rgb * 2 - 1, tgt_rgb * 2 - 1).mean()

                # Track TF vs AR separately for curriculum
                if is_ar_step:
                    loss_lpips_ar_list.append(lpips_val)
                else:
                    loss_lpips_tf_list.append(lpips_val)

            # C. IDM Loss (Variable Span)
            if t >= 1:
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

        # Aggregate
        loss_sem_total = torch.stack(loss_sem_list).mean()
        loss_lpips_tf = torch.stack(loss_lpips_tf_list).mean() if loss_lpips_tf_list else torch.tensor(0.0, device=DEVICE)
        loss_lpips_ar = torch.stack(loss_lpips_ar_list).mean() if loss_lpips_ar_list else torch.tensor(0.0, device=DEVICE)
        loss_idm_total = torch.stack(loss_idm_list).mean() if loss_idm_list else torch.tensor(0.0, device=DEVICE)

        # Combined LPIPS (TF + AR upweighted)
        loss_lpips_total = loss_lpips_tf + AR_LOSS_WEIGHT * loss_lpips_ar

        # Track metrics for Curriculum
        if loss_lpips_tf_list:
            prev_lpips_tf = loss_lpips_tf.item()
        if loss_lpips_ar_list:
            prev_lpips_ar = loss_lpips_ar.item()

        total_loss = (SEMANTIC_WEIGHT * loss_sem_total) + \
                     (LPIPS_WEIGHT * loss_lpips_total) + \
                     (IDM_LOSS_WEIGHT * loss_idm_total)

    # 5. Backward
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # 6. Log
    if global_step % LOG_STEPS == 0:
        print(f"[Step {global_step}] Loss: {total_loss.item():.4f} | Sem: {loss_sem_total.item():.4f} | IDM: {loss_idm_total.item():.4f}")
        if wandb:
            wandb.log({
                "train/loss": total_loss.item(),
                "train/loss_texture": loss_sem_total.item(),
                "train/loss_lpips": loss_lpips_total.item(),
                "train/loss_lpips_tf": loss_lpips_tf.item(),
                "train/loss_lpips_ar": loss_lpips_ar.item(),
                "train/loss_idm": loss_idm_total.item(),
                "curriculum/ar_len": ar_len,
                "curriculum/lpips_ratio": prev_lpips_ar / (prev_lpips_tf + 1e-6),
                "train/grad_norm": float(norm)
            }, step=global_step)
            
    if global_step % IMAGE_LOG_STEPS == 0:
        log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)

    if global_step % MILESTONE_SAVE_STEPS == 0:
        torch.save(model.state_dict(), f"./checkpoints/{MODEL_OUT_PREFIX}-step{global_step}.pt")

    global_step += 1
