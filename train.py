#!/usr/bin/env python3
"""
Full World Model Training Script with Semantic & Spatial Loss.
------------------------------------------------------------
Checklist before running:
1. Ensure 'WorldModelConvFiLM' class is defined/imported.
2. Update VQVAE_PATH to point to your trained 'vqvae.pt'.
3. Update DATA_DIR to your dataset location.
"""

import os, time, json, math, numpy as np, torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import checkpoint
from torchvision.utils import make_grid

# World Model and VQ-VAE imports (required for standalone execution)
from vq_vae.vq_vae import VQVAE
from model_convGru import WorldModelConvFiLM

import wandb
if 'WANDB_API_KEY' in os.environ:
    wandb.login(key=os.environ['WANDB_API_KEY'])
else:
    raise Exception("No wandb key found")


# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PATHS (UPDATE THESE) ---
DATA_DIR = "../preprocessedv4"
VQVAE_PATH = "./checkpoints/vqvae_v2.1.6__epoch100.pt" 
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 28
EPOCHS = 50
LR = 3e-5             
WARMUP_STEPS = 500   
MIN_LR = 1e-6         
USE_CHECKPOINTING = False 

# --- LOSS WEIGHTS ---
# v4.6.5: Focus on perceptual sharpness via LPIPS
# Only 2 loss components: Semantic + LPIPS (neighbor removed)
SEMANTIC_WEIGHT = 10.0   # Embedding MSE (token prediction accuracy)
LPIPS_WEIGHT = 2.0       # Perceptual loss (sharpness, visual quality) - v4.6.5: 1.0 -> 2.0
                         # Combined with LPIPS_FREQ=1, this makes LPIPS the dominant signal
LPIPS_FREQ = 1           # v4.6.5: Every step (was 5) - eliminates periodic flashing artifact
GUMBEL_TAU_STEPS = 20000 # Gumbel-Softmax annealing: 1.0→0.1 over this many steps

# --- CURRICULUM ---
CURRICULUM_SEQ_LEN = False      # Keep disabled (GPU memory risk)
BASE_SEQ_LEN = 20               # v4.5.1 OOM fix: Increased from 16 to accommodate AR rollout
MAX_SEQ_LEN = 50                # Keep same
SEQ_LEN_INCREASE_STEPS = 5000   # Not used (SEQ_LEN disabled)

# v4.7.0: Restore v4.6.4 AR curriculum (fixes action conditioning via exposure to AR rollout)
CURRICULUM_AR = True             # Re-enabled: AR rollout during training
AR_START_STEP = 5000            # v4.6.4: Start AR after model stabilizes
AR_RAMP_STEPS = 10000           # v4.6.4: Ramp AR length gradually
AR_ROLLOUT_MAX = 25             # v4.6.4: Maximum AR rollout length (restored from v4.6.6's 0)
AR_VALIDATION_FREQ = 500        # v4.6.6: Validate AR every N steps
AR_VALIDATION_STEPS = 5         # v4.6.6: Number of AR rollouts per validation

# v4.6.6: Single-step AR mix for noise robustness (Option 2 from efficient-ar-robustness.md)
AR_MIX_ENABLED = True           # Enable occasional AR step during training
AR_MIX_PROB = 0.05              # 5% of steps use previous prediction (~0.5% overhead)

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v4.7.0-step0"
MODEL_OUT_PREFIX = "ochre-v4.7.0"
RESUME_PATH = ""

LOG_STEPS = 10
IMAGE_LOG_STEPS = 500 # Log visual reconstruction every N steps
REPORT_SPIKES = True
SPIKE_LOSS = 15.0
SPIKE_GRAD = 400
EMERGENCY_SAVE_INTERVAL_HRS = 11.8
MILESTONE_SAVE_STEPS = 10000  # v4.6.6: Save checkpoint every N steps

# v4.6.6: Timing & throughput tracking
TIMING_EMA_ALPHA = 0.1  # Smoothing factor for exponential moving average (0.1 = ~10 step window)

# ==========================================
# LOSS & HELPER FUNCTIONS
# ==========================================

class SemanticCodebookLoss(nn.Module):
    """
    v4.6.0: Computes MSE in Embedding Space with Gumbel-Softmax.
    Prevents "Gray Soup" by rewarding visually similar textures.
    Gumbel-Softmax forces discrete commitments during forward pass.
    """
    def __init__(self, codebook_tensor):
        super().__init__()
        # codebook_tensor: (Num_Embeddings, Embedding_Dim)
        self.register_buffer('codebook', codebook_tensor.clone().detach())
        self.codebook_size = codebook_tensor.shape[0]

    def forward(self, logits, target_indices, global_step=0):
        # v4.6.0: Gumbel-Softmax with annealing
        # tau: 1.0 (soft) → 0.1 (hard) over GUMBEL_TAU_STEPS
        tau = max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9)

        # 1. Gumbel-Softmax -> Discrete (hard=True) selection
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        pred_vectors = torch.matmul(probs, self.codebook) # (B*S, Emb_Dim)

        # 2. Ground Truth Vector
        target_vectors = F.embedding(target_indices, self.codebook) # (B*S, Emb_Dim)

        # 3. MSE
        return F.mse_loss(pred_vectors, target_vectors)

# --- REMOVED LOSS FUNCTIONS ---
# v4.5 → v4.6.0: Removed entropy_regularization_loss, sharpness_loss, temporal_consistency_loss
#                (caused conflicting gradients and mode collapse in v4.5.2)
# v4.6.5: Removed neighborhood_token_loss (NEIGHBOR_WEIGHT=0 since v4.6.4, unused computation)

def log_images_to_wandb(vqvae, Z_target, logits, global_step):
    """
    Visualizes Ground Truth vs Prediction.
    Shows interleaved GT/Pred pairs for easier comparison.
    """
    if wandb is None: return
    with torch.no_grad():
        # Take the last frame of the sequence
        gt_indices = Z_target[:, -1]  # (B, H, W)
        pred_indices = logits.argmax(dim=1)  # (B, H, W)

        # Decode - VQ-VAE decoder outputs in [0, 1] range (sigmoid activation)
        gt_rgb = vqvae.decode_code(gt_indices)  # (B, 3, IMAGE_H, IMAGE_W)
        pred_rgb = vqvae.decode_code(pred_indices)  # (B, 3, IMAGE_H, IMAGE_W)

        # Visualize first 4 samples
        n = min(4, gt_rgb.size(0))
        vis_gt = gt_rgb[:n].detach().cpu()
        vis_pred = pred_rgb[:n].detach().cpu()

        # Clamp to [0, 1] range to ensure proper visualization
        vis_gt = torch.clamp(vis_gt, 0.0, 1.0)
        vis_pred = torch.clamp(vis_pred, 0.0, 1.0)

        # Interleave GT and Pred for side-by-side comparison
        # Pattern: [GT0, Pred0, GT1, Pred1, GT2, Pred2, GT3, Pred3]
        interleaved = []
        for i in range(n):
            interleaved.append(vis_gt[i])
            interleaved.append(vis_pred[i])
        interleaved_tensor = torch.stack(interleaved, dim=0)

        # Create grid: 2 columns (GT | Pred), n rows (samples)
        grid = make_grid(interleaved_tensor, nrow=2, normalize=False, value_range=(0, 1), padding=2)

        # Convert from (C, H, W) to (H, W, C) and scale to [0, 255] for wandb
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/reconstruction": wandb.Image(grid_np, caption=f"Left: GT | Right: Pred (Step {global_step})")
        }, step=global_step)


def log_ar_rollout_to_wandb(model, vqvae, Z_seq, A_seq, Z_target, global_step, ar_len):
    """
    v4.7.0: Enhanced AR rollout visualization.
    Displays GT vs AR predictions at current AR rollout length to catch action conditioning issues early.

    Args:
        model: World model
        vqvae: VQ-VAE decoder
        Z_seq: (B, K, H, W) input token sequence
        A_seq: (B, K, A) action sequence (ground truth from context)
        Z_target: (B, K, H, W) target token sequence
        global_step: Current training step
        ar_len: Current AR rollout length
    """
    if wandb is None or ar_len == 0: return

    with torch.no_grad():
        B, K, H, W = Z_seq.shape
        device = Z_seq.device

        # Use first sample from batch for visualization
        z_seq = Z_seq[0:1]  # (1, K, H, W)
        a_seq = A_seq[0:1]  # (1, K, A)
        z_target = Z_target[0:1]  # (1, K, H, W)

        # Pre-compute embeddings and FiLM for efficiency
        X_seq = model.compute_embeddings(z_seq)
        Gammas_seq, Betas_seq = model.compute_film(a_seq)

        # Determine where AR rollout starts (last ar_len frames)
        ar_start = K - ar_len

        # Teacher-forced warmup (frames 0 to ar_start-1)
        h_state = model.init_state(1, device=device)
        for t in range(ar_start):
            x_t = X_seq[:, t]
            g_t = Gammas_seq[:, :, t]
            b_t = Betas_seq[:, :, t]
            _, h_state = model.step(None, None, h_state, x_t=x_t, gammas_t=g_t, betas_t=b_t)

        # AR rollout (frames ar_start to K-1)
        ar_pred_frames = []
        gt_frames = []
        x_prev = X_seq[:, ar_start-1] if ar_start > 0 else X_seq[:, 0]

        for t in range(ar_start, K):
            g_t = Gammas_seq[:, :, t]
            b_t = Betas_seq[:, :, t]

            # AR prediction
            logits_t, h_state = model.step(None, None, h_state, x_t=x_prev, gammas_t=g_t, betas_t=b_t)
            pred_tokens = logits_t.argmax(dim=1)  # (1, H, W)

            # Decode predictions and GT
            pred_rgb = vqvae.decode_code(pred_tokens)[0]  # (3, IMG_H, IMG_W)
            gt_rgb = vqvae.decode_code(z_target[:, t])[0]  # (3, IMG_H, IMG_W)

            ar_pred_frames.append(pred_rgb.cpu())
            gt_frames.append(gt_rgb.cpu())

            # Update for next step
            x_prev = model._embed_tokens(pred_tokens)

        # Create visualization: show every 5th frame or all if < 5
        num_ar_frames = len(ar_pred_frames)
        if num_ar_frames <= 5:
            # Show all frames
            indices = list(range(num_ar_frames))
        else:
            # Show evenly spaced frames (first, middle points, last)
            indices = [0, num_ar_frames//4, num_ar_frames//2, 3*num_ar_frames//4, num_ar_frames-1]

        # Build interleaved GT/Pred grid
        vis_frames = []
        for idx in indices:
            gt_frame = torch.clamp(gt_frames[idx], 0.0, 1.0)
            pred_frame = torch.clamp(ar_pred_frames[idx], 0.0, 1.0)
            vis_frames.append(gt_frame)
            vis_frames.append(pred_frame)

        vis_tensor = torch.stack(vis_frames, dim=0)
        grid = make_grid(vis_tensor, nrow=2, normalize=False, value_range=(0, 1), padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/ar_rollout": wandb.Image(
                grid_np,
                caption=f"AR Rollout (len={ar_len}) | Left: GT | Right: AR Pred | Step {global_step}"
            )
        }, step=global_step)


def validate_action_conditioning(model, vqvae, Z_seq, A_seq, Z_target, global_step):
    """
    v4.7.0: Action-specific validation to catch action conditioning failures.
    Tests model response to different action types: static, camera movements, player movement.

    Args:
        model: World model
        vqvae: VQ-VAE decoder
        Z_seq: (B, K, H, W) input token sequence
        A_seq: (B, K, A) action sequence
        Z_target: (B, K, H, W) target token sequence
        global_step: Current training step

    Returns:
        dict: Metrics for different action conditions
    """
    if wandb is None: return {}

    with torch.no_grad():
        device = Z_seq.device
        # Use first sample
        z_start = Z_seq[0:1, 0]  # (1, H, W) - starting frame
        x_start = model._embed_tokens(z_start)
        h_start = model.init_state(1, device=device)

        # Action format: [yaw, pitch, move_x, move_z, action_5] (5-dim)
        # Test different action conditions
        test_actions = {
            'static': torch.zeros(1, 5, device=device),  # No movement
            'camera_left': torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]], device=device),  # Yaw left
            'camera_right': torch.tensor([[-0.5, 0.0, 0.0, 0.0, 0.0]], device=device),  # Yaw right
            'move_forward': torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0]], device=device),  # Forward
        }

        # Predict from same starting state with different actions
        predictions = {}
        for action_name, action_vec in test_actions.items():
            # Get FiLM parameters for this action
            gammas, betas = model.film(action_vec)  # (L, 1, C, 1, 1)

            # Single-step prediction
            logits, _ = model.step(None, None, h_start, x_t=x_start, gammas_t=gammas, betas_t=betas)
            pred_tokens = logits.argmax(dim=1)  # (1, H, W)
            pred_rgb = vqvae.decode_code(pred_tokens)[0]  # (3, IMG_H, IMG_W)

            predictions[action_name] = pred_rgb.cpu()

        # Visualize: show all action-conditioned predictions side-by-side
        vis_frames = []
        for action_name in ['static', 'camera_left', 'camera_right', 'move_forward']:
            pred_frame = torch.clamp(predictions[action_name], 0.0, 1.0)
            vis_frames.append(pred_frame)

        vis_tensor = torch.stack(vis_frames, dim=0)
        grid = make_grid(vis_tensor, nrow=4, normalize=False, value_range=(0, 1), padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        wandb.log({
            "visuals/action_conditioning": wandb.Image(
                grid_np,
                caption=f"Action Test | Static | Camera L | Camera R | Move Fwd | Step {global_step}"
            )
        }, step=global_step)

        # Compute diversity metrics: how different are action-conditioned predictions?
        static_pred = predictions['static'].flatten()
        camera_l_pred = predictions['camera_left'].flatten()
        camera_r_pred = predictions['camera_right'].flatten()
        move_fwd_pred = predictions['move_forward'].flatten()

        # L2 distance between predictions (higher = better action response)
        diff_camera_l = (static_pred - camera_l_pred).pow(2).mean().sqrt().item()
        diff_camera_r = (static_pred - camera_r_pred).pow(2).mean().sqrt().item()
        diff_move_fwd = (static_pred - move_fwd_pred).pow(2).mean().sqrt().item()

        # Average action response magnitude
        action_response = (diff_camera_l + diff_camera_r + diff_move_fwd) / 3

        return {
            'action_response/camera_left_diff': diff_camera_l,
            'action_response/camera_right_diff': diff_camera_r,
            'action_response/move_forward_diff': diff_move_fwd,
            'action_response/average': action_response,
        }


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
        try:
            data = np.load(path, mmap_mode='r') 
        except ValueError:
            data = np.load(path)
        
        # Slicing
        tokens = data["tokens"]   
        actions = data["actions"] 

        Z_seq = tokens[start:start + self.seq_len]
        A_seq = actions[start:start + self.seq_len]
        Z_target_seq = tokens[start + 1:start + self.seq_len + 1]
        
        return (
            torch.tensor(np.array(Z_seq), dtype=torch.long),        
            torch.tensor(np.array(A_seq), dtype=torch.float32),     
            torch.tensor(np.array(Z_target_seq), dtype=torch.long), 
            idx, vid_idx, start                
        )

if wandb:
    wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
                config=dict(batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS))

# A. Load VQ-VAE (Frozen) for Loss & Vis
print(f"📥 Loading VQ-VAE from {VQVAE_PATH}...")
if not os.path.exists(VQVAE_PATH):
    raise FileNotFoundError(f"VQVAE checkpoint not found at {VQVAE_PATH}")

vqvae_ckpt = torch.load(VQVAE_PATH, map_location=DEVICE)

# Instantiate VQVAE using the config found in the checkpoint or defaults
conf = vqvae_ckpt.get("config", {})
vqvae_model = VQVAE(
    embedding_dim=conf.get("embedding_dim", 384),
    num_embeddings=conf.get("codebook_size", 1024)
).to(DEVICE)

# v4.5.2: Fix VQ-VAE loading - checkpoint has separate encoder/quantizer/decoder dicts
if "encoder" in vqvae_ckpt and "decoder" in vqvae_ckpt and "quantizer" in vqvae_ckpt:
    # Load component-wise (v2.1.6 format)
    vqvae_model.encoder.load_state_dict(vqvae_ckpt["encoder"])
    vqvae_model.decoder.load_state_dict(vqvae_ckpt["decoder"])
    # Load quantizer state (embedding, cluster_size, embedding_avg)
    for key, value in vqvae_ckpt["quantizer"].items():
        if hasattr(vqvae_model.vq_vae, key):
            getattr(vqvae_model.vq_vae, key).copy_(value)
    print("✅ VQ-VAE loaded from component-wise checkpoint")
elif "model_state" in vqvae_ckpt:
    # Unified state dict format
    vqvae_model.load_state_dict(vqvae_ckpt["model_state"], strict=False)
    print("✅ VQ-VAE loaded from model_state")
else:
    # Try loading directly (legacy fallback)
    vqvae_model.load_state_dict(vqvae_ckpt, strict=False)
    print("⚠️  VQ-VAE loaded from root dict (legacy)")

vqvae_model.eval()
vqvae_model.requires_grad_(False)

# B. Extract codebook info from VQ-VAE
# VQ-VAE stores embedding as (embedding_dim, num_embeddings), but SemanticCodebookLoss expects (num_embeddings, embedding_dim)
codebook_raw = vqvae_model.vq_vae.embedding.clone().detach()  # (D, K)
codebook = codebook_raw.t()  # Transpose to (K, D)
codebook_size = codebook.shape[0]  # Now correctly extracts num_embeddings
print(f"VQ-VAE codebook shape (after transpose): {codebook.shape}, size: {codebook_size}")

# C. Initialize World Model with correct codebook size
model = WorldModelConvFiLM(
    codebook_size=codebook_size,
    action_dim=5,
    H=18,
    W=32,
    use_checkpointing=USE_CHECKPOINTING,
    zero_init_head=False  # Disable zero-init to prevent mode collapse at cold start
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

# D. Setup Semantic Loss with loaded codebook
semantic_criterion = SemanticCodebookLoss(codebook).to(DEVICE)
print(f"Semantic Loss Initialized with codebook shape {codebook.shape}")

# E. Setup LPIPS Perceptual Loss (v4.6.0)

import lpips
lpips_criterion = lpips.LPIPS(net='alex').to(DEVICE)
lpips_criterion.eval()  # Frozen, only for loss computation
lpips_criterion.requires_grad_(False)
print(f"✅ LPIPS Loss Initialized (AlexNet backend)")


# F. Resume Logic
global_step = 0
start_epoch = 1
if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f" Resuming training from {RESUME_PATH}")
    ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)

def save_checkpoint(epoch, global_step, is_emergency=False, is_milestone=False):
    # v4.6.6: Step-based naming instead of epoch-based
    if is_emergency:
        save_name = f"{MODEL_OUT_PREFIX}-step{global_step}-emergency.pt"
    elif is_milestone:
        # Milestone saves every 10k steps
        save_name = f"{MODEL_OUT_PREFIX}-step{global_step}.pt"
    else:
        # End of epoch saves (legacy, less important)
        save_name = f"{MODEL_OUT_PREFIX}-epoch{epoch}.pt"

    save_path = f"/kaggle/working/{save_name}"
    print(f"⏰ Saving checkpoint to {save_path}")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, save_path)


# --- TRAINING ---
last_emergency_save_time = time.time()

# v4.6.6: Timing tracking with exponential moving average
timing_stats = {
    'step_total': None,      # Total time per step (EMA)
    'data_load': None,       # Time to load batch
    'forward': None,         # Forward pass time
    'lpips': None,           # LPIPS computation time
    'backward': None,        # Backward pass time
    'optimizer': None,       # Optimizer step time
    'ar_validation': None,   # AR validation time
}

# Helper function to compute curriculum params
def compute_curriculum_params(step):
    ar_len = 0
    if CURRICULUM_AR and step > AR_START_STEP:
        progress = (step - AR_START_STEP) / AR_RAMP_STEPS
        progress = max(0.0, min(1.0, progress))
        ar_len = int(progress * AR_ROLLOUT_MAX)

    # v4.5 OOM fix: Lock seq_len to BASE_SEQ_LEN to prevent memory growth
    # Previous logic allowed seq_len to grow with ar_len, causing OOM
    current_base = BASE_SEQ_LEN
    if CURRICULUM_SEQ_LEN:
        current_base = min(BASE_SEQ_LEN + (step // SEQ_LEN_INCREASE_STEPS), MAX_SEQ_LEN)

    seq_len = current_base  # Fixed at BASE_SEQ_LEN (16)
    ar_len = min(ar_len, seq_len - 1)  # Cap ar_len to seq_len - 1
    if ar_len < 0: ar_len = 0

    return seq_len, ar_len

# Initialize curriculum
prev_seq_len, prev_ar_len = compute_curriculum_params(global_step)
dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=prev_seq_len)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)  # v4.5.2: Keep workers alive
loader_iter = iter(loader)
print(f"Initial Curriculum: seq_len={prev_seq_len}, ar_len={prev_ar_len}")

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    # Infinite batch loop (dataloader reloads when curriculum changes)
    while True:
        # Check if curriculum needs update
        seq_len, ar_len = compute_curriculum_params(global_step)

        if seq_len != prev_seq_len:
            print(f"Curriculum changed: seq_len {prev_seq_len}->{seq_len}, ar_len {prev_ar_len}->{ar_len}")
            prev_seq_len, prev_ar_len = seq_len, ar_len
            # Reload dataset and dataloader
            dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
            if len(dataset) == 0:
                print("WARNING: Empty dataset, breaking epoch")
                break
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)  # v4.5.2: Keep workers alive
            loader_iter = iter(loader)

        # Get next batch
        t_step_start = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            # Epoch complete
            break

        t_data_end = time.time()
        Z_seq, A_seq, Z_target, _, _, _ = batch
        Z_seq, A_seq, Z_target = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target.to(DEVICE)
        B, K, H, W = Z_seq.shape
        step_losses = []

        # LR Warmup
        if global_step <= WARMUP_STEPS:
            new_lr = MIN_LR + (global_step / WARMUP_STEPS) * (LR - MIN_LR)
            for pg in optimizer.param_groups: pg['lr'] = new_lr

        optimizer.zero_grad()

        # Embeddings & FiLM
        X_seq = model.compute_embeddings(Z_seq)
        Gammas_seq, Betas_seq = model.compute_film(A_seq)
        h_state = model.init_state(B, device=DEVICE)

        # v4.6.6: Primarily teacher-forced with occasional single-step AR mix
        # AR rollout moved to validation-only (see AR validation section below)

        step_losses = []         # Loss per timestep
        lpips_loss_steps = []    # v4.6.2: Track LPIPS losses across sequence
        logits_last = None
        prev_pred_tokens = None  # v4.6.6: Track previous prediction for AR mix
        ar_mix_count = 0         # v4.6.6: Count AR mix steps for diagnostics

        t_forward_start = time.time()
        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            # v4.6.6: ar_len always 0 (teacher-forced only, except AR mix)
            ar_cutoff = K  # All steps are teacher-forced

            for t in range(K):
                # v4.6.6: Single-step AR mix for noise robustness
                use_ar_step = (AR_MIX_ENABLED and
                               t > 0 and
                               prev_pred_tokens is not None and
                               random.random() < AR_MIX_PROB)

                if use_ar_step:
                    # Use previous prediction (detached to prevent gradient backprop through time)
                    with torch.no_grad():
                        x_in = model.compute_embeddings(prev_pred_tokens.unsqueeze(1))[:, 0]  # (B, H, W) -> (B, 1, H, W) -> (B, D, H, W)
                    ar_mix_count += 1
                else:
                    # Normal teacher forcing (ground truth)
                    x_in = X_seq[:, t]

                # Step
                g_t = Gammas_seq[:, :, t]
                b_t = Betas_seq[:, :, t]

                if USE_CHECKPOINTING:
                    logits_t, h_state = checkpoint(model.step, torch.tensor(0, device=DEVICE), torch.tensor(0, device=DEVICE), h_state, x_in, g_t, b_t, use_reentrant=False)
                else:
                    logits_t, h_state = model.step(None, None, h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)

                logits_last = logits_t

                # v4.6.6: Store prediction for potential next AR step
                with torch.no_grad():
                    prev_pred_tokens = logits_t.argmax(dim=1).detach()

                # --- LOSS CALCULATION (v4.6.5: Optimized) ---
                # 1. Semantic Loss with Gumbel-Softmax (Handle Blur + Force Discrete)
                logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                target_flat = Z_target[:, t].reshape(-1)
                loss_texture = semantic_criterion(logits_flat, target_flat, global_step=global_step)

                # 3. LPIPS Perceptual Loss (v4.6.2: Differentiable via Gumbel-Softmax)
                loss_lpips = torch.tensor(0.0, device=DEVICE)
                t_lpips_start = time.time()
                if lpips_criterion is not None and (t % LPIPS_FREQ == 0):
                    # v4.6.2 FIX: Use Gumbel-Softmax for differentiable token selection
                    # Reuse same tau as semantic loss for consistency
                    tau = max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9)

                    # Get soft token probabilities (differentiable)
                    logits_flat_lpips = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))  # (B*H*W, C)
                    probs = F.gumbel_softmax(logits_flat_lpips, tau=tau, hard=False, dim=-1)  # (B*H*W, C)
                    probs = probs.reshape(logits_t.size(0), logits_t.size(2), logits_t.size(3), -1)  # (B, H, W, C)

                    # Soft embedding lookup: weighted sum over VQ-VAE codebook
                    # codebook is already transposed (K, D) from line 295
                    soft_embeddings = torch.matmul(probs, codebook)  # (B, H, W, D)
                    soft_embeddings = soft_embeddings.permute(0, 3, 1, 2)  # (B, D, H, W)

                    # Decode soft embeddings directly (gradients flow!)
                    pred_rgb = vqvae_model.decoder(soft_embeddings)  # (B, 3, 128, 72)

                    # Target (no gradients needed)
                    with torch.no_grad():
                        target_tokens = Z_target[:, t]  # (B, H, W)
                        target_rgb = vqvae_model.decode_code(target_tokens)  # (B, 3, 128, 72)

                    # LPIPS expects inputs in [-1, 1] range
                    pred_rgb_norm = pred_rgb * 2.0 - 1.0
                    target_rgb_norm = target_rgb * 2.0 - 1.0

                    # Compute perceptual loss (gradients now flow through entire chain!)
                    loss_lpips = lpips_criterion(pred_rgb_norm, target_rgb_norm).mean()

                    # Track LPIPS for averaging
                    lpips_loss_steps.append(loss_lpips)

                t_lpips_end = time.time()
                # Update LPIPS timing (only when LPIPS was computed)
                if lpips_criterion is not None and (t % LPIPS_FREQ == 0):
                    lpips_time = t_lpips_end - t_lpips_start
                    if timing_stats['lpips'] is None:
                        timing_stats['lpips'] = lpips_time
                    else:
                        timing_stats['lpips'] = (1 - TIMING_EMA_ALPHA) * timing_stats['lpips'] + TIMING_EMA_ALPHA * lpips_time

                # Weighted Sum (v4.6.6: Only 2 components - neighbor removed)
                loss_step = (SEMANTIC_WEIGHT * loss_texture) + \
                            (LPIPS_WEIGHT * loss_lpips)
                step_losses.append(loss_step)

            loss = torch.stack(step_losses).mean()

        t_forward_end = time.time()

        t_backward_start = time.time()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # v4.5: Tighter clipping (was 1.0)
        t_backward_end = time.time()

        t_optimizer_start = time.time()
        scaler.step(optimizer)
        scaler.update()
        t_optimizer_end = time.time()

        # v4.6.6: Update timing statistics with EMA
        data_time = t_data_end - t_step_start
        forward_time = t_forward_end - t_forward_start
        backward_time = t_backward_end - t_backward_start
        optimizer_time = t_optimizer_end - t_optimizer_start
        step_time = t_optimizer_end - t_step_start

        if timing_stats['data_load'] is None:
            # First step - initialize
            timing_stats['data_load'] = data_time
            timing_stats['forward'] = forward_time
            timing_stats['backward'] = backward_time
            timing_stats['optimizer'] = optimizer_time
            timing_stats['step_total'] = step_time
        else:
            # EMA update
            timing_stats['data_load'] = (1 - TIMING_EMA_ALPHA) * timing_stats['data_load'] + TIMING_EMA_ALPHA * data_time
            timing_stats['forward'] = (1 - TIMING_EMA_ALPHA) * timing_stats['forward'] + TIMING_EMA_ALPHA * forward_time
            timing_stats['backward'] = (1 - TIMING_EMA_ALPHA) * timing_stats['backward'] + TIMING_EMA_ALPHA * backward_time
            timing_stats['optimizer'] = (1 - TIMING_EMA_ALPHA) * timing_stats['optimizer'] + TIMING_EMA_ALPHA * optimizer_time
            timing_stats['step_total'] = (1 - TIMING_EMA_ALPHA) * timing_stats['step_total'] + TIMING_EMA_ALPHA * step_time

        # --- DIAGNOSTICS & LOGGING ---
        if (loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD) and REPORT_SPIKES:
            print(f"\nSPIKE: step={global_step} loss={loss.item():.2f} grad={float(grad_norm):.2f}")
            print(f"   (Semantic: {loss_texture.item():.4f}, LPIPS: {lpips_loss_avg:.4f})")

        total_loss += loss.item()
        
        # Periodic Logs
        if global_step % LOG_STEPS == 0:
            # v4.6.6: Calculate LPIPS average
            if len(lpips_loss_steps) > 0:
                lpips_loss_avg = torch.stack(lpips_loss_steps).mean().item()
            else:
                lpips_loss_avg = 0.0

            # v4.6.6: Print timing summary
            if timing_stats['step_total'] is not None:
                throughput = 1.0 / timing_stats['step_total'] if timing_stats['step_total'] > 0 else 0
                print(f"[Step {global_step}] Loss: {loss.item():.4f} | "
                      f"{throughput:.2f} steps/s | "
                      f"Total: {timing_stats['step_total']*1000:.1f}ms "
                      f"(Data: {timing_stats['data_load']*1000:.0f}ms, "
                      f"Fwd: {timing_stats['forward']*1000:.0f}ms, "
                      f"Bwd: {timing_stats['backward']*1000:.0f}ms, "
                      f"Opt: {timing_stats['optimizer']*1000:.0f}ms"
                      f"{', LPIPS: ' + str(int(timing_stats['lpips']*1000)) + 'ms' if timing_stats['lpips'] is not None else ''})")
            else:
                print(f"[Step {global_step}] Loss: {loss.item():.4f}")

            # Calculate Entropy (Uncertainty)
            with torch.no_grad():
                if logits_last is not None:
                    probs = F.softmax(logits_last.float(), dim=1) # (B, C, H, W)
                    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
                    confidence = probs.max(dim=1)[0].mean()
                    unique_codes = logits_last.argmax(dim=1).unique().numel()

                    # v4.5: NEW Spatial Gradient Magnitude (sharpness proxy)
                    spatial_grad_x = (logits_last[:, :, :, 1:] - logits_last[:, :, :, :-1]).abs().mean()
                    spatial_grad_y = (logits_last[:, :, 1:, :] - logits_last[:, :, :-1, :]).abs().mean()
                    spatial_gradient = (spatial_grad_x + spatial_grad_y) / 2

                    # v4.5: NEW Confidence distribution stats
                    top1_conf = probs.max(dim=1)[0]
                    confidence_std = top1_conf.std()
                    confidence_min = top1_conf.min()
                else:
                    # Fallback values for first step
                    entropy = torch.tensor(0.0)
                    confidence = torch.tensor(0.0)
                    unique_codes = 0
                    spatial_gradient = torch.tensor(0.0)
                    confidence_std = torch.tensor(0.0)
                    confidence_min = torch.tensor(0.0)

                # Action responsiveness diagnostics
                # v4.6.5: Use FiLM magnitude as proxy (removed expensive perturbation test)
                gamma_magnitude = Gammas_seq.abs().mean().item()
                beta_magnitude = Betas_seq.abs().mean().item()
                action_sensitivity = (gamma_magnitude + beta_magnitude) / 2

            if wandb:
                log_dict = {
                    # Core metrics
                    "train/loss": loss.item(),
                    "train/loss_texture": loss_texture.item(),

                    # v4.6.2: LPIPS perceptual loss (averaged across sequence)
                    "train/loss_lpips": lpips_loss_avg,

                    # v4.6.0: Gumbel-Softmax tau annealing
                    "train/gumbel_tau": max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9),

                    # v4.6.6: AR mix diagnostics
                    "ar_mix/enabled": 1.0 if AR_MIX_ENABLED else 0.0,
                    "ar_mix/probability": AR_MIX_PROB if AR_MIX_ENABLED else 0.0,
                    "ar_mix/actual_frequency": ar_mix_count / K if K > 0 else 0.0,

                    # Action diagnostics
                    "action_diagnostics/film_gamma_magnitude": gamma_magnitude,
                    "action_diagnostics/film_beta_magnitude": beta_magnitude,
                    "action_diagnostics/sensitivity": action_sensitivity,

                    # Existing diagnostics
                    "diagnostics/entropy": entropy.item(),
                    "diagnostics/confidence": confidence.item(),
                    "diagnostics/unique_codes": unique_codes,

                    # v4.5: NEW diagnostics
                    "diagnostics/spatial_gradient": spatial_gradient.item(),
                    "diagnostics/confidence_std": confidence_std.item(),
                    "diagnostics/confidence_min": confidence_min.item(),

                    "train/grad_norm": float(grad_norm),
                    # v4.6.5: seq_len and ar_len removed (constants: 20 and 0)

                    # v4.6.6: Timing & throughput metrics
                    "timing/step_total_ms": timing_stats['step_total'] * 1000 if timing_stats['step_total'] else 0,
                    "timing/data_load_ms": timing_stats['data_load'] * 1000 if timing_stats['data_load'] else 0,
                    "timing/forward_ms": timing_stats['forward'] * 1000 if timing_stats['forward'] else 0,
                    "timing/backward_ms": timing_stats['backward'] * 1000 if timing_stats['backward'] else 0,
                    "timing/optimizer_ms": timing_stats['optimizer'] * 1000 if timing_stats['optimizer'] else 0,
                    "timing/throughput_steps_per_sec": 1.0 / timing_stats['step_total'] if timing_stats['step_total'] and timing_stats['step_total'] > 0 else 0,
                }

                # Add LPIPS timing if available
                if timing_stats['lpips'] is not None:
                    log_dict["timing/lpips_ms"] = timing_stats['lpips'] * 1000

                # Add AR validation timing if available
                if timing_stats['ar_validation'] is not None:
                    log_dict["timing/ar_validation_ms"] = timing_stats['ar_validation'] * 1000

                wandb.log(log_dict, step=global_step)

        # Image Logs (Visual Confirmation)
        if global_step % IMAGE_LOG_STEPS == 0:
            log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)
            # v4.7.0: Also log AR rollout visualization when AR is active
            if ar_len > 0:
                log_ar_rollout_to_wandb(model, vqvae_model, Z_seq, A_seq, Z_target, global_step, ar_len)
            # v4.7.0: Action conditioning validation
            action_metrics = validate_action_conditioning(model, vqvae_model, Z_seq, A_seq, Z_target, global_step)
            if wandb and action_metrics:
                wandb.log(action_metrics, step=global_step)

        # v4.6.5: AR Validation
        if global_step % AR_VALIDATION_FREQ == 0 and global_step > 0:
            t_ar_val_start = time.time()
            model.eval()
            with torch.no_grad(), autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                # Perform AR_VALIDATION_STEPS rollouts
                validation_ar_losses = []
                validation_teacher_losses = []

                for _ in range(AR_VALIDATION_STEPS):
                    # Use current batch for validation
                    val_X_seq = X_seq
                    val_h_state = model.init_state(B, device=DEVICE)

                    # Teacher-forced pass (ground truth)
                    teacher_val_losses = []
                    for t in range(K):
                        x_in = val_X_seq[:, t]
                        g_t = Gammas_seq[:, :, t]
                        b_t = Betas_seq[:, :, t]

                        logits_t, val_h_state = model.step(None, None, val_h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)

                        # Compute loss
                        logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                        target_flat = Z_target[:, t].reshape(-1)
                        loss_t = semantic_criterion(logits_flat, target_flat, global_step=global_step)
                        teacher_val_losses.append(loss_t)

                    teacher_val_loss = torch.stack(teacher_val_losses).mean()
                    validation_teacher_losses.append(teacher_val_loss)

                    # AR rollout pass (autoregressive)
                    val_h_state = model.init_state(B, device=DEVICE)
                    ar_val_losses = []
                    x_prev = val_X_seq[:, 0]  # Start with first GT frame

                    for t in range(K):
                        g_t = Gammas_seq[:, :, t]
                        b_t = Betas_seq[:, :, t]

                        # First frame uses GT, rest use predictions
                        if t == 0:
                            x_in = val_X_seq[:, t]
                        else:
                            x_in = x_prev

                        logits_t, val_h_state = model.step(None, None, val_h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)

                        # Compute loss
                        logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                        target_flat = Z_target[:, t].reshape(-1)
                        loss_t = semantic_criterion(logits_flat, target_flat, global_step=global_step)
                        ar_val_losses.append(loss_t)

                        # Use prediction for next step
                        z_pred = logits_t.argmax(dim=1)
                        x_prev = model._embed_tokens(z_pred)

                    ar_val_loss = torch.stack(ar_val_losses).mean()
                    validation_ar_losses.append(ar_val_loss)

                # Average across validation rollouts
                avg_teacher_val_loss = torch.stack(validation_teacher_losses).mean().item()
                avg_ar_val_loss = torch.stack(validation_ar_losses).mean().item()
                validation_ar_gap = avg_ar_val_loss - avg_teacher_val_loss

                # Log validation metrics
                if wandb:
                    wandb.log({
                        "validation/teacher_loss": avg_teacher_val_loss,
                        "validation/ar_loss": avg_ar_val_loss,
                        "validation/ar_loss_gap": validation_ar_gap,
                    }, step=global_step)

                print(f"[Step {global_step}] Validation AR gap: {validation_ar_gap:.4f} (teacher: {avg_teacher_val_loss:.4f}, ar: {avg_ar_val_loss:.4f})")

            model.train()  # Return to training mode

            t_ar_val_end = time.time()
            ar_val_time = t_ar_val_end - t_ar_val_start
            if timing_stats['ar_validation'] is None:
                timing_stats['ar_validation'] = ar_val_time
            else:
                timing_stats['ar_validation'] = (1 - TIMING_EMA_ALPHA) * timing_stats['ar_validation'] + TIMING_EMA_ALPHA * ar_val_time

        # v4.6.6: Milestone Save (every 10k steps)
        if global_step > 0 and global_step % MILESTONE_SAVE_STEPS == 0:
            save_checkpoint(epoch, global_step, is_milestone=True)

        # Emergency Save (time-based)
        if (time.time() - last_emergency_save_time) > EMERGENCY_SAVE_INTERVAL_HRS * 3600:
            save_checkpoint(epoch, global_step, is_emergency=True)
            last_emergency_save_time = time.time()

        global_step += 1

    print(f"Epoch {epoch}: mean loss {total_loss / len(loader):.4f}")
    save_checkpoint(epoch, global_step)

if wandb: wandb.finish()
print("Training complete.")