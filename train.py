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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torchvision.utils import make_grid
from torch import autocast

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

print("Device:", DEVICE)

# --- PATHS ---
DATA_DIR = "../preprocessedv4"
VQVAE_PATH = "./checkpoints/vqvae_v2.1.6__epoch100.pt" 
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32         # v4.7.4: 48 -> 32
MAX_STEPS = 300_000  # v4.7.0: Step-based training (replaces EPOCHS)
LR = 3e-5             # v4.7.4: 4.5e-5 -> 3e-5
WARMUP_STEPS = 750      # v4.7.3: 1.5× warmup (500 → 750)
MIN_LR = 1e-6         # v4.7.4: 1.5e-6 -> 1e-6
USE_CHECKPOINTING = False 

# --- LOSS WEIGHTS ---
# v4.10.1: Rebalance to fight mode collapse while maintaining action conditioning
SEMANTIC_WEIGHT = 1.0    # v4.10.1: Increased from 0.5 to encourage codebook diversity
LPIPS_WEIGHT = 3.0       # v4.10.1: Increased from 2.0 to fight repetitive patterns
                         # Combined with LPIPS_FREQ=1, this makes LPIPS the dominant signal
LPIPS_FREQ = 1           # v4.6.5: Every step (was 5) - eliminates periodic flashing artifact
GUMBEL_TAU_STEPS = 20000 # Gumbel-Softmax annealing: 1.0→0.1 over this many steps

# v4.7.4: AR loss upweighting (emphasize deployment condition)
AR_LOSS_WEIGHT = 2.5     # Static upweighting of AR loss vs TF loss
                         # Rationale: AR is deployment condition, TF and AR were improving at same rate
                         # despite AR being harder task (3 frames vs 17 frames contributing to loss)

# --- CURRICULUM ---
# v4.7.1: seq_len curriculum removed - seq_len dynamically grows with ar_len
# Live inference uses CTX_LEN=6, so training seq_len doesn't need to match
BASE_SEQ_LEN = 20               # Minimum sequence length (will grow to ar_len + 1)
CURRICULUM_SEQ_LEN = False      # Disabled (seq_len now auto-adjusts to AR curriculum)
MAX_SEQ_LEN = 50                # Not used
SEQ_LEN_INCREASE_STEPS = 5000   # Not used

# v4.7.1: AR curriculum with adaptive brake (TIER 2 - PRIMARY)
# Warm up with teacher forcing, then guarantee some AR exposure with adaptive quality control
CURRICULUM_AR = True             # Re-enabled: AR rollout during training
AR_WARMUP_STEPS = 5000          # Warm up before starting AR
AR_MIN_LEN = 10                 # v4.9.0: Increased from 3 to force longer rollouts
AR_MAX_LEN = 25                 # Maximum AR rollout length

# Adaptive AR brake: prevent LPIPS degradation by monitoring AR vs TF quality
ADAPTIVE_AR_BRAKE = True
AR_BRAKE_EMA_ALPHA = 0.98       # Smoothing for LPIPS tracking
AR_BRAKE_HYSTERESIS = 0.05      # Prevent oscillation
AR_BRAKE_RATIO_UPPER = 2.5      # v4.11.0: Increased from 2.0 - allow curriculum to progress past current 1.5 ratio
AR_BRAKE_RATIO_LOWER = 1.6      # v4.9.0: Increased from 1.3 - allow progression at current 1.5 ratio

# v4.6.6: Single-step AR mix DISABLED in v4.7.1 (replaced by guaranteed AR exposure)
AR_MIX_ENABLED = False          # Disabled: v4.7.1 uses guaranteed AR exposure instead
AR_MIX_PROB = 0.0               # Not used

# --- ACTION CONDITIONING (TIER 1) ---
# v4.11.0: Reduced action ranking weight to balance with stronger IDM
ACTION_RANK_WEIGHT = 1.0        # Reduced from 2.0 - make room for multi-step IDM
ACTION_RANK_FREQ = 1            # v4.9.0: Every step (was 5) - continuous action gradients
ACTION_RANK_MARGIN = 0.05       # Margin for ranking hinge loss

# NOTE: ACTION_RANK_NUM_NEG removed - hardcoded to 1 for efficiency
# NOTE: ACTION_NOISE_SCALE not implemented - reserved for future use

# --- INVERSE DYNAMICS MODULE (v4.11.0: Variable-Span "Time Telescope") ---
IDM_LOSS_WEIGHT = 0.5  # v4.10.1: Reduced from 1.0 to reduce dominance over reconstruction
                       # v4.10.0 @ 19.5k: IDM too strong → mode collapse (unique_codes ~35)
                       # Need to rebalance: maintain action conditioning while improving diversity

# v4.11.0: Multi-step IDM configuration
MAX_IDM_SPAN = 5       # Maximum look-ahead frames for IDM (Time Telescope)
                       # Movement velocity visible over 3-5 frames, not 1 frame
MOVEMENT_WEIGHT = 10.0 # Penalty multiplier for Move_X / Move_Z actions
                       # Single-step supervision insufficient for movement (weak gradient)
JUMP_WEIGHT = 5.0      # Penalty multiplier for Jump action
                       # Multi-step supervision helps capture jump arc dynamics

# --- OPTIMIZATION ---
# v4.11.0: Further increase FiLM LR to provide more action gradient headroom
FILM_LR_MULT = 25.0             # Increased from 15.0 (v4.10.1 still showed 16× imbalance)
                                # Target: stronger action gradients for movement/yaw learning

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v4.11.0-step0"
MODEL_OUT_PREFIX = "ochre-v4.11.0"
RESUME_PATH = ""

LOG_STEPS = 10
IMAGE_LOG_STEPS = 500 # Log visual reconstruction every N steps
REPORT_SPIKES = True
SPIKE_LOSS = 15.0
SPIKE_GRAD = 400
EMERGENCY_SAVE_INTERVAL_HRS = 11.8
MILESTONE_SAVE_STEPS = 10000  # v4.6.6: Save checkpoint every N steps

# --- ACTION VALIDATION ---
# v4.8.1: Multi-step action conditioning validation
ACTION_VALIDATION_STEPS = [1, 5, 10]  # Test action response at different rollout lengths
ACTION_VISUAL_ROLLOUT_LEN = 30         # Full rollout length for visual logging

# v4.7.2: Fine-grained timing & throughput tracking
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


def compute_multistep_action_response(model, vqvae, z_start, action_vec, num_steps, device):
    """
    v4.8.1: Perform multi-step AR rollout with a fixed action.

    Args:
        model: World model
        vqvae: VQ-VAE decoder
        z_start: (1, H, W) starting frame tokens
        action_vec: (1, 5) action to apply at each step
        num_steps: Number of AR steps to roll out
        device: torch device

    Returns:
        list of (3, IMG_H, IMG_W) RGB frames
    """
    with torch.no_grad():
        x_t = model._embed_tokens(z_start)
        h_state = model.init_state(1, device=device)

        frames = []
        for _ in range(num_steps):
            # Get FiLM parameters for this action
            gammas, betas = model.film(action_vec)  # (L, 1, C, 1, 1)

            # Predict next frame
            logits, h_state = model.step(None, None, h_state, x_t=x_t, gammas_t=gammas, betas_t=betas)
            pred_tokens = logits.argmax(dim=1)  # (1, H, W)
            pred_rgb = vqvae.decode_code(pred_tokens)[0]  # (3, IMG_H, IMG_W)

            frames.append(pred_rgb.cpu())

            # Update for next step (AR)
            x_t = model._embed_tokens(pred_tokens)

        return frames


def validate_action_conditioning(model, vqvae, Z_seq, A_seq, Z_target, global_step):
    """
    v4.8.1: Multi-step action conditioning validation to catch degradation over AR rollouts.
    Tests model response to different action types at multiple rollout lengths.

    Args:
        model: World model
        vqvae: VQ-VAE decoder
        Z_seq: (B, K, H, W) input token sequence
        A_seq: (B, K, A) action sequence
        Z_target: (B, K, H, W) target token sequence
        global_step: Current training step

    Returns:
        dict: Metrics for different action conditions and rollout lengths
    """
    if wandb is None: return {}

    with torch.no_grad():
        device = Z_seq.device
        # Use first sample
        z_start = Z_seq[0:1, 0]  # (1, H, W) - starting frame

        # Action format: [yaw, pitch, move_x, move_z, action_5] (5-dim)
        # Test different action conditions
        test_actions = {
            'static': torch.zeros(1, 5, device=device),  # No movement
            'camera_left': torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]], device=device),  # Yaw left
            'camera_right': torch.tensor([[-0.5, 0.0, 0.0, 0.0, 0.0]], device=device),  # Yaw right
            'move_forward': torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0]], device=device),  # Forward
        }

        # v4.8.1: Multi-step rollouts for each action
        rollout_predictions = {}  # {action_name: {num_steps: [frames]}}
        for action_name, action_vec in test_actions.items():
            rollout_predictions[action_name] = {}
            for num_steps in ACTION_VALIDATION_STEPS:
                frames = compute_multistep_action_response(model, vqvae, z_start, action_vec, num_steps, device)
                rollout_predictions[action_name][num_steps] = frames

        # v4.8.1: Compute action response metrics at each rollout length
        metrics = {}
        for num_steps in ACTION_VALIDATION_STEPS:
            # Get final frame from each rollout
            static_frame = rollout_predictions['static'][num_steps][-1].flatten()
            camera_l_frame = rollout_predictions['camera_left'][num_steps][-1].flatten()
            camera_r_frame = rollout_predictions['camera_right'][num_steps][-1].flatten()
            move_fwd_frame = rollout_predictions['move_forward'][num_steps][-1].flatten()

            # L2 distance between final frames (higher = better action response)
            diff_camera_l = (static_frame - camera_l_frame).pow(2).mean().sqrt().item()
            diff_camera_r = (static_frame - camera_r_frame).pow(2).mean().sqrt().item()
            diff_move_fwd = (static_frame - move_fwd_frame).pow(2).mean().sqrt().item()

            # Average action response magnitude
            action_response = (diff_camera_l + diff_camera_r + diff_move_fwd) / 3

            # Store metrics with step suffix
            step_suffix = f"_{num_steps}step" if num_steps > 1 else ""
            metrics[f'action_response/camera_left_diff{step_suffix}'] = diff_camera_l
            metrics[f'action_response/camera_right_diff{step_suffix}'] = diff_camera_r
            metrics[f'action_response/move_forward_diff{step_suffix}'] = diff_move_fwd
            metrics[f'action_response/average{step_suffix}'] = action_response

        # v4.8.1: 30-frame rollout visualization - replaces old AR rollout
        # Generate 30-frame rollouts for visual logging
        rollout_30_predictions = {}
        for action_name, action_vec in test_actions.items():
            frames_30 = compute_multistep_action_response(model, vqvae, z_start, action_vec, ACTION_VISUAL_ROLLOUT_LEN, device)
            rollout_30_predictions[action_name] = frames_30

        # Show evenly spaced frames from 30-frame rollout (6 frames per action)
        num_vis_frames = 6
        vis_frames_30 = []
        for action_name in ['static', 'camera_left', 'camera_right', 'move_forward']:
            frames = rollout_30_predictions[action_name]
            # Select evenly spaced frames
            indices = [int(i * (ACTION_VISUAL_ROLLOUT_LEN - 1) / (num_vis_frames - 1)) for i in range(num_vis_frames)]
            for idx in indices:
                vis_frames_30.append(torch.clamp(frames[idx], 0.0, 1.0))

        # Create grid: 4 rows (actions) x 6 columns (time)
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

        # Convert uint16 to int32 (PyTorch doesn't support uint16)
        Z_seq_int = np.array(Z_seq, dtype=np.int32)
        Z_target_int = np.array(Z_target_seq, dtype=np.int32)

        return (
            torch.tensor(Z_seq_int, dtype=torch.long),
            torch.tensor(np.array(A_seq), dtype=torch.float32),
            torch.tensor(Z_target_int, dtype=torch.long),
            idx, vid_idx, start
        )

if wandb:
    wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
                config=dict(batch_size=BATCH_SIZE, lr=LR, max_steps=MAX_STEPS))

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
    print("VQ-VAE loaded from component-wise checkpoint")
elif "model_state" in vqvae_ckpt:
    # Unified state dict format
    vqvae_model.load_state_dict(vqvae_ckpt["model_state"], strict=False)
    print("VQ-VAE loaded from model_state")
else:
    # Try loading directly (legacy fallback)
    vqvae_model.load_state_dict(vqvae_ckpt, strict=False)
    print("WARNING: VQ-VAE loaded from root dict (legacy)")

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
    idm_max_span=MAX_IDM_SPAN,
    H=18,
    W=32,
    use_checkpointing=USE_CHECKPOINTING,
    zero_init_head=False  # Disable zero-init to prevent mode collapse at cold start
).to(DEVICE)

# v4.7.1: Optimizer with FiLM parameter group (higher LR for action pathway)
film_params = [p for n, p in model.named_parameters() if ("film" in n or "action" in n)]
dynamics_params = [p for n, p in model.named_parameters() if ("film" not in n and "action" not in n)]

optimizer = torch.optim.AdamW([
    {'params': dynamics_params, 'lr': LR},
    {'params': film_params, 'lr': LR * FILM_LR_MULT}
], lr=LR)
print(f"Optimizer: Base LR={LR}, FiLM LR={LR * FILM_LR_MULT} ({FILM_LR_MULT}x multiplier)")
scaler = GradScaler(enabled=(DEVICE == "cuda"))

# D. Setup Semantic Loss with loaded codebook
semantic_criterion = SemanticCodebookLoss(codebook).to(DEVICE)
print(f"Semantic Loss Initialized with codebook shape {codebook.shape}")

# E. Setup LPIPS Perceptual Loss (v4.6.0)

import lpips
lpips_criterion = lpips.LPIPS(net='alex').to(DEVICE)
lpips_criterion.eval()  # Frozen, only for loss computation
lpips_criterion.requires_grad_(False)
print(f"LPIPS Loss Initialized (AlexNet backend)")


# F. Resume Logic
global_step = 0
if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"📥 Resuming training from {RESUME_PATH}")
    ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    global_step = ckpt.get("global_step", 0)
    print(f"Resumed from step {global_step}")
else:
    print("Starting fresh training run")

def save_checkpoint(global_step, is_emergency=False, is_milestone=False):
    # v4.7.0: Step-based checkpointing (epoch removed)
    if is_emergency:
        save_name = f"{MODEL_OUT_PREFIX}-step{global_step}-emergency.pt"
    elif is_milestone:
        save_name = f"{MODEL_OUT_PREFIX}-step{global_step}.pt"
    else:
        # Periodic saves (every 10k steps if called manually)
        save_name = f"{MODEL_OUT_PREFIX}-step{global_step}.pt"

    # Ensure checkpoint directory exists
    os.makedirs("./checkpoints", exist_ok=True)

    save_path = f"./checkpoints/{save_name}"
    print(f"Saving checkpoint to {save_path}")
    torch.save({
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, save_path)


# --- TRAINING ---
last_emergency_save_time = time.time()

# v4.7.2: Fine-grained timing tracking (EMA)
# Notes:
# - Times are per training step (one batch), averaged with EMA.
# - LPIPS is timed as both total LPIPS time across all timesteps in the sequence and per-LPIPS-call time.
timing_stats = {
    'step_total': None,          # Total time per step (EMA)
    'data_load': None,           # Time to load batch

    # Forward breakdown
    'forward_total': None,       # Total forward section time
    'embed_film': None,          # X_seq + Gammas/Betas + init_state
    'model_step': None,          # Sum of model.step calls across sequence
    'loss_semantic': None,       # Sum of semantic loss compute across sequence
    'loss_lpips_total': None,    # Sum of LPIPS blocks across sequence (total)
    'loss_lpips_call': None,     # Mean time per LPIPS call (EMA)
    'action_rank': None,         # Action ranking loss block

    # Backward breakdown
    'backward_total': None,      # Total backward section time
    'backward': None,            # scaler.scale(loss).backward()
    'unscale': None,             # scaler.unscale_
    'grad_clip': None,           # clip_grad_norm_

    # Optimizer breakdown
    'optimizer_total': None,     # Total optimizer section time
    'optimizer_step': None,      # scaler.step(optimizer)
    'scaler_update': None,       # scaler.update()
}

# v4.7.1: AR curriculum with adaptive brake
# Global state for AR brake (initialized outside function)
current_ar_len = 0
lpips_tf_ema = None
lpips_ar_ema = None
brake_increase_count = 0
brake_decrease_count = 0
brake_stable_count = 0

def compute_ar_len_from_brake(step, lpips_tf_avg, lpips_ar_avg):
    """
    v4.7.1: Compute AR length with adaptive brake.
    Guarantees minimum AR exposure after warmup, with quality-based adjustments.
    """
    global current_ar_len, lpips_tf_ema, lpips_ar_ema, brake_increase_count, brake_decrease_count, brake_stable_count

    # Check if AR curriculum is enabled
    if not CURRICULUM_AR:
        return 0

    # Phase 1: Warmup (teacher-forcing only)
    if step < AR_WARMUP_STEPS:
        return 0

    # Phase 2: Initialize AR at minimum length
    if current_ar_len == 0:
        current_ar_len = AR_MIN_LEN
        print(f"AR curriculum started: ar_len={current_ar_len} (step {step})")
        return current_ar_len

    # Phase 3: Adaptive brake adjustments
    if ADAPTIVE_AR_BRAKE and lpips_ar_avg > 0:
        # Update EMA trackers
        if lpips_tf_ema is None:
            lpips_tf_ema = lpips_tf_avg
            lpips_ar_ema = lpips_ar_avg
        else:
            lpips_tf_ema = AR_BRAKE_EMA_ALPHA * lpips_tf_ema + (1 - AR_BRAKE_EMA_ALPHA) * lpips_tf_avg
            lpips_ar_ema = AR_BRAKE_EMA_ALPHA * lpips_ar_ema + (1 - AR_BRAKE_EMA_ALPHA) * lpips_ar_avg

        # Compute quality ratio (AR LPIPS / TF LPIPS)
        ratio = lpips_ar_ema / (lpips_tf_ema + 1e-6)

        # Apply brake logic with hysteresis
        prev_ar_len = current_ar_len
        if ratio > (AR_BRAKE_RATIO_UPPER + AR_BRAKE_HYSTERESIS):
            # AR quality degraded too much - reduce AR intensity
            current_ar_len = max(AR_MIN_LEN, current_ar_len - 2)
            if current_ar_len != prev_ar_len:
                brake_decrease_count += 1
                print(f"AR brake: REDUCE ar_len {prev_ar_len}->{current_ar_len} (ratio={ratio:.2f} > {AR_BRAKE_RATIO_UPPER}, step {step})")
            else:
                brake_stable_count += 1  # Pinned at AR_MIN_LEN
        elif ratio < (AR_BRAKE_RATIO_LOWER - AR_BRAKE_HYSTERESIS):
            # AR quality close to TF - allow increases
            current_ar_len = min(AR_MAX_LEN, current_ar_len + 2)
            if current_ar_len != prev_ar_len:
                brake_increase_count += 1
                print(f"AR brake: INCREASE ar_len {prev_ar_len}->{current_ar_len} (ratio={ratio:.2f} < {AR_BRAKE_RATIO_LOWER}, step {step})")
            else:
                brake_stable_count += 1  # Pinned at AR_MAX_LEN
        else:
            # Within acceptable range - no change
            brake_stable_count += 1

    return current_ar_len

def compute_curriculum_params(step, lpips_tf_avg=0, lpips_ar_avg=0):
    """
    v4.7.1: Compute curriculum parameters with adaptive AR brake.
    """
    ar_len = compute_ar_len_from_brake(step, lpips_tf_avg, lpips_ar_avg)

    # seq_len dynamically grows to accommodate AR rollout
    seq_len = max(BASE_SEQ_LEN, ar_len + 1)

    return seq_len, ar_len

# Initialize curriculum
prev_seq_len, prev_ar_len = compute_curriculum_params(global_step)
dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=prev_seq_len)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, pin_memory=True, persistent_workers=True)
loader_iter = iter(loader)
print(f"Initial Curriculum: seq_len={prev_seq_len}, ar_len={prev_ar_len}")
print(f"Training target: {MAX_STEPS:,} steps")

model.train()

# v4.7.1: Track prev step LPIPS for adaptive brake
prev_lpips_tf_avg = 0.0
prev_lpips_ar_avg = 0.0

# v4.7.0: Step-based training loop (no epochs)
while global_step < MAX_STEPS:
    # Check if curriculum needs update (uses LPIPS from previous step)
    seq_len, ar_len = compute_curriculum_params(global_step, prev_lpips_tf_avg, prev_lpips_ar_avg)

    if seq_len != prev_seq_len:
        print(f"Curriculum changed: seq_len {prev_seq_len}->{seq_len}, ar_len {prev_ar_len}->{ar_len}")
        prev_seq_len, prev_ar_len = seq_len, ar_len
        # Reload dataset and dataloader
        dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
        if len(dataset) == 0:
            print("WARNING: Empty dataset, stopping training")
            break
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
        loader_iter = iter(loader)

    # Get next batch
    t_step_start = time.perf_counter()
    try:
        batch = next(loader_iter)
    except StopIteration:
        # Dataloader exhausted, restart it (infinite training)
        loader_iter = iter(loader)
        batch = next(loader_iter)

    t_data_end = time.perf_counter()
    Z_seq, A_seq, Z_target, _, _, _ = batch
    Z_seq, A_seq, Z_target = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target.to(DEVICE)
    B, K, H, W = Z_seq.shape
    step_losses = []

    # LR Warmup (v4.7.1: Preserve FiLM multiplier)
    if global_step <= WARMUP_STEPS:
        warmup_progress = global_step / WARMUP_STEPS
        base_lr = MIN_LR + warmup_progress * (LR - MIN_LR)
        # Set LR for each param group (dynamics: base, FiLM: base * multiplier)
        optimizer.param_groups[0]['lr'] = base_lr  # dynamics
        optimizer.param_groups[1]['lr'] = base_lr * FILM_LR_MULT  # FiLM

    optimizer.zero_grad()

    # v4.7.2: Forward timing breakdown
    t_forward_start = time.perf_counter()
    t_embed_film_start = time.perf_counter()

    # Embeddings & FiLM
    X_seq = model.compute_embeddings(Z_seq)
    Gammas_seq, Betas_seq = model.compute_film(A_seq)
    h_state = model.init_state(B, device=DEVICE)

    t_embed_film_end = time.perf_counter()

    # v4.7.1: Training loop with adaptive AR curriculum (ar_len determined by brake)
    step_losses = []         # Loss per timestep
    teacher_losses = []      # v4.7.0: Track teacher-forcing losses
    ar_losses = []           # v4.7.0: Track autoregressive losses
    lpips_loss_steps = []    # v4.6.2: Track LPIPS losses across sequence
    lpips_tf_steps = []      # v4.7.1: Track TF LPIPS separately
    lpips_ar_steps = []      # v4.7.1: Track AR LPIPS separately
    logits_last = None       # v4.9.0: Track logits for AR gradient flow
    prev_pred_tokens = None  # v4.6.6: Track previous prediction for AR mix
    ar_mix_count = 0         # v4.6.6: Count AR mix steps for diagnostics

    # v4.10.0: IDM loss accumulation
    idm_losses = []

    # v4.11.0: Buffer for Time Telescope IDM
    h_buffer = []  # Stores detached hidden states for variable-span IDM

    # v4.9.0: Action ranking (Option A) - capture a single realistic hidden state/input
    # Instead of caching all h_states (huge memory), sample one timestep and capture h_state and x_in used there.
    do_action_rank = (global_step % ACTION_RANK_FREQ == 0 and K > 1)
    t_rank = random.randint(1, K - 1) if do_action_rank else None
    h_rank = None
    x_rank = None

    model_step_time_total = 0.0
    loss_semantic_time_total = 0.0
    lpips_time_total = 0.0
    lpips_call_count = 0
    action_rank_time = 0.0

    with autocast('cuda' if DEVICE == "cuda" else 'cpu'):
        # v4.7.0: Restore AR curriculum - use ar_len to determine AR rollout
        ar_cutoff = K - ar_len  # Steps [0, ar_cutoff) are teacher-forced, [ar_cutoff, K) are AR

        # v4.11.0: Reuse tensors to reduce per-step allocations
        # NOTE: Do not reuse a single dt tensor with in-place updates across timesteps.
        # dt is used as indices into nn.Embedding; mutating it in-place before backward
        # triggers autograd "modified by an inplace operation" version-counter errors.
        action_weights = torch.tensor(
            [1.0, 1.0, MOVEMENT_WEIGHT, MOVEMENT_WEIGHT, JUMP_WEIGHT],
            device=DEVICE,
        )

        for t in range(K):
            # Determine input source based on AR curriculum and AR mix
            use_ar_rollout = (t >= ar_cutoff and t > 0 and prev_pred_tokens is not None)

            # v4.6.6: Single-step AR mix for noise robustness (independent of curriculum)
            use_ar_mix = (AR_MIX_ENABLED and
                          t > 0 and
                          t < ar_cutoff and  # Only apply AR mix in teacher-forced region
                          prev_pred_tokens is not None and
                          random.random() < AR_MIX_PROB)

            is_ar_step = use_ar_rollout or use_ar_mix

            # Teacher forcing: use current frame embedding as input
            # Dataset alignment: Z_target[:, t] is next frame, A_seq[:, t] is action at t
            # So we use X_seq[:, t] (current frame) + A_seq[:, t] (action) to predict Z_target[:, t] (next frame)
            if t == 0:
                # First frame: use ground truth
                x_in = X_seq[:, 0]
            elif is_ar_step:
                # v4.10.0: Detached AR (reverted from v4.9.0)
                # Gradient detachment prevents BPTT instability
                # IDM provides action conditioning instead
                with torch.no_grad():
                    x_in = model.compute_embeddings(prev_pred_tokens.unsqueeze(1))[:, 0]

                if use_ar_mix:
                    ar_mix_count += 1
            else:
                # Teacher forcing: Use current ground truth frame
                x_in = X_seq[:, t]

            # Step
            g_t = Gammas_seq[:, :, t]
            b_t = Betas_seq[:, :, t]

            # v4.10.0: Capture previous hidden state for IDM (detach to truncate BPTT).
            # We want IDM gradients to shape the current transition, not backprop through the whole unroll.
            h_prev_for_idm = h_state.detach()

            # v4.9.0: Option A - capture the realistic hidden state and actual input used at t_rank
            # Capture BEFORE the step (this is the state used to predict frame t).
            if do_action_rank and t_rank is not None and t == t_rank:
                # Detach to avoid backpropagating through the whole unroll (keeps memory bounded).
                h_rank = h_state.detach().clone()
                x_rank = x_in

            t_model_step_start = time.perf_counter()
            if USE_CHECKPOINTING:
                logits_t, h_state = checkpoint(
                    model.step,
                    torch.tensor(0, device=DEVICE),
                    torch.tensor(0, device=DEVICE),
                    h_state,
                    x_in,
                    g_t,
                    b_t,
                    use_reentrant=False,
                )
            else:
                logits_t, h_state = model.step(None, None, h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)
            model_step_time_total += (time.perf_counter() - t_model_step_start)

            # v4.11.0: Store detached state for IDM buffer
            # Only store top layer to save memory (reduce from ~2.8GB to ~0.9GB)
            h_buffer.append(h_state[-1].detach())

            # v4.11.0: Variable-Span IDM ("Time Telescope")
            loss_idm_step = torch.tensor(0.0, device=DEVICE)

            if t >= 1:  # Need at least 1 past frame
                # Random span: k ∈ [1, min(t, MAX_IDM_SPAN)]
                max_lookback = min(t, MAX_IDM_SPAN)
                k = random.randint(1, max_lookback)

                # Retrieve states (buffer index: t → buffer[-1], t-k → buffer[-1-k])
                # h_buffer now stores only top layer tensors (memory optimized)
                h_end = h_buffer[-1]      # Current state, top layer
                h_start = h_buffer[-1-k]  # Past state k steps ago, top layer

                # Cumulative action target: sum(A_seq[:, t-k+1:t+1]) → (B, 5)
                action_segment = A_seq[:, t-k+1:t+1]  # (B, k, 5)
                action_target = action_segment.sum(dim=1)  # (B, 5)

                # Predict with time-delta embedding
                dt_tensor = torch.full((B,), k, device=DEVICE, dtype=torch.long)
                pred_action = model.idm(h_start, h_end, dt_tensor)  # (B, 5)

                # Weighted loss: 10× movement, 5× jump
                raw_loss = F.mse_loss(pred_action, action_target, reduction='none')  # (B, 5)
                loss_idm_step = (raw_loss * action_weights).mean()

            idm_losses.append(loss_idm_step)

            logits_last = logits_t

            # v4.6.6: Store prediction for potential next AR step
            with torch.no_grad():
                prev_pred_tokens = logits_t.argmax(dim=1).detach()

            # --- LOSS CALCULATION (v4.6.5: Optimized) ---
            # 1. Semantic Loss with Gumbel-Softmax (Handle Blur + Force Discrete)
            if (SEMANTIC_WEIGHT > 0):
                t_semantic_start = time.perf_counter()
                logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                target_flat = Z_target[:, t].reshape(-1)
                loss_texture = semantic_criterion(logits_flat, target_flat, global_step=global_step)
                loss_semantic_time_total += (time.perf_counter() - t_semantic_start)
            else:
                loss_texture = torch.tensor(0.0, device=DEVICE)

            # 3. LPIPS Perceptual Loss (v4.6.2: Differentiable via Gumbel-Softmax)
            loss_lpips = torch.tensor(0.0, device=DEVICE)
            t_lpips_start = time.perf_counter()
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

                # v4.7.1: Track TF vs AR LPIPS separately
                if is_ar_step:
                    lpips_ar_steps.append(loss_lpips)
                else:
                    lpips_tf_steps.append(loss_lpips)

            t_lpips_end = time.perf_counter()
            # Update LPIPS timing (only when LPIPS was computed)
            if lpips_criterion is not None and (t % LPIPS_FREQ == 0):
                lpips_time = t_lpips_end - t_lpips_start
                lpips_time_total += lpips_time
                lpips_call_count += 1

            # Weighted Sum (v4.6.6: Only 2 components - neighbor removed)
            loss_step = (SEMANTIC_WEIGHT * loss_texture) + \
                        (LPIPS_WEIGHT * loss_lpips)
            step_losses.append(loss_step)
            if is_ar_step:
                ar_losses.append(loss_step)
            else:
                teacher_losses.append(loss_step)

        # v4.7.4: Apply AR loss upweighting (emphasize deployment condition)
        loss_teacher = torch.stack(teacher_losses).mean() if teacher_losses else torch.tensor(0.0, device=DEVICE)
        loss_ar = torch.stack(ar_losses).mean() if ar_losses else torch.tensor(0.0, device=DEVICE)
        loss = loss_teacher + AR_LOSS_WEIGHT * loss_ar

        # v4.9.0: Action-contrastive ranking loss with realistic hidden state (Option A)
        t_action_rank_start = time.perf_counter()
        loss_action_rank = torch.tensor(0.0, device=DEVICE)
        if do_action_rank:
            if h_rank is None or x_rank is None or t_rank is None:
                raise RuntimeError("Action ranking enabled but failed to capture h_rank/x_rank during unroll.")

            # True action: should predict target better
            g_true = Gammas_seq[:, :, t_rank]
            b_true = Betas_seq[:, :, t_rank]
            logits_true, _ = model.step(None, None, h_rank, x_t=x_rank, gammas_t=g_true, betas_t=b_true)

            # Compute loss for true action
            logits_true_flat = logits_true.permute(0, 2, 3, 1).reshape(-1, logits_true.size(1))
            target_rank_flat = Z_target[:, t_rank].reshape(-1)
            L_true = semantic_criterion(logits_true_flat, target_rank_flat, global_step=global_step)

            # Negative action: shuffle actions across batch (wrong action)
            perm = torch.randperm(B, device=DEVICE)
            a_neg = A_seq[:, t_rank][perm]  # Shuffled actions
            g_neg, b_neg = model.compute_film(a_neg.unsqueeze(1))  # (B, 1, A) -> (L, B, C, 1, 1)

            # Compute prediction with wrong action
            # Detach state for the negative branch to avoid gradients that "make negatives worse" via state dynamics.
            logits_neg, _ = model.step(None, None, h_rank.detach(), x_t=x_rank,
                                      gammas_t=g_neg[:, :, 0], betas_t=b_neg[:, :, 0])

            # Compute loss for negative action
            logits_neg_flat = logits_neg.permute(0, 2, 3, 1).reshape(-1, logits_neg.size(1))
            L_neg = semantic_criterion(logits_neg_flat, target_rank_flat, global_step=global_step)

            # Ranking loss: penalize if true action loss is not better than negative loss
            # margin: L_true should be at least ACTION_RANK_MARGIN better than L_neg
            loss_action_rank = F.relu(ACTION_RANK_MARGIN + L_true - L_neg)

            # Add to total loss
            loss = loss + ACTION_RANK_WEIGHT * loss_action_rank
        action_rank_time = time.perf_counter() - t_action_rank_start

        # v4.10.0: Add IDM loss
        loss_idm = torch.stack(idm_losses).mean() if idm_losses else torch.tensor(0.0, device=DEVICE)
        loss = loss + IDM_LOSS_WEIGHT * loss_idm

    t_forward_end = time.perf_counter()
    t_backward_start = time.perf_counter()
    t_backward_compute_start = time.perf_counter()
    scaler.scale(loss).backward()
    t_backward_compute_end = time.perf_counter()

    t_unscale_start = time.perf_counter()
    scaler.unscale_(optimizer)
    t_unscale_end = time.perf_counter()
    
    # v4.7.0: Track specific gradient norms (FiLM vs Dynamics)
    grad_film = 0.0
    grad_dynamics = 0.0
    if global_step % LOG_STEPS == 0:
        film_params = [p for n, p in model.named_parameters() if ("film" in n or "action" in n) and p.grad is not None]
        dynamics_params = [p for n, p in model.named_parameters() if ("film" not in n and "action" not in n) and p.grad is not None]
        
        if film_params:
            grad_film = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in film_params]), 2).item()
        if dynamics_params:
            grad_dynamics = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in dynamics_params]), 2).item()

    t_clip_start = time.perf_counter()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # v4.10.0: Revert to v4.8.1 clipping (v4.9.0 used 1.0 for BPTT headroom)
    t_clip_end = time.perf_counter()
    t_backward_end = time.perf_counter()

    t_optimizer_start = time.perf_counter()
    t_opt_step_start = time.perf_counter()
    scaler.step(optimizer)
    t_opt_step_end = time.perf_counter()
    t_scaler_update_start = time.perf_counter()
    scaler.update()
    t_scaler_update_end = time.perf_counter()
    t_optimizer_end = time.perf_counter()

    # v4.6.6: Update timing statistics with EMA
    data_time = t_data_end - t_step_start
    forward_time = t_forward_end - t_forward_start
    backward_time = t_backward_end - t_backward_start
    optimizer_time = t_optimizer_end - t_optimizer_start
    step_time = t_optimizer_end - t_step_start

    embed_film_time = t_embed_film_end - t_embed_film_start
    forward_total_time = forward_time
    backward_compute_time = t_backward_compute_end - t_backward_compute_start
    unscale_time = t_unscale_end - t_unscale_start
    grad_clip_time = t_clip_end - t_clip_start
    optimizer_step_time = t_opt_step_end - t_opt_step_start
    scaler_update_time = t_scaler_update_end - t_scaler_update_start

    lpips_call_time = (lpips_time_total / lpips_call_count) if lpips_call_count > 0 else 0.0

    if timing_stats['data_load'] is None:
        # First step - initialize
        timing_stats['step_total'] = step_time
        timing_stats['data_load'] = data_time

        timing_stats['forward_total'] = forward_total_time
        timing_stats['embed_film'] = embed_film_time
        timing_stats['model_step'] = model_step_time_total
        timing_stats['loss_semantic'] = loss_semantic_time_total
        timing_stats['loss_lpips_total'] = lpips_time_total
        timing_stats['loss_lpips_call'] = lpips_call_time
        timing_stats['action_rank'] = action_rank_time

        timing_stats['backward_total'] = backward_time
        timing_stats['backward'] = backward_compute_time
        timing_stats['unscale'] = unscale_time
        timing_stats['grad_clip'] = grad_clip_time

        timing_stats['optimizer_total'] = optimizer_time
        timing_stats['optimizer_step'] = optimizer_step_time
        timing_stats['scaler_update'] = scaler_update_time
    else:
        # EMA update
        timing_stats['step_total'] = (1 - TIMING_EMA_ALPHA) * timing_stats['step_total'] + TIMING_EMA_ALPHA * step_time
        timing_stats['data_load'] = (1 - TIMING_EMA_ALPHA) * timing_stats['data_load'] + TIMING_EMA_ALPHA * data_time

        timing_stats['forward_total'] = (1 - TIMING_EMA_ALPHA) * timing_stats['forward_total'] + TIMING_EMA_ALPHA * forward_total_time
        timing_stats['embed_film'] = (1 - TIMING_EMA_ALPHA) * timing_stats['embed_film'] + TIMING_EMA_ALPHA * embed_film_time
        timing_stats['model_step'] = (1 - TIMING_EMA_ALPHA) * timing_stats['model_step'] + TIMING_EMA_ALPHA * model_step_time_total
        timing_stats['loss_semantic'] = (1 - TIMING_EMA_ALPHA) * timing_stats['loss_semantic'] + TIMING_EMA_ALPHA * loss_semantic_time_total
        timing_stats['loss_lpips_total'] = (1 - TIMING_EMA_ALPHA) * timing_stats['loss_lpips_total'] + TIMING_EMA_ALPHA * lpips_time_total
        timing_stats['loss_lpips_call'] = (1 - TIMING_EMA_ALPHA) * timing_stats['loss_lpips_call'] + TIMING_EMA_ALPHA * lpips_call_time
        timing_stats['action_rank'] = (1 - TIMING_EMA_ALPHA) * timing_stats['action_rank'] + TIMING_EMA_ALPHA * action_rank_time

        timing_stats['backward_total'] = (1 - TIMING_EMA_ALPHA) * timing_stats['backward_total'] + TIMING_EMA_ALPHA * backward_time
        timing_stats['backward'] = (1 - TIMING_EMA_ALPHA) * timing_stats['backward'] + TIMING_EMA_ALPHA * backward_compute_time
        timing_stats['unscale'] = (1 - TIMING_EMA_ALPHA) * timing_stats['unscale'] + TIMING_EMA_ALPHA * unscale_time
        timing_stats['grad_clip'] = (1 - TIMING_EMA_ALPHA) * timing_stats['grad_clip'] + TIMING_EMA_ALPHA * grad_clip_time

        timing_stats['optimizer_total'] = (1 - TIMING_EMA_ALPHA) * timing_stats['optimizer_total'] + TIMING_EMA_ALPHA * optimizer_time
        timing_stats['optimizer_step'] = (1 - TIMING_EMA_ALPHA) * timing_stats['optimizer_step'] + TIMING_EMA_ALPHA * optimizer_step_time
        timing_stats['scaler_update'] = (1 - TIMING_EMA_ALPHA) * timing_stats['scaler_update'] + TIMING_EMA_ALPHA * scaler_update_time

    # --- DIAGNOSTICS & LOGGING ---
    # v4.6.6: Calculate LPIPS average (needed for spike detection and logging)
    if len(lpips_loss_steps) > 0:
        lpips_loss_avg = torch.stack(lpips_loss_steps).mean().item()
    else:
        lpips_loss_avg = 0.0

    if (loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD) and REPORT_SPIKES:
        print(f"\nSPIKE: step={global_step} loss={loss.item():.2f} grad={float(grad_norm):.2f}")
        print(f"   (Semantic: {loss_texture.item():.4f}, LPIPS: {lpips_loss_avg:.4f})")

    # v4.7.1: Compute TF/AR LPIPS averages (for adaptive brake)
    if len(lpips_tf_steps) > 0:
        prev_lpips_tf_avg = torch.stack(lpips_tf_steps).mean().item()
    else:
        prev_lpips_tf_avg = 0.0

    if len(lpips_ar_steps) > 0:
        prev_lpips_ar_avg = torch.stack(lpips_ar_steps).mean().item()
    else:
        prev_lpips_ar_avg = 0.0

    # Periodic Logs
    if global_step % LOG_STEPS == 0:
        # v4.7.2: Print fine-grained timing summary
        if timing_stats['step_total'] is not None:
            throughput = 1.0 / timing_stats['step_total'] if timing_stats['step_total'] > 0 else 0
            lpips_call_ms = int(timing_stats['loss_lpips_call'] * 1000) if timing_stats['loss_lpips_call'] is not None else 0
            print(
                f"[Step {global_step}] Loss: {loss.item():.4f} "
                f"(TF: {loss_teacher.item():.3f}, AR: {loss_ar.item():.3f}, "
                f"IDM: {loss_idm.item():.3f}, Rank: {loss_action_rank.item():.3f}) | "
                f"{throughput:.2f} steps/s | "
                f"Total: {timing_stats['step_total']*1000:.1f}ms "
                f"(Data: {timing_stats['data_load']*1000:.0f}ms, "
                f"Emb+FiLM: {timing_stats['embed_film']*1000:.0f}ms, "
                f"Step: {timing_stats['model_step']*1000:.0f}ms, "
                f"Sem: {timing_stats['loss_semantic']*1000:.0f}ms, "
                f"LPIPS: {timing_stats['loss_lpips_total']*1000:.0f}ms"
                f"{f' ({lpips_call_ms}ms/call)' if lpips_call_ms > 0 else ''}, "
                f"Rank: {timing_stats['action_rank']*1000:.0f}ms, "
                f"Bwd: {timing_stats['backward_total']*1000:.0f}ms, "
                f"Opt: {timing_stats['optimizer_total']*1000:.0f}ms)"
            )
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
            # v4.7.2: Timing percentages (relative to total step time)
            step_total_s = timing_stats['step_total'] if timing_stats['step_total'] else 0.0
            if step_total_s > 0:
                def _pct(value_s: float | None) -> float:
                    return (float(value_s) / step_total_s) * 100.0 if value_s else 0.0
            else:
                def _pct(value_s: float | None) -> float:
                    return 0.0

            log_dict = {
                # Core metrics
                "train/loss": loss.item(),
                "train/loss_teacher": loss_teacher.item(),  # v4.7.0
                "train/loss_ar": loss_ar.item(),            # v4.7.0
                "train/loss_texture": loss_texture.item(),
                "train/loss_lpips": lpips_loss_avg, # v4.6.2: LPIPS perceptual loss (averaged across sequence)

                # v4.7.1: TF vs AR LPIPS split (for adaptive brake)
                "train/loss_lpips_tf": prev_lpips_tf_avg,
                "train/loss_lpips_ar": prev_lpips_ar_avg,

                # v4.7.1: Action ranking loss
                "action_rank/loss": loss_action_rank.item(),

                # v4.11.0: IDM diagnostics
                "idm/loss": loss_idm.item(),
                "idm/loss_per_action": loss_idm.item() / 5,  # Normalize by action_dim
                "idm/max_span": MAX_IDM_SPAN,  # Track for reference

                # v4.6.0: Gumbel-Softmax tau annealing
                "train/gumbel_tau": max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9),

                # v4.7.1: AR curriculum diagnostics with adaptive brake
                "curriculum/seq_len": seq_len,
                "curriculum/ar_len": ar_len,
                "curriculum/ar_cutoff": ar_cutoff,
                "curriculum/lpips_tf_ema": lpips_tf_ema if lpips_tf_ema is not None else 0.0,
                "curriculum/lpips_ar_ema": lpips_ar_ema if lpips_ar_ema is not None else 0.0,
                "curriculum/lpips_ratio": (lpips_ar_ema / (lpips_tf_ema + 1e-6)) if (lpips_ar_ema and lpips_tf_ema) else 0.0,

                # Action diagnostics
                "action_diagnostics/film_gamma_magnitude": gamma_magnitude,
                "action_diagnostics/film_beta_magnitude": beta_magnitude,
                "action_diagnostics/sensitivity": action_sensitivity,
                "action_diagnostics/action_magnitude": A_seq.abs().mean().item(),  # v4.10.0: Track dataset action distribution

                # Existing diagnostics
                "diagnostics/entropy": entropy.item(),
                "diagnostics/confidence": confidence.item(),
                "diagnostics/unique_codes": unique_codes,

                # v4.5: NEW diagnostics
                "diagnostics/confidence_std": confidence_std.item(),
                "diagnostics/confidence_min": confidence_min.item(),

                "train/grad_norm": float(grad_norm),
                "grad/film_norm": grad_film,           # v4.7.0
                "grad/dynamics_norm": grad_dynamics,   # v4.7.0
                "grad/spatial_gradient": spatial_gradient.item(),

                # v4.7.2: Timing & throughput metrics
                # Keep total step time in ms + LPIPS ms/call as absolute units; everything else is logged as percentages below.
                "timing/step_total_ms": timing_stats['step_total'] * 1000 if timing_stats['step_total'] else 0,
                "timing/loss_lpips_call_ms": timing_stats['loss_lpips_call'] * 1000 if timing_stats['loss_lpips_call'] else 0,
                "timing/throughput_steps_per_sec": 1.0 / timing_stats['step_total'] if timing_stats['step_total'] and timing_stats['step_total'] > 0 else 0
            }

            wandb.log(log_dict, step=global_step)

    # Image Logs (Visual Confirmation)
    if global_step % IMAGE_LOG_STEPS == 0:
        log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)
        # v4.8.1: Multi-step action conditioning validation (replaces old AR rollout viz)
        action_metrics = validate_action_conditioning(model, vqvae_model, Z_seq, A_seq, Z_target, global_step)
        if wandb and action_metrics:
            wandb.log(action_metrics, step=global_step)

    # v4.6.6: Milestone Save (every 10k steps)
    if global_step > 0 and global_step % MILESTONE_SAVE_STEPS == 0:
        save_checkpoint(global_step, is_milestone=True)

    # Emergency Save (time-based)
    if (time.time() - last_emergency_save_time) > EMERGENCY_SAVE_INTERVAL_HRS * 3600:
        save_checkpoint(global_step, is_emergency=True)
        last_emergency_save_time = time.time()

    global_step += 1

# Training complete
print(f"Training complete! Reached {global_step} steps (target: {MAX_STEPS})")
save_checkpoint(global_step, is_milestone=True)
if wandb: wandb.finish()
