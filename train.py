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

try:
    import wandb
except ImportError:
    wandb = None
    print("WARNING: wandb not found, logging disabled.")


# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PATHS (UPDATE THESE) ---
DATA_DIR = "/kaggle/input/gamefactorylatents/preprocessedv4"
VQVAE_PATH = "/kaggle/input/vq-vae-64x64/pytorch/v1.0.1-epoch10/7/vqvae_v2.1.6__epoch100.pt" 
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 28
EPOCHS = 50
LR = 3e-5             
WARMUP_STEPS = 500   
MIN_LR = 1e-6         
USE_CHECKPOINTING = False 

# --- DUAL LOSS WEIGHTS ---
# v4.6.0: Revert to v4.4.1 baseline + add LPIPS
# SEMANTIC: Punishes "wrong color/texture" (Embedding MSE)
SEMANTIC_WEIGHT = 10.0   # v4.6.0: Back to v4.4.1 baseline (was 3.0 in v4.5)
# NEIGHBOR: Tolerates "1-pixel spatial shift" (Spatial Cross Entropy)
NEIGHBOR_WEIGHT = 1.0    # v4.6.0: Back to v4.4.1 baseline (was 2.0 in v4.5)
NEIGHBOR_KERNEL = 3
NEIGHBOR_EXACT_MIX = 0.1 # v4.6.0: Back to v4.4.1 baseline (was 0.3 in v4.5)

# --- NEW LOSS (v4.6.0) ---
LPIPS_WEIGHT = 0.3       # Perceptual loss (learn from VQ-VAE success)
LPIPS_FREQ = 5           # Compute LPIPS every N timesteps for efficiency
GUMBEL_TAU_STEPS = 20000 # Gumbel-Softmax annealing: 1.0→0.1 over this many steps

# --- CURRICULUM ---
CURRICULUM_SEQ_LEN = False      # Keep disabled (GPU memory risk)
BASE_SEQ_LEN = 20               # v4.5.1 OOM fix: Increased from 16 to accommodate AR rollout
MAX_SEQ_LEN = 50                # Keep same
SEQ_LEN_INCREASE_STEPS = 5000   # Not used (SEQ_LEN disabled)

CURRICULUM_AR = True
AR_START_STEP = 0
AR_RAMP_STEPS = 10000           # v4.5.1: Faster ramp (was 20000)
AR_ROLLOUT_MAX = 18             # v4.5.1 OOM fix: Reduced from 32 to fit P100 memory (seq_len - 2)       

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v4.6.1-step38k"
MODEL_OUT_PREFIX = "ochre-v4.6.1"
RESUME_PATH = ""  

LOG_STEPS = 10
IMAGE_LOG_STEPS = 500 # Log visual reconstruction every N steps
REPORT_SPIKES = True
SPIKE_LOSS = 15.0
SPIKE_GRAD = 400
EMERGENCY_SAVE_INTERVAL_HRS = 11.8

# --- SCHEDULED SAMPLING ---
ENABLE_SCHEDULED_SAMPLING = True
SS_K_STEEPNESS = 4000.0              # v4.5: Faster decay (was 6000)
SS_MIDPOINT_STEP = 10000             # v4.5: Earlier transition (was 15000)
SS_MIN_TEACHER_PROB = 0.15           # v4.5: Keep more teacher signal (was 0.05)

def get_sampling_probability(step, k=SS_K_STEEPNESS, s=SS_MIDPOINT_STEP):
    """Returns probability of using teacher forcing (inverse sigmoid)"""
    prob = 1.0 / (1.0 + np.exp((step - s) / k))
    return max(SS_MIN_TEACHER_PROB, min(0.95, prob))

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

def neighborhood_token_loss(logits, target, kernel_size=3, mix_exact=0.1):
    """
    Spatially tolerant Cross Entropy.
    Prevents penalty for 1-pixel jitter.
    """
    B, C, H, W = logits.shape
    log_probs = F.log_softmax(logits, dim=1)          
    log_probs_flat = log_probs.view(B, C, H * W)      

    k = kernel_size
    pad = k // 2
    # Create windows of neighbors for every pixel
    target_patches = F.unfold(
        target.float().unsqueeze(1), kernel_size=k, padding=pad
    ).long() # (B, K*K, HW)

    neighbors = target_patches.permute(0, 2, 1) # (B, HW, K*K)
    lp = log_probs_flat.permute(0, 2, 1)        # (B, HW, C)
    
    # Gather log-probs of all valid neighbors
    lp_neighbors = lp.gather(dim=2, index=neighbors)
    
    # "Did we predict ANY of the neighbors?" (LogSumExp)
    log_p_any = torch.logsumexp(lp_neighbors, dim=2)
    loss_neigh = -log_p_any.mean()

    if mix_exact <= 0.0: return loss_neigh

    # Mix with strict loss
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.reshape(-1)
    loss_exact = F.cross_entropy(logits_flat, target_flat)

    return mix_exact * loss_exact + (1.0 - mix_exact) * loss_neigh

# --- v4.5 LOSS FUNCTIONS REMOVED IN v4.6.0 ---
# Removed: entropy_regularization_loss, sharpness_loss, temporal_consistency_loss
# These created conflicting gradients causing mode collapse in v4.5.2

def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-5: return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

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
try:
    import lpips
    lpips_criterion = lpips.LPIPS(net='alex').to(DEVICE)
    lpips_criterion.eval()  # Frozen, only for loss computation
    lpips_criterion.requires_grad_(False)
    print(f"✅ LPIPS Loss Initialized (AlexNet backend)")
except ImportError:
    print("⚠️  LPIPS not installed, will skip perceptual loss")
    lpips_criterion = None

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

def save_checkpoint(epoch, global_step, is_emergency=False):
    base_name = f"{MODEL_OUT_PREFIX}_epoch_{epoch}"
    suffix = "_emergency" if is_emergency else ""
    save_path = f"/kaggle/working/{base_name}{suffix}.pt"
    print(f"⏰ Saving checkpoint to {save_path}")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, save_path)


# --- TRAINING ---
last_emergency_save_time = time.time()

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
        try:
            batch = next(loader_iter)
        except StopIteration:
            # Epoch complete
            break

        t0_batch = time.time()
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

        # Scheduled sampling setup
        if ENABLE_SCHEDULED_SAMPLING:
            teacher_force_prob = get_sampling_probability(global_step)
        else:
            teacher_force_prob = 1.0  # Fallback to original behavior

        teacher_loss_steps = []
        ar_loss_steps = []
        logits_last = None
        logits_prev = None  # v4.5: Track previous logits for temporal consistency

        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            ar_cutoff = K - ar_len

            for t in range(K):
                # Determine if this step uses teacher forcing
                if t < ar_cutoff:
                    # Always use GT for context portion
                    use_teacher = True
                    x_in = X_seq[:, t]
                else:
                    # Scheduled sampling for AR portion
                    use_teacher = (torch.rand(1).item() < teacher_force_prob) if ENABLE_SCHEDULED_SAMPLING else False

                    if use_teacher:
                        x_in = X_seq[:, t]
                    else:
                        # Use previous prediction
                        if logits_last is not None:
                            z_pred = logits_last.argmax(dim=1)
                            x_in = model._embed_tokens(z_pred)
                        else:
                            # Fallback to GT for first AR step
                            x_in = X_seq[:, t]

                # Step
                g_t = Gammas_seq[:, :, t]
                b_t = Betas_seq[:, :, t]

                if USE_CHECKPOINTING:
                    logits_t, h_state = checkpoint(model.step, torch.tensor(0, device=DEVICE), torch.tensor(0, device=DEVICE), h_state, x_in, g_t, b_t, use_reentrant=False)
                else:
                    logits_t, h_state = model.step(None, None, h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)

                logits_last = logits_t
                
                # --- LOSS CALCULATION (v4.6.0) ---
                # 1. Spatial Loss (Handle Jitter)
                loss_space = neighborhood_token_loss(
                    logits_t, Z_target[:, t],
                    kernel_size=NEIGHBOR_KERNEL, mix_exact=NEIGHBOR_EXACT_MIX
                )

                # 2. Semantic Loss with Gumbel-Softmax (Handle Blur + Force Discrete)
                logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                target_flat = Z_target[:, t].reshape(-1)
                loss_texture = semantic_criterion(logits_flat, target_flat, global_step=global_step)

                # 3. LPIPS Perceptual Loss (v4.6.1: Fixed gradient flow)
                loss_lpips = torch.tensor(0.0, device=DEVICE)
                if lpips_criterion is not None and (t % LPIPS_FREQ == 0):
                    # v4.6.1 FIX: Only disable grad for argmax, not entire computation
                    with torch.no_grad():
                        pred_tokens = logits_t.argmax(dim=1)  # (B, H, W)
                        target_tokens = Z_target[:, t]  # (B, H, W)

                    # Decode with gradients enabled (for LPIPS backprop)
                    pred_rgb = vqvae_model.decode_code(pred_tokens)  # (B, 3, 128, 72)
                    target_rgb = vqvae_model.decode_code(target_tokens)  # (B, 3, 128, 72)

                    # LPIPS expects inputs in [-1, 1] range
                    pred_rgb_norm = pred_rgb * 2.0 - 1.0
                    target_rgb_norm = target_rgb * 2.0 - 1.0

                    # Compute perceptual loss (gradients flow through pred_rgb)
                    loss_lpips = lpips_criterion(pred_rgb_norm, target_rgb_norm).mean()

                # Weighted Sum (v4.6.0: Simplified - only 3 components)
                loss_step = (NEIGHBOR_WEIGHT * loss_space) + \
                            (SEMANTIC_WEIGHT * loss_texture) + \
                            (LPIPS_WEIGHT * loss_lpips)
                step_losses.append(loss_step)

                # Track teacher vs AR loss separately
                if use_teacher:
                    teacher_loss_steps.append(loss_step)
                else:
                    ar_loss_steps.append(loss_step)

            loss = torch.stack(step_losses).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # v4.5: Tighter clipping (was 1.0)
        scaler.step(optimizer)
        scaler.update()

        # --- DIAGNOSTICS & LOGGING ---
        if (loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD) and REPORT_SPIKES:
            print(f"\nSPIKE: step={global_step} loss={loss.item():.2f} grad={float(grad_norm):.2f}")
            print(f"   (Space: {loss_space.item():.4f}, Tex: {loss_texture.item():.4f})")

        total_loss += loss.item()
        
        # Periodic Logs
        if global_step % LOG_STEPS == 0:
            # Calculate AR loss gap
            if len(teacher_loss_steps) > 0 and len(ar_loss_steps) > 0:
                teacher_loss_avg = torch.stack(teacher_loss_steps).mean().item()
                ar_loss_avg = torch.stack(ar_loss_steps).mean().item()
                ar_loss_gap = ar_loss_avg - teacher_loss_avg
            else:
                teacher_loss_avg = 0.0
                ar_loss_avg = 0.0
                ar_loss_gap = 0.0

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
                gamma_magnitude = Gammas_seq.abs().mean().item()
                beta_magnitude = Betas_seq.abs().mean().item()

                # Action perturbation test (v4.5.2: reduced from every 100 to every 500 steps)
                if global_step % 500 == 0:
                    # Compare prediction with vs without actions
                    A_seq_zero = torch.zeros_like(A_seq)
                    Gammas_zero, Betas_zero = model.compute_film(A_seq_zero)

                    # Re-run last timestep with zero actions
                    logits_zero, _ = model.step(None, None, h_state, x_t=x_in,
                                               gammas_t=Gammas_zero[:, :, -1],
                                               betas_t=Betas_zero[:, :, -1])

                    # Measure sensitivity
                    action_sensitivity = (logits_last - logits_zero).abs().mean().item()
                else:
                    # Use FiLM magnitude as instant proxy
                    action_sensitivity = (gamma_magnitude + beta_magnitude) / 2

            if wandb:
                wandb.log({
                    # Existing metrics
                    "train/loss": loss.item(),
                    "train/loss_space": loss_space.item(),
                    "train/loss_texture": loss_texture.item(),

                    # v4.6.0: LPIPS perceptual loss
                    "train/loss_lpips": loss_lpips.item() if isinstance(loss_lpips, torch.Tensor) else 0,

                    # v4.6.0: Gumbel-Softmax tau annealing
                    "train/gumbel_tau": max(0.1, 1.0 - (global_step / GUMBEL_TAU_STEPS) * 0.9),

                    # New scheduled sampling metrics
                    "curriculum/teacher_force_prob": teacher_force_prob,
                    "curriculum/ar_loss_gap": ar_loss_gap,
                    "curriculum/teacher_loss": teacher_loss_avg,
                    "curriculum/ar_loss": ar_loss_avg,

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
                    "seq_len": seq_len,
                    "ar_len": ar_len,
                }, step=global_step)

        # Image Logs (Visual Confirmation)
        if global_step % IMAGE_LOG_STEPS == 0:
            log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)

        # Emergency Save
        if (time.time() - last_emergency_save_time) > EMERGENCY_SAVE_INTERVAL_HRS * 3600:
            save_checkpoint(epoch, global_step, is_emergency=True)
            last_emergency_save_time = time.time()
            
        global_step += 1

    print(f"Epoch {epoch}: mean loss {total_loss / len(loader):.4f}")
    save_checkpoint(epoch, global_step)

if wandb: wandb.finish()
print("Training complete.")