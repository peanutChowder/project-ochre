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
    print("⚠️  wandb not found, logging disabled.")


# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PATHS (UPDATE THESE) ---
DATA_DIR = "/kaggle/input/minerl-64x64-vqvae-latents-wasd-pitch-yaw"
VQVAE_PATH = "/kaggle/working/vqvae.pt" # Points to the checkpoint saved by your VQVAE script
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-5             
WARMUP_STEPS = 500   
MIN_LR = 1e-6         
USE_CHECKPOINTING = False 

# --- DUAL LOSS WEIGHTS ---
# SEMANTIC: Punishes "wrong color/texture" (Embedding MSE)
SEMANTIC_WEIGHT = 10.0  
# NEIGHBOR: Tolerates "1-pixel spatial shift" (Spatial Cross Entropy)
NEIGHBOR_WEIGHT = 1.0   
NEIGHBOR_KERNEL = 3     
NEIGHBOR_EXACT_MIX = 0.1 # 10% strict exact-pixel loss, 90% neighbor tolerant

# --- CURRICULUM ---
CURRICULUM_SEQ_LEN = False
BASE_SEQ_LEN = 1       
MAX_SEQ_LEN = 50       
SEQ_LEN_INCREASE_STEPS = 5000  

CURRICULUM_AR = True
AR_START_STEP = 0      
AR_RAMP_STEPS = 30000    
AR_ROLLOUT_MAX = 49       

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v4.5-semantic-vis"
MODEL_OUT_PREFIX = "ochre-v4.5"
RESUME_PATH = ""  

LOG_STEPS = 10
IMAGE_LOG_STEPS = 500 # Log visual reconstruction every N steps
REPORT_SPIKES = True
SPIKE_LOSS = 15.0 
SPIKE_GRAD = 400
EMERGENCY_SAVE_INTERVAL_HRS = 11.8

# ==========================================
# LOSS & HELPER FUNCTIONS
# ==========================================

class SemanticCodebookLoss(nn.Module):
    """
    Computes MSE in Embedding Space.
    Prevents "Gray Soup" by rewarding visually similar textures.
    """
    def __init__(self, codebook_tensor):
        super().__init__()
        # codebook_tensor: (Num_Embeddings, Embedding_Dim)
        self.register_buffer('codebook', codebook_tensor.clone().detach())

    def forward(self, logits, target_indices):
        # 1. Softmax -> Expected Vector
        probs = F.softmax(logits, dim=-1) 
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

def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-5: return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def log_images_to_wandb(vqvae, Z_target, logits, global_step):
    """
    Visualizes Ground Truth vs Prediction.
    """
    if wandb is None: return
    with torch.no_grad():
        # Take the last frame of the sequence
        gt_indices = Z_target[:, -1] 
        pred_indices = logits.argmax(dim=1) 
        
        # Decode
        gt_rgb = vqvae.decode_code(gt_indices)
        pred_rgb = vqvae.decode_code(pred_indices)
        
        # Visualize first 4
        n = min(4, gt_rgb.size(0))
        vis_gt = gt_rgb[:n].detach().cpu()
        vis_pred = pred_rgb[:n].detach().cpu()
        
        # Stack: Top=GT, Bottom=Pred
        grid = make_grid(torch.cat([vis_gt, vis_pred], dim=0), nrow=n, normalize=False)
        
        wandb.log({
            "visuals/reconstruction": wandb.Image(grid, caption="Top: GT | Bottom: Pred")
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
# We assume the user's provided VQVAE class structure matches the checkpoint
conf = vqvae_ckpt.get("config", {})
vqvae_model = VQVAE(
    embedding_dim=conf.get("embedding_dim", 256), # Default to standard if missing
    num_embeddings=conf.get("codebook_size", 1024)
).to(DEVICE)

# Load state dict (handle potential key mismatches)
state_dict = vqvae_ckpt
if "model_state" in vqvae_ckpt: state_dict = vqvae_ckpt["model_state"]
elif "quantizer" in vqvae_ckpt: # Legacy check
    # If the user saved separate dicts, we might need to reconstruct, but
    # standard VQVAE script saves the whole model in root or model_state
    pass 

# Clean up keys if necessary (e.g. remove 'module.' prefix)
# Then load
try:
    vqvae_model.load_state_dict(state_dict, strict=False)
except Exception as e:
    print(f"⚠️ Note: strict loading failed ({e}). Attempting non-strict...")
    # This is fine for us as long as decoder and quantizer.embedding load
    
vqvae_model.eval()
vqvae_model.requires_grad_(False)
print("✅ VQ-VAE Loaded.")

# B. Initialize World Model
model = WorldModelConvFiLM(action_dim=5, H=18, W=32, use_checkpointing=USE_CHECKPOINTING).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

# C. Setup Semantic Loss with loaded codebook
# The codebook tensor is typically: vqvae_model.vq_vae.embedding
codebook = vqvae_model.vq_vae.embedding.clone().detach()
semantic_criterion = SemanticCodebookLoss(codebook).to(DEVICE)
print(f"✅ Semantic Loss Initialized with codebook shape {codebook.shape}")

# D. Resume Logic
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

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    # Curriculum update
    ar_len = 0
    if CURRICULUM_AR and global_step > AR_START_STEP:
        progress = (global_step - AR_START_STEP) / AR_RAMP_STEPS
        progress = max(0.0, min(1.0, progress))
        ar_len = int(progress * AR_ROLLOUT_MAX)

    current_base = BASE_SEQ_LEN
    if CURRICULUM_SEQ_LEN:
         current_base = min(BASE_SEQ_LEN + (global_step // SEQ_LEN_INCREASE_STEPS), MAX_SEQ_LEN)
    
    seq_len = min(max(current_base, ar_len + 1), MAX_SEQ_LEN)
    ar_len = min(ar_len, seq_len - 1)
    if ar_len < 0: ar_len = 0
    
    print(f"🧩 Curriculum: seq_len={seq_len}, ar_len={ar_len}")

    dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
    if len(dataset) == 0: continue
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, pin_memory=True, persistent_workers=False)

    for batch_idx, batch in enumerate(loader):
        t0_batch = time.time()
        Z_seq, A_seq, Z_target, _, _, _ = batch
        Z_seq, A_seq, Z_target = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target.to(DEVICE)
        B, K, H, W = Z_seq.shape
        step_losses = []
        logits_last = None

        # LR Warmup
        if global_step <= WARMUP_STEPS:
            new_lr = MIN_LR + (global_step / WARMUP_STEPS) * (LR - MIN_LR)
            for pg in optimizer.param_groups: pg['lr'] = new_lr

        optimizer.zero_grad()
        
        # Embeddings & FiLM
        X_seq = model.compute_embeddings(Z_seq)
        Gammas_seq, Betas_seq = model.compute_film(A_seq)
        h_state = model.init_state(B, device=DEVICE)
        
        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            ar_cutoff = K - ar_len
            
            for t in range(K):
                # Autoregressive Switch
                if t >= ar_cutoff:
                    # In training, we use hard argmax or gumbel. 
                    # Using argmax here for simplicity in 'Stateful' simulation
                    z_in = logits_last.argmax(dim=1) 
                    x_in = None 
                else:
                    z_in = torch.tensor(0, device=DEVICE)
                    x_in = X_seq[:, t]

                # Step
                g_t = Gammas_seq[:, :, t]
                b_t = Betas_seq[:, :, t]
                
                if USE_CHECKPOINTING:
                    logits_t, h_state = checkpoint(model.step, z_in, torch.tensor(0, device=DEVICE), h_state, x_in, g_t, b_t, use_reentrant=False)
                else:
                    logits_t, h_state = model.step(z_in, None, h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)
                
                logits_last = logits_t
                
                # --- LOSS CALCULATION ---
                # 1. Spatial Loss (Handle Jitter)
                loss_space = neighborhood_token_loss(
                    logits_t, Z_target[:, t],
                    kernel_size=NEIGHBOR_KERNEL, mix_exact=NEIGHBOR_EXACT_MIX
                )
                
                # 2. Semantic Loss (Handle Blur)
                # Flatten for codebook processing
                logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                target_flat = Z_target[:, t].reshape(-1)
                loss_texture = semantic_criterion(logits_flat, target_flat)
                
                # Weighted Sum
                loss_step = (NEIGHBOR_WEIGHT * loss_space) + (SEMANTIC_WEIGHT * loss_texture)
                step_losses.append(loss_step)

            loss = torch.stack(step_losses).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # --- DIAGNOSTICS & LOGGING ---
        if (loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD) and REPORT_SPIKES:
            print(f"\n⚡ Spike: step={global_step} loss={loss.item():.2f} grad={float(grad_norm):.2f}")
            print(f"   (Space: {loss_space.item():.4f}, Tex: {loss_texture.item():.4f})")

        total_loss += loss.item()
        
        # Periodic Logs
        if global_step % LOG_STEPS == 0:
            # Calculate Entropy (Uncertainty)
            with torch.no_grad():
                probs = F.softmax(logits_last.float(), dim=1) # (B, C, H, W)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
                confidence = probs.max(dim=1)[0].mean()
                unique_codes = logits_last.argmax(dim=1).unique().numel()

            if wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_space": loss_space.item(),
                    "train/loss_texture": loss_texture.item(),
                    "diagnostics/entropy": entropy.item(),
                    "diagnostics/confidence": confidence.item(),
                    "diagnostics/unique_codes": unique_codes,
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
print("✅ Training complete.")