#!/usr/bin/env python3
"""
Checklist before running:
1. Ensure 'WorldModelConvFiLM' class is defined/imported.
2. Ensure 'VQVAE' class (and dependencies) is defined/imported.
3. Update VQVAE_PATH to point to your trained 'vqvae.pt'.
4. Update DATA_DIR to your dataset location.
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

# --- PATHS ---
DATA_DIR = "/kaggle/input/gamefactorylatents/preprocessedv4"
VQVAE_PATH = "/kaggle/input/vq-vae-64x64/pytorch/v1.0.1-epoch10/7/vqvae_v2.1.6__epoch100.pt"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
MODEL_OUT_PREFIX = "ochre-v4.3.2"
RESUME_PATH = ""  

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 16
EPOCHS = 50           # Total number of "Pseudo-Epochs" to run
STEPS_PER_EPOCH = 10000 # Force curriculum update/dataset refresh every N steps
LR = 3e-5             
WARMUP_STEPS = 500   
MIN_LR = 1e-6         
USE_CHECKPOINTING = True 

# --- CONTEXT WINDOW SETTINGS ---
GT_CONTEXT_LEN = 16       
MAX_SEQ_LEN = 100 

# Autoregressive Rollout 
CURRICULUM_AR = True
AR_START_STEP = 0      
AR_ROLLOUT_MAX = 49    
AR_RAMP_EPOCHS = 5     

# --- DUAL LOSS WEIGHTS ---
SEMANTIC_WEIGHT = 10.0  
NEIGHBOR_WEIGHT = 1.0   
NEIGHBOR_KERNEL = 3     
NEIGHBOR_EXACT_MIX = 0.1 

# --- LOGGING ---
PROJECT = "project-ochre"
RUN_NAME = "v4.3.2-epoch0"
LOG_STEPS = 10
IMAGE_LOG_STEPS = 10000 
REPORT_SPIKES = True
SPIKE_LOSS = 15.0 
SPIKE_GRAD = 400
EMERGENCY_SAVE_INTERVAL_HRS = 11.8

# ==========================================
# LOSS & HELPER FUNCTIONS
# ==========================================

class SemanticCodebookLoss(nn.Module):
    def __init__(self, codebook_tensor):
        super().__init__()
        self.register_buffer('codebook', codebook_tensor.clone().detach())

    def forward(self, logits, target_indices):
        """
        Memory Efficient (Chunked) Forward Pass.
        Handles both (B, C, H, W) and (B, H, W, C) logits automatically.
        """
        # [FIX] Handle Permutation:
        # If logits are (B, C, H, W), move C to the end -> (B, H, W, C)
        if logits.dim() == 4 and logits.shape[1] == self.codebook.shape[0]:
             logits = logits.permute(0, 2, 3, 1)
        
        # Flatten batch/spatial dims for chunking
        # Now shape is guaranteed to be (..., Codebook_Size)
        logits_flat = logits.reshape(-1, logits.size(-1)) # (N, 1024)
        targets_flat = target_indices.reshape(-1)         # (N)
        
        # Chunk size: Process 4096 pixels at a time
        CHUNK_SIZE = 4096 
        total_loss = 0.0
        n_samples = logits_flat.size(0)
        
        # Iterate in chunks to keep peak memory low
        for i in range(0, n_samples, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, n_samples)
            logits_chunk = logits_flat[i : end]
            target_chunk = targets_flat[i : end]
            
            # 1. Softmax -> Expected Vector
            probs = F.softmax(logits_chunk, dim=-1)
            pred_vectors = torch.matmul(probs, self.codebook)
            
            # 2. Ground Truth Vector
            target_vectors = F.embedding(target_chunk, self.codebook)
            
            # 3. MSE (Sum reduction, we divide by total later)
            chunk_loss = F.mse_loss(pred_vectors, target_vectors, reduction='sum')
            total_loss += chunk_loss

        # Average over total samples
        return total_loss / (n_samples * self.codebook.shape[1])

def neighborhood_token_loss(logits, target, kernel_size=3, mix_exact=0.1):
    B, C, H, W = logits.shape
    log_probs = F.log_softmax(logits, dim=1)          
    log_probs_flat = log_probs.view(B, C, H * W)      

    k = kernel_size
    pad = k // 2
    target_patches = F.unfold(
        target.float().unsqueeze(1), kernel_size=k, padding=pad
    ).long()

    neighbors = target_patches.permute(0, 2, 1) 
    lp = log_probs_flat.permute(0, 2, 1)        
    lp_neighbors = lp.gather(dim=2, index=neighbors)
    log_p_any = torch.logsumexp(lp_neighbors, dim=2)
    loss_neigh = -log_p_any.mean()

    if mix_exact <= 0.0: return loss_neigh

    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.reshape(-1)
    loss_exact = F.cross_entropy(logits_flat, target_flat)

    return mix_exact * loss_exact + (1.0 - mix_exact) * loss_neigh

def log_images_to_wandb(vqvae, Z_target, logits, global_step):
    if wandb is None: return
    with torch.no_grad():
        gt_indices = Z_target[:, -1] 
        pred_indices = logits.argmax(dim=1) 
        
        gt_rgb = vqvae.decode_code(gt_indices)
        pred_rgb = vqvae.decode_code(pred_indices)
        
        n = min(4, gt_rgb.size(0))
        vis_gt = gt_rgb[:n].detach().cpu()
        vis_pred = pred_rgb[:n].detach().cpu()
        
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

# ==========================================
# 5. SETUP & TRAINING LOOP
# ==========================================

if wandb:
    wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
                config=dict(batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS))

# --- A. Load VQ-VAE (Frozen) ---
print(f"📥 Loading VQ-VAE from {VQVAE_PATH}...")
if not os.path.exists(VQVAE_PATH):
    raise FileNotFoundError(f"VQVAE checkpoint not found at {VQVAE_PATH}")

vqvae_ckpt = torch.load(VQVAE_PATH, map_location=DEVICE)

# 1. Extract Config
conf = vqvae_ckpt.get("config", {})
LOADED_EMB_DIM = conf.get("embedding_dim", 384)
LOADED_CB_SIZE = conf.get("codebook_size", 1024) 
print(f"   Detected VQ-VAE Config -> Dim: {LOADED_EMB_DIM}, Codebook Size: {LOADED_CB_SIZE}")

# 2. Instantiate VQ-VAE
vqvae_model = VQVAE(
    embedding_dim=LOADED_EMB_DIM,
    num_embeddings=LOADED_CB_SIZE
).to(DEVICE)

# 3. Load Weights Correctly
try:
    print("   Loading VQ-VAE sub-modules...")
    vqvae_model.encoder.load_state_dict(vqvae_ckpt["encoder"])
    vqvae_model.decoder.load_state_dict(vqvae_ckpt["decoder"])
    vqvae_model.vq_vae.load_state_dict(vqvae_ckpt["quantizer"])
    print("✅ VQ-VAE Weights Loaded Successfully.")
except KeyError as e:
    print(f"⚠️ Standard loading failed ({e}). Attempting fallback to full state_dict...")
    if "model_state" in vqvae_ckpt:
        vqvae_model.load_state_dict(vqvae_ckpt["model_state"], strict=False)

vqvae_model.eval()
vqvae_model.requires_grad_(False)

# --- B. Initialize World Model ---
model = WorldModelConvFiLM(
    action_dim=5, 
    H=18, 
    W=32, 
    codebook_size=LOADED_CB_SIZE,
    use_checkpointing=USE_CHECKPOINTING
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

# --- C. Setup Semantic Loss ---
codebook_raw = vqvae_model.vq_vae.embedding.clone().detach() 
if codebook_raw.shape[0] == LOADED_EMB_DIM and codebook_raw.shape[1] == LOADED_CB_SIZE:
    codebook = codebook_raw.t() 
else:
    codebook = codebook_raw

semantic_criterion = SemanticCodebookLoss(codebook).to(DEVICE)
print(f"✅ Semantic Loss Initialized. Shape: {codebook.shape}")

# --- D. Resume Logic ---
global_step = 0
start_epoch = 1

if RESUME_PATH and os.path.exists(RESUME_PATH):
    print(f"🔄 Resuming training from {RESUME_PATH}")
    ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)
    print(f"   Resumed at Global Step: {global_step}")
else:
    print("⚠️  No resume path found. Starting from Step 0.")

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


# --- E. TRAINING LOOP ---
last_emergency_save_time = time.time()

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    # --- Curriculum Calculation ---
    ar_len = 0
    if CURRICULUM_AR and global_step > AR_START_STEP:
        total_ramp_steps = AR_RAMP_EPOCHS * STEPS_PER_EPOCH
        progress = (global_step - AR_START_STEP) / total_ramp_steps
        progress = max(0.0, min(1.0, progress))
        ar_len = int(progress * AR_ROLLOUT_MAX)
    
    seq_len = GT_CONTEXT_LEN + ar_len
    if seq_len > MAX_SEQ_LEN:
        seq_len = MAX_SEQ_LEN
    
    # Cap AR len if it consumes the entire sequence
    ar_len = min(ar_len, seq_len - 1)
    if ar_len < 0: ar_len = 0
    
    print(f"🧩 Curriculum: seq_len={seq_len}, ar_len={ar_len}, gt_ctx={seq_len - ar_len} (Step {global_step})")

    dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
    if len(dataset) == 0: 
        print(f"⚠️ Dataset empty. Skipping epoch.")
        continue
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, pin_memory=True, persistent_workers=False)

    steps_in_this_epoch = 0

    for batch_idx, batch in enumerate(loader):
        t0_batch = time.time()
        Z_seq, A_seq, Z_target, _, _, _ = batch
        Z_seq, A_seq, Z_target = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target.to(DEVICE)
        B, K, H, W = Z_seq.shape
        step_losses = []
        logits_last = None

        if global_step <= WARMUP_STEPS:
            new_lr = MIN_LR + (global_step / WARMUP_STEPS) * (LR - MIN_LR)
            for pg in optimizer.param_groups: pg['lr'] = new_lr

        optimizer.zero_grad()
        
        # Pre-compute Embeddings & FiLM
        X_seq = model.compute_embeddings(Z_seq)
        Gammas_seq, Betas_seq = model.compute_film(A_seq)
        h_state = model.init_state(B, device=DEVICE)
        
        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            ar_cutoff = K - ar_len
            
            # Create a dummy tensor that requires grad for checkpoint consistency
            # This is never used mathematically if x_t is passed, but ensures signature validity
            dummy_z = torch.tensor(0, device=DEVICE)
            dummy_a = torch.tensor(0, device=DEVICE)

            for t in range(K):
                # Prepare Inputs
                if t >= ar_cutoff:
                    # AR Mode: Use previous prediction
                    z_idx = logits_last.argmax(dim=1) 
                    
                    # [CHECKPOINT SAFETY FIX]
                    # Compute embedding HERE so we always pass a tensor to model.step
                    # instead of passing None. This ensures 'checkpoint' works correctly.
                    x_in = model._embed_tokens(z_idx) 
                else:
                    # Teacher Forcing: Use pre-computed embedding
                    x_in = X_seq[:, t]

                g_t = Gammas_seq[:, :, t]
                b_t = Betas_seq[:, :, t]
                
                if USE_CHECKPOINTING:
                    # We pass x_in (Embedding) which has requires_grad=True
                    # We pass dummy_z, dummy_a which are ignored inside step because x_in is present
                    logits_t, h_state = checkpoint(
                        model.step, 
                        dummy_z, 
                        dummy_a, 
                        h_state, 
                        x_in, 
                        g_t, 
                        b_t, 
                        use_reentrant=False
                    )
                else:
                    logits_t, h_state = model.step(None, None, h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)
                
                logits_last = logits_t
                
                # --- Loss Calculation ---
                # 1. Spatial Loss
                loss_space = neighborhood_token_loss(
                    logits_t, Z_target[:, t],
                    kernel_size=NEIGHBOR_KERNEL, mix_exact=NEIGHBOR_EXACT_MIX
                )
                
                # 2. Semantic Loss (Chunked for Memory Safety)
                # Reshaping handled inside the class now
                loss_texture = semantic_criterion(logits_t, Z_target[:, t])
                
                loss_step = (NEIGHBOR_WEIGHT * loss_space) + (SEMANTIC_WEIGHT * loss_texture)
                step_losses.append(loss_step)

            loss = torch.stack(step_losses).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if (loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD) and REPORT_SPIKES:
            print(f"\n⚡ Spike: step={global_step} loss={loss.item():.2f} grad={float(grad_norm):.2f}")

        total_loss += loss.item()
        
        # Logs
        if global_step % LOG_STEPS == 0:
            with torch.no_grad():
                probs = F.softmax(logits_last.float(), dim=1)
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

        if global_step % IMAGE_LOG_STEPS == 0:
            log_images_to_wandb(vqvae_model, Z_target, logits_last, global_step)

        if (time.time() - last_emergency_save_time) > EMERGENCY_SAVE_INTERVAL_HRS * 3600:
            save_checkpoint(epoch, global_step, is_emergency=True)
            last_emergency_save_time = time.time()
            
        global_step += 1
        steps_in_this_epoch += 1

        if steps_in_this_epoch >= STEPS_PER_EPOCH:
            print(f"🔄 Pseudo-epoch limit reached ({STEPS_PER_EPOCH} steps). Updating curriculum...")
            break 

    print(f"Epoch {epoch}: mean loss {total_loss / (len(loader) + 1e-6):.4f}")
    save_checkpoint(epoch, global_step)

if wandb: wandb.finish()
print("✅ Training complete.")