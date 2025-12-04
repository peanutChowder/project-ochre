#!/usr/bin/env python3
"""
Checklist:
- Update output model name
- Update run name
- update vqvae checkpoint
- update world model checkpoint
"""
import os, time, json, math, numpy as np, torch, random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import checkpoint

try:
    import wandb
except ImportError:
    wandb = None
    print("⚠️  wandb not found, logging disabled.")

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/kaggle/input/minerl-64x64-vqvae-latents-wasd-pitch-yaw"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-5             # Base learning rate after warmup
WARMUP_STEPS = 500   # Number of steps for linear LR warmup
MIN_LR = 1e-6         # Starting learning rate for warmup
USE_CHECKPOINTING = False # Disable to improve throughput; enable if OOM

# --- CURRICULUM 1: SEQUENCE LENGTH (Context Window) ---
# Gradually increases the BPTT horizon / memory length
CURRICULUM_SEQ_LEN = False
BASE_SEQ_LEN = 1       # Start with short context
MAX_SEQ_LEN = 50       # Target max context
SEQ_LEN_INCREASE_STEPS = 5000  # Increase seq_len by 1 every N steps

# --- CURRICULUM 2: AUTOREGRESSIVE ROLLOUT (Deterministic Tail) ---
# We gradually increase the number of frames at the END of the sequence
# that are generated autoregressively (feeding predictions back in).
CURRICULUM_AR = True
AR_START_STEP = 0      # Step to start introducing AR frames
AR_RAMP_STEPS = 30000    # Steps to reach AR_ROLLOUT_MAX
AR_ROLLOUT_MAX = 49       # Max number of frames to generate autoregressively (MAX_SEQ_LEN - 1)


PROJECT = "project-ochre"
RUN_NAME = "v4.2-step60k"
MODEL_OUT_PREFIX = "ochre-v4.2"
resume_path = "checkpoints/ochrev4_fused.pt"  

LOG_STEPS = 10
REPORT_SPIKES = True
SPIKE_LOSS = 9
SPIKE_GRAD = 400
EMERGENCY_SAVE_INTERVAL_HRS = 11.8  # Save a checkpoint every N hours


# ---------- DATASET ----------
class GTTokenDataset(Dataset):
    """Loads pre-tokenized MineRL sequences for autoregressive world model training."""
    def __init__(self, manifest_path, root_dir, seq_len=6):
        with open(manifest_path, "r") as f:
            self.entries = json.load(f)["sequences"]
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.index_map = []
        for vid_idx, meta in enumerate(self.entries):
            L = meta["length"]
            # We need seq_len inputs + 1 target.
            # actions length is L-1.
            # We need valid indices for tokens[start : start+seq_len+1]
            # Max index is start+seq_len. This must be < L.
            # So start < L - seq_len.
            if L > seq_len + 1:
                for i in range(L - (seq_len + 1)):
                    self.index_map.append((vid_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid_idx, start = self.index_map[idx]
        meta = self.entries[vid_idx]
        path = os.path.join(self.root_dir, meta["file"])
        
        # Optimization: Use mmap_mode='r' to avoid loading the full file if possible.
        # If the file is compressed .npz, this might not mmap but will still load.
        # If uncompressed, it saves huge I/O.
        try:
            data = np.load(path, mmap_mode='r') 
        except ValueError:
            # Fallback if mmap_mode is not supported for this file type
            data = np.load(path)
            
        tokens = data["tokens"]   # shape (T, H, W) of discrete VQ-VAE indices over time
        actions = data["actions"] # shape (T-1, A) of continuous action vectors per frame

        # Take a window of length seq_len from this video.
        # Z_seq are the input tokens at times [t, ..., t+seq_len-1]
        # Z_target_seq are the next-step tokens [t+1, ..., t+seq_len]
        Z_seq = tokens[start:start + self.seq_len]
        # actions[i] is action between tokens[i] and tokens[i+1]
        A_seq = actions[start:start + self.seq_len]
        Z_target_seq = tokens[start + 1:start + self.seq_len + 1]
        
        # If mapped, copy to memory now to release file handle references if needed, 
        # or just return as is (torch will copy when converting to tensor).
        # explicitly converting to numpy array helps ensure we don't pass a memmap to torch if not needed,
        # but torch.tensor() handles numpy arrays fine.
        
        return (
            torch.tensor(np.array(Z_seq), dtype=torch.long),        # (K, H, W)
            torch.tensor(np.array(A_seq), dtype=torch.float32),     # (K, A)
            torch.tensor(np.array(Z_target_seq), dtype=torch.long), # (K, H, W)
            idx,                  
            vid_idx,             
            start                
        )

# Initialize wandb
if wandb:
    wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
                config=dict(batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS))
else:
    print("⚠️  wandb not found, logging disabled.")

# ---------- MODEL ----------
# Instantiate model with correct action dimension (5) and latent dims (18x32)
model = WorldModelConvFiLM(action_dim=5, H=18, W=32, use_checkpointing=USE_CHECKPOINTING).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
global_step = 0
start_epoch = 1
if os.path.exists(resume_path):
    print(f" Resuming training from {resume_path}")
    resume_checkpoint = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(resume_checkpoint["model_state"])
    
    if resume_checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        print(" Optimizer state loaded.")
    else:
        print(" ⚠️ Optimizer state not found in checkpoint (likely due to architecture upgrade). Starting with fresh optimizer.")
        
    start_epoch = resume_checkpoint["epoch"] + 1
    global_step = resume_checkpoint.get("global_step", 0)
    print(f"Resumed at epoch {start_epoch}, global step {global_step}")
else:
    print(" No checkpoint found — starting fresh.")


def save_checkpoint(epoch, global_step, model, optimizer, is_emergency=False):
    """Saves a training checkpoint."""
    base_name = f"{MODEL_OUT_PREFIX}_epoch_{epoch}"
    suffix = "_emergency" if is_emergency else ""
    save_path = f"/kaggle/working/{base_name}{suffix}.pt"
    
    save_type = "emergency" if is_emergency else "epoch"
    print(f"⏰ Saving {save_type} checkpoint to {save_path}")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, save_path)


# ---------- TRAINING LOOP ----------
last_emergency_save_time = time.time()
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    # --- 1. Determine AR Length (Curriculum) ---
    ar_len = 0
    if CURRICULUM_AR and global_step > AR_START_STEP:
        progress = (global_step - AR_START_STEP) / AR_RAMP_STEPS
        progress = max(0.0, min(1.0, progress))
        ar_len = int(progress * AR_ROLLOUT_MAX)

    # --- 2. Determine Loaded Sequence Length ---
    # We check if seq_len curriculum is active, otherwise default to BASE
    current_base_seq_len = BASE_SEQ_LEN
    if CURRICULUM_SEQ_LEN:
         rollout_increments = global_step // SEQ_LEN_INCREASE_STEPS
         current_base_seq_len = min(BASE_SEQ_LEN + rollout_increments, MAX_SEQ_LEN)
    
    # Dynamic Loading: Load max of (Base, AR+1)
    # This ensures we always have 1 context frame + ar_len frames
    seq_len = max(current_base_seq_len, ar_len + 1)
    seq_len = min(seq_len, MAX_SEQ_LEN)
    
    # Cap AR len if it exceeds available sequence
    ar_len = min(ar_len, seq_len - 1)
    if ar_len < 0: ar_len = 0

    print(f"🧩 Curriculum: seq_len={seq_len}, ar_len={ar_len} (cutoff at t={seq_len - ar_len})")

    # Rebuild dataset and DataLoader each epoch to avoid stale worker state when seq_len changes
    dataset = GTTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
    if len(dataset) == 0:
        print(f"Warning: No sequences available for seq_len={seq_len}. Skipping epoch {epoch}.")
        continue
    
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False,
    )
    print(f"Dataset loaded: {len(dataset)} sequence windows.")

    for batch_idx, batch in enumerate(loader):
        t0_batch = time.time()
        # Each batch is a tuple of tensors/lists from __getitem__ stacked along batch dim.
        Z_seq, A_seq, Z_target_seq, idxs, vid_idxs, starts = batch
        # Move token sequences and actions to GPU/CPU device.
        Z_seq, A_seq, Z_target_seq = Z_seq.to(DEVICE), A_seq.to(DEVICE), Z_target_seq.to(DEVICE)

        # Z_seq has shape (B, K, H, W):
        #   B = batch size (number of different windows),
        #   K = sequence length (number of time steps),
        #   H, W = spatial dimensions of the VQ-VAE latent grid.
        B, K, H, W = Z_seq.shape
        step_losses = []
        logits_last = None

        # --- LR Scheduler: Warmup ---
        if global_step < WARMUP_STEPS:
            # Linear warmup from MIN_LR to LR
            new_lr = MIN_LR + (global_step / WARMUP_STEPS) * (LR - MIN_LR)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        elif global_step == WARMUP_STEPS:
            # Ensure LR is set to the base value after warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
        # --- End LR Scheduler ---

        # Reset gradients for this batch.
        optimizer.zero_grad()
        
        # Optimization: Pre-compute embeddings and FiLM parameters for the whole sequence
        # This avoids repeating these operations inside the sequential loop
        X_seq = model.compute_embeddings(Z_seq) # (B, K, C, H, W)
        Gammas_seq, Betas_seq = model.compute_film(A_seq)    # Stacked tensors (L, B, K, C, 1, 1)

        # Initialize the hidden state for the ConvGRU layers for this batch.
        h_state = model.init_state(B, device=DEVICE)
        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            # Iterate over time steps 0..K-1 and unroll the world model.
            # Define cutoff step: everything >= this step uses model prediction
            ar_cutoff = K - ar_len
            
            for t in range(K):
                # Deterministic AR Logic:
                # If t < ar_cutoff: Teacher Forcing (GT)
                # If t >= ar_cutoff: Autoregressive (Model Pred)
                # Note: t=0 is always GT because loop starts at 0 and ar_len <= K-1, so ar_cutoff >= 1
                use_pred = (t >= ar_cutoff)

                if use_pred:
                    # Use predicted token indices from previous step.
                    # We take argmax to get hard indices. This acts as a "detach" for the token choice,
                    # but gradients still flow through h_state.
                    z_in = logits_last.argmax(dim=1) # (B, H, W)
                    x_in = None # Signal model.step to embed z_in
                else:
                    # Use Ground Truth (Teacher Forcing)
                    # We use the pre-computed embeddings for efficiency.
                    z_in = torch.tensor(0, device=DEVICE) # Dummy, ignored if x_in is provided
                    x_in = X_seq[:, t]

                # FiLM params always come from GT actions (actions are exogenous/observed)
                g_t = Gammas_seq[:, :, t]
                b_t = Betas_seq[:, :, t]
                
                # Pass inputs to step
                if USE_CHECKPOINTING:
                    # checkpoint requires tensors. None is valid for x_in if it's not used for grad?
                    # If x_in is None, we must pass it as None.
                    # z_in is a tensor (indices).
                    logits_t, h_state = checkpoint(
                        model.step, 
                        z_in, 
                        torch.tensor(0, device=DEVICE), # a_t is unused as we pass g_t/b_t
                        h_state, 
                        x_in, 
                        g_t, 
                        b_t, 
                        use_reentrant=False
                    )
                else:
                    logits_t, h_state = model.step(z_in, None, h_state, x_t=x_in, gammas_t=g_t, betas_t=b_t)
                
                logits_last = logits_t
                # logits_t is (B, codebook_size, H, W). We flatten batch and spatial dims
                # so that each position predicts one token class.
                logits_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, logits_t.size(1))
                # Z_target_seq[:, t] is (B, H, W), we flatten it to match logits_flat rows.
                target_flat = Z_target_seq[:, t].reshape(-1)
                loss_t = F.cross_entropy(logits_flat, target_flat)
                step_losses.append(loss_t)

            # Final loss is the mean over all time-step losses in the unroll.
            loss = torch.stack(step_losses).mean()

        # Backpropagate the mixed-precision loss using GradScaler for numerical stability.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Clip gradients to avoid exploding gradients at long horizons.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if (loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD) and REPORT_SPIKES:
            print("\n Large spike --------------")
            print(f"  global_step={global_step}")
            print(f"  loss={loss.item():.4f}, grad_norm={float(grad_norm):.2f}")

            # Additional diagnostics for spike batches
            loss_first = step_losses[0].item()
            loss_last = step_losses[-1].item()
            ppl = math.exp(loss.item())
            ppl_first = math.exp(loss_first)
            ppl_last = math.exp(loss_last)

            with torch.no_grad():
                last_logits = logits_last
                last_logits_f32 = last_logits.float()
                last_target = Z_target_seq[:, -1]
                pred_last = last_logits.argmax(dim=1)
                acc_last = (pred_last == last_target).float().mean().item()
                p_last = torch.softmax(last_logits_f32, dim=1)
                entropy_last = (-p_last * torch.log(torch.clamp(p_last, min=1e-9))).sum(dim=1).mean().item()
                conf_last = p_last.max(dim=1).values.mean().item()

            print(f"  loss_first={loss_first:.4f}, loss_last={loss_last:.4f}")
            print(f"  ppl={ppl:.2f}, ppl_first={ppl_first:.2f}, ppl_last={ppl_last:.2f}")
            print(f"  acc_last={acc_last:.4f}, entropy_last={entropy_last:.4f}, conf_last={conf_last:.4f}")
            print(f"  seq_len={seq_len}")
            print("  Offending batch samples:")
            for vid, start in zip(vid_idxs.tolist(), starts.tolist()):
                print(f"    - video {vid}, start index {start}")

        total_loss += loss.item()
        global_step += 1
        dt_batch = time.time() - t0_batch
        throughput = BATCH_SIZE / dt_batch

        # Emergency save if it's been a long time
        if (time.time() - last_emergency_save_time) > EMERGENCY_SAVE_INTERVAL_HRS * 3600:
            save_checkpoint(epoch, global_step, model, optimizer, is_emergency=True)
            last_emergency_save_time = time.time()

        if global_step % LOG_STEPS == 0:
            # Perplexity estimates (in nats -> exp(loss))
            loss_first = step_losses[0].item()
            loss_last = step_losses[-1].item()
            ppl = math.exp(loss.item())
            ppl_first = math.exp(loss_first)
            ppl_last = math.exp(loss_last)

            # Last-step token accuracy and entropy for quick health signal
            with torch.no_grad():
                last_logits = logits_last
                last_target = Z_target_seq[:, -1]
                pred_last = last_logits.argmax(dim=1)
                acc_last = (pred_last == last_target).float().mean().item()
                p_last = torch.softmax(last_logits, dim=1)
                entropy_last = (-p_last * torch.log(torch.clamp(p_last, min=1e-9))).sum(dim=1).mean().item()
                conf_last = p_last.max(dim=1).values.mean().item()

            lr = optimizer.param_groups[0].get("lr", 0.0)
            if wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/ppl": ppl,
                    "train/loss_first": loss_first,
                    "train/loss_last": loss_last,
                    "train/ppl_first": ppl_first,
                    "train/ppl_last": ppl_last,
                    "train/acc_last": acc_last,
                    "train/entropy_last": entropy_last,
                    "train/conf_last": conf_last,
                    "train/grad_norm": float(grad_norm),
                    "train/throughput": throughput,
                    "seq_len": seq_len,
                    "ar_len": ar_len,
                    "lr": lr,
                }, step=global_step)

    epoch_loss = total_loss / len(loader)
    print(f"Epoch {epoch}: mean loss {epoch_loss:.4f}")
    if wandb:
        wandb.log({
            "epoch": epoch,
            "mean_loss": epoch_loss,
            "mean_ppl": math.exp(epoch_loss),
        })

    # Save checkpoint to Kaggle working directory
    save_checkpoint(epoch, global_step, model, optimizer)

if wandb:
    wandb.finish()
print("✅ Training complete.")

