#!/usr/bin/env python3
"""
Checklist:
- Update output model name
- Update run name
- update vqvae checkpoint
- update world model checkpoint
"""
import os, time, json, math, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
import wandb

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/kaggle/input/minerl-64x64-vqvae-latents-wasd-pitch-yaw"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
BATCH_SIZE = 4
BASE_SEQ_LEN = 15      # initial context
MAX_SEQ_LEN = 40       # maximum unroll window
EPOCHS = 50
LR = 3e-4
CURRICULUM_UNROLL = True  # gradually increase sequence length
SPIKE_LOSS = 7
SPIKE_GRAD = 400

PROJECT = "project-ochre"
RUN_NAME = "v3.0-epoch0"
MODEL_OUT_PREFIX = "ochre-"
resume_path = ""  


wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
           config=dict(batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS))

# ---------- DATASET ----------
class MineRLTokenDataset(Dataset):
    """Loads pre-tokenized MineRL sequences for autoregressive world model training."""
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
        data = np.load(path)  # loads arrays like {"tokens": ..., "actions": ...}
        tokens = data["tokens"]   # shape (T, H, W) of discrete VQ-VAE indices over time
        actions = data["actions"] # shape (T, A) of continuous action vectors per frame

        # Take a window of length seq_len from this video.
        # Z_seq are the input tokens at times [t, ..., t+seq_len-1]
        # Z_target_seq are the next-step tokens [t+1, ..., t+seq_len]
        Z_seq = tokens[start:start + self.seq_len]
        A_seq = actions[start:start + self.seq_len]
        Z_target_seq = tokens[start + 1:start + self.seq_len + 1]

        return (
            torch.tensor(Z_seq, dtype=torch.long),        # (K, H, W)
            torch.tensor(A_seq, dtype=torch.float32),     # (K, A)
            torch.tensor(Z_target_seq, dtype=torch.long), # (K, H, W)
            idx,                  
            vid_idx,             
            start                
        )

# ---------- MODEL ----------
model = WorldModelConvFiLM().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))
global_step = 0
start_epoch = 1
if os.path.exists(resume_path):
    print(f" Resuming training from {resume_path}")
    checkpoint = torch.load(resume_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)
    print(f"Resumed at epoch {start_epoch}, global step {global_step}")
else:
    print(" No checkpoint found — starting fresh.")

# ---------- TRAINING LOOP ----------
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    # Curriculum: gradually increase context length
    seq_len = BASE_SEQ_LEN
    if CURRICULUM_UNROLL:
        seq_len = min(BASE_SEQ_LEN + (epoch // 5), MAX_SEQ_LEN)
    print(f"🧩 Curriculum unrolling active → seq_len = {seq_len}")

    # Rebuild dataset and DataLoader each epoch to avoid stale worker state when seq_len changes
    dataset = MineRLTokenDataset(MANIFEST_PATH, DATA_DIR, seq_len=seq_len)
    if len(dataset) == 0:
        print(f"Warning: No sequences available for seq_len={seq_len}. Skipping epoch {epoch}.")
        continue
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False,
    )
    print(f"Dataset loaded: {len(dataset)} sequence windows.")

    for batch_idx, batch in enumerate(loader):
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

        # Reset gradients for this batch.
        optimizer.zero_grad()
        # Initialize the hidden state for the ConvGRU layers for this batch.
        h_list = model.init_state(B, device=DEVICE)
        with autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            # Iterate over time steps 0..K-1 and unroll the world model.
            for t in range(K):
                # Z_seq[:, t] has shape (B, H, W) of token indices at time t.
                # A_seq[:, t] has shape (B, A) of actions at time t.
                logits_t, h_list = model.step(Z_seq[:, t], A_seq[:, t], h_list)
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

        if loss.item() > SPIKE_LOSS or float(grad_norm) > SPIKE_GRAD:
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
                last_target = Z_target_seq[:, -1]
                pred_last = last_logits.argmax(dim=1)
                acc_last = (pred_last == last_target).float().mean().item()
                p_last = torch.softmax(last_logits, dim=1)
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

        if global_step % 50 == 0:
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
                "seq_len": seq_len,
                "lr": lr,
            }, step=global_step)

    epoch_loss = total_loss / len(loader)
    print(f"Epoch {epoch}: mean loss {epoch_loss:.4f}")
    wandb.log({
        "epoch": epoch,
        "mean_loss": epoch_loss,
        "mean_ppl": math.exp(epoch_loss),
    })

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"/kaggle/working/{MODEL_OUT_PREFIX}_epoch_{epoch}.pt")

wandb.finish()
print("✅ Training complete.")
