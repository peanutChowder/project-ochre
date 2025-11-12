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
import wandb
from model_convGru import WorldModelConvFiLM

# ---------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/kaggle/input/minerl-64x64-vqvae-latents-wasd-pitch-yaw"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
BATCH_SIZE = 4
BASE_SEQ_LEN = 4      # initial context
MAX_SEQ_LEN = 8       # maximum unroll window
EPOCHS = 20
LR = 3e-4
CURRICULUM_UNROLL = True  # gradually increase sequence length

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
        data = np.load(path)
        tokens = data["tokens"]      # (T,16,16)
        actions = data["actions"]    # (T,4)
        i = start
        Z_seq = tokens[i:i+self.seq_len]
        A_seq = actions[i:i+self.seq_len]
        Z_target_seq = tokens[i+1:i+self.seq_len+1]
        return (
            torch.tensor(Z_seq, dtype=torch.long),
            torch.tensor(A_seq, dtype=torch.float32),
            torch.tensor(Z_target_seq, dtype=torch.long),
        )

# ---------- MODEL ----------
model = WorldModelConvFiLM().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
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
        Z_seq, A_seq, Z_target_seq = [x.to(DEVICE) for x in batch]

        # forward pass (multi-step unrolling)
        logits_list = model(Z_seq, A_seq, return_all=True)
        step_losses = []
        for t, logits in enumerate(logits_list):
            # Flatten spatial + batch to N, keep class dimension as C
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            target_flat = Z_target_seq[:, t].reshape(-1)
            loss_t = F.cross_entropy(logits_flat, target_flat)
            step_losses.append(loss_t)

        loss = torch.stack(step_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
                last_logits = logits_list[-1]
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
