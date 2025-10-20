#!/usr/bin/env python3
"""
Checklist:
- Update input model path to latest
- Update run name
- update prev hardcoded epoch
"""
import os, time, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
import wandb

# ---------- CONFIG ----------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 4
SEQ_LEN     = 7
LATENT_C    = 4
LATENT_H    = LATENT_W = 32
EPOCHS      = 2
LR          = 1e-4
DATA_DIR    = "/kaggle/input/minerl-navigate-spliced-120k/preprocessedv2.0"

PROJECT     = "project-ochre"
RUN_NAME    = "v2.0-run1-epoch0"
PREV_HARDCODED_EPOCH = 0

LATENT_NOISE = False  # Flag to control latent noise injection

# Safety limit: Kaggle runtime cutoff ~12 hours → stop early
MAX_TRAIN_HOURS = 11.9

# ---------- HELPERS ----------
def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model + optimizer state from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    if "model_state" in ckpt:
        # New format (comprehensive checkpoint)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"✅ Loaded full checkpoint from {checkpoint_path}")
    else:
        # Old format (only model weights)
        model.load_state_dict(ckpt)
        start_epoch = PREV_HARDCODED_EPOCH + 1
        global_step = 0
        print(f"⚠️ Loaded model weights only (no optimizer state): {checkpoint_path}")

    return start_epoch, global_step


# ---------- W&B ----------
wandb.init(project=PROJECT, name=RUN_NAME, resume="allow",
           config=dict(batch_size=BATCH_SIZE, lr=LR, epochs=EPOCHS))

# ---------- Dataset ----------
dataset = MineRLFrameTripletDataset(DATA_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=2, pin_memory=True)
print(f"Dataset size: {len(dataset)} frame triplets")

print("Quick dataset smoke test: f.shape, a.shape, n.shape, n2.shape")
for i, (f, a, n, n2) in enumerate(loader):
    print(f.shape, a.shape, n.shape, n2.shape)
    if i > 5: break
print("-----------------")

# ---------- VAE (frozen) ----------
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to(DEVICE)
vae.eval()

# ---------- Model ----------
model = TemporalTransformer(latent_channels=LATENT_C, H=LATENT_H, W=LATENT_W).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ---------- Resume / Initialization ----------
latest_ckpt = ""
if latest_ckpt:
    start_epoch, global_step = load_checkpoint(model, optimizer, latest_ckpt)
else:
    print("🚀 No existing checkpoint found. Starting fresh.")
    start_epoch, global_step = 1, 0

# ---------- Training Loop ----------
start_time = time.time()
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for frame_t, action_t, frame_next, frame_next2 in loader:
        frame_t    = frame_t.to(DEVICE, dtype=torch.float32)
        action_t   = action_t.to(DEVICE, dtype=torch.float32)
        frame_next = frame_next.to(DEVICE, dtype=torch.float32)
        _ = frame_next2  # Placeholder for future use (noise-injection, scheduled sampling, etc.)

        with torch.no_grad():
            latents_t = vae.encode(frame_t).latent_dist.sample()
            target    = vae.encode(frame_next).latent_dist.sample()
            if LATENT_NOISE:
                noise = torch.randn_like(latents_t) * 0.1
                latents_t = latents_t + noise

        pred = model(latents_t.unsqueeze(1), action_t.unsqueeze(1))  # Add seq dim of 1
        loss = criterion(pred.squeeze(1), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        wandb.log({"train/loss": loss.item(), "train/step": global_step})
        wandb.log({
            "latent/pred_mean": pred.mean().item(),
            "latent/pred_std": pred.std().item(),
            "latent/target_mean": target.mean().item(),
            "latent/target_std": target.std().item(),
        })

        # Safety save before Kaggle timeout
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours > MAX_TRAIN_HOURS:
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, f"/kaggle/working/temporal_transformer_epoch{epoch}_half.pt")
            wandb.log({"safety_checkpoint_saved": epoch})
            print("⚠️ Safety save triggered: stopping before timeout.")
            wandb.finish()
            exit(0)

    # --- End of epoch ---
    epoch_loss = running_loss / len(loader)
    print(f"Epoch {epoch}: mean loss {epoch_loss:.4f}")

    wandb.log({"epoch": epoch, "epoch_loss": epoch_loss})
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, f"/kaggle/working/temporal_transformer_epoch{epoch}_full.pt")

wandb.finish()
print("✅ Training complete.")