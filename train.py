

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
EPOCHS      = 20
LR          = 1e-4
DATA_DIR    = "/kaggle/input/minerl-navigate-spliced-120k/preprocessed"

PROJECT     = "project-ochre"
RUN_NAME    = "run6-startFromEpoch13.5"
PREV_HARDCODED_EPOCH = 0

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
dataset = MineRLSequenceDataset(DATA_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=2, pin_memory=True)
print(f"Dataset size: {len(dataset)} sequences")

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
latest_ckpt = "/kaggle/input/project-ochre/pytorch/default/6/temporal_transformer_epoch13_half.pt"
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

    for frames, actions in loader:
        frames  = frames.to(DEVICE, dtype=torch.float32)
        actions = actions.to(DEVICE, dtype=torch.float32)

        # vae is frozen so dont need gradients
        with torch.no_grad():
            latents = vae.encode(frames[:, :-1].reshape(-1, 3, 256, 256)).latent_dist.sample()
            latents = latents.view(frames.size(0), SEQ_LEN-1, LATENT_C, LATENT_H, LATENT_W)
            target  = vae.encode(frames[:, -1]).latent_dist.sample()

        pred = model(latents, actions)
        loss = criterion(pred, target)

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