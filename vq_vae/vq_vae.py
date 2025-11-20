"""
Checklist:
- Update output model name
- Update run name
- Update checkpoint path
"""
import os
import glob
import time
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torchvision import transforms, utils

try:
    import lpips
except ImportError:
    lpips = None

try:
    import wandb
except ImportError:
    wandb = None

# Configuration constants for Kaggle environment
DATA_DIR = '/kaggle/input/dataset'
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
EMBEDDING_DIM = 384
CODEBOOK_SIZE = 4096
BETA = 0.25
EMA_DECAY = 0.99
# Target training resolution, matching 16:9 aspect ratio (e.g., 640x360 -> 128x72).
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 128
LOG_EVERY = 50
SAVE_DIR = '/kaggle/working'
NUM_WORKERS = 4
USE_LPIPS = False
USE_WANDB = True
RUN_NAME = "v2.0-epoch0"
LOAD_FROM_SAVE = ""
EMERGENCY_SAVE_HOURS = 11.8


class ResidualBlock(nn.Module):
    """
    Simple residual block: Conv -> ReLU -> Conv + skip connection.
    Operates at fixed channel count and preserves spatial resolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """
    Encoder for 128x72 images producing 16x9 latent feature maps.
    """
    def __init__(self, in_channels=3, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Output spatial size (for 128x72 input): 128x72 -> 64x36 -> 32x18 -> 16x9
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1)   # 128x72 -> 64x36
        self.res1 = ResidualBlock(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)          # 64x36 -> 32x18
        self.res2 = ResidualBlock(128)
        self.conv3 = nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1) # 32x18 -> 16x9
        self.res3 = ResidualBlock(embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.relu(self.conv2(x))
        x = self.res2(x)
        x = self.relu(self.conv3(x))
        x = self.res3(x)
        return x


class Decoder(nn.Module):
    """
    Decoder for 16x9 latent feature maps producing 128x72 images.
    """
    def __init__(self, embedding_dim=256, out_channels=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Upsample three times by stride=2 transpose conv with residual refinement at each scale.
        self.res_latent = ResidualBlock(embedding_dim)
        self.conv_trans1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)  # 16x9 -> 32x18
        self.res_mid1 = ResidualBlock(128)
        self.conv_trans2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)           # 32x18 -> 64x36
        self.res_mid2 = ResidualBlock(128)
        self.conv_trans3 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)  # 64x36 -> 128x72
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.res_latent(x)
        x = self.relu(self.conv_trans1(x))
        x = self.res_mid1(x)
        x = self.relu(self.conv_trans2(x))
        x = self.res_mid2(x)
        x = self.conv_trans3(x)
        x = torch.sigmoid(x)
        return x


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average updates.
    """
    def __init__(self, embedding_dim=256, num_embeddings=2048, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.randn(embedding_dim, num_embeddings, dtype=torch.float32)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings, dtype=torch.float32))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, inputs):
        """
        inputs: (B, D, H, W)
        returns: quantized (B, D, H, W), vq_loss, perplexity, encodings
        """
        inputs_f32 = inputs.float()
        inputs_perm = inputs_f32.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat_input = inputs_perm.view(-1, self.embedding_dim)  # (B*H*W, D)
        flat_input = flat_input.clamp(-10.0, 10.0)
        emb = self.embedding

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(emb ** 2, dim=0, keepdim=True)
            - 2.0 * flat_input.matmul(emb)
        )

        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)  # (B*H*W, num_embeddings)

        quantized = encodings.float().matmul(emb.t())
        quantized = quantized.view(inputs_perm.shape).permute(0, 3, 1, 2).contiguous()
        quantized = quantized.to(inputs.dtype)

        if self.training:
            with torch.no_grad():
                enc_f = encodings.float()
                # EMA cluster sizes
                batch_cluster = enc_f.sum(0)
                self.cluster_size.mul_(self.decay).add_(batch_cluster, alpha=1.0 - self.decay)
                # EMA embedding sums
                embed_sum = flat_input.t().matmul(enc_f)  # (D, K)
                self.embedding_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)
                # Normalize with Laplace smoothing
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                new_emb = self.embedding_avg / cluster_size.unsqueeze(0)
                self.embedding.copy_(new_emb)
                # Reseed dead codes from current batch encoder outputs
                dead_mask = (batch_cluster == 0)
                if dead_mask.any():
                    dead_indices = torch.nonzero(dead_mask, as_tuple=False).squeeze(1)
                    num_dead = dead_indices.numel()
                    if flat_input.numel() > 0:
                        rand_ids = torch.randint(0, flat_input.shape[0], (num_dead,), device=flat_input.device)
                        self.embedding[:, dead_indices] = flat_input[rand_ids].t()

                self.embedding.copy_(torch.nan_to_num(self.embedding, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-3.0, 3.0))
                self.embedding_avg.copy_(torch.nan_to_num(self.embedding_avg, nan=0.0))
                self.cluster_size.copy_(torch.nan_to_num(self.cluster_size, nan=0.0))

                        

        inputs_4loss = inputs_f32
        e_latent_loss = F.mse_loss(quantized.detach().float(), inputs_4loss)
        q_latent_loss = F.mse_loss(quantized.float(), inputs_4loss.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized.to(inputs.dtype) - inputs).detach()

        avg_probs = torch.mean(encodings.float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, embedding_dim=256, num_embeddings=2048, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(3, embedding_dim)
        self.pre_vq_conv = nn.Identity()  # no extra conv before quantization
        self.vq_vae = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder = Decoder(embedding_dim, 3)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_vq_conv(z_e)
        quantized, vq_loss, perplexity, encoding_indices = self.vq_vae(z_e)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity, encoding_indices
    
    def decode_code(self, code_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete code indices back into an RGB image.
        code_indices: (B, H, W) or (H, W)
        Returns (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH) in [0,1] when trained at that size.
        """
        if code_indices.dim() == 2:
            code_indices = code_indices.unsqueeze(0)
        elif code_indices.dim() == 3 and code_indices.size(0) == 1:
            # ensure consistent dtype and batch dimension
            code_indices = code_indices.clone()
        else:
            raise ValueError(f"Unexpected shape for code_indices: {tuple(code_indices.shape)}")

        code_indices = code_indices.long()  # ensure integer indices
        emb = self.vq_vae.embedding  # (D, K)
        # F.embedding expects (...,) and adds trailing dimension D
        z_q = F.embedding(code_indices, emb.t())  # (B, H, W, D)
        if z_q.dim() == 3:  # safety fallback if batch dim squeezed
            z_q = z_q.unsqueeze(0)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        return self.decoder(z_q)


class FlatFolderDataset(Dataset):
    """
    Dataset that recursively loads all PNG/JPG images from a folder (flat folder).
    """
    def __init__(self, root_dir, height: int, width: int):
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.paths = []
        for ext in ('**/*.png', '**/*.jpg', '**/*.jpeg'):
            self.paths.extend(glob.glob(os.path.join(root_dir, ext), recursive=True))
        self.paths = sorted(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img


def save_reconstructions(x, x_recon, epoch, save_dir):
    """
    Save an 8x2 grid of original and reconstructed images side by side.
    """
    n = min(8, x.size(0))
    comparison = torch.cat([x[:n], x_recon[:n]])
    grid = utils.make_grid(comparison, nrow=n)
    utils.save_image(grid, os.path.join(save_dir, f'recon_epoch_{epoch}.png'))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup dataset and dataloader
    dataset = FlatFolderDataset(DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = VQVAE(
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=CODEBOOK_SIZE,
        commitment_cost=BETA,
        decay=EMA_DECAY,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 1
    global_step = 0
    if (LOAD_FROM_SAVE):
        print(f"Resuming from checkpoint '{LOAD_FROM_SAVE}'")
        checkpoint = torch.load(LOAD_FROM_SAVE, map_location=device)
        try:
            model.encoder.load_state_dict(checkpoint["encoder"])
            model.vq_vae.load_state_dict(checkpoint["quantizer"])
            model.decoder.load_state_dict(checkpoint["decoder"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            saved_lr = checkpoint["config"]["lr"]
            global_step = checkpoint.get("global_step", 0)
            print(f"Retrieved learning rate '{saved_lr}'")
            print(f"Successfully loaded checkpoint, beginning training from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print("No checkpoint found - starting from epoch 0")
            

    # LPIPS loss
    if USE_LPIPS and lpips is not None:
        perceptual_loss_fn = lpips.LPIPS(net='alex').to(device)
        print("LPIPS enabled.")
    else:
        perceptual_loss_fn = None
        if USE_LPIPS:
            print("Warning: lpips package not found, disabling LPIPS loss.")

    # W&B
    use_wandb = USE_WANDB and (wandb is not None)
    if use_wandb:
        wandb.init(project="vqvae", name=RUN_NAME, config={ #type: ignore
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "embedding_dim": EMBEDDING_DIM,
            "codebook_size": CODEBOOK_SIZE,
            "beta": BETA,
            "ema_decay": EMA_DECAY,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "log_every": LOG_EVERY,
            "num_workers": NUM_WORKERS,
            "use_lpips": USE_LPIPS,
        })
        wandb.watch(model, log="all") #type: ignore

    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    start_time = time.time()
    emergency_saved = False

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_recon_loss = 0.0
        running_vq_loss = 0.0
        running_total_loss = 0.0
        running_perplexity = 0.0
        total_codes = 0
        used_codes = set()
        steps = 0

        for step, x in enumerate(dataloader, 1):
            x = x.to(device)
            optimizer.zero_grad()

            # Run quantizer explicitly in float32 (outside autocast) to avoid fp16 overflow.
            with torch.autocast('cuda', enabled=(device.type == 'cuda')):
                z_e = model.encoder(x)
                z_e = model.pre_vq_conv(z_e)
            quantized, vq_loss, perplexity, encoding_indices = model.vq_vae(z_e)
            with torch.autocast('cuda', enabled=(device.type == 'cuda')):
                x_recon = model.decoder(quantized)
                recon_loss = F.mse_loss(x_recon, x)
                if perceptual_loss_fn is not None:
                    lpips_loss = perceptual_loss_fn(x_recon, x).mean()
                    recon_loss = recon_loss + lpips_loss
                loss = recon_loss + vq_loss

            if not torch.isfinite(loss):
                print("⚠️  Skipping batch due to non-finite loss.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_recon_loss += recon_loss.item()
            running_vq_loss += vq_loss.item()
            running_total_loss += loss.item()
            running_perplexity += perplexity.item()

            # Track code usage
            used_codes.update(encoding_indices.cpu().numpy())
            total_codes += encoding_indices.numel()

            steps += 1
            global_step += 1

            if step % LOG_EVERY == 0:
                if steps == 0:
                    print(f"Epoch {epoch}: no valid steps (all skipped). Skipping averages.")
                    continue
                avg_recon = running_recon_loss / steps
                avg_vq = running_vq_loss / steps
                avg_total = running_total_loss / steps
                avg_perplexity = running_perplexity / steps
                code_usage_frac = len(used_codes) / CODEBOOK_SIZE
                dead_codes = CODEBOOK_SIZE - len(used_codes)

                emb_safe = torch.nan_to_num(model.vq_vae.embedding, nan=0.0)
                embedding_norms = emb_safe.norm(dim=0)
                ema_codebook_variance = torch.nan_to_num(embedding_norms.var(), nan=0.0).item()

                print(f"Epoch {epoch} Step {step} | Recon Loss: {avg_recon:.6f} | VQ Loss: {avg_vq:.6f} | "
                      f"Total Loss: {avg_total:.6f} | Perplexity: {avg_perplexity:.4f} | "
                      f"Code Usage: {code_usage_frac:.4f} | Dead Codes: {dead_codes} | EMA Codebook Var: {ema_codebook_variance:.6f}")

                if use_wandb:
                    wandb.log({
                        "train/recon_loss": avg_recon,
                        "train/vq_loss": avg_vq,
                        "train/total_loss": avg_total,
                        "train/perplexity": avg_perplexity,
                        "train/code_usage_frac": code_usage_frac,
                        "train/dead_codes": dead_codes,
                        "train/ema_codebook_variance": ema_codebook_variance,
                    }, step=global_step)

            # Emergency save at 11.8 hrs
            if not emergency_saved:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= EMERGENCY_SAVE_HOURS:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    emergency_path = os.path.join(SAVE_DIR, 'vqvae_emergency.pt')
                    checkpoint = {
                        "epoch": epoch,
                        "encoder": model.encoder.state_dict(),
                        "quantizer": model.vq_vae.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": {
                            "embedding_dim": EMBEDDING_DIM,
                            "codebook_size": CODEBOOK_SIZE,
                            "beta": BETA,
                            "ema_decay": EMA_DECAY,
                            "lr": LR,
                        },
                    }
                    torch.save(checkpoint, emergency_path)
                    print(f"⏰ Emergency checkpoint saved to {emergency_path} after {elapsed_hours:.2f} hours.")
                    emergency_saved = True

        if steps == 0:
            print(f"Epoch {epoch}: no valid steps (all skipped). Skipping averages.")
            continue

        # Epoch end logging
        avg_recon = running_recon_loss / steps
        avg_vq = running_vq_loss / steps
        avg_total = running_total_loss / steps
        avg_perplexity = running_perplexity / steps
        code_usage_frac = len(used_codes) / CODEBOOK_SIZE
        dead_codes = CODEBOOK_SIZE - len(used_codes)
        emb_safe = torch.nan_to_num(model.vq_vae.embedding, nan=0.0)
        embedding_norms = emb_safe.norm(dim=0)
        ema_codebook_variance = torch.nan_to_num(embedding_norms.var(), nan=0.0).item()

        print(f"Epoch {epoch} completed. Avg Recon Loss: {avg_recon:.6f} | Avg VQ Loss: {avg_vq:.6f} | "
              f"Avg Total Loss: {avg_total:.6f} | Avg Perplexity: {avg_perplexity:.4f} | "
              f"Code Usage Fraction: {code_usage_frac:.4f} | Dead Codes: {dead_codes} | EMA Codebook Var: {ema_codebook_variance:.6f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "val/avg_recon_loss": avg_recon,
                "val/avg_vq_loss": avg_vq,
                "val/avg_total_loss": avg_total,
                "val/avg_perplexity": avg_perplexity,
                "val/avg_code_usage_frac": code_usage_frac,
                "val/dead_codes": dead_codes,
                "val/ema_codebook_variance": ema_codebook_variance,
            }, step=global_step)

        # Save checkpoint
        os.makedirs(SAVE_DIR, exist_ok=True)
        latest_path = os.path.join(SAVE_DIR, 'vqvae.pt')
        epoch_path = os.path.join(SAVE_DIR, f'vqvae_epoch_{epoch}.pt')
        checkpoint = {
            "epoch": epoch,
            "encoder": model.encoder.state_dict(),
            "quantizer": model.vq_vae.state_dict(),
            "decoder": model.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "config": {
                "embedding_dim": EMBEDDING_DIM,
                "codebook_size": CODEBOOK_SIZE,
                "beta": BETA,
                "ema_decay": EMA_DECAY,
                "lr": LR,
            },
        }
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, epoch_path)

        # Save reconstructions
        model.eval()
        with torch.no_grad():
            x = next(iter(dataloader))
            x = x.to(device)
            x_recon, _, _, _ = model(x)
            save_reconstructions(x.cpu(), x_recon.cpu(), epoch, SAVE_DIR)


if __name__ == "__main__":
    main()
