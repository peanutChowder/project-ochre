import os
import glob
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
DATA_DIR = '/kaggle/input/dataset'  # Replace with actual dataset path in Kaggle
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
EMBEDDING_DIM = 256
CODEBOOK_SIZE = 2048
BETA = 0.25
EMA_DECAY = 0.99
IMAGE_SIZE = 64
LOG_EVERY = 50
SAVE_DIR = '/kaggle/working'
NUM_WORKERS = 4
USE_LPIPS = False
USE_WANDB = True


class Encoder(nn.Module):
    """
    Encoder for 64x64 images producing 16x16 latent feature maps.
    """
    def __init__(self, in_channels=3, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Output spatial size: 64 -> 32 -> 16 (downsample twice by stride=2)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1)  # 64->32
        self.conv2 = nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1)  # 32->16
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """
    Decoder for 16x16 latent feature maps producing 64x64 images.
    """
    def __init__(self, embedding_dim=256, out_channels=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Upsample twice by stride=2 transpose conv
        self.conv_trans1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)  # 16->32
        self.conv_trans2 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)  # 32->64
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
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

        embedding = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', embedding.clone())

    def forward(self, inputs):
        """
        inputs: (B, D, H, W)
        returns: quantized (B, D, H, W), vq_loss, perplexity, encodings
        """
        # Flatten input to (B*H*W, D)
        inputs_perm = inputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat_input = inputs_perm.view(-1, self.embedding_dim)  # (B*H*W, D)

        # Compute distances to embeddings
        emb = self.embedding

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(emb ** 2, dim=0)
            - 2 * torch.matmul(flat_input, emb)
        )

        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)  # (B*H*W, num_embeddings)

        # Quantize
        quantized = torch.matmul(encodings, emb.t())  # (B*H*W, D)
        quantized = quantized.view(inputs_perm.shape)  # (B, H, W, D)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        if self.training:
            # EMA cluster size update
            cluster_size = torch.sum(encodings, dim=0)  # (num_embeddings,)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # EMA embedding average update
            embed_sum = torch.matmul(flat_input.t(), encodings)  # (D, num_embeddings)
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace smoothing of cluster size
            n = torch.sum(self.cluster_size.data)
            cluster_size = (
                (self.cluster_size.data + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            # Normalize embedding average with smoothed cluster size
            self.embedding.data.copy_(self.embedding_avg.data / cluster_size.unsqueeze(0))

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
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


class FlatFolderDataset(Dataset):
    """
    Dataset that recursively loads all PNG/JPG images from a folder (flat folder).
    """
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.image_size = image_size
        self.paths = []
        for ext in ('**/*.png', '**/*.jpg', '**/*.jpeg'):
            self.paths.extend(glob.glob(os.path.join(root_dir, ext), recursive=True))
        self.paths = sorted(self.paths)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
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
    dataset = FlatFolderDataset(DATA_DIR, IMAGE_SIZE)
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
        wandb.init(project="vqvae", config={ #type: ignore
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "embedding_dim": EMBEDDING_DIM,
            "codebook_size": CODEBOOK_SIZE,
            "beta": BETA,
            "ema_decay": EMA_DECAY,
            "image_size": IMAGE_SIZE,
            "log_every": LOG_EVERY,
            "num_workers": NUM_WORKERS,
            "use_lpips": USE_LPIPS,
        })
        wandb.watch(model, log="all") #type: ignore

    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    for epoch in range(1, EPOCHS + 1):
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
            with torch.autocast('cuda', enabled=(device.type == 'cuda')):
                x_recon, vq_loss, perplexity, encoding_indices = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                if perceptual_loss_fn is not None:
                    lpips_loss = perceptual_loss_fn(x_recon, x).mean()
                    recon_loss = recon_loss + lpips_loss
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
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

            if step % LOG_EVERY == 0:
                avg_recon = running_recon_loss / steps
                avg_vq = running_vq_loss / steps
                avg_total = running_total_loss / steps
                avg_perplexity = running_perplexity / steps
                code_usage_frac = len(used_codes) / CODEBOOK_SIZE
                dead_codes = CODEBOOK_SIZE - len(used_codes)

                # EMA codebook variance: variance of embedding magnitudes
                embedding_norms = model.vq_vae.embedding.norm(dim=0)
                ema_codebook_variance = embedding_norms.var().item()

                print(f"Epoch {epoch} Step {step} | Recon Loss: {avg_recon:.6f} | VQ Loss: {avg_vq:.6f} | "
                      f"Total Loss: {avg_total:.6f} | Perplexity: {avg_perplexity:.4f} | "
                      f"Code Usage: {code_usage_frac:.4f} | Dead Codes: {dead_codes} | EMA Codebook Var: {ema_codebook_variance:.6f}")

                if use_wandb:
                    wandb.log({
                        "step": (epoch - 1) * len(dataloader) + step,
                        "recon_loss": avg_recon,
                        "vq_loss": avg_vq,
                        "total_loss": avg_total,
                        "perplexity": avg_perplexity,
                        "code_usage_frac": code_usage_frac,
                        "dead_codes": dead_codes,
                        "ema_codebook_variance": ema_codebook_variance,
                    })

        # Epoch end logging
        avg_recon = running_recon_loss / steps
        avg_vq = running_vq_loss / steps
        avg_total = running_total_loss / steps
        avg_perplexity = running_perplexity / steps
        code_usage_frac = len(used_codes) / CODEBOOK_SIZE
        dead_codes = CODEBOOK_SIZE - len(used_codes)
        embedding_norms = model.vq_vae.embedding.norm(dim=0)
        ema_codebook_variance = embedding_norms.var().item()

        print(f"Epoch {epoch} completed. Avg Recon Loss: {avg_recon:.6f} | Avg VQ Loss: {avg_vq:.6f} | "
              f"Avg Total Loss: {avg_total:.6f} | Avg Perplexity: {avg_perplexity:.4f} | "
              f"Code Usage Fraction: {code_usage_frac:.4f} | Dead Codes: {dead_codes} | EMA Codebook Var: {ema_codebook_variance:.6f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "avg_recon_loss": avg_recon,
                "avg_vq_loss": avg_vq,
                "avg_total_loss": avg_total,
                "avg_perplexity": avg_perplexity,
                "avg_code_usage_frac": code_usage_frac,
                "dead_codes": dead_codes,
                "ema_codebook_variance": ema_codebook_variance,
            })

        # Save checkpoint
        os.makedirs(SAVE_DIR, exist_ok=True)
        latest_path = os.path.join(SAVE_DIR, 'vqvae.pt')
        epoch_path = os.path.join(SAVE_DIR, f'vqvae_epoch_{epoch}.pt')
        torch.save(model.state_dict(), latest_path)
        torch.save(model.state_dict(), epoch_path)

        # Save reconstructions
        model.eval()
        with torch.no_grad():
            x = next(iter(dataloader))
            x = x.to(device)
            x_recon, _, _, _ = model(x)
            save_reconstructions(x.cpu(), x_recon.cpu(), epoch, SAVE_DIR)


if __name__ == "__main__":
    main()
