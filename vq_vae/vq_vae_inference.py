import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
import random
import argparse
from vq_vae import Encoder, Decoder, VectorQuantizerEMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
])

def load_images(data_dir, num_samples):
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    selected_files = random.sample(all_files, min(num_samples, len(all_files)))
    images = []
    for f in selected_files:
        img = Image.open(os.path.join(data_dir, f)).convert("RGB")
        img = transform(img)
        images.append(img)
    return torch.stack(images)

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Inference")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="inference_recons.png")
    parser.add_argument('--num_samples', type=int, default=16)
    args = parser.parse_args()

    encoder = Encoder().to(device)
    quantizer = VectorQuantizerEMA(num_embeddings=512, embedding_dim=64).to(device)
    decoder = Decoder().to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'encoder' in checkpoint and 'decoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
        quantizer.load_state_dict(checkpoint['quantizer'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        # Assume flat state dict (model.state_dict())
        combined_model = nn.Module()
        combined_model.encoder = encoder
        combined_model.quantizer = quantizer
        combined_model.decoder = decoder
        combined_model.load_state_dict(checkpoint, strict=False)

    encoder.eval()
    quantizer.eval()
    decoder.eval()

    imgs = load_images(args.data_dir, args.num_samples).to(device)

    with torch.no_grad():
        with torch.autocast('cuda', enabled=(device.type == 'cuda')):
            z_e = encoder(imgs)
            z_q, q_loss, perplexity, _ = quantizer(z_e)
            recons = decoder(z_q)

            recon_loss = torch.mean((recons - imgs) ** 2).item()
            perplexity_val = perplexity.item()

    # Make grid: originals and reconstructions side by side
    comparison = torch.cat([imgs, recons])
    grid = make_grid(comparison, nrow=args.num_samples)
    save_image(grid, args.output_path)

    print(f"Reconstruction loss: {recon_loss:.6f}")
    print(f"Perplexity: {perplexity_val:.6f}")
    print(f"Saved inference comparison image to {args.output_path}")

if __name__ == "__main__":
    main()
