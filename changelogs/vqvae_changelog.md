# vq-vae

## Current training version

### v1.0

- Dataset: trained on MineRL frames, first 20 frames skipped, skipping every other frame
- 16x16 latent designed for 64x64 images
- Encoder: 4 convolutional layers with ReLU activations for 64x64 images and 16x16 latent 
- Decoder: mirrored architecture of the encoder with transposed convolutions
- Quantizer: EMA-based vector quantization with exponential moving average updates.
- Codebook size: 512 embeddings.
- Embedding dimension: 64.
- Training Configurations:
- Loss: combination of reconstruction loss (mean squared error) and commitment loss with a weight of 0.25.
- Automatic Mixed Precision (AMP) enabled for faster training and reduced memory usage.

Results:
- Good metric improvements + epoch outputs until final epoch
- Final epoch experienced complete collapse - black outputs
- Cause: loss hit infinity 

### v1.0.1
- Gradient clipping set to a maximum norm of 1.0 to stabilize training.

Results after 10 epochs ~30min:
- Rapid descent across all loss metrics + perplexity in epoch 1-2
- Gradual minor loss + perplexity metric increases in epoch 3-10

Results after 10 more epochs, LPIPS enabled
- No visually discernible difference. Images still have a pixelated blur, likely due to MSE. Future training should encourage sharpness in pixels.

### v2.0

Dataset
- Changed to 25.3k of randomly sampled images from GameFactory, guaranteed 5 frame gaps.

Architecture:
- Input resolution: 64x36 (16:9), downsampled from GameFactory.
- Latent grid: 16x9 (144 tokens per frame) via two stride-2 conv downsamples.
- Encoder/Decoder: added residual blocks
- Quantizer: embedding_dim=384, codebook_size=4096.

Results:
- Convergence within 1 epoch, solid outputs. Though was tested on train set.
- Nearly matches input images, blocks are just missing the pixelated minecraft look.

**v2.0.1**
- Increased input resolution to: 128 x 72, old input image was too constraining on model.
- Extended encoder/decoder with third sampling stage to handle new res.

Epoch 20 Results:
- General structure present
- Poor sharpness diff between predicted and gt
- Some low-detail images (e.g. sky) show "vqvae" patterns rather than smooth colours

**v2.0.2**
- Enabled LPIPS
- Increased batch to 192

Epoch 40 Results (cont. from v2.0.1):
- Only some scenes show general pattern captured, e.g. desert scene gets a bunch of weird pink coloured blobs
- Grid pattern is strongly imprinted onto images, especially sky - turns into grey grid instead of smooth blue gradient

**v2.0.3**
- Reduced codebook size to 1024
- Added in train set output samples to determine generalizability 

Epoch 30 Results:
- Strange patches in images, general scene patterns not captured in some biomes (e.g. water)

**v2.0.4**
- Applied 0.3 * LPIPS 
- Added separate wandb logging for MSE vs LPIPS loss

Epoch 30 Results:
- Realized via wandb LPIPS was always 0. 

**v2.0.5**
- Added missing LPIPS installation in kaggle

Epoch 60 (cont. from 2.0.4-epoch30) Results:
- Better reconstruction in some images. LPIPS seems to bias model towards memorizing local patterns, to the point of "screen burn in" where images of sky have small patches of grass appearing.
- Any fine texture is represented as grid noise

**v2.1.0**
- Reduced 3 -> 2 down/upsampling layers to now support 32x18 latents
- Added LPIPS schedule, +0.05 per 5 epochs
- Increased batch size 192 -> 256

Results:
- Just a grey image