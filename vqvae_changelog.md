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

Results:
- Rapid descent across all loss metrics + perplexity in epoch 1-2
- Gradual minor loss + perplexity metric increases in epoch 3-10

