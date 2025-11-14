# project-ochre

## Current training version

### v1.1

- 6 frame input + yaw/axis data, single frame prediction.

Results: 
- Trained for ~14 epochs, ~90 hours of A100
- Decent video generation (inference.py) with 6 frame latent input. Some videos perform very well (ocean with islands, snow biome) whereas others tend to collapse (tree biomes turn to brownish-grey blur)
- Poor video generation (inference.py) with 1 frame duplicated x6 latent input. Very quickly collapses to some blurry grey-brown average.
- Horrible live inference (live_inference.py) performance. Completely nonresponsive to movements. Shows some change when fed 6-frame latent initial input, but over time collapses to grey-brown average.

Data
- Each child of preprocessed data corresponds to a random 7-frame subset of all the mp4s

---
### v2.0
- 1 frame input + yaw/axis data, single frame prediction.
- Latent noise injection - gaussian noise in inputs to encourage recovery from previous error drift
- Self-sampling - occasionally using predicted frame 1 as input to predict frame 2 applied after **1.2** epochs
- Removing 1s from start of each mp4 due to sudden teleporting

Data 
- Each child folder of preprocessed data now corresponds to a mp4
- Downsampling FPS - max pooling actions and sampling frames for 8 FPS

**v2.0.2**
- Bug fixes due to import errors

**v2.0.3**
- Added wandb logging for avg loss/10k steps
- Added wandb epoch completion indicator (%)

**v2.0.4**
- Added safety checkpoint saves after final save failed from v2.0.2

**v2.0.5**
- Fixed wandb x-axis steps not counted properly due to batched updates
- Decreased average loss calulation to per 1k steps

**v2.1.0**
- Enabled latent noise injection
- Fixed loss/1k steps divisor 

Results:
- Trained for 1.8 epochs (limited due to training script inefficiency)
- 0.6 epochs / 12hrs
- Live inference maintains minecraft-ness much better
- Live inference eventually collapses to still scene
- Does not appear to respond to user input

---
### v3.0
Dataset:
- Overhauled preprocessing and alignment; do not assume fixed 20 FPS.
- Use metadata.json (duration_ms, duration_steps) + decoded MP4 frames to map steps→frames via linspace.
- Downsample in step space to target FPS; aggregate actions between kept frames.
- Replaced Gaussian VAE with our trained VQ‑VAE (vq_vae/checkpoints) at 64×64.

Preprocessing (preprocess.py):
- VAE encoding during preprocessing saves lots of train time
- Outputs per‑trajectory .npz with:
  - tokens: [K, 16, 16] uint16 VQ‑VAE code indices for kept frames
  - actions: [K−1, 4] float32 = [yaw, pitch, move_x, move_z]
- Action aggregation per window (kept_state[i]→kept_state[i+1]):
  - yaw/pitch: mean of MineRL camera deltas; clipped to ±15° and scaled to [-1,1]
  - move_x = right − left, move_z = forward − back (averaged)
- Emits manifest.json with [{file, length=K}] for boundary‑safe sampling.
- Guarantees len(actions) = len(tokens) − 1 and strict per‑video boundaries.

VQVAE:
- New VQVAE trained from ground up to support 64x64 input/outputs
- Improved train+inference time with lower resolution

Training (train.py + model):
- Autoregressive multi‑step rollout loss with curriculum unroll (BASE_SEQ_LEN->MAX_SEQ_LEN).
- Dataset samples n‑length windows strictly within a single trajectory using manifest boundaries.
- Model conditions a ConvGRU token predictor via FiLM on 4D actions; codebook size matches VQ‑VAE (default 2048).

**Results, 9 epochs**
- Immediately recognizable minecraft world
- Does not respond well to inputs -- seems to shift only on initial delta in movements/camera
- No collapse! Maintains "minecraft-ness".
- Not yet generalized to orientations. Movements + camera result in a few pixels appearing/disappearing, but frame of reference does not change



**Results, 36 epochs**
- Improvement on continued frame changes as movement+camera keys are held down
- Detail improvement - medium objects like trees are beginning to appear
- Still no improvement on orientation - camera movement only changes groups of pixels at a time rather than shifting entire scene.
- Unique scenes depending on sequence of camera+movement produced!
- Suspect improved dataset or autoregressive strategy needed for model to understanding orientation changes
- W (forward) results in the most significant scene changes - suspect due to dominance in dataset
- Still struggles with dense scenes (shrubs, forests) - becomes a blurry mix