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

<u>Results, 9 epochs</u>
- Immediately recognizable minecraft world
- Does not respond well to inputs -- seems to shift only on initial delta in movements/camera
- No collapse! Maintains "minecraft-ness".
- Not yet generalized to orientations. Movements + camera result in a few pixels appearing/disappearing, but frame of reference does not change



<u>Results, 36 epochs</u>
- Improvement on continued frame changes as movement+camera keys are held down
- Detail improvement - medium objects like trees are beginning to appear
- Still no improvement on orientation - camera movement only changes groups of pixels at a time rather than shifting entire scene.
- Unique scenes depending on sequence of camera+movement produced!
- Suspect improved dataset or autoregressive strategy needed for model to understanding orientation changes
- W (forward) results in the most significant scene changes - suspect due to dominance in dataset
- Still struggles with dense scenes (shrubs, forests) - becomes a blurry mix

**v3.0.1**
- Increased base rollout length to 15

<u>Results, 43 epochs</u>
- Continued training from 36 epoch checkpoint, increased base rollout so seq_len was 22-23 frames
- Stronger reaction to camera movements. Sudden mouse movements could completely change the scene
- "w" results are beginning to show "enlarging" effect of moving forwards
- Understanding general structure of terrain better - separation of water/land/sky
- Still confused by detailed terrain like bushes
- Beginning to learn unwanted details, e.g. the pixel-ness of the VAE

**v4.0**
Dataset:
- Changed to GameFactory, 69.6hrs with unbiased action sampling + focus on isolated actions
- Native 16FPS, 640x360 source videos → resized to 128x72 and encoded with VQ‑VAEv2.1.6 to 32x18 latent grids.

Temporal Training:
- Added mixed‑precision training (autocast + GradScaler) for more memory efficient rollouts.
- Switched to streaming autoregressive loss over time steps (using `model.step`) to avoid storing all logits and reduce peak VRAM.
- Updated world model to `action_dim=5` and latent dims `H=18, W=32`, `[yaw, pitch, move_x, move_z, jump]` actions aligned with GameFactory controls.

Preprocessing
- New `preprocess.py` path for GameFactory: uses `metadata/*.json` + `video/*.mp4` and writes `preprocessedv4`.
- Action encoding: `actions[t] = [yaw_delta, pitch_delta, move_x, move_z, jump]` where yaw/pitch are normalized deltas, `move_x/move_z` are in {−1,0,+1} from `ad/ws`, and `jump` comes from `scs` (Space).
- Outputs per‑video `.npz` with `tokens: [K, 18, 32]` and `actions: [K−1, 5]`, plus a `manifest.json`

**v4.0.1**
- Fixed gradient collapse issue due to applying FiLM to each timestep, causing gradient explosion over longer steps