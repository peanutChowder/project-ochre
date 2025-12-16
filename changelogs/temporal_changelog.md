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

### v4.0
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

<u>Results, 61k Steps</u>
- Good understanding of camera pitch: Looking up/down maintains the structure of sky/land. However fully looking up into sky/ground sometimes causes model collapse into only sky or ground.
- Limited understanding of camera yaw: In initial ~1s, camera maintains orientation of things when looking left/right. However as longer rollouts cause average blurring, e.g. desert cactus becomes just desert-blob, changes in yaw dont look like they do anything.
- No understanding of movement: WASD+Jump do not seem to do anything 
Runtime
- ~5k Steps / hr

**v4.1**
Training Optimizations & Architecture Fusion:
- Fused ConvGRU gates: Refactored `ConvGRUCell` to compute `z` and `r` gates in a single fused convolution, reducing kernel launch overhead.
- Sequence Pre-computation: Updated `WorldModelConvFiLM` and `train.py` to pre-compute embeddings and FiLM parameters for the entire sequence in parallel *before* the recurrent unroll loop. This moves heavy non-recurrent compute out of the critical path.
- Gradient Checkpointing: Added `use_checkpointing` support to `WorldModelConvFiLM` to trade compute for memory, enabling significantly longer sequence rollouts on limited VRAM.
- Data Loading: Updated `GTTokenDataset` to use `mmap_mode='r'` for `.npz` files, drastically reducing disk I/O and RAM usage for small window samples.

Rollout:
- Step-based rollout: Increasing rollout to every 10k steps

**v4.1.1**
- Enabled checkpointing: was being bypassed.
- Increased batch size from 16 -> 32
- Decreased rollout intervals to 5k steps

Results:
- Steps/hr **decreased**, ~5K/hr -> 1.4K/hr
- Trained at 22 seq_len rollout
- No discernible increase in quality

**v4.2**
live_inference.py
- Added persistent hidden state to allow models to retain larger context window during inference

train.py
- Disabled gradient checkpointing due to poor tradeoff of saved memory but large speed slowdown
- Added autoregressive unrolling curriculum - mistakenly confused previous seq_len (input context window) for AR unrolling curriculm
  - `ar_len` ramps up from 0 to `AR_ROLLOUT_MAX` over `AR_RAMP_STEPS` starting after `AR_START_STEP`
- Moved FiLM modulation to before GRU to carry action information into hidden state

**v4.2.1**
- Decreased batch size 32 -> 16, v4.2 ran into OOM error.

live_inference.py
- Added stochastic sampling option
- replaced rolling token/action window with single current token +action per step

Results:
- Frame quality: outputs are sharper, but are only made up of "ground"
  - e.g.: in desert biomes, no cacti or blocks are present - instead, a sandy terraria-like environment has a sharp boundary with the sky, but no biome features.
  - e.g.: in grass/forest biomes, a greenish average appears. Indistuinguishable features, only made up of a green blur.
- Frame quality: Some scenes are beginning to over-learn the vqvae's grid pattern
- Movement: Even without movement, the scene has constant localized shifts - e.g. little spikes in the ground shift around, but general scene remains stable
  - Cannot discern whether new action conditioning is working due to this constant shifting
- Camera rollout: Excellect maintenance of ground/sky when looking up/down, but looking left/right is not discernible due to lack of detail and the constant shifting spikes in scene.

**v4.3**
train.py
- Dual Loss: Implemented SemanticCodebookLoss (Embedding MSE) to fix "blurry" texture predictions and neighborhood_token_loss to tolerate spatial jitter.
- Shape Correction: Dynamically loads codebook_size (1024) from the VQ-VAE checkpoint and passes it to the World Model (fixing the default 2048 mismatch).

vqvae.py 
- Batch Support: Updated decode_code to allow batched inputs (B, H, W) by removing the restriction size(0) == 1 to enable visualization logs

live_inference.py
- Shape correction: Fixed the codebook_size to 1024

**v4.4**
train.py
- AR Curriculum fix: Dataloader now reloads with recomputed `ar_len`, `seq_len` change. v4.3 previously had AR permanently set to 0.
- Scheduled Sampling: Implemented inverse sigmoid curriculum for teacher forcing probability (lines 77-86), addresses exposure bias by gradually transitioning from teacher forcing (95%) to autoregressive (5%) over 30k steps
- AR Curriculum Optimization: Reduced AR_ROLLOUT_MAX from 49->24 steps and AR_RAMP_STEPS from 30k->20k for gentler learning curve; increased BASE_SEQ_LEN from 1→16 for better context
- AR Loss Gap Tracking: Added separate loss tracking for teacher-forced vs autoregressive steps to quantify exposure bias
- Action Diagnostics: Implemented FiLM magnitude tracking (gamma/beta) and action sensitivity testing via perturbation analysis every 100 steps

model_convGru.py
- Public Embedding Method: Added _embed_tokens() helper (lines 133-142) to support scheduled sampling by enabling re-embedding of predicted tokens during autoregressive rollout

Results:
- `unique_codes` collapses to 0 by step 70. Does not change at all.

**v4.4.1**
- Fixed CUDA assertion error due to bug causing world model to initialize `codebook_size=embedding_dim=384` instead of `codebook_size=1024`
- Scheduled Sampling Formula Fix: Corrected inverse sigmoid to use `1/(1+exp(...))` instead of `k/(k+exp(...))` for proper decay curve

Results, 40k steps:
- with `--greedy` flag
  - Maintains general shape and colors of biome - e.g. desert scenes are blurry sand land, looking up away from land temporarily and back still maintains land shape
  - Tracks camera movement fairly well
  - Small localized shaky changes throughout terrain without movement input - e.g. little bumps forming in a flat grassland and disappearing
  - Still lacks most biome terrain details - e.g. 3D hills, block shapes, trees, cacti... etc. Only exception is water - looking left and right in a grass biome can eventually lead to small blue spots that eventually evolve into shapes of lakes.
  - Visual quality is too blurry to discern if movement has an impact
- with `--temperature 0.1`
  - Performs similarly to `--greedy` flag
- with default flags (only checkpoints and context provided)
  - Entire world always has constant shifting, has less blur and more intense visuals but land/sky border dissolves. In grassy biomes brown dirt/tree textures phase in and out, even without any input

**v4.5**
train.py
- Loss Rebalancing: Adjusted weights to prevent model averaging/hedging behavior
  - SEMANTIC_WEIGHT: 10.0 -> 3.0 (reduce dominance of embedding-space loss)
  - NEIGHBOR_WEIGHT: 1.0 -> 2.0 (increase spatial precision)
  - NEIGHBOR_EXACT_MIX: 0.1 -> 0.3 (more exact-pixel pressure)
- New Loss Functions:
  - `entropy_regularization_loss()`: Penalizes overconfident predictions below target entropy (4.5), encourages codebook exploration
  - `sharpness_loss()`: Rewards high-confidence committed predictions, discourages blurry averaging
  - `temporal_consistency_loss()`: Penalizes rapid prediction changes between timesteps via KL divergence, reduces shaky artifacts
- Curriculum Acceleration:
  - AR_RAMP_STEPS: 20k -> 10k steps (Faster AR ramp)
  - AR_ROLLOUT_MAX: 24 -> 32 steps 
  - SS_K_STEEPNESS: 6000 -> 4000 (faster scheduled sampling decay)
  - SS_MIDPOINT_STEP: 15k -> 10k (earlier teacher forcing transition)
  - SS_MIN_TEACHER_PROB: 0.05 -> 0.15 (maintain more teacher signal to reduce exposure bias)
- Gradient Stability: Tighter gradient clipping (1.0 -> 0.5) to address sporadic spikes
- Diagnostics:
  - `spatial_gradient`: Spatial gradient magnitude as sharpness proxy
  - `confidence_std`, `confidence_min`: Confidence distribution statistics
  - Individual loss component tracking: `loss_entropy`, `loss_sharpness`, `loss_temporal`
- Wandb visualization fix: Fixed WandB image logging to properly convert [0,1] float tensors to uint8 numpy arrays

Target Improvements:
- Entropy: Increase from 2.5-3.0 toward 4.0-4.5 (more exploration)
- AR loss gap: Stabilize or decrease from 2.0 (reduce exposure bias)
- Spatial gradient: Increase (sharper predictions)
- Visual quality: More committed/sharp details, reduced blurriness
- Temporal stability: Reduced shaky artifacts without input

Results after 7.7k steps:
- OOM after 7.7k steps
- Unique code decrease from v4.4: ~85 -> ~60 
- Entropy increase from v4.4: ~2.5 -> ~3.5
- Overall loss decrease from v4.4: ~6 -> ~5


**v4.5.1** 
train.py
- OOM Fixes: 
  - BASE_SEQ_LEN: 16 -> 20 (accommodate AR rollout)
  - AR_ROLLOUT_MAX: 32 -> 18 (fit P100 memory constraints)
  - Fixed compute_curriculum_params() to lock seq_len at BASE_SEQ_LEN 
  - Previous bug: seq_len = max(BASE_SEQ_LEN, ar_len + 1) caused 16->33 growth during training
  - seq_len fixed at 20, ar_len capped at 19 max



**v4.5.2** 
train.py
  - Fixed VQ-VAE decoder not loading correctly, causing static/noise images in WandB
  - Checkpoint uses component-wise format (encoder/decoder/quantizer dicts), properly loading each component separately
  - Visual logs now show actual decoded RGB images instead of raw token noise
- Performance Optimizations (estimated +4-7% throughput):
  - Action sensitivity diagnostic: Reduced frequency from every 100 to every 500 steps 
    - Uses FiLM magnitude as instant proxy between exact computations
    - Saves ~3-4% throughput by avoiding full forward pass
  - DataLoader: Set persistent_workers=True 
    - Keeps worker processes alive between epochs
    - Eliminates worker restart overhead (~0.5-2% gain)
  - Temporal loss: Removed redundant detach() operation (line 452)
    - Gradient blocking moved to temporal_consistency_loss function (line 187)
    - Reduces memory fragmentation (~0.5-1% gain)
- Expected cumulative improvement: +4-7% throughput (3,390 -> 3,526-3,627 steps/hr)

Results, step 68k:
- Visual Quality (`--greedy`): Highly blurry outputs, vague terrain shapes but no definitive features or recognizable objects. Some semblance of biomes (possible forests) but not a significant step up from previous versions despite metric improvements.
- Issues:
  - `unique_codes`: Dropped 35-40% (95-100 → 60-75) - severe mode collapse
  - `loss_texture`: Collapsed to near-zero (~0.05 vs v4.4's ~0.7-0.9) - extreme confidence in limited vocabulary
  - `sensitivity`: Dropped 75-90% (0.5 → 0.05-0.2) - model barely responding to action inputs despite stronger FiLM magnitudes
  - `grad_norm`: Increased instability (spikes to 10 vs v4.4's stable ~1-2) - optimization struggling with multiple loss objectives
- Positive Changes:
  - Curriculum losses improved: `teacher_loss` (8→2-3), `ar_loss` (10→3-4), `ar_loss_gap` (1→0.5-1)
  - `spatial_gradient` increased (0.15→0.22-0.25) but doesn't translate to visual quality
- Diagnosis: Model solving losses by overfitting to narrow solution space - high confidence on few codes rather than diverse exploration. Visual blur + low sensitivity suggest losses not aligned with perceptual quality or action conditioning goals.
- Potential fix?: Increase entropy regularization weight/target to force broader codebook usage, potentially rebalance loss weights to prioritize diversity over confidence.

