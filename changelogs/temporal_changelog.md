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


**v4.6.0**
train.py
- Loss Rebalancing: Reverted to v4.4.1 baseline to remove conflicting gradients
  - SEMANTIC_WEIGHT: 3.0 → 10.0 (back to v4.4.1 stable baseline)
  - NEIGHBOR_WEIGHT: 2.0 → 1.0 (back to v4.4.1)
  - NEIGHBOR_EXACT_MIX: 0.3 → 0.1 (back to v4.4.1)
- Removed v4.5 Loss Functions: Deleted entropy_regularization_loss(), sharpness_loss(), temporal_consistency_loss()
  - These created conflicting gradients in v4.5/v4.5.2 causing mode collapse
  - Sharpness (↑ confidence) vs Entropy (↑ diversity) = opposing forces
  - Temporal (↓ changes) vs Sensitivity (↑ responsiveness) = contradiction
- New Loss: Gumbel-Softmax in SemanticCodebookLoss
  - Forces discrete token selection during forward pass (hard=True)
  - Prevents soft averaging that causes blur
  - Annealing schedule: tau 1.0 → 0.1 over 20k steps (GUMBEL_TAU_STEPS)
  - Maintains differentiability through Gumbel gradient estimator
- New Loss: LPIPS Perceptual Loss (LPIPS_WEIGHT = 0.3)
  - Learns from VQ-VAE v2.1.5 success: LPIPS fixed identical blur/averaging problem
  - Decodes predicted and target tokens to RGB, computes perceptual distance
  - Applied every 5 timesteps for efficiency (LPIPS_FREQ = 5)
  - VQ-VAE decoder already loaded for visualization → no additional memory cost
  - LPIPS expects inputs in [-1, 1] range (pred_rgb * 2.0 - 1.0)
- Simplified Loss Calculation: Only 3 components (was 5 in v4.5.2)
  - loss_step = (NEIGHBOR_WEIGHT * loss_space) + (SEMANTIC_WEIGHT * loss_texture) + (LPIPS_WEIGHT * loss_lpips)
- Updated Logging:
  - Added: train/loss_lpips (track perceptual loss)
  - Added: train/gumbel_tau (monitor annealing progress)
  - Removed: v4.5 loss component logs (loss_entropy, loss_sharpness, loss_temporal)

Hypothesis:
- Gumbel-Softmax forces discrete commitments → prevents averaging
- LPIPS provides perceptual sharpness signal → what "sharp" looks like in pixel space
- Complementary mechanisms: discrete selection + perceptual feedback = sharp predictions
- Based on VQ-VAE empirical proof that LPIPS fixes MSE-based blur

Expected Outcomes:
- Conservative: Recover to v4.4.1 levels (unique_codes ~90-110, stable but still blurry)
- Optimistic: Significantly exceed v4.4.1, approaching VQ-VAE sharpness levels
- Visual: Sharp recognizable features (trees, blocks, textures)
- Metrics: loss_lpips should decrease, gumbel_tau should anneal 1.0→0.1

Critical Context:
- **No world model version has achieved true sharpness yet** (v4.4.1 was blurry blob)
- VQ-VAE produces sharp GT images → proves capability exists
- World model uses MSE semantic loss → same failure mode as early VQ-VAE
- VQ-VAE v2.1.5 changelog: "Grey averaging fixed! Sharpness fixed!" with LPIPS
- This approach targets root cause: token-space MSE loss insufficient for sharpness

Results, step 38k:
- **Metrics (WandB Charts)**:
  - `unique_codes`: 60-75 range throughout training (stable, no collapse vs v4.5.2, but lower than v4.4.1's 95-100)
  - `entropy`: ~1.5-2.0 stable (lower than v4.4.1's 2.5-3.0, suggests more confident predictions)
  - `confidence`: ~0.6-0.65 (healthy range, not overconfident)
  - `spatial_gradient`: Increased from 0.05→0.32 (sharpness proxy improving)
  - `loss_lpips`: Flat at ~0.0 throughout training (LPIPS not contributing signal - likely issue)
  - `gumbel_tau`: Clean annealing from 1.0→0.1 over 20k steps as designed
  - `loss_texture`: 0.04-0.08 range (very low, similar confidence issue as v4.5.2)
  - `loss_space`: Decreased 5.0→0.2 (spatial loss improving)
  - `train/loss`: Overall loss decreased 6.0→0.8 (smooth learning curve)
  - `grad_norm`: Stable 0.5-1.5 (no spikes, healthy gradients vs v4.5.2's instability)
  - `teacher_force_prob`: Clean decay 0.95→0.15 (scheduled sampling working)
  - `ar_loss_gap`: 0.2-0.6 range (exposure bias well controlled)
  - `action_diagnostics/sensitivity`: 0.5-0.6 stable (MUCH better than v4.5.2's 0.05-0.2)
  - `film_gamma_magnitude`: 0.85-0.95 (strong FiLM signal)
  - `film_beta_magnitude`: 0.16-0.22 (healthy modulation)

- **Visual Quality (live_inference.py --greedy @ 38k steps)**:
  - **Best visual acuity achieved so far** in any world model version
  - Grassland scenes: Beginning to show **texture details** - dirt visible underneath green top of grass blocks (first time seeing block-level detail!)
  - Desert scenes: Still blurry mess, no texture detail
  - Water scenes: Maintains coherent blue regions, better than v4.4.1
  - Forest/dense biomes: Still struggle with detail, but trees more distinguishable than v4.4.1
  - Overall: Still blurry, but **incremental improvement** - some scenes show micro-textures
  - Camera tracking: Maintains orientation well, stable over time
  - Temporal stability: Reduced shaky artifacts vs v4.4.1

- **WandB Visual Logging (GT vs Pred comparisons)**:
  - Step 38k samples show predictions maintaining general color/structure
  - GT images remain sharp (VQ-VAE working correctly)
  - Pred images show some texture attempting to form, but still averaged
  - Better than v4.4.1 samples (less blob-like), worse than GT (still blurry)

- **Key Observations**:
  - **Gumbel-Softmax worked**: No mode collapse, stable training, better gradients
  - **LPIPS did NOT work**: loss_lpips stayed at ~0.0 (likely implementation issue - may need debugging)
  - `loss_texture` extremely low (0.04-0.08) suggests model very confident in narrow predictions
  - `unique_codes` only 60-75 (vs 95-100 in v4.4.1) - still some mode restriction
  - **Visual improvement despite LPIPS failure** - Gumbel-Softmax alone helped
  - Action sensitivity MUCH better (0.5-0.6 vs v4.5.2's 0.05-0.2)

- **Diagnosis**:
  - Gumbel-Softmax successfully forces discrete commitments → prevents soft averaging
  - LPIPS flat at zero → either (1) not being computed correctly, (2) pred_rgb == target_rgb exactly (unlikely), or (3) gradient flow issue
  - Model learning but still constrained to ~60-75 codes → need broader exploration
  - Texture detail emerging in some biomes (grass) but not others (desert) → biome-specific learning
  - Still far from VQ-VAE sharpness, but **clear incremental progress**

- **Next Steps**:
  1. Debug LPIPS implementation - check if loss is actually being computed
  2. If LPIPS working: Increase LPIPS_WEIGHT from 0.3 to 1.0
  3. If LPIPS broken: Fix computation, ensure gradients flow
  4. Consider increasing unique_codes target - maybe reduce Gumbel tau annealing speed
  5. Continue training to 50k-60k steps to see if texture detail improves further


**v4.6.1**
train.py
- LPIPS Gradient Flow Fix: Corrected critical bug preventing LPIPS from contributing to training
  - v4.6.0 bug: `with torch.no_grad():` wrapped entire LPIPS computation block (lines 489-499)
- Increased batch size to 28, since v4.6.0 stayed at ~50% gpu memory usage

**v4.6.2**
train.py
- LPIPS Gradient Flow Fix (ATTEMPT 2 - COMPLETE): Use Gumbel-Softmax for differentiable token selection
  - Replaced argmax with soft Gumbel-Softmax probabilities (`hard=False`)
  - Soft embedding lookup via weighted sum over VQ-VAE codebook
  - Decode soft embeddings directly through VQ-VAE decoder (bypassing quantizer)
  - Gradients now flow: `loss_lpips → pred_rgb → decoder → soft_embeddings → probs → logits_t`
  - Uses same tau annealing schedule as semantic loss for consistency

Results, step 59k:
- Visual quality improvements
  - First version showing recognizable micro-features: sand hills, sand crevices, trees, cacti
  - Transitioned from "land vs sky blobs" to distinct environmental details
  - Features phase in/out during inference but maintain coherence
  - Significant improvement over all previous versions
  - Best visual quality achieved so far
  - Not yet at VQ-VAE GT quality but clear trajectory toward it


- Temperature Sensitivity:
  - Greedy/low temp: Best accuracy, minimal flickering, fewer unique features
  - High temp (1.0): More diverse features (sand hills, vegetation), increased flickering BUT significantly reduced vs previous versions
  - Previous versions at temp=1.0: "massive mess of colors" - v4.6.2 maintains coherent features
  - Temperature resilience demonstrates robust learned representations

- Action Response:
  - Best camera input response across all versions
  - Clear reaction to yaw/pitch/movement in live_inference.py
  - Not yet detailed enough to fully differentiate specific action types (work in progress)

- Metrics Summary:
  - `loss_lpips`: 0.24-0.25 stable
  - `unique_codes`: 85-95 
  - `spatial_gradient`: 0.34-0.36 (well above >0.2 target for sharp edges)
  - `action_sensitivity`: 0.5-0.6 (exceeds >0.3 target)
  - `confidence`: 0.60-0.65 (healthy 0.5-0.8 range)
  - `loss_texture`: 0.05-0.065 
  - `grad_norm`: <2.5 (stable training)
  - `ar_loss_gap`: 0.25-0.3 (exposure bias controlled)

**v4.6.3**
train.py
- Increased `LPIPS_WEIGHT` from 0.3 to 1.0

Results, step 80k:
- Visual Quality:
  - Marginal improvement over v4.6.2: slightly more individual blocks visible in terrain
  - Sky-ground separation marginally sharper, less color bleeding
  - Still predominantly Gaussian blur with no 3D block texture detail
  - Individual blocks "vibrate" between frames (temporal flickering reduced but still significant)
  - Long rollouts (>5-10s) still fail biome conditioning - looking away and back changes biome

- Metrics Summary (vs v4.6.2 @ 59k):
  - `loss_lpips`: 0.225-0.235 (improved from 0.24-0.25, -6%)
  - `unique_codes`: 90-100 (healthier diversity, up from 85-95)
  - `ar_loss`: 0.65-0.7 (improved from 0.75-0.8, -10%)
  - `spatial_gradient`: 0.34-0.36 (unchanged)
  - `action_sensitivity`: 0.55-0.6 (stable)
  - `loss_texture`: 0.048-0.06 (unchanged, still very low)
  - `grad_norm`: <2.5 (stable training)

- Key Observations:
  - LPIPS weight increase (0.3→1.0) had limited impact - only 5-10% visual improvement
  - LPIPS loss decreased marginally (0.24→0.23) suggesting local optimum reached
  - Metrics remain healthy but visual quality plateau evident
  - Gap to VQ-VAE ground truth decreased, but remains large
  - Model confident (`loss_texture` very low) but still produces blur (confidence != sharpness)

- Limitations:
  - No 3D block texture (grass blocks lack top/side distinction)
  - Gaussian blur dominates most terrain
  - Temporal flickering - block details appear/disappear sporadically
  - Long rollout conditioning failure - biome changes during extended inference


**v4.6.4**
train.py
- Removed neighborhood loss: `NEIGHBOR_WEIGHT` from 1.0 to 0.0 due to potentially allowing 3x3 neighborhood jitter
- Increased AR from 18 -> 25

Results, step 101k:
- Visual Quality:
  - 2x sharpness improvement - spatial_gradient: 0.34-0.36 -> 0.6-0.7 (biggest gain across all versions)
  - LPIPS improved 20%: 0.24-0.25 -> 0.18-0.20 
  - Critical artifact discovered: Global flashing every 5 frames during inference
    - Frames 0-4: Gaussian blur, low detail
    - Frame 5: Flash of sharp features (sand hills, texture visible)
    - Pattern repeats indefinitely (confirmed >1min rollout)
    - Different from v4.6.3's localized jittering - now synchronized global flash
    - Highly likely due to LPIPS only computed at every 5th timestep

- Metrics Summary:
  - `spatial_gradient`: 0.6-0.7 (doubled from 0.34-0.36)
  - `loss_lpips`: 0.18-0.20 (improved 20% from 0.24-0.25)
  - `unique_codes`: 65-78 (dropped 25% from 90-100) - WARNING
  - `entropy`: 0.3-0.4 (halved from 0.6-1.0) - WARNING
  - `confidence`: 0.88-0.92 (increased from 0.8)
  - `loss_space`: 4.5-5.0 (increased from ~2.0 but weight is 0.0)
  - `ar_loss_gap`: ~0.0 (no exposure bias)
  - `grad_norm`: <2.0 (stable)


**v4.6.5**

train.py
- Core Changes:
  - `LPIPS_FREQ`: 5 -> 1 (every step) - eliminates 5-frame periodic flashing artifact
  - `LPIPS_WEIGHT`: 1.0 -> 2.0 - LPIPS becomes dominant gradient signal (53% vs 47% semantic)
  - `CURRICULUM_AR`: Disabled during training - AR moved to validation-only, per 500 steps - since AR loss gap was negative-ish to zero

- Optimizations:
  - Removed 0-weighted neighborhood loss computation 
  - Removed scheduled sampling logic (always teacher-forced)
  - Removed action perturbation test (redundant diagnostic)
  - Removed unused variables and constant logging

- Expected Behavior:
  - No periodic flashing (LPIPS every step provides consistent signal)
  - Strong perceptual gradient (~10x stronger than v4.6.4)
  - Potential mode collapse risk - keep watch on `unique_codes`

Results - skipped due to GPU budget exhausted

**v4.6.6**

train.py

- Single-step AR mix: 5% of training steps use AR vs 20 frame rollout in v4.6.4
- Removed unused WandB metrics and unused variables, added `ar_mix/actual_frequency` tracking
- Checkpoint saving: step-based naming (e.g., `ochre-v4.6.6-step80k.pt`), auto-save every 10k steps

- Expected Behavior:
  - `validation/ar_loss_gap` stays <0.5 (AR capability preserved)
  - Training stability maintained (no grad_norm spikes)
  - 40-60x more efficient than v4.6.4 full AR rollout
  - If AR gap >0.5: increase AR_MIX_PROB to 0.10
  - If unstable: reduce to 0.02 or disable

**v4.6.7**

train.py

- Timing instrumentation: Added comprehensive timing tracking with EMA smoothing (α=0.1, ~10 step window) for throughput analysis and bottleneck identification
- WandB metrics: New `timing/*` namespace tracking step_total, data_load, forward, lpips, backward, optimizer, ar_validation (all in ms) + throughput_steps_per_sec
- Console output: Prints timing breakdown every 10 steps with steps/sec and section-level breakdown (e.g., "1.24 steps/s | Total: 806ms (Data: 50ms, Fwd: 620ms, Bwd: 85ms, Opt: 45ms, LPIPS: 480ms)")

- Expected Behavior:
  - Minimal overhead (<0.1% from time.time() calls)
  - Stable metrics via EMA smoothing
  - Identifies performance bottlenecks (e.g., LPIPS-bound vs forward-bound)

**v4.7.0**

train.py
- Restored v4.6.4 AR curriculum: `CURRICULUM_AR=True`, `AR_ROLLOUT_MAX=25`, gradual ramp 5k-15k steps
- Real-time Monitoring: Added `train/loss_teacher` vs `train/loss_ar` and component-specific `grad/film_norm` vs `grad/dynamics_norm`
- Validation Update: Removed offline AR validation loop (speedup); added explicit `ar_len` logging
- Action Metrics: `action_response/camera_left_diff` etc. to verify FiLM efficacy
- Step-based training: Removed epoch loop in favour of step-based
- seq_len deprecation: Made AR len override seq_len which previously capped max AR to 19.

model_convGru.py
- FiLM capacity: Internal MLP hidden dim 256 -> 512 for stronger action encoding

Target: Fix v4.6.6 action conditioning failure (duplicated tails during camera movement) via stronger FiLM + AR exposure

Results, step 50k:
- Visual Quality:
  - Reconstruction: Significantly blurry (all samples), scene structure preserved but details lost (e.g. birch biome only has green, no white birch trunks)
  - AR Rollout: Stable blur across 25 frames, no compounding/splitting/drift (major improvement over v4.6.6)
  - Action Conditioning: Non-functional (outputs identical for different actions)
- Metrics Summary:
  - `loss_lpips`: 0.23-0.24 (doubled from v4.6.6's 0.11-0.13 during AR ramp-up 10k-15k)
  - `spatial_gradient`: 0.65 (degraded from v4.6.6's 0.98-1.02)
  - `unique_codes`: 85-90 (down from v4.6.6's 96-98)
  - `action_response/average`: 0.015 (30% of target >0.05, broken like v4.6.6)
  - `ar_loss_gap`: 0.3 (excellent AR stability)
- Key Observations:
  - AR stability achieved: 25-frame rollouts coherent, no v4.6.6 catastrophic failures
  - Visual quality degraded: LPIPS doubled coinciding with AR curriculum 10k-15k
  - Action conditioning still broken: FiLM capacity increase (256->512) ineffective
  - Trade-off: Solved AR stability, lost single-frame sharpness vs v4.6.6

**v4.7.1**

train.py
- Adaptive AR Brake: Guaranteed minimum AR exposure (ar_len>=3 after warmup) with quality-based adjustments (TF vs AR LPIPS split)
  - Brake logic: If AR LPIPS >1.8× TF → reduce ar_len; if <1.3× TF -> increase ar_len (with hysteresis to prevent oscillation)
  - EMA smoothing (alpha=0.98) for stable feedback signal
- Action-Contrastive Ranking Loss: Explicit correctness supervision every 10 steps
  - Hinge loss: true action should predict target better than shuffled action by margin 0.05
  - Weight: 0.5, uses fresh h_state for efficiency (~10-15% overhead vs ~30% for replay)
- FiLM LR Multiplier: 3× base LR for FiLM/action parameters to address gradient imbalance (dynamics 5.6× stronger than FiLM in v4.7.0)
- Enhanced Logging: TF vs AR LPIPS split, brake state counters (increase/decrease/stable), action ranking loss
- Critical Bug Fixes (pre-training):
  - Fixed LR warmup overwriting FiLM multiplier (would have broken action training)
  - Added CURRICULUM_AR gate to brake function
  - Enhanced brake observability with state counters

Target: Fix v4.7.0's LPIPS degradation (0.12→0.24) and action conditioning failure via adaptive AR curriculum + explicit action supervision

Expected: TF LPIPS 0.11-0.15, AR LPIPS <2× TF, action_response >0.05 (vs 0.015)

**v4.7.2**

train.py
- Replaced coarse forward/backward timings with per-step EMA breakdown (embed+FiLM, model_step, semantic, LPIPS total + ms/call, action_rank, backward sub-steps, optimizer sub-steps)

**v4.7.3**

train.py
- Turned `SEMANTIC_WEIGHT = 0` and added conditional skip computation when 0
- Doubled `BATCH_SIZE`, `LR`, `MIN_LR`, `WARMUP_STEPS`

Results, step 40k:
- `live_inference.py` quality faces the same issues as previous verison - strong initial scene that maintains quality until camera movements triggered -> scene collapse
  - WASD+jump fully ignored
- Wandb indicates poor FiLM gradients, increasing but then *declining* gamma and sensitivity.
- AR len never increases: Starts at AR len 3, but due to AR loss decreasing slowly at the *same* rate as TF loss, LPIPS ratio stays around ~1.5.

- Possible culprits:
  - Insufficient AR learning signals vs TF signals
  - Weak action conditioning still - ~0.03 `film_norm` vs ~0.4 `dynamics_norm` -> 13x slower learning for FiLM

**v4.7.4**

train.py
- Re-enabled semantic loss: `SEMANTIC_WEIGHT = 0.5` (provides gradients to action pathway)
- AR loss upweighting: `AR_LOSS_WEIGHT = 2.5` (emphasizes deployment condition, AR now ~40% of loss vs 15%)
- FiLM gradient balancing: `FILM_LR_MULT = 15.0` (addresses 13× gradient imbalance: film_norm=0.03 vs dynamics_norm=0.4)
- Stronger action supervision: `ACTION_RANK_WEIGHT = 2.0`, `ACTION_RANK_FREQ = 5` (4× stronger, 2× more frequent)

Target: Fix v4.7.3's gradient imbalance + LPIPS ratio stuck at 1.45-1.5 preventing ar_len progression

Results, step 50k:
- Gradient imbalance worsened: 13× → 33× (film_norm: 0.03→0.09, dynamics_norm: 0.4→3.0)
  - FILM_LR_MULT=15 insufficient - dynamics gradients increased 6× while film only 3×
- Action conditioning still broken: film_gamma declining (6.5→3.0), WASD/jump non-functional
  - action_response metrics <0.03 (target >0.05), action_rank/loss stuck at 0.049 (target <0.01)
- LPIPS ratio stuck at 1.45 - ar_len remains at 3, no progression
- Visual improvements: 3D block edges visible during camera movement (new), WandB samples nearly match GT
- Live inference: Initial frames strong, static rollouts stable, but camera input causes quality drop (better than v4.7.3 though - retains 3D structure for <20s before blob collapse)
- Diagnosis: Gradient sign conflict suspected - film_norm increasing but gamma declining suggests reconstruction losses dominating and learning "ignore actions"

**v4.8.0**

train.py
- Potentially critical fix for FiLM conditioning: teacher forcing now uses previous frame (not target frame) as input
  - Previous versions: `x_in = X_seq[:, t]` during teacher forcing (target frame embedding → actions irrelevant)
  - v4.8.0: `x_in = X_seq[:, t-1]` during teacher forcing, so must use hidden state and actions to predict next frame, much more like `live_inference.py`
Target: Force model to learn action conditioning by removing visual shortcut that made actions irrelevant

**v4.8.1**

train.py
- Multi-step action conditioning validation: measures action response at 1, 5, and 10 step rollouts
  - New metrics: `action_response/average_{1,5,10}step` to catch degradation over longer AR sequences
  - Replaced old AR rollout visualization with 30-frame action rollouts (4 actions × 6 timesteps)
  - Logs to `visuals/action_rollout_30step` - shows if action conditioning persists or collapses during extended rollouts
Target: Early detection of action conditioning failures that only appear during multi-step inference (matching live_inference.py failure mode)

**v4.9.0**

train.py
- AR gradient flow via Gumbel-Softmax: Removed gradient detachment in AR rollout to enable BPTT through time
  - Soft token selection allows gradients from frame t+10 to improve action conditioning at frame t
- Aggressive curriculum: `AR_MIN_LEN = 10` (3->10), `AR_BRAKE_RATIO_LOWER = 1.6` (1.3->1.6), `AR_BRAKE_RATIO_UPPER = 2.0` (1.8->2.0)
  - Forces longer AR rollouts (10 frames) where visual shortcuts should fail, making actions necessary
- Action ranking improvements: `ACTION_RANK_FREQ = 1` (every step, was 5), realistic hidden states from forward pass
  - 5x stronger action supervision signal (every step vs every 5)
  - Captures actual h_state during unroll instead of fresh init (memory efficient single-timestep sampling)
- Gradient clipping increased: 0.5->1.0 to provide headroom for longer BPTT gradient chains
Target: Combined approach - attack action conditioning from multiple angles (visual shortcut + gradient flow + stronger supervision)

Results, step 80k:
- Catastrophic failure: Gradient explosion to ~800 in both FiLM and dynamics pathways (10-15k onset)
- Visual quality degraded progressively (10k good → 40k severe → 80k dark/poor)
  - Solid color blocks (brown, gray, orange, black) - repetitive patterns
  - Unique codes: 25-30 (2.4-2.9% of 1024 codebook) vs v4.8.1's 40-43
- LPIPS stable at 0.30-0.35 while visuals collapsed (metric failure)
- Noiser action conditioning metrics: action_rank/loss converged to v4.8.1 baseline (0.048), response metrics weak/noisy
- AR curriculum instability: ar_len reached 20-25 (higher than v4.8.1's 17) but oscillated, quality didn't improve
- BPTT failure mechanism: 10-step gradient chain caused exponential growth (clipping 800→1.0 destroyed 99.875% of gradient info)
- v4.9.0 objectively worse than v4.8.1 across all meaningful metrics - BPTT not viable without architectural changes

**v4.10.0**

- Inverse Dynamics Module: Auxiliary head predicts actions from hidden state transitions (h_t -> h_{t+1}), forces model to encode action info in states
- Reverted v4.9.0 BPTT (gradient explosion to ~800) → stable detached AR from v4.8.1
- Kept: AR_MIN_LEN=10, ACTION_RANK_FREQ=1, realistic h_state, loosened brake (1.6/2.0)
Target: Stable action conditioning via auxiliary supervision (DreamerV3 approach)

**v4.10.1**

- Loss rebalancing to fix mode collapse @ v4.10.0 step 19.5k (unique_codes ~35, repetitive green stripes)
- Increased reconstruction: SEMANTIC 0.5→1.0 (+100%), LPIPS 2.0→3.0 (+50%); Reduced action: IDM 1.0→0.5 (-50%)
Target: Maintain action conditioning (film_gamma >3.0, ranking <0.01) while recovering diversity (unique_codes >70)

Results, step 80k:
- First run achieving functional action conditioning: action_rank 0.003-0.005 (10× better than v4.8.1), stable FiLM magnitudes (3.2-3.3), IDM converged (0.018-0.025)
- Best 3D structure preservation: Approximate block shapes retained vs previous 2D collapse
- Gradient imbalance improved: 16× (vs v4.8.1's 25×)
- Unique codes recovered: 43-45 (vs v4.10.0's 35)
- Camera response hierarchy: Pitch (up/down) excellent; Yaw (left/right) degrades over rollouts; WASD minimal; Jump partial (upward drift only)
- Hierarchical action learning observed: Model learned by gradient signal strength (pitch > yaw > movement)
- AR-sharpness trade-off: AR curriculum reduces reconstruction sharpness (~20% LPIPS increase) for rollout stability

**v4.11.0**

- Variable-span IDM ("Time Telescope"): predict cumulative action from (h_{t-k}, h_t) with time-delta embedding, k val range [1..5]
- Movement-weighted IDM loss: 10× Move_X/Move_Z, 5× Jump
- FiLM clamping: clamp gammas/betas to [-5, 5] to reduce 1-frame flash artifacts
- Gradient/curriculum tuning: FILM_LR_MULT 15→25, AR_BRAKE_RATIO_UPPER 2.0→2.5, ACTION_RANK_WEIGHT 2.0→1.0
Target: Improve WASD responsiveness and yaw stability over longer rollouts while preventing FiLM overshoot artifacts

### v5.0

Dataset:
- 15D discrete action encoding: yaw(5 one-hot) + pitch(3 one-hot) + WASD(4 multi-hot) + jump/sprint/sneak(3)

model_convGru.py
- Capacity scaling defaults: embed_dim 320, hidden_dim 640, layers 6; residual connections every 2 layers
- FiLM scaling: internal dim 512→max(1024, 2×hidden_dim) (e.g. 1280 @ 640)
- IDM scaling: aggressive pooling 9×16→4×8 + deeper MLP (+dropout) to control params; keep time-delta embedding + variable-span k∈[1..5]

train.py
- IDM training fix: CE for yaw/pitch bins + BCE for WASD/jump/sprint/sneak over the k-window; ensure IDM gradients hit current dynamics state (no full BPTT)
Target: Remove action-gradient interference (movement vs camera) and add enough capacity for sharper recon + functional movement mechanics

### v6.1

Motivation: v5.0 converged on reconstruction/IDM metrics but failed in live inference with “action watermark/flash” shortcuts and weak/no coherent multi-step motion.

model_convGru.py
- Separate FiLM pathways (camera vs movement) to reduce gradient competition.
- Add lightweight causal temporal attention over pooled recent hidden states to support explicit multi-frame comparison.
- Replace additive temporal position embeddings with relative attention-logit bias (avoid leaking position into V / shortcut patterns).
- Redesign IDM to strided-conv compressor + dt embedding for cheaper, more stable auxiliary supervision.

train.py
- New v6.1 config/loop (`hidden_dim=512`, temporal ctx=8); remove action-ranking; increase reliance on IDM; simplify logging + weights-only checkpointing.
- Fixes discovered during review: correct component-wise VQ-VAE loading; AR brake guardrails; pooled temporal buffer passed into `model.step`.

live_inference.py
- Maintain/pass pooled temporal buffer during warmup + rollout so temporal attention is active at inference (train/inference parity).

### v6.2

Motivation: v6.1 still shows weak/unclear controllability and rapid “2D blob / texture field” collapse under heavy camera input, with early signs of token diversity collapse/plateaus. Dataset diagnostics also show extreme camera+movement co-occurrence (~87%), making movement learning easy to mask.

model_convGru.py
- Add explicit camera warp primitive (discrete yaw/pitch bins → small pixel shifts) applied on the embedding input path: yaw uses horizontal wrap shift; pitch uses vertical shift with zero padding. Intended as a cheap spatial transport bias to reduce “camera mush”.

train.py
- Teacher-forcing token corruption (random token replacement) with a ramp schedule to weaken the “copy z_t” shortcut and improve robustness to action injection.
- Anti-collapse guardrails: slower Gumbel hardening with higher tau floor + temporary entropy bonus early in training; gate AR growth using `unique_codes` to avoid pushing AR while in a collapsed regime.
- Action-conditional LPIPS reweighting: ramp a boost multiplier for movement-active frames (WASD/jump/sprint/sneak) to amplify weak movement gradients under co-occurrence; log `train/lpips_movement_boost` and `train/movement_active_pct`.

Results, step 100k:
- Anti-collapse succeeded: `unique_codes` stabilized ~40–42; entropy/confidence trends healthy (no v6.1-style collapse).
- AR curriculum reached ~15 steps but did not hit max; `lpips_ratio` ~1.5–2.0 and AR LPIPS worsened over training.
- Live inference regressed: strong zero-input drift to a memorized stone texture field within seconds, long-rollout degradation persists.
- Camera controllability worse than prior best: yaw/pitch cause sudden snaps rather than progressive motion (camera warp likely shortcut/instability).
- Movement still non-functional: WASD/jump show no observable effect despite 4× LPIPS movement reweighting; IDM low loss suggests “action-echo” shortcut (actions encoded in h, not used for visual causality).
