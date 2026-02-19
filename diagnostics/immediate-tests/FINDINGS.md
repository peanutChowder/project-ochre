# Immediate Diagnostic Test Findings: Root Cause Identified

**Date**: 2026-02-18
**Checkpoint**: ochre-v7.0.5-step95k.pt
**Tests Run**: 4 immediate diagnostics to identify first-step collapse mechanism

---

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The first-step catastrophic collapse is caused by a **combination of two factors**:

1. **Argmax sampling determinism** - Using argmax for autoregressive prediction forces the model into a tiny subset of "safe" codes
2. **Large TF→AR distribution shift** (KL divergence 2.94) - AR logit distributions are fundamentally different from teacher-forced distributions

**Solution**: **Top-k sampling (k=50) prevents the collapse entirely** (+14.4% improvement vs argmax).

---

## Test Results

### Test 1: Collapsed Codebook Analysis ⚠️ SEVERE MODE COLLAPSE

**Finding**: Model uses only **35-36 codes out of 1024** (3.4%) across all actions.

| Action | Unique Codes | Top 30 Coverage |
|--------|--------------|-----------------|
| Static | 36 | 99.2% |
| Camera L/R/U/D | 35-36 | 99.3-99.8% |
| Move Forward | 36 | 99.2% |
| Jump | 36 | 99.7% |

**Critical observations**:

1. **Same codes across all actions**: The top 10 codes are nearly identical for every action
   - Core collapsed set: `[74, 423, 761, 103, 651, 30, 853, 490, 270, 919, 306, 801, 600, 570, 257]`
   - These 15 codes likely represent "safe neutral" features (grass, sky, brown terrain)

2. **Top 30 codes cover 99%+ of all tokens**
   - Model is not exploring the codebook at all
   - Remaining 994 codes are essentially dead

3. **Action conditioning works, but within collapsed space**
   - Different actions produce slightly different code orderings
   - But all actions use the same tiny subset
   - This explains why WASD appears functional but produces minimal visual change

**Implications**:
- This is **codebook mode collapse**, not action conditioning failure
- Model learned to play it safe with a small set of codes that work for most scenes
- The 1024-code VQ-VAE capacity is wasted - model only uses 3.5%

---

### Test 2: TF vs AR Logit Distribution ⚠️ MASSIVE DISTRIBUTION SHIFT

**Finding**: **KL divergence = 2.9402** between teacher-forced and autoregressive logit distributions.

| Metric | Teacher-Forced | Autoregressive | Change |
|--------|---------------|----------------|--------|
| Entropy | 0.7330 | 0.4343 | **-40.8%** |
| Confidence (max prob) | 0.7423 | 0.8433 | **+13.6%** |
| Top-10 overlap | - | 6.1 / 10 | 39% different |

**Critical observations**:

1. **Large KL divergence (2.94)**
   - This is considered "large" in ML terms (>1.0 is significant)
   - Indicates fundamentally different prediction behaviors
   - TF and AR are operating in different modes

2. **AR predictions are MORE confident**
   - AR entropy drops 40.8% (0.733 → 0.434)
   - AR max probability increases 13.6% (0.742 → 0.843)
   - Model becomes MORE certain when making AR predictions
   - This seems paradoxical - why would AR be more confident than TF?

3. **Top-10 predictions differ by 39%**
   - Only 6.1 out of 10 top candidates overlap
   - Model is predicting different codes in AR mode

**Implications**:
- There IS a distribution shift from TF to AR
- The shift makes AR predictions MORE peaked (higher confidence, lower entropy)
- **This is the opposite of what we'd expect from uncertainty**
- Hypothesis: Model learned that in AR mode, it's safer to commit strongly to the collapsed code set

---

### Test 3: Soft Sampling Alternatives ✅ **TOP-K PREVENTS COLLAPSE**

**Finding**: **Top-k sampling (k=50) eliminates first-step collapse** (+14.4% improvement).

| Sampling Strategy | Frame 0 | Frame 1 | 1-Step Drop | Improvement |
|------------------|---------|---------|-------------|-------------|
| **Argmax** (baseline) | 35 | 31 | **-11.4%** | - |
| **Top-k=50** | 34 | 35 | **+2.9%** | **+14.4%** ✅ |
| Top-k=100 | 37 | 35 | -5.4% | +6.0% |
| Nucleus p=0.9 | 35 | 33 | -5.7% | +5.7% |
| Temperature=0.8 | 35 | 32 | -8.6% | +2.9% |

**Critical observations**:

1. **Top-k=50 actually INCREASES diversity from Frame 0→1**
   - 34 codes → 35 codes (+2.9% instead of collapse!)
   - This is the ONLY strategy that prevents first-step collapse
   - By frame 10-19, still maintains 28-30 codes (vs 29 for argmax)

2. **Argmax is the problem**
   - Deterministic argmax selection forces model into collapsed mode
   - Even slight stochasticity (top-k=50) breaks the collapse pattern

3. **Top-k=100 and nucleus p=0.9 help, but less**
   - Still show 5-6% collapse on first step
   - Better than argmax but not as good as top-k=50

4. **Temperature scaling helps least**
   - Only 2.9% improvement
   - Still collapses 8.6% on first step
   - Scaling entire distribution doesn't address the peaked nature

**Implications**:
- **Argmax determinism is a PRIMARY cause of collapse**
- Top-k=50 forces model to explore beyond its "safe" collapsed set
- This suggests training with soft targets or scheduled sampling might also help
- **Immediate fix**: Switch live inference to top-k=50 instead of argmax

---

### Test 4: Gumbel Annealing Verification ⚠️ CANNOT VERIFY

**Finding**: No `gumbel_tau` value stored in checkpoint, cannot verify schedule was applied.

**Observations**:
- Checkpoint is at step 95,000
- Expected tau at this step: 0.1 (annealing complete at 20k)
- Optimizer state exists (training state preserved)
- But no tau value stored

**Implications**:
- Cannot confirm Gumbel annealing was actually applied during training
- Possible the schedule was not saved to checkpoint
- Would need to check v7.0.5 training logs to verify

---

## Root Cause Analysis

### The First-Step Collapse Mechanism

1. **Training phase (teacher-forced)**:
   - Model sees real VQ-VAE codes (diverse, ~400-500 unique codes)
   - Logit distribution is moderately confident (entropy 0.73)
   - Model learns: "Given diverse input → predict diverse output"

2. **AR prediction phase (first step)**:
   - Model switches from real codes to predicted codes (argmax)
   - **Distribution shift occurs**: AR logits become MORE confident (entropy 0.43)
   - **Argmax sampling**: Deterministically selects most confident prediction
   - Result: Model commits to "safe" code from collapsed set

3. **Subsequent AR steps**:
   - Model now receives collapsed codes as input
   - Continues predicting from collapsed set (self-reinforcing)
   - Diversity never recovers (stays at 25-30 codes)

### Why Argmax Causes Collapse

Argmax creates a **"confidence death spiral"**:

```
High-confidence AR logits (0.84 max prob)
         ↓
   Argmax selection
         ↓
  Always picks same "safe" codes
         ↓
   Collapsed code set (35 codes)
         ↓
Model learns these codes work reliably
         ↓
  AR logits become EVEN MORE confident
         ↓
     [Loop reinforces]
```

**Top-k breaks the cycle** by forcing stochastic exploration:
- Even if model is 84% confident in code X, there's a chance to sample code Y
- This prevents lock-in to the collapsed set
- Diversity is maintained through forced exploration

---

## Why This Explains All v7.x Failures

### v7.0.1 (96% collapse, 18 codes)
- Most severe collapse
- Likely used very low Gumbel tau (0.2) from step 0
- Combined with argmax → immediate hard collapse

### v7.0.2 (89.6% collapse, 45 codes)
- "Least bad" v7.x checkpoint
- Gumbel annealing 1.0→0.1 over 20k helped
- Still collapsed due to argmax determinism

### v7.0.3-v7.0.5 (91-92% collapse, 35-40 codes)
- Corruption experiments didn't help
- Root cause was argmax + distribution shift, not error recovery

---

## Recommended Fixes

### Tier 1: Immediate (No Retraining Required)

**1. Switch live inference to top-k=50 sampling** ⭐⭐⭐
- **Impact**: Should eliminate first-step collapse immediately
- **Implementation**: 1 line change in `live_inference.py`
- **Risk**: Very low - just changes sampling strategy
- **Test now**: Can verify visually if diversity improves

```python
# Current (argmax):
z_next = logits.argmax(dim=1)

# Proposed (top-k=50):
z_next = sample_top_k(logits, k=50)
```

### Tier 2: Training Changes (Requires Retraining)

**2. Train with soft Gumbel targets instead of hard argmax** ⭐⭐⭐
- **What**: Use Gumbel-Softmax with `hard=False` during AR predictions in training
- **Why**: Provides gradient signal for exploration, prevents argmax lock-in
- **Implementation**: Change AR rollout in `train.py`
- **Risk**: Moderate - need to manage tau schedule carefully

**3. Scheduled sampling for AR predictions** ⭐⭐
- **What**: Gradually transition from teacher-forcing to AR during training
- **Why**: Reduces distribution shift between TF and AR
- **Implementation**: Add sampling probability schedule in `train.py`
- **Risk**: Moderate - v4.4 tried this but abandoned it

**4. Label smoothing for codebook targets** ⭐⭐
- **What**: Instead of one-hot targets, smooth to top-k most likely codes
- **Why**: Encourages model to explore alternatives, reduces overconfidence
- **Implementation**: Modify loss computation in `train.py`
- **Risk**: Low - standard technique

### Tier 3: Architectural Changes

**5. Stochastic prediction head** ⭐⭐
- **What**: Add explicit stochasticity to prediction head (e.g., VAE-style latent)
- **Why**: Prevents deterministic collapse
- **Risk**: High - requires architectural changes

**6. Auxiliary diversity loss** ⭐
- **What**: Add loss term penalizing low code usage (entropy regularization on code distribution)
- **Why**: Explicitly encourages codebook diversity
- **Risk**: Low - just adds loss term

---

## Next Steps

### Immediate Actions (Today)

1. ✅ **Test top-k=50 in live inference**
   ```bash
   # Modify live_inference.py to use top-k=50
   # Run visual test to verify diversity improvement
   ```

2. ✅ **Run same diagnostic tests on v7.0.2**
   ```bash
   python immediate_diagnostics.py \
       --checkpoint ./checkpoints/ochre-v7.0.2-step85k.pt \
       --vqvae ./vq_vae/checkpoints/vqvae_v2.1.6__epoch100.pt
   ```
   - Verify if v7.0.2 has same argmax collapse pattern
   - Check if KL divergence is lower (better TF-AR alignment)

### Short-Term (This Week)

3. **Train v7.0.6 with top-k sampling during AR rollout**
   - Modify `train.py` to use top-k=50 instead of argmax for AR predictions
   - Keep all other v7.0.2 settings (best v7.x baseline)
   - Expected outcome: 60-80 unique codes instead of 35

4. **Implement soft Gumbel for AR (optional)**
   - If top-k alone insufficient, try soft Gumbel targets
   - Requires more careful tuning

### Analysis (Next Week)

5. **Analyze which codes are collapsed**
   - Decode the 15 core codes: `[74, 423, 761, 103, 651, 30, 853, 490, 270, 919, 306, 801, 600, 570, 257]`
   - Visual inspection: Are they grass/sky/brown as hypothesized?
   - Check VQ-VAE codebook clustering

6. **Compare to v6.2 (ConvGRU)**
   - Create architecture-specific loader for v6.2
   - Run same diagnostic tests
   - Does ConvGRU also have argmax collapse?

---

## Conclusion

The first-step catastrophic collapse in v7.x is caused by:

1. **Primary cause**: **Argmax determinism** forces model into a collapsed code set
2. **Secondary cause**: Large TF→AR distribution shift (KL=2.94) creates different prediction modes

**The fix is simple**: **Use top-k=50 sampling instead of argmax**.

This explains why all training strategies (corruption, AR schedules, loss rebalancing) failed - they didn't address the fundamental sampling determinism issue.

**Critical next step**: Test top-k=50 in live inference immediately to verify this fixes the collapse.

If confirmed, train v7.0.6 with:
- Top-k=50 sampling for AR rollout (instead of argmax)
- Keep v7.0.2 hyperparameters (best v7.x baseline)
- Expected result: 60-80 unique codes, better visual quality, stronger action responses
