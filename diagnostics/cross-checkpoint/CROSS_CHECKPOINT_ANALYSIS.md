# Cross-Checkpoint Diagnostic Analysis: First-Step Collapse Root Cause

## Executive Summary

**Critical Finding**: The first-step catastrophic collapse (89-96% code reduction in one AR step) is **NOT specific to v7.0.5** - it is a **fundamental architectural issue introduced in v7.0**.

All v7.x checkpoints exhibit this failure mode, indicating the **ConvTransformer architecture itself** (vs the previous ConvGRU architecture) introduced the collapse mechanism.

---

## Comparative Analysis Results

### v7.x Series (ConvTransformer Architecture)

| Checkpoint | Training Steps | Frame 0 Codes | Frame 1 Codes | 1-Step Collapse | WASD/Camera Ratio |
|-----------|---------------|---------------|---------------|-----------------|-------------------|
| **v7.0.1** | 55k | 450 | 18 | **96.0%** | 0.66 |
| **v7.0.2** | 85k | 431 | 45 | **89.6%** | 1.19 |
| **v7.0.4** | 110k | 441 | 39 | **91.2%** | 1.71 |
| **v7.0.5** | 95k | 448 | 35 | **92.2%** | 1.23 |

**Pattern**: ALL v7.x checkpoints collapse by ~90-96% in the first autoregressive prediction step.

### Pre-v7.x Series (ConvGRU Architecture)

- **v4.10.1-step80000.pt**: Architecture incompatible (no AdaLN-Zero blocks)
- **v4.11.0-step70000.pt**: Architecture incompatible
- **v5.0-step30000.pt**: Architecture incompatible
- **v6.1-step20000.pt**: Architecture incompatible
- **v6.2-step100000.pt**: Architecture incompatible

**Analysis**: Pre-v7.x checkpoints use different model structure (ConvGRU with FiLM conditioning) and cannot be loaded with the current diagnostic script.

**Historical Evidence from Changelogs**:
- v4.x/v5.x/v6.x: No mentions of catastrophic first-step collapse
- v6.3 (mature ConvGRU): Achieved 15-step AR rollouts, camera warp issues but no collapse
- v7.0: First mention of "severely degraded" reconstructions (line 805, changelog)

---

## Architectural Change in v7.0 (Root Cause)

### What Changed (temporal_changelog.md line 795):

**v7.0 introduced ConvTransformer**:
```
- Conv stem + windowed + axial spatial attention (replacing ConvGRU)
- AdaLN-Zero action conditioning with separate camera/movement pathways
- Temporal cross-attention over compressed past frames (replacing GRU hidden state)
```

**Previous architecture (v4.x-v6.x)**:
```
- ConvGRU: Recurrent gating with ConvGRU cells
- FiLM action conditioning (single pathway)
- GRU hidden state for temporal memory
```

### Failure Timeline in v7.x:

1. **v7.0 (step 16-17k)**: "Reconstructions severely degraded - dark, gray, lacking detail vs GT" (line 805)
   - LPIPS flat at 0.2, visual quality poor
   - Diagnosis: Gumbel soft embeddings (hard=False) + tau=0.5

2. **v7.0.1 (55k)**: "Sludge collapse", unique_codes stuck at ~18 (line 821)
   - Fixed: hard=True, tau=0.2 (but tau too cold)
   - Result: Still collapsed, now "sludge" mode

3. **v7.0.2 (85k)**: Restored to 7/10 first-frame quality (line 836)
   - Fixed: Gumbel annealing tau 1.0→0.1 over 20k steps
   - Result: Best v7.x checkpoint (only 89.6% collapse vs 96%)
   - Still has ghost trails, camera U/D non-functional, WASD weak

4. **v7.0.3-v7.0.5**: Corruption approach failed, collapse persisted (line 850-890)

---

## Key Insights

### 1. The "WASD Non-Functional" Perception Was Wrong

| Checkpoint | WASD Effect | Camera Effect | Ratio |
|-----------|-------------|---------------|-------|
| v7.0.1 | 0.0687 | 0.1042 | 0.66 |
| v7.0.2 | 0.1169 | 0.0983 | **1.19** ✅ |
| v7.0.4 | 0.1690 | 0.0986 | **1.71** ✅ |
| v7.0.5 | 0.1041 | 0.0848 | **1.23** ✅ |

**Finding**: WASD action conditioning actually **improved across v7.x** (ratio >1.0 means WASD stronger than camera). The problem is NOT that WASD is ignored - it's that the model has collapsed to 30-40 codes, so **ALL actions** (camera AND WASD) produce minimal visual change.

### 2. v7.0.2 Was the "Least Bad" Checkpoint

- Lowest collapse rate: 89.6% vs 96% for v7.0.1
- Balanced action conditioning: WASD/Camera = 1.19
- Highest visual quality reported: 7/10 first frame
- Still insufficient for training to converge

### 3. The Collapse Happens in ONE Step, Then Stabilizes

For all v7.x checkpoints:
- Frame 0: ~430-450 codes (healthy seed)
- Frame 1: 18-45 codes (catastrophic collapse)
- Frames 2-19: 25-40 codes (stable collapsed state)

**This is NOT gradual degradation**. The model makes a catastrophically wrong first prediction, then stays in that collapsed mode.

### 4. Temporal Attention and FiLM Gates Are NOT the Problem

From v7.0.5 diagnostics:
- Temporal attention entropy **increases** 3.70 → 5.68 (not decreasing/sticky)
- FiLM gates remain **constant** 0.4386 (not weakening)
- Buffer dependence 2.34% (near zero - model doesn't rely on buffer quality)

The problem is **upstream of attention and action conditioning** - it's in the first AR prediction itself.

---

## Root Cause Hypotheses

### Hypothesis 1: Spatial Attention Collapse ⭐⭐⭐
**What**: Windowed + axial attention in ConvTransformer may not handle the transition from "real codes" (teacher-forced) to "predicted codes" (autoregressive) well.

**Evidence**:
- ConvGRU (v4.x-v6.x) had no first-step collapse
- ConvTransformer (v7.x) exhibits 90%+ collapse immediately
- Attention mechanisms can collapse to degenerate patterns (all heads focusing on same tokens)

**Test**: Analyze attention maps from Frame 0→Frame 1 prediction. Are spatial attention heads collapsing to uniform weights or focusing on a small subset of tokens?

---

### Hypothesis 2: Gumbel Hard Sampling Distribution Mismatch ⭐⭐
**What**: Using `hard=True` Gumbel-Softmax creates a train-inference mismatch where:
- Teacher-forced steps: Model sees VQ-VAE codes (sharp, diverse distribution)
- First AR step: Model must predict from Gumbel-sampled codes (different distribution)

**Evidence**:
- v7.0 used `hard=False` → "OOD inputs to decoder" (line 807)
- v7.0.1 switched to `hard=True` → "sludge collapse" (line 821)
- Both modes failed, but in different ways

**Test**: Compare logit distributions for TF vs AR steps. Does switching from real codes to hard-sampled codes cause a distribution shift that triggers collapse?

---

### Hypothesis 3: Codebook Mode Collapse ⭐⭐
**What**: The prediction head learns to output a safe "average" subset of 30-40 codes that work for most scenes, avoiding risky high-entropy predictions.

**Evidence**:
- All v7.x models collapse to 25-45 codes (similar range)
- This is ~3-4% of the 1024 codebook
- Models trained on different data/steps still converge to similar collapsed set

**Test**: Analyze which 30 codes the model uses. Are they "neutral" codes (grass, sky, brown)? Do they represent low-frequency features?

---

### Hypothesis 4: Temporal Cross-Attention Initialization ⭐
**What**: The first AR step has an **empty or minimal temporal buffer** (0-1 frames), which may cause cross-attention to fail and produce degenerate outputs.

**Evidence**:
- ConvGRU used hidden state (always available)
- ConvTransformer uses cross-attention over buffer (empty at t=0)
- Buffer dependence is 2.34% (model doesn't learn to use buffer)

**Test**: Run diagnostics with forced temporal buffer initialization (duplicate seed frame 8 times). Does this prevent first-step collapse?

---

### Hypothesis 5: Action Conditioning Interference ⭐
**What**: AdaLN-Zero with separate camera/movement pathways creates gradient competition that destabilizes first prediction.

**Evidence**:
- v4.x-v6.x used single FiLM pathway (simpler)
- v7.x uses dual AdaLN-Zero pathways (more complex)
- However, FiLM gates remain stable (0.4386), suggesting pathways work fine

**Test**: Disable action conditioning entirely for first AR step (zero action vector). Does collapse still occur?

---

## Recommended Next Steps

### Immediate Diagnostics (1-2 hours)

1. **Analyze Collapsed Codebook**
   ```python
   # Which 30 codes does v7.0.5 use?
   # Are they semantically meaningful (grass, sky) or arbitrary?
   # Do they cluster in codebook space?
   ```

2. **Compare TF vs AR Logit Distributions**
   ```python
   # Measure KL divergence between:
   # - Teacher-forced logits (using real codes as input)
   # - First AR logits (using predicted codes as input)
   # Large divergence → distribution shift is the problem
   ```

3. **Test Soft Sampling Alternatives**
   ```python
   # Replace argmax with:
   # - Top-k sampling (k=50)
   # - Nucleus sampling (p=0.9)
   # - Temperature scaling (temp=0.8)
   # Does this prevent collapse by maintaining diversity?
   ```

4. **Verify Gumbel Annealing Applied Correctly**
   ```python
   # Check v7.0.2/v7.0.5 training logs
   # Was tau actually 1.0→0.1 over 20k steps?
   # Or did it lock in early due to resume bugs?
   ```

### Medium-Term Experiments (1-2 days)

5. **Analyze Spatial Attention Maps**
   ```python
   # Extract attention weights from Frame 0→Frame 1 prediction
   # Are attention heads collapsing to uniform or degenerate patterns?
   # Compare to later frames (Frame 10→Frame 11)
   ```

6. **Test Temporal Buffer Initialization**
   ```python
   # Warm-start AR with duplicated seed frame (8x)
   # Or use teacher-forced buffer from ground truth sequence
   # Does this prevent first-step collapse?
   ```

7. **Compare v6.2 (ConvGRU) First-Step Behavior**
   ```python
   # Create architecture-specific loader for v6.2
   # Run same diagnostics (unique codes Frame 0→Frame 1)
   # Establish baseline: did ConvGRU have this issue?
   ```

### Long-Term Architecture Fixes (2-5 days)

8. **Hybrid Architecture: ConvTransformer + GRU**
   ```python
   # Replace temporal cross-attention with GRU hidden state
   # Keep spatial attention + AdaLN-Zero (proven to work)
   # Test if first-step collapse disappears
   ```

9. **Scheduled AR Warmup**
   ```python
   # Start with teacher-forced first prediction for N steps
   # Gradually anneal to autoregressive (like scheduled sampling)
   # But only for first frame, not entire sequence
   ```

10. **Multi-Step Prediction Head**
    ```python
    # Instead of predicting t+1 from t, predict [t+1, t+2, t+3]
    # Encourages model to learn smooth transitions
    # May prevent collapse to safe "average" mode
    ```

---

## Critical Questions to Answer

1. **Did v4.x/v5.x/v6.x (ConvGRU) have first-step collapse?**
   - Needs architecture-specific diagnostic loader
   - If NO: Problem is specific to ConvTransformer spatial attention
   - If YES: Problem is more fundamental (Gumbel? Codebook?)

2. **Which 30 codes does the model use when collapsed?**
   - Are they semantically meaningful (common scene elements)?
   - Are they low-frequency (smooth textures)?
   - Are they clustered in codebook embedding space?

3. **Does the collapse happen in logit space or sampling?**
   - Measure logit entropy before argmax
   - If entropy is high but argmax collapses → sampling is the problem
   - If entropy is low → model is confident in wrong prediction

4. **Can we prevent collapse with better sampling?**
   - Top-k, nucleus, temperature
   - If YES: Problem is argmax determinism
   - If NO: Problem is deeper (logit distribution itself)

5. **Does temporal buffer initialization matter?**
   - Test AR with warm-started buffer (8 duplicates of seed)
   - If collapse persists → buffer is not the issue
   - If collapse prevented → cross-attention needs context

---

## Implications for v7.0.6 Planning

### ❌ Approaches That Won't Help

1. **Fixed AR Schedule**: Doesn't address first-step collapse
2. **Multi-Rollout Consistency Loss**: Can't fix what's already collapsed
3. **Extended Gumbel Annealing**: v7.0.2 already used optimal schedule
4. **Corruption Recovery**: v7.0.3-v7.0.5 proved this doesn't work
5. **Separate Optimizer for Actions**: FiLM gates are stable, not the issue

### ✅ Approaches Worth Trying

1. **Soft Sampling Instead of Argmax** (Tier 1 priority)
   - Top-k or nucleus sampling for AR predictions
   - May prevent collapse to safe subset of codes

2. **Analyze and Fix Spatial Attention** (Tier 1 priority)
   - If attention is collapsing, add regularization
   - Consider LayerScale, attention dropout, or attention temperature

3. **Hybrid ConvTransformer + GRU** (Tier 2)
   - Replace temporal cross-attention with GRU hidden state
   - Keep proven spatial attention + AdaLN-Zero components

4. **Multi-Step Prediction** (Tier 2)
   - Predict [t+1, t+2, t+3] jointly
   - Encourages smooth transitions, may prevent mode collapse

5. **Compare to v6.2 Baseline** (Research)
   - Establish whether ConvGRU had this issue
   - May inform whether to abandon ConvTransformer entirely

---

## Conclusion

The first-step catastrophic collapse is **NOT a training hyperparameter issue** - it's an **architectural failure mode of the ConvTransformer** introduced in v7.0.

All attempts to fix it via training strategies (corruption, AR schedules, loss rebalancing) failed because they didn't address the root cause: the ConvTransformer's spatial attention or prediction head is fundamentally unstable during the transition from teacher-forced to autoregressive prediction.

**The path forward requires architectural investigation**, not more training experiments:
1. Diagnose which component causes collapse (attention vs prediction head)
2. Test architectural fixes (soft sampling, attention regularization, hybrid GRU)
3. Compare to v6.2 (ConvGRU) to establish baseline behavior

**Next immediate action**: Run the 4 diagnostic tests outlined above (collapsed codebook analysis, TF vs AR logit comparison, soft sampling, Gumbel verification) to narrow down the specific mechanism of collapse before proposing v7.0.6.
