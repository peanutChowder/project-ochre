# Debug Tools Plan

## Goal

The current eval stack is good at answering:

- is checkpoint B better than checkpoint A?
- did a training intervention improve scalar metrics or visual artifacts?

It is much weaker at answering:

- why does the model fail under longer autoregressive rollout?
- where does the failure first enter the system?
- is the bottleneck temporal memory, self-conditioning, local token jitter, or action/memory entanglement?

This document outlines new **debug-oriented** tools whose purpose is not checkpoint ranking, but **failure-source isolation**.

The design principle is:

- every tool should test one concrete hypothesis
- outputs should make it easier to reject or support a causal explanation
- tools should be runnable on existing checkpoints so we can learn from past runs before training again

---

## 1. `self_conditioning_gap_eval.py`

### Question

How quickly does the model diverge when it must condition on its own predictions instead of GT-conditioned history?

### Motivation

This is the most direct probe for exposure bias / self-conditioning fragility.

The model may produce strong single-frame predictions and still fail under AR because:

- the first few self-generated frames move the latent state off-manifold
- later predictions are then conditioned on a degraded history
- quality and controllability both decay even if the model is locally competent

### What it should compare

For the same context and GT action sequence:

- teacher-forced rollout
- mixed rollout
- pure autoregressive rollout

At each frame, compare:

- token accuracy
- token agreement to GT
- LPIPS / PSNR
- edge metrics
- divergence between teacher-forced and AR predictions

### Why it helps

This isolates whether long-horizon failure is caused by:

- immediate transition to self-conditioning
- gradual accumulation of small errors
- or a later instability unrelated to the first AR steps

If the gap opens immediately, the model lacks recovery under self-generated context.
If the gap grows slowly, the issue is compounding drift.

### Expected value

Very high. This should be the first debug tool built.

---

## 2. `action_memory_swap_eval.py`

### Question

When the model fails, is it because action conditioning is weak, or because memory dominates the prediction incorrectly?

### Motivation

Long-horizon failure could come from bad interaction between:

- current action
- recent temporal context

The model may:

- over-trust history and under-react to action changes
- underuse history and depend almost entirely on the latest token
- or entangle the two in ways that prevent clean control

### What it should test

Run paired probes such as:

- same buffer, different action
- same action, different buffer
- swap recent buffer states across contexts
- swap only the newest or oldest buffer slots

Measure how much logits / predictions move under each manipulation.

### Why it helps

This tells us whether the model is:

- memory-dominated
- action-dominated
- or properly balancing both

That matters for deciding whether future work should focus on:

- better temporal memory
- better action disentanglement
- or better robustness to self-generated history

### Expected value

Very high.

---

## 3. `temporal_buffer_ablation_eval.py`

### Question

Is the temporal buffer actually helping, or is it ignored / harmful?

### Motivation

We have repeatedly assumed that better use of temporal history would improve rollouts.
That assumption has not been tested directly.

The model may be failing because:

- it ignores older buffer states entirely
- it over-relies on the newest state
- older states carry noise forward and destabilize rollout
- the temporal buffer is used, but not in a way that improves scene maintenance

### What it should test

Run the same rollout under controlled buffer variants:

- full buffer
- empty buffer
- last-state-only buffer
- truncated buffers of length `1`, `2`, `4`, `8`
- shuffled buffer order
- corrupted / noisy buffer states

Measure:

- transition metrics
- per-frame quality metrics
- divergence from the normal rollout
- token jitter / stability metrics

### Why it helps

This tool tells us whether the temporal-memory mechanism is:

- useful
- weak
- or actively amplifying error

If buffer ablations barely change output, more AR depth was never likely to help.
If longer buffers hurt more than shorter ones, the model is propagating noise through memory.

### Summary requirement

This tool should absorb the old `buffer_usage_summary.py` idea instead of splitting it out into a second script.

In addition to the full ablation runs, it should emit a compact summary such as:

- performance vs buffer length
- sensitivity to dropping each slot
- relative importance of newest vs oldest slots

### Expected value

Very high.

---

## 4. `rollout_recovery_eval.py`

### Question

Can the model recover after a small mistake, or do small perturbations cause permanent drift?

### Motivation

A stable long-horizon model does not need to be perfect at every step.
It needs to be able to absorb and recover from small errors.

Without testing recovery directly, we cannot tell whether the model's true weakness is:

- error accumulation
- poor local robustness
- or weak temporal correction dynamics

### What it should do

Start from a clean rollout, then inject a controlled perturbation at step `t`:

- wrong token patch (5-10% random token replacement in one frame)
- wrong full latent frame
- corrupted buffer state

Then continue rollout and measure:

- recovery time (frames until quality returns to within threshold of unperturbed trajectory)
- how much divergence spreads
- whether quality / control return toward the unperturbed trajectory

### Why it helps

This isolates error-correction ability directly.

If a tiny perturbation causes permanent drift, scheduled sampling or consistency-style training becomes much more justified.
If the model recovers quickly, the bottleneck lies elsewhere.

### Expected value

High.

---

## 5. `token_jitter_eval.py`

### Question

Is temporal instability caused by many small token flips or by rare larger scene failures?

### Motivation

Current flicker metrics are useful but too coarse.
They tell us that frames are unstable, but not whether the instability is:

- local token wobble
- edge jitter
- patch-level structural churn
- or full-scene drift

### What it should compute

Frame-to-frame, for predicted rollouts:

- number of token positions that change
- distribution of changed-token clusters
- persistent vs transient token flips
- changes restricted to GT-stable regions
- optionally compare against GT frame-to-frame token changes

Outputs should include:

- scalar summaries
- per-frame curves
- optional heatmaps for changed-token density

### Why it helps

This tool distinguishes:

- a model that is "almost stable but jittery"
- from one that is periodically rebuilding the scene incorrectly

That difference matters for training decisions:

- local jitter suggests token-level consistency or spatial masking
- large episodic failure suggests recovery / robustness interventions

### Expected value

High.

---

## Suggested Build Order

If we want maximum information quickly:

1. `self_conditioning_gap_eval.py`
2. `action_memory_swap_eval.py`
3. `temporal_buffer_ablation_eval.py`
4. `rollout_recovery_eval.py`
5. `token_jitter_eval.py`

---

## Why These Tools Matter

Recent runs have often been guided by plausible hypotheses, but not enough causal evidence.

These debug tools would let us answer questions like:

- did longer AR fail because the model cannot recover from its own mistakes?
- is the model even listening to current actions under AR, or is stale memory dominating?
- is the temporal buffer helping or hurting?
- is flicker caused by local token jitter or larger structural failures?
- are current actions being overridden by stale memory?

That would allow future runs to be based on:

- demonstrated failure mechanisms

rather than mostly educated guesses.
