# Mixed-Policy Rollout (Teacher + Draft Lane)

Last updated: 04/17/2026.

This document explains the mixed-policy rollout implementation in verl, what it depends on, and why rollout correction (importance weighting / rejection) is strongly recommended.

## What Mixed-Policy Means in verl

Mixed-policy rollout runs two generation lanes per prompt:

- `on-policy` lane: standard actor rollout (policy being optimized).
- `mixed` lane: speculative decoding with a teacher target model and a draft model synchronized from the actor.

Per-prompt multiplicities are controlled by:

- `algorithm.mixed_policy.k_on`: number of on-policy samples.
- `algorithm.mixed_policy.k_off`: number of mixed-lane samples.
- effective `rollout.n = k_on + k_off` (normalized at startup).

## Runtime Architecture

At initialization:

1. Main actor-rollout workers are created as usual.
2. When `k_off > 0`, a second rollout worker group (`mixed_rollout_pool`) is created.
3. Mixed rollout config is built from `actor_rollout_ref` with these key overrides:
   - `model.path = teacher_model_path` (teacher target model).
   - speculative decode enabled and configured under `rollout.mtp`.
   - speculative draft model path points to the actor checkpoint path.
4. A collective group is created between actor workers and mixed rollout workers for actor->draft sync.

During training:

1. Actor is updated by PPO as usual.
2. Regular checkpoint-engine weight update refreshes the normal actor rollout lane.
3. Mixed draft sync broadcasts actor parameters and rewrites keys into the draft namespace, then loads them into the mixed lane draft model.
4. Mixed lane teacher target weights are not overwritten during draft sync.
5. On and mixed generated batches are concatenated and trained together.

## Why Rollout Correction Matters for Mixed Policy

Mixed batches are intentionally off-policy:

- on-policy samples come from (approximately) the actor rollout distribution;
- mixed samples come from a teacher+draft speculative distribution.

Without correction, those mixed samples bias PPO gradients toward the behavior policy mismatch. In practice this can produce unstable or low-quality updates when `k_off` is non-trivial.

For this reason, the mixed-policy startup normalization sets a default decoupled token-level TIS correction when `k_off > 0` and no explicit correction is configured:

- `rollout_is = "token"`
- `rollout_is_threshold = 2.0`
- `bypass_mode = false`

This computes importance weights from rollout log-probs and applies them in the actor loss path through `rollout_is_weights`, while optionally allowing additional rejection sampling if configured.

## vLLM Dependency (Modified Version)

This mixed-policy integration relies on vLLM behavior not available in plain upstream releases by default. The required capabilities are:

1. Speculative config overrides for mixed lane setup (teacher-draft mixing with entropy-aware knobs).
2. Draft-target key namespace routing when loading updated weights (draft-prefixed keys must be applied to drafter, not teacher target).
3. Support for mixed lane actor naming isolation (to avoid actor-name collisions when two lanes run in the same Ray job).
4. KV-cache clear control during incremental draft weight streaming.

In this project, these capabilities are implemented in a patched vLLM fork (https://github.com/matteosantelmo/vllm/tree/easd) under:

- `vllm/config/speculative.py`
- `vllm/engine/arg_utils.py`
- `vllm/compilation/decorators.py`
- `vllm/v1/sample/rejection_sampler.py`
- `vllm/v1/spec_decode/eagle.py`
- `vllm/v1/spec_decode/draft_model.py`
- `vllm/v1/worker/gpu_model_runner.py`

If you run mixed-policy with an unpatched vLLM build, initialization may succeed but runtime behavior (especially draft sync / mixture behavior) can silently degrade.

## Practical Checks

When validating a mixed-policy run:

1. Confirm mixed lane teacher loads from real checkpoint (not `load_format=dummy` which would result in randomly initialized weights).
2. Confirm draft sync updates only draft-prefixed weights in mixed lane.
3. Track per-source metrics (`mixed_policy/on/*`, `mixed_policy/mixed/*`) and rollout-correction diagnostics (`rollout_corr/*`).
4. Keep tokenizer mapping identical between actor and teacher (enforced by startup compatibility checks).
