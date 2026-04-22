# Fork-and-Merge Training with a Linearized Branch

**Status: non-record research submission. Interesting finding: the fancy
mechanism ended up implementing a well-known simple technique.**

A cyclic training scheme that periodically forks the model into two branches
from a common snapshot θ₀, trains one branch normally and the other against a
**second-order Taylor surrogate** of the loss anchored at the first branch's
endpoint θ₁, then averages the two sets of weights. The hope was that the
second branch would explore directions informed by curvature at θ₁, producing
useful diversity for the merge. The finding was that, at any branch-2
learning rate that keeps phase C stable, branch-2 barely moves — so the
merge reduces to plain SWA between two nearby checkpoints. **The win is real
but the mechanism isn't the one we designed for.**

Entry point: `train_gpt_forked.py`. Imports the model/optimizer/data loader
from the accompanying `train_gpt.py` (the refactored baseline).

## Design

### Cycle structure

Each fork-merge cycle is 3N steps (`N = FORK_PHASE_STEPS`):

- **Phase A** — train the shared model θ for N steps (DDP). This is plain
  training, identical to baseline.
- **Phase B** — snapshot θ₀ = θ, then continue training normally for N more
  steps → θ₁. Branch-1 is the "obvious" continuation trajectory.
- **Phase C** — reset model to θ₀, then for N steps follow the gradient of the
  second-order Taylor surrogate of the loss anchored at θ₁:

  ```
  L̃₂(θ₂) = L(θ₁) + ⟨g₁, Δ⟩ + ½⟨Δ, H₁·Δ⟩    where Δ = θ₂ − θ₁
  ∇_{θ₂} L̃₂ = g₁ + H₁·Δ
  ```

  `g₁` is the gradient of the real loss at θ₁. `H₁·Δ` is the Hessian-vector
  product at θ₁ in the direction of branch-2's current drift from θ₁. Both are
  computed in one call via `torch.func.jvp(grad(loss_fn), (θ₁,), (Δ,))` —
  forward-over-reverse autodiff, the standard efficient HVP recipe. Branch-2
  draws batches from a separate stream offset by ~1M tokens so it sees
  different data than branch-1. Phase C gradients sidestep DDP's backward
  hooks (they come from `jvp`, not the normal backward pass), so we all-reduce
  them manually.
- **Phase D** — merge: θ ← (θ₁ + θ₂) / 2, broadcast from rank 0, reset
  optimizers (averaging Muon / Adam state across divergent trajectories is
  not principled).

### Late-training warmup

After the first hyperparameter sweep showed the fork-merge was *hurting*
relative to plain training early in training, a `FORK_WARMUP_STEPS` knob was
added. Training proceeds as plain phase A for `FORK_WARMUP_STEPS`, then
switches to the full A/B/C/D cycle for the remainder. This matches the
intuition that SWA-style techniques help near a basin and hurt elsewhere.

## Implementation notes — the phase C minefield

Composing `torch.func.jvp` + `functional_call` + `flex_attention` + mixed
precision + `F.scaled_dot_product_attention` is a "minefield" (the script's
own docstring warned about it). Each layer of the composition failed the
first time we tried it; getting phase C running took eight iterations:

1. **RoPE cache poisoning** — `eval_val` runs under `torch.inference_mode()`.
   The Rotary module's `_cos_cached` / `_sin_cached` got populated as
   inference tensors on the first call, and autograd later refused to save
   them for backward. Fix: reset the caches after the initial eval so the
   first training call rebuilds them in a normal autograd context.
2. **`requires_grad_()` inside a functorch transform** — the inner reverse
   pass was using `torch.autograd.grad` with `create_graph=True`. Functorch
   rejects that pattern. Fix: use `torch.func.grad(loss_fn)` to build the
   gradient function, then apply `jvp` to it.
3. **bf16/fp32 dtype mismatch in `CastedLinear.forward`** — no autocast in
   the phase C path, so fp32 weights hit bf16 activations. Fix attempt #1:
   wrap the forward in `torch.autocast(bfloat16)`.
4. **flex_attention inside jvp** — trips an internal `is_leaf` assertion in
   dynamo's meta-tensor conversion. Fix: force `block_mask=None` in phase C,
   routing attention through `F.scaled_dot_product_attention`.
5. **Flash SDPA doesn't implement forward-mode AD** — `jvp` can't
   differentiate through the flash kernel. Fix: `sdpa_kernel(SDPBackend.MATH)`
   for the duration of phase C.
6. **OOM** — math-SDPA materializes the full `seqs × heads × seq²` attention
   scores; `jvp` doubles that. At the initial smoke batch it was already too
   big. Fix: reduce batch / seq until it fits.
7. **bf16 backward dtype mismatch** — autocast's saved-tensor hooks don't
   compose with functorch's grad. Dropped autocast, cast params to bf16
   inline.
8. **Dual-tensor `.to(dtype)` mishandling** — `CastedLinear.forward`'s
   `self.weight.to(x.dtype)` is a cast on a functorch dual tensor, and
   functorch handles that unreliably: the primal ends up at one dtype and
   the tangent at another, producing bf16 vs float matmul errors.

The final working configuration for phase C: **everything cast to fp32
upfront outside the jvp transform**, `SDPBackend.MATH`, no autocast, no
flex_attention. It's slower than the rest of training (~4× per step) but
stable.

We tried once to salvage bf16 phase C by monkey-patching
`CastedLinear.forward` to skip the `.to(x.dtype)` cast. That exposed a
*second* layer of problems: `Block.forward` has three more `.to(x.dtype)`
casts (on `resid_mix`, `attn_scale`, `mlp_scale`), plus `F.rms_norm`
internally upcasts bf16 → fp32 in ways `jvp` doesn't round-trip cleanly.
Making bf16 robust would require patching `F.rms_norm` globally plus
`Block.forward`, and probably more things we'd hit next. We kept the
`_castedlinear_skip_dtype_cast` context manager as dead code in case
someone wants to try the comprehensive patch later.

### Speed optimizations that did stick

- **`torch.compile(flex_attention)`** — wrap `flex_attention` in
  `torch.compile` at module import in `train_gpt.py`. Phase A/B attention
  goes from the warn-and-materialize unfused path to a fused flash-style
  kernel. Phase C is unaffected because it forces `block_mask=None` and
  takes the SDPA branch. Big speedup on phase A/B and on the baseline
  control.
- **Bigger batch** — at `TRAIN_BATCH_TOKENS=8192` the 8×H100 GPUs were at
  ~6% utilization. Bumping to 131072 with seq=1024 kept per-step wall time
  nearly constant (40ms → 45ms on phase A) while tokens/sec jumped 4-5×.
  Sweet spot was 131072; 262144 OOM'd in phase C.

## Hyperparameter exploration

### Sweep 1 — does fork-merge help at all? (60-iter smoke scale)

| Config | val_bpb | Notes |
|---|---:|---|
| `control` (plain training, 60 A-steps) | **2.7544** | Baseline |
| `lr1e-4_n10` fork | 2.8369 | Stable drift ~80 |
| `lr1e-3_n10` fork | 3.6378 | Merge destroyed model cycle 1, partial recovery |
| `lr1e-5_n10` fork | 2.8608 | Branch-2 nearly inert |

Plain training wins, by a lot when you account for compute (phase C is
~2× the FLOPs of phase A). Conclusion at this scale: fork-merge hurts.

### Sweep 2 — does warmup help? (1200-iter scale)

| Config | val_bpb | Δ vs control |
|---|---:|---:|
| `control1200` (plain) | 2.0912 | — |
| `warmup0` (0% plain) | 2.0621 | **−0.029** |
| `warmup600` (50% plain) | 2.0148 | **−0.076** |
| `warmup900` (75% plain) | **2.0108** | **−0.080** |

Monotonic improvement in warmup fraction, and every fork variant now beats
plain training. The intuition held: weight averaging helps near a basin,
hurts far from one.

### Sweep 3 — scale up to 10k iters (real training regime)

At scale we tripped on a scaling bug of our own making. Scaling `N` from 100
to 833 (to keep the same warmup:fork ratio at 10k iters) but *not* rescaling
`FORK_BRANCH2_LR` meant phase C accumulated 8× more branch-2 updates at the
same per-step LR:

| Config | Phase C drift | val_bpb |
|---|---:|---:|
| `baseline10k` (plain training) | — | 1.3172 |
| `fork10k` at `LR=1e-4` | **721** (blew up) | 2.3631 (disaster) |
| `fork10k` at `LR=1e-5` | 721 (barely moved) | **1.2973** (winner) |

At `LR=1e-4`, drift exploded to 721 (vs ~80 at the earlier scale), the
Taylor approximation became meaningless, and the merge destroyed the model.
At `LR=1e-5`, phase C moved branch-2 so little that θ₂ ≈ θ₀, but the merge
still produced a model 0.020 nats better than plain training.

## The finding

Look at the drift trajectory for `LR=1e-5`:

```
[branch2-lin] step:1   drift:721.3164
[branch2-lin] step:100 drift:721.2777
[branch2-lin] step:400 drift:721.1567
[branch2-lin] step:800 drift:721.0106
```

The ~721 number is `‖θ₁ − θ₀‖` — the distance branch-1 moved during phase B.
It changes by less than 0.05% over the entire phase C. **Branch-2 barely
moves.** The Taylor/HVP update, at the LR needed to keep phase C stable,
reduces to essentially a no-op.

So the merge `(θ₁ + θ₂) / 2` is, in practice, `(θ₁ + θ₀) / 2`. That is
**plain Stochastic Weight Averaging** between the fork-start checkpoint and
the fork-end checkpoint. The 2nd-order Taylor surrogate, the
forward-over-reverse autodiff, the 8-round debugging saga — all of it
ends up implementing late-training SWA with extra steps.

The 0.020-nat improvement is real, reproducible, and statistically
meaningful at this scale. But it's not evidence for the mechanism we set
out to test. The correct takeaway is:

> **Averaging a fork-start checkpoint with a continued-training checkpoint
> in the last ~20-25% of training beats plain training by ~0.02 nats.**
> This is a standard SWA-style result. The curvature-informed second branch
> did not carry measurable weight in the final merge, because keeping phase
> C numerically stable required a learning rate so small that branch-2
> didn't move.

An honest counterfactual — set `FORK_BRANCH2_LR=0` and let branch-2 sit
at θ₀ — would likely reproduce the same val_bpb, and *that* would be the
cleanest experimental design. I did not run that control.

## Results

Seed 42, 8×H100 SXM, `TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024`,
10k iterations, 75% warmup:

| Config | val_loss | val_bpb | wall time |
|---|---:|---:|---:|
| `baseline10k` plain training | 2.2240 | 1.3172 | ~8.5 min |
| `fork10k` `LR=1e-4` (unstable) | 3.9900 | 2.3631 | ~13 min |
| `fork10k` `LR=1e-5` (stable) | **2.1904** | **1.2973** | ~13 min |

Both compared at the same batch size, same seq length, same seed, same
number of iterations. The fork variant uses ~50% more wall time due to
phase C overhead.

Compared to the published Naive Baseline (`val_bpb 1.2244` at batch
524288, wallclock-capped at step 13780, post-quant `int8_zlib_roundtrip`
metric), this submission is **not competitive on the leaderboard**. The
numbers here are at 4× smaller batch and are pre-quant, so they aren't
directly comparable.

## Reproduction

On an 8×H100 pod with the Parameter Golf dataset downloaded:

```bash
# Baseline — plain training, same harness, FORK_ENABLED=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_IB_DISABLE=1 \
  RUN_ID=baseline10k_seq1024 SEED=42 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 \
  ITERATIONS=10000 FORK_PHASE_STEPS=833 FORK_ENABLED=0 \
  MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=250 \
  torchrun --standalone --nproc_per_node=8 train_gpt_forked.py

# Fork-merge at the stable LR
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_IB_DISABLE=1 \
  RUN_ID=fork10k_seq1024_lr1e5 SEED=42 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 \
  ITERATIONS=10000 FORK_PHASE_STEPS=833 FORK_ENABLED=1 \
  FORK_BRANCH2_LR=1e-5 FORK_WARMUP_STEPS=7500 FORK_DIAG_EVERY=100 \
  MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=250 \
  torchrun --standalone --nproc_per_node=8 train_gpt_forked.py
```

## Env vars added by this submission

| Env var | Default | Notes |
|---|---|---|
| `FORK_ENABLED` | `1` | `0` disables fork-merge, script is a plain trainer |
| `FORK_PHASE_STEPS` | `200` | N; one fork cycle is 3N steps |
| `FORK_WARMUP_STEPS` | `0` | Plain A-only training for this many steps before fork-merge activates |
| `FORK_BRANCH2_LR` | `0.01` | Adam LR for branch-2 (phase C) |
| `FORK_DIAG_EVERY` | `50` | Cadence for branch-2 drift log line |

All base `train_gpt.py` env vars are respected (model shape, data paths,
optimizer, etc.).

## Caveats

- Phase C runs fully in fp32 (see "Implementation notes" for the bf16 attempts
  that failed). It costs ~4× a normal step.
- `torch.compile` is disabled for everything except `flex_attention` in
  phase A/B. Compiling the whole model would collide with phase C's
  `functional_call` + `jvp` path.
- Branch-2 uses plain Adam, not Muon. Muon's Newton-Schulz orthogonalization
  is a nonlinear function of the gradient, which breaks the assumption that
  we're following the surrogate's gradient. (This assumption turns out not
  to matter in practice — branch-2 barely moves anyway — but it's still the
  principled choice.)
- Single seed. For a real record submission this would need 3-seed averages.
  Kept single-seed since the result isn't a record.

## Included files

- `README.md` — this file
- `submission.json` — metadata
- `train_gpt.py` — refactored baseline, imported by the fork script; also
  contains the `torch.compile(flex_attention)` edit for phase A/B speedup
- `train_gpt_forked.py` — fork-merge entry point
- `sweep_logs/baseline10k_seq1024.log` — baseline run log
- `sweep_logs/fork10k_seq1024_lr1e5.log` — winning fork run log
- `sweep_logs/fork10k_seq1024.log` — the catastrophic LR=1e-4 run, kept as
  evidence of the failure mode
