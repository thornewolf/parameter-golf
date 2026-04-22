# Fork-and-Merge Training with a Linearized Branch

Experimental training loop that periodically forks the model into two branches,
trains one normally and the other against a **second-order Taylor surrogate** of
the loss around the first branch, then averages the two sets of weights.

Entry point: `train_gpt_forked.py`. The baseline `train_gpt.py` is included
unchanged — the forked script imports `GPT`, `Muon`, the data loader, etc.
from it.

## Cycle structure (length 3N, N = `FORK_PHASE_STEPS`)

- **Phase A** — train the shared model θ for N steps (DDP).
- **Phase B** — snapshot θ₀ = θ, continue normally for N steps → θ₁ (DDP).
- **Phase C** — reset model to θ₀, then for N steps follow the gradient of
  `L̃₂(θ₂) = L(θ₁) + ⟨g₁,Δ⟩ + ½⟨Δ,H₁Δ⟩` where `Δ = θ₂ − θ₁`.
  The HVP `H₁·Δ` is computed once per step via `torch.func.jvp` of the gradient
  function (forward-over-reverse). Branch-2 draws from a separate loader offset
  by ~1M tokens so it sees different batches than branch-1. DDP hooks don't fire
  for jvp-computed grads, so they are all-reduced manually. → θ₂.
- **Phase D** — merge: θ ← (θ₁ + θ₂) / 2, broadcast from rank 0, reset
  optimizers (averaging momentum across divergent trajectories is not
  principled).

## Knobs

| Env var | Default | Notes |
|---|---|---|
| `FORK_ENABLED` | `1` | Set to `0` for plain DDP A/B control. |
| `FORK_PHASE_STEPS` | `200` | N; one cycle is 3N steps. |
| `FORK_DIAG_EVERY` | `50` | Branch-2 drift-log cadence. |
| `FORK_BRANCH2_LR` | `0.01` | Plain Adam LR on branch-2. |

All base `train_gpt.py` env vars are respected (model shape, data paths, etc.).

## Caveats

- Research experiment, not a record attempt. The 2nd-order surrogate degrades as
  `‖θ₂ − θ₁‖` grows — watch the `drift` diagnostic.
- Branch-2 uses plain Adam, not Muon. Muon's Newton-Schulz orthogonalization is
  a nonlinear function of the gradient, which breaks the surrogate-following
  assumption.
- `torch.compile` is off (jvp + `functional_call` + compile is not worth fighting).
- Phase C is the expensive one — roughly double a normal step per micro-batch
  because of the extra forward pass inside the jvp. Expect the 10-min wallclock
  cap to hit far fewer tokens than baseline.

## Results (seed 42, 8xH100 SXM)

<!-- TODO: fill in after the RunPod run completes -->

| Seed | Config | val_loss | val_bpb | artifact_bytes |
|------|--------|----------|---------|----------------|
| 42   | FORK_ENABLED=1 |     |         |                |
| 42   | FORK_ENABLED=0 (control) |  |  |           |

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py` (base, imported by the forked script)
- `train_gpt_forked.py` (entry point)
- `train_seed42.log`
