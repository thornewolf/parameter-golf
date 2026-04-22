"""
train_gpt_forked.py — fork-and-merge training with a linearized branch.

Multi-GPU via DDP data-parallelism within each phase.

Training runs in repeating cycles of length 3N:

  Phase A (normal):     Train one shared model θ for N steps (DDP).
  Phase B (branch-1):   Snapshot θ₀ = θ. Continue training normally for N more
                        steps → θ₁ (DDP).
  Phase C (branch-2):   Reset model to θ₀. For N steps, compute gradients of the
                        second-order Taylor surrogate of the loss around frozen θ₁:
                            L̃₂(θ₂) = L(θ₁) + ⟨g₁, Δ⟩ + ½⟨Δ, H₁ Δ⟩
                        whose gradient ∇_{θ₂} L̃₂ = g₁ + H₁·Δ. The H₁·Δ term is
                        the mixed-Hessian-vector product, computed via
                        torch.func.jvp of the gradient function (forward-over-
                        reverse autodiff). Branch-2 sees different batches than
                        branch-1. Gradients are all-reduced manually across
                        ranks since DDP's backward hooks don't fire for jvp-
                        computed grads. → θ₂.
  Phase D (merge):      θ ← (θ₁ + θ₂) / 2 on every rank. Broadcast from rank 0
                        as a safety belt.

Multi-GPU notes:
  * Data parallelism only — both branches run on all ranks, one at a time.
    A true dual-branch split (half the ranks each) would roughly halve phase
    wall-clock but requires a cross-group merge; deliberately out of scope.
  * Phase-C grads are accumulated locally across micro-steps, then all-reduced
    once per optimizer step. This is the same amortization DDP does for real
    grads, just done by hand.
  * torch.compile is disabled (jvp+functional_call+compile is a minefield).

Caveats, stated plainly:
  * This is a research experiment. The second-order Taylor approximation
    degrades as ‖θ₂ − θ₁‖ grows; watch the drift diagnostic.
  * Branch-2 uses plain Adam everywhere. Muon's Newton-Schulz orthogonalization
    is a nonlinear function of the gradient, which breaks the "we're following
    the surrogate's gradient" assumption the trick relies on.
"""

from __future__ import annotations

import copy
import functools
import math
import os
import random as _random
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

# Reuse the heavy lifting from the reference script.
from train_gpt import (  # type: ignore
    CONTROL_TENSOR_NAME_PATTERNS,
    CastedLinear,
    DistributedTokenLoader,
    GPT,
    Hyperparameters as BaseHyperparameters,
    Muon,
    build_doc_block_mask,
    build_sentencepiece_luts,
    eval_val,
    get_cuda_device,
    get_distributed_config,
    load_tokenizer,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


# -----------------------------
# HYPERPARAMETERS (extends the base set)
# -----------------------------


class Hyperparameters(BaseHyperparameters):
    # Length of each phase (A, B, C are all this many steps; D is instantaneous).
    fork_phase_steps = int(os.environ.get("FORK_PHASE_STEPS", 200))
    # Turn the whole fork-merge thing off for A/B comparison against a plain run.
    fork_enabled = os.environ.get("FORK_ENABLED", "1") == "1"
    # How often (in branch-2 steps) to log the linearization drift diagnostic.
    fork_diag_every = int(os.environ.get("FORK_DIAG_EVERY", 50))
    # Branch-2 uses plain Adam; keep LR conservative since linearized gradients
    # can be biased as θ₂ drifts from θ₁.
    fork_branch2_lr = float(os.environ.get("FORK_BRANCH2_LR", 0.01))


# -----------------------------
# PARAM DICT UTILITIES
# -----------------------------


def params_as_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def buffers_as_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: b for name, b in model.named_buffers()}


def clone_param_dict(pd: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in pd.items()}


def load_param_dict_into(model: nn.Module, pd: dict[str, Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in pd:
                p.data.copy_(pd[name].to(device=p.device, dtype=p.dtype))


def average_param_dicts(
    a: dict[str, Tensor], b: dict[str, Tensor]
) -> dict[str, Tensor]:
    return {k: 0.5 * (a[k] + b[k]) for k in a}


def reset_rope_caches(model: nn.Module) -> None:
    # `eval_val` runs under torch.inference_mode(), which means any Rotary cos/sin
    # tables built *during* that call are stored as inference tensors and can't
    # later be saved for backward. If we call eval_val before any training has
    # warmed the cache, the first training forward blows up with
    # "Inference tensors cannot be saved for backward". Invalidate any stored
    # tables so the next Rotary.forward rebuilds them in a normal autograd
    # context. Duck-typed so we don't have to import the Rotary class.
    for m in model.modules():
        if hasattr(m, "_cos_cached") and hasattr(m, "_sin_cached"):
            m._cos_cached = None
            m._sin_cached = None
            m._seq_len_cached = 0


def broadcast_params_from_rank0(model: nn.Module) -> None:
    """After the merge, make sure every rank has bit-identical params.

    They *should* already match if rank 0 did the same arithmetic on the same
    inputs as every other rank, but bf16 associativity and cross-rank bit-drift
    over long runs make this a cheap safety belt.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        if b.is_floating_point():
            dist.broadcast(b.data, src=0)


# -----------------------------
# LINEARIZED-LOSS GRADIENT
# -----------------------------


def _loss_of_params(
    model: nn.Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    x: Tensor,
    y: Tensor,
    block_mask,
) -> Tensor:
    # Functional view: loss as a pure function of the parameter dict.
    #
    # CastedLinear keeps its weight in fp32 and casts to x.dtype at matmul
    # time. Real training uses torch.autocast to unify dtypes to bf16; but
    # autocast's saved-tensor hooks don't compose reliably with functorch,
    # so the backward under grad() ends up with bf16/fp32 mismatches.
    # Instead, cast every param to bf16 inline before functional_call.
    # jvp differentiates through .to(bf16) cleanly — the gradient returned
    # by grad(loss_fn) will be in the caller's original param dtypes.
    #
    # SDPBackend.MATH is also required: flash / mem-efficient SDPA kernels
    # don't implement forward-mode AD, which jvp needs.
    params_bf16 = {k: v.to(torch.bfloat16) for k, v in params.items()}
    with sdpa_kernel(SDPBackend.MATH):
        return functional_call(
            model, {**params_bf16, **buffers}, args=(x, y, block_mask)
        )


def linearized_loss_and_grad(
    model: nn.Module,
    theta1: dict[str, Tensor],
    theta2: dict[str, Tensor],
    buffers: dict[str, Tensor],
    x: Tensor,
    y: Tensor,
    block_mask,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Second-order Taylor surrogate of the loss around frozen θ₁:

        L̃₂(θ₂) = L(θ₁) + ⟨g₁, Δ⟩ + ½⟨Δ, H₁ Δ⟩    where Δ = θ₂ − θ₁

    Its gradient is:

        ∇_{θ₂} L̃₂ = g₁ + H₁ Δ

    The H₁·Δ term is the mixed-Hessian-vector product — the thing that makes
    branch-2's trajectory actually differ from a naive SGD on θ₁'s loss.

    We compute g₁ + H₁·Δ in one shot using torch.func.jvp applied to the
    functorch grad of the loss: jvp(grad(loss_fn), (θ₁,), (Δ,)) returns
    (grad(loss_fn)(θ₁), ∂_ε grad(loss_fn)(θ₁+εΔ)|_{ε=0}) = (g₁, H₁·Δ). That's
    forward-over-reverse autodiff — the standard efficient HVP recipe, and the
    functorch form is required (plain torch.autograd.grad with
    requires_grad_() is not allowed inside a jvp transform).
    """
    delta = {name: theta2[name] - theta1[name] for name in theta1}

    def loss_fn(params: dict[str, Tensor]) -> Tensor:
        return _loss_of_params(model, params, buffers, x, y, block_mask)

    grad_of_loss = grad(loss_fn)
    g1, Hdelta = jvp(grad_of_loss, (theta1,), (delta,))
    out_grad = {name: g1[name] + Hdelta[name] for name in g1}

    # Also compute the surrogate value itself for logging.
    with torch.no_grad():
        loss_at_theta1 = _loss_of_params(model, theta1, buffers, x, y, block_mask)
        linear_term = sum((g1[n] * delta[n]).sum() for n in g1)
        quadratic_term = 0.5 * sum((Hdelta[n] * delta[n]).sum() for n in Hdelta)
        surrogate_loss = loss_at_theta1 + linear_term + quadratic_term

    return surrogate_loss.detach(), out_grad


# -----------------------------
# OPTIMIZER SETUP (reused for branches)
# -----------------------------


def make_optimizers_standard(
    model: nn.Module, args: Hyperparameters
) -> list[torch.optim.Optimizer]:
    """Recreate the same optimizer split the base script uses."""
    block_named = list(model.blocks.named_parameters())
    matrix_params = [
        p for n, p in block_named
        if p.ndim == 2
        and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for n, p in block_named
        if p.ndim < 2
        or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opts: list[torch.optim.Optimizer] = [
        torch.optim.Adam(
            [{"params": [model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        ),
        Muon(
            matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            momentum_warmup_start=args.muon_momentum_warmup_start,
            momentum_warmup_steps=args.muon_momentum_warmup_steps,
        ),
        torch.optim.Adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        ),
    ]
    for group in opts[1].param_groups:
        group["base_lr"] = args.matrix_lr
    if model.lm_head is not None:
        opts.insert(1, torch.optim.Adam(
            [{"params": [model.lm_head.weight], "lr": args.head_lr,
              "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        ))
    return opts


def make_optimizer_branch2_adam(
    model: nn.Module, args: Hyperparameters
) -> torch.optim.Optimizer:
    """Branch-2 under linearized updates: plain Adam on everything.

    Muon's orthogonalization is a nonlinear function of the gradient, which
    breaks the assumption that we're following the Taylor surrogate's gradient.
    Plain Adam keeps the semantics honest.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(
        [{"params": params, "lr": args.fork_branch2_lr,
          "base_lr": args.fork_branch2_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )


def allreduce_grads(model: nn.Module, world_size: int) -> None:
    """Manual DDP-style all-reduce of .grad over ranks, for phase C.

    Phase A/B use DDP which hooks into backward; phase C computes grads via
    torch.func.jvp which sidesteps those hooks, so we all-reduce by hand. We
    average rather than sum to match DDP's default.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    scale = 1.0 / world_size
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.mul_(scale)


# -----------------------------
# MAIN
# -----------------------------


def main() -> None:
    args = Hyperparameters()

    distributed, rank, world_size, local_rank, grad_accum_steps = (
        get_distributed_config()
    )
    device = get_cuda_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    # Reproducibility. Same seed across ranks for model init (DDP requires it);
    # we only vary seeds for things that should differ per rank, which in this
    # script is nothing — DistributedTokenLoader already shards the stream.
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = load_tokenizer(args.tokenizer_path, args.vocab_size)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}_forked.txt"
        print(logfile)

    def log(msg: str) -> None:
        if not master:
            return
        print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log(f"fork_enabled:{args.fork_enabled} fork_phase_steps:{args.fork_phase_steps}")
    log(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    # -------- Build the model. Single instance; we snapshot/restore weights
    # to implement the phases rather than holding two live modules. --------
    # No torch.compile: functional_call + jvp through a compiled module is
    # a minefield we don't need to fight today.
    base_model = GPT(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)
    base_model.doc_attn_gate = args.doc_attn_gate
    base_model.register_buffer(
        "is_boundary_token_lut", is_boundary_token_lut, persistent=False
    )

    # Wrap in DDP for the real-training phases. We intentionally use the bare
    # module (via base_model) for the functional-call path in phase C.
    if distributed:
        ddp_model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        ddp_model = base_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"model_params:{n_params}")

    optimizers = make_optimizers_standard(base_model, args)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    # Branch-2 loader: same shards, same rank/world_size split, but we advance
    # the underlying stream to a different offset so branch-2 never sees the
    # same token window as branch-1 during phase C. Every rank advances the
    # same amount so the per-rank slicing still produces disjoint batches
    # across ranks — just from a different region of the corpus.
    branch2_loader = DistributedTokenLoader(
        args.train_files, rank, world_size, device
    )
    BRANCH2_OFFSET_TOKENS = 1 << 20  # ~1M tokens
    skipped = 0
    while skipped < BRANCH2_OFFSET_TOKENS:
        avail = branch2_loader.stream.tokens.numel() - branch2_loader.stream.pos
        take = min(avail, BRANCH2_OFFSET_TOKENS - skipped)
        branch2_loader.stream.pos += take
        skipped += take
        if branch2_loader.stream.pos >= branch2_loader.stream.tokens.numel():
            branch2_loader.stream._advance_file()

    run_eval_val = functools.partial(
        eval_val, args, ddp_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )

    # -------- Real forward/backward training step (phase A, B). --------
    def train_step_real(loader: DistributedTokenLoader) -> Tensor:
        grad_scale = 1.0 / grad_accum_steps
        total_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            if distributed:
                ddp_model.require_backward_grad_sync = (
                    micro == grad_accum_steps - 1
                )
            x, y = loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            block_mask = build_doc_block_mask(
                is_boundary_token_lut, x, args.doc_attn_gate
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = ddp_model(x, y, block_mask)
            total_loss += loss.detach()
            (loss * grad_scale).backward()
        return total_loss / grad_accum_steps

    # -------- Linearized step (phase C). --------
    def train_step_linearized(
        theta1: dict[str, Tensor],
        loader: DistributedTokenLoader,
    ) -> tuple[Tensor, float]:
        grad_scale = 1.0 / grad_accum_steps
        total_loss = torch.zeros((), device=device)
        for p in base_model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        current_theta2 = params_as_dict(base_model)
        current_buffers = buffers_as_dict(base_model)

        for _ in range(grad_accum_steps):
            x, y = loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            # flex_attention runs through torch._dynamo and trips internal
            # meta-tensor assertions when composed with torch.func.jvp. Force
            # the block_mask=None branch so the attention module uses plain
            # SDPA (which composes fine with functorch). The semantic cost is
            # that phase C loses the document-boundary attention mask — ok for
            # this experiment, since the fork-merge behavior is the point.
            block_mask = None
            surrogate_loss, surrogate_grad = linearized_loss_and_grad(
                base_model, theta1, current_theta2, current_buffers,
                x, y, block_mask,
            )
            total_loss += surrogate_loss

            for name, p in base_model.named_parameters():
                g = surrogate_grad[name].to(dtype=p.dtype)
                if p.grad is None:
                    p.grad = g.detach() * grad_scale
                else:
                    p.grad.add_(g.detach(), alpha=grad_scale)

        # All-reduce across ranks — DDP hooks didn't fire for jvp-computed grads.
        allreduce_grads(base_model, world_size)

        # Diagnostic: how far has θ₂ drifted from θ₁? Compute in fp64 for
        # stability, then combine across ranks so the master log reflects the
        # globally agreed drift (which should equal the local drift, but log
        # what's actually true).
        drift_sq = torch.zeros((), device=device, dtype=torch.float64)
        for name in theta1:
            drift_sq += (current_theta2[name] - theta1[name]).pow(2).sum().to(
                dtype=torch.float64
            )
        if distributed:
            # All ranks should have identical params, so this is a no-op check;
            # still cheap and catches silent divergence.
            dist.all_reduce(drift_sq, op=dist.ReduceOp.MAX)
        drift = math.sqrt(drift_sq.item())
        return total_loss / grad_accum_steps, drift

    # -------- Phase drivers. --------
    def run_phase_real(
        n_steps: int, loader: DistributedTokenLoader,
        opts: list[torch.optim.Optimizer], tag: str,
    ) -> None:
        base_model.train()
        for s in range(n_steps):
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss = train_step_real(loader)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), args.grad_clip_norm
                )
            for opt in opts:
                opt.step()
            if (s + 1) % max(args.train_log_every, 1) == 0 or s == 0:
                log(f"[{tag}] step:{s + 1}/{n_steps} loss:{loss.item():.4f}")

    def run_phase_linearized(
        n_steps: int, theta1: dict[str, Tensor],
        loader: DistributedTokenLoader,
    ) -> None:
        opt = make_optimizer_branch2_adam(base_model, args)
        base_model.train()
        for s in range(n_steps):
            opt.zero_grad(set_to_none=True)
            surrogate_loss, drift = train_step_linearized(theta1, loader)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), args.grad_clip_norm
                )
            opt.step()
            if ((s + 1) % max(args.fork_diag_every, 1) == 0) or s == 0:
                log(
                    f"[branch2-lin] step:{s + 1}/{n_steps} "
                    f"surrogate_loss:{surrogate_loss.item():.4f} drift:{drift:.4f}"
                )

    # -------- Main loop: A, (B, C, D), A, (B, C, D), … --------
    N = args.fork_phase_steps
    steps_per_cycle = 3 * N
    num_cycles = max(args.iterations // steps_per_cycle, 1)

    log(f"cycles:{num_cycles} steps_per_cycle:{steps_per_cycle}")

    # Initial val
    val_loss, val_bpb = run_eval_val()
    log(f"step:0 val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
    reset_rope_caches(base_model)

    t0 = time.perf_counter()
    global_step = 0
    for cycle in range(num_cycles):
        log(f"=== cycle {cycle + 1}/{num_cycles} ===")

        # Phase A: normal training on single model.
        run_phase_real(N, train_loader, optimizers, tag="phaseA")
        global_step += N

        if not args.fork_enabled:
            val_loss, val_bpb = run_eval_val()
            log(f"step:{global_step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
            continue

        # Snapshot θ₀ (identical on all ranks since DDP kept params in sync).
        theta0 = clone_param_dict(params_as_dict(base_model))
        opt_state_pre_fork = [copy.deepcopy(o.state_dict()) for o in optimizers]

        # Phase B: branch-1 = continue normally from θ₀ for N steps.
        run_phase_real(N, train_loader, optimizers, tag="phaseB-branch1")
        global_step += N
        theta1 = clone_param_dict(params_as_dict(base_model))

        # Reset to θ₀ for branch-2, plus restore optimizer state so the post-
        # merge optimizer (which we recreate below) starts from a clean slate.
        load_param_dict_into(base_model, theta0)
        for o, s in zip(optimizers, opt_state_pre_fork):
            o.load_state_dict(s)
        if distributed:
            dist.barrier()

        # Phase C: branch-2 trains with 2nd-order Taylor surrogate gradients.
        run_phase_linearized(N, theta1, branch2_loader)
        global_step += N
        theta2 = clone_param_dict(params_as_dict(base_model))

        # Phase D: merge.
        merged = average_param_dicts(theta1, theta2)
        load_param_dict_into(base_model, merged)
        broadcast_params_from_rank0(base_model)
        # Averaging momentum buffers from two divergent trajectories is not
        # principled. Reset optimizers post-merge.
        optimizers = make_optimizers_standard(base_model, args)

        val_loss, val_bpb = run_eval_val()
        elapsed = time.perf_counter() - t0
        log(
            f"[merged] step:{global_step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
            f"elapsed:{elapsed:.1f}s"
        )

    # Final eval
    val_loss, val_bpb = run_eval_val()
    log(f"final step:{global_step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
