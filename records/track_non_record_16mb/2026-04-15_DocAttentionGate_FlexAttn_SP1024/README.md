# Document-Boundary Attention Gate (FlexAttention)

Non-record experiment forked from `train_gpt_apr_14.py`. The motivation: training packs many unrelated FineWeb documents back-to-back into each fixed-length sequence, but stock attention is purely causal so token *i* in a new document can still attend to the end of the previous document on the same row. This fork gates attention so each token only sees prior tokens **within its own document**.

## Approach

- **Document IDs per row.** The existing `is_boundary_token_lut` (built in `build_sentencepiece_luts` from `sp.is_control | is_unknown | is_unused`) marks the SentencePiece `<s>` token — the doc separator in these shards (one every ~1247 tokens in the val split). We derive `doc_id = is_boundary_token_lut[input_ids].cumsum(dim=1)` per row, which increments at every `<s>`.
- **FlexAttention BlockMask.** For each packed batch we build a `BlockMask` with `mask_mod = (q_idx >= kv_idx) & (doc_id[b, q_idx] == doc_id[b, kv_idx])`. This folds the causal triangle and the block-diagonal document partition into one kernel. `create_block_mask(..., _compile=True)` keeps the block construction on the Triton fast path.
- **Attention call.** `CausalSelfAttention.forward` branches: gate-off uses `F.scaled_dot_product_attention(is_causal=True, attn_mask=None)` (Flash SDPA); gate-on uses `flex_attention(q, k, v, block_mask=..., enable_gqa=...)`. Same tensor layouts, same GQA settings.
- **No tokenizer / data / loss changes.** Shards, tokenizer, targets, BPB metric, RoPE positions, optimizer — all unchanged. When `DOC_ATTN_GATE=0` the fork is bitwise-identical to the original baseline.

### Why FlexAttention

An earlier iteration passed a dense `(B, 1, S, S)` bool mask through SDPA. Flash SDPA rejects custom masks, so dispatch fell to cuDNN / mem-efficient, which cut throughput in half. FlexAttention consumes the structured `BlockMask` directly via a Triton kernel and keeps flash-level performance with the block-diagonal mask. Step-avg on 8×H100 SXM: **~43.9 ms/step** (vs ~30 ms/step for Flash-only causal — a ~47% overhead, but within the 10-minute cap at fewer total steps).

## Configuration

- Base layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TIE_EMBEDDINGS=1`
- Sequence length / batch: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Wallclock cap: `MAX_WALLCLOCK_SECONDS=600` (default)
- Gate toggle: `DOC_ATTN_GATE=1` (default on in this fork)

## Command

```bash
RUN_ID=docgate_flex \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
DOC_ATTN_GATE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The bundled `train_gpt.py` is a snapshot of `train_gpt_apr_14_docgate.py` at the time of the run.

## Results (1 seed, 8×H100 SXM, 600s cap)

| | val_loss | val_bpb |
|---|---:|---:|
| baseline (`train_gpt_apr_14.py`) | 2.07185845 | 1.22707128 |
| **doc-attn-gate (this run)** | **2.06904804** | **1.22540679** |
| delta | **−0.00281 nats** | **−0.00166 bpb** |

Modest positive signal from a single seed. Below the 0.005-nat bar for a record submission at `p<0.01`. Given FlexAttention's per-step overhead, the gated run reaches fewer total training steps inside the 600s cap than the baseline yet still produces lower val_loss — per-step the gate helps more than the raw delta suggests.

## Limitations / next steps

- **Single seed.** A 3-seed A/B (`SEED=1337,1338,1339`) is needed to measure whether this delta is noise.
- **Short sequences.** At `TRAIN_SEQ_LEN=1024` and ~1 `<s>` per 1247 tokens, only roughly half of training rows have a mid-sequence boundary to gate. Longer seq_len would amplify the effect and amortize FlexAttention's compile overhead.
- **Composable.** The gate is orthogonal to sliding-window eval, embedding tricks, and most quantization schemes; a natural next experiment is stacking it on top of a stronger stack.

## Included files

- `train_gpt.py` — snapshot of `train_gpt_apr_14_docgate.py` used for this run
- `submission.json` — leaderboard metadata
- `train.log` — (**TODO: drop in the training log from the RunPod pod here**)
