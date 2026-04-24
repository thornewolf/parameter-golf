# Baseline Clone + Leaderboard Stack Ablation Plan

Status: planned research submission. No score claim yet.

This folder is intended to answer a narrow question: which repeated leaderboard
techniques actually explain the path from the clean baseline toward the current
top stack when enabled one development at a time?

## Files

- `train_gpt_baseline.py` -- byte-for-byte clone of repository root
  `train_gpt.py` at branch creation time.
- `train_gpt_common_stack.py` -- readable decompressed copy of the current top
  leaderboard trainer from `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`.
- `ablations.csv` -- planned runs, boolean feature columns, and exact env
  overrides.
- `submission.json` -- metadata placeholder for this non-record research entry.

## Techniques Mined From The Leaderboard

The recurring high-signal techniques in the top entries are:

1. Larger SentencePiece vocabularies, especially SP4096 and SP8192.
2. 11 physical transformer layers at 512d with 4x MLP.
3. Partial RoPE, layerwise norm scaling, LeakyReLU(0.5)^2 MLPs, and U-Net skips.
4. MuonEq-R / row-normalized Muon with high weight decay for compression.
5. EMA weight averaging during late training.
6. Depth recurrence over middle layers.
7. Parallel residual attention/MLP blocks in later layers.
8. Higher learned QK gain, around 5.0 to 5.25.
9. GPTQ/SDClip int6 matrices plus int8 embeddings.
10. Sliding-window evaluation and legal score-first TTT.

The top script already exposes most of these as env knobs. A few are coupled:
LeakyReLU(0.5)^2, the shuffled loader, FlashAttention 3, and the GPTQ pipeline
are structural in `train_gpt_common_stack.py`. Treat the first common-stack
control as measuring this harness jump, not as the original baseline.

## RunPod Setup

Use the repository RunPod flow, but download all tokenizer/data variants needed
by the matrix:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
git checkout codex/2026-04-24-ablation-stack

python3 data/cached_challenge_fineweb.py --variant sp1024
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest

pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

Baseline smoke:

```bash
RUN_ID=A000_baseline_sp1024_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-24_AblationStack/train_gpt_baseline.py
```

Common-stack template:

```bash
DATA_DIR=./data \
RUN_ID=A130_full_ttt_seed42 \
SEED=42 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-24_AblationStack/train_gpt_common_stack.py
```

## Execution Plan

1. Run `A000` and `A010` first. This separates the original baseline from the
   common-stack harness.
2. Run the progressive bridge rows `A020` through `A130` with seed 42.
3. Run the leave-one-out rows `L010` through `L080` with seed 42.
4. Promote only the top three candidates to seeds 42, 314, and 999.
5. For any record-like claim, compare 3-seed means and report pre-quant,
   quantized, sliding, TTT, artifact bytes, train time, and eval time.

## Planned Bridge Table

| run_id | entrypoint | vocab | shape | xsa | muon_eqr_wd | ema | recurrence | parallel | qk5.25 | gptq_sdclip | sliding | ttt |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|
| A000 | baseline | 1024 | 9L/2x | false | false | false | false | false | false | false | false | false |
| A010 | common control | 1024 | 9L/2x | false | false | false | false | false | false | false | false | false |
| A020 | common stack | 4096 | 9L/2x | false | false | false | false | false | false | false | false | false |
| A030 | common stack | 8192 | 9L/2x | false | false | false | false | false | false | false | false | false |
| A040 | common stack | 8192 | 11L/2x | false | false | false | false | false | false | false | false | false |
| A050 | common stack | 8192 | 11L/4x | false | false | false | false | false | false | false | false | false |
| A060 | common stack | 8192 | 11L/4x | true | false | false | false | false | false | false | false | false |
| A070 | common stack | 8192 | 11L/4x | true | true | false | false | false | false | false | false | false |
| A080 | common stack | 8192 | 11L/4x | true | true | true | false | false | false | false | false | false |
| A090 | common stack | 8192 | 11L/4x | true | true | true | true | false | false | false | false | false |
| A100 | common stack | 8192 | 11L/4x | true | true | true | true | true | false | false | false | false |
| A110 | common stack | 8192 | 11L/4x | true | true | true | true | true | true | false | false | false |
| A120 | common stack | 8192 | 11L/4x | true | true | true | true | true | true | true | false | false |
| A125 | common stack | 8192 | 11L/4x | true | true | true | true | true | true | true | true | false |
| A130 | common stack | 8192 | 11L/4x | true | true | true | true | true | true | true | true | true |

`ablations.csv` contains exact environment settings for these plus the
leave-one-out and interaction rows.

## Measurement Rules

- Do not use validation loss to choose a checkpoint during training.
- Keep `MAX_WALLCLOCK_SECONDS=600` for candidate runs; use shorter smoke runs
  only for syntax/performance checks and mark them as smoke.
- Prefer `VAL_LOSS_EVERY=0` for full runs to avoid paying repeated validation
  cost during training.
- Archive raw logs under this folder before writing any result table.
- Treat tokenizer changes as separate experimental phases because BPB is the
  target metric and tokenizer mistakes can dominate apparent gains.
