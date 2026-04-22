# Refactor of the Baseline train_gpt.py

Refactor of `train_gpt.py`. Default hyperparameters match the Naive Baseline
(VOCAB_SIZE=1024, 9L x 512d, 8H / 4KV, MLP 2x, tied embeddings). No ML changes
intended — the submission is about code clarity / structure.

## Results (seed 42, 8xH100 SXM)

<!-- TODO: fill in after the RunPod run completes -->

| Seed | val_loss | val_bpb | artifact_bytes |
|------|----------|---------|----------------|
| 42   |          |         |                |

Single-seed run. Not a statistically-significant record claim; intended as a
refactor/readability submission rather than a new leaderboard entry.

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=refactor_sp1024_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
