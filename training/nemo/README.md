# NeMo ASR Curriculum Training Pipeline

This directory contains code that trains a NeMo CTC-BPE speech recognition model (`EncDecCTCModelBPE`)
on a custom feature dataset, with optional **curriculum learning**: the model
periodically re-ranks the training set by its own word error rate (WER) and
trains on progressively larger, harder-ordered subsets.

## What it does, in plain terms

1. Loads a NeMo run config (a YAML/OmegaConf-style config, based on how it's used).
2. Loads a pretrained CTC-BPE model (e.g. a Conformer-CTC checkpoint from
   NVIDIA NGC, or a local `.nemo` file) **once**, and disables its built-in
   SpecAugment (`model.spec_augmentation = None`) — augmentation is expected
   to be handled separately, via the dataset's own `augmentation` config.
3. Builds a training dataset and a validation dataset from JSON/JSONL
   "schema" manifests, using `SharedASRDataset`.
4. Optionally loads a SentencePiece tokenizer from a directory.
5. If curriculum learning is **enabled** in the config:
   - Runs several "stages," each stage using a larger fraction of the training
     data than the last (default schedule: `0.2 → 0.5 → 0.7 → 1.0`).
   - At the start of each stage, it scores the *entire* training set with the
     *current* model state to rank samples by WER (`rank_samples_by_wer`).
   - It then builds a `CurriculumSampler` that restricts training to the
     `active_size` easiest (or otherwise selected — depends on
     `rank_samples_by_wer`'s ordering) samples for that stage.
   - It creates a fresh PyTorch Lightning trainer for each stage and calls
     `train_nemo(...)`, training the *same* model object further.
6. If curriculum learning is **disabled**, it just builds one training
   dataloader from the full dataset and trains once via `train_nemo(...)`.
7. Returns the experiment directory (`exp_dir`) produced by the last
   (or only) `train_nemo` call.

## Requirements

- Python environment with:
  - `torch`
  - `nemo_toolkit[asr]` (provides `nemo.collections.asr.models.EncDecCTCModelBPE`)


The main.py script auto-adds the project root to `sys.path` at import time, so you
can run it from anywhere inside the repo without manually setting
`PYTHONPATH` — it walks up two parent directories from the script's own
location (`parents[2]`) to find the root. **This means the script must live
exactly two directories below the project root** for that logic to resolve
correctly; if you move the file, update this or set `PYTHONPATH` yourself.

## Inputs you need to prepare

| Input | Required? | Notes |
|---|---|---|
| NeMo config file (`--config`) | Yes | Must define `model.tokenizer.dir` and, if you don't pass `--pretrained_model`, `model.init_from_pretrained_model`. May also define an `augmentation` block and a `curriculum` block. |
| Training schema (`--train_schema`) | Yes | JSON or JSONL manifest consumed by `SharedASRDataset`. |
| Validation schema (`--val_schema`) | Yes | Same format as training schema. |
| Tokenizer directory (`--tokenizer_dir`) | Optional | Path to a SentencePiece tokenizer directory; if omitted, `tokenizer` is `None` and whatever `SharedASRDataset`/`train_nemo` does by default applies. |
| Pretrained model (`--pretrained_model`) | Optional | Overrides `config.model.init_from_pretrained_model`. Can be an NGC model name or a path to a `.nemo` checkpoint, per NeMo's `from_pretrained` conventions. |

### Config keys the script reads directly

- `model.tokenizer.dir` — **must exist on disk**, or the script raises
  `FileNotFoundError` before doing anything else.
- `model.init_from_pretrained_model` — fallback pretrained model name/path.
- `augmentation` — passed straight through to both datasets as
  `config={"augmentation": augmentation_cfg}`.
- `curriculum.enabled` — turns curriculum mode on/off (default: off).
- `curriculum.schedule` — list of active-fraction floats per stage
  (default: `[0.2, 0.5, 0.7, 1.0]`).
- `curriculum.score_batch_size` — batch size used when re-scoring the full
  training set each stage (default: falls back to `--val_batch_size`).

## Running it

Basic run, no curriculum:

```bash
cd training/nemo;
python main.py \
  --config path/to/nemo_config.yaml \
  --train_schema path/to/train.jsonl \
  --val_schema path/to/val.jsonl \
  --feature_base_dir /data/features \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8 \
  --pin_memory
```

With a tokenizer and an explicit pretrained model override:

```bash
cd training/nemo;
python main.py \
  --config path/to/nemo_config.yaml \
  --train_schema path/to/train.jsonl \
  --val_schema path/to/val.jsonl \
  --pretrained_model stt_en_conformer_ctc_large \
  --tokenizer_dir path/to/tokenizer_dir \
  --train_batch_size 8 \
  --val_batch_size 8
```

To use curriculum learning, set `curriculum.enabled: true` (and optionally
`curriculum.schedule` / `curriculum.score_batch_size`) inside your config
file — there's no separate CLI flag for it.

## Command-line arguments

| Flag | Default | Description |
|---|---|---|
| `--config` | *(required)* | Path to the NeMo training config file. |
| `--train_schema` | *(required)* | Path to the training dataset JSON/JSONL manifest. |
| `--val_schema` | *(required)* | Path to the validation dataset JSON/JSONL manifest. |
| `--feature_base_dir` | `None` | Base directory used to resolve relative feature paths in the schema. |
| `--feature_key` | `feature_path` | Schema key holding the feature file path. |
| `--text_key` | `text` | Schema key holding the transcript text. |
| `--pretrained_model` | `None` | Overrides `config.model.init_from_pretrained_model`. |
| `--tokenizer_dir` | `None` | Path to a SentencePiece tokenizer directory. |
| `--train_batch_size` | `4` | Training batch size. |
| `--val_batch_size` | `4` | Validation batch size. |
| `--num_workers` | `4` | Number of DataLoader worker processes. |
| `--pin_memory` | `False` (flag) | Enables pinned memory in DataLoaders. |
| `--drop_last` | `False` (flag) | Drop the last incomplete training batch. |

## How data flows through the dataloaders

Both training and scoring dataloaders are built by the internal
`_build_dataloader` helper, which:

- Wraps `SharedASRDataset.nemo_collate_fn` as the `collate_fn`, passing in
  the tokenizer, a `training` flag, and the dataset's own config.
- Shuffles only when `training=True`.
- Only applies `drop_last` when `training=True` (validation/scoring never
  drops the last batch).
- Sets `persistent_workers=True` whenever `num_workers > 0`.
- Accepts an optional `sampler` — used during curriculum stages to restrict
  training to a ranked subset (`CurriculumSampler`); left as `None`
  (default random shuffling over the full set) otherwise.

## Output

`main()` returns `exp_dir`, the experiment directory produced by
`train_nemo(...)` (presumably where checkpoints, logs, and NeMo's
experiment manager output land — check `training.nemo.trainer.train_nemo`
for exact behavior). In curriculum mode, `exp_dir` is overwritten each stage
and the function's return value reflects only the **last** stage's directory.
