# Training

Fine-tunes a NeMo ASR model from a pretrained checkpoint. Supports standard training and curriculum learning.

## Files

```
training/
└── nemo/
    └── audio_pipeline.py   ← entry point; run this directly
```

Supporting utilities live in `../utils/`:

| File | Purpose |
|------|---------|
| `model_utils.py` | Load pretrained model, set up tokenizer/data/optimiser |
| `curriculun_utils.py` | Score manifest by per-sample WER, write stage manifests |

## Setup

Config is in `../config/nemo_audio_config.yaml`. Edit it before running.

Set your HuggingFace token if the model is gated:

```bash
echo "HF_TOKEN=hf_..." > .env
```

## Usage

Run from the **repo root**.

### Standard training

```bash
python training/nemo/audio_pipeline.py \
  --model_class nemo.collections.asr.models.EncDecCTCModelBPE \
  --config config/nemo_audio_config.yaml
```

Pass `--pretrained_model` to override the model name in the config:

```bash
python training/nemo/audio_pipeline.py \
  --model_class nemo.collections.asr.models.EncDecCTCModelBPE \
  --config config/nemo_audio_config.yaml \
  --pretrained_model nvidia/parakeet-ctc-1.1b
```

`--pretrained_model` also accepts a local `.nemo` checkpoint path.

### Curriculum training

Add `--curriculum`. The model scores every training sample by WER, then trains in stages from easiest to hardest.

```bash
python training/nemo/audio_pipeline.py \
  --model_class nemo.collections.asr.models.EncDecCTCModelBPE \
  --config config/nemo_audio_config.yaml \
  --curriculum
```

## Config

Minimum required fields — see `config/nemo_audio_config.yaml` for a full example.

```yaml
model:
  init_from_pretrained_model: nvidia/parakeet-ctc-1.1b   # or omit and use --pretrained_model
  tokenizer_dir: ./kinyarwanda/tokenizer_spe_bpe_v1024
  tokenizer_type: bpe
  train_ds:
    manifest_filepath: /path/to/train.json
    batch_size: 8
  validation_ds:
    manifest_filepath: /path/to/val.json
    batch_size: 16
  optim:
    name: adamw
    lr: 0.0001

trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 50
  precision: bf16

exp_manager:
  exp_dir: ./nemo_experiments
```

### Additional fields for curriculum training

```yaml
curriculum:
  schedule: [0.2, 0.5, 0.7, 1.0]   # fraction of data active at each stage
  epochs_per_stage: [5, 5, 5, 10]   # epochs to train at each stage (required)
  warmup_epochs: 3                   # optional: train on full data first
  score_batch_size: 16               # batch size used during WER scoring
```

`schedule` and `epochs_per_stage` must have the same length. `warmup_epochs` defaults to `0` (no warmup); `score_batch_size` defaults to `16`.

## CLI reference

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_class` | yes | Dotted path to the NeMo model class |
| `--config` | yes | Path to YAML config file |
| `--pretrained_model` | no | Model name (HF/NeMo Hub) or `.nemo` path; falls back to `model.init_from_pretrained_model` in config |
| `--curriculum` | no | Enable curriculum learning |
