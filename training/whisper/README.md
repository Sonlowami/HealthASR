# Whisper fine-tuning (Kinyarwanda + Kidaw'ida)

Fine-tunes `Sunbird/asr-whisper-large-v3-salt` on a combined bilingual dataset.
Config: `config/whisper_config.yaml` — edit dataset paths there.

## Setup (on orchard)

```bash
pip install -r training/whisper/requirements.txt
echo "HF_TOKEN=hf_..." > .env   # account must have accepted the Sunbird model terms
```

Data expectations per language (set in the config):

- a manifest (TSV/CSV/JSON/JSONL) with an audio-path column and a text column
  (column names are auto-detected, e.g. `audio_path` + `transcript`)
- audio readable by `datasets` (WAV/MP3); resampled to 16 kHz on the fly

Kidaw'ida comes from the `data-cleaning` pipeline (`*_processed.tsv`).
Kinyarwanda paths point at the teammate-processed Track B data.

## Usage (from repo root, inside tmux)

```bash
# 1. zero-shot baseline — run this first
python training/whisper/train.py --config config/whisper_config.yaml --eval_only

# 2. standard fine-tuning
python training/whisper/train.py --config config/whisper_config.yaml

# 3. curriculum: WER-rank at each stage (current model) → 20/50/70/100% easiest
python training/whisper/train.py --config config/whisper_config.yaml --curriculum

# 4. evaluate the fine-tuned model: set checkpoint: ./whisper_experiments/kin-dav/final
#    in the config, then run --eval_only again

# 5. QAT + short finetune (any --model_path; data from config/whisper_qat.yaml)
#    schemes: int8_weight_qat | int4_weight_qat | int6_weight_qat | float8_weight_qat
#    results.json includes WER/CER/CES/combined_error/params/macs/size (mate schema)
python training/whisper/qat_finetune.py \
  --config config/whisper_qat.yaml \
  --model_path /path/to/any/whisper/final \
  --quant int8_weight_qat \
  --quant int4_weight_qat \
  --output_dir ./whisper_qat_out

# 6. FFN structural pruning 10/20/50% + recovery FT (mate-style torch-pruning)
python training/whisper/prune_finetune.py \
  --config config/whisper_prune.yaml \
  --model_path /path/to/any/whisper/final \
  --ratio 0.1 --ratio 0.2 --ratio 0.5 \
  --output_dir ./whisper_prune_out

# 7. Prune × QAT grid (mate nested JSON): 10/20/50% × int4/int6/int8
#    Needs pruned+FT dirs under prune_root (.../ffn_10/final etc.)
python training/whisper/prune_qat_finetune.py \
  --config config/whisper_prune_qat.yaml \
  --model_path /path/to/baseline/final \
  --prune_root /path/to/whisper_kin_dav_prune \
  --seed_results /path/to/whisper_kin_dav_prune/results.json \
  --output_dir ./whisper_prune_qat_out
```

## Notes

- Language tokens (SALT scheme): Kinyarwanda `kin` = 50350, Kidaw'ida uses
  Swahili `swa` = 50318 as proxy. Set per language in the config.
- `oversample` repeats a language's train set to counter data imbalance.
- Curriculum: at the start of each stage, rank train clips by WER with the
  *current* model (saved as `wer_difficulty_{lang}_stage{N}.npy`), then keep
  the easiest fraction (e.g. 20% → 50% → 70% → 100%) per language. Re-scoring
  updates hardness as the model improves; existing stage `.npy` files are
  reused on `--resume`.
- Early stopping monitors validation loss (paper recipe:
  `early_stopping_patience` eval rounds x `eval_steps`).
- Anything under `training:` in the YAML is passed straight to HF
  `Seq2SeqTrainingArguments`.
