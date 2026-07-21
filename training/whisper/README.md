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

# 3. curriculum: score per-sample WER, train in easiest-first stages
python training/whisper/train.py --config config/whisper_config.yaml --curriculum

# 4. evaluate the fine-tuned model: set checkpoint: ./whisper_experiments/kin-dav/final
#    in the config, then run --eval_only again
```

## Notes

- Language tokens (SALT scheme): Kinyarwanda `kin` = 50350, Kidaw'ida uses
  Swahili `swa` = 50318 as proxy. Set per language in the config.
- `oversample` repeats a language's train set to counter data imbalance.
- Curriculum ranks WER *within* each language, so every stage stays bilingual.
- Early stopping monitors validation loss (paper recipe:
  `early_stopping_patience` eval rounds x `eval_steps`).
- Anything under `training:` in the YAML is passed straight to HF
  `Seq2SeqTrainingArguments`.
