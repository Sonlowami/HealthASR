# Data cleaning

Part of the [HealthASR](https://github.com/Sonlowami/HealthASR) repository — this directory handles dataset cleaning, preprocessing, feature extraction, and augmentation. It sits alongside `tokenizers/`; see that folder for tokenizer training and NeMo manifest conversion.

Multilingual ASR data pipeline for Kidaw'ida, Kinyarwanda, and Swahili .

**Pipeline:** raw → clean → preprocess → extract → augment

Each step is one script with a single entry function that processes all languages and splits (train, dev, test).

All commands below are run from **`data-cleaning/`** (this directory), not the repo root.

---

## Current status

| Language    | Raw | Clean | Preprocess | Features | Augment |
|-------------|-----|-------|------------|----------|---------|
| Kidaw'ida   | ✓   | ✓     | ✓          | ✓        | ✓       |
| Kinyarwanda | —   | —     | —          | —        | —       |
| Swahili     | —   | —     | —          | —        | —       |

### Kidaw'ida results (after full pipeline)

| Split | Raw clips | Kept after clean | Notes                          |
|-------|-----------|------------------|--------------------------------|
| train | 2,098     | 1,639            | 459 removed (&lt; 3 s)         |
| dev   | 1,276     | 1,080            | 196 removed (&lt; 3 s)         |
| test  | 1,004     | 964              | 40 removed (&lt; 3 s)          |

- Zero missing audio, empty transcripts, or corrupt files
- Full report: `outputs/statistics/cleaning_report.json`

---

## Folder structure

```text
data-cleaning/
├── data/
│   ├── raw/                ← original downloads (never modified, local only)
│   │   └── kidawida/
│   ├── cleaned/            ← filtered TSV manifests (local only)
│   │   └── kidawida/manifests/
│   └── processed/          ← 16 kHz WAV + training manifests (local only)
│       └── kidawida/
│           ├── audio/{train,dev,test}/
│           └── manifests/*_processed.tsv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_feature_extraction.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── cleaning.py
│   ├── preprocessing.py
│   └── features.py
├── outputs/                ← reports, plots, features (local only)
│   ├── statistics/
│   ├── figures/
│   └── features/
│       └── kidawida/
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

## Setup

Clone the full HealthASR repo, then work inside this directory:

```bash
git clone https://github.com/Sonlowami/HealthASR.git
cd HealthASR/data-cleaning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download raw datasets into `data/raw/<language>/` locally — they are not stored on GitHub.

---

## Pipeline steps

### 1. Clean — `python -m src.cleaning`

**Reads:** `data/raw/<lang>/` (manifests + audio in `clips/` or `audio/`)

**Does:**
1. **Verify** — check manifests load, columns exist, every audio file is on disk
2. **Filter** — remove clips with empty text, missing/corrupt audio, or duration &lt; 3 s

**Writes:**
- `data/cleaned/<lang>/manifests/{train,dev,test}.tsv`
- `outputs/statistics/cleaning_report.json` (includes verify results under `"verify"`)

Aborts if verification fails for a language.

### 2. Preprocess — `python -m src.preprocessing`

**Reads:** cleaned manifests; audio still loaded from `data/raw/`

**Does:**
- Resample to **16 kHz mono WAV**
- Normalize transcripts: lowercase, trim whitespace, collapse extra spaces
- Add `audio_path` and `transcript` columns

**Writes:**
- `data/processed/<lang>/audio/{train,dev,test}/*.wav`
- `data/processed/<lang>/manifests/{split}_processed.tsv`

### 3. Extract — `python -m src.features extract`

**Reads:** processed WAV + `*_processed.tsv`

**Does:**
- Compute **80-bin log-mel spectrograms** (16 kHz, 10 ms frames)
- Save each clip as a `.npy` array

**Writes:**
- `outputs/features/<lang>/{train,dev,test}/*.npy`
- `outputs/features/<lang>/{split}_features.tsv` (adds `feature_path`, `feature_shape`)

### 4. Augment — `python -m src.features augment`

**Reads:** train features only

**Does:**
- **SpecAugment** on train: time masking (max 40 frames) + frequency masking (max 15 mel bins)
- One augmented copy per training clip (`*_aug0.npy`)
- Dev and test are never augmented

**Writes:**
- `outputs/features/<lang>/train_augmented/*.npy`
- `outputs/features/<lang>/train_augmented.tsv`

---

## Run the pipeline

From **`data-cleaning/`** with venv active:

```bash
# All steps, all languages
python run_pipeline.py

# One language only
python run_pipeline.py --language kidawida
```

Or step by step:

```bash
python -m src.cleaning
python -m src.preprocessing
python -m src.features extract
python -m src.features augment

# add --language kidawida to any command above
```

---

## Training data (handoff)

Two options depending on what the model expects:

### Option A — WAV + transcript (Whisper, many Conformer setups)

```text
data/processed/<language>/manifests/train_processed.tsv
data/processed/<language>/manifests/dev_processed.tsv
data/processed/<language>/manifests/test_processed.tsv
data/processed/<language>/audio/
```

Key columns: **`audio_path`**, **`transcript`**

### Option B — Precomputed log-mel features

```text
outputs/features/<language>/train_features.tsv
outputs/features/<language>/train_augmented.tsv   ← extra train copies
outputs/features/<language>/dev_features.tsv
outputs/features/<language>/test_features.tsv
outputs/features/<language>/
```

Key columns: **`feature_path`**, **`transcript`**, **`feature_shape`**

Train on original + augmented train manifests for more data. Use dev/test as-is for evaluation.

---

## Add a new language

1. Download Common Voice (or similar) into `data/raw/<name>/`
2. Ensure `train`, `dev`, `test` manifests exist (`.tsv`, `.json`, or `.jsonl`)
3. Put audio in `clips/` or `audio/`
4. Register in `src/config.py`:

```python
"kinyarwanda": {"code": "rw", "dir": RAW_ROOT / "kinyarwanda"},
```

5. Run `python run_pipeline.py --language kinyarwanda`

---

## Settings (`src/config.py`)

| Setting            | Value   | Purpose                          |
|--------------------|---------|----------------------------------|
| `MIN_DURATION_SEC` | 3.0     | Drop clips shorter than this     |
| `SAMPLE_RATE`      | 16000   | Resample target (Hz)             |
| `N_MELS`           | 80      | Mel bands in spectrogram         |
| `HOP_LENGTH`       | 160     | 10 ms per frame at 16 kHz        |

---

## Notebooks

Run **after** the scripts to inspect outputs — they do not replace the pipeline.

1. **`01_data_preparation.ipynb`** — raw data layout, cleaning stats, sample clips
2. **`02_feature_extraction.ipynb`** — processed WAV, log-mel shapes, spectrogram plots

---

## Manifest formats

Auto-detected by `data_loader.py`. Normalized to `sentence` + `path`.

**Text columns:** `sentence`, `text`, `transcript`, `transcription`  
**Audio columns:** `path`, `audio`, `audio_path`, `file`, `filename`, `audio_filepath`
