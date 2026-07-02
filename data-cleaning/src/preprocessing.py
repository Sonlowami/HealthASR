"""
Step 3 of the pipeline: preprocess cleaned data for model training.

Reads:  data/cleaned/<lang>/manifests/*.tsv
Writes: data/processed/<lang>/audio/{split}/*.wav
        data/processed/<lang>/manifests/*_processed.tsv  ← use this for training

Run:
    python -m src.preprocessing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .config import CLEANED_ROOT, PROCESSED_ROOT, SAMPLE_RATE, SPLITS
from .utils import clips_dir, iter_languages, resolve_audio


def run_preprocess(language: str | None = None) -> int:
    """
    Preprocess all cleaned splits (train, dev, test) for every language.

    For each clip:
        1. Load original audio from data/raw/
        2. Resample to 16 kHz mono, save as WAV in data/processed/
        3. Lowercase and normalize transcript text
        4. Write updated manifest with audio_path and transcript columns
    """
    for name, meta in iter_languages(language):
        print(f"\nPreprocessing {name}...")
        lang_dir = meta["dir"]

        for split in SPLITS:
            manifest = CLEANED_ROOT / name / "manifests" / f"{split}.tsv"
            if not manifest.is_file():
                raise FileNotFoundError(f"Run clean first: {manifest}")

            df = pd.read_csv(manifest, sep="\t")
            adir = clips_dir(lang_dir)
            out_audio = PROCESSED_ROOT / name / "audio" / split
            rows = []

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
                src = resolve_audio(lang_dir, adir, row["path"])
                dst = out_audio / Path(row["path"]).with_suffix(".wav").name
                dst.parent.mkdir(parents=True, exist_ok=True)

                y, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
                sf.write(dst, y, SAMPLE_RATE)

                r = row.to_dict()
                r["transcript"] = re.sub(r"\s+", " ", str(row["sentence"]).lower().strip())
                r["audio_path"] = str(dst.relative_to(PROCESSED_ROOT / name))
                rows.append(r)

            out = PROCESSED_ROOT / name / "manifests" / f"{split}_processed.tsv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, sep="\t", index=False)

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess cleaned ASR data to 16 kHz WAV")
    p.add_argument("--language", help="Process one language only")
    args = p.parse_args()
    sys.exit(run_preprocess(args.language))
