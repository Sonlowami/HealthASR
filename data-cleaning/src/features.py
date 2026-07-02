"""
Step 4 & 5 of the pipeline: feature extraction and augmentation.

extract — compute 80-bin log-mel spectrograms (.npy)
augment — apply SpecAugment-style time + frequency masking (train only)

Optional if your model loads WAV directly (Whisper, Conformer).

Run:
    python -m src.features extract
    python -m src.features augment
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import FEATURES_DIR, HOP_LENGTH, N_FFT, N_MELS, PROCESSED_ROOT, SAMPLE_RATE, SPLITS
from .utils import iter_languages


def _mask(mel: np.ndarray, axis: int, max_w: int, n: int = 2) -> np.ndarray:
    """
    SpecAugment masking helper.

    axis=0 → frequency masking (horizontal band across all time)
    axis=1 → time masking (vertical strip across all mel bins)
    Masked regions are filled with the mean value of the spectrogram.
    """
    out = mel.copy()
    size = out.shape[axis]
    for _ in range(n):
        w = np.random.randint(1, min(max_w, size) + 1)       # random mask width
        start = np.random.randint(0, max(1, size - w))       # random start position
        sl = [slice(None)] * 2
        sl[axis] = slice(start, start + w)
        out[tuple(sl)] = out.mean()
    return out


def run_extract(language: str | None = None) -> int:
    """
    Extract log-mel features for all languages and splits (train, dev, test).

    Reads WAV from data/processed/, saves .npy to outputs/features/.
    Also writes a manifest TSV with feature_path and feature_shape per clip.
    """
    for name, _ in iter_languages(language):
        print(f"\nFeatures {name}...")

        for split in SPLITS:
            manifest = PROCESSED_ROOT / name / "manifests" / f"{split}_processed.tsv"
            df = pd.read_csv(manifest, sep="\t")
            feat_dir = FEATURES_DIR / name / split
            paths, shapes = [], []

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
                wav = PROCESSED_ROOT / name / row["audio_path"]
                y, sr = librosa.load(wav, sr=SAMPLE_RATE, mono=True)

                # Mel spectrogram → log scale (shape: 80 mel bins × time frames)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
                log_mel = librosa.power_to_db(mel, ref=np.max)

                out = feat_dir / f"{Path(row['audio_path']).stem}.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, log_mel)

                paths.append(str(out.relative_to(FEATURES_DIR / name)))
                shapes.append(str(list(log_mel.shape)))

            df["feature_path"], df["feature_shape"] = paths, shapes
            df.to_csv(FEATURES_DIR / name / f"{split}_features.tsv", sep="\t", index=False)

    return 0


def run_augment(language: str | None = None) -> int:
    """
    Augment training features for all languages (fixed seed for reproducibility).

    Creates one augmented copy of each training spectrogram.
    Applies time masking then frequency masking (SpecAugment).
    Only runs on train split — dev/test are never augmented.
    """
    np.random.seed(42)

    for name, _ in iter_languages(language):
        print(f"\nAugment {name}...")

        manifest = FEATURES_DIR / name / "train_features.tsv"
        df = pd.read_csv(manifest, sep="\t")
        aug_dir = FEATURES_DIR / name / "train_augmented"
        rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="  augment"):
            mel = np.load(FEATURES_DIR / name / row["feature_path"])

            # Time mask (max 40 frames) then frequency mask (max 15 mel bins)
            aug = _mask(_mask(mel, axis=1, max_w=40), axis=0, max_w=15)

            stem = Path(row["feature_path"]).stem
            path = aug_dir / f"{stem}_aug0.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, aug)

            r = row.to_dict()
            r["feature_path"] = str(path.relative_to(FEATURES_DIR / name))
            r["feature_shape"] = str(list(aug.shape))
            rows.append(r)

        pd.DataFrame(rows).to_csv(FEATURES_DIR / name / "train_augmented.tsv", sep="\t", index=False)

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract or augment log-mel features")
    p.add_argument("action", choices=["extract", "augment"])
    p.add_argument("--language", help="Process one language only")
    args = p.parse_args()
    fn = run_extract if args.action == "extract" else run_augment
    sys.exit(fn(args.language))
