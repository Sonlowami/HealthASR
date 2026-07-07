#!/usr/bin/env python3
"""
Run the full ASR data pipeline in order.

Steps:
    1. clean      — verify + filter bad clips → data/cleaned/
    2. preprocess — 16 kHz WAV + transcripts → data/processed/
    3. extract    — log-mel features → outputs/features/
    4. augment    — SpecAugment on train features

Usage:
    python run_pipeline.py
    python run_pipeline.py --language kidawida
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.cleaning import run_clean
from src.features import run_extract
from src.preprocessing import run_preprocess


def main() -> int:
    """Run each step sequentially; stop if any step returns non-zero."""
    parser = argparse.ArgumentParser(description="Run the full ASR data pipeline")
    parser.add_argument("--language", help="Process one language only (e.g. kidawida)")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Root directory containing per-language raw data folders. "
             "Only affects the 'clean' step (overrides the default raw-data root).",
    )
    parser.add_argument(
        "--cleaned_dir",
        type=Path,
        default=None,
        help="Root directory to write cleaned manifests/report to. "
             "Only affects the 'clean' step (overrides config.CLEANED_ROOT/STATS_DIR).",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=None,
        help="Root directory to write processed audio and manifests to. "
                "Only affects the 'preprocess' step (overrides config.PROCESSED_ROOT)."
    )

    parser.add_argument(
        "--features_dir",
        type=Path,
        default=None,
        help="Root directory to write features to. "
                "Only affects the 'extract' and 'augment' steps (overrides config.FEATURES_DIR)."
        )

    args = parser.parse_args()

    STEPS = [
    ("clean", lambda lang: run_clean(lang, args.dataset_root, args.cleaned_dir)),
    ("preprocess", lambda lang: run_preprocess(lang, args.dataset_root, args.processed_dir, args.cleaned_dir)),
    ("extract", lambda lang: run_extract(lang, args.processed_dir, args.features_dir)),
    #("augment", run_augment),
]

    for name, fn in STEPS:
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        code = fn(args.language)
        if code != 0:
            print(f"\nPipeline stopped at: {name}")
            return code

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
