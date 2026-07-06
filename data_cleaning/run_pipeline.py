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
from src.features import run_augment, run_extract
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
        "--output_root",
        type=Path,
        default=None,
        help="Root directory to write cleaned manifests/report to. "
             "Only affects the 'clean' step (overrides config.CLEANED_ROOT/STATS_DIR).",
    )
    args = parser.parse_args()

    STEPS = [
    ("clean", lambda lang: run_clean(lang, args.dataset_root, args.output_root)),
    ("preprocess", run_preprocess),
    ("extract", run_extract),
    ("augment", run_augment),
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
