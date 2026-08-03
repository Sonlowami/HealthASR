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
import sys

from src.cleaning import run_clean
from src.features import run_augment, run_extract
from src.preprocessing import run_preprocess

STEPS = [
    ("clean", run_clean),
    ("preprocess", run_preprocess),
    ("extract", run_extract),
    ("augment", run_augment),
]


def main() -> int:
    """Run each step sequentially; stop if any step returns non-zero."""
    parser = argparse.ArgumentParser(description="Run the full ASR data pipeline")
    parser.add_argument("--language", help="Process one language only (e.g. kidawida)")
    args = parser.parse_args()

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
