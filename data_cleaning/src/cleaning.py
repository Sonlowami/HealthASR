"""
Clean raw data: verify integrity, then filter bad clips.

Verification runs automatically inside run_clean before any filtering.

Run:
    python -m src.cleaning
    python -m src.cleaning --language kidawida
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from tqdm import tqdm

from .config import CLEANED_ROOT, MIN_DURATION_SEC, SPLITS, STATS_DIR
from .data_loader import load_split
from .utils import audio_duration, clips_dir, iter_languages, resolve_audio


def verify_language(name: str, lang_dir) -> dict:
    """
    Read-only integrity check for one language folder.

    Checks:
        - folder exists
        - train/dev/test manifest loads (TSV, JSON, or JSONL)
        - required columns present
        - every listed audio file exists on disk

    Does NOT open audio files or remove any rows.
    Returns a report dict with "ok": True if everything passes.
    """
    report = {"language": name, "path": str(lang_dir), "splits": {}, "issues": [], "ok": False}

    if not lang_dir.is_dir():
        report["issues"].append(f"Directory not found: {lang_dir}")
        return report

    adir = clips_dir(lang_dir)
    missing_total = 0

    for split in SPLITS:
        info = {"manifest": None, "rows": 0, "missing_audio": 0, "empty_transcripts": 0}
        try:
            df, fmt = load_split(lang_dir, split)
            info["manifest"] = f"{split}.{fmt}"
            info["rows"] = len(df)

            empty = df["sentence"].isna() | (df["sentence"].str.strip() == "")
            info["empty_transcripts"] = int(empty.sum())

            missing = [p for p in df["path"] if not resolve_audio(lang_dir, adir, p).is_file()]
            info["missing_audio"] = len(missing)
            missing_total += info["missing_audio"]

        except FileNotFoundError:
            report["issues"].append(f"Missing manifest for {split}")
        except ValueError as exc:
            report["issues"].append(str(exc))
        report["splits"][split] = info

    report["clip_files_on_disk"] = len(list(adir.glob("*"))) if adir.is_dir() else 0
    report["ok"] = not report["issues"] and missing_total == 0
    return report


def run_clean(language: str | None = None) -> int:
    """
    Verify then clean all languages and splits (train, dev, test).

    For each split, removes rows where:
        - transcript is empty
        - audio file is missing
        - audio is corrupt (cannot be read)
        - duration < MIN_DURATION_SEC

    Calls verify_language first — aborts if check fails.
    Saves data/cleaned/<lang>/manifests/*.tsv and cleaning_report.json
    (includes verify results under the "verify" key).
    """
    all_stats = []

    for name, meta in iter_languages(language):
        if not meta["dir"].is_dir():
            print(f"Skip {name}: not found")
            continue

        print(f"\nVerifying {name}...")
        report = verify_language(name, meta["dir"])
        for s, info in report["splits"].items():
            print(f"  {s}: {info.get('manifest', '?')} rows={info['rows']} missing={info['missing_audio']}")
        if not report["ok"]:
            for issue in report["issues"]:
                print(f"  ! {issue}")
            print(f"Aborting {name} — fix raw data first.")
            return 1

        print(f"Cleaning {name}...")
        out_dir = CLEANED_ROOT / name / "manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        lang_stats = {"language": name, "verify": report, "splits": {}}
        lang_dir = meta["dir"]
        adir = clips_dir(lang_dir)

        for split in SPLITS:
            df, _ = load_split(lang_dir, split)
            stats = {"input": len(df), "empty_transcript": 0, "missing_file": 0, "corrupt_audio": 0, "too_short": 0, "kept": 0}
            kept = []

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
                text = row["sentence"].strip() if pd.notna(row["sentence"]) else ""
                if not text:
                    stats["empty_transcript"] += 1
                    continue

                apath = resolve_audio(lang_dir, adir, row["path"])
                if not apath.is_file():
                    stats["missing_file"] += 1
                    continue

                ok, dur = audio_duration(apath)
                if not ok or dur is None:
                    stats["corrupt_audio"] += 1
                    continue

                if dur < MIN_DURATION_SEC:
                    stats["too_short"] += 1
                    continue

                r = row.to_dict()
                r["duration_sec"] = dur
                kept.append(r)

            stats["kept"] = len(kept)
            pd.DataFrame(kept).to_csv(out_dir / f"{split}.tsv", sep="\t", index=False)
            lang_stats["splits"][split] = stats
            print(f"  {split}: {stats['kept']}/{stats['input']} kept")

        all_stats.append(lang_stats)

    if not all_stats:
        print("No languages processed.")
        return 1

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / "cleaning_report.json"
    path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify and clean raw ASR data")
    p.add_argument("--language", help="Process one language only")
    sys.exit(run_clean(p.parse_args().language))
