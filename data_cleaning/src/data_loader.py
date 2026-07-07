"""
Load dataset manifests from TSV, JSON, or JSONL.

All formats are normalized to two columns: sentence (text), path (audio).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AUDIO_FIELDS, TEXT_FIELDS


def _pick_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    """
    Match a column name from a list of accepted names.
    Handles case differences (e.g. 'Text' vs 'text').
    """
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name in lower:
            return lower[name]
    return None


def load_split(lang_dir: Path, split: str) -> tuple[pd.DataFrame, str]:
    """
    Load one data split (train, dev, or test) from the first manifest found:
        {split}.tsv  →  {split}.jsonl  →  {split}.json

    Returns:
        df  — dataframe with normalized columns: sentence, path
        ext — file extension used ('tsv', 'jsonl', or 'json')
    """
    for ext in ("tsv", "jsonl", "json"):
        path = lang_dir / f"{split}.{ext}"
        if not path.is_file():
            continue

        # --- Parse file by type ---
        if ext == "tsv":
            df = pd.read_csv(path, sep="\t")
        elif ext == "jsonl":
            df = pd.read_json(path, lines=True)  # one JSON object per line
        else:
            raw = json.loads(path.read_text())
            if isinstance(raw, list):
                # [{...}, {...}, ...]
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
                # {"data": [{...}, {...}]}
                df = pd.DataFrame(raw["data"])
            elif isinstance(raw, dict):
                # {clip_id: {"path": ..., "sentence": ...}, ...}
                df = pd.DataFrame.from_dict(raw, orient="index").reset_index().rename(
            columns={"index": "clip_id"}
        )
            else:
                raise ValueError(f"Unrecognized JSON manifest shape in {path}")
        # --- Map varying column names to standard sentence + path ---
        text_col = _pick_column(df, TEXT_FIELDS)
        audio_col = _pick_column(df, AUDIO_FIELDS)
        if not text_col or not audio_col:
            raise ValueError(f"{path.name}: need text + audio columns, got {list(df.columns)}")

        out = df.copy()
        out["sentence"] = out[text_col].astype(str)
        out["path"] = out[audio_col].astype(str)
        return out, ext

    raise FileNotFoundError(f"No {split}.tsv/.json/.jsonl in {lang_dir}")
