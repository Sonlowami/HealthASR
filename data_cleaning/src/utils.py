"""
Shared helpers used across cleaning, preprocessing, and features.

Handles finding audio files on disk and checking clip duration.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import torchaudio

from pathlib import Path
import re
import subprocess

import imageio_ffmpeg


_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)"
)

from .config import LANGUAGES


def clips_dir(lang_dir: Path) -> Path:
    """
    Find where audio files live inside a language folder.
    Checks clips/, audio/, wavs/ — returns first found, else defaults to clips/.
    """
    for name in ("clips", "audio", "wavs"):
        d = lang_dir / name
        if d.is_dir():
            return d
    return lang_dir / "clips"


def resolve_audio(lang_dir: Path, audio_dir: Path, path_val: str) -> Path:
    """
    Locate an audio file from a manifest path value.
    Tries: absolute path → relative to audio_dir → filename only → relative to lang_dir.
    """
    p = Path(path_val)
    for candidate in (p, audio_dir / path_val, audio_dir / p.name, lang_dir / path_val):
        if candidate.is_file():
            return candidate
    # Return expected path even if missing (caller checks .is_file())
    return audio_dir / p.name

def audio_duration(path: Path) -> tuple[bool, float | None]:
    """
    Check if audio is readable and return its length in seconds.
    Returns (False, None) for corrupt or unsupported files.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        match = _DURATION_RE.search(result.stderr)
        if match:
            h, m, s = match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
            return True, duration

    except Exception as e:
        print(f"Error reading audio file: {path}")
        print(f"Exception: {e}")

    return False, None

def iter_languages(only: str | None = None):
    """
    Loop over languages registered in config.py.
    Pass only='kidawida' to process a single language.
    Yields: (language_name, {"code": ..., "dir": ...})
    """
    for name, meta in LANGUAGES.items():
        if only and name != only:
            continue
        yield name, meta
