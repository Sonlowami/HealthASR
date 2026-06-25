"""
Project settings and folder paths.

Edit LANGUAGES when you add a new dataset.
All other modules import constants from here.
"""
from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data folders (pipeline writes forward, never modifies raw/) ---
RAW_ROOT = PROJECT_ROOT / "data" / "raw"           # original downloads
CLEANED_ROOT = PROJECT_ROOT / "data" / "cleaned"   # after verify + clean
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"  # 16 kHz WAV + transcripts (for training)

# --- Generated outputs (reports, plots, features) ---
STATS_DIR = PROJECT_ROOT / "outputs" / "statistics"   # JSON reports
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"      # plots from notebooks
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"    # log-mel .npy files

# Register each language: name → raw folder with train/dev/test manifests + audio
LANGUAGES = {
    "kidawida": {"code": "dav", "dir": RAW_ROOT / "kidawida"},
    # "kinyarwanda": {"code": "rw", "dir": RAW_ROOT / "kinyarwanda"},
    # "swahili": {"code": "sw", "dir": RAW_ROOT / "swahili"},
}

SPLITS = ("train", "dev", "test")

# Cleaning: clips shorter than this are removed
MIN_DURATION_SEC = 3.0

# Preprocessing + features (standard ASR settings)
SAMPLE_RATE = 16_000   # Hz — Whisper / Conformer expect 16 kHz
N_MELS = 80            # mel bands in spectrogram
N_FFT = 400
HOP_LENGTH = 160       # 10 ms per frame at 16 kHz

# Column names accepted in TSV / JSON / JSONL manifests (see data_loader.py)
TEXT_FIELDS = ("sentence", "text", "transcript", "transcription")
AUDIO_FIELDS = ("path", "audio", "audio_path", "file", "filename", "audio_filepath")
