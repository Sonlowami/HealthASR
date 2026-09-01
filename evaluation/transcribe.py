import torch
import pandas as pd
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

from evaluation.main import _clean_text_list
from utils.model_utils import resolve_model_class, load_model

def transcribe_unlabeled_audio(
    model,
    audio_paths: list[str],
    device,
    batch_size: int = 4,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Runs inference on audio with NO ground-truth transcript, using NeMo's
    high-level model.transcribe() API (confirmed signature: `audio=`,
    `return_hypotheses`, `use_lhotse=True` by default) rather than the
    manifest-driven run_model_inference() path, which requires a `text`
    field to build its dataloader.

    Uses whatever decoding strategy is already configured on `model` --
    does not reset or override it, so it inherits e.g. the beam-search fix
    if setup_model()/change_decoding_strategy() already ran on this model.

    Returns a DataFrame with columns [utterance_id, prediction] (or
    prediction_{run_label} if given), and optionally writes it to output_csv.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        raw_output = model.transcribe(
            audio=audio_paths,
            batch_size=batch_size,
            return_hypotheses=False,  # explicit: we want plain text, not Hypothesis objects
        )

    # TranscriptionReturnType's exact shape isn't nailed down by the signature alone
    # (some NeMo paths return a bare List[str], others a (hyps, ...) tuple, and
    # return_hypotheses=False isn't universally guaranteed to suppress Hypothesis
    # objects for every model class) -- normalize defensively rather than assume.
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    predictions = []
    for item in raw_output:
        if isinstance(item, str):
            predictions.append(item)
        elif hasattr(item, "text"):
            predictions.append(item.text)
        elif hasattr(item, "words"):
            predictions.append(" ".join(item.words))
        else:
            raise TypeError(f"Unrecognized transcribe() output element type: {type(item)}")

    predictions = _clean_text_list(predictions)

    col_name = "transcription"
    result_df = pd.DataFrame({"utterance_id": audio_paths, col_name: predictions})

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Saved {len(result_df)} transcripts -> {output_csv}")

    model.train()
    return result_df

def build_test_path_from_manifest(manifest_path: str) -> list[str]:
    """
    Given a manifest path, return a list of audio file paths for testing.
    Assumes the manifest is a JSONL or CSV/TSV with an "audio_filepath" column.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    ext = manifest_path.suffix.lower()
    if ext == ".jsonl":
        audio_paths = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                audio_paths.append(entry.get("audio_filepath"))
        return audio_paths
    elif ext in [".csv", ".tsv"]:
        sep = '\t' if ext == ".tsv" else ','
        df = pd.read_csv(manifest_path, sep=sep)
        if "audio_filepath" not in df.columns:
            raise ValueError(f"Manifest file {manifest_path} is missing 'audio_filepath' column")
        return df["audio_filepath"].tolist()

def discover_audio_paths_from_dir(audio_dir: str) -> list[str]:
    """
    Given a directory, return a list of audio file paths for testing.
    Recursively searches for .wav files in the directory.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory not found: {audio_dir}")

    audio_paths = list(audio_dir.rglob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"No .wav files found in directory: {audio_dir}")

    return [str(p) for p in audio_paths]

def main(
    model_path: str,
    model_class: str,
    audio_paths: list[str],
    device: str = "cuda",
    batch_size: int = 4,
    output_csv: str | None = None,
):
    model_class = resolve_model_class(model_class)
    model = load_model(model_class, model_path, device=device)
    return transcribe_unlabeled_audio(
        model=model,
        audio_paths=audio_paths,
        device=device,
        batch_size=batch_size,
        output_csv=output_csv,
    )

def build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe unlabeled audio with a NeMo ASR model")
    parser.add_argument("--model_class", type=str, required=True, help="dotted path to the Nemo clas")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the NeMo ASR model (.nemo file)")
    parser.add_argument("--manifest_path", type=str, required=False, help="Path to the manifest file containing audio file paths")
    parser.add_argument("--audio_paths", type=str, nargs="+", required=True, help="List of audio file paths to transcribe")
    parser.add_argument("--audio_base_path", type=str, default="", help="Base path to prepend to audio file paths from the manifest")
    parser.add_argument("--audio_dir", type=str, nargs="?", help="Optional directory containing audio files (if not using manifest)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save the output DataFrame as CSV")
    return parser

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.manifest_path:
        audio_paths = build_test_path_from_manifest(args.manifest_path)
        if args.audio_base_path:
            audio_paths = [str(Path(args.audio_base_path) / Path(p)) for p in audio_paths]
    elif args.audio_dir:
        audio_paths = discover_audio_paths_from_dir(args.audio_dir)
    else:
        audio_paths = args.audio_paths

    main(
        model_path=args.model_path,
        model_class=args.model_class,
        audio_paths=audio_paths,
        device=args.device,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
    )

