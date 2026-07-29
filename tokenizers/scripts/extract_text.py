import pandas as pd
from pathlib import Path
import json
import os

def load_existing_sentences(output_path: str) -> list[str]:
    """
    Load sentences already written to output_path, if it exists. Returns
    a list of the sentences in order.
    """
    if not os.path.isfile(output_path):
        return [], set()

    with open(output_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
    return lines


def load_train_dataframe(train_file: Path) -> pd.DataFrame:
    """
    Load a train file into a DataFrame, handling csv/tsv/json — including
    JSON shaped as either a list of records or a dict keyed by clip id
    (the latter needs orient='index', or pd.read_json silently produces
    an empty/garbled frame -- see the manifest-loading fix earlier).
    """
    if train_file.suffix == '.csv':
        return pd.read_csv(train_file)
    elif train_file.suffix == '.tsv':
        return pd.read_csv(train_file, sep='\t')
    elif train_file.suffix == '.json':
        raw = json.loads(train_file.read_text())
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
            return pd.DataFrame(raw["data"])
        elif isinstance(raw, dict):
            return pd.DataFrame.from_dict(raw, orient="index")
        else:
            raise ValueError(f"Unrecognized JSON shape in {train_file}")
    else:
        raise ValueError(f"Unsupported file format: {train_file.suffix}")


def load_seen_audio_paths(sidecar_path: str) -> set[str]:
    """Load the set of audio_paths already extracted, from a sidecar tracking file."""
    if not os.path.isfile(sidecar_path):
        return set()
    with open(sidecar_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def save_seen_audio_paths(sidecar_path: str, seen_paths: set[str]) -> None:
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(seen_paths)))


def extract_text(dataset_path: str, output_path: str) -> str:
    """
    Extract sentence information from the documents. Supports tsv, csv and JSON.
    Expects a file dataset_path/train.[csv, tsv, json]

    Merges into an existing output_path rather than overwriting it. Dedup
    is keyed on audio_path (consistent with convert_to_nemo_manifest),
    tracked via a sidecar file next to output_path -- output_path itself
    stays plain sentence-per-line text for the tokenizer trainer.
    """
    dataset_dir = Path(dataset_path)

    train_file = None
    for ext in ['.csv', '.tsv', '.json']:
        candidate = dataset_dir / f'train{ext}'
        if candidate.exists():
            train_file = candidate
            break

    if train_file is None:
        raise FileNotFoundError(f"No train file found in {dataset_path}")

    sidecar_path = output_path + '.seen_audio_paths'

    try:
        df = load_train_dataframe(train_file)

        existing_lines = load_existing_sentences(output_path)
        seen_paths = load_seen_audio_paths(sidecar_path)
        new_lines = list(existing_lines)

        added, skipped_duplicates, skipped_empty, skipped_missing_key = 0, 0, 0, 0
        for _, row in df.iterrows():
            try:
                audio_path = row['audio_path']
            except KeyError:
                skipped_missing_key += 1
                continue

            if audio_path in seen_paths:
                skipped_duplicates += 1
                continue

            sentence = str(row['sentence']).strip()
            if not sentence:
                skipped_empty += 1
                continue

            new_lines.append(sentence)
            seen_paths.add(audio_path)
            added += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        save_seen_audio_paths(sidecar_path, seen_paths)

        print(f"Added {added} new sentences, skipped {skipped_duplicates} duplicates "
              f"(by audio_path), skipped {skipped_empty} empty, {skipped_missing_key} "
              f"missing audio_path. Output now has {len(new_lines)} total sentences at {output_path}.")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from train dataset files.")
    parser.add_argument("--dataset_path", help="Path to the dataset directory")
    parser.add_argument("--output_path", help="Path to write the extracted text")
    args = parser.parse_args()

    try:
        extract_text(args.dataset_path, args.output_path)
        print("Extracted text successfully")
    except ValueError as e:
        print(f"Can't extract text: {e}")