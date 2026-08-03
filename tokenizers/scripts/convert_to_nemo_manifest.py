import pandas as pd
import json
import os
from tqdm import tqdm

def load_data(input_path):
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".json":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert dict → list of entries
        if isinstance(data, dict):
            data = list(data.values())
        return data

    elif ext in [".csv", ".tsv"]:
        sep = '\t' if ext == ".tsv" else ','
        df = pd.read_csv(input_path, sep=sep)

        # Expect columns: path, sentence, sentence_domain
        required_cols = {"path", "sentence",}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {ext}")

def load_existing_manifest(path: str) -> tuple[list[dict], set[str]]:
    """
    Load an existing JSONL manifest if present. Returns (entries, seen_paths)
    where seen_paths is the set of audio_filepath values already present,
    used to skip duplicates on merge. Returns ([], set()) if the file
    doesn't exist yet.
    """
    if not os.path.isfile(path):
        return [], set()

    entries = []
    seen_paths = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line {line_num} in existing manifest: {line[:80]!r}")
                continue
            entries.append(entry)
            seen_paths.add(entry.get("audio_filepath"))
    return entries, seen_paths


def convert_to_nemo_manifest(input_path, output_path, audio_base_path):
    data = load_data(input_path)

    existing_entries, seen_paths = load_existing_manifest(output_path)
    manifest_lines = list(existing_entries)

    added, skipped_duplicates, skipped_missing_key = 0, 0, 0

    for i, entry in enumerate(tqdm(data, total=len(data))):
        try:
            audio_file = entry.get("audio_path") or entry.get("path")
            if audio_file is None:
                raise KeyError("audio_path/path")
            audio_path = os.path.join(audio_base_path, audio_file)
            if audio_path in seen_paths:
                skipped_duplicates += 1
                continue

            transcription = entry['sentence']
            duration = entry['duration_sec']

            manifest_lines.append({
                "audio_filepath": audio_path,
                "duration": float(duration),
                "text": str(transcription).strip()
            })
            seen_paths.add(audio_path)
            added += 1

        except KeyError as e:
            print(f"Skipping entry {i} due to missing key: {e}")
            skipped_missing_key += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in manifest_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"Added {added} new entries, skipped {skipped_duplicates} duplicates, "
          f"skipped {skipped_missing_key} for missing keys. "
          f"Manifest now has {len(manifest_lines)} total entries at {output_path}.")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert to nemo manifest file")
    parser.add_argument("--input_path", help="Path to the input file: supports json, csv, tsv")
    parser.add_argument("--output_path", help="Path to write the extracted text")
    parser.add_argument("--audio_base_path", help="Path to the audio base")
    args = parser.parse_args()

    convert_to_nemo_manifest(args.input_path, args.output_path, args.audio_base_path)