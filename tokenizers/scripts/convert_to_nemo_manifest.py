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
        required_cols = {"path", "sentence", "sentence_domain"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def convert_to_nemo_manifest(input_path, output_path, audio_base_path):
    data = load_data(input_path)

    manifest_lines = []

    for i, entry in enumerate(tqdm(data, total=len(data))):
        try:
            audio_file = entry['path'] + '.wav'
            audio_path = os.path.join(audio_base_path, audio_file)

            transcription = entry['sentence']
            duration = entry['duration_sec']

            manifest_lines.append({
                "audio_filepath": audio_path,
                "duration": float(duration),
                "text": str(transcription).strip()
            })

        except KeyError as e:
            print(f"Skipping entry {i} due to missing key: {e}")

    # Write the entire manifest as a single JSON array (JSONL previously)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_lines, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert to nemo manifest file")
    parser.add_argument("--input_path", help="Path to the input file: supports json, csv, tsv")
    parser.add_argument("--output_path", help="Path to write the extracted text")
    parser.add_argument("--audio_base_path", help="Path to the audio base")
    args = parser.parse_args()

    convert_to_nemo_manifest(args.input_path, args.output_path, args.audio_base_path)