import pandas as pd
from pathlib import Path

def extract_text(dataset_path: str, output_path: str) -> str:
    """
    Extract sentence information from the documents. Support tsv, csv and JSON.
    Expects a file dataset_path/train.[csv, tsv, json]
    """
    dataset_dir = Path(dataset_path)
    
    # Check for train file with supported extensions
    train_file = None
    for ext in ['.csv', '.tsv', '.json']:
        candidate = dataset_dir / f'train{ext}'
        if candidate.exists():
            train_file = candidate
            break
    
    if train_file is None:
        raise FileNotFoundError(f"No train file found in {dataset_path}")
    
    # Load the file based on extension
    try:
        if train_file.suffix == '.csv':
            df = pd.read_csv(train_file)
        elif train_file.suffix == '.tsv':
            df = pd.read_csv(train_file, sep='\t')
        elif train_file.suffix == '.json':
            df = pd.read_json(train_file)
        else:
            raise ValueError(f"Unsupported file format: {train_file.suffix}")
        
        # Accumulate sentences into buffer, space separated
        text_buffer = '\n'.join(df['sentence'].astype(str).tolist())
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.write(text_buffer)

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