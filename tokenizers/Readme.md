# Tokenizers

This directory contains language-specific tokenizer artifacts plus the helper scripts used to prepare training text, convert metadata into NeMo-style manifests, and train tokenizers.

## Directory Layout

```text
tokenizers/
├── train_nemo.json
├── kidawida/
│   ├── text_corpus/
│   │   └── document.txt
│   └── tokenizer_spe_bpe_v1024/
│       ├── tokenizer.model
│       ├── tokenizer.vocab
│       └── vocab.txt
├── kinyarwanda/
│   └── tokenizer_spe_bpe_v1024/
│       ├── tokenizer.model
│       ├── tokenizer.vocab
│       └── vocab.txt
└── scripts/
    ├── convert_to_nemo_manifest.py
    ├── extract_text.py
    ├── process_asr_text_tokenizer.py
    └── tokenizer.sh
```

## What The Language Folders Contain

The language folders hold generated tokenizer outputs for each supported language.

* `kidawida/` contains a prepared text corpus in `text_corpus/document.txt` and a SentencePiece tokenizer trained with vocabulary size 1024.
* `kinyarwanda/` contains a SentencePiece tokenizer trained with vocabulary size 1024.
* `tokenizer.model` is the serialized tokenizer model.
* `tokenizer.vocab` is the tokenizer vocabulary file produced by SentencePiece.
* `vocab.txt` is a text vocabulary artifact kept alongside the tokenizer files.

## Script Reference

### `scripts/convert_to_nemo_manifest.py`

Converts dataset metadata into a NeMo-style manifest structure.

What it does:

* Accepts `.json`, `.csv`, or `.tsv` input.
* For `.csv` and `.tsv`, it expects the columns `path`, `sentence`, and `sentence_domain`.
* For JSON input, it accepts either a list of records or a dictionary of records.
* Builds one output entry per sample with:
  * `audio_filepath`: the base audio path joined with `path + ".wav"`
  * `text`: the sentence text, stripped of surrounding whitespace
* Writes the result as a pretty-printed JSON array.

How to run it:

```bash
python scripts/convert_to_nemo_manifest.py \
  --input_path <input.json|input.csv|input.tsv> \
  --output_path <output_manifest.json> \
  --audio_base_path <base_directory_for_audio_files>
```

Example:

```bash
python scripts/convert_to_nemo_manifest.py \
  --input_path ./train.csv \
  --output_path ./train_nemo.json \
  --audio_base_path /mnt/c/Users/uwilo/Datasets/Kidawida
```

Notes:

* The script assumes the audio files are `.wav` files.
* It does not validate that the audio files exist on disk.

### `scripts/extract_text.py`

Extracts all transcript lines from a dataset directory into a plain text file.

What it does:

* Looks inside the dataset directory for one of these files, in order:
  * `train.csv`
  * `train.tsv`
  * `train.json`
* Loads the file with pandas.
* Reads the `sentence` column and joins the values with newline characters.
* Writes the extracted text to the requested output file.

How to run it:

```bash
python scripts/extract_text.py \
  --dataset_path <dataset_directory> \
  --output_path <output_text_file>
```

Example:

```bash
python scripts/extract_text.py \
  --dataset_path ./kidawida \
  --output_path ./kidawida/text_corpus/document.txt
```

Notes:

* The input file must contain a `sentence` column.
* If no supported `train.*` file exists in the dataset directory, the script raises `FileNotFoundError`.

### `scripts/process_asr_text_tokenizer.py`

Trains a tokenizer from either a manifest file or a plain text corpus.

What it does:

* Accepts either `--manifest` or `--data_file`.
* If `--manifest` is used, it reads one or more NeMo manifest files and extracts the `text` field from each JSON line.
* It writes those transcripts into `text_corpus/document.txt` under the output directory.
* It then trains a tokenizer into a directory named from the tokenizer type and vocabulary size.

Supported tokenizer modes:

* `--tokenizer spe` trains a SentencePiece tokenizer through NeMo helpers.
* `--tokenizer wpe` trains a HuggingFace `BertWordPieceTokenizer`.

Common output locations:

* `tokenizer_spe_<spe_type>_v<vocab_size>` for SentencePiece.
* `tokenizer_wpe_v<vocab_size>` for WordPiece.
* If SentencePiece flags such as `--spe_pad`, `--spe_bos`, or `--spe_eos` are used, those suffixes are appended to the output directory name.
* If `--spe_max_sentencepiece_length` is set to a positive value, the length limit is also included in the directory name.

How to run it with a manifest:

```bash
python scripts/process_asr_text_tokenizer.py \
  --manifest <manifest1.json,manifest2.json> \
  --data_root <output_directory> \
  --vocab_size 1024 \
  --tokenizer spe \
  --spe_type bpe \
  --log
```

How to run it with a plain text file:

```bash
python scripts/process_asr_text_tokenizer.py \
  --data_file <document.txt> \
  --data_root <output_directory> \
  --vocab_size 1024 \
  --tokenizer wpe \
  --log
```

Example for the current tokenizer folders:

```bash
python scripts/process_asr_text_tokenizer.py \
  --manifest ./train_nemo.json \
  --data_root ./kinyarwanda \
  --vocab_size 1024 \
  --tokenizer spe \
  --spe_type bpe \
  --spe_remove_extra_whitespaces \
  --log
```

Notes:

* When `--manifest` is used, the script expects JSON-line NeMo manifests with a `text` field on each line.
* If the output corpus file already exists, it is reused.
* The script depends on the `nemo` and `tokenizers` Python packages.

### `scripts/tokenizer.sh`

This file is a reference command showing how the tokenizer training script can be invoked.

What it does:

* Points to a manifest file.
* Selects an output directory for tokenizer artifacts.
* Sets the vocabulary size and tokenizer type.
* Enables SentencePiece BPE mode and whitespace cleanup.

How to use it:

* Treat it as a template.
* Update the hard-coded paths so they match your local environment.
* Run the equivalent `python scripts/process_asr_text_tokenizer.py ...` command directly, or execute the shell script after updating the paths.

## Typical Workflow

1. Prepare a dataset file with transcript text.
2. If needed, convert metadata into a NeMo manifest with `convert_to_nemo_manifest.py`.
3. Extract all transcript lines into a corpus with `extract_text.py`.
4. Train the tokenizer with `process_asr_text_tokenizer.py`.
5. Store the resulting tokenizer files under the language directory that matches the dataset.

## Dependencies

The scripts rely on:

* `pandas`
* `tqdm`
* `tokenizers`
* `nemo`

If you are running them in a fresh environment, install the required packages before training or conversion.