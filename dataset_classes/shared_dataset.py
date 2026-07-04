# shared_asr_dataset.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import csv

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
import os
from typing import Tuple
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


def load_spt_tokenizer(
    tokenizer_dir: str,
    legacy: bool = False,
    ignore_extra_whitespaces: bool = True,
    trim_spm_separator_after_special_token: bool = True,
    spm_separator: str = "▁",
) -> Tuple[SentencePieceTokenizer, str]:
    """
    Load a NeMo SentencePieceTokenizer from a tokenizer directory.

    Expected files in tokenizer_dir:
      - tokenizer.model   (required)
      - vocab.txt         (optional, but commonly produced by create_spt_model)

    Returns:
      (tokenizer, vocab_path)

    Raises:
      ValueError: if tokenizer_dir is invalid or tokenizer.model is missing.
    """
    if not tokenizer_dir or not os.path.isdir(tokenizer_dir):
        raise ValueError(f"tokenizer_dir must be a valid directory, got: {tokenizer_dir}")

    model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    vocab_path = os.path.join(tokenizer_dir, "vocab.txt")

    if not os.path.isfile(model_path):
        raise ValueError(f"Missing tokenizer model file: {model_path}")

    tokenizer = SentencePieceTokenizer(
        model_path=model_path,
        special_tokens=None,  # required unless legacy=True and you explicitly add special tokens
        legacy=legacy,
        ignore_extra_whitespaces=ignore_extra_whitespaces,
        trim_spm_separator_after_special_token=trim_spm_separator_after_special_token,
        spm_separator=spm_separator,
    )

    return tokenizer, vocab_path

# Expected global CONFIG, for example:
# CONFIG = {
#     "augmentation": {
#         "use_time_mask": True,
#         "use_freq_mask": True,
#         "time_mask_param": 30,
#         "freq_mask_param": 15,
#         "num_time_masks": 2,
#         "num_freq_masks": 2,
#     }
# }



def _resolve_feature_path(base_dir: Optional[Union[str, Path]], feature_path: str) -> Path:
    fp = Path(feature_path)
    if fp.is_absolute() or base_dir is None:
        return fp
    return Path(base_dir) / fp


def _load_features(feature_path: Union[str, Path]) -> torch.Tensor:
    feature_path = Path(feature_path)
    suffix = feature_path.suffix.lower()

    if suffix == ".pt":
        features = torch.load(feature_path, map_location="cpu")
    elif suffix == ".npy":
        features = np.load(feature_path)
    elif suffix == ".npz":
        archive = np.load(feature_path)
        features = archive["features"] if "features" in archive else archive[next(iter(archive.files))]
    else:
        raise ValueError(f"Unsupported feature file type: {feature_path}")

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    elif not torch.is_tensor(features):
        features = torch.tensor(features)

    features = features.float()

    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor, got shape {tuple(features.shape)} from {feature_path}")

    # Canonical in-memory layout: [T, 80]
    if features.shape[-1] != 80 and features.shape[0] == 80:
        features = features.transpose(0, 1).contiguous()

    if features.shape[-1] != 80:
        raise ValueError(f"Expected one feature axis of size 80, got shape {tuple(features.shape)} from {feature_path}")

    return features


def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a manifest file in csv, tsv, json or jsonl format and return list of dicts.

    - .csv : comma-separated, header required
    - .tsv or .tsc : tab-separated, header required
    - .json : expects a JSON list of objects or a single object (wrapped into list)
    - .jsonl: newline-delimited JSON, one object per line
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [dict(r) for r in reader]
    if suffix in (".tsv", ".tsc"):
        with p.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            return [dict(r) for r in reader]
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Each line in jsonl must be a JSON object")
                records.append(obj)
        return records
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
            if isinstance(obj, list):
                return [dict(o) for o in obj]
            if isinstance(obj, dict):
                return [obj]
            raise ValueError("Unsupported JSON manifest structure; expected list or object")

    raise ValueError(f"Unsupported manifest file type: {p}")


class SharedASRDataset(Dataset):
    def __init__(
        self,
        manifest_schema_path: Union[str, Path],
        training: bool,
        feature_key: str = "feature_path",
        text_key: str = "text",
        feature_base_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.schema_path = Path(manifest_schema_path)
        self.training = bool(training)
        self.feature_key = feature_key
        self.text_key = text_key
        self.feature_base_dir = Path(feature_base_dir) if feature_base_dir is not None else None
        self.config = config or {}
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, Any]]:
        rows = load_manifest(self.schema_path)
        normalized: List[Dict[str, Any]] = []

        for row in rows:
            if self.feature_key in row:
                feature_path = row[self.feature_key]
            elif "feature_filepath" in row:
                feature_path = row["feature_filepath"]
            elif "audio_filepath" in row:
                feature_path = row["audio_filepath"]
            else:
                raise KeyError(
                    f"Missing feature path key. Expected '{self.feature_key}', 'feature_filepath', or 'audio_filepath'."
                )

            if self.text_key in row:
                text = row[self.text_key]
            elif "transcription" in row:
                text = row["transcription"]
            else:
                raise KeyError(f"Missing text key. Expected '{self.text_key}' or 'transcription'.")

            resolved = str(_resolve_feature_path(self.feature_base_dir, str(feature_path)))
            normalized.append({**row, self.feature_key: resolved, self.text_key: text})

        return normalized

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        feature_path = row.get(self.feature_key) or row.get("feature_filepath") or row.get("audio_filepath")
        text = row.get(self.text_key) or row.get("transcription")
        features = _load_features(feature_path)

        return {
            "input_features": features,  # [T, 80]
            "text": text,
            "feature_path": str(feature_path),
            "num_frames": int(features.shape[0]),
        }

    @staticmethod
    def _pad_features(features: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(features) else 0
        feat_dim = int(features[0].shape[1]) if len(features) else 80

        padded = torch.zeros((len(features), max_len, feat_dim), dtype=torch.float32)
        for i, feat in enumerate(features):
            padded[i, : feat.shape[0]] = feat
        return padded, lengths

    @staticmethod
    def _pad_labels(
        label_ids: Sequence[Sequence[int]],
        pad_id: int = 0,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(ids) for ids in label_ids], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(label_ids) else 0

        padded = torch.full((len(label_ids), max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(label_ids):
            if len(ids):
                padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        attention_mask = padded.ne(pad_id)
        labels = padded.masked_fill(~attention_mask, ignore_index)
        return labels, lengths

    def _make_aug_transforms(self) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        aug_cfg = (self.config or {}).get("augmentation", {})
        use_time_mask = aug_cfg.get("use_time_mask", True)
        use_freq_mask = aug_cfg.get("use_freq_mask", True)
        time_mask_param = int(aug_cfg.get("time_mask_param", 30))
        freq_mask_param = int(aug_cfg.get("freq_mask_param", 15))

        time_mask = T.TimeMasking(time_mask_param=time_mask_param) if use_time_mask else None
        freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param) if use_freq_mask else None
        return time_mask, freq_mask

    def _apply_augmentation(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return feat

        # torchaudio masking expects [batch_or_channel, freq, time] in common use.
        # For our canonical [T, 80], transpose to [1, 80, T], mask, then transpose back.
        aug_cfg = (self.config or {}).get("augmentation", {})
        num_time_masks = int(aug_cfg.get("num_time_masks", 2))
        num_freq_masks = int(aug_cfg.get("num_freq_masks", 2))

        time_mask, freq_mask = self._make_aug_transforms()

        x = feat.transpose(0, 1).unsqueeze(0)  # [1, 80, T]

        if freq_mask is not None:
            for _ in range(num_freq_masks):
                x = freq_mask(x)

        if time_mask is not None:
            for _ in range(num_time_masks):
                x = time_mask(x)

        return x.squeeze(0).transpose(0, 1).contiguous()  # back to [T, 80]

    @classmethod
    def hf_collate_fn(
        cls,
        features: List[Dict[str, Any]],
        processor: Any,
        training: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Hugging Face collate function for precomputed features.

        Expected output:
        - input_features: [B, T, 80]
        - labels: [B, L] with padding masked as -100
        """
        config = config or {}
        dataset_like = cls.__new__(cls)  # lightweight namespace for helpers
        dataset_like.training = training
        dataset_like.config = config

        processed_features: List[torch.Tensor] = []
        labels_list: List[List[int]] = []

        for item in features:
            feat = item["input_features"]
            if not torch.is_tensor(feat):
                feat = torch.tensor(feat)
            feat = feat.float()

            if feat.ndim != 2:
                raise ValueError(f"Expected 2D features, got shape {tuple(feat.shape)}")

            if feat.shape[-1] != 80 and feat.shape[0] == 80:
                feat = feat.transpose(0, 1).contiguous()

            feat = SharedASRDataset._apply_aug_static(feat, training=training, config=config)
            processed_features.append(feat)

            if "labels" in item and item["labels"] is not None:
                labels_list.append(item["labels"])
            else:
                labels_list.append(processor.tokenizer(item["text"]).input_ids)

        input_features, feature_lengths = cls._pad_features(processed_features)
        labels, label_lengths = cls._pad_labels(
            labels_list,
            pad_id=processor.tokenizer.pad_token_id,
            ignore_index=-100,
        )

        return {
            "input_features": input_features,
            "labels": labels,
            "input_lengths": feature_lengths,
            "label_lengths": label_lengths,
        }

    @classmethod
    def nemo_collate_fn(
        cls,
        features: List[Dict[str, Any]],
        tokenizer: Any,
        training: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Collate function that returns unpacked tuple for EncDecCTCModel."""
        config = config or {}
        processed_features: List[torch.Tensor] = []
        token_ids_list: List[List[int]] = []
    
        for item in features:
            feat = item["input_features"]
            if not torch.is_tensor(feat):
                feat = torch.tensor(feat)
            feat = feat.float()
    
            if feat.ndim != 2:
                raise ValueError(f"Expected 2D features, got shape {tuple(feat.shape)}")
    
            if feat.shape[-1] != 80 and feat.shape[0] == 80:
                feat = feat.transpose(0, 1).contiguous()
    
            feat = SharedASRDataset._apply_aug_static(feat, training=training, config=config)
            processed_features.append(feat)
    
            if "labels" in item and item["labels"] is not None:
                token_ids_list.append(item["labels"])
            else:
                token_ids_list.append(tokenizer.text_to_ids(item["text"]))
    
        input_features, feature_lengths = cls._pad_features(processed_features)
        input_features = input_features.transpose(1, 2).contiguous()
        labels, label_lengths = cls._pad_labels(
            token_ids_list,
            pad_id=getattr(tokenizer, "pad_id", 0),
            ignore_index=-100,
        )
    
        return DALIOutputs({
            'processed_signal': input_features,
            'processed_signal_len': feature_lengths,
            'transcript': labels,
            'transcript_len': label_lengths,
        })

    @staticmethod
    def _apply_aug_static(feat: torch.Tensor, training: bool, config: Optional[Dict[str, Any]]) -> torch.Tensor:
        if not training:
            return feat

        aug_cfg = (config or {}).get("augmentation", {})
        use_time_mask = aug_cfg.get("use_time_mask", True)
        use_freq_mask = aug_cfg.get("use_freq_mask", True)
        time_mask_param = int(aug_cfg.get("time_mask_param", 30))
        freq_mask_param = int(aug_cfg.get("freq_mask_param", 15))
        num_time_masks = int(aug_cfg.get("num_time_masks", 2))
        num_freq_masks = int(aug_cfg.get("num_freq_masks", 2))

        x = feat.transpose(0, 1).unsqueeze(0)  # [1, 80, T]

        if use_freq_mask:
            freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
            for _ in range(num_freq_masks):
                x = freq_mask(x)

        if use_time_mask:
            time_mask = T.TimeMasking(time_mask_param=time_mask_param)
            for _ in range(num_time_masks):
                x = time_mask(x)

        return x.squeeze(0).transpose(0, 1).contiguous()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SharedASRDataset")
    parser.add_argument("--manifest", type=str, help="Path to manifest file (csv, tsv, json, or jsonl)")
    parser.add_argument("--feature_key", type=str, default="feature_path", help="Key for feature path in manifest")
    parser.add_argument("--text_key", type=str, default="text", help="Key for text in manifest")
    parser.add_argument("--feature_base_dir", type=str, default=None, help="Base directory for feature paths")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Path to tokenizer directory (optional)")
    #parser.add_argument("--tokenizer_type", type=str, default="bpe", help="Tokenizer type (bpe, wordpiece, etc.)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Manifest file not found: {manifest_path}")
        exit(1)
    loaded_records = load_manifest(manifest_path)
    print(f"Loaded {len(loaded_records)} records from {manifest_path}")

    dataset = SharedASRDataset(
        manifest_schema_path=args.manifest,
        training=False,
        feature_key=args.feature_key,
        text_key=args.text_key,
        feature_base_dir=args.feature_base_dir,
    )
    tokenizer = load_spt_tokenizer(args.tokenizer_dir)[0] if args.tokenizer_dir else None
    print(f"Dataset length: {len(dataset)}")
    print(f"Tokenizer: {tokenizer}")

    nemo_loader  = DataLoader(dataset, batch_size=2, collate_fn=lambda x: SharedASRDataset.nemo_collate_fn(x, tokenizer=tokenizer))
    print(f"Created DataLoader with batch size 2 and collate_fn for NeMo model.")
    print(f"Iterating over DataLoader...")
    for batch in nemo_loader:
        print(f"Batch keys: {list(vars(batch).keys())}")
        print(f"Batch shapes:", batch._outs[0].shape)
        break  # Just test one batch