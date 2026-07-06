from __future__ import annotations

import argparse
from importlib import import_module
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from nemo.collections.asr.models import EncDecCTCModelBPE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")
from dataset_classes.shared_dataset import SharedASRDataset, load_spt_tokenizer
from training.curriculum import CurriculumSampler, rank_samples_by_wer
from training.nemo.trainer import load_config, load_nemo_run_config, create_trainer, train_nemo


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train a NeMo ASR model on shared feature datasets.")
	parser.add_argument("--config", required=True, help="Path to the NeMo training config file.")
	parser.add_argument("--train_schema", required=True, help="Path to the training dataset JSON or JSONL file.")
	parser.add_argument("--val_schema", required=True, help="Path to the validation dataset JSON or JSONL file.")
	parser.add_argument(
		"--feature_base_dir",
		default=None,
		help="Base directory used to resolve relative feature paths in the dataset schema.",
	)
	parser.add_argument("--feature_key", default="feature_path", help="Schema key containing the feature path.")
	parser.add_argument("--text_key", default="text", help="Schema key containing the transcript text.")
	parser.add_argument(
		"--pretrained_model",
		default=None,
		help="NeMo pretrained model name or path. Defaults to config.model.init_from_pretrained_model.",
	)
	parser.add_argument("--tokenizer_dir", help="Path to the tokenizer directory (optional).")
	parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size.")
	parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size.")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")
	parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory in data loaders.")
	parser.add_argument("--drop_last", action="store_true", help="Drop the last incomplete training batch.")
	return parser


def _build_dataloader(
	dataset: SharedASRDataset,
	*,
	tokenizer,
	batch_size: int,
	num_workers: int,
	pin_memory: bool,
	drop_last: bool,
	training: bool,
	sampler=None,
) -> DataLoader:
	collate_fn = lambda batch: SharedASRDataset.nemo_collate_fn(  # noqa: E731
		batch,
		tokenizer=tokenizer,
		training=training,
		config=dataset.config,
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=training,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last if training else False,
		persistent_workers=num_workers > 0,
		collate_fn=collate_fn,
		sampler=sampler,
	)


def main() -> str:
	parser = build_arg_parser()
	args = parser.parse_args()

	raw_cfg = load_config(args.config)
	cfg = load_nemo_run_config(raw_cfg)

	tokenizer_dir = Path(cfg.model.tokenizer.dir)
	if not tokenizer_dir.exists():
		raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

	model_name = args.pretrained_model or cfg.model.get("init_from_pretrained_model")
	if not model_name:
		raise ValueError(
			"Missing pretrained model name. Pass --pretrained-model or set config.model.init_from_pretrained_model."
		)

	augmentation_cfg = cfg.get("augmentation", {})
	train_dataset = SharedASRDataset(
		manifest_schema_path=args.train_schema,
		training=True,
		feature_key=args.feature_key,
		text_key=args.text_key,
		feature_base_dir=args.feature_base_dir,
		config={"augmentation": augmentation_cfg},
	)
	val_dataset = SharedASRDataset(
		manifest_schema_path=args.val_schema,
		training=False,
		feature_key=args.feature_key,
		text_key=args.text_key,
		feature_base_dir=args.feature_base_dir,
		config={"augmentation": augmentation_cfg},
	)

	tokenizer = load_spt_tokenizer(args.tokenizer_dir)[0] if args.tokenizer_dir else None

	curriculum_cfg = cfg.get("curriculum", {})
	curriculum_enabled = bool(curriculum_cfg.get("enabled", False))

	curriculum_sampler = None
	if curriculum_enabled:
		model = EncDecCTCModelBPE.from_pretrained(model_name)
		model.spec_augmentation = None

		score_batch_size = int(curriculum_cfg.get("score_batch_size", args.val_batch_size))
		score_loader = _build_dataloader(
			train_dataset,
			tokenizer=tokenizer,
			batch_size=score_batch_size,
			num_workers=args.num_workers,
			pin_memory=args.pin_memory,
			drop_last=False,
			training=False,
		)
		ranked = rank_samples_by_wer(model, score_loader, tokenizer)
		ordered_indices = [item.sample_id for item in ranked]
		active_size = int(curriculum_cfg.get("active_size", 1.0) * len(ordered_indices))
		curriculum_sampler = CurriculumSampler(ordered_indices, active_size=active_size)
	else:
		model = EncDecCTCModelBPE.from_pretrained(model_name)
		model.spec_augmentation = None  # Disable NeMo's built-in SpecAugment, since we handle augmentation in the dataset.

	train_loader = _build_dataloader(
		train_dataset,
		tokenizer=tokenizer,
		batch_size=args.train_batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_memory,
		drop_last=args.drop_last,
		training=True,
		sampler=curriculum_sampler,
	)
	val_loader = _build_dataloader(
		val_dataset,
		tokenizer=tokenizer,
		batch_size=args.val_batch_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_memory,
		drop_last=False,
		training=False,
	)

	#model.change_vocabulary(new_tokenizer_type=tokenizer, new_tokenizer_dir=args.tokenizer_dir)

	trainer = create_trainer(cfg)
	exp_dir = train_nemo(
		model=model,
		model_cfg=cfg,
		trainer=trainer,
		train_dataloader=train_loader,
		val_dataloader=val_loader,
	)

	return exp_dir


if __name__ == "__main__":
	main()



