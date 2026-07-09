from __future__ import annotations

import argparse
from importlib import import_module
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from torch.utils.data import DataLoader
from nemo.collections.asr.models import EncDecCTCModelBPE
import lightning.pytorch as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")
from dataset_classes.shared_dataset import SharedASRDataset, load_spt_tokenizer
from training.curriculum import CurriculumSampler, rank_samples_by_wer
from training.nemo.trainer import load_config, load_nemo_run_config, create_trainer, train_nemo


def resolve_model_class(dotted_path: str):
	"""
	Resolve a model class from a dotted path string, e.g.
	'nemo.collections.asr.models.EncDecCTCModelBPE' ->
	the EncDecCTCModelBPE class object.
	"""
	module_path, _, class_name = dotted_path.rpartition(".")
	if not module_path:
		raise ValueError(
			f"'{dotted_path}' is not a valid dotted path to a class "
			"(expected e.g. 'nemo.collections.asr.models.EncDecCTCModelBPE')."
		)
	module = import_module(module_path)
	return getattr(module, class_name)


class NemoASRPipeline:
	"""
	Trains a NeMo ASR model on shared feature datasets, with optional
	curriculum learning.

	To use a different NeMo model class, change `model_class` — either by
	setting it on an instance, subclassing, or via the --model_class CLI
	flag. Everything else (dataset loading, dataloaders, curriculum
	staging, trainer construction) is unaffected by the choice of model
	class, as long as that class exposes `.from_pretrained(name)` and a
	`.spec_augmentation` attribute, matching NeMo's usual ASR model API.

	Example — swap the model without touching anything else:
		pipeline = NemoASRPipeline(args)
		pipeline.model_class = EncDecRNNTBPEModel
		pipeline.run()
	"""

	#: Change this attribute to swap the underlying NeMo model class.
	model_class = EncDecCTCModelBPE

	def __init__(self, args: argparse.Namespace):
		self.args = args
		self.cfg = None
		self.model = None
		self.tokenizer = None
		self.train_dataset: SharedASRDataset | None = None
		self.val_dataset: SharedASRDataset | None = None
		self.val_loader: DataLoader | None = None

		if getattr(args, "model_class", None):
			self.model_class = resolve_model_class(args.model_class)

	# ---------- setup steps ----------

	def load_config(self) -> None:
		raw_cfg = load_config(self.args.config)
		self.cfg = load_nemo_run_config(raw_cfg)

		tokenizer_dir = Path(self.cfg.model.tokenizer.dir)
		if not tokenizer_dir.exists():
			raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

	def load_tokenizer(self) -> None:
		self.tokenizer = (
			load_spt_tokenizer(self.args.tokenizer_dir)[0]
			if self.args.tokenizer_dir
			else None
		)

	def authenticate_huggingface(self) -> None:
		"""
		Load HF_TOKEN from a .env file (or the environment) and log in to
		Hugging Face Hub, so from_pretrained() can pull gated model repos.
		No-op if HF_TOKEN isn't set anywhere — ungated models still work.
		"""
		load_dotenv()  # populates os.environ from a .env file if present
		token = os.environ.get("HF_TOKEN")
		if token:
			hf_login(token=token)
		else:
			print("No HF_TOKEN found in environment/.env — skipping HF login.")

	def load_model(self) -> None:
		model_name = self.args.pretrained_model or self.cfg.model.get("init_from_pretrained_model")
		if not model_name:
			raise ValueError(
				"Missing pretrained model name. Pass --pretrained_model or set "
				"config.model.init_from_pretrained_model."
			)
		self.authenticate_huggingface()
		# The only line that depends on which model class is in use.
		self.model = self.model_class.from_pretrained(model_name)
		self.model.spec_augmentation = None

	def build_datasets(self) -> None:
		augmentation_cfg = self.cfg.get("augmentation", {})

		self.train_dataset = SharedASRDataset(
			manifest_schema_path=self.args.train_schema,
			training=True,
			feature_key=self.args.feature_key,
			text_key=self.args.text_key,
			feature_base_dir=self.args.feature_base_dir,
			config={"augmentation": augmentation_cfg},
		)

		self.val_dataset = SharedASRDataset(
			manifest_schema_path=self.args.val_schema,
			training=False,
			feature_key=self.args.feature_key,
			text_key=self.args.text_key,
			feature_base_dir=self.args.feature_base_dir,
			config={"augmentation": augmentation_cfg},
		)

	def build_dataloader(
		self,
		dataset: SharedASRDataset,
		*,
		batch_size: int,
		training: bool,
		drop_last: bool,
		sampler=None,
	) -> DataLoader:
		collate_fn = lambda batch: SharedASRDataset.nemo_collate_fn(  # noqa: E731
			batch,
			tokenizer=self.tokenizer,
			training=training,
			config=dataset.config,
		)
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=self.args.num_workers,
			pin_memory=self.args.pin_memory,
			drop_last=drop_last if training else False,
			persistent_workers=self.args.num_workers > 0,
			collate_fn=collate_fn,
			sampler=sampler,
		)

	# ---------- training ----------

	def _train_stage(self, train_loader: DataLoader, trainer: pl.Trainer = None, perform_setup: bool = True) -> str:
		if trainer is None:
			trainer = create_trainer(self.cfg)
			
		return train_nemo(
			model=self.model,
			model_cfg=self.cfg,
			trainer=trainer,
			train_dataloader=train_loader,
			val_dataloader=self.val_loader,
			perform_setup=perform_setup
		)

	def _run_curriculum(self) -> str:
		curriculum_cfg = self.cfg.get("curriculum", {})
		schedule = curriculum_cfg.get("schedule", [0.2, 0.5, 0.7, 1.0])
		score_batch_size = int(curriculum_cfg.get("score_batch_size", self.args.val_batch_size))
		epochs_per_stage = curriculum_cfg.get("epochs_per_stage", None)
		if epochs_per_stage is not None and len(epochs_per_stage) != len(schedule):
			raise ValueError(
                f"curriculum.epochs_per_stage has {len(epochs_per_stage)} entries "
                f"but curriculum.schedule has {len(schedule)}. They must be the same length."
            )
		exp_dir = None
		cumulative_epochs = 0
		base_epochs = self.cfg.trainer.get("max_epochs", 1)
		trainer = create_trainer(self.cfg)
		for stage_idx, active_fraction in enumerate(schedule, start=1):
			if epochs_per_stage is not None:
				stage_epochs = int(epochs_per_stage[stage_idx - 1])
			else:
				stage_epochs = base_epochs
			cumulative_epochs += stage_epochs
			self.cfg.trainer.max_epochs = cumulative_epochs
			trainer.fit_loop.max_epochs = cumulative_epochs
			print(
                f"\n========== Curriculum Stage {stage_idx}/{len(schedule)} "
                f"(active_size={active_fraction:.2f}, additional_epochs={stage_epochs}, cumulative_epochs={cumulative_epochs}) =========="
            )
			score_loader = self.build_dataloader(
                self.train_dataset,
                batch_size=score_batch_size,
                training=False,
                drop_last=False,
            )
			ranked = rank_samples_by_wer(model=self.model, dataloader=score_loader, tokenizer=self.tokenizer)
			ordered_indices = [sample.sample_id for sample in ranked]
			active_size = max(1, int(len(ordered_indices) * active_fraction))
			sampler = CurriculumSampler(ordered_indices, active_size=active_size)
			train_loader = self.build_dataloader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                training=True,
                drop_last=self.args.drop_last,
                sampler=sampler,
            )
			is_first_stage = (stage_idx == 1)
			current_exp_dir = self._train_stage(
                train_loader, 
                trainer=trainer, 
                perform_setup=is_first_stage
            )
			if is_first_stage:
				exp_dir = current_exp_dir
			return exp_dir

	def _run_single_pass(self) -> str:
		train_loader = self.build_dataloader(
			self.train_dataset,
			batch_size=self.args.train_batch_size,
			training=True,
			drop_last=self.args.drop_last,
		)
		return self._train_stage(train_loader)

	def run(self) -> str:
		"""Run the full pipeline end to end and return the experiment directory."""
		self.load_config()
		self.load_tokenizer()
		self.build_datasets()
		self.load_model()  # loaded ONCE; reused across curriculum stages

		self.val_loader = self.build_dataloader(
			self.val_dataset,
			batch_size=self.args.val_batch_size,
			training=False,
			drop_last=False,
		)

		curriculum_enabled = bool(self.cfg.get("curriculum", {}).get("enabled", False))
		if curriculum_enabled:
			return self._run_curriculum()
		return self._run_single_pass()


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
	parser.add_argument(
		"--model_class",
		default=None,
		help=(
			"Dotted path to the NeMo model class to use, e.g. "
			"'nemo.collections.asr.models.EncDecCTCModelBPE' or "
			"'nemo.collections.asr.models.EncDecRNNTBPEModel'. "
			"Defaults to EncDecCTCModelBPE."
		),
	)
	parser.add_argument("--tokenizer_dir", help="Path to the tokenizer directory (optional).")
	parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size.")
	parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size.")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")
	parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory in data loaders.")
	parser.add_argument("--drop_last", action="store_true", help="Drop the last incomplete training batch.")
	return parser


def main() -> str:
	parser = build_arg_parser()
	args = parser.parse_args()
	pipeline = NemoASRPipeline(args)
	return pipeline.run()


if __name__ == "__main__":
	load_dotenv()
	import wandb
	wandb.login(key=os.environ.get("WANDB_API_KEY", None))
	main()