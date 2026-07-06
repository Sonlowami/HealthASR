import os
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.utils.exp_manager import exp_manager


def load_config(config_filepath: str) -> DictConfig:
    """
    Load a YAML/JSON config file into a DictConfig object.

    Args:
        config_filepath: Path to the config file (YAML or JSON)

    Returns:
        DictConfig: The loaded configuration
    """
    path = Path(config_filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_filepath}")

    # OmegaConf.load supports yaml and json
    cfg = OmegaConf.load(path)
    return cfg


def load_nemo_run_config(config: DictConfig) -> DictConfig:
    """
    Extract and normalize NeMo training config from a DictConfig object.

    Required output fields:
      - model.tokenizer.dir
      - model.optim

    Accepts either:
      A) "already-nested" schema:
         model:
           tokenizer:
             dir: ...
           optim: {...}
      B) "flat" schema:
         tokenizer_dir: ...
         optimizer: {...}

    Args:
        config: DictConfig object containing the configuration

    Returns:
        DictConfig: Normalized config with required NeMo training fields
    """
    # Detect whether config is nested already
    has_nested = (
        "model" in config
        and "optim" in config.model
    )

    if has_nested:
        # Normalize and validate required keys
        required = [
            "model.tokenizer.dir",
            "model.optim",
        ]
        for k in required:
            if OmegaConf.select(config, k) is None:
                raise ValueError(f"Missing required config key: {k}")
        return config

    # Flat schema -> convert to notebook-like nested schema
    mapped = {
        "model": {
            "tokenizer": {"dir": OmegaConf.select(config, "tokenizer_dir")},
            "optim": OmegaConf.select(config, "optimizer")
        },
        # Optional passthrough blocks used in your notebook flow
        "trainer": OmegaConf.select(config, "trainer") or {},
        "exp_manager": OmegaConf.select(config, "exp_manager") or {},
    }
    cfg = OmegaConf.create(mapped)

    required = [
        "model.tokenizer.dir",
        "model.optim",
    ]
    for k in required:
        if OmegaConf.select(cfg, k) is None:
            raise ValueError(
                f"Missing required config value after mapping: {k}. "
                "Expected flat keys: tokenizer_dir, train_manifest_path, "
                "val_manifest_path, train_batch_size, val_batch_size, optimizer."
            )

    return cfg


def create_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Initialize Lightning Trainer similarly to the notebook:
      trainer = pl.Trainer(**trainer_config, logger=False, enable_checkpointing=False)
    """
    trainer_cfg = OmegaConf.to_container(cfg.get("trainer", {}), resolve=True) or {}

    # Keep notebook behavior; exp_manager will attach logger/checkpointing.
    trainer = pl.Trainer(
        **trainer_cfg,
        logger=False,
        enable_checkpointing=False,
    )
    return trainer


def _transfer_dali_outputs_to_device(batch: DALIOutputs, device: torch.device) -> DALIOutputs:
    if batch.has_processed_signal:
        payload = {
            "processed_signal": batch[0].to(device, non_blocking=True),
            "processed_signal_len": batch[1].to(device, non_blocking=True),
            "transcript": batch[2].to(device, non_blocking=True),
            "transcript_len": batch[3].to(device, non_blocking=True),
        }
    else:
        payload = {
            "audio": batch[0].to(device, non_blocking=True),
            "audio_len": batch[1].to(device, non_blocking=True),
            "transcript": batch[2].to(device, non_blocking=True),
            "transcript_len": batch[3].to(device, non_blocking=True),
        }
    transferred = DALIOutputs(payload)
    if hasattr(batch, "sample_indices"):
        transferred.sample_indices = batch.sample_indices
    return transferred


def train_nemo(
    model,
    model_cfg: DictConfig,
    trainer: pl.Trainer,
    train_dataloader=None,
    val_dataloader=None,
) -> str:
    """
    Configure and train a NeMo ASR model (EncDecCTCModelBPE-style), based on notebook flow.

    Steps:
      1) exp_manager(trainer, cfg.exp_manager)
      2) connect config paths/sizes to model cfg
      3) change vocabulary + setup training/validation data
      4) setup optimization
      5) trainer.fit(model)

    Returns:
      exp_dir path from exp_manager
    """
    # 1) Experiment manager
    exp_manager_cfg = model_cfg.get("exp_manager", {})
    exp_dir = exp_manager(trainer, exp_manager_cfg)

    # 2) Update model config
    nemo_cfg = model.cfg
    nemo_cfg.tokenizer.dir = model_cfg.model.tokenizer.dir

    # 3) Tokenizer + datasets
    model.change_vocabulary(
        new_tokenizer_dir=nemo_cfg.tokenizer.dir,
        new_tokenizer_type="bpe",
    )

    # 4) Optimizer
    nemo_cfg.optim = OmegaConf.create(model_cfg.model.optim)
    model.setup_optimization(optim_config=nemo_cfg.optim)

    def _transfer_batch_to_device(self, batch, device, dataloader_idx: int = 0):
        if isinstance(batch, DALIOutputs):
            self._healthasr_last_batch_size = int(batch._outs[0].shape[0])
            return _transfer_dali_outputs_to_device(batch, device)
        return batch

    type(model).transfer_batch_to_device = _transfer_batch_to_device

    original_log = model.log

    def _log_with_default_batch_size(*args, **kwargs):
        if args and args[0] == "global_step" and "batch_size" not in kwargs:
            kwargs["batch_size"] = getattr(model, "_healthasr_last_batch_size", 1)
        return original_log(*args, **kwargs)

    model.log = _log_with_default_batch_size

    # 5) Train
    # NOTE: for NeMo ASR models, trainer.fit(model) is usually enough after setup_* calls.
    # If explicit dataloaders are passed, Lightning accepts them, but NeMo already wires loaders internally.
    if train_dataloader is not None or val_dataloader is not None:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        trainer.fit(model)

    return str(exp_dir) if exp_dir is not None else ""