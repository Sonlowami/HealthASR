from xml.parsers.expat import model

from dotenv import load_dotenv
from huggingface_hub import login as hf_login
import os
from pathlib import Path
import sys
import torch.nn as nn
from importlib import import_module
from omegaconf import DictConfig, OmegaConf, open_dict
import lightning.pytorch as pl
from nemo.collections.asr.models import EncDecRNNTModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

def authenticate_huggingface() -> None:
		"""
		Load HF_TOKEN from a .env file (or the environment) and log in to
		Hugging Face Hub, so from_pretrained() can pull gated model repos.
		No-op if HF_TOKEN isn't set anywhere — ungated models still work.
		"""
		load_dotenv()
		token = os.environ.get("HF_TOKEN")
		if token:
			hf_login(token=token)
		else:
			print("No HF_TOKEN found in environment/.env — skipping HF login.")
			
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
			
def load_model(model_name, model_class) -> None:
		"""
		Load a model from a local checkpoint or a pretrained model name on Hugging Face."""
		if not model_name:
			raise ValueError(
				"Missing pretrained model name. Pass --pretrained_model or set "
				"config.model.init_from_pretrained_model."
			)
		try:
			authenticate_huggingface()
		except Exception as e:
			print(f"Error occurred while authenticating with Hugging Face: {e}")
		# The only line that depends on which model class is in use.
		if os.path.exists(model_name):
			# If the model name is a local path, load from the local checkpoint.
			model = model_class.restore_from(model_name)
		else:
			# Otherwise, load from a pretrained model name (Hugging Face or NeMo Hub).
			model = model_class.from_pretrained(model_name)
		model.spec_augmentation = None
		return model
		
def create_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    Initialize Lightning Trainer similarly to the notebook:
      trainer = pl.Trainer(**trainer_config, logger=False, enable_checkpointing=False)
    """
    trainer_cfg = OmegaConf.to_container(cfg.get("trainer", {}), resolve=True) or {}

    trainer = pl.Trainer(
        **trainer_cfg,
        logger=False,
        enable_checkpointing=False,
    )
    return trainer

def setup_model(model, cfg: DictConfig, change_vocab=True) -> None:
    """
    Set up the model for training or evaluation.
    """
    model_cfg = model.cfg
    model_cfg.tokenizer.dir = cfg['model']['tokenizer_dir']
    model_cfg.tokenizer.type = cfg['model']['tokenizer_type']
    model_cfg.train_ds.manifest_filepath = cfg['model']['train_ds']['manifest_filepath']
    model_cfg.validation_ds.manifest_filepath = cfg['model']['validation_ds'].get('dav_manifest_filepath')
    model_cfg.decoding.strategy = cfg['model']['decoding']['strategy']
	

	# Clear leftover tarred-dataset config inherited from the pretrained
    # checkpoint (NVIDIA's original training setup used tarred/webdataset
    # shards on their own internal storage — not applicable here).
    for ds_key in ("train_ds", "validation_ds"):
        ds_cfg = model_cfg[ds_key]
        with open_dict(ds_cfg):
            if "is_tarred" in ds_cfg or True:  # force the key to exist either way
                ds_cfg.is_tarred = False
            ds_cfg.tarred_audio_filepaths = None
            if "shard_manifests" in ds_cfg:
                ds_cfg.shard_manifests = False
    if change_vocab:
        model.change_vocabulary(new_tokenizer_dir=model_cfg.tokenizer.dir, new_tokenizer_type=model_cfg.tokenizer.type)
    model.change_decoding_strategy(decoding_cfg=model_cfg.decoding)
    model_cfg.train_ds.batch_size = 6
    model_cfg.validation_ds.batch_size = 6
    model_cfg.train_ds.max_duration = 30
	
    model.setup_training_data(model_cfg.train_ds)
    model.setup_validation_data(model_cfg.validation_ds)
    model_cfg.optim = OmegaConf.create(cfg['model']['optim'])
    model.setup_optimization(optim_config=model_cfg.optim)

def setup_model_for_validation(model, cfg: DictConfig) -> None:
    """
    Set up the model for evaluation (validation), without touching training data.
    """
    model_cfg = model.cfg
    decoding_config = model_cfg.decoding
    decoding_config.strategy = "beam"
    decoding_config.beam.beam_size = 8
    model.change_decoding_strategy(decoding_cfg=decoding_config)
    model_cfg.validation_ds.batch_size = cfg["model"]["validation_ds"].get("batch_size", 16)

def get_hypotheses(model, log_probs_or_encoded, encoded_len):
    """
    Normalize CTC vs RNNT decoding into a single call. CTC's forward()
    already returns log_probs directly decodable via ctc_decoder_predictions_tensor.
    RNNT's forward() only returns encoder output -- decoding to text requires
    a separate call through the model's RNNT decoding module.
    """
    if isinstance(model, EncDecRNNTModel):
        # RNNT: log_probs_or_encoded is actually `encoded` (encoder output),
        # not log_probs -- decode via the joint network / RNNT decoding.
        best_hyp = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=log_probs_or_encoded,
            encoded_lengths=encoded_len,
            return_hypotheses=True,
        )
        best_hyp = model.decoding.decode_hypothesis(best_hyp)
        return best_hyp
    else:
        return model.decoding.ctc_decoder_predictions_tensor(log_probs_or_encoded, encoded_len)


def run_model_forward(model, signal, signal_len):
    """
    Dispatch forward() correctly for CTC (3-tuple) vs RNNT (2-tuple)
    return signatures, and return a normalized (encoder_output, encoded_len)
    pair regardless of model type.
    """
    if isinstance(model, EncDecRNNTModel):
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        return encoded, encoded_len
    else:
        log_probs, encoded_len, _ = model.forward(input_signal=signal, input_signal_length=signal_len)
        return log_probs, encoded_len


class _KwargsForwardWrapper(nn.Module):
    """
    thop calls the profiled module positionally (model(*inputs)), but
    NeMo's forward() is decorated with @typecheck() and requires
    input_signal=/input_signal_length= as keywords. This thin wrapper
    accepts positional args from thop and forwards them as kwargs to the
    real model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_signal, input_signal_length):
        return self.model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
