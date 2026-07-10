from dotenv import load_dotenv
from huggingface_hub import login as hf_login
import os
from pathlib import Path
import sys
from importlib import import_module
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

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
		authenticate_huggingface()
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

def setup_model(model, cfg: DictConfig) -> None:
    """
    Set up the model for training or evaluation.
    """
    model_cfg = model.cfg
    model_cfg.tokenizer.dir = cfg['model']['tokenizer_dir']
    model_cfg.tokenizer.type = cfg['model']['tokenizer_type']
    model_cfg.train_ds.manifest_filepath = cfg['model']['train_ds']['manifest_filepath']
    model_cfg.validation_ds.manifest_filepath = cfg['model']['validation_ds']['manifest_filepath']
	
    model.change_vocabulary(new_tokenizer_dir=model_cfg.tokenizer.dir, new_tokenizer_type=model_cfg.tokenizer.type)
    model_cfg.train_ds.batch_size = 6
    model_cfg.validation_ds.batch_size = 6
    model_cfg.train_ds.max_duration = 30
	
    model.setup_training_data(model_cfg.train_ds)
    model.setup_validation_data(model_cfg.validation_ds)
    model_cfg.optim = OmegaConf.create(cfg['model']['optim'])
    model.setup_optimization(optim_config=model_cfg.optim)

	