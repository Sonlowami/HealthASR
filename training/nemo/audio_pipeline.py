import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

import utils.model_utils as model_utils
import argparse
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf

class AudioNemoTrainer:
    def __init__(self, model_name: str | None, model_class: str, cfg: dict):
        self.model_name = model_name or cfg.get("model", {}).get("init_from_pretrained_model")
        self.model_class = model_utils.resolve_model_class(model_class)
        self.cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        self.trainer = model_utils.create_trainer(self.cfg)
        self.model = None
        self._is_setup = False

    def setup_model(self):
        """
        Set up the model for training or evaluation.
        """
        if self._is_setup:
            return self.model
        model = model_utils.load_model(self.model_name, self.model_class)
        model_utils.setup_model(model, self.cfg)
        self.model = model
        self._is_setup = True
    
    def train_nemo(self):
        """
        Train the model using the provided trainer.
        """
        exp_dir = exp_manager(self.trainer, cfg=self.model.cfg.exp_manager)
        self.trainer.fit(self.model)
        return exp_dir

    def train(self):
        """
        Train the model.
        """
        self.setup_model()
        exp_dir = self.train_nemo()
        return exp_dir
    

    def evaluate(self):
        model = self.setup_modelw()
        results = self.trainer.validate(model)
        return results[0] if results else {}
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Train a NeMo model.")
    parser.add_argument("--pretrained_model", type=str, required=False, default=None, help="Name of the model to load.")
    parser.add_argument("--model_class", type=str, required=True, help="Dotted path to the model class.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    trainer = AudioNemoTrainer(args.pretrained_model, args.model_class, cfg)
    print(trainer.evaluate())
    #print(f"Training completed. Experiment directory: {exp_dir}")