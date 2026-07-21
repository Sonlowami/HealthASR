import sys
from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

import utils.model_utils as model_utils
import argparse
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf, open_dict
import utils.curriculun_utils as cutils

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
    
    def train_nemo(self, setup_exp_manager: bool = True):
        """
        Train the model using the provided trainer.
        """
        exp_dir = exp_manager(self.trainer, cfg=self.cfg.exp_manager) if setup_exp_manager else None
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
        self.setup_model()
        val_dataloader = self.model._validation_dl
        results = self.trainer.validate(dataloaders=val_dataloader)
        return results[0] if results else {}
    

class CurriculumAudioNemoTrainer(AudioNemoTrainer):
    def __init__(self, model_name, model_class, cfg):
        super().__init__(model_name, model_class, cfg)
        self.curriculum_cfg = cfg.get("curriculum", {})

    def run_curriculum(self):
        self.setup_model()
        schedule = self.curriculum_cfg.get("schedule", [0.2, 0.5, 0.7, 1.0])
        epochs_per_stage = self.curriculum_cfg.get("epochs_per_stage")
        warmup_epochs = int(self.curriculum_cfg.get("warmup_epochs", 0))
        score_batch_size = self.curriculum_cfg.get("score_batch_size", 16)
        base_manifest = self.cfg["model"]["train_ds"]["manifest_filepath"]

        # Deterministic epoch boundaries — same every run, since schedule/
        # epochs_per_stage/warmup_epochs are static config values.
        boundaries = []  # list of (label, start_epoch, end_epoch)
        cursor = 0
        if warmup_epochs > 0:
            boundaries.append(("warmup", cursor, cursor + warmup_epochs))
            cursor += warmup_epochs
        for i, frac in enumerate(schedule):
            end = cursor + int(epochs_per_stage[i])
            boundaries.append((f"stage_{i+1}", cursor, end))
            cursor = end

        # Check for a checkpoint from a previous (crashed/interrupted) run.
        resume_epoch = None
        if self.cfg.exp_manager.get("resume_if_exists", False):
            ckpt_dir = Path(self.cfg.exp_manager.get("explicit_log_dir"))
            resume_epoch, resume_ckpt_path = cutils.find_last_checkpoint(str(ckpt_dir))
            if resume_epoch is not None:
                print(f"Found checkpoint at epoch {resume_epoch} — will skip already-completed "
                    f"curriculum stages and preload weights for the stage still in progress.")
                device = self.trainer.strategy.root_device
                cutils.preload_weights(self.model, resume_ckpt_path, device)

        first_fit_call = True
        exp_dir = None

        for label, start_epoch, end_epoch in boundaries:
            print(f"\n=== Curriculum {label} (epochs {start_epoch}-{end_epoch}) ===")
            if resume_epoch is not None and end_epoch <= resume_epoch:
                print(f"Skipping {label} (epochs {start_epoch}-{end_epoch}) — already completed.")
                continue

            print(f"\n=== Curriculum {label} (epochs {start_epoch}-{end_epoch}) ===")

            if label != "warmup":
                # score using CURRENT weights — either freshly initialized
                # (genuine fresh run) or preloaded above (resumed run)
                ranked = cutils.score_manifest(self.model, self.trainer, base_manifest, batch_size=score_batch_size)
                stage_manifest = f"/tmp/curriculum_{label}.jsonl"
                active_fraction = schedule[int(label.split("_")[1]) - 1]
                cutils.write_stage_manifest(ranked, active_fraction, stage_manifest)

                stage_ds_cfg = copy.deepcopy(self.model.cfg.train_ds)
                with open_dict(stage_ds_cfg):
                    stage_ds_cfg.manifest_filepath = stage_manifest
                self.model.setup_training_data(stage_ds_cfg)

            self.trainer.fit_loop.max_epochs = end_epoch
            exp_dir = self.train_nemo(setup_exp_manager=first_fit_call)
            first_fit_call = False

        return exp_dir

    # def run_curriculum(self):
    #     self.setup_model()
    #     schedule = self.curriculum_cfg.get("schedule", [0.2, 0.5, 0.7, 1.0])
    #     epochs_per_stage = self.curriculum_cfg.get("epochs_per_stage")
    #     score_batch_size = self.curriculum_cfg.get("score_batch_size", 16)
    #     warmup_epochs = int(self.curriculum_cfg.get("warmup_epochs", 0))
    #     base_manifest = self.cfg["model"]["train_ds"]["manifest_filepath"]

    #     cumulative_epochs = 0
    #     exp_dir = None
    #     is_first_train = True

    #     # --- 1. Optional Warmup Phase ---
    #     if warmup_epochs > 0:
    #         print(f"\n=== Curriculum Warmup: Training on full dataset for {warmup_epochs} epochs ===")
    #         cumulative_epochs += warmup_epochs
    #         self.trainer.fit_loop.max_epochs = cumulative_epochs
            
    #         # Initialize exp_manager only on the very first fit
    #         if is_first_train:
    #             exp_dir = exp_manager(self.trainer, cfg=self.cfg.exp_manager)
    #             is_first_train = False
                
    #         self.trainer.fit(self.model)

    #     # --- 2. Curriculum Stages ---
    #     for stage_idx, active_fraction in enumerate(schedule, start=1):
    #         print(f"\n=== Curriculum stage {stage_idx}/{len(schedule)} (fraction={active_fraction}) ===")

    #         # Score using the model's current weights
    #         ranked = cutils.score_manifest(self.model, self.trainer, base_manifest, batch_size=score_batch_size)

    #         stage_manifest = f"/tmp/curriculum_stage_{stage_idx}.jsonl"
    #         cutils.write_stage_manifest(ranked, active_fraction, stage_manifest)

    #         # Update the dataset dynamically for this stage
    #         stage_ds_cfg = copy.deepcopy(self.model.cfg.train_ds)
    #         with open_dict(stage_ds_cfg):
    #             stage_ds_cfg.manifest_filepath = stage_manifest
    #         self.model.setup_training_data(stage_ds_cfg)

    #         # Accumulate epochs and update the trainer
    #         cumulative_epochs += int(epochs_per_stage[stage_idx - 1])
    #         self.trainer.fit_loop.max_epochs = cumulative_epochs

    #         # Initialize exp_manager if warmup was skipped and this is the first run
    #         if is_first_train:
    #             exp_dir = exp_manager(self.trainer, cfg=self.cfg.exp_manager)
    #             is_first_train = False

    #         self.trainer.fit(self.model)

    #     return exp_dir
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Train a NeMo model.")
    parser.add_argument("--pretrained_model", type=str, required=False, default=None, help="Name of the model to load.")
    parser.add_argument("--model_class", type=str, required=True, help="Dotted path to the model class.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.curriculum:
        print("Using curriculum learning...")
        trainer = CurriculumAudioNemoTrainer(args.pretrained_model, args.model_class, cfg)
        print(trainer.run_curriculum())
    else:
        trainer = AudioNemoTrainer(args.pretrained_model, args.model_class, cfg)
    print(trainer.train())
    #print(f"Training completed. Experiment directory: {exp_dir}")