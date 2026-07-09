from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Assuming standard ASR metric utilities or common fallbacks are available
# If editdistance isn't installed, a basic DP Levenshtein function can be swapped in.
try:
    import editdistance
except ImportError:
    editdistance = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_classes.shared_dataset import SharedASRDataset, load_spt_tokenizer
from training.nemo.main import NemoASRPipeline, resolve_model_class


def compute_levenshtein_distance(ref: list | str, hyp: list | str) -> int:
    """Computes minimum edit distance between two sequences."""
    if editdistance:
        return editdistance.eval(ref, hyp)
    
    # Fallback standard DP implementation if external library is missing
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


class NemoValidationPipeline(NemoASRPipeline):
    """
    Runs a standalone validation pass on a pretrained NeMo ASR model
    to compute global Word Error Rate (WER) and Character Error Rate (CER).
    """

    def run_evaluation(self) -> tuple[float, float]:
        """
        Executes the evaluation loop and returns a tuple of (global_wer, global_cer).
        """
        # 1) Setup steps using inherited infrastructure
        self.load_config()
        self.load_tokenizer()
        sample_predicted = []
        sample_references = []
        
        # Build validation dataset exclusively
        augmentation_cfg = self.cfg.get("augmentation", {})
        self.val_dataset = SharedASRDataset(
            manifest_schema_path=self.args.val_schema,
            training=False,
            feature_key=self.args.feature_key,
            text_key=self.args.text_key,
            feature_base_dir=self.args.feature_base_dir,
            config={"augmentation": augmentation_cfg},
        )
        
        self.val_loader = self.build_dataloader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            training=False,
            drop_last=False,
        )
        
        self.load_model()
        # self.model.change_vocabulary(
        #     new_tokenizer_dir=self.cfg.model.tokenizer.dir,
        #     new_tokenizer_type=self.cfg.model.tokenizer.type,
        # )
        
        # 2) Prepare model for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        total_word_edits = 0
        total_words_ref = 0
        total_char_edits = 0
        total_chars_ref = 0

        print(f"Starting pipeline validation evaluation on device: {device}...")

        # 3) Evaluation Loop
        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack standard NeMo feature collate structure
                features, feature_lengths, targets, target_lengths = batch
                
                features = features.to(device)
                feature_lengths = feature_lengths.to(device)

                # Forward pass to obtain log probabilities
                log_probs, encoded_len, _ = self.model(
                    processed_signal=features, 
                    processed_signal_length=feature_lengths
                )

                # Use NeMo's native CTC decoding helper to extract hypotheses strings
                hypotheses = self.model.decoding.ctc_decoder_predictions_tensor(
                    log_probs, 
                    encoded_len
                )

                # Decode target tokens back to reference strings
                targets = targets.cpu().numpy()
                target_lengths = target_lengths.cpu().numpy()
                
                references = []
                for t, t_len in zip(targets, target_lengths):
                    # Strip out padding tokens before decoding
                    valid_tokens = t[:t_len].tolist()
                    ref_text = self.tokenizer.ids_to_text(valid_tokens)
                    references.append(ref_text)

                # 4) Accumulate global distance metrics
                for ref, hyp in zip(references, hypotheses):
                    ref_words = ref.split()
                    hyp_words = hyp.words

                    # Word metrics
                    total_word_edits += compute_levenshtein_distance(ref_words, hyp_words)
                    total_words_ref += len(ref_words)

                    # Character metrics
                    hyp_chars = " ".join(hyp_words)
                    total_char_edits += compute_levenshtein_distance(ref, hyp_chars)
                    total_chars_ref += len(ref)
                    if len(sample_predicted) < 5:  # Sample a few predictions for inspection
                        sample_predicted.append(hyp_chars)
                        sample_references.append(ref)

        # 5) Calculate final global proportions
        global_wer = (total_word_edits / total_words_ref) if total_words_ref > 0 else 0.0
        global_cer = (total_char_edits / total_chars_ref) if total_chars_ref > 0 else 0.0

        print("\n================ Evaluation Results ================")
        print(f"Global Word Error Rate (WER):      {global_wer * 100:.2f}%")
        print(f"Global Character Error Rate (CER): {global_cer * 100:.2f}%")
        print("====================================================\n")

        return global_wer, global_cer, sample_predicted, sample_references


def main() -> tuple[float, float]:
    from training.nemo.main import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args()
    
    val_pipeline = NemoValidationPipeline(args)
    return val_pipeline.run_evaluation()


if __name__ == "__main__":
    global_wer, global_cer, sample_predicted, sample_references = main()
    print("Validation completed successfully.")
    print("Sample predictions: ", sample_predicted)
    print("Sample references: ", sample_references)
    print(f"Final Global WER: {global_wer * 100:.2f}%, CER: {global_cer * 100:.2f}%")
