import math
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")
from utils.curriculun_utils import compute_levenshtein_distance


class ASREvaluator:
    """
    Computes WER, CER, and Compression Efficiency Score (CES) for ASR
    model evaluation, following the Digital Umuganda challenge metrics.
    """

    def __init__(self):
        self.wer: float | None = None
        self.cer: float | None = None
        self.ces: float | None = None

    def compute_wer(self, references: list[str], hypotheses: list[str]) -> float:
        """Word Error Rate over a list of reference/hypothesis transcript pairs."""
        total_edits = 0
        total_words = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words, hyp_words = ref.split(), hyp.split()
            total_edits += compute_levenshtein_distance(ref_words, hyp_words)
            total_words += len(ref_words)
        self.wer = total_edits / total_words if total_words > 0 else 0.0
        return self.wer

    def compute_cer(self, references: list[str], hypotheses: list[str]) -> float:
        """Character Error Rate over a list of reference/hypothesis transcript pairs."""
        total_edits = 0
        total_chars = 0
        for ref, hyp in zip(references, hypotheses):
            total_edits += compute_levenshtein_distance(ref, hyp)
            total_chars += len(ref)
        self.cer = total_edits / total_chars if total_chars > 0 else 0.0
        return self.cer

    def compute_combined_error(self) -> float:
        """CombinedError = 0.4 * WER + 0.6 * CER (Digital Umuganda challenge score)."""
        if self.wer is None or self.cer is None:
            raise ValueError("Call compute_wer() and compute_cer() before compute_combined_error().")
        return 1 - (0.4 * self.wer + 0.6 * self.cer)
    
    def compute_ces(self, size_baseline_bytes: int, size_pruned_bytes: int, cer_baseline: float) -> float:
        """
        CES: resource saving (X) is on-disk model size reduction (bytes),
        used identically for every compression technique -- pruning,
        quantization, or any combination. This is the single, canonical
        definition; there is no separate params/MACs-based variant, so CES
        values remain comparable across techniques regardless of how each
        one achieves its size reduction.
        Uses self.cer (set via compute_cer()) as CER_pruned.
        """
        if self.cer is None:
            raise ValueError("Call compute_cer() before compute_ces() -- it's used as CER_pruned.")

        x = (size_baseline_bytes - size_pruned_bytes) / size_baseline_bytes
        y = max(0.0, 1 - (self.cer - cer_baseline) / cer_baseline)

        self.ces = math.sqrt((1 - x) ** 2 + (1 - y) ** 2)
        return self.ces

    def __to_dict__(self) -> dict:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "ces": self.ces,
            "combined_error": self.compute_combined_error() if self.wer is not None and self.cer is not None else None,
        }