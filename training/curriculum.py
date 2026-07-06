from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from jiwer import wer as compute_wer
from torch.utils.data import Sampler


@dataclass(frozen=True)
class RankedSample:
    sample_id: int
    wer: float


def rank_samples_by_wer(model, dataloader, tokenizer) -> List[RankedSample]:
    """Rank samples in ascending WER using an already-loaded ASR model."""
    model.eval()
    ranked: List[RankedSample] = []
    device = model.device

    with torch.no_grad():
        for batch in dataloader:
            sample_ids = getattr(batch, "sample_indices", None)
            if sample_ids is None:
                raise ValueError("Curriculum scoring requires batch.sample_indices to be present.")

            if batch.has_processed_signal:
                signal = batch[0].to(device, non_blocking=True)
                signal_len = batch[1]
                references = batch[2].to(device, non_blocking=True)
                reference_len = batch[3]
                _, encoded_len, greedy_predictions = model.forward(
                    processed_signal=signal, processed_signal_length=signal_len
                )
            else:
                signal = batch[0].to(device, non_blocking=True)
                signal_len = batch[1]
                references = batch[2].to(device, non_blocking=True)
                reference_len = batch[3]
                _, encoded_len, greedy_predictions = model.forward(input_signal=signal, input_signal_length=signal_len)

            predicted_ids = greedy_predictions

            for row_idx, sample_id in enumerate(sample_ids):
                ref_ids = references[row_idx][: int(reference_len[row_idx].item())].tolist()
                hyp_ids = predicted_ids[row_idx][: int(encoded_len[row_idx].item())].tolist()
                ref_text = tokenizer.ids_to_text(ref_ids)
                hyp_text = tokenizer.ids_to_text(hyp_ids)
                ranked.append(RankedSample(sample_id=int(sample_id), wer=float(compute_wer(ref_text, hyp_text))))

    ranked.sort(key=lambda item: item.wer)
    return ranked


class CurriculumSampler(Sampler[int]):
    """Sampler that yields a prefix of pre-ranked sample ids."""

    def __init__(self, ordered_indices: Sequence[int], active_size: int):
        self.ordered_indices = list(ordered_indices)
        self.active_size = int(active_size)

    def __iter__(self):
        yield from self.ordered_indices[: self.active_size]

    def __len__(self):
        return min(self.active_size, len(self.ordered_indices))
