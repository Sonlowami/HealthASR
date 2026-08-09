import nncf
from nncf.torch.strip import StripFormat
import onnx
import numpy as np
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_utils import KwargsForwardWrapper

def build_calibration_dataset_from_loader(val_loader, device=None) -> nncf.Dataset:
    if val_loader is None:
        raise RuntimeError(
            "val_loader is None -- a validation dataloader must be set up "
            "before calibration/quantization."
        )

    def transform_fn(batch):
        signal, signal_len, _, _ = batch
        if device is not None:
            signal = signal.to(device)
            signal_len = signal_len.to(device)
        return signal, signal_len

    return nncf.Dataset(val_loader, transform_fn)

def build_onnx_calibration_dataset(nemo_model, val_loader, input_names: tuple[str, str], device) -> nncf.Dataset:
    """
    input_names: confirmed via the runtime error to be something like
    ('audio_signal', 'length') -- verify against
    onnx.load(path).graph.input for the exact names before trusting this.
    The exported graph starts AFTER preprocessing (rank-3 mel features),
    not from raw waveform -- must run nemo_model.preprocessor first,
    mirroring NeMo's own RNNT reference script.
    """
    def transform_fn(batch):
        signal, signal_len, _, _ = batch
        signal, signal_len = signal.to(device), signal_len.to(device)
        with torch.no_grad():
            processed_audio, processed_audio_len = nemo_model.preprocessor(
                input_signal=signal, length=signal_len
            )
        return {
            input_names[0]: processed_audio.cpu().numpy(),
            input_names[1]: processed_audio_len.cpu().numpy(),
        }
    return nncf.Dataset(val_loader, transform_fn)

def quantize_onnx_model(model, val_loader, input_names: tuple[str, str], device=None):
    """
    input_names: the ONNX graph's actual input names (check via
    onnx.load('model.onnx').graph.input -- don't assume; NeMo's exporter
    may not name them exactly 'input_signal'/'input_signal_length').
    """
    calibration_dataset = build_onnx_calibration_dataset(model, val_loader, input_names, device)
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        #ignored_scope=nncf.IgnoredScope(patterns=[".*pos_enc.*", ".*featurizer.*"]),
        )
    return quantized_model

def quantize_ctc_onnx(model, onnx_path: str, val_loader, device) -> str:
    onnx_model = onnx.load(onnx_path)
    input_names = tuple(inp.name for inp in onnx_model.graph.input)
    calibration_dataset = build_onnx_calibration_dataset(model, val_loader, input_names, device)
    quantized = nncf.quantize(onnx_model, calibration_dataset)
    quantized_path = onnx_path.replace(".onnx", "_int8.onnx")
    onnx.save(quantized, quantized_path)
    return quantized_path


def quantize_rnnt_onnx(encoder_path: str, decoder_joint_path: str, nemo_model, val_loader, device) -> tuple[str, str]:
    """
    Encoder: same calibration shape as CTC's single graph -- quantized normally.
    decoder_joint: called autoregressively inside ONNXGreedyBatchedRNNTInfer's
    per-step decode loop with carried hidden state -- its calibration input
    shape isn't yet confirmed against that class's actual session.run() calls
    (flagged as open work last turn). NOT quantized here yet -- returned
    as-is, so results stay honest about what was actually compressed rather
    than silently claiming full RNNT quantization.
    """
    quantized_encoder_path = quantize_ctc_onnx(nemo_model, encoder_path, val_loader, device)
    print("  NOTE: decoder_joint graph is NOT yet quantized (calibration shape "
          "unconfirmed for its autoregressive per-step inputs) -- only the "
          "encoder was compressed. RNNT compression numbers currently "
          "understate potential savings.")
    return quantized_encoder_path, decoder_joint_path


def quantize_model(model):
    """
    nncf.quantize()'s internal deepcopy(model) chokes on wrapt-based
    FunctionWrapper/BoundFunctionWrapper objects living as instance
    attributes: model.training_step (monkeypatched elsewhere in the
    pipeline) and model._validation_dl.collate_fn (NeMo's own
    typecheck-decorated collate function). Strip both from the MODEL
    object before quantizing -- calibration_dataset already holds its
    own independent reference to the dataloader, so nulling
    model._validation_dl doesn't affect calibration itself, only what
    deepcopy(model) has to walk.
    """
    instance_overrides = {}
    for attr in ("training_step", "trainer", "log"):
        if attr in model.__dict__:
            instance_overrides[attr] = model.__dict__.pop(attr)

    saved_validation_dl = model._validation_dl
    sample_batch = next(iter(saved_validation_dl))
    signal, signal_len, _, _ = sample_batch
    device = next(model.parameters()).device
    signal, signal_len = signal.to(device), signal_len.to(device)

    model._validation_dl = None
    device = next(model.parameters()).device

    try:
        model.eval()
        calibration_dataset = build_calibration_dataset_from_loader(saved_validation_dl, device=device)
        # Set maximum length so nncf does not retrigger statistics computation when it meets a longer sequence
        MAX_SEQ_LENGTH = 10000
        model.encoder.update_max_seq_length(seq_length=MAX_SEQ_LENGTH, device=device)


        wrapped = KwargsForwardWrapper(model)
        # Ignore positional encoding layers during quantization.
        # Why? Because nncf + nemo is quite a headache.
        # nncf expects a tensor with dim >= 2. It tried to compute statistics for 1-d tensors from pe, triggering IndexOutOfRangeError.
        quantized_wrapper = nncf.quantize(
            wrapped,
            calibration_dataset,
            ignored_scope=nncf.IgnoredScope(patterns=[".*pos_enc.*", ".*featurizer.*"]),
            )
        # nncf doesn't work well stripping nemo models.
        # quantized_wrapper = nncf.strip(
        #     quantized_wrapper,
        #     example_input=example_input,
        #     strip_format=StripFormat.DQ
        #     )
        model = quantized_wrapper.model
    except Exception as e:
        print(f"Quantization failed: {e}")
        raise e

    finally:
        model._validation_dl = saved_validation_dl
        for attr, value in instance_overrides.items():
            model.__dict__[attr] = value

    return model