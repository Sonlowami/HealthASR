import nncf
import sys
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
    example_input = (signal, signal_len)

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
            ignored_scope=nncf.IgnoredScope(patterns=[".*pos_enc.*"]),
            )
        quantized_model = nncf.strip(
            quantized_wrapper,
            example_input=example_input,
            strip_format=nncf.torch.strip.StripFormat.DQ
            )
        model = quantized_model.model
    except Exception as e:
        print(f"Quantization failed: {e}")
        raise e

    finally:
        model._validation_dl = saved_validation_dl
        for attr, value in instance_overrides.items():
            model.__dict__[attr] = value

    return model