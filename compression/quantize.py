import nncf


def build_calibration_dataset(model) -> nncf.Dataset:
    def transform_fn(batch):
        signal, signal_len, _, _ = batch
        # positional order must match EncDecCTCModelBPE.forward's first two
        # params (input_signal, input_signal_length) -- NNCF calls
        # model(*transform_fn(batch)) during calibration.
        return signal, signal_len

    return nncf.Dataset(model._validation_dl, transform_fn)

def quantize_model(model):
    """
    nncf.quantize()'s internal deepcopy() can't handle a wrapt.FunctionWrapper
    stored as an instance attribute (model.training_step here appears to be
    monkeypatched with one, likely by decorator-based instrumentation. I couldn't find where this happened).
    Strip any such instance-level overrides
    before quantizing, restore them after.
    """
    instance_overrides = {}
    for attr in ("training_step", "trainer", "log"):
        if attr in model.__dict__:
            instance_overrides[attr] = model.__dict__.pop(attr)

    try:
        model.eval()
        calibration_dataset = build_calibration_dataset(model)
        quantized_model = nncf.quantize(model, calibration_dataset)
    finally:
        for attr, value in instance_overrides.items():
            model.__dict__[attr] = value

    return quantized_model