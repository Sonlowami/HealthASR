import nncf

def build_calibration_dataset_from_loader(val_loader) -> nncf.Dataset:
    if val_loader is None:
        raise RuntimeError(
            "val_loader is None -- a validation dataloader must be set up "
            "before calibration/quantization."
        )

    def transform_fn(batch):
        signal, signal_len, _, _ = batch
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
    model._validation_dl = None

    try:
        model.eval()
        calibration_dataset = build_calibration_dataset_from_loader(saved_validation_dl)
        quantized_model = nncf.quantize(model, calibration_dataset)
    finally:
        model._validation_dl = saved_validation_dl
        for attr, value in instance_overrides.items():
            model.__dict__[attr] = value

    return quantized_model