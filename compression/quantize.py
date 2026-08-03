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
    model.eval()
    calibration_dataset = build_calibration_dataset(model)
    quantized_model = nncf.quantize(model, calibration_dataset)
    return quantized_model