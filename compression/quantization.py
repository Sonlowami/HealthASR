from torchao.quantization import quantize_, Int8WeightOnlyConfig

def quantize_model(model, config=None):
    """
    In-place weight quantization via torchao.
    walks named_modules() directly and swaps matched layers' weights for
    a quantized tensor subclass. Default filter_fn only matches nn.Linear;
    Conv1d/Conv2d layers (encoder subsampling, depthwise conv modules)
    are NOT touched unless a custom filter_fn is passed.
    """
    if config is None:
        config = Int8WeightOnlyConfig()
    quantize_(model, config)
    return model

