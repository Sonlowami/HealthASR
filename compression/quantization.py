from torchao.quantization import quantize_, Int8WeightOnlyConfig

def quantize_model(model, config=None):
    if config is None:
        config = Int8WeightOnlyConfig()
    was_training = model.training
    model.eval()
    quantize_(model, config)
    if was_training:
        model.train()
    return model

