# Evaluation pipeline

This directory contains the evaluation entrypoint for the ASR pipeline. The main script in [evaluation/main.py](main.py) can be used in two modes:

1. Normal inference/evaluation
2. Quantization-aware training (QAT) followed by optional fine-tuning and re-evaluation

The pipeline restores a NeMo model checkpoint, prepares validation data for the requested languages, runs inference, and reports WER, CER, and compression efficiency metrics.

## What the pipeline does

For each model supplied via `--model_paths`, the script:

- loads the checkpoint through the configured model class,
- sets up validation data for the requested languages,
- runs inference over the validation loader,
- computes WER and CER,
- optionally applies quantization-aware evaluation and fine-tuning,
- writes a JSON report to the requested output directory.

The output structure is written to `results.json` and contains:

- a baseline section with model size, parameter count, MACs estimate (when available), and language-level metrics,
- a quantization section with the results for the configured QAT variants.

## Normal inference mode

Use the evaluation script without the quantization flags when you only want to measure a checkpoint's baseline performance.

Typical flow:

1. Restore the model from the checkpoint.
2. Set up validation data for the selected languages.
3. Run inference on the validation set.
4. Compute WER/CER.
5. Optionally compare against a baseline file and compute CES.

This is the default behavior when `--quantize` is not provided.

## Quantization-aware training (QAT) flow

When `--quantize` is enabled, the script switches into a QAT-oriented workflow. The intent is to measure the model before quantization, simulate quantization during training, optionally fine-tune the model under that simulation, and then persist the learned weights back to a plain linear model with a real quantized configuration.

### High-level flow

1. Load the checkpoint and measure its baseline performance.
2. Apply QAT preparation to the model.
3. Measure performance under the fake-quantized configuration.
4. If `--finetune` is supplied, fine-tune the QAT-prepared model.
5. Convert the fine-tuned weights back into a plain model and apply the corresponding plain quantization configuration for persistence.
6. Measure the persisted quantized model again.
7. Report the baseline and post-quantization results.

This is the workflow the code implements:

- get model
- measure performance
- do QAT
- optionally fine-tune
- convert back to linear for persisting quantization
- measure again
- report findings

## How QAT works

Quantization-aware training makes the forward pass behave as if weights and/or activations are quantized, while keeping the underlying parameters in floating point so gradients can still flow.

A simple way to think about it is:

$$
\tilde{w} = s \cdot \mathrm{clip}\left(\mathrm{round}\left(\frac{w}{s}\right), q_{\min}, q_{\max}\right)
$$

where:

- $w$ is the real-valued weight,
- $s$ is the scale,
- $q_{\min}$ and $q_{\max}$ define the quantization range,
- $\tilde{w}$ is the fake-quantized value used during the forward pass.

The important detail is that the model still learns with floating-point weights, but the forward path is exposed to quantization effects. That usually gives a better outcome than applying quantization only after training.

## Current QAT configuration

The current configuration in the code uses two fake-quantization setups:

- `float8_weight_qat`: uses `Float8FakeQuantizeConfig`
- `int8_weight_qat`: uses `IntxFakeQuantizeConfig(torch.int8, "per_channel", is_symmetric=True)

Both are wrapped in `QATConfig` with `step="prepare"`, which is the prepare-time QAT stage used by the evaluation entrypoint.

## Persisting quantized weights

After fine-tuning, the script does not persist the fake-quantized wrapper directly. Instead, it:

- extracts the fine-tuned weights from the QAT-prepared model,
- loads them into a fresh plain model,
- applies the corresponding plain quantization configuration for persistence.

The base quantization configs used for persistence are:

- `Float8WeightOnlyConfig` for the float8 path
- `Int8WeightOnlyConfig` for the int8 path

This keeps the persisted model in a regular quantized form rather than a fake-quantized training wrapper.

## Command-line usage

The entrypoint is invoked through the script in [evaluation/main.py](main.py). The main flags are:

- `--model_paths`: one or more checkpoint paths
- `--model_class`: dotted path to the model class
- `--config`: NeMo config used for validation setup
- `--languages`: language names or codes to evaluate
- `--baseline_file`: optional baseline JSON for CES comparison
- `--output_dir`: directory for `results.json`
- `--quantize`: enable QAT preparation and evaluation
- `--finetune`: fine-tune the QAT-prepared model before persistence and re-evaluation

Example shape:

```bash
python evaluation/main.py \
  --model_paths /path/to/model.nemo \
  --model_class <your.model.Class> \
  --config /path/to/config.yaml \
  --languages kinyarwanda english \
  --output_dir ./results \
  --quantize --finetune
```

## Notes

- The evaluation script uses the model’s validation data setup from the provided NeMo config.
- The metrics are computed with the same evaluator logic used elsewhere in the project.
- The exact quantization and persistence behavior is implemented in [evaluation/main.py](main.py) and [compression/quantization.py](../compression/quantization.py).
