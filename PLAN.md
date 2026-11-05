# Conversion and release record

This document records how the TensorFold-compatible Cyber-Frost Qwen3.8 Flash Next MLX Q8 artifact was produced. Commands use placeholder paths; do not place model weights in this Git repository.

## Source

- Repository: `Blackfrost-AI/CYBER-FROST-3.8-BF16`
- Pinned revision: `3e30d25ddb0e802389d62e55e9ff31ad05f0c96e`
- Source size: approximately 335.3 GiB across 131 SafeTensors shards
- Architecture: `Qwen4ExpForConditionalGeneration` / `qwen4_exp`
- License: Qwen Community License 1.0

The source download was verified against the pinned Hugging Face revision before conversion.

## Conversion environment

- Host: Apple M2 Max, 96GB unified memory
- MLX-VLM: 0.7.12
- MLX: 0.32.4
- Transformers: 5.22.1
- SafeTensors: 0.8.0

MLX-VLM 0.6.3 did not include the required `qwen4_exp` implementation in the tested environment, so the conversion used 0.7.12.

## MLX Q8 conversion

```bash
mlx_vlm.convert \
  --hf-path /path/to/CYBER-FROST-3.8-BF16 \
  --mlx-path /path/to/CYBER-FROST-3.8-MLX-8bit \
  --quantize \
  --q-bits 8 \
  --q-group-size 32 \
  --dtype bfloat16 \
  --trust-remote-code \
  --mtp \
  --mtp-output /path/to/CYBER-FROST-3.8-MLX-8bit-mtp
```

The converter reported 9.018 effective bits per weight. It emitted:

- Base model: 41 shards, 3,767 indexed tensors, approximately 186 GiB
- MTP sidecar: 76 tensors, approximately 2.8 GiB

The global quantization metadata is affine 8-bit/group-32. The Qwen4Exp predicate uses affine 8-bit/group-64 for router/shared-expert gates, group-32 for PLE shards, and skips multimodal modules.

## Why the MTP sidecar was embedded

The generic base passed MLX-VLM structural loading, but TensorFold native serving rejected it with `MissingFlashTensor`. TensorFold's Flash Next implementation expects native MTP tensors inside the target checkpoint rather than attached as an external sidecar.

The repository tool performs that packaging step:

```bash
python3 tools/embed_tensorfold_mtp.py \
  /path/to/CYBER-FROST-3.8-MLX-8bit \
  /path/to/CYBER-FROST-3.8-MLX-8bit-mtp
```

It:

1. Reads the 76 sidecar tensor headers.
2. Prefixes the tensor names with `language_model.mtp.`.
3. Writes `model-mtp.safetensors` without modifying tensor payloads.
4. Adds those tensors to `model.safetensors.index.json`.
5. Refuses to overwrite an existing embedded MTP file.

## Structural validation

- Every base shard header matched the weight index.
- All 3,767 base tensors were accounted for.
- All 76 MTP tensors were embedded and indexed.
- Final index total: 3,843 tensors.
- Lazy strict-load validation passed for the base and the original MLX-VLM MTP sidecar.
- TensorFold native 1.2.3 format admission passed after embedding.

```bash
/opt/homebrew/bin/tensorfold info /path/to/final-checkpoint
```

Expected summary:

```text
family: Qwen3.8 Flash Next (qwen4_exp)
format: 8-bit, group 32
weights: 188.96 GiB (202892816404 bytes)
startup budget: 217.99 GiB
cold pack budget: 21.65 GiB
```

## Studio runtime

The final artifact was transferred to an Apple M3 Ultra Mac Studio with 256GB unified memory and served with native TensorFold 1.2.3. The public launcher is [`scripts/serve-tensorfold.sh`](scripts/serve-tensorfold.sh).

The tested profile uses:

- `FLASHNEXT_LOAD_LIMIT_GB=240`
- 100,000-token context
- 8,192-token maximum output
- thinking enabled
- `xhigh` reasoning effort
- loopback port 18080

OMP configuration examples are in [`examples/`](examples/). Benchmark results and limitations are in [`RESULTS.md`](RESULTS.md).

## Release checklist

- [x] Pin and verify source revision
- [x] Convert BF16 to MLX Q8
- [x] Validate base and MTP tensor structure
- [x] Embed native MTP tensors for TensorFold
- [x] Pass TensorFold 1.2.3 format admission
- [x] Load and generate on M3 Ultra 256GB
- [x] Exercise 100K OMP context configuration with tools and thinking
- [x] Record decode, prefill, acceptance, memory, and concurrency caveats
- [x] Include upstream license and attribution
- [x] Publish scripts separately from model weights
