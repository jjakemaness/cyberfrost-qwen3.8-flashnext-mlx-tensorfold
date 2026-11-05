# Benchmark and validation results

All measurements below were collected on 2026-11-05. They describe this specific artifact and runtime configuration, not every Cyber-Frost or Qwen3.8 deployment.

## Final artifact

- Source: `Blackfrost-AI/CYBER-FROST-3.8-BF16`
- Pinned source revision: `3e30d25ddb0e802389d62e55e9ff31ad05f0c96e`
- Architecture: Qwen3.8 Flash Next / `qwen4_exp`
- Quantization: MLX affine 8-bit, global group size 32
- Artifact size: 188.96 GiB / 202,892,816,404 bytes
- Layout: 41 base shards plus `model-mtp.safetensors`
- Index: 3,843 tensors, including 76 embedded `language_model.mtp.*` tensors
- TensorFold admission: passed with native TensorFold 1.2.3
- TensorFold reported startup budget: 217.99 GiB
- TensorFold reported cold-pack budget: 21.65 GiB

## Hardware

- Mac Studio `Mac15,14`
- Apple M3 Ultra
- 256GB unified memory
- Native TensorFold 1.2.3

## Earlier MLX-VLM baseline

The initial Q8 base was served by MLX-VLM 0.7.12 / MLX 0.32.4 without MTP.

- Prompt: `Explain three practical ways to improve home Wi-Fi coverage.`
- Input: 133 tokens
- Output limit: 128 tokens
- Temperature: 0
- Prompt processing: 18.212 tok/s
- Decode: 35.237 tok/s
- Peak memory: 201.826 GB

A matched test with the converter's separate MTP sidecar produced 34.181 tok/s and 45.8% draft-token acceptance, so that sidecar path was not used as the final runtime.

## TensorFold native MTP results

The final checkpoint embeds the MTP tensors in the layout required by TensorFold. Clean live OMP requests with thinking enabled and reasoning effort `xhigh` produced:

| Completion tokens | Decode seconds | Decode speed | Accepted/drafted |
| ---: | ---: | ---: | ---: |
| 143 | 1.246 | 114.0 tok/s | 99/123 (80.5%) |
| 337 | 3.177 | 105.8 tok/s | 228/280 (81.4%) |

Requests observed while OMP had concurrent helpers waiting produced 72.9–77.1 tok/s. Treat 106–114 tok/s as the clean single-request range observed here, not a guaranteed minimum.

### Prefill

Warm incremental prompt sections were observed around 950–1,270 tok/s, including 1,253 and 1,272 tok/s samples. This is cache-assisted incremental prefill and is not directly comparable to the 18.212 tok/s cold 133-token MLX-VLM baseline.

### Context and OMP

- TensorFold server context: 100,000 tokens
- OMP model context window: 100,000 tokens
- Maximum output: 8,192 tokens
- Thinking: enabled
- Reasoning effort: `xhigh`
- Tools: enabled through the OMP provider configuration

## Cybersecurity refusal screen

The model produced 0 refusals across a local 20-prompt cybersecurity behavior screen. This result measures refusal behavior only. It is not a standardized benchmark and does not establish correctness, exploit reliability, safety, or general cyber capability.

## Interpretation

- Clean TensorFold decode was approximately 3× the earlier 35.237 tok/s MLX-VLM baseline.
- The improvement is associated with the TensorFold native Flash Next path and embedded MTP speculative drafting.
- Decode comparisons are more meaningful than the prefill comparison because the prefill workloads and cache states differ.
- Speed changes with concurrency, prompt cache state, context length, thermal state, and draft acceptance.
