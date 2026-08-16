# Qwen3.8-27B NVFP4 weight audit

This report records what was checked before accepting the local
`Qwen3.8-27B-NVFP4-Corrected` checkpoint. It is an audit of exact checkpoint bytes,
quantization semantics, and the selected production kernel. It is not a claim that
quantized inference is mathematically identical to BF16 inference.

## Immutable inputs

| Input | Pinned identity |
|---|---|
| Official reference | `Qwen/Qwen3.8-27B` |
| Official revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Converted source | `unsloth/Qwen3.8-27B-NVFP4` |
| Converted revision | `16b6615af3548b88e2d8e382457bc705b00479cf` |
| Corrected model file | `5fd70b38b3708e47adc1e9e9ab90f5d688ec01177d0718fdd16678696fdb0988` |
| Corrected manifest | `3a86177c30b97035d27ad0cf516fc4c2ddb83701c4de4fc6adcb23c7c2531bfc` |
| Full audit JSON | `audit-results/source-weight-audit-16b6615a-vs-1d4bf0f2.json` |
| Full audit JSON SHA-256 | `5404c27fc14d0e0ca8caffb9459d72521b6150811eb09d6d7d2839c9bf28093a` |

Both source trees were verified against their checked-in manifests before any
comparison. The auditor opens safetensors directly and does not trust tensor names,
quantization metadata, or publisher prose in isolation.

## Complete tensor accounting

The official checkpoint has 1,199 tensors. The converted checkpoint has 1,968:

- 798 tensors represented directly at reference precision;
- 233 FP8 weight tensors and 233 FP8 scale tensors;
- 168 packed NVFP4 weight tensors and 504 NVFP4 metadata tensors;
- 32 static KV-calibration tensors.

Those sets are disjoint and account for every converted model tensor. The audit
dequantized every FP8 and NVFP4 tensor and compared it with the same official BF16
tensor. This was a full conversion audit, not sparse sampling.

## FP8 result

The stored semantics are E4M3 values multiplied by one floating scale per output
channel. The scale tensors are BF16 in this snapshot.

Across 10,624,696,320 dequantized elements:

| Metric | Result |
|---|---:|
| Relative L2 error | 0.0266015023 |
| Cosine similarity | 0.9996461412 |
| Reference RMS | 0.0149812568 |
| Approximation RMS | 0.0149726860 |
| Mean absolute error | 0.0002646446 |
| Maximum absolute error | 0.0546875 |
| E4M3 endpoint fraction | 0.0002779797 |

The low endpoint fraction argues against widespread FP8 clipping. It does not turn
FP8 into a lossless representation.

## NVFP4 result

The checkpoint uses low-nibble-first E2M1 values in groups of 16. Each group has an
E4M3 block scale and each tensor has a global scale divisor. The independent audit
unpacks the nibbles explicitly and computes:

`dequantized = E2M1(code) * E4M3(block_scale) / global_scale`

Across all 168 NVFP4 matrices:

| Metric | Result |
|---|---:|
| Relative L2 error | 0.1082005947 |
| Cosine similarity | 0.9941543195 |
| E2M1 endpoint fraction | 0.1113897493 |
| E4M3 scale endpoint fraction | 0.0000001592 |

The worst matrix by relative L2 was
`model.language_model.layers.0.mlp.down_proj.weight`, at 0.1534641 relative L2 and
0.988319 cosine similarity. This deliberately became the production-kernel test
matrix; choosing an easy layer would weaken the test.

The NVFP4 comparison establishes correct format interpretation and credible
alignment with the official weights. The error is real and materially larger than
FP8. No benchmark, perplexity, or task-quality result should be inferred from these
matrix metrics alone.

## The offset-RMSNorm defect and correction

Of the 798 tensors nominally retained at reference precision, 637 were byte-identical
to the official checkpoint. The other 161 were exactly the ordinary language-model
RMSNorm tensors:

- 64 input layer norms;
- 64 post-attention layer norms;
- 16 query norms and 16 key norms in the full-attention layers;
- the final language-model norm.

Qwen3.5-family RMSNorm stores an offset `w` and applies gain `1 + w`. The conversion
path temporarily materialized `1 + w` in BF16, then subtracted one. That BF16
round-trip loses low bits. Comparing effective gains, the aggregate relative L2 was
0.00186876, cosine similarity was 0.999998256, and maximum absolute difference was
0.0078125. Small is not the same as intended, so the unmodified converted snapshot
is retained only as source evidence and is not served.

`scripts/repair-offset-norms.py` creates the deployable tree by:

1. verifying the complete Unsloth and official manifests;
2. copying the source snapshot to a private temporary sibling;
3. locating the exact safetensors byte range for each of the 161 expected names;
4. requiring BF16 dtype, matching shapes and equal byte lengths;
5. replacing only those ranges with official BF16 bytes;
6. proving by an independently streamed synthetic digest that no other byte differs;
7. verifying the complete corrected manifest; and
8. publishing with an atomic directory rename.

It does not deserialize and rewrite the 23 GB file, modify quantized projections, or
accept a partial result. Existing targets, symlinks, unknown layouts, missing names,
changed manifests, and any extra byte change are hard errors. The wrapper
`scripts/repair-model.sh` runs it with no network inside the immutable pinned Docker
image; no Python dependency is installed on the host.

## Production NVFP4 kernel test

`scripts/nvfp4_kernel_unit.py` loads the complete worst-error matrix directly from
the corrected checkpoint: shape 5,120 by 17,408, weight divisor 2,752 and input
divisor 161. On the RTX 5090 it requires vLLM to select
`FlashInferCutlassNvFp4LinearKernel`; a different kernel is a failure.

For batches M=1, 17, and 129, the test independently unpacks the actual checkpoint
weights, independently dequantizes the FlashInfer-swizzled activation, performs the
BF16 reference matmul, and compares the production kernel output:

| M | Max abs | Relative L2 | Cosine | Elements outside `atol=.1, rtol=.1` |
|---:|---:|---:|---:|---:|
| 1 | 0.125 | 0.003161705 | 0.999995470 | 0 |
| 17 | 0.25 | 0.003603501 | 0.999993920 | 0 |
| 129 | 1.0 | 0.003809735 | 0.999993205 | 0.000003040 |

This proves that the selected SM 12.0 production kernel implements the checkpoint's
stored NVFP4 arithmetic to tight BF16 numerical tolerance across decode-like and
prefill-like shapes. It does not prove that NVFP4 matches an official BF16 model's
answers. That remaining difference is the intended weight quantization tradeoff.

## KV metadata is not the runtime KV format

The 32 checkpoint KV-scale tensors are BF16 calibration metadata for a static FP8
cache path. The selected runtime uses TurboQuant K8V4, which dynamically stores FP8
keys and four-bit values with per-vector metadata. The checkpoint KV scales neither
pre-quantize the cache nor participate in the deployed TurboQuant kernels.

## Reproduction

The heavyweight source audit requires both fully downloaded snapshots and is run
inside the pinned Docker dependency boundary. Its committed result is the JSON file
above. The deployable correction is recreated with:

```text
./scripts/repair-model.sh
```

The immutable runtime embeds the GPU kernel audit at
`/opt/qwen38/nvfp4_kernel_unit.py`. It must be run with the corrected model mounted
read-only at `/model` and the GPU assigned. A passing test prints all three measured
rows; a format, shape, kernel-selection, or tolerance change exits nonzero.
