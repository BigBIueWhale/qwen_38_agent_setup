# Qwen3.8 context, MRoPE, and TurboQuant K8V4 audit

This report derives the deployed context implementation from the pinned model
configuration and the exact vLLM code. It then records independent numerical tests
of multimodal positions and the actual K8V4 store/decode kernels.

## Model geometry and native context

The corrected checkpoint retains the official Qwen3.8 configuration:

| Property | Exact value |
|---|---:|
| Native total context | 262,144 tokens |
| Language layers | 64 |
| Full-attention layers | 16: 3, 7, 11, ..., 63 |
| Other layers | 48 Gated DeltaNet layers |
| Query heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| RoPE theta | 10,000,000 |
| Partial rotary factor | 0.25 |
| Rotary dimensions | 64, or 32 frequency pairs |
| MRoPE sections | `[11, 11, 10]` |
| MRoPE layout | interleaved |
| Native RoPE type | default, with no scaling |

The one supported deployment sets `--max-model-len 262144` and does not set YaRN or
`VLLM_ALLOW_LONG_MAX_MODEL_LEN`. Context means prompt plus all generated reasoning,
tool syntax, and final output. It is not a prompt-only allowance.

Static YaRN is deliberately absent. Alibaba documents it as an option beyond the
native window, but it changes scaling even for shorter sequences and does not supply
memory. Local allocation experiments reached roughly 335,872 tokens before the
32-GiB card ran out of memory; they did not prove retrieval quality. One million K8V4
tokens would require about 23.126 GiB for raw full-attention KV alone and cannot
coexist with the deployed weights on this GPU.

## Text and image positions

Text-only prompts receive ordinary one-dimensional positions copied across all
three MRoPE axes. Image placeholders remain exactly where the originating message or
tool result put them. For each image run:

- the temporal coordinate is the current text position;
- height and width enumerate the post-merge vision grid;
- the next text position advances by `max(grid_height, grid_width)`, not by the
  number of vision tokens;
- generated tokens continue one-dimensionally after the maximum multimodal
  coordinate using the model's position delta.

`scripts/qwen38_context_unit.py` independently implements the released Transformers
5.15 image-position algorithm, then compares it element-for-element with
`Qwen3_5ForConditionalGeneration._get_mrope_input_positions` from pinned vLLM. Its
three-image prompt deliberately reverses the transport feature list; offsets still
determine chronology. It also compares the exact interleaved frequency-axis map and
generated-token continuation.

The pinned result is:

```text
PASS qwen38_context: native=262144 layers=64 full_layers=16 head_dim=256
rotary_dim=64 mrope=[11,11,10]/interleaved images=3 prompt_tokens=99
delta=-38 raw_k8v4=6509559808B/6.0625GiB
```

This proves alignment between the released model semantics, Transformers 5.15, and
the pinned vLLM implementation for text, interleaved images, and continuation. It
does not assert an undocumented Alibaba training distribution for extreme aspect
ratios; the separate image contract is intentionally stricter.

## Exact K8V4 storage arithmetic

Only the 16 full-attention layers have a growing KV cache. For head dimension 256,
one KV head and one token occupy:

| Component | Bytes |
|---|---:|
| Key, E4M3 FP8 | 256 |
| Value, two 4-bit codes per byte | 128 |
| Value scale, FP16 | 2 |
| Value minimum, FP16 | 2 |
| Total slot | 388 |

Therefore the raw native cache is:

```text
388 bytes * 4 KV heads * 16 layers * 262144 tokens
= 6,509,559,808 bytes
= 6.0625 GiB
```

The launch reserves 6,925,634,765 bytes. vLLM's hybrid cache manager pages and
aligns both full-attention cache and recurrent-layer state, reporting 264,115 cache
tokens. The extra 1,971 cache slots are allocator headroom; the API context remains
the official 262,144 total tokens.

K8V4 is not four-bit for both K and V. Keys are direct E4M3; values use an affine
four-bit representation per 256-element vector. It is also not BF16 KV and is not
defined by the checkpoint's static KV calibration scales.

## Prefill and decode behavior

For prefill, the cache store quantizes the incoming chunk first and attention then
consumes the same BF16 K/V tensors directly through FlashAttention — the store
happens before attention in the deployed order, and the raw tensors, not the
quantized copies, feed the current chunk's scores. Future decode steps read the
compressed history.
The fused TurboQuant decode path dequantizes keys and values for score, softmax, and
value accumulation; the returned attention output is BF16. Continuations of at most
128 tokens use the fused decode path. Larger attention queries use the dedicated
dequantization workspace followed by FlashAttention.

The local workspace patch does not change quantization or attention mathematics. On
the exact K8V4 FP8-key path, it aliases final BF16 continuation buffers into the
already reserved 1,024-MiB dequantization workspace instead of allocating a second
full-length pair. The patch has strict geometry and lifetime guards and leaves other
TurboQuant modes unchanged.

## Independent store and decode test

`scripts/turboquant_k8v4_unit.py` imports vLLM's installed production Triton store
and fused decode launchers. For Qwen's exact `D=256`, `Hq=24`, `Hkv=4` geometry it:

1. stores 37 BF16 K/V tokens across multiple cache blocks;
2. independently constructs every expected key byte, value nibble, FP16 scale, and
   FP16 minimum;
3. requires exact E4M3 key and metadata bytes;
4. permits a one-code value difference only when compiler reassociation lands
   within `2e-5` of an integer rounding boundary;
5. independently dequantizes the actual cache bytes;
6. performs explicit FP32 grouped-query scores, softmax, and value accumulation; and
7. compares the fused production output to that reference.

The actual RTX 5090 result was:

```text
PASS turboquant_k8v4: D=256 Hq=24 Hkv=4 tokens=37 slot=388
boundary_choices=18 max_abs=0.00381172 min_cosine=0.999998331
```

The companion vLLM test
`tests/quantization/test_turboquant.py::test_qwen38_k8v4_bf16_gqa_matches_packed_reference`
freezes the same model-specific contract in the reconstructed source tree.

These tests prove implementation agreement with the explicitly packed cache and a
high-precision attention reference for the tested shapes. They do not prove that a
four-bit value cache is quality-equivalent to BF16 KV across every downstream task.
That empirical quality risk remains explicit; K8V4 is selected because BF16 KV
cannot provide the native context alongside these weights on a 32-GiB card.

## Vision memory interaction

The 6.45-GiB cache reservation and native text context are never reduced to enable
vision. The complete BF16 vision tower remains loaded. Around vision encoding only,
the runtime temporarily releases the reclaimable 1,024-MiB TurboQuant workspace and
a fixed 640-MiB raw CUDA reserve, then recreates and verifies both. This creates
1,664 MiB of transient working room without changing weights, KV capacity, prefill
chunk size, CUDA graphs, image pixels, or attention precision.

Image preprocessing uses the released dynamic-resolution grid. Position IDs and
prefix-cache identity include the image at its chronological message location.
Identical image bytes are separately keyed by SHA-256 in the multimodal processor
cache. Moving identical bytes can therefore hit the image cache while correctly
missing the rendered-prefix cache.

## What is proved and what remains empirical

Proved by exact configuration/code checks and numerical execution:

- native geometry, full-attention layer pattern, and raw byte arithmetic;
- no RoPE scaling in the supported profile;
- text, image, and generated-token MRoPE positions;
- exact K8V4 cache representation and tight fused-decode agreement;
- production NVFP4 kernel selection and arithmetic, documented separately;
- full native allocation and maximum configured image execution on the target GPU.

Not claimed:

- BF16-equivalent model or KV-cache task quality;
- quality beyond the native context;
- knowledge of Alibaba's complete unpublished image/aspect training distribution;
- transfer of the measured margins to another driver, GPU, or package build.

Any change to model bytes, vLLM commit, Transformers, TurboQuant, GPU architecture,
head geometry, cache dtype, or context limit requires rerunning these tests under a
new explicit runtime profile. There is no compatibility fallback.
