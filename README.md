---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
---

## Qwen3.8-27B NVFP4 — correctness-first local agent server

This front section is the authoritative record for the one local deployment in this
directory. The upstream Qwen model card follows it verbatim under the heading
Qwen3.8-27B. The upstream card describes the model family and Alibaba recommendations;
this section records the stricter local security, quality, reproducibility, memory,
protocol, image, and agent contracts that were actually built and proved on this
workstation.

### Bottom line

The deployment is complete and healthy. There is one supported mode:

| Property | Locked value |
|---|---|
| Checkpoint | Corrected `unsloth/Qwen3.8-27B-NVFP4`, with 161 official BF16 offset-RMSNorm tensors restored |
| Checkpoint revision | 16b6615af3548b88e2d8e382457bc705b00479cf |
| Served name | qwen3.8-27b-nvfp4-k8v4 |
| Weights | Mixed NVFP4/FP8 Compressed Tensors; fragile state and vision remain BF16 |
| KV cache | TurboQuant K8V4: FP8 keys, packed 4-bit values |
| Total context | Native 262,144 prompt-plus-generation tokens |
| Vision | Full BF16 tower, FlashAttention, full released processor pixel budget |
| Images | At most 15 static inline PNGs, 16,777,216 pixels each, aspect ratio at most 30:1 |
| Video/audio | Rejected |
| Thinking | Always enabled at xhigh; high and max are exact aliases |
| Historical thinking | Omitted by default and by the supported agent client |
| Reasoning ceiling | 262,144 generated reasoning tokens, subject to remaining context |
| Final-answer ceiling | 131,072 generated final tokens, subject to remaining context |
| MTP/speculation | Disabled |
| CPU/KV offload | Zero |
| Batching | One sequence; 2,048-token chunked prefill |
| Listener | 127.0.0.1:8000 only |
| Agent client | Qwen Code 0.21.12 at b965d5f8c24f48e65fb0b17c7d45f34ca4ce8f38 |
| Agent image | sha256:9393fe2c53b34ba220ef86a930ab6ea2c6c7ad23a439af85f6c98cc446fe2f15 |
| Agent-service implementation | a0ddc3dc815b658513c62661d650cf540ba869e8 |
| Agent-service release lock | a8e5a63402f1c443a288d92b65e3fcdcfc9d7211 |
| Agent-service image | sha256:0f3b8096b8c18207acd2483d046de20efce31a226ebd8fe6f8ff2e98e9463b6e |
| Agent-service listener | 127.0.0.1:8090 only |
| Runtime profile | socket-isolated-nonroot-vision-k8v4-agent-v15 |
| Runtime image | sha256:dd70ee4a13f89ecaba05fcf627414f6e35489923155e4d7356d2c730b3baee44 |

This is not a text-only profile with an optional vision switch. It is not a
one-million-token profile. It has no MTP, eager-mode, lower-quality image, alternate
cache, alternate port, wildcard-listener, or silent fallback variant. Historical
runtime images and caches are retained only as recovery evidence; none is accepted
by the current scripts.

The total context limit means exactly:

    prompt tokens + reasoning tokens + tool/final tokens <= 262144

It does not mean 262,144 prompt tokens followed by additional output.

### The only everyday commands

From this directory:

    ./start.sh
    ./status.sh
    ./stop.sh

All three take no arguments. A supplied mode, port, model, or tuning option is an
error.

- start.sh starts the one profile and its two fixed relays, waits on the exact vLLM
  and relay readiness events, then validates the complete live configuration before
  reporting success. Re-running it validates the existing owned topology rather than
  starting a duplicate.
- status.sh validates host prerequisites, nine ordered vLLM transformations, every reviewed
  source and test file, the model manifest, image archive, image identity and labels,
  command and environment, mounts, runtime packages, API identity, listener,
  hardening, and live health. HEALTHY means all checks passed.
- stop.sh removes only the exact project-labelled ingress, bridge, and vLLM
  containers, removes only its owned socket, and verifies that port 8000 is free. It
  never kills an unknown process or deletes weights, images, caches, patches, or test
  evidence.

Startup uses a single event-driven log-follow deadline, not sleep-based busy waiting.
A failed startup prints the final logs, removes only the exact failed project
container, and leaves durable inputs intact. No command guesses a replacement,
continues after a mismatch, or calls a mutable network fallback.

Advanced reproducibility operations are deliberately separate from serving mode:

    ./scripts/build-vllm.sh check
    ./scripts/build-vllm.sh build
    ./scripts/restore-images.sh

The check reconstructs the source tree from the pinned upstream commit through all nine
landmark-aware transformations. The build runs offline from the exact base image and fails unless it produces
the pinned image ID. Restore verifies the pinned local archive before loading it.

### Security and host boundary

The machine is assumed exposed to the public Internet on every non-loopback
interface. vLLM therefore runs under Docker `--network none` and binds only its own
private namespace loopback at 127.0.0.1:8000. A minimal fixed bridge sharing that
namespace connects the private loopback to one project Unix socket. A separately
pinned minimal ingress is the only host-network component; it binds exactly host
127.0.0.1:8000 and connects only to that socket. Neither component has a dynamic
target or configuration language. status.sh validates both relays, the socket,
namespace identities, exact listener, and the absence of Docker port mappings.

The vLLM container runs as `2000:0` with cap-drop ALL, no-new-privileges, restart=no,
a read-only root, a read-only model mount, and one dedicated labelled cache volume.
The only durable writable runtime state is that exact v15 volume mounted at
`/home/vllm/.cache/vllm`, owned `2000:0` mode 0770; all CUDA, Triton, TorchInductor,
FlashInfer, Hugging Face, XDG, and vLLM caches are rooted beneath it. `/tmp` is a
bounded 2 GiB executable tmpfs and `/run` is a bounded 64 MiB non-executable tmpfs.
There is no writable `/root` mount. status.sh validates the user, root flag, exact
mount source/options/ownership, cache environment, and absence of extra mounts. The
model checkpoint is never modified.

Docker is the dependency-execution boundary. Tokenizers, protocol clients, build
tools, Python libraries, and integration tests run in the serving container or in
disposable/project-owned containers. Nothing in this project authorizes installing
host Python/npm packages, editing host application configuration, or touching the
user's host Claude Code installation, credentials, history, sessions, or settings.
Host interaction is limited to project files, Docker lifecycle control, and explicit
hardware/listener diagnostics such as nvidia-smi and ss.

For the paired agent, “temporary” is a lifecycle and ownership guarantee, not a
blanket RAM-only requirement. Each session container, staged workspace, scratch
tree, and stream is fresh and never adopted from stale state; cleanup occurs only
after required evidence is captured and terminal state is durable. Failed capture
retains raw evidence. Bounded tmpfs is used only where the locked runtime calls for
it, while Docker images, release artifacts, terminal records, and result bundles are
deliberately durable. Storage medium is not treated as a substitute for namespace,
mount, retention, and teardown correctness.

The observed unrelated wildcard listeners on other ports predate this project and
are outside its scope.

### Model identity and provenance

The exact model snapshot is Qwen3.8-27B, not Qwen3.6, not a renamed derivative, and
not an MoE model:

- repository: unsloth/Qwen3.8-27B-NVFP4
- revision: 16b6615af3548b88e2d8e382457bc705b00479cf
- immutable conversion source: Qwen3.8-27B-NVFP4-Unsloth
- official BF16 reference: Qwen/Qwen3.8-27B at
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
- deployable directory: Qwen3.8-27B-NVFP4-Corrected
- model.safetensors: 22,568,192,096 bytes,
  5fd70b38b3708e47adc1e9e9ab90f5d688ec01177d0718fdd16678696fdb0988
- model_mtp.safetensors: 849,400,392 bytes,
  1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe
- tokenizer.json: 19,989,325 bytes,
  06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523
- model index: 1,968 tensors and 23,417,592,488 tensor bytes
- top-level deployable manifest:
  manifests/model-corrected-16b6615a-norms-1d4bf0f2.sha256
- manifest hash:
  3a86177c30b97035d27ad0cf516fc4c2ddb83701c4de4fc6adcb23c7c2531bfc

The manifest covers all thirteen deployable files: weights, MTP weights, tokenizer,
vocabulary, both templates/configuration surfaces, image and video processor
configuration, index, README, and attributes. Missing, changed, or unexpected
top-level files fail status. A moving main branch is never accepted.

The unusually early publication time of the Unsloth quant was not treated as proof
of authenticity. Every tensor was accounted for against the official BF16 checkpoint:
all 233 FP8 and 168 NVFP4 matrices were independently dequantized, and all 798
reference-precision tensors were compared. The selected checkpoint is genuinely
derived from Qwen3.8-27B.

The quantization recipe is mixed:

- most MLP projections use dynamic NVFP4 W4A4 with group size 16;
- full-attention projections, important Gated DeltaNet projections, the LM head, and
  the final eight MLP layers use FP8;
- fragile recurrent/control state and the complete vision tower remain BF16.

This is a practical quality/context compromise, not a claim of lossless weight
quantization. No publisher has supplied a comprehensive BF16-versus-this-checkpoint
perplexity or task-quality delta. The rejected RadixArk partial download remains
historical data and is not selected, resumed, or silently deleted.

The full audit measured aggregate FP8 relative L2 0.0266015 / cosine 0.9996461 and
aggregate NVFP4 relative L2 0.1082006 / cosine 0.9941543 against official BF16. The
worst NVFP4 tensor was layer 0's down projection at relative L2 0.1534641. The live
production-kernel audit intentionally loads that complete 5,120 x 17,408 matrix and
requires `FlashInferCutlassNvFp4LinearKernel`. Across M=1, 17, and 129 it stayed below
0.00381 relative L2 and above 0.999993 cosine against independent dequantization plus
BF16 matmul. This establishes implementation correctness to a tight tolerance; it
does not erase the model-quality cost of four-bit weights.

The exact M=1/17/129 relative-L2 results were 0.003161705, 0.003603501, and
0.003809735; cosine similarities were 0.999995470, 0.999993920, and 0.999993205.
Maximum absolute error was 1.0 in BF16 output units and the independent packed-weight
mismatch was 0.000003040. The selector check is mandatory: a generic or silently
different linear kernel is not accepted as equivalent evidence.

That exhaustive comparison found a separate conversion defect in 161 language
RMSNorm tensors. Qwen stores offset weights and applies gain `1 + w`; the conversion
path had rounded `1 + w` to BF16 before subtracting one. The deployable snapshot
restores those 161 byte ranges from the exact official revision. A streamed semantic
digest proves every other byte is unchanged. `./scripts/repair-model.sh` recreates
the corrected directory atomically, offline, and inside the pinned Docker boundary.
The source conversion is retained as evidence but is never mounted by the supported
runtime. Full methods, aggregate errors, kernel comparisons, and limitations are in
`docs/qwen38-weight-quantization-audit.md`.

The snapshot includes separate MTP tensors, but presence on disk is not activation.
The launch has no speculative configuration and logs speculative_config=None, so MTP
is not loaded or used.

### Exact vLLM and image provenance

The vLLM submodule is pinned at:

    9df9b0b0a1816b6d0d0f6ecd0da563cc37fd72f5

It is intentionally reconstructed by eleven ordered, reviewed semantic transformations:

| Patch | SHA-256 |
|---|---|
| patches/vllm-turboquant-k8v4-direct-workspace.patch | a9721067f1a7ee9497a4bd51e47e3a474561189e881b4704bfc4beac8ea48380 |
| patches/vllm-enforce-auto-tool-schema.patch | 4f75c793a9c2cdcfb2fd0768ba49a4e34748d3a37d8392b07d3592ca50939c07 |
| patches/vllm-qwen38-agent-defaults-and-thinking.patch | 6428d2cfa77f28e57e117999d0ec8fab5430856c985ba530e04885c2f5c420b7 |
| patches/vllm-qwen38-separate-final-response-budget.patch | f20d7dff41931248272842ed2c7a163c6f013e405ccf35733c40ff131a2fc503 |
| patches/vllm-qwen-implicit-tool-grammar-boundary.patch | d231c6e2e7040c4cd4b38432cb8c794805afddbf2c6e4f7ff6febb78e3fd9f48 |
| patches/vllm-anthropic-validation-http400.patch | 030b64be104e6ef57a40f6bae740dfa9d4634a420c6c93a395f62bfb98d6d053 |
| patches/vllm-tool-truncation-finish-reason.patch | 1a220f6db9b40967d867b3cfb1a92d95d907ca059718ffe61772b4cb4409f551 |
| patches/vllm-qwen38-vision-runtime.patch | f92603724861da5b5a364f43e57d3f95ef43a9dded8ae645278373850db3140f |
| patches/vllm-qwen38-numerical-audits.patch | a73aa2f2ae3f82010eb2bafcdf663c2fe14854c30165dbc4d8457725bc3b6632 |
| patches/vllm-turboquant-fail-closed-guards.patch | 0ecf95ab8ee25a76d5412ce44aafafe13992b2cb373d6010acf5bc119dc8f47b |
| patches/vllm-kv-offload-pinning-fail-closed.patch | 1857071c38d081bb95e3cca12153cebce096649084950b99229104fdae029ca6 |

The reconstructed tree has exactly thirty-two reviewed runtime-source changes,
seven reviewed existing-test changes, and one reviewed new workspace test. The
landmark-aware Python patcher calculates every mutation before writing, validates
unique structural landmarks and complete pre/post hashes, performs atomic
transactions with rollback, and is itself covered by seven failure-path tests. The
unified diffs remain review artifacts, but they do not select mutation locations.
The build check rejects an extra dirty file, ambiguous landmark, missing hunk, wrong
stage, changed final hash, whitespace error, partial intermediate state, concurrent
source drift, or a patch that does not recreate the exact live tree.

Pinned build inputs and products:

| Item | Identity |
|---|---|
| Immutable base tag | qwen38-vllm:main-9df9b0b |
| Immutable base ID | sha256:fa4a002a88b7043a1a89966dea8a500fe9696f84e75730d9da916f916048d401 |
| Runtime tag | qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v15 |
| Runtime ID | sha256:dd70ee4a13f89ecaba05fcf627414f6e35489923155e4d7356d2c730b3baee44 |
| Offline archive | artifacts/qwen38-vllm-images-runtime-v15.tar |
| Archive size | 8,557,675,008 bytes, mode 0600 |
| Archive SHA-256 | a80766d9560a419b9c051fc84d9beca1f1a3ac9ab508c99cf29d218b71bef43c |
| Runtime Dockerfile SHA-256 | 17f72538ee71292e4cf0a2ce804e52a4d26413a034286590aef76008fbd4fcec |
| Docker context allowlist SHA-256 | a15c81d0be5c474d9f0cd5e8b1d3f89b5eb7266ce60d45476069de9499f6b103 |
| Build verifier SHA-256 | d231f88f8f3e1418ff2fb68498762c2d48cfec91b881728116acd64cba1a84a9 |
| Runtime validator SHA-256 | 6954f0a81c1be056e2ad882f68249aa34e24d29aa325aa05fe04e477f8ef3781 |
| Runtime lock SHA-256 | 6498c8fd4ac52306fd79360c80911b6e01f2eb7b2f562b18299f08277cb4aced |

The final runtime layer does no package resolution or installation. It is built with
pull=false, network=none, provenance=false, an exact base ID, an allowlisted context,
upstream installed-file hashes, final installed-file hashes, and build-time invariant
tests. Independent offline builds produced the identical v13 image ID.

The local repository identity is Ronen Zyroff <rzyroff@gmail.com>. Global Git
configuration is untouched. Large checkpoint trees, image archives, caches,
credentials, transient output, and editor state are ignored; the compact hashes,
manifests, patch artifacts, validation scripts, and recovery instructions are tracked.

### Exact launch contract

config/runtime-v1.sh is the single source of truth consumed by start, status, stop,
restore, and build verification. The exact server argument semantics are:

    /model
    --served-model-name qwen3.8-27b-nvfp4-k8v4
    --host 127.0.0.1 --port 8000
    --model-impl vllm
    --config-format hf
    --load-format safetensors
    --tokenizer /model
    --chat-template /opt/qwen38/chat_template.jinja
    --chat-template-content-format openai
    --generation-config /model
    --override-generation-config
      {"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,
       "presence_penalty":0.0,"repetition_penalty":1.0,
       "thinking_token_budget":262144,"final_response_token_budget":131072}
    --quantization compressed-tensors
    --dtype bfloat16
    --kv-cache-dtype turboquant_k8v4
    --max-model-len 262144
    --max-num-seqs 1
    --max-num-batched-tokens 2048
    --kv-cache-memory 6925634765
    --cpu-offload-gb 0
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{"cpu_bytes_to_use":7747584000,"eviction_policy":"arc"}}'
    --enable-prefix-caching
    --enable-chunked-prefill
    --attention-config.flash_attn_version=2
    --kernel-config.enable_flashinfer_autotune=False
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --default-chat-template-kwargs
      {"enable_thinking":true,"preserve_thinking":false,
       "reasoning_effort":"xhigh","add_vision_id":false}
    --limit-mm-per-prompt
      {"image":{"count":15,"width":4096,"height":4096},"video":0}
    --mm-processor-kwargs
      {"size":{"longest_edge":16777216,"shortest_edge":65536}}
    --mm-processor-device cpu
    --no-mm-device-do-normalize
    --mm-encoder-tp-mode weights
    --mm-processor-cache-gb 4
    --mm-processor-cache-type lru
    --mm-hasher-algorithm sha256
    --mm-tensor-ipc direct_rpc
    --no-skip-mm-profiling

The exact relevant environment includes:

    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    VLLM_NO_USAGE_STATS=1
    VLLM_ENFORCE_STRICT_TOOL_CALLING=1
    VLLM_QWEN38_STRICT_IMAGE_CONTRACT=1
    VLLM_QWEN38_VISION_HEADROOM_BYTES=671088640
    VLLM_MAX_IMAGE_PIXELS=16777216
    GLOO_SOCKET_IFNAME=lo
    NCCL_SOCKET_IFNAME=lo

Consequences:

- There is no language-model-only flag. The complete vision model is loaded.
- There is no speculative/MTP argument.
- CUDA graphs remain enabled. Vision was not bought by forcing eager text execution.
- The measured 2,048-token prefill chunk remains fixed. Vision was not bought by
  reducing text prefill performance.
- FlashAttention 2 is explicit because automatic selection otherwise changes the
  TurboQuant path. FlashInfer autotuning is explicitly off so probe-OOM-and-fallback
  behavior cannot be mistaken for normal startup.
- CPU offload and KV offload are exactly zero.
- Multimodal profiling is mandatory and cannot be skipped to obtain a deceptively
  optimistic allocation.
- All unquantized model computation, including the entire vision tower, uses BF16.
- trust_remote_code remains false. Current vLLM resolves the architecture natively as
  Qwen3_5ForConditionalGeneration.
- Request-side media limits, processor overrides, and lower image-detail choices are
  rejected by the strict image patch rather than silently replacing the profile.

### Agent defaults: xhigh thinking, exact sampling, long output

The one deployment is explicitly configured for complex agentic work. A client that
omits Qwen-specific fields receives all of these server-side defaults:

    enable_thinking       = true
    reasoning_effort      = xhigh
    preserve_thinking     = false
    add_vision_id         = false

    temperature           = 1.0
    top_p                 = 0.95
    top_k                 = 20
    min_p                 = 0.0
    presence_penalty      = 0.0
    repetition_penalty    = 1.0

    thinking_token_budget       = 262144
    final_response_token_budget = 131072

These sampling values are Alibaba's published Qwen3.8 thinking-mode tuple.
repetition_penalty=1.0 is neutral. The historical Qwen3.6 repetition detector is not
used; SamplingParams.repetition_detection is None. Adding a heuristic repetition
intervention would alter the learned distribution and is not part of this profile.

xhigh is the canonical effort. Omitted effort, OpenAI high, OpenAI max, and the
supported Anthropic high/max-style forms render the same xhigh model token IDs.
Medium, low, disabled thinking, and incompatible Anthropic controls are rejected with
HTTP 400. The supported agent client always sends xhigh and never asks for a weaker
mode.

preserve_thinking=false does not turn reasoning off. It removes completed hidden
reasoning blocks from older turns while retaining visible answers, native tool calls,
typed tool results, and the current unresolved user/tool chain. This is the selected
single-long-thread policy: scarce context is spent on the task and its durable
outcomes instead of replaying every old hidden trace. A direct diagnostic API request
can explicitly test preserve_thinking=true, and that behavior was verified, but it is
not the supported agent-service policy or another launch mode.

Alibaba's 262,144 reasoning and 131,072 final ceilings are recommendations for
frameworks that distinguish the phases within a much larger window. Locally they are
hard server defaults but not reservations and not additive capacity. The usable
generation allowance is the minimum of the phase ceiling, the request's total
generation ceiling, and physical context remaining after exact rendering. A short
prompt can use extensive reasoning; a nearly full prompt cannot.

The final counter starts only after the explicit reasoning-end marker. Tool XML is a
structured tool phase, not visible final prose. EOS and stop sequences may end
earlier; min_tokens cannot cross a hard phase ceiling. Chat, Anthropic Messages, and
Responses all inherit the same defaults. Clients may lower a phase ceiling for a
deliberate request but cannot null or raise the server's final-response ceiling. A
live five-real-token final-ceiling probe stopped at exactly five final tokens.

Prompts are not claimed deterministic. Correctness tests compare structure, typed
semantics, exact rendering, and repeated pass rates rather than pretending a fixed
seed makes long sampled GPU trajectories byte-identical.

### KV cache, context length, and VRAM

TurboQuant K8V4 is the only cache format:

- keys are stored at FP8;
- values are packed to 4-bit;
- every live vector retains the TurboQuant scale/zero-point metadata;
- K8V4 is not K4V4, ordinary FP8 cache, BF16 cache, or a two/three-bit format;
- the checkpoint's static FP8 cache calibration metadata does not pre-quantize the
  runtime cache and does not define this K8V4 path.

For Qwen3.8's four KV heads in each of sixteen full-attention layers, the pinned code
accounts 388 bytes per head per token: 256 FP8 key bytes, 128 packed value bytes, and
four FP16 scale/zero-point bytes. The raw cache is therefore 24,832 bytes per token:

- 6.0625 GiB at 262,144 tokens;
- about 23.126 GiB at one million tokens.

vLLM must also page and align the hybrid Gated DeltaNet state. The explicit
6,925,634,765-byte allocation reports 264,115 cache-token capacity, 1.01 times native
maximum concurrency. That leaves only 1,971 cache tokens beyond native, not a useful
extended-context tier.

The reviewed TurboQuant patch reuses the already reserved 1,024 MiB dequantization
workspace as the final BF16 continuation buffers on the exact K8V4 key-FP8 path. It
eliminates duplicate full-length K/V buffers and progressive runtime allocations
without changing K4V4/MSE modes. Focused kernel tests passed.

This storage path is numerically tested rather than accepted from its name. The
runtime's actual Triton store produced the independently constructed FP8 key bytes,
packed V4 nibbles, and FP16 affine metadata for Qwen's exact D=256 geometry. The
actual fused GQA decode matched an explicit FP32 score/softmax/accumulation reference
with maximum absolute difference 0.00381172 and minimum cosine 0.999998331. Eighteen
value codes fell on floating-point rounding boundaries; all were within the stated
`2e-5` boundary interval and differed by only one code step. All stable codes, key
bytes, scales, and minima matched exactly.
The accepted case used D=256, 24 query heads, 4 KV heads, 37 tokens, the exact
388-byte per-head slot, and 18 independently classified rounding-boundary choices.

The numerical runner itself was also treated fail-closed. An initial invocation while
the serving engine still owned the GPU failed with CUDA OOM; subsequent runner
attempts exposed a read-only `/home/vllm/.triton` target and then incorrect uid/exec
tmpfs contracts. None was reported as a numerical result. The accepted audit ran
only after exclusive GPU ownership, as non-root, with its executable cache/scratch
mount explicitly owned by that user. This distinction preserves the failed evidence
without mislabelling an orchestration failure as a TurboQuant or NVFP4 defect.

Vision adds the complete BF16 tower and transient encoder/MLP activations. vLLM logs
21.34 GiB loaded model memory, 6.45 GiB KV reservation, 1,024 MiB reclaimable
TurboQuant workspace, and about 0.06 GiB CUDA graph capture. The vision patch
temporarily releases that workspace plus a fixed 640 MiB raw headroom around vision
encoding, then restores them exactly. It does not change cache capacity, text
prefill size, graphs, weight precision, or image precision. Logs from maximum images
show release/restoration and no OOM, retry, preemption, or fallback.

On the exact final v13 live image:

- immediately before the fifteen-image maximum-quality run: 31,647 MiB used,
  464 MiB free;
- after the complete backend suite and recorded v8 agent acceptance: 31,797 MiB used,
  314 MiB free;
- total reported VRAM: 32,607 MiB.

The post-suite reading is an observed residency point, not a claim that generation
always peaks at exactly that number. The small global free number is not the memory available during a vision
encode; the exact patch deliberately makes 1,664 MiB available around that phase.
The full-context and maximum-image runs completed, so residency is proven by
execution rather than inferred from an idle screenshot.

The native context is 262,144 total tokens. config.json, tokenizer_config.json, the
upstream card, exact tokenizer, vLLM request accounting, and the live boundary agree:

    262143 prompt + 1 output = 262144 total  -> accepted
    262144 prompt + 1 output = 262145 total -> HTTP 400

The accepted multimodal boundary included all fifteen maximum-size images and
245,745 tokens of multimodal expansion. No YaRN, long-context environment override,
or nominal one-million setting is enabled.

Static-YaRN allocation experiments were separately bracketed: engines initialized
through 335,872 configured tokens and OOMed at 337,920. That is only an allocation
edge, with tens of MiB margin and no extended-context retrieval/quality proof.
Static YaRN also changes short-context position scaling. It is therefore rejected as
a supported mode. One million tokens physically cannot coexist with these weights
and a 23.126-GiB raw K8V4 cache on this 32-GiB card.

Every long-context constructor uses the real checkpoint tokenizer inside Docker,
applies the real chat template, and requires Transformers, vLLM /tokenize, and final
usage.prompt_tokens to agree. There is no character division, target//8, token-density
estimate, padding fudge, tokenizer substitution, or host tokenizer.

The model-position audit separately freezes the 64-layer / 16-full-attention
geometry, 256-dimensional heads, 0.25 partial RoPE, 10,000,000 theta, and interleaved
MRoPE sections `[11,11,10]`. An independent implementation of the released
Transformers 5.15 image-position algorithm matched vLLM element-for-element for
interleaved image placements and generated-token continuation, even when the
transport feature list was deliberately reversed. Complete derivations, test code,
results, and the limits of the claim are in
`docs/qwen38-context-turboquant-audit.md`.

The accepted context probe covered all 64 language layers, 16 cached full-attention
layers, 256-dimensional heads, 64 rotary dimensions, and interleaved sections
`[11,11,10]`; its 99-token mixed prompt produced the expected continuation delta of
-38. The same first-principles calculation yields exactly 6,509,559,808 raw K8V4
bytes, or 6.0625 GiB, at 262,144 tokens.

### Verified: the norm repair restores exactly the official values

The offset-RMSNorm damage mechanism and its repair are now empirically
proven, not just derived. For every sampled repaired tensor, every element
of the pinned Unsloth export satisfies

    bf16(1 + w_official) - 1 == w_unsloth

with a 100.0000% match — the export's norms are exactly the official norms
passed through the exporter's `1 + w` BF16 round trip, and nothing else.
Measured damage in the export this repair reverses: only 6–49% of elements
per tensor survived identical, up to 3.9% of layer-0 input-norm weights
were zeroed outright (|w| below the 2^-8 grid), and per-element error
reached 0.0078 against |w| medians as low as 0.033. Because the corrected
snapshot byte-restores the official ranges (and its manifest proves nothing
else changed), the repair cannot damage the model: for these 161 tensors,
corrected == official ground truth. Completeness is now also proven, not
assumed: with the official BF16 checkpoint restored on disk, a sweep of
all 783 BF16 tensors shared by the export and the official reference found
622 byte-identical and exactly the 161 known norms carrying the round-trip
signature — no unexpected damage anywhere, and nothing differing for any
other reason. The repair set was exhaustive.

### Final deferred audit: quantized-weight context correctness

After every other service, lifecycle, benchmark, documentation, release, and
deployment task is complete, the final task is a mathematical end-to-end audit of
how this deployed mixed NVFP4/FP8 checkpoint handles context versus the exact
official BF16 reference at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. This item is deliberately recorded as
pending; the existing tensor reconstruction, isolated production-matmul, MRoPE, and
TurboQuant tests are necessary evidence, but they do not by themselves establish
end-to-end contextual equivalence.

The audit must use identical real rendered/tokenized inputs and compare matched
BF16-reference and deployed executions through multimodal token placement and
MRoPE, chunked and unchunked prefill, every Gated DeltaNet recurrent state, every
full-attention layer, K/V creation before cache compression, K8V4 storage and
reconstruction, continuation hidden states/logits, and controlled long-context
retrieval at multiple positions up to the native 262,144-token boundary. It must
separate at least three causal conditions rather than report one blended score:

1. official BF16 weights with BF16 K/V state as the reference;
2. deployed mixed NVFP4/FP8 weights with BF16 K/V state, isolating weight
   quantization and conversion error;
3. the same deployed weights with TurboQuant K8V4, isolating the additional cache
   error and any interaction with position or accumulated context length.

The report must derive tolerances before observing results, include per-layer and
end-to-end error growth rather than only aggregate cosine similarity, test text and
chronologically interleaved full-quality images, and distinguish implementation
correctness from model-quality equivalence. Any unavailable BF16 execution path,
OOM, kernel substitution, shortened-context proxy, or incomparable rendering is a
failed/incomplete audit condition, not permission to silently weaken the claim.

### Exact image-quality contract

Qwen3.8-27B is natively trained and post-trained as a vision-language model. Alibaba
publishes visual and multimodal-agent benchmarks, a dynamic-resolution processor,
and a released image processor with:

    longest_edge = 16777216 pixels
    shortest_edge = 65536 pixels
    patch_size = 16
    temporal_patch_size = 2
    spatial merge = 2

Those released processor values justify using the complete 16,777,216-pixel image
budget. They do not publish the exact distribution of aspect ratios or crops used
throughout training. A permissive utility constant or an extreme resize formula is
not proof that the model learned equal quality at that ratio. The deployment
therefore separates three claims:

1. Official: the model is a native vision-language model and the released image
   processor exposes the pixel range above.
2. Unpublished: Alibaba has not documented a complete learned aspect-ratio
   distribution for Qwen3.8-27B.
3. Locally proved: full-budget images work at square and at both portrait/landscape
   endpoints through 30:1; the deployment rejects anything beyond 30:1.

The transport/decoder contract is intentionally narrower and fail-closed:

- only an exact inline data:image/png;base64, URL is accepted;
- remote URLs, file URLs, JPEG, WebP, GIF, BMP, SVG, animated PNG, and video are
  rejected before inference;
- source PNG must be static, eight-bit RGB or RGBA;
- palette, grayscale, 16-bit, and RGB tRNS forms are rejected;
- RGBA is deterministically composited onto pinned white and passed as RGB;
- transport remains lossless PNG; there is no JPEG transcode;
- source and processed image limits are both 16,777,216 pixels;
- source aspect ratio must be at most 30:1 in either orientation;
- official 32-pixel neural-grid alignment remains part of Qwen preprocessing;
- detail=auto and detail=high use the same full-quality path;
- detail=low is rejected;
- add_vision_id is fixed false;
- callers cannot provide image embeddings, PIL objects, UUIDs, audio/video/file
  parts, media-IO overrides, per-request processor overrides, or alternate limits;
- at most fifteen images are accepted; the sixteenth is HTTP 400;
- image dimensions declared by the serving limit are 4,096 by 4,096, while other
  shapes within the same pixel/aspect contract are accepted and processed on the
  official grid.

Full quality here has a precise meaning: maximum released processor pixel budget,
complete BF16 vision weights and activations, lossless transport, official dynamic
resolution, no low-detail path, no silent downscale from an over-limit source, and no
vision quantization. It does not falsely mean that a neural patch encoder preserves
every source pixel as an independent token. Sources outside the contract are rejected
with the violated bound rather than silently changed.

The maximum was proved, not estimated:

- fifteen distinct 4,096 x 4,096 images were encoded in one request;
- every image used the full 16,777,216 source pixels;
- the model transcribed all thirty independent pixel strings exactly;
- the request used 246,022 prompt tokens and completed normally;
- a sixteenth image was rejected before inference;
- the exact native-context boundary with those fifteen images also passed.

Aspect proof used six independent fresh near-full-pixel images, alternating
22,080 x 736 and 736 x 22,080. Each had 16,250,880 pixels and 15,870 visual tokens.
The model recovered codes placed at both far ends of every image. A 31:1 control,
21,824 x 704, was rejected with an explicit HTTP 400 naming the 30:1 bound.

### Where images live in an agent history

Images are not moved to the newest user message. They remain in the exact
chronological content position at which the user or tool supplied them:

    user text
    assistant reasoning/tool call
    tool result: text -> image -> text
    assistant acknowledgement
    later user question
    assistant response

The patched OpenAI path accepts a typed content list on the originating tool role.
The patched Anthropic converter retains a tool_result's text/image/text sequence in
the corresponding tool response instead of inventing a later user image turn. The
Qwen template emits each vision marker at that content position. Transport-only tool
IDs are validated and correlated, while Qwen's positional tool XML remains ordered.

OpenAI and Anthropic representations of the same history produced exactly identical
16,562 prompt-token IDs. Marker ordering proved:

    tool start < preceding tool text < vision marker <
    following tool text < tool end < later acknowledgement < later question

Moving the same image to another turn changed prompt token IDs as expected. The
model therefore sees the image in the chronology in which Qwen's interleaved
vision-language training and template expect it, rather than a clumped approximation.

Old images remain in message history unless normal context management removes that
history. They are not re-downloaded: only inline bytes are allowed. Two caches have
different jobs:

- vLLM prefix caching reuses the unchanged rendered/token prefix;
- the multimodal processor cache keys exact image bytes with SHA-256 and reuses the
  image preprocessing/encoder input independently.

The final v13 text-history probe sent a 65,529-token cold prompt with zero prefix
hits, reused 64,480 tokens on the warm continuation, and reused zero tokens for an
equivalent fresh-salt control. Warm TTFT was 32.233 times faster than cold. The final
chronological image-history probe then hit 14,560 prefix tokens plus the multimodal
cache and improved TTFT by 17.089 times. Changing image bytes caused zero
multimodal hits. Moving identical bytes caused a multimodal hit but zero prefix hit,
which proves the caches are not being conflated. That deliberately moved history is
a negative cache/render control: because its text contradicts its media chronology,
semantic OCR is not treated as an acceptance invariant. Every chronologically valid
OpenAI/Anthropic stream and non-stream request still required and returned the exact
pixel-only value.

This cache behavior does not depend on preserving old hidden reasoning. Omitting
completed hidden traces stabilizes the durable message prefix and gives the main
thread more useful lifetime; retained visible/tool/image history remains cacheable.

### Tool calling and protocol correctness

The server explicitly selects the qwen3 reasoning parser and qwen3_coder tool parser.
Automatic tool choice is enabled, parallel tool calls are disabled in the supported
agent policy, and every selected tool is constrained to its declared JSON schema
whether strict is omitted, false, or true. A normal non-tool answer and
tool_choice=none remain valid. Unknown names, wrong nested types, extra properties,
duplicate IDs, orphan results, missing or out-of-order results, and incomplete chains
fail closed.

A narrow scheduler patch retains the implicit Qwen tool-start token at the exact
reasoning-to-tool grammar boundary. Without it, post-generation parsing could
recognize a call that decoder-time structural grammar had not constrained. Real
token sensitivity tests show that the fixed path rejects an unknown tool, wrong type,
and extra property at their first invalid token.

Streaming and non-streaming paths are separately tested. An incomplete generation is
never promoted into a successful executable call:

- Chat preserves finish_reason=length;
- Anthropic preserves stop_reason=max_tokens;
- Responses marks the response and function item incomplete, emits no executable
  arguments-done/completed terminal, and ends with response.incomplete.

Thirty controlled real-token prefix cuts produced the same typed semantics in
streaming and non-streaming parser paths. Live independently sampled fault injection
also passed; no claim of byte-identical sampled prose was needed.

Complete tool loops passed three of three OpenAI trials and three of three Anthropic
trials. OpenAI Chat, Anthropic Messages, and OpenAI Responses passed streaming and
non-streaming tool calls plus typed tool-result continuation. Equivalent protocol
histories passed render -> tokenize -> parse -> rerender with token-identical
semantics. Anthropic validation problems return HTTP 400 invalid_request_error, not a
misleading server 500.

The installed-image focused suites passed:

- 125 TurboQuant cases, two platform-inapplicable skips;
- 286 parser/structural-output/Anthropic conversion cases with ten intentional
  generic-policy deselections replaced by project-specific assertions;
- 382 Qwen streaming/replay cases;
- vision workspace, image-contract, and vision-MLP units during the immutable build.

### Validation record for v13 and the paired production agent

The source/image invariants and numerical results below were obtained from the exact
pinned v13 image. The protocol, cache, native-boundary, and full-image tests are
rerun after every runtime-profile change before that image is accepted:

1. Runtime source reconstruction and immutable offline build: pass; two builds
   produced the same ID.
2. Full corrected model/tokenizer/template/processor manifest and exact 161-range
   official-norm repair proof: pass.
3. Exact network-none vLLM namespace, fixed bridge/Unix socket, host-loopback-only
   ingress, and no published Docker ports: pass.
4. MTP/speculation absent, CPU offload zero, CUDA graphs retained: pass.
5. Exact xhigh defaults, high/max alias identity, low/disabled rejection: pass.
6. Default preserve_thinking=false and explicit diagnostic restoration: pass; omission
   saved 1,798 real tokens in the controlled history.
7. Exact five-real-token final-response phase stop: pass.
8. OpenAI/Anthropic tool loops, adversarial grammar, orphan rejection, and
   streaming/non-streaming round trip: pass.
9. Responses streaming/non-streaming tool loop and incomplete-output gates: pass.
10. Fifteen full-pixel distinct-image transcription and sixteenth-image rejection:
    pass.
11. Portrait/landscape 30:1 far-end perception and 31:1 rejection: pass.
12. Chronological OpenAI/Anthropic tool-result image parity: pass.
13. Cold/warm image and prompt-cache counters plus time separation: pass.
14. Exact 262,143-prompt-plus-one-output native boundary with fifteen images: pass;
    262,145 total rejected.
15. Logs after maximum image/context work: no CUDA OOM, allocation retry, preemption,
    semantic fallback, or failed restoration.
16. Complete official-BF16 versus converted tensor audit: pass; 1,199 official and
    1,968 converted tensors fully accounted, 233 FP8 and 168 NVFP4 tensors fully
    dequantized, and all 798 reference-precision tensors compared.
17. Actual K8V4 Triton store/fused decode versus independently packed FP32 reference:
    pass; maximum absolute difference 0.00381172 and minimum cosine 0.999998331.
18. Actual worst-layer NVFP4 production kernel versus independent dequantization and
    BF16 matmul at M=1/17/129: pass; required FlashInfer-CUTLASS selector.
19. Exact text/image MRoPE implementation versus independent Transformers 5.15
    semantics and continuation positions: pass.
20. Pinned Qwen Code archive plus exact semantic reconstruction: pass; exactly 61
    changed/new files and 2,427 assertions across 23 focused test files; the full
    no-cache build reproduced agent image
    sha256:9393fe2c53b34ba220ef86a930ab6ea2c6c7ad23a439af85f6c98cc446fe2f15.
21. All pinned Rust component tests and clean release images: pass; 44 service,
    9 broker, 3 relay, 2 capture, and 2 agent-exec tests. The same no-cache release
    exactly reproduced the locked relay, capture, broker, and service image IDs.
22. Real Qwen Code hostile-workspace text, full-resolution PNG vision, PTY shell,
    write/read, and final response: pass in seven turns. Ordinary QWEN.md and
    AGENTS.md guidance remained active while `.env`, MCP, hooks, skills, rules,
    memory, output language, workspace settings, and slash commands remained inert;
    the model read exact code `VISION_AGENT_PTY_4827` and the shell emitted exact
    marker `QWEN38_AGENT_ISOLATION_OK`.
23. Locked-mode authentication revalidation: pass. An earlier candidate exposed an
    upstream late `.env` load after tools; the accepted narrow source repair keeps
    environment/workspace loading disabled on every later auth validation. The full
    source matrix and a fresh hostile live session proved the marker absent.
24. Real Qwen Code prefix and image caching: pass. The recorded v8 cache acceptance
    added 296,939 prompt tokens; vLLM recorded 241,280 local prefix hits, 55,659
    locally computed tokens, and 3 multimodal-cache hits across 20 requests.
25. PTY and foreground subagent: pass. A separate real PTY session produced exact
    shell/file output. One Explore subagent returned through correlated parent/child
    tool events, used native list/read plus a shell byte check, and was independently
    verified by the main thread. Explore retained writable conversion/scratch tools;
    its trusted effect journal proved zero workspace/artifact effects for that task.
26. Live stack isolation and lifecycle: pass. vLLM and the service are network-none;
    only two minimal fixed ingress relays use host networking. The service has no raw
    Docker socket; a network-none typed broker is its sole holder. Backend and agent
    roots are read-only, agents have no route/DNS/GPU/published port, model access is
    through the exact socket relay, session output is unmounted from the agent, and
    all components are capability-free with no-new-privileges.
27. Readiness-boundary cancellation: pass. The request acknowledged in 785 ms and
    terminated in 1,863 ms with zero turns, exit 143, an empty event stream, a
    complete nine-file bundle, no fabricated success, and no session-container
    leftovers.
28. Late capture subscriber correctness: pass. A production benchmark run exposed
    that Docker `logs --follow --since 0s` starts at relative "now" and can miss an
    already-emitted `CAPTURE_COMPLETE`. The broker now uses exact Unix epoch `0`;
    its new unit test freezes that replay contract. The service continued to fail
    closed during the faulty run and retained its evidence instead of parsing
    unproved output.
29. Clean repaired production release: pass. The pinned build ran 44 service, 9
    broker, 3 relay, 2 capture, and 2 agent-exec Rust tests, all 2,427 Qwen Code
    assertions, and reproduced all five final image IDs. A fresh five-turn
    production tool round trip then completed normally with a clean eleven-file
    bundle and no session containers.
30. Production-service SWE-rebench pilot: pass. Both history-policy variants ran
    through real production session creation and `/wait`; neither Harbor nor the
    evaluator called vLLM. The supported `preserve_thinking=false` run resolved all
    11 evaluator checks in 61 turns. The explicit `true` diagnostic also resolved
    all 11, in 83 turns and with 48.0% more aggregate input-token traffic. The two
    earlier orchestration failures are retained as infrastructure evidence and are
    not model scores.

The supported status currently reports:

    HEALTHY
    endpoint 127.0.0.1:8000
    262144 total-token context
    15 inline static PNG images, 16777216 pixels each
    BF16 vision tower, aspect ratio <= 30:1
    xhigh thinking, old traces omitted by default
    explicit Qwen3.8 sampling
    reasoning/final ceilings 262144 / 131072

### Historical audits and client direction

The historical /home/user/Desktop/qwen_36_agent_setup repository was audited idea by
idea against Qwen3.8, the current checkpoint/template, and current vLLM. Old hunks
were not copied blindly. Durable principles—strict schemas, exact tokenization,
malformed-chain rejection, fail-closed launch verification—were retained. Obsolete
Qwen3.6 egress renaming and its repetition detector were rejected. New current-code
defects in TurboQuant workspace use, grammar boundaries, phase budgets, truncation,
protocol validation, and vision handling received narrow current patches and tests.
The detailed audit is docs/qwen36-to-qwen38-audit.md.

The original /home/user/Desktop/agent_service is the selected client/orchestration
project, not a duplicated launcher here. Its durable outer design is singleton
ownership, copied workspaces, no-GPU/no-Internet agent containers, narrow Unix-socket
proxying to loopback vLLM, cancellation, durable JSONL/bundles, labels, orphan
recovery, and ordered teardown.

The chosen and deployed client is pinned Qwen Code 0.21.12 at
b965d5f8c24f48e65fb0b17c7d45f34ca4ce8f38. The official release archive is pinned
by SHA-256 61beddff8bde1dd2654c8714f927b46ab7cf9822b8561d11e3a2b8e085b5e745,
and the landmark-aware source transformation is independently pinned. It runs only
inside immutable agent image
sha256:9393fe2c53b34ba220ef86a930ab6ea2c6c7ad23a439af85f6c98cc446fe2f15.
The accepted service sends this exact outgoing policy on every main and foreground
subagent turn:

- contextWindowSize 262144;
- exact server/model identity;
- xhigh mandatory thinking;
- preserveThinking false;
- Alibaba thinking sampling tuple;
- total generation ceiling 262144 and separate final ceiling 131072;
- exact /tokenize count on the same rendered request before generation;
- no character/image-token heuristic or tokenizer fallback;
- splitToolMedia false so tool images stay in their originating tool result;
- typed content parts and PNG-only image tools matching the strict backend;
- no client retry/downgrade, XML recovery, implicit continuation, or partial-call
  execution;
- one long main thread with late exact compaction and only sequential foreground
  subagents.

The service is the updated original `/home/user/Desktop/agent_service`, with release
implementation commit a0ddc3dc815b658513c62661d650cf540ba869e8 and release-lock
commit a8e5a63402f1c443a288d92b65e3fcdcfc9d7211, not a copied launcher. Its
current release carries the workspace over the connection as a hash-committed
zip (no shared-filesystem input paths and no host input mount), returns the
result bundle over the connection with its own SHA-256 commitment, runs
sessions concurrently because serving capacity is governed above the service,
and asserts only functional host properties rather than one specific
computer's software identities. Its own
README and lock files are authoritative for Qwen Code source and
transformation, the five exact component images, package snapshot, tools, copied
workspace envelope, typed Docker broker, socket relays, stream capture, effect
journals, cancellation, bundles, and 127.0.0.1:8090 listener. Repeated real
tool-and-image sessions proved prefix and multimodal caching through Qwen Code; the
frontend compatibility cache counter is deliberately not trusted over vLLM's
authoritative counters.

The final coding pilot likewise measured the deployed pair rather than bypassing it.
The harness submitted the same pinned SWE-rebench task to real production
`POST /v1/agent/sessions`, waited on the production notification endpoint, required
the service's durable terminal bundle, and invoked the immutable evaluator only on
that captured post-session workspace. The normal `preserve_thinking=false` session
resolved all 11 evaluator checks in 61 turns and 1,090,658 ms. An explicitly typed
`true` diagnostic also resolved all 11 in 83 turns and 1,081,835 ms, while sending
1,636,513 more aggregate input tokens through Qwen Code. This single sampled pair
supports the normal policy's context efficiency but is not a population-level
quality or latency claim. Exact production release IDs, input hashes, failure
classification, bundle/patch hashes, and replay procedure are recorded in
`/home/user/Desktop/agent_service/docs/production-swe-rebench-pilot.md`.

Codex Responses and Anthropic Messages protocol surfaces are proven on the server,
but neither creates another supported client mode. Host Claude Code remains entirely
untouched. There is exactly one accepted local-agent behavior: the pinned Qwen Code
container through the isolated service into this pinned vLLM backend.

### Software/hardware record and the host contract

Host requirements are functional, not identity pins: the invoked tools must
exist, Docker must respond with its NVIDIA runtime and the
apparmor/seccomp/cgroupns isolation features active, and exactly one GPU with
at least 32,607 MiB — the calibration floor of the locked VRAM/KV budget —
must be present. Exact host software versions, binary hashes, and GPU/driver
identity are deliberately not asserted; pinning them tied the deployment to
one specific computer without making inference any more correct.

Everything inside the pinned images remains exact. The container runtime
record is:

- container CUDA 13.0.3; Python 3.12.3;
- vLLM 0.27.2rc1.dev106+g9df9b0b0a;
- PyTorch 2.13.0+cu130 with CUDA 13.0 (compute capability 12.0 kernels);
- Transformers 5.15.0; Tokenizers 0.22.2; Safetensors 0.8.0;
- Compressed Tensors 0.17.0; FlashInfer 0.6.16.post3;
- Triton 3.7.1; NumPy 2.3.5; FastAPI 0.136.3; Uvicorn 0.52.3.

The environment this profile was originally validated on ran Ubuntu 24.04
with an RTX 5090 on driver 595.71.05 and Docker 29.7.2; that is a historical
observation, not a gate. An update to any locked container component still
requires a new explicit profile/image version and the complete relevant
acceptance suite. A version string in prose is not a pin; the immutable
image IDs, hashes, labels, source reconstruction, and live checks are.

Known nonfatal log noise is documented rather than hidden:

- Transformers emits ERROR-labelled docstring validation messages for missing
  min_frames/max_frames fields; they are documentation noise, not a video path.
- Optional ROCm import warnings occur in the CUDA image; live selection remains CUDA,
  SM 12.0, NVFP4, TurboQuant, and FlashAttention.
- First use of an unseen long-context/image shape can JIT-compile a Triton kernel;
  successful compilation is warmup, not a fallback.
- Deliberate low/disabled-thinking tests log template exceptions and HTTP 400; that is
  expected fail-closed evidence.

Do not silence these messages by enabling remote code, weakening offline mode,
changing cache/image precision, adding retries, or installing host packages.

### Licensing and third-party scope

Original material in this repository for which the repository author owns the
copyright is released under [The Unlicense](LICENSE), SPDX identifier
`Unlicense`. This public-domain dedication does not and cannot relicense material
owned by Alibaba, Unsloth, vLLM contributors, NVIDIA, dependency authors, or any
other third party.

The Qwen model/configuration/tokenizer/model-card material and the pinned vLLM
submodule retain their upstream terms and notices. Review patches or generated
transformations containing modified upstream source likewise remain subject to
the applicable upstream license. The exact scope and preserved Apache-2.0 text
are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). Model weights, local Docker
archives, and caches remain outside Git and are not relicensed by this repository.

# Qwen3.8-27B

> [!Note]
> This repository contains model weights and configuration files for the post-trained model in the Hugging Face Transformers format. 
>
> These artifacts are compatible with Hugging Face Transformers, vLLM, SGLang, TokenSpeed, etc.

> [!Tip]
> For users seeking managed, scalable inference without infrastructure maintenance, the official Qwen API service is provided by [Qwen Cloud](https://www.qwencloud.com).
> In particular, **Qwen3.8-27B** will be available as a hosted version with more production features, e.g., 1M context length by default, official built-in tools. For more information, please refer to the [Qwen3.8-27B Overview](https://www.qwencloud.com/models/qwen3.8-27b). The service is coming soon. Stay tuned for updates.

Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, we are pleased to introduce Qwen3.8, the most capable generation in the Qwen open-model family to date.

Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.

## Qwen3.8 Highlights

Qwen3.8-27B features the following enhancements:
- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.


## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 27B
    - Hidden Dimension: 5120
    - Token Embedding: 248,320 (Padded)
    - Number of Layers: 64
    - Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Gated Attention:
        - Number of Attention Heads: 24 for Q and 4 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
    - Feed Forward Network:
        - Intermediate Dimension: 17,408
    - LM Output: 248,320 (Padded)
    - MTP (Multi-Token Prediction): trained with multiple steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.


## Benchmark Results

### Text Performance
<style>
.vl-table th{font-size:15px!important;line-height:1.2}
.vl-table td:not(.benchmark-cell):not([colspan]){font-size:15px;line-height:1.2;vertical-align:middle}
.vl-table .benchmark-cell{padding:12px 10px 12px 18px!important;vertical-align:middle}
.vl-table .benchmark-capability{font-size:15px;font-weight:600;line-height:1.22;color:#171717}
.vl-table .benchmark-name{margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B}
.vl-table .metric-stack{display:flex;flex-direction:column;gap:7px;padding:3px 0}
.vl-table .metric-label{font-size:10px;font-weight:400;line-height:1.1;color:#777}
.vl-table .metric-value{margin-top:2px;font-size:15px;line-height:1.15;color:#171717}
</style>
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr>
<th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.6-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Muse Glimmer-30B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Opus4.6 Max</th></tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Coding</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic terminal coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Terminal Bench 2.1 (Terminus)</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">73.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.4</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">64.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">51.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>78.2</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-bench Pro</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>61.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">57.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">51.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">53.4</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Repo-level code generation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">NL2Repo-Bench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">42.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">36.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">41.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>47.6</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Agentic coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">DeepSWE 1.1</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>42.2</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">13.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">14.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">QwenSWEBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>79.0</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">49.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">59.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.8</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agent</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Long-horizon office work</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CoWorkBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>70.7</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">61.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">68.2</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Professional job tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">JobBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>33.4</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">21.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Frontier agentic tasks</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Agents' Last Exam</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>20.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>42.9</strong></div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">10.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">27.3</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@1</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">13.2</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Score</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">33.6</div></div></div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
</tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General</td></tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Instruction following</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">IFBench</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>79.5</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">69.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">79.1</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">77.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">GPQA Diamond</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">89.2</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">87.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">90.3</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">83.5</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.3</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multidisciplinary reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">HLE</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">30.8</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">24.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">34.7</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">22.0</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>40.0</strong></td>
</tr>
<tr>
<td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Competitive coding</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">LiveCodeBench v6</div></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>90.3</strong></td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">83.9</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.6</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td>
<td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">88.8</td>
</tr>
</tbody>
</table>
<div style="margin-top:12px;font-size:11px;line-height:1.5;color:rgba(0,0,0,0.72)">
<ol style="margin:0;padding-left:20px">
<li>SWE-bench Pro: Except for Opus4.6 Max, which uses the officially reported score, all models are evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window. Problematic tasks were corrected, and all baseline models were re-evaluated on the refined benchmark.</li>
<li>NL2Repo-Bench: Evaluated with the Claude Code harness. To prevent reward hacking, we disable Bash commands that attempt to access the specific repository, such as pip download, pip install, and git clone.</li>
<li>DeepSWE 1.1: Evaluated with the Claude Code harness at temp=1.0, top_p=0.95, and a 256K context window.</li>
<li>QwenSWEBench: In-house coding benchmark for evaluating models' software engineering capabilities. Evaluated with the Claude Code harness. Reporting avg@3 with an 8-hour timeout, max_tokens=32,768, temperature=1.0, and a 256K context window.</li>
<li>CoWorkBench: In-house cowork benchmark for evaluating long-horizon tasks across computer science, finance, law, medical, and other productivity domains.</li>
<li>HLE: Judged by GPT-4o.</li>
<li>The best result in each row is shown in bold.</li>
<li>Empty cells (--) indicate that results are not yet available or not applicable.</li>
</ol>
</div>
</div>

### VL Performance
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:16px 0">
<table class="vl-table" style="width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px">
<thead><tr><th style="padding:10px 7px;text-align:left;font-weight:600;border-bottom:2px solid #0A2EFE;color:#0A2EFE"></th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;background:rgba(10, 46, 254, 0.08);">Qwen3.8-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.6-27B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Qwen3.7-Plus</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Muse Glimmer-30B</th><th style="padding:10px 7px;text-align:center;font-weight:500;border-bottom:2px solid #0A2EFE;color:#0A2EFE;font-size: 14px;width:14.00%;">Opus4.6 Max</th></tr></thead>
<tbody>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">Agentic Multimodal Intelligence</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Computer use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OSWorld-Verified</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>84.3</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">63.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">65.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">72.7</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Browser use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">WebArena-Verified</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>64.8</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">48.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">55.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Mobile use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">AndroidWorld</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>81.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">70.3</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">81.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.0</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Application recreation</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RecreationBench</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>47.1</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">29.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.2</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal tool use</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ClawEval-MM</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>57.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">56.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">42.6</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">50.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>57.4</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>60.1</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Pass@3</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">52.5</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Average</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">54.7</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Multimodal software engineering</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">SWE-MM</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>38.6</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">25.7</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">30.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">27.1</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual web development</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">Vision2Web</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>62.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">45.0</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">42.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td></tr>
<tr><td colspan="6" style="padding:8px 12px;font-weight:600;color:#0A2EFE;border-bottom:1px solid rgba(10, 46, 254, 0.2);background:#D6DAFC">General Multimodal Intelligence</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Visual math problem solving</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">MathVision</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">90.0</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>94.6</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.1</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.3</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">65.5</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">General visual reasoning</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">BabyVision</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>65.7</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.6</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">28.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">64.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">70.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">12.6</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Scientific chart analysis</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">CharXiv (RQ)</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">83.7</div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>90.2</strong></div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">78.4</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717"><strong>85.8</strong></div></div><div style="margin-top:7px"><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">With CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">85.9</div></div></div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">78.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><div class="metric-stack" style="padding:3px 0"><div><div class="metric-label" style="font-size:10px;font-weight:400;line-height:1.1;color:#777">Without CI</div><div class="metric-value" style="margin-top:2px;font-size:15px;line-height:1.15;color:#171717">66.0</div></div></div></td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Document intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">OmniDocBench 1.5</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">91.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">89.4</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>91.4</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">75.8</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">86.6</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Real-world perception</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">RealWorldQA</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">85.9</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">84.1</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>86.9</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">73.9</td></tr>
<tr><td class="benchmark-cell" style="padding:7px 7px;padding-left:20px;border-bottom:1px solid rgba(128, 128, 128, 0.15);"><div class="benchmark-capability" style="font-size:15px;font-weight:600;line-height:1.22;color:#171717">Embodied intelligence</div><div class="benchmark-name" style="margin-top:4px;font-size:11px;font-weight:400;line-height:1.2;color:#6B6B6B">ERQA</div></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);background:rgba(10, 46, 254, 0.08);vertical-align:middle;font-size:15px;line-height:1.2;">65.5</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">62.5</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;"><strong>69.8</strong></td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">--</td><td style="padding:7px 7px;text-align:center;border-bottom:1px solid rgba(128, 128, 128, 0.15);vertical-align:middle;font-size:15px;line-height:1.2;">40.8</td></tr>
</tbody>
</table>
<div style="margin-top:12px;font-size:11px;line-height:1.5;color:rgba(0,0,0,0.72)">
<ol style="margin:0;padding-left:20px">
<li>MathVision, BabyVision, and CharXiv (RQ): Where both settings are available, cells report “Without CI” and “With CI” separately; otherwise, only the available setting is shown. A small number of incorrect ground-truth annotations in MathVision and CharXiv (RQ) were corrected following manual verification, and all reported scores on those benchmarks were computed using the corrected annotations.</li>
<li>MathVision: Qwen3.8-27B is evaluated using the fixed prompt: “Please reason step by step, and put your final answer within <code>\boxed{}</code>.” For the remaining models, we report the higher score from two prompt variants—one with and one without the <code>\boxed{}</code> formatting requirement.</li>
<li>WebArena-Verified: Scores are computed with the official WebArena-Verified grader under the OSWorld scaffold.</li>
<li>RecreationBench: An in-house, long-horizon application-recreation benchmark designed to evaluate hybrid-agent capabilities across five platforms: desktop (Ubuntu, macOS, and Windows), mobile (Android), and the web.</li>
<li>ClawEval-MM: Scores are reported as “Pass@3 / average score.” Pass@3 is the percentage of tasks passed in at least one of three trials; the average score is the mean benchmark score across the three trials.</li>
<li>Vision2Web: Scores are averaged across the frontend, webpage, and website categories. Evaluations use the Claude Code harness and are judged by <code>gpt-5.4-2026-03-05</code>.</li>
<li>SWE-MM: Scores are evaluated on the Claude Code harness using the public dev split of SWE-bench Multimodal, with the modifications described in Appendix 8.3 of the Claude Opus 4.7 system card.</li>
<li>Empty cells (--) indicate that results are not yet available or not applicable.</li>
</ol></div>
</div>


## Quickstart

For streamlined integration, we recommend using Qwen3.8 via APIs.

### Serving Qwen3.8

> [!Important]
> Inference efficiency and throughput vary significantly across frameworks. 
> We recommend using the latest framework versions to ensure optimal performance and compatibility.
> For production workloads or high-throughput scenarios, dedicated serving engines such as SGLang, vLLM, or TokenSpeed are recommended.

Qwen3.8 can be deployed with popular inference frameworks, e.g.:

- [SGLang](https://www.sglang.io/): [Qwen3.8 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [vLLM](https://vllm.ai/): [Qwen3.8 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [TokenSpeed](https://lightseek.org/tokenspeed/): [Qwen3.8 Recipe](https://lightseek.org/tokenspeed/recipes/models#qwen3-8)


### API Usage

> [!Important]
> Qwen3.8 models operate in thinking mode by default, generating thinking content signified by `<think>\n...</think>\n\n` before producing the final response.
> To disable thinking content and obtain a direct response, refer to the examples [here](#instruct-or-non-thinking-mode).


> [!Tip]
> We recommend using the following sets of sampling parameters for generation:
> - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
> - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
>
> Please note that the support for sampling parameters varies according to inference frameworks.


Qwen3.8 comes with official support for `reasoning_effort`, which can be used to adjust reasoning depth and control cost:  
  - `xhigh` (default): for complex tasks demanding thorough analysis
  - `medium`: balancing accuracy and speed
  - `low`: efficient reasoning optimizing for speed and cost


In addition, `preserve_thinking` is enabled by default for all workloads for the best out-of-the-box experience. To disable preserved thinking, refer to the examples [here](#disable-preserved-thinking).

> [!Tip]
> In multi-turn agentic tasks, lower reasoning effort does not always reduce overall task completion time. Although it may produce faster per-turn responses, it can also lead to insufficient analysis, more failures, and repeated retries, which may increase total latency and token consumption.


#### Chat Completions API

The Chat Completions API can be used with most inference frameworks, as well as [Qwen Cloud](https://www.qwencloud.com/).
Before starting, make sure the OpenAI Python SDK is installed and the API key and the API base URL are configured, e.g.:
```shell
pip install -U openai

# Set the following accordingly
export OPENAI_BASE_URL='your-base-url'
export OPENAI_API_KEY='your-api-key'
```

##### Text-Only Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [{"role": "user", "content": "Write a Python function to merge two sorted linked lists."}]

completion = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,  # on by default
            "preserve_thinking": True, # on by default
        },
    },
    reasoning_effort="xhigh",  # xhigh by default; supported levels are xhigh, medium, and low
    stream=True,
    stream_options={"include_usage": True},
)

reasoning_content = ""
answer_content = ""
is_answering = False
print("\n" + "=" * 20 + "Reasoning" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
        continue

    delta = chunk.choices[0].delta

    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content
    elif hasattr(delta, "reasoning") and delta.reasoning is not None:
        if not is_answering:
            print(delta.reasoning, end="", flush=True)
        reasoning_content += delta.reasoning

    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Answer" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content

messages.append({
    "role": "assistant",
    "content": answer_content,
    "reasoning_content": reasoning_content,
    "reasoning": reasoning_content,
})
```


##### Image Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
            },
            {
                "type": "text",
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
)
print("Chat response:", chat_response)
```

##### Video Input

```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4"
                }
            },
            {
                "type": "text",
                "text": "How many porcelain jars were discovered in the niches located in the primary chamber of the tomb?"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
)

# When vLLM is launched with `--media-io-kwargs '{"video": {"num_frames": -1}}'`,
# video frame sampling can be configured via `extra_body` (e.g., by setting `fps`).
# This feature is currently supported only in vLLM.
#
# By default, `fps=2` and `do_sample_frames=True`.
# With `do_sample_frames=True`, you can customize the `fps` value to set your desired video sampling rate.
# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen3.8-27B",
#     messages=messages,
#     extra_body={
#         "mm_processor_kwargs": {"fps": 2, "do_sample_frames": True},
#     }, 
# )

print("Chat response:", chat_response)
```


##### Instruct (or Non-Thinking) Mode

Qwen3.8-27B will think by default before responding.
You can obtain a direct response from the model without thinking by configuring the API parameters. 
For example,
```python
from openai import OpenAI
# Configured by environment variables
client = OpenAI()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
                }
            },
            {
                "type": "text",
                "text": "Where is this?"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }, 
)
print("Chat response:", chat_response)
```

> [!Note]
> If you are using APIs from Qwen Cloud, in addition to changing `model`, please use `"enable_thinking": False` instead of `"chat_template_kwargs": {"enable_thinking": False}`.


##### Disable Preserved Thinking


By default, Qwen3.8 retains thinking blocks from all historical messages, maintaining a complete reasoning trace across the conversation. This behavior, known as preserved thinking, ensures full context continuity and is especially beneficial for agent scenarios where decision consistency and reduced redundant reasoning are critical. It also improves KV cache utilization, optimizing inference efficiency in both thinking and non-thinking modes.

If you prefer to retain only the thinking blocks from the latest user message, you can disable this behavior by setting `preserve_thinking` to `False`:

```python
from openai import OpenAI

# Configured by environment variables
client = OpenAI()
messages = [...]
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"preserve_thinking": False},
    },
)
print("Chat response:", chat_response)
```

> [!Note]
> If you are using APIs from Qwen Cloud, in addition to changing `model`, please use `"preserve_thinking": False` directly instead of wrapping it in `chat_template_kwargs`.


## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

2. **Adequate Output Length**: To optimize performance on agentic tasks, we recommend allocating sufficient output length to allow the model to generate detailed and comprehensive responses. For frameworks that support separate token limits for internal reasoning and final outputs, we suggest the following configuration within the 1M context length:
    
    - Reasoning Content: Set the maximum output length to 262,144 tokens.
    - Final Response: Set the maximum output length to 131,072 tokens.

    These settings provide the necessary capacity for complex reasoning while ensuring ample space for high-quality final deliverables.

3. **Processing Ultra-Long Texts**: Qwen3.8-27B natively supports context lengths of up to 262,144 tokens. For long-horizon tasks where the total length (including both input and output) exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively, e.g., YaRN.

    YaRN is currently supported by several inference frameworks, e.g., vLLM, SGLang, and TokenSpeed. 
    In general, there are two approaches to enabling YaRN for supported frameworks:

    - Modifying the model configuration file:
        
        In the `config.json` file, change the `rope_parameters` fields in `text_config` to:
        ```json
        {
            "mrope_interleaved": true,
            "mrope_section": [
                11,
                11,
                10
            ],
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "factor": 4.0,
            "original_max_position_embeddings": 262144,
        }
        ```

    - Passing command line arguments:

        For vLLM, you can use
        ```shell
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
        ```

        For SGLang, you can use
        ```shell
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --context-length 1000000
        ```

        For TokenSpeed, you can use
        ```shell
        TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 tokenspeed serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
        ```
    
    > [!NOTE]
    > All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts.**
    > We advise modifying the `rope_parameters` configuration only when processing long contexts is required. 
    > It is also recommended to modify the `factor` as needed. For example, if the typical context length for your application is 524,288 tokens, it would be better to set `factor` as 2.0. 


4. **Long Video Understanding**: To optimize inference efficiency for plain text and images, the `size` parameter in the released `video_preprocessor_config.json` is conservatively configured. It is recommended to set the `longest_edge` parameter in the video_preprocessor_config file to 469,762,048 (corresponding to 224k video tokens) to enable higher frame-rate sampling for hour-scale videos and thereby achieve superior performance. For example,
    ```json
    {"longest_edge": 469762048, "shortest_edge": 4096}
    ```

    Alternatively, override the default values via engine startup parameters. For implementation details, refer to: [vLLM](https://github.com/vllm-project/vllm/pull/34330) / [SGLang](https://github.com/sgl-project/sglang/pull/18467).


## Citation

If you find our work helpful, feel free to give us a cite.


```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
