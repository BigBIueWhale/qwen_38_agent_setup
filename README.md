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

The runtime image `config/runtime-v1.sh` pins awaits release validation;
`./scripts/build-vllm.sh check` states whether it was built from this source revision,
and `start.sh` and `status.sh` refuse before serving when it was not.
Prior live results below describe the versions that earned them.
The audit resolution record in `docs/model-output-and-audit-fixes.md` lists the
source fixes and their container validation. Anthropic tool results now refuse
unsupported nested content, preserve `is_error` in the rendered result, and return
request rejections as native Anthropic errors on both Messages routes.
There is one supported mode:

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
| MTP/speculation | Disabled |
| CPU weight offload | Zero |
| KV offload | One declared resident user context, pinned host tier in /dev/shm; shared prefixes and whole-agent context eviction |
| Batching | One sequence; 2,048-token chunked prefill |
| Listener | 127.0.0.1:8000 only |
| Agent client | Qwen Code 0.21.12 at b965d5f8c24f48e65fb0b17c7d45f34ca4ce8f38 |
| Agent-service release | Pinned by the agent-service release lock, which owns every agent and service image identity |
| Agent-service listener | 127.0.0.1:8090 only |
| Launch profile and cache volume | socket-isolated-nonroot-vision-k8v4-agent-v21 |
| Image profile | Declared in `config/runtime-v1.sh`, and baked into the image as its profile label by the build |
| Runtime image | Pinned in `config/runtime-v1.sh` by the build that made it |

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
- status.sh validates host prerequisites, every ordered vLLM transformation, every reviewed
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
    ./scripts/build-vllm.sh materialise DIRECTORY
    ./scripts/save-images.sh
    ./scripts/restore-images.sh

The check reconstructs the source tree from the pinned upstream commit through every
landmark-aware transformation and assembles the build context from that
reconstruction. The build does the same, then runs offline from the exact base image
on that context, runs the units that need the GPU in the image it made on this
host's GPU, and pins what it made only once they pass: a build of the inputs the
pinned image was built from must reproduce its ID and fails otherwise, and a build of
any other inputs writes its own ID into `config/runtime-v1.sh`. A host without the
validated GPU is refused before anything is built. Materialise writes the verified
reconstruction to a new directory, as a tree in which to author a stage; nothing
reads it. `serve-check` is the check `start.sh` and `status.sh` run first: it
also refuses, before any container is created, a revision whose image inputs
are not the ones the pinned image was built from, since that image does not
carry the source the revision reviews. The mode argument is required, so none of
these happens by default. Save
archives the pinned images and pins the archive by the same rule, and restore
verifies the pinned local archive before loading it.

`check` verifies the Dockerfile's hashes and packaging contracts and runs build
units offline, the Qwen grammar unit among them; it does not execute the
Dockerfile. No build assertion is written only inside it any more: what the build
alone still proves is the image's own assembly -- the copied modes, the removed
modules and the upstream-verifier hashes -- together with the vision MLP unit,
which runs only against the installed tree; the vision workspace unit runs in
check as well. The parser unit parses as
serving does: the reasoning and tool-call parsers the launch names, composed into
the two-pass parser serving builds, with the launch's default template
arguments, on the served model's tokenizer and generation files, each checked
against the model manifest before it is mounted, fed the deltas the native
decoder produces. The image does not carry that tokenizer, so the build runs this
unit in check and in build rather than inside the Dockerfile. The probes that
parse fixed outputs locally build the same composition (`scripts/probe_parser.py`),
named by `run-probe.sh` from the launch. The vLLM test files the reviewed stages
modify are review artifacts the check hashes and does not execute: no pinned
input of the check can run them -- the base image has no test runner, and many
need a GPU or hub downloads -- so what this deployment relies on is asserted by
the units above.

The live probes are launched the same way, through one launcher for the whole
suite:

    ./scripts/run-probe.sh <name>_probe.py [probe arguments]

Each `scripts/test-*.sh` runner is one probe with its canonical arguments. The
launcher refuses a container that is not the locked name running the pinned
image, stages the suite into that container's bounded scratch tmpfs, and runs
the named probe by file. That is what lets a probe carry the identity the
backend requires of every generative caller: `scripts/probe_scope.py` mints a
new `kv_scope` for each conversation a probe holds, from the probe's own file
name, a fresh run id, a sequence number and a label. A probe sends that ID on
every request that continues the conversation, including a redraw of one of its
turns, and mints a new one for a fork, a control or an independent request.
This naming convention belongs to the probe; the backend treats the ID as
opaque. The launcher runs probes by file, which supplies the name used by this
convention.

A probe runs only against the live backend, so `check` proves the half of it
that needs no backend: `scripts/probe_requests_unit.py` loads every probe as the
program the launcher runs and calls its `offline_requests()`, which builds one
request through every body builder the probe sends through -- the functions its
live run calls, with a value of a response's shape where a body continues one --
and validates each body with the request model its route parses, taken from the
API server's own app built offline from the reviewed runtime. A body the probe
sends to be served must be taken; one it sends to be refused must be taken, or
refused for the reason the probe holds the server's refusal to; and every route
a probe names must be one its offline requests reach. A probe that cannot build
its requests fails the check rather than a run on the card.

The build is reproducible on a given host. Every file in the build context
carries `SOURCE_DATE_EPOCH` as its mtime and fixed permissions, and the export
clamps every layer timestamp to it, so reconstructing the source tree afresh for
each build does not change the image ID. Byte-compilation writes `.pyc` files
whose headers embed each copied source file's mtime; those headers therefore
carry `SOURCE_DATE_EPOCH` too, rather than the moment some checkout was written,
which was one of the two measured causes of different image IDs across hosts.
The other remains: the host Docker and buildx versions are deliberately not
pinned, and differing builders can serialise identically-specified layers
differently. Whether two hosts now produce one ID has not been measured, so the
offline archive rather than a rebuild is how a second machine obtains the pinned
image; it makes both machines byte-identical by construction rather than by two
builds happening to agree.

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
The only durable writable runtime state is that exact v21 volume mounted at
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

It is intentionally reconstructed by the ordered, reviewed semantic transformations below:

| Patch | SHA-256 |
|---|---|
| patches/vllm-turboquant-k8v4-direct-workspace.patch | a9721067f1a7ee9497a4bd51e47e3a474561189e881b4704bfc4beac8ea48380 |
| patches/vllm-enforce-auto-tool-schema.patch | 4f75c793a9c2cdcfb2fd0768ba49a4e34748d3a37d8392b07d3592ca50939c07 |
| patches/vllm-qwen38-agent-defaults-and-thinking.patch | bdba0512ed3c997e259856a2909d61bd24b87cf262f4f3a62875a8354a380a0a |
| patches/vllm-qwen38-separate-final-response-budget.patch | f2d88c9ec42e62dcb079d77136d8cbb11443a12304d632924c2e0071098e2b00 |
| patches/vllm-qwen-implicit-tool-grammar-boundary.patch | d231c6e2e7040c4cd4b38432cb8c794805afddbf2c6e4f7ff6febb78e3fd9f48 |
| patches/vllm-anthropic-validation-http400.patch | 2950d1e6659bc961aeb55e8061fad84ee8f9b1295dfb6793f9efa365d28432e5 |
| patches/vllm-tool-truncation-finish-reason.patch | 7ee55158f31b5243b14b9d25ac250511a2fec7f8fca2101d2990d105eaa6b5a0 |
| patches/vllm-qwen38-vision-runtime.patch | 03e091a252702011556eca614c0b41dee9b3643973431004415f3ee2ae987227 |
| patches/vllm-qwen38-numerical-audits.patch | dc0b947db3727b522427a204edd1a930d637476a0f66d7d30e2da65c144ac944 |
| patches/vllm-turboquant-fail-closed-guards.patch | b140445625c63a85d4cae5d878b1de472a9645b63e78c34249848082120b9ba3 |
| patches/vllm-kv-offload-pinning-fail-closed.patch | 3a72f88d679a493d6ae0db6c05e171f9f0be6d44dca958ee351e6099df61a8ec |
| patches/vllm-generation-requires-agent-id.patch | f9e387eb2a57a3c4177d10637faf2f44eb05436d09823cca83d55d4a1d6c20e7 |
| patches/vllm-attention-growth-keeps-prefix-hash.patch | a6c38a841c05bcd4f5bfc573c99f1c4a849e7399af05b1e53096e15a43a97632 |
| patches/vllm-grouped-kv-specs-use-layer-geometry.patch | 6bb249bc143a179ca317c72d2bf70ec118baa6f12dca19c0a59da2e3c935b814 |
| patches/vllm-agent-grouped-offload-retention.patch | 36140417721a858ed623a605874cd7c094171e01acead53d76c851ea3929b551 |
| patches/vllm-agentless-generation-routes-unmounted.patch | c485cf9d7d862c0f4214cd625d598fd8903c43c0e42947e9f847ce4052b156ff |
| patches/vllm-kv-capacity-in-declared-users.patch | f06f1becfcc7c56bb3507d0cfb91e37991a9b89d75112fe69cea76f389a484f4 |
| patches/vllm-kv-declaration-within-physical-bound.patch | a8386795dc7792ed06f63d92159c22323b986bb93e25a0798417243e00a643e5 |
| patches/vllm-exact-reasoning-usage.patch | 34a3291cda667e89ffa97f399b821a06adf9a0b14c7429b121e2b01492b7a8e6 |
| patches/vllm-anthropic-input-fidelity.patch | 3252e25a6c2e9d8ee0eec4cb383fc292bff2afaac2e3becdc1006c68b3b02c3c |
| patches/vllm-qwen-exact-tool-language.patch | e0bdd47262490c88bc600b858e3320efdd4dc4c80c218761fe01378b0e8c8134 |
| patches/vllm-png-source-admission.patch | b1b684a96d7243ae647d4d8ce2fe69b7330b3ab243ea77903c4cfb346bc80549 |
| patches/vllm-kv-physical-free-memory.patch | dfafb63c87ea6383cdac064514339bd65481c89d96fbfa0740bd844244cf830d |
| patches/vllm-qwen-single-call-grammar.patch | 878ba3d98284a326784ffced00a64b38dd827cbc80f136cf1e582df469c3eced |
| patches/vllm-responses-history-integrity.patch | d2c6343087fc287eb6afe315cdfb9caa2909de140c82b57ee9d0c3ed9473983c |
| patches/vllm-responses-stream-identity.patch | 9a3f1fb54f3e22f3df621ab681e675f7a916849bdceb6e555242df29d6028095 |
| patches/vllm-anthropic-terminal-metadata.patch | 532d0087c42445ffa19f0d1ce190fd72b9eb238fda4100f17aa071af598dee6c |
| patches/vllm-generation-sampling-resolution.patch | 8fb1c0311f4e7faa03780914d609bbb109fb609e759eb07196bac2040adfea84 |
| patches/vllm-sampling-decoding-boundary.patch | e5909686b9aa4e591a35e66bcd78fb0fbc7b0ea56bf2210779d934d86c7b28c8 |
| patches/vllm-token-generation-result-integrity.patch | bcb8e56c7f9f53563c16bc3adb8fc10796d0463ac5b85e52ecb367948efbb93e |
| patches/vllm-raw-image-token-transport.patch | 7ff7f7fdc9d72fec7948de4cd8968e157756345ab8487281a36689ed3aa82a63 |
| patches/vllm-xml-text-fidelity.patch | fea2ea6b6837aa30c59a5758837eb039649af16bffd49ffcad603058d59742d3 |
| patches/vllm-phase-aware-parser-terminals.patch | 8310845e39bed950883182690894f3ed94d0e55c8ef16d89c750d925fef65b21 |
| patches/vllm-input-stream-agent-identity.patch | caee1588ff260cb866404a91889b3962f8f94b468fa5f5eb81720f1e8c88d67a |
| patches/vllm-tool-output-completion.patch | 51ec1e129e3467b6569e3557bebd57b2cd2c0ab809c81fb13d9e9a84300e4ce4 |
| patches/vllm-one-way-thinking-boundary.patch | 63a69aea8a184a3875f50673a55058fb9e14dc3178abd8c5510de2770022f346 |
| patches/vllm-schema-faithful-xml.patch | 97e73d566da1bbde490e78da1143d6052e93309b3bb67c90c56fda582af207c2 |
| patches/vllm-token-text-provenance.patch | 954b36cb444f7e644e29d13f7a9d3c000512a0d616bbf2b6cd3cb8f4e880dd44 |
| patches/vllm-precise-request-errors.patch | 5620e9394e9d636b6f875bb9a04c21d54b5a9629f6a6651cda3be79d98be43f2 |
| patches/vllm-qwen-canonical-parameter-framing.patch | d438f9106c4a989d64837c21f5491d946065e6721570ad47664513347f150064 |
| patches/vllm-qwen-owned-tool-grammar.patch | 8eca87e7046eb5b01f37ebc93cafaec01479558ef102201dd74ef3ae8ad8a665 |
| patches/vllm-qwen-unique-tool-parameters.patch | 6a76a61c743807215555cbd6b3bbdd8fcaba4abcaca69ef000d301df6c792d3b |
| patches/vllm-generation-admission-before-response.patch | 6418c42eb3fca2492473e5411f9463d21d1473a0cc55187a2a9a530ddfc44b86 |
| patches/vllm-kv-scope-single-flight.patch | b84b915c38e2c1d32b278bf88dcd5ac7228b9a680b746e2842783dbe1942540a |
| patches/vllm-template-authored-control-tokens.patch | 2a6e8b31826cf06d52acb3c87cae0c6c68a7acc2c8f01dd7848bc5e948977228 |
| patches/vllm-nvfp4-native-kernel-required.patch | 9d9ce188b6670d687a725c4cdca37478f9dc78ef9f19685ddbcf5e9edd53b8b7 |
| patches/vllm-qwen-arguments-read-by-grammar.patch | 9bf29aed999f1cfe12a58cc98b91fccafe615dfb7c22f13e9c681d1b54034837 |
| patches/vllm-startup-plan-admission-bound.patch | 340e4147f03da36ba5ca85db6c0d2f29bf7dbce0aabe4b536e591a582e2dea2c |
| patches/vllm-template-refusals-name-their-parameter.patch | 1f428332be39e4fb7f3c7f7eb5d6e3297f5dc70638fcd7f81d69b32f69d1fa26 |
| patches/vllm-qwen-repeated-parameter-refusal.patch | af405e3be4a649264786bf7bc924c3e4053579eddde47030d9776eb1bc1c73c0 |
| patches/vllm-generated-tokens-survive-parsing.patch | 3709ac24d4f098a27ffa57fedf9d3dec81a1e08392da07d225bb8b62d2e8ee2e |
| patches/vllm-include-reasoning-shapes-the-response.patch | d3b41899464142ffffb04efd0b19647153e322b10c7f9fc1e14af0f290af98d1 |
| patches/vllm-unspecified-tool-choice-is-the-default.patch | 8db1159ea73f23e20a7635b276f56abce085d9b0d83b3ea0d4599f275c5e8746 |
| patches/vllm-call-only-answer-keeps-the-blank-line.patch | 57c69104cb5b569050098993f4abb4e7f88299a43b6ac41f22e267dd8ab177a2 |
| patches/vllm-responses-tools-are-one-function-list.patch | aa54ae92344ca7c9bdfe014a9676b7b63716e3c677e64cee4a8a6ac3b0c104e5 |
| patches/vllm-batch-parse-starts-where-the-prompt-leaves.patch | bc8dc94b88990fbf99549ac8a9b8825bfa02a5225809071e73eab869e12de8db |
| patches/vllm-derender-text-is-the-detokenizers.patch | 9d8f6d45beba5c79671444be4e9604c4352c7fe5346d78573bfc4fbc91bfb4b6 |
| patches/vllm-output-constraints-refused-beside-tool-calls.patch | b82f6259428441aee55d157443bcedc1d98fb247ef9a521106408671a24ae533 |
| patches/vllm-batch-invariance-substitutes-no-nvfp4-kernel.patch | c79baaee0a51275f5f522d266fe6b315c200e6951920c2454d5c80cadbb92f44 |
| patches/vllm-render-carries-every-image-chat-renders.patch | de53b6f2c86c53552a133d0ef93422c73570b53d1eace1d9e8f835accf9f5181 |
| patches/vllm-rendered-prompts-are-never-truncated.patch | 72e74c44c7c8b8ff3db8c7f73124fa82e196244f2b0f2f666a2315f3cef4cdc9 |
| patches/vllm-kv-transfer-params-are-declared.patch | 9742698af30ad79158f5430fb22987cafe4667f457771dd66b29b38a7376ce0e |
| patches/vllm-responses-refuses-tools-the-template-is-never-given.patch | ee2c83118ee21815cfb582d432c61b9bebbde1845858f4c294bab0ce392abf85 |
| patches/vllm-chat-stream-carries-every-token-logprob.patch | ca15dadd152454fe5b3fbcb710b8c7b5ce3221c038617b9e0d0985953aecbf47 |

The reconstructed tree's runtime-source and test changes, new files and
deletions are counted by ./scripts/build-vllm.sh check, which derives and prints
them; they are not restated here. The landmark-aware Python patcher calculates every mutation
(including file deletions) before writing, validates unique structural landmarks
and complete pre/post hashes, performs atomic transactions with rollback, and is
itself covered by failure-path tests that the check runs. The
unified diffs remain review artifacts, but they do not select mutation locations:
each stage's data and its review diff must describe the same blocks at the same
place. Every hunk must land at the line its header names, every index line must
name the blobs the stage transforms, and every created or deleted file must be
declared as one, so a landmark that matched an identical block elsewhere in a file
is refused rather than applied. The compiler that writes the data anchors each
hunk at that line instead of choosing the nearest occurrence of its block.
The build check rejects an ambiguous landmark, missing hunk, misplaced hunk, wrong
stage, changed final hash, whitespace error, partial intermediate state, concurrent
source drift, or a reconstruction whose footprint is not exactly the patch set's.

The image is built from that reconstruction and from nothing else. Every stage is
applied from the pinned commit to a worktree of the build's own under complete
pre/post hashes, and git must then find that worktree changed in exactly the paths
the committed stage data changes, creates and deletes -- a footprint the framework
derives from that data, never a list kept by hand. The build context is copied out
of the reconstruction: each file `containers/Dockerfile.runtime` copies, from the
reconstruction when it lies under `vllm/` and from this repository otherwise, and the
Dockerfile itself, each at mode 0644 with `SOURCE_DATE_EPOCH` as its mtime. The CPU
units run against those same copies. What the check proves is therefore what the
image carries, and nothing but this repository's files, the pinned commit and the
base image can enter it.

The `vllm/` submodule is the pinned upstream checkout and nothing else. No check or
build reads its files -- they reach it only through git, for the pinned commit's
objects -- and git reports an edit there like any other uncommitted change, which
`start.sh` and `status.sh` refuse. A stage is authored in a tree of its own:
`./scripts/build-vllm.sh materialise DIRECTORY` runs the same verification as `check`
and writes the verified reconstruction to DIRECTORY, a new detached worktree of the
submodule at the pinned commit, with the reconstruction recorded in its index.
`git -C DIRECTORY diff` is then exactly what the author changes on top of it (after
`git add --intent-to-add` for a new file), which is the review diff
`patches/source_patch_v1/compile_review_diff.py` compiles into stage data. The verb
refuses a directory that exists, so it never writes over work, and a reconstruction
that fails any step is removed rather than left behind. The tree is removed with
`git -C vllm worktree remove --force DIRECTORY`.

Pinned build inputs and products:

| Item | Identity |
|---|---|
| Immutable base tag | qwen38-vllm:main-9df9b0b |
| Immutable base ID | sha256:fa4a002a88b7043a1a89966dea8a500fe9696f84e75730d9da916f916048d401 |
| Runtime Dockerfile SHA-256 | 319a602d81d5f30696609ad36c9e3ea421c08ef5d97623222eeec6ba1866f4fe |
| Build verifier SHA-256 | 1cfd9e55f63e92d586d5c19ae48ad235f235ceb884a9cc3919ca0f59fe99f908 |
| Runtime validator SHA-256 | e743d2bdca2bb34960f3b5bc8240d0361a7479539a2980a4aae552be29c74f7e |

The runtime image's profile, tag and archive name, which every release advances
together, are declared in `config/runtime-v1.sh`, and the archive lives under
`artifacts/` by that name. The runtime image's ID and the archive's SHA-256 live
in the lock alone, which is also where `agent_service` reads them and their
history. The build writes the first and `./scripts/save-images.sh` the second,
each together with the digest of the inputs it was produced from, and a later run
of the same inputs must reproduce what is pinned: a build is refused if it makes
another image, and a save if it writes other bytes.

The runtime tag names the pinned image and nothing else. A build used to load its
image under that tag before comparing the image ID with the pin, so a build that
did not reproduce the pin took the tag from the pinned image, which was left
untagged and indistinguishable from a failed build. That happened twice: the v23
tag on the machine that cut v23 names an image no lock ever pinned, while the
pinned v23 image is untagged, and the first v19 image lost its tag when v19 was
re-pinned. Now `./scripts/build-vllm.sh build` loads its image under a name of
the run's own, checks it by its ID, and moves the runtime tag only to the image it
then pins -- one that reproduced the pin, or the build of new inputs that wrote
it; `./scripts/restore-images.sh` moves it only by loading
the archive the lock pins. Every runtime image also carries an identity tag --
the release its archive is named for and the image's own ID, derived in
[`config/runtime-v1.sh`](config/runtime-v1.sh) from values the lock already holds:
`IMAGE_IDENTITY_TAG` for the pinned image. A build that does not reproduce the pin
is kept under its own identity tag and named in the refusal, and the pinned image
keeps its identity tag when a re-pin moves the runtime tag on. An identity tag
that already names another image is refused, never moved. Which of
these images and archives a host keeps is decided by `agent_service`'s
`./collect.sh`, where every session record and benchmark pass that references a
backend release lives; its README states the policy.

Every reviewed runtime file, including both TurboQuant kernels, is copied and
hash-checked against its upstream and patched identities. A CPU Triton-interpreter
build unit executes the installed K8V4 store with finite, NaN, infinity and
metadata-overflow inputs and checks the decode hardware guard. A separate recipe
unit refuses missing copies or installed-file checks. CPU interpretation is not GPU
acceptance: the two units that need the GPU -- the installed K8V4 store and fused
decode against PyTorch references, and the checkpoint's worst-error NVFP4 layer
through the production kernel -- run in every image `./scripts/build-vllm.sh build`
makes, on the host's GPU with the verified model read-only at `/model`, as the
server's user and in its environment, and the build pins the image only if both
pass. They are the only shipped units the CPU unit loop does not run.

The final runtime layer does no package resolution or installation. It is built with
pull=false, network=none, provenance=false, an exact base ID, a context assembled
from the reconstruction, upstream installed-file hashes, final installed-file hashes,
and build-time invariant tests. Independent offline builds produced the identical v13
image ID.

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
       "presence_penalty":0.0,"repetition_penalty":1.0}
    --quantization compressed-tensors
    --dtype bfloat16
    --kv-cache-dtype turboquant_k8v4
    --max-model-len 262144
    --max-num-seqs 1
    --max-num-batched-tokens 2048
    --kv-cache-users 1
    --cpu-offload-gb 0
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{"cpu_kv_cache_users":1}}'
    --enable-prefix-caching
    --enable-chunked-prefill
    --attention-config.flash_attn_version=2
    --kernel-config.enable_flashinfer_autotune=False
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --enable-prompt-tokens-details
    --default-chat-template-kwargs
      {"enable_thinking":true,"reasoning_effort":"xhigh","add_vision_id":false}
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
    VLLM_ENFORCE_STRICT_TOOL_CALLING=1
    GLOO_SOCKET_IFNAME=lo
    NCCL_SOCKET_IFNAME=lo

The image itself sets `DO_NOT_TRACK=1`, so usage reporting is off whatever
starts it: vLLM's usage reporter (stats.vllm.ai, on by default upstream) and
huggingface_hub's telemetry both read that one variable, and the build asserts
both are off in the image it produces.

Consequences:

- There is no language-model-only flag. The complete vision model is loaded.
- There is no speculative/MTP argument.
- CUDA graphs remain enabled. Vision was not bought by forcing eager text execution.
- The measured 2,048-token prefill chunk remains fixed. Vision was not bought by
  reducing text prefill performance.
- FlashAttention 2 is explicit because automatic selection otherwise changes the
  TurboQuant path. FlashInfer autotuning is explicitly off so probe-OOM-and-fallback
  behavior cannot be mistaken for normal startup.
- CPU weight offload is exactly zero. KV offload is not: the OffloadingConnector runs
  in kv_both role with a pinned host tier in /dev/shm sized as one declared resident
  user context (bytes derived in-engine from max_model_len and the KV cache spec).
  Agent IDs decide what this tier retains, not what a lookup matches; see
  [Shared prefixes and agent IDs](#shared-prefixes-and-agent-ids).
- A request's `kv_transfer_params` are parameters of the configured KV connector,
  and each connector declares the keys it takes. This launch's CPU tier takes
  `max_offload_tokens` alone: `kv_load_tiers` selects among secondary tiers it has
  none of, and `do_remote_prefill` belongs to the NIXL, Mooncake and MoRIIO
  connectors. Any other key, and a `kv_transfer_params` that is not an object
  (including one sent through `vllm_xargs`), is refused with a 400 naming the key,
  the connector and what it takes, before the request reaches the engine. A
  connector that declares nothing takes nothing: an out-of-tree connector, and
  LMCache's (without `use_native`) and FlexKV's adapters, which hand requests to
  their own packages. Their callers remove the key; their operators declare it by
  overriding `KVConnectorBase_V1.get_kv_transfer_params_keys`, or for LMCache set
  `use_native`, whose adapter declares its keys. A taken key's value is not
  checked here: a malformed `max_offload_tokens` is still logged and ignored by
  the connector.
- Multimodal profiling is mandatory and cannot be skipped to obtain a deceptively
  optimistic allocation.
- All unquantized model computation, including the entire vision tower, uses BF16.
- trust_remote_code remains false. Current vLLM resolves the architecture natively as
  Qwen3_5ForConditionalGeneration.
- Request-side media limits, processor overrides, and lower image-detail choices are
  rejected by the image's own media contract rather than silently replacing the profile.

### Only the template writes control tokens

The served tokenizer marks `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` and the
vision, audio and speech markers as special tokens, and `<tool_call>`,
`</tool_call>`, `<think>`, `</think>`, `<tool_response>`, `</tool_response>` and
the fill-in-the-middle markers as added tokens that are not special;
`split_special_tokens` is false. Upstream vLLM renders the chat template to one
string and tokenizes that string whole, so request text that spells one of those
markers -- a file the model reads, a command's output, a tool result, a user
message, the model's own earlier turn as the client re-sends it -- becomes that
control token: a forged turn boundary, thinking boundary or tool call, and a way
for a repository to end a turn and open another inside what should be inert
content. Setting `split_special_tokens` would not end it: the template's own
`<|im_start|>` is in the same string and would be split too, and `<tool_call>`
and `</think>` are not special, so they would not be split at all.

The template renders with authorship (`vllm/renderers/template_authorship.py`,
stage `template-authored-control-tokens`). Every character of the rendered prompt
carries whether the template or the request wrote it. The template's constants
and the tokenizer's special-token variables are the template's; every string the
request supplies -- messages, reasoning, tool calls and their arguments, tool
results, tool schemas, documents, request template arguments -- is the request's,
and authorship follows the text through the operations templates apply to it
(concatenation, macros, `trim`, slicing, `split`, `join`, `replace`, `tojson`,
`safe`). Encoding matches the added vocabulary only in text the template wrote
and encodes the text between those tokens with no added vocabulary at all, so
request text can only become ordinary vocabulary (every id below 248,044),
whatever bytes it holds. Nothing the model reads is rewritten: a file holding
`<|im_end|>` reaches the model as those ten characters, in the ordinary tokens
that spell them. The template's markers are the ids they always were, and a
prompt whose request text spells no added token encodes to exactly the ids the
whole-string encoding gives. A chat is rendered straight to token ids; its text
is kept for logging and `echo`, never tokenized again.

Text produced by a template operation that does not carry authorship has no
author. If such text spells an added token, nobody can say whether the template
or the request wrote it, and rendering refuses with a server error naming the
token and its position. `scripts/template_authorship_unit.py` proves both halves
for the served template in every build and check: statically, every filter and
method it applies keeps authorship or yields no text; by rendering, a
conversation that takes every branch of the template, with every request field
spelling every added token of the served vocabulary, encodes to exactly the
added tokens of the same conversation with those spellings removed. vLLM's
`string` content format writes multimodal placeholders into message text, so it
is refused for multimodal requests (this deployment serves `openai`), and chat
rendering needs a tokenizers-backed tokenizer.

Proven offline against the served tokenizer and template, no model: every
request body of the two recorded runs (548 prompts, 61,395,519 tokens, both
bundles hash-verified) renders through the served chat path to identical ids,
identical text and the same 95,404 added-token ids before and after, and every
prompt's ids decode back to its text. Every added-token spelling placed in a
file read, a command output, a user message, the system prompt, the model's
history (content, reasoning, tool-call names and arguments), a tool schema, a
continued final message, and beside an image yields exactly the template's
markers; before, the same content forged up to 313 more, and an `<|image_pad|>`
spelled in the text ahead of an image took that image's 70 embedding positions
while the image's own placeholder stayed a single unexpanded token.

What changes for the model, stated rather than hidden: when the model itself
emits a control-token id where the template writes none -- the tool grammar
lets a string argument contain `</think>`, `<tool_call>` or `<|image_pad|>` as
an id, and an answer may carry any special token its format does not act on --
the parser returns its text, the client keeps that text, and it returns in its
history as the ordinary tokens that spell it. Before, every ordinary-token spelling the model wrote came back
as a control token it never wrote. The mismatch that remains begins with the
model's own emission of a control id where no template would write one, and the
agent service records the generated ids that show it.

Surfaces where the client writes the prompt itself have no template, and their
client is the author of every token: `/v1/completions` with a text prompt and
`/inference/v1/generate` with token ids. Every chat surface -- Chat Completions,
Responses and Anthropic Messages -- has the template as the one writer of its
prompt: nothing a chat request carries, for the KV connector or otherwise, takes
the rendered prompt's place.

### Shared prefixes and agent IDs

Every generation request supplies an opaque `kv_scope` agent ID containing a
non-whitespace character, and one ID names one line of work: send the same ID
for every request that continues a conversation, and a new, never-used ID for a
fork or a subagent. The generation request models require the field and the
served OpenAPI schema declares it required; the render models they share their
shape with (`/v1/chat/completions/render`, `/v1/completions/render`, and the
token request those return) allocate no KV and have no such field. A request
that omits the ID, or sends a blank one, is refused with HTTP 400 stating that
rule. The same rule is the engine's invariant for every other caller:
`require_kv_scope` in `vllm/vllm/v1/engine/input_processor.py` refuses a direct
SamplingParams caller that names no line of work. The supported generation
APIs are Chat Completions (including batch Chat), Completions, Responses,
Anthropic Messages and token-in-token-out generate: five protocol families and
six routes. Generative scoring and Cohere are not mounted; neither is part of
this deployment's agent-identity contract. Pooling allocates no KV and does not
require an ID.

Every generation route finishes admission before it answers. Input validation,
the ID rules and the overlap rule below are applied by `AsyncLLM.admit` before
a streaming response is returned, so each refusal is an HTTP 400 on streaming
and non-streaming requests alike, never an error inside a stream already
answered 200. The admitted stream owns its request: a client that disconnects,
or a response that is never read, aborts it. What can still fail after the
status line (the engine failing mid-generation) ends a Chat, Completions,
Anthropic or token stream with its error chunk, and a Responses stream with its
`error` event.

One line of work generates one sequence per request, so the ID can name it:
`n` must be 1 on Chat Completions, Completions and token generation (and on the
batch route, as must `best_of`), and a Completions request carries one prompt.
Several samples or several prompts are several lines: send one request each,
each with its own new ID; they share any common prompt prefix by content
(below). The batch route takes one ID per conversation, as a list in the order
of `messages`, all distinct.

A line of work also has at most one request in flight. The frontend refuses a
second request under an ID whose earlier request has not finished, with HTTP 400
naming the ID and the request in flight: wait for it to finish or abort it, since
a continuation must not overlap its predecessor, and give a concurrent line of
work its own new ID. The frontend answers this because it is one API server
process that admits every generation request; the engine refuses to start with
more than one (`--api-server-count` other than 1). An ID is free again as soon
as its request's last output is produced, before the response's last byte, so a
continuation sent after reading a response to its end is admitted. The cost: a
client that times out and re-sends before the server has aborted its first
request (the server aborts it when it notices the disconnect) gets this
refusal, and must retry once that abort has happened.

The ID never enters a block hash and never restricts a lookup. Prefix lookup
matches content and `cache_salt` alone, in the GPU block pool and the CPU tier,
as upstream vLLM does: any request may reuse any resident prefix whose tokens
and salt it shares, whether its ID is new or already has cached blocks. No ID
choice can serve one agent another agent's state, since reuse needs the same
tokens, and none isolates agents. Isolation is `cache_salt`, a hash input the
server never derives from the ID. The ID has no prescribed format and declares
no parent or lineage. Any request can observe shared-prefix hits through
latency, so IDs carry no authentication or confidentiality promise; no timing
padding is added.

The ID controls one thing: **what the CPU tier retains.** `CPUOffloadingManager`
in `vllm/vllm/v1/kv_offload/cpu/manager.py` keeps at most one finished context
per ID. The ID's next request drops it as soon as that request retains any data
(`retain_context`), and completion makes the finished request the ID's retained
context (`on_request_finished`). Concurrent requests of one ID keep separate
working sets only while active. Under pressure, `_prepare_store` reclaims
unreferenced chunks first, then releases whole IDs in least-recently-used order,
never the storing ID; references held by surviving contexts keep their shared
data. Retained contexts are bounded by the tier's chunk count: past it, the
least recently used retained context is dropped. GPU block eviction is
per-block LRU and ignores the ID.

So the ID the harness sends decides which retained context a request replaces
and which line pressure releases as a unit. It does not decide what a request
matches, and it does not decide what stays resident: whether an earlier context
is still there to match depends on both tiers' capacity, least-recently-used
order and the retained-context bound.

| Caller case | Send | What the code does |
| --- | --- | --- |
| Next turn of the same conversation | The same ID | The new turn replaces the ID's retained context; chunks only the old context used become the first reclaimed. |
| Retry or redraw of the same turn | The ID the conversation continues under | The redraw replaces the discarded attempt's retained context. |
| Fork sharing history up to a point | A new ID, first used by the fork's first request | The fork matches the parent's history while it is resident. Parent and fork then keep separate retained contexts over the shared chunks. |
| Subagent with its own conversation | A new ID | It matches any identical resident prefix, such as the system prompt and tools. The parent's retained context stays under the parent's ID, released only by whole-ID pressure or the retained-context bound. |

A waiting parent survives a subagent only while both fit. This deployment runs
one sequence at a time (`--max-num-seqs 1`), so a foreground subagent runs after
the parent's request finished. In the CPU tier the parent is then the least
recently used ID besides the subagent: when a subagent store needs more rows
than free and unreferenced ones, `_prepare_store` releases the parent first
(after any older ID) and reclaims every chunk the subagent does not share with
it. The tier holds one full-length context (`cpu_kv_cache_users:1`), so the
parent's retained context survives exactly while the parent's chunks and the
subagent's unshared chunks, each context with its own trailing recurrent-state
chunk per recurrent group, fit in that one context's rows. The GPU pool is one
context too (`--kv-cache-users 1`) and evicts the parent's freed blocks first as
the subagent allocates beyond the free pool; losing the parent's recurrent-state
blocks there forfeits its GPU hit and leaves the CPU copy, if it survived, as
the parent's only cached context.

Each of several forks or subagents needs its own ID, and the server refuses
the ways one ID could name several at once: overlapping requests, `n > 1`,
several prompts, and a batch that repeats an ID.

Scoping mistakes never change output. They cost retention:

- **One ID for two lines of work**, such as a subagent or fork on the parent's
  ID: each line's next request drops the other's retained context, so while the
  parent waits, its chunks the child does not use become the first reclaimed
  under CPU pressure. That wait is what the CPU tier is configured for
  (`config/runtime-v1.sh`).
- **A new ID per turn or retry of one conversation:** the prefix still matches.
  Each abandoned ID keeps its last context retained as a whole agent, however,
  so least-recently-used release and the retained-context bound can take other
  agents' contexts first.

Beyond that, nothing verifies correct use. The server checks that the ID is
present and contains a non-whitespace character, that it has one request in
flight and one sequence per request, and that an input stream keeps one ID
across its chunks: a change is refused before dispatch and aborts only that
stream's request. It cannot tell a continuation, retry, fork or subagent apart.
Correct scoping rests on the caller. Its effects are visible only as
prefill latency, the served cached-token count
(`prompt_tokens_details.cached_tokens` on Chat Completions; see
[Exact served usage](#exact-served-usage)) and the server's prefix-cache metrics.

See [the KV design](docs/kv-user-count-and-agent-scope-design.md) for CPU
window availability, retention, secondary storage and capacity math.

### Agent defaults: xhigh thinking, exact sampling, long output

The one deployment is explicitly configured for complex agentic work. A client that
omits Qwen-specific fields receives all of these server-side defaults:

    enable_thinking       = true
    reasoning_effort      = xhigh
    add_vision_id         = false

    temperature           = 1.0
    top_p                 = 0.95
    top_k                 = 20
    min_p                 = 0.0
    presence_penalty      = 0.0
    repetition_penalty    = 1.0

These sampling values are Alibaba's published Qwen3.8 thinking-mode tuple.
repetition_penalty=1.0 is neutral. The historical Qwen3.6 repetition detector is not
used; SamplingParams.repetition_detection is None. Adding a heuristic repetition
intervention would alter the learned distribution and is not part of this profile.

xhigh is the canonical effort. Omitted effort, OpenAI high, OpenAI max, and the
supported Anthropic high/max-style forms render the same xhigh model token IDs.
Medium, low, disabled thinking, and incompatible Anthropic controls are rejected with
HTTP 400. The supported agent client always sends xhigh and never asks for a weaker
mode.

Reasoning already in the conversation is never dropped. Every assistant turn is
rendered with its reasoning, and `preserve_thinking` accepts only `true`: omitting
it renders exactly as `true`, and any other value that reaches the template is
rejected with HTTP 400.

Dropping reasoning from earlier turns is a good idea when it is correctly
implemented: it trades compute for slower context growth and stability. For
Qwen3.8-27B it is not correctly implemented, even though the upstream card below
documents `preserve_thinking: false` as supported. The model's own template keeps
thinking only for assistant turns after the latest `role: "user"` message. A
one-prompt agent task therefore sheds nothing, and an agent client that injects
reminders as user messages moves the cut into the middle of the current task; Qwen
Code does so every third tool turn (`ACTIVE_TODO_REMINDER_REFRESH_TURNS = 3`). What
survives depends on when the client injects a message, not on any rule. Inventing a
rule instead, such as keeping the last K assistant turns, is no fix: it renders a
history the model was never trained on. Qwen3.8-27B was trained with preserved
thinking, as nearly all current models are, and there is no correct off switch for
it.

The guarantee therefore lives in the served template rather than in a launch
default, so it holds for every caller: `scripts/derive-chat-template.py` derives that
template from the model's own through landmark-anchored stages, and
`./scripts/build-vllm.sh check` refuses unless it reproduces the served bytes.
`--default-chat-template-kwargs` accordingly names no retention field, and the
upstream card's "Disable Preserved Thinking" section below does not apply to this
deployment. The derivation and the rejected alternative are recorded in
`docs/qwen36-to-qwen38-audit.md`.

No phase budget is served. The usable generation allowance is the minimum of the
request's total generation ceiling and the physical context remaining after exact
rendering. A short prompt can use extensive reasoning; a nearly full prompt cannot.

A request may still set its own phase budgets: `thinking_token_budget` for reasoning
and `final_response_token_budget` for the visible answer, or Anthropic's
`thinking.budget_tokens` for the first, and each is applied exactly as sent. The
final counter starts only after the explicit reasoning-end marker. Tool XML is a
structured tool phase, not visible final prose. EOS and stop sequences may end
earlier; min_tokens cannot cross a request's phase budget. Chat, Completions,
Responses, Anthropic Messages, and `/inference/v1/generate` are the five supported
generation families; batch Chat shares Chat's policy. All inherit the configured
sampling defaults. A live five-real-token final-budget probe stopped at exactly five
final tokens.

Generation settings resolve after the rendered prompt length and server defaults
are known. Omitting the total token limit uses the remaining window subject to the
configured maximum. Explicit neutral sampling values keep their meaning, including
through the render-to-generate JSON boundary. `/generate` preserves supplied fields
until resolution, then constructs a fresh validated engine request; a temporary
engine default cannot reject an otherwise valid request or truncate its output.

The token-generation transport selects delta output for streaming and complete
output for batch requests; `output_kind` is an internal engine setting. Both
transports require an observed terminal for every requested choice and preserve
`stop_reason`, including a matched string, stop token ID or final-budget cause.
An empty token list can carry a valid terminal. Parallel choices retain their
own token sequences and contribute to the same total usage. Missing, repeated or
contradictory engine terminals are server errors. A streaming error ends with its
error event and no `[DONE]` marker. Derender preserves the supplied terminal cause
and accepts a completed empty answer -- the model's end of turn alone -- on both
Chat and Completions. A choice that reports a stop on a stop token ends on it, as
every generation the engine stops does: `stop_reason` null on one of the model's
EOS ids, which derender reads as the engine does, and a stop token ID on that ID.
Any other is refused as HTTP 400 whose parameter names the choice, and no token
loses its text. Derender does not cut the text at a matched stop string: the
string and whatever the ids decode to after it stay in its text, where Chat cuts
the text at the string.

Beam search is unsupported. Chat, Completions, batch Chat and their render requests
refuse `use_beam_search: true` as HTTP 400 with parameter `use_beam_search`, before
rendering or engine dispatch, for both stream settings. Their schemas admit only
false or omission. The served generation path applies sampling, grammar, phase
budgets and output parsing together.

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
temporarily releases that workspace around vision encoding, then restores it
exactly. It does not change cache capacity, text prefill size, graphs, weight
precision, or image precision. Logs from maximum images show release/restoration
and no OOM, retry, preemption, or fallback.

No other reserve is held for the encoder. A reserve held through text execution
and released around encoding would add nothing to the encoder's room -- its bytes
are free during encoding whether or not it was held -- and would only take them
from text execution. The v13 image below held one, 640 MiB of raw CUDA memory
released and restored with the workspace; its free readings are taken with it
held.

The declared pool is admitted against a bound the engine derives at every startup
from what it profiles; no number in the lock or the launcher sizes it. The bound is
the device memory free when this instance started, minus what serving holds beside
its pool:

- the residents measured when profiling ends: weights, non-torch allocations,
  every workspace (the 1,024 MiB continuation workspace among them) and the input
  batch's block tables;
- the peak of the profiled phases above them, each run at its largest in the
  residency serving runs it in: the vision encoder at its full budget with the
  continuation workspace released, then a 2,048-token text step with attention
  executed against a stand-in pool at the full 262,144-token context, its encoder
  outputs still cached and the workspace resident, then its sampler and its prompt
  log probabilities, each with the most log probabilities a request is admitted
  with. The peak is therefore the largest of these, each with the workspace it
  holds;
- the CUDA-graph estimate and the frontend's multimodal reservation.

A warm pass before the measured one takes one-time compile and autotuning memory
out of the measurement, and the stand-in pool's bytes come off the peak because the
declared pool takes its place. The startup log states every term.

The text step shows what it ran by running it. It runs over as many requests as a
step holds and, when that splits the step, once more as one request -- the longest
chunk a request is scheduled, which TurboQuant's continuation path reaches only
alone (this launch, at one sequence, runs it once). A layer that holds KV cache or
state reads its attention metadata only on the path that executes against that
cache or state, and the dummy step records which layers read it; the continuation
workspace is requested only by the continuation path, and the workspace manager
records the request. Startup is refused, naming the layer or the workspace, unless
every layer of the model's pool read its metadata and every reserved reclaimable
workspace was requested, so a profile that stopped running attention or a state
update refuses startup instead of printing a smaller peak. A layer that reads its
metadata and then returns before its state update is not caught by the first
witness; for TurboQuant the second catches it.

Prompt log probabilities (chat's `prompt_logprobs`, or `echo` with `top_logprobs`)
are computed as upstream's V2 runner computes them: the logits of at most 1,024
prompt rows at a time, and only the requested scores. Computed as upstream's V1
runner computes them -- a float32 log-softmax over the 248,320-token vocabulary for
every row of a 2,048-token chunk, beside the chunk's bfloat16 logits, about
2.8 GiB -- the phase would not fit beside this launch's one-user pool, and the
profile, which runs every phase serving runs, would refuse the pool. Computed this
way it holds about 0.47 GiB of logits, below the encoder's peak (by arithmetic
from tensor shapes, not measured).

Only the V1 model runner holds a declared pool. Upstream selects its V2 runner by
default for dense, non-hybrid models; that runner's startup profile runs no
attention, builds its attention backends and their workspaces only after
profiling, charges no CUDA-graph memory and samples without log probabilities,
which upstream's utilization holdback absorbs and this bound does not charge. A
configuration with `--kv-cache-users` is therefore one the V2 runner does not
support: by default such a model is served by V1, and forcing V2 is refused at
configuration with the cause and the V1 runner as the next action. This launch's
hybrid model is served by V1 either way.

The derivation assumes that the caching allocator places each phase's peak,
counted in allocated bytes, in the room the bound leaves: zero fragmentation. It
charges the pool its exact bytes, not the allocator's rounding of each pool tensor
up to a 2 MiB multiple (at most 2 MiB per tensor, about 25 MB for this pool's
sixteen). A run on the card confirms these assumptions and that the profiled text
step runs on this model; it does not decide the bound, and a profile that fails
stops startup rather than admitting the pool. By the v29 startup log's arithmetic
(bound 6.93 GiB, pool 6.45 GiB, profiled peak 1.85 GiB, a 0.48 GiB margin), the
one-user pool stays admitted unless the largest of the text step's, its sampler's
and its prompt scoring's peaks with the workspace resident, plus the residents the
profile now counts beside it (attention builders, block tables, TurboQuant's
per-layer arange cache), exceeds about 2.3 GiB. Reading the
code puts that near 1.75 GiB; the profile decides it at each startup.

On the exact final v13 live image:

- immediately before the fifteen-image maximum-quality run: 31,647 MiB used,
  464 MiB free;
- after the complete backend suite and recorded v8 agent acceptance: 31,797 MiB used,
  314 MiB free;
- total reported VRAM: 32,607 MiB.

The post-suite reading is an observed residency point, not a claim that generation
always peaks at exactly that number. The small global free number is not the memory available during a vision
encode: the 1,024 MiB workspace is released around that phase (on v13, with the
640 MiB reserve).
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
internal engine feature list was deliberately reversed. Complete derivations, test code,
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

The transport/decoder contract is intentionally narrower and fail-closed, and it is
the image's own: it holds for every caller and every launch of the image, not only
for a stack started by this repository's launcher.

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

The token generation API uses the same raw-image boundary. `/render` carries the
original accepted inline PNG parts in `content_parts`, in image order, beside the
fully rendered `token_ids`. `/generate` decodes and processes those images through
the native processor, computes their hashes, and validates every complete image
span against the supplied tokens. It preserves the exact tokens without a second
placeholder expansion. Processor cache hits still provide complete image data to
the engine. Caller-supplied tensors, hashes, placeholder positions, UUIDs and
unknown content parts are refused at the request boundary. The render result can
be sent directly to `/generate`, including its sampling settings and cache salt;
generation requires an agent ID as described in the cache contract.

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

Chat, Anthropic and Responses replay share the same ordered tool-history gate.
Every declared call requires one result with its transport ID, in call order,
before the next turn. Orphaned, missing, duplicate and mismatched results are
request errors before rendering. Responses preserves every supplied reasoning,
summary and assistant text block in order; text blocks concatenate without
trimming or invented separators. Mixed text/media/refusal parts remain available
to the shared content parser. Summary-only reasoning retains all summary text;
encrypted reasoning is unsupported and refused explicitly.

Responses streaming keeps each output item's ID and each function call's
`call_id` through its added/done events and the terminal response. Terminal
output uses the completed stream items, so callers can replay it with results
correlated using the IDs first received in the stream; each streamed item is
exactly the concatenation of its deltas, which the parser unit asserts for every
engine chunking. Usage, the response's own status and every item's status are
the same on both paths: in a truncated response the item the limit cut -- the
last -- is incomplete, and the items it finished before the cut are completed.
A message carries no log probabilities. They would be those of the tokens its
text came from, and the model writes reasoning, message and calls as one token
sequence in which a single token can end the message and begin a call, so no list
of whole tokens is the message's: `include: ["message.output_text.logprobs"]` and
a non-zero `top_logprobs` are refused, each by name. Chat Completions reports the
log probability of every token a choice generated, in order -- the end of
reasoning, the calls and the end of turn included -- the same in the full
response and across the streamed chunks, for every engine chunking.

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

### Tool calling and protocol correctness

The server explicitly selects the qwen3 reasoning parser and qwen3_coder tool parser.
Automatic tool choice is enabled, parallel tool calls are disabled in the supported
agent policy, and every selected tool is constrained to its declared JSON schema
whether strict is omitted, false, or true. A normal non-tool answer and
tool_choice=none remain valid. Unknown names, wrong nested types, extra properties,
duplicate IDs, orphan results, missing or out-of-order results, and incomplete chains
fail closed.

The exact `<tool_call>\n<function=` trigger starts a constrained call. While
reasoning is active, the reasoning boundary requires its actual token ID. After
that boundary, tool wrappers are recognized through either added tokens or
ordinary text tokens, matching XGrammar's text language. Ordinary-token thinking
marker spellings inside reasoning remain reasoning. Both tokenizations are
checked with native XGrammar and through streaming and batch parsing.

The parser deletes nothing the model generated. A special token its format does
not act on -- the vision and audio markers among them, which the served template
itself spells -- is the text it decodes to, in reasoning, in the answer and inside
a call's arguments, and a second `</think>` after reasoning has ended is content,
as a literal `<think>` there already was. A `<think>` the model writes inside its
reasoning is part of it: every served generation prompt ends inside the `<think>`
the template wrote, and the parser reads the prompt. Only a generation whose
prompt left the opener to the model -- a template that writes none, or derender,
which is given no prompt -- opens its reasoning with `<think>`, and only as its
first token. Upstream's Qwen grammar deletes every `<think>` in reasoning, the
model's own included. Upstream's parser engine deletes every
special token its format does not act on by default. Here that deletion erased the
model's output from the record and, beside the content ids the batch tool pass
splits on, left an id whose text was gone, which the token-position scanner
refuses: a whole response failed with 500 and a stream lost the token or failed,
by how its deltas were grouped. The stop token a generation ended on carries no
text unless the caller asked to see stop text, on every route: the detokenizer and
the derender route decide it in one place, and the parser still receives the
token. A parser format that forwards content ids is refused when it is
registered, at startup, if it acts on a terminal in content outside its tool
language in any configuration it builds.

`include_reasoning` decides only what a response shows: whether reasoning has
ended, and so when a grammar starts constraining, is read from the prompt the
model continues, on Chat Completions as on Responses. The complete-output parse
reads that one decision as the stream does, so a continued final message -- whose
prompt already closed reasoning -- is the answer on both transports, never
reasoning in one and content in the other. Derender is given no prompt and parses
as if reasoning were open, as upstream's derender does: a continued final
message's answer is reasoning there, and empty with `include_reasoning` off. The
"parser parity" that upstream's derender guide
(`vllm/docs/serving/online_serving/derenderer.md`) states -- the same content,
reasoning and tool-call split as a served Chat request -- does not hold here for
a continued final message.

A tool choice that is not specified -- omitted or `null` -- is `auto` when tools
are declared and `none` otherwise. The chat request and the Anthropic conversion
decide it once, so one value decides both whether the call grammar is armed and
whether calls are parsed; `none` is the only choice under which a call-shaped
span is text. A forced (named) or `required` choice may begin with the blank line
the template writes after `</think>`, so the call-only answer the model is
trained on -- `</think>`, a blank line, `<tool_call>` -- is generable; nothing
else may precede the call. XGrammar writes that line only inside a grammar that
also covers the reasoning, which vLLM never builds, since the reasoning parser
starts the grammar once reasoning has ended.

On Responses, a namespace's functions are offered to the model under flat names
(`namespace__name`), and the prompt, the call grammar and the parser read that
one list, so a call to one of them is admitted and parsed by that name. A named
choice must use a name the model is offered; `allowed_tools` arms the grammar to
exactly the listed functions in its mode; `required` needs a function to call; and
a choice no grammar can enforce -- a hosted, MCP or custom tool -- is refused
naming `tool_choice` rather than left to publish the call as text. Every declared
tool is in that list or refused: the template is given functions and nothing else,
so a hosted, MCP or custom tool, a namespace member that is not a function, or a
namespace with no tools is a 400 naming `tools[i]`, under every choice and with or
without a tool server, rather than a tool dropped from the prompt.

A prompt the template renders is never cut. Responses `truncation: "auto"` and chat
`truncate_prompt_tokens` or `truncation_side` would cut tokens from the head of the
rendered prompt -- the template's markers, the system text, the task and the tool
definitions -- so each is refused with a 400 naming the parameter; shorten the
input, or let the client compact it. Completions keep `truncate_prompt_tokens`,
whose prompt is the caller's own text.

An output constraint cannot hold beside a callable tool, because the call grammar
covers the whole output. A request that could call a tool and also constrains its
output -- `response_format` or `structured_outputs` on chat, `text.format` or
`structured_outputs` on Responses, `output_config.format` on Anthropic -- is
refused with a 400 naming that parameter: `tool_choice` `none` keeps the
constraint, and a forced call of a function whose parameters are the schema
returns structured data through the grammar.

Qwen's natural `</think>` token or implicit `<tool_call>` token ends thinking
once per generation. The deployed V1 thinking-budget tracker and grammar
use that parser-derived boundary. Later marker text in a response or a tool
value cannot restart the budget or inject a forced closer into the call.
The implicit trigger itself belongs to grammar content. The existing final
response counter still starts only at the explicit reasoning closer.

Only a call with its observed `</tool_call>` wrapper and a model EOS terminal
becomes an executable call. Both configured Qwen EOS tokens have that meaning.
An EOS-cut or malformed span stays verbatim text. A caller stop returns the
raw span with its stop cause; a length terminal retains a diagnostic prefix.
Streaming holds call entries and following content until the terminal makes
that decision, while preceding content and reasoning continue to stream.

Caller stop strings and stop-token IDs apply to text outside armed calls.
The native structural grammar suspends them inside a call, including literal
wrapper text in parameter values. A stop overlapping the closing wrapper
cannot remove part of the call. Stop tokens remain available as literal
argument content; only model EOS tokens serve as grammar terminators.
Unknown emitted names and their whitespace remain exact for client error
feedback. The parser does not silently delete or rename them.

Request errors are classified at their cause. Deliberate template guards,
image-data refusals and context-length refusals are typed client errors.
Unexpected Python exceptions, template bugs and server configuration failures
remain server errors; a raw ValueError does not imply HTTP 400. Each template
guard names the template variable it refuses, and the refusal names the request
parameter that supplied it -- `reasoning_effort`, `reasoning.effort`,
`chat_template_kwargs.preserve_thinking`, `messages` or `input` -- and no
parameter when the value came from the server's own defaults.

Reasoning and tool boundaries retain their actual token positions through
Unicode decoding and caller-stop holdback. A stripped marker cannot bind to
an earlier prose lookalike. Visible partial markers remain text; finishing
does not recreate hidden text. Batch and streaming use the same rule.

Qwen XML argument decoding uses the complete tool schema, including local
references, enum/const constraints and composition. The complete decoded object
must validate. A schema-valid raw string keeps all its bytes. Grammar padding
is removed only when the schema requires that interpretation; ambiguous values
prefer their original string representation. Encoded JSON objects, arrays and
numbers keep their original representation. Cut or untypable values stay raw
for diagnostics and client validation. All parameter constraints are available
before argument JSON is emitted; executable calls still wait for EOS.
If one call repeats a parameter name, decoding refuses the call before either
value can replace the other. This covers complete output and an unfinished
trailing parameter on either transport, including schemas that allow additional
properties. The response is the unprocessable-entity refusal, status 422, that
a well-formed request gets when what it leads to cannot be served, typed as
`RepeatedToolParameterError` -- inside a stream already under way, its error
event with that type and code -- naming the call's tool and the repeated
parameter and no request parameter, and no call is published. The model wrote
the repeat; the request was sound and the server did not fail, so the refusal is
no 5xx, and its next action is the caller's: generate the response again.

A raw parameter value -- a string, or a value whose schema leaves its type
open -- may carry neither its own `</parameter>` closer nor the next
parameter's `<parameter=` opener. vLLM owns the Qwen structural tag so it can
exclude both: excluding only the closer let a value absorb the following opener,
which produced not a parse error but a different, well-formed, schema-satisfying
call. Every other production reproduces XGrammar's Qwen language exactly, with
one deliberate exception: a string parameter declaring `pattern`, `format`,
`minLength` or `maxLength` is refused, naming the tool and the property.
XGrammar emits such a string as a regex, and that regex cannot carry the
exclusion — xgrammar's regex engine has no lookahead, so forbidding a fixed
substring is only an unrolled DFA, which cannot then carry a length bound. There
is no grammar that both bounds the value and keeps the opener ungenerable, so
there is one string channel rather than a second one that drops the exclusion;
declare the parameter without the constraint and validate it in the tool. A
string argument therefore cannot carry the literal `<parameter=`, the same
limitation the format already had for the exact closer; tool authors needing
either sequence in a string must use another representation.

A JSON value -- an array, object, number, boolean or null parameter -- may
carry both markers, and `</function>`, inside its strings: the grammar admits
any JSON string there. The parser reads argument text by the grammar's own
productions rather than by a pattern of its own. Which production each
parameter's value is written in is decided once, the grammar is rendered from
that decision and the parser reads by it, so a JSON value ends at the first
closer outside its strings, and `</function>`, a newline and `</tool_call>` end
the call unless its arguments can only be read as still inside a parameter --
then they are that parameter's text. A declared parameter is
read by its declared production while its slot is still open, and by the
undeclared-name production only after that. A call that names a parameter twice
in that reading is refused as above.

A parameter value is framed by the transport, not by the value. The served
template renders `<parameter=NAME>`, a newline, the value, a newline, and
`</parameter>` — the framing the model was trained to emit and the framing the
grammar pads by construction — and the parser removes exactly one newline from
each end and nothing else. That inverse is exact, so every value round-trips
unchanged, including one that begins or ends with a newline of its own, which
travels as two. A value the model hugs against its tags therefore decodes to
the same string as one it frames, which makes the model's slot-bound framing
habit unable to decide what reaches a tool. Parsing, history rendering and
grammar acceptance are checked together. Streaming and batch parsing also return
all content before, between and after calls byte for byte, whitespace-only content
included, so a caller records exactly what the model produced. Upstream vLLM's
shared parser deletes whitespace-only content before a call and strips content
beside calls; this deployment does not. That changes the response, not the next
prompt: the model's own template trims every message's content before rendering
it (`render_content(...)|trim`), and so does the served template derived from it,
so a turn whose content is whitespace or carries surrounding whitespace renders
byte for byte as the trimmed or empty turn upstream would have returned.
Historical reasoning remains intact.

A narrow scheduler patch retains the implicit Qwen tool-start token at the exact
reasoning-to-tool grammar boundary. Without it, post-generation parsing could
recognize a call that decoder-time structural grammar had not constrained. Real
token sensitivity tests show that the fixed path rejects an unknown tool, wrong type,
and extra property at their first invalid token.

Streaming and non-streaming paths are separately tested. An incomplete generation is
never promoted into a successful executable call:

- Chat preserves finish_reason=length;
- Anthropic preserves stop_reason=max_tokens;
- Responses ends a cut generation with response.incomplete, never
  response.completed; the item the limit cut -- the last -- is incomplete, a cut
  call emits no arguments-done, and items finished before the cut are completed,
  on both transports.

Anthropic streaming and batch responses derive their terminal metadata from the
same observed completion cause and output. A matched stop string is reported as
`stop_reason: stop_sequence` with the exact string in `stop_sequence`. Completed
tool-use output reports `tool_use`, including a named call that Chat reports as
`stop`. Token limits retain `max_tokens`. A stream that lacks its completion
metadata, usage or terminal marker ends with an API error and no `message_stop`.

Thirty controlled real-token prefix cuts produced the same typed semantics in
streaming and non-streaming parser paths. Live independently sampled fault injection
also passed; no claim of byte-identical sampled prose was needed.

Complete tool loops passed three of three OpenAI trials and three of three Anthropic
trials. OpenAI Chat, Anthropic Messages, and OpenAI Responses passed streaming and
non-streaming tool calls plus typed tool-result continuation. Equivalent protocol
histories passed render -> tokenize -> parse -> rerender with token-identical
semantics. Anthropic validation problems return HTTP 400 invalid_request_error, not a
misleading server 500, whether they are typed request validation or the engine's own
request validation.

Generation names its agent or does not happen. The protocol probe sends every
one of the six mounted generation routes — Chat Completions and its batch form, Completions,
Responses, Anthropic Messages, and token-in-token-out generate — a non-streaming
request that is complete except for `kv_scope`, and each answers HTTP 400 naming the field: as
`error.param` on the OpenAI-shaped surfaces, and as an `invalid_request_error`
whose message names it on the Anthropic surface, because the Anthropic router
answers the request model's refusal in its own envelope, exactly as it answers
typed request validation. The probe also sends the five streaming routes the same request with
`stream: true`, which must be refused with HTTP 400 before any status line, and
holds one conversation's stream open while a non-streaming and a streaming
request under its ID must each be refused with HTTP 400 naming that ID, after
which the open stream must still finish. The agent prefix-cache probe forks its
conversation: after the parent's cached continuation, the parent's history plus
a new question is sent under a new ID and the parent's salt, and the fork's
served `cached_tokens`, metric delta and shared prefix are recorded; the fork
must reuse at least what the parent's continuation reused. These streaming,
overlap and fork checks have not yet run against a live server; they were
exercised only against a loopback stub. Every probe mints one ID per
conversation with `scripts/probe_scope.py`, so the suite runs under the rule it
proves.

The installed-image focused suites passed:

- 125 TurboQuant cases, two platform-inapplicable skips;
- 286 parser/structural-output/Anthropic conversion cases with ten intentional
  generic-policy deselections replaced by project-specific assertions;
- 382 Qwen streaming/replay cases;
- vision workspace, image-contract, and vision-MLP units during the immutable build.

### Exact served usage

Every Chat Completions response — non-streaming, and streaming with
`stream_options.include_usage` — carries
`usage.completion_tokens_details.reasoning_tokens`: the number of generated token
ids before the id at which the reasoning parser left its reasoning state
(`</think>`, or the implicit `<tool_call>` that ends reasoning on this grammar),
or every generated id when the generation never left it. It is read from the
parser engine's own token-id split, the same walk that decides where reasoning
ends in the streamed deltas; it is never a re-tokenisation of the reasoning text
and never a character estimate. `completion_tokens` counts every generated id,
so `completion_tokens - reasoning_tokens` is the exact content-plus-marker count.
The count is defined only for a grammar whose reasoning state is entered once and
left on a token-id terminal; for any other grammar the field is absent rather than
approximate. Responses serves `usage.output_tokens_details.reasoning_tokens` from
the same count, read through the same function and summed over a turn's
generations; it is absent, never zero and never a failed response, when a
generation had no exact split. `--enable-prompt-tokens-details`
serves `prompt_tokens_details.cached_tokens`, the prompt tokens the scheduler
actually reused from the prefix cache, so a client never has to invent a zero for
it. The exact-reasoning-usage build unit exercises the composed qwen3 parsers on
CPU inside the immutable build and holds every reasoning parser to one count: the
whole-generation count a parser exposes is the count its feed took for every
format built on the engine, and none for a parser that splits on text, so it is
absent exactly where usage is absent and never an error. The protocol probe
holds both fields to the returned token ids on every trial, streamed and
unstreamed.

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
4. MTP/speculation absent, CPU weight offload and KV offload both zero, CUDA
   graphs retained: pass.
5. Exact xhigh defaults, high/max alias identity, low/disabled rejection: pass.
6. Exact five-real-token final-response phase stop: pass.
7. OpenAI/Anthropic tool loops, adversarial grammar, orphan rejection, and
   streaming/non-streaming round trip: pass.
8. Responses streaming/non-streaming tool loop and incomplete-output gates: pass.
9. Fifteen full-pixel distinct-image transcription and sixteenth-image rejection:
    pass.
10. Portrait/landscape 30:1 far-end perception and 31:1 rejection: pass.
11. Chronological OpenAI/Anthropic tool-result image parity: pass.
12. Cold/warm image and prompt-cache counters plus time separation: pass.
13. Exact 262,143-prompt-plus-one-output native boundary with fifteen images: pass;
    262,145 total rejected.
14. Logs after maximum image/context work: no CUDA OOM, allocation retry, preemption,
    semantic fallback, or failed restoration.
15. Complete official-BF16 versus converted tensor audit: pass; 1,199 official and
    1,968 converted tensors fully accounted, 233 FP8 and 168 NVFP4 tensors fully
    dequantized, and all 798 reference-precision tensors compared.
16. Actual K8V4 Triton store/fused decode versus independently packed FP32 reference:
    pass; maximum absolute difference 0.00381172 and minimum cosine 0.999998331.
17. Actual worst-layer NVFP4 production kernel versus independent dequantization and
    BF16 matmul at M=1/17/129: pass; required FlashInfer-CUTLASS selector.
18. Exact text/image MRoPE implementation versus independent Transformers 5.15
    semantics and continuation positions: pass.
19. Pinned Qwen Code archive plus exact semantic reconstruction: pass; exactly 61
    changed/new files and 2,427 assertions across 23 focused test files; the full
    no-cache build reproduced the agent image its own release lock pins.
20. All pinned Rust component tests and clean release images: pass; 44 service,
    9 broker, 3 relay, 2 capture, and 2 agent-exec tests. The same no-cache release
    exactly reproduced the locked relay, capture, broker, and service image IDs.
21. Real Qwen Code hostile-workspace text, full-resolution PNG vision, PTY shell,
    write/read, and final response: pass in seven turns. Ordinary QWEN.md and
    AGENTS.md guidance remained active while `.env`, MCP, hooks, skills, rules,
    memory, output language, workspace settings, and slash commands remained inert;
    the model read exact code `VISION_AGENT_PTY_4827` and the shell emitted exact
    marker `QWEN38_AGENT_ISOLATION_OK`.
22. Locked-mode authentication revalidation: pass. An earlier candidate exposed an
    upstream late `.env` load after tools; the accepted narrow source repair keeps
    environment/workspace loading disabled on every later auth validation. The full
    source matrix and a fresh hostile live session proved the marker absent.
23. Real Qwen Code prefix and image caching: pass. The recorded v8 cache acceptance
    added 296,939 prompt tokens; vLLM recorded 241,280 local prefix hits, 55,659
    locally computed tokens, and 3 multimodal-cache hits across 20 requests.
24. PTY and foreground subagent: pass. A separate real PTY session produced exact
    shell/file output. One Explore subagent returned through correlated parent/child
    tool events, used native list/read plus a shell byte check, and was independently
    verified by the main thread. Explore retained writable conversion/scratch tools;
    its trusted effect journal proved zero workspace/artifact effects for that task.
25. Live stack isolation and lifecycle: pass. vLLM and the service are network-none;
    only two minimal fixed ingress relays use host networking. The service has no raw
    Docker socket; a network-none typed broker is its sole holder. Backend and agent
    roots are read-only, agents have no route/DNS/GPU/published port, model access is
    through the exact socket relay, session output is unmounted from the agent, and
    all components are capability-free with no-new-privileges.
26. Readiness-boundary cancellation: pass. The request acknowledged in 785 ms and
    terminated in 1,863 ms with zero turns, exit 143, an empty event stream, a
    complete nine-file bundle, no fabricated success, and no session-container
    leftovers.
27. Late capture subscriber correctness: pass. A production benchmark run exposed
    that Docker `logs --follow --since 0s` starts at relative "now" and can miss an
    already-emitted `CAPTURE_COMPLETE`. The broker now uses exact Unix epoch `0`;
    its new unit test freezes that replay contract. The service continued to fail
    closed during the faulty run and retained its evidence instead of parsing
    unproved output.
28. Clean repaired production release: pass. The pinned build ran 44 service, 9
    broker, 3 relay, 2 capture, and 2 agent-exec Rust tests, all 2,427 Qwen Code
    assertions, and reproduced all five final image IDs. A fresh five-turn
    production tool round trip then completed normally with a clean eleven-file
    bundle and no session containers.
29. Production-service SWE-rebench pilot: pass. The pilot task ran through real
    production session creation and `/wait`; neither Harbor nor the evaluator
    called vLLM. The production session resolved all 11 evaluator checks in 61
    turns. The two earlier orchestration failures are retained as infrastructure
    evidence and are not model scores.

The supported status currently reports:

    HEALTHY
    endpoint 127.0.0.1:8000
    262144 total-token context
    15 inline static PNG images, 16777216 pixels each
    BF16 vision tower, aspect ratio <= 30:1
    xhigh thinking
    explicit Qwen3.8 sampling

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
b965d5f8c24f48e65fb0b17c7d45f34ca4ce8f38. The agent-service repository pins the
official release archive by SHA-256, pins the landmark-aware source
transformation independently, and pins the immutable agent image the client runs
inside; its release lock is the one place those identities are recorded.
The accepted service sends this exact outgoing policy on every main and foreground
subagent turn:

- contextWindowSize 262144;
- exact server/model identity;
- xhigh mandatory thinking;
- Alibaba thinking sampling tuple;
- no phase budget, from the server or the client; the client issues every turn
  with its turn room `C`, derived from the served window, as its limit and admits
  no configured ceiling of its own;
- exact /tokenize count on the same rendered request before generation, and
  every tool result's text bounded once, where the model's copy of it is made,
  in the UTF-8 bytes of its NFC form;
- no character/image-token heuristic or tokenizer fallback;
- splitToolMedia false so tool images stay in their originating tool result;
- typed content parts and PNG-only image tools matching the strict backend;
- no client retry/downgrade, XML recovery, implicit continuation, or partial-call
  execution;
- one long main thread compacted when the exactly-counted request reaches its
  share of the window, and only sequential foreground subagents.

The service is the updated original `/home/user/Desktop/agent_service`, at the
release its own `config/release.lock.json` records, not a copied launcher.
Identity this repository does not derive is named by its owning repository
rather than restated here, because an unowned copy is what drifts. Its
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
The harness submitted the pinned SWE-rebench task to real production
`POST /v1/agent/sessions`, waited on the production notification endpoint, required
the service's durable terminal bundle, and invoked the immutable evaluator only on
that captured post-session workspace. The production session resolved all 11
evaluator checks in 61 turns and 1,090,658 ms. One completed task is a lifecycle
proof for the deployed pair, not a benchmark-suite score. Exact production release
IDs, input hashes, failure classification, and bundle/patch hashes are recorded in
`/home/user/Desktop/agent_service/docs/production-swe-rebench-pilot.md`. That
harness cannot be replayed: it is pinned to the agent-service release it ran
against, and the service has since removed the folder-path submission and the
notification endpoint it depended on.

Codex Responses and Anthropic Messages protocol surfaces are proven on the server,
but neither creates another supported client mode. Host Claude Code remains entirely
untouched. There is exactly one accepted local-agent behavior: the pinned Qwen Code
container through the isolated service into this pinned vLLM backend.

### Software/hardware record and the host contract

Host requirements are functional, not identity pins: the invoked tools must
exist, Docker must respond with its NVIDIA runtime, the daemon must report the
container isolation [`scripts/host-isolation.sh`](scripts/host-isolation.sh)
requires, and exactly one GPU of compute capability 12.0 must be present.
Whether its memory holds the model and the declared KV pool is the engine's
measurement at startup, which refuses the pool naming each term; the launcher
keeps no second memory number beside it.
Exact host software versions, binary hashes, and GPU/driver identity are
deliberately not asserted; pinning them tied the deployment to one specific
computer without making inference any more correct.

Two different facts refuse a GPU, each by its own statement. A GPU without
native FP4 cannot compute the checkpoint's NVFP4 W4A4 layers: vLLM's kernel
selection used to fall through to Marlin (W4A16, 16-bit activations) or
emulation with a warning, serving different arithmetic, and refuses at
layer construction, before any weight loads, naming the device capability and
why each native kernel is unavailable (stage `nvfp4-native-kernel-required`,
proven on CPU by `scripts/native_fp4_selection_unit.py`). Only naming the
substitute with `--linear-backend`, which this locked command does not, serves
it -- under `VLLM_BATCH_INVARIANT` too, whose emulation kernel is refused like any
other substitute unless named, so the refusal does not depend on what the launch
leaves unset. Separately, `VALIDATED_CUDA_CAPABILITY` declares the one capability the
image is built for (the base image compiles with `TORCH_CUDA_ARCH_LIST=12.0`)
and every GPU gate ran on; the launcher refuses any other before starting
anything, as outside the validated lock, and says how to validate one. It names
an instruction set, not a card: every 12.0 GPU passes, and a B200 is refused
because this image was never built or validated for 10.0, not because it lacks
FP4. The runtime report the running container is checked against is software
versions only.

The isolation rule is one file that agent_service carries byte-identically. It
parses `docker info` SecurityOptions into names and attributes and requires
AppArmor with the default profile, seccomp with the builtin profile, and a
private cgroup namespace. A daemon-wide mode that can only add a restriction
the containers already impose (`no-new-privileges`) is accepted; one that
changes the model the stack runs under (`userns`, `rootless`, `selinux`), and
anything the rule cannot interpret, is refused with the failed property, the
requirement, the report, and the next action. It asserts properties rather than a string, so Docker
29.7.2, which reports `name=apparmor`, and Docker 29.8.1, which reports
`name=apparmor,profile=default`, both pass, and no Docker or containerd version
is pinned.

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
