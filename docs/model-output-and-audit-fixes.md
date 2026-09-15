# Model-output and deployment audit resolutions

This record accompanies the unchanged `qwen38-deployment-correctness-audit.md`.
The independent triage and handling policy supplied with the implementation brief
decide the resolutions. Source validation here does not certify a built image or a
live release. The v23 image and archive are awaiting adoption.

## Finding 10: ship and execute the TurboQuant guards

The runtime Dockerfile now copies both reviewed non-SoA kernels and verifies their
upstream and patched hashes. The context explicitly admits every copied input.
The build and runtime installed-file checks cover all reviewed runtime sources;
the runtime check previously also omitted the twenty KV sizing/scope files that
the image already copied and the build verified.

The new `scripts/turboquant_guard_unit.py` executes the installed store kernel with
Triton's CPU interpreter. It tests unchanged finite key/value/metadata bytes at
D=256 and four KV heads, isolated and all-NaN vectors, both infinities, and fp16
metadata overflow. It separately tests the installed decode helper's refusal of
SM 8.0 and its E4M3 result on SM 8.9, 9.0 and 12.0 with mocked capabilities. No
CUDA device or model is used. The Dockerfile runs this test in a fresh process
with interpretation enabled; this environment applies only to that build test.

Executing the test exposed an additional defect in the original guard: Triton's
minimum/maximum reductions use non-propagating NaN semantics, so an isolated NaN
can leave both extrema finite. The reviewed guard transformation now checks for
NaN in every active lane before accepting the metadata. Finite lanes still use
the same arithmetic and produce the same bytes.

`scripts/runtime_image_unit.py` checks that every reviewed runtime mutation has
its copy, context entry, build verification and runtime verification, and that
the guard test actually runs in the Dockerfile. This prevents a future label from
claiming a source mutation the image does not contain. All runtime shell-contract
tests invoked by `build-vllm.sh check` now execute in a disposable container too.
Its disposable worktree/export paths honor the standard `TMPDIR` variable.

Validation: `TMPDIR=/tmp/codex-fix ./scripts/build-vllm.sh check` passed with
16 framework/recipe tests, the exact template derivation and thirteen-stage source
reconstruction, and three kernel-guard tests. The untouched base image is a
negative control: the same kernel-guard suite must fail against its unpatched
store and decode modules. Triton interpreter warnings on intentionally invalid
floating-point input are retained as evidence.

The full `turboquant_k8v4_unit.py` GPU store/fused-decode numerical acceptance
remains required at release. CPU interpretation proves these guards and the
packaging regression; it does not prove GPU code generation or performance.

## Findings 5, 3, 6 and new A: Anthropic input and error fidelity

`vllm-anthropic-input-fidelity.patch` adds the fourteenth runtime source stage.
The tool-result converter refuses documents, search results and unknown items
with `VLLMValidationError` before rendering. Missing image sources and malformed
text/reference items are also refused. Supported text and media retain their
order inside the originating tool response. When the caller sets `is_error`, the
result begins with `Tool result flagged is_error: true.`; the caller's content
follows it unchanged. This represents a caller-supplied status the template has
no separate field for.

Both `/v1/messages` and `/v1/messages/count_tokens` use the same exception
classification as OpenAI. Template, image-gate and prompt-length rejections are
HTTP 400 `invalid_request_error`. Framework validation, HTTP errors and engine
errors use Anthropic envelopes on these paths, including deployments with an
ASGI root path. Other protocols retain their own envelope. OpenAI error chunks
forwarded through the Anthropic stream keep their classified status semantics;
the converter emits an `error` and stops without a success terminal.

The existing shared classifier still handles raw `ValueError`, `TypeError` and
`OverflowError` as client rejections. This is necessary while the pinned renderer
and input utilities themselves raise those types; duplicating a narrower local
classifier caused the original 500 defect. Unclassified server failures remain
500 `api_error`. HTTP statuses without a dedicated Anthropic error name retain
their HTTP status and use the corresponding client/server error family.

Validation: 122 tests in the extended Anthropic conversion and validation-handler
suites pass in a disposable, network-disabled base-image container using the
preexisting offline pytest packages. The tests exercise both stream settings and
both routes, real template/image/length rejection code, nested-content refusal,
error markers with text/media, returned engine errors, framework validation with
root paths, OpenAI envelope preservation, and terminal stream errors. No listener,
engine or model is started. The five newly patched shared error modules are
copied and checked against their upstream and final hashes in the image recipe,
build verifier and installed runtime verifier.

## Finding 13, S11, D2 and batch parity: the Qwen parser's tool language

`vllm-qwen-exact-tool-language.patch` adds the fifteenth runtime source stage.
Only a real `<tool_call>` token followed by exactly `\n<function=` starts a
call. Bare function markup, stray openers, empty wrappers and unfinished
function headers return as content. With no declared tools or `tool_choice:
none`, the model's call-shaped text remains visible and produces no call.

A parameter value ends only at the exact `</parameter>` delimiter.
`</function>`, `</tool_call>`, `</think>`, `<parameter=` and variations such as
`</parameter >` remain part of that value. The format cannot carry the exact
parameter closer inside a string; tool authors needing that sequence must use
another representation, such as encoded input decoded by their tool.

The batch tool pass receives the generated IDs after the parser's first exact
reasoning boundary. Streaming retains those IDs when detokenization delays
the corresponding text. Later markers inside a value cannot move the boundary.
Content before and after calls keeps its order, and the parser exposes whether
each call reached its actual closing wrapper. The remaining serving-layer EOS
and stop-control work must use this observation before promotion is complete.

Validation: 3,833 parser-engine tests pass, including the preserved regression
tests, 24 adversarial value/chunk combinations and two delayed-boundary cases.
The build now runs five CPU tests against the installed parser, covering exact
triggers, disabled tools, reserved value markup, token/text lookalikes, batch
parity and observed closure. `build-vllm.sh check` runs that unit against the
reviewed runtime overlay. The four newly patched engine modules are copied and
verified in the image, build and runtime checks. The older reasoning-usage build
unit now declares its test tool and uses marker IDs outside the ASCII range;
the old mock assigned several letters the same IDs as reserved markers.

This stage does not finish caller stop suppression, EOS promotion, schema type
conversion or context-dependent scanner alignment. They remain open.

## Finding 11: admit encoded PNG source pixels

`vllm-png-source-admission.patch` adds the sixteenth runtime source stage.
The decoder checks the source IHDR chunk before normalization or loading:
bit depth must be 8, and color type must be RGB (2) or RGBA (6). Pillow's
decoded mode remains a separate consistency check. A 16-bit source is refused
with its bit depth and color type instead of being silently reduced to 8 bits.

Validation: five image admission tests pass, including hand-encoded valid
16-bit RGB, RGBA and grayscale-with-alpha PNGs that Pillow exposes as RGB/RGBA.
The existing CPU vision-contract build unit now exercises those three cases
alongside accepted 8-bit RGB/RGBA, white alpha compositing, animation refusal,
image size/aspect checks, inline URL validation and tool-image ordering.
The unit passes without network, engine or GPU use. Its Anthropic fixture now
includes the `is_error` field required by the real input model. Backend `check`
now executes the parser, vision and reasoning CPU contract units against the
complete reviewed runtime overlay as well as its existing kernel guards.

## Finding 16: physical KV capacity includes initial GPU residents

`vllm-kv-physical-free-memory.patch` adds the seventeenth runtime source stage.
The physical capacity bound starts with the free-memory snapshot taken before
model loading. The profiler reports subsequent consumption and transient peak
headroom, so subtracting that amount from the whole card incorrectly made
earlier residents available a second time. The corrected bound charges initial
residents, model/profile consumption, recurring activation headroom, CUDA graphs
and frontend reservations once each. Utilization continues to provide an
informational estimate and the startup free-memory requirement; it cannot
silently resize or veto a declaration that fits the physical bound.

Validation: 18 worker tests pass in an offline CPU container. Sixteen combinations
call the real `Worker.determine_available_memory` with mocked snapshots and
profiling operations, varying initial occupancy, activation headroom, graph
memory and utilization. They verify the bound and the separate estimate; the
two startup-plan tests still pass. No GPU or model profiling was performed.

## S6: the decoder owns the number of calls

`vllm-qwen-single-call-grammar.patch` adds the eighteenth runtime source stage.
The request's `parallel_tool_calls` value reaches the Qwen structural-tag
builder. With `false`, XGrammar's existing Qwen format stops after its first
complete call. Automatic choice still allows an ordinary text answer;
required choice still requires a call; named choice already contains one tag.
The existing schema and reasoning prefix retain their behavior.

The Chat response layer returns every call actually produced in both streaming
and batch responses. Its old filtering module is deleted, including its image
copy and bytecode. Where a call-count grammar is inactive, output cannot be
silently changed to make it appear that the model emitted only one call.

Validation: 95 tests pass, including actual XGrammar acceptance/rejection for
each choice and reasoning setting, request-to-grammar propagation, and real
streaming/batch response generators supplied with deterministic output. The
installed-parser build unit also exercises the grammar call limit. No engine,
model, network listener or GPU was started.

## Findings 7 and 14: Responses history integrity

`vllm-responses-history-integrity.patch` adds the nineteenth runtime source stage.
Chat's transport-ID validation now lives in the shared chat utility and also runs
after Responses history construction, before rendering. The same ordered,
complete call/result sequence is required for every rendered protocol. Boundary
refusals use `VLLMValidationError`, including encrypted reasoning that cannot be
rendered, so they do not depend on broad Python exception classification.

Responses converts every supplied text and reasoning block in order, preserving
whitespace and concatenating text without invented separators. Mixed content
retains its complete part list for the shared content parser. Empty messages
remain empty. Previous-response reconstruction uses the same conversion as
explicit replay, including thinking and tool calls, without mutating supplied
history. Summary-only reasoning keeps all supplied summary blocks with the
existing explicit warning; a summary is the available representation when the
caller provides no full reasoning text.

Validation: 91 utility/boundary tests pass in the offline CPU container. They
cover typed, dictionary and easy-input replay, both stream settings, every
malformed call/result sequence on Chat and Responses, mixed content, empty
content, preserved thinking, nonmutating reconstruction and the real Responses
request method refusing invalid history before its renderer is called.

## Remaining implementation

Parser language, stop handling, schema conversion, protocol translation, image
admission, remaining triage findings, client recovery verification and final
backend/client pin adoption are still under review. Their completion is not
implied by the finding-10 validation above.
