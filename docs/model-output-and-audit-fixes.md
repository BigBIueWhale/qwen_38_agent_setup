# Model-output and deployment audit resolutions

This record accompanies the unchanged `qwen38-deployment-correctness-audit.md`.
The independent triage and handling policy supplied with the implementation brief
decide the resolutions. Source validation here does not certify a built image or a
live release. The v23 image and archive are awaiting adoption.

## Qwen terminal recognition follows the grammar phase

`vllm-phase-aware-parser-terminals.patch` represents token authority in each
format's terminal declarations. While Qwen is reasoning, an actual boundary
token ends that phase. Once reasoning has ended, the exact tool-call trigger
and wrapper closer are recognized through either added tokens or ordinary text
tokens, as XGrammar recognizes them. A literal thinking-marker spelling in
ordinary tokens does not end reasoning. Parameter values retain their existing
exact closer and preserve other markup.

Every parser configuration and consumer uses the same terminal representation.
The unused transition setting that allowed different fallback transitions in
text-only input, and its synthetic tests, are deleted. Vocabularies without an
atomic token for a format terminal retain text recognition: there is no atomic
token whose identity could distinguish that spelling. This is the existing
format behavior, and Qwen's deployed vocabulary has its boundary tokens.

Validation: 3,998 offline parser tests and eight installed CPU tests pass. The
new tests cover both wrapper tokenizations, explicit and implicit reasoning
boundaries, literal boundary lookalikes and multiple streaming chunk sizes.
Native XGrammar accepts valid calls and refuses unknown names through both
tokenizations. Against the previous runtime, 25 focused cases fail and 23
unaffected cases pass. The earlier installed test requiring ordinary-token
calls after reasoning to remain prose was incorrect and is replaced. Its old
passing result is not evidence of grammar equality. Four newly modified parser
modules join the complete image provenance checks.

The integrated generator and `build-vllm.sh check` pass with 27 reviewed stages,
103 deployment inputs, 19 framework/recipe tests and every installed CPU unit.
The standalone reasoning-usage unit also constructs the typed declarations.

This correction leaves terminal promotion, caller-stop control, schema
conversion, phase accounting and exact detokenizer provenance open.

## XML string and content fidelity

`vllm-xml-text-fidelity.patch` preserves every byte inside an XML string parameter,
including leading/trailing newlines and whitespace-only values. The converter's
newline-trimming helper is deleted. Named derivation stages remove inserted value
padding from the historical tool-call template and its instruction example. The
model's source template stays unchanged, and historical reasoning is retained.

The shared parser also removes its batch-only content-stripping setting and all
consumers of that setting. Nonempty content around calls now retains the same
bytes on both transports; the existing whitespace-only gap normalization remains.
Four newly modified parser configurations join the full image provenance cascade.

Validation: 3,954 offline parser tests pass. The previous runtime fails 101 focused
controls, with 20 unaffected controls passing. The source tests cover multiple
chunk sizes, empty and whitespace-only strings, Unicode, embedded reserved
markers, surrounding content and incomplete string diagnostics. Five obsolete
expectations now require preserved whitespace. Seven installed parser tests pass,
including actual derived-template rendering, XGrammar acceptance and stream/batch
round trips. Template retention now also runs in the ordinary CPU source check.
Schema coercion, incomplete non-string diagnostics, terminal promotion and exact
token/text provenance remain separate open obligations.

The integrated generator and `build-vllm.sh check` pass with 26 reviewed stages,
102 deployment inputs, 19 framework/recipe tests and all installed CPU units.

## Finding 20: raw images through the token generation boundary

`vllm-raw-image-token-transport.patch` gives render and generate one media
contract: original inline PNG parts, in image order, alongside the fully rendered
token IDs. Generate decodes and processes the images, computes native hashes, and
validates their complete Qwen image spans without tokenizing or expanding the
prompt again. Processor-only caching supplies complete native image data even on
a cache hit. Cache salt and required generation identity survive the handoff.
Caller tensors, hashes, placeholder positions, UUIDs, unsupported media and extra
request fields are refused. The tensor serializer, its tests, both extractor
methods and the old feature protocol are deleted.

Validation: 216 offline CPU cases pass, including real render/JSON/generate methods
on both transports, native PNG decoding and hashes, complete span rejection, and
the actual image processor with unequal image sizes, both cached and uncached.
Thirty controls against the previous runtime fail because it admits the removed
media representations. The updated live media tests consume this same contract;
their source is reconstructed but those live tests were not run. No model,
weights, GPU or service was used. The installed raw-media unit checks the request
boundary and exact rendered-token geometry. Four newly modified runtime modules
join all image COPY, upstream/final hash, context and runtime checks. The image
recipe removes the obsolete serializer and compiled bytecode and verifies its
absence.

The integrated generator and `build-vllm.sh check` pass with 25 reviewed stages,
101 deployment inputs, 19 framework/recipe tests and all installed CPU units.

## Token generation: complete results and preserved terminal causes

`vllm-token-generation-result-integrity.patch` makes the transport select the
engine's DELTA or FINAL_ONLY output and removes `output_kind` from supplied
sampling settings. Shared validation requires each requested choice's terminal
and the request's completion event. It detects missing, repeated, out-of-range or
contradictory choices and engine errors. Batch errors return HTTP 500; streaming
errors emit an error event without a final usage report or `[DONE]`. There is no
invented default `stop` cause.

Empty-token terminal choices are preserved, sparse parallel output is indexed by
the requested `n`, and continuous usage totals every choice. The token protocol
carries `stop_reason` through all four Chat/Completion derender paths. A complete
empty answer remains empty; null or absent token lists are invalid input.

Validation: 160 offline CPU cases pass, including real `ParentRequest` aggregation,
serving methods on both transports, all terminal causes, empty output, broken
engine sequences, render serialization and derender JSON round trips. Sixteen
controls against the previous runtime fail on the corrected behavior. A live
derender test's obsolete empty-output refusal is replaced by its correct success
expectation; that live test is included in source reconstruction but is not run.
The installed `generate_result_unit.py` verifies transport ownership, parallel
empty terminals, stop causes and refusal of incomplete output. The newly modified
derenderer joins every upstream/final image hash, COPY, context and runtime check.
The integrated generator and `build-vllm.sh check` pass with 24 reviewed stages,
99 deployment inputs, 19 framework/recipe tests and every installed CPU unit.

## Finding 22: refuse unsupported decoding at the request boundary

`vllm-sampling-decoding-boundary.patch` declares beam search unsupported through
one shared request field type. Chat, Completions, batch Chat and render requests
refuse true before any rendering or engine call. The native HTTP 400 identifies
`use_beam_search`; it cannot misreport missing cache identity. JSON schemas expose
false as the sole supported value. The former Chat/Completion beam conversion
methods, dispatch branches and scattered capability checks are deleted.

Validation: 193 offline CPU tests pass, including the same boundary error on both
stream settings, native error-envelope conversion, request schemas, default and
explicit sampling selection, the request-input bounds and sampling/render suites.
Eight negative controls against the previous runtime admit the unsupported flag
and fail. Five live logprob cases were deliberately deselected; no service, model,
GPU or listener was started. Existing live beam-success tests are replaced by
boundary coverage; the LoRA test verifies refusal before adapter lookup and batch
Completion coverage now uses supported sampling. Source-only edits to live tests
are parsed and included in exact reconstruction but have not been run live.

The two newly modified serving modules join every image copy, upstream/final hash,
context, build and installed-runtime check. The installed phase-budget unit checks
this refusal on both transports. Integrated generator and `build-vllm.sh check`
pass with 23 stages, 97 deployment inputs and all installed CPU units.

## Finding 19: resolve sampling policy before engine validation

`vllm-generation-sampling-resolution.patch` supplies one typed representation of
the settings sent to `/generate`. Omitted fields remain omitted until the server
knows the model defaults and exact prompt length. Resolution creates a fresh
`SamplingParams` and passes it explicitly to both output converters. It preserves
stop-ID merging, required opaque agent identity and transfer metadata without
mutating the input. The former provided-key tracking and post-construction
defaulting are deleted.

Render output serializes every resolved public setting, including default-valued
ones. Thus an explicit maximum of 16 tokens or neutral sampling value survives
JSON transport. Internal engine bookkeeping fields are not request settings.
Completions uses the remaining window when the total limit is omitted and inherits
configured presence, thinking and final budgets. Chat also leaves an omitted
presence penalty available for model defaulting; Responses applies configured
`min_p`. All five generation surfaces inherit the same configured sampling policy.
A shared resolver enforces the server's final-answer ceiling even when a caller
sends null or the unset sentinel.

Validation: 148 offline CPU tests pass, including actual streaming and batch
`ServingTokens` dispatch, Chat/Completion render JSON round trips, all-surface
defaults, explicit neutral values, nested-object independence, input schemas and
budget validation. Three controls against the previous runtime reproduce early
`min_tokens=64` rejection against 16, lost explicit `max_tokens=16` serialization,
and Completions' implicit 16-token limit. The installed phase-budget unit now
checks these resolution and serialization invariants and runs during `check` as
well as image construction. The integrated generator and `build-vllm.sh check`
pass with 22 reviewed stages, 96 deployment inputs and all installed CPU units.

Correct initial and implicit reasoning-phase boundaries remain a separate open
obligation with finding 17; supplying the correct budget alone does not fix its
token accounting.

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
The original five installed CPU tests covered exact triggers, disabled tools,
reserved value markup, batch parity and observed closure. The ordinary-token
tool-call expectation in that set was incorrect; the phase-aware terminal
correction above replaces it and expands the installed checks to eight tests.
`build-vllm.sh check` runs that unit against the
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

## Finding 8: one identity through Responses streaming and completion

`vllm-responses-stream-identity.patch` adds the twentieth runtime source stage.
The stream's completed output items are the terminal response's output. Batch
parsing happens once, and both transports call one finalizer for status, usage,
raw-message metadata, output logging and storage. Streaming no longer re-enters
batch generation with an empty iterator or reinitializes tool sessions at the
end. A completed stream cannot acquire fresh item or function-call IDs by being
parsed a second time.

Text log probabilities retain their token bytes and alternatives in the completed
item and terminal response. A truncated final text item is incomplete on both
transports; the existing incomplete-call rule still suppresses arguments.done
on the interrupted call. This change does not complete the remaining EOS/stop
promotion work.

Validation: 39 tests pass in an offline CPU container. Eight terminal-identity
cases cover text, one/two calls, mixed reasoning/text/calls and length/normal ends;
they verify added/done/terminal IDs, exact item payloads, usage, event sequence,
absence of reparsing, and successful history replay using the original stream
call IDs. Four additional cases check log probabilities and text status on both
transports. The existing reasoning-usage fixture now has a coherent decoder and
the full Qwen marker vocabulary; its expected count includes every generated ID
before the reasoning boundary. One preexisting strict xfail remains for upstream
Harmony zero-delta item lifecycle; the deployed Qwen path does not use Harmony.

## Finding 9 and shared prefix caching

`vllm-shared-prefix-cache-and-user-capacity.patch` defines the twelfth source
stage. Every generation uses an opaque required agent ID. An ID with no cached
blocks may acquire an initial shared prefix; an existing ID matches its acquired
or computed data. GPU and CPU share a membership catalog. No parent, lineage
declaration or ID structure is used.

CPU retention protects complete working sets across cache groups. Releasing an
agent preserves other retained contexts' shared references. A finished request
keeps its complete context for its next turn. Transfer pins protect physical
data separately, allocation plans precede eviction, and failed dependencies
invalidate only the contexts that need them. User-count sizing uses normalized
block/window geometry, including grouped specs and EAGLE verification.

Coalesced CPU chunks advertise only written data. Each lookup checks the actual
candidate window, and filling missing data preserves existing agents' acquired
subsets. Secondary storage receives canonical-complete entries. Selection inside
a larger chunk acquires only the selected extent; a 72-token fork cannot acquire
later attention data from a 96-token physical entry. Native stores preserve every
source agent's acquired subset when copying shared data from GPU to CPU.

Fresh IDs can observe initial shared hits through latency. IDs provide cache
accounting and matching semantics, without authentication or confidentiality
promises, agent-derived cache salts, or artificial timing padding.

Validation: 544 scheduler, cache, geometry, protocol and tiering tests pass in
offline CPU containers. The production patch framework reconstructs all twenty
stages and matches the tested sources exactly; its thirteen transaction tests
pass. The installed-image CPU unit covers initial forks, existing-agent matching,
GPU/CPU membership, physical-copy preservation, surviving shared references and
sparse fills. The required backend check passes, including exact reconstruction,
the 94-file deployment-input manifest and the installed shared-prefix CPU unit.
The v23 image and archive remain awaiting adoption.

## Findings 4 and 12: Anthropic terminal metadata

`vllm-anthropic-terminal-metadata.patch` makes streaming and batch responses use
one translation of the observed Chat completion cause and actual tool-use
output. A matched stop string becomes `stop_sequence` with its exact contents.
Named tools report `tool_use` even when Chat reports `stop`; Chat's named-tool
behavior remains correct. Token limits take precedence and retain `max_tokens`.
The obsolete finish-reason map is deleted.

Streaming retains content carried in the first chunk, captures the matched stop
alongside the finish reason, and requires terminal usage and the terminal marker
before reporting a completed message. Missing, repeated or out-of-order terminal
data produces a native API error with no success marker.

Validation: 127 offline CPU tests pass, including the streaming/batch terminal
matrix, exact stop-string preservation, named calls, length precedence and
incomplete-stream errors. A four-case control against the previous runtime
reproduces incorrect `end_turn` metadata for stop strings and named tool calls.
The reviewed source stage compiles with only the two intended source/test changes.
The required backend check passes for all twenty-one source stages, the 95-file
deployment-input manifest and installed CPU units. Image adoption remains pending.

## Remaining implementation

Parser language, stop handling, schema conversion, protocol translation, image
admission, remaining triage findings, client recovery verification and final
backend/client pin adoption are still under review. Their completion is not
implied by the finding-10 validation above.
