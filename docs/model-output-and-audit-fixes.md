# Model-output and deployment audit resolutions

This record accompanies the unchanged `qwen38-deployment-correctness-audit.md`.
The independent triage and handling policy supplied with the implementation brief
decide the resolutions. Source validation here does not certify a built image or a
live release. The v23 image and archive are awaiting adoption.

## Findings #1 and #13: tool commitment and caller text stops

`vllm-tool-output-completion.patch` commits Qwen calls only when the actual
closing wrapper was observed and generation ended at model EOS. EOS-cut and
malformed spans remain verbatim text; length keeps its diagnostic prefix.
The parser preserves unknown and padded names for client error feedback.
Chat, Responses and batch derender use the same terminal-aware parser methods.
Streaming holds call entries and subsequent content until the terminal;
preceding content and reasoning still stream.

Sampling parameters distinguish all configured EOS IDs from caller stops.
The deployed V1 scheduler probes the native grammar before allowing a caller
stop token. A text matcher for the same structural grammar protects stop
strings during detokenization, including overlaps with a closing wrapper.
It is created only for requests with structural output and stop strings.
There is no wrapper-depth heuristic, new wire protocol, or retry path.

Validation: 4,358 parser, Chat, Responses, derender, stop and EOS tests pass
in offline CPU containers. This includes 40 tests through the real output
processor and the pinned Qwen tokenizer, checking batch/stream output and
exact sampled IDs. The installed parser unit has 12 tests plus 126 subtests.
The build check reconstructs and verifies every runtime file and runs the
installed unit. No model, service or GPU was run.

The client review and release dependencies are recorded at the end of this document.

## Precise request errors on the served APIs

`vllm-precise-request-errors.patch` removes the generic ValueError/TypeError/
OverflowError-to-400 mapping and exception-class-name matching for templates.
The HF renderer supplies a typed callback for deliberate template guards;
request kwargs cannot replace it. Template syntax/undefined-variable errors
and unexpected renderer exceptions keep their original server cause.

Image option and source gates, malformed base64 and native image-decoding
failures carry typed validation errors at the input boundary. The pinned
pixel-limit environment invariant is a server error. Unexpected errors in
the image processing pipeline remain server errors. Prompt lengths that
exhaust or exceed the window also carry typed validation errors.

The shared classifier still maps explicit unimplemented operations to HTTP
501, and typed client/server exceptions and generation failures retain their
declared meanings. Raw exception handlers remain registered inside the metrics
middleware so the recorded status matches the response. These registrations
do not turn unknown errors into client refusals.

Fourteen causal controls fail before the fix. The changed request/image/metrics
suites and focused renderer cases pass offline; installed image and template
units verify the typed refusals and internal-error distinction. No model ran.

## Finding #23: exact token positions through text stops

`vllm-token-text-provenance.patch` tracks native ByteLevel ASCII terminal
positions while consuming every generated token ID. V1 output and the scanner
share the existing incremental decoder and its existing UTF-8 handling.
Visible text consumes those spans in order. Unicode bytes and text holdback
cannot move a real marker onto a prose lookalike. A terminal is released only
when all of its text is visible; a visible fragment remains ordinary text,
and a permanently stripped suffix is never emitted at finish.

Scope: four runtime files in the scanner, its parser feed, the shared decoder
helper and V1 detokenizer. No transport fields or engine protocol change.
Other tokenizer formats retain their existing scanner behavior; the pinned
Qwen ByteLevel path does not use textual anchor searches.

Validation: 4,333 parser/native-stop tests and 221 Chat/Responses tests pass
(one existing expected failure). This includes 64 native scanner cases and
64 Qwen output-processor cases for literal versus real markers, split UTF-8,
invalid UTF-8 replacement, stop buffering, partial markers, both transports
and exact reasoning counts. The installed parser unit has 15 tests.
Twenty-nine corrected controls failed before the fix. No model ran.

## Findings #2 and #18: schema-faithful Qwen XML arguments

`vllm-schema-faithful-xml.patch` places decoding at the Qwen XML boundary,
before raw parameter text loses its lexical meaning. The shared engine passes
the tool name to the format converter on both transports; Qwen resolves the
complete parameter schema instead of applying generic type coercion afterward.
Nested JSON values keep their types and original serialized representation.

The decoder checks string and JSON interpretations against resolved property
constraints and validates the complete object. Known discriminator values narrow
conditional and disjunctive branches before joint interpretation checks. Strings
retain their bytes when valid; schema-required grammar padding is distinguished
from value whitespace. A string enum containing "null" stays a string, even
when null is also a declared type. Unclosed parameters and values with no valid
interpretation remain raw; no partial value is repaired or coerced.

Validation includes references and escaped pointers, enum/const/composition,
conditional discriminators, native XGrammar acceptance, exact numeric/JSON
representation, unfinished diagnostics, and batch/stream cuts. The installed
parser unit has 14 tests. Parser, Chat and Responses regression results are
recorded with the required source/image reconstruction check. No model ran.

## Finding #17: one-way thinking boundary on V1

`vllm-one-way-thinking-boundary.patch` exposes the atomic boundary already
derived by the parser for exact reasoning usage. Qwen ends thinking at its
natural closer or implicit tool-start token. The V1 thinking-budget holder
latches that boundary and stops forcing. A later thinking opener in answer
text or a tool value cannot reopen the phase. The forced closing sequence
remains distinct from the tokens that naturally end thinking.

The grammar detects the first boundary in the newly generated IDs and keeps
the implicit tool trigger. Earlier prompt history cannot make it trim that
trigger away. Existing final-response counting is unchanged. The correction
touches seven runtime files in the parser, reasoning configuration, V1
budget holder and structured-output manager.

Validation: 64 CPU budget/manager cases pass, including 20 new Qwen cases;
3,998 parser cases and 62 native stop/OutputProcessor cases also pass. The
installed phase-budget unit exercises natural and implicit boundaries,
retained output, chunking, prompt history, unchanged logits and native
grammar acceptance of a call with literal thinking markers in its value.
No model or GPU was run.

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

Terminal promotion and caller stops are covered by the EOS tool-output stage;
schema conversion and the V1 thinking boundary have their own stages above.
Exact token/text provenance is covered by its dedicated stage above.

## XML string and content fidelity

`vllm-xml-text-fidelity.patch` preserves every byte of an unconstrained XML string,
including whitespace-only values. The schema stage above handles padding only
where the complete schema requires it. The model's source template stays
unchanged, and historical reasoning is retained. Where a value's newline framing
ends and the value begins is settled by the canonical-framing stage below.

The shared parser also removes its batch-only content-stripping setting and all
consumers of that setting. Nonempty content around calls now retains the same
bytes on both transports; the existing whitespace-only gap normalization remains.
Four newly modified parser configurations join the full image provenance cascade.

Validation: 3,954 offline parser tests pass. The previous runtime fails 101 focused
controls, with 20 unaffected controls passing. The source tests cover multiple
chunk sizes, empty and whitespace-only strings, Unicode, embedded reserved
markers, surrounding content and incomplete string diagnostics. Seven installed
parser tests pass,
including actual derived-template rendering, XGrammar acceptance and stream/batch
round trips. Template retention now also runs in the ordinary CPU source check.
Schema decoding, incomplete diagnostics, terminal promotion and token/text
provenance are covered by the dedicated stages above.

The integrated generator and `build-vllm.sh check` pass with 26 reviewed stages,
102 deployment inputs, 19 framework/recipe tests and all installed CPU units.

## The decoder owns the Qwen value language

`vllm-qwen-owned-tool-grammar.patch` adds the thirty-fifth runtime source
stage. vLLM registers its own `qwen_3_coder` structural tag, joining `hermes`,
`minimax` and `kimi_k3`, and no longer consumes XGrammar's builtin Qwen
template for this model.

The reason is one rule the structural-tag API cannot express. XGrammar emits
the Qwen value channel from C++ as `TagDispatch(excludes=("</parameter>"))`,
excluding a value's own closer and nothing else, so `<parameter=` is ordinary
value text. A value could therefore absorb the next parameter's opener, and the
result was not a parse error: it was a different parse. The captured failure is
exactly that -- one dropped `</` merged a path, an opener and a hundred
kilobytes of body into a single argument, the required chain was then satisfied
by an empty second parameter, and the call was published as well formed,
ordered and schema-satisfying.

The vLLM-owned builder excludes the opener as well. At the token that would
complete `<parameter=`, the bitmask forbids it and the only legal continuations
are more value text or the real closer, so the slip self-corrects at decode
time into the call the model meant rather than being repaired afterwards.
Everything else reproduces XGrammar's Qwen language exactly: the `[ \n\t]*`
padding around every value, the ordered required chain, optional properties,
additional properties, string enums and constants, local `$ref` resolution, and
JSON productions for every non-string type. The one production it does not have
is XGrammar's constrained-string regex: a string declaring `pattern`, `format`,
`minLength` or `maxLength` is refused, naming the tool and the property, because
the exclusion cannot be written inside a length-bounded regex -- xgrammar's
regex engine has no lookahead, so forbidding a fixed substring is only an
unrolled DFA, which cannot then carry a `{m,n}` bound. There is one string
channel, not a second one that drops the exclusion. The call-count limit is
applied while building instead of by mutating a returned tag, and the
now-unreachable mutation is deleted.

What this forbids is real and worth stating: a tool argument can no longer
carry the literal sequence `<parameter=`. That extends a limitation the format
already had for the exact closer, and it is visible to the model rather than
silent -- a tool author needing either sequence must use another
representation, such as encoded input their tool decodes.

Validation: the built tag is compared against XGrammar's own Qwen tag over
eighteen schema shapes -- required, optional, additional, typed, enum, constant,
union, nested object, array, local reference, reference to a string, and
length-, minimum- and pattern-constrained strings -- across automatic and
required choice, both reasoning settings, and `strict` true, false and absent.
Every accepted and rejected string agrees except the merge itself, which the
owned grammar rejects and XGrammar accepts, and the three constrained-string
shapes, which the owned grammar refuses to build at all rather than build
without the exclusion. The call-count behaviour agrees with the deleted
mutation on single, duplicate and trailing-text calls. Tests
that placed the opener inside a value move with the grammar and still assert
that every other marker stays value text.

## Canonical parameter framing

`vllm-qwen-canonical-parameter-framing.patch` adds the thirty-fourth runtime
source stage, and the served template's derivation drops its Stage C.

The Qwen XML transport pads every parameter: the grammar writes `[ \n\t]*` on
both sides of a value by construction, and the model's own template renders
`<parameter=NAME>` newline VALUE newline `</parameter>`. A derivation stage had
deleted that framing from the served template and from the in-context
instruction example, on the reading that whitespace between the tags belongs to
the string. That reading is right about a decoded value and wrong about the
transport. The model was not retrained, so it kept emitting the framing it was
trained on and the deployment began reading that framing as data: a written file
gained a blank first line and an extra trailing newline, silently and on disk.

The framing is now restored and inverted exactly once.
`_unframe_parameter_value` removes one leading and one trailing newline, so
`decode(encode(v)) == v` for every `v` — a value that itself begins or ends with
a newline travels as two and keeps its own — and nothing becomes inexpressible.
A value the model hugs against its tags decodes to the same string as one it
frames, which makes the model's slot-bound framing habit irrelevant rather than
load-bearing. The one parameter whose closer has not been generated yet has no
observed trailing newline, so only its leading newline is removed and a trailing
newline there stays value text.

Validation: the served template reproduces from the model's own through the
remaining named stages and matches its pinned bytes; the round trip from
rendered history back through the parser returns every value unchanged, over
empty, newline-only, both-framed, Unicode and embedded-marker values. The
installed parser unit carries the canonical framing in its shared call helper,
so the grammar-acceptance, schema-typing and history round-trip checks all
exercise the transport the model actually emits.

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
`min_p`. All five supported generation families inherit the same configured
sampling policy: Chat (including batch Chat), Completions, Responses, Anthropic
Messages and token-in-token-out generate. They expose six generation routes;
generative scoring and Cohere are not mounted.
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

The separate finding-17 stage above supplies the initial and implicit reasoning
boundaries; budget resolution and phase accounting are both required.

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

The shared classifier accepts typed request causes. The precise-error stage
types the deployed template/image/context producers and removes raw Python
and template-name inference. Unexpected failures remain 500 `api_error`.
HTTP statuses without a dedicated Anthropic error name retain their HTTP
status and use the corresponding client/server error family.

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
The exact `<tool_call>\n<function=` trigger starts a call after reasoning.
The phase-aware stage above recognizes both valid tokenizations of that trigger.
Bare function markup, stray openers, empty wrappers and unfinished
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
each call reached its actual closing wrapper. The EOS tool-output stage uses
this observation and the terminal cause to decide promotion.

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

The dedicated stages above complete caller stop suppression, EOS promotion,
schema conversion and exact scanner positions.

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
builder. With `false`, the grammar stops after its first complete call.
Automatic choice still allows an ordinary text answer; required choice still
requires a call; named choice already contains one tag. The schema and
reasoning prefix retain their behavior. This stage applied the limit by
mutating the tag XGrammar returned; the Qwen grammar stage below owns the
builder and applies it while building.

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
transports. The EOS tool-output stage subsequently completes the call-promotion
rule, retaining interrupted raw text instead of executable calls.

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

## Shared prefixes: stable identity within an input stream

`vllm-input-stream-agent-identity.patch` binds an input stream's required opaque
agent ID before reading chunks. Every chunk must keep that exact ID. A different
ID is rejected before dispatch, and the existing input-error path aborts only
the affected generation request. This keeps an ID change from continuing another
agent's live KV under inconsistent sampling metadata. A different agent submits
a new generation request and follows ordinary prefix selection and acquisition.

Validation: 30 offline source tests pass using actual input admission, stream
dispatch and generation error handling. Both supported output modes, first and
later chunk changes, exact opaque IDs, invalid IDs, valid sampling updates and
request-local abort are covered. Eight controls reproduce the old acceptance of
changed IDs; 22 unaffected cases pass against that source. The installed shared
prefix CPU unit also exercises accepted and rejected stream identities. The
required backend check passes for all 28 reviewed stages and 104 deployment
inputs, with AsyncLLM included in upstream, final, image and installed hashes.
The v23 image and archive remain awaiting adoption.

## Owner verification: supported generation routes

Commit `8bb4f92` unnecessarily remounted generative scoring. Section 3 does not
require that API, so the follow-up removes its registration and serving state
from the shared-prefix transformation again. The scoring-specific deployment
changes, source hashes, image copies, tests and serving-document changes are
removed. Cohere remains unmounted. No shared-prefix ownership rule changes.

The supported set is five protocol families and six generation routes, including
batch Chat. The source protocol test and installed shared-prefix CPU unit check
this exact supported set and the absence of both excluded API families. This
also agrees with the image recipe's existing route-absence assertion. Historical
scoring tests from `8bb4f92` are not evidence for a supported deployment surface.

Validation: the installed route assertion fails against the remounted runtime
and passes after excision. The shared-prefix CPU unit and all 72 protocol cases
pass in offline containers. No cache ownership or eviction behavior changes.

## Owner verification: asserted degeneracy policy cases

`scripts/tool_output_degeneracy_probe.py` now asserts the actual reasoning,
content and decoded calls required by S1, S2, S7, S8 and S11, alongside the
existing stream/batch comparisons. Thirteen cases cover bare/quoted functions,
stray and empty wrappers, prose surrounding a call, all five S8 thinking shapes,
an EOS-cut wrapper and reserved markup inside a string value. Each runs in batch
and streams of one token, seven tokens and the complete response.

Controlled replay uses the deployed native incremental decoder and the
terminal-aware parser methods, so EOS finalization is exercised. The pinned
tokenizer replay passes all 13 asserted cases and the existing 30 prefix
comparisons. A negative control makes both transports delete the same bare
function text: equality still holds, but the S1 policy assertion fails. This is
offline CPU evidence; the live model portion of the probe was not run.

## Owner verification: retained Qwen name-validation fallback

`qwen3_config` deliberately retains `validate_tool_names=False`, an explicit
exception to the policy's suggested `True` defense in depth, as permitted by
follow-up ruling 3. Native strict grammar already rejects undeclared names.
When a complete, exactly wrapped unknown or whitespace-padded name nevertheless
reaches the parser, its unchanged name and arguments remain available for the
client's S5 unknown-tool error result. The brief requires that feedback behavior
to stay. `True` instead demotes the entire call to raw text in both transports,
which routes the client through the slipped-markup notice rather than the
unknown-tool result. It does not delete that raw text.

This preserves diagnostics without trimming names or deriving calls from prose;
the exact trigger, observed closing wrapper and EOS gates still apply. The
backend never executes a tool, and clients must validate registered names and
arguments before execution. The installed parser unit already asserts unknown
and padded name preservation for batch and both streaming chunk sizes. Six
offline comparisons verify the behavior of both flag values, and actual
XGrammar rejects both names while accepting the declared control. Evidence:
`/tmp/codex-fix/followup-probe/name-validation.log`. Runtime and its contract
remain unchanged; the retained-fallbacks section of the implementation report
now records this decision explicitly.

## Serving-machine build: stale assertion types and parser fixture

The v23 build failed inside the Dockerfile unit block because its full-context
`get_max_tokens` assertion still caught `ValueError` after `8833ce1` correctly
retyped that refusal. The assertion now catches `VLLMValidationError`, with its
import moved before first use. Runtime refusal types and the prohibition on broad
Python-exception-to-400 classification remain unchanged.

The sibling sweep found two further mismatched assertions: malformed tool history
in the Dockerfile must catch `VLLMValidationError`, and the raw-image generation
test must expect that same type for decoder rejection on both stream settings.
The latter is included in the precise-request-errors reviewed transformation and
regenerated landmarks. Executing the inline block also exposed a stale parser
fixture that bypassed initialization; it now constructs `Qwen3Parser` normally
with a mock tokenizer, preserving the actual reasoning-boundary state.

All 102 original `pytest.raises(ValueError)` occurrences in the generated stages
were audited against their final source. Repeated before/review landmarks include
historical image/history assertions already changed by later stages and deleted
CPU-policy tests. The 13 genuine remaining assertion sites cover exact-count
invariants, capacity and offload configuration/factory errors; their exception
types remain unchanged. The Dockerfile's offload-config `ValueError` assertion
also remains unchanged.

The optional run of these genuine `ValueError` cases was not all green: 12
parameterized cases passed, five sizing cases could not construct their default
model metadata offline, and three connector cases stopped at the preexisting
missing `_build_config` helper. Their production paths still raise `ValueError`;
these fixture problems are outside the retyped-refusal correction and remain
unchanged. They are recorded in `build-refusal/genuine-value-errors-final.log`.

The original full-context assertion fails against the typed runtime, and the old
raw-image test fails on both transports. After correction, all 37 raw-image and
API-utils cases pass. The entire Dockerfile inline Python unit block, including
its invoked CPU units, passes when extracted verbatim and executed in a disposable
offline container. No image build was run here.

`build-vllm.sh check` does **not** execute the Dockerfile: it checks its identity
and packaging contracts and runs selected CPU units. That gap explains why the
serving machine's real build first exposed these stale inline assertions. The
separate inline-block execution above verifies this correction without claiming
that a new image was built. Logs and the detailed sweep are under
`/tmp/codex-fix/build-refusal/`.

## Retained decisions and release boundary

Finding #15 is workstation archive state, not a source defect. Archive verification
remains mandatory. Finding #21's pre-SM-8.9 refusal remains deliberate: this is the
SM 12.0/K8V4 deployment, with no older-device MSE fallback. The guard unit checks
that refusal. The Chat half of #12 already follows its protocol and is unchanged.

The owner stopped the expanded phase draft. Its sampling-failure/retry protocol,
streaming-session redesign and undeployed runner/Rust/speculative changes were
discarded without a backend commit. The retained #17 stage is the seven-file V1
thinking-boundary correction described above. The agent retry commit `ddb85d0`
was reverted by `2566435`; S9 was never implemented or committed.

The client review lives in
[`agent_service/docs/model-output-handling-review.md`](../../agent_service/docs/model-output-handling-review.md).
The final implementation report and exact validation logs are under
`/tmp/codex-fix/`. CPU source checks do not certify GPU execution or deployment.
The coordinator must build and adopt the v23 image and archive, then synchronize
the agent backend image ID before cutting the dependent agent-service release.
The launch profile and cache volume remain v21; the image profile is v23.
