# Qwen3.6 historical setup → Qwen3.8 current deployment audit

Date of audit: 2026-08-15 through 2026-08-16

This is an evidence record, not a migration recipe. The historical repository was
read as a set of bug reports, design ideas, and test hypotheses. None of its runtime
monkey patches was copied into this project. Every item below was re-evaluated against
the pinned Qwen3.8 checkpoint, its real tokenizer/template, the pinned current vLLM
source, and the actual RTX 5090 runtime.

## Pins compared

- Historical repository: `/home/user/Desktop/qwen_36_agent_setup`, Git commit
  `79249a8b7a5eeb7174050decb4864fedacc4667e`.
- Historical model: `QuantTrio/Qwen3.6-27B-AWQ`, revision
  `9b507bdc9afafb87b7898700cc2a591aa6639461`.
- Historical vLLM: `8cd174fa358326d5cc4195446be2ebcd65c481ce`.
- Current model: `unsloth/Qwen3.8-27B-NVFP4`, revision
  `16b6615af3548b88e2d8e382457bc705b00479cf`.
- Current vLLM: `9df9b0b0a1816b6d0d0f6ecd0da563cc37fd72f5`.

The historical clone was left clean. It is outside this repository and is not a
runtime dependency.

## Architecture conclusion

Qwen3.8-27B is not a new vLLM architecture relative to Qwen3.6-27B. Selected fields
from the two pinned `config.json` files are identical:

| Field | Qwen3.6-27B | Qwen3.8-27B |
|---|---:|---:|
| outer architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| outer/text model type | `qwen3_5` / `qwen3_5_text` | `qwen3_5` / `qwen3_5_text` |
| hidden size | 5,120 | 5,120 |
| intermediate size | 17,408 | 17,408 |
| layers | 64 | 64 |
| full-attention/GDN pattern | 16 full + 48 linear, repeating 3:1 | same |
| attention heads / KV heads | 24 / 4 | 24 / 4 |
| full-attention head dimension | 256 | 256 |
| GDN key/value heads | 16 / 48 | 16 / 48 |
| GDN key/value head dimension | 128 / 128 | 128 / 128 |
| native maximum positions | 262,144 | 262,144 |

Therefore, the relevant changes are post-training, weight/quantization recipe,
tokenizer metadata, chat-template behavior, recommended sampling/output policy, and
the large intervening vLLM frontend/parser changes. Tool-calling improvements must
not be attributed to a fictitious `Qwen3.8ForConditionalGeneration` implementation.

## Prompt/template conclusions from real tokenization

The Qwen3.6 and Qwen3.8 templates were rendered using their actual tokenizers inside
Docker. The checks were not character-count estimates.

- The historical Qwen3.6 template rejects `developer`; the selected Qwen3.8 template
  merges a leading run of `system` and `developer` messages into the system block.
  This directly improves compatibility with coding-agent clients.
- Qwen3.8 introduces explicit reasoning-effort instructions. Its upstream template
  accepts `xhigh`, `medium`, and `low`, with `high` mapped to `xhigh`.
- This deployment intentionally narrows that policy: Qwen `xhigh` is the only mode;
  Anthropic/OpenAI aliases `high` and `max` map to the exact same prompt; `medium`,
  `low`, and disabled thinking fail with HTTP 400 rather than silently degrading.
- The Qwen3.8 template has stronger historical-tool-call validation: missing names or
  malformed argument types fail loudly. The project retains those checks.

Live policy evidence from the final image:

- `xhigh`, `high`, and `max` rendered identical token IDs;
- OpenAI low/disabled and Anthropic low/disabled requests all returned HTTP 400;
- Anthropic adaptive/max returned a thinking block;
- legacy `reasoning_content` was normalized on real Chat Completions ingress.

## Historical workaround review, one item at a time

### 1. `monkey_patch_qwen3_coder.py` — truncated XML parser crash

Historical idea: make truncated tool XML safe and fail loudly if the old parser
landmark changes.

Disposition: **do not carry the patch**. Current vLLM replaced the old independent
`Qwen3CoderToolParser` path with the unified streaming Parser Engine and current
`vllm/parser/qwen3.py`. The obsolete function body the patch replaced is not the
current serving path. Applying the wrapper would either fail or restore a superseded
parser model. Current reconstruction/truncation tests passed. That does not imply
the entire decoder-time grammar integration was correct; item 5 records the separate
current-code boundary defect found by this audit.

Good idea retained: parser behavior is tested at both unit and live protocol levels,
and malformed historical tool calls fail rather than being coerced silently.

### 2. `monkey_patch_hybrid_kv_allocator.py` — hybrid-cache capacity reporting

Historical idea: prevent under-reporting caused by treating only one hybrid cache
group as authoritative.

Disposition: **upstream/current behavior supersedes it**. Current vLLM contains
group-aware hybrid KV capacity accounting and reports the measured capacity directly.
The actual final runtime reported 264,115 GPU KV-cache tokens and 1.01× concurrency
at a 262,144-token request length. No historical allocator wrapper is installed.

Good idea retained: measured allocator output, not a hand-computed theoretical cache
size, is the deployment authority.

### 3. `monkey_patch_reasoning_field_egress.py` — output alias

Historical idea: rename vLLM's `reasoning` to the client-specific
`reasoning_content` field.

Disposition: **reject the global rename**. Current vLLM intentionally emits
`reasoning`; its history documents removal of `reasoning_content` output as a
breaking-client change (`005fa01756`). Qwen's own examples tolerate either field,
and the native Anthropic endpoint emits typed thinking blocks. Replacing the current
field globally would create another compatibility fork.

Known boundary: clients that insist on `reasoning_content` *on output* need a client
adapter. This server accepts the legacy alias on Chat Completions *input*.

### 4. `monkey_patch_reasoning_field_ingest.py` — legacy input alias

Historical idea: normalize assistant-history `reasoning_content` to `reasoning` so
preserved reasoning is not silently dropped.

Disposition: **fixed upstream**, by vLLM commit `346cf163a1`; no patch copied. A live
Chat Completions request proved the alias is normalized.

Narrow caveat: `/tokenize` uses its own `TokenizeChatRequest` model and bypasses the
Chat Completions request normalizer. That diagnostic route is not inference ingress.
Use `reasoning` there.

### 5. `monkey_patch_tool_call_in_think_detector.py` — tool call before `</think>`

Historical idea: detect a Qwen tool call emitted while the old reasoning parser still
believes it is inside thinking.

Disposition: **the historical warning patch is not carried, but its invariant was not
fully superseded**. Current `vllm/parser/qwen3.py` explicitly defines
`(REASONING, TOOL_START)` as an implicit reasoning-end transition, so post-generation
streaming and non-streaming parsing reconstruct the call correctly. The historical
patch only warned; it did not repair grammar enforcement.

First-principles tracing found a different current defect. The structured-output
scheduler recorded the `<tool_call>` token itself as the final reasoning token and
trimmed it before advancing XGrammar. Qwen's triggered grammar begins with
`<tool_call>\n<function=`. Without the trigger, an unknown function and schema-invalid
arguments remain ordinary free text and are accepted even though the final parser
can still return a plausible structured call.

The new current-source patch makes `Qwen3Parser.extract_content_ids()` retain an
unpaired tool-start marker and makes `StructuredOutputManager` record the implicit
reasoning boundary immediately before that suffix. With the real tokenizer,
`<tool_call>` is the single special token 248058. Feeding it to the real XGrammar
matcher rejected an unknown function at token 5, a nested wrong type at token 18,
and an extra property at token 20. A sensitivity control dropping only token 248058
accepted the unknown call, proving the regression test detects the original bypass.

### 6. `monkey_patch_default_sampling_params.py` — model-specific defaults

Historical idea: server-side explicit model-recommended defaults are essential
because local clients often omit sampling fields.

Disposition: **adopt the principle, reject the Qwen3.6 values and monkey-patch
mechanism**. Qwen3.8's published thinking values are different and now explicit:

```text
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
presence_penalty=0.0, repetition_penalty=1.0
```

They are passed through vLLM's pinned `--override-generation-config`. A small reviewed
source patch extends the same default path with separate reasoning/final ceilings and
Anthropic thinking conversion. The historical precise-coding temperature of 0.6 and
81,920-token combined cap were not imported.

### 7. `monkey_patch_repetition_detection_default.py` — automatic loop detector

Historical idea: forcibly stop repeated patterns as a defense against Qwen3.6 looping.

Disposition: **reject**. Qwen3.8's new recommended configuration specifies neutral
`repetition_penalty=1.0` and does not recommend vLLM's repetition detector. The final
build test asserts `SamplingParams.repetition_detection is None`. A detector that can
terminate legitimate repetitive code/data is not a correctness-preserving default.

The project does not claim repetition is impossible; it follows the new training-time
recommendation and uses proper stochastic sampling. It also rejects the idea that a
fixed seed or “deterministic prompt” proves deterministic GPU/model behavior.

### 8. `monkey_patch_qwen3_coder_grammar.py` — schema-constrained tools

Historical idea: once the model chooses a tool, constrain the generated arguments to
the declared JSON schema instead of trusting post-hoc parsing.

Disposition: **adopt narrowly after current-code review**. Current vLLM normally
activates structural schema constraints in automatic tool mode only if the client
marks the tool `strict:true`. Claude-like clients may omit that flag, and XGrammar
deliberately replaces the parameter schema with `true` for explicit `strict:false`.
The reviewed current-source patch forces the real schema for omitted, false, and true
strict values only when the exact structural-tag model is `qwen_3_coder`. It leaves
the decision *whether* to call a tool to the model, keeps `tool_choice=none`
unconstrained, and preserves upstream non-strict behavior for `llama`, `qwen_3_5`,
`deepseek_v4`, and every other parser. The runtime environment variable
`VLLM_ENFORCE_STRICT_TOOL_CALLING=1` is explicit.

This is not copied historical code. The implementation targets current structural
tags and was exercised through current bitmask/JIT grammar generation and live tool
calls.

### 9. `monkey_patch_request_memory_snapshot.py` — startup-memory slack

Historical idea: compensate for vLLM's own CUDA initialization footprint when using a
near-1.0 `gpu_memory_utilization` fraction.

Disposition: **not needed and not installed**. This deployment uses an exact
`--kv-cache-memory 6925634765`, not a utilization heuristic. The current vLLM path
with explicit cache bytes skips the profiling decision that motivated the historical
slack. Adding an environment-tunable GiB of slack would weaken the exact-memory lock.

### 10. `monkey_patch_tool_role_media_preserve.py` — media in tool results

Historical idea: preserve image/video parts attached to `tool` messages.

Disposition: **adopt the invariant through current-source changes, not the historical
patch**. Vision is now inseparable from the one supported profile. The current patch
accepts typed content on the originating OpenAI tool role, keeps Anthropic
`tool_result` text/image/text parts together, validates tool IDs, and renders the
vision marker at the same chronological Qwen-template position. It does not detach
the image into the newest user turn.

Equivalent OpenAI and Anthropic histories rendered to exactly the same 16,562 token
IDs. Moving the image changed the token IDs. The deployed Qwen Code client pins
`splitToolMedia=false` and `toolResultContentFormat=parts`; a real native
`read_file` image tool turn remained in later requests, producing one multimodal
cache hit in its first four-turn session and two of two hits in the repeated session.
This directly proves the model, template, vLLM frontend, and client agree on the
chronology.

### 11. `monkey_patch_mm_cache_validator_eviction.py` — multimodal cache drift

Historical idea: if validation rejects a request after sender-cache mutation, restore
the sender/receiver cache invariant.

Disposition: **fixed upstream and exercised here**. Current vLLM includes the P0/P1
multimodal processor-cache recovery change (`3962042304`), so no historical cache
validator patch is installed. The one profile uses a 4-GiB SHA-256-keyed multimodal
processor cache. Cold/warm tests separately varied image bytes and chronological
position, proving that byte identity controls the multimodal cache while stable token
prefix controls the prefix cache. Real Qwen Code tool-image history then produced the
expected live cache hits without sender/receiver drift.

### 12. `monkey_patch_qwen3_coder_streaming_truncation.py` — dropped partial args

Historical idea: preserve cumulative partial tool arguments and make truncation
observable in streaming output.

Disposition: **do not carry the old parser wrapper, but do not assume the rewritten
frontend is therefore correct**. Current vLLM's unified Parser Engine replaced the
old target/state layout and has dedicated Qwen delta-boundary tests (`a0df04e477`).
The installed-image suites completed 286 focused parser/grammar/Anthropic-conversion
tests and 382 Qwen streaming/replay tests, yet new fault injection still found four
current-code defects: Chat streaming could relabel an in-tool `length` cutoff as
`tool_calls`; Responses streaming could emit executable/completed events for a
truncated call; Qwen batch parsing dropped the final unterminated parameter prefix;
and generic batch parsing could strand text after a tool. Narrow fixes target the
current code rather than resurrecting the historical wrapper. A 30-prefix corpus with
the real tokenizer now proves stream/batch semantic parity, and live truncation tests
cover Chat, Anthropic, and Responses. The implicit grammar-trigger bug in item 5 was
also outside the old wrapper's scope and retains its dedicated sensitivity test.

### 13. `launch_with_patches.py` and `sitecustomize.py` — propagation machinery

Historical idea: runtime patches must either appear in every spawned interpreter or
cause startup to fail.

Disposition: **retain the fail-closed objective, reject runtime monkey-patching**. The
current changes are baked into an immutable Docker image. The build verifies upstream
file hashes, patch hashes, live diffs, installed file hashes, and functional tests.
Startup then re-hashes all twenty-nine reviewed runtime files, seven reviewed test
files, the template, Dockerfile, allowlist, and build units. There is no
launcher/sitecustomize registry that can drift between processes.

## Current-source changes that actually remain

Nine reviewed diff artifacts reconstruct twenty-nine runtime-source changes, seven
reviewed test changes, and one reviewed new workspace test:

| Diff | Purpose | SHA-256 |
|---|---|---|
| `patches/vllm-turboquant-k8v4-direct-workspace.patch` | remove unsafe duplicate K8V4 continuation workspace allocations | `a9721067f1a7ee9497a4bd51e47e3a474561189e881b4704bfc4beac8ea48380` |
| `patches/vllm-enforce-auto-tool-schema.patch` | schema-constrain Qwen auto tool arguments for strict omitted/false/true without broadening other parsers | `4f75c793a9c2cdcfb2fd0768ba49a4e34748d3a37d8392b07d3592ca50939c07` |
| `patches/vllm-qwen38-agent-defaults-and-thinking.patch` | explicit model defaults, Anthropic thinking controls, phase-ceiling request plumbing, fail-closed ordered tool-result correlation | `6428d2cfa77f28e57e117999d0ec8fab5430856c985ba530e04885c2f5c420b7` |
| `patches/vllm-qwen38-separate-final-response-budget.patch` | count final tokens only after explicit reasoning end and enforce a separate hard ceiling | `f20d7dff41931248272842ed2c7a163c6f013e405ccf35733c40ff131a2fc503` |
| `patches/vllm-qwen-implicit-tool-grammar-boundary.patch` | retain an implicit Qwen tool-start token as the structural grammar trigger and recover a final partial Qwen parameter consistently | `d231c6e2e7040c4cd4b38432cb8c794805afddbf2c6e4f7ff6febb78e3fd9f48` |
| `patches/vllm-anthropic-validation-http400.patch` | report Anthropic request-conversion validation failures as HTTP 400, not HTTP 500 | `030b64be104e6ef57a40f6bae740dfa9d4634a420c6c93a395f62bfb98d6d053` |
| `patches/vllm-tool-truncation-finish-reason.patch` | preserve truncation terminals, fail Responses incomplete events closed, flush deferred batch content, and carry phase budgets through Responses | `1a220f6db9b40967d867b3cfb1a92d95d907ca059718ffe61772b4cb4409f551` |
| `patches/vllm-qwen38-vision-runtime.patch` | enforce lossless static-PNG ingress, chronological tool media, full BF16 image processing, and exact reclaimable vision workspace without reducing context, graphs, or prefill chunking | `f92603724861da5b5a364f43e57d3f95ef43a9dded8ae645278373850db3140f` |
| `patches/vllm-qwen38-numerical-audits.patch` | add exact TurboQuant K8V4, Qwen3.8 context/MRoPE, and real-checkpoint NVFP4 production-kernel audits to the immutable build | `a73aa2f2ae3f82010eb2bafcdf663c2fe14854c30165dbc4d8457725bc3b6632` |

The separate final-response ceiling is deliberately a hard ceiling: EOS and configured
stop sequences still win, but `min_tokens` cannot force generation past it. Multi-token
reasoning-end delimiters are handled. Unit tests cover exact boundary, serialization,
EOS precedence, `min_tokens`, invalid values, and server-ceiling clamping.

One honest limit remains: the final phase begins when the output contains an explicit
reasoning-end token sequence. If a malformed generation never ends reasoning, the
separate final counter cannot begin; the request's total `max_tokens` and the physical
context limit still bound it. A tool call is a separate structured phase, not the
visible final-response phase; Qwen's tool-start marker can end reasoning implicitly in
the parser without reclassifying tool arguments as final prose.

## Tool-result correlation is part of prompt correctness

The OpenAI and Anthropic protocols assign an ID to every tool call and return that ID
with the corresponding result. Qwen's rendered tool-history XML does not include
those transport IDs. It represents a group of calls followed by a group of results,
so association inside the model prompt is positional. This makes apparently harmless
API leniency unsafe: accepting two results in reverse order silently teaches the model
that each result belongs to the wrong call.

The reviewed request validator therefore requires all of the following before either
API renders the conversation:

- every assistant tool call has a nonempty, unique ID;
- its results form one immediately following, uninterrupted group;
- that group contains exactly one result per call, in declared call order; and
- no orphan, duplicate, unknown, missing, interrupted, or incomplete result exists.

There is intentionally no guessing, ID repair, positional reordering, or best-effort
fallback. OpenAI validation errors are HTTP 400. Anthropic request conversion uses the
same canonical validator and the narrow router patch maps its Pydantic validation
failure to HTTP 400 `invalid_request_error`; unrelated server exceptions remain HTTP
500. Live adversarial requests proved both orphan-result paths, while successful SSE
tests proved valid call/result/continuation chains through both protocols.

## Current-upstream and K8V4 implementation checks

The pinned vLLM commit was compared with remote `main` at
`d4801990a45792c7081652f8ebea4ee56ceb67f9` on 2026-08-15. Remote was ten commits
ahead. Its only intervening parser change added streaming reasoning-token counting;
it did not repair the implicit Qwen tool-trigger boundary, Qwen-only non-strict schema
contract, ordered result correlation, or Anthropic validation mapping. Keeping the
reviewed pin is therefore deliberate.

The checkpoint's static symmetric per-tensor FP8 `kv_cache_scheme` is calibration
metadata for the ordinary FP8 cache path, not pre-quantized KV data stored in the
weights. The selected TurboQuant attention implementation does not read the model
layers' `_k_scale`/`_v_scale`; its runtime store kernel quantizes keys to FP8 and
values to packed 4-bit with a per-vector FP16 scale and zero point. Consequently,
the explicit `--kv-cache-dtype turboquant_k8v4` argument defines this deployment's
live KV semantics. vLLM's generic `+1.17% PPL` source comment is not a quality result
for this Qwen3.8 checkpoint and is not treated as one.

## Separate budgets and the physical context window

Qwen3.8 recommends a reasoning ceiling of 262,144 and a final-response ceiling of
131,072 **within a one-million-token context deployment**. Those are phase ceilings,
not a promise that 393,216 generated tokens fit inside this workstation's native
262,144-token profile.

This server defaults both ceilings explicitly across Chat Completions, Anthropic
Messages, and Responses. Responses exposes validated vLLM extension fields for both
budgets; omitted values inherit the server lock and the final ceiling cannot be raised
or nulled out. Every request is also constrained
by:

```text
prompt tokens + all generated tokens <= 262,144
all generated tokens <= request max_tokens
reasoning tokens <= 262,144
visible final-response tokens <= 131,072
```

Consequently, the effective budget is the minimum of the applicable limits and the
remaining physical window. A client may lower the final-response ceiling but cannot
raise or null out the server's 131,072 ceiling. A live request with a five-token final
ceiling produced separated reasoning, exactly five final tokens according to the real
served tokenizer, `finish_reason="length"`, and
`stop_reason="final_response_token_budget"`.

## Bottom-line decisions

- No historical monkey patch was copied.
- The architecture sameness justified reusing vLLM's Qwen3.5 model implementation;
  it did not justify carrying old frontend workarounds.
- The best historical ideas retained were fail-closed verification, server-side
  explicit defaults, real-tokenizer testing, schema-constrained tool arguments, and
  measured cache capacity.
- The most important obsolete ideas were the old parser wrappers, output-field rename,
  repetition detector, utilization-slack patch, and historical multimodal monkey
  patches. Vision instead received a current-source implementation with a narrower
  lossless contract and direct runtime proof.
- Qwen3.8's stronger template/tool behavior and new official sampling/budget guidance
  make several Qwen3.6 complaints void, but they do not remove the need for narrow
  current-source fixes or live long-context/tool validation.
- The original `agent_service` now runs the pinned Qwen Code client with the exact
  vLLM tokenizer, full-quality chronological PNG tools, foreground-only subagents,
  xhigh thinking defaults, and live cache/lifecycle acceptance. It does
  not resurrect the historical Qwen3.6 launcher or create a second model mode.
