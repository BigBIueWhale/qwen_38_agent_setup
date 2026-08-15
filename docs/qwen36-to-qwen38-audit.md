# Qwen3.6 historical setup → Qwen3.8 current deployment audit

Date of audit: 2026-08-15

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
  `a767244d27bd76589a3e3b2ab4e64032c4ebc7af`.
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
- The selected Qwen3.8 template defaults to preserving old hidden reasoning traces.
  This project deliberately changes the default to `preserve_thinking=false` to
  conserve the physical context window. An explicit `true` restores the traces.
- Omission applies to hidden traces from completed historical turns. Reasoning within
  the current user/tool chain remains available to the model, and visible assistant
  answers, tool calls, and tool results remain in history. This is context hygiene,
  not conversation compaction and not deletion of the agent's observable work.
- The Qwen3.8 template has stronger historical-tool-call validation: missing names or
  malformed argument types fail loudly. The project retains those checks.

Live policy evidence from the final image:

- default omission rendered 68 tokens versus 1,866 with explicit preservation for
  the same synthetic history: 1,798 real tokenizer tokens saved;
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
parser model. Current parser tests, including Qwen boundary/truncation cases, passed.

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
Use `reasoning` there. The default `preserve_thinking=false` is unaffected because it
intentionally omits old hidden traces.

### 5. `monkey_patch_tool_call_in_think_detector.py` — tool call before `</think>`

Historical idea: detect a Qwen tool call emitted while the old reasoning parser still
believes it is inside thinking.

Disposition: **superseded by the unified Qwen state machine**. Current
`vllm/parser/qwen3.py` explicitly defines `(REASONING, TOOL_START)` as an implicit
reasoning-end transition. Both streaming and non-streaming behavior share the Parser
Engine. The live OpenAI and Anthropic trials exercised model-generated tool calls
with reasoning enabled and passed 3/3 per protocol.

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
marks the tool `strict:true`. Claude-like clients may omit that flag. The reviewed
current-source patch in `vllm/tool_parsers/structural_tag_registry.py` forces the
schema for `auto` tool choice while leaving the decision *whether* to call a tool to
the model. `tool_choice=none` remains unconstrained. Runtime environment variable
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

Disposition: **known current caveat but intentionally unreachable**. Current chat
ingress still has media-flattening behavior worth fixing for a vision agent. The only
supported profile is truly text-only via `--language-model-only`; all multimodal
limits are zero, so carrying a complex media monkey patch would add untestable attack
surface with no reachable feature. Enabling vision requires a new versioned profile
and a fresh audit of this item.

### 11. `monkey_patch_mm_cache_validator_eviction.py` — multimodal cache drift

Historical idea: if validation rejects a request after sender-cache mutation, restore
the sender/receiver cache invariant.

Disposition: **fixed upstream and unreachable here**. Current vLLM includes the P0/P1
multimodal processor-cache recovery change (`3962042304`). In addition, the text-only
profile does not create a usable multimodal request path. No patch is installed.

### 12. `monkey_patch_qwen3_coder_streaming_truncation.py` — dropped partial args

Historical idea: preserve cumulative partial tool arguments and make truncation
observable in streaming output.

Disposition: **do not carry the old parser wrapper**. Current vLLM's unified streaming
Parser Engine replaced the target/state layout, has dedicated Qwen delta-boundary
tests (`a0df04e477`), and shares transition semantics with non-streaming parsing. The
current parser-focused Docker test runs completed with 98 parser tests plus 124
targeted tool tests passing, including streaming boundary coverage. The project-owned
live probes separately exercised non-streaming OpenAI and Anthropic tool-call and
continuation invariants.

### 13. `launch_with_patches.py` and `sitecustomize.py` — propagation machinery

Historical idea: runtime patches must either appear in every spawned interpreter or
cause startup to fail.

Disposition: **retain the fail-closed objective, reject runtime monkey-patching**. The
current changes are baked into an immutable Docker image. The build verifies upstream
file hashes, patch hashes, live diffs, installed file hashes, and functional tests.
Startup then re-hashes all ten installed vLLM files, the template, and phase-budget
test. There is no launcher/sitecustomize registry that can drift between processes.

## Current-source changes that actually remain

Only four reviewed diff artifacts are applied to ten source files:

| Diff | Purpose | SHA-256 |
|---|---|---|
| `patches/vllm-turboquant-k8v4-direct-workspace.patch` | remove unsafe duplicate K8V4 continuation workspace allocations | `a9721067f1a7ee9497a4bd51e47e3a474561189e881b4704bfc4beac8ea48380` |
| `patches/vllm-enforce-auto-tool-schema.patch` | schema-constrain auto tool arguments even when clients omit `strict:true` | `eb5141db2aa702c9cc7dbcf1c8116e4dff37e832d1e5ce6a0ded3f38d66f4510` |
| `patches/vllm-qwen38-agent-defaults-and-thinking.patch` | explicit model defaults, Anthropic thinking controls, phase-ceiling request plumbing | `e1a31e2408603ebc174fc59e6a1e1e6dbfc5abf1d92221c8d976aa361f7d2423` |
| `patches/vllm-qwen38-separate-final-response-budget.patch` | count final tokens only after explicit reasoning end and enforce a separate hard ceiling | `f20d7dff41931248272842ed2c7a163c6f013e405ccf35733c40ff131a2fc503` |

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

## Separate budgets and the physical context window

Qwen3.8 recommends a reasoning ceiling of 262,144 and a final-response ceiling of
131,072 **within a one-million-token context deployment**. Those are phase ceilings,
not a promise that 393,216 generated tokens fit inside this workstation's native
262,144-token profile.

This server defaults both ceilings explicitly, but every request is also constrained
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
  repetition detector, utilization-slack patch, and multimodal patches in a text-only
  service.
- Qwen3.8's stronger template/tool behavior and new official sampling/budget guidance
  make several Qwen3.6 complaints void, but they do not remove the need for narrow
  current-source fixes or live long-context/tool validation.
