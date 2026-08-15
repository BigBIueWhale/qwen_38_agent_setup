# Original `agent_service` use case and current client audit

Audit date: 2026-08-15

This is an inspiration and compatibility record, not an instruction to install a
client on the host. The only supported runtime in this repository remains the one
text-only, loopback-only vLLM profile. Any future coding client and orchestration
service must be pinned, built, configured, and run in project-owned Docker images.

## Exact source pins inspected

| Project | Exact revision inspected | Date / version |
|---|---|---|
| Original `BigBIueWhale/agent_service` | `7c6f9ea66cb12678217f7427513234518323d13c` | 2026-05-03 |
| Current Qwen Code | `b965d5f8c24f48e65fb0b17c7d45f34ca4ce8f38` | 2026-08-14, `0.21.12` |
| Current OpenAI Codex | `00f6a8a60e5c5e93d185c7fe67fd596b7e62240f` | 2026-08-15 |
| Current Claude Code public repository | `0fa8c19d50f70f9f383fb6ff5ce5209575267d21` | 2026-08-14, changelog `2.1.233` |

The original repository is cloned at `/home/user/Desktop/agent_service` and was left
clean. The three current-client research clones were disposable source-audit inputs
under `/tmp`; they are not runtime dependencies. No host client, configuration,
credential, session, or history directory was read or modified.

## What remains valuable in `agent_service`

The original use case is broader than choosing a CLI. It provides a durable execution
envelope around an autonomous coding agent:

- a singleton, lifecycle-explicit session API with create, inspect, cancel, and
  delete operations;
- input validation and a per-session copied workspace, so the agent does not work
  directly in the caller's source tree;
- distinct staging, output, event, artifact, terminal-record, and final-result data;
- a no-GPU agent container on an internal isolated network with DNS and the default
  route blocked;
- a deliberately narrow two-hop `socat` bridge from that network to the model server
  bound on host loopback, without giving the agent general host-network access;
- a separate loopback-only, read-only ttyd observation path;
- cancellation, a wall-clock limit, explicit resource limits, ordered teardown,
  project/session labels, and orphan cleanup after an orchestrator crash;
- a structured JSONL event stream and preserved forensic bundle even when the agent
  process fails.

Those properties remain useful whether the inner client eventually is Qwen Code,
Claude Code, Codex, or another protocol-compatible client. `qwen serve`, ACP, a client
daemon, or a browser UI may replace parts of the inner control plane; none replaces
the outer copied-workspace boundary, network isolation, lifecycle, result bundling,
or recovery semantics.

## What must not be copied from the historical service

The original repository pins Qwen Code `0.15.6`, Qwen3.6-27B AWQ, port 8001, a
152,000-token vLLM window, 32,768 output tokens, temperature 0.6, historical
repetition/compression workarounds, and vision-capable request settings. Those are
incompatible with this repository's selected model, current client, measured memory
profile, and one-mode text-only policy.

Its local-development document installs Node/Qwen Code and edits `~/.qwen/settings.json`
and shell startup files. That directly violates this project's Docker-only client
boundary and must not be followed here. Its container initialization loop polls a
sentinel with `sleep`; readiness in this project is event-driven. A model-identity
probe that only warns and continues is also unacceptable: every mismatch must stop
before the agent runs.

The original service's examples encourage subagents. Subagents can still be useful
for genuinely independent work, but they create separate context and fragment a long
thread. They must not be the default operating pattern for this deployment. The
quality-first default is one continuous main thread, completed-thinking omission,
prefix reuse, and compaction only when the physical window actually requires it.

## Current client comparison

### Qwen Code `0.21.12`: best-aligned candidate, not yet an installed mode

Qwen Code is currently the most directly aligned client candidate because its generic
OpenAI provider uses Chat Completions, the exact protocol on which this deployment's
Qwen rendering, parser, strict tool grammar, stream reconstruction, and full-history
round trip were tested. It exposes explicit `samplingParams`, `extra_body`,
`thinkingMandatory`, `contextWindowSize`, modalities, timeouts, and retry policy.

Several current behaviors materially supersede the historical `0.15.6` complaints:

- provider entries are arrays of `ModelConfig`; the obsolete wrapped provider shape
  is not accepted;
- a complete history is retained rather than dropping an arbitrary tail;
- output capacity is dynamically clamped to the real context window and prompt
  count, with a safety margin, rather than estimated from characters;
- compaction has explicit warning/automatic/hard thresholds, a 20,000-token summary
  reserve, state validation, truncation detection, and a three-consecutive-failure
  circuit breaker instead of a permanent first-failure latch;
- current compaction restores recent file/image references, plan state, and active
  agent reminders;
- the client supports streaming headless output, ACP, `qwen serve`, SDK/daemon
  lifecycle surfaces, and session continuation.

Two defaults are dangerous for this exact local model if left implicit. First, the
current token-limit table classifies names matching `qwen3.<digit>` as a commercial
one-million-token input model with a 65,536-token output limit. Our physical/native
window is 262,144, so `contextWindowSize: 262144` is mandatory. Second, internal side
queries may request `includeThoughts: false`; on a generic non-DashScope vLLM route
that normally becomes `chat_template_kwargs.enable_thinking: false`, which this
server correctly rejects. `thinkingMandatory: true` makes Qwen Code strip every such
disable shape before transmission, allowing the server's mandatory-xhigh default to
remain authoritative.

The following is the reviewed provider fragment for a future pinned, Docker-contained
Qwen Code client. It is a design input, not a second launch mode and not permission to
write host settings. `baseUrl` must point to the narrow model proxy inside the sealed
agent network; it must never be changed to a wildcard listener.

```json
{
  "model": {
    "name": "qwen3.8-27b-nvfp4-k8v4"
  },
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  },
  "modelProviders": {
    "openai": [
      {
        "id": "qwen3.8-27b-nvfp4-k8v4",
        "name": "Local Qwen3.8-27B NVFP4 K8V4",
        "envKey": "QWEN38_LOCAL_API_KEY",
        "baseUrl": "http://MODEL_PROXY_INSIDE_AGENT_NETWORK:8000/v1",
        "capabilities": {
          "agent": true,
          "vision": false
        },
        "generationConfig": {
          "timeout": 7200000,
          "maxRetries": 0,
          "thinkingMandatory": true,
          "contextWindowSize": 262144,
          "modalities": {
            "image": false,
            "pdf": false,
            "audio": false,
            "video": false
          },
          "samplingParams": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": 262144
          },
          "extra_body": {
            "reasoning_effort": "xhigh",
            "thinking_token_budget": 262144,
            "final_response_token_budget": 131072,
            "chat_template_kwargs": {
              "enable_thinking": true,
              "preserve_thinking": false,
              "reasoning_effort": "xhigh"
            }
          }
        }
      }
    ]
  }
}
```

The future client container must additionally pin these deployment environment
variables; current Qwen Code explicitly documents that they are not settings keys:

```text
QWEN_STREAM_IDLE_TIMEOUT_MS=600000
QWEN_STREAM_MAX_LIFETIME_MS=7200000
```

The first allows ten minutes of silence during a very large prefill or reasoning
phase, while the second makes the same two-hour outer request bound apply even to a
stream that keeps producing small chunks. Their stock values are four and fifteen
minutes respectively, which are too short for this profile's maximum-generation
contract. Neither guard is disabled, and the outer session manager still has the
independent authority to cancel the run sooner.

`max_tokens` is the total-generation ceiling, not the final-response phase ceiling.
Setting it to 262,144 allows Qwen Code to clamp it to its prompt-dependent safe
remainder (using the real/API prompt count and its explicit margin), after which vLLM
enforces the exact physical remainder; setting it to 131,072 would unnecessarily cut
possible reasoning in half on a short prompt. The independent server-enforced
`final_response_token_budget` still caps visible final output at 131,072. On the
native 262,144-token profile the upstream one-million-context recommendation cannot
simultaneously spend 262,144 reasoning tokens and 131,072 final tokens; prompt,
reasoning, tools, and final output always share the one physical window.

`maxRetries: 0` is deliberate for the defensive profile. It makes a transport error
observable instead of silently replaying an ambiguous long generation. A future
orchestrator may implement an explicit, state-aware retry only after proving that it
cannot duplicate an already executed tool result. The long timeout is not permission
to run forever: the outer session manager retains cancellation and a separate
wall-clock deadline.

This fragment still needs an end-to-end run from a pinned Qwen Code Docker image
through the future isolated proxy before it becomes supported. In particular, the
test must record the exact outgoing bodies for main and side queries, verify real API
usage counts drive compaction, and prove that no internal path disables thinking,
enables vision, changes sampling, retries silently, or selects another model.

### OpenAI Codex: Responses protocol is promising, client behavior is not proven

Current Codex has removed `wire_api = "chat"`; custom providers must use the Responses
API. It builds streamed `/v1/responses` requests with developer instructions, typed
tools, `tool_choice: auto`, explicit full history, `parallel_tool_calls`, reasoning
effort, `store: false`, a stable `prompt_cache_key`, and extra client metadata. A
custom provider can explicitly set a base URL, environment key, retry counts, and
stream idle timeout, while top-level configuration can set
`model_context_window = 262144` and an explicit auto-compaction limit.

The live project-owned Responses probe passed non-streaming and SSE tool selection,
strict typed arguments, `function_call_output` continuation, xhigh reasoning,
completed response events, and semantic transport equivalence. vLLM's request model
accepts Codex's extra metadata and `reasoning.encrypted_content` include request.
The v6 request model also applies the pinned `thinking_token_budget=262144` and
`final_response_token_budget=131072` defaults to Responses requests. It accepts
explicit smaller vLLM-extension values while retaining the server's final ceiling as
a hard upper bound.

That does **not** establish complete Codex compatibility. Current Codex's normal
request struct does not send `max_output_tokens` and cannot lower the Qwen-specific
phase budgets per request. The server now supplies and enforces both defaults, so
omission no longer bypasses them; however, Codex may still ask for reasoning-summary/
encrypted-content semantics designed for OpenAI models and has not been run
end-to-end against this Qwen model's actual system prompt and complete tool palette.
The server defaults prevent a silent sampling or phase-budget downgrade, but a
branded client working against protocol probes is not enough for a correctness claim.
Codex remains a candidate for a future pinned Docker experiment, not the current
recommendation.

### Claude Code: strong agent UX, weakest auditable model fit

The current public material documents custom gateways/models, xhigh/max effort,
streaming headless output, session resume, and prompt caching. The live native
Anthropic Messages endpoint in this project passes typed thinking/tool calls and
maps `high`, `xhigh`, and `max` to the mandatory Qwen xhigh policy.

Claude Code itself is proprietary. Its complete model assumptions, system prompt,
tool schemas, compaction behavior for an unknown custom Qwen model, and request
rewriting cannot be audited from the public repository. A passing Messages endpoint
probe is not an end-to-end Claude Code result. It must not be tested by touching the
user's host installation. If evaluated later, it must run from a pinned disposable
container with nonessential traffic disabled and all egress blocked except the narrow
model proxy.

## Design decision for the original use case

Do not couple the outer service to a particular CLI. Preserve `agent_service`'s
session/workspace/network/result architecture, but define a small client-adapter
boundary whose output is a normalized event stream and terminal result. The first
implementation candidate should be pinned Qwen Code `0.21.12` because it is the
model-native, Chat-protocol-aligned option and exposes all required Qwen controls.
Codex and Claude Code should remain isolated experimental adapters until each passes
the same full contract, not marketing-based substitutes.

The acceptance contract for any adapter is:

1. exact client source/package/image identity and configuration are pinned;
2. the client container has no GPU, no Internet/DNS/default route, and reaches only
   the narrow model proxy; every published observer/API socket is loopback-only;
3. model identity, 262,144 context, text-only modalities, xhigh thinking,
   `preserve_thinking=false`, all sampling parameters, both phase ceilings, and zero
   silent retries/fallbacks are proven from captured outgoing requests;
4. streamed and non-streamed tool turns have the same typed semantics, and complete
   tool-call/result history round-trips to identical model token IDs; partial deltas
   remain non-executable until the protocol's overall successful terminal, while
   `length`, `max_tokens`, `response.incomplete`, missing terminals, and malformed
   output fail closed without executing a buffered call;
5. API usage counts use the real tokenizer and govern output clamping/compaction;
6. a timed multi-turn run demonstrates actual prefix-cache hits and lower TTFT,
   instead of merely checking that prefix caching was enabled;
7. cancellation, timeout, process failure, malformed events, and model/protocol
   mismatches all produce explicit terminal records and preserve forensics;
8. the default is one long main thread. Compaction is delayed until the measured
   window requires it, completed hidden thinking is omitted to preserve useful
   context, and subagents are opt-in for genuinely independent work.

No current client has yet passed this entire outer-service contract. That is the
remaining integration boundary, not a defect in the validated vLLM serving profile.
