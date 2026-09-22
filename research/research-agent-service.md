# agent_service + qwen_38_agent_setup: how the two repositories fit together

Research report. Read-only analysis of two local clones; nothing was built or run.

**Path abbreviations used in citations**
- `A/` = `/home/user/bigbiuewhale/agent_service` (single commit `f3bd4cb`, 2026-09-22, "Pin the release image archive")
- `B/` = `/home/user/qwen_38_agent_setup` (HEAD `fb8d39c`, 2026-09-22, "Adopt the v26 runtime image and archive identities")
- `QCP` = `A/patches/qwen-code-0.21.12-agent-service.patch` (the 10.9 MB Qwen Code review diff). `SPP` = `B/patches/vllm-shared-prefix-cache-and-user-capacity.patch`. Line numbers for patches are line numbers inside the patch file.

**Evidence labels**
- **MEASURED**: the repos record a live run on hardware (usually with a session ID, date or release).
- **SOURCE-TESTED**: the repos record passing CPU/unit tests or build-time checks, but no live GPU run.
- **DESIGN / PLANNED**: the docs describe intent, obligations or open work.
- **INFERENCE**: my own reading of the code, not stated in the repos.

**Identity note.** On the author's machine the repos live at `/home/user/Desktop/agent_service` and `/home/user/Desktop/Qwen_best_model_ever`. This clone of the backend (`B/`) is `Qwen_best_model_ever`: its container label is `CONTAINER_LABEL="Qwen_best_model_ever"` (`B/config/runtime-v1.sh:7`). agent_service names it as `backend.project_dir` (`A/config/stack.lock.json:128`) and links it as `../Qwen_best_model_ever/README.md` (`A/README.md:89`). The two repos are coupled in several places:
- the backend's central model socket lives under agent_service's `.runtime` directory (`B/config/runtime-v1.sh:17-18`);
- the backend uses the fixed-relay image that agent_service builds (`B/config/runtime-v1.sh:10-13`; `A/config/release.lock.json`, `images.relay`);
- agent_service's `start.sh` launches the backend's `start.sh` when the backend is absent (`A/start.sh:169-175`).

---

## 1. Topology

### 1.1 Containers and processes

All containers use `--cap-drop ALL`, `--security-opt no-new-privileges:true`, `--restart no` and `--read-only`. None uses Docker port publication (`A/README.md:833-836`, `B/README.md:150-151`).

| # | Container name | Owner / launcher | Image | Network | User | Main mounts and limits | Role |
|---|---|---|---|---|---|---|---|
| 1 | `qwen38-agent-native` | B `scripts/run-agent.sh:262-285` | `qwen38-vllm:…-runtime-v26` = `sha256:dcc869c0…` (`B/config/runtime-v1.sh:8-9`) | `--network none` | `2000:0` | `--gpus all`; `/model` ro; cache volume `qwen38-vllm-cache-socket-isolated-nonroot-vision-agent-v21` → `/home/vllm/.cache/vllm`; tmpfs `/tmp` 2g exec and `/run` 64m noexec; `--shm-size 8g` | vLLM, listening on 127.0.0.1:8000 in its own network namespace |
| 2 | `qwen38-model-bridge` | B `run-agent.sh:379-399` | fixed relay `sha256:568e6bc1…` | `container:<vLLM id>` | 1000:1000 | host `…/agent_service/.runtime/model-socket` → `/sock` (rw); 32m memory; 32 PIDs | role `model-bridge`: Unix `/sock/relay.sock` → TCP 127.0.0.1:8000 inside the vLLM namespace |
| 3 | `qwen38-model-ingress` | B `run-agent.sh:401-420` | fixed relay | `--network host` | 1000:1000 | model-socket dir ro | role `model-ingress`: binds **host** 127.0.0.1:8000 → `/sock/relay.sock` |
| 4 | `qwen38-docker-broker` | A `start.sh:180-201` | broker `sha256:983e5e01…` (FROM scratch, static docker CLI 29.7.2) | `--network none` | 1000:984 | `/var/run/docker.sock` (bind, readonly); `.runtime/control` rw; state dir ro; 128m; 64 PIDs | the only holder of the Docker socket. Typed operations: `preflight`, `sweep_orphans`, `create_session`, `prove_session_quiescent`, `session_logs`, `wait_session`, `wait_agent_ready`, `wait_capture_complete`, `stop_session`, `remove_session` (`A/src/bin/docker_broker.rs:513-523`) |
| 5 | `qwen38-agent-service` | A `start.sh:203-227` | service `sha256:ebc43aff…` | `--network none` | 1000:1000 | `config/` ro; backend `manifests/` ro; `state` rw; `results` rw; `control` ro; model-socket dir ro; tmpfs `/tmp` 256m noexec; 2g; 512 PIDs | Rust/axum HTTP API on 127.0.0.1:8090 inside its own namespace. No Docker client and no Docker socket (`A/README.md:869-876`) |
| 6 | `qwen38-service-bridge` | A `start.sh:229-249` | fixed relay | `container:<service id>` | 1000:1000 | `.runtime/service-socket` → `/sock` | role `service-bridge`: Unix → TCP 127.0.0.1:8090 in the service namespace |
| 7 | `qwen38-service-ingress` | A `start.sh:251-270` | fixed relay | `--network host` | 1000:1000 | service-socket dir ro | role `service-ingress`: binds **host** 127.0.0.1:8090 |
| 8 | `agent-<session-id>` (one per session) | broker `create_session` (`docker_broker.rs:1069-1111`, name at `:2071-2073`) | agent `sha256:c371a572…` | `--network none` | 1000:1000 | staged → `/workspace` rw; artifacts → `/artifacts` rw; control → `/run/agent` ro; streams → `/streams` ro; tmpfs `/tmp` 8g (exec) and `/qwen-runtime` 2g noexec; 32g memory; 4096 PIDs | Qwen Code 0.21.12 (patched). The entrypoint chain is `run_agent.sh` → `agent_exec` (Landlock) → `node /opt/qwen-code/dist/cli.js` |
| 9 | `agent-capture-<session-id>` | broker (`:1122-1177`) | capture `sha256:1658271f…` | `container:<agent id>` | 1000:1000 | streams rw; output rw; 32m | the only owner of `/output`. Receives stdout and stderr over `/streams/events.sock` and `/streams/stderr.sock` and writes `/output/events.jsonl` and `/output/qwen.stderr` (`A/src/bin/session_capture.rs:1-20`) |
| 10 | `agent-model-<session-id>` | broker (`:1178-1234`) | fixed relay, role `agent-model` | `container:<agent id>` | 1000:1000 | central model-socket dir → `/sock` ro; 32m | binds 127.0.0.1:18000 inside the agent's namespace → central Unix socket |

Resource figures come from `A/config/stack.lock.json:36-125` and `A/config/broker-policy-v1.json`.

### 1.2 Listeners and sockets

| Endpoint | Where it lives | Bound by | Evidence |
|---|---|---|---|
| TCP 127.0.0.1:8000 | vLLM's private network namespace | vLLM (`--host 127.0.0.1 --port 8000`) | `B/config/runtime-v1.sh:223-225,261-265` |
| TCP 127.0.0.1:8000 | **host** | model-ingress relay | `A/src/bin/fixed_relay.rs:75-79` |
| Unix `…/.runtime/model-socket/relay.sock` (1000:1000, mode 0660) | host directory | model-bridge | `fixed_relay.rs:15,164-181`; `B/scripts/run-agent.sh:393-399` |
| TCP 127.0.0.1:8090 | service's private namespace | agent_service (`SERVICE_READY … listen=127.0.0.1:8090 network=none`) | `A/start.sh:226-227` |
| TCP 127.0.0.1:8090 | **host** | service-ingress relay | `fixed_relay.rs:85-89` |
| Unix `…/.runtime/service-socket/relay.sock` | host directory | service-bridge | `A/start.sh:244-249` |
| Unix `…/.runtime/control/broker.sock` (1000:984, mode 0660) | host directory | broker | `A/config/stack.lock.json:55`; `A/start.sh:199-201` |
| TCP 127.0.0.1:18000 | each agent's private namespace | session relay (`agent-model`) | `fixed_relay.rs:90-94`; `docker_broker.rs:1238-1246` |
| Unix `<state>/sessions/<id>/streams/{events,stderr}.sock` | per session | capture sidecar | `A/src/staging.rs:15-17` |
| `<state>/sessions/<id>/control/start-gate.lock` (flock) | per session | service holds the lock; the agent wrapper blocks on it | `A/src/staging.rs:213-234`; `A/docker/config/run_agent.sh:211-216` |

The only host TCP listeners are 127.0.0.1:8000 and 127.0.0.1:8090 (`A/README.md:833-836`). Each is owned by a host-network relay whose compiled role fixes its address; a relay accepts exactly one role argument and nothing else (`fixed_relay.rs:1-5,103-123`).

### 1.3 The model path from the agent

```
Qwen Code (agent netns) --HTTP--> 127.0.0.1:18000   [agent-model relay, same netns]
   --> Unix /sock/relay.sock = host .runtime/model-socket/relay.sock   [bound by model-bridge]
   --> model-bridge, which runs in the vLLM netns --> TCP 127.0.0.1:8000 --> vLLM
```

- **The agent never touches host 127.0.0.1:8000 or the model-ingress.** The ingress exists only for host-side callers such as `status.sh`/`start.sh` health checks (`B/scripts/run-agent.sh:422`).
- The probes do not use the ingress either. They run inside the vLLM container via `docker exec` (`B/scripts/run-probe.sh`).
- The agent_service container also reaches vLLM directly over the central Unix socket, using `curl --unix-socket`. It does this only for startup preflight (`/version`, `/v1/models`, `/tokenize`) (`A/src/api.rs:1984-2059`).
- The diagram itself is in `A/README.md:838-858`.

**Every relay is a protocol-blind byte pipe.** `relay_unix_to_tcp` and `relay_tcp_to_unix` simply call `tokio::io::copy_bidirectional` (`fixed_relay.rs:528-564`). No relay parses HTTP, injects headers, or adds identity. Agent identity (`kv_scope`) therefore has to come from the client (§2).

### 1.4 What "socket-isolated-nonroot" means

The profile name `socket-isolated-nonroot-vision-k8v4-agent-v21` appears at `B/config/runtime-v1.sh:4`, and the image profile is `-v26` (`:5`). The repos never define the term. Its meaning follows from `B/README.md:142-161` (INFERENCE):
- **Socket-isolated:** vLLM runs under `--network none`, binds only its private loopback, and is reachable only through one project Unix socket. A minimal bridge that shares its namespace owns that socket. The only host-network component is a fixed ingress bound to host 127.0.0.1:8000.
- **Non-root:** the container runs as `2000:0` with cap-drop ALL, no-new-privileges, a read-only root, a read-only model mount, and one labelled cache volume owned `2000:0` mode 0770. `/tmp` is a 2 GiB exec tmpfs and `/run` a 64 MiB noexec tmpfs.
- `status.sh` validates all of this (`B/README.md:150-151,159-161`).

### 1.5 The protocol the agent CLI speaks to vLLM

**It is OpenAI Chat Completions, streamed, plus vLLM's non-OpenAI `/tokenize`.** Anthropic Messages and OpenAI Responses are served and probed on the backend, but the agent does not use them.

| Evidence | Location |
|---|---|
| Sealed provider is `modelProviders.openai[...]` with `baseUrl: http://127.0.0.1:18000/v1`, `auth.selectedType/enforcedType: openai`, `envKey QWEN38_LOCAL_API_KEY=local-loopback-only`, `exactTokenCounting: "vllm"`, `maxRetries: 0`, `timeout: 86400000` | `A/docker/config/settings.json:25-33,78-100` |
| The build-time fixture only accepts `POST /v1/chat/completions` with `stream: true`, `Authorization: Bearer <credential>`, and a nonempty `kv_scope` string. `POST /tokenize` takes `messages` for exact sizing or `prompt` for preflight; `GET /v1/models` must come first | `A/docker/tests/check_composition.py:103,128-145,729` |
| The headless-CLI smoke asserts that generations go to `/v1/chat/completions` and carry `kv_scope == session` | `A/docker/scripts/check_headless_cli.py:76-79,187-190` |
| Qwen Code's pipeline builds `OpenAI.Chat.ChatCompletionCreateParams`, then writes `parallel_tool_calls=false` and `kv_scope` last | QCP:169329-169331 |
| Wrapper preflight: `GET http://127.0.0.1:18000/v1/models` (requires one model, `max_model_len == 262144`) and `POST http://127.0.0.1:18000/tokenize` | `A/docker/config/run_agent.sh:218-236` |
| README: "The live model protocol is OpenAI Chat Completions with vLLM's `qwen3_coder` tool parser and `qwen3` reasoning parser." Responses coverage is "for future Codex compatibility" | `A/README.md:634-647` |
| The client was chosen because "its generic OpenAI provider uses Chat Completions" | `B/docs/agent-service-and-client-audit.md:77-80` |
| "Codex Responses and Anthropic Messages protocol surfaces are proven on the server, but neither creates another supported client mode" | `B/README.md:1229-1232` |

### 1.6 Sandboxing: Landlock, seccomp, AppArmor and no-network

**Agent process (`agent_exec`)**
- Sandbox ID: `landlock-fs-v4-write-roots-v1+private-devpts-rw-v1+output-unmounted-v1` (`A/src/bin/agent_exec.rs:20`).
- Requires Landlock ABI ≥4 (`:402-412`).
- Write access is limited to `/workspace`, `/artifacts`, `/tmp` and `/qwen-runtime`, plus `/dev/null` (write/truncate) and `/dev/pts` (WRITE_FILE only, for the PTY shell) (`:377-391`).
- Refuses to run if `/output` exists (`:147-149`).
- Closes every descriptor ≥3 before `exec` (`:186-192`).
- Fixed argv: `--input-format=text --approval-mode=yolo --output-format=stream-json --strict-tools=<10 tools> --foreground-agents-only --max-subagent-depth=1 --max-tool-calls=-1 --max-session-turns=N` (`:94-102,194-204`).

**Wrapper (`run_agent.sh`)**
- Asserts uid:gid 1000:1000 (`:117-118`).
- Asserts the exact private devpts/ptmx contract (`:119-124`).
- Asserts no-network: the only interface is `lo`, the only IPv4 address is 127.0.0.1/8, there are no IPv6 addresses, and both IPv4 and IPv6 route tables are empty (`:171-193`).
- Runs the runtime-contract and toolchain verifiers (`:157-169`).
- Requires the exact attestation string within 15 s (`:253-262`).
- Sets `umask 077` (`:103`).

**Relays**
- Sandbox ID `landlock-net-v4+seccomp-socket-v2` (`fixed_relay.rs:16`).
- A Landlock network rule allows exactly one TCP port and one direction (`:293-385`).
- A stacked seccomp filter allows only AF_UNIX or AF_INET SOCK_STREAM sockets; TCP-listening roles get Unix only after binding (`:405-450`).
- A second seccomp filter forbids any further `bind` (`:452-467`).
- The relay self-proves the sandbox with negative probes before announcing readiness: arbitrary connect, arbitrary bind and UDP must all be denied (`:493-526`).

**Host requirements**
- Docker security options must include `apparmor`, `seccomp,profile=builtin` and `cgroupns`; the container AppArmor profile must be `docker-default` (`A/config/stack.lock.json:280-289`; `B/config/runtime-v1.sh:219-220`).
- One GPU with ≥32,607 MiB (`B/config/runtime-v1.sh:221`; `B/README.md:1236-1242`).
- Host software versions are deliberately **not** pinned (`B/README.md:1236-1242`).

---

## 2. The 35 vLLM patches

vLLM is pinned at `9df9b0b0a1816b6d0d0f6ecd0da563cc37fd72f5` and rebuilt through 35 ordered, landmark-aware transformations (`B/README.md:260-302`). The patch files are plain unified diffs with no prose header. The one-line purposes below come from each stage's `rationale` in `B/patches/source_patch_v1/contracts_vllm.py:2621-3155` (the line given is the contract key). "Ord." is the application order in the `B/README.md:268-302` table.

### 2.1 Grouped table

**Tool-call grammar and parser fidelity (12)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 2 | `vllm-enforce-auto-tool-schema.patch` | 1,736 | With automatic choice, every call the model begins is constrained to its declared schema; OpenAI `strict` and XGrammar's `strict=false` fallback can no longer loosen a Qwen schema | 2955 |
| 5 | `vllm-qwen-implicit-tool-grammar-boundary.patch` | 4,230 | Keep the implicit `<tool_call>` token that ends reasoning, so the decoder grammar is not one token behind and the call stays constrained | 3000 |
| 7 | `vllm-tool-truncation-finish-reason.patch` | 14,841 | A partial call at max_tokens is never promoted to an executable call; Chat keeps `length`, Responses stays `incomplete` | 3030 |
| 15 | `vllm-qwen-exact-tool-language.patch` | 71,992 | Match the exact trigger; no calls invented from unarmed prose; keep content order and wrapper closure; batch and stream parity | 2909 |
| 18 | `vllm-qwen-single-call-grammar.patch` | 17,000 | The grammar, not a response filter, owns the call count; no produced call is silently deleted | 2870 |
| 26 | `vllm-xml-text-fidelity.patch` | 10,868 | Whitespace inside XML string values and the surrounding content is preserved byte-for-byte | 2756 |
| 27 | `vllm-phase-aware-parser-terminals.patch` | 31,908 | Reasoning boundaries use token-ID authority; the tool trigger is recognized as text after reasoning | 2741 |
| 29 | `vllm-tool-output-completion.patch` | 92,182 | Only a closed wrapper at model EOS is executable; caller stops are suspended inside armed calls | 2712 |
| 31 | `vllm-schema-faithful-xml.patch` | 33,584 | Decode XML arguments against the complete schema (refs, enum/const, composition) and validate the result | 2682 |
| 32 | `vllm-token-text-provenance.patch` | 22,785 | A stop-stripped boundary ID never binds to a prose lookalike | 2668 |
| 34 | `vllm-qwen-canonical-parameter-framing.patch` | 7,414 | Invert the transport's `\n` framing around parameter values exactly once | 2638 |
| 35 | `vllm-qwen-owned-tool-grammar.patch` | 31,772 | vLLM owns the Qwen structural tag so that no value can contain `<parameter=`; string parameters with `pattern/format/minLength/maxLength` are refused | 2622 |

**Thinking, budgets and sampling (6)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 3 | `vllm-qwen38-agent-defaults-and-thinking.patch` | 11,934 | Server defaults reach every protocol; only xhigh thinking is accepted (high/max are aliases); tool-result correlation is validated | 2970 |
| 4 | `vllm-qwen38-separate-final-response-budget.patch` | 8,490 | A `final_response_token_budget` counter that starts only after a real reasoning-end marker | 2985 |
| 13 | `vllm-exact-reasoning-usage.patch` | 34,714 | `completion_tokens_details.reasoning_tokens` counted exactly from the parser's token-ID split | 3138 |
| 22 | `vllm-generation-sampling-resolution.patch` | 42,229 | Resolve sampling settings after the prompt length and defaults are known, then build one fresh engine request | 2813 |
| 23 | `vllm-sampling-decoding-boundary.patch` | 32,255 | Beam search is refused at request validation (it bypassed budgets, grammar and `kv_scope`) | 2798 |
| 30 | `vllm-one-way-thinking-boundary.patch` | 19,007 | Thinking ends once; the V1 budget holder and the grammar share the parser's boundary | 2697 |

**KV cache, memory and agent identity (6)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 1 | `vllm-turboquant-k8v4-direct-workspace.patch` | 5,257 | The K8V4 continuation dequantizes straight into the reserved 1,024 MiB workspace (no duplicate K/V buffers) | 2939 |
| 10 | `vllm-turboquant-fail-closed-guards.patch` | 4,746 | Poison NaN metadata, refuse non-finite prefill, refuse SM < 8.9 | 3077 |
| 11 | `vllm-kv-offload-pinning-fail-closed.patch` | 1,691 | A `cudaHostRegister` failure now raises instead of warning | 3097 |
| 12 | `vllm-shared-prefix-cache-and-user-capacity.patch` | 417,363 | Required opaque `kv_scope`; shared-prefix membership catalog; `--kv-cache-users` and `cpu_kv_cache_users` sizing; whole-agent eviction | 3118 |
| 17 | `vllm-kv-physical-free-memory.patch` | 6,094 | The declared KV capacity cannot spend memory already occupied before profiling | 2883 |
| 28 | `vllm-input-stream-agent-identity.patch` | 7,901 | An input stream keeps one `kv_scope`; a change is refused before dispatch | 2727 |

**Vision (3)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 8 | `vllm-qwen38-vision-runtime.patch` | 65,969 | Chronological tool-result media; strict lossless static PNG; full pixel budget; BF16 tower; workspace released and restored around encoding | 3045 |
| 16 | `vllm-png-source-admission.patch` | 2,738 | Admit only encoded IHDR bit depth 8 with colour type 2 or 6, so 16-bit PNGs cannot slip through Pillow | 2896 |
| 25 | `vllm-raw-image-token-transport.patch` | 71,591 | `/render`→`/generate` carry the original PNGs; caller tensors, hashes and UUIDs are refused | 2769 |

**Protocol identity and fidelity (6)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 6 | `vllm-anthropic-validation-http400.patch` | 2,150 | Anthropic validation and engine-validation errors (for example a missing `kv_scope`) become 400 `invalid_request_error` | 3014 |
| 14 | `vllm-anthropic-input-fidelity.patch` | 40,295 | Refuse unrenderable tool-result items, render `is_error`, and use Anthropic error envelopes | 2924 |
| 19 | `vllm-responses-history-integrity.patch` | 23,420 | Responses replay keeps every text and reasoning block and runs the shared tool-ID correlation gate | 2857 |
| 20 | `vllm-responses-stream-identity.patch` | 24,633 | The terminal response reuses the streamed item and call IDs | 2844 |
| 21 | `vllm-anthropic-terminal-metadata.patch` | 16,615 | `stop_sequence`, `tool_use` and `max_tokens` derived from the observed cause on both transports | 2829 |
| 24 | `vllm-token-generation-result-integrity.patch` | 32,466 | `/generate` keeps every choice and the real terminal cause | 2784 |

**Error precision (1)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 33 | `vllm-precise-request-errors.patch` | 39,724 | Only a known request cause becomes 4xx; internal Python and template errors stay 5xx | 2654 |

**Numerical audits (1)**

| Ord. | Patch | Bytes | Purpose | Contract line |
|---|---|---|---|---|
| 9 | `vllm-qwen38-numerical-audits.patch` | 9,388 | A K8V4 oracle test at Qwen3.8's exact BF16 D=256 / Hq=24 / Hkv=4 geometry | 3062 |

Aggregate counts: 100 reviewed runtime-source changes, 2 new runtime sources, 7 runtime deletions, 71 test changes, 12 new tests and 3 test deletions (`B/README.md:304-306`). Five of the seven deletions are the old CPU-offload ARC/LRU policy package, which SPP deletes (SPP:8693-9071).

### 2.2 The agent identity: request field `kv_scope`

**Carrier**
- `kv_scope` is a **top-level JSON body field**, not an HTTP header.
- It is declared as `kv_scope: str | None = Field(default=None, min_length=1, pattern=r"\S", …)` on six request models: Anthropic Messages, Chat Completions, batch Chat, Completions, Responses, and token-in/token-out `GenerateRequest` (SPP:6486, 6669, 6699, 6741, 6774, 6809).
- The OpenAI surfaces copy it into `SamplingParams.extra_args["kv_scope"]`. Anthropic passes it through its Chat translation (SPP:6503-6512).
- `Request.kv_scope` is materialized from `extra_args` so the offload connector can read it (SPP:9617).
- Neither repo has any `x-kv*`, `x-agent*` or `x-user*` header (grep over both repos).

**Gate**
- `require_kv_scope()` in `vllm/v1/engine/input_processor.py` raises `VLLMValidationError(parameter="kv_scope")` when the field is absent, not a string, or whitespace-only (SPP:7861-7881).
- It is called for every `SamplingParams` request in `process_inputs`.
- Render-only and pooling requests need no ID (`B/docs/kv-user-count-and-agent-scope-design.md:32-35`).
- Each of the six generation routes (`/v1/chat/completions`, `/v1/chat/completions/batch`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/inference/v1/generate`) returns HTTP 400 naming the field when it is missing. On the OpenAI shapes this is `error.param`; on Anthropic it is an `invalid_request_error` (`B/README.md:1018-1026`).
- Cohere and `/generative_scoring` are deliberately unmounted because they cannot name an agent (SPP:6514-6621; `B/README.md:479-482`).
- The ID is opaque: whitespace-only is invalid, and any other string is kept byte-exact (for example `"  opaque:/agent  "`, a test case at `B/patches/vllm-input-stream-agent-identity.patch`).
- The ID provides "neither authentication nor a confidentiality boundary": a fresh ID can detect shared-prefix hits through latency. `cache_salt` is independent of it (`kv-user-count-and-agent-scope-design.md:182-189`; `B/README.md:490-494`).

### 2.3 Shared prefix cache and user capacity

**Membership catalog** (`PrefixCacheIndex`, new file `vllm/v1/core/prefix_cache.py`, SPP:7496-7752)
- Physical entries are keyed by content (chained block hash + KV group ID). Agent IDs never enter the hashes.
- Each entry records which agents hold references to it, either the full entry or a selected subset.
- `view(agent)` returns "permit everything" when the agent owns nothing. Otherwise it permits only content the agent has acquired or computed.

**Resulting rule** (`kv-user-count-and-agent-scope-design.md:10-30`)
- An ID with **no** cached blocks (GPU or CPU) may acquire any matching shared prefix. This is an implicit fork.
- Once it has blocks, it matches only its own acquired or computed data.
- Example from the doc: B can reuse A's prefix P but never A's continuation A1. Releasing A preserves B's references.
- GPU and CPU share one catalog. A block resident in either tier makes the agent "existing" (`:16-20`).

**Capacity is declared in user contexts, not bytes**
- `--kv-cache-users N` replaces `--kv-cache-memory-bytes`, `--kv-offloading-size` and `--kv-offloading-backend`, all of which are removed (SPP:5440-5485 and the `arg_utils.py` hunk).
- GPU sizing: `blocks_per_user = Σ_groups ceil(max_memory_usage_bytes / page_size_bytes)` and `pool_blocks = N·blocks_per_user + 1` (the +1 is the null block).
- If that does not fit the **physical** bound, meaning total VRAM minus every measured resident, startup refuses. `num_gpu_blocks_override` is refused alongside it (SPP `kv_cache_utils.py` hunk around line 7429; design doc 139-157).
- Host tier: `cpu_kv_cache_users: N` is required in `kv_connector_extra_config`, and unknown keys are refused. `chunks_per_user = Σ_groups (F_g for full attention, else min(F_g, window_chunks + eagle))` and `num_blocks = N·chunks_per_user` (SPP:9236-9301; design doc 159-180).

**Deployed values**
- `--kv-cache-users 1` and `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_kv_cache_users":1}}'` (`B/config/runtime-v1.sh:289,324-325`).
- This is the "one declared resident user context" in `B/README.md:40,452-459`.
- The config comment estimates about 29.6 GiB in use on a 31.8 GiB card: weights 21.3 GiB + activation peak 1.9 GiB + a 6.4 GiB one-context pool (`B/config/runtime-v1.sh:294-298`).

**Whole-agent context eviction in the host tier** (the CPU manager rewrite, SPP `vllm/v1/kv_offload/cpu/manager.py`)
- A `RetainedContext` holds a request's complete working set across all KV groups.
- Finishing a request keeps its context for the next turn, and a new turn replaces the agent's previous idle context (SPP:8303).
- Stores are planned before anything is evicted (SPP:8546-8570):
  1. reclaim unreferenced idle entries, including obsolete windows;
  2. if that is not enough, release **entire agents** in LRU order, excluding the incoming agent (`if victim == agent_id: continue`);
  3. if space still cannot be made, decline the store with no partial eviction.
- Shared references held by surviving contexts protect their entries.
- A failed transfer invalidates only the contexts that depended on it.
- The same rules are stated in prose at `kv-user-count-and-agent-scope-design.md:72-105`.
- On the GPU tier the catalog also caps agent *records* at the tier's entry capacity and releases the oldest complete agent record at that bound (`prefix_cache.py _make_room`; design doc 101-105). Physical GPU blocks are still recycled by the block pool.

**Why the host tier exists**
- "A foreground subagent's context competes for the one-context GPU pool of the session that launched it, so at long main contexts any subagent run evicts the main agent's blocks … losing the few GDN state blocks collapses the whole GPU prefix hit … Offloading keeps those blocks in DDR5 and restores them by DMA" (`B/config/runtime-v1.sh:303-309`).
- Pages move as opaque int8 bytes at 24,832 B/token, with no dequantization (`:311-314`).

### 2.4 Pinned host tier in /dev/shm and "KV offload pinning fail-closed"

- The host region is an mmap in the container's `/dev/shm`. The vLLM container runs with `--shm-size 8g` (`B/scripts/run-agent.sh:278`), and status checks that `ShmSize == 8589934592` (`B/scripts/runtime-common.sh:575-588`).
- It holds about 6.6 GB for one resident context, "pre-faulted and page-locked — hard, unswappable host memory" (`B/config/runtime-v1.sh:316-323`).
- Upstream only *warned* when `cudaHostRegister` failed and then carried on over unpinned DMA. The patch replaces that with `raise RuntimeError(…)` naming RLIMIT_MEMLOCK, host memory, or the region size as likely causes (`B/patches/vllm-kv-offload-pinning-fail-closed.patch:1-42`; rationale at contracts line 3097).
- At readiness, `assert_kv_offload_pinned` requires:
  - zero `cudaHostRegister failed` log lines;
  - zero `MADV_POPULATE_WRITE is not supported` fallbacks;
  - a positive `cpu_kv_cache_users` in the live argv;
  - exactly one `Created`/`Opened existing mmap file` line.
  (`B/scripts/runtime-common.sh:958-998`)
- Historical sizing: the older byte-denominated host tier was "the declared 7,747,584,000-byte ARC host tier" (`B/README.md:1069-1070`). Those byte constants "were hand-measured on this one GPU and silently wrong anywhere else" and have been replaced by user counts (`B/config/runtime-v1.sh:281-288`).

### 2.5 Input-stream agent identity

- `AsyncLLM` streaming input binds `agent_id = require_kv_scope(sampling_params)` before it reads any chunk.
- Each chunk's `SamplingParams` may change generation settings. If `require_kv_scope(sp) != agent_id`, it raises `VLLMValidationError(parameter="kv_scope")` *before dispatch*, and only that stream's request is aborted (`B/patches/vllm-input-stream-agent-identity.patch:139-161`).
- The tests cover both output kinds, first-chunk and later-chunk changes, and a parameter set of invalid IDs (`:223-275` of the patch).
- Rationale: "An input stream resumes one live request and its acquired KV … A different agent enters through normal generation admission" (contracts line 2727).

### 2.6 Responses history integrity

Before the patch, Responses replay dropped everything after the first text or reasoning block (`item.content[0].text`), skipped previous reasoning output, and bypassed Chat's tool-ID correlation check.

The patch (`B/patches/vllm-responses-history-integrity.patch`):
- moves the correlation check into `chat_utils.validate_tool_result_correlation`. Every call needs exactly one result, in declared order, before the next turn; orphaned, missing, duplicate, mismatched or out-of-order results raise `VLLMValidationError`;
- calls that check from both Chat and Responses;
- rebuilds previous-response output through the same converter;
- concatenates every text and reasoning block with no invented separators;
- refuses encrypted reasoning explicitly;
- deep-copies the supplied history instead of mutating it.

Validation: "91 utility/boundary tests pass in the offline CPU container" (`B/docs/model-output-and-audit-fixes.md:555-577`).

### 2.7 Anthropic input fidelity

- A tool result's content items must be `text`, `image` or `tool_reference`. Any other item, such as a `document` block, is **refused** rather than silently dropped.
- An image source must be a nonempty URL, or base64 with both `media_type` and `data`.
- `is_error: true` is rendered as a first line, `"Tool result flagged is_error: true."`.
- All errors are classified through `create_error_response` and mapped to Anthropic error types by HTTP status (400 → `invalid_request_error`, 500 → `api_error`, and so on).
- Mid-stream errors become Anthropic `error` events.

Sources: `B/patches/vllm-anthropic-input-fidelity.patch`, `vllm/entrypoints/anthropic/*` hunks; rationale at contracts line 2924. This closed Codex audit findings 3, 5 and 6 (`B/docs/qwen38-deployment-correctness-audit.md`, findings 3, 5, 6; `model-output-and-audit-fixes.md:425`).

### 2.8 How agent_service sets and uses the identity

- **Client side only.** Patched Qwen Code adds `GenerationContext(kvScope)`, which refuses an empty ID (QCP:157082-157161). A `GenerationClient` attaches it to every generate, stream and count request.
- The provider pipeline writes `typed['kv_scope'] = request.generationContext.kvScope` **after** provider and `extra_body` decoration, so settings cannot override it (QCP:169329-169331).
- **Scope assignment.**
  - The main session's scope is the conversation's session ID.
  - A tool-launched child (subagent) uses the ID of the call that spawned it.
  - Workflow and utility invocations get their own distinct IDs.
  - Batch requests, streams, JSON side-queries and exact counting all carry the scope.
  (`A/patches/README.md:36-42`; `A/README.md:455-466`)
- The build-time composition gate checks that `kv_scope` equals the `session_id` recorded in the captured stream (`A/docker/tests/check_composition.py:694-698`).
- INFERENCE: the Qwen Code session ID is minted inside the CLI. It is not the service's `s-<64hex>` handle, since `agent_exec` passes no session-id argument (`agent_exec.rs:94-102`).
- **What the service knows about KV offload.** The runtime contract declares `kv_offload: true`, `OffloadingConnector`, `kv_both`, `kv_offload_host_user_contexts: 1`, `kv_offload_scope_eviction: true` (`A/config/agent-runtime-contract-v1.json:66-70`).
- The sealed main-session prompt tells the model: "Delegate generously … to a foreground `general-purpose` or `Explore` subagent … The backend can retain your KV cache in DDR5 while the child runs, but cache eviction can require prefix processing again" (QCP:170267).
- The README states the consequence: "Returning to a parent can reuse backend state while its cache remains retained … eviction of its own cached context can require reprocessing" (`A/README.md:348-351`).
- agent_service sets no `cache_salt` (grep over A: none).

### 2.9 Status: proven or planned

- **SOURCE-TESTED.** Shared-prefix and identity rules pass "544 scheduler, cache, geometry, protocol and tiering tests … in offline CPU containers" (`B/docs/model-output-and-audit-fixes.md:604-637`). Input-stream identity passes 30 tests (`:661-678`).
- **BUILT.** The v26 runtime image carrying these patches is pinned and rebuilt reproducibly: "A second no-cache build of this tree printed 'Verified reproducible image ID'" (commit `fb8d39c` message; `B/README.md:351-355`).
- **LIVE GPU validation of `kv_scope` sharing and whole-agent eviction is not recorded.**
  - The README still says "The v23 source tests cover these rules; image adoption is pending" (`B/README.md:459`) and "The v23 source revision awaits its offline image build, adoption and release validation" (`:18-19`).
  - The design doc says "Source validation does not claim that an awaiting-adoption image is already serving these rules" (`kv-user-count-and-agent-scope-design.md:200-203`).
- **The live prefix-cache measurements predate `kv_scope`.** They come from the v13 backend and the v8 agent acceptance (§5).

---

## 3. The agent_service request lifecycle

### 3.1 HTTP API

The API listens on 127.0.0.1:8090 only (`A/README.md:1195-1207`; routes at `A/src/api.rs:69-91`).

| Method and path | Behaviour |
|---|---|
| `GET /healthz` | Returns `ok` |
| `POST /v1/agent/sessions` | Multipart upload. 202 for a new durable acceptance; 200 for an idempotent replay; 409 for a conflicting replay |
| `GET /v1/agent/sessions` | Returns `{"sessions":[…]}` |
| `GET /v1/agent/sessions/{id}` | Pure read of state, progress and terminal fields |
| `GET /v1/agent/sessions/{id}/bundle` | Streams `bundle.tar.zst` with `Content-Length` and `X-Bundle-SHA256`. 409 while running; 404 when no bundle was accepted (`api.rs:441-504`) |
| `POST /v1/agent/sessions/{id}/cancel` | 202 while running (durable cancel intent recorded); 200 if already terminal; 409 `session_finalizing` if terminal selection already won (`A/src/runtime.rs:1789-1854`) |
| `DELETE /v1/agent/sessions/{id}` | 204 on success; 409 if running ("cancel first"); 404 if unknown (`runtime.rs:1863-1920`) |

Status mapping is at `A/src/error.rs:99-121`. The service has no authentication. The repos describe none, and the API relies on the loopback-only bind (INFERENCE).

### 3.2 Creating a session

**Body.** Exactly two ordered multipart parts (`A/README.md:1220-1230`; `api.rs:122-237`):
1. `request` (`application/json`, ≤2 MiB = `request_body_limit_bytes` 2,097,152) containing `{"prompt", "max_session_turns"?, "archive_bytes", "archive_sha256"}`, with `deny_unknown_fields`;
2. `archive` (`application/zip`).

Any extra part is refused.

**Checks before the archive is spooled**
- The archive commitment must be 64 lowercase hex characters, and `0 < bytes ≤ MAX_ARCHIVE_BYTES` = 200 GiB + 64 MiB = 214,815,473,664 (`A/src/validation.rs:167-189`; `A/src/config.rs:52-63`).
- `max_session_turns` must be a JSON integer in 1..2000; omitted means 400. Zero, negative and non-integer values are refused by name and never clamped (`validation.rs:97-160`; `config.rs:91-111`).

**Prompt** (`validation.rs:56-95`; `config.rs:26-44`)
- Non-empty after trim, no NUL.
- Must already be in **NFC**; a non-NFC prompt is refused with an instruction to normalize it.
- At most **32,768 bytes** (M = W/8).
- Longer material belongs in the workspace. The prompt reaches Qwen through stdin, not argv (`A/README.md:1243-1253`).

**Idempotency**
- The `Idempotency-Key` header is required and is itself the **session ID**: `s-` followed by exactly 64 lowercase hex characters from 32 CSPRNG bytes (`api.rs:540-562`; `runtime.rs:4231-4237`).
- A replay with identical prompt, archive commitment and turn budget is a pure lookup. Any mismatch is 409 `idempotency_conflict` (`runtime.rs:1306-1345,2511-2525`).
- `run.sh` mints the key from `/dev/urandom` (`A/run.sh:62-68`).
- It keeps a durable byte-exact receipt (`request.json` + `archive.zip`, mode 0600) in `.runtime/client-submissions/<id>/` and retries only transient failures (408/425/429/5xx, or curl errors) up to 5 times with backoff, replaying the same receipt. `resubmit.sh` replays after an interruption (`A/scripts/submission-common.sh:144-358`; `A/resubmit.sh`).

**Upload spool**
- The archive streams to `state/spool/upload-<uuid>/archive.zip.partial` while being hashed.
- Acceptance requires the received byte count and SHA-256 to match the declaration.
- The file is then fsynced, renamed to `archive.zip`, and the directory fsynced. Any failure removes the spool (`api.rs:312-412`).

**Durable acceptance**
- Structural validation of the zip runs **before** acceptance (`runtime.rs:1347-1364`).
- `accepted.json` is published with no `.await` between its first visible byte and supervisor spawn, so an accepted operation can never be left without an owner (`runtime.rs:1389-1400`).
- The response returns immediately. "The operation never belongs to a connection" (`A/README.md:1237-1243`).
- There is **no waiting endpoint**; callers poll (`A/README.md:1241-1243`).

### 3.3 Staging and the hostile-workspace defence

**Structural checks on the zip.** Before anything stages, and again while extracting (`A/src/staging.rs:377-646`; README `876-882`):
- entry names are canonical relative UTF-8: not empty, no NUL, no leading `/`, no empty, `.` or `..` components (`staging.rs:950-991`);
- only directory, regular-file and symlink entries are allowed; any other Unix mode type is refused (`:993-1017`);
- no duplicate names after directory-slash normalization. The count is cross-checked against the raw end-of-central-directory record, Zip64 aware, so shadowed entries are caught;
- single-part archives only, with no trailing bytes after the EOCD (`:530-646`);
- no file may also be used as a parent directory (`:476-497`);
- declared totals must stay within **200,000 regular files, 250,000 entries (implied directories included), 200 GiB of content** (`config.rs:52-54`; `staging.rs:1045-1059`);
- a symlink target must be 1..1 MiB and contain no NUL (`:439-445,748-753`; `MAX_TARGET_BYTES`, `:1071`);
- decompressed bytes are counted as they stream, so a lying size header fails closed (`:844-850`);
- the entry must remain unchanged between validation and extraction (`:674-703`).

**Extraction**
- Directories become 0755. Files become 0644, or 0755 if any execute bit was set. Dangerous mode bits are stripped (`staging.rs:879-890`; README `900-903`).
- Symlinks are staged as **opaque links that are never followed**. They resolve only inside the agent's mount namespace, where Landlock still governs writes (`staging.rs:22-29,730-792`).
- **Only the outermost archive is extracted.** A zip inside the workspace stays an ordinary file (test at `staging.rs:1400-1517`).
- The consumed archive is deleted right after extraction (`A/src/session.rs:314-336`).

**No files or directories are stripped or excluded from the workspace.** There is no name filter in `staging.rs`, and `run.sh` zips the whole folder, dotfiles included (`zip -0 -ryq … .`, `A/scripts/submission-common.sh:185`). The hostile-workspace defence is **configuration isolation**: these files stay in `/workspace` as ordinary data ("These files remain ordinary copied source files and may be inspected", `A/README.md:894-895`; QWEN.md `:20-22`), but Qwen Code does not treat them as configuration.

What locked mode does **not load or enable**, per `A/README.md:884-896`:
- workspace `.qwen/settings.json`
- workspace `.env`
- project hooks, extensions and skills
- `.qwen/rules`
- `.qwen/output-language.md`
- `.mcp.json`
- `--mcp-config`
- session-injected MCP servers
- managed memory, auto-memory/dream, team memory and synchronization
- auto-skill
- custom slash commands and workflows
- include directories
- permission-rule persistence

`A/docker/config/QWEN.md:15-19` adds **model/provider overrides** and **extra include directories** to the list. A leading `/` in a prompt is literal task text, and init must report `slash_commands: []` (README `892-894`; QWEN.md `:23-24`).

In code:
- locked mode forces `mcpServers` to the top tier only and skips pending MCP gating;
- it sets `contextRuleExcludes: ['**']`, which excludes Qwen conditional rules;
- it removes `onPersistPermissionRule`.
(QCP:33300-33360)

What stays active: ordinary project `QWEN.md` and `AGENTS.md` remain task-level guidance (README `884-886,1297-1308`).

The acceptance run used contradictory fixtures for `.env`, `.qwen/settings.json`, MCP, hooks, rules, skills, memory, output language and slash commands. Session `s-f75db894…` completed seven turns and all five hostile marker files remained absent (MEASURED; `A/README.md:1297-1308`; `B/README.md:1100-1109`). Workspace `.env` loading had also leaked in through a later authentication-revalidation path; this was repaired (`B/README.md:1106-1109`).

### 3.4 Sealed configuration inside the agent image

The image carries these files, hash-checked at build and again at runtime (`A/docker/Dockerfile:204-259`; `run_agent.sh:130-169`):
- `/opt/agent/settings.json`
- `QWEN.md`
- `system.md`
- `deployment-contract.md`
- `runtime-contract.json`
- `toolchain-manifest.json`

`QWEN_HOME=/opt/agent` is read-only (`run_agent.sh:9,105-107`).

**`system.md`** (38 lines, `A/docker/config/system.md`)
- Engineering discipline that applies to both the main session and subagents.
- Inspect before changing; follow project `QWEN.md`/`AGENTS.md` unless they conflict with the sealed contract.
- Report verification truthfully; tool calls are sequential; use absolute paths.

**`deployment-contract.md`** (`A/docker/config/deployment-contract.md`)
- Filesystem (`:3-24`):
  - `/workspace` is read-write and kept.
  - `/artifacts` starts empty and is kept, for deliberate deliverables.
  - `/tmp` is bounded scratch for "derived document pages, archive extraction, databases, compiler probes, indexes, media conversion" and is discarded.
  - `/output` is not mounted.
  - uid 1000, read-only root, no GPU and no Docker socket.
- Network (`:26-44`): `--network none`; the only model path is `http://127.0.0.1:18000/v1`; no apt/pip/npm/cargo/go/maven/git/curl.
- Subagents (`:46-54`): Explore is "not mechanically read-only". It may render or extract PDFs, unpack archives, build databases or indexes, and even modify the workspace. Its workspace and artifact state is content-hashed before and after it runs (effect journal).
- Vision and documents (`:56-69`): see §7.
- Per `A/artifacts/…/INCIDENTS.md:213-217`, the contract also ends with a session-start timestamp computed once at process start.

**`QWEN.md`** (43 lines)
- Nothing is written back to the operator's source (`:8-11`).
- Project files are not configuration (`:13-24`).
- Only native structured tool calls (`:26-29`).
- The turn budget is finite and reaching it is an ordinary outcome; there is no wall-clock cutoff (`:31-38`).
- Each tool result is held to one inline block, with the cut part recoverable via `read_file` (`:39-43`).

**`settings.json`** (`A/docker/config/settings.json`)
- `approvalMode: "yolo"`, `sandbox: false` (`:68-71`).
- `maxSessionTurns: 400`, `maxWallTimeSeconds: -1`, `maxToolCalls: -1`, `maxToolCallsPerTurn: 1000`, `maxSubagentDepth: 1`, `skipLoopDetection: true`, `skipNextSpeakerCheck: true`, `chatCompression.maxRecentImagesToRetain: 0` (`:30-45`).
- All memory, auto-skill and team features off (`:60-67`); telemetry off.
- `modalities {image: true, pdf: false, audio: false, video: false}`, `splitToolMedia: false`, `toolResultContentFormat: "parts"`, `contextWindowSize: 262144`, `thinkingMandatory: true`, `strictToolCalling: true` (`:92-106`).
- Sampling `1.0 / 0.95 / 20 / 0.0 / 0.0 / 1.0`; `extra_body` sets xhigh effort (`:107-122`).

**Tools.** The allowlist is exactly 10 tools: `agent, edit, glob, grep_search, list_directory, notebook_edit, read_file, run_shell_command, todo_write, write_file` (`A/README.md:560-573`; `A/config/stack.lock.json:105-116`). There are no web tools.

### 3.5 Context policy, turns, circuit breakers and subagents

**The context window is partitioned exactly** (`A/README.md:198-270`; `A/patches/README.md:70-109`). W = 262,144:

| Quantity | Formula | Value at W = 262,144 |
|---|---|---|
| D, static preamble | 3W/64 | 12,288 tokens |
| M, largest single inline block | W/8 | 32,768 bytes (UTF-8 NFC) |
| F, per-message framing | measured | 61 bytes |
| C, turn generation room (every turn's `max_tokens`) | derived | 69,509 tokens |
| T, compaction trigger | W − C − D | 180,347 tokens |

- The measured preamble is 7,813 tokens, plus 46 for startup context. The worst-case bound is 11,075, leaving 1,213 spare (`A/README.md:218-228`).
- "prompt + reasoning + tools + final ≤ 262,144" (`A/README.md:150-152`; `B/README.md:56-60`).
- **No phase budget is served** by default. The server and client send none. `thinking_token_budget` and `final_response_token_budget` exist only as per-request options (`B/README.md:552-565`; `A/README.md:147-156`; commit `5cc4187`, "Serve no phase budget by default").
- A turn that generates all of C ends the session as `error_incomplete_generation` (`A/README.md:756-762`).
- Every turn is admitted by an exact `/tokenize` of the rendered request. There is no heuristic estimate or fallback (`:313-333`).

**Turn budget**
- Default 400, ceiling 2,000, counted in turns only. Reaching it exits 53 with `error_max_turns` (`A/README.md:335-375`).
- A subagent counts as one parent turn but has its own budget (`:343-348`).

**Circuit breakers**
- "Cumulative tool calls have no separate cutoff, and one model turn still has a 1,000-tool-call circuit breaker for degenerate output" (`A/README.md:372-374`; `settings.json:37`; `A/config/agent-runtime-contract-v1.json:137`).
- In practice `parallel_tool_calls: false` plus the backend's single-call grammar make each turn one call. A turn carrying more is refused as a defect (`A/README.md:161-167,321-324`).
- A third consecutive "slipped final message" ends the run as `error_slipped_final_message` (`:775-800`).
- Qwen Code's always-on identical-call guard (`consecutive_identical_tool_calls`) killed several benchmark runs (`A/artifacts/…/loop-halt-forensics.md:6-10`; `INCIDENTS.md:80-84`). It is separate from the `skipLoopDetection: true` setting. The current patch concern `correctable-tool-call-repetition` answers a repeated identical call with a corrective tool result first, and only a repeat after that ends the run (`A/patches/source_patch_v1/contracts_qwen_code.py:7625-7639`). The stream contract's name for a loop-detector halt is `error_loop_detected` (`A/README.md:686-687`). The 2026-08 incident was still reported as `error_during_execution` (`INCIDENTS.md:82-84`).

**Stream bounds**
- Before the first chunk, only the 24 h request timeout applies, because the backend runs one generation at a time.
- After that, 4 minutes without a chunk counts as a stall, and the total lifetime is `max_tokens` at a declared 12 tok/s floor: about 5,792 s at C (`A/README.md:764-773`).

**Subagents**
- Only `general-purpose` and `Explore`.
- Foreground only, sequential (one at a time), maximum depth 1.
- No background work, teams, forks, worktrees, model overrides or nesting (`A/config/agent-runtime-contract-v1.json:116-131`; `A/README.md:376-379,524-532`; `agent_exec.rs:99-100`).
- A child owns its spawning call ID as its `kv_scope` (§2.8).

**Images**
- At most 15 per *rendered request*, not per session.
- Static 8-bit RGB/RGBA PNG only, ≤16,777,216 px, aspect ratio ≤30:1, ≤100 MiB on disk.
- The original bytes are sent inline as `data:image/png;base64,`.
- JPEG, WebP, GIF/APNG, BMP, SVG (as an image), grayscale, palette, 16-bit and tRNS images are rejected.
- Compaction drops old pixels.
(`A/README.md:584-630`; `A/config/agent-runtime-contract-v1.json:84-104`; server side at `B/README.md:780-800`)

**Compaction**
- Compaction produces a forced tool call with six sections, at most M bytes, carrying the last turn verbatim (`A/README.md:282-311,487-495`).

### 3.6 Progress, terminal states, cancellation and deletion

**Progress**
- Phases: `Accepted, Staging, PreparingAgent, CreatingTopology, AwaitingReadiness, RunningAgent, Cancelling, CapturingOutput, TearingDown, Bundling, PersistingTerminal, Terminal` (`A/src/progress.rs:38-52`).
- Progress is a complete snapshot, replaced atomically (`progress.json.next` → fsync → rename → dir fsync). It holds at most 4,096 events of at most 4 KiB each (`progress.rs:1-34`).
- Readers poll `progress_revision` and `progress_events` together with live counters: `staged_*`, `output_event_bytes`, `num_turns`, and `observed_output_tokens` / `observed_reasoning_tokens` / `observed_subagent_scope_count` / `observed_unaccounted_records` (`A/docs/session-resource.md:24-41`; `A/README.md:727-737`).

**Terminal states** (`A/protocol/stream-contract-v1.json:1615`, `terminalOutcome`)

| Subtype | Exit code |
|---|---|
| `success` | 0 |
| `error_during_execution` | 1 |
| `error_timeout` | 55 |
| `error_max_turns` | 53 |
| `error_max_tool_calls` | 55 |
| `error_loop_detected` | 1 |
| `error_incomplete_generation` | 1 |
| `error_slipped_final_message` | 1 |
| `error_cancelled` | 130 |

The service surfaces the subtype as `terminal.agent_result.agent_result_subtype` (`A/README.md:680-719`).

**Cancellation**
- `persist_cancel_intent` is made durable before the cancellation token becomes observable (`runtime.rs:1840-1844`).
- The start gate prevents Qwen from starting while the broker's create transaction is still in flight (`session.rs:430-557`).
- The wrapper traps TERM→143 and INT→130, runs Qwen under `setsid`, and forwards the signal to the whole process group (`run_agent.sh:61-101`; README `909-915`).
- `stop.sh` uses `docker stop --timeout -1`, so shutdown cancels sessions and finishes teardown with no deadline (`A/README.md:1182-1188`; `runtime.rs:1927-1936`).

**Deletion**
- DELETE removes one terminal record and its bundle, via a durable delete intent (`runtime.rs:1908-1918`).
- Nothing is pruned automatically by age or count (`A/README.md:1190-1193`).

### 3.7 The result bundle

**Mechanics** (`A/src/bundle.rs:35-240`)
- `tar --zstd --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --format=posix --pax-option=delete=atime,delete=ctime -C <session> staged artifacts control output` (`:113-134`).
- The tree is snapshotted before and after tar, and the archive is rejected if anything changed in between (`:150-163`).
- Written as an exclusive `.partial`, fsynced, SHA-256 computed, published by hard link, directory fsynced (`:99-205`).
- Symlinks are archived as links, never dereferenced. Special files are refused (`:1-8,361-383`).
- There is no `--ignore-failed-read` path (`A/README.md:903-907`).

**Required members** (`bundle.rs:36-66`)
- Directories `staged/`, `artifacts/`, `control/`, `output/`.
- Files `control/prompt.txt`, `control/turn-budget.json`, `output/ready.json`, `output/events.jsonl`, `output/qwen.stderr`, `output/response.txt`.
- `output/qwen-exit-code` is present only when a Docker wait observed an exit (`A/docs/session-resource.md:131-135`).
- `control/start-gate.lock` is never removed, so it is always bundled (INFERENCE from `staging.rs:217-234` plus the fact that `control/` is bundled whole).
- Failure paths add `control/setup-failure.txt` and `control/container-logs.txt` (`session.rs:1808-1831`).

**"The 9 files."** The nine-file bundle is the readiness-boundary cancellation acceptance: zero turns, exit 143, empty event stream (`A/README.md:1319-1323`; `B/README.md:1124-1127`). The repos **do not list** the nine files. Under the current code, a zero-turn bundle holds 7–8 fixed control/output files plus whatever the workspace and artifacts contained. That run also predates the `turn-budget.json` record (INFERENCE). Other recorded counts are workspace-dependent: 19 files in the hostile-workspace run (`A/README.md:1308`) and 11 files in the five-turn production smoke (`B/README.md:1137-1138`).

**Terminal resource.** `terminal.bundle = {sha256, compressed_bytes, uncompressed_bytes, file_count, artifacts_file_count}`, or `null` (`A/docs/session-resource.md:95-111`). `bundle.sh` checks the downloaded bytes against the resource hash, the byte count and the `X-Bundle-SHA256` header, then publishes with no-clobber (`A/bundle.sh:54-88`).

### 3.8 Durable-acceptance and atomic-write guarantees

- **Session layout** is created exclusively and never adopted from stale state. A partial create is removed; recovery validates the layout but never repairs it (`staging.rs:75-156`; README `928-947`).
- **Upload spool:** exact bytes and hash, fsync, rename, dir fsync (`api.rs:312-412`).
- **Acceptance** has no `.await` window between publication and ownership (`runtime.rs:1389-1400`). Startup reads every result directory strictly and **refuses to start** if any record is unreadable, naming each one. It never translates, skips or deletes (`A/README.md:1209-1218`).
- **Terminal persistence:** "`create_new`, write, `fsync`, same-directory hard-link publication, directory `fsync`". If persistence fails, the body stays in memory and is marked erroneous (`A/README.md:1260-1264`).
- **Control records** (`prompt.txt`, `turn-budget.json` in canonical form `{"max_session_turns":N}\n` mode 0444, `start-gate.lock`) are written with `create_new` and fsynced (`staging.rs:158-234`). The launcher re-verifies the budget record byte-for-byte (`agent_exec.rs:254-318`).
- **Capture:** the broker replays `CAPTURE_COMPLETE` with `docker logs --since 0`, replacing a racy `--since 0s` (`A/README.md:917-926`).

---

## 4. The agent image

**Identity**
- Tag `qwen38-agent:0.21.12-b965d5f8-v9`, ID `sha256:c371a572…d0f` (`A/config/stack.lock.json:87-88`; `A/config/release.lock.json`).
- Built `FROM noble-toolchain`, the pinned base `sha256:4a6de2f8…`, which is built `FROM ubuntu@sha256:019e8eb2…` (Ubuntu 24.04 "noble") with Node copied from `node@sha256:d649c27d…` (`A/docker/Dockerfile.base:14-18,31`; `stack.lock.json:6-9,25-34`).
- The apt snapshot is `20260814T120000Z`, and every package version is verified after install (`Dockerfile.base:41-71`).

**Apt packages: 65 top-level pins** (`A/config/agent-apt-packages.lock:1-65`; "the toolchain set has 65 packages", `A/README.md:987-988`):

aria2, bat, build-essential, bzip2, ca-certificates, cargo 1.75.0, clang 18, cmake 3.28.3, coreutils, curl, diffutils, fd-find, ffmpeg 6.1.1, file, findutils, fzf, gawk, gdb 15.1, git-lfs, git 2.43.0, golang-go 1.22, graphviz 2.42.2, grep, imagemagick 6.9.12.98, iproute2, jq 1.7.1, less, lsof, maven 3.8.7, netcat-openbsd, ninja-build, openjdk-21-jdk-headless, openssh-client, pandoc 3.1.3, patch, pkg-config, poppler-utils 24.02.0, postgresql-client 16, procps, psmisc, python3-dev, python3-pip, python3-pytest 7.4.4, python3-venv, python3 3.12.3, qpdf 11.9.0, ripgrep 14.1.0, rsync, rustc 1.75.0, sed, shellcheck 0.9.0, socat, sqlite3 3.45.1, strace 6.8, tar, tmux, unzip, util-linux 2.39.3, vim-tiny, wget, xz-utils, yq 3.1.0, zip, zsh, zstd.

The service/broker runtime set is 5 packages (ca-certificates, curl, iproute2, tar, zstd) (`A/config/service-apt-packages.lock`).

**Non-apt toolchains**
- Go 1.25.13 from a checksum-pinned archive at `/usr/local/bin/go`.
- Node 22.23.2 and npm 10.9.8.
- The Qwen Code 0.21.12 build under `/opt/qwen-code`.
- (`Dockerfile.base:90-111`; `A/docker/config/toolchain-manifest.json:52-206`)
- The manifest verifies 46 command paths and 19 version probes. Examples: `pdftotext -v` = 24.02.0, `pandoc` 3.1.3, `convert` = ImageMagick 6.9.12-98 Q16, `rustc 1.75.0`, `java 21.0.11` (`toolchain-manifest.json`).

**pip and Python packages.** None are installed or listed. There is no pip lock, and runtime installation is impossible offline. The earlier benchmark dossier observed "bare Python + PyYAML" (`A/artifacts/…/failure-dossier.md:90-91`); pytest has since been added from apt (`INCIDENTS.md:222-225`).

**Present and absent tools**

| Present | Absent (not in lock or manifest) |
|---|---|
| Node 22, Python 3.12 (stdlib + apt python3-pytest/yaml), Go 1.25.13 (+ apt 1.22), Rust 1.75, Java 21 + Maven, GCC 13 / Clang 18, CMake/Ninja, Git/LFS, GDB/strace/shellcheck, rg/fd/fzf/bat, jq/yq, sqlite3, psql, **pandoc**, **poppler** (`pdftotext`, `pdftoppm`), **qpdf**, **ImageMagick**, **ffmpeg**, **graphviz**, `file`, archivers (tar/zip/unzip/xz/zstd/bzip2), rsync, tmux, zsh, vim-tiny; curl/wget/aria2/ssh/socat/netcat are present but useless under network none | **No Ghidra, no tesseract/OCR, no LibreOffice**, no browser, no Gradle, no pip packages such as pydantic/numpy/pandas/markdown libraries (INFERENCE from the lock), no `sudo`, no SSH server, no ttyd, no browser server, no runtime installer (`A/README.md:581-582`) |

The README summary list matches (`A/README.md:575-582`).

**Size**
- The agent image size is **not recorded** in either repo.
- The generic toolchain base is "a 4.98 GB toolchain base and a 147 MB runtime base" (`A/artifacts/swe-rebench-2026-07-production-service/INCIDENTS.md:431-433`).
- The backend image archive is 8,561,236,480 bytes (`B/README.md:353-355`).

**Reproducibility**
- `./build.sh` rebuilds every image with `--no-cache --pull=false` and `rewrite-timestamp=true` at `SOURCE_DATE_EPOCH=1786725153`. It refuses unless each image ID equals the locked one (`A/build.sh:88-130`; README `1007-1025,1101-1105`).
- "Two independent `--no-cache` agent builds must produce the exact locked image ID; a cached rebuild alone is not accepted as reproducibility evidence" (`A/README.md:1023-1025`).
- "A full clean no-cache release build reproduced the exact locked agent, relay, capture, broker, and service image IDs" (`:1045-1048`).
- `release.sh` loops (build → adopt → commit) until `./build.sh` agrees (`:1126-1134`). The current `release.lock.json` hashes match the tree's `stack.lock.json` and `build-inputs.sha256` (verified by `sha256sum`: `73fe1889…`, `a8a432d4…`).
- Images are **not** reproducible across hosts. A second machine gets them from the release archive (`A/README.md:983-985`).
- Java's JKS timestamps are normalized by a dedicated tool (`:1017-1023`).
- npm audit reports 68 advisories (3 low, 35 moderate, 27 high, 3 critical); they are deliberately left unfixed (`:1061-1066`).

**Test counts** (README): 117 Rust tests (100 service, 9 broker, 3 relay, 2 capture, 3 agent-exec) and "5,259 tests across fifty-seven focused test files" (`A/README.md:1037-1046`). The backend README records older counts; see §9.

**Sandbox assertions.** See §1.6. The Landlock filesystem sandbox and the attestation are verified on every start (`run_agent.sh:239-262`). Agent seccomp is Docker's default builtin profile; the stacked custom filters apply only to the relays. The AppArmor `docker-default` profile is a host-contract check (§1.6).

---

## 5. Measured numbers, checked against the source text

| Number | Exact text or derivation | Citation | Status and notes |
|---|---|---|---|
| KV bytes per token = **24,832** | "388 bytes per head per token: 256 FP8 key bytes, 128 packed value bytes, and four FP16 scale/zero-point bytes … 24,832 bytes per token". 388 × 4 KV heads × 16 layers = 24,832 | `B/README.md:605-607`; `B/docs/qwen38-context-turboquant-audit.md:71-90` | Arithmetic from pinned code (checked) |
| **6.0625 GiB** at 262,144 tokens | 6,509,559,808 bytes = 6.0625 × 2³⁰ | `B/README.md:609,697-698`; audit doc `:84-90` | Arithmetic (checked); the unit test printed `raw_k8v4=6509559808B/6.0625GiB` (audit doc `:61-63`) |
| ~23.126 GiB at 1M tokens | stated | `B/README.md:610`; `A/README.md:105` | Arithmetic |
| Weights **21.34 GiB** | "vLLM logs 21.34 GiB loaded model memory, 6.45 GiB KV reservation, 1,024 MiB reclaimable TurboQuant workspace, and about 0.06 GiB CUDA graph capture" | `B/README.md:641-643`; `A/README.md:106` | MEASURED (v13-era logs) |
| KV allocation 6,925,634,765 B → **264,115** cache tokens (1.01× native; 1,971 spare) | stated | `B/README.md:612-615`; audit doc `:92-95` | MEASURED under the old byte flag. Current sizing is `--kv-cache-users 1`; the config comment estimates the pool at ~6.4 GiB (`B/config/runtime-v1.sh:296-297`) |
| VRAM 32,607 MiB total; 31,647 used / 464 free before the 15-image run; 31,797 / 314 after the suite | stated | `B/README.md:649-655` | MEASURED on v13 |
| Native boundary: 262,143 + 1 accepted; 262,144 + 1 → 400 | stated | `B/README.md:663-667` | MEASURED |
| 15 × 4096² images in one request, 246,022 prompt tokens, 30 strings transcribed; the 16th rejected | stated | `B/README.md:820-827` | MEASURED |
| Aspect: 22,080×736 and the reverse at 16,250,880 px (15,870 visual tokens) pass; 31:1 (21,824×704) → 400 | stated | `B/README.md:829-832` | MEASURED |
| **Prefix-cache TTFT (text): 32.233× warm/cold**; 65,529-token cold prompt, 0 hits; 64,480 tokens reused warm; fresh-salt control reused 0 | stated | `B/README.md:885-887`; `A/README.md:808-813` | MEASURED (v13 probe) |
| Image-history probe: 14,560 prefix tokens + multimodal hit, **17.089×** TTFT; OpenAI and Anthropic histories render identical **16,562** token IDs | stated | `B/README.md:867-868,887-889`; `A/README.md:815-821` | MEASURED |
| Qwen Code v8 acceptance: **296,939** prompt tokens over **20** requests; **241,280** local prefix hits; **55,659** computed; **3** multimodal hits | stated | `B/README.md:1110-1112`; `A/README.md:823-828` | MEASURED. 241,280 + 55,659 = 296,939 (checked). **81.3% hit rate is my derivation** (241,280 / 296,939 = 0.8126); the repos never state it |
| Repeated 4-turn image task: **4.044140×** mean-TTFT improvement | "4.044140-times mean-TTFT improvement on the repeated four-turn image task" | `B/docs/agent-service-and-client-audit.md:197-199` | MEASURED (Qwen Code through the proxy) |
| SWE-rebench pilot: **11/11** evaluator tests, **61 turns**, 1,090,658 ms wall, 1,084,315 ms agent, 60/60 tool calls, input 3,407,979 / output 38,461 tokens, 5,301-byte patch | table | `A/docs/production-swe-rebench-pilot.md:20-36`; `B/README.md:1139-1143,1216-1221`; `A/README.md:1336-1339` | MEASURED on release `7a329f6`, vLLM image `sha256:587e8710…` (older), `preserve_thinking=false` session. The **other variant (true) also resolved, in 83 turns / 1,081,835 ms** (`A/artifacts/…/runs/…/pair-summary-release-7a329f6.json`, `comparison`). "One completed task is a lifecycle proof … not a benchmark-suite score" |
| Readiness-boundary cancellation: ack **785 ms**, teardown **1,863 ms**, 0 turns, exit 143, empty stream, nine-file bundle | stated | `A/README.md:1319-1324`; `B/README.md:1124-1127` | MEASURED (session `s-cd0e7f51…`, older 32-hex handle era) |
| K8V4 kernel vs FP32 reference: max_abs 0.00381172, min cosine 0.999998331, 18 boundary choices | stated | `B/README.md:622-631,1088-1089`; audit doc `:134-139` | MEASURED on RTX 5090 |
| NVFP4 worst-layer kernel: rel-L2 0.003161705 / 0.003603501 / 0.003809735 (M = 1/17/129) | stated | `B/README.md:229-242` | MEASURED |
| Weight audit: FP8 rel-L2 0.0266015 / cos 0.9996461; NVFP4 0.1082006 / 0.9941543; 161 RMSNorm tensors restored | stated | `B/README.md:229-231,244-253,700-721` | MEASURED (offline tensor audit) |
| Decode speed ~30 tok/s | "262,144 tokens at the measured ~30 tok/s is ~2.4 h" | `A/artifacts/…/INCIDENTS.md:256-259` | MEASURED (informal) |
| Turn latency ~35–40 s at 60–200k context; "fixed ~38 s/turn" | stated | `A/artifacts/…/failure-dossier.md:107-109,166-169` | MEASURED (benchmark forensics) |
| Long-corpus Test A: 1,554,654 words, ~427 reads, ~970 turns, ~18 h, ~28 serial compactions; per-compaction success 15/18 (83.3%) → 0.6% single-attempt survival | stated | `A/artifacts/…/INCIDENTS.md:326-347`; `A/artifacts/long-context-harness/README.md:20,40-44,55-76` | MEASURED plus arithmetic. The probe (26 × 50 KB chunks) passed with 3 compactions; Test A failed 4 times; Test B never succeeded; Test C has not been attempted |
| Static-YaRN allocation edge: initialized at 335,872; OOM at 337,920 | stated | `B/README.md:673-678` | MEASURED; rejected as a mode |
| Bundle of 2.9 GB for one benchmark workspace | stated | `INCIDENTS.md:113` | MEASURED |

---

## 6. What agent_service explicitly does not do

| Not done | Quote or evidence |
|---|---|
| **No capacity or placement decisions; assumes it may sit behind a load balancer** | "How many sessions run at once is deliberately not this service's decision: it cannot know it is not one worker behind a load balancer, so serving capacity and placement are governed above it" (`A/README.md:8-11`); "There is no serving-capacity gate: sessions run concurrently … a placement decision for whatever sits above this service" (`:1231-1236`). The user directive behind it is at `B/transcripts/codex-session-01a000ca-user-messages.json`, message 154 |
| **No queue or scheduler in the service** | Concurrent sessions "interleave their model turns through [the backend's] queue and compete for its prefix cache; that is a throughput property of the chosen backend, not a correctness property of this service" (`:1233-1236`). The backend runs one sequence (`--max-num-seqs 1`), and queued requests wait (`A/README.md:766-768`) |
| **No multi-user scheduling or fairness; no auth** | Sessions are isolated but not scheduled (above). No API auth is described (INFERENCE). `kv_scope` "provide[s] cache accounting, with no authentication or confidentiality promise" (`:464-466`) |
| **No other client modes or fallbacks** | "no Claude mode, Codex mode, Qwen3.6 mode, text-only mode, reduced-context mode, alternate port, retry downgrade, heuristic context clamp, XML tool recovery, host client installation, or compatibility fallback" (`:13-17`) |
| **No MCP, hooks, skills, memory or workflows** | See §3.3 (`:528-531,884-896`) |
| **No UI** | "The old ttyd observer, browser UI, client daemon, and alternate adapter paths were not retained" (`B/docs/agent-service-and-client-audit.md:46-48`); "There is no `sudo`, SSH server, ttyd, browser server" (`A/README.md:581-582`) |
| **No waiting endpoint, callbacks or streaming progress** | "There is no waiting endpoint … callers poll" (`:1241-1243`). The earlier `/wait` endpoint and folder-path body were removed 2026-08-18 (`:1346-1348`) |
| **No second upload protocol; no host input mounts** | "one workspace archive plus text prompt — so there is no second upload protocol" (`:586-588`); "mounts no host input tree at all" (`:870-871`) |
| **No wall-clock limit; no cumulative tool-call cap** | `:335-339,372-374` |
| **No background, parallel, nested or team subagents** | `:376-379,524-527` |
| **No retries** | `maxRetries: 0` (`settings.json:94`); "no client retry/downgrade, XML recovery, implicit continuation, or partial-call execution" (`B/README.md:1193-1194`) |
| **No network access for agents; no runtime installs** | deployment-contract `:26-44` |
| **No compatibility with older record formats; no automatic pruning** | `A/README.md:1209-1218,1190-1193` |
| **No write-back to the caller's folder** | QWEN.md `:10-11`; README `:900-903` |
| **KV offload: not controlled by the service, but relied upon** | The service does not configure vLLM. It declares the backend's offload in its runtime contract (`A/config/agent-runtime-contract-v1.json:66-70`), and the prompt tells the model that the DDR5 host tier may keep the parent's KV while a child runs (QCP:170267). So KV offload **is used**, as a backend property |

**Open obligations the service itself lists (DESIGN / PLANNED).** Six broader implementation obligations are open: daemon metrics, provider-stream lifetime, reasoning stored by reference, prompt-hook failure policy, atomic speculative file/history acceptance, and resident background AgentTool settlement (`A/README.md:552-558`; `A/patches/README.md:358-436`). Also open:
- the compaction/continuation obligations in `A/docs/design/correctness-round7.md:19,35`;
- Incident 7's emitter half (`INCIDENTS.md:292-298`);
- the backend's end-to-end BF16 versus NVFP4/K8V4 context audit (`B/README.md:723-753`; `A/README.md:90-95`).

---

## 7. Feeding a Markdown knowledge base or Obsidian vault into a session

This section combines code-backed facts with clearly labelled inferences.

**1. Prepare the submission**
- `./run.sh /abs/canonical/vault /abs/prompt.txt [--max-session-turns=N]`.
- The folder must be an absolute, canonical, non-symlink directory (`A/run.sh:40-48`).
- The prompt file must be 1..32,768 bytes (`:53-58`) and the service also requires NFC (§3.2). Put long instructions in files in the vault, for example a root `AGENTS.md` or `QWEN.md` with the vault's conventions. Those stay active as project guidance (`A/README.md:884-886`).

**2. Serialization**
- `(cd vault && zip -0 -ryq archive.zip .)` produces a stored (uncompressed) archive. It recurses into **hidden directories** and stores symlinks as links (`A/scripts/submission-common.sh:167-187`).
- An empty folder becomes the canonical 22-byte empty zip (`:179-183`).
- The upload deadline is 300 s + archive_cap / 100 MiB/s (`:71-77`), about 2,348 s (INFERENCE from the arithmetic).

**3. Limits** (§3.2–3.3)
- ≤200,000 regular files, ≤250,000 entries including implied directories, ≤200 GiB of content, archive ≤200 GiB + 64 MiB.
- Entry names must be valid UTF-8 with no `..` components.
- Entry types: directories, regular files and symlinks only. FIFOs and devices are rejected.
- Symlink targets ≤1 MiB. Symlinks stay symlinks; they are never followed on the host and resolve only inside the agent container.

**4. What happens to `.obsidian/`, `.git/`, `.trash/`, `.canvas` and Kanban notes**
- They are **preserved verbatim**: no path filter exists (`staging.rs`, entire extract path). Their mode bits are normalized to 0644/0755.
- `.obsidian/plugins/*.js` are plain files. Nothing executes them unless the model chooses to run them in the shell.
- The hostile-config list in §3.3 is about Qwen-specific files (`.qwen/*`, `.mcp.json`, `.env`). Those are also kept but never loaded as configuration.
- Nested zips (for example attachment archives) are **not** extracted.
- INFERENCE: upstream Qwen Code's `context.fileFiltering.respectGitIgnore` and `respectQwenIgnore` default to `true` (upstream docs table inside QCP:8640-8656), and the sealed `settings.json` does not override them. So `glob`, `grep_search`, `list_directory` and `read_file` may skip or refuse paths matched by the vault's `.gitignore` or `.qwenignore` (read_file errors "ignored by .qwenignore pattern(s)", QCP:233567-233574). The shell (`rg`, `fd`, `cat`) is unaffected.

**5. Reading**
- `.md`, `.canvas` (JSON) and Kanban `.md` are text. `read_file` requires `offset`, where 0 means the start. Pages hold at most M = 32,768 bytes of NFC text, and every partial read names the exact next call (QCP:234221; `A/README.md:240-260`).
- Search with `grep_search`/`glob`, or `rg`/`fd`/`jq`/`yq`/`pandoc` in the shell.
- Oversized tool results are kept in a session artifact store (500 MiB quota) and paged back (`A/README.md:511-514`; `A/patches/README.md:249-273`).
- The main-session prompt tells the model to delegate broad reading to foreground subagents (QCP:170267).

**6. Binary files**
- `read_file` returns "Cannot display content of binary file" for binary content (QCP:233425-233436).
- Range parameters on binary, PDF, image, notebook or SVG files are refused (QCP:234221).
- Use shell tools instead, for example `sqlite3`, `unzip -p`, or Python stdlib.

**7. Images in the vault**
- Only static 8-bit RGB/RGBA PNG (≤16,777,216 px, ≤30:1, ≤100 MiB) can be *viewed*, via `read_file`. At most 15 can be in one rendered request.
- JPEG/WebP/GIF/SVG attachments are rejected by `read_file`. The contract allows "deliberately render[ing]" derived images into `/tmp`, and "A derived image enters `read_file` only after it satisfies the exact PNG contract" (`deployment-contract.md:63-67`).
- INFERENCE: ImageMagick may pick palette or grayscale PNG output for low-colour images, which the contract rejects. Forcing RGB (for example `PNG24:`) would be needed. The repos do not say this.
- On compaction, old pixels are dropped (`A/README.md:618-625`).

**8. PDFs**
- Settings `modalities.pdf: false` (`settings.json:103`). "PDF handling is local computation, not direct PDF vision. Poppler/QPDF/Pandoc/ImageMagick may extract or deliberately render pages into scratch … never triggers an online or silent lossy fallback" (`deployment-contract.md:63-67`). "PDF remains text extraction only" (`A/README.md:609-610`).
- `read_file` on a PDF is whole-file text extraction. There is no `pages` parameter; it was removed after it caused loop-guard kills (`INCIDENTS.md:170-193`; QCP:234221 removes it).
- A PDF too large for one read fails with this guidance: "Extract the pages you need as text in the shell with `pdftotext -layout -f FIRST -l LAST -- <path> -`, then read that output" (QCP:257128-257141).
- No OCR tool exists (no tesseract). INFERENCE: a scanned PDF can only be read by rendering pages with `pdftoppm -png` to PNG and viewing them (≤15 per request).
- "Explore … may render or extract PDFs" (`deployment-contract.md:48-51`).

**9. Editing**
- Tools: `edit` (string replacement), `write_file`, `notebook_edit`, or any shell program. Landlock allows writes only under `/workspace`, `/artifacts`, `/tmp` and `/qwen-runtime` (`agent_exec.rs:377-391`).
- Put generated wiki pages or an index into the vault (`/workspace`), or deliverables into `/artifacts`.
- Scratch indexes or databases belong in `/tmp` (8 GiB tmpfs, discarded) (`deployment-contract.md:9-13`).
- INFERENCE: the patch notes read/write evidence for overwrite permission ("a file this session wrote itself is still one it is allowed to overwrite", `A/patches/README.md:182-185`), which implies read-before-overwrite rules.
- INFERENCE: new files the agent creates probably get mode 0600 / dirs 0700 under the wrapper's `umask 077` (`run_agent.sh:103`). tar preserves modes.

**10. Getting the vault back**
- Poll `./session.sh <id>`. When the session is terminal, run `./bundle.sh <id> /abs/out.tar.zst`; it verifies SHA-256, byte count and header, and allows `curl --max-time 900` (`A/bundle.sh:58-63`).
- `tar --zstd -xf` gives:
  - `staged/`: the **entire final vault**, including `.obsidian/`. Not a diff; no patch is produced;
  - `artifacts/`;
  - `control/` (prompt, turn budget, start gate);
  - `output/` (`events.jsonl` with the full event stream, `response.txt` with the final answer, `ready.json`, `qwen.stderr`, `qwen-exit-code`).
- **All member mtimes are epoch 0 and owners are 0:0** (`bundle.rs:113-127`). INFERENCE: syncing back into a live Obsidian vault should compare content (`rsync --checksum` or `git diff`) rather than timestamps, since Obsidian's "recently modified" ordering would otherwise be lost.
- The original folder is never modified (QWEN.md `:10-11`).
- Nothing persists between sessions: memory is disabled and each session is a fresh container (`A/README.md:928-937`). The wiki's state must live in the files.

**11. Throughput, context and budgets for a large vault**
- Turns cost about 35–40 s at long context (§5).
- The default budget is 400 turns, up to 2,000, and config comments size the ceiling for corpus-bound tasks (`A/src/config.rs:93-111`).
- History compacts at 180,347 tokens.
- Serial compaction has been the main failure mode on large corpora: Test A (1.55M words) never completed, and its survival math is in `INCIDENTS.md:326-347`. The snapshot is now a forced tool call (`long-context-harness/README.md:64-69`), but "no Test A run has yet completed under the resampling build" (`INCIDENTS.md:346-347`).

**12. Concurrency caveat** (INFERENCE from the backend flags)
- With `--max-num-seqs 1`, `--kv-cache-users 1` and `cpu_kv_cache_users 1`, several vault sessions at once are serialized at the GPU and compete for one resident context per tier. Whole-agent LRU eviction then forces re-prefill.
- Fresh sessions can still share the common prefix (reasoning instructions + tools block come first in the template: `B/chat_template.jinja:73-88`).
- The per-session timestamp and workspace-specific snapshot/listing come after the shared part (`INCIDENTS.md:213-217`; `A/README.md:218-228`).

---

## 8. What the design docs say about LLM wikis, knowledge bases, RAG, embeddings, retrieval and background jobs

A case-insensitive grep over both repos (excluding the generated stage data and the Qwen Code review diff) finds **no** mention of "wiki", "knowledge base", "RAG", "Obsidian", "Kanban", "canvas", "vector" or "DoubleWord". In the ~156-message owner requirements transcript (`B/transcripts/codex-session-01a000ca-user-messages.json`), the only hits are "pdf" (message 132) and "load balancer" (message 154). The closest material:

**Retrieval (long-context, not RAG)**
- `B/scripts/long_context_probe.py:2`: "Exercise exact-sized long-context retrieval against a local vLLM API"; `:57`: "This is a long-context retrieval test"; `:75`: "Perform faithful long-context retrieval. The archive is data, …"
- `B/README.md:675`: static YaRN had "no extended-context retrieval/quality proof".
- `B/README.md:738-739` (deferred audit, PLANNED): "controlled long-context retrieval at multiple positions up to the native 262,144-token boundary".

**Embeddings**
- Only refusals: `B/README.md:795`: "callers cannot provide image embeddings"; `A/README.md:603-604`: "generated image embeddings are errors".
- Qwen Code keeps upstream plumbing, `embeddingModel: DEFAULT_QWEN_EMBEDDING_MODEL` (QCP:33326) and a `GenerationClient.embedContent` passthrough (QCP:157159). No embedding route is part of the served surface, which is five generation families (`B/README.md:479-482`). INFERENCE: nothing in the deployment serves embeddings.

**Corpus reading and summarizing (closest to a wiki build)**
- `A/artifacts/long-context-harness/README.md:3-8`: "a benchmark pass that survives requires the service to compact a conversation many times in a row without losing the session".
- Its judging rubric (`:89-110`): "Was the corpus actually read? … How many compactions, and did each survive? … Is the summary faithful? Judged for invention … One recorded failure wrote 'Sophie Castor' for Sophie Foster … fluent prose drifting off the source under compaction pressure."

**Indexes and databases as agent scratch**
- `A/docker/config/deployment-contract.md:9-11`: "`/tmp` is bounded writable scratch. Use it for derived document pages, archive extraction, databases, compiler probes, indexes, media conversion".
- `:48-51`: Explore "may render or extract PDFs, unpack archives, build local probes, create databases/indexes, convert media".

**Context economy guidance to the model**
- QCP:170267: "Context length is the most important thing in AI, and it is immoral to waste it … Delegate generously … broad searches, surveying unfamiliar code, reproducing a failure, converting documents".

**Background or long-running jobs**
- There is no batch-inference or DoubleWord-style design. The service is itself an asynchronous, connection-independent job API: "`run.sh` returns as soon as the service has durably accepted it; the session does not belong to the connection" (`A/README.md:1161-1162`); "There is no waiting endpoint: the operation never belongs to a connection, and callers poll" (`:1241-1243`).
- Owner rationale, transcript message 149: "I come from a philosophy that I hate connection-oriented stuff … 'Forcing somebody to get progress' might actually be a bad idea".
- `A/docker/config/QWEN.md:31-38` ("Long-running work"): "There is no Qwen wall-clock cutoff. Use each shell call's explicit timeout carefully and keep long-running commands observable." Subagents never run in the background (§3.5).
- The backend mounts `/v1/chat/completions/batch`, where one batch shares one `kv_scope` (SPP test `test_a_batch_submits_every_conversation_under_the_one_agent`). With `--max-num-seqs 1` there is no parallelism, and agent_service does not use the route.

---

## 9. Discrepancies and documentation drift

1. **Image adoption vs. README.** `B/README.md:18-19,459` and the KV design doc (`:200-203`) say the v23 source revision "awaits … adoption", yet the runtime image is pinned at v26 (`B/config/runtime-v1.sh:5-9`; commit `fb8d39c`). No live validation of the `kv_scope` and whole-agent-eviction behaviour on v24–v26 is recorded.
2. **"eighteen maximum-size images".** `B/README.md:669-670` says the accepted boundary "included all eighteen maximum-size images", contradicting the 15-image cap. Line 1081 says "with fifteen images".
3. **Qwen Code patch size.** `A/README.md:1037-1038` says "exactly 104 changed/new files" and `B/README.md:1094-1096` says "61 changed/new files … 2,427 assertions across 23 focused test files". The current review diff and stage data cover **799 paths** (81 new, 27 deleted) (`grep -c '^diff --git'` on QCP; `FINAL_FILES` in `A/patches/source_patch_v1/generated_qwen_code_stage.py`). The 35 semantic concerns are confirmed (`contracts_qwen_code.py:7350-…`).
4. **Rust test counts.** `A/README.md:1045-1046` says 117 tests (100 service …); `B/README.md:1097-1099,1134-1136` and the pilot doc (`:136-139`) say 44 service, 9 broker, 3 relay, 2 capture, 2 agent-exec. These are different eras.
5. **Subagent policy tone.** `B/docs/agent-service-and-client-audit.md:66-71` (2026-08-15) says subagents "must not be the default operating pattern", while the current sealed prompt says "Delegate generously" (QCP:170267).
6. **Thinking retention.** The same audit doc (line 272) says "completed hidden thinking is omitted", while both READMEs now say historical reasoning is always preserved and `preserve_thinking` other than `true` is refused (`B/README.md:524-550`; `A/README.md:169-196`; `B/chat_template.jinja:58-60`).
7. **Measurements from older backends.** The SWE-rebench pilot ran on vLLM image `sha256:587e8710…` and release `7a329f6`. The v13 prefix-cache numbers predate `kv_scope`, the user-count sizing and the ARC-policy removal.
8. **Historical host tier.** The "7,747,584,000-byte ARC host tier" (`B/README.md:1070`) is superseded by `cpu_kv_cache_users: 1` (~6.6 GB) (`B/config/runtime-v1.sh:281-288,316-318`).
9. **Unverified external audit.** `B/docs/qwen38-deployment-correctness-audit.md` is an unverified Codex log (its own header says "None of its findings has been verified"). Its finding 9 ("scope eviction can destroy another surviving agent's prefix") is addressed by the shared-prefix stage (`B/docs/model-output-and-audit-fixes.md:604-637`, source-tested only).
