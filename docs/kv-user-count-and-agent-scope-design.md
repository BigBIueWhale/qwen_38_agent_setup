# KV sizing by declared users, eviction by agent identity — design note

Status: reviewed design for the `kv-user-count-sizing-and-scope-eviction`
source stage. Authored while runtime-v15 benchmarks were still executing;
the image pins (`EXPECTED_IMAGE_ID`, archive name/hash, profile label) are
deliberately untouched and belong to the release step that follows the
benchmark runs.

## 1. Problem

Two tiers of KV cache exist and both were sized in raw bytes that are only
valid on one machine:

- VRAM: `--kv-cache-memory 6925634765` — a hand-measured constant. The
  number encodes "one 262,144-token context" for exactly this GPU, this
  quantization, this hybrid layout. On a B200 it is silently wrong.
- Host RAM: `kv_connector_extra_config.cpu_bytes_to_use: 7747584000` — the
  same disease. The byte cost of a context only exists post-engine-init
  (page sizes, hybrid group structure, TP sharding), which is why the
  operator had to bake measured constants.

Recency alone fails at one provable moment: when a subagent finishes, its
context is dead AND most-recently-used, so a recency policy protects a
corpse and evicts the live main agent's blocks. A recency policy can only
*approximate* that deadness, so the server accepts a protocol by which a
client that tracks agent lifetimes can *state* it instead.

## 2. Decisions

### 2.1 Sizing currency: user contexts, declared, required

Both tiers are declared as counts of resident full-length user contexts.
Bytes are derived inside vLLM at the point where the KV cache spec exists.

- VRAM: `--kv-cache-users N` (new `CacheConfig.kv_cache_users`). Required:
  engine initialization fails if it is absent while the model has KV cache
  groups. Derivation, per worker, in `get_kv_cache_configs`
  (vllm/v1/core/kv_cache_utils.py):

      blocks_per_user = sum over groups of
          cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
      needed_blocks  = N * blocks_per_user + 1        # BlockPool block 0 is the null block
      needed_bytes   = needed_blocks * _pool_bytes_per_block(...)

  `blocks_per_user` is the exact expression `get_max_concurrency_for_kv_cache_config`
  already uses, and `_pool_bytes_per_block` mirrors the divisor of every
  layout branch in `get_kv_cache_config_from_groups`, so the round trip
  bytes -> num_blocks is exact for the uniform, packed, and general
  layouts. Cross-check against the live server: the measured
  "Maximum concurrency 1.01x" (264,115 / 262,144 tokens) solves only as a
  134-block pool over a 133-block request — i.e. today's hand-tuned byte
  constant is `1 * blocks_per_user + 1` in disguise. The new flag makes
  that identity the contract instead of an accident.

  The declaration is authoritative, exactly as the byte flag was.
  Profiling's conservative availability estimate (the utilization budget
  minus every observed resident; it preferred ~1.3 GiB less than the
  proven production pool) is informational only; it can neither shrink
  nor veto the declared pool. What refuses a declaration is physical
  capacity: the whole card minus every measured resident — weights,
  non-torch allocations, the transient activation peak that recurs each
  forward pass, frontend reservations, and the CUDA-graph charge when
  opted in. The utilization factor is deliberately not charged against
  the pool: its holdback is a discretionary reserve the superseded byte
  flag also ignored, and the pinned production footprint (29.6 GiB of a
  31.8 GiB card) sits beyond the 0.9 budget while fitting the card with
  slack. `gpu_memory_utilization` keeps its real jobs — the startup
  free-memory requirement and the estimate's budget. "The estimate would
  prefer less" proceeds; "the card physically cannot hold N contexts"
  refuses with the per-user byte cost in the message. Capacity is never
  silently clamped to "what fits".

- Host RAM: `kv_connector_extra_config.cpu_kv_cache_users: N` (required by
  `CPUOffloadingSpec`, inherited by `TieringOffloadingSpec`). Derivation in
  the spec, from geometry the offload boundary already normalizes:

      tokens_per_chunk_g = tokens_per_block_g * blocks_per_chunk
      chunks_per_user    = sum over groups of
          cdiv(max_model_len, tokens_per_chunk_g)              # full attention
          min(window_chunks_g, cdiv(max_model_len, tpc_g))     # windowed / recurrent
      num_cpu_chunks     = N * chunks_per_user

  The windowed term is deliberate, not an approximation: the load path
  (`_sliding_window_lookup`) reads exactly the trailing
  `sliding_window_size_in_chunks` chunks of a windowed group, and a Mamba
  group in `align` mode resumes from a single trailing state chunk. A
  full-length re-entry therefore touches `chunks_per_user` chunks and no
  more. Counting every boundary snapshot instead would double this box's
  footprint past its 8 GiB `/dev/shm` for the same declared "1 user";
  intermediate-boundary snapshots are opportunistic cache that lives in
  slack and ages out. For this deployment: 127 attention chunks + 1 GDN
  state chunk = 128 chunks/user (~6.6 GB at the 51,650,560 B aligned
  chunk), inside the existing `--shm-size 8g`.

  The per-group window is computed once, in
  `build_offloading_config` (offloading/config.py), stored on
  `OffloadingGroupConfig.sliding_window_size_in_chunks`, and consumed by
  both the sizing math and the connector scheduler. The scheduler's local
  recomputation (`get_sliding_window_size_in_chunks`) moves to the config
  builder so sizing and load planning can never disagree.
  `OffloadingModelConfig` gains `max_model_len` for the same reason.

  Chunk *counts* are topology-free; chunk *bytes* already scale through
  `worker_kv_bytes_per_block * num_copies` with the existing
  `world_size` / `replicated_layout` handling in cpu/spec.py, which this
  change does not touch. That is the whole TP=1 vs TP=8 story for the CPU
  tier; on the GPU tier, derivation runs per worker on projected groups
  and the min-reduce across workers is upstream's unchanged mechanism.

### 2.2 Identity: `kv_scope` on every generation protocol

Two request fields, placed beside `cache_salt` / `kv_transfer_params` on
every generation surface this image can import — the proven set is five:
openai chat_completion, openai completion, openai responses, anthropic,
and scale_out token_in_token_out. Cohere is deliberately not in the set:
its protocol models hard-import the optional `cohere` SDK, which a
`--network none` build cannot install, so upstream's `/cohere/v2/chat`
existed only as a try/except-import accident. That guarded registration
is excised from the server assembly (`generate/api_router.py`) along
with the endpoint-only `--cohere-is-reasoning-model` knob, and the image
build asserts that no `/cohere` route is registered — the endpoint's
absence is an enforced fact, not an installation state. The
`cohere_format` renderer flag survives: it serves `--tokenizer-mode
cohere` model families over the OpenAI endpoints and is unrelated to
the excised HTTP surface.

- `kv_scope: str | null` — the agent scope owning this request's KV.
  Encodes the harness convention verbatim (agent_service commit 00be562):
  `null`/absent = the main session line, otherwise the spawning
  `tool_use_id`. The null scope is shared bookkeeping for every main line;
  it is never releasable, so its width is harmless — ownership is only
  ever *acted on* through explicit release of a named scope, and
  tool_use ids are unique.
- `kv_scope_release: list[str] | null` — scopes terminated since the
  previous request. The harness attaches these to the first request it
  sends after observing a subagent's terminal result (in practice: the
  parent's resume request). In-band release is sufficient: release only
  matters when someone still runs to benefit from the freed space; if no
  request ever follows, nothing competes.

Transport mirrors `kv_transfer_params` exactly: each protocol's sampling
conversion writes the fields into `SamplingParams.extra_args`;
`Request.__init__` materializes them as typed attributes next to
`kv_transfer_params`; `InputProcessor._validate_params` rejects non-string
scopes, empty strings, and non-list releases (VLLMValidationError — a 400,
not a crash). No new EngineCoreRequest plumbing, no per-endpoint privileged
path — Anthropic gets the same two lines as everyone else.

Delivery to the manager: `_create_req_context` copies `request.kv_scope`
onto `ReqContext.kv_scope`; every existing manager call already carries
the ReqContext. Release executes in
`OffloadingConnectorScheduler.on_new_request` — before the manager
registers the new request — via `OffloadingManager.release_scope(scope)`.
The scheduler calls `connector.on_new_request(request)` when the request
is added, before its first lookup, so a resuming parent sees the freed
space in the same scheduling step.

### 2.3 One eviction mode: scope-owned live-LRU

`CPUOffloadingManager` stops delegating to a pluggable `CachePolicy` and
owns a single structure:

- every chunk carries an owner scope: the scope of the request that last
  stored or touched it. Touch transfers ownership, so a prefix chunk
  shared by a subagent and its parent stays alive if the parent used it
  after the subagent did; scope death only takes what the dead scope alone
  was keeping warm.
- `release_scope(s)`: drop the scope's idle chunks immediately (with
  BlockRemoved events); chunks with an in-flight store or load are marked
  dead and reclaimed at `complete_store` / `complete_load` — release is a
  state transition, never a data race with the DMA engine.
- capacity pressure among *live* chunks stays plain LRU (the ordered
  evictable list the old LRU policy proved). Recency was only ever wrong
  about dead scopes, and identity now states death exactly; ARC existed to
  approximate that statement and has nothing left to approximate.

The manager keeps its existing ref-count / write-pending accounting,
events, threshold tracker, and stats surface. `BlockStatus` moves into
manager.py. The tiering primary tier (`CPUPrimaryTierOffloadingManager`)
inherits the mode; `TieringOffloadingManager.release_scope` forwards to
the primary tier — secondary (fs/obj/p2p) tiers have their own lifecycle
and receive the base-class default. The base-class default is an explicit
documented no-op *on the abstract manager only*: a tier without identity
tracking has nothing to drop as a unit; it is not a configuration
fallback, and the pinned profile's manager implements the full semantics.

### 2.4 Supersede: what stops being accepted, what is deleted

Fails startup (not ignored, not deprecated):

- `cpu_bytes_to_use`, `eviction_policy`, `cache_policy_module_path` in
  `kv_connector_extra_config` — `CPUOffloadingSpec` now validates the
  extra config against an exact key allowlist and names the offender;
  the error for the three superseded keys states the replacement.
- `--kv-cache-memory[-bytes]` — the EngineArgs field, CLI flag, and
  `LLM(kv_cache_memory_bytes=...)` kwarg are deleted; supplying the flag
  dies in argparse, the kwarg dies as an unexpected argument.
- `--kv-offloading-size` / `kv_offloading_backend` — deleted with their
  `VllmConfig` synthesis; the flag existed only to manufacture
  `cpu_bytes_to_use`.
- `--num-gpu-blocks-override` alongside `--kv-cache-users` — rejected at
  derivation. It is block-denominated sizing and cannot coexist with a
  declared user count; it is *rejected* rather than deleted because it was
  never part of this profile's configuration and its removal would gut
  unrelated upstream unit scaffolding — the acceptance surface is closed,
  which is the invariant that matters.

Deleted outright (files removed from the tree and `rm`'d from the image):

- `vllm/v1/kv_offload/cpu/policies/{__init__,base,factory,lru,arc}.py`
- `tests/v1/kv_offload/cpu/policies/{__init__,test_factory}.py`

Retained deliberately, with changed or unchanged roles:

- `CacheConfig.kv_cache_memory_bytes` survives as *internal availability
  state only*: the CPU backend derives it from `VLLM_CPU_KVCACHE_SPACE`,
  and the startup-plan cache re-applies a previously *measured*
  availability. Neither is a user KV-size declaration; the field's
  docstring and the gpu_worker log that used to advertise the flag are
  rewritten accordingly. The persisted plan can only shortcut profiling —
  the N-context demand must still fit inside whatever it reports.
- `SimpleCPUOffloadConnector` and the NIXL/P2P connectors are different
  connectors with their own contracts, not alternate modes of this one;
  they are out of scope exactly as the Anthropic-only-privilege rule is in
  scope: the boundary is the `OffloadingConnector` spec family
  (`CPUOffloadingSpec` and its `TieringOffloadingSpec` subclass), which is
  converted whole.

### 2.5 Patch-framework extension: expressible deletion

The landmark framework already validates "deleted-file hunks"
(`review_after == ""` requires `after == ""` describing the whole file)
but could not commit them: commit wrote zero-byte files and the reader
refuses files without a terminal newline. The stage above requires true
deletion, so the transaction learns it end-to-end:

- `FileIdentity.after_sha256: str | None` — `None` means absent after.
- empty pristine files (`policies/__init__.py` is 0 bytes) cannot carry a
  hunk (`before == after == ""`), so a hunkless deletion is evidenced by
  the review diff's `deleted file mode` header instead: the parser
  returns those paths, and `_verify_review_artifact` requires the set of
  hunkless deletions in the diff to equal the stage's edit-less
  `after_sha256 is None` identities — the Python data and the reviewed
  diff still describe byte-identical transformations.
- `plan()` deletes the key after proving the edit result is empty;
  `commit()` unlinks with the same backup/rollback discipline as writes;
  state reading tolerates absence only for planned deletions and reads
  0-byte files as `""`.
- `build-vllm.sh`'s worktree comparison learns the ` D ` porcelain code:
  a deleted path must be absent in both the patched verification worktree
  and the live tree.

## 3. Subagent lifecycle the protocol admits

The sequence below is the contract the server implements; it is what a
client that tracks agent lifetimes sends to use scope ownership.

1. Main agent runs; its requests carry `kv_scope: null`. Stored/touched
   chunks are owned by the null scope.
2. Subagent spawned from `tool_use` `toolu_X`; its requests carry
   `kv_scope: "toolu_X"`. Its stores are owned by `toolu_X`; any shared
   prefix chunks it touches transfer to `toolu_X` (and transfer back the
   moment the parent touches them again).
3. Subagent finishes. Nothing happens yet — vLLM cannot know, and does
   not guess.
4. The parent's next request carries `kv_scope: null,
   kv_scope_release: ["toolu_X"]`. At `on_new_request`, before lookup,
   every chunk still owned by `toolu_X` is dropped as a unit (in-flight
   transfers drain first). The parent's own chunks were never candidates:
   they are owned by `null`.
5. The parent's lookup then runs against a pool where the dead context
   occupies nothing — the exact inversion of the ARC failure, on both
   this box (1 declared user) and a DGX (32 declared users, where a dead
   scope would otherwise squat on 1/32 of the durable tier indefinitely).

The GPU tier keeps upstream recency: its pool is `N * blocks_per_user`
deep, every offloaded chunk also exists in the CPU tier, and a dead
scope's VRAM blocks are recycled within one pool cycle of allocation
pressure — the tier self-heals, and reordering its eviction would buy a
transient win at the cost of threading scope state through the core block
pool. The durable tier is where corpses rot, and that is where identity
now lives. Revisit only if a measured DGX workload shows the transient
VRAM squat mattering.

## 4. File map

vLLM worktree (runtime):

| File | Change |
| --- | --- |
| vllm/config/cache.py | + `kv_cache_users`; − `kv_offloading_size`/`kv_offloading_backend`; `kv_cache_memory_bytes` re-documented as internal |
| vllm/config/vllm.py | − byte-flag → `cpu_bytes_to_use` synthesis |
| vllm/engine/arg_utils.py | + `--kv-cache-users`; − `--kv-cache-memory-bytes`, `--kv-offloading-size` |
| vllm/entrypoints/llm.py | − `kv_cache_memory_bytes` kwarg |
| vllm/v1/core/kv_cache_utils.py | user-count derivation + fail-closed gates in `get_kv_cache_configs` |
| vllm/v1/worker/gpu_worker.py | profiling log rewritten (no dead-flag advice; plan still saved) |
| vllm/v1/kv_offload/config.py | + `max_model_len`, per-group `sliding_window_size_in_chunks` |
| vllm/v1/kv_offload/base.py | `ReqContext.kv_scope`; `OffloadingManager.release_scope` |
| vllm/v1/kv_offload/cpu/spec.py | `cpu_kv_cache_users` derivation; exact-key allowlist; policy knobs gone |
| vllm/v1/kv_offload/cpu/manager.py | scope-owned live-LRU; `release_scope`; `BlockStatus` moves in |
| vllm/v1/kv_offload/cpu/policies/* | deleted |
| vllm/v1/kv_offload/tiering/spec.py, tiering/manager.py | follow the single mode; release forwards to primary |
| .../kv_connector/v1/offloading/config.py | window classification moves here; new config fields |
| .../kv_connector/v1/offloading/scheduler.py | consume stored windows; scope into ReqContext; release at on_new_request |
| five protocol files | `kv_scope`, `kv_scope_release` beside `cache_salt`/`kv_transfer_params` |
| vllm/entrypoints/generate/api_router.py | Cohere registration and serving state excised; absence enforced |
| vllm/entrypoints/openai/cli_args.py | − `--cohere-is-reasoning-model` (endpoint-only knob) |
| vllm/v1/request.py | typed `kv_scope`/`kv_scope_release` attributes |
| vllm/v1/engine/input_processor.py | fail-closed field validation |

vLLM worktree (tests): rewrite `tests/v1/kv_offload/cpu/test_manager.py`
around the single mode + scope release; delete the policies tests; update
spec/connector construction in `tests/v1/kv_offload/test_factory.py`,
`tests/v1/kv_connector/unit/test_offloading_connector.py`,
`offloading_connector/{utils,test_events,test_config}.py`,
`test_hma_auto_config.py`; new `tests/entrypoints/test_kv_scope_protocol.py`
proving both fields and their extra_args propagation on all five importable
surfaces plus the enforced absence of any /cohere route;
sizing-derivation unit coverage.

Repo: `config/runtime-v1.sh` (VLLM_ARGS + hashes), `scripts/build-vllm.sh`
(EXPECTED_STATUS incl. ` D ` lines, cmp loop, new hash wiring, manifest
line count), `containers/Dockerfile.runtime` (new COPY/verify rows, `rm`
of deleted modules), `patches/vllm-kv-user-count-sizing-and-scope-eviction.patch`,
`patches/source_patch_v1/{framework,test_framework,compile_review_diff,contracts_vllm,generated_vllm_stages,manifest}`,
`config/deployment-inputs.sha256` (70 → 71 lines).

## 5. Interim state until the release step

This section described the window between authoring and the
post-benchmark rebuild, during which `config/runtime-v1.sh` named flags
only the next image understood and `start.sh` failed closed at vLLM
argument parsing. That window is closed: the v16 release step re-pinned
`EXPECTED_IMAGE_ID`, the archive pair, and the profile label together,
with the image rebuilt twice to prove the pinned ID reproducible. The
v15 archive and cache volume are retained as a rollback path until the
new stack is proven healthy. Nothing in this change could be *silently*
half-adopted at any point.
