# Shared prefix caching and KV capacity

## Identity and matching

Every generation request supplies `kv_scope`, an opaque, nonempty agent ID
that names one line of work. Use the same ID for every request that continues a
conversation, and a new, never-used ID for a fork or a subagent. Whitespace-only
IDs are invalid; other strings are preserved exactly. The server assigns no
meaning to punctuation, session names, or the relationship between IDs.

A line of work has at most one request in flight and generates one sequence per
request. The frontend refuses a second request under an ID whose earlier
request has not finished, and refuses `n > 1` or several Completions prompts
under one ID; the batch route names one distinct ID per conversation. The one
API server process admits every generation request, which is what lets it see
every request in flight; the engine refuses to start with several. Every
refusal is made at admission, which each generation surface completes before it
answers, so it is an HTTP error of its own rather than an error inside a stream.

Prefix lookup matches content and `cache_salt` alone, as upstream vLLM does.
The ID never enters a block hash and never restricts a lookup: every request
may reuse any resident prefix whose tokens and salt it shares, whether its ID
is new or already has cached blocks. `cache_salt` is what isolates callers.
The server stores no parent pointer, lineage, or declaration of a fork.

For example:

1. Agent A computes prefix P and continuation A1.
2. Agent B requests P and continuation B1. B reuses P from A's cached data,
   then computes B1.
3. A later request of either agent reuses P, A1 and B1 wherever its own
   tokens continue them, while they remain resident.
4. Releasing A removes A's retention references. B's retained context keeps
   P and B1 resident while B is retained.

The generation request models require the ID and the served schema declares
it; the render models they share their shape with have no ID. It travels
through the sampling parameters to the engine request and offload request
context. The common engine validation holds the same rules for direct sampling
callers. Rendering and pooling do not allocate a generation context and do not
require an ID.
An ongoing input stream belongs to its admitted agent ID. Per-chunk sampling
parameters may change generation settings but must retain that exact ID.
An ID change is refused before dispatch and aborts only that stream's request.

## Content availability

Physical cache entries identify content by the existing chained prefix hash
and KV group. The GPU block pool keeps upstream's per-block LRU eviction. When
an attention block grows, its earlier immutable prefix stays reachable under
its shorter hash, so a request whose prefix ends inside the block still hits
it; a recurrent block replaces its checkpoint.

A CPU chunk can coalesce several GPU blocks. The CPU manager records, for each
resident row, the canonical content its key names and the content actually
written there. Attention entries describe their hash-sized data units;
recurrent entries describe their final checkpoint. A lookup checks the data
needed at its candidate resume boundary against the row's written content, so
a chunk that holds only part of a window serves exactly the frontiers that part
covers.

## CPU contexts and eviction

The agent ID groups retention. CPU retention records a request's complete
working set across all KV groups: the full-attention prefix, each required
attention window or recurrent state, and any partial tail. Each dependency
includes its required data extent. Physical transfer pins are separate from
these context references.

Finishing a request retains its complete context for the next turn. A new
nonempty turn replaces the previous idle turn from the same agent. Concurrent
requests retain their separate working sets while active. Missing or failed
dependencies prevent an incomplete finished context from being retained.

Capacity allocation is planned before eviction:

1. Reclaim unreferenced, idle entries, including obsolete windows left in spare
   capacity after a context advanced.
2. If more space is needed, release whole agents in least-recently-used order,
   excluding the incoming agent. Release every exclusively held, unpinned
   entry of the chosen victims. Shared references from surviving contexts
   continue to protect their entries.
3. If the required space cannot be made available, decline the store without
   a partial eviction. Transfer pins protect bytes until the transfer ends.

Releasing one agent never removes another retained context's reference to
shared data. A failed transfer invalidates each context that depended on the
failed entry. Other contexts retain their complete working sets. Active
contexts may have pending dependencies; completion of their transfers makes
those dependencies readable.

Retained contexts are bounded by the tier's chunk count. Without sharing, each
one holds at least one chunk of its own, so no more could stay resident;
sharing must not let identical content under ever more IDs grow this metadata
without bound. Past the bound, the least recently used retained context is
dropped; active requests keep theirs. This is separate from physical byte
accounting: three agents sharing two chunks still consume two physical chunks.

## Coalesced windows and secondary storage

A CPU chunk can contain several GPU blocks. Window recycling can leave its
leading slots unwritten. The CPU manager tracks the key's canonical contents
separately from the data actually available in that row. Store descriptions
name only non-null GPU sources; lookup checks the data needed at the candidate
resume boundary against that availability.

For example, with 16-token GPU blocks, three per CPU chunk, and an 80-token
sliding window, the 96-token frontier may need only tokens 16 through 96.
The first CPU chunk can safely lack tokens 0 through 16. That chunk cannot
serve a shorter request whose window needs those bytes.

Computation or a secondary promotion may fill an incomplete CPU row in place.
The row must be idle. The fill becomes a pending write; once complete, the row
serves every frontier its content now covers, for every agent. A failed fill
invalidates the row and its dependent contexts.

Secondary storage keys promise canonical entries. Incomplete window chunks
are available to valid local window lookups but are not advertised or exported
as canonical entries. A partial-tail key already defines its shorter valid
span and can be exported once that span is complete. Physical storage
transfers do not select a generation prefix or retain an agent context.
An entry transferred before a generation supplies its hash chain serves a
generation lookup once that lookup describes its content.

The native CPU and tiering managers share these rules. The alternate simple
CPU connector is upstream's: it matches content and salt and keeps no agent
retention.

## Capacity declared in user contexts

Operators declare GPU capacity with `--kv-cache-users N` and native CPU
offload capacity with `cpu_kv_cache_users: N`. Both are positive integer
counts of full-length contexts. Runtime KV geometry determines the byte
cost; cache sharing can allow more contexts to coexist within that capacity.

For each GPU worker:

```text
blocks_per_user = sum over KV groups of
    ceil(spec.max_memory_usage_bytes(vllm_config) / spec.page_size_bytes)
pool_blocks = N * blocks_per_user + 1
pool_bytes = pool_blocks * bytes_per_pool_block
```

The additional block is the pool's null block. Pool byte geometry accounts
for uniform, packed and general layouts. The declared demand must fit the
physical memory remaining after measured resident allocations and execution
peaks. Profiling's utilization estimate cannot silently shrink the declared
pool. A demand that cannot fit refuses with its capacity cost.

For native CPU offload, let `T_g` be a group's tokens per GPU block multiplied
by `blocks_per_chunk`, and let `F_g = ceil(max_model_len / T_g)`:

```text
chunks_per_user = sum over KV groups of
    F_g                                      for full attention
    min(F_g, window_chunks_g + eagle_extra_g) for windowed/recurrent groups
pool_chunks = N * chunks_per_user
```

A recurrent group needs its trailing checkpoint. EAGLE attention includes
the additional checkpoint its lookup verifies. Window geometry, recurrent
mode and EAGLE classification are normalized at the offload configuration
boundary and used by both sizing and scheduling. Grouped layer specs use
the same block geometry as engine hash sizing, including context parallelism.

CPU row bytes include the worker layout, its number of copies and alignment.
Shared data is charged once physically. Transfer pins consume space in the
same pool; concurrent transfers and advancing working sets can cause stores
to be declined or whole contexts to be evicted. The user count sizes the
resident working sets and is not a promise to retain every historical
checkpoint or admit unlimited in-flight copies.

## Cache-hit timing

Any request can observe shared-prefix reuse through latency, whatever its ID.
IDs group retention; they provide neither matching restrictions,
authentication nor a confidentiality boundary. No artificial timing padding is
added. `cache_salt` remains an independent cache-key input, is what isolates
callers, and is never derived from an agent ID.

## Validation and adoption

The source tests exercise content matching under new and existing IDs, shared
eviction, whole-agent retention and its bound, attention prefix aliases, sparse
window availability and fills, transfer failures, complete-context retention,
secondary promotion/export, geometry and sizing, and required opaque IDs on
generation surfaces, admission before the status line on every generation route, and the
one-request-in-flight rule. Scheduler and manager tests use metadata and
disposable containers;
they do not require a GPU or model execution.

The deployment's landmark transformations, reviewed diffs, source hashes and
Docker unit assertions must reconstruct these sources exactly. The backend
build and release adopt the image and archive pins together. Source validation
does not claim that an awaiting-adoption image is already serving these rules.
