# Shared prefix caching and KV capacity

## Identity and matching

Every generation request supplies `kv_scope`, an opaque, nonempty agent ID.
Use the same ID for successive requests from the same agent. Whitespace-only
IDs are invalid; other strings are preserved exactly. The server assigns no
meaning to punctuation, session names, or the relationship between IDs.

An agent with no cached blocks of its own may match a prefix in the shared
cache. Selecting that prefix makes it a fork: the agent acquires references
to the selected data. Once it has cached blocks, matching uses only the data
it has acquired or computed. Computation extends that cache. The server
stores no parent pointer, lineage, or declaration of a fork.

The rule applies across GPU and CPU together. A GPU-resident block makes the
agent existing even if its CPU cache is empty, and a CPU-resident block does
the same if its GPU cache is empty. After all of an agent's cached blocks
have been evicted, its next lookup again has no cache of its own and may
acquire an initial shared prefix.

For example:

1. Agent A computes prefix P and continuation A1.
2. New agent B requests P and continuation B1. B can acquire P from A's
   cached data, then compute B1.
3. B's later requests may reuse P and B1. They cannot acquire A1 merely
   because A1 is also resident.
4. Releasing A removes A's references. B keeps P and B1 while its context
   remains resident.

The required ID travels through each generation protocol's sampling
parameters to the engine request and offload request context. The common
engine validation also covers direct sampling callers. Rendering and
pooling do not allocate a generation context and do not require an ID.

## Content and membership

`PrefixCacheIndex` is the shared membership catalog. Physical cache entries
identify content by the existing chained prefix hash and KV group. Agent
IDs do not enter those hashes. Attention entries describe their hash-sized
data units; recurrent entries describe their final checkpoint.

A lookup captures a stable membership view before querying groups. Inspecting
a candidate does not acquire it. Acquisition follows selection of a usable
prefix, or computation. This keeps one group's tentative hit from changing
the remaining groups' eligibility during the same lookup.

A physical entry can have multiple agent references. A complete selection
uses the entry's shared content description; a shorter selection records
its acquired subset. This matters when a common resume boundary ends inside
a larger GPU block or coalesced CPU chunk. Selecting 72 tokens cannot acquire
later data merely because the physical entry extends to 96 tokens.

The catalog follows insertion, removal, immutable aliases and physical tier
copies. Moving a GPU block's hashes registers the destination before removing
the source, so a move does not erase surviving memberships. Attention growth
preserves earlier immutable prefix aliases. Replacing recurrent state removes
the replaced checkpoint's membership.

Native CPU stores coalesce source entries while preserving every source agent's
acquired subset, even when a different agent issues the store. The catalog
collects the whole transfer batch before changing memberships, since ownership
record eviction can also change its source records. Copying a shared prefix
cannot leave its other users dependent on the GPU copy alone.

## CPU contexts and eviction

CPU retention records a request's complete working set across all KV groups:
the full-attention prefix, each required attention window or recurrent state,
and any partial tail. Each dependency includes its required data extent.
Physical transfer pins are separate from these context references.

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

Cached agent records are bounded by each tier's cache-entry capacity. At that
metadata bound the tier releases the oldest complete agent record. This is
separate from physical byte accounting: three agents sharing two chunks
still consume two physical chunks. Full memberships share one content
description, and subsets are stored only when selection needs them.

## Coalesced windows and secondary storage

A CPU chunk can contain several GPU blocks. Window recycling can leave its
leading slots unwritten. The CPU manager tracks the key's canonical contents
separately from the data actually available in that row. Store descriptions
name only non-null GPU sources; lookup checks the data needed at the candidate
resume boundary against both availability and the agent's membership.

For example, with 16-token GPU blocks, three per CPU chunk, and an 80-token
sliding window, the 96-token frontier may need only tokens 16 through 96.
The first CPU chunk can safely lack tokens 0 through 16. That chunk cannot
serve a shorter request whose window needs those bytes.

Computation or a secondary promotion may fill an incomplete CPU row in place.
The row must be idle. The fill becomes a pending write, preserves every
existing agent's acquired subset, and grants the producer only its produced
data. A failed fill invalidates the row and its dependent contexts.

Secondary storage keys promise canonical entries. Incomplete window chunks
are available to valid local window lookups but are not advertised or exported
as canonical entries. A partial-tail key already defines its shorter valid
span and can be exported once that span is complete. Physical storage
transfers do not select a generation prefix or create an agent membership.
An entry transferred before a generation supplies its hash chain cannot be
acquired until its content has been described.

The native CPU and tiering managers share these rules. The alternate simple
CPU connector also uses the GPU membership catalog, acquires selected hits,
and preserves membership when copying physical entries.

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

A fresh agent ID can observe shared-prefix reuse through latency. Existing
agents query their own acquired cache, but anyone able to submit a fresh ID
can attempt an initial match. IDs provide cache accounting and matching
semantics; they provide neither authentication nor a confidentiality boundary.
No artificial timing padding is added. `cache_salt` remains an independent
cache-key input and is never derived from an agent ID.

## Validation and adoption

The source tests exercise initial forks, existing-agent matching, shared
eviction, GPU/CPU membership, aliases and physical copies, shortened selections,
sparse window fills, transfer failures, complete-context retention, secondary
promotion/export, geometry and sizing, and required opaque IDs on generation
surfaces. Scheduler and manager tests use metadata and disposable containers;
they do not require a GPU or model execution.

The deployment's landmark transformations, reviewed diffs, source hashes and
Docker unit assertions must reconstruct these sources exactly. The backend
build and release adopt the image and archive pins together. Source validation
does not claim that an awaiting-adoption image is already serving these rules.
