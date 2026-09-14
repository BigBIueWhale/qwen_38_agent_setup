# Qwen3.8 deployment correctness audit (Codex, unverified)

Produced on 2026-09-14 by Codex CLI 0.153.4 running gpt-6-astra, in two `/goal` passes in one
conversation, launched from this repository. The first pass audited the deployment against how
agent clients such as Claude Code drive a model API. The second audited the correctness and
completeness of this repository's vLLM patches, and looked for missing ones. It read code and
pinned public sources only; it ran nothing and changed nothing.

Everything below is Codex's own append-only log, kept verbatim. None of its findings has been
verified by this project yet. Some entries reflect the state of the workstation where it ran,
not the deployment itself; for example, a runtime archive that lives only on the serving machine.

---

## Clones

## Audited
- Brief, complete owner-only history (1–168), complete README: read; pinned upstream 9df9b0b0a1816b6d0d0f6ecd0da563cc37fd72f5; initial worktree clean at 1c63b01400bcaefcbaae531e57fcd310b9de1366 except intentionally patched vLLM. No previous goal turn to classify.
- Scope inventory: thirteen patch artifacts; patched source diff spans 91 tracked files; full area audit pending. Inspection only; no runtime/build/test execution.

## Confirmed issues

## Dismissed

## Clones
- https://huggingface.co/Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (downloaded chat_template.jinja only).
- https://github.com/anthropics/anthropic-sdk-python@f31b3a2c750429e211af7a1d7bb4b8fc61caedd3 (v0.67.0).
- https://github.com/QwenLM/qwen-code@b965d5f8c24f48e65fb0b17c7d45f34ca4ce8f38 (v0.21.12).

## Audited
- Official Qwen template, Unsloth template, served template, complete derivation script: comparison read; retention refusal is an intentional change, not an upstream limitation; parser/protocol propagation still under audit.

## Dismissed
- Developer-role and consecutive leading-system merging: Unsloth extension preserves ordinary single-system Qwen rendering; no concrete wrong result established.
- Tool-history JSON strings rejected by Jinja: vLLM deserializes historical function arguments before applying the template; not a normal OpenAI round-trip break by itself.

## Clones
- https://github.com/mlc-ai/xgrammar@557becfb64c503ae9c04344b0047661f43f44320 (v0.2.3; comparator from pinned vLLM CUDA test requirements, installed version not yet established).

## Confirmed issues
1. **High — custom stops promote truncated XML into executable calls.** File: `vllm/vllm/entrypoints/openai/chat_completion/serving.py:770,1091`; `vllm/vllm/parser/qwen3.py:81`; `vllm/vllm/v1/engine/output_processor.py:672`. Trigger: auto tool calling with a string argument and `stop=["HALT"]`, when a schema-valid generation reaches `<tool_call>\n<function=run>\n<parameter=command>\necho HALT` before closing the parameter/function/call. Wrong result: detokenization cuts at HALT with finish `stop`; the parser manufactures JSON from the unfinished parameter; Chat reports `tool_calls`, and Anthropic reports `tool_use`, rather than incomplete output. Why wrong: a stop-string match does not prove complete tool syntax; the new truncation guards only preserve `length`. The README explicitly forbids incomplete generations becoming executable calls. Verification: verified by reading the stop, parser, and both serving paths; no generation performed.
2. **High — `$ref` tool argument types are lost after constrained generation.** File: `vllm/vllm/tool_parsers/utils.py:1007`; `vllm/vllm/parser/engine/parser_engine.py:262,394,1211`. Trigger: a tool schema such as `{"type":"object","properties":{"count":{"$ref":"#/$defs/count"}},"$defs":{"count":{"type":"integer"}},"required":["count"]}` and a complete Qwen call with `<parameter=count>\n7\n</parameter>`. Wrong result: served arguments contain `{"count":"7"}`; the declared integer becomes a string on both parser paths. Why wrong: the XML converter makes every parameter a string; type recovery does not resolve `$ref` and defaults to string, then returns the call without validating its schema. This violates the declared tool schema even when generation obeyed it. Verification: verified by reading; XGrammar v0.2.3's schema parser/reference resolver is corroborating source, not a claim about the installed package version.
3. **High — Anthropic tool failures lose their failure status.** File: `vllm/vllm/entrypoints/anthropic/serving.py:407,450`; `vllm/vllm/entrypoints/anthropic/protocol.py:57`. Trigger: a user tool_result with `is_error:true` and ordinary stdout/result text (e.g. a command emits `Done` but exits unsuccessfully). Wrong result: the model receives exactly the same tool response as with `is_error:false`; the failure bit is accepted and discarded. Why wrong: the Anthropic tool_result schema carries `is_error` separately from content; Qwen Code v0.21.12's Anthropic converter sets it for failed tool results (`converter.ts:677`). A long tool loop loses the only status distinction and can treat failure as success. Verification: verified by reading; no client issue reported.
4. **Medium — Anthropic custom stop metadata is wrong.** File: `vllm/vllm/entrypoints/anthropic/serving.py:644,838`. Trigger: `stop_sequences:["END"]` actually matches in an ordinary text completion. Wrong result: both modes report `end_turn`; the matched `stop_sequence` is never populated. Why wrong: the pinned Anthropic SDK's generated `types/message.py` requires `stop_reason=stop_sequence` and the matched string for this case; Chat already retains `choice.stop_reason`, but the converter ignores it. Verification: verified by reading.
5. **Medium — unsupported nested Anthropic tool-result content is silently deleted.** File: `vllm/vllm/entrypoints/anthropic/serving.py:424`; `vllm/vllm/entrypoints/anthropic/protocol.py:56`. Trigger: a tool_result containing a document block, such as a text document (`type:document`, source `type:text`, `media_type:text/plain`, `data` with tool output), valid in Anthropic SDK v0.67.0's `ToolResultBlockParam`. Wrong result: the request is accepted, the document is ignored, and an empty or partial tool response is sent to the model. Why wrong: the profile deliberately rejects unsupported media/file parts; it must not accept and erase tool output. Only text/image/tool_reference branches are handled, with no rejection branch. Verification: verified by reading.

## Audited
- Anthropic text/image tool-result conversion and stop metadata: issues 3–5; plain inline images retain their originating result and ordering.
- Parser schema coercion and stop-string termination: issues 1–2; grammar/parser alternate-syntax mismatch remains under investigation.

## Dismissed
- Historical Chat `reasoning_content` alias dropped: the request model normalizes it to `reasoning` before chat_utils builds the template's compatibility alias.
- Missing Anthropic final cache/input usage updates: SDK v0.67.0's stream accumulator explicitly applies input/cache fields from message_delta; final totals can replace initial uncached values.

## Clones
- https://github.com/openai/openai-python@15afa21e54952c06e2ac4d3e3a82f144c2cf9ed9 (v2.26.0).

## Confirmed issues
6. **Medium — most Anthropic validation errors still use the wrong error envelope/type.** File: `vllm/vllm/entrypoints/serve/exception_handling/handlers/validation.py:130`; `vllm/vllm/entrypoints/anthropic/api_router.py:38`. Trigger: `/v1/messages` with `max_tokens:0` or `thinking.type:disabled` (request-model validation), or a valid-shaped request rejected by the served template/image validation. Wrong result: the former receives an OpenAI error envelope (`error.type:"Bad Request"`, no top-level `type:"error"`); returned backend errors retain types such as `BadRequestError` rather than `invalid_request_error`. Why wrong: the route's new catch only handles exceptions raised after FastAPI validation and does not normalize ErrorResponse returns; README promises Anthropic HTTP 400 invalid_request_error for both paths, and the pinned Anthropic schema defines that envelope/type. Verification: verified by reading router, error builder, and registered exception handlers.
7. **High — Responses accepts miscorrelated tool history and binds results positionally.** File: `vllm/vllm/entrypoints/openai/responses/serving.py:624`; `vllm/vllm/entrypoints/openai/responses/utils.py:310`; `chat_template.jinja:156`. Trigger: two Responses function_call items A/B followed by function_call_output B/A, or a missing/orphan output. Wrong result: no ChatCompletionRequest correlation validator is instantiated; call IDs are discarded by the template and B's result is rendered in A's position. Why wrong: the IDs carry protocol correlation, while Qwen's training template uses order; the Chat/Anthropic validator exists specifically to prevent this corruption, but Responses bypasses it. Verification: verified by reading the Responses model validators, input converter, preprocess_chat path, and positional template.
8. **High — Responses stream IDs disagree with its own terminal response.** File: `vllm/vllm/entrypoints/openai/responses/streaming_events.py:1028`; `vllm/vllm/entrypoints/openai/responses/serving.py:1556`; `vllm/vllm/entrypoints/openai/responses/utils.py:101`. Trigger: any completed streamed function call; a client handles output_item/arguments events and also retains response.completed.output for the next turn. Wrong result: the streamed function item/call get fresh random IDs unrelated to the parser's IDs; terminal construction reparses accumulated text and creates different item IDs and call IDs. Why wrong: one streamed output item must retain one identity; executing under the streamed call_id then replaying the terminal output yields a result referencing a different call, and event consumers cannot correlate the terminal snapshot. Verification: verified by reading; the selected non-Harmony SimpleContext path recreates the final response after emitting stream items.

## Audited
- Responses input conversion/correlation and stream identity: issues 7–8; store defaults, incomplete item transitions, phase budgets, cancellation, and context/cache paths still pending.
- Anthropic exception routing: issue 6; HTTP status is 400 on these triggers, but the promised protocol body is not supplied.

## Dismissed
- Responses mixed-channel delta loss: `_process_simple_streaming_events` calls `split_delta` before the state processor, avoiding the apparent TOOL_CALL-priority loss.
- Responses zero reasoning usage by default: final generator derives the exact count from accumulated token IDs through the Qwen reasoning adapter when the context counter is zero.
- Reading the documented v22 image archive: path is absent in this worktree; no installed XGrammar version inferred from it. Source-only audit continues.

## Audited
- Complete runtime config, start/status/stop wrappers and implementations, runtime-common, build-vllm: read; archive presence is a mandatory start/status gate, independent of live image health.
- Docker runtime COPY inventory and CPU build assertions: two modified kernel files are not installed; numerical units are compiled but the GPU kernel audit is not run during image build.
- CPU offload manager, sizing, connector store/load planning: scope grouping does not model shared-prefix ownership.

## Confirmed issues
9. **Medium — scope eviction can destroy another surviving agent's prefix.** File: `vllm/vllm/v1/kv_offload/cpu/manager.py:238,270,317`; `vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:344,642`. Trigger: agents A and B use the same system/tool prefix without different cache salts; A stores it first, B reuses it and stores its distinct tail; a later agent creates pressure that drains A's group far enough to remove the shared prefix. Wrong result: B's own group survives, but B no longer has a whole restorable context and must prefill the missing prefix. Why wrong: hashes do not include kv_scope; existing keys are skipped at store and retain only A as owner; eviction cannot account for B's dependence. This violates the declared whole-context survival contract. Group draining also stops at the immediate block deficit (`manager.py:278`), so the policy is not atomic whole-agent eviction. Verification: verified by reading; no claim of incorrect attention values on a cache miss.
10. **Medium — the image build omits both reviewed TurboQuant kernel guards.** File: `containers/Dockerfile.runtime:282,441,1092`; `scripts/build-vllm.sh:73`. Trigger: build the declared runtime from its upstream base after applying all thirteen transformations. Wrong result: `triton_turboquant_store.py` and `triton_turboquant_decode.py` are modified and checked in the source reconstruction, but neither is copied into the image, patched there, or included in installed-file verification; the image nevertheless advertises the guard patch's digest. Why wrong: the patch's only runtime changes (non-finite value metadata handling and refusal of the alternative FP8 format on older GPUs) are absent from the recipe's installed changes. Verification: omission verified by reading all COPY/RUN instructions; actual inherited kernel bytes and live numerical effects are not established because the image archive is absent and execution is prohibited.

## Clones
- https://github.com/huggingface/transformers@5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9 (v5.15.0).
- https://github.com/python-pillow/Pillow@10.4.0 (tag; downloaded PngImagePlugin.py and libImaging/Unpack.c); also read PngImagePlugin.py at tag 12.0.0 as a cross-version comparator. Installed Pillow version not asserted.

## Confirmed issues
11. **Medium — the PNG bit-depth gate admits 16-bit source images and silently drops their low bytes.** File: `vllm/vllm/multimodal/media/image.py:119`. Trigger: a static inline 16-bit truecolor RGB/RGBA PNG within the count, size and aspect bounds, such as an image exported by an image-processing tool. Wrong result: the request passes the purported eight-bit-only test; Pillow decodes to eight-bit pixels, discarding the lower byte of each 16-bit sample. Why wrong: checking `image.mode in ("RGB", "RGBA")` does not check PNG source depth. Pillow's pinned PNG mode table maps `(16,2)` to RGB and `(16,6)` to RGBA; its `unpackRGB16B`/`unpackRGBA16B` copy bytes 0/2/4[/6] only. README:650–651 expressly requires rejecting 16-bit inputs, rather than accepting a lossy conversion. Verification: verified by reading local validation and Pillow 10.4.0/12.0.0 source; no image decoded or test run.

## Audited
- All nine copied unit scripts: read; no additional unit-induced runtime defect established. GPU audits are standalone acceptance programs, not live evidence from this audit.
- Vision content/connector/decoder guards, exact processor settings, workspace lifetime, in-place GELU change: issue 11; normal RGB/RGBA routing and phase restoration show no additional defect by reading.

## Confirmed issues
12. **High — forced named tool calls are reported as finished assistant turns.** File: `vllm/vllm/entrypoints/openai/chat_completion/serving.py:770,1004,1091`; `vllm/vllm/entrypoints/anthropic/serving.py:563,644,838`. Trigger: a complete normally stopped call with Chat `tool_choice:{"type":"function","function":{"name":"read_file"}}` or Anthropic `tool_choice:{"type":"tool","name":"read_file"}`. Wrong result: the tool-call payload is returned with Chat `finish_reason:"stop"`; Anthropic converts it to `stop_reason:"end_turn"` despite its tool_use block. Why wrong: the code expressly excludes named choices from tool-call finish promotion. OpenAI SDK v2.26.0's Choice schema documents tool_calls when a tool is called, and Anthropic SDK v0.67.0 documents tool_use; neither makes this named-choice exception. Clients using the terminal reason to continue their tool loop can finish without executing the requested call. Verification: verified by reading both modes and the pinned generated protocol documentation.
13. **High — the parser interprets ungrammared bare function markup as executable tools.** File: `vllm/vllm/parser/qwen3.py:150`; `vllm/vllm/parser/engine/parser_engine_config.py:97`; `vllm/vllm/parser/engine/parser_engine.py:421,1209`. Trigger: auto tools are supplied and, after </think>, generation produces `<function=not_a_tool>\n<parameter=x>\n1\n</parameter>\n</function>` as ordinary text (for example while explaining that syntax). Wrong result: the fallback starts a tool slot without <tool_call>; any nonempty function name is accepted, arguments are not schema-validated, and a normal stop yields an executable tool terminal. Why wrong: the parser's accepted language is broader than the constrained tool language. The pinned XGrammar v0.2.3 comparator triggers only on `<tool_call>\n<function=` and allows this bare markup as free text. Verification: parser behavior verified by reading; decoder reachability inferred from the pinned dependency comparator, since the installed XGrammar package cannot be inspected under this audit's restrictions.
14. **Medium — Responses replay silently drops all but the first content block.** File: `vllm/vllm/entrypoints/openai/responses/utils.py:279,300,329`. Trigger: replay an assistant message with two valid output_text blocks, or a reasoning item with two reasoning_text blocks, in the next Responses input. Wrong result: only element zero reaches the template; the remaining assistant text/reasoning disappears. Why wrong: these protocol fields are arrays, and every supplied block belongs to the history; the converter accepts the item but indexes `[0]` rather than preserving its contents. Verification: verified by reading the typed and dictionary input paths and the pinned OpenAI input/output schemas.
15. **Medium — the current deployment lacks the archive required by both start and status.** File: `config/runtime-v1.sh:28`; `scripts/runtime-common.sh:213`; `scripts/run-agent.sh:16`; `scripts/status-agent.sh:13`. Trigger: invoke the documented start/status command in this supplied workspace. Wrong result: even if all earlier checks pass and the runtime image already exists, `check_pinned_build_inputs` fails because `artifacts/qwen38-vllm-images-runtime-v22.tar` is absent; the documented restore path cannot supply it either. Why wrong: README declares the current deployment complete and healthy, but the mandatory recovery input is missing and lifecycle validation cannot succeed. Verification: verified by reading the mandatory file gate and read-only stat of the exact path; no assertion about a currently running container.

## Audited
- Complete patch transaction framework, patchset assembly, diff compiler, all thirteen semantic contracts, generated identity/stage/final-manifest structure: read; immutable reconstruction checks constrain generated blocks to the review artifacts. No additional concrete transaction defect established.
- Remaining config/CLI/LLM, route removal, completion and token-in-token-out identity diffs: read; identity reaches the engine gate on each mounted generation surface.
- Qwen3.5/Qwen3VL image MRoPE and next-token positions compared to Transformers v5.15.0: normal static-image coordinate construction agrees; no issue established.
- Named-tool finish reasons and Responses replay arrays: issues 12 and 14. Alternate bare-function grammar/parser boundary: issue 13, with dependency-version inference explicitly bounded.

## Dismissed
- Responses TODO says disconnect is unhandled: route uses with_cancellation and StreamingResponse; AsyncLLM.generate aborts on CancelledError/GeneratorExit. The TODO alone is not a defect.
- Responses store-disabled default: upstream deliberately disables memory retention and returns store=false; no enabled-store-only defect reported for the locked profile.
- Reclaimable buffers invalidate decode CUDA graphs: selected decode buffers use the separate persistent workspace; continuation buffers are used outside the selected decode graph path.

## Clones

## Audited
- Entire Docker recipe, complete transactional framework tests, runtime-common contract script, restore path, generated thirteen-stage inventory: read; issue 10 remains the concrete build omission. No scripts executed.
- Selected TurboQuant AoS K8V4 store, split decode, split reduction, bulk dequantization, BF16 continuation workspace and FlashAttention routing: read; no additional numerical/indexing defect established for the pinned single-sequence profile. Hardware/inherited-image guards remain issue 10.
- Context token accounting, phase stop checks, prompt-plus-output clamp and full-context refusal: read; ordinary requests preserve the 262,144-token combined bound. Exact reasoning token accounting and disconnect cleanup reviewed across all three protocols.
- GPU context-capacity calculation: issue 16. CPU full-attention/recurrent window sizing, store/load/pin lifecycle and scope eviction reviewed; issue 9 remains confirmed.
- Issue 15 reference correction: archive declaration is config/runtime-v1.sh:26; mandatory missing-file gate is scripts/runtime-common.sh:207–211 (not the nearby hash-verification line). Missing archive remains verified; v22 adoption comments are not treated as evidence of live deployment health.

## Confirmed issues
16. **Medium — the physical GPU capacity bound omits allocations predating the profiling snapshot.** File: `vllm/vllm/v1/worker/gpu_worker.py:577`; `vllm/vllm/utils/mem_utils.py:317`; `vllm/vllm/v1/core/kv_cache_utils.py:2272`. Trigger: another GPU consumer leaves enough free memory for the default 0.92 startup requirement, but its allocation plus the model's measured non-KV peak and the declared one-context pool exceed total VRAM. For example, about 2.4 GiB pre-existing use is below the startup allowance on the approximately 31.8 GiB card, yet adding it to the documented approximately 29.6 GiB model/peak/cache footprint exceeds the card. Wrong result: initialization's authoritative capacity check accepts the declaration; allocation or a later peak can OOM rather than rejecting the declaration against actual free capacity. Why wrong: `non_kv_cache_memory` uses `before_create.free_memory - after_profile.free_memory` plus transient peak, so prior allocations (also pre-snapshot NCCL) are excluded; subtracting this delta from total_memory counts occupied memory as available. The launch checks only total GPU memory and do not require an otherwise empty device. Verification: omitted-byte accounting and gate reachability verified by reading; the OOM consequence is inferred for the stated capacity inequality, not observed.

## Dismissed
- Empty TurboQuant decode splits read stale scratch: stage 2 checks the same nonempty split bounds and skips unwritten splits.
- Vision feature list order corrupts MRoPE: `_iter_mm_grid_hw` sorts by placeholder offset before coordinate construction.
- Recurrent snapshot retention alone proves loss of full-context capacity: not established for the selected group's ordering and completed-store lifecycle; no additional finding reported from that candidate.
- GPU capacity wrongly charges discretionary utilization reserve: the code deliberately ignores that reserve for pool sizing; issue 16 concerns already occupied bytes, not the intentional reserve policy.

## Clones

## Audited
- Final coverage complete: owner-only requirements history and authoritative README; deployment config and lifecycle; image recipe/build/recovery; all thirteen landmark transformations and patch framework; modified runtime sources and copied units; derived template; Chat Completions, Responses and Anthropic protocol paths; reasoning, grammar, tool round trips, budgets, errors, usage and cancellation; GPU/CPU KV sizing, offload ownership/pinning and selected TurboQuant path; strict vision ingestion, processor/MRoPE and workspace lifecycle.
- Final source-only result: 7 high and 9 medium findings; none critical. Explicit limits remain attached to findings 10 (installed image bytes unavailable), 11/13 (pinned dependency comparators), and 16 (OOM consequence inferred). Archive absence reconfirmed; repository remains unchanged by this audit. No project execution, build, test, service request, listener or model call performed.

## Confirmed issues
- Final report retains issues 1–16 above, ordered by severity. Issue 15 final locations: config/runtime-v1.sh:26; scripts/runtime-common.sh:208; scripts/run-agent.sh:15; scripts/status-agent.sh:12.

## Dismissed
- No remaining in-scope candidate is being held open; unproved numerical/runtime behavior is not reported as observed failure.

## Clones

## Audited
- Follow-up brief re-read in full. Previous goal turn classified as progress (completed first-pass evidence audit); this follow-up audits all thirteen patch guarantees and their callers, including uncovered upstream gaps. Existing issues 1–16 will not be re-reported.
- Current state: main HEAD 3dff26d; changes since first-pass baseline are documentation/comment wording and its input hash only; derived template and all thirteen runtime transformations unchanged. vLLM remains pinned at 9df9b0b0a1816b6d0d0f6ecd0da563cc37fd72f5 with the same 91 tracked changed files and four added tests.
- Patch guarantees re-established from CONTRACTS, review artifacts and current README. Stage 4 intentionally starts the final counter only at explicit reasoning-end; implicit tool XML is deliberately excluded. Stage 5's implicit boundary still needs to agree with the sampler's reasoning-budget state.

## Confirmed issues
17. **High — implicit tool entry does not end the sampler's thinking budget, allowing it to overwrite the tool grammar.** Missing companion logic for stages 4/5. File: `vllm/vllm/v1/sample/thinking_budget_state.py:61,254,564`; `vllm/vllm/v1/sample/sampler.py:404`; `vllm/vllm/v1/worker/gpu_model_runner.py:4704`; `vllm/vllm/v1/core/sched/scheduler.py:1908`. Trigger: a supported lowered thinking budget (e.g. 64), short reasoning followed directly by <tool_call> without </think>, then a tool argument long enough to cross the budget while inside a constrained integer/object/string. Wrong result: the parser and grammar have entered tool content, but ThinkingBudgetStateHolder recognizes only the configured </think> sequence and continues counting tool tokens as thinking; it forces </think> with a 1e9 logit after the grammar has masked that token out. For a grammar position that cannot accept the marker, scheduler acceptance fails and terminates the valid request as FINISHED_ERROR. Why wrong: the new implicit-boundary patch promises the same token ends reasoning and begins schema-constrained tool content, but leaves an independent upstream phase tracker and post-mask forcing unchanged. Verification: tracker omission, mask/forcing order, and rejection-to-error path verified by reading; occurrence on a particular sampled generation is conditional on the concrete stated output prefix, not observed in a model run.

## Dismissed
- No explicit </think> means final-response counter never starts: intentional and expressly documented for the structured tool phase; not a new issue.
- Added token-ID keyword breaks unrelated engine-backed reasoning classes: engine-backed implementations use ParserEngineReasoningAdapter, which was updated with the keyword; no unmatched implementation found.

## Clones
- No additional dependencies downloaded in this follow-up; pinned comparators from the first pass retained.

## Audited
- Stage 2 schema enforcement: decoder constraints do not survive all XML-to-JSON coercions; new issue 18 is independent of issue 2's unresolved references.
- Stages 3/4 defaults and ceilings: Chat/Responses/Anthropic inherit the configured final bound; mounted Completion and token-in-token-out paths remain uncovered (19).
- Stage 8 strict media identity: ordinary Chat/Responses/Anthropic routes reach the gate; token-in-token-out's direct multimodal tracker bypasses it (20).
- Stages 1/10 alternate callers: the shared FP8 guard also runs for MSE keys, violating preserved upstream behavior (21); selected RTX 5090 K8V4 does not hit this refusal.
- Stage 12 identity: regular sampling forwards kv_scope, but Chat/Completion beam-search subrequests discard it (22).

## Confirmed issues
- Issue 17 reference/trigger refinement: the CUDA logit overwrite is thinking_budget_state.py:580, after the natural-end-only lookup at :251–252. A decisive grammar position is the required function header after implicit <tool_call>, with a lowered budget that expires there; the forced </think> cannot satisfy the header. No inference about whether a particular free-form string grammar allows that text is required.
18. **High — valid XML enum values are coerced into schema-invalid JSON.** Missing parser-side companion to stage 2. File: `vllm/vllm/parser/engine/parser_engine.py:262,394`; `vllm/vllm/tool_parsers/utils.py:1007,1080,1115`. Trigger: an auto tool has a property `{"type":["string","null"],"enum":["null"]}`, and generation completes `<parameter=mode>\nnull\n</parameter>`. The only schema-valid value is the string "null". Wrong result: type extraction retains both string and null, and coercion prioritizes null, emitting `{"mode":null}` as a completed executable call. No full-schema check follows. Why wrong: XML erases the string/primitive spelling distinction; validating generated XML does not authorize converting to a value forbidden by enum. This requires no $ref and remains after resolving issue 2. Both streaming argument conversion and batch parsing use the same coercion. Verification: wrong conversion and terminal reachability verified by reading; pinned XGrammar v0.2.3's XML GenerateEnum strips JSON string quotes, supporting the stated generated representation. Installed dependency version remains unverified.
19. **Medium — two mounted generation surfaces bypass the hard final-response ceiling.** Incomplete stages 3/4. File: `vllm/vllm/entrypoints/openai/completion/protocol.py:386,418`; `vllm/vllm/entrypoints/scale_out/token_in_token_out/serving.py:132,210`; `vllm/vllm/sampling_params.py:698`. Trigger: a scoped Completion request with a short valid Qwen prompt and max_tokens:200000, or an equivalent `/inference/v1/generate` request using direct SamplingParams, omits phase budgets and generates an explicit </think> followed by more than 131072 tokens. Wrong result: final_response_token_budget remains None, so the scheduler's hard final bound is absent; Completion also omits the thinking default, and token-in-token-out applies only max_tokens from server defaults (also losing top_p/top_k defaults). Direct token-in-token-out final budgets above the server bound are not clamped. Why wrong: stage 3 promises server defaults on every protocol and clients may lower but never raise/null the final ceiling; these mounted surfaces were included in stage 12 but not in the phase/default propagation. InputProcessor's later generation-config update only merges EOS and does not repair the omission. Verification: parameter flow and absent enforcement verified by reading; the long sampled output is the concrete conditional trigger, not an observed run.
20. **High — token-in-token-out accepts image UUIDs and can reuse another image's processed pixels.** Incomplete stage 8. File: `vllm/vllm/entrypoints/scale_out/token_in_token_out/serving.py:145–164`; `vllm/vllm/entrypoints/chat_utils.py:1554`; `vllm/vllm/multimodal/processing/inputs.py:39–62`; `vllm/vllm/multimodal/processing/processor.py:1520,1608`. Trigger: two valid scoped `/inference/v1/generate` calls supply same-shaped PNGs A and B in content_parts using `{"type":"image_url","url":"data:image/png;base64,...","uuid":"screenshot"}`, with the same UUID and matching image placeholders. Wrong result: this endpoint calls the multimodal parser directly and never invokes the strict content-part gate. Both images get the same hash from UUID plus fixed processor kwargs; with the configured 4 GiB LRU processor cache, B reuses A's processed image while A's entry remains resident. Why wrong: the gate explicitly forbids UUIDs so identity derives from actual PNG bytes; accepting a new screenshot under a stable client label must not silently replace its pixels with an older screenshot. The endpoint also accepts preprocessed features/hashes outside the strict input gate. Verification: mounted route, UUID propagation, hash construction and cached-item reuse verified by reading; no model or image processing run.
21. **Medium — the FP8 hardware refusal also disables retained MSE-key TurboQuant modes.** Stage 10 regression against stage 1's preserved upstream path. File: `vllm/vllm/v1/attention/ops/triton_turboquant_decode.py:25,563`; `vllm/vllm/v1/attention/backends/turboquant_attn.py:1086,1117`. Trigger: the reconstructed source serves a normal decode or continuation on SM 8.0/A100 with `turboquant_k3v4_nc` or `turboquant_4bit_nc` (key_fp8=False). Wrong result: these paths unconditionally call _use_fp8_e4b15, which now raises the K8V4 FP8-support error before launching an MSE kernel. Why wrong: MSE keys use centroids/rotation, not either FP8 key encoding; only the key_fp8 store branch correctly makes the hardware check conditional. Stage 1 expressly preserves this distinct upstream path, and stage 10's incompatible-FP8 rejection need not affect it. Verification: verified by reading every caller and preset definition. Scope: reconstructed patch regression on another supported upstream mode/device, not a failure of the locked RTX 5090 K8V4 profile; installed-kernel omission remains issue 10.
22. **Medium — beam search drops a supplied kv_scope and then rejects its own subrequest.** Incomplete stage 12. File: `vllm/vllm/entrypoints/openai/chat_completion/serving.py:338,363`; `vllm/vllm/entrypoints/openai/completion/serving.py:198`; `vllm/vllm/entrypoints/generate/beam_search/online.py:61`; `vllm/vllm/v1/engine/input_processor.py:51`. Trigger: an otherwise valid non-streaming Chat or Completion request sets use_beam_search:true, n:1, and kv_scope:"agent-A". Wrong result: the route takes to_beam_search_params instead of the patched to_sampling_params; the online beam loop creates fresh SamplingParams without extra_args, so its first engine generation fails with "kv_scope is required" although the caller supplied it. Why wrong: stage 12 promises scope propagation through every generation path; this accepted upstream generation mode was neither adapted nor explicitly rejected as unsupported. Verification: accepted field/branch, missing propagation, and unconditional engine gate verified by reading.

## Dismissed
- Oldest block versus latest aggregate agent activity for eviction: the interleaved-owner unit explicitly expects eviction by the owner of the oldest block; no independent new policy finding added beyond issue 9.
- Vision workspace exception cleanup and mmap registration failure cleanup: encoder release uses finally restoration, and the sole mmap-worker factory cleans the region on constructor failure; no additional normal-profile defect established.

## Audited
- Stage 13 generated-ID accounting, composed/direct parser paths, adapters, suppressed reasoning, all- reasoning truncation, implicit tool exit, batch and streaming usage: the counter's ordinary ID arithmetic is sound; stop-stripped text can still move the parser's apparent boundary (23).
- Stop-budget caller coverage: scheduler checks each appended token, including multiple accepted tokens in one update; no additional final-budget overshoot established. Stop and cancellation do not require resetting a completed request's phase index.
- Transaction framework follow-up: pristine/final classification, review-byte checks, stage/final validation, optimistic commit checks, deletion and rollback paths read; no additional concrete defect established for its disposable-tree contract.

## Confirmed issues
23. **Medium — a stop-stripped real reasoning marker can be rebound to a literal lookalike in earlier reasoning.** Missing scanner logic exposed on the batch path by stage 13. File: `vllm/vllm/v1/engine/detokenizer.py:125–141`; `vllm/vllm/v1/engine/output_processor.py:401–405`; `vllm/vllm/parser/engine/token_id_scanner.py:213,265`; `vllm/vllm/entrypoints/openai/chat_completion/serving.py:984`. Trigger: non-streaming Chat with stop:["HALT</think>"] and generated text `Example: </think> is a delimiter. HALT</think>`, where the first </think> is ordinary-token prose and the last is the real special-token boundary. Wrong result: detokenization removes HALT plus the real marker from text but retains their generated IDs. The batch parser now receives both; reconstructed text no longer matches, and the scanner's rfind binds the real marker's ID to the surviving literal occurrence. It returns ` is a delimiter. ` as final content even though all those tokens preceded the actual reasoning boundary; reasoning usage still counts them as reasoning. Why wrong: stage 13 promises the content split and usage share the actual token-ID boundary, and the scanner explicitly promises literal lookalikes do not steal real anchors. A stream that already emitted the literal as reasoning differs from this batch result. Verification: verified by reading the stop stripping, retained-ID handoff, anchor fallback, and parser transitions for the concrete stated token sequence; no generation observed.

## Dismissed
- Tool parsing disabled invalidates implicit reasoning usage: the skip-tool projection explicitly emits REASONING_END at the implicit boundary and forwards the marker as content, so the counter still matches that transition.
- A multi-token scheduler update skips a final-response delimiter or overshoots the budget: _update_request_with_output appends/checks one token at a time and trims the rest on stopping.

## Clones
- Follow-up used only existing pinned public source; no new clone/download or execution.

## Audited
- 1 — turboquant-k8v4-direct-workspace: direct final-layout dequant, cache/current-token placement, BF16 continuation, split decode, persistent/reclaimable allocation and retained MSE branch reviewed. No new selected-profile layout defect; cross-stage MSE regression is 21.
- 2 — enforce-auto-tool-schema: automatic choice/strictness, structural-tag registration, grammar compilation and batch/stream conversion reviewed. Existing 2/13; new constraint-coercion gap 18.
- 3 — qwen38-agent-defaults-and-thinking: default propagation, always-thinking validation, effort aliases, tool-history validation and all mounted protocol constructors reviewed. Existing 7; new uncovered defaults 19.
- 4 — qwen38-separate-final-response-budget: natural/forced end sequences, per-token scheduler check, min_tokens, total/context cap, phase index lifetime and protocol propagation reviewed. New uncovered callers 19; interaction with independent thinking forcing 17.
- 5 — qwen-implicit-tool-grammar-boundary: direct/composed reasoning parser, content-ID extraction, scheduler grammar boundary/acceptance and sampler tracker reviewed. New missing independent phase-tracker update 17.
- 6 — anthropic-validation-http400: messages/count_tokens, Pydantic/engine/template validation, returned ErrorResponse and generic/streaming error paths reviewed. Existing 6; no additional confirmed issue.
- 7 — tool-truncation-finish-reason: Chat/Responses batch and stream, Anthropic conversion, complete/length/custom-stop/error/cancellation terminals reviewed. Existing 1/4/8/12; no additional confirmed issue from this stage.
- 8 — qwen38-vision-runtime: chronological mixed tool results, all strict-media and override gates/callers, PNG decoding, processor/MRoPE, in-place GELU, workspace release/restore and failure paths reviewed. Existing 5/11; new token-in-token-out identity bypass 20.
- 9 — qwen38-numerical-audits: advertised byte layout, BF16 multi-block/GQA oracle, quantization rounding tolerance and copied numerical acceptance programs reviewed. No additional source-established runtime finding; numerical execution remains prohibited.
- 10 — turboquant-fail-closed-guards: store metadata poisoning, prefill finite check, all shared FP8-helper callers and image installation reviewed. Existing 10; new unintended MSE refusal 21.
- 11 — kv-offload-host-pinning-fail-closed: registration result handling, platform early return, sole registration caller, constructor failure cleanup and transfer shutdown reviewed. No additional confirmed defect.
- 12 — kv-user-count-sizing-and-scope-eviction: GPU/CPU derived sizing and null block, shared/recurrent groups, plain/tiered store callers, ownership/eviction, engine gate, public protocol fields, offline/direct sampling and beam subrequests reviewed. Existing 9/16; new dropped beam identity 22.
- 13 — exact-reasoning-usage: grammar eligibility, direct/composed adapters, batch ID handoff, incremental IDs, aggregate usage, suppression, implicit/explicit/no boundary, truncation and stop-text alignment reviewed. New scanner gap 23; ordinary counter arithmetic has no additional established defect.
- Follow-up coverage complete across review diffs, generated landmark transformations, semantic contracts, reconstructed source, relevant tests/copied units and uncovered upstream callers. Original deployment/build/template/protocol coverage remains applicable because runtime sources have not changed since the first pass.
- Final follow-up result: seven new findings, numbered 17–23: three high (17,18,20), four medium (19,21,22,23), none critical. Findings describe source-established behavior under their stated triggers; no sampled occurrence or live installed-image result is claimed. Issue 21 is specifically a reconstructed-source upstream regression outside the selected RTX 5090 K8V4 mode.
- Main worktree remains unchanged by this audit; pre-existing reconstructed vLLM diff remains 91 tracked files, 3024 insertions and 1733 deletions. No fixes, tests, builds, installs, project execution, model/local-service calls, Docker or listeners performed.

## Confirmed issues
- Final follow-up order by severity: 17, 18, 20; then 19, 21, 22, 23. Issues 1–16 were not re-reported; issue 17's refined trigger/line correction is recorded above.

## Dismissed
- No remaining candidate is held open. Runtime-only numerical behavior, cache residency observations and sampled trajectories were not promoted to observed failures.
