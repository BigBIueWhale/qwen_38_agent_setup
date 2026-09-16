"""Semantic contracts for the pinned vLLM source transformations.

The generated landmark module proves byte identity with the reviewed diffs.
This module independently states the defect, the intended behavioral invariant,
and the condition under which each local change should disappear.  A maintainer
cannot bless a new upstream hash without also satisfying these source-structure
checks and the behavioral suites run by the immutable build.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .framework import (
    PatchRefusedError,
    forbid_text,
    require_python_symbols,
    require_text,
)

State = Mapping[str, str]
Validator = Callable[[State], None]


@dataclass(frozen=True)
class SemanticContract:
    rationale: str
    removal_condition: str
    validate_before: Validator
    validate_after: Validator


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PatchRefusedError(message)


def _source(state: State, path: str, *, label: str) -> str:
    _require(path in state, f"{label}: missing {path}")
    return state[path]


def _parse(state: State, path: str, *, label: str) -> ast.Module:
    source = _source(state, path, label=label)
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise PatchRefusedError(f"{label}: invalid Python in {path}: {exc}") from exc


def _find_symbol(state: State, path: str, qualname: str, *, label: str) -> ast.AST:
    tree = _parse(state, path, label=label)
    found: dict[str, ast.AST] = {}

    def visit(body: Sequence[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, node.name))
                found[name] = node
                visit(node.body, (*parents, node.name))

    visit(tree.body, ())
    _require(qualname in found, f"{label}: missing Python symbol {path}:{qualname}")
    return found[qualname]


def _symbol_source(state: State, path: str, qualname: str, *, label: str) -> str:
    source = _source(state, path, label=label)
    node = _find_symbol(state, path, qualname, label=label)
    segment = ast.get_source_segment(source, node)
    _require(segment is not None, f"{label}: cannot recover source for {qualname}")
    return segment


def _require_in_symbol(
    state: State,
    path: str,
    qualname: str,
    needles: Sequence[str],
    *,
    label: str,
) -> str:
    source = _symbol_source(state, path, qualname, label=label)
    for needle in needles:
        _require(
            needle in source,
            f"{label}: {path}:{qualname} lacks required construct {needle!r}",
        )
    return source


def _require_ordered(
    source: str, needles: Sequence[str], *, label: str, location: str
) -> None:
    cursor = 0
    for needle in needles:
        found = source.find(needle, cursor)
        _require(
            found >= 0,
            f"{label}: {location} lacks ordered construct {needle!r}",
        )
        cursor = found + len(needle)


def _if_node(
    node: ast.AST,
    test: str,
    *,
    label: str,
    location: str,
    contains: str | None = None,
) -> ast.If:
    matches = [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.If)
        and ast.unparse(candidate.test) == test
        and (contains is None or contains in _branch_source(candidate.body))
    ]
    _require(
        len(matches) == 1,
        f"{label}: expected exactly one {test!r} branch in {location}; "
        f"found {len(matches)}",
    )
    return matches[0]


def _branch_source(branch: Sequence[ast.stmt]) -> str:
    return "\n".join(ast.unparse(statement) for statement in branch)


def _validate_turbo_before(state: State) -> None:
    label = "turboquant direct workspace precondition"
    path = "vllm/v1/attention/backends/turboquant_attn.py"
    require_python_symbols(
        state,
        path,
        {
            "TurboQuantAttentionImpl._continuation_prefill": (
                "self",
                "layer",
                "query",
                "key_chunk",
                "val_chunk",
                "kv_cache",
                "block_table",
                "cached_len",
                "seq_len",
                "Pi",
                "centroids",
            )
        },
        label=label,
    )
    source = _require_in_symbol(
        state,
        path,
        "TurboQuantAttentionImpl._continuation_prefill",
        (
            "buf_shape = (1, Hk, alloc_len, D)",
            "k_full = torch.empty(seq_len, Hk, D",
            "v_full = torch.empty(seq_len, Hk, D",
        ),
        label=label,
    )
    _require(
        "full_shape = (full_alloc_len, Hk, D)" not in source,
        f"{label}: upstream already contains a direct final-layout workspace",
    )


def _validate_turbo_after(state: State) -> None:
    label = "turboquant direct workspace result"
    path = "vllm/v1/attention/backends/turboquant_attn.py"
    qualname = "TurboQuantAttentionImpl._continuation_prefill"
    node = _find_symbol(state, path, qualname, label=label)
    source = _require_in_symbol(
        state,
        path,
        qualname,
        (
            "full_shape = (full_alloc_len, Hk, D)",
            "k_cached = k_full_buf[:alloc_len].transpose(0, 1).unsqueeze(0)",
            "v_cached = v_full_buf[:alloc_len].transpose(0, 1).unsqueeze(0)",
            "k_full[cached_len:] = key_chunk",
            "v_full[cached_len:] = val_chunk",
        ),
        label=label,
    )
    direct = _if_node(
        node,
        "self.tq_config.key_fp8",
        label=label,
        location=f"{path}:{qualname}",
        contains="full_shape = (full_alloc_len, Hk, D)",
    )
    direct_source = _branch_source(direct.body)
    _require(
        "full_shape = (full_alloc_len, Hk, D)" in direct_source
        and (
            "get_simultaneous" in direct_source
            or "get_reclaimable_simultaneous" in direct_source
        ),
        f"{label}: K8V4 branch does not acquire the final-layout workspace",
    )
    _require(
        "torch.empty" not in direct_source,
        f"{label}: K8V4 branch still allocates progressive K/V tensors",
    )
    mse = _if_node(
        node,
        "not self.tq_config.key_fp8",
        label=label,
        location=f"{path}:{qualname}",
        contains="torch.empty",
    )
    mse_source = _branch_source(mse.body)
    _require(
        mse_source.count("torch.empty") == 2,
        f"{label}: MSE-key compatibility branch no longer owns two final buffers",
    )
    _require_ordered(
        source,
        (
            "if self.tq_config.key_fp8:",
            "full_shape = (full_alloc_len, Hk, D)",
            "if not self.tq_config.key_fp8:",
            "k_full[cached_len:] = key_chunk",
            "v_full[cached_len:] = val_chunk",
        ),
        label=label,
        location=f"{path}:{qualname}",
    )


def _validate_schema_before(state: State) -> None:
    label = "Qwen automatic-tool schema precondition"
    path = "vllm/tool_parsers/structural_tag_registry.py"
    require_python_symbols(
        state,
        path,
        {"get_model_structural_tag": ("model", "tools", "tool_choice", "reasoning")},
        label=label,
    )
    _require_in_symbol(
        state,
        path,
        "get_model_structural_tag",
        ('if tool_choice == "auto" and not _any_tool_strict(tools):', "return None"),
        label=label,
    )
    forbid_text(state, path, 'model != "qwen_3_coder"', label=label)


def _validate_schema_after(state: State) -> None:
    label = "Qwen automatic-tool schema result"
    path = "vllm/tool_parsers/structural_tag_registry.py"
    source = _require_in_symbol(
        state,
        path,
        "get_model_structural_tag",
        (
            'model != "qwen_3_coder"',
            'if model == "qwen_3_coder":',
            'function.pop("strict", None)',
        ),
        label=label,
    )
    _require_ordered(
        source,
        (
            'model != "qwen_3_coder"',
            "return None",
            "dumped_tools =",
            'if model == "qwen_3_coder":',
            'function.pop("strict", None)',
        ),
        label=label,
        location=f"{path}:get_model_structural_tag",
    )


def _validate_defaults_before(state: State) -> None:
    label = "Qwen3.8 agent defaults precondition"
    chat = "vllm/entrypoints/openai/chat_completion/protocol.py"
    anthropic = "vllm/entrypoints/anthropic/protocol.py"
    require_python_symbols(
        state,
        chat,
        {"ChatCompletionRequest.to_sampling_params": (
            "self",
            "max_tokens",
            "default_sampling_params",
        )},
        label=label,
    )
    forbid_text(state, chat, "_validate_tool_result_correlation", label=label)
    forbid_text(state, anthropic, "class AnthropicThinkingConfig", label=label)


def _validate_defaults_after(state: State) -> None:
    label = "Qwen3.8 agent defaults result"
    model = "vllm/config/model.py"
    anthropic_protocol = "vllm/entrypoints/anthropic/protocol.py"
    anthropic_serving = "vllm/entrypoints/anthropic/serving.py"
    chat = "vllm/entrypoints/openai/chat_completion/protocol.py"
    require_python_symbols(
        state,
        anthropic_protocol,
        {
            "AnthropicOutputConfig.validate_correctness_first_effort": ("self",),
            "AnthropicThinkingConfig.validate_budget_shape": ("self",),
        },
        label=label,
    )
    require_python_symbols(
        state,
        chat,
        {
            "ChatCompletionRequest._validate_tool_result_correlation": ("self",),
            "ChatCompletionRequest.to_sampling_params": (
                "self",
                "max_tokens",
                "default_sampling_params",
            ),
        },
        label=label,
    )
    for needle in (
        '"presence_penalty"',
        '"thinking_token_budget"',
        '"final_response_token_budget"',
    ):
        require_text(state, model, needle, label=label)
    correlation = _symbol_source(
        state, chat, "ChatCompletionRequest._validate_tool_result_correlation", label=label
    )
    if "validate_tool_result_correlation(self.messages)" in correlation:
        correlation = _symbol_source(
            state, "vllm/entrypoints/chat_utils.py",
            "validate_tool_result_correlation", label=label,
        )
    for invariant in (
        "is orphaned",
        "is missing its transport id",
        "repeats transport id",
        "before all results",
        "no complete result sequence",
    ):
        _require(invariant in correlation, f"{label}: missing tool-history gate {invariant!r}")
    sampling = _symbol_source(
        state, chat, "ChatCompletionRequest.to_sampling_params", label=label
    )
    for needle in (
        'default_sampling_params.get("presence_penalty", 0.0)',
        "thinking_token_budget = default_sampling_params.get(",
        '"thinking_token_budget"',
        "final_response_token_budget=final_response_token_budget",
    ):
        _require(needle in sampling, f"{label}: sampling default/clamp missing {needle!r}")
    if "resolve_final_response_token_budget(" in sampling:
        _require_in_symbol(state, "vllm/sampling_params.py",
            "resolve_final_response_token_budget", (
                "validate_final_response_token_budget(requested)",
                "validate_final_response_token_budget(server_budget)",
                "server_budget if requested is None else min(requested, server_budget)",
            ), label=label)
    else:
        _require(
            "min(\n                    final_response_token_budget, server_final_response_budget" in sampling,
            f"{label}: missing final-response ceiling",
        )
    _require(
        "max(\n                    final_response_token_budget" not in sampling,
        f"{label}: client can raise the server final-response ceiling",
    )
    _require_in_symbol(
        state,
        anthropic_serving,
        "AnthropicServingMessages._build_base_request",
        (
            'chat_template_kwargs["enable_thinking"] = True',
            'thinking_kwargs["thinking_token_budget"] = thinking.budget_tokens',
        ),
        label=label,
    )


def _validate_phase_before(state: State) -> None:
    label = "separate final-response budget precondition"
    forbid_text(
        state,
        "vllm/sampling_params.py",
        "validate_final_response_token_budget",
        label=label,
    )
    forbid_text(
        state,
        "vllm/v1/request.py",
        "final_response_start_index",
        label=label,
    )


def _validate_phase_after(state: State) -> None:
    label = "separate final-response budget result"
    sampling = "vllm/sampling_params.py"
    scheduler = "vllm/v1/core/sched/utils.py"
    processor = "vllm/v1/engine/input_processor.py"
    request = "vllm/v1/request.py"
    require_python_symbols(
        state,
        sampling,
        {"validate_final_response_token_budget": ("value",)},
        label=label,
    )
    require_python_symbols(
        state,
        scheduler,
        {"check_stop": None},
        label=label,
    )
    validator = _symbol_source(
        state, sampling, "validate_final_response_token_budget", label=label
    )
    for needle in ("isinstance(value, (bool, float))", "if value == -1:", "if value <= 0:"):
        _require(needle in validator, f"{label}: missing budget validation {needle!r}")
    stop = _symbol_source(state, scheduler, "check_stop", label=label)
    _require_ordered(
        stop,
        (
            "final_budget = sampling_params.final_response_token_budget",
            "if request.final_response_start_index is None:",
            "for end_sequence in sampling_params._reasoning_end_token_sequences:",
            "request.num_output_tokens < sampling_params.min_tokens",
            "request.status = RequestStatus.FINISHED_STOPPED",
            "if final_budget_reached:",
            'request.stop_reason = "final_response_token_budget"',
            "request.num_tokens >= max_model_len",
        ),
        label=label,
        location=f"{scheduler}:check_stop",
    )
    process = _symbol_source(state, processor, "InputProcessor.process_inputs", label=label)
    _require_ordered(
        process,
        (
            "if sampling_params.final_response_token_budget is not None:",
            "reasoning_config.natural_reasoning_end_token_ids",
            "reasoning_config.reasoning_end_token_ids",
            "if not end_sequences:",
            "sampling_params._reasoning_end_token_sequences = end_sequences",
        ),
        label=label,
        location=f"{processor}:InputProcessor.process_inputs",
    )
    require_text(
        state,
        request,
        "self.final_response_start_index: int | None = None",
        label=label,
    )


def _validate_grammar_before(state: State) -> None:
    label = "implicit Qwen tool-boundary precondition"
    forbid_text(
        state,
        "vllm/parser/qwen3.py",
        "def extract_content_ids(self, input_ids",
        label=label,
    )
    forbid_text(
        state,
        "vllm/v1/structured_output/__init__.py",
        "return idx - len(content_ids)",
        label=label,
    )


def _validate_grammar_after(state: State) -> None:
    label = "implicit Qwen tool-boundary result"
    parser = "vllm/parser/qwen3.py"
    structured = "vllm/v1/structured_output/__init__.py"
    require_python_symbols(
        state,
        parser,
        {"Qwen3Parser.extract_content_ids": ("self", "input_ids")},
        label=label,
    )
    require_python_symbols(
        state,
        structured,
        {"StructuredOutputManager._find_reasoning_end_index": (
            "reasoner",
            "all_token_ids",
            "start",
        )},
        label=label,
    )
    parser_source = _symbol_source(
        state, parser, "Qwen3Parser.extract_content_ids", label=label
    )
    for needle in (
        "content_ids = super().extract_content_ids(input_ids)",
        "for i in range(len(input_ids) - 1, -1, -1):",
        "return input_ids[i:]",
        "return input_ids",
    ):
        _require(needle in parser_source, f"{label}: missing Qwen boundary rule {needle!r}")
    manager = _symbol_source(
        state,
        structured,
        "StructuredOutputManager._find_reasoning_end_index",
        label=label,
    )
    _require_ordered(
        manager,
        (
            "if reasoner.is_reasoning_end_streaming(prefix, [token]):",
            'getattr(reasoner, "extract_content_ids", None)',
            "return idx - len(content_ids)",
            "return idx",
        ),
        label=label,
        location=f"{structured}:StructuredOutputManager._find_reasoning_end_index",
    )


def _validate_anthropic_400_before(state: State) -> None:
    label = "Anthropic validation status precondition"
    path = "vllm/entrypoints/anthropic/api_router.py"
    source = _source(state, path, label=label)
    _require("except ValidationError as e:" not in source, f"{label}: fix already present")
    _require("VLLMValidationError" not in source, f"{label}: fix already present")
    _require(source.count("except Exception as e:") >= 2, f"{label}: generic handlers drifted")


def _validate_anthropic_400_after(state: State) -> None:
    label = "Anthropic validation status result"
    path = "vllm/entrypoints/anthropic/api_router.py"
    require_python_symbols(
        state,
        path,
        {
            "create_messages": ("request", "raw_request"),
            "count_tokens": ("request", "raw_request"),
        },
        label=label,
    )
    # The later fidelity stage routes these same validation failures through
    # the shared exception classifier. Reassert that stronger contract when
    # validating the final tree; the initial stage still has its local catches.
    if "def refuse(" in _source(state, path, label=label):
        _validate_anthropic_inputs_after(state)
        return
    # The engine's own request validation -- a generation that names no agent,
    # for one -- is a client error exactly as typed request validation is, and
    # it reaches this router as VLLMValidationError rather than pydantic's.
    require_text(
        state,
        path,
        "from vllm.exceptions import VLLMValidationError",
        count=1,
        label=label,
    )
    for qualname in ("create_messages", "count_tokens"):
        source = _symbol_source(state, path, qualname, label=label)
        _require_ordered(
            source,
            (
                "except (ValidationError, VLLMValidationError) as e:",
                "status_code=HTTPStatus.BAD_REQUEST.value",
                'type="invalid_request_error"',
                "message=sanitize_message(str(e))",
                "except Exception as e:",
                "status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value",
            ),
            label=label,
            location=f"{path}:{qualname}",
        )


def _validate_truncation_before(state: State) -> None:
    label = "truncated tool-call precondition"
    path = "vllm/entrypoints/openai/chat_completion/serving.py"
    source = _source(state, path, label=label)
    _require(
        "if tools_streamed[i] and not tool_choice_function_name:" in source,
        f"{label}: legacy stream promotion landmark missing",
    )
    forbid_text(
        state,
        "vllm/entrypoints/openai/responses/streaming_events.py",
        "incomplete: bool = False",
        label=label,
    )


def _validate_truncation_after(state: State) -> None:
    label = "truncated tool-call result"
    chat = "vllm/entrypoints/openai/chat_completion/serving.py"
    responses_protocol = "vllm/entrypoints/openai/responses/protocol.py"
    responses_serving = "vllm/entrypoints/openai/responses/serving.py"
    events = "vllm/entrypoints/openai/responses/streaming_events.py"
    utils = "vllm/entrypoints/openai/responses/utils.py"
    parser = "vllm/parser/engine/parser_engine.py"
    require_text(
        state,
        chat,
        'output.finish_reason == "stop"',
        count=2,
        label=label,
    )
    require_python_symbols(
        state,
        events,
        {
            "emit_simple_tool_call_done": ("state", "incomplete"),
            "SimpleStreamingEventProcessor.close_current": ("self", "incomplete"),
        },
        label=label,
    )
    require_python_symbols(
        state,
        utils,
        {"build_response_output_items": (
            "reasoning",
            "content",
            "tool_calls",
            "logprobs",
            "tools",
            "incomplete",
        )},
        label=label,
    )
    event_source = _symbol_source(state, events, "emit_simple_tool_call_done", label=label)
    _require_ordered(
        event_source,
        (
            "if state.has_emitted_tool_call_delta and not incomplete:",
            "ResponseFunctionCallArgumentsDoneEvent",
            'status="incomplete" if incomplete else "completed"',
        ),
        label=label,
        location=f"{events}:emit_simple_tool_call_done",
    )
    for path, needles in (
        (
            responses_protocol,
            ("class ResponseIncompleteEvent", "| ResponseIncompleteEvent"),
        ),
        (
            responses_serving,
            (
                'incomplete=final_output.finish_reason == "length"',
                'incomplete=final_finish_reason == "length"',
                'type="response.incomplete"',
            ),
        ),
    ):
        for needle in needles:
            require_text(state, path, needle, label=label)
    require_text(
        state,
        utils,
        'status="incomplete" if incomplete else "completed"',
        count=2,
        label=label,
    )
    parser_terminal = ast.unparse(
        _find_symbol(
            state,
            parser,
            "ParserEngine._events_to_delta",
            label=label,
        )
    )
    if "earlier_deferred" in parser_terminal:
        _require(
            "if finished or not seen_tool_event or (not tool_call_deltas):"
            in parser_terminal
            and "content_parts.insert(0, earlier_deferred)" in parser_terminal
            and "content_parts.append(deferred_after_call)" in parser_terminal,
            f"{label}: terminal parse must release all text in generation order",
        )
    else:
        _require(
            "self._deferred_content and (finished or not seen_tool_event or "
            "(not tool_call_deltas))" in parser_terminal,
            f"{label}: batch terminal parse can strand deferred content",
        )


def _validate_vision_before(state: State) -> None:
    label = "Qwen3.8 vision runtime precondition"
    _require(
        "tests/v1/worker/test_workspace.py" not in state,
        f"{label}: new workspace test unexpectedly exists",
    )
    forbid_text(
        state,
        "vllm/envs.py",
        "VLLM_QWEN38_STRICT_IMAGE_CONTRACT",
        label=label,
    )
    require_text(
        state,
        "vllm/entrypoints/anthropic/serving.py",
        "tool_image_urls: list[str] = []",
        label=label,
    )
    forbid_text(
        state,
        "vllm/v1/worker/workspace.py",
        "get_reclaimable_simultaneous",
        label=label,
    )


def _validate_vision_after(state: State) -> None:
    label = "Qwen3.8 vision runtime result"
    anthropic = "vllm/entrypoints/anthropic/serving.py"
    chat = "vllm/entrypoints/chat_utils.py"
    connector = "vllm/multimodal/media/connector.py"
    image = "vllm/multimodal/media/image.py"
    render = "vllm/renderers/params.py"
    vision_model = "vllm/model_executor/models/qwen3_vl.py"
    turbo = "vllm/v1/attention/backends/turboquant_attn.py"
    runner = "vllm/v1/worker/gpu_model_runner.py"
    workspace = "vllm/v1/worker/workspace.py"

    require_python_symbols(
        state,
        anthropic,
        {"AnthropicServingMessages._convert_user_tool_result": (
            "cls",
            "block",
            "openai_messages",
        )},
        label=label,
    )
    conversion = _symbol_source(
        state, anthropic, "AnthropicServingMessages._convert_user_tool_result", label=label
    )
    for needle in (
        "tool_content_parts",
        '"content": tool_content_parts if has_image else (tool_text or "")',
    ):
        _require(needle in conversion, f"{label}: tool media chronology missing {needle!r}")
    _require(
        '"role": "user"' not in conversion,
        f"{label}: tool-result image is still detached into a synthetic user turn",
    )

    require_python_symbols(
        state,
        chat,
        {"_enforce_qwen38_strict_content_part": ("part",)},
        label=label,
    )
    strict_part = _symbol_source(
        state, chat, "_enforce_qwen38_strict_content_part", label=label
    )
    for needle in (
        "if not envs.VLLM_QWEN38_STRICT_IMAGE_CONTRACT:",
        "if part.get(\"uuid\") is not None:",
        'if part_type == "image_url":',
        'detail not in ("auto", "high")',
        'elif part_type == "input_image":',
    ):
        _require(needle in strict_part, f"{label}: pre-I/O media gate missing {needle!r}")

    for qualname in ("MediaConnector.fetch_image", "MediaConnector.fetch_image_async"):
        connector_source = _symbol_source(state, connector, qualname, label=label)
        _require(
            'not image_url.startswith(\n            "data:image/png;base64,"' in connector_source,
            f"{label}: {qualname} does not reject non-canonical URLs before I/O",
        )

    image_source = _symbol_source(state, image, "ImageMediaIO.load_bytes", label=label)
    for needle in (
        "required_max_pixels = 16_777_216",
        'if image.format != "PNG":',
        'image.mode not in ("RGB", "RGBA")',
        "QWEN38_MAX_PROVEN_IMAGE_ASPECT_RATIO",
        '"transparency" in image.info',
    ):
        _require(needle in image_source, f"{label}: decoded-image gate missing {needle!r}")
    require_text(
        state,
        image,
        "QWEN38_MAX_PROVEN_IMAGE_ASPECT_RATIO = 30",
        label=label,
    )
    render_source = _symbol_source(state, render, "ChatParams.with_defaults", label=label)
    for needle in ("if self.media_io_kwargs:", "if self.mm_processor_kwargs:", '"add_vision_id"'):
        _require(needle in render_source, f"{label}: request override gate missing {needle!r}")

    mlp = _symbol_source(state, vision_model, "Qwen3_VisionMLP.forward", label=label)
    _require_ordered(
        mlp,
        (
            "hidden_states = self.linear_fc1(x)",
            "if self._use_inplace_gelu_tanh and not torch.is_grad_enabled():",
            "torch.ops.aten.gelu.out",
            "else:",
            "hidden_states = self.act_fn(hidden_states)",
            "mlp_output = self.linear_fc2(hidden_states)",
        ),
        label=label,
        location=f"{vision_model}:Qwen3_VisionMLP.forward",
    )

    for needle in (
        '_CONTINUATION_WORKSPACE_NAME = "turboquant_continuation_prefill"',
        '_VISION_HEADROOM_WORKSPACE_NAME = "qwen38_vision_encoder_headroom"',
        "reserve_raw_cuda_headroom",
    ):
        require_text(state, turbo, needle, label=label)
    require_text(
        state,
        turbo,
        "get_reclaimable_simultaneous",
        count=3,
        label=label,
    )
    runner_source = _symbol_source(state, runner, "GPUModelRunner._execute_mm_encoder", label=label)
    _require_ordered(
        runner_source,
        (
            "if not scheduler_output.scheduled_encoder_inputs:",
            "return []",
            "with release_reclaimable_workspaces():",
            "return self._execute_mm_encoder_with_released_workspace",
        ),
        label=label,
        location=f"{runner}:GPUModelRunner._execute_mm_encoder",
    )

    require_python_symbols(
        state,
        workspace,
        {
            "WorkspaceManager.get_reclaimable_simultaneous": (
                "self",
                "name",
                "*shapes_and_dtypes",
            ),
            "WorkspaceManager.reserve_raw_cuda_headroom": ("self", "name", "size"),
            "WorkspaceManager.release_reclaimable_workspaces": ("self",),
            "WorkspaceManager.restore_reclaimable_workspaces": ("self",),
            "release_reclaimable_workspaces": (),
        },
        label=label,
    )
    context = _symbol_source(state, workspace, "release_reclaimable_workspaces", label=label)
    _require_ordered(
        context,
        (
            "released_bytes = manager.release_reclaimable_workspaces()",
            "try:",
            "yield released_bytes",
            "finally:",
            "manager.restore_reclaimable_workspaces()",
        ),
        label=label,
        location=f"{workspace}:release_reclaimable_workspaces",
    )
    for needle, count in (
        ('ctypes.CDLL("libcudart.so.13")', 1),
        ("runtime.cudaMemset(pointer, 0, size)", 1),
        ("runtime.cudaDeviceSynchronize()", 1),
        ("if self._reclaimable_workspaces_released:", 3),
        ("self._reclaimable_workspaces_released = True", 1),
        ("self._reclaimable_workspaces_released = False", 2),
    ):
        require_text(state, workspace, needle, count=count, label=label)

    test_contracts = {
        "tests/entrypoints/unit_tests/test_chat_utils.py": (
            "test_qwen38_strict_content_part_contract",
            "test_qwen38_strict_request_overrides_fail_closed",
        ),
        "tests/multimodal/media/test_image.py": (
            "test_qwen38_strict_image_contract_accepts_rgb_and_rgba",
            "test_qwen38_strict_image_contract_rejects_ambiguous_inputs",
        ),
        "tests/v1/worker/test_gpu_model_runner_mm_gather.py": (
            "test_text_step_does_not_release_reclaimable_workspace",
            "test_encoder_step_releases_workspace_only_around_encoder",
        ),
        "tests/v1/worker/test_workspace.py": (
            "test_reclaimable_workspace_release_preserves_primary",
            "test_reclaimable_workspace_context_restores_after_error",
            "test_nested_reclaimable_workspace_release_fails_closed",
            "test_raw_cuda_headroom_is_physically_freed_and_restored",
        ),
    }
    for path, symbols in test_contracts.items():
        require_python_symbols(
            state,
            path,
            {symbol: None for symbol in symbols},
            label=label,
        )


def _validate_numerical_audits_before(state: State) -> None:
    label = "Qwen3.8 numerical-audit precondition"
    backend = "vllm/v1/attention/backends/turboquant_attn.py"
    test = "tests/quantization/test_turboquant.py"
    require_text(
        state,
        backend,
        "unpack FP16 values, softmax + weighted sum",
        label=label,
    )
    require_text(
        state,
        backend,
        "For turboquant_k3v4_nc head_dim=256: "
        "[100 bytes key | 512 bytes value] = 612",
        label=label,
    )
    forbid_text(
        state,
        test,
        "test_qwen38_k8v4_bf16_gqa_matches_packed_reference",
        label=label,
    )


def _validate_numerical_audits_after(state: State) -> None:
    label = "Qwen3.8 numerical-audit result"
    backend = "vllm/v1/attention/backends/turboquant_attn.py"
    test = "tests/quantization/test_turboquant.py"
    for needle in (
        "dequantize packed V",
        "float32 online softmax and value accumulation",
        "key_packed_size + value_packed_size",
        "256-byte E4M3 key | 128-byte 4-bit V",
        "= 388 bytes per KV head/token",
    ):
        require_text(state, backend, needle, label=label)
    require_python_symbols(
        state,
        test,
        {
            "TestStoreDecodeRoundTrip."
            "test_qwen38_k8v4_bf16_gqa_matches_packed_reference": ("self",)
        },
        label=label,
    )
    source = _symbol_source(
        state,
        test,
        "TestStoreDecodeRoundTrip."
        "test_qwen38_k8v4_bf16_gqa_matches_packed_reference",
        label=label,
    )
    for needle in (
        'preset = "turboquant_k8v4"',
        "d = 256",
        "num_kv_heads = 4",
        "num_query_heads = 24",
        "dtype=torch.bfloat16",
        "assert cfg.slot_size == 388",
        "actual_codes_cpu[..., 0::2]",
        "boundary_distance > 2e-5",
        "torch.softmax(scores, dim=-1)",
        "assert similarities.min().item() > 0.999",
    ):
        _require(needle in source, f"{label}: numerical oracle lacks {needle!r}")


def _validate_tq_guards_before(state: State) -> None:
    label = "turboquant fail-closed guards precondition"
    store = "vllm/v1/attention/ops/triton_turboquant_store.py"
    decode = "vllm/v1/attention/ops/triton_turboquant_decode.py"
    backend = "vllm/v1/attention/backends/turboquant_attn.py"
    forbid_text(state, store, "meta_ok", label=label)
    forbid_text(state, decode, "Refusing a silent fp8e4b15", label=label)
    forbid_text(
        state, backend, "received non-finite K/V activations", label=label
    )


def _validate_tq_guards_after(state: State) -> None:
    label = "turboquant fail-closed guards result"
    store = "vllm/v1/attention/ops/triton_turboquant_store.py"
    decode = "vllm/v1/attention/ops/triton_turboquant_decode.py"
    backend = "vllm/v1/attention/backends/turboquant_attn.py"
    # Insane value vectors poison both metadata fields with propagating
    # NaN instead of laundering into finite codes; the predicate covers
    # NaN inputs and fp16 metadata overflow and is bit-neutral otherwise.
    require_text(
        state, store, "sc_f16 = tl.where(meta_ok, v_scale, poison)", label=label
    )
    require_text(
        state, store, "zr_f16 = tl.where(meta_ok, val_min, poison)", label=label
    )
    require_text(
        state, store, "((val_max - val_min) < 982560.0)", label=label
    )
    require_text(
        state, store,
        "tl.sum((d_mask & (val_vec != val_vec)).to(tl.int32), axis=0) == 0",
        label=label,
    )
    # The stored-key byte contract is E4M3: SM < 8.9 refuses instead of a
    # silent fp8e4b15 format switch.
    require_text(state, decode, "Refusing a silent fp8e4b15", label=label)
    forbid_text(state, decode, "1 if cap < (8, 9) else 0", label=label)
    # Prefill chunks fail closed on non-finite activations before storing;
    # the per-token decode path deliberately relies on kernel poisoning.
    require_text(
        state,
        backend,
        "turboquant store received non-finite K/V activations",
        label=label,
    )


def _validate_kv_pin_before(state: State) -> None:
    label = "KV offload host pinning precondition"
    worker = "vllm/v1/kv_offload/cpu/gpu_worker.py"
    forbid_text(state, worker, "Continuing would serve every offloaded KV", label=label)
    require_text(
        state,
        worker,
        "transfers will still work but may be slower (unpinned DMA)",
        label=label,
    )


def _validate_kv_pin_after(state: State) -> None:
    label = "KV offload host pinning result"
    worker = "vllm/v1/kv_offload/cpu/gpu_worker.py"
    # A failed registration is fatal, not a downgrade to unpinned DMA.
    require_text(state, worker, "raise RuntimeError(", label=label)
    require_text(
        state, worker, "Continuing would serve every offloaded KV", label=label
    )
    forbid_text(
        state,
        worker,
        "transfers will still work but may be slower (unpinned DMA)",
        label=label,
    )
    # The platform check above it is a capability test, not a failure, and
    # must keep returning quietly on non-CUDA builds.
    require_text(
        state,
        worker,
        "cudaHostRegister is only ",
        label=label,
    )


def _validate_shared_prefix_cache_before(state: State) -> None:
    label = "shared prefix cache precondition"
    spec = "vllm/v1/kv_offload/cpu/spec.py"
    manager = "vllm/v1/kv_offload/cpu/manager.py"
    cache = "vllm/config/cache.py"
    # The byte-denominated and policy-selecting world this stage replaces.
    require_text(
        state,
        spec,
        'cpu_bytes_to_use must be specified in kv_connector_extra_config',
        label=label,
    )
    require_text(
        state, spec, 'self.extra_config.get("eviction_policy", "lru")', label=label
    )
    require_text(state, manager, "CachePolicyFactory", count=3, label=label)
    require_text(state, cache, "kv_offloading_size: float | None = None", label=label)
    require_text(
        state, "vllm/v1/kv_offload/cpu/policies/factory.py", '"arc"', label=label
    )
    for path in (
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "vllm/entrypoints/openai/completion/protocol.py",
        "vllm/entrypoints/openai/responses/protocol.py",
        "vllm/entrypoints/anthropic/protocol.py",
        "vllm/entrypoints/scale_out/token_in_token_out/protocol.py",
    ):
        forbid_text(state, path, "kv_scope", label=label)
    generate_router = "vllm/entrypoints/generate/api_router.py"
    require_text(
        state, generate_router, "register_cohere_api_router(app)", label=label
    )
    require_text(
        state, generate_router, "state.cohere_serving_chat_v2", label=label
    )
    require_text(
        state,
        "vllm/entrypoints/openai/cli_args.py",
        "cohere_is_reasoning_model: bool = True",
        label=label,
    )


def _validate_shared_prefix_cache_after(state: State) -> None:
    label = "shared prefix cache result"
    spec = "vllm/v1/kv_offload/cpu/spec.py"
    manager = "vllm/v1/kv_offload/cpu/manager.py"
    cache = "vllm/config/cache.py"
    kv_utils = "vllm/v1/core/kv_cache_utils.py"

    for path in (
        "vllm/v1/kv_offload/cpu/policies/__init__.py",
        "vllm/v1/kv_offload/cpu/policies/base.py",
        "vllm/v1/kv_offload/cpu/policies/factory.py",
        "vllm/v1/kv_offload/cpu/policies/lru.py",
        "vllm/v1/kv_offload/cpu/policies/arc.py",
        "tests/v1/kv_offload/cpu/policies/__init__.py",
        "tests/v1/kv_offload/cpu/policies/test_factory.py",
    ):
        _require(path not in state, f"{label}: deleted policy module is present: {path}")
    require_text(state, spec, "cpu_kv_cache_users must be specified", label=label)
    require_text(state, spec, "Unknown kv_connector_extra_config keys", label=label)
    require_text(state, spec, "self.num_blocks = cpu_kv_cache_users * chunks_per_user", label=label)
    require_python_symbols(state, manager, {
        "CPUOffloadingManager.__init__": ["self", "num_blocks", "enable_events"],
    }, label=label)

    membership = "vllm/v1/core/prefix_cache.py"
    for text in (
        "class PrefixCacheView:",
        "return self.keys is None or self.keys.issuperset(content)",
        "if agent_id not in self._owned:",
        "return PrefixCacheView(None)",
        "return PrefixCacheView(frozenset(owned))",
        "def extend_content(",
        "entry.agents[agent_id] = acquired",
        "def copy_membership(",
        "def copy_content_memberships(",
    ):
        require_text(state, membership, text, label=label)
    require_text(state, membership, "len(state.agents) + added <= state.capacity", count=2, label=label)
    for text in (
        "self.prefix_cache = index",
        "req_context.set_state(self.prefix_cache.view(self._agent_id(req_context)))",
        "self._references.setdefault(key, set()).add(req_id)",
        "not required or not self._has_content(key, required, req_context)",
        "self._references.get(key, set()) <= released_requests",
        "if victim == agent_id:",
        "self.prefix_cache.release_agent(self.cache_tier, victim)",
        "if self._blocks[key].ref_cnt != 0:",
        "self.prefix_cache.extend_content(self.cache_tier, key, content[key])",
        "self.prefix_cache.copy_content_memberships(",
        "        if key not in self._complete_blocks:",
    ):
        require_text(state, manager, text, label=label)
    require_text(state, "vllm/v1/core/kv_cache_manager.py",
                 "self.block_pool.prefix_cache.view(request.kv_scope)", count=2, label=label)
    require_text(state, "vllm/v1/simple_kv_offload/manager.py",
                 "copy_membership(", label=label)
    tiering = "vllm/v1/kv_offload/tiering/manager.py"
    require_text(state, tiering, "if success and complete_keys:", label=label)
    require_text(state, tiering, "self.primary_tier.begin_lookup(req_context)", label=label)

    # GPU tier: the byte flag is gone, the count is required, and the pool
    # is derived rather than filled to whatever memory happened to be free.
    require_text(state, cache, "kv_cache_users: int | None = None", label=label)
    forbid_text(state, cache, "kv_offloading_size", label=label)
    require_text(
        state,
        "vllm/engine/arg_utils.py",
        '"--kv-cache-users", **cache_kwargs["kv_cache_users"]',
        label=label,
    )
    forbid_text(state, "vllm/engine/arg_utils.py", "kv-cache-memory-bytes", label=label)
    forbid_text(state, "vllm/engine/arg_utils.py", "kv_offloading_size", label=label)
    require_text(
        state, kv_utils, "needed_blocks = users * per_user_blocks + 1", label=label
    )
    require_text(
        state, kv_utils, "--kv-cache-users was not", label=label
    )
    require_text(
        state,
        kv_utils,
        "num_gpu_blocks_override cannot be combined",
        label=label,
    )
    forbid_text(state, "vllm/config/vllm.py", "cpu_bytes_to_use", label=label)

    # Every generation surface carries the same opaque identity through
    # the common sampling-parameter channel.
    # Six identity surfaces over five modules: the chat module carries both
    # the single-conversation request and the batch, and a batch is one
    # caller, so its conversations are submitted under the one agent.
    for path, surfaces in (
        ("vllm/entrypoints/openai/chat_completion/protocol.py", 2),
        ("vllm/entrypoints/openai/completion/protocol.py", 1),
        ("vllm/entrypoints/openai/responses/protocol.py", 1),
        ("vllm/entrypoints/anthropic/protocol.py", 1),
        ("vllm/entrypoints/scale_out/token_in_token_out/protocol.py", 1),
    ):
        require_text(
            state, path, "kv_scope: str | None = Field(", count=surfaces, label=label
        )
    require_text(
        state,
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "return ChatCompletionRequest.model_validate(data)",
        label=label,
    )
    generate_router = "vllm/entrypoints/generate/api_router.py"
    forbid_text(state, generate_router, "register_cohere_api_router", label=label)
    forbid_text(state, generate_router, "CohereServingChatV2", label=label)
    forbid_text(state, generate_router, "cohere_serving_chat_v2", label=label)
    # The Cohere-model renderer format is a tokenizer-mode feature, not the
    # HTTP endpoint; it must survive the endpoint excision.
    require_text(state, generate_router, '"cohere_format"', count=3, label=label)
    forbid_text(
        state,
        "vllm/entrypoints/openai/cli_args.py",
        "cohere_is_reasoning_model",
        label=label,
    )
    require_text(
        state,
        "vllm/entrypoints/openai/cli_args.py",
        'cohere_format: str = "cmd4"',
        label=label,
    )
    # Generation requires an agent ID, enforced on the path into the
    # engine rather than on the request models -- the render endpoints share
    # those models and allocate nothing, so a model-level requirement would
    # demand an identity from a caller that owns no context.
    input_processor = "vllm/v1/engine/input_processor.py"
    require_text(
        state, input_processor, "def require_kv_scope(params: SamplingParams) -> str:",
        label=label,
    )
    require_text(state, input_processor, "require_kv_scope(params)", count=1, label=label)
    _require_ordered(
        _source(state, input_processor, label=label),
        (
            'scope = params.extra_args.get("kv_scope") if params.extra_args else None',
            "if scope is None:",
            "kv_scope is required for generation",
            'parameter="kv_scope",',
        ),
        label=label,
        location=input_processor,
    )
    # No surface may be mounted that reaches the engine without being able to
    # name an agent; /generative_scoring is excised for that reason, exactly
    # as the Cohere surface is.
    forbid_text(
        state, generate_router, "register_generative_scoring_api_router", label=label
    )
    forbid_text(state, generate_router, "ServingGenerativeScoring", label=label)
    scheduler = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    require_text(
        state, "vllm/v1/request.py",
        'self.kv_scope = sampling_params.extra_args.get("kv_scope")',
        label=label,
    )
    require_text(
        state,
        "vllm/v1/kv_offload/base.py",
        "kv_scope: str | None = None",
        label=label,
    )
    # The window classification is derived once at the offloading boundary.
    require_text(
        state,
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
        "def get_sliding_window_size_in_chunks(",
        label=label,
    )
    forbid_text(
        state, scheduler, "def get_sliding_window_size_in_chunks(", label=label
    )


def _validate_reasoning_usage_before(state: State) -> None:
    label = "exact reasoning usage precondition"
    parser = "vllm/parser/engine/parser_engine.py"
    abstract = "vllm/parser/abstract_parser.py"
    adapters = "vllm/parser/engine/adapters.py"
    protocol = "vllm/entrypoints/openai/engine/protocol.py"
    chat = "vllm/entrypoints/openai/chat_completion/serving.py"
    # The counter this stage replaces: a depth count between a generated
    # <think> and </think>, which is zero for a prompt that pre-fills the
    # opener and blind to Qwen's implicit <tool_call> end.
    require_text(
        state,
        parser,
        "if token_id == start_id:\n                depth += 1",
        label=label,
    )
    forbid_text(state, parser, "_resolve_reasoning_boundary", label=label)
    forbid_text(state, parser, "reasoning_token_count", label=label)
    # The batch split drops the generated ids it is handed.
    require_text(
        state,
        abstract,
        "reasoning, content = self.extract_reasoning(model_output, request)",
        label=label,
    )
    forbid_text(state, abstract, "generated_token_count", label=label)
    forbid_text(state, adapters, "batch_token_ids", label=label)
    forbid_text(state, protocol, "CompletionTokenUsageInfo", label=label)
    forbid_text(state, chat, "completion_tokens_details", label=label)
    _require(
        "tests/parser/engine/test_reasoning_token_count.py" not in state,
        f"{label}: new reasoning-count test unexpectedly exists",
    )


def _validate_reasoning_usage_after(state: State) -> None:
    label = "exact reasoning usage result"
    parser = "vllm/parser/engine/parser_engine.py"
    abstract = "vllm/parser/abstract_parser.py"
    adapters = "vllm/parser/engine/adapters.py"
    protocol = "vllm/entrypoints/openai/engine/protocol.py"
    chat = "vllm/entrypoints/openai/chat_completion/serving.py"
    test = "tests/parser/engine/test_reasoning_token_count.py"

    # One boundary, resolved from the grammar; one count, advanced on every
    # feed; the depth counter gone rather than kept beside it.
    require_python_symbols(
        state,
        parser,
        {
            "ParserEngine._resolve_reasoning_boundary": ("self",),
            "ParserEngine._account_reasoning_tokens": (
                "self",
                "delta_text",
                "delta_token_ids",
            ),
            "ParserEngine.batch_token_ids": ("self", "token_ids"),
            "ParserEngine.reasoning_token_count": ("self",),
            "ParserEngine.count_reasoning_tokens": ("self", "token_ids"),
        },
        label=label,
    )
    forbid_text(state, parser, "if token_id == start_id:", label=label)
    forbid_text(state, parser, "if depth > 0:", label=label)
    boundary = _symbol_source(
        state, parser, "ParserEngine._resolve_reasoning_boundary", label=label
    )
    _require_ordered(
        boundary,
        (
            "transition.next_state is ParserState.REASONING",
            "re-enters reasoning",
            "leaves = transition.next_state is not ParserState.REASONING",
            "announces = EventType.REASONING_END in transition.events",
            "if leaves != announces:",
            "not a token id in this vocabulary",
        ),
        label=label,
        location=f"{parser}:ParserEngine._resolve_reasoning_boundary",
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine._feed",
        ("self._account_reasoning_tokens(delta_text, delta_token_ids)",),
        label=label,
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine._account_reasoning_tokens",
        (
            "self._reasoning_fed_without_ids = True",
            "if token_id in self._reasoning_boundary_ids:",
        ),
        label=label,
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine.extract_reasoning",
        ("self._feed(model_output, self._batch_token_ids)",),
        label=label,
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine.reasoning_token_count",
        (
            "if self._reasoning_boundary_refusal is not None:",
            "return None",
            "if self._reasoning_fed_without_ids:",
            "raise ValueError(",
        ),
        label=label,
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine.parse_delta",
        ("self._stream_state.generated_token_count += len(delta_token_ids)",),
        label=label,
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine.parse",
        ("self._stream_state.generated_token_count = len(model_output_token_ids)",),
        label=label,
    )

    # The composed parser the endpoint runs: ids reach the reasoning engine
    # on the batch path too, every delta is counted, and the count is read
    # through one property on every Parser.
    require_text(
        state, abstract, "    generated_token_count: int = 0\n", label=label
    )
    require_python_symbols(
        state,
        abstract,
        {
            "Parser.reasoning_token_count": ("self",),
            "Parser.generated_token_count": ("self",),
            "DelegatingParser.extract_reasoning": (
                "self",
                "model_output",
                "request",
                "model_output_token_ids",
            ),
            "DelegatingParser.reasoning_token_count": ("self",),
        },
        label=label,
    )
    forbid_text(
        state,
        abstract,
        "reasoning, content = self.extract_reasoning(model_output, request)",
        label=label,
    )
    _require_in_symbol(
        state,
        abstract,
        "DelegatingParser.parse",
        (
            "self._stream_state.generated_token_count = len(model_output_token_ids)",
            "model_output, request, model_output_token_ids",
        ),
        label=label,
    )
    _require_in_symbol(
        state,
        abstract,
        "DelegatingParser.parse_delta",
        ("state.generated_token_count += len(delta_token_ids)",),
        label=label,
    )
    _require_in_symbol(
        state,
        abstract,
        "DelegatingParser.extract_reasoning",
        (
            "if self._reasoning_parser.engine_based_streaming:",
            "model_output_token_ids=model_output_token_ids",
        ),
        label=label,
    )
    require_python_symbols(
        state,
        adapters,
        {
            "ParserEngineReasoningAdapter.extract_reasoning": (
                "self",
                "model_output",
                "request",
                "model_output_token_ids",
            ),
            "ParserEngineReasoningAdapter.reasoning_token_count": ("self",),
        },
        label=label,
    )
    require_text(
        state,
        adapters,
        "self._parser_engine.batch_token_ids(model_output_token_ids)",
        label=label,
    )

    # The wire field is OpenAI's own shape and is absent, never zero, when
    # no exact split was made.
    require_python_symbols(
        state, protocol, {"CompletionTokenUsageInfo": None}, label=label
    )
    require_text(state, protocol, "    reasoning_tokens: int\n", label=label)
    require_text(
        state,
        protocol,
        "completion_tokens_details: CompletionTokenUsageInfo | None = None",
        label=label,
    )

    # Both chat paths report it from the parsers that split the choices,
    # refuse a parser that missed generated ids, and never estimate.
    require_python_symbols(
        state,
        chat,
        {
            "_reasoning_token_count": ("parser", "generated_token_count"),
            "_make_completion_tokens_details": ("reasoning_token_counts",),
        },
        label=label,
    )
    _require_in_symbol(
        state,
        chat,
        "_reasoning_token_count",
        (
            "count = parser.reasoning_token_count",
            "if parser.generated_token_count != generated_token_count:",
            "reasoning token accounting refused",
        ),
        label=label,
    )
    require_text(state, chat, "completion_tokens_details=", count=6, label=label)
    _require_in_symbol(
        state,
        chat,
        "OpenAIServingChat.chat_completion_stream_generator",
        (
            "completion_tokens_details = _make_completion_tokens_details(",
            "parsers, previous_num_tokens, strict=True",
            "completion_tokens_details=completion_tokens_details,",
        ),
        label=label,
    )
    _require_in_symbol(
        state,
        chat,
        "OpenAIServingChat.chat_completion_full_generator",
        (
            "reasoning_token_counts: list[int | None] = []",
            "_reasoning_token_count(parser, len(token_ids))",
            "reasoning_token_counts.append(None)",
            "completion_tokens_details=_make_completion_tokens_details(",
        ),
        label=label,
    )
    for absent in ("estimate", "len(reasoning)", "tokenizer.encode(reasoning"):
        forbid_text(state, chat, absent, label=label)

    require_python_symbols(
        state,
        test,
        {
            "TestStreaming.test_counts_ids_before_the_explicit_end": None,
            "TestStreaming.test_counts_ids_before_the_implicit_tool_call_end": None,
            "TestStreaming.test_counts_every_id_when_reasoning_never_ends": None,
            "TestBatch.test_batch_split_is_made_on_ids_and_counted": None,
            "TestBatch.test_batch_and_stream_agree_on_the_same_generation": None,
            "TestRefusals.test_text_fed_without_ids_refuses_the_count": None,
            "TestRefusals.test_a_grammar_without_one_id_boundary_serves_no_count": None,
        },
        label=label,
    )


def _validate_anthropic_inputs_before(state: State) -> None:
    forbid_text(
        state, "vllm/entrypoints/anthropic/protocol.py",
        "def for_status(", label="Anthropic input fidelity precondition",
    )


def _validate_anthropic_inputs_after(state: State) -> None:
    label = "Anthropic input fidelity result"
    router = "vllm/entrypoints/anthropic/api_router.py"
    serving = "vllm/entrypoints/anthropic/serving.py"
    protocol = "vllm/entrypoints/anthropic/protocol.py"
    errors = "vllm/entrypoints/serve/exception_handling/error_response.py"
    for name in ("create_messages", "count_tokens"):
        _require_in_symbol(state, router, name, (
            "except Exception as e:", f'return refuse(e, route="{name}")',
        ), label=label)
    _require_in_symbol(state, router, "refuse", (
        "error = create_error_response(exc)", "return translate_error_response(error)",
    ), label=label)
    _require_in_symbol(state, router, "translate_error_response", (
        "status_code=response.error.code", "AnthropicErrorResponse.for_status(",
    ), label=label)
    _require_in_symbol(state, errors, "error_json_response", (
        "is_anthropic_api_path(get_route_path(request.scope))",
        "AnthropicErrorResponse.for_status(", "content = error.model_dump()",
        "status_code=error.error.code",
    ), label=label)
    for name in ("exception", "http", "validation", "vllm_error"):
        path = f"vllm/entrypoints/serve/exception_handling/handlers/{name}.py"
        require_text(state, path, "return error_json_response(req, err)",
                     count=2 if name == "vllm_error" else 1, label=label)
        forbid_text(state, path, "return JSONResponse(", label=label)
    for needle in (
        '400: "invalid_request_error"', '500: "api_error"',
        '529: "overloaded_error"', "def for_status(",
    ):
        require_text(state, protocol, needle, label=label)
    _require_in_symbol(state, serving, "AnthropicServingMessages._convert_user_tool_result", (
        "raise VLLMValidationError(", "if block.is_error:", "TOOL_RESULT_ERROR_LINE",
        "tool_content_parts.insert(", "item_where",
    ), label=label)
    _require_in_symbol(state, serving, "AnthropicServingMessages._convert_image_source_to_url", (
        'source.get("type") == "url"', 'source.get("type") == "base64"',
        "raise VLLMValidationError(",
    ), label=label)
    _require_in_symbol(state, serving, "AnthropicServingMessages.message_stream_converter", (
        "failure = ErrorResponse.model_validate(payload)",
        "AnthropicErrorResponse.for_status(", 'type="api_error"',
    ), label=label)
    forbid_text(state, serving, 'type="internal_error"', label=label)
    require_python_symbols(state,
        "tests/entrypoints/anthropic/test_anthropic_messages_conversion.py", {
            "TestToolResultFidelity.test_an_item_the_rendering_cannot_carry_is_refused": None,
            "TestToolResultFidelity.test_is_error_is_stated_before_media_parts": None,
            "TestErrorEnvelope.test_real_request_gates_return_400_before_streaming": None,
            "TestErrorEnvelope.test_stream_forwards_the_classified_error_and_never_reports_success": None,
        }, label=label)


def _validate_qwen_language_before(state: State) -> None:
    forbid_text(state, "vllm/parser/qwen3.py", 'tool_preamble_text="\\n"',
                label="Qwen exact tool language precondition")


def _validate_qwen_language_after(state: State) -> None:
    label = "Qwen exact tool language result"
    qwen = "vllm/parser/qwen3.py"
    engine = "vllm/parser/engine/streaming_parser_engine.py"
    parser = "vllm/parser/engine/parser_engine.py"
    abstract = "vllm/parser/abstract_parser.py"
    for text in ('tool_preamble_text="\\n"',
                 '(ParserState.TOOL_PARAM_VALUE, "PARAM_END")',
                 'batch_tool_pass_uses_ids = True'):
        require_text(state, qwen, text, label=label)
    for text in ('(ParserState.CONTENT, "FUNC_PREFIX")',
                 '(ParserState.TOOL_BETWEEN, "FUNC_PREFIX")',
                 '(ParserState.TOOL_NAME, "FUNC_END")'):
        forbid_text(state, qwen, text, label=label)
    _require_in_symbol(state, qwen, "Qwen3Parser._check_skip_tool_parsing", (
        'tool_choice == "none" or not tools', 'self.skip_tool_parsing = True',
    ), label=label)
    _require_in_symbol(state, qwen, "Qwen3Parser.extract_batch_content_ids", (
        'boundary = self.reasoning_token_count', 'offset = boundary - token_offset',
        'return token_ids[offset:]',
    ), label=label)
    _require_in_symbol(state, engine, "StreamingParserEngine._on_terminal", (
        'self._preamble_text == self.config.tool_preamble_text',
        'self._in_skipped_parameter_value', 'terminal == "PARAM_END"',
    ), label=label)
    _require_in_symbol(state, engine, "StreamingParserEngine.finish", (
        'events.extend(self._abandon_preamble())',
        'events.extend(self._abandon_call_header(""))',
    ), label=label)
    _require_in_symbol(state, parser, "ParserEngine._events_to_delta", (
        'content_parts.append(deferred_after_call)',
        'content_parts.insert(0, earlier_deferred)',
        'case EventType.TOOL_CALL_CLOSED:', 'case EventType.TOOL_CALL_ABANDONED:',
    ), label=label)
    _require_in_symbol(state, abstract, "DelegatingParser.parse", (
        'content_token_ids=self._content_token_ids(model_output_token_ids)',
    ), label=label)
    _require_in_symbol(state, abstract, "DelegatingParser.parse_delta", (
        'state.reasoning_content_token_ids.extend(',
        'token_offset=state.generated_token_count - len(delta_token_ids)',
        'state.reasoning_content_token_ids = []',
    ), label=label)
    require_python_symbols(state,
        "tests/parser/engine/test_delegating_replay.py", {
            "test_qwen_value_markup_is_identical_in_batch_and_stream": None,
        }, label=label)
    require_python_symbols(state,
        "tests/parser/engine/test_reasoning_token_count.py", {
            "TestStreaming.test_boundary_ids_wait_for_detokenized_text": None,
        }, label=label)


def _validate_png_source_before(state: State) -> None:
    forbid_text(state, "vllm/multimodal/media/image.py", "bit_depth, color_type",
                label="PNG source admission precondition")


def _validate_png_source_after(state: State) -> None:
    label = "PNG source admission result"
    _require_in_symbol(state, "vllm/multimodal/media/image.py", "ImageMediaIO.load_bytes", (
        'bit_depth, color_type = data[24:26]',
        'if bit_depth != 8 or color_type not in (2, 6):',
        'detected IHDR',
    ), label=label)
    source = _symbol_source(state, "vllm/multimodal/media/image.py",
                            "ImageMediaIO.load_bytes", label=label)
    _require_ordered(source, ('bit_depth, color_type', 'image = normalize_image(image)',
                             'image.load()'), label=label, location="PNG source gate")
    require_python_symbols(state, "tests/multimodal/media/test_image.py", {
        "test_qwen38_rejects_sixteen_bit_source_before_pillow_reduces_it": None,
    }, label=label)


def _validate_kv_physical_before(state: State) -> None:
    require_text(state, "vllm/v1/worker/gpu_worker.py",
                 "kv_physical_bound = (\n            self.init_snapshot.total_memory",
                 label="KV physical bound precondition")


def _validate_kv_physical_after(state: State) -> None:
    label = "KV physical bound result"
    _require_in_symbol(state, "vllm/v1/worker/gpu_worker.py",
                       "Worker.determine_available_memory", (
        "kv_physical_bound = (\n            self.init_snapshot.free_memory",
        "- profile_result.non_kv_cache_memory",
        "- cudagraph_memory_estimate_applied",
        "int(kv_physical_bound)",
    ), label=label)
    require_python_symbols(state, "tests/v1/worker/test_gpu_worker.py", {
        "test_physical_bound_charges_preexisting_residents_once": None,
    }, label=label)


def _validate_single_call_before(state: State) -> None:
    require_text(state, "vllm/entrypoints/openai/chat_completion/serving.py",
                 "choice_data = maybe_filter_parallel_tool_calls(choice_data, request)",
                 count=2, label="Call-count grammar precondition")


def _validate_single_call_after(state: State) -> None:
    label = "Call-count grammar result"
    _require_in_symbol(state, "vllm/tool_parsers/structural_tag_registry.py",
                       "get_model_structural_tag", (
        'model == "qwen_3_coder" and parallel_tool_calls is False',
        'suffix.stop_after_first = True',
        'get_xgrammar_model_structural_tag(', 'reasoning=reasoning',
        'isinstance(suffix, TagFormat)', 'raise RuntimeError(',
    ), label=label)
    _require_in_symbol(state, "vllm/tool_parsers/abstract_tool_parser.py",
                       "ToolParser.get_structural_tag", (
        'parallel_tool_calls=request.parallel_tool_calls',
    ), label=label)
    forbid_text(state, "vllm/entrypoints/openai/chat_completion/serving.py",
                "maybe_filter_parallel_tool_calls", label=label)
    _require("vllm/entrypoints/serve/utils/tool_calls_utils.py" not in state,
             f"{label}: obsolete response filter survived")
    require_python_symbols(state, "tests/tool_parsers/test_structural_tag_registry.py", {
        "test_qwen3_auto_with_parallel_off_ends_the_turn_after_one_call": None,
        "test_single_qwen_call_preserves_the_builtin_reasoning_prefix": None,
    }, label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/chat_completion/test_parallel_tool_call_integrity.py", {
            "test_response_does_not_discard_calls_when_parallel_is_false": None,
        }, label=label)


def _validate_responses_history_before(state: State) -> None:
    label = "Responses history precondition"
    require_text(state, "vllm/entrypoints/openai/responses/utils.py",
                 "output_text = item.content[0].text", label=label)
    forbid_text(state, "vllm/entrypoints/chat_utils.py",
                "def validate_tool_result_correlation(", label=label)


def _validate_responses_history_after(state: State) -> None:
    label = "Responses history result"
    utils = "vllm/entrypoints/openai/responses/utils.py"
    _require_in_symbol(state, utils, "construct_input_messages", (
        "list(prev_response_output or [])", "validate_tool_result_correlation(messages)",
        "construct_chat_messages_with_tool_call(new_items)",
    ), label=label)
    _require_in_symbol(state, utils, "_construct_message_from_response_item", (
        '"".join(block.text for block in item.content)',
        '"".join(block.text for block in item.summary)',
        "_assistant_content(item.content)", "return deepcopy(item)",
    ), label=label)
    forbid_text(state, utils, "item.content[0]", label=label)
    forbid_text(state, utils, "item.summary[0]", label=label)
    _require_in_symbol(state, "vllm/entrypoints/openai/chat_completion/protocol.py",
                       "ChatCompletionRequest._validate_tool_result_correlation", (
        "validate_tool_result_correlation(self.messages)",
    ), label=label)
    _require_in_symbol(state, "vllm/entrypoints/chat_utils.py",
                       "validate_tool_result_correlation", (
        "raise VLLMValidationError(", "result_id != expected_id", "pending_ids.pop(0)",
    ), label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/responses/test_responses_utils.py", {
            "test_replayed_blocks_preserve_every_byte_and_reasoning": None,
            "test_tool_history_correlation_is_shared_across_surfaces": None,
            "test_responses_history_is_validated_before_rendering": None,
        }, label=label)


def _validate_responses_identity_before(state: State) -> None:
    require_text(state, "vllm/entrypoints/openai/responses/serving.py",
                 "async def empty_async_generator():",
                 label="Responses stream identity precondition")


def _validate_responses_identity_after(state: State) -> None:
    label = "Responses stream identity result"
    serving = "vllm/entrypoints/openai/responses/serving.py"
    _require_in_symbol(state, serving, "OpenAIServingResponses.responses_stream_generator", (
        "output.append(event_data.item)", "isinstance(event_data, ResponseOutputItemDoneEvent)",
        "self._finalize_response(", "event_data.output_index != len(output)",
    ), label=label)
    source = _symbol_source(state, serving,
        "OpenAIServingResponses.responses_stream_generator", label=label)
    _require("responses_full_generator(" not in source,
             f"{label}: streaming must not re-enter batch generation")
    forbid_text(state, serving, "empty_async_generator", label=label)
    _require_in_symbol(state, serving, "OpenAIServingResponses.responses_full_generator", (
        "self._collect_response_output(", "self._finalize_response(",
    ), label=label)
    _require_in_symbol(state, serving, "OpenAIServingResponses._finalize_response", (
        "output=output", "usage=usage", "is_streaming=request.stream",
        'if finish_reason == "length"',
    ), label=label)
    _require_in_symbol(state, "vllm/entrypoints/openai/responses/streaming_events.py",
        "emit_simple_content_done", (
            "logprobs=state.accumulated_logprobs or None",
            'status="incomplete" if incomplete else "completed"',
        ), label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/responses/test_serving_responses.py", {
            "test_terminal_response_uses_streamed_items_and_ids": None,
            "test_text_output_retains_logprobs_and_status_on_both_transports": None,
        }, label=label)


def _validate_anthropic_terminal_before(state: State) -> None:
    require_text(state, "vllm/entrypoints/anthropic/serving.py",
                 "self.stop_reason_map = {",
                 label="Anthropic terminal metadata precondition")


def _validate_anthropic_terminal_after(state: State) -> None:
    label = "Anthropic terminal metadata result"
    serving = "vllm/entrypoints/anthropic/serving.py"
    stop = _require_in_symbol(state, serving, "_anthropic_stop_metadata", (
        'finish_reason == "length"',
        'stop_reason="max_tokens", stop_sequence=None',
        'isinstance(stop_reason, str)',
        'stop_reason="stop_sequence", stop_sequence=stop_reason',
        'stop_reason="tool_use" if has_tool_use else "end_turn"',
        "raise GenerationError(",
    ), label=label)
    _require_ordered(stop, (
        'finish_reason == "length"', 'isinstance(stop_reason, str)',
        'stop_reason="tool_use" if has_tool_use else "end_turn"',
    ), label=label, location="_anthropic_stop_metadata")
    _require_in_symbol(state, serving, "AnthropicServingMessages.messages_full_converter", (
        "_anthropic_stop_metadata(", "choice.finish_reason, choice.stop_reason",
        "bool(choice.message.tool_calls)", "result.stop_sequence = stop.stop_sequence",
    ), label=label)
    _require_in_symbol(state, serving, "AnthropicServingMessages.message_stream_converter", (
        "_anthropic_stop_metadata(", "finish_reason, matched_stop, has_tool_use",
        "matched_stop = origin_chunk.choices[0].stop_reason",
        "has_tool_use = True", "if not sent_message_delta:",
        "origin_chunk.usage is None or sent_message_delta",
        'raise GenerationError("Chat stream ended without its terminal marker")',
    ), label=label)
    forbid_text(state, serving, "stop_reason_map", label=label)
    require_python_symbols(state,
        "tests/entrypoints/anthropic/test_anthropic_messages_conversion.py", {
            "TestAnthropicTerminalMetadata.test_stream_and_batch_report_observed_cause": None,
            "TestAnthropicTerminalMetadata.test_invalid_completion_cause_is_never_reported_as_success": None,
            "TestAnthropicTerminalMetadata.test_incomplete_or_invalid_stream_has_no_success_terminal": None,
        }, label=label)


def _validate_sampling_resolution_before(state: State) -> None:
    label = "generation sampling resolution precondition"
    require_text(state,
        "vllm/entrypoints/scale_out/token_in_token_out/protocol.py",
        "def is_sampling_param_provided(", label=label)
    require_text(state, "vllm/entrypoints/openai/completion/protocol.py",
        "max_tokens: int | None = 16", label=label)


def _validate_sampling_resolution_after(state: State) -> None:
    label = "generation sampling resolution result"
    protocol = "vllm/entrypoints/scale_out/token_in_token_out/protocol.py"
    serving = "vllm/entrypoints/scale_out/token_in_token_out/serving.py"
    _require_in_symbol(state, protocol,
        "SamplingParamsInput.__get_pydantic_core_schema__", (
            "get_type_hints(SamplingParams)", "required=False",
            "isinstance(value, SamplingParams)",
            "return {name: getattr(value, name) for name in fields}",
            'excluded = {"skip_clone", "output_text_buffer_length"',
            'extra_behavior="forbid"',
        ), label=label)
    _require_in_symbol(state, protocol, "GenerateRequest.to_sampling_params", (
        "deepcopy({**default_sampling_params, **self.sampling_params})",
        "resolve_final_response_token_budget(",
        'extra_args["kv_scope"] = self.kv_scope',
        "return SamplingParams(**values)",
    ), label=label)
    source = _symbol_source(state, serving, "ServingTokens.serve_tokens", label=label)
    _require_ordered(source, (
        "max_tokens = get_max_tokens(",
        "sampling_params = request.to_sampling_params(",
        "sampling_params.n > max_num_seqs",
        "msgspec.msgpack.encode(sampling_params)",
        "self.engine_client.generate(",
    ), label=label, location="ServingTokens.serve_tokens")
    for path in (protocol, serving):
        for obsolete in ("is_sampling_param_provided", "_sampling_params_provided_keys"):
            forbid_text(state, path, obsolete, label=label)
    for surface, request_type in (
        ("chat_completion", "ChatCompletionRequest"),
        ("completion", "CompletionRequest"),
        ("responses", "ResponsesRequest"),
    ):
        path = f"vllm/entrypoints/openai/{surface}/protocol.py"
        _require_in_symbol(state, path, f"{request_type}.to_sampling_params", (
            "resolve_final_response_token_budget(",
            'default_sampling_params.get("presence_penalty", 0.0)',
            '"thinking_token_budget"',
            "final_response_token_budget=final_response_token_budget",
        ), label=label)
    completion = "vllm/entrypoints/openai/completion/protocol.py"
    require_text(state, completion, "max_tokens: int | None = None", label=label)
    forbid_text(state, completion, "normalize_null_max_tokens", label=label)
    for surface in ("chat_completion", "completion"):
        require_text(state, f"vllm/entrypoints/openai/{surface}/protocol.py",
            "presence_penalty: float | None = None", label=label)
    require_text(state, "vllm/entrypoints/openai/responses/protocol.py",
        'min_p=default_sampling_params.get("min_p", 0.0)', label=label)
    require_python_symbols(state,
        "tests/entrypoints/scale_out/token_in_token_out/test_protocol.py", {
            "test_omitted_settings_remain_omitted_until_resolution": None,
            "test_configured_sampling_policy_reaches_every_generation_surface": None,
            "test_rendered_requests_keep_resolved_sampling_through_generate_json": None,
            "test_serving_resolves_once_before_dispatch_on_both_transports": None,
        }, label=label)


def _validate_sampling_boundary_before(state: State) -> None:
    label = "sampling decoding boundary precondition"
    for surface in ("chat_completion", "completion"):
        require_text(state, f"vllm/entrypoints/openai/{surface}/serving.py",
            "if request.use_beam_search:", label=label)


def _validate_sampling_boundary_after(state: State) -> None:
    label = "sampling decoding boundary result"
    protocol = "vllm/entrypoints/openai/engine/protocol.py"
    _require_in_symbol(state, protocol, "_reject_beam_search", (
        "if value is True:", "raise VLLMValidationError(",
        '"Beam search is not supported by this deployment."',
        'parameter="use_beam_search"',
    ), label=label)
    require_text(state, protocol,
        "Literal[False], BeforeValidator(_reject_beam_search)", label=label)
    for surface, names in (
        ("chat_completion", ("ChatCompletionRequest", "BatchChatCompletionRequest")),
        ("completion", ("CompletionRequest",)),
    ):
        path = f"vllm/entrypoints/openai/{surface}/protocol.py"
        for name in names:
            _require_in_symbol(state, path, name,
                ("use_beam_search: BeamSearchDisabled = False",), label=label)
        forbid_text(state, path, "to_beam_search_params", label=label)
        serving = f"vllm/entrypoints/openai/{surface}/serving.py"
        for obsolete in ("use_beam_search", "BeamSearchParams", "self.beam_search("):
            forbid_text(state, serving, obsolete, label=label)
        require_text(state, serving, "self.engine_client.generate(", label=label)
    forbid_text(state, "vllm/entrypoints/scale_out/render/serving.py",
        "use_beam_search", label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/test_beam_search_boundary.py", {
            "test_unsupported_beam_search_returns_the_same_boundary_error": None,
            "test_supported_sampling_does_not_require_a_decoding_flag": None,
            "test_request_schema_advertises_the_supported_decoding_value": None,
        }, label=label)


def _validate_generate_result_before(state: State) -> None:
    label = "token generation result baseline"
    require_text(state,
        "vllm/entrypoints/scale_out/token_in_token_out/serving.py",
        'finish_reason=output.finish_reason if output.finish_reason else "stop"',
        label=label)


def _validate_generate_result_after(state: State) -> None:
    label = "token generation result integrity"
    protocol = "vllm/entrypoints/scale_out/token_in_token_out/protocol.py"
    serving = "vllm/entrypoints/scale_out/token_in_token_out/serving.py"
    _require_in_symbol(state, protocol, "GenerateRequest.to_sampling_params", (
        'values["output_kind"]',
        "RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY",
    ), label=label)
    _require_in_symbol(state, protocol,
        "SamplingParamsInput.__get_pydantic_core_schema__", (
            '"output_text_buffer_length", "output_kind"}',
        ), label=label)
    _require_in_symbol(state, protocol, "GenerateResponseChoice", (
        "finish_reason: GenerateFinishReason",
        "stop_reason: int | str | None = None",
        "token_ids: list[int]",
    ), label=label)
    _require_in_symbol(state, serving, "_GenerateOutputState.accept", (
        "raise GenerationError(", "i in indices or i in terminals",
        "res.finished != (len(terminals) == self.n)",
    ), label=label)
    for method in ("serve_tokens_full_generator", "serve_tokens_stream_generator"):
        _require_in_symbol(state, serving, f"ServingTokens.{method}", (
            "state.accept(res)", "state.finish()", "stop_reason=output.stop_reason",
        ), label=label)
    forbid_text(state, serving,
        'output.finish_reason if output.finish_reason else "stop"', label=label)
    forbid_text(state, serving, "[0] * len(res.outputs)", label=label)
    derender = "vllm/renderers/online_derenderer.py"
    for method in ("_derender_chat", "_derender_completion",
                   "derender_chat_stream", "derender_completion_stream"):
        _require_in_symbol(state, derender, f"OnlineDerenderer.{method}", (
            "stop_reason=choice.stop_reason",
        ), label=label)
    forbid_text(state, derender, "has empty or null token_ids", label=label)
    require_python_symbols(state,
        "tests/entrypoints/scale_out/token_in_token_out/test_generate_stream.py", {
            "test_terminal_cause_survives_empty_completion": None,
            "test_parallel_sampling_keeps_staggered_choices_and_total_usage": None,
            "test_generation_integrity_failures_are_errors_on_both_transports": None,
        }, label=label)


def _validate_raw_image_before(state: State) -> None:
    require_text(state,
        "vllm/entrypoints/scale_out/token_in_token_out/protocol.py",
        "features: MultiModalFeatures | None = None",
        label="raw image transport baseline")


def _validate_raw_image_after(state: State) -> None:
    label = "raw image token transport"
    protocol = "vllm/entrypoints/scale_out/token_in_token_out/protocol.py"
    _require_in_symbol(state, protocol, "GenerateRequest", (
        'model_config = ConfigDict(extra="forbid")',
        "content_parts: list[InlinePNGPart] | None",
    ), label=label)
    _require_in_symbol(state, protocol, "InlinePNGSource", (
        'model_config = ConfigDict(extra="forbid")',
        '^data:image/png;base64,', 'Literal["auto", "high"]',
    ), label=label)
    serving = "vllm/entrypoints/scale_out/token_in_token_out/serving.py"
    _require_in_symbol(state, serving, "ServingTokens.serve_tokens", (
        "mm_parser.parse_image(part.image_url.url)",
        "process_rendered_multimodal_async(",
        "request.token_ids, mm_data, cache_salt=request.cache_salt",
    ), label=label)
    for path in (protocol, serving,
                 "vllm/entrypoints/scale_out/render/serving.py",
                 "vllm/entrypoints/scale_out/derender/serving.py"):
        for obsolete in ("MultiModalFeatures", "PlaceholderRangeInfo",
                         "mm_serde", "_extract_mm_features"):
            forbid_text(state, path, obsolete, label=label)
    for path in ("vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py",
                 "tests/entrypoints/scale_out/token_in_token_out/test_mm_serde.py"):
        _require(path not in state, f"{label}: obsolete serializer remains: {path}")
    _require_in_symbol(state, "vllm/renderers/base.py",
        "BaseRenderer.process_rendered_multimodal_async", (
            "RenderedPromptTokens(list(token_ids))",
            "skip_mm_cache=True", 'engine_input["cache_salt"] = cache_salt',
        ), label=label)
    processor = "vllm/multimodal/processing/processor.py"
    _require_in_symbol(state, processor,
        "BaseMultiModalProcessor._apply_hf_processor_main", (
            "isinstance(prompt, RenderedPromptTokens)",
            "prompt_ids = list(prompt.token_ids)",
        ), label=label)
    _require_in_symbol(state, processor, "BaseMultiModalProcessor.apply", (
        "self._find_rendered_prompt_placeholders(",
        "self._maybe_apply_prompt_updates(",
    ), label=label)
    _require_in_symbol(state, "vllm/model_executor/models/qwen3_vl.py",
        "Qwen3VLMultiModalProcessor._find_rendered_prompt_placeholders", (
            "spans != expected", "VLLMValidationError(",
            "config.image_token_id", "config.vision_start_token_id",
            "config.vision_end_token_id",
        ), label=label)
    require_python_symbols(state,
        "tests/entrypoints/scale_out/token_in_token_out/test_raw_images.py", {
            "test_rendered_image_tokens_are_validated_without_second_expansion": None,
            "test_render_json_generate_preserves_source_images_tokens_salt_and_identity": None,
            "test_native_image_processing_retains_exact_spans_and_full_cache_data": None,
        }, label=label)


def _validate_canonical_framing_before(state: State) -> None:
    forbid_text(state, "vllm/parser/qwen3.py", "_unframe_parameter_value",
                label="Canonical parameter framing precondition")


def _validate_canonical_framing_after(state: State) -> None:
    label = "Canonical parameter framing result"
    qwen = "vllm/parser/qwen3.py"
    require_python_symbols(
        state, qwen, {"_unframe_parameter_value": ("value", "complete")}, label=label
    )
    # One newline off each end, and the trailing one only once the closer has
    # actually been observed: that is the exact inverse of the transport's
    # framing, which is what makes the round trip a bijection.
    _require_in_symbol(state, qwen, "_unframe_parameter_value", (
        'if value.startswith("\\n"):', "value = value[1:]",
        'if complete and value.endswith("\\n"):', "value = value[:-1]",
    ), label=label)
    _require_in_symbol(state, qwen, "_qwen3_arg_converter", (
        "_unframe_parameter_value(value, complete=True)",
        "_unframe_parameter_value(value, complete=False)",
    ), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_framing_habit_cannot_change_a_decoded_value": None,
        "test_parameter_string_bytes_survive_every_transport_cut": None,
        "test_partial_string_diagnostic_preserves_raw_value_bytes": None,
    }, label=label)


def _validate_xml_fidelity_before(state: State) -> None:
    require_text(state, "vllm/parser/qwen3.py", "def _trim_wrapping_newlines(",
                 label="XML text fidelity baseline")


def _validate_xml_fidelity_after(state: State) -> None:
    label = "XML text fidelity"
    # This stage's own guarantee is that the old newline-trimming helper is
    # gone. How a value then reaches ``params`` is asserted by the
    # canonical-framing contract, which owns that shape: asserting a bare
    # ``params[name] = value`` here would forbid the exact transport inverse
    # that replaced the helper, and this validator also runs against the
    # complete tree.
    forbid_text(state, "vllm/parser/qwen3.py", "_trim_wrapping_newlines", label=label)
    for path in ("vllm/parser/engine/parser_engine.py",
                 "vllm/parser/engine/parser_engine_config.py",
                 "vllm/parser/deepseek_v32.py", "vllm/parser/deepseek_v4.py",
                 "vllm/parser/inkling.py", "vllm/parser/kimi_k2.py"):
        forbid_text(state, path, "strip_content_whitespace_with_tools", label=label)
        forbid_text(state, path, "_strip_content_ws_with_tools", label=label)
    _require_in_symbol(state, "vllm/parser/engine/parser_engine.py",
        "ParserEngine._strip_content_whitespace", (
            "and not content.strip()", 'content = ""', "return content or None",
        ), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_parameter_string_bytes_survive_every_transport_cut": None,
        "test_partial_string_diagnostic_preserves_raw_value_bytes": None,
    }, label=label)


def _validate_phase_terminals_before(state: State) -> None:
    require_text(state, "vllm/parser/engine/streaming_parser_engine.py",
                 "strict = self._token_id_terminal_names if self._ever_had_token_ids else None",
                 label="Phase-aware terminal baseline")


def _validate_phase_terminals_after(state: State) -> None:
    label = "Phase-aware parser terminals"
    config = "vllm/parser/engine/parser_engine_config.py"
    engine = "vllm/parser/engine/streaming_parser_engine.py"
    require_python_symbols(state, config, {"TokenTerminal": None}, label=label)
    require_text(state, config,
                 "required_states: frozenset[ParserState] = frozenset(ParserState)",
                 label=label)
    for path in (config, engine, "vllm/parser/engine/parser_engine.py",
                 "tests/parser/engine/test_engine.py"):
        forbid_text(state, path, "skip_in_token_id_mode", label=label)
    forbid_text(state, engine, "_token_id_terminal_names", label=label)
    _require_in_symbol(state, engine, "StreamingParserEngine._process_lex_tokens", (
        "for tok in tokens:",
        "self.state in self._token_required_states.get(tok.terminal, ())",
        "events.extend(self._on_content(tok.value))",
        "events.extend(self._on_terminal(tok.terminal, tok.value))",
    ), label=label)
    _require_in_symbol(state, "vllm/parser/qwen3.py", "qwen3_config", (
        '"THINK_END": TokenTerminal(think_end)',
        '"TOOL_START": TokenTerminal(tool_start, frozenset({ParserState.REASONING}))',
        '"TOOL_END": TokenTerminal(tool_end, frozenset())',
    ), label=label)
    require_python_symbols(state,
        "tests/parser/engine/test_qwen_terminal_authority.py", {
            "test_content_phase_tool_language_is_independent_of_tokenization": None,
            "test_ordinary_token_trigger_does_not_end_inactive_reasoning": None,
            "test_implicit_reasoning_end_admits_text_encoded_call_closer": None,
            "test_literal_thinking_marker_does_not_move_the_real_token_boundary": None,
            "test_native_grammar_arms_both_tokenizations": None,
        }, label=label)


def _validate_stream_identity_before(state: State) -> None:
    label = "Input-stream agent identity baseline"
    path = "vllm/v1/engine/async_llm.py"
    _require_in_symbol(state, path, "AsyncLLM._add_streaming_input_request", (
        "sp = input_chunk.sampling_params",
        "sp = sampling_params",
        "params=sp,",
    ), label=label)
    forbid_text(state, path, "require_kv_scope(sp)", label=label)


def _validate_stream_identity_after(state: State) -> None:
    label = "Input-stream agent identity"
    source = _require_in_symbol(state, "vllm/v1/engine/async_llm.py",
        "AsyncLLM._add_streaming_input_request", (
            "agent_id = require_kv_scope(sampling_params)",
            "if require_kv_scope(sp) != agent_id:",
            'parameter="kv_scope",',
            "queue.put(InputStreamError(error))",
        ), label=label)
    _require(
        source.index("agent_id = require_kv_scope(sampling_params)")
        < source.index("async for input_chunk in input_stream:")
        < source.index("if require_kv_scope(sp) != agent_id:")
        < source.index("\n                    req = self.input_processor.process_inputs("),
        f"{label}: identity must be bound before reading and checked before dispatch",
    )
    require_python_symbols(state,
        "tests/v1/streaming_input/test_async_llm_streaming.py", {
            "test_input_stream_retains_opaque_id_and_accepts_new_sampling_params": None,
            "test_input_stream_refuses_id_change_before_dispatch_and_aborts_only_its_request": None,
        }, label=label)


def _validate_tool_completion_before(state: State) -> None:
    label = "Tool completion baseline"
    require_text(state, "vllm/sampling_params.py", "_eos_token_id: int | None", label=label)
    forbid_text(state, "vllm/parser/abstract_parser.py", "def parse_output(", label=label)


def _validate_tool_completion_after(state: State) -> None:
    label = "EOS tool commitment and structural text stops"
    abstract = "vllm/parser/abstract_parser.py"
    parser = "vllm/parser/engine/parser_engine.py"
    _require_in_symbol(state, parser, "ParserEngine._completed_tool_events", (
        'terminal[0] == "length"', 'if span.closed',
        'terminal == ("stop", None)', 'value=raw_calls[withheld].text',
    ), label=label)
    _require_in_symbol(state, abstract, "Parser.parse_output_delta", (
        'self.tool_calls_complete is None', 'state.withholding',
        'if not finished:', 'self.parse_output(', 'state.emitted_content',
    ), label=label)
    require_text(state, "vllm/parser/qwen3.py", "validate_tool_names=False", label=label)
    forbid_text(state, parser, "slot.name.strip()", label=label)
    for path, symbol, method in (
        ("vllm/entrypoints/openai/chat_completion/serving.py",
         "OpenAIServingChat.chat_completion_full_generator", "parser.parse_output("),
        ("vllm/entrypoints/openai/chat_completion/serving.py",
         "OpenAIServingChat.chat_completion_stream_generator", "parser.parse_output_delta("),
        ("vllm/renderers/online_derenderer.py", "OnlineDerenderer._derender_chat",
         "parser.parse_output("),
    ):
        _require_in_symbol(state, path, symbol, (method, "stop_reason"), label=label)
    require_text(state, "vllm/entrypoints/openai/responses/serving.py",
                 "parser.parse_output_delta(", label=label)
    require_text(state, "vllm/entrypoints/openai/responses/context.py",
                 "parser.parse_output(", label=label)
    _require_in_symbol(state, "vllm/sampling_params.py",
        "SamplingParams.update_from_generation_config", (
            "self._eos_token_ids", "set(self.stop_token_ids or ())",
        ), label=label)
    forbid_text(state, "vllm/sampling_params.py", "_eos_token_id:", label=label)
    _require_in_symbol(state, "vllm/v1/core/sched/utils.py", "check_stop", (
        "not sampling_params.ignore_eos",
        "last_token_id in sampling_params.eos_token_ids",
        "if allow_stop_token and", "request.stop_reason = last_token_id",
    ), label=label)
    _require_in_symbol(state, "vllm/v1/core/sched/scheduler.py",
        "Scheduler._update_request_with_output", (
            "self.structured_output_manager.allows_stop_token(",
        ), label=label)
    _require_in_symbol(state, "vllm/v1/structured_output/backend_xgrammar.py",
        "XgrammarGrammar.allows_text_stop", (
            "self.matcher.is_completed()", "self.matcher.fork()",
            "probe.accept_string(", "probe.is_completed()",
        ), label=label)
    _require_in_symbol(state, "vllm/v1/structured_output/stop_checker.py",
        "StructuralTagStopChecker.feed", (
            "self.reasoner.is_reasoning_end_streaming(",
            "self.reasoner.extract_content_ids(", "self.matcher.accept_string(char)",
            "if not complete or not next_complete:",
        ), label=label)
    _require_in_symbol(state, "vllm/v1/engine/detokenizer.py", "check_stop_strings", (
        "protected_spans", "stop_index = output_text.find(stop_str, stop_index + 1)",
    ), label=label)
    require_python_symbols(state,
        "tests/v1/structured_output/test_backend_xgrammar_stop_tokens.py", {
            "test_structural_stop_tokens_remain_valid_values_and_resume_in_text": None,
            "test_structural_stop_cannot_remove_the_closing_wrapper": None,
            "test_structural_stop_through_qwen_output_processor": None,
        }, label=label)


def _validate_thinking_boundary_before(state: State) -> None:
    label = "V1 thinking-boundary baseline"
    path = "vllm/v1/sample/thinking_budget_state.py"
    _require_in_symbol(state, path, "ThinkingBudgetStateHolder.__init__", (
        "self.think_end_token_ids",
    ), label=label)
    forbid_text(state, path, "self.reasoning_boundary_token_ids", label=label)


def _validate_thinking_boundary_after(state: State) -> None:
    label = "One-way V1 thinking boundary"
    _require_in_symbol(state, "vllm/parser/engine/parser_engine.py",
        "ParserEngine.reasoning_boundary_token_ids", (
            "return self._reasoning_boundary_ids",
        ), label=label)
    _require_in_symbol(state, "vllm/parser/engine/adapters.py",
        "ParserEngineReasoningAdapter.is_reasoning_end_streaming", (
            "self._parser_engine.is_reasoning_end_streaming(",
        ), label=label)
    _require_in_symbol(state, "vllm/config/reasoning.py",
        "ReasoningConfig.initialize_token_ids", (
            "reasoning_parser.reasoning_boundary_token_ids",
        ), label=label)
    _require_in_symbol(state, "vllm/parser/qwen3.py",
        "Qwen3Parser.is_reasoning_end_streaming", (
            "token in self.reasoning_boundary_token_ids for token in delta_ids",
        ), label=label)
    source = _require_in_symbol(state, "vllm/v1/sample/thinking_budget_state.py",
        "ThinkingBudgetStateHolder._update_think_state", (
            'if not state["reasoning_ended"]:',
            'token in self.reasoning_boundary_token_ids',
            'if state["reasoning_ended"]:',
            'state["in_think"] = state["in_end"] = False',
            'state["force_index"] = []',
        ), label=label)
    _require(
        source.index('if state["reasoning_ended"]:')
        < source.index('if state.get("thinking_token_budget", -1) == -1:'),
        f"{label}: the completed phase must bypass budget forcing",
    )
    _require_in_symbol(state, "vllm/v1/structured_output/__init__.py",
        "StructuredOutputManager._find_reasoning_end_index", (
            "if token in reasoner.reasoning_boundary_token_ids",
        ), label=label)
    require_python_symbols(state, "tests/v1/logits_processors/test_correctness.py", {
        "TestQwenThinkingBoundary.test_budget_never_forces_after_a_natural_boundary": None,
        "TestQwenThinkingBoundary.test_resumed_request_retains_its_completed_thinking_phase": None,
        "TestQwenThinkingBoundary.test_grammar_keeps_the_current_tool_trigger_with_prompt_history": None,
    }, label=label)


def _validate_xml_schema_before(state: State) -> None:
    label = "Qwen XML schema baseline"
    forbid_text(state, "vllm/parser/qwen3.py", "class _XMLParameterSchema", label=label)
    _require_in_symbol(state, "vllm/parser/engine/parser_engine.py",
        "ParserEngine._flush_arg_converter", (
            "final_json = converter(slot.args, False)",
            "self._fix_arg_types(final_json, slot.name)",
        ), label=label)


def _validate_xml_schema_after(state: State) -> None:
    label = "Schema-faithful Qwen XML arguments"
    qwen = "vllm/parser/qwen3.py"
    _require_in_symbol(state, qwen, "Qwen3Parser._convert_tool_arguments", (
        "find_tool_schema(self._tools, func_name)", "_qwen3_arg_converter(",
    ), label=label)
    _require_in_symbol(state, qwen, "_qwen3_arg_converter", (
        "if schema and m is None:", "return _decode_xml_parameters(params, schema)",
    ), label=label)
    _require_in_symbol(state, qwen, "_decode_xml_parameters", (
        "object_pairs_hook=_unique_json_object", "parse_constant=_reject_non_json_number",
        "context.known", "context.field(name)", "context.validator.is_valid(decoded)",
        "json.dumps(name, ensure_ascii=False)",
    ), label=label)
    _require_in_symbol(state, qwen, "_XMLParameterSchema._project", (
        'schema.get("$ref")', 'schema.get("properties", {})',
        '("allOf", "anyOf", "oneOf")', 'self._condition_is_known(schema["if"])',
    ), label=label)
    _require_in_symbol(state, qwen, "qwen3_config", (
        "stream_arg_deltas=False", "arg_converter=_qwen3_arg_converter",
    ), label=label)
    for symbol in ("_compute_arg_delta", "_flush_arg_converter", "_build_extracted_result"):
        _require_in_symbol(state, "vllm/parser/engine/parser_engine.py",
            "ParserEngine." + symbol, ("self._convert_tool_arguments(",), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_xml_typing_preserves_the_resolved_schema": None,
        "test_native_xml_grammar_and_parser_agree_on_value_types": None,
        "test_unclosed_typed_parameter_keeps_its_raw_diagnostic": None,
        "test_stable_discriminator_resolves_many_parameter_types": None,
    }, label=label)


def _validate_token_positions_before(state: State) -> None:
    _require_in_symbol(state, "vllm/parser/engine/token_id_scanner.py",
        "TokenIDScanner.scan", ("self._decode_token(tid)",
                                "self._recover_holdback_text("),
        label="Token/text position baseline")


def _validate_token_positions_after(state: State) -> None:
    label = "Native ByteLevel token positions"
    scanner = "vllm/parser/engine/token_id_scanner.py"
    _require_in_symbol(state, scanner, "TokenIDScanner.__init__", (
        "isinstance(backend, Tokenizer)", "isinstance(backend.decoder, ByteLevel)",
        ".isascii()", "_ByteLevelTokenPositions(",
    ), label=label)
    _require_in_symbol(state, scanner, "TokenIDScanner.scan", (
        "return self._token_positions.scan(delta_text, delta_token_ids)",
    ), label=label)
    positions = _require_in_symbol(state, scanner, "_ByteLevelTokenPositions", (
        "NativeDecodeStream(self.tokenizer)", "self.decoder.step(token_id)",
        "decoded.endswith(spelling)", "self.pending.popleft()", "self.offset",
    ), label=label)
    _require(".find(" not in positions and ".rfind(" not in positions,
             label + ": marker placement must not search for its spelling")
    _require_in_symbol(state, scanner, "_ByteLevelTokenPositions.finish", (
        "TextChunk(item.text[: self.offset])", "self.pending.clear()",
    ), label=label)
    _require_in_symbol(state, scanner, "TokenIDScanner.reset", (
        "self._token_positions.reset()",
    ), label=label)
    _require_in_symbol(state, "vllm/parser/engine/streaming_parser_engine.py",
        "StreamingParserEngine.feed", ("not self._scanner.tracks_token_positions",),
        label=label)
    _require_in_symbol(state, "vllm/v1/engine/detokenizer.py",
        "FastIncrementalDetokenizer.__init__", (
            "self.decoder = NativeDecodeStream(", "ids=request.prompt_token_ids",
        ), label=label)
    _require_in_symbol(state, "vllm/v1/engine/detokenizer.py",
        "FastIncrementalDetokenizer.decode_next", ("self.decoder.step(next_token_id)",),
        label=label)
    require_python_symbols(state, "vllm/tokenizers/detokenizer_utils.py", {
        "NativeDecodeStream.step": None,
    }, label=label)
    require_python_symbols(state, "tests/parser/engine/test_token_id_scanner.py", {
        "TestNativeByteLevelPositions.test_literal_and_real_marker_positions": None,
    }, label=label)
    require_python_symbols(state,
        "tests/v1/structured_output/test_backend_xgrammar_stop_tokens.py", {
            "test_qwen_stop_stripped_boundary_never_rebinds_to_prose": None,
        }, label=label)


def _validate_precise_errors_before(state: State) -> None:
    require_text(state,
        "vllm/entrypoints/serve/exception_handling/error_response.py",
        "elif isinstance(exc, (ValueError, TypeError, OverflowError)):",
        label="Precise request error baseline")


def _validate_precise_errors_after(state: State) -> None:
    label = "Precise request error classification"
    errors = "vllm/entrypoints/serve/exception_handling/error_response.py"
    forbid_text(state, errors, "(ValueError, TypeError, OverflowError)", label=label)
    forbid_text(state, errors, '__name__ == "TemplateError"', label=label)
    _require_in_symbol(state, errors, "create_error_response", (
        "isinstance(exc, VLLMValidationError)", "param = exc.parameter",
        "status_code = HTTPStatus.INTERNAL_SERVER_ERROR",
    ), label=label)
    hf = "vllm/renderers/hf.py"
    source = _require_in_symbol(state, hf, "safe_apply_chat_template", (
        'resolved_kwargs["raise_exception"] = _raise_template_validation_error',
        "plain = tokenizer.apply_chat_template(",
    ), label=label)
    _require("except Exception" not in source and "raise ValueError(str(e))" not in source,
             label + ": template bugs must retain their original server cause")
    _require_in_symbol(state, hf, "_raise_template_validation_error", (
        'raise VLLMValidationError(message, parameter="messages")',
    ), label=label)
    _require_in_symbol(state, "vllm/multimodal/media/image.py", "ImageMediaIO.load_bytes", (
        "raise VLLMServerError(", "raise VLLMValidationError(",
        "Image.open(BytesIO(data))", "image.load()",
    ), label=label)
    loader = _find_symbol(state, "vllm/multimodal/media/image.py",
                          "ImageMediaIO.load_bytes", label=label)
    decoder_calls = []
    for node in ast.walk(loader):
        if isinstance(node, ast.With) and any(
            ast.unparse(item.context_expr) == "_image_decoder_errors()"
            for item in node.items
        ):
            _require(len(node.body) == 1, label + ": decoder boundary contains pipeline work")
            decoder_calls.append(ast.unparse(node.body[0].value.func))
    _require(decoder_calls == ["Image.open", "image.load"],
             label + ": only native byte decoders may translate format failures")
    _require_in_symbol(state, "vllm/multimodal/media/image.py", "ImageMediaIO.load_base64", (
        "decoded = pybase64.b64decode(data, validate=True)", "raise VLLMValidationError(",
    ), label=label)
    source = _require_in_symbol(state, "vllm/entrypoints/serve/utils/api_utils.py",
        "get_max_tokens", ("raise VLLMValidationError(",), label=label)
    _require("raise ValueError" not in source, label + ": context refusals must be typed")
    require_python_symbols(state,
        "tests/entrypoints/anthropic/test_anthropic_messages_conversion.py", {
            "test_internal_template_failure_keeps_its_server_cause": None,
            "test_template_programming_error_is_not_a_request_refusal": None,
            "test_unrelated_exception_named_template_error_is_a_server_error": None,
        }, label=label)
    require_python_symbols(state, "tests/multimodal/media/test_image.py", {
        "test_internal_image_pipeline_error_keeps_its_server_cause": None,
        "test_invalid_base64_image_is_a_typed_request_error": None,
    }, label=label)


def validate_final(state: State) -> None:
    """Reassert every durable semantic invariant on the complete tree.

    Iterates the registry itself rather than a restated name list: a
    hand-list here was a second copy of the stage set, and a contract
    missing from it would have had its final validation silently skipped —
    drift with no failure. build_patchset separately requires the generated
    stages and this registry to name identical sets, so registry iteration
    is complete by construction.
    """
    for contract in CONTRACTS.values():
        contract.validate_after(state)


CONTRACTS: Mapping[str, SemanticContract] = {
    "qwen-canonical-parameter-framing": SemanticContract(
        rationale=(
            "The Qwen XML transport pads every parameter value, and the model "
            "emits the framing it was trained on. Reading that framing as data "
            "wrote blank first lines and extra trailing newlines to disk. The "
            "parser must invert the framing exactly once, so every value round "
            "trips unchanged and a hugged value means what a framed one means."
        ),
        removal_condition=(
            "Remove when upstream decodes the Qwen XML transport's framing as "
            "framing on both batch and streaming output, or when the served "
            "model is retrained to emit no framing at all."
        ),
        validate_before=_validate_canonical_framing_before,
        validate_after=_validate_canonical_framing_after,
    ),
    "precise-request-errors": SemanticContract(
        rationale=(
            "Only a known request cause should become a client error. Type "
            "template guards, image decoding/admission and context refusals "
            "at their producers. Preserve internal Python and template errors "
            "as server errors on every API surface."
        ),
        removal_condition=(
            "Remove when upstream carries request causes through these deployed "
            "producers and no longer classifies raw Python/template failures as 400."
        ),
        validate_before=_validate_precise_errors_before,
        validate_after=_validate_precise_errors_after,
    ),
    "token-text-provenance": SemanticContract(
        rationale=(
            "A stop-stripped boundary ID must never bind to a visible prose "
            "lookalike. Native ByteLevel ASCII terminals use decoder positions "
            "across Unicode and text holdback; finishing preserves only visible "
            "fragments and never recreates stripped marker text."
        ),
        removal_condition=(
            "Remove when upstream preserves these token/text positions in the "
            "deployed parser on both batch and streaming V1 output."
        ),
        validate_before=_validate_token_positions_before,
        validate_after=_validate_token_positions_after,
    ),
    "schema-faithful-xml": SemanticContract(
        rationale=(
            "The Qwen XML converter must resolve the complete tool schema and "
            "validate its decoded object. Keep strings when valid, distinguish "
            "grammar padding through constraints, preserve encoded JSON values, "
            "and leave cut or untypable parameters raw. Narrow stable schema "
            "branches before checking joint interpretations."
        ),
        removal_condition=(
            "Remove when upstream decodes Qwen XML with the same complete-schema "
            "and lexical-fidelity guarantees on batch and streaming output."
        ),
        validate_before=_validate_xml_schema_before,
        validate_after=_validate_xml_schema_after,
    ),
    "one-way-thinking-boundary": SemanticContract(
        rationale=(
            "Qwen ends thinking once, at its natural closer or implicit tool "
            "boundary. The deployed V1 budget holder must stop forcing there, "
            "and the grammar must receive the current tool trigger even when "
            "history contains earlier reasoning markers. Use the parser's "
            "existing atomic boundary metadata for both consumers."
        ),
        removal_condition=(
            "Remove when upstream uses the same parser-derived one-way boundary "
            "for V1 thinking budgets and structured-output advancement."
        ),
        validate_before=_validate_thinking_boundary_before,
        validate_after=_validate_thinking_boundary_after,
    ),
    "tool-output-completion": SemanticContract(
        rationale=(
            "Only observed closed wrappers at model EOS are executable calls. "
            "Keep slips verbatim, length diagnostic, and caller stops distinct. "
            "Suspend text controls inside the native grammar's armed tags so "
            "they cannot truncate a call or forbid literal argument values."
        ),
        removal_condition=(
            "Remove when upstream commits calls identically on the deployed "
            "batch and streaming APIs, preserves model EOS identity, and "
            "suspends caller text controls at the same native grammar boundary."
        ),
        validate_before=_validate_tool_completion_before,
        validate_after=_validate_tool_completion_after,
    ),
    "input-stream-agent-identity": SemanticContract(
        rationale=(
            "An input stream resumes one live request and its acquired KV. "
            "Bind its opaque agent ID before reading chunks and refuse an "
            "ID change before dispatch. A different agent enters through "
            "normal generation admission and initial-prefix acquisition."
        ),
        removal_condition=(
            "Remove when upstream enforces the same stable input-stream "
            "identity and request-local validation/abort behavior."
        ),
        validate_before=_validate_stream_identity_before,
        validate_after=_validate_stream_identity_after,
    ),
    "phase-aware-parser-terminals": SemanticContract(
        rationale=(
            "Qwen reasoning boundaries require atomic token provenance, while "
            "tool grammar recognizes its exact text trigger after reasoning. "
            "Declare token authority by state and update every format consumer "
            "so ordinary-token calls accepted by the grammar survive parsing."
        ),
        removal_condition=(
            "Remove when upstream represents each format's state-dependent "
            "terminal authority and preserves Qwen grammar acceptance across "
            "both added-token and ordinary-token spellings."
        ),
        validate_before=_validate_phase_terminals_before,
        validate_after=_validate_phase_terminals_after,
    ),
    "xml-text-fidelity": SemanticContract(
        rationale=(
            "Whitespace inside an XML string is its value. Preserve those bytes "
            "and surrounding nonempty content on both transports; remove "
            "converter padding removal and the batch-only stripping setting."
        ),
        removal_condition=(
            "Remove when upstream preserves exact XML string and nonempty "
            "content bytes on both transports without a stripping mode."
        ),
        validate_before=_validate_xml_fidelity_before,
        validate_after=_validate_xml_fidelity_after,
    ),
    "raw-image-token-transport": SemanticContract(
        rationale=(
            "Caller tensor/hash/cache-only features bypass image admission. Carry "
            "the original PNG through render/generate and derive native features "
            "and identity from admitted bytes. Validate already-rendered image "
            "spans completely without a second expansion or caller positions."
        ),
        removal_condition=(
            "Remove when upstream supplies the same raw-image-only transport and "
            "complete processed-prompt validation, with no caller cache/tensor "
            "bypass and complete data on processor cache hits."
        ),
        validate_before=_validate_raw_image_before,
        validate_after=_validate_raw_image_after,
    ),
    "token-generation-result-integrity": SemanticContract(
        rationale=(
            "The raw-token endpoint must preserve all choices and actual terminal "
            "causes. Transport-owned engine output, complete terminal evidence and "
            "empty-output forwarding prevent lost choices and false success; "
            "derender preserves the same stop cause on both API transports."
        ),
        removal_condition=(
            "Remove when upstream provides the same complete token result contract, "
            "including sparse parallel updates, terminal errors and stop metadata."
        ),
        validate_before=_validate_generate_result_before,
        validate_after=_validate_generate_result_after,
    ),
    "sampling-decoding-boundary": SemanticContract(
        rationale=(
            "The beam loop bypasses phase budgets, grammar and parsing and drops "
            "the required agent identity, producing a misleading kv_scope error. "
            "Represent supported decoding at request validation and remove the "
            "unreachable Chat/Completion beam conversion and dispatch paths."
        ),
        removal_condition=(
            "Remove when served beam decoding honors the same complete generation "
            "policy, or upstream provides the same truthful shared refusal and "
            "request schema on Chat, Completion, batch and render requests."
        ),
        validate_before=_validate_sampling_boundary_before,
        validate_after=_validate_sampling_boundary_after,
    ),
    "generation-sampling-resolution": SemanticContract(
        rationale=(
            "Constructing engine settings before model defaults and prompt length "
            "rejects valid input against a temporary 16-token cap and loses omitted "
            "settings. Serializing default-valued engine fields loses explicit "
            "request semantics. Preserve supplied values until resolution, then "
            "construct one fresh engine request with consistent server defaults."
        ),
        removal_condition=(
            "Remove when every generation surface resolves configured sampling "
            "and phase ceilings before engine validation, and render-to-generate "
            "serialization preserves all resolved values including neutral ones."
        ),
        validate_before=_validate_sampling_resolution_before,
        validate_after=_validate_sampling_resolution_after,
    ),
    "anthropic-terminal-metadata": SemanticContract(
        rationale=(
            "Mapping only Chat finish_reason loses matched stop sequences and "
            "reports a forced tool call as end_turn. Both transports must derive "
            "native terminal metadata from the observed cause and tool-use output, "
            "and an incomplete stream must never receive a success terminal."
        ),
        removal_condition=(
            "Remove when upstream preserves matched stop strings, reports named "
            "tool output as tool_use, gives truncation precedence, and ends "
            "streaming and batch responses with the same observed metadata."
        ),
        validate_before=_validate_anthropic_terminal_before,
        validate_after=_validate_anthropic_terminal_after,
    ),
    "responses-stream-identity": SemanticContract(
        rationale=(
            "Reparsing streamed output minted replacement item and call IDs in "
            "the terminal response, breaking replay correlation. Finalization "
            "must use the actual streamed items, preserving status and logprobs."
        ),
        removal_condition=(
            "Remove when upstream uses one item identity from stream creation "
            "through terminal output, with equivalent metadata on both transports."
        ),
        validate_before=_validate_responses_identity_before,
        validate_after=_validate_responses_identity_after,
    ),
    "responses-history-integrity": SemanticContract(
        rationale=(
            "Responses replay dropped all but the first text/reasoning block and "
            "bypassed Chat's tool ID correlation. Preserve the supplied content and "
            "validate every positional tool history through one shared boundary."
        ),
        removal_condition=(
            "Remove when upstream preserves all Responses history blocks and "
            "enforces the same tool-result correlation on every rendered surface."
        ),
        validate_before=_validate_responses_history_before,
        validate_after=_validate_responses_history_after,
    ),
    "qwen-single-call-grammar": SemanticContract(
        rationale=(
            "A response filter silently removed model-produced calls when parallel "
            "calls were disabled. The Qwen grammar must own the count, while every "
            "call actually produced remains on both streaming and batch responses."
        ),
        removal_condition=(
            "Remove when upstream enforces Qwen's call count while decoding and "
            "preserves the actual tool output on every response path."
        ),
        validate_before=_validate_single_call_before,
        validate_after=_validate_single_call_after,
    ),
    "kv-physical-free-memory": SemanticContract(
        rationale=(
            "The declared KV capacity cannot spend memory already occupied before "
            "profiling. Bound it by initially free memory minus the profile delta, "
            "recurring activation peak, CUDA graph and frontend reservations."
        ),
        removal_condition=(
            "Remove when upstream's authoritative bound includes pre-snapshot "
            "residents exactly once and keeps utilization as an estimate only."
        ),
        validate_before=_validate_kv_physical_before,
        validate_after=_validate_kv_physical_after,
    ),
    "png-source-admission": SemanticContract(
        rationale=(
            "Pillow presents 16-bit truecolour and grey+alpha PNGs as RGB/RGBA, "
            "silently bypassing the source-pixel contract. Admit only the encoded "
            "IHDR bit depth 8 and colour types 2 or 6 before image conversion."
        ),
        removal_condition=(
            "Remove when pinned upstream validates encoded PNG bit depth and colour "
            "type before Pillow can silently reduce disallowed source samples."
        ),
        validate_before=_validate_png_source_before,
        validate_after=_validate_png_source_after,
    ),
    "qwen-exact-tool-language": SemanticContract(
        rationale=(
            "The parser invented calls from unarmed prose, consumed reserved markup "
            "inside valid parameter values, and diverged in batch. Match the exact "
            "trigger, keep disabled-tool output, carry the exact content IDs through "
            "the reasoning split, and preserve content order and wrapper closure."
        ),
        removal_condition=(
            "Remove when upstream matches the Qwen grammar's trigger and parameter "
            "language, preserves disabled-tool text, and provides equivalent batch "
            "and streaming results with exact reasoning-to-content token handoff."
        ),
        validate_before=_validate_qwen_language_before,
        validate_after=_validate_qwen_language_after,
    ),
    "anthropic-input-fidelity": SemanticContract(
        rationale=(
            "Anthropic tool results silently lost unsupported content and is_error; "
            "renderer ValueErrors became 500s and framework errors used OpenAI "
            "envelopes. Refuse unrenderable input, render the caller's failure flag, "
            "and preserve the shared HTTP classification in Anthropic errors."
        ),
        removal_condition=(
            "Remove when pinned upstream preserves all supported tool-result input "
            "or refuses it and uses the same exception classification with native "
            "Anthropic envelopes at request and streaming boundaries."
        ),
        validate_before=_validate_anthropic_inputs_before,
        validate_after=_validate_anthropic_inputs_after,
    ),
    "turboquant-k8v4-direct-workspace": SemanticContract(
        rationale=(
            "The pinned K8V4 continuation-prefill path dequantized into two reserved "
            "buffers and then allocated two progressively growing final K/V tensors. "
            "K8V4 needs no inverse key rotation, so its dequantization can target the "
            "final FlashAttention-contiguous workspace directly; MSE-key modes retain "
            "their distinct rotated path."
        ),
        removal_condition=(
            "Remove only when pinned upstream independently dequantizes K8V4 into a "
            "bounded final-layout workspace, performs no progressive K/V allocation, "
            "and preserves tested MSE-key behavior."
        ),
        validate_before=_validate_turbo_before,
        validate_after=_validate_turbo_after,
    ),
    "enforce-auto-tool-schema": SemanticContract(
        rationale=(
            "Qwen automatic tool choice must leave the choice to the model while "
            "constraining every begun function name and argument object. OpenAI's "
            "optional strict annotation and XGrammar's strict=false fallback must not "
            "turn a declared Qwen schema into unconstrained JSON."
        ),
        removal_condition=(
            "Remove when upstream offers a pinned fail-closed Qwen parser policy that "
            "enforces all advertised schemas under auto choice regardless of omitted "
            "or false transport strictness, with adversarial token-level tests."
        ),
        validate_before=_validate_schema_before,
        validate_after=_validate_schema_after,
    ),
    "qwen38-agent-defaults-and-thinking": SemanticContract(
        rationale=(
            "Server defaults must reach every protocol, only correctness-first thinking "
            "controls are accepted, a client may lower but never raise the final ceiling, "
            "and Qwen's ID-less prompt representation makes malformed tool-result "
            "correlation unsafe to guess."
        ),
        removal_condition=(
            "Remove only after upstream propagates the same defaults and phase ceilings "
            "through Chat and Anthropic, enforces equivalent thinking policy, and rejects "
            "orphaned, duplicate, missing, and out-of-order tool histories."
        ),
        validate_before=_validate_defaults_before,
        validate_after=_validate_defaults_after,
    ),
    "qwen38-separate-final-response-budget": SemanticContract(
        rationale=(
            "Alibaba specifies separate reasoning and visible-response ceilings. The "
            "visible counter must begin only after a real natural or forced reasoning-end "
            "token sequence; min_tokens cannot override the hard phase ceiling, and a "
            "missing delimiter must not invent a final phase."
        ),
        removal_condition=(
            "Remove when upstream exposes an equivalent validated final-response budget, "
            "tracks both configured reasoning-end forms in the scheduler, and passes the "
            "exact delimiter/min-token/context-boundary tests."
        ),
        validate_before=_validate_phase_before,
        validate_after=_validate_phase_after,
    ),
    "qwen-implicit-tool-grammar-boundary": SemanticContract(
        rationale=(
            "Qwen may begin a tool call directly from reasoning with <tool_call>. That "
            "token simultaneously ends reasoning and begins structured content; trimming "
            "it leaves decoder-time grammar one token behind and the call unconstrained."
        ),
        removal_condition=(
            "Remove when upstream preserves implicit structured-start tokens across the "
            "reasoning boundary and token-sensitivity tests reject invalid tool names and "
            "arguments at the first invalid token."
        ),
        validate_before=_validate_grammar_before,
        validate_after=_validate_grammar_after,
    ),
    "anthropic-validation-http400": SemanticContract(
        rationale=(
            "Typed Pydantic request/translation validation and the engine's own request "
            "validation (VLLMValidationError on the path into the engine -- a generation "
            "naming no kv_scope, for one) are client errors and must be returned as "
            "sanitized Anthropic invalid_request_error HTTP 400. Unexpected server "
            "failures remain logged HTTP 500; the categories must not be merged."
        ),
        removal_condition=(
            "Remove when upstream maps Pydantic and VLLMValidationError failures in both "
            "messages and count_tokens to Anthropic HTTP 400 while preserving generic "
            "exception 500s."
        ),
        validate_before=_validate_anthropic_400_before,
        validate_after=_validate_anthropic_400_after,
    ),
    "tool-truncation-finish-reason": SemanticContract(
        rationale=(
            "A parser may recognize a partial tool prefix at max_tokens. Neither Chat nor "
            "Responses may promote that prefix to an executable terminal: length/incomplete "
            "must survive streaming and batch paths, and Responses must omit arguments.done "
            "and response.completed."
        ),
        removal_condition=(
            "Remove when upstream preserves engine truncation across Chat and Responses "
            "stream/batch parsing, marks every partial item incomplete, and exposes no "
            "successful execution boundary under controlled token cuts."
        ),
        validate_before=_validate_truncation_before,
        validate_after=_validate_truncation_after,
    ),
    "qwen38-vision-runtime": SemanticContract(
        rationale=(
            "The sole deployment requires chronological tool-result media, canonical "
            "lossless/static PNG inputs, the released full pixel budget, BF16 vision, and "
            "enough phase-local VRAM without reducing text context. Strict validation must "
            "occur before I/O; mutually exclusive encoder/text workspaces must release and "
            "restore even on failure, with physical driver-visible headroom."
        ),
        removal_condition=(
            "Remove only when pinned upstream natively preserves tool-media chronology, "
            "provides the same fail-closed image/override contract, avoids the maximum-image "
            "vision MLP temporary, and offers tested exception-safe reclaimable CUDA "
            "workspaces with equivalent full-context/full-image residency."
        ),
        validate_before=_validate_vision_before,
        validate_after=_validate_vision_after,
    ),
    "qwen38-numerical-audits": SemanticContract(
        rationale=(
            "The pinned backend documentation describes a non-existent FP16 value "
            "cache and its one-token test cannot validate attention scores, GQA head "
            "mapping, block addressing, or packed bytes. The deployment requires an "
            "oracle at Qwen3.8's actual BF16 D=256/Hq=24/Hkv=4 geometry."
        ),
        removal_condition=(
            "Remove only when pinned upstream documents the exact 388-byte K8V4 slot "
            "and independently verifies multi-block BF16 GQA store/decode against a "
            "byte-level encoder and explicit attention reference at D=256."
        ),
        validate_before=_validate_numerical_audits_before,
        validate_after=_validate_numerical_audits_after,
    ),
    "turboquant-fail-closed-guards": SemanticContract(
        rationale=(
            "A CPU-only mathematical audit of the deployed K8V4 numerics found "
            "three silent hazards: NaN value vectors laundered into finite "
            "quantized codes, unguarded fp16 metadata overflow for pathological "
            "value ranges, and a silent switch to the incompatible fp8e4b15 key "
            "format on SM < 8.9 hardware. All three now fail closed: insane "
            "value vectors poison their stored metadata with propagating NaN, "
            "prefill chunks refuse non-finite activations outright, and "
            "pre-8.9 devices are rejected. Every guard is bit-neutral for "
            "finite in-range inputs on the deployed hardware."
        ),
        removal_condition=(
            "Remove only when pinned upstream turboquant fails closed on "
            "non-finite inputs and refuses incompatible FP8 key formats "
            "by itself."
        ),
        validate_before=_validate_tq_guards_before,
        validate_after=_validate_tq_guards_after,
    ),
    "kv-offload-host-pinning-fail-closed": SemanticContract(
        rationale=(
            "The CPU KV offload keeps evicted K8V4 pages in host RAM so a "
            "returning subagent does not force re-ingestion, and its whole "
            "value depends on that region being page-locked: unpinned DMA is "
            "the slow path the offload exists to avoid. Upstream logged a "
            "warning when cudaHostRegister failed and carried on, and nothing "
            "downstream records which path was taken, so a failed "
            "registration degraded every transfer for the life of the process "
            "while the deployment still reported healthy. It now raises. The "
            "non-CUDA early return above it is untouched: that is a platform "
            "capability test, not a failure, and other backends legitimately "
            "run the offload without host registration."
        ),
        removal_condition=(
            "Remove only when pinned upstream refuses to start the CPU "
            "offload rather than falling back to unpinned DMA."
        ),
        validate_before=_validate_kv_pin_before,
        validate_after=_validate_kv_pin_after,
    ),
    "shared-prefix-cache-and-user-capacity": SemanticContract(
        rationale=(
            "Both KV tiers are sized from declared full-length user contexts. "
            "A required opaque agent ID selects shared-prefix membership: an "
            "ID with no cached blocks can acquire its initial prefix, and an "
            "existing ID matches its acquired or computed data. GPU and CPU "
            "share one catalog. Whole-context eviction preserves surviving "
            "agents' shared references; sparse CPU chunks advertise only "
            "written data, and secondary storage receives canonical entries. "
            "Every served generation surface uses the common identity gate."
        ),
        removal_condition=(
            "Remove when pinned upstream provides the same user-count sizing, "
            "initial-fork matching, shared GPU/CPU membership, complete-context "
            "retention, and content-accurate offload transfer contracts across "
            "all generation consumers."
        ),
        validate_before=_validate_shared_prefix_cache_before,
        validate_after=_validate_shared_prefix_cache_after,
    ),
    "exact-reasoning-usage": SemanticContract(
        rationale=(
            "The client could only estimate the share of a completion that "
            "was hidden reasoning: Chat Completions served no "
            "completion_tokens_details, and the one reasoning counter in the "
            "tree counted ids between a generated <think> and </think>, which "
            "is zero for this deployment's template -- it pre-fills the "
            "opener into the prompt -- and blind to Qwen's implicit "
            "<tool_call> end. The parser engine already splits reasoning from "
            "content at a token id, so the count is taken there: the ids "
            "before the id the grammar leaves reasoning at, resolved once "
            "from the transition table, advanced on every feed, summed across "
            "choices exactly as completion_tokens is, and served on both chat "
            "paths -- the batch split now receives the generated ids it used "
            "to drop. A grammar without one id-marked boundary, a parser fed "
            "text without its ids, or a parser that missed generated ids "
            "yields no count or a refusal, never an estimate; the field is "
            "absent, not zero, when no exact split was made."
        ),
        removal_condition=(
            "Remove only when pinned upstream serves completion_tokens_details."
            "reasoning_tokens on Chat Completions from the parser's own token-id "
            "boundary, with the prompt-side opener and the implicit tool-call "
            "end both counted correctly, on the streaming and batch paths."
        ),
        validate_before=_validate_reasoning_usage_before,
        validate_after=_validate_reasoning_usage_after,
    ),
}
