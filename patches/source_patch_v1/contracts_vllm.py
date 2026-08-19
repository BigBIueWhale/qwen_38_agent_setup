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
        "min(\n                    final_response_token_budget, server_final_response_budget",
        "final_response_token_budget=final_response_token_budget",
    ):
        _require(needle in sampling, f"{label}: sampling default/clamp missing {needle!r}")
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
        {"check_stop": ("request", "max_model_len")},
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
            "if last_token_id == sampling_params.eos_token_id:",
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
    for qualname in ("create_messages", "count_tokens"):
        source = _symbol_source(state, path, qualname, label=label)
        _require_ordered(
            source,
            (
                "except ValidationError as e:",
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


def validate_final(state: State) -> None:
    """Reassert every durable semantic invariant on the complete tree."""
    for name in (
        "turboquant-k8v4-direct-workspace",
        "enforce-auto-tool-schema",
        "qwen38-agent-defaults-and-thinking",
        "qwen38-separate-final-response-budget",
        "qwen-implicit-tool-grammar-boundary",
        "anthropic-validation-http400",
        "tool-truncation-finish-reason",
        "qwen38-vision-runtime",
        "qwen38-numerical-audits",
        "turboquant-fail-closed-guards",
    ):
        CONTRACTS[name].validate_after(state)


CONTRACTS: Mapping[str, SemanticContract] = {
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
            "Typed Pydantic request/translation validation is a client error and must be "
            "returned as sanitized Anthropic invalid_request_error HTTP 400. Unexpected "
            "server failures remain logged HTTP 500; the categories must not be merged."
        ),
        removal_condition=(
            "Remove when upstream maps Pydantic validation failures in both messages and "
            "count_tokens to Anthropic HTTP 400 while preserving generic exception 500s."
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
}
