"""Semantic contracts for the pinned vLLM source transformations.

The generated landmark module proves byte identity with the reviewed diffs.
This module independently states the defect, the intended behavioral invariant,
and the condition under which each local change should disappear.  A maintainer
cannot bless a new upstream hash without also satisfying these source-structure
checks and the CPU units the check and the build execute.  The vLLM tests a
contract names are review artifacts: they are hashed, and nothing runs them.
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
    # Both final-layout buffers come from one acquisition from the shared
    # workspace manager. Which acquisition is the vision runtime's to pin: it
    # makes the continuation workspace reclaimable around encoding.
    acquisitions = [
        candidate
        for candidate in ast.walk(direct)
        if isinstance(candidate, ast.Assign)
        and ast.unparse(candidate.targets) == "(k_full_buf, v_full_buf)"
        and isinstance(candidate.value, ast.Call)
        and isinstance(candidate.value.func, ast.Attribute)
        and ast.unparse(candidate.value.func.value) == "current_workspace_manager()"
        and [ast.unparse(arg) for arg in candidate.value.args[-2:]]
        == ["(full_shape, qdtype)", "(full_shape, qdtype)"]
    ]
    _require(
        "full_shape = (full_alloc_len, Hk, D)" in direct_source
        and len(acquisitions) == 1,
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
    forbid_text(state, "vllm/entrypoints/chat_utils.py",
                "def validate_tool_result_correlation(", label=label)
    forbid_text(state, anthropic, "class AnthropicThinkingConfig", label=label)
    require_text(state, "vllm/entrypoints/anthropic/serving.py",
                 '"id": block.id or f"call_{int(time.time())}",', label=label)


def _validate_defaults_after(state: State) -> None:
    label = "Qwen3.8 agent defaults result"
    model = "vllm/config/model.py"
    anthropic_protocol = "vllm/entrypoints/anthropic/protocol.py"
    anthropic_serving = "vllm/entrypoints/anthropic/serving.py"
    chat = "vllm/entrypoints/openai/chat_completion/protocol.py"
    chat_utils = "vllm/entrypoints/chat_utils.py"
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
    # One boundary validates a positional tool history before it is rendered;
    # a refusal names where the caller sent the offending message or call.
    _require_in_symbol(
        state, chat, "ChatCompletionRequest._validate_tool_result_correlation", (
            "validate_tool_result_correlation(",
            "ToolHistoryOrigin.chat(self.messages)",
        ), label=label,
    )
    require_python_symbols(state, chat_utils, {
        "ToolHistoryOrigin.chat": ("cls", "messages"),
        "validate_tool_result_correlation": ("messages", "origin"),
    }, label=label)
    correlation = _require_in_symbol(
        state, chat_utils, "validate_tool_result_correlation", (
            "raise VLLMValidationError(", "result_id != expected_id",
            "pending_ids.pop(0)", "parameter=parameter", "parameter=call_parameter",
            "only an assistant message declares tool calls",
        ), label=label,
    )
    forbid_text(state, chat_utils, "messages[{message_index}]", label=label)
    # Anthropic records, as it converts, the message or content block each
    # chat message and call came from, and refuses its history in those terms
    # before building the chat request; a call sent without an id is refused
    # as such, never given one.
    conversion = _require_in_symbol(
        state, anthropic_serving,
        "AnthropicServingMessages._convert_anthropic_to_openai_request", (
            "validate_tool_result_correlation(",
            'messages=origins, calls=call_origins, result_id="tool_use_id"',
        ), label=label,
    )
    _require_ordered(conversion, (
        "validate_tool_result_correlation(", "cls._build_base_request(",
    ), label=label, location="_convert_anthropic_to_openai_request")
    _require_in_symbol(
        state, anthropic_serving, "AnthropicServingMessages._convert_message_content", (
            'block_origin = ("messages", f"{where}.content[{position}]")',
            "origins.extend([block_origin] * (len(openai_messages) - results_before))",
            "call_blocks.extend([block_origin] * (len(tool_calls) - calls_before))",
        ), label=label,
    )
    _require_in_symbol(
        state, anthropic_serving, "AnthropicServingMessages._convert_tool_use_block", (
            '"id": block.id or "",',
        ), label=label,
    )
    forbid_text(state, anthropic_serving, "time.time()", label=label)
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
        "final_response_token_budget = resolve_final_response_token_budget(",
        "final_response_token_budget=final_response_token_budget",
    ):
        _require(needle in sampling, f"{label}: sampling default/clamp missing {needle!r}")
    # A client may lower the server's final-response ceiling, never raise it.
    _require_in_symbol(state, "vllm/sampling_params.py",
        "resolve_final_response_token_budget", (
            "validate_final_response_token_budget(requested)",
            "validate_final_response_token_budget(server_budget)",
            "server_budget if requested is None else min(requested, server_budget)",
        ), label=label)
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
        "def validate_final_response_token_budget(",
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
    _require("def refuse(" not in source, f"{label}: fix already present")
    _require(source.count("except Exception as e:") == 2, f"{label}: generic handlers drifted")


def _validate_anthropic_400_after(state: State) -> None:
    label = "Anthropic validation status result"
    path = "vllm/entrypoints/anthropic/api_router.py"
    protocol = "vllm/entrypoints/anthropic/protocol.py"
    require_python_symbols(
        state,
        path,
        {
            "create_messages": ("request", "raw_request"),
            "count_tokens": ("request", "raw_request"),
            "refuse": ("exc", "route"),
            "translate_error_response": ("response",),
        },
        label=label,
    )
    # A route classifies a failure as the OpenAI surfaces do: typed request
    # validation and the engine's own request validation -- a generation that
    # names no agent, for one -- are client errors, and only a failure the
    # classifier cannot lay at the client's door is a logged server error.
    for name in ("create_messages", "count_tokens"):
        _require_in_symbol(state, path, name, (
            "except Exception as e:", f'return refuse(e, route="{name}")',
        ), label=label)
    _require_in_symbol(state, path, "refuse", (
        "error = create_error_response(exc)",
        "if error.error.code >= HTTPStatus.INTERNAL_SERVER_ERROR.value:",
        "logger.exception(",
        "return translate_error_response(error)",
    ), label=label)
    _require_in_symbol(state, path, "translate_error_response", (
        "status_code=response.error.code", "AnthropicErrorResponse.for_status(",
    ), label=label)
    forbid_text(state, path, "except (ValidationError, VLLMValidationError)", label=label)
    forbid_text(state, path, 'type="internal_error"', label=label)
    for needle in (
        '400: "invalid_request_error"', '500: "api_error"',
        '529: "overloaded_error"', "def for_status(",
    ):
        require_text(state, protocol, needle, label=label)


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
    # The stream hands the limit to the item it closes: a call open at the
    # limit is closed incomplete and sends no arguments.done.
    _require_in_symbol(
        state, events, "SimpleStreamingEventProcessor.close_current",
        ("incomplete=incomplete",),
        label=label,
    )
    # The generation's cut is the one keyword-only parameter of the batch
    # item builder.
    builder = _find_symbol(state, utils, "build_response_output_items", label=label)
    _require(
        [argument.arg for argument in builder.args.kwonlyargs] == ["incomplete"],
        f"{label}: {utils}:build_response_output_items must take incomplete as "
        "its one keyword-only parameter",
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
    # The batch marks items as the stream closes them: the item the limit
    # cut -- the last -- is incomplete, every earlier one completed.
    _require_in_symbol(
        state, utils, "build_response_output_items",
        (
            "if incomplete",
            'return "incomplete" if index == last else "completed"',
        ),
        label=label,
    )
    require_text(state, utils, "status=status(len(outputs)),", count=3, label=label)
    forbid_text(state, utils, 'status="incomplete" if incomplete else "completed"', label=label)
    parser_terminal = ast.unparse(
        _find_symbol(
            state,
            parser,
            "ParserEngine._events_to_delta",
            label=label,
        )
    )
    # A finished parse has no later delta to wait for: it releases every
    # held-back text, in the order the model produced it.
    _require(
        "if finished or not seen_tool_event or (not tool_call_deltas):"
        in parser_terminal
        and "content_parts.insert(0, earlier_deferred)" in parser_terminal
        and "content_parts.append(deferred_after_call)" in parser_terminal,
        f"{label}: terminal parse must release all text in generation order",
    )
    require_python_symbols(state, "tests/parser/engine/test_parser_engine.py", {
        "TestPostToolContentDeferral.test_text_after_tool_released_in_order_when_finished": None,
        "TestPostToolContentDeferral.test_text_held_back_by_an_earlier_delta_comes_first": None,
    }, label=label)


def _validate_vision_before(state: State) -> None:
    label = "Qwen3.8 vision runtime precondition"
    _require(
        "tests/v1/worker/test_workspace.py" not in state,
        f"{label}: new workspace test unexpectedly exists",
    )
    forbid_text(
        state,
        "vllm/entrypoints/chat_utils.py",
        "_enforce_qwen38_strict_content_part",
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
        "if part.get(\"uuid\") is not None:",
        'if part_type == "image_url":',
        'detail not in ("auto", "high")',
        'elif part_type == "input_image":',
    ):
        _require(needle in strict_part, f"{label}: pre-I/O media gate missing {needle!r}")
    # The contract is the served image's own: it holds for every caller and
    # every launch, so none of its gates consults a launch setting.
    _require(
        "envs." not in strict_part,
        f"{label}: the pre-I/O media gate depends on a launch setting",
    )

    for qualname in ("MediaConnector.fetch_image", "MediaConnector.fetch_image_async"):
        connector_source = _symbol_source(state, connector, qualname, label=label)
        _require(
            'if not image_url.startswith(\n            "data:image/png;base64,"' in connector_source,
            f"{label}: {qualname} does not reject non-canonical URLs before I/O",
        )
        _require(
            "if envs." not in connector_source,
            f"{label}: {qualname} gates the image contract on a launch setting",
        )
    for qualname, needle in (
        ("ImageMediaIO.__init__", 'if image_mode != "RGB":'),
        ("ImageMediaIO.__init__", "if self.rgba_background_color != (255, 255, 255):"),
        ("ImageMediaIO.load_base64", 'if media_type != "image/png":'),
    ):
        _require(
            needle in _symbol_source(state, image, qualname, label=label),
            f"{label}: {qualname} does not apply the image contract unconditionally",
        )

    image_source = _symbol_source(state, image, "ImageMediaIO.load_bytes", label=label)
    _require(
        "if envs." not in image_source,
        f"{label}: the decoded-image gate depends on a launch setting",
    )
    for needle in (
        "if w * h > QWEN38_MAX_IMAGE_PIXELS:",
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
    require_text(
        state,
        image,
        "QWEN38_MAX_IMAGE_PIXELS = 16_777_216",
        label=label,
    )
    render_source = _symbol_source(state, render, "ChatParams.with_defaults", label=label)
    for needle in ("if self.media_io_kwargs:", "if self.mm_processor_kwargs:", '"add_vision_id"'):
        _require(needle in render_source, f"{label}: request override gate missing {needle!r}")
    _require(
        "envs." not in render_source,
        f"{label}: the request override gate depends on a launch setting",
    )

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

    require_text(
        state,
        turbo,
        '_CONTINUATION_WORKSPACE_NAME = "turboquant_continuation_prefill"',
        label=label,
    )
    # The encoder's room is the continuation workspace it releases plus what
    # text execution leaves free. A reserve held through text execution and
    # released around encoding adds nothing to that room -- its bytes are free
    # then whether or not it was held -- and only takes them from text
    # execution, so no such reserve, and no setting sizing one, exists.
    for path, retired in (
        (turbo, "reserve_raw_cuda_headroom"),
        (turbo, "VLLM_QWEN38_VISION_HEADROOM_BYTES"),
        (workspace, "reserve_raw_cuda_headroom"),
        (workspace, "cudaMalloc"),
        ("vllm/envs.py", "VLLM_QWEN38_VISION_HEADROOM_BYTES"),
    ):
        forbid_text(state, path, retired, label=label)
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
            "WorkspaceManager.release_reclaimable_workspaces": ("self",),
            "WorkspaceManager.restore_reclaimable_workspaces": ("self",),
            "release_reclaimable_workspaces": (),
        },
        label=label,
    )
    # A reclaimable view is refused while a graph is captured: its allocation
    # is released and restored at a new address, which a graph would keep.
    _require_in_symbol(state, workspace, "WorkspaceManager.get_reclaimable_simultaneous", (
        'if self._device.type == "cuda" and torch.cuda.is_current_stream_capturing():',
    ), label=label)
    require_python_symbols(state, "tests/v1/worker/test_workspace.py", {
        "test_reclaimable_workspace_is_refused_during_graph_capture": None,
    }, label=label)
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
        ("if self._reclaimable_workspaces_released:", 2),
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


def _validate_agent_id_before(state: State) -> None:
    label = "generation agent ID precondition"
    for path in (
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "vllm/entrypoints/openai/completion/protocol.py",
        "vllm/entrypoints/openai/responses/protocol.py",
        "vllm/entrypoints/anthropic/protocol.py",
        "vllm/entrypoints/scale_out/token_in_token_out/protocol.py",
        "vllm/entrypoints/openai/chat_completion/batch_serving.py",
        "vllm/entrypoints/openai/run_batch.py",
        "vllm/sampling_params.py",
        "vllm/v1/engine/__init__.py",
        "vllm/v1/engine/async_llm.py",
        "vllm/engine/protocol.py",
        "vllm/entrypoints/generate/base/serving.py",
    ):
        forbid_text(state, path, "kv_scope", label=label)
    _require_in_symbol(
        state, "vllm/v1/engine/async_llm.py",
        "AsyncLLM.notify_kv_transfer_request_rejected",
        ('extra_args={"kv_transfer_params": dict(kv_transfer_params)},',),
        label=label,
    )
    require_text(
        state, "vllm/entrypoints/openai/chat_completion/api_router.py",
        "async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):",
        label=label,
    )
    require_text(
        state, "vllm/entrypoints/openai/completion/api_router.py",
        "async def create_completion(request: CompletionRequest, raw_request: Request):",
        label=label,
    )
    require_text(
        state, "vllm/entrypoints/scale_out/token_in_token_out/api_router.py",
        "async def generate(request: GenerateRequest, raw_request: Request):",
        label=label,
    )


def _validate_agent_id_after(state: State) -> None:
    label = "generation agent ID result"
    # Every generation request model requires the agent ID, and the served
    # schema says so; the render models it shares its shape with allocate no
    # KV and have no ID. The chat module carries the single conversation and
    # the batch, which names one line of work per conversation.
    chat = "vllm/entrypoints/openai/chat_completion/protocol.py"
    completion = "vllm/entrypoints/openai/completion/protocol.py"
    tokens = "vllm/entrypoints/scale_out/token_in_token_out/protocol.py"
    require_text(
        state, "vllm/entrypoints/openai/engine/protocol.py",
        "KvScope: TypeAlias = Annotated[", label=label,
    )
    for path in (
        chat, completion, tokens,
        "vllm/entrypoints/openai/responses/protocol.py",
        "vllm/entrypoints/anthropic/protocol.py",
    ):
        require_text(state, path, "    kv_scope: KvScope\n", label=label)
        forbid_text(state, path, "kv_scope: str | None", label=label)
    require_text(state, chat, "class ChatCompletionGenerationRequest(ChatCompletionRequest):",
                 label=label)
    require_text(state, chat, "kv_scope: list[KvScope] = Field(", label=label)
    require_text(state, chat, "ChatCompletionGenerationRequest.model_validate(", label=label)
    require_text(state, chat, 'require_one_sequence(data.get("n"), "n")', count=2, label=label)
    require_text(state, completion,
                 "class CompletionGenerationRequest(CompletionRequest):", label=label)
    require_text(state, completion,
                 'require_one_sequence(count, "prompt", "the number of prompts")',
                 label=label)
    require_text(state, tokens, "class TokenGenerationRequest(GenerateRequest):", label=label)
    # Each generation route takes the generation model; rendering keeps the
    # shared shape.
    for path, text in (
        ("vllm/entrypoints/openai/chat_completion/api_router.py",
         "    request: ChatCompletionGenerationRequest, raw_request: Request\n"),
        ("vllm/entrypoints/openai/completion/api_router.py",
         "    request: CompletionGenerationRequest, raw_request: Request\n"),
        ("vllm/entrypoints/scale_out/token_in_token_out/api_router.py",
         "async def generate(request: TokenGenerationRequest, raw_request: Request):"),
        ("vllm/entrypoints/openai/chat_completion/batch_serving.py",
         "single_requests = request.to_chat_completion_requests()"),
        ("vllm/entrypoints/anthropic/serving.py",
         "return ChatCompletionGenerationRequest("),
        ("vllm/entrypoints/openai/run_batch.py",
         "return ChatCompletionGenerationRequest.model_validate(value)"),
        # /invocations dispatches by the generation endpoints' own types.
        ("vllm/entrypoints/generate/factories.py",
         "(ChatCompletionGenerationRequest, (chat, create_chat_completion)),"),
        ("vllm/entrypoints/generate/factories.py",
         "(CompletionGenerationRequest, (completion, create_completion)),"),
    ):
        require_text(state, path, text, label=label)
    # The same rule holds for every caller at the one boundary every
    # generation passes, stated once and reused by the request models.
    sampling = "vllm/sampling_params.py"
    for text in (
        "def require_kv_scope_value(value: object) -> str:",
        "def require_one_sequence(",
        "send the same kv_scope for every request that continues a conversation, ",
        "and a new, never-used kv_scope for a fork or a subagent",
    ):
        require_text(state, sampling, text, label=label)
    # The engine request type holds the rule: no generation can be built or
    # decoded without the ID, whichever producer builds it -- the input
    # processor, a caller that builds an engine request itself, or the
    # engine's own notice that a KV-transfer request was refused.
    _require_ordered(
        _symbol_source(state, sampling, "require_kv_scope", label=label),
        (
            "scope = require_kv_scope_value(",
            'params.extra_args.get("kv_scope") if params.extra_args else None',
            'require_one_sequence(params.n, "n")',
        ),
        label=label,
        location=sampling,
    )
    _require_in_symbol(state, "vllm/v1/engine/__init__.py", "EngineCoreRequest.__post_init__", (
        "if self.sampling_params is not None:",
        "require_kv_scope(self.sampling_params)",
    ), label=label)
    forbid_text(state, "vllm/v1/engine/input_processor.py", "kv_scope", label=label)
    _require_in_symbol(
        state, "vllm/v1/engine/async_llm.py",
        "AsyncLLM.notify_kv_transfer_request_rejected",
        ("kv_scope: str,", '"kv_scope": kv_scope,'),
        label=label,
    )
    require_python_symbols(state, "vllm/engine/protocol.py", {
        "EngineClient.notify_kv_transfer_request_rejected": (
            "self", "request_id", "kv_transfer_params", "kv_scope",
            "data_parallel_rank",
        ),
    }, label=label)
    _require_in_symbol(
        state, "vllm/entrypoints/generate/base/serving.py",
        "GenerateBaseServing._with_kv_transfer_rejection_cleanup",
        ("ChatCompletionGenerationRequest", "CompletionGenerationRequest",
         "request.kv_scope,"),
        label=label,
    )
    require_python_symbols(state, "tests/entrypoints/test_kv_scope_protocol.py", {
        "test_invocations_dispatch_by_the_generation_request_types": None,
    }, label=label)
    require_python_symbols(state, "tests/v1/engine/test_engine_request_identity.py", {
        "test_a_generation_without_an_identity_cannot_be_built": None,
        "test_a_generation_of_several_sequences_cannot_be_built": None,
        "test_a_pooling_request_names_no_line_of_work": None,
        "test_the_engine_decodes_the_identity_it_was_sent": None,
        "test_a_rejected_remote_prefill_notifies_under_the_requests_own_scope": None,
        "test_a_request_refused_as_it_is_decoded_is_that_requests_error": None,
    }, label=label)
    # A request the engine refuses as it decodes it is that request's error,
    # returned to its client; the input thread it used to end goes on.
    core = "vllm/v1/engine/core.py"
    _require_ordered(
        _symbol_source(state, core, "EngineCoreProc._receive_add_request", label=label),
        (
            "add_request_decoder.decode(data_frames)",
            "except VLLMValidationError:",
            "self._handle_refused_add_request(data_frames)",
            "return None",
            "return self.preprocess_add_request(req)",
        ),
        label=label,
        location=f"{core}:EngineCoreProc._receive_add_request",
    )
    _require_in_symbol(state, core, "EngineCoreProc.process_input_sockets", (
        "request = self._receive_add_request(",
    ), label=label)
    _require_in_symbol(state, core, "EngineCoreProc._handle_refused_add_request", (
        "self._send_error_outputs_to_client([request_id], client_index)",
    ), label=label)
    # A request a test builds is a request the server admits: it names its
    # line of work.
    kimi_k3_tests = "tests/tool_use/test_kimi_k3_tool_parser.py"
    require_text(state, kimi_k3_tests, '        kv_scope="agent",\n', count=2, label=label)
    require_text(state, kimi_k3_tests, '            "kv_scope": "agent",\n', label=label)


def _validate_attention_prefix_hash_before(state: State) -> None:
    label = "attention prefix hash precondition"
    block_pool = "vllm/v1/core/block_pool.py"
    # added: the pool drops a grown block's earlier hash whatever the block holds.
    forbid_text(state, block_pool, "is_recurrent", label=label)
    require_text(
        state, block_pool,
        "removed_hashes = self._remove_cached_block_hashes(blk)", label=label,
    )


def _validate_attention_prefix_hash_after(state: State) -> None:
    label = "attention prefix hash result"
    # Attention growth keeps the block's earlier immutable prefix reachable.
    require_text(state, "vllm/v1/core/block_pool.py", "*, is_recurrent: bool,",
                 count=2, label=label)


def _validate_grouped_geometry_before(state: State) -> None:
    label = "grouped KV spec geometry precondition"
    kv_utils = "vllm/v1/core/kv_cache_utils.py"
    # added: classification reads the group's wrapper spec.
    forbid_text(state, kv_utils, "get_kv_cache_spec_for_block_geometry", label=label)
    require_text(state, kv_utils, "if isinstance(g.kv_cache_spec, AttentionSpec)",
                 label=label)


def _validate_grouped_geometry_after(state: State) -> None:
    label = "grouped KV spec geometry result"
    kv_utils = "vllm/v1/core/kv_cache_utils.py"
    offload_config = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py"
    scheduler = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    # added: every classification resolves a grouped spec to its layers' geometry.
    require_python_symbols(state, kv_utils, {
        "get_kv_cache_spec_for_block_geometry": ["kv_cache_spec"],
    }, label=label)
    _require_in_symbol(state, kv_utils, "resolve_kv_cache_block_sizes", (
        "group_specs = [get_kv_cache_spec_for_block_geometry(g.kv_cache_spec) for g in groups]",
        "for spec in group_specs",
    ), label=label)
    require_text(
        state, offload_config,
        "if isinstance(get_kv_cache_spec_for_block_geometry(group.kv_cache_spec), AttentionSpec)",
        label=label,
    )
    require_text(state, offload_config, '"mamba_cache_mode", None)', label=label)
    require_text(state, scheduler, "if spec.config.groups[idx].mamba_cache_mode in (",
                 label=label)
    require_text(
        state, scheduler,
        'requires_cow_source=spec.config.groups[idx].mamba_cache_mode == "align",',
        label=label,
    )
    # That the window classification resolves the grouped spec is pinned by
    # kv-capacity-in-declared-users, the stage that moves the classification
    # to the offloading boundary: before it, where this stage writes it, and
    # after it, where it stays.


def _validate_agent_retention_before(state: State) -> None:
    label = "agent-grouped retention precondition"
    spec = "vllm/v1/kv_offload/cpu/spec.py"
    manager = "vllm/v1/kv_offload/cpu/manager.py"
    # The policy-selecting world this stage replaces.
    require_text(
        state, spec, 'self.extra_config.get("eviction_policy", "lru")', label=label
    )
    require_text(state, manager, "CachePolicyFactory", count=3, label=label)
    require_text(
        state, "vllm/v1/kv_offload/cpu/policies/factory.py", '"arc"', label=label
    )


def _validate_agent_retention_after(state: State) -> None:
    label = "agent-grouped retention result"
    spec = "vllm/v1/kv_offload/cpu/spec.py"
    manager = "vllm/v1/kv_offload/cpu/manager.py"
    tiering = "vllm/v1/kv_offload/tiering/manager.py"
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
    require_python_symbols(state, manager, {
        "CPUOffloadingManager.__init__": ["self", "num_blocks", "enable_events"],
    }, label=label)
    # added: no spec reads a policy or reuse-count store-filter key.
    for path in (spec, "vllm/v1/kv_offload/tiering/spec.py"):
        for text in ("eviction_policy", "store_threshold"):
            forbid_text(state, path, text, label=label)
    forbid_text(state, "vllm/v1/kv_offload/cpu/common.py", "STORES_SKIPPED", label=label)
    # No metrics test configures the reuse threshold or asserts its counter.
    forbid_text(state,
                "tests/v1/kv_connector/unit/offloading_connector/test_metrics.py",
                "store_threshold", label=label)

    # Lookup matches content and cache_salt alone, in every tier, as upstream
    # does: no membership view, acquisition or per-agent catalog exists to
    # make an ID with cached blocks miss data another ID computed.
    for path in (
        "vllm/v1/core/block_pool.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
        manager,
        "vllm/v1/kv_offload/base.py",
        tiering,
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    ):
        for text in ("prefix_cache.", "PrefixCache", "cache_view", "begin_lookup"):
            forbid_text(state, path, text, label=label)
    # The agent ID groups retention. Whole agents are released in
    # least-recently-used order, never the storing one; references held by
    # surviving contexts protect shared chunks; retained contexts are bounded
    # by the chunk count; sparse rows record the data actually written.
    for text in (
        "self._references.setdefault(key, set()).add(req_id)",
        "not required or not self._has_content(key, required, req_context)",
        "self._references.get(key, set()) <= released_requests",
        "if victim == agent_id:",
        "self._forget_agent(victim)",
        "if self._blocks[key].ref_cnt != 0:",
        "while len(self._idle_context) > self._num_blocks:",
        "        if key not in self._complete_blocks:",
    ):
        require_text(state, manager, text, label=label)
    require_text(state, manager, "self._available_content[key] = content[key]",
                 count=2, label=label)
    require_text(state, tiering, "if success and complete_keys:", label=label)
    # The engine request carries the ID to the offload connector.
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
    require_python_symbols(state, "tests/v1/engine/test_engine_request_identity.py", {
        "test_the_notice_reaches_the_offload_tier_as_a_request_of_that_agent": None,
        "test_a_pooling_model_is_refused_an_agent_scoped_offload_tier": None,
    }, label=label)
    # A pooling request carries no agent ID, so a spec whose manager accounts
    # per agent is refused a pooling model where the connector is built.
    require_text(state, "vllm/v1/kv_offload/base.py",
                 "ACCOUNTS_KV_PER_AGENT: ClassVar[bool] = False", label=label)
    require_text(state, "vllm/v1/kv_offload/cpu/spec.py",
                 "ACCOUNTS_KV_PER_AGENT = True", label=label)
    _require_ordered(
        _symbol_source(state,
                       "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
                       "OffloadingConnector.__init__", label=label),
        (
            "spec_cls.ACCOUNTS_KV_PER_AGENT",
            'vllm_config.model_config.runner_type == "pooling"',
            "raise ValueError(",
            "offloading_config = build_offloading_config(vllm_config, kv_cache_config)",
        ),
        label=label,
        location="OffloadingConnector.__init__",
    )
    # Whether a context is idle is the idle map's to say; no second flag.
    forbid_text(state, manager, "context.active", label=label)


def _validate_agentless_routes_before(state: State) -> None:
    label = "agentless generation routes precondition"
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


def _validate_agentless_routes_after(state: State) -> None:
    label = "agentless generation routes result"
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
    # No surface may be mounted that reaches the engine without being able to
    # name an agent: neither /generative_scoring nor the Cohere chat endpoint
    # is registered or served.
    forbid_text(
        state, generate_router, "register_generative_scoring_api_router", label=label
    )
    forbid_text(state, generate_router, "ServingGenerativeScoring", label=label)


def _validate_declared_capacity_before(state: State) -> None:
    label = "declared KV capacity precondition"
    spec = "vllm/v1/kv_offload/cpu/spec.py"
    cache = "vllm/config/cache.py"
    scheduler = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    # The byte-denominated world this stage replaces.
    require_text(
        state,
        spec,
        'cpu_bytes_to_use must be specified in kv_connector_extra_config',
        label=label,
    )
    require_text(state, cache, "kv_offloading_size: float | None = None", label=label)
    # The window classification this stage moves, resolving the grouped spec
    # to its layers' geometry as grouped-kv-specs-use-layer-geometry wrote it.
    _require_in_symbol(state, scheduler, "get_sliding_window_size_in_chunks", (
        "kv_cache_spec = get_kv_cache_spec_for_block_geometry(kv_cache_spec)",
    ), label=label)


def _validate_declared_capacity_after(state: State) -> None:
    label = "declared KV capacity result"
    spec = "vllm/v1/kv_offload/cpu/spec.py"
    cache = "vllm/config/cache.py"
    kv_utils = "vllm/v1/core/kv_cache_utils.py"
    scheduler = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    require_text(state, spec, "cpu_kv_cache_users must be specified", label=label)
    # A key the spec does not act on is refused, naming the spec; only the
    # tiering spec, which certifies a canonical layout, accepts that request.
    require_text(state, spec, "does not act on kv_connector_extra_config keys",
                 label=label)
    forbid_text(state, spec, '"canonical_layout",', label=label)
    require_text(state, "vllm/v1/kv_offload/tiering/spec.py", '"canonical_layout",',
                 label=label)
    require_python_symbols(state, "tests/v1/kv_offload/test_factory.py", {
        "test_only_the_spec_that_certifies_a_canonical_layout_accepts_it": None,
    }, label=label)
    require_text(state, spec, "self.num_blocks = cpu_kv_cache_users * chunks_per_user", label=label)

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
    # added: the backend selector and the environment switch existed only to
    # route the removed byte size; neither survives it.
    forbid_text(state, cache, "kv_offloading_backend", label=label)
    forbid_text(state, "vllm/engine/arg_utils.py", "kv_offloading_backend", label=label)
    forbid_text(state, "vllm/envs.py", "VLLM_USE_SIMPLE_KV_OFFLOAD", label=label)
    forbid_text(state, "vllm/config/vllm.py", "VLLM_USE_SIMPLE_KV_OFFLOAD", label=label)
    # The window classification is derived once at the offloading boundary,
    # from the grouped spec's layer geometry.
    require_text(
        state,
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
        "def get_sliding_window_size_in_chunks(",
        label=label,
    )
    _require_in_symbol(
        state,
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
        "get_sliding_window_size_in_chunks",
        ("kv_cache_spec = get_kv_cache_spec_for_block_geometry(kv_cache_spec)",),
        label=label,
    )
    forbid_text(
        state, scheduler, "def get_sliding_window_size_in_chunks(", label=label
    )


def _validate_physical_bound_before(state: State) -> None:
    label = "KV declaration physical bound precondition"
    worker = "vllm/v1/worker/gpu_worker.py"
    # added: the worker reports only the utilization estimate.
    forbid_text(state, worker, "kv_physical_bound", label=label)
    require_text(
        state, worker,
        "maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)",
        label=label,
    )


def _validate_physical_bound_after(state: State) -> None:
    label = "KV declaration physical bound result"
    worker = "vllm/v1/worker/gpu_worker.py"
    kv_utils = "vllm/v1/core/kv_cache_utils.py"
    # added: the worker returns the physical bound, the declaration is checked
    # against it, and no refusal advises raising the utilization estimate.
    _require_ordered(
        _symbol_source(state, worker, "Worker.determine_available_memory", label=label),
        ("kv_physical_bound", "return reserve_mm_ipc_gpu_memory(", "kv_physical_bound"),
        label=label,
        location=f"{worker}:Worker.determine_available_memory",
    )
    require_text(
        state, kv_utils,
        "The declaration is AUTHORITATIVE against the bound the workers", label=label,
    )
    # The bound is the memory free at startup profiling, not the card's, and
    # the refusal names the next actions that exist.
    require_text(state, kv_utils, 'f"but the KV bound is only "', label=label)
    require_text(state, kv_utils, 'f"memory free at startup profiling, minus the residents it "',
                 label=label)
    _require_in_symbol(state, kv_utils, "get_kv_cache_configs", (
        '"free the device memory other processes hold",',
        '*(["declare fewer users"] if users > 1 else []),',
        '"reduce max_model_len",',
    ), label=label)
    require_python_symbols(state, "tests/v1/core/test_kv_cache_users_sizing.py", {
        "test_one_declared_user_is_not_told_to_declare_fewer": None,
    }, label=label)
    forbid_text(state, kv_utils, "total device memory minus", label=label)
    forbid_text(state, kv_utils, "Try increasing `gpu_memory_utilization`", label=label)


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
    # Counts of their own beside it: a zero default for a parser that cannot
    # count, depth counts on the text-split parsers, and two format overrides.
    require_text(
        state, "vllm/reasoning/abs_reasoning_parsers.py",
        "# By default, assume the parser cannot detect reasoning spans.", label=label,
    )
    require_text(
        state, "vllm/reasoning/basic_parsers.py",
        "Uses a depth counter so nested spans are handled safely", label=label,
    )
    require_text(
        state, "vllm/reasoning/minimax_m3_reasoning_parser.py",
        "depth = 1 if self._initial_in_reasoning else 0", label=label,
    )
    require_text(
        state, "vllm/parser/kimi_k2.py",
        "return super().count_reasoning_tokens(token_ids)", label=label,
    )
    require_text(state, "vllm/parser/inkling.py", "def count_reasoning_tokens(", label=label)
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
            "ParserEngine._ids_before_boundary": ("self", "token_ids"),
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
            "counted = self._ids_before_boundary(delta_token_ids)",
        ),
        label=label,
    )
    _require_in_symbol(
        state,
        parser,
        "ParserEngine._ids_before_boundary",
        ("if token_id in self._reasoning_boundary_ids:",),
        label=label,
    )
    # The whole-generation count is that number, by that scan: absent with
    # it, never an error, and no format or text-split parser keeps another.
    _require_in_symbol(
        state,
        parser,
        "ParserEngine.count_reasoning_tokens",
        (
            "if self._reasoning_boundary_refusal is not None:\n            return None",
            "return self._ids_before_boundary(token_ids)",
        ),
        label=label,
    )
    _require(
        "raise" not in _symbol_source(
            state, parser, "ParserEngine.count_reasoning_tokens", label=label),
        f"{label}: the whole-generation reasoning count raises",
    )
    counted = "    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int | None:\n"
    basic = "vllm/reasoning/basic_parsers.py"
    for path in (parser, adapters, "vllm/reasoning/abs_reasoning_parsers.py", basic):
        require_text(state, path, counted, label=label)
    for path, qualname in (
        ("vllm/reasoning/abs_reasoning_parsers.py", "ReasoningParser.count_reasoning_tokens"),
        (basic, "BaseThinkingReasoningParser.count_reasoning_tokens"),
    ):
        _require_in_symbol(
            state, path, qualname, ('"""\n        return None',), label=label
        )
    forbid_text(state, basic, "Uses a depth counter", label=label)
    for path in (
        "vllm/reasoning/minimax_m3_reasoning_parser.py",
        "vllm/parser/kimi_k2.py",
        "vllm/parser/inkling.py",
    ):
        forbid_text(state, path, "def count_reasoning_tokens(", label=label)
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

    # One reader of the count, beside the parsers: every surface reports it
    # from the parser that split the output, refuses a parser that missed
    # generated ids, and never estimates.
    require_python_symbols(
        state,
        abstract,
        {"reasoning_token_usage": ("parser", "generated_token_count")},
        label=label,
    )
    _require_in_symbol(
        state,
        abstract,
        "reasoning_token_usage",
        (
            "count = parser.reasoning_token_count",
            "if parser.generated_token_count != generated_token_count:",
            "reasoning token accounting refused",
        ),
        label=label,
    )
    require_python_symbols(
        state,
        chat,
        {"_make_completion_tokens_details": ("reasoning_token_counts",)},
        label=label,
    )
    forbid_text(state, chat, "def _reasoning_token_count(", label=label)
    require_text(state, chat, "reasoning_token_usage(", count=5, label=label)
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
            "reasoning_token_usage(parser, len(token_ids))",
            "reasoning_token_counts.append(None)",
            "completion_tokens_details=_make_completion_tokens_details(",
        ),
        label=label,
    )
    for absent in ("estimate", "len(reasoning)", "tokenizer.encode(reasoning"):
        forbid_text(state, chat, absent, label=label)

    # Responses reads the same count from the same parser -- per generation,
    # summed over a parsable context's turns -- and serves it absent, never
    # zero and never a failed response, when no exact split was made.
    responses_context = "vllm/entrypoints/openai/responses/context.py"
    responses_serving = "vllm/entrypoints/openai/responses/serving.py"
    _require_in_symbol(
        state, responses_context, "SimpleContext.num_reasoning_tokens",
        ("reasoning_token_usage(self.response_parser, self.num_output_tokens)",),
        label=label,
    )
    _require_in_symbol(
        state, responses_context, "ParsableContext.append_output",
        ("reasoning_token_usage(self.response_parser, len(completion.token_ids))",
         "self._turn_reasoning_tokens.append(None)"),
        label=label,
    )
    _require_in_symbol(
        state, responses_context, "ParsableContext.num_reasoning_tokens",
        ("if None in self._turn_reasoning_tokens:",), label=label,
    )
    # Harmony counts its own reasoning channel; the parsed contexts read the
    # parser and keep no counter beside it.
    for context in ("SimpleContext.__init__", "ParsableContext.__init__"):
        _require(
            "self.num_reasoning_tokens" not in _symbol_source(
                state, responses_context, context, label=label),
            f"{label}: {context} keeps a reasoning counter beside the parser's",
        )
    forbid_text(state, responses_serving, "count_reasoning_tokens(", label=label)
    require_text(
        state, responses_serving,
        "num_reasoning_tokens = context.num_reasoning_tokens", label=label,
    )
    require_text(
        state, "vllm/entrypoints/openai/responses/protocol.py",
        "    reasoning_tokens: int | None = None\n", label=label,
    )
    require_python_symbols(
        state,
        "tests/entrypoints/openai/responses/test_reasoning_usage_context.py",
        {
            "test_the_context_reads_the_parsers_count": None,
            "test_a_parser_that_split_other_ids_is_refused": None,
        },
        label=label,
    )

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
            "TestOneCount.test_every_engine_format_counts_a_generation_as_it_feeds_it": None,
            "TestOneCount.test_a_parser_that_splits_on_text_reports_no_count": None,
        },
        label=label,
    )
    require_text(
        state, test, "assert engine.count_reasoning_tokens([65, 99, 66]) is None",
        label=label,
    )
    require_text(
        state, "tests/reasoning/test_base_thinking_reasoning_parser.py",
        "assert parser.count_reasoning_tokens(token_ids) is None", label=label,
    )
    require_text(
        state, "tests/reasoning/test_minimax_m3_reasoning_parser.py",
        "assert parser.count_reasoning_tokens(output_ids) is None", count=3, label=label,
    )


def _validate_anthropic_inputs_before(state: State) -> None:
    forbid_text(
        state, "vllm/entrypoints/anthropic/protocol.py",
        "def is_anthropic_api_path(", label="Anthropic input fidelity precondition",
    )


def _validate_anthropic_inputs_after(state: State) -> None:
    label = "Anthropic input fidelity result"
    serving = "vllm/entrypoints/anthropic/serving.py"
    errors = "vllm/entrypoints/serve/exception_handling/error_response.py"
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
            "TestToolHistoryRefusal.test_a_refusal_names_the_block_the_caller_sent": None,
            "TestToolHistoryRefusal."
            "test_an_unmerged_inline_system_message_is_named_where_it_was_sent": None,
            "TestToolHistoryRefusal.test_a_correlated_history_converts_whole": None,
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
                 '(ParserState.TOOL_BETWEEN, "TOOL_END")',
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
    # The language is the grammar's, so the Qwen tool parser is neither
    # selected nor built while strict tool calling arms no grammar.
    _require_in_symbol(state, "vllm/tool_parsers/qwen3_engine_tool_parser.py",
        "Qwen3EngineToolParser.require_servable", (
            "if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:",
            "Next: unset VLLM_ENFORCE_STRICT_TOOL_CALLING",
        ), label=label)
    _require_in_symbol(state, "vllm/parser/parser_manager.py",
        "ParserManager.get_tool_parser", ("parser.require_servable()",),
        label=label)
    _require_in_symbol(state, "vllm/tool_parsers/abstract_tool_parser.py",
        "ToolParser.__init__", ("self.require_servable()",), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen3.py", {
        "test_the_qwen_tool_parser_is_never_served_without_its_grammar": None,
    }, label=label)
    require_python_symbols(state,
        "tests/parser/engine/test_reasoning_token_count.py", {
            "TestStreaming.test_boundary_ids_wait_for_detokenized_text": None,
        }, label=label)
    # Every override of the extraction this stage gives the content's ids
    # takes them and hands them on: an override that does not is a TypeError
    # on every complete parse of its format.
    _require_in_symbol(state, "vllm/parser/kimi_k3.py", "KimiK3Parser._extract_tool_calls", (
        "content_token_ids: Sequence[int] = (),",
        "content, request, enable_auto_tools, content_token_ids",
    ), label=label)
    _require_in_symbol(state, "vllm/parser/mistral.py",
        "MistralParser.extract_tool_calls_from_content", (
            "content_token_ids: Sequence[int] = (),",
            "content, request, content_token_ids",
        ), label=label)


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
    worker = "vllm/v1/worker/gpu_worker.py"
    runner = "vllm/v1/worker/gpu_model_runner.py"
    _require_in_symbol(state, worker, "Worker.determine_available_memory", (
        "kv_physical_bound = ",
        "self.init_snapshot.free_memory\n"
        "            - profile_result.non_kv_cache_memory\n"
        "            - cudagraph_memory_estimate_applied",
    ), label=label)
    # What the bound subtracts is what serving holds beside its pool: a warm
    # pass outside the window, then the measured pass with a stand-in pool
    # living only inside it and its bytes taken off the peak it held.
    _require_ordered(
        _symbol_source(state, worker, "Worker.determine_available_memory", label=label),
        (
            "with profiling_kv_cache():",
            "profile()",
            "memory_profiling(",
            "profiling_kv_cache() as stand_in_pool_bytes,",
            "profile()",
            "profile_result.transient_peak_headroom -= stand_in_pool_bytes",
            "profile_result.non_kv_cache_memory -= stand_in_pool_bytes",
        ),
        label=label,
        location=f"{worker}:Worker.determine_available_memory",
    )
    # Each phase runs in the residency serving runs it in: the encoder with
    # the reclaimable workspace released, the text step with it resident,
    # then its sampler and its prompt log probabilities at the most a request
    # is admitted with. That the text step ran attention and every state
    # update is witnessed by the step itself and refused at startup when it
    # is missing, so no argument the step is called with is pinned here.
    _require_ordered(
        _symbol_source(state, runner, "GPUModelRunner.profile_served_phases",
                       label=label),
        (
            "with release_reclaimable_workspaces():",
            "self._run_dummy_encoder()",
            "num_logprobs = self._largest_admitted_logprob_count()",
            "with manager.witness_reclaimable_requests() as requested:",
            "self._run_dummy_text_step(",
            "layer_reads=layer_reads,",
            "self._require_text_step_ran(",
        ),
        label=label,
        location=f"{runner}:GPUModelRunner.profile_served_phases",
    )
    # The witnesses are produced where the served code runs: the dummy step
    # records which layers read the metadata it built, and the workspace
    # manager which reclaimable workspaces were requested.
    _require_in_symbol(state, runner, "GPUModelRunner._dummy_run", (
        "attn_metadata = _record_layer_reads(attn_metadata, layer_reads)",
    ), label=label)
    _require_in_symbol(state, "vllm/v1/worker/workspace.py",
                       "WorkspaceManager.get_reclaimable_simultaneous", (
        "self._witnessed_requests.add(name)",
    ), label=label)
    _require_ordered(
        _symbol_source(state, runner, "GPUModelRunner._run_dummy_text_step",
                       label=label),
        (
            "self._dummy_sampler_run(last_hidden_states, num_logprobs)",
            "self._dummy_prompt_logprobs_run(hidden_states, num_logprobs)",
        ),
        label=label,
        location=f"{runner}:GPUModelRunner._run_dummy_text_step",
    )
    # Serving scores a chunk's prompt through the function the profile runs
    # at its largest, never a float32 log-softmax over the vocabulary for
    # every row of the chunk.
    for symbol in ("GPUModelRunner._get_prompt_logprobs_dict",
                   "GPUModelRunner._dummy_prompt_logprobs_run"):
        _require_in_symbol(state, runner, symbol, (
            "compute_prompt_logprobs_with_chunking(",
        ), label=label)
    _require(
        "self.sampler.compute_logprobs(" not in _symbol_source(
            state, runner, "GPUModelRunner._get_prompt_logprobs_dict", label=label
        ),
        f"{label}: prompt log probabilities take a full-vocabulary log-softmax "
        "the profile does not run",
    )
    _require_ordered(
        _symbol_source(state, runner, "GPUModelRunner.profiling_kv_cache", label=label),
        (
            "stand_in = self._init_minimal_kv_cache_for_profiling()",
            "yield stand_in.num_blocks * _pool_bytes_per_block(",
            "finally:",
            "self._cleanup_profiling_kv_cache()",
        ),
        label=label,
        location=f"{runner}:GPUModelRunner.profiling_kv_cache",
    )
    _require_in_symbol(state, runner, "GPUModelRunner._dummy_run", (
        "np.maximum(num_scheduled_tokens, profile_seq_lens)",
    ), label=label)
    # The V2 runner, whose profile runs no attention, holds no declared pool.
    _require_in_symbol(state, "vllm/config/vllm.py",
                       "VllmConfig._get_v2_model_runner_unsupported_features", (
        "if self.cache_config.kv_cache_users is not None:",
        '"a KV pool declared with --kv-cache-users',
    ), label=label)
    require_python_symbols(state, "tests/test_config.py", {
        "test_v2_model_runner_serves_no_declared_kv_pool": None,
    }, label=label)
    forbid_text(state, worker, "Residents allocated after profiling", label=label)
    require_python_symbols(state, "tests/v1/worker/test_gpu_worker.py", {
        "test_physical_bound_charges_preexisting_residents_once": None,
        "test_served_profile_encodes_with_workspace_released_and_attends_in_text": None,
        "test_served_profile_refuses_a_text_step_that_skipped_what_serving_runs": None,
        "test_largest_admitted_logprob_count": None,
        "test_served_text_step_samples_then_scores_the_prompt": None,
        "test_dummy_prompt_logprobs_scores_every_prompt_row_of_the_step": None,
        "test_prompt_logprobs_are_scored_by_the_profiled_function": None,
        "test_profiling_kv_cache_yields_stand_in_bytes_and_always_removes_it": None,
        "test_dummy_context_covers_its_own_query": None,
    }, label=label)
    require_python_symbols(state, "tests/v1/worker/test_workspace.py", {
        "test_witness_names_the_reserved_workspaces_a_block_never_requested": None,
    }, label=label)


def _validate_single_call_before(state: State) -> None:
    require_text(state, "vllm/entrypoints/openai/chat_completion/serving.py",
                 "choice_data = maybe_filter_parallel_tool_calls(choice_data, request)",
                 count=2, label="Call-count grammar precondition")


def _validate_single_call_after(state: State) -> None:
    label = "Call-count grammar result"
    # This stage's durable guarantee is that the response layer no longer drops
    # calls and that the request's limit reaches the tag builder. Where the
    # limit is then applied is owned by the Qwen grammar contract: this stage
    # applied it to XGrammar's returned tag, the Qwen builder now applies it
    # while building, and this validator also runs against the complete tree.
    _require_in_symbol(state, "vllm/tool_parsers/structural_tag_registry.py",
                       "get_model_structural_tag", (
        "parallel_tool_calls",
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
    forbid_text(state, "vllm/entrypoints/openai/responses/utils.py",
                "validate_tool_result_correlation", label=label)


def _validate_responses_history_after(state: State) -> None:
    label = "Responses history result"
    utils = "vllm/entrypoints/openai/responses/utils.py"
    # One boundary for every surface; each surface says where each message
    # and call came from, so a refusal names the field the caller sent.
    _require_in_symbol(state, utils, "construct_input_messages", (
        "list(prev_response_output or [])",
        "_convert_response_items(",
        'ToolHistoryOrigin(messages=origins, calls=call_origins, result_id="call_id")',
    ), label=label)
    _require_in_symbol(state, utils, "construct_chat_messages_with_tool_call", (
        "_convert_response_items(",
    ), label=label)
    _require_in_symbol(state, utils, "_construct_message_from_response_item", (
        '"".join(block.text for block in item.content)',
        '"".join(block.text for block in item.summary)',
        "_assistant_content(item.content)", "return deepcopy(item)",
    ), label=label)
    forbid_text(state, utils, "item.content[0]", label=label)
    forbid_text(state, utils, "item.summary[0]", label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/responses/test_responses_utils.py", {
            "test_replayed_blocks_preserve_every_byte_and_reasoning": None,
            "test_tool_history_correlation_is_shared_across_surfaces": None,
            "test_responses_history_is_validated_before_rendering": None,
            "test_a_responses_history_refusal_names_the_input_item_sent": None,
            "test_a_call_outside_an_assistant_message_is_named_by_the_call": None,
        }, label=label)


def _validate_responses_identity_before(state: State) -> None:
    label = "Responses stream identity precondition"
    serving = "vllm/entrypoints/openai/responses/serving.py"
    require_text(state, serving, "async def empty_async_generator():", label=label)
    require_text(state, serving, "def _create_response_logprobs(", label=label)
    require_text(state, "vllm/entrypoints/openai/responses/protocol.py",
                 "def is_include_output_logprobs(self) -> bool:", label=label)


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
    events = "vllm/entrypoints/openai/responses/streaming_events.py"
    _require_in_symbol(state, events, "emit_simple_content_done", (
        'status="incomplete" if incomplete else "completed"',
    ), label=label)
    # A message carries no log probabilities: the parser reports each item's
    # text, never its tokens, and one token can end the message and begin a
    # call. The request refuses both fields that ask for them, and nothing
    # on either transport builds them.
    protocol = "vllm/entrypoints/openai/responses/protocol.py"
    _require_in_symbol(state, protocol, "ResponsesRequest.refuse_log_probabilities", (
        'if self.include and "message.output_text.logprobs" in self.include:',
        'parameter="include"', "if self.top_logprobs:", 'parameter="top_logprobs"',
        "/v1/chat/completions",
    ), label=label)
    for path, needles in (
        (protocol, ("is_include_output_logprobs", "logprobs=self.top_logprobs")),
        (serving, ("is_include_output_logprobs", "_create_response_logprobs",
                   "_create_stream_response_logprobs", "_topk_logprobs")),
        (events, ("accumulated_logprobs",)),
        ("vllm/tool_parsers/poolside_v1_tool_parser.py", ("is_include_output_logprobs",)),
    ):
        for needle in needles:
            forbid_text(state, path, needle, label=label)
    _require_in_symbol(state, serving,
        "OpenAIServingResponses._process_simple_streaming_events",
        ("processor.emit_delta(dm)",), label=label)
    require_python_symbols(state, events, {
        "emit_simple_content_delta": ("state", "delta"),
        "SimpleStreamingEventProcessor.emit_delta": ("self", "delta_message"),
    }, label=label)
    require_python_symbols(state, "vllm/entrypoints/openai/responses/utils.py", {
        "build_response_output_items": (
            "reasoning", "content", "tool_calls", "tools", "incomplete"),
    }, label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/responses/test_serving_responses.py", {
            "test_terminal_response_uses_streamed_items_and_ids": None,
            "test_text_output_keeps_its_status_on_both_transports": None,
            "test_log_probabilities_are_refused_naming_the_field": None,
            "test_a_request_asking_for_no_log_probabilities_is_served": None,
        }, label=label)
    require_python_symbols(state, "tests/entrypoints/openai/responses/test_basic.py",
        {"test_logprobs_are_refused": None}, label=label)


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
        "return SamplingParams(**values)",
    ), label=label)
    # A rendered request names no line of work; the generation request adds
    # its own to the one resolved engine request.
    _require_in_symbol(state, protocol, "TokenGenerationRequest.to_sampling_params", (
        "super().to_sampling_params(max_tokens, default_sampling_params)",
        '"kv_scope": self.kv_scope',
    ), label=label)
    source = _symbol_source(state, serving, "ServingTokens.serve_tokens", label=label)
    _require_ordered(source, (
        "max_tokens = get_max_tokens(",
        "sampling_params = request.to_sampling_params(",
        "sampling_params.n > max_num_seqs",
        "msgspec.msgpack.encode(sampling_params)",
        "result_generator = ",
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
        require_text(state, serving, "sampling_params = request.to_sampling_params(",
                     label=label)
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
            "test_parallel_sampling_is_refused_before_dispatch": None,
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


def _validate_qwen_grammar_before(state: State) -> None:
    forbid_text(state, "vllm/tool_parsers/structural_tag_registry.py",
                "get_qwen_3_coder_structural_tag",
                label="Qwen-owned tool grammar precondition")


def _validate_qwen_grammar_after(state: State) -> None:
    label = "Qwen-owned tool grammar result"
    registry = "vllm/tool_parsers/structural_tag_registry.py"
    # A call limit is held by the grammar the request arms or refused, by one
    # refusal and nowhere else: an XGrammar builtin format takes none; under
    # "auto" a format arming its grammar only for a strict tool holds none
    # without one; and a tool parser whose format arms no grammar -- it names
    # none, or VLLM_ENFORCE_STRICT_TOOL_CALLING is off -- holds none unless
    # its own grammar takes the limit, as Mistral's does where it is armed.
    _require_in_symbol(state, registry, "require_call_limit_held", (
        "parallel_tool_calls false is held only by the tool-call grammar",
        "_ALLOW_PARALLEL_CALLS",
        'parameter="parallel_tool_calls",',
    ), label=label)
    require_text(state, registry, 'parameter="parallel_tool_calls"', label=label)
    forbid_text(state, registry, "parallel_tool_calls false cannot be enforced",
                label=label)
    _require_in_symbol(state, registry, "get_model_structural_tag", (
        "if model in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:",
        'next_action="Mark the tools strict: true"',
    ), label=label)
    _require_in_symbol(state, registry, "_unarmed_cause", (
        "has no tool-call grammar",
        "VLLM_ENFORCE_STRICT_TOOL_CALLING off",
    ), label=label)
    _require_in_symbol(state, registry, "require_unarmed_call_limit_held", (
        "_unarmed_cause(model, parser_name)",
        "if model in _VLLM_STRUCTURAL_TAG_REGISTRY:",
        "named_choice_held=named_choice_held",
    ), label=label)
    # Naming the one function is offered only where a named choice is held.
    _require_in_symbol(state, registry, "require_call_limit_held", (
        "_ALLOW_PARALLEL_CALLS + (_NAME_THE_FUNCTION if named_choice_held else \"\")",
    ), label=label)
    # A forced choice -- required, a named function, allowed_tools -- is held
    # by a grammar the parse reads back, or refused by one refusal naming
    # tool_choice: every format holds one once armed, so the switch is
    # offered wherever a parser names a format.
    _require_in_symbol(state, registry, "require_choice_held", (
        'if tool_choice in ("auto", "none"):',
        "is held only by a ",
        'parameter="tool_choice",',
    ), label=label)
    _require_in_symbol(state, registry, "require_unarmed_choice_held", (
        "_unarmed_cause(model, parser_name)",
        "Serve with VLLM_ENFORCE_STRICT_TOOL_CALLING on, which arms ",
    ), label=label)
    tool_parser = "vllm/tool_parsers/abstract_tool_parser.py"
    _require_ordered(_symbol_source(state, tool_parser, "ToolParser.get_structural_tag",
                                    label=label), (
        "forced_choice_held = self.forced_choice_held_without_grammar(request)",
        "if isinstance(request.tool_choice, ToolChoiceAllowed) or not (",
        "require_unarmed_choice_held(",
        "if not self.own_grammar_holds_call_limit(request):",
        "require_unarmed_call_limit_held(",
        "named_choice_held=forced_choice_held,",
    ), label=label, location=f"{tool_parser}:ToolParser.get_structural_tag")
    _require_in_symbol(state, tool_parser,
                       "ToolParser.forced_choice_held_without_grammar", (
        "return self.supports_required_and_named",
    ), label=label)
    _require_in_symbol(state, tool_parser,
                       "ToolParser.own_grammar_holds_call_limit", (
        "return False",
    ), label=label)
    # Mistral's declaration is the predicate its adjust_request branches on,
    # so what it claims and what it arms cannot disagree.
    _require_in_symbol(state, "vllm/parser/mistral.py", "MistralParser.adjust_request", (
        "if not self.arms_grammar(request):",
    ), label=label)
    _require_in_symbol(state, "vllm/tool_parsers/mistral_tool_parser.py",
                       "MistralToolParser.own_grammar_holds_call_limit", (
        "return self._parser_engine.arms_grammar(request)",
    ), label=label)
    _require_in_symbol(state, "vllm/tool_parsers/mistral_tool_parser.py",
                       "MistralToolParser.forced_choice_held_without_grammar", (
        "return self._parser_engine.arms_grammar(request) or (",
        "and self._parser_engine._is_pre_v11",
    ), label=label)
    # With its format unarmed, Kimi K2 holds a forced choice by the call's
    # JSON schema, as every other format parser does there.
    _require_in_symbol(state, "vllm/tool_parsers/kimi_k2_tool_parser.py",
                       "KimiK2ToolParser.adjust_request", (
        "request = ToolParser.adjust_request(self, request)",
    ), label=label)
    require_python_symbols(state, "tests/tool_parsers/test_structural_tag_registry.py", {
        "test_a_builtin_format_refuses_a_call_limit_it_cannot_hold": None,
        "test_a_forced_choice_is_one_call_in_a_builtin_format": None,
        "test_auto_without_a_strict_tool_refuses_a_call_limit": None,
        "test_a_parser_without_a_tool_grammar_refuses_a_call_limit": None,
        "test_strict_tool_calling_off_refuses_a_call_limit": None,
        "test_mistral_grammar_holds_its_call_limit": None,
        "test_a_parser_that_holds_no_forced_choice_refuses_it": ("choice", "sample_tools"),
        "test_a_parser_that_reads_the_call_schema_holds_a_forced_choice": None,
        "test_strict_tool_calling_off_offers_the_switch_for_a_forced_choice": None,
        "test_naming_the_function_is_offered_only_where_a_named_choice_is_held": None,
        "test_mistral_holds_a_forced_choice_where_its_grammar_or_legacy_array_does": None,
        "test_kimi_k2_holds_a_forced_choice_by_the_call_schema": None,
    }, label=label)
    require_python_symbols(state, registry, {
        "get_qwen_3_coder_structural_tag": (
            "tools", "builtin_tools", "tool_choice", "reasoning",
            "parallel_tool_calls",
        ),
        "_qwen_raw_value": (),
        "_qwen_value": None,
        "_qwen_arguments": ("parameters", "tool"),
        "_qwen_resolve": ("prop", "root", "seen"),
        "_qwen_definitions": ("root",),
        "_qwen_embed": ("schema", "root"),
    }, label=label)
    # A property is embedded in the tag on its own, so the root definition
    # tables must travel with it or its references dangle and the tool stops
    # building at all. The parameters document may itself be a wrapper, whose
    # unresolved ``properties`` would leave every argument unconstrained.
    # Both JSON productions -- an unfollowable local reference and every
    # other non-string type -- carry the root definition tables.
    require_text(state, registry, "_qwen_embed(resolved, root)", count=2,
                 label=label)
    for text in (
        "resolved = _qwen_resolve(parameters, root)",
        'properties = resolved.get("properties")',
        'required = set(resolved.get("required") or [])',
        # An unresolvable local reference is refused by the JSON channel,
        # never answered with the raw any-text channel.
        'reference = resolved.get("$ref")',
    ):
        require_text(state, registry, text, label=label)
    require_text(state, registry, 'reference.startswith("#/")', count=2,
                 label=label)
    require_text(state, registry, '@register_vllm_structural_tag("qwen_3_coder")',
                 label=label)
    require_text(state, registry, '_QWEN_PARAM_OPEN = "<parameter="', label=label)
    require_text(state, registry, '_QWEN_PARAM_CLOSE = "</parameter>"', label=label)
    # The whole point of owning the tag: an unconstrained value may carry
    # neither its own closer nor the next parameter's opener, so a value can
    # never absorb the opener and publish one call as another.
    _require_in_symbol(state, registry, "_qwen_raw_value", (
        "excludes=[_QWEN_PARAM_OPEN, _QWEN_PARAM_CLOSE]",
    ), label=label)
    # There is no second string channel to fall into. XGrammar emits a pattern-
    # or length-constrained string as a regex, and the opener exclusion cannot
    # live inside one: its regex engine has no lookahead, so forbidding a fixed
    # substring is only an unrolled DFA, which cannot then carry a length bound.
    # The declaration is refused by name, naming the tool, the property and the
    # fix, rather than served on a channel that silently drops the exclusion.
    forbid_text(state, registry, "_QWEN_STRING_CONSTRAINTS", label=label)
    forbid_text(state, registry, 'RegexFormat(pattern="[^]"', label=label)
    require_text(state, registry,
                 "for key in _QWEN_REFUSED_STRING_KEYS if key in resolved",
                 label=label)
    _require_in_symbol(state, registry, "_qwen_value", (
        "raise VLLMValidationError(",
        "Declare the parameter without it and validate it in the tool.",
        'parameter="tools",',
    ), label=label)
    # A schema this transport cannot express, and a forced choice naming no
    # declared function, are both the caller's to fix, so both answer as typed
    # request refusals carrying the field to edit. The module's remaining bare
    # ValueError is the one fault the caller cannot fix -- a deployment whose
    # tool parser names a structural-tag model this module does not build --
    # and it stays a server error on purpose.
    require_text(state, registry, "from vllm.exceptions import VLLMValidationError",
                 label=label)
    _require_in_symbol(state, registry, "get_qwen_3_coder_structural_tag", (
        "raise VLLMValidationError(",
        'parameter="tool_choice",',
    ), label=label)
    _require_in_symbol(state, registry, "get_model_structural_tag", (
        "raise ValueError(f\"Unknown format type: {model}",
    ), label=label)
    # The call count is decided while building, not by mutating a returned tag.
    _require_in_symbol(state, registry, "get_qwen_3_coder_structural_tag", (
        "single_call = parallel_tool_calls is False",
        "stop_after_first=single_call",
    ), label=label)
    forbid_text(state, registry, "suffix.stop_after_first = True", label=label)
    _require_in_symbol(state, "vllm/parser/qwen3.py", "_qwen3_arg_converter", (
        "_unframe_parameter_value(value, complete=True)",
    ), label=label)
    # The dispatch calls every registered builder with one argument list, so
    # every builder takes it -- the request's call limit included -- and holds
    # the limit wherever its format can express a single call. The formats a
    # refusal offers are read from the registry, the only list of them.
    builder_params = (
        "tools", "builtin_tools", "tool_choice", "reasoning", "parallel_tool_calls",
    )
    require_python_symbols(state, registry, {
        "get_hermes_structural_tag": builder_params,
        "get_minimax_structural_tag": builder_params,
        "get_kimi_k3_structural_tag": builder_params,
    }, label=label)
    require_python_symbols(state, "vllm/parser/harmony.py", {
        "get_harmony_structural_tag": builder_params,
    }, label=label)
    for builder in (
        "get_hermes_structural_tag", "get_minimax_structural_tag",
        "get_kimi_k3_structural_tag",
    ):
        _require_in_symbol(state, registry, builder, (
            "single_call = parallel_tool_calls is False",
        ), label=label)
    for retired in ("VLLM_BUILTIN_STRUCTURAL_TAG_MODELS",
                    "SUPPORTED_STRUCTURAL_TAG_MODELS"):
        forbid_text(state, registry, retired, label=label)
    _require_in_symbol(state, registry, "get_model_structural_tag", (
        "XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS.union(_VLLM_STRUCTURAL_TAG_REGISTRY)",
    ), label=label)
    require_python_symbols(state, "tests/tool_parsers/test_structural_tag_registry.py", {
        "test_every_registered_builder_takes_the_dispatch_arguments": None,
        "test_qwen3_value_cannot_absorb_the_next_parameter_opener": None,
        "test_qwen3_auto_with_parallel_off_ends_the_turn_after_one_call": None,
        "test_qwen3_auto_tool_choice_is_constrained_without_strict": None,
        "test_qwen3_nested_reference_keeps_the_root_definition_tables": None,
        "test_qwen3_root_composition_still_binds_every_property": None,
        "test_qwen3_unresolvable_local_reference_is_refused": None,
        "test_qwen3_external_reference_stays_unconstrained": None,
        # The refusal offers what the registry builds, read from it.
        "test_supported_structural_tag_models_include_vllm_builtins": None,
    }, label=label)


def _validate_canonical_framing_before(state: State) -> None:
    forbid_text(state, "vllm/parser/qwen3.py", "_unframe_parameter_value",
                label="Canonical parameter framing precondition")


def _validate_unique_parameters_before(state: State) -> None:
    forbid_text(
        state, "vllm/parser/qwen3.py", "Qwen XML repeats parameter",
        label="Qwen unique parameter precondition",
    )


def _validate_unique_parameters_after(state: State) -> None:
    label = "Qwen unique tool parameters"
    qwen = "vllm/parser/qwen3.py"
    # A name already in the arguments is refused before either value can
    # replace the other, on complete and unfinished output alike; how the
    # refusal is answered is qwen-repeated-parameter-refusal's.
    _require_in_symbol(state, qwen, "_qwen3_arg_converter", (
        "if name in params:",
        "_unframe_parameter_value(value, complete=True)",
        "_unframe_parameter_value(value, complete=False)",
    ), label=label)
    _require_in_symbol(state, qwen, "Qwen3Parser._convert_tool_arguments", (
        "Qwen XML argument decoding failed: {exc}",
        "raise RuntimeError(",
    ), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen3.py", {
        "TestArgConverter.test_repeated_parameter_cannot_overwrite_model_output": None,
    }, label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_repeated_parameter_refuses_the_call_instead_of_replacing_text": None,
    }, label=label)


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
                 "vllm/parser/inkling.py", "vllm/parser/kimi_k2.py",
                 "vllm/parser/mistral.py"):
        forbid_text(state, path, "strip_content_whitespace_with_tools", label=label)
        forbid_text(state, path, "_strip_content_ws_with_tools", label=label)
        forbid_text(state, path, "drop_whitespace_only_content_before_tools", label=label)
        forbid_text(state, path, "_strip_content_whitespace", label=label)
        forbid_text(state, path, "_content_has_nonws", label=label)
    delegating = "vllm/parser/abstract_parser.py"
    forbid_text(state, delegating, 'if content and content.strip() == "":', label=label)
    _require_in_symbol(state, delegating, "DelegatingParser._extract_tool_calls", (
        "content = tool_call_info.content", "return [], content",
    ), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_parameter_string_bytes_survive_every_transport_cut": None,
        "test_partial_string_diagnostic_preserves_raw_value_bytes": None,
        "test_content_around_tool_calls_is_verbatim": None,
        "test_whitespace_without_a_call_survives_tool_choice": None,
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
    # The terminal reaches every override of the extraction, as the ids do.
    _require_in_symbol(state, "vllm/parser/kimi_k3.py", "KimiK3Parser._extract_tool_calls", (
        "output_terminal: tuple[str | None, str | int | None] | None = None,",
        "content_token_ids,\n                output_terminal,",
    ), label=label)
    _require_in_symbol(state, "vllm/parser/mistral.py",
        "MistralParser.extract_tool_calls_from_content", (
            "output_terminal: tuple[str | None, str | int | None] | None = None,",
            "content, request, content_token_ids, output_terminal",
        ), label=label)


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
        "if schema and ", "return _decode_xml_parameters(params, schema)",
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
    # The template's deliberate guard is a typed request refusal; every other
    # template failure keeps its server cause.
    source = _require_in_symbol(state, hf, "safe_apply_chat_template", (
        "raise_exception",
    ), label=label)
    _require("except Exception" not in source and "raise ValueError(str(e))" not in source,
             label + ": template bugs must retain their original server cause")
    require_text(state, hf, "raise VLLMValidationError(message, parameter=", label=label)
    _require_in_symbol(state, "vllm/multimodal/media/image.py", "ImageMediaIO.load_bytes", (
        "raise VLLMValidationError(", "Image.open(BytesIO(data))", "image.load()",
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
    require_text(state, "tests/multimodal/media/test_connector.py",
                 'pytest.raises(VLLMValidationError, match="Failed to load image")',
                 count=2, label=label)


def _validate_admission_before(state: State) -> None:
    label = "generation admission precondition"
    engine = "vllm/v1/engine/async_llm.py"
    forbid_text(state, engine, "async def admit(", label=label)
    require_text(state, "vllm/entrypoints/openai/chat_completion/serving.py",
                 "generator = self.engine_client.generate(", label=label)
    require_text(state, "vllm/entrypoints/openai/responses/serving.py",
                 "TypeAdapter(StreamingResponsesResponse).validate_json(error_json)",
                 label=label)


def _validate_admission_after(state: State) -> None:
    label = "generation admission result"
    require_text(state, "vllm/engine/protocol.py", "    async def admit(\n", label=label)
    engine = "vllm/v1/engine/async_llm.py"
    _require_in_symbol(state, engine, "AsyncLLM.admit", (
        "stream = self._admitted_stream(",
        "admitted = await anext(stream)",
        "assert admitted is _ADMITTED",
    ), label=label)
    _require_in_symbol(state, engine, "AsyncLLM.generate", (
        "stream = await self.admit(", "await stream.aclose()",
    ), label=label)
    _require_ordered(
        _symbol_source(state, engine, "AsyncLLM._admitted_stream", label=label),
        ("q = await self.add_request(", "yield _ADMITTED", "out = q.get_nowait()"),
        label=label, location=engine,
    )
    # Admission is all or nothing: a failed engine submission is aborted.
    _require_in_symbol(state, engine, "AsyncLLM._add_request", (
        "except BaseException:", "await self.abort(request.request_id, internal=True)",
    ), label=label)
    # Every served generation surface admits before it answers.
    for path, text in (
        ("vllm/entrypoints/openai/chat_completion/serving.py",
         "generator = await self.engine_client.admit("),
        ("vllm/entrypoints/openai/completion/serving.py",
         "generator = await self.engine_client.admit("),
        ("vllm/entrypoints/openai/chat_completion/batch_serving.py",
         "stream = await self.engine_client.admit("),
        ("vllm/entrypoints/openai/responses/serving.py",
         "return await self.engine_client.admit("),
        ("vllm/entrypoints/scale_out/token_in_token_out/serving.py",
         "result_generator = await self.engine_client.admit("),
    ):
        require_text(state, path, text, label=label)
        forbid_text(state, path, "self.engine_client.generate(", label=label)
    require_text(state, "vllm/entrypoints/openai/chat_completion/batch_serving.py",
                 "admitted.push_async_callback(stream.aclose)", label=label)
    responses = "vllm/entrypoints/openai/responses/serving.py"
    forbid_text(state, responses, "TypeAdapter(StreamingResponsesResponse)", label=label)
    _require_in_symbol(state, responses, "OpenAIServingResponses.responses_stream_generator", (
        "except GenerationError as e:", "except Exception as e:",
        "self._stream_error_event(e)",
    ), label=label)
    require_text(state, "vllm/entrypoints/openai/responses/protocol.py",
                 "    | ResponseErrorEvent\n", label=label)
    require_python_symbols(state, "tests/entrypoints/test_generation_admission.py", {
        "test_an_admission_refusal_is_a_400_before_any_status_line": None,
        "test_a_responses_failure_after_the_status_line_ends_with_its_error_event": None,
    }, label=label)


def _validate_native_fp4_before(state: State) -> None:
    label = "native NVFP4 kernel precondition"
    kernels = "vllm/model_executor/kernels/linear/__init__.py"
    require_text(state, kernels, "NVFP4 linear falling back to the slow and unoptimized",
                 label=label)
    forbid_text(state, kernels, "does not compute in FP4", label=label)


def _validate_native_fp4_after(state: State) -> None:
    label = "native NVFP4 kernel required"
    kernels = "vllm/model_executor/kernels/linear/__init__.py"
    _require_in_symbol(state, kernels, "init_nvfp4_linear_kernel", (
        "substitutes = () if use_a16 else (*a16_kernels, EmulationNvFp4LinearKernel)",
        "requested = _LINEAR_BACKEND_KERNEL_MAP.get(linear_backend, set())",
        "if kernel_cls in substitutes and kernel_cls not in requested:",
        "capability.as_version_str()", "does not compute in FP4",
    ), label=label)
    forbid_text(state, kernels, "NVFP4 linear falling back to the slow and unoptimized",
                label=label)
    require_python_symbols(state,
        "tests/model_executor/kernels/test_nvfp4_native_selection.py", {
            "test_a_gpu_without_native_fp4_refuses_instead_of_marlin": None,
            "test_emulation_is_not_substituted_either": None,
            "test_a_native_kernel_is_selected_ahead_of_substitutes": None,
            "test_a_named_substitute_is_served": None,
            "test_weight_only_checkpoints_keep_marlin": None,
        }, label=label)


def _validate_template_authorship_before(state: State) -> None:
    label = "template-authored control tokens precondition"
    hf = "vllm/renderers/hf.py"
    require_text(state, hf, "plain = tokenizer.apply_chat_template(", label=label)
    forbid_text(state, hf, "TemplateTokenEncoder", label=label)
    require_text(state, "vllm/renderers/online_renderer.py",
                 "def _reused_prompt_token_ids(", label=label)


def _validate_template_authorship_after(state: State) -> None:
    label = "template-authored control tokens"
    hf = "vllm/renderers/hf.py"
    authorship = "vllm/renderers/template_authorship.py"
    # The prompt of a chat is the template's text encoded with authorship;
    # nothing renders it to a string for a later whole-string encoding.
    forbid_text(state, hf, "tokenizer.apply_chat_template(", label=label)
    forbid_text(state, hf, "parse_dec_only_prompt", label=label)
    _require_in_symbol(state, hf, "safe_apply_chat_template", (
        "render_chat_template(", "special_tokens=tokenizer.special_tokens_map",
        "raise_exception=_",
    ), label=label)
    _require_in_symbol(state, hf, "HfRenderer.__init__", (
        "isinstance(self.tokenizer, TokenizersBackend)",
        "TemplateTokenEncoder(self.tokenizer.backend_tokenizer.to_str())",
    ), label=label)
    _require_in_symbol(state, hf, "HfRenderer._render_conversation", (
        "if self._template_encoder is None:",
        "self._template_encoder.encode(text)",
        "TokensPrompt(prompt_token_ids=encoded.ids, prompt=str(text))",
        'content_format == "string" and mm_data',
    ), label=label)
    # Every chat surface renders its prompt through the template; no request
    # field supplies the prompt's ids in its place.
    online = "vllm/renderers/online_renderer.py"
    forbid_text(state, online, "kv_transfer_params", label=label)
    _require_in_symbol(state, online, "OnlineRenderer.preprocess_chat", (
        "renderer.render_chat_async(",
    ), label=label)
    _require(
        "tokens_input(" not in _symbol_source(
            state, online, "OnlineRenderer.preprocess_chat", label=label),
        f"{label}: a chat prompt is built from ids the request supplied",
    )
    require_python_symbols(
        state, "tests/entrypoints/openai/chat_completion/test_serving_chat.py", {
            "test_make_request_with_harmony_renders_the_messages": ("monkeypatch",),
        }, label=label)
    _require_in_symbol(state, authorship, "TemplateTokenEncoder.__init__", (
        'spec["added_tokens"] = []',
    ), label=label)
    _require_in_symbol(state, authorship, "TemplateTokenEncoder.encode", (
        "self._ordinary.encode_batch(",
    ), label=label)
    _require_in_symbol(state, authorship, "TemplateTokenEncoder._template_added_tokens", (
        "label & _AUTHOR == REQUEST", "text.authors(start, end) != {TEMPLATE}",
        "raise ControlTokenAuthorshipError(",
    ), label=label)
    _require_in_symbol(state, authorship, "render_chat_template", (
        "written_by_request(list(conversation))", "written_by_template(v)",
    ), label=label)
    require_python_symbols(state, authorship, {
        "AuthoredText": None,
        "AuthoringEnvironment": None,
        "GenerationTracker": None,
        "_AuthoringCodeGenerator.visit_Const": None,
        "_AuthoringCodeGenerator._output_child_pre": None,
    }, label=label)
    _require_in_symbol(state, "vllm/entrypoints/chat_utils.py",
                       "_parse_chat_message_content_part", (
        "AuthoredText.written_by(", "PROMPT_EMBEDS_PLACEHOLDER_TOKEN, TEMPLATE",
    ), label=label)
    require_python_symbols(state, "tests/renderers/test_template_authorship.py", {
        "test_request_text_spelling_added_tokens_stays_text": None,
        "test_prompt_without_spellings_encodes_as_the_whole_string": None,
        "test_authorship_follows_the_operations_templates_use": None,
        "test_text_without_an_author_cannot_spell_an_added_token": None,
    }, label=label)


def _validate_single_flight_before(state: State) -> None:
    forbid_text(state, "vllm/v1/engine/output_processor.py", "kv_scope",
                label="kv_scope single-flight precondition")


def _validate_single_flight_after(state: State) -> None:
    label = "kv_scope single-flight result"
    processor = "vllm/v1/engine/output_processor.py"
    for text in (
        "self.kv_scope_requests: dict[str, RequestState] = {}",
        "already has a request in flight",
        "a continuation must not overlap its predecessor",
        "self.kv_scope_requests[scope] = req_state",
    ):
        require_text(state, processor, text, label=label)
    require_text(state, processor, "self._release_kv_scope(req_state)", count=2,
                 label=label)
    # One API server process admits every generation request; several would
    # each see only their own share of the requests in flight.
    _require_in_symbol(state, "vllm/v1/engine/async_llm.py", "AsyncLLM.__init__", (
        "if client_count != 1:", "Serve with --api-server-count 1.",
    ), label=label)
    require_python_symbols(state, "tests/v1/engine/test_kv_scope_single_flight.py", {
        "test_an_overlapping_request_under_one_id_is_refused_by_name": None,
        "test_admission_refuses_the_overlap_before_the_engine_sees_it": None,
        "test_only_one_api_server_process_may_admit_generation": None,
    }, label=label)
    require_python_symbols(state, "tests/entrypoints/test_generation_admission.py", {
        "test_an_overlapping_request_under_one_scope_is_a_400_naming_it": None,
    }, label=label)


def _validate_grammar_read_arguments_before(state: State) -> None:
    label = "Qwen arguments read by the grammar precondition"
    forbid_text(state, "vllm/tool_parsers/structural_tag_registry.py",
                "def read_qwen_arguments(", label=label)
    require_text(state, "vllm/parser/qwen3.py",
                 '(ParserState.TOOL_PARAM_VALUE, "PARAM_END")', label=label)


def _validate_grammar_read_arguments_after(state: State) -> None:
    label = "Qwen arguments read by the grammar"
    registry = "vllm/tool_parsers/structural_tag_registry.py"
    qwen = "vllm/parser/qwen3.py"
    engine = "vllm/parser/engine/streaming_parser_engine.py"
    config = "vllm/parser/engine/parser_engine_config.py"
    require_python_symbols(state, registry, {
        "QwenArgumentReading": None,
        "read_qwen_arguments": ("text", "parameters"),
        "_qwen_productions": ("parameters",),
        "_qwen_production": ("prop", "root"),
        "_qwen_value": ("value", "tool", "parameter"),
        "_qwen_value_ends": ("value", "text", "start"),
        "_qwen_json_closer": ("text", "position"),
    }, label=label)
    # One definition: the grammar is rendered from the productions the reader
    # reads by, so the two cannot disagree about where a value ends.
    for symbol in ("_qwen_arguments", "read_qwen_arguments"):
        _require_in_symbol(state, registry, symbol, (
            "arguments = _qwen_productions(parameters)",
        ), label=label)
    # A JSON value ends at the first closer outside its strings, and is a JSON
    # value up to it; a raw value ends at its first closer with no opener
    # before it.
    _require_in_symbol(state, registry, "_qwen_json_closer", (
        "elif text.startswith(_QWEN_PARAM_CLOSE, index):",
    ), label=label)
    _require_in_symbol(state, registry, "_qwen_value_ends", (
        "_QWEN_JSON.raw_decode(text, begin)",
        "elif opener < 0 or opener > closer:",
    ), label=label)
    # The parser reads by that definition, and keeps the repeat refusal.
    _require_in_symbol(state, qwen, "_qwen3_arg_converter", (
        "reading = read_qwen_arguments(raw_args, schema or {})",
        "if schema and unfinished is None:",
        "if name in params:",
    ), label=label)
    _require_in_symbol(state, qwen, "_qwen3_arguments_reading", (
        "return ArgumentsReading(reading.complete, reading.inside_parameter)",
    ), label=label)
    _require_in_symbol(state, qwen, "qwen3_config", (
        "arguments_reading=_qwen3_arguments_reading",
        "(EventType.TOOL_CALL_END, EventType.TOOL_CALL_CLOSED)",
    ), label=label)
    _require_in_symbol(state, qwen, "Qwen3Parser._tool_arguments_reading", (
        "find_tool_schema(self._tools, func_name)",
    ), label=label)
    # The engine has no second notion of being inside a value: the format's
    # reading of the arguments decides whether the closing sequence ends the
    # call and whether a held function closer was text inside a parameter.
    for path in (qwen, engine, config):
        forbid_text(state, path, "TOOL_PARAM_VALUE", label=label)
    require_python_symbols(state, config, {"ArgumentsReading": None}, label=label)
    require_text(state, config,
                 "arguments_reading: Callable[[str], ArgumentsReading] | None = None",
                 label=label)
    _require_in_symbol(state, engine, "StreamingParserEngine._on_terminal", (
        "self._call_reading()", "self._resume_arguments(",
        "reading.complete or not reading.inside_parameter",
    ), label=label)
    _require_in_symbol(state, engine, "StreamingParserEngine._emit_for_state", (
        "self._call_reading().inside_parameter",
    ), label=label)
    _require_in_symbol(state, "vllm/parser/engine/parser_engine.py",
        "ParserEngine.__init__", (
            "arguments_reading=self._tool_arguments_reading",
        ), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_json_value_strings_carry_parameter_markup": None,
        "test_open_declared_slot_reads_its_declared_production": None,
        "test_whole_body_closes_at_the_wrapper_while_a_json_reading_is_open": None,
        "test_truncated_json_value_is_the_unfinished_parameter": None,
        "test_closing_sequence_inside_a_raw_value_does_not_end_the_call": None,
    }, label=label)
    require_python_symbols(state, "tests/tool_parsers/test_structural_tag_registry.py", {
        "test_qwen3_arguments_are_read_by_the_grammar_productions": None,
        "test_qwen3_reader_has_no_reading_of_text_the_grammar_never_writes": None,
    }, label=label)


def _validate_plan_bound_before(state: State) -> None:
    require_text(state, "vllm/v1/worker/gpu_worker.py",
                 "- int(self.total_consumed)",
                 label="Startup plan admission bound precondition")


def _validate_plan_bound_after(state: State) -> None:
    label = "Startup plan persists the admission bound"
    worker = "vllm/v1/worker/gpu_worker.py"
    # One quantity: determine_available_memory derives the bound once and
    # admits against it; the plan persists that same value, before the
    # reservation a plan-shortcut boot applies to it again.
    _require_in_symbol(state, worker, "Worker.determine_available_memory", (
        "self.kv_physical_bound = int(",
        "self.kv_physical_bound,",
    ), label=label)
    _require_in_symbol(state, worker, "Worker.compile_or_warm_up_model", (
        "maybe_save_startup_plan(self, self.kv_physical_bound)",
    ), label=label)
    forbid_text(state, worker, "- int(self.total_consumed)", label=label)
    # A plan is keyed on the code that derived it: builds that patch one
    # upstream commit share its version string.
    _require_in_symbol(state, "vllm/v1/worker/startup_plan.py",
                       "compute_plan_fingerprint", (
        '"source": installed_source_digest(),',
    ), label=label)
    _require_in_symbol(state, "vllm/v1/worker/startup_plan.py",
                       "installed_source_digest", (
        'root.rglob("*.py")',
    ), label=label)
    require_python_symbols(state, "tests/v1/worker/test_gpu_worker.py", {
        "test_plan_shortcut_boot_admits_against_the_profiled_bound": None,
        "test_installed_source_digest_covers_every_source_file": None,
    }, label=label)


def _validate_refusal_parameter_before(state: State) -> None:
    require_text(state, "vllm/renderers/hf.py",
                 "def _raise_template_validation_error(",
                 label="Template refusal parameter precondition")


def _validate_refusal_parameter_after(state: State) -> None:
    label = "Template refusals name the request parameter"
    hf = "vllm/renderers/hf.py"
    params = "vllm/renderers/params.py"
    # The template names the variable it refuses; the refusal names the
    # request parameter that supplied it, and none it cannot attribute.
    require_python_symbols(state, hf, {"_template_refusal": ("parameter_names",)},
                           label=label)
    _require_in_symbol(state, hf, "_template_refusal", (
        "def raise_exception(message: str, variable: str | None = None) -> NoReturn:",
        "raise VLLMValidationError(message, parameter=parameter_names.get(variable))",
    ), label=label)
    _require_in_symbol(state, hf, "safe_apply_chat_template", (
        "raise_exception=_template_refusal(parameter_names or {})",
    ), label=label)
    require_text(state, hf, "parameter_names=params.parameter_names", label=label)
    forbid_text(state, hf, 'parameter="messages"', label=label)
    forbid_text(state, hf, "_raise_template_validation_error", label=label)
    # One record of where each template variable came from, built in the same
    # merge that sets its value, on every request surface that renders.
    require_python_symbols(state, params, {
        "request_chat_template_kwargs": ("chat_template_kwargs", "fields"),
    }, label=label)
    require_text(state, params,
                 "parameter_names: dict[str, str] = field(default_factory=dict)",
                 label=label)
    _require_in_symbol(state, params, "ChatParams.with_defaults", (
        "parameter_names=self.parameter_names",
    ), label=label)
    for path, symbol, conversation, effort in (
        ("vllm/entrypoints/openai/chat_completion/protocol.py",
         "ChatCompletionRequest.build_chat_params", '"messages"', '"reasoning_effort")'),
        ("vllm/entrypoints/openai/responses/protocol.py",
         "ResponsesRequest.build_chat_params", '"input"', '"reasoning.effort")'),
    ):
        _require_in_symbol(state, path, symbol, (
            "request_chat_template_kwargs(",
            f"parameter_names.update(messages={conversation}, tools=\"tools\")",
            effort, "parameter_names=parameter_names",
        ), label=label)
    _require_in_symbol(state, "vllm/entrypoints/serve/tokenize/protocol.py",
        "TokenizeChatRequest.build_chat_params", (
            "request_chat_template_kwargs(",
            'parameter_names.update(messages="messages", tools="tools")',
            "parameter_names=parameter_names",
        ), label=label)
    require_python_symbols(state, "vllm/renderers/template_authorship.py", {
        "_raise_exception": ("message", "variable"),
    }, label=label)
    require_python_symbols(state, "tests/renderers/test_hf.py", {
        "test_template_refusal_names_the_request_parameter": None,
        "test_template_refusal_of_a_server_default_names_no_request_parameter": None,
    }, label=label)


def _validate_repeated_parameter_refusal_before(state: State) -> None:
    label = "Qwen repeated parameter refusal precondition"
    require_text(state, "vllm/parser/qwen3.py",
                 'raise ValueError(f"Qwen XML repeats parameter {name!r}")',
                 count=2, label=label)
    forbid_text(state, "vllm/entrypoints/openai/engine/protocol.py",
                "RepeatedToolParameterError", label=label)


def _validate_repeated_parameter_refusal_after(state: State) -> None:
    label = "Qwen repeated parameter refusal"
    protocol = "vllm/entrypoints/openai/engine/protocol.py"
    errors = "vllm/entrypoints/serve/exception_handling/error_response.py"
    qwen = "vllm/parser/qwen3.py"
    # The refusal is typed, names the call and the parameter, and states the
    # next action; the request was sound and the server did not fail, so it
    # is a 422 on the one error surface every refusal is answered on.
    require_python_symbols(state, protocol, {
        "RepeatedToolParameterError": None,
        "RepeatedToolParameterError.__init__": (
            "self", "tool", "repeated_parameter",
        ),
    }, label=label)
    # It is the unprocessable-entity refusal, whose request parameter is none.
    require_text(state, protocol,
                 "class RepeatedToolParameterError(VLLMUnprocessableEntityError):",
                 label=label)
    _require_in_symbol(state, protocol, "RepeatedToolParameterError.__init__", (
        "{repeated_parameter!r} more than once.",
        "the response again.",
        "self.repeated_parameter = repeated_parameter",
    ), label=label)
    _require_in_symbol(state, errors, "create_error_response", (
        "if isinstance(exc, RepeatedToolParameterError):",
        'err_type = "RepeatedToolParameterError"',
        "status_code = HTTPStatus.UNPROCESSABLE_ENTITY",
    ), label=label)
    # The converter signals a repeat with an exception the engine's
    # provisional-converter fallback (ValueError, TypeError) cannot catch,
    # and the parser answers it with the refusal; no call is published.
    require_text(state, qwen, "class _RepeatedParameter(Exception):", label=label)
    require_text(state, qwen, "raise _RepeatedParameter(name)", count=2, label=label)
    forbid_text(state, qwen, "Qwen XML repeats parameter", label=label)
    forbid_text(state, qwen, "inspect the generated call", label=label)
    _require_in_symbol(state, qwen, "Qwen3Parser._convert_tool_arguments", (
        "except _RepeatedParameter as repeat:",
        "raise RepeatedToolParameterError(func_name, repeat.name) from repeat",
    ), label=label)
    # A refusal inside a stream is answered as one, not logged as a failure.
    _require_in_symbol(state, "vllm/entrypoints/openai/chat_completion/serving.py",
        "OpenAIServingChat.chat_completion_stream_generator", (
            "except RepeatedToolParameterError as e:",
        ), label=label)
    _require_in_symbol(state, "vllm/entrypoints/openai/responses/serving.py",
        "OpenAIServingResponses.responses_stream_generator", (
            "except RepeatedToolParameterError as e:",
        ), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen_xml_fidelity.py", {
        "test_repeated_parameter_refuses_the_call_instead_of_replacing_text": None,
    }, label=label)
    require_python_symbols(
        state,
        "tests/entrypoints/openai/chat_completion/"
        "test_repeated_tool_parameter_refusal.py",
        {
            "test_the_refusal_is_a_typed_422_naming_the_call_and_parameter": None,
            "test_a_stream_ends_in_the_refusal_and_publishes_no_call": None,
            "test_a_whole_response_is_refused_before_any_body": None,
        },
        label=label,
    )


def _validate_generated_tokens_before(state: State) -> None:
    label = "Generated tokens survive parsing precondition"
    require_text(state, "vllm/parser/engine/streaming_parser_engine.py",
                 "def _build_drop_info(", label=label)
    require_text(state, "vllm/parser/qwen3.py",
                 '(ParserState.CONTENT, "THINK_END"): Transition(', label=label)


def _validate_generated_tokens_after(state: State) -> None:
    label = "Generated tokens survive parsing"
    engine = "vllm/parser/engine/streaming_parser_engine.py"
    scanner = "vllm/parser/engine/token_id_scanner.py"
    # No terminal deletes what the model generated: a token the format does
    # not act on is the text it decodes to, in every state.
    for path in (engine, scanner):
        for retired in ("DROP_TERMINAL", "__DROP__", "_build_drop_info", "_has_drops"):
            forbid_text(state, path, retired, label=label)
    for path in ("vllm/parser/engine/parser_engine_config.py", "vllm/parser/gemma4.py"):
        forbid_text(state, path, "preserve_tokens", label=label)
    # Reasoning ends once; a later closer is content, as a later opener is.
    forbid_text(state, "vllm/parser/qwen3.py", '(ParserState.CONTENT, "THINK_END")',
                label=label)
    # A grammar whose batch tool pass splits on the forwarded content ids must
    # forward every byte after the boundary; registering the format refuses
    # one that acts on a terminal in content outside its tool language, in
    # any configuration it builds, before any request.
    _require_in_symbol(state, "vllm/parser/engine/parser_engine.py",
                       "ParserEngine.check_format", (
        "if not cls.batch_tool_pass_uses_ids:",
        "for config in cls.engine_configs():",
        "if state is ParserState.CONTENT and terminal not in tool_terminals",
        "or keep the text-only batch tool pass.",
    ), label=label)
    _require_in_symbol(state, "vllm/parser/engine/adapters.py", "make_adapters",
                       ("parser_engine_cls.check_format()",), label=label)
    _require_in_symbol(state, "vllm/parser/qwen3.py", "Qwen3Parser.engine_configs",
                       ("(cls.engine_config(True), cls.engine_config(False))",),
                       label=label)
    # A format that builds its own configuration names it, so registration
    # checks what is served; no request repeats the check.
    _require_in_symbol(state, "vllm/parser/nemotron_v3.py", "NemotronV3Parser.engine_config",
                       ("return nemotron_v3_config(thinking=thinking)",), label=label)
    forbid_text(state, "vllm/parser/engine/parser_engine.py",
                "and terminal not in self._engine._tool_terminals", label=label)
    require_python_symbols(state, "tests/parser/engine/test_parser_engine.py", {
        "TestSpecialTokensAreText.test_special_token_by_id_stays_in_content": None,
        "TestSpecialTokensAreText.test_special_token_stays_in_tool_args": None,
        "TestSpecialTokensAreText.test_special_tokens_stay_with_skip_tool_parsing": None,
        "TestFormatRegistration."
        "test_forwarding_ids_while_acting_in_content_is_refused_at_registration": None,
        "TestFormatRegistration.test_forwarding_ids_requires_naming_every_configuration": None,
        "TestFormatRegistration.test_every_format_that_forwards_ids_registers": None,
    }, label=label)
    require_python_symbols(state, "tests/parser/engine/test_replay.py", {
        "TestSpecialTokenReplay.test_special_tokens_survive_as_text": None,
    }, label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen3_reasoning.py", {
        "TestNonStreaming.test_second_think_end_is_content": None,
        "TestStreaming.test_streaming_second_think_end_is_content": None,
    }, label=label)


_CHAT_SERVING = "vllm/entrypoints/openai/chat_completion/serving.py"


def _validate_include_reasoning_before(state: State) -> None:
    require_text(state, _CHAT_SERVING,
                 "if not request.include_reasoning:\n                reasoning_ended = True",
                 label="Response-only include_reasoning precondition")


def _validate_include_reasoning_after(state: State) -> None:
    label = "Response-only include_reasoning"
    source = _symbol_source(state, _CHAT_SERVING,
                            "OpenAIServingChat._create_chat_completion", label=label)
    start = source.find("session_id = self._get_session_id(")
    end = source.find("reasoning_ended=reasoning_ended,", start)
    _require(start >= 0 and end > start,
             f"{label}: the reasoning state is no longer decided before admission")
    decision = [line for line in source[start:end].splitlines()
                if not line.strip().startswith("#")]
    # What the response shows never decides when a grammar starts: the state
    # comes from the parser's reading of the prompt the model continues.
    _require(not any("include_reasoning" in line for line in decision),
             f"{label}: the reasoning state is derived from include_reasoning")
    _require_ordered("\n".join(decision), (
        "if request._grammar_from_parser:",
        "reasoning_ended = True",
        "reasoning_ended = parser.is_reasoning_end(prompt_token_ids or [])",
    ), label=label, location=f"{_CHAT_SERVING}:OpenAIServingChat._create_chat_completion")
    require_python_symbols(state,
        "tests/entrypoints/openai/chat_completion/test_serving_chat.py", {
            "test_include_reasoning_leaves_the_reasoning_state_to_the_prompt": None,
        }, label=label)


_CHAT_PROTOCOL = "vllm/entrypoints/openai/chat_completion/protocol.py"


def _validate_unspecified_tool_choice_before(state: State) -> None:
    require_text(state, _CHAT_PROTOCOL,
                 'if "tool_choice" not in data and data.get("tools"):',
                 label="Unspecified tool choice precondition")


def _validate_unspecified_tool_choice_after(state: State) -> None:
    label = "Unspecified tool choice"
    anthropic = "vllm/entrypoints/anthropic/serving.py"
    # Omitted and null are one value, decided once, so arming and parsing
    # read the same choice; the field holds no null after validation.
    _require_in_symbol(state, _CHAT_PROTOCOL, "ChatCompletionRequest.check_tool_usage", (
        'if data.get("tool_choice") is None:',
        'data["tool_choice"] = "auto" if data.get("tools") else "none"',
        'if data["tool_choice"] == "none":',
    ), label=label)
    forbid_text(state, _CHAT_PROTOCOL,
                'if "tool_choice" not in data and data.get("tools"):', label=label)
    require_text(state, _CHAT_PROTOCOL,
                 '        | ChatCompletionNamedToolChoiceParam\n    ) = "none"\n',
                 label=label)
    _require_in_symbol(state, anthropic, "AnthropicServingMessages._convert_tool_choice", (
        'req.tool_choice = "auto" if anthropic_request.tools else "none"',
    ), label=label)
    forbid_text(state, anthropic, "req.tool_choice = None", label=label)
    for path in ("vllm/parser/abstract_parser.py", _CHAT_SERVING):
        forbid_text(state, path, "request.tool_choice is None", label=label)
    forbid_text(state, _CHAT_SERVING, "not request.tool_choice", label=label)
    forbid_text(state, "vllm/tool_parsers/structural_tag_registry.py",
                "if tool_choice is None:", label=label)
    forbid_text(state, "vllm/parser/mistral.py",
                "case None:\n                tool_choice = MistralToolChoiceEnum.auto",
                label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/chat_completion/test_unspecified_tool_choice.py", {
            "test_an_unspecified_tool_choice_is_the_default_for_its_tools": None,
            "test_calls_are_parsed_exactly_where_the_grammar_is_armed": None,
        }, label=label)
    require_python_symbols(state,
        "tests/entrypoints/anthropic/test_anthropic_messages_conversion.py", {
            "TestUnspecifiedToolChoice.test_with_tools_is_auto": None,
            "TestUnspecifiedToolChoice.test_without_tools_is_none": None,
        }, label=label)


def _validate_call_only_separator_before(state: State) -> None:
    _require_in_symbol(state, "vllm/tool_parsers/structural_tag_registry.py",
                       "get_qwen_3_coder_structural_tag", ("suffix_tag = tags[0]",),
                       label="Call-only answer separator precondition")


def _validate_call_only_separator_after(state: State) -> None:
    label = "Call-only answer separator"
    registry = "vllm/tool_parsers/structural_tag_registry.py"
    require_text(state, registry, '_QWEN_THINK_SUFFIX = "\\n\\n"', label=label)
    # The template's blank line after </think> may begin a forced or required
    # answer, and nothing else may; the reasoning-inclusive grammar writes it
    # once, after its own closer.
    _require_in_symbol(state, registry, "get_qwen_3_coder_structural_tag", (
        "separator = ConstStringFormat(value=_QWEN_THINK_SUFFIX)",
        "OptionalFormat(content=separator),",
        "# Free text before the trigger already carries the blank line.",
    ), label=label)
    forbid_text(state, registry, "suffix_tag = tags[0]", label=label)
    require_python_symbols(state, "tests/tool_parsers/test_structural_tag_registry.py", {
        "test_qwen3_call_only_answer_may_follow_the_template_blank_line": None,
    }, label=label)



def _validate_responses_function_list_before(state: State) -> None:
    label = "Responses one function list precondition"
    registry = "vllm/tool_parsers/structural_tag_registry.py"
    require_text(state, registry, "dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]",
                 label=label)
    require_text(state, "vllm/entrypoints/openai/responses/utils.py",
                 "def extract_function_tool_names(", label=label)
    forbid_text(state, "vllm/entrypoints/openai/responses/protocol.py",
                "def check_tool_choice_calls_offered_functions(", label=label)


def _validate_responses_function_list_after(state: State) -> None:
    label = "Responses one function list"
    registry = "vllm/tool_parsers/structural_tag_registry.py"
    # The grammar is given the functions the template offers the model, under
    # the flat names it offers them by; the parser resolves the same names.
    _require_in_symbol(state, registry, "_dump_tools_for_xgrammar", (
        "iter_response_function_tool_dicts([tool])",
    ), label=label)
    require_text(state, registry, "dumped_tools = _dump_tools_for_xgrammar(tools)", label=label)
    forbid_text(state, registry, "_dump_tool_for_xgrammar(", label=label)
    _require_in_symbol(state, registry, "get_qwen_3_coder_structural_tag", (
        "tool_choice 'required' needs a function the model can call",
    ), label=label)
    require_python_symbols(state, "vllm/tool_parsers/utils.py", {
        "response_function_tool_names": ("tools",),
    }, label=label)
    forbid_text(state, "vllm/entrypoints/openai/responses/utils.py",
                "def extract_function_tool_names(", label=label)
    # Every non-string choice is the one a grammar enforces, or refused.
    _require_in_symbol(state, "vllm/entrypoints/openai/responses/protocol.py",
        "ResponsesRequest.check_tool_choice_calls_offered_functions", (
            "offered = response_function_tool_names(self.tools)",
            "isinstance(choice, ToolChoiceFunction)",
            "isinstance(choice, ToolChoiceAllowed)",
            'parameter=f"tool_choice.tools[{index}]"',
            "cannot be enforced",
        ), label=label)
    # allowed_tools reaches the tool grammar the composed parser arms; where
    # none is armed, the tool parser has refused it (qwen-owned-tool-grammar).
    _require_in_symbol(state, "vllm/parser/abstract_parser.py",
        "DelegatingParser._apply_structural_tag", ("ToolChoiceAllowed,",), label=label)
    forbid_text(state, "vllm/parser/abstract_parser.py",
                "tool_choice allowed_tools is enforced by a tool-call", label=label)
    # A choice that lets the model call is refused alike on chat and on
    # Responses while no tool parser parses the calls: one refusal, which both
    # routes ask before rendering.
    renderer = "vllm/renderers/online_renderer.py"
    _require_in_symbol(state, renderer, "OnlineRenderer.require_tool_choice_parsed", (
        "--enable-auto-tool-choice",
        'parameter="tool_choice",',
    ), label=label)
    _require_in_symbol(state, renderer, "OnlineRenderer.render_chat", (
        "self.require_tool_choice_parsed(request.tool_choice, self.parser)",
    ), label=label)
    forbid_text(state, renderer, "tool_parsing_unavailable", label=label)
    _require_in_symbol(state, "vllm/entrypoints/openai/responses/serving.py",
        "OpenAIServingResponses._make_request", (
            "self.online_renderer.require_tool_choice_parsed(",
        ), label=label)
    require_python_symbols(state, "tests/tool_use/test_responses_request_validations.py", {
        "test_a_tool_choice_no_parser_parses_is_refused_alike": None,
        "test_responses_request_names_a_namespace_function_by_its_flat_name": None,
        "test_responses_request_refuses_a_name_the_model_is_not_offered": None,
        "test_responses_request_allowed_tools_lists_offered_functions": None,
        "test_responses_request_allowed_tools_refuses_what_it_cannot_enforce": None,
        "test_responses_request_refuses_a_hosted_tool_choice": None,
    }, label=label)
    require_python_symbols(state, "tests/tool_parsers/test_structural_tag_registry.py", {
        "test_qwen3_admits_the_names_the_responses_prompt_offers": None,
        "test_qwen3_required_refuses_tools_that_offer_no_function": None,
    }, label=label)
    # _make_request asks the renderer's tool-choice check, so the tests that
    # drive it give their renderer OnlineRenderer's own.
    require_text(state, "tests/entrypoints/openai/responses/test_responses_utils.py",
                 "    renderer.require_tool_choice_parsed = partial(\n"
                 "        OnlineRenderer.require_tool_choice_parsed, renderer\n"
                 "    )\n", label=label)

def _validate_batch_parse_from_prompt_before(state: State) -> None:
    label = "Batch parse from the prompt state precondition"
    abstract = "vllm/parser/abstract_parser.py"
    forbid_text(state, abstract, "_reasoning_ended_in_prompt", label=label)
    _require_in_symbol(state, abstract, "DelegatingParser.parse_delta", (
        "if not state.prompt_reasoning_checked and prompt_token_ids is not None:",
    ), label=label)
    # The whole-generation reasoning count starts from the configured state.
    _require_in_symbol(
        state, "vllm/parser/engine/parser_engine.py",
        "ParserEngine.count_reasoning_tokens",
        ("if self.parser_engine_config.initial_state is not ParserState.REASONING:",),
        label=label,
    )
    # Qwen's grammar absorbs every <think> in reasoning.
    require_text(state, "vllm/parser/qwen3.py",
                 '(ParserState.REASONING, "THINK_START"): Transition(', label=label)


def _validate_batch_parse_from_prompt_after(state: State) -> None:
    label = "Batch parse from the prompt state"
    abstract = "vllm/parser/abstract_parser.py"
    engine = "vllm/parser/engine/parser_engine.py"
    signature = ("self", "model_output", "request", "prompt_token_ids", "finish_reason",
                 "stop_reason", "enable_auto_tools", "model_output_token_ids")
    require_python_symbols(state, abstract, {
        "Parser.parse_output": signature,
        "DelegatingParser.parse_output": signature,
        "DelegatingParser._reasoning_ended_in_prompt": ("self", "prompt_token_ids"),
    }, label=label)
    # One decision where the prompt leaves reasoning, read by both paths.
    require_text(state, abstract, "adjust_initial_state_from_prompt(", label=label)
    _require_in_symbol(state, abstract, "DelegatingParser._reasoning_ended_in_prompt", (
        "self._reasoning_parser.adjust_initial_state_from_prompt(prompt_token_ids)",
    ), label=label)
    _require_in_symbol(state, abstract, "DelegatingParser.parse_delta", (
        "state.reasoning_ended = self._reasoning_ended_in_prompt(prompt_token_ids)",
    ), label=label)
    _require_in_symbol(state, abstract, "DelegatingParser.parse_output", (
        "if self._reasoning_ended_in_prompt(prompt_token_ids):",
    ), label=label)
    _require_in_symbol(state, abstract, "Parser.parse_output_delta", (
        "prompt_token_ids=prompt_token_ids",
    ), label=label)
    # A grammar's prompt state survives the batch parse's reset, and with it
    # whether the prompt itself opened the reasoning it leaves.
    require_python_symbols(state, engine, {
        "ParserEngine._start_in": ("self", "state", "reasoning_opened"),
        "ParserEngine.parse_output": signature,
    }, label=label)
    _require_in_symbol(state, engine, "ParserEngine._reset", (
        "initial_state = self._prompt_initial_state",
        "reasoning_opened = self._prompt_opened_reasoning",), label=label)
    # A generation opens its reasoning only with its first token, and only
    # where no prompt opened it; anywhere else the opener is text it wrote.
    streaming = "vllm/parser/engine/streaming_parser_engine.py"
    qwen = "vllm/parser/qwen3.py"
    require_text(state, "vllm/parser/engine/parser_engine_config.py",
                 "reasoning_opener: str | None = None", label=label)
    _require_in_symbol(state, streaming, "StreamingParserEngine.reset", (
        "self.config.reasoning_opener is not None",
        "and self.state is ParserState.REASONING",
        "and not reasoning_opened",), label=label)
    _require_in_symbol(state, streaming, "StreamingParserEngine._on_terminal", (
        "if terminal == self.config.reasoning_opener:",), label=label)
    _require_in_symbol(state, streaming, "StreamingParserEngine._emit_for_state", (
        "self._awaits_opener = False",), label=label)
    forbid_text(state, qwen, '(ParserState.REASONING, "THINK_START")', label=label)
    _require_in_symbol(state, qwen, "qwen3_config", (
        'reasoning_opener="THINK_START" if thinking else None',), label=label)
    _require_in_symbol(state, qwen, "Qwen3Parser.adjust_initial_state_from_prompt", (
        "self._start_in(ParserState.REASONING, reasoning_opened=True)",), label=label)
    require_python_symbols(state, "tests/parser/engine/test_qwen3_reasoning.py", {
        "TestNonStreaming.test_opener_inside_reasoning_is_text": None,
        "TestNonStreaming.test_opener_after_a_prompt_that_opened_reasoning_is_text": None,
        "TestNonStreaming.test_prompt_without_an_opener_leaves_it_to_the_model": None,
        "TestStreaming.test_streaming_opener_inside_reasoning_is_text": None,
        "TestDelegatingGeneratedOpener."
        "test_a_generated_opener_after_an_opened_prompt_is_reasoning_text": None,
    }, label=label)
    # The whole-generation reasoning count starts there too, so it is the
    # count the feed takes from that state.
    _require_in_symbol(state, engine, "ParserEngine.count_reasoning_tokens", (
        "initial_state = self._prompt_initial_state",), label=label)
    _require_in_symbol(state, engine, "ParserEngine.parse_output", (
        "self.adjust_initial_state_from_prompt(prompt_token_ids)",), label=label)
    for path, starts in (("vllm/parser/gemma4.py", 1), ("vllm/parser/inkling.py", 3)):
        forbid_text(state, path, "self._engine.reset(initial_state=", label=label)
        require_text(state, path, "self._start_in(ParserState.", count=starts, label=label)
    # Every complete-output parse is handed the prompt it continues.
    for path, symbol, needle in (
        ("vllm/entrypoints/openai/chat_completion/serving.py",
         "OpenAIServingChat.chat_completion_full_generator",
         "prompt_token_ids=final_res.prompt_token_ids"),
        ("vllm/entrypoints/openai/chat_completion/batch_serving.py",
         "OpenAIServingChatBatch.chat_completion_full_generator_batch",
         "prompt_token_ids=final_res.prompt_token_ids"),
        ("vllm/entrypoints/openai/responses/serving.py",
         "OpenAIServingResponses._collect_response_output",
         "prompt_token_ids=final_res.prompt_token_ids"),
        ("vllm/entrypoints/openai/responses/serving.py",
         "OpenAIServingResponses._make_response_output_items",
         "prompt_token_ids=prompt_token_ids"),
        ("vllm/entrypoints/openai/responses/context.py",
         "ParsableContext.append_output", "prompt_token_ids=output.prompt_token_ids"),
        ("vllm/renderers/online_derenderer.py", "OnlineDerenderer._derender_chat",
         "prompt_token_ids=prompt_token_ids"),
    ):
        _require_in_symbol(state, path, symbol, (needle,), label=label)
    # Derender renders the prompt from the request it is given, with the
    # renderer and settings /render renders with, refuses a request that does
    # not render, and parses from that prompt; nothing parses without it.
    derenderer = "vllm/renderers/online_derenderer.py"
    _require_in_symbol(state, derenderer, "OnlineDerenderer.__init__", (
        "self.online_renderer = OnlineRenderer(",
        "self.parser: type[Parser] | None = self.online_renderer.parser",
    ), label=label)
    _require_ordered(_symbol_source(state, derenderer, "OnlineDerenderer.derender_chat",
                                    label=label), (
        "if self.parser is not None and chat_request is not None:",
        "chat_request = chat_request.model_copy(deep=True)",
        "rendered = await self.online_renderer.render_chat(",
        "chat_request, skip_mm_cache=True",
        "if isinstance(rendered, ErrorResponse):",
        "return rendered",
        "prompt_token_ids = extract_prompt_components(",
        "generate_response, chat_request, prompt_token_ids",
    ), label=label, location=f"{derenderer}:OnlineDerenderer.derender_chat")
    _require_in_symbol(state, derenderer, "OnlineDerenderer._derender_chat", (
        "prompt_token_ids: list[int] | None,\n    ) -> list[ChatCompletionResponseChoice]:",
    ), label=label)
    forbid_text(state, derenderer, "read as the opener and has no text", label=label)
    forbid_text(state, derenderer, "ParserManager.get_parser(", label=label)
    _require_in_symbol(state, "vllm/entrypoints/scale_out/derender/serving.py",
                       "ServingDerender.derender_chat_response", (
                           "if isinstance(choices, ErrorResponse):\n"
                           "            return choices",
                       ), label=label)
    forbid_text(state, "vllm/entrypoints/openai/chat_completion/batch_serving.py",
                "parser.parse(", label=label)
    for path, tests in (
        ("tests/entrypoints/openai/responses/test_serving_responses.py",
         {"test_a_continued_final_message_is_the_answer_on_both_responses_paths": None}),
        ("tests/parser/engine/test_gemma4_streaming_reasoning.py",
         {"TestGemma4PromptOpenReasoning."
          "test_batch_parse_starts_where_the_prompt_leaves_reasoning": None}),
        ("tests/parser/engine/test_reasoning_token_count.py",
         {"TestOneCount."
          "test_the_whole_generation_count_starts_where_the_prompt_leaves": None}),
    ):
        require_python_symbols(state, path, tests, label=label)

def _validate_derender_stop_text_before(state: State) -> None:
    label = "Derender stop-token text precondition"
    forbid_text(state, "vllm/v1/engine/detokenizer.py", "def split_stop_token(", label=label)
    forbid_text(state, "vllm/sampling_params.py", "def model_eos_token_ids(", label=label)
    require_text(state, "vllm/renderers/online_derenderer.py",
                 "choice.token_ids, skip_special_tokens=False", label=label)


def _validate_derender_stop_text_after(state: State) -> None:
    label = "Derender stop-token text"
    detokenizer = "vllm/v1/engine/detokenizer.py"
    derender = "vllm/renderers/online_derenderer.py"
    # One decision of which generated ids carry text, for every route.
    require_python_symbols(state, detokenizer, {
        "split_stop_token": ("token_ids", "stop_terminated", "include_stop_str_in_output"),
        "ended_on_stop_token": ("finish_reason", "stop_reason"),
    }, label=label)
    _require_in_symbol(state, detokenizer, "BaseIncrementalDetokenizer.update", (
        "split_stop_token(",
    ), label=label)
    _require_in_symbol(state, derender, "_text_token_ids", (
        "stop_terminated = ended_on_stop_token(finish_reason, stop_reason)",
        "split_stop_token(", "stop_terminated=stop_terminated",
        # The caller's stop is held to its ids, never repaired by them.
        "ends_on_it = last in eos_token_ids", "ends_on_it = last == stop_reason",
        "raise VLLMValidationError(", "parameter=field",
    ), label=label)
    # Chat with and without a parser, completion, and both streams.
    require_text(state, derender, "_text_token_ids(", count=5, label=label)
    require_text(state, derender, "eos_token_ids=self.eos_token_ids", count=4, label=label)
    forbid_text(state, derender, "choice.token_ids, skip_special_tokens=False", label=label)
    # One definition of the model's EOS ids, read from the engine's inputs.
    sampling = "vllm/sampling_params.py"
    require_python_symbols(state, sampling, {
        "model_eos_token_ids": ("generation_config", "eos_token_id"),
    }, label=label)
    _require_in_symbol(state, sampling, "SamplingParams.update_from_generation_config", (
        "model_eos_token_ids(generation_config, eos_token_id)",
    ), label=label)
    _require_in_symbol(state, derender, "OnlineDerenderer.__init__", (
        "model_eos_token_ids(",
        "model_config.try_get_generation_config(), renderer.get_eos_token_id()",
    ), label=label)
    require_python_symbols(
        state, "tests/entrypoints/scale_out/derender/test_terminal_metadata.py", {
            "test_derender_keeps_empty_output_and_observed_terminal": None,
            "test_derender_only_commits_closed_calls_at_eos": None,
            "test_derender_reads_the_model_eos_ids_the_engine_reads": None,
            "test_derender_refuses_a_stop_its_ids_do_not_end_on": None,
            "test_a_contradicted_stop_is_a_400_naming_the_choice": None,
        }, label=label)

def _validate_output_constraint_beside_tools_before(state: State) -> None:
    label = "Output constraint beside tool calls precondition"
    forbid_text(state, "vllm/entrypoints/openai/engine/protocol.py",
                "def output_constraint_beside_tool_calls(", label=label)
    require_text(state, "vllm/entrypoints/openai/chat_completion/protocol.py",
                 "You can only either use constraints for structured outputs ", label=label)


def _validate_output_constraint_beside_tools_after(state: State) -> None:
    label = "Output constraint beside tool calls"
    # One refusal, named by each surface's own parameter, wherever a call can
    # be made: the tool grammar would otherwise replace the caller's constraint.
    require_python_symbols(state, "vllm/entrypoints/openai/engine/protocol.py", {
        "output_constraint_beside_tool_calls": None,
    }, label=label)
    for path in (
        "vllm/entrypoints/openai/chat_completion/protocol.py",
        "vllm/entrypoints/openai/responses/protocol.py",
        "vllm/entrypoints/anthropic/protocol.py",
    ):
        require_text(state, path, "raise output_constraint_beside_tool_calls(", label=label)
    # Upstream's partial rule (json/regex/choice beside a named choice only,
    # naming no parameter) is subsumed.
    forbid_text(state, "vllm/entrypoints/openai/chat_completion/protocol.py",
                "You can only either use constraints for structured outputs ", label=label)
    forbid_text(state, "vllm/entrypoints/anthropic/serving.py",
                "output_config.format and output_config.format.json_schema", label=label)
    require_python_symbols(state, "tests/entrypoints/openai/test_output_constraint_beside_tools.py", {
        "test_chat_refuses_a_constraint_beside_a_callable_tool": None,
        "test_chat_keeps_the_constraint_when_no_tool_can_be_called": None,
        "test_responses_refuses_a_constraint_beside_a_callable_tool": None,
        "test_anthropic_refuses_an_output_format_beside_a_callable_tool": None,
        "test_anthropic_refuses_an_output_format_without_a_schema": None,
    }, label=label)


def _validate_batch_invariant_native_fp4_before(state: State) -> None:
    label = "Batch-invariant NVFP4 substitute precondition"
    kernels = "vllm/model_executor/kernels/linear/__init__.py"
    _require_in_symbol(state, kernels, "init_nvfp4_linear_kernel", (
        '"kernel is not supported on this platform; falling back to "',
    ), label=label)
    forbid_text(state, kernels, "def refuse_substitute(", label=label)


def _validate_batch_invariant_native_fp4_after(state: State) -> None:
    label = "Batch-invariant NVFP4 substitute refused"
    kernels = "vllm/model_executor/kernels/linear/__init__.py"
    # VLLM_BATCH_INVARIANT's emulation kernel is a substitute like any other:
    # served only when --linear-backend names it.
    _require_in_symbol(state, kernels, "init_nvfp4_linear_kernel", (
        "def refuse_substitute(",
        "EmulationNvFp4LinearKernel in substitutes",
        "and EmulationNvFp4LinearKernel not in requested",
        "VLLM_BATCH_INVARIANT needs the batch-invariant native FP4",
        "if kernel_cls in substitutes and kernel_cls not in requested:",
    ), label=label)
    forbid_text(state, kernels,
                '"kernel is not supported on this platform; falling back to "',
                label=label)
    require_python_symbols(state,
        "tests/model_executor/kernels/test_nvfp4_native_selection.py", {
            "test_batch_invariance_does_not_substitute_emulation": None,
            "test_batch_invariance_serves_named_emulation": None,
            "test_batch_invariance_uses_cutlass_where_it_runs": None,
            "test_a_disabled_native_kernel_is_named_as_the_next_action": None,
        }, label=label)

def _validate_render_every_image_before(state: State) -> None:
    label = "Render carries every image precondition"
    forbid_text(state, "vllm/entrypoints/chat_utils.py", "IMAGE_PART_TYPES", label=label)
    require_text(state, "vllm/entrypoints/scale_out/render/serving.py",
                 'if part.get("type") == "image_url"', label=label)


def _validate_render_every_image_after(state: State) -> None:
    label = "Render carries every image"
    chat_utils = "vllm/entrypoints/chat_utils.py"
    render = "vllm/entrypoints/scale_out/render/serving.py"
    # One decision of what an image part is, shared by chat and render.
    require_text(state, chat_utils, 'IMAGE_PART_TYPES = ("image_url", "input_image")',
                 label=label)
    require_text(state, chat_utils, "elif part_type in IMAGE_PART_TYPES:", label=label)
    _require_in_symbol(state, chat_utils, "content_part_image", (
        "_parse_chat_message_content_mm_part(part)",
        "if part_type not in IMAGE_PART_TYPES:",
    ), label=label)
    require_text(state, render, "content_part_image(part)", label=label)
    forbid_text(state, render, 'if part.get("type") == "image_url"', label=label)
    require_text(state, chat_utils, "input_image.detail is required", label=label)
    # A rendered image span without its image is refused, not read as text.
    require_python_symbols(state, "vllm/renderers/base.py", {
        "BaseRenderer.require_no_rendered_media": ("self", "token_ids"),
    }, label=label)
    require_text(state, "vllm/entrypoints/scale_out/token_in_token_out/serving.py",
                 "renderer.require_no_rendered_media(request.token_ids)", label=label)
    # Every processor decides by one rule: the ids it embeds media at, read
    # from its own output for one dummy item of each modality it accepts,
    # stand only inside a supplied item's span.
    processor = "vllm/multimodal/processing/processor.py"
    _require_in_symbol(state, processor,
        "BaseMultiModalProcessor.rendered_media_token_ids", (
            "self.dummy_inputs.get_dummy_processor_inputs(",
            "self._maybe_apply_prompt_updates(",
            "get_added_vocab()",
        ), label=label)
    _require_in_symbol(state, processor,
        "BaseMultiModalProcessor._find_rendered_prompt_placeholders", (
            "self.rendered_media_token_ids.items()",
            "if token in modality_of and position not in spanned:",
        ), label=label)
    # Derived once, where the renderer builds its processor: a processor that
    # cannot derive it refuses startup, not a request.
    _require_in_symbol(state, "vllm/renderers/base.py", "BaseRenderer.__init__", (
        "dict(self.mm_processor.rendered_media_token_ids)",
    ), label=label)
    require_python_symbols(
        state, "tests/entrypoints/scale_out/token_in_token_out/test_raw_media_boundary.py", {
            "test_render_carries_every_image_chat_renders": None,
            "test_input_image_names_its_required_detail": None,
            "test_rendered_media_spans_require_their_images": None,
            "test_every_processor_reads_its_media_ids_from_its_own_output": None,
            "test_a_processor_without_an_override_refuses_media_ids_as_text": None,
            "test_rendered_media_holds_no_span_beyond_its_images": None,
        }, label=label)

def _validate_rendered_prompt_truncation_before(state: State) -> None:
    label = "Rendered prompt truncation precondition"
    require_text(state, "vllm/entrypoints/openai/responses/protocol.py",
                 'truncate_prompt_tokens=-1 if self.truncation != "disabled" else None,',
                 label=label)
    forbid_text(state, "vllm/entrypoints/openai/chat_completion/protocol.py",
                "def refuse_prompt_truncation(", label=label)


def _validate_rendered_prompt_truncation_after(state: State) -> None:
    label = "Rendered prompt truncation"
    chat = "vllm/entrypoints/openai/chat_completion/protocol.py"
    responses = "vllm/entrypoints/openai/responses/protocol.py"
    # A prompt the template renders is never cut: the request that would ask
    # for it is refused by name, and nothing tokenizes it with a truncation.
    require_python_symbols(state, chat, {
        "ChatCompletionRequest.refuse_prompt_truncation": ("cls", "data"),
    }, label=label)
    require_python_symbols(state, responses, {
        "ResponsesRequest.refuse_prompt_truncation": ("cls", "truncation"),
    }, label=label)
    require_text(state, responses, 'truncation: Literal["auto", "disabled"] = "disabled"',
                 label=label)
    for path in (responses, "vllm/entrypoints/openai/responses/serving.py"):
        forbid_text(state, path, 'truncation != "disabled"', label=label)
    for path in (chat, "vllm/entrypoints/openai/chat_completion/serving.py"):
        forbid_text(state, path, "truncate_prompt_tokens=", label=label)
    require_python_symbols(state, "tests/entrypoints/openai/test_prompt_truncation_refused.py", {
        "test_chat_refuses_prompt_truncation": None,
        "test_chat_tokenizes_without_truncation": None,
        "test_responses_refuses_truncation_auto": None,
        "test_responses_refuses_a_null_truncation": None,
        "test_responses_tokenizes_without_truncation": None,
    }, label=label)

def _validate_kv_transfer_params_keys_before(state: State) -> None:
    label = "declared kv_transfer_params keys precondition"
    # added: connectors declare the request keys their protocol defines.
    forbid_text(state, "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
                "KVTransferParamsKeys", label=label)
    forbid_text(state, "vllm/v1/engine/input_processor.py", "kv_transfer_params",
                label=label)
    require_text(state, "vllm/entrypoints/generate/base/serving.py",
                 "self.has_kv_connector = kv_transfer_config is not None", label=label)
    scheduler = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    require_text(state, scheduler, 'KV_LOAD_TIERS_KEY = "kv_load_tiers"', label=label)
    require_text(state, scheduler, 'params.get("max_offload_tokens")', label=label)


def _validate_kv_transfer_params_keys_after(state: State) -> None:
    label = "declared kv_transfer_params keys result"
    v1 = "vllm/distributed/kv_transfer/kv_connector/v1/"
    # A connector that declares nothing takes nothing; namespaces are prefixes.
    # It says what its operator does to declare what it reads.
    _require_in_symbol(state, v1 + "base.py",
                       "KVConnectorBase_V1.get_kv_transfer_params_keys", (
        "return KVTransferParamsKeys(",
        "undeclared=(",
        "overriding KVConnectorBase_V1.get_kv_transfer_params_keys",
    ), label=label)
    _require_in_symbol(state, v1 + "base.py", "KVTransferParamsKeys.defines", (
        "key in self.keys or key.startswith(tuple(self.namespaces))",
    ), label=label)
    # Each declared key carries the shape of the value its connector reads;
    # a key two declarations define holds its value to both.
    require_python_symbols(state, v1 + "base.py", {
        "KVTransferParamShape": None,
        "KVTransferParamsKeys.shape_of": ("self", "key"),
        "object_with": ("fields", "required"),
    }, label=label)
    _require_in_symbol(state, v1 + "base.py", "KVTransferParamsKeys.__or__", (
        "_all_of(union[name], shape) if name in union else shape",
    ), label=label)
    # The same declaration states the keys a value is read with and those it
    # is never read beside; null and false ask for nothing; a key two
    # declarations define is read with what either reads it with.
    _require_in_symbol(state, v1 + "base.py", "KVTransferParamShape", (
        "requires: tuple[str, ...] = ()",
        "excludes: tuple[str, ...] = ()",
        "def requiring(self, *keys: str)",
        "def excluding(self, *keys: str)",
    ), label=label)
    _require_in_symbol(state, v1 + "base.py", "asks", (
        "return value is not None and value is not False",
    ), label=label)
    _require_in_symbol(state, v1 + "base.py", "_all_of", (
        "requires=tuple(dict.fromkeys(first.requires + second.requires))",
        "excludes=tuple(dict.fromkeys(first.excludes + second.excludes))",
    ), label=label)
    _require_in_symbol(state, v1 + "multi_connector.py",
                       "MultiConnector.get_kv_transfer_params_keys", (
        "keys |= connector_cls.get_kv_transfer_params_keys(child_config)",
    ), label=label)
    _require_in_symbol(state, "vllm/distributed/kv_transfer/kv_connector/factory.py",
                       "KVConnectorFactory.get_kv_transfer_params_keys", (
        "if vllm_config.kv_transfer_config is None:",
        "return KVTransferParamsKeys()",
        "connector_cls.get_kv_transfer_params_keys(vllm_config)",
    ), label=label)
    # The CPU tier acts on the store cap; the tier filter selects only among
    # secondary tiers, so only a tiering spec that has one takes it.
    _require_in_symbol(state, v1 + "offloading_connector.py",
                       "OffloadingConnector.get_kv_transfer_params_keys", (
        "keys={MAX_OFFLOAD_TOKENS_KEY: NON_NEGATIVE_INTEGER}",
        "spec_cls.get_kv_transfer_params_keys(extra_config)",
    ), label=label)
    require_text(state, "vllm/v1/kv_offload/base.py",
                 'KV_LOAD_TIERS_KEY = "kv_load_tiers"', label=label)
    require_text(state, "vllm/v1/kv_offload/base.py",
                 "KV_LOAD_TIERS_SHAPE = list_of(", label=label)
    forbid_text(state, v1 + "offloading/scheduler.py",
                'KV_LOAD_TIERS_KEY = "kv_load_tiers"', label=label)
    _require_in_symbol(state, v1 + "offloading/scheduler.py",
                       "RequestOffloadState.__post_init__",
                       ("params.get(MAX_OFFLOAD_TOKENS_KEY)",), label=label)
    _require_in_symbol(state, "vllm/v1/kv_offload/tiering/spec.py",
                       "TieringOffloadingSpec.get_kv_transfer_params_keys", (
        "keys={KV_LOAD_TIERS_KEY: KV_LOAD_TIERS_SHAPE}",
        "tier_cls.get_kv_transfer_params_keys(tier_config)",
    ), label=label)
    # Every in-tree connector that reads request keys declares them, including
    # those its request_finished returns for the peer node.
    for path, qualname, needles in (
        (v1 + "nixl/connector.py", "NixlBaseConnector",
         ('"do_remote_prefill"', '"transfer_mode"',
          # A proxy fills what a prefill node does not use with null.
          '"remote_block_ids": _REMOTE_BLOCK_IDS',
          '"remote_engine_id": nullable(STRING)',
          '"remote_request_id": nullable(STRING)',
          '"remote_host": nullable(STRING)',
          '"remote_port": nullable(INTEGER)')),
        # Each mode reads do_remote_prefill with the keys it indexes for it.
        (v1 + "nixl/connector.py", "NixlPullConnector",
         ('"do_remote_prefill": BOOLEAN.requiring(',
          '"remote_block_ids", *cls._REMOTE_REQUEST_KEYS',
          '"remote_block_ids": _REMOTE_BLOCK_IDS.requiring(')),
        (v1 + "nixl/connector.py", "NixlPushConnector",
         ('"do_remote_prefill": BOOLEAN.requiring(',
          '*cls._REMOTE_REQUEST_KEYS, "tp_size"')),
        (v1 + "mooncake/mooncake_connector.py", "MooncakeConnector",
         ('"remote_bootstrap_addr"',
          '"transfer_id", "remote_engine_id", "remote_bootstrap_addr"',
          'do_remote_decode = BOOLEAN.requiring("transfer_id")')),
        (v1 + "moriio/moriio_connector.py", "MoRIIOConnector",
         ('"is_request_leader"',
          '"remote_block_ids": nullable(list_of(INTEGER, "integers"))',
          '"remote_engine_id": nullable(STRING)')),
        (v1 + "lmcache_connector.py", "LMCacheConnectorV1",
         ("if not cls._uses_native_adapter(vllm_config):",
          "with use_native set in ",
          'namespaces={"lmcache.": ANY_VALUE}')),
        (v1 + "lmcache_mp_connector.py", "LMCacheMPConnectorUpstream",
         ('"num_lmcache_extra_cached_tokens"',)),
        (v1 + "example_hidden_states_connector.py", "ExampleHiddenStatesConnector",
         ('"hidden_states_path"', '"include_output_tokens"',
          '"allow_custom_save_path", False',
          "takes hidden_states_path only where")),
        ("vllm/v1/kv_offload/tiering/p2p/manager.py", "P2PSecondaryTierManager",
         ("REMOTE_PREFILLER_KEY: peer", "REMOTE_DECODER_KEY: object_with(",
          "REMOTE_KV_SOURCE_KEY: peer",
          'required=("kv_request_id", "remote_host", "remote_port")',
          'required=("kv_request_id",)',
          "REMOTE_PREFILLER_KEY: peer.excluding(",
          "REMOTE_DECODER_KEY, REMOTE_KV_SOURCE_KEY",
          ").excluding(REMOTE_PREFILLER_KEY)",
          "REMOTE_KV_SOURCE_KEY: peer.excluding(REMOTE_PREFILLER_KEY)")),
    ):
        _require_in_symbol(state, path, f"{qualname}.get_kv_transfer_params_keys",
                           needles, label=label)
    # Admission refuses what no configured connector takes, and a non-object.
    processor = "vllm/v1/engine/input_processor.py"
    _require_in_symbol(state, processor, "InputProcessor.__init__", (
        "KVConnectorFactory.get_kv_transfer_params_keys(",
    ), label=label)
    _require_in_symbol(state, processor, "InputProcessor._validate_params", (
        "self._validate_kv_transfer_params(params)",
    ), label=label)
    _require_in_symbol(state, processor, "InputProcessor._validate_kv_transfer_params", (
        "refusal = self.kv_transfer_params_refusal(kv_transfer_params)",
        'parameter="kv_transfer_params"',
    ), label=label)
    _require_ordered(
        _symbol_source(state, processor, "InputProcessor.kv_transfer_params_refusal",
                       label=label),
        (
            "if not isinstance(kv_transfer_params, dict):",
            "declared.defines(key)",
            "return self._undefined_keys_refusal(refused)",
            "if not shape.conforms(value):",
            # Then each value that asks is read only with its companions, and
            # never beside a key it excludes that asks too.
            "if not asks(value):",
            "for companion in shape.requires",
            "if kv_transfer_params.get(companion) is None",
            "for other in shape.excludes",
            "if asks(kv_transfer_params.get(other))",
        ),
        label=label,
        location=f"{processor}:InputProcessor.kv_transfer_params_refusal",
    )
    # Every refusal's next action is one the caller can take, and the
    # reused-prompt-ids key is refused for what this server does instead.
    require_text(state, processor, 'REUSED_PROMPT_IDS_KEY = "prompt_token_ids"',
                 label=label)
    forbid_text(state, processor, "send the request to a server whose", label=label)
    # Only a connector that takes do_remote_prefill is told of a refusal.
    serving = "vllm/entrypoints/generate/base/serving.py"
    forbid_text(state, serving, "has_kv_connector", label=label)
    _require_in_symbol(state, serving, "GenerateBaseServing.__init__", (
        'self.input_processor.kv_transfer_params_keys.defines("do_remote_prefill")',
    ), label=label)
    # The notice reaches the connector without crossing admission, so it
    # carries only parameters admission takes.
    _require_in_symbol(state, serving,
                       "GenerateBaseServing._with_kv_transfer_rejection_cleanup", (
        "self.notifies_remote_prefill_rejection and request.kv_transfer_params",
        "self.input_processor.kv_transfer_params_refusal(kv_transfer_params)",
    ), label=label)
    require_python_symbols(state, "tests/v1/kv_connector/unit/test_kv_transfer_params_keys.py", {
        "test_a_connector_that_declares_nothing_takes_nothing": None,
        "test_the_cpu_offload_tier_takes_only_its_store_cap": None,
        "test_the_tier_filter_is_taken_only_beside_a_secondary_tier": None,
        "test_multi_connector_takes_what_any_child_takes": None,
        "test_lmcache_takes_its_namespace_only_through_its_native_adapter": None,
        "test_every_key_a_connector_reads_or_returns_is_declared": None,
        "test_a_connector_that_declares_nothing_reads_nothing": None,
        "test_a_key_two_connectors_define_holds_its_value_to_both": None,
        "test_each_value_is_held_to_what_its_connector_reads": None,
        "test_a_key_two_connectors_define_is_read_with_what_either_reads_it_with": None,
        "test_only_null_and_false_ask_for_nothing": None,
        "test_a_proxy_fills_the_keys_naming_a_remote_request_with_null": None,
        "test_a_value_is_read_with_the_keys_its_connector_reads_it_with": None,
        "test_a_multi_connector_reads_a_value_with_what_its_children_read_it_with": None,
        "test_a_remote_prefill_requires_every_key_its_connector_indexes_for_it": None,
        "test_a_hidden_states_path_is_taken_only_where_custom_paths_are_allowed": None,
        "test_a_peer_host_is_an_address_or_a_host_name": ("host",),
        "test_a_mooncake_node_is_refused_what_its_role_never_does": ("kv_role",),
    }, label=label)
    # A value a connector reads inside the engine core is held to the values
    # it can act on, not only to its JSON type: a P2P peer's ID, host and port
    # (the parsers read empty ones as no peer, the transport refused the rest
    # in the engine core), a NIXL block-ID list (one per KV cache group), and
    # the remote action a Mooncake node's role lets it take (its scheduler
    # asserts the other).
    base = "vllm/distributed/kv_transfer/kv_connector/v1/base.py"
    for construct in (
        'NON_EMPTY_STRING = _shape(\n    "a non-empty string"',
        'PORT = _shape(\n    "an integer from 1 to 65535",',
        'HOST = _shape("a host name or an IP address", _names_a_host)',
    ):
        require_text(state, base, construct, label=label)
    require_python_symbols(state, base, {
        "_names_a_host": ("value",),
        "non_empty": ("shape", "described"),
        "asking_nothing": ("described",),
    }, label=label)
    require_text(state, "vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py",
                 '"a non-empty list of lists of integers, one per KV cache group",',
                 label=label)
    _require_in_symbol(state, "vllm/v1/kv_offload/tiering/p2p/manager.py",
                       "P2PSecondaryTierManager.get_kv_transfer_params_keys", (
        '"kv_request_id": NON_EMPTY_STRING,',
        '"remote_host": HOST,',
        '"remote_port": PORT,',
        '{"kv_request_id": NON_EMPTY_STRING}, required=("kv_request_id",)',
    ), label=label)
    _require_in_symbol(state,
        "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py",
        "MooncakeConnector.get_kv_transfer_params_keys", (
            'if kv_role == "kv_producer":',
            '"false on this kv_producer node, which prefills for a decode "',
            'if kv_role == "kv_consumer":',
            '"false on this kv_consumer node, which decodes what a prefill "',
        ), label=label)
    require_python_symbols(state, "tests/v1/engine/test_kv_transfer_params_admission.py", {
        "test_the_cpu_offload_tier_admits_its_store_cap": None,
        "test_a_key_the_configured_connector_does_not_take_is_refused_naming_both": None,
        "test_without_a_connector_every_key_is_refused": None,
        "test_a_connector_that_declares_nothing_is_named_as_declaring_nothing": None,
        "test_kv_transfer_params_through_vllm_xargs_must_be_an_object": None,
        "test_no_notice_without_a_connector_that_takes_remote_prefill": None,
        "test_a_connector_that_takes_remote_prefill_is_told_of_the_refusal": None,
        "test_reused_prompt_ids_are_refused_for_what_the_server_does_instead": None,
        "test_lmcache_without_its_native_adapter_names_use_native": None,
        "test_a_value_the_connector_cannot_read_is_refused_naming_its_shape": None,
        "test_parameters_admission_refuses_reach_no_connector_as_a_notice": None,
        "test_a_key_filled_with_null_or_false_asks_for_nothing": None,
        "test_a_value_without_the_keys_it_is_read_with_is_refused_naming_them": None,
        "test_a_key_beside_one_it_is_never_read_with_is_refused": None,
        "test_a_notice_names_the_blocks_only_with_the_keys_that_name_them": None,
    }, label=label)
    require_python_symbols(state, "tests/entrypoints/openai/chat_completion/test_chat_completion.py", {
        "test_kv_transfer_params_no_connector_takes_are_refused": None,
    }, label=label)


def _validate_responses_tools_never_given_before(state: State) -> None:
    label = "Responses tools the template is never given precondition"
    utils = "vllm/entrypoints/openai/responses/utils.py"
    require_text(state, utils, 'if not tools or (tool_choice == "none" and '
                 "exclude_tools_when_tool_choice_none):", label=label)
    forbid_text(state, utils, "_tool_the_template_is_never_given", label=label)
    require_text(state, "tests/entrypoints/openai/responses/test_parsable_context.py",
                 "async def test_mcp_tool_call(", label=label)


def _validate_responses_tools_never_given_after(state: State) -> None:
    label = "Responses tools the template is never given"
    utils = "vllm/entrypoints/openai/responses/utils.py"
    # The template's tool list refuses, before any return and so under every
    # choice, each declared tool it would not be given.
    forbid_text(state, utils, 'if not tools or (tool_choice == "none" and '
                "exclude_tools_when_tool_choice_none):", label=label)
    require_python_symbols(state, utils, {
        "_tool_the_template_is_never_given": ("parameter", "kind", "namespace"),
    }, label=label)
    _require_in_symbol(state, utils, "_tool_the_template_is_never_given", (
        "with or without a tool ",
        "Declare what it does as a function tool your client executes.",
    ), label=label)
    _require_in_symbol(state, utils, "construct_tool_dicts", (
        'raise _tool_the_template_is_never_given(f"tools[{index}]", tool.type)',
        'parameter=f"tools[{index}]"',
        'f"tools[{index}].tools[{member_index}]"',
        'if tool_choice == "none" and exclude_tools_when_tool_choice_none:',
    ), label=label)
    for path, tests in (
        ("tests/entrypoints/openai/responses/test_responses_utils.py", {
            "test_a_tool_the_chat_template_is_never_given_is_refused": None,
            "test_the_chat_template_is_given_every_declared_function": None,
            "test_a_tool_the_chat_template_is_never_given_is_refused_before_rendering":
                None,
        }),
        ("tests/tool_use/test_responses_request_validations.py", {
            "test_responses_request_leaves_hosted_tools_to_the_route": None,
        }),
        ("tests/entrypoints/openai/responses/test_parsable_context.py", {
            "test_a_hosted_tool_is_refused_beside_a_tool_server": None,
        }),
    ):
        require_python_symbols(state, path, tests, label=label)
    forbid_text(state, "tests/entrypoints/openai/responses/test_parsable_context.py",
                "async def test_mcp_tool_call(", label=label)
    require_text(state, "tests/entrypoints/openai/responses/test_responses_utils.py",
                 "    serving.parser = SimpleNamespace(tool_parser_cls=object)\n",
                 label=label)


def _validate_chat_stream_logprobs_before(state: State) -> None:
    label = "chat stream log probabilities precondition"
    require_text(state, "vllm/entrypoints/openai/chat_completion/serving.py",
                 "not request.return_token_ids or hide_stream_metadata", label=label)
    require_text(state, "vllm/tool_parsers/poolside_v1_tool_parser.py",
                 'wants_logprobs = getattr(request, "logprobs", None)', label=label)


def _validate_chat_stream_logprobs_after(state: State) -> None:
    label = "chat stream log probabilities result"
    serving = "vllm/entrypoints/openai/chat_completion/serving.py"
    # A chat choice's log probabilities are those of every token it generated.
    # A step the parser releases no text for still sends its chunk when the
    # chunk carries the step's token ids or log probabilities, so the stream
    # reports the list the full response reports.
    source = _require_in_symbol(state, serving,
        "OpenAIServingChat.chat_completion_stream_generator", (
            "include_token_ids = (",
            "request.return_token_ids and not hide_stream_metadata",
            "and not include_token_ids",
            "and logprobs is None",
        ), label=label)
    _require_ordered(source, (
        "if hide_stream_metadata:", "logprobs = None", "include_token_ids = (",
        "if delta_message is None:", "and not include_token_ids",
        "and logprobs is None", "continue", "delta_message = DeltaMessage()",
    ), label=label, location=f"{serving}:chat_completion_stream_generator")
    forbid_text(state, serving, "not request.return_token_ids or hide_stream_metadata",
                label=label)
    # No parser emits a placeholder delta to keep a step's log probabilities.
    poolside = "vllm/tool_parsers/poolside_v1_tool_parser.py"
    forbid_text(state, poolside, "wants_logprobs", label=label)
    forbid_text(state, poolside, 'return DeltaMessage(content="")', label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/chat_completion/test_serving_chat.py", {
            "test_a_stream_reports_the_log_probability_of_a_step_with_no_text": None,
        }, label=label)


def _validate_messages_one_rule_before(state: State) -> None:
    label = "chat messages read by one rule precondition"
    require_text(state, "vllm/entrypoints/openai/chat_completion/protocol.py",
                 'reasoning_content = msg.pop("reasoning_content", None)', label=label)
    forbid_text(state, "vllm/entrypoints/chat_utils.py",
                "def normalize_request_messages(", label=label)
    forbid_text(state, "vllm/entrypoints/serve/tokenize/protocol.py",
                "normalize_request_messages", label=label)


def _validate_messages_one_rule_after(state: State) -> None:
    label = "chat messages read by one rule result"
    chat_utils = "vllm/entrypoints/chat_utils.py"
    # One function reads a caller's chat messages: a past turn's reasoning
    # under its legacy name is the turn's reasoning, on every request type
    # that carries messages, so /tokenize counts what chat renders.
    _require_in_symbol(state, chat_utils, "normalize_request_messages", (
        'msg["tool_calls"] = list(tool_calls)',
        'reasoning_content = msg.pop("reasoning_content", None)',
        'if reasoning_content is not None and msg.get("reasoning") is None:',
        'msg["reasoning"] = reasoning_content',
    ), label=label)
    for path, qualname in (
        ("vllm/entrypoints/openai/chat_completion/protocol.py",
         "ChatCompletionRequest._normalize_messages_before"),
        ("vllm/entrypoints/serve/tokenize/protocol.py",
         "TokenizeChatRequest._normalize_messages_before"),
        ("vllm/entrypoints/pooling/base/protocol.py",
         "ChatRequestMixin._normalize_messages_before"),
    ):
        _require_in_symbol(state, path, qualname, (
            "return normalize_request_messages(data)",
        ), label=label)
        forbid_text(state, path, 'msg.pop("reasoning_content"', label=label)
    require_python_symbols(state,
        "tests/tool_use/test_chat_completion_request_validations.py", {
            "test_reasoning_content_normalized_to_reasoning": ("request_type",),
            "test_reasoning_takes_precedence_over_reasoning_content": ("request_type",),
            "test_no_reasoning_fields_unchanged": ("request_type",),
        }, label=label)


def _validate_responses_unhonoured_fields_before(state: State) -> None:
    label = "Responses unhonoured fields precondition"
    serving = "vllm/entrypoints/openai/responses/serving.py"
    require_text(state, serving,
                 "def _make_not_found_error(self, response_id: str) -> ErrorResponse:",
                 label=label)
    forbid_text(state, serving, 'parameter="max_tool_calls"', label=label)
    require_text(state, "vllm/entrypoints/openai/responses/protocol.py",
                 '"prompt template is not supported", parameter="prompt"', label=label)


def _validate_responses_unhonoured_fields_after(state: State) -> None:
    label = "Responses unhonoured fields result"
    serving = "vllm/entrypoints/openai/responses/serving.py"
    protocol = "vllm/entrypoints/openai/responses/protocol.py"
    # A server that stores no responses refuses previous_response_id at
    # intake, naming the field, the cause and both next actions.
    _require_in_symbol(state, serving,
        "OpenAIServingResponses._validate_create_responses_input", (
            "if request.previous_response_id is not None and not self.enable_store:",
            "this server stores no responses",
            "conversation so far in `input`",
            "`VLLM_ENABLE_RESPONSES_API_STORE=1`",
            "status_code=HTTPStatus.BAD_REQUEST,",
            'param="previous_response_id",',
        ), label=label)
    # A stored-response lookup names the field the id came in, and a server
    # that stores nothing says so.
    _require_in_symbol(state, serving, "OpenAIServingResponses._make_not_found_error", (
        'self, response_id: str, parameter: str = "response_id"',
        "if not self.enable_store:",
        "This server stores no responses",
        "param=parameter,",
    ), label=label)
    # max_tool_calls counts built-in calls; where the server runs one itself
    # it is refused before anything is admitted.
    create = _require_in_symbol(state, serving, "OpenAIServingResponses._create_responses", (
        'prev_response_id, parameter="previous_response_id"',
        "if request.max_tool_calls is not None and available_tools:",
        'parameter="max_tool_calls",',
    ), label=label)
    _require_ordered(create, (
        "available_tools = builtin_tool_list",
        "if request.max_tool_calls is not None and available_tools:",
        "generator = await self._generate_with_builtin_tools(",
    ), label=label, location=f"{serving}:_create_responses")
    _require_in_symbol(state, protocol, "ResponsesRequest.validate_prompt", (
        "holds no prompt templates",
        "as instructions and input, and omit prompt.",
        'parameter="prompt",',
    ), label=label)
    forbid_text(state, protocol, "prompt template is not supported", label=label)
    # A parameter of the pinned API the request does not declare is refused,
    # the set read from the client types; of reasoning and text only effort
    # and format are applied; the rest is refused unless it asks only for what
    # the server does anyway. One refusal, one table of reasons.
    for construct in (
        "_API_PARAMETERS = frozenset(\n"
        "    ResponseCreateParamsStreaming.__required_keys__\n"
        "    | ResponseCreateParamsStreaming.__optional_keys__\n)",
        '    "reasoning": frozenset({"effort"}),\n'
        '    "text": frozenset({"format"}),\n',
        '    "reasoning.context": frozenset({"auto"}),\n'
        '    "text.verbosity": frozenset({"medium"}),\n'
        '    "service_tier": frozenset({"auto", "default"}),\n'
        '    "stream_options.include_obfuscation": frozenset({False}),\n',
    ):
        _require(_source(state, protocol, label=label).count(construct) == 1,
                 f"{label}: {protocol} lacks the single construct {construct!r}")
    _require_in_symbol(state, protocol, "_refuse_not_applied", (
        "_NOT_APPLIED.get(",
        "parameter=parameter,",
    ), label=label)
    applied = _require_in_symbol(state, protocol,
        "ResponsesRequest.refuse_what_is_not_applied", (
            "for name in sorted(_API_PARAMETERS - type(self).model_fields.keys()):",
            "for owner, applied in _APPLIED_SETTINGS.items():",
            "sorted(type(settings).model_fields.keys() - applied)",
            '_refuse_not_applied("service_tier", self.service_tier)',
            '_refuse_not_applied("stream_options.include_obfuscation", obfuscation)',
            '_refuse_not_applied("include", "reasoning.encrypted_content")',
        ), label=label)
    require_text(state, protocol,
                 '    @model_validator(mode="after")\n'
                 '    def refuse_what_is_not_applied(self) -> "ResponsesRequest":\n',
                 label=label)
    _require("raise " not in applied,
             f"{label}: ResponsesRequest.refuse_what_is_not_applied refuses "
             "other than through _refuse_not_applied")
    for parameter in ("conversation", "context_management", "moderation",
                      "prompt_cache_options", "prompt_cache_retention",
                      "reasoning.summary", "reasoning.generate_summary",
                      "reasoning.context", "reasoning.mode", "text.verbosity",
                      "service_tier", "stream_options.include_obfuscation",
                      "include"):
        require_text(state, protocol, f'    "{parameter}": (\n', label=label)
    # The route refuses what its launch cannot serve: a store sent true with
    # the store off, Harmony history without Harmony, and the outputs of a
    # code interpreter it runs itself.
    _require_in_symbol(state, serving,
        "OpenAIServingResponses._validate_create_responses_input", (
            "if request.previous_input_messages and not self.use_harmony:",
            'param="previous_input_messages",',
            '"store" in request.model_fields_set',
            'param="store",',
        ), label=label)
    _require_ordered(create, (
        "if request.max_tool_calls is not None and available_tools:",
        '"code_interpreter_call.outputs" in request.include',
        'and "python" in available_tools',
        'value="code_interpreter_call.outputs",',
        "generator = await self._generate_with_builtin_tools(",
    ), label=label, location=f"{serving}:_create_responses")
    forbid_text(state, serving, "we opted\n            # to implicitly disable store",
                label=label)
    require_python_symbols(state,
        "tests/entrypoints/openai/responses/test_serving_responses.py", {
            "test_previous_response_id_on_a_server_that_stores_nothing_is_refused": None,
            "test_an_unknown_previous_response_id_is_named_by_its_field": None,
            "test_max_tool_calls_is_refused_where_built_in_calls_are_not_counted": (
                "runs_the_tool",),
            "test_store_sent_true_on_a_server_that_stores_nothing_is_refused": None,
            "test_previous_input_messages_without_harmony_are_refused": None,
            "test_code_interpreter_outputs_are_refused_where_the_server_runs_the_code": (
                "runs_the_tool",),
        }, label=label)
    require_python_symbols(state, "tests/tool_use/test_responses_request_validations.py", {
        "test_a_prompt_template_is_refused_naming_what_to_send_instead": None,
        "test_a_parameter_this_server_does_not_apply_is_refused": (
            "fields", "parameter"),
        "test_a_value_that_asks_for_what_the_server_does_is_served": ("fields",),
        "test_every_responses_parameter_is_applied_or_refused": None,
    }, label=label)


_CONFTEST = "tests/conftest.py"
_GPU_MARK = (
    '        "gpu: the test needs a GPU (a GPU build of vLLM resolves no platform "\n'
)
_NETWORK_MARK = (
    '        "network: the test needs the network: the Hugging Face hub or a remote URL",\n'
)


def _validate_test_declarations_before(state: State) -> None:
    label = "Reviewed test declarations precondition"
    require_text(state, _CONFTEST, "def pytest_collection_modifyitems(config, items):\n",
                 label=label)
    forbid_text(state, _CONFTEST, "def pytest_configure(config):", label=label)


def _validate_test_declarations_after(state: State) -> None:
    label = "Reviewed test declarations result"
    # The tree's own tests register what a test may need, so the marks mean
    # the same under any runner.
    _require_in_symbol(state, _CONFTEST, "pytest_configure", (
        'config.addinivalue_line(\n        "markers",\n' + _GPU_MARK,
        'config.addinivalue_line(\n        "markers",\n' + _NETWORK_MARK,
    ), label=label)
    # A test this tree refuses by design says so, with the exception and the
    # cause, strictly: a change that lets it pass fails until the mark goes.
    connector = "tests/multimodal/media/test_connector.py"
    require_text(state, connector,
                 "IMAGE_CONTRACT_REFUSES = pytest.mark.xfail(\n    strict=True,\n"
                 "    raises=VLLMUnprocessableEntityError,\n",
                 label=label)
    require_text(state, connector, "Refused by the Qwen3.8 image contract",
                 label=label)
    require_text(state, connector, "\n@IMAGE_CONTRACT_REFUSES\n", count=3,
                 label=label)
    # The round trip of the admitted form runs: its reference images are not
    # fetched, and only its non-PNG cases are refused.
    _require_in_symbol(state, connector, "url_images", (
        "ImageAsset(base).read_bytes(ext)",
    ), label=label)
    forbid_text(state, connector, "local_asset_server.get_image_asset(", label=label)
    require_text(state, connector,
                 '            suffix, marks=() if suffix == ".png" else '
                 "IMAGE_CONTRACT_REFUSES\n",
                 label=label)
    # A module that cannot be imported without a GPU says so where the runner
    # reads it without importing it: its module-level pytestmark.
    require_text(state, "tests/distributed/test_rocm_quick_reduce.py",
                 "pytestmark = [\n    pytest.mark.gpu,\n", label=label)
    require_text(state, "tests/v1/logits_processors/test_correctness.py",
                 "\npytestmark = pytest.mark.gpu\n", label=label)


_INPUT_PROCESSOR = "vllm/v1/engine/input_processor.py"
_GENERATE_BASE_SERVING = "vllm/entrypoints/generate/base/serving.py"


def _validate_priority_refusal_before(state: State) -> None:
    label = "Priority refusal precondition"
    forbid_text(state, _INPUT_PROCESSOR, "def _validate_priority(", label=label)
    require_text(state, _GENERATE_BASE_SERVING,
                 "                except ValueError:\n                    pass\n",
                 label=label)


def _validate_priority_refusal_after(state: State) -> None:
    label = "Priority refusal result"
    # Admission, which every route that reaches the engine passes, refuses a
    # priority the scheduler would not apply, before anything else it checks.
    _require_ordered(_symbol_source(state, _INPUT_PROCESSOR,
                                    "InputProcessor.process_inputs", label=label), (
        ") -> EngineCoreRequest:\n        self._validate_priority(priority)\n",
        "self._validate_params(params, supported_tasks)",
    ), label=label, location=f"{_INPUT_PROCESSOR}:InputProcessor.process_inputs")
    _require_in_symbol(state, _INPUT_PROCESSOR, "InputProcessor._validate_priority", (
        "if priority == 0:\n            return\n",
        'if policy != "priority":',
        'parameter="priority",',
        '"arrival order whatever its priority. Send priority 0 or omit "',
        '"it, or serve with --scheduling-policy priority."',
    ), label=label)
    # A header the route reads the priority from is refused when it is not one.
    _require_in_symbol(state, _GENERATE_BASE_SERVING, "GenerateBaseServing._get_priority", (
        "raise VLLMValidationError(",
        "parameter=PRIORITY_HEADER,",
    ), label=label)
    forbid_text(state, _GENERATE_BASE_SERVING,
                "                except ValueError:\n                    pass\n", label=label)
    require_python_symbols(state, "tests/v1/engine/test_priority_admission.py", {
        "test_a_priority_first_come_first_served_would_not_apply_is_refused": ("priority",),
        "test_a_priority_that_asks_for_nothing_or_is_applied_is_admitted": None,
        "test_a_priority_header_that_is_not_an_integer_is_refused": ("header",),
        "test_a_priority_header_that_is_an_integer_is_the_priority": None,
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
    "qwen-repeated-parameter-refusal": SemanticContract(
        rationale=(
            "A Qwen tool call that names one parameter twice was refused with "
            "a RuntimeError, so the response was an HTTP 500 whose message "
            "told the caller to inspect the call the response had discarded. "
            "The model wrote the repeat; the request was sound and the server "
            "did not fail. The parser answers it with the unprocessable-entity "
            "refusal (422), typed as its own and naming the call and the "
            "parameter, on the error surface every refusal is answered on, "
            "and no call is published."
        ),
        removal_condition=(
            "Remove when pinned upstream answers a Qwen XML tool call that "
            "repeats a parameter with a typed refusal naming it."
        ),
        validate_before=_validate_repeated_parameter_refusal_before,
        validate_after=_validate_repeated_parameter_refusal_after,
    ),
    "template-refusals-name-their-parameter": SemanticContract(
        rationale=(
            "Every chat template refusal became a request error naming "
            "parameter 'messages', so a caller who sent an unsupported "
            "reasoning_effort or preserve_thinking: false was told the "
            "conversation was at fault. The template names the variable each "
            "refusal concerns, every surface records the request parameter "
            "behind each template variable it sets, and the refusal names that "
            "parameter, or none when the value did not come from the request."
        ),
        removal_condition=(
            "Remove when pinned upstream names the request parameter behind "
            "the template variable a chat template refuses."
        ),
        validate_before=_validate_refusal_parameter_before,
        validate_after=_validate_refusal_parameter_after,
    ),
    "startup-plan-admission-bound": SemanticContract(
        rationale=(
            "kv-physical-free-memory bounds the declared KV capacity by memory "
            "initially free, but the startup plan still persisted the card's "
            "total memory minus measured residents, so a plan-shortcut boot "
            "admitted against a larger bound than the profiled boot it "
            "replaces whenever memory was occupied before profiling. The plan "
            "persists the one bound determine_available_memory derives, keyed "
            "on the installed source that derived it, since builds that patch "
            "one upstream commit share its version string."
        ),
        removal_condition=(
            "Remove when pinned upstream persists the same admission bound its "
            "profiled boot admits against."
        ),
        validate_before=_validate_plan_bound_before,
        validate_after=_validate_plan_bound_after,
    ),
    "qwen-arguments-read-by-grammar": SemanticContract(
        rationale=(
            "The Qwen grammar admits a JSON value whose strings carry "
            "</parameter>, <parameter=NAME> or </function>, but the parser cut "
            "every value at its first </parameter> and ended the call at the "
            "first </function> after one: a grammar-legal call was published "
            "truncated, refused as a repeated parameter (HTTP 500) or returned "
            "as content. The grammar's productions are decided once; the "
            "grammar is rendered from them and the parser reads argument text "
            "by them, and a call ends only at its closing sequence after "
            "arguments that read whole."
        ),
        removal_condition=(
            "Remove when pinned upstream reads Qwen XML tool arguments by the "
            "same productions as the grammar that constrained them."
        ),
        validate_before=_validate_grammar_read_arguments_before,
        validate_after=_validate_grammar_read_arguments_after,
    ),
    "nvfp4-native-kernel-required": SemanticContract(
        rationale=(
            "Upstream's automatic NVFP4 kernel selection falls through to Marlin "
            "(weight-only W4A16) or emulation on a GPU without native FP4 and "
            "only warns, so a W4A4 checkpoint is served with different "
            "arithmetic. Selection refuses an implicit substitute at layer "
            "construction, naming the device capability and why each native "
            "kernel is unavailable; a substitute named with --linear-backend is "
            "still served."
        ),
        removal_condition=(
            "Remove when pinned upstream refuses to substitute a weight-only or "
            "emulated kernel for an NVFP4 W4A4 layer unless it is requested."
        ),
        validate_before=_validate_native_fp4_before,
        validate_after=_validate_native_fp4_after,
    ),
    "template-authored-control-tokens": SemanticContract(
        rationale=(
            "Upstream renders the chat template to one string and tokenizes it "
            "whole, so request text (a file the model read, command output, a "
            "tool result, the model's own re-sent history) that spells "
            "<|im_end|>, </think> or <tool_call> becomes that control token: a "
            "forged turn boundary, thinking boundary or tool call. The template "
            "renders with authorship and only text the template wrote may match "
            "the added vocabulary; request text is encoded without it. The text "
            "is never changed, and a prompt whose request text spells no added "
            "token encodes to the ids it always did. The template is also the "
            "only writer of a chat prompt: upstream's chat renderer takes the ids "
            "in kv_transfer_params.prompt_token_ids, which disaggregated serving "
            "uses to hand a prefill node's ids to the decode node, as the prompt "
            "in place of rendering, so a caller could write any id -- a turn "
            "boundary, a tool call, a thinking boundary -- past the template on "
            "Chat Completions, Responses and Anthropic Messages alike. A guarantee "
            "with that door is not one, so every chat request is rendered. The "
            "cost: a decode node renders the messages it is sent, as the prefill "
            "node did, rather than reusing that node's ids."
        ),
        removal_condition=(
            "Remove when pinned upstream encodes chat-template output so that "
            "only the template's own text can produce added-token ids, and "
            "renders every chat request rather than taking its ids from the "
            "request."
        ),
        validate_before=_validate_template_authorship_before,
        validate_after=_validate_template_authorship_after,
    ),
    "kv-scope-single-flight": SemanticContract(
        rationale=(
            "One kv_scope names one line of work, yet upstream runs two "
            "requests under one ID at once: each replaces the other's retained "
            "context and neither continues the other. The one API server "
            "process knows every request in flight, so admission refuses an "
            "overlapping request under the same ID by name, before the engine "
            "or any response sees it; finishing or aborting the earlier "
            "request frees the ID. Several API server processes would each see "
            "only their own share, so the engine refuses to start with more "
            "than one."
        ),
        removal_condition=(
            "Remove when pinned upstream refuses a second in-flight request "
            "under one agent identity at admission."
        ),
        validate_before=_validate_single_flight_before,
        validate_after=_validate_single_flight_after,
    ),
    "generation-admission-before-response": SemanticContract(
        rationale=(
            "Upstream streaming surfaces return their response before the "
            "engine admits the request, so every admission refusal (a missing "
            "or malformed kv_scope, an overlapping request, input validation) "
            "arrives after HTTP 200 as an in-stream error, and Responses cuts "
            "its stream after response.in_progress with no error event at all. "
            "Admission is a step of its own that every generation surface "
            "completes before it answers; the admitted stream owns its request "
            "and aborts it when closed, cancelled or dropped; Responses ends "
            "every later failure with its error event."
        ),
        removal_condition=(
            "Remove when pinned upstream admits a generation request before "
            "answering on every surface and ends every Responses stream "
            "failure with an error event."
        ),
        validate_before=_validate_admission_before,
        validate_after=_validate_admission_after,
    ),
    "qwen-unique-tool-parameters": SemanticContract(
        rationale=(
            "Qwen's additional-properties grammar can emit a parameter name "
            "already emitted in the same call. A dictionary assignment then "
            "silently replaced the first model value. Refuse repeated names "
            "before schema conversion on both complete and partial output."
        ),
        removal_condition=(
            "Remove when upstream's Qwen XML converter refuses repeated "
            "parameter names before any decoded tool call is published."
        ),
        validate_before=_validate_unique_parameters_before,
        validate_after=_validate_unique_parameters_after,
    ),
    "qwen-owned-tool-grammar": SemanticContract(
        rationale=(
            "XGrammar's Qwen value channel excludes only the parameter closer, "
            "so a value could absorb the next parameter's opener and publish a "
            "different call that was still well formed, ordered and "
            "schema-satisfying. The exclusion cannot be reached through the "
            "structural-tag API, so vLLM owns the Qwen tag and excludes the "
            "opener too, reproducing every other production exactly. Owning a "
            "builder means taking the dispatch's one argument list, the "
            "request's call limit included: every registered builder takes it "
            "and holds the limit wherever its format can express one call, and "
            "a refusal offers the formats the registry builds, read from it. A "
            "limit no grammar holds -- an XGrammar builtin format, which takes "
            "none, or 'auto' without a strict tool where the format arms its "
            "grammar only for one, or a tool parser whose format arms no grammar "
            "(it names none, or VLLM_ENFORCE_STRICT_TOOL_CALLING is off) and "
            "whose own grammar takes no limit -- is refused naming "
            "parallel_tool_calls, by one refusal, since the response layer "
            "drops no call."
        ),
        removal_condition=(
            "Remove when XGrammar's Qwen template excludes the parameter opener "
            "from unconstrained values, or lets a caller supply the exclusions."
        ),
        validate_before=_validate_qwen_grammar_before,
        validate_after=_validate_qwen_grammar_after,
    ),
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
            "and all surrounding content on both transports, including "
            "whitespace-only text and unfulfilled tool choices. Shared parsing "
            "and delegation retain content without a whitespace deletion mode. "
            "Upstream's deletion shaped only the response: the model's own "
            "template, and the served one derived from it, trims every "
            "message's content before rendering, so the next prompt is "
            "byte-identical with or without it and the trained shape is kept."
        ),
        removal_condition=(
            "Remove when upstream preserves exact XML string and all "
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
            "uses the actual streamed items, so both transports report the same "
            "items, IDs and statuses. A message carries no log probabilities: "
            "they would be those of the tokens its text came from, the parser "
            "reports each item's text and has no token spans, and one generated "
            "token can end the message and begin a call, so no list of whole "
            "tokens is a message's. The request refuses include "
            "message.output_text.logprobs and a non-zero top_logprobs by name, "
            "and nothing on either transport builds Responses log probabilities."
        ),
        removal_condition=(
            "Remove when upstream uses one item identity from stream creation "
            "through terminal output, and either attributes every generated "
            "token to the one item that wrote it or refuses message log "
            "probabilities."
        ),
        validate_before=_validate_responses_identity_before,
        validate_after=_validate_responses_identity_after,
    ),
    "responses-history-integrity": SemanticContract(
        rationale=(
            "Responses replay dropped all but the first text/reasoning block and "
            "bypassed Chat's tool ID correlation. Preserve the supplied content and "
            "validate every positional tool history through one shared boundary, "
            "whose refusal names the field the caller sent -- input[k] or "
            "previous_response_id on Responses, messages[i] on chat -- and the id "
            "field each protocol uses (call_id or tool_call_id), never a position "
            "in the converted chat list the caller never saw."
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
            "profiling, nor memory serving holds beside its pool. Bound it by "
            "initially free memory minus what stays resident after a profile that "
            "runs each phase at its largest in the residency serving runs it in -- "
            "the encoder with the reclaimable workspace released, the text step "
            "attending to a stand-in pool at full context with it resident, its "
            "sampler and its prompt log probabilities at the most a request is "
            "admitted with, computed as serving computes them -- the peak of "
            "those phases above that with the stand-in pool taken off, CUDA "
            "graph and frontend reservations. The profile witnesses what its "
            "text step ran -- every layer holding KV cache or state read its "
            "metadata, every reserved reclaimable workspace was requested -- "
            "and refuses startup otherwise. The V2 runner, whose profile runs "
            "no attention, holds no declared pool."
        ),
        removal_condition=(
            "Remove when upstream's authoritative bound includes pre-snapshot "
            "residents exactly once, profiles attention and every workspace in "
            "the phase that holds it on both model runners, and keeps "
            "utilization as an estimate only."
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
            "A call is the grammar's exact trigger language and nothing looser: a "
            "bare function opener, or a tool-call opener the trigger does not "
            "follow, is content, and markup inside a parameter value is the "
            "value's text. With tool_choice none, or no tools, a call-shaped span "
            "is forwarded as the text it was. The batch tool pass splits on the "
            "generated ids after the reasoning boundary, as streaming does, so a "
            "text lookalike of a marker is content on both transports; a finished "
            "parse reports a call cut before its wrapper as open. Because the "
            "language is the grammar's, the Qwen tool parser is neither selected "
            "nor built while VLLM_ENFORCE_STRICT_TOOL_CALLING arms no grammar."
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
            "a refusal raised before a route ran -- typed body validation, an HTTP or "
            "framework error -- used the OpenAI envelope, and a stream replaced a "
            "classified failure with an internal_error. Refuse unrenderable input, "
            "render the caller's failure flag, and answer every refusal on a Messages "
            "path, in a stream too, in the Anthropic envelope for its status."
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
            "correlation unsafe to guess. One function resolves the final ceiling, and "
            "one boundary validates a tool history, its refusal naming where the "
            "caller sent the offending message or call: messages[i] on chat; on "
            "Anthropic, which records each origin as it converts, the message or "
            "content block and its tool_use_id, never a position in the converted "
            "chat list. An Anthropic tool_use sent without an id is refused as one, "
            "never given an id invented from the clock."
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
            "The Messages routes answered every failure as HTTP 500 internal_error, "
            "typed Pydantic request/translation validation and the engine's own "
            "request validation (VLLMValidationError on the path into the engine -- a "
            "generation naming no kv_scope, for one) included. Both routes classify a "
            "failure with the classifier the OpenAI surfaces use and answer it, "
            "sanitized, at that status with the Anthropic error type the status "
            "stands for: request validation is an invalid_request_error HTTP 400, "
            "and only a failure the classifier cannot lay at the client's door is a "
            "logged server error; the categories must not be merged."
        ),
        removal_condition=(
            "Remove when upstream classifies messages and count_tokens failures as its "
            "OpenAI surfaces do and answers them with the Anthropic error type for "
            "their status."
        ),
        validate_before=_validate_anthropic_400_before,
        validate_after=_validate_anthropic_400_after,
    ),
    "tool-truncation-finish-reason": SemanticContract(
        rationale=(
            "A parser may recognize a partial tool prefix at max_tokens. Neither Chat nor "
            "Responses may promote that prefix to an executable terminal: length/incomplete "
            "must survive streaming and batch paths, and Responses must emit neither "
            "arguments.done for the call the limit cut nor response.completed. Both "
            "Responses transports mark items as the stream closes them: the item the "
            "limit cut, the last, is incomplete; items the model finished before it "
            "are completed. A finished parse releases every held-back text, in the "
            "order the model produced it."
        ),
        removal_condition=(
            "Remove when upstream preserves engine truncation across Chat and Responses "
            "stream/batch parsing, marks the item the limit cut incomplete on both "
            "transports, and exposes no execution boundary for it under controlled "
            "token cuts."
        ),
        validate_before=_validate_truncation_before,
        validate_after=_validate_truncation_after,
    ),
    "qwen38-vision-runtime": SemanticContract(
        rationale=(
            "The sole deployment requires chronological tool-result media, canonical "
            "lossless/static PNG inputs, the released full pixel budget, BF16 vision, and "
            "enough phase-local VRAM without reducing text context. Strict validation must "
            "occur before I/O, and the image contract is the served image's own: it holds "
            "for every caller and every launch, so no setting of how the image is started "
            "serves JPEG, a remote URL or a low-detail image. Mutually exclusive "
            "encoder/text workspaces must release and restore even on failure."
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
    "generation-requires-agent-id": SemanticContract(
        rationale=(
            "A generation names the line of work it belongs to with an opaque "
            "agent ID, kv_scope: the same ID for every request that continues a "
            "conversation, a new, never-used ID for a fork or a subagent, with no "
            "lineage or format declared. Generation request models require it and "
            "the served schema says so; the render models they share their shape "
            "with allocate no KV and have no ID; a batch names one ID per "
            "conversation. One line of work generates one sequence per request, so "
            "n > 1, a batch best_of > 1, or several prompts under one ID are "
            "refused by name; /invocations dispatches by the same generation "
            "types, so a body that names its ID reaches the engine with it. The "
            "engine request type holds the same rule: no "
            "generation can be built, or decoded by the engine, without the ID, so "
            "it binds every producer -- the input processor, a caller that builds "
            "an engine request itself, and the engine's own notice that a "
            "KV-transfer request was refused before admission, which names the "
            "refused request's ID. A request refused as the engine decodes it "
            "(one changed after it was built) is that request's error, returned "
            "to its client; it used to end the engine's input thread, after which "
            "the engine received nothing. The ID is the key agent-grouped-offload-retention "
            "groups what the KV tiers retain by, and that tier refuses a request "
            "without one; a notice without one ended the engine core for every user."
        ),
        removal_condition=(
            "Remove when pinned upstream requires an agent ID naming one line of "
            "work on every generation surface and at its engine boundary, with one "
            "sequence per ID."
        ),
        validate_before=_validate_agent_id_before,
        validate_after=_validate_agent_id_after,
    ),
    "attention-growth-keeps-prefix-hash": SemanticContract(
        rationale=(
            "An attention block only appends: when a block cached under a partial "
            "prefix grows to a longer one, the data its earlier prefix names is "
            "unchanged. Upstream removed the earlier prefix-cache hash when the "
            "longer one was registered, so a later request ending at the earlier "
            "point recomputed data the block still held. The block keeps every hash "
            "it has carried until it is evicted, when all are released together. A "
            "recurrent-state block is different -- its state at the longer prefix "
            "replaces the earlier checkpoint -- so its earlier hash is removed. "
            "Every caller states which kind of block it caches."
        ),
        removal_condition=(
            "Remove when pinned upstream keeps a grown attention block reachable "
            "under each shorter prefix it holds."
        ),
        validate_before=_validate_attention_prefix_hash_before,
        validate_after=_validate_attention_prefix_hash_after,
    ),
    "grouped-kv-specs-use-layer-geometry": SemanticContract(
        rationale=(
            "A KV cache group whose layers share one attention type but differ in "
            "shape carries a UniformTypeKVCacheSpecs wrapper, which is neither an "
            "attention nor a Mamba spec. Upstream classified groups by that wrapper: "
            "block-size resolution and the offloading boundary missed the "
            "decode-context-parallel factor of a grouped attention group and the "
            "cache mode of a grouped Mamba group, and the offload scheduler's window "
            "classification asserted a full-attention spec and stopped the engine. "
            "Every classification resolves a grouped spec to its layers' common "
            "geometry first, so grouped and ungrouped descriptions of the same "
            "layers yield the same block sizes, windows and recurrent-state "
            "handling; the offloading boundary records each group's resolved Mamba "
            "cache mode, which the scheduler reads for alignment and copy-on-write."
        ),
        removal_condition=(
            "Remove when pinned upstream classifies UniformTypeKVCacheSpecs groups "
            "by their layers' spec in block-size resolution and KV offloading."
        ),
        validate_before=_validate_grouped_geometry_before,
        validate_after=_validate_grouped_geometry_after,
    ),
    "agent-grouped-offload-retention": SemanticContract(
        rationale=(
            "The CPU offload tier keeps, for each agent ID, the complete context its "
            "next turn reads, and knows what every chunk holds. A request publishes "
            "one working set across all KV groups -- the full-attention prefix, the "
            "trailing window or recurrent state each group resumes from, and any "
            "partial tail -- and only chunks a working set retains are stored. A "
            "finished request's context is retained for the agent's next turn when "
            "every chunk holds the data it needs, replacing the agent's previous "
            "one. Capacity pressure reclaims unreferenced old windows first, then "
            "releases whole agents in least-recently-used order, never the storing "
            "one, while references held by surviving contexts protect shared chunks; "
            "retained contexts are bounded by the chunk count. A coalesced chunk "
            "advertises only the data actually written to it: lookup matches the "
            "data a resume point requires, a fill completes a sparse chunk in place, "
            "and secondary storage receives only complete canonical entries. Prefix "
            "lookup matches content and cache_salt alone in every tier, as upstream "
            "does; the ID groups retention and never restricts a match. The engine "
            "request carries the ID to the offload connector. Upstream's pluggable "
            "block-eviction policy sees request context only when blocks are "
            "touched, never on insert, eviction or request finish, so it cannot "
            "release whole agents; eviction is the manager's own, no policy or "
            "policy-selection key exists, and stores are not filtered by reuse "
            "count. A pooling request carries no ID, so a spec whose manager "
            "accounts per agent is refused a pooling model where the connector is "
            "built; its manager raised on the first one, in the engine core."
        ),
        removal_condition=(
            "Remove when pinned upstream retains complete per-agent contexts, "
            "releases whole agents under pressure, and tracks the data each offload "
            "chunk holds, across all generation consumers."
        ),
        validate_before=_validate_agent_retention_before,
        validate_after=_validate_agent_retention_after,
    ),
    "agentless-generation-routes-unmounted": SemanticContract(
        rationale=(
            "Every mounted route that reaches the generative engine is an identity "
            "surface. /generative_scoring builds its sampling parameters without an "
            "agent ID, and the Cohere chat endpoint converts each request into a "
            "render-shaped chat request without one, so every generation either "
            "makes is refused at the engine boundary for naming no line of work; a "
            "mounted route whose only answer is that refusal is a half-present "
            "surface. Upstream mounts the Cohere endpoint only when "
            "VLLM_ENABLE_COHERE_API=1 and the optional cohere SDK imports; neither "
            "route is registered here, no handler is built, and the Cohere-only "
            "cohere_is_reasoning_model flag does not exist. The Cohere prompt "
            "format, a tokenizer-mode feature, stays."
        ),
        removal_condition=(
            "Remove when pinned upstream's /generative_scoring and Cohere chat "
            "requests name an agent ID, or those routes stop reaching the "
            "generative engine."
        ),
        validate_before=_validate_agentless_routes_before,
        validate_after=_validate_agentless_routes_after,
    ),
    "kv-capacity-in-declared-users": SemanticContract(
        rationale=(
            "KV capacity is declared as a count of resident full-length user "
            "contexts -- --kv-cache-users for the GPU pool, cpu_kv_cache_users for "
            "the CPU offload tier -- and bytes are derived where page sizes and the "
            "group structure exist, so one declaration is correct on any device and "
            "parallel layout. The GPU pool holds exactly the declared contexts plus "
            "the null block: surplus memory stays unclaimed, a shortfall refuses at "
            "startup, and auto-fit searches the longest length whose declared "
            "contexts fit. The CPU tier holds the declared contexts in chunks, "
            "full-attention groups at full length and windowed or recurrent groups "
            "at the trailing window a re-entry reads, which retention makes their "
            "whole footprint; the window classification is derived once at the "
            "offloading boundary because load planning and sizing must agree on "
            "it. The other inputs that sized these two tiers are a second, byte- or "
            "block-denominated mode that encodes one machine: "
            "--kv-cache-memory-bytes sized the GPU "
            "pool in bytes; --kv-offloading-size sized the CPU tier in GiB by "
            "writing a cpu_bytes_to_use entry; --kv-offloading-backend only chose "
            "which connector that size configured and did nothing without it; "
            "VLLM_USE_SIMPLE_KV_OFFLOAD only switched that same written "
            "configuration to the byte-sized SimpleCPUOffloadConnector. None of "
            "them exists, the CPU spec does not read cpu_bytes_to_use, "
            "num_gpu_blocks_override "
            "beside a declaration is refused, and a kv_connector_extra_config key "
            "the CPU spec does not read refuses at startup instead of being "
            "ignored."
        ),
        removal_condition=(
            "Remove when pinned upstream sizes both KV tiers from a declared count "
            "of full-length contexts and accepts no byte- or block-denominated "
            "sizing input."
        ),
        validate_before=_validate_declared_capacity_before,
        validate_after=_validate_declared_capacity_after,
    ),
    "kv-declaration-within-physical-bound": SemanticContract(
        rationale=(
            "The declared GPU capacity is checked against what the card can "
            "physically hold -- its memory minus every measured resident: weights, "
            "non-torch allocations, the recurring activation peak and the opted-in "
            "CUDA-graph charge -- not against profiling's estimate, which also "
            "withholds the (1 - gpu_memory_utilization) reserve. A declaration the "
            "card holds proceeds when the estimate would prefer less; one beyond "
            "the physical bound refuses at startup naming the bound and what frees "
            "it. The estimate stays informational, and the startup plan persists "
            "the physical bound rather than the estimate."
        ),
        removal_condition=(
            "Remove when pinned upstream checks a declared KV capacity against "
            "physical device capacity and keeps the utilization holdback an "
            "estimate only."
        ),
        validate_before=_validate_physical_bound_before,
        validate_after=_validate_physical_bound_after,
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
            "to drop -- and on Responses, which reads the same count through "
            "the same function, per turn, instead of a second count of its own. "
            "The whole-generation count the reasoning-parser interface exposes "
            "is that same number, by the same scan: absent where usage is "
            "absent, never an error, and kept by no format or text-split parser "
            "beside it -- a parser that splits on text has no exact id count "
            "and reports none rather than a depth count usage declines. "
            "A grammar without one id-marked boundary, a parser fed text "
            "without its ids, or a parser that missed generated ids yields no "
            "count or a refusal, never an estimate; the field is absent, not "
            "zero, when no exact split was made."
        ),
        removal_condition=(
            "Remove only when pinned upstream serves completion_tokens_details."
            "reasoning_tokens on Chat Completions from the parser's own token-id "
            "boundary, with the prompt-side opener and the implicit tool-call "
            "end both counted correctly, on the streaming and batch paths, and "
            "its reasoning-parser count is that number or absent."
        ),
        validate_before=_validate_reasoning_usage_before,
        validate_after=_validate_reasoning_usage_after,
    ),
    "generated-tokens-survive-parsing": SemanticContract(
        rationale=(
            "The parser engine deleted every special token that is not one of "
            "its format's terminals, and Qwen's grammar deleted a second "
            "</think>, in reasoning, in the answer and inside a call's "
            "arguments: a file write lost the vision markers the served template "
            "itself spells. With the content ids forwarded beside the text, the "
            "deleted token's id remained, so a whole response was a 500 and a "
            "stream lost the token or failed by how its deltas were grouped. A "
            "token the format does not act on is the text it decodes to, in "
            "every state, which is how the grammar matches it; a stop token adds "
            "nothing only because the detokenizer gives it no text. A grammar "
            "that forwards content ids is refused at construction if it acts on "
            "a terminal in content outside its tool language."
        ),
        removal_condition=(
            "Remove when upstream's parser engine deletes no generated token and "
            "its Qwen grammar reads a second </think> as content."
        ),
        validate_before=_validate_generated_tokens_before,
        validate_after=_validate_generated_tokens_after,
    ),
    "include-reasoning-shapes-the-response": SemanticContract(
        rationale=(
            "Chat Completions read include_reasoning=false as reasoning having "
            "already ended, which started the tool or output grammar at the "
            "first generated token, inside the reasoning the template opens; "
            "under auto tools the grammar excludes </think>, so the model could "
            "only call a tool, stop or run to its limit, and the answer came back "
            "empty. Responses documents the same field as hiding reasoning "
            "without affecting inference. Whether reasoning has ended is read "
            "from the prompt the model continues; include_reasoning only shapes "
            "the response."
        ),
        removal_condition=(
            "Remove when upstream's chat route no longer derives the reasoning "
            "state from include_reasoning."
        ),
        validate_before=_validate_include_reasoning_before,
        validate_after=_validate_include_reasoning_after,
    ),
    "unspecified-tool-choice-is-the-default": SemanticContract(
        rationale=(
            "An explicit null tool_choice with tools armed no grammar, which "
            "needs auto, required or a named choice, while the parser read null "
            "as auto: a stream published a call no grammar constrained, a whole "
            "response deleted the call and its text, and parallel_tool_calls "
            "false was held by nothing. A choice that is not specified, omitted "
            "or null, is auto with tools and none without, decided once in the "
            "chat request and in the Anthropic conversion, so null never reaches "
            "the parser and one value decides both whether the call grammar is "
            "armed and whether calls are parsed."
        ),
        removal_condition=(
            "Remove when upstream defaults a null tool_choice exactly as an "
            "omitted one."
        ),
        validate_before=_validate_unspecified_tool_choice_before,
        validate_after=_validate_unspecified_tool_choice_after,
    ),
    "call-only-answer-keeps-the-blank-line": SemanticContract(
        rationale=(
            "Under tool_choice required or a named function the Qwen grammar "
            "began with <tool_call>, because XGrammar writes the template's "
            "blank line after </think> only inside a reasoning prefix vLLM never "
            "builds: the reasoning parser starts the grammar after reasoning "
            "ends. A model trained on </think>, a blank line, then the call was "
            "masked from that line on every forced or required call. The answer "
            "may now begin with it, and nothing else may precede the call; it "
            "stays optional because a <tool_call> that ends reasoning itself, "
            "or a model that does not reason, starts the grammar at the call."
        ),
        removal_condition=(
            "Remove when upstream's Qwen builder admits the template's blank "
            "line before a forced or required call."
        ),
        validate_before=_validate_call_only_separator_before,
        validate_after=_validate_call_only_separator_after,
    ),
    "responses-tools-are-one-function-list": SemanticContract(
        rationale=(
            "Responses offers a namespace's functions to the model under flat "
            "names (namespace__name), and the parser resolves those names, but "
            "the tool grammar was given the tools unflattened: XGrammar classed "
            "a namespace as a builtin tool and the Qwen builder drops builtins, "
            "so a call to a namespace function was masked at its first name "
            "token, and with only namespace tools the grammar admitted any text, "
            "an un-offered call included. The grammar now reads the one list "
            "the template does. A named choice must use a name the model is "
            "offered (the bare member name was accepted and the grammar then "
            "raised a 500); allowed_tools arms the grammar to its listed subset "
            "and mode instead of arming and parsing nothing while the tools "
            "stay in the prompt; a choice no grammar enforces (hosted, MCP, "
            "custom) is refused naming tool_choice; and required with no "
            "function to call is a 400, where it was a KeyError 500. A choice "
            "that lets the model call is refused while no tool parser parses "
            "the calls, by the refusal chat gives: Responses rendered it and "
            "returned the calls as text."
        ),
        removal_condition=(
            "Remove when pinned upstream gives the tool grammar the Responses "
            "functions under the names the prompt offers, and refuses every "
            "tool_choice it cannot enforce."
        ),
        validate_before=_validate_responses_function_list_before,
        validate_after=_validate_responses_function_list_after,
    ),
    "batch-parse-starts-where-the-prompt-leaves": SemanticContract(
        rationale=(
            "The stream decided from the prompt whether reasoning had already "
            "ended; the complete-output parse never saw the prompt and always "
            "began in reasoning. A final message the caller asks to continue -- "
            "Responses turns this on by itself for an in_progress or incomplete "
            "last item, chat with continue_final_message -- ends its prompt after "
            "</think>, so the batch answer was filed as reasoning (and hidden with "
            "include_reasoning off) and counted as reasoning tokens, while every "
            "chunking of the stream read it as content. Upstream has the same "
            "asymmetry. One decision now reads the prompt for both paths; "
            "parse_output takes the prompt ids, every complete-output caller "
            "passes them, and a grammar's prompt state survives the batch reset "
            "(gemma4 and inkling seeded only the stream) and is where the "
            "whole-generation reasoning count starts, so that count is the one "
            "the feed takes. The same reading decides who writes Qwen's opener: "
            "its grammar deleted every <think> in reasoning, the model's own "
            "included, because it could not tell an opener from text. A prompt "
            "ending inside <think>, as every served generation prompt does, "
            "opened reasoning, so a <think> the model writes is reasoning text; "
            "elsewhere only a generation's first token can be the opener, which "
            "a template that leaves the opener to the model expects. Derender "
            "is given the request /render rendered but not the prompt ids, and "
            "parsed as if reasoning were open and its opener left to the "
            "model: a continued final message was reasoning, and a <think> the "
            "model wrote as its first token lost its text, though the prompt "
            "had opened reasoning. It renders that prompt again now, with the "
            "renderer and settings /render renders with (0.12 to 0.17 s for a "
            "210,035-token history and 0.7 s with a 4096x4096 PNG, measured on "
            "CPU), "
            "refuses a request that does not render as /render refuses it, and "
            "parses from the prompt as chat does: a second render on that route "
            "rather than a token of the model's text lost, and no new protocol."
        ),
        removal_condition=(
            "Remove when pinned upstream's complete-output parse starts from the "
            "state the prompt leaves, as its streaming parse does."
        ),
        validate_before=_validate_batch_parse_from_prompt_before,
        validate_after=_validate_batch_parse_from_prompt_after,
    ),
    "derender-text-is-the-detokenizers": SemanticContract(
        rationale=(
            "The serving detokenizer gives the stop token a generation ended on "
            "no text unless the caller asked to see stop text; derender decoded "
            "every id itself, so the model's <|im_end|> (and a caller's stop "
            "token id) reached content on the derender route, in batch and in "
            "the stream, chat and completion alike. The parser engine's "
            "special-token deletion had hidden the end-of-turn text there until "
            "generated-tokens-survive-parsing removed it. One function now "
            "decides which generated ids carry text, and both the detokenizer "
            "and every derender path call it; the parser still receives every "
            "id. Derender's batch decode is its stream's incremental decode, so "
            "a length cut inside a character holds the partial bytes back, as "
            "serving does, instead of failing the scanner with a 500. The "
            "detokenizer is told by the engine that stopped that the last id is "
            "a stop token; derender is told by its caller, and holds the caller "
            "to it: a stop with no stop_reason ends on one of the model's EOS "
            "ids, which one function reads from the generation config and the "
            "tokenizer for the engine and derender alike, and a stop on a stop "
            "token id ends on that id. Any other is a 400 naming the choice, "
            "and no id loses its text. Derender does not cut the text at a "
            "matched stop string, as upstream does not."
        ),
        removal_condition=(
            "Remove when pinned upstream's derender gives a generation's stop "
            "token no text unless stop text is requested, as its detokenizer does, "
            "and refuses a stop the choice's ids do not end on."
        ),
        validate_before=_validate_derender_stop_text_before,
        validate_after=_validate_derender_stop_text_after,
    ),
    "output-constraints-refused-beside-tool-calls": SemanticContract(
        rationale=(
            "Whenever a request could call a tool, the parser armed the tool "
            "grammar over the whole output and discarded the caller's output "
            "constraint -- response_format and structured_outputs on chat, "
            "text.format and structured_outputs on Responses, and Anthropic's "
            "output_config.format, which is converted to response_format after "
            "the chat request is built -- and the Responses echo of text came "
            "back null. Upstream replaced it under required, named or strict "
            "auto; this fork arms every Qwen auto request, so it replaced it "
            "always. A composed grammar has no shape the model was trained on "
            "and no upstream counterpart, so the combination is refused with a "
            "400 naming the constraint as the surface spells it, and the next "
            "actions that exist: tool_choice none keeps the constraint, or a "
            "forced call of a function whose parameters are the schema. An "
            "Anthropic output format without a schema, which was ignored, is "
            "refused too. Upstream's partial refusal is subsumed."
        ),
        removal_condition=(
            "Remove when pinned upstream refuses, or composes, an output "
            "constraint beside a callable tool on every surface instead of "
            "replacing it."
        ),
        validate_before=_validate_output_constraint_beside_tools_before,
        validate_after=_validate_output_constraint_beside_tools_after,
    ),
    "batch-invariance-substitutes-no-nvfp4-kernel": SemanticContract(
        rationale=(
            "With VLLM_BATCH_INVARIANT set and the batch-invariant CUTLASS "
            "kernel unsupported -- any GPU without native FP4 -- selection "
            "forced the emulation kernel, logged it at INFO and returned before "
            "the refusal that keeps a substitute from serving W4A4 layers. The "
            "refusal held for this deployment only because its launch does not "
            "set the variable. Emulation under batch invariance is now a "
            "substitute like any other: served only when --linear-backend names "
            "it, and otherwise refused with the CUTLASS reason and the next "
            "actions that exist. The refusal's headline no longer says the "
            "device has no native kernel when one was only disabled by "
            "VLLM_DISABLED_KERNELS; it names that variable as the next action."
        ),
        removal_condition=(
            "Remove when pinned upstream refuses to substitute a non-FP4 kernel "
            "for NVFP4 W4A4 layers under VLLM_BATCH_INVARIANT unless it is named."
        ),
        validate_before=_validate_batch_invariant_native_fp4_before,
        validate_after=_validate_batch_invariant_native_fp4_after,
    ),
    "render-carries-every-image-chat-renders": SemanticContract(
        rationale=(
            "Chat renders an image_url and an input_image part alike, but the "
            "render route carried only image_url parts to generation: an "
            "input_image rendered its image pad tokens and arrived with no "
            "image, so generate returned 200 and the model read bare pad "
            "tokens as text. A mix of the two was refused naming the caller's "
            "token_ids, and shapes chat accepts -- an extra key, a null uuid, a "
            "plain-string item -- made render fail with a 500. One decision of "
            "what an image part is (chat's own per-part parse) now feeds both, "
            "and render carries exactly the images chat rendered, in order, in "
            "the transport's one image shape. input_image.detail is required, "
            "as upstream's type declares, and refused naming detail where its "
            "absence was a 500; the generate text path refuses an image span "
            "that arrives without its image. Every processor decides that by "
            "one rule: the added-vocabulary ids it embeds media at, read from "
            "its own output for one dummy item of each modality it accepts, "
            "stand only inside a supplied item's span -- the processor's "
            "per-item span search alone found nothing in ids that supply no "
            "media, and nothing past the last supplied item."
        ),
        removal_condition=(
            "Remove when pinned upstream's render route carries every image "
            "part chat renders, read by chat's own part parser."
        ),
        validate_before=_validate_render_every_image_before,
        validate_after=_validate_render_every_image_after,
    ),
    "rendered-prompts-are-never-truncated": SemanticContract(
        rationale=(
            "Responses truncation \"auto\" set truncate_prompt_tokens to the "
            "room max_output_tokens left, and the served tokenizer cuts from "
            "the left, so an overflowing prompt lost its head: the template's "
            "markers, the system text, the operator's task and the tool "
            "definitions, with nothing said. An explicit null truncated as "
            "well and then failed at response construction, and the mapping was "
            "decided three times. Chat cut the same head through "
            "truncate_prompt_tokens and truncation_side. A rendered prompt cut "
            "from the left starts mid-message, a shape no template writes. Both "
            "surfaces refuse the parameter by name and tokenize without a "
            "truncation; Completions keeps truncate_prompt_tokens, whose prompt "
            "is the caller's own text with no template around it."
        ),
        removal_condition=(
            "Remove when pinned upstream truncates a rendered chat or Responses "
            "prompt only by whole input items, never by a token cut from the "
            "head, or refuses to."
        ),
        validate_before=_validate_rendered_prompt_truncation_before,
        validate_after=_validate_rendered_prompt_truncation_after,
    ),
    "kv-transfer-params-are-declared": SemanticContract(
        rationale=(
            "A request's kv_transfer_params are parameters of the configured KV "
            "connector, and no connector said which it takes, so a key none reads "
            "was accepted and never read: the caller believed a parameter acted. "
            "Here the CPU tier acts on max_offload_tokens alone; kv_load_tiers "
            "selects among secondary tiers this spec has none of; do_remote_prefill "
            "is NIXL's, yet the frontend's rejection notice acted on it whatever "
            "connector was configured. A string sent as kv_transfer_params through "
            "vllm_xargs reached the connector, whose first read of it raised in the "
            "engine core for every user. Each connector declares the keys its "
            "protocol defines, as configured: those it reads and those it returns "
            "for a peer, which a proxy hands on as the peer's own (NIXL's "
            "transfer_mode). MultiConnector takes what any child takes; the offload "
            "connector adds what its spec and tiers act on; a connector that "
            "declares none takes none -- an out-of-tree one, and LMCache's and "
            "FlexKV's delegated adapters, whose packages read what nothing here "
            "establishes, each naming what its operator does to declare them. Each "
            "declared key carries the shape of the value its connector reads -- "
            "inside the engine core, where the P2P tier's .get on a string, or a "
            "list hashed as an ID, raised for every request the engine served. "
            "The same declaration states the keys a value is read with and those "
            "it is never read beside: NIXL's and Mooncake's decode node record a "
            "remote prefill by the keys naming the remote request -- for an "
            "aborted request too, whose notice then raised KeyError in the engine "
            "core -- and skipped a transfer missing them; the P2P tier dropped a "
            "source beside remote_prefiller and ran without a peer object's "
            "missing fields. Null and false ask for nothing, so upstream's "
            "proxies, which fill the keys a prefill node does not use with null, "
            "are admitted. The hidden-states connector takes a request's path "
            "only where its operator allows custom save paths. "
            "Admission refuses any other key, a value of another shape, a value "
            "without its companions or beside a key it excludes, and a "
            "non-object, with a 400 naming the key, the connector and what it "
            "takes, the shape or the other keys, and a next action the caller can "
            "take; upstream's "
            "reused prompt ids, which no layer here takes, are refused for what "
            "this server does instead. Only a connector that takes "
            "do_remote_prefill is told of a refused request's remote-prefill "
            "blocks, and only by parameters admission takes, since the notice "
            "reaches the connector without crossing admission. A shape holds a "
            "value to what its connector can act on, not only its JSON type: a "
            "P2P peer's request ID and host must be non-empty and its port from "
            "1 to 65535 (the parsers read an empty one as no peer and the request "
            "ran without its transfer; the transport raised on a negative port "
            "or a host its address grammar refuses, inside the engine core); a "
            "NIXL block-ID list is one list per KV cache group, never empty "
            "(the pull scheduler counted the prompt as remote and asserted on "
            "nothing to receive); and a Mooncake node is refused the remote "
            "action its role never takes -- a kv_producer pulling a remote "
            "prefill, a kv_consumer serving a remote decode -- which its "
            "scheduler asserts against inside the engine core."
        ),
        removal_condition=(
            "Remove when pinned upstream has each KV connector declare the request "
            "kv_transfer_params keys it takes and refuses a key no configured "
            "connector takes."
        ),
        validate_before=_validate_kv_transfer_params_keys_before,
        validate_after=_validate_kv_transfer_params_keys_after,
    ),
    "responses-refuses-tools-the-template-is-never-given": SemanticContract(
        rationale=(
            "On the chat-template path a Responses request's tools reach the "
            "model only as the functions construct_tool_dicts gives the "
            "template, on both contexts. A hosted, MCP or custom tool, a custom "
            "tool inside a namespace, and an empty namespace were left out of "
            "the prompt silently; with only such tools under auto the Qwen "
            "grammar admitted any text, and an empty namespace under required "
            "reached XGrammar's normalize_tool_choice, whose bare ValueError was "
            "a 500. A tool server does not change that: ParsableContext only "
            "runs a call the model was never told it could make, which upstream's "
            "own test marks xfail. The template's tool list now refuses each "
            "such tool with a 400 naming tools[i] or tools[i].tools[j] and its "
            "kind, under every tool choice and before anything is rendered. "
            "Harmony, which describes a tool server's browser and python to the "
            "model, never builds this list, and the request model still accepts "
            "a hosted tool for it."
        ),
        removal_condition=(
            "Remove when pinned upstream gives the chat template every declared "
            "Responses tool, or refuses a tool it does not give it."
        ),
        validate_before=_validate_responses_tools_never_given_before,
        validate_after=_validate_responses_tools_never_given_after,
    ),
    "chat-stream-carries-every-token-logprob": SemanticContract(
        rationale=(
            "A chat choice's log probabilities are those of every token it "
            "generated -- the end of reasoning, the calls and the end of turn "
            "included -- and the full response reports one per generated id. "
            "The stream skipped a step whose delta the parser left empty (a "
            "reasoning end, a held marker) unless token ids were requested, so "
            "that step's log probabilities never reached the caller and the "
            "streamed list depended on how the engine grouped tokens. The step "
            "is now sent whenever it carries log probabilities or token ids, "
            "and the Poolside parser no longer emits an empty content delta to "
            "keep them."
        ),
        removal_condition=(
            "Remove when pinned upstream sends every generated token's log "
            "probability on the chat stream whatever the parser releases."
        ),
        validate_before=_validate_chat_stream_logprobs_before,
        validate_after=_validate_chat_stream_logprobs_after,
    ),
    "chat-messages-read-by-one-rule": SemanticContract(
        rationale=(
            "Chat Completions renamed a message's legacy reasoning_content to "
            "reasoning before validation, and the message parser reads only "
            "reasoning, while /tokenize's chat form and the pooling chat forms "
            "kept the legacy name: /tokenize rendered a history whose past "
            "turns carry reasoning_content without that reasoning, so its count "
            "omitted tokens chat renders. One function now reads a caller's "
            "chat messages for every request type that carries them. The "
            "tool-history gate stays chat's: a client counts a history whose "
            "last calls have no results yet, and /tokenize counts it."
        ),
        removal_condition=(
            "Remove when pinned upstream normalizes reasoning_content on every "
            "request type that carries chat messages."
        ),
        validate_before=_validate_messages_one_rule_before,
        validate_after=_validate_messages_one_rule_after,
    ),
    "responses-refuses-what-it-cannot-honour": SemanticContract(
        rationale=(
            "previous_response_id on a server that stores no responses was "
            "looked up in an empty store and answered 404 'Response with id "
            "not found' naming response_id: the wrong field, the wrong cause "
            "and no next action. It is refused at intake with a 400 naming "
            "previous_response_id, the store being off, and what to send "
            "instead; a lookup that misses names the field the id came in, and "
            "retrieving on a server that stores nothing says so. The prompt "
            "template refusal names why and what to send instead. "
            "max_tool_calls counts built-in tool calls: the template path runs "
            "none, so it holds there; a server that runs a requested built-in "
            "tool itself, uncounted, refuses it before admission. Every other "
            "Responses parameter was accepted whether or not it was applied: "
            "the API parameters the request model does not declare (as "
            "conversation, moderation, context_management) passed as extra "
            "keys, and reasoning.summary, text.verbosity, a processing tier, "
            "stream obfuscation, encrypted reasoning, a store sent true without "
            "a store, Harmony history without Harmony, and a code interpreter's "
            "outputs were served without what they asked. Each is refused, "
            "naming it, its cause and what to send instead; the undeclared "
            "parameters are read from the pinned client types, so none can be "
            "added unrefused, and a value that asks only for what the server "
            "does anyway is served."
        ),
        removal_condition=(
            "Remove when pinned upstream refuses previous_response_id without "
            "a store by its field and cause, applies or refuses max_tool_calls "
            "where it runs built-in tools, and refuses every Responses "
            "parameter it does not apply."
        ),
        validate_before=_validate_responses_unhonoured_fields_before,
        validate_after=_validate_responses_unhonoured_fields_after,
    ),
    "reviewed-tests-declare-what-they-need": SemanticContract(
        rationale=(
            "The test files the patch set changes or adds were hashed and "
            "executed by nothing, so commits broke them unnoticed. A runner "
            "executes them now, and where a test runs is the test's own "
            "statement: tests/conftest.py registers a gpu and a network mark, "
            "and a test that needs a GPU or the network -- the Hugging Face hub "
            "or a remote URL -- carries the mark on itself, its class, or its "
            "module's pytestmark, which is read without importing the module, "
            "so a module that cannot even be imported without a GPU says so the "
            "same way. Every test that carries neither runs with no GPU and no "
            "network, and fails there if it needs one unsaid. Upstream's tests "
            "of image fetching the Qwen3.8 image contract refuses are strict "
            "xfails naming the contract and the exception it raises, as is each "
            "non-PNG case of the base64 round trip, whose reference images are "
            "read from the asset store rather than fetched, so its PNG cases -- "
            "the form the contract admits -- run and must pass."
        ),
        removal_condition=(
            "Remove when pinned upstream's tests declare what they need of the "
            "machine in a form a runner reads before importing them."
        ),
        validate_before=_validate_test_declarations_before,
        validate_after=_validate_test_declarations_after,
    ),
    "priority-is-refused-where-nothing-orders-by-it": SemanticContract(
        rationale=(
            "Every request model's priority field says a priority other than 0 "
            "raises an error unless the server schedules by priority, and none "
            "did: under the default first-come-first-served scheduler the "
            "priority was carried to the engine and never read, so a caller was "
            "told an order applied that nothing applied. Admission, which every "
            "route that reaches the engine passes (chat, completions, Responses "
            "and its built-in tool turns, generate, pooling, and the offline "
            "LLM), refuses it there, before the request reaches the engine, "
            "naming priority, the policy, and both next actions; 0, which asks "
            "for nothing, and any priority under priority scheduling are "
            "admitted. A route that reads the priority from its "
            "X-Vllm-Priority header served a header that was not an integer as "
            "if it had not been sent, and refuses it."
        ),
        removal_condition=(
            "Remove when pinned upstream refuses a priority its scheduler does "
            "not apply, as its request models already state."
        ),
        validate_before=_validate_priority_refusal_before,
        validate_after=_validate_priority_refusal_after,
    ),
}
