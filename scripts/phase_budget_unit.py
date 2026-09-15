#!/usr/bin/env python3
"""CPU-only unit checks for Qwen's separate reasoning/final response caps."""

from __future__ import annotations

import msgspec

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus


def make_request(final_budget: int, max_tokens: int = 100) -> Request:
    params = SamplingParams(
        max_tokens=max_tokens,
        thinking_token_budget=20,
        final_response_token_budget=final_budget,
    )
    params._reasoning_end_token_sequences = [[90, 91], [91]]
    return Request("phase-budget-unit", [1, 2], params, None)


def append_and_check(request: Request, token: int, expected: bool) -> None:
    request.append_output_token_ids(token)
    actual = check_stop(request, 1_000)
    assert actual is expected, (token, expected, actual, request.output_token_ids)


request = make_request(3)
round_trip_params = msgspec.msgpack.decode(
    msgspec.msgpack.encode(request.sampling_params), type=SamplingParams
)
assert round_trip_params.final_response_token_budget == 3
assert round_trip_params._reasoning_end_token_sequences == [[90, 91], [91]]
for reasoning_token in (10, 11, 12, 90):
    append_and_check(request, reasoning_token, False)
append_and_check(request, 91, False)
assert request.final_response_start_index == 5
append_and_check(request, 20, False)
append_and_check(request, 21, False)
append_and_check(request, 22, True)
assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
assert request.stop_reason == "final_response_token_budget"

eos_at_cap = make_request(1)
assert eos_at_cap.sampling_params is not None
eos_at_cap.sampling_params.update_from_generation_config({}, eos_token_id=99)
append_and_check(eos_at_cap, 91, False)
append_and_check(eos_at_cap, 99, True)
assert eos_at_cap.status == RequestStatus.FINISHED_STOPPED
assert eos_at_cap.stop_reason is None

hard_cap_over_minimum = make_request(1)
assert hard_cap_over_minimum.sampling_params is not None
hard_cap_over_minimum.sampling_params.min_tokens = 100
append_and_check(hard_cap_over_minimum, 91, False)
append_and_check(hard_cap_over_minimum, 55, True)
assert hard_cap_over_minimum.stop_reason == "final_response_token_budget"

no_end = make_request(2, max_tokens=3)
append_and_check(no_end, 40, False)
append_and_check(no_end, 41, False)
append_and_check(no_end, 42, True)
assert no_end.stop_reason is None

defaults = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "thinking_token_budget": 262_144,
    "final_response_token_budget": 131_072,
}


def chat_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="qwen3.8",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=200_000,
        **kwargs,
    )


assert (
    chat_request().to_sampling_params(200_000, defaults).final_response_token_budget
    == 131_072
)
assert (
    chat_request(final_response_token_budget=200_000)
    .to_sampling_params(200_000, defaults)
    .final_response_token_budget
    == 131_072
)
assert (
    chat_request(final_response_token_budget=5)
    .to_sampling_params(200_000, defaults)
    .final_response_token_budget
    == 5
)
assert (
    chat_request(final_response_token_budget=None)
    .to_sampling_params(200_000, defaults)
    .final_response_token_budget
    == 131_072
)


def responses_request(**kwargs) -> ResponsesRequest:
    return ResponsesRequest(
        model="qwen3.8",
        input="test",
        max_output_tokens=200_000,
        **kwargs,
    )


responses_defaults = responses_request().to_sampling_params(200_000, defaults)
assert responses_defaults.thinking_token_budget == 262_144
assert responses_defaults.final_response_token_budget == 131_072
responses_explicit = responses_request(
    thinking_token_budget=128,
    final_response_token_budget=5,
).to_sampling_params(200_000, defaults)
assert responses_explicit.thinking_token_budget == 128
assert responses_explicit.final_response_token_budget == 5
assert (
    responses_request(final_response_token_budget=200_000)
    .to_sampling_params(200_000, defaults)
    .final_response_token_budget
    == 131_072
)

for invalid in (0, -2, True, 1.5):
    try:
        SamplingParams(final_response_token_budget=invalid)
    except VLLMValidationError:
        pass
    else:
        raise AssertionError(f"invalid final response budget accepted: {invalid!r}")

# Every generation protocol resolves the configured sampling policy before
# constructing the engine request. Render output preserves explicit values.
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest

completion = CompletionRequest(model="qwen3.8", prompt="test", kv_scope="agent")
assert completion.max_tokens is None
completion_params = completion.to_sampling_params(200_000, defaults)
assert completion_params.thinking_token_budget == 262_144
assert completion_params.final_response_token_budget == 131_072
assert completion_params.top_p == 0.95 and completion_params.top_k == 20

supplied = GenerateRequest.model_validate({
    "token_ids": [1, 2, 3], "kv_scope": "agent", "sampling_params": {"min_tokens": 64},
})
resolved = supplied.to_sampling_params(200_000, defaults)
assert resolved.max_tokens == 200_000 and resolved.min_tokens == 64
assert resolved.top_p == 0.95 and resolved.top_k == 20
assert resolved.final_response_token_budget == 131_072
assert supplied.sampling_params == {"min_tokens": 64}

rendered = GenerateRequest(
    token_ids=[1, 2, 3], kv_scope="agent", sampling_params=SamplingParams(max_tokens=16),
)
restored = GenerateRequest.model_validate_json(rendered.model_dump_json())
assert restored.sampling_params["max_tokens"] == 16
resolved = restored.to_sampling_params(16, defaults)
assert resolved.max_tokens == 16 and resolved.top_p == 1.0 and resolved.top_k == 0
assert resolved.final_response_token_budget == 131_072
assert resolved.extra_args["kv_scope"] == "agent"

policy = {**defaults, "presence_penalty": 0.3, "min_p": 0.01}
for request in (chat_request(), responses_request(), completion, supplied):
    params = request.to_sampling_params(200_000, policy)
    for key, value in policy.items():
        assert getattr(params, key) == value, (type(request).__name__, key)

for request_type, prompt in (
    (ChatCompletionRequest, {"messages": [{"role": "user", "content": "test"}]}),
    (CompletionRequest, {"prompt": "test"}),
):
    for stream in (False, True):
        try:
            request_type(model="qwen3.8", use_beam_search=True, stream=stream, **prompt)
        except VLLMValidationError as exc:
            assert exc.parameter == "use_beam_search"
            assert "Beam search is not supported" in str(exc)
        else:
            raise AssertionError("beam decoding bypassed the sampling policy boundary")

def check_one_way_thinking_boundary():
    from pathlib import Path
    import runpy
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    import torch
    import xgrammar as xgr

    from vllm.config import ReasoningConfig
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
    from vllm.v1.sample.logits_processor import BatchUpdate
    from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.sampling_params import StructuredOutputsParams

    helpers = runpy.run_path(
        str(Path(__file__).with_name("tool_output_parser_unit.py"))
    )
    markers, tool_schema = helpers["MARKERS"], helpers["TOOL"]
    call, decode, encode = (helpers[name] for name in ("call", "decode", "encode"))
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = markers
    tokenizer.decode.side_effect = decode
    tokenizer.encode.side_effect = lambda text, **_: encode(text)
    tokenizer.all_special_tokens = list(markers)
    tokenizer.all_special_ids = list(markers.values())
    config = ReasoningConfig(reasoning_parser="qwen3")
    with patch(
        "vllm.config.reasoning.cached_tokenizer_from_config", return_value=tokenizer
    ):
        config.initialize_token_ids(SimpleNamespace())
    reasoner_cls = ReasoningParserManager.get_reasoning_parser("qwen3")
    tools = ChatCompletionRequest(messages=[], tools=[tool_schema]).tools
    tag = get_model_structural_tag("qwen_3_coder", tools, "auto", False)
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo([])).compile_structural_tag(tag)
    start, end, tool = (markers[m] for m in ("<think>", "</think>", "<tool_call>"))
    assert config.reasoning_boundary_token_ids == frozenset({end, tool})
    value = "literal <think></think></tool_call> " + "x" * 100
    content = encode(call(value))

    with patch("vllm.utils.torch_utils.PIN_MEMORY", False):
        for implicit in (False, True):
            for resumed in (False, True):
                for chunk_size in (1, 1000):
                    prompt = [start, 10, end, 20, start]
                    output = [10] + ([] if implicit else [end]) + content
                    holder = ThinkingBudgetStateHolder(
                        config, 1, 0, torch.device("cpu"), False
                    )
                    holder.sync_batch(BatchUpdate(
                        batch_size=1, removed=(), moved=(),
                        added=[(0, SamplingParams(thinking_token_budget=8),
                                prompt, output if resumed else [])],
                    ))
                    prefixes = ([output] if resumed else [
                        output[:i] for i in range(
                            chunk_size, len(output) + chunk_size, chunk_size
                        )
                    ])
                    for prefix in prefixes:
                        holder.update_state([prefix], None)
                        logits = torch.zeros((1, max(markers.values()) + 1))
                        logits[:, end] = -float("inf")
                        before = logits.clone()
                        holder.apply_to_logits(logits, False, None)
                        torch.testing.assert_close(logits, before)

                    params = SamplingParams(structured_outputs=StructuredOutputsParams(
                        structural_tag=tag.model_dump_json()
                    ))
                    request = Request("thinking-boundary", prompt, params, None,
                                      reasoning_ended=False)
                    request.append_output_token_ids(output)
                    manager = StructuredOutputManager.__new__(StructuredOutputManager)
                    manager.reasoner_cls = reasoner_cls
                    manager.tokenizer = tokenizer
                    manager.enable_in_reasoning = False
                    assert manager.should_advance(request, new_token_ids=output)
                    trimmed = manager.trim_reasoning_for_advance(request, output)
                    assert trimmed == content
                    matcher = xgr.GrammarMatcher(compiled)
                    assert matcher.accept_string(decode(trimmed))
                    assert matcher.is_completed()


check_one_way_thinking_boundary()
print("phase-budget-unit: PASS")
