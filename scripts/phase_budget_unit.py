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

print("phase-budget-unit: PASS")
