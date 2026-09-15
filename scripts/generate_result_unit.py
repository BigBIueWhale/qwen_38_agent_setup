#!/usr/bin/env python3
"""CPU-only proof that installed token generation preserves terminal evidence."""
import asyncio
import json

from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind


def output(index, reason, *, finished):
    return RequestOutput(
        request_id="installed-unit", prompt=None, prompt_token_ids=[1, 2],
        prompt_logprobs=None, finished=finished,
        outputs=[CompletionOutput(
            index=index, text="", token_ids=[], cumulative_logprob=None,
            logprobs=None, finish_reason=reason,
            stop_reason="END" if reason == "stop" else None,
        )],
    )


async def check():
    serving = object.__new__(ServingTokens)
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    for stream in (False, True):
        request = GenerateRequest(token_ids=[1, 2], sampling_params={"n": 2},
                                  kv_scope="unit", stream=stream)
        params = request.to_sampling_params(100, {})
        assert params.output_kind == (
            RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY)
        assert "output_kind" not in request.model_dump_json()
        for complete in (False, True):
            async def results():
                yield output(1, "stop", finished=False)
                if complete:
                    yield output(0, "stop", finished=True)

            metadata = RequestResponseMetadata(request_id="installed-unit")
            arguments = (request, results(), "installed-unit", "model", metadata, params)
            if stream:
                events = [event async for event in serving.serve_tokens_stream_generator(*arguments)]
                if complete:
                    assert events[-1] == "data: [DONE]\n\n"
                    choices = [json.loads(event[6:])["choices"][0] for event in events[:-1]]
                    assert {choice["index"] for choice in choices} == {0, 1}
                    assert all(choice["stop_reason"] == "END" for choice in choices)
                else:
                    assert "data: [DONE]\n\n" not in events
                    assert json.loads(events[-1][6:])["error"]["code"] == 500
            else:
                response = await serving.serve_tokens_full_generator(*arguments)
                if complete:
                    assert [choice.index for choice in response.choices] == [0, 1]
                    assert all(choice.stop_reason == "END" for choice in response.choices)
                    assert all(choice.token_ids == [] for choice in response.choices)
                else:
                    assert isinstance(response, ErrorResponse) and response.error.code == 500
            assert (metadata.final_usage_info is not None) == complete
    print("Installed token-generation terminal contract passed")


asyncio.run(check())
