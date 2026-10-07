#!/usr/bin/env python3
"""Verify installed shared-prefix semantics using CPU metadata only."""

import asyncio
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    _create_req_context,
)
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionGenerationRequest
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind, SamplingParams, require_kv_scope
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM, InputStreamError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request


def request(agent, content, required=None, produced=None):
    return ReqContext(
        req_id=agent,
        kv_scope=agent,
        prefix_content=content,
        prefix_required=required or {},
        prefix_store=produced or {},
    )


def store(manager, context, keys):
    output = manager.prepare_store(keys, context)
    assert output is not None
    manager.complete_store(output.keys_to_store, context)
    return output


async def check_stream_identity():
    for next_agent in ('agent', 'other'):
        engine = MagicMock(spec=AsyncLLM)
        engine.model_config = SimpleNamespace(is_encoder_decoder=False)
        engine.get_supported_tasks = AsyncMock(return_value=('generate',))
        engine._validate_streaming_input_sampling_params = AsyncLLM._validate_streaming_input_sampling_params
        engine._add_request = AsyncMock()

        def process_inputs(request_id, prompt, params, resumable=False, **kwargs):
            require_kv_scope(params)
            return SimpleNamespace(
                request_id=request_id, external_req_id=None,
                prompt_token_ids=prompt['prompt_token_ids'], prompt_embeds=None,
                sampling_params=params.clone(), resumable=resumable,
            )

        engine.input_processor = SimpleNamespace(
            process_inputs=process_inputs, assign_request_id=InputProcessor.assign_request_id,
        )
        initial = SamplingParams(output_kind=RequestOutputKind.DELTA,
                                 extra_args={'kv_scope': 'agent'})
        following = SamplingParams(output_kind=RequestOutputKind.DELTA,
                                   extra_args={'kv_scope': next_agent})

        async def chunks():
            yield StreamingInput(prompt={'prompt_token_ids': [42]})
            yield StreamingInput(prompt={'prompt_token_ids': [43]}, sampling_params=following)

        queue = await AsyncLLM._add_streaming_input_request(engine, 'req', chunks(), initial)
        task = queue._input_stream_task
        assert task is not None
        await task
        requests = [call.args[0] for call in engine._add_request.call_args_list]
        if next_agent == 'agent':
            assert len(requests) == 3
            assert queue.get_nowait() is None
        else:
            assert len(requests) == 2
            try:
                queue.get_nowait()
            except InputStreamError as error:
                assert isinstance(error.cause, VLLMValidationError)
                assert error.cause.parameter == 'kv_scope'
            else:
                raise AssertionError('Input stream accepted a different agent ID')
        assert all(req.sampling_params.extra_args['kv_scope'] == 'agent' for req in requests)
        assert not requests[-1].resumable


async def check_rejection_notice_identity():
    """A refused remote-prefill request's notice reaches the CPU tier as a
    request of the refused request's own agent, and releases nothing it kept.
    Without the identity the tier raised, and an exception there ends the
    engine core for every user."""
    for extra_args in (None, {}, {'kv_scope': ' '}):
        try:
            EngineCoreRequest(
                request_id='r', prompt_token_ids=[0], mm_features=None,
                sampling_params=SamplingParams(max_tokens=1, extra_args=extra_args),
                pooling_params=None, arrival_time=0.0, lora_request=None,
                cache_salt=None, data_parallel_rank=None,
            )
        except VLLMValidationError as error:
            assert error.parameter == 'kv_scope', error.parameter
        else:
            raise AssertionError(f'an engine request generated with {extra_args!r}')

    notify = AsyncMock()
    serving = SimpleNamespace(
        has_kv_connector=True,
        engine_client=SimpleNamespace(notify_kv_transfer_request_rejected=notify),
        _get_data_parallel_rank=lambda raw_request: None,
    )
    refused = ChatCompletionGenerationRequest(
        model='m', messages=[{'role': 'user', 'content': 'hi'}], kv_scope='agent',
        kv_transfer_params={'do_remote_prefill': True},
    )
    refusal = ErrorResponse(error=ErrorInfo(message='no', type='BadRequest', code=400))

    async def rejected():
        return refusal

    assert await GenerateBaseServing._with_kv_transfer_rejection_cleanup(
        serving, rejected(), refused, None) is refusal
    (request_id, params, scope), _ = notify.await_args
    assert (request_id, scope) == (refused.request_id, 'agent'), (request_id, scope)

    engine_core = SimpleNamespace(add_request_async=AsyncMock())
    await AsyncLLM.notify_kv_transfer_request_rejected(
        SimpleNamespace(engine_core=engine_core), request_id, params, scope)
    (notice,), _ = engine_core.add_request_async.await_args
    assert notice.abort_immediately
    notice = Request.from_engine_core_request(notice, None)
    assert notice.kv_scope == 'agent'

    manager = CPUOffloadingManager(4)
    earlier = request('agent', {b'turn': (b'turn',)})
    store(manager, earlier, [b'turn'])
    manager.retain_context([b'turn'], earlier)
    manager.on_request_finished(earlier)
    context = _create_req_context(notice)
    manager.on_new_request(context)
    manager.on_request_finished(context)
    assert manager._idle_context == {'agent': 'agent'}, manager._idle_context
    assert manager.lookup(b'turn', earlier) is LookupResult.HIT


def main():
    from fastapi import FastAPI

    from vllm.entrypoints.generate.api_router import register_generate_api_routers
    from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers

    app = FastAPI()
    app.state.args = SimpleNamespace(tokens_only=False)
    register_generate_api_routers(app)
    register_scale_out_api_routers(app, ("generate",))
    paths = {route.path for route in app.routes}
    assert "/generative_scoring" not in paths, paths
    assert not any(path.startswith("/cohere") for path in paths), paths
    assert {
        "/v1/chat/completions", "/v1/chat/completions/batch", "/v1/completions",
        "/v1/responses", "/v1/messages", "/inference/v1/generate",
    } <= paths, paths
    # /invocations, mounted beside them, dispatches by the request types the
    # generation endpoints take: a body naming its line of work reaches them
    # with it; one naming none is refused naming kv_scope, an error the
    # dispatcher -- which skips only pydantic mismatches -- does not swallow.
    import pydantic

    from vllm.entrypoints.generate.factories import get_generate_invocation_types

    for request_type, _ in get_generate_invocation_types(("generate",)):
        adapter = pydantic.TypeAdapter(request_type)
        body = {"model": "unit", "kv_scope": "agent"}
        body.update({"messages": [{"role": "user", "content": "hi"}]}
                    if "messages" in request_type.model_fields else {"prompt": "hi"})
        named = adapter.validate_python(body)
        extra_args = named.to_sampling_params(8, {}).extra_args or {}
        assert extra_args.get("kv_scope") == "agent", (
            f"/invocations dropped the kv_scope sent to {request_type.__name__}")
        del body["kv_scope"]
        try:
            adapter.validate_python(body)
        except VLLMValidationError as error:
            assert error.parameter == "kv_scope", request_type
        else:
            raise AssertionError(f"{request_type.__name__} generates for no agent")
    asyncio.run(check_stream_identity())
    asyncio.run(check_rejection_notice_identity())

    # Lookup matches content alone. The membership catalog that once let only
    # an ID without cached blocks match another ID's data is gone.
    assert importlib.util.find_spec('vllm.v1.core.prefix_cache') is None
    manager = CPUOffloadingManager(7)
    content = {key: (key,) for key in
               (b'prefix', b'producer-tail', b'fork-tail', b'own', b'other')}
    producer = request('producer', content)
    store(manager, producer, [b'prefix', b'producer-tail'])
    manager.on_request_finished(producer)

    fork = request('fork', content)
    assert manager.lookup(b'prefix', fork) is LookupResult.HIT
    manager.prepare_load([b'prefix'], fork)
    manager.complete_load([b'prefix'], fork)
    store(manager, fork, [b'fork-tail'])
    # The fork now has cached data of its own and still matches every
    # resident prefix it names, including data only the producer computed.
    for key in (b'prefix', b'fork-tail', b'producer-tail'):
        assert manager.lookup(key, fork) is LookupResult.HIT
    manager.on_request_finished(fork)

    # Whole-agent retention: pressure releases the least recently used ID,
    # never the storing one, and references a survivor holds keep shared data.
    incoming = request('incoming', content)
    output = store(manager, incoming, [b'own', b'other'])
    assert output.evicted_keys == []
    pressure = request('pressure', {key: (key,) for key in (b'p1', b'p2', b'p3')})
    output = store(manager, pressure, [b'p1', b'p2', b'p3'])
    assert output.evicted_keys == [b'producer-tail']
    assert 'producer' not in manager._agents and 'fork' in manager._agents
    assert manager.lookup(b'prefix', fork) is LookupResult.HIT

    # An incomplete window chunk serves its written suffix. Filling it uses
    # the same row; the complete row then serves every reader, including the
    # agent that wrote only the suffix.
    manager = CPUOffloadingManager(3, enable_events=True)
    canonical, suffix = (b'early', b'middle', b'end'), (b'middle', b'end')
    window = request('window', {b'end': canonical}, {b'end': suffix}, {b'end': suffix})
    store(manager, window, [b'end'])
    slot = manager._blocks[b'end'].block_id
    assert manager.lookup(b'end', window) is LookupResult.HIT
    assert manager._lookup(b'end', ReqContext('storage')) is LookupResult.MISS
    assert not list(manager.take_events())
    earlier = request('earlier', {b'end': canonical})
    assert manager.lookup(b'end', earlier) is LookupResult.MISS
    output = store(manager, earlier, [b'end'])
    assert list(output.store_spec.block_ids) == [slot]
    assert manager._num_allocated_blocks == 1
    assert manager._available_content[b'end'] == canonical
    assert manager.lookup(b'end', request('window', {b'end': canonical})) is LookupResult.HIT
    assert manager._lookup(b'end', ReqContext('storage')) is LookupResult.HIT
    assert len(list(manager.take_events())) == 1
    print('Shared prefix cache CPU unit PASS')


if __name__ == '__main__':
    main()
