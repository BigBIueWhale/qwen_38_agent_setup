#!/usr/bin/env python3
"""Verify installed shared-prefix semantics using CPU metadata only."""

import asyncio
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from vllm.engine.protocol import StreamingInput
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM, InputStreamError
from vllm.v1.engine.input_processor import InputProcessor, require_kv_scope
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


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
    asyncio.run(check_stream_identity())

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
