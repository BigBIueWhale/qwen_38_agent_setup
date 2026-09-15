#!/usr/bin/env python3
"""Verify installed shared-prefix semantics using CPU metadata only."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from vllm.engine.protocol import StreamingInput
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.prefix_cache import PrefixCacheIndex
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


async def check_scoring_identity():
    from fastapi import FastAPI
    from pydantic import ValidationError

    from vllm.entrypoints.generate.api_router import register_generate_api_routers
    from vllm.entrypoints.generate.generative_scoring.serving import (
        GenerativeScoringRequest, GenerativeScoringResponse, ServingGenerativeScoring,
    )
    from vllm.logprobs import Logprob
    from vllm.outputs import CompletionOutput, RequestOutput

    app = FastAPI()
    register_generate_api_routers(app)
    assert '/generative_scoring' in {route.path for route in app.routes}

    body = dict(query=[10], items=[[11], [12]], label_token_ids=[20])
    for fields in ({}, {'kv_scope': None}, {'kv_scope': ''}, {'kv_scope': ' \t'},
                   {'kv_scope': 7}, {'kv_scope': False}):
        try:
            GenerativeScoringRequest(**body, **fields)
        except ValidationError as error:
            assert error.errors()[0]['loc'] == ('kv_scope',)
        else:
            raise AssertionError('Scoring accepted a missing or invalid agent ID')

    serving = object.__new__(ServingGenerativeScoring)
    serving._check_model = AsyncMock(return_value=None)
    serving.model_config = SimpleNamespace(get_vocab_size=lambda: 100, max_model_len=100)
    serving.renderer = SimpleNamespace(tokenizer=MagicMock())
    serving._maybe_get_adapters = MagicMock(return_value=None)
    serving._log_inputs = MagicMock()
    serving.models = SimpleNamespace(model_name=lambda lora: 'model')
    seen = []

    async def generate(prompt, params, request_id, **kwargs):
        seen.append(require_kv_scope(params))
        yield RequestOutput(
            request_id=request_id, prompt=None,
            prompt_token_ids=prompt['prompt_token_ids'], prompt_logprobs=None,
            outputs=[CompletionOutput(
                index=0, text='', token_ids=[20], cumulative_logprob=-0.5,
                logprobs=[{20: Logprob(logprob=-0.5)}], finish_reason='length',
            )], finished=True,
        )

    serving.engine_client = SimpleNamespace(errored=False, generate=generate)
    response = await serving.create_generative_scoring(
        GenerativeScoringRequest(**body, kv_scope=' opaque: scoring-agent '),
    )
    assert isinstance(response, GenerativeScoringResponse)
    assert seen == [' opaque: scoring-agent ', ' opaque: scoring-agent ']
    assert [item.score for item in response.data] == [1.0, 1.0]


def main():
    asyncio.run(check_stream_identity())
    asyncio.run(check_scoring_identity())
    index = PrefixCacheIndex()
    gpu = object()
    index.register_tier(gpu, 4)
    manager = CPUOffloadingManager(4)
    manager.bind_prefix_cache(index)
    index.insert(gpu, b'prefix', (b'prefix',))
    index.acquire(gpu, 'source-agent', [b'prefix'])
    content = {key: (key,) for key in (b'prefix', b'producer-tail', b'fork-tail')}
    producer = request('producer', content)
    store(manager, producer, [b'prefix', b'producer-tail'])
    index.remove(gpu, b'prefix')
    assert index.view('source-agent').keys == frozenset((b'prefix',))

    fork = request('fork', content)
    manager.begin_lookup(fork)
    assert manager.lookup(b'prefix', fork) is LookupResult.HIT
    manager.prepare_load([b'prefix'], fork)
    manager.complete_load([b'prefix'], fork)
    store(manager, fork, [b'fork-tail'])
    manager.begin_lookup(fork)
    assert manager.lookup(b'prefix', fork) is LookupResult.HIT
    assert manager.lookup(b'fork-tail', fork) is LookupResult.HIT
    assert manager.lookup(b'producer-tail', fork) is LookupResult.MISS

    # GPU membership makes the agent existing even with no CPU membership.
    index.insert(gpu, b'gpu-only', (b'gpu-only',))
    index.acquire(gpu, 'gpu-agent', [b'gpu-only'])
    gpu_agent = request('gpu-agent', content)
    manager.begin_lookup(gpu_agent)
    assert manager.lookup(b'prefix', gpu_agent) is LookupResult.MISS
    index.release_agent(manager.cache_tier, 'producer')
    manager.begin_lookup(fork)
    assert manager.lookup(b'prefix', fork) is LookupResult.HIT

    # An incomplete window chunk serves its written suffix. Filling it
    # preserves each previous user's acquired extent and uses the same row.
    manager = CPUOffloadingManager(3, enable_events=True)
    canonical, suffix = (b'early', b'middle', b'end'), (b'middle', b'end')
    window = request('window', {b'end': canonical}, {b'end': suffix}, {b'end': suffix})
    store(manager, window, [b'end'])
    slot = manager._blocks[b'end'].block_id
    manager.begin_lookup(window)
    assert manager.lookup(b'end', window) is LookupResult.HIT
    assert manager._lookup(b'end', ReqContext('storage')) is LookupResult.MISS
    assert not list(manager.take_events())
    earlier = request('earlier', {b'end': canonical})
    manager.begin_lookup(earlier)
    assert manager.lookup(b'end', earlier) is LookupResult.MISS
    output = store(manager, earlier, [b'end'])
    assert list(output.store_spec.block_ids) == [slot]
    assert manager._num_allocated_blocks == 1
    assert manager.prefix_cache.view('window').keys == frozenset(suffix)
    assert manager.prefix_cache.view('earlier').keys == frozenset(canonical)
    assert manager._lookup(b'end', ReqContext('storage')) is LookupResult.HIT
    assert len(list(manager.take_events())) == 1
    print('Shared prefix cache CPU unit PASS')


if __name__ == '__main__':
    main()
