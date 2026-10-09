#!/usr/bin/env python3
"""Verify installed shared-prefix semantics using CPU metadata only."""

import asyncio
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import vllm.v1.engine.input_processor as input_processor_module
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    offloading_connector as offloading_connector_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    NON_NEGATIVE_INTEGER,
    KVTransferParamsKeys,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
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
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.p2p.manager import (
    _annotate_req_context,
    _parse_dest,
    _parse_source,
)
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


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


# The connector config/runtime-v1.sh launches: the CPU tier alone.
DEPLOYED_KV_TRANSFER = KVTransferConfig(
    kv_connector='OffloadingConnector', kv_role='kv_both',
    kv_connector_extra_config={'cpu_kv_cache_users': 1},
)
NIXL_KV_TRANSFER = KVTransferConfig(kv_connector='NixlConnector', kv_role='kv_both')
MOONCAKE_KV_TRANSFER = KVTransferConfig(kv_connector='MooncakeConnector',
                                        kv_role='kv_both')
# What a NIXL prefill node returns for its decode node, which upstream's proxy
# hands on as the decode request's own.
NIXL_REMOTE_PREFILL = {'do_remote_prefill': True, 'remote_block_ids': [[1, 2]],
                       'remote_engine_id': 'e', 'remote_request_id': 'r',
                       'remote_host': 'h', 'remote_port': 5600, 'tp_size': 1}
P2P_KV_TRANSFER = KVTransferConfig(
    kv_connector='OffloadingConnector', kv_role='kv_both',
    kv_connector_extra_config={'spec_name': 'TieringOffloadingSpec',
                               'cpu_kv_cache_users': 1,
                               'secondary_tiers': [{'type': 'p2p'}]},
)


def admission(kv_transfer_config):
    """The real InputProcessor, its tokenizer and multimodal parts stubbed."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(try_get_generation_config=dict,
                                     return_sampling_mask=False),
        cache_config=None, lora_config=None, scheduler_config=None,
        speculative_config=None, structured_outputs_config=None,
        observability_config=None, use_v2_model_runner=False,
        reasoning_config=None, kv_transfer_config=kv_transfer_config,
    )
    with patch.object(input_processor_module, 'InputPreprocessor'):
        return InputProcessor(
            vllm_config, SimpleNamespace(_executor=None, tokenizer=None),
            mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda _: False),
        )


def chat(**fields):
    return ChatCompletionGenerationRequest(
        model='m', messages=[{'role': 'user', 'content': 'hi'}], kv_scope='agent',
        **fields,
    )


def admit(processor, **fields):
    params = chat(**fields).to_sampling_params(8, {})
    with patch.object(SamplingParams, 'verify'):
        processor._validate_params(params, ('generate',))


def refusal(processor, **fields):
    try:
        admit(processor, **fields)
    except VLLMValidationError as error:
        assert error.parameter == 'kv_transfer_params', error.parameter
        return str(error)
    raise AssertionError(f'admitted {fields!r}')


def serving_over(processor):
    notify = AsyncMock()
    serving = GenerateBaseServing(SimpleNamespace(
        model_config=None, renderer=None, input_processor=processor,
        vllm_config=processor.vllm_config, notify_kv_transfer_request_rejected=notify,
    ), None, request_logger=None)
    return serving, notify


async def refuse(serving, request):
    refusal = ErrorResponse(error=ErrorInfo(message='no', type='BadRequest', code=400))

    async def rejected():
        return refusal

    assert await serving._with_kv_transfer_rejection_cleanup(
        rejected(), request, None) is refusal


async def check_kv_transfer_params_admission():
    """A request reaches the engine naming only kv_transfer_params the
    configured connector takes. The deployed CPU tier takes max_offload_tokens
    alone: it has no secondary tier for kv_load_tiers to select and no remote
    prefill. Unrefused, both were accepted and never read, and a non-object
    sent through vllm_xargs reached the connector, whose first read of it
    raised in the engine core for every user."""
    deployed = admission(DEPLOYED_KV_TRANSFER)
    assert deployed.kv_transfer_params_keys == KVTransferParamsKeys(
        keys={'max_offload_tokens': NON_NEGATIVE_INTEGER}), deployed.kv_transfer_params_keys
    admit(deployed, kv_transfer_params={'max_offload_tokens': 64})
    for key in ('do_remote_prefill', 'kv_load_tiers'):
        message = refusal(deployed, kv_transfer_params={key: True})
        assert f'key {key!r} is not a parameter of OffloadingConnector' in message, message
        assert '(it takes max_offload_tokens). Next: remove it.' in message, message
    # No connector takes upstream's reused prompt ids: the refusal names what
    # this server does instead of sending the request elsewhere.
    for processor in (deployed, admission(None), admission(NIXL_KV_TRANSFER)):
        message = refusal(processor, kv_transfer_params={'prompt_token_ids': [1]})
        assert "takes no prompt ids from a request. Next: remove it." in message, message
        assert 'send the request to a server' not in message, message
    for smuggled in ('x', ['a'], 3):
        message = refusal(deployed, vllm_xargs={'kv_transfer_params': smuggled})
        assert 'must be an object of KV connector parameters' in message, message
    message = refusal(admission(None), kv_transfer_params={'max_offload_tokens': 1})
    assert 'reaches no KV connector: this server configures none' in message, message
    # A value of a key the connector takes is held to what it reads.
    for value in (-1, '64', 1.5, True):
        message = refusal(deployed, kv_transfer_params={'max_offload_tokens': value})
        assert "'max_offload_tokens' must be a non-negative integer" in message, message

    # No connector here holds remote-prefill blocks, so a refused request
    # that names some sends no notice into the engine.
    for processor in (deployed, admission(None)):
        serving, notify = serving_over(processor)
        await refuse(serving, chat(kv_transfer_params={'do_remote_prefill': True}))
        notify.assert_not_awaited()


def check_kv_transfer_values_are_what_their_connector_reads():
    """The P2P tier reads its peer objects with .get in on_new_request, inside
    EngineCore.add_request, where a value of another shape raised and ended the
    engine core for every user. Admission takes exactly the values the tier's
    own reader reads, and refuses the ones that raised there."""
    p2p = admission(P2P_KV_TRANSFER)
    peer = {'kv_request_id': 't', 'remote_host': '10.0.0.1', 'remote_port': 5710}
    taken = [{'remote_prefiller': peer}, {'remote_kv_source': peer},
             {'remote_decoder': {'kv_request_id': 't'}},
             {'remote_decoder': {'kv_request_id': 't'}, 'remote_kv_source': peer}]
    refused = [{'remote_kv_source': 'x'}, {'remote_decoder': 'x'},
               {'remote_prefiller': [1]}]
    # Unhashable, it raises later, where the tier keys its sessions by it.
    message = refusal(p2p, kv_transfer_params={
        'remote_prefiller': {**peer, 'kv_request_id': ['t']}})
    assert "'remote_prefiller' must be an object whose kv_request_id" in message, message
    for params in taken:
        admit(p2p, kv_transfer_params=params)
        _annotate_req_context(ReqContext(req_id='r', kv_transfer_params=params,
                                         kv_scope='agent'))
    for params in refused:
        ((key, _),) = params.items()
        message = refusal(p2p, kv_transfer_params=params)
        assert f'{key!r} must be an object whose kv_request_id' in message, message
        assert 'Next: send it in that shape, or remove it.' in message, message
        try:
            _annotate_req_context(ReqContext(req_id='r', kv_transfer_params=params,
                                             kv_scope='agent'))
        except (AttributeError, TypeError):
            pass
        else:
            raise AssertionError(f'the P2P tier reads {params!r}; admission refuses it')


def check_kv_transfer_values_are_read_with_their_companions():
    """A value that asks a connector to act is read only together with the
    keys its connector reads it with, and never beside one it excludes; the
    connectors' own readers either raised on such a request inside the engine
    core -- NIXL's and Mooncake's decode node record a remote prefill by those
    keys, for an aborted request too -- or acted on one key and dropped the
    other. A null or false asks for nothing: upstream's proxies fill the keys
    a prefill node does not use with it."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlConnectorMetadata,
    )

    nixl = admission(NIXL_KV_TRANSFER)
    admit(nixl, kv_transfer_params=NIXL_REMOTE_PREFILL)
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv('r', [], dict(NIXL_REMOTE_PREFILL))
    assert meta.reqs_to_recv['r'].remote.host == 'h', meta.reqs_to_recv
    admit(nixl, kv_transfer_params={
        'do_remote_decode': True, 'do_remote_prefill': False, 'remote_engine_id': None,
        'remote_block_ids': None, 'remote_host': None, 'remote_port': None})
    for dropped, listed in (
        (('remote_block_ids',), "'remote_block_ids'"),
        (('remote_host', 'remote_port'), "'remote_host' and 'remote_port'"),
        (('remote_block_ids', 'remote_engine_id', 'remote_request_id', 'remote_host',
          'remote_port'), "'remote_block_ids', 'remote_engine_id', "
                          "'remote_request_id', 'remote_host' and 'remote_port'"),
    ):
        params = {key: value for key, value in NIXL_REMOTE_PREFILL.items()
                  if key not in dropped}
        message = refusal(nixl, kv_transfer_params=params)
        assert "key 'do_remote_prefill' is a boolean (True), which NixlConnector " \
            'acts on only together with ' in message, message
        assert f'this request sends no value for {listed}. Next: send ' in message, message
        try:
            NixlConnectorMetadata().add_new_req_to_recv('r', [], params)
        except KeyError:
            pass
        else:
            raise AssertionError(f'NIXL records {params!r}; admission refuses it')
        # A null is no value: admission refuses it beside do_remote_prefill.
        refusal(nixl, kv_transfer_params={**NIXL_REMOTE_PREFILL,
                                          **dict.fromkeys(dropped)})

    mooncake = admission(MOONCAKE_KV_TRANSFER)
    complete = {'do_remote_prefill': True, 'transfer_id': 't',
                'remote_engine_id': 'e', 'remote_bootstrap_addr': 'h:1'}
    admit(mooncake, kv_transfer_params=complete)
    MooncakeConnectorMetadata().add_new_req('r', [], dict(complete))
    params = {'do_remote_prefill': True, 'transfer_id': 't'}
    message = refusal(mooncake, kv_transfer_params=params)
    assert "no value for 'remote_engine_id' and 'remote_bootstrap_addr'" in message, message
    try:
        MooncakeConnectorMetadata().add_new_req('r', [], params)
    except KeyError:
        pass
    else:
        raise AssertionError(f'Mooncake records {params!r}; admission refuses it')
    message = refusal(mooncake, kv_transfer_params={'do_remote_decode': True})
    assert "no value for 'transfer_id'. Next: send it beside" in message, message

    p2p = admission(P2P_KV_TRANSFER)
    peer = {'kv_request_id': 't', 'remote_host': '10.0.0.1', 'remote_port': 5710}
    for params in ({'remote_prefiller': {}}, {'remote_kv_source': {'kv_request_id': 't'}},
                   {'remote_prefiller': {'kv_request_id': 't', 'remote_host': 'h'}}):
        ((key, _),) = params.items()
        message = refusal(p2p, kv_transfer_params=params)
        assert f'{key!r} must be an object whose kv_request_id is a string' in message, message
        assert _parse_source(params) is None, params
    message = refusal(p2p, kv_transfer_params={'remote_decoder': {}})
    assert "'remote_decoder' must be an object whose kv_request_id is a string" in message
    assert _parse_dest({'remote_decoder': {}}).kv_request_id is None
    other = {**peer, 'kv_request_id': 'other'}
    for beside, value in (('remote_kv_source', other),
                          ('remote_decoder', {'kv_request_id': 'other'})):
        params = {'remote_prefiller': peer, beside: value}
        message = refusal(p2p, kv_transfer_params=params)
        assert f"'remote_prefiller' is sent beside {beside!r}" in message, message
        assert 'Next: remove ' in message, message
    # The tier read the prefiller and dropped the other source.
    assert _parse_source({'remote_prefiller': peer,
                          'remote_kv_source': other}).kv_request_id == 't'
    legal = {'remote_decoder': {'kv_request_id': 'd'}, 'remote_kv_source': peer}
    admit(p2p, kv_transfer_params=legal)
    assert (_parse_source(legal).kv_request_id, _parse_dest(legal).kv_request_id) == (
        't', 'd')


def check_pooling_offload_is_refused_at_startup():
    """The CPU tier accounts the host KV cache per agent ID, which a pooling
    request does not carry: its manager raises on the first one, in the engine
    core. A pooling model is refused that tier when the connector is built."""
    context = ReqContext(req_id='pooling', kv_transfer_params=None, kv_scope=None)
    try:
        CPUOffloadingManager(4).on_new_request(context)
    except Exception as error:
        assert 'agent ID' in str(error), error
    else:
        raise AssertionError('the CPU tier took a request without an agent ID')

    class Built(Exception):
        pass

    with patch.object(offloading_connector_module, 'build_offloading_config',
                      side_effect=Built):
        for runner_type in ('pooling', 'generate'):
            vllm_config = SimpleNamespace(
                kv_transfer_config=DEPLOYED_KV_TRANSFER,
                model_config=SimpleNamespace(runner_type=runner_type))
            try:
                OffloadingConnector(vllm_config, None, None)
            except ValueError as error:
                assert runner_type == 'pooling', error
                assert 'accounts the offloaded KV cache per agent ID' in str(error)
                assert 'Next: serve this pooling model without the OffloadingConnector' in str(error)
            except Built:
                assert runner_type == 'generate', runner_type
            else:
                raise AssertionError('the connector was built without its config')


def check_a_refused_engine_request_is_its_own_error():
    """The engine request type refuses on decode what it refuses when built,
    so a request changed after it was built is refused as it is decoded. That
    is the request's error, returned to its client; it used to end the input
    thread, after which the engine received nothing, silently."""
    built = EngineCoreRequest(
        request_id='changed', prompt_token_ids=[0], mm_features=None,
        sampling_params=SamplingParams(max_tokens=1, extra_args={'kv_scope': 'agent'}),
        pooling_params=None, arrival_time=0.0, lora_request=None, cache_salt=None,
        data_parallel_rank=None, client_index=3,
    )
    built.sampling_params = SamplingParams(max_tokens=1)
    frames = MsgpackEncoder().encode(built)
    errors, preprocessed = [], []
    engine = SimpleNamespace(
        _send_error_outputs_to_client=lambda ids, client: errors.append((ids, client)),
        preprocess_add_request=lambda req: preprocessed.append(req) or (req, 0),
    )
    engine._handle_refused_add_request = (
        lambda frames: EngineCoreProc._handle_refused_add_request(engine, frames))
    decoder = MsgpackDecoder(EngineCoreRequest)
    assert EngineCoreProc._receive_add_request(engine, decoder, frames) is None
    assert errors == [(['changed'], 3)], errors
    assert preprocessed == [], preprocessed

    built.sampling_params = SamplingParams(max_tokens=1, extra_args={'kv_scope': 'agent'})
    received = EngineCoreProc._receive_add_request(
        engine, decoder, MsgpackEncoder().encode(built))
    assert received is not None and received[0].request_id == 'changed', received
    assert errors == [(['changed'], 3)], errors


async def check_rejection_notice_identity():
    """A connector that takes do_remote_prefill is told of a refused request
    that named remote-prefill blocks. The notice reaches the CPU tier (beside
    such a connector under MultiConnector) as a request of the refused
    request's own agent, and releases nothing it kept. Without the identity
    the tier raised, and an exception there ends the engine core for every
    user."""
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

    serving, notify = serving_over(admission(NIXL_KV_TRANSFER))
    # The notice reaches the connector without crossing admission: one whose
    # parameters admission refuses names no blocks the connector can read,
    # and is not sent.
    await refuse(serving, chat(kv_transfer_params={'do_remote_prefill': True,
                                                   'remote_engine_id': ['e']}))
    notify.assert_not_awaited()
    # Nor one whose remote prefill lacks the keys that name its blocks: the
    # connector records the notice's transfer by them, and raised without.
    await refuse(serving, chat(kv_transfer_params={'do_remote_prefill': True}))
    notify.assert_not_awaited()
    refused = chat(kv_transfer_params=NIXL_REMOTE_PREFILL)
    await refuse(serving, refused)
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
    asyncio.run(check_kv_transfer_params_admission())
    check_kv_transfer_values_are_what_their_connector_reads()
    check_kv_transfer_values_are_read_with_their_companions()
    check_pooling_offload_is_refused_at_startup()
    check_a_refused_engine_request_is_its_own_error()
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
