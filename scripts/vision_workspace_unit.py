#!/usr/bin/env python3
"""Build-time invariants for phase-safe multimodal workspace reuse."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.config import vllm as vllm_config
from vllm.config.compilation import CompilationMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.attention import attention as attention_layer
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.utils import mem_utils
from vllm.v1.attention.backends import turboquant_attn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker import gpu_model_runner, gpu_worker, workspace


def test_workspace_lifetime() -> None:
    original_empty_cache = torch.accelerator.empty_cache
    original_manager = workspace._manager
    empty_cache_calls = []
    torch.accelerator.empty_cache = lambda: empty_cache_calls.append(True)
    try:
        manager = workspace.WorkspaceManager(torch.device("cpu"))
        (primary,) = manager.get_simultaneous(((64,), torch.uint8))
        primary.fill_(17)
        primary_ptr = primary.data_ptr()
        manager.get_reclaimable_simultaneous(
            "phase-local",
            ((64,), torch.float32),
            ((32,), torch.float16),
        )
        manager.lock()
        empty_cache_calls.clear()
        workspace._manager = manager

        try:
            with workspace.release_reclaimable_workspaces() as released:
                assert released == 512
                assert primary.data_ptr() == primary_ptr
                assert primary.tolist() == [17] * 64
                try:
                    manager.get_reclaimable_simultaneous(
                        "phase-local", ((64,), torch.float32)
                    )
                except AssertionError as exc:
                    assert "Model phases must not overlap" in str(exc)
                else:
                    raise AssertionError("released workspace was still accessible")
                raise RuntimeError("controlled encoder failure")
        except RuntimeError as exc:
            assert str(exc) == "controlled encoder failure"
        else:
            raise AssertionError("controlled encoder failure was lost")

        (restored,) = manager.get_reclaimable_simultaneous(
            "phase-local", ((64,), torch.float32)
        )
        assert restored.shape == (64,)
        assert empty_cache_calls == [True, True]
        try:
            manager.get_reclaimable_simultaneous(
                "phase-local", ((1024,), torch.float32)
            )
        except AssertionError as exc:
            assert "is locked" in str(exc)
        else:
            raise AssertionError("locked reclaimable workspace grew")

        try:
            with workspace.release_reclaimable_workspaces():
                with workspace.release_reclaimable_workspaces():
                    pass
        except AssertionError as exc:
            assert "already released" in str(exc)
        else:
            raise AssertionError("nested workspace release was accepted")

        manager.get_reclaimable_simultaneous(
            "phase-local", ((64,), torch.float32)
        )
        assert empty_cache_calls == [True, True, True, True]
    finally:
        workspace._manager = original_manager
        torch.accelerator.empty_cache = original_empty_cache


def test_graph_capture_refuses_reclaimable_views() -> None:
    # A graph would keep the address the workspace is restored away from, so
    # the views are refused while a capture is under way, and only then.
    original_capturing = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    try:
        manager = workspace.WorkspaceManager(torch.device("cuda"))
        try:
            manager.get_reclaimable_simultaneous("phase-local", ((64,), torch.uint8))
        except AssertionError as error:
            assert "being captured" in str(error), error
            assert "--enforce-eager" in str(error), error
        else:
            raise AssertionError("A capture was given reclaimable workspace views")
        assert manager._reclaimable_workspaces == {}
        # The capture state of a CUDA stream says nothing about a CPU workspace.
        cpu = workspace.WorkspaceManager(torch.device("cpu"))
        (view,) = cpu.get_reclaimable_simultaneous("phase-local", ((64,), torch.uint8))
        assert view.numel() == 64
    finally:
        torch.cuda.is_current_stream_capturing = original_capturing


def test_turboquant_reservation_routing() -> None:
    calls = []

    class FakeWorkspaceManager:
        def get_simultaneous(self, *shapes_and_dtypes):
            calls.append(("primary", None, shapes_and_dtypes))

        def get_reclaimable_simultaneous(self, name, *shapes_and_dtypes):
            calls.append(("reclaimable", name, shapes_and_dtypes))

    original_manager = turboquant_attn.current_workspace_manager
    original_initialized = turboquant_attn.is_workspace_manager_initialized
    turboquant_attn.current_workspace_manager = lambda: FakeWorkspaceManager()
    turboquant_attn.is_workspace_manager_initialized = lambda: True
    try:
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=16,
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
            ),
            model_config=SimpleNamespace(
                max_model_len=8192,
                dtype=torch.float16,
                get_num_attention_heads=lambda parallel_config: 8,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                decode_context_parallel_size=1,
            ),
            attention_config=SimpleNamespace(
                tq_max_kv_splits_for_cuda_graph=4,
            ),
        )
        cache_spec = FullAttentionSpec(
            block_size=32,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            state_content_bytes=102,
        )
        turboquant_attn.TurboQuantMetadataBuilder(
            kv_cache_spec=cache_spec,
            layer_names=["layers.0.self_attn.attn"],
            vllm_config=vllm_config,
            device=torch.device("cuda"),
        )
    finally:
        turboquant_attn.current_workspace_manager = original_manager
        turboquant_attn.is_workspace_manager_initialized = original_initialized

    # The continuation workspace is the only phase-local reservation: a
    # raw reserve would be a call this fake manager does not answer.
    assert [call[0] for call in calls] == ["primary", "reclaimable"]
    assert calls[1][1] == "turboquant_continuation_prefill"
    assert calls[1][2] == (
        ((1, 4, 8192, 128), torch.float16),
        ((1, 4, 8192, 128), torch.float16),
    )


class _ReachedKernel(Exception):
    """The served path went on past its witness to a GPU kernel."""


def test_served_paths_witness_their_own_execution() -> None:
    # The profile's witnesses are produced by the served code itself: an
    # attention layer reads its metadata through the attention op, and a
    # TurboQuant continuation requests the reclaimable workspace, only on
    # the path that executes; GDN reads its metadata only on the path that
    # updates its state. Without metadata both return having read nothing.
    max_model_len, block, q_len = 4096, 32, 2048
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(tq_max_kv_splits_for_cuda_graph=4),
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            dtype=torch.float16,
            get_num_attention_heads=lambda parallel_config: 8,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1, enable_chunked_prefill=True, max_num_batched_tokens=q_len
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, decode_context_parallel_size=1
        ),
    )

    class Launcher:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise _ReachedKernel
            return launch

    saved_config = turboquant_attn.get_current_vllm_config
    saved_dequant = turboquant_attn._tq_full_dequant_kv
    saved_manager = workspace._manager
    turboquant_attn.get_current_vllm_config = lambda: vllm_config
    turboquant_attn._tq_full_dequant_kv = Launcher()
    workspace._manager = manager = workspace.WorkspaceManager(torch.device("cpu"))
    try:
        spec = FullAttentionSpec(
            block_size=block, num_kv_heads=4, head_size=128, head_size_v=128,
            dtype=torch.uint8, state_content_bytes=102,
        )
        turboquant_attn.TurboQuantMetadataBuilder(
            spec, ["layers.0.self_attn.attn"], vllm_config, torch.device("cpu")
        )
        impl = turboquant_attn.TurboQuantAttentionImpl(
            num_heads=8, head_size=128, scale=1.0, num_kv_heads=4,
            kv_cache_dtype="turboquant_k8v4",
        )
        attn = "layers.0.self_attn.attn"
        attn_layer = SimpleNamespace(
            impl=impl,
            kv_cache=torch.zeros(
                (max_model_len // block, block, 4, 256), dtype=torch.uint8
            ),
        )

        def tq_metadata(seq_len):
            return turboquant_attn.TurboQuantMetadata(
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                slot_mapping=torch.full((q_len,), -1, dtype=torch.int64),
                block_table=torch.zeros(
                    (1, max_model_len // block), dtype=torch.int32
                ),
                query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
                num_actual_tokens=q_len, max_query_len=q_len, max_seq_len=seq_len,
                is_prefill=True, num_decodes=0, num_decode_tokens=0,
                query_start_loc_cpu=torch.tensor([0, q_len], dtype=torch.int32),
                seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            )

        def attend(metadata):
            reads: set[str] = set()
            if metadata is not None:
                metadata = gpu_model_runner._record_layer_reads(
                    {attn: metadata}, reads
                )
            context = ForwardContext(
                no_compile_layers={attn: attn_layer},
                attn_metadata=metadata,
                slot_mapping={},
            )
            query = torch.zeros(q_len, 8 * 128, dtype=torch.float16)
            key = torch.zeros(q_len, 4 * 128, dtype=torch.float16)
            with (
                override_forward_context(context),
                manager.witness_reclaimable_requests() as requested,
            ):
                try:
                    attention_layer.unified_attention_with_output(
                        query, key, key,
                        torch.empty(q_len, 8 * 128, dtype=torch.float16), attn,
                    )
                    reached = False
                except _ReachedKernel:
                    reached = True
            return reads, requested, reached

        # No metadata: zeros, and nothing read or requested.
        assert attend(None) == (set(), set(), False)
        # A first prefill chunk attends within the step: read, no workspace.
        assert attend(tq_metadata(q_len)) == ({attn}, set(), False)
        # A continuation at full context requests the workspace and goes on
        # to dequantise the cached K/V into it.
        assert attend(tq_metadata(max_model_len)) == (
            {attn}, {_CONTINUATION}, True
        )
    finally:
        turboquant_attn.get_current_vllm_config = saved_config
        turboquant_attn._tq_full_dequant_kv = saved_dequant
        workspace._manager = saved_manager

    class StateUpdate:
        def __get__(self, layer, owner):
            raise _ReachedKernel

    warm_ups = []
    Layer = type("Layer", (), {
        "_forward_core": qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core,
        "conv1d": StateUpdate(),
    })
    gdn = Layer()
    gdn.prefix = "layers.1.linear_attn"
    gdn._warmup_prefill_kernels = lambda mixed_qkv, v_dim: warm_ups.append(v_dim)
    gdn.enable_packed_recurrent_decode = False
    gdn.kv_cache = (torch.zeros(2, 64, 3), torch.zeros(2, 4, 16, 16))
    tokens = 64
    gdn_metadata = GDNAttentionMetadata(
        num_prefills=1, num_prefill_tokens=tokens, num_decodes=0,
        num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=tokens,
        has_initial_state=torch.ones(1, dtype=torch.bool),
        non_spec_query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(1, dtype=torch.int32),
    )
    for metadata, expected in ((None, (set(), False)),
                               (gdn_metadata, ({gdn.prefix}, True))):
        reads: set[str] = set()
        if metadata is not None:
            metadata = gpu_model_runner._record_layer_reads(
                {gdn.prefix: metadata}, reads
            )
        context = ForwardContext(
            no_compile_layers={gdn.prefix: gdn},
            attn_metadata=metadata,
            slot_mapping={},
        )
        with override_forward_context(context):
            try:
                qwen_gdn_linear_attn.qwen_gdn_attention_core(
                    torch.zeros(tokens, 64), torch.zeros(tokens, 4),
                    torch.zeros(tokens, 4), torch.zeros(tokens, 4, 16),
                    layer_name=gdn.prefix,
                )
                reached = False
            except _ReachedKernel:
                reached = True
        assert (reads, reached) == expected, (metadata is None, reads, reached)
    assert warm_ups == [0]


def test_model_runner_phase_boundary() -> None:
    events = []

    @contextmanager
    def tracked_release():
        events.append("release")
        try:
            yield
        finally:
            events.append("restore")

    original_release = gpu_model_runner.release_reclaimable_workspaces
    gpu_model_runner.release_reclaimable_workspaces = tracked_release
    try:
        text_runner = SimpleNamespace()
        text_step = SimpleNamespace(scheduled_encoder_inputs={})
        assert gpu_model_runner.GPUModelRunner._execute_mm_encoder(
            text_runner, text_step
        ) == []
        assert events == []

        def encode(scheduler_output):
            assert scheduler_output.scheduled_encoder_inputs
            events.append("encode")
            return [torch.ones(1)]

        vision_runner = SimpleNamespace(
            _execute_mm_encoder_with_released_workspace=encode,
        )
        vision_step = SimpleNamespace(
            scheduled_encoder_inputs={"request": [0]},
        )
        outputs = gpu_model_runner.GPUModelRunner._execute_mm_encoder(
            vision_runner, vision_step
        )
        assert len(outputs) == 1
        assert events == ["release", "encode", "restore"]
    finally:
        gpu_model_runner.release_reclaimable_workspaces = original_release


class _Device:
    """A device whose allocator the profiler reads: what is allocated (named
    allocations plus the live workspaces), its peak, and free memory."""

    def __init__(self, total: int, others: int, manager) -> None:
        self.total, self.others, self.manager = total, others, manager
        self.non_torch = 0
        self.named: dict[str, int] = {}
        self.peak = 0

    def workspaces(self) -> int:
        size = self.manager._workspace_size_bytes
        reclaimable = self.manager._reclaimable_workspaces.values()
        return sum(size(ws) for ws in self.manager._current_workspaces) + sum(
            size(ws) for wss in reclaimable for ws in wss
        )

    def allocated(self) -> int:
        return sum(self.named.values()) + self.workspaces()

    def alloc(self, name: str, nbytes: int) -> None:
        assert name not in self.named, name
        self.named[name] = nbytes
        self.peak = max(self.peak, self.allocated())

    def free(self, name: str) -> None:
        del self.named[name]

    def stats(self, device=None) -> dict:
        self.peak = max(self.peak, self.allocated())
        return {
            "allocated_bytes.all.peak": self.peak,
            "allocated_bytes.all.current": self.allocated(),
        }

    def memory_info(self, device=None) -> tuple[int, int]:
        used = self.others + self.non_torch + self.allocated()
        return self.total - used, self.total

    def reset_peak(self, device=None) -> None:
        self.peak = self.allocated()


_CONTINUATION = "turboquant_continuation_prefill"


def _profiled_bound(
    sizes: dict[str, int], skipped: str | None = None
) -> tuple[int, int]:
    """Run the worker's derivation over the runner's profile, both as built,
    on a device where each phase allocates `sizes`; return the bound and
    what serving holds beside its pool at its worst moment. The pool holds
    two model layers and a drafter's; each model layer reads its metadata
    when the step built it, and the continuation phase requests the
    reclaimable workspace when one request holds the step -- except what
    `skipped` names ("layer" or "workspace")."""
    total, others, weights, non_torch = 1 << 30, 3 << 20, 600 << 20, 40 << 20
    reclaimable, primary = sizes["reclaimable"], 1 << 20
    builders, input_batch, per_block, blocks = 3 << 20, 2 << 20, 5 << 20, 2
    max_model_len, max_num_tokens, max_logprobs, vocab = 4096, 2048, 20, 1000
    model_layers = {
        "layers.0.self_attn.attn": object(),
        "layers.1.linear_attn": object(),
    }
    drafter_layer = "drafter.layers.0.self_attn.attn"
    forward_context = {**model_layers, drafter_layer: object()}

    manager = workspace.WorkspaceManager(torch.device("cpu"))
    device = _Device(total, others, manager)

    class EncoderCache(dict):
        def clear(self) -> None:
            if "encoder_outputs" in device.named:
                device.free("encoder_outputs")
            super().clear()

    def init_minimal(runner):
        device.alloc("stand_in", blocks * per_block)
        device.alloc("builders", builders)
        manager.get_simultaneous(((primary,), torch.uint8))
        manager.get_reclaimable_simultaneous(
            _CONTINUATION, ((reclaimable,), torch.uint8)
        )
        if "input_batch" not in device.named:
            device.alloc("input_batch", input_batch)
        runner.kv_cache_config = SimpleNamespace(
            num_blocks=blocks,
            kv_cache_groups=[SimpleNamespace(layer_names=list(forward_context))],
        )
        return runner.kv_cache_config

    def cleanup(runner) -> None:
        device.free("stand_in")
        device.free("builders")
        del runner.kv_cache_config

    def embed_multimodal(**inputs):
        assert manager._reclaimable_workspaces_released or not any(
            manager._reclaimable_workspace_sizes.values()
        ), "the encoder ran beside a reserved reclaimable workspace"
        device.alloc("encoder_transient", sizes["encoder"])
        device.alloc("encoder_outputs", sizes["encoder_outputs"])
        device.free("encoder_transient")
        return [torch.zeros(4, 8)]

    def dummy_run(runner, num_tokens, is_profile=False, force_attention=False,
                  profile_seq_lens=None, max_num_reqs=None, layer_reads=None):
        assert is_profile
        text = sizes["text"]
        if force_attention:
            # Attention executes against the stand-in pool, over the longest
            # context a request holds; its first call autotunes once.
            assert "stand_in" in device.named
            assert profile_seq_lens == max_model_len
            metadata = gpu_model_runner._record_layer_reads(
                {name: object() for name in forward_context}, layer_reads
            )
            for name in model_layers:
                if not (skipped == "layer" and name == "layers.1.linear_attn"):
                    metadata[name]
            if max_num_reqs == 1 and skipped != "workspace":
                manager.get_reclaimable_simultaneous(
                    _CONTINUATION, ((reclaimable,), torch.uint8)
                )
            text += sizes["attention"]
            if not runner.autotuned:
                runner.autotuned = True
                device.alloc("autotune_scratch", sizes["autotune"])
                device.free("autotune_scratch")
        device.alloc("text", text)
        device.free("text")
        return torch.zeros(max_num_tokens, 8), torch.zeros(1, 8)

    def sampler(runner, hidden_states, max_num_logprobs=None):
        # The sampler at the most log probabilities a request is admitted with.
        assert max_num_logprobs == max_logprobs, max_num_logprobs
        device.alloc("sampler", sizes["sampler"])
        device.free("sampler")

    def prompt_logprobs(target_ids, hidden_states, logits_fn, count, mode):
        # The step's prompt log probabilities at their largest: every prompt
        # row of the step, at the most a request is admitted with.
        assert len(target_ids) == hidden_states.shape[0] == max_num_tokens
        assert count == max_logprobs, count
        device.alloc("prompt_logprobs", sizes["prompt_logprobs"])
        device.free("prompt_logprobs")

    def profile_cudagraph_memory(runner) -> int:
        init_minimal(runner)
        cleanup(runner)
        return 0

    budget = SimpleNamespace(
        get_encoder_budget=lambda: 16384,
        mm_max_toks_per_item={"image": 16384},
        get_modality_with_max_tokens=lambda: "image",
        mm_max_items_per_batch={"image": 1},
    )
    shared = {
        name: getattr(gpu_model_runner.GPUModelRunner, name)
        for name in (
            "profile_run",
            "profile_served_phases",
            "profiling_kv_cache",
            "_run_dummy_encoder",
            "_run_dummy_text_step",
            "_require_text_step_ran",
            "_dummy_prompt_logprobs_run",
            "_largest_admitted_logprob_count",
        )
        if hasattr(gpu_model_runner.GPUModelRunner, name)
    }
    Runner = type("Runner", (), {
        **shared,
        "_init_minimal_kv_cache_for_profiling": init_minimal,
        "_cleanup_profiling_kv_cache": cleanup,
        "_dummy_run": dummy_run,
        "_dummy_sampler_run": sampler,
        "_get_mm_dummy_batch": lambda runner, modality, count: {},
        "_sync_device": lambda runner: None,
        "profile_cudagraph_memory": profile_cudagraph_memory,
    })
    runner = Runner()
    runner.autotuned = False
    runner.vllm_config = SimpleNamespace()
    runner.supports_mm_inputs = True
    runner.model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(skip_mm_profiling=False)
    )
    runner.mm_budget = budget
    runner.model_config.max_logprobs = max_logprobs
    runner.model_config.get_vocab_size = lambda: vocab
    runner.model_config.logprobs_mode = "raw_logprobs"
    runner.model = SimpleNamespace(
        embed_multimodal=embed_multimodal, compute_logits=None
    )
    runner.device = torch.device("cpu")
    runner.encoder_cache = EncoderCache()
    runner.get_model = lambda: SimpleNamespace(
        modules=lambda: iter(model_layers.values())
    )
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )
    runner.max_num_reqs = 4
    runner.max_num_tokens = max_num_tokens
    runner.max_model_len = max_model_len
    runner.is_pooling_model = False
    runner.model_memory_usage = weights

    accelerator = torch.accelerator
    saved = {
        name: getattr(accelerator, name)
        for name in ("memory_stats", "get_memory_info", "memory_reserved",
                     "empty_cache", "reset_peak_memory_stats")
    }
    saved_manager = workspace._manager
    saved_divisor = kv_cache_utils._pool_bytes_per_block
    saved_mem_platform = mem_utils.current_platform
    saved_worker_platform = gpu_worker.current_platform
    saved_reserve = gpu_worker.reserve_mm_ipc_gpu_memory
    saved_pp = gpu_model_runner.get_pp_group
    saved_prompt_logprobs = getattr(
        gpu_model_runner, "compute_prompt_logprobs_with_chunking", None
    )
    accelerator.memory_stats = device.stats
    accelerator.get_memory_info = device.memory_info
    accelerator.memory_reserved = lambda device_=None: device.allocated()
    accelerator.empty_cache = lambda: None
    accelerator.reset_peak_memory_stats = device.reset_peak
    workspace._manager = manager
    kv_cache_utils._pool_bytes_per_block = lambda config, groups: per_block
    mem_utils.current_platform = SimpleNamespace(
        is_integrated_gpu=lambda index: False, device_name="device"
    )
    gpu_worker.current_platform = SimpleNamespace(is_cuda_alike=lambda: True)
    gpu_worker.reserve_mm_ipc_gpu_memory = lambda bound, *args: bound
    gpu_model_runner.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    gpu_model_runner.compute_prompt_logprobs_with_chunking = prompt_logprobs
    try:
        init_snapshot = mem_utils.MemorySnapshot(device=torch.device("cpu"))
        device.alloc("weights", weights)
        device.non_torch = non_torch
        worker = SimpleNamespace(
            use_v2_model_runner=False,
            model_runner=runner,
            init_snapshot=init_snapshot,
            requested_memory=int(total * 0.92),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=None, gpu_memory_utilization=0.92
            ),
            model_config=SimpleNamespace(multimodal_config=None),
            parallel_config=SimpleNamespace(_api_process_count=1),
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
            ),
        )
        bound = gpu_worker.Worker.determine_available_memory(worker)
    finally:
        for name, fn in saved.items():
            setattr(accelerator, name, fn)
        workspace._manager = saved_manager
        kv_cache_utils._pool_bytes_per_block = saved_divisor
        mem_utils.current_platform = saved_mem_platform
        gpu_worker.current_platform = saved_worker_platform
        gpu_worker.reserve_mm_ipc_gpu_memory = saved_reserve
        gpu_model_runner.get_pp_group = saved_pp
        gpu_model_runner.compute_prompt_logprobs_with_chunking = saved_prompt_logprobs

    # Serving's worst moment beside its pool: the residents (weights,
    # non-torch, the primary workspace, the input batch, the attention
    # metadata builders) and the largest phase -- the encoder with the
    # reclaimable workspace released, or the text step, the sampler or a
    # chunk's prompt log probabilities with it resident -- each with the
    # encoder outputs it holds.
    phase = max(
        sizes["encoder"] + sizes["encoder_outputs"],
        reclaimable + sizes["encoder_outputs"] + sizes["text"] + sizes["attention"],
        reclaimable + sizes["encoder_outputs"] + sizes["sampler"],
        reclaimable + sizes["encoder_outputs"] + sizes["prompt_logprobs"],
    )
    held = weights + non_torch + primary + input_batch + builders + phase
    return bound, total - others - held


def test_kv_bound_refuses_a_text_step_that_skips_what_serving_runs() -> None:
    # The text step's own execution is the witness: a model layer that never
    # read its metadata, or a reserved reclaimable workspace never requested,
    # refuses startup by name. A drafter's layer is not the model's step.
    mib = 1 << 20
    sizes = {
        "reclaimable": 64 * mib, "encoder_outputs": 8 * mib, "text": 16 * mib,
        "attention": 6 * mib, "sampler": 4 * mib, "prompt_logprobs": 12 * mib,
        "autotune": 200 * mib, "encoder": 40 * mib,
    }
    for skipped, named in (
        ("layer", "layers.1.linear_attn"),
        ("workspace", repr(_CONTINUATION)),
    ):
        try:
            _profiled_bound(sizes, skipped=skipped)
        except RuntimeError as error:
            assert named in str(error), error
            assert "Next:" in str(error), error
        else:
            raise AssertionError(f"a text step that skipped the {skipped} was admitted")


def test_kv_bound_counts_each_phase_with_the_workspaces_it_holds() -> None:
    mib = 1 << 20
    base = {
        "reclaimable": 64 * mib,
        "encoder_outputs": 8 * mib,
        "text": 16 * mib,
        "attention": 6 * mib,
        "sampler": 4 * mib,
        "prompt_logprobs": 12 * mib,
        # Larger than every phase: it may not reach the bound.
        "autotune": 200 * mib,
    }
    for regime, sizes in (
        ("text step", {**base, "encoder": 40 * mib}),
        ("encoder", {**base, "encoder": 120 * mib}),
        ("sampler", {**base, "encoder": 40 * mib, "sampler": 30 * mib}),
        (
            "prompt log probabilities",
            {**base, "encoder": 40 * mib, "prompt_logprobs": 50 * mib},
        ),
    ):
        bound, room = _profiled_bound(sizes)
        assert bound == room, (
            f"{regime} phase dominating: the KV bound is {bound} B but serving "
            f"leaves {room} B beside its residents and the larger phase "
            f"({bound - room:+d} B)"
        )


def test_serving_scores_the_prompt_as_the_profile_measures() -> None:
    # Serving computes a chunk's prompt log probabilities through the one
    # function the startup profile runs at its largest, and through no
    # full-vocabulary computation of its own that the profile never runs.
    calls = []

    def computed(target_ids, hidden_states, logits_fn, count, mode):
        calls.append((target_ids.tolist(), hidden_states.shape[0], count, mode))
        rows = len(target_ids)
        return (
            torch.zeros(rows, count + 1, dtype=torch.int64),
            torch.zeros(rows, count + 1),
            torch.zeros(rows, dtype=torch.int64),
        )

    def unprofiled(*args, **kwargs):
        raise AssertionError(
            "prompt log probabilities were computed outside the function the "
            "startup profile measures"
        )

    request = SimpleNamespace(
        prompt_token_ids=[5, 6, 7, 8, 9],
        num_computed_tokens=0,
        in_progress_prompt_logprobs_cpu=None,
    )
    runner = SimpleNamespace(
        num_prompt_logprobs={"request": 2},
        requests={"request": request},
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(req_id_to_index={"request": 0}),
        query_start_loc=SimpleNamespace(np=np.array([0, 5])),
        model=SimpleNamespace(compute_logits=unprofiled),
        sampler=SimpleNamespace(
            compute_logprobs=unprofiled, gather_logprobs=unprofiled
        ),
        model_config=SimpleNamespace(logprobs_mode="raw_logprobs"),
        _sync_device=lambda: None,
    )
    saved = getattr(gpu_model_runner, "compute_prompt_logprobs_with_chunking", None)
    saved_h2d = gpu_model_runner.async_tensor_h2d
    gpu_model_runner.compute_prompt_logprobs_with_chunking = computed
    # A CPU stand-in for the pinned host-to-device copy.
    gpu_model_runner.async_tensor_h2d = lambda data, device, dtype=None: (
        torch.tensor(data, dtype=dtype)
    )
    try:
        scored = gpu_model_runner.GPUModelRunner._get_prompt_logprobs_dict(
            runner, torch.zeros(5, 8), {"request": 5}
        )
    finally:
        gpu_model_runner.compute_prompt_logprobs_with_chunking = saved
        gpu_model_runner.async_tensor_h2d = saved_h2d
    assert calls == [([6, 7, 8, 9], 4, 2, "raw_logprobs")], calls
    assert list(scored) == ["request"], scored


def test_v2_runner_holds_no_declared_pool() -> None:
    # The V2 runner's startup profile runs no attention and holds none of the
    # workspaces, CUDA graphs or log-probability memory serving holds beside
    # the pool, and nothing charges a holdback for them: a model V2 would
    # serve by default is served by V1 when it declares a pool, and V2 forced
    # is refused with the cause and a possible next action.
    def selection(kv_cache_users, architecture="Qwen3ForCausalLM",
                  only_v2=False, prefill_context_parallel_size=1):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="dense", architectures=[architecture],
                architecture=architecture, requires_v2_model_runner=only_v2,
                runner_type="generate", is_moe=False, is_quantized=False,
                is_diffusion=False, is_attention_free=False, use_mla=False,
                logits_processors=None, enable_prompt_embeds=False,
            ),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=prefill_context_parallel_size,
                tensor_parallel_size=1,
                pipeline_parallel_size=1, distributed_executor_backend="mp",
                enable_dbo=False, enable_elastic_ep=False,
            ),
            compilation_config=SimpleNamespace(
                mode=CompilationMode.VLLM_COMPILE,
                pass_config=SimpleNamespace(enable_sp=False),
            ),
            speculative_config=None,
            cache_config=SimpleNamespace(
                kv_sharing_fast_prefill=False, kv_cache_users=kv_cache_users
            ),
        )
        for name in ("_dflash_needs_multi_kv_group",
                     "_only_v2_model_runner_serves",
                     "_is_default_v2_model_runner_model",
                     "_get_v2_model_runner_unsupported_features"):
            setattr(config, name,
                    getattr(vllm_config.VllmConfig, name).__get__(config))
        return config

    # That a model only the V2 runner implements is the model's own
    # declaration, read by the registry as its other interfaces are.
    from vllm.model_executor.models.longcat_flash_ngram import (
        LongcatFlashNgramForCausalLM,
    )
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.model_executor.models.registry import _ModelInfo

    for model_cls, only_v2 in ((LongcatFlashNgramForCausalLM, True),
                               (Qwen3ForCausalLM, False)):
        info = _ModelInfo.from_model_cls(model_cls)
        assert info.requires_v2_model_runner is only_v2, (model_cls, info)

    selects_v2 = vllm_config.VllmConfig.use_v2_model_runner.fget
    saved_triton, saved_env = vllm_config.HAS_TRITON, os.environ.pop(
        "VLLM_USE_V2_MODEL_RUNNER", None
    )
    vllm_config.HAS_TRITON = True
    try:
        assert selects_v2(selection(None)) is True
        declared = selection(1)
        assert selects_v2(declared) is False, "V2 selected for a declared pool"
        try:
            vllm_config.VllmConfig._validate_v2_model_runner(declared)
        except ValueError as refusal:
            assert "a KV pool declared with --kv-cache-users" in str(refusal)
            assert "Next: " in str(refusal) and "V1 model runner" in str(refusal)
        else:
            raise AssertionError("V2 forced with a declared pool was admitted")
        # What only the V2 runner serves selects it, and the refusal names what
        # the same table lists: a launch option the operator can drop, after
        # which V1 serves the rest; a model only V2 implements, which no
        # launch of this server serves -- before its weights load, whether or
        # not the pool is declared, since a model with a KV cache is served
        # here only with one.
        pcp = selection(1, prefill_context_parallel_size=2)
        assert selects_v2(pcp) is True
        try:
            vllm_config.VllmConfig._validate_v2_model_runner(pcp)
        except ValueError as refusal:
            assert "serve without prefill context parallelism" in str(refusal)
            assert "so that the V1 model runner serves it" in str(refusal)
            assert "serve without the listed features" not in str(refusal)
        else:
            raise AssertionError("PCP with a declared pool was admitted on V2")
        for users in (1, None):
            longcat = selection(users, "LongcatFlashNgramForCausalLM", only_v2=True)
            assert selects_v2(longcat) is True, "a V2-only model routed to V1"
            (need,) = [need for need, _ in longcat._only_v2_model_runner_serves()]
            try:
                vllm_config.VllmConfig._validate_v2_model_runner(longcat)
            except ValueError as refusal:
                assert need in str(refusal), (need, str(refusal))
                assert "no launch of this server serves this model" in str(refusal)
                assert "V1 model runner" not in str(refusal)
            else:
                raise AssertionError("a V2-only model with a KV cache was admitted")
    finally:
        vllm_config.HAS_TRITON = saved_triton
        if saved_env is not None:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = saved_env


def test_kv_bound_refusal_names_only_possible_actions() -> None:
    # A declaration beyond the bound is refused with the actions that can make
    # it fit; declaring fewer users is one only while more than one is.
    from vllm.config import CacheConfig

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float32
    )
    for users, fewer in ((1, False), (4, True)):
        cache = CacheConfig()
        cache.kv_cache_users = users
        config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, original_max_model_len=64),
            cache_config=cache,
            speculative_config=None,
            kv_transfer_config=None,
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1, prefill_context_parallel_size=1
            ),
        )
        # One context fits; the declared users and the null block do not.
        bound = spec.page_size_bytes * (4 * users + 1) - 1
        try:
            kv_cache_utils.get_kv_cache_configs(config, [{"layer": spec}], [bound])
        except ValueError as refusal:
            message = str(refusal)
        else:
            raise AssertionError(f"--kv-cache-users {users} beyond the bound was admitted")
        assert f"--kv-cache-users {users} requires" in message, message
        assert "Next: free the device memory other processes hold" in message, message
        assert ("declare fewer users" in message) is fewer, message
        assert message.endswith("or reduce max_model_len."), message


def test_startup_plan_is_keyed_on_the_code_that_derived_it() -> None:
    # A persisted plan replaces the profile; builds that patch one upstream
    # commit share its version string, so a plan recorded by other code --
    # another derivation of the bound -- must never match.
    import tempfile
    from pathlib import Path

    import vllm
    from vllm.v1.worker import startup_plan

    platform = SimpleNamespace(
        get_device_name=lambda device_id=0: "device",
        get_device_total_memory=lambda device_id=0: 1 << 30,
        get_device_capability=lambda device_id=0: (12, 0),
    )
    config = SimpleNamespace(compute_hash=lambda: "config")
    digest = getattr(startup_plan, "installed_source_digest", None)
    clear = getattr(digest, "cache_clear", lambda: None)
    saved_platform, saved_file = startup_plan.current_platform, vllm.__file__
    fingerprints = []
    with tempfile.TemporaryDirectory() as root:
        package = Path(root) / "vllm"
        package.mkdir()
        (package / "__init__.py").write_text("")
        source = package / "gpu_worker.py"
        startup_plan.current_platform = platform
        vllm.__file__ = str(package / "__init__.py")
        try:
            for derivation in ("bound = 1\n", "bound = 2\n"):
                source.write_text(derivation)
                clear()
                fingerprints.append(
                    startup_plan.compute_plan_fingerprint(config, 0, 1)
                )
        finally:
            startup_plan.current_platform = saved_platform
            vllm.__file__ = saved_file
            clear()
    assert fingerprints[0] != fingerprints[1], (
        "a startup plan recorded by other code would be adopted"
    )


def test_profiled_context_covers_its_own_query() -> None:
    # The served profile gives every dummy request the longest context a
    # request holds; a request whose own share of the step is longer keeps a
    # context that covers its query.
    from contextlib import nullcontext

    class Stop(Exception):
        pass

    for tokens, seqs, context, expected in (
        (2048, 1, 262144, [262144]),
        (29, 3, 10, [10, 10, 11]),
    ):
        captured = []

        def copy_(source, non_blocking=False, expected=expected):
            captured.extend(source[: len(expected)].tolist())
            raise Stop

        step = SimpleNamespace(num_tokens=tokens, num_reqs=None)
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(multimodal_config=None),
                parallel_config=SimpleNamespace(num_ubatches=1),
            ),
            uniform_decode_query_len=1,
            max_num_tokens=tokens,
            scheduler_config=SimpleNamespace(max_num_seqs=seqs),
            _determine_batch_execution_and_padding=lambda **kwargs: (
                CUDAGraphMode.NONE, step, False, None, None
            ),
            dcp_world_size=1,
            parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
            _get_slot_mappings=lambda **kwargs: (None, None),
            synchronize_input_prep=nullcontext,
            optimistic_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
            seq_lens=SimpleNamespace(copy_=copy_),
        )
        try:
            gpu_model_runner.GPUModelRunner._dummy_run(
                runner, tokens, is_profile=True, force_attention=True,
                profile_seq_lens=context,
            )
        except Stop:
            pass
        assert captured == expected, (tokens, seqs, context, captured)


def test_dummy_step_records_the_layers_that_read_its_metadata() -> None:
    # The dummy text step builds attention metadata only when attention is
    # forced, and hands the layers a record of which of them read their own
    # entry; it splits its tokens over at most the requests it is given.
    from contextlib import nullcontext

    import numpy as np

    class Stop(Exception):
        pass

    layer = "layers.0.self_attn.attn"
    built = []

    def build_attention_metadata(**kwargs):
        built.append(kwargs["num_reqs"])
        return {layer: object()}, None

    captured = []

    def set_forward_context(attn_metadata, *args, **kwargs):
        captured.append(attn_metadata)
        raise Stop

    saved_context = gpu_model_runner.set_forward_context
    saved_pp = gpu_model_runner.get_pp_group
    gpu_model_runner.set_forward_context = set_forward_context
    gpu_model_runner.get_pp_group = lambda: SimpleNamespace(is_first_rank=True)
    try:
        for force_attention, max_num_reqs, seq_lens in (
            (True, None, [4096, 4096, 4096, 4096]),
            (True, 1, [4096]),
            (False, None, None),
        ):
            built.clear()
            captured.clear()
            reads: set[str] = set()
            copied = []
            step = SimpleNamespace(num_tokens=2048, num_reqs=None)
            runner = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(multimodal_config=None),
                    parallel_config=SimpleNamespace(num_ubatches=1),
                ),
                uniform_decode_query_len=1,
                max_num_tokens=2048,
                scheduler_config=SimpleNamespace(max_num_seqs=4),
                _determine_batch_execution_and_padding=lambda **kwargs: (
                    CUDAGraphMode.NONE, step, False, None, None
                ),
                dcp_world_size=1,
                parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
                _get_slot_mappings=lambda **kwargs: (None, None),
                synchronize_input_prep=nullcontext,
                optimistic_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
                seq_lens=SimpleNamespace(
                    copy_=lambda source, non_blocking=False: copied.append(
                        source.tolist()
                    )
                ),
                _get_cumsum_and_arange=lambda tokens, out: np.cumsum(tokens),
                query_pos=SimpleNamespace(np=np.zeros(4096, dtype=np.int64)),
                query_start_loc=SimpleNamespace(
                    np=np.zeros(9, dtype=np.int32), copy_to_gpu=lambda: None
                ),
                input_batch=SimpleNamespace(
                    block_table=SimpleNamespace(commit_block_table=lambda n: None)
                ),
                positions=torch.zeros(2048, dtype=torch.int64),
                _build_attention_metadata=build_attention_metadata,
                speculative_config=None,
                lora_config=None,
                maybe_dummy_run_with_lora=lambda *args: nullcontext(),
                _init_model_kwargs=lambda: {},
                supports_mm_inputs=False,
                enable_prompt_embeds=False,
                input_ids=SimpleNamespace(gpu=torch.zeros(2048, dtype=torch.int64)),
                uses_mrope=False,
                uses_xdrope_dim=0,
                maybe_randomize_inputs=lambda *args, **kwargs: nullcontext(),
            )
            try:
                gpu_model_runner.GPUModelRunner._dummy_run(
                    runner, 2048, is_profile=True, force_attention=force_attention,
                    profile_seq_lens=4096 if force_attention else None,
                    max_num_reqs=max_num_reqs, layer_reads=reads,
                )
            except Stop:
                pass
            (metadata,) = captured
            if not force_attention:
                assert metadata is None and not built and not copied
                continue
            assert built == [len(seq_lens)], built
            assert copied[0][: len(seq_lens)] == seq_lens, copied
            assert reads == set()
            metadata[layer]
            assert reads == {layer}
    finally:
        gpu_model_runner.set_forward_context = saved_context
        gpu_model_runner.get_pp_group = saved_pp


if __name__ == "__main__":
    test_workspace_lifetime()
    test_graph_capture_refuses_reclaimable_views()
    test_turboquant_reservation_routing()
    test_served_paths_witness_their_own_execution()
    test_model_runner_phase_boundary()
    test_kv_bound_counts_each_phase_with_the_workspaces_it_holds()
    test_kv_bound_refuses_a_text_step_that_skips_what_serving_runs()
    test_serving_scores_the_prompt_as_the_profile_measures()
    test_v2_runner_holds_no_declared_pool()
    test_kv_bound_refusal_names_only_possible_actions()
    test_startup_plan_is_keyed_on_the_code_that_derived_it()
    test_profiled_context_covers_its_own_query()
    test_dummy_step_records_the_layers_that_read_its_metadata()
    print("vision workspace unit: passed")
