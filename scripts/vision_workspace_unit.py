#!/usr/bin/env python3
"""Build-time invariants for phase-safe multimodal workspace reuse."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.utils import mem_utils
from vllm.v1.attention.backends import turboquant_attn
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


def _profiled_bound(sizes: dict[str, int]) -> tuple[int, int]:
    """Run the worker's derivation over the runner's profile, both as built,
    on a device where each phase allocates `sizes`; return the bound and
    what serving holds beside its pool at its worst moment."""
    total, others, weights, non_torch = 1 << 30, 3 << 20, 600 << 20, 40 << 20
    reclaimable, primary = sizes["reclaimable"], 1 << 20
    builders, input_batch, per_block, blocks = 3 << 20, 2 << 20, 5 << 20, 2
    max_model_len = 4096

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
            "turboquant_continuation_prefill", ((reclaimable,), torch.uint8)
        )
        if "input_batch" not in device.named:
            device.alloc("input_batch", input_batch)
        return SimpleNamespace(num_blocks=blocks, kv_cache_groups=["layer-group"])

    def cleanup(runner) -> None:
        device.free("stand_in")
        device.free("builders")

    def embed_multimodal(**inputs):
        assert manager._reclaimable_workspaces_released or not any(
            manager._reclaimable_workspace_sizes.values()
        ), "the encoder ran beside a reserved reclaimable workspace"
        device.alloc("encoder_transient", sizes["encoder"])
        device.alloc("encoder_outputs", sizes["encoder_outputs"])
        device.free("encoder_transient")
        return [torch.zeros(4, 8)]

    def dummy_run(runner, num_tokens, is_profile=False, force_attention=False,
                  profile_seq_lens=None):
        assert is_profile
        text = sizes["text"]
        if force_attention:
            # Attention executes against the stand-in pool, over the longest
            # context a request holds; its first call autotunes once.
            assert "stand_in" in device.named
            assert profile_seq_lens == max_model_len
            text += sizes["attention"]
            if not runner.autotuned:
                runner.autotuned = True
                device.alloc("autotune_scratch", sizes["autotune"])
                device.free("autotune_scratch")
        device.alloc("text", text)
        device.free("text")
        return torch.zeros(4, 8), torch.zeros(1, 8)

    def sampler(runner, hidden_states):
        device.alloc("sampler", sizes["sampler"])
        device.free("sampler")

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
    runner.model = SimpleNamespace(embed_multimodal=embed_multimodal)
    runner.encoder_cache = EncoderCache()
    runner.max_num_tokens = 2048
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

    # Serving's worst moment beside its pool: the residents (weights,
    # non-torch, the primary workspace, the input batch, the attention
    # metadata builders) and the larger phase -- the encoder with the
    # reclaimable workspace released, or the text step or sampler with it
    # resident -- each with the encoder outputs it holds.
    phase = max(
        sizes["encoder"] + sizes["encoder_outputs"],
        reclaimable + sizes["encoder_outputs"] + sizes["text"] + sizes["attention"],
        reclaimable + sizes["encoder_outputs"] + sizes["sampler"],
    )
    held = weights + non_torch + primary + input_batch + builders + phase
    return bound, total - others - held


def test_kv_bound_counts_each_phase_with_the_workspaces_it_holds() -> None:
    mib = 1 << 20
    base = {
        "reclaimable": 64 * mib,
        "encoder_outputs": 8 * mib,
        "text": 16 * mib,
        "attention": 6 * mib,
        "sampler": 4 * mib,
        # Larger than every phase: it may not reach the bound.
        "autotune": 200 * mib,
    }
    for regime, sizes in (
        ("text step", {**base, "encoder": 40 * mib}),
        ("encoder", {**base, "encoder": 120 * mib}),
        ("sampler", {**base, "encoder": 40 * mib, "sampler": 30 * mib}),
    ):
        bound, room = _profiled_bound(sizes)
        assert bound == room, (
            f"{regime} phase dominating: the KV bound is {bound} B but serving "
            f"leaves {room} B beside its residents and the larger phase "
            f"({bound - room:+d} B)"
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


if __name__ == "__main__":
    test_workspace_lifetime()
    test_graph_capture_refuses_reclaimable_views()
    test_turboquant_reservation_routing()
    test_model_runner_phase_boundary()
    test_kv_bound_counts_each_phase_with_the_workspaces_it_holds()
    test_profiled_context_covers_its_own_query()
    print("vision workspace unit: passed")
