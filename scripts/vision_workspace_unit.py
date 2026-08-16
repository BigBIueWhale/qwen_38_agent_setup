#!/usr/bin/env python3
"""Build-time invariants for phase-safe multimodal workspace reuse."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends import turboquant_attn
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker import gpu_model_runner, workspace


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


def test_turboquant_reservation_routing() -> None:
    calls = []

    class FakeWorkspaceManager:
        def get_simultaneous(self, *shapes_and_dtypes):
            calls.append(("primary", None, shapes_and_dtypes))

        def get_reclaimable_simultaneous(self, name, *shapes_and_dtypes):
            calls.append(("reclaimable", name, shapes_and_dtypes))

        def reserve_raw_cuda_headroom(self, name, size):
            calls.append(("raw-headroom", name, size))

    original_manager = turboquant_attn.current_workspace_manager
    original_initialized = turboquant_attn.is_workspace_manager_initialized
    original_headroom = turboquant_attn.envs.VLLM_QWEN38_VISION_HEADROOM_BYTES
    turboquant_attn.current_workspace_manager = lambda: FakeWorkspaceManager()
    turboquant_attn.is_workspace_manager_initialized = lambda: True
    turboquant_attn.envs.VLLM_QWEN38_VISION_HEADROOM_BYTES = 1024
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
        turboquant_attn.envs.VLLM_QWEN38_VISION_HEADROOM_BYTES = original_headroom

    assert [call[0] for call in calls] == [
        "primary",
        "reclaimable",
        "raw-headroom",
    ]
    assert calls[1][1] == "turboquant_continuation_prefill"
    assert calls[1][2] == (
        ((1, 4, 8192, 128), torch.float16),
        ((1, 4, 8192, 128), torch.float16),
    )
    assert calls[2] == (
        "raw-headroom",
        "qwen38_vision_encoder_headroom",
        1024,
    )


def test_raw_cuda_headroom_lifetime() -> None:
    allocations = []
    frees = []
    original_allocate = workspace._raw_cuda_malloc_committed
    original_free = workspace._raw_cuda_free
    original_empty_cache = torch.accelerator.empty_cache
    workspace._raw_cuda_malloc_committed = lambda size: (
        allocations.append(size) or (0x1000 + len(allocations))
    )
    workspace._raw_cuda_free = frees.append
    torch.accelerator.empty_cache = lambda: None
    try:
        manager = workspace.WorkspaceManager(torch.device("cuda"))
        manager.reserve_raw_cuda_headroom("physical-headroom", 1024)
        manager.lock()
        assert manager.release_reclaimable_workspaces() == 1024
        try:
            manager.reserve_raw_cuda_headroom("physical-headroom", 1024)
        except AssertionError as exc:
            assert "Model phases must not overlap" in str(exc)
        else:
            raise AssertionError("released raw headroom was accessible")
        assert manager.restore_reclaimable_workspaces() == 1024
        assert allocations == [1024, 1024]
        assert frees == [0x1001]
        manager.reserve_raw_cuda_headroom("physical-headroom", 1024)
        assert allocations == [1024, 1024]
        try:
            manager.reserve_raw_cuda_headroom("physical-headroom", 2048)
        except AssertionError as exc:
            assert "changed size" in str(exc)
        else:
            raise AssertionError("raw headroom size change was accepted")
    finally:
        workspace._raw_cuda_malloc_committed = original_allocate
        workspace._raw_cuda_free = original_free
        torch.accelerator.empty_cache = original_empty_cache


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


if __name__ == "__main__":
    test_workspace_lifetime()
    test_turboquant_reservation_routing()
    test_raw_cuda_headroom_lifetime()
    test_model_runner_phase_boundary()
    print("vision workspace unit: passed")
