#!/usr/bin/env python3
"""Prove that an NVFP4 W4A4 layer gets a native FP4 kernel or a refusal at load.

The served checkpoint's MLPs are NVFP4 W4A4. vLLM's automatic kernel selection
used to fall through to Marlin (W4A16, 16-bit activations) or emulation on a
GPU without native FP4, warning and serving different arithmetic. Selection
now refuses at layer construction, naming the device capability and why each
native kernel is unavailable, unless the operator names the substitute with
--linear-backend. VLLM_BATCH_INVARIANT=1 is held to the same rule: its
deterministic kernel is the native CUTLASS one, and emulation is served only
when named. This runs selection on CPU with each kernel's availability stated,
as `turboquant_guard_unit.py` does for its hardware guard.
"""

from __future__ import annotations

import os

import vllm.model_executor.kernels.linear as linear
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_NVFP4_KERNELS,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel
from vllm.model_executor.kernels.linear.nvfp4.emulation import (
    EmulationNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
from vllm.platforms import PlatformEnum
from vllm.platforms.interface import DeviceCapability

KERNELS = _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA]
ABSENT = object()


def select(
    capability: tuple[int, int],
    supported: set[type],
    backend: str = "auto",
    batch_invariant: bool = False,
):
    """Run selection on a CUDA device of `capability` where only `supported` run."""
    platform = linear.current_platform
    patched = [
        (platform, "_enum"),
        (platform, "get_device_capability"),
        (linear, "_get_linear_backend"),
    ]
    patched += [
        (kernel, name)
        for kernel in KERNELS
        for name in ("is_supported", "can_implement")
    ]
    saved = [(owner, name, vars(owner).get(name, ABSENT)) for owner, name in patched]
    platform._enum = PlatformEnum.CUDA
    platform.get_device_capability = lambda device_id=0: DeviceCapability(*capability)
    linear._get_linear_backend = lambda: backend
    for kernel in KERNELS:
        result = (
            (True, None)
            if kernel in supported
            else (False, "unavailable on this device")
        )
        kernel.is_supported = classmethod(
            lambda cls, compute_capability=None, r=result: r
        )
        kernel.can_implement = classmethod(lambda cls, config: (True, None))
    if batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    try:
        return init_nvfp4_linear_kernel()
    finally:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)
        for owner, name, value in saved:
            if value is ABSENT:
                delattr(owner, name)
            else:
                setattr(owner, name, value)


def main() -> None:
    # A GPU without native FP4 (an H200): Marlin and the FlashInfer kernels that
    # check no capability are available; nothing native is.
    try:
        select((9, 0), {MarlinNvFp4LinearKernel, FlashInferTrtllmNvFp4LinearKernel})
    except ValueError as refusal:
        message = str(refusal)
        assert "native FP4 kernel" in message, message
        assert "compute capability 9.0" in message, message
        assert (
            "FlashInferCutlassNvFp4LinearKernel: unavailable on this device" in message
        ), message
        assert "MarlinNvFp4LinearKernel, does not compute in FP4" in message, message
    else:
        raise AssertionError("a GPU without native FP4 was served a substitute kernel")

    # The served card selects the native kernel the GPU unit measures.
    kernel = select(
        (12, 0), {FlashInferCutlassNvFp4LinearKernel, MarlinNvFp4LinearKernel}
    )
    assert isinstance(kernel, FlashInferCutlassNvFp4LinearKernel), type(kernel)

    # A substitute the operator names is served as named.
    kernel = select((9, 0), {MarlinNvFp4LinearKernel}, backend="marlin")
    assert isinstance(kernel, MarlinNvFp4LinearKernel), type(kernel)

    # VLLM_BATCH_INVARIANT on a GPU without the native CUTLASS kernel: emulation
    # is a substitute, refused unless named, whatever the environment sets.
    try:
        kernel = select(
            (9, 0),
            {MarlinNvFp4LinearKernel, EmulationNvFp4LinearKernel},
            batch_invariant=True,
        )
    except ValueError as refusal:
        message = str(refusal)
        assert "VLLM_BATCH_INVARIANT needs" in message, message
        assert "compute capability 9.0" in message, message
        assert "EmulationNvFp4LinearKernel, does not compute in FP4" in message, message
        assert "unset VLLM_BATCH_INVARIANT" in message, message
    else:
        raise AssertionError(
            f"VLLM_BATCH_INVARIANT served {type(kernel).__name__} on a GPU without "
            "native FP4"
        )
    kernel = select(
        (9, 0), {EmulationNvFp4LinearKernel}, backend="emulation", batch_invariant=True
    )
    assert isinstance(kernel, EmulationNvFp4LinearKernel), type(kernel)
    kernel = select(
        (12, 0),
        {CutlassNvFp4LinearKernel, EmulationNvFp4LinearKernel},
        batch_invariant=True,
    )
    assert isinstance(kernel, CutlassNvFp4LinearKernel), type(kernel)
    print(
        "NATIVE_FP4_SELECTION_UNIT_OK refused=9.0 native=12.0 named=marlin "
        "batch-invariant=refused-9.0,named-emulation,cutlass-12.0"
    )


if __name__ == "__main__":
    main()
