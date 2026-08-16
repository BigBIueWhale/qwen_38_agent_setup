#!/usr/bin/env python3
"""Build-time checks for the Qwen3 vision-MLP peak-memory optimization."""

import torch
from torch import nn

from vllm.model_executor.models.qwen3_vl import Qwen3_VisionMLP


class _FakeFc1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.output = value.clone()
        return self.output


class _FakeFc2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_ptr: int | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.input_ptr = value.data_ptr()
        return value


def _make_mlp(act_fn: nn.Module) -> Qwen3_VisionMLP:
    # Linear construction requires a distributed vLLM context.  The behavior
    # under test begins after fc1, so install deterministic stand-ins while
    # retaining the production class's actual forward implementation.
    mlp = Qwen3_VisionMLP.__new__(Qwen3_VisionMLP)
    nn.Module.__init__(mlp)
    mlp.linear_fc1 = _FakeFc1()
    mlp.linear_fc2 = _FakeFc2()
    mlp.act_fn = act_fn
    mlp._use_inplace_gelu_tanh = (
        isinstance(act_fn, nn.GELU) and act_fn.approximate == "tanh"
    )
    return mlp


def test_inference_alias_and_exactness() -> None:
    mlp = _make_mlp(nn.GELU(approximate="tanh"))
    source = torch.linspace(-5, 5, 4096, dtype=torch.bfloat16).reshape(64, 64)
    expected = nn.functional.gelu(source.clone(), approximate="tanh")

    with torch.inference_mode():
        output = mlp(source)

    fc1_output = mlp.linear_fc1.output
    assert fc1_output is not None
    assert mlp._use_inplace_gelu_tanh
    assert mlp.linear_fc2.input_ptr == fc1_output.data_ptr()
    assert output.data_ptr() == fc1_output.data_ptr()
    assert torch.equal(output, expected)


def test_other_activations_are_unchanged() -> None:
    mlp = _make_mlp(nn.SiLU())
    source = torch.linspace(-3, 3, 1024, dtype=torch.float32).reshape(32, 32)
    expected = nn.functional.silu(source)

    with torch.inference_mode():
        output = mlp(source)

    assert not mlp._use_inplace_gelu_tanh
    assert torch.equal(output, expected)


def test_autograd_uses_out_of_place_path() -> None:
    mlp = _make_mlp(nn.GELU(approximate="tanh"))
    source = torch.linspace(-2, 2, 256, dtype=torch.float32, requires_grad=True)
    output = mlp(source)
    output.sum().backward()

    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


if __name__ == "__main__":
    test_inference_alias_and_exactness()
    test_other_activations_are_unchanged()
    test_autograd_uses_out_of_place_path()
    print("vision MLP unit: passed")
