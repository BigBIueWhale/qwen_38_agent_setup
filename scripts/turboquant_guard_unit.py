#!/usr/bin/env python3
"""Execute the installed K8V4 store guards on CPU with Triton's interpreter.

Run in a fresh process with TRITON_INTERPRET=1 and CUDA_VISIBLE_DEVICES=.
This exercises real kernel code at build time; GPU numerical acceptance remains
in turboquant_k8v4_unit.py and is required separately at release.
"""

import os
import unittest
from unittest.mock import patch

import torch

from vllm.v1.attention.ops import triton_turboquant_decode as decode
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store


class TurboQuantGuardTest(unittest.TestCase):
    def test_store_preserves_finite_bytes_and_poisons_invalid_metadata(self):
        self.assertEqual(os.environ.get("TRITON_INTERPRET"), "1")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        # Every lane is an exact quantization level, away from rounding ties.
        value = torch.arange(256, dtype=torch.float32).remainder(16)
        values = value.repeat(7, 4, 1)
        for row, invalid in enumerate((float("nan"), float("inf"), -float("inf")), 1):
            values[row, 0, 7] = invalid
        values[4, 0] = float("nan")
        values[5, 0] = -70000.0  # Minimum cannot be represented as finite fp16.
        values[6, 0, 7] = 1000000.0  # Scale cannot be represented as finite fp16.
        keys = torch.ones_like(values)
        cache = torch.zeros((1, 16, 4, 388), dtype=torch.uint8)
        triton_turboquant_store(
            keys, values, cache, torch.arange(7, dtype=torch.int32),
            torch.eye(256), torch.arange(15, dtype=torch.float32),
            mse_bits=4, key_packed_size=256, value_quant_bits=4, key_fp8=True,
        )
        expected_values = (
            value.to(torch.uint8)[::2] | (value.to(torch.uint8)[1::2] << 4)
        )
        for row in range(7):
            for head in range(4):
                with self.subTest(row=row, head=head):
                    torch.testing.assert_close(
                        cache[0, row, head, :256],
                        keys[row, head].to(torch.float8_e4m3fn).view(torch.uint8),
                        rtol=0, atol=0,
                    )
                    metadata = cache[0, row, head, 384:388].view(torch.float16)
                    if row != 0 and head == 0:
                        self.assertTrue(torch.isnan(metadata).all().item())
                    else:
                        torch.testing.assert_close(
                            cache[0, row, head, 256:384], expected_values,
                            rtol=0, atol=0,
                        )
                        torch.testing.assert_close(
                            metadata, torch.tensor([1.0, 0.0], dtype=torch.float16),
                            rtol=0, atol=0,
                        )

    def test_decode_refuses_an_incompatible_key_format(self):
        with patch.dict(decode._FP8_E4B15, {}, clear=True), patch.object(
            decode.current_platform, "is_cuda_alike", return_value=True
        ), patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            with self.assertRaisesRegex(RuntimeError, "Refusing a silent fp8e4b15"):
                decode._use_fp8_e4b15()
            self.assertNotIn(0, decode._FP8_E4B15)

    def test_decode_keeps_e4m3_on_supported_devices(self):
        for capability in ((8, 9), (9, 0), (12, 0)):
            with self.subTest(capability=capability), patch.dict(
                decode._FP8_E4B15, {}, clear=True
            ), patch.object(
                decode.current_platform, "is_cuda_alike", return_value=True
            ), patch.object(torch.cuda, "get_device_capability", return_value=capability):
                self.assertEqual(decode._use_fp8_e4b15(), 0)


if __name__ == "__main__":
    unittest.main()
