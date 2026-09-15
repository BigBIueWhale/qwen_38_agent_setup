#!/usr/bin/env python3
"""Prove the image recipe carries every reviewed runtime mutation."""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patches.source_patch_v1.generated_vllm_stages import FINAL_FILES, GENERATED_STAGES


class RuntimeImageTest(unittest.TestCase):
    def test_every_reviewed_runtime_file_is_copied_and_verified(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        joined = recipe.replace("\\\n", " ")
        copies = dict(re.findall(r"^COPY --chmod=0644\s+(\S+)\s+(\S+)$", joined, re.M))
        allowed = set((ROOT / ".dockerignore").read_text().splitlines())
        build = (ROOT / "scripts/build-vllm.sh").read_text()
        runtime = (ROOT / "scripts/runtime-common.sh").read_text()
        for path in FINAL_FILES:
            if not path.startswith("vllm/"):
                continue
            with self.subTest(path=path):
                source = f"vllm/{path}"
                installed = f"/usr/local/lib/python3.12/dist-packages/{path}"
                self.assertEqual(copies.get(source), installed)
                self.assertIn("!" + source, allowed)
                # Both the upstream verifier and the final layer hash this path.
                self.assertGreaterEqual(recipe.count(installed), 3)
                self.assertGreaterEqual(build.count(installed), 2)
                self.assertGreaterEqual(runtime.count(installed), 2)

    def test_reviewed_deletions_cannot_survive_the_runtime_layer(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        for stage in GENERATED_STAGES:
            for entry in stage["files"]:
                path = entry["path"]
                if path.startswith("vllm/") and entry["after_sha256"] is None:
                    # The removed CPU-policy package is deleted as a whole.
                    self.assertTrue(path.startswith("vllm/v1/kv_offload/cpu/policies/"))
                    self.assertIn(
                        "test ! -e /usr/local/lib/python3.12/dist-packages/"
                        "vllm/v1/kv_offload/cpu/policies", recipe,
                    )

    def test_kernel_guard_is_executed_during_build(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        self.assertIn(
            "RUN CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 "
            "python3 /opt/qwen38/turboquant_guard_unit.py", recipe,
        )


if __name__ == "__main__":
    unittest.main()
