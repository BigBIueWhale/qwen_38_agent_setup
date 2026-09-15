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
    def test_recipe_hashes_match_the_reviewed_source_identities(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text().replace("\\\n", " ")
        pins = dict(re.findall(
            r'readonly (\w+)="([0-9a-f]{64})"',
            (ROOT / "config/runtime-v1.sh").read_text(),
        ))
        upstream = {}
        for stage in GENERATED_STAGES:
            for entry in stage["files"]:
                upstream.setdefault(entry["path"], entry["before_sha256"])
        checked = set()
        for name, kind, path in re.findall(
            r'"\$\{(\w+_(UPSTREAM|PATCHED)_FILE_SHA256)\}"\s+'
            r'/usr/local/lib/python3.12/dist-packages/(\S+)', recipe,
        ):
            with self.subTest(name=name, path=path):
                expected = upstream[path] if kind == "UPSTREAM" else FINAL_FILES[path]
                self.assertEqual(pins[name], expected)
                checked.add((kind, path))
        for path in FINAL_FILES:
            if path.startswith("vllm/"):
                for kind in ("UPSTREAM", "PATCHED"):
                    self.assertIn((kind, path), checked)

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

    def test_parser_unit_is_executed_during_build(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        self.assertIn(
            "RUN CUDA_VISIBLE_DEVICES= "
            "python3 /opt/qwen38/tool_output_parser_unit.py", recipe,
        )


if __name__ == "__main__":
    unittest.main()
