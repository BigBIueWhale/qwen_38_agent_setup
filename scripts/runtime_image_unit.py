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
                self.assertIn(("PATCHED", path), checked)
                if upstream[path] is None:
                    verifier = recipe.split("FROM upstream-verifier AS runtime", 1)[0]
                    self.assertIn(
                        "RUN test ! -e /usr/local/lib/python3.12/dist-packages/" + path,
                        verifier,
                    )
                else:
                    self.assertIn(("UPSTREAM", path), checked)

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
                    # The CPU policies are removed as a package; other
                    # superseded modules must be individually absent.
                    removed = (
                        "vllm/v1/kv_offload/cpu/policies"
                        if path.startswith("vllm/v1/kv_offload/cpu/policies/")
                        else path
                    )
                    self.assertIn(
                        "test ! -e /usr/local/lib/python3.12/dist-packages/"
                        + removed, recipe,
                    )

    def test_every_declared_build_argument_is_supplied(self):
        # An ARG the recipe declares but the build never passes expands to the
        # empty string inside the image. That turns a hash assertion into a
        # malformed sha256sum line rather than a mismatch, and `check` cannot
        # catch it because it does not execute the recipe -- which is how
        # QWEN_GRAMMAR_UNIT_SHA256 reached a build and failed it at step 151.
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        build = (ROOT / "scripts/build-vllm.sh").read_text()
        passed = set(re.findall(r'--build-arg "(\w+)=', build))
        # SOURCE_DATE_EPOCH is BuildKit's own reproducibility argument; it is
        # honoured by the builder, not declared by the recipe.
        builtin = {"SOURCE_DATE_EPOCH"}
        declared, valueless = set(), set()
        for line in recipe.splitlines():
            match = re.match(r"ARG\s+(\w+)(=.*)?$", line.strip())
            if match:
                declared.add(match.group(1))
                if match.group(2) is None:
                    valueless.add(match.group(1))
        # An ARG with no default and no value is the empty string downstream.
        self.assertEqual(sorted(valueless - passed), [])
        # An argument the recipe never declares is silently discarded, so a
        # misspelled one would never reach the layer that asserts on it.
        self.assertEqual(sorted(passed - declared - builtin), [])

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

    def test_shared_prefix_unit_is_executed_during_build(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        self.assertIn(
            "RUN CUDA_VISIBLE_DEVICES= python3 /opt/qwen38/shared_prefix_cache_unit.py",
            recipe,
        )

    def test_grammar_unit_is_executed_during_build(self):
        # The image must run the shipped file, not a copy of its assertions:
        # `build-vllm.sh check` runs the same path, so a recipe that stopped
        # executing it would leave the two able to drift again.
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        self.assertIn(
            "RUN CUDA_VISIBLE_DEVICES= python3 /opt/qwen38/qwen_grammar_unit.py",
            recipe,
        )


if __name__ == "__main__":
    unittest.main()
