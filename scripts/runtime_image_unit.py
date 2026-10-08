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
        build = (ROOT / "scripts/build-vllm.sh").read_text()
        runtime = (ROOT / "scripts/runtime-common.sh").read_text()
        for path in FINAL_FILES:
            if not path.startswith("vllm/"):
                continue
            with self.subTest(path=path):
                source = f"vllm/{path}"
                installed = f"/usr/local/lib/python3.12/dist-packages/{path}"
                self.assertEqual(copies.get(source), installed)
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
        # The parser unit parses on the served model's tokenizer, which the
        # image does not carry, so the build's unit loop runs it -- in check and
        # in build -- with the manifest-checked files and the launch's parsers.
        script = (ROOT / "scripts/build-vllm.sh").read_text()
        loop = next(
            line for line in script.splitlines() if line.startswith("for unit in ")
        )
        self.assertIn(" tool_output_parser_unit ", loop)
        self.assertIn('"${served_model_mounts[@]}"', script)
        self.assertIn('"${served_parser_env[@]}"', script)
        for option in ("--reasoning-parser", "--tool-call-parser",
                       "--default-chat-template-kwargs"):
            self.assertIn(f"$(launch_arg_value {option})", script)

    def test_probe_requests_unit_is_executed_during_check(self):
        # The probes are not in the image; check runs every probe's offline
        # request construction from the project, against the reviewed runtime.
        script = (ROOT / "scripts/build-vllm.sh").read_text()
        self.assertIn('--volume "${PROJECT_DIR}/scripts:/probes:ro" '
                      '"${parser_unit_mounts[@]}"', script)
        self.assertIn('"${BASE_IMAGE_TAG}" /probes/probe_requests_unit.py', script)
        # A failure fails the gate: nothing after the unit discards its status.
        self.assertRegex(script, r'/probes/probe_requests_unit\.py\n')

    def test_shared_prefix_unit_is_executed_during_build(self):
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        self.assertIn(
            "RUN CUDA_VISIBLE_DEVICES= python3 /opt/qwen38/shared_prefix_cache_unit.py",
            recipe,
        )

    def test_every_shipped_unit_is_executed(self):
        # A unit the image ships and nothing runs reports as coverage it never
        # gives: qwen38_context_unit, which needs only the served config, was
        # shipped and hashed for weeks and run by nothing, and the two GPU units
        # were run by nothing until a build ran them. A unit runs in the image's
        # own layers, in the build's CPU unit loop, or -- only if it needs the
        # GPU -- in the image a build has made, on the GPU, before the build
        # pins it. Those are named here, with the reason, so no other unit can
        # quietly join them and leave the loop that check runs.
        needs_the_card = {
            # Triton store and fused decode on CUDA, against PyTorch references.
            "turboquant_k8v4_unit",
            # The checkpoint's worst-error NVFP4 layer on an SM 12.0 card.
            "nvfp4_kernel_unit",
        }
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        script = (ROOT / "scripts/build-vllm.sh").read_text()
        shipped = set(re.findall(
            r"^COPY --chmod=0644 scripts/(\w+_unit)\.py /opt/qwen38/", recipe, re.M,
        ))
        in_image = set(re.findall(
            r"^RUN [^\n]*python3 /opt/qwen38/(\w+_unit)\.py", recipe, re.M,
        ))
        loop = next(
            line for line in script.splitlines() if line.startswith("for unit in ")
        )
        in_check = set(loop.split(" in ", 1)[1].split(";", 1)[0].split())
        in_check |= set(re.findall(r"/context/scripts/(\w+_unit)\.py", script))
        runtime = (ROOT / "scripts/runtime-common.sh").read_text()
        on_the_card = set(re.search(
            r"^readonly -a GPU_RELEASE_UNITS=\(([^)]*)\)$", runtime, re.M,
        ).group(1).split())
        self.assertEqual(on_the_card, needs_the_card)
        self.assertEqual(sorted(shipped - in_image - in_check), sorted(on_the_card))
        self.assertEqual(on_the_card & (in_image | in_check), set())

    def test_grammar_unit_is_executed_during_build(self):
        # The image must run the shipped file, not a copy of its assertions:
        # `build-vllm.sh check` runs the same path, so a recipe that stopped
        # executing it would leave the two able to drift again.
        recipe = (ROOT / "containers/Dockerfile.runtime").read_text()
        self.assertIn(
            "RUN CUDA_VISIBLE_DEVICES= python3 /opt/qwen38/qwen_grammar_unit.py",
            recipe,
        )



class BuildScriptTest(unittest.TestCase):
    """The build reads the verified reconstruction, never the vllm/ checkout."""

    def setUp(self):
        self.script = (ROOT / "scripts/build-vllm.sh").read_text()

    def test_the_mode_argument_is_required(self):
        self.assertNotIn('MODE="${1:-build}"', self.script)
        self.assertIn('MODE="$1"', self.script)

    def test_usage_names_every_mode(self):
        usage = self.script.split("<<'USAGE'", 1)[1].split("USAGE", 1)[0]
        for mode in ("build", "check", "serve-check", "materialise"):
            with self.subTest(mode=mode):
                self.assertIn(f"\n  {mode}", usage)

    def test_serving_refuses_an_image_not_built_from_these_inputs(self):
        # The refusal comes from the run that just derived the inputs, after
        # every check and before anything is served.
        summary = self.script.split('if [[ "${MODE}" == "check" || "${MODE}" == "serve-check" ]]; then', 1)[1]
        summary = summary.split("\n  exit 0\n", 1)[0]
        self.assertIn(
            'require_image_built_from_inputs "${image_inputs_sha256}" "${context_file_count}"',
            summary,
        )
        runtime = (ROOT / "scripts/runtime-common.sh").read_text()
        self.assertIn('"${COMMON_SCRIPT_DIR}/build-vllm.sh" serve-check\n', runtime)
        self.assertNotIn('"${COMMON_SCRIPT_DIR}/build-vllm.sh" check\n', runtime)

    def test_the_vllm_checkout_is_reached_only_through_git(self):
        # The submodule directory is used only as the repository that holds the
        # pinned commit; no path into its checked-out files appears, so nothing
        # an edit there leaves can reach a check, a context or an image.
        self.assertNotIn('"${VLLM_DIR}/', self.script)
        self.assertEqual(
            self.script.count("${VLLM_DIR}"),
            self.script.count('git -C "${VLLM_DIR}"'),
        )

    def test_a_build_pins_only_an_image_whose_gpu_units_passed(self):
        # The GPU units run in the image the build has just made, after its
        # installed bytes and label are verified and before anything pins it or
        # moves the runtime tag to it; a host that cannot run them is refused
        # before the image is built. What the gate itself does is the runtime
        # contract test's.
        script = self.script
        gate = script.index('\nrun_gpu_release_units "${actual_image_id}"\n')
        self.assertEqual(script.count('run_gpu_release_units "'), 1)
        self.assertLess(
            script.index("Built image carries the wrong runtime profile label."), gate)
        self.assertLess(gate, script.index('settle_produced_identity pin_outcome "${RUNTIME_LOCK}"'))
        self.assertLess(gate, script.index('docker tag "${actual_image_id}" "${IMAGE_TAG}"'))
        early = script.index(
            'if [[ "${MODE}" == build ]]; then\n  check_host_prerequisites\nfi\n')
        self.assertLess(early, script.index("docker buildx build"))
        self.assertLess(early, script.index("BUILD_EXPORT_DIR=\"$(mktemp"))

    def test_the_image_is_built_from_the_assembled_context(self):
        invocation = self.script.split("docker buildx build", 1)[1]
        invocation = invocation.split("\ndocker load", 1)[0]
        self.assertIn('--file "${BUILD_CONTEXT}/containers/Dockerfile.runtime"', invocation)
        self.assertTrue(invocation.rstrip().endswith('"${BUILD_CONTEXT}"'))
        self.assertNotIn("${PROJECT_DIR}", invocation)


if __name__ == "__main__":
    unittest.main()
