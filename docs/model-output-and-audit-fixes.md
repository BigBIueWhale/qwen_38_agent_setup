# Model-output and deployment audit resolutions

This record accompanies the unchanged `qwen38-deployment-correctness-audit.md`.
The independent triage and handling policy supplied with the implementation brief
decide the resolutions. Source validation here does not certify a built image or a
live release. The v23 image and archive are awaiting adoption.

## Finding 10: ship and execute the TurboQuant guards

The runtime Dockerfile now copies both reviewed non-SoA kernels and verifies their
upstream and patched hashes. The context explicitly admits every copied input.
The build and runtime installed-file checks cover all reviewed runtime sources;
the runtime check previously also omitted the twenty KV sizing/scope files that
the image already copied and the build verified.

The new `scripts/turboquant_guard_unit.py` executes the installed store kernel with
Triton's CPU interpreter. It tests unchanged finite key/value/metadata bytes at
D=256 and four KV heads, isolated and all-NaN vectors, both infinities, and fp16
metadata overflow. It separately tests the installed decode helper's refusal of
SM 8.0 and its E4M3 result on SM 8.9, 9.0 and 12.0 with mocked capabilities. No
CUDA device or model is used. The Dockerfile runs this test in a fresh process
with interpretation enabled; this environment applies only to that build test.

Executing the test exposed an additional defect in the original guard: Triton's
minimum/maximum reductions use non-propagating NaN semantics, so an isolated NaN
can leave both extrema finite. The reviewed guard transformation now checks for
NaN in every active lane before accepting the metadata. Finite lanes still use
the same arithmetic and produce the same bytes.

`scripts/runtime_image_unit.py` checks that every reviewed runtime mutation has
its copy, context entry, build verification and runtime verification, and that
the guard test actually runs in the Dockerfile. This prevents a future label from
claiming a source mutation the image does not contain. All runtime shell-contract
tests invoked by `build-vllm.sh check` now execute in a disposable container too.
Its disposable worktree/export paths honor the standard `TMPDIR` variable.

Validation: `TMPDIR=/tmp/codex-fix ./scripts/build-vllm.sh check` passed with
16 framework/recipe tests, the exact template derivation and thirteen-stage source
reconstruction, and three kernel-guard tests. The untouched base image is a
negative control: the same kernel-guard suite must fail against its unpatched
store and decode modules. Triton interpreter warnings on intentionally invalid
floating-point input are retained as evidence.

The full `turboquant_k8v4_unit.py` GPU store/fused-decode numerical acceptance
remains required at release. CPU interpretation proves these guards and the
packaging regression; it does not prove GPU code generation or performance.

## Remaining implementation

Parser language, stop handling, schema conversion, protocol translation, image
admission, remaining triage findings, client recovery verification and final
backend/client pin adoption are still under review. Their completion is not
implied by the finding-10 validation above.
