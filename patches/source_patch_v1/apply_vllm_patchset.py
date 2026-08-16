#!/usr/bin/env python3
"""Apply the one pinned vLLM patch set transactionally.

There is intentionally no fuzzy, force, partial, skip, or compatibility mode.
The source must be wholly pristine or already equal to the complete reviewed
result.  Every other state is a typed refusal with no planned writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts_vllm import CONTRACTS, validate_final
from .framework import (
    FileIdentity,
    LandmarkEdit,
    PatchRefusedError,
    PatchSet,
    PatchStage,
    PatchWriteError,
    SourcePatchTransaction,
)
from .generated_vllm_stages import (
    FINAL_FILES,
    GENERATED_STAGES,
    IDENTITY_FILES,
    SOURCE_REVISION,
)


def build_patchset() -> PatchSet:
    generated_names = tuple(stage["name"] for stage in GENERATED_STAGES)
    contract_names = tuple(CONTRACTS)
    if set(generated_names) != set(contract_names):
        raise PatchRefusedError(
            "vLLM generated stages and semantic contracts disagree: "
            f"generated={generated_names!r}, contracts={contract_names!r}"
        )
    if len(generated_names) != len(set(generated_names)):
        raise PatchRefusedError(f"duplicate generated stage: {generated_names!r}")

    stages: list[PatchStage] = []
    for generated in GENERATED_STAGES:
        name = generated["name"]
        contract = CONTRACTS[name]
        stages.append(
            PatchStage(
                name=name,
                rationale=contract.rationale,
                removal_condition=contract.removal_condition,
                review_patch=generated["review_patch"],
                review_sha256=generated["review_sha256"],
                files=tuple(FileIdentity(**item) for item in generated["files"]),
                edits=tuple(LandmarkEdit(**item) for item in generated["edits"]),
                validate_before=contract.validate_before,
                validate_after=contract.validate_after,
            )
        )
    return PatchSet(
        name="qwen38-vllm-source-patch-v1",
        source_revision=SOURCE_REVISION,
        identity_files=IDENTITY_FILES,
        stages=tuple(stages),
        final_files=FINAL_FILES,
        validate_final=validate_final,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply or idempotently verify the sole pinned, landmark-aware vLLM "
            "source transformation."
        )
    )
    parser.add_argument(
        "source_root",
        type=Path,
        help="private disposable vLLM source tree to validate and transform",
    )
    parser.add_argument(
        "artifact_root",
        type=Path,
        help="project root containing the exact reviewed patch artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        patchset = build_patchset()
        transaction = SourcePatchTransaction(
            args.source_root,
            args.artifact_root,
            patchset,
        )
        result = transaction.apply()
    except PatchRefusedError as exc:
        print(f"SOURCE PATCH REFUSED: {exc}", file=sys.stderr)
        return 1
    except PatchWriteError as exc:
        print(f"SOURCE PATCH WRITE FAILURE: {exc}", file=sys.stderr)
        return 1

    print(
        f"{patchset.name}: {result.state}; source revision "
        f"{patchset.source_revision}; stages={len(result.stages)}; "
        f"changed_files={len(result.changed_files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
