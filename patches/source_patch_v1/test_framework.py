"""Failure-semantics tests for the transactional source patch framework."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import framework
from .framework import (
    FileIdentity,
    LandmarkEdit,
    PatchRefusedError,
    PatchSet,
    PatchStage,
    PatchWriteError,
    SourcePatchTransaction,
    require_python_symbols,
    sha256_bytes,
    sha256_text,
)


def _noop(_state) -> None:
    return None


def _review(path: str, before: str, after: str) -> str:
    old_lines = before.count("\n")
    new_lines = after.count("\n")
    old = "".join(f"-{line}" for line in before.splitlines(keepends=True))
    new = "".join(f"+{line}" for line in after.splitlines(keepends=True))
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,{old_lines} +1,{new_lines} @@\n"
        f"{old}{new}"
    )


def _stage(
    artifact_root: Path,
    *,
    name: str,
    transformations: tuple[tuple[str, str, str], ...],
    validate_before=_noop,
    validate_after=_noop,
) -> PatchStage:
    review_path = f"{name}.patch"
    review = "".join(
        _review(path, before, after) for path, before, after in transformations
    )
    (artifact_root / review_path).write_text(
        review, encoding="utf-8", newline="\n"
    )
    return PatchStage(
        name=name,
        rationale="Synthetic defect used to prove transaction failure semantics.",
        removal_condition="Remove when this framework test is removed.",
        review_patch=review_path,
        review_sha256=sha256_bytes(review.encode("utf-8")),
        files=tuple(
            FileIdentity(
                path=path,
                before_sha256=sha256_text(before) if before else None,
                after_sha256=sha256_text(after),
            )
            for path, before, after in transformations
        ),
        edits=tuple(
            LandmarkEdit(
                name=f"{name}:{path}",
                path=path,
                before=before,
                after=after,
                review_before=before,
                review_after=after,
            )
            for path, before, after in transformations
        ),
        validate_before=validate_before,
        validate_after=validate_after,
    )


def _patchset(
    stages: tuple[PatchStage, ...],
    *,
    final_files: dict[str, str],
    final_validator=_noop,
) -> PatchSet:
    return PatchSet(
        name="synthetic-transaction",
        source_revision="synthetic-revision",
        identity_files={"identity.txt": sha256_text("pinned-revision\n")},
        stages=stages,
        final_files=final_files,
        validate_final=final_validator,
    )


class SourcePatchTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="source-patch-test-")
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.artifact = root / "artifact"
        self.source.mkdir()
        self.artifact.mkdir()
        (self.source / "identity.txt").write_text(
            "pinned-revision\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_apply_is_exact_idempotent_and_preserves_mode(self) -> None:
        before = "def value():\n    return 1\n"
        after = "def value():\n    return 2\n"
        target = self.source / "module.py"
        target.write_text(before, encoding="utf-8")
        target.chmod(0o750)

        def validate_final(state) -> None:
            require_python_symbols(
                state,
                "module.py",
                {"value": ()},
                label="synthetic final contract",
            )

        stage = _stage(
            self.artifact,
            name="change-value",
            transformations=(("module.py", before, after),),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset(
                (stage,),
                final_files={"module.py": sha256_text(after)},
                final_validator=validate_final,
            ),
        )

        first = transaction.apply()
        first_stat = target.stat()
        first_bytes = target.read_bytes()
        second = transaction.apply()
        second_stat = target.stat()

        self.assertEqual(first.state, "applied")
        self.assertEqual(second.state, "already-applied")
        self.assertEqual(second.changed_files, ())
        self.assertEqual(first_bytes, after.encode())
        self.assertEqual(first_stat.st_mode, second_stat.st_mode)
        self.assertEqual(first_stat.st_mtime_ns, second_stat.st_mtime_ns)
        self.assertEqual(first_stat.st_ctime_ns, second_stat.st_ctime_ns)
        self.assertEqual(first_stat.st_ino, second_stat.st_ino)
        self.assertEqual(first_stat.st_mode & 0o777, 0o750)

    def test_new_file_hunk_must_describe_and_create_complete_file(self) -> None:
        after = "def created():\n    return True\n"
        stage = _stage(
            self.artifact,
            name="create-file",
            transformations=(("created.py", "", after),),
        )
        patchset = _patchset(
            (stage,), final_files={"created.py": sha256_text(after)}
        )

        result = SourcePatchTransaction(
            self.source, self.artifact, patchset
        ).apply()

        self.assertEqual(result.state, "applied")
        self.assertEqual(
            (self.source / "created.py").read_text(encoding="utf-8"), after
        )
        self.assertEqual((self.source / "created.py").stat().st_mode & 0o777, 0o644)

    def test_unknown_source_drift_refuses_without_writes(self) -> None:
        before = "def value():\n    return 1\n"
        after = "def value():\n    return 2\n"
        drift = "def value():\n    return 999\n"
        target = self.source / "module.py"
        target.write_text(drift, encoding="utf-8")
        stage = _stage(
            self.artifact,
            name="change-value",
            transformations=(("module.py", before, after),),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset((stage,), final_files={"module.py": sha256_text(after)}),
        )
        before_stat = target.stat()

        with self.assertRaisesRegex(
            PatchRefusedError, "neither wholly pristine nor wholly final"
        ):
            transaction.apply()

        after_stat = target.stat()
        self.assertEqual(target.read_text(encoding="utf-8"), drift)
        self.assertEqual(before_stat.st_mtime_ns, after_stat.st_mtime_ns)
        self.assertEqual(before_stat.st_ctime_ns, after_stat.st_ctime_ns)
        self.assertFalse(tuple(self.source.rglob("*.qwen-source-patch.*")))

    def test_exact_intermediate_state_refuses_without_finishing_it(self) -> None:
        first = "def value():\n    return 1\n"
        intermediate = "def value():\n    return 2\n"
        final = "def value():\n    return 3\n"
        target = self.source / "module.py"
        target.write_text(intermediate, encoding="utf-8")
        stage_one = _stage(
            self.artifact,
            name="stage-one",
            transformations=(("module.py", first, intermediate),),
        )
        stage_two = _stage(
            self.artifact,
            name="stage-two",
            transformations=(("module.py", intermediate, final),),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset(
                (stage_one, stage_two),
                final_files={"module.py": sha256_text(final)},
            ),
        )
        prior = target.stat()

        with self.assertRaisesRegex(PatchRefusedError, "module.py=intermediate"):
            transaction.apply()

        current = target.stat()
        self.assertEqual(target.read_text(encoding="utf-8"), intermediate)
        self.assertEqual(prior.st_mtime_ns, current.st_mtime_ns)
        self.assertEqual(prior.st_ctime_ns, current.st_ctime_ns)

    def test_review_artifact_drift_refuses_before_source_writes(self) -> None:
        before = "def value():\n    return 1\n"
        after = "def value():\n    return 2\n"
        target = self.source / "module.py"
        target.write_text(before, encoding="utf-8")
        stage = _stage(
            self.artifact,
            name="change-value",
            transformations=(("module.py", before, after),),
        )
        (self.artifact / stage.review_patch).write_text("drift\n", encoding="utf-8")
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset((stage,), final_files={"module.py": sha256_text(after)}),
        )
        prior = target.stat()

        with self.assertRaisesRegex(PatchRefusedError, "review diff SHA-256 drift"):
            transaction.apply()

        current = target.stat()
        self.assertEqual(target.read_text(encoding="utf-8"), before)
        self.assertEqual(prior.st_mtime_ns, current.st_mtime_ns)
        self.assertEqual(prior.st_ctime_ns, current.st_ctime_ns)

    def test_source_change_between_plan_and_commit_is_refused(self) -> None:
        before = "def value():\n    return 1\n"
        after = "def value():\n    return 2\n"
        target = self.source / "module.py"
        target.write_text(before, encoding="utf-8")
        stage = _stage(
            self.artifact,
            name="change-value",
            transformations=(("module.py", before, after),),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset((stage,), final_files={"module.py": sha256_text(after)}),
        )
        planned, result = transaction.plan()
        target.write_text("def value():\n    return 7\n", encoding="utf-8")

        with self.assertRaisesRegex(
            PatchRefusedError, "neither wholly pristine nor wholly final"
        ):
            transaction.commit(planned, result)

        self.assertIn("return 7", target.read_text(encoding="utf-8"))

    def test_second_replace_failure_rolls_back_first_file_and_modes(self) -> None:
        a_before = "def a():\n    return 1\n"
        a_after = "def a():\n    return 2\n"
        b_before = "def b():\n    return 1\n"
        b_after = "def b():\n    return 2\n"
        (self.source / "a.py").write_text(a_before, encoding="utf-8")
        (self.source / "b.py").write_text(b_before, encoding="utf-8")
        (self.source / "a.py").chmod(0o740)
        (self.source / "b.py").chmod(0o640)
        stage = _stage(
            self.artifact,
            name="two-files",
            transformations=(
                ("a.py", a_before, a_after),
                ("b.py", b_before, b_after),
            ),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset(
                (stage,),
                final_files={
                    "a.py": sha256_text(a_after),
                    "b.py": sha256_text(b_after),
                },
            ),
        )
        real_replace = os.replace

        def fail_second_patch_temp(src, dst) -> None:
            if Path(dst).name == "b.py" and ".qwen-source-patch." in Path(src).name:
                raise OSError("injected second replacement failure")
            real_replace(src, dst)

        with patch.object(framework.os, "replace", fail_second_patch_temp):
            with self.assertRaisesRegex(
                PatchWriteError, "caller must discard this disposable tree"
            ):
                transaction.apply()

        self.assertEqual((self.source / "a.py").read_text(), a_before)
        self.assertEqual((self.source / "b.py").read_text(), b_before)
        self.assertEqual((self.source / "a.py").stat().st_mode & 0o777, 0o740)
        self.assertEqual((self.source / "b.py").stat().st_mode & 0o777, 0o640)
        self.assertFalse(tuple(self.source.rglob("*.qwen-source-patch.*")))
        self.assertFalse(tuple(self.source.rglob("*.qwen-rollback.*")))


if __name__ == "__main__":
    unittest.main()
