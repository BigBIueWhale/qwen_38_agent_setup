"""Failure-semantics tests for the transactional source patch framework."""

from __future__ import annotations

import ast
import os
import sys
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
    git_blob_id,
    sha256_bytes,
    sha256_text,
)


def _noop(_state) -> None:
    return None


def _deletion_review(path: str, before: str) -> str:
    """git-style deletion. An empty ``before`` is the hunkless form: the
    'deleted file mode' header is then the entire review evidence."""
    header = (
        f"diff --git a/{path} b/{path}\n"
        f"deleted file mode 100644\n"
    )
    if before == "":
        return header
    old_lines = before.count("\n")
    old = "".join(f"-{line}" for line in before.splitlines(keepends=True))
    return (
        header
        + f"--- a/{path}\n"
        + "+++ /dev/null\n"
        + f"@@ -1,{old_lines} +0,0 @@\n"
        + old
    )


def _deletion_stage(
    artifact_root: Path,
    *,
    name: str,
    deletions: tuple[tuple[str, str], ...],
    review_override: str | None = None,
) -> PatchStage:
    review_path = f"{name}.patch"
    review = (
        review_override
        if review_override is not None
        else "".join(_deletion_review(path, before) for path, before in deletions)
    )
    (artifact_root / review_path).write_text(review, encoding="utf-8", newline="\n")
    return PatchStage(
        name=name,
        rationale="Synthetic deletion used to prove transaction failure semantics.",
        removal_condition="Remove when this framework test is removed.",
        review_patch=review_path,
        review_sha256=sha256_bytes(review.encode("utf-8")),
        files=tuple(
            FileIdentity(
                path=path,
                before_sha256=sha256_text(before),
                after_sha256=None,
            )
            for path, before in deletions
        ),
        edits=tuple(
            LandmarkEdit(
                name=f"{name}:{path}",
                path=path,
                before=before,
                after="",
                review_before=before,
                review_after="",
            )
            for path, before in deletions
            if before != ""
        ),
        validate_before=_noop,
        validate_after=_noop,
    )


def _review(path: str, before: str, after: str) -> str:
    """git-style whole-file review; a created file is declared as one."""
    old_lines = before.count("\n")
    new_lines = after.count("\n")
    old = "".join(f"-{line}" for line in before.splitlines(keepends=True))
    new = "".join(f"+{line}" for line in after.splitlines(keepends=True))
    if before == "":
        return (
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{path}\n"
            f"@@ -0,0 +1,{new_lines} @@\n"
            f"{new}"
        )
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

    def _twin_stage(self, *, review: str, landmark_first: bool) -> PatchStage:
        """A file with two identical blocks; the stage changes one of them."""
        block = "handler()\n    value = 1\n"
        changed = "handler()\n    value = 2\n"
        before = "# first\n" + block + "# second\n" + block
        after = (
            "# first\n" + changed + "# second\n" + block
            if landmark_first
            else "# first\n" + block + "# second\n" + changed
        )
        marker = "# first\n" if landmark_first else "# second\n"
        (self.source / "twin.txt").write_text(before, encoding="utf-8")
        (self.artifact / "twin.patch").write_text(review, encoding="utf-8", newline="\n")
        return PatchStage(
            name="twin",
            rationale="Synthetic placement used to prove the proof of placement.",
            removal_condition="Remove when this framework test is removed.",
            review_patch="twin.patch",
            review_sha256=sha256_bytes(review.encode("utf-8")),
            files=(FileIdentity("twin.txt", sha256_text(before), sha256_text(after)),),
            edits=(
                LandmarkEdit(
                    name="twin.txt:landmark-1",
                    path="twin.txt",
                    before=marker + block,
                    after=marker + changed,
                    review_before=block,
                    review_after=changed,
                ),
            ),
            validate_before=_noop,
            validate_after=_noop,
        ), after

    def test_a_hunk_must_land_at_the_line_its_header_names(self) -> None:
        # The review names the second block (line 5); the data's landmark
        # changes the first (line 2). Both describe the same blocks, so only
        # placement tells them apart.
        review = (
            "diff --git a/twin.txt b/twin.txt\n--- a/twin.txt\n+++ b/twin.txt\n"
            "@@ -5,2 +5,2 @@\n handler()\n-    value = 1\n+    value = 2\n"
        )
        stage, after = self._twin_stage(review=review, landmark_first=True)
        patchset = _patchset((stage,), final_files={"twin.txt": sha256_text(after)})
        with self.assertRaisesRegex(
            PatchRefusedError, r"lands at line 2 of twin\.txt, but its review "
            r"hunk's header places it at line 5"
        ):
            SourcePatchTransaction(self.source, self.artifact, patchset).apply()
        self.assertNotIn("value = 2", (self.source / "twin.txt").read_text())

        stage, after = self._twin_stage(review=review, landmark_first=False)
        patchset = _patchset((stage,), final_files={"twin.txt": sha256_text(after)})
        SourcePatchTransaction(self.source, self.artifact, patchset).apply()
        self.assertEqual((self.source / "twin.txt").read_text(), after)

    def test_an_index_line_must_name_the_blobs_the_stage_transforms(self) -> None:
        block = "handler()\n    value = 1\n"
        before = "# first\n" + block + "# second\n" + block
        stage, after = self._twin_stage(review="", landmark_first=False)
        for named_after, refused in ((git_blob_id(after), False), ("0" * 40, True)):
            review = (
                "diff --git a/twin.txt b/twin.txt\n"
                f"index {git_blob_id(before)[:10]}..{named_after[:10]} 100644\n"
                "--- a/twin.txt\n+++ b/twin.txt\n"
                "@@ -5,2 +5,2 @@\n handler()\n-    value = 1\n+    value = 2\n"
            )
            stage, after = self._twin_stage(review=review, landmark_first=False)
            patchset = _patchset(
                (stage,), final_files={"twin.txt": sha256_text(after)}
            )
            transaction = SourcePatchTransaction(self.source, self.artifact, patchset)
            if refused:
                with self.assertRaisesRegex(PatchRefusedError, "index line for twin.txt"):
                    transaction.plan()
            else:
                self.assertEqual(transaction.plan()[1].state, "planned")

    def test_a_created_file_must_be_declared_as_one(self) -> None:
        after = "def created():\n    return True\n"
        stage = _stage(
            self.artifact, name="create-file", transformations=(("created.py", "", after),)
        )
        undeclared = _review("created.py", "", after).replace("new file mode 100644\n", "")
        (self.artifact / stage.review_patch).write_text(undeclared, encoding="utf-8")
        stage = PatchStage(**{
            **stage.__dict__, "review_sha256": sha256_bytes(undeclared.encode("utf-8"))
        })
        patchset = _patchset((stage,), final_files={"created.py": sha256_text(after)})
        with self.assertRaisesRegex(
            PatchRefusedError, "created.py's path pair says created, but its header does not"
        ):
            SourcePatchTransaction(self.source, self.artifact, patchset).apply()
        self.assertFalse((self.source / "created.py").exists())

    def test_the_compiler_anchors_a_hunk_at_its_header_line(self) -> None:
        sys.path.insert(0, str(Path(framework.__file__).parent))
        try:
            import compile_review_diff as compiler
        finally:
            sys.path.pop(0)
        block, changed = "handler()\n    value = 1\n", "handler()\n    value = 2\n"
        current = "# first\n" + block + "# second\n" + block
        before, after = compiler.expand_to_unique_landmark(current, block, changed, 5)
        self.assertEqual(
            current.replace(before, after, 1),
            "# first\n" + block + "# second\n" + changed,
        )
        with self.assertRaisesRegex(
            compiler.PatchRefusedError, "does not start at line 4"
        ):
            compiler.expand_to_unique_landmark(current, block, changed, 4)

    def _landmark_patchset(
        self, *, source: str, result: str, edit: LandmarkEdit, hunk: str
    ) -> PatchSet:
        """A one-hunk stage over module.py, its source written in place."""
        review = (
            "diff --git a/module.py b/module.py\n--- a/module.py\n+++ b/module.py\n"
            + hunk
        )
        (self.source / "module.py").write_text(source, encoding="utf-8")
        (self.artifact / "landmark.patch").write_text(
            review, encoding="utf-8", newline="\n"
        )
        stage = PatchStage(
            name="landmark",
            rationale="Synthetic landmark used to prove the after-block rule.",
            removal_condition="Remove when this framework test is removed.",
            review_patch="landmark.patch",
            review_sha256=sha256_bytes(review.encode("utf-8")),
            files=(
                FileIdentity("module.py", sha256_text(source), sha256_text(result)),
            ),
            edits=(edit,),
            validate_before=_noop,
            validate_after=_noop,
        )
        return _patchset((stage,), final_files={"module.py": sha256_text(result)})

    def test_a_deletion_that_ends_a_file_compiles_and_applies(self) -> None:
        sys.path.insert(0, str(Path(framework.__file__).parent))
        try:
            import compile_review_diff as compiler
        finally:
            sys.path.pop(0)
        source = "def kept():\n    return 1\n\n\ndef dropped():\n    return 2\n"
        result = "def kept():\n    return 1\n"
        review_before = "    return 1\n\n\ndef dropped():\n    return 2\n"
        review_after = "    return 1\n"
        # What the deletion keeps is part of what it replaces: the block itself,
        # not a sign that the source already holds the result.
        before, after = compiler.expand_to_unique_landmark(
            source, review_before, review_after, 2
        )
        self.assertEqual(source.replace(before, after, 1), result)
        edit = LandmarkEdit(
            name="module.py:landmark-1", path="module.py", before=before,
            after=after, review_before=review_before, review_after=review_after,
        )
        hunk = "@@ -2,5 +2,1 @@\n     return 1\n-\n-\n-def dropped():\n-    return 2\n"
        patchset = self._landmark_patchset(
            source=source, result=result, edit=edit, hunk=hunk
        )
        self.assertEqual(
            SourcePatchTransaction(self.source, self.artifact, patchset).apply().state,
            "applied",
        )
        self.assertEqual((self.source / "module.py").read_text(), result)

    def test_an_after_block_outside_its_landmark_is_refused(self) -> None:
        # The same trailing deletion, over a source that already holds the text
        # it keeps elsewhere: that copy is outside the landmark, so refused.
        kept = "def other():\n    return 1\n\n\n"
        source = kept + "def kept():\n    return 1\n\n\ndef dropped():\n    return 2\n"
        review_before = "    return 1\n\n\ndef dropped():\n    return 2\n"
        trailing = LandmarkEdit(
            name="module.py:landmark-1", path="module.py", before=review_before,
            after="    return 1\n", review_before=review_before,
            review_after="    return 1\n",
        )
        # A partial application of an ordinary change: its result is already
        # in the file.
        partial = LandmarkEdit(
            name="module.py:landmark-1", path="module.py", before="value = 1\n",
            after="value = 2\n", review_before="value = 1\n",
            review_after="value = 2\n",
        )
        # A result that straddles the landmark's edge is outside it too.
        straddling = LandmarkEdit(
            name="module.py:landmark-1", path="module.py", before="b\nc\n",
            after="c\nd\n", review_before="b\nc\n", review_after="c\nd\n",
        )
        trailing_hunk = (
            "@@ -6,5 +6,1 @@\n     return 1\n-\n-\n-def dropped():\n-    return 2\n"
        )
        cases = (
            (source, trailing, trailing_hunk),
            ("value = 2\nvalue = 1\n", partial, "@@ -2,1 +2,1 @@\n-value = 1\n+value = 2\n"),
            ("a\nb\nc\nd\n", straddling, "@@ -2,2 +2,2 @@\n-b\n-c\n+c\n+d\n"),
        )
        for case_source, edit, hunk in cases:
            with self.subTest(edit=edit.before):
                result = case_source.replace(edit.before, edit.after, 1)
                patchset = self._landmark_patchset(
                    source=case_source, result=result, edit=edit, hunk=hunk
                )
                with self.assertRaisesRegex(
                    PatchRefusedError,
                    r"after block already appears 1 time\(s\) in module\.py "
                    r"outside its before landmark",
                ):
                    SourcePatchTransaction(self.source, self.artifact, patchset).apply()
                self.assertEqual((self.source / "module.py").read_text(), case_source)

    def _stated_patchset(
        self, *, source: str, result: str, edits: tuple[LandmarkEdit, ...], review: str
    ) -> PatchSet:
        """A stage over notes.txt whose review diff is exactly *review*."""
        (self.source / "notes.txt").write_text(source, encoding="utf-8")
        (self.artifact / "stated.patch").write_text(review, encoding="utf-8", newline="\n")
        stage = PatchStage(
            name="stated",
            rationale="Synthetic review used to prove what a review diff may state.",
            removal_condition="Remove when this framework test is removed.",
            review_patch="stated.patch",
            review_sha256=sha256_bytes(review.encode("utf-8")),
            files=(FileIdentity("notes.txt", sha256_text(source), sha256_text(result)),),
            edits=edits,
            validate_before=_noop,
            validate_after=_noop,
        )
        return _patchset((stage,), final_files={"notes.txt": sha256_text(result)})

    # Two changes ten lines apart: the first adds a line, so the second's old
    # block starts one line above its new block.
    _TWO_HUNK_SOURCE = "".join(f"line {n}\n" for n in range(1, 21))
    _TWO_HUNK_RESULT = _TWO_HUNK_SOURCE.replace(
        "line 3\n", "line 3\nadded\n"
    ).replace("line 15\n", "line fifteen\n")

    def _two_hunk_edits(self) -> tuple[LandmarkEdit, ...]:
        first_before = "line 2\nline 3\nline 4\n"
        second_before = "line 14\nline 15\nline 16\n"
        return (
            LandmarkEdit(
                name="notes.txt:landmark-1", path="notes.txt",
                before=first_before, after="line 2\nline 3\nadded\nline 4\n",
                review_before=first_before,
                review_after="line 2\nline 3\nadded\nline 4\n",
            ),
            LandmarkEdit(
                name="notes.txt:landmark-2", path="notes.txt",
                before=second_before, after="line 14\nline fifteen\nline 16\n",
                review_before=second_before,
                review_after="line 14\nline fifteen\nline 16\n",
            ),
        )

    def _two_hunk_review(
        self, *, second_old: int = 14, second_new: int = 15, first_counts: str = "3 +2,4",
        header: str = "index 0000000..0000000 100644\n",
    ) -> str:
        source, result = self._TWO_HUNK_SOURCE, self._TWO_HUNK_RESULT
        header = header.replace(
            "0000000..0000000",
            f"{git_blob_id(source)[:7]}..{git_blob_id(result)[:7]}",
        )
        return (
            "diff --git a/notes.txt b/notes.txt\n" + header
            + "--- a/notes.txt\n+++ b/notes.txt\n"
            + f"@@ -2,{first_counts} @@\n line 2\n line 3\n+added\n line 4\n"
            + f"@@ -{second_old},3 +{second_new},3 @@\n line 14\n-line 15\n"
            "+line fifteen\n line 16\n"
        )

    def _apply_stated(self, review: str) -> None:
        SourcePatchTransaction(
            self.source, self.artifact,
            self._stated_patchset(
                source=self._TWO_HUNK_SOURCE, result=self._TWO_HUNK_RESULT,
                edits=self._two_hunk_edits(), review=review,
            ),
        ).apply()

    def test_every_coordinate_a_hunk_header_states_is_proven(self) -> None:
        # The true review applies.
        self._apply_stated(self._two_hunk_review())
        self.assertEqual(
            (self.source / "notes.txt").read_text(), self._TWO_HUNK_RESULT
        )
        cases = (
            # The second hunk's old block starts where its new block does less
            # the line the first added: 14, not 15.
            (self._two_hunk_review(second_old=15),
             r"starts at line 14 of notes\.txt before the stage, but its review "
             r"hunk's header says line 15"),
            (self._two_hunk_review(second_new=14),
             r"lands at line 15 of notes\.txt, but its review hunk's header "
             r"places it at line 14"),
            # A count that claims a line the body does not hold, or misses one
            # it does.
            (self._two_hunk_review(first_counts="3 +2,5"),
             r"a hunk ends 0 old and 1 new line\(s\) short of its header's counts"),
            (self._two_hunk_review(first_counts="3 +2,3"),
             r"a hunk holds more new lines than its header's 3"),
        )
        for review, refusal in cases:
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(PatchRefusedError, refusal):
                    self._apply_stated(review)
                self.assertEqual(
                    (self.source / "notes.txt").read_text(), self._TWO_HUNK_SOURCE
                )

    def test_a_hunk_header_follows_gits_count_rules(self) -> None:
        # git omits a count of one.
        edit = LandmarkEdit(
            name="notes.txt:landmark-1", path="notes.txt", before="a\n",
            after="b\n", review_before="a\n", review_after="b\n",
        )
        review = (
            "diff --git a/notes.txt b/notes.txt\n--- a/notes.txt\n+++ b/notes.txt\n"
            "@@ -1 +1 @@\n-a\n+b\n"
        )
        transaction = SourcePatchTransaction(
            self.source, self.artifact,
            self._stated_patchset(source="a\n", result="b\n", edits=(edit,), review=review),
        )
        self.assertEqual(transaction.plan()[1].state, "planned")
        # An empty range is named by the line before it: a created file's old
        # range is 0,0, a deleted file's new range 0,0.
        after = "created\n"
        created = _stage(
            self.artifact, name="create-file",
            transformations=(("created.py", "", after),),
        )
        wrong = _review("created.py", "", after).replace("@@ -0,0 +1,1 @@", "@@ -1,0 +1,1 @@")
        (self.artifact / created.review_patch).write_text(wrong, encoding="utf-8")
        created = PatchStage(**{
            **created.__dict__, "review_sha256": sha256_bytes(wrong.encode("utf-8"))
        })
        with self.assertRaisesRegex(
            PatchRefusedError, r"starts at line 0 of created\.py before the stage, "
            r"but its review hunk's header says line 1"
        ):
            SourcePatchTransaction(
                self.source, self.artifact,
                _patchset((created,), final_files={"created.py": sha256_text(after)}),
            ).plan()

    def test_a_stated_mode_is_the_files_mode(self) -> None:
        for file_mode, stated, refused in (
            (0o644, "100644", False), (0o755, "100755", False),
            (0o755, "100644", True), (0o644, "100755", True),
        ):
            with self.subTest(file_mode=oct(file_mode), stated=stated):
                review = self._two_hunk_review(
                    header=f"index 0000000..0000000 {stated}\n"
                )
                patchset = self._stated_patchset(
                    source=self._TWO_HUNK_SOURCE, result=self._TWO_HUNK_RESULT,
                    edits=self._two_hunk_edits(), review=review,
                )
                (self.source / "notes.txt").chmod(file_mode)
                transaction = SourcePatchTransaction(self.source, self.artifact, patchset)
                if refused:
                    with self.assertRaisesRegex(
                        PatchRefusedError,
                        rf"states mode {stated} for notes\.txt, whose mode is "
                        rf"{'100755' if file_mode & 0o100 else '100644'}",
                    ):
                        transaction.plan()
                else:
                    self.assertEqual(transaction.plan()[1].state, "planned")
        # A created file is written 0644, so it is declared 100644.
        after = "created\n"
        created = _stage(
            self.artifact, name="create-file",
            transformations=(("created.py", "", after),),
        )
        executable = _review("created.py", "", after).replace(
            "new file mode 100644", "new file mode 100755"
        )
        (self.artifact / created.review_patch).write_text(executable, encoding="utf-8")
        created = PatchStage(**{
            **created.__dict__, "review_sha256": sha256_bytes(executable.encode("utf-8"))
        })
        with self.assertRaisesRegex(
            PatchRefusedError, r"states mode 100755 for created\.py, whose mode is 100644"
        ):
            SourcePatchTransaction(
                self.source, self.artifact,
                _patchset((created,), final_files={"created.py": sha256_text(after)}),
            ).plan()

    def test_a_review_diff_states_nothing_unread(self) -> None:
        true_review = self._two_hunk_review()
        cases = (
            # A mode change: no stage changes a mode.
            (true_review.replace(
                "index ", "old mode 100644\nnew mode 100755\nindex ", 1),
             r"unsupported review diff line 'old mode 100644\\n'"),
            # A path pair naming another file.
            (true_review.replace("+++ b/notes.txt", "+++ b/vllm/notes.txt"),
             r"notes\.txt's path pair names another file"),
            # A path pair that says the file is created.
            (true_review.replace("--- a/notes.txt", "--- /dev/null"),
             r"notes\.txt's path pair says created, but its header does not"),
            # A line outside every hunk.
            (true_review + "-stray\n", r"unsupported review diff line '-stray\\n'"),
            (true_review.replace("index ", "similarity index 90%\nindex ", 1),
             r"unsupported review diff line 'similarity index 90%\\n'"),
            ("preamble\n" + true_review, r"line before the first file section"),
        )
        for review, refusal in cases:
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(PatchRefusedError, refusal):
                    self._apply_stated(review)

    def test_a_files_hunks_are_stated_in_order(self) -> None:
        source, result = self._TWO_HUNK_SOURCE, self._TWO_HUNK_RESULT
        first, second = self._two_hunk_edits()
        swapped = (
            "diff --git a/notes.txt b/notes.txt\n--- a/notes.txt\n+++ b/notes.txt\n"
            "@@ -14,3 +14,3 @@\n line 14\n-line 15\n+line fifteen\n line 16\n"
            "@@ -2,3 +2,4 @@\n line 2\n line 3\n+added\n line 4\n"
        )
        with self.assertRaisesRegex(
            PatchRefusedError, r"starts inside or above the one before it in notes\.txt"
        ):
            SourcePatchTransaction(
                self.source, self.artifact,
                self._stated_patchset(
                    source=source, result=result,
                    edits=(
                        LandmarkEdit(**{**second.__dict__, "name": "notes.txt:landmark-1"}),
                        LandmarkEdit(**{**first.__dict__, "name": "notes.txt:landmark-2"}),
                    ),
                    review=swapped,
                ),
            ).apply()

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


class DeletionTransactionTests(unittest.TestCase):
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

    def test_nonempty_deletion_applies_and_is_idempotent(self) -> None:
        doomed = "def dead_code():\n    return None\n"
        (self.source / "doomed.py").write_text(doomed, encoding="utf-8")
        stage = _deletion_stage(
            self.artifact,
            name="delete-doomed",
            deletions=(("doomed.py", doomed),),
        )
        transaction = SourcePatchTransaction(
            self.source, self.artifact, _patchset((stage,), final_files={})
        )

        first = transaction.apply()
        second = transaction.apply()

        self.assertEqual(first.state, "applied")
        self.assertEqual(first.changed_files, ("doomed.py",))
        self.assertEqual(second.state, "already-applied")
        self.assertFalse((self.source / "doomed.py").exists())

    def test_empty_file_deletion_is_hunkless_and_applies(self) -> None:
        (self.source / "__init__.py").write_bytes(b"")
        stage = _deletion_stage(
            self.artifact,
            name="delete-empty-marker",
            deletions=(("__init__.py", ""),),
        )
        transaction = SourcePatchTransaction(
            self.source, self.artifact, _patchset((stage,), final_files={})
        )

        result = transaction.apply()

        self.assertEqual(result.state, "applied")
        self.assertFalse((self.source / "__init__.py").exists())
        self.assertEqual(transaction.apply().state, "already-applied")

    def test_deletion_without_review_marker_is_refused(self) -> None:
        doomed = "def dead_code():\n    return None\n"
        (self.source / "doomed.py").write_text(doomed, encoding="utf-8")
        # Review diff shows the emptying hunk but not the deleted-file
        # header: the Python data and the review evidence disagree about
        # whether the file ceases to exist. Its path pair either says the
        # file is deleted, which its header does not, or that it survives,
        # which the stage does not.
        for new_path, refusal in (
            ("/dev/null", "doomed.py's path pair says deleted, but its header does not"),
            ("b/doomed.py", "declares deletions"),
        ):
            with self.subTest(new_path=new_path):
                review = (
                    f"diff --git a/doomed.py b/doomed.py\n"
                    f"--- a/doomed.py\n"
                    f"+++ {new_path}\n"
                    f"@@ -1,2 +0,0 @@\n"
                    + "".join(f"-{line}" for line in doomed.splitlines(keepends=True))
                )
                stage = _deletion_stage(
                    self.artifact,
                    name="delete-doomed",
                    deletions=(("doomed.py", doomed),),
                    review_override=review,
                )
                transaction = SourcePatchTransaction(
                    self.source, self.artifact, _patchset((stage,), final_files={})
                )
                with self.assertRaisesRegex(PatchRefusedError, refusal):
                    transaction.apply()
                self.assertTrue((self.source / "doomed.py").exists())

    def test_deletion_landmark_mismatch_refuses_without_writes(self) -> None:
        expected = "def dead_code():\n    return None\n"
        drifted = "def dead_code():\n    return 1\n"
        (self.source / "doomed.py").write_text(drifted, encoding="utf-8")
        stage = _deletion_stage(
            self.artifact,
            name="delete-doomed",
            deletions=(("doomed.py", expected),),
        )
        transaction = SourcePatchTransaction(
            self.source, self.artifact, _patchset((stage,), final_files={})
        )

        with self.assertRaises(PatchRefusedError):
            transaction.apply()
        self.assertEqual(
            (self.source / "doomed.py").read_text(encoding="utf-8"), drifted
        )

    def test_resurrecting_a_deleted_file_is_refused(self) -> None:
        doomed = "def dead_code():\n    return None\n"
        reborn = "def dead_code():\n    return 2\n"
        (self.source / "doomed.py").write_text(doomed, encoding="utf-8")
        delete_stage = _deletion_stage(
            self.artifact,
            name="delete-doomed",
            deletions=(("doomed.py", doomed),),
        )
        recreate_stage = _stage(
            self.artifact,
            name="recreate-doomed",
            transformations=(("doomed.py", "", reborn),),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset(
                (delete_stage, recreate_stage),
                final_files={"doomed.py": sha256_text(reborn)},
            ),
        )

        with self.assertRaisesRegex(PatchRefusedError, "resurrected"):
            transaction.apply()
        self.assertEqual(
            (self.source / "doomed.py").read_text(encoding="utf-8"), doomed
        )

    def test_failed_commit_restores_deleted_files(self) -> None:
        doomed = "def dead_code():\n    return None\n"
        kept_before = "def value():\n    return 1\n"
        kept_after = "def value():\n    return 2\n"
        (self.source / "doomed.py").write_text(doomed, encoding="utf-8")
        (self.source / "kept.py").write_text(kept_before, encoding="utf-8")
        delete_stage = _deletion_stage(
            self.artifact,
            name="delete-doomed",
            deletions=(("doomed.py", doomed),),
        )
        edit_stage = _stage(
            self.artifact,
            name="edit-kept",
            transformations=(("kept.py", kept_before, kept_after),),
        )
        transaction = SourcePatchTransaction(
            self.source,
            self.artifact,
            _patchset(
                (delete_stage, edit_stage),
                final_files={"kept.py": sha256_text(kept_after)},
            ),
        )

        real_replace = os.replace

        def failing_replace(src, dst):
            # The unlink of doomed.py happens first (changed_files order);
            # failing the kept.py replacement must roll the deletion back.
            # Rollback restores use a .qwen-rollback. temp name and must be
            # allowed through, or the failure being tested would also break
            # the recovery being asserted.
            if str(dst).endswith("kept.py") and ".qwen-rollback." not in str(src):
                raise OSError("synthetic replace failure")
            return real_replace(src, dst)

        with patch.object(framework.os, "replace", failing_replace):
            with self.assertRaises(PatchWriteError):
                transaction.apply()

        self.assertEqual(
            (self.source / "doomed.py").read_text(encoding="utf-8"), doomed
        )
        self.assertEqual(
            (self.source / "kept.py").read_text(encoding="utf-8"), kept_before
        )


class WorktreeStatusTests(unittest.TestCase):
    """The footprint a build checks its reconstruction against is derived."""

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

    def _tree(self) -> dict[str, str]:
        return {
            path.relative_to(self.source).as_posix(): sha256_bytes(path.read_bytes())
            for path in self.source.rglob("*")
            if path.is_file()
        }

    def test_status_is_exactly_what_the_transaction_changes(self) -> None:
        # One path of each kind: changed, created, deleted, changed and then
        # restored by a later stage, and untouched. Only the first three differ
        # from the source revision afterwards.
        original = {
            "pkg/changed.py": "def value():\n    return 1\n",
            "pkg/doomed.py": "def dead_code():\n    return None\n",
            "pkg/restored.py": "def flag():\n    return True\n",
            "z_untouched.py": "def other():\n    return 0\n",
        }
        for path, text in original.items():
            target = self.source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        changed_after = "def value():\n    return 2\n"
        created = "def fresh():\n    return 3\n"
        restored_middle = "def flag():\n    return False\n"
        first = _stage(
            self.artifact,
            name="first",
            transformations=(
                ("pkg/changed.py", original["pkg/changed.py"], changed_after),
                ("pkg/created.py", "", created),
                ("pkg/restored.py", original["pkg/restored.py"], restored_middle),
            ),
        )
        second = _stage(
            self.artifact,
            name="second",
            transformations=(
                ("pkg/restored.py", restored_middle, original["pkg/restored.py"]),
            ),
        )
        third = _deletion_stage(
            self.artifact,
            name="third",
            deletions=(("pkg/doomed.py", original["pkg/doomed.py"]),),
        )
        patchset = _patchset(
            (first, second, third),
            final_files={
                "pkg/changed.py": sha256_text(changed_after),
                "pkg/created.py": sha256_text(created),
                "pkg/restored.py": sha256_text(original["pkg/restored.py"]),
            },
        )
        before = self._tree()

        SourcePatchTransaction(self.source, self.artifact, patchset).apply()

        after = self._tree()
        self.assertEqual(
            patchset.worktree_status(),
            (" M pkg/changed.py", " D pkg/doomed.py", "?? pkg/created.py"),
        )
        observed = {
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        }
        self.assertEqual(
            {line[3:] for line in patchset.worktree_status()}, observed
        )


# What a validator reads of the code is a text test: whether a string occurs
# in it (``in`` / ``not in`` beside a string constant), a string search or
# string test of it, or code text -- a source, a symbol's source, an unparsed
# node -- compared with a string. A validator's own constants compared with
# each other (a loop over literal names) read nothing of the code.
_TEXT_METHODS = frozenset({
    "find", "rfind", "index", "rindex", "count", "startswith", "endswith",
    "search", "match", "fullmatch", "findall", "finditer",
})
_CODE_TEXT = frozenset({"_source", "_symbol_source", "_branch_source", "unparse"})
_REFUSALS = frozenset({"PatchRefusedError", "Exception", "BaseException"})


def _is_code_text(node: ast.expr) -> bool:
    return any(
        isinstance(inner, ast.Call)
        and (
            (isinstance(inner.func, ast.Name) and inner.func.id in _CODE_TEXT)
            or (isinstance(inner.func, ast.Attribute) and inner.func.attr in _CODE_TEXT)
        )
        or isinstance(inner, ast.Subscript)
        and isinstance(inner.value, ast.Name)
        and inner.value.id == "state"
        for inner in ast.walk(node)
    )


def _reads_code_text(node: ast.AST, derived: frozenset[str] = frozenset()) -> bool:
    """Whether *node* holds a text test of the code, or a name a text test
    computed (*derived*)."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in derived:
            return True
        if isinstance(inner, ast.Compare):
            sides = [inner.left, *inner.comparators]
            strings = [
                side for side in sides
                if isinstance(side, ast.Constant) and isinstance(side.value, str)
            ]
            if strings and any(isinstance(op, (ast.In, ast.NotIn)) for op in inner.ops):
                return True
            if (
                strings
                and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in inner.ops)
                and any(_is_code_text(side) for side in sides)
            ):
                return True
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in _TEXT_METHODS
        ):
            return True
    return False


def _alternatives(node: ast.AST) -> bool:
    """Whether a comprehension iterates over alternatives the validator names
    -- a literal tuple, list or set of locations -- rather than over the parts
    of one location it reads (its lines, its nodes)."""
    return isinstance(node, (ast.Tuple, ast.List, ast.Set))


def _shape_choices(function: ast.FunctionDef) -> list[str]:
    """Each place where *function* chooses, by the code's text, what to check
    or accepts more than one shape of it. See ContractShapeTests."""
    choices: list[str] = []
    where = function.name

    # Names a text test computed, to a fixed point: a choice can be made one
    # assignment away from the test.
    derived: set[str] = set()
    while True:
        before = len(derived)
        for node in ast.walk(function):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                value = node.value
                if value is not None and _reads_code_text(value, frozenset(derived)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        derived.update(
                            name.id for name in ast.walk(target)
                            if isinstance(name, ast.Name)
                        )
        if len(derived) == before:
            break
    derived_names = frozenset(derived)

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def requirement(node: ast.AST) -> ast.expr | None:
        """The condition of the requirement *node* lies in, if any."""
        while node in parents:
            parent = parents[node]
            if (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id == "_require"
                and parent.args
                and node is parent.args[0]
            ):
                return node
            node = parent
        return None

    def search_loop(node: ast.AST) -> bool:
        """An ``if`` that filters a loop over the parts of one location: the
        imperative form of a search, which a requirement then holds to what
        it found."""
        parent = parents.get(node)
        return (
            isinstance(parent, ast.For)
            and node in parent.body
            and not _alternatives(parent.iter)
        )

    def judged_elsewhere(node: ast.AST) -> bool:
        """Whether a short-circuit outside a requirement is part of a test
        another rule judges: a branch's test, a comprehension's filter, or
        the value of an assignment, whose name then counts as the test."""
        while node in parents:
            parent = parents[node]
            if isinstance(parent, (ast.If, ast.IfExp, ast.While)) and node is parent.test:
                return True
            if isinstance(parent, ast.comprehension) and node in parent.ifs:
                return True
            if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                return node is parent.value
            if isinstance(parent, ast.stmt):
                return False
            node = parent
        return False

    def polarity(node: ast.AST, condition: ast.expr) -> bool:
        """Whether *node* is asserted (True) or denied (False) by *condition*."""
        positive = True
        while node is not condition:
            parent = parents[node]
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                positive = not positive
            node = parent
        return positive

    for node in ast.walk(function):
        line = getattr(node, "lineno", function.lineno)
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            if _reads_code_text(node.test, derived_names) and not (
                isinstance(node, ast.If) and search_loop(node)
            ):
                choices.append(f"{where}:{line} branches on the code")
        elif isinstance(node, ast.Match):
            if _reads_code_text(node.subject, derived_names):
                choices.append(f"{where}:{line} branches on the code")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for generator in node.generators:
                if _alternatives(generator.iter) and any(
                    _reads_code_text(test, derived_names) for test in generator.ifs
                ):
                    choices.append(
                        f"{where}:{line} chooses among locations by the code"
                    )
        elif isinstance(node, ast.BoolOp):
            if not any(_reads_code_text(v, derived_names) for v in node.values):
                continue
            condition = requirement(node)
            if condition is None:
                if not judged_elsewhere(node):
                    choices.append(f"{where}:{line} short-circuits on the code")
                continue
            positive = polarity(node, condition)
            if isinstance(node.op, ast.Or) == positive:
                choices.append(f"{where}:{line} accepts either of two shapes")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("any", "all")
            and node.args
            and _reads_code_text(node.args[0], derived_names)
        ):
            condition = requirement(node)
            if condition is None:
                continue
            if (node.func.id == "any") == polarity(node, condition):
                choices.append(f"{where}:{line} accepts any of several shapes")
        elif isinstance(node, ast.Try):
            for handler in node.handlers:
                caught = (
                    [] if handler.type is None
                    else handler.type.elts if isinstance(handler.type, ast.Tuple)
                    else [handler.type]
                )
                if handler.type is None or any(
                    isinstance(kind, ast.Name) and kind.id in _REFUSALS for kind in caught
                ):
                    choices.append(f"{where}:{line} chooses by whether a check fails")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_validate_")
        ):
            choices.append(f"{where}:{line} runs {node.func.id}")
    return choices


class ContractShapeTests(unittest.TestCase):
    """Every semantic contract validator pins one shape of the code it checks.

    validate_final runs every stage's validate_after again on the final tree,
    so a validator that decides by the code's text which shape to check -- or
    hands its check to another stage's validator -- accepts a shape the code
    has left beside the one it has, and a regression to the left shape passes
    it. A stage that rewrites a construct an earlier contract pins is written
    where the construct is first written instead, or the construct is pinned
    by the stage that moves it.

    A validator chooses by the code's text wherever a text test of the code
    decides control flow (an ``if``, a conditional expression, a ``while``, a
    ``match``, a short-circuit ``and`` / ``or`` outside a requirement, or a
    ``try`` that catches a check's refusal), filters a validator-named set of
    locations, or makes a requirement accept alternatives (``or`` or ``any``
    asserted, ``and`` or ``all`` denied); or wherever it runs another stage's
    validator. A name a text test computed counts as the test. Not choices:
    a search over the parts of one location (its lines, its nodes), by a
    comprehension or a loop's ``if``, which requirements then hold to what it
    found -- a search followed by an exact-count requirement; a conjunction a
    requirement asserts; and tests of the validator's own constants.
    """

    def test_no_validator_chooses_by_the_code(self) -> None:
        path = Path(__file__).with_name("contracts_vllm.py")
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        choices = [
            choice
            for function in module.body
            if isinstance(function, ast.FunctionDef)
            and function.name.startswith("_validate_")
            for choice in _shape_choices(function)
        ]
        self.assertEqual(choices, [])

    def test_the_detector_sees_every_form_of_choice(self) -> None:
        chooses = {
            "if": 'if "def f(" in _source(state, a, label=l):\n    check(a)',
            "elif-derived": 'moved = "def f(" in _source(state, a, label=l)\n'
                            'if moved:\n    check(a)',
            "conditional": 'check(a if "x" in s else b)',
            "while": 'while s.find("x") >= 0:\n    s = s[1:]',
            "method": 'if s.startswith("def"):\n    check(a)',
            "count": 'if s.count("x") == 1:\n    check(a)',
            "unparse": 'if ast.unparse(node) == "f()":\n    check(a)',
            "comprehension": 'found = [p for p in (a, b) if "def f(" in _source(state, p)]\n'
                             '_require(len(found) == 1, l)\ncheck(found[0])',
            "location loop": 'for p in (a, b):\n    if "def f(" in _source(state, p):\n'
                             '        check(p)',
            "or": '_require("x" in s or "y" in s, l)',
            "denied and": '_require(not ("x" in s and "y" in s), l)',
            "any": '_require(any("x" in _source(state, p) for p in (a, b)), l)',
            "denied all": '_require(not all("x" in t for t in texts), l)',
            "short-circuit": 'check("x" in s and a)',
            "try": 'try:\n    check(a)\nexcept PatchRefusedError:\n    check(b)',
            "bare try": 'try:\n    check(a)\nexcept:\n    check(b)',
            "delegation": '_validate_other_after(state)',
        }
        pins = {
            "requirement": '_require("x" not in s and "y" not in s, l)',
            "conjunction": '_require("x" in s and len(found) == 1, l)',
            "denied any": '_require(not any("x" in line for line in lines), l)',
            "search": 'found = [n for n in ast.walk(t) if ast.unparse(n) == "f()"]\n'
                      '_require(len(found) == 1, l)',
            "comment filter": 'code = [line for line in s.splitlines()\n'
                              '        if not line.strip().startswith("#")]\n'
                              '_require(not any("x" in line for line in code), l)',
            "search loop": 'for node in ast.walk(t):\n'
                           '    if ast.unparse(node) == "f()":\n'
                           '        found.append(node)\n'
                           '_require(len(found) == 1, l)',
            "own constants": 'for name in ("a", "b"):\n'
                             '    require_text(state, name, "x", '
                             'count=2 if name == "a" else 1, label=l)',
        }

        def function(body: str) -> ast.FunctionDef:
            source = "def _validate_case(state):\n" + "".join(
                f"    {line}\n" for line in body.splitlines()
            )
            return ast.parse(source).body[0]

        for name, body in chooses.items():
            with self.subTest(chooses=name):
                self.assertNotEqual(_shape_choices(function(body)), [], body)
        for name, body in pins.items():
            with self.subTest(pins=name):
                self.assertEqual(_shape_choices(function(body)), [], body)
