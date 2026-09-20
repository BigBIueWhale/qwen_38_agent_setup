#!/usr/bin/env python3
"""Failure-path tests for the live-tree materialising verb.

The verb's whole purpose is that it cannot destroy hand-authored work, so the
tests that matter most are the ones proving it refuses: an unexplained
difference must leave every path on disk exactly as it was, including the
paths the same run could have explained.
"""

import os
import tempfile
import unittest
from pathlib import Path

from .framework import PatchRefusedError, sha256_bytes
from .materialise_live_tree import (
    ABSENT,
    MaterialiseRefusedError,
    MaterialiseWriteError,
    apply_actions,
    deleted_paths,
    materialise,
    pristine_identities,
    read_shipped_identities,
)

ROOT = Path(__file__).resolve().parents[2]

OLD = "old reviewed body\n"
NEW = "new reviewed body\n"
UPSTREAM = "pristine upstream body\n"
MINE = "my unreviewed work in progress\n"

OLD_ID = sha256_bytes(OLD.encode())
NEW_ID = sha256_bytes(NEW.encode())
UPSTREAM_ID = sha256_bytes(UPSTREAM.encode())
MINE_ID = sha256_bytes(MINE.encode())

STALE = "vllm/stale.py"
OTHER = "vllm/other.py"
GONE = "vllm/gone.py"


def _put(root: Path, path: str, text: str) -> None:
    target = root.joinpath(*path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def _digest_of(root: Path, path: str) -> str:
    target = root.joinpath(*path.split("/"))
    if not target.exists():
        return ABSENT
    return sha256_bytes(target.read_bytes())


class MaterialiseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.live = base / "live"
        self.reconstruction = base / "reconstruction"
        self.live.mkdir()
        self.reconstruction.mkdir()

    def run_verb(self, *, final_files, deleted, pristine, shipped):
        return materialise(
            live_root=self.live,
            reconstruction_root=self.reconstruction,
            final_files=final_files,
            deleted=deleted,
            pristine=pristine,
            shipped=shipped,
        )

    def test_stale_file_is_written_and_cites_the_revision_that_shipped_it(self):
        _put(self.live, STALE, OLD)
        _put(self.reconstruction, STALE, NEW)
        actions, refusals, reviewed = self.run_verb(
            final_files={STALE: NEW_ID},
            deleted=set(),
            pristine={STALE: UPSTREAM_ID},
            shipped={STALE: {OLD_ID: "6af1414"}},
        )
        self.assertEqual(refusals, ())
        self.assertEqual(reviewed, 1)
        self.assertEqual([a.verb for a in actions], ["write"])
        self.assertEqual(actions[0].live, OLD_ID)
        self.assertEqual(actions[0].target, NEW_ID)
        self.assertIn("6af1414", actions[0].proof)
        self.assertEqual(_digest_of(self.live, STALE), NEW_ID)

    def test_unexplained_difference_refuses_and_writes_nothing_at_all(self):
        # The run holds one difference it can explain and one it cannot. The
        # explainable one must NOT be written: a partial materialisation would
        # leave a tree nobody can reason about afterwards.
        _put(self.live, STALE, OLD)
        _put(self.reconstruction, STALE, NEW)
        _put(self.live, OTHER, MINE)
        _put(self.reconstruction, OTHER, NEW)
        before = {
            STALE: _digest_of(self.live, STALE),
            OTHER: _digest_of(self.live, OTHER),
        }

        actions, refusals, reviewed = self.run_verb(
            final_files={STALE: NEW_ID, OTHER: NEW_ID},
            deleted=set(),
            pristine={STALE: UPSTREAM_ID, OTHER: UPSTREAM_ID},
            shipped={STALE: {OLD_ID: "6af1414"}},
        )

        self.assertEqual(actions, ())
        self.assertEqual(reviewed, 0)
        self.assertEqual([r.path for r in refusals], [OTHER])
        self.assertEqual(refusals[0].live, MINE_ID)
        self.assertEqual(refusals[0].target, NEW_ID)
        self.assertEqual(_digest_of(self.live, STALE), before[STALE])
        self.assertEqual(_digest_of(self.live, OTHER), before[OTHER])
        self.assertEqual(self.live.joinpath(*OTHER.split("/")).read_text(), MINE)

    def test_pristine_bytes_are_provably_stale(self):
        _put(self.live, STALE, UPSTREAM)
        _put(self.reconstruction, STALE, NEW)
        actions, refusals, _ = self.run_verb(
            final_files={STALE: NEW_ID},
            deleted=set(),
            pristine={STALE: UPSTREAM_ID},
            shipped={},
        )
        self.assertEqual(refusals, ())
        self.assertEqual(actions[0].proof, "pristine upstream identity")
        self.assertEqual(_digest_of(self.live, STALE), NEW_ID)

    def test_path_new_in_the_patch_set_is_created_from_absence(self):
        _put(self.reconstruction, STALE, NEW)
        actions, refusals, _ = self.run_verb(
            final_files={STALE: NEW_ID},
            deleted=set(),
            pristine={STALE: None},
            shipped={},
        )
        self.assertEqual(refusals, ())
        self.assertEqual([a.verb for a in actions], ["create"])
        self.assertEqual(actions[0].live, ABSENT)
        self.assertEqual(_digest_of(self.live, STALE), NEW_ID)

    def test_locally_deleted_upstream_path_is_refused(self):
        # The path exists upstream, so its absence is a local deletion no
        # committed revision accounts for.
        _put(self.reconstruction, STALE, NEW)
        actions, refusals, _ = self.run_verb(
            final_files={STALE: NEW_ID},
            deleted=set(),
            pristine={STALE: UPSTREAM_ID},
            shipped={},
        )
        self.assertEqual(actions, ())
        self.assertEqual([r.path for r in refusals], [STALE])
        self.assertEqual(refusals[0].live, ABSENT)
        self.assertFalse(self.live.joinpath(*STALE.split("/")).exists())

    def test_reviewed_deletion_removes_a_shipped_file(self):
        _put(self.live, GONE, OLD)
        actions, refusals, _ = self.run_verb(
            final_files={},
            deleted={GONE},
            pristine={GONE: UPSTREAM_ID},
            shipped={GONE: {OLD_ID: "50601f7"}},
        )
        self.assertEqual(refusals, ())
        self.assertEqual([a.verb for a in actions], ["delete"])
        self.assertEqual(actions[0].target, ABSENT)
        self.assertFalse(self.live.joinpath(*GONE.split("/")).exists())

    def test_reviewed_deletion_refuses_unexplained_bytes(self):
        _put(self.live, GONE, MINE)
        actions, refusals, _ = self.run_verb(
            final_files={},
            deleted={GONE},
            pristine={GONE: UPSTREAM_ID},
            shipped={GONE: {OLD_ID: "50601f7"}},
        )
        self.assertEqual(actions, ())
        self.assertEqual([r.path for r in refusals], [GONE])
        self.assertEqual(self.live.joinpath(*GONE.split("/")).read_text(), MINE)

    def test_already_materialised_tree_is_a_no_op(self):
        _put(self.live, STALE, NEW)
        _put(self.reconstruction, STALE, NEW)
        actions, refusals, reviewed = self.run_verb(
            final_files={STALE: NEW_ID},
            deleted=set(),
            pristine={STALE: UPSTREAM_ID},
            shipped={},
        )
        self.assertEqual((actions, refusals, reviewed), ((), (), 1))

    def test_symlink_where_a_source_belongs_is_refused(self):
        target = self.live.joinpath(*STALE.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(self.live / "elsewhere.py")
        _put(self.reconstruction, STALE, NEW)
        with self.assertRaises(MaterialiseRefusedError):
            self.run_verb(
                final_files={STALE: NEW_ID},
                deleted=set(),
                pristine={STALE: UPSTREAM_ID},
                shipped={STALE: {OLD_ID: "6af1414"}},
            )

    def test_reconstruction_drift_between_proof_and_copy_is_refused(self):
        from .materialise_live_tree import Action

        _put(self.reconstruction, STALE, MINE)
        drifted = Action(
            path=STALE, verb="write", live=OLD_ID, target=NEW_ID, proof="test"
        )
        with self.assertRaises(MaterialiseWriteError):
            apply_actions(
                (drifted,),
                live_root=self.live,
                reconstruction_root=self.reconstruction,
            )
        self.assertFalse(self.live.joinpath(*STALE.split("/")).exists())

    def test_written_file_keeps_the_reconstruction_mode(self):
        _put(self.live, STALE, OLD)
        _put(self.reconstruction, STALE, NEW)
        os.chmod(self.reconstruction.joinpath(*STALE.split("/")), 0o600)
        self.run_verb(
            final_files={STALE: NEW_ID},
            deleted=set(),
            pristine={STALE: UPSTREAM_ID},
            shipped={STALE: {OLD_ID: "6af1414"}},
        )
        mode = os.stat(self.live.joinpath(*STALE.split("/"))).st_mode & 0o777
        self.assertEqual(mode, 0o600)


class ShippedIdentityFileTests(unittest.TestCase):
    def _parse(self, text: str):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "shipped"
            source.write_text(text)
            return read_shipped_identities(source)

    def test_newest_revision_wins_for_a_repeated_identity(self):
        shipped = self._parse(
            f"{OLD_ID} {STALE} 5e94c31\n{OLD_ID} {STALE} 6af1414\n"
        )
        self.assertEqual(shipped[STALE][OLD_ID], "5e94c31")

    def test_malformed_line_is_refused(self):
        with self.assertRaises(MaterialiseRefusedError):
            self._parse(f"{OLD_ID} {STALE}\n")

    def test_non_digest_is_refused(self):
        with self.assertRaises(MaterialiseRefusedError):
            self._parse(f"{'z' * 64} {STALE} 5e94c31\n")

    def test_absolute_path_is_refused(self):
        # Path shape is the patcher's rule, reused rather than restated, so
        # its refusal is what surfaces here.
        with self.assertRaises(PatchRefusedError):
            self._parse(f"{OLD_ID} /etc/passwd 5e94c31\n")


class CommittedDataTests(unittest.TestCase):
    def test_deleted_and_final_paths_are_disjoint_and_complete(self):
        # The verb reasons over exactly final ∪ deleted; if a stage could both
        # delete a path and leave it in FINAL_FILES the classification would be
        # ambiguous. The patcher forbids resurrection; assert it here too.
        from .generated_vllm_stages import FINAL_FILES

        self.assertEqual(set(FINAL_FILES) & deleted_paths(), set())

    def test_every_reviewed_path_has_a_pristine_identity(self):
        from .generated_vllm_stages import FINAL_FILES

        pristine = pristine_identities()
        for path in set(FINAL_FILES) | deleted_paths():
            with self.subTest(path=path):
                self.assertIn(path, pristine)


class BuildScriptContractTests(unittest.TestCase):
    """The verb must stay a named mode that no other mode can reach."""

    def setUp(self) -> None:
        self.script = (ROOT / "scripts/build-vllm.sh").read_text()

    def test_the_mode_argument_is_required(self):
        self.assertNotIn('MODE="${1:-build}"', self.script)
        self.assertIn('MODE="$1"', self.script)

    def test_usage_names_every_mode(self):
        usage = self.script.split("<<'USAGE'", 1)[1].split("USAGE", 1)[0]
        for mode in ("build", "check", "materialise"):
            with self.subTest(mode=mode):
                self.assertIn(f"\n  {mode}", usage)

    def test_the_dispatch_accepts_exactly_the_documented_modes(self):
        self.assertIn("  build|check|materialise)", self.script)

    def test_materialise_is_reachable_only_by_name(self):
        # The invocation must sit under an explicit mode guard, so neither
        # check nor build can write the live tree as a side effect.
        guard = 'if [[ "${MODE}" == "materialise" ]]; then'
        self.assertIn(guard, self.script)
        after_guard = self.script.split(guard, 1)[1]
        self.assertIn(
            "patches.source_patch_v1.materialise_live_tree", after_guard
        )
        self.assertEqual(
            self.script.count("patches.source_patch_v1.materialise_live_tree"), 1
        )

    def test_the_live_tree_is_mounted_writable_only_for_the_verb(self):
        writable = '--volume "${VLLM_DIR}:/live:rw"'
        self.assertEqual(self.script.count(writable), 1)

    def test_an_unnamed_live_path_is_refused_in_every_mode(self):
        # The footprint question is asked once, unconditionally. A mode that
        # could skip it could write over hand-authored work.
        guard = 'if [[ -n "${unnamed_live_paths}" ]]; then'
        self.assertIn(guard, self.script)
        before = self.script.split(guard, 1)[0]
        self.assertNotIn("materialise", before.rsplit("actual_status=", 1)[1])

    def test_only_materialise_tolerates_a_tree_that_is_behind(self):
        # Exact equality stays the rule everywhere else; relaxing it is what
        # lets the verb repair a tree missing paths a pulled stage added.
        self.assertIn(
            'if [[ "${MODE}" != "materialise" && '
            '"${actual_status}" != "${EXPECTED_STATUS}" ]]; then',
            self.script,
        )

    def test_every_behind_the_tree_refusal_names_the_verb(self):
        # These messages named a file and read as a patch defect twice. Each
        # one now says what to do about it.
        remedy = 'echo "Next: ./scripts/build-vllm.sh materialise" >&2'
        self.assertEqual(self.script.count(remedy), 3)
        for symptom in (
            'echo "Reviewed patches do not reproduce ${relative_path}." >&2',
            'echo "Reviewed patches do not reproduce deletion of '
            '${relative_path}." >&2',
            'echo "The live tree is behind the reviewed patch set." >&2',
        ):
            with self.subTest(symptom=symptom):
                self.assertIn(symptom, self.script)
                self.assertIn(remedy, self.script.split(symptom, 1)[1][:400])


if __name__ == "__main__":
    unittest.main()
