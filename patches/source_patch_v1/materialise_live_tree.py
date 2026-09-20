#!/usr/bin/env python3
"""Write the live vLLM tree from an already verified reconstruction.

``containers/Dockerfile.runtime`` copies its runtime sources out of the live
``vllm/`` tree, so that tree is a build input -- but it is not a managed one.
Pulling a change to a patch stage updates the reviewed diffs on every machine
and leaves that tree behind on all of them, after which ``build-vllm.sh``
compares a correct reconstruction against stale bytes and names the file whose
new stage the tree is missing, which reads as a patch defect and is not one.
This is the explicit verb that closes that gap.  It runs only when it is asked
for by name: ``check`` and ``build`` read the live tree and never write it.

The live tree is also where a human edits to author a new stage, so a verb that
overwrote it blindly could destroy exactly the work this repository exists to
capture.  That is prevented by construction rather than by care.  A difference
is written only when the live bytes are *provably stale*, meaning one of:

  * they are an identity this repository itself shipped for that path at some
    committed revision -- the complete patched result recorded in a committed
    ``FINAL_FILES`` -- or
  * they are the pristine upstream identity the patch set starts that path
    from, recorded in the committed stage data.

Both proofs are read from committed data alone; neither is inferred from the
tree being examined.  A single difference that cannot be proved stale refuses
the whole run and writes nothing at all.  Materialising only the explainable
subset would leave a tree that neither this program nor an operator could
reason about afterwards, so there is no partial mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .framework import PatchRefusedError, _safe_relative_path, sha256_bytes
from .generated_vllm_stages import FINAL_FILES, GENERATED_STAGES

ABSENT = "absent"


class MaterialiseRefusedError(RuntimeError):
    """The live tree holds bytes no committed revision explains."""


# Committed identities are validated by the patcher's own path rules, so a
# malformed one raises the patcher's refusal. It is caught alongside this
# module's, rather than re-validated here, so there is one rule and one
# implementation of it.


class MaterialiseWriteError(RuntimeError):
    """A write was attempted and did not land as planned."""


@dataclass(frozen=True)
class Action:
    """One proved-stale difference between the live tree and the reconstruction."""

    path: str
    verb: str
    live: str
    target: str
    proof: str


@dataclass(frozen=True)
class Refusal:
    """One difference committed data does not explain."""

    path: str
    live: str
    target: str
    reason: str


def pristine_identities() -> dict[str, str | None]:
    """The upstream identity of every path the patch set touches.

    The first stage naming a path records what that path held before any stage
    ran.  ``None`` means the path does not exist upstream at all, so upstream
    absence -- not any digest -- is its pristine state.
    """
    first: dict[str, str | None] = {}
    for stage in GENERATED_STAGES:
        for entry in stage["files"]:
            first.setdefault(entry["path"], entry["before_sha256"])
    return first


def deleted_paths() -> set[str]:
    """Paths the patch set removes; the reconstruction has no such file."""
    return {
        entry["path"]
        for stage in GENERATED_STAGES
        for entry in stage["files"]
        if entry["after_sha256"] is None
    }


def read_shipped_identities(source: Path) -> dict[str, dict[str, str]]:
    """Parse ``digest path revision`` lines of committed final identities.

    The caller produces these from this repository's own history: every
    ``FINAL_FILES`` entry of every committed revision of the generated stage
    data.  Paths in that data never contain whitespace, which the generator
    and this parser both enforce, so the three fields are unambiguous.
    """
    shipped: dict[str, dict[str, str]] = {}
    for number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise MaterialiseRefusedError(
                f"{source}:{number}: expected 'digest path revision', got {line!r}"
            )
        digest, path, revision = fields
        if len(digest) != 64 or digest.strip("0123456789abcdef"):
            raise MaterialiseRefusedError(
                f"{source}:{number}: not a lowercase sha256 digest: {digest!r}"
            )
        _safe_relative_path(path)
        # git log is newest first, so the first revision seen for an identity
        # is the most recent one that shipped it -- the useful one to cite.
        shipped.setdefault(path, {}).setdefault(digest, revision)
    return shipped


def _live_digest(live_root: Path, path: str) -> str:
    """Digest of the live file, ``ABSENT``, or a refusal for anything else.

    Only regular non-symlink files can be reasoned about: a symlink or a
    directory standing where a source file belongs is local state this program
    has no committed identity for.
    """
    candidate = live_root.joinpath(*PurePosixPath(path).parts)
    if candidate.is_symlink():
        raise MaterialiseRefusedError(
            f"{path}: the live tree holds a symlink where a source file belongs; "
            "no writes performed"
        )
    if not candidate.exists():
        return ABSENT
    if not candidate.is_file():
        raise MaterialiseRefusedError(
            f"{path}: the live tree holds a non-regular file; no writes performed"
        )
    return sha256_bytes(candidate.read_bytes())


def plan_materialisation(
    *,
    live_root: Path,
    final_files: Mapping[str, str],
    deleted: frozenset[str] | set[str],
    pristine: Mapping[str, str | None],
    shipped: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[Action, ...], tuple[Refusal, ...]]:
    """Classify every difference between the live tree and the reconstruction.

    The reconstruction is described by ``final_files`` and ``deleted``, which
    the caller has already proved the reconstruction satisfies.  Nothing here
    writes; the caller decides, and only a wholly explained plan may proceed.
    """
    actions: list[Action] = []
    refusals: list[Refusal] = []
    for path in sorted(set(final_files) | set(deleted)):
        _safe_relative_path(path)
        target = ABSENT if path in deleted else final_files[path]
        live = _live_digest(live_root, path)
        if live == target:
            continue

        known = shipped.get(path, {})
        upstream = pristine.get(path)
        if live in known:
            proof = f"final identity shipped at {known[live]}"
        elif live != ABSENT and live == upstream:
            proof = "pristine upstream identity"
        elif live == ABSENT and upstream is None:
            proof = "absent upstream; this path is new in the patch set"
        else:
            refusals.append(
                Refusal(
                    path=path,
                    live=live,
                    target=target,
                    reason=(
                        "no committed revision has shipped these bytes for this "
                        "path, and they are not its pristine upstream identity"
                        if live != ABSENT
                        else "this path exists upstream, so its absence here is "
                        "a local deletion no committed revision explains"
                    ),
                )
            )
            continue

        if target == ABSENT:
            verb = "delete"
        elif live == ABSENT:
            verb = "create"
        else:
            verb = "write"
        actions.append(
            Action(path=path, verb=verb, live=live, target=target, proof=proof)
        )
    return tuple(actions), tuple(refusals)


def _write_file(destination: Path, data: bytes, mode: int) -> None:
    """Replace one path atomically, so no reader ever sees a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{destination.name}.qwen-materialise.", dir=destination.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, destination)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def apply_actions(
    actions: tuple[Action, ...], *, live_root: Path, reconstruction_root: Path
) -> None:
    """Copy each proved-stale path across from the verified reconstruction.

    The bytes come from the reconstruction and are re-checked against the
    identity the caller proved it satisfies, so a reconstruction that drifted
    between verification and copy cannot reach the live tree.  Emptied
    directories are left in place: git does not track directories either, and
    removing one is a deletion this verb has no committed identity for.
    """
    for action in actions:
        parts = PurePosixPath(action.path).parts
        destination = live_root.joinpath(*parts)
        if action.verb == "delete":
            destination.unlink()
            continue
        source = reconstruction_root.joinpath(*parts)
        data = source.read_bytes()
        found = sha256_bytes(data)
        if found != action.target:
            raise MaterialiseWriteError(
                f"{action.path}: the reconstruction no longer holds the verified "
                f"identity; expected {action.target}, found {found}"
            )
        _write_file(destination, data, os.stat(source).st_mode & 0o777)


def verify_live_tree(
    *,
    live_root: Path,
    final_files: Mapping[str, str],
    deleted: frozenset[str] | set[str],
) -> int:
    """Re-read every reviewed path and prove the live tree is now the result."""
    for path in sorted(set(final_files) | set(deleted)):
        expected = ABSENT if path in deleted else final_files[path]
        found = _live_digest(live_root, path)
        if found != expected:
            raise MaterialiseWriteError(
                f"{path}: the live tree does not hold the reconstruction's "
                f"identity after materialising; expected {expected}, found {found}"
            )
    return len(set(final_files) | set(deleted))


def materialise(
    *,
    live_root: Path,
    reconstruction_root: Path,
    final_files: Mapping[str, str],
    deleted: frozenset[str] | set[str],
    pristine: Mapping[str, str | None],
    shipped: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[Action, ...], tuple[Refusal, ...], int]:
    """Plan, and write only a plan in which every difference is proved stale.

    This is the whole decision: a refusal returns before ``apply_actions`` is
    reachable, so "one unexplained path writes nothing at all" is a property of
    the control flow rather than of the caller remembering to check.
    """
    actions, refusals = plan_materialisation(
        live_root=live_root,
        final_files=final_files,
        deleted=deleted,
        pristine=pristine,
        shipped=shipped,
    )
    if refusals:
        return (), refusals, 0
    if actions:
        apply_actions(
            actions, live_root=live_root, reconstruction_root=reconstruction_root
        )
    reviewed = verify_live_tree(
        live_root=live_root, final_files=final_files, deleted=deleted
    )
    return actions, (), reviewed


def _report_refusals(refusals: tuple[Refusal, ...]) -> None:
    print(
        "MATERIALISE REFUSED: the live vLLM tree holds bytes no committed "
        "revision of this repository explains.",
        file=sys.stderr,
    )
    for refusal in refusals:
        print(f"  {refusal.path}", file=sys.stderr)
        print(f"    live           {refusal.live}", file=sys.stderr)
        print(f"    reconstruction {refusal.target}", file=sys.stderr)
        print(f"    {refusal.reason}", file=sys.stderr)
    print(
        "\nNothing was written. These are either hand-authored edits that belong "
        "in a patch stage, or a damaged tree.\n"
        "Next: if the bytes are your work, compile them into a reviewed stage "
        "with patches/source_patch_v1/compile_review_diff.py and commit the "
        "result, then run this verb again. If they are not yours and nothing is "
        "lost by discarding them, restore the named paths deliberately -- this "
        "verb will not decide that for you.",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the live vLLM tree from a verified reconstruction, and only "
            "where the live bytes are provably stale."
        )
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        required=True,
        help="verified disposable worktree the reviewed patch set just produced",
    )
    parser.add_argument(
        "--live",
        type=Path,
        required=True,
        help="unmanaged live vLLM tree the runtime image is built from",
    )
    parser.add_argument(
        "--shipped",
        type=Path,
        required=True,
        help="committed final identities, as 'digest path revision' lines",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        final_files = dict(FINAL_FILES)
        deleted = deleted_paths()
        actions, refusals, reviewed = materialise(
            live_root=args.live,
            reconstruction_root=args.reconstruction,
            final_files=final_files,
            deleted=deleted,
            pristine=pristine_identities(),
            shipped=read_shipped_identities(args.shipped),
        )
        if refusals:
            _report_refusals(refusals)
            return 1
        if not actions:
            print(
                f"ALREADY MATERIALISED: the live tree already equals the "
                f"reconstruction across all {reviewed} reviewed paths; "
                f"nothing written."
            )
            return 0
    except (MaterialiseRefusedError, PatchRefusedError) as exc:
        print(f"MATERIALISE REFUSED: {exc}", file=sys.stderr)
        return 1
    except MaterialiseWriteError as exc:
        print(f"MATERIALISE WRITE FAILURE: {exc}", file=sys.stderr)
        return 1

    print(f"MATERIALISED {len(actions)} path(s) from the verified reconstruction:")
    for action in actions:
        print(f"  {action.verb:<6} {action.path}")
        print(f"    live           {action.live}  ({action.proof})")
        print(f"    reconstruction {action.target}")
    print(
        f"The live tree now holds the reconstruction's identity for all "
        f"{reviewed} reviewed paths.\n"
        f"Next: run ./scripts/build-vllm.sh check to verify the whole tree."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
