"""Transactional, landmark-aware source transformations.

The source patchers in this project are deliberately stricter than a unified
diff application.  Every patch set is tied to exact source identities, every
edit names and validates the source block it understands, and the complete
result is planned and checked before a disposable source tree is changed.

Unified diffs remain review evidence.  They are parsed independently and must
describe the exact same old/new blocks as the Python transformation data.
They are never used to decide where or how to edit a file.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class PatchRefusedError(RuntimeError):
    """The source does not satisfy a patcher's complete contract."""


class PatchWriteError(RuntimeError):
    """A fully validated plan could not be committed to the disposable tree."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PatchRefusedError(message)


def _safe_relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    _require(raw != "", "empty source path")
    _require(not path.is_absolute(), f"absolute source path is forbidden: {raw!r}")
    _require(".." not in path.parts, f"parent traversal is forbidden: {raw!r}")
    _require("." not in path.parts, f"non-canonical source path: {raw!r}")
    _require(str(path) == raw, f"non-canonical source path: {raw!r}")
    return path


@dataclass(frozen=True)
class LandmarkEdit:
    """One exact, named source transformation.

    ``before`` and ``after`` include the surrounding source landmarks needed
    to identify the intended construct.  An edit applies only when ``before``
    occurs exactly once and ``after`` does not already occur outside that
    block.  Whole-file hashes provide the stronger outer identity boundary.
    """

    name: str
    path: str
    before: str
    after: str
    review_before: str
    review_after: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path)
        _require(bool(self.name.strip()), "landmark edit has an empty name")
        _require(self.before != self.after, f"{self.name}: before equals after")
        _require(
            self.review_before != self.review_after,
            f"{self.name}: review before equals review after",
        )
        _require("\r" not in self.before, f"{self.name}: CR in before landmark")
        _require("\r" not in self.after, f"{self.name}: CR in after landmark")
        _require(
            self.review_before in self.before,
            f"{self.name}: review-before block is not inside the source landmark",
        )
        _require(
            self.review_after in self.after,
            f"{self.name}: review-after block is not inside the result landmark",
        )
        if self.review_before == "":
            _require(
                self.before == "" and self.review_after == self.after,
                f"{self.name}: a new-file hunk must describe the complete file",
            )
            before_prefix, before_suffix = "", ""
            after_prefix, after_suffix = "", ""
        elif self.review_after == "":
            _require(
                self.after == "" and self.review_before == self.before,
                f"{self.name}: a deleted-file hunk must describe the complete file",
            )
            before_prefix, before_suffix = "", ""
            after_prefix, after_suffix = "", ""
        else:
            before_prefix, before_suffix = self.before.split(self.review_before, 1)
            after_prefix, after_suffix = self.after.split(self.review_after, 1)
        _require(
            (before_prefix, before_suffix) == (after_prefix, after_suffix),
            f"{self.name}: expanded review context differs across old/new blocks",
        )


@dataclass(frozen=True)
class FileIdentity:
    path: str
    before_sha256: str | None
    after_sha256: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path)
        if self.before_sha256 is not None:
            _require(
                bool(re.fullmatch(r"[0-9a-f]{64}", self.before_sha256)),
                f"{self.path}: invalid before SHA-256",
            )
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", self.after_sha256)),
            f"{self.path}: invalid after SHA-256",
        )


StateValidator = Callable[[Mapping[str, str]], None]


@dataclass(frozen=True)
class PatchStage:
    name: str
    rationale: str
    removal_condition: str
    review_patch: str
    review_sha256: str
    files: tuple[FileIdentity, ...]
    edits: tuple[LandmarkEdit, ...]
    validate_before: StateValidator
    validate_after: StateValidator

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "patch stage has an empty name")
        _require(bool(self.rationale.strip()), f"{self.name}: empty rationale")
        _require(
            bool(self.removal_condition.strip()),
            f"{self.name}: empty removal condition",
        )
        _safe_relative_path(self.review_patch)
        _require(
            bool(re.fullmatch(r"[0-9a-f]{64}", self.review_sha256)),
            f"{self.name}: invalid review SHA-256",
        )
        file_paths = [contract.path for contract in self.files]
        _require(
            len(file_paths) == len(set(file_paths)),
            f"{self.name}: duplicate file identity",
        )
        edit_paths = {edit.path for edit in self.edits}
        _require(
            edit_paths == set(file_paths),
            f"{self.name}: edit/file sets disagree: "
            f"edits={sorted(edit_paths)!r} files={sorted(file_paths)!r}",
        )


@dataclass(frozen=True)
class PatchSet:
    name: str
    source_revision: str
    identity_files: Mapping[str, str]
    stages: tuple[PatchStage, ...]
    final_files: Mapping[str, str]
    validate_final: StateValidator

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "patch set has an empty name")
        _require(bool(self.source_revision.strip()), f"{self.name}: empty revision")
        _require(bool(self.stages), f"{self.name}: no stages")
        for path, digest in self.identity_files.items():
            _safe_relative_path(path)
            _require(
                bool(re.fullmatch(r"[0-9a-f]{64}", digest)),
                f"{self.name}: invalid identity digest for {path}",
            )
        for path, digest in self.final_files.items():
            _safe_relative_path(path)
            _require(
                bool(re.fullmatch(r"[0-9a-f]{64}", digest)),
                f"{self.name}: invalid final digest for {path}",
            )


@dataclass(frozen=True)
class PatchResult:
    state: str
    changed_files: tuple[str, ...]
    stages: tuple[str, ...]


@dataclass(frozen=True)
class _Backup:
    data: bytes | None
    mode: int | None


def require_text(
    state: Mapping[str, str],
    path: str,
    needle: str,
    *,
    count: int = 1,
    label: str,
) -> None:
    _require(path in state, f"{label}: missing {path}")
    actual = state[path].count(needle)
    _require(
        actual == count,
        f"{label}: expected {count} occurrence(s) of {needle!r} in {path}; "
        f"found {actual}",
    )


def forbid_text(
    state: Mapping[str, str], path: str, needle: str, *, label: str
) -> None:
    require_text(state, path, needle, count=0, label=label)


def require_python_symbols(
    state: Mapping[str, str],
    path: str,
    symbols: Mapping[str, Sequence[str] | None],
    *,
    label: str,
) -> None:
    """Require Python class/function qualnames and optional parameter names."""

    _require(path in state, f"{label}: missing {path}")
    try:
        tree = ast.parse(state[path], filename=path)
    except SyntaxError as exc:
        raise PatchRefusedError(f"{label}: {path} is not valid Python: {exc}") from exc

    found: dict[str, list[str]] = {}

    def visit(body: Sequence[ast.stmt], parents: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join((*parents, node.name))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [arg.arg for arg in node.args.posonlyargs]
                    args += [arg.arg for arg in node.args.args]
                    if node.args.vararg is not None:
                        args.append(f"*{node.args.vararg.arg}")
                    args += [arg.arg for arg in node.args.kwonlyargs]
                    if node.args.kwarg is not None:
                        args.append(f"**{node.args.kwarg.arg}")
                    found[qualname] = args
                else:
                    found[qualname] = []
                visit(node.body, (*parents, node.name))

    visit(tree.body, ())
    for qualname, expected_params in symbols.items():
        _require(
            qualname in found,
            f"{label}: required Python symbol {qualname!r} missing from {path}",
        )
        if expected_params is not None:
            _require(
                found[qualname] == list(expected_params),
                f"{label}: {path}:{qualname} parameters changed; "
                f"expected {list(expected_params)!r}, got {found[qualname]!r}",
            )


@dataclass(frozen=True)
class _ParsedReviewEdit:
    path: str
    before: str
    after: str
    old_start: int
    new_start: int


_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@"
)


def _parse_review_diff(data: bytes, *, label: str) -> tuple[_ParsedReviewEdit, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchRefusedError(f"{label}: review diff is not UTF-8: {exc}") from exc
    _require("\r" not in text, f"{label}: review diff contains CR bytes")
    lines = text.splitlines(keepends=True)
    edits: list[_ParsedReviewEdit] = []
    current_path: str | None = None
    index = 0
    while index < len(lines):
        raw = lines[index]
        match = _DIFF_HEADER.match(raw.rstrip("\n"))
        if match:
            _require(
                match.group(1) == match.group(2),
                f"{label}: rename/copy diffs are not supported: {raw!r}",
            )
            current_path = match.group(1)
            _safe_relative_path(current_path)
            index += 1
            continue
        hunk_match = _HUNK_HEADER.match(raw.rstrip("\n"))
        if hunk_match:
            _require(current_path is not None, f"{label}: hunk without file header")
            before: list[str] = []
            after: list[str] = []
            index += 1
            while index < len(lines):
                hunk_line = lines[index]
                if _DIFF_HEADER.match(hunk_line.rstrip("\n")) or _HUNK_HEADER.match(
                    hunk_line.rstrip("\n")
                ):
                    break
                if hunk_line.startswith("\\ No newline at end of file"):
                    raise PatchRefusedError(
                        f"{label}: files without terminal newline are unsupported"
                    )
                prefix = hunk_line[:1]
                payload = hunk_line[1:]
                _require(
                    prefix in {" ", "+", "-"},
                    f"{label}: unsupported hunk line {hunk_line!r}",
                )
                if prefix in {" ", "-"}:
                    before.append(payload)
                if prefix in {" ", "+"}:
                    after.append(payload)
                index += 1
            edits.append(
                _ParsedReviewEdit(
                    current_path,
                    "".join(before),
                    "".join(after),
                    int(hunk_match.group(1)),
                    int(hunk_match.group(2)),
                )
            )
            continue
        index += 1
    _require(edits, f"{label}: review diff contains no hunks")
    return tuple(edits)


class SourcePatchTransaction:
    """Plan, validate, and commit one exact patch set."""

    def __init__(self, source_root: Path, artifact_root: Path, patchset: PatchSet):
        self.source_root = source_root.resolve(strict=True)
        self.artifact_root = artifact_root.resolve(strict=True)
        self.patchset = patchset
        _require(self.source_root.is_dir(), f"source root is not a directory")
        _require(self.artifact_root.is_dir(), f"artifact root is not a directory")

    def _path(self, relative: str, *, existing: bool) -> Path:
        safe = _safe_relative_path(relative)
        candidate = self.source_root.joinpath(*safe.parts)
        parent = candidate.parent.resolve(strict=True)
        _require(
            parent == self.source_root or self.source_root in parent.parents,
            f"{self.patchset.name}: path escapes source root: {relative}",
        )
        if existing:
            _require(candidate.exists(), f"{self.patchset.name}: missing {relative}")
            info = candidate.lstat()
            _require(
                stat.S_ISREG(info.st_mode),
                f"{self.patchset.name}: {relative} is not a regular file",
            )
            _require(not candidate.is_symlink(), f"{relative} is a symlink")
        else:
            _require(
                not candidate.exists() and not candidate.is_symlink(),
                f"{relative} unexpectedly exists",
            )
        return candidate

    def _artifact_path(self, relative: str) -> Path:
        safe = _safe_relative_path(relative)
        candidate = self.artifact_root.joinpath(*safe.parts)
        parent = candidate.parent.resolve(strict=True)
        _require(
            parent == self.artifact_root or self.artifact_root in parent.parents,
            f"{self.patchset.name}: artifact path escapes root: {relative}",
        )
        _require(
            candidate.is_file() and not candidate.is_symlink(),
            f"{self.patchset.name}: missing regular review artifact {relative}",
        )
        return candidate

    def _read_text(self, relative: str) -> str:
        path = self._path(relative, existing=True)
        data = path.read_bytes()
        _require(b"\x00" not in data, f"{relative}: NUL byte in text source")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PatchRefusedError(f"{relative}: source is not UTF-8: {exc}") from exc
        _require("\r" not in text, f"{relative}: CR bytes are forbidden")
        _require(text.endswith("\n"), f"{relative}: no terminal newline")
        return text

    def _all_paths(self) -> tuple[str, ...]:
        paths: set[str] = set(self.patchset.identity_files)
        paths.update(self.patchset.final_files)
        for stage in self.patchset.stages:
            paths.update(contract.path for contract in stage.files)
        return tuple(sorted(paths))

    def _read_state(self) -> dict[str, str]:
        state: dict[str, str] = {}
        new_paths = {
            contract.path
            for stage in self.patchset.stages
            for contract in stage.files
            if contract.before_sha256 is None
        }
        for path in self._all_paths():
            candidate = self.source_root.joinpath(*PurePosixPath(path).parts)
            if not candidate.exists() and path in new_paths:
                continue
            state[path] = self._read_text(path)
        return state

    def _verify_identity(self, state: Mapping[str, str]) -> None:
        for path, digest in self.patchset.identity_files.items():
            _require(path in state, f"identity file missing: {path}")
            actual = sha256_text(state[path])
            _require(
                actual == digest,
                f"{self.patchset.name}: identity drift in {path}; "
                f"expected {digest}, got {actual}",
            )

    def _verify_review_artifact(self, stage: PatchStage) -> None:
        path = self._artifact_path(stage.review_patch)
        data = path.read_bytes()
        actual_digest = sha256_bytes(data)
        _require(
            actual_digest == stage.review_sha256,
            f"{stage.name}: review diff SHA-256 drift; expected "
            f"{stage.review_sha256}, got {actual_digest}",
        )
        parsed = _parse_review_diff(data, label=stage.name)
        _require(
            len(parsed) == len(stage.edits),
            f"{stage.name}: review diff has {len(parsed)} hunks but the Python "
            f"patcher has {len(stage.edits)} landmark transformations",
        )
        expected = tuple(
            _ParsedReviewEdit(
                edit.path,
                edit.review_before,
                edit.review_after,
                parsed_edit.old_start,
                parsed_edit.new_start,
            )
            for edit, parsed_edit in zip(stage.edits, parsed)
        )
        _require(
            parsed == expected,
            f"{stage.name}: Python landmarks and review diff describe "
            "different transformations",
        )

    @staticmethod
    def _matches(state: Mapping[str, str], expected: Mapping[str, str]) -> bool:
        return all(
            path in state and sha256_text(state[path]) == digest
            for path, digest in expected.items()
        )

    def _overall_pristine(self) -> dict[str, str | None]:
        first: dict[str, str | None] = {}
        for stage in self.patchset.stages:
            for contract in stage.files:
                first.setdefault(contract.path, contract.before_sha256)
        return first

    def _classify_mixed_state(self, state: Mapping[str, str]) -> str:
        pristine = self._overall_pristine()
        rows: list[str] = []
        for path in sorted(self.patchset.final_files):
            if path not in state:
                rows.append(f"{path}=absent")
                continue
            digest = sha256_text(state[path])
            if digest == self.patchset.final_files[path]:
                kind = "final"
            elif pristine.get(path) == digest:
                kind = "pristine"
            else:
                intermediates = {
                    contract.after_sha256
                    for stage in self.patchset.stages
                    for contract in stage.files
                    if contract.path == path
                }
                kind = "intermediate" if digest in intermediates else "unknown"
            rows.append(f"{path}={kind}:{digest}")
        return "; ".join(rows)

    def plan(self) -> tuple[dict[str, str], PatchResult]:
        state = self._read_state()
        self._verify_identity(state)
        for stage in self.patchset.stages:
            self._verify_review_artifact(stage)

        if self._matches(state, self.patchset.final_files):
            self.patchset.validate_final(state)
            return dict(state), PatchResult(
                state="already-applied",
                changed_files=(),
                stages=tuple(stage.name for stage in self.patchset.stages),
            )

        pristine = self._overall_pristine()
        is_pristine = True
        for path, digest in pristine.items():
            if digest is None:
                if path in state:
                    is_pristine = False
            elif path not in state or sha256_text(state[path]) != digest:
                is_pristine = False
        _require(
            is_pristine,
            f"{self.patchset.name}: source is neither wholly pristine nor "
            f"wholly final; no writes performed. "
            f"classification: {self._classify_mixed_state(state)}",
        )

        planned = dict(state)
        changed: set[str] = set()
        for stage in self.patchset.stages:
            stage.validate_before(planned)
            contracts = {contract.path: contract for contract in stage.files}
            for path, contract in contracts.items():
                if contract.before_sha256 is None:
                    _require(path not in planned, f"{stage.name}: {path} already exists")
                else:
                    _require(path in planned, f"{stage.name}: missing {path}")
                    actual = sha256_text(planned[path])
                    _require(
                        actual == contract.before_sha256,
                        f"{stage.name}: before hash drift in {path}; expected "
                        f"{contract.before_sha256}, got {actual}",
                    )

            for edit in stage.edits:
                current = planned.get(edit.path, "")
                before_count = current.count(edit.before)
                after_count = current.count(edit.after)
                _require(
                    before_count == 1,
                    f"{stage.name}:{edit.name}: expected one before landmark "
                    f"in {edit.path}, found {before_count}; no writes performed",
                )
                _require(
                    after_count == 0,
                    f"{stage.name}:{edit.name}: after block already appears "
                    f"{after_count} time(s) in {edit.path}; source is partial or "
                    "the landmarks overlap; no writes performed",
                )
                planned[edit.path] = current.replace(edit.before, edit.after, 1)
                changed.add(edit.path)

            for path, contract in contracts.items():
                _require(path in planned, f"{stage.name}: failed to produce {path}")
                actual = sha256_text(planned[path])
                _require(
                    actual == contract.after_sha256,
                    f"{stage.name}: final hash mismatch in {path}; expected "
                    f"{contract.after_sha256}, got {actual}; no writes performed",
                )
                if path.endswith(".py"):
                    try:
                        ast.parse(planned[path], filename=path)
                    except SyntaxError as exc:
                        raise PatchRefusedError(
                            f"{stage.name}: transformed {path} is invalid Python: {exc}"
                        ) from exc
            stage.validate_after(planned)

        _require(
            self._matches(planned, self.patchset.final_files),
            f"{self.patchset.name}: complete result does not match final manifest; "
            "no writes performed",
        )
        self.patchset.validate_final(planned)
        return planned, PatchResult(
            state="planned",
            changed_files=tuple(sorted(changed)),
            stages=tuple(stage.name for stage in self.patchset.stages),
        )

    def commit(self, planned: Mapping[str, str], result: PatchResult) -> PatchResult:
        # Re-plan immediately before preparing writes. This makes commit reject
        # source, identity, review-artifact, or semantic-contract drift that
        # occurred after the caller obtained its plan.
        current_plan, current_result = self.plan()
        _require(
            current_result == result and current_plan == dict(planned),
            f"{self.patchset.name}: source changed between plan and commit; "
            "no writes performed",
        )
        if current_result.state == "already-applied":
            return current_result

        temporary: dict[str, Path] = {}
        backups: dict[str, _Backup] = {}
        replaced: list[str] = []
        try:
            # Capture every original before creating any temporary output. The
            # bytes and permission bits form the optimistic-concurrency guard
            # checked again immediately before each replacement.
            for relative in result.changed_files:
                destination = self.source_root.joinpath(
                    *_safe_relative_path(relative).parts
                )
                if destination.exists():
                    self._path(relative, existing=True)
                    info = destination.stat()
                    backups[relative] = _Backup(
                        destination.read_bytes(), stat.S_IMODE(info.st_mode)
                    )
                else:
                    self._path(relative, existing=False)
                    backups[relative] = _Backup(None, None)

            for relative in result.changed_files:
                destination = self.source_root.joinpath(
                    *_safe_relative_path(relative).parts
                )
                backup = backups[relative]
                descriptor, raw_temp = tempfile.mkstemp(
                    prefix=f".{destination.name}.qwen-source-patch.",
                    dir=destination.parent,
                )
                temp_path = Path(raw_temp)
                temporary[relative] = temp_path
                try:
                    data = planned[relative].encode("utf-8")
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temp_path, backup.mode or 0o644)
                except BaseException:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    raise

            for relative in result.changed_files:
                destination = self.source_root.joinpath(
                    *_safe_relative_path(relative).parts
                )
                backup = backups[relative]
                if backup.data is None:
                    _require(
                        not destination.exists() and not destination.is_symlink(),
                        f"{self.patchset.name}: new destination {relative} appeared "
                        "during commit",
                    )
                else:
                    self._path(relative, existing=True)
                    info = destination.stat()
                    _require(
                        destination.read_bytes() == backup.data
                        and stat.S_IMODE(info.st_mode) == backup.mode,
                        f"{self.patchset.name}: {relative} changed during commit",
                    )
                # Review evidence is part of the transaction's trusted input,
                # so it is rechecked at the final mutation boundary as well.
                for stage in self.patchset.stages:
                    self._verify_review_artifact(stage)
                os.replace(temporary[relative], destination)
                replaced.append(relative)
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)

            verified = self._read_state()
            _require(
                self._matches(verified, self.patchset.final_files),
                f"{self.patchset.name}: post-write final manifest mismatch",
            )
            self.patchset.validate_final(verified)
        except BaseException as exc:
            rollback_errors: list[str] = []
            for relative in reversed(replaced):
                destination = self.source_root.joinpath(
                    *_safe_relative_path(relative).parts
                )
                prior = backups[relative]
                try:
                    if prior.data is None:
                        destination.unlink(missing_ok=True)
                    else:
                        descriptor, raw_restore = tempfile.mkstemp(
                            prefix=f".{destination.name}.qwen-rollback.",
                            dir=destination.parent,
                        )
                        restore = Path(raw_restore)
                        with os.fdopen(descriptor, "wb") as handle:
                            handle.write(prior.data)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.chmod(restore, prior.mode or 0o644)
                        os.replace(restore, destination)
                    directory_fd = os.open(destination.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except BaseException as rollback_exc:
                    rollback_errors.append(f"{relative}: {rollback_exc!r}")
            detail = ""
            if rollback_errors:
                detail = f"; rollback errors: {rollback_errors!r}"
            raise PatchWriteError(
                f"{self.patchset.name}: commit failed after full validation: "
                f"{exc!r}{detail}. The caller must discard this disposable tree."
            ) from exc
        finally:
            for temp_path in temporary.values():
                temp_path.unlink(missing_ok=True)
        return PatchResult(
            state="applied",
            changed_files=result.changed_files,
            stages=result.stages,
        )

    def apply(self) -> PatchResult:
        planned, result = self.plan()
        return self.commit(planned, result)
