"""Transactional, landmark-aware source transformations.

The source patchers in this project are deliberately stricter than a unified
diff application.  Every patch set is tied to exact source identities, every
edit names and validates the source block it understands, and the complete
result is planned and checked before a disposable source tree is changed.

Unified diffs remain review evidence.  They are parsed independently and must
describe the exact same old/new blocks as the Python transformation data, at
the same place: every hunk's header names where its old block starts in the
file before the stage and where its new block starts after it, and how many
lines each holds, and its body holds exactly those lines; every index line
names the blobs the stage transforms; every created or deleted file is
declared as one; and every mode a section states is the file's mode in the
tree the stage transforms.  Each of these is proven against the replay of the
stage data, never taken from the diff.  They are never used to decide where or
how to edit a file.
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


def git_blob_id(text: str) -> str:
    """The object id git gives a file's bytes, as a review diff's index line names it."""
    data = text.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# Digest of the empty file. A deletion of an empty file cannot carry a
# landmark hunk (its before and after blocks would both be empty), so it is
# the one transformation evidenced purely by the review diff's
# "deleted file mode" header instead of by hunks.
EMPTY_FILE_SHA256 = sha256_text("")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise PatchRefusedError(message)


def after_outside_landmark(current: str, before: str, after: str) -> int:
    """How often *after* occurs in *current* other than wholly inside the one
    occurrence of *before* that an edit replaces.

    Any such occurrence means the source already holds the edit's result, or
    part of it, so the edit is refused.  An occurrence wholly inside that
    block is the block itself: the text a deletion keeps is part of the text
    it replaces, as it always is for a deletion that ends a file.  Every
    occurrence is counted, overlapping ones included, so one that straddles
    the block's edge counts as outside it.
    """
    start = current.index(before)
    end = start + len(before)
    outside = 0
    position = current.find(after)
    while position != -1:
        if position < start or position + len(after) > end:
            outside += 1
        position = current.find(after, position + 1)
    return outside


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
        _require(
            self.review_before == "" or self.before.count(self.review_before) == 1,
            f"{self.name}: the review-before block occurs more than once inside "
            "its landmark, so the landmark does not say where it is",
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
    """before None = file created; after None = file deleted."""

    path: str
    before_sha256: str | None
    after_sha256: str | None

    def __post_init__(self) -> None:
        _safe_relative_path(self.path)
        if self.before_sha256 is not None:
            _require(
                bool(re.fullmatch(r"[0-9a-f]{64}", self.before_sha256)),
                f"{self.path}: invalid before SHA-256",
            )
        if self.after_sha256 is not None:
            _require(
                bool(re.fullmatch(r"[0-9a-f]{64}", self.after_sha256)),
                f"{self.path}: invalid after SHA-256",
            )
        _require(
            self.before_sha256 is not None or self.after_sha256 is not None,
            f"{self.path}: a file cannot be both absent before and after",
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
        paths_requiring_edits = {
            contract.path
            for contract in self.files
            if not (
                contract.after_sha256 is None
                and contract.before_sha256 == EMPTY_FILE_SHA256
            )
        }
        _require(
            edit_paths == paths_requiring_edits,
            f"{self.name}: edit/file sets disagree: "
            f"edits={sorted(edit_paths)!r} files={sorted(paths_requiring_edits)!r}",
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

    def worktree_status(self) -> tuple[str, ...]:
        """How ``git status --short --untracked-files=all`` reports the result.

        This is the patch set's whole footprint on a checkout of its source
        revision: a path is listed when its final state differs from its
        pristine one -- `` M`` changed, `` D`` deleted, ``??`` created -- and
        every other path is untouched. The pristine state of a path is the
        first identity a stage requires of it, and the final state is
        ``final_files`` (absent for a deleted path), so the listing is derived
        from the data the transaction proves rather than kept beside it.
        Tracked paths come first and untracked ones after, each in path order,
        as git lists them.
        """
        pristine: dict[str, str | None] = {}
        for stage in self.stages:
            for contract in stage.files:
                pristine.setdefault(contract.path, contract.before_sha256)
        tracked: list[str] = []
        untracked: list[str] = []
        for path in sorted(pristine):
            before, after = pristine[path], self.final_files.get(path)
            if before is None:
                if after is not None:
                    untracked.append(f"?? {path}")
            elif after is None:
                tracked.append(f" D {path}")
            elif after != before:
                tracked.append(f" M {path}")
        return tuple(tracked + untracked)


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
    """One hunk: its blocks, and the coordinates its header states."""

    path: str
    before: str
    after: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass(frozen=True)
class _ParsedReviewFile:
    """What one file's section of a review diff declares about the file."""

    path: str
    created: bool
    deleted: bool
    # The abbreviated blob ids of the file before and after the stage, when the
    # section writes an index line.
    blobs: tuple[str, str] | None
    # The git file mode the section states (its new or deleted file mode, or
    # its index line's), when it states one.
    mode: str | None = None


_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
# git omits a count of one; a count of zero names the line before the range.
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
_DELETED_FILE_HEADER = re.compile(r"^deleted file mode (\d{6})$")
_NEW_FILE_HEADER = re.compile(r"^new file mode (\d{6})$")
_INDEX_HEADER = re.compile(
    r"^index ([0-9a-f]{7,40})\.\.([0-9a-f]{7,40})(?: (\d{6}))?$"
)
_OLD_FILE_HEADER = re.compile(r"^--- (?:a/(.+)|(/dev/null))$")
_NEW_FILE_PATH_HEADER = re.compile(r"^\+\+\+ (?:b/(.+)|(/dev/null))$")
# The modes git gives a regular file. The framework edits regular files only
# and never changes a mode, so these are the only modes a section can state.
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})


def git_file_mode(st_mode: int) -> str:
    """The mode git records for a regular file, from its permission bits."""
    return "100755" if st_mode & stat.S_IXUSR else "100644"


def git_range_start(first_line: int, count: int) -> int:
    """The start git writes for a range of *count* lines whose first line is
    *first_line*: an empty range is named by the line before it."""
    return first_line if count else first_line - 1


def _parse_review_diff(
    data: bytes, *, label: str
) -> tuple[tuple[_ParsedReviewEdit, ...], dict[str, _ParsedReviewFile]]:
    """Parse a review diff in git's unified grammar, and nothing looser.

    Each file section is a ``diff --git`` header, then at most one new or
    deleted file mode, at most one index line, and -- when the section has
    hunks -- the ``---``/``+++`` pair and its hunks. A hunk's body is
    delimited by its header's counts, as git delimits it: it holds exactly
    the counted old and new lines, and what follows is a hunk or a section.
    Any other line is refused, so nothing a review diff states goes unread.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchRefusedError(f"{label}: review diff is not UTF-8: {exc}") from exc
    _require("\r" not in text, f"{label}: review diff contains CR bytes")
    lines = text.splitlines(keepends=True)
    edits: list[_ParsedReviewEdit] = []
    files: dict[str, _ParsedReviewFile] = {}
    current_path: str | None = None
    # Where in its section's grammar the parser stands: "header" (mode and
    # index lines may follow), "old" / "new" (the path pair is open), "hunks".
    phase = "none"
    path_pair: tuple[str | None, str | None] = (None, None)
    index = 0

    def declare(**declared: object) -> None:
        files[current_path] = _ParsedReviewFile(
            **{**files[current_path].__dict__, **declared}
        )

    def declare_mode(mode: str, where: str) -> None:
        _require(
            mode in _REGULAR_FILE_MODES,
            f"{label}: {current_path} {where} states mode {mode}, which is "
            "not a regular file's",
        )
        stated = files[current_path].mode
        _require(
            stated in (None, mode),
            f"{label}: {current_path} states modes {stated} and {mode}",
        )
        declare(mode=mode)

    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip("\n")
        match = _DIFF_HEADER.match(line)
        if match:
            _require(
                phase in ("none", "header", "hunks"),
                f"{label}: {current_path} opens its path pair without a hunk",
            )
            _require(
                match.group(1) == match.group(2),
                f"{label}: rename/copy diffs are not supported: {raw!r}",
            )
            current_path = match.group(1)
            _safe_relative_path(current_path)
            _require(
                current_path not in files,
                f"{label}: {current_path} has more than one section",
            )
            files[current_path] = _ParsedReviewFile(
                current_path, created=False, deleted=False, blobs=None
            )
            phase = "header"
            index += 1
            continue
        _require(
            current_path is not None,
            f"{label}: line before the first file section: {raw!r}",
        )
        section = files[current_path]
        if phase == "header":
            deleted = _DELETED_FILE_HEADER.match(line)
            created = _NEW_FILE_HEADER.match(line)
            index_match = _INDEX_HEADER.match(line)
            if (deleted or created) and not (
                section.created or section.deleted or section.blobs
            ):
                declare(deleted=bool(deleted), created=bool(created))
                declare_mode(
                    (deleted or created).group(1),
                    "deleted file mode" if deleted else "new file mode",
                )
                index += 1
                continue
            if index_match and section.blobs is None:
                declare(blobs=(index_match.group(1), index_match.group(2)))
                if index_match.group(3) is not None:
                    declare_mode(index_match.group(3), "index line")
                index += 1
                continue
            old = _OLD_FILE_HEADER.match(line)
            if old:
                path_pair = (old.group(1), old.group(2))
                phase = "old"
                index += 1
                continue
        elif phase == "old":
            new = _NEW_FILE_PATH_HEADER.match(line)
            _require(new, f"{label}: {current_path}: '---' without '+++'")
            old_path, old_null = path_pair
            new_path, new_null = new.group(1), new.group(2)
            _require(
                old_path in (None, current_path) and new_path in (None, current_path)
                and not (old_null and new_null),
                f"{label}: {current_path}'s path pair names another file",
            )
            _require(
                bool(old_null) == section.created
                and bool(new_null) == section.deleted,
                f"{label}: {current_path}'s path pair says "
                f"{'created' if old_null else 'deleted' if new_null else 'modified'}"
                ", but its header does not",
            )
            phase = "new"
            index += 1
            continue
        hunk_match = _HUNK_HEADER.match(line)
        _require(
            hunk_match and phase in ("new", "hunks"),
            f"{label}: {current_path}: unsupported review diff line {raw!r}",
        )
        old_start, old_count, new_start, new_count = (
            int(hunk_match.group(1)),
            int(hunk_match.group(2) or 1),
            int(hunk_match.group(3)),
            int(hunk_match.group(4) or 1),
        )
        before: list[str] = []
        after: list[str] = []
        index += 1
        old_left, new_left = old_count, new_count
        while old_left or new_left:
            _require(
                index < len(lines),
                f"{label}: {current_path}: a hunk ends {old_left} old and "
                f"{new_left} new line(s) short of its header's counts",
            )
            hunk_line = lines[index]
            if hunk_line.startswith("\\ No newline at end of file"):
                raise PatchRefusedError(
                    f"{label}: files without terminal newline are unsupported"
                )
            prefix = hunk_line[:1]
            payload = hunk_line[1:]
            _require(
                prefix in {" ", "+", "-"},
                f"{label}: {current_path}: a hunk ends {old_left} old and "
                f"{new_left} new line(s) short of its header's counts at "
                f"{hunk_line!r}",
            )
            if prefix in {" ", "-"}:
                _require(
                    old_left > 0,
                    f"{label}: {current_path}: a hunk holds more old lines "
                    f"than its header's {old_count}",
                )
                old_left -= 1
                before.append(payload)
            if prefix in {" ", "+"}:
                _require(
                    new_left > 0,
                    f"{label}: {current_path}: a hunk holds more new lines "
                    f"than its header's {new_count}",
                )
                new_left -= 1
                after.append(payload)
            index += 1
        edits.append(
            _ParsedReviewEdit(
                current_path,
                "".join(before),
                "".join(after),
                old_start,
                old_count,
                new_start,
                new_count,
            )
        )
        phase = "hunks"
    _require(
        phase in ("none", "header", "hunks"),
        f"{label}: {current_path} opens its path pair without a hunk",
    )
    _require(
        edits or any(file.deleted for file in files.values()),
        f"{label}: review diff contains no hunks",
    )
    return tuple(edits), files


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
        if data == b"":
            # An empty file is a well-defined fixed point with no line
            # structure to police; refusing it would make empty pristine
            # files (package __init__ markers) untouchable by deletions.
            return ""
        _require(b"\x00" not in data, f"{relative}: NUL byte in text source")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PatchRefusedError(f"{relative}: source is not UTF-8: {exc}") from exc
        _require("\r" not in text, f"{relative}: CR bytes are forbidden")
        _require(text.endswith("\n"), f"{relative}: no terminal newline")
        return text

    def _deleted_paths(self) -> set[str]:
        return {
            contract.path
            for stage in self.patchset.stages
            for contract in stage.files
            if contract.after_sha256 is None
        }

    def _all_paths(self) -> tuple[str, ...]:
        paths: set[str] = set(self.patchset.identity_files)
        paths.update(self.patchset.final_files)
        for stage in self.patchset.stages:
            paths.update(contract.path for contract in stage.files)
        return tuple(sorted(paths))

    def _read_state(self) -> dict[str, str]:
        state: dict[str, str] = {}
        absent_ok = {
            contract.path
            for stage in self.patchset.stages
            for contract in stage.files
            if contract.before_sha256 is None
        }
        absent_ok |= self._deleted_paths()
        for path in self._all_paths():
            candidate = self.source_root.joinpath(*PurePosixPath(path).parts)
            if not candidate.exists() and path in absent_ok:
                continue
            state[path] = self._read_text(path)
        return state

    def _read_modes(self) -> dict[str, str]:
        """The git mode of every file the patch set names that exists."""
        modes: dict[str, str] = {}
        for path in self._all_paths():
            candidate = self.source_root.joinpath(*PurePosixPath(path).parts)
            if candidate.exists():
                modes[path] = git_file_mode(candidate.lstat().st_mode)
        return modes

    def _verify_identity(self, state: Mapping[str, str]) -> None:
        for path, digest in self.patchset.identity_files.items():
            _require(path in state, f"identity file missing: {path}")
            actual = sha256_text(state[path])
            _require(
                actual == digest,
                f"{self.patchset.name}: identity drift in {path}; "
                f"expected {digest}, got {actual}",
            )

    def _verify_review_artifact(
        self, stage: PatchStage
    ) -> tuple[tuple[_ParsedReviewEdit, ...], dict[str, _ParsedReviewFile]]:
        path = self._artifact_path(stage.review_patch)
        data = path.read_bytes()
        actual_digest = sha256_bytes(data)
        _require(
            actual_digest == stage.review_sha256,
            f"{stage.name}: review diff SHA-256 drift; expected "
            f"{stage.review_sha256}, got {actual_digest}",
        )
        parsed, declared = _parse_review_diff(data, label=stage.name)
        _require(
            len(parsed) == len(stage.edits),
            f"{stage.name}: review diff has {len(parsed)} hunks but the Python "
            f"patcher has {len(stage.edits)} landmark transformations",
        )
        _require(
            sorted(declared) == sorted(contract.path for contract in stage.files),
            f"{stage.name}: review diff has sections for {sorted(declared)!r} "
            "but the Python patcher changes "
            f"{sorted(contract.path for contract in stage.files)!r}",
        )
        for kind, parsed_paths, stage_paths in (
            (
                "deletions",
                {path for path, file in declared.items() if file.deleted},
                {c.path for c in stage.files if c.after_sha256 is None},
            ),
            (
                "creations",
                {path for path, file in declared.items() if file.created},
                {c.path for c in stage.files if c.before_sha256 is None},
            ),
        ):
            _require(
                parsed_paths == stage_paths,
                f"{stage.name}: review diff declares {kind} "
                f"{sorted(parsed_paths)!r} but the Python patcher declares "
                f"{sorted(stage_paths)!r}",
            )
        # The blocks are the stage data's; a hunk's counts are its body's, by
        # the parse. Where each hunk starts is proven by the replay (plan).
        _require(
            [(edit.path, edit.before, edit.after) for edit in parsed]
            == [
                (edit.path, edit.review_before, edit.review_after)
                for edit in stage.edits
            ],
            f"{stage.name}: Python landmarks and review diff describe "
            "different transformations",
        )
        return parsed, declared

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
        for path in sorted(self._deleted_paths()):
            if path not in state:
                rows.append(f"{path}=deleted")
                continue
            digest = sha256_text(state[path])
            kind = "pristine" if pristine.get(path) == digest else "unknown"
            rows.append(f"{path}={kind}:{digest}")
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
        reviews = {
            stage.name: self._verify_review_artifact(stage)
            for stage in self.patchset.stages
        }

        deleted_paths = self._deleted_paths()
        if self._matches(state, self.patchset.final_files) and all(
            path not in state for path in deleted_paths
        ):
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
        # The framework writes a file it creates 0644 and keeps every other
        # file's mode, so a file's mode through every stage is its mode here.
        modes = self._read_modes()
        for stage in self.patchset.stages:
            stage.validate_before(planned)
            contracts = {contract.path: contract for contract in stage.files}
            coordinates, declared = reviews[stage.name]
            for path, review_file in declared.items():
                actual_mode = modes.get(path, "100644")
                _require(
                    review_file.mode in (None, actual_mode),
                    f"{stage.name}: the review diff states mode "
                    f"{review_file.mode} for {path}, whose mode is "
                    f"{actual_mode}; no writes performed",
                )
            # Per file, how far this stage's earlier hunks moved later lines,
            # and the line after the last one's new block.
            shift: dict[str, int] = {}
            end: dict[str, int] = {}
            stage_before = {path: planned.get(path) for path in contracts}
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

            for edit, coordinate in zip(stage.edits, coordinates):
                current = planned.get(edit.path, "")
                if edit.after == "":
                    # Deleted-file hunk: by the LandmarkEdit grammar its
                    # before block describes the complete file, so identity
                    # is exact equality; the substring-count discipline is
                    # meaningless against an empty after block (the empty
                    # string occurs everywhere).
                    _require(
                        current == edit.before,
                        f"{stage.name}:{edit.name}: deleted-file landmark "
                        f"does not match the whole of {edit.path}; no "
                        "writes performed",
                    )
                    landed = 0
                else:
                    before_count = current.count(edit.before)
                    _require(
                        before_count == 1,
                        f"{stage.name}:{edit.name}: expected one before landmark "
                        f"in {edit.path}, found {before_count}; no writes performed",
                    )
                    after_count = after_outside_landmark(
                        current, edit.before, edit.after
                    )
                    _require(
                        after_count == 0,
                        f"{stage.name}:{edit.name}: after block already appears "
                        f"{after_count} time(s) in {edit.path} outside its before "
                        "landmark; source is partial or the landmarks overlap; no "
                        "writes performed",
                    )
                    # A created file's hunk has no old block, and starts the file.
                    landed = (
                        current.index(edit.before)
                        + len(edit.before.split(edit.review_before, 1)[0])
                        if edit.review_before
                        else 0
                    )
                # Where the edit lands is proven, not only what it says: a
                # file's hunks apply in order, so the text this one meets holds
                # the earlier ones' results and none of the later ones', which
                # lie below it. Its new block starts where it lands, and its
                # old block there less what the earlier hunks moved it; git
                # names an empty block by the line before it.
                line = current.count("\n", 0, landed) + 1
                _require(
                    line >= end.get(edit.path, 1),
                    f"{stage.name}:{edit.name}: its review hunk starts inside or "
                    f"above the one before it in {edit.path}; no writes performed",
                )
                _require(
                    coordinate.new_start == git_range_start(line, coordinate.new_count),
                    f"{stage.name}:{edit.name}: lands at line {line} of "
                    f"{edit.path}, but its review hunk's header places it at "
                    f"line {coordinate.new_start}; no writes performed",
                )
                old_start = git_range_start(
                    line - shift.get(edit.path, 0), coordinate.old_count
                )
                _require(
                    coordinate.old_start == old_start,
                    f"{stage.name}:{edit.name}: starts at line {old_start} of "
                    f"{edit.path} before the stage, but its review hunk's header "
                    f"says line {coordinate.old_start}; no writes performed",
                )
                shift[edit.path] = (
                    shift.get(edit.path, 0) + coordinate.new_count - coordinate.old_count
                )
                end[edit.path] = line + coordinate.new_count
                planned[edit.path] = (
                    "" if edit.after == ""
                    else current.replace(edit.before, edit.after, 1)
                )
                changed.add(edit.path)

            for path, contract in contracts.items():
                _require(path in planned, f"{stage.name}: failed to produce {path}")
                if contract.after_sha256 is None:
                    _require(
                        planned[path] == "",
                        f"{stage.name}: deletion of {path} left residual "
                        "content; no writes performed",
                    )
                    del planned[path]
                    changed.add(path)
                    continue
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
            for path, review_file in declared.items():
                if review_file.blobs is None:
                    continue
                transformed = tuple(
                    git_blob_id(text) if text is not None else "0" * 40
                    for text in (stage_before[path], planned.get(path))
                )
                _require(
                    all(
                        blob.startswith(named)
                        for blob, named in zip(transformed, review_file.blobs)
                    ),
                    f"{stage.name}: the review diff's index line for {path} names "
                    f"{'..'.join(review_file.blobs)}, but the stage transforms "
                    f"{transformed[0][:len(review_file.blobs[0])]}.."
                    f"{transformed[1][:len(review_file.blobs[1])]}; no writes performed",
                )
            stage.validate_after(planned)

        _require(
            self._matches(planned, self.patchset.final_files),
            f"{self.patchset.name}: complete result does not match final manifest; "
            "no writes performed",
        )
        _require(
            all(path not in planned for path in deleted_paths),
            f"{self.patchset.name}: a later stage resurrected a deleted file; "
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
                if relative not in planned:
                    # Planned deletion: nothing to stage; the unlink happens
                    # at the same optimistic-concurrency boundary as writes.
                    continue
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
                if relative not in planned:
                    os.unlink(destination)
                else:
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
            _require(
                all(path not in verified for path in self._deleted_paths()),
                f"{self.patchset.name}: post-write deletion still present",
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
