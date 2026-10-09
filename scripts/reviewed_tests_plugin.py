"""The reviewed-test runner's pytest plugin, and the merge of its shards' reports.

scripts/build-vllm.sh loads it into pytest (``-p reviewed_tests_plugin``) and
runs it as a script to merge what its shards report.

The reviewed test files are the test_*.py files the patch set changes or adds;
build-vllm.sh derives them from the verified stage data and names them in
REVIEWED_TEST_FILES. pytest is pointed at the tree's tests/ directory, and this
plugin collects those files and nothing else: no other test file, and no
conftest.py of a directory that holds none of them, is imported.
``--reviewed-shard I/N`` takes the I-th of N shares of them, by sorted position.

A test that needs what a machine may lack says so itself, with the mark the
tree's tests/conftest.py registers for it -- ``gpu`` or ``network`` -- on the
test, its class, or its module's ``pytestmark``. Nothing else decides where a
test runs:

  --reviewed-set offline  every reviewed test that declares neither, run with
                          no GPU and no network (check). A module whose
                          pytestmark names a need is not imported at all, so a
                          module that cannot even be imported without a GPU
                          says so the same way. A test that needs one and does
                          not say so fails here, which is what makes it say so.
  --reviewed-set release  every reviewed test, in the built image, on the GPU
                          and with the network (build).
  --reviewed-set collection
                          every reviewed test file imported and its tests built
                          as the release set collects them, and none run, with
                          no GPU and no network (check). A collection error
                          fails the run, so a module that cannot be collected
                          is found by check rather than by a build that can then
                          pin nothing. The one exception is a module that
                          declares a GPU in its module-level pytestmark and
                          cannot be imported without one: it is excused only
                          when its imports and its declaration -- the module's
                          source up to and including that pytestmark -- run
                          here, so what fails is what follows the declaration
                          of the need, and it is named with its error.

A reviewed file that runs no test, skips no module and declares no need fails
the run: nothing the runner is given goes unexecuted unsaid. A runtime module the
patch set deletes (REVIEWED_DELETED_RUNTIME_FILES) cannot be imported: the image
does not carry it, so no test may find it either.

Each shard ends its output with one ``REVIEWED_TESTS_JSON`` line; ``python3
reviewed_tests_plugin.py merge LOG...`` adds the shards up and prints what the
run did and did not execute, the lines build-vllm.sh repeats in its summary.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

NEEDS = ("gpu", "network")
OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")


def _paths(variable: str) -> tuple[str, ...]:
    return tuple(os.environ.get(variable, "").split())


class _DeletedRuntimeModules(importlib.abc.MetaPathFinder):
    def __init__(self, paths: tuple[str, ...]) -> None:
        self.modules = frozenset(
            path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
            for path in paths
        )

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            raise ModuleNotFoundError(
                f"No module named {fullname!r}: the reviewed patch set deletes it",
                name=fullname,
            )
        return None


def _declared_needs(module: Path) -> tuple[str, ...]:
    """The needs a module's top-level ``pytestmark`` names, read without
    importing it."""
    tree = ast.parse(module.read_bytes(), filename=str(module))
    needs: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr in NEEDS
                    and isinstance(sub.value, ast.Attribute)
                    and sub.value.attr == "mark"
                ):
                    needs.add(sub.attr)
    return tuple(need for need in NEEDS if need in needs)


def _declaration_runs(module: Path, root: Path) -> str | None:
    """Run a module's source up to and including its module-level
    ``pytestmark``, in a module of the name pytest imports it under; None when
    that runs, or the error that stopped it."""
    source = module.read_text()
    ends = [
        node.end_lineno
        for node in ast.parse(source, filename=str(module)).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        )
    ]
    if not ends:
        return "the module declares nothing at module level"
    head = "".join(source.splitlines(keepends=True)[: ends[-1]])
    package = module.parent
    parts = [module.stem]
    while (package / "__init__.py").exists():
        parts.insert(0, package.name)
        package = package.parent
    spec = importlib.util.spec_from_file_location(".".join(parts), module)
    namespace = importlib.util.module_from_spec(spec)
    try:
        exec(compile(head, str(module), "exec"), namespace.__dict__)
    except BaseException as error:  # noqa: BLE001 - the error is the answer
        return f"{type(error).__name__}: {error}"
    return None


if __name__ != "__main__":
    import pytest

    sys.meta_path.insert(
        0, _DeletedRuntimeModules(_paths("REVIEWED_DELETED_RUNTIME_FILES"))
    )
    _skipped_modules: set[str] = set()
    _uncollected_modules: dict[str, str] = {}

    class _Run:
        def __init__(self, config) -> None:
            self.reviewed_set = config.getoption("reviewed_set")
            index, _, count = config.getoption("reviewed_shard").partition("/")
            if not (
                index.isdigit() and count.isdigit() and 0 < int(index) <= int(count)
            ):
                raise pytest.UsageError("--reviewed-shard takes I/N, 1 <= I <= N")
            every = sorted(_paths("REVIEWED_TEST_FILES"))
            if not every:
                raise pytest.UsageError("REVIEWED_TEST_FILES names no reviewed test file")
            mine = every[int(index) - 1 :: int(count)]
            self.root = config.rootpath
            self.reviewed = {self.root / relative: relative for relative in mine}
            self.support = _paths("REVIEWED_TEST_SUPPORT_FILES")
            self.declared_modules: dict[str, tuple[str, ...]] = {}
            self.declared_tests: Counter[str] = Counter()
            self.collected_files: set[str] = set()
            self.collected_tests = 0
            self.unaccounted: list[str] = []
            self.gpu_only: dict[str, str] = {}
            self.collection_errors: dict[str, str] = {}

        def wanted(self, path: Path) -> bool | None:
            if path.is_dir():
                return any(file.is_relative_to(path) for file in self.reviewed)
            if path.name.startswith("test_") and path.suffix == ".py":
                return path in self.reviewed
            return None

    _RUN = pytest.StashKey[_Run]()

    def pytest_addoption(parser):
        parser.addoption(
            "--reviewed-set",
            dest="reviewed_set",
            choices=("offline", "release", "collection"),
            required=True,
            help="offline: the reviewed tests that declare no need, with no GPU "
            "and no network (check). release: every reviewed test, on the GPU "
            "with the network (build). collection: every reviewed test file "
            "collected as release collects it, and none run (check).",
        )
        parser.addoption(
            "--reviewed-shard",
            dest="reviewed_shard",
            default="1/1",
            help="I/N: the I-th of N shares of the reviewed files, by sorted "
            "position.",
        )

    def pytest_configure(config):
        run = _Run(config)
        config.stash[_RUN] = run
        if run.reviewed_set == "collection":
            config.option.collectonly = True
            config.option.continue_on_collection_errors = True

    def pytest_ignore_collect(collection_path, config):
        run = config.stash[_RUN]
        path = Path(collection_path)
        wanted = run.wanted(path)
        if wanted is False:
            return True
        if wanted and path.is_file() and run.reviewed_set == "offline":
            needs = _declared_needs(path)
            if needs:
                run.declared_modules[run.reviewed[path]] = needs
                return True
        return None

    def pytest_collectreport(report):
        # A module that skips itself at import ran: its skip is its outcome,
        # and its reason is reported with the others.
        if report.skipped and report.nodeid.endswith(".py"):
            _skipped_modules.add(report.nodeid)
        if report.failed and report.nodeid.endswith(".py"):
            lines = [line for line in str(report.longrepr).splitlines() if line.strip()]
            errors = [line[1:].strip() for line in lines if line.startswith("E ")]
            _uncollected_modules[report.nodeid] = (errors or lines or ["no error text"])[-1]

    def pytest_collection_modifyitems(config, items):
        run = config.stash[_RUN]
        for item in items:
            for mark in item.iter_markers():
                if mark.name in NEEDS and (mark.args or mark.kwargs):
                    raise pytest.UsageError(
                        f"{item.nodeid}: the {mark.name} mark takes no arguments; "
                        "it states only that the test needs one"
                    )
        run.collected_files = {
            item.path.relative_to(run.root).as_posix() for item in items
        }
        run.collected_tests = len(items)
        if run.reviewed_set != "offline":
            return
        kept, declared = [], []
        for item in items:
            if any(item.get_closest_marker(need) for need in NEEDS):
                declared.append(item)
                run.declared_tests[item.path.relative_to(run.root).as_posix()] += 1
            else:
                kept.append(item)
        if declared:
            config.hook.pytest_deselected(items=declared)
            items[:] = kept

    def pytest_sessionfinish(session, exitstatus):
        run = session.config.stash[_RUN]
        for relative, error in sorted(_uncollected_modules.items()):
            module = run.root / relative
            if run.reviewed_set == "collection" and "gpu" in _declared_needs(module):
                stopped = _declaration_runs(module, run.root)
                if stopped is None:
                    run.gpu_only[relative] = error
                    continue
                error = f"{error}; its imports and declaration stop here too: {stopped}"
            run.collection_errors[relative] = error
        accounted = (
            run.collected_files
            | set(run.declared_modules)
            | _skipped_modules
            | set(_uncollected_modules)
        )
        run.unaccounted = sorted(set(run.reviewed.values()) - accounted)
        if run.reviewed_set == "collection":
            # Collection errors stop nothing in this set, so the set decides:
            # every module collected, or excused by the need it declares.
            session.exitstatus = (
                pytest.ExitCode.TESTS_FAILED
                if run.collection_errors or run.unaccounted
                else pytest.ExitCode.OK
            )
        elif run.unaccounted and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _reason(report) -> str:
        if getattr(report, "wasxfail", ""):
            return report.wasxfail
        if isinstance(report.longrepr, tuple):
            return report.longrepr[2].removeprefix("Skipped: ")
        return str(report.longrepr).splitlines()[0]

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(terminalreporter, exitstatus, config):
        run = config.stash[_RUN]
        stats = terminalreporter.stats
        files = {
            report.nodeid.split("::", 1)[0]
            for outcome in OUTCOMES
            for report in stats.get(outcome, [])
            if report.nodeid
        }
        imported = {
            Path(module.__file__).resolve()
            for module in list(sys.modules.values())
            if getattr(module, "__file__", None)
        }
        for relative in run.unaccounted:
            terminalreporter.write_line(
                f"REVIEWED_TESTS NOT RUN: {relative} ran no test, skipped no "
                "module and declares no need"
            )
        for relative, error in run.collection_errors.items():
            terminalreporter.write_line(
                f"REVIEWED_TESTS NOT COLLECTED: {relative}: {error}"
            )
        report = {
            "set": run.reviewed_set,
            "reviewed_files": len(run.reviewed),
            "ran_files": sorted(files),
            "counts": {outcome: len(stats.get(outcome, [])) for outcome in OUTCOMES},
            "reasons": {
                outcome: Counter(_reason(r) for r in stats.get(outcome, []))
                for outcome in ("skipped", "xfailed")
            },
            "declared_modules": run.declared_modules,
            "declared_tests": dict(run.declared_tests),
            "support": list(run.support),
            "support_imported": sorted(
                r for r in run.support if (run.root / r).resolve() in imported
            ),
            "unaccounted": run.unaccounted,
            "collected_files": len(run.collected_files),
            "collected_tests": run.collected_tests,
            "gpu_only": run.gpu_only,
            "skipped_modules": sorted(_skipped_modules),
            "collection_errors": run.collection_errors,
        }
        terminalreporter.write_line("REVIEWED_TESTS_JSON " + json.dumps(report))


def _merge(logs: list[str]) -> int:
    reports = []
    for log in logs:
        lines = [
            line.removeprefix("REVIEWED_TESTS_JSON ")
            for line in Path(log).read_text(errors="replace").splitlines()
            if line.startswith("REVIEWED_TESTS_JSON ")
        ]
        if len(lines) != 1:
            print(f"REVIEWED_TESTS NOT RUN: {log} holds {len(lines)} reports, not one")
            return 1
        reports.append(json.loads(lines[0]))
    (reviewed_set,) = {r["set"] for r in reports}
    counts: Counter[str] = Counter()
    reasons = {"skipped": Counter(), "xfailed": Counter()}
    declared_modules: dict[str, list[str]] = {}
    declared_tests: Counter[str] = Counter()
    ran, imported, unaccounted = set(), set(), []
    gpu_only: dict[str, str] = {}
    collection_errors: dict[str, str] = {}
    collected_files = collected_tests = 0
    skipped_modules: set[str] = set()
    for r in reports:
        skipped_modules.update(r["skipped_modules"])
        gpu_only.update(r["gpu_only"])
        collection_errors.update(r["collection_errors"])
        collected_files += r["collected_files"]
        collected_tests += r["collected_tests"]
        counts.update(r["counts"])
        for outcome, counter in reasons.items():
            counter.update(r["reasons"][outcome])
        declared_modules.update(r["declared_modules"])
        declared_tests.update(r["declared_tests"])
        ran.update(r["ran_files"])
        imported.update(r["support_imported"])
        unaccounted += r["unaccounted"]
    for outcome, counter in reasons.items():
        for reason, count in counter.most_common():
            print(f"REVIEWED_TESTS {outcome} {count}: {reason}")
    for relative in sorted(unaccounted):
        print(f"REVIEWED_TESTS NOT RUN: {relative}")
    for relative, error in sorted(collection_errors.items()):
        print(f"REVIEWED_TESTS NOT COLLECTED: {relative}: {error}")
    reviewed = sum(r["reviewed_files"] for r in reports)
    if reviewed_set == "collection":
        text = (
            f"Reviewed test collection: of the {reviewed} reviewed test files, "
            "check collected as build's release set collects them, with no GPU "
            f"and no network, {collected_files}, {collected_tests} tests, and ran "
            "none"
        )
        if skipped_modules:
            text += (
                f"; {len(skipped_modules)} skip themselves at import here, so only "
                "build collects what follows the skip: "
                + ", ".join(sorted(skipped_modules))
            )
        if gpu_only:
            text += (
                f"; {len(gpu_only)} declare a GPU and cannot be imported without "
                "one, so only build collects them whole, and their imports and "
                "declarations ran here: "
                + "; ".join(f"{path} ({error})" for path, error in sorted(gpu_only.items()))
            )
        if collection_errors or unaccounted:
            text += (
                f"; {len(collection_errors) + len(unaccounted)} could not be "
                "collected, so build could not run the release set"
            )
        print("REVIEWED_TESTS_SUMMARY " + text + ".")
        return 1 if collection_errors or unaccounted else 0
    where = (
        "check ran here, with no GPU and no network,"
        if reviewed_set == "offline"
        else "build ran in the image it made, on the GPU and with the network,"
    )
    text = (
        f"Reviewed tests: of the {reviewed} reviewed test files, {where} "
        f"{len(ran)}: {counts['passed']} tests passed, {counts['skipped']} skipped "
        f"by their own condition and {counts['xfailed']} failed as their xfail "
        "marks state"
    )
    bad = counts["failed"] + counts["error"] + counts["xpassed"]
    if bad:
        text += f"; {bad} failed, errored or passed against a strict xfail"
    if reviewed_set == "offline":
        partly = set(declared_tests) - set(declared_modules)
        tests = sum(declared_tests[path] for path in partly)
        text += (
            f"; {len(declared_modules) + len(partly)} declare that they need a GPU "
            f"or the network -- {len(declared_modules)} as whole modules, and "
            f"{tests} tests in {len(partly)} more -- and only build runs those"
        )
    support = reports[0]["support"]
    text += (
        f"; {len(imported)} of the {len(support)} reviewed support modules ran "
        "through the tests that import them."
    )
    print("REVIEWED_TESTS_SUMMARY " + text)
    return 1 if unaccounted or collection_errors or bad else 0


if __name__ == "__main__":
    if sys.argv[1:2] != ["merge"] or len(sys.argv) < 3:
        raise SystemExit("usage: reviewed_tests_plugin.py merge LOG...")
    raise SystemExit(_merge(sys.argv[2:]))
