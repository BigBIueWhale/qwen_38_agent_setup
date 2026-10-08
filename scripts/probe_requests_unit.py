#!/usr/bin/env python3
"""Every probe builds, offline, each kind of request it sends, and its route takes it.

A probe runs only against the live backend, so a probe that cannot build its
requests is noticed only when someone runs it on the card -- and until then it
reads as coverage. vision_history_cache_probe built a render-only Anthropic
request without its agent ID, which the protocol refuses, and crashed before
any generation on every run from 2026-10-01 on. This unit is where such a
probe fails instead: in the check, against the reconstruction's own protocol.

Each ``scripts/*_probe.py`` exposes ``offline_requests()``. It builds, with no
server, one request through every body builder the probe sends through --
the same functions its live run calls, so the two cannot differ -- running
any conversion a body needs on the way (an Anthropic request converted for the
render route, say). A body that a live run derives from an earlier response
takes a value of the shape that response has. It returns
``(route, body, refusal)`` triples: ``body`` is ``None`` for a GET, and
``refusal`` is ``None`` for a request the route must take, or the text the
probe holds the server's refusal of it to.

The unit loads each probe as the program, as ``scripts/run-probe.sh`` runs it,
so its agent IDs are its own. The routes and the request model each parses
come from the API server's own app, built offline, never from a list kept
here. Every POST body is validated as JSON by its route's request model: one
the probe sends to be served must be taken, and one it sends to be refused
must be taken or refused naming what the probe holds the refusal to (a route
that refuses it later, while serving, takes it here). Every route the probe
names must be among the routes its offline requests reach.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
import typing
from argparse import Namespace
from pathlib import Path

from fastapi.routing import APIRoute
from pydantic import TypeAdapter

from vllm.entrypoints.openai.api_server import build_app

PROBES = Path(__file__).resolve().parent


def served_routes() -> tuple[dict[str, TypeAdapter], set[str]]:
    """The POST routes with their request models, and the GET routes, of the
    app the server builds for a generating model."""
    args = Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        enable_fault_tolerance=False,
        root_path=None,
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[],
    )
    app = build_app(args, supported_tasks=("generate",))
    posts: dict[str, TypeAdapter] = {}
    gets: set[str] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if "GET" in route.methods:
            gets.add(route.path)
        if "POST" in route.methods:
            model = typing.get_type_hints(route.endpoint).get("request")
            if model is not None:
                posts[route.path] = TypeAdapter(model)
    return posts, gets


def load_as_program(path: Path) -> types.ModuleType:
    """Import a probe as the program a run executes, so probe_scope names its
    agents after it."""
    for name in [n for n in sys.modules if n == "probe_scope" or n.endswith("_probe")]:
        del sys.modules[name]
    program = types.ModuleType("__main__")
    program.__file__ = str(path)
    saved = sys.modules["__main__"]
    sys.modules["__main__"] = program
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    finally:
        sys.modules["__main__"] = saved
    return module


def named_routes(path: Path, routes: set[str]) -> set[str]:
    """Every served route the probe's source names, in any string."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in routes
    }


def check_probe(
    path: Path, posts: dict[str, TypeAdapter], gets: set[str]
) -> list[str]:
    module = load_as_program(path)
    build = getattr(module, "offline_requests", None)
    if not callable(build):
        return [f"{path.name} exposes no offline_requests()"]
    try:
        requests = build()
    except Exception as error:
        return [
            f"{path.name}: building its requests offline raised "
            f"{type(error).__name__}: {error}"
        ]
    if not isinstance(requests, list) or not requests:
        return [f"{path.name}: offline_requests() returned no requests"]
    failures: list[str] = []
    reached: set[str] = set()
    refused = 0
    for index, request in enumerate(requests):
        label = f"{path.name} request {index}"
        if not (isinstance(request, tuple) and len(request) == 3):
            failures.append(f"{label} is not a (route, body, refusal) triple")
            continue
        route, body, refusal = request
        reached.add(route)
        label += f" ({route})"
        if body is None:
            if route not in gets:
                failures.append(f"{label}: the app serves no GET {route}")
            continue
        if route not in posts:
            failures.append(f"{label}: the app serves no POST {route}")
            continue
        try:
            # The body crosses the wire as JSON, and the route validates
            # what it decodes.
            posts[route].validate_python(json.loads(json.dumps(body, ensure_ascii=False)))
        except Exception as error:
            refused += 1
            text = f"{type(error).__name__}: {error}"
            if refusal is None:
                failures.append(f"{label}: the route refuses it: {text}")
            elif refusal.lower() not in text.lower():
                failures.append(
                    f"{label}: the route refuses it, but not for the reason the "
                    f"probe holds it to ({refusal!r}): {text}"
                )
    unreached = sorted(named_routes(path, posts.keys() | gets) - reached)
    if unreached:
        failures.append(
            f"{path.name} names routes its offline requests never reach: {unreached}"
        )
    if not failures:
        print(
            f"{path.name}: {len(requests)} requests ({refused} refused as the "
            f"probe holds them to be), routes {sorted(reached)}"
        )
    return failures


def main() -> None:
    sys.path.insert(0, str(PROBES))
    posts, gets = served_routes()
    probes = sorted(PROBES.glob("*_probe.py"))
    if not probes:
        raise SystemExit(f"no probes found in {PROBES}")
    failures: list[str] = []
    for path in probes:
        failures.extend(check_probe(path, posts, gets))
    if failures:
        raise SystemExit(
            "A probe cannot build the requests it sends:\n  "
            + "\n  ".join(failures)
            + "\nNext: make the probe build each request its route takes, "
            "through the builder its live run sends through."
        )
    print(f"probe requests unit: {len(probes)} probes passed")


if __name__ == "__main__":
    main()
