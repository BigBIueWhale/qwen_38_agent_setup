#!/usr/bin/env python3
"""Opaque agent IDs for a probe run: one per conversation.

Generation requires a ``kv_scope`` agent ID naming one line of work, which is
one conversation. Every probe follows the same discipline:

- Mint one ID per conversation with ``new_conversation`` and send it on every
  request of that conversation: each continuation of its history, and each
  retry or redraw of one of its turns.
- Mint a new ID for anything that is not that conversation: a fork of its
  history, a subagent, a fresh control, or an independent request.
- A line of work has at most one request in flight; the server refuses a
  second request under an ID whose earlier request has not finished. Requests
  that run concurrently are different lines of work with distinct IDs.
- A request generates one sequence (``n`` is 1, a Completions request carries
  one prompt), so several samples or prompts are several requests, each with
  its own new ID. The batch route takes one distinct ID per conversation, as a
  list in the order of its ``messages``.

The ID groups what the CPU KV tier retains. Prefix lookup matches content and
cache_salt, whatever the ID, so a fork under a new ID still reuses the resident
prefix it shares with its parent.

An ID is the probe's file stem, a fresh run ID, a process-wide sequence number
and the conversation's label, so no call returns an ID any earlier call, in
this run or another, returned. This convention is local to the probe suite; the
backend interprets no ID structure or lineage. ``scripts/run-probe.sh`` stages
the suite into the serving container and runs the named probe by file.
"""

from __future__ import annotations

import __main__
import itertools
import re
import threading
import uuid
from pathlib import Path

_script = Path(getattr(__main__, "__file__", ""))
if not _script.is_file() or not _script.name.endswith("_probe.py"):
    raise SystemExit(
        "a probe names its agents after the probe file it runs from; launch it "
        "through scripts/run-probe.sh rather than from stdin or -c"
    )

_RUN = f"{_script.stem}/{uuid.uuid4().hex}"
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_sequence = itertools.count(1)
_sequence_lock = threading.Lock()


def new_conversation(label: str) -> str:
    """Return a new, never-used agent ID for one conversation.

    ``label`` says which conversation this is for a reader of the server's
    logs; it must be a non-empty run of letters, digits, ``.``, ``_`` or
    ``-`` that starts with a letter or digit. Uniqueness never rests on it:
    the sequence number makes each call's ID new even when labels repeat.
    """
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise ValueError(f"conversation label {label!r} is not a plain token")
    with _sequence_lock:
        number = next(_sequence)
    return f"{_RUN}/{number}-{label}"
