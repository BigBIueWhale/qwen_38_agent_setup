#!/usr/bin/env python3
"""The agent a probe run is.

Every generative request names the agent that owns its KV cache, and the
backend refuses one that does not: the host KV tier files each block under
its owner and gives up whole agents under pressure, so a request naming no
agent would produce blocks no single context could ever reclaim. The harness
names a main agent by its session id and a subagent by the tool call that
spawned it -- one id per agent, held for that agent's life.

A probe is a caller in exactly that position, and its agent is one run of
the probe. The identity is minted once per process from two real facts:
which probe is running -- the file the interpreter was given -- and a fresh
run id, so two runs of the same probe are two agents, as two sessions of the
harness are. There is no default and no fallback: a probe fed to the
interpreter from stdin or as an inline command has no name here, and a
made-up one would be exactly the anonymous label the requirement exists to
refuse. ``scripts/run-probe.sh`` stages the suite into the serving container
and runs the named probe by file.
"""

from __future__ import annotations

import __main__
import uuid
from pathlib import Path

_script = Path(getattr(__main__, "__file__", ""))
if not _script.is_file() or not _script.name.endswith("_probe.py"):
    raise SystemExit(
        "a probe names its agent after the probe file it runs from; launch it "
        "through scripts/run-probe.sh rather than from stdin or -c"
    )

KV_SCOPE = f"{_script.stem}/{uuid.uuid4().hex}"
