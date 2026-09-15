#!/usr/bin/env python3
"""One opaque agent identity for a complete probe run.

Generation requires a stable agent ID for shared-prefix matching and cache
accounting. An ID with no cached blocks may acquire an initial shared prefix;
an existing ID matches its own acquired cache and extends it.

Each probe run is one agent. The probe's file name and a fresh run ID identify
it for the life of the process. This convention is local to the probe suite;
the backend interprets no ID structure or lineage. ``scripts/run-probe.sh``
stages the suite into the serving container and runs the named probe by file.
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
