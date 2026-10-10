#!/usr/bin/env python3
"""The parser serving builds, for a probe that parses locally.

A probe that checks a policy by parsing a fixed output must parse it the way the
served route does: with the parser ``ParserManager`` builds for the reasoning
parser and the tool-call parser the launch selects -- for this launch, the one
Qwen engine both adapt -- and the launch's default template arguments, so what
the probe holds is the route's parser, not one built beside it.
``scripts/run-probe.sh`` names what the launch gives ``--reasoning-parser``,
``--tool-call-parser`` and ``--default-chat-template-kwargs`` in
``SERVED_REASONING_PARSER``, ``SERVED_TOOL_CALL_PARSER`` and
``SERVED_CHAT_TEMPLATE_KWARGS``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.parser import ParserManager


def _launch_value(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise SystemExit(
            f"{name} is not set. Run the probe through ./scripts/run-probe.sh, "
            "which names the parsers and template arguments the launch serves."
        ) from None


def served_parser(tokenizer: Any, tools: Any) -> Any:
    """A fresh instance of the parser the served route builds for one output."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name=_launch_value("SERVED_TOOL_CALL_PARSER"),
        reasoning_parser_name=_launch_value("SERVED_REASONING_PARSER"),
        enable_auto_tools=True,
    )
    return parser_cls(
        tokenizer,
        tools,
        chat_template_kwargs=json.loads(_launch_value("SERVED_CHAT_TEMPLATE_KWARGS")),
    )
