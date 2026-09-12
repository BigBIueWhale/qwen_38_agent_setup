#!/usr/bin/env python3
"""Prove the served template cannot be made to discard historical reasoning.

`./scripts/build-vllm.sh check` proves the template's bytes are exactly what the
derivation produces from the model's own template. That is a statement about
bytes, not about what they do: a future edit could move a stage and its pinned
hash together and reconstruction would still succeed. This renders the installed
template and asserts the property itself.

The property: every assistant turn already in the conversation is rendered with
its reasoning, whatever the request asks, and a request that asks for anything
else is refused rather than served a shorter prompt. It is enforced in the
served artifact rather than in a launch argument, so it holds for every caller of
this model and not only for callers who share this repository's configuration.
"""

from __future__ import annotations

import re

from jinja2 import BaseLoader, Environment
from jinja2.ext import loopcontrols


TEMPLATE_PATH = "/opt/qwen38/chat_template.jinja"
REASONING_MARKER = re.compile(r"HISTORICAL_REASONING_(\d+)")
EMPTY_THINKING = "<think>\n\n</think>"


class TemplateRefusal(Exception):
    pass


def load_template(path: str = TEMPLATE_PATH):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    environment = Environment(loader=BaseLoader(), extensions=[loopcontrols])

    def raise_exception(message: str) -> None:
        raise TemplateRefusal(message)

    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(source)


def agent_history(turns: int, reminder_after: tuple[int, ...] = ()) -> list[dict]:
    """One task prompt, then assistant/tool turns, with reminders injected.

    The reminders matter: an injected `role: "user"` message is what used to move
    the template's retention cut, so a rule that survives them is a rule the
    client cannot influence.
    """
    messages: list[dict] = [{"role": "user", "content": "Do the task."}]
    for index in range(1, turns + 1):
        messages.append(
            {
                "role": "assistant",
                "content": f"step {index}",
                "reasoning_content": f"HISTORICAL_REASONING_{index}",
            }
        )
        messages.append({"role": "tool", "content": f"result {index}"})
        if index in reminder_after:
            messages.append(
                {
                    "role": "user",
                    "content": "<system-reminder>active todo</system-reminder>",
                }
            )
    return messages


def rendered_reasoning(template, messages: list[dict], **kwargs) -> list[int]:
    prompt = template.render(
        messages=messages, add_generation_prompt=True, **kwargs
    )
    assert EMPTY_THINKING not in prompt, (
        "a rendered turn carries an empty thinking block, which tells the model "
        "that turn thought nothing rather than that its thinking is not shown"
    )
    return sorted(int(match) for match in REASONING_MARKER.findall(prompt))


def main() -> None:
    template = load_template()
    messages = agent_history(6, reminder_after=(2, 4))
    expected = list(range(1, 7))

    # Omitted and the one accepted value render every historical turn's
    # reasoning, across injected reminders.
    for label, kwargs in (("omitted", {}), ("true", {"preserve_thinking": True})):
        observed = rendered_reasoning(template, messages, **kwargs)
        assert observed == expected, (label, observed, expected)

    # Anything else is refused. `0`, `"false"` and `None` are included because a
    # client that means `false` can spell it in several ways, and a template that
    # accepted any of them would discard reasoning while appearing to refuse.
    for asked in (False, 0, "false", None, "no"):
        try:
            template.render(
                messages=messages,
                add_generation_prompt=True,
                preserve_thinking=asked,
            )
        except TemplateRefusal as refusal:
            assert "cannot be discarded" in str(refusal), str(refusal)
        else:
            raise AssertionError(
                f"preserve_thinking={asked!r} was accepted; the served template "
                "must refuse every request to discard historical reasoning"
            )

    # A conversation with no assistant turns yet renders and refuses the same.
    opening = [{"role": "user", "content": "Do the task."}]
    assert rendered_reasoning(template, opening) == []

    print("CHAT_TEMPLATE_RETENTION_UNIT_OK turns=6 reminders=2 refusals=5")


if __name__ == "__main__":
    main()
