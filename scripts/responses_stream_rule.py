"""The one rule a Responses stream is held to: it is its own snapshot.

Both checks of it read this module -- the parser unit, offline in check and in
the image beside it, and ``responses_protocol_probe.py`` against the live
server -- so the rule cannot be narrowed in one of them and stay strict in the
other. Events are the JSON objects the server streams, one per ``data:`` line.
"""

from __future__ import annotations

DELTAS = {
    "response.output_text.delta": "text",
    "response.reasoning_text.delta": "text",
    "response.function_call_arguments.delta": "arguments",
}


def item_text(item: dict) -> str:
    if item.get("type") == "function_call":
        return item.get("arguments") or ""
    return "".join(part.get("text", "") for part in item.get("content") or [])


# Every done event that states an item's text, and where it states it.
STATED = {
    "response.output_text.done": lambda event: event["text"],
    "response.reasoning_text.done": lambda event: event["text"],
    "response.content_part.done": lambda event: event["part"].get("text"),
    "response.reasoning_part.done": lambda event: event["part"].get("text"),
    "response.function_call_arguments.done": lambda event: event["arguments"],
    "response.output_item.done": lambda event: item_text(event["item"]),
}


def assert_stream_is_its_snapshot(events: list[dict], completed: dict) -> None:
    """One generation, one record: every item of the completed response is the
    concatenation of the deltas streamed under its index, and no other index
    streams any; every done event that states an item's text states exactly
    what its deltas built; the completed response is the done items in order,
    with contiguous sequence numbers."""
    numbers = [event["sequence_number"] for event in events]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise AssertionError(f"Responses stream sequence numbers are not contiguous: {numbers}")
    built: dict[int, str] = {}
    done: dict[int, dict] = {}
    for event in events:
        kind = event.get("type")
        if kind in DELTAS:
            index = event["output_index"]
            built[index] = built.get(index, "") + event["delta"]
        elif kind in STATED:
            index = event["output_index"]
            if STATED[kind](event) != built.get(index):
                raise AssertionError(
                    f"Responses {kind} of item {index} states "
                    f"{STATED[kind](event)!r}; its deltas built {built.get(index)!r}"
                )
            if kind == "response.output_item.done":
                done[index] = event["item"]
    snapshot = dict(enumerate(completed["output"]))
    texts = {index: item_text(item) for index, item in snapshot.items()}
    if texts != built:
        raise AssertionError(
            f"Responses completed items {texts!r} are not what their deltas "
            f"built {built!r}"
        )
    if done != snapshot:
        raise AssertionError(
            "Responses completed output differs from the streamed done items: "
            f"{completed['output']!r} != {[done[i] for i in sorted(done)]!r}"
        )
