#!/usr/bin/env python3
"""CPU-only unit checks for the served exact reasoning token count."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import (
    _make_completion_tokens_details,
    _reasoning_token_count,
)
from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokenUsageInfo,
    UsageInfo,
)
from vllm.parser import ParserManager
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.qwen3 import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Qwen3Parser,
)

# The deployed grammar's markers under the ids the mock tokenizer resolves;
# every other token is one ASCII character carrying its code point.
VOCAB = {
    THINK_START: 98,
    THINK_END: 99,
    TOOL_CALL_START: 100,
    TOOL_CALL_END: 101,
    "<|im_end|>": 102,
}
ID_TO_TEXT = {token_id: text for text, token_id in VOCAB.items()}


def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = dict(VOCAB)
    tokenizer.decode.side_effect = lambda ids: "".join(
        ID_TO_TEXT.get(i, chr(i)) for i in ids
    )
    tokenizer.all_special_tokens = list(VOCAB)
    tokenizer.all_special_ids = list(VOCAB.values())
    return tokenizer


def token_id(piece: str) -> int:
    return VOCAB[piece] if piece in VOCAB else ord(piece)


def tokens(*pieces: str) -> tuple[str, list[int]]:
    return "".join(pieces), [token_id(piece) for piece in pieces]


def make_request() -> MagicMock:
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    request.include_reasoning = True
    return request


def make_parser(tokenizer):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="qwen3_coder",
        reasoning_parser_name="qwen3",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    return parser_cls(tokenizer, [], chat_template_kwargs={"enable_thinking": True})


def stream(parser, request, deltas):
    outputs = []
    for index, pieces in enumerate(deltas):
        text, ids = tokens(*pieces)
        outputs.append(
            parser.parse_delta(
                delta_text=text,
                delta_token_ids=ids,
                request=request,
                prompt_token_ids=[1, 2, 3],
                finished=index == len(deltas) - 1,
            )
        )
    return outputs


def joined(outputs, field: str) -> str:
    return "".join(getattr(delta, field) or "" for delta in outputs if delta)


tokenizer = make_tokenizer()
request = make_request()

# Streaming: the count is the position of the boundary id, whether the
# boundary is the explicit </think> or Qwen's implicit <tool_call>, and it is
# taken from the same split the client receives.
explicit = make_parser(tokenizer)
outputs = stream(explicit, request, [("A", "B"), ("C", THINK_END, "D"), ("E",)])
assert joined(outputs, "reasoning") == "ABC", outputs
assert joined(outputs, "content") == "DE", outputs
assert explicit.reasoning_token_count == 3
assert explicit.generated_token_count == 6
assert _reasoning_token_count(explicit, 6) == 3

call = (
    TOOL_CALL_START,
    *"\n<function=f>\n<parameter=x>1</parameter>\n</function>\n",
    TOOL_CALL_END,
)
implicit = make_parser(tokenizer)
outputs = stream(implicit, request, [("A", "B"), ("C",), call])
assert joined(outputs, "reasoning") == "ABC", outputs
assert any(delta and delta.tool_calls for delta in outputs), outputs
assert implicit.reasoning_token_count == 3
assert implicit.generated_token_count == 3 + len(call)

# A generation that never leaves reasoning is all reasoning, special tokens
# generated inside it included: the ids are what completion_tokens counts.
unfinished = make_parser(tokenizer)
stream(unfinished, request, [("A", THINK_START, "B"), ("C", "<|im_end|>")])
assert unfinished.reasoning_token_count == 5
assert unfinished.generated_token_count == 5

# Batch: the split is made on the ids, exactly as it is delta by delta.
batched = make_parser(tokenizer)
text, ids = tokens("A", "B", "C", THINK_END, "D", "E")
reasoning, content, tool_calls = batched.parse(
    text, request, enable_auto_tools=True, model_output_token_ids=ids
)
assert (reasoning, content, tool_calls) == ("ABC", "DE", None), (
    reasoning,
    content,
    tool_calls,
)
assert batched.reasoning_token_count == 3
assert batched.generated_token_count == len(ids)
assert batched.reasoning_parser.count_reasoning_tokens(ids) == 3

# The usage field the endpoint builds: summed across choices, absent rather
# than zero when a choice has no exact count, and refused when the parser
# did not see the ids the choice generated.
assert _make_completion_tokens_details([3, 2]) == CompletionTokenUsageInfo(
    reasoning_tokens=5
)
assert _make_completion_tokens_details([3, None]) is None
assert _reasoning_token_count(None, 6) is None
try:
    _reasoning_token_count(explicit, 7)
except ValueError as exc:
    assert "handed 6 generated ids" in str(exc), str(exc)
else:
    raise AssertionError("a parser that missed generated ids was not refused")

usage = UsageInfo(
    prompt_tokens=10,
    completion_tokens=6,
    total_tokens=16,
    completion_tokens_details=_make_completion_tokens_details([3]),
)
served = json.loads(usage.model_dump_json(exclude_unset=True, exclude_none=True))
assert served["completion_tokens_details"] == {"reasoning_tokens": 3}, served
without = json.loads(UsageInfo(prompt_tokens=1).model_dump_json(exclude_none=True))
assert "completion_tokens_details" not in without, without

# Refusals: text fed without its ids has no position in the id stream, and a
# grammar without one id-marked, mode-independent boundary serves no count.
text_only = Qwen3Parser(tokenizer)
text_only.extract_reasoning("AB" + THINK_END + "C", request)
try:
    text_only.reasoning_token_count
except ValueError as exc:
    assert "without its token ids" in str(exc), str(exc)
else:
    raise AssertionError("a text-only feed produced a reasoning token count")

disabled = Qwen3Parser(tokenizer, chat_template_kwargs={"enable_thinking": False})
stream(disabled, request, [("A", "B")])
assert disabled.reasoning_token_count == 0
assert disabled.count_reasoning_tokens([65, 66]) == 0

for transitions, reason in (
    (
        {
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT, (EventType.REASONING_END,)
            ),
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING, (EventType.REASONING_START,)
            ),
        },
        "re-enters reasoning",
    ),
    (
        {(ParserState.REASONING, "THINK_END"): Transition(ParserState.CONTENT, ())},
        "without announcing",
    ),
):
    engine = ParserEngine(
        tokenizer,
        parser_engine_config=ParserEngineConfig(
            name="unit-grammar",
            initial_state=ParserState.REASONING,
            terminals={"THINK_START": THINK_START, "THINK_END": THINK_END},
            token_id_terminals={"THINK_START": THINK_START, "THINK_END": THINK_END},
            transitions=transitions,
        ),
    )
    assert engine.reasoning_token_count is None
    try:
        engine.count_reasoning_tokens([65, 99, 66])
    except ValueError as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError(f"an unmodelled grammar was counted: {reason}")

print("reasoning-usage-unit: PASS")
