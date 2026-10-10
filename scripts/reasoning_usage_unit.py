#!/usr/bin/env python3
"""CPU-only unit checks for the served exact reasoning token count."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from unittest.mock import MagicMock

import vllm.parser as parser_package

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import (
    _make_completion_tokens_details,
)
from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokenUsageInfo,
    UsageInfo,
)
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import DelegatingParser, reasoning_token_usage
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    TokenTerminal,
    Transition,
)
from vllm.parser.inkling import InklingParser
from vllm.parser.kimi_k2 import KimiK2Parser
from vllm.parser.qwen3 import (
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Qwen3Parser,
)
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

# The deployed grammar's markers under the ids the mock tokenizer resolves;
# every other token is one ASCII character carrying its code point.
VOCAB = {
    THINK_START: 998,
    THINK_END: 999,
    TOOL_CALL_START: 1000,
    TOOL_CALL_END: 1001,
    "<|im_end|>": 1002,
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
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    request.tools = [ChatCompletionToolsParam.model_validate({
        "type": "function", "function": {"name": "f", "parameters": {"type": "object"}},
    })]
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


def stream(parser, request, deltas, prompt_token_ids=(1, 2, 3)):
    outputs = []
    for index, pieces in enumerate(deltas):
        text, ids = tokens(*pieces)
        outputs.append(
            parser.parse_delta(
                delta_text=text,
                delta_token_ids=ids,
                request=request,
                prompt_token_ids=list(prompt_token_ids),
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
assert reasoning_token_usage(explicit, 6) == 3

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
assert batched.count_reasoning_tokens(ids) == 3

# The usage field the endpoint builds: summed across choices, absent rather
# than zero when a choice has no exact count, and refused when the parser
# did not see the ids the choice generated.
assert _make_completion_tokens_details([3, 2]) == CompletionTokenUsageInfo(
    reasoning_tokens=5
)
assert _make_completion_tokens_details([3, None]) is None
assert reasoning_token_usage(None, 6) is None
try:
    reasoning_token_usage(explicit, 7)
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
            token_id_terminals={
                "THINK_START": TokenTerminal(THINK_START),
                "THINK_END": TokenTerminal(THINK_END),
            },
            transitions=transitions,
        ),
    )
    assert reason in engine._reasoning_boundary_refusal, reason
    assert engine.reasoning_token_count is None
    assert engine.count_reasoning_tokens([65, 99, 66]) is None


# One count: the whole-generation count a reasoning parser exposes is the
# count its feed took on the same ids -- exact, or absent exactly where the
# served usage is absent, never an error -- for every format built on the
# engine, and taken from where the prompt leaves the grammar.
class TerminalVocab(dict):
    """Every terminal a format asks for is one id: only its grammar decides."""

    def get(self, text, default=None):
        if text not in self:
            self[text] = 5000 + len(self)
        return self[text]

    def decode(self, ids):
        by_id = {token_id: text for text, token_id in self.items()}
        return "".join(by_id.get(token_id, chr(token_id)) for token_id in ids)


formats = []
for module_info in pkgutil.iter_modules(parser_package.__path__):
    module = importlib.import_module(
        f"{parser_package.__name__}.{module_info.name}"
    )
    formats += [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, ParserEngine)
        and obj is not ParserEngine
        and obj.__module__ == module.__name__
    ]
assert {Qwen3Parser, KimiK2Parser, InklingParser} <= set(formats), formats
for parser_cls in formats:
    for enable_thinking in (True, False):
        vocab = TerminalVocab()
        terminal_tokenizer = MagicMock()
        terminal_tokenizer.get_vocab.return_value = vocab
        terminal_tokenizer.decode.side_effect = vocab.decode
        terminal_tokenizer.all_special_tokens = []
        terminal_tokenizer.all_special_ids = []
        engine = parser_cls(
            terminal_tokenizer,
            [],
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        boundary = sorted(engine._reasoning_boundary_ids)
        for ids in ([65, 66, 67], [65, 66, *boundary[:1], 67]):
            with engine.batch_token_ids(ids):
                engine.extract_reasoning(vocab.decode(ids), request)
            assert engine.count_reasoning_tokens(ids) == engine.reasoning_token_count, (
                parser_cls.__name__,
                enable_thinking,
                ids,
            )


# A parser that splits on text serves no count, and its own count is that
# same absent one rather than a depth count usage declines.
text_vocab = {**VOCAB, "<mm:think>": 1003, "</mm:think>": 1004}
text_ids = {token_id: text for text, token_id in text_vocab.items()}
for reasoning_parser_cls, opener, closer in (
    (DeepSeekR1ReasoningParser, THINK_START, THINK_END),
    (MiniMaxM3ReasoningParser, "<mm:think>", "</mm:think>"),
):
    text_tokenizer = MagicMock()
    text_tokenizer.get_vocab.return_value = dict(text_vocab)
    text_tokenizer.decode.side_effect = lambda ids: "".join(
        text_ids.get(i, chr(i)) for i in ids
    )
    text_tokenizer.encode.side_effect = lambda text, **_: (
        [text_vocab[text]] if text in text_vocab else [ord(c) for c in text]
    )
    text_split = type(
        "TextSplit",
        (DelegatingParser,),
        {"reasoning_parser_cls": reasoning_parser_cls, "tool_parser_cls": None},
    )(text_tokenizer)
    ids = [text_vocab[opener], ord("A"), ord("B"), text_vocab[closer], ord("C")]
    split = text_split.parse(
        text_tokenizer.decode(ids), request, model_output_token_ids=ids
    )
    assert split[:2] == ("AB", "C"), (reasoning_parser_cls.__name__, split)
    assert text_split.reasoning_token_count is None
    assert text_split.reasoning_parser.count_reasoning_tokens(ids) is None, (
        reasoning_parser_cls.__name__
    )


class PromptOpenedGrammar(ParserEngine):
    """Configured to start in content; a prompt ending in the opener starts
    the generation inside reasoning, as a template that pre-fills it does."""

    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, parser_engine_config=ParserEngineConfig(
            name="unit-prompt-opened",
            initial_state=ParserState.CONTENT,
            terminals={"THINK_START": THINK_START, "THINK_END": THINK_END},
            token_id_terminals={
                "THINK_START": TokenTerminal(THINK_START),
                "THINK_END": TokenTerminal(THINK_END),
            },
            transitions={
                (ParserState.REASONING, "THINK_END"): Transition(
                    ParserState.CONTENT, (EventType.REASONING_END,)
                ),
            },
        ))

    def adjust_initial_state_from_prompt(self, prompt_token_ids):
        if prompt_token_ids and prompt_token_ids[-1] == self.vocab[THINK_START]:
            self._start_in(ParserState.REASONING)


prompt_opened = PromptOpenedGrammar(tokenizer)
stream(prompt_opened, request, [("A", "B"), (THINK_END, "C")],
       prompt_token_ids=(1, VOCAB[THINK_START]))
_, ids = tokens("A", "B", THINK_END, "C")
assert prompt_opened.reasoning_token_count == 2
assert prompt_opened.count_reasoning_tokens(ids) == 2


# Responses reports the same count chat does, read from the parser that split
# the output: exact for the deployed grammar, and absent -- never a failed
# response -- for a grammar that re-enters reasoning, as gemma4, deepseek_v4,
# glm45/glm47 and ling3 do.
def responses_usage(parser, text, ids):
    import asyncio

    from vllm.entrypoints.openai.responses.context import SimpleContext
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.sampling_params import SamplingParams

    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.enable_auto_tools = True
    request = ResponsesRequest(input="q", kv_scope="unit", store=False)
    context = SimpleContext(response_parser=parser)
    completion = CompletionOutput(
        index=0, text=text, token_ids=ids, cumulative_logprob=None, logprobs=None,
        finish_reason="stop", stop_reason=None,
    )
    context.append_output(RequestOutput(
        request_id="unit", prompt="q", prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None, outputs=[completion], finished=True,
    ))
    output = serving._collect_response_output(request, context, tokenizer)
    response = asyncio.run(serving._finalize_response(
        request, SamplingParams(max_tokens=16), context, "unit", 1, output,
    ))
    return response.usage.output_tokens_details.reasoning_tokens


text, ids = tokens("A", "B", "C", THINK_END, "D", "E")
assert responses_usage(make_parser(tokenizer), text, ids) == 3


class ReenteringGrammar(ParserEngine):
    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, parser_engine_config=ParserEngineConfig(
            name="unit-reentering",
            initial_state=ParserState.REASONING,
            terminals={"THINK_START": THINK_START, "THINK_END": THINK_END},
            token_id_terminals={
                "THINK_START": TokenTerminal(THINK_START),
                "THINK_END": TokenTerminal(THINK_END),
            },
            transitions={
                (ParserState.REASONING, "THINK_END"): Transition(
                    ParserState.CONTENT, (EventType.REASONING_END,)
                ),
                (ParserState.CONTENT, "THINK_START"): Transition(
                    ParserState.REASONING, (EventType.REASONING_START,)
                ),
            },
        ))


reentering_reasoning, _ = make_adapters(ReenteringGrammar)


class Reentering(DelegatingParser):
    reasoning_parser_cls = reentering_reasoning
    tool_parser_cls = None


assert responses_usage(Reentering(tokenizer), text, ids) is None

print("reasoning-usage-unit: PASS")
