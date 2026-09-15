#!/usr/bin/env python3
"""CPU checks of the installed Qwen parser's wire-visible language."""

import json
import unittest
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager


MARKERS = {
    "<think>": 20000,
    "</think>": 20001,
    "<tool_call>": 20002,
    "</tool_call>": 20003,
}
TEXT = {v: k for k, v in MARKERS.items()}
TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}
PARSER = ParserManager.get_parser(
    tool_parser_name="qwen3_coder", reasoning_parser_name="qwen3",
    enable_auto_tools=True,
)


def encode(text):
    ids = []
    while text:
        marker = next((m for m in MARKERS if text.startswith(m)), None)
        if marker:
            ids.append(MARKERS[marker])
            text = text[len(marker):]
        else:
            ids.append(ord(text[0]))
            text = text[1:]
    return ids


def decode(ids):
    return "".join(TEXT.get(i, chr(i)) for i in ids)


def call(value):
    return (
        "<tool_call>\n<function=write>\n<parameter=text>" + value
        + "</parameter>\n</function>\n</tool_call>"
    )


def parse(text, chunk_size, *, tools=None, choice="auto", ids=None):
    request = ChatCompletionRequest(
        model="unit", messages=[{"role": "user", "content": "test"}],
        tools=[TOOL] if tools is None else tools or None,
        tool_choice=choice if tools != [] else None,
    )
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = MARKERS
    tokenizer.decode.side_effect = decode
    tokenizer.all_special_tokens = list(MARKERS)
    tokenizer.all_special_ids = list(MARKERS.values())
    parser = PARSER(
        tokenizer, request.tools, chat_template_kwargs={"enable_thinking": True}
    )
    ids = encode(text) if ids is None else ids
    if chunk_size is None:
        reasoning, content, calls = parser.parse(
            text, request, enable_auto_tools=True, model_output_token_ids=ids,
        )
        return (
            reasoning or "", content or "",
            [(c.name, c.arguments) for c in calls or []], parser.tool_calls_complete,
        )
    reasoning, content, calls = "", "", {}
    for start in range(0, len(ids), chunk_size):
        group = ids[start:start + chunk_size]
        delta = parser.parse_delta(
            decode(group), group, request, prompt_token_ids=[1, 2, 3],
            finished=start + chunk_size >= len(ids),
        )
        if not delta:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for c in delta.tool_calls:
            name, args = calls.get(c.index, ("", ""))
            if c.function:
                name += c.function.name or ""
                args += c.function.arguments or ""
            calls[c.index] = name, args
    return reasoning, content, list(calls.values()), parser.tool_calls_complete


class ToolOutputParserTest(unittest.TestCase):
    def test_call_limit_is_decided_by_the_grammar(self):
        from xgrammar import Grammar
        from xgrammar.testing import _is_grammar_accept_string

        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        tools = ChatCompletionRequest(
            messages=[], tools=[TOOL], tool_choice="auto"
        ).tools
        for choice in ("auto", "required"):
            for parallel in (None, True, False):
                with self.subTest(choice=choice, parallel=parallel):
                    tag = get_model_structural_tag(
                        "qwen_3_coder", tools, choice, False,
                        parallel_tool_calls=parallel,
                    )
                    grammar = Grammar.from_structural_tag(tag)
                    self.assertTrue(_is_grammar_accept_string(grammar, call("one")))
                    self.assertEqual(
                        _is_grammar_accept_string(
                            grammar, call("one") + "\n" + call("two")
                        ), parallel is not False,
                    )

    def test_unarmed_text_never_becomes_a_call(self):
        for body in (
            "<function=write><parameter=text>x</parameter></function>",
            "<tool_call> prose </tool_call>", "<tool_call>",
            "<tool_call>\n<function=wr", "<tool_call>\n\n<function=write>",
            "<tool_call><function=write>",
        ):
            for chunk in (1, 3, 13, None):
                with self.subTest(body=body, chunk=chunk):
                    self.assertEqual(parse("plan</think>" + body, chunk)[:3],
                                     ("plan", body, []))

    def test_values_keep_every_marker_except_the_parameter_closer(self):
        for value in (
            "before</function>after", "before</tool_call></think>after",
            "before<parameter=literal>after", "before< /parameter >after",
            "before</parameter >after", "before</tool_call><tool_call></think>after",
        ):
            for chunk in (1, 3, 13, None):
                with self.subTest(value=value, chunk=chunk):
                    result = parse("plan</think>before" + call(value) + "after", chunk)
                    self.assertEqual(result[:2], ("plan", "beforeafter"))
                    self.assertEqual(len(result[2]), 1)
                    self.assertEqual(result[2][0][0], "write")
                    self.assertEqual(json.loads(result[2][0][1]), {"text": value})
                    self.assertTrue(result[3])

    def test_parameter_history_round_trip_preserves_string_bytes(self):
        from chat_template_retention_unit import load_template
        from xgrammar import Grammar
        from xgrammar.testing import _is_grammar_accept_string
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        template = load_template()
        tools = ChatCompletionRequest(messages=[], tools=[TOOL]).tools
        grammar = Grammar.from_structural_tag(get_model_structural_tag(
            "qwen_3_coder", tools, "auto", False))
        for value in ("", "\n", "\n\n", "\nfirst\n", "first\n", "\nfirst",
                      " \t\r\nfirst\n\t ", "雪\nשלום\n",
                      "\n</function></tool_call><think>\n", '\n"quoted"\n'):
            messages = [{"role": "user", "content": "test"}, {
                "role": "assistant", "reasoning_content": "plan", "content": "",
                "tool_calls": [{"type": "function", "function": {
                    "name": "write", "arguments": {"text": value}}}],
            }]
            rendered = template.render(messages=messages, tools=[TOOL],
                                       add_generation_prompt=False)
            expected = call(value)
            self.assertIn(expected + "<|im_end|>", rendered)
            self.assertTrue(_is_grammar_accept_string(grammar, expected))
            for chunk in (1, 3, 13, None):
                with self.subTest(value=value, chunk=chunk):
                    result = parse("plan\n</think>\n before \t\n" + expected
                                   + "\n after \t\n", chunk)
                    self.assertEqual(result[:2],
                                     ("plan\n", "\n before \t\n\n after \t\n"))
                    self.assertEqual(json.loads(result[2][0][1]), {"text": value})

    def test_disabled_tools_keep_the_model_output(self):
        body = call("one</function></tool_call></think>two")
        for settings in ({"tools": []}, {"choice": "none"}):
            for chunk in (1, 3, 13, None):
                with self.subTest(settings=settings, chunk=chunk):
                    self.assertEqual(parse("plan</think>" + body, chunk, **settings)[:3],
                                     ("plan", body, []))

    def test_batch_distinguishes_marker_text_from_marker_ids(self):
        body = call("literal syntax")
        text = "plan</think>" + body
        ids = encode("plan</think>") + [ord(c) for c in body]
        for chunk in (1, 13, None):
            with self.subTest(chunk=chunk):
                self.assertEqual(parse(text, chunk, ids=ids)[:3], ("plan", body, []))

    def test_only_an_observed_wrapper_closes_the_call(self):
        body = call("complete value").removesuffix("</tool_call>")
        for chunk in (1, 13, None):
            with self.subTest(chunk=chunk):
                result = parse("plan</think>" + body, chunk)
                self.assertEqual(len(result[2]), 1)
                self.assertFalse(result[3])


if __name__ == "__main__":
    unittest.main()
