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
    """One call carrying *value* in the transport's canonical framing.

    ``<parameter=NAME>`` ``\n`` VALUE ``\n`` ``</parameter>`` is what the
    served template renders and what the model is trained to emit; the
    parser removes exactly that framing, so *value* survives unchanged.
    """
    return (
        "<tool_call>\n<function=write>\n<parameter=text>\n" + value
        + "\n</parameter>\n</function>\n</tool_call>"
    )


def parse(text, chunk_size, *, tools=None, choice="auto", ids=None,
          finish="stop", stop=None):
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
        reasoning, content, calls = parser.parse_output(
            text, request, enable_auto_tools=True, model_output_token_ids=ids,
            finish_reason=finish, stop_reason=stop,
        )
        return (
            reasoning or "", content or "",
            [(c.name, c.arguments) for c in calls or []], parser.tool_calls_complete,
        )
    reasoning, content, calls = "", "", {}
    for start in range(0, len(ids), chunk_size):
        group = ids[start:start + chunk_size]
        last = start + chunk_size >= len(ids)
        delta = parser.parse_output_delta(
            decode(group), group, request, prompt_token_ids=[1, 2, 3],
            finish_reason=finish if last else None, stop_reason=stop if last else None,
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
    def test_native_byte_positions_survive_stops_and_unicode(self):
        from types import SimpleNamespace
        from tokenizers import Tokenizer, decoders, models
        from vllm.parser.engine.token_id_scanner import (
            PreLexedTerminal, TokenIDScanner,
        )

        vocabulary = {chr(i): i for i in range(33, 127)}
        vocabulary.update({"Ġ": 32, "é": 200, "Ľ": 201, "ª": 202,
                           "</think>": 300, "<tool_call>": 301})
        native = Tokenizer(models.BPE(vocabulary, []))
        native.decoder = decoders.ByteLevel()
        before = "Example: </think> literal "
        ids = [200, 201, 202, 32] + [ord(c) for c in before] + [300]
        decoded = native.decode(ids, skip_special_tokens=False)
        for chunk_size in (1, 3, 1000):
            for holdback in (0, 9, 1000):
                for cut in (0, 3, 8):
                    with self.subTest(chunk=chunk_size, holdback=holdback, cut=cut):
                        scanner = TokenIDScanner({300: "THINK_END", 301: "TOOL_START"},
                            SimpleNamespace(backend_tokenizer=native))
                        sent, items = 0, []
                        end = len("雪 " + before) + cut
                        for start in range(0, len(ids), chunk_size):
                            group = ids[start:start + chunk_size]
                            done = start + chunk_size >= len(ids)
                            available = native.decode(ids[:start + len(group)],
                                skip_special_tokens=False).rstrip("\ufffd")
                            visible = min(end, max(0, len(available)
                                          - (0 if done else holdback)))
                            items.extend(scanner.scan(available[sent:visible], group))
                            sent = visible
                        items.extend(scanner.flush_pending())
                        self.assertEqual("".join(item.text for item in items), decoded[:end])
                        self.assertEqual([item.terminal for item in items
                                          if isinstance(item, PreLexedTerminal)],
                                         ["THINK_END"] if cut == 8 else [])

    def test_xml_values_follow_the_complete_schema(self):
        from jsonschema import Draft202012Validator
        from xgrammar import Grammar
        from xgrammar.testing import _is_grammar_accept_string
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        for schema, value, expected in (
            ({"$defs": {"count": {"type": "integer"}}, "properties": {
                "text": {"$ref": "#/$defs/count"}}}, "\n42\n", 42),
            ({"properties": {"text": {
                "type": ["string", "null"], "enum": ["null"]}}},
             "\nnull\n", "null"),
            ({"properties": {"text": {"type": "object", "properties": {
                "count": {"type": "string"}}, "required": ["count"],
                "additionalProperties": False}}}, '{"count":"42"}', {"count": "42"}),
        ):
            schema = {"type": "object", "required": ["text"],
                      "additionalProperties": False, **schema}
            tool = {"type": "function", "function": {
                "name": "write", "parameters": schema,
            }}
            tools = ChatCompletionRequest(messages=[], tools=[tool]).tools
            grammar = Grammar.from_structural_tag(get_model_structural_tag(
                "qwen_3_coder", tools, "auto", False,
            ))
            self.assertTrue(_is_grammar_accept_string(grammar, call(value)))
            for chunk in (None, 1, 13):
                with self.subTest(value=value, chunk=chunk):
                    result = parse("plan</think>" + call(value), chunk, tools=[tool])
                    self.assertEqual(json.loads(result[2][0][1]), {"text": expected})
                    self.assertTrue(Draft202012Validator(schema).is_valid(
                        json.loads(result[2][0][1])
                    ))
                    self.assertTrue(result[3])

    def test_unfinished_numeric_parameter_stays_a_raw_diagnostic(self):
        tool = {"type": "function", "function": {
            "name": "write", "parameters": {
                "type": "object", "properties": {"text": {"type": "integer"}},
            },
        }}
        for chunk in (None, 1, 13):
            with self.subTest(chunk=chunk):
                result = parse("plan</think><tool_call>\n<function=write>\n"
                               "<parameter=text> 123", chunk, tools=[tool],
                               finish="length")
                self.assertEqual(json.loads(result[2][0][1]), {"text": " 123"})
                self.assertFalse(result[3])

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

    def test_values_keep_every_marker_except_the_two_parameter_markers(self):
        """Only the two markers the grammar excludes are structural.

        A value cannot carry ``</parameter>`` (its own closer) or
        ``<parameter=`` (the next parameter's opener, whose absorption is what
        published one call as another). Everything else -- including a bare
        ``<parameter`` with no ``=`` -- is the value's own text.
        """
        for value in (
            "before</function>after", "before</tool_call></think>after",
            "before<parameter literal>after", "before< /parameter >after",
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

    def test_tool_grammar_recognizes_both_tokenizations_after_reasoning(self):
        import xgrammar as xgr
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        vocabulary = [chr(i) for i in range(128)] + list(MARKERS) + ["<eos>"]
        info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[132])
        tools = ChatCompletionRequest(messages=[], tools=[TOOL]).tools
        tag = get_model_structural_tag("qwen_3_coder", tools, "auto", False)
        grammar = xgr.GrammarCompiler(info, max_threads=1).compile_structural_tag(tag)
        grammar_ids = {token_id: 128 + index
                       for index, token_id in enumerate(MARKERS.values())}
        body = call("literal syntax")
        text = "plan</think>" + body
        for body_ids in (encode(body), [ord(c) for c in body]):
            matcher = xgr.GrammarMatcher(grammar)
            for token_id in body_ids:
                self.assertTrue(matcher.accept_token(grammar_ids.get(token_id, token_id)))
            self.assertTrue(matcher.accept_token(132))
            self.assertTrue(matcher.is_terminated())
            ids = encode("plan</think>") + body_ids
            for chunk in (1, 13, None):
                with self.subTest(chunk=chunk, body_ids=body_ids):
                    result = parse(text, chunk, ids=ids)
                    self.assertEqual(result[:2], ("plan", ""))
                    self.assertEqual(result[2], [("write", '{"text": "literal syntax"}')])
                    self.assertTrue(result[3])

    def test_ordinary_tool_and_think_spellings_leave_reasoning_inactive(self):
        reasoning = "plan </think> " + call("literal syntax")
        for chunk in (1, 13, None):
            with self.subTest(chunk=chunk):
                result = parse(reasoning, chunk, ids=[ord(c) for c in reasoning])
                self.assertEqual(result[:3], (reasoning, "", []))

    def test_only_an_observed_wrapper_closes_the_call(self):
        body = call("complete value").removesuffix("</tool_call>")
        for chunk in (1, 13, None):
            with self.subTest(chunk=chunk):
                result = parse("plan</think>" + body, chunk)
                self.assertEqual(result[:3], ("plan", body, []))
                self.assertFalse(result[3])

    def test_caller_stop_and_length_keep_their_distinct_meaning(self):
        body = call("complete value")
        for chunk in (1, 13, None):
            for stop in ("HALT", 42):
                with self.subTest(chunk=chunk, stop=stop):
                    result = parse("plan</think>" + body, chunk, stop=stop)
                    self.assertEqual(result[:3], ("plan", body, []))
            result = parse("plan</think>" + body[:-5], chunk, finish="length")
            self.assertEqual(len(result[2]), 1)
            self.assertFalse(result[3])

    def test_unknown_and_padded_names_are_not_deleted_or_renamed(self):
        for name in ("unknown", " write "):
            body = call("kept").replace("function=write", "function=" + name)
            for chunk in (1, 13, None):
                with self.subTest(name=name, chunk=chunk):
                    result = parse("plan</think>" + body, chunk)
                    self.assertEqual(result[2], [(name, '{"text": "kept"}')])

    def test_native_grammar_suspends_text_stops_inside_values(self):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
        from vllm.v1.structured_output.stop_checker import StructuralTagStopChecker

        tools = ChatCompletionRequest(messages=[], tools=[TOOL]).tools
        tag = get_model_structural_tag("qwen_3_coder", tools, "auto", False)
        params = SamplingParams(stop=["HALT"], structured_outputs=StructuredOutputsParams(
            structural_tag=tag.model_dump_json()))
        request = EngineCoreRequest(
            request_id="unit", prompt_token_ids=[], mm_features=None,
            sampling_params=params, pooling_params=None, arrival_time=0,
            lora_request=None, cache_salt=None, data_parallel_rank=None,
        )

        class TextDetokenizer(BaseIncrementalDetokenizer):
            def decode_next(self, token):
                return chr(token)

        body = call("HALT </tool_call> HALT")
        detok = TextDetokenizer(request)
        detok.stop_checker = StructuralTagStopChecker(MagicMock(), request, None)
        self.assertEqual(detok.update(list(map(ord, body + " after HALT")), False), "HALT")
        self.assertEqual(detok.output_text, body + " after ")

    def test_every_model_eos_is_an_eos_terminal(self):
        from vllm import SamplingParams
        from vllm.v1.core.sched.utils import check_stop
        from vllm.v1.request import Request

        for token in (248046, 248044):
            params = SamplingParams(stop_token_ids=[7], extra_args={"kv_scope": "unit"})
            params.update_from_generation_config({"eos_token_id": [248046, 248044]}, 248046)
            request = Request("unit", [1], params, None)
            request.append_output_token_ids(token)
            self.assertTrue(check_stop(request, 1024))
            self.assertIsNone(request.stop_reason)
            self.assertEqual(params.stop_token_ids, [7])


if __name__ == "__main__":
    unittest.main()
