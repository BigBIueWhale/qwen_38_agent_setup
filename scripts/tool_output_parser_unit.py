#!/usr/bin/env python3
"""CPU checks of the installed Qwen parser's wire-visible language, run as
serving runs it: the parser composition the launch selects, on the served
tokenizer, fed the deltas its native decoder produces.

The build mounts the served model's tokenizer and generation files, each checked
against the model manifest, at ``SERVED_MODEL``, and names what the launch gives
``--reasoning-parser``, ``--tool-call-parser`` and
``--default-chat-template-kwargs`` in ``SERVED_REASONING_PARSER``,
``SERVED_TOOL_CALL_PARSER`` and ``SERVED_CHAT_TEMPLATE_KWARGS``. Nothing here has
a second tokenizer or a second composition to fall back to: run any other way,
the unit fails at import, naming what it lacks.
"""

import json
import os
import unittest
from unittest.mock import MagicMock

from tokenizers import Tokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import NativeDecodeStream


def _served(name):
    try:
        return os.environ[name]
    except KeyError:
        raise SystemExit(
            f"{name} is not set. This unit runs the parser the launch serves on the "
            "served tokenizer; run it through ./scripts/build-vllm.sh check."
        ) from None


SERVED_MODEL = _served("SERVED_MODEL")
TOKENIZER = get_tokenizer(SERVED_MODEL)
with open(os.path.join(SERVED_MODEL, "generation_config.json")) as config:
    MODEL_EOS = tuple(json.load(config)["eos_token_id"])
PARSER = ParserManager.get_parser(
    tool_parser_name=_served("SERVED_TOOL_CALL_PARSER"),
    reasoning_parser_name=_served("SERVED_REASONING_PARSER"),
    enable_auto_tools=True,
)
CHAT_TEMPLATE_KWARGS = json.loads(_served("SERVED_CHAT_TEMPLATE_KWARGS"))
MARKERS = {
    marker: TOKENIZER.convert_tokens_to_ids(marker)
    for marker in ("<think>", "</think>", "<tool_call>", "</tool_call>")
}
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


def _ordinary_tokenizer():
    """The served pipeline without its added vocabulary: what spells a marker
    with ordinary tokens, as request text does once the template has rendered."""
    spec = json.loads(TOKENIZER.backend_tokenizer.to_str())
    spec["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(spec))


_ORDINARY = _ordinary_tokenizer()


def encode(text):
    """The ids the model generates for *text*: added-token spellings are ids."""
    return TOKENIZER.encode(text, add_special_tokens=False)


def ordinary(text):
    """The ids that spell *text* without any added or special token."""
    return _ORDINARY.encode(text, add_special_tokens=False).ids


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


def request_for(*, tools=None, choice="auto"):
    fields = {} if tools == [] else {"tool_choice": choice}
    return ChatCompletionRequest(
        model="unit", messages=[{"role": "user", "content": "test"}],
        tools=[TOOL] if tools is None else tools or None, **fields,
    )


def generation(text, ids, finish, stop):
    """The ids serving hands the parser and the text of each: a model EOS ends
    the ids and the detokenizer gives it no text."""
    ids = encode(text) if ids is None else list(ids)
    if finish == "stop" and stop is None:
        ids.append(MODEL_EOS[0])
    decoder = NativeDecodeStream(TOKENIZER.backend_tokenizer)
    texts = [decoder.step(token) or "" for token in ids]
    if finish == "stop" and stop is None:
        texts[-1] = ""
    return ids, texts


def parse(text, chunk_size, *, tools=None, choice="auto", ids=None,
          finish="stop", stop=None):
    request = request_for(tools=tools, choice=choice)
    parser = PARSER(TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)
    ids, texts = generation(text, ids, finish, stop)
    if chunk_size is None:
        reasoning, content, calls = parser.parse_output(
            "".join(texts), request, enable_auto_tools=True,
            model_output_token_ids=ids, finish_reason=finish, stop_reason=stop,
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
            "".join(texts[start:start + chunk_size]), group, request,
            prompt_token_ids=[1, 2, 3],
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


def grammar_matcher(request):
    """The decoding grammar the request arms, as the engine compiles it: on
    the served vocabulary, with the model's EOS ids as its terminators."""
    import xgrammar as xgr

    armed = PARSER(
        TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS
    ).adjust_request(request)
    info = xgr.TokenizerInfo.from_huggingface(TOKENIZER, stop_token_ids=list(MODEL_EOS))
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_structural_tag(
        armed.structured_outputs.structural_tag
    )
    return xgr.GrammarMatcher(compiled)


class ToolOutputParserTest(unittest.TestCase):
    def test_content_survives_tools_without_whitespace_normalization(self):
        for choice in ('auto', 'required', {
            'type': 'function', 'function': {'name': 'write'},
        }):
            for chunk in (None, 1, 13):
                with self.subTest(choice=choice, chunk=chunk):
                    text = ('plan</think> \t' + call('value') + '\n\n'
                            + call('value') + '\r\n')
                    reasoning, content, calls, complete = parse(
                        text, chunk, choice=choice,
                    )
                    self.assertEqual(reasoning, 'plan')
                    self.assertEqual(content, ' \t\n\n\r\n')
                    self.assertEqual([json.loads(args) for _, args in calls],
                                     [{'text': 'value'}, {'text': 'value'}])
                    self.assertTrue(complete)
                    self.assertEqual(parse('plan</think> \t\r\n', chunk,
                                           choice=choice)[1], ' \t\r\n')

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

    def test_json_values_may_carry_the_parameter_markers(self):
        """A JSON value ends at the first closer outside its strings.

        The grammar admits any JSON string in an array or object parameter,
        including one spelling ``</parameter>``, ``<parameter=NAME>``,
        ``</function>`` or ``</tool_call>``. Each call below is accepted by
        the served grammar and must be published exactly as written -- not
        cut at the marker, refused as a repeated parameter, or demoted to
        content -- in the batch and every streamed chunking. A declared
        parameter whose slot is still open is read by its declared
        production, so the declared reading wins over the undeclared one
        that would split the same text into a repeat.
        """
        from xgrammar import Grammar
        from xgrammar.testing import _is_grammar_accept_string
        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        todo = {"type": "function", "function": {"name": "todo_write", "parameters": {
            "type": "object", "required": ["todos"], "additionalProperties": False,
            "properties": {"todos": {"type": "array", "items": {
                "type": "object", "required": ["content", "status"],
                "properties": {"content": {"type": "string"},
                               "status": {"type": "string",
                                          "enum": ["pending", "completed"]}}}}}}}}
        listing = {"type": "function", "function": {"name": "list_directory",
            "parameters": {"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string"},
                "ignore": {"type": "array", "items": {"type": "string"}}}}}}
        for tool, arguments in (
            (todo, {"todos": [{"content": "close with </parameter> tag",
                               "status": "pending"}]}),
            (todo, {"todos": [{"content": "a</parameter>\n<parameter=todos>\nb",
                               "status": "pending"}]}),
            (todo, {"todos": [{"content": "a</parameter> then </function> b",
                               "status": "completed"}]}),
            (todo, {"todos": [{"content": "</parameter></function></tool_call>",
                               "status": "pending"}]}),
            (listing, {"path": "/a",
                       "ignore": ["x</parameter>\n<parameter=path>\ny"]}),
            (listing, {"path": "/a", "ignore": ["*.log</parameter>"]}),
        ):
            name = tool["function"]["name"]
            body = ("<tool_call>\n<function=" + name + ">\n" + "".join(
                "<parameter=" + key + ">\n"
                + (value if isinstance(value, str) else json.dumps(value))
                + "\n</parameter>\n" for key, value in arguments.items()
            ) + "</function>\n</tool_call>")
            tools = ChatCompletionRequest(messages=[], tools=[tool]).tools
            grammar = Grammar.from_structural_tag(get_model_structural_tag(
                "qwen_3_coder", tools, "auto", False))
            self.assertTrue(_is_grammar_accept_string(grammar, body), body)
            for chunk in (None, 1, 3, 13):
                with self.subTest(arguments=arguments, chunk=chunk):
                    result = parse("plan</think>" + body, chunk, tools=[tool])
                    self.assertEqual(result[:2], ("plan", ""))
                    self.assertEqual([(n, json.loads(a)) for n, a in result[2]],
                                     [(name, arguments)])
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
        body = call("literal syntax")
        text = "plan</think>" + body
        for body_ids in (encode(body), ordinary(body)):
            matcher = grammar_matcher(request_for())
            for token_id in body_ids:
                self.assertTrue(matcher.accept_token(token_id))
            self.assertTrue(matcher.accept_token(MODEL_EOS[0]))
            self.assertTrue(matcher.is_terminated())
            ids = encode("plan</think>") + body_ids
            for chunk in (1, 13, None):
                with self.subTest(chunk=chunk, ordinary=body_ids != encode(body)):
                    result = parse(text, chunk, ids=ids)
                    self.assertEqual(result[:2], ("plan", ""))
                    self.assertEqual(result[2], [("write", '{"text": "literal syntax"}')])
                    self.assertTrue(result[3])

    def test_ordinary_tool_and_think_spellings_leave_reasoning_inactive(self):
        reasoning = "plan </think> " + call("literal syntax")
        for chunk in (1, 13, None):
            with self.subTest(chunk=chunk):
                result = parse(reasoning, chunk, ids=ordinary(reasoning))
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

    def test_a_repeated_parameter_is_refused_naming_the_call(self):
        from vllm.entrypoints.openai.engine.protocol import (
            RepeatedToolParameterError,
        )
        from vllm.entrypoints.serve.exception_handling.error_response import (
            create_error_response,
        )

        parameters = dict(TOOL["function"]["parameters"], additionalProperties=True)
        tool = dict(TOOL, function=dict(TOOL["function"], parameters=parameters))
        first = "<tool_call>\n<function=write>\n<parameter=text>\nfirst\n</parameter>\n"
        for second, finish in (
            ("<parameter=text>\nsecond\n</parameter>\n</function>\n</tool_call>",
             "stop"),
            ("<parameter=text>\nsec", "length"),
        ):
            for chunk in (1, 13, None):
                with self.subTest(finish=finish, chunk=chunk):
                    with self.assertRaises(RepeatedToolParameterError) as refused:
                        parse("plan</think>" + first + second, chunk,
                              tools=[tool], finish=finish)
                    self.assertEqual(
                        (refused.exception.tool,
                         refused.exception.repeated_parameter),
                        ("write", "text"),
                    )
                    error = create_error_response(refused.exception).error
                    self.assertEqual(
                        (error.code, error.type, error.param),
                        (422, "RepeatedToolParameterError", None),
                    )

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
            structural_tag=tag.model_dump_json()), extra_args={"kv_scope": "unit"})
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

        for token in MODEL_EOS:
            params = SamplingParams(stop_token_ids=[7], extra_args={"kv_scope": "unit"})
            params.update_from_generation_config(
                {"eos_token_id": list(MODEL_EOS)}, MODEL_EOS[0]
            )
            request = Request("unit", [1], params, None)
            request.append_output_token_ids(token)
            self.assertTrue(check_stop(request, 1024))
            self.assertIsNone(request.stop_reason)
            self.assertEqual(params.stop_token_ids, [7])


    def test_every_generated_token_survives_parsing(self):
        """The parser deletes nothing the model generated.

        Every special token of the served tokenizer that the format does not
        act on -- the vision and audio markers among them -- is the text it
        decodes to, in reasoning, in the answer, beside a call and inside a
        value; so is a second ``</think>``. Batch and every chunking read the
        same ids the same way, so text and ids cannot disagree. A model EOS
        ends a generation, so it is not one of them.
        """
        specials = [
            token for token in TOKENIZER.all_special_tokens
            if token not in MARKERS
            and TOKENIZER.convert_tokens_to_ids(token) not in MODEL_EOS
        ]
        self.assertIn("<|image_pad|>", specials)
        for token in [*specials, "</think>"]:
            cases = [
                ("pl" + token + "an</think>\n\nanswer",
                 ("pl" + token + "an", "\n\nanswer", [])),
                ("plan</think>\n\nA" + token + "B",
                 ("plan", "\n\nA" + token + "B", [])),
                ("plan</think>\n\nSee " + token + ".\n" + call("v"),
                 ("plan", "\n\nSee " + token + ".\n",
                  [("write", json.dumps({"text": "v"}))])),
                ("plan</think>\n\n" + call("a" + token + "b"),
                 ("plan", "\n\n",
                  [("write", json.dumps({"text": "a" + token + "b"},
                                        ensure_ascii=False))])),
            ]
            if token == "</think>":
                # Inside reasoning a closer ends it; that is not this token.
                cases = cases[1:]
            for text, expected in cases:
                for chunk in (None, 1, 3, 1000):
                    with self.subTest(token=token, text=text, chunk=chunk):
                        self.assertEqual(parse(text, chunk)[:3], expected)

    def test_the_grammar_is_armed_exactly_where_calls_are_parsed(self):
        """One value decides both: an omitted or null choice is "auto" with
        tools, and "none" is the only choice under which nothing is a call."""
        omitted = object()
        text = "plan</think>\n\n" + call("one")
        for choice in (omitted, None, "auto", "required",
                       {"type": "function", "function": {"name": "write"}}, "none"):
            fields = {} if choice is omitted else {"tool_choice": choice}
            request = ChatCompletionRequest(
                model="unit", messages=[{"role": "user", "content": "test"}],
                tools=[TOOL], **fields,
            )
            armed = PARSER(
                TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS
            ).adjust_request(request)
            is_armed = (
                armed.structured_outputs is not None
                and armed.structured_outputs.structural_tag is not None
            )
            self.assertEqual(is_armed, armed.tool_choice != "none")
            for chunk in (None, 1, 13):
                with self.subTest(choice=choice, chunk=chunk):
                    result = parse(text, chunk, choice=(
                        armed.tool_choice if isinstance(armed.tool_choice, str)
                        else armed.tool_choice.model_dump()
                    ))
                    self.assertEqual(bool(result[2]), is_armed)
                    if not is_armed:
                        self.assertEqual(result[:2], ("plan", "\n\n" + call("one")))

    def test_a_call_only_answer_follows_the_template_blank_line(self):
        """A forced or required call is generable in the shape the template
        renders and the model is trained on: ``</think>``, a blank line, the
        call. The grammar starts after the reasoning closer, so the blank line
        is its first token; nothing else may precede the call."""
        from chat_template_retention_unit import load_template

        rendered = load_template().render(messages=[
            {"role": "user", "content": "test"},
            {"role": "assistant", "reasoning_content": "plan", "content": "",
             "tool_calls": [{"type": "function", "function": {
                 "name": "write", "arguments": {"text": "one"}}}]},
        ], tools=[TOOL], add_generation_prompt=False)
        self.assertIn("plan\n</think>\n\n" + call("one") + "<|im_end|>", rendered)
        for choice in ("required", {"type": "function", "function": {"name": "write"}}):
            for prefix, accepted in (("\n\n", True), ("", True), ("\n", False),
                                     ("Calling.\n\n", False)):
                with self.subTest(choice=choice, prefix=prefix):
                    matcher = grammar_matcher(request_for(choice=choice))
                    walked = all(
                        matcher.accept_token(token)
                        for token in encode(prefix + call("one"))
                    )
                    self.assertEqual(
                        walked and matcher.accept_token(MODEL_EOS[0]), accepted
                    )
            for chunk in (None, 1, 13):
                with self.subTest(choice=choice, chunk=chunk):
                    self.assertEqual(
                        parse("plan</think>\n\n" + call("one"), chunk, choice=choice)[:3],
                        ("plan", "\n\n", [("write", '{"text": "one"}')]),
                    )


if __name__ == "__main__":
    unittest.main()
