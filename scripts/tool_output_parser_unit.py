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

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

from tokenizers import Tokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import NativeDecodeStream


def asyncio_run(awaitable):
    return asyncio.run(awaitable)


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
    GENERATION_CONFIG = json.load(config)
MODEL_EOS = tuple(GENERATION_CONFIG["eos_token_id"])
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


def rendered_prompt(messages, *, continue_final=False):
    """The ids of the prompt the served template renders for *messages*.

    ``continue_final`` leaves the final assistant message open, as
    ``continue_final_message`` does: the render ends right after that
    message's text rather than at the end of turn the template writes.
    """
    from chat_template_retention_unit import load_template

    text = load_template().render(
        messages=messages, add_generation_prompt=not continue_final,
        **CHAT_TEMPLATE_KWARGS,
    )
    if continue_final:
        final = messages[-1]["content"]
        text = text[:text.rindex(final) + len(final)]
    return encode(text)


# The generation prompt of a fresh turn: reasoning is open where it ends.
OPEN_PROMPT = rendered_prompt([{"role": "user", "content": "test"}])


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
          finish="stop", stop=None, request=None, prompt=OPEN_PROMPT):
    if request is None:
        request = request_for(tools=tools, choice=choice)
    parser = PARSER(TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)
    ids, texts = generation(text, ids, finish, stop)
    if chunk_size is None:
        reasoning, content, calls = parser.parse_output(
            "".join(texts), request, prompt_token_ids=prompt, enable_auto_tools=True,
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
            prompt_token_ids=prompt,
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


def _served_model_config():
    """The model config the chat route reads, from the served files."""
    from dataclasses import dataclass, field

    from transformers import AutoConfig

    from vllm.config.multimodal import MultiModalConfig

    @dataclass
    class ServedModelConfig:
        task = "generate"
        runner_type = "generate"
        model = SERVED_MODEL
        tokenizer = SERVED_MODEL
        trust_remote_code = False
        tokenizer_mode = "auto"
        max_model_len = 4096
        tokenizer_revision = None
        multimodal_config = MultiModalConfig()
        hf_config = hf_text_config = AutoConfig.from_pretrained(
            SERVED_MODEL, local_files_only=True
        )
        logits_processors = None
        diff_sampling_param = None
        allowed_local_media_path = ""
        allowed_media_domains = None
        encoder_config = None
        generation_config = "auto"
        override_generation_config: dict = field(default_factory=dict)
        media_io_kwargs: dict = field(default_factory=dict)
        skip_tokenizer_init = False
        is_encoder_decoder = False
        is_multimodal_model = False
        renderer_num_workers = 1
        enable_prompt_embeds = False

        def get_diff_sampling_param(self):
            return {}

    return ServedModelConfig()


def _served_chat():
    """The chat route as the launch serves it -- template, parsers, renderer
    -- over a mocked engine, which is returned with it."""
    from types import SimpleNamespace

    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.online_renderer import OnlineRenderer
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.v1.engine.async_llm import AsyncLLM

    from chat_template_retention_unit import TEMPLATE_PATH

    with open(TEMPLATE_PATH) as handle:
        template = handle.read()
    config = _served_model_config()
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = config
    engine.input_processor = MagicMock()
    engine.renderer = HfRenderer(
        SimpleNamespace(model_config=config,
                        parallel_config=SimpleNamespace(_api_process_rank=0)),
        cached_tokenizer_from_config(config),
    )
    launch = dict(
        enable_auto_tools=True,
        tool_parser=_served("SERVED_TOOL_CALL_PARSER"),
        reasoning_parser=_served("SERVED_REASONING_PARSER"),
        default_chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    )
    serving = OpenAIServingChat(
        engine,
        OpenAIServingModels(engine_client=engine, base_model_paths=[
            BaseModelPath(name="unit", model_path=SERVED_MODEL)]),
        response_role="assistant",
        online_renderer=OnlineRenderer(
            model_config=config, renderer=engine.renderer, request_logger=None,
            chat_template=template, chat_template_content_format="auto", **launch,
        ),
        chat_template=template,
        chat_template_content_format="auto",
        request_logger=None,
        **launch,
    )
    return serving, engine


def _chat_route_admits(*, include_reasoning, tools):
    """Everything the chat route hands the engine for one request: the
    prompt, the sampling parameters and the admission arguments."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionGenerationRequest,
    )

    serving, engine = _served_chat()
    request = ChatCompletionGenerationRequest(
        model="unit", messages=[{"role": "user", "content": "what is 1+1?"}],
        kv_scope="unit", include_reasoning=include_reasoning, max_tokens=8,
        **({"tools": tools} if tools else {}),
    )
    try:
        asyncio_run(serving.create_chat_completion(request))
    except Exception:
        pass  # the mocked engine generates nothing; admission has happened
    if engine.admit.call_args is None:
        raise AssertionError("the chat route admitted nothing")
    engine_input, params = engine.admit.call_args.args[:2]
    admitted = dict(engine.admit.call_args.kwargs)
    admitted.pop("trace_headers", None)
    admitted["prompt"] = {k: v for k, v in dict(engine_input).items()
                          if k != "arrival_time"}
    admitted["params"] = repr(params)
    return admitted


def _chat_logprobs(served, text, chunk_size, finish="stop", tools=None):
    """The log probabilities the *served* chat route (``_served_chat()``)
    reports for one generation in which the k-th generated id has log
    probability -(k + 1)/100: the full response's, or every streamed chunk's
    in order, as (token, logprob)."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionGenerationRequest,
    )
    from vllm.logprobs import Logprob
    from vllm.outputs import CompletionOutput, RequestOutput

    serving, engine = served
    ids, texts = generation(text, None, finish, None)
    logprobs = [{token: Logprob(logprob=-(k + 1) / 100, rank=1)}
                for k, token in enumerate(ids)]
    step = chunk_size or len(ids)

    async def outputs():
        for start in range(0, len(ids), step):
            last = start + step >= len(ids)
            yield RequestOutput(
                request_id="unit", prompt="p", prompt_token_ids=OPEN_PROMPT,
                prompt_logprobs=None, finished=last, outputs=[CompletionOutput(
                    index=0, text="".join(texts[start:start + step]),
                    token_ids=ids[start:start + step], cumulative_logprob=None,
                    logprobs=logprobs[start:start + step],
                    finish_reason=finish if last else None, stop_reason=None,
                )],
            )

    engine.admit = AsyncMock(return_value=outputs())
    request = ChatCompletionGenerationRequest(
        model="unit", messages=[{"role": "user", "content": "test"}],
        kv_scope="unit", logprobs=True, top_logprobs=0, max_tokens=1000,
        stream=chunk_size is not None, **({"tools": tools} if tools else {}),
    )

    async def collect():
        response = await serving.create_chat_completion(request)
        if chunk_size is None:
            return [(entry.token, entry.logprob)
                    for entry in response.choices[0].logprobs.content]
        reported = []
        async for line in response:
            if not line.startswith("data: {"):
                continue
            for choice in json.loads(line[len("data: "):])["choices"]:
                for entry in (choice.get("logprobs") or {}).get("content") or []:
                    reported.append((entry["token"], entry["logprob"]))
        return reported

    return ids, asyncio_run(collect())


def _responses_events(text, chunk_size, finish="stop"):
    """The served Responses stream for one generation: the launch's parsers,
    the request adjusted as rendering adjusts it, engine outputs of
    *chunk_size* ids decoded as the detokenizer decodes them, after the
    generation prompt of a fresh turn."""
    from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
    from vllm.entrypoints.openai.responses.context import SimpleContext
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
    from vllm.outputs import CompletionOutput, RequestOutput

    engine = MagicMock()
    engine.model_config.max_model_len = 10_000
    engine.model_config.hf_config.model_type = "qwen3"
    engine.model_config.get_diff_sampling_param.return_value = {}
    serving = OpenAIServingResponses(
        engine_client=engine, models=MagicMock(), online_renderer=MagicMock(),
        request_logger=None, chat_template=None, chat_template_content_format="auto",
        reasoning_parser=_served("SERVED_REASONING_PARSER"),
        tool_parser=_served("SERVED_TOOL_CALL_PARSER"), enable_auto_tools=True,
    )
    request = ResponsesRequest.model_validate({
        "model": "unit", "input": "test", "kv_scope": "unit", "stream": True,
        "tools": [{"type": "function", "name": "write",
                   "parameters": TOOL["function"]["parameters"]}],
    })
    serving.parser(
        TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS
    ).adjust_request(request)
    ids, texts = generation(text, None, finish, None)
    context = SimpleContext(response_parser=serving.parser(
        TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS))

    async def outputs():
        for start in range(0, len(ids), chunk_size):
            last = start + chunk_size >= len(ids)
            context.append_output(RequestOutput(
                request_id="unit", prompt=None, prompt_token_ids=OPEN_PROMPT,
                prompt_logprobs=None, finished=last,
                outputs=[CompletionOutput(
                    index=0, text="".join(texts[start:start + chunk_size]),
                    token_ids=ids[start:start + chunk_size], cumulative_logprob=None,
                    logprobs=None, finish_reason=finish if last else None,
                    stop_reason=None,
                )],
            ))
            yield context

    async def collect():
        return [event async for event in serving.responses_stream_generator(
            request, request.to_sampling_params(1000, {}), outputs(), context,
            "unit", TOKENIZER, RequestResponseMetadata(request_id="unit"),
            created_time=1,
        )]

    return asyncio_run(collect())


def _item_text(item):
    if item.type == "function_call":
        return item.arguments
    return "".join(part.text for part in (item.content or []))


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


    def test_derender_reads_a_generation_as_the_chat_route_does(self):
        """Derender hands the parser the text the serving detokenizer gives.

        The stop token a generation ends on -- a model EOS or a caller's stop
        token id -- has no text unless the caller asked to see stop text, and a
        length cut inside a character shows nothing of it; derender decides
        both where the detokenizer does, so the same ids parse to the same
        message on the chat route and on derender, batch and stream. Derender
        reads a stop from its caller, not from the engine, so it accepts one
        exactly where the engine, on the served generation config, would have
        stopped with that stop_reason, and refuses any other, naming the choice.
        """
        import types

        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            GenerateResponse,
            GenerateStreamResponse,
        )
        from vllm.exceptions import VLLMValidationError
        from vllm.renderers.online_derenderer import OnlineDerenderer
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.utils import check_stop
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.engine.detokenizer import IncrementalDetokenizer
        from vllm.v1.request import Request

        derenderer = OnlineDerenderer(
            types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3_5"), model="unit",
                try_get_generation_config=lambda: GENERATION_CONFIG,
            ),
            types.SimpleNamespace(
                get_tokenizer=lambda: TOKENIZER,
                get_eos_token_id=lambda: TOKENIZER.eos_token_id, _executor=None,
            ),
            request_logger=None, chat_template=None,
            chat_template_content_format="openai", enable_auto_tools=True,
            tool_parser=_served("SERVED_TOOL_CALL_PARSER"),
            reasoning_parser=_served("SERVED_REASONING_PARSER"),
            default_chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        )
        crab = encode("plan</think>\n\nCrab: \U0001F980")
        stop_id = encode(" Done")[0]
        cases = [
            ("answer", encode("plan</think>\n\nThe answer is 4.") + [MODEL_EOS[0]],
             "stop", None, False),
            ("call", encode("plan</think>\n\n" + call("one")) + [MODEL_EOS[1]],
             "stop", None, False),
            ("caller stop id", encode("plan</think>\n\nAll") + [stop_id],
             "stop", stop_id, False),
            ("stop text asked for", encode("plan</think>\n\nYes") + [MODEL_EOS[0]],
             "stop", None, True),
            ("cut inside a character", crab[:-1], "length", None, False),
        ]
        for name, ids, finish, stop, include_stop in cases:
            with self.subTest(case=name):
                params = SamplingParams(
                    skip_special_tokens=False,
                    include_stop_str_in_output=include_stop,
                    extra_args={"kv_scope": "unit"},
                )
                detokenizer = IncrementalDetokenizer.from_new_request(
                    TOKENIZER,
                    EngineCoreRequest(
                        request_id="unit", prompt_token_ids=encode("prompt"),
                        mm_features=None, sampling_params=params,
                        pooling_params=None, arrival_time=0.0, lora_request=None,
                        cache_salt=None, data_parallel_rank=None,
                    ),
                )
                # The engine reports a stop on a stop token as STOP.
                detokenizer.update(list(ids), finish == "stop")
                served = detokenizer.get_next_output_text(finished=True, delta=False)
                request = request_for().model_copy(
                    update={"include_stop_str_in_output": include_stop})
                parser = PARSER(
                    TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS
                )
                chat = parser.parse_output(
                    served, request, prompt_token_ids=OPEN_PROMPT,
                    enable_auto_tools=True, model_output_token_ids=list(ids),
                    finish_reason=finish, stop_reason=stop,
                )
                response = GenerateResponse(request_id="unit", choices=[{
                    "index": 0, "finish_reason": finish, "stop_reason": stop,
                    "token_ids": list(ids),
                }])
                message = derenderer._derender_chat(response, request)[0].message
                self.assertEqual(
                    (message.reasoning, message.content,
                     [(c.function.name, c.function.arguments)
                      for c in message.tool_calls]),
                    (chat[0], chat[1], [(c.name, c.arguments) for c in chat[2] or []]),
                )
                completion = CompletionRequest(
                    model="unit", prompt="prompt", skip_special_tokens=False,
                    include_stop_str_in_output=include_stop,
                )
                (choice,), _, _ = derenderer._derender_completion(
                    [response], None, completion)
                self.assertEqual(choice.text, served)
                streamed, state = "", None
                for start in range(0, len(ids), 3):
                    last = start + 3 >= len(ids)
                    chunk = GenerateStreamResponse(request_id="unit", choices=[{
                        "index": 0, "token_ids": list(ids[start:start + 3]),
                        "finish_reason": finish if last else None,
                        "stop_reason": stop if last else None,
                    }])
                    delta, state = asyncio_run(derenderer.derender_completion_stream(
                        "unit", chunk, state, completion_request=completion))
                    streamed += delta.choices[0].text
                self.assertEqual(streamed, served)

        request = request_for()
        completion = CompletionRequest(model="unit", prompt="prompt")
        body = encode("plan</think>\n\nThe answer is 4.")
        for last in (*MODEL_EOS, stop_id, None):
            ids = body + [last] if last is not None else body
            params = SamplingParams(stop_token_ids=[stop_id], extra_args={"kv_scope": "unit"})
            params.update_from_generation_config(GENERATION_CONFIG, TOKENIZER.eos_token_id)
            self.assertEqual(derenderer.eos_token_ids, params.eos_token_ids)
            engine = Request("unit", [1], params, None)
            engine.append_output_token_ids(ids[-1])
            stopped = check_stop(engine, 1024)
            for case_ids, stop in ((ids, None), (ids, stop_id), ([], None), ([], stop_id)):
                accepted = bool(case_ids) and stopped and engine.stop_reason == stop
                choice = {"index": 0, "finish_reason": "stop", "stop_reason": stop,
                          "token_ids": list(case_ids)}
                response = GenerateResponse(request_id="unit", choices=[choice])
                chunk = GenerateStreamResponse(request_id="unit", choices=[choice])
                for field, derender in (
                    ("generate_response.choices[0]",
                     lambda: derenderer._derender_chat(response, request)),
                    ("generate_responses[0].choices[0]",
                     lambda: derenderer._derender_completion([response], None, completion)),
                    ("generate_chunk.choices[0]",
                     lambda: asyncio_run(derenderer.derender_completion_stream(
                         "unit", chunk, None, completion_request=completion))),
                ):
                    with self.subTest(last=last, ids=len(case_ids), stop=stop, field=field):
                        if accepted:
                            derender()
                            continue
                        with self.assertRaises(VLLMValidationError) as refused:
                            derender()
                        self.assertEqual(refused.exception.parameter, field)
                        self.assertEqual(response.choices[0].token_ids, list(case_ids))

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

    def test_a_format_whose_ids_and_text_could_disagree_never_registers(self):
        """The served format gives its batch tool pass the content ids, which is
        sound only while it acts on no content terminal outside its tool
        language. That is a property of the format alone, so registering the
        format decides it, before any request: the served one registers, and
        the same format acting on one more content terminal does not.
        """
        import dataclasses

        from vllm.parser.engine.adapters import make_adapters
        from vllm.parser.engine.parser_engine_config import ParserState, Transition

        served = PARSER.reasoning_parser_cls._parser_engine_cls
        self.assertTrue(served.batch_tool_pass_uses_ids)
        served.check_format()

        class ActsOnAnOpenerInContent(served):
            @classmethod
            def engine_config(cls, thinking):
                config = super().engine_config(thinking)
                transitions = {
                    **config.transitions,
                    (ParserState.CONTENT, "THINK_START"):
                        Transition(next_state=ParserState.CONTENT),
                }
                return dataclasses.replace(config, transitions=transitions)

        with self.assertRaisesRegex(ValueError, r"acts on \['THINK_START'\]"):
            make_adapters(ActsOnAnOpenerInContent)

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

    def test_a_responses_choice_enforces_exactly_the_offered_functions(self):
        """Responses offers a namespace's function under its flat name. The
        prompt, the armed grammar and the parser read that one list: the
        grammar admits a call to exactly the offered names the choice allows,
        both transports parse it alike, and a choice the grammar cannot
        enforce is refused naming the parameter, never a server error."""
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.entrypoints.openai.responses.utils import construct_tool_dicts
        from vllm.exceptions import VLLMValidationError

        schema = TOOL["function"]["parameters"]
        function = {"type": "function", "name": "write", "parameters": schema}
        namespace = {"type": "namespace", "name": "fs", "description": "files",
                     "tools": [{"type": "function", "name": "save",
                                "parameters": schema}]}
        hosted = {"type": "web_search_preview"}

        def request(choice, tools=(function, namespace)):
            return ResponsesRequest.model_validate(
                {"model": "unit", "input": "test", "tools": list(tools),
                 "tool_choice": choice, "kv_scope": "unit"})

        def named(name):
            return call("one").replace("<function=write>", f"<function={name}>")

        def allowed(mode, name):
            return {"type": "allowed_tools", "mode": mode,
                    "tools": [{"type": "function", "name": name}]}

        enforced = (
            ("auto", {"write", "fs__save"}),
            ("required", {"write", "fs__save"}),
            ({"type": "function", "name": "fs__save"}, {"fs__save"}),
            (allowed("auto", "fs__save"), {"fs__save"}),
            (allowed("required", "write"), {"write"}),
        )
        for choice, callable_names in enforced:
            offered = {tool["function"]["name"] for tool in
                       construct_tool_dicts(request(choice).tools, choice)}
            self.assertEqual(offered, {"write", "fs__save"})
            for name in (*offered, "save"):
                with self.subTest(choice=choice, name=name):
                    matcher = grammar_matcher(request(choice))
                    walked = all(matcher.accept_token(token)
                                 for token in encode("\n\n" + named(name)))
                    self.assertEqual(
                        walked and matcher.accept_token(MODEL_EOS[0]),
                        name in callable_names,
                    )
            for name in callable_names:
                for chunk in (None, 1, 13):
                    with self.subTest(choice=choice, name=name, chunk=chunk):
                        self.assertEqual(
                            parse("plan</think>\n\n" + named(name), chunk,
                                  request=request(choice))[2],
                            [(name, '{"text": "one"}')],
                        )
        refused = (
            ({"type": "function", "name": "save"}, (function, namespace), "tool_choice"),
            (hosted, (function, namespace, hosted), "tool_choice"),
            (allowed("auto", "save"), (function, namespace), "tool_choice.tools[0]"),
            ("required", (hosted,), "tool_choice"),
        )
        for choice, tools, parameter in refused:
            with self.subTest(refused=choice, tools=tools):
                with self.assertRaises(VLLMValidationError) as refusal:
                    grammar_matcher(request(choice, tools))
                self.assertEqual(refusal.exception.parameter, parameter)

    def test_a_responses_tool_the_template_is_never_given_is_refused(self):
        """The chat template is given function tools and the functions of a
        namespace, and nothing else, so a declared tool of any other kind, a
        namespace member that is not a function, or a namespace with no tools
        would never reach the model -- with a tool server that serves every
        kind, on either context. The route refuses the request with a 400
        naming that tool and its kind, under every tool choice, before
        anything is rendered or admitted."""
        import typing
        from unittest.mock import AsyncMock, patch

        from openai.types.responses.tool import Tool

        from vllm.entrypoints.mcp.tool_server import ToolServer
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
        from vllm.entrypoints.serve.exception_handling.error_response import (
            create_error_response,
        )
        from vllm.exceptions import VLLMValidationError

        schema = TOOL["function"]["parameters"]
        function = {"type": "function", "name": "write", "parameters": schema}
        never_given = {
            "file_search": {"vector_store_ids": ["unit"]},
            "computer": {},
            "computer_use_preview": {"display_height": 1, "display_width": 1,
                                     "environment": "linux"},
            "web_search": {},
            "web_search_2025_08_26": {},
            "mcp": {"server_label": "code_interpreter", "server_url": "http://unit"},
            "code_interpreter": {"container": {"type": "auto"}},
            "programmatic_tool_calling": {},
            "image_generation": {},
            "local_shell": {},
            "shell": {},
            "custom": {"name": "patch"},
            "tool_search": {},
            "web_search_preview": {},
            "web_search_preview_2025_03_11": {},
            "apply_patch": {},
        }
        declarable = {
            kind for member in typing.get_args(typing.get_args(Tool)[0])
            for kind in typing.get_args(member.model_fields["type"].annotation)
        }
        self.assertEqual(declarable, {"function", "namespace", *never_given})
        namespace = {"type": "namespace", "name": "fs", "description": "files",
                     "tools": [{"type": "function", "name": "save",
                                "parameters": schema},
                               {"type": "custom", "name": "patch"}]}
        empty = dict(namespace, tools=[])
        declared = [
            *((tools, parameter, f"is a {kind!r} tool")
              for kind, fields in never_given.items()
              for tools, parameter in (([{"type": kind, **fields}], "tools[0]"),
                                       ([function, {"type": kind, **fields}],
                                        "tools[1]"))),
            ([function, namespace], "tools[1].tools[1]",
             "is a 'custom' tool in namespace 'fs'"),
            ([empty], "tools[0]", "is namespace 'fs' with no tools"),
            ([function, empty], "tools[1]", "is namespace 'fs' with no tools"),
        ]

        engine = MagicMock()
        engine.errored = False
        engine.model_config.max_model_len = 10_000
        engine.model_config.hf_config.model_type = "qwen3"
        engine.model_config.get_diff_sampling_param.return_value = {}
        tool_server = MagicMock(spec=ToolServer)
        tool_server.has_tool.return_value = True
        renderer = MagicMock()
        renderer.preprocess_chat = AsyncMock(
            side_effect=AssertionError("the chat template was given the request"))
        serving = OpenAIServingResponses(
            engine_client=engine, models=MagicMock(), online_renderer=renderer,
            request_logger=None, chat_template=None,
            chat_template_content_format="auto",
            reasoning_parser=_served("SERVED_REASONING_PARSER"),
            tool_parser=_served("SERVED_TOOL_CALL_PARSER"), enable_auto_tools=True,
            tool_server=tool_server,
        )
        offered = ResponsesRequest.model_validate(
            {"model": "unit", "input": "test", "kv_scope": "unit",
             "tools": [function, dict(namespace, tools=namespace["tools"][:1])]})
        with self.assertRaisesRegex(AssertionError, "chat template was given"):
            asyncio_run(serving.create_responses(offered))
        renderer.preprocess_chat.reset_mock()
        for parsable in ("0", "1"):
            for tools, parameter, cause in declared:
                for choice in ("auto", "required", "none"):
                    with self.subTest(parsable=parsable, tools=tools, choice=choice), \
                            patch.dict(os.environ, {
                                "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": parsable}):
                        request = ResponsesRequest.model_validate(
                            {"model": "unit", "input": "test", "tools": tools,
                             "tool_choice": choice, "kv_scope": "unit"})
                        with self.assertRaises(VLLMValidationError) as refused:
                            asyncio_run(serving.create_responses(request))
                        error = create_error_response(refused.exception).error
                        self.assertEqual((error.code, error.param), (400, parameter))
                        self.assertIn(f"{parameter} {cause}", error.message)
        renderer.preprocess_chat.assert_not_called()
        engine.admit.assert_not_called()

    def test_a_continued_final_message_parses_alike_on_every_path(self):
        """A prompt that already closed reasoning -- a continued final
        message, which Responses selects by itself for an in-progress or
        incomplete last item -- leaves every generated token outside
        reasoning: on the batch parse exactly as on every chunking of the
        stream, with no reasoning counted."""
        prompt = rendered_prompt(
            [{"role": "user", "content": "test"},
             {"role": "assistant", "content": "The ans"}],
            continue_final=True,
        )
        self.assertTrue(TOKENIZER.decode(prompt).endswith("</think>\n\nThe ans"))
        ids, texts = generation("wer is 4.", None, "stop", None)
        for chunk in (None, 1, 3, 1000):
            with self.subTest(chunk=chunk):
                self.assertEqual(
                    parse("wer is 4.", chunk, prompt=prompt)[:3],
                    ("", "wer is 4.", []),
                )
                parser = PARSER(TOKENIZER, None, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)
                request = request_for(tools=[])
                if chunk is None:
                    parser.parse_output("".join(texts), request, prompt_token_ids=prompt,
                                        model_output_token_ids=ids,
                                        finish_reason="stop", stop_reason=None)
                else:
                    for start in range(0, len(ids), chunk):
                        last = start + chunk >= len(ids)
                        parser.parse_output_delta(
                            "".join(texts[start:start + chunk]), ids[start:start + chunk],
                            request, prompt_token_ids=prompt,
                            finish_reason="stop" if last else None, stop_reason=None,
                        )
                self.assertEqual(parser.reasoning_token_count, 0)
        # The open prompt still starts in reasoning, on every path alike.
        for chunk in (None, 1, 1000):
            with self.subTest(open=chunk):
                self.assertEqual(parse("plan</think>\n\nok", chunk)[:3],
                                 ("plan", "\n\nok", []))

    def test_responses_reports_one_generation_alike_on_both_transports(self):
        """The Responses route publishes one generation the same way whether it
        streams or not: the same items, statuses and reasoning usage. A token
        limit leaves the item it cut incomplete, and only that one; no
        arguments.done is sent for a cut call."""
        import asyncio

        from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
        from vllm.entrypoints.openai.responses.context import SimpleContext
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
        from vllm.outputs import CompletionOutput, RequestOutput

        engine = MagicMock()
        engine.model_config.max_model_len = 100000
        engine.model_config.get_diff_sampling_param.return_value = {}
        serving = OpenAIServingResponses(
            engine_client=engine, models=MagicMock(), online_renderer=MagicMock(),
            request_logger=None, chat_template=None, chat_template_content_format="auto",
            reasoning_parser=_served("SERVED_REASONING_PARSER"),
            tool_parser=_served("SERVED_TOOL_CALL_PARSER"), enable_auto_tools=True,
        )
        tool = {"type": "function", **TOOL["function"]}

        async def publish(text, finish, prompt, chunk):
            request = ResponsesRequest.model_validate({
                "model": "unit", "input": "test", "tools": [tool], "kv_scope": "unit",
                "stream": chunk is not None,
            })
            serving.parser(TOKENIZER, request.tools,
                           chat_template_kwargs=CHAT_TEMPLATE_KWARGS).adjust_request(request)
            context = SimpleContext(response_parser=serving.parser(
                TOKENIZER, request.tools, chat_template_kwargs=CHAT_TEMPLATE_KWARGS))
            ids, texts = generation(text, None, finish, None)
            step = chunk or len(ids)

            async def outputs():
                for start in range(0, len(ids), step):
                    last = start + step >= len(ids)
                    context.append_output(RequestOutput(
                        request_id="unit", prompt="p", prompt_token_ids=prompt,
                        prompt_logprobs=None, outputs=[CompletionOutput(
                            index=0, text="".join(texts[start:start + step]),
                            token_ids=ids[start:start + step], cumulative_logprob=None,
                            logprobs=None, finish_reason=finish if last else None,
                            stop_reason=None,
                        )], finished=last,
                    ))
                    yield context

            args = (request, request.to_sampling_params(1000, {}), outputs(), context,
                    "unit", TOKENIZER, RequestResponseMetadata(request_id="unit"))
            if chunk is None:
                return await serving.responses_full_generator(*args, created_time=1), None
            events = [event async for event in
                      serving.responses_stream_generator(*args, created_time=1)]
            done = [event.arguments for event in events
                    if event.type == "response.function_call_arguments.done"]
            return events[-1].response, done

        def shape(response):
            return response.status, response.usage.output_tokens_details.reasoning_tokens, [
                (item.type, item.status,
                 "".join(part.text for part in getattr(item, "content", None) or [])
                 if item.type != "function_call" else item.arguments)
                for item in response.output
            ]

        continued = rendered_prompt(
            [{"role": "user", "content": "test"},
             {"role": "assistant", "content": "The ans"}],
            continue_final=True,
        )
        cut_second = ("plan</think>\n\n" + call("one") + "\n"
                      + "<tool_call>\n<function=write>\n<parameter=text>\ntw")
        cases = {
            "two calls, the limit cuts the second": (cut_second, "length", OPEN_PROMPT, (
                "incomplete", len(encode("plan")), [
                    ("reasoning", "completed", "plan"),
                    ("message", "completed", "\n\n\n"),
                    ("function_call", "completed", '{"text": "one"}'),
                    ("function_call", "incomplete", '{"text": "tw"}'),
                ]), ['{"text": "one"}']),
            "the limit cuts reasoning": ("plan more pla", "length", OPEN_PROMPT, (
                "incomplete", len(encode("plan more pla")),
                [("reasoning", "incomplete", "plan more pla")]), []),
            "a continued final message": ("wer is 4.", "stop", continued, (
                "completed", 0, [("message", "completed", "wer is 4.")]), []),
        }
        for name, (text, finish, prompt, expected, done) in cases.items():
            for chunk in (None, 1, 4):
                with self.subTest(case=name, chunk=chunk):
                    response, arguments_done = asyncio.run(
                        publish(text, finish, prompt, chunk))
                    self.assertEqual(shape(response), expected)
                    if chunk is not None:
                        self.assertEqual(arguments_done, done)

    def test_include_reasoning_changes_nothing_the_route_admits(self):
        """include_reasoning shapes the response, never what the model may
        generate: the chat route admits the same prompt, sampling parameters
        and reasoning state with it on or off, and that state is the prompt's
        (reasoning is open at the end of the served generation prompt)."""
        for tools in (None, [TOOL]):
            with self.subTest(tools=bool(tools)):
                shown, hidden = (
                    _chat_route_admits(include_reasoning=flag, tools=tools)
                    for flag in (True, False)
                )
                self.assertIs(shown["reasoning_ended"], False)
                self.assertEqual(shown, hidden)

    def test_a_responses_stream_is_its_own_snapshot(self):
        """Each streamed item is the concatenation of its deltas; its done event
        carries exactly that; the terminal response is the done items, in
        order. For every engine chunking, on the served parsers."""
        deltas = {
            "response.output_text.delta", "response.reasoning_text.delta",
            "response.function_call_arguments.delta",
        }
        generations = (
            ("plan</think>\n\nThe answer \u03a9.", "stop"),
            ("plan</think>\n\nLet me look.\n\n" + call("a b"), "stop"),
            ("plan</think>\n\n" + call("a") + "\n" + call("b"), "stop"),
            ("plan</think>\n\n" + call("a") + "\n<tool_call>\n<function=write>\n"
             "<parameter=text>\nb", "length"),
            ("plan and more pla", "length"),
        )
        for text, finish in generations:
            for chunk in (1, 2, 3, 5, 64):
                with self.subTest(text=text, chunk=chunk):
                    events = _responses_events(text, chunk, finish)
                    built, done = {}, {}
                    for event in events:
                        if event.type in deltas:
                            built[event.output_index] = (
                                built.get(event.output_index, "") + event.delta
                            )
                        elif event.type == "response.output_item.done":
                            done[event.output_index] = event.item
                    self.assertEqual(
                        {index: _item_text(item) for index, item in done.items()
                         if index in built},
                        built,
                    )
                    terminal = events[-1].response
                    self.assertEqual(
                        [item.model_dump() for item in terminal.output],
                        [done[index].model_dump() for index in sorted(done)],
                    )

    def test_a_responses_history_refusal_names_the_input_item(self):
        """Responses validates its tool history after converting it to chat
        messages, whose positions the caller never sent: a refusal names the
        input item and the field the caller set."""
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.entrypoints.openai.responses.utils import construct_input_messages
        from vllm.exceptions import VLLMValidationError

        history = [{"role": "user", "content": "hi"}] + [
            {"type": "function_call", "id": "fc" + cid, "call_id": cid,
             "name": "write", "arguments": "{}"} for cid in ("A", "B")
        ] + [{"type": "function_call_output", "call_id": cid, "output": "x"}
             for cid in ("B", "A")]
        request = ResponsesRequest.model_validate(
            {"model": "unit", "input": history, "kv_scope": "unit"})
        with self.assertRaises(VLLMValidationError) as refused:
            construct_input_messages(request_input=request.input)
        self.assertEqual(refused.exception.parameter, "input")
        self.assertIn("input[3] has call_id 'B'; expected 'A'", str(refused.exception))

    def test_an_anthropic_history_refusal_names_the_block_sent(self):
        """An Anthropic request is validated after conversion to chat
        messages, whose positions and id field the caller never sent: a
        refusal names the content block and its tool_use_id or id."""
        from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
        from vllm.exceptions import VLLMValidationError

        def refusal(calls, results):
            request = AnthropicMessagesRequest.model_validate({
                "model": "unit", "max_tokens": 8, "kv_scope": "unit",
                "system": "Be terse.",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Calling."}]
                     + [{"type": "tool_use", "name": "write", "input": {},
                         **({"id": cid} if cid else {})} for cid in calls]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": cid, "content": "x"}
                        for cid in results]},
                ]})
            with self.assertRaises(VLLMValidationError) as refused:
                AnthropicServingMessages._convert_anthropic_to_openai_request(
                    request, merge_inline_system=True)
            self.assertEqual(refused.exception.parameter, "messages")
            return str(refused.exception)

        self.assertIn("Tool result at messages[2].content[1] has tool_use_id 'A'; "
                      "expected 'B'", refusal(["A", "B"], ["A", "A"]))
        self.assertIn("Tool call messages[1].content[1] is missing its transport id",
                      refusal([None], ["A"]))

    def test_a_responses_message_carries_no_log_probabilities(self):
        """A message's log probabilities would be those of the tokens its text
        came from, and the served parser divides one generated token between
        the message and a call: after "See:" the grammar admits the ordinary
        " <" token, whose space ends the message and whose "<" opens the call.
        No list of whole tokens is the message's, so the route refuses each
        field that asks for one, naming it."""
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
        from vllm.exceptions import VLLMValidationError

        message, body = "\n\nSee: ", call("v")
        answer = ordinary(message + body)
        shared = [TOKENIZER.decode([token]) for token in answer].index(" <")
        self.assertEqual(TOKENIZER.decode(answer[:shared]) + " ", message)
        matcher = grammar_matcher(request_for())
        for token in [*answer, MODEL_EOS[0]]:
            self.assertTrue(matcher.accept_token(token))
        self.assertTrue(matcher.is_terminated())
        for chunk in (None, 1, shared + 1):
            with self.subTest(chunk=chunk):
                self.assertEqual(
                    parse("plan</think>" + message + body, chunk,
                          ids=encode("plan</think>") + answer)[:3],
                    ("plan", message, [("write", '{"text": "v"}')]),
                )
        for field, value in (("include", ["message.output_text.logprobs"]),
                             ("top_logprobs", 5)):
            with self.subTest(field=field):
                with self.assertRaises(VLLMValidationError) as refused:
                    ResponsesRequest.model_validate(
                        {"model": "unit", "input": "test", "kv_scope": "unit",
                         field: value})
                self.assertEqual(refused.exception.parameter, field)

    def test_chat_reports_every_generated_tokens_log_probability(self):
        """Chat's log probabilities are the choice's: one per generated id, in
        order, the end of reasoning, the calls and the end of turn included --
        in the full response and across the streamed chunks alike, for every
        engine chunking, including steps the parser releases no text for."""
        generations = (
            ("plan</think>\n\nThe answer.", "stop", None),
            ("plan</think>\n\nLet me look.\n\n" + call("a b"), "stop", [TOOL]),
            ("plan</think>\n\n" + call("a") + "\n<tool_call>\n<function=write>\n"
             "<parameter=text>\nb", "length", [TOOL]),
        )
        served = _served_chat()
        for text, finish, tools in generations:
            for chunk in (None, 1, 2, 3, 5, 64):
                with self.subTest(text=text, chunk=chunk):
                    ids, reported = _chat_logprobs(served, text, chunk, finish, tools)
                    self.assertEqual(
                        [logprob for _, logprob in reported],
                        [-(k + 1) / 100 for k in range(len(ids))],
                    )


if __name__ == "__main__":
    unittest.main()
