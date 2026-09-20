#!/usr/bin/env python3
"""Prove the built runtime's CPU-side contract, starting with the Qwen grammar.

The Qwen tool grammar is the part of this deployment that no other unit covers:
``get_model_structural_tag("qwen_3_coder", ...)`` renders a declared schema as
the XML transport, and the exclusion of ``<parameter=`` from the raw string
channel is what makes an argument merge -- one call published as another --
ungrammatical rather than merely unlikely. A change to the builder's dumped
shape is invisible to every other gate in this repository.

These assertions used to live in a heredoc inside ``containers/Dockerfile.runtime``
and therefore ran only during ``docker buildx build``. When stage 35 moved
``qwen_3_coder`` to the vLLM-owned builder the heredoc still read the old
``content.json_schema`` path, and the build died on ``KeyError`` after hours --
twice. As a file the same assertions are one of the units
``./scripts/build-vllm.sh check`` runs in seconds, and the image runs this very
file rather than a copy of it, so the two can never disagree.

Both callers reach the same paths: the units and the served template ship at
``/opt/qwen38`` in the image, and ``build-vllm.sh check`` mounts the template
there and runs this file beside its siblings in ``scripts/``. Facts about the
image's own assembly -- COPY modes, the removed modules, the other installed
units' execution -- stay in the Dockerfile, which is the only place they exist.
"""

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponseIncompleteEvent
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
)
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.parser.qwen3 import Qwen3Parser, _qwen3_arg_converter
from vllm.v1.structured_output import StructuredOutputManager
from pydantic import ValidationError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from unittest.mock import MagicMock
import json
from pathlib import Path

# The numerical audit units ship in the image but run later, on the GPU host.
# A syntax error in one of them would otherwise surface only there. They sit
# beside this file in both callers -- ``/opt/qwen38`` in the image, ``scripts/``
# in the check -- so one expression covers both.
_UNIT_DIR = Path(__file__).resolve().parent
for audit_path in (
    _UNIT_DIR / "turboquant_k8v4_unit.py",
    _UNIT_DIR / "qwen38_context_unit.py",
    _UNIT_DIR / "nvfp4_kernel_unit.py",
):
    compile(audit_path.read_text(encoding="utf-8"), str(audit_path), "exec")

tool = ChatCompletionToolsParam.model_validate(
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["path", "max_bytes"],
                "additionalProperties": False,
            },
        },
    }
)
assert tool.function.strict is None
assert json.loads(
    _qwen3_arg_converter("<parameter=path>/partial", partial=False)
) == {"path": "/partial"}
assert ResponseIncompleteEvent is not None
processor = SimpleStreamingEventProcessor()
processor.state.current_state = _StateType.TOOL_CALL
processor.state.current_item_id = "fc_build"
processor.state.tool_call_id = "call_build"
processor.state.tool_call_name = "read_file"
processor.state.tool_call_index = 0
processor.state.accumulated_text = '{"path":"/partial"}'
processor.state.has_emitted_tool_call_delta = True
incomplete_events = processor.close_current(incomplete=True)
assert [event.type for event in incomplete_events] == ["response.output_item.done"]
assert incomplete_events[0].item.status == "incomplete"
qwen_tag = get_model_structural_tag("qwen_3_coder", [tool], "auto", False)
assert qwen_tag is not None
assert get_model_structural_tag("qwen_3_coder", [tool], "none", False) is None
assert get_model_structural_tag("llama", [tool], "auto", False) is None
assert get_model_structural_tag("qwen_3_5", [tool], "auto", False) is None

try:
    get_max_tokens(
        max_model_len=262144,
        max_tokens=1,
        input_length=262144,
        default_sampling_params={"max_tokens": 65536},
    )
except VLLMValidationError as exc:
    boundary_error = str(exc)
    assert "equals model's maximum context length (262144)" in boundary_error
    assert "leaving no room for the required output token" in boundary_error
else:
    raise AssertionError("a full-context prompt was allowed to request output")

# A Qwen tool schema is load-bearing even when OpenAI's optional strict flag
# is omitted or explicitly false. Other structural-tag models retain upstream
# opt-in semantics. The vLLM-owned Qwen builder renders that schema as the XML
# transport rather than as one embedded JSON document: each declared parameter
# becomes a named tag in the schema's own order, a string rides the
# unconstrained channel, every other declared type keeps its JSON sub-schema
# verbatim, and "additionalProperties": false is the absence of any element
# that could carry an undeclared name. Assert that rendering, and that all
# three spellings of strict are one and the same grammar.
strict_dumps = []
for strict_value in (None, False, True):
    strict_tool = tool.model_copy(deep=True)
    strict_tool.function.strict = strict_value
    tag = get_model_structural_tag("qwen_3_coder", [strict_tool], "auto", False)
    assert tag is not None
    call_tag = tag.model_dump()["format"]["tags"][0]
    assert call_tag["begin"] == "<tool_call>\n<function=read_file>\n"
    assert call_tag["end"] == "\n</function>\n</tool_call>"
    # A schema the builder erased is a star of any-named parameters accepting
    # any value, never the declared-order sequence.
    body = call_tag["content"]
    assert body["type"] == "sequence", body
    declared = [e for e in body["elements"] if e["type"] != "regex"]
    # Exactly the two declared parameters, in the declared order, each bound
    # by name and each required -- an optional one arrives wrapped in
    # "optional", and an undeclared one would need the trailing "star" that
    # "additionalProperties": false is precisely the absence of.
    assert [e["type"] for e in declared] == ["tag", "tag"], declared
    assert [e["begin"] for e in declared] == [
        "<parameter=path>",
        "<parameter=max_bytes>",
    ], declared
    assert [e["end"] for e in declared] == ["</parameter>", "</parameter>"]
    # The string parameter rides the unconstrained channel, which excludes its
    # own closer and the next parameter's opener. That second exclusion is what
    # makes the argument merge that once published one call as another
    # ungrammatical rather than merely unlikely.
    path_value = declared[0]["content"]
    assert path_value["type"] == "any_text", path_value
    assert path_value["excludes"] == ["<parameter=", "</parameter>"]
    # Every other declared type carries its JSON sub-schema verbatim.
    max_bytes_value = declared[1]["content"]
    assert max_bytes_value["type"] == "sequence", max_bytes_value
    embedded = [e for e in max_bytes_value["elements"] if e["type"] != "regex"]
    assert [e["type"] for e in embedded] == ["json_schema"], embedded
    assert embedded[0]["json_schema"] == {"type": "integer"}
    strict_dumps.append(call_tag)
assert strict_dumps[0] == strict_dumps[1] == strict_dumps[2]

# There is one string channel and it is the excluded raw one. A string
# parameter carrying pattern, format, minLength or maxLength has no grammar at
# all -- the exclusion that makes an argument merge ungrammatical cannot be
# written inside a length-bounded regex -- so the registry refuses it instead
# of building a second channel that drops the exclusion. The refusal names the
# tool and the property because the fix is an edit to that one declaration.
for refused_key, refused_value in (
    ("pattern", "^[a-z]+$"),
    ("format", "uuid"),
    ("minLength", 1),
    ("maxLength", 64),
):
    constrained = ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "update_todo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todo_id": {"type": "string", refused_key: refused_value}
                    },
                    "required": ["todo_id"],
                    "additionalProperties": False,
                },
            },
        }
    )
    try:
        get_model_structural_tag("qwen_3_coder", [constrained], "auto", False)
    except ValueError as exc:
        refusal = str(exc)
        assert "update_todo" in refusal, refusal
        assert "todo_id" in refusal, refusal
        assert refused_key in refusal, refusal
        assert "validate it in the tool" in refusal, refusal
    else:
        raise AssertionError(f"a string declaring {refused_key} was given a grammar")

# The undeclared-name channel is a value production too, and is refused on the
# same terms under the name the schema gives it.
extra_constrained = ChatCompletionToolsParam.model_validate(
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "parameters": {
                "type": "object",
                "properties": {"body": {"type": "string"}},
                "required": ["body"],
                "additionalProperties": {"type": "string", "maxLength": 8},
            },
        },
    }
)
try:
    get_model_structural_tag("qwen_3_coder", [extra_constrained], "auto", False)
except ValueError as exc:
    refusal = str(exc)
    assert "write_note" in refusal, refusal
    assert "additionalProperties" in refusal, refusal
else:
    raise AssertionError("a constrained additionalProperties string got a grammar")

# Exactly those four keys are refused. An ordinary string, annotations and all,
# still rides the one channel and still carries both exclusions.
annotated = ChatCompletionToolsParam.model_validate(
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {
                        "type": "string",
                        "title": "Body",
                        "description": "Note text.",
                        "default": "",
                    }
                },
                "required": ["body"],
                "additionalProperties": False,
            },
        },
    }
)
annotated_tag = get_model_structural_tag("qwen_3_coder", [annotated], "auto", False)
annotated_body = annotated_tag.model_dump()["format"]["tags"][0]["content"]
annotated_declared = [e for e in annotated_body["elements"] if e["type"] != "regex"]
assert [e["begin"] for e in annotated_declared] == ["<parameter=body>"]
annotated_value = annotated_declared[0]["content"]
assert annotated_value["type"] == "any_text", annotated_value
assert annotated_value["excludes"] == ["<parameter=", "</parameter>"]

# Decoder-time boundary invariant: an explicit </think> is excluded from the
# grammar, while an implicit <tool_call> reasoning terminator is retained as
# the Qwen structural grammar's trigger token.
tokenizer = MagicMock()
tokenizer.get_vocab.return_value = {
    "<think>": 101, "</think>": 102, "<tool_call>": 103, "</tool_call>": 104,
}
parser = Qwen3Parser(tokenizer)
assert parser.extract_content_ids([101, 11, 102, 21]) == [21]
assert parser.extract_content_ids([101, 11, 103]) == [103]
assert parser.is_reasoning_end_streaming([101, 11, 103], [103])
implicit_end = StructuredOutputManager._find_reasoning_end_index(
    parser, [101, 11, 103], 2
)
assert implicit_end == 1
request_stub = type("Request", (), {})()
request_stub.all_token_ids = [101, 11, 103]
request_stub.structured_output_request = type("Structured", (), {})()
request_stub.structured_output_request.reasoning_end_token_index = implicit_end
assert StructuredOutputManager.trim_reasoning_for_advance(
    object(), request_stub, [103]
) == [103]

# The Qwen history wire format is positional and does not expose transport
# IDs to the model. Require a complete, ordered one-result-per-call sequence
# rather than allowing corrupted tool context or silently reordering it.
valid_history = [
    {"role": "user", "content": "test"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "call_a", "content": "A"},
    {"role": "tool", "tool_call_id": "call_b", "content": "B"},
    {"role": "user", "content": "continue"},
]
ChatCompletionRequest(model="qwen3.8", messages=valid_history, max_tokens=1)
invalid_histories = (
    [
        {"role": "user", "content": "test"},
        {"role": "tool", "tool_call_id": "orphan", "content": "bad"},
    ],
    [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_b", "content": "B"},
        {"role": "tool", "tool_call_id": "call_a", "content": "A"},
    ],
    [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    ],
)
for invalid_history in invalid_histories:
    try:
        ChatCompletionRequest(
            model="qwen3.8", messages=invalid_history, max_tokens=1
        )
    except VLLMValidationError:
        pass
    else:
        raise AssertionError(f"malformed tool history was accepted: {invalid_history}")

defaults = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "thinking_token_budget": 262144,
    "final_response_token_budget": 131072,
}
request = ChatCompletionRequest(
    model="qwen3.8",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=1024,
)
sampling = request.to_sampling_params(1024, defaults)
assert sampling.temperature == 1.0
assert sampling.top_p == 0.95
assert sampling.top_k == 20
assert sampling.min_p == 0.0
assert sampling.presence_penalty == 0.0
assert sampling.repetition_penalty == 1.0
assert sampling.repetition_detection is None
assert sampling.thinking_token_budget == 262144
assert sampling.final_response_token_budget == 131072

anthropic = AnthropicMessagesRequest(
    model="qwen3.8",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=1024,
    thinking={"type": "enabled", "budget_tokens": 512},
)
converted = AnthropicServingMessages._build_base_request(
    anthropic, [{"role": "user", "content": "test"}]
)
assert converted.thinking_token_budget == 512
assert converted.chat_template_kwargs == {"enable_thinking": True}

adaptive = AnthropicMessagesRequest(
    model="qwen3.8",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=1024,
    thinking={"type": "adaptive"},
    output_config={"effort": "max"},
)
adaptive_converted = AnthropicServingMessages._build_base_request(
    adaptive, [{"role": "user", "content": "test"}]
)
AnthropicServingMessages._handle_output_config(adaptive_converted, adaptive)
assert adaptive_converted.thinking_token_budget is None
assert adaptive_converted.reasoning_effort == "max"

for rejected in (
    {"thinking": {"type": "disabled"}},
    {"output_config": {"effort": "low"}},
):
    try:
        AnthropicMessagesRequest(
            model="qwen3.8",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1024,
            **rejected,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError(f"unsafe thinking configuration was accepted: {rejected}")

def raise_exception(message):
    raise RuntimeError(message)

environment = ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
    extensions=["jinja2.ext.loopcontrols"],
)
template = environment.from_string(open("/opt/qwen38/chat_template.jinja").read())
base_context = {
    "messages": [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "visible answer",
            "reasoning_content": "OLD_HIDDEN_TRACE",
        },
        {"role": "user", "content": "next"},
    ],
    "add_generation_prompt": True,
    "enable_thinking": True,
    "tools": None,
    "raise_exception": raise_exception,
}
rendered = template.render(**base_context, reasoning_effort="max")
assert "OLD_HIDDEN_TRACE" in rendered
assert template.render(**base_context, reasoning_effort="high") == rendered
for rejected_context in (
    {"reasoning_effort": "medium"},
    {"reasoning_effort": "low"},
    {"enable_thinking": False},
):
    context = dict(base_context)
    context.update(rejected_context)
    try:
        template.render(**context)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"unsafe template configuration was accepted: {rejected_context}")

# Every generation surface carries an opaque agent ID through the common
# sampling-parameter channel. Shared-cache behavior has its own installed unit.
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.engine.input_processor import require_kv_scope

scoped = ChatCompletionRequest(
    model="qwen3.8",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=8,
    kv_scope="agent_scope",
)
scoped_sampling = scoped.to_sampling_params(8, defaults)
assert scoped_sampling.extra_args["kv_scope"] == "agent_scope"
for scoped_request in (
    CompletionRequest(model="m", prompt="hi", kv_scope="agent_scope"),
    ResponsesRequest(model="m", input="hi", kv_scope="agent_scope"),
    AnthropicMessagesRequest(
        model="m",
        max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
        kv_scope="agent_scope",
    ),
    GenerateRequest(
        token_ids=[1], sampling_params=SamplingParams(), kv_scope="agent_scope"
    ),
):
    assert scoped_request.kv_scope == "agent_scope"

# Generation requires an agent ID, and the requirement lives on the path
# into the engine rather than on these models: /v1/chat/completions/render
# renders through the very same ChatCompletionRequest while allocating
# nothing, so a model-level requirement would demand an identity from an
# endpoint that owns no context. Prove both directions here, plus the fact
# that keeps them separable -- the shared model still builds unscoped.
assert require_kv_scope(scoped_sampling) == "agent_scope"
unscoped = ChatCompletionRequest(
    model="qwen3.8",
    messages=[{"role": "user", "content": "test"}],
    max_tokens=8,
)
assert unscoped.kv_scope is None
unscoped_sampling = unscoped.to_sampling_params(8, defaults)
assert "kv_scope" not in (unscoped_sampling.extra_args or {})
try:
    require_kv_scope(unscoped_sampling)
except VLLMValidationError as exc:
    assert exc.parameter == "kv_scope", exc.parameter
    assert "required for generation" in str(exc), str(exc)
else:
    raise AssertionError("generation admitted a request that names no agent")
for malformed in ("", " ", "\t\n", 0, [], {}):
    try:
        require_kv_scope(SamplingParams(max_tokens=1, extra_args={"kv_scope": malformed}))
    except VLLMValidationError as exc:
        assert exc.parameter == "kv_scope", exc.parameter
    else:
        raise AssertionError(f"malformed kv_scope accepted: {malformed!r}")

# One batch is one caller: every conversation is submitted under its agent.
batched = BatchChatCompletionRequest(
    model="qwen3.8",
    messages=[
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ],
    max_tokens=8,
    kv_scope="agent_scope",
)
derived = [batched.to_chat_completion_request(m) for m in batched.messages]
assert [d.kv_scope for d in derived] == ["agent_scope", "agent_scope"]
assert require_kv_scope(derived[0].to_sampling_params(8, defaults)) == "agent_scope"

assert hasattr(EngineArgs, "kv_cache_users")
assert not hasattr(EngineArgs, "kv_cache_memory_bytes")
assert not hasattr(EngineArgs, "kv_offloading_size")
try:
    CPUOffloadingSpec._validate_extra_config({"definitely_unknown_knob": 1})
except ValueError:
    pass
else:
    raise AssertionError("unknown offload configuration key accepted")

# The Cohere HTTP surface must be absent as an enforced fact of the server
# assembly, not as the accident of the uninstallable cohere SDK: register
# every generate router in-image and prove no /cohere route exists while
# the five importable identity surfaces are all mounted.
from fastapi import FastAPI
from vllm.entrypoints.generate.api_router import register_generate_api_routers

from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from types import SimpleNamespace

route_app = FastAPI()
# The token-in-token-out attach reads app.state.args (tokens_only servers
# mount an extra abort route); provide the minimal state a real server has.
route_app.state.args = SimpleNamespace(tokens_only=False)
register_generate_api_routers(route_app)
register_scale_out_api_routers(route_app, ("generate",))
route_paths = {getattr(route, "path", "") for route in route_app.routes}
assert not any(path.startswith("/cohere") for path in route_paths), route_paths
# /generative_scoring is withheld for the mirror-image reason: it reaches the
# engine's generative path but names no agent, so its blocks could not be
# filed under any single context. Every mounted generative route here is an
# identity surface.
assert "/generative_scoring" not in route_paths, sorted(route_paths)
for required_path in (
    "/v1/chat/completions",
    "/v1/chat/completions/batch",
    "/v1/completions",
    "/v1/responses",
    "/v1/messages",
    "/inference/v1/generate",
):
    assert required_path in route_paths, (required_path, sorted(route_paths))

print("qwen-grammar-unit: PASS")
