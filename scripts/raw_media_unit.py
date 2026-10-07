#!/usr/bin/env python3
"""CPU-only validation of installed raw-image inputs and rendered token spans."""
from types import SimpleNamespace

from pydantic import ValidationError
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs, RenderedPromptTokens
from vllm.multimodal.processing.processor import MultiModalProcessingInfo, PromptReplacement

for fields in (
    {"features": {"mm_hashes": {"image": ["untrusted"]}}},
    {"content_parts": [{"type": "image_url", "uuid": "untrusted"}]},
    {"content_parts": [{"type": "audio_url", "url": "untrusted"}]},
):
    try:
        GenerateRequest(token_ids=[1], sampling_params={}, **fields)
    except ValidationError:
        pass
    else:
        raise AssertionError("Untrusted media representation was accepted")

processor = object.__new__(Qwen3VLMultiModalProcessor)
processor.info = SimpleNamespace(
    get_tokenizer=lambda: None,
    get_hf_config=lambda: SimpleNamespace(
        image_token_id=101, vision_start_token_id=102,
        vision_end_token_id=103, video_token_id=104),
)
updates = {"image": [[PromptReplacement("image", [101], [101, 101]).resolve(0)]]}
info = MultiModalProcessingInfo(
    kwargs={"image": [None]}, hashes={"image": ["native"]}, prompt_updates=updates)
processor._cached_apply_hf_processor = lambda inputs, timing: (
    list(inputs.prompt.token_ids), info, False)
items = SimpleNamespace(get_all_counts=lambda: {"image": 1})
for tokens, valid in (
    ([7, 102, 101, 101, 103, 8], True),
    ([7, 102, 101, 103, 8], False),
    ([7, 102, 101, 101, 101, 103, 8], False),
    ([7, 102, 101, 101, 103, 101], False),
):
    try:
        output = processor.apply(
            ProcessorInputs(RenderedPromptTokens(tokens), items),
            TimingContext(enabled=False))
    except VLLMValidationError:
        assert not valid
    else:
        assert valid
        assert output["prompt_token_ids"] == tokens
        span = output["mm_placeholders"]["image"][0]
        assert (span.offset, span.length) == (2, 2)

# Ids that supply no image hold no image span: the rendered-span check the
# image path runs, with no images, refuses one before it reaches the model as
# bare placeholder text; plain text passes unchanged.
from vllm.renderers.base import BaseRenderer

text_route = SimpleNamespace(mm_processor=processor)
BaseRenderer.require_no_rendered_media(text_route, [7, 8, 9])
for tokens in ([7, 102, 101, 101, 103, 8], [7, 101, 8]):
    try:
        BaseRenderer.require_no_rendered_media(text_route, tokens)
    except VLLMValidationError as error:
        assert error.parameter == "token_ids"
    else:
        raise AssertionError("Rendered image tokens were accepted as text")

# One decision of what an image part is. Every image part shape the chat
# parser renders as an image -- both part types, extra keys, a null uuid, beside
# a plain-string item -- is carried by the render route, in order, in the
# transport's one image shape; an input_image without its required detail is
# the caller's mistake, refused naming it.
import asyncio
import base64
import io
from unittest.mock import AsyncMock

from PIL import Image

from vllm.entrypoints.chat_utils import _parse_chat_message_content_part
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.scale_out.render.serving import ServingRender

buffer = io.BytesIO()
Image.new("RGB", (8, 8)).save(buffer, format="PNG")
png = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
parts = [
    {"type": "image_url", "image_url": {"url": png}},
    {"type": "image_url", "image_url": {"url": png, "detail": "high"}},
    {"type": "input_image", "image_url": png, "detail": "auto"},
    {"type": "image_url", "image_url": {"url": png, "name": "x.png"}},
    {"type": "image_url", "image_url": {"url": png}, "cache_control": {}},
    {"type": "image_url", "image_url": {"url": png}, "uuid": None},
    "plain text",
    {"type": "text", "text": "describe"},
]
rendered_images = []
recorder = SimpleNamespace(
    model_config=SimpleNamespace(enable_prompt_embeds=False),
    parse_image=lambda url, uuid: rendered_images.append(url),
)
for part in parts:
    _parse_chat_message_content_part(
        part, recorder, wrap_dicts=True, interleave_strings=False)
assert len(rendered_images) == 6, rendered_images

render = object.__new__(ServingRender)
render._check_model = AsyncMock(return_value=None)
render.model_config = SimpleNamespace(max_model_len=64)
render.default_sampling_params = {}
render.override_max_tokens = None
render.online_renderer = SimpleNamespace(render_chat=AsyncMock(
    return_value=([], [{"type": "token", "prompt_token_ids": [7, 8]}])))
request = ChatCompletionRequest(model="unit", messages=[
    {"role": "user", "content": parts[:3]},
    {"role": "user", "content": parts[3:]},
])
import vllm.entrypoints.scale_out.render.serving as render_module

render_module.extract_prompt_components = lambda config, engine_input: SimpleNamespace(
    token_ids=engine_input["prompt_token_ids"])
render_module.extract_prompt_len = lambda config, engine_input: 2
generate = asyncio.run(render.render_chat_request(request))
assert [part.image_url.url for part in generate.content_parts] == rendered_images
assert [part.image_url.detail for part in generate.content_parts] == [
    "auto", "high", "auto", "auto", "auto", "auto"]
GenerateRequest.model_validate_json(generate.model_dump_json())

try:
    _parse_chat_message_content_part(
        {"type": "input_image", "image_url": png}, recorder,
        wrap_dicts=True, interleave_strings=False)
except VLLMValidationError as error:
    assert error.parameter == "detail"
else:
    raise AssertionError("An input_image without detail was accepted")

# A prompt the chat template renders is never truncated: a token-count cut
# from the tokenizer's default side removes the template's markers, the system
# text and the tool definitions first. Chat and Responses refuse the parameter
# that would ask for it, and neither tokenizes with a truncation.
model_config = SimpleNamespace(max_model_len=64)
for build, parameter in (
    (lambda: ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}],
         "truncate_prompt_tokens": -1}), "truncate_prompt_tokens"),
    (lambda: ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}],
         "truncation_side": "left"}), "truncation_side"),
    (lambda: ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "kv_scope": "agent", "truncation": "auto",
         "max_output_tokens": 8}), "truncation"),
):
    try:
        build()
    except VLLMValidationError as refusal:
        assert refusal.parameter == parameter, (parameter, refusal.parameter)
    else:
        raise AssertionError(f"{parameter} truncation of a chat prompt was accepted")
for request in (
    ChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}],
         "max_completion_tokens": 8}),
    ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "kv_scope": "agent", "max_output_tokens": 8}),
):
    assert request.build_tok_params(model_config).truncate_prompt_tokens is None
print("Installed raw-media and rendered-prompt contract passed")
