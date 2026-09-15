#!/usr/bin/env python3
"""CPU-only validation of installed raw-image inputs and rendered token spans."""
from types import SimpleNamespace

from pydantic import ValidationError
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
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

print("Installed raw-media and rendered-prompt contract passed")
