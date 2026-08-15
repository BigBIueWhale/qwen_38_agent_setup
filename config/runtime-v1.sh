#!/usr/bin/env bash
# Single source of truth for the only supported serving profile.

readonly PROFILE_VERSION="single-loopback-text-k8v4-v1"
readonly CONTAINER_NAME="qwen38-agent-native"
readonly CONTAINER_LABEL="Qwen_best_model_ever"
readonly IMAGE_TAG="qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v1"
readonly EXPECTED_IMAGE_ID="sha256:bcbc6241a543e3308720cf16401451a98ae6ed5f34ab41b48b9555e9065e5e6c"
readonly BASE_IMAGE_TAG="qwen38-vllm:main-9df9b0b"
readonly EXPECTED_BASE_IMAGE_ID="sha256:fa4a002a88b7043a1a89966dea8a500fe9696f84e75730d9da916f916048d401"
readonly IMAGE_ARCHIVE_NAME="qwen38-vllm-images-runtime-v1.tar"
readonly IMAGE_ARCHIVE_SHA256="a226481417a7ff381e931051da3b5a30189b75894f2b58802419690c283626c4"

readonly MODEL_DIR_NAME="Qwen3.8-27B-NVFP4-Unsloth"
readonly MODEL_REPOSITORY="unsloth/Qwen3.8-27B-NVFP4"
readonly MODEL_REVISION="a767244d27bd76589a3e3b2ab4e64032c4ebc7af"
readonly MODEL_SHA256="c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05"
readonly MTP_SHA256="1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe"
readonly MODEL_MANIFEST_NAME="model-snapshot-a767244d.sha256"
readonly MODEL_MANIFEST_SHA256="43ebab0147f818aad9887f5d2db9b88eb25111c0e458c7e40c5af508fa39019b"
MODEL_FILES=(
  .gitattributes
  README.md
  chat_template.jinja
  config.json
  generation_config.json
  model.safetensors
  model.safetensors.index.json
  model_mtp.safetensors
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)
readonly -a MODEL_FILES

readonly VLLM_COMMIT="9df9b0b0a1816b6d0d0f6ecd0da563cc37fd72f5"
readonly PATCH_DIFF_SHA256="a9721067f1a7ee9497a4bd51e47e3a474561189e881b4704bfc4beac8ea48380"
readonly PATCHED_FILE_SHA256="0aa02c874d33c1113a49cb1aab49cfdc53e6a0d77fdc9ba91f7f89e6bddc0367"
readonly UPSTREAM_FILE_SHA256="48994be137f3d25d4ee4f79ba2b89b0a6c3d988085079ffea1d241a34c2c755f"
readonly SOURCE_DATE_EPOCH="1786751423"
readonly RUNTIME_DOCKERFILE_SHA256="53df4a70a621788e754fd56dc0198e77af1e35b0a2ab2aaa1fc435db1e8f8993"
readonly DOCKERIGNORE_SHA256="632f8086e0ba807a2f4dd411ba4b23fb59df0e1a696582ab1d63a98b59d739af"

readonly EXPECTED_DOCKER_VERSION="29.7.2"
readonly EXPECTED_NVIDIA_CONTAINER_CLI_VERSION="1.19.1"
readonly EXPECTED_GPU_NAME="NVIDIA GeForce RTX 5090"
readonly EXPECTED_GPU_MEMORY_MIB="32607"
readonly EXPECTED_DRIVER_VERSION="595.71.05"

readonly LISTEN_HOST="127.0.0.1"
readonly LISTEN_PORT="8000"
readonly ENDPOINT="http://${LISTEN_HOST}:${LISTEN_PORT}"
readonly SERVED_MODEL="qwen3.8-27b-nvfp4-k8v4"
readonly MAX_MODEL_LEN="262144"
readonly CACHE_VOLUME="qwen38-vllm-cache-single-loopback-v1"
readonly STARTUP_TIMEOUT_SECONDS="600"

readonly EXPECTED_RUNTIME_REPORT=$'python=3.12.3\nvllm=0.27.2rc1.dev106+g9df9b0b0a\ntorch=2.13.0+cu130\ntransformers=5.15.0\ntokenizers=0.22.2\nsafetensors=0.8.0\ncompressed-tensors=0.17.0\nflashinfer-python=0.6.16.post3\ntriton=3.7.1\nnumpy=2.3.5\nfastapi=0.136.3\nuvicorn=0.52.3\ntorch_cuda=13.0\ncuda_capability=12.0'

RUNTIME_ENV=(
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "DO_NOT_TRACK=1"
  "VLLM_NO_USAGE_STATS=1"
  "VLLM_DEBUG_WORKSPACE=1"
  "GLOO_SOCKET_IFNAME=lo"
  "NCCL_SOCKET_IFNAME=lo"
)
readonly -a RUNTIME_ENV

VLLM_ARGS=(
  /model
  --served-model-name "${SERVED_MODEL}"
  --host "${LISTEN_HOST}"
  --port "${LISTEN_PORT}"
  --language-model-only
  --model-impl vllm
  --config-format hf
  --load-format safetensors
  --tokenizer /model
  --chat-template /model/chat_template.jinja
  --chat-template-content-format openai
  --generation-config /model
  --quantization compressed-tensors
  --dtype bfloat16
  --kv-cache-dtype turboquant_k8v4
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs 1
  --max-num-batched-tokens 2048
  --kv-cache-memory 6925634765
  --cpu-offload-gb 0
  --enable-prefix-caching
  --enable-chunked-prefill
  --attention-config.flash_attn_version=2
  --kernel-config.enable_flashinfer_autotune=False
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs
  '{"enable_thinking":true,"preserve_thinking":true}'
)
readonly -a VLLM_ARGS
