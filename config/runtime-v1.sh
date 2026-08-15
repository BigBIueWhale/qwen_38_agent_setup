#!/usr/bin/env bash
# Single source of truth for the only supported serving profile.

readonly PROFILE_VERSION="single-loopback-text-k8v4-agent-v3"
readonly CONTAINER_NAME="qwen38-agent-native"
readonly CONTAINER_LABEL="Qwen_best_model_ever"
readonly IMAGE_TAG="qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v3"
readonly EXPECTED_IMAGE_ID="sha256:5bf315d0db49b61a24addaeae96a1551e87b307cf3da6168f306a7e6c71e6205"
readonly BASE_IMAGE_TAG="qwen38-vllm:main-9df9b0b"
readonly EXPECTED_BASE_IMAGE_ID="sha256:fa4a002a88b7043a1a89966dea8a500fe9696f84e75730d9da916f916048d401"
readonly IMAGE_ARCHIVE_NAME="qwen38-vllm-images-runtime-v3.tar"
readonly IMAGE_ARCHIVE_SHA256="8d85d9360eae5ca0fe78ea36cdaefb3b04abcacb91ffc827ef7cdfc9ecf3c96d"

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
readonly TURBOQUANT_PATCH_DIFF_SHA256="a9721067f1a7ee9497a4bd51e47e3a474561189e881b4704bfc4beac8ea48380"
readonly TOOL_SCHEMA_PATCH_DIFF_SHA256="eb5141db2aa702c9cc7dbcf1c8116e4dff37e832d1e5ce6a0ded3f38d66f4510"
readonly AGENT_DEFAULTS_PATCH_DIFF_SHA256="e1a31e2408603ebc174fc59e6a1e1e6dbfc5abf1d92221c8d976aa361f7d2423"
readonly PHASE_BUDGET_PATCH_DIFF_SHA256="f20d7dff41931248272842ed2c7a163c6f013e405ccf35733c40ff131a2fc503"
readonly TURBOQUANT_PATCHED_FILE_SHA256="0aa02c874d33c1113a49cb1aab49cfdc53e6a0d77fdc9ba91f7f89e6bddc0367"
readonly TOOL_SCHEMA_PATCHED_FILE_SHA256="7b18b3b6f5f230b73726a9a8ede72b8599aaeffed1e45142e116198a4633ba54"
readonly MODEL_CONFIG_PATCHED_FILE_SHA256="6a0b5fdcb292fef440ee59321b7db437dae2cd5fd80eb2372fa3647fb163a3cf"
readonly ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256="d7808b7aafdac29f3bbf5e8787dad83f913bba9b8626d52c2e6a92fb8ef87207"
readonly ANTHROPIC_SERVING_PATCHED_FILE_SHA256="b283dc7f47fe1cbb57f5a9cc4000a27ff5106e94d5caf152e5f7142e8a056e19"
readonly CHAT_PROTOCOL_PATCHED_FILE_SHA256="c96dccf12a0a931929bbe8da86776d386dbc7db548cb2f9ab209136f911da0f2"
readonly SAMPLING_PARAMS_PATCHED_FILE_SHA256="cbd49b4d7a8b84f7cc2dfb43ea13381337d5408d4eaadcf8d1b91082c269a06b"
readonly SCHED_UTILS_PATCHED_FILE_SHA256="bf00f90553b05358a2671466eabe3c3f2caee6b64a6ad64ad5544f7ffc997aa2"
readonly INPUT_PROCESSOR_PATCHED_FILE_SHA256="2b9e64486ce316fb4bc2293f18b1f005ae2e4b9adc60e39331a5add342b4b018"
readonly REQUEST_PATCHED_FILE_SHA256="9894e1e7d12850796c04f2f17116f7cea58daaa78ec9e24294df136c37b41e60"
readonly AGENT_CHAT_TEMPLATE_SHA256="07d9cf1a50bc702b27832586af016188d4cb5787e9a88847a5611237f722343e"
readonly PHASE_BUDGET_UNIT_SHA256="428b8a26b6fdf9612cc6cca43c4a0aa1f952f7157e448348b73130cac2e1543d"
readonly TURBOQUANT_UPSTREAM_FILE_SHA256="48994be137f3d25d4ee4f79ba2b89b0a6c3d988085079ffea1d241a34c2c755f"
readonly TOOL_SCHEMA_UPSTREAM_FILE_SHA256="015b989c567c6794e6dbbba72af88694470421adab13775c95b50efe9eedd2b7"
readonly MODEL_CONFIG_UPSTREAM_FILE_SHA256="17c687232886184f0390f38fc1c2c8ae078eaf24ebd1960a6b0c6a0669a35a98"
readonly ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256="a159978048c4cc8a409ca74b638618c5ccd2b1ffbc6a922702bc09b2918f9dc4"
readonly ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256="cc99303714b88b7138ff5411cc367a66f98bf851c55f63bff226f562f5b528ef"
readonly CHAT_PROTOCOL_UPSTREAM_FILE_SHA256="cb756e3d18e9061a2b306f305e10bd71d43f01ad1b236d0e9cbbb8756cd504dc"
readonly SAMPLING_PARAMS_UPSTREAM_FILE_SHA256="a29d80a2dc533c9a560f96acc3538fffcd00bf0412a6a7aa105d46553afb8359"
readonly SCHED_UTILS_UPSTREAM_FILE_SHA256="85e82eae555a03497ad2ac1540ed562a6c36fc26185aa6233725c914816aa1b3"
readonly INPUT_PROCESSOR_UPSTREAM_FILE_SHA256="f9a7946a16acc2374ff2bdfc22f212cb43461d9ef4d99c5e19a536339f11212f"
readonly REQUEST_UPSTREAM_FILE_SHA256="0287844f70eeaeb077d714e833a4b449a15e045a6516f8530182e357a5bec82f"
readonly SOURCE_DATE_EPOCH="1786751423"
readonly RUNTIME_DOCKERFILE_SHA256="a8bfe7e14c537958d54f014c3056c7c50a453dfd33a0c632af1b3e8beb42905f"
readonly DOCKERIGNORE_SHA256="32cf68ab7d096293d4b90179e906fbf67f1949557dc3f626cab60c43fa25cc56"

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
readonly CACHE_VOLUME="qwen38-vllm-cache-single-loopback-agent-v3"
readonly STARTUP_TIMEOUT_SECONDS="600"

readonly EXPECTED_RUNTIME_REPORT=$'python=3.12.3\nvllm=0.27.2rc1.dev106+g9df9b0b0a\ntorch=2.13.0+cu130\ntransformers=5.15.0\ntokenizers=0.22.2\nsafetensors=0.8.0\ncompressed-tensors=0.17.0\nflashinfer-python=0.6.16.post3\ntriton=3.7.1\nnumpy=2.3.5\nfastapi=0.136.3\nuvicorn=0.52.3\ntorch_cuda=13.0\ncuda_capability=12.0'

RUNTIME_ENV=(
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "DO_NOT_TRACK=1"
  "VLLM_NO_USAGE_STATS=1"
  "VLLM_DEBUG_WORKSPACE=1"
  "VLLM_ENFORCE_STRICT_TOOL_CALLING=1"
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
  --chat-template /opt/qwen38/chat_template.jinja
  --chat-template-content-format openai
  --generation-config /model
  --override-generation-config
  '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"thinking_token_budget":262144,"final_response_token_budget":131072}'
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
  '{"enable_thinking":true,"preserve_thinking":false,"reasoning_effort":"xhigh"}'
)
readonly -a VLLM_ARGS
