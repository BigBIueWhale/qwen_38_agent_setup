#!/usr/bin/env bash
# Single source of truth for the only supported serving profile.

readonly PROFILE_VERSION="single-loopback-text-k8v4-agent-v6"
readonly CONTAINER_NAME="qwen38-agent-native"
readonly CONTAINER_LABEL="Qwen_best_model_ever"
readonly IMAGE_TAG="qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v6"
readonly EXPECTED_IMAGE_ID="sha256:3452967bf6d1dca98122042e6d1b4445a4d5addd5ed6947e84a94e003fcac884"
readonly BASE_IMAGE_TAG="qwen38-vllm:main-9df9b0b"
readonly EXPECTED_BASE_IMAGE_ID="sha256:fa4a002a88b7043a1a89966dea8a500fe9696f84e75730d9da916f916048d401"
readonly IMAGE_ARCHIVE_NAME="qwen38-vllm-images-runtime-v6.tar"
readonly IMAGE_ARCHIVE_SHA256="ad6a38a0658e3d57b8037ebd4b3dcfe91b3d9eae3733fc78ee15ae7c755b8d08"

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
readonly TOOL_SCHEMA_PATCH_DIFF_SHA256="4f75c793a9c2cdcfb2fd0768ba49a4e34748d3a37d8392b07d3592ca50939c07"
readonly AGENT_DEFAULTS_PATCH_DIFF_SHA256="6428d2cfa77f28e57e117999d0ec8fab5430856c985ba530e04885c2f5c420b7"
readonly PHASE_BUDGET_PATCH_DIFF_SHA256="f20d7dff41931248272842ed2c7a163c6f013e405ccf35733c40ff131a2fc503"
readonly IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256="d231c6e2e7040c4cd4b38432cb8c794805afddbf2c6e4f7ff6febb78e3fd9f48"
readonly ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256="030b64be104e6ef57a40f6bae740dfa9d4634a420c6c93a395f62bfb98d6d053"
readonly TOOL_TRUNCATION_PATCH_DIFF_SHA256="1a220f6db9b40967d867b3cfb1a92d95d907ca059718ffe61772b4cb4409f551"
readonly TURBOQUANT_PATCHED_FILE_SHA256="0aa02c874d33c1113a49cb1aab49cfdc53e6a0d77fdc9ba91f7f89e6bddc0367"
readonly TOOL_SCHEMA_PATCHED_FILE_SHA256="e88b5cd98ace7c76453552f5f08264e0be23d1a5bc9b9d15cc0f39ba75ec043e"
readonly MODEL_CONFIG_PATCHED_FILE_SHA256="6a0b5fdcb292fef440ee59321b7db437dae2cd5fd80eb2372fa3647fb163a3cf"
readonly ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256="d7808b7aafdac29f3bbf5e8787dad83f913bba9b8626d52c2e6a92fb8ef87207"
readonly ANTHROPIC_SERVING_PATCHED_FILE_SHA256="b283dc7f47fe1cbb57f5a9cc4000a27ff5106e94d5caf152e5f7142e8a056e19"
readonly CHAT_PROTOCOL_PATCHED_FILE_SHA256="0dae9a86a71ec3bbe1d55e55e7c8c7e2c14a6a8f5b1fa3f98a5223a63165e77b"
readonly SAMPLING_PARAMS_PATCHED_FILE_SHA256="cbd49b4d7a8b84f7cc2dfb43ea13381337d5408d4eaadcf8d1b91082c269a06b"
readonly SCHED_UTILS_PATCHED_FILE_SHA256="bf00f90553b05358a2671466eabe3c3f2caee6b64a6ad64ad5544f7ffc997aa2"
readonly INPUT_PROCESSOR_PATCHED_FILE_SHA256="2b9e64486ce316fb4bc2293f18b1f005ae2e4b9adc60e39331a5add342b4b018"
readonly REQUEST_PATCHED_FILE_SHA256="9894e1e7d12850796c04f2f17116f7cea58daaa78ec9e24294df136c37b41e60"
readonly QWEN3_PARSER_PATCHED_FILE_SHA256="e5c192fda3ceba5c1686a790fd29b4ba663abdc9cbb7cc292634f4c503fd28e4"
readonly STRUCTURED_OUTPUT_PATCHED_FILE_SHA256="f458a20495496d1bade5785addc50b6d655a81fd9d655912b703c4ed2e04314b"
readonly ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256="05d17ec4f7ef1f05bdf95b6ce0d04ab80d1f5f8d0bbc130a551ea0512897e56e"
readonly CHAT_SERVING_PATCHED_FILE_SHA256="da0922ec020e0d4cf0111af1bf00b534348ce13e6e19ac7ce50b3d6cbf58653a"
readonly RESPONSES_CONTEXT_PATCHED_FILE_SHA256="45aabb486f12047609dac95a8f05bd48ce653e9c139cb1195f44c2df3b114423"
readonly RESPONSES_PROTOCOL_PATCHED_FILE_SHA256="ae15671c1863efa07573a6f2282fb7c7e364c0af7a778e50151241b501a5468e"
readonly RESPONSES_SERVING_PATCHED_FILE_SHA256="ee5f461f39c7a03fb4147d6f045c127b311785def7689e607002676de50f2f83"
readonly RESPONSES_STREAMING_PATCHED_FILE_SHA256="5400a68d6219ca3944edb8a6d077da5e0ad0c767c34759dec7d35463dd1090b2"
readonly RESPONSES_UTILS_PATCHED_FILE_SHA256="6c70148e6de4a9806f2e4e8fe3e02659780e86b6886601bb8a60b377235dc29d"
readonly PARSER_ENGINE_PATCHED_FILE_SHA256="9ffce8a3aac1d885cbbd4de269201ef32bdd6089b5f92d61f94eebf1130a5faf"
readonly AGENT_CHAT_TEMPLATE_SHA256="07d9cf1a50bc702b27832586af016188d4cb5787e9a88847a5611237f722343e"
readonly PHASE_BUDGET_UNIT_SHA256="913266638d302de31cdeae1acfdc5a568a01513481a57e5a4e7e9cbe258a99df"
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
readonly QWEN3_PARSER_UPSTREAM_FILE_SHA256="8a7ee658322de7b736ea5b0f802d70dd07a124b5878b4f8ad2f99eca8e1d35fb"
readonly STRUCTURED_OUTPUT_UPSTREAM_FILE_SHA256="355f6f1193c15d5d6901a0f567e2e16005e3681f04f70079c6ba11e020b4d33a"
readonly ANTHROPIC_API_ROUTER_UPSTREAM_FILE_SHA256="0e5f655dd0ff66cfb5c53e5413ca90713eb5d5b16547e8ef01891845fb047e37"
readonly CHAT_SERVING_UPSTREAM_FILE_SHA256="a42294241a5a2f0cfb115dfa09ded8cb08647f9c6ad98d22c71d3fac3bae3520"
readonly RESPONSES_CONTEXT_UPSTREAM_FILE_SHA256="04f25b6fb1180c9e4e24045d77f8eed880ea3e26b01f070951a1ae6112ba129b"
readonly RESPONSES_PROTOCOL_UPSTREAM_FILE_SHA256="0d7335e0ea5b361f26dba13223c277b2af3eed0c2a86333a657b4ec04e4fd343"
readonly RESPONSES_SERVING_UPSTREAM_FILE_SHA256="ac586de722ddada5032c760da0dfb8faae281fbec4e5d324204cafdad00b5e30"
readonly RESPONSES_STREAMING_UPSTREAM_FILE_SHA256="cf1d8f5e0619148374ce10be15b1a9f7640016d810f1fe766c2dd451a918aa1f"
readonly RESPONSES_UTILS_UPSTREAM_FILE_SHA256="577100edd0951f7f2936d2b37b7b4ec9a03d85088b35e49de6c0e9633a59adc2"
readonly PARSER_ENGINE_UPSTREAM_FILE_SHA256="3ac89a7f22f0e4f0d3f6f2365d79f64da9da217969793f6a33a6db9cf5ef60ff"
readonly SOURCE_DATE_EPOCH="1786751423"
readonly RUNTIME_DOCKERFILE_SHA256="3ad07d29169e00e6b48441d21abbfee78c1d1d06261f9e92ec2808d1daa0ebaf"
readonly DOCKERIGNORE_SHA256="cacac15870fe9dee962785d6ab7d3289c73adf02d5ff05ace1be1c5e5e8b1ff1"

readonly EXPECTED_DOCKER_VERSION="29.7.2"
readonly EXPECTED_NVIDIA_CONTAINER_CLI_VERSION="1.19.1"
readonly EXPECTED_GPU_NAME="NVIDIA GeForce RTX 5090"
readonly EXPECTED_GPU_MEMORY_MIB="32607"
readonly EXPECTED_DRIVER_VERSION="595.71.05"
readonly EXPECTED_BASH_VERSION="5.2.21(1)-release"
readonly EXPECTED_GIT_VERSION_REPORT="git version 2.43.0"
readonly EXPECTED_SHA256SUM_VERSION_REPORT="sha256sum (GNU coreutils) 9.4"
readonly EXPECTED_SS_VERSION_REPORT="ss utility, iproute2-6.1.0"

readonly LISTEN_HOST="127.0.0.1"
readonly LISTEN_PORT="8000"
readonly ENDPOINT="http://${LISTEN_HOST}:${LISTEN_PORT}"
readonly SERVED_MODEL="qwen3.8-27b-nvfp4-k8v4"
readonly MAX_MODEL_LEN="262144"
readonly CACHE_VOLUME="qwen38-vllm-cache-single-loopback-agent-v6"
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
