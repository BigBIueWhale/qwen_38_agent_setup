#!/usr/bin/env bash
# Single source of truth for the only supported serving profile.

readonly PROFILE_VERSION="socket-isolated-nonroot-vision-k8v4-agent-v14"
readonly IMAGE_PROFILE_VERSION="socket-isolated-nonroot-vision-k8v4-agent-v14"
readonly CONTAINER_NAME="qwen38-agent-native"
readonly CONTAINER_LABEL="Qwen_best_model_ever"
readonly IMAGE_TAG="qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v14"
readonly EXPECTED_IMAGE_ID="sha256:587e8710c6630edd249f19b46837c12ebe5b5dcdc98486e215ac48a66644dc7f"
readonly RELAY_IMAGE_TAG="qwen38-fixed-relay:1.0.0"
readonly EXPECTED_RELAY_IMAGE_ID="sha256:5153a46bc03fa920b0d09000eca1848af255010bda99cc50e8a6110ebcd02690"
readonly RELAY_SOURCE_SHA256="051dc82af7b9b12e229f9a127183d051ef47a6d44f03d99346762e84bd69c815"
readonly RELAY_SANDBOX="landlock-net-v4+seccomp-socket-v2"
readonly AGENT_SERVICE_PROFILE="qwen38-agent-service-v3"
readonly MODEL_BRIDGE_NAME="qwen38-model-bridge"
readonly MODEL_INGRESS_NAME="qwen38-model-ingress"
readonly AGENT_SERVICE_RUNTIME_ROOT="/home/user/Desktop/agent_service/.runtime"
readonly MODEL_SOCKET_DIR="/home/user/Desktop/agent_service/.runtime/model-socket"
readonly RELAY_MEMORY="32m"
readonly RELAY_PIDS_LIMIT="32"
readonly BASE_IMAGE_TAG="qwen38-vllm:main-9df9b0b"
readonly EXPECTED_BASE_IMAGE_ID="sha256:fa4a002a88b7043a1a89966dea8a500fe9696f84e75730d9da916f916048d401"
readonly IMAGE_ARCHIVE_NAME="qwen38-vllm-images-runtime-v14.tar"
readonly IMAGE_ARCHIVE_SHA256="a80766d9560a419b9c051fc84d9beca1f1a3ac9ab508c99cf29d218b71bef43c"

readonly MODEL_DIR_NAME="Qwen3.8-27B-NVFP4-Corrected"
readonly MODEL_REPOSITORY="unsloth/Qwen3.8-27B-NVFP4"
readonly MODEL_REVISION="16b6615af3548b88e2d8e382457bc705b00479cf"
readonly OFFICIAL_MODEL_REPOSITORY="Qwen/Qwen3.8-27B"
readonly OFFICIAL_MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
readonly MODEL_CORRECTION="restore-161-offset-rmsnorms-from-official-bf16-v1"
readonly MODEL_SHA256="5fd70b38b3708e47adc1e9e9ab90f5d688ec01177d0718fdd16678696fdb0988"
readonly MTP_SHA256="1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe"
readonly MODEL_MANIFEST_NAME="model-corrected-16b6615a-norms-1d4bf0f2.sha256"
readonly MODEL_MANIFEST_SHA256="3a86177c30b97035d27ad0cf516fc4c2ddb83701c4de4fc6adcb23c7c2531bfc"
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
readonly VISION_RUNTIME_PATCH_DIFF_SHA256="f92603724861da5b5a364f43e57d3f95ef43a9dded8ae645278373850db3140f"
readonly NUMERICAL_AUDITS_PATCH_DIFF_SHA256="a73aa2f2ae3f82010eb2bafcdf663c2fe14854c30165dbc4d8457725bc3b6632"
readonly TURBOQUANT_GUARDS_PATCH_DIFF_SHA256="0ecf95ab8ee25a76d5412ce44aafafe13992b2cb373d6010acf5bc119dc8f47b"
readonly SOURCE_PATCH_MANIFEST_SHA256="539a777cb9e367a4922270f768214cbafbe4b0c9a46c5a90252df74a39c02226"
readonly TURBOQUANT_PATCHED_FILE_SHA256="59faab97fa3331028ac76dc63732e6eca239d0f7a83b79e18fd30c6d9b92c2df"
readonly TOOL_SCHEMA_PATCHED_FILE_SHA256="e88b5cd98ace7c76453552f5f08264e0be23d1a5bc9b9d15cc0f39ba75ec043e"
readonly MODEL_CONFIG_PATCHED_FILE_SHA256="6a0b5fdcb292fef440ee59321b7db437dae2cd5fd80eb2372fa3647fb163a3cf"
readonly ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256="d7808b7aafdac29f3bbf5e8787dad83f913bba9b8626d52c2e6a92fb8ef87207"
readonly ANTHROPIC_SERVING_PATCHED_FILE_SHA256="0e67a46639b5369fad8de21a31799b4fe2bfbbd93c5d1df02911f0f98a08a43c"
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
readonly WORKSPACE_PATCHED_FILE_SHA256="b859dfdc5676f90a0b00718e34adcd0a02d266be1543ca146eebb724a9235c00"
readonly GPU_MODEL_RUNNER_PATCHED_FILE_SHA256="a7bed200b304fdc17320a30178ded7669d4677e787947470b872ef0ec14b6c8b"
readonly API_UTILS_PATCHED_FILE_SHA256="5c6fbd5ff02c042d6f96bfbe7f4d784f97dbfa8029bc09e059c66c16a807f74b"
readonly ENVS_PATCHED_FILE_SHA256="b171841f774cf1265cc67d93ed3c7c3cef77bedfbcf01ffb346bbbb43853fe3a"
readonly CHAT_UTILS_PATCHED_FILE_SHA256="ca23415158a124c1c53b21bee6e22ab0ee7b433f8c9389e208d32a52162fb947"
readonly MEDIA_CONNECTOR_PATCHED_FILE_SHA256="8b3998c4427fac24e5b92ac0b7f85950c13c43d9b43f17d4b464a9776e1bfaa5"
readonly IMAGE_MEDIA_PATCHED_FILE_SHA256="4ef1af2c5ede9651d2ae490934cf798cab512a56dd7800978cafcb8ab6e1b8f9"
readonly RENDER_PARAMS_PATCHED_FILE_SHA256="2ba9da75d73c77333bb3e66bf5fff7e4afe6af59e681dffb7f74c306320a7381"
readonly QWEN3_VL_MODEL_PATCHED_FILE_SHA256="e271b7bbda10dc047d36b96fdbe9a7fd1806f1390bf7bb0aa3d3948f7f037cfa"
readonly AGENT_CHAT_TEMPLATE_SHA256="07d9cf1a50bc702b27832586af016188d4cb5787e9a88847a5611237f722343e"
readonly PHASE_BUDGET_UNIT_SHA256="913266638d302de31cdeae1acfdc5a568a01513481a57e5a4e7e9cbe258a99df"
readonly VISION_WORKSPACE_UNIT_SHA256="34f6ef1c477794de5e8b349c2da1dd491607a5618498358aa8b86085336a3df8"
readonly VISION_CONTRACT_UNIT_SHA256="3a35831a58641f360ebc8c2c961c44deac1024e0a3431363ab68f803d86b4740"
readonly VISION_MLP_UNIT_SHA256="857ba547a099c6ba646210eb33dd7b159bf9a1972d1772ea396071c8d8e4f2e3"
readonly TURBOQUANT_K8V4_UNIT_SHA256="721cdf880ac0236f8b829894458e56ec0c8e9bffb1bc200aef92d42f25d913ca"
readonly QWEN38_CONTEXT_UNIT_SHA256="77696c508ea77ffa8e63eed616783b648656bd81612b7d763ebf4505fdd9f5b2"
readonly NVFP4_KERNEL_UNIT_SHA256="2fce56060c9589d46e50371c8de456a6b9a65b906d95d9e3e1079cc70f790302"
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
readonly WORKSPACE_UPSTREAM_FILE_SHA256="d0650393bc657064acc97fe2b227ebff8f85799a8f727a8c136098f1f79964df"
readonly GPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256="7d5888ea176f34441553a4a0262f137433369e7b86076503ea54fbaee52d5554"
readonly API_UTILS_UPSTREAM_FILE_SHA256="14ca06f57d110b05f561812f84115bd7c380ad18e0fd0d0b2acaaab0e21fdd74"
readonly ENVS_UPSTREAM_FILE_SHA256="6bca0f24e5e9eec31b374c17cd9a7dac8cc548c95a9846826c08c62d4e04189f"
readonly CHAT_UTILS_UPSTREAM_FILE_SHA256="e77285d290ec7ad0fd8aaff8bcccfa9be3e77ff7fd355a1fb4b77bae0e8686e1"
readonly MEDIA_CONNECTOR_UPSTREAM_FILE_SHA256="b368a50c80d8fe01cf150cea02f39477d9c0ff76bfaea1a2b72883d8081a7326"
readonly IMAGE_MEDIA_UPSTREAM_FILE_SHA256="2605d8ca98c29fa9f2b208358f1049e2c60948b7494fcb08c4b334fe8acb1a38"
readonly RENDER_PARAMS_UPSTREAM_FILE_SHA256="b9667d21614cc474e881124b44762db6a1d43efe21b9ec0350fc5c80229bf67d"
readonly QWEN3_VL_MODEL_UPSTREAM_FILE_SHA256="b7ae6775e74cbcdb6e62d7fca9284e848f1f653caf53912f4dec64dc16ab96e3"
readonly SOURCE_DATE_EPOCH="1786751423"
readonly RUNTIME_DOCKERFILE_SHA256="17f72538ee71292e4cf0a2ce804e52a4d26413a034286590aef76008fbd4fcec"
readonly DOCKERIGNORE_SHA256="a15c81d0be5c474d9f0cd5e8b1d3f89b5eb7266ce60d45476069de9499f6b103"

# Functional host contract only. Exact host software versions, binary
# hashes, and GPU/driver identity are deliberately not pinned: they tie the
# deployment to one specific computer without making inference any more
# correct. The isolation features below and the GPU-memory calibration floor
# for the locked VRAM budget are the properties the profile actually
# depends on; everything inside the pinned images remains exact.
readonly EXPECTED_DOCKER_SECURITY_OPTIONS='["name=apparmor","name=seccomp,profile=builtin","name=cgroupns"]'
readonly EXPECTED_CONTAINER_APPARMOR_PROFILE="docker-default"
readonly MINIMUM_GPU_MEMORY_MIB="32607"

readonly LISTEN_HOST="127.0.0.1"
readonly LISTEN_PORT="8000"
readonly ENDPOINT="http://${LISTEN_HOST}:${LISTEN_PORT}"
readonly SERVED_MODEL="qwen3.8-27b-nvfp4-k8v4"
readonly MAX_MODEL_LEN="262144"
readonly CACHE_VOLUME="qwen38-vllm-cache-socket-isolated-nonroot-vision-agent-v14"
readonly TMP_TMPFS_OPTIONS="rw,nosuid,nodev,exec,size=2g,mode=1777"
readonly RUN_TMPFS_OPTIONS="rw,nosuid,nodev,noexec,size=64m,uid=2000,gid=0,mode=0700"
readonly STARTUP_TIMEOUT_SECONDS="600"

readonly EXPECTED_RUNTIME_REPORT=$'python=3.12.3\nvllm=0.27.2rc1.dev106+g9df9b0b0a\ntorch=2.13.0+cu130\ntransformers=5.15.0\ntokenizers=0.22.2\nsafetensors=0.8.0\ncompressed-tensors=0.17.0\nflashinfer-python=0.6.16.post3\ntriton=3.7.1\nnumpy=2.3.5\nfastapi=0.136.3\nuvicorn=0.52.3\ntorch_cuda=13.0\ncuda_capability=12.0'

RUNTIME_ENV=(
  "HOME=/home/vllm"
  "VLLM_CACHE_ROOT=/home/vllm/.cache/vllm"
  "XDG_CACHE_HOME=/home/vllm/.cache/vllm/xdg-cache"
  "XDG_CONFIG_HOME=/home/vllm/.cache/vllm/xdg-config"
  "CUDA_CACHE_PATH=/home/vllm/.cache/vllm/cuda"
  "HF_HOME=/home/vllm/.cache/vllm/huggingface"
  "TRITON_HOME=/home/vllm/.cache/vllm/triton"
  "TRITON_CACHE_DIR=/home/vllm/.cache/vllm/triton/cache"
  "TORCHINDUCTOR_CACHE_DIR=/home/vllm/.cache/vllm/torchinductor"
  "FLASHINFER_WORKSPACE_BASE=/home/vllm/.cache/vllm/flashinfer"
  "PYTHONDONTWRITEBYTECODE=1"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "DO_NOT_TRACK=1"
  "VLLM_NO_USAGE_STATS=1"
  "VLLM_DEBUG_WORKSPACE=1"
  "VLLM_ENFORCE_STRICT_TOOL_CALLING=1"
  "VLLM_QWEN38_STRICT_IMAGE_CONTRACT=1"
  "VLLM_QWEN38_VISION_HEADROOM_BYTES=671088640"
  "VLLM_MAX_IMAGE_PIXELS=16777216"
  "GLOO_SOCKET_IFNAME=lo"
  "NCCL_SOCKET_IFNAME=lo"
)
readonly -a RUNTIME_ENV

VLLM_ARGS=(
  /model
  --served-model-name "${SERVED_MODEL}"
  --host "${LISTEN_HOST}"
  --port "${LISTEN_PORT}"
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
  '{"enable_thinking":true,"preserve_thinking":false,"reasoning_effort":"xhigh","add_vision_id":false}'
  --limit-mm-per-prompt
  '{"image":{"count":15,"width":4096,"height":4096},"video":0}'
  --mm-processor-kwargs
  '{"size":{"longest_edge":16777216,"shortest_edge":65536}}'
  --mm-processor-device cpu
  --no-mm-device-do-normalize
  --mm-encoder-tp-mode weights
  --mm-processor-cache-gb 4
  --mm-processor-cache-type lru
  --mm-hasher-algorithm sha256
  --mm-tensor-ipc direct_rpc
  --no-skip-mm-profiling
)
readonly -a VLLM_ARGS
