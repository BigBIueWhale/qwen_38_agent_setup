#!/usr/bin/env bash
# Single source of truth for the only supported serving profile.

readonly PROFILE_VERSION="socket-isolated-nonroot-vision-k8v4-agent-v15"
readonly IMAGE_PROFILE_VERSION="socket-isolated-nonroot-vision-k8v4-agent-v15"
readonly CONTAINER_NAME="qwen38-agent-native"
readonly CONTAINER_LABEL="Qwen_best_model_ever"
readonly IMAGE_TAG="qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v15"
readonly EXPECTED_IMAGE_ID="sha256:c7dc10bafdc1a44853b6e73fee487e1bd951995532e4326662283a29eebc7573"
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
readonly IMAGE_ARCHIVE_NAME="qwen38-vllm-images-runtime-v15.tar"
readonly IMAGE_ARCHIVE_SHA256="24ca0fcc3a92bad9092afbbbd1c705e0a1972f34219010804b786799c382d3c9"

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
readonly KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256="1857071c38d081bb95e3cca12153cebce096649084950b99229104fdae029ca6"
readonly KV_USERS_SCOPE_PATCH_DIFF_SHA256="e9905e065913ba338da6ec1d61d2f350f48ac30eaa22de05dd491d2767521e66"
readonly SOURCE_PATCH_MANIFEST_SHA256="7d975b00045e7f07fa50fdb35e591c35a43c9b38de74df0871ae56d90d5aca19"
readonly TURBOQUANT_PATCHED_FILE_SHA256="ccda36577e4fb0052f370169dce4b649bad890b8b440a82e584acd3dd92a6d86"
readonly TOOL_SCHEMA_PATCHED_FILE_SHA256="e88b5cd98ace7c76453552f5f08264e0be23d1a5bc9b9d15cc0f39ba75ec043e"
readonly MODEL_CONFIG_PATCHED_FILE_SHA256="6a0b5fdcb292fef440ee59321b7db437dae2cd5fd80eb2372fa3647fb163a3cf"
readonly ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256="aba739f2db3d34580fc558f91f9cbb712ed683012bb95767b625f1469ddaaac3"
readonly ANTHROPIC_SERVING_PATCHED_FILE_SHA256="5f6aea9c0fb658cac7ecded12e24bad63a27cc9c719a6dc1c4d0ca19104172d2"
readonly CHAT_PROTOCOL_PATCHED_FILE_SHA256="9367f449954034b5aa615a6d5ae7e7f0e08e2066eec594c7b7d07375edde69f0"
readonly SAMPLING_PARAMS_PATCHED_FILE_SHA256="cbd49b4d7a8b84f7cc2dfb43ea13381337d5408d4eaadcf8d1b91082c269a06b"
readonly SCHED_UTILS_PATCHED_FILE_SHA256="bf00f90553b05358a2671466eabe3c3f2caee6b64a6ad64ad5544f7ffc997aa2"
readonly INPUT_PROCESSOR_PATCHED_FILE_SHA256="3c51e8935a92d2841199ae03659c4dcfcdea061f2d45dd1e2714ee5f3d9a7d06"
readonly REQUEST_PATCHED_FILE_SHA256="51e064ae80165a16e699f4e1742abf71bbbe59107a5d57d130956ef443b411d0"
readonly QWEN3_PARSER_PATCHED_FILE_SHA256="e5c192fda3ceba5c1686a790fd29b4ba663abdc9cbb7cc292634f4c503fd28e4"
readonly STRUCTURED_OUTPUT_PATCHED_FILE_SHA256="f458a20495496d1bade5785addc50b6d655a81fd9d655912b703c4ed2e04314b"
readonly ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256="05d17ec4f7ef1f05bdf95b6ce0d04ab80d1f5f8d0bbc130a551ea0512897e56e"
readonly CHAT_SERVING_PATCHED_FILE_SHA256="da0922ec020e0d4cf0111af1bf00b534348ce13e6e19ac7ce50b3d6cbf58653a"
readonly RESPONSES_CONTEXT_PATCHED_FILE_SHA256="45aabb486f12047609dac95a8f05bd48ce653e9c139cb1195f44c2df3b114423"
readonly RESPONSES_PROTOCOL_PATCHED_FILE_SHA256="8d174a7f706fcbf8b7b70f3fca283e6eb8816e02aa87da34aeb27aed1e720090"
readonly RESPONSES_SERVING_PATCHED_FILE_SHA256="ee5f461f39c7a03fb4147d6f045c127b311785def7689e607002676de50f2f83"
readonly RESPONSES_STREAMING_PATCHED_FILE_SHA256="5400a68d6219ca3944edb8a6d077da5e0ad0c767c34759dec7d35463dd1090b2"
readonly RESPONSES_UTILS_PATCHED_FILE_SHA256="6c70148e6de4a9806f2e4e8fe3e02659780e86b6886601bb8a60b377235dc29d"
readonly PARSER_ENGINE_PATCHED_FILE_SHA256="9ffce8a3aac1d885cbbd4de269201ef32bdd6089b5f92d61f94eebf1130a5faf"
readonly KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256="0cd50f3deea7c8e91de072dde5d55f96edb6c0b35525c2d144ade7ae4e1b8a91"
readonly WORKSPACE_PATCHED_FILE_SHA256="b859dfdc5676f90a0b00718e34adcd0a02d266be1543ca146eebb724a9235c00"
readonly GPU_MODEL_RUNNER_PATCHED_FILE_SHA256="a7bed200b304fdc17320a30178ded7669d4677e787947470b872ef0ec14b6c8b"
readonly API_UTILS_PATCHED_FILE_SHA256="5c6fbd5ff02c042d6f96bfbe7f4d784f97dbfa8029bc09e059c66c16a807f74b"
readonly ENVS_PATCHED_FILE_SHA256="44dcae7ec3cf943de5c2e11125adf7e75676b12e627ed02c213e8dd38049f371"
readonly CHAT_UTILS_PATCHED_FILE_SHA256="ca23415158a124c1c53b21bee6e22ab0ee7b433f8c9389e208d32a52162fb947"
readonly MEDIA_CONNECTOR_PATCHED_FILE_SHA256="8b3998c4427fac24e5b92ac0b7f85950c13c43d9b43f17d4b464a9776e1bfaa5"
readonly IMAGE_MEDIA_PATCHED_FILE_SHA256="4ef1af2c5ede9651d2ae490934cf798cab512a56dd7800978cafcb8ab6e1b8f9"
readonly RENDER_PARAMS_PATCHED_FILE_SHA256="2ba9da75d73c77333bb3e66bf5fff7e4afe6af59e681dffb7f74c306320a7381"
readonly QWEN3_VL_MODEL_PATCHED_FILE_SHA256="e271b7bbda10dc047d36b96fdbe9a7fd1806f1390bf7bb0aa3d3948f7f037cfa"
readonly CACHE_CONFIG_PATCHED_FILE_SHA256="82ab839cacb2e30f62f485c9e3ea32440fbf27beef00d1c60220f9776eb1ef43"
readonly VLLM_CONFIG_PATCHED_FILE_SHA256="30f612691ee2a5a1511484fbcece4bd89ade72c4309e786771826cf12fad38df"
readonly ARG_UTILS_PATCHED_FILE_SHA256="88582e97c98ffcd16416e48eeea3db415cab1f33673c7ff8c1613fa83aad1eac"
readonly LLM_ENTRYPOINT_PATCHED_FILE_SHA256="79f9bb1212884746964a347f7e4b39087b5ac084b1d72821a12efd2fb85bcb03"
readonly KV_CACHE_UTILS_PATCHED_FILE_SHA256="9bb65b9c96b22fbef74897ba1c74747936744218a988b49f42033602013a243b"
readonly GPU_WORKER_PATCHED_FILE_SHA256="2867bd3bc8449b5805bdfe2f0d0e2c557b42a864ca3eb3c0dca548d577605806"
readonly STARTUP_PLAN_PATCHED_FILE_SHA256="2f4f50c34201390e50e10b578bc4cd964a4f5729334225fc30d815bb704aa81f"
readonly KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256="36187561239d7fc212f0376b35b110b6507af060c18f0fad9308b2a332c4f7be"
readonly KV_OFFLOAD_BASE_PATCHED_FILE_SHA256="bbaf124d4360e5627f303a39a972f2f547b2eb9dd5deb6b6ae41d4e1e557b816"
readonly KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256="4e658d9969daa2549f6efcb2b0ee37e2de4fe6c3bd6b94c08a987836aa48d843"
readonly KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256="4a72e675a10e0f2d353477f1e75d32cc418b4f580730d6e1506855c1e50c2a9a"
readonly KV_TIERING_SPEC_PATCHED_FILE_SHA256="483427e773767a1696559eb566b4899dc4fa0751c75c477126d57eb57c714dc0"
readonly KV_TIERING_MANAGER_PATCHED_FILE_SHA256="246ac66fc93751ec7b17316129367f0a7cf38156932edd25f50d606a41283df5"
readonly OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256="3e038d98736a4dfc5bd83508bc537fea386273b615e794cba7e56ef1ff95505c"
readonly OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256="b343f5dc4a071a16820289d99dadfdb72e9afd8915b05a4a4b380e2b7071d704"
readonly COMPLETION_PROTOCOL_PATCHED_FILE_SHA256="ab3ac9361b8299fd890b2c268ff9fafd423c3802656d628f115614c6934d01ac"
readonly COHERE_PROTOCOL_PATCHED_FILE_SHA256="b8bf3aa474074b17fae8b60c3fe4c02f4bb36c2cdc43ac20f6190d0a11cc4269"
readonly COHERE_SERVING_PATCHED_FILE_SHA256="dd0a61acd644ae52303d56ed0f482f8df67129c24674cc6cb06a02d8aac51adc"
readonly TITOTO_PROTOCOL_PATCHED_FILE_SHA256="e36dfe2f1220a2105cf9adec949aa37f6e85fb716bffe09a29d38123b246de85"
readonly TITOTO_SERVING_PATCHED_FILE_SHA256="9241f7a9e3c7c8440448ba3d575ba4e5becadcc95195f1abb6953cf4523fa8bd"
readonly AGENT_CHAT_TEMPLATE_SHA256="07d9cf1a50bc702b27832586af016188d4cb5787e9a88847a5611237f722343e"
readonly PHASE_BUDGET_UNIT_SHA256="913266638d302de31cdeae1acfdc5a568a01513481a57e5a4e7e9cbe258a99df"
readonly VISION_WORKSPACE_UNIT_SHA256="34f6ef1c477794de5e8b349c2da1dd491607a5618498358aa8b86085336a3df8"
readonly VISION_CONTRACT_UNIT_SHA256="3a35831a58641f360ebc8c2c961c44deac1024e0a3431363ab68f803d86b4740"
readonly VISION_MLP_UNIT_SHA256="857ba547a099c6ba646210eb33dd7b159bf9a1972d1772ea396071c8d8e4f2e3"
readonly TURBOQUANT_K8V4_UNIT_SHA256="2121146ae781bb94bd4ae257fb6a26c40ef7f3b212e626845d0939756fe8a494"
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
readonly KV_OFFLOAD_WORKER_UPSTREAM_FILE_SHA256="5105d0455682704e93c69e7f21554e0ec959225484af000184c1014456dd095a"
readonly WORKSPACE_UPSTREAM_FILE_SHA256="d0650393bc657064acc97fe2b227ebff8f85799a8f727a8c136098f1f79964df"
readonly GPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256="7d5888ea176f34441553a4a0262f137433369e7b86076503ea54fbaee52d5554"
readonly API_UTILS_UPSTREAM_FILE_SHA256="14ca06f57d110b05f561812f84115bd7c380ad18e0fd0d0b2acaaab0e21fdd74"
readonly ENVS_UPSTREAM_FILE_SHA256="6bca0f24e5e9eec31b374c17cd9a7dac8cc548c95a9846826c08c62d4e04189f"
readonly CHAT_UTILS_UPSTREAM_FILE_SHA256="e77285d290ec7ad0fd8aaff8bcccfa9be3e77ff7fd355a1fb4b77bae0e8686e1"
readonly MEDIA_CONNECTOR_UPSTREAM_FILE_SHA256="b368a50c80d8fe01cf150cea02f39477d9c0ff76bfaea1a2b72883d8081a7326"
readonly IMAGE_MEDIA_UPSTREAM_FILE_SHA256="2605d8ca98c29fa9f2b208358f1049e2c60948b7494fcb08c4b334fe8acb1a38"
readonly RENDER_PARAMS_UPSTREAM_FILE_SHA256="b9667d21614cc474e881124b44762db6a1d43efe21b9ec0350fc5c80229bf67d"
readonly QWEN3_VL_MODEL_UPSTREAM_FILE_SHA256="b7ae6775e74cbcdb6e62d7fca9284e848f1f653caf53912f4dec64dc16ab96e3"
readonly CACHE_CONFIG_UPSTREAM_FILE_SHA256="8790351601be188ed1cc7ad9de2af1978b238269da4724873af1185c91165ef6"
readonly VLLM_CONFIG_UPSTREAM_FILE_SHA256="9048c652dd028757b972e36cfd46e1b3ce8db072fa1a5141c5899a0ed8bff0e1"
readonly ARG_UTILS_UPSTREAM_FILE_SHA256="75636c2a7903738f0e8954394fcedfced318d30a41598a8aca533be2ea2c38c4"
readonly LLM_ENTRYPOINT_UPSTREAM_FILE_SHA256="52de4ac99489e004ef6c61d0bedc84aa96020dd58b8bd1ae500814b548b2b83e"
readonly KV_CACHE_UTILS_UPSTREAM_FILE_SHA256="088f2201bee86fade694e78141b6e99a5cd0cdd23c5c7ab3526dd119f76e4aec"
readonly GPU_WORKER_UPSTREAM_FILE_SHA256="7ed4d59ee05cfefcdf16ffc901767e3c8a51d0fb0da4309cbd7b389bca96b7d2"
readonly STARTUP_PLAN_UPSTREAM_FILE_SHA256="84bcbb4f7a9fd7c8d10c9c895aadd9673d44c767b9132368d11eb5e017afe86b"
readonly KV_OFFLOAD_CONFIG_UPSTREAM_FILE_SHA256="92fbcc19d9e863c80676f03540c4a4b68b3943979b80123cc219ff114a92f955"
readonly KV_OFFLOAD_BASE_UPSTREAM_FILE_SHA256="8c999cc328d61e8cbaa0bc27b9f2488eb5323218807717ded69e9f6ecdfe7ef4"
readonly KV_OFFLOAD_CPU_SPEC_UPSTREAM_FILE_SHA256="f7e3fda4ea318c87aa740cf02b466db0798d3a9c2b5660e899ed095e087bd222"
readonly KV_OFFLOAD_CPU_MANAGER_UPSTREAM_FILE_SHA256="64b7ade9508fc4d1af5d7e67f030c05b3fe4b5dca0bcb6e91469ca0a127906c2"
readonly KV_TIERING_SPEC_UPSTREAM_FILE_SHA256="ff882d9e406e084d845cdc476771ace68a55843efe7735d74525cdf45b13cb77"
readonly KV_TIERING_MANAGER_UPSTREAM_FILE_SHA256="ebf34d67e83071b88be0e955c399f524c026965d8d41a0e3ffce68fd8ebee90c"
readonly OFFLOAD_CONNECTOR_CONFIG_UPSTREAM_FILE_SHA256="d400d0b0fadc06f2ad60a1356a6fee730a187dbcc4e48656da523de813419ec9"
readonly OFFLOAD_CONNECTOR_SCHEDULER_UPSTREAM_FILE_SHA256="616e7fd4cb0d09064cbc4d5735f607b37964c6be3b81e26de00d5913e0a9a3e3"
readonly COMPLETION_PROTOCOL_UPSTREAM_FILE_SHA256="fce5b234ae7f7d4cddea55357d18a3cb79e979f4d249a96968146bf9ef3e25c0"
readonly COHERE_PROTOCOL_UPSTREAM_FILE_SHA256="ebddd59bb5fc982552b104fd0e8ea16a4bd6e6ec444773ca77080f8c41f79ffb"
readonly COHERE_SERVING_UPSTREAM_FILE_SHA256="6dfd722db056b9a701145f233b5c9cf7d829cd54e97c86aabfe211b8538ebc2d"
readonly TITOTO_PROTOCOL_UPSTREAM_FILE_SHA256="89cc87f17da8223c0467361d89daa0eb74d2647a896af88682a061c4e0c39e0d"
readonly TITOTO_SERVING_UPSTREAM_FILE_SHA256="9e840e76e30863769ca653eabfa89c1e0e17c998751cd29da5313192404419ce"
readonly POLICY_PKG_INIT_UPSTREAM_FILE_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
readonly POLICY_BASE_UPSTREAM_FILE_SHA256="a019b4b59ee4acacca12e6b4057ad27c80d99b2da7f7a72f1484e6a0b5bd2eda"
readonly POLICY_FACTORY_UPSTREAM_FILE_SHA256="3c8d4be50478408b434d65851bb5f67003dcab54748ae7f3eb8fc0439378e8c6"
readonly POLICY_LRU_UPSTREAM_FILE_SHA256="d9fa56860f8a9d34ba7c16b9061da9d74c6fdefa2e538778add4fb582e0febfe"
readonly POLICY_ARC_UPSTREAM_FILE_SHA256="ff12419f9cb4fb84c4029ff3346319d43d02365a8c74748c35fbbdb3066e91d6"
readonly SOURCE_DATE_EPOCH="1786751423"
readonly RUNTIME_DOCKERFILE_SHA256="65bca0c0f9db13f9ca57d86674ff98acdd355762608d9b58eeb5b49ceed39d23"
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
readonly CACHE_VOLUME="qwen38-vllm-cache-socket-isolated-nonroot-vision-agent-v15"
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
  # Both KV tiers are declared as counts of resident 262,144-token user
  # contexts; vLLM derives the bytes post-engine-init from the KV cache spec,
  # where page sizes and the hybrid group structure actually exist. The byte
  # constants this replaces (6,925,634,765 B VRAM / 7,747,584,000 B DDR5) were
  # hand-measured on this one GPU and silently wrong anywhere else; the same
  # two declarations here are correct unchanged on a B200 or a TP-8 DGX.
  # Supplying the old byte or eviction knobs is a startup failure, not a
  # fallback.
  --kv-cache-users 1
  --cpu-offload-gb 0
  # Host-RAM KV offload. A foreground subagent's context competes for the
  # one-context GPU pool of the session that launched it, so at long main
  # contexts any subagent run evicts the main agent's blocks. On this hybrid
  # model that is disproportionately expensive: losing the few GDN state blocks
  # collapses the whole GPU prefix hit, forcing a full re-prefill rather than
  # just the evicted tail. Offloading keeps those blocks in DDR5 and restores
  # them by DMA instead.
  #
  # The copy is opaque bytes: the cache registers as int8 and whole pages move
  # by pointer, and K8V4 keeps its fp16 scale and min inline in each 388-byte
  # slot, so a token costs the same in DDR5 as in VRAM (24,832 B across the 16
  # full-attention layers). Nothing is dequantized.
  #
  # cpu_kv_cache_users:1 = one resident full-length context: every attention
  # chunk plus the trailing recurrent-state chunk a full-length re-entry
  # actually reads (~6.6 GB, inside the container's 8 GiB /dev/shm mmap,
  # pre-faulted and page-locked — hard, unswappable host memory). Eviction is
  # scope-aware, not a policy knob: each chunk is owned by the agent scope
  # that last used it, and a request's kv_scope_release drops a terminated
  # subagent's chunks as a unit. ARC existed to approximate exactly that
  # deadness from recency; the harness now states it, so there is nothing
  # left to select.
  --kv-transfer-config
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_kv_cache_users":1}}'
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
