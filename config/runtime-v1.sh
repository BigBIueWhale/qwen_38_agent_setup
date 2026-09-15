#!/usr/bin/env bash
# Single source of truth for the only supported serving profile.

readonly PROFILE_VERSION="socket-isolated-nonroot-vision-k8v4-agent-v21"
readonly IMAGE_PROFILE_VERSION="socket-isolated-nonroot-vision-k8v4-agent-v23"
readonly CONTAINER_NAME="qwen38-agent-native"
readonly CONTAINER_LABEL="Qwen_best_model_ever"
readonly IMAGE_TAG="qwen38-vllm:qwen38-27b-nvfp4-k8v4-runtime-v23"
# AWAITING ADOPTION for the v23 image. ./scripts/build-vllm.sh builds it and
# refuses at this check, reporting the ID it produced; adopt that ID here.
# The refusal is what keeps the stale value from being deployable.
readonly EXPECTED_IMAGE_ID="sha256:695780692e2d9ea863f7081520e02ade0997ee7486a78e05968d2d62efe4b839"
readonly RELAY_IMAGE_TAG="qwen38-fixed-relay:1.0.0"
readonly EXPECTED_RELAY_IMAGE_ID="sha256:0e1c8be9644e7a5e09b1fbdf697be22c11b6c106b9cd167d3f046f78f9aa3657"
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
readonly IMAGE_ARCHIVE_NAME="qwen38-vllm-images-runtime-v23.tar"
# AWAITING ADOPTION, like EXPECTED_IMAGE_ID: the restore path verifies this
# hash before docker load, so a stale value fails on the bytes.
readonly IMAGE_ARCHIVE_SHA256="3a0a9e2aa84df4adf3993aa47d922bfc925850e72f7049347722a5406f7cbc31"

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
readonly AGENT_DEFAULTS_PATCH_DIFF_SHA256="c3a7315c32f8b117066e851b123b7a1b446c2fcd4c8a1f2616774c8f9d668509"
readonly PHASE_BUDGET_PATCH_DIFF_SHA256="f20d7dff41931248272842ed2c7a163c6f013e405ccf35733c40ff131a2fc503"
readonly IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256="d231c6e2e7040c4cd4b38432cb8c794805afddbf2c6e4f7ff6febb78e3fd9f48"
readonly ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256="b4c3327ca4e513b9a58edc3e9aca978d324a27032511f9868d5f941411941bcf"
readonly TOOL_TRUNCATION_PATCH_DIFF_SHA256="1a220f6db9b40967d867b3cfb1a92d95d907ca059718ffe61772b4cb4409f551"
readonly VISION_RUNTIME_PATCH_DIFF_SHA256="f92603724861da5b5a364f43e57d3f95ef43a9dded8ae645278373850db3140f"
readonly NUMERICAL_AUDITS_PATCH_DIFF_SHA256="a73aa2f2ae3f82010eb2bafcdf663c2fe14854c30165dbc4d8457725bc3b6632"
readonly TURBOQUANT_GUARDS_PATCH_DIFF_SHA256="7282d1d4d7a17b40ab8626c82f478bbb938c548451b7793df8233562a9e24c7c"
readonly KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256="1857071c38d081bb95e3cca12153cebce096649084950b99229104fdae029ca6"
readonly SHARED_PREFIX_CACHE_PATCH_DIFF_SHA256="736183bab22bb200053d38990ef0a51d7721a711542f241bde07f723aa2ce892"
readonly EXACT_REASONING_USAGE_PATCH_DIFF_SHA256="c6a880c0a15056792286f74bf32a4e554f70de05a82615522086ef4ca1cf2db3"
readonly SOURCE_PATCH_MANIFEST_SHA256="b9560a7c9f86648b0b6a58b4f02768937e8239740723fec96b01307ecc738d8a"
# Cardinality of config/deployment-inputs.sha256. The hash manifest alone
# proves the listed bytes but cannot see a quietly grown or shrunk allowlist,
# so the reviewed file count is pinned as well. It is declared exactly once,
# here: this value used to live independently in the generator and in the
# runtime validator, and the two copies drifted the first time a build input
# was added — the validator then refused a correct manifest. Every consumer
# (build-vllm.sh, runtime-common.sh, generate-deployment-input-manifest.sh)
# reads this declaration.
readonly DEPLOYMENT_INPUT_FILE_COUNT="94"
readonly TURBOQUANT_PATCHED_FILE_SHA256="ccda36577e4fb0052f370169dce4b649bad890b8b440a82e584acd3dd92a6d86"
readonly TURBOQUANT_STORE_PATCHED_FILE_SHA256="298645bff68c6adab58261862602b86e7e714c3552a9fd89102d9ccd2b83e9f7"
readonly TURBOQUANT_DECODE_PATCHED_FILE_SHA256="dab8b65ab7ddd6582de16e1fc7b1360ab0061b4a2a2b114f5d87ea0532fd726f"
readonly TOOL_SCHEMA_PATCHED_FILE_SHA256="b6ddd5a890f31922b2f23cc7b84fd39b42783f65c8db121ba3d402b1e3266288"
readonly MODEL_CONFIG_PATCHED_FILE_SHA256="6a0b5fdcb292fef440ee59321b7db437dae2cd5fd80eb2372fa3647fb163a3cf"
readonly ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256="4f648cfd2ddfb8fab0f611de597b2943e0041de679887ed9bb0292234bdf96cd"
readonly ANTHROPIC_SERVING_PATCHED_FILE_SHA256="01b02c09a64be7e0de8921694c1537a4972d37c4e6c861716df461c2b0debc5c"
readonly CHAT_PROTOCOL_PATCHED_FILE_SHA256="be4c43d9d8fdce12021e771ddef1992962cd3c913096ac1e789688b5b4265cfa"
readonly SAMPLING_PARAMS_PATCHED_FILE_SHA256="cbd49b4d7a8b84f7cc2dfb43ea13381337d5408d4eaadcf8d1b91082c269a06b"
readonly SCHED_UTILS_PATCHED_FILE_SHA256="bf00f90553b05358a2671466eabe3c3f2caee6b64a6ad64ad5544f7ffc997aa2"
readonly INPUT_PROCESSOR_PATCHED_FILE_SHA256="27944f76eb87665136d73ce3b9330abd4af02fb85e62b03bb334c7a219e3bbbe"
readonly REQUEST_PATCHED_FILE_SHA256="6281dcb0f3562cf6cc365e8fa43b1fd8d4fe06e136900fd49d2cbe718cbd0839"
readonly QWEN3_PARSER_PATCHED_FILE_SHA256="2c0d5e5bec9e3b504894d278eeaeded0f282e627260f1f35689869dd1fdf9bb0"
readonly STRUCTURED_OUTPUT_PATCHED_FILE_SHA256="f458a20495496d1bade5785addc50b6d655a81fd9d655912b703c4ed2e04314b"
readonly ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256="8d7d742f6e7d9f2fa9bbe1e57d6adbd4f2753075a083c7a7f93d39fc5ab474b6"
readonly CHAT_SERVING_PATCHED_FILE_SHA256="6d7e623168292c83e8a2eebd97ee124dc2ea03a4de5a7c36dc21d461ec623337"
readonly RESPONSES_CONTEXT_PATCHED_FILE_SHA256="45aabb486f12047609dac95a8f05bd48ce653e9c139cb1195f44c2df3b114423"
readonly RESPONSES_PROTOCOL_PATCHED_FILE_SHA256="bc0f16f37ee45ec9a5329c3b9bf51991c55ef89454f13b890128dd9f8e134983"
readonly RESPONSES_SERVING_PATCHED_FILE_SHA256="2eb659b4b126d551ea60516d499e79de420162da90c9cc7dafd220b1f52646cd"
readonly RESPONSES_STREAMING_PATCHED_FILE_SHA256="1d39608c0ddfb5466661fbe42d44f8c8b3584e9eeab36f3093c41734183efeba"
readonly RESPONSES_UTILS_PATCHED_FILE_SHA256="8d266a6a9a0f2d3c2d748e2bb4e8e69b2cdfb4a0488224e2a28d356cdf046b79"
readonly PARSER_ENGINE_PATCHED_FILE_SHA256="163224e7847cbd29bce9e291adfc4f307aace7109f52fbfd671a1b645cb9542a"
readonly KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256="0cd50f3deea7c8e91de072dde5d55f96edb6c0b35525c2d144ade7ae4e1b8a91"
readonly WORKSPACE_PATCHED_FILE_SHA256="b859dfdc5676f90a0b00718e34adcd0a02d266be1543ca146eebb724a9235c00"
readonly GPU_MODEL_RUNNER_PATCHED_FILE_SHA256="a7bed200b304fdc17320a30178ded7669d4677e787947470b872ef0ec14b6c8b"
readonly API_UTILS_PATCHED_FILE_SHA256="5c6fbd5ff02c042d6f96bfbe7f4d784f97dbfa8029bc09e059c66c16a807f74b"
readonly ENVS_PATCHED_FILE_SHA256="44dcae7ec3cf943de5c2e11125adf7e75676b12e627ed02c213e8dd38049f371"
readonly CHAT_UTILS_PATCHED_FILE_SHA256="9d939e56e812a583becfa18b5421488adc2087fba9922bd4e69dbccb012ca89a"
readonly MEDIA_CONNECTOR_PATCHED_FILE_SHA256="8b3998c4427fac24e5b92ac0b7f85950c13c43d9b43f17d4b464a9776e1bfaa5"
readonly IMAGE_MEDIA_PATCHED_FILE_SHA256="0ad95048460398831c58ace5c4f1d400eb127c4c2d2afb5b3df8cafb8c66f85f"
readonly RENDER_PARAMS_PATCHED_FILE_SHA256="2ba9da75d73c77333bb3e66bf5fff7e4afe6af59e681dffb7f74c306320a7381"
readonly QWEN3_VL_MODEL_PATCHED_FILE_SHA256="e271b7bbda10dc047d36b96fdbe9a7fd1806f1390bf7bb0aa3d3948f7f037cfa"
readonly CACHE_CONFIG_PATCHED_FILE_SHA256="82ab839cacb2e30f62f485c9e3ea32440fbf27beef00d1c60220f9776eb1ef43"
readonly VLLM_CONFIG_PATCHED_FILE_SHA256="30f612691ee2a5a1511484fbcece4bd89ade72c4309e786771826cf12fad38df"
readonly ARG_UTILS_PATCHED_FILE_SHA256="88582e97c98ffcd16416e48eeea3db415cab1f33673c7ff8c1613fa83aad1eac"
readonly LLM_ENTRYPOINT_PATCHED_FILE_SHA256="79f9bb1212884746964a347f7e4b39087b5ac084b1d72821a12efd2fb85bcb03"
readonly KV_CACHE_UTILS_PATCHED_FILE_SHA256="83cfaccc6607e8b850484aab0373bbfb326c072d606952dd42f21e41192dc8e7"
readonly GPU_WORKER_PATCHED_FILE_SHA256="e40be2cf5a83b8d69a2d4e620486838e3ae970226f265805c558541d95e54d9b"
readonly STARTUP_PLAN_PATCHED_FILE_SHA256="2f4f50c34201390e50e10b578bc4cd964a4f5729334225fc30d815bb704aa81f"
readonly KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256="50daea7891442fa779743796c343fe0a09bdcd0266910bf83b35509e533c3b89"
readonly KV_OFFLOAD_BASE_PATCHED_FILE_SHA256="e8fff9428338aa2c0d86c2ae1bdbb35e1e2338620e121656ac4f6da0fbd6779a"
readonly KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256="02bb64e4052092229e372002acdedb5206d3709698e88ff4c5e33b26da836e93"
readonly KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256="c6761a1151887f4fb4c3223de0e57f29e523a9568a4652c646c2fbb5f8ccecbf"
readonly KV_TIERING_SPEC_PATCHED_FILE_SHA256="a78615eeb2befe97739461b1db84b4fa79ebd3690fc63a807e2bed1ef2dc12f8"
readonly KV_TIERING_MANAGER_PATCHED_FILE_SHA256="c1119810cd34feb028e739f35ccce8fbf1793997e13766fa1d595fd8c9efaae4"
readonly OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256="328033f5240090ed684eeb4a96e67658b6839be9ccb6886483569c7bc2702c21"
readonly OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256="3afc05d7389a2590ef012279492c9526d5f0afe54b7909c9e5f8e8fd1857cc53"
readonly GENERATE_API_ROUTER_PATCHED_FILE_SHA256="dffeda2c3ccc6cfe3d4945720a7378bab34e7c7c9959d020a9643675895a3ffd"
readonly CLI_ARGS_PATCHED_FILE_SHA256="2c74b481652e1b7154df7836a98eb3ef1377092dc8ac4ae02095160907b5e36e"
readonly COMPLETION_PROTOCOL_PATCHED_FILE_SHA256="5b70feb0a6a59b6c763d64f5c8ab5fbc99750ff98cc055f4db70958b7bfc98ac"
readonly TITOTO_PROTOCOL_PATCHED_FILE_SHA256="2f668f2796cfc767777771508f2ad439bf96ba9d560f288212e3d675d99e2a7d"
readonly TITOTO_SERVING_PATCHED_FILE_SHA256="c055a75cd9148521bdf921110dcd54d7fa6fad17d2cfd552c23e775946beff7a"
readonly ABSTRACT_PARSER_PATCHED_FILE_SHA256="c3ab24e70dcabf75cd8cb662e56bd369dfcb6a9e66814e1e35f2840bc3920c4d"
readonly PARSER_ADAPTERS_PATCHED_FILE_SHA256="cda9c48f5b64c60224961bd75c8a5667caaa1494da581f5bb7ab10cec07ecd8b"
readonly ENGINE_PROTOCOL_PATCHED_FILE_SHA256="86001520ac9ec6e51d3dce8f0617461d497a75e83b2c2c7bb51e7459c8610dd0"
# Derived, not edited: scripts/derive-chat-template.py reconstructs these
# bytes from the model's own template through named stages, and
# ./scripts/build-vllm.sh check refuses if it cannot reproduce them.
readonly AGENT_CHAT_TEMPLATE_SHA256="07f545cd8ed9232f2b24d79010fad187f92e5b25b532448eb9017c0f8b8c2088"
readonly PHASE_BUDGET_UNIT_SHA256="913266638d302de31cdeae1acfdc5a568a01513481a57e5a4e7e9cbe258a99df"
readonly VISION_WORKSPACE_UNIT_SHA256="34f6ef1c477794de5e8b349c2da1dd491607a5618498358aa8b86085336a3df8"
readonly VISION_CONTRACT_UNIT_SHA256="3fb65d1767a55ca6536a45e6c3d72f1d6b08232d135f79eb2c4e65f220528832"
readonly VISION_MLP_UNIT_SHA256="857ba547a099c6ba646210eb33dd7b159bf9a1972d1772ea396071c8d8e4f2e3"
readonly TURBOQUANT_K8V4_UNIT_SHA256="2121146ae781bb94bd4ae257fb6a26c40ef7f3b212e626845d0939756fe8a494"
readonly CHAT_TEMPLATE_RETENTION_UNIT_SHA256="2baf37580ecf709b3c9d615ea08e3a337738ab0ea20fb35439e01b669e25ba2d"
readonly QWEN38_CONTEXT_UNIT_SHA256="77696c508ea77ffa8e63eed616783b648656bd81612b7d763ebf4505fdd9f5b2"
readonly NVFP4_KERNEL_UNIT_SHA256="2fce56060c9589d46e50371c8de456a6b9a65b906d95d9e3e1079cc70f790302"
readonly REASONING_USAGE_UNIT_SHA256="09175ce6a490f0bdfee6203cff06d8c7022518965cf1c7a452643093c04ffcad"
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
readonly GENERATE_API_ROUTER_UPSTREAM_FILE_SHA256="e428a6de4c01659de90e3cd88aed179595f4e48ffdf02ece0a6649750ebc71ab"
readonly CLI_ARGS_UPSTREAM_FILE_SHA256="5b6c1c61bc9d25c703086b92a3e0040d21cc054305d9a0fbe69767dfd26af646"
readonly COMPLETION_PROTOCOL_UPSTREAM_FILE_SHA256="fce5b234ae7f7d4cddea55357d18a3cb79e979f4d249a96968146bf9ef3e25c0"
readonly TITOTO_PROTOCOL_UPSTREAM_FILE_SHA256="89cc87f17da8223c0467361d89daa0eb74d2647a896af88682a061c4e0c39e0d"
readonly TITOTO_SERVING_UPSTREAM_FILE_SHA256="9e840e76e30863769ca653eabfa89c1e0e17c998751cd29da5313192404419ce"
readonly POLICY_PKG_INIT_UPSTREAM_FILE_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
readonly POLICY_BASE_UPSTREAM_FILE_SHA256="a019b4b59ee4acacca12e6b4057ad27c80d99b2da7f7a72f1484e6a0b5bd2eda"
readonly POLICY_FACTORY_UPSTREAM_FILE_SHA256="3c8d4be50478408b434d65851bb5f67003dcab54748ae7f3eb8fc0439378e8c6"
readonly POLICY_LRU_UPSTREAM_FILE_SHA256="d9fa56860f8a9d34ba7c16b9061da9d74c6fdefa2e538778add4fb582e0febfe"
readonly POLICY_ARC_UPSTREAM_FILE_SHA256="ff12419f9cb4fb84c4029ff3346319d43d02365a8c74748c35fbbdb3066e91d6"
readonly ABSTRACT_PARSER_UPSTREAM_FILE_SHA256="e567186750002ed7d0f5c5efeaffc9b9cfbec18060bdc080420b24cade713e13"
readonly PARSER_ADAPTERS_UPSTREAM_FILE_SHA256="dc1c1317dbfb298e54b8d94ca0e66d2b0cb1e481c35cdcc60a815284bd8a6ef7"
readonly ENGINE_PROTOCOL_UPSTREAM_FILE_SHA256="1c11f63c48fb3a48fdcc60371cb8eff4f03ed28ea7fc226450379b20bf8aa319"
readonly TURBOQUANT_STORE_UPSTREAM_FILE_SHA256="6e6e2fe74a307d40f0be786ccbaea76d989e3c7b5985f3144a5218c61bf6d902"
readonly TURBOQUANT_DECODE_UPSTREAM_FILE_SHA256="8e52678136449e4bbca2195fbcbb87426c955a2b1b8422e7ab9511e45ee5f5c6"
readonly TURBOQUANT_GUARD_UNIT_SHA256="657189807e2966824c556a08eb78a9ed7891331f6de4f3b563cf6cfded15cf47"
readonly SOURCE_DATE_EPOCH="1786751423"
readonly RUNTIME_DOCKERFILE_SHA256="eb0f0baddef109dbe25f6a92f8a2b84ad9c18647892131411bb8f298646c9234"
readonly DOCKERIGNORE_SHA256="00b93440d4684980fc932594c0353e72bdb1d12b69165dd7898b28e03119f00d"

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
readonly CACHE_VOLUME="qwen38-vllm-cache-socket-isolated-nonroot-vision-agent-v21"
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
  # The declaration is authoritative over the pool size, exactly as the byte
  # flag it replaced was: profiling's conservative availability estimate
  # (utilization budget minus every observed resident) may prefer a smaller
  # pool — it preferred ~1.3 GiB less than this pool through months of
  # production — but cannot shrink or veto it. What CAN refuse the
  # declaration is physical capacity: the whole card minus every measured
  # resident. Here weights 21.3 GiB + activation peak 1.9 GiB + this
  # 6.4 GiB one-context pool total ~29.6 GiB on the 31.8 GiB card — inside
  # the card, beyond the 0.9-utilization paper budget, exactly as the
  # proven byte-flag profile ran. Utilization stays at the vLLM default;
  # it governs the startup free-memory requirement and the informational
  # estimate, not the declared pool.
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
  # pre-faulted and page-locked — hard, unswappable host memory). Shared
  # chunks carry each acquiring agent's references. Only an ID with no cached
  # blocks may fork a shared prefix; an existing ID matches its own cache.
  # Pressure releases complete agent contexts while preserving the shared
  # references and complete working sets of surviving contexts.
  --kv-transfer-config
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_kv_cache_users":1}}'
  --enable-prefix-caching
  --enable-chunked-prefill
  --attention-config.flash_attn_version=2
  --kernel-config.enable_flashinfer_autotune=False
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  # Usage is served exactly or not at all: completion_tokens_details counts
  # the generated ids before the reasoning-end id (the reviewed
  # exact-reasoning-usage stage), and prompt_tokens_details reports the
  # prefix-cached prompt tokens the scheduler actually reused. Without this
  # flag vLLM omits the cached count and a client can only invent a zero.
  --enable-prompt-tokens-details
  --default-chat-template-kwargs
  '{"enable_thinking":true,"reasoning_effort":"xhigh","add_vision_id":false}'
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

readonly ERROR_RESPONSE_UPSTREAM_FILE_SHA256="91e23742a98c6ab629ec1fb41d63e3537b4a618c2f1310a32ee79e384b31d047"

readonly ERROR_RESPONSE_PATCHED_FILE_SHA256="897ad7da6633bee28d379f70a84ac51f6889b63214fc33977ef67c929a69584a"

readonly EXCEPTION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256="20a933aab30e36fd72c03f07915625fceaf76651314a0686c6e23416c389b4f6"

readonly EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256="644dbe4bca975923d66f5a929b10f61efb182667e33cd04f09ab496891af2764"

readonly HTTP_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256="a4642e0b141e078cf64a6e4192cb7c0f61cc9917bc561534a79da44dc087f78d"

readonly HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256="41021774f2fb3555129c74b8768ce200973f5fbcd0a5dbf8a017863657368868"

readonly VALIDATION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256="015286b0a27faa08bea1cc3286afa66b922f24926356d316cb3cddb795c6704b"

readonly VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256="eabace8c08e9412607fcaaaed0180ceaa5e57c8c812121663b0c3becc1c37c2c"

readonly VLLM_ERROR_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256="049ed8a85212377dc8dadadeada6d7acc9078e40e9b574cd7a702341a45b691f"

readonly VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256="973d2f726ce22110eece27e8dbf77b4fc0f0b64d229a5468e286dcd29fae4709"

readonly ANTHROPIC_INPUTS_PATCH_DIFF_SHA256="c2063d509fc90929f7d6018796f753da6445f12a4b4b19181e377f772b923a49"

readonly PARSER_EVENTS_UPSTREAM_FILE_SHA256="493543e5832b721c67640c09a6ad664823423a98afe0bb53ba78518608047f0a"

readonly PARSER_EVENTS_PATCHED_FILE_SHA256="d0ed492bbe28c19b6ec0446770a21754bfa844a70888ef5706587b0bbea51405"

readonly PARSER_ENGINE_CONFIG_UPSTREAM_FILE_SHA256="f7350e0ca9124001684f1f874ee72bf6a34932d3e4b84cc84567bbccf2f3e4b9"

readonly PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256="f746ba34d2b7606037d4229529e97da019bed0abda83b5f62f98559db613a233"

readonly STREAMING_PARSER_ENGINE_UPSTREAM_FILE_SHA256="bd2060822cd587efaaf8a1e0c282dd95642ce97976098461617cdf3e8d9afc6d"

readonly STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256="387a342be77da9928001c697e2e46c2895cd3ecd96dca9109ed18c4578a3a027"

readonly TOKEN_ID_SCANNER_UPSTREAM_FILE_SHA256="c9db6d6a29865d65ba1adc523015488aad4d7dbef46b7539041e9efbe81bd034"

readonly TOKEN_ID_SCANNER_PATCHED_FILE_SHA256="476d20aa1bc0e340ada1310dfaa909d2dae8236d049bdb5258b1147a0c63e373"

readonly QWEN_LANGUAGE_PATCH_DIFF_SHA256="fe4e46cb7444c80646537da63ab1ac12c54e7eebb04c0735a4243d8b7e7943d2"

readonly TOOL_OUTPUT_PARSER_UNIT_SHA256="7d55309fcfae4fc321a8a969bf77e73152b4abacac290c3aa0d57649e19584ae"

readonly PNG_SOURCE_PATCH_DIFF_SHA256="b9091c5c227151ec00131a854d927a9405396244a9a59bd4d6e297dd67ea3306"

readonly KV_PHYSICAL_PATCH_DIFF_SHA256="21f8993033c78971d4f7a660fe9906e054ec658139e83fc37b7121f1d8d91289"

readonly TOOL_PARSER_ABSTRACT_UPSTREAM_FILE_SHA256="5826dee6676d2ffc88856ab498c6271296b17c6f743a96646a2fc49a9008d1d7"

readonly TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256="91f4f3184e7f0eb6bc9e76d9ce4d3063cff58e0de8afc409038139e48314d0c8"

readonly TOOL_CALL_FILTER_UPSTREAM_FILE_SHA256="8439c3798fefaecd689f499b48ed4e151fff104b4c92c992ce4455ded25cb186"

readonly SINGLE_CALL_PATCH_DIFF_SHA256="2ae587bdde25b974cd88c5c351162fe809ee92b11d22ac0037f38411c8467b8b"

readonly RESPONSES_HISTORY_PATCH_DIFF_SHA256="117c17c114d7e91e57045a8e80b6eeac283c1ab56bdaf135019ad3c372a42186"

readonly RESPONSES_IDENTITY_PATCH_DIFF_SHA256="eccec34b8dd211f444065ef60b6b8075161efc790b61db4640104e3747763478"

readonly OFFLOADING_CONNECTOR_PATCHED_FILE_SHA256="0c46f6fb9c8b25c626c3d95b1463176e95f8a700afb2219e3ce4d267547ff5bb"
readonly OFFLOADING_CONNECTOR_UPSTREAM_FILE_SHA256="b5ddf7c1c8c50f6183dcdc4247759865b88e2b1b415e605c7993f615534912e0"

readonly BLOCK_POOL_PATCHED_FILE_SHA256="fb1a35481812980aed5370ee54a684261673547a359fff763f13422ed3802140"
readonly BLOCK_POOL_UPSTREAM_FILE_SHA256="ddee56dccb2208411b3a035918e917ce8f56a9858471e9ca12b420d5d79bc69c"

readonly KV_CACHE_COORDINATOR_PATCHED_FILE_SHA256="fd0373ae5fb314eb1a2a8c064560835224459a0147c10ea238094915f22f52a5"
readonly KV_CACHE_COORDINATOR_UPSTREAM_FILE_SHA256="e88a023b2364907140520bc333f25e3477eb8c6a3fff2cb8eeabca8e6e1d25ef"

readonly KV_CACHE_MANAGER_PATCHED_FILE_SHA256="af5f1204087629e4467b294eadd982a3255e01581b42a58a0ec3758c404286c5"
readonly KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256="2d20c3d98845cfd8d88a2f66b8fc6402ea1fcfbee16879d2364a4a9e45b8fa47"

readonly PREFIX_CACHE_PATCHED_FILE_SHA256="74cdbe60273641df0df7e0158da777314fe74ca29ccb539f16340ec7a2687ef4"

readonly SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256="e4d62562736aed93d394ba43f82654b207dd6f926a707e0b015d105129fd2131"
readonly SINGLE_TYPE_KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256="dc6d2e64cc4cd8cbb3f4bd77d2d035038658d75b6e52f9ace0cf637a4897f71f"

readonly KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256="9144fd869fa0c6df122081be01f2d81fa70f2e407f187564aad874e679b832c4"
readonly KV_OFFLOAD_CPU_COMMON_UPSTREAM_FILE_SHA256="0b7952a73376f04eb89a974e443d2dd5ce27b1a1001d8ebb628bce755c913f67"

readonly SIMPLE_KV_OFFLOAD_MANAGER_PATCHED_FILE_SHA256="bb8358c21057634fae53c34346588b7dd854b66fe6b6085947aac25f9ebbaef5"
readonly SIMPLE_KV_OFFLOAD_MANAGER_UPSTREAM_FILE_SHA256="d17d29556e61b82f6a8b3da998b995622bc0aed8ae395249bf315d21998fef9a"

readonly SHARED_PREFIX_CACHE_UNIT_SHA256="7516f9518f91b711f9bc0f4a56966e228703a134e6c83a6eb6d03d6acb29e818"
