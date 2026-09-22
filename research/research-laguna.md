# Laguna (Poolside): fact-checked research report

Date: 2026-09-22. Question: the team said "we'll probably switch to Laguna (the best one) instead of Ornith". What is Laguna, and does it fit the offline patched-vLLM stack (NVFP4 weights, TurboQuant K8V4 KV cache, Blackwell, 262,144-token native context, thinking always on, vision required, agentic coding, no PRC-published models)?

## How to read this report

- **FACT (direct)**: I read the primary artifact myself, as source code or a recipe file on GitHub.
- **FACT (mirror)**: a third-party copy of a primary artifact, such as a copy of an HF `config.json` or safetensors-index totals. The copy is checked against another source where possible.
- **FACT (snippet)**: text returned by the search engine from the primary page. **I could not open the page itself.** See "Access limits" below.
- **INFERENCE**: my own calculation or judgment.

**Access limits (important):** in this session the egress proxy returned HTTP 403 (EGRESS_BLOCKED) for `huggingface.co`, including `https://huggingface.co/api/models?search=laguna` and every raw `config.json`. It did the same for `poolside.ai`, `arxiv.org`, `venturebeat.com`, `marktechpost.com`, `docs.vllm.ai`/`recipes.vllm.ai`, Reddit and HN. I checked with `curl` and WebFetch. The reachable sources were GitHub (git, raw files, code search), PyPI and web search. As a result:
- the config values below come from verified GitHub copies of the HF `config.json` files;
- the model-card benchmark tables come from search snippets and card snapshots.

Section 7 lists the items to re-check on a machine that can reach HF.

---

## 0. Bottom line

1. **FACT: "Laguna" is Poolside's open-weights agentic-coding MoE family.** Poolside is a US-founded lab with a US/Paris footprint. Variants:
   - **XS.2 / XS 2.1**: 33B-A3B
   - **S 2.1**: 118B-A8B
   - **M.1**: 225B-A23B
   
   There is **no 397B Laguna**. No other LLM called "Laguna" exists in 2026. Every other hit was a car, a sailboat, a place or a hackathon adapter. Sources: [transformers doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/laguna.md), [vLLM recipes](https://github.com/vllm-project/recipes/tree/main/models/poolside), [Wikipedia snippet](https://en.wikipedia.org/wiki/Poolside_AI).
2. **"The best one" most plausibly means Laguna S 2.1 (118B-A8B, released 2026-07-21).**
   - Poolside's own HF collection calls it "Our most capable model to date" (FACT, snippet via [radar mirror of HF metadata](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1-nvfp4.json)).
   - It beats M.1 on every agentic-coding benchmark both models publish.
   - The largest model is still **M.1 (225B-A23B)**. It is older, uses global attention in all 70 layers and has a 256K context.
3. **Hard blocker: no Laguna model has vision.**
   - The architecture is `LagunaForCausalLM`, text-only, with no `vision_config` in any config.
   - The recipes list `tasks: [text]`.
   - The only image path is a community hackathon LoRA adapter on XS.2, `poolside-laguna-hackathon/laguna-vision`, described as "weakly grounded, 12/80 strict passes" (FACT, snippet: [HF](https://huggingface.co/poolside-laguna-hackathon/laguna-vision)).
   
   Your requirements list says vision is mandatory, so Laguna cannot replace Ornith 1:1.
4. **Benchmarks:** on the benchmarks Poolside publishes, Laguna S 2.1 is roughly in the **Ornith 1.5 35B-A3B tier, not the Ornith 1.5 397B tier**:

   | Benchmark | Laguna S 2.1 | Ornith 1.5 35B-A3B | Ornith 1.5 397B |
   |---|---|---|---|
   | Terminal-Bench 2.1 | 70.2 | 67.8 | 86.1 |
   | Toolathlon | 49.7 ("Toolathlon Verified") | 48.7 | 71.2 |

   Poolside publishes no SWE-bench Verified, GPQA, MCP-Atlas, BrowseComp, HLE or long-context numbers for S 2.1.
5. **KV cache:**
   - Laguna S 2.1: 12 global layers × 8 KV heads × 128 = **49,152 B/token BF16** and **18,816 B/token K8V4**. That is 1.6× Ornith 397B. S 2.1 also carries a small fixed sliding-window KV (72 MiB BF16) instead of a recurrent state.
   - Laguna M.1: **286,720 B/token BF16**, which is 9.3× Ornith 397B, and 26.8 GiB per 256K user even at K8V4.
6. **Policy:** Laguna is cleaner than Ornith on your "no PRC" rule:
   - Laguna comes from a non-PRC publisher and Poolside says it was trained from scratch.
   - Ornith comes from a non-PRC publisher, DeepReinforce in California, but builds on Qwen3.5, which is Alibaba (PRC). Your stated policy accepts that.

---

## 1. Identification

### 1a. What "Laguna" is

**FACT (direct):** the transformers docs describe it as "Laguna is Poolside's mixture-of-experts language model family". The files carry "Copyright 2026 Poolside and the HuggingFace Inc. team". Laguna was contributed to Transformers on **2026-04-28**. Sources: [laguna.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/laguna.md), [configuration_laguna.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/laguna/configuration_laguna.py).

**Publisher: Poolside (Poolside AI), HF org [`poolside`](https://huggingface.co/poolside).**
- FACT (snippet): Wikipedia describes it as "an American artificial intelligence company… founded in 2023 with headquarters in San Francisco", founded by Jason Warner (ex-CTO of GitHub) and Eiso Kant ([Wikipedia](https://en.wikipedia.org/wiki/Poolside_AI)).
- FACT (snippet): in 2023 it was reported to have relocated its HQ from SF to Paris ([Sifted](https://sifted.eu/articles/poolside-raises-126m-relocated-france-news), [Dealroom](https://app.dealroom.co/news/note/relocation-of-hq-from-sf-to-paris-for-poolside-ai)).
- FACT (snippet): VentureBeat calls it an "American AI startup" ([VB](https://venturebeat.com/technology/american-ai-startup-poolside-launches-free-high-performing-open-model-laguna-xs-2-for-local-agentic-coding)).
- **Country: US, with a France presence. Not PRC.**

**What the card says about the team (FACT, snippet):** "Poolside trains models from scratch: our own data, our own infrastructure, our own reinforcement learning" ([poolside.ai/models](https://poolside.ai/models), via [radar mirror](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1-nvfp4.json)). The technical report is authored as "Poolside Team, research@poolside.ai" ([arXiv 2605.27605](https://arxiv.org/pdf/2605.27605), [PDF](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf)).

### 1b. Variants, repos, dates and licenses

| Model | Total / active | Official HF repos (all published by `poolside`) | Release | License |
|---|---|---|---|---|
| **Laguna XS.2** | 33B / 3B | `poolside/Laguna-XS.2`, `-FP8`, `-NVFP4`, `-INT4`, `Laguna-XS.2-speculator.dflash` | 2026-04-28 | Apache-2.0 |
| **Laguna M.1** | 225B / 23B (config-derived 225.8B / 23.5B) | `poolside/Laguna-M.1`, `Laguna-M.1-base`, `-FP8`, `-NVFP4` | Announced and API-only 2026-04-28; **open weights 2026-06-18** | Apache-2.0 |
| **Laguna XS 2.1** | 33.44B / 3B | `poolside/Laguna-XS-2.1`, `-FP8`, `-NVFP4`, `-INT4`, `-GGUF`, `-DFlash`, `-DFlash-FP8/-NVFP4/-INT4` | 2026-07-02 | OpenMDW-1.1 |
| **Laguna S 2.1** | 117.56B / ~8B (card says 8.5B for NVFP4) | `poolside/Laguna-S-2.1`, `-FP8`, `-NVFP4` (re-uploaded Aug 2026), `-INT4`, `-GGUF`, `-DFlash`, `-DFlash-FP8/-NVFP4/-INT4` | **2026-07-21** | OpenMDW-1.1 |
| Laguna-tiny-per-element | tiny test model | `poolside/Laguna-tiny-per-element` | — | — |

Sources for the table:
- Repo ids: FACT (direct) from [vLLM recipes](https://github.com/vllm-project/recipes/tree/main/models/poolside), the [SGLang cookbook S-2.1](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-S-2.1.mdx), the [SGLang cookbook M.1](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-M.1.mdx) and [vLLM tests/models/registry.py](https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py).
- Dates:
  - XS.2 recipe `date_added: 2026-04-28`.
  - M.1 open weights: Poolside's X post "releasing the weights for Laguna M.1 … base and post-trained … under Apache 2.0" ([X](https://x.com/poolsideai/status/2067623353230217448), snippet). The recipe `date_added` is 2026-06-18, and AINews covered it on [2026-06-19](https://github.com/smol-ai/ainews-web-2025/blob/main/src/content/issues/26-06-19-not-much.md).
  - XS 2.1 on 2026-07-02 ([Poolside blog](https://poolside.ai/blog/introducing-laguna-xs-2-1), snippet).
  - S 2.1 on 2026-07-21 ([VentureBeat](https://venturebeat.com/infrastructure/poolside-drops-laguna-s-2-1-an-open-weight-coding-model-that-beats-rivals-10x-its-size), [MarkTechPost](https://www.marktechpost.com/2026/07/21/poolside-releases-laguna-s-2-1/); snippets).
- Licenses:
  - The XS 2.1 license change was announced on X ([X](https://x.com/poolsideai/status/2072700435610112321), snippet).
  - M.1 is Apache-2.0 per the [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-M.1.mdx) (FACT, direct).
  - S 2.1 is `license:openmdw-1.1` per the HF API metadata mirror ([radar](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1.json)).
  - One secondary page says M.1 is OpenMDW. That contradicts the primary sources, so I discard it.

**Quantized exports.**
- FACT: all FP8, NVFP4 and INT4 exports are **Poolside's own**.
- Third-party exports found: GGUF from `unsloth`, `linuxid10t` (M.1), `AtomicChat` and `vcruz305`, and MLX from `mlx-community/Laguna-S-2.1-oQ2e/oQ3e/oQ4e` (snippets: [unsloth](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF), [mlx-community](https://huggingface.co/mlx-community/Laguna-S-2.1-oQ4e)).
- I found no third-party NVFP4 or FP8 exports.

### 1c. Base-model provenance

- **FACT (snippet): Poolside says Laguna is trained from scratch.**
  - M.1 is "trained completely in-house and from scratch on 30T tokens, using 6,144 … Hopper GPUs".
  - M.1 and XS.2 were pre-trained on 6,144 and 2,048 H200s respectively, over more than 30T tokens of web, code and synthetic data, with the Muon (Moonlight) optimizer.
  - Sources: [tech report](https://arxiv.org/pdf/2605.27605), [Poolside blog](https://poolside.ai/blog/laguna-a-deeper-dive).
- **FACT (snippet): S 2.1 reused XS 2.1's pre-training data.** Training started 2026-05-22 on 4,096 H200s, and S 2.1 launched in under 9 weeks ([VentureBeat](https://venturebeat.com/infrastructure/poolside-drops-laguna-s-2-1-an-open-weight-coding-model-that-beats-rivals-10x-its-size)).
- **FACT (mirror):** HF metadata shows `base_model: None` for `Laguna-S-2.1` and `Laguna-XS-2.1` ([radar S](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1.json), [radar XS](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-xs-2-1.json)).
- **INFERENCE: the architecture supports the from-scratch claim.** It does not match Qwen, Gemma or Llama:
  - vocabulary of 100,352 tokens (Qwen3.5 uses 248,320);
  - sigmoid router with a DeepSeek-style aux-loss-free bias;
  - softplus attention-output gate;
  - per-layer head counts;
  - 512-token SWA with its own RoPE.

  Nothing indicates a Chinese base.

### 1d. Other things called "Laguna"

None of these is an LLM:
- the province (Philippines), Laguna Beach, cars (Renault Laguna, [Vanderhall Laguna](https://en.wikipedia.org/wiki/Vanderhall_Laguna)) and sailboats ([Laguna 26](https://en.wikipedia.org/wiki/Laguna_26));
- the community adapter `poolside-laguna-hackathon/laguna-vision`, which is built on Laguna.

I found no GPU codename or dataset named Laguna. Searches for "Laguna 397B", "laguna-ai", "Laguna DeepReinforce" and "Laguna Ornith" returned only Poolside Laguna pages or Laguna-vs-Ornith comparison pages.

---

## 2. Architecture and KV-cache arithmetic

### 2a. Config values

**FACT (mirror)** from verbatim copies of the HF `config.json` files on GitHub:
- S 2.1 BF16, FP8 and NVFP4, and M.1: [chrisaboyd capacity-planner assets](https://github.com/chrisaboyd/Samples/tree/main/llms/capacity-planner/lib/tests/assets).
- XS 2.1: [Layr-Labs fixture](https://github.com/Layr-Labs/mlxfast-challenge/blob/main/fixtures/poolside_laguna_xs_2_1_nvfp4_config.json).
- XS.2: [copy](https://github.com/cooker-c/ai-model-security-assessment-tool/blob/main/sample_models/Laguna-XS.2/config.json).

**Validation (INFERENCE, strong):** I computed the parameter count from these configs:
- S 2.1: **117,561,977,600**, which exactly equals HF's `parameters` field ([radar](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1.json)).
- XS 2.1: 33,442,617,088, also an exact match.
- M.1: 225.80B total and 23.46B active.

The architecture is also confirmed by the [vLLM recipes](https://github.com/vllm-project/recipes/tree/main/models/poolside), the [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-S-2.1.mdx) and [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/laguna.cpp).

| Field | **Laguna S 2.1** | **Laguna M.1** | **Laguna XS 2.1** (XS.2 same shape) |
|---|---|---|---|
| Total / active params | 117.56B / ~8.4B (7.8B excluding embeddings) | 225.8B / 23.5B | 33.44B / ~3.0B |
| Layers | 48 | 70 | 40 |
| Attention | Hybrid **SWA + global**, window 512; layer_types FULL at i%4==0 | **Global (full) in all 70 layers**, `sliding_window: 0` | Hybrid SWA + global, window 512, same pattern |
| Full-attention layers | **12** (0,4,…,44) | **70** | **10** |
| Linear attention (GDN/Mamba) | **None** | None | None |
| num_key_value_heads | 8 (all layers) | 8 | 8 |
| Q heads | 48 on global layers, 72 on SWA layers (`num_attention_heads_per_layer`) | 64 | 48 on global, 64 on SWA |
| head_dim | 128 | 128 | 128 |
| hidden_size | 3072 | 4096 | 2048 |
| Experts | 256 routed, top-10, plus 1 shared (1024 intermediate); layer 0 dense (12288) | 256 routed, top-16, plus 1 shared (1024); layers 0–2 dense (16384) | 256 routed, top-8, plus 1 shared (512); layer 0 dense (8192) |
| Attention output gate | softplus, per-head | softplus, per-element | per-head |
| max_position_embeddings | **1,048,576** (BF16). The recipe says the quantized checkpoints are "configured for 256K" | **262,144** | 262,144 |
| RoPE, global layers | YaRN factor 128 over 8192, θ=5e5, partial_rotary 0.5, attention_factor 1.485 | YaRN factor 64 over 4096, θ=5e5, full rotary | YaRN factor 32 over 8192, partial 0.5 |
| RoPE, SWA layers | default θ=1e4, full rotary | n/a | default θ=1e4 |
| Vocab | 100,352 | 100,352 | 100,352 |
| Vision tower | **No** (`LagunaForCausalLM`, no vision_config) | **No** | **No** |
| Thinking | Interleaved thinking; `enable_thinking` chat-template flag, **off by default** in the template; can be forced on server-side with `--default-chat-template-kwargs '{"enable_thinking": true}'` | Same flag; the vLLM recipe enables it at launch | Same as S 2.1 |

Sources for the thinking row: [vLLM recipes](https://github.com/vllm-project/recipes/tree/main/models/poolside) and the [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-S-2.1.mdx).

### 2b. KV-cache bytes per token

These are INFERENCE: arithmetic on the FACT configs.

Formulas:
- BF16 = 2 × full_layers × kv_heads × head_dim × 2 B
- FP8 = the same × 1 B
- TurboQuant K8V4 = full_layers × kv_heads × (head_dim × 1 B for K + head_dim × 0.5 B for V + 4 B scale/zero)

The K8V4 per-head slot matches upstream vLLM's `TurboQuantConfig.slot_size` exactly: the key is `head_dim` bytes and the value is `head_dim×4/8 + 4` bytes ([vLLM turboquant/config.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/turboquant/config.py)). With your Ornith inputs it reproduces 15×2×(256+128+4) = **11,640 B**.

- **Laguna S 2.1** (12 × 8 × 128):
  - BF16: 2×12×8×128×2 = **49,152 B/token**
  - FP8: **24,576**
  - K8V4: 12×8×(128+64+4) = 12×8×196 = **18,816**
- **Laguna M.1** (70 × 8 × 128):
  - BF16: 2×70×8×128×2 = **286,720 B/token**
  - FP8: **143,360**
  - K8V4: 70×8×196 = **109,760**
- **Laguna XS 2.1** (10 × 8 × 128):
  - BF16: **40,960**
  - FP8: **20,480**
  - K8V4: 10×8×196 = **15,680**
- Ornith 1.5 397B (15 × 2 × 256): BF16 30,720; FP8 15,360; K8V4 11,640 (your figures, reproduced)
- Ornith 1.5 35B-A3B (10 × 2 × 256): BF16 20,480; FP8 10,240; K8V4 10×2×388 = 7,760

**GiB per user.** 1 GiB = 2^30 B. For example, 49,152 × 262,144 / 2^30 = 12.000.

| Model | BF16 @262,144 | FP8 @262,144 | K8V4 @262,144 | BF16 @1,048,576 | FP8 @1,048,576 | K8V4 @1,048,576 |
|---|---|---|---|---|---|---|
| **Laguna S 2.1** | **12.00** | 6.00 | **4.59** | 48.00 | 24.00 | **18.38** |
| **Laguna M.1** | **70.00** | 35.00 | **26.80** | 280.0 † | 140.0 † | 107.19 † |
| Laguna XS 2.1 | 10.00 | 5.00 | 3.83 | 40.0 † | 20.0 † | 15.31 † |
| Ornith 1.5 397B | 7.50 | 3.75 | 2.84 | 30.00 | 15.00 | 11.37 |
| Ornith 1.5 35B-A3B | 5.00 | 2.50 | 1.90 | 20.00 | 10.00 | 7.58 |

† Beyond the model's configured max_position_embeddings of 262,144, so not a supported setting.

Ratios versus Ornith 1.5 397B, per token:
- S 2.1: ×1.60 at BF16, ×1.62 at K8V4
- M.1: ×9.33 at BF16, ×9.43 at K8V4
- XS 2.1 versus Ornith 35B: ×2.0

**Fixed per-sequence state** (INFERENCE):
- **Laguna has no recurrent state.** Its SWA layers hold a fixed window instead:
  - S 2.1: 36 layers × 512 tokens × 2 × 8 × 128 → **72 MiB BF16**, 36 MiB FP8, about 27.6 MiB K8V4
  - XS 2.1: 30 layers → 60 / 30 / 23 MiB
  - vLLM also reserves the in-flight prefill chunk on top of the window, so the real figure is slightly higher.
- For comparison, Ornith's recurrent state, assuming it keeps its Qwen3.5 base dimensions:
  - Qwen3.5-397B-A17B: 60 layers, `full_attention_interval` 4, so 45 GatedDeltaNet layers; `linear_num_value_heads` 64, 128×128, `mamba_ssm_dtype` float32. That gives 45 × 64 × 128 × 128 × 4 B = **180 MiB** per sequence, plus about 3 MiB of conv state.
  - Qwen3.5-35B-A3B: 30 GDN layers × 32 value heads → **60 MiB**.
  - Source for the Qwen3.5 values: [MaxText copy of the Qwen3.5 HF configs](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/hf_model_configs.py). I did not verify Ornith's own config (HF blocked).

**Two practical caveats** (FACT from code or measurement; the impact is INFERENCE):
1. **Upstream vLLM TurboQuant treats Laguna as a non-hybrid model, so it adds "boundary skip" layers.**
   - The code keeps the first 2 and last 2 attention layers unquantized. It disables this only for Mamba or linear-attention hybrids ([vLLM turboquant/config.py `get_boundary_skip_layers`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/turboquant/config.py), [arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)).
   - For S 2.1, layer 0 is a global layer. Global KV becomes 11×8×196 + 1×2×8×128×2 = **21,344 B/token**: 5.21 GiB at 256K and 20.84 GiB at 1M.
   - For M.1: 66×8×196 + 4×4,096 = **119,872 B/token**, or 29.27 GiB at 256K.
   - Ornith is hybrid, so no skip applies there. Your patched vLLM may behave differently.
2. **The DFlash speculator adds full-context KV.**
   - A third-party capacity planner, measured against real vLLM startups on RTX PRO 6000, reports that the S-2.1 DFlash drafter's **6 layers "all allocate full-context KV"**. That adds +50% on top of the 12 target layers ([chrisaboyd reference.rs](https://github.com/chrisaboyd/Samples/blob/main/llms/capacity-planner/lib/tests/reference.rs)).
   - This is consistent with a DGX Spark run: 32.07 GiB of KV held 918,419 tokens, about 37.5 KB/token, which matches (12+6)×8×128×2×1 B = 36,864 B plus SWA ([sparkrun runbook](https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/laguna-s-2.1.md)).

---

## 3. Benchmarks

Poolside's cards are coding-only. Sources:
- **S 2.1**: the model-card table dated 2026-07-21, via an HF card snapshot ([radar laguna-s-2-1.md](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1.md)), cross-checked with search snippets from [the Poolside blog](https://poolside.ai/blog/introducing-laguna-s-2-1), [VentureBeat](https://venturebeat.com/infrastructure/poolside-drops-laguna-s-2-1-an-open-weight-coding-model-that-beats-rivals-10x-its-size) and [latent.space](https://www.latent.space/p/ainews-laguna-s-21-released-cheaper). Non-thinking numbers come from [ailmanac](https://github.com/derob98/ailmanac/blob/main/docs/models/poolside-laguna-family.mdx) (secondary).
- **XS 2.1**: card table via [radar laguna-xs-2-1.md](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-xs-2-1.md). It matches the vLLM recipe text "SWE-bench Multilingual 57.7% → 63.1%, SWE-bench Verified 69.9% → 70.9%".
- **M.1**:
  - SWE-V 74.6 is FACT (direct) from the [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/poolside/Laguna-M.1.yaml).
  - The other M.1 numbers come from secondary summaries, and those summaries **disagree**: [AINews/Reddit](https://github.com/smol-ai/ainews-web-2025/blob/main/src/content/issues/26-06-19-not-much.md) versus [hammer/labs](https://github.com/hammer/labs/blob/main/data/outputs/poolside/laguna-m1.yaml).

| Benchmark | **Laguna S 2.1** (thinking) | Laguna M.1 | Laguna XS 2.1 | **Ornith 1.5 397B** (your figures) | Ornith 1.5 35B-A3B (your figures) | Claude Opus 4.8 (as quoted) |
|---|---|---|---|---|---|---|
| Terminal-Bench 2.1 | **70.2** (60.4 without thinking) | — | — | **86.1** | 67.8 | 84.6 on Poolside's compiled leaderboard (snippet, [BenchLM](https://benchlm.ai/compare/claude-opus-4-8-vs-laguna-s-2-1)); 85.0 as quoted by Ornith ([aiweekly](https://aiweekly.co/alerts/deepreinforce-releases-ornith-15-in-397b-35b-and-9b-sizes), snippet) |
| Terminal-Bench 2.0 | — | 45.8 (AINews) / 56.9 (hammer) / 40.7 (one snippet): **unresolved** | 37.5 | — | — | — |
| SWE-bench Verified | **not reported** | **74.6** | 70.9 | 86.0 | 79.0 | 88.6 ([BenchLM](https://benchlm.ai/compare/claude-opus-4-8-vs-laguna-s-2-1), snippet) |
| SWE-bench Pro (public) | **59.4** | 49.2 (46.9 in one snippet) | 47.6 | 65.1 (third-party snippet, [benchgen](https://benchgen.com/models/ornith-deepreinforce/ornith-1-5-397b); not in your list) | — | — |
| SWE-bench Multilingual | **78.5** | 63.1 (AINews) / 73.3 (hammer): **unresolved** | 63.1 | — | — | — |
| GPQA Diamond | not reported | not reported | not reported | 92.8 | 89.2 | — |
| MCP-Atlas | not reported | not reported | not reported | 80.0 | 70.2 | — |
| Toolathlon | **49.7** ("Toolathlon Verified"; may not be the same split) | not reported | not reported | 71.2 | 48.7 | — |
| BrowseComp | not reported | not reported | not reported | 86.6 | — | — |
| HLE | not reported | not reported | not reported | — | — | — |
| DeepSWE v1.1 | 40.4 (16.5 without thinking) | — | — | 56.0 (snippet) | — | 59 (as quoted by Ornith, snippet) |
| SWE Atlas Codebase QnA | 46.2 | — | — | — | — | — |
| Long context (MRCR / RULER / NIAH) | **none published** | none | none | ? | ? | — |

Notes on the table:
- **Where your Ornith columns come from (INFERENCE).** I matched your six 397B numbers to TB2.1 / SWE-V / GPQA / MCP-Atlas / Toolathlon / BrowseComp using the order of your 35B list. A snippet confirms GPQA 92.8 and BrowseComp 86.6. One snippet gives Ornith 397B TB2.1 as 85.1, not 86.1.
- **Opus 4.8 on the Laguna card.** The S 2.1 card table does not include Opus 4.8. Its closed-model row is **Claude Fable 5**: TB2.1 88, SWE-Pro 80.3, DeepSWE 70. The open rows are Tencent Hy3, Inkling, Nemotron 3 Ultra, DeepSeek-V4-Pro-Max, Kimi K3, Qwen 3.7 Max and Muse Spark 1.1.
- **How Poolside ran its evals** (FACT, snippet):
  - Laude Institute Harbor framework, Poolside's own agent harness, up to 500 steps.
  - pass@1 averaged over 4 attempts (SWE-V, SWE-Multilingual), 2 (SWE-Pro) or 5 (TB2.0).
  - Full trajectories are published at trajectories.poolside.ai.
  - Sources: [poolside.ai/models](https://poolside.ai/models) via [radar](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-xs-2-1.json), and [the-decoder](https://the-decoder.com/poolsides-laguna-s-2-1-is-a-small-open-weight-coding-model-that-punches-well-above-its-size/).
- **INFERENCE:** on the published numbers, Laguna S 2.1 is **16–22 points behind Ornith 1.5 397B** on TB2.1 and Toolathlon, and roughly **level with Ornith 1.5 35B-A3B**. It has a stronger SWE-Pro and SWE-Multilingual profile, but there is nothing to compare on GPQA, MCP-Atlas, BrowseComp or long context. "Switching to the best Laguna" would be a **downgrade from Ornith 397B** on every benchmark you track that Poolside also reports.

---

## 4. Serving

### 4a. Checkpoint sizes

| Checkpoint | Size | Status / source |
|---|---|---|
| S 2.1 BF16 | **235,129,327,775 B** (235.1 GB / 219.0 GiB) | FACT (mirror of HF `storage_bytes`, [radar](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1.json)); recipe says "~235GB" |
| S 2.1 FP8 (block 128×128) | **131,264,796,160 B** (122.25 GiB) | FACT (mirror: safetensors-index `total_size`, [reference.rs](https://github.com/chrisaboyd/Samples/blob/main/llms/capacity-planner/lib/tests/reference.rs)). By bytes it is FP8 79% + BF16 21%; the recipe says "~121GB" |
| S 2.1 NVFP4, **current** | **~67 GB, 14 shards**; loads as 69.45 GiB including the 2.1 GB drafter. Recipe: "~72GB … fits a single B200/B300", `vram_minimum_gb: 86` | FACT (third-party measurement, [sparkrun runbook](https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/laguna-s-2.1.md); [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/poolside/Laguna-S-2.1.yaml)) |
| S 2.1 NVFP4, **earlier revision** | **99,697,287,856 B, 49 shards**; experts in layers 40–47 left in BF16; "NVFP4 53% + BF16 47%" | FACT (mirror: config plus shard headers, [chrisaboyd](https://github.com/chrisaboyd/Samples/blob/main/llms/capacity-planner/lib/tests/assets/laguna-s-2.1-nvfp4-config.json)) |
| S 2.1 INT4 (W4A16) | ~72 GB | FACT (vLLM recipe) |
| S 2.1 DFlash drafter | 2,229,962,896 B | FACT (mirror, [reference.rs](https://github.com/chrisaboyd/Samples/blob/main/llms/capacity-planner/lib/tests/reference.rs)) |
| M.1 BF16 | "~450 GB"; 225.8B × 2 = 451.6 GB | FACT (vLLM recipe) |
| M.1 FP8 | `vram_minimum_gb` 270 | FACT (vLLM recipe) |
| M.1 NVFP4 | "~135 GB" | FACT (vLLM recipe) |
| XS 2.1 BF16 | 66,889,022,016 B | FACT (mirror) |
| XS 2.1 FP8 / NVFP4 / INT4 | ~40 / ~26 / ~29 GB of VRAM | FACT (vLLM recipe) |

On the two NVFP4 revisions:
- The HF repo was updated 2026-08-25, and its storage shows 315.5 GB across revisions, which suggests it was re-uploaded ([radar](https://github.com/686f6c61/radar-huggingface-dataset/blob/main/data/poolside/laguna-s-2-1-nvfp4.json)). **Verify the current `model.safetensors.index.json` before you size anything.**
- The NVFP4 scheme is **W4A4** (FP4 activations, group 16) on routed experts only. Attention, router, shared experts, layer-0 MLP and lm_head stay BF16 (older config copy).

### 4b. Minimum software versions

- **vLLM** (FACT, direct):
  - XS.2, XS 2.1 and M.1 need ≥ **0.21.0** (PR #41129). Quantized XS needs ≥ **0.22.0** for the FP8-KV per-layer head fix, #42650.
  - **S 2.1 needs ≥ 0.25.0**, and ≥ 0.25.1 for DFlash (PR #46853).
  - Sources: [recipes](https://github.com/vllm-project/recipes/tree/main/models/poolside), [runbook](https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/laguna-s-2.1.md) ("0.24.0 doesn't support Laguna" S 2.1).
  - My own check of tagged sources: `laguna.py` is present in v0.22.0 through v0.30.0, and `laguna_dflash.py` from v0.25.0 onward. PyPI release dates are v0.25.0 on 2026-07-11 and v0.30.0 on 2026-09-22 ([PyPI](https://pypi.org/project/vllm/#history)).
- **transformers** (FACT, direct): native `models/laguna` first appears in tag **v5.7.0**. The checkpoints still ship `auto_map` custom code, so recipes pass `--trust-remote-code`. **INFERENCE:** for an offline stack, audit `configuration_laguna.py` and `modeling_laguna.py` in the repo before use.
- **SGLang** (FACT, direct): "On transformers ≥ 5.10 the standalone chat_template.jinja auto-loads" ([cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-S-2.1.mdx)).

### 4c. Recommended vLLM flags

FACT, direct, from the [recipes](https://github.com/vllm-project/recipes/tree/main/models/poolside):

`--enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --trust-remote-code --max-model-len 262144`

- Add `--default-chat-template-kwargs '{"enable_thinking": true}'` to force thinking on.
- Optional DFlash: `--speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash","num_speculative_tokens":15,"method":"dflash"}' --moe-backend triton`.
- FP8 needs `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0`.
- Sampling: temperature 1.0, top_k 20 for M.1; temperature 0.7, top_k 20 in the S/XS examples.

### 4d. Support in vLLM main

- **FACT (direct):** `LagunaForCausalLM` → `vllm/model_executor/models/laguna.py`, with SupportsPP, SupportsLoRA and SupportsEagle3. `DFlashLagunaForCausalLM` → `laguna_dflash.py`. Both are registered in [registry.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py). `poolside_v1` is registered in both `vllm/reasoning/__init__.py` and `vllm/tool_parsers/__init__.py`.
- **Open vLLM issues** (FACT, titles):
  - [#49379](https://github.com/vllm-project/vllm/issues/49379): "Laguna-S-2.1 poolside_v1 reasoning parser falls back to IdentityReasoningParser"
  - [#49711](https://github.com/vllm-project/vllm/issues/49711): poolside_v1 reports zero Responses reasoning_tokens
  - [#57623](https://github.com/vllm-project/vllm/issues/57623): the nested-RoPE validator patch leaks to later configs; related to the closed [#57188](https://github.com/vllm-project/vllm/issues/57188), "Laguna global RoPE validator mutates later flat rope_parameters"

### 4e. TurboQuant with Laguna

- **FACT (direct):** upstream vLLM main has `turboquant_k8v4` (plus 4bit_nc, k3v4_nc and 3bit_nc) in `CacheDType` ([cache.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)). The backend accepts a `sliding_window` argument ([turboquant_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/turboquant_attn.py)). Enabling TQ forces FlashAttention 2 ([arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)).
- **INFERENCE:** all Laguna layers are standard softmax attention, both global and SWA, so TQ can in principle apply to all 48 layers, not just the 12 global ones. Only the 12 global layers matter for context-scaled memory.
  - Poolside validates only an **FP8 KV cache**; its quantized checkpoints "ship an FP8-quantized KV cache".
  - I found **no report of K8V4 on Laguna**. Laguna's softplus output gate and partial-rotary global layers are untested with TQ. Treat K8V4 as an A/B item.
  - The boundary-skip behavior (§2b) means upstream vLLM keeps layers 0, 1, 46 and 47 unquantized.

### 4f. Do the weights fit?

INFERENCE, weights only; KV comes out of what is left. Capacities (vendor specs, not re-fetched): RTX PRO 6000 Blackwell 96 GB, which is about an 86 GiB vLLM budget at 0.9 utilization as measured in [reference.rs](https://github.com/chrisaboyd/Samples/blob/main/llms/capacity-planner/lib/tests/reference.rs); B200 180 GB, so 8× is 1,440 GB; H200 141 GB, so 2× is 282 GB.

| Model / precision | 1× RTX PRO 6000 (96 GB) | 2× H200 (282 GB) | 8× B200 (1,440 GB) |
|---|---|---|---|
| **S 2.1 NVFP4** (current ~67 GiB) | **Yes.** About 12–16 GiB is left for KV → ~2–3 × 256K users at K8V4 (4.6 GiB each), or ~2 at FP8. **No** if the 99.7 GB revision is what gets served. | NVFP4 is "Blackwell only" per the recipe; use INT4 W4A16 (~72 GB) or FP8 instead | Yes |
| S 2.1 FP8 (122.25 GiB) | No. TP2 on **2× RTX PRO 6000** was measured working: 16.7 GiB of KV per GPU at 262,144 context, 3.29× concurrency with DFlash | Yes (recipe: "TP2 on H200") | Yes |
| S 2.1 BF16 (219 GiB) | No | Borderline; the recipe minimum is 282 GB, which is exactly 2× H200 | Yes |
| **M.1 NVFP4** (~135 GB) | No. 2× RTX PRO 6000 holds the weights but leaves little KV (26.8 GiB per 256K user at K8V4) | Blackwell-only format | Yes |
| M.1 FP8 (~226 GB+; recipe minimum 270 GB) | No | Borderline; the recipe says a 4× H200/B200 node | Yes |
| M.1 BF16 (~451 GB) | No | No | Yes |
| XS 2.1 NVFP4 / FP8 / BF16 (26 / 40 / 62 GiB) | Yes | Yes | Yes |

**Quirks** (FACT, direct, [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-S-2.1.mdx)):
- "BF16 reasons approximately 2× longer than FP8/INT4 on AIME25 (median 34.8k vs 16.9k tokens)".
- FP8 in SGLang needs `SGLANG_SHARED_EXPERT_TP1=1`.
- With DFlash, the flashinfer backend "breaks this hybrid-SWA model at tp ≥ 4 on Blackwell".

---

## 5. Provenance and the "no PRC" rule

| | Laguna (all variants) | Ornith 1.5 (35B-A3B, 397B) |
|---|---|---|
| Publisher | Poolside. American per [Wikipedia](https://en.wikipedia.org/wiki/Poolside_AI) (HQ San Francisco), with a Paris HQ relocation reported in 2023 ([Sifted](https://sifted.eu/articles/poolside-raises-126m-relocated-france-news)). Not PRC. | DeepReinforce (`ornith-ai`, formerly also `deepreinforce-ai` on HF). Santa Clara / San Jose, CA per [Crunchbase](https://www.crunchbase.com/organization/deepreinforce-ai) and [wiki](https://ai.miraheze.org/wiki/DeepReinforce) (snippets). Not PRC. |
| Base model | **Trained from scratch** according to the tech report and cards; HF `base_model: None` | Built on **Qwen3.5** (Alibaba, **PRC**) and Gemma 4 (Google). The 35B and 397B use the `qwen3_5_moe` architecture ([MindStudio](https://www.mindstudio.ai/blog/ornith-1-5-35b-a3b-benchmarks), [Simon Willison](https://simonwillison.net/2026/Jun/29/ornith/); snippets) |
| Under your policy | **Compliant, with no PRC lineage at all** | **Compliant** under the "non-PRC retrain of a Chinese base" exception |

INFERENCE: if the team's real concern is PRC *lineage* rather than the PRC *publisher*, Laguna is the stricter-compliant choice. Under the policy as written, both are allowed.

---

## 6. Independent views and comparisons

These are aggregator or community sources. They are **not** independently re-run evals.

- **BenchLM, Laguna M.1 vs Ornith-1.0-35B:** "tied on the BenchAlign overall score". Ornith edges coding, 65.9 vs 64.8, and leads agentic tasks, 64.2 vs 45.8 ([BenchLM](https://benchlm.ai/compare/laguna-m-1-vs-ornith-1-0-35b), snippet).
- **BenchLM, Laguna S 2.1 vs Ornith-1.0-397B:** tied overall. Ornith leads coding, 74.6 vs 59.4, and SWE-Pro, 62.2 vs 59.4. Laguna's advantage is the 1M context ([BenchLM](https://benchlm.ai/compare/laguna-s-2-1-vs-ornith-1-0-397b), snippet). This compares **Ornith 1.0**; I found no Ornith **1.5** vs Laguna comparison.
- **HF discussion "BenchMAXXXED"** on Laguna-S-2.1 ([#28](https://huggingface.co/poolside/Laguna-S-2.1/discussions/28), snippet):
  - a user found it "definitely worse than Qwen 3.6 27b and … 3.6 35b" in their own app;
  - it is "sensitive to quantization and setup".
- **latent.space AINews** on the S 2.1 launch: one commenter says the claims "sound too good to be true" ([latent.space](https://www.latent.space/p/ainews-laguna-s-21-released-cheaper), snippet).
- **r/LocalLLaMA on M.1** (via AINews 2026-06-19, direct):
  - "competitive with open models like Devstral 2 and GLM-4.7 but below DeepSeek-V4 Flash / Qwen3.5 on several listed metrics";
  - "potentially the strongest US-trained open-weight coding model" ([AINews](https://github.com/smol-ai/ainews-web-2025/blob/main/src/content/issues/26-06-19-not-much.md)).
- **Secondary summary of Poolside's own stated limitations** ([ailmanac](https://github.com/derob98/ailmanac/blob/main/docs/models/poolside-laguna-family.mdx), secondary):
  - trouble with "slight schema variations" in third-party harnesses;
  - occasional incorrectly-escaped JSON in nested tool calls;
  - overthinking on math.
- **HN threads exist** for [Laguna S 2.1](https://news.ycombinator.com/item?id=48995261) and [XS.2/M.1](https://news.ycombinator.com/item?id=47936511), and there is a [YouTube XS 2.1 vs Ornith 1.0 video](https://www.youtube.com/watch?v=cSMs-qeS8ME). None of these were fetchable, so their content is not summarized.
- **I found no vLLM issue comparing Ornith and Laguna.**

---

## 7. Open items to verify on a host with HF access

1. `curl -s 'https://huggingface.co/api/models?author=poolside&search=laguna&full=true'`: confirm the repo list and `lastModified`.
2. `poolside/Laguna-S-2.1-NVFP4`: the current `model.safetensors.index.json` total_size (67 GB or 99.7 GB?), the `quantization_config.ignore` list, `max_position_embeddings` (256K or 1M), and any `kv_cache_scheme`.
3. The `poolside/Laguna-M.1` README benchmark table: settle SWE-Multilingual (63.1 vs 73.3) and TB2.0 (45.8 vs 56.9).
4. The `LICENSE.md` of each repo you would use: Apache-2.0 (XS.2, M.1) or OpenMDW-1.1 (XS 2.1, S 2.1).
5. An A/B test of `--kv-cache-dtype turboquant_k8v4` against `fp8` on Laguna S 2.1, run in your patched vLLM. Nobody has published this.

---

## Sources

- **Read directly:**
  - [vLLM recipes/poolside](https://github.com/vllm-project/recipes/tree/main/models/poolside) (commit 09dea3c, 2026-09-22)
  - [SGLang cookbook S-2.1](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-S-2.1.mdx) and [M.1](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Poolside/Laguna-M.1.mdx)
  - transformers [config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/laguna/configuration_laguna.py) and [doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/laguna.md)
  - vLLM [laguna.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/laguna.py), [registry](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py), [tests registry](https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py), [cache.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py), [TQ config](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/turboquant/config.py), [TQ backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/turboquant_attn.py)
  - [llama.cpp laguna.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/laguna.cpp)
- **Config.json copies:** [chrisaboyd assets and reference.rs](https://github.com/chrisaboyd/Samples/tree/main/llms/capacity-planner/lib/tests), [Layr-Labs XS 2.1](https://github.com/Layr-Labs/mlxfast-challenge/blob/main/fixtures/poolside_laguna_xs_2_1_nvfp4_config.json), [XS.2 copy](https://github.com/cooker-c/ai-model-security-assessment-tool/blob/main/sample_models/Laguna-XS.2/config.json), [Qwen3.5 configs (MaxText)](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/hf_model_configs.py)
- **HF metadata and card snapshots:** [radar-huggingface-dataset/poolside](https://github.com/686f6c61/radar-huggingface-dataset/tree/main/data/poolside)
- **Third-party runbook:** [sparkrun-recipes](https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/laguna-s-2.1.md)
- **Search snippets only** (pages blocked): poolside.ai blogs ([S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1), [XS 2.1](https://poolside.ai/blog/introducing-laguna-xs-2-1), [deeper dive](https://poolside.ai/blog/laguna-a-deeper-dive)), [arXiv 2605.27605](https://arxiv.org/pdf/2605.27605), [VentureBeat](https://venturebeat.com/infrastructure/poolside-drops-laguna-s-2-1-an-open-weight-coding-model-that-beats-rivals-10x-its-size), [MarkTechPost](https://www.marktechpost.com/2026/07/21/poolside-releases-laguna-s-2-1/), [Wikipedia](https://en.wikipedia.org/wiki/Poolside_AI), [BenchLM](https://benchlm.ai/compare/laguna-s-2-1-vs-ornith-1-0-397b), [aiweekly on Ornith 1.5](https://aiweekly.co/alerts/deepreinforce-releases-ornith-15-in-397b-35b-and-9b-sizes)
