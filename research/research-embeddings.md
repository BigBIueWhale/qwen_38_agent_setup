# Embedding models for offline Hebrew + English RAG under the "no PRC publisher" rule

Research date: 2026-09-22. Every line is tagged **FACT** (read from a named source) or **INFERENCE** (my reasoning or arithmetic).

## 0. What I could reach, and how the numbers were produced

- **FACT.** From this session the egress proxy blocks `huggingface.co` (HTTP 403 on CONNECT, which also covers `/raw/main/README.md`, `/raw/main/config.json` and `/api/models/...`). It also blocks `hf-mirror.com`, `arxiv.org`, `blogs.bing.com`, `labs.ai.azure.com`, `techcommunity.microsoft.com`, `ai.google.dev`, `codabench.org` and most news sites. GitHub, raw.githubusercontent.com and PyPI are reachable.
- **FACT.** I therefore took model metadata from three places:
  - the MTEB model registry (<https://github.com/embeddings-benchmark/mteb>, commit `34d7c32`, 2026-09-22);
  - the MTEB results repository (<https://github.com/embeddings-benchmark/results>, commit `fe1eb05`, 2026-09-22);
  - the vLLM source (<https://github.com/vllm-project/vllm>, commit `74370a0`, 2026-09-22).

  Hugging Face model-card text is quoted only where search-engine snippets of those pages returned it. Those claims are labelled "(via search snippet of the HF card)".
- **FACT.** The MMTEB scores below are **computed by me**. I ran `mteb` from repo HEAD (`ResultCache(...).load_results(tasks="MTEB(Multilingual, v2)")` followed by `get_benchmark_result()`) against the official results repo. The script is in `scratchpad/scripts/mmteb.py`. The numbers reproduce the vendors' own claims, which validates the method:
  - Qwen3-Embedding-8B: 70.58 computed, 70.58 in the Qwen README (<https://github.com/QwenLM/Qwen3-Embedding>).
  - Harrier-27B: 74.27 computed, 74.3 in the press (<https://www.marktechpost.com/2026/03/30/microsoft-ai-releases-harrier-oss-v1-a-new-family-of-multilingual-embedding-models-hitting-sota-on-multilingual-mteb-v2/>).
  - pplx-embed-4B retrieval: 69.66 computed, 69.66 per Perplexity (<https://www.marktechpost.com/2026/02/26/perplexity-just-released-pplx-embed-new-sota-qwen3-bidirectional-embedding-models-for-web-scale-retrieval-tasks/>).

---

## 1. Microsoft `harrier-oss-v1-27b`

**It exists.** The MTEB registry lists it as open-weights at revision `0c0fc62f…`, and the MTEB results repo holds a full 131-task MMTEB run for it (<https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/harrier_models.py>, <https://github.com/embeddings-benchmark/results/tree/main/results/microsoft__harrier-oss-v1-27b>). I could not confirm whether the repository is gated, because huggingface.co is unreachable from here. Nothing I found says it is gated or removed. Community GGUF re-uploads exist (<https://huggingface.co/Abiray/harrier-oss-v1-27b-GGUF>).

| Item | Value | Tag and source |
|---|---|---|
| Publisher | Microsoft (Bing team), HF org `microsoft` | FACT: <https://the-decoder.com/microsofts-bing-team-open-sources-harrier-embedding-model/>, <https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model> (via search snippets) |
| Base model | **Google Gemma 3 27B (`google/gemma-3-27b-pt`)**, not Qwen | FACT: MTEB registry `adapted_from="google/gemma-3-27b-pt"` (harrier_models.py). A third-party Rust implementation loads it as `Gemma3TextModel` (<https://github.com/ikcore/ai-harrier-embedding>, README and `crates/harrier_core/src/registry.rs`) |
| Architecture | Decoder-only, last-token pooling, L2 normalisation | FACT: card text quoted in <https://github.com/unslothai/unsloth/issues/4984>; <https://labs.ai.azure.com/innovations/harrier-oss-v1/> (via search snippet) |
| Parameters | 27,008,663,808, of which 1,409,630,208 are embedding-table parameters | FACT: MTEB registry |
| Embedding dimension | 5,376 | FACT: MTEB registry; Rust SDK registry.rs |
| Max sequence length | 32,768 tokens per the card and the Rust SDK. MTEB metadata says `max_tokens=131072`, which is probably Gemma-3's `max_position_embeddings` | FACT: <https://github.com/ikcore/ai-harrier-embedding> (registry.rs); press: "32,000-token context window" (<https://the-decoder.com/microsofts-bing-team-open-sources-harrier-embedding-model/>). INFERENCE on the 131072 origin |
| Instruction format | Queries: `Instruct: {one-sentence task}\nQuery: {text}`. Documents: no instruction. Named prompts shipped include `web_search_query`, `sts_query` and `bitext_query` | FACT: MTEB `instruction_template` in harrier_models.py (with `apply_instruction_to_passages=False`); Rust SDK `task.rs`; HF card (via search snippet) |
| License | MIT | FACT: MTEB registry `license="mit"`; <https://the-decoder.com/microsofts-bing-team-open-sources-harrier-embedding-model/> |
| Release | Weights dated 2026-03-27 (MTEB `release_date`). Press coverage began 2026-03-30; Bing blog post April 2026 | FACT: harrier_models.py; MarkTechPost URL above |
| Languages | Card: "delivers strong … quality across **94 languages**". Bing blog and press: "more than 100 languages". The card lists languages "including but not limited to" a set that **includes Hebrew**, alongside Arabic, Persian, Hindi and others | FACT (via search snippet of the HF card, returned identically in two separate searches): <https://huggingface.co/microsoft/harrier-oss-v1-27b>. MTEB metadata uses the 100-language XLM-R list, which contains Hebrew |
| MMTEB (Multilingual, v2) | **Mean(Task) 74.27, Mean(TaskType) 64.20.** Retrieval 78.27, STS 79.99, Bitext 86.02, Classification 79.95, Clustering 58.93, Reranking 67.35 | FACT (computed from the results repo, see §0) |
| MMTEB rank | #1 by Mean(Task) and #1 by Borda among the roughly 50 strongest full-coverage models I loaded, as of 2026-09-22. It was also #1 at release | FACT: computation; <https://www.neowin.net/news/microsofts-new-harrier-models-top-benchmarks-outperforming-googles-gemini-embedding-2/> |
| Zero-shot share | 78%: part of the MMTEB training splits (MIRACL, MrTidy, FEVER, …) were in the training data | FACT: MTEB leaderboard "Zero-shot" column computed from `harrier_training_data` |
| Hebrew scores | Belebele retrieval nDCG@10: heb→heb 94.97, heb→eng 95.79, eng→heb 92.54. MASSIVE-intent (he) 82.13. Tatoeba heb-eng F1 93.58 | FACT (results repo JSONs) |
| vLLM | vLLM maps the `Gemma3TextModel` architecture to its embedding registry (`_EMBEDDING_MODELS["Gemma3TextModel"]`) and runs it causally unless `use_bidirectional_attention` is set. Current docs document `--runner pooling` / `--convert embed`; `--task embed` is the old interface and no longer appears in the docs. Harrier-0.6B has been served on vLLM 0.24.0. Open bug: PR #48214 fixes a CUDA index-out-of-bounds crash with nine concurrent ~32k-character inputs during chunked prefill. The newest vLLM on PyPI is 0.30.0 (2026-09-22) | FACT: <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py>, <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/config.py> (`Gemma3TextModelConfig`), <https://github.com/vllm-project/vllm/blob/main/docs/models/pooling_models/README.md>, <https://github.com/vllm-project/vllm/pull/48214>, <https://pypi.org/project/vllm/> |
| vLLM for the 27B specifically | No public report found of anyone serving the 27B on vLLM | INFERENCE: the architecture path exists, so `vllm serve microsoft/harrier-oss-v1-27b --runner pooling` should work. Test it on your patched build. vLLM does not add the `Instruct:` prefix; the client must prepend it to queries |
| VRAM | Weights: BF16 54.0 GB = **50.3 GiB**. FP8 ≈ **25.2 GiB** (≈26.5 GiB if the 1.41B-parameter embedding table stays in BF16). NVFP4 ≈ 16 GiB. Add activation and "KV" working memory on top | FACT: the Rust SDK says "~54 GB of bf16 memory" and MTEB says `memory_usage_mb=51515`. INFERENCE: the FP8 and NVFP4 figures are arithmetic. No official FP8 or NVFP4 checkpoint was found, so the quality loss is unmeasured |
| Throughput | No official throughput figures published | INFERENCE: about 54 GFLOP per token, roughly 45× a 0.6B model. Indexing 50M tokens at an effective 200 TFLOP/s takes about 3.75 h, against about 5 min for a 0.6B model. Its vectors are 21 KB each in fp32, so 1M chunks ≈ 20 GiB of vectors, against 3.8 GiB at 1,024 dimensions |

### Other Harrier sizes (all Microsoft, MIT, 32k context, same instruction format)

| Model | Base (origin) | Params | Dims | MMTEB Mean(Task) | Retrieval | Belebele heb→heb / heb→eng / eng→heb |
|---|---|---|---|---|---|---|
| harrier-oss-v1-270m | `google/gemma-3-270m` (Google) | 268M | 640 | 66.55 | 66.38 | 78.30 / 81.85 / 67.77 |
| harrier-oss-v1-0.6b | **`Qwen/Qwen3-0.6B` (Alibaba, PRC)** | 596M | 1,024 | 69.01 | 70.75 | 84.21 / 89.58 / 72.58 |
| harrier-oss-v1-27b | `google/gemma-3-27b-pt` (Google) | 27.0B | 5,376 | 74.27 | 78.27 | 94.97 / 95.79 / 92.54 |

- **FACT.** The bases come from the MTEB registry `adapted_from` fields (harrier_models.py). The Rust SDK agrees: "`Qwen3Model` (0.6B) and `Gemma3TextModel` (270M, 27B)" (<https://github.com/ikcore/ai-harrier-embedding>). The scores are computed from the results repo.
- **INFERENCE.** harrier-oss-v1-0.6b is itself a Qwen model retrained by a non-PRC organisation. It has the same provenance status as Ornith 1.5.

### Microsoft's other open embedding families

| Model | Base | Params / dims / context | License | MMTEB | Hebrew (Belebele heb→heb) | Sources |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-large-instruct | XLM-RoBERTa-large (Meta) | 560M / 1,024 / **514 tokens** | MIT | 63.22 (Retrieval 57.11) | **91.54** (heb→eng 91.91, eng→heb 82.21) | FACT: MTEB registry (e5_instruct.py); <https://github.com/microsoft/unilm/tree/master/e5>; <https://arxiv.org/abs/2402.05672> (authors at Microsoft Corporation) |
| intfloat/multilingual-e5-large / base / small | XLM-R / MiniLM | 560M / 278M / 118M, 512 tokens | MIT | — (not every MMTEB task was run) | large: 92.54 | FACT: MTEB registry and results repo |
| intfloat/e5-mistral-7b-instruct | Mistral-7B-v0.1 (France) | 7.1B / 4,096 / 32k | MIT | 60.25 | 79.48 | FACT: MTEB registry and results repo. The E5 README lists it as the LLM-based model (<https://github.com/microsoft/unilm/blob/master/e5/README.md>) |

- **FACT.** No Microsoft embedding release newer than Harrier-OSS-v1 was found. Searches for a "harrier-oss-v2" return nothing (<https://blogs.bing.com/search/April-2026/Microsoft-Open-Sources-Industry-Leading-Embedding-Model>).
- **INFERENCE.** "intfloat" is the personal Hugging Face account of the Microsoft E5 authors. It is not the official `microsoft` org that Harrier uses.

---

## 2. "A Qwen embedding retrained by Taiwanese or Singaporeans"

**FACT.** Qwen3-Embedding (0.6B / 4B / 8B; 1,024 / 2,560 / 4,096 dimensions; 32k context; Apache-2.0; "over 100 languages") is published by Alibaba's Qwen team, which is a PRC organisation. It is refused under your rule. Its MMTEB scores are 64.33 / 69.45 / 70.58 (<https://github.com/QwenLM/Qwen3-Embedding>, <https://huggingface.co/Qwen/Qwen3-Embedding-8B>).

### Most likely the model you are thinking of: Octen-Embedding (Octen, San Francisco and Singapore)

- **FACT.** Octen describes itself as "headquartered in both San Francisco and Singapore". It was founded by Kuan Zou, formerly head of AI search at Alibaba Cloud and earlier at Baidu. Its team includes people from Meta, Google, TikTok, Alibaba, Baidu, DeepSeek and Xiaohongshu. It raised a US$10M seed round led by Square Peg in April 2026. Sources: <https://www.prnewswire.com/news-releases/octen-sets-new-global-benchmark-for-search-infrastructure-launches-worlds-fastest-web-search-api-for-the-agentic-era-302749728.html>, <https://siliconangle.com/2026/04/22/octen-raises-10m-seed-funding-speed-ai-agent-search-queries/>, <https://www.squarepeg.vc/blog/investment-notes-octen-us-10m-seed>.
- **FACT.** The models are Octen-Embedding-0.6B, 4B and 8B (plus INT8 variants). They are fine-tuned **directly from `Qwen/Qwen3-Embedding-*`**, carry an Apache-2.0 license, have 32k context, and keep Qwen3's 1,024 / 2,560 / 4,096 dimensions. Their MTEB metadata uses the Qwen3 73-language list, which includes Hebrew. Octen-8B was #1 on the RTEB retrieval leaderboard in early 2026. Sources: <https://huggingface.co/Octen/Octen-Embedding-8B>, <https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/octen_models.py>.
- **FACT (computed).** Octen-8B scores MMTEB Mean(Task) 67.84 and Retrieval 71.61. These cover only 129 of 131 tasks (AILAStatutes and MIRACLRetrievalHardNegatives were not submitted), so they are not an official leaderboard score. Belebele Hebrew: heb→heb 96.78, heb→eng 99.14, eng→heb 96.30, the best of any non-PRC-published open model here. Octen-0.6B and Octen-4B have no MMTEB runs.
- **INFERENCE.** This is the best match for "a Qwen embedding retrained by Singaporeans". It sits closer to the line than Ornith does:
  - It is a fine-tune of the Chinese *embedding* model itself, not a continued pre-train of a base LLM.
  - Its founder comes from Alibaba Cloud search.

  By the letter of your rule (publisher is non-PRC) it passes. By the spirit of the rule, have it signed off explicitly. The architecture is unchanged Qwen3, so vLLM should serve it the same way it serves Qwen3-Embedding (`Qwen3Model` / `Qwen3ForCausalLM` are listed in <https://github.com/vllm-project/vllm/blob/main/docs/models/pooling_models/embed.md>).

### Taiwan and Singapore: what actually exists

- **FACT.** No Taiwanese Qwen-based or other embedding model was found. MediaTek Research's Breeze models are Mistral-based LLMs (<https://github.com/mtkresearch/MR-Models>). TAIDE and Academia Sinica returned no embedding models. The MTEB registry contains no Taiwanese entry.
- **FACT.** AI Singapore released SEA-LION-ModernBERT-Embedding-300M and 600M. They are ModernBERT trained from scratch with the Gemma-3 tokenizer, so they are **not** Qwen. They have 8,192-token context and cover Burmese, Chinese, English, Filipino, Indonesian, Khmer, Lao, Malay, Tamil, Thai and Vietnamese, with **no Hebrew**. Sources: <https://huggingface.co/aisingapore/SEA-LION-ModernBERT-Embedding-600M>, <https://sea-lion.ai/blog/bridging-the-semantic-gap-announcing-the-sea-lion-embedding-suite/>, <https://arxiv.org/abs/2606.03027> (via search snippets).
- **FACT.** Sea AI Lab's Sailor and Sailor2 are Qwen-based LLMs, not embedders (<https://github.com/sail-sg/sailor2>).

### Other Qwen-derived embedders from non-PRC publishers

| Model | Org / country | Base (origin) | Params / dims / context | License | MMTEB | Hebrew Belebele (h→h / h→e / e→h) | vLLM | Sources |
|---|---|---|---|---|---|---|---|---|
| harrier-oss-v1-0.6b | Microsoft / US | Qwen3-0.6B (PRC) | 0.6B / 1,024 / 32k | MIT | 69.01 | 84.21 / 89.58 / 72.58 | Yes: served on vLLM 0.24.0 (PR #48214) | see §1 |
| pplx-embed-v1-0.6b and 4b | Perplexity / US | Qwen3-0.6B and Qwen3-4B base (PRC), converted to bidirectional with diffusion pretraining | 0.6B / 1,024 and 4B / 2,560, 32k | MIT | Retrieval only: 65.41 (0.6B), 69.66 (4B) | 88.06 / 87.45 / 71.27 (0.6B); 87.11 / 91.39 / 73.75 (4B) | Not listed in vLLM docs (custom bidirectional) | <https://research.perplexity.ai/articles/pplx-embed-state-of-the-art-embedding-models-for-web-scale-retrieval>; MTEB perplexity_models.py |
| jina-embeddings-v5-text-small | Jina AI (Berlin), owned by Elastic (US) since 2025-10-09 | Qwen3-0.6B-Base, distilled from Qwen3-Embedding-4B | 0.6B / 1,024 (Matryoshka) / 32k | **CC-BY-NC-4.0** | 67.00 | 89.44 / 93.26 / 85.35 | Yes, as `JinaEmbeddingsV5Model` | <https://huggingface.co/jinaai/jina-embeddings-v5-text-small>; <https://www.businesswire.com/news/home/20251009619654/en/Elastic-Completes-Acquisition-of-Jina-AI-a-Leader-in-Frontier-Models-for-Multimodal-and-Multilingual-Search>; vLLM embed.md |
| voyage-4-nano | Voyage AI (MongoDB) / US | Bidirectional Qwen3 | 346M / 2,048 / 32k | Apache-2.0 | no full run | no data | Yes, as `VoyageQwen3BidirectionalEmbedModel` | <https://huggingface.co/voyageai/voyage-4-nano>; vLLM tests/models/registry.py |
| Qwen3-Embedding-Scandi-0.6B, dinghy-law, Euler-Legal, RTriever-4B, Querit | assorted (DK, US, …) | Qwen3-Embedding | — | mostly Apache-2.0 | domain or language-specific, no Hebrew | — | — | MTEB registry |

### Non-Qwen open candidates (non-PRC publisher)

| Model | Org / country | Base (origin) | Params / dims / context | Hebrew listed? | License | MMTEB | Hebrew Belebele h→h | vLLM | Sources |
|---|---|---|---|---|---|---|---|---|---|
| llama-embed-nemotron-8b | NVIDIA / US | Llama-3.1-8B (Meta) | 7.5B / 4,096 / 32k | Not in MTEB's evaluated-language list | **Non-commercial** (customized NSCLv1) | 69.46 | 94.85 | Likely, via `LlamaBidirectionalModel` (INFERENCE) | <https://huggingface.co/nvidia/llama-embed-nemotron-8b>; <https://huggingface.co/blog/nvidia/llama-embed-nemotron-8b> |
| Nemotron-3-Embed-8B and 1B (July 2026) | NVIDIA / US | Ministral-3-8B / 3B (Mistral, France), converted to bidirectional | 8B / 4,096; 1.1B / 2,048; 32k | **No** (34 languages, Hebrew absent) | OpenMDW-1.1 | no full run; #1 on RTEB | no data | Not listed in vLLM docs | <https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16>; <https://huggingface.co/blog/nvidia/nemotron-3-embed-wins-rteb> |
| EmbeddingGemma-300m | Google / US | Gemma 3 (T5Gemma init) | 308M / 768 (MRL) / **2,048** | "100+ spoken languages" | Gemma terms | 61.15 | 73.96 (weak) | Yes, as `Gemma3TextModel` | <https://huggingface.co/google/embeddinggemma-300m>; vLLM embed.md |
| granite-embedding-311m-multilingual-r2 (2026-04-29) | IBM / US | ModernBERT-style, IBM-trained | 311M / 768 (MRL) / 32,768 | 200+ languages, 52 "enhanced"; the MTEB list of the 52 includes Hebrew | Apache-2.0 | 55.96 | 81.08 | Probably, via `ModernBertModel` (INFERENCE) | <https://github.com/ibm-granite/granite-embedding-models> |
| snowflake-arctic-embed-l-v2.0 | Snowflake / US | `BAAI/bge-m3-retromae` (**PRC BAAI lineage**) | 568M / 1,024 / 8k | Yes | Apache-2.0 | 57.03 | 89.76 | Yes, as `XLMRobertaModel` | MTEB registry |
| nomic-embed-text-v2-moe | Nomic / US | nomic-xlm-2048 | 475M / 768 / **512** | Yes | Apache-2.0 | 57.64 | 90.36 | Yes, as `NomicBertModel` | MTEB registry; vLLM embed.md |
| BidirLM-1B-Embedding | French academic group (Boizard, Colombo et al.) | Gemma-3-1B | 1.0B / 1,152 / 32k | Yes | Apache-2.0 | 63.21 | 87.32 | not verified | <https://arxiv.org/abs/2604.02045>; MTEB registry |

**FACT.** Refused by publisher under your rule, and listed only for reference: KaLM-Embedding-Gemma3-12B (Tencent, MMTEB 72.32, a Gemma base with a PRC publisher), F2LLM-v2 (Ant Group), bge-m3 (BAAI) and Seed1.6-embedding (ByteDance, closed). Scores are from the results repo.

**FACT.** Closed models, for reference only: gemini-embedding-001 (MMTEB 68.37, Hebrew Belebele 95.76), OpenAI text-embedding-3-large (Hebrew 79.84), Cohere embed-v4 and Voyage-4 (no full MMTEB run). Source: the results repo.

---

## 3. Hebrew-specific evaluation: what exists

- **FACT.** MTEB has **no Hebrew benchmark**: no "MTEB(heb)" exists in `mteb/benchmarks`. Hebrew appears only as a subset of multilingual tasks (<https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks>):
  - Retrieval: BelebeleRetrieval (heb↔heb, heb↔eng), MKQARetrieval (he), WebFAQRetrieval (heb).
  - Classification: HebrewSentimentAnalysis(.v3), MassiveIntent and MassiveScenario (he), SIB200 and MultilingualSentiment.
  - Clustering: SIB200ClusteringS2S.
  - Bitext mining: Flores, NTREX, Tatoeba, BibleNLP, WebFAQ.
- **FACT.** Inside MMTEB v2, Hebrew contributes only to Belebele, MassiveIntent, SIB200 clustering and bitext mining. A model's MMTEB average says little about Hebrew specifically.
- **FACT.** Belebele is small: 900 questions per language variant (<https://arxiv.org/abs/2308.16884>). It is also near saturation for the top models (95–97 nDCG@10).
- **FACT.** Hebrew community evaluations:
  - The **Hebrew Semantic Retrieval National Challenge** (MAFAT / Israel National NLP Program, on Codabench): a Hebrew-only retrieval benchmark scored with NDCG@20 (<https://www.codabench.org/competitions/9950/>, via search snippet; the page is unreachable from here).
  - Dicta's `neodictabert-bilingual-embed` (Israel; 768 dimensions; 4,096 context; CC-BY-4.0) reports #10 on the challenge's private phase (<https://huggingface.co/dicta-il/neodictabert-bilingual-embed>, via search snippet).
  - Sefaria's **Rabbinic-Embedding-Leaderboard** covers Rabbinic, not modern, Hebrew (<https://huggingface.co/datasets/Sefaria/Rabbinic-Embedding-Leaderboard>).
- **FACT (computed from the results repo).** Hebrew subset scores for the main candidates:

| Model | Belebele h→h | h→e | e→h | MASSIVE-intent he (accuracy) | Tatoeba heb-eng F1 |
|---|---|---|---|---|---|
| Octen-Embedding-8B | 96.78 | 99.14 | 96.30 | 79.38 | 90.02 |
| harrier-oss-v1-27b | 94.97 | 95.79 | 92.54 | 82.13 | 93.58 |
| llama-embed-nemotron-8b | 94.85 | 96.75 | 93.61 | 77.16 | 88.85 |
| multilingual-e5-large-instruct | 91.54 | 91.91 | 82.21 | 63.50 | 91.52 |
| jina-embeddings-v5-text-small | 89.44 | 93.26 | 85.35 | 84.90 | 73.23 |
| pplx-embed-v1-0.6b | 88.06 | 87.45 | 71.27 | — | — |
| harrier-oss-v1-0.6b | 84.21 | 89.58 | 72.58 | 63.95 | 85.44 |
| granite-embedding-311m-multilingual-r2 | 81.08 | 79.06 | 58.09 | 48.52 | 65.24 |
| embeddinggemma-300m | 73.96 | 84.45 | 62.43 | 52.59 | 51.64 |
| *(PRC, reference)* Qwen3-Embedding-8B | 96.86 | 98.78 | 95.98 | 79.32 | 89.93 |

**INFERENCE.** Among small models, multilingual-e5-large-instruct is noticeably stronger on Hebrew than harrier-0.6b, even though harrier-0.6b has the far better MMTEB average. The public data cannot settle Hebrew wiki retrieval quality; you need an in-house Hebrew and English query set.

---

## 4. Comparison table (open weights, publisher not PRC)

VRAM figures are weights only, computed as parameters × 2 bytes (BF16) and × 1 byte (FP8). Runtime memory comes on top. The FP8 figures are INFERENCE.

| Model | Org / country | Base origin | Params | Dims | Context | Hebrew | License | MMTEB Mean(Task) | Weights BF16 / FP8 | vLLM |
|---|---|---|---|---|---|---|---|---|---|---|
| harrier-oss-v1-27b | Microsoft / US | Gemma-3-27B (Google, US) | 27.0B | 5,376 | 32k | Listed on card; Belebele 95.0 | MIT | **74.27** | 50.3 / 25.2–26.5 GiB | Architecture supported (`Gemma3TextModel`); not tested publicly |
| Octen-Embedding-8B | Octen / US + Singapore | Qwen3-Embedding-8B (Alibaba, **PRC**) | 7.57B | 4,096 | 32k | Yes; Belebele 96.8 | Apache-2.0 | 67.84* (129/131 tasks) | 14.1 / 7.0 GiB | Yes (Qwen3 architecture) |
| llama-embed-nemotron-8b | NVIDIA / US | Llama-3.1-8B (Meta, US) | 7.5B | 4,096 | 32k | Not listed; Belebele 94.9 | **Non-commercial** | 69.46 | 14.0 / 7.0 GiB | Likely |
| Nemotron-3-Embed-8B | NVIDIA / US | Ministral-3-8B (Mistral, FR) | 7.95B | 4,096 | 32k | **No** | OpenMDW-1.1 | n/a | 14.8 / 7.4 GiB | Not in docs |
| pplx-embed-v1-4b | Perplexity / US | Qwen3-4B (**PRC**) | 4.0B | 2,560 | 32k | Yes; Belebele 87.1 | MIT | Retrieval 69.66 only | 7.5 / 3.7 GiB | Not in docs |
| harrier-oss-v1-0.6b | Microsoft / US | Qwen3-0.6B (**PRC**) | 0.6B | 1,024 | 32k | Yes; Belebele 84.2 | MIT | 69.01 | 1.1 / 0.6 GiB | Yes (served on 0.24.0) |
| jina-emb-v5-text-small | Jina (DE) / Elastic (US) | Qwen3-0.6B (**PRC**) | 0.6B | 1,024 | 32k | Yes; Belebele 89.4 | **CC-BY-NC-4.0** | 67.00 | 1.1 / 0.6 GiB | Yes |
| harrier-oss-v1-270m | Microsoft / US | Gemma-3-270m (Google) | 0.27B | 640 | 32k | Yes; Belebele 78.3 | MIT | 66.55 | 0.5 / 0.25 GiB | Yes (`Gemma3TextModel`) |
| multilingual-e5-large-instruct | Microsoft / US | XLM-R-large (Meta) | 0.56B | 1,024 | **514** | Yes; Belebele 91.5 | MIT | 63.22 | 1.0 / 0.5 GiB | Yes (`XLMRobertaModel`) |
| embeddinggemma-300m | Google / US | Gemma 3 | 0.31B | 768 | 2,048 | Yes; Belebele 74.0 | Gemma | 61.15 | 0.6 / 0.3 GiB | Yes |
| granite-emb-311m-multi-r2 | IBM / US | IBM ModernBERT | 0.31B | 768 | 32k | Yes; Belebele 81.1 | Apache-2.0 | 55.96 | 0.6 / 0.3 GiB | Probably (`ModernBertModel`) |

Sources for this table: §1–§3 above. Every MMTEB and Belebele number is computed from <https://github.com/embeddings-benchmark/results> at commit `fe1eb05`.

---

## 5. Recommendation for offline Hebrew + English RAG over Markdown wikis

1. **Best quality with clean provenance: harrier-oss-v1-27b.** Use it if you can give it a card, or about 30 GiB of one.
   - FACT: it is #1 on MMTEB (74.3; retrieval 78.3), strong on Hebrew (Belebele 95.0 / 95.8 / 92.5), MIT-licensed, and built on Google Gemma 3 with no Chinese lineage.
   - INFERENCE on the costs:
     - On a shared 96 GiB RTX PRO 6000, FP8 weights (about 26 GiB) plus runtime memory take roughly 30% of the card away from LLM weights and KV cache.
     - Indexing compute is about 45× that of a 0.6B model.
     - Vectors are 5.25× larger than at 1,024 dimensions.
     - No official FP8 or NVFP4 checkpoint exists. Validate the FP8 quality loss yourself, or run BF16 (about 50 GiB).
     - A dedicated small card, or the B200, is the natural home for it.
2. **If your policy officer accepts a direct Qwen3-Embedding fine-tune from a non-PRC company: Octen-Embedding-8B, or 4B.**
   - FACT: it has the best Hebrew scores of any open non-PRC-published model, is Apache-2.0, weighs 14 GiB in BF16, and uses the standard Qwen3 architecture in vLLM.
   - INFERENCE: it is provenance-wise weaker than Ornith, because it is the Chinese embedder lightly re-tuned by an ex-Alibaba team. Get an explicit decision before using it.
3. **Cheap default, under 2 GiB, both from Microsoft and MIT-licensed:**
   - **multilingual-e5-large-instruct** is best on Hebrew per GiB. Its 512-token limit is fine for chunked Markdown (INFERENCE), but its English and multilingual retrieval is an older generation (retrieval 57.1).
   - **harrier-oss-v1-0.6b** has 32k context and strong overall scores (retrieval 70.8) but weaker Hebrew (Belebele 84.2). It is Qwen-lineage, so it has the same status as Ornith.
   - Run both against your own queries.
4. **Avoid, or only if the license allows:**
   - jina-v5-text-small: CC-BY-NC.
   - llama-embed-nemotron-8b: non-commercial.
   - Nemotron-3-Embed: Hebrew is not a supported language.
   - EmbeddingGemma-300m: weak Hebrew and 2k context.
   - Granite R2: weak Hebrew numbers.

**INFERENCE (practical).**
- Build a 200–300-query Hebrew and English gold set from your actual wikis before choosing. The public Hebrew signal is one small task, Belebele.
- Whatever model you pick:
  - Prepend `Instruct: …\nQuery: ` to queries client-side for Harrier, Octen and Qwen-style models, because vLLM does not add it.
  - Cap `--max-model-len` at your chunk size, for example 2,048–8,192, rather than 32k. This saves memory and avoids the long-input chunked-prefill bug (vLLM PR #48214).
  - Re-index the whole corpus if you change embedders; vectors are not portable between models.
