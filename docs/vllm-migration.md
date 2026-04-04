# Switching from Ollama to vLLM

## Why Switch?

| Metric | Ollama | vLLM |
|---|---|---|
| Concurrent requests | Queued (sequential) | Continuous batching (parallel) |
| Memory management | Static KV cache | PagedAttention (3-5x more concurrent requests) |
| Single request speed (4090) | ~30 tok/s | ~50-80 tok/s |
| 5 concurrent users | Each waits in queue (~5x latency) | Batched together (~1.5x latency) |
| API | OpenAI-compatible | OpenAI-compatible (same) |

**Bottom line**: With 10+ concurrent students chatting, Ollama queues them one by one. vLLM serves them in parallel with continuous batching.

---

## What Changes in the Code

### Total files touched: 2 (config + factory)

### 1. `app/config/settings.py`

```diff
- ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
- ollama_model: str = Field(default="llama3.2:3b", validation_alias="OLLAMA_MODEL")
+ llm_base_url: str = Field(default="http://localhost:8000/v1", validation_alias="LLM_BASE_URL")
+ llm_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", validation_alias="LLM_MODEL")
```

### 2. `app/llm/factory.py`

```diff
- from llama_index.llms.ollama import Ollama
+ from llama_index.llms.openai_like import OpenAILike

- _llm: Ollama | None = None
+ _llm: OpenAILike | None = None

- def get_llm() -> Ollama:
+ def get_llm() -> OpenAILike:
      global _llm
      if _llm is None:
-         _llm = Ollama(
-             model=settings.ollama_model,
-             base_url=settings.ollama_base_url,
-             request_timeout=600.0,
-             additional_kwargs={"num_ctx": 4096}
-         )
+         _llm = OpenAILike(
+             model=settings.llm_model,
+             api_base=settings.llm_base_url,
+             api_key="not-needed",          # vLLM doesn't require auth by default
+             timeout=600.0,
+             is_chat_model=True,
+             context_window=8192,            # llama3.1 supports up to 128k
+         )
      return _llm
```

### 3. `requirements.txt` / `pyproject.toml`

```diff
- llama-index-llms-ollama
+ llama-index-llms-openai-like
```

That's it. **No other files change.** The rest of the pipeline (condenser, synthesizer, quiz formatter) all call `get_llm()` — they don't care whether the LLM behind it is Ollama or vLLM.

---

## How to Run vLLM on Your Server

```bash
# Install vLLM
pip install vllm

# Start the server (uses your RTX 4090 GPU)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enable-prefix-caching
```

### Key vLLM flags:
- `--gpu-memory-utilization 0.85` — use 85% of your 12GB VRAM (~10.2GB)
- `--max-model-len 8192` — limits context window to save VRAM
- `--enable-prefix-caching` — reuses computation for shared system prompts (your RAG prompts all start the same)

### Environment variables for RAG engine:
```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

---

## What Stays the Same

- **Embeddings**: FastEmbed (CPU-only, no GPU) — completely unaffected
- **Reranker**: BAAI/bge-reranker-base (CPU) — completely unaffected
- **Qdrant**: Vector search — unaffected
- **NATS/Workers**: Message queue — unaffected
- **All pipeline logic**: Condenser, hybrid retriever, quiz generation — all just call `get_llm()`

---

## When to Switch

- **Stay on Ollama** during development and testing (simpler to manage)
- **Switch to vLLM** when deploying for real users (especially 10+ concurrent)
- The switch is reversible — just change the 2 files back
