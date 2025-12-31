# Inference Pipeline Documentation
## Please ensure you have set the cache location or its unchanged from previous run or the model will download again causing errors or disk space constraints.

This directory contains the scripts and modules for the RAG inference pipeline. It handles retrieval, reranking, and generation of answers using the processed data.

## Scripts Overview

### 1. CLI Chat Interface (`cli_chat.py`)
The main entry point for interacting with the RAG system via the command line. It initializes the pipeline and provides a streaming chat interface.

**Usage:**
```bash
python cli_chat.py [options]
```

**Arguments:**
- `--no-unload`: Keep models loaded in memory between queries. Useful for faster response times if you have sufficient VRAM (approx. 80GB+ for full pipeline). By default, models are loaded and unloaded sequentially to save memory.

**Features:**
- Streaming responses.
- Displays source citations (Paper Title and Filename).
- Logs session details (queries, answers, timings, rankings) to `data/session_logs/`.

---

### 2. Chat Pipeline (`chat_pipeline.py`)
Orchestrates the full 4-stage RAG process. It manages model loading/unloading and data flow between stages.

**Pipeline Stages:**
1.  **Hybrid Retrieval:** Combines BM25 (keyword) and Embedding (semantic) search at the paper level to select the top `k` relevant papers.
2.  **Chunk Retrieval:** Searches for the top `m` relevant chunks within the selected papers using dense embeddings.
3.  **Reranking:** Uses a cross-encoder (Qwen) to rerank the `m` chunks and select the top `n` most relevant ones.
4.  **Generation:** Feeds the top `n` chunks as context to the LLM (Qwen) to generate the final answer.

**Key Methods:**
- `answer(query, ...)`: Returns the complete answer and metadata.
- `answer_streaming(query, ...)`: Yields tokens for real-time display.

---

### 3. BM25 Search (`bm25_search.py`)
Handles keyword-based retrieval using the pre-built BM25 index.

**Usage:**
```python
from inference.bm25_search import BM25Searcher
searcher = BM25Searcher(artifacts_dir="...")
searcher.load_bm25_artifacts()
results = searcher.search("query", k=10)
```

---

### 4. Embedding Search (`embedding_search.py`)
Handles semantic retrieval using vector embeddings (FAISS indices). Supports both paper-level (Stage 1) and chunk-level (Stage 2) search.

**Usage:**
```python
from inference.embedding_search import EmbeddingSearch
searcher = EmbeddingSearch()
searcher.load("path/to/embeddings")
results = searcher.search("query", index_num=1, k=10)
```

---

### 5. Reranker (`reranker.py`)
Implements the reranking stage using a Qwen-based cross-encoder. It scores the relevance of query-chunk pairs to improve precision.

**Usage:**
```python
from inference.reranker import Reranker
reranker = Reranker()
reranked_results = reranker.rerank(query, candidates, top_k=5)
```

---

### 6. Generator (`generator.py`)
Manages the Large Language Model (LLM) for the final answer generation. It constructs the prompt with retrieved contexts and handles streaming output.

**Usage:**
```python
from inference.generator import QwenGenerator
generator = QwenGenerator()
response = generator.generate(query, contexts)
```

---

## Configuration

The pipeline relies on a central configuration file (typically `config.yaml`) loaded via `utils.config_loader`. Key parameters include:
- **Paths:** Locations of embeddings, BM25 artifacts, and logs.
- **Models:** HuggingFace model names for embeddings, reranker, and generator.
- **Retrieval:** Parameters `k` (papers), `m` (chunks), `n` (reranked), and weights for hybrid search.
- **Generation:** Temperature, max tokens, etc.

## Memory Management

To support running on limited hardware (e.g., single A100 40GB or smaller), the pipeline supports **Sequential Loading**:
- Models are loaded only when needed for their specific stage.
- They are immediately unloaded (and GPU memory cleared) after use.
- Use the `--no-unload` flag in `cli_chat.py` to disable this behavior for higher performance on large-memory systems.
