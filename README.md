# LitLens: Local RAG Capstone Project

**A State-of-the-Art, Fully Local, Memory-Optimized Retrieval Augmented Generation System**

Welcome to the core codebase of our advanced RAG system. This project represents a leap forward in local document intelligence, designed to process, index, and query vast repositories of academic literature without a single byte leaving your infrastructure.

## Key Highlights

*   **100% Local & Private:** Built to run entirely on your hardware. No APIs, no data leaks, total privacy.
*   **Memory Optimized:** Engineered with a sophisticated sequential loading architecture. We maximize performance on limited hardware by dynamically managing VRAM, allowing powerful models to run on standard compute resources.
*   **Production Ready:** From robust ingestion pipelines handling PDF noise to a high-throughput inference engine, every component is built for stability and scale.
*   **State-of-the-Art Models:** Leverages the latest in open-source AI (Qwen, Phi, etc.) for embedding, reranking, and generation.

---

## The Funnel Architecture

We approach the challenge of "finding a needle in a haystack" through a highly efficient, multi-stage funnel. Instead of overwhelming the LLM with context, we progressively narrow down the search space to ensure maximum precision and relevance.

1.  **The Wide Net (Stage 1: Paper Selection)**
    *   *Input:* The entire corpus of N papers.
    *   *Action:* We use a Hybrid Search (BM25 Keyword + Dense Embeddings) to identify the top `k` most relevant **papers**.
    *   *Result:* A focused subset of documents most likely to contain the answer.

2.  **The Deep Dive (Stage 2: Chunk Retrieval)**
    *   *Input:* The selected `k` papers.
    *   *Action:* We perform a granular dense retrieval search within these papers to find the top `m` specific **paragraphs/chunks**.
    *   *Result:* High-resolution context fragments.

3.  **The Precision Filter (Stage 3: Reranking)**
    *   *Input:* The top `m` chunks.
    *   *Action:* A powerful Cross-Encoder (Reranker) scores every query-chunk pair to filter out false positives and rank them by true semantic relevance.
    *   *Result:* The top `n` "gold standard" chunks.

4.  **The Synthesis (Stage 4: Generation)**
    *   *Input:* The top `n` chunks.
    *   *Action:* Our Generative LLM synthesizes these facts into a coherent, cited, and accurate answer.
    *   *Result:* A final response grounded in truth.

---

## Project Structure

This repository is organized into two main logical components, each with its own detailed documentation:

### 1. [Ingestion Pipeline](./ingestion/readme.md)
*Located in `src/ingestion/`*
*   **Purpose:** Turning raw PDFs into a searchable, clean knowledge base.
*   **Features:** OCR correction, noise classification, layout analysis, summarization, and vector indexing.
*   **Read the full guide inside the folder.**

### 2. [Inference Engine](./inference/readme.md)
*Located in `src/inference/`*
*   **Purpose:** The brain of the system. Handles the 4-stage retrieval and chat interface.
*   **Features:** CLI chat, streaming responses, hybrid search, and sequential model management.
*   **Read the full guide inside the folder.**

---

*Built with performance, privacy, and precision in mind.*
