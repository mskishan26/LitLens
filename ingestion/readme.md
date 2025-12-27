# Ingestion Pipeline Documentation

This directory contains the scripts and modules for the data ingestion pipeline of the RAG system. The pipeline processes academic PDF documents, extracts text, cleans it, generates summaries, and creates embeddings for retrieval.

## Scripts Overview

### 1. Page Rotation Correction (`page_rotation.py`)
Detects and corrects the orientation of PDF pages using Tesseract OSD. This ensures that all pages are upright before text extraction.

**Usage:**
```bash
python page_rotation.py --input_dir /path/to/input_pdfs --output_dir /path/to/rotated_pdfs [options]
```

**Arguments:**
- `--input_dir`: Directory containing input PDF files (Required).
- `--output_dir`: Directory to save corrected PDFs (Required).
- `--confidence`: Confidence threshold for orientation detection (0-15, default: 5.0). Higher values reduce false positives.
- `--workers`: Number of parallel worker processes (default: CPU cores - 2).
- `--dpi`: DPI for page rendering during detection (default: 300).
- `--single_file`: Process a single file instead of a directory.

---

### 2. Marker PDF Extraction (`marker_extraction.py`)
Extracts text and layout information from PDFs using the Marker library. It converts PDFs to Markdown format.

**Usage:**
```bash
python marker_extraction.py --input_dir /path/to/rotated_pdfs --output_dir /path/to/markdown_output
```

**Arguments:**
- `--input_dir`: Directory containing PDF files (Required).
- `--output_dir`: Directory to save output Markdown and JSON files (Required).

**Environment Variables:**
- `MARKER_MODEL_CACHE`: Path to cache Marker models (default: `/scratch/sathishbabu.ki/marker-models`).
- `MARKER_LOG_DIR`: Path for logs.

---

### 3. Noise Classification (`noise_classifer.py`)
Filters the extracted Markdown content to remove irrelevant text such as running headers, footers, and page numbers, keeping only the main academic content.

**Usage:**
```bash
python noise_classifer.py --input_dir /path/to/markdown_output --output_dir /path/to/clean_markdown
```

**Arguments:**
- `--input_dir`: Directory containing `*.md` files (Required).
- `--output_dir`: Directory to save cleaned Markdown files (Required).

**Model:** Uses `microsoft/Phi-4-mini-instruct` for classification.

---

### 4. Summary Generation (`summary_generator.py`)
Generates comprehensive summaries of the research papers using a large language model. It supports direct summarization for shorter papers and recursive chunked summarization for longer ones.

**Usage:**
```bash
python summary_generator.py --input_dir /path/to/clean_markdown --output_dir /path/to/summaries [options]
```

**Arguments:**
- `--input_dir`: Directory containing `*.md` files (Required).
- `--output_dir`: Directory to save summary Markdown files (Required).
- `--no-skip-existing`: Force regeneration even if a summary already exists.
- `--max-context`: Override the model's max context window size.
- `--output-budget`: Override the output token budget.

**Model:** Uses `microsoft/phi-4-mini-instruct`.

---

### 5. BM25 Index Generation (`bm25_generator.py`)
Creates a BM25 index from the processed Markdown documents for keyword-based retrieval.

**Usage:**
```bash
python bm25_generator.py --input_dir /path/to/clean_markdown --output_dir /path/to/artifacts
```

**Arguments:**
- `--input_dir`: Directory containing `*.md` files (Required).
- `--output_dir`: Directory to save BM25 artifacts (index, corpus, metadata) (Required).

---

### 6. Embedding Generation (`embedding_generator.py`)
Generates vector embeddings for the documents using a two-stage approach:
1.  **Stage 1 (Paper Level):** Large chunks for high-level retrieval.
2.  **Stage 2 (Chunk Level):** Smaller semantic chunks for detailed retrieval.

**Usage:**
```bash
python embedding_generator.py --stage1_input_dir /path/to/stage1_md --stage2_input_dir /path/to/stage2_md --output_dir /path/to/embeddings
```

**Arguments:**
- `--stage1_input_dir`: Directory containing Markdown files for Stage 1 (Required).
- `--stage2_input_dir`: Directory containing Markdown files for Stage 2 (Required).
- `--output_dir`: Directory to save FAISS indices and metadata (Required).

**Model:** Uses `Qwen/Qwen3-Embedding-8B`.

---

## Pipeline Orchestration

### `ingestion_pipeline.sh`
A shell script that automates the entire ingestion process, running the above scripts in the correct order.

### `ingestion_pipeline_interactive.sh`
An interactive version of the pipeline script, likely allowing for step-by-step execution or user input.