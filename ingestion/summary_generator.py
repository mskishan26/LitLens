#!/usr/bin/env python3
"""
Minimal, production-grade summarizer (refactor of summary_generator.py).

Key points:
- Assumes configs are provided from elsewhere (see CONFIG STUBS).
- Tokenizer is a utility using tiktoken (cached encoder).
- Model loading is lazy; GPU memory auto-detected via torch if available, else CLI fallback.
- Chunking, direct vs chunked summarization, synthesis preserved.
- Simple file I/O helpers with atomic write and skip-existing support.
- Logging via standard logging module.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from utils.logger import get_ingestion_logger
import torch

# External libs used by original code
try:
    import tiktoken
except Exception as e:
    raise RuntimeError("tiktoken required: pip install tiktoken") from e

try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise RuntimeError("vllm required: pip install vllm") from e

# Optional: detect GPU memory
def detect_gpu_memory_gb() -> Optional[int]:
    """Return GPU memory (GB) if a CUDA-capable GPU is available via torch; else None."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # total_memory is bytes
            total_gb = int(props.total_memory // (1024 ** 3))
            return max(40, total_gb)  # normalize small GPUs to 40 as default lower bound
    except Exception:
        pass
    return None


# =========================
# CONFIG (stub - replace)
# =========================
# Replace the dataclasses below with imports from your project's config module.
@dataclass
class SummaryModelConfig:
    name: str = "microsoft/phi-4-mini-instruct"
    max_context: int = 16384
    token_budget_output: int = 2500
    token_buffer: int = 1000

    @property
    def max_input_tokens(self) -> int:
        return self.max_context - self.token_buffer - self.token_budget_output


@dataclass
class SummaryGPUConfig:
    memory_gb: int = 40

    @property
    def utilization(self) -> float:
        return 0.75 if self.memory_gb <= 40 else 0.85


@dataclass
class SummaryChunkingConfig:
    max_chunk_tokens: int = 11000
    overlap_tokens: int = 500
    chunk_summary_max_tokens: int = 1500


# Setup logging
logger = get_ingestion_logger(Path(__file__).stem, max_files=5)

# =========================
# Prompts (unchanged content)
# =========================
SUMMARY_PROMPT = """Summarize this research paper comprehensively in 5-7 flowing paragraphs. Write in dense, information-rich prose that maximizes keyword coverage for search indexing.

Include the following in natural, connected paragraphs:
- Define all technical terms, methods, and statistical concepts when first mentioned
- State the specific research problem with quantitative scope if available
- Describe study design, sample characteristics, and statistical models in detail
- Report all numerical findings: effect sizes, p-values, confidence intervals, model parameters
- Explain practical recommendations and common pitfalls to avoid
- Mention related methods, application domains, and computational tools
- Use varied vocabulary and precise technical language throughout

CRITICAL: Write only as much as the paper content warrants. Do NOT pad or hallucinate to reach a word count. If the paper is short, the summary should be proportionally shorter. Stop naturally when all key information is covered.

Write directly - do NOT include meta-commentary, introductory statements, or explanations about what you're doing. Start immediately with the paper's content.

Paper text:
{paper_text}

Summary:"""

CHUNK_SUMMARY_PROMPT = """Summarize this section comprehensively with maximum information density. Extract:

- All technical terms, method names, and statistical concepts
- All numbers: sample sizes, effect sizes, p-values, confidence intervals, parameters
- Study design details, software tools, and computational approaches
- Key findings, patterns, and comparisons
- Practical recommendations and contextual information

Write in dense prose - multiple concepts per sentence. Use precise technical vocabulary. Write only what the content supports - do NOT pad or add filler.

{content}

Summary:"""


SYNTHESIS_PROMPT = """You are given multiple independently generated summaries of chunks from a single research paper. Your task is to synthesize these into ONE comprehensive, flowing summary.

Requirements:
1. Write dense, information-rich paragraphs with all the important technical terms, methods, and statistical concepts
2. Eliminate ALL redundancy - each fact mentioned ONCE
3. Prioritize: methods, results (with numbers), and key findings
4. Maintain technical precision and vocabulary
5. Write naturally flowing prose, NOT a list of facts
6. Do NOT add meta-commentary or section labels

Chunk summaries:
{chunk_summaries}

Final synthesis:"""


# =========================
# Sampling params factory
# =========================
def make_sampling_params(max_tokens: int, task_type: str = "summary") -> SamplingParams:
    """Return SamplingParams that match original heuristics."""
    if task_type == "summary":
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.92,
            repetition_penalty=1.05,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=["---", "\n\n\n\n", "In summary, the", "To conclude,"],
        )
    else:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.05,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )


# =========================
# Token utilities (no class)
# =========================
_encoder_cache = {}


def get_encoder(model_name: str = "gpt-4o-mini"):
    """Return cached tiktoken encoder for a model name."""
    if model_name not in _encoder_cache:
        try:
            _encoder_cache[model_name] = tiktoken.encoding_for_model(model_name)
            logger.info(f"Tokenizer: loaded tiktoken encoder for {model_name}")
        except Exception as e:
            logger.error(f"Unable to load tokenizer encoding for {model_name}: {e}")
            raise
    return _encoder_cache[model_name]


def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    enc = get_encoder(model_name)
    return len(enc.encode(text))


# =========================
# Model loader (lazy)
# =========================
class ModelManager:
    """Lightweight manager for a single vLLM model instance."""

    def __init__(self, model_config: SummaryModelConfig, gpu_config: SummaryGPUConfig):
        self.model_config = model_config
        self.gpu_config = gpu_config
        self._llm: Optional[LLM] = None
        atexit.register(self.cleanup)

    def initialize(self) -> None:
        if self._llm is not None:
            logger.debug("Model already initialized")
            return
        logger.info(
            f"Initializing vLLM model '{self.model_config.name}' "
            f"on {self.gpu_config.memory_gb}GB GPU (util={self.gpu_config.utilization})"
        )
        try:
            self._llm = LLM(
                model=self.model_config.name,
                max_model_len=self.model_config.max_context,
                tensor_parallel_size=1,
                max_num_seqs=1,
                gpu_memory_utilization=self.gpu_config.utilization,
                enable_chunked_prefill=True,
                dtype="float16",
                enforce_eager=False,
                trust_remote_code=True,
            )
            logger.info("Model loaded")
        except Exception as e:
            logger.exception("Failed to initialize model")
            raise RuntimeError(e)

    def generate(self, prompt: str, sampling: SamplingParams) -> str:
        if self._llm is None:
            raise RuntimeError("Model not initialized. Call initialize().")
        try:
            output = self._llm.generate([prompt], sampling)
            return output[0].outputs[0].text.strip()
        except Exception:
            logger.exception("Generation error")
            raise

    def cleanup(self) -> None:
        if self._llm is None:
            return
        try:
            logger.info("Cleaning up model resources")
            del self._llm
            self._llm = None
            # to prevent nccl group warning
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("Error during model cleanup")


# =========================
# Chunking utilities
# =========================
def chunk_by_markdown_headers(
    text: str,
    tokenizer_model: str,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Split the markdown text by headers, greedily pack sections into chunks
    with an overlapping tail for context.
    """
    sections = re.split(r'(\n#{1,6}\s+[^\n]+|^#{1,6}\s+[^\n]+)', text, flags=re.MULTILINE)

    # reconstruct
    structured = []
    i = 0
    while i < len(sections):
        if i == 0 and sections[0].strip():
            structured.append(sections[0])
            i += 1
        elif i < len(sections) and ('#' in sections[i]):
            header = sections[i]
            content = sections[i + 1] if i + 1 < len(sections) else ""
            structured.append(header + content)
            i += 2
        else:
            i += 1

    chunks: List[str] = []
    current = ""
    previous_tail = ""

    for sec in structured:
        sec_tokens = count_tokens(sec, tokenizer_model)
        current_tokens = count_tokens(current, tokenizer_model) if current else 0

        # if single section larger than max, split by paragraphs
        if sec_tokens > max_chunk_tokens:
            logger.warning("Section too large (%d tokens), splitting into paragraphs", sec_tokens)
            if current:
                chunks.append(current)
                previous_tail = _get_tail_text(current, current_tokens, overlap_tokens)

            paragraphs = sec.split("\n\n")
            temp = previous_tail if previous_tail else ""
            for p in paragraphs:
                p_tokens = count_tokens(p, tokenizer_model)
                temp_tokens = count_tokens(temp, tokenizer_model) if temp else 0
                if temp_tokens + p_tokens > max_chunk_tokens:
                    if temp:
                        chunks.append(temp)
                    temp = p
                else:
                    temp = temp + ("\n\n" if temp else "") + p
            if temp:
                current = temp
                previous_tail = ""
            else:
                current = ""
            continue

        # normal greedy add
        if current and (current_tokens + sec_tokens > max_chunk_tokens):
            chunks.append(current)
            previous_tail = _get_tail_text(current, current_tokens, overlap_tokens)
            current = (previous_tail + "\n\n" + sec) if previous_tail else sec
        else:
            current = current + ("\n\n" if current else "") + sec

    if current:
        chunks.append(current)

    logger.info("Created %d chunks", len(chunks))
    return chunks


def _get_tail_text(text: str, token_count: int, overlap_tokens: int) -> str:
    """Return a tail substring that approximates overlap_tokens tokens."""
    if not text or token_count < overlap_tokens:
        return ""
    chars_per_token = len(text) / max(token_count, 1)
    tail_chars = int(overlap_tokens * chars_per_token)
    return text[-tail_chars:] if tail_chars < len(text) else ""


# =========================
# Summarizer (Final Production Version)
# =========================
class Summarizer:
    def __init__(
        self,
        model_manager: ModelManager,
        tokenizer_model: str,
        model_config: SummaryModelConfig,
        chunking_config: SummaryChunkingConfig,
    ):
        self.model_manager = model_manager
        self.tokenizer_model = tokenizer_model
        self.model_config = model_config
        self.chunking_config = chunking_config

    def summarize(self, content: str) -> str:
        if not content or not content.strip():
            raise ValueError("Empty content")

        tokens = count_tokens(content, self.tokenizer_model)
        max_direct = self.model_config.max_input_tokens
        logger.info("Input: %d tokens | direct threshold: %d", tokens, max_direct)

        if tokens <= max_direct:
            logger.info("Strategy: Direct summarization")
            return self._summarize_direct(content)
        else:
            logger.info("Strategy: Recursive chunked summarization")
            return self._process_recursive(content, depth=0)

    def _summarize_direct(self, content: str) -> str:
        prompt = SUMMARY_PROMPT.format(paper_text=content)
        sampling = make_sampling_params(self.model_config.token_budget_output, "summary")
        return self.model_manager.generate(prompt, sampling)

    def _process_recursive(self, content: str, depth: int, max_depth: int = 3) -> str:
        """
        Recursive loop: Chunk -> Summarize -> Check Size -> (Finalize or Recurse)
        """
        indent = "  " * depth

        # --- SAFETY BREAK (MAX DEPTH) ---
        if depth >= max_depth:
            logger.warning(f"{indent}Max recursion depth ({max_depth}) reached. Engaging fallback compression.")
            
            # CRITICAL FIX: Do not truncate. Re-chunk safely to force-fit the content.
            # We use smaller chunks (4000 tokens) to guarantee the model can process them.
            safe_chunks = chunk_by_markdown_headers(
                content,
                tokenizer_model=self.tokenizer_model,
                max_chunk_tokens=4000, 
                overlap_tokens=0, # No overlap needed for this emergency compression
            )
            logger.info(f"{indent}Fallback: Re-chunked into {len(safe_chunks)} safe segments")

            safe_summaries = []
            for i, ch in enumerate(safe_chunks):
                # Summarize each safe chunk
                safe_summaries.append(self._summarize_single_chunk(ch))
            
            # Combine the safe summaries
            combined_final = "\n\n".join(safe_summaries)
            
            # Final Check: If even the combined safe summaries are too big (rare but possible),
            # we MUST hard truncate to avoid a crash.
            final_tokens = count_tokens(combined_final, self.tokenizer_model)
            if final_tokens > self.model_config.max_input_tokens:
                logger.warning(f"{indent}Fallback result still too large ({final_tokens}), performing hard truncation.")
                # Calculate exact char limit based on token ratio to be safe
                ratio = self.model_config.max_input_tokens / final_tokens
                safe_chars = int(len(combined_final) * ratio * 0.95) # 5% buffer
                combined_final = combined_final[:safe_chars] + "\n[...Truncated...]"

            prompt = SYNTHESIS_PROMPT.format(chunk_summaries=combined_final)
            sampling = make_sampling_params(self.model_config.token_budget_output, "summary")
            return self.model_manager.generate(prompt, sampling)

        # --- NORMAL RECURSION STEPS ---

        # Step 1: Chunk
        # Use 8k chunks for intermediate steps (summaries of summaries) to maintain context
        current_chunk_size = self.chunking_config.max_chunk_tokens if depth == 0 else 8000
            
        chunks = chunk_by_markdown_headers(
            content,
            tokenizer_model=self.tokenizer_model,
            max_chunk_tokens=current_chunk_size,
            overlap_tokens=self.chunking_config.overlap_tokens,
        )
        logger.info(f"{indent}Depth {depth}: Split content into {len(chunks)} chunks")

        # Step 2: Summarize chunks
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"{indent}Depth {depth}: Summarizing chunk {i+1}/{len(chunks)}")
            summary = self._summarize_single_chunk(chunk)
            chunk_summaries.append(summary)

        # Step 3: Combine
        combined = "\n\n".join([f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)])
        combined_tokens = count_tokens(combined, self.tokenizer_model)
        logger.info(f"{indent}Depth {depth}: Combined summaries = {combined_tokens} tokens")

        # Step 4: Decision
        if combined_tokens <= self.model_config.max_input_tokens:
            logger.info(f"{indent}Depth {depth}: Fits in context. Generating final synthesis.")
            prompt = SYNTHESIS_PROMPT.format(chunk_summaries=combined)
            sampling = make_sampling_params(self.model_config.token_budget_output, "summary")
            return self.model_manager.generate(prompt, sampling)
        else:
            logger.info(f"{indent}Depth {depth}: Result too large. Recursing...")
            return self._process_recursive(combined, depth + 1, max_depth)

    def _summarize_single_chunk(self, chunk: str) -> str:
        """Helper to summarize a raw text chunk."""
        chunk_tokens = count_tokens(chunk, self.tokenizer_model)
        
        # Safety truncation for individual chunks
        if chunk_tokens > self.model_config.max_input_tokens:
            chars_to_keep = int(len(chunk) * (self.model_config.max_input_tokens / chunk_tokens))
            chunk = chunk[:chars_to_keep]

        prompt = CHUNK_SUMMARY_PROMPT.format(content=chunk)
        sampling = make_sampling_params(self.chunking_config.chunk_summary_max_tokens, "chunk")
        return self.model_manager.generate(prompt, sampling)


# =========================
# File I/O helpers
# =========================
def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    # replace is atomic on POSIX
    tmp.replace(path)


def read_text(path: Path) -> str:
    data = path.read_text(encoding="utf-8")
    if not data.strip():
        raise ValueError(f"Empty file: {path}")
    return data


def prepare_input_files(input_path: Path) -> Tuple[List[Path], bool]:
    """Return list of input files and whether batch mode (directory)"""
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if input_path.is_file():
        return [input_path], False

    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.name.endswith(".md")])
        if not files:
            raise FileNotFoundError(f"No *filtered.md files found in: {input_path}")
        return files, True

    raise ValueError(f"Invalid input path: {input_path}")


def get_output_path(input_file: Path, output_arg: Optional[Path], batch_mode: bool, output_dir: Optional[Path]) -> Path:
    if batch_mode:
        out_dir = output_dir or (input_file.parent / "summaries")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = input_file.name.replace("filtered.md", "summary.md")
        if name == input_file.name:
            name = input_file.stem + "_summary.md"
        return out_dir / name
    else:
        if output_arg:
            return output_arg
        name = input_file.name.replace("filtered.md", "summary.md")
        if name == input_file.name:
            name = input_file.stem + "_summary.md"
        return input_file.with_name(name)


# =========================
# CLI and orchestration
# =========================
def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize research papers using vLLM")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *.md files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save summary md files.")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=True,
                        help="Regenerate even if summary exists")
    
    # Model Config Overrides
    parser.add_argument("--max-context", type=int, help="Override max context window")
    parser.add_argument("--output-budget", type=int, help="Override output token budget")
    
    args = parser.parse_args(argv)

    input_path = Path(args.input_dir)
    try:
        files, batch_mode = prepare_input_files(input_path)
    except Exception as e:
        logger.error("Input error: %s", e)
        return 2

    output_arg = Path(args.output_dir) if args.output_dir else None
    output_dir = None
    if batch_mode:
        output_dir = output_arg if output_arg else (input_path / "summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Batch mode: %d files; output_dir=%s; skip_existing=%s", len(files), output_dir, args.skip_existing)

    # Determine GPU config
    detected = detect_gpu_memory_gb()
    gpu_mem = detected or 40
    gpu_cfg = SummaryGPUConfig(memory_gb=gpu_mem)

    # Load configs (stub -> replace/import real config as desired)
    model_cfg = SummaryModelConfig()
    if args.max_context:
        model_cfg.max_context = args.max_context
    if args.output_budget:
        model_cfg.token_budget_output = args.output_budget

    chunk_cfg = SummaryChunkingConfig()

    # Model manager and summarizer
    manager = ModelManager(model_cfg, gpu_cfg)
    manager.initialize()
    summarizer = Summarizer(manager, tokenizer_model="gpt-4o-mini", model_config=model_cfg, chunking_config=chunk_cfg)

    # Process files
    stats = {"successful": 0, "skipped": 0, "failed": 0}
    for i, fp in enumerate(files, 1):
        logger.info("Processing %s (%d/%d)", fp.name, i, len(files))
        out_path = get_output_path(fp, output_arg, batch_mode, output_dir)
        if args.skip_existing and out_path.exists():
            logger.info("Skipping existing: %s", out_path)
            stats["skipped"] += 1
            continue
        try:
            text = read_text(fp)
            summary = summarizer.summarize(text)
            atomic_write(out_path, summary)
            tcount = count_tokens(summary)
            logger.info("Saved %s (%d chars, ~%d tokens)", out_path, len(summary), tcount)
            stats["successful"] += 1
        except Exception as e:
            logger.exception("Failed to process %s", fp)
            stats["failed"] += 1

    logger.info("Done. success=%d skipped=%d failed=%d total=%d",
                stats["successful"], stats["skipped"], stats["failed"], len(files))

    manager.cleanup()
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(cli())
