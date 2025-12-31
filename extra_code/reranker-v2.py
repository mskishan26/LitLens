"""
Jina-compatible reranker for Stage 3 of RAG pipeline.
Uses the official Jina Reranker v3 model with its native rerank() method.
Jina's rerank method is incredibly straightforward as compared to the qwen method with the tokenizer and the padding, 
although I have yet to check if we can change something in between those steps, otherwise best to write a wrapper.
"""

import torch
from transformers import AutoModel
from typing import List, Tuple, Dict, Optional
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from utils.logger import (
    get_logger, 
    log_stage_start, 
    log_retrieval_metrics,
)
from utils.config_loader import load_config

logger = get_logger(__name__)


class Reranker:
    """
    Jina-based cross-encoder reranker for Stage 3 of RAG pipeline.
    
    Production Safety Features:
    - No lazy I/O: Requires chunk_text in all candidates (no disk reads during inference)
    - Timeout handling: Falls back to Stage 2 ordering if reranking hangs
    - Context window protection: Caps candidates to ~100k tokens
    - Reranker limit: Maximum 100 candidates to avoid cross-encoder overhead
    - Singleton executor: Thread pool created once at init, reused across requests
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: str = "jinaai/jina-reranker-v3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        max_length: int = 1024,
        timeout_seconds: int = 30,
        max_workers: int = 1
    ):
        """
        Initialize Jina reranker.
        
        Args:
            config_path: Path to config file
            model_name: HuggingFace model name for reranker
            device: 'cuda' or 'cpu'
            batch_size: Batch size for reranking (used for manual batching if needed)
            max_length: Maximum sequence length for reranker
            timeout_seconds: Timeout for reranking operation (default: 30s)
            max_workers: Number of worker threads for reranking (default: 1)
        """
        self.config = load_config(config_path) if config_path else {}
        model_name = model_name or self.config.get('models', {}).get('reranker', 'jinaai/jina-reranker-v3')
        
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.timeout_seconds = timeout_seconds
        
        logger.info(f"Loading Jina reranker model '{model_name}' on {device}")
        
        # Load model using AutoModel - this loads Jina's custom model class
        # which includes the built-in rerank() method
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True,
        )
        
        # Move to device if not using device_map
        if device == 'cuda':
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Create singleton executor for timeout handling
        # Workers will be reused across requests, no executor leaks
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Jina reranker loaded successfully")
        logger.info(f"  Device: {device}, Batch size: {batch_size}, Max length: {max_length}")
        logger.info(f"  Timeout: {timeout_seconds}s, Worker threads: {max_workers}")
    
    def __del__(self):
        """Clean up executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def shutdown(self):
        """Explicitly shutdown the executor pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logger.info("Reranker executor pool shutdown")
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5,
        return_scores: bool = True,
        max_context_tokens: int = 30_000  # Conservative limit for context window
    ) -> List[Tuple[float, Dict, str]]:
        """
        Rerank candidates using the Jina reranker model.
        
        Args:
            query: Search query
            candidates: List of (distance, metadata, chunk_text) from Stage 2
            top_k: Number of top results to return after reranking
            return_scores: If True, return reranker scores; if False, return ranks
            max_context_tokens: Maximum tokens across all chunks (default: 100k)
        
        Returns:
            List of (score/rank, metadata, chunk_text) tuples, sorted by relevance
        """
        if not candidates:
            logger.warning("No candidates provided for reranking")
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates for query: '{query[:100]}...'")
        
        # Extract texts from candidates
        texts = []
        valid_candidates = []
        
        for dist, meta, chunk_text in candidates:
            # Production safety: chunk_text should ALWAYS be provided
            # If None, skip this candidate to avoid blocking I/O
            if chunk_text is None:
                logger.warning(
                    f"Skipping candidate with missing chunk_text: "
                    f"{meta.get('file_path', 'unknown')} chunk {meta.get('chunk_index', '?')}. "
                    f"This indicates an issue in Stage 2 retrieval."
                )
                continue
            
            texts.append(chunk_text)
            valid_candidates.append((dist, meta, chunk_text))
        
        if not texts:
            logger.warning("No valid candidates after loading texts")
            return []
        
        # Issue #4: Cap candidates to avoid context window overflow
        # Estimate tokens (rough: 1 token ≈ 4 chars) and limit candidates
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars / 4
        
        if estimated_tokens > max_context_tokens:
            # Truncate to fit context window, keeping Stage 2 ordering
            logger.warning(
                f"Estimated tokens ({estimated_tokens:.0f}) exceeds max ({max_context_tokens}). "
                f"Truncating from {len(texts)} to fit context window."
            )
            
            cumulative_tokens = 0
            max_candidates = 0
            for i, text in enumerate(texts):
                text_tokens = len(text) / 4
                if cumulative_tokens + text_tokens > max_context_tokens:
                    break
                cumulative_tokens += text_tokens
                max_candidates = i + 1
            
            texts = texts[:max_candidates]
            valid_candidates = valid_candidates[:max_candidates]
            logger.info(f"Truncated to {max_candidates} candidates (~{cumulative_tokens:.0f} tokens)")
        
        # Cap to 100 candidates for reranker (Issue #4)
        max_rerank_candidates = min(100, len(texts))
        if len(texts) > max_rerank_candidates:
            logger.info(f"Capping reranking from {len(texts)} to {max_rerank_candidates} candidates")
            texts = texts[:max_rerank_candidates]
            valid_candidates = valid_candidates[:max_rerank_candidates]
        
        # Issue #3: Add timeout handling with fallback using singleton executor
        try:
            # Submit to singleton executor (reuses workers across requests)
            future = self._executor.submit(self.model.rerank, query, texts, top_n=None)
            
            try:
                rerank_results = future.result(timeout=self.timeout_seconds)
            except FuturesTimeoutError:
                logger.error(
                    f"Reranking timed out after {self.timeout_seconds}s. "
                    f"Falling back to Stage 2 ordering (top {top_k} from embeddings)."
                )
                # Worker will finish in background and be available for next request
                # Fallback: return top_k from Stage 2 ordering
                fallback_results = valid_candidates[:top_k]
                if not return_scores:
                    fallback_results = [(i + 1, meta, text) for i, (_, meta, text) in enumerate(fallback_results)]
                return fallback_results
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to Stage 2 ordering.")
            fallback_results = valid_candidates[:top_k]
            if not return_scores:
                fallback_results = [(i + 1, meta, text) for i, (_, meta, text) in enumerate(fallback_results)]
            return fallback_results
        
        # Build a mapping from original index to score
        index_to_score = {r['index']: r['relevance_score'] for r in rerank_results}
        
        # Combine scores with candidates
        reranked = []
        for idx, (_, meta, text) in enumerate(valid_candidates):
            score = index_to_score.get(idx, 0.0)
            reranked.append((float(score), meta, text))
        
        # Sort by score (higher is better)
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        result = reranked[:top_k]
        
        # Optionally convert to ranks
        if not return_scores:
            result = [(i + 1, meta, text) for i, (_, meta, text) in enumerate(result)]
        
        logger.info(f"Reranking complete, returning top {len(result)} results")
        if result:
            logger.info(f"  Top score: {result[0][0]:.4f}, Bottom score: {result[-1][0]:.4f}")
        
        return result
    
    def rerank_with_details(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank and return detailed results with both original and reranker scores.
        
        Args:
            query: Search query
            candidates: List of (distance, metadata, chunk_text) from Stage 2
            top_k: Number of top results to return
        
        Returns:
            List of dicts with detailed ranking information
        """
        start_time = time.perf_counter()
        log_stage_start(logger, "reranker", k=k)
        
        # Get reranked results
        reranked = self.rerank(query, candidates, top_k=top_k, return_scores=True)
        
        # Add detailed information
        detailed_results = []
        for rank, (rerank_score, meta, text) in enumerate(reranked, 1):
            # Find original distance and rank from candidates
            original_dist = None
            original_rank = None
            for orig_rank, (dist, orig_meta, _) in enumerate(candidates, 1):
                if orig_meta == meta:
                    original_dist = dist
                    original_rank = orig_rank
                    break
            
            detailed_results.append({
                'rank': rank,
                'rerank_score': rerank_score,
                'original_distance': original_dist,
                'original_rank': original_rank,
                'rank_improvement': original_rank - rank if original_rank else None,
                'metadata': meta,
                'text': text
            })
        
        return detailed_results


def get_optimal_reranker_batch_size() -> int:
    """Determine optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 4
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if gpu_memory_gb >= 70:  # A100 80GB, H100
        return 32
    elif gpu_memory_gb >= 35:  # A100 40GB
        return 16
    else:  # V100 16GB or smaller
        return 8