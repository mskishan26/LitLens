"""
Jina-compatible reranker for Stage 3 of RAG pipeline.
"""

import torch
from transformers import AutoModel
from typing import List, Tuple, Dict, Optional, Any
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Import your custom logger utilities
from utils.logger import (
    get_logger, 
    log_stage_start, 
    log_retrieval_metrics,
    log_stage_end
)
from utils.config_loader import load_config

logger = get_logger(__name__)

class Reranker:
    """
    Jina-based cross-encoder reranker for Stage 3 of RAG pipeline.
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
        self.config = load_config(config_path) if config_path else {}
        # Default to config, fallback to arg, fallback to hardcoded
        model_name = model_name or self.config.get('models', {}).get('reranker', 'jinaai/jina-reranker-v3')
        
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.timeout_seconds = timeout_seconds
        
        logger.info(f"Loading Jina reranker model '{model_name}' on {device}")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True,
        )
        
        if device == 'cuda':
            self.model = self.model.to(device)
        
        self.model.eval()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("Jina reranker loaded successfully", extra={
            "device": device,
            "timeout": timeout_seconds
        })
    
    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5,
        return_scores: bool = True,
        max_context_tokens: int = 30_000
    ) -> Tuple[List[Tuple[float, Dict, str]], Dict[str, Any]]:
        """
        Core reranking logic.
        
        Returns:
            Tuple containing:
            1. List of sorted results (score, meta, text)
            2. Dict of execution metadata (method used, token count, success status)
        """
        start_time = time.perf_counter()
        
        # 1. Validation
        if not candidates:
            return [], {"method": "empty_input", "success": True}

        # 2. Extract Texts & Validate
        texts = []
        valid_candidates = []
        
        for dist, meta, chunk_text in candidates:
            if chunk_text is None:
                continue 
            texts.append(chunk_text)
            valid_candidates.append((dist, meta, chunk_text))
        
        if not texts:
            return [], {"method": "no_valid_text", "success": False}

        # 3. Context Window Protection
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars / 4
        
        # Truncate if necessary (Stage 2 order preserved)
        if estimated_tokens > max_context_tokens:
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

        # 4. Cap Candidate Count (Hard limit for cross-encoder speed)
        max_rerank_candidates = min(100, len(texts))
        if len(texts) > max_rerank_candidates:
            texts = texts[:max_rerank_candidates]
            valid_candidates = valid_candidates[:max_rerank_candidates]

        # 5. Execution with Timeout
        method = "jina_rerank"
        rerank_results = []
        
        try:
            future = self._executor.submit(self.model.rerank, query, texts, top_n=None)
            rerank_results = future.result(timeout=self.timeout_seconds)
            
        except (FuturesTimeoutError, Exception) as e:
            # FALLBACK LOGIC
            is_timeout = isinstance(e, FuturesTimeoutError)
            method = "fallback_timeout" if is_timeout else "fallback_error"
            
            logger.error(
                f"Reranking failed ({method}). Reverting to Stage 2 scores.",
                exc_info=not is_timeout,  # Don't print stack trace for simple timeouts
                extra={"error": str(e), "timeout_setting": self.timeout_seconds}
            )
            
            # Create fake "reranker" results based on original order
            # We normalize original distances to a fake 0-1 score to maintain API contract
            # or just return the candidates as is if downstream can handle it.
            # Here we preserve the list but mark method as fallback.
            
            # Simple fallback: Return top_k of the VALID candidates (already sorted by Stage 2)
            top_results = valid_candidates[:top_k]
            
            # If caller expects scores (floats), the original might be distances (unbounded).
            # We just pass them through, but the 'method' flag warns the caller.
            return top_results, {
                "method": method,
                "success": False,
                "candidates_processed": len(valid_candidates),
                "error": str(e)
            }

        # 6. Process Successful Results
        # Map Jina results back to valid_candidates indices
        index_to_score = {r['index']: r['relevance_score'] for r in rerank_results}
        
        final_list = []
        for idx, (_, meta, text) in enumerate(valid_candidates):
            score = index_to_score.get(idx, 0.0)
            final_list.append((float(score), meta, text))
            
        # Sort by new score
        final_list.sort(key=lambda x: x[0], reverse=True)
        final_results = final_list[:top_k]

        execution_time = (time.perf_counter() - start_time) * 1000
        
        return final_results, {
            "method": method,
            "success": True,
            "candidates_processed": len(texts),
            "duration_ms": execution_time
        }

    def rerank_with_details(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Wrapper that handles logging, metrics, and detailed rank comparison.
        """
        start_time = time.perf_counter()
        
        # LOG START
        log_stage_start(logger, "reranker", top_k=top_k, input_count=len(candidates))

        # 1. OPTIMIZATION: Build O(1) lookup map for original ranks
        # Using id(meta) ensures we match the specific object instance from Stage 2
        # Structure: id(meta) -> (original_rank_index, original_distance)
        original_rank_map = {}
        for idx, (dist, meta, _) in enumerate(candidates):
            original_rank_map[id(meta)] = (idx + 1, dist)

        # 2. Call Core Rerank
        reranked_results, run_info = self.rerank(
            query, 
            candidates, 
            top_k=top_k, 
            return_scores=True
        )

        # 3. Build Detailed Response
        detailed_results = []
        
        for rank_idx, (score, meta, text) in enumerate(reranked_results):
            current_rank = rank_idx + 1
            
            # O(1) Lookup
            orig_info = original_rank_map.get(id(meta))
            
            if orig_info:
                orig_rank, orig_dist = orig_info
                rank_change = orig_rank - current_rank # Positive = Improved
            else:
                # Should not happen if candidates list wasn't mutated externally
                orig_rank, orig_dist, rank_change = (None, None, 0)

            detailed_results.append({
                'rank': current_rank,
                'rerank_score': score,
                'original_distance': orig_dist,
                'original_rank': orig_rank,
                'rank_improvement': rank_change,
                'metadata': meta,
                'text': text
            })

        # 4. LOG METRICS (Compatible with logger.py)
        duration = (time.perf_counter() - start_time) * 1000
        
        # Determine top score safely
        top_score = detailed_results[0]['rerank_score'] if detailed_results else 0.0
        
        log_retrieval_metrics(
            logger,
            stage="reranker",
            count=len(detailed_results),
            duration_ms=duration,
            top_score=top_score,
            # Extra fields specifically requested
            method=run_info.get('method', 'unknown'),
            fallback_triggered=not run_info.get('success', True),
            candidates_in=len(candidates),
            candidates_reranked=run_info.get('candidates_processed', 0)
        )

        return detailed_results