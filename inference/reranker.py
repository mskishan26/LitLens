"""
Jina-compatible reranker for Stage 3 of RAG pipeline.
V3: Merged version with:
    - Timeout fallback (restored from v1)
    - Internal thread safety 
    - GPU memory management (cache clearing, not CPU offload)
    - Optional AsyncReranker wrapper for async pipelines
"""

import torch
from transformers import AutoModel
from typing import List, Tuple, Dict, Optional, Any
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
    Jina-based cross-encoder reranker with:
    - Timeout protection for hung inference
    - Thread-safe execution
    - GPU memory management (clears compute overhead, keeps model loaded)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: str = "jinaai/jina-reranker-v3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        max_length: int = 1024,
        timeout_seconds: int = 30,
        max_workers: int = 1,
        auto_clear_cache: bool = True,  # Clear GPU cache after each rerank
    ):
        self.config = load_config(config_path) if config_path else {}
        model_name = model_name or self.config.get('models', {}).get('reranker', 'jinaai/jina-reranker-v3')
        
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.timeout_seconds = timeout_seconds
        self.auto_clear_cache = auto_clear_cache
        
        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Loading Jina reranker model '{model_name}' on {device}")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True,
        )
        
        if device == 'cuda':
            self.model = self.model.to(device)
        
        self.model.eval()
        
        logger.info("Jina reranker loaded successfully", extra={
            "device": device,
            "timeout_seconds": timeout_seconds,
            "auto_clear_cache": auto_clear_cache,
        })
    
    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    # =========================================================================
    # GPU MEMORY MANAGEMENT
    # =========================================================================
    
    def clear_compute_cache(self) -> None:
        """
        Clear GPU activation/compute memory without moving model weights.
        
        Safe to call even when vLLM is running — only frees unreferenced memory.
        vLLM's KV cache and weights are held in live tensors, so untouched.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared GPU compute cache")
    
    def get_model_memory_mb(self) -> float:
        """Approximate GPU memory used by model weights (not compute)."""
        if self.device != 'cuda':
            return 0.0
        params = sum(p.numel() for p in self.model.parameters())
        # float16 = 2 bytes per param
        return (params * 2) / (1024 * 1024)
    
    # =========================================================================
    # CORE RERANKING (with timeout protection)
    # =========================================================================
    
    def _rerank_batch_internal(
        self,
        query: str,
        texts: List[str],
        batch_size: int,
    ) -> Dict[int, float]:
        """
        Internal batched reranking. Runs inside executor for timeout protection.
        Returns: {global_index: score}
        """
        all_scores = {}
        
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            
            batch_results = self.model.rerank(
                query,
                batch_texts,
                top_n=None
            )
            
            for r in batch_results:
                global_idx = start_idx + r["index"]
                all_scores[global_idx] = float(r["relevance_score"])
        
        return all_scores
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5,
        return_scores: bool = True,
        truncate_texts: bool = False,
        max_candidates: int = 50,
        batch_size: int = 5,
        max_context_tokens: int = 2048,
    ) -> Tuple[List[Tuple[float, Dict, str]], Dict[str, Any]]:
        """
        Memory-safe batched reranking with timeout protection.
        
        Strategy:
        - Cap candidates
        - Run batched rerank with timeout
        - Clear compute cache after (keeps model loaded)
        - Fallback to Stage 2 scores on timeout/error
        """
        start_time = time.perf_counter()

        if not candidates:
            return [], {"method": "empty_input", "success": True}

        # 1. Sanitize + cap candidates
        valid_candidates = []
        for dist, meta, text in candidates:
            if text:
                valid_candidates.append((dist, meta, text))

        if not valid_candidates:
            return [], {"method": "no_valid_text", "success": False}

        valid_candidates = valid_candidates[:max_candidates]
        texts = [t for _, _, t in valid_candidates]

        # 2. Token safety (approximate truncation)
        def approx_tokens(s: str) -> int:
            return len(s) // 4

        if truncate_texts:
            truncated_texts = []
            for t in texts:
                if approx_tokens(t) > max_context_tokens:
                    truncated_texts.append(t[: max_context_tokens * 4])
                else:
                    truncated_texts.append(t)
        else:
            truncated_texts = texts

        # 3. Run reranking with timeout protection
        method = "jina_rerank_batched"
        all_scores = {}
        timed_out = False
        error_msg = None

        try:
            with self._lock:
                future = self._executor.submit(
                    self._rerank_batch_internal,
                    query,
                    truncated_texts,
                    batch_size,
                )
                all_scores = future.result(timeout=self.timeout_seconds)

        except FuturesTimeoutError:
            timed_out = True
            error_msg = f"Reranking timed out after {self.timeout_seconds}s"
            logger.warning(error_msg, extra={
                "candidates": len(valid_candidates),
                "timeout": self.timeout_seconds,
            })
            method = "fallback_timeout"

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Batched reranking failed, falling back to Stage 2 scores",
                exc_info=True,
                extra={"error": error_msg}
            )
            method = "fallback_error"

        finally:
            # 4. Clear compute cache (keeps model loaded, frees activations)
            if self.auto_clear_cache:
                self.clear_compute_cache()

        # 5. Handle fallback case
        if timed_out or error_msg:
            return valid_candidates[:top_k], {
                "method": method,
                "success": False,
                "candidates_processed": len(valid_candidates),
                "error": error_msg,
                "timed_out": timed_out,
            }

        # 6. Merge + global sort
        reranked = []
        for idx, (_, meta, text) in enumerate(valid_candidates):
            score = all_scores.get(idx, 0.0)
            reranked.append((score, meta, text))

        reranked.sort(key=lambda x: x[0], reverse=True)
        final_results = reranked[:top_k]

        duration_ms = (time.perf_counter() - start_time) * 1000

        return final_results, {
            "method": method,
            "success": True,
            "candidates_processed": len(valid_candidates),
            "batch_size": batch_size,
            "duration_ms": duration_ms,
        }

    def rerank_with_details(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Wrapper with logging, metrics, and detailed rank comparison.
        """
        start_time = time.perf_counter()
        
        log_stage_start(logger, "reranker", top_k=top_k, input_count=len(candidates))

        # Build O(1) lookup map for original ranks
        original_rank_map = {}
        for idx, (dist, meta, _) in enumerate(candidates):
            c_id = meta.get('chroma_id')
            if c_id is not None:
                original_rank_map[c_id] = (idx + 1, dist)

        # Core rerank
        reranked_results, run_info = self.rerank(
            query, 
            candidates, 
            top_k=top_k, 
            return_scores=True
        )

        # Build detailed response
        detailed_results = []
        
        for rank_idx, (score, meta, text) in enumerate(reranked_results):
            current_rank = rank_idx + 1
            c_id = meta.get('chroma_id')
            orig_info = original_rank_map.get(c_id)
            
            if orig_info:
                orig_rank, orig_dist = orig_info
                rank_change = orig_rank - current_rank
            else:
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

        duration = (time.perf_counter() - start_time) * 1000
        top_score = detailed_results[0]['rerank_score'] if detailed_results else 0.0
        
        log_retrieval_metrics(
            logger,
            stage="reranker",
            count=len(detailed_results),
            duration_ms=duration,
            top_score=top_score,
            method=run_info.get('method', 'unknown'),
            fallback_triggered=not run_info.get('success', True),
            candidates_in=len(candidates),
            candidates_reranked=run_info.get('candidates_processed', 0)
        )

        return detailed_results


# =============================================================================
# ASYNC WRAPPER (Optional, for asyncio pipelines)
# =============================================================================

class AsyncReranker:
    """
    Async wrapper for use in asyncio pipelines.
    
    Delegates all logic to the core Reranker — this is just an async interface.
    Timeout and memory management are handled by the underlying Reranker.
    """
    
    def __init__(self, reranker: Reranker):
        self.reranker = reranker
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5,
        **kwargs
    ) -> Tuple[List[Tuple[float, Dict, str]], Dict[str, Any]]:
        """Async version of rerank."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.reranker.rerank(query, candidates, top_k, **kwargs)
        )
    
    async def rerank_with_details(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5
    ) -> List[Dict]:
        """Async version of rerank_with_details."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.reranker.rerank_with_details(query, candidates, top_k)
        )
    
    def clear_compute_cache(self) -> None:
        """Proxy to underlying reranker."""
        self.reranker.clear_compute_cache()
    
    @property
    def device(self) -> str:
        return self.reranker.device