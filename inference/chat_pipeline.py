"""
Production RAG Pipeline (Refactored)
====================================

Five-stage retrieval pipeline with:
1. Hybrid retrieval: BM25 + Paper-level embeddings → k papers
2. Chunk retrieval: Chunk-level embeddings filtered by k papers → m chunks  
3. Reranking: Jina reranker → n chunks
4. Generation: Qwen LLM with context
5. Hallucination check: HHEM verification of claims against sources

Key improvements over original:
- RequestTracer integration for full data capture
- Clean async/sync separation
- Modular stage methods
- Better error handling with stage-level recovery
- Memory-efficient sequential model loading
"""

import torch
import gc
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, List, Any, Generator, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
import time

from utils.logger import (
    get_logger, 
    set_request_context, 
    clear_request_context,
    log_request_summary
)
from utils.config_loader import load_config

logger = get_logger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def clear_gpu_memory():
    """Aggressively clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_reranker_batch_size() -> int:
    """
    Determine optimal batch size for reranker based on available GPU memory.
    
    Returns:
        Batch size (4-16 depending on GPU memory)
    """
    if not torch.cuda.is_available():
        return 4
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory >= 24:
            return 16
        elif total_memory >= 16:
            return 12
        elif total_memory >= 8:
            return 8
        else:
            return 4
    except Exception:
        return 8


# =============================================================================
# Data Classes for Pipeline Results
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run."""
    k_papers: int
    m_chunks: int
    n_reranked: int
    bm25_weight: float
    embedding_weight: float
    temperature: float
    include_citations: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'k_papers': self.k_papers,
            'm_chunks': self.m_chunks,
            'n_reranked': self.n_reranked,
            'bm25_weight': self.bm25_weight,
            'embedding_weight': self.embedding_weight,
            'temperature': self.temperature,
            'include_citations': self.include_citations,
        }


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""
    query: str
    answer: str
    timestamp: str
    config: Dict[str, Any]
    
    # Metadata
    papers_selected: int = 0
    chunks_retrieved: int = 0
    contexts_used: int = 0
    elapsed_seconds: float = 0.0
    
    # Stage outputs (for debugging/analysis)
    contexts: Optional[List[Dict]] = None
    hallucination_result: Optional[Dict] = None
    
    # Trace ID for correlation
    trace_id: Optional[str] = None
    
    def to_dict(self, include_contexts: bool = False) -> Dict[str, Any]:
        result = {
            'query': self.query,
            'answer': self.answer,
            'timestamp': self.timestamp,
            'config': self.config,
            'metadata': {
                'papers_selected': self.papers_selected,
                'chunks_retrieved': self.chunks_retrieved,
                'contexts_used': self.contexts_used,
                'elapsed_seconds': self.elapsed_seconds,
                'trace_id': self.trace_id,
            }
        }
        
        if include_contexts and self.contexts:
            result['contexts'] = self.contexts
        
        if self.hallucination_result:
            result['hallucination_check'] = self.hallucination_result
        
        return result


# =============================================================================
# RAG Pipeline Class
# =============================================================================

class RAGPipeline:
    """
    Memory-efficient RAG pipeline with hybrid retrieval and sequential model loading.
    
    Supports both sync and async operation modes.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        no_unload: bool = False,
        enable_hallucination_check: bool = True,
        enable_tracing: bool = True
    ):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to config file (optional)
            device: Computation device ('cuda' or 'cpu')
            no_unload: If True, keep models loaded between queries
            enable_hallucination_check: If True, run verification on answers
            enable_tracing: If True, capture full execution traces
        """
        self.config = load_config(config_path)
        self.no_unload = no_unload
        self.enable_hallucination_check = enable_hallucination_check
        self.enable_tracing = enable_tracing
        
        # Paths
        self.embeddings_path = Path(self.config['paths']['embeddings'])
        self.bm25_artifacts_path = Path(self.config['paths']['bm25_artifacts'])
        self.traces_path = Path(self.config['paths'].get('traces', 'outputs/traces'))
        
        # Model configs
        self.embedding_model = self.config['models']['embedding']
        self.reranker_model = self.config['models']['reranker']
        self.generator_model = self.config['models']['generator']
        self.truncate_dim = self.config['retrieval']['truncate_dim']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model instances (loaded on-demand)
        self.bm25_searcher = None
        self.embedding_searcher = None
        self.reranker = None
        self.async_generator = None  # vLLM async generator (used for both sync and streaming)
        self.hallucination_checker = None
        
        # Session management
        self.session_logger = None
        self.current_tracer = None
        
        logger.info("RAG Pipeline initialized", extra={
            'device': self.device,
            'embeddings_path': str(self.embeddings_path),
            'bm25_path': str(self.bm25_artifacts_path),
            'hallucination_check': self.enable_hallucination_check,
            'tracing': self.enable_tracing,
            'no_unload': self.no_unload,
        })
        
        if self.no_unload:
            self._preload_all_models()
    
    # =========================================================================
    # Model Loading/Unloading
    # =========================================================================
    
    def _preload_all_models(self):
        """Pre-load all models into memory."""
        logger.info("Pre-loading all models...")
        self._load_bm25()
        self._load_embedding_search()
        self._load_reranker()
        self._load_generator()
        if self.enable_hallucination_check:
            self._load_hallucination_checker()
        logger.info("All models pre-loaded")
    
    def _load_bm25(self):
        """Load BM25 search index."""
        if self.bm25_searcher is None:
            logger.info("Loading BM25 searcher")
            from inference.bm25_search import BM25Searcher
            
            self.bm25_searcher = BM25Searcher(str(self.bm25_artifacts_path))
            self.bm25_searcher.load_bm25_artifacts()
            self._log_memory("BM25 loaded")
    
    def _unload_bm25(self):
        """Unload BM25 searcher."""
        if self.no_unload:
            return
        if self.bm25_searcher is not None:
            logger.debug("Unloading BM25 searcher")
            del self.bm25_searcher
            self.bm25_searcher = None
            clear_gpu_memory()
    
    def _load_embedding_search(self):
        """Load embedding search system."""
        if self.embedding_searcher is None:
            logger.info("Loading embedding searcher")
            from inference.embedding_search import EmbeddingSearch
            
            self.embedding_searcher = EmbeddingSearch(
                embedding_model_name=self.embedding_model,
                device=self.device,
                truncate_dim=self.truncate_dim
            )
            self.embedding_searcher.load(self.embeddings_path)
            self._log_memory("Embedding searcher loaded")
    
    def _unload_embedding_search(self):
        """Unload embedding search system."""
        if self.no_unload:
            return
        if self.embedding_searcher is not None:
            logger.debug("Unloading embedding searcher")
            if hasattr(self.embedding_searcher, 'model'):
                del self.embedding_searcher.model
            del self.embedding_searcher
            self.embedding_searcher = None
            clear_gpu_memory()
    
    def _load_reranker(self):
        """Load reranker model."""
        if self.reranker is None:
            logger.info("Loading reranker")
            from inference.reranker import Reranker
            
            batch_size = get_optimal_reranker_batch_size()
            self.reranker = Reranker(
                model_name=self.reranker_model,
                device=self.device,
                batch_size=batch_size
            )
            self._log_memory("Reranker loaded")
    
    def _unload_reranker(self):
        """Unload reranker model."""
        if self.no_unload:
            return
        if self.reranker is not None:
            logger.debug("Unloading reranker")
            if hasattr(self.reranker, 'model'):
                del self.reranker.model
            del self.reranker
            self.reranker = None
            clear_gpu_memory()
    
    def _load_generator(self):
        """
        Load generator model.
        
        Note: The generator uses vLLM's AsyncLLMEngine which requires async.
        For sync usage, we run the async code in an event loop.
        """
        if self.async_generator is None:
            logger.info("Loading generator (vLLM async engine)")
            from inference.generator import AsyncQwenGenerator
            
            self.async_generator = AsyncQwenGenerator(
                model_name=self.generator_model,
                gpu_memory_utilization=0.9
            )
            
            # Initialize synchronously using asyncio.run or get_event_loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, schedule as task
                raise RuntimeError("Use async generator loading in async context")
            except RuntimeError:
                # No running loop, we can use asyncio.run
                asyncio.run(self.async_generator.initialize())
            
            self._log_memory("Generator loaded")
    
    def _unload_generator(self):
        """Unload generator."""
        if self.no_unload:
            return
        if self.async_generator is not None:
            logger.debug("Unloading generator")
            try:
                loop = asyncio.get_running_loop()
                # In async context - can't easily cleanup sync
                pass
            except RuntimeError:
                asyncio.run(self.async_generator.cleanup())
            self.async_generator = None
            clear_gpu_memory()
    
    async def _load_async_generator(self):
        """Load async generator for streaming."""
        if self.async_generator is None:
            logger.info("Loading async generator")
            from inference.generator import AsyncQwenGenerator
            
            self.async_generator = AsyncQwenGenerator(
                model_name=self.generator_model,
                gpu_memory_utilization=0.9
            )
            await self.async_generator.initialize()
            self._log_memory("Async generator loaded")
    
    async def _unload_async_generator(self):
        """Unload async generator."""
        if self.no_unload:
            return
        if self.async_generator is not None:
            logger.debug("Unloading async generator")
            await self.async_generator.cleanup()
            self.async_generator = None
            clear_gpu_memory()
    
    def _load_hallucination_checker(self):
        """Load hallucination checker."""
        if self.hallucination_checker is None:
            logger.info("Loading hallucination checker")
            from inference.hallucination_checker import HallucinationChecker
            
            self.hallucination_checker = HallucinationChecker(device=self.device)
            self._log_memory("Hallucination checker loaded")
    
    def _unload_hallucination_checker(self):
        """Unload hallucination checker."""
        if self.no_unload:
            return
        if self.hallucination_checker is not None:
            logger.debug("Unloading hallucination checker")
            self.hallucination_checker.cleanup()
            del self.hallucination_checker
            self.hallucination_checker = None
            clear_gpu_memory()
    
    def _log_memory(self, context: str = ""):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.debug(f"GPU Memory ({context}): {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # =========================================================================
    # Session/Tracing Management
    # =========================================================================
    
    def init_session_logger(
        self, 
        session_dir: Optional[Path] = None, 
        session_id: Optional[str] = None
    ) -> Path:
        """Initialize session logger for JSONL logging."""
        from inference.session_logger import SessionLogger
        
        session_dir = session_dir or Path(self.config['paths'].get('session_logs', 'outputs/sessions'))
        self.session_logger = SessionLogger(
            session_dir=session_dir,
            session_id=session_id
        )
        logger.info(f"Session logger initialized: {self.session_logger.session_file}")
        return self.session_logger.session_file
    
    def _init_tracer(self, conversation_id: Optional[str] = None):
        """Initialize request tracer if enabled."""
        if not self.enable_tracing:
            return None
        
        from inference.request_tracer import RequestTracer
        
        self.current_tracer = RequestTracer(
            conversation_id=conversation_id,
            auto_register=True
        )
        return self.current_tracer
    
    def _get_default_config(self) -> PipelineConfig:
        """Get default pipeline configuration from config file."""
        return PipelineConfig(
            k_papers=self.config['retrieval']['k_papers'],
            m_chunks=self.config['retrieval']['m_chunks'],
            n_reranked=self.config['retrieval']['n_reranked'],
            bm25_weight=self.config['retrieval']['bm25_weight'],
            embedding_weight=self.config['retrieval']['embedding_weight'],
            temperature=self.config['generation']['temperature'],
            include_citations=self.config['generation']['include_citations'],
        )
    
    # =========================================================================
    # Stage 1: Hybrid Retrieval (BM25 + Embedding)
    # =========================================================================
    
    def _stage1_hybrid_retrieval(
        self,
        query: str,
        k: int,
        bm25_weight: float,
        embedding_weight: float,
        timings: Dict[str, float]
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """
        Stage 1: Combine BM25 and paper-level embedding search.
        
        Returns:
            Tuple of (selected_files set, retrieval_data dict)
        """
        logger.info(f"Stage 1: Hybrid retrieval (k={k}, bm25={bm25_weight:.2f}, emb={embedding_weight:.2f})")
        
        if not abs(bm25_weight + embedding_weight - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {bm25_weight + embedding_weight}")
        
        retrieval_data = {
            'bm25_results': [],
            'bm25_scores': None,
            'embedding_results': [],
            'file_scores': {},
        }
        
        # --- BM25 Search ---
        self._load_bm25()
        
        t0 = time.perf_counter()
        bm25_results = self.bm25_searcher.search(query, k=k * 2)
        t1 = time.perf_counter()
        timings['bm25'] = t1 - t0
        
        # Try to get raw scores
        try:
            if hasattr(self.bm25_searcher, 'bm25') and self.bm25_searcher.bm25 is not None:
                tokenized_query = query.lower().split()
                scores = self.bm25_searcher.bm25.get_scores(tokenized_query)
                retrieval_data['bm25_scores'] = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.debug(f"Could not get BM25 scores: {e}")
        
        bm25_files = set(bm25_results)
        retrieval_data['bm25_results'] = bm25_results
        
        # Capture to tracer
        if self.current_tracer:
            self.current_tracer.capture_bm25(
                results=bm25_results,
                scores=retrieval_data.get('bm25_scores'),
                duration_ms=timings['bm25'] * 1000,
                k=k * 2
            )
        
        self._unload_bm25()
        logger.info(f"  BM25 retrieved {len(bm25_files)} papers")
        
        # --- Embedding Search (Paper-level) ---
        self._load_embedding_search()
        
        t0 = time.perf_counter()
        emb_results = self.embedding_searcher.search(query, collection_num=1, k=k * 2)
        t1 = time.perf_counter()
        timings['embedding_paper'] = t1 - t0
        
        emb_files = {meta['file_path'] for _, meta, _ in emb_results}
        retrieval_data['embedding_results'] = emb_results
        
        # Capture to tracer
        if self.current_tracer:
            self.current_tracer.capture_embedding_paper(
                results=emb_results,
                duration_ms=timings['embedding_paper'] * 1000,
                k=k * 2
            )
        
        logger.info(f"  Embedding retrieved {len(emb_files)} papers")
        
        # --- Combine with Weighted Voting ---
        all_files = bm25_files.union(emb_files)
        file_scores = {}
        
        for file in all_files:
            # BM25 score: rank-based
            bm25_score = (k * 2 - bm25_results.index(file)) if file in bm25_results else 0
            
            # Embedding score: rank-based
            emb_score = 0
            for idx, (_, meta, _) in enumerate(emb_results):
                if meta['file_path'] == file:
                    emb_score = k * 2 - idx
                    break
            
            # Normalize to [0, 1] and apply weights
            bm25_norm = bm25_score / (k * 2) if k > 0 else 0
            emb_norm = emb_score / (k * 2) if k > 0 else 0
            
            file_scores[file] = bm25_weight * bm25_norm + embedding_weight * emb_norm
        
        retrieval_data['file_scores'] = file_scores
        
        # Get top-k by combined score
        top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected_files = {file for file, _ in top_files}
        
        # Capture hybrid to tracer
        if self.current_tracer:
            self.current_tracer.capture_hybrid(
                file_scores=file_scores,
                bm25_results=bm25_results,
                emb_results=emb_results,
                selected_files=selected_files,
                bm25_weight=bm25_weight,
                embedding_weight=embedding_weight,
                k=k
            )
        
        # Log to session logger
        if self.session_logger:
            self.session_logger.log_bm25_results(bm25_results)
            self.session_logger.log_embedding_stage1_results(emb_results)
            self.session_logger.log_hybrid_scores(
                file_scores=file_scores,
                bm25_results=bm25_results,
                emb_results=emb_results,
                selected_files=selected_files,
                bm25_weight=bm25_weight,
                embedding_weight=embedding_weight,
                k=k
            )
        
        logger.info(f"  Combined ranking selected {len(selected_files)} papers")
        
        return selected_files, retrieval_data
    
    # =========================================================================
    # Stage 2: Chunk Retrieval
    # =========================================================================
    
    def _stage2_chunk_retrieval(
        self,
        query: str,
        file_filter: Set[str],
        m: int,
        timings: Dict[str, float]
    ) -> List[Tuple[float, Dict, str]]:
        """
        Stage 2: Retrieve chunks from selected papers.
        
        Returns:
            List of (distance, metadata, text) tuples
        """
        logger.info(f"Stage 2: Chunk retrieval (m={m}, filtered to {len(file_filter)} papers)")
        
        # Embedding searcher should still be loaded from Stage 1
        if self.embedding_searcher is None:
            self._load_embedding_search()
        
        t0 = time.perf_counter()
        chunk_results = self.embedding_searcher.search(
            query=query,
            collection_num=2,
            k=m,
            file_path_filter=file_filter
        )
        t1 = time.perf_counter()
        timings['embedding_chunk'] = t1 - t0
        
        # Capture to tracer
        if self.current_tracer:
            self.current_tracer.capture_embedding_chunk(
                results=chunk_results,
                duration_ms=timings['embedding_chunk'] * 1000,
                m=m,
                file_filter=file_filter
            )
        
        # Log to session logger
        if self.session_logger:
            self.session_logger.log_chunk_results(chunk_results)
        
        logger.info(f"  Retrieved {len(chunk_results)} chunks")
        
        # Unload embedding search now
        self._unload_embedding_search()
        
        return chunk_results
    
    # =========================================================================
    # Stage 3: Reranking
    # =========================================================================
    
    def _stage3_reranking(
        self,
        query: str,
        chunks: List[Tuple[float, Dict, str]],
        n: int,
        timings: Dict[str, float]
    ) -> List[Dict]:
        """
        Stage 3: Rerank chunks using cross-encoder.
        
        Returns:
            List of reranked result dicts
        """
        logger.info(f"Stage 3: Reranking ({len(chunks)} chunks → top {n})")
        
        self._load_reranker()
        
        t0 = time.perf_counter()
        reranked = self.reranker.rerank_with_details(
            query=query,
            candidates=chunks,
            top_k=n
        )
        t1 = time.perf_counter()
        timings['reranker'] = t1 - t0
        
        # Capture to tracer
        if self.current_tracer:
            self.current_tracer.capture_reranker(
                reranked=reranked,
                original_chunks=chunks,
                duration_ms=timings['reranker'] * 1000,
                n=n
            )
        
        # Log to session logger
        if self.session_logger:
            self.session_logger.log_reranker_results(reranked, chunks)
        
        logger.info(f"  Selected top {len(reranked)} chunks")
        
        self._unload_reranker()
        
        return reranked
    
    # =========================================================================
    # Stage 4: Generation (Sync wrapper around async)
    # =========================================================================
    
    def _stage4_generation(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float,
        include_citations: bool,
        timings: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Stage 4: Generate answer using LLM.
        
        Uses async generator internally but provides sync interface.
        
        Returns:
            Generation response dict with 'answer' key
        """
        logger.info(f"Stage 4: Generating answer ({len(contexts)} contexts)")
        
        self._load_generator()
        
        async def _generate():
            """Inner async function for generation."""
            full_response = ""
            async for token in self.async_generator.generate_streaming(
                query=query,
                contexts=contexts,
                temperature=temperature,
                include_citations=include_citations
            ):
                full_response += token
            return full_response
        
        t0 = time.perf_counter()
        
        # Run async generation in sync context
        answer = asyncio.run(_generate())
        
        t1 = time.perf_counter()
        timings['generator'] = t1 - t0
        
        response = {
            'answer': answer,
            'num_contexts_used': len(contexts),
            'generation_params': {
                'temperature': temperature,
                'include_citations': include_citations,
            }
        }
        
        # Capture to tracer
        if self.current_tracer:
            self.current_tracer.capture_generator(
                answer=answer,
                contexts=contexts,
                duration_ms=timings['generator'] * 1000,
                temperature=temperature,
                include_citations=include_citations
            )
        
        # Log to session logger
        if self.session_logger:
            self.session_logger.log_generation(
                answer=answer,
                contexts=contexts,
                generation_params=response.get('generation_params', {}),
                prompt_length_tokens=None
            )
        
        logger.info(f"  Generated {len(answer.split())} words")
        
        self._unload_generator()
        
        return response
    
    # =========================================================================
    # Stage 4: Generation (Async Streaming)
    # =========================================================================
    
    async def _stage4_generation_streaming(
        self,
        query: str,
        contexts: List[Dict],
        temperature: float,
        include_citations: bool,
        timings: Dict[str, float]
    ) -> AsyncGenerator[str, None]:
        """
        Stage 4: Stream generated answer tokens.
        
        Yields:
            Generated tokens
        """
        logger.info(f"Stage 4: Streaming answer ({len(contexts)} contexts)")
        
        await self._load_async_generator()
        
        t0 = time.perf_counter()
        accumulated = ""
        
        async for token in self.async_generator.generate_streaming(
            query=query,
            contexts=contexts,
            temperature=temperature,
            include_citations=include_citations
        ):
            accumulated += token
            yield token
        
        t1 = time.perf_counter()
        timings['generator'] = t1 - t0
        
        # Capture to tracer (after streaming completes)
        if self.current_tracer:
            self.current_tracer.capture_generator(
                answer=accumulated,
                contexts=contexts,
                duration_ms=timings['generator'] * 1000,
                temperature=temperature,
                include_citations=include_citations
            )
        
        # Log to session logger
        if self.session_logger:
            self.session_logger.log_generation(
                answer=accumulated,
                contexts=contexts,
                generation_params={'temperature': temperature, 'include_citations': include_citations},
                prompt_length_tokens=None
            )
        
        logger.info(f"  Streamed {len(accumulated.split())} words")
        
        await self._unload_async_generator()
    
    # =========================================================================
    # Stage 5: Hallucination Check
    # =========================================================================
    
    def _stage5_hallucination_check(
        self,
        answer: str,
        contexts: List[Dict],
        timings: Dict[str, float]
    ) -> Optional[Any]:
        """
        Stage 5: Verify claims against source documents.
        
        Returns:
            HallucinationResult or None
        """
        if not self.enable_hallucination_check:
            return None
        
        logger.info("Stage 5: Hallucination check")
        
        self._load_hallucination_checker()
        
        t0 = time.perf_counter()
        try:
            result = self.hallucination_checker.check_answer_sync(
                answer=answer,
                contexts=contexts
            )
            
            logger.info(
                f"  {result.num_grounded}/{result.num_claims} claims grounded "
                f"({result.grounding_ratio:.0%})"
            )
            
        except Exception as e:
            logger.error(f"Hallucination check failed: {e}", exc_info=True)
            result = None
        
        t1 = time.perf_counter()
        timings['hallucination_check'] = t1 - t0
        
        # Capture to tracer
        if self.current_tracer and result:
            self.current_tracer.capture_hallucination(
                result=result,
                duration_ms=timings['hallucination_check'] * 1000
            )
        
        # Log to session logger
        if self.session_logger and result:
            self.session_logger.log_hallucination_check(result)
        
        self._unload_hallucination_checker()
        
        return result
    
    # =========================================================================
    # Main Entry Points
    # =========================================================================
    
    def answer(
        self,
        query: str,
        k: Optional[int] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        embedding_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        include_citations: Optional[bool] = None,
        return_contexts: bool = False,
        conversation_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Answer a question using the full RAG pipeline (sync).
        
        Args:
            query: User question
            k: Number of papers to retrieve (Stage 1)
            m: Number of chunks to retrieve (Stage 2)
            n: Number of chunks after reranking (Stage 3)
            bm25_weight: Weight for BM25 in hybrid retrieval
            embedding_weight: Weight for embedding in hybrid retrieval
            temperature: LLM sampling temperature
            include_citations: Whether to request citations
            return_contexts: If True, include contexts in result
            conversation_id: Optional conversation ID for tracing
        
        Returns:
            PipelineResult with answer and metadata
        """
        # Build config
        defaults = self._get_default_config()
        config = PipelineConfig(
            k_papers=k or defaults.k_papers,
            m_chunks=m or defaults.m_chunks,
            n_reranked=n or defaults.n_reranked,
            bm25_weight=bm25_weight if bm25_weight is not None else defaults.bm25_weight,
            embedding_weight=embedding_weight if embedding_weight is not None else defaults.embedding_weight,
            temperature=temperature if temperature is not None else defaults.temperature,
            include_citations=include_citations if include_citations is not None else defaults.include_citations,
        )
        
        # Set request context for logging
        req_id = set_request_context(conversation_id=conversation_id)
        
        # Initialize tracer
        tracer = self._init_tracer(conversation_id=conversation_id)
        if tracer:
            tracer.start_request(query=query, config=config.to_dict())
        
        # Initialize session logger entry
        if self.session_logger:
            self.session_logger.start_entry(query, config.to_dict())
        
        logger.info("=" * 60)
        logger.info(f"Query: {query}")
        logger.info("=" * 60)
        
        start_time = time.perf_counter()
        timings: Dict[str, float] = {}
        
        try:
            # Stage 1: Hybrid Retrieval
            selected_papers, _ = self._stage1_hybrid_retrieval(
                query=query,
                k=config.k_papers,
                bm25_weight=config.bm25_weight,
                embedding_weight=config.embedding_weight,
                timings=timings
            )
            
            # Stage 2: Chunk Retrieval
            chunk_results = self._stage2_chunk_retrieval(
                query=query,
                file_filter=selected_papers,
                m=config.m_chunks,
                timings=timings
            )
            
            # Stage 3: Reranking
            reranked_chunks = self._stage3_reranking(
                query=query,
                chunks=chunk_results,
                n=config.n_reranked,
                timings=timings
            )
            
            # Stage 4: Generation
            generation_response = self._stage4_generation(
                query=query,
                contexts=reranked_chunks,
                temperature=config.temperature,
                include_citations=config.include_citations,
                timings=timings
            )
            
            # Stage 5: Hallucination Check
            hallucination_result = self._stage5_hallucination_check(
                answer=generation_response['answer'],
                contexts=reranked_chunks,
                timings=timings
            )
            
            # Build result
            elapsed = time.perf_counter() - start_time
            
            result = PipelineResult(
                query=query,
                answer=generation_response['answer'],
                timestamp=datetime.now().isoformat(),
                config=config.to_dict(),
                papers_selected=len(selected_papers),
                chunks_retrieved=len(chunk_results),
                contexts_used=len(reranked_chunks),
                elapsed_seconds=elapsed,
                contexts=reranked_chunks if return_contexts else None,
                hallucination_result=hallucination_result.to_dict() if hallucination_result else None,
                trace_id=tracer.get_trace().trace_id if tracer else None,
            )
            
            # Log timings
            if self.session_logger:
                self.session_logger.log_timings(timings)
                self.session_logger.finish_entry()
            
            # Finalize tracer
            if tracer:
                trace = tracer.finish_request(success=True)
                # Save trace
                self.traces_path.mkdir(parents=True, exist_ok=True)
                tracer.save(trace, self.traces_path)
            
            # Log summary
            log_request_summary(
                logger,
                total_duration_ms=elapsed * 1000,
                stages=timings,
                success=True,
                papers_selected=len(selected_papers),
                chunks_retrieved=len(chunk_results)
            )
            
            logger.info("=" * 60)
            logger.info(f"Pipeline completed in {elapsed:.2f}s")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            if self.session_logger:
                self.session_logger.log_error(str(e))
                self.session_logger.finish_entry()
            
            if tracer:
                tracer.finish_request(success=False, error=str(e))
            
            raise
        
        finally:
            clear_request_context()
            if tracer:
                tracer.cleanup()
    
    async def answer_streaming(
        self,
        query: str,
        k: Optional[int] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        embedding_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        include_citations: Optional[bool] = None,
        conversation_id: Optional[str] = None
    ) -> AsyncGenerator[Any, None]:
        """
        Answer with streaming output.
        
        Yields:
            First: dict with contexts and metadata
            Then: str tokens as they're generated
            Finally: dict with hallucination results (if enabled)
        """
        # Build config
        defaults = self._get_default_config()
        config = PipelineConfig(
            k_papers=k or defaults.k_papers,
            m_chunks=m or defaults.m_chunks,
            n_reranked=n or defaults.n_reranked,
            bm25_weight=bm25_weight if bm25_weight is not None else defaults.bm25_weight,
            embedding_weight=embedding_weight if embedding_weight is not None else defaults.embedding_weight,
            temperature=temperature if temperature is not None else defaults.temperature,
            include_citations=include_citations if include_citations is not None else defaults.include_citations,
        )
        
        req_id = set_request_context(conversation_id=conversation_id)
        tracer = self._init_tracer(conversation_id=conversation_id)
        if tracer:
            tracer.start_request(query=query, config=config.to_dict())
        
        if self.session_logger:
            self.session_logger.start_entry(query, config.to_dict())
        
        logger.info(f"Streaming query: {query}")
        
        start_time = time.perf_counter()
        timings: Dict[str, float] = {}
        accumulated_answer = ""
        
        try:
            # Stages 1-3 (sync)
            selected_papers, _ = self._stage1_hybrid_retrieval(
                query=query,
                k=config.k_papers,
                bm25_weight=config.bm25_weight,
                embedding_weight=config.embedding_weight,
                timings=timings
            )
            
            chunk_results = self._stage2_chunk_retrieval(
                query=query,
                file_filter=selected_papers,
                m=config.m_chunks,
                timings=timings
            )
            
            reranked_chunks = self._stage3_reranking(
                query=query,
                chunks=chunk_results,
                n=config.n_reranked,
                timings=timings
            )
            
            # Yield metadata first
            yield {
                'type': 'metadata',
                'contexts': reranked_chunks,
                'papers_selected': len(selected_papers),
                'chunks_retrieved': len(chunk_results),
            }
            
            # Stage 4: Streaming generation
            async for token in self._stage4_generation_streaming(
                query=query,
                contexts=reranked_chunks,
                temperature=config.temperature,
                include_citations=config.include_citations,
                timings=timings
            ):
                accumulated_answer += token
                yield {'type': 'token', 'content': token}
            
            # Stage 5: Hallucination check
            hallucination_result = self._stage5_hallucination_check(
                answer=accumulated_answer,
                contexts=reranked_chunks,
                timings=timings
            )
            
            if hallucination_result:
                yield {
                    'type': 'hallucination',
                    'result': hallucination_result.to_dict()
                }
            
            # Log timings
            if self.session_logger:
                self.session_logger.log_timings(timings)
                self.session_logger.finish_entry()
            
            if tracer:
                trace = tracer.finish_request(success=True)
                self.traces_path.mkdir(parents=True, exist_ok=True)
                tracer.save(trace, self.traces_path)
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Streaming pipeline completed in {elapsed:.2f}s")
            
            # Final summary
            yield {
                'type': 'complete',
                'elapsed_seconds': elapsed,
                'trace_id': tracer.get_trace().trace_id if tracer else None,
            }
            
        except Exception as e:
            logger.error(f"Streaming pipeline failed: {e}", exc_info=True)
            
            if self.session_logger:
                self.session_logger.log_error(str(e))
                self.session_logger.finish_entry()
            
            if tracer:
                tracer.finish_request(success=False, error=str(e))
            
            yield {'type': 'error', 'message': str(e)}
            
        finally:
            clear_request_context()
            if tracer:
                tracer.cleanup()
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def cleanup(self):
        """Unload all models and free memory."""
        logger.info("Cleaning up pipeline")
        
        # Force unload
        original_no_unload = self.no_unload
        self.no_unload = False
        
        self._unload_bm25()
        self._unload_embedding_search()
        self._unload_reranker()
        self._unload_generator()
        self._unload_hallucination_checker()
        
        self.no_unload = original_no_unload
        
        logger.info("Cleanup complete")


# =============================================================================
# CLI / Testing
# =============================================================================

def main():
    """Example usage and testing."""
    import os
    import json
    
    # Optimize memory
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        enable_hallucination_check=False,
        enable_tracing=False,
        device='cpu'
    )
    
    # Initialize session logger
    # session_file = pipeline.init_session_logger()
    # print(f"Session logging to: {session_file}")
    
    # Test query
    query = "What is matrix suppression and how does it relate to ion competition in mass spectrometry?"
    
    print(f"\nQuery: {query}\n")
    print("=" * 60)
    
    # Get answer
    result = pipeline.answer(
        query=query,
        return_contexts=True,
        conversation_id="test-session"
    )
    
    # Display result
    print("\nANSWER:")
    print("-" * 60)
    print(result.answer)
    print("-" * 60)
    
    print(f"\nMetadata:")
    print(f"  Papers selected: {result.papers_selected}")
    print(f"  Chunks retrieved: {result.chunks_retrieved}")
    print(f"  Contexts used: {result.contexts_used}")
    print(f"  Elapsed: {result.elapsed_seconds:.2f}s")
    print(f"  Trace ID: {result.trace_id}")
    
    if result.hallucination_result:
        hr = result.hallucination_result
        print(f"\nHallucination Check:")
        print(f"  Claims: {hr['num_claims']}")
        print(f"  Grounded: {hr['num_grounded']}")
        print(f"  Ratio: {hr['grounding_ratio']:.0%}")
    
    # Save result
    output_path = Path(pipeline.config['paths'].get('outputs', 'outputs')) / "rag_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(include_contexts=True), f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved result to: {output_path}")
    
    # Cleanup
    pipeline.cleanup()
    print("\nDone!")


async def main_streaming():
    """Test streaming mode."""
    import os
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    pipeline = RAGPipeline(
        enable_hallucination_check=True,
        enable_tracing=True
    )
    
    query = "What are the key challenges in MALDI imaging?"
    
    print(f"\nStreaming Query: {query}\n")
    print("=" * 60)
    print("Response: ", end="", flush=True)
    
    async for item in pipeline.answer_streaming(
        query=query,
        conversation_id="streaming-test"
    ):
        if item['type'] == 'token':
            print(item['content'], end="", flush=True)
        elif item['type'] == 'metadata':
            pass  # Could show sources here
        elif item['type'] == 'hallucination':
            print(f"\n\n[Hallucination check: {item['result']['grounding_ratio']:.0%} grounded]")
        elif item['type'] == 'complete':
            print(f"\n\n[Completed in {item['elapsed_seconds']:.2f}s]")
        elif item['type'] == 'error':
            print(f"\n\nERROR: {item['message']}")
    
    pipeline.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--stream':
        asyncio.run(main_streaming())
    else:
        main()