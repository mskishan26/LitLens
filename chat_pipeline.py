"""
RAG Chat Pipeline V2 - With Clean DynamoDB Persistence
=======================================================

Simplified pipeline with:
- TraceBuilder for accumulating all pipeline stages
- Save-at-start pattern: trace saved immediately, updated on completion
- Structured answer storage (components stored separately, frontend renders)
- Eager hallucination model loading option
- Single message_id (serves as both message and trace identifier)

Usage:
    pipeline = RAGPipelineV2(
        config_path="config.yaml",
        use_dynamodb=True,
        eager_load_hallucination=True,  # Load model at startup
    )
    await pipeline.initialize()
    
    async for event in pipeline.answer_stream(
        query="What is X?",
        conversation_id="chat-123",
        user_id="user-456",
        is_anonymous=False,
        enable_hallucination_check=True,
    ):
        print(event)
"""

import asyncio
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
import os

from utils.config_loader import load_config
from utils.logger import get_logger, set_request_context, clear_request_context

logger = get_logger(__name__)


def _reranker_to_trace_format(results: List[Dict]) -> Dict[str, Dict]:
    """Convert reranker results to trace-friendly format."""
    return {
        chunk["metadata"]["chroma_id"]: {
            "rank": chunk.get("rank"),
            "rerank_score": chunk.get("rerank_score"),
            "original_distance": chunk.get("original_distance"),
            "original_rank": chunk.get("original_rank"),
        }
        for chunk in results
    }


@dataclass
class RetrievalResult:
    """Container for retrieval stage outputs."""
    reranked_results: List[Dict]
    selected_papers: Set[str]
    req_id: str
    retrieval_duration_ms: float


class RAGPipelineV2:
    """
    Concurrent RAG Pipeline with GPU-serialized generation.
    Clean DynamoDB persistence integration.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_dynamodb: bool = False,
        dynamodb_persistence: Optional[Any] = None,
        eager_load_hallucination: bool = True,  # Load HHEM at startup
    ):
        self.config = load_config(config_path)
        
        self.paths = self.config["paths"]
        self.models = self.config["models"]
        self.pipeline_config = self.config["pipeline_config"]
        self.setup_config = self.config["setup_config"]
        
        # Device configs
        self.embedding_config = self.setup_config["embedding"]
        self.reranker_config = self.setup_config["reranker"]
        self.generator_config = self.setup_config["generator"]
        self.hallucination_config = self.setup_config["hallucination_eval"]
        
        # Components (lazy loaded)
        self.bm25: Any = None
        self.embedding: Any = None
        self.reranker: Any = None
        self.generator: Any = None
        self.hallucination_checker: Any = None
        
        # Persistence
        self.use_dynamodb = use_dynamodb
        self.persistence = dynamodb_persistence
        self.enable_tracing = self.pipeline_config.get("enable_tracing", True)
        
        # Hallucination loading strategy
        self.eager_load_hallucination = eager_load_hallucination
        
        # For SQLite fallback
        if self.enable_tracing and not use_dynamodb:
            self.trace_db_path = os.path.join(self.paths.get("logs", "logs"), "traces.db")
            self._sqlite_tracer = None
        
        # GPU semaphore
        self._gpu_semaphore = asyncio.Semaphore(
            self.pipeline_config.get("max_concurrent_generation", 1)
        )
        self._generation_queue_depth = 0
        self._queue_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing RAG Pipeline V2...")
        
        import torch
        from inference.bm25_search import BM25Searcher
        from inference.embedding_search import EmbeddingSearch
        from inference.reranker import Reranker
        from inference.generator import AsyncQwenGenerator
        from inference.hallucination_checker import HallucinationChecker
        
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.warning("CUDA not available, using CPU")
        
        # SQLite tracer if not using DynamoDB
        if self.enable_tracing and not self.use_dynamodb:
            from utils.request_tracer import RequestTracer
            self._sqlite_tracer = RequestTracer(db_path=self.trace_db_path)
            logger.info(f"Using SQLite tracer at {self.trace_db_path}")
        
        # BM25
        logger.info("Loading BM25...")
        self.bm25 = BM25Searcher(artifacts_dir=self.paths["bm25_artifacts"])
        self.bm25.load_bm25_artifacts()
        
        # Embedding
        logger.info("Loading Embedding model...")
        self.embedding = EmbeddingSearch(
            embedding_model_name=self.models["embedding"].get("path") or self.models["embedding"]["id"],
            **self.embedding_config,
        )
        self.embedding.load(Path(self.paths["embeddings"]))
        
        # Reranker
        logger.info("Loading Reranker...")
        self.reranker = Reranker(
            model=self.models["reranker"].get("path") or self.models["reranker"]["id"],
            **self.reranker_config,
        )
        
        # Generator
        logger.info("Loading Generator (vLLM)...")
        self.generator = AsyncQwenGenerator(
            model_name=self.models["generator"].get("path") or self.models["generator"]["id"],
            **self.generator_config,
        )
        await self.generator.initialize()
        await self._warmup_generator()
        
        # Hallucination checker
        logger.info("Initializing Hallucination Checker...")
        self.hallucination_checker = HallucinationChecker(
            generator=self.generator,
            hallucination_model=self.models["hallucination_eval"].get("path") or self.models["hallucination_eval"]["id"],
            **self.hallucination_config,
        )
        
        # Eager load HHEM model to avoid cold start on first hallucination check
        if self.eager_load_hallucination:
            logger.info("Eager loading HHEM model...")
            await self._warmup_hallucination_checker()
        
        logger.info("Pipeline V2 initialized successfully")

    async def _warmup_generator(self):
        """Warmup vLLM to capture CUDA graphs."""
        logger.info("Warming up vLLM...")
        try:
            async with self._gpu_semaphore:
                async for _ in self.generator.generate_streaming(
                    query="warmup",
                    contexts=[{"text": "warmup context", "metadata": {}, "score": 1.0}],
                    temperature=0.7,
                    max_tokens=5,
                ):
                    pass
            logger.info("vLLM warmup complete")
        except Exception as e:
            logger.error(f"vLLM warmup failed: {e}")

    async def _warmup_hallucination_checker(self):
        """
        Warmup hallucination checker by loading the HHEM model.
        This avoids cold start latency on the first hallucination check.
        """
        try:
            # Force model load by calling internal method
            self.hallucination_checker._load_hhem()
            logger.info("HHEM model loaded successfully")
        except Exception as e:
            logger.warning(f"HHEM warmup failed (will load on first use): {e}")

    async def cleanup(self):
        """Free all resources."""
        import torch
        logger.info("Cleaning up pipeline...")
        
        if self.hallucination_checker:
            self.hallucination_checker.cleanup()
        if self.generator:
            await self.generator.cleanup()
        if self._sqlite_tracer:
            self._sqlite_tracer.close()
        
        self.hallucination_checker = None
        self.generator = None
        self.bm25 = None
        self.embedding = None
        self.reranker = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def answer_stream(
        self,
        query: str,
        conversation_id: str = "default-session",
        user_id: Optional[str] = None,
        is_anonymous: bool = False,
        enable_hallucination_check: bool = False,
        message_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query through the RAG pipeline.
        
        Yields events:
        - status: Pipeline stage updates
        - context: Retrieved sources
        - token: Generated text tokens
        - hallucination: Verification results (if enabled)
        - done: Completion with message_id (which is also the trace identifier)
        - error: Error details
        
        Args:
            query: The user's question
            conversation_id: Chat/conversation ID (set by frontend)
            user_id: Firebase UID (set by frontend after auth)
            is_anonymous: Whether this is an anonymous Firebase user
            enable_hallucination_check: Whether to run hallucination checking
        """
        start_time = time.perf_counter()
        req_id = set_request_context(conversation_id=conversation_id)
        user_id = user_id or "anonymous"
        
        # Initialize trace
        trace = None
        if self.enable_tracing and self.use_dynamodb and self.persistence:
            from dynamo_persistence import TraceBuilder
            trace = TraceBuilder(
                chat_id=conversation_id,
                user_id=user_id,
                query=query,
                is_anonymous=is_anonymous,
                message_id=message_id,
            )
            
            # Save trace immediately with status="running"
            # This ensures abandoned requests are visible
            await self.persistence.save_trace_start(trace)
            logger.info(f"[{req_id[:8]}] Trace started: message_id={trace.message_id}")
        
        # Accumulators for structured storage
        accumulated_answer = ""
        sources_for_storage = []
        hallucination_result = None
        
        try:
            # ===== STAGES 1-3: Retrieval =====
            retrieval_result = await self._run_retrieval(
                query=query,
                req_id=req_id,
                trace=trace,
            )
            
            yield {
                "type": "status",
                "stage": "retrieval_complete",
                "papers_found": len(retrieval_result.selected_papers),
                "chunks_reranked": len(retrieval_result.reranked_results),
            }
            
            # Prepare sources (store full data for trace, minimal for chat)
            sources_for_storage = [
                {
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": r["rerank_score"],
                }
                for r in retrieval_result.reranked_results
            ]
            
            yield {"type": "context", "data": sources_for_storage}
            
            # ===== STAGES 4-5: Generation + Hallucination =====
            async for event in self._run_generation(
                query=query,
                reranked_results=retrieval_result.reranked_results,
                req_id=req_id,
                trace=trace,
                enable_hallucination_check=enable_hallucination_check,
            ):
                # Accumulate answer tokens
                if event.get("type") == "token":
                    accumulated_answer += event.get("content", "")
                elif event.get("type") == "hallucination":
                    hallucination_result = event
                
                yield event
            
            # ===== PERSIST =====
            total_duration = (time.perf_counter() - start_time) * 1000
            
            if trace and self.persistence:
                trace.complete()
                
                # Save structured data (frontend renders the display)
                # This updates the trace (created at start) and saves message + metadata
                await self.persistence.save_turn(
                    trace=trace,
                    answer=accumulated_answer,  # Raw answer, not pre-formatted
                    sources=sources_for_storage,
                    hallucination_result=hallucination_result,
                )
                logger.info(f"[{req_id[:8]}] Saved to DynamoDB: message_id={trace.message_id}")
            
            # message_id is now the sole identifier (no separate trace_id)
            yield {
                "type": "done",
                "message_id": trace.message_id if trace else None,
                "chat_id": conversation_id,
                "total_duration_ms": total_duration,
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            
            if trace and self.persistence:
                failed_stage = self._detect_failed_stage(trace)
                trace.fail(failed_stage, str(e))
                
                # Still save what we have (updates the running trace to failed)
                await self.persistence.save_turn(
                    trace=trace,
                    answer=accumulated_answer or "",
                    sources=sources_for_storage,
                    hallucination_result=hallucination_result,
                )
            
            yield {
                "type": "error",
                "message": str(e),
                "message_id": trace.message_id if trace else None,
            }
        
        finally:
            clear_request_context()

    def _detect_failed_stage(self, trace) -> str:
        """Determine which stage failed based on what's in the trace."""
        stages_order = [
            "bm25", "embedding_paper", "hybrid_fusion", "embedding_chunk",
            "reranker", "generator", "hallucination"
        ]
        
        for stage in stages_order:
            if stage not in trace.stages:
                return stage
        
        return "unknown"

    # =========================================================================
    # RETRIEVAL STAGES (1-3)
    # =========================================================================

    async def _run_retrieval(
        self,
        query: str,
        req_id: str,
        trace: Optional[Any],
    ) -> RetrievalResult:
        """Run stages 1-3: BM25, Embedding, Hybrid Fusion, Chunk Retrieval, Reranking."""
        import torch
        start_time = time.perf_counter()
        
        # ===== Stage 1a: BM25 Search =====
        logger.info(f"[{req_id[:8]}] Stage 1a: BM25 Search")
        bm25_start = time.perf_counter()
        bm25_output = self.bm25.search(query, k=self.pipeline_config["k_papers"] * 2)
        bm25_results = [fp for fp, _ in bm25_output]
        bm25_duration = (time.perf_counter() - bm25_start) * 1000
        
        if trace:
            trace.add_stage(
                "bm25",
                {
                    "results": [
                        {"file_path": fp, "score": round(s, 4)} 
                        for fp, s in bm25_output[:50]
                    ],
                    "total": len(bm25_output),
                },
                duration_ms=bm25_duration,
                papers_retrieved=len(bm25_output),
            )
        
        # ===== Stage 1b: Embedding Paper Search =====
        logger.info(f"[{req_id[:8]}] Stage 1b: Embedding Paper Search")
        emb_start = time.perf_counter()
        emb_paper_results = self.embedding.search(
            query, collection_num=1, k=self.pipeline_config["k_papers"] * 2
        )
        emb_duration = (time.perf_counter() - emb_start) * 1000
        
        if trace:
            trace.add_stage(
                "embedding_paper",
                {
                    "results": [
                        {"chroma_id": r[1].get("chroma_id"), "distance": round(r[0], 4)}
                        for r in emb_paper_results[:50]
                    ],
                    "total": len(emb_paper_results),
                },
                duration_ms=emb_duration,
            )
        
        # ===== Stage 1c: Hybrid Fusion =====
        logger.info(f"[{req_id[:8]}] Stage 1c: Hybrid Fusion")
        fusion_start = time.perf_counter()
        selected_papers, fusion_scores = self._hybrid_fusion(
            bm25_results, emb_paper_results,
            k=self.pipeline_config["k_papers"],
            return_scores=True,
        )
        fusion_duration = (time.perf_counter() - fusion_start) * 1000
        
        if trace:
            trace.add_stage(
                "hybrid_fusion",
                {
                    "selected_papers": list(selected_papers),
                    "fusion_scores": {
                        k: round(v, 4) 
                        for k, v in sorted(fusion_scores.items(), key=lambda x: -x[1])[:20]
                    },
                    "num_selected": len(selected_papers),
                },
                duration_ms=fusion_duration,
            )
        
        # ===== Stage 2: Chunk Retrieval =====
        logger.info(f"[{req_id[:8]}] Stage 2: Chunk Retrieval")
        chunk_start = time.perf_counter()
        chunk_results = self.embedding.search(
            query,
            collection_num=2,
            k=self.pipeline_config["m_chunks"],
            file_path_filter=selected_papers,
        )
        chunk_duration = (time.perf_counter() - chunk_start) * 1000
        
        if trace:
            trace.add_stage(
                "embedding_chunk",
                {
                    "results": [
                        {"chroma_id": r[1].get("chroma_id"), "distance": round(r[0], 4)}
                        for r in chunk_results[:100]
                    ],
                    "total": len(chunk_results),
                },
                duration_ms=chunk_duration,
                chunks_retrieved=len(chunk_results),
            )
        
        # ===== Stage 3: Reranking =====
        logger.info(f"[{req_id[:8]}] Stage 3: Reranking")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        rerank_start = time.perf_counter()
        reranked_results = self.reranker.rerank_with_details(
            query,
            candidates=chunk_results,
            top_k=self.pipeline_config["n_reranked"],
        )
        rerank_duration = (time.perf_counter() - rerank_start) * 1000
        
        if trace:
            trace.add_stage(
                "reranker",
                {
                    "results": _reranker_to_trace_format(reranked_results),
                    "num_reranked": len(reranked_results),
                },
                duration_ms=rerank_duration,
                chunks_reranked=len(reranked_results),
            )
        
        retrieval_duration = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{req_id[:8]}] Retrieval complete: {retrieval_duration:.0f}ms")
        
        return RetrievalResult(
            reranked_results=reranked_results,
            selected_papers=selected_papers,
            req_id=req_id,
            retrieval_duration_ms=retrieval_duration,
        )

    # =========================================================================
    # GENERATION STAGES (4-5)
    # =========================================================================

    async def _run_generation(
        self,
        query: str,
        reranked_results: List[Dict],
        req_id: str,
        trace: Optional[Any],
        enable_hallucination_check: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run stage 4 (generation) and optionally stage 5 (hallucination)."""
        
        # Queue management
        async with self._queue_lock:
            self._generation_queue_depth += 1
            queue_pos = self._generation_queue_depth
        
        if queue_pos > 1:
            logger.info(f"[{req_id[:8]}] Waiting for GPU (position {queue_pos})")
            yield {"type": "status", "stage": "waiting_for_gpu", "queue_position": queue_pos}
        
        wait_start = time.perf_counter()
        
        async with self._gpu_semaphore:
            wait_time = (time.perf_counter() - wait_start) * 1000
            
            async with self._queue_lock:
                self._generation_queue_depth -= 1
            
            if wait_time > 100:
                logger.info(f"[{req_id[:8]}] GPU acquired after {wait_time:.0f}ms")
            
            # ===== Stage 4: Generation =====
            logger.info(f"[{req_id[:8]}] Stage 4: Generation")
            gen_start = time.perf_counter()
            
            accumulated_text = ""
            token_count = 0
            ttft = None
            
            async for token in self.generator.generate_streaming(
                query=query,
                contexts=reranked_results,
                temperature=self.generator_config["temperature"],
                include_citations=self.pipeline_config["include_citations"],
            ):
                if ttft is None:
                    ttft = (time.perf_counter() - gen_start) * 1000
                accumulated_text += token
                token_count += 1
                yield {"type": "token", "content": token}
            
            gen_duration = (time.perf_counter() - gen_start) * 1000
            
            if trace:
                trace.add_stage(
                    "generator",
                    {
                        "answer_length": len(accumulated_text),
                        "completion_tokens": token_count,
                        "ttft_ms": round(ttft, 2) if ttft else None,
                    },
                    duration_ms=gen_duration,
                )
            
            # ===== Stage 5: Hallucination Check (Optional) =====
            if enable_hallucination_check and self.hallucination_checker:
                logger.info(f"[{req_id[:8]}] Stage 5: Hallucination Check")
                hal_start = time.perf_counter()
                
                hal_contexts = [
                    {"text": r["text"], "metadata": r["metadata"]}
                    for r in reranked_results
                ]
                
                hal_result = await self.hallucination_checker.check(
                    answer=accumulated_text,
                    contexts=hal_contexts,
                )
                hal_duration = (time.perf_counter() - hal_start) * 1000
                
                if trace:
                    trace.add_stage(
                        "hallucination",
                        {
                            "num_claims": hal_result["num_claims"],
                            "num_grounded": hal_result["num_grounded"],
                            "grounding_ratio": hal_result["grounding_ratio"],
                            "unsupported_claims": hal_result["unsupported_claims"],
                            "verifications": hal_result["verifications"],
                        },
                        duration_ms=hal_duration,
                        grounding_ratio=hal_result["grounding_ratio"],
                    )
                
                yield {
                    "type": "hallucination",
                    "grounding_ratio": hal_result["grounding_ratio"],
                    "num_claims": hal_result["num_claims"],
                    "num_grounded": hal_result["num_grounded"],
                    "unsupported_claims": hal_result["unsupported_claims"],
                    "verifications": hal_result["verifications"],
                }

    # =========================================================================
    # TITLE GENERATION (for chat rename)
    # =========================================================================

    async def generate_chat_title(
        self,
        queries: List[str],
        max_tokens: int = 30,
    ) -> str:
        """
        Generate a concise title for a chat based on the first few queries.
        
        Uses the already-loaded generator to create a descriptive title.
        
        Args:
            queries: List of user queries (typically first 2)
            max_tokens: Maximum tokens for title generation
        
        Returns:
            Generated title string
        """
        if not self.generator:
            raise RuntimeError("Generator not initialized")
        
        if not queries:
            return "New Chat"
        
        # Build a simple prompt for title generation
        queries_text = "\n".join(f"- {q}" for q in queries[:3])
        prompt = f"""Based on these user queries to a research assistant, generate a brief, specific title (5-8 words max):

{queries_text}

Title:"""
        
        # Use the generator directly with minimal context
        async with self._gpu_semaphore:
            title_parts = []
            async for token in self.generator.generate_streaming(
                query=prompt,
                contexts=[],  # No context needed for title generation
                temperature=0.3,  # Lower temperature for more focused output
                max_tokens=max_tokens,
            ):
                title_parts.append(token)
                # Stop at newline or if we have enough
                if "\n" in token or len(title_parts) > max_tokens:
                    break
        
        title = "".join(title_parts).strip()
        
        # Clean up: remove quotes, excessive punctuation, newlines
        title = title.strip('"\'').split("\n")[0].strip()
        
        # Fallback if generation failed
        if not title or len(title) < 3:
            title = queries[0][:50]
        
        return title[:100]  # Cap at 100 chars

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _hybrid_fusion(
        self,
        bm25_list: List[str],
        emb_list: List[Tuple],
        k: int,
        return_scores: bool = False,
    ) -> Tuple[Set[str], Dict[str, float]] | Set[str]:
        """Fuse BM25 and embedding results using weighted rank fusion."""
        scores = {}
        bm25_w = self.pipeline_config["bm25_weight"]
        emb_w = self.pipeline_config["embedding_weight"]
        
        for rank, file_path in enumerate(bm25_list):
            score = 1.0 - (rank / len(bm25_list))
            scores[file_path] = scores.get(file_path, 0) + (score * bm25_w)
        
        for rank, (_, meta, _) in enumerate(emb_list):
            file_path = meta.get("file_path")
            if file_path:
                score = 1.0 - (rank / len(emb_list))
                scores[file_path] = scores.get(file_path, 0) + (score * emb_w)
        
        sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = {f for f, _ in sorted_files[:k]}
        
        if return_scores:
            return selected, scores
        return selected

    @property
    def generation_queue_depth(self) -> int:
        """Number of queries waiting for GPU."""
        return self._generation_queue_depth
