"""
RAG Chat Pipeline V2 - Concurrent Execution
============================================

Key improvements over V1:
1. Retrieval stages (1-3) separated from Generation (4)
2. GPU semaphore for serialized generation
3. Reranker auto-offloads to CPU after use
4. Multiple queries can progress through retrieval while waiting for GPU

Architecture:
    Query A: [Retrieval] -> [Rerank] -> [===== GENERATING =====]
    Query B: [Retrieval] -> [Rerank] -> [WAITING] -> [GENERATING]
                                         ^
                                         GPU semaphore
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

def reranker_tracer(a: List[Dict]) -> Dict[str, Dict]:
    """Convert reranker results to tracer format."""
    tracer_dict = {}
    for chunk in a:
        chroma_id = chunk['metadata']['chroma_id']
        tracer_dict[chroma_id] = {}
        for metric in ['rank', 'rerank_score', 'original_distance', 'original_rank', 'rank_improvement']:
            tracer_dict[chroma_id][metric] = chunk.get(metric)
    return tracer_dict


@dataclass
class RetrievalResult:
    """Container for retrieval stage outputs."""
    reranked_results: List[Dict]
    selected_papers: Set[str]
    trace_id: Optional[str]
    req_id: str
    retrieval_duration_ms: float
    
    # For tracing
    bm25_output: Optional[Any] = None
    emb_paper_results: Optional[Any] = None
    chunk_results: Optional[Any] = None
    fusion_scores: Optional[Dict] = None


class RAGPipelineV2:
    """
    Concurrent RAG Pipeline with GPU-serialized generation.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        
        self.paths = self.config['paths']
        self.models = self.config['models']
        self.pipeline_config = self.config['pipeline_config']
        self.setup_config = self.config['setup_config']
        
        # Device configuration
        self.embedding_config = self.setup_config['embedding']
        self.reranker_config = self.setup_config['reranker']
        self.generator_config = self.setup_config['generator']
        self.hallucination_config = self.setup_config['hallucination_eval']
        
        # Components (lazy loaded)
        self.bm25: Any = None
        self.embedding: Any = None
        self.reranker: Any = None
        self.generator: Any = None
        self.hallucination_checker: Any = None
        self.tracer: Any = None
        
        # Tracing
        self.enable_tracing = self.pipeline_config.get('enable_tracing')
        if self.enable_tracing:
            self.trace_db_path = os.path.join(self.paths.get('logs'), 'traces.db')
        
        # GPU SEMAPHORE - Key for concurrent execution
        self._gpu_semaphore = asyncio.Semaphore(self.pipeline_config.get('max_concurrent_generation', 1))
        self._generation_queue_depth = 0
        self._queue_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize all pipeline components with lazy imports."""
        logger.info("Initializing RAG Pipeline V2 (concurrent mode)...")
        
        import torch 
        from inference.bm25_search import BM25Searcher
        from inference.embedding_search import EmbeddingSearch
        from inference.reranker import Reranker  # Use V2 reranker if available
        from inference.generator import AsyncQwenGenerator
        from inference.hallucination_checker import HallucinationChecker
        from utils.request_tracer import RequestTracer
        
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.warning("CUDA not available, forcing all devices to CPU")
            self.embedding_device = "cpu"
            self.reranker_device = "cpu"
            self.generator_device = "cpu"
            self.hallucination_device = "cpu"

        # 1. Tracer
        if self.enable_tracing:
            self.tracer = RequestTracer(db_path=self.trace_db_path)

        # 2. BM25
        self.bm25 = BM25Searcher(artifacts_dir=self.paths['bm25_artifacts'])
        self.bm25.load_bm25_artifacts()
        
        # 3. Embedding Search
        self.embedding = EmbeddingSearch(
            embedding_model_name=self.models['embedding'].get('path') or self.models['embedding']['id'],
            **self.embedding_config
        )
        self.embedding.load(Path(self.paths['embeddings']))
        
        # 4. Reranker (with auto-offload support)
        self.reranker = Reranker(
            model=self.models['reranker'].get('path') or self.models['reranker']['id'],
            **self.reranker_config,
        )
        
        # 5. Generator
        self.generator = AsyncQwenGenerator(
            model_name=self.models['generator'].get('path') or self.models['generator']['id'],
            **self.generator_config,
        )
        await self.generator.initialize()
        
        # Warmup to capture CUDA graphs immediately
        await self.warmup()
        
        # 6. Hallucination Checker
        self.hallucination_checker = HallucinationChecker(
            generator=self.generator,
            hallucination_model=self.models['hallucination_eval'].get('path') or self.models['hallucination_eval']['id'],
            **self.hallucination_config,
        )
        
        logger.info("Pipeline V2 initialized (concurrent mode enabled)")

    async def cleanup(self):
        """Free all resources."""
        import torch
        logger.info("Cleaning up pipeline resources...")
        
        if self.hallucination_checker:
            self.hallucination_checker.cleanup()
            self.hallucination_checker = None
        
        if self.generator:
            await self.generator.cleanup()
            self.generator = None
        
        if self.tracer:
            self.tracer.close()
            self.tracer = None
        
        self.bm25 = None
        self.embedding = None
        self.reranker = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def warmup(self):
        """
        Run a dummy generation to force vLLM to capture CUDA graphs.
        This prevents the 'Capturing CUDA graphs' lag during the first user query.
        """
        logger.info("Starting vLLM warmup sequence...")
        try:
            # We acquire the semaphore to ensure warmup respects the concurrency lock
            async with self._gpu_semaphore:
                # Use a dummy context and query to trigger the generation path
                # Short tokens are enough to trigger graph capture
                dummy_contexts = [{"text": "Warmup context", "metadata": {}, "score": 1.0}]
                async for _ in self.generator.generate_streaming(
                    query="warmup",
                    contexts=dummy_contexts,
                    temperature=0.7,
                    max_tokens=5
                ):
                    pass
            logger.info("vLLM warmup complete. CUDA graphs captured.")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    # =========================================================================
    # STAGE 1-3: RETRIEVAL (Can run concurrently for multiple queries)
    # =========================================================================
    
    async def _run_retrieval_stages(
        self,
        query: str,
        conversation_id: str,
        req_id: str,
    ) -> RetrievalResult:
        """
        Run stages 1-3 (BM25, Embedding, Reranking).
        
        This method does NOT hold the GPU semaphore, allowing multiple
        queries to complete retrieval while waiting for generation.
        """
        import torch
        start_time = time.perf_counter()
        trace_id = None
        
        # Start tracing
        if self.tracer:
            trace_id = self.tracer.start_trace(
                query=query,
                conversation_id=conversation_id,
                req_id=req_id
            )
        
        # ===== STAGE 1: Hybrid Retrieval =====
        logger.info(f"[{req_id[:8]}] Stage 1: Hybrid Retrieval")
        
        # BM25
        bm25_start = time.perf_counter()
        bm25_output = self.bm25.search(
            query, 
            k=self.pipeline_config['k_papers'] * 2
        )
        bm25_results, bm25_scores = map(list, zip(*bm25_output))
        bm25_duration = (time.perf_counter() - bm25_start) * 1000
        
        if self.tracer and trace_id:
            self.tracer.capture_bm25(trace_id, bm25_output, bm25_duration)
        
        # Embedding for Papers
        emb_start = time.perf_counter()
        emb_paper_results = self.embedding.search(
            query, 
            collection_num=1, 
            k=self.pipeline_config['k_papers'] * 2
        )
        emb_duration = (time.perf_counter() - emb_start) * 1000
        
        emb_tracer = [[item[1].get('chroma_id', 'N/A'), round(item[0], 4)] 
                      for item in emb_paper_results]
        
        if self.tracer and trace_id:
            self.tracer.capture_embedding_paper(trace_id, emb_tracer, emb_duration)
        
        # Hybrid Fusion
        selected_papers, fusion_scores = self._hybrid_fusion(
            bm25_results, emb_paper_results, 
            k=self.pipeline_config['k_papers'],
            return_scores=True
        )
        
        if self.tracer and trace_id:
            self.tracer.capture_hybrid_fusion(trace_id, list(selected_papers), fusion_scores)
        
        # ===== STAGE 2: Chunk Retrieval =====
        logger.info(f"[{req_id[:8]}] Stage 2: Chunk Retrieval")
        
        chunk_start = time.perf_counter()
        chunk_results = self.embedding.search(
            query,
            collection_num=2,
            k=self.pipeline_config['m_chunks'],
            file_path_filter=selected_papers
        )
        chunk_duration = (time.perf_counter() - chunk_start) * 1000
        
        chunk_tracer = [[item[1].get('chroma_id', 'N/A'), round(item[0], 4)] 
                        for item in chunk_results]
        
        if self.tracer and trace_id:
            self.tracer.capture_embedding_chunk(trace_id, chunk_tracer, chunk_duration)
        
        # ===== STAGE 3: Reranking =====
        logger.info(f"[{req_id[:8]}] Stage 3: Reranking")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        rerank_start = time.perf_counter()
        reranked_results = self.reranker.rerank_with_details(
            query,
            candidates=chunk_results,
            top_k=self.pipeline_config['n_reranked']
        )
        rerank_duration = (time.perf_counter() - rerank_start) * 1000
        
        # NOTE: Reranker auto-offloads to CPU here (if enabled)
        # This frees GPU memory for the generator
        
        if self.tracer and trace_id:
            self.tracer.capture_reranker(
                trace_id, 
                reranker_tracer(reranked_results), 
                rerank_duration
            )
        
        retrieval_duration = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{req_id[:8]}] Retrieval complete in {retrieval_duration:.0f}ms")
        
        return RetrievalResult(
            reranked_results=reranked_results,
            selected_papers=selected_papers,
            trace_id=trace_id,
            req_id=req_id,
            retrieval_duration_ms=retrieval_duration,
            bm25_output=bm25_output,
            emb_paper_results=emb_paper_results,
            chunk_results=chunk_results,
            fusion_scores=fusion_scores,
        )

    # =========================================================================
    # STAGE 4: GENERATION (GPU-serialized)
    # =========================================================================
    
    async def _run_generation(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        enable_hallucination_check: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run generation stage under GPU semaphore.
        
        This ensures only one query is generating at a time,
        while others can complete retrieval stages.
        """
        trace_id = retrieval_result.trace_id
        req_id = retrieval_result.req_id
        reranked_results = retrieval_result.reranked_results
        
        # Track queue depth
        async with self._queue_lock:
            self._generation_queue_depth += 1
            queue_pos = self._generation_queue_depth
        
        if queue_pos > 1:
            logger.info(f"[{req_id[:8]}] Waiting for GPU (position {queue_pos} in queue)")
            yield {"type": "status", "stage": "waiting_for_gpu", "queue_position": queue_pos}
        
        wait_start = time.perf_counter()
        
        # ===== ACQUIRE GPU SEMAPHORE =====
        async with self._gpu_semaphore:
            wait_time = (time.perf_counter() - wait_start) * 1000
            
            async with self._queue_lock:
                self._generation_queue_depth -= 1
            
            if wait_time > 100:
                logger.info(f"[{req_id[:8]}] GPU acquired after {wait_time:.0f}ms wait")
            
            # ===== STAGE 4: Generation =====
            logger.info(f"[{req_id[:8]}] Stage 4: Generation")
            gen_start = time.perf_counter()
            
            accumulated_text = ""
            token_count = 0
            ttft = None
            
            async for token in self.generator.generate_streaming(
                query=query,
                contexts=reranked_results,
                temperature=self.generator_config['temperature'],
                include_citations=self.pipeline_config['include_citations']
            ):
                if ttft is None:
                    ttft = (time.perf_counter() - gen_start) * 1000
                accumulated_text += token
                token_count += 1
                yield {"type": "token", "content": token}
            
            gen_duration = (time.perf_counter() - gen_start) * 1000
            
            if self.tracer and trace_id:
                self.tracer.capture_generator(
                    trace_id,
                    answer=accumulated_text,
                    duration_ms=gen_duration,
                    completion_tokens=token_count,
                    ttft_ms=ttft
                )
            
            # ===== STAGE 5: Hallucination Check (Optional) =====
            if enable_hallucination_check and self.hallucination_checker:
                logger.info(f"[{req_id[:8]}] Stage 5: Hallucination Check")
                hal_start = time.perf_counter()
                
                hal_contexts = [
                    {"text": r['text'], "metadata": r['metadata']} 
                    for r in reranked_results
                ]
                
                hallucination_result = await self.hallucination_checker.check(
                    answer=accumulated_text,
                    contexts=hal_contexts
                )
                hal_duration = (time.perf_counter() - hal_start) * 1000
                
                if self.tracer and trace_id:
                    self.tracer.capture_hallucination(
                        trace_id,
                        verifications=hallucination_result['verifications'],
                        grounding_ratio=hallucination_result['grounding_ratio'],
                        unsupported_claims=hallucination_result['unsupported_claims'],
                        duration_ms=hal_duration
                    )
                
                yield {
                    "type": "hallucination",
                    "grounding_ratio": hallucination_result['grounding_ratio'],
                    "num_claims": hallucination_result['num_claims'],
                    "num_grounded": hallucination_result['num_grounded'],
                    "unsupported_claims": hallucination_result['unsupported_claims'],
                    "verifications": hallucination_result['verifications']
                }

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================
    
    async def answer_stream(
        self, 
        query: str, 
        conversation_id: str = "default-session",
        enable_hallucination_check: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query through the concurrent RAG pipeline.
        
        Flow:
        1. Retrieval stages run immediately (no GPU lock)
        2. Generation acquires GPU semaphore (queued if busy)
        3. Other queries can complete retrieval while waiting
        """
        start_time = time.perf_counter()
        req_id = set_request_context(conversation_id=conversation_id)
        
        try:
            # ===== STAGES 1-3: Retrieval (concurrent) =====
            retrieval_result = await self._run_retrieval_stages(
                query=query,
                conversation_id=conversation_id,
                req_id=req_id,
            )
            
            yield {
                "type": "status", 
                "stage": "retrieval_complete",
                "papers_found": len(retrieval_result.selected_papers),
                "chunks_reranked": len(retrieval_result.reranked_results),
            }
            
            yield {
                "type": "context", 
                "data": [
                    {
                        "text": r['text'], 
                        "metadata": r['metadata'], 
                        "score": r['rerank_score']
                    } 
                    for r in retrieval_result.reranked_results
                ]
            }
            
            # ===== STAGE 4-5: Generation (GPU-serialized) =====
            async for update in self._run_generation(
                query=query,
                retrieval_result=retrieval_result,
                enable_hallucination_check=enable_hallucination_check,
            ):
                yield update
            
            # Finish tracing
            if self.tracer and retrieval_result.trace_id:
                self.tracer.finish_trace(retrieval_result.trace_id, success=True)
            
            total_duration = (time.perf_counter() - start_time) * 1000
            logger.info(f"[{req_id[:8]}] Request completed in {total_duration:.0f}ms")
            
            yield {
                "type": "done", 
                "trace_id": retrieval_result.trace_id,
                "total_duration_ms": total_duration
            }

        except Exception as e:
            logger.error(f"Pipeline Error: {e}", exc_info=True)
            yield {"type": "error", "message": str(e)}
        
        finally:
            clear_request_context()

    def _hybrid_fusion(
        self, 
        bm25_list: List[str], 
        emb_list: List[Tuple], 
        k: int,
        return_scores: bool = False
    ) -> Tuple[Set[str], Dict[str, float]] | Set[str]:
        """Fuse BM25 and embedding results using weighted rank fusion."""
        scores = {}
        bm25_w = self.pipeline_config['bm25_weight']
        emb_w = self.pipeline_config['embedding_weight']
        
        for rank, file_path in enumerate(bm25_list):
            score = 1.0 - (rank / len(bm25_list))
            scores[file_path] = scores.get(file_path, 0) + (score * bm25_w)

        for rank, (_, meta, _) in enumerate(emb_list):
            file_path = meta.get('file_path')
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


# =============================================================================
# TEST: Concurrent Query Simulation
# =============================================================================

async def test_concurrent_execution():
    """
    Test that demonstrates concurrent query handling.
    
    Without proper concurrency:
        Query A (3s) + Query B (2s) = 5s total
    
    With concurrent retrieval:
        Query A: [1s retrieval] [=== 2s generation ===]
        Query B:     [0.5s retrieval] [wait] [=== 1.5s generation ===]
        Total: ~3.5s (overlapped retrieval)
    """
    print("\n" + "="*70)
    print("CONCURRENT EXECUTION TEST")
    print("="*70)
    
    # Mock pipeline that simulates timing
    class MockPipeline:
        def __init__(self):
            self._gpu_semaphore = asyncio.Semaphore(1)
            self._queue_depth = 0
        
        async def process_query(self, name: str, retrieval_time: float, gen_time: float):
            # Retrieval (no lock)
            print(f"[{name}] Starting retrieval...")
            await asyncio.sleep(retrieval_time)
            print(f"[{name}] Retrieval complete, waiting for GPU...")
            
            # Generation (locked)
            wait_start = time.perf_counter()
            async with self._gpu_semaphore:
                wait_time = (time.perf_counter() - wait_start) * 1000
                if wait_time > 10:
                    print(f"[{name}] GPU acquired after {wait_time:.0f}ms wait")
                else:
                    print(f"[{name}] GPU acquired immediately")
                
                print(f"[{name}] === GENERATING ===")
                await asyncio.sleep(gen_time)
                print(f"[{name}] === DONE ===")
    
    pipeline = MockPipeline()
    
    start = time.perf_counter()
    
    # Launch queries with slight offset
    async def run():
        task_a = asyncio.create_task(pipeline.process_query("A", 1.0, 2.0))
        await asyncio.sleep(0.3)  # Query B arrives 0.3s later
        task_b = asyncio.create_task(pipeline.process_query("B", 0.5, 1.5))
        await asyncio.gather(task_a, task_b)
    
    await run()
    
    total = time.perf_counter() - start
    sequential = 1.0 + 2.0 + 0.5 + 1.5  # 5s
    
    print(f"\n{'='*70}")
    print(f"Total time: {total:.2f}s")
    print(f"Sequential would be: {sequential:.1f}s")
    print(f"Time saved: {sequential - total:.2f}s ({(1 - total/sequential)*100:.0f}% faster)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import os
    import multiprocessing as mp
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Run concurrent test
    asyncio.run(test_concurrent_execution())