"""
Integration Test: Concurrent Pipeline Execution
================================================

Tests real pipeline components with actual queries to validate:
1. Two requests can run retrieval concurrently
2. Generation is serialized (one at a time)
3. Request 3 blocks until one of the first two completes

Query: "What are the tradeoffs and assumptions of intensity normalization 
        using an internal standard reference of MALDI MSI datasets with 
        multiple samples?"

Run with: python test_concurrent_pipeline.py
"""

import asyncio
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Optional
from dataclasses import dataclass, field

# Set up environment before torch imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# =============================================================================
# BACKEND WITH DUAL SEMAPHORE PATTERN
# =============================================================================

@dataclass
class RequestMetrics:
    """Track timing for a single request."""
    request_id: str
    query: str
    
    # Timestamps
    entered_pipeline: float = 0.0
    started_retrieval: float = 0.0
    finished_retrieval: float = 0.0
    started_waiting_gpu: float = 0.0
    acquired_gpu: float = 0.0
    finished_generation: float = 0.0
    exited_pipeline: float = 0.0
    
    # Computed durations (filled after completion)
    wait_for_pipeline_ms: float = 0.0
    retrieval_ms: float = 0.0
    wait_for_gpu_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    
    def compute_durations(self):
        """Calculate all durations from timestamps."""
        self.wait_for_pipeline_ms = (self.started_retrieval - self.entered_pipeline) * 1000
        self.retrieval_ms = (self.finished_retrieval - self.started_retrieval) * 1000
        self.wait_for_gpu_ms = (self.acquired_gpu - self.started_waiting_gpu) * 1000
        self.generation_ms = (self.finished_generation - self.acquired_gpu) * 1000
        self.total_ms = (self.exited_pipeline - self.entered_pipeline) * 1000


class RAGBackend:
    """
    Backend layer with dual semaphore pattern.
    
    - pipeline_semaphore: Max 2 requests in pipeline at any time
    - gpu_semaphore: Max 1 request generating at any time
    """
    
    def __init__(
        self,
        pipeline,  # RAGPipeline instance
        max_concurrent_pipeline: int = 2,
        max_concurrent_gpu: int = 1,
    ):
        self.pipeline = pipeline
        self._pipeline_semaphore = asyncio.Semaphore(max_concurrent_pipeline)
        self._gpu_semaphore = asyncio.Semaphore(max_concurrent_gpu)
        
        # Tracking
        self._active_requests: Dict[str, RequestMetrics] = {}
        self._completed_requests: list[RequestMetrics] = []
        self._request_counter = 0
        self._lock = asyncio.Lock()
    
    async def query(
        self,
        query: str,
        conversation_id: str = "default",
        enable_hallucination_check: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query with concurrency control.
        """
        # Create metrics tracker
        async with self._lock:
            self._request_counter += 1
            request_id = f"req-{self._request_counter:03d}"
        
        metrics = RequestMetrics(request_id=request_id, query=query[:50])
        metrics.entered_pipeline = time.perf_counter()
        
        async with self._lock:
            self._active_requests[request_id] = metrics
        
        try:
            # ===== PIPELINE GATE (max 2) =====
            pipeline_wait_start = time.perf_counter()
            
            if self._pipeline_semaphore.locked():
                print(f"  [{request_id}] ⏳ Waiting for pipeline slot...")
                yield {"type": "status", "stage": "waiting_pipeline", "request_id": request_id}
            
            async with self._pipeline_semaphore:
                metrics.started_retrieval = time.perf_counter()
                
                if metrics.started_retrieval - pipeline_wait_start > 0.01:
                    print(f"  [{request_id}] 🎫 Got pipeline slot after {(metrics.started_retrieval - pipeline_wait_start)*1000:.0f}ms")
                
                # ===== STAGE 1-3: RETRIEVAL (CPU) =====
                print(f"  [{request_id}] 🔍 Starting retrieval...")
                yield {"type": "status", "stage": "retrieval", "request_id": request_id}
                
                retrieval_result = await self.pipeline.retrieve(query)
                metrics.finished_retrieval = time.perf_counter()
                
                print(f"  [{request_id}] ✅ Retrieval complete ({(metrics.finished_retrieval - metrics.started_retrieval)*1000:.0f}ms)")
                
                yield {
                    "type": "context",
                    "request_id": request_id,
                    "num_chunks": len(retrieval_result.reranked_results),
                }
                
                # ===== GPU GATE (max 1) =====
                metrics.started_waiting_gpu = time.perf_counter()
                
                if self._gpu_semaphore.locked():
                    print(f"  [{request_id}] ⏳ Waiting for GPU...")
                    yield {"type": "status", "stage": "waiting_gpu", "request_id": request_id}
                
                async with self._gpu_semaphore:
                    metrics.acquired_gpu = time.perf_counter()
                    
                    gpu_wait = (metrics.acquired_gpu - metrics.started_waiting_gpu) * 1000
                    if gpu_wait > 10:
                        print(f"  [{request_id}] 🎮 Got GPU after {gpu_wait:.0f}ms wait")
                    else:
                        print(f"  [{request_id}] 🎮 Got GPU immediately")
                    
                    # ===== STAGE 4: GENERATION =====
                    print(f"  [{request_id}] 💬 Generating...")
                    yield {"type": "status", "stage": "generating", "request_id": request_id}
                    
                    token_count = 0
                    async for update in self.pipeline.generate(
                        query=query,
                        retrieval_result=retrieval_result,
                        enable_hallucination_check=enable_hallucination_check,
                    ):
                        if update.get("type") == "token":
                            token_count += 1
                        yield update
                    
                    metrics.finished_generation = time.perf_counter()
                    print(f"  [{request_id}] ✅ Generation complete ({token_count} tokens, {(metrics.finished_generation - metrics.acquired_gpu)*1000:.0f}ms)")
            
            metrics.exited_pipeline = time.perf_counter()
            metrics.compute_durations()
            
            yield {
                "type": "done",
                "request_id": request_id,
                "metrics": {
                    "wait_pipeline_ms": metrics.wait_for_pipeline_ms,
                    "retrieval_ms": metrics.retrieval_ms,
                    "wait_gpu_ms": metrics.wait_for_gpu_ms,
                    "generation_ms": metrics.generation_ms,
                    "total_ms": metrics.total_ms,
                }
            }
        
        except Exception as e:
            print(f"  [{request_id}] ❌ Error: {e}")
            yield {"type": "error", "request_id": request_id, "message": str(e)}
        
        finally:
            async with self._lock:
                if request_id in self._active_requests:
                    del self._active_requests[request_id]
                self._completed_requests.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all completed requests."""
        if not self._completed_requests:
            return {"message": "No completed requests"}
        
        return {
            "total_requests": len(self._completed_requests),
            "requests": [
                {
                    "id": m.request_id,
                    "wait_pipeline_ms": round(m.wait_for_pipeline_ms, 1),
                    "retrieval_ms": round(m.retrieval_ms, 1),
                    "wait_gpu_ms": round(m.wait_for_gpu_ms, 1),
                    "generation_ms": round(m.generation_ms, 1),
                    "total_ms": round(m.total_ms, 1),
                }
                for m in self._completed_requests
            ]
        }


# =============================================================================
# MODIFIED PIPELINE WITH SEPARATE RETRIEVE/GENERATE
# =============================================================================

@dataclass
class RetrievalResult:
    """Container for retrieval outputs."""
    reranked_results: list
    selected_papers: set
    trace_id: Optional[str] = None
    req_id: Optional[str] = None


class RAGPipelineWithSplit:
    """
    RAG Pipeline with separate retrieve() and generate() methods
    for backend-level concurrency control.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        embedding_device: str = "cpu",
        reranker_device: str = "cuda",
        generator_device: str = "cuda",
        hallucination_device: str = "cpu",
    ):
        from utils.config_loader import load_config
        
        self.config = load_config(config_path)
        self.paths = self.config['paths']
        self.models = self.config['models']
        self.retrieval_conf = self.config['retrieval']
        self.gen_conf = self.config['generation']
        
        self.embedding_device = embedding_device
        self.reranker_device = reranker_device
        self.generator_device = generator_device
        self.hallucination_device = hallucination_device
        
        # Components (lazy loaded)
        self.bm25 = None
        self.embedding = None
        self.reranker = None
        self.generator = None
        self.hallucination_checker = None
    
    async def initialize(self):
        """Initialize all components."""
        import torch
        from inference.bm25_search import BM25Searcher
        from inference.embedding_search import EmbeddingSearch
        from inference.reranker import Reranker
        from inference.generator import AsyncQwenGenerator
        from inference.hallucination_checker import HallucinationChecker
        
        print("Initializing pipeline components...")
        
        # BM25
        self.bm25 = BM25Searcher(artifacts_dir=self.paths['bm25_artifacts'])
        self.bm25.load_bm25_artifacts()
        print("  ✓ BM25 loaded")
        
        # Embedding
        self.embedding = EmbeddingSearch(
            embedding_model_name=self.models['embedding'],
            device=self.embedding_device,
            truncate_dim=self.retrieval_conf.get('truncate_dim')
        )
        self.embedding.load(Path(self.paths['embeddings']))
        print(f"  ✓ Embedding loaded ({self.embedding_device})")
        
        # Reranker
        self.reranker = Reranker(
            model_name=self.models['reranker'],
            device=self.reranker_device,
            batch_size=4,
            timeout_seconds=60,
        )
        print(f"  ✓ Reranker loaded ({self.reranker_device})")
        
        # Generator
        self.generator = AsyncQwenGenerator(
            model_name=self.models['generator'],
            gpu_memory_utilization=0.4,
            tensor_parallel_size=1,
        )
        await self.generator.initialize()
        print(f"  ✓ Generator loaded ({self.generator_device})")
        
        # Hallucination checker
        self.hallucination_checker = HallucinationChecker(
            generator=self.generator,
            device=self.hallucination_device,
        )
        print(f"  ✓ Hallucination checker ready ({self.hallucination_device})")
        
        print("Pipeline initialized!")
    
    async def cleanup(self):
        """Clean up resources."""
        import torch
        
        if self.hallucination_checker:
            self.hallucination_checker.cleanup()
        if self.generator:
            await self.generator.cleanup()
        
        self.bm25 = None
        self.embedding = None
        self.reranker = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def retrieve(self, query: str) -> RetrievalResult:
        """
        Run stages 1-3: BM25 + Embedding + Reranking.
        CPU-bound, can run concurrently.
        """
        import torch
        
        # Stage 1: BM25
        bm25_output = self.bm25.search(query, k=self.retrieval_conf['k_papers'] * 2)
        bm25_results, _ = map(list, zip(*bm25_output))
        
        # Stage 1: Embedding (papers)
        emb_paper_results = self.embedding.search(
            query,
            collection_num=1,
            k=self.retrieval_conf['k_papers'] * 2,
        )
        
        # Hybrid fusion
        selected_papers = self._hybrid_fusion(
            bm25_results,
            emb_paper_results,
            k=self.retrieval_conf['k_papers'],
        )
        
        # Stage 2: Chunk retrieval
        chunk_results = self.embedding.search(
            query,
            collection_num=2,
            k=self.retrieval_conf['m_chunks'],
            file_path_filter=selected_papers,
        )
        
        # Stage 3: Reranking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        reranked_results = self.reranker.rerank_with_details(
            query,
            candidates=chunk_results,
            top_k=self.retrieval_conf['n_reranked'],
        )
        
        return RetrievalResult(
            reranked_results=reranked_results,
            selected_papers=selected_papers,
        )
    
    async def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        enable_hallucination_check: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run stage 4: Generation (and optionally stage 5: hallucination check).
        GPU-bound, should be serialized.
        """
        accumulated_text = ""
        
        async for token in self.generator.generate_streaming(
            query=query,
            contexts=retrieval_result.reranked_results,
            temperature=self.gen_conf['temperature'],
            include_citations=self.gen_conf['include_citations'],
        ):
            accumulated_text += token
            yield {"type": "token", "content": token}
        
        # Hallucination check (optional)
        if enable_hallucination_check and self.hallucination_checker:
            hal_contexts = [
                {"text": r['text'], "metadata": r['metadata']}
                for r in retrieval_result.reranked_results
            ]
            
            hal_result = await self.hallucination_checker.check(
                answer=accumulated_text,
                contexts=hal_contexts,
            )
            
            yield {
                "type": "hallucination",
                "grounding_ratio": hal_result['grounding_ratio'],
                "num_claims": hal_result['num_claims'],
                "unsupported_claims": hal_result['unsupported_claims'],
            }
    
    def _hybrid_fusion(self, bm25_list, emb_list, k: int) -> set:
        """Weighted rank fusion of BM25 and embedding results."""
        scores = {}
        bm25_w = self.retrieval_conf['bm25_weight']
        emb_w = self.retrieval_conf['embedding_weight']
        
        for rank, file_path in enumerate(bm25_list):
            score = 1.0 - (rank / len(bm25_list))
            scores[file_path] = scores.get(file_path, 0) + (score * bm25_w)
        
        for rank, (_, meta, _) in enumerate(emb_list):
            file_path = meta.get('file_path')
            if file_path:
                score = 1.0 - (rank / len(emb_list))
                scores[file_path] = scores.get(file_path, 0) + (score * emb_w)
        
        sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {f for f, _ in sorted_files[:k]}


# =============================================================================
# TEST RUNNER
# =============================================================================

async def run_concurrent_test():
    """
    Test concurrent execution with real components.
    
    Scenario:
    - 3 requests arrive with slight delays
    - Max 2 in pipeline, max 1 generating
    - Request 3 should wait for pipeline slot
    """
    
    TEST_QUERY = (
        "What are the tradeoffs and assumptions of intensity normalization "
        "using an internal standard reference of MALDI MSI datasets with "
        "multiple samples?"
    )
    
    print("\n" + "=" * 70)
    print("CONCURRENT PIPELINE TEST")
    print("=" * 70)
    print(f"\nQuery: {TEST_QUERY[:70]}...")
    print(f"\nConfig: max_pipeline=2, max_gpu=1")
    print("=" * 70)
    
    # Initialize
    pipeline = RAGPipelineWithSplit(
        embedding_device="cpu",
        reranker_device="cuda",
        generator_device="cuda",
        hallucination_device="cpu",
    )
    
    await pipeline.initialize()
    
    backend = RAGBackend(
        pipeline=pipeline,
        max_concurrent_pipeline=2,
        max_concurrent_gpu=1,
    )
    
    # Track results
    results = {}
    
    async def run_request(request_num: int, delay: float = 0.0):
        """Run a single request and collect results."""
        if delay > 0:
            await asyncio.sleep(delay)
        
        print(f"\n[Request {request_num}] 🚀 Starting...")
        
        tokens = []
        final_metrics = None
        
        async for update in backend.query(
            query=TEST_QUERY,
            conversation_id=f"test-{request_num}",
            enable_hallucination_check=False,
        ):
            if update.get("type") == "token":
                tokens.append(update["content"])
            elif update.get("type") == "done":
                final_metrics = update.get("metrics", {})
        
        results[request_num] = {
            "tokens": len(tokens),
            "response_preview": "".join(tokens)[:200],
            "metrics": final_metrics,
        }
        
        print(f"\n[Request {request_num}] 🏁 Finished! ({len(tokens)} tokens)")
    
    # Launch 3 concurrent requests
    print("\n" + "-" * 70)
    print("Launching 3 concurrent requests...")
    print("  - Request 1: Immediate")
    print("  - Request 2: After 100ms")
    print("  - Request 3: After 200ms (should wait for pipeline slot)")
    print("-" * 70)
    
    start_time = time.perf_counter()
    
    await asyncio.gather(
        run_request(1, delay=0.0),
        run_request(2, delay=0.1),
        run_request(3, delay=0.2),
    )
    
    total_time = time.perf_counter() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summary = backend.get_summary()
    
    for req in summary.get("requests", []):
        print(f"\n{req['id']}:")
        print(f"  Wait for pipeline: {req['wait_pipeline_ms']:>8.1f} ms")
        print(f"  Retrieval:         {req['retrieval_ms']:>8.1f} ms")
        print(f"  Wait for GPU:      {req['wait_gpu_ms']:>8.1f} ms")
        print(f"  Generation:        {req['generation_ms']:>8.1f} ms")
        print(f"  ─────────────────────────────")
        print(f"  Total:             {req['total_ms']:>8.1f} ms")
    
    print("\n" + "-" * 70)
    print(f"Wall clock time: {total_time:.2f}s")
    
    # Calculate theoretical times
    avg_retrieval = sum(r['retrieval_ms'] for r in summary['requests']) / 3
    avg_generation = sum(r['generation_ms'] for r in summary['requests']) / 3
    
    sequential_time = 3 * (avg_retrieval + avg_generation) / 1000
    print(f"Sequential would be: ~{sequential_time:.1f}s")
    print(f"Speedup: {sequential_time / total_time:.2f}x")
    
    # Verify concurrency pattern
    print("\n" + "-" * 70)
    print("CONCURRENCY VERIFICATION")
    print("-" * 70)
    
    requests = summary['requests']
    
    # Check: Request 3 should have waited for pipeline
    req3 = next((r for r in requests if r['id'] == 'req-003'), None)
    if req3 and req3['wait_pipeline_ms'] > 100:
        print("✅ Request 3 waited for pipeline slot (as expected)")
    else:
        print("⚠️  Request 3 did not wait for pipeline (unexpected)")
    
    # Check: At least one request waited for GPU
    gpu_waiters = [r for r in requests if r['wait_gpu_ms'] > 100]
    if gpu_waiters:
        print(f"✅ {len(gpu_waiters)} request(s) waited for GPU (serialization working)")
    else:
        print("⚠️  No requests waited for GPU (check semaphore)")
    
    # Check: Retrieval overlap occurred
    # If retrieval ran concurrently, total time < sum of all stages
    sum_all = sum(r['total_ms'] for r in requests)
    if total_time * 1000 < sum_all * 0.8:
        print(f"✅ Concurrent execution achieved ({sum_all/1000:.1f}s work in {total_time:.1f}s)")
    else:
        print("⚠️  Limited concurrency benefit")
    
    print("=" * 70)
    
    # Cleanup
    await pipeline.cleanup()
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        results = asyncio.run(run_concurrent_test())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)