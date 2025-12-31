"""
RAG Chat Pipeline - Updated
===========================

4-Stage RAG Pipeline with:
- Multi-device support (separate devices for embedding, reranker, generator)
- Optional hallucination detection (toggle-able)
- Request tracing with SQLite storage
- Memory-safe lazy imports

Chainlit Integration:
    The `enable_hallucination_check` parameter can be controlled via a Chainlit
    toggle switch. See the example Chainlit integration at the bottom.
"""

import asyncio
import time
import os
import gc
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, AsyncGenerator, Tuple

from utils.logger import (
    get_logger, 
    set_request_context, 
    clear_request_context,
    get_request_context
)
from utils.config_loader import load_config

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


class RAGPipeline:
    """
    Orchestrates the 4-Stage RAG Pipeline with:
    - Multi-device configuration
    - Optional hallucination checking
    - Request tracing
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        # Device configuration
        embedding_device: str = "cpu",
        reranker_device: str = "cuda",
        generator_device: str = "cuda",
        hallucination_device: str = "cpu",
        # Feature flags
        enable_tracing: bool = True,
        trace_db_path: str = "traces/request_traces.db"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            config_path: Path to configuration YAML file.
            embedding_device: Device for embedding model ("cpu" or "cuda").
            reranker_device: Device for reranker model ("cpu" or "cuda").
            generator_device: Device for generator model (typically "cuda").
            hallucination_device: Device for HHEM model ("cpu" recommended for 0.1B).
            enable_tracing: Whether to enable request tracing.
            trace_db_path: Path to SQLite database for traces.
        """
        self.config = load_config(config_path)
        
        self.paths = self.config['paths']
        self.models = self.config['models']
        self.retrieval_conf = self.config['retrieval']
        self.gen_conf = self.config['generation']
        
        # Device configuration
        self.embedding_device = embedding_device
        self.reranker_device = reranker_device
        self.generator_device = generator_device
        self.hallucination_device = hallucination_device
        
        # Component instances (lazy loaded)
        self.bm25: Any = None
        self.embedding: Any = None
        self.reranker: Any = None
        self.generator: Any = None
        self.hallucination_checker: Any = None
        
        # Tracing
        self.enable_tracing = enable_tracing
        self.trace_db_path = trace_db_path
        self.tracer: Any = None

    async def initialize(self):
        """
        Initialize all pipeline components.
        
        CRITICAL: All heavy imports happen here (lazy loading) to prevent
        CUDA initialization before multiprocessing spawn is configured.
        """
        logger.info("Initializing RAG Pipeline components...")
        logger.info(f"Device config: embedding={self.embedding_device}, "
                   f"reranker={self.reranker_device}, generator={self.generator_device}, "
                   f"hallucination={self.hallucination_device}")

        # --- LAZY IMPORTS ---
        import torch 
        from inference.bm25_search import BM25Searcher
        from inference.embedding_search import EmbeddingSearch
        from inference.reranker import Reranker
        from inference.generator import AsyncQwenGenerator
        from hallucination_checker import HallucinationChecker
        from request_tracer import RequestTracer
        
        # Validate CUDA availability
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.warning("CUDA not available, forcing all devices to CPU")
            self.embedding_device = "cpu"
            self.reranker_device = "cpu"
            self.generator_device = "cpu"
            self.hallucination_device = "cpu"

        # 1. Initialize Tracer (if enabled)
        if self.enable_tracing:
            self.tracer = RequestTracer(db_path=self.trace_db_path)
            logger.info("Request tracer initialized")

        # 2. Load BM25 (CPU only - no device option)
        self.bm25 = BM25Searcher(artifacts_dir=self.paths['bm25_artifacts'])
        self.bm25.load_bm25_artifacts()
        logger.info("BM25 searcher loaded")
        
        # 3. Load Embedding Search
        self.embedding = EmbeddingSearch(
            embedding_model_name=self.models['embedding'],
            device=self.embedding_device,
            truncate_dim=self.retrieval_conf.get('truncate_dim')
        )
        self.embedding.load(Path(self.paths['embeddings']))
        logger.info(f"Embedding search loaded on {self.embedding_device}")
        
        # 4. Load Reranker
        self.reranker = Reranker(
            model_name=self.models['reranker'],
            device=self.reranker_device,
            batch_size=4,
            timeout_seconds=60
        )
        logger.info(f"Reranker loaded on {self.reranker_device}")
        
        # 5. Load Generator
        self.generator = AsyncQwenGenerator(
            model_name=self.models['generator'],
            gpu_memory_utilization=0.4,
            tensor_parallel_size=1
        )
        await self.generator.initialize()
        logger.info("Generator initialized")
        
        # 6. Initialize Hallucination Checker (lazy - model loaded on first use)
        self.hallucination_checker = HallucinationChecker(
            generator=self.generator,
            device=self.hallucination_device,
            threshold=self.config.get('hallucination', {}).get('threshold', 0.5)
        )
        logger.info(f"Hallucination checker ready (device={self.hallucination_device})")
        
        logger.info("Pipeline initialized successfully")

    async def cleanup(self):
        """Free all resources."""
        import torch
        import torch.distributed as dist

        logger.info("Cleaning up pipeline resources...")
        
        # Cleanup components in reverse order
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
            torch.cuda.synchronize()

        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error destroying process group: {e}")

        logger.info("Pipeline resources cleaned up")

    async def answer_stream(
        self, 
        query: str, 
        conversation_id: str = "default-session",
        enable_hallucination_check: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query through the RAG pipeline with streaming output.
        
        Args:
            query: The user's question.
            conversation_id: Chat session identifier.
            enable_hallucination_check: Whether to run hallucination detection.
                Can be toggled per-request (e.g., via Chainlit switch).
        
        Yields:
            Dicts with types: "status", "context", "token", "hallucination", "done", "error"
        """
        import torch
        
        start_time = time.perf_counter()
        trace_id = None
        
        # Set up request context
        req_id = set_request_context(conversation_id=conversation_id)
        
        try:
            # Start tracing
            if self.tracer:
                trace_id = self.tracer.start_trace(
                    query=query,
                    conversation_id=conversation_id,
                    req_id=req_id
                )
            
            # =========================================================
            # STAGE 1: Hybrid Retrieval (BM25 + Embedding for Papers)
            # =========================================================
            logger.info("Stage 1: Hybrid Retrieval")
            stage_start = time.perf_counter()
            
            # BM25 Search
            bm25_start = time.perf_counter()
            bm25_search_output = self.bm25.search(
                query, 
                k=self.retrieval_conf['k_papers'] * 2
            )
            bm25_results, bm25_scores = map(list, zip(*bm25_search_output))
            bm25_duration = (time.perf_counter() - bm25_start) * 1000
            
            # Capture BM25 trace
            if self.tracer and trace_id:
                self.tracer.capture_bm25(trace_id, bm25_search_output, bm25_duration)
            
            # Embedding Search for Papers
            emb_start = time.perf_counter()
            emb_paper_results = self.embedding.search(
                query, 
                collection_num=1, 
                k=self.retrieval_conf['k_papers'] * 2
            )
            emb_duration = (time.perf_counter() - emb_start) * 1000
            
            # Tracer format for embedding results
            emb_paper_results_tracer = [
                [item[1].get('chroma_id', 'N/A'), round(item[0], 4)] 
                for item in emb_paper_results
            ]
            
            if self.tracer and trace_id:
                self.tracer.capture_embedding_paper(
                    trace_id, 
                    emb_paper_results_tracer, 
                    emb_duration
                )
            
            # Hybrid Fusion
            selected_papers, fusion_scores = self._hybrid_fusion(
                bm25_results, 
                emb_paper_results, 
                k=self.retrieval_conf['k_papers'],
                return_scores=True
            )
            
            if self.tracer and trace_id:
                self.tracer.capture_hybrid_fusion(
                    trace_id, 
                    list(selected_papers), 
                    fusion_scores
                )
            
            yield {
                "type": "status", 
                "stage": "stage_1_complete", 
                "papers_found": len(selected_papers)
            }

            # =========================================================
            # STAGE 2: Chunk Retrieval
            # =========================================================
            logger.info("Stage 2: Chunk Retrieval")
            chunk_start = time.perf_counter()
            
            chunk_results = self.embedding.search(
                query,
                collection_num=2,
                k=self.retrieval_conf['m_chunks'],
                file_path_filter=selected_papers
            )
            chunk_duration = (time.perf_counter() - chunk_start) * 1000
            
            emb_chunk_results_tracer = [
                [item[1].get('chroma_id', 'N/A'), round(item[0], 4)] 
                for item in chunk_results
            ]
            
            if self.tracer and trace_id:
                self.tracer.capture_embedding_chunk(
                    trace_id, 
                    emb_chunk_results_tracer, 
                    chunk_duration
                )
            
            yield {
                "type": "status", 
                "stage": "stage_2_complete", 
                "chunks_found": len(chunk_results)
            }

            # =========================================================
            # STAGE 3: Reranking
            # =========================================================
            logger.info("Stage 3: Reranking")
            rerank_start = time.perf_counter()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reranked_results = self.reranker.rerank_with_details(
                query,
                candidates=chunk_results,
                top_k=self.retrieval_conf['n_reranked']
            )
            rerank_duration = (time.perf_counter() - rerank_start) * 1000
            
            # Tracer format for reranker
            reranker_trace_data = reranker_tracer(reranked_results)
            
            if self.tracer and trace_id:
                self.tracer.capture_reranker(
                    trace_id, 
                    reranker_trace_data, 
                    rerank_duration
                )
            
            # Yield context for display
            yield {
                "type": "context", 
                "data": [
                    {
                        "text": r['text'], 
                        "metadata": r['metadata'], 
                        "score": r['rerank_score']
                    } 
                    for r in reranked_results
                ]
            }

            # =========================================================
            # STAGE 4: Generation
            # =========================================================
            logger.info("Stage 4: Generation")
            gen_start = time.perf_counter()
            
            accumulated_text = ""
            token_count = 0
            ttft = None
            
            async for token in self.generator.generate_streaming(
                query=query,
                contexts=reranked_results,
                temperature=self.gen_conf['temperature'],
                include_citations=self.gen_conf['include_citations']
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

            # =========================================================
            # STAGE 5: Hallucination Check (Optional)
            # =========================================================
            hallucination_result = None
            
            if enable_hallucination_check and self.hallucination_checker:
                logger.info("Stage 5: Hallucination Check")
                hal_start = time.perf_counter()
                
                # Prepare contexts for hallucination checker
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
                
                # Yield hallucination results
                yield {
                    "type": "hallucination",
                    "grounding_ratio": hallucination_result['grounding_ratio'],
                    "num_claims": hallucination_result['num_claims'],
                    "num_grounded": hallucination_result['num_grounded'],
                    "unsupported_claims": hallucination_result['unsupported_claims'],
                    "verifications": hallucination_result['verifications']
                }
                
                logger.info(
                    f"Hallucination check: {hallucination_result['grounding_ratio']:.0%} grounded, "
                    f"{len(hallucination_result['unsupported_claims'])} unsupported claims"
                )

            # Finish tracing
            if self.tracer and trace_id:
                self.tracer.finish_trace(trace_id, success=True)
            
            total_duration = (time.perf_counter() - start_time) * 1000
            logger.info(f"Request completed in {total_duration:.0f}ms")
            
            yield {
                "type": "done", 
                "trace_id": trace_id,
                "total_duration_ms": total_duration
            }

        except Exception as e:
            logger.error(f"Pipeline Error: {e}", exc_info=True)
            
            if self.tracer and trace_id:
                self.tracer.finish_trace(trace_id, success=False, error_message=str(e))
            
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
        """
        Fuse BM25 and embedding results using weighted rank fusion.
        
        Args:
            bm25_list: List of file paths from BM25.
            emb_list: List of (distance, metadata, text) from embedding search.
            k: Number of papers to select.
            return_scores: Whether to return fusion scores for tracing.
        
        Returns:
            Set of selected file paths, optionally with fusion scores.
        """
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
        selected = {f for f, _ in sorted_files[:k]}
        
        if return_scores:
            return selected, scores
        return selected


# ==============================================================================
# CHAINLIT INTEGRATION EXAMPLE
# ==============================================================================
"""
Example Chainlit app with hallucination toggle:

```python
import chainlit as cl
from chat_pipeline import RAGPipeline

pipeline = None

@cl.on_chat_start
async def start():
    global pipeline
    
    # Initialize pipeline once
    pipeline = RAGPipeline(
        embedding_device="cpu",
        reranker_device="cuda",
        generator_device="cuda",
        hallucination_device="cpu"
    )
    await pipeline.initialize()
    
    # Create hallucination toggle switch
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Switch(
                id="hallucination_check",
                label="Enable Hallucination Detection",
                initial=False,
                description="Verify claims against source documents"
            )
        ]
    ).send()
    
    cl.user_session.set("hallucination_check", False)

@cl.on_settings_update
async def settings_update(settings):
    cl.user_session.set("hallucination_check", settings.get("hallucination_check", False))

@cl.on_message
async def main(message: cl.Message):
    # Get toggle state
    enable_hallucination = cl.user_session.get("hallucination_check", False)
    conversation_id = cl.user_session.get("conversation_id", "default")
    
    # Create streaming message
    msg = cl.Message(content="")
    await msg.send()
    
    # Process through pipeline
    async for update in pipeline.answer_stream(
        query=message.content,
        conversation_id=conversation_id,
        enable_hallucination_check=enable_hallucination
    ):
        if update["type"] == "token":
            await msg.stream_token(update["content"])
        
        elif update["type"] == "hallucination":
            # Add hallucination info as a step
            async with cl.Step(name="Hallucination Check") as step:
                ratio = update["grounding_ratio"]
                unsupported = update["unsupported_claims"]
                
                step.output = f"**Grounding: {ratio:.0%}** ({update['num_grounded']}/{update['num_claims']} claims verified)\\n\\n"
                
                if unsupported:
                    step.output += "**Unverified Claims:**\\n"
                    for claim in unsupported:
                        step.output += f"- {claim}\\n"
        
        elif update["type"] == "context":
            # Optionally show sources
            sources = update["data"][:3]
            elements = [
                cl.Text(
                    name=f"Source {i+1}",
                    content=src["text"][:500],
                    display="side"
                )
                for i, src in enumerate(sources)
            ]
            msg.elements = elements
        
        elif update["type"] == "error":
            msg.content = f"Error: {update['message']}"
    
    await msg.update()
```
"""


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. SETUP MULTIPROCESSING FIRST (Before ANY logic runs)
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

    # 2. Set environment variables
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    # 3. Run Pipeline
    async def main():
        # Configure devices: embedding and hallucination on CPU, rest on GPU
        pipeline = RAGPipeline(
            embedding_device="cpu",
            reranker_device="cuda",
            generator_device="cuda",
            hallucination_device="cpu",
            enable_tracing=True,
            trace_db_path="traces/request_traces.db"
        )
        
        try:
            await pipeline.initialize()
            
            query = "What is the impact of matrix effects?"
            print(f"Query: {query}\n")
            print("-" * 50)
            
            # Test with hallucination check enabled
            async for update in pipeline.answer_stream(
                query=query,
                conversation_id="test-session",
                enable_hallucination_check=True  # Toggle this!
            ):
                if update['type'] == 'token':
                    print(update['content'], end="", flush=True)
                
                elif update['type'] == 'status':
                    print(f"\n[{update['stage']}]", flush=True)
                
                elif update['type'] == 'hallucination':
                    print(f"\n\n{'='*50}")
                    print("HALLUCINATION CHECK RESULTS")
                    print(f"{'='*50}")
                    print(f"Grounding Ratio: {update['grounding_ratio']:.0%}")
                    print(f"Claims: {update['num_grounded']}/{update['num_claims']} verified")
                    
                    if update['unsupported_claims']:
                        print(f"\nUnsupported Claims:")
                        for claim in update['unsupported_claims']:
                            print(f"  ⚠️  {claim}")
                    print(f"{'='*50}\n")
                
                elif update['type'] == 'error':
                    print(f"\nERROR: {update['message']}")
                
                elif update['type'] == 'done':
                    print(f"\n\n[Completed in {update['total_duration_ms']:.0f}ms]")
                    print(f"[Trace ID: {update['trace_id']}]")
                    
        except Exception as e:
            print(f"\nCRITICAL FAILURE: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await pipeline.cleanup()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")