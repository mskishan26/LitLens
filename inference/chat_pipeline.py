import asyncio
import time
import torch
import os
import gc
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, AsyncGenerator, Tuple

# === CRITICAL FIX: MEMORY FRAGMENTATION ===
# Set this before importing torch to allow PyTorch to use fragmented memory segments
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# setting up multiple models, to prevent startup errors
import torch.distributed as dist
import torch.multiprocessing as mp
# Import your components
from inference.bm25_search import BM25Searcher
from inference.embedding_search import EmbeddingSearch
from inference.reranker import Reranker
from inference.generator import AsyncQwenGenerator
from inference.request_tracer import RequestTracer, create_tracer

from utils.logger import get_logger, set_request_context, clear_request_context
from utils.config_loader import load_config

logger = get_logger(__name__)

class RAGPipeline:
    """
    Orchestrates the 4-Stage RAG Pipeline with memory safety optimizations.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        
        self.paths = self.config['paths']
        self.models = self.config['models']
        self.retrieval_conf = self.config['retrieval']
        self.gen_conf = self.config['generation']
        
        # Component Instances
        self.bm25: Optional[BM25Searcher] = None
        self.embedding: Optional[EmbeddingSearch] = None
        self.reranker: Optional[Reranker] = None
        self.generator: Optional[AsyncQwenGenerator] = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def initialize(self):
        """
        Initialize models with strict memory budgeting for single-GPU setup.
        """
        logger.info("Initializing RAG Pipeline components...")
        
        # 1. Load BM25 (CPU only, safe)
        self.bm25 = BM25Searcher(artifacts_dir=self.paths['bm25_artifacts'])
        self.bm25.load_bm25_artifacts()
        
        # 2. Load Embedding Search (GPU Model 1)
        # Uses ~600MB-1GB VRAM depending on model
        self.embedding = EmbeddingSearch(
            embedding_model_name=self.models['embedding'],
            device=self.device,
            truncate_dim=self.retrieval_conf.get('truncate_dim')
        )
        self.embedding.load(Path(self.paths['embeddings']))
        
        # 3. Load Reranker (GPU Model 2)
        # Uses ~1-2GB VRAM. We set batch_size=4 to prevent OOM spikes during inference.
        self.reranker = Reranker(
            model_name=self.models['reranker'],
            device=self.device,
            batch_size=4  # <--- REDUCED BATCH SIZE FOR SAFETY
        )
        
        # 4. Load Generator (GPU Model 3 - The Memory Hog)
        # CRITICAL FIX: Utilization set to 0.4.
        # On a 16GB card:
        #   0.4 * 16GB = ~6.4GB for vLLM KV Cache + Weights
        #   Leaves ~9.6GB for Embedding(1GB) + Reranker(2GB) + System Overhead
        self.generator = AsyncQwenGenerator(
            model_name=self.models['generator'],
            gpu_memory_utilization=0.4, # <--- LOWERED FROM 0.65
            tensor_parallel_size=1
        )
        await self.generator.initialize()
        
        logger.info("Pipeline initialized successfully with memory optimizations.")

    async def cleanup(self):
        """Free resources and handle distributed shutdown."""
        logger.info("Cleaning up pipeline resources...")
        
        # 1. Cleanup Generator (vLLM)
        if self.generator:
            # vLLM doesn't always have a clean shutdown method, but deleting it helps
            await self.generator.cleanup()
            self.generator = None
        
        # 2. Cleanup other components
        self.bm25 = None
        self.embedding = None
        self.reranker = None
        
        # 3. Force Garbage Collection to release GPU memory pointers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 4. CRITICAL FIX: Clean up the distributed process group (NCCL)
        # This fixes the "destroy_process_group() was not called" error
        if dist.is_initialized():
            logger.info("Destroying distributed process group...")
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error destroying process group: {e}")

        logger.info("Pipeline resources cleaned up.")

    async def answer_stream(
        self, 
        query: str, 
        conversation_id: str = "default-session"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        set_request_context(conversation_id=conversation_id)
        tracer = create_tracer(conversation_id=conversation_id)
        trace_id = tracer.start_request(query, config=self.config)
        
        start_time = time.perf_counter()
        
        try:
            # =================================================================
            # STAGE 1: Hybrid Retrieval
            # =================================================================
            logger.info(f"Stage 1: Hybrid Retrieval")
            
            # BM25 (CPU)
            bm25_results = self.bm25.search(query, k=self.retrieval_conf['k_papers'] * 2)
            tracer.capture_bm25(bm25_results, k=self.retrieval_conf['k_papers'] * 2)
            
            # Embedding (GPU - Low VRAM impact)
            emb_paper_results = self.embedding.search(
                query, 
                collection_num=1, 
                k=self.retrieval_conf['k_papers'] * 2
            )
            tracer.capture_embedding_paper(emb_paper_results, k=self.retrieval_conf['k_papers'] * 2)
            
            # Fusion
            selected_papers = self._hybrid_fusion(
                bm25_results, 
                emb_paper_results, 
                k=self.retrieval_conf['k_papers']
            )
            
            # Capture Hybrid metrics
            tracer.capture_hybrid(
                file_scores={}, 
                bm25_results=bm25_results,
                emb_results=emb_paper_results,
                selected_files=selected_papers,
                bm25_weight=self.retrieval_conf['bm25_weight'],
                embedding_weight=self.retrieval_conf['embedding_weight'],
                k=self.retrieval_conf['k_papers']
            )
            
            yield {"type": "status", "stage": "stage_1_complete", "papers_found": len(selected_papers)}

            # =================================================================
            # STAGE 2: Filtered Chunk Retrieval
            # =================================================================
            logger.info(f"Stage 2: Chunk Retrieval")
            
            chunk_results = self.embedding.search(
                query,
                collection_num=2,
                k=self.retrieval_conf['m_chunks'],
                file_path_filter=selected_papers
            )
            
            tracer.capture_embedding_chunk(
                chunk_results, 
                m=self.retrieval_conf['m_chunks'], 
                file_filter=selected_papers
            )
            
            yield {"type": "status", "stage": "stage_2_complete", "chunks_found": len(chunk_results)}

            # =================================================================
            # STAGE 3: Reranking (Peak Memory Usage)
            # =================================================================
            logger.info("Stage 3: Reranking")
            
            # Explicitly clear cache before the heavy reranker step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reranked_results = self.reranker.rerank_with_details(
                query,
                candidates=chunk_results,
                top_k=self.retrieval_conf['n_reranked']
            )
            
            tracer.capture_reranker(
                reranked_results, 
                original_chunks=chunk_results, 
                n=self.retrieval_conf['n_reranked']
            )
            
            yield {
                "type": "context", 
                "data": [
                    {"text": r['text'], "metadata": r['metadata'], "score": r['rerank_score']} 
                    for r in reranked_results
                ]
            }

            # =================================================================
            # STAGE 4: Generation
            # =================================================================
            logger.info("Stage 4: Generation")
            
            accumulated_text = ""
            async for token in self.generator.generate_streaming(
                query=query,
                contexts=reranked_results,
                temperature=self.gen_conf['temperature'],
                include_citations=self.gen_conf['include_citations']
            ):
                accumulated_text += token
                yield {"type": "token", "content": token}
            
            tracer.capture_generator(
                answer=accumulated_text,
                contexts=reranked_results,
                duration_ms=(time.perf_counter() - start_time) * 1000
            )
            
            tracer.finish_request(success=True)
            yield {"type": "done", "trace_id": trace_id}

        except Exception as e:
            logger.error(f"Pipeline Error: {e}", exc_info=True)
            # Try to capture error state in trace
            tracer.finish_request(success=False, error=str(e))
            yield {"type": "error", "message": str(e)}
        
        finally:
            clear_request_context()

    def _hybrid_fusion(self, bm25_list, emb_list, k):
        """Simple rank fusion logic."""
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

if __name__ == "__main__":
    # FIX: Force 'spawn' method before any CUDA init happens
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Context might already be set

    # Also set env var here for safety 
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    async def main():
        pipeline = RAGPipeline()
        
        try:
            await pipeline.initialize()
            
            query = "What is the impact of matrix effects?"
            print(f"Query: {query}\n")
            
            async for update in pipeline.answer_stream(query):
                if update['type'] == 'token':
                    print(update['content'], end="", flush=True)
                elif update['type'] == 'error':
                    print(f"\nERROR: {update['message']}")
                    
        except Exception as e:
            print(f"\nCRITICAL FAILURE: {e}")
        finally:
            # Ensure cleanup always runs even if there is a crash
            await pipeline.cleanup()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass