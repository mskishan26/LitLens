import asyncio
import time
import os
import gc
import sys
# Standard standard libs are safe at the top
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, AsyncGenerator, Tuple

# Utility imports are usually safe (unless they import torch internally)
from utils.logger import get_logger, set_request_context, clear_request_context
from utils.config_loader import load_config

logger = get_logger(__name__)

def reranker_tracer(a):
    tracer_dict = {}
    for chunk in a:
        chroma_id = chunk['metadata']['chroma_id']
        tracer_dict[chroma_id] = {}
        for metric in ['rank', 'rerank_score', 'original_distance', 'original_rank', 'rank_improvement']:
            tracer_dict[chroma_id][metric] = chunk[metric]
    return tracer_dict

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
        # We use 'Any' type hint because the classes aren't imported yet
        self.bm25: Any = None
        self.embedding: Any = None
        self.reranker: Any = None
        self.generator: Any = None
        
        # Determine device safely later
        self.device = "cpu"

    async def initialize(self):
        """
        Initialize models.
        CRITICAL FIX: All heavy imports are moved HERE (Lazy Loading).
        This prevents CUDA from initializing before 'spawn' is set in main.
        """
        logger.info("Initializing RAG Pipeline components...")

        # --- LAZY IMPORTS START ---
        # Import torch and inference modules ONLY when this method runs
        import torch 
        from inference.bm25_search import BM25Searcher
        from inference.embedding_search import EmbeddingSearch
        from inference.reranker import Reranker
        from inference.generator import AsyncQwenGenerator
        # --- LAZY IMPORTS END ---
        
        # Now it is safe to check for CUDA
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        # Keeping CPU as per your request for now, or mix match:
        # self.device = "cpu" # For BM25/Embedding if desired

        # 1. Load BM25 (CPU only)
        self.bm25 = BM25Searcher(artifacts_dir=self.paths['bm25_artifacts'])
        self.bm25.load_bm25_artifacts()
        
        # 2. Load Embedding Search (GPU Model 1)
        # It is safe to init GPU models now because spawn is already set
        self.embedding = EmbeddingSearch(
            embedding_model_name=self.models['embedding'],
            device=self.device,
            truncate_dim=self.retrieval_conf.get('truncate_dim')
        )
        self.embedding.load(Path(self.paths['embeddings']))
        
        # 3. Load Reranker (GPU Model 2)
        self.reranker = Reranker(
            model_name=self.models['reranker'],
            device=self.device,
            batch_size=4,
            timeout_seconds=60
        )
        
        # 4. Load Generator (GPU Model 3)
        self.generator = AsyncQwenGenerator(
            model_name=self.models['generator'],
            gpu_memory_utilization=0.4,
            tensor_parallel_size=1
        )
        await self.generator.initialize()
        
        logger.info("Pipeline initialized successfully with lazy imports.")

    async def cleanup(self):
        """Free resources."""
        # Need local imports since they aren't global
        import torch
        import torch.distributed as dist

        logger.info("Cleaning up pipeline resources...")
        
        if self.generator:
            await self.generator.cleanup()
            self.generator = None
        
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

        logger.info("Pipeline resources cleaned up.")

    async def answer_stream(
        self, 
        query: str, 
        conversation_id: str = "default-session"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        import torch
        start_time = time.perf_counter()
        
        try:
            # STAGE 1: Hybrid
            logger.info(f"Stage 1: Hybrid Retrieval")
            
            bm25_search_output = self.bm25.search(query, k=self.retrieval_conf['k_papers'] * 2)
            bm25_results, bm25_scores = map(list, zip(*bm25_search_output))
            # print(bm25_search_output)
            
            emb_paper_results = self.embedding.search(
                query, 
                collection_num=1, 
                k=self.retrieval_conf['k_papers'] * 2
            )
            emb_paper_results_tracer = [[item[1].get('chroma_id', 'N/A'), round(item[0], 4)] for item in emb_paper_results]
            # print(emb_paper_results_tracer)
            
            selected_papers = self._hybrid_fusion(
                bm25_results, 
                emb_paper_results, 
                k=self.retrieval_conf['k_papers']
            )
            
            # tracer.capture_hybrid(
            #     file_scores={}, 
            #     bm25_results=bm25_results,
            #     emb_results=emb_paper_results,
            #     selected_files=selected_papers,
            #     bm25_weight=self.retrieval_conf['bm25_weight'],
            #     embedding_weight=self.retrieval_conf['embedding_weight'],
            #     k=self.retrieval_conf['k_papers']
            # )
            yield {"type": "status", "stage": "stage_1_complete", "papers_found": len(selected_papers)}

            # STAGE 2: Chunks
            logger.info(f"Stage 2: Chunk Retrieval")
            chunk_results = self.embedding.search(
                query,
                collection_num=2,
                k=self.retrieval_conf['m_chunks'],
                file_path_filter=selected_papers
            )
            emb_chunk_results_tracer = [[item[1].get('chroma_id', 'N/A'), round(item[0], 4)] for item in emb_paper_results]
            # print(emb_chunk_results_tracer)
            # tracer.capture_embedding_chunk(
            #     chunk_results, m=self.retrieval_conf['m_chunks'], file_filter=selected_papers
            # )
            yield {"type": "status", "stage": "stage_2_complete", "chunks_found": len(chunk_results)}

            # STAGE 3: Reranking
            logger.info("Stage 3: Reranking")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reranked_results = self.reranker.rerank_with_details(
                query,
                candidates=chunk_results,
                top_k=self.retrieval_conf['n_reranked']
            )
            print(reranker_tracer(reranked_results))
            # tracer.capture_reranker(reranked_results, original_chunks=chunk_results, n=self.retrieval_conf['n_reranked'])
            
            yield {
                "type": "context", 
                "data": [{"text": r['text'], "metadata": r['metadata'], "score": r['rerank_score']} for r in reranked_results]
            }

            # STAGE 4: Generation
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
            
            # tracer.capture_generator(
            #     answer=accumulated_text,
            #     duration_ms=(time.perf_counter() - start_time) * 1000
            # )
            # tracer.finish_request(success=True)
            yield {"type": "done", "trace_id": trace_id}

        except Exception as e:
            logger.error(f"Pipeline Error: {e}", exc_info=True)
            # tracer.finish_request(success=False, error=str(e))
            yield {"type": "error", "message": str(e)}
        finally:
            clear_request_context()

    def _hybrid_fusion(self, bm25_list, emb_list, k):
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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. SETUP MULTIPROCESSING FIRST (Before ANY logic runs)
    import multiprocessing as mp
    try:
        # This MUST happen before any import triggers CUDA
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

    # 2. Set Env Var
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

    # 3. Run Pipeline
    async def main():
        # Because we used lazy imports, this __init__ is lightweight and safe
        pipeline = RAGPipeline() 
        
        try:
            # The heavy imports (and CUDA init) happen HERE, 
            # AFTER mp.set_start_method has definitely run.
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
            await pipeline.cleanup()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass