"""
Production RAG Pipeline
========================

Three-stage retrieval pipeline with memory-efficient model loading:
1. Hybrid retrieval: BM25 + Paper-level embeddings → k papers
2. Chunk retrieval: Chunk-level embeddings filtered by k papers → m chunks  
3. Reranking: Qwen reranker → n chunks
4. Generation: Qwen LLM with context

Models are loaded/unloaded sequentially to optimize memory usage.
"""

import torch
import gc
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
from datetime import datetime
import time


from utils.logger import get_chat_logger
from utils.config_loader import load_config

logger = get_chat_logger(__name__)


def clear_gpu_memory():
    """Aggressively clear GPU cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class RAGPipeline:
    """Memory-efficient RAG pipeline with hybrid retrieval and sequential model loading."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        no_unload: bool = False
    ):
        """
        Initialize pipeline with model configurations from config file.
        
        Args:
            config_path: Path to config file (optional)
            device: Computation device ('cuda' or 'cpu')
            no_unload: If True, keep models loaded in memory between queries
        """
        self.config = load_config(config_path)
        self.no_unload = no_unload
        
        self.embeddings_path = Path(self.config['paths']['embeddings'])
        self.bm25_artifacts_path = Path(self.config['paths']['bm25_artifacts'])
        
        self.embedding_model = self.config['models']['embedding']
        self.reranker_model = self.config['models']['reranker']
        self.generator_model = self.config['models']['generator']
        
        self.truncate_dim = self.config['retrieval']['truncate_dim']
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model instances (loaded on-demand)
        self.bm25_searcher = None
        self.embedding_searcher = None
        self.reranker = None
        self.generator = None
        
        logger.info("RAG Pipeline initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embeddings: {self.embeddings_path}")
        logger.info(f"  BM25: {self.bm25_artifacts_path}")
        logger.info(f"  No Unload: {self.no_unload}")

        if self.no_unload:
            self.load_all_models()
    
    def load_all_models(self):
        """Pre-load all models into memory."""
        logger.info("Pre-loading all models...")
        self._load_bm25()
        self._load_embedding_search()
        self._load_reranker()
        self._load_generator()
        logger.info("All models pre-loaded.")

    def _load_bm25(self):
        """Load BM25 search index."""
        if self.bm25_searcher is None:
            logger.info("Loading BM25 searcher")
            from inference.bm25_search import BM25Searcher
            
            self.bm25_searcher = BM25Searcher(str(self.bm25_artifacts_path))
            self.bm25_searcher.load_bm25_artifacts()
            self._log_memory()
    
    def _unload_bm25(self):
        """Unload BM25 searcher."""
        if self.no_unload:
            return
            
        if self.bm25_searcher is not None:
            logger.info("Unloading BM25 searcher")
            del self.bm25_searcher
            self.bm25_searcher = None
            clear_gpu_memory()
    
    def _load_embedding_search(self):
        """Load embedding search system (both indices)."""
        if self.embedding_searcher is None:
            logger.info("Loading embedding searcher")
            from inference.embedding_search import EmbeddingSearch
            
            self.embedding_searcher = EmbeddingSearch(
                embedding_model_name=self.embedding_model,
                device=self.device,
                truncate_dim=self.truncate_dim
            )
            self.embedding_searcher.load(self.embeddings_path)
            self._log_memory()
    
    def _unload_embedding_search(self):
        """Unload embedding search system."""
        if self.no_unload:
            return

        if self.embedding_searcher is not None:
            logger.info("Unloading embedding searcher")
            del self.embedding_searcher.model
            del self.embedding_searcher
            self.embedding_searcher = None
            clear_gpu_memory()
    
    def _load_reranker(self):
        """Load reranker model."""
        if self.reranker is None:
            logger.info("Loading reranker")
            from inference.reranker import Reranker, get_optimal_reranker_batch_size
            
            batch_size = get_optimal_reranker_batch_size()
            self.reranker = Reranker(
                model_name=self.reranker_model,
                device=self.device,
                batch_size=batch_size
            )
            self._log_memory()
    
    def _unload_reranker(self):
        """Unload reranker model."""
        if self.no_unload:
            return

        if self.reranker is not None:
            logger.info("Unloading reranker")
            del self.reranker.model
            del self.reranker
            self.reranker = None
            clear_gpu_memory()
    
    def _load_generator(self):
        """Load generator model."""
        if self.generator is None:
            logger.info("Loading generator")
            from inference.generator import QwenGenerator
            
            self.generator = QwenGenerator(
                model_name=self.generator_model,
                device=self.device
            )
            self._log_memory()
    
    def _unload_generator(self):
        """Unload generator model."""
        if self.no_unload:
            return

        if self.generator is not None:
            logger.info("Unloading generator")
            del self.generator.model
            del self.generator
            self.generator = None
            clear_gpu_memory()
    
    def _log_memory(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.debug(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def _log_to_session_file(self, file_path: Path, header: str, content: str):
        """Append formatted log entry to session file."""
        if not file_path:
            return
            
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{header}\n")
                f.write(f"{'-'*len(header)}\n")
                f.write(f"{content}\n")
        except Exception as e:
            logger.error(f"Failed to write to session log: {e}")

    def _hybrid_retrieval(
        self,
        query: str,
        k: int,
        bm25_weight: float,
        embedding_weight: float,
        session_file: Optional[Path] = None,
        timings: Optional[Dict[str, float]] = None
    ) -> Set[str]:
        """
        Stage 1: Combine BM25 and paper-level embedding search.
        """
        logger.info(f"Stage 1: Hybrid retrieval (k={k}, bm25={bm25_weight:.2f}, emb={embedding_weight:.2f})")
        
        # Validate weights
        if not abs(bm25_weight + embedding_weight - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {bm25_weight + embedding_weight}")
        
        # BM25 search
        self._load_bm25()
        
        t0 = time.perf_counter()
        bm25_results = self.bm25_searcher.search(query, k=k * 2)  # Get more for combining
        t1 = time.perf_counter()
        if timings is not None:
            timings['bm25'] = timings.get('bm25', 0.0) + (t1 - t0)
            
        bm25_files = set(bm25_results)
        self._unload_bm25()
        
        if session_file:
            bm25_log = "\n".join([f"{i+1}. {f}" for i, f in enumerate(bm25_results)])
            self._log_to_session_file(session_file, "BM25 Rankings (Stage 1)", bm25_log)
        
        logger.info(f"  BM25 retrieved {len(bm25_files)} papers")
        
        # Embedding search (paper-level)
        self._load_embedding_search()
        
        t0 = time.perf_counter()
        emb_results = self.embedding_searcher.search(query, collection_num=1, k=k * 2)
        t1 = time.perf_counter()
        if timings is not None:
            timings['embedding'] = timings.get('embedding', 0.0) + (t1 - t0)
            
        emb_files = {meta['file_path'] for _, meta, _ in emb_results}
        
        if session_file:
            emb_log = "\n".join([f"{i+1}. {meta.get('file_path', 'unknown')} (score: {dist:.4f})" 
                               for i, (dist, meta, _) in enumerate(emb_results)])
            self._log_to_session_file(session_file, "Embedding Rankings (Stage 1)", emb_log)
        
        logger.info(f"  Embedding retrieved {len(emb_files)} papers")
        
        # Combine results with weighted voting
        all_files = bm25_files.union(emb_files)
        file_scores = {}
        
        for file in all_files:
            bm25_score = (k * 2 - bm25_results.index(file)) if file in bm25_results else 0
            emb_score = 0
            
            for idx, (_, meta, _) in enumerate(emb_results):
                if meta['file_path'] == file:
                    emb_score = k * 2 - idx
                    break
            
            # Normalize to [0, 1] and apply weights
            bm25_norm = bm25_score / (k * 2) if k > 0 else 0
            emb_norm = emb_score / (k * 2) if k > 0 else 0
            
            file_scores[file] = bm25_weight * bm25_norm + embedding_weight * emb_norm
        
        # Get top-k by combined score
        top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected_files = {file for file, _ in top_files}
        
        logger.info(f"  Combined ranking selected {len(selected_files)} papers")
        
        return selected_files

    def _chunk_retrieval(
        self,
        query: str,
        file_filter: Set[str],
        m: int,
        timings: Optional[Dict[str, float]] = None
    ) -> list:
        """
        Stage 2: Retrieve chunks from selected papers.
        """
        logger.info(f"Stage 2: Chunk retrieval (m={m}, filtered to {len(file_filter)} papers)")
        
        # Search chunk-level index with file filter
        t0 = time.perf_counter()
        chunk_results = self.embedding_searcher.search(
            query=query,
            collection_num=2,
            k=m,
            file_path_filter=file_filter
        )
        t1 = time.perf_counter()
        if timings is not None:
            timings['embedding'] = timings.get('embedding', 0.0) + (t1 - t0)
        
        logger.info(f"  Retrieved {len(chunk_results)} chunks")
        
        # Unload embedding search now that we're done with retrieval
        self._unload_embedding_search()
        
        return chunk_results
    
    def _rerank_chunks(
        self,
        query: str,
        chunks: list,
        n: int,
        session_file: Optional[Path] = None,
        timings: Optional[Dict[str, float]] = None
    ) -> list:
        """
        Stage 3: Rerank chunks using Qwen reranker.
        """
        logger.info(f"Stage 3: Reranking (n={n})")
        
        self._load_reranker()
        
        t0 = time.perf_counter()
        reranked = self.reranker.rerank_with_details(
            query=query,
            candidates=chunks,
            top_k=n
        )
        t1 = time.perf_counter()
        if timings is not None:
            timings['reranker'] = timings.get('reranker', 0.0) + (t1 - t0)
        
        if session_file:
            rerank_log = ""
            for item in reranked:
                rank = item['rank']
                score = item['rerank_score']
                title = item['metadata'].get('paper_title', 'Unknown')
                fname = Path(item['metadata'].get('file_path', '')).name
                rerank_log += f"{rank}. [{score:.4f}] {title} ({fname})\n"
            self._log_to_session_file(session_file, "Reranker Rankings (Stage 3)", rerank_log)
        
        logger.info(f"  Selected top {len(reranked)} chunks")
        
        self._unload_reranker()
        
        return reranked
    
    def _generate_answer(
        self,
        query: str,
        contexts: list,
        temperature: float,
        include_citations: bool,
        timings: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Stage 4: Generate answer using Qwen LLM.
        """
        logger.info("Stage 4: Generating answer")
        
        self._load_generator()
        
        t0 = time.perf_counter()
        response = self.generator.generate(
            query=query,
            contexts=contexts,
            temperature=temperature,
            include_citations=include_citations
        )
        t1 = time.perf_counter()
        if timings is not None:
            timings['generator'] = timings.get('generator', 0.0) + (t1 - t0)
        
        logger.info(f"  Generated {len(response['answer'].split())} words")
        
        self._unload_generator()
        
        return response
    
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
        return_metadata: bool = False,
        session_file: Optional[Path] = None
    ) -> Dict:
        """
        Answer a question using the full RAG pipeline.
        """
        # Use config defaults if not provided
        k = k or self.config['retrieval']['k_papers']
        m = m or self.config['retrieval']['m_chunks']
        n = n or self.config['retrieval']['n_reranked']
        bm25_weight = bm25_weight if bm25_weight is not None else self.config['retrieval']['bm25_weight']
        embedding_weight = embedding_weight if embedding_weight is not None else self.config['retrieval']['embedding_weight']
        temperature = temperature if temperature is not None else self.config['generation']['temperature']
        include_citations = include_citations if include_citations is not None else self.config['generation']['include_citations']

        logger.info("="*80)
        logger.info(f"Query: {query}")
        logger.info("="*80)
        
        if session_file:
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*40 + "\n")
                f.write(f"Question: {query}\n")
                f.write("="*40 + "\n")
        
        start_time = datetime.now()
        timings = {}
        
        try:
            # Stage 1: Hybrid retrieval
            selected_papers = self._hybrid_retrieval(
                query=query,
                k=k,
                bm25_weight=bm25_weight,
                embedding_weight=embedding_weight,
                session_file=session_file,
                timings=timings
            )
            
            # Stage 2: Chunk retrieval (need embedding_searcher loaded)
            chunk_results = self._chunk_retrieval(
                query=query,
                file_filter=selected_papers,
                m=m,
                timings=timings
            )
            
            # Stage 3: Reranking
            reranked_chunks = self._rerank_chunks(
                query=query,
                chunks=chunk_results,
                n=n,
                session_file=session_file,
                timings=timings
            )
            
            # Stage 4: Generation
            generation_response = self._generate_answer(
                query=query,
                contexts=reranked_chunks,
                temperature=temperature,
                include_citations=include_citations,
                timings=timings
            )
            
            if session_file:
                self._log_to_session_file(session_file, "Generator Output", generation_response['answer'])
            
            # Build response
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if session_file:
                total_exec = sum(timings.values())
                meta_log = f"total time: {elapsed:.2f}s\n"
                meta_log += f"total execution time: {total_exec:.2f}s\n"
                meta_log += f"bm25 step: {timings.get('bm25', 0):.2f}s\n"
                meta_log += f"embedding step: {timings.get('embedding', 0):.2f}s\n"
                meta_log += f"reranker step: {timings.get('reranker', 0):.2f}s\n"
                meta_log += f"generator step: {timings.get('generator', 0):.2f}s\n\n"
                
                meta_log += f"Papers selected: {len(selected_papers)}\n"
                meta_log += f"Chunks retrieved: {len(chunk_results)}\n"
                meta_log += f"Contexts used: {generation_response.get('num_contexts_used', n)}"
                self._log_to_session_file(session_file, "Metadata", meta_log)
            
            response = {
                'query': query,
                'answer': generation_response['answer'],
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'k_papers': k,
                    'm_chunks': m,
                    'n_reranked': n,
                    'bm25_weight': bm25_weight,
                    'embedding_weight': embedding_weight,
                    'temperature': temperature
                }
            }
            
            if return_metadata:
                response['metadata'] = {
                    'papers_selected': len(selected_papers),
                    'chunks_retrieved': len(chunk_results),
                    'contexts_used': generation_response.get('num_contexts_used', n),
                    'elapsed_seconds': elapsed,
                    'generation_params': generation_response.get('generation_params', {})
                }
            
            logger.info("="*80)
            logger.info(f"Pipeline completed in {elapsed:.2f}s")
            logger.info("="*80)
            
            return response
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    def answer_streaming(
        self,
        query: str,
        k: Optional[int] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        embedding_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        include_citations: Optional[bool] = None,
        return_metadata: bool = False,
        session_file: Optional[Path] = None
    ):
        """
        Answer a question using the full RAG pipeline with streaming output.
        
        Yields:
            str: Tokens of the answer
            dict: Metadata (contexts, etc.) - typically yielded once
        """
        # Use config defaults if not provided
        k = k or self.config['retrieval']['k_papers']
        m = m or self.config['retrieval']['m_chunks']
        n = n or self.config['retrieval']['n_reranked']
        bm25_weight = bm25_weight if bm25_weight is not None else self.config['retrieval']['bm25_weight']
        embedding_weight = embedding_weight if embedding_weight is not None else self.config['retrieval']['embedding_weight']
        temperature = temperature if temperature is not None else self.config['generation']['temperature']
        include_citations = include_citations if include_citations is not None else self.config['generation']['include_citations']

        logger.info("="*80)
        logger.info(f"Streaming Query: {query}")
        logger.info("="*80)
        
        if session_file:
            with open(session_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*40 + "\n")
                f.write(f"Question: {query}\n")
                f.write("="*40 + "\n")
        
        start_time = datetime.now()
        timings = {}
        
        try:
            # Stage 1: Hybrid retrieval
            selected_papers = self._hybrid_retrieval(
                query=query,
                k=k,
                bm25_weight=bm25_weight,
                embedding_weight=embedding_weight,
                session_file=session_file,
                timings=timings
            )
            
            # Stage 2: Chunk retrieval
            chunk_results = self._chunk_retrieval(
                query=query,
                file_filter=selected_papers,
                m=m,
                timings=timings
            )
            
            # Stage 3: Reranking
            reranked_chunks = self._rerank_chunks(
                query=query,
                chunks=chunk_results,
                n=n,
                session_file=session_file,
                timings=timings
            )
            
            # Yield metadata first so UI can show sources immediately if desired
            # (or we can yield it last, but having it available is nice)
            metadata_payload = {
                'contexts': reranked_chunks,
                'papers_selected': len(selected_papers),
                'chunks_retrieved': len(chunk_results)
            }
            yield metadata_payload
            
            # Stage 4: Generation (Streaming)
            logger.info("Stage 4: Generating answer (streaming)")
            self._load_generator()
            
            # We use the generator's streaming method
            # Note: generate_streaming yields tokens
            token_generator = self.generator.generate_streaming(
                query=query,
                contexts=reranked_chunks,
                temperature=temperature,
                include_citations=include_citations
            )
            
            accumulated_answer = ""
            
            t0 = time.perf_counter()
            for token in token_generator:
                accumulated_answer += token
                yield token
            t1 = time.perf_counter()
            timings['generator'] = timings.get('generator', 0.0) + (t1 - t0)
            
            if session_file:
                self._log_to_session_file(session_file, "Generator Output", accumulated_answer)
            
            self._unload_generator()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if session_file:
                total_exec = sum(timings.values())
                meta_log = f"total time: {elapsed:.2f}s\n"
                meta_log += f"total execution time: {total_exec:.2f}s\n"
                meta_log += f"bm25 step: {timings.get('bm25', 0):.2f}s\n"
                meta_log += f"embedding step: {timings.get('embedding', 0):.2f}s\n"
                meta_log += f"reranker step: {timings.get('reranker', 0):.2f}s\n"
                meta_log += f"generator step: {timings.get('generator', 0):.2f}s\n\n"
                
                meta_log += f"Papers selected: {len(selected_papers)}\n"
                meta_log += f"Chunks retrieved: {len(chunk_results)}\n"
                self._log_to_session_file(session_file, "Metadata", meta_log)
            
            logger.info("="*80)
            logger.info(f"Pipeline completed in {elapsed:.2f}s")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def cleanup(self):
        """Unload all models and free memory."""
        logger.info("Cleaning up pipeline")
        # Force unload everything
        self.no_unload = False
        self._unload_bm25()
        self._unload_embedding_search()
        self._unload_reranker()
        self._unload_generator()
        logger.info("Cleanup complete")


def main():
    """Example usage."""
    import os
    import json
    
    # Optimize memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Example query
    query = "What is matrix suppression and how does it relate to ion competition in mass spectrometry?"
    
    # Get answer
    result = pipeline.answer(
        query=query,
        return_metadata=True
    )
    
    # Display result
    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)
    print(result['answer'])
    print("="*80)
    
    # Save result
    # Load config just to get the output path, or access it via pipeline.config
    output_path = Path(pipeline.config['paths']['outputs']) / "rag_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved result to {output_path}")
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()
