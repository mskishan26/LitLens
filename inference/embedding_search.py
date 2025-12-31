"""
We are using l2 for now, so lower the score the better.
Need to configure the metadata inside the chromadb.
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
from functools import lru_cache
import time

from utils.logger import (
    get_logger, 
    log_stage_start, 
    log_retrieval_metrics,
)

logger = get_logger(__name__)


class EmbeddingSearch:
    """
    Handles loading ChromaDB collections and performing search queries.
    Thread-safe singleton for FastAPI deployment.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to prevent multiple model loads in FastAPI workers."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        embedding_model_name: str = "infgrad/Jasper-Token-Compression-600M",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        truncate_dim: Optional[int] = 1024
    ):
        """
        Initialize embedding search (singleton - only runs once).
        
        Args:
            embedding_model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            truncate_dim: Truncate embeddings to this dimension (e.g., 1024 from 4096)
        """
        # Skip if already initialized (singleton pattern)
        if self._initialized:
            return
            
        self.device = device
        
        logger.info(f"Loading embedding model '{embedding_model_name}' on {device}")
        
        model_kwargs = {
            'dtype': torch.bfloat16 if device == 'cuda' else torch.float32,
            'attn_implementation': "sdpa",
            'trust_remote_code': True
        }
        
        if truncate_dim:
            self.model = SentenceTransformer(
                embedding_model_name, 
                device=device,
                truncate_dim=truncate_dim,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
                tokenizer_kwargs={"padding_side": "left"}
            )
            self.embedding_dim = truncate_dim
        else:
            self.model = SentenceTransformer(
                embedding_model_name, 
                device=device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
                tokenizer_kwargs={"padding_side": "left"}
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.client = None
        self.collection1 = None
        self.collection2 = None
        
        self._initialized = True
        logger.info(f"Initialized with embedding dimension: {self.embedding_dim}")
    
    def load(self, input_path: Path):
        """
        Load ChromaDB collections from disk.
        ChromaDB PersistentClient runs on CPU RAM and is thread-safe for concurrent reads.
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise ValueError(f"ChromaDB path does not exist: {input_path}")
        
        logger.info(f"Loading ChromaDB from {input_path}")
        
        try:
            # Initialize persistent client (thread-safe for reads)
            self.client = chromadb.PersistentClient(
                path=str(input_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False  # Safety: prevent accidental resets
                )
            )
            
            # Load collections
            self.collection1 = self.client.get_collection(name="paper_level")
            self.collection2 = self.client.get_collection(name="chunk_level")
            
            # Get metadata
            meta1 = self.collection1.metadata
            meta2 = self.collection2.metadata
            
            logger.info(f"Loaded collections:")
            logger.info(f"  Collection 1 (paper-level): {self.collection1.count()} vectors")
            logger.info(f"  Collection 2 (chunk-level): {self.collection2.count()} vectors")
            logger.info(f"  Embedding dimension: {self.embedding_dim}")
            
        except ValueError as e:
            # Collection doesn't exist
            logger.error(f"ChromaDB collection not found: {e}")
            raise ValueError(f"Required collections not found in {input_path}. Did you run embedding generation?")
        except Exception as e:
            # Corrupted DB or other issues
            logger.error(f"Error loading ChromaDB: {e}")
            raise RuntimeError(f"Failed to load ChromaDB from {input_path}. Database may be corrupted. Error: {e}")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
        
        Returns:
            Query embedding array
            
        Note:
            - prompt_name="query": Jasper uses asymmetric query/document encoding
            - compression_ratio=0: Disabled for short queries (token compression helps with long docs)
        """
        try:
            embedding = self.model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name="query",  # Asymmetric encoding: query prompt
                compression_ratio=0    # No compression for queries (already short)
            )
            
            return embedding[0]  # Return 1D array for single query
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise RuntimeError(f"Failed to encode query. Error: {e}")
    
    def search(
        self,
        query: str,
        collection_num: int = 1,
        k: int = 5,
        file_path_filter: Optional[set] = None
    ) -> List[Tuple[float, Dict, str]]:
        """
        Search a collection for relevant chunks.
        
        Args:
            query: Search query
            collection_num: 1 or 2 for which collection
            k: Number of results (capped at 100 for safety)
            file_path_filter: Set of file paths to restrict search to
        
        Returns:
            List of (distance, metadata, chunk_text) tuples
            
        Raises:
            ValueError: If collections not loaded or invalid parameters
            RuntimeError: If search fails
        """
        if self.collection1 is None or self.collection2 is None:
            raise ValueError("Collections not loaded. Call load() first.")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.perf_counter()
        log_stage_start(logger, "embedding", k=k, collection=collection_num)
        
        # Cap k to prevent excessive memory usage
        k = min(k, 100)
        
        collection = self.collection1 if collection_num == 1 else self.collection2
        
        try:
            query_embedding = self._embed_query(query)
            
            # Build where filter if file_path_filter is provided
            where_filter = None
            if file_path_filter:
                # ChromaDB uses $in operator for filtering
                where_filter = {"file_path": {"$in": list(file_path_filter)}}
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert ChromaDB results to expected format
            output = []
            top_score = 0.0
            if results['ids'][0]:  # Check if we got any results
                top_score = results['distances'][0][0]
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    metadata['chroma_id'] = results['ids'][0][i]
                    document = results['documents'][0][i]
                    
                    output.append((float(distance), metadata, document))

            duration_ms = (time.perf_counter() - start_time) * 1000
            log_retrieval_metrics(
                logger, 
                stage="embedding", 
                count=len(output), 
                duration_ms=duration_ms, 
                top_score=top_score,
                collection=collection_num
            )

            return output
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise RuntimeError(f"Search failed. Error: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about loaded collections."""
        if self.collection1 is None or self.collection2 is None:
            return {"status": "not_loaded"}
        
        try:
            meta1 = self.collection1.metadata
            meta2 = self.collection2.metadata
            
            return {
                "status": "loaded",
                "collection1": {
                    "purpose": meta1.get('purpose', 'unknown'),
                    "total_vectors": self.collection1.count(),
                    "dimension": self.embedding_dim,
                    "chunking_strategy": meta1.get('chunking_strategy', 'unknown'),
                    "chunk_size": meta1.get('chunk_size', 'unknown')
                },
                "collection2": {
                    "purpose": meta2.get('purpose', 'unknown'),
                    "total_vectors": self.collection2.count(),
                    "dimension": self.embedding_dim,
                    "chunking_strategy": meta2.get('chunking_strategy', 'unknown'),
                    "chunk_size": meta2.get('chunk_size', 'unknown')
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_document_by_id(self, doc_id: str, collection_num: int = 1) -> Optional[Dict]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            doc_id: Document ID
            collection_num: 1 or 2 for which collection
        
        Returns:
            Dictionary with document, metadata, and embedding, or None if not found
        """
        if self.collection1 is None or self.collection2 is None:
            logger.warning("Attempted to get document before collections loaded")
            return None
            
        collection = self.collection1 if collection_num == 1 else self.collection2
        
        try:
            result = collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0] if result['embeddings'] else None
                }
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
        
        return None
    
    def list_papers(self, collection_num: int = 1, limit: int = 1000) -> List[str]:
        """
        Get list of unique paper file paths in a collection.
        
        Args:
            collection_num: 1 or 2 for which collection
            limit: Maximum number of documents to fetch (prevents OOM)
        
        Returns:
            List of unique file paths
            
        Note:
            This fetches metadata for up to 'limit' documents. For very large collections,
            consider implementing pagination or using a separate papers index.
        """
        if self.collection1 is None or self.collection2 is None:
            logger.warning("Attempted to list papers before collections loaded")
            return []
            
        collection = self.collection1 if collection_num == 1 else self.collection2
        
        try:
            # Get documents with limit to prevent OOM
            results = collection.get(
                limit=limit,
                include=["metadatas"]
            )
            
            # Extract unique file paths
            file_paths = set()
            for metadata in results['metadatas']:
                if 'file_path' in metadata:
                    file_paths.add(metadata['file_path'])
            
            sorted_paths = sorted(list(file_paths))
            
            if len(results['metadatas']) == limit:
                logger.warning(f"Hit limit of {limit} documents. Some papers may be missing from list.")
            
            return sorted_paths
            
        except Exception as e:
            logger.error(f"Error listing papers: {e}")
            return []
    
    def health_check(self) -> Dict[str, any]:
        """
        Health check for monitoring/debugging.
        
        Returns:
            Dictionary with system status
        """
        return {
            "model_loaded": self._initialized,
            "model_device": self.device if self._initialized else None,
            "collections_loaded": self.collection1 is not None and self.collection2 is not None,
            "embedding_dim": self.embedding_dim if self._initialized else None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        }


# FastAPI integration helper
@lru_cache(maxsize=1)
def get_embedding_searcher(
    model_name: str = "infgrad/Jasper-Token-Compression-600M",
    db_path: str = "/path/to/chromadb"
) -> EmbeddingSearch:
    """
    FastAPI dependency for getting embedding searcher singleton.
    Use with Depends() in your FastAPI routes.
    
    Example:
        @app.get("/search")
        async def search_endpoint(
            query: str,
            searcher: EmbeddingSearch = Depends(get_embedding_searcher)
        ):
            results = searcher.search(query)
            return results
    """
    searcher = EmbeddingSearch(embedding_model_name=model_name)
    searcher.load(Path(db_path))
    return searcher


def main():
    """Test function"""
    from utils.logger import set_request_context, clear_request_context 
    set_request_context(req_id="embed-test", conversation_id="test-session")
    embedding_searcher = EmbeddingSearch()
    embedding_searcher.load("/scratch/sathishbabu.ki/data_files/embeddings")
    
    # Health check
    print("Health check:", embedding_searcher.health_check())
    
    try:
        # Test search
        results = embedding_searcher.search(
            query="What are the tradeoffs and assumptions of intensity normalization using an internal standard reference of MALDI MSI datasets with multiple samples?",
            k=5,
            collection_num = 2
        )
        
        print(f"\nFound {len(results)} results:")
        for i, (distance, metadata, text) in enumerate(results, 1):
            print(f"\n{i}. Distance: {distance:.4f}")
            print(f"   File: {metadata.get('file_path', 'unknown')}")
            print(f"   Text: {text[:200]}...")
    finally:
        clear_request_context()


if __name__ == "__main__":
    main()