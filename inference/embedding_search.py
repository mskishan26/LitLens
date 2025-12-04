import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch

from utils.logger import get_chat_logger

logger = get_chat_logger(__name__)


class EmbeddingSearch:
    """Handles loading ChromaDB collections and performing search queries."""
    
    def __init__(
        self,
        embedding_model_name: str = "infgrad/Jasper-Token-Compression-600M",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        truncate_dim: Optional[int] = 1024
    ):
        """
        Initialize embedding search.
        
        Args:
            embedding_model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            truncate_dim: Truncate embeddings to this dimension (e.g., 1024 from 4096)
        """
        self.device = device
        
        logger.info(f"Loading embedding model '{embedding_model_name}' on {device}")
        
        model_kwargs = {
            'torch_dtype': torch.bfloat16 if device == 'cuda' else torch.float32,
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
        
        logger.info(f"Initialized with embedding dimension: {self.embedding_dim}")
    
    def load(self, input_path: Path):
        """Load ChromaDB collections from disk."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise ValueError(f"ChromaDB path does not exist: {input_path}")
        
        logger.info(f"Loading ChromaDB from {input_path}")
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=str(input_path),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Load collections
        try:
            self.collection1 = self.client.get_collection(name="paper_level")
            self.collection2 = self.client.get_collection(name="chunk_level")
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
            raise ValueError(f"Could not load collections from {input_path}. Error: {e}")
        
        # Get metadata
        meta1 = self.collection1.metadata
        meta2 = self.collection2.metadata
        
        logger.info(f"Loaded collections:")
        logger.info(f"  Collection 1 (paper-level): {self.collection1.count()} vectors")
        logger.info(f"  Collection 2 (chunk-level): {self.collection2.count()} vectors")
        logger.info(f"  Embedding dimension: {self.embedding_dim}")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
        
        Returns:
            Query embedding array
        """
        embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name="query",
            compression_ratio=0
        )
        
        return embedding[0]  # Return 1D array for single query
    
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
            k: Number of results
            file_path_filter: Set of file paths to restrict search to
        
        Returns:
            List of (distance, metadata, chunk_text) tuples
        """
        if self.collection1 is None or self.collection2 is None:
            raise ValueError("Collections not loaded. Call load() first.")
        
        collection = self.collection1 if collection_num == 1 else self.collection2
        
        logger.info(f"Searching collection {collection_num} for query: '{query[:100]}...'")
        
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
        if results['ids'][0]:  # Check if we got any results
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                
                output.append((float(distance), metadata, document))
        
        logger.info(f"Found {len(output)} results")
        return output
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about loaded collections."""
        if self.collection1 is None or self.collection2 is None:
            return {"status": "not_loaded"}
        
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
    
    def get_document_by_id(self, doc_id: str, collection_num: int = 1) -> Optional[Dict]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            doc_id: Document ID
            collection_num: 1 or 2 for which collection
        
        Returns:
            Dictionary with document, metadata, and embedding, or None if not found
        """
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
    
    def list_papers(self, collection_num: int = 1) -> List[str]:
        """
        Get list of unique paper file paths in a collection.
        
        Args:
            collection_num: 1 or 2 for which collection
        
        Returns:
            List of unique file paths
        """
        collection = self.collection1 if collection_num == 1 else self.collection2
        
        # Get all documents (ChromaDB might have limits, so be careful with large collections)
        results = collection.get(
            include=["metadatas"]
        )
        
        # Extract unique file paths
        file_paths = set()
        for metadata in results['metadatas']:
            if 'file_path' in metadata:
                file_paths.add(metadata['file_path'])
        
        return sorted(list(file_paths))