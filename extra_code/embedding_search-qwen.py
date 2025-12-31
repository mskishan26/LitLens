import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
from sentence_transformers import SentenceTransformer
import torch

from utils.logger import get_chat_logger

logger = get_chat_logger(__name__)


class EmbeddingSearch:
    """Handles loading indices and performing search queries."""
    
    def __init__(
        self,
        embedding_model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        truncate_dim: Optional[int] = None
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
            'dtype': torch.float16 if device == 'cuda' else torch.float32
        }
        
        if truncate_dim:
            self.model = SentenceTransformer(
                embedding_model_name, 
                device=device,
                truncate_dim=truncate_dim,
                model_kwargs=model_kwargs
            )
            self.embedding_dim = truncate_dim
        else:
            self.model = SentenceTransformer(
                embedding_model_name, 
                device=device,
                model_kwargs=model_kwargs
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.index1 = None
        self.index2 = None
        self.metadata1: List[Dict] = []
        self.metadata2: List[Dict] = []
        self.config = {}
        
        logger.info(f"Initialized with embedding dimension: {self.embedding_dim}")
    
    def load(self, input_path: Path):
        """Load both indices and metadata from disk."""
        input_path = Path(input_path)
        
        logger.info(f"Loading RAG store from {input_path}")
        
        self.index1 = faiss.read_index(str(input_path / "index1_paper_level.faiss"))
        self.index2 = faiss.read_index(str(input_path / "index2_chunk_level.faiss"))
        
        with open(input_path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.metadata1 = data['metadata1']
            self.metadata2 = data['metadata2']
            self.config = data.get('config', {})
        
        logger.info(f"Loaded indices:")
        logger.info(f"  Index 1 ({self.config.get('index1_purpose', 'unknown')}): {self.index1.ntotal} vectors")
        logger.info(f"  Index 2 ({self.config.get('index2_purpose', 'unknown')}): {self.index2.ntotal} vectors")
    
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
            prompt_name="query"
        )
        
        return embedding
    
    def search(
        self,
        query: str,
        index_num: int = 1,
        k: int = 5,
        file_path_filter: Optional[set] = None
    ) -> List[Tuple[float, Dict, str]]:
        """
        Search an index for relevant chunks.
        
        Args:
            query: Search query
            index_num: 1 or 2 for which index
            k: Number of results
            file_path_filter: Set of file paths to restrict search to
        
        Returns:
            List of (distance, metadata, chunk_text) tuples
        """
        if self.index1 is None or self.index2 is None:
            raise ValueError("Indices not loaded. Call load() first.")
        
        index = self.index1 if index_num == 1 else self.index2
        metadata_list = self.metadata1 if index_num == 1 else self.metadata2
        
        logger.info(f"Searching index {index_num} for query: '{query[:100]}...'")
        
        query_embedding = self._embed_query(query)
        
        search_k = k * 10 if file_path_filter else k
        distances, indices = index.search(query_embedding, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            meta = metadata_list[idx]
            
            if file_path_filter and meta['file_path'] not in file_path_filter:
                continue
            
            results.append((float(dist), meta, None))
            
            if len(results) >= k:
                break
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_index_stats(self) -> Dict:
        """Get statistics about loaded indices."""
        if self.index1 is None or self.index2 is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "index1": {
                "purpose": self.config.get('index1_purpose', 'unknown'),
                "total_vectors": self.index1.ntotal,
                "dimension": self.embedding_dim
            },
            "index2": {
                "purpose": self.config.get('index2_purpose', 'unknown'),
                "total_vectors": self.index2.ntotal,
                "dimension": self.embedding_dim
            },
            "config": self.config
        }