import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from nltk.tokenize import word_tokenize
import nltk

from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from utils.logger import get_chat_logger
from utils.config_loader import load_config

logger_chat = get_chat_logger(__name__)

class BM25Searcher:
    """Search class for querying pre-built BM25 indices."""
    
    def __init__(self, artifacts_dir: str):
        """
        Initialize BM25 searcher.
        
        Args:
            artifacts_dir: Directory containing BM25 artifacts
            
        Raises:
            FileNotFoundError: If artifacts_dir doesn't exist
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.metadata: List[Dict[str, str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: Optional[List[List[str]]] = None
        self._loaded = False
        
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory does not exist: {artifacts_dir}"
            )

    def load_bm25_artifacts(self) -> None:
        """
        Load BM25 index and metadata from disk.
        
        Raises:
            FileNotFoundError: If required artifact files are missing
            IOError: If loading fails
        """
        required_files = ['bm25_index.pkl', 'tokenized_corpus.pkl', 'metadata.json']
        
        # Check all required files exist
        for filename in required_files:
            filepath = self.artifacts_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required artifact file missing: {filepath}"
                )
        
        try:
            logger_chat.info(f"Loading BM25 artifacts from {self.artifacts_dir}")
            
            # Load BM25 index
            with open(self.artifacts_dir / 'bm25_index.pkl', 'rb') as f:
                self.bm25 = pickle.load(f)
            
            # Load tokenized corpus
            with open(self.artifacts_dir / 'tokenized_corpus.pkl', 'rb') as f:
                self.tokenized_corpus = pickle.load(f)
            
            # Load metadata
            with open(self.artifacts_dir / 'metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self._loaded = True
            logger_chat.info(
                f"Successfully loaded {len(self.metadata)} documents"
            )
            
        except IOError as e:
            logger_chat.error(f"Failed to load artifacts: {e}")
            raise
        except Exception as e:
            logger_chat.error(f"Unexpected error loading artifacts: {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 30) -> List[str]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            k: Number of top results to return (default: 30)
            
        Returns:
            List of filenames for top-k matching documents
            
        Raises:
            RuntimeError: If artifacts not loaded
            ValueError: If k is invalid
        """
        if not self._loaded or self.bm25 is None:
            raise RuntimeError(
                "BM25 artifacts not loaded. Call load_bm25_artifacts() first."
            )
        
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        if k > len(self.metadata):
            logger_chat.warning(
                f"k={k} exceeds corpus size ({len(self.metadata)}), "
                f"returning all documents"
            )
            k = len(self.metadata)
        
        logger_chat.info(f"Searching for: '{query}' (top-{k})")
        
        try:
            # Tokenize query (simple split, matching BM25 expectations)
            tokenized_query = query.lower().split()
            
            # Get scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_n_indices = scores.argsort()[-k:][::-1]
            
            # Collect results
            filtered_files = []
            # logger_chat.info(f"\nBM25 Top {k} results:")
            for rank, idx in enumerate(top_n_indices, 1):
                filename = self.metadata[idx]['filename']
                score = scores[idx]
                # logger_chat.info(f"{rank:3d}. {filename:50s} (score: {score:.4f})")
                filtered_files.append(filename)
            
            return filtered_files
            
        except Exception as e:
            logger_chat.error(f"Search failed: {e}", exc_info=True)
            raise

def main() -> None:
    """
    Main function for BM25 index generation or search.
    
    Args:
        mode: Either 'generate' to create index or 'search' to query
    """
    config = load_config()

    artifacts_dir = config['paths']['bm25_artifacts']
    
    try:
        logger_chat.info("=== BM25 Search ===")
        
        # Load and search
        searcher = BM25Searcher(artifacts_dir=artifacts_dir)
        searcher.load_bm25_artifacts()
        
        # Example queries
        test_queries = [
            'What is MALDI?',
            'linear mixed effect models',
            'causal inference'
        ]
        
        for query in test_queries:
            results = searcher.search(query, k=10)
            print(f"\nQuery: '{query}' returned {len(results)} results")
        
        logger_chat.info("=== Search Complete ===")
            
    except Exception as e:
        logger_chat.error(f"Fatal error in main: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()