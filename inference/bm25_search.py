import re
import json
import pickle
import string
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from nltk.tokenize import word_tokenize
import nltk
import time

from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from utils.logger import (
    get_logger, 
    log_stage_start, 
    log_retrieval_metrics,
)

logger = get_logger(__name__)

class BM25Searcher:
    """Search class for querying pre-built BM25 indices."""
    
    # Query constraints
    MAX_QUERY_LENGTH = 1000  # characters
    MAX_TOKENS = 100  # tokens after preprocessing
    
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
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory does not exist: {artifacts_dir}"
            )

    def load_bm25_artifacts(self) -> None:
        """
        Load BM25 index and metadata from disk with thread safety.
        
        Raises:
            FileNotFoundError: If required artifact files are missing
            IOError: If loading fails
        """
        with self._lock:
            # Skip if already loaded
            if self._loaded:
                logger.info("Artifacts already loaded, skipping")
                return
                
            required_files = ['bm25_index.pkl', 'tokenized_corpus.pkl', 'metadata.json']
            
            # Check all required files exist
            for filename in required_files:
                filepath = self.artifacts_dir / filename
                if not filepath.exists():
                    raise FileNotFoundError(
                        f"Required artifact file missing: {filepath}"
                    )
            
            try:
                logger.info(f"Loading BM25 artifacts from {self.artifacts_dir}")
                
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
                logger.info(
                    f"Successfully loaded {len(self.metadata)} documents"
                )
                
            except IOError as e:
                logger.error(f"Failed to load artifacts: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error loading artifacts: {e}", exc_info=True)
                raise

    def _preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess and tokenize query to match corpus preprocessing.
        
        Must match BM25IndexBuilder.better_tokenize() logic:
        1. Basic sanitization (strip, length check)
        2. Lowercase conversion
        3. NLTK word_tokenize
        4. Filter to alphanumeric + hyphen tokens only
        5. Token count limit
        
        Note: No stopword removal to match corpus indexing.
        
        Args:
            query: Raw query string
            
        Returns:
            List of preprocessed tokens (empty list for invalid queries)
        """
        # Sanitization: strip whitespace
        query = query.strip()
        
        # Return empty list for empty query (graceful degradation)
        if not query:
            logger.debug("Empty query received, returning empty token list")
            return []
        
        # Check query length
        if len(query) > self.MAX_QUERY_LENGTH:
            logger.warning(
                f"Query length ({len(query)}) exceeds max ({self.MAX_QUERY_LENGTH}), truncating"
            )
            query = query[:self.MAX_QUERY_LENGTH]
        
        # Lowercase (matching corpus preprocessing)
        query = query.lower()
        
        # Tokenize using NLTK (matching corpus preprocessing)
        try:
            tokens = word_tokenize(query)
        except Exception as e:
            logger.warning(f"word_tokenize failed, falling back to split: {e}")
            tokens = query.split()
        
        # Filter to alphanumeric + hyphen only (matching corpus preprocessing)
        # Pattern: ^[a-zA-Z0-9\-]+$
        tokens = [t for t in tokens if re.match(r'^[a-zA-Z0-9\-]+$', t)]
        
        # Return empty list if no valid tokens (graceful degradation)
        if not tokens:
            logger.debug("No valid tokens after preprocessing, returning empty list")
            return []
        
        # Limit token count
        if len(tokens) > self.MAX_TOKENS:
            logger.warning(
                f"Token count ({len(tokens)}) exceeds max ({self.MAX_TOKENS}), truncating"
            )
            tokens = tokens[:self.MAX_TOKENS]
        
        return tokens

    def search(self, query: str, k: int = 30) -> List[str]:
        """
        Search for documents matching the query with thread safety.
        
        Args:
            query: Search query string
            k: Number of top results to return (default: 30)
            
        Returns:
            List of filenames for top-k matching documents (empty list for invalid queries)
            
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
            logger.warning(
                f"k={k} exceeds corpus size ({len(self.metadata)}), "
                f"returning all documents"
            )
            k = len(self.metadata)
        
        start_time = time.perf_counter()
        log_stage_start(logger, "bm25", query=query, k=k)
        
        try:
            # Preprocess query with sanitization
            tokenized_query = self._preprocess_query(query)
            
            # Return empty list if no valid tokens (graceful degradation)
            if not tokenized_query:
                logger.info("No valid tokens in query, returning empty results")
                return []
            
            logger.debug(f"Preprocessed query: {tokenized_query}")
            
            # Thread-safe read operation (BM25 scoring is read-only)
            with self._lock:
                # Get scores
                scores = self.bm25.get_scores(tokenized_query)
                
                # Get top-k results
                top_n_indices = scores.argsort()[-k:][::-1]
            
            # Collect results (metadata access is read-only, no lock needed after scoring)
            filtered_files = []
            top_score = 0.0

            if len(top_n_indices) > 0:
                top_score = float(scores[top_n_indices[0]])

            for idx in top_n_indices:
                filename = self.metadata[idx]['filename']
                filtered_files.append(filename)

            duration_ms = (time.perf_counter() - start_time) * 1000

            log_retrieval_metrics(
                logger, 
                stage="bm25", 
                count=len(filtered_files), 
                duration_ms=duration_ms, 
                top_score=top_score
            )
            
            return filtered_files
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise

def main() -> None:
    """
    Main function for BM25 index generation or search.
    """
    from utils.config_loader import load_config
    from utils.logger import set_request_context, clear_request_context 
    config = load_config()
    artifacts_dir = config['paths']['bm25_artifacts']
    
    # UPDATED: Set request context for standalone execution
    # This ensures logs have a req_id and don't look broken in dev/prod
    req_id = set_request_context(conversation_id="cli-test-session")
    
    try:
        logger.info(f"=== BM25 Search CLI (Req ID: {req_id}) ===")
        
        # Load and search
        searcher = BM25Searcher(artifacts_dir=artifacts_dir)
        searcher.load_bm25_artifacts()
        
        # Example queries
        test_queries = [
            'What is MALDI?',
            'linear mixed effect models',
            'causal inference',
            ' '
        ]
        
        for query in test_queries:
            try:
                results = searcher.search(query, k=10)
                print(f"Query: '{query}' returned {len(results)} results")
            except Exception as e:
                print(f"Query '{query}' failed: {e}")
        
        logger.info("=== Search Complete ===")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        # UPDATED: Clean up context
        clear_request_context()

if __name__ == '__main__':
    main()