"""
JSONL Session Logger for RAG Pipeline
=======================================

Comprehensive logging format that captures all stages of the RAG pipeline for:
1. Session replay / chat history reconstruction
2. Debugging retrieval and ranking quality
3. Analyzing stage-by-stage performance

Each query-response pair is logged as a single JSON line with the following structure:
- Session metadata
- Query information
- Stage 1: BM25 + Paper-level embedding (hybrid retrieval) 
- Stage 2: Chunk-level embedding retrieval
- Stage 3: Reranking with score changes
- Stage 4: Generation with final sources
- Timing breakdown
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import uuid

from utils.logger import get_chat_logger

logger = get_chat_logger(__name__)


@dataclass
class BM25Result:
    """BM25 search result for a single document."""
    rank: int
    filename: str
    score: float  # BM25 score from the searcher


@dataclass  
class EmbeddingResult:
    """Embedding search result for a single document/chunk."""
    rank: int
    file_path: str
    distance: float  # Cosine/L2 distance from ChromaDB
    paper_title: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    # For chunk-level results
    text_preview: Optional[str] = None  # First N chars for debugging


@dataclass
class HybridScoreBreakdown:
    """Detailed hybrid score breakdown for a single paper."""
    file_path: str
    bm25_rank: Optional[int]  # None if not in BM25 results
    bm25_raw_score: Optional[float]  # Normalized BM25 score [0,1]
    embedding_rank: Optional[int]  # None if not in embedding results
    embedding_distance: Optional[float]  # Raw distance
    embedding_normalized_score: Optional[float]  # Normalized [0,1]
    combined_score: float  # Final weighted score
    selected: bool  # Whether this paper was selected for Stage 2


@dataclass
class RerankerResult:
    """Reranker result with before/after comparison."""
    final_rank: int
    rerank_score: float
    # Original retrieval info
    original_rank: int  # Rank from Stage 2
    original_distance: float  # Distance from Stage 2
    rank_change: int  # original_rank - final_rank (positive = improved)
    # Document info
    file_path: str
    paper_title: Optional[str]
    chunk_id: Optional[str]
    chunk_index: Optional[int]
    # Content
    text: str
    text_length: int
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalSource:
    """Final source used in generation."""
    rank: int
    paper_title: str
    file_path: str
    chunk_id: Optional[str]
    rerank_score: float
    text_preview: str  # Truncated for readability


@dataclass
class StageTimings:
    """Timing breakdown for each stage."""
    bm25_seconds: float = 0.0
    embedding_stage1_seconds: float = 0.0  # Paper-level
    embedding_stage2_seconds: float = 0.0  # Chunk-level
    reranker_seconds: float = 0.0
    generator_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class ClaimVerificationLog:
    """Log entry for a single claim verification."""
    claim: str
    claim_index: int
    is_grounded: bool
    supporting_doc_indices: List[int]
    max_score: float
    doc_scores: List[Dict[str, Any]]  # [{doc_index, label, probability}, ...]


@dataclass
class HallucinationLog:
    """Log entry for hallucination check results."""
    claims: List[str]
    verifications: List[Dict]  # Serialized ClaimVerificationLog
    num_claims: int
    num_grounded: int
    num_unsupported: int
    grounding_ratio: float
    unsupported_claims: List[str]


@dataclass
class SessionEntry:
    """Complete entry for a single query-response in the session."""
    # Session metadata
    session_id: str
    entry_id: str
    timestamp: str
    
    # Query
    query: str
    
    # Pipeline configuration
    config: Dict[str, Any]
    
    # Stage 1: Hybrid Retrieval
    stage1_bm25_results: List[Dict]  # Serialized BM25Result
    stage1_embedding_results: List[Dict]  # Serialized EmbeddingResult
    stage1_hybrid_scores: List[Dict]  # Serialized HybridScoreBreakdown
    stage1_selected_papers: List[str]  # Final paper paths
    
    # Stage 2: Chunk Retrieval
    stage2_chunk_results: List[Dict]  # Serialized EmbeddingResult with chunks
    
    # Stage 3: Reranking
    stage3_reranked_results: List[Dict]  # Serialized RerankerResult
    
    # Stage 4: Generation
    answer: str
    final_sources: List[Dict]  # Serialized FinalSource
    generation_params: Dict[str, Any]
    prompt_length_tokens: Optional[int] = None
    
    # Stage 5: Hallucination Check (optional)
    hallucination_check: Optional[Dict] = None  # Serialized HallucinationLog
    
    # Timings
    timings: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: str = "success"  # "success", "error", "partial"
    error_message: Optional[str] = None


class SessionLogger:
    """
    JSONL-based session logger for RAG pipeline.
    
    Each session is a .jsonl file where each line is a complete query-response entry.
    This format supports:
    - Streaming writes (append mode)
    - Easy parsing and filtering
    - Hybrid use: chat replay AND debugging
    """
    
    def __init__(
        self, 
        session_dir: Path,
        session_id: Optional[str] = None,
        text_preview_length: int = 500
    ):
        """
        Initialize session logger.
        
        Args:
            session_dir: Directory to store session files
            session_id: Optional custom session ID (auto-generated if None)
            text_preview_length: Max chars for text previews in logs
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or self._generate_session_id()
        self.session_file = self.session_dir / f"session_{self.session_id}.jsonl"
        self.text_preview_length = text_preview_length
        
        # Current entry being built
        self._current_entry: Optional[Dict[str, Any]] = None
        self._entry_count = 0
        
        logger.info(f"Session logger initialized: {self.session_file}")
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        return datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to preview length."""
        if len(text) <= self.text_preview_length:
            return text
        return text[:self.text_preview_length] + "..."
    
    def start_entry(self, query: str, config: Dict[str, Any]) -> str:
        """
        Start a new entry for a query.
        
        Args:
            query: The user's query
            config: Pipeline configuration
            
        Returns:
            Entry ID
        """
        self._entry_count += 1
        entry_id = f"{self.session_id}_{self._entry_count:04d}"
        
        self._current_entry = {
            'session_id': self.session_id,
            'entry_id': entry_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'config': config,
            'stage1_bm25_results': [],
            'stage1_embedding_results': [],
            'stage1_hybrid_scores': [],
            'stage1_selected_papers': [],
            'stage2_chunk_results': [],
            'stage3_reranked_results': [],
            'answer': '',
            'final_sources': [],
            'generation_params': {},
            'prompt_length_tokens': None,
            'hallucination_check': None,
            'timings': {},
            'status': 'in_progress',
            'error_message': None
        }
        
        logger.debug(f"Started entry {entry_id}")
        return entry_id
    
    def log_bm25_results(
        self,
        results: List[str],
        scores: Optional[List[float]] = None
    ):
        """
        Log BM25 search results.
        
        Args:
            results: List of filenames from BM25 search
            scores: Optional list of BM25 scores (if available)
        """
        if self._current_entry is None:
            logger.warning("No active entry for BM25 results")
            return
        
        bm25_entries = []
        for rank, filename in enumerate(results, 1):
            entry = {
                'rank': rank,
                'filename': filename,
                'score': scores[rank - 1] if scores else None
            }
            bm25_entries.append(entry)
        
        self._current_entry['stage1_bm25_results'] = bm25_entries
        logger.debug(f"Logged {len(bm25_entries)} BM25 results")
    
    def log_embedding_stage1_results(
        self,
        results: List[Tuple[float, Dict, str]]
    ):
        """
        Log paper-level embedding search results.
        
        Args:
            results: List of (distance, metadata, text) tuples from embedding search
        """
        if self._current_entry is None:
            logger.warning("No active entry for Stage 1 embedding results")
            return
        
        emb_entries = []
        for rank, (distance, metadata, text) in enumerate(results, 1):
            entry = {
                'rank': rank,
                'file_path': metadata.get('file_path', ''),
                'distance': float(distance),
                'paper_title': metadata.get('paper_title'),
                'text_preview': self._truncate_text(text) if text else None
            }
            emb_entries.append(entry)
        
        self._current_entry['stage1_embedding_results'] = emb_entries
        logger.debug(f"Logged {len(emb_entries)} Stage 1 embedding results")
    
    def log_hybrid_scores(
        self,
        file_scores: Dict[str, float],
        bm25_results: List[str],
        emb_results: List[Tuple[float, Dict, str]],
        selected_files: Set[str],
        bm25_weight: float,
        embedding_weight: float,
        k: int
    ):
        """
        Log detailed hybrid score breakdown.
        
        Args:
            file_scores: Dict of {file_path: combined_score}
            bm25_results: List of filenames from BM25
            emb_results: List of (distance, metadata, text) from embeddings
            selected_files: Set of selected file paths
            bm25_weight: Weight for BM25 scores
            embedding_weight: Weight for embedding scores  
            k: Number of papers retrieved per method
        """
        if self._current_entry is None:
            logger.warning("No active entry for hybrid scores")
            return
        
        # Build lookup structures
        bm25_ranks = {f: i for i, f in enumerate(bm25_results, 1)}
        emb_info = {}
        for rank, (dist, meta, _) in enumerate(emb_results, 1):
            fp = meta.get('file_path', '')
            emb_info[fp] = {'rank': rank, 'distance': float(dist)}
        
        hybrid_entries = []
        for file_path, combined_score in file_scores.items():
            # BM25 info
            bm25_rank = bm25_ranks.get(file_path)
            bm25_norm = (k * 2 - (bm25_rank - 1)) / (k * 2) if bm25_rank else 0
            
            # Embedding info
            emb_data = emb_info.get(file_path, {})
            emb_rank = emb_data.get('rank')
            emb_dist = emb_data.get('distance')
            emb_norm = (k * 2 - (emb_rank - 1)) / (k * 2) if emb_rank else 0
            
            entry = {
                'file_path': file_path,
                'bm25_rank': bm25_rank,
                'bm25_normalized_score': round(bm25_norm * bm25_weight, 4) if bm25_rank else None,
                'embedding_rank': emb_rank,
                'embedding_distance': emb_dist,
                'embedding_normalized_score': round(emb_norm * embedding_weight, 4) if emb_rank else None,
                'combined_score': round(combined_score, 4),
                'selected': file_path in selected_files
            }
            hybrid_entries.append(entry)
        
        # Sort by combined score descending
        hybrid_entries.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self._current_entry['stage1_hybrid_scores'] = hybrid_entries
        self._current_entry['stage1_selected_papers'] = list(selected_files)
        logger.debug(f"Logged {len(hybrid_entries)} hybrid scores, {len(selected_files)} selected")
    
    def log_chunk_results(
        self,
        results: List[Tuple[float, Dict, str]]
    ):
        """
        Log chunk-level embedding search results (Stage 2).
        
        Args:
            results: List of (distance, metadata, text) tuples
        """
        if self._current_entry is None:
            logger.warning("No active entry for chunk results")
            return
        
        chunk_entries = []
        for rank, (distance, metadata, text) in enumerate(results, 1):
            entry = {
                'rank': rank,
                'file_path': metadata.get('file_path', ''),
                'distance': float(distance),
                'paper_title': metadata.get('paper_title'),
                'chunk_id': metadata.get('chunk_id'),
                'chunk_index': metadata.get('chunk_index'),
                'text_preview': self._truncate_text(text) if text else None,
                'text_length': len(text) if text else 0,
                'metadata': {k: v for k, v in metadata.items() 
                            if k not in ['file_path', 'paper_title', 'chunk_id', 'chunk_index']}
            }
            chunk_entries.append(entry)
        
        self._current_entry['stage2_chunk_results'] = chunk_entries
        logger.debug(f"Logged {len(chunk_entries)} chunk results")
    
    def log_reranker_results(
        self,
        reranked: List[Dict],
        original_chunks: List[Tuple[float, Dict, str]]
    ):
        """
        Log reranker results with before/after comparison.
        
        Args:
            reranked: List of dicts from reranker.rerank_with_details()
            original_chunks: Original chunks from Stage 2 for comparison
        """
        if self._current_entry is None:
            logger.warning("No active entry for reranker results")
            return
        
        # Build original rank lookup
        original_ranks = {}
        for rank, (dist, meta, _) in enumerate(original_chunks, 1):
            key = (meta.get('file_path', ''), meta.get('chunk_index', 0))
            original_ranks[key] = {'rank': rank, 'distance': float(dist)}
        
        rerank_entries = []
        for item in reranked:
            meta = item.get('metadata', {})
            key = (meta.get('file_path', ''), meta.get('chunk_index', 0))
            orig_info = original_ranks.get(key, {})
            
            orig_rank = item.get('original_rank') or orig_info.get('rank', 0)
            final_rank = item.get('rank', 0)
            
            entry = {
                'final_rank': final_rank,
                'rerank_score': round(item.get('rerank_score', 0), 6),
                'original_rank': orig_rank,
                'original_distance': item.get('original_distance') or orig_info.get('distance'),
                'rank_change': orig_rank - final_rank if orig_rank else None,
                'file_path': meta.get('file_path', ''),
                'paper_title': meta.get('paper_title'),
                'chunk_id': meta.get('chunk_id'),
                'chunk_index': meta.get('chunk_index'),
                'text': item.get('text', ''),
                'text_length': len(item.get('text', '')),
                'metadata': {k: v for k, v in meta.items() 
                            if k not in ['file_path', 'paper_title', 'chunk_id', 'chunk_index']}
            }
            rerank_entries.append(entry)
        
        self._current_entry['stage3_reranked_results'] = rerank_entries
        logger.debug(f"Logged {len(rerank_entries)} reranked results")
    
    def log_generation(
        self,
        answer: str,
        contexts: List[Dict],
        generation_params: Dict[str, Any],
        prompt_length_tokens: Optional[int] = None
    ):
        """
        Log generation results and final sources.
        
        Args:
            answer: Generated answer text
            contexts: Contexts used for generation (from reranker)
            generation_params: Generation parameters used
            prompt_length_tokens: Prompt length in tokens
        """
        if self._current_entry is None:
            logger.warning("No active entry for generation results")
            return
        
        self._current_entry['answer'] = answer
        self._current_entry['generation_params'] = generation_params
        self._current_entry['prompt_length_tokens'] = prompt_length_tokens
        
        # Build final sources list
        final_sources = []
        for ctx in contexts:
            meta = ctx.get('metadata', {})
            source = {
                'rank': ctx.get('rank', 0),
                'paper_title': meta.get('paper_title', 'Unknown'),
                'file_path': meta.get('file_path', ''),
                'chunk_id': meta.get('chunk_id'),
                'rerank_score': round(ctx.get('rerank_score', 0), 6),
                'text_preview': self._truncate_text(ctx.get('text', ''))
            }
            final_sources.append(source)
        
        self._current_entry['final_sources'] = final_sources
        logger.debug(f"Logged generation with {len(final_sources)} sources")
    
    def log_timings(self, timings: Dict[str, float]):
        """
        Log timing breakdown.
        
        Args:
            timings: Dict of {stage_name: seconds}
        """
        if self._current_entry is None:
            logger.warning("No active entry for timings")
            return
        
        self._current_entry['timings'] = {
            'bm25_seconds': round(timings.get('bm25', 0), 4),
            'embedding_seconds': round(timings.get('embedding', 0), 4),
            'reranker_seconds': round(timings.get('reranker', 0), 4),
            'generator_seconds': round(timings.get('generator', 0), 4),
            'hallucination_check_seconds': round(timings.get('hallucination_check', 0), 4),
            'total_seconds': round(sum(timings.values()), 4)
        }
        logger.debug(f"Logged timings: {self._current_entry['timings']}")
    
    def log_hallucination_check(self, hallucination_result: Any):
        """
        Log hallucination check results.
        
        Args:
            hallucination_result: HallucinationResult from hallucination_checker
        """
        if self._current_entry is None:
            logger.warning("No active entry for hallucination check")
            return
        
        # Convert to dict if it has a to_dict method, otherwise use as-is
        if hasattr(hallucination_result, 'to_dict'):
            result_dict = hallucination_result.to_dict()
        elif isinstance(hallucination_result, dict):
            result_dict = hallucination_result
        else:
            logger.warning(f"Unknown hallucination result type: {type(hallucination_result)}")
            return
        
        # Store the full result
        self._current_entry['hallucination_check'] = {
            'claims': result_dict.get('claims', []),
            'verifications': result_dict.get('verifications', []),
            'num_claims': result_dict.get('num_claims', 0),
            'num_grounded': result_dict.get('num_grounded', 0),
            'num_unsupported': result_dict.get('num_unsupported', 0),
            'grounding_ratio': round(result_dict.get('grounding_ratio', 0), 4),
            'unsupported_claims': result_dict.get('unsupported_claims', [])
        }
        
        logger.debug(
            f"Logged hallucination check: {result_dict.get('num_grounded', 0)}/"
            f"{result_dict.get('num_claims', 0)} claims grounded"
        )

    def log_error(self, error_message: str):
        """Log an error for the current entry."""
        if self._current_entry is None:
            logger.warning("No active entry for error")
            return
        
        self._current_entry['status'] = 'error'
        self._current_entry['error_message'] = error_message
        logger.debug(f"Logged error: {error_message}")
    
    def finish_entry(self) -> Optional[str]:
        """
        Finish the current entry and write to JSONL file.
        
        Returns:
            Entry ID if successful, None otherwise
        """
        if self._current_entry is None:
            logger.warning("No active entry to finish")
            return None
        
        # Set final status
        if self._current_entry['status'] == 'in_progress':
            self._current_entry['status'] = 'success'
        
        entry_id = self._current_entry['entry_id']
        
        try:
            # Write as single JSON line
            with open(self.session_file, 'a', encoding='utf-8') as f:
                json.dump(self._current_entry, f, ensure_ascii=False, default=str)
                f.write('\n')
            
            logger.info(f"Entry {entry_id} written to {self.session_file}")
            
        except Exception as e:
            logger.error(f"Failed to write entry {entry_id}: {e}")
            return None
        finally:
            self._current_entry = None
        
        return entry_id
    
    @contextmanager
    def entry_context(self, query: str, config: Dict[str, Any]):
        """
        Context manager for entry lifecycle.
        
        Usage:
            with session_logger.entry_context(query, config):
                session_logger.log_bm25_results(...)
                ...
        """
        entry_id = self.start_entry(query, config)
        try:
            yield entry_id
        except Exception as e:
            self.log_error(str(e))
            raise
        finally:
            self.finish_entry()
    
    @classmethod
    def load_session(cls, session_file: Path) -> List[Dict]:
        """
        Load all entries from a session file.
        
        Args:
            session_file: Path to .jsonl session file
            
        Returns:
            List of entry dictionaries
        """
        entries = []
        with open(session_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    
    @classmethod
    def load_for_chat_replay(cls, session_file: Path) -> List[Dict[str, str]]:
        """
        Load session in simplified format for chat replay.
        
        Args:
            session_file: Path to .jsonl session file
            
        Returns:
            List of {query, answer, timestamp, sources} dicts
        """
        entries = cls.load_session(session_file)
        chat_history = []
        
        for entry in entries:
            if entry.get('status') == 'success':
                sources = [s['paper_title'] for s in entry.get('final_sources', [])]
                chat_history.append({
                    'query': entry['query'],
                    'answer': entry['answer'],
                    'timestamp': entry['timestamp'],
                    'sources': sources
                })
        
        return chat_history
    
    @classmethod
    def load_for_debugging(cls, session_file: Path, entry_id: Optional[str] = None) -> Dict:
        """
        Load session with full debugging information.
        
        Args:
            session_file: Path to .jsonl session file
            entry_id: Optional specific entry ID to load
            
        Returns:
            Full debugging dict or specific entry
        """
        entries = cls.load_session(session_file)
        
        if entry_id:
            for entry in entries:
                if entry.get('entry_id') == entry_id:
                    return entry
            return {}
        
        return {
            'session_file': str(session_file),
            'total_entries': len(entries),
            'successful': sum(1 for e in entries if e.get('status') == 'success'),
            'errors': sum(1 for e in entries if e.get('status') == 'error'),
            'entries': entries
        }


# Utility functions for analysis

def analyze_reranker_impact(entry: Dict) -> Dict:
    """
    Analyze the impact of reranking on retrieval quality.
    
    Args:
        entry: Single session entry
        
    Returns:
        Analysis dict with metrics
    """
    reranked = entry.get('stage3_reranked_results', [])
    
    if not reranked:
        return {'error': 'No reranked results'}
    
    rank_changes = [r.get('rank_change', 0) for r in reranked if r.get('rank_change') is not None]
    
    return {
        'num_results': len(reranked),
        'avg_rank_change': sum(rank_changes) / len(rank_changes) if rank_changes else 0,
        'max_improvement': max(rank_changes) if rank_changes else 0,
        'max_degradation': min(rank_changes) if rank_changes else 0,
        'results_improved': sum(1 for r in rank_changes if r > 0),
        'results_degraded': sum(1 for r in rank_changes if r < 0),
        'results_unchanged': sum(1 for r in rank_changes if r == 0),
        'top_score': reranked[0].get('rerank_score') if reranked else None,
        'score_spread': reranked[0].get('rerank_score', 0) - reranked[-1].get('rerank_score', 0) if len(reranked) > 1 else 0
    }


def analyze_hybrid_retrieval(entry: Dict) -> Dict:
    """
    Analyze hybrid retrieval effectiveness.
    
    Args:
        entry: Single session entry
        
    Returns:
        Analysis dict
    """
    hybrid = entry.get('stage1_hybrid_scores', [])
    selected = set(entry.get('stage1_selected_papers', []))
    
    if not hybrid:
        return {'error': 'No hybrid scores'}
    
    # Count papers from each source
    bm25_only = sum(1 for h in hybrid if h.get('bm25_rank') and not h.get('embedding_rank'))
    emb_only = sum(1 for h in hybrid if h.get('embedding_rank') and not h.get('bm25_rank'))
    both = sum(1 for h in hybrid if h.get('bm25_rank') and h.get('embedding_rank'))
    
    # Selected from each source
    selected_bm25_only = sum(1 for h in hybrid if h.get('selected') and h.get('bm25_rank') and not h.get('embedding_rank'))
    selected_emb_only = sum(1 for h in hybrid if h.get('selected') and h.get('embedding_rank') and not h.get('bm25_rank'))
    selected_both = sum(1 for h in hybrid if h.get('selected') and h.get('bm25_rank') and h.get('embedding_rank'))
    
    return {
        'total_candidates': len(hybrid),
        'selected_count': len(selected),
        'bm25_only_candidates': bm25_only,
        'embedding_only_candidates': emb_only,
        'both_candidates': both,
        'selected_from_bm25_only': selected_bm25_only,
        'selected_from_embedding_only': selected_emb_only,
        'selected_from_both': selected_both,
        'hybrid_benefit': selected_bm25_only + selected_emb_only  # Papers only found by one method
    }


def get_session_summary(session_file: Path) -> Dict:
    """
    Get summary statistics for a session.
    
    Args:
        session_file: Path to .jsonl session file
        
    Returns:
        Summary dict
    """
    entries = SessionLogger.load_session(session_file)
    
    if not entries:
        return {'error': 'Empty session'}
    
    timings = [e.get('timings', {}).get('total_seconds', 0) for e in entries]
    
    return {
        'session_file': str(session_file),
        'total_queries': len(entries),
        'successful': sum(1 for e in entries if e.get('status') == 'success'),
        'errors': sum(1 for e in entries if e.get('status') == 'error'),
        'avg_time_seconds': sum(timings) / len(timings) if timings else 0,
        'min_time_seconds': min(timings) if timings else 0,
        'max_time_seconds': max(timings) if timings else 0,
        'first_query_time': entries[0].get('timestamp') if entries else None,
        'last_query_time': entries[-1].get('timestamp') if entries else None
    }