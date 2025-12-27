"""
Request Tracer for RAG Pipeline
================================

Structured data capture for all pipeline stages. Unlike SessionLogger (which focuses on
human-readable JSONL for replay), RequestTracer captures complete intermediate outputs
in machine-readable format for:

1. Debugging: Full visibility into each stage's inputs/outputs
2. Analysis: Performance profiling, retrieval quality metrics
3. Caching: Potential reuse of intermediate results
4. Evaluation: Ground truth comparison, A/B testing
5. Export: Training data generation, pipeline optimization

Architecture:
- RequestTrace: Immutable snapshot of a single request's complete execution
- StageData: Per-stage input/output/timing/metadata container
- RequestTracer: Manager that builds traces and provides export/query capabilities

Thread Safety:
- Uses contextvars for per-request isolation (same pattern as logger.py)
- Immutable dataclasses for trace data prevent mutation bugs

Usage:
    from inference.request_tracer import RequestTracer, get_current_tracer

    # At request start
    tracer = RequestTracer()
    tracer.start_request(query="What is X?", config={...})

    # In each pipeline stage
    tracer.capture_bm25(results=[...], scores=[...], duration_ms=45.2)
    tracer.capture_embedding_stage1(results=[...], duration_ms=120.5)
    # ... etc

    # At request end
    trace = tracer.finish_request()

    # Export
    trace.to_dict()  # Full JSON-serializable dict
    trace.to_json()  # JSON string
    tracer.save(trace, "/path/to/traces")  # Save to disk
"""

from __future__ import annotations

import json
import uuid
import time
import gzip
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    TypedDict, Literal, Union
)
from dataclasses import dataclass, field, asdict
from contextvars import ContextVar
from enum import Enum
import threading

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class StageStatus(str, Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Stage Data Classes (Immutable snapshots)
# =============================================================================

@dataclass(frozen=True)
class BM25Result:
    """Single BM25 result."""
    rank: int
    filename: str
    score: Optional[float] = None


@dataclass(frozen=True)
class EmbeddingResult:
    """Single embedding search result."""
    rank: int
    file_path: str
    distance: float
    paper_title: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    text: Optional[str] = None
    text_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HybridScore:
    """Hybrid retrieval score breakdown."""
    file_path: str
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    embedding_rank: Optional[int]
    embedding_distance: Optional[float]
    combined_score: float
    selected: bool


@dataclass(frozen=True)
class RerankerResult:
    """Single reranker result with rank comparison."""
    final_rank: int
    rerank_score: float
    original_rank: int
    original_distance: float
    rank_change: int  # positive = improved
    file_path: str
    paper_title: Optional[str]
    chunk_id: Optional[str]
    chunk_index: Optional[int]
    text: str
    text_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimVerification:
    """Single claim verification result."""
    claim_index: int
    claim: str
    is_grounded: bool
    supporting_doc_indices: Tuple[int, ...]  # Tuple for immutability
    max_score: float
    doc_scores: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class HallucinationCheckResult:
    """Complete hallucination check output."""
    claims: Tuple[str, ...]
    verifications: Tuple[ClaimVerification, ...]
    num_claims: int
    num_grounded: int
    num_unsupported: int
    grounding_ratio: float
    unsupported_claims: Tuple[str, ...]


# =============================================================================
# Stage Data Container
# =============================================================================

@dataclass
class StageData:
    """
    Container for a single stage's execution data.
    Mutable during capture, frozen when trace is finalized.
    """
    name: str
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Input parameters
    input_params: Dict[str, Any] = field(default_factory=dict)
    
    # Output data (stage-specific)
    output_count: int = 0
    output_data: Any = None  # Will be stage-specific dataclass or list
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    def start(self):
        """Mark stage as started."""
        self.status = StageStatus.RUNNING
        self.start_time = time.perf_counter()
    
    def finish(self, success: bool = True, error: Optional[Exception] = None):
        """Mark stage as finished."""
        self.end_time = time.perf_counter()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if error:
            self.status = StageStatus.FAILED
            self.error = str(error)
            self.error_type = type(error).__name__
        elif success:
            self.status = StageStatus.SUCCESS
        else:
            self.status = StageStatus.FAILED
    
    def skip(self, reason: str = ""):
        """Mark stage as skipped."""
        self.status = StageStatus.SKIPPED
        self.metrics['skip_reason'] = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            'name': self.name,
            'status': self.status.value,
            'duration_ms': self.duration_ms,
            'input_params': self.input_params,
            'output_count': self.output_count,
            'metrics': self.metrics,
        }
        
        if self.error:
            result['error'] = self.error
            result['error_type'] = self.error_type
        
        # Convert output_data based on type
        if self.output_data is not None:
            if isinstance(self.output_data, (list, tuple)):
                result['output_data'] = [
                    asdict(item) if hasattr(item, '__dataclass_fields__') else item
                    for item in self.output_data
                ]
            elif hasattr(self.output_data, '__dataclass_fields__'):
                result['output_data'] = asdict(self.output_data)
            else:
                result['output_data'] = self.output_data
        
        return result


# =============================================================================
# Request Trace (Complete execution snapshot)
# =============================================================================

@dataclass
class RequestTrace:
    """
    Complete trace of a single request's execution.
    Immutable after finalization.
    """
    # Identity
    trace_id: str
    request_id: str
    conversation_id: Optional[str]
    
    # Timing
    timestamp: str  # ISO format
    total_duration_ms: Optional[float] = None
    
    # Query
    query: str = ""
    query_hash: str = ""  # For deduplication/caching
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Stage data
    stages: Dict[str, StageData] = field(default_factory=dict)
    
    # Final outputs
    final_answer: str = ""
    final_sources: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status
    status: str = "in_progress"  # in_progress, success, error, partial
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire trace to JSON-serializable dict."""
        return {
            'trace_id': self.trace_id,
            'request_id': self.request_id,
            'conversation_id': self.conversation_id,
            'timestamp': self.timestamp,
            'total_duration_ms': self.total_duration_ms,
            'query': self.query,
            'query_hash': self.query_hash,
            'config': self.config,
            'stages': {name: stage.to_dict() for name, stage in self.stages.items()},
            'final_answer': self.final_answer,
            'final_sources': self.final_sources,
            'status': self.status,
            'error_message': self.error_message,
            'metadata': self.metadata,
        }
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_stage(self, name: str) -> Optional[StageData]:
        """Get stage data by name."""
        return self.stages.get(name)
    
    def get_timing_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown for all stages."""
        return {
            name: stage.duration_ms or 0.0
            for name, stage in self.stages.items()
            if stage.duration_ms is not None
        }
    
    def get_retrieval_funnel(self) -> Dict[str, int]:
        """Get document counts through the retrieval funnel."""
        return {
            name: stage.output_count
            for name, stage in self.stages.items()
        }


# =============================================================================
# Request Tracer (Manager class)
# =============================================================================

# Context variable for current tracer (async-safe)
_current_tracer: ContextVar[Optional['RequestTracer']] = ContextVar(
    'current_tracer', default=None
)


def get_current_tracer() -> Optional['RequestTracer']:
    """Get the current request's tracer."""
    return _current_tracer.get()


def set_current_tracer(tracer: Optional['RequestTracer']) -> None:
    """Set the current request's tracer."""
    _current_tracer.set(tracer)


class RequestTracer:
    """
    Manager for capturing request execution data.
    
    One tracer per request. Builds a RequestTrace incrementally
    as stages complete, then finalizes to an immutable snapshot.
    """
    
    # Stage names (canonical)
    STAGE_BM25 = "bm25"
    STAGE_EMBEDDING_PAPER = "embedding_paper"
    STAGE_HYBRID = "hybrid"
    STAGE_EMBEDDING_CHUNK = "embedding_chunk"
    STAGE_RERANKER = "reranker"
    STAGE_GENERATOR = "generator"
    STAGE_HALLUCINATION = "hallucination"
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        auto_register: bool = True
    ):
        """
        Initialize tracer for a request.
        
        Args:
            request_id: Unique request ID (auto-generated if None)
            conversation_id: Conversation/session ID
            auto_register: If True, register as current tracer in contextvars
        """
        self.request_id = request_id or str(uuid.uuid4())[:12]
        self.conversation_id = conversation_id
        
        self._trace: Optional[RequestTrace] = None
        self._start_time: Optional[float] = None
        self._finalized = False
        self._lock = threading.Lock()
        
        if auto_register:
            set_current_tracer(self)
        
        logger.debug(f"RequestTracer initialized: {self.request_id}")
    
    def start_request(
        self,
        query: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracing a request.
        
        Args:
            query: User query
            config: Pipeline configuration
            metadata: Additional metadata
        
        Returns:
            Trace ID
        """
        with self._lock:
            if self._trace is not None:
                logger.warning("Tracer already has active trace, overwriting")
            
            trace_id = f"{self.request_id}_{datetime.now().strftime('%H%M%S%f')}"
            
            self._trace = RequestTrace(
                trace_id=trace_id,
                request_id=self.request_id,
                conversation_id=self.conversation_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                query=query,
                query_hash=hashlib.md5(query.encode()).hexdigest()[:8],
                config=config.copy(),
                metadata=metadata or {},
            )
            
            self._start_time = time.perf_counter()
            self._finalized = False
            
            logger.debug(f"Request trace started: {trace_id}")
            return trace_id
    
    def _ensure_stage(self, name: str) -> StageData:
        """Get or create stage data."""
        if self._trace is None:
            raise RuntimeError("No active trace. Call start_request() first.")
        
        if name not in self._trace.stages:
            self._trace.stages[name] = StageData(name=name)
        
        return self._trace.stages[name]
    
    # =========================================================================
    # Stage 1: BM25
    # =========================================================================
    
    def capture_bm25(
        self,
        results: List[str],
        scores: Optional[List[float]] = None,
        duration_ms: Optional[float] = None,
        k: Optional[int] = None,
        tokenized_query: Optional[List[str]] = None
    ) -> None:
        """
        Capture BM25 search results.
        
        Args:
            results: List of filenames returned
            scores: Optional BM25 scores per result
            duration_ms: Stage duration
            k: Requested number of results
            tokenized_query: Preprocessed query tokens
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_BM25)
            
            stage.input_params = {
                'k': k,
                'tokenized_query': tokenized_query,
            }
            
            # Build structured results
            bm25_results = []
            for rank, filename in enumerate(results, 1):
                score = scores[rank - 1] if scores and len(scores) >= rank else None
                bm25_results.append(BM25Result(
                    rank=rank,
                    filename=filename,
                    score=score
                ))
            
            stage.output_data = tuple(bm25_results)
            stage.output_count = len(bm25_results)
            stage.duration_ms = duration_ms
            stage.status = StageStatus.SUCCESS
            
            # Metrics
            if scores:
                stage.metrics['top_score'] = max(scores) if scores else None
                stage.metrics['score_range'] = (min(scores), max(scores)) if scores else None
    
    # =========================================================================
    # Stage 1b: Paper-level Embedding
    # =========================================================================
    
    def capture_embedding_paper(
        self,
        results: List[Tuple[float, Dict, str]],
        duration_ms: Optional[float] = None,
        k: Optional[int] = None
    ) -> None:
        """
        Capture paper-level embedding search results.
        
        Args:
            results: List of (distance, metadata, text) tuples
            duration_ms: Stage duration
            k: Requested number of results
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_EMBEDDING_PAPER)
            
            stage.input_params = {'k': k}
            
            emb_results = []
            for rank, (distance, metadata, text) in enumerate(results, 1):
                emb_results.append(EmbeddingResult(
                    rank=rank,
                    file_path=metadata.get('file_path', ''),
                    distance=float(distance),
                    paper_title=metadata.get('paper_title'),
                    text=text[:500] if text else None,  # Truncate for storage
                    text_length=len(text) if text else 0,
                    metadata=dict(metadata),
                ))
            
            stage.output_data = tuple(emb_results)
            stage.output_count = len(emb_results)
            stage.duration_ms = duration_ms
            stage.status = StageStatus.SUCCESS
            
            if emb_results:
                stage.metrics['top_distance'] = emb_results[0].distance
    
    # =========================================================================
    # Stage 1c: Hybrid Combination
    # =========================================================================
    
    def capture_hybrid(
        self,
        file_scores: Dict[str, float],
        bm25_results: List[str],
        emb_results: List[Tuple[float, Dict, str]],
        selected_files: Set[str],
        bm25_weight: float,
        embedding_weight: float,
        k: int
    ) -> None:
        """
        Capture hybrid retrieval combination results.
        
        Args:
            file_scores: Combined scores per file
            bm25_results: BM25 result filenames
            emb_results: Embedding results
            selected_files: Final selected files
            bm25_weight: BM25 weight used
            embedding_weight: Embedding weight used
            k: Number of papers to select
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_HYBRID)
            
            stage.input_params = {
                'bm25_weight': bm25_weight,
                'embedding_weight': embedding_weight,
                'k': k,
                'bm25_count': len(bm25_results),
                'embedding_count': len(emb_results),
            }
            
            # Build lookup for embedding distances
            emb_lookup = {}
            for rank, (dist, meta, _) in enumerate(emb_results, 1):
                fp = meta.get('file_path', '')
                emb_lookup[fp] = {'rank': rank, 'distance': float(dist)}
            
            # Build hybrid scores
            hybrid_scores = []
            for file_path, combined_score in file_scores.items():
                bm25_rank = bm25_results.index(file_path) + 1 if file_path in bm25_results else None
                bm25_score = (k * 2 - (bm25_rank - 1)) / (k * 2) * bm25_weight if bm25_rank else None
                
                emb_info = emb_lookup.get(file_path, {})
                emb_rank = emb_info.get('rank')
                emb_dist = emb_info.get('distance')
                
                hybrid_scores.append(HybridScore(
                    file_path=file_path,
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                    embedding_rank=emb_rank,
                    embedding_distance=emb_dist,
                    combined_score=combined_score,
                    selected=file_path in selected_files
                ))
            
            # Sort by combined score
            hybrid_scores.sort(key=lambda x: x.combined_score, reverse=True)
            
            stage.output_data = tuple(hybrid_scores)
            stage.output_count = len(selected_files)
            stage.status = StageStatus.SUCCESS
            
            # Metrics
            bm25_only = sum(1 for h in hybrid_scores if h.bm25_rank and not h.embedding_rank)
            emb_only = sum(1 for h in hybrid_scores if h.embedding_rank and not h.bm25_rank)
            both = sum(1 for h in hybrid_scores if h.bm25_rank and h.embedding_rank)
            
            stage.metrics = {
                'total_candidates': len(hybrid_scores),
                'selected_count': len(selected_files),
                'bm25_only': bm25_only,
                'embedding_only': emb_only,
                'overlap': both,
                'unique_from_hybrid': bm25_only + emb_only,
            }
    
    # =========================================================================
    # Stage 2: Chunk-level Embedding
    # =========================================================================
    
    def capture_embedding_chunk(
        self,
        results: List[Tuple[float, Dict, str]],
        duration_ms: Optional[float] = None,
        m: Optional[int] = None,
        file_filter: Optional[Set[str]] = None
    ) -> None:
        """
        Capture chunk-level embedding search results.
        
        Args:
            results: List of (distance, metadata, text) tuples
            duration_ms: Stage duration
            m: Requested number of chunks
            file_filter: Files searched within
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_EMBEDDING_CHUNK)
            
            stage.input_params = {
                'm': m,
                'file_filter_count': len(file_filter) if file_filter else None,
            }
            
            chunk_results = []
            for rank, (distance, metadata, text) in enumerate(results, 1):
                chunk_results.append(EmbeddingResult(
                    rank=rank,
                    file_path=metadata.get('file_path', ''),
                    distance=float(distance),
                    paper_title=metadata.get('paper_title'),
                    chunk_id=metadata.get('chunk_id'),
                    chunk_index=metadata.get('chunk_index'),
                    text=text,  # Full text for chunks (needed for reranking)
                    text_length=len(text) if text else 0,
                    metadata={k: v for k, v in metadata.items() 
                              if k not in ['file_path', 'paper_title', 'chunk_id', 'chunk_index']},
                ))
            
            stage.output_data = tuple(chunk_results)
            stage.output_count = len(chunk_results)
            stage.duration_ms = duration_ms
            stage.status = StageStatus.SUCCESS
            
            if chunk_results:
                stage.metrics['top_distance'] = chunk_results[0].distance
                stage.metrics['avg_chunk_length'] = sum(c.text_length for c in chunk_results) / len(chunk_results)
    
    # =========================================================================
    # Stage 3: Reranker
    # =========================================================================
    
    def capture_reranker(
        self,
        reranked: List[Dict],
        original_chunks: List[Tuple[float, Dict, str]],
        duration_ms: Optional[float] = None,
        n: Optional[int] = None,
        method: str = "jina_rerank",
        fallback_triggered: bool = False
    ) -> None:
        """
        Capture reranker results with rank changes.
        
        Args:
            reranked: List of dicts from reranker.rerank_with_details()
            original_chunks: Original chunks for comparison
            duration_ms: Stage duration
            n: Requested number of results
            method: Reranking method used
            fallback_triggered: Whether fallback was used
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_RERANKER)
            
            stage.input_params = {
                'n': n,
                'input_count': len(original_chunks),
            }
            
            # Build original rank lookup
            original_ranks = {}
            for rank, (dist, meta, _) in enumerate(original_chunks, 1):
                key = (meta.get('file_path', ''), meta.get('chunk_index', 0))
                original_ranks[key] = {'rank': rank, 'distance': float(dist)}
            
            reranker_results = []
            for item in reranked:
                meta = item.get('metadata', {})
                key = (meta.get('file_path', ''), meta.get('chunk_index', 0))
                orig_info = original_ranks.get(key, {})
                
                orig_rank = item.get('original_rank') or orig_info.get('rank', 0)
                final_rank = item.get('rank', 0)
                
                reranker_results.append(RerankerResult(
                    final_rank=final_rank,
                    rerank_score=float(item.get('rerank_score', 0)),
                    original_rank=orig_rank,
                    original_distance=float(item.get('original_distance') or orig_info.get('distance', 0)),
                    rank_change=orig_rank - final_rank if orig_rank else 0,
                    file_path=meta.get('file_path', ''),
                    paper_title=meta.get('paper_title'),
                    chunk_id=meta.get('chunk_id'),
                    chunk_index=meta.get('chunk_index'),
                    text=item.get('text', ''),
                    text_length=len(item.get('text', '')),
                    metadata={k: v for k, v in meta.items() 
                              if k not in ['file_path', 'paper_title', 'chunk_id', 'chunk_index']},
                ))
            
            stage.output_data = tuple(reranker_results)
            stage.output_count = len(reranker_results)
            stage.duration_ms = duration_ms
            stage.status = StageStatus.SUCCESS
            
            # Metrics
            rank_changes = [r.rank_change for r in reranker_results if r.original_rank]
            stage.metrics = {
                'method': method,
                'fallback_triggered': fallback_triggered,
                'top_score': reranker_results[0].rerank_score if reranker_results else None,
                'avg_rank_change': sum(rank_changes) / len(rank_changes) if rank_changes else 0,
                'improved': sum(1 for r in rank_changes if r > 0),
                'degraded': sum(1 for r in rank_changes if r < 0),
                'unchanged': sum(1 for r in rank_changes if r == 0),
            }
    
    # =========================================================================
    # Stage 4: Generator
    # =========================================================================
    
    def capture_generator(
        self,
        answer: str,
        contexts: List[Dict],
        duration_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        ttft_ms: Optional[float] = None,
        temperature: Optional[float] = None,
        include_citations: bool = True
    ) -> None:
        """
        Capture generator output.
        
        Args:
            answer: Generated answer text
            contexts: Contexts used for generation
            duration_ms: Total generation time
            prompt_tokens: Input token count
            completion_tokens: Output token count
            ttft_ms: Time to first token
            temperature: Sampling temperature used
            include_citations: Whether citations were requested
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_GENERATOR)
            
            stage.input_params = {
                'context_count': len(contexts),
                'temperature': temperature,
                'include_citations': include_citations,
            }
            
            # Store answer and source mapping
            stage.output_data = {
                'answer': answer,
                'answer_length': len(answer),
                'answer_word_count': len(answer.split()),
                'sources': [
                    {
                        'rank': ctx.get('rank', i + 1),
                        'paper_title': ctx.get('metadata', {}).get('paper_title', 'Unknown'),
                        'file_path': ctx.get('metadata', {}).get('file_path', ''),
                        'rerank_score': ctx.get('rerank_score'),
                    }
                    for i, ctx in enumerate(contexts)
                ],
            }
            
            stage.output_count = 1  # Single answer
            stage.duration_ms = duration_ms
            stage.status = StageStatus.SUCCESS
            
            stage.metrics = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'ttft_ms': ttft_ms,
                'tokens_per_second': completion_tokens / (duration_ms / 1000) if duration_ms and completion_tokens else None,
            }
            
            # Update trace final answer
            if self._trace:
                self._trace.final_answer = answer
                self._trace.final_sources = stage.output_data['sources']
    
    # =========================================================================
    # Stage 5: Hallucination Check
    # =========================================================================
    
    def capture_hallucination(
        self,
        result: Any,  # HallucinationResult from hallucination_checker
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Capture hallucination check results.
        
        Args:
            result: HallucinationResult object or dict
            duration_ms: Stage duration
        """
        with self._lock:
            stage = self._ensure_stage(self.STAGE_HALLUCINATION)
            
            # Convert to dict if needed
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            elif isinstance(result, dict):
                result_dict = result
            else:
                stage.status = StageStatus.FAILED
                stage.error = f"Unknown result type: {type(result)}"
                return
            
            # Build structured verification results
            verifications = []
            for v in result_dict.get('verifications', []):
                if isinstance(v, dict):
                    verifications.append(ClaimVerification(
                        claim_index=v.get('claim_index', 0),
                        claim=v.get('claim', ''),
                        is_grounded=v.get('is_grounded', False),
                        supporting_doc_indices=tuple(v.get('supporting_docs', [])),
                        max_score=float(v.get('max_score', 0)),
                        doc_scores=tuple(v.get('doc_scores', [])),
                    ))
            
            hall_result = HallucinationCheckResult(
                claims=tuple(result_dict.get('claims', [])),
                verifications=tuple(verifications),
                num_claims=result_dict.get('num_claims', 0),
                num_grounded=result_dict.get('num_grounded', 0),
                num_unsupported=result_dict.get('num_unsupported', 0),
                grounding_ratio=float(result_dict.get('grounding_ratio', 0)),
                unsupported_claims=tuple(result_dict.get('unsupported_claims', [])),
            )
            
            stage.output_data = hall_result
            stage.output_count = hall_result.num_claims
            stage.duration_ms = duration_ms
            stage.status = StageStatus.SUCCESS
            
            stage.metrics = {
                'num_claims': hall_result.num_claims,
                'num_grounded': hall_result.num_grounded,
                'num_unsupported': hall_result.num_unsupported,
                'grounding_ratio': hall_result.grounding_ratio,
                'is_fully_grounded': hall_result.num_unsupported == 0,
            }
    
    # =========================================================================
    # Stage Error Capture
    # =========================================================================
    
    def capture_error(self, stage_name: str, error: Exception) -> None:
        """Capture an error for a stage."""
        with self._lock:
            stage = self._ensure_stage(stage_name)
            stage.finish(success=False, error=error)
    
    # =========================================================================
    # Finalization
    # =========================================================================
    
    def finish_request(self, success: bool = True, error: Optional[str] = None) -> RequestTrace:
        """
        Finalize the request trace.
        
        Args:
            success: Whether request succeeded
            error: Error message if failed
        
        Returns:
            Finalized RequestTrace
        """
        with self._lock:
            if self._trace is None:
                raise RuntimeError("No active trace to finish")
            
            if self._finalized:
                logger.warning("Trace already finalized")
                return self._trace
            
            # Calculate total duration
            if self._start_time:
                self._trace.total_duration_ms = (time.perf_counter() - self._start_time) * 1000
            
            # Set status
            if error:
                self._trace.status = "error"
                self._trace.error_message = error
            elif success:
                # Check if all stages succeeded
                all_success = all(
                    s.status in (StageStatus.SUCCESS, StageStatus.SKIPPED)
                    for s in self._trace.stages.values()
                )
                self._trace.status = "success" if all_success else "partial"
            else:
                self._trace.status = "error"
            
            self._finalized = True
            
            logger.debug(f"Request trace finalized: {self._trace.trace_id}, status={self._trace.status}")
            
            return self._trace
    
    def get_trace(self) -> Optional[RequestTrace]:
        """Get current trace (may be in progress)."""
        return self._trace
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def save(
        self,
        trace: RequestTrace,
        output_dir: Union[str, Path],
        compress: bool = False
    ) -> Path:
        """
        Save trace to disk.
        
        Args:
            trace: RequestTrace to save
            output_dir: Directory to save to
            compress: If True, gzip compress the output
        
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        filename = f"trace_{trace.trace_id}.json"
        if compress:
            filename += ".gz"
        
        filepath = output_dir / filename
        
        json_data = trace.to_json(indent=2)
        
        if compress:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_data)
        
        logger.info(f"Trace saved: {filepath}")
        return filepath
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> RequestTrace:
        """
        Load trace from disk.
        
        Args:
            filepath: Path to trace file
        
        Returns:
            RequestTrace object
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Reconstruct RequestTrace from dict
        # Note: This is a shallow reconstruction; stage data remains as dicts
        trace = RequestTrace(
            trace_id=data['trace_id'],
            request_id=data['request_id'],
            conversation_id=data.get('conversation_id'),
            timestamp=data['timestamp'],
            total_duration_ms=data.get('total_duration_ms'),
            query=data.get('query', ''),
            query_hash=data.get('query_hash', ''),
            config=data.get('config', {}),
            final_answer=data.get('final_answer', ''),
            final_sources=data.get('final_sources', []),
            status=data.get('status', 'unknown'),
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {}),
        )
        
        # Reconstruct stages
        for name, stage_data in data.get('stages', {}).items():
            stage = StageData(
                name=name,
                status=StageStatus(stage_data.get('status', 'pending')),
                duration_ms=stage_data.get('duration_ms'),
                input_params=stage_data.get('input_params', {}),
                output_count=stage_data.get('output_count', 0),
                output_data=stage_data.get('output_data'),  # Keep as dict
                metrics=stage_data.get('metrics', {}),
                error=stage_data.get('error'),
                error_type=stage_data.get('error_type'),
            )
            trace.stages[name] = stage
        
        return trace
    
    def cleanup(self) -> None:
        """Clean up tracer resources."""
        if get_current_tracer() is self:
            set_current_tracer(None)
        
        self._trace = None
        self._finalized = False


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracer(
    request_id: Optional[str] = None,
    conversation_id: Optional[str] = None
) -> RequestTracer:
    """Create and register a new tracer for the current request."""
    return RequestTracer(
        request_id=request_id,
        conversation_id=conversation_id,
        auto_register=True
    )


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_trace(trace: RequestTrace) -> Dict[str, Any]:
    """
    Analyze a trace for quality metrics and bottlenecks.
    
    Returns dict with:
    - timing_breakdown: Per-stage timing
    - bottleneck: Slowest stage
    - retrieval_funnel: Document counts through pipeline
    - retrieval_efficiency: How much filtering each stage does
    """
    timings = trace.get_timing_breakdown()
    funnel = trace.get_retrieval_funnel()
    
    # Find bottleneck
    bottleneck = max(timings.items(), key=lambda x: x[1]) if timings else (None, 0)
    
    # Calculate retrieval efficiency
    efficiency = {}
    prev_count = None
    for stage_name in [
        RequestTracer.STAGE_BM25,
        RequestTracer.STAGE_EMBEDDING_CHUNK,
        RequestTracer.STAGE_RERANKER
    ]:
        count = funnel.get(stage_name, 0)
        if prev_count is not None and prev_count > 0:
            efficiency[stage_name] = count / prev_count
        prev_count = count
    
    return {
        'timing_breakdown': timings,
        'total_ms': trace.total_duration_ms,
        'bottleneck': {'stage': bottleneck[0], 'duration_ms': bottleneck[1]},
        'retrieval_funnel': funnel,
        'retrieval_efficiency': efficiency,
        'status': trace.status,
        'error': trace.error_message,
    }


def compare_traces(trace1: RequestTrace, trace2: RequestTrace) -> Dict[str, Any]:
    """
    Compare two traces (e.g., for A/B testing).
    
    Returns comparison of timing, retrieval quality, and outputs.
    """
    t1 = analyze_trace(trace1)
    t2 = analyze_trace(trace2)
    
    return {
        'trace1_id': trace1.trace_id,
        'trace2_id': trace2.trace_id,
        'same_query': trace1.query_hash == trace2.query_hash,
        'timing_diff_ms': (t2.get('total_ms') or 0) - (t1.get('total_ms') or 0),
        'timing_comparison': {
            stage: (t2['timing_breakdown'].get(stage, 0) - t1['timing_breakdown'].get(stage, 0))
            for stage in set(t1['timing_breakdown']) | set(t2['timing_breakdown'])
        },
        'funnel_comparison': {
            stage: {
                'trace1': t1['retrieval_funnel'].get(stage, 0),
                'trace2': t2['retrieval_funnel'].get(stage, 0),
            }
            for stage in set(t1['retrieval_funnel']) | set(t2['retrieval_funnel'])
        },
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    """Demo usage of RequestTracer."""
    
    print("=== RequestTracer Demo ===\n")
    
    # Create tracer
    tracer = create_tracer(conversation_id="demo-session")
    
    # Start request
    trace_id = tracer.start_request(
        query="What is matrix suppression in mass spectrometry?",
        config={
            'k_papers': 10,
            'm_chunks': 50,
            'n_reranked': 5,
            'bm25_weight': 0.3,
            'embedding_weight': 0.7,
        }
    )
    print(f"Trace started: {trace_id}")
    
    # Simulate BM25 stage
    tracer.capture_bm25(
        results=['paper1.md', 'paper2.md', 'paper3.md'],
        scores=[12.5, 10.2, 8.7],
        duration_ms=45.2,
        k=10
    )
    print("✓ BM25 captured")
    
    # Simulate embedding stage
    tracer.capture_embedding_paper(
        results=[
            (0.15, {'file_path': 'paper1.md', 'paper_title': 'Ion Suppression'}, 'Sample text...'),
            (0.22, {'file_path': 'paper4.md', 'paper_title': 'Mass Spec Basics'}, 'More text...'),
        ],
        duration_ms=120.5,
        k=10
    )
    print("✓ Paper embedding captured")
    
    # Simulate hybrid
    tracer.capture_hybrid(
        file_scores={'paper1.md': 0.85, 'paper2.md': 0.7, 'paper4.md': 0.65},
        bm25_results=['paper1.md', 'paper2.md', 'paper3.md'],
        emb_results=[
            (0.15, {'file_path': 'paper1.md'}, ''),
            (0.22, {'file_path': 'paper4.md'}, ''),
        ],
        selected_files={'paper1.md', 'paper2.md'},
        bm25_weight=0.3,
        embedding_weight=0.7,
        k=2
    )
    print("✓ Hybrid captured")
    
    # Simulate chunk retrieval
    tracer.capture_embedding_chunk(
        results=[
            (0.12, {'file_path': 'paper1.md', 'chunk_index': 0, 'paper_title': 'Ion Suppression'}, 'Chunk 1 text here...'),
            (0.18, {'file_path': 'paper1.md', 'chunk_index': 1, 'paper_title': 'Ion Suppression'}, 'Chunk 2 text here...'),
        ],
        duration_ms=85.3,
        m=50,
        file_filter={'paper1.md', 'paper2.md'}
    )
    print("✓ Chunk embedding captured")
    
    # Simulate reranker
    tracer.capture_reranker(
        reranked=[
            {'rank': 1, 'rerank_score': 0.95, 'original_rank': 2, 'original_distance': 0.18, 
             'metadata': {'file_path': 'paper1.md', 'chunk_index': 1, 'paper_title': 'Ion Suppression'},
             'text': 'Matrix suppression is a phenomenon...'},
            {'rank': 2, 'rerank_score': 0.88, 'original_rank': 1, 'original_distance': 0.12,
             'metadata': {'file_path': 'paper1.md', 'chunk_index': 0, 'paper_title': 'Ion Suppression'},
             'text': 'In mass spectrometry...'},
        ],
        original_chunks=[
            (0.12, {'file_path': 'paper1.md', 'chunk_index': 0}, 'Chunk 1...'),
            (0.18, {'file_path': 'paper1.md', 'chunk_index': 1}, 'Chunk 2...'),
        ],
        duration_ms=1830.0,
        n=5,
        method='jina_rerank'
    )
    print("✓ Reranker captured")
    
    # Simulate generator
    tracer.capture_generator(
        answer="Matrix suppression is a phenomenon in mass spectrometry where...",
        contexts=[
            {'rank': 1, 'metadata': {'paper_title': 'Ion Suppression', 'file_path': 'paper1.md'}, 'rerank_score': 0.95},
        ],
        duration_ms=3200.5,
        prompt_tokens=540,
        completion_tokens=120,
        ttft_ms=250.0,
        temperature=0.3
    )
    print("✓ Generator captured")
    
    # Finish and get trace
    trace = tracer.finish_request()
    print(f"\n✓ Trace finalized: status={trace.status}")
    
    # Analyze
    analysis = analyze_trace(trace)
    print(f"\nAnalysis:")
    print(f"  Total time: {analysis['total_ms']:.1f}ms")
    print(f"  Bottleneck: {analysis['bottleneck']['stage']} ({analysis['bottleneck']['duration_ms']:.1f}ms)")
    print(f"  Retrieval funnel: {analysis['retrieval_funnel']}")
    
    # Export
    print(f"\nJSON preview (first 500 chars):")
    print(trace.to_json(indent=2)[:500] + "...")
    
    # Cleanup
    tracer.cleanup()
    print("\n✓ Demo complete")