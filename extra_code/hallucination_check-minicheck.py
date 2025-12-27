"""
Hallucination Checker Module
============================

Uses MiniCheck-RoBERTa-Large to verify claims in generated answers against
source documents. Each claim is checked against all source documents, and
is considered grounded if at least one document supports it.

Flow:
1. Split generated answer into independent claims using the generator LLM
2. For each claim, check against each source document using MiniCheck
3. A claim is "grounded" if ANY document supports it (label=1)
4. A claim is "unsupported" if ALL documents have label=0

Refactored for:
- Async generator integration (AsyncQwenGenerator)
- Unified structured logging
- Proper resource management via context managers
- Production-grade error handling
"""

import torch
import asyncio
import time
import re
import uuid
from typing import List, Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

from vllm.sampling_params import SamplingParams

from utils.logger import get_logger, log_stage_start, log_generation_metrics
from utils.config_loader import load_config

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    claim_index: int
    is_grounded: bool
    supporting_docs: List[int]
    doc_scores: List[Dict[str, Any]]
    max_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HallucinationResult:
    """Complete hallucination check result for an answer."""
    answer: str
    claims: List[str]
    verifications: List[ClaimVerification]
    num_claims: int
    num_grounded: int
    num_unsupported: int
    grounding_ratio: float
    unsupported_claims: List[str]
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'claims': self.claims,
            'verifications': [v.to_dict() for v in self.verifications],
            'num_claims': self.num_claims,
            'num_grounded': self.num_grounded,
            'num_unsupported': self.num_unsupported,
            'grounding_ratio': self.grounding_ratio,
            'unsupported_claims': self.unsupported_claims,
            'duration_ms': self.duration_ms
        }
    
    @property
    def is_fully_grounded(self) -> bool:
        """Returns True if all claims are grounded."""
        return self.num_unsupported == 0 and self.num_claims > 0


# =============================================================================
# Exceptions
# =============================================================================

class HallucinationCheckerError(Exception):
    """Hallucination checker specific exception with context."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        
        logger.error(
            f"HallucinationCheckerError: {message}",
            extra={
                'original_error': str(original_error) if original_error else None,
                'error_type': type(original_error).__name__ if original_error else None
            },
            exc_info=original_error
        )


# =============================================================================
# Main Hallucination Checker Class
# =============================================================================

class HallucinationChecker:
    """
    Checks for hallucinations in generated answers using MiniCheck.
    
    Supports both sync and async operation modes:
    - Async: Uses AsyncQwenGenerator for claim splitting (recommended)
    - Sync: Uses provided generator or falls back to simple splitting
    """
    
    CLAIM_SPLIT_SYSTEM_PROMPT = (
        "You are a helpful assistant that splits text into atomic, verifiable claims. "
        "Output ONLY a numbered list with no other text."
    )
    
    CLAIM_SPLIT_USER_TEMPLATE = """Split the following text into independent, atomic claims. Each claim should:
- Be a single, self-contained statement
- Be verifiable against source documents
- Not depend on other claims for context

Output ONLY a numbered list of claims, one per line:
1. [claim]
2. [claim]

Text to split:
{answer}"""

    # MiniCheck configuration
    DEFAULT_MINICHECK_MODEL = 'roberta-large'
    DEFAULT_CACHE_DIR = './ckpts'
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        minicheck_model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        grounding_threshold: float = 0.5
    ):
        """
        Initialize hallucination checker.
        
        Args:
            config_path: Path to config file
            minicheck_model: MiniCheck model name ('roberta-large', 'deberta-v3-large', etc.)
            cache_dir: Directory to cache MiniCheck model
            device: Computation device (auto-detected if None)
            grounding_threshold: Minimum probability threshold for grounding (default 0.5)
        """
        self.config = load_config(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.minicheck_model_name = minicheck_model or self.DEFAULT_MINICHECK_MODEL
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.grounding_threshold = grounding_threshold
        
        # MiniCheck loaded on demand
        self.minicheck_scorer = None
        self._minicheck_loaded = False
        
        logger.info(
            f"HallucinationChecker initialized",
            extra={
                'minicheck_model': self.minicheck_model_name,
                'device': self.device,
                'grounding_threshold': self.grounding_threshold
            }
        )
    
    # =========================================================================
    # MiniCheck Model Management
    # =========================================================================
    
    def _load_minicheck(self) -> None:
        """Load MiniCheck model (lazy loading)."""
        if self._minicheck_loaded:
            return
        
        log_stage_start(logger, "minicheck_loading", model=self.minicheck_model_name)
        start_time = time.perf_counter()
        
        try:
            from minicheck.minicheck import MiniCheck
            
            self.minicheck_scorer = MiniCheck(
                model_name=self.minicheck_model_name,
                cache_dir=self.cache_dir
            )
            self._minicheck_loaded = True
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"MiniCheck model loaded successfully",
                extra={'duration_ms': duration_ms, 'model': self.minicheck_model_name}
            )
            
        except ImportError as e:
            raise HallucinationCheckerError(
                "MiniCheck not installed. Install with: pip install minicheck",
                original_error=e
            )
        except Exception as e:
            raise HallucinationCheckerError(
                f"Failed to load MiniCheck model: {e}",
                original_error=e
            )
    
    def _unload_minicheck(self) -> None:
        """Unload MiniCheck model to free memory."""
        if self.minicheck_scorer is not None:
            logger.info("Unloading MiniCheck model")
            del self.minicheck_scorer
            self.minicheck_scorer = None
            self._minicheck_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # =========================================================================
    # Claim Splitting (Async)
    # =========================================================================
    
    async def split_into_claims_async(
        self,
        answer: str,
        generator: "AsyncQwenGenerator"
    ) -> List[str]:
        """
        Split an answer into independent claims using the async generator.
        
        Args:
            answer: Generated answer text
            generator: Initialized AsyncQwenGenerator instance
        
        Returns:
            List of claim strings
        """
        if not answer or not answer.strip():
            return []
        
        log_stage_start(logger, "claim_splitting", answer_length=len(answer))
        start_time = time.perf_counter()
        
        try:
            # Build the claim splitting prompt
            user_message = self.CLAIM_SPLIT_USER_TEMPLATE.format(answer=answer)
            
            # Format using the generator's tokenizer
            messages = [
                {"role": "system", "content": self.CLAIM_SPLIT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            prompt = generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Use low temperature for consistent claim extraction
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=1024,
                skip_special_tokens=True
            )
            
            # Generate using vLLM async engine
            request_id = str(uuid.uuid4())
            claims_text = ""
            
            async for request_output in generator.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            ):
                claims_text = request_output.outputs[0].text
            
            # Parse claims from the response
            claims = self._parse_claims(claims_text)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Claim splitting complete",
                extra={
                    'num_claims': len(claims),
                    'duration_ms': duration_ms,
                    'answer_length': len(answer)
                }
            )
            
            return claims
            
        except Exception as e:
            raise HallucinationCheckerError(
                f"Claim splitting failed: {e}",
                original_error=e
            )
    
    def split_into_claims_sync(self, answer: str) -> List[str]:
        """
        Fallback sync claim splitting using simple heuristics.
        
        Use this when no generator is available. Less accurate than LLM-based splitting.
        
        Args:
            answer: Generated answer text
        
        Returns:
            List of claim strings (sentence-based splitting)
        """
        if not answer or not answer.strip():
            return []
        
        logger.info("Using heuristic claim splitting (no generator provided)")
        
        # Simple sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter very short sentences
                claims.append(sentence)
        
        logger.info(f"Extracted {len(claims)} claims via heuristic splitting")
        return claims
    
    def _parse_claims(self, claims_text: str) -> List[str]:
        """
        Parse claims from LLM output.
        
        Args:
            claims_text: Raw text output from LLM
        
        Returns:
            List of cleaned claim strings
        """
        claims = []
        
        # Match numbered list format: "1. claim" or "1) claim"
        pattern = r'^\s*\d+[\.\)]\s*(.+)$'
        
        for line in claims_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                claim = match.group(1).strip()
                if claim and len(claim) > 5:
                    claims.append(claim)
            elif line and not line[0].isdigit() and len(line) > 20:
                # Fallback: substantial non-numbered lines
                claims.append(line)
        
        # Deduplicate while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_lower = claim.lower()
            if claim_lower not in seen:
                seen.add(claim_lower)
                unique_claims.append(claim)
        
        return unique_claims
    
    # =========================================================================
    # Claim Verification
    # =========================================================================
    
    def verify_claims(
        self,
        claims: List[str],
        documents: List[str]
    ) -> List[ClaimVerification]:
        """
        Verify each claim against all documents using MiniCheck.
        
        A claim is considered grounded if at least one document supports it.
        
        Args:
            claims: List of claim strings
            documents: List of source document texts
        
        Returns:
            List of ClaimVerification results
        """
        if not claims:
            return []
        
        if not documents:
            logger.warning("No documents provided for verification")
            return [
                ClaimVerification(
                    claim=claim,
                    claim_index=i,
                    is_grounded=False,
                    supporting_docs=[],
                    doc_scores=[],
                    max_score=0.0
                )
                for i, claim in enumerate(claims)
            ]
        
        # Load MiniCheck model
        self._load_minicheck()
        
        log_stage_start(
            logger, "claim_verification",
            num_claims=len(claims),
            num_documents=len(documents)
        )
        start_time = time.perf_counter()
        
        verifications = []
        
        for claim_idx, claim in enumerate(claims):
            # Build pairs: each claim against each document
            docs_for_claim = documents
            claims_for_scoring = [claim] * len(documents)
            
            try:
                pred_labels, raw_probs, _, _ = self.minicheck_scorer.score(
                    docs=docs_for_claim,
                    claims=claims_for_scoring
                )
            except Exception as e:
                logger.error(
                    f"MiniCheck scoring failed for claim {claim_idx}",
                    extra={'error': str(e), 'claim': claim[:100]}
                )
                pred_labels = [0] * len(documents)
                raw_probs = [0.0] * len(documents)
            
            # Collect results per document
            doc_scores = []
            supporting_docs = []
            
            for doc_idx, (label, prob) in enumerate(zip(pred_labels, raw_probs)):
                doc_scores.append({
                    'doc_index': doc_idx,
                    'label': int(label),
                    'probability': float(prob)
                })
                
                # Use threshold for determining support
                if label == 1 or prob >= self.grounding_threshold:
                    supporting_docs.append(doc_idx)
            
            is_grounded = len(supporting_docs) > 0
            max_score = max(raw_probs) if raw_probs else 0.0
            
            verification = ClaimVerification(
                claim=claim,
                claim_index=claim_idx,
                is_grounded=is_grounded,
                supporting_docs=supporting_docs,
                doc_scores=doc_scores,
                max_score=float(max_score)
            )
            verifications.append(verification)
            
            logger.debug(
                f"Claim {claim_idx} verified",
                extra={
                    'is_grounded': is_grounded,
                    'supporting_docs': supporting_docs,
                    'max_score': max_score
                }
            )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Claim verification complete",
            extra={
                'num_claims': len(claims),
                'num_grounded': sum(1 for v in verifications if v.is_grounded),
                'duration_ms': duration_ms
            }
        )
        
        return verifications
    
    # =========================================================================
    # Full Pipeline Methods
    # =========================================================================
    
    async def check_answer_async(
        self,
        answer: str,
        contexts: List[Dict],
        generator: "AsyncQwenGenerator"
    ) -> HallucinationResult:
        """
        Full async hallucination check pipeline.
        
        Args:
            answer: Generated answer text
            contexts: List of context dicts from reranker (with 'text' key)
            generator: Initialized AsyncQwenGenerator instance
        
        Returns:
            HallucinationResult with all verification details
        """
        log_stage_start(logger, "hallucination_check")
        start_time = time.perf_counter()
        
        # Extract document texts from contexts
        documents = self._extract_documents(contexts)
        
        # Step 1: Split answer into claims (async)
        claims = await self.split_into_claims_async(answer, generator)
        
        if not claims:
            logger.warning("No claims extracted from answer")
            return self._create_empty_result(answer, start_time)
        
        # Step 2: Verify each claim (sync - MiniCheck is not async)
        verifications = self.verify_claims(claims, documents)
        
        # Step 3: Aggregate results
        return self._aggregate_results(answer, claims, verifications, start_time)
    
    def check_answer_sync(
        self,
        answer: str,
        contexts: List[Dict],
        claims: Optional[List[str]] = None
    ) -> HallucinationResult:
        """
        Sync hallucination check pipeline.
        
        Use this when you don't have an async generator or already have claims.
        
        Args:
            answer: Generated answer text
            contexts: List of context dicts from reranker (with 'text' key)
            claims: Optional pre-extracted claims (skips claim splitting if provided)
        
        Returns:
            HallucinationResult with all verification details
        """
        log_stage_start(logger, "hallucination_check_sync")
        start_time = time.perf_counter()
        
        # Extract document texts
        documents = self._extract_documents(contexts)
        
        # Step 1: Get claims
        if claims is None:
            claims = self.split_into_claims_sync(answer)
        
        if not claims:
            logger.warning("No claims to verify")
            return self._create_empty_result(answer, start_time)
        
        # Step 2: Verify claims
        verifications = self.verify_claims(claims, documents)
        
        # Step 3: Aggregate results
        return self._aggregate_results(answer, claims, verifications, start_time)
    
    def _extract_documents(self, contexts: List[Dict]) -> List[str]:
        """Extract document texts from context dicts."""
        documents = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                text = ctx.get('text', '')
                if text:
                    documents.append(text)
            elif isinstance(ctx, str):
                documents.append(ctx)
        
        logger.info(f"Extracted {len(documents)} documents from contexts")
        return documents
    
    def _create_empty_result(self, answer: str, start_time: float) -> HallucinationResult:
        """Create result for empty claims case."""
        duration_ms = (time.perf_counter() - start_time) * 1000
        return HallucinationResult(
            answer=answer,
            claims=[],
            verifications=[],
            num_claims=0,
            num_grounded=0,
            num_unsupported=0,
            grounding_ratio=1.0,
            unsupported_claims=[],
            duration_ms=duration_ms
        )
    
    def _aggregate_results(
        self,
        answer: str,
        claims: List[str],
        verifications: List[ClaimVerification],
        start_time: float
    ) -> HallucinationResult:
        """Aggregate verification results into final result."""
        num_grounded = sum(1 for v in verifications if v.is_grounded)
        num_unsupported = len(verifications) - num_grounded
        unsupported_claims = [v.claim for v in verifications if not v.is_grounded]
        grounding_ratio = num_grounded / len(claims) if claims else 1.0
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        result = HallucinationResult(
            answer=answer,
            claims=claims,
            verifications=verifications,
            num_claims=len(claims),
            num_grounded=num_grounded,
            num_unsupported=num_unsupported,
            grounding_ratio=grounding_ratio,
            unsupported_claims=unsupported_claims,
            duration_ms=duration_ms
        )
        
        logger.info(
            f"Hallucination check complete",
            extra={
                'num_claims': len(claims),
                'num_grounded': num_grounded,
                'num_unsupported': num_unsupported,
                'grounding_ratio': grounding_ratio,
                'duration_ms': duration_ms
            }
        )
        
        return result
    
    def cleanup(self) -> None:
        """Unload models and free memory."""
        self._unload_minicheck()
        logger.info("HallucinationChecker cleanup complete")


# =============================================================================
# Context Manager for Resource Management
# =============================================================================

@asynccontextmanager
async def hallucination_checker_context(
    config_path: Optional[str] = None,
    minicheck_model: Optional[str] = None,
    **init_kwargs
):
    """
    Async context manager for HallucinationChecker with automatic cleanup.
    
    Usage:
        async with hallucination_checker_context() as checker:
            result = await checker.check_answer_async(answer, contexts, generator)
    """
    checker = HallucinationChecker(
        config_path=config_path,
        minicheck_model=minicheck_model,
        **init_kwargs
    )
    
    try:
        yield checker
    finally:
        checker.cleanup()


# =============================================================================
# Utility Functions
# =============================================================================

def format_hallucination_output(
    result: HallucinationResult,
    verbose: bool = False
) -> str:
    """
    Format hallucination check result for display.
    
    Args:
        result: HallucinationResult from checker
        verbose: If True, show all claims; if False, only unsupported
    
    Returns:
        Formatted string for display
    """
    lines = []
    
    if result.num_unsupported > 0:
        lines.append(
            f"\n⚠️  Grounding Check: {result.num_grounded}/{result.num_claims} "
            f"claims supported ({result.grounding_ratio:.0%})"
        )
        lines.append("\nUnsupported claims:")
        for i, claim in enumerate(result.unsupported_claims, 1):
            lines.append(f"  {i}. {claim}")
    else:
        lines.append(
            f"\n✓ All {result.num_claims} claims are grounded in source documents."
        )
    
    if result.duration_ms:
        lines.append(f"\n[Checked in {result.duration_ms:.0f}ms]")
    
    if verbose:
        lines.append("\n\nAll claims:")
        for v in result.verifications:
            status = "✓" if v.is_grounded else "✗"
            docs = f"(docs: {v.supporting_docs})" if v.supporting_docs else "(no support)"
            claim_preview = v.claim[:80] + "..." if len(v.claim) > 80 else v.claim
            lines.append(f"  {status} {claim_preview} {docs}")
    
    return "\n".join(lines)


def quick_grounding_check(
    answer: str,
    contexts: List[Dict],
    threshold: float = 0.7
) -> bool:
    """
    Quick check if answer is sufficiently grounded.
    
    Args:
        answer: Generated answer text
        contexts: List of context dicts
        threshold: Minimum grounding ratio to pass
    
    Returns:
        True if grounding_ratio >= threshold
    """
    checker = HallucinationChecker()
    try:
        result = checker.check_answer_sync(answer, contexts)
        return result.grounding_ratio >= threshold
    finally:
        checker.cleanup()


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    from utils.logger import set_request_context, clear_request_context
    
    async def test_with_generator():
        """Test async hallucination checking with generator."""
        # Import here to avoid circular imports
        from inference.generator import AsyncQwenGenerator, async_generator_context
        
        set_request_context(req_id="test-hallucination-1")
        
        mock_contexts = [
            {
                'text': (
                    "Matrix suppression is a phenomenon in mass spectrometry where "
                    "components in the sample matrix interfere with the ionization process. "
                    "This effect is particularly pronounced in electrospray ionization (ESI)."
                ),
                'metadata': {'paper_title': 'Ion Suppression in Mass Spectrometry'}
            },
            {
                'text': (
                    "Ion competition effects are a major source of matrix suppression in ESI-MS. "
                    "When multiple species are present in the electrospray droplet, they compete "
                    "for the limited charge available."
                ),
                'metadata': {'paper_title': 'Analytical Challenges in LC-MS/MS'}
            }
        ]
        
        # Mock answer with one hallucinated claim
        mock_answer = (
            "Matrix suppression is a phenomenon in mass spectrometry where matrix "
            "components interfere with ionization. It is particularly common in ESI "
            "(electrospray ionization). Ion competition is a major cause of this effect. "
            "Matrix suppression was first discovered in 1985 by Dr. Smith."
        )
        
        print("\n" + "=" * 60)
        print("ASYNC HALLUCINATION CHECK TEST")
        print("=" * 60)
        
        async with async_generator_context(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        ) as generator:
            async with hallucination_checker_context() as checker:
                result = await checker.check_answer_async(
                    answer=mock_answer,
                    contexts=mock_contexts,
                    generator=generator
                )
                
                print(format_hallucination_output(result, verbose=True))
                print(f"\nResult dict keys: {list(result.to_dict().keys())}")
        
        clear_request_context()
    
    def test_sync_verification():
        """Test sync verification with pre-defined claims."""
        set_request_context(req_id="test-hallucination-sync")
        
        mock_contexts = [
            {
                'text': (
                    "Matrix suppression is a phenomenon in mass spectrometry where "
                    "components in the sample matrix interfere with the ionization process."
                )
            },
            {
                'text': (
                    "Ion competition effects are a major source of matrix suppression in ESI-MS."
                )
            }
        ]
        
        # Pre-defined claims for sync testing
        mock_claims = [
            "Matrix suppression is a phenomenon in mass spectrometry.",
            "Matrix components interfere with ionization.",
            "Ion competition causes matrix suppression.",
            "Matrix suppression was discovered in 1985 by Dr. Smith."  # Hallucination
        ]
        
        print("\n" + "=" * 60)
        print("SYNC HALLUCINATION CHECK TEST")
        print("=" * 60)
        
        checker = HallucinationChecker()
        try:
            result = checker.check_answer_sync(
                answer="Test answer",
                contexts=mock_contexts,
                claims=mock_claims
            )
            
            print(format_hallucination_output(result, verbose=True))
            
            # Show detailed verification
            print("\nDetailed scores:")
            for v in result.verifications:
                status = "GROUNDED" if v.is_grounded else "UNSUPPORTED"
                print(f"\n[{status}] {v.claim}")
                print(f"  Max score: {v.max_score:.4f}")
                print(f"  Supporting docs: {v.supporting_docs}")
        finally:
            checker.cleanup()
        
        clear_request_context()
    
    # Run sync test (doesn't require generator)
    test_sync_verification()
    
    # Uncomment to run async test (requires generator setup)
    # asyncio.run(test_with_generator())