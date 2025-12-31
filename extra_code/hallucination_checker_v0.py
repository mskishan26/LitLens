"""
Hallucination Checker Module
=============================

Uses MiniCheck-RoBERTa-Large to verify claims in generated answers against
source documents. Each claim is checked against all source documents, and
is considered grounded if at least one document supports it.

Flow:
1. Split generated answer into independent claims using the generator LLM
2. For each claim, check against each source document using MiniCheck
3. A claim is "grounded" if ANY document supports it (label=1)
4. A claim is "unsupported" if ALL documents have label=0
"""

import torch
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import re

from utils.logger import get_chat_logger
from utils.config_loader import load_config

logger = get_chat_logger(__name__)


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    claim_index: int
    is_grounded: bool  # True if ANY document supports it
    supporting_docs: List[int]  # Indices of documents that support this claim
    doc_scores: List[Dict[str, Any]]  # Detailed scores per document
    max_score: float  # Highest probability score across all docs
    
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
    grounding_ratio: float  # num_grounded / num_claims
    unsupported_claims: List[str]  # List of claims that are not grounded
    
    def to_dict(self) -> Dict:
        return {
            'answer': self.answer,
            'claims': self.claims,
            'verifications': [v.to_dict() for v in self.verifications],
            'num_claims': self.num_claims,
            'num_grounded': self.num_grounded,
            'num_unsupported': self.num_unsupported,
            'grounding_ratio': self.grounding_ratio,
            'unsupported_claims': self.unsupported_claims
        }


class HallucinationChecker:
    """
    Checks for hallucinations in generated answers using MiniCheck.
    
    Uses a two-step process:
    1. Split answer into claims using the generator model
    2. Verify each claim against source documents using MiniCheck
    """
    
    CLAIM_SPLIT_PROMPT = """Split the following text into independent, atomic claims. Each claim should:
- Be a single, self-contained statement
- Be verifiable against source documents
- Not depend on other claims for context

Output ONLY a numbered list of claims, one per line, in the format:
1. [claim]
2. [claim]
...

Do not include any other text or explanation.

Text to split:
{answer}"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        minicheck_model: str = 'roberta-large',
        cache_dir: str = './ckpts',
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize hallucination checker.
        
        Args:
            config_path: Path to config file
            minicheck_model: MiniCheck model name ('roberta-large', 'deberta-v3-large', etc.)
            cache_dir: Directory to cache MiniCheck model
            device: Computation device
        """
        self.config = load_config(config_path)
        self.device = device
        self.minicheck_model_name = minicheck_model
        self.cache_dir = cache_dir
        
        # Models loaded on demand
        self.minicheck_scorer = None
        self._minicheck_loaded = False
        
        logger.info(f"HallucinationChecker initialized")
        logger.info(f"  MiniCheck model: {minicheck_model}")
        logger.info(f"  Device: {device}")
    
    def _load_minicheck(self):
        """Load MiniCheck model."""
        if self._minicheck_loaded:
            return
        
        logger.info(f"Loading MiniCheck model: {self.minicheck_model_name}")
        try:
            from minicheck.minicheck import MiniCheck
            
            self.minicheck_scorer = MiniCheck(
                model_name=self.minicheck_model_name,
                cache_dir=self.cache_dir
            )
            self._minicheck_loaded = True
            logger.info("MiniCheck model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import MiniCheck: {e}")
            logger.error("Install with: pip install minicheck")
            raise
        except Exception as e:
            logger.error(f"Failed to load MiniCheck: {e}")
            raise
    
    def _unload_minicheck(self):
        """Unload MiniCheck model to free memory."""
        if self.minicheck_scorer is not None:
            logger.info("Unloading MiniCheck model")
            del self.minicheck_scorer
            self.minicheck_scorer = None
            self._minicheck_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def split_into_claims(
        self, 
        answer: str, 
        generator=None
    ) -> List[str]:
        """
        Split an answer into independent claims using the generator LLM.
        
        Args:
            answer: Generated answer text
            generator: QwenGenerator instance (if None, will load one)
        
        Returns:
            List of claim strings
        """
        if not answer or not answer.strip():
            return []
        
        logger.info(f"Splitting answer into claims (length: {len(answer)} chars)")
        
        # Build prompt for claim splitting
        prompt = self.CLAIM_SPLIT_PROMPT.format(answer=answer)
        
        # Use provided generator or load one
        own_generator = False
        if generator is None:
            logger.info("Loading generator for claim splitting")
            from inference.generator import QwenGenerator
            generator = QwenGenerator(device=self.device)
            own_generator = True
        
        try:
            # Generate claims using a simple message format
            messages = [
                {"role": "system", "content": "You are a helpful assistant that splits text into atomic claims."},
                {"role": "user", "content": prompt}
            ]
            
            prompt_text = generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            with torch.no_grad():
                inputs = generator.tokenizer([prompt_text], return_tensors="pt").to(generator.device)
                
                outputs = generator.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,  # Low temperature for consistent splitting
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                claims_text = generator.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Parse claims from numbered list
            claims = self._parse_claims(claims_text)
            logger.info(f"Extracted {len(claims)} claims")
            
            return claims
            
        finally:
            if own_generator:
                generator.cleanup()
    
    def _parse_claims(self, claims_text: str) -> List[str]:
        """
        Parse claims from LLM output.
        
        Args:
            claims_text: Raw text output from LLM
        
        Returns:
            List of cleaned claim strings
        """
        claims = []
        
        # Try numbered list format first: "1. claim" or "1) claim"
        pattern = r'^\s*\d+[\.\)]\s*(.+)$'
        
        for line in claims_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                claim = match.group(1).strip()
                if claim and len(claim) > 5:  # Filter out very short claims
                    claims.append(claim)
            elif line and not line[0].isdigit():
                # Fallback: treat non-numbered lines as claims if they're substantial
                if len(line) > 20:  # Minimum length for a claim
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
        
        self._load_minicheck()
        
        logger.info(f"Verifying {len(claims)} claims against {len(documents)} documents")
        
        verifications = []
        
        for claim_idx, claim in enumerate(claims):
            # Build pairs: each claim against each document
            docs_for_claim = documents  # All documents
            claims_for_scoring = [claim] * len(documents)  # Same claim repeated
            
            # Score all pairs at once for efficiency
            try:
                pred_labels, raw_probs, _, _ = self.minicheck_scorer.score(
                    docs=docs_for_claim,
                    claims=claims_for_scoring
                )
            except Exception as e:
                logger.error(f"MiniCheck scoring failed for claim {claim_idx}: {e}")
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
                
                if label == 1:
                    supporting_docs.append(doc_idx)
            
            # A claim is grounded if ANY document supports it
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
                f"Claim {claim_idx}: grounded={is_grounded}, "
                f"supporting_docs={supporting_docs}, max_score={max_score:.4f}"
            )
        
        return verifications
    
    def check_answer(
        self,
        answer: str,
        contexts: List[Dict],
        generator=None
    ) -> HallucinationResult:
        """
        Full hallucination check pipeline.
        
        Args:
            answer: Generated answer text
            contexts: List of context dicts from reranker (with 'text' key)
            generator: Optional QwenGenerator instance for claim splitting
        
        Returns:
            HallucinationResult with all verification details
        """
        logger.info("Starting hallucination check")
        
        # Extract document texts from contexts
        documents = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                text = ctx.get('text', '')
                if text:
                    documents.append(text)
            elif isinstance(ctx, str):
                documents.append(ctx)
        
        if not documents:
            logger.warning("No valid documents found in contexts")
        
        logger.info(f"Extracted {len(documents)} documents from contexts")
        
        # Step 1: Split answer into claims
        claims = self.split_into_claims(answer, generator)
        
        if not claims:
            logger.warning("No claims extracted from answer")
            return HallucinationResult(
                answer=answer,
                claims=[],
                verifications=[],
                num_claims=0,
                num_grounded=0,
                num_unsupported=0,
                grounding_ratio=1.0,  # No claims = nothing to verify
                unsupported_claims=[]
            )
        
        # Step 2: Verify each claim
        verifications = self.verify_claims(claims, documents)
        
        # Step 3: Aggregate results
        num_grounded = sum(1 for v in verifications if v.is_grounded)
        num_unsupported = len(verifications) - num_grounded
        unsupported_claims = [v.claim for v in verifications if not v.is_grounded]
        
        grounding_ratio = num_grounded / len(claims) if claims else 1.0
        
        result = HallucinationResult(
            answer=answer,
            claims=claims,
            verifications=verifications,
            num_claims=len(claims),
            num_grounded=num_grounded,
            num_unsupported=num_unsupported,
            grounding_ratio=grounding_ratio,
            unsupported_claims=unsupported_claims
        )
        
        logger.info(
            f"Hallucination check complete: "
            f"{num_grounded}/{len(claims)} claims grounded ({grounding_ratio:.1%})"
        )
        
        return result
    
    def cleanup(self):
        """Unload models and free memory."""
        self._unload_minicheck()


def format_hallucination_output(result: HallucinationResult, verbose: bool = False) -> str:
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
        lines.append(f"\n⚠️  Grounding Check: {result.num_grounded}/{result.num_claims} claims supported ({result.grounding_ratio:.0%})")
        lines.append("\nUnsupported claims:")
        for i, claim in enumerate(result.unsupported_claims, 1):
            lines.append(f"  {i}. {claim}")
    else:
        lines.append(f"\n✓ All {result.num_claims} claims are grounded in source documents.")
    
    if verbose:
        lines.append("\n\nAll claims:")
        for v in result.verifications:
            status = "✓" if v.is_grounded else "✗"
            docs = f"(docs: {v.supporting_docs})" if v.supporting_docs else "(no support)"
            lines.append(f"  {status} {v.claim[:80]}... {docs}")
    
    return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Test with mock data
    checker = HallucinationChecker()
    
    # Mock contexts (would come from reranker in real usage)
    mock_contexts = [
        {
            'text': "Matrix suppression is a phenomenon in mass spectrometry where components in the sample matrix interfere with the ionization process. This effect is particularly pronounced in electrospray ionization (ESI)."
        },
        {
            'text': "Ion competition effects are a major source of matrix suppression in ESI-MS. When multiple species are present in the electrospray droplet, they compete for the limited charge available."
        }
    ]
    
    # Mock answer
    mock_answer = """Matrix suppression is a phenomenon in mass spectrometry where matrix components interfere with ionization. 
    It is particularly common in ESI (electrospray ionization). 
    Ion competition is a major cause of this effect.
    Matrix suppression was first discovered in 1985 by Dr. Smith."""
    
    # Manual claim splitting for testing (without generator)
    mock_claims = [
        "Matrix suppression is a phenomenon in mass spectrometry where matrix components interfere with ionization.",
        "Matrix suppression is particularly common in ESI (electrospray ionization).",
        "Ion competition is a major cause of matrix suppression.",
        "Matrix suppression was first discovered in 1985 by Dr. Smith."
    ]
    
    # Test verification
    documents = [ctx['text'] for ctx in mock_contexts]
    verifications = checker.verify_claims(mock_claims, documents)
    
    print("\n" + "="*60)
    print("HALLUCINATION CHECK TEST")
    print("="*60)
    
    for v in verifications:
        status = "GROUNDED" if v.is_grounded else "UNSUPPORTED"
        print(f"\n[{status}] {v.claim}")
        print(f"  Max score: {v.max_score:.4f}")
        print(f"  Supporting docs: {v.supporting_docs}")
    
    # Summary
    grounded = sum(1 for v in verifications if v.is_grounded)
    print(f"\n{'='*60}")
    print(f"Summary: {grounded}/{len(verifications)} claims grounded")
    print("="*60)
    
    checker.cleanup()