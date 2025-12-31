"""
Hallucination Checker - Simplified
Uses HHEM-2.1-Open to verify claims against source documents.
"""

import torch
import re
import uuid
import time
from typing import List, Dict, Optional, Any
from vllm.sampling_params import SamplingParams

from utils.logger import get_logger
from utils.config_loader import load_config

logger = get_logger(__name__)

CLAIM_SPLIT_PROMPT = """Split this text into independent, atomic claims. Output ONLY a numbered list:

Text: {answer}"""


class HallucinationChecker:
    """Checks for hallucinations using HHEM-2.1-Open."""
    
    def __init__(
        self,
        generator: "AsyncQwenGenerator",
        config_path: Optional[str] = None,
        hhem_model: str = "vectara/hallucination_evaluation_model",
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        self.config = load_config(config_path)
        self.generator = generator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hhem_model_name = hhem_model
        self.threshold = threshold
        self.hhem_model = None
    
    def _load_hhem(self) -> None:
        """Lazy load HHEM model."""
        if self.hhem_model is not None:
            return
        
        from transformers import AutoModelForSequenceClassification
        
        self.hhem_model = AutoModelForSequenceClassification.from_pretrained(
            self.hhem_model_name,
            trust_remote_code=True
        )
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.hhem_model = self.hhem_model.to(self.device)
        
        self.hhem_model.eval()
        logger.info(f"HHEM model loaded: {self.hhem_model_name}")
    
    async def _split_claims(self, answer: str) -> List[str]:
        """Split answer into claims using the generator."""
        if not answer or not answer.strip():
            return []
        
        messages = [
            {"role": "system", "content": "You split text into atomic claims. Output ONLY a numbered list."},
            {"role": "user", "content": CLAIM_SPLIT_PROMPT.format(answer=answer)}
        ]
        
        try:
            prompt = self.generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback if tokenizer doesn't support enable_thinking
            prompt = self.generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            skip_special_tokens=True
        )
        
        request_id = str(uuid.uuid4())
        claims_text = ""
        
        async for output in self.generator.engine.generate(prompt, sampling_params, request_id=request_id):
            claims_text = output.outputs[0].text
        
        # Parse numbered list
        claims = []
        for line in claims_text.strip().split('\n'):
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line.strip())
            if match:
                claim = match.group(1).strip()
                if claim and len(claim) > 5:
                    claims.append(claim)
        
        # Deduplicate
        seen = set()
        return [c for c in claims if not (c.lower() in seen or seen.add(c.lower()))]
    
    def _verify_claims(self, claims: List[str], documents: List[str]) -> List[Dict]:
        """Verify claims against documents using HHEM."""
        if not claims:
            return []
        
        if not documents:
            return [
                {"claim": c, "is_grounded": False, "max_score": 0.0, "supporting_docs": []}
                for c in claims
            ]
        
        self._load_hhem()
        
        # Create (doc, claim) pairs
        pairs = [(doc, claim) for claim in claims for doc in documents]
        
        with torch.no_grad():
            scores = self.hhem_model.predict(pairs)
        
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
        
        # Organize scores by claim
        num_docs = len(documents)
        results = []
        
        for i, claim in enumerate(claims):
            claim_scores = scores[i * num_docs : (i + 1) * num_docs]
            supporting = [j for j, s in enumerate(claim_scores) if s >= self.threshold]
            max_score = max(claim_scores) if claim_scores else 0.0
            
            results.append({
                "claim": claim,
                "is_grounded": len(supporting) > 0,
                "max_score": float(max_score),
                "supporting_docs": supporting
            })
        
        return results
    
    async def check(self, answer: str, contexts: List[Dict]) -> Dict[str, Any]:
        """
        Check answer for hallucinations.
        
        Returns dict with: claims, verifications, grounding_ratio, unsupported_claims
        """
        start = time.perf_counter()
        
        # Extract document texts
        documents = []
        for ctx in contexts:
            text = ctx.get('text', '') if isinstance(ctx, dict) else ctx
            if text:
                documents.append(text)
        
        # Split into claims
        try:
            claims = await self._split_claims(answer)
        except Exception as e:
            logger.warning(f"LLM claim splitting failed, using fallback: {e}")
            # Fallback: simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
            claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not claims:
            return {
                "answer": answer,
                "claims": [],
                "verifications": [],
                "num_claims": 0,
                "num_grounded": 0,
                "grounding_ratio": 1.0,
                "unsupported_claims": [],
                "duration_ms": (time.perf_counter() - start) * 1000
            }
        
        # Verify
        verifications = self._verify_claims(claims, documents)
        
        num_grounded = sum(1 for v in verifications if v["is_grounded"])
        unsupported = [v["claim"] for v in verifications if not v["is_grounded"]]
        
        return {
            "answer": answer,
            "claims": claims,
            "verifications": verifications,
            "num_claims": len(claims),
            "num_grounded": num_grounded,
            "grounding_ratio": num_grounded / len(claims),
            "unsupported_claims": unsupported,
            "duration_ms": (time.perf_counter() - start) * 1000
        }
    
    def cleanup(self) -> None:
        """Free HHEM model memory."""
        if self.hhem_model is not None:
            del self.hhem_model
            self.hhem_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import asyncio
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    from inference.generator import AsyncQwenGenerator, async_generator_context
    
    async def test():
        contexts = [
            {"text": "Matrix suppression is a phenomenon in mass spectrometry where components interfere with ionization."},
            {"text": "Ion competition effects are a major source of matrix suppression in ESI-MS."}
        ]
        
        answer = (
            "Matrix suppression is a phenomenon in mass spectrometry. "
            "Ion competition is a major cause. "
            "It was discovered in 1985 by Dr. Smith."  # hallucination
        )
        
        async with async_generator_context(tensor_parallel_size=1, gpu_memory_utilization=0.9) as generator:
            checker = HallucinationChecker(generator=generator, device="cpu")
            result = await checker.check(answer, contexts)
            print(result)
            print(f"Claims: {result['num_claims']}")
            print(f"Grounded: {result['num_grounded']}")
            print(f"Ratio: {result['grounding_ratio']:.0%}")
            print(f"Unsupported: {result['unsupported_claims']}")
            
            checker.cleanup()
    
    asyncio.run(test())