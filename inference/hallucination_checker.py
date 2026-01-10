"""
Hallucination Checker - Simplified
Uses HHEM-2.1-Open to verify claims.
"""

import torch
import re
import uuid
import time
from typing import List, Dict, Optional, Any
from vllm.sampling_params import SamplingParams
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.logger import get_logger

logger = get_logger(__name__)

CLAIM_SPLIT_PROMPT = """Split this text into independent, atomic claims. Output ONLY a numbered list:

Text: {answer}"""


class HallucinationChecker:
    """Checks for hallucinations using HHEM-2.1-Open."""
    
    def __init__(
        self,
        generator: "AsyncQwenGenerator",
        hallucination_model: str = "vectara/hallucination_evaluation_model",
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        self.generator = generator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = hallucination_model
        self.threshold = threshold
        self.hallucination_model = None
        self.tokenizer = None
    
    def _load_hhem(self) -> None:
        """Lazy load HHEM model and tokenizer."""
        if self.hallucination_model is not None:
            return
        
        # Explicitly load Tokenizer and force model_max_length
        # HHEM-2.1 supports long context, but config defaults to 512.
        try:
            logger.info(f"Loading HHEM tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                model_max_length=8192
            )
        except Exception as e:
            logger.warning(f"Could not load custom tokenizer, falling back to default: {e}")

        logger.info(f"Loading HHEM model from {self.model_name}...")
        self.hallucination_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.hallucination_model = self.hallucination_model.to(self.device)
        
        self.hallucination_model.eval()
        logger.info("HHEM model loaded successfully.")
    
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
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = self.generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        sampling_params = SamplingParams(
            temperature=0.1, top_p=0.9, max_tokens=1024, skip_special_tokens=True
        )
        
        request_id = str(uuid.uuid4())
        claims_text = ""
        
        async for output in self.generator.engine.generate(prompt, sampling_params, request_id=request_id):
            claims_text = output.outputs[0].text
        
        claims = []
        for line in claims_text.strip().split('\n'):
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line.strip())
            if match:
                claim = match.group(1).strip()
                if claim and len(claim) > 5:
                    claims.append(claim)
        
        seen = set()
        return [c for c in claims if not (c.lower() in seen or seen.add(c.lower()))]
    
    def _verify_claims(self, claims: List[str], documents: List[str]) -> List[Dict]:
        """Verify claims against documents using HHEM."""
        if not claims: return []
        if not documents:
            return [{"claim": c, "is_grounded": False, "max_score": 0.0, "supporting_docs": []} for c in claims]
        
        self._load_hhem()
        
        pairs = [[doc, claim] for claim in claims for doc in documents]
        
        with torch.no_grad():
            if self.tokenizer:
                inputs = self.tokenizer(
                    pairs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=8192 
                )
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.hallucination_model(**inputs)
                scores = torch.sigmoid(outputs.logits).squeeze()
            else:
                scores = self.hallucination_model.predict(pairs)
        
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
        if isinstance(scores, float): 
            scores = [scores]
        
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
        start = time.perf_counter()
        
        documents = []
        for ctx in contexts:
            text = ctx.get('text', '') if isinstance(ctx, dict) else ctx
            if text: documents.append(text)
        
        try:
            claims = await self._split_claims(answer)
        except Exception as e:
            logger.warning(f"LLM claim splitting failed, using fallback: {e}")
            sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
            claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not claims:
            return {
                "answer": answer, "claims": [], "verifications": [], "num_claims": 0,
                "num_grounded": 0, "grounding_ratio": 1.0, "unsupported_claims": [],
                "duration_ms": (time.perf_counter() - start) * 1000
            }
        
        verifications = self._verify_claims(claims, documents)
        
        num_grounded = sum(1 for v in verifications if v["is_grounded"])
        unsupported = [v["claim"] for v in verifications if not v["is_grounded"]]
        
        return {
            "answer": answer, "claims": claims, "verifications": verifications,
            "num_claims": len(claims), "num_grounded": num_grounded,
            "grounding_ratio": num_grounded / len(claims),
            "unsupported_claims": unsupported,
            "duration_ms": (time.perf_counter() - start) * 1000
        }
    
    def cleanup(self) -> None:
        if self.hallucination_model is not None:
            del self.hallucination_model
            self.hallucination_model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()