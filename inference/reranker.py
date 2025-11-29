import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

from utils.logger import get_chat_logger
from utils.config_loader import load_config

logger = get_chat_logger(__name__)


class Reranker:
    """Qwen-based reranker for Stage 3 of RAG pipeline (chat/query time only)."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
        max_length: int = 8192
    ):
        """
        Initialize Qwen reranker.
        
        Args:
            config_path: Path to config file
            model_name: HuggingFace model name for reranker (overrides config)
            device: 'cuda' or 'cpu'
            batch_size: Batch size for reranking (smaller for large models)
            max_length: Maximum sequence length for reranker
        """
        self.config = load_config(config_path)
        model_name = model_name or self.config['models']['reranker']
        
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        logger.info(f"Loading reranker model '{model_name}' on {device}")
        
        # Load tokenizer with left padding (required for causal LM batching)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Set padding token (use eos_token as pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load causal LM model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device)
        
        self.model.eval()
        
        # Get token IDs for "yes" and "no" (used for scoring)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # Define prompt format (from official Qwen reranker)
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        logger.info(f"Reranker loaded successfully")
        logger.info(f"  Device: {device}, Batch size: {batch_size}, Max length: {max_length}")
        logger.info(f"  Token IDs - yes: {self.token_true_id}, no: {self.token_false_id}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[float, Dict, str]]:
        """
        Rerank candidates using the Qwen reranker model.
        
        Args:
            query: Search query
            candidates: List of (distance, metadata, chunk_text) from Stage 2
                        Note: chunk_text might be None, will load if needed
            top_k: Number of top results to return after reranking
            return_scores: If True, return reranker scores; if False, return ranks
        
        Returns:
            List of (score/rank, metadata, chunk_text) tuples, sorted by relevance
        """
        if not candidates:
            logger.warning("No candidates provided for reranking")
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates for query: '{query[:100]}...'")
        
        # Extract texts from candidates (load if None)
        texts = []
        valid_candidates = []
        
        for dist, meta, chunk_text in candidates:
            if chunk_text is None:
                try:
                    chunk_text = self._load_chunk_from_metadata(meta)
                except Exception as e:
                    logger.error(f"Could not load text for {meta['file_path']}: {e}")
                    continue
            
            texts.append(chunk_text)
            valid_candidates.append((dist, meta, chunk_text))
        
        if not texts:
            logger.warning("No valid candidates after loading texts")
            return []
        
        # Compute reranker scores
        scores = self._compute_scores(query, texts)
        
        # Combine scores with candidates
        reranked = []
        for score, (_, meta, text) in zip(scores, valid_candidates):
            reranked.append((float(score), meta, text))
        
        # Sort by score (higher is better)
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        result = reranked[:top_k]
        
        # Optionally convert to ranks
        if not return_scores:
            result = [(i + 1, meta, text) for i, (_, meta, text) in enumerate(result)]
        
        logger.info(f"Reranking complete, returning top {len(result)} results")
        return result
    
    def rerank_with_details(
        self,
        query: str,
        candidates: List[Tuple[float, Dict, str]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank and return detailed results with both original and reranker scores.
        
        Args:
            query: Search query
            candidates: List of (distance, metadata, chunk_text) from Stage 2
            top_k: Number of top results to return
        
        Returns:
            List of dicts with detailed ranking information
        """
        logger.info(f"Performing detailed reranking for top {top_k} results")
        
        # Get reranked results
        reranked = self.rerank(query, candidates, top_k=top_k, return_scores=True)
        
        # Add detailed information
        detailed_results = []
        for rank, (rerank_score, meta, text) in enumerate(reranked, 1):
            # Find original distance and rank from candidates
            original_dist = None
            original_rank = None
            for orig_rank, (dist, orig_meta, _) in enumerate(candidates, 1):
                if orig_meta == meta:
                    original_dist = dist
                    original_rank = orig_rank
                    break
            
            detailed_results.append({
                'rank': rank,
                'rerank_score': rerank_score,
                'original_distance': original_dist,
                'original_rank': original_rank,
                'rank_improvement': original_rank - rank if original_rank else None,
                'metadata': meta,
                'text': text
            })
        
        return detailed_results
    
    def _load_chunk_from_metadata(self, meta: Dict) -> str:
        """
        Load the specific chunk from file using metadata.
        
        Args:
            meta: Chunk metadata with token positions
        
        Returns:
            Chunk text
        """
        with open(meta['file_path'], 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        if 'token_start' in meta and 'token_end' in meta:
            # Use token positions to extract chunk
            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            chunk_tokens = tokens[meta['token_start']:meta['token_end']]
            return self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        else:
            # Fallback: approximate position
            chunk_idx = meta.get('chunk_index', 0)
            chunk_size = meta.get('chunk_size', 800)
            
            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
            start = chunk_idx * chunk_size
            end = start + chunk_size
            chunk_tokens = tokens[start:min(end, len(tokens))]
            return self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
    
    def _format_instruction(
        self, 
        query: str, 
        doc: str, 
        instruction: Optional[str] = None
    ) -> str:
        """
        Format query and document into the reranker prompt format.
        
        Args:
            query: Search query
            doc: Document/chunk text
            instruction: Optional task instruction
        
        Returns:
            Formatted prompt string
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
    
    def _process_inputs(self, pairs: List[str]) -> Dict:
        """
        Tokenize and prepare inputs following official Qwen format.
        
        Args:
            pairs: List of formatted query-document strings
        
        Returns:
            Tokenized inputs ready for model
        """
        # Tokenize without padding first
        # We truncate the middle part (query + doc) to ensure the final sequence 
        # (prefix + middle + suffix) fits within max_length
        truncation_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=truncation_length
        )
        
        input_ids_list = inputs['input_ids']
        
        # Add prefix and suffix tokens
        final_input_ids = []
        for ele in input_ids_list:
            final_input_ids.append(self.prefix_tokens + ele + self.suffix_tokens)
        
        # Manual padding to avoid tokenizer warnings about efficient usage
        # We pad to the longest sequence in the batch (left padding)
        max_len = max(len(ids) for ids in final_input_ids)
        
        padded_input_ids = []
        attention_masks = []
        pad_token_id = self.tokenizer.pad_token_id
        
        for ids in final_input_ids:
            pad_len = max_len - len(ids)
            # Left padding
            padded_ids = [pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
            
        # Convert to tensors and move to device
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long, device=self.device)
        }
    
    def _compute_scores(self, query: str, texts: List[str]) -> np.ndarray:
        """
        Compute reranker scores for query-text pairs using yes/no token logits.
        
        Args:
            query: Search query
            texts: List of candidate texts
        
        Returns:
            Array of scores (higher = more relevant)
        """
        scores = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Reranking"):
            batch_texts = texts[i:i + self.batch_size]
            
            # Format query-document pairs
            pairs = [self._format_instruction(query, text) for text in batch_texts]
            
            # Process inputs
            with torch.no_grad():
                inputs = self._process_inputs(pairs)
                
                # Get model outputs
                outputs = self.model(**inputs)
                
                # Extract logits from last token position
                batch_logits = outputs.logits[:, -1, :]
                
                # Get yes/no token logits
                true_vector = batch_logits[:, self.token_true_id]
                false_vector = batch_logits[:, self.token_false_id]
                
                # Stack and compute log softmax
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                
                # Convert to probabilities (exp of log probability for "yes")
                batch_probs = batch_scores[:, 1].exp().cpu().numpy()
                scores.extend(batch_probs)
        
        return np.array(scores)


def get_optimal_reranker_batch_size() -> int:
    """Determine optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 1
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if gpu_memory_gb >= 70:  # A100 80GB, H100
        return 8
    elif gpu_memory_gb >= 35:  # A100 40GB
        return 4
    else:  # V100 16GB or smaller
        return 2