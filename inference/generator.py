import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from contextlib import contextmanager

# Setup logging

from utils.logger import get_chat_logger
from utils.config_loader import load_config

logger_chat = get_chat_logger(__name__)


class GeneratorError(Exception):
    """Custom exception for generator errors with context preservation."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        """
        Initialize error with context.
        
        Args:
            message: Error description
            context: Dict with debugging context (query, params, GPU state, etc.)
            original_error: The original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_error = original_error
        
        # Log full error details immediately when exception is created
        logger_chat.error(
            f"GeneratorError: {message}",
            extra={
                'error_context': self.context,
                'original_error': str(original_error) if original_error else None,
                'original_error_type': type(original_error).__name__ if original_error else None
            },
            exc_info=original_error
        )
    
    def __str__(self):
        """Include context in string representation."""
        base_msg = self.message
        if self.context:
            context_str = ', '.join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" | Context: {context_str}"
        if self.original_error:
            base_msg += f" | Caused by: {type(self.original_error).__name__}: {self.original_error}"
        return base_msg


class QwenGenerator:
    """Qwen-based text generator for RAG pipeline Stage 4."""
    
    # Configuration constants
    DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-1M"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_P = 0.9
    MAX_CONTEXT_WINDOW = 48_000  # 1M tokens for Qwen2.5-1M
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """
        Initialize Qwen generator for Stage 4 (final generation) of RAG pipeline.
        
        Args:
            config_path: Path to config file
            model_name: HuggingFace model name for generation (overrides config)
            device: 'cuda' or 'cpu'
            max_new_tokens: Maximum tokens to generate in response (overrides config)
            temperature: Sampling temperature (lower = more deterministic) (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            
        Raises:
            GeneratorError: If model loading fails
        """
        self.config = load_config(config_path)
        
        # Load defaults from config or constants
        self.model_name = model_name or self.config['models']['generator']
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_TOKENS
        self.temperature = temperature if temperature is not None else self.config['generation']['temperature']
        self.top_p = top_p or self.DEFAULT_TOP_P
        self.device = device
        
        logger_chat.info(f"Initializing generator: model={self.model_name}, device={device}")
        
        try:
            self._load_model()
        except Exception as e:
            error_context = {
                'model_name': model_name,
                'device': device,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            raise GeneratorError(
                f"Model loading failed: {e}",
                context=error_context,
                original_error=e
            )
        
        logger_chat.info(
            f"Generator initialized successfully - "
            f"max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}"
        )
    
    def _load_model(self):
        """Load tokenizer and model with error handling."""
        logger_chat.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set padding token (use eos_token as pad_token if missing)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger_chat.info(f"Loading model from {self.model_name}")
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # Log memory usage if CUDA
        if self.device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            logger_chat.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer based on query and retrieved contexts.
        
        Args:
            query: User's question
            contexts: List of context dicts from reranker
            system_prompt: Optional custom system prompt
            include_citations: Whether to prompt model to cite sources
            max_contexts: Maximum number of contexts to include
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dict containing generated answer and metadata
            
        Raises:
            GeneratorError: If generation fails
        """
        try:
            logger_chat.info(
                f"Generating response for query (length={len(query)}), "
                f"using {len(contexts)} contexts (max={max_contexts})"
            )
            
            # Limit contexts
            contexts = contexts[:max_contexts]
            
            # Build generation parameters
            gen_params = self._prepare_generation_params(**generation_kwargs)
            
            # Build messages
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt or self._get_default_system_prompt(include_citations)
                },
                {
                    "role": "user", 
                    "content": self._format_user_message(query, contexts)
                }
            ]
            
            # Check token budget
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            prompt_tokens = len(self.tokenizer.encode(prompt_text))
            if prompt_tokens + self.max_new_tokens > self.MAX_CONTEXT_WINDOW:
                logger_chat.warning(
                    f"Prompt too long: {prompt_tokens} tokens + {self.max_new_tokens} "
                    f"new tokens exceeds {self.MAX_CONTEXT_WINDOW}"
                )
            
            # Generate response
            with torch.no_grad():
                inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
                
                # Ensure attention_mask is present
                if 'attention_mask' not in inputs:
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                
                logger_chat.debug(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
                
                outputs = self.model.generate(
                    **inputs,
                    **gen_params
                )
                
                # Decode only new tokens
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            logger_chat.info(f"Generation complete, answer length: {len(answer)} chars")
            
            # Prepare response
            response = {
                'answer': answer.strip(),
                'query': query,
                'num_contexts_used': len(contexts),
                'contexts': contexts,
                'generation_params': gen_params,
                'prompt_length': inputs['input_ids'].shape[1]
            }
            
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            
            error_context = {
                'query_length': len(query),
                'num_contexts': len(contexts),
                'prompt_length': inputs['input_ids'].shape[1] if 'inputs' in locals() else 'unknown',
                'max_new_tokens': self.max_new_tokens,
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'generation_params': gen_params if 'gen_params' in locals() else {}
            }
            
            raise GeneratorError(
                "GPU out of memory during generation",
                context=error_context,
                original_error=e
            )
        except Exception as e:
            error_context = {
                'query_length': len(query),
                'num_contexts': len(contexts),
                'max_contexts': max_contexts,
                'include_citations': include_citations,
                'generation_params': gen_params if 'gen_params' in locals() else {},
                'step': 'unknown'  # Could track which step failed
            }
            
            # Try to determine where it failed
            if 'inputs' not in locals():
                error_context['step'] = 'tokenization'
            elif 'outputs' not in locals():
                error_context['step'] = 'model_generation'
            else:
                error_context['step'] = 'decoding'
            
            raise GeneratorError(
                f"Generation failed: {e}",
                context=error_context,
                original_error=e
            )
    
    def _prepare_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare generation parameters with defaults.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            Dict of generation parameters
        """
        params = {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'do_sample': self.temperature > 0,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        params.update(kwargs)
        return params
    
    def _get_default_system_prompt(self, include_citations: bool = True) -> str:
        """Get default system prompt for academic paper QA."""
        base_prompt = """You are a helpful research assistant specializing in biostatistics and causal inference. Your task is to answer questions based on the provided academic paper excerpts.

Instructions:
- Provide accurate, well-reasoned answers based on the given contexts
- If the contexts don't contain enough information to answer fully, acknowledge this
- Use technical terminology appropriately
- Be concise but comprehensive"""
        
        if include_citations:
            base_prompt += "\n- When referencing information, cite the relevant paper using [Paper N] notation"
        
        return base_prompt
    
    def _format_user_message(self, query: str, contexts: List[Dict]) -> str:
        """Format the user message with query and contexts."""
        message = "Context from relevant papers:\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            paper_title = ctx['metadata'].get('paper_title', 'Unknown Paper')
            text = ctx['text']
            
            message += f"[Paper {i}] {paper_title}\n"
            message += f"{text}\n\n"
        
        message += f"Question: {query}\n\n"
        message += "Please provide a detailed answer based on the contexts above."
        
        return message
    
    def _build_prompt(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True
    ) -> str:
        """
        Build the full prompt for generation (for inspection/debugging).
        
        Args:
            query: User's question
            contexts: Retrieved contexts
            system_prompt: Optional system prompt
            include_citations: Whether to include citation instructions
        
        Returns:
            Full formatted prompt string
        """
        messages = [
            {
                "role": "system", 
                "content": system_prompt or self._get_default_system_prompt(include_citations)
            },
            {
                "role": "user", 
                "content": self._format_user_message(query, contexts)
            }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_streaming(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        **generation_kwargs
    ):
        """
        Generate answer with streaming output (yields tokens as generated).
        
        Args:
            query: User's question
            contexts: Retrieved contexts
            system_prompt: Optional system prompt
            include_citations: Whether to include citations
            **generation_kwargs: Additional parameters
        
        Yields:
            Generated tokens one at a time
            
        Raises:
            GeneratorError: If streaming generation fails
        """
        try:
            logger_chat.info("Starting streaming generation")
            
            gen_params = self._prepare_generation_params(**generation_kwargs)
            
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt or self._get_default_system_prompt(include_citations)
                },
                {
                    "role": "user", 
                    "content": self._format_user_message(query, contexts)
                }
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Ensure attention_mask is present
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Setup streaming
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs_with_streamer = {
                **gen_params, 
                'streamer': streamer
            }
            
            thread = Thread(
                target=self.model.generate,
                kwargs={
                    'input_ids': inputs['input_ids'], 
                    'attention_mask': inputs['attention_mask'],
                    **generation_kwargs_with_streamer
                }
            )
            thread.start()
            
            for token in streamer:
                yield token
            
            thread.join()
            
            logger_chat.info("Streaming generation complete")
            
        except Exception as e:
            error_context = {
                'query_length': len(query),
                'num_contexts': len(contexts),
                'include_citations': include_citations,
                'generation_params': gen_params if 'gen_params' in locals() else {},
                'streaming': True
            }
            
            raise GeneratorError(
                f"Streaming generation failed: {e}",
                context=error_context,
                original_error=e
            )

    def generate_with_multiple_temperatures(
        self,
        query: str,
        contexts: List[Dict],
        temperatures: List[float],
        **kwargs
    ) -> List[Dict]:
        """Generate answers with multiple temperatures for comparison."""
        results = []
        for temp in temperatures:
            logger_chat.info(f"Generating with temperature {temp}")
            response = self.generate(
                query, 
                contexts, 
                temperature=temp,
                **kwargs
            )
            response['temperature'] = temp
            results.append(response)
        return results
    
    def cleanup(self):
        """Release GPU memory and cleanup resources."""
        logger_chat.info("Cleaning up generator resources")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger_chat.info("GPU cache cleared")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


def get_optimal_generator_settings() -> Dict[str, Any]:
    """
    Determine optimal settings based on available GPU memory.
    
    Returns:
        Dict with recommended device, max_tokens, and batch_size
    """
    if not torch.cuda.is_available():
        logger_chat.info("No CUDA available, using CPU settings")
        return {
            'device': 'cpu',
            'max_new_tokens': 512,
            'batch_size': 1
        }
    
    # Get total GPU memory in GB
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger_chat.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb >= 70:  # A100 80GB, H100
        settings = {
            'device': 'cuda',
            'max_new_tokens': 2048,
            'batch_size': 4
        }
    elif gpu_memory_gb >= 35:  # A100 40GB
        settings = {
            'device': 'cuda',
            'max_new_tokens': 1024,
            'batch_size': 2
        }
    else:  # V100 16GB or smaller
        settings = {
            'device': 'cuda',
            'max_new_tokens': 512,
            'batch_size': 1
        }
    
    logger_chat.info(f"Optimal settings: {settings}")
    return settings


def save_response(response: Dict, output_path: Path):
    """
    Save generated response to JSON file.
    
    Args:
        response: Response dict from generator
        output_path: Path to save JSON file
        
    Raises:
        IOError: If file writing fails
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable version
        save_data = {
            'answer': response['answer'],
            'query': response['query'],
            'num_contexts_used': response['num_contexts_used'],
            'generation_params': response['generation_params'],
            'prompt_length': response['prompt_length'],
            'contexts': [
                {
                    'rank': ctx.get('rank'),
                    'rerank_score': ctx.get('rerank_score'),
                    'paper_title': ctx['metadata'].get('paper_title'),
                    'text_preview': ctx['text'][:500] + '...' if len(ctx['text']) > 500 else ctx['text']
                }
                for ctx in response['contexts']
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger_chat.info(f"Response saved to {output_path}")
        
    except Exception as e:
        error_context = {
            'output_path': str(output_path),
            'output_exists': output_path.exists() if output_path else False,
            'parent_exists': output_path.parent.exists() if output_path else False,
            'response_keys': list(response.keys()) if response else []
        }
        
        logger_chat.error(
            f"Failed to save response: {e}",
            extra={'error_context': error_context},
            exc_info=True
        )
        raise IOError(f"Failed to save response to {output_path}: {e}") from e


@contextmanager
def generator_context(*args, **kwargs):
    """
    Context manager for automatic generator cleanup.
    
    Usage:
        with generator_context() as gen:
            response = gen.generate(query, contexts)
    """
    generator = QwenGenerator(*args, **kwargs)
    try:
        yield generator
    finally:
        generator.cleanup()


# Usage example
if __name__ == "__main__":
    import os
    
    # Set environment for memory efficiency
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger_chat.info("="*80)
    logger_chat.info("GENERATOR STANDALONE TEST")
    logger_chat.info("="*80)
    logger_chat.info("NOTE: This loads ONLY the generator model to test generation.")
    logger_chat.info("For full pipeline, use rag_pipeline.py instead.")
    logger_chat.info("="*80)
    
    # Stage 4: Generation (ONLY)
    logger_chat.info("Loading Generator")
    settings = get_optimal_generator_settings()
    
    with generator_context(
        device=settings['device'],
        max_new_tokens=settings['max_new_tokens'],
        temperature=0.7
    ) as generator:
        
        # Create mock contexts for testing
        logger_chat.info("Creating Mock Contexts")
        logger_chat.info("(In real usage, these come from the reranker)")
        
        mock_contexts = [
            {
                'rank': 1,
                'rerank_score': 0.95,
                'metadata': {
                    'paper_title': 'Ion Suppression in Mass Spectrometry',
                    'file_path': '/path/to/paper1.txt'
                },
                'text': """Matrix suppression is a phenomenon in mass spectrometry where components in the sample matrix interfere with the ionization process. This effect is particularly pronounced in electrospray ionization (ESI) where ion competition can significantly reduce signal intensity. The presence of salts, lipids, or other matrix components can compete with analytes for charge during the ionization process, leading to reduced sensitivity and accuracy in quantification."""
            },
            {
                'rank': 2,
                'rerank_score': 0.89,
                'metadata': {
                    'paper_title': 'Analytical Challenges in LC-MS/MS',
                    'file_path': '/path/to/paper2.txt'
                },
                'text': """Ion competition effects are a major source of matrix suppression in ESI-MS. When multiple species are present in the electrospray droplet, they compete for the limited charge available. Compounds with higher proton affinity or gas-phase basicity tend to be preferentially ionized, suppressing the signal of other analytes. This is why careful sample preparation and chromatographic separation are essential for accurate quantification."""
            },
            {
                'rank': 3,
                'rerank_score': 0.82,
                'metadata': {
                    'paper_title': 'Strategies for Minimizing Matrix Effects',
                    'file_path': '/path/to/paper3.txt'
                },
                'text': """Several strategies can be employed to minimize matrix suppression effects. Isotope dilution methods using stable isotope-labeled internal standards are particularly effective because the internal standard experiences the same matrix effects as the analyte. Sample dilution, solid-phase extraction, and improved chromatographic separation can also reduce matrix effects by separating interfering compounds from analytes of interest."""
            }
        ]
        
        query = "What is matrix suppression and how does it relate to ion competition in mass spectrometry?"
        
        logger_chat.info(f"Query: {query}")
        logger_chat.info(f"Number of mock contexts: {len(mock_contexts)}")
        
        # Generate answer
        logger_chat.info("="*80)
        logger_chat.info("GENERATING ANSWER")
        logger_chat.info("="*80)
        
        response = generator.generate(
            query=query,
            contexts=mock_contexts,
            include_citations=True,
            max_contexts=5
        )
        
        # Display result
        logger_chat.info("="*80)
        logger_chat.info("GENERATED ANSWER")
        logger_chat.info("="*80)
        logger_chat.info(response['answer'])
        logger_chat.info("="*80)
        logger_chat.info(
            f"Contexts used: {response['num_contexts_used']}, "
            f"Prompt length: {response['prompt_length']} tokens, "
            f"Temp: {response['generation_params']['temperature']}, "
            f"Top-p: {response['generation_params']['top_p']}"
        )
        logger_chat.info("="*80)
        
        # Save response
        # Save response
        output_path = Path(generator.config['paths']['outputs']) / "generator_test_response.json"
        save_response(response, output_path)
        
        # Optional: Try different temperatures
        logger_chat.info("="*80)
        logger_chat.info("BONUS: Comparing Different Temperatures")
        logger_chat.info("="*80)
        
        responses = generator.generate_with_multiple_temperatures(
            query=query,
            contexts=mock_contexts,
            temperatures=[0.3, 0.7, 1.0]
        )
        
        for resp in responses:
            logger_chat.info(f"\n--- Temperature: {resp['temperature']} ---")
            logger_chat.info(resp['answer'][:300] + "...")