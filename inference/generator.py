"""
Lean Qwen Generator for RAG Pipeline
Focuses on async streaming with proper error handling and separation of concerns.
Refactored to support unified structured logging.
"""

import torch
import asyncio
import time
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from typing import List, Dict, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager
import uuid

# Updated logger imports to match new logger.py interface
from utils.logger import get_logger, log_stage_start, log_generation_metrics
from utils.config_loader import load_config

# Initialize component-specific logger
logger = get_logger(__name__)


class GeneratorError(Exception):
    """Generator-specific exception with context."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        
        # Log error using the structured logger
        logger.error(
            f"GeneratorError: {message}",
            extra={
                'original_error': str(original_error) if original_error else None,
                'error_type': type(original_error).__name__ if original_error else None
            },
            exc_info=original_error
        )


class AsyncQwenGenerator:
    """Async Qwen generator with streaming support for RAG chatbot."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.4,
        top_p: Optional[float] = 0.7,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = 8_000,
    ):
        """
        Initialize async Qwen generator.
        
        Args:
            config_path: Path to config file
            model_name: HuggingFace model name (overrides config)
            max_new_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length
        """
        
        # Load configuration
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p or self.DEFAULT_TOP_P
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer = None
        
        logger.info(
            f"Generator config loaded: model={self.model_name}, "
            f"max_tokens={self.max_new_tokens}, temp={self.temperature}"
        )
    
    async def initialize(self):
        """Initialize the async engine. Call this at app startup."""
        if self.engine is not None:
            logger.warning("Engine already initialized")
            return
        
        logger.info(f"Initializing AsyncLLMEngine with {self.model_name}")
        
        try:
            # Build engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype='float16' if torch.cuda.is_available() else 'float32',
                max_model_len=self.max_model_len,
                # Remove hardcoded quantization - only add if model requires it
                # quantization="awq_marlin",  # Only if using AWQ models
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Get tokenizer for prompt formatting
            self.tokenizer = await self.engine.get_tokenizer()
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Engine initialized. GPU memory: {memory_allocated:.2f} GB")
            else:
                logger.info("Engine initialized on CPU")
                
        except Exception as e:
            error_msg = f"Failed to initialize engine: {str(e)}"
            logger.error(error_msg)
            raise GeneratorError(error_msg, original_error=e)
    
    async def cleanup(self):
        """Clean up engine resources. Call this at app shutdown."""
        if self.engine is not None:
            logger.info("Shutting down AsyncLLMEngine")
            # vLLM's AsyncEngine doesn't have explicit cleanup, but we can set to None
            self.engine = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
    
    def _format_prompt(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5
    ) -> str:
        """
        Format prompt using chat template.
        """
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful research assistant. Answer questions based on the "
                "provided context from academic papers. "
            )
            if include_citations:
                system_prompt += (
                    "Cite specific sources when making claims by referencing the paper titles. "
                )
        
        # Limit contexts
        selected_contexts = contexts[:max_contexts] if max_contexts > 0 else contexts
        
        # Build context string
        context_parts = []
        for i, ctx in enumerate(selected_contexts, 1):
            paper_title = ctx.get('metadata', {}).get('paper_title', f'Source {i}')
            text = ctx.get('text', '')
            context_parts.append(f"[{i}] {paper_title}\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        # Build user message
        if context_str:
            user_message = (
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer the question based on the context provided. "
            )
            if include_citations:
                user_message += "Cite sources using [number] notation."
        else:
            user_message = query
        
        # Format with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        return prompt
    
    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        """Create sampling parameters for generation."""
        return SamplingParams(
            temperature=kwargs.get('temperature', self.temperature),
            top_p=kwargs.get('top_p', self.top_p),
            max_tokens=kwargs.get('max_tokens', self.max_new_tokens),
            skip_special_tokens=True,
        )
    
    async def generate_streaming(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5,
        request_id:str = 'abc', #None #replace to none later
        **generation_kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated tokens as they're produced.
        Captures metrics (TTFT, total time, tokens) for the structured logger.
        
        Yields:
            Generated text tokens
        """
        if self.engine is None:
            raise GeneratorError("Engine not initialized. Call initialize() first.")
        
        # Track timing for logging
        start_time = time.perf_counter()
        ttft_ms = None
        
        try:
            # Log stage start
            log_stage_start(logger, "generation", model=self.model_name)
            
            # Format prompt
            prompt = self._format_prompt(
                query=query,
                contexts=contexts,
                system_prompt=system_prompt,
                include_citations=include_citations,
                max_contexts=max_contexts
            )
            
            # Create sampling params
            sampling_params = self._create_sampling_params(**generation_kwargs)
            
            # Track generated text for delta calculation and stats
            previous_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            async for request_output in self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id
            ):
                # Calculate TTFT on first iteration
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start_time) * 1000
                
                # Get current full text
                current_text = request_output.outputs[0].text
                
                # Capture accurate token counts from vLLM output
                if hasattr(request_output, 'prompt_token_ids'):
                    prompt_tokens = len(request_output.prompt_token_ids)
                if hasattr(request_output.outputs[0], 'token_ids'):
                    completion_tokens = len(request_output.outputs[0].token_ids)
                
                # Calculate delta (new tokens only)
                if current_text.startswith(previous_text):
                    delta = current_text[len(previous_text):]
                    if delta:
                        yield delta
                        previous_text = current_text
                else:
                    # Fallback: yield entire current text if mismatch
                    yield current_text
                    previous_text = current_text
            
            # Log final metrics on completion
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            
            log_generation_metrics(
                logger,
                duration_ms=total_duration_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
                model=self.model_name
            )
            
        except Exception as e:
            error_msg = f"Streaming generation failed: {str(e)}"
            logger.error(error_msg)
            raise GeneratorError(error_msg, original_error=e)
    
    async def generate(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate complete response (non-streaming).
        Useful for testing and cases where full response is needed at once.
        """
        full_response = ""
        
        # Re-use streaming method to ensure logging logic is consistent
        async for token in self.generate_streaming(
            query=query,
            contexts=contexts,
            system_prompt=system_prompt,
            include_citations=include_citations,
            max_contexts=max_contexts,
            **generation_kwargs
        ):
            full_response += token
        
        return {
            'answer': full_response,
            'num_contexts_used': min(len(contexts), max_contexts),
            'query': query,
            'generation_params': {
                'temperature': generation_kwargs.get('temperature', self.temperature),
                'top_p': generation_kwargs.get('top_p', self.top_p),
                'max_tokens': generation_kwargs.get('max_tokens', self.max_new_tokens)
            }
        }


@asynccontextmanager
async def async_generator_context(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
    **init_kwargs
):
    """
    Context manager for AsyncQwenGenerator with automatic cleanup.
    Use this in FastAPI lifespan for proper startup/shutdown.
    """
    generator = AsyncQwenGenerator(
        config_path=config_path,
        model_name=model_name,
        **init_kwargs
    )
    
    try:
        await generator.initialize()
        yield generator
    finally:
        await generator.cleanup()


# Maybe needed in the backend layer?
def format_simple_prompt(query: str, contexts: List[Dict], max_contexts: int = 5) -> str:
    """
    Simple prompt formatting without tokenizer (for API layer previews).
    """
    selected = contexts[:max_contexts]
    
    context_parts = []
    for i, ctx in enumerate(selected, 1):
        title = ctx.get('metadata', {}).get('paper_title', f'Source {i}')
        text = ctx.get('text', '')
        context_parts.append(f"[{i}] {title}\n{text}")
    
    context_str = "\n\n".join(context_parts)
    
    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer based on the context and cite sources using [number]."
    )


def estimate_prompt_tokens(prompt: str) -> int:
    """
    Rough token estimate (4 chars ≈ 1 token for English).
    For accurate counts, use the actual tokenizer.
    """
    return len(prompt) // 4

if __name__ == "__main__":
    import os
    # Set context variables for testing the logger
    from utils.logger import set_request_context, clear_request_context
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    async def test_streaming():
        """Test streaming generation with mock contexts."""
        
        # Mimic a request context for logging
        set_request_context(req_id="test-gen-1", conversation_id="test-conv")
        
        mock_contexts = [
            {
                'rank': 1,
                'rerank_score': 0.95,
                'metadata': {
                    'paper_title': 'Ion Suppression in Mass Spectrometry',
                    'file_path': '/path/to/paper1.txt'
                },
                'text': (
                    "Matrix suppression is a phenomenon in mass spectrometry where "
                    "components in the sample matrix interfere with the ionization process."
                )
            }
        ]
        
        query = "What is matrix suppression?"
        
        async with async_generator_context(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.65,
        ) as generator:
            
            print(f"\nQuery: {query}\n")
            print("Response: ", end="", flush=True)
            
            async for token in generator.generate_streaming(
                query=query,
                contexts=mock_contexts,
                include_citations=True
            ):
                print(token, end="", flush=True)
            
            print("\n\n✓ Streaming test complete")
            clear_request_context()
    
    async def test_chat_loop():
        """Interactive chat loop for manual testing."""
        set_request_context(req_id="manual-chat")
        
        async with async_generator_context(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        ) as generator:
            
            print("\n=== Chat Loop (type 'q' to quit) ===\n")
            
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'q':
                    break
                
                if not user_input:
                    continue
                
                print("AI: ", end="", flush=True)
                
                async for token in generator.generate_streaming(
                    query=user_input,
                    contexts=[],  # Direct chat without RAG
                    include_citations=False
                ):
                    print(token, end="", flush=True)
                
                print("\n")
        
        clear_request_context()
    
    # Run tests
    try:
        asyncio.run(test_streaming())
        # asyncio.run(test_chat_loop())  # Uncomment for interactive testing
    except KeyboardInterrupt:
        print("\nInterrupted")