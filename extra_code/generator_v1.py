import torch
import asyncio
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as AsyncSamplingParams
from typing import List, Dict, Optional, Any, AsyncGenerator
from pathlib import Path
import json
import uuid
from contextlib import contextmanager, asynccontextmanager

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
    """Qwen-based text generator for RAG pipeline Stage 4 using vLLM."""
    
    # Configuration constants
    DEFAULT_MODEL = "Qwen/Qwen3-4B-AWQ" #"Qwen/Qwen2.5-14B-Instruct-1M"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_P = 0.9
    MAX_CONTEXT_WINDOW = 8_000  # Conservative limit for Qwen2.5-1M
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = MAX_CONTEXT_WINDOW,
    ):
        """
        Initialize Qwen generator for Stage 4 (final generation) of RAG pipeline.
        
        Args:
            config_path: Path to config file
            model_name: HuggingFace model name for generation (overrides config)
            device: 'cuda' or 'cpu' (note: vLLM primarily supports CUDA)
            max_new_tokens: Maximum tokens to generate in response (overrides config)
            temperature: Sampling temperature (lower = more deterministic) (overrides config)
            top_p: Nucleus sampling parameter (overrides config)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length (context + generation)
            
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
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        logger_chat.info(f"Initializing vLLM generator: model={self.model_name}, device={device}")
        
        if device == "cpu":
            logger_chat.warning("vLLM has limited CPU support. Consider using CUDA for better performance.")
        
        try:
            self._load_model()
        except Exception as e:
            error_context = {
                'model_name': model_name,
                'device': device,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'tensor_parallel_size': tensor_parallel_size,
                'gpu_memory_utilization': gpu_memory_utilization
            }
            raise GeneratorError(
                f"Model loading failed: {e}",
                context=error_context,
                original_error=e
            )
        
        logger_chat.info(
            f"Generator initialized successfully - "
            f"max_tokens={self.max_new_tokens}, temp={self.temperature}, top_p={self.top_p}"
        )
    
    def _load_model(self):
        """Load vLLM model with error handling."""
        logger_chat.info(f"Loading vLLM model from {self.model_name}")
        
        # Build vLLM initialization kwargs
        llm_kwargs = {
            'model': self.model_name,
            'trust_remote_code': True,
            'tensor_parallel_size': self.tensor_parallel_size,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'quantization': "awq_marlin",
        }
        
        # Add max_model_len if specified
        if self.max_model_len:
            llm_kwargs['max_model_len'] = self.max_model_len
        
        # Set dtype based on device
        if self.device == 'cuda':
            llm_kwargs['dtype'] = 'float16'
        else:
            llm_kwargs['dtype'] = 'float32'
        
        print(f'MAX MODEL LEN: {self.max_model_len}')
        self.llm = LLM(**llm_kwargs)
        
        # Get tokenizer from vLLM for chat template
        self.tokenizer = self.llm.get_tokenizer()
        
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
            
            # Build sampling parameters
            sampling_params = self._prepare_sampling_params(**generation_kwargs)
            
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
            
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Check token budget
            prompt_tokens = len(self.tokenizer.encode(prompt_text))
            if prompt_tokens + self.max_new_tokens > self.MAX_CONTEXT_WINDOW:
                logger_chat.warning(
                    f"Prompt too long: {prompt_tokens} tokens + {self.max_new_tokens} "
                    f"new tokens exceeds {self.MAX_CONTEXT_WINDOW}"
                )
            
            logger_chat.debug(f"Prompt tokens: {prompt_tokens}")
            
            # Generate response using vLLM
            outputs = self.llm.generate([prompt_text], sampling_params)
            
            # Extract generated text
            answer = outputs[0].outputs[0].text
            
            logger_chat.info(f"Generation complete, answer length: {len(answer)} chars")
            
            # Prepare response
            response = {
                'answer': answer.strip(),
                'query': query,
                'num_contexts_used': len(contexts),
                'contexts': contexts,
                'generation_params': {
                    'max_tokens': sampling_params.max_tokens,
                    'temperature': sampling_params.temperature,
                    'top_p': sampling_params.top_p,
                },
                'prompt_length': prompt_tokens
            }
            
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            
            error_context = {
                'query_length': len(query),
                'num_contexts': len(contexts),
                'prompt_length': prompt_tokens if 'prompt_tokens' in locals() else 'unknown',
                'max_new_tokens': self.max_new_tokens,
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
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
                'step': 'unknown'
            }
            
            # Try to determine where it failed
            if 'prompt_text' not in locals():
                error_context['step'] = 'prompt_building'
            elif 'outputs' not in locals():
                error_context['step'] = 'model_generation'
            else:
                error_context['step'] = 'output_extraction'
            
            raise GeneratorError(
                f"Generation failed: {e}",
                context=error_context,
                original_error=e
            )
    
    def _prepare_sampling_params(self, **kwargs) -> SamplingParams:
        """
        Prepare vLLM sampling parameters with defaults.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            SamplingParams object
        """
        # Start with defaults
        params = {
            'max_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
        }
        
        # Handle temperature=0 case (greedy decoding)
        if params['temperature'] == 0:
            params['temperature'] = 0.0
            # For greedy decoding, we can leave top_p as is
        
        # Add any additional supported vLLM parameters
        supported_params = {'top_k', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'stop', 'seed'}
        for key in supported_params:
            if key in kwargs:
                params[key] = kwargs[key]
        
        return SamplingParams(**params)
    
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
            add_generation_prompt=True,
            enable_thinking=False
        )
    
    def generate_batch(
        self,
        queries: List[str],
        contexts_list: List[List[Dict]],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries in a batch (vLLM optimized).
        
        Args:
            queries: List of user questions
            contexts_list: List of context lists for each query
            system_prompt: Optional custom system prompt
            include_citations: Whether to prompt model to cite sources
            max_contexts: Maximum number of contexts to include per query
            **generation_kwargs: Additional generation parameters
        
        Returns:
            List of response dicts
            
        Raises:
            GeneratorError: If batch generation fails
        """
        try:
            if len(queries) != len(contexts_list):
                raise ValueError("Number of queries must match number of context lists")
            
            logger_chat.info(f"Batch generating responses for {len(queries)} queries")
            
            sampling_params = self._prepare_sampling_params(**generation_kwargs)
            
            # Build all prompts
            prompts = []
            prompt_tokens_list = []
            limited_contexts_list = []
            
            for query, contexts in zip(queries, contexts_list):
                contexts = contexts[:max_contexts]
                limited_contexts_list.append(contexts)
                
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
                
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(prompt_text)
                prompt_tokens_list.append(len(self.tokenizer.encode(prompt_text)))
            
            # Generate all responses in batch
            outputs = self.llm.generate(prompts, sampling_params)
            
            # Build response dicts
            responses = []
            for i, output in enumerate(outputs):
                answer = output.outputs[0].text
                responses.append({
                    'answer': answer.strip(),
                    'query': queries[i],
                    'num_contexts_used': len(limited_contexts_list[i]),
                    'contexts': limited_contexts_list[i],
                    'generation_params': {
                        'max_tokens': sampling_params.max_tokens,
                        'temperature': sampling_params.temperature,
                        'top_p': sampling_params.top_p,
                    },
                    'prompt_length': prompt_tokens_list[i]
                })
            
            logger_chat.info(f"Batch generation complete for {len(queries)} queries")
            return responses
            
        except Exception as e:
            error_context = {
                'num_queries': len(queries),
                'max_contexts': max_contexts,
                'include_citations': include_citations,
            }
            raise GeneratorError(
                f"Batch generation failed: {e}",
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
        
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger_chat.info("GPU cache cleared")


class AsyncQwenGenerator:
    """Async Qwen-based text generator using vLLM AsyncLLMEngine for streaming."""
    
    # Configuration constants
    DEFAULT_MODEL = "Qwen/Qwen3-4B-AWQ"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_P = 0.9
    MAX_CONTEXT_WINDOW = 8_000
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = DEFAULT_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = MAX_CONTEXT_WINDOW,
    ):
        """
        Initialize async Qwen generator with streaming support.
        
        Args:
            config_path: Path to config file
            model_name: HuggingFace model name for generation (overrides config)
            device: 'cuda' or 'cpu'
            max_new_tokens: Maximum tokens to generate in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
        """
        self.config = load_config(config_path)
        
        self.model_name = model_name or self.config['models']['generator']
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_TOKENS
        self.temperature = temperature if temperature is not None else self.config['generation']['temperature']
        self.top_p = top_p or self.DEFAULT_TOP_P
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer = None
        
        logger_chat.info(f"Initializing async vLLM generator: model={self.model_name}")
    
    async def start(self):
        """Start the async engine."""
        if self.engine is not None:
            logger_chat.warning("Engine already started")
            return
        
        logger_chat.info(f"Starting AsyncLLMEngine for {self.model_name}")
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype='float16' if self.device == 'cuda' else 'float32',
            max_model_len=self.max_model_len,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Get tokenizer
        self.tokenizer = await self.engine.get_tokenizer()
        
        logger_chat.info("AsyncLLMEngine started successfully")
    
    def stop(self):
        """Stop the async engine and cleanup."""
        if self.engine is not None:
            logger_chat.info("Stopping AsyncLLMEngine")
            # vLLM 0.8.x cleanup
            if hasattr(self.engine, 'shutdown'):
                self.engine.shutdown()
            self.engine = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger_chat.info("AsyncLLMEngine stopped")
    
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
    
    def _prepare_sampling_params(self, **kwargs) -> SamplingParams:
        """Prepare vLLM sampling parameters."""
        params = {
            'max_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'top_p': kwargs.get('top_p', self.top_p),
        }
        
        supported_params = {'top_k', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'stop', 'seed'}
        for key in supported_params:
            if key in kwargs:
                params[key] = kwargs[key]
        
        return SamplingParams(**params)
    
    async def generate_streaming(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5,
        **generation_kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate answer with streaming output (yields tokens as generated).
        
        Args:
            query: User's question
            contexts: Retrieved contexts
            system_prompt: Optional system prompt
            include_citations: Whether to include citations
            max_contexts: Maximum contexts to use
            **generation_kwargs: Additional parameters
        
        Yields:
            Generated tokens one at a time
        """
        if self.engine is None:
            raise GeneratorError("Engine not started. Call await generator.start() first.")
        
        try:
            logger_chat.info("Starting streaming generation")
            
            contexts = contexts[:max_contexts]
            sampling_params = self._prepare_sampling_params(**generation_kwargs)
            
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
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            request_id = str(uuid.uuid4())
            
            # Stream results
            results_generator = self.engine.generate(
                prompt_text,
                sampling_params,
                request_id=request_id
            )
            
            previous_text = ""
            async for request_output in results_generator:
                # Get the new text since last iteration
                current_text = request_output.outputs[0].text
                new_text = current_text[len(previous_text):]
                previous_text = current_text
                
                if new_text:
                    yield new_text
            
            logger_chat.info("Streaming generation complete")
            
        except Exception as e:
            error_context = {
                'query_length': len(query),
                'num_contexts': len(contexts),
                'streaming': True
            }
            raise GeneratorError(
                f"Streaming generation failed: {e}",
                context=error_context,
                original_error=e
            )
    
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
        Generate answer (non-streaming, collects full response).
        
        Args:
            query: User's question
            contexts: List of context dicts from reranker
            system_prompt: Optional custom system prompt
            include_citations: Whether to prompt model to cite sources
            max_contexts: Maximum number of contexts to include
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dict containing generated answer and metadata
        """
        if self.engine is None:
            raise GeneratorError("Engine not started. Call await generator.start() first.")
        
        try:
            logger_chat.info(
                f"Generating response for query (length={len(query)}), "
                f"using {len(contexts)} contexts"
            )
            
            contexts = contexts[:max_contexts]
            sampling_params = self._prepare_sampling_params(**generation_kwargs)
            
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
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            prompt_tokens = len(self.tokenizer.encode(prompt_text))
            request_id = str(uuid.uuid4())
            
            # Collect full response
            final_output = None
            async for request_output in self.engine.generate(
                prompt_text,
                sampling_params,
                request_id=request_id
            ):
                final_output = request_output
            
            answer = final_output.outputs[0].text
            
            logger_chat.info(f"Generation complete, answer length: {len(answer)} chars")
            
            return {
                'answer': answer.strip(),
                'query': query,
                'num_contexts_used': len(contexts),
                'contexts': contexts,
                'generation_params': {
                    'max_tokens': sampling_params.max_tokens,
                    'temperature': sampling_params.temperature,
                    'top_p': sampling_params.top_p,
                },
                'prompt_length': prompt_tokens
            }
            
        except Exception as e:
            error_context = {
                'query_length': len(query),
                'num_contexts': len(contexts),
                'max_contexts': max_contexts,
            }
            raise GeneratorError(
                f"Generation failed: {e}",
                context=error_context,
                original_error=e
            )
    
    async def generate_batch_streaming(
        self,
        queries: List[str],
        contexts_list: List[List[Dict]],
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        max_contexts: int = 5,
        **generation_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate answers for multiple queries, yielding results as they complete.
        
        Args:
            queries: List of user questions
            contexts_list: List of context lists for each query
            system_prompt: Optional custom system prompt
            include_citations: Whether to prompt model to cite sources
            max_contexts: Maximum number of contexts per query
            **generation_kwargs: Additional generation parameters
        
        Yields:
            Dict with query index and streaming tokens or final result
        """
        if self.engine is None:
            raise GeneratorError("Engine not started. Call await generator.start() first.")
        
        if len(queries) != len(contexts_list):
            raise ValueError("Number of queries must match number of context lists")
        
        logger_chat.info(f"Batch streaming generation for {len(queries)} queries")
        
        sampling_params = self._prepare_sampling_params(**generation_kwargs)
        
        # Build all prompts and create request IDs
        prompts = []
        request_ids = []
        limited_contexts = []
        
        for i, (query, contexts) in enumerate(zip(queries, contexts_list)):
            contexts = contexts[:max_contexts]
            limited_contexts.append(contexts)
            
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
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt_text)
            request_ids.append(str(uuid.uuid4()))
        
        # Submit all requests
        generators = []
        for prompt, request_id in zip(prompts, request_ids):
            gen = self.engine.generate(prompt, sampling_params, request_id=request_id)
            generators.append(gen)
        
        # Track previous text for each request to compute deltas
        previous_texts = [""] * len(queries)
        
        # Process all generators concurrently
        async def process_generator(idx: int, gen):
            nonlocal previous_texts
            async for output in gen:
                current_text = output.outputs[0].text
                new_text = current_text[len(previous_texts[idx]):]
                previous_texts[idx] = current_text
                
                if output.finished:
                    yield {
                        'index': idx,
                        'type': 'complete',
                        'answer': current_text.strip(),
                        'query': queries[idx],
                        'num_contexts_used': len(limited_contexts[idx]),
                        'contexts': limited_contexts[idx],
                    }
                elif new_text:
                    yield {
                        'index': idx,
                        'type': 'token',
                        'token': new_text
                    }
        
        # Merge all async generators
        async def merge_generators():
            tasks = [process_generator(i, gen) for i, gen in enumerate(generators)]
            
            async def consume(idx, task):
                results = []
                async for item in task:
                    results.append(item)
                return results
            
            all_results = await asyncio.gather(*[consume(i, t) for i, t in enumerate(tasks)])
            for results in all_results:
                for item in results:
                    yield item
        
        async for item in merge_generators():
            yield item
    
    def cleanup(self):
        """Release GPU memory and cleanup resources."""
        logger_chat.info("Cleaning up generator resources")
        
        if hasattr(self, 'llm'):
            del self.llm
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
        Dict with recommended device, max_tokens, batch_size, and vLLM-specific settings
    """
    if not torch.cuda.is_available():
        logger_chat.info("No CUDA available, using CPU settings")
        return {
            'device': 'cpu',
            'max_new_tokens': 512,
            'batch_size': 1,
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.9,
        }
    
    # Get total GPU memory in GB
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    num_gpus = torch.cuda.device_count()
    logger_chat.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB, GPUs: {num_gpus}")
    
    if gpu_memory_gb >= 70:  # A100 80GB, H100
        settings = {
            'device': 'cuda',
            'max_new_tokens': 2048,
            'batch_size': 8,
            'tensor_parallel_size': min(num_gpus, 2),
            'gpu_memory_utilization': 0.9,
        }
    elif gpu_memory_gb >= 35:  # A100 40GB
        settings = {
            'device': 'cuda',
            'max_new_tokens': 1024,
            'batch_size': 4,
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.9,
        }
    else:  # V100 16GB or smaller
        settings = {
            'device': 'cuda',
            'max_new_tokens': 512,
            'batch_size': 2,
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.85,
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


@asynccontextmanager
async def async_generator_context(*args, **kwargs):
    """
    Async context manager for AsyncQwenGenerator with automatic cleanup.
    
    Usage:
        async with async_generator_context() as gen:
            async for token in gen.generate_streaming(query, contexts):
                print(token, end='', flush=True)
    """
    generator = AsyncQwenGenerator(*args, **kwargs)
    try:
        await generator.start()
        yield generator
    finally:
        generator.stop()


# Usage example
# if __name__ == "__main__":
#     import os
    
#     # Set environment for memory efficiency
#     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
#     logger_chat.info("="*80)
#     logger_chat.info("GENERATOR STANDALONE TEST (vLLM)")
#     logger_chat.info("="*80)
#     logger_chat.info("NOTE: This loads ONLY the generator model to test generation.")
#     logger_chat.info("For full pipeline, use rag_pipeline.py instead.")
#     logger_chat.info("="*80)
    
#     # Stage 4: Generation (ONLY)
#     logger_chat.info("Loading Generator")
#     settings = get_optimal_generator_settings()
    
#     # Create mock contexts for testing
#     mock_contexts = [
#         {
#             'rank': 1,
#             'rerank_score': 0.95,
#             'metadata': {
#                 'paper_title': 'Ion Suppression in Mass Spectrometry',
#                 'file_path': '/path/to/paper1.txt'
#             },
#             'text': """Matrix suppression is a phenomenon in mass spectrometry where components in the sample matrix interfere with the ionization process. This effect is particularly pronounced in electrospray ionization (ESI) where ion competition can significantly reduce signal intensity. The presence of salts, lipids, or other matrix components can compete with analytes for charge during the ionization process, leading to reduced sensitivity and accuracy in quantification."""
#         },
#         {
#             'rank': 2,
#             'rerank_score': 0.89,
#             'metadata': {
#                 'paper_title': 'Analytical Challenges in LC-MS/MS',
#                 'file_path': '/path/to/paper2.txt'
#             },
#             'text': """Ion competition effects are a major source of matrix suppression in ESI-MS. When multiple species are present in the electrospray droplet, they compete for the limited charge available. Compounds with higher proton affinity or gas-phase basicity tend to be preferentially ionized, suppressing the signal of other analytes. This is why careful sample preparation and chromatographic separation are essential for accurate quantification."""
#         },
#         {
#             'rank': 3,
#             'rerank_score': 0.82,
#             'metadata': {
#                 'paper_title': 'Strategies for Minimizing Matrix Effects',
#                 'file_path': '/path/to/paper3.txt'
#             },
#             'text': """Several strategies can be employed to minimize matrix suppression effects. Isotope dilution methods using stable isotope-labeled internal standards are particularly effective because the internal standard experiences the same matrix effects as the analyte. Sample dilution, solid-phase extraction, and improved chromatographic separation can also reduce matrix effects by separating interfering compounds from analytes of interest."""
#         }
#     ]
    
#     query = "What is matrix suppression and how does it relate to ion competition in mass spectrometry?"
    
#     # ==========================================
#     # Test 1: Synchronous generation (QwenGenerator)
#     # ==========================================
#     logger_chat.info("="*80)
#     logger_chat.info("TEST 1: SYNCHRONOUS GENERATION")
#     logger_chat.info("="*80)
    
#     with generator_context(
#         device=settings['device'],
#         max_new_tokens=settings['max_new_tokens'],
#         temperature=0.7,
#         tensor_parallel_size=settings['tensor_parallel_size'],
#         gpu_memory_utilization=settings['gpu_memory_utilization'],
#     ) as generator:
        
#         logger_chat.info(f"Query: {query}")
#         logger_chat.info(f"Number of mock contexts: {len(mock_contexts)}")
        
#         # Generate answer
#         response = generator.generate(
#             query=query,
#             contexts=mock_contexts,
#             include_citations=True,
#             max_contexts=5
#         )
        
#         # Display result
#         logger_chat.info("="*80)
#         logger_chat.info("GENERATED ANSWER (Sync)")
#         logger_chat.info("="*80)
#         logger_chat.info(response['answer'])
#         logger_chat.info("="*80)
#         logger_chat.info(
#             f"Contexts used: {response['num_contexts_used']}, "
#             f"Prompt length: {response['prompt_length']} tokens, "
#             f"Temp: {response['generation_params']['temperature']}, "
#             f"Top-p: {response['generation_params']['top_p']}"
#         )
        
#         # Save response
#         output_path = Path(generator.config['paths']['outputs']) / "generator_test_response.json"
#         save_response(response, output_path)
        
#         # Test batch generation
#         logger_chat.info("="*80)
#         logger_chat.info("TEST: BATCH GENERATION")
#         logger_chat.info("="*80)
        
#         queries = [query, "What strategies can minimize matrix effects?"]
#         contexts_list = [mock_contexts, mock_contexts]
        
#         batch_responses = generator.generate_batch(
#             queries=queries,
#             contexts_list=contexts_list,
#             include_citations=True
#         )
        
#         for i, resp in enumerate(batch_responses):
#             logger_chat.info(f"\n--- Query {i+1} ---")
#             logger_chat.info(resp['answer'][:300] + "...")
    
#     # ==========================================
#     # Test 2: Async streaming generation (AsyncQwenGenerator)
#     # ==========================================
#     logger_chat.info("="*80)
#     logger_chat.info("TEST 2: ASYNC STREAMING GENERATION")
#     logger_chat.info("="*80)
    
#     async def test_async_streaming():
#         async with async_generator_context(
#             device=settings['device'],
#             max_new_tokens=settings['max_new_tokens'],
#             temperature=0.7,
#             tensor_parallel_size=settings['tensor_parallel_size'],
#             gpu_memory_utilization=settings['gpu_memory_utilization'],
#         ) as generator:
            
#             logger_chat.info("Streaming response:")
#             print("\n", end='')
            
#             full_response = ""
#             async for token in generator.generate_streaming(
#                 query=query,
#                 contexts=mock_contexts,
#                 include_citations=True,
#                 max_contexts=5
#             ):
#                 print(token, end='', flush=True)
#                 full_response += token
            
#             print("\n")
#             logger_chat.info(f"Streaming complete. Total length: {len(full_response)} chars")
            
#             # Also test non-streaming async generate
#             logger_chat.info("="*80)
#             logger_chat.info("TEST: ASYNC NON-STREAMING")
#             logger_chat.info("="*80)
            
#             response = await generator.generate(
#                 query=query,
#                 contexts=mock_contexts,
#                 include_citations=True
#             )
#             logger_chat.info(f"Async response length: {len(response['answer'])} chars")
    
#     # Run async tests
#     asyncio.run(test_async_streaming())
    
#     logger_chat.info("="*80)
#     logger_chat.info("ALL TESTS COMPLETE")
#     logger_chat.info("="*80)

# Usage example 2
if __name__ == "__main__":
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    async def chat_loop():
        # Auto-detect best GPU settings
        settings = get_optimal_generator_settings()
        
        # Use the async context manager to handle start/stop/cleanup
        async with async_generator_context(
            device=settings['device'],
            max_new_tokens=settings['max_new_tokens'],
            tensor_parallel_size=settings['tensor_parallel_size'],
            gpu_memory_utilization=settings['gpu_memory_utilization'],
        ) as generator:
            
            print("\n--- Model Ready (Type 'q' to quit) ---\n")
            
            while True:
                # Use standard input for simplicity
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'q':
                    print("Exiting...")
                    break
                
                if not user_input:
                    continue

                print("AI: ", end="", flush=True)
                
                # Directly stream tokens from your generator
                # Note: We pass an empty list for contexts to focus on direct chat
                async for token in generator.generate_streaming(
                    query=user_input,
                    contexts=[], 
                    include_citations=False
                ):
                    print(token, end="", flush=True)
                print("\n")

    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        pass