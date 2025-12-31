#!/usr/bin/env python3
"""
RAG Evaluation Dataset Generator
================================
Generates synthetic test questions for RAG pipeline evaluation using local LLMs.

This script uses vLLM served as an OpenAI-compatible API to avoid in-process conflicts.

Usage:
1. Start vLLM server first (see instructions below)
2. Run this script

To start vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8000 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --trust-remote-code

Or use the helper script: bash start_vllm_server.sh
"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import hashlib

# Disable proxies for local vLLM server connection
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ['NO_PROXY'] = '0.0.0.0,127.0.0.1'

# Document processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# OpenAI client for vLLM
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # Paths
    pdf_dir: str = "/scratch/sathishbabu.ki/data_files/input_pdf"
    output_file: str = "rag_eval_dataset.json"
    checkpoint_file: str = "generation_checkpoint.json"
    
    # vLLM server settings
    vllm_base_url: str = "http://0.0.0.0:8000/v1"
    model_name: str = "/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    
    # Embedding model
    embed_model: str = "BAAI/bge-large-en-v1.5"
    
    # Generation settings
    num_corpus_questions: int = 350
    num_domain_questions: int = 50
    
    # Chunk settings
    chunk_size: int = 1500
    chunk_overlap: int = 200
    
    # LLM generation params
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Processing
    batch_size: int = 5  # Process chunks in batches
    max_retries: int = 3
    retry_delay: float = 2.0

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class GeneratedQuestion:
    question: str
    ground_truth: str  # Expected answer based on context
    contexts: List[str]  # Source contexts used
    question_type: str  # simple, reasoning, multi_context
    source_files: List[str]
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class DomainQuestion:
    question: str
    question_type: str = "domain"
    topic: str = ""
    difficulty: str = "medium"
    answerable: bool = True  # Whether it's expected to be answerable from corpus

# ============================================================================
# DOCUMENT LOADING
# ============================================================================
def load_documents(pdf_dir: str) -> List[Document]:
    """Load all PDFs from directory."""
    documents = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    print(f"\n📁 Loading {len(pdf_files)} PDFs from {pdf_dir}...")
    
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source"] = pdf_path.name
                doc.metadata["file_path"] = str(pdf_path)
                
            documents.extend(docs)
        except Exception as e:
            print(f"  ⚠ Failed to load {pdf_path.name}: {e}")
    
    print(f"✓ Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_hash"] = hashlib.md5(
            chunk.page_content.encode()
        ).hexdigest()[:8]
    
    print(f"✓ Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

# ============================================================================
# LLM CLIENT
# ============================================================================

# Disable proxies for local connections
import httpx

class VLLMClient:
    """Client for vLLM OpenAI-compatible server."""
    
    def __init__(self, base_url: str, model_name: str, config: Config):
        # Create httpx client with no proxies and reasonable timeout
        http_client = httpx.Client(
            # proxies=None,  # Explicitly disable proxies
            timeout=httpx.Timeout(300.0, connect=30.0),  # 5 min timeout for generation
        )
        
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",  # vLLM doesn't require API key
            http_client=http_client,
        )
        self.model_name = model_name
        self.config = config
        self._check_connection()
    
    def _check_connection(self):
        """Verify server is running and get actual model name."""
        try:
            print(f"  Attempting to connect to {self.client.base_url}...")
            models = self.client.models.list()
            available = [m.id for m in models.data]
            print(f"✓ Connected to vLLM server. Available models: {available}")
            
            # Auto-detect model name if needed
            if self.model_name not in available and len(available) > 0:
                old_name = self.model_name
                self.model_name = available[0]
                print(f"  ⚠ Model name adjusted: {old_name[-50:]}... → {self.model_name[-50:]}")
                
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to vLLM server at {self.client.base_url}. "
                f"Please start it first with:\n"
                f"  python -m vllm.entrypoints.openai.api_server --model {self.model_name} --port 8000\n"
                f"Error: {e}"
            )
    
    def generate(self, prompt: str, system_prompt: str = None, 
                 temperature: float = None, max_tokens: int = None) -> str:
        """Generate text from prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise e
        return ""

# ============================================================================
# QUESTION GENERATORS
# ============================================================================

SYSTEM_PROMPT = """You are an expert at creating high-quality evaluation questions for RAG (Retrieval-Augmented Generation) systems. 
Your questions should:
1. Be answerable from the provided context
2. Test comprehension, not just keyword matching
3. Vary in complexity and type
4. Be clear and unambiguous"""

def generate_simple_question(client: VLLMClient, context: str, source: str) -> Optional[GeneratedQuestion]:
    """Generate a simple factual question from context."""
    
    prompt = f"""Based on the following context, generate ONE simple factual question that can be directly answered from the text.

Context:
{context[:2000]}

Respond in this exact JSON format:
{{"question": "Your question here", "answer": "The answer from the context", "key_phrase": "Key phrase from context that answers it"}}

Generate only the JSON, nothing else."""

    try:
        response = client.generate(prompt, SYSTEM_PROMPT, temperature=0.5)
        
        # Clean and parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        data = json.loads(response)
        
        return GeneratedQuestion(
            question=data["question"],
            ground_truth=data["answer"],
            contexts=[context],
            question_type="simple",
            source_files=[source],
            difficulty="easy"
        )
    except Exception as e:
        return None


def generate_reasoning_question(client: VLLMClient, context: str, source: str) -> Optional[GeneratedQuestion]:
    """Generate a question requiring reasoning/inference."""
    
    prompt = f"""Based on the following context, generate ONE question that requires reasoning or inference to answer.
The question should NOT be directly answered by a single sentence, but require understanding multiple parts of the text.

Context:
{context[:2000]}

Respond in this exact JSON format:
{{"question": "Your reasoning question", "answer": "Detailed answer requiring inference", "reasoning": "Brief explanation of reasoning needed"}}

Generate only the JSON, nothing else."""

    try:
        response = client.generate(prompt, SYSTEM_PROMPT, temperature=0.6)
        
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        data = json.loads(response)
        
        return GeneratedQuestion(
            question=data["question"],
            ground_truth=data["answer"],
            contexts=[context],
            question_type="reasoning",
            source_files=[source],
            difficulty="medium",
            metadata={"reasoning": data.get("reasoning", "")}
        )
    except Exception as e:
        return None


def generate_multi_context_question(client: VLLMClient, contexts: List[str], 
                                     sources: List[str]) -> Optional[GeneratedQuestion]:
    """Generate a question requiring multiple contexts to answer."""
    
    combined = "\n\n---\n\n".join([f"[Context {i+1}]:\n{c[:1000]}" for i, c in enumerate(contexts[:3])])
    
    prompt = f"""Based on the following multiple contexts, generate ONE question that requires information from AT LEAST TWO contexts to fully answer.

{combined}

Respond in this exact JSON format:
{{"question": "Your multi-context question", "answer": "Complete answer using multiple contexts", "contexts_used": [1, 2]}}

Generate only the JSON, nothing else."""

    try:
        response = client.generate(prompt, SYSTEM_PROMPT, temperature=0.6)
        
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        data = json.loads(response)
        
        return GeneratedQuestion(
            question=data["question"],
            ground_truth=data["answer"],
            contexts=contexts,
            question_type="multi_context",
            source_files=sources,
            difficulty="hard",
            metadata={"contexts_used": data.get("contexts_used", [])}
        )
    except Exception as e:
        return None


def generate_domain_questions(client: VLLMClient, num_questions: int = 50) -> List[DomainQuestion]:
    """Generate domain-specific questions for biostatistics/causal inference."""
    
    print(f"\n🔬 Generating {num_questions} domain-specific questions...")
    
    topics = [
        "propensity score methods",
        "instrumental variables",
        "difference-in-differences",
        "regression discontinuity",
        "SUTVA assumption",
        "confounding adjustment",
        "selection bias",
        "mediation analysis",
        "sensitivity analysis",
        "matching estimators",
        "inverse probability weighting",
        "doubly robust estimation",
        "causal graphs and DAGs",
        "randomization inference",
        "survival analysis causality"
    ]
    
    all_questions = []
    questions_per_batch = 10
    
    for batch_start in range(0, num_questions, questions_per_batch):
        batch_topics = random.sample(topics, min(5, len(topics)))
        
        prompt = f"""Generate {questions_per_batch} challenging questions about biostatistics and causal inference.

Focus on these topics: {', '.join(batch_topics)}

Requirements:
1. Mix of conceptual questions (what/why) and methodological questions (how/when)
2. Include some questions that test common misconceptions
3. Vary difficulty from intermediate to expert
4. Some questions should be very specific (likely NOT answerable from a general paper collection)

Output as a JSON array:
[
  {{"question": "...", "topic": "...", "difficulty": "medium|hard|expert", "answerable_from_papers": true|false}},
  ...
]

Generate exactly {questions_per_batch} questions. Output ONLY the JSON array."""

        try:
            response = client.generate(prompt, temperature=0.8, max_tokens=3000)
            
            # Parse response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            questions = json.loads(response)
            
            for q in questions:
                all_questions.append(DomainQuestion(
                    question=q["question"],
                    topic=q.get("topic", ""),
                    difficulty=q.get("difficulty", "medium"),
                    answerable=q.get("answerable_from_papers", True)
                ))
                
        except Exception as e:
            print(f"  ⚠ Batch generation failed: {e}")
            continue
        
        if len(all_questions) >= num_questions:
            break
    
    print(f"✓ Generated {len(all_questions)} domain questions")
    return all_questions[:num_questions]

# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load generation checkpoint if exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"generated_questions": [], "processed_chunks": [], "domain_questions": []}


def save_checkpoint(checkpoint_file: str, data: Dict):
    """Save generation checkpoint."""
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f)


def generate_corpus_questions(client: VLLMClient, chunks: List[Document], 
                               num_questions: int, checkpoint_file: str) -> List[GeneratedQuestion]:
    """Generate questions from document corpus."""
    
    print(f"\n📝 Generating {num_questions} corpus-based questions...")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    questions = [GeneratedQuestion(**q) for q in checkpoint.get("generated_questions", [])]
    processed_chunks = set(checkpoint.get("processed_chunks", []))
    
    print(f"  Resuming from checkpoint: {len(questions)} questions already generated")
    
    # Distribution of question types
    target_simple = int(num_questions * 0.5)
    target_reasoning = int(num_questions * 0.3)
    target_multi = num_questions - target_simple - target_reasoning
    
    # Count existing by type
    simple_count = sum(1 for q in questions if q.question_type == "simple")
    reasoning_count = sum(1 for q in questions if q.question_type == "reasoning")
    multi_count = sum(1 for q in questions if q.question_type == "multi_context")
    
    # Shuffle chunks for variety
    available_chunks = [c for c in chunks if c.metadata["chunk_hash"] not in processed_chunks]
    random.shuffle(available_chunks)
    
    pbar = tqdm(total=num_questions, initial=len(questions), desc="Generating questions")
    
    chunk_idx = 0
    while len(questions) < num_questions and chunk_idx < len(available_chunks):
        chunk = available_chunks[chunk_idx]
        chunk_hash = chunk.metadata["chunk_hash"]
        
        # Skip if already processed
        if chunk_hash in processed_chunks:
            chunk_idx += 1
            continue
        
        context = chunk.page_content
        source = chunk.metadata.get("source", "unknown")
        
        # Decide question type based on current distribution
        if simple_count < target_simple:
            q = generate_simple_question(client, context, source)
            if q:
                questions.append(q)
                simple_count += 1
                pbar.update(1)
        
        elif reasoning_count < target_reasoning:
            q = generate_reasoning_question(client, context, source)
            if q:
                questions.append(q)
                reasoning_count += 1
                pbar.update(1)
        
        elif multi_count < target_multi and chunk_idx + 2 < len(available_chunks):
            # Get 2-3 related chunks for multi-context
            related_chunks = available_chunks[chunk_idx:chunk_idx+3]
            contexts = [c.page_content for c in related_chunks]
            sources = [c.metadata.get("source", "unknown") for c in related_chunks]
            
            q = generate_multi_context_question(client, contexts, sources)
            if q:
                questions.append(q)
                multi_count += 1
                pbar.update(1)
        
        else:
            # Fallback to simple
            q = generate_simple_question(client, context, source)
            if q:
                questions.append(q)
                simple_count += 1
                pbar.update(1)
        
        processed_chunks.add(chunk_hash)
        chunk_idx += 1
        
        # Save checkpoint periodically
        if len(questions) % 10 == 0:
            save_checkpoint(checkpoint_file, {
                "generated_questions": [asdict(q) for q in questions],
                "processed_chunks": list(processed_chunks)
            })
    
    pbar.close()
    
    # Final checkpoint save
    save_checkpoint(checkpoint_file, {
        "generated_questions": [asdict(q) for q in questions],
        "processed_chunks": list(processed_chunks)
    })
    
    print(f"\n✓ Generated {len(questions)} corpus questions:")
    print(f"  - Simple: {simple_count}")
    print(f"  - Reasoning: {reasoning_count}")
    print(f"  - Multi-context: {multi_count}")
    
    return questions

# ============================================================================
# OUTPUT
# ============================================================================

def save_dataset(corpus_questions: List[GeneratedQuestion], 
                 domain_questions: List[DomainQuestion],
                 output_file: str, config: Config):
    """Save complete evaluation dataset."""
    
    dataset = {
        "metadata": {
            "total_questions": len(corpus_questions) + len(domain_questions),
            "corpus_questions": len(corpus_questions),
            "domain_questions": len(domain_questions),
            "model": config.model_name,
            "embedding_model": config.embed_model,
            "chunk_size": config.chunk_size,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question_type_distribution": {
                "simple": sum(1 for q in corpus_questions if q.question_type == "simple"),
                "reasoning": sum(1 for q in corpus_questions if q.question_type == "reasoning"),
                "multi_context": sum(1 for q in corpus_questions if q.question_type == "multi_context"),
                "domain": len(domain_questions)
            }
        },
        "corpus_questions": [asdict(q) for q in corpus_questions],
        "domain_questions": [asdict(q) for q in domain_questions],
    }
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Saved dataset to {output_file}")
    return dataset

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RAG EVALUATION DATASET GENERATOR")
    print("=" * 70)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  PDF Directory: {config.pdf_dir}")
    print(f"  Model: {config.model_name}")
    print(f"  Embeddings: {config.embed_model}")
    print(f"  Target corpus questions: {config.num_corpus_questions}")
    print(f"  Target domain questions: {config.num_domain_questions}")
    print(f"  Output: {config.output_file}")
    print("=" * 70)
    
    # Step 1: Load documents
    print("\n[1/5] Loading documents...")
    documents = load_documents(config.pdf_dir)
    
    # Step 2: Chunk documents
    print("\n[2/5] Chunking documents...")
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    
    # Step 3: Initialize LLM client
    print("\n[3/5] Connecting to vLLM server...")
    client = VLLMClient(config.vllm_base_url, config.model_name, config)
    
    # Step 4: Generate corpus questions
    print("\n[4/5] Generating corpus questions...")
    corpus_questions = generate_corpus_questions(
        client, chunks, config.num_corpus_questions, config.checkpoint_file
    )
    
    # Step 5: Generate domain questions
    print("\n[5/5] Generating domain questions...")
    domain_questions = generate_domain_questions(client, config.num_domain_questions)
    
    # Save final dataset
    print("\n💾 Saving dataset...")
    dataset = save_dataset(corpus_questions, domain_questions, config.output_file, config)
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nDataset: {config.output_file}")
    print(f"Total questions: {dataset['metadata']['total_questions']}")
    print("\nQuestion distribution:")
    for qtype, count in dataset['metadata']['question_type_distribution'].items():
        print(f"  {qtype}: {count}")
    
    # Cleanup checkpoint
    if os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)
        print(f"\n✓ Cleaned up checkpoint file")


if __name__ == "__main__":
    main()