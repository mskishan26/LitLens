#!/usr/bin/env python3
"""
RAG Evaluation Dataset Generator v2
===================================
Generates expert-level domain questions for RAG evaluation.

This version generates questions that domain experts would actually ask:
- Practical tradeoffs and comparisons
- Tool/method recommendations
- Synthesis across multiple sources
- Open-ended explanations and overviews
"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import hashlib
import re

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

# OpenAI client for vLLM
from openai import OpenAI
import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # Paths
    pdf_dir: str = "/scratch/sathishbabu.ki/data_files/input_pdf"
    output_file: str = "rag_eval_dataset.json"
    checkpoint_file: str = "generation_checkpoint.json"
    
    # vLLM server settings - use 0.0.0.0 instead of localhost for cluster compatibility
    vllm_base_url: str = "http://0.0.0.0:8000/v1"
    model_name: str = "/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    
    # Generation settings
    num_questions: int = 400
    
    # Chunk settings - sized to fit in context window with room for output
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # LLM generation params
    temperature: float = 0.8  # Higher for more creative questions
    max_tokens: int = 1024  # Keep reasonable given 4096 context limit
    
    # Processing
    max_retries: int = 3
    retry_delay: float = 2.0

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class ExpertQuestion:
    question: str
    question_type: str  # comparison, tradeoff, practical, synthesis, explanation, troubleshooting
    topics: List[str]  # Key concepts/methods mentioned
    difficulty: str  # intermediate, advanced, expert
    requires_synthesis: bool  # Whether it likely needs multiple sources
    source_context: str = ""  # The context that inspired this question
    metadata: Dict[str, Any] = field(default_factory=dict)

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


def extract_key_terms(documents: List[Document], client: 'VLLMClient') -> List[str]:
    """Extract key domain terms from the corpus for question generation."""
    
    # Sample some documents
    sample_docs = random.sample(documents, min(20, len(documents)))
    combined_text = "\n\n".join([d.page_content[:1000] for d in sample_docs])
    
    prompt = f"""Analyze this scientific text and extract key domain-specific terms, methods, tools, and concepts.

Text sample:
{combined_text[:3000]}

Extract and categorize:
1. METHODS: Statistical/computational methods mentioned
2. TOOLS: Software tools or packages
3. CONCEPTS: Key domain concepts
4. MARKERS: Biological markers, proteins, genes
5. DATA_TYPES: Types of data or experiments

Output as JSON:
{{"methods": [...], "tools": [...], "concepts": [...], "markers": [...], "data_types": [...]}}

Only output the JSON, nothing else."""

    try:
        response = client.generate(prompt, temperature=0.3)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        terms = json.loads(response.strip())
        all_terms = []
        for category in terms.values():
            if isinstance(category, list):
                all_terms.extend(category)
        
        print(f"✓ Extracted {len(all_terms)} domain terms")
        return all_terms
    except Exception as e:
        print(f"  ⚠ Term extraction failed: {e}")
        return []

# ============================================================================
# LLM CLIENT
# ============================================================================
class VLLMClient:
    """Client for vLLM OpenAI-compatible server."""
    
    def __init__(self, base_url: str, model_name: str, config: Config):
        # Force 0.0.0.0 instead of localhost
        base_url = base_url.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        
        print(f"  Creating HTTP client for: {base_url}")
        
        http_client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
        
        # Test raw connection first
        print(f"  Testing raw HTTP connection...")
        try:
            test_response = http_client.get(f"{base_url}/models")
            print(f"  Raw HTTP test: {test_response.status_code}")
        except Exception as e:
            print(f"  Raw HTTP test failed: {e}")
            raise
        
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
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
            
            if self.model_name not in available and len(available) > 0:
                old_name = self.model_name
                self.model_name = available[0]
                print(f"  ⚠ Model name adjusted to: {self.model_name[-60:]}")
                
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to vLLM server at {self.client.base_url}.\n"
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
                    print(f"  Retry {attempt+1}/{self.config.max_retries}: {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    raise e
        return ""

# ============================================================================
# EXPERT QUESTION GENERATION
# ============================================================================

EXPERT_SYSTEM_PROMPT = """You are a senior researcher who asks deep, practical questions when reading scientific papers. 
Your questions are the kind that come up during:
- Lab meetings and journal clubs
- Method selection and experimental design
- Troubleshooting and data analysis
- Literature reviews and grant writing

Your questions should be:
1. PRACTICAL - About real decisions researchers face
2. COMPARATIVE - Comparing methods, tools, or approaches  
3. NUANCED - Acknowledging tradeoffs and assumptions
4. DOMAIN-SPECIFIC - Using proper technical terminology
5. OPEN-ENDED - Requiring synthesis and explanation, not just yes/no"""


QUESTION_TEMPLATES = [
    # Comparison questions
    "What is the difference between {term1} and {term2}? Which would be better for {scenario}?",
    "Compare the tradeoffs between {method1} and {method2} for {application}.",
    
    # Practical/applied questions  
    "What are the current options for {task}? What are their limitations?",
    "If I have {data_situation}, what approach would you recommend for {goal}?",
    "Does {method} work well for {edge_case}? What assumptions might be violated?",
    
    # Synthesis questions
    "Generate an overview of current tools for {domain} that {criteria}.",
    "List examples of studies that have done {analysis_type} for {application}.",
    "What are the best practices for {task} when dealing with {challenge}?",
    
    # Explanation questions
    "Explain how {method} handles {specific_aspect}.",
    "What is {concept}? How does it relate to {related_concept}?",
    
    # Troubleshooting questions
    "I'm seeing {problem} when using {method}. What might be causing this?",
    "What are common pitfalls when applying {method} to {data_type}?",
    
    # Marker/validation questions
    "Is {marker} a good indicator of {condition}? What's the evidence?",
    "How reliable is {method} for detecting {target} in {context}?",
]


def generate_expert_questions_from_context(
    client: VLLMClient, 
    context: str, 
    source: str,
    num_questions: int = 5
) -> List[ExpertQuestion]:
    """Generate expert-level questions inspired by document context."""
    
    prompt = f"""You are a domain expert reading this scientific text. Generate {num_questions} questions that a researcher might ask after reading this.

TEXT FROM: {source}
---
{context[:2000]}
---

Generate questions like:
- "What are the tradeoffs of [method] for [application]?"
- "What is the difference between [tool1] and [tool2]?"
- "For experiments with [constraint], what are the options for [task]?"
- "Is [marker/method] reliable for [purpose]?"
- "List tools for [domain] that [have certain features]."

Requirements:
1. PRACTICAL questions researchers would actually need
2. Reference SPECIFIC methods/tools from the text
3. Ask about TRADEOFFS and COMPARISONS
4. Vary question types

Output as JSON array:
[{{"question": "...", "type": "comparison|tradeoff|practical|synthesis|explanation", "topics": ["topic1"], "difficulty": "intermediate|advanced|expert", "requires_synthesis": true/false}}]

Generate exactly {num_questions} questions. Output ONLY the JSON array."""

    try:
        response = client.generate(prompt, EXPERT_SYSTEM_PROMPT, temperature=0.85)
        
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        # Find JSON array
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        questions_data = json.loads(response)
        
        questions = []
        for q in questions_data:
            if isinstance(q, dict) and "question" in q:
                questions.append(ExpertQuestion(
                    question=q["question"],
                    question_type=q.get("type", "practical"),
                    topics=q.get("topics", []),
                    difficulty=q.get("difficulty", "advanced"),
                    requires_synthesis=q.get("requires_synthesis", False),
                    source_context=context[:500],
                    metadata={"source_file": source}
                ))
        
        return questions
        
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"  ⚠ Question generation error: {e}")
        return []


def generate_cross_document_questions(
    client: VLLMClient,
    chunks: List[Document],
    num_questions: int = 50
) -> List[ExpertQuestion]:
    """Generate questions that require synthesizing across multiple documents."""
    
    print(f"\n🔗 Generating {num_questions} cross-document synthesis questions...")
    
    questions = []
    
    # Sample pairs/triplets of chunks from different sources
    sources = list(set(c.metadata.get("source", "") for c in chunks))
    
    for _ in tqdm(range(num_questions), desc="Cross-doc questions"):
        # Pick 2-3 random chunks from different sources if possible
        if len(sources) >= 2:
            selected_sources = random.sample(sources, min(3, len(sources)))
            selected_chunks = []
            for src in selected_sources:
                src_chunks = [c for c in chunks if c.metadata.get("source") == src]
                if src_chunks:
                    selected_chunks.append(random.choice(src_chunks))
        else:
            selected_chunks = random.sample(chunks, min(3, len(chunks)))
        
        combined_context = "\n\n---\n\n".join([
            f"[From {c.metadata.get('source', 'unknown')}]:\n{c.page_content[:800]}"
            for c in selected_chunks
        ])
        
        prompt = f"""Based on these excerpts from different papers, generate 1 question requiring synthesis across sources.

{combined_context}

Generate a question that connects concepts across the sources - comparisons, integrations, or overviews.

Output as JSON:
{{"question": "...", "type": "synthesis", "topics": [...], "difficulty": "expert", "requires_synthesis": true}}

Output ONLY the JSON object."""

        try:
            response = client.generate(prompt, temperature=0.8, max_tokens=500)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            # Find JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                response = response[start:end]
            
            q_data = json.loads(response)
            
            if "question" in q_data:
                questions.append(ExpertQuestion(
                    question=q_data["question"],
                    question_type="synthesis",
                    topics=q_data.get("topics", []),
                    difficulty="expert",
                    requires_synthesis=True,
                    source_context=combined_context[:500],
                    metadata={"sources": [c.metadata.get("source") for c in selected_chunks]}
                ))
        except Exception as e:
            continue
    
    print(f"✓ Generated {len(questions)} cross-document questions")
    return questions


def generate_edge_case_questions(
    client: VLLMClient,
    domain_terms: List[str],
    num_questions: int = 30
) -> List[ExpertQuestion]:
    """Generate questions about edge cases, limitations, and assumptions."""
    
    print(f"\n⚠️ Generating {num_questions} edge case/limitation questions...")
    
    if not domain_terms:
        domain_terms = ["the methods", "statistical analysis", "data preprocessing"]
    
    prompt = f"""Generate {num_questions} critical questions about edge cases and limitations in methods.

Domain terms: {', '.join(domain_terms[:20])}

Question types:
- When methods/assumptions FAIL
- EDGE CASES that cause problems  
- LIMITATIONS not mentioned in papers
- TROUBLESHOOTING common issues

Output as JSON array:
[{{"question": "...", "type": "troubleshooting", "topics": [...], "difficulty": "advanced"}}]

Generate exactly {num_questions} questions. Output ONLY the JSON array."""

    try:
        response = client.generate(prompt, temperature=0.85)
        
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        questions_data = json.loads(response)
        
        questions = []
        for q in questions_data:
            if isinstance(q, dict) and "question" in q:
                questions.append(ExpertQuestion(
                    question=q["question"],
                    question_type=q.get("type", "troubleshooting"),
                    topics=q.get("topics", []),
                    difficulty=q.get("difficulty", "advanced"),
                    requires_synthesis=True,
                    metadata={"category": "edge_case"}
                ))
        
        print(f"✓ Generated {len(questions)} edge case questions")
        return questions
        
    except Exception as e:
        print(f"  ⚠ Edge case generation failed: {e}")
        return []


def generate_overview_questions(
    client: VLLMClient,
    domain_terms: List[str],
    num_questions: int = 20
) -> List[ExpertQuestion]:
    """Generate questions asking for overviews and comprehensive lists."""
    
    print(f"\n📋 Generating {num_questions} overview/listing questions...")
    
    prompt = f"""Generate {num_questions} questions asking for overviews, lists, or summaries.

Domain context: {', '.join(domain_terms[:15])}

Examples:
- "List current tools for [domain] that [criteria]."
- "What approaches exist for [task]? Compare strengths/weaknesses."
- "What software options exist for [task]? Which are open source?"

Output as JSON array:
[{{"question": "...", "type": "synthesis", "topics": [...], "difficulty": "advanced"}}]

Generate exactly {num_questions} questions. Output ONLY the JSON array."""

    try:
        response = client.generate(prompt, temperature=0.8)
        
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        questions_data = json.loads(response)
        
        questions = []
        for q in questions_data:
            if isinstance(q, dict) and "question" in q:
                questions.append(ExpertQuestion(
                    question=q["question"],
                    question_type="synthesis",
                    topics=q.get("topics", []),
                    difficulty=q.get("difficulty", "advanced"),
                    requires_synthesis=True,
                    metadata={"category": "overview"}
                ))
        
        print(f"✓ Generated {len(questions)} overview questions")
        return questions
        
    except Exception as e:
        print(f"  ⚠ Overview generation failed: {e}")
        return []

# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load generation checkpoint if exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"questions": [], "processed_chunks": []}


def save_checkpoint(checkpoint_file: str, questions: List[ExpertQuestion], processed: List[str]):
    """Save generation checkpoint."""
    with open(checkpoint_file, 'w') as f:
        json.dump({
            "questions": [asdict(q) for q in questions],
            "processed_chunks": processed
        }, f)


def generate_all_questions(
    client: VLLMClient, 
    chunks: List[Document],
    config: Config
) -> List[ExpertQuestion]:
    """Main question generation pipeline."""
    
    all_questions = []
    
    # Load checkpoint
    checkpoint = load_checkpoint(config.checkpoint_file)
    existing_questions = [ExpertQuestion(**q) for q in checkpoint.get("questions", [])]
    processed_chunks = set(checkpoint.get("processed_chunks", []))
    
    if existing_questions:
        print(f"  Resuming from checkpoint: {len(existing_questions)} questions")
        all_questions.extend(existing_questions)
    
    # Calculate how many more we need
    remaining = config.num_questions - len(all_questions)
    if remaining <= 0:
        print(f"  Already have {len(all_questions)} questions, target is {config.num_questions}")
        return all_questions
    
    # Distribution of question types
    # 60% from individual contexts, 20% cross-document, 10% edge cases, 10% overviews
    context_questions_needed = int(remaining * 0.6)
    cross_doc_needed = int(remaining * 0.2)
    edge_case_needed = int(remaining * 0.1)
    overview_needed = remaining - context_questions_needed - cross_doc_needed - edge_case_needed
    
    # Extract domain terms first
    print("\n🔍 Extracting domain terminology...")
    domain_terms = extract_key_terms(chunks[:50], client)
    
    # 1. Generate questions from individual contexts
    print(f"\n📝 Generating {context_questions_needed} context-based questions...")
    
    available_chunks = [c for c in chunks if c.metadata["chunk_hash"] not in processed_chunks]
    random.shuffle(available_chunks)
    
    questions_per_chunk = 3
    chunks_needed = (context_questions_needed // questions_per_chunk) + 1
    
    pbar = tqdm(total=context_questions_needed, desc="Context questions")
    context_questions = []
    
    for chunk in available_chunks[:chunks_needed]:
        if len(context_questions) >= context_questions_needed:
            break
            
        new_qs = generate_expert_questions_from_context(
            client, 
            chunk.page_content, 
            chunk.metadata.get("source", "unknown"),
            num_questions=questions_per_chunk
        )
        
        context_questions.extend(new_qs)
        processed_chunks.add(chunk.metadata["chunk_hash"])
        pbar.update(len(new_qs))
        
        # Checkpoint every 20 questions
        if len(context_questions) % 20 == 0:
            save_checkpoint(config.checkpoint_file, 
                          all_questions + context_questions, 
                          list(processed_chunks))
    
    pbar.close()
    all_questions.extend(context_questions[:context_questions_needed])
    
    # 2. Cross-document synthesis questions
    cross_doc_qs = generate_cross_document_questions(client, chunks, cross_doc_needed)
    all_questions.extend(cross_doc_qs)
    
    # 3. Edge case questions
    edge_case_qs = generate_edge_case_questions(client, domain_terms, edge_case_needed)
    all_questions.extend(edge_case_qs)
    
    # 4. Overview questions
    overview_qs = generate_overview_questions(client, domain_terms, overview_needed)
    all_questions.extend(overview_qs)
    
    # Final save
    save_checkpoint(config.checkpoint_file, all_questions, list(processed_chunks))
    
    return all_questions

# ============================================================================
# OUTPUT
# ============================================================================

def deduplicate_questions(questions: List[ExpertQuestion]) -> List[ExpertQuestion]:
    """Remove duplicate or very similar questions."""
    seen = set()
    unique = []
    
    for q in questions:
        # Normalize question for comparison
        normalized = q.question.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Simple dedup by exact match
        if normalized not in seen:
            seen.add(normalized)
            unique.append(q)
    
    print(f"  Deduplication: {len(questions)} → {len(unique)} questions")
    return unique


def save_dataset(questions: List[ExpertQuestion], output_file: str, config: Config):
    """Save the evaluation dataset."""
    
    # Deduplicate
    questions = deduplicate_questions(questions)
    
    # Count by type
    type_counts = {}
    difficulty_counts = {}
    synthesis_count = 0
    
    for q in questions:
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
        difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
        if q.requires_synthesis:
            synthesis_count += 1
    
    dataset = {
        "metadata": {
            "total_questions": len(questions),
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.model_name.split("/")[-1] if "/" in config.model_name else config.model_name,
            "source_directory": config.pdf_dir,
            "question_type_distribution": type_counts,
            "difficulty_distribution": difficulty_counts,
            "requires_synthesis_count": synthesis_count,
        },
        "questions": [asdict(q) for q in questions]
    }
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Saved {len(questions)} questions to {output_file}")
    return dataset

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RAG EVALUATION DATASET GENERATOR v2")
    print("Expert-Level Question Generation")
    print("=" * 70)
    
    config = Config()
    
    print(f"\nConfiguration:")
    print(f"  PDF Directory: {config.pdf_dir}")
    print(f"  Model: {config.model_name.split('/')[-1]}")
    print(f"  Target questions: {config.num_questions}")
    print(f"  Output: {config.output_file}")
    print("=" * 70)
    
    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    documents = load_documents(config.pdf_dir)
    
    # Step 2: Chunk documents
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents, config.chunk_size, config.chunk_overlap)
    
    # Step 3: Connect to LLM
    print("\n[3/4] Connecting to vLLM server...")
    client = VLLMClient(config.vllm_base_url, config.model_name, config)
    
    # Step 4: Generate questions
    print("\n[4/4] Generating expert questions...")
    questions = generate_all_questions(client, chunks, config)
    
    # Save
    print("\n💾 Saving dataset...")
    dataset = save_dataset(questions, config.output_file, config)
    
    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nDataset: {config.output_file}")
    print(f"Total questions: {dataset['metadata']['total_questions']}")
    print("\nQuestion types:")
    for qtype, count in dataset['metadata']['question_type_distribution'].items():
        print(f"  {qtype}: {count}")
    print(f"\nRequires synthesis: {dataset['metadata']['requires_synthesis_count']}")
    
    # Show sample questions
    print("\n" + "-" * 70)
    print("SAMPLE QUESTIONS:")
    print("-" * 70)
    for q in random.sample(questions, min(5, len(questions))):
        print(f"\n[{q.question_type.upper()}] {q.question}")
    
    # Cleanup checkpoint
    if os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)
        print(f"\n✓ Cleaned up checkpoint file")


if __name__ == "__main__":
    main()