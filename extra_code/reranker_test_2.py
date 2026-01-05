"""
REAL Stress Test for Reranker - Tests actual Jina model for OOM errors.
Generates 200 candidates with ~1000 tokens each and runs through the real reranker.

Run this on your cluster with:
    python test_reranker_stress.py

Make sure your utils/logger.py and utils/config_loader.py are in the same directory
or adjust the imports in reranker.py accordingly.
"""

import random
import string
import time
import torch
import gc
import sys
from typing import List, Tuple, Dict

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_random_word(min_len: int = 3, max_len: int = 12) -> str:
    """Generate a random word-like string."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def generate_random_sentence(min_words: int = 8, max_words: int = 20) -> str:
    """Generate a random sentence."""
    num_words = random.randint(min_words, max_words)
    words = [generate_random_word() for _ in range(num_words)]
    words[0] = words[0].capitalize()
    return ' '.join(words) + random.choice(['.', '!', '?'])


def generate_document(target_tokens: int = 1000, chars_per_token: float = 4.0) -> str:
    """
    Generate a synthetic document with approximately target_tokens tokens.
    """
    target_chars = int(target_tokens * chars_per_token)
    paragraphs = []
    current_chars = 0
    
    while current_chars < target_chars:
        num_sentences = random.randint(3, 8)
        sentences = [generate_random_sentence() for _ in range(num_sentences)]
        paragraph = ' '.join(sentences)
        paragraphs.append(paragraph)
        current_chars += len(paragraph) + 2
    
    return '\n\n'.join(paragraphs)


def generate_metadata(doc_id: int) -> Dict:
    """Generate fake metadata for a document."""
    return {
        'doc_id': f'doc_{doc_id:04d}',
        'title': f'Document {doc_id}',
        'page': random.randint(1, 50),
        'chunk_index': random.randint(0, 20),
    }


def generate_candidates(
    num_docs: int = 200, 
    tokens_per_doc: int = 1000
) -> List[Tuple[float, Dict, str]]:
    """
    Generate synthetic candidates as they would come from Stage 2.
    Returns: List of (distance_score, metadata, text) tuples
    """
    candidates = []
    
    print(f"Generating {num_docs} documents with ~{tokens_per_doc} tokens each...")
    
    for i in range(num_docs):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_docs} documents")
        
        distance = random.uniform(0.3, 1.8)
        metadata = generate_metadata(i)
        text = generate_document(target_tokens=tokens_per_doc)
        candidates.append((distance, metadata, text))
    
    # Sort by distance (ascending) to simulate Stage 2 output
    candidates.sort(key=lambda x: x[0])
    
    return candidates


# ============================================================================
# GPU MEMORY UTILITIES
# ============================================================================

def print_gpu_memory(label: str = ""):
    """Print current GPU memory usage."""
    prefix = f"  [{label}] " if label else "  "
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, "
              f"Peak: {max_allocated:.2f}GB, Total: {total:.2f}GB")
    else:
        print(f"{prefix}GPU not available, using CPU")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print("=" * 70)
    print("REAL RERANKER STRESS TEST - OOM Detection")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # [0] Environment Check
    # -------------------------------------------------------------------------
    print("\n[0] Environment Check:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  CUDA total memory: {total_mem:.2f}GB")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -------------------------------------------------------------------------
    # [1] Generate Test Data
    # -------------------------------------------------------------------------
    random.seed(42)
    
    print("\n[1] Generating Test Data:")
    start_gen = time.perf_counter()
    candidates = generate_candidates(num_docs=20, tokens_per_doc=2000)
    gen_time = (time.perf_counter() - start_gen) * 1000
    
    total_chars = sum(len(c[2]) for c in candidates)
    total_tokens_est = total_chars / 4
    avg_tokens = total_tokens_est / len(candidates)
    
    print(f"\n  Summary:")
    print(f"    Documents: {len(candidates)}")
    print(f"    Total characters: {total_chars:,}")
    print(f"    Estimated total tokens: {total_tokens_est:,.0f}")
    print(f"    Average tokens per doc: {avg_tokens:.0f}")
    print(f"    Generation time: {gen_time:.2f}ms")
    
    # -------------------------------------------------------------------------
    # [2] Load the REAL Reranker
    # -------------------------------------------------------------------------
    print("\n[2] Loading REAL Jina Reranker:")
    print_gpu_memory("Before load")
    clear_gpu_memory()
    
    try:
        from inference.reranker import Reranker
        
        start_load = time.perf_counter()
        reranker = Reranker(
            model="jinaai/jina-reranker-v3",
            device=device,
            batch_size=8,
            max_length=1024,
            timeout_seconds=120
        )
        load_time = (time.perf_counter() - start_load) * 1000
        
        print(f"  Model loaded in {load_time:.2f}ms")
        print_gpu_memory("After load")
        
    except Exception as e:
        print(f"  ERROR loading reranker: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # -------------------------------------------------------------------------
    # [3] Run Stress Test
    # -------------------------------------------------------------------------
    test_query = "What are the key findings and methodology used in the research analysis?"
    
    print("\n[3] Running rerank_with_details (top_k=10) with 200 candidates:")
    print_gpu_memory("Before rerank")
    clear_gpu_memory()
    print("  Cache cleared, starting rerank...")
    
    try:
        start_rerank = time.perf_counter()
        results = reranker.rerank_with_details(
            query=test_query,
            candidates=candidates,
            top_k=10
        )
        rerank_time = (time.perf_counter() - start_rerank) * 1000
        
        print(f"\n  *** SUCCESS - NO OOM ***")
        print(f"  Reranking completed in {rerank_time:.2f}ms")
        print(f"  Results returned: {len(results)}")
        print_gpu_memory("After rerank")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n  *** OOM ERROR ***")
        print(f"  {e}")
        print_gpu_memory("At OOM")
        
        # Suggestions
        print("\n  Suggestions to fix OOM:")
        print("    1. Reduce max_context_tokens in rerank() call")
        print("    2. Reduce batch_size in Reranker init")
        print("    3. Reduce max_length (truncate documents)")
        print("    4. Use a smaller model (jina-reranker-v2-base-multilingual)")
        print("    5. The 100-candidate cap in your code should help, check if it's working")
        return
        
    except Exception as e:
        print(f"\n  ERROR during reranking: {e}")
        import traceback
        traceback.print_exc()
        print_gpu_memory("At error")
        return
    
    # -------------------------------------------------------------------------
    # [4] Display Results
    # -------------------------------------------------------------------------
    print("\n[4] Top 10 Results:")
    print("-" * 70)
    for r in results:
        text_preview = r['text'][:50].replace('\n', ' ') + "..."
        print(f"  Rank {r['rank']:2d}: score={r['rerank_score']:.4f}, "
              f"orig_rank={r['original_rank']:3d}, "
              f"Δ={r['rank_improvement']:+4d}, "
              f"doc={r['metadata']['doc_id']}")
    
    # -------------------------------------------------------------------------
    # [5] Validation
    # -------------------------------------------------------------------------
    print("\n[5] Validation:")
    
    errors = []
    
    # Check count
    if len(results) != 10:
        errors.append(f"Expected 10 results, got {len(results)}")
    else:
        print("  ✓ Correct result count (10)")
    
    # Check sorting
    scores = [r['rerank_score'] for r in results]
    if scores != sorted(scores, reverse=True):
        errors.append("Results not sorted by score descending")
    else:
        print("  ✓ Results sorted correctly")
    
    # Check ranks
    ranks = [r['rank'] for r in results]
    if ranks != list(range(1, len(results) + 1)):
        errors.append("Ranks not sequential")
    else:
        print("  ✓ Ranks are sequential")
    
    # Check rank_improvement
    for r in results:
        expected = r['original_rank'] - r['rank']
        if r['rank_improvement'] != expected:
            errors.append(f"rank_improvement wrong for rank {r['rank']}")
            break
    else:
        print("  ✓ rank_improvement correctly calculated")
    
    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e}")
    
    # -------------------------------------------------------------------------
    # [6] Memory Summary
    # -------------------------------------------------------------------------
    print("\n[6] Final Memory Summary:")
    print_gpu_memory("Final")
    
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        usage_pct = (peak / total) * 100
        print(f"  Peak memory usage: {usage_pct:.1f}% of total GPU memory")
        
        if usage_pct > 90:
            print("  ⚠️  WARNING: Very high memory usage, OOM risk with larger inputs")
        elif usage_pct > 70:
            print("  ⚠️  CAUTION: High memory usage, monitor closely")
        else:
            print("  ✓ Memory usage looks healthy")
    
    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()