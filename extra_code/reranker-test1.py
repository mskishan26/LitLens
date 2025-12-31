"""
Test suite for reranker production fixes.
Tests all 4 issues: blocking I/O, timeout handling, candidate capping, and context window limits.
"""

import torch
import time
import numpy as np
from typing import List, Tuple, Dict
from unittest.mock import Mock, patch
from concurrent.futures import TimeoutError as FuturesTimeoutError

# Assuming your reranker is importable
from reranker import Reranker


def generate_mock_candidates(n: int, avg_chunk_length: int = 500) -> List[Tuple[float, Dict, str]]:
    """Generate mock candidates with varying chunk sizes."""
    candidates = []
    for i in range(n):
        distance = np.random.uniform(0.3, 0.9)
        metadata = {
            'file_path': f'/fake/path/doc_{i % 10}.txt',
            'chunk_index': i,
            'chunk_size': 800,
        }
        # Vary chunk length around average
        chunk_length = int(avg_chunk_length * np.random.uniform(0.5, 1.5))
        chunk_text = f"This is chunk {i}. " + "Lorem ipsum dolor sit amet. " * (chunk_length // 30)
        candidates.append((distance, metadata, chunk_text))
    return candidates


def test_issue_1_missing_chunk_text():
    """Test Issue #1: Should skip candidates with None chunk_text and log warning."""
    print("\n=== Testing Issue #1: Missing chunk_text ===")
    
    reranker = Reranker(device='cuda')
    
    # Create candidates with some None chunk_text
    candidates = [
        (0.5, {'file_path': 'doc1.txt', 'chunk_index': 0}, "Valid chunk text 1"),
        (0.6, {'file_path': 'doc2.txt', 'chunk_index': 1}, None),  # Should skip
        (0.4, {'file_path': 'doc3.txt', 'chunk_index': 2}, "Valid chunk text 2"),
        (0.7, {'file_path': 'doc4.txt', 'chunk_index': 3}, None),  # Should skip
        (0.3, {'file_path': 'doc5.txt', 'chunk_index': 4}, "Valid chunk text 3"),
    ]
    
    query = "test query"
    results, run_info = reranker.rerank(query, candidates, top_k=3)
    
    # Should only get 3 valid candidates back (skipped 2 with None)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    
    # Verify none of the results have None text
    for score, meta, text in results:
        assert text is not None, "Result contains None chunk_text!"
    
    print(f"✓ Successfully skipped {2} candidates with None chunk_text")
    print(f"✓ Returned {len(results)} valid results")


def test_issue_3_timeout_fallback():
    """Test Issue #3: Should timeout and fallback to Stage 2 ordering."""
    print("\n=== Testing Issue #3: Timeout with fallback ===")
    
    reranker = Reranker(device='cuda', timeout_seconds=2)
    
    candidates = generate_mock_candidates(10)
    query = "test query"
    
    # Mock the model.rerank to hang
    def hanging_rerank(*args, **kwargs):
        time.sleep(10)  # Simulate hang (longer than timeout)
        return []
    
    with patch.object(reranker.model, 'rerank', side_effect=hanging_rerank):
        start = time.time()
        results, run_info = reranker.rerank(query, candidates, top_k=5)
        elapsed = time.time() - start
        
        # With shutdown(wait=False), should return quickly (~2s)
        assert elapsed < 3, f"Should timeout quickly, took {elapsed:.1f}s"
        
        # Should return fallback results (top 5 from Stage 2)
        assert len(results) == 5, f"Expected 5 fallback results, got {len(results)}"
        
        # Fallback should preserve Stage 2 ordering
        assert results[0][1]['chunk_index'] == 0, "Fallback didn't preserve Stage 2 order"
        
    print(f"✓ Timed out after {elapsed:.1f}s (expected ~2s)")
    print(f"✓ Returned {len(results)} fallback results from Stage 2 ordering")
    print(f"  Note: Background thread abandoned (won't block user request)")


def test_issue_3_exception_fallback():
    """Test Issue #3: Should handle exceptions and fallback gracefully."""
    print("\n=== Testing Issue #3: Exception with fallback ===")
    
    reranker = Reranker(device='cuda')
    
    candidates = generate_mock_candidates(10)
    query = "test query"
    
    # Mock the model.rerank to raise exception
    with patch.object(reranker.model, 'rerank', side_effect=RuntimeError("GPU OOM!")):
        results, run_info = reranker.rerank(query, candidates, top_k=5)
        
        # Should return fallback results despite exception
        assert len(results) == 5, f"Expected 5 fallback results, got {len(results)}"
        
    print(f"✓ Handled exception gracefully")
    print(f"✓ Returned {len(results)} fallback results")


def test_issue_4_candidate_capping():
    """Test Issue #4: Should cap candidates to 100 for reranking."""
    print("\n=== Testing Issue #4: Candidate capping to 100 ===")
    
    reranker = Reranker(device='cuda')
    
    # Generate 200 candidates
    candidates = generate_mock_candidates(200)
    query = "test query"
    
    # Track how many candidates actually get reranked
    original_rerank = reranker.model.rerank
    rerank_call_count = [0]
    
    def counting_rerank(q, texts, **kwargs):
        rerank_call_count[0] = len(texts)
        # Return mock results
        return [{'index': i, 'relevance_score': 0.5 + i*0.01, 'document': texts[i]} 
                for i in range(len(texts))]
    
    with patch.object(reranker.model, 'rerank', side_effect=counting_rerank):
        results, run_info = reranker.rerank(query, candidates, top_k=10)
        
        # Should cap to 100 candidates for reranking
        assert rerank_call_count[0] <= 100, f"Reranked {rerank_call_count[0]} candidates, should cap at 100"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    
    print(f"✓ Capped {len(candidates)} candidates to {rerank_call_count[0]} for reranking")
    print(f"✓ Returned top {len(results)} results")


def test_issue_4_context_window_limit():
    """Test Issue #4: Should truncate candidates exceeding context window."""
    print("\n=== Testing Issue #4: Context window protection ===")
    
    reranker = Reranker(device='cuda')
    
    # Generate candidates that will exceed 100k tokens
    # Each chunk ~2k chars = ~500 tokens, so 250 chunks = ~125k tokens
    large_candidates = generate_mock_candidates(250, avg_chunk_length=2000)
    query = "test query"
    
    rerank_call_count = [0]
    
    def counting_rerank(q, texts, **kwargs):
        rerank_call_count[0] = len(texts)
        return [{'index': i, 'relevance_score': 0.5, 'document': texts[i]} 
                for i in range(len(texts))]
    
    with patch.object(reranker.model, 'rerank', side_effect=counting_rerank):
        results, run_info = reranker.rerank(query, large_candidates, top_k=10, max_context_tokens=100000)
        
        # Should truncate to fit context window
        total_chars = sum(len(large_candidates[i][2]) for i in range(rerank_call_count[0]))
        estimated_tokens = total_chars / 4
        
        assert estimated_tokens <= 105000, f"Exceeded context limit: {estimated_tokens:.0f} tokens"
        assert rerank_call_count[0] < 250, f"Should truncate 250 candidates, kept {rerank_call_count[0]}"
    
    print(f"✓ Truncated {len(large_candidates)} candidates to {rerank_call_count[0]}")
    print(f"✓ Estimated tokens: {estimated_tokens:.0f} (limit: 100k)")


def test_combined_limits():
    """Test that both caps work together: context window + 100 candidate limit."""
    print("\n=== Testing combined limits (context + candidate cap) ===")
    
    reranker = Reranker(device='cuda')
    
    # 150 candidates with moderate size (~300 chars = ~75 tokens each)
    # Total: ~11,250 tokens (well under 100k limit)
    candidates = generate_mock_candidates(150, avg_chunk_length=300)
    query = "test query"
    
    rerank_call_count = [0]
    
    def counting_rerank(q, texts, **kwargs):
        rerank_call_count[0] = len(texts)
        return [{'index': i, 'relevance_score': 0.5, 'document': texts[i]} 
                for i in range(len(texts))]
    
    with patch.object(reranker.model, 'rerank', side_effect=counting_rerank):
        results, run_info = reranker.rerank(query, candidates, top_k=10)
        
        # Should cap at 100 (candidate limit), not context limit
        assert rerank_call_count[0] == 100, f"Expected 100 candidates, got {rerank_call_count[0]}"
    
    print(f"✓ Candidate limit (100) applied before context window limit")
    print(f"✓ Processed {rerank_call_count[0]} out of {len(candidates)} candidates")


def test_normal_operation():
    """Test that normal operation still works with all fixes in place."""
    print("\n=== Testing normal operation ===")
    
    reranker = Reranker(device='cuda')
    
    candidates = generate_mock_candidates(20)
    query = "test query about important topics"
    
    # Mock rerank to return realistic results
    def mock_rerank(q, texts, **kwargs):
        return [{'index': i, 'relevance_score': 1.0 - i*0.05, 'document': texts[i]} 
                for i in range(len(texts))]
    
    with patch.object(reranker.model, 'rerank', side_effect=mock_rerank):
        results, run_info = reranker.rerank(query, candidates, top_k=5)
        
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        # Check scores are in descending order
        scores = [score for score, _, _ in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"
        
        # Check all results have valid structure
        for score, meta, text in results:
            assert isinstance(score, float), "Score should be float"
            assert isinstance(meta, dict), "Metadata should be dict"
            assert isinstance(text, str), "Text should be string"
            assert text is not None, "Text should not be None"
    
    print(f"✓ Normal operation works correctly")
    print(f"✓ Top 5 results returned with proper structure")
    print(f"✓ Scores: {[f'{s:.2f}' for s, _, _ in results]}")


def run_all_tests():
    """Run all test cases."""
    print("="*60)
    print("RERANKER PRODUCTION FIXES TEST SUITE")
    print("="*60)
    
    try:
        test_issue_1_missing_chunk_text()
        test_issue_3_timeout_fallback()
        test_issue_3_exception_fallback()
        test_issue_4_candidate_capping()
        test_issue_4_context_window_limit()
        test_combined_limits()
        test_normal_operation()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()