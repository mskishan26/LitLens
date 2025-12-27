"""
Test script for the Reranker class.
Tests reranking functionality with mock academic paper chunks.
"""

import sys
from pathlib import Path

# # Add parent directory to path for imports
# sys.path.append(str(Path(__file__).parent.parent))

from reranker import Reranker


def create_mock_candidates(query_type="causal_inference"):
    """
    Create mock candidates with varying relevance to test reranking.
    
    Returns:
        List of (distance, metadata, chunk_text) tuples
    """
    if query_type == "causal_inference":
        query = "What are propensity score methods in causal inference?"
        
        candidates = [
            # High relevance - directly about propensity scores
            (0.35, 
             {"file_path": "paper1.md", "chunk_index": 0, "paper_title": "Propensity Score Methods"},
             "Propensity score methods are widely used in observational studies to estimate causal effects. "
             "The propensity score is the conditional probability of receiving treatment given observed covariates. "
             "Common methods include propensity score matching, weighting, and stratification."),
            
            # Medium relevance - mentions causality but not propensity scores
            (0.45,
             {"file_path": "paper2.md", "chunk_index": 1, "paper_title": "Causal Inference Overview"},
             "Causal inference seeks to determine cause-and-effect relationships from data. "
             "Various methods exist including randomized experiments, instrumental variables, "
             "and regression discontinuity designs. These methods help address confounding."),
            
            # Low relevance - statistics but not causal inference
            (0.50,
             {"file_path": "paper3.md", "chunk_index": 2, "paper_title": "Regression Models"},
             "Linear regression models are fundamental tools in statistics. "
             "They allow us to model relationships between variables and make predictions. "
             "Model diagnostics include checking residual plots and variance inflation factors."),
            
            # Very high relevance - specific propensity score technique
            (0.40,
             {"file_path": "paper4.md", "chunk_index": 0, "paper_title": "Inverse Probability Weighting"},
             "Inverse probability of treatment weighting (IPTW) uses propensity scores to create "
             "a pseudo-population where treatment assignment is independent of measured confounders. "
             "This method weights each observation by the inverse of their propensity score."),
            
            # Medium-high relevance - related methodology
            (0.42,
             {"file_path": "paper5.md", "chunk_index": 3, "paper_title": "Matching Methods"},
             "Matching methods pair treated and control units with similar characteristics. "
             "Covariate balance is crucial for valid causal inference. Various distance metrics "
             "can be used including Mahalanobis distance and propensity score distances."),
            
            # Irrelevant - completely different topic
            (0.60,
             {"file_path": "paper6.md", "chunk_index": 1, "paper_title": "Deep Learning"},
             "Convolutional neural networks have revolutionized computer vision tasks. "
             "These models learn hierarchical features through multiple layers of convolution "
             "and pooling operations applied to image data."),
        ]
        
    else:  # "survival_analysis"
        query = "How do Cox proportional hazards models work?"
        
        candidates = [
            (0.38,
             {"file_path": "paper7.md", "chunk_index": 0, "paper_title": "Cox Models"},
             "The Cox proportional hazards model is a semi-parametric method for survival analysis. "
             "It models the hazard ratio as a function of covariates without making assumptions "
             "about the baseline hazard function."),
            
            (0.55,
             {"file_path": "paper8.md", "chunk_index": 2, "paper_title": "Logistic Regression"},
             "Logistic regression models binary outcomes using a logit link function. "
             "The model estimates log odds ratios for each predictor variable."),
            
            (0.41,
             {"file_path": "paper9.md", "chunk_index": 1, "paper_title": "Survival Analysis"},
             "Survival analysis studies time-to-event data. Common methods include "
             "Kaplan-Meier curves, log-rank tests, and parametric survival models."),
        ]
    
    return query, candidates


def test_basic_reranking():
    """Test basic reranking functionality."""
    print("=" * 80)
    print("TEST 1: Basic Reranking")
    print("=" * 80)
    
    # Initialize reranker (will use config if available, or defaults)
    print("\nInitializing reranker...")
    reranker = Reranker(
        model_name="jinaai/jina-reranker-v3", #"Qwen/Qwen3-Reranker-4B",    
        batch_size=4,
        max_length=8192
    )
    
    # Get test data
    query, candidates = create_mock_candidates("causal_inference")
    
    print(f"\nQuery: {query}")
    print(f"Number of candidates: {len(candidates)}\n")
    
    # Show original ranking
    print("Original ranking (by distance):")
    for rank, (dist, meta, text) in enumerate(candidates, 1):
        print(f"  {rank}. [{dist:.3f}] {meta['paper_title']}: {text[:80]}...")
    
    # Rerank
    print("\nPerforming reranking...")
    reranked = reranker.rerank(query, candidates, top_k=5, return_scores=True)
    
    # Show reranked results
    print("\nReranked results:")
    for rank, (score, meta, text) in enumerate(reranked, 1):
        print(f"  {rank}. [score={score:.4f}] {meta['paper_title']}: {text[:80]}...")
    
    return reranker, query, candidates


def test_detailed_reranking(reranker, query, candidates):
    """Test detailed reranking with rank improvements."""
    print("\n" + "=" * 80)
    print("TEST 2: Detailed Reranking")
    print("=" * 80)
    
    detailed = reranker.rerank_with_details(query, candidates, top_k=5)
    
    print(f"\nDetailed reranking results for: '{query}'\n")
    
    for result in detailed:
        print(f"Rank {result['rank']}: {result['metadata']['paper_title']}")
        print(f"  Rerank score: {result['rerank_score']:.4f}")
        print(f"  Original distance: {result['original_distance']:.3f}")
        print(f"  Original rank: {result['original_rank']}")
        print(f"  Rank improvement: {result['rank_improvement']:+d}")
        print(f"  Text: {result['text'][:100]}...")
        print()


def test_different_query(reranker):
    """Test with a different query type."""
    print("=" * 80)
    print("TEST 3: Different Query Type")
    print("=" * 80)
    
    query, candidates = create_mock_candidates("survival_analysis")
    
    print(f"\nQuery: {query}")
    print(f"Number of candidates: {len(candidates)}\n")
    
    print("Original ranking:")
    for rank, (dist, meta, text) in enumerate(candidates, 1):
        print(f"  {rank}. [{dist:.3f}] {meta['paper_title']}")
    
    reranked = reranker.rerank(query, candidates, top_k=3, return_scores=True)
    
    print("\nReranked results:")
    for rank, (score, meta, text) in enumerate(reranked, 1):
        print(f"  {rank}. [score={score:.4f}] {meta['paper_title']}")


def test_empty_candidates(reranker):
    """Test edge case with empty candidates."""
    print("\n" + "=" * 80)
    print("TEST 4: Edge Case - Empty Candidates")
    print("=" * 80)
    
    query = "test query"
    candidates = []
    
    print(f"\nQuery: {query}")
    print(f"Number of candidates: {len(candidates)}")
    
    result = reranker.rerank(query, candidates, top_k=5)
    print(f"Result: {result}")
    print("✓ Handled empty candidates correctly")


def test_single_candidate(reranker):
    """Test edge case with single candidate."""
    print("\n" + "=" * 80)
    print("TEST 5: Edge Case - Single Candidate")
    print("=" * 80)
    
    query = "propensity scores"
    candidates = [
        (0.35, 
         {"file_path": "paper1.md", "chunk_index": 0, "paper_title": "Test Paper"},
         "This is a test document about propensity score matching.")
    ]
    
    print(f"\nQuery: {query}")
    print(f"Number of candidates: {len(candidates)}")
    
    result = reranker.rerank(query, candidates, top_k=5)
    print(f"\nResult: {len(result)} candidates returned")
    print(f"Score: {result[0][0]:.4f}")
    print("✓ Handled single candidate correctly")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RERANKER TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Basic reranking
        reranker, query, candidates = test_basic_reranking()
        
        # Test 2: Detailed reranking
        test_detailed_reranking(reranker, query, candidates)
        
        # Test 3: Different query type
        test_different_query(reranker)
        
        # Test 4: Empty candidates
        test_empty_candidates(reranker)
        
        # Test 5: Single candidate
        test_single_candidate(reranker)
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())