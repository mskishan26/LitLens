import sys
import os
import argparse
from pathlib import Path
import textwrap

# -----------------------------------------------------------------------------
# Path Setup - Centralized Entry Point Configuration
# -----------------------------------------------------------------------------
# Add src to python path (assuming this script is in src/inference)
# This allows all other modules to just import from 'utils', 'inference', etc.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Suppress console logs for cleaner CLI experience
# DO NOT FORGET TO SET THE HF_HOME VARIABLE, IT WILL REDOWNLOAD TO THE CACHE LOCATION AND EAT YOUR HOME DIR STORAGE
os.environ['HF_HOME'] = "/scratch/sathishbabu.ki/vllm_models/vllm/.cache/huggingface"
os.environ['RAG_CONSOLE_LOG_LEVEL'] = 'WARNING'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'

# Suppress transformers loading bars and info logs
try:
    from transformers import logging as hf_logging
    from transformers.utils import logging as hf_utils_logging
    hf_logging.set_verbosity_error()
    hf_utils_logging.disable_progress_bar()
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from utils.config_loader import load_config
from utils.logger import get_chat_logger
from inference.chat_pipeline import RAGPipeline

logger = get_chat_logger("cli_chat")


def display_sources(sources: list):
    """Display sources in a formatted way."""
    if not sources:
        return
    
    print("-" * 40)
    print("Sources:")
    for i, ctx in enumerate(sources, 1):
        if isinstance(ctx, dict):
            # Handle reranker output format
            if 'metadata' in ctx:
                title = ctx['metadata'].get('paper_title', 'Unknown Paper')
                fname = Path(ctx['metadata'].get('file_path', '')).name
                score = ctx.get('rerank_score', 0)
                print(f"{i}. {title}")
                print(f"   File: {fname} | Score: {score:.4f}")
            else:
                title = ctx.get('paper_title', 'Unknown Paper')
                fname = Path(ctx.get('file_path', '')).name
                print(f"{i}. {title}")
                print(f"   File: {fname}")
    print("-" * 40)


def display_hallucination_check(hallucination_result):
    """Display hallucination check results."""
    if hallucination_result is None:
        return
    
    # Handle both HallucinationResult object and dict
    if hasattr(hallucination_result, 'to_dict'):
        result = hallucination_result
        num_claims = result.num_claims
        num_grounded = result.num_grounded
        num_unsupported = result.num_unsupported
        grounding_ratio = result.grounding_ratio
        unsupported_claims = result.unsupported_claims
    else:
        # Dict format
        num_claims = hallucination_result.get('num_claims', 0)
        num_grounded = hallucination_result.get('num_grounded', 0)
        num_unsupported = hallucination_result.get('num_unsupported', 0)
        grounding_ratio = hallucination_result.get('grounding_ratio', 0)
        unsupported_claims = hallucination_result.get('unsupported_claims', [])
    
    if num_claims == 0:
        return
    
    print("\n" + "=" * 40)
    print("GROUNDING CHECK")
    print("=" * 40)
    
    if num_unsupported > 0:
        print(f"⚠️  {num_grounded}/{num_claims} claims supported ({grounding_ratio:.0%})")
        print("\nUnsupported claims:")
        for i, claim in enumerate(unsupported_claims, 1):
            # Wrap long claims
            if len(claim) > 80:
                print(f"  {i}. {claim[:77]}...")
            else:
                print(f"  {i}. {claim}")
    else:
        print(f"✓ All {num_claims} claims are grounded in source documents.")
    
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="RAG CLI Chat")
    parser.add_argument("--no-unload", action="store_true", help="Keep models loaded in memory between queries")
    parser.add_argument("--session-id", type=str, default=None, help="Custom session ID for logging")
    parser.add_argument("--replay", type=str, default=None, help="Path to JSONL session file to replay")
    parser.add_argument("--no-hallucination-check", action="store_true", help="Disable hallucination checking")
    args = parser.parse_args()

    # Handle replay mode
    if args.replay:
        from inference.session_logger import SessionLogger
        
        replay_file = Path(args.replay)
        if not replay_file.exists():
            print(f"Error: Session file not found: {replay_file}")
            return
        
        print(f"\n{'='*60}")
        print(f"Replaying session: {replay_file}")
        print(f"{'='*60}\n")
        
        history = SessionLogger.load_for_chat_replay(replay_file)
        for entry in history:
            print(f"[{entry['timestamp']}]")
            print(f"User: {entry['query']}")
            print(f"\nAssistant: {entry['answer']}")
            if entry['sources']:
                print(f"\nSources: {', '.join(entry['sources'][:3])}")
            print("\n" + "-"*40 + "\n")
        
        print(f"Total entries: {len(history)}")
        return

    print("Initializing RAG Pipeline... (this may take a minute)")
    
    # Load config to get session log path
    try:
        config = load_config()
        session_log_dir = Path(config['paths']['session_logs'])
        session_log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not load config for session logs: {e}")
        session_log_dir = Path(".")
        
    try:
        pipeline = RAGPipeline(
            no_unload=args.no_unload,
            enable_hallucination_check=not args.no_hallucination_check
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # Initialize JSONL session logger
    session_file = pipeline.init_session_logger(
        session_dir=session_log_dir,
        session_id=args.session_id
    )
    
    print("\n" + "="*60)
    print("RAG CLI Chat")
    print(f"Session Log (JSONL): {session_file}")
    print(f"Hallucination Check: {'Enabled' if pipeline.enable_hallucination_check else 'Disabled'}")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("Type 'debug' to show last query's debug info.")
    print("="*60 + "\n")

    last_entry_id = None
    
    while True:
        try:
            query = input("\nUser: ").strip()
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit', 'q'):
                print("Goodbye!")
                break
            
            # Debug command: show last query's detailed info
            if query.lower() == 'debug':
                if last_entry_id and session_file:
                    from inference.session_logger import SessionLogger, analyze_reranker_impact, analyze_hybrid_retrieval
                    
                    entry = SessionLogger.load_for_debugging(session_file, last_entry_id)
                    if entry:
                        print("\n" + "="*40)
                        print(f"Debug info for: {entry.get('query', 'N/A')[:50]}...")
                        print("="*40)
                        
                        # Timings
                        timings = entry.get('timings', {})
                        print(f"\nTimings:")
                        for k, v in timings.items():
                            print(f"  {k}: {v:.3f}s")
                        
                        # Reranker analysis
                        rerank_analysis = analyze_reranker_impact(entry)
                        print(f"\nReranker Impact:")
                        print(f"  Avg rank change: {rerank_analysis.get('avg_rank_change', 0):.2f}")
                        print(f"  Results improved: {rerank_analysis.get('results_improved', 0)}")
                        print(f"  Results degraded: {rerank_analysis.get('results_degraded', 0)}")
                        
                        # Hybrid retrieval analysis
                        hybrid_analysis = analyze_hybrid_retrieval(entry)
                        print(f"\nHybrid Retrieval:")
                        print(f"  Total candidates: {hybrid_analysis.get('total_candidates', 0)}")
                        print(f"  From BM25 only: {hybrid_analysis.get('selected_from_bm25_only', 0)}")
                        print(f"  From Embedding only: {hybrid_analysis.get('selected_from_embedding_only', 0)}")
                        print(f"  From both: {hybrid_analysis.get('selected_from_both', 0)}")
                        
                        print("="*40)
                    else:
                        print("Could not load debug info for last query.")
                else:
                    print("No previous query to debug.")
                continue
            
            print("\nAssistant: ", end="", flush=True)
            
            # Use streaming for better experience and to get sources
            stream = pipeline.answer_streaming(
                query, 
                return_metadata=True,
                session_file=None  # Don't use legacy text logging
            )
            
            sources = []
            answer_accumulated = ""
            hallucination_result = None
            
            try:
                for chunk in stream:
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        answer_accumulated += chunk
                    elif isinstance(chunk, dict):
                        if 'contexts' in chunk:
                            sources = chunk['contexts']
                        elif 'hallucination_check' in chunk:
                            hallucination_result = chunk['hallucination_check']
            except Exception as e:
                print(f"\n[Error during generation: {e}]")
                logger.error(f"Generation error: {e}", exc_info=True)
            
            print("\n")
            
            display_sources(sources)
            
            # Display hallucination check results
            display_hallucination_check(hallucination_result)
            
            # Track last entry ID for debug command
            if pipeline.session_logger:
                last_entry_id = f"{pipeline.session_logger.session_id}_{pipeline.session_logger._entry_count:04d}"

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}")

    print("Cleaning up...")
    pipeline.cleanup()
    
    # Print session summary
    if session_file and session_file.exists():
        from inference.session_logger import get_session_summary
        summary = get_session_summary(session_file)
        print(f"\nSession Summary:")
        print(f"  Total queries: {summary.get('total_queries', 0)}")
        print(f"  Successful: {summary.get('successful', 0)}")
        print(f"  Avg time: {summary.get('avg_time_seconds', 0):.2f}s")
        print(f"  Session file: {session_file}")


if __name__ == "__main__":
    main()