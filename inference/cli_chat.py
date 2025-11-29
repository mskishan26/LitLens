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
os.environ['RAG_CONSOLE_LOG_LEVEL'] = 'WARNING'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

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

def main():
    parser = argparse.ArgumentParser(description="RAG CLI Chat")
    parser.add_argument("--no-unload", action="store_true", help="Keep models loaded in memory between queries")
    args = parser.parse_args()

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
        pipeline = RAGPipeline(no_unload=args.no_unload)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    # Create session log file
    from datetime import datetime
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_file = session_log_dir / f"chat_session_{session_id}.txt"
    
    print("\n" + "="*60)
    print("RAG CLI Chat")
    print(f"Session Log: {session_file.absolute()}")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            query = input("\nUser: ").strip()
            if not query:
                continue
                
            if query.lower() in ('exit', 'quit', 'q'):
                print("Goodbye!")
                break
            
            print("\nAssistant: ", end="", flush=True)
            
            # Use streaming for better experience and to get sources
            stream = pipeline.answer_streaming(
                query, 
                return_metadata=True,
                session_file=session_file
            )
            
            sources = []
            answer_accumulated = ""
            
            try:
                for chunk in stream:
                    if isinstance(chunk, str):
                        print(chunk, end="", flush=True)
                        answer_accumulated += chunk
                    elif isinstance(chunk, dict) and 'contexts' in chunk:
                        sources = chunk['contexts']
            except Exception as e:
                print(f"\n[Error during generation: {e}]")
                logger.error(f"Generation error: {e}", exc_info=True)
            
            print("\n")
            
            if sources:
                print("-" * 40)
                print("Sources:")
                for i, ctx in enumerate(sources, 1):
                    title = ctx['metadata'].get('paper_title', 'Unknown Paper')
                    fname = Path(ctx['metadata'].get('file_path', '')).name
                    print(f"{i}. {title}")
                    print(f"   File: {fname}")
                    # Optional: Print snippet
                    # snippet = textwrap.shorten(ctx['text'], width=100, placeholder="...")
                    # print(f"   Snippet: {snippet}")
                print("-" * 40)

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}")

    print("Cleaning up...")
    pipeline.cleanup()

if __name__ == "__main__":
    main()
