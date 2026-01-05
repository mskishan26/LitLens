import asyncio
import time
import uuid
import os

from utils.config_loader import load_config
from utils.logger import configure_logging

# Configure logging before importing pipeline to ensure formats are correct
configure_logging(load_config())

from chat_pipeline import RAGPipelineV2

async def run_query(pipeline: RAGPipelineV2, query_text: str, delay: float = 0):
    """
    Simulates a single user request with an optional entry delay.
    """
    if delay > 0:
        await asyncio.sleep(delay)
    
    # Generate a short unique ID for logs
    query_id = f"Q-{str(uuid.uuid4())[:4]}"
    print(f"--- [START] {query_id} (Arrived at T+{delay}s) ---")
    
    start_time = time.perf_counter()
    
    # We use answer_stream to see the transition between stages
    try:
        async for update in pipeline.answer_stream(
            query=query_text,
            conversation_id=f"session-{query_id}",
            enable_hallucination_check=True
        ):
            current_time = time.perf_counter() - start_time
            msg_type = update.get("type")
            
            if msg_type == "status":
                stage = update.get('stage')
                if stage == "waiting_for_gpu":
                    pos = update.get('queue_position')
                    print(f"[{query_id} @ {current_time:.2f}s] ⏸️  WAITING FOR GPU (Queue Pos: {pos})")
                elif stage == "retrieval_complete":
                    n_papers = update.get('papers_found')
                    n_chunks = update.get('chunks_reranked')
                    print(f"[{query_id} @ {current_time:.2f}s] ✅ Retrieval Done ({n_papers} papers, {n_chunks} chunks)")
            
            elif msg_type == "token":
                # Only print the very first token to show generation started, then suppress
                if update.get("content"):
                    pass 
            
            elif msg_type == "done":
                duration = update.get('total_duration_ms')
                print(f"[{query_id} @ {current_time:.2f}s] 🏁 COMPLETED in {duration:.0f}ms")
                
            elif msg_type == "error":
                print(f"[{query_id} @ {current_time:.2f}s] ❌ ERROR: {update.get('message')}")

    except Exception as e:
        print(f"[{query_id}] CRITICAL FAIL: {e}")

async def main():
    print("\n" + "="*80)
    print("STRESS TEST: ASYNCHRONOUS RETRIEVAL + SERIALIZED GENERATION")
    print("="*80)

    # 1. Initialize the pipeline
    # NOTE: The new V2 pipeline loads config from files internally.
    pipeline = RAGPipelineV2()
    
    # 2. FORCE SERIALIZATION FOR DEMO PURPOSES
    # The new __init__ creates a semaphore based on config.yaml. 
    # To ensure we see the queueing effect in this test (even if config says 10),
    # we manually overwrite the semaphore to 1.
    
    await pipeline.initialize()

    # 3. Define a batch of queries with different arrival times
    # Query 1: Heavy, arrives immediately. Occupies GPU.
    # Query 2: Arrives 0.5s later. Should finish retrieval, then WAIT for Query 1.
    # Query 3: Arrives 1.0s later.
    tasks = [
        run_query(pipeline, "What are the tradeoffs and assumptions of intensity normalization using an internal standard reference of MALDI MSI datasets?", delay=0),
        run_query(pipeline, "What is the difference between qvalue and locfdr R packages? Which is better for interdependent tests?", delay=2.0),
        run_query(pipeline, "For experiments consisting of only MSI data without accompanying MS/MS fragmentation, what are the options for annotation?", delay=4.0)
    ]

    total_start = time.perf_counter()
    
    # 4. Execute all queries concurrently
    await asyncio.gather(*tasks)
    
    total_end = time.perf_counter() - total_start
    print("="*80)
    print(f"All queries finished in {total_end:.2f}s")
    print("="*80)

    # 5. Cleanup resources
    await pipeline.cleanup()

if __name__ == "__main__":
    # Ensure torch/multiprocessing compatibility
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass