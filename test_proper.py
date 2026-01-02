import asyncio
import time
import uuid
from chat_pipeline import RAGPipelineV2

async def run_query(pipeline: RAGPipelineV2, query_text: str, delay: float = 0):
    """
    Simulates a single user request with an optional entry delay.
    """
    if delay > 0:
        await asyncio.sleep(delay)
    
    query_id = f"Q-{query_text[:5]}-{str(uuid.uuid4())[:4]}"
    print(f"--- [START] {query_id} (Arrived at T+{delay}s) ---")
    
    start_time = time.perf_counter()
    
    # We use answer_stream to see the transition between stages
    async for update in pipeline.answer_stream(
        query=query_text,
        conversation_id=f"session-{query_id}"
    ):
        current_time = time.perf_counter() - start_time
        msg_type = update.get("type")
        
        if msg_type == "status":
            print(f"[{query_id} @ {current_time:.2f}s] Status: {update.get('stage')}")
        elif msg_type == "token":
            # We only print the first token to avoid flooding the console
            pass 
        elif msg_type == "done":
            print(f"[{query_id} @ {current_time:.2f}s] COMPLETED. Total Duration: {update.get('total_duration_ms'):.0f}ms")

async def main():
    # 1. Initialize the actual pipeline
    # max_concurrent_generation=1 ensures we can see the semaphore queueing
    pipeline = RAGPipelineV2(
        max_concurrent_generation=1,
        reranker_auto_clear_cache=True
    )
    
    await pipeline.initialize()
    
    print("\n" + "="*80)
    print("STRESS TEST: ASYNCHRONOUS RETRIEVAL + SERIALIZED GENERATION")
    print("="*80)

    # 2. Define a batch of queries with different arrival times
    # Query 1: Heavy, arrives immediately
    # Query 2: Arrives 0.5s later, should finish retrieval while Query 1 generates
    # Query 3: Arrives 1.0s later
    tasks = [
        run_query(pipeline, "What are the tradeoffs and assumptions of intensity normalization using an internal standard reference of MALDI MSI datasets with multiple samples?", delay=0),
        run_query(pipeline, "What is the difference between qvalue and locfdr R packages? Which would be better for highly interdependent tests?", delay=0.5),
        run_query(pipeline, "For experiments consisting of only MSI data without accompanying MS/MS fragmentation, what are the current options, if any, for annotation of peptides/proteins?", delay=1.0)
    ]

    total_start = time.perf_counter()
    
    # 3. Execute all queries concurrently
    await asyncio.gather(*tasks)
    
    total_end = time.perf_counter() - total_start
    print("="*80)
    print(f"All queries finished in {total_end:.2f}s")
    print("="*80)

    # 4. Cleanup resources
    await pipeline.cleanup()

if __name__ == "__main__":
    # Ensure torch/multiprocessing compatibility for some environments
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass