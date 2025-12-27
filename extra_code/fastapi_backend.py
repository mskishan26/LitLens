"""
FastAPI backend integration for embedding search.
Shows proper startup/shutdown, error handling, and endpoint design.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from embedding_search_fixed import EmbeddingSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    collection_num: int = Field(1, ge=1, le=2, description="Collection to search (1 or 2)")
    k: int = Field(5, ge=1, le=100, description="Number of results")
    file_path_filter: Optional[List[str]] = Field(None, description="Optional list of file paths to filter")


class SearchResult(BaseModel):
    distance: float
    file_path: str
    chunk_id: Optional[str] = None
    text: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    collections_loaded: bool
    collection_stats: Optional[Dict] = None
    system_info: Optional[Dict] = None


# Initialize FastAPI app
app = FastAPI(
    title="Academic Paper RAG API",
    description="Embedding search for academic papers in biostatistics and causal inference",
    version="1.0.0"
)


# Global state for embedding searcher
embedding_searcher: Optional[EmbeddingSearch] = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize embedding searcher on startup.
    This runs once when the server starts, not per worker.
    """
    global embedding_searcher
    
    try:
        logger.info("Initializing embedding searcher...")
        
        # Configure these via environment variables in production
        MODEL_NAME = "infgrad/Jasper-Token-Compression-600M"
        DB_PATH = "/mnt/e/data_files/embeddings"  # Replace with your path
        
        embedding_searcher = EmbeddingSearch(embedding_model_name=MODEL_NAME)
        embedding_searcher.load(Path(DB_PATH))
        
        logger.info("✓ Embedding searcher initialized successfully")
        logger.info(f"  Health: {embedding_searcher.health_check()}")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize embedding searcher: {e}")
        raise RuntimeError(f"Startup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global embedding_searcher
    logger.info("Shutting down...")
    embedding_searcher = None


# Dependency to get searcher instance
def get_searcher() -> EmbeddingSearch:
    """FastAPI dependency for accessing the embedding searcher."""
    if embedding_searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding searcher not initialized. Check server logs."
        )
    return embedding_searcher


# === API Endpoints ===

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Academic Paper RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(searcher: EmbeddingSearch = Depends(get_searcher)):
    """
    Health check endpoint for monitoring.
    """
    try:
        system_info = searcher.health_check()
        collection_stats = searcher.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            model_loaded=system_info.get("model_loaded", False),
            collections_loaded=system_info.get("collections_loaded", False),
            collection_stats=collection_stats,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            collections_loaded=False
        )


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    searcher: EmbeddingSearch = Depends(get_searcher)
):
    """
    Search for relevant paper chunks.
    
    Example request:
    ```json
    {
        "query": "What are causal inference assumptions?",
        "collection_num": 2,
        "k": 10,
        "file_path_filter": ["paper1.pdf", "paper2.pdf"]
    }
    ```
    """
    try:
        # Convert file_path_filter list to set if provided
        file_filter = set(request.file_path_filter) if request.file_path_filter else None
        
        # Perform search
        results = searcher.search(
            query=request.query,
            collection_num=request.collection_num,
            k=request.k,
            file_path_filter=file_filter
        )
        
        # Format results
        formatted_results = [
            SearchResult(
                distance=distance,
                file_path=metadata.get("file_path", "unknown"),
                chunk_id=metadata.get("chunk_id"),
                text=text,
                metadata=metadata
            )
            for distance, metadata, text in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/search", response_model=SearchResponse)
async def search_get(
    query: str = Query(..., min_length=1, max_length=2000),
    collection_num: int = Query(1, ge=1, le=2),
    k: int = Query(5, ge=1, le=100),
    searcher: EmbeddingSearch = Depends(get_searcher)
):
    """
    GET version of search endpoint for simple queries.
    
    Example: /search?query=causal+inference&collection_num=2&k=10
    """
    try:
        results = searcher.search(
            query=query,
            collection_num=collection_num,
            k=k,
            file_path_filter=None
        )
        
        formatted_results = [
            SearchResult(
                distance=distance,
                file_path=metadata.get("file_path", "unknown"),
                chunk_id=metadata.get("chunk_id"),
                text=text,
                metadata=metadata
            )
            for distance, metadata, text in results
        ]
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total_results=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers", response_model=Dict[str, Any])
async def list_papers(
    collection_num: int = Query(1, ge=1, le=2),
    limit: int = Query(1000, ge=1, le=5000),
    searcher: EmbeddingSearch = Depends(get_searcher)
):
    """
    List all unique paper file paths in a collection.
    """
    try:
        papers = searcher.list_papers(collection_num=collection_num, limit=limit)
        
        return {
            "collection_num": collection_num,
            "total_papers": len(papers),
            "papers": papers,
            "note": "Limited to first 1000 documents" if limit == 1000 else None
        }
        
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}", response_model=Dict[str, Any])
async def get_document(
    doc_id: str,
    collection_num: int = Query(1, ge=1, le=2),
    searcher: EmbeddingSearch = Depends(get_searcher)
):
    """
    Retrieve a specific document by ID.
    """
    try:
        doc = searcher.get_document_by_id(doc_id, collection_num)
        
        if doc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found in collection {collection_num}"
            )
        
        return doc
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict[str, Any])
async def collection_stats(searcher: EmbeddingSearch = Depends(get_searcher)):
    """Get collection statistics."""
    try:
        return searcher.get_collection_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # For development only - use gunicorn/uvicorn in production
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
