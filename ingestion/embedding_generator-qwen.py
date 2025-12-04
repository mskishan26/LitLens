import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import argparse
import os

from utils.logger import get_ingestion_logger

logger = get_ingestion_logger(Path(__file__).stem, max_files=5)


class EmbeddingConfig:
    """Configuration for embedding generation."""
    MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
    TRUNCATE_DIM = 1024
    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_CHUNK_OVERLAP = 200
    
    # Stage 1 Config (Paper Level)
    STAGE1_CHUNKING_STRATEGY = 'uniform'
    STAGE1_CHUNK_SIZE = 19000
    STAGE1_CHUNK_OVERLAP = 1500
    
    # Stage 2 Config (Chunk Level)
    STAGE2_CHUNKING_STRATEGY = 'paragraph'
    STAGE2_CHUNK_SIZE = 800
    STAGE2_CHUNK_OVERLAP = 2


class EmbeddingGenerator:
    """Handles document chunking, embedding generation, and index building."""
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        truncate_dim: Optional[int] = None,
        default_chunk_size: Optional[int] = None,
        default_chunk_overlap: Optional[int] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            embedding_model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            truncate_dim: Truncate embeddings to this dimension
            default_chunk_size: Default target tokens per chunk
            default_chunk_overlap: Default overlap tokens between chunks
        """
        # Load defaults from EmbeddingConfig if not provided
        embedding_model_name = embedding_model_name or EmbeddingConfig.MODEL_NAME
        truncate_dim = truncate_dim or EmbeddingConfig.TRUNCATE_DIM
        default_chunk_size = default_chunk_size or EmbeddingConfig.DEFAULT_CHUNK_SIZE
        default_chunk_overlap = default_chunk_overlap or EmbeddingConfig.DEFAULT_CHUNK_OVERLAP
        
        self.device = device
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        
        logger.info(f"Loading embedding model '{embedding_model_name}' on {device}")
        
        model_kwargs = {
            'dtype': torch.float16 if device == 'cuda' else torch.float32,
        }
        
        if truncate_dim:
            self.model = SentenceTransformer(
                embedding_model_name, 
                device=device,
                truncate_dim=truncate_dim,
                model_kwargs=model_kwargs
            )
            self.embedding_dim = truncate_dim
        else:
            self.model = SentenceTransformer(
                embedding_model_name, 
                device=device,
                model_kwargs=model_kwargs
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.tokenizer = self.model.tokenizer
        
        # Initialize two FAISS indices
        self.index1 = faiss.IndexFlatL2(self.embedding_dim)
        self.index2 = faiss.IndexFlatL2(self.embedding_dim)
        
        self.metadata1: List[Dict] = []
        self.metadata2: List[Dict] = []
        
        logger.info(f"Initialized with embedding dimension: {self.embedding_dim}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using markdown convention (\n\n)."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _chunk_text_fixed(
        self, 
        text: str, 
        metadata: Dict,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Tuple[str, Dict]]:
        """
        Chunk text by tokens with fixed size and overlap.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to chunks
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap tokens between chunks
        
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_meta = metadata.copy()
            chunk_meta.update({
                'chunk_index': chunk_index,
                'chunk_method': 'fixed',
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'token_start': start_idx,
                'token_end': end_idx,
                'token_length': end_idx - start_idx
            })
            
            chunks.append((chunk_text, chunk_meta))
            start_idx += chunk_size - chunk_overlap
            chunk_index += 1
        
        for _, meta in chunks:
            meta['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_text_uniform(
        self, 
        text: str, 
        metadata: Dict,
        target_chunk_size: int,
        chunk_overlap: int
    ) -> List[Tuple[str, Dict]]:
        """
        Chunk text into approximately equal-sized chunks.
        
        Creates k chunks of approximately equal size, rather than fixed-size chunks
        where the last one might be very small.
        
        Args:
            text: Text to chunk
            metadata: Base metadata
            target_chunk_size: Target tokens per chunk (e.g., 29000)
            chunk_overlap: Overlap tokens between chunks
        
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        if total_tokens <= target_chunk_size:
            chunk_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            chunk_meta = metadata.copy()
            chunk_meta.update({
                'chunk_index': 0,
                'chunk_method': 'uniform',
                'target_chunk_size': target_chunk_size,
                'chunk_overlap': chunk_overlap,
                'token_start': 0,
                'token_end': total_tokens,
                'token_length': total_tokens,
                'total_chunks': 1
            })
            return [(chunk_text, chunk_meta)]
        
        effective_size = target_chunk_size - chunk_overlap
        num_chunks = max(1, int(np.ceil((total_tokens - chunk_overlap) / effective_size)))
        actual_chunk_size = int(np.ceil((total_tokens + (num_chunks - 1) * chunk_overlap) / num_chunks))
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < total_tokens:
            end_idx = min(start_idx + actual_chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_meta = metadata.copy()
            chunk_meta.update({
                'chunk_index': chunk_index,
                'chunk_method': 'uniform',
                'target_chunk_size': target_chunk_size,
                'actual_chunk_size': actual_chunk_size,
                'chunk_overlap': chunk_overlap,
                'token_start': start_idx,
                'token_end': end_idx,
                'token_length': end_idx - start_idx,
                'total_chunks': num_chunks
            })
            
            chunks.append((chunk_text, chunk_meta))
            start_idx += actual_chunk_size - chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def _chunk_text_paragraph(
        self,
        text: str,
        metadata: Dict,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Tuple[str, Dict]]:
        """
        Chunk text by paragraphs with target token size.
        
        Combines consecutive paragraphs until target size is reached.
        Overlap is in number of paragraphs, not tokens.
        
        Args:
            text: Text to chunk
            metadata: Base metadata
            chunk_size: Target tokens per chunk
            chunk_overlap: Number of paragraphs to overlap
        
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        paragraphs = self._split_paragraphs(text)
        
        if not paragraphs:
            return []
        
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(paragraphs):
            current_chunk_paragraphs = []
            current_tokens = 0
            start_i = i  # Track starting position to prevent infinite loops
            
            while i < len(paragraphs):
                para = paragraphs[i]
                para_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))
                
                if current_tokens + para_tokens > chunk_size and current_chunk_paragraphs:
                    break
                
                current_chunk_paragraphs.append(para)
                current_tokens += para_tokens
                i += 1
            
            if current_chunk_paragraphs:
                chunk_text = '\n\n'.join(current_chunk_paragraphs)
                
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    'chunk_index': chunk_index,
                    'chunk_method': 'paragraph',
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'num_paragraphs': len(current_chunk_paragraphs),
                    'token_length': current_tokens
                })
                
                chunks.append((chunk_text, chunk_meta))
                chunk_index += 1
                
                # Calculate overlap: move back by overlap amount, but ensure forward progress
                overlap_paragraphs = min(chunk_overlap, len(current_chunk_paragraphs) - 1)
                i = max(i - overlap_paragraphs, start_i + 1)
            else:
                break
        
        for _, meta in chunks:
            meta['total_chunks'] = len(chunks)
        
        return chunks
    
    def _embed_texts(
        self,
        texts: List[str],
        batch_size: int = 8,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for texts with Qwen3 prompt format.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            is_query: Whether texts are queries (vs documents)
        
        Returns:
            Normalized embeddings array
        """
        prompt_name = "query" if is_query else None
        
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size}, is_query={is_query})")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name=prompt_name
        )
        
        return embeddings
    
    def add_documents(
        self,
        markdown_files: List[Path],
        index_num: int = 1,
        chunking_strategy: str = 'fixed',
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        batch_size: int = 8
    ) -> int:
        """
        Add documents to specified index.
        
        Args:
            markdown_files: List of markdown file paths
            index_num: 1 or 2 (paper-level or chunk-level)
            chunking_strategy: 'fixed', 'uniform', or 'paragraph'
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap in tokens (or paragraphs for paragraph strategy)
            batch_size: Batch size for embedding generation
        
        Returns:
            Number of chunks added
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.default_chunk_overlap
        
        index = self.index1 if index_num == 1 else self.index2
        metadata_list = self.metadata1 if index_num == 1 else self.metadata2
        
        chunk_fn_map = {
            'fixed': self._chunk_text_fixed,
            'uniform': self._chunk_text_uniform,
            'paragraph': self._chunk_text_paragraph
        }
        
        if chunking_strategy not in chunk_fn_map:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        chunk_fn = chunk_fn_map[chunking_strategy]
        
        logger.info(f"Processing {len(markdown_files)} files for index {index_num}")
        logger.info(f"Strategy: {chunking_strategy}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        
        all_chunks = []
        all_metadata = []
        
        for file_path in tqdm(markdown_files, desc=f"Chunking files for index {index_num}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                paper_title = file_path.stem
                first_line = text.split('\n')[0].strip()
                if first_line.startswith('#'):
                    paper_title = first_line.lstrip('#').strip()
                
                base_metadata = {
                    'paper_title': paper_title,
                    'file_path': str(file_path)
                }
                
                chunks = chunk_fn(text, base_metadata, chunk_size, chunk_overlap)
                
                for chunk_text, chunk_meta in chunks:
                    all_chunks.append(chunk_text)
                    all_metadata.append(chunk_meta)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not all_chunks:
            logger.warning("No chunks generated")
            return 0
        
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self._embed_texts(all_chunks, batch_size=batch_size, is_query=False)
        
        index.add(embeddings)
        metadata_list.extend(all_metadata)
        
        logger.info(f"Added {len(all_chunks)} chunks to index {index_num}")
        return len(all_chunks)
    
    def save(self, output_path: Path):
        """Save both indices and metadata to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index1, str(output_path / "index1_paper_level.faiss"))
        faiss.write_index(self.index2, str(output_path / "index2_chunk_level.faiss"))
        
        with open(output_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadata1': self.metadata1,
                'metadata2': self.metadata2,
                'config': {
                    'embedding_dim': self.embedding_dim,
                    'default_chunk_size': self.default_chunk_size,
                    'default_chunk_overlap': self.default_chunk_overlap,
                    'index1_purpose': 'paper_level',
                    'index2_purpose': 'chunk_level'
                }
            }, f)
        
        logger.info(f"Saved RAG store to {output_path}")
        logger.info(f"  Index 1 (paper-level): {self.index1.ntotal} vectors")
        logger.info(f"  Index 2 (chunk-level): {self.index2.ntotal} vectors")


def get_optimal_batch_sizes():
    """Determine optimal batch sizes based on available GPU memory."""
    if not torch.cuda.is_available():
        return 1, 2
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if gpu_memory_gb >= 70:
        return 2, 12
    elif gpu_memory_gb >= 35:
        return 1, 6
    else:
        return 1, 4


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for RAG system.")
    parser.add_argument("--stage1_input_dir", type=str, required=True, help="Directory containing markdown files for Stage 1")
    parser.add_argument("--stage2_input_dir", type=str, required=True, help="Directory containing markdown files for Stage 2")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save embeddings and indices")
    
    args = parser.parse_args()
    
    stage1_input_dir = Path(args.stage1_input_dir)
    stage2_input_dir = Path(args.stage2_input_dir)
    output_dir = Path(args.output_dir)
    
    if not stage1_input_dir.exists():
        logger.error(f"Stage 1 input directory does not exist: {stage1_input_dir}")
        return
    if not stage2_input_dir.exists():
        logger.error(f"Stage 2 input directory does not exist: {stage2_input_dir}")
        return
        
    # Auto-detect optimal batch sizes
    stage1_batch_size, stage2_batch_size = get_optimal_batch_sizes()
    logger.info(f"Using batch sizes: Stage 1={stage1_batch_size}, Stage 2={stage2_batch_size}")
    
    # Set environment variable for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Stage 1: Large chunks for paper-level retrieval
    logger.info("Starting Stage 1: Paper-level indexing...")
    markdown_files_stage1 = list(stage1_input_dir.glob("**/*.md"))
    
    if not markdown_files_stage1:
        logger.warning(f"No markdown files found in {stage1_input_dir}")
    else:
        generator.add_documents(
            markdown_files_stage1, 
            index_num=1,
            chunking_strategy=EmbeddingConfig.STAGE1_CHUNKING_STRATEGY,
            chunk_size=EmbeddingConfig.STAGE1_CHUNK_SIZE,
            chunk_overlap=EmbeddingConfig.STAGE1_CHUNK_OVERLAP,
            batch_size=stage1_batch_size
        )
    
    # Stage 2: Smaller semantic chunks for detailed retrieval
    logger.info("Starting Stage 2: Chunk-level indexing...")
    markdown_files_stage2 = list(stage2_input_dir.glob("**/*.md"))
    
    if not markdown_files_stage2:
        logger.warning(f"No markdown files found in {stage2_input_dir}")
    else:
        generator.add_documents(
            markdown_files_stage2,
            index_num=2,
            chunking_strategy=EmbeddingConfig.STAGE2_CHUNKING_STRATEGY,
            chunk_size=EmbeddingConfig.STAGE2_CHUNK_SIZE,
            chunk_overlap=EmbeddingConfig.STAGE2_CHUNK_OVERLAP,
            batch_size=stage2_batch_size
        )
    
    # Save indices
    generator.save(output_dir)
    logger.info("Embedding generation completed successfully.")


if __name__ == "__main__":
    main()