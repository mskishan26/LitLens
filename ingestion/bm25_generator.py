import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from nltk.tokenize import word_tokenize
import nltk

from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Setup logging
from utils.logger import get_ingestion_logger


logger_ingestion = get_ingestion_logger(Path(__file__).stem, max_files=5)


class BM25IndexBuilder:
    """Builder class for creating BM25 indices from markdown documents."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize BM25 index builder.
        
        Args:
            input_dir: Directory containing *filtered.md files
            output_dir: Directory to save BM25 artifacts
            
        Raises:
            FileNotFoundError: If input_dir doesn't exist
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.documents: List[str] = []
        self.metadata: List[Dict[str, str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: Optional[List[List[str]]] = None
        
        # Validate input directory
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    def _load_markdown_files(self) -> None:
        """
        Load all filtered markdown files from input directory.
        
        Raises:
            ValueError: If no *filtered.md files found
            IOError: If file reading fails
        """
        md_files = list(self.input_dir.glob("*.md"))
        
        if not md_files:
            raise ValueError(f"No *filtered.md files found in {self.input_dir}")
        
        logger_ingestion.info(f"Found {len(md_files)} markdown files to process")
        
        for md_file in tqdm(md_files, desc="Loading markdown files"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents.append(content)
                    self.metadata.append({
                        'filename': md_file.name,
                        'filepath': str(md_file)
                    })
            except IOError as e:
                logger_ingestion.warning(f"Failed to read {md_file}: {e}")
                continue
            except Exception as e:
                logger_ingestion.error(f"Unexpected error reading {md_file}: {e}")
                continue
        
        if not self.documents:
            raise ValueError("No documents were successfully loaded")
        
        logger_ingestion.info(f"Successfully loaded {len(self.documents)} documents")

    def preprocess_markdown(self, text: str) -> str:
        """
        Clean markdown syntax and artifacts from text.
        
        Args:
            text: Raw markdown text
            
        Returns:
            Cleaned text with markdown removed
        """
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove LaTeX/Math expressions
        text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
        text = re.sub(r'\$[^$]+\$', '', text)
        
        # Remove markdown links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove markdown headers
        text = re.sub(r'#+\s+', '', text)
        
        # Remove markdown emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove table separators and pipes
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'-{3,}', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def better_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with preprocessing and filtering.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of cleaned, lowercase tokens
        """
        # Clean markdown
        clean_text = self.preprocess_markdown(text)
        
        # Tokenize
        tokens = word_tokenize(clean_text.lower())
        
        # Filter to alphanumeric tokens only
        tokens = [t for t in tokens if re.match(r'^[a-zA-Z0-9\-]+$', t)]
        
        return tokens
    
    def create_bm25_index(self) -> bool:
        """
        Create BM25 index from loaded documents.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            RuntimeError: If index creation fails critically
        """
        try:
            # Load documents
            self._load_markdown_files()
            
            # Tokenize corpus
            logger_ingestion.info("Tokenizing corpus...")
            self.tokenized_corpus = [
                self.better_tokenize(doc) 
                for doc in tqdm(self.documents, desc="Tokenizing")
            ]
            
            # Create BM25 index
            logger_ingestion.info("Building BM25 index...")
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            logger_ingestion.info("Successfully created BM25 index")
            return True
            
        except (FileNotFoundError, ValueError) as e:
            logger_ingestion.error(f"Input validation error: {e}")
            raise
        except Exception as e:
            logger_ingestion.error(f"Failed to create BM25 index: {e}", exc_info=True)
            raise RuntimeError(f"BM25 index creation failed: {e}") from e

    def save_bm25_artifacts(self) -> None:
        """
        Save BM25 index and metadata to disk.
        
        Raises:
            ValueError: If BM25 index hasn't been created
            IOError: If saving fails
        """
        if self.bm25 is None or self.tokenized_corpus is None:
            raise ValueError(
                "BM25 index not created. Call create_bm25_index() first."
            )
        
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger_ingestion.info(f"Saving artifacts to {self.output_dir}")
            
            # Save BM25 object
            with open(self.output_dir / 'bm25_index.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
            
            # Save tokenized corpus
            with open(self.output_dir / 'tokenized_corpus.pkl', 'wb') as f:
                pickle.dump(self.tokenized_corpus, f)
            
            # Save metadata
            with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            
            logger_ingestion.info(
                f"Successfully saved BM25 artifacts to {self.output_dir}"
            )
            
        except IOError as e:
            logger_ingestion.error(f"Failed to save artifacts: {e}")
            raise
        except Exception as e:
            logger_ingestion.error(f"Unexpected error saving artifacts: {e}", exc_info=True)
            raise

import argparse

def main() -> None:
    """
    Main function for BM25 index generation.
    """
    parser = argparse.ArgumentParser(description="Generate BM25 index from markdown documents.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *.md files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save BM25 artifacts")
    
    args = parser.parse_args()
    
    try:
        logger_ingestion.info("=== BM25 Index Generation ===")
        
        # Create and save index
        builder = BM25IndexBuilder(input_dir=args.input_dir, output_dir=args.output_dir)
        builder.create_bm25_index()
        builder.save_bm25_artifacts()
        
        logger_ingestion.info("=== Generation Complete ===")     
    except Exception as e:
        logger_ingestion.error(f"Fatal error in main: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()