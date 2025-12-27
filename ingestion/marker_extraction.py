import os
import sys
from utils.logger import get_ingestion_logger
from pathlib import Path
import json
import argparse
from datetime import datetime

# Setup logging
logger = get_ingestion_logger(Path(__file__).stem, max_files=5)

# Configure custom model weights storage location
MODEL_CACHE_DIR = os.environ.get('MARKER_MODEL_CACHE', '/scratch/sathishbabu.ki/marker-models')
os.environ['XDG_CACHE_HOME'] = MODEL_CACHE_DIR

# Log environment configuration
logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")
logger.info(f"XDG_CACHE_HOME set to: {os.environ['XDG_CACHE_HOME']}")

# Import marker modules after setting environment variables
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser


def process_single_pdf(pdf_file, output_dir, artifact_dict, config):
    """
    Process a single PDF file with Marker.
    
    Args:
        pdf_file: Path to PDF file
        output_dir: Output directory
        artifact_dict: Marker model dictionary
        config: Marker configuration
    
    Returns:
        tuple: (success: bool, filename: str)
    """
    pdf_name = pdf_file.stem
    output_name = pdf_name
    
    try:
        # Extract with Marker
        config_parser = ConfigParser(config)
        
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )
        
        rendered = converter(str(pdf_file))

        structured = {"pages": []}
        
        # Some Marker versions use .pages, some use .page_data. Try both.
        pages = getattr(rendered, "pages", None) or getattr(rendered, "page_data", [])
        
        for p in pages:
            page_obj = {
                "page_number": getattr(p, "page_number", None),
                "blocks": []
            }
        
            # blocks might be under .blocks or .elements depending on Marker version
            blocks = getattr(p, "blocks", None) or getattr(p, "elements", [])
        
            for b in blocks:
                page_obj["blocks"].append({
                    "type": getattr(b, "type", None),
                    "bbox": getattr(b, "bbox", None),
                    "text": getattr(b, "text", "") or getattr(b, "cleaned_text", "")
                })
        
            structured["pages"].append(page_obj)

        # Save output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        markdown_file = output_path / f"{output_name}.md"
        metadata_file = output_path / f"{output_name}_metadata.json"
        structured_file = output_path / f"{output_name}_layout.json"
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(rendered.markdown)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(rendered.metadata, f, indent=2)

        with open(structured_file, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)
        
        logger.info(f" Successfully processed: {output_name}")
        logger.info(f"  Markdown: {markdown_file}")
        logger.info(f"  Pages: {rendered.metadata.get('pages', 'N/A')}")
        
        return True, pdf_file.name
        
    except Exception as e:
        logger.error(f" Failed to process {pdf_file.name}: {str(e)}", exc_info=True)
        return False, pdf_file.name


def process_directory(input_dir: str, output_dir: str):
    """
    Process all PDFs in input directory sequentially (GPU-bound, no parallelism).
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save output files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Recursively find all PDFs
    pdf_files = sorted(list(input_path.rglob("*.pdf")))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in: {input_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Configuration for Marker
    config = {
        "output_format": "markdown",
        "paginate_output": False,
        "disable_image_extraction": True,  # Changed
        "force_ocr": True,
        "format_lines": True,
        "strip_existing_ocr": True,  # Changed - removes bad OCR first
        "extract_tables": True,
        "use_llm": False,  # Consider True for better layout understanding
        # "debug": True,  # Will show more info
        "keep_pageheader_in_output": False,
        "keep_pagefooter_in_output": False,
        "disable_tqdm": True,
    }
    
    # Create model dictionary once (shared across all PDFs)
    logger.info("Loading Marker models...")
    artifact_dict = create_model_dict()
    logger.info("Models loaded successfully")
    
    # Process PDFs sequentially
    success_count = 0
    failed_files = []
    skipped_count = 0
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n[{idx}/{len(pdf_files)}] {pdf_file.name}")
        
        pdf_name = pdf_file.stem
        output_name = pdf_name
        
        markdown_file = Path(output_dir) / f"{output_name}.md"
        metadata_file = Path(output_dir) / f"{output_name}_metadata.json"
        
        # Skip if output already exists
        if markdown_file.exists() and metadata_file.exists():
            logger.info(f"Skipping already processed file: {output_name}")
            skipped_count += 1
            continue
        
        # Process with Marker
        success, filename = process_single_pdf(
            pdf_file, output_dir, artifact_dict, config
        )
        
        if success:
            success_count += 1
        else:
            failed_files.append(filename)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files: {len(pdf_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Skipped (already processed): {skipped_count}")
    logger.info(f"Failed: {len(failed_files)}")
    
    if failed_files:
        logger.warning("\nFailed files:")
        for filename in failed_files:
            logger.warning(f"  - {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from academic PDFs using Marker (sequential, GPU-optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python marker_prod.py /path/to/pdfs /path/to/output
  
Environment Variables:
  MARKER_MODEL_CACHE: Custom location for model weights (default: /scratch/sathishbabu.ki/marker-models)
  MARKER_LOG_DIR: Custom location for logs (default: ./logs)
  
Notes:
  - Processes PDFs sequentially to avoid GPU memory issues
  - Skips already processed files automatically
  - Output naming: input.pdf → output.md
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Log job info
    logger.info("="*60)
    logger.info("MARKER PDF EXTRACTION JOB STARTED")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Processing mode: Sequential (GPU-optimized, no parallelism)")
    logger.info("="*60)
    
    process_directory(args.input_dir, args.output_dir)
    
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Job completed")


if __name__ == "__main__":
    main()