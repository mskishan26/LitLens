#!/usr/bin/env python3
"""
Concurrent PDF Orientation Corrector

Detects and corrects PDF page orientations using Tesseract OSD with parallel processing.
Optimized for multi-core systems to process multiple PDFs simultaneously.
Uses PyMuPDF (fitz) instead of pdf2image/poppler for better compatibility.
"""

import os
import sys
from utils.logger import get_ingestion_logger
from pathlib import Path
import argparse
from datetime import datetime
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"ERROR: Missing required dependencies: {e}")
    print("Install with: pip install PyMuPDF opencv-python pillow")
    print("Also ensure tesseract-ocr is installed on your system")
    sys.exit(1)

# Setup logging
logger = get_ingestion_logger(Path(__file__).stem, max_files=5)


def render_page_to_image(page, dpi=300):
    """
    Render a PDF page to PIL Image using PyMuPDF.
    
    Args:
        page: fitz.Page object
        dpi: Resolution for rendering
    
    Returns:
        PIL.Image: Rendered page image
    """
    # Calculate zoom factor from DPI (72 is base DPI)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    # Render page to pixmap
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    return img


def detect_page_orientation(page_image, use_preprocessing=True):
    """
    Detect orientation with confidence scoring and optional preprocessing.
    
    Args:
        page_image: PIL Image object
        use_preprocessing: Apply image preprocessing for better detection
    
    Returns:
        tuple: (rotation_angle, confidence)
        - rotation_angle: 0, 90, 180, or 270 degrees
        - confidence: 0-15 (Tesseract OSD confidence scale)
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        try:
            if use_preprocessing:
                # Convert to grayscale
                img_array = np.array(page_image.convert('L'))
                
                # Increase contrast
                img_array = cv2.equalizeHist(img_array)
                
                # Binarize with Otsu's method
                _, img_array = cv2.threshold(
                    img_array, 0, 255, 
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                
                processed_image = Image.fromarray(img_array)
                processed_image.save(tmp.name, "PNG")
            else:
                page_image.save(tmp.name, "PNG")
            
            # Run Tesseract OSD
            result = subprocess.run(
                ["tesseract", tmp.name, "stdout", "--psm", "0"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return 0, 0.0
            
            rotation = 0
            confidence = 0.0
            
            for line in result.stdout.splitlines():
                if "Rotate:" in line:
                    rotation = int(line.split(":")[1].strip())
                elif "Orientation confidence:" in line:
                    confidence = float(line.split(":")[1].strip())
            
            return rotation, confidence
            
        except Exception as e:
            logger.debug(f"Orientation detection failed: {e}")
            return 0, 0.0
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)


def correct_single_pdf(pdf_path, output_path, confidence_threshold=5.0, dpi=300):
    """
    Detect and correct orientation for a single PDF file.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to save corrected PDF
        confidence_threshold: Minimum confidence to apply rotation (0-15 scale)
        dpi: DPI for page rendering (higher = better detection, slower)
    
    Returns:
        dict: Processing results including rotation info
    """
    pdf_name = Path(pdf_path).name
    
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        
        logger.info(f"Processing {pdf_name} ({num_pages} pages)")
        
        # Detect orientation for each page
        rotations = {}
        rejected = {}
        
        for i in range(num_pages):
            page = doc[i]
            
            # Render page to image
            page_image = render_page_to_image(page, dpi=dpi)
            
            # Detect orientation
            rotation, confidence = detect_page_orientation(
                page_image, 
                use_preprocessing=True
            )
            
            if rotation != 0:
                if confidence >= confidence_threshold:
                    rotations[i] = rotation
                    logger.info(
                        f"  Page {i+1}/{num_pages}: applying {rotation}° rotation "
                        f"(confidence: {confidence:.2f})"
                    )
                else:
                    rejected[i] = (rotation, confidence)
                    logger.debug(
                        f"  Page {i+1}/{num_pages}: rejecting {rotation}° rotation "
                        f"(low confidence: {confidence:.2f})"
                    )
        
        # Log summary
        if rotations:
            logger.info(f"  Applied {len(rotations)} rotation(s)")
        else:
            logger.info(f"  No rotations needed")
            
        if rejected:
            logger.info(f"  Rejected {len(rejected)} low-confidence rotation(s)")
        
        # Apply rotations using PyMuPDF
        for page_num, rotation_angle in rotations.items():
            page = doc[page_num]
            # PyMuPDF uses counter-clockwise rotation, but we need clockwise
            # So we negate the angle
            current_rotation = page.rotation
            new_rotation = (current_rotation + rotation_angle) % 360
            page.set_rotation(new_rotation)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save corrected PDF
        doc.save(str(output_path))
        doc.close()
        
        return {
            'success': True,
            'input_path': str(pdf_path),
            'output_path': str(output_path),
            'pages': num_pages,
            'rotations_applied': len(rotations),
            'rotations_rejected': len(rejected),
            'rotation_details': rotations,
            'rejected_details': {
                k: {'angle': v[0], 'confidence': v[1]} 
                for k, v in rejected.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_name}: {str(e)}", exc_info=True)
        return {
            'success': False,
            'input_path': str(pdf_path),
            'output_path': str(output_path),
            'error': str(e)
        }


def process_pdf_worker(args):
    """
    Worker function for parallel processing.
    Unpacks arguments and calls correct_single_pdf.
    """
    pdf_file, input_dir, output_dir, confidence_threshold, dpi = args
    
    # Generate output path with naming convention: filename_parentfolder
    pdf_name = pdf_file.stem
    parent_folder = pdf_file.parent.name
    
    # If PDF is directly in input_dir, don't add parent folder suffix
    if pdf_file.parent == input_dir:
        output_name = f"{pdf_name}.pdf"
    else:
        output_name = f"{pdf_name}_{parent_folder}.pdf"
    
    output_path = output_dir / output_name
    
    # Skip if already processed
    if output_path.exists():
        logger.info(f"Skipping {pdf_file.name} (already processed)")
        return {
            'success': True,
            'input_path': str(pdf_file),
            'output_path': str(output_path),
            'skipped': True
        }
    
    return correct_single_pdf(
        pdf_file, 
        output_path, 
        confidence_threshold=confidence_threshold,
        dpi=dpi
    )


def process_directory(input_dir, output_dir, confidence_threshold=5.0, 
                      num_workers=6, dpi=300):
    """
    Process all PDFs in input directory with parallel processing.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save corrected PDFs
        confidence_threshold: Confidence threshold for rotation (0-15)
        num_workers: Number of parallel worker processes
        dpi: DPI for page rendering
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs recursively
    pdf_files = sorted(list(input_path.rglob("*.pdf")))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in: {input_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    logger.info(f"Using {num_workers} worker processes")
    logger.info(f"Orientation confidence threshold: {confidence_threshold}")
    logger.info(f"DPI: {dpi}")
    
    # Prepare arguments for workers
    worker_args = [
        (pdf_file, input_path, output_path, confidence_threshold, dpi)
        for pdf_file in pdf_files
    ]
    
    # Process PDFs in parallel
    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        future_to_pdf = {
            executor.submit(process_pdf_worker, args): args[0] 
            for args in worker_args
        }
        
        # Process completed jobs
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.get('skipped'):
                    skipped_count += 1
                elif result['success']:
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Worker exception for {pdf_file.name}: {e}")
                failed_count += 1
                results.append({
                    'success': False,
                    'input_path': str(pdf_file),
                    'error': str(e)
                })
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files: {len(pdf_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Skipped (already processed): {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    
    # Log failed files
    if failed_count > 0:
        logger.warning("\nFailed files:")
        for result in results:
            if not result['success']:
                logger.warning(f"  - {result['input_path']}")
                logger.warning(f"    Error: {result.get('error', 'Unknown')}")
    
    # Log rotation statistics
    total_rotations = sum(
        result.get('rotations_applied', 0) 
        for result in results if result['success']
    )
    total_rejected = sum(
        result.get('rotations_rejected', 0) 
        for result in results if result['success']
    )
    
    if total_rotations > 0 or total_rejected > 0:
        logger.info("\nRotation Statistics:")
        logger.info(f"  Total rotations applied: {total_rotations}")
        logger.info(f"  Total rotations rejected: {total_rejected}")


def main():
    parser = argparse.ArgumentParser(
        description='Correct PDF page orientations in parallel using Tesseract OSD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with 6 workers (default confidence)
    python page_rotation.py --input_dir /path/to/input --output_dir /path/to/output --workers 6
    
    # Conservative threshold for fewer false positives
    python page_rotation.py --input_dir /path/to/input --output_dir /path/to/output --confidence 7.0 --workers 8
    
    # Higher DPI for better detection (slower)
    python page_rotation.py --input_dir /path/to/input --output_dir /path/to/output --dpi 400 --workers 4

Naming Convention:
    Input:  /input/subdir/paper.pdf
    Output: /output/paper_subdir.pdf
    
    Input:  /input/paper.pdf (no subdirectory)
    Output: /output/paper.pdf

Confidence Scale (0-15):
    3-5:  Moderate threshold (default: 5.0)
    7-10: Conservative threshold
    Higher values = fewer false positives, more selective rotation

Dependencies:
    pip install PyMuPDF opencv-python pillow
    System: tesseract-ocr (apt-get install tesseract-ocr)
        """
    )
    
    parser.add_argument(
        '--input_dir',
        required=True,
        type=str, 
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str, 
        help='Directory to save corrected PDFs'
    )
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=5.0,
        help='Confidence threshold for orientation detection (0-15, default: 5.0)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=max(1, multiprocessing.cpu_count() - 2),
        help=f'Number of parallel worker processes (default: {max(1, multiprocessing.cpu_count() - 2)} = CPU cores - 2)'
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for page rendering (default: 300, higher = better but slower)'
    )
    parser.add_argument(
        '--single_file', 
        action='store_true',
        help='Change the CLI to accept a single file path, calls correct_single_pdf'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0 <= args.confidence <= 15):
        logger.error(
            f"Confidence must be between 0 and 15, got: {args.confidence}"
        )
        sys.exit(1)
    
    if args.workers < 1:
        logger.error(f"Workers must be >= 1, got: {args.workers}")
        sys.exit(1)
    
    # Log job info
    logger.info("="*60)
    logger.info("ORIENTATION CORRECTION JOB STARTED")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Confidence threshold: {args.confidence}")
    logger.info(f"Worker processes: {args.workers}")
    logger.info(f"Available CPU cores: {multiprocessing.cpu_count()}")
    logger.info("="*60)
    
    if args.single_file:
        correct_single_pdf(
            args.input_dir,
            args.output_dir,
            confidence_threshold=args.confidence,
            dpi=args.dpi
        )
    else:
        process_directory(
            args.input_dir,
            args.output_dir,
            confidence_threshold=args.confidence,
            num_workers=args.workers,
            dpi=args.dpi
        )
    
    logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Job completed")


if __name__ == "__main__":
    main()