#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartGlassOCR v4.0
Advanced OCR engine optimized for smart glasses with enhanced text processing
No AI dependencies - Optimized for better image processing and OCR results

Copyright (c) 2025
"""

import os
import uuid
import time
import re
import logging
import json
import string
import math
import threading
import numpy as np
import concurrent.futures
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter

# Import from local modules
from .model import ImageType, ProcessingStrategy, DocumentStructure, ImageStats
from .utils import MemoryManager, clean_text, order_points, generate_unique_filename
from .image_processing import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("smart_glass_ocr.log")
    ]
)
logger = logging.getLogger("SmartGlass-OCR")

# Check image processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.error("OpenCV not available. Image processing will be limited.")

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.error("PIL not available. Image processing will be limited.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.error("Tesseract not available. OCR functionality will be disabled.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.error("pdf2image not available. PDF processing will be disabled.")

# NLP libraries - using minimal dependencies
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.probability import FreqDist
    
    # Download NLTK resources if needed
    nltk_resources = ['punkt', 'stopwords']
    for resource in nltk_resources:
        try:
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    
    NLTK_AVAILABLE = True
    
    # Load stopwords with better error handling
    try:
        from nltk.corpus import stopwords
        STOPWORDS_EN = set(stopwords.words('english'))
    except:
        STOPWORDS_EN = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                      "when", "where", "how", "which", "who", "whom", "this", "that", "these",
                      "those", "then", "just", "so", "than", "such", "both", "through", "about",
                      "for", "is", "of", "while", "during", "to", "from"}
        logging.warning("Failed to load English stopwords, using fallback")
    
    try:
        STOPWORDS_ID = set(stopwords.words('indonesian'))
    except:
        # Fallback stopwords for Indonesian
        STOPWORDS_ID = {'yang', 'dan', 'di', 'ini', 'itu', 'dari', 'dengan', 'untuk', 'pada', 'adalah',
                        'ke', 'tidak', 'ada', 'oleh', 'juga', 'akan', 'bisa', 'dalam', 'saya', 'kamu', 
                        'kami', 'mereka', 'dia', 'nya', 'tersebut', 'dapat', 'sebagai', 'telah', 'bahwa',
                        'atau', 'jika', 'maka', 'sudah', 'saat', 'ketika', 'karena'}
        logging.warning("Failed to load Indonesian stopwords, using fallback")
    
    # Combined stopwords
    STOPWORDS = STOPWORDS_EN.union(STOPWORDS_ID)
    
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS = set()
    logging.warning("NLTK libraries not available, using simplified text processing")

# Try to load more specific OCR models
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # Initialize reader in a separate thread to avoid blocking startup
    def init_easyocr():
        global reader
        reader = easyocr.Reader(['en', 'id'])  # Initialize with English and Indonesian
        logging.info("EasyOCR initialized successfully")
        
    easyocr_thread = threading.Thread(target=init_easyocr)
    easyocr_thread.daemon = True  # Set as daemon so it doesn't block program exit
    easyocr_thread.start()
    
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available, using Tesseract only")

# Try to load PaddleOCR as another option
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
    # Initialize in a separate thread
    def init_paddleocr():
        global paddle_ocr
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logging.info("PaddleOCR initialized successfully")
        
    paddle_thread = threading.Thread(target=init_paddleocr)
    paddle_thread.daemon = True
    paddle_thread.start()
    
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    logging.warning("PaddleOCR not available")

class SmartGlassOCR:
    """Advanced OCR engine optimized for smart glasses with enhanced image processing"""
    
    def __init__(self, config=None):
        """
        Initialize the OCR engine with specified configuration
        
        Args:
            config: Dictionary with configuration parameters
        """
        # Default configuration
        self.config = {
            "upload_folder": "/tmp/ocr_uploads",
            "allowed_extensions": {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'},
            "max_file_size": 16 * 1024 * 1024,  # 16 MB
            "tesseract_path": None,
            "tesseract_data_path": None,
            "default_language": "eng+ind",
            "summary_length": 200,
            "summary_style": "concise",  # Options: concise, detailed, bullets, structured
            "use_gpu": False,
            "max_workers": 4,      # For parallel processing
            "ocr_timeout": 30,     # Timeout for OCR process in seconds
            "preprocessing_level": "auto",  # auto, minimal, standard, aggressive
            "debug_mode": False,
            "cache_processed_images": True,
            "cache_size_mb": 500,  # Maximum cache size in MB
            "min_confidence_threshold": 60.0,  # Minimum acceptable confidence
            "use_all_available_engines": True,  # Try all OCR engines
            "perform_ocr_verification": True,   # Verify OCR results
            "auto_rotate": True,                # Auto-rotate images if needed
            "max_image_dimension": 3000,        # Maximum image dimension for processing
            "save_debug_images": False,         # Save debug images
            "debug_output_dir": "/tmp/ocr_debug",
            "enable_text_correction": True,     # Enable post-OCR text correction
            "enable_structured_extraction": True,  # Enable structured data extraction
            "enhance_scientific_text": True,    # Enhance scientific notation and formulas
            "enhance_table_detection": True,    # Improved table detection and extraction
            "language_specific_processing": True, # Apply language-specific optimizations
            "apply_contextual_corrections": True, # Use contextual clues for corrections
            "extract_key_insights": True,       # Extract key insights from text
            "organized_output_format": True,    # Provide well-organized output
            "confidence_scoring": "weighted",   # How to calculate confidence: simple, weighted, adaptive
            "lightweight_mode": False,          # Mode for limited resources devices
            "offline_mode": True,               # Use only locally available methods
            "enhanced_image_processing": True,  # Use enhanced image processing techniques
            "multi_page_processing": True,      # Process multi-page documents
            "adaptive_binarization": True,      # Use adaptive binarization for better text extraction
            "edge_enhancement": True,           # Enhance edges for better text detection
            "noise_reduction": True,            # Apply noise reduction techniques
            "shadow_removal": True,             # Remove shadows from images
            "perspective_correction": True,     # Correct perspective distortion
            "contrast_enhancement": True,       # Enhance contrast
            "text_line_detection": True,        # Detect text lines for better OCR
        }
        
        # Override with user config
        if config:
            self.config.update(config)
        
        # Ensure upload directory exists
        os.makedirs(self.config["upload_folder"], exist_ok=True)
        
        # Create debug directory if needed
        if self.config["save_debug_images"]:
            os.makedirs(self.config["debug_output_dir"], exist_ok=True)
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(self.config["cache_size_mb"])
        
        # Initialize image processor
        self.image_processor = ImageProcessor(self.config)
        
        # Configure Tesseract
        if TESSERACT_AVAILABLE:
            self._configure_tesseract()
        
        # Initialize OCR engines
        self.ocr_engines = {}
        
        if TESSERACT_AVAILABLE:
            self.ocr_engines["tesseract"] = {
                "available": True,
                "best_for": [ImageType.DOCUMENT, ImageType.HIGH_CONTRAST, ImageType.RECEIPT, 
                             ImageType.BOOK_PAGE, ImageType.NEWSPAPER, ImageType.FORM]
            }
        
        # Initialize EasyOCR (will be done in background)
        if EASYOCR_AVAILABLE:
            self.ocr_engines["easyocr"] = {
                "available": True,
                "best_for": [ImageType.NATURAL, ImageType.SIGNAGE, ImageType.MIXED, 
                             ImageType.HANDWRITTEN, ImageType.PRESENTATION]
            }
        
        # Initialize PaddleOCR (will be done in background)
        if PADDLE_OCR_AVAILABLE:
            self.ocr_engines["paddleocr"] = {
                "available": True,
                "best_for": [ImageType.MIXED, ImageType.LOW_QUALITY, ImageType.NATURAL, 
                             ImageType.SCIENTIFIC, ImageType.TABLE]
            }
        
        # Track processing performance for adaptive optimization
        self.processing_stats = {
            "image_types": {},
            "processing_times": {},
            "success_rates": {}
        }
        
        # Initialize version
        self.version = "4.1.0"  # Updated version with enhanced image processing
        
        logger.info(f"SmartGlassOCR v{self.version} initialized")
        logger.info(f"Available OCR engines: {list(self.ocr_engines.keys())}")
        logger.info(f"Running in {'lightweight' if self.config['lightweight_mode'] else 'standard'} mode")
    
    def _configure_tesseract(self):
        """Configure Tesseract based on OS and user settings"""
        if self.config["tesseract_path"]:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]
        elif os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        elif os.path.exists('/usr/bin/tesseract'):
            # Linux path
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        elif os.path.exists('/usr/local/bin/tesseract'):
            # macOS path
            pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        
        # Configure Tesseract data path if provided
        if self.config["tesseract_data_path"]:
            os.environ["TESSDATA_PREFIX"] = self.config["tesseract_data_path"]
        
        # Verify Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Error getting Tesseract version: {e}")

    def process_file(self, file_path: str, original_filename: str = None, language: str = None, 
                    page: int = 0, summary_length: int = None, 
                    summary_style: str = None) -> dict:
        """
        Process a file (image or PDF) and extract text with summarization
        
        Args:
            file_path: Path to the file
            original_filename: Original filename (if different from file_path)
            language: OCR language (default from config)
            page: Page number for PDF (0-based)
            summary_length: Maximum summary length
            summary_style: Style of summary (concise, detailed, bullets, structured)
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        # Set original filename if not provided
        if original_filename is None:
            original_filename = os.path.basename(file_path)
        
        # Use default values if not provided
        language = language or self.config["default_language"]
        summary_length = summary_length or self.config["summary_length"]
        summary_style = summary_style or self.config["summary_style"]
        
        # Check file extension
        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in self.config["allowed_extensions"]:
            return {"status": "error", "message": "Unsupported file type"}
        
        try:
            # Handle PDF vs image
            is_pdf = ext == 'pdf'
            
            if is_pdf:
                if not PDF2IMAGE_AVAILABLE:
                    return {"status": "error", "message": "PDF processing not available"}
                
                # Convert PDF to image
                logger.info(f"Converting PDF to image: {file_path}, page {page}")
                image_path, total_pages = self._convert_pdf_to_image(file_path, page)
                
                if not image_path:
                    return {"status": "error", "message": "Failed to convert PDF to image"}
                    
                # Process the image
                image_results = self._process_image(image_path, language)
                
                # Add PDF-specific metadata
                image_results["metadata"].update({
                    "file_type": "pdf",
                    "page": page,
                    "total_pages": total_pages
                })
                
                # Clean up temporary image
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary PDF image: {e}")
            else:
                # Process the image directly
                logger.info(f"Processing image: {file_path}")
                image_results = self._process_image(file_path, language)
                image_results["metadata"]["file_type"] = "image"
            
            # Generate summary if text was extracted successfully
            if image_results["status"] in ["success", "partial_success"] and image_results.get("text"):
                from .text_processing import generate_summary, detect_document_structure, extract_key_insights
                
                text = image_results.get("text", "")
                
                # Use enhanced extractive summarization
                summary = generate_summary(text, max_length=summary_length, style=summary_style)
                image_results["summary"] = summary
                
                # Extract document structure
                structure = detect_document_structure(text)
                image_results["document_structure"] = structure.value
                
                # Extract key insights if enabled
                if self.config["extract_key_insights"] and len(text) > 200:
                    insights = extract_key_insights(text)
                    image_results["key_insights"] = insights
            else:
                image_results["summary"] = ""
            
            # Add processing time
            processing_time = time.time() - start_time
            image_results["metadata"]["processing_time_ms"] = round(processing_time * 1000, 2)
            
            # Update processing stats
            self._update_processing_stats(image_results, processing_time)
            
            # Apply organized output format if enabled
            if self.config["organized_output_format"]:
                from .information_extraction import organize_output
                image_results = organize_output(image_results)
            
            return image_results
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error", 
                "message": f"Processing failed: {str(e)}",
                "metadata": {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }
    
    def process_batch(self, file_paths: List[str], language: str = None, 
                     summary_length: int = None, summary_style: str = None) -> Dict[str, dict]:
        """
        Process multiple files in parallel
        
        Args:
            file_paths: List of file paths to process
            language: OCR language (default from config)
            summary_length: Maximum summary length
            summary_style: Style of summary (concise, detailed, bullets, structured)
            
        Returns:
            Dictionary mapping file paths to their OCR results
        """
        language = language or self.config["default_language"]
        summary_length = summary_length or self.config["summary_length"]
        summary_style = summary_style or self.config["summary_style"]
        
        results = {}
        
        # Determine optimal number of workers based on file count and CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        optimal_workers = min(cpu_count, len(file_paths), self.config["max_workers"])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit tasks
            future_to_path = {
                executor.submit(
                    self.process_file, 
                    path, 
                    language=language, 
                    summary_length=summary_length,
                    summary_style=summary_style
                ): path for path in file_paths
            }
            
            # Process completions
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=self.config["ocr_timeout"])
                    results[path] = result
                except concurrent.futures.TimeoutError:
                    results[path] = {
                        "status": "error",
                        "message": "OCR process timed out"
                    }
                except Exception as e:
                    results[path] = {
                        "status": "error",
                        "message": f"Error: {str(e)}"
                    }
        
        return results
    
    def _process_image(self, image_path: str, language: str) -> dict:
        """Process an image through enhanced pipeline and OCR"""
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            return {"status": "error", "message": "Required image processing libraries not available"}
        
        try:
            # Step 1: Read and analyze the image to determine optimal processing
            image = cv2.imread(image_path)
            
            if image is None:
                # Try with PIL if OpenCV fails
                try:
                    pil_image = Image.open(image_path)
                    if pil_image.mode == 'RGBA':
                        pil_image = pil_image.convert('RGB')
                    temp_jpg = f"{image_path}_temp.jpg"
                    pil_image.save(temp_jpg)
                    image = cv2.imread(temp_jpg)
                    os.remove(temp_jpg)
                except Exception as e:
                    logger.error(f"Error converting image: {e}")
                    return {"status": "error", "message": "Failed to read image"}
            
            if image is None:
                return {"status": "error", "message": "Failed to read image file"}
            
            # Step 2: Analyze image with enhanced analysis for better type detection
            image_stats = self.image_processor.analyze_image(image)
            
            # For ID cards, force lightweight mode for faster processing
            if image_stats.image_type == ImageType.ID_CARD:
                self.config["lightweight_mode"] = True
                logger.info("Detected ID card, enabling lightweight mode for faster processing")
            
            logger.info(f"Enhanced image analysis: {image_stats.image_type.value}, " 
                    f"{image_stats.width}x{image_stats.height}, "
                    f"brightness: {image_stats.brightness:.1f}, "
                    f"contrast: {image_stats.contrast:.1f}, "
                    f"blur: {image_stats.blur:.1f}")
            
            # Step 3: Determine the best processing strategy based on image type and stats
            strategy = self.image_processor.determine_processing_strategy(image_stats)
            logger.info(f"Using processing strategy: {strategy.value}")
            
            # If auto rotation is enabled, check and rotate if needed
            if self.config["auto_rotate"] and image_stats.image_type != ImageType.NATURAL:
                image = self.image_processor.auto_rotate(image)
            
            # Step 4: Apply optimized preprocessing with enhanced methods
            processed_images, image_data = self.image_processor.preprocess_image(image, image_stats, strategy)
            
            # Save debug images if enabled
            if self.config["save_debug_images"]:
                self._save_debug_images(image_data, image_path)
            
            # Step 5: Perform OCR using the optimal engine sequence with improved confidence scoring
            from .ocr_engines import OCREngineManager
            
            # Initialize OCR engine manager
            ocr_manager = OCREngineManager(self.config)
            
            # Perform OCR
            best_engine, text, confidence, layout_info = ocr_manager.perform_ocr(
                processed_images, image_data, language, image_stats
            )
            
            # Step 6: Apply enhanced rule-based text correction if enabled
            if self.config["enable_text_correction"] and len(text) > 10:
                from .text_processing import post_process_text
                text = post_process_text(text, image_stats.image_type)
            
            # Step 7: Clean and format the extracted text with improved formatting
            from .text_processing import format_text
            formatted_text = format_text(text, layout_info)
            
            # Step 8: Extract additional information with rule-based methods
            from .text_processing import detect_language
            detected_language = detect_language(formatted_text)
            
            # Extract structured information if enabled
            structured_info = None
            if self.config["enable_structured_extraction"] and formatted_text:
                from .information_extraction import extract_structured_info
                structured_info = extract_structured_info(formatted_text, image_stats.image_type)
            
            # Clean up memory if not caching
            if not self.config["cache_processed_images"]:
                image_data.clear()
            
            # Determine status based on enhanced criteria
            status = "success"
            if confidence < 30 or len(formatted_text.strip()) < 5:
                status = "poor_quality"
            elif confidence < 60:
                status = "partial_success"
            
            # Prepare result with enhanced metadata
            result = {
                "status": status,
                "text": formatted_text,
                "confidence": confidence,
                "metadata": {
                    "detected_language": detected_language,
                    "structured_info": structured_info,
                    "image_type": image_stats.image_type.value,
                    "best_engine": best_engine,
                    "layout_info": layout_info,
                    "image_stats": {
                        "width": image_stats.width,
                        "height": image_stats.height,
                        "brightness": round(image_stats.brightness, 2),
                        "contrast": round(image_stats.contrast, 2),
                        "blur": round(image_stats.blur, 2),
                        "edge_density": round(image_stats.edge_density, 4),
                        "aspect_ratio": round(image_stats.aspect_ratio, 2),
                        "text_regions": image_stats.text_regions,
                        "table_likelihood": round(image_stats.table_likelihood, 2),
                        "form_likelihood": round(image_stats.form_likelihood, 2),
                        "text_confidence": round(image_stats.text_confidence, 2)
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"Processing failed: {str(e)}"}
    
    def _convert_pdf_to_image(self, pdf_path: str, page_num: int = 0) -> Tuple[Optional[str], int]:
        """
        Convert a PDF page to high-quality image for OCR
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-based)
            
        Returns:
            Tuple of (path to converted image, total pages)
        """
        if not PDF2IMAGE_AVAILABLE:
            logger.error("pdf2image library not available")
            return None, 0
            
        try:
            # Check total pages
            try:
                from PyPDF2 import PdfReader
                pdf = PdfReader(pdf_path)
                total_pages = len(pdf.pages)
            except Exception as e:
                logger.error(f"Error reading PDF: {e}")
                # Try with pdf2image directly if PyPDF2 fails
                try:
                    pages = convert_from_path(pdf_path, 72, first_page=1, last_page=1)
                    # Just to get a page count
                    temp_path = f"{pdf_path}_temp_count.jpg"
                    pages[0].save(temp_path, 'JPEG')
                    os.remove(temp_path)
                    
                    pages = convert_from_path(pdf_path, 72)
                    total_pages = len(pages)
                except Exception as e2:
                    logger.error(f"Error with pdf2image: {e2}")
                    total_pages = 1
            
            # Validate page number
            if page_num < 0 or page_num >= total_pages:
                logger.error(f"Invalid page number: {page_num}, total pages: {total_pages}")
                return None, total_pages
            
            # Convert with pdf2image using high DPI
            try:
                # Use higher DPI for better OCR quality
                pages = convert_from_path(
                    pdf_path,
                    600,  # High DPI for better quality
                    first_page=page_num + 1,
                    last_page=page_num + 1
                )
                
                if pages:
                    # Save as high-quality PNG (better than JPEG for text)
                    image_path = f"{pdf_path}_page_{page_num}.png"
                    pages[0].save(image_path, 'PNG')
                    return image_path, total_pages
            except Exception as e:
                logger.error(f"Error using pdf2image: {e}")
                
                # Try with lower DPI if higher fails (sometimes happens with memory issues)
                try:
                    pages = convert_from_path(
                        pdf_path,
                        300,  # Lower DPI
                        first_page=page_num + 1,
                        last_page=page_num + 1
                    )
                    
                    if pages:
                        image_path = f"{pdf_path}_page_{page_num}.png"
                        pages[0].save(image_path, 'PNG')
                        return image_path, total_pages
                except:
                    pass
            
            # Return error
            return None, total_pages
            
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            return None, 0
    
    def _save_debug_images(self, image_data, original_path):
        """
        Save processed images for debugging purposes
        
        Args:
            image_data: Dictionary of processed images
            original_path: Path to the original image
        """
        if not self.config["save_debug_images"]:
            return
        
        try:
            # Create a subdirectory for this image
            basename = os.path.basename(original_path)
            debug_dir = os.path.join(self.config["debug_output_dir"], 
                                     f"{basename}_{int(time.time())}")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save each processed image
            for name, img in image_data.items():
                out_path = os.path.join(debug_dir, f"{name}.jpg")
                cv2.imwrite(out_path, img)
            
            logger.info(f"Saved debug images to {debug_dir}")
        except Exception as e:
            logger.error(f"Error saving debug images: {e}")
    
    def _update_processing_stats(self, results: Dict, processing_time: float):
        """
        Update processing statistics for adaptive optimization
        
        Args:
            results: OCR results dictionary
            processing_time: Time taken to process the image
        """
        if not results or 'status' not in results or 'metadata' not in results:
            return
        
        # Get key information
        status = results['status']
        image_type = results['metadata'].get('image_type')
        engine = results['metadata'].get('best_engine', '').split('_')[0]
        
        if not image_type or not engine:
            return
        
        # Initialize dict for this image type if needed
        if image_type not in self.processing_stats['image_types']:
            self.processing_stats['image_types'][image_type] = {
                'count': 0,
                'success_count': 0,
                'processing_times': {},
                'success_rates': {}
            }
        
        # Update stats
        stats = self.processing_stats['image_types'][image_type]
        stats['count'] += 1
        
        # Update success count
        if status == 'success':
            stats['success_count'] += 1
        
        # Update processing time for this engine
        if engine not in stats['processing_times']:
            stats['processing_times'][engine] = []
        
        stats['processing_times'][engine].append(processing_time)
        
        # Limit list size to avoid memory issues
        if len(stats['processing_times'][engine]) > 10:
            stats['processing_times'][engine] = stats['processing_times'][engine][-10:]
        
        # Update success rates
        success_rate = stats['success_count'] / stats['count']
        
        if engine not in stats['success_rates']:
            stats['success_rates'][engine] = success_rate
        else:
            # Rolling average (70% old, 30% new)
            stats['success_rates'][engine] = stats['success_rates'][engine] * 0.7 + success_rate * 0.3
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics and performance metrics
        
        Returns:
            Dictionary with usage and performance statistics
        """
        return {
            "version": self.version,
            "engines_available": list(self.ocr_engines.keys()),
            "processing_stats": self.processing_stats,
            "cache_usage": {
                "size_bytes": self.memory_manager.current_usage,
                "item_count": len(self.memory_manager.cache)
            }
        }
    
    def clear_cache(self):
        """Clear the image cache to free memory"""
        self.memory_manager.clear_cache()
        logger.info("Cache cleared")

# Additional utility functions for standalone usage

    def process_file(self, file_path: str, original_filename: str = None, language: str = None, 
                    page: int = 0, summary_length: int = None, 
                    summary_style: str = None) -> dict:
        """
        Process a file (image or PDF) and extract text with summarization
        
        Args:
            file_path: Path to the file
            original_filename: Original filename (if different from file_path)
            language: OCR language (default from config)
            page: Page number for PDF (0-based)
            summary_length: Maximum summary length
            summary_style: Style of summary (concise, detailed, bullets, structured)
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        # Set original filename if not provided
        if original_filename is None:
            original_filename = os.path.basename(file_path)
        
        # Use default values if not provided
        language = language or self.config["default_language"]
        summary_length = summary_length or self.config["summary_length"]
        summary_style = summary_style or self.config["summary_style"]
        
        # Check file extension
        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in self.config["allowed_extensions"]:
            return {"status": "error", "message": "Unsupported file type"}
        
        try:
            # Handle PDF vs image
            is_pdf = ext == 'pdf'
            
            if is_pdf:
                if not PDF2IMAGE_AVAILABLE:
                    return {"status": "error", "message": "PDF processing not available"}
                
                # Convert PDF to image
                logger.info(f"Converting PDF to image: {file_path}, page {page}")
                image_path, total_pages = self._convert_pdf_to_image(file_path, page)
                
                if not image_path:
                    return {"status": "error", "message": "Failed to convert PDF to image"}
                    
                # Process the image
                image_results = self._process_image(image_path, language)
                
                # Ensure metadata exists
                if "metadata" not in image_results:
                    image_results["metadata"] = {}
                
                # Add PDF-specific metadata
                image_results["metadata"].update({
                    "file_type": "pdf",
                    "page": page,
                    "total_pages": total_pages
                })
                
                # Clean up temporary image
                try:
                    os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary PDF image: {e}")
            else:
                # Process the image directly
                logger.info(f"Processing image: {file_path}")
                image_results = self._process_image(file_path, language)
                
                # Ensure metadata exists
                if "metadata" not in image_results:
                    image_results["metadata"] = {}
                    
                image_results["metadata"]["file_type"] = "image"
            
            # Generate summary if text was extracted successfully
            if image_results["status"] in ["success", "partial_success"] and image_results.get("text"):
                from .text_processing import generate_summary, detect_document_structure, extract_key_insights
                
                text = image_results.get("text", "")
                
                # Use enhanced extractive summarization
                summary = generate_summary(text, max_length=summary_length, style=summary_style)
                image_results["summary"] = summary
                
                # Extract document structure
                structure = detect_document_structure(text)
                image_results["document_structure"] = structure.value
                
                # Extract key insights if enabled
                if self.config["extract_key_insights"] and len(text) > 200:
                    insights = extract_key_insights(text)
                    image_results["key_insights"] = insights
            else:
                image_results["summary"] = ""
            
            # Add processing time
            processing_time = time.time() - start_time
            
            # Ensure metadata exists
            if "metadata" not in image_results:
                image_results["metadata"] = {}
                
            image_results["metadata"]["processing_time_ms"] = round(processing_time * 1000, 2)
            
            # Update processing stats
            self._update_processing_stats(image_results, processing_time)
            
            # Apply organized output format if enabled
            if self.config["organized_output_format"]:
                from .information_extraction import organize_output
                image_results = organize_output(image_results)
            
            return image_results
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error", 
                "message": f"Processing failed: {str(e)}",
                "metadata": {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }

def process_directory(directory_path, output_dir=None, language=None, summary_length=200, summary_style="concise"):
    """Process all supported files in a directory"""
    # Create OCR engine
    ocr = SmartGlassOCR()
    
    # Get list of supported files
    supported_extensions = ocr.config["allowed_extensions"]
    files = []
    
    for ext in supported_extensions:
        import glob
        files.extend(glob.glob(os.path.join(directory_path, f"*.{ext}")))
    
    if not files:
        print(f"No supported files found in {directory_path}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each file
    results = {}
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        if output_dir:
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_ocr.txt")
        else:
            output_file = None
        
        file_result = process_file(
            file_path=file_path,
            output=output_file,
            language=language,
            summary_length=summary_length,
            summary_style=summary_style
        )
        
        results[file_path] = file_result
    
    print(f"Processed {len(files)} files")
    return results

if __name__ == "__main__":
    # Add required imports for standalone usage
    import glob
    import sys
    
    # Main function from command line
    if len(sys.argv) > 1:
        process_file(sys.argv[1])
    else:
        print("Usage: python smartglass_ocr.py [file_path]")