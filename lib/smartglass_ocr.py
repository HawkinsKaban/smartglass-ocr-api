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
from enum import Enum
from functools import lru_cache
from collections import Counter

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

# Image processing libraries
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

# Define image type classification with enhanced specificity
class ImageType(Enum):
    DOCUMENT = "document"           # Clear document with structured text
    NATURAL = "natural"             # Natural scene with text
    SIGNAGE = "signage"             # Signs, banners, displays
    HANDWRITTEN = "handwritten"     # Handwritten text
    MIXED = "mixed"                 # Mixed content
    LOW_QUALITY = "low_quality"     # Blurry or low quality image
    HIGH_CONTRAST = "high_contrast" # High contrast image (black text on white)
    RECEIPT = "receipt"             # Receipt or ticket
    ID_CARD = "id_card"             # ID card or license
    SCIENTIFIC = "scientific"       # Scientific document with formulas
    PRESENTATION = "presentation"   # Slides or presentation material
    BOOK_PAGE = "book_page"         # Book or magazine page
    NEWSPAPER = "newspaper"         # Newspaper or article
    FORM = "form"                   # Form with fields and entries
    TABLE = "table"                 # Table with rows and columns

@dataclass
class ImageStats:
    """Statistical features of an image"""
    width: int
    height: int
    brightness: float
    contrast: float
    blur: float
    edge_density: float
    text_regions: int  # Number of potential text regions
    aspect_ratio: float
    image_type: ImageType
    # Added new metrics for better image analysis
    table_likelihood: float = 0.0
    form_likelihood: float = 0.0
    color_variance: float = 0.0
    text_confidence: float = 0.0

# Enhanced processing strategies with more specific options
class ProcessingStrategy(Enum):
    """Strategies for image processing"""
    MINIMAL = "minimal"             # Basic processing
    STANDARD = "standard"           # Standard processing
    AGGRESSIVE = "aggressive"       # Heavy processing for difficult images
    DOCUMENT = "document"           # Optimized for documents
    NATURAL = "natural"             # Optimized for natural scenes
    RECEIPT = "receipt"             # Optimized for receipts/tickets
    ID_CARD = "id_card"             # Optimized for ID cards
    BOOK = "book"                   # Optimized for book pages
    TABLE = "table"                 # Optimized for tables and structured data
    HANDWRITTEN = "handwritten"     # Optimized for handwritten text
    MULTI_COLUMN = "multi_column"   # Optimized for multi-column layouts
    SCIENTIFIC = "scientific"       # Optimized for scientific documents
    FORM = "form"                   # Optimized for forms

# Define document structure type for better analysis
class DocumentStructure(Enum):
    """Types of document structures for improved text organization"""
    PLAIN_TEXT = "plain_text"       # Simple flowing text
    PARAGRAPHS = "paragraphs"       # Text with paragraph breaks
    HEADERS_AND_CONTENT = "headers_and_content"  # Headers with content sections
    BULLET_POINTS = "bullet_points" # Lists with bullet points
    TABLE = "table"                 # Tabular data
    FORM = "form"                   # Form with fields
    MULTI_COLUMN = "multi_column"   # Multi-column layout
    SCIENTIFIC = "scientific"       # Scientific with formulas
    MIXED = "mixed"                 # Mixed structure types

class MemoryManager:
    """Manages memory usage for image processing"""
    
    def __init__(self, max_cache_size_mb=500):
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.current_usage = 0
        self.cache = {}
        self.lock = threading.Lock()
    
    def add_to_cache(self, key, image):
        """Add an image to cache if there's enough space"""
        if not isinstance(image, np.ndarray):
            return False
        
        image_size = image.nbytes
        
        with self.lock:
            # Check if we need to make space
            if self.current_usage + image_size > self.max_cache_size:
                self._clean_cache(image_size)
            
            # Add to cache if there's space
            if self.current_usage + image_size <= self.max_cache_size:
                self.cache[key] = {
                    'image': image,
                    'size': image_size,
                    'last_access': time.time()
                }
                self.current_usage += image_size
                return True
            
            return False
    
    def get_from_cache(self, key):
        """Get an image from cache"""
        with self.lock:
            if key in self.cache:
                # Update last access time
                self.cache[key]['last_access'] = time.time()
                return self.cache[key]['image']
            return None
    
    def _clean_cache(self, required_space):
        """Clear enough space in the cache"""
        if not self.cache:
            return
        
        # Sort items by last access time
        items = sorted(self.cache.items(), key=lambda x: x[1]['last_access'])
        
        # Remove oldest items until we have enough space
        for key, item in items:
            self.current_usage -= item['size']
            del self.cache[key]
            
            if self.current_usage + required_space <= self.max_cache_size:
                break
    
    def clear_cache(self):
        """Clear the entire cache"""
        with self.lock:
            self.cache.clear()
            self.current_usage = 0

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
    
    def process_file(self, file_path: str, language: str = None, page: int = 0, 
                    summary_length: int = None, summary_style: str = None) -> dict:
        """
        Process a file (image or PDF) and extract text with summarization
        
        Args:
            file_path: Path to the file
            language: OCR language (default from config)
            page: Page number for PDF (0-based)
            summary_length: Maximum summary length
            summary_style: Style of summary (concise, detailed, bullets, structured)
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        # Use default values if not provided
        language = language or self.config["default_language"]
        summary_length = summary_length or self.config["summary_length"]
        summary_style = summary_style or self.config["summary_style"]
        
        # Check file extension
        ext = file_path.split('.')[-1].lower()
        if ext not in self.config["allowed_extensions"]:
            return {"status": "error", "message": "Unsupported file type"}
        
        try:
            # If lightweight mode is enabled, use simplified processing
            if self.config["lightweight_mode"]:
                logger.info("Using lightweight processing mode")
                
                # Simplified processing for lightweight mode
                if ext == 'pdf' and PDF2IMAGE_AVAILABLE:
                    # Simple PDF conversion
                    image_path, total_pages = self._convert_pdf_to_image(file_path, page, dpi=200)
                    
                    if not image_path:
                        return {"status": "error", "message": "Failed to convert PDF to image"}
                    
                    # Simple OCR with Tesseract only
                    if TESSERACT_AVAILABLE:
                        try:
                            # Convert to PIL image for Tesseract
                            pil_img = Image.open(image_path)
                            text = pytesseract.image_to_string(pil_img, lang=language)
                            
                            # Clean up temporary image
                            try:
                                os.remove(image_path)
                            except:
                                pass
                            
                            return {
                                "status": "success",
                                "text": text,
                                "metadata": {
                                    "file_type": "pdf",
                                    "page": page,
                                    "total_pages": total_pages,
                                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                                }
                            }
                        except Exception as e:
                            logger.error(f"Error in lightweight PDF OCR: {e}")
                            return {"status": "error", "message": f"OCR failed: {str(e)}"}
                    else:
                        return {"status": "error", "message": "Tesseract not available for OCR"}
                
                # Simple image processing
                elif CV2_AVAILABLE and TESSERACT_AVAILABLE:
                    try:
                        # Basic OCR with minimal processing
                        image = cv2.imread(file_path)
                        if image is None:
                            return {"status": "error", "message": "Failed to read image"}
                        
                        # Convert to grayscale
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        # Basic thresholding
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Convert to PIL image for Tesseract
                        pil_img = Image.fromarray(binary)
                        text = pytesseract.image_to_string(pil_img, lang=language)
                        
                        return {
                            "status": "success",
                            "text": text,
                            "metadata": {
                                "file_type": "image",
                                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                            }
                        }
                    except Exception as e:
                        logger.error(f"Error in lightweight image OCR: {e}")
                        return {"status": "error", "message": f"OCR failed: {str(e)}"}
                else:
                    return {"status": "error", "message": "Required libraries not available"}
            
            # Regular, full-featured processing mode:
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
                text = image_results.get("text", "")
                
                # Use enhanced extractive summarization
                summary = self._generate_enhanced_extractive_summary(text, max_length=summary_length, style=summary_style)
                image_results["summary"] = summary
                
                # Extract document structure
                structure = self._detect_document_structure(text)
                image_results["document_structure"] = structure.value
                
                # Extract key insights if enabled
                if self.config["extract_key_insights"] and len(text) > 200:
                    insights = self._extract_key_insights(text)
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
                image_results = self._organize_output(image_results)
            
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
        """
        Process an image through enhanced pipeline and OCR
        
        Args:
            image_path: Path to the image
            language: OCR language
            
        Returns:
            Dictionary with OCR results
        """
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
            image_stats = self._enhanced_image_analysis(image)
            logger.info(f"Enhanced image analysis: {image_stats.image_type.value}, " 
                       f"{image_stats.width}x{image_stats.height}, "
                       f"brightness: {image_stats.brightness:.1f}, "
                       f"contrast: {image_stats.contrast:.1f}, "
                       f"blur: {image_stats.blur:.1f}, "
                       f"table_likelihood: {image_stats.table_likelihood:.2f}")
            
            # Step 3: Determine the best processing strategy based on image type and stats
            strategy = self._determine_enhanced_processing_strategy(image_stats)
            logger.info(f"Using processing strategy: {strategy.value}")
            
            # If auto rotation is enabled, check and rotate if needed
            if self.config["auto_rotate"] and image_stats.image_type != ImageType.NATURAL:
                image = self._advanced_auto_rotate(image)
            
            # Step 4: Apply optimized preprocessing with enhanced methods
            processed_images, image_data = self._enhanced_preprocess_image(image, image_stats, strategy)
            
            # Save debug images if enabled
            if self.config["save_debug_images"]:
                self._save_debug_images(image_data, image_path)
            
            # Step 5: Perform OCR using the optimal engine sequence with improved confidence scoring
            best_engine, text, confidence, layout_info = self._perform_enhanced_ocr(
                processed_images, image_data, language, image_stats
            )
            
            # Step 6: Apply enhanced rule-based text correction if enabled
            if self.config["enable_text_correction"] and len(text) > 10:
                text = self._enhanced_post_process_text(text, image_stats.image_type)
            
            # Step 7: Clean and format the extracted text with improved formatting
            formatted_text = self._enhanced_text_formatting(text, layout_info)
            
            # Step 8: Extract additional information with rule-based methods
            detected_language = self._enhanced_language_detection(formatted_text)
            
            # Extract structured information if enabled
            structured_info = None
            if self.config["enable_structured_extraction"] and formatted_text:
                structured_info = self._extract_rule_based_structured_info(
                    formatted_text, image_stats.image_type
                )
            
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
    
    def _enhanced_image_analysis(self, image) -> ImageStats:
        """
        Perform enhanced image analysis for better type detection
        
        Args:
            image: OpenCV image
            
        Returns:
            ImageStats object with image characteristics
        """
        # Get dimensions
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Calculate color variance for determining if image is color or grayscale
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_variance = np.std(hsv[:,:,0]) + np.std(hsv[:,:,1])
        else:
            gray = image
            color_variance = 0.0
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate blur level (variance of Laplacian)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / (width * height)
        
        # Detect potential text regions with improved method
        text_regions = self._detect_enhanced_text_regions(gray)
        
        # Calculate text confidence based on edge analysis
        text_confidence = self._calculate_text_confidence(gray, edges)
        
        # Check for table structures - horizontal and vertical lines
        table_likelihood = self._detect_table_likelihood(gray, edges)
        
        # Check for form structures - boxed regions and labels
        form_likelihood = self._detect_form_likelihood(gray, text_regions)
        
        # Determine image type with enhanced algorithm
        image_type = self._determine_enhanced_image_type(
            width, height, brightness, contrast, blur, edge_density, 
            text_regions, aspect_ratio, color_variance, table_likelihood, 
            form_likelihood, text_confidence
        )
        
        return ImageStats(
            width=width,
            height=height,
            brightness=brightness,
            contrast=contrast,
            blur=blur,
            edge_density=edge_density,
            text_regions=len(text_regions),
            aspect_ratio=aspect_ratio,
            image_type=image_type,
            table_likelihood=table_likelihood,
            form_likelihood=form_likelihood,
            color_variance=color_variance,
            text_confidence=text_confidence
        )
    
    def _detect_enhanced_text_regions(self, gray_image) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential text regions with enhanced methodology
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of (x, y, w, h) tuples for potential text regions
        """
        # Apply MSER (Maximally Stable Extremal Regions) for text detection
        # This is more accurate than simple binary thresholding for text regions
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)
            
            text_regions = []
            for region in regions:
                # Get bounding box for each region
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter regions by size and aspect ratio to find likely text areas
                area = w * h
                if (area > 100 and area < 10000 and 0.1 < w/h < 10 and
                    h > 8 and w > 8):  # Minimum size for text
                    text_regions.append((x, y, w, h))
            
            # Merge overlapping regions to get text lines
            if text_regions:
                text_regions = self._merge_overlapping_regions(text_regions)
                
            return text_regions
            
        except Exception as e:
            logger.warning(f"MSER detection failed: {e}, falling back to threshold-based detection")
            
            # Fallback to threshold-based detection
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Perform morphological operations to separate text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to identify potential text regions
            text_regions = []
            min_area = 100  # Minimum area to consider
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter based on size and aspect ratio
                if area > min_area and 0.1 < w/h < 15:
                    text_regions.append((x, y, w, h))
            
            return text_regions
    
    def _merge_overlapping_regions(self, regions):
        """
        Merge overlapping text regions to get text lines/paragraphs
        
        Args:
            regions: List of (x, y, w, h) tuples
            
        Returns:
            List of merged (x, y, w, h) tuples
        """
        if not regions:
            return []
        
        # Sort regions by y-coordinate
        sorted_regions = sorted(regions, key=lambda r: r[1])
        
        merged_regions = []
        current_group = [sorted_regions[0]]
        current_y = sorted_regions[0][1]
        
        for i in range(1, len(sorted_regions)):
            region = sorted_regions[i]
            # If region is close to the current line (vertically)
            if abs(region[1] - current_y) < region[3] * 0.5:
                current_group.append(region)
            else:
                # Process the current group - merge horizontally close regions
                merged_line = self._merge_line_regions(current_group)
                merged_regions.extend(merged_line)
                
                # Start a new group
                current_group = [region]
                current_y = region[1]
        
        # Process the last group
        if current_group:
            merged_line = self._merge_line_regions(current_group)
            merged_regions.extend(merged_line)
        
        return merged_regions
    
    def _merge_line_regions(self, line_regions):
        """
        Merge horizontally close regions in a line
        
        Args:
            line_regions: List of regions in approximately the same line
            
        Returns:
            List of merged regions
        """
        if not line_regions:
            return []
        
        # Sort regions horizontally
        sorted_line = sorted(line_regions, key=lambda r: r[0])
        
        merged_line = []
        current_region = list(sorted_line[0])
        
        for i in range(1, len(sorted_line)):
            region = sorted_line[i]
            # If regions are horizontally close
            if region[0] <= current_region[0] + current_region[2] + 10:
                # Merge regions
                x = min(current_region[0], region[0])
                y = min(current_region[1], region[1])
                w = max(current_region[0] + current_region[2], region[0] + region[2]) - x
                h = max(current_region[1] + current_region[3], region[1] + region[3]) - y
                current_region = [x, y, w, h]
            else:
                merged_line.append(tuple(current_region))
                current_region = list(region)
        
        # Add the last region
        merged_line.append(tuple(current_region))
        
        return merged_line
    
    def _calculate_text_confidence(self, gray_image, edges) -> float:
        """
        Calculate confidence that the image contains text based on edge patterns
        
        Args:
            gray_image: Grayscale image
            edges: Edge-detected image
            
        Returns:
            Confidence score (0-100)
        """
        # Simple version - calculate edge patterns typical for text
        height, width = gray_image.shape
        
        # Text has a specific ratio of horizontal to vertical edges
        kernel_h = np.ones((1, 5), np.uint8)
        kernel_v = np.ones((5, 1), np.uint8)
        
        # Erode the edges to find horizontal and vertical components
        horizontal = cv2.erode(edges, kernel_h, iterations=1)
        vertical = cv2.erode(edges, kernel_v, iterations=1)
        
        # Count pixels
        h_pixels = np.count_nonzero(horizontal)
        v_pixels = np.count_nonzero(vertical)
        total_edge_pixels = np.count_nonzero(edges)
        
        if total_edge_pixels == 0:
            return 0.0
        
        # Text typically has a balanced ratio of horizontal to vertical edges
        # with more horizontal than vertical in Latin-based scripts
        h_v_ratio = h_pixels / (v_pixels + 1)  # Add 1 to avoid division by zero
        
        # Good text range is around 1.2 to 2.5 for h/v ratio
        ratio_score = 0.0
        if 1.0 <= h_v_ratio <= 3.0:
            # Optimal range
            ratio_score = 100.0
        elif 0.5 <= h_v_ratio < 1.0 or 3.0 < h_v_ratio <= 5.0:
            # Less optimal but still possibly text
            ratio_score = 60.0
        else:
            # Unlikely to be text
            ratio_score = 30.0
        
        # Edge density - text typically has a specific range of edge density
        edge_density = total_edge_pixels / (width * height)
        
        density_score = 0.0
        if 0.02 <= edge_density <= 0.15:
            # Optimal text density
            density_score = 100.0
        elif 0.01 <= edge_density < 0.02 or 0.15 < edge_density <= 0.25:
            # Less optimal but possible
            density_score = 60.0
        else:
            # Either too sparse or too dense for typical text
            density_score = 30.0
        
        # Combine scores - give more weight to ratio which is more text-specific
        # Weighted average
        confidence = (ratio_score * 0.6) + (density_score * 0.4)
        
        return min(100.0, confidence)
    
    def _detect_table_likelihood(self, gray_image, edges) -> float:
        """
        Detect the likelihood that the image contains tables
        
        Args:
            gray_image: Grayscale image
            edges: Edge-detected image
            
        Returns:
            Likelihood score (0-100)
        """
        # Detect horizontal and vertical lines which are typical for tables
        height, width = gray_image.shape
        
        # Define kernels for horizontal and vertical lines
        kernel_h = np.ones((1, int(width/30)), np.uint8)  # Horizontal kernel
        kernel_v = np.ones((int(height/30), 1), np.uint8)  # Vertical kernel
        
        # Use morphology to find lines
        # Assuming white text on black background after edge detection
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.bitwise_or(horizontal, vertical)
        
        # Count pixels to find the extent of potential table structure
        h_pixels = np.count_nonzero(horizontal)
        v_pixels = np.count_nonzero(vertical)
        
        # Calculate the density of lines
        h_density = h_pixels / (width * height)
        v_density = v_pixels / (width * height)
        
        # A good table should have a balanced distribution of horizontal and vertical lines
        # and the density should be within a certain range
        
        # Check the balance between horizontal and vertical lines
        if h_pixels == 0 or v_pixels == 0:
            balance_score = 0.0  # No lines in one direction means no table
        else:
            # Calculate how balanced the lines are (closer to 1.0 is more balanced)
            balance_ratio = h_pixels / v_pixels if h_pixels <= v_pixels else v_pixels / h_pixels
            balance_score = balance_ratio * 100.0
        
        # Check the density
        density_score = 0.0
        combined_density = h_density + v_density
        
        if 0.002 <= combined_density <= 0.05:
            # Optimal range for tables
            density_score = 100.0
        elif combined_density < 0.002:
            # Too few lines
            density_score = max(0, combined_density * 50000)  # Scale up to 100
        elif combined_density > 0.05:
            # Too many lines, might be a dense document or noise
            density_score = max(0, 100 - (combined_density - 0.05) * 2000)
        
        # Check for intersections of horizontal and vertical lines
        # Tables typically have intersections at cell corners
        intersections = cv2.bitwise_and(horizontal, vertical)
        intersection_count = np.count_nonzero(intersections)
        
        # More intersections usually means more likelihood of a table
        intersection_score = min(100.0, intersection_count / 5.0)
        
        # Combine the scores with appropriate weights
        likelihood = (
            (balance_score * 0.3) + 
            (density_score * 0.4) + 
            (intersection_score * 0.3)
        )
        
        return min(100.0, likelihood)
    
    def _detect_form_likelihood(self, gray_image, text_regions) -> float:
        """
        Detect the likelihood that the image contains a form
        
        Args:
            gray_image: Grayscale image
            text_regions: Detected text regions
            
        Returns:
            Likelihood score (0-100)
        """
        height, width = gray_image.shape
        
        # Forms typically have:
        # 1. Text regions aligned in a structured way
        # 2. Boxes or lines for input fields
        # 3. Labels followed by blank spaces or underlines
        
        # Use binary thresholding to detect boxes and lines
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines (typically used in forms for input fields)
        kernel_h = np.ones((1, 20), np.uint8)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        
        # Detect rectangular boxes (common in forms)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular and square contours
        rect_count = 0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            if peri > 100:  # Ignore very small contours
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:  # Rectangle has 4 corners
                    rect_count += 1
        
        # Calculate horizontal line density
        h_pixels = np.count_nonzero(horizontal)
        h_density = h_pixels / (width * height)
        
        # Forms usually have a certain density of horizontal lines
        line_score = 0.0
        if 0.001 <= h_density <= 0.02:
            line_score = 100.0 * (h_density / 0.02)
        elif h_density > 0.02:
            line_score = 100.0 - min(100.0, (h_density - 0.02) * 5000)
        
        # Analyze text region alignment
        alignment_score = 0.0
        if len(text_regions) > 5:
            # Collect x-coordinates for potential label alignment
            x_coords = [region[0] for region in text_regions]
            
            # Check for consistent alignment (common in forms where labels align)
            from collections import Counter
            coord_counter = Counter([coord // 10 * 10 for coord in x_coords])  # Group within 10px
            
            # Find the most common x-coordinate (potential label column)
            if coord_counter:
                most_common_count = coord_counter.most_common(1)[0][1]
                alignment_ratio = most_common_count / len(text_regions)
                
                # Higher ratio means more aligned text regions
                alignment_score = alignment_ratio * 100.0
        
        # Check for box presence (typical in forms)
        box_score = min(100.0, rect_count * 10.0)
        
        # Combine scores - weighing alignment and boxes more as they're more form-specific
        likelihood = (
            (line_score * 0.3) + 
            (alignment_score * 0.4) + 
            (box_score * 0.3)
        )
        
        return min(100.0, likelihood)
    
    def _determine_enhanced_image_type(self, width, height, brightness, 
                                     contrast, blur, edge_density, 
                                     text_regions, aspect_ratio, color_variance,
                                     table_likelihood, form_likelihood,
                                     text_confidence) -> ImageType:
        """
        Determine the type of image with enhanced classification
        
        Args:
            Various image characteristics
            
        Returns:
            ImageType enum representing the image type
        """
        # Define weighted characteristics for image types
        # Each characteristic has a weight and a range of values for each image type
        
        # Calculate scores for each image type
        scores = {}
        
        # Check if the image is blurry or low quality
        if blur < 100:
            scores[ImageType.LOW_QUALITY] = 100
        else:
            scores[ImageType.LOW_QUALITY] = max(0, 100 - blur/10)
        
        # Check for ID card (specific aspect ratio, multiple text regions)
        id_card_score = 0
        if 1.4 < aspect_ratio < 1.8 and 4 <= len(text_regions) <= 15:
            id_card_score = 80
            # Additional check for ID-like layout: header + multiple fields
            if form_likelihood > 50:
                id_card_score += 20
        scores[ImageType.ID_CARD] = id_card_score
        
        # Check for receipt (tall and narrow)
        receipt_score = 0
        if aspect_ratio < 0.6 and len(text_regions) > 5:
            receipt_score = 70
            # Additional check for typical receipt layout with aligned prices
            if form_likelihood > 30 and text_confidence > 60:
                receipt_score += 30
        scores[ImageType.RECEIPT] = receipt_score
        
        # Check for document
        document_score = 0
        if edge_density > 0.04 and contrast > 40 and blur > 300:
            document_score = 60
            if text_confidence > 70:
                document_score += 20
            # Check for paragraph structure
            if len(text_regions) > 10:
                document_score += 20
        scores[ImageType.DOCUMENT] = document_score
        
        # Check for high contrast document
        high_contrast_score = 0
        if contrast > 70 and brightness > 180 and edge_density > 0.04:
            high_contrast_score = 80
            if text_confidence > 80:
                high_contrast_score += 20
        scores[ImageType.HIGH_CONTRAST] = high_contrast_score
        
        # Check for signage
        signage_score = 0
        if (width > 2*height or height > 2*width) and edge_density < 0.1 and contrast > 50:
            signage_score = 70
            # Signs typically have large text with high contrast
            if text_confidence > 70 and len(text_regions) < 5 and color_variance > 20:
                signage_score += 30
        scores[ImageType.SIGNAGE] = signage_score
        
        # Check for handwritten text
        handwritten_score = 0
        if 0.02 < edge_density < 0.06 and 20 < contrast < 60:
            handwritten_score = 60
            # Handwritten text typically has more irregular edges
            if blur < 300 and text_confidence < 70:
                handwritten_score += 40
        scores[ImageType.HANDWRITTEN] = handwritten_score
        
        # Check for natural scene
        natural_score = 0
        if edge_density < 0.04 and contrast > 30:
            natural_score = 60
            # Natural scenes typically have higher color variance
            if color_variance > 30 and text_confidence < 60:
                natural_score += 40
        scores[ImageType.NATURAL] = natural_score
        
        # Check for form
        form_score = form_likelihood
        scores[ImageType.FORM] = form_score
        
        # Check for book page
        book_page_score = 0
        if 0.65 < aspect_ratio < 0.85 and edge_density > 0.05 and text_confidence > 70:
            book_page_score = 80
            # Book pages typically have dense, aligned text
            if len(text_regions) > 15:
                book_page_score += 20
        scores[ImageType.BOOK_PAGE] = book_page_score
        
        # Check for scientific document
        scientific_score = 0
        if edge_density > 0.05 and contrast > 50:
            # Scientific documents often have formulas, diagrams, and tables
            if table_likelihood > 40:
                scientific_score += 40
            # Look for potential formula patterns
            if text_confidence > 60 and len(text_regions) > 5:
                scientific_score += 30
        scores[ImageType.SCIENTIFIC] = scientific_score
        
        # Check for presentation
        presentation_score = 0
        if brightness > 200 and contrast > 60:
            presentation_score = 40
            # Presentations often have large text with high contrast
            if len(text_regions) < 10 and text_confidence > 70:
                presentation_score += 30
            # Presentations often have a distinct aspect ratio
            if 1.2 < aspect_ratio < 1.8:
                presentation_score += 30
        scores[ImageType.PRESENTATION] = presentation_score
        
        # Check for newspaper
        newspaper_score = 0
        if edge_density > 0.06 and contrast > 50:
            newspaper_score = 50
            # Newspapers often have multiple columns
            if len(text_regions) > 20:
                newspaper_score += 30
            # Check for multi-column layout
            text_x_positions = [region[0] for region in text_regions]
            if text_x_positions:
                x_positions_set = set([x // 50 for x in text_x_positions])  # Group by 50px
                if len(x_positions_set) >= 3:  # Multiple columns
                    newspaper_score += 20
        scores[ImageType.NEWSPAPER] = newspaper_score
        
        # Check for table structure
        table_score = table_likelihood
        scores[ImageType.TABLE] = table_score
        
        # Get the highest scoring image type
        best_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # If two types have very close scores, favor the more specific type
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0][1] - sorted_scores[1][1] < 10:
            # If scores are close, choose more specific type
            specific_types = [
                ImageType.ID_CARD, ImageType.RECEIPT, ImageType.SCIENTIFIC, 
                ImageType.FORM, ImageType.BOOK_PAGE, ImageType.NEWSPAPER,
                ImageType.TABLE
            ]
            
            if sorted_scores[1][0] in specific_types and sorted_scores[0][0] not in specific_types:
                best_type = sorted_scores[1][0]
        
        # Special case: if no type has a strong score, default to MIXED
        if scores[best_type] < 50:
            return ImageType.MIXED
        
        # Special case: if table score is very high, override other types
        if scores[ImageType.TABLE] > 70:
            return ImageType.TABLE
        
        return best_type
    
    def _determine_enhanced_processing_strategy(self, image_stats: ImageStats) -> ProcessingStrategy:
        """
        Determine the optimal processing strategy with enhanced logic
        
        Args:
            image_stats: ImageStats object with image characteristics
            
        Returns:
            ProcessingStrategy enum value
        """
        # Check configured preprocessing level
        if self.config["preprocessing_level"] != "auto":
            return ProcessingStrategy(self.config["preprocessing_level"])
        
        # Otherwise, determine based on image type and characteristics
        image_type = image_stats.image_type
        
        # Direct mappings for specific image types
        type_to_strategy = {
            ImageType.DOCUMENT: ProcessingStrategy.DOCUMENT,
            ImageType.HIGH_CONTRAST: ProcessingStrategy.MINIMAL,
            ImageType.RECEIPT: ProcessingStrategy.RECEIPT,
            ImageType.ID_CARD: ProcessingStrategy.ID_CARD,
            ImageType.NATURAL: ProcessingStrategy.NATURAL,
            ImageType.HANDWRITTEN: ProcessingStrategy.HANDWRITTEN,
            ImageType.BOOK_PAGE: ProcessingStrategy.BOOK,
            ImageType.SCIENTIFIC: ProcessingStrategy.SCIENTIFIC,
            ImageType.FORM: ProcessingStrategy.FORM,
            ImageType.NEWSPAPER: ProcessingStrategy.MULTI_COLUMN,
            ImageType.TABLE: ProcessingStrategy.TABLE
        }
        
        if image_type in type_to_strategy:
            return type_to_strategy[image_type]
        
        # For image types not in the mapping, use characteristics to determine
        if image_type == ImageType.LOW_QUALITY:
            return ProcessingStrategy.AGGRESSIVE
        
        if image_type == ImageType.MIXED:
            # For mixed content, decide based on specific characteristics
            if image_stats.blur < 300:
                return ProcessingStrategy.AGGRESSIVE
            elif image_stats.text_confidence > 70:
                return ProcessingStrategy.DOCUMENT
            else:
                return ProcessingStrategy.STANDARD
        
        # Default to standard processing
        return ProcessingStrategy.STANDARD
    
    def _advanced_auto_rotate(self, image) -> np.ndarray:
        """
        Advanced auto-rotation detection and correction
        
        Args:
            image: OpenCV image
            
        Returns:
            Rotated image if needed, otherwise original image
        """
        # Convert to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try multiple methods for rotation detection and use the most confident one
        
        # Method 1: Hough Line Transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        angles_hough = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = np.arctan((y2 - y1) / (x2 - x1)) * 180.0 / np.pi
                    angles_hough.append(angle)
        
        # Method 2: Text Region Analysis
        # Detect text regions and analyze their alignment
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal kernel
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles_regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                # OpenCV's minAreaRect returns angles in [-90, 0)
                # Normalize to [-45, 45)
                if angle < -45:
                    angle = 90 + angle
                angles_regions.append(angle)
        
        # Determine the most reliable rotation angle
        rotation_angle = 0
        confidence = 0
        
        # Process Hough angles if we have enough
        if len(angles_hough) > 5:
            # Filter extreme angles
            filtered_angles = [a for a in angles_hough if -45 < a < 45]
            if filtered_angles:
                # Use median for robustness against outliers
                median_angle = np.median(filtered_angles)
                # Calculate confidence based on consistency
                angle_std = np.std(filtered_angles)
                hough_confidence = max(0, 100 - angle_std * 10)  # Lower std = higher confidence
                
                if hough_confidence > confidence:
                    rotation_angle = median_angle
                    confidence = hough_confidence
        
        # Process region angles if we have enough
        if len(angles_regions) > 3:
            median_angle = np.median(angles_regions)
            # Calculate confidence based on consistency
            angle_std = np.std(angles_regions)
            region_confidence = max(0, 100 - angle_std * 10)
            
            if region_confidence > confidence:
                rotation_angle = -median_angle  # Negate since we want to correct the rotation
                confidence = region_confidence
        
        # Only apply rotation if we're confident enough and angle is significant
        if confidence > 30 and abs(rotation_angle) > 1.0:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Calculate rotation matrix
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Apply rotation with border replication for better quality
            rotated = cv2.warpAffine(image, M, (width, height), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Auto-rotated image by {rotation_angle:.2f} degrees with {confidence:.1f}% confidence")
            return rotated
        
        # Return original if no rotation needed or not confident enough
        return image
    
    def _enhanced_preprocess_image(self, image, image_stats: ImageStats, 
                                 strategy: ProcessingStrategy) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Preprocess the image with enhanced techniques based on image type
        
        Args:
            image: OpenCV image
            image_stats: ImageStats object with image characteristics
            strategy: Processing strategy to apply
            
        Returns:
            Tuple of (list of processing methods, dict mapping methods to processed images)
        """
        # Dictionary to store processed images in memory
        processed_images = []
        image_data = {}
        
        # Convert to grayscale if not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Store grayscale version
        image_data["gray"] = gray
        processed_images.append("gray")
        
        # Apply resizing if needed
        resized_image = self._resize_for_optimal_ocr(gray, image_stats)
        if resized_image is not gray:
            image_data["resized"] = resized_image
            processed_images.append("resized")
            base_image = resized_image
        else:
            base_image = gray
        
        # Apply strategy-specific preprocessing
        if strategy == ProcessingStrategy.MINIMAL:
            # Just apply basic Otsu thresholding
            _, otsu = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
        
        elif strategy == ProcessingStrategy.DOCUMENT:
            # Optimized for document images
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Apply Otsu thresholding
            _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
            _, denoised_otsu = cv2.threshold(denoised, 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["denoised_otsu"] = denoised_otsu
            processed_images.append("denoised_otsu")
        
        elif strategy == ProcessingStrategy.RECEIPT:
            # Optimized for receipts
            # High contrast adjustment
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Stronger adaptive thresholding
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 4)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Special receipt processing - line removal for better text recognition
            kernel_h = np.ones((1, 20), np.uint8)
            eroded_h = cv2.erode(adaptive, kernel_h, iterations=1)
            dilated_h = cv2.dilate(eroded_h, kernel_h, iterations=1)
            removed_lines = cv2.subtract(adaptive, dilated_h)
            image_data["removed_lines"] = removed_lines
            processed_images.append("removed_lines")
            
            # Deskew specifically for receipts which are often slightly tilted
            try:
                coords = np.column_stack(np.where(removed_lines > 0))
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                if abs(angle) > 0.5:  # Only deskew if there's a significant angle
                    (h, w) = removed_lines.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    deskewed = cv2.warpAffine(removed_lines, M, (w, h), 
                                          flags=cv2.INTER_CUBIC, 
                                          borderMode=cv2.BORDER_REPLICATE)
                    image_data["deskewed"] = deskewed
                    processed_images.append("deskewed")
            except:
                pass  # Skip deskewing if it fails
            
            # Add sharpened version for better text recognition
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
            _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["sharpened_thresh"] = sharp_thresh
            processed_images.append("sharpened_thresh")
        
        elif strategy == ProcessingStrategy.ID_CARD:
            # Optimized for ID cards
            # Noise reduction for small text
            bilateral = cv2.bilateralFilter(base_image, 9, 75, 75)
            image_data["bilateral"] = bilateral
            processed_images.append("bilateral")
            
            # Enhance contrast to make small text more readable
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(bilateral)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Apply Otsu thresholding
            _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
            
            # Apply adaptive thresholding - often better for variable lighting in ID cards
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Edge enhancement for better text detection in security features
            edges = cv2.Canny(contrast_enhanced, 50, 150)
            dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
            edge_enhanced = cv2.addWeighted(contrast_enhanced, 0.8, dilated_edges, 0.2, 0)
            image_data["edge_enhanced"] = edge_enhanced
            processed_images.append("edge_enhanced")
            
            # Add local adaptive thresholding with smaller window for fine details
            adaptive_small = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 7, 2)
            image_data["adaptive_small"] = adaptive_small
            processed_images.append("adaptive_small")
        
        elif strategy == ProcessingStrategy.NATURAL:
            # Optimized for natural scenes
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Apply adaptive thresholding with larger block sizes for natural scenes
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 25, 15)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Apply bilateral filter to preserve edges while removing noise
            bilateral = cv2.bilateralFilter(contrast_enhanced, 11, 17, 17)
            image_data["bilateral"] = bilateral
            processed_images.append("bilateral")
            
            # Edge emphasizing for better text detection in natural scenes
            edges = cv2.Canny(contrast_enhanced, 30, 120)
            dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
            edge_enhanced = cv2.addWeighted(contrast_enhanced, 0.7, dilated_edges, 0.3, 0)
            image_data["edge_enhanced"] = edge_enhanced
            processed_images.append("edge_enhanced")
            
            # Shadow removal technique for outdoor scenes
            _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
            image_data["shadow_removed"] = closed
            processed_images.append("shadow_removed")
            
            # Additional processing for better text extraction
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
            image_data["sharpened"] = sharpened
            processed_images.append("sharpened")
            
            # Otsu on sharpened image
            _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["sharp_thresh"] = sharp_thresh
            processed_images.append("sharp_thresh")
        
        elif strategy == ProcessingStrategy.HANDWRITTEN:
            # Optimized for handwritten text
            # Apply stronger bilateral filtering to smooth but preserve edges
            bilateral = cv2.bilateralFilter(base_image, 15, 40, 40)
            image_data["bilateral"] = bilateral
            processed_images.append("bilateral")
            
            # Increase contrast to make pen/pencil marks stand out
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(bilateral)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Modified adaptive thresholding for handwritten text
            # Use larger block size and higher constant for better noise filtering
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 21, 10)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Thin handwritten strokes to help with recognition
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(adaptive, kernel, iterations=1)
            image_data["thinned"] = eroded
            processed_images.append("thinned")
            
            # Edge enhancement for better stroke detection
            edges = cv2.Canny(contrast_enhanced, 30, 120)
            edge_enhanced = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
            edge_enhanced = 255 - edge_enhanced  # Invert for better visualization
            image_data["edge_enhanced"] = edge_enhanced
            processed_images.append("edge_enhanced")
            
            # Apply special morphological operations for handwritten text
            kernel_line = np.ones((1, 5), np.uint8)
            dilated_horiz = cv2.dilate(contrast_enhanced, kernel_line, iterations=1)
            dilated_horiz = cv2.erode(dilated_horiz, kernel_line, iterations=1)
            image_data["enhanced_strokes"] = dilated_horiz
            processed_images.append("enhanced_strokes")
        
        elif strategy == ProcessingStrategy.BOOK:
            # Optimized for book pages
            # Apply contrast enhancement for faded text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Apply denoising for typical book scan noise
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
            image_data["denoised"] = denoised
            processed_images.append("denoised")
            
            # Apply Otsu thresholding which works well for book pages
            _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
            
            # Remove page curvature shadows
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            background = cv2.morphologyEx(denoised, cv2.MORPH_DILATE, morph_kernel)
            normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
            image_data["normalized"] = normalized
            processed_images.append("normalized")
            
            # Enhance text edges for better recognition
            edges = cv2.Canny(denoised, 50, 150)
            dilated_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            edge_enhanced = cv2.addWeighted(denoised, 0.8, dilated_edges, 0.2, 0)
            image_data["edge_enhanced"] = edge_enhanced
            processed_images.append("edge_enhanced")
            
            # Sharpening for clearer text
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["sharpened_thresh"] = sharp_thresh
            processed_images.append("sharpened_thresh")
        
        elif strategy == ProcessingStrategy.TABLE:
            # Optimized for tables
            # Use line detection and enhancement
            # Detect horizontal and vertical lines
            kernel_h = np.ones((1, 40), np.uint8)
            kernel_v = np.ones((40, 1), np.uint8)
            
            _, binary = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
            
            # Combine lines and dilate to get table structure
            table_structure = cv2.bitwise_or(horizontal, vertical)
            table_structure = cv2.dilate(table_structure, np.ones((3,3), np.uint8), iterations=1)
            
            # Invert for display and OCR
            table_structure = 255 - table_structure
            image_data["table_structure"] = table_structure
            processed_images.append("table_structure")
            
            # Get cells from table by removing grid lines
            cells = cv2.bitwise_and(255 - binary, table_structure)
            image_data["table_cells"] = cells
            processed_images.append("table_cells")
            
            # Apply adaptive thresholding for text in cells
            adaptive = cv2.adaptiveThreshold(base_image, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Apply enhanced processing for table text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
            
            # Special processing to isolate text in table cells
            dilated_lines = cv2.dilate(horizontal + vertical, np.ones((2,2), np.uint8), iterations=1)
            text_only = cv2.bitwise_and(otsu, cv2.bitwise_not(dilated_lines))
            image_data["text_only"] = text_only
            processed_images.append("text_only")
        
        elif strategy == ProcessingStrategy.MULTI_COLUMN:
            # Optimized for multi-column layouts like newspapers
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
            image_data["denoised"] = denoised
            processed_images.append("denoised")
            
            # Apply Otsu thresholding
            _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
            
            # Create an enhanced version for column detection
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(otsu, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=2)
            image_data["column_segmented"] = eroded
            processed_images.append("column_segmented")
            
            # Perform vertical projection to detect column boundaries
            vertical_projection = np.sum(otsu, axis=0) / 255
            
            # Smooth the projection to reduce noise
            kernel_size = max(3, int(width / 100))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            vertical_projection_smoothed = cv2.GaussianBlur(vertical_projection.reshape(-1, 1), 
                                                          (1, kernel_size), 0).flatten()
            
            # Create visual representation of column detection
            column_visual = otsu.copy()
            threshold = np.mean(vertical_projection_smoothed) * 0.5
            
            # Mark detected column boundaries
            for i in range(1, len(vertical_projection_smoothed) - 1):
                if (vertical_projection_smoothed[i] < threshold and 
                    vertical_projection_smoothed[i-1] > threshold):
                    cv2.line(column_visual, (i, 0), (i, height), 127, 2)
            
            image_data["column_detected"] = column_visual
            processed_images.append("column_detected")
            
            # Enhanced processing for multi-column text
            adaptive = cv2.adaptiveThreshold(denoised, 255, 
                                           cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, 15, 8)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
        
        elif strategy == ProcessingStrategy.SCIENTIFIC:
            # Optimized for scientific documents with formulas
            # Enhance contrast first
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Apply adaptive thresholding with careful parameters to preserve formula symbols
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 15, 8)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Special morphological operation to preserve small details in formulas
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            image_data["formula_preserved"] = opened
            processed_images.append("formula_preserved")
            
            # Enhanced edge detection for formula symbols
            edges = cv2.Canny(contrast_enhanced, 30, 130)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            symbol_enhanced = cv2.addWeighted(contrast_enhanced, 0.85, dilated_edges, 0.15, 0)
            image_data["symbol_enhanced"] = symbol_enhanced
            processed_images.append("symbol_enhanced")
            
            # Special processing for mathematical symbols
            # Sharpen the image to enhance small details
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
            _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["sharp_thresh"] = sharp_thresh
            processed_images.append("sharp_thresh")
            
            # Special processing for subscripts and superscripts
            # Use a larger kernel for text and a smaller kernel for symbols
            text_kernel = np.ones((3, 3), np.uint8)
            text_cleaned = cv2.morphologyEx(sharp_thresh, cv2.MORPH_OPEN, text_kernel)
            symbol_kernel = np.ones((1, 1), np.uint8)
            symbol_only = cv2.subtract(sharp_thresh, text_cleaned)
            symbol_cleaned = cv2.morphologyEx(symbol_only, cv2.MORPH_OPEN, symbol_kernel)
            combined = cv2.bitwise_or(text_cleaned, symbol_cleaned)
            image_data["formula_enhanced"] = combined
            processed_images.append("formula_enhanced")
        
        elif strategy == ProcessingStrategy.FORM:
            # Optimized for forms
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 15, 5)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Find and enhance form fields
            kernel = np.ones((1, 20), np.uint8)
            horizontal = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((20, 1), np.uint8)
            vertical = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
            
            # Combine to get form structure
            form_structure = cv2.bitwise_or(horizontal, vertical)
            
            # Enhance form field boundaries
            form_enhanced = cv2.bitwise_and(adaptive, cv2.bitwise_not(form_structure))
            image_data["form_enhanced"] = form_enhanced
            processed_images.append("form_enhanced")
            
            # Special processing for form text
            # Create a mask of likely text areas
            _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel_text = np.ones((3, 15), np.uint8)
            text_areas = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_text)
            text_only = cv2.bitwise_and(binary, text_areas)
            image_data["text_only"] = text_only
            processed_images.append("text_only")
            
            # Extract field labels with special processing
            # Labels are usually aligned and have consistent format
            labels_image = cv2.bitwise_xor(binary, form_enhanced)
            kernel_label = np.ones((1, 15), np.uint8)
            labels_processed = cv2.morphologyEx(labels_image, cv2.MORPH_CLOSE, kernel_label)
            image_data["labels"] = labels_processed
            processed_images.append("labels")
        
        elif strategy == ProcessingStrategy.STANDARD:
            # Standard processing for general cases
            # Apply Otsu thresholding
            _, otsu = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
            
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(base_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            _, contrast_otsu = cv2.threshold(contrast_enhanced, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["contrast_otsu"] = contrast_otsu
            processed_images.append("contrast_otsu")
            
            # Add edge enhancement for better text detection
            edges = cv2.Canny(base_image, 50, 150)
            dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
            edge_enhanced = cv2.addWeighted(base_image, 0.8, dilated_edges, 0.2, 0)
            image_data["edge_enhanced"] = edge_enhanced
            processed_images.append("edge_enhanced")
        
        elif strategy == ProcessingStrategy.AGGRESSIVE:
            # Aggressive processing for difficult images
            # Super contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(base_image)
            image_data["contrast"] = contrast_enhanced
            processed_images.append("contrast")
            
            # Strong denoising
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 15, 9, 21)
            image_data["denoised"] = denoised
            processed_images.append("denoised")
            
            # Sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            image_data["sharpened"] = sharpened
            processed_images.append("sharpened")
            
            # Otsu on sharpened
            _, sharpened_otsu = cv2.threshold(sharpened, 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["sharpened_otsu"] = sharpened_otsu
            processed_images.append("sharpened_otsu")
            
            # Strong adaptive thresholding
            adaptive = cv2.adaptiveThreshold(sharpened, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 13, 5)
            image_data["adaptive"] = adaptive
            processed_images.append("adaptive")
            
            # Try inverting the image as sometimes it helps
            inverted = cv2.bitwise_not(base_image)
            _, inverted_otsu = cv2.threshold(inverted, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["inverted_otsu"] = inverted_otsu
            processed_images.append("inverted_otsu")
            
            # Try histogram stretching
            stretched = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX)
            image_data["stretched"] = stretched
            processed_images.append("stretched")
            
            # Multi-scale processing - try different scales
            # Sometimes downscaling and then upscaling removes noise
            height, width = base_image.shape
            down_scale = cv2.resize(base_image, (width//2, height//2), interpolation=cv2.INTER_AREA)
            up_scale = cv2.resize(down_scale, (width, height), interpolation=cv2.INTER_CUBIC)
            _, multi_scale_otsu = cv2.threshold(up_scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["multi_scale_otsu"] = multi_scale_otsu
            processed_images.append("multi_scale_otsu")
            
            # Add local contrast enhancement with different parameters
            clahe_strong = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
            strong_contrast = clahe_strong.apply(base_image)
            _, strong_otsu = cv2.threshold(strong_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["strong_otsu"] = strong_otsu
            processed_images.append("strong_otsu")
        
        # Check for glare in the image (common in smart glasses captures)
        # This will automatically address glare in all strategies if present
        if self._has_glare(base_image):
            logger.info("Glare detected, applying advanced glare reduction")
            glare_reduced = self._advanced_glare_reduction(base_image)
            image_data["glare_reduced"] = glare_reduced
            processed_images.append("glare_reduced")
            
            # Apply Otsu on glare reduced image
            _, glare_otsu = cv2.threshold(glare_reduced, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["glare_otsu"] = glare_otsu
            processed_images.append("glare_otsu")
        
        # Always add original image as final option
        image_data["original"] = image
        processed_images.append("original")
        
        return processed_images, image_data
    
    def _resize_for_optimal_ocr(self, image, image_stats: ImageStats) -> np.ndarray:
        """Resize the image for optimal OCR performance"""
        height, width = image_stats.height, image_stats.width
        max_dimension = self.config["max_image_dimension"]
        
        # For handwritten text, use a smaller max dimension
        if image_stats.image_type == ImageType.HANDWRITTEN:
            max_dimension = min(max_dimension, 1500)  # Limit size for handwritten images
        
        # If image is very large, scale it down more aggressively
        if width > max_dimension or height > max_dimension:
            scale_factor = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Return original if no resizing needed
        return image
    
    def _has_glare(self, image, threshold_percent=5) -> bool:
        """
        Check if the image has significant glare (bright areas)
        
        Args:
            image: Grayscale image
            threshold_percent: Percentage of bright pixels to consider as glare
            
        Returns:
            Boolean indicating presence of glare
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create mask for very bright regions
        _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of bright pixels
        bright_percent = (np.count_nonzero(bright_mask) / bright_mask.size) * 100
        
        return bright_percent > threshold_percent
    
    def _advanced_glare_reduction(self, image) -> np.ndarray:
        """
        Advanced glare reduction using multiple techniques
        
        Args:
            image: Grayscale image
            
        Returns:
            Image with reduced glare
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create a mask for bright regions (potential glare)
        _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Dilate the mask to cover glare areas fully
        kernel = np.ones((9,9), np.uint8)
        dilated_bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        
        # Method 1: Inpaint the glare regions
        try:
            inpainted = cv2.inpaint(gray, dilated_bright_mask, 5, cv2.INPAINT_TELEA)
            
            # Method 2: Blend with median filtered version for smooth transitions
            median_filtered = cv2.medianBlur(gray, 11)
            
            # Create a smoother blend mask
            blend_mask = cv2.GaussianBlur(dilated_bright_mask, (21, 21), 0) / 255.0
            
            # Combine methods based on the blend mask
            glare_reduced = (1 - blend_mask) * gray + blend_mask * inpainted
            
            # Convert to uint8
            glare_reduced = glare_reduced.astype(np.uint8)
            
            # Apply final contrast enhancement to recover details
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(glare_reduced)
        except:
            # Fallback to simpler method if inpainting fails
            mean_filter = cv2.blur(gray, (15, 15))
            glare_reduced = gray.copy()
            glare_reduced[dilated_bright_mask > 0] = mean_filter[dilated_bright_mask > 0]
            
            # Apply contrast enhancement to recover details
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(glare_reduced)
    
    def _perform_enhanced_easyocr_with_fallback(self, image, layout_info):
        """Enhanced EasyOCR with better error handling"""
        try:
            # Try standard EasyOCR process
            text, confidence, method, regions = self._perform_enhanced_easyocr(image, layout_info)
            if text and confidence > 0:
                return text, confidence, method, regions
        except Exception as e:
            logger.warning(f"Primary EasyOCR method failed: {e}, trying alternatives")
        
        # Try with different preprocessing if primary method fails
        try:
            # Apply different preprocessing 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # Try EasyOCR again with preprocessed image
            return self._perform_enhanced_easyocr(enhanced, layout_info)
        except Exception as e:
            logger.error(f"All EasyOCR methods failed: {e}")
            return "", 0, "error", []
    
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
    
    def _perform_enhanced_ocr(self, processed_images: List[str], 
                             image_data: Dict[str, np.ndarray],
                             language: str, 
                             image_stats: ImageStats) -> Tuple[str, str, float, Dict]:
        """
        Perform OCR using enhanced methods with layout analysis
        
        Args:
            processed_images: List of processing method names
            image_data: Dictionary mapping method names to processed images
            language: OCR language
            image_stats: Image statistics
            
        Returns:
            Tuple of (best engine name, text result, confidence score, layout info)
        """
        available_engines = list(self.ocr_engines.keys())
        
        if not available_engines:
            logger.error("No OCR engines available")
            return "none", "", 0, {}
        
        # Determine optimal engine sequence based on image type
        if self.config["use_all_available_engines"]:
            # Try all engines in order of likely effectiveness for this image type
            engine_sequence = self._determine_optimal_engine_sequence(image_stats.image_type)
        else:
            # Use only the first available best engine
            for engine in self._determine_optimal_engine_sequence(image_stats.image_type):
                if engine in available_engines:
                    engine_sequence = [engine]
                    break
            else:
                engine_sequence = [available_engines[0]]
        
        # Filter to only include available engines
        engine_sequence = [e for e in engine_sequence if e in available_engines]
        
        # Detect layout for better text organization
        layout_info = self._detect_document_layout(image_data, image_stats)
        
        # Try each engine in sequence
        best_text = ""
        best_confidence = 0
        best_engine = ""
        
        for engine in engine_sequence:
            # Call the appropriate OCR function based on engine
            if engine == "tesseract":
                text, confidence, method, page_layout = self._perform_enhanced_tesseract_ocr(
                    processed_images, image_data, language, layout_info
                )
                engine_name = f"tesseract_{method}"
                # Update layout with tesseract info if available
                if page_layout:
                    layout_info.update(page_layout)
            
            elif engine == "easyocr":
                text, confidence, method, regions = self._perform_enhanced_easyocr(
                    list(image_data.values())[0],  # Use first preprocessed image
                    layout_info
                )
                engine_name = f"easyocr_{method}"
                # Update layout with region info
                if regions:
                    layout_info["text_regions"] = regions
            
            elif engine == "paddleocr":
                text, confidence, method, regions = self._perform_enhanced_paddleocr(
                    list(image_data.values())[0],  # Use first preprocessed image
                    layout_info
                )
                engine_name = f"paddleocr_{method}"
                # Update layout with region info
                if regions:
                    layout_info["text_regions"] = regions
            
            else:
                continue
            
            # Update best result if this is better using weighted confidence calculation
            weighted_confidence = self._calculate_weighted_confidence(text, confidence, engine)
            if weighted_confidence > best_confidence and len(text.strip()) > 0:
                best_text = text
                best_confidence = confidence  # Keep original confidence for reporting
                best_engine = engine_name
            
            # Early stopping if we have a good result
            if confidence > 80 and len(text.strip()) > 20:
                logger.info(f"Early stopping with engine {engine_name}, confidence {confidence:.1f}")
                break
        
        # If no good results, try fallback methods
        if best_confidence < 30 or len(best_text.strip()) < 10:
            logger.info("Using enhanced fallback OCR methods")
            fallback_text, fallback_conf, fallback_method, fallback_layout = self._perform_enhanced_fallback_ocr(
                image_data, language, layout_info
            )
            
            # Use fallback results if they're better
            if fallback_conf > best_confidence or (len(fallback_text.strip()) > len(best_text.strip())):
                best_text = fallback_text
                best_confidence = fallback_conf
                best_engine = f"fallback_{fallback_method}"
                # Update layout
                if fallback_layout:
                    layout_info.update(fallback_layout)
        
        return best_engine, best_text, best_confidence, layout_info
    
    def _determine_optimal_engine_sequence(self, image_type: ImageType) -> List[str]:
        """
        Determine the optimal sequence of OCR engines based on the image type
        
        Args:
            image_type: Type of image
            
        Returns:
            List of OCR engine names in preferred order
        """
        # Define preferred sequence based on image type
        type_to_engine_sequence = {
            ImageType.DOCUMENT: ["tesseract", "paddleocr", "easyocr"],
            ImageType.HIGH_CONTRAST: ["tesseract", "paddleocr", "easyocr"],
            ImageType.RECEIPT: ["tesseract", "paddleocr", "easyocr"],
            ImageType.ID_CARD: ["tesseract", "easyocr", "paddleocr"],
            ImageType.NATURAL: ["easyocr", "paddleocr", "tesseract"],
            ImageType.SIGNAGE: ["easyocr", "paddleocr", "tesseract"],
            ImageType.HANDWRITTEN: ["easyocr", "paddleocr", "tesseract"],
            ImageType.MIXED: ["paddleocr", "easyocr", "tesseract"],
            ImageType.LOW_QUALITY: ["paddleocr", "easyocr", "tesseract"],
            ImageType.BOOK_PAGE: ["tesseract", "paddleocr", "easyocr"],
            ImageType.NEWSPAPER: ["tesseract", "paddleocr", "easyocr"],
            ImageType.SCIENTIFIC: ["tesseract", "paddleocr", "easyocr"],
            ImageType.PRESENTATION: ["easyocr", "tesseract", "paddleocr"],
            ImageType.FORM: ["tesseract", "paddleocr", "easyocr"],
            ImageType.TABLE: ["tesseract", "paddleocr", "easyocr"]
        }
        
        # Use defined sequence or fallback to default
        if image_type in type_to_engine_sequence:
            return type_to_engine_sequence[image_type]
        
        # Default sequence if not found
        return ["tesseract", "easyocr", "paddleocr"]
    
    def _calculate_weighted_confidence(self, text: str, raw_confidence: float, engine: str) -> float:
        """
        Calculate weighted confidence based on text quality and engine reliability
        
        Args:
            text: Extracted text
            raw_confidence: Raw confidence score
            engine: OCR engine used
            
        Returns:
            Weighted confidence score
        """
        if not text.strip():
            return 0
        
        # Start with the raw confidence
        weighted_confidence = raw_confidence
        
        # Text length factor - longer text with high confidence is usually more reliable
        # but we don't want to overly penalize short text that's correct
        text_length = len(text.strip())
        if text_length < 20:
            length_factor = 0.8
        elif text_length < 50:
            length_factor = 0.9
        elif text_length < 100:
            length_factor = 1.0
        else:
            length_factor = 1.1
        
        # Word count factor - more words usually means more complete text
        word_count = len(text.split())
        if word_count < 3:
            word_factor = 0.8
        elif word_count < 10:
            word_factor = 0.9
        else:
            word_factor = 1.0
        
        # Engine reliability factor based on past performance
        engine_factor = 1.0
        engine_key = engine
        
        if engine_key in self.processing_stats.get("success_rates", {}):
            success_rate = self.processing_stats["success_rates"][engine_key]
            if success_rate > 0.8:
                engine_factor = 1.2
            elif success_rate > 0.6:
                engine_factor = 1.1
            elif success_rate < 0.4:
                engine_factor = 0.9
            elif success_rate < 0.2:
                engine_factor = 0.8
        
        # Text quality factor - check for gibberish or nonsensical text
        # Simple heuristic: high ratio of non-alphanumeric characters often indicates poor OCR
        clean_text = re.sub(r'\s+', '', text)
        if clean_text:
            non_alnum_ratio = sum(1 for c in clean_text if not c.isalnum()) / len(clean_text)
            if non_alnum_ratio > 0.4:
                quality_factor = 0.7
            elif non_alnum_ratio > 0.3:
                quality_factor = 0.8
            elif non_alnum_ratio > 0.2:
                quality_factor = 0.9
            else:
                quality_factor = 1.0
        else:
            quality_factor = 0.5
        
        # Calculate final weighted confidence
        weighted_confidence = weighted_confidence * length_factor * word_factor * engine_factor * quality_factor
        
        # Cap at 100
        return min(100.0, weighted_confidence)
    
    def _detect_document_layout(self, image_data: Dict[str, np.ndarray], 
                               image_stats: ImageStats) -> Dict:
        """
        Detect document layout for better text organization
        
        Args:
            image_data: Dictionary of processed images
            image_stats: Image statistics
            
        Returns:
            Dictionary with layout information
        """
        # Get original or grayscale image
        if "original" in image_data:
            img = image_data["original"]
            if len(img.shape) > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
        elif "gray" in image_data:
            gray = image_data["gray"]
        else:
            gray = list(image_data.values())[0]
        
        # Default layout info
        layout_info = {
            "document_type": image_stats.image_type.value,
            "columns": 1,
            "has_table": image_stats.table_likelihood > 70,
            "has_form": image_stats.form_likelihood > 70,
            "orientation": "portrait" if image_stats.height > image_stats.width else "landscape",
            "regions": []
        }
        
        # Detect text regions
        try:
            # Apply binary thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort regions by position
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter small noise
                if area > 100:
                    regions.append({"x": x, "y": y, "width": w, "height": h, "type": "text"})
            
            # Sort regions by vertical position
            regions.sort(key=lambda r: r["y"])
            
            # Try to detect columns
            if len(regions) > 5:
                # Collect x-coordinates of regions
                x_centers = [r["x"] + r["width"]//2 for r in regions]
                
                # Check for distribution of x centers
                x_hist, bins = np.histogram(x_centers, bins=10)
                peaks = [i for i, val in enumerate(x_hist) if val > len(regions)/20]
                
                # Multiple peaks might indicate columns
                if len(peaks) > 1 and len(peaks) <= 3:  # Cap at 3 columns for reasonability
                    layout_info["columns"] = len(peaks)
            
            # Add regions to layout info
            layout_info["regions"] = regions
            
            # Detect potential headers or titles
            if regions:
                # Headers are typically at the top and wider
                top_regions = [r for r in regions if r["y"] < gray.shape[0] * 0.2]
                if top_regions:
                    # Find the widest region in the top
                    widest = max(top_regions, key=lambda r: r["width"])
                    if widest["width"] > gray.shape[1] * 0.5:  # If it spans at least half the width
                        layout_info["has_header"] = True
                        widest["type"] = "header"
        
        except Exception as e:
            logger.warning(f"Layout detection error: {e}")
        
        return layout_info
    
    def _perform_enhanced_tesseract_ocr(self, processed_images: List[str], 
                                      image_data: Dict[str, np.ndarray], 
                                      language: str,
                                      layout_info: Dict) -> Tuple[str, float, str, Dict]:
        """
        Perform OCR using Tesseract with enhanced parameters
        
        Args:
            processed_images: List of processing method names
            image_data: Dictionary mapping method names to processed images
            language: OCR language
            layout_info: Document layout information
            
        Returns:
            Tuple of (text result, confidence score, method name, page layout info)
        """
        if not TESSERACT_AVAILABLE:
            return "", 0, "unavailable", {}
        
        best_result = ""
        best_confidence = 0
        best_length = 0
        best_method = None
        page_layout = {}
        
        # Optimize PSM (Page Segmentation Mode) based on layout information
        if layout_info.get("has_table", False):
            psm_modes = [6, 4, 3]  # Prefer mode 6 (block of text) for tables
        elif layout_info.get("columns", 1) > 1:
            psm_modes = [3, 1, 11]  # Mode 3 (fully automatic) works well for multi-column
        elif layout_info.get("has_form", False):
            psm_modes = [4, 6, 3]  # Mode 4 (single column) for forms
        else:
            psm_modes = [6, 11, 3, 4]  # Default order
        
        # Early stopping threshold
        confidence_threshold = 85.0
        
        # Track all results for debugging
        all_results = []
        
        # Process each image with different PSM modes
        for img_type in processed_images:
            # Skip processing if we already have a good result
            if best_confidence > confidence_threshold and best_length > 20:
                break
                
            img = image_data[img_type]
            
            for psm in psm_modes:
                try:
                    # Skip processing if we already have a good result
                    if best_confidence > confidence_threshold and best_length > 20:
                        break
                        
                    # Custom config for this particular run with enhanced options
                    custom_config = f'-l {language} --oem 1 --psm {psm}'
                    
                    # For tables, add table detection options
                    if layout_info.get("has_table", False) and psm in [6, 4]:
                        custom_config += ' --dpi 300'
                    
                    # For scientific texts, improve digit and formula recognition
                    if "image_type" in layout_info and layout_info["document_type"] == "scientific":
                        custom_config += ' -c tessedit_char_whitelist="0123456789.+-=()[]{}<>ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"'
                    
                    # Convert to PIL image for Tesseract
                    if isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(img.astype('uint8'))
                    else:
                        pil_img = img
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(pil_img, config=custom_config)
                    
                    # Get confidence data and page segmentation info
                    data = pytesseract.image_to_data(pil_img, config=custom_config, 
                                                    output_type=pytesseract.Output.DICT)
                    
                    # Extract page layout information for better text organization
                    page_segmentation = {}
                    try:
                        # Group words into paragraphs and blocks
                        paragraphs = {}
                        for i in range(len(data['text'])):
                            if data['text'][i].strip():
                                block_num = data['block_num'][i]
                                par_num = data['par_num'][i]
                                line_num = data['line_num'][i]
                                word_num = data['word_num'][i]
                                
                                # Create paragraph key
                                para_key = f"{block_num}_{par_num}"
                                
                                if para_key not in paragraphs:
                                    paragraphs[para_key] = {
                                        "text": [],
                                        "confidence": [],
                                        "bbox": [
                                            data['left'][i], 
                                            data['top'][i], 
                                            data['left'][i] + data['width'][i], 
                                            data['top'][i] + data['height'][i]
                                        ]
                                    }
                                else:
                                    # Update bounding box
                                    paragraphs[para_key]["bbox"][0] = min(paragraphs[para_key]["bbox"][0], data['left'][i])
                                    paragraphs[para_key]["bbox"][1] = min(paragraphs[para_key]["bbox"][1], data['top'][i])
                                    paragraphs[para_key]["bbox"][2] = max(paragraphs[para_key]["bbox"][2], 
                                                                          data['left'][i] + data['width'][i])
                                    paragraphs[para_key]["bbox"][3] = max(paragraphs[para_key]["bbox"][3], 
                                                                          data['top'][i] + data['height'][i])
                                
                                # Add word and confidence
                                paragraphs[para_key]["text"].append(data['text'][i])
                                paragraphs[para_key]["confidence"].append(int(data['conf'][i]) if data['conf'][i] != '-1' else 0)
                        
                        # Create structured page layout
                        page_segmentation["paragraphs"] = []
                        for para_key, para_data in paragraphs.items():
                            # Join text with spaces
                            para_text = " ".join(para_data["text"])
                            # Calculate average confidence
                            conf_values = [c for c in para_data["confidence"] if c > 0]
                            avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0
                            
                            # Add to page segmentation
                            page_segmentation["paragraphs"].append({
                                "text": para_text,
                                "confidence": avg_conf,
                                "bbox": para_data["bbox"]
                            })
                        
                        # Sort paragraphs by vertical position
                        page_segmentation["paragraphs"].sort(key=lambda p: p["bbox"][1])
                    except Exception as e:
                        logger.warning(f"Error extracting page layout: {e}")
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Get text length (non-whitespace)
                    text_length = len(''.join(text.split()))
                    
                    # Calculate word count
                    word_count = len(text.split())
                    
                    # Store results for debugging
                    all_results.append({
                        "method": img_type,
                        "psm": psm,
                        "confidence": avg_confidence,
                        "text_length": text_length,
                        "word_count": word_count,
                        "sample": text[:50] + "..." if len(text) > 50 else text
                    })
                    
                    # Calculate overall score
                    # Balance between confidence and text length
                    score = avg_confidence * 0.8 + (min(100, text_length) / 100) * 20
                    
                    # Penalty for very short results with high confidence
                    if avg_confidence > 80 and word_count < 3 and text_length < 15:
                        score -= 15
                    
                    # Bonus for table mode if image has table
                    if layout_info.get("has_table", False) and psm in [6, 4]:
                        score += 10
                    
                    # Update best result if this is better
                    if score > best_confidence or (
                            score == best_confidence and text_length > best_length):
                        best_confidence = avg_confidence
                        best_result = text
                        best_length = text_length
                        best_method = f"{img_type}_psm{psm}"
                        page_layout = page_segmentation
                        
                        logger.info(f"New best Tesseract OCR result: {img_type}, PSM {psm}, "
                                  f"Confidence: {avg_confidence:.1f}, Length: {text_length}, "
                                  f"Words: {word_count}")
                    
                except Exception as e:
                    logger.error(f"Tesseract OCR error for {img_type} with PSM {psm}: {e}")
        
        # If in debug mode, log all results
        if self.config["debug_mode"]:
            logger.debug(f"All Tesseract OCR results: {all_results}")
        
        return best_result, best_confidence, best_method, page_layout
    
    def _perform_enhanced_easyocr(self, image, layout_info: Dict) -> Tuple[str, float, str, List]:
        """
        Perform OCR using EasyOCR with enhanced processing
        
        Args:
            image: Image to process
            layout_info: Document layout information
            
        Returns:
            Tuple of (text result, confidence score, method name, regions)
        """
        if not EASYOCR_AVAILABLE or not 'reader' in globals():
            return "", 0, "unavailable", []
        
        try:
            # Wait for initialization if needed
            if 'easyocr_thread' in globals() and easyocr_thread.is_alive():
                logger.info("Waiting for EasyOCR initialization to complete...")
                easyocr_thread.join(timeout=10)
            
            # Save image to file for EasyOCR (it works better with files)
            temp_path = f"/tmp/easyocr_temp_{uuid.uuid4()}.jpg"
            cv2.imwrite(temp_path, image)
            
            # Configure EasyOCR based on document type
            paragraph = False
            detail = 0  # 0 for fastest mode, 1 for more accurate
            
            # Use paragraph mode for documents and book pages
            if layout_info.get("document_type") in ["document", "book_page", "newspaper"]:
                paragraph = True
                detail = 1  # More detailed for documents
            
            # For natural scenes or signage, use detail mode for better accuracy
            elif layout_info.get("document_type") in ["natural", "signage", "mixed"]:
                detail = 1
            
            # Perform OCR with EasyOCR
            results = reader.readtext(temp_path, paragraph=paragraph, detail=detail)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if not results:
                return "", 0, "no_text", []
            
            # Process results based on format (depends on paragraph mode)
            if paragraph:
                # Paragraph mode returns combined text blocks
                text = " ".join([r[1] for r in results if r[1].strip()])
                confidences = [r[2] * 100 for r in results if r[1].strip()]  # Scale to 0-100
                regions = [{"bbox": r[0], "text": r[1], "confidence": r[2] * 100} for r in results if r[1].strip()]
            else:
                # Extract text and confidence
                texts = []
                confidences = []
                regions = []
                
                for (bbox, text, conf) in results:
                    if text.strip():
                        texts.append(text)
                        confidences.append(conf * 100)  # Scale to 0-100
                        regions.append({"bbox": bbox, "text": text, "confidence": conf * 100})
                
                # Join text with appropriate spacing
                text = ' '.join(texts)
            
            if not text:
                return "", 0, "empty", []
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Sort regions by vertical position for better text organization
            regions.sort(key=lambda r: r["bbox"][0][1])  # Sort by y-coordinate of top-left point
            
            logger.info(f"EasyOCR result: {len(regions)} text regions, "
                      f"Confidence: {avg_confidence:.1f}")
            
            return text, avg_confidence, "enhanced", regions
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0, "error", []
    
    def _perform_enhanced_paddleocr(self, image, layout_info: Dict) -> Tuple[str, float, str, List]:
        """
        Perform OCR using PaddleOCR with enhanced processing
        
        Args:
            image: Image to process
            layout_info: Document layout information
            
        Returns:
            Tuple of (text result, confidence score, method name, regions)
        """
        if not PADDLE_OCR_AVAILABLE or not 'paddle_ocr' in globals():
            return "", 0, "unavailable", []
        
        try:
            # Wait for initialization if needed
            if 'paddle_thread' in globals() and paddle_thread.is_alive():
                logger.info("Waiting for PaddleOCR initialization to complete...")
                paddle_thread.join(timeout=10)
            
            # Save image to file for PaddleOCR
            temp_path = f"/tmp/paddleocr_temp_{uuid.uuid4()}.jpg"
            cv2.imwrite(temp_path, image)
            
            # Configure PaddleOCR based on document type
            use_angle_cls = True  # Enable rotation detection
            
            # Adjust params for different document types
            rec_model_dir = None  # Default model
            
            # Perform OCR with PaddleOCR
            results = paddle_ocr.ocr(temp_path, cls=use_angle_cls, rec_model_dir=rec_model_dir)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if not results or not results[0]:
                return "", 0, "no_text", []
            
            # Extract text and confidence
            texts = []
            confidences = []
            regions = []
            
            for line in results[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    conf = line[1][1]
                    
                    if text.strip():
                        texts.append(text)
                        confidences.append(conf * 100)  # Scale to 0-100
                        regions.append({
                            "bbox": bbox,
                            "text": text,
                            "confidence": conf * 100
                        })
            
            if not texts:
                return "", 0, "empty", []
            
            # Sort regions by vertical position
            regions.sort(key=lambda r: r["bbox"][0][1])  # Sort by y-coordinate
            
            # Process text based on document type
            if layout_info.get("document_type") in ["document", "book_page", "newspaper"]:
                # For document-type images, try to preserve paragraph structure
                # Group lines that are close together vertically
                paragraphs = []
                current_paragraph = []
                prev_y = None
                
                for region in regions:
                    # Get average y-coordinate of bottom of bounding box
                    current_y = (region["bbox"][2][1] + region["bbox"][3][1]) / 2
                    
                    if prev_y is not None and abs(current_y - prev_y) > 30:  # Threshold for new paragraph
                        if current_paragraph:
                            paragraphs.append(" ".join(current_paragraph))
                            current_paragraph = []
                    
                    current_paragraph.append(region["text"])
                    prev_y = current_y
                
                # Add the last paragraph
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                
                # Join paragraphs with double newlines
                full_text = "\n\n".join(paragraphs)
            else:
                # For other types, just join with spaces
                full_text = ' '.join(texts)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"PaddleOCR result: {len(texts)} text regions, "
                      f"Confidence: {avg_confidence:.1f}")
            
            return full_text, avg_confidence, "enhanced", regions
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return "", 0, "error", []
    
    def _perform_enhanced_fallback_ocr(self, image_data: Dict[str, np.ndarray], 
                                     language: str, layout_info: Dict) -> Tuple[str, float, str, Dict]:
        """
        Perform advanced fallback OCR for difficult images
        
        Args:
            image_data: Dictionary of processed images
            language: OCR language
            layout_info: Document layout information
            
        Returns:
            Tuple of (text result, confidence score, method name, layout info)
        """
        try:
            # Select the original or grayscale image
            if "gray" in image_data:
                gray = image_data["gray"]
            elif "original" in image_data:
                original = image_data["original"]
                if len(original.shape) > 2:
                    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                else:
                    gray = original
            else:
                # Get any image from the dictionary
                for img_name, img in image_data.items():
                    if len(img.shape) > 2:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img
                    break
                else:
                    return "", 0, "no_image", {}
            
            # Apply advanced processing techniques
            
            # 1. Multi-scale OCR approach - sometimes scaling helps
            scales = [1.0, 1.5, 0.75]
            scale_results = []
            
            for scale in scales:
                if scale != 1.0:
                    # Resize the image
                    h, w = gray.shape
                    scaled = cv2.resize(gray, (int(w * scale), int(h * scale)), 
                                      interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
                else:
                    scaled = gray
                
                # Apply extreme contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                enhanced = clahe.apply(scaled)
                
                # Apply adaptive thresholding
                adaptive = cv2.adaptiveThreshold(enhanced, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 21, 11)
                
                # Try multiple PSM modes for Tesseract
                psm_options = [6, 3, 11]
                for psm in psm_options:
                    try:
                        # Convert to PIL image for Tesseract
                        pil_img = Image.fromarray(adaptive)
                        
                        # Custom config
                        custom_config = f'-l {language} --oem 1 --psm {psm}'
                        
                        # Get text
                        text = pytesseract.image_to_string(pil_img, config=custom_config)
                        
                        if not text.strip():
                            continue
                        
                        # Get confidence data
                        data = pytesseract.image_to_data(pil_img, config=custom_config, 
                                                      output_type=pytesseract.Output.DICT)
                        
                        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        # Add to results if text is meaningful
                        if len(text.strip()) > 10 and avg_confidence > 30:
                            scale_results.append({
                                "text": text,
                                "confidence": avg_confidence,
                                "method": f"scale_{scale}_psm{psm}"
                            })
                    except Exception as e:
                        logger.warning(f"Error in scale OCR: {e}")
            
            # 2. Region-based OCR - attempt to extract text from specific regions
            region_results = []
            
            # If we have region information from layout
            if "regions" in layout_info and layout_info["regions"]:
                for region in layout_info["regions"]:
                    try:
                        # Extract region
                        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                        if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= gray.shape[1] and y + h <= gray.shape[0]:
                            roi = gray[y:y+h, x:x+w]
                            
                            # Apply specialized processing for small regions
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
                            roi_enhanced = clahe.apply(roi)
                            
                            # Scale up small regions
                            if roi.shape[0] < 50 or roi.shape[1] < 100:
                                scale_factor = max(2, 150 / max(roi.shape[0], roi.shape[1]))
                                roi_enhanced = cv2.resize(roi_enhanced, None, fx=scale_factor, fy=scale_factor, 
                                                       interpolation=cv2.INTER_CUBIC)
                            
                            # Apply adaptive thresholding
                            roi_binary = cv2.adaptiveThreshold(roi_enhanced, 255, 
                                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                            cv2.THRESH_BINARY, 11, 5)
                            
                            # Try OCR on region
                            pil_roi = Image.fromarray(roi_binary)
                            custom_config = f'-l {language} --oem 1 --psm 6'
                            
                            roi_text = pytesseract.image_to_string(pil_roi, config=custom_config)
                            
                            if roi_text.strip():
                                region_results.append({
                                    "text": roi_text,
                                    "region": (x, y, w, h),
                                    "method": "region_based"
                                })
                    except Exception as e:
                        logger.warning(f"Error in region OCR: {e}")
            
            # 3. Try document warping as a last resort
            warping_result = None
            try:
                warped = self._try_advanced_document_warping(gray)
                
                if warped is not None:
                    # Apply Otsu thresholding
                    _, warped_otsu = cv2.threshold(warped, 0, 255, 
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Convert to PIL image for Tesseract
                    pil_img = Image.fromarray(warped_otsu.astype('uint8'))
                    
                    # Try OCR with document PSM
                    custom_config = f'-l {language} --oem 1 --psm 6'
                    
                    text = pytesseract.image_to_string(pil_img, config=custom_config)
                    
                    if text.strip():
                        # Get confidence
                        data = pytesseract.image_to_data(pil_img, config=custom_config, 
                                                      output_type=pytesseract.Output.DICT)
                        
                        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        warping_result = {
                            "text": text,
                            "confidence": avg_confidence,
                            "method": "warped_document"
                        }
            except Exception as e:
                logger.warning(f"Document warping error: {e}")
            
            # Combine and select the best result
            all_results = scale_results + ([warping_result] if warping_result else [])
            
            if all_results:
                # Sort by confidence
                all_results.sort(key=lambda x: x["confidence"], reverse=True)
                best = all_results[0]
                
                # If region results are available, incorporate them
                if region_results:
                    # Sort regions by vertical position
                    region_results.sort(key=lambda x: x["region"][1])
                    region_text = "\n".join([r["text"] for r in region_results])
                    
                    # If region text is longer and seems more complete, use it
                    if len(region_text.strip()) > len(best["text"].strip()) * 1.2:
                        return region_text, 50.0, "region_combined", {"regions": region_results}
                
                return best["text"], best["confidence"], best["method"], {}
            
            # If no results from advanced methods, try one more desperate approach
            # Extreme binarization with multiple thresholds
            best_text = ""
            best_confidence = 0
            best_method = ""
            
            for threshold in [127, 100, 150, 80, 180]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                
                # Add dilation to connect broken text
                kernel = np.ones((2, 2), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=1)
                
                # Try OCR on this binary image
                try:
                    # Convert to PIL image for Tesseract
                    pil_img = Image.fromarray(dilated.astype('uint8'))
                    
                    # Try with PSM 6
                    custom_config = f'-l {language} --oem 1 --psm 6'
                    
                    text = pytesseract.image_to_string(pil_img, config=custom_config)
                    
                    if text.strip():
                        # Get confidence
                        data = pytesseract.image_to_data(pil_img, config=custom_config, 
                                                      output_type=pytesseract.Output.DICT)
                        
                        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        if avg_confidence > best_confidence or len(text) > len(best_text) * 1.5:
                            best_text = text
                            best_confidence = avg_confidence
                            best_method = f"extreme_binary_{threshold}"
                except Exception as e:
                    logger.warning(f"Fallback OCR error with threshold {threshold}: {e}")
            
            if best_text:
                return best_text, best_confidence, best_method, {}
            
            # If nothing worked, return empty result
            return "", 0, "all_failed", {}
            
        except Exception as e:
            logger.error(f"Enhanced fallback OCR error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "", 0, "error", {}
    
    def _try_advanced_document_warping(self, gray_image):
        """
        Try to detect and warp document corners with enhanced methods
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Warped image if successful, None otherwise
        """
        # Try multiple approaches and return the best result
        
        # Approach 1: Standard contour-based approach
        try:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if it's large enough to be a document
            if cv2.contourArea(largest_contour) < 0.2 * gray_image.shape[0] * gray_image.shape[1]:
                return None
            
            # Approximate the contour to find corners
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If it has 4 corners, it might be a document
            if len(approx) == 4:
                # Order the points for perspective transform
                pts = np.array([pt[0] for pt in approx])
                rect = self._order_points(pts)
                
                # Get dimensions
                width = max(
                    int(np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))),
                    int(np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2)))
                )
                
                height = max(
                    int(np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))),
                    int(np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2)))
                )
                
                # Create destination points
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")
                
                # Compute perspective transform
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(gray_image, M, (width, height))
                
                return warped
        except Exception as e:
            logger.warning(f"Standard warping approach failed: {e}")
        
        # Approach 2: Hough line-based approach to find document edges
        try:
            # Enhance edges
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Find lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) < 4:
                return None
            
            # Extend lines and find intersections
            h, w = gray_image.shape
            corners = []
            
            # Group lines into horizontal and vertical
            h_lines = []
            v_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 45 or angle > 135:  # Horizontal
                    h_lines.append(line[0])
                else:  # Vertical
                    v_lines.append(line[0])
            
            # Need at least 2 lines in each direction
            if len(h_lines) < 2 or len(v_lines) < 2:
                return None
            
            # Find 4 extreme lines (top, bottom, left, right)
            h_lines.sort(key=lambda l: l[1] + l[3])  # Sort by y
            v_lines.sort(key=lambda l: l[0] + l[2])  # Sort by x
            
            top_line = h_lines[0]
            bottom_line = h_lines[-1]
            left_line = v_lines[0]
            right_line = v_lines[-1]
            
            # Find intersections of these lines
            def line_intersection(line1, line2):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2
                
                # Check if lines are parallel
                d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if d == 0:
                    return None
                
                # Calculate intersection point
                x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
                y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
                
                # Check if intersection is within image bounds
                if 0 <= x < w and 0 <= y < h:
                    return (int(x), int(y))
                return None
            
            # Find the four corners
            top_left = line_intersection(top_line, left_line)
            top_right = line_intersection(top_line, right_line)
            bottom_left = line_intersection(bottom_line, left_line)
            bottom_right = line_intersection(bottom_line, right_line)
            
            # If all corners found, warp the image
            if all([top_left, top_right, bottom_left, bottom_right]):
                src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                
                # Calculate width and height for the new image
                width = int(max(np.linalg.norm(np.array(top_right) - np.array(top_left)),
                              np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))))
                height = int(max(np.linalg.norm(np.array(bottom_left) - np.array(top_left)),
                               np.linalg.norm(np.array(bottom_right) - np.array(top_right))))
                
                dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
                
                # Get perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Warp the image
                warped = cv2.warpPerspective(gray_image, M, (width, height))
                
                return warped
        except Exception as e:
            logger.warning(f"Hough line warping approach failed: {e}")
        
        # No successful approach
        return None
    
    def _order_points(self, pts):
        """
        Order points for perspective transform
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered points [top-left, top-right, bottom-right, bottom-left]
        """
        # Convert to numpy array if not already
        pts = np.array(pts, dtype=np.float32)
        
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
        
        # Difference of coordinates
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
        
        return rect
    
    def _enhanced_post_process_text(self, text: str, image_type: ImageType) -> str:
        """
        Apply enhanced rule-based post-processing to extracted text
        
        Args:
            text: Extracted text
            image_type: Type of image
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Apply specific corrections based on image type
        if image_type == ImageType.RECEIPT:
            # Fix common receipt OCR errors
            text = self._fix_receipt_text(text)
        elif image_type == ImageType.ID_CARD:
            # Fix common ID card OCR errors
            text = self._fix_id_card_text(text)
        elif image_type == ImageType.SCIENTIFIC:
            # Fix formula and scientific notation
            text = self._fix_scientific_text(text)
        elif image_type == ImageType.FORM:
            # Fix form field text
            text = self._fix_form_text(text)
        elif image_type == ImageType.TABLE:
            # Fix table text
            text = self._fix_table_text(text)
        
        # Apply general text corrections
        text = self._apply_general_text_corrections(text)
        
        # Enhance text organization
        text = self._enhance_text_organization(text, image_type)
        
        return text
    
    def _fix_receipt_text(self, text: str) -> str:
        """
        Fix common OCR errors in receipts with enhanced corrections
        
        Args:
            text: Receipt text
            
        Returns:
            Corrected text
        """
        # Fix currency symbols and amounts
        text = re.sub(r'([0-9]+)\.([0-9]{2})([^0-9])', r'$\1.\2\3', text)
        
        # Fix percentage signs
        text = re.sub(r'([0-9]+)[,.]([0-9]+)o\/?', r'\1.\2%', text)
        
        # Fix common receipt words
        replacements = {
            r'\bTOTAI\b': 'TOTAL',
            r'\bSUBTOTAI\b': 'SUBTOTAL',
            r'\bCASI-I\b': 'CASH',
            r'\bCHANGI\b': 'CHANGE',
            r'\bDISCOUNI\b': 'DISCOUNT',
            r'\bITEMS\b': 'ITEMS',
            r'\bTAX\b': 'TAX',
            r'\bDUE\b': 'DUE',
            r'\bDATE\b': 'DATE',
            r'\bTIME\b': 'TIME',
            r'\bTHANI< YOU\b': 'THANK YOU',
            r'\bTHANKS\b': 'THANKS',
            r'\bCARD\b': 'CARD',
            r'\bCASHIER\b': 'CASHIER',
            r'\bINVOICE\b': 'INVOICE',
            r'\bNO\.\b': 'NO.',
            r'\bDESCRIPTION\b': 'DESCRIPTION',
            r'\bQTY\b': 'QTY',
            r'\bPRICE\b': 'PRICE',
            r'\bAMOUNT\b': 'AMOUNT',
            r'\bDISCOUNT\b': 'DISCOUNT',
            r'\bSUBTOTAL\b': 'SUBTOTAL',
            r'\bTAX\b': 'TAX',
            r'\bTOTAL\b': 'TOTAL',
            r'\bPMT\b': 'PAYMENT',
            r'\bVAT\b': 'VAT',
            r'\bCASH\b': 'CASH',
            r'\bCARD\b': 'CARD',
            r'\bDEBIT\b': 'DEBIT',
            r'\bCREDIT\b': 'CREDIT',
            r'\bTHANK YOU\b': 'THANK YOU'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Enhanced detection of receipt items
        lines = text.split('\n')
        formatted_lines = []
        
        # Flag to identify item sections
        in_item_section = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                formatted_lines.append("")
                continue
            
            # Check for section headers
            if re.match(r'^(ITEM|DESCRIPTION|PRODUCT|GOODS)S?', line, re.IGNORECASE):
                in_item_section = True
                formatted_lines.append(line)
                continue
            
            # Check for end of item section
            if in_item_section and re.match(r'^(SUBTOTAL|TOTAL|TAX|DISCOUNT)', line, re.IGNORECASE):
                in_item_section = False
            
            # Format receipt items if in item section
            if in_item_section:
                # Try to detect and format item with quantity and price
                item_match = re.search(r'^(.+?)(?:\s+(\d+))?(?:\s+(?:x|@)\s+)?([0-9.,]+)', line)
                
                if item_match:
                    item_name = item_match.group(1).strip()
                    quantity = item_match.group(2) or "1"
                    price = item_match.group(3).strip()
                    
                    formatted_line = f"{item_name}: {quantity} x ${price}"
                    formatted_lines.append(formatted_line)
                else:
                    formatted_lines.append(line)
            else:
                # Regular line
                formatted_lines.append(line)
        
        # Rejoin lines
        text = '\n'.join(formatted_lines)
        
        # Format total, subtotal, tax lines
        text = re.sub(r'(?i)subtotal\s*[:,]?\s*[$]?([0-9.,]+)', r'SUBTOTAL: $\1', text)
        text = re.sub(r'(?i)tax\s*[:,]?\s*[$]?([0-9.,]+)', r'TAX: $\1', text)
        text = re.sub(r'(?i)total\s*[:,]?\s*[$]?([0-9.,]+)', r'TOTAL: $\1', text)
        
        return text
    
    def _fix_id_card_text(self, text: str) -> str:
        """
        Fix common OCR errors in ID cards with enhanced corrections
        
        Args:
            text: ID card text
            
        Returns:
            Corrected text
        """
        # Fix common ID card fields with enhanced patterns
        replacements = {
            r'\bNAME\b': 'NAME',
            r'\bADDRESS\b': 'ADDRESS',
            r'\bDOB\b': 'DATE OF BIRTH',
            r'\bDATI?E? OF BIRTH\b': 'DATE OF BIRTH',
            r'\bEXP(?:IRATION)? ?DATI?E?\b': 'EXPIRATION DATE',
            r'\bSEX\b': 'SEX',
            r'\bGENDER\b': 'GENDER',
            r'\bHEIGH[TI]\b': 'HEIGHT',
            r'\bWEIGH[TI]\b': 'WEIGHT',
            r'\bEYES?\b': 'EYES',
            r'\bHAIR\b': 'HAIR',
            r"\b(?:DRIVER'?S|DRIV|ORIV) LIC[, ]?(?:NO|NUM|NUMBER)\b": 'DRIVER\'S LICENSE NO',
            r'\bISSUE(?:D)? DATE\b': 'ISSUE DATE',
            r'\bIDENTITY(?:\s+CARD)?\b': 'IDENTITY CARD',
            r'\bID(?:\s+CARD)?\b': 'ID CARD',
            r'\bCITIZEN\b': 'CITIZEN',
            r'\bNATIONALITY\b': 'NATIONALITY',
            r'\bPLACE OF BIRTH\b': 'PLACE OF BIRTH',
            r'\bRELIGION\b': 'RELIGION',
            r'\bMARITAL STATUS\b': 'MARITAL STATUS',
            r'\bBLOOD(?: TYPE)?\b': 'BLOOD TYPE',
            r'\bOCCUPATION\b': 'OCCUPATION',
            r'\bSIGNATURE\b': 'SIGNATURE'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Format common ID fields with colon for better readability
        id_fields = [
            'NAME', 'ADDRESS', 'DATE OF BIRTH', 'EXPIRATION DATE', 'SEX', 'GENDER',
            'HEIGHT', 'WEIGHT', 'EYES', 'HAIR', 'DRIVER\'S LICENSE NO', 'ISSUE DATE',
            'PLACE OF BIRTH', 'NATIONALITY', 'RELIGION', 'MARITAL STATUS', 'BLOOD TYPE',
            'OCCUPATION'
        ]
        
        for field in id_fields:
            # Add colon if field exists but doesn't have one
            pattern = f'({field})\\s+([^:\\n]+)'
            replacement = r'\1: \2'
            text = re.sub(pattern, replacement, text)
        
        # Format dates correctly (various formats)
        date_patterns = [
            # MM/DD/YYYY or similar
            (r'(\d{1,2})[/\-\.\\](\d{1,2})[/\-\.\\](\d{2,4})', r'\1/\2/\3'),
            # MMDDYYYY
            (r'(\d{2})(\d{2})(\d{4})', r'\1/\2/\3'),
            # DD-MMM-YYYY
            (r'(\d{1,2})[-\s](JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[-\s](\d{2,4})', 
            r'\1 \2 \3')
        ]
        
        for pattern, replacement in date_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Correct ID numbers (format various ID numbers)
        # Remove unwanted spaces and characters in ID numbers
        id_pattern = r'(?:ID|NUMBER|LICENSE|NO|#)[:.\s-]*([A-Z0-9][\sA-Z0-9\-\.]{4,})'
        matches = re.finditer(id_pattern, text, re.IGNORECASE)
        
        for match in matches:
            raw_id = match.group(1)
            # Remove spaces and unwanted characters
            clean_id = re.sub(r'[^A-Z0-9]', '', raw_id, flags=re.IGNORECASE)
            # Format based on length
            if len(clean_id) in [9, 10]:
                # Common ID lengths - add hyphens for readability
                formatted_id = '-'.join([clean_id[:3], clean_id[3:6], clean_id[6:]])
                text = text.replace(raw_id, formatted_id)
        
        return text
    
    def _fix_scientific_text(self, text: str) -> str:
        """
        Fix scientific and mathematical notation
        
        Args:
            text: Scientific text
            
        Returns:
            Corrected text
        """
        # Fix common scientific notation errors
        # Fix superscripts
        text = re.sub(r'(\d)[\^](\d+)', r'\1\u00B2', text)  # x^2 -> x²
        text = re.sub(r'(\d)[\^]2', r'\1\u00B2', text)  # x^2 -> x²
        text = re.sub(r'(\d)[\^]3', r'\1\u00B3', text)  # x^3 -> x³
        
        # Fix subscripts
        text = re.sub(r'([A-Za-z])_(\d)', r'\1\u208\2', text)  # H_2O -> H₂O
        
        # Fix common scientific symbols
        replacements = {
            r'(?<=\d)x(?=\d)': '×',         # 2x3 -> 2×3
            'alpha': 'α',
            'beta': 'β',
            'gamma': 'γ',
            'delta': 'δ',
            'epsilon': 'ε',
            'theta': 'θ',
            'lambda': 'λ',
            'micro': 'µ',
            'pi': 'π',
            'sigma': 'σ',
            'Sigma': 'Σ',
            'tau': 'τ',
            'phi': 'φ',
            'omega': 'ω',
            'Omega': 'Ω',
            'approx': '≈',
            'neq': '≠',
            'leq': '≤',
            'geq': '≥',
            r'(?<!\w)inf(?!\w)': '∞',
            'sqrt': '√',
            'integral': '∫',
            'nabla': '∇',
            'union': '∪',
            'intersect': '∩',
            'in': '∈',
            'notin': '∉',
            'subset': '⊂',
            'superset': '⊃',
            'partial': '∂',
            'sum': '∑',
            'product': '∏',
            'deg(ree)?s?': '°',
            r'\+/-': '±',
            r'\(\+/-\)': '±'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix chemical formulas
        text = re.sub(r'([A-Z][a-z]?)(\d+)', r'\1\u208\2', text)  # CO2 -> CO₂
        
        # Fix common unit errors
        units = {
            r'([0-9]+)([^0-9\s]+[Cc])': r'\1 °C',  # Fix Celsius
            r'([0-9]+)([^0-9\s]+[Ff])': r'\1 °F',  # Fix Fahrenheit
            r'([0-9]+)([^0-9\s]+[Kk])': r'\1 K',   # Fix Kelvin
            r'([0-9]+)([^0-9\s]*)[Mm][Ll]': r'\1 ml',  # Fix milliliters
            r'([0-9]+)([^0-9\s]*)[Mm][Gg]': r'\1 mg',  # Fix milligrams
            r'([0-9]+)([^0-9\s]*)[Kk][Gg]': r'\1 kg',  # Fix kilograms
            r'([0-9]+)([^0-9\s]*)[Cc][Mm]': r'\1 cm',  # Fix centimeters
            r'([0-9]+)([^0-9\s]*)[Mm][Mm]': r'\1 mm',  # Fix millimeters
            r'([0-9]+)([^0-9\s]*)[Kk][Mm]': r'\1 km'   # Fix kilometers
        }
        
        for pattern, replacement in units.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _fix_form_text(self, text: str) -> str:
        """
        Fix common errors in form text
        
        Args:
            text: Form text
            
        Returns:
            Corrected text
        """
        # Fix common form field labels
        form_fields = {
            r'\b(?:F|f)irst\s*(?:N|n)ame\b': 'First Name',
            r'\b(?:L|l)ast\s*(?:N|n)ame\b': 'Last Name',
            r'\b(?:M|m)iddle\s*(?:N|n)ame\b': 'Middle Name',
            r'\b(?:F|f)ull\s*(?:N|n)ame\b': 'Full Name',
            r'\b(?:A|a)ddress\b': 'Address',
            r'\b(?:C|c)ity\b': 'City',
            r'\b(?:S|s)tate\b': 'State',
            r'\b(?:Z|z)ip\s*(?:C|c)ode\b': 'Zip Code',
            r'\b(?:P|p)ostal\s*(?:C|c)ode\b': 'Postal Code',
            r'\b(?:C|c)ountry\b': 'Country',
            r'\b(?:E|e)mail\b': 'Email',
            r'\b(?:P|p)hone\b': 'Phone',
            r'\b(?:M|m)obile\b': 'Mobile',
            r'\b(?:D|d)ate\s*(?:O|o)f\s*(?:B|b)irth\b': 'Date of Birth',
            r'\b(?:G|g)ender\b': 'Gender',
            r'\b(?:O|o)ccupation\b': 'Occupation',
            r'\b(?:C|c)ompany\b': 'Company',
            r'\b(?:D|d)epartment\b': 'Department',
            r'\b(?:S|s)ignature\b': 'Signature',
            r'\b(?:D|d)ate\b': 'Date'
        }
        
        for pattern, replacement in form_fields.items():
            text = re.sub(pattern, replacement, text)
        
        # Ensure form field labels are followed by colons
        for field in form_fields.values():
            # Add colon if field exists but doesn't have one
            pattern = f'({field})\\s+([^:\\n]+)'
            replacement = r'\1: \2'
            text = re.sub(pattern, replacement, text)
        
        # Fix check boxes
        text = re.sub(r'\[\s*[xX✓✔]\s*\]', '☑', text)  # Checked box
        text = re.sub(r'\[\s*\]', '☐', text)  # Empty box
        
        # Fix form structure - newlines after each field
        for field in form_fields.values():
            pattern = f'({field}:\\s+[^\\n]+)([^\\n])'
            replacement = r'\1\n\2'
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _fix_table_text(self, text: str) -> str:
        """
        Fix common errors in table text
        
        Args:
            text: Table text
            
        Returns:
            Corrected text
        """
        # Attempt to detect table structure
        lines = text.split('\n')
        
        # Check if this looks like a table with columns
        # Look for delimiter patterns
        if any('|' in line for line in lines) or any('\t' in line for line in lines):
            # Already has delimiters, normalize them
            formatted_lines = []
            
            for line in lines:
                # Replace tabs with pipe delimiters
                line = line.replace('\t', ' | ')
                
                # Normalize pipe delimiters (ensure spaces around them)
                line = re.sub(r'\s*\|\s*', ' | ', line)
                
                # Remove empty columns
                line = re.sub(r'\|\s+\|', '|', line)
                
                # Add to formatted lines
                formatted_lines.append(line)
            
            # Join lines
            table_text = '\n'.join(formatted_lines)
            
            # Try to detect header and add separator
            if len(formatted_lines) > 1:
                if '|' in formatted_lines[0] and '|' in formatted_lines[1]:
                    # Insert separator after header
                    header_parts = formatted_lines[0].split('|')
                    separator_parts = ['-' * len(part.strip()) for part in header_parts]
                    separator_line = '|'.join(separator_parts)
                    
                    formatted_lines.insert(1, separator_line)
                    table_text = '\n'.join(formatted_lines)
            
            return table_text
            
        else:
            # Try to detect columns based on space alignment
            # Check for consistent spacing that might indicate columns
            # Look for words with large gaps between them
            words_positions = []
            
            for line in lines:
                # Find all word positions in the line
                positions = []
                for match in re.finditer(r'\S+', line):
                    positions.append((match.start(), match.end()))
                words_positions.append(positions)
            
            # Need enough lines to detect a pattern
            if len(words_positions) > 2:
                # Try to find column boundaries
                col_starts = {}
                col_ends = {}
                
                for positions in words_positions:
                    for start, end in positions:
                        col_starts[start] = col_starts.get(start, 0) + 1
                        col_ends[end] = col_ends.get(end, 0) + 1
                
                # Find potential column boundaries (frequent positions)
                line_count = len(words_positions)
                threshold = line_count * 0.4  # At least 40% of lines should have a boundary here
                
                potential_cols = sorted([
                    pos for pos, count in col_starts.items() if count >= threshold
                ] + [
                    pos for pos, count in col_ends.items() if count >= threshold
                ])
                
                # Merge close positions
                col_boundaries = []
                curr_boundary = None
                
                for pos in potential_cols:
                    if curr_boundary is None:
                        curr_boundary = pos
                    elif pos - curr_boundary < 5:  # Merge if very close
                        curr_boundary = (curr_boundary + pos) // 2
                    else:
                        col_boundaries.append(curr_boundary)
                        curr_boundary = pos
                
                if curr_boundary is not None:
                    col_boundaries.append(curr_boundary)
                
                # If we have enough column boundaries, format as a table
                if len(col_boundaries) >= 2:
                    formatted_lines = []
                    
                    for line in lines:
                        if not line.strip():
                            formatted_lines.append("")
                            continue
                        
                        # Insert pipe delimiters at column boundaries
                        new_line = ""
                        last_pos = 0
                        
                        for boundary in col_boundaries:
                            if boundary > len(line):
                                continue
                            
                            new_line += line[last_pos:boundary] + " | "
                            last_pos = boundary
                        
                        if last_pos < len(line):
                            new_line += line[last_pos:]
                        
                        # Clean up multiple delimiters
                        new_line = re.sub(r'\|\s+\|', '|', new_line)
                        
                        formatted_lines.append(new_line)
                    
                    # Try to add a separator after header
                    if len(formatted_lines) > 1:
                        header_parts = formatted_lines[0].split('|')
                        separator_parts = ['-' * len(part.strip()) for part in header_parts]
                        separator_line = '|'.join(separator_parts)
                        
                        formatted_lines.insert(1, separator_line)
                    
                    return '\n'.join(formatted_lines)
        
        # If no table structure detected, return original text
        return text
    
    def _apply_general_text_corrections(self, text: str) -> str:
        """
        Apply enhanced general text corrections for OCR output
        
        Args:
            text: OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return ""
        
        # Remove invalid unicode characters
        text = ''.join(c for c in text if ord(c) < 65536)
        
        # Fix quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("''", '"').replace(",,", '"')
        text = text.replace("'", "'").replace("`", "'")
        
        # Fix bullet points and normalize them
        text = re.sub(r'[\*\+\-‣▪•●·](?:\s+|\n)', '• ', text)
        
        # Fix common OCR letter confusions
        text = re.sub(r'(?<=\d)l(?=\d)', '1', text)  # Digit + l + digit → 1
        text = re.sub(r'(?<=\d)I(?=\d)', '1', text)  # Digit + I + digit → 1
        text = re.sub(r'(?<=\d)O(?=\d)', '0', text)  # Digit + O + digit → 0
        text = re.sub(r'(?<=\d)S(?=\d)', '5', text)  # Digit + S + digit → 5
        text = re.sub(r'(?<=\d)Z(?=\d)', '2', text)  # Digit + Z + digit → 2
        text = re.sub(r'(?<=\d)B(?=\d)', '8', text)  # Digit + B + digit → 8
        
        # Fix space issues
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # No space between word and capital letter
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)  # No space between letter and digit
        text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)  # No space between digit and letter
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.!?,:;])([A-Z0-9])', r'\1 \2', text)
        
        # Fix common merged words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Use context to fix common errors
        common_errors = {
            'tbe': 'the',
            'arid': 'and',
            'ofthe': 'of the',
            'forthe': 'for the',
            'tothe': 'to the',
            'inthe': 'in the',
            'fromthe': 'from the',
            'onthe': 'on the',
            'withthe': 'with the',
            'atthe': 'at the',
            'isthe': 'is the',
            'wasthe': 'was the',
            'asthe': 'as the',
            'bythe': 'by the',
            'thatthe': 'that the',
            'butthe': 'but the',
            'andthe': 'and the',
            'Tbis': 'This',
            'ca11': 'call',
            'cornpany': 'company',
            'frorn': 'from',
            'systern': 'system',
            'rnay': 'may',
            'Iine': 'line',
            'tirne': 'time',
            'Iist': 'list',
            'Iike': 'like',
            'sirnple': 'simple',
            'sarne': 'same',
            'frorntbe': 'from the'
        }
        
        for error, correction in common_errors.items():
            text = re.sub(r'\b' + error + r'\b', correction, text)
        
        # Fix newlines - remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphens at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Fix URLs and email addresses
        # Email pattern
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        for email in emails:
            # Common OCR errors in emails
            fixed_email = email.replace(' ', '').replace(',', '.').replace(';', '.')
            text = text.replace(email, fixed_email)
        
        # Fix URL pattern
        url_pattern = r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*\b'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            # Common OCR errors in URLs
            fixed_url = url.replace(' ', '').replace(',', '.').replace(';', '.')
            text = text.replace(url, fixed_url)
        
        return text
    
    def _enhance_text_organization(self, text: str, image_type: ImageType) -> str:
        """
        Enhance text organization based on image type and content
        
        Args:
            text: Text to organize
            image_type: Type of image
            
        Returns:
            Organized text
        """
        # Add appropriate structure based on image type
        if image_type == ImageType.DOCUMENT or image_type == ImageType.BOOK_PAGE:
            # Preserve paragraph structure
            text = self._organize_document_text(text)
        elif image_type == ImageType.FORM:
            # Organize form with proper field formatting
            text = self._organize_form_text(text)
        elif image_type == ImageType.RECEIPT:
            # Organize receipt with sections
            text = self._organize_receipt_text(text)
        elif image_type == ImageType.ID_CARD:
            # Organize ID card fields
            text = self._organize_id_card_text(text)
        elif image_type == ImageType.TABLE:
            # Organize table format
            text = self._organize_table_text(text)
        else:
            # Default organization
            text = self._default_text_organization(text)
        
        return text
    
    def _organize_document_text(self, text: str) -> str:
        """
        Organize document text with proper paragraph structure
        
        Args:
            text: Document text
            
        Returns:
            Organized text
        """
        # Split into lines while preserving paragraph structure
        lines = text.split('\n')
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            # Clean the line
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append("")  # Add empty line to preserve structure
            elif line.startswith('•') or line.startswith('-') or re.match(r'^\d+[\.\)]', line):
                # This is a list item - preserve as standalone line
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            elif re.match(r'^[A-Z][A-Z\s]+:?', line) or re.match(r'^[A-Z][A-Za-z\s]+:', line):
                # This is likely a heading or a form field label (like "NAME:")
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            elif len(line) < 40 and not line.endswith(('.', '?', '!')):
                # Short line that doesn't end with punctuation - might be a heading
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            else:
                # Regular text line - add to current paragraph
                # But check if it should start a new paragraph based on content
                if (current_paragraph and 
                    (line[0].isupper() or re.match(r'^[0-9]', line)) and
                    current_paragraph[-1].endswith(('.', '!', '?'))):
                    # This line starts with a capital letter or number and previous line 
                    # ended with punctuation - likely a new paragraph
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
        
        # Join formatted lines, preserving structure
        return '\n'.join(formatted_lines)
    
    def _organize_form_text(self, text: str) -> str:
        """
        Organize form text with field labels and values clearly separated
        
        Args:
            text: Form text
            
        Returns:
            Organized text
        """
        # Split into lines
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Try to identify form field patterns: "Label: Value" or "Label Value"
            label_value_match = re.match(r'^([A-Za-z\s]+):\s*(.+)', line)
            if label_value_match:
                label = label_value_match.group(1).strip()
                value = label_value_match.group(2).strip()
                
                # Format as "Label: Value"
                formatted_lines.append(f"{label}: {value}")
            else:
                # Try to match label without colon
                label_value_match = re.match(r'^([A-Za-z\s]+)\s{2,}(.+)', line)
                if label_value_match:
                    label = label_value_match.group(1).strip()
                    value = label_value_match.group(2).strip()
                    
                    # Format as "Label: Value"
                    formatted_lines.append(f"{label}: {value}")
                else:
                    # No special formatting needed
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _organize_receipt_text(self, text: str) -> str:
        """
        Organize receipt text with clear sections
        
        Args:
            text: Receipt text
            
        Returns:
            Organized text
        """
        # Split into lines
        lines = text.split('\n')
        formatted_lines = []
        
        # Extract header (store name, address, date)
        header_lines = []
        item_lines = []
        total_lines = []
        footer_lines = []
        
        # Track current section
        section = "header"
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Check for transition to items section
            if section == "header" and (
                re.match(r'^ITEM|^DESCRIPTION|^QTY|PRICE|^-+', line, re.IGNORECASE) or
                re.match(r'^={5,}', line)
            ):
                section = "items"
                continue
            
            # Check for transition to totals section
            if section == "items" and (
                re.match(r'^SUBTOTAL|^TAX|^TOTAL|^={5,}', line, re.IGNORECASE)
            ):
                section = "totals"
            
            # Check for transition to footer
            if section == "totals" and (
                re.match(r'^THANK|^RETURN|^EXCHANGE|^POLICY|^RECEIPT', line, re.IGNORECASE)
            ):
                section = "footer"
            
            # Add line to appropriate section
            if section == "header":
                header_lines.append(line)
            elif section == "items":
                item_lines.append(line)
            elif section == "totals":
                total_lines.append(line)
            elif section == "footer":
                footer_lines.append(line)
        
        # Format header - usually store name, address, date
        if header_lines:
            formatted_lines.extend(header_lines)
            formatted_lines.append("")  # Add separator
        
        # Format items section
        if item_lines:
            formatted_lines.append("ITEMS:")
            formatted_lines.extend(["  " + line for line in item_lines])
            formatted_lines.append("")  # Add separator
        
        # Format totals section
        if total_lines:
            formatted_lines.append("TOTALS:")
            formatted_lines.extend(total_lines)
            formatted_lines.append("")  # Add separator
        
        # Format footer section
        if footer_lines:
            formatted_lines.extend(footer_lines)
        
        return '\n'.join(formatted_lines)
    
    def _organize_id_card_text(self, text: str) -> str:
        """
        Organize ID card text with fields clearly labeled
        
        Args:
            text: ID card text
            
        Returns:
            Organized text
        """
        # Split into lines
        lines = text.split('\n')
        formatted_lines = []
        
        # Common ID card fields
        id_fields = [
            'NAME', 'ADDRESS', 'DATE OF BIRTH', 'DOB', 'EXPIRATION DATE', 'SEX', 'GENDER',
            'HEIGHT', 'WEIGHT', 'EYES', 'HAIR', 'DRIVER\'S LICENSE', 'ISSUE DATE',
            'PLACE OF BIRTH', 'NATIONALITY', 'RELIGION', 'MARITAL STATUS', 'BLOOD TYPE',
            'OCCUPATION', 'ID NUMBER', 'SIGNATURE'
        ]
        
        # Extract field values
        field_values = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match field: value pattern
            field_match = None
            for field in id_fields:
                pattern = f'^{field}\\s*:?\\s*(.+)'
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    field_match = (field, match.group(1).strip())
                    break
            
            if field_match:
                field, value = field_match
                field_values[field.upper()] = value
            else:
                # Check if line contains a field
                for field in id_fields:
                    if field.upper() in line.upper():
                        # Extract the value (text after the field)
                        parts = re.split(field, line, flags=re.IGNORECASE)
                        if len(parts) > 1 and parts[1].strip():
                            field_values[field.upper()] = parts[1].strip()
                            break
        
        # Format the output in a standard way
        # First add the name if available
        if 'NAME' in field_values:
            formatted_lines.append(f"NAME: {field_values['NAME']}")
        
        # Add ID number if available
        for field in ['ID NUMBER', 'DRIVER\'S LICENSE']:
            if field in field_values:
                formatted_lines.append(f"{field}: {field_values[field]}")
                break
        
        # Add other fields in a logical order
        field_order = [
            'ADDRESS', 'DATE OF BIRTH', 'DOB', 'PLACE OF BIRTH', 'SEX', 'GENDER',
            'HEIGHT', 'WEIGHT', 'EYES', 'HAIR', 'BLOOD TYPE', 'NATIONALITY',
            'RELIGION', 'MARITAL STATUS', 'OCCUPATION', 'ISSUE DATE',
            'EXPIRATION DATE'
        ]
        
        for field in field_order:
            if field in field_values:
                formatted_lines.append(f"{field}: {field_values[field]}")
        
        # Add any remaining fields
        for field, value in field_values.items():
            if field not in ['NAME', 'ID NUMBER', 'DRIVER\'S LICENSE'] + field_order:
                formatted_lines.append(f"{field}: {value}")
        
        return '\n'.join(formatted_lines)
    
    def _organize_table_text(self, text: str) -> str:
        """
        Organize text from table structure
        
        Args:
            text: Table text
            
        Returns:
            Organized text in table format
        """
        lines = text.split('\n')
        formatted_lines = []
        
        # Handle pipe-delimited tables
        if any('|' in line for line in lines):
            # Already has pipe delimiters, clean up format
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    formatted_lines.append("")
                    continue
                
                # Normalize pipe delimiters (ensure spaces around them)
                line = re.sub(r'\s*\|\s*', ' | ', line.strip())
                
                # Add leading/trailing pipes if missing
                if not line.startswith('|'):
                    line = '| ' + line
                if not line.endswith('|'):
                    line = line + ' |'
                
                formatted_lines.append(line)
                
                # Add separator after header if not present
                if i == 0 and len(lines) > 1:
                    # Check if the next line isn't already a separator
                    if not lines[1].strip().startswith('--') and not lines[1].strip().startswith('=='):
                        # Create separator matching first row format
                        columns = line.count('|') - 1
                        separator = '|' + '|'.join([' --- ' for _ in range(columns)]) + '|'
                        formatted_lines.append(separator)
            
            return '\n'.join(formatted_lines)
        
        # For non-delimited tables, try to identify columns through spacing
        # Find consistent space patterns that might indicate columns
        if len(lines) > 2:
            # Detect potential column boundaries through whitespace patterns
            whitespace_cols = []
            for line in lines[:5]:  # Use first few lines for pattern detection
                if not line.strip():
                    continue
                
                # Find columns of whitespace
                prev_char = ''
                col_start = -1
                for i, char in enumerate(line):
                    if char.isspace() and prev_char not in string.whitespace:
                        col_start = i
                    elif not char.isspace() and prev_char in string.whitespace and col_start >= 0:
                        # End of whitespace column
                        if i - col_start >= 2:  # Minimum 2 spaces for column
                            whitespace_cols.append((col_start, i))
                        col_start = -1
                    prev_char = char
            
            # If we have consistent whitespace columns, convert to table format
            if whitespace_cols:
                # Group nearby column boundaries
                boundaries = []
                for start, end in sorted(whitespace_cols, key=lambda x: x[0]):
                    if not boundaries or start > boundaries[-1] + 3:
                        boundaries.append(start)
                
                # If we have boundaries, format as a table
                if len(boundaries) >= 1:
                    for line in lines:
                        if not line.strip():
                            formatted_lines.append("")
                            continue
                        
                        # Insert pipe delimiters at boundaries
                        new_line = "| "
                        last_pos = 0
                        
                        for boundary in boundaries:
                            if boundary < len(line):
                                new_line += line[last_pos:boundary].strip() + " | "
                                last_pos = boundary
                        
                        # Add remaining text
                        if last_pos < len(line):
                            new_line += line[last_pos:].strip() + " |"
                        
                        formatted_lines.append(new_line)
                    
                    # Add separator after header
                    if len(formatted_lines) > 0:
                        columns = formatted_lines[0].count('|') - 1
                        separator = '|' + '|'.join([' --- ' for _ in range(columns)]) + '|'
                        formatted_lines.insert(1, separator)
                    
                    return '\n'.join(formatted_lines)
        
        # If table structure not detected, just return cleaned text
        return '\n'.join([line.strip() for line in lines])
    
    def _default_text_organization(self, text: str) -> str:
        """
        Default text organization for general text
        
        Args:
            text: General text
            
        Returns:
            Organized text
        """
        # Split text into lines
        lines = text.split('\n')
        
        # Remove excess empty lines (more than 2 consecutive)
        formatted_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if not prev_empty:
                    formatted_lines.append("")
                    prev_empty = True
            else:
                formatted_lines.append(line)
                prev_empty = False
        
        # Join lines
        return '\n'.join(formatted_lines)
    
    def _enhanced_text_formatting(self, text: str, layout_info: Dict) -> str:
        """
        Format extracted text with enhanced layout information
        
        Args:
            text: Raw OCR text
            layout_info: Layout information
            
        Returns:
            Formatted text
        """
        if not text:
            return ""
        
        # Use layout information to format text appropriately
        document_structure = None
        if layout_info:
            # Determine document structure from layout info
            if layout_info.get("has_table", False):
                document_structure = DocumentStructure.TABLE
            elif layout_info.get("has_form", False):
                document_structure = DocumentStructure.FORM
            elif layout_info.get("columns", 1) > 1:
                document_structure = DocumentStructure.MULTI_COLUMN
            elif layout_info.get("document_type") == "scientific":
                document_structure = DocumentStructure.SCIENTIFIC
            elif "paragraphs" in layout_info and len(layout_info["paragraphs"]) > 1:
                document_structure = DocumentStructure.PARAGRAPHS
            else:
                # Determine structure from text content
                document_structure = self._detect_document_structure(text)
        else:
            # Determine structure from text content
            document_structure = self._detect_document_structure(text)
        
        # Format based on document structure
        if document_structure == DocumentStructure.PLAIN_TEXT:
            formatted_text = self._format_plain_text(text)
        elif document_structure == DocumentStructure.PARAGRAPHS:
            formatted_text = self._format_paragraphs(text, layout_info)
        elif document_structure == DocumentStructure.HEADERS_AND_CONTENT:
            formatted_text = self._format_headers_and_content(text, layout_info)
        elif document_structure == DocumentStructure.BULLET_POINTS:
            formatted_text = self._format_bullet_points(text)
        elif document_structure == DocumentStructure.TABLE:
            formatted_text = self._format_table(text, layout_info)
        elif document_structure == DocumentStructure.FORM:
            formatted_text = self._format_form(text, layout_info)
        elif document_structure == DocumentStructure.MULTI_COLUMN:
            formatted_text = self._format_multi_column(text, layout_info)
        elif document_structure == DocumentStructure.SCIENTIFIC:
            formatted_text = self._format_scientific(text)
        else:  # MIXED or other
            formatted_text = self._default_formatting(text)
        
        # Remove unwanted characters but preserve useful ones
        formatted_text = re.sub(r'[^\w\s.!?,;:()"\'•\-\n]', '', formatted_text)
        
        return formatted_text.strip()
    
    def _detect_document_structure(self, text: str) -> DocumentStructure:
        """
        Detect the document structure based on text content
        
        Args:
            text: Text to analyze
            
        Returns:
            Document structure enum
        """
        # Count various features
        bullet_count = len(re.findall(r'(?:^|\n)[•\-*+]', text))
        numbered_list_count = len(re.findall(r'(?:^|\n)\d+[\.\)]', text))
        table_row_count = len(re.findall(r'(?:^|\n)[\w\s]+\|[\w\s]+\|', text))
        form_field_count = len(re.findall(r'(?:^|\n)[\w\s]+:', text))
        header_count = len(re.findall(r'(?:^|\n)[A-Z][A-Z\s]+(?:\n|$)', text))
        paragraph_count = len(re.findall(r'\n\s*\n', text))
        formula_count = len(re.findall(r'[=+\-*/^]|sqrt|sin|cos|tan|log', text))
        
        # Check for multi-column layout (hard to detect from text alone)
        lines = text.split('\n')
        if len(lines) > 10:
            # Check for lines that are unusually short and staggered
            short_line_count = 0
            for line in lines:
                if 5 < len(line.strip()) < 40:
                    short_line_count += 1
            
            if short_line_count > len(lines) * 0.6:
                return DocumentStructure.MULTI_COLUMN
        
        # Determine dominant structure
        if table_row_count > 5:
            return DocumentStructure.TABLE
        elif bullet_count + numbered_list_count > 5:
            return DocumentStructure.BULLET_POINTS
        elif form_field_count > 5:
            return DocumentStructure.FORM
        elif header_count > 2 and paragraph_count > 1:
            return DocumentStructure.HEADERS_AND_CONTENT
        elif paragraph_count > 1:
            return DocumentStructure.PARAGRAPHS
        elif formula_count > 3:
            return DocumentStructure.SCIENTIFIC
        elif len(text.strip()) < 100:
            return DocumentStructure.PLAIN_TEXT
        else:
            return DocumentStructure.MIXED
    
    def _format_plain_text(self, text: str) -> str:
        """
        Format plain text with minimal processing
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text
        """
        # Just clean up spaces and newlines
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(line for line in lines if line)
    
    def _format_paragraphs(self, text: str, layout_info: Dict) -> str:
        """
        Format text with paragraph structure
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted text with paragraphs
        """
        # Use paragraph information from layout if available
        if layout_info and "paragraphs" in layout_info:
            # Sort paragraphs by vertical position
            paragraphs = sorted(layout_info["paragraphs"], key=lambda p: p["bbox"][1])
            
            # Join paragraphs with proper spacing
            return "\n\n".join(p["text"] for p in paragraphs)
        
        # Otherwise, use text-based paragraph detection
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                # Add to current paragraph
                current_paragraph.append(line)
        
        # Add final paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double newlines
        return '\n\n'.join(paragraphs)
    
    def _format_headers_and_content(self, text: str, layout_info: Dict) -> str:
        """
        Format text with headers and content
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted text with headers and content
        """
        lines = text.split('\n')
        formatted_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                formatted_lines.append("")
                i += 1
                continue
            
            # Check if this line might be a header
            is_header = False
            if re.match(r'^[A-Z][A-Z\s]+', line) or re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}', line):
                # All caps or Title Case with few words - likely a header
                is_header = True
            elif i < len(lines) - 1 and not lines[i+1].strip():
                # Followed by an empty line - could be a header
                is_header = True
            
            if is_header:
                # Add header with proper formatting
                formatted_lines.append("")  # Spacing before header
                formatted_lines.append(line)
                formatted_lines.append("")  # Spacing after header
                i += 1
                
                # Collect content under this header
                content_lines = []
                while i < len(lines) and (not lines[i].strip() or 
                                        not re.match(r'^[A-Z][A-Z\s]+', lines[i].strip())):
                    if lines[i].strip():
                        content_lines.append(lines[i].strip())
                    i += 1
                
                # Format the content as paragraphs
                if content_lines:
                    current_paragraph = []
                    for content_line in content_lines:
                        if not content_line:
                            if current_paragraph:
                                formatted_lines.append(' '.join(current_paragraph))
                                current_paragraph = []
                        else:
                            current_paragraph.append(content_line)
                    
                    if current_paragraph:
                        formatted_lines.append(' '.join(current_paragraph))
                
                # Don't increment i since the while loop already moved to the next header
            else:
                # Regular line
                formatted_lines.append(line)
                i += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_bullet_points(self, text: str) -> str:
        """
        Format text with bullet points
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text with bullet points
        """
        lines = text.split('\n')
        formatted_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                formatted_lines.append("")
                i += 1
                continue
            
            # Check for bullet point or numbered list
            bullet_match = re.match(r'^([•\-*+]|\d+[\.\)])(.+)', line)
            
            if bullet_match:
                # Get bullet/number and content
                bullet = bullet_match.group(1)
                content = bullet_match.group(2).strip()
                
                # Standardize bullets
                if bullet not in ['•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
                    bullet = '•'
                
                # Add formatted bullet point
                formatted_lines.append(f"{bullet} {content}")
                
                # Look for continuation lines (indented)
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r'^([•\-*+]|\d+[\.\)])', lines[i].strip()):
                    formatted_lines.append(f"  {lines[i].strip()}")
                    i += 1
            else:
                # Regular line
                formatted_lines.append(line)
                i += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_table(self, text: str, layout_info: Dict) -> str:
        """
        Format text as a table
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted table text
        """
        lines = text.split('\n')
        table_lines = []
        
        # Find table rows
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if this might be a table row
            # Tables typically have consistent column alignment or delimiters
            
            # Check for explicit delimiters
            if '|' in line or '\t' in line:
                table_lines.append(line)
                i += 1
                continue
            
            # Check for space-aligned columns (multiple spaces between columns)
            if re.search(r'\S+\s{2,}\S+', line):
                table_lines.append(line)
                i += 1
                continue
            
            # Not a table row
            i += 1
        
        # If no table content found, return original text
        if not table_lines:
            return text
        
        # Format the table
        formatted_table = []
        delimiter = '|' if any('|' in line for line in table_lines) else None
        
        # Try to determine column boundaries for space-delimited tables
        column_boundaries = []
        if not delimiter:
            # Find potential column boundaries from space patterns
            for line in table_lines[:min(5, len(table_lines))]:
                positions = [m.start() for m in re.finditer(r'\s{2,}', line)]
                if positions:
                    column_boundaries.append(positions)
            
            # Find common boundaries
            if column_boundaries:
                # Group close positions
                all_positions = [pos for positions in column_boundaries for pos in positions]
                all_positions.sort()
                
                common_boundaries = []
                current_group = [all_positions[0]]
                
                for pos in all_positions[1:]:
                    if pos - current_group[-1] < 3:
                        current_group.append(pos)
                    else:
                        # Calculate average position for this group
                        common_boundaries.append(sum(current_group) // len(current_group))
                        current_group = [pos]
                
                if current_group:
                    common_boundaries.append(sum(current_group) // len(current_group))
                
                # Use common boundaries to format table
                for line in table_lines:
                    formatted_line = line
                    for boundary in reversed(common_boundaries):
                        if boundary < len(line):
                            formatted_line = formatted_line[:boundary] + ' | ' + formatted_line[boundary:].lstrip()
                    formatted_table.append(formatted_line)
            else:
                # Just use the original table lines
                formatted_table = table_lines
        else:
            # For explicitly delimited tables, just clean up spacing
            for line in table_lines:
                parts = [part.strip() for part in line.split('|')]
                formatted_table.append(' | '.join(parts))
        
        # Add a separator after the header row
        if len(formatted_table) > 1:
            header = formatted_table[0]
            separator = ''
            
            # Create a separator matching the structure of the header
            if '|' in header:
                parts = header.split('|')
                separator = '|'.join('-' * len(part.strip()) for part in parts)
            else:
                separator = '-' * len(header)
            
            formatted_table.insert(1, separator)
        
        # Join the formatted table
        return '\n'.join(formatted_table)
    
    def _format_form(self, text: str, layout_info: Dict) -> str:
        """
        Format text as a form
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted form text
        """
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Check for form field pattern: Label: Value
            field_match = re.match(r'^([A-Za-z\s]+):\s*(.+)', line)
            
            if field_match:
                # Already in the right format
                formatted_lines.append(line)
            else:
                # Check for field pattern without colon
                field_match = re.match(r'^([A-Za-z\s]+)\s{2,}(.+)', line)
                
                if field_match:
                    label = field_match.group(1).strip()
                    value = field_match.group(2).strip()
                    formatted_lines.append(f"{label}: {value}")
                else:
                    # Regular line
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_multi_column(self, text: str, layout_info: Dict) -> str:
        """
        Format text from multi-column layout
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted multi-column text
        """
        # This is challenging from just OCR text without positional information
        # Try to detect column breaks from text patterns
        
        lines = text.split('\n')
        
        # Check if we have regions from layout info
        if layout_info and "regions" in layout_info:
            # Sort regions by column then by vertical position
            columns = {}
            
            for region in layout_info["regions"]:
                # Get region center x-coordinate
                center_x = region["x"] + region["width"] // 2
                
                # Determine which column this belongs to
                column_idx = center_x // 200  # Simple column bucketing
                
                if column_idx not in columns:
                    columns[column_idx] = []
                
                columns[column_idx].append(region)
            
            # Sort regions in each column by y-coordinate
            for column_idx in columns:
                columns[column_idx].sort(key=lambda r: r["y"])
            
            # Reconstruct text by column
            formatted_text = ""
            
            # Process columns from left to right
            for column_idx in sorted(columns.keys()):
                column_text = "\n\n".join([r.get("text", "") for r in columns[column_idx]])
                
                if formatted_text:
                    formatted_text += "\n\n--- Next Column ---\n\n"
                
                formatted_text += column_text
            
            return formatted_text
        
        # Without layout info, try to detect column breaks heuristically
        # This is challenging and error-prone without positional info
        
        # Look for very short lines that may indicate column wrapping
        short_line_threshold = 30
        short_lines = [i for i, line in enumerate(lines) if 0 < len(line.strip()) < short_line_threshold]
        
        # If more than half the lines are short, likely multi-column
        if len(short_lines) > len(lines) * 0.5:
            # Try to recognize column boundaries by line patterns
            # This is a simplistic approach and may not work well
            formatted_lines = []
            formatted_lines.append("NOTE: This text appears to be in multiple columns. " +
                                 "The content below has been reformatted as a single column.")
            formatted_lines.append("")
            
            current_paragraph = []
            for line in lines:
                line = line.strip()
                
                if not line:
                    if current_paragraph:
                        formatted_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                    formatted_lines.append("")
                else:
                    # If line starts with capital letter and previous line was short,
                    # it might be a new paragraph or column break
                    if (current_paragraph and 
                        line[0].isupper() and 
                        len(current_paragraph[-1]) < short_line_threshold):
                        
                        # Check if this might be a continuation by checking for sentence ending
                        if current_paragraph[-1].endswith(('.', '!', '?', ':', ';')):
                            # Likely a new paragraph
                            formatted_lines.append(' '.join(current_paragraph))
                            current_paragraph = [line]
                        else:
                            # Might be continuation or column break
                            # Try to detect by checking for semantic connection
                            if len(current_paragraph[-1].split()) < 4:
                                # Very short line, likely a column break
                                formatted_lines.append(' '.join(current_paragraph))
                                current_paragraph = [line]
                            else:
                                # Probably continuation
                                current_paragraph.append(line)
                    else:
                        current_paragraph.append(line)
            
            # Add final paragraph
            if current_paragraph:
                formatted_lines.append(' '.join(current_paragraph))
            
            return '\n'.join(formatted_lines)
        
        # If not detected as multi-column, format as paragraphs
        return self._format_paragraphs(text, layout_info)
    
    def _format_scientific(self, text: str) -> str:
        """
        Format scientific text with formulas
        
        Args:
            text: Scientific text
            
        Returns:
            Formatted scientific text
        """
        lines = text.split('\n')
        formatted_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                formatted_lines.append("")
                i += 1
                continue
            
            # Check if this line might contain a formula
            formula_indicators = ['=', '+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos', 'tan', 'log']
            is_formula = any(indicator in line for indicator in formula_indicators)
            
            if is_formula:
                # Add formula with proper spacing
                formatted_lines.append("")  # Space before formula
                formatted_lines.append(line)
                formatted_lines.append("")  # Space after formula
            else:
                # Regular text - format as paragraphs
                if i > 0 and formatted_lines and formatted_lines[-1] and not line.startswith(' '):
                    # Continue paragraph
                    formatted_lines[-1] += ' ' + line
                else:
                    # New paragraph or indented line
                    formatted_lines.append(line)
            
            i += 1
        
        return '\n'.join(formatted_lines)
    
    def _default_formatting(self, text: str) -> str:
        """
        Default text formatting for mixed content
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text
        """
        # Split into lines while preserving paragraph structure
        lines = text.split('\n')
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            # Clean the line
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append("")  # Add empty line to preserve structure
            else:
                # Add to current paragraph
                current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
        
        # Join formatted lines
        return '\n'.join(formatted_lines)
    
    def _enhanced_language_detection(self, text: str) -> str:
        """
        Enhanced language detection with better accuracy using rules
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        if not text or len(text) < 20:
            return "unknown"
        
        # Enhanced keywords for language detection
        # Add more common words for better accuracy
        id_keywords = [
            'yang', 'dengan', 'dan', 'untuk', 'dari', 'pada', 'adalah', 'ini', 'itu',
            'dalam', 'tidak', 'akan', 'saya', 'kamu', 'kami', 'mereka', 'bisa', 'oleh',
            'jika', 'telah', 'sudah', 'harus', 'dapat', 'karena', 'kepada', 'maka',
            'tentang', 'setiap', 'seperti', 'juga', 'ada', 'sebuah', 'tersebut',
            'anda', 'sangat', 'kemudian', 'saat', 'selama', 'masih', 'lebih',
            'belum', 'ketika', 'kita', 'baru', 'perlu'
        ]
        
        en_keywords = [
            'the', 'is', 'are', 'and', 'for', 'that', 'have', 'with', 'this', 'from',
            'they', 'will', 'would', 'there', 'their', 'what', 'about', 'which',
            'when', 'one', 'all', 'been', 'but', 'not', 'you', 'your', 'who',
            'more', 'has', 'was', 'were', 'can', 'said', 'out', 'use', 'into',
            'some', 'than', 'other', 'time', 'now', 'only', 'like', 'just'
        ]
        
        # Count keyword occurrences
        text_lower = ' ' + text.lower() + ' '
        id_count = sum(1 for word in id_keywords if f' {word} ' in text_lower)
        en_count = sum(1 for word in en_keywords if f' {word} ' in text_lower)
        
        # Calculate weighted scores
        id_score = id_count / len(id_keywords)
        en_score = en_count / len(en_keywords)
        
        # Add text pattern analysis for better detection
        # Indonesian patterns
        id_patterns = [r'\bakan\s+\w+\b', r'\bsedang\s+\w+\b', r'\btelah\s+\w+\b']
        id_pattern_count = sum(1 for pattern in id_patterns if re.search(pattern, text_lower))
        id_score += id_pattern_count * 0.1
        
        # English patterns
        en_patterns = [r'\bwill\s+\w+\b', r'\bhave\s+\w+\b', r'\bhas\s+\w+\b']
        en_pattern_count = sum(1 for pattern in en_patterns if re.search(pattern, text_lower))
        en_score += en_pattern_count * 0.1
        
        # Make decision based on threshold
        if id_score > 0.15 and id_score > en_score:
            return 'id'
        elif en_score > 0.15:
            return 'en'
        else:
            # Check for other language indicators
            # These are simplified character set checks
            
            # Check for Latin script dominance
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            total_chars = sum(1 for c in text if c.isalpha())
            
            if total_chars > 0:
                latin_ratio = latin_chars / total_chars
                
                if latin_ratio > 0.9:
                    # Probably a Latin script language other than English or Indonesian
                    return 'latin'
                elif latin_ratio < 0.3:
                    # Probably a non-Latin script language
                    return 'non-latin'
            
            return 'unknown'
    
    def _extract_rule_based_structured_info(self, text: str, 
                                           image_type: ImageType = None) -> Optional[Dict[str, Any]]:
        """
        Extract structured information using rule-based methods
        
        Args:
            text: Input text
            image_type: Optional image type for specialized extraction
            
        Returns:
            Dictionary of extracted fields or None
        """
        if not text:
            return None
        
        # Choose the appropriate extraction strategy based on image type
        if image_type == ImageType.ID_CARD:
            return self._extract_id_card_info(text)
        elif image_type == ImageType.RECEIPT:
            return self._extract_receipt_info(text)
        elif image_type == ImageType.FORM:
            return self._extract_form_info(text)
        elif image_type == ImageType.TABLE:
            return self._extract_table_info(text)
        
        # Default to generic form extraction
        return self._extract_generic_info(text)
    
    def _extract_id_card_info(self, text: str) -> Dict[str, str]:
        """
        Extract information from ID card text
        
        Args:
            text: ID card text
            
        Returns:
            Dictionary of extracted fields
        """
        # Common ID card fields with regex patterns
        field_patterns = {
            'name': r'(?:name|nama)[\s:]+([^\n]+)',
            'date_of_birth': r'(?:date of birth|birth date|birthdate|dob|tanggal lahir)[\s:]+([^\n]+)',
            'gender': r'(?:gender|sex|jenis kelamin)[\s:]+([^\n]+)',
            'address': r'(?:address|alamat)[\s:]+([^\n]+)',
            'id_number': r'(?:id|no|number|nomor)[\s:]+([A-Z0-9\-\s]+)',
            'expiration_date': r'(?:expiration|expiry|exp|berlaku sampai)[\s:]+([^\n]+)',
            'issue_date': r'(?:issue|issued|date of issue|tanggal dikeluarkan)[\s:]+([^\n]+)',
            'nationality': r'(?:nationality|negara|warga negara|citizenship)[\s:]+([^\n]+)',
            'place_of_birth': r'(?:place of birth|birthplace|tempat lahir)[\s:]+([^\n]+)',
            'blood_type': r'(?:blood|blood type|golongan darah)[\s:]+([^\n]+)',
            'marital_status': r'(?:marital status|status perkawinan)[\s:]+([^\n]+)',
            'occupation': r'(?:occupation|job|pekerjaan)[\s:]+([^\n]+)',
            'religion': r'(?:religion|agama)[\s:]+([^\n]+)'
        }
        
        # Extract fields from text
        extracted_info = {}
        text_lower = text.lower()
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Only add if not empty
                if value:
                    extracted_info[field] = value
        
        # Also try a different approach - look for field labels followed by value
        field_labels = {
            'name': ['name', 'nama'],
            'date_of_birth': ['date of birth', 'birth date', 'birthdate', 'dob', 'tanggal lahir'],
            'gender': ['gender', 'sex', 'jenis kelamin'],
            'address': ['address', 'alamat'],
            'id_number': ['id', 'no', 'number', 'nomor', 'nomor kartu'],
            'expiration_date': ['expiration', 'expiry', 'exp', 'berlaku sampai'],
            'issue_date': ['issue', 'issued', 'date of issue', 'tanggal dikeluarkan'],
            'nationality': ['nationality', 'negara', 'warga negara', 'citizenship'],
            'place_of_birth': ['place of birth', 'birthplace', 'tempat lahir'],
            'blood_type': ['blood', 'blood type', 'golongan darah'],
            'marital_status': ['marital status', 'status perkawinan'],
            'occupation': ['occupation', 'job', 'pekerjaan'],
            'religion': ['religion', 'agama']
        }
        
        for field, labels in field_labels.items():
            if field in extracted_info:
                continue  # Already extracted
                
            for label in labels:
                # Try label followed by colon
                pattern = f"\\b{re.escape(label)}\\s*:\\s*([^\\n]+)"
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        extracted_info[field] = value
                        break
                        
                # Try label at the beginning of a line
                pattern = f"^\\s*{re.escape(label)}\\s+([^\\n]+)"
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        extracted_info[field] = value
                        break
        
        return extracted_info
    
    def _extract_receipt_info(self, text: str) -> Dict[str, Any]:
        """
        Extract information from receipt text
        
        Args:
            text: Receipt text
            
        Returns:
            Dictionary of extracted fields
        """
        receipt_info = {
            'items': []
        }
        
        # Extract merchant/store name - usually found at the top
        lines = text.split('\n')
        if lines and lines[0].strip():
            receipt_info['merchant'] = lines[0].strip()
        
        # Extract date
        date_match = re.search(r'(?:date|tanggal)[\s:]+([0-9/\-\.]+)', text.lower())
        if date_match:
            receipt_info['date'] = date_match.group(1).strip()
        else:
            # Try simple date pattern
            date_pattern = r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b'
            date_match = re.search(date_pattern, text)
            if date_match:
                receipt_info['date'] = date_match.group(1)
        
        # Extract time
        time_match = re.search(r'(?:time|waktu)[\s:]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)', text.lower())
        if time_match:
            receipt_info['time'] = time_match.group(1).strip()
        else:
            # Try simple time pattern
            time_pattern = r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b'
            time_match = re.search(time_pattern, text)
            if time_match:
                receipt_info['time'] = time_match.group(1)
        
        # Extract subtotal
        subtotal_match = re.search(r'(?:subtotal|sub[\s-]?total)[\s:]+\$?([0-9\.,]+)', text.lower())
        if subtotal_match:
            receipt_info['subtotal'] = subtotal_match.group(1).strip()
        
        # Extract tax
        tax_match = re.search(r'(?:tax|vat|pajak)[\s:]+\$?([0-9\.,]+)', text.lower())
        if tax_match:
            receipt_info['tax'] = tax_match.group(1).strip()
        
        # Extract total
        total_match = re.search(r'(?:total|amount|jumlah)[\s:]+\$?([0-9\.,]+)', text.lower())
        if total_match:
            receipt_info['total'] = total_match.group(1).strip()
        
        # Extract payment method
        payment_methods = ['cash', 'card', 'credit', 'debit', 'visa', 'mastercard', 'amex', 'american express',
                        'discover', 'tunai', 'kartu', 'kredit', 'debit']
        for method in payment_methods:
            if method.lower() in text.lower():
                receipt_info['payment_method'] = method
                break
        
        # Extract items - this is more complex
        # Look for the items section
        items_section = None
        in_items = False
        items_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check for the start of the items section
            if re.match(r'^(?:items?|description|qty|quantity|item price)', line, re.IGNORECASE):
                in_items = True
                continue
            
            # Check for the end of the items section
            if in_items and re.match(r'^(?:subtotal|total|tax|amount)', line, re.IGNORECASE):
                in_items = False
                continue
            
            # Collect item lines
            if in_items and line:
                items_lines.append(line)
        
        # Parse item lines
        for line in items_lines:
            # Try to match item patterns:
            # 1. ItemName Quantity Price
            # 2. ItemName Price
            # 3. Quantity x ItemName Price
            
            # Pattern 1: ItemName Quantity Price
            match = re.match(r'(.+?)\s+(\d+)\s+\$?([0-9\.,]+)', line)
            if match:
                item_name = match.group(1).strip()
                quantity = match.group(2)
                price = match.group(3)
                receipt_info['items'].append({
                    'name': item_name,
                    'quantity': quantity,
                    'price': price
                })
                continue
            
            # Pattern 2: ItemName Price
            match = re.match(r'(.+?)\s+\$?([0-9\.,]+)', line)
            if match:
                item_name = match.group(1).strip()
                price = match.group(2)
                receipt_info['items'].append({
                    'name': item_name,
                    'quantity': '1',
                    'price': price
                })
                continue
            
            # Pattern 3: Quantity x ItemName Price
            match = re.match(r'(\d+)(?:\s*[xX]\s*)(.+?)\s+\$?([0-9\.,]+)', line)
            if match:
                quantity = match.group(1)
                item_name = match.group(2).strip()
                price = match.group(3)
                receipt_info['items'].append({
                    'name': item_name,
                    'quantity': quantity,
                    'price': price
                })
                continue
            
            # If no pattern matches, just add as item name
            if line:
                receipt_info['items'].append({
                    'name': line,
                    'quantity': '1',
                    'price': '0.00'
                })
        
        return receipt_info
    
    def _extract_form_info(self, text: str) -> Dict[str, str]:
        """
        Extract information from form text
        
        Args:
            text: Form text
            
        Returns:
            Dictionary of extracted fields
        """
        # Look for field-value pairs using regex
        # Common pattern: Field: Value or Field - Value
        field_value_pattern = r'([A-Za-z\s]+[A-Za-z])[\s:]+(.+)'
        
        # Extract all field-value pairs
        form_info = {}
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(field_value_pattern, line)
            if match:
                field = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                
                # Only add if value is meaningful
                if value and not re.match(r'^[:\-,.;]*', value):
                    form_info[field] = value
        
        # Try special patterns for common form fields if they weren't found
        field_patterns = {
            'name': r'(?:name|nama)[\s:]+([^\n]+)',
            'email': r'(?:email|e-mail)[\s:]+([^\n]+)',
            'phone': r'(?:phone|telephone|tel|hp|handphone)[\s:]+([^\n]+)',
            'address': r'(?:address|alamat)[\s:]+([^\n]+)',
            'date': r'(?:date|tanggal)[\s:]+([^\n]+)',
            'company': r'(?:company|perusahaan)[\s:]+([^\n]+)',
            'department': r'(?:department|departemen)[\s:]+([^\n]+)'
        }
        
        for field, pattern in field_patterns.items():
            if field not in form_info:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        form_info[field] = value
        
        return form_info
    
    def _extract_table_info(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract information from table text
        
        Args:
            text: Table text
            
        Returns:
            Dictionary with extracted table data
        """
        # Split into lines
        lines = text.split('\n')
        
        # Check if this is a pipe-delimited table
        if any('|' in line for line in lines):
            return self._extract_delimited_table(lines, '|')
        
        # Check if this is a tab-delimited table
        if any('\t' in line for line in lines):
            return self._extract_delimited_table(lines, '\t')
        
        # If no delimiters, try to detect columns by whitespace
        return self._extract_space_delimited_table(lines)
    
    def _extract_delimited_table(self, lines: List[str], delimiter: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract data from a delimiter-separated table
        
        Args:
            lines: Text lines
            delimiter: Column delimiter
            
        Returns:
            Dictionary with extracted table data
        """
        table_data = {
            'headers': [],
            'rows': []
        }
        
        # Remove empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return table_data
        
        # Extract headers from the first row
        headers_line = non_empty_lines[0]
        headers = [h.strip() for h in headers_line.split(delimiter)]
        # Remove empty headers
        headers = [h for h in headers if h]
        
        if not headers:
            return table_data
        
        table_data['headers'] = headers
        
        # Start from the second row (skip header and separator rows)
        data_start = 1
        while data_start < len(non_empty_lines):
            if all(c == '-' or c == '=' or c.isspace() for c in non_empty_lines[data_start]):
                # This is a separator row
                data_start += 1
            else:
                break
        
        # Process data rows
        for i in range(data_start, len(non_empty_lines)):
            row = non_empty_lines[i]
            
            # Skip separator rows
            if all(c == '-' or c == '=' or c.isspace() for c in row):
                continue
                
            # Split row by delimiter
            values = [v.strip() for v in row.split(delimiter)]
            
            # Create row data with header mapping
            row_data = {}
            for j, value in enumerate(values):
                if j < len(headers):
                    row_data[headers[j]] = value
            
            if row_data:
                table_data['rows'].append(row_data)
        
        return table_data
    
    def _extract_space_delimited_table(self, lines: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract data from a space-delimited table (with column alignment)
        
        Args:
            lines: Text lines
            
        Returns:
            Dictionary with extracted table data
        """
        table_data = {
            'headers': [],
            'rows': []
        }
        
        # Remove empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return table_data
        
        # Try to detect column boundaries from spacing patterns
        # We'll look at word positions in the first few rows
        column_starts = []
        max_analysis_rows = min(5, len(non_empty_lines))
        
        for i in range(max_analysis_rows):
            # Find start positions of words in this line
            positions = [m.start() for m in re.finditer(r'\S+', non_empty_lines[i])]
            
            if i == 0:
                # First row - initialize with positions
                column_starts = positions
            else:
                # Merge with existing positions - keep positions that are close to existing ones
                merged_positions = []
                for pos in positions:
                    # Find closest existing column start
                    closest = min(column_starts, key=lambda x: abs(x - pos))
                    
                    # If close enough, adjust existing; otherwise add new
                    if abs(closest - pos) < 5:
                        # Update the existing position as average
                        idx = column_starts.index(closest)
                        column_starts[idx] = (column_starts[idx] + pos) // 2
                    else:
                        merged_positions.append(pos)
                
                # Add new positions
                column_starts.extend(merged_positions)
                column_starts.sort()
        
        if not column_starts:
            return table_data
        
        # Extract headers from first row
        header_line = non_empty_lines[0]
        headers = []
        
        for i in range(len(column_starts)):
            start = column_starts[i]
            end = column_starts[i+1] if i < len(column_starts) - 1 else len(header_line)
            header = header_line[start:end].strip()
            if header:
                headers.append(header)
        
        if not headers:
            return table_data
        
        table_data['headers'] = headers
        
        # Skip header and separator rows
        data_start = 1
        while data_start < len(non_empty_lines):
            if all(c == '-' or c == '=' or c.isspace() for c in non_empty_lines[data_start]):
                # This is a separator row
                data_start += 1
            else:
                break
        
        # Process data rows
        for i in range(data_start, len(non_empty_lines)):
            row = non_empty_lines[i]
            
            # Skip separator rows
            if all(c == '-' or c == '=' or c.isspace() for c in row):
                continue
            
            # Extract values based on column positions
            values = []
            for j in range(len(column_starts)):
                start = column_starts[j]
                end = column_starts[j+1] if j < len(column_starts) - 1 else len(row)
                
                if start < len(row):
                    value = row[start:end].strip()
                    values.append(value)
                else:
                    values.append("")
            
            # Create row data with header mapping
            row_data = {}
            for j, value in enumerate(values):
                if j < len(headers):
                    row_data[headers[j]] = value
            
            if row_data:
                table_data['rows'].append(row_data)
        
        return table_data
    
    def _extract_generic_info(self, text: str) -> Dict[str, str]:
        """
        Extract generic key-value information from any text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted key-value pairs
        """
        # Look for field-value pairs using regex
        # Common pattern: Field: Value or Field - Value
        field_value_pattern = r'([A-Za-z][A-Za-z\s]{2,20})[\s:]+([^\n:]{2,100})'
        
        # Extract all field-value pairs
        info = {}
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            matches = re.finditer(field_value_pattern, line)
            for match in matches:
                field = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                
                # Only add if value is meaningful
                if value and not re.match(r'^[:\-,.;]*', value):
                    info[field] = value
        
        return info
    
    def _generate_enhanced_extractive_summary(self, text: str, max_length: int = 200, style: str = "concise") -> str:
        """
        Generate an enhanced extractive summary without relying on AI models
        
        Args:
            text: Input text
            max_length: Maximum summary length
            style: Summary style (concise, detailed, bullets, structured)
            
        Returns:
            Extractive summary
        """
        if not text or len(text) < 100:
            return text[:max_length] if text else ""
        
        # Fall back to enhanced extractive summarization
        try:
            # Approach depends on whether NLTK is available
            if NLTK_AVAILABLE:
                # Use NLTK for better text segmentation and frequency analysis
                return self._generate_extractive_summary_nltk(text, max_length, style)
            else:
                # Use regex-based summarization without NLTK
                return self._generate_extractive_summary_regex(text, max_length, style)
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            
            # Very simple fallback - first few sentences or characters
            sentences = re.split(r'(?<=[.!?])\s+', text)
            summary = sentences[0]
            
            i = 1
            while i < len(sentences) and len(summary) + len(sentences[i]) + 1 <= max_length:
                summary += " " + sentences[i]
                i += 1
            
            return summary
    
    def _generate_extractive_summary_nltk(self, text: str, max_length: int = 200, style: str = "concise") -> str:
        """
        Generate extractive summary using NLTK
        
        Args:
            text: Input text
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            Extractive summary
        """
        # Extract title if available
        title = ""
        first_line = text.split('\n', 1)[0].strip()
        if len(first_line) < 100 and not first_line.endswith(('.', '!', '?')):
            title = first_line
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Use improved TextRank-inspired algorithm for important sentences
        sentence_scores = {}
        
        # Tokenize text into words
        tokens = word_tokenize(text.lower())
        
        # Use POS tagging if available to filter non-important words
        filtered_tokens = []
        try:
            pos_tags = nltk.pos_tag(tokens)
            for word, tag in pos_tags:
                if word not in STOPWORDS and word not in string.punctuation:
                    filtered_tokens.append(word)
        except:
            # Basic filtering if POS tagging fails
            filtered_tokens = [word for word in tokens if word not in STOPWORDS and word not in string.punctuation]
        
        # Calculate word frequencies
        word_freq = FreqDist(filtered_tokens)
        
        # Calculate sentence scores based on word frequencies
        for i, sentence in enumerate(sentences):
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Tokenize sentence
            sent_words = word_tokenize(sentence.lower())
            
            # Filter stopwords and punctuation
            sent_words = [word for word in sent_words if word not in STOPWORDS and word not in string.punctuation]
            
            if not sent_words:
                continue
            
            # Calculate score based on word frequency of important words
            score = sum(word_freq[word] for word in sent_words) / len(sent_words)
            
            # Position bias based on style
            if style == "concise":
                # Strong bias for first and last sentences
                if i < len(sentences) * 0.1:  # First 10%
                    score *= 1.5
                elif i > len(sentences) * 0.9:  # Last 10%
                    score *= 1.2
            else:  # "detailed" or other styles
                # Milder position bias
                if i < len(sentences) * 0.2:  # First 20%
                    score *= 1.25
                elif i > len(sentences) * 0.8:  # Last 20%
                    score *= 1.1
            
            # Extra boost for sentences with key terms
            # Customize key terms based on likely document type
            key_terms = ["summary", "conclusion", "result", "important", "significant", 
                        "key", "main", "primary", "critical", "essential", "crucial"]
            
            # Check if any words match key terms
            if any(word in key_terms for word in sent_words):
                score *= 1.2
            
            sentence_scores[i] = score
        
        # Determine how many sentences to include based on style and max_length
        avg_sent_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 20
        target_sentences = max(1, int(max_length / avg_sent_length))
        
        if style == "detailed":
            target_sentences = min(int(target_sentences * 1.5), len(sentences))
        elif style == "concise":
            target_sentences = max(1, int(target_sentences * 0.7))
        
        # Get top scoring sentences
        top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in top_indices[:target_sentences]]
        
        # Sort indices to preserve original order
        top_indices.sort()
        
        # Extract key sentences
        key_sentences = [sentences[i] for i in top_indices if i < len(sentences)]
        
        # Build summary
        if title:
            # If we have a title, include it
            summary_parts = [title]
            remaining_length = max_length - len(title)
            
            # Add sentences up to remaining length
            for sentence in key_sentences:
                if len(sentence) + 2 <= remaining_length:
                    summary_parts.append(sentence)
                    remaining_length -= len(sentence) + 2
                else:
                    break
            
            summary = title
            if len(summary_parts) > 1:
                summary += ". " + " ".join(summary_parts[1:])
        else:
            # Without title, just join key sentences
            summary = " ".join(key_sentences)
        
        # Format summary based on style
        if style == "bullets":
            return self._format_as_bullet_points(summary)
        elif style == "structured":
            return self._format_as_structured_summary(summary)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _generate_extractive_summary_regex(self, text: str, max_length: int = 200, style: str = "concise") -> str:
        """
        Generate extractive summary using regex (when NLTK is not available)
        
        Args:
            text: Input text
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on position and keywords
        sentence_scores = {}
        
        # Simplified stopwords if NLTK is not available
        simple_stopwords = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                          "when", "where", "how", "which", "who", "whom", "this", "that", "these",
                          "those", "then", "just", "so", "than", "such", "both", "through", "about",
                          "for", "is", "of", "while", "during", "to", "from"}
        
        # Build a simple word frequency counter
        word_counts = {}
        for sentence in sentences:
            # Split into words and count frequencies
            words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
            for word in words:
                if word not in simple_stopwords:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Score sentences
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.strip()) < 10:
                continue
            
            # Base score
            score = 0
            
            # Position score - first and last sentences are important
            if i == 0:
                score += 5  # First sentence
            elif i == len(sentences) - 1:
                score += 3  # Last sentence
            elif i < len(sentences) * 0.1:
                score += 2  # Early sentences
            
            # Word importance score
            words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
            if words:
                # Average frequency of non-stopwords
                word_score = sum(word_counts.get(word, 0) for word in words 
                               if word not in simple_stopwords) / len(words)
                score += word_score
            
            # Keyword score
            key_terms = ["summary", "conclusion", "result", "important", "significant", 
                        "key", "main", "primary", "critical", "essential", "crucial"]
            for term in key_terms:
                if term in sentence.lower():
                    score += 3
                    break
            
            sentence_scores[i] = score
        
        # Determine how many sentences to include
        avg_sent_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 20
        target_sentences = max(1, int(max_length / avg_sent_length))
        
        if style == "detailed":
            target_sentences = min(int(target_sentences * 1.5), len(sentences))
        elif style == "concise":
            target_sentences = max(1, int(target_sentences * 0.7))
        
        # Get top scoring sentences
        top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in top_indices[:target_sentences]]
        
        # Sort indices to preserve original order
        top_indices.sort()
        
        # Extract key sentences
        summary = " ".join(sentences[i] for i in top_indices if i < len(sentences))
        
        # Format summary based on style
        if style == "bullets":
            return self._format_as_bullet_points(summary)
        elif style == "structured":
            return self._format_as_structured_summary(summary)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _format_as_bullet_points(self, summary: str) -> str:
        """
        Format summary as bullet points
        
        Args:
            summary: Text summary
            
        Returns:
            Bullet point formatted summary
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        # Format as bullet points
        bullet_points = []
        
        for sentence in sentences:
            if sentence.strip():
                # Clean the sentence
                clean_sentence = sentence.strip()
                
                # Ensure it ends with proper punctuation
                if not clean_sentence[-1] in ['.', '!', '?']:
                    clean_sentence += '.'
                
                # Add bullet point
                bullet_points.append(f"• {clean_sentence}")
        
        return '\n'.join(bullet_points)
    
    def _format_as_structured_summary(self, summary: str) -> str:
        """
        Format as structured summary with sections
        
        Args:
            summary: Text summary
            
        Returns:
            Structured summary
        """
        # Extract potential entities for structured summary
        sections = {
            "SUMMARY": summary
        }
        
        # Try to extract key entities
        people = []
        organizations = []
        locations = []
        dates = []
        
        # Look for potential names (capitalized phrases)
        name_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', summary)
        if name_matches:
            people = list(set(name_matches))[:3]  # Top 3 unique names
        
        # Look for potential organizations
        org_patterns = [
            r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)+\s+(?:Inc|Corp|Co|Ltd|LLC|Company|Association|Organization)\b',
            r'\b[A-Z][A-Z]+\b'  # Acronyms
        ]
        org_matches = []
        for pattern in org_patterns:
            org_matches.extend(re.findall(pattern, summary))
        
        if org_matches:
            organizations = list(set(org_matches))[:3]  # Top 3 unique
        
        # Look for potential locations
        loc_patterns = [
            r'\b[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)?\b'  # City, Country format
        ]
        loc_matches = []
        for pattern in loc_patterns:
            loc_matches.extend(re.findall(pattern, summary))
        
        if loc_matches:
            locations = list(set(loc_matches))[:3]  # Top 3 unique
        
        # Look for dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',                   # MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b',  # Month Day, Year
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{2,4}\b'  # Day Month Year
        ]
        date_matches = []
        for pattern in date_patterns:
            date_matches.extend(re.findall(pattern, summary))
        
        if date_matches:
            dates = list(set(date_matches))[:2]  # Top 2 unique
        
        # Add sections if not empty
        if people:
            sections["PEOPLE"] = ", ".join(people)
        if organizations:
            sections["ORGANIZATIONS"] = ", ".join(organizations)
        if locations:
            sections["LOCATIONS"] = ", ".join(locations)
        if dates:
            sections["DATES"] = ", ".join(dates)
        
        # Format structured summary
        formatted_summary = []
        
        for section, content in sections.items():
            formatted_summary.append(f"{section}:")
            formatted_summary.append(content)
            formatted_summary.append("")  # Empty line between sections
        
        return "\n".join(formatted_summary).strip()
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """
        Extract key insights using rule-based methods
        
        Args:
            text: Input text
            
        Returns:
            List of key insights
        """
        insights = []
        
        # Extract key facts using simple heuristics
        if NLTK_AVAILABLE:
            try:
                # Tokenize sentences
                sentences = sent_tokenize(text)
                
                # Look for sentences with insight markers
                insight_markers = [
                    r'\b(?:key|main|important|significant|critical)\s+(?:point|fact|finding|result|conclusion)\b',
                    r'\b(?:in\s+summary|to\s+summarize|in\s+conclusion|concluding|therefore)\b',
                    r'\b(?:must|should|need\s+to|have\s+to)\b',
                    r'\b(?:increased|decreased|improved|reduced|enhanced|caused)\b'
                ]
                
                fact_sentences = []
                for sentence in sentences:
                    if any(re.search(marker, sentence, re.IGNORECASE) for marker in insight_markers):
                        fact_sentences.append(sentence)
                
                # Take top 3 fact sentences
                for sentence in fact_sentences[:3]:
                    insights.append(sentence)
                
                # If no marker-based insights, use TextRank to find key sentences
                if not fact_sentences and len(sentences) > 3:
                    # Simplified TextRank implementation
                    sentence_scores = {}
                    
                    # Tokenize text into words
                    words = [w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in STOPWORDS]
                    word_freq = FreqDist(words)
                    
                    for i, sentence in enumerate(sentences):
                        if len(sentence) < 15:  # Skip very short sentences
                            continue
                        
                        # Calculate score based on word frequencies
                        words_in_sentence = [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
                        score = sum(word_freq[w] for w in words_in_sentence) / max(1, len(words_in_sentence))
                        
                        # Boost score for sentences at beginning or end
                        if i < len(sentences) * 0.2:  # First 20%
                            score *= 1.25
                        elif i > len(sentences) * 0.8:  # Last 20%
                            score *= 1.1
                        
                        sentence_scores[i] = score
                    
                    # Get top scoring sentences
                    top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
                    top_indices = [idx for idx, _ in top_indices[:2]]  # Take top 2
                    
                    # Add key sentences as insights
                    for idx in sorted(top_indices):
                        if idx < len(sentences):
                            insights.append(sentences[idx])
            
            except Exception as e:
                logger.warning(f"Fact extraction failed: {e}")
        else:
            # Simple regex-based fact extraction without NLTK
            # Split into sentences using regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Define insight marker patterns
            insight_patterns = [
                r'(?:key|main|important|significant|critical).{0,20}(?:point|fact|finding|conclusion)',
                r'(?:in\s+summary|to\s+summarize|in\s+conclusion|concluding|therefore)',
                r'(?:must|should|need to|have to)',
                r'increase|decrease|improve|reduce|enhance|cause'
            ]
            
            # Look for sentences with insight markers
            for sentence in sentences:
                for pattern in insight_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        insights.append(sentence)
                        break
                        
                # Limit to 5 insights
                if len(insights) >= 5:
                    break
            
            # If not enough insight sentences, add first and last sentences
            if len(insights) < 2 and len(sentences) > 2:
                if sentences[0] not in insights:
                    insights.append(sentences[0])  # First sentence
                if sentences[-1] not in insights and sentences[-1] != sentences[0]:
                    insights.append(sentences[-1])  # Last sentence
        
        # Limit to 5 insights
        return insights[:5]
    
    def _organize_output(self, results: Dict) -> Dict:
        """
        Organize output for better readability and structure
        
        Args:
            results: OCR results dictionary
            
        Returns:
            Organized results dictionary
        """
        # Create a structured output with clear sections
        organized = {
            "status": results.get("status", ""),
            "metadata": {
                "processing_info": {
                    "time_ms": results.get("metadata", {}).get("processing_time_ms", 0),
                    "detected_language": results.get("metadata", {}).get("detected_language", "unknown"),
                    "image_type": results.get("metadata", {}).get("image_type", "unknown"),
                    "ocr_engine": results.get("metadata", {}).get("best_engine", "unknown"),
                    "confidence": results.get("confidence", 0)
                }
            }
        }
        
        # Add text content section
        if "text" in results and results["text"]:
            organized["content"] = {
                "full_text": results["text"]
            }
            
            # Add summary if available
            if "summary" in results and results["summary"]:
                organized["content"]["summary"] = results["summary"]
            
            # Add document structure if available
            if "document_structure" in results:
                organized["content"]["document_structure"] = results["document_structure"]
            
            # Add key insights if available
            if "key_insights" in results:
                organized["content"]["key_insights"] = results["key_insights"]
        
        # Add structured information if available
        if "metadata" in results and "structured_info" in results["metadata"] and results["metadata"]["structured_info"]:
            organized["structured_data"] = results["metadata"]["structured_info"]
        
        # Add detailed image stats for debugging
        if "metadata" in results and "image_stats" in results["metadata"]:
            organized["metadata"]["image_stats"] = results["metadata"]["image_stats"]
        
        return organized
    
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

def process_file(file_path, output=None, language=None, page=0, summary_length=200, summary_style="concise"):
    """Process a file in standalone mode with enhanced options"""
    # Create OCR engine
    ocr = SmartGlassOCR()
    
    # Process the file
    results = ocr.process_file(
        file_path=file_path,
        language=language,
        page=page,
        summary_length=summary_length,
        summary_style=summary_style
    )
    
    # Print results to console
    if results["status"] == "success" or results["status"] == "partial_success":
        print(f"Status: {results['status']} (Confidence: {results.get('confidence', 0):.1f}%)")
        print(f"Image Type: {results['metadata'].get('image_type', 'unknown')}")
        print(f"OCR Engine: {results['metadata'].get('best_engine', 'unknown')}")
        print(f"Processing Time: {results['metadata'].get('processing_time_ms', 0):.1f} ms")
        print(f"Detected Language: {results['metadata'].get('detected_language', 'unknown')}")
        print("\n--- Summary ---")
        print(results.get("summary", "No summary available"))
        print("\n--- Full Text ---")
        print(results.get("text", "No text extracted"))
    else:
        print(f"Error: {results.get('message', 'Unknown error')}")
    
    # Save to output file if specified
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            if results["status"] == "success" or results["status"] == "partial_success":
                f.write("--- Summary ---\n")
                f.write(results.get("summary", "No summary available"))
                f.write("\n\n--- Full Text ---\n")
                f.write(results.get("text", "No text extracted"))
            else:
                f.write(f"Error: {results.get('message', 'Unknown error')}")
        print(f"Results saved to {output}")
    
    return results

def process_directory(directory_path, output_dir=None, language=None, summary_length=200, summary_style="concise"):
    """Process all supported files in a directory"""
    # Create OCR engine
    ocr = SmartGlassOCR()
    
    # Get list of supported files
    supported_extensions = ocr.config["allowed_extensions"]
    files = []
    
    for ext in supported_extensions:
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

    

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartGlassOCR - Advanced OCR engine optimized for smart glasses")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-l", "--language", default=None, help="OCR language (e.g., 'eng', 'eng+ind')")
    parser.add_argument("-p", "--page", type=int, default=0, help="Page number for PDF (0-based)")
    parser.add_argument("-s", "--summary-length", type=int, default=200, help="Maximum summary length")
    parser.add_argument("--summary-style", choices=["concise", "detailed", "bullets", "structured"], 
                       default="concise", help="Summary style")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set up debug logging if enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        process_file(
            file_path=args.input,
            output=args.output,
            language=args.language,
            page=args.page,
            summary_length=args.summary_length,
            summary_style=args.summary_style
        )
    elif os.path.isdir(args.input):
        process_directory(
            directory_path=args.input,
            output_dir=args.output,
            language=args.language,
            summary_length=args.summary_length,
            summary_style=args.summary_style
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0

if __name__ == "__main__":
    # Add required imports for standalone usage
    import glob
    import sys
    
    sys.exit(main())