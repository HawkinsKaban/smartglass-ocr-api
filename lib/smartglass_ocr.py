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
            # ID card specific settings
            "id_card_timeout": 600,             # 10 minutes timeout for ID cards
            "id_card_resize_width": 1000,       # Resize ID cards to this width
            "id_card_use_tesseract_only": True, # Use only tesseract for ID cards
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
        
        # Initialize markdown formatter
        try:
            from app.core.markdown_formatter import MarkdownFormatter
            self.markdown_formatter = MarkdownFormatter()
        except ImportError:
            self.markdown_formatter = None
            logger.warning("Markdown formatter not available")
        
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

    def process_id_card(self, file_path: str, language: str = "ind") -> dict:
        """
        Specialized method for processing Indonesian ID cards (KTP) with optimized parameters
        
        Args:
            file_path: Path to the ID card image
            language: OCR language (default 'ind' for Indonesian)
            
        Returns:
            Dictionary with OCR results optimized for ID structure
        """
        start_time = time.time()
        try:
            import cv2
            import numpy as np
            import pytesseract
            
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                return {"status": "error", "message": "Failed to read image file"}
            
            # Resize to manageable dimensions
            h, w = image.shape[:2]
            target_width = self.config.get("id_card_resize_width", 1000)
            if w > target_width:
                scale = target_width / w
                image = cv2.resize(image, (int(w * scale), int(h * scale)), 
                                interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply aggressive preprocessing optimized for ID cards
            # CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
            
            # Binarize using adaptive thresholding
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 21, 10)
            
            # Function to extract specific fields based on position
            def extract_field(img, field_name, regex_pattern=None):
                field_config = "--psm 6 --oem 1"
                text = pytesseract.image_to_string(img, lang=language, config=field_config)
                if regex_pattern:
                    import re
                    match = re.search(regex_pattern, text)
                    if match:
                        return match.group(1).strip()
                return text.strip()
            
            # Extract text using direct Tesseract call with optimized PSM mode
            text = pytesseract.image_to_string(binary, lang=language, config="--psm 3 --oem 1")
            
            # Extract structured data - common KTP fields
            fields = {}
            
            # NIK pattern: 16 digits with possible spacing/formatting
            nik_pattern = r'NIK[:\s]+([0-9\s]+)'
            fields["NIK"] = extract_field(binary, "NIK", nik_pattern)
            
            # Other common fields
            fields["Name"] = extract_field(binary, "Name", r'Nama[:\s]+(.+)')
            fields["Place/DOB"] = extract_field(binary, "Place/DOB", r'(?:Tempat|Tempat/Tgl)[\s.]+Lahir[:\s]+(.+)')
            fields["Gender"] = extract_field(binary, "Gender", r'(?:Jenis|Jenis\s+Kelamin)[:\s]+(.+)')
            fields["Address"] = extract_field(binary, "Address", r'Alamat[:\s]+(.+)')
            fields["Religion"] = extract_field(binary, "Religion", r'Agama[:\s]+(.+)')
            fields["Marital Status"] = extract_field(binary, "Marital Status", r'Status\s+Perkawinan[:\s]+(.+)')
            fields["Occupation"] = extract_field(binary, "Occupation", r'Pekerjaan[:\s]+(.+)')
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "text": text,
                "confidence": 75.0,  # Standard confidence for ID card detection
                "metadata": {
                    "detected_language": "ind",
                    "image_type": "id_card",
                    "best_engine": "tesseract_id_card",
                    "structured_info": fields,
                    "processing_time_ms": round(processing_time * 1000, 2)
                }
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Error in ID card processing: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"ID card processing failed: {str(e)}",
                "metadata": {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }

    def _process_image(self, image_path: str, language: str) -> dict:
        """
        Enhanced image processing with improved text detection in various conditions
        
        Args:
            image_path: Path to the image
            language: OCR language
            
        Returns:
            Dictionary with OCR results
        """
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            return {"status": "error", "message": "Required image processing libraries not available"}
        
        try:
            # Step 1: Enhanced file type detection
            is_id_card = self._check_if_id_card(image_path)
            if is_id_card:
                logger.info("ID card detected, using optimized ID card processing")
                return self.process_id_card(image_path, language=language)
            
            # Step 2: Read and analyze the image
            image = None
            try:
                # Try multiple reading methods for robustness
                image = cv2.imread(image_path)
                
                if image is None:
                    # If OpenCV fails, try with PIL
                    pil_image = Image.open(image_path)
                    if pil_image.mode == 'RGBA':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert PIL image to numpy array
                    image = np.array(pil_image)
                    
                    # Convert RGB to BGR (OpenCV format)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = image[:, :, ::-1].copy()
                
                if image is None or image.size == 0:
                    # If both methods fail, try to convert to JPG and read again
                    temp_jpg = f"{image_path}_temp.jpg"
                    pil_image = Image.open(image_path)
                    pil_image.convert('RGB').save(temp_jpg, quality=95)
                    image = cv2.imread(temp_jpg)
                    try:
                        os.remove(temp_jpg)
                    except:
                        pass
            except Exception as e:
                logger.error(f"Error reading image: {e}")
                
            if image is None:
                return {"status": "error", "message": "Failed to read image file"}
            
            # Step 3: Enhanced image analysis
            image_stats = self.image_processor.analyze_image(image)
            
            # Step 4: Apply advanced image corrections based on image type
            if image_stats.image_type in [ImageType.LOW_QUALITY, ImageType.NATURAL]:
                # Apply additional corrections for low quality images
                try:
                    # Check for blur and apply correction
                    if image_stats.blur < 100:
                        # Apply sharpening to reduce blur
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        image = cv2.filter2D(image, -1, kernel)
                    
                    # Check for low contrast and apply correction
                    if image_stats.contrast < 40:
                        # Apply CLAHE for better contrast
                        if len(image.shape) > 2:
                            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                            cl = clahe.apply(l)
                            merged = cv2.merge((cl, a, b))
                            image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                        else:
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                            image = clahe.apply(image)
                    
                    # Check for uneven lighting
                    if image_stats.brightness < 50 or image_stats.brightness > 200:
                        # Apply adaptive histogram equalization
                        if len(image.shape) > 2:
                            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            v = clahe.apply(v)
                            merged = cv2.merge((h, s, v))
                            image = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
                        else:
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            image = clahe.apply(image)
                    
                    # If there's perspective distortion, correct it
                    if image_stats.image_type == ImageType.NATURAL and self.config.get("perspective_correction", True):
                        corrected = self._try_perspective_correction(image)
                        if corrected is not None:
                            image = corrected
                except Exception as e:
                    logger.warning(f"Advanced image correction failed: {e}")
            
            # Step 5: Improved strategy selection
            strategy = self.image_processor.determine_processing_strategy(image_stats)
            logger.info(f"Enhanced image analysis: {image_stats.image_type.value}, " 
                      f"{image_stats.width}x{image_stats.height}, "
                      f"brightness: {image_stats.brightness:.1f}, "
                      f"contrast: {image_stats.contrast:.1f}")
            logger.info(f"Using improved processing strategy: {strategy.value}")
            
            # Step 6: Auto orientation correction if needed
            if self.config.get("auto_rotate", True) and image_stats.image_type not in [ImageType.NATURAL, ImageType.SIGNAGE]:
                # Enhanced auto-rotation detection
                try:
                    import pytesseract
                    
                    # Convert to grayscale if needed
                    if len(image.shape) > 2:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = image
                    
                    # Get orientation info from Tesseract
                    osd = pytesseract.image_to_osd(gray)
                    angle = int(re.search(r'Rotate: (\d+)', osd).group(1))
                    
                    if angle != 0:
                        logger.info(f"Auto-rotating image by {angle} degrees")
                        
                        # Rotate image
                        h, w = image.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                except Exception as e:
                    logger.warning(f"Auto-rotation detection failed: {e}")
                    
                    # Fallback to simpler heuristic-based rotation
                    image = self.image_processor.auto_rotate(image)
            
            # Step 7: Apply improved preprocessing with enhanced methods
            processed_images, image_data = self.image_processor.preprocess_image(image, image_stats, strategy)
            
            # Step 8: Apply multi-engine OCR with improved confidence scoring
            from .ocr_engines import OCREngineManager
            
            # Initialize OCR engine manager with custom configuration for this image
            custom_config = self.config.copy()
            
            # Tailor OCR approach based on image type
            if image_stats.image_type == ImageType.ID_CARD:
                custom_config["use_all_available_engines"] = False  # Only use Tesseract for ID cards
                custom_config["lightweight_mode"] = True
            elif image_stats.image_type == ImageType.HANDWRITTEN:
                # For handwritten text, prioritize EasyOCR if available
                if "easyocr" in self.ocr_engines and self.ocr_engines["easyocr"]["available"]:
                    custom_config["preferred_engine"] = "easyocr"
            elif image_stats.image_type == ImageType.SIGNAGE:
                # For signage, try all engines with aggressive preprocessing
                custom_config["use_all_available_engines"] = True
                custom_config["preprocessing_level"] = "aggressive"
            
            ocr_manager = OCREngineManager(custom_config)
            
            # Perform OCR with multiple engines
            best_engine, text, confidence, layout_info = ocr_manager.perform_ocr(
                processed_images, image_data, language, image_stats
            )
            
            # Step 9: Apply enhanced language-specific text correction
            if self.config.get("enable_text_correction", True) and len(text) > 10:
                from .text_processing import post_process_text
                
                # Detect language for language-specific corrections
                from .text_processing import detect_language
                detected_language = detect_language(text)
                
                # Apply language-specific corrections
                text = post_process_text(text, image_stats.image_type)
                
                # Apply additional corrections for Indonesian text
                if detected_language == 'ind':
                    # Fix common Indonesian OCR errors
                    text = self._fix_indonesian_text(text)
            
            # Step 10: Apply better text formatting and organization
            from .text_processing import format_text
            formatted_text = format_text(text, layout_info)
            
            # Step 11: Extract additional information
            from .text_processing import detect_language
            detected_language = detect_language(formatted_text)
            
            # Extract structured information if enabled
            structured_info = None
            if self.config.get("enable_structured_extraction", True) and formatted_text:
                from .information_extraction import extract_structured_info
                structured_info = extract_structured_info(formatted_text, image_stats.image_type)
            
            # Clean up memory if not caching
            if not self.config.get("cache_processed_images", True):
                image_data.clear()
            
            # Determine status based on enhanced criteria
            status = "success"
            if confidence < 30 or len(formatted_text.strip()) < 5:
                status = "poor_quality"
            elif confidence < 60:
                status = "partial_success"
            
            # Step 12: Prepare enhanced result with more detailed metadata
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
                    "processing_strategy": strategy.value,
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
    
    def _fix_indonesian_text(self, text: str) -> str:
        """
        Apply specific corrections for Indonesian text
        
        Args:
            text: Indonesian text with potential OCR errors
            
        Returns:
            Corrected text
        """
        if not text:
            return ""
        
        # Common Indonesian OCR errors
        replacements = {
            # Letter confusions
            'l<epada': 'kepada',
            'l<ami': 'kami',
            'l<arena': 'karena',
            'bal1wa': 'bahwa',
            'adala11': 'adalah',
            'dala1n': 'dalam',
            'merniliki': 'memiliki',
            'rnengenai': 'mengenai',
            'dalarn': 'dalam',
            'rnasa': 'masa',
            
            # Common word errors
            'Nornor': 'Nomor',
            'nornor': 'nomor',
            'Narna': 'Nama',
            'narna': 'nama',
            'Ternpat': 'Tempat',
            'ternpat': 'tempat',
            'Tgl': 'Tgl.',
            'pernerintah': 'pemerintah',
            'Provinsl': 'Provinsi',
            'Kabupaten/l<ota': 'Kabupaten/Kota',
            'Kecarnatan': 'Kecamatan',
            'Kelura11an': 'Kelurahan',
            'Jenis l<elarnin': 'Jenis Kelamin',
            'Golongan Dara11': 'Golongan Darah',
            'l<ecarnatan': 'Kecamatan',
            'Agarna': 'Agama',
            
            # Fix Indonesian abbreviations
            'RT/RVV': 'RT/RW',
            'RT /RW': 'RT/RW',
            'PROVINSI': 'PROVINSI',
            'KAB\\.': 'KAB.',
            'KEL\\.': 'KEL.',
            'KEC\\.': 'KEC.'
        }
        
        # Apply replacements
        for error, correction in replacements.items():
            text = re.sub(r'\b' + re.escape(error) + r'\b', correction, text)
        
        # Fix NIK format (16 digits for Indonesian ID cards)
        nik_matches = re.search(r'(?:NIK|N[l1]K)\s*:?\s*([0-9\s\.,]+)', text, re.IGNORECASE)
        if nik_matches:
            nik_raw = nik_matches.group(1)
            # Extract digits only
            nik_digits = ''.join(c for c in nik_raw if c.isdigit())
            if 15 <= len(nik_digits) <= 17:  # Allow for small OCR errors
                # Enforce 16 digits
                nik_digits = nik_digits[:16].zfill(16)
                # Format with spaces for readability
                formatted_nik = ' '.join([nik_digits[i:i+4] for i in range(0, len(nik_digits), 4)])
                # Replace in text
                text = re.sub(r'(?:NIK|N[l1]K)\s*:?\s*[0-9\s\.,]+', f'NIK: {formatted_nik}', text, flags=re.IGNORECASE)
        
        # Fix Indonesian date format (DD-MM-YYYY)
        date_matches = re.finditer(r'(\d{1,2})[/\-\.\\](\d{1,2})[/\-\.\\](\d{2,4})', text)
        for match in date_matches:
            day, month, year = match.groups()
            
            # Check if this might be a valid date
            try:
                day_int = int(day)
                month_int = int(month)
                year_int = int(year)
                
                if 1 <= day_int <= 31 and 1 <= month_int <= 12:
                    # Ensure 4-digit year
                    if year_int < 100:
                        year_int = 2000 + year_int if year_int < 50 else 1900 + year_int
                    
                    formatted_date = f"{day.zfill(2)}-{month.zfill(2)}-{str(year_int).zfill(4)}"
                    text = text.replace(match.group(0), formatted_date)
            except:
                # If date parsing fails, keep original
                pass
        
        # Fix Indonesian address formatting
        address_pattern = r'(?:ALAMAT|Alamat)\s*:?\s*(.+?)(?=\n\s*(?:RT/RW|PROVINSI|KABUPATEN|KECAMATAN|KELURAHAN|NIK|AGAMA|\s*$))'
        address_match = re.search(address_pattern, text, re.IGNORECASE | re.DOTALL)
        if address_match:
            address = address_match.group(1).strip()
            # Remove extra spaces and clean up
            address = re.sub(r'\s+', ' ', address)
            # Replace in text
            text = re.sub(address_pattern, f'Alamat: {address}', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Add missing colons for Indonesian ID fields
        id_fields = [
            'NAMA', 'TEMPAT/TGL LAHIR', 'JENIS KELAMIN', 'ALAMAT', 'AGAMA',
            'STATUS PERKAWINAN', 'PEKERJAAN', 'KEWARGANEGARAAN', 'BERLAKU HINGGA',
            'GOL. DARAH', 'RT/RW', 'KELURAHAN', 'KECAMATAN', 'PROVINSI', 'KABUPATEN'
        ]
        
        for field in id_fields:
            # Pattern: field name without colon followed by text
            pattern = f'({field})\s+([^\n:]+)'
            replacement = f'\\1: \\2'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _try_perspective_correction(self, image):
        """
        Attempt to correct perspective distortion in images
        
        Args:
            image: Input image with possible perspective distortion
            
        Returns:
            Corrected image or None if correction fails
        """
        try:
            import cv2
            import numpy as np
            
            # Convert to grayscale
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Blur and find edges
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 75, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Look for rectangular contours
            for contour in contours[:5]:  # Check only the 5 largest contours
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # If we have a quadrilateral
                if len(approx) == 4:
                    # Get bounding rect corners
                    rect = np.zeros((4, 2), dtype="float32")
                    
                    # Extract corners from approx
                    for i in range(4):
                        rect[i] = approx[i][0]
                    
                    # Order the corners
                    rect = order_points(rect)
                    
                    # Calculate width and height
                    (tl, tr, br, bl) = rect
                    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    
                    # Use max width and height
                    max_width = max(int(width_a), int(width_b))
                    max_height = max(int(height_a), int(height_b))
                    
                    # Define destination points
                    dst = np.array([
                        [0, 0],
                        [max_width - 1, 0],
                        [max_width - 1, max_height - 1],
                        [0, max_height - 1]
                    ], dtype="float32")
                    
                    # Calculate perspective transform
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(image, M, (max_width, max_height))
                    
                    # Check if warped image makes sense
                    if warped.shape[0] > 100 and warped.shape[1] > 100:
                        return warped
            
            # No suitable quadrilateral found
            return None
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}")
            return None

    def _check_if_id_card(self, image_path: str) -> bool:
        """Quick check if an image is an ID card to use optimized processing"""
        try:
            import cv2
            import numpy as np
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Simple checks first
            h, w = image.shape[:2]
            aspect_ratio = w / h
            
            # ID cards typically have aspect ratios between 1.4 and 1.7
            if not (1.4 < aspect_ratio < 1.8):
                return False
                
            # Quick OCR to check for ID card text (e.g., "NIK", "KTP", etc.)
            if TESSERACT_AVAILABLE:
                try:
                    import pytesseract
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Resize for faster processing
                    scale = min(1, 1000 / max(h, w))
                    resized = cv2.resize(gray, (int(w * scale), int(h * scale)), 
                                        interpolation=cv2.INTER_AREA)
                    # Run quick OCR check
                    text = pytesseract.image_to_string(resized, lang='ind', config='--psm 11 --oem 1')
                    text_lower = text.lower()
                    
                    # Check for common Indonesian ID card text
                    id_keywords = ['nik', 'ktp', 'provinsi', 'kabupaten', 'kecamatan', 
                                  'agama', 'status perkawinan', 'kewarganegaraan']
                    
                    keyword_count = sum(1 for kw in id_keywords if kw in text_lower)
                    if keyword_count >= 2:
                        logger.info(f"ID card detected with {keyword_count} keywords")
                        return True
                except Exception as e:
                    logger.warning(f"Error in quick ID check: {e}")
            
            return False
        except Exception as e:
            logger.warning(f"Error in ID card check: {e}")
            return False
    
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
    
    def _save_markdown_file(self, md_content: str, original_filename: str) -> str:
        """
        Save markdown content to a file
        
        Args:
            md_content: Markdown content to save
            original_filename: Original filename for context
            
        Returns:
            Filename of the saved markdown file
        """
        try:
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            base_name = ''.join(c for c in base_name if c.isalnum() or c in '-_.')
            md_filename = f"{base_name}_{int(time.time())}.md"
            
            # Get markdown folder
            if hasattr(self, 'config') and 'upload_folder' in self.config:
                markdown_folder = os.path.join(os.path.dirname(self.config['upload_folder']), 'markdown')
            else:
                markdown_folder = os.path.join(os.getcwd(), 'data', 'markdown')
            
            # Ensure directory exists
            os.makedirs(markdown_folder, exist_ok=True)
            
            file_path = os.path.join(markdown_folder, md_filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Markdown file saved: {md_filename}")
            return md_filename
        except Exception as e:
            logger.error(f"Error saving markdown file: {e}")
            return f"error_saving_{int(time.time())}.md"
    
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

    def process_file(self, file_path: str, original_filename: str = None, language: str = None, 
                    page: int = 0, summary_length: int = None, 
                    summary_style: str = None) -> Tuple[dict, str]:
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
            Tuple of (OCR results dict, Markdown filename)
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
            return {"status": "error", "message": "Unsupported file type"}, ""
        
        try:
            # Handle PDF vs image
            is_pdf = ext == 'pdf'
            
            if is_pdf:
                if not PDF2IMAGE_AVAILABLE:
                    return {"status": "error", "message": "PDF processing not available"}, ""
                
                # Convert PDF to image
                logger.info(f"Converting PDF to image: {file_path}, page {page}")
                image_path, total_pages = self._convert_pdf_to_image(file_path, page)
                
                if not image_path:
                    return {"status": "error", "message": "Failed to convert PDF to image"}, ""
                    
                # Process the image
                image_results = self._process_image(image_path, language)
                
                # Add PDF-specific metadata
                if "metadata" not in image_results:
                    image_results["metadata"] = {}
                    
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
            
            if "metadata" not in image_results:
                image_results["metadata"] = {}
                
            image_results["metadata"]["processing_time_ms"] = round(processing_time * 1000, 2)
            
            # Update processing stats
            self._update_processing_stats(image_results, processing_time)
            
            # Apply organized output format if enabled
            if self.config["organized_output_format"]:
                from .information_extraction import organize_output
                image_results = organize_output(image_results)
            
            # Generate markdown file
            md_filename = ""
            if self.markdown_formatter:
                try:
                    md_content = self.markdown_formatter.format_ocr_results(image_results, original_filename)
                    md_filename = self._save_markdown_file(md_content, original_filename)
                except Exception as e:
                    logger.error(f"Error generating markdown: {e}")
            
            return image_results, md_filename
            
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
            }, ""

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