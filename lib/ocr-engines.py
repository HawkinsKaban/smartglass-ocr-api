#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR engine wrappers for SmartGlassOCR
Handles Tesseract, EasyOCR, and PaddleOCR integration
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from PIL import Image

from .models import ImageType

logger = logging.getLogger("SmartGlass-OCREngines")

class TesseractOCR:
    """Wrapper for Tesseract OCR with enhanced processing"""
    
    def __init__(self, config=None):
        """
        Initialize Tesseract OCR with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.available = self._check_availability()
        
        if self.available:
            self._configure_tesseract()
    
    def _check_availability(self) -> bool:
        """Check if Tesseract is available"""
        try:
            import pytesseract
            return True
        except ImportError:
            logger.warning("Tesseract not available")
            return False
    
    def _configure_tesseract(self):
        """Configure Tesseract based on OS and user settings"""
        import pytesseract
        
        if self.config.get("tesseract_path"):
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
        if self.config.get("tesseract_data_path"):
            os.environ["TESSDATA_PREFIX"] = self.config["tesseract_data_path"]
        
        # Verify Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Error getting Tesseract version: {e}")
    
    def process(self, processed_images: List[str], image_data: Dict[str, np.ndarray], 
               language: str, layout_info: Dict) -> Tuple[str, float, str, Dict]:
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
        if not self.available:
            return "", 0, "unavailable", {}
        
        import pytesseract
        
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
        if self.config.get("debug_mode", False):
            logger.debug(f"All Tesseract OCR results: {all_results}")
        
        return best_result, best_confidence, best_method, page_layout

class EasyOCR:
    """Wrapper for EasyOCR with enhanced processing"""
    
    def __init__(self, config=None):
        """
        Initialize EasyOCR with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.available = self._check_availability()
        self.reader = None
        
        if self.available:
            self._initialize_reader()
    
    def _check_availability(self) -> bool:
        """Check if EasyOCR is available"""
        try:
            import easyocr
            return True
        except ImportError:
            logger.warning("EasyOCR not available")
            return False
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader"""
        if not self.available:
            return
        
        import easyocr
        
        # Initialize in a separate thread to avoid blocking
        import threading
        
        def init_reader():
            try:
                self.reader = easyocr.Reader(['en', 'id'])  # Initialize with English and Indonesian
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing EasyOCR: {e}")
        
        # Start initialization in a separate thread
        thread = threading.Thread(target=init_reader)
        thread.daemon = True
        thread.start()
        thread.join()  # Wait for initialization to complete
    
    def process(self, image, layout_info: Dict) -> Tuple[str, float, str, List]:
        """
        Perform OCR using EasyOCR with enhanced processing
        
        Args:
            image: Image to process
            layout_info: Document layout information
            
        Returns:
            Tuple of (text result, confidence score, method name, regions)
        """
        if not self.available or self.reader is None:
            return "", 0, "unavailable", []
        
        try:
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
            
            # Save image to file for EasyOCR (it works better with files)
            import uuid
            import os
            temp_path = f"/tmp/easyocr_temp_{uuid.uuid4()}.jpg"
            cv2.imwrite(temp_path, image)
            
            # Perform OCR with EasyOCR
            results = self.reader.readtext(temp_path, paragraph=paragraph, detail=detail)
            
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

class PaddleOCR:
    """Wrapper for PaddleOCR with enhanced processing"""
    
    def __init__(self, config=None):
        """
        Initialize PaddleOCR with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.available = self._check_availability()
        self.paddle_ocr = None
        
        if self.available:
            self._initialize_paddle()
    
    def _check_availability(self) -> bool:
        """Check if PaddleOCR is available"""
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            logger.warning("PaddleOCR not available")
            return False
    
    def _initialize_paddle(self):
        """Initialize PaddleOCR"""
        if not self.available:
            return
        
        from paddleocr import PaddleOCR
        
        # Initialize in a separate thread to avoid blocking
        import threading
        
        def init_paddle():
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR: {e}")
        
        # Start initialization in a separate thread
        thread = threading.Thread(target=init_paddle)
        thread.daemon = True
        thread.start()
        thread.join()  # Wait for initialization to complete
    
    def process(self, image, layout_info: Dict) -> Tuple[str, float, str, List]:
        """
        Perform OCR using PaddleOCR with enhanced processing
        
        Args:
            image: Image to process
            layout_info: Document layout information
            
        Returns:
            Tuple of (text result, confidence score, method name, regions)
        """
        if not self.available or self.paddle_ocr is None:
            return "", 0, "unavailable", []
        
        try:
            # Save image to file for PaddleOCR
            import uuid
            import os
            temp_path = f"/tmp/paddleocr_temp_{uuid.uuid4()}.jpg"
            cv2.imwrite(temp_path, image)
            
            # Configure PaddleOCR based on document type
            use_angle_cls = True  # Enable rotation detection
            
            # Adjust params for different document types
            rec_model_dir = None  # Default model
            
            # Perform OCR with PaddleOCR
            results = self.paddle_ocr.ocr(temp_path, cls=use_angle_cls, rec_model_dir=rec_model_dir)
            
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

class OCREngineManager:
    """Manages available OCR engines and selects the optimal one for each image type"""
    
    def __init__(self, config=None):
        """
        Initialize OCR engine manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize OCR engines
        self.engines = {
            "tesseract": TesseractOCR(config),
            "easyocr": EasyOCR(config),
            "paddleocr": PaddleOCR(config)
        }
        
        # Define best image types for each engine
        self.engine_preferences = {
            "tesseract": [
                ImageType.DOCUMENT, ImageType.HIGH_CONTRAST, ImageType.RECEIPT, 
                ImageType.BOOK_PAGE, ImageType.NEWSPAPER, ImageType.FORM, ImageType.TABLE
            ],
            "easyocr": [
                ImageType.NATURAL, ImageType.SIGNAGE, ImageType.MIXED, 
                ImageType.HANDWRITTEN, ImageType.PRESENTATION
            ],
            "paddleocr": [
                ImageType.MIXED, ImageType.LOW_QUALITY, ImageType.NATURAL, 
                ImageType.SCIENTIFIC, ImageType.TABLE
            ]
        }
        
        # Initialize performance tracking for adaptive optimization
        self.performance_stats = {
            "image_types": {},
            "processing_times": {},
            "success_rates": {}
        }
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        return [name for name, engine in self.engines.items() if engine.available]
    
    def determine_optimal_engine_sequence(self, image_type: ImageType) -> List[str]:
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
            # Filter to only available engines
            return [e for e in type_to_engine_sequence[image_type] if self.engines[e].available]
        
        # Default sequence if not found
        return [e for e in ["tesseract", "easyocr", "paddleocr"] if self.engines[e].available]
    
    def perform_ocr(self, processed_images: List[str], 
                   image_data: Dict[str, np.ndarray],
                   language: str, 
                   image_stats) -> Tuple[str, str, float, Dict]:
        """
        Perform OCR using the optimal engine sequence with improved confidence scoring
        
        Args:
            processed_images: List of processing method names
            image_data: Dictionary mapping method names to processed images
            language: OCR language
            image_stats: Image statistics
            
        Returns:
            Tuple of (best engine name, text result, confidence score, layout info)
        """
        available_engines = self.get_available_engines()
        
        if not available_engines:
            logger.error("No OCR engines available")
            return "none", "", 0, {}
        
        # Determine optimal engine sequence based on image type
        if self.config.get("use_all_available_engines", True):
            # Try all engines in order of likely effectiveness for this image type
            engine_sequence = self.determine_optimal_engine_sequence(image_stats.image_type)
        else:
            # Use only the first available best engine
            for engine in self.determine_optimal_engine_sequence(image_stats.image_type):
                if engine in available_engines:
                    engine_sequence = [engine]
                    break
            else:
                engine_sequence = [available_engines[0]]
        
        # Detect layout for better text organization
        layout_info = {
            "document_type": image_stats.image_type.value,
            "columns": 1,
            "has_table": image_stats.table_likelihood > 70,
            "has_form": image_stats.form_likelihood > 70,
            "orientation": "portrait" if image_stats.height > image_stats.width else "landscape"
        }
        
        # Try each engine in sequence
        best_text = ""
        best_confidence = 0
        best_engine = ""
        
        for engine_name in engine_sequence:
            engine = self.engines[engine_name]
            
            # Call the appropriate OCR function based on engine
            if engine_name == "tesseract":
                text, confidence, method, page_layout = engine.process(
                    processed_images, image_data, language, layout_info
                )
                engine_result = f"tesseract_{method}"
                # Update layout with tesseract info if available
                if page_layout:
                    layout_info.update(page_layout)
            
            elif engine_name == "easyocr":
                # Get first image for EasyOCR
                first_image = list(image_data.values())[0]
                text, confidence, method, regions = engine.process(
                    first_image, layout_info
                )
                engine_result = f"easyocr_{method}"
                # Update layout with region info
                if regions:
                    layout_info["text_regions"] = regions
            
            elif engine_name == "paddleocr":
                # Get first image for PaddleOCR
                first_image = list(image_data.values())[0]
                text, confidence, method, regions = engine.process(
                    first_image, layout_info
                )
                engine_result = f"paddleocr_{method}"
                # Update layout with region info
                if regions:
                    layout_info["text_regions"] = regions
            
            else:
                continue
            
            # Update best result if this is better
            weighted_confidence = self._calculate_weighted_confidence(text, confidence, engine_name)
            if weighted_confidence > best_confidence and len(text.strip()) > 0:
                best_text = text
                best_confidence = confidence  # Keep original confidence for reporting
                best_engine = engine_result
            
            # Early stopping if we have a good result
            if confidence > 80 and len(text.strip()) > 20:
                logger.info(f"Early stopping with engine {engine_result}, confidence {confidence:.1f}")
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
        
        if engine_key in self.performance_stats.get("success_rates", {}):
            success_rate = self.performance_stats["success_rates"][engine_key]
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
        clean_text = text.replace(" ", "").replace("\n", "")
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
            import pytesseract
            from PIL import Image
            
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
            if "text_regions" in layout_info and layout_info["text_regions"]:
                for region in layout_info["text_regions"]:
                    try:
                        # Extract region
                        bbox = region["bbox"]
                        # Convert bbox format based on engine
                        if isinstance(bbox, list) and len(bbox) == 4 and isinstance(bbox[0], list):
                            # EasyOCR/PaddleOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            x = min(point[0] for point in bbox)
                            y = min(point[1] for point in bbox)
                            w = max(point[0] for point in bbox) - x
                            h = max(point[1] for point in bbox) - y
                        elif isinstance(bbox, list) and len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                            # Standard bbox format: [x, y, w, h]
                            x, y, w, h = bbox
                        else:
                            continue
                            
                        # Ensure coordinates are valid
                        x, y, w, h = int(x), int(y), int(w), int(h)
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
                # Use utilities.order_points
                from .utils import order_points
                
                # Apply document warping
                # Detect edges
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Check if it's large enough to be a document
                    if cv2.contourArea(largest_contour) > 0.2 * gray.shape[0] * gray.shape[1]:
                        # Approximate the contour to find corners
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        # If it has 4 corners, it might be a document
                        if len(approx) == 4:
                            # Order the points for perspective transform
                            pts = np.array([pt[0] for pt in approx])
                            rect = order_points(pts)
                            
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
                            warped = cv2.warpPerspective(gray, M, (width, height))
                            
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
