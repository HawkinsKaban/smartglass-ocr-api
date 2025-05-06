"""
SmartGlass OCR API - OCR Processor
Wrapper around the SmartGlassOCR engine with improved error handling
"""

import os
import time
import logging
from typing import Dict, Any, Tuple
from flask import current_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ocr_api.log")
    ]
)
logger = logging.getLogger("OCR-Processor")

class OCRProcessor:
    """
    OCR Processor that wraps the SmartGlassOCR engine
    and handles Markdown generation with improved error handling
    """
    
    def __init__(self):
        """Initialize the OCR Processor with better error handling"""
        try:
            # Import the SmartGlassOCR engine
            from lib.smartglass_ocr import SmartGlassOCR
            self.ocr_engine = SmartGlassOCR()
            
            # Import the Markdown formatter
            from app.core.markdown_formatter import MarkdownFormatter
            self.markdown_formatter = MarkdownFormatter()
            
            logger.info("OCR Processor initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise RuntimeError(f"OCR system initialization failed: {e}")
        except Exception as e:
            logger.error(f"Error initializing OCR Processor: {e}")
            raise RuntimeError(f"OCR system initialization failed: {e}")
    
    def process_file(self, file_path: str, original_filename: str = None, language: str = None, 
                    page: int = 0, summary_length: int = None, 
                    summary_style: str = None) -> Tuple[Dict[str, Any], str]:
        """
        Process a file with OCR and generate markdown with improved error handling
        
        Args:
            file_path: Path to the file to process
            original_filename: Original filename for display
            language: OCR language
            page: Page number for PDF (0-based)
            summary_length: Maximum summary length
            summary_style: Summary style (concise, detailed, bullets, structured)
            
        Returns:
            Tuple of (OCR results dict, Markdown filename)
        """
        logger.info(f"Processing file: {original_filename}")
        start_time = time.time()
        
        try:
            # Default values from config
            if language is None:
                language = current_app.config.get('DEFAULT_LANGUAGE')
            if summary_length is None:
                summary_length = current_app.config.get('DEFAULT_SUMMARY_LENGTH')
            if summary_style is None:
                summary_style = current_app.config.get('DEFAULT_SUMMARY_STYLE')
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine if this is likely a banner/signage image by analyzing content
            is_likely_signage = self._check_if_signage(file_path)
            
            # Process the file with OCR with timeout
            timeout = current_app.config.get('OCR_TIMEOUT', 120)
            
            # Modify OCR settings for signage if detected
            original_config = {}
            if is_likely_signage and hasattr(self.ocr_engine, 'config'):
                # Store original config
                original_config = self.ocr_engine.config.copy()
                
                # Apply optimized settings for signage
                self.ocr_engine.config["preprocessing_level"] = "aggressive"
                self.ocr_engine.config["use_all_available_engines"] = True
                self.ocr_engine.config["adaptive_binarization"] = True
                self.ocr_engine.config["edge_enhancement"] = True
                self.ocr_engine.config["perspective_correction"] = True
            
            # Process the file
            results = self._process_with_timeout(
                file_path=file_path,
                language=language,
                page=page,
                summary_length=summary_length,
                summary_style=summary_style,
                timeout=timeout
            )
            
            # Restore original OCR settings if changed
            if is_likely_signage and original_config and hasattr(self.ocr_engine, 'config'):
                self.ocr_engine.config = original_config
            
            # Add signage-specific information if detected
            if is_likely_signage:
                content_type, description = self._analyze_signage_content(results.get('text', ''))
                
                # Add metadata if not present
                if 'metadata' not in results:
                    results['metadata'] = {}
                
                # Add signage-specific metadata
                results['metadata']['is_outdoor_signage'] = True
                results['metadata']['content_type'] = content_type
                results['metadata']['description'] = description
                
                # Ensure original_text is preserved
                if 'text' in results:
                    results['original_text'] = results['text']
                
                # Update summary with signage description
                results['summary'] = description
                
                # Add key insights for signage
                results['key_insights'] = [
                    f"This appears to be a {content_type} sign/banner",
                    f"The text is most likely in {self._detect_language(results.get('text', ''))}",
                    description
                ]
            
            # Generate markdown content
            if isinstance(results, tuple):
                # Unpack the tuple if it's a tuple
                result_dict, _ = results
                md_content = self.markdown_formatter.format_ocr_results(result_dict, original_filename)
            else:
                # It's already a dictionary
                md_content = self.markdown_formatter.format_ocr_results(results, original_filename)
            
            # Save markdown file
            md_filename = self._save_markdown_file(md_content, original_filename)
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"File processed in {processing_time:.2f} seconds: {original_filename}")
            
            return results, md_filename
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {
                "status": "error",
                "message": f"File not found: {str(e)}",
                "metadata": {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }, ""
        except TimeoutError:
            logger.error(f"OCR processing timed out for {original_filename}")
            return {
                "status": "error", 
                "message": "OCR processing timed out. Try with a smaller image or PDF.",
                "metadata": {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }, ""
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
    
    def _check_if_signage(self, file_path):
        """
        Check if this image is likely a banner or signage
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Boolean indicating if the image is likely signage
        """
        try:
            import cv2
            import numpy as np
            
            # Read image
            image = cv2.imread(file_path)
            
            if image is None:
                return False
            
            # Convert to grayscale
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate key metrics
            height, width = image.shape[:2]
            contrast = np.std(gray)
            aspect_ratio = width / height
            
            # Calculate color variance (useful for signage)
            if len(image.shape) > 2:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                color_std = np.std(hsv[:,:,1])  # Saturation standard deviation
                has_strong_colors = color_std > 50
            else:
                has_strong_colors = False
            
            # Calculate edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.count_nonzero(edges)
            edge_density = edge_pixels / (width * height)
            
            # Look for rectangular contours (common in signage)
            has_rectangular_contour = False
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # If we find a rectangle that's large enough to be a sign
                if len(approx) == 4 and cv2.contourArea(contour) > 0.3 * (width * height):
                    has_rectangular_contour = True
                    break
            
            # Detect text regions
            text_regions_count = 0
            try:
                import pytesseract
                from PIL import Image
                
                pil_image = Image.fromarray(gray)
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                text_regions_count = sum(1 for conf in data['conf'] if conf != '-1' and int(conf) > 60)
            except:
                # If text detection fails, estimate based on edge patterns
                text_regions_count = 5  # Default assumption
            
            # Detect if image is a sign/banner
            is_signage = (
                (contrast > 60 or has_strong_colors) and  # Good contrast or strong colors
                (edge_density < 0.08) and  # Not too dense with edges
                (aspect_ratio > 1.5 or aspect_ratio < 0.67 or has_rectangular_contour) and  # Wide/tall or has rectangular border
                (text_regions_count > 0 and text_regions_count < 15)  # Some text, but not too much
            )
            
            return is_signage
            
        except Exception as e:
            logger.warning(f"Error checking if signage: {e}")
            return False
    
    def _analyze_signage_content(self, text):
        """
        Analyze signage content to determine type and generate description
        
        Args:
            text: Extracted text from signage
            
        Returns:
            Tuple of (content_type, description)
        """
        if not text:
            return "unknown", "Could not determine the content of this sign."
        
        # Lowercase for matching
        text_lower = text.lower()
        
        # Identify different types of signs/banners
        # Promotional/advertising
        if any(word in text_lower for word in ['beli', 'diskon', 'gratis', 'promo', 'sale', 'discount', 'free', 
                                             'special', 'offer', 'limited', 'new', 'buy']):
            # Check for restaurant promotion
            if any(word in text_lower for word in ['menu', 'restoran', 'restaurant', 'makanan', 'food', 'makan', 
                                                  'cafe', 'kafe', 'kebab', 'pizza', 'burger']):
                return "restaurant_promotion", "This is a promotional sign for a restaurant or food establishment offering special deals."
            
            # Check for retail promotion
            if any(word in text_lower for word in ['toko', 'store', 'shop', 'mall', 'retail', 'belanja', 'shopping']):
                return "retail_promotion", "This is a retail promotion sign advertising sales or special offers."
            
            # General promotion
            return "promotion", "This appears to be a promotional sign or banner advertising a special offer or deal."
        
        # Property/real estate
        if any(word in text_lower for word in ['jual', 'sewa', 'dijual', 'disewakan', 'rent', 'sale', 'property', 
                                             'rumah', 'apartemen', 'tanah', 'house', 'apartment', 'land', 'estate']):
            return "property", "This is a property sign that appears to be advertising rental or sale information."
        
        # Government/official
        if any(word in text_lower for word in ['kantor', 'pemerintah', 'dinas', 'kementerian', 'departemen', 
                                             'ministry', 'office', 'government', 'official', 'agency', 'badan']):
            return "government", "This is an official sign from a government office or agency."
        
        # Religious
        if any(word in text_lower for word in ['masjid', 'mosque', 'gereja', 'church', 'temple', 'kuil', 
                                              'agama', 'religion', 'allah', 'tuhan', 'god', 'muhammad', 'jesus']):
            return "religious", "This appears to be a religious sign or announcement."
        
        # Celebratory/Event
        if any(word in text_lower for word in ['selamat', 'congratulations', 'peringatan', 'commemoration', 
                                              'acara', 'event', 'festival', 'celebration', 'memperingati']):
            return "celebration", "This is a celebration or commemorative banner for a special event or occasion."
        
        # Directional/Navigation
        if any(word in text_lower for word in ['arah', 'direction', 'jalan', 'road', 'belok', 'turn', 'km', 
                                              'meter', 'parkir', 'parking', 'masuk', 'enter', 'keluar', 'exit']):
            return "directional", "This is a directional or navigation sign."
        
        # Warning/Safety
        if any(word in text_lower for word in ['awas', 'warning', 'bahaya', 'danger', 'hati-hati', 'caution', 
                                              'peringatan', 'larangan', 'dilarang', 'prohibited', 'stop']):
            return "warning", "This is a warning or safety sign alerting of potential hazards or prohibitions."
        
        # Try to generate a more specific description based on the text content
        lines = text.split('\n')
        if len(lines) >= 2:
            main_text = lines[0]
            sub_text = ' '.join(lines[1:])
            
            return "general", f"This sign contains the main text \"{main_text}\" followed by additional information."
        else:
            return "general", "This appears to be a general informational sign or banner."
    
    def _detect_language(self, text):
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code
        """
        # Simple heuristic-based language detection
        if not text:
            return "unknown"
        
        # Count occurrences of language-specific characters
        text = text.lower()
        
        # Indonesian-specific words
        id_words = ['yang', 'dan', 'di', 'ke', 'ada', 'pada', 'ini', 'untuk', 'dengan', 'tidak', 'dari']
        id_count = sum(1 for word in id_words if word in text.split())
        
        # English-specific words
        en_words = ['the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'at', 'from', 'by']
        en_count = sum(1 for word in en_words if word in text.split())
        
        if id_count > en_count:
            return "ind"
        elif en_count > id_count:
            return "eng"
        else:
            # Default to combined if can't determine
            return "eng+ind"
    
    def _process_with_timeout(self, file_path, language, page, summary_length, summary_style, timeout):
        """Process file with timeout to prevent hanging"""
        import threading
        import queue

        result_queue = queue.Queue()
        
        def target_function():
            try:
                # Process the file
                result = self.ocr_engine.process_file(
                    file_path=file_path,
                    language=language,
                    page=page,
                    summary_length=summary_length,
                    summary_style=summary_style
                )
                result_queue.put(result)
            except Exception as e:
                result_queue.put({"status": "error", "message": str(e)})
        
        thread = threading.Thread(target=target_function)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # If thread is still running after timeout
            raise TimeoutError("OCR processing timed out")
        
        if result_queue.empty():
            return {"status": "error", "message": "Processing failed with no result"}
        
        return result_queue.get()
    
    def _save_markdown_file(self, md_content: str, original_filename: str) -> str:
        """
        Save markdown content to a file with better error handling
        
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
            file_path = os.path.join(current_app.config['MARKDOWN_FOLDER'], md_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Markdown file saved: {md_filename}")
            return md_filename
        except Exception as e:
            logger.error(f"Error saving markdown file: {e}")
            return f"error_saving_{int(time.time())}.md"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get OCR engine statistics
        
        Returns:
            Dictionary with OCR engine statistics
        """
        try:
            return self.ocr_engine.get_statistics()
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}