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
    
    def process_file(self, file_path: str, original_filename: str, language: str = None, 
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
            
            # Process the file with OCR with timeout
            timeout = current_app.config.get('OCR_TIMEOUT', 120)
            results = self._process_with_timeout(
                file_path=file_path,
                language=language,
                page=page,
                summary_length=summary_length,
                summary_style=summary_style,
                timeout=timeout
            )
            
            # Generate markdown content
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
    
    def _process_with_timeout(self, file_path, language, page, summary_length, summary_style, timeout):
        """Process file with timeout to prevent hanging"""
        import threading
        import queue

        result_queue = queue.Queue()
        
        def target_function():
            try:
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