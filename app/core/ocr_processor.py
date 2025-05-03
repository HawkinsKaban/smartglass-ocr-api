"""
SmartGlass OCR API - OCR Processor
Wrapper around the SmartGlassOCR engine
"""

import os
import time
import logging
from typing import Dict, Any, Tuple
from flask import current_app

# Import the SmartGlassOCR engine
from lib.smartglass_ocr import SmartGlassOCR

# Import the Markdown formatter
from app.core.markdown_formatter import MarkdownFormatter

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
    and handles Markdown generation
    """
    
    def __init__(self):
        """Initialize the OCR Processor"""
        self.ocr_engine = SmartGlassOCR()
        logger.info("OCR Processor initialized")
    
    def process_file(self, file_path: str, original_filename: str, language: str = None, 
                    page: int = 0, summary_length: int = None, 
                    summary_style: str = None) -> Tuple[Dict[str, Any], str]:
        """
        Process a file with OCR and generate markdown
        
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
        
        # Default values from config
        if language is None:
            language = current_app.config.get('DEFAULT_LANGUAGE')
        if summary_length is None:
            summary_length = current_app.config.get('DEFAULT_SUMMARY_LENGTH')
        if summary_style is None:
            summary_style = current_app.config.get('DEFAULT_SUMMARY_STYLE')
        
        # Process the file with OCR
        results = self.ocr_engine.process_file(
            file_path=file_path,
            language=language,
            page=page,
            summary_length=summary_length,
            summary_style=summary_style
        )
        
        # Generate markdown content
        md_content = MarkdownFormatter.format_ocr_results(results, original_filename)
        
        # Save markdown file
        md_filename = self._save_markdown_file(md_content, original_filename)
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.info(f"File processed in {processing_time:.2f} seconds: {original_filename}")
        
        return results, md_filename
    
    def _save_markdown_file(self, md_content: str, original_filename: str) -> str:
        """
        Save markdown content to a file
        
        Args:
            md_content: Markdown content to save
            original_filename: Original filename for context
            
        Returns:
            Filename of the saved markdown file
        """
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        base_name = ''.join(c for c in base_name if c.isalnum() or c in '-_.')
        md_filename = f"{base_name}_{int(time.time())}.md"
        file_path = os.path.join(current_app.config['MARKDOWN_FOLDER'], md_filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown file saved: {md_filename}")
        return md_filename
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get OCR engine statistics
        
        Returns:
            Dictionary with OCR engine statistics
        """
        return self.ocr_engine.get_statistics()