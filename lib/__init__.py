"""
SmartGlass OCR API - Library Module
"""

# Export important classes and functions
from .model import ImageType, ProcessingStrategy, DocumentStructure, ImageStats
from .utils import generate_unique_filename, get_available_libraries, clean_text, order_points, MemoryManager
from .smartglass_ocr import SmartGlassOCR, process_file, process_directory