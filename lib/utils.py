#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions and classes for SmartGlassOCR
"""

import os
import time
import uuid
import logging
import threading
import re
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

logger = logging.getLogger("SmartGlass-Utils")

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

def calculate_hash(image):
    """
    Calculate a hash for an image to use as cache key
    
    Args:
        image: Image data as numpy array
        
    Returns:
        Hash value as string
    """
    if not isinstance(image, np.ndarray):
        return None
    
    # Simple hash based on image shape and a sample of pixels
    try:
        shape_hash = hash(image.shape)
        
        # Get a downsampled version of the image for hashing
        height, width = image.shape[:2]
        sample_factor = max(1, min(width, height) // 50)  # Downsample to roughly 50x50 or less
        
        if len(image.shape) > 2:  # Color image
            sample = image[::sample_factor, ::sample_factor, 0].flatten()  # Use first channel
        else:  # Grayscale
            sample = image[::sample_factor, ::sample_factor].flatten()
        
        # Calculate hash from sampled data
        data_hash = hash(tuple(sample[::max(1, len(sample)//100)]))  # Further reduce to ~100 values
        
        return f"{shape_hash}_{data_hash}"
    except Exception as e:
        logger.warning(f"Error calculating image hash: {e}")
        return None

def is_valid_language(language_code: str) -> bool:
    """
    Check if a language code is valid
    
    Args:
        language_code: Language code to check
        
    Returns:
        True if valid, False otherwise
    """
    # List of supported language codes in Tesseract
    # This is a subset of the most common ones
    valid_codes = {
        "eng", "ind", "ara", "bul", "cat", "ces", "chi_sim", "chi_tra", 
        "dan", "deu", "ell", "fin", "fra", "glg", "heb", "hin", "hun", 
        "ita", "jpn", "kor", "nld", "nor", "pol", "por", "ron", "rus", 
        "spa", "swe", "tha", "tur", "ukr", "vie"
    }
    
    # Check if it's a simple code
    if language_code in valid_codes:
        return True
    
    # Check if it's a compound code (e.g., eng+fra)
    if "+" in language_code:
        parts = language_code.split("+")
        return all(part in valid_codes for part in parts)
    
    return False

def format_confidence(confidence: float) -> str:
    """
    Format confidence score for display
    
    Args:
        confidence: Confidence score (0-100)
        
    Returns:
        Formatted confidence string
    """
    if confidence >= 90:
        return f"Very High ({confidence:.1f}%)"
    elif confidence >= 75:
        return f"High ({confidence:.1f}%)"
    elif confidence >= 60:
        return f"Good ({confidence:.1f}%)"
    elif confidence >= 40:
        return f"Moderate ({confidence:.1f}%)"
    elif confidence >= 20:
        return f"Low ({confidence:.1f}%)"
    else:
        return f"Very Low ({confidence:.1f}%)"

def safe_filename(filename: str) -> str:
    """
    Make a filename safe for all operating systems
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Ensure it's not too long
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:255 - len(ext) - 1]
        filename = f"{name}.{ext}" if ext else name
    
    return filename

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension in lowercase
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension without dot
    """
    return file_path.split('.')[-1].lower() if '.' in file_path else ''

# Added missing functions below:

def generate_unique_filename(filename: str) -> str:
    """
    Generate a unique filename for storage
    
    Args:
        filename: The original filename
        
    Returns:
        A unique filename with timestamp and UUID
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    # Use safe_filename instead of secure_filename since we're not importing werkzeug
    secure_name = safe_filename(filename)
    base, ext = os.path.splitext(secure_name)
    return f"{base}_{timestamp}_{unique_id}{ext}"

def get_available_libraries() -> Dict[str, bool]:
    """
    Check which OCR libraries are available in the system
    
    Returns:
        Dictionary mapping library names to availability status
    """
    libraries = {
        "cv2": False,
        "PIL": False,
        "pytesseract": False,
        "pdf2image": False,
        "nltk": False,
        "easyocr": False,
        "paddleocr": False
    }
    
    # Check OpenCV
    try:
        import cv2
        libraries["cv2"] = True
    except ImportError:
        pass
    
    # Check PIL
    try:
        from PIL import Image
        libraries["PIL"] = True
    except ImportError:
        pass
    
    # Check Tesseract
    try:
        import pytesseract
        libraries["pytesseract"] = True
    except ImportError:
        pass
    
    # Check PDF2Image
    try:
        import pdf2image
        libraries["pdf2image"] = True
    except ImportError:
        pass
    
    # Check NLTK
    try:
        import nltk
        libraries["nltk"] = True
    except ImportError:
        pass
    
    # Check EasyOCR
    try:
        import easyocr
        libraries["easyocr"] = True
    except ImportError:
        pass
    
    # Check PaddleOCR
    try:
        from paddleocr import PaddleOCR
        libraries["paddleocr"] = True
    except ImportError:
        pass
    
    return libraries

def clean_text(text: str) -> str:
    """
    Clean and normalize OCR text results
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove invalid unicode characters
    text = ''.join(c for c in text if ord(c) < 65536)
    
    # Fix common OCR errors
    replacements = {
        # Letter and number confusion
        'l': '1',  # lowercase L to 1 when between numbers
        'O': '0',  # capital O to 0 when between numbers
        'I': '1',  # capital I to 1 when between numbers
        
        # Common word errors
        'tbe': 'the',
        'arid': 'and',
        'rnay': 'may',
        'Iine': 'line',
        'tirne': 'time',
    }
    
    # Apply replacements in context (only when between numbers)
    for char, replacement in [('l', '1'), ('O', '0'), ('I', '1')]:
        text = re.sub(f'(?<=\\d){char}(?=\\d)', replacement, text)
    
    # Replace error words (whole word only)
    for error, correction in replacements.items():
        if error not in ['l', 'O', 'I']:  # Skip the ones we did above
            text = re.sub(f'\\b{error}\\b', correction, text)
    
    # Fix space issues
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between lowercase and uppercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    # Fix newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines with double newline
    
    return text.strip()

def order_points(pts):
    """
    Order points in a rectangle in top-left, top-right, bottom-right, bottom-left order
    
    Args:
        pts: Array of points (4 points for a rectangle)
        
    Returns:
        Ordered points
    """
    # Sort the points based on their x-coordinates
    pts = pts.astype(np.float32)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Grab the left-most and right-most points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    # Sort the left-most according to their y-coordinates
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    
    # Calculate the Euclidean distance from the top-left corner to the right-most points
    # The point with the largest distance is the bottom-right corner
    d = np.sqrt(((tl[0] - right_most[:, 0]) ** 2) + ((tl[1] - right_most[:, 1]) ** 2))
    if len(d) > 0:
        (br, tr) = right_most[np.argsort(d)[::-1], :]
    else:
        # Handle case with fewer than 4 points
        (br, tr) = right_most, right_most
    
    # Return the coordinates in order: top-left, top-right, bottom-right, bottom-left
    return np.array([tl, tr, br, bl], dtype=np.float32)

def convert_numpy_types(obj):
    """
    Mengkonversi tipe data NumPy ke tipe data Python native untuk JSON serialization
    
    Args:
        obj: Object yang akan dikonversi
        
    Returns:
        Object yang sudah dikonversi ke tipe data Python native
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj
    

def clean_response_text(text):
    """
    Clean text for API responses by removing newlines and special characters
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    import re
    
    # Replace newlines with spaces
    text = re.sub(r'\n+', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace backslashes
    text = text.replace('\\', '')
    
    # Replace other special characters if needed
    text = text.replace('\t', ' ')
    text = text.replace('\r', '')
    
    return text.strip()