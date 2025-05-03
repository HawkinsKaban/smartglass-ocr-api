#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions and classes for SmartGlassOCR
Includes memory management, file handling, and other helper functions
"""

import os
import time
import threading
import numpy as np
import logging
import uuid
import re
from pathlib import Path

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

def generate_unique_filename(base_path, extension=".jpg"):
    """Generate a unique filename with timestamp and UUID"""
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{Path(base_path).stem}_{timestamp}_{unique_id}{extension}"
    return filename

def get_available_libraries():
    """Check which libraries are available in the environment"""
    libraries = {}
    
    # Check for OpenCV
    try:
        import cv2
        libraries["cv2"] = True
    except ImportError:
        libraries["cv2"] = False
    
    # Check for PIL
    try:
        from PIL import Image
        libraries["pil"] = True
    except ImportError:
        libraries["pil"] = False
    
    # Check for Tesseract
    try:
        import pytesseract
        libraries["tesseract"] = True
    except ImportError:
        libraries["tesseract"] = False
    
    # Check for PDF processing
    try:
        from pdf2image import convert_from_path
        libraries["pdf2image"] = True
    except ImportError:
        libraries["pdf2image"] = False
    
    # Check for NLP libraries
    try:
        import nltk
        libraries["nltk"] = True
        
        # Check NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            libraries["nltk_punkt"] = True
        except LookupError:
            libraries["nltk_punkt"] = False
            
        try:
            nltk.data.find('corpora/stopwords')
            libraries["nltk_stopwords"] = True
        except LookupError:
            libraries["nltk_stopwords"] = False
            
    except ImportError:
        libraries["nltk"] = False
        libraries["nltk_punkt"] = False
        libraries["nltk_stopwords"] = False
    
    # Check for EasyOCR
    try:
        import easyocr
        libraries["easyocr"] = True
    except ImportError:
        libraries["easyocr"] = False
    
    # Check for PaddleOCR
    try:
        from paddleocr import PaddleOCR
        libraries["paddleocr"] = True
    except ImportError:
        libraries["paddleocr"] = False
    
    return libraries

def clean_text(text, keep_newlines=True):
    """
    Clean text by removing unwanted characters and normalizing whitespace
    
    Args:
        text: Text to clean
        keep_newlines: Whether to preserve newlines
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove invalid unicode characters
    text = ''.join(c for c in text if ord(c) < 65536)
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Handle newlines based on parameter
    if keep_newlines:
        # Replace multiple newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
    else:
        # Replace newlines with spaces
        text = re.sub(r'\n', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
    
    return text.strip()

def order_points(pts):
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