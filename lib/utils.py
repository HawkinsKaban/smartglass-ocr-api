#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions and classes for SmartGlassOCR
"""

import time
import logging
import threading
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