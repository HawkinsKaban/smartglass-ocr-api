import cv2
import numpy as np
import os
import logging

logger = logging.getLogger("ID-Card-Utils")

def preprocess_id_card(file_path, target_width=1000):
    """
    Specialized preprocessing for Indonesian ID cards
    
    Args:
        file_path: Path to the ID card image
        target_width: Target width for resizing
        
    Returns:
        Path to preprocessed image
    """
    try:
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return file_path
            
        # Resize image to manageable size
        h, w = img.shape[:2]
        if w > target_width:
            scale = target_width / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), 
                           interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply normalization and enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Create a new optimized image file
        output_path = f"{os.path.splitext(file_path)[0]}_optimized.jpg"
        cv2.imwrite(output_path, denoised, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return output_path
    except Exception as e:
        logger.error(f"Error preprocessing ID card: {e}")
        return file_path