#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image processing functionality for SmartGlassOCR
Includes image analysis, preprocessing, and enhancement
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

from .model import ImageType, ImageStats, ProcessingStrategy

logger = logging.getLogger("SmartGlass-ImageProcessing")

class ImageProcessor:
    """Image processor with enhanced analysis and preprocessing"""
    
    def __init__(self, config=None):
        """
        Initialize the image processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def analyze_image(self, image) -> ImageStats:
        """
        Perform enhanced image analysis for better type detection
        
        Args:
            image: OpenCV image
            
        Returns:
            ImageStats object with image characteristics
        """
        # Get dimensions
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Calculate color variance for determining if image is color or grayscale
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_variance = np.std(hsv[:,:,0]) + np.std(hsv[:,:,1])
        else:
            gray = image
            color_variance = 0.0
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate blur level (variance of Laplacian)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / (width * height)
        
        # Detect potential text regions with improved method
        text_regions = self._detect_text_regions(gray)
        
        # Calculate text confidence based on edge analysis
        text_confidence = self._calculate_text_confidence(gray, edges)
        
        # Check for table structures - horizontal and vertical lines
        table_likelihood = self._detect_table_likelihood(gray, edges)
        
        # Check for form structures - boxed regions and labels
        form_likelihood = self._detect_form_likelihood(gray, text_regions)
        
        # Determine image type with enhanced algorithm
        image_type = self._determine_image_type(
            width, height, brightness, contrast, blur, edge_density, 
            text_regions, aspect_ratio, color_variance, table_likelihood, 
            form_likelihood, text_confidence
        )
        
        return ImageStats(
            width=width,
            height=height,
            brightness=brightness,
            contrast=contrast,
            blur=blur,
            edge_density=edge_density,
            text_regions=len(text_regions),
            aspect_ratio=aspect_ratio,
            image_type=image_type,
            table_likelihood=table_likelihood,
            form_likelihood=form_likelihood,
            color_variance=color_variance,
            text_confidence=text_confidence
        )
    
    def preprocess_image(self, image, image_stats: ImageStats, 
                       strategy: ProcessingStrategy) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Preprocess the image with enhanced techniques based on image type
        
        Args:
            image: OpenCV image
            image_stats: ImageStats object with image characteristics
            strategy: Processing strategy to apply
            
        Returns:
            Tuple of (list of processing methods, dict mapping methods to processed images)
        """
        # Dictionary to store processed images in memory
        processed_images = []
        image_data = {}
        
        # Get dimensions for reference
        height, width = image_stats.height, image_stats.width
        
        # Convert to grayscale if not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Store grayscale version
        image_data["gray"] = gray
        processed_images.append("gray")
        
        # Apply resizing if needed
        resized_image = self._resize_for_optimal_ocr(gray, image_stats)
        if not np.array_equal(resized_image, gray):
            image_data["resized"] = resized_image
            processed_images.append("resized")
            base_image = resized_image
        else:
            base_image = gray
        
        # Apply strategy-specific preprocessing
        if strategy == ProcessingStrategy.MINIMAL:
            # Just apply basic Otsu thresholding
            _, otsu = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["otsu"] = otsu
            processed_images.append("otsu")
        
        elif strategy == ProcessingStrategy.DOCUMENT:
            # Optimized for document images
            self._preprocess_document(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.RECEIPT:
            # Optimized for receipts
            self._preprocess_receipt(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.ID_CARD:
            # Optimized for ID cards
            self._preprocess_id_card(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.NATURAL:
            # Optimized for natural scenes
            self._preprocess_natural(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.HANDWRITTEN:
            # Optimized for handwritten text
            self._preprocess_handwritten(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.BOOK:
            # Optimized for book pages
            self._preprocess_book(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.TABLE:
            # Optimized for tables
            self._preprocess_table(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.MULTI_COLUMN:
            # Optimized for multi-column layouts like newspapers
            self._preprocess_multi_column(base_image, image_data, processed_images, width, height)
        
        elif strategy == ProcessingStrategy.SCIENTIFIC:
            # Optimized for scientific documents with formulas
            self._preprocess_scientific(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.FORM:
            # Optimized for forms
            self._preprocess_form(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.SIGNAGE:
            # Optimized for outdoor signs and banners
            self._preprocess_signage(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.STANDARD:
            # Standard processing for general cases
            self._preprocess_standard(base_image, image_data, processed_images)
        
        elif strategy == ProcessingStrategy.AGGRESSIVE:
            # Aggressive processing for difficult images
            self._preprocess_aggressive(base_image, image_data, processed_images)
        
        # Check for glare in the image (common in smart glasses captures)
        # This will automatically address glare in all strategies if present
        if self._has_glare(base_image):
            logger.info("Glare detected, applying advanced glare reduction")
            glare_reduced = self._advanced_glare_reduction(base_image)
            image_data["glare_reduced"] = glare_reduced
            processed_images.append("glare_reduced")
            
            # Apply Otsu on glare reduced image
            _, glare_otsu = cv2.threshold(glare_reduced, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["glare_otsu"] = glare_otsu
            processed_images.append("glare_otsu")
        
        # Always add original image as final option
        image_data["original"] = image
        processed_images.append("original")
        
        return processed_images, image_data
    
    # Add the new preprocessing method for signage
    def _preprocess_signage(self, base_image, image_data, processed_images):
        """
        Optimized preprocessing for outdoor signage and banners
        
        Args:
            base_image: Base image to process
            image_data: Dictionary to store processed images
            processed_images: List to track processing methods
        """
        # 1. Apply adaptive histogram equalization for better contrast in varying lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast_enhanced"] = contrast_enhanced
        processed_images.append("contrast_enhanced")
        
        # 2. Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
        image_data["bilateral"] = bilateral
        processed_images.append("bilateral")
        
        # 3. Apply adaptive thresholding to handle uneven lighting
        adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 8)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # 4. Try Otsu thresholding as well
        _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # 5. Special processing for reflective/glare conditions
        # Detect bright spots that might be glare
        _, bright_mask = cv2.threshold(base_image, 220, 255, cv2.THRESH_BINARY)
        
        # If glare is detected, apply special processing
        if np.count_nonzero(bright_mask) > 0.05 * base_image.size:
            # Create a mask of glare areas (dilated to cover full glare area)
            kernel = np.ones((7, 7), np.uint8)
            glare_mask = cv2.dilate(bright_mask, kernel, iterations=2)
            
            # Use inpainting to remove glare
            glare_reduced = cv2.inpaint(base_image, glare_mask, 5, cv2.INPAINT_TELEA)
            image_data["glare_reduced"] = glare_reduced
            processed_images.append("glare_reduced")
            
            # Apply thresholding on glare-reduced image
            _, glare_otsu = cv2.threshold(glare_reduced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["glare_otsu"] = glare_otsu
            processed_images.append("glare_otsu")
        
        # 6. Try morphological operations to enhance text
        kernel = np.ones((2, 2), np.uint8)
        morph_open = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        image_data["morph_open"] = morph_open
        processed_images.append("morph_open")
        
        # 7. Edge enhancement for clearer text boundaries
        edges = cv2.Canny(bilateral, 30, 130)
        edge_dilated = cv2.dilate(edges, kernel, iterations=1)
        # Convert edges to float32 to avoid overflow in addWeighted
        edges_float = edge_dilated.astype(np.float32)
        edge_enhanced = cv2.addWeighted(base_image, 0.7, edges_float, 0.3, 0).astype(np.uint8)
        image_data["edge_enhanced"] = edge_enhanced
        processed_images.append("edge_enhanced")
        
        # 8. Try perspective correction if there's a clear rectangular shape
        corrected = self._try_perspective_correction(base_image)
        if corrected is not None:
            image_data["perspective_corrected"] = corrected
            processed_images.append("perspective_corrected")
            
            # Apply Otsu to the corrected image
            _, corrected_otsu = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_data["corrected_otsu"] = corrected_otsu
            processed_images.append("corrected_otsu")
    
    def _try_perspective_correction(self, image):
        """
        Try to correct perspective distortion in an image
        
        Args:
            image: Image to correct
            
        Returns:
            Corrected image or None if correction failed
        """
        try:
            # Find contours in the image
            edges = cv2.Canny(image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if it's large enough to be a sign
            image_area = image.shape[0] * image.shape[1]
            if cv2.contourArea(largest_contour) < 0.2 * image_area:
                return None
                
            # Approximate the contour to find corners
            peri = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
            
            # If it has 4 corners, it might be a rectangular sign
            if len(approx) == 4:
                # Convert points to right format
                pts = np.array([point[0] for point in approx], dtype=np.float32)
                
                # Sort points (top-left, top-right, bottom-right, bottom-left)
                s = pts.sum(axis=1)
                rect = np.zeros((4, 2), dtype=np.float32)
                rect[0] = pts[np.argmin(s)]  # Top-left: smallest sum
                rect[2] = pts[np.argmax(s)]  # Bottom-right: largest sum
                
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]  # Top-right: smallest difference
                rect[3] = pts[np.argmax(diff)]  # Bottom-left: largest difference
                
                # Calculate width and height of the destination image
                width_a = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
                width_b = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
                max_width = max(int(width_a), int(width_b))
                
                height_a = np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
                height_b = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))
                max_height = max(int(height_a), int(height_b))
                
                # Set destination points
                dst = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype=np.float32)
                
                # Calculate perspective transform matrix
                M = cv2.getPerspectiveTransform(rect, dst)
                
                # Apply perspective transform
                warped = cv2.warpPerspective(image, M, (max_width, max_height))
                return warped
            
            return None
            
        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}")
            return None
    
    def _detect_text_regions(self, gray_image) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential text regions with enhanced methodology
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of (x, y, w, h) tuples for potential text regions
        """
        # Apply MSER (Maximally Stable Extremal Regions) for text detection
        # This is more accurate than simple binary thresholding for text regions
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)
            
            text_regions = []
            for region in regions:
                # Get bounding box for each region
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter regions by size and aspect ratio to find likely text areas
                area = w * h
                if (area > 100 and area < 10000 and 0.1 < w/h < 10 and
                    h > 8 and w > 8):  # Minimum size for text
                    text_regions.append((x, y, w, h))
            
            # Merge overlapping regions to get text lines
            if text_regions:
                text_regions = self._merge_overlapping_regions(text_regions)
                
            return text_regions
            
        except Exception as e:
            logger.warning(f"MSER detection failed: {e}, falling back to threshold-based detection")
            
            # Fallback to threshold-based detection
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Perform morphological operations to separate text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to identify potential text regions
            text_regions = []
            min_area = 100  # Minimum area to consider
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter based on size and aspect ratio
                if area > min_area and 0.1 < w/h < 15:
                    text_regions.append((x, y, w, h))
            
            return text_regions
    
    def _merge_overlapping_regions(self, regions):
        """
        Merge overlapping text regions to get text lines/paragraphs
        
        Args:
            regions: List of (x, y, w, h) tuples
            
        Returns:
            List of merged (x, y, w, h) tuples
        """
        if not regions:
            return []
        
        # Sort regions by y-coordinate
        sorted_regions = sorted(regions, key=lambda r: r[1])
        
        merged_regions = []
        current_group = [sorted_regions[0]]
        current_y = sorted_regions[0][1]
        
        for i in range(1, len(sorted_regions)):
            region = sorted_regions[i]
            # If region is close to the current line (vertically)
            if abs(region[1] - current_y) < region[3] * 0.5:
                current_group.append(region)
            else:
                # Process the current group - merge horizontally close regions
                merged_line = self._merge_line_regions(current_group)
                merged_regions.extend(merged_line)
                
                # Start a new group
                current_group = [region]
                current_y = region[1]
        
        # Process the last group
        if current_group:
            merged_line = self._merge_line_regions(current_group)
            merged_regions.extend(merged_line)
        
        return merged_regions
    
    def _merge_line_regions(self, line_regions):
        """
        Merge horizontally close regions in a line
        
        Args:
            line_regions: List of regions in approximately the same line
            
        Returns:
            List of merged regions
        """
        if not line_regions:
            return []
        
        # Sort regions horizontally
        sorted_line = sorted(line_regions, key=lambda r: r[0])
        
        merged_line = []
        current_region = list(sorted_line[0])
        
        for i in range(1, len(sorted_line)):
            region = sorted_line[i]
            # If regions are horizontally close
            if region[0] <= current_region[0] + current_region[2] + 10:
                # Merge regions
                x = min(current_region[0], region[0])
                y = min(current_region[1], region[1])
                w = max(current_region[0] + current_region[2], region[0] + region[2]) - x
                h = max(current_region[1] + current_region[3], region[1] + region[3]) - y
                current_region = [x, y, w, h]
            else:
                merged_line.append(tuple(current_region))
                current_region = list(region)
        
        # Add the last region
        merged_line.append(tuple(current_region))
        
        return merged_line
    
    def _calculate_text_confidence(self, gray_image, edges) -> float:
        """
        Calculate confidence that the image contains text based on edge patterns
        
        Args:
            gray_image: Grayscale image
            edges: Edge-detected image
            
        Returns:
            Confidence score (0-100)
        """
        # Simple version - calculate edge patterns typical for text
        height, width = gray_image.shape
        
        # Text has a specific ratio of horizontal to vertical edges
        kernel_h = np.ones((1, 5), np.uint8)
        kernel_v = np.ones((5, 1), np.uint8)
        
        # Erode the edges to find horizontal and vertical components
        horizontal = cv2.erode(edges, kernel_h, iterations=1)
        vertical = cv2.erode(edges, kernel_v, iterations=1)
        
        # Count pixels
        h_pixels = np.count_nonzero(horizontal)
        v_pixels = np.count_nonzero(vertical)
        total_edge_pixels = np.count_nonzero(edges)
        
        if total_edge_pixels == 0:
            return 0.0
        
        # Text typically has a balanced ratio of horizontal to vertical edges
        # with more horizontal than vertical in Latin-based scripts
        h_v_ratio = h_pixels / (v_pixels + 1)  # Add 1 to avoid division by zero
        
        # Good text range is around 1.2 to 2.5 for h/v ratio
        ratio_score = 0.0
        if 1.0 <= h_v_ratio <= 3.0:
            # Optimal range
            ratio_score = 100.0
        elif 0.5 <= h_v_ratio < 1.0 or 3.0 < h_v_ratio <= 5.0:
            # Less optimal but still possibly text
            ratio_score = 60.0
        else:
            # Unlikely to be text
            ratio_score = 30.0
        
        # Edge density - text typically has a specific range of edge density
        edge_density = total_edge_pixels / (width * height)
        
        density_score = 0.0
        if 0.02 <= edge_density <= 0.15:
            # Optimal text density
            density_score = 100.0
        elif 0.01 <= edge_density < 0.02 or 0.15 < edge_density <= 0.25:
            # Less optimal but possible
            density_score = 60.0
        else:
            # Either too sparse or too dense for typical text
            density_score = 30.0
        
        # Combine scores - give more weight to ratio which is more text-specific
        # Weighted average
        confidence = (ratio_score * 0.6) + (density_score * 0.4)
        
        return min(100.0, confidence)
    
    def _detect_table_likelihood(self, gray_image, edges) -> float:
        """
        Detect the likelihood that the image contains tables
        
        Args:
            gray_image: Grayscale image
            edges: Edge-detected image
            
        Returns:
            Likelihood score (0-100)
        """
        # Detect horizontal and vertical lines which are typical for tables
        height, width = gray_image.shape
        
        # Define kernels for horizontal and vertical lines
        kernel_h = np.ones((1, int(width/30)), np.uint8)  # Horizontal kernel
        kernel_v = np.ones((int(height/30), 1), np.uint8)  # Vertical kernel
        
        # Use morphology to find lines
        # Assuming white text on black background after edge detection
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.bitwise_or(horizontal, vertical)
        
        # Count pixels to find the extent of potential table structure
        h_pixels = np.count_nonzero(horizontal)
        v_pixels = np.count_nonzero(vertical)
        
        # Calculate the density of lines
        h_density = h_pixels / (width * height)
        v_density = v_pixels / (width * height)
        
        # A good table should have a balanced distribution of horizontal and vertical lines
        # and the density should be within a certain range
        
        # Check the balance between horizontal and vertical lines
        if h_pixels == 0 or v_pixels == 0:
            balance_score = 0.0  # No lines in one direction means no table
        else:
            # Calculate how balanced the lines are (closer to 1.0 is more balanced)
            balance_ratio = h_pixels / v_pixels if h_pixels <= v_pixels else v_pixels / h_pixels
            balance_score = balance_ratio * 100.0
        
        # Check the density
        density_score = 0.0
        combined_density = h_density + v_density
        
        if 0.002 <= combined_density <= 0.05:
            # Optimal range for tables
            density_score = 100.0
        elif combined_density < 0.002:
            # Too few lines
            density_score = max(0, combined_density * 50000)  # Scale up to 100
        elif combined_density > 0.05:
            # Too many lines, might be a dense document or noise
            density_score = max(0, 100 - (combined_density - 0.05) * 2000)
        
        # Check for intersections of horizontal and vertical lines
        # Tables typically have intersections at cell corners
        intersections = cv2.bitwise_and(horizontal, vertical)
        intersection_count = np.count_nonzero(intersections)
        
        # More intersections usually means more likelihood of a table
        intersection_score = min(100.0, intersection_count / 5.0)
        
        # Combine the scores with appropriate weights
        likelihood = (
            (balance_score * 0.3) + 
            (density_score * 0.4) + 
            (intersection_score * 0.3)
        )
        
        return min(100.0, likelihood)
    
    def _detect_form_likelihood(self, gray_image, text_regions) -> float:
        """
        Detect the likelihood that the image contains a form
        
        Args:
            gray_image: Grayscale image
            text_regions: Detected text regions
            
        Returns:
            Likelihood score (0-100)
        """
        height, width = gray_image.shape
        
        # Forms typically have:
        # 1. Text regions aligned in a structured way
        # 2. Boxes or lines for input fields
        # 3. Labels followed by blank spaces or underlines
        
        # Use binary thresholding to detect boxes and lines
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detect horizontal lines (typically used in forms for input fields)
        kernel_h = np.ones((1, 20), np.uint8)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        
        # Detect rectangular boxes (common in forms)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular and square contours
        rect_count = 0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            if peri > 100:  # Ignore very small contours
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:  # Rectangle has 4 corners
                    rect_count += 1
        
        # Calculate horizontal line density
        h_pixels = np.count_nonzero(horizontal)
        h_density = h_pixels / (width * height)
        
        # Forms usually have a certain density of horizontal lines
        line_score = 0.0
        if 0.001 <= h_density <= 0.02:
            line_score = 100.0 * (h_density / 0.02)
        elif h_density > 0.02:
            line_score = 100.0 - min(100.0, (h_density - 0.02) * 5000)
        
        # Analyze text region alignment
        alignment_score = 0.0
        if len(text_regions) > 5:
            # Collect x-coordinates for potential label alignment
            x_coords = [region[0] for region in text_regions]
            
            # Check for consistent alignment (common in forms where labels align)
            from collections import Counter
            coord_counter = Counter([coord // 10 * 10 for coord in x_coords])  # Group within 10px
            
            # Find the most common x-coordinate (potential label column)
            if coord_counter:
                most_common_count = coord_counter.most_common(1)[0][1]
                alignment_ratio = most_common_count / len(text_regions)
                
                # Higher ratio means more aligned text regions
                alignment_score = alignment_ratio * 100.0
        
        # Check for box presence (typical in forms)
        box_score = min(100.0, rect_count * 10.0)
        
        # Combine scores - weighing alignment and boxes more as they're more form-specific
        likelihood = (
            (line_score * 0.3) + 
            (alignment_score * 0.4) + 
            (box_score * 0.3)
        )
        
        return min(100.0, likelihood)
    
    def _determine_image_type(self, width, height, brightness, 
                           contrast, blur, edge_density, 
                           text_regions, aspect_ratio, color_variance,
                           table_likelihood, form_likelihood,
                           text_confidence) -> ImageType:
        """
        Determine the type of image with enhanced classification
        
        Args:
            Various image characteristics
            
        Returns:
            ImageType enum representing the image type
        """
        # Define weighted characteristics for image types
        # Each characteristic has a weight and a range of values for each image type
        
        # Calculate scores for each image type
        scores = {}
        
        # Check if the image is blurry or low quality
        if blur < 100:
            scores[ImageType.LOW_QUALITY] = 100
        else:
            scores[ImageType.LOW_QUALITY] = max(0, 100 - blur/10)
        
        # Check for ID card (specific aspect ratio, multiple text regions)
        id_card_score = 0
        if 1.4 < aspect_ratio < 1.8 and 4 <= len(text_regions) <= 15:
            id_card_score = 80
            # Additional check for ID-like layout: header + multiple fields
            if form_likelihood > 50:
                id_card_score += 20
        scores[ImageType.ID_CARD] = id_card_score
        
        # Check for receipt (tall and narrow)
        receipt_score = 0
        if aspect_ratio < 0.6 and len(text_regions) > 5:
            receipt_score = 70
            # Additional check for typical receipt layout with aligned prices
            if form_likelihood > 30 and text_confidence > 60:
                receipt_score += 30
        scores[ImageType.RECEIPT] = receipt_score
        
        # Check for document
        document_score = 0
        if edge_density > 0.04 and contrast > 40 and blur > 300:
            document_score = 60
            if text_confidence > 70:
                document_score += 20
            # Check for paragraph structure
            if len(text_regions) > 10:
                document_score += 20
        scores[ImageType.DOCUMENT] = document_score
        
        # Check for high contrast document
        high_contrast_score = 0
        if contrast > 70 and brightness > 180 and edge_density > 0.04:
            high_contrast_score = 80
            if text_confidence > 80:
                high_contrast_score += 20
        scores[ImageType.HIGH_CONTRAST] = high_contrast_score
        
        # Check for signage
        signage_score = 0
        if (width > 2*height or height > 2*width) and edge_density < 0.1 and contrast > 50:
            signage_score = 70
            # Signs typically have large text with high contrast
            if text_confidence > 70 and len(text_regions) < 5 and color_variance > 20:
                signage_score += 30
        scores[ImageType.SIGNAGE] = signage_score
        
        # Check for handwritten text
        handwritten_score = 0
        if 0.02 < edge_density < 0.06 and 20 < contrast < 60:
            handwritten_score = 60
            # Handwritten text typically has more irregular edges
            if blur < 300 and text_confidence < 70:
                handwritten_score += 40
        scores[ImageType.HANDWRITTEN] = handwritten_score
        
        # Check for natural scene
        natural_score = 0
        if edge_density < 0.04 and contrast > 30:
            natural_score = 60
            # Natural scenes typically have higher color variance
            if color_variance > 30 and text_confidence < 60:
                natural_score += 40
        scores[ImageType.NATURAL] = natural_score
        
        # Check for form
        form_score = form_likelihood
        scores[ImageType.FORM] = form_score
        
        # Check for book page
        book_page_score = 0
        if 0.65 < aspect_ratio < 0.85 and edge_density > 0.05 and text_confidence > 70:
            book_page_score = 80
            # Book pages typically have dense, aligned text
            if len(text_regions) > 15:
                book_page_score += 20
        scores[ImageType.BOOK_PAGE] = book_page_score
        
        # Check for scientific document
        scientific_score = 0
        if edge_density > 0.05 and contrast > 50:
            # Scientific documents often have formulas, diagrams, and tables
            if table_likelihood > 40:
                scientific_score += 40
            # Look for potential formula patterns
            if text_confidence > 60 and len(text_regions) > 5:
                scientific_score += 30
        scores[ImageType.SCIENTIFIC] = scientific_score
        
        # Check for presentation
        presentation_score = 0
        if brightness > 200 and contrast > 60:
            presentation_score = 40
            # Presentations often have large text with high contrast
            if len(text_regions) < 10 and text_confidence > 70:
                presentation_score += 30
            # Presentations often have a distinct aspect ratio
            if 1.2 < aspect_ratio < 1.8:
                presentation_score += 30
        scores[ImageType.PRESENTATION] = presentation_score
        
        # Check for newspaper
        newspaper_score = 0
        if edge_density > 0.06 and contrast > 50:
            newspaper_score = 50
            # Newspapers often have multiple columns
            if len(text_regions) > 20:
                newspaper_score += 30
            # Check for multi-column layout
            text_x_positions = [region[0] for region in text_regions]
            if text_x_positions:
                x_positions_set = set([x // 50 for x in text_x_positions])  # Group by 50px
                if len(x_positions_set) >= 3:  # Multiple columns
                    newspaper_score += 20
        scores[ImageType.NEWSPAPER] = newspaper_score
        
        # Check for table structure
        table_score = table_likelihood
        scores[ImageType.TABLE] = table_score
        
        # Get the highest scoring image type
        best_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # If two types have very close scores, favor the more specific type
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0][1] - sorted_scores[1][1] < 10:
            # If scores are close, choose more specific type
            specific_types = [
                ImageType.ID_CARD, ImageType.RECEIPT, ImageType.SCIENTIFIC, 
                ImageType.FORM, ImageType.BOOK_PAGE, ImageType.NEWSPAPER,
                ImageType.TABLE
            ]
            
            if sorted_scores[1][0] in specific_types and sorted_scores[0][0] not in specific_types:
                best_type = sorted_scores[1][0]
        
        # Special case: if no type has a strong score, default to MIXED
        if scores[best_type] < 50:
            return ImageType.MIXED
        
        # Special case: if table score is very high, override other types
        if scores[ImageType.TABLE] > 70:
            return ImageType.TABLE
        
        return best_type
    
    def _resize_for_optimal_ocr(self, image, image_stats: ImageStats) -> np.ndarray:
        """Resize the image for optimal OCR performance"""
        height, width = image_stats.height, image_stats.width
        max_dimension = self.config.get("max_image_dimension", 3000)
        
        # For handwritten text, use a smaller max dimension
        if image_stats.image_type == ImageType.HANDWRITTEN:
            max_dimension = min(max_dimension, 1500)  # Limit size for handwritten images
        
        # If image is very large, scale it down more aggressively
        if width > max_dimension or height > max_dimension:
            scale_factor = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Return original if no resizing needed
        return image
    
    def _has_glare(self, image, threshold_percent=5) -> bool:
        """
        Check if the image has significant glare (bright areas)
        
        Args:
            image: Grayscale image
            threshold_percent: Percentage of bright pixels to consider as glare
            
        Returns:
            Boolean indicating presence of glare
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create mask for very bright regions
        _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of bright pixels
        bright_percent = (np.count_nonzero(bright_mask) / bright_mask.size) * 100
        
        return bright_percent > threshold_percent
    
    def _advanced_glare_reduction(self, image) -> np.ndarray:
        """
        Advanced glare reduction using multiple techniques
        
        Args:
            image: Grayscale image
            
        Returns:
            Image with reduced glare
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create a mask for bright regions (potential glare)
        _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Dilate the mask to cover glare areas fully
        kernel = np.ones((9,9), np.uint8)
        dilated_bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        
        # Method 1: Inpaint the glare regions
        try:
            inpainted = cv2.inpaint(gray, dilated_bright_mask, 5, cv2.INPAINT_TELEA)
            
            # Method 2: Blend with median filtered version for smooth transitions
            median_filtered = cv2.medianBlur(gray, 11)
            
            # Create a smoother blend mask
            blend_mask = cv2.GaussianBlur(dilated_bright_mask, (21, 21), 0) / 255.0
            
            # Combine methods based on the blend mask
            glare_reduced = (1 - blend_mask) * gray + blend_mask * inpainted
            
            # Convert to uint8
            glare_reduced = glare_reduced.astype(np.uint8)
            
            # Apply final contrast enhancement to recover details
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(glare_reduced)
        except:
            # Fallback to simpler method if inpainting fails
            mean_filter = cv2.blur(gray, (15, 15))
            glare_reduced = gray.copy()
            glare_reduced[dilated_bright_mask > 0] = mean_filter[dilated_bright_mask > 0]
            
            # Apply contrast enhancement to recover details
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(glare_reduced)
    
    # Strategy-specific preprocessing methods
    def _preprocess_document(self, base_image, image_data, processed_images):
        """Optimized preprocessing for document images"""
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Apply Otsu thresholding
        _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
        _, denoised_otsu = cv2.threshold(denoised, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["denoised_otsu"] = denoised_otsu
        processed_images.append("denoised_otsu")
    
    def _preprocess_receipt(self, base_image, image_data, processed_images):
        """Optimized preprocessing for receipts"""
        # High contrast adjustment
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Stronger adaptive thresholding
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 4)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Special receipt processing - line removal for better text recognition
        kernel_h = np.ones((1, 20), np.uint8)
        eroded_h = cv2.erode(adaptive, kernel_h, iterations=1)
        dilated_h = cv2.dilate(eroded_h, kernel_h, iterations=1)
        removed_lines = cv2.subtract(adaptive, dilated_h)
        image_data["removed_lines"] = removed_lines
        processed_images.append("removed_lines")
        
        # Deskew specifically for receipts which are often slightly tilted
        try:
            coords = np.column_stack(np.where(removed_lines > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Only deskew if there's a significant angle
                (h, w) = removed_lines.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                deskewed = cv2.warpAffine(removed_lines, M, (w, h), 
                                      flags=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_REPLICATE)
                image_data["deskewed"] = deskewed
                processed_images.append("deskewed")
        except:
            pass  # Skip deskewing if it fails
        
        # Add sharpened version for better text recognition
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["sharpened_thresh"] = sharp_thresh
        processed_images.append("sharpened_thresh")
    
    def _preprocess_id_card(self, base_image, image_data, processed_images):
        """Optimized preprocessing for ID cards"""
        # Noise reduction for small text
        bilateral = cv2.bilateralFilter(base_image, 9, 75, 75)
        image_data["bilateral"] = bilateral
        processed_images.append("bilateral")
        
        # Enhance contrast to make small text more readable
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(bilateral)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Apply Otsu thresholding
        _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # Apply adaptive thresholding - often better for variable lighting in ID cards
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Edge enhancement for better text detection in security features
        edges = cv2.Canny(contrast_enhanced, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        edge_enhanced = cv2.addWeighted(contrast_enhanced, 0.8, dilated_edges.astype(np.float32), 0.2, 0)
        image_data["edge_enhanced"] = edge_enhanced.astype(np.uint8)
        processed_images.append("edge_enhanced")
        
        # Add local adaptive thresholding with smaller window for fine details
        adaptive_small = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 7, 2)
        image_data["adaptive_small"] = adaptive_small
        processed_images.append("adaptive_small")
    
    def _preprocess_natural(self, base_image, image_data, processed_images):
        """Optimized preprocessing for natural scenes"""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Apply adaptive thresholding with larger block sizes for natural scenes
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 25, 15)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Apply bilateral filter to preserve edges while removing noise
        bilateral = cv2.bilateralFilter(contrast_enhanced, 11, 17, 17)
        image_data["bilateral"] = bilateral
        processed_images.append("bilateral")
        
        # Edge emphasizing for better text detection in natural scenes
        edges = cv2.Canny(contrast_enhanced, 30, 120)
        dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        try:
            dilated_edges_float = dilated_edges.astype(np.float32)
            contrast_enhanced_float = contrast_enhanced.astype(np.float32)
            edge_enhanced = cv2.addWeighted(contrast_enhanced_float, 0.7, dilated_edges_float, 0.3, 0)
            image_data["edge_enhanced"] = edge_enhanced.astype(np.uint8)
        except Exception as e:
            # Jika gagal, gunakan gambar asli saja
            image_data["edge_enhanced"] = contrast_enhanced
            logger.warning(f"Edge enhancement failed: {e}, using original image")
        
        # Shadow removal technique for outdoor scenes
        _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
        image_data["shadow_removed"] = closed
        processed_images.append("shadow_removed")
        
        # Additional processing for better text extraction
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        image_data["sharpened"] = sharpened
        processed_images.append("sharpened")
        
        # Otsu on sharpened image
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["sharp_thresh"] = sharp_thresh
        processed_images.append("sharp_thresh")
    
    def _preprocess_handwritten(self, base_image, image_data, processed_images):
        """Optimized preprocessing for handwritten text"""
        # Apply stronger bilateral filtering to smooth but preserve edges
        bilateral = cv2.bilateralFilter(base_image, 15, 40, 40)
        image_data["bilateral"] = bilateral
        processed_images.append("bilateral")
        
        # Increase contrast to make pen/pencil marks stand out
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(bilateral)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Modified adaptive thresholding for handwritten text
        # Use larger block size and higher constant for better noise filtering
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 21, 10)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Thin handwritten strokes to help with recognition
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(adaptive, kernel, iterations=1)
        image_data["thinned"] = eroded
        processed_images.append("thinned")
        
        # Edge enhancement for better stroke detection
        edges = cv2.Canny(contrast_enhanced, 30, 120)
        edge_enhanced = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
        edge_enhanced = 255 - edge_enhanced  # Invert for better visualization
        image_data["edge_enhanced"] = edge_enhanced
        processed_images.append("edge_enhanced")
        
        # Apply special morphological operations for handwritten text
        kernel_line = np.ones((1, 5), np.uint8)
        dilated_horiz = cv2.dilate(contrast_enhanced, kernel_line, iterations=1)
        dilated_horiz = cv2.erode(dilated_horiz, kernel_line, iterations=1)
        image_data["enhanced_strokes"] = dilated_horiz
        processed_images.append("enhanced_strokes")
    
    def _preprocess_book(self, base_image, image_data, processed_images):
        """Optimized preprocessing for book pages"""
        # Apply contrast enhancement for faded text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Apply denoising for typical book scan noise
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
        image_data["denoised"] = denoised
        processed_images.append("denoised")
        
        # Apply Otsu thresholding which works well for book pages
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # Remove page curvature shadows
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background = cv2.morphologyEx(denoised, cv2.MORPH_DILATE, morph_kernel)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        image_data["normalized"] = normalized
        processed_images.append("normalized")
        
        # Enhance text edges for better recognition
        edges = cv2.Canny(denoised, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edge_enhanced = cv2.addWeighted(denoised.astype(np.float32), 0.8, dilated_edges.astype(np.float32), 0.2, 0).astype(np.uint8)
        image_data["edge_enhanced"] = edge_enhanced.astype(np.uint8)
        processed_images.append("edge_enhanced")
        
        # Sharpening for clearer text
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["sharpened_thresh"] = sharp_thresh
        processed_images.append("sharpened_thresh")
    
    def _preprocess_table(self, base_image, image_data, processed_images):
        """Optimized preprocessing for tables"""
        # Use line detection and enhancement
        # Detect horizontal and vertical lines
        kernel_h = np.ones((1, 40), np.uint8)
        kernel_v = np.ones((40, 1), np.uint8)
        
        _, binary = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
        
        # Combine lines and dilate to get table structure
        table_structure = cv2.bitwise_or(horizontal, vertical)
        table_structure = cv2.dilate(table_structure, np.ones((3,3), np.uint8), iterations=1)
        
        # Invert for display and OCR
        table_structure = 255 - table_structure
        image_data["table_structure"] = table_structure
        processed_images.append("table_structure")
        
        # Get cells from table by removing grid lines
        cells = cv2.bitwise_and(255 - binary, table_structure)
        image_data["table_cells"] = cells
        processed_images.append("table_cells")
        
        # Apply adaptive thresholding for text in cells
        adaptive = cv2.adaptiveThreshold(base_image, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Apply enhanced processing for table text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        _, otsu = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # Special processing to isolate text in table cells
        dilated_lines = cv2.dilate(horizontal + vertical, np.ones((2,2), np.uint8), iterations=1)
        text_only = cv2.bitwise_and(otsu, cv2.bitwise_not(dilated_lines))
        image_data["text_only"] = text_only
        processed_images.append("text_only")
    
    def _preprocess_multi_column(self, base_image, image_data, processed_images, width=None, height=None):
        """
        Optimized preprocessing for multi-column layouts
        
        Args:
            base_image: Base image to process
            image_data: Dictionary to store processed images
            processed_images: List to track processing methods
            width: Optional width parameter (can be passed from caller)
            height: Optional height parameter (can be passed from caller)
        """
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        
        # Get image dimensions for processing (if not provided)
        if width is None or height is None:
            height, width = base_image.shape[:2]
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
        image_data["denoised"] = denoised
        processed_images.append("denoised")
        
        # Apply Otsu thresholding
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # Create an enhanced version for column detection
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(otsu, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        image_data["column_segmented"] = eroded
        processed_images.append("column_segmented")
        
        # Perform vertical projection to detect column boundaries
        vertical_projection = np.sum(otsu, axis=0) / 255
        
        # Calculate kernel size based on width
        kernel_size = max(3, int(width / 100))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        vertical_projection_smoothed = cv2.GaussianBlur(
            vertical_projection.reshape(-1, 1), (1, kernel_size), 0
        ).flatten()
        
        # Create visual representation of column detection
        column_visual = otsu.copy()
        threshold = np.mean(vertical_projection_smoothed) * 0.5
        
        # Mark detected column boundaries
        for i in range(1, len(vertical_projection_smoothed) - 1):
            if (vertical_projection_smoothed[i] < threshold and 
                vertical_projection_smoothed[i-1] > threshold):
                cv2.line(column_visual, (i, 0), (i, height), 127, 2)
        
        image_data["column_detected"] = column_visual
        processed_images.append("column_detected")
        
        # Enhanced processing for multi-column text
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
        )
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
    
    def _preprocess_scientific(self, base_image, image_data, processed_images):
        """Optimized preprocessing for scientific documents with formulas"""
        # Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Apply adaptive thresholding with careful parameters to preserve formula symbols
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 15, 8)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Special morphological operation to preserve small details in formulas
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        image_data["formula_preserved"] = opened
        processed_images.append("formula_preserved")
        
        # Enhanced edge detection for formula symbols
        edges = cv2.Canny(contrast_enhanced, 30, 130)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # Convert edges to float32 to avoid overflow in addWeighted
        edges_float = dilated_edges.astype(np.float32)
        symbol_enhanced = cv2.addWeighted(contrast_enhanced, 0.85, edges_float, 0.15, 0)
        image_data["symbol_enhanced"] = symbol_enhanced.astype(np.uint8)
        processed_images.append("symbol_enhanced")
        
        # Special processing for mathematical symbols
        # Sharpen the image to enhance small details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        _, sharp_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["sharp_thresh"] = sharp_thresh
        processed_images.append("sharp_thresh")
        
        # Special processing for subscripts and superscripts
        # Use a larger kernel for text and a smaller kernel for symbols
        text_kernel = np.ones((3, 3), np.uint8)
        text_cleaned = cv2.morphologyEx(sharp_thresh, cv2.MORPH_OPEN, text_kernel)
        symbol_kernel = np.ones((1, 1), np.uint8)
        symbol_only = cv2.subtract(sharp_thresh, text_cleaned)
        symbol_cleaned = cv2.morphologyEx(symbol_only, cv2.MORPH_OPEN, symbol_kernel)
        combined = cv2.bitwise_or(text_cleaned, symbol_cleaned)
        image_data["formula_enhanced"] = combined
        processed_images.append("formula_enhanced")
    
    def _preprocess_form(self, base_image, image_data, processed_images):
        """Optimized preprocessing for forms"""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 15, 5)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Find and enhance form fields
        kernel = np.ones((1, 20), np.uint8)
        horizontal = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((20, 1), np.uint8)
        vertical = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
        
        # Combine to get form structure
        form_structure = cv2.bitwise_or(horizontal, vertical)
        
        # Enhance form field boundaries
        form_enhanced = cv2.bitwise_and(adaptive, cv2.bitwise_not(form_structure))
        image_data["form_enhanced"] = form_enhanced
        processed_images.append("form_enhanced")
        
        # Special processing for form text
        # Create a mask of likely text areas
        _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_text = np.ones((3, 15), np.uint8)
        text_areas = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_text)
        text_only = cv2.bitwise_and(binary, text_areas)
        image_data["text_only"] = text_only
        processed_images.append("text_only")
        
        # Extract field labels with special processing
        # Labels are usually aligned and have consistent format
        labels_image = cv2.bitwise_xor(binary, form_enhanced)
        kernel_label = np.ones((1, 15), np.uint8)
        labels_processed = cv2.morphologyEx(labels_image, cv2.MORPH_CLOSE, kernel_label)
        image_data["labels"] = labels_processed
        processed_images.append("labels")
    
    def _preprocess_standard(self, base_image, image_data, processed_images):
        """Standard processing for general cases"""
        # Apply Otsu thresholding
        _, otsu = cv2.threshold(base_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["otsu"] = otsu
        processed_images.append("otsu")
        
        # Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(base_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        _, contrast_otsu = cv2.threshold(contrast_enhanced, 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["contrast_otsu"] = contrast_otsu
        processed_images.append("contrast_otsu")
        
        # Add edge enhancement for better text detection
        edges = cv2.Canny(base_image, 50, 150)
        dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        # Convert edges to float32 to avoid overflow in addWeighted
        edges_float = dilated_edges.astype(np.float32)
        edge_enhanced = cv2.addWeighted(base_image, 0.8, edges_float, 0.2, 0)
        image_data["edge_enhanced"] = edge_enhanced.astype(np.uint8)
        processed_images.append("edge_enhanced")
    
    def _preprocess_aggressive(self, base_image, image_data, processed_images):
        """Aggressive processing for difficult images"""
        # Super contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(base_image)
        image_data["contrast"] = contrast_enhanced
        processed_images.append("contrast")
        
        # Strong denoising
        denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 15, 9, 21)
        image_data["denoised"] = denoised
        processed_images.append("denoised")
        
        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        image_data["sharpened"] = sharpened
        processed_images.append("sharpened")
        
        # Otsu on sharpened
        _, sharpened_otsu = cv2.threshold(sharpened, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["sharpened_otsu"] = sharpened_otsu
        processed_images.append("sharpened_otsu")
        
        # Strong adaptive thresholding
        adaptive = cv2.adaptiveThreshold(sharpened, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 13, 5)
        image_data["adaptive"] = adaptive
        processed_images.append("adaptive")
        
        # Try inverting the image as sometimes it helps
        inverted = cv2.bitwise_not(base_image)
        _, inverted_otsu = cv2.threshold(inverted, 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["inverted_otsu"] = inverted_otsu
        processed_images.append("inverted_otsu")
        
        # Try histogram stretching
        stretched = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX)
        image_data["stretched"] = stretched
        processed_images.append("stretched")
        
        # Multi-scale processing - try different scales
        # Sometimes downscaling and then upscaling removes noise
        height, width = base_image.shape
        down_scale = cv2.resize(base_image, (width//2, height//2), interpolation=cv2.INTER_AREA)
        up_scale = cv2.resize(down_scale, (width, height), interpolation=cv2.INTER_CUBIC)
        _, multi_scale_otsu = cv2.threshold(up_scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["multi_scale_otsu"] = multi_scale_otsu
        processed_images.append("multi_scale_otsu")
        
        # Add local contrast enhancement with different parameters
        clahe_strong = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        strong_contrast = clahe_strong.apply(base_image)
        _, strong_otsu = cv2.threshold(strong_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data["strong_otsu"] = strong_otsu
        processed_images.append("strong_otsu")

    def determine_processing_strategy(self, image_stats):
        """
        Determine the best processing strategy based on image type and stats
        
        Args:
            image_stats: ImageStats object with image characteristics
            
        Returns:
            ProcessingStrategy enum for optimal processing
        """
        # Choose strategy based on image type
        image_type = image_stats.image_type
        
        # Use special strategies for specific image types
        if image_type == ImageType.DOCUMENT:
            return ProcessingStrategy.DOCUMENT
        elif image_type == ImageType.NATURAL:
            return ProcessingStrategy.NATURAL
        elif image_type == ImageType.RECEIPT:
            return ProcessingStrategy.RECEIPT
        elif image_type == ImageType.ID_CARD:
            return ProcessingStrategy.ID_CARD
        elif image_type == ImageType.HANDWRITTEN:
            return ProcessingStrategy.HANDWRITTEN
        elif image_type == ImageType.BOOK_PAGE:
            return ProcessingStrategy.BOOK
        elif image_type == ImageType.TABLE:
            return ProcessingStrategy.TABLE
        elif image_type == ImageType.SCIENTIFIC:
            return ProcessingStrategy.SCIENTIFIC
        elif image_type == ImageType.FORM:
            return ProcessingStrategy.FORM
        elif image_type == ImageType.NEWSPAPER:
            return ProcessingStrategy.MULTI_COLUMN
        elif image_type == ImageType.SIGNAGE:  # Add this condition for signage
            return ProcessingStrategy.SIGNAGE
        
        # For low quality images, use aggressive processing
        if image_type == ImageType.LOW_QUALITY:
            return ProcessingStrategy.AGGRESSIVE
        
        # Consider other image characteristics
        if image_stats.image_type == ImageType.HIGH_CONTRAST:
            # For high contrast images, minimal processing is often better
            return ProcessingStrategy.MINIMAL
        
        # Default to standard processing
        return ProcessingStrategy.STANDARD
    
    def auto_rotate(self, image):
        """
        Auto-rotate image based on orientation detection
        
        Args:
            image: OpenCV image
            
        Returns:
            Rotated image
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert to grayscale if needed
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use edge detection to find lines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width/4, maxLineGap=20)
            
            if lines is None or len(lines) == 0:
                return image  # No rotation needed
            
            # Compute angles of lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Avoid division by zero
                    continue
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                angles.append(angle)
            
            # Find dominant angle (horizontal or vertical lines)
            angle_counts = {}
            for angle in angles:
                # Normalize angle to 0-180 range
                norm_angle = angle % 180
                # Group similar angles (within 2 degrees)
                grouped_angle = round(norm_angle / 2) * 2
                angle_counts[grouped_angle] = angle_counts.get(grouped_angle, 0) + 1
            
            if not angle_counts:
                return image  # No valid angles found
            
            # Get most common angle
            dominant_angle = max(angle_counts.items(), key=lambda x: x[1])[0]
            
            # Convert to rotation angle (adjust to make horizontal/vertical)
            if 45 <= dominant_angle <= 135:
                # Closer to vertical
                rotation_angle = dominant_angle - 90
            else:
                # Closer to horizontal
                rotation_angle = dominant_angle
            
            # Only rotate if angle is significant
            if abs(rotation_angle) < 1:
                return image  # No significant rotation needed
            
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        except Exception as e:
            logger.warning(f"Auto-rotation failed: {e}")
            return image  # Return original if rotation fails