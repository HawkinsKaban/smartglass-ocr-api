#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data models and enums for SmartGlassOCR
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any, Set

# Define image type classification with enhanced specificity
class ImageType(Enum):
    DOCUMENT = "document"           # Clear document with structured text
    NATURAL = "natural"             # Natural scene with text
    SIGNAGE = "signage"             # Signs, banners, displays
    HANDWRITTEN = "handwritten"     # Handwritten text
    MIXED = "mixed"                 # Mixed content
    LOW_QUALITY = "low_quality"     # Blurry or low quality image
    HIGH_CONTRAST = "high_contrast" # High contrast image (black text on white)
    RECEIPT = "receipt"             # Receipt or ticket
    ID_CARD = "id_card"             # ID card or license
    SCIENTIFIC = "scientific"       # Scientific document with formulas
    PRESENTATION = "presentation"   # Slides or presentation material
    BOOK_PAGE = "book_page"         # Book or magazine page
    NEWSPAPER = "newspaper"         # Newspaper or article
    FORM = "form"                   # Form with fields and entries
    TABLE = "table"                 # Table with rows and columns

@dataclass
class ImageStats:
    """Statistical features of an image"""
    width: int
    height: int
    brightness: float
    contrast: float
    blur: float
    edge_density: float
    text_regions: int  # Number of potential text regions
    aspect_ratio: float
    image_type: ImageType
    # Added new metrics for better image analysis
    table_likelihood: float = 0.0
    form_likelihood: float = 0.0
    color_variance: float = 0.0
    text_confidence: float = 0.0

# Enhanced processing strategies with more specific options
class ProcessingStrategy(Enum):
    """Strategies for image processing"""
    MINIMAL = "minimal"             # Basic processing
    STANDARD = "standard"           # Standard processing
    AGGRESSIVE = "aggressive"       # Heavy processing for difficult images
    DOCUMENT = "document"           # Optimized for documents
    NATURAL = "natural"             # Optimized for natural scenes
    RECEIPT = "receipt"             # Optimized for receipts/tickets
    ID_CARD = "id_card"             # Optimized for ID cards
    BOOK = "book"                   # Optimized for book pages
    TABLE = "table"                 # Optimized for tables and structured data
    HANDWRITTEN = "handwritten"     # Optimized for handwritten text
    MULTI_COLUMN = "multi_column"   # Optimized for multi-column layouts
    SCIENTIFIC = "scientific"       # Optimized for scientific documents
    FORM = "form"                   # Optimized for forms
    SIGNAGE = "signage"             # Optimized for outdoor signs and banners

# Define document structure type for better analysis
class DocumentStructure(Enum):
    """Types of document structures for improved text organization"""
    PLAIN_TEXT = "plain_text"       # Simple flowing text
    PARAGRAPHS = "paragraphs"       # Text with paragraph breaks
    HEADERS_AND_CONTENT = "headers_and_content"  # Headers with content sections
    BULLET_POINTS = "bullet_points" # Lists with bullet points
    TABLE = "table"                 # Tabular data
    FORM = "form"                   # Form with fields
    MULTI_COLUMN = "multi_column"   # Multi-column layout
    SCIENTIFIC = "scientific"       # Scientific with formulas
    MIXED = "mixed"                 # Mixed structure types
    SIGNAGE = "signage"             # Signage and banner text
@dataclass
class TextRegion:
    """Represents a region of text in an image"""
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    confidence: float = 0.0
    type: str = "text"  # Can be "text", "title", "header", "list_item", etc.

@dataclass
class OCRResult:
    """Structured representation of OCR results"""
    text: str
    confidence: float
    engine: str
    regions: List[TextRegion] = None
    layout_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = []
        if self.layout_info is None:
            self.layout_info = {}