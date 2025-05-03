#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text processing functionality for SmartGlassOCR
Includes text correction, formatting, and analysis
"""

import re
import logging
import string
from typing import List, Dict, Tuple, Optional, Union, Any

from .model import ImageType, DocumentStructure

logger = logging.getLogger("SmartGlass-TextProcessing")

# Try to load NLP libraries - using minimal dependencies
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.probability import FreqDist
    
    # Download NLTK resources if needed
    nltk_resources = ['punkt', 'stopwords']
    for resource in nltk_resources:
        try:
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    
    NLTK_AVAILABLE = True
    
    # Load stopwords with better error handling
    try:
        from nltk.corpus import stopwords
        STOPWORDS_EN = set(stopwords.words('english'))
    except:
        STOPWORDS_EN = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                      "when", "where", "how", "which", "who", "whom", "this", "that", "these",
                      "those", "then", "just", "so", "than", "such", "both", "through", "about",
                      "for", "is", "of", "while", "during", "to", "from"}
        logger.warning("Failed to load English stopwords, using fallback")
    
    try:
        STOPWORDS_ID = set(stopwords.words('indonesian'))
    except:
        # Fallback stopwords for Indonesian
        STOPWORDS_ID = {'yang', 'dan', 'di', 'ini', 'itu', 'dari', 'dengan', 'untuk', 'pada', 'adalah',
                        'ke', 'tidak', 'ada', 'oleh', 'juga', 'akan', 'bisa', 'dalam', 'saya', 'kamu', 
                        'kami', 'mereka', 'dia', 'nya', 'tersebut', 'dapat', 'sebagai', 'telah', 'bahwa',
                        'atau', 'jika', 'maka', 'sudah', 'saat', 'ketika', 'karena'}
        logger.warning("Failed to load Indonesian stopwords, using fallback")
    
    # Combined stopwords
    STOPWORDS = STOPWORDS_EN.union(STOPWORDS_ID)
    
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS = set()
    logger.warning("NLTK libraries not available, using simplified text processing")

class TextProcessor:
    """Text processor with enhanced correction and formatting"""
    
    def __init__(self, config=None):
        """
        Initialize the text processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def post_process_text(text: str, image_type) -> str:
        """
        Apply rule-based post-processing to extracted text
        
        Args:
            text: Extracted text
            image_type: Type of image
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Remove invalid unicode characters
        text = ''.join(c for c in text if ord(c) < 65536)
        
        # Fix quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("''", '"').replace(",,", '"')
        text = text.replace("'", "'").replace("`", "'")
        
        # Fix bullet points and normalize them
        text = re.sub(r'[\*\+\-‣▪•●·](?:\s+|\n)', '• ', text)
        
        # Fix common OCR letter confusions
        text = re.sub(r'(?<=\d)l(?=\d)', '1', text)  # Digit + l + digit → 1
        text = re.sub(r'(?<=\d)I(?=\d)', '1', text)  # Digit + I + digit → 1
        text = re.sub(r'(?<=\d)O(?=\d)', '0', text)  # Digit + O + digit → 0
        text = re.sub(r'(?<=\d)S(?=\d)', '5', text)  # Digit + S + digit → 5
        text = re.sub(r'(?<=\d)Z(?=\d)', '2', text)  # Digit + Z + digit → 2
        text = re.sub(r'(?<=\d)B(?=\d)', '8', text)  # Digit + B + digit → 8
        
        # Fix space issues
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # No space between word and capital letter
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)  # No space between letter and digit
        text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)  # No space between digit and letter
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.!?,:;])([A-Z0-9])', r'\1 \2', text)
        
        # Fix common merged words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix newlines - remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphens at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Specific processing based on image type
        if hasattr(image_type, 'value'):
            image_type_value = image_type.value.lower()
            if 'receipt' in image_type_value:
                # Apply receipt-specific corrections
                text = _fix_receipt_text(text)
            elif 'id_card' in image_type_value:
                # Apply ID card-specific corrections
                text = _fix_id_card_text(text)
            elif 'form' in image_type_value:
                # Apply form-specific corrections
                text = _fix_form_text(text)
            elif 'table' in image_type_value:
                # Apply table-specific corrections
                text = _fix_table_text(text)
        
        return text.strip()
    
    def _fix_receipt_text(self, text: str) -> str:
        """
        Fix common OCR errors in receipts with enhanced corrections
        
        Args:
            text: Receipt text
            
        Returns:
            Corrected text
        """
        # Fix currency symbols and amounts
        text = re.sub(r'([0-9]+)\.([0-9]{2})([^0-9])', r'$\1.\2\3', text)
        
        # Fix percentage signs
        text = re.sub(r'([0-9]+)[,.]([0-9]+)o\/?', r'\1.\2%', text)
        
        # Fix common receipt words
        replacements = {
            r'\bTOTAI\b': 'TOTAL',
            r'\bSUBTOTAI\b': 'SUBTOTAL',
            r'\bCASI-I\b': 'CASH',
            r'\bCHANGI\b': 'CHANGE',
            r'\bDISCOUNI\b': 'DISCOUNT',
            r'\bITEMS\b': 'ITEMS',
            r'\bTAX\b': 'TAX',
            r'\bDUE\b': 'DUE',
            r'\bDATE\b': 'DATE',
            r'\bTIME\b': 'TIME',
            r'\bTHANI< YOU\b': 'THANK YOU',
            r'\bTHANKS\b': 'THANKS',
            r'\bCARD\b': 'CARD',
            r'\bCASHIER\b': 'CASHIER',
            r'\bINVOICE\b': 'INVOICE',
            r'\bNO\.\b': 'NO.',
            r'\bDESCRIPTION\b': 'DESCRIPTION',
            r'\bQTY\b': 'QTY',
            r'\bPRICE\b': 'PRICE',
            r'\bAMOUNT\b': 'AMOUNT',
            r'\bDISCOUNT\b': 'DISCOUNT',
            r'\bSUBTOTAL\b': 'SUBTOTAL',
            r'\bTAX\b': 'TAX',
            r'\bTOTAL\b': 'TOTAL',
            r'\bPMT\b': 'PAYMENT',
            r'\bVAT\b': 'VAT',
            r'\bCASH\b': 'CASH',
            r'\bCARD\b': 'CARD',
            r'\bDEBIT\b': 'DEBIT',
            r'\bCREDIT\b': 'CREDIT',
            r'\bTHANK YOU\b': 'THANK YOU'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Enhanced detection of receipt items
        lines = text.split('\n')
        formatted_lines = []
        
        # Flag to identify item sections
        in_item_section = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                formatted_lines.append("")
                continue
            
            # Check for section headers
            if re.match(r'^(ITEM|DESCRIPTION|PRODUCT|GOODS)S?', line, re.IGNORECASE):
                in_item_section = True
                formatted_lines.append(line)
                continue
            
            # Check for end of item section
            if in_item_section and re.match(r'^(SUBTOTAL|TOTAL|TAX|DISCOUNT)', line, re.IGNORECASE):
                in_item_section = False
            
            # Format receipt items if in item section
            if in_item_section:
                # Try to detect and format item with quantity and price
                item_match = re.search(r'^(.+?)(?:\s+(\d+))?(?:\s+(?:x|@)\s+)?([0-9.,]+)', line)
                
                if item_match:
                    item_name = item_match.group(1).strip()
                    quantity = item_match.group(2) or "1"
                    price = item_match.group(3).strip()
                    
                    formatted_line = f"{item_name}: {quantity} x ${price}"
                    formatted_lines.append(formatted_line)
                else:
                    formatted_lines.append(line)
            else:
                # Regular line
                formatted_lines.append(line)
        
        # Rejoin lines
        text = '\n'.join(formatted_lines)
        
        # Format total, subtotal, tax lines
        text = re.sub(r'(?i)subtotal\s*[:,]?\s*[$]?([0-9.,]+)', r'SUBTOTAL: $\1', text)
        text = re.sub(r'(?i)tax\s*[:,]?\s*[$]?([0-9.,]+)', r'TAX: $\1', text)
        text = re.sub(r'(?i)total\s*[:,]?\s*[$]?([0-9.,]+)', r'TOTAL: $\1', text)
        
        return text
    
    def _fix_id_card_text(self, text: str) -> str:
        """
        Fix common OCR errors in ID cards with enhanced corrections
        
        Args:
            text: ID card text
            
        Returns:
            Corrected text
        """
        # Fix common ID card fields with enhanced patterns
        replacements = {
            r'\bNAME\b': 'NAME',
            r'\bADDRESS\b': 'ADDRESS',
            r'\bDOB\b': 'DATE OF BIRTH',
            r'\bDATI?E? OF BIRTH\b': 'DATE OF BIRTH',
            r'\bEXP(?:IRATION)? ?DATI?E?\b': 'EXPIRATION DATE',
            r'\bSEX\b': 'SEX',
            r'\bGENDER\b': 'GENDER',
            r'\bHEIGH[TI]\b': 'HEIGHT',
            r'\bWEIGH[TI]\b': 'WEIGHT',
            r'\bEYES?\b': 'EYES',
            r'\bHAIR\b': 'HAIR',
            r"\b(?:DRIVER'?S|DRIV|ORIV) LIC[, ]?(?:NO|NUM|NUMBER)\b": 'DRIVER\'S LICENSE NO',
            r'\bISSUE(?:D)? DATE\b': 'ISSUE DATE',
            r'\bIDENTITY(?:\s+CARD)?\b': 'IDENTITY CARD',
            r'\bID(?:\s+CARD)?\b': 'ID CARD',
            r'\bCITIZEN\b': 'CITIZEN',
            r'\bNATIONALITY\b': 'NATIONALITY',
            r'\bPLACE OF BIRTH\b': 'PLACE OF BIRTH',
            r'\bRELIGION\b': 'RELIGION',
            r'\bMARITAL STATUS\b': 'MARITAL STATUS',
            r'\bBLOOD(?: TYPE)?\b': 'BLOOD TYPE',
            r'\bOCCUPATION\b': 'OCCUPATION',
            r'\bSIGNATURE\b': 'SIGNATURE'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Format common ID fields with colon for better readability
        id_fields = [
            'NAME', 'ADDRESS', 'DATE OF BIRTH', 'EXPIRATION DATE', 'SEX', 'GENDER',
            'HEIGHT', 'WEIGHT', 'EYES', 'HAIR', 'DRIVER\'S LICENSE NO', 'ISSUE DATE',
            'PLACE OF BIRTH', 'NATIONALITY', 'RELIGION', 'MARITAL STATUS', 'BLOOD TYPE',
            'OCCUPATION'
        ]
        
        for field in id_fields:
            # Add colon if field exists but doesn't have one
            pattern = f'({field})\\s+([^:\\n]+)'
            replacement = r'\1: \2'
            text = re.sub(pattern, replacement, text)
        
        # Format dates correctly (various formats)
        date_patterns = [
            # MM/DD/YYYY or similar
            (r'(\d{1,2})[/\-\.\\](\d{1,2})[/\-\.\\](\d{2,4})', r'\1/\2/\3'),
            # MMDDYYYY
            (r'(\d{2})(\d{2})(\d{4})', r'\1/\2/\3'),
            # DD-MMM-YYYY
            (r'(\d{1,2})[-\s](JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[-\s](\d{2,4})', 
             r'\1 \2 \3')
        ]
        
        for pattern, replacement in date_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Correct ID numbers (format various ID numbers)
        # Remove unwanted spaces and characters in ID numbers
        id_pattern = r'(?:ID|NUMBER|LICENSE|NO|#)[:.\s-]*([A-Z0-9][\sA-Z0-9\-\.]{4,})'
        matches = re.finditer(id_pattern, text, re.IGNORECASE)
        
        for match in matches:
            raw_id = match.group(1)
            # Remove spaces and unwanted characters
            clean_id = re.sub(r'[^A-Z0-9]', '', raw_id, flags=re.IGNORECASE)
            # Format based on length
            if len(clean_id) in [9, 10]:
                # Common ID lengths - add hyphens for readability
                formatted_id = '-'.join([clean_id[:3], clean_id[3:6], clean_id[6:]])
                text = text.replace(raw_id, formatted_id)
        
        return text
    
    def _fix_scientific_text(self, text: str) -> str:
        """
        Fix scientific and mathematical notation
        
        Args:
            text: Scientific text
            
        Returns:
            Corrected text
        """
        # Fix common scientific notation errors
        # Fix superscripts
        text = re.sub(r'(\d)[\^](\d+)', r'\1\u00B2', text)  # x^2 -> x²
        text = re.sub(r'(\d)[\^]2', r'\1\u00B2', text)  # x^2 -> x²
        text = re.sub(r'(\d)[\^]3', r'\1\u00B3', text)  # x^3 -> x³
        
        # Fix subscripts
        text = re.sub(r'([A-Za-z])_(\d)', r'\1\u208\2', text)  # H_2O -> H₂O
        
        # Fix common scientific symbols
        replacements = {
            r'(?<=\d)x(?=\d)': '×',         # 2x3 -> 2×3
            'alpha': 'α',
            'beta': 'β',
            'gamma': 'γ',
            'delta': 'δ',
            'epsilon': 'ε',
            'theta': 'θ',
            'lambda': 'λ',
            'micro': 'µ',
            'pi': 'π',
            'sigma': 'σ',
            'Sigma': 'Σ',
            'tau': 'τ',
            'phi': 'φ',
            'omega': 'ω',
            'Omega': 'Ω',
            'approx': '≈',
            'neq': '≠',
            'leq': '≤',
            'geq': '≥',
            r'(?<!\w)inf(?!\w)': '∞',
            'sqrt': '√',
            'integral': '∫',
            'nabla': '∇',
            'union': '∪',
            'intersect': '∩',
            'in': '∈',
            'notin': '∉',
            'subset': '⊂',
            'superset': '⊃',
            'partial': '∂',
            'sum': '∑',
            'product': '∏',
            'deg(ree)?s?': '°',
            r'\+/-': '±',
            r'\(\+/-\)': '±'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix chemical formulas
        text = re.sub(r'([A-Z][a-z]?)(\d+)', r'\1\u208\2', text)  # CO2 -> CO₂
        
        # Fix common unit errors
        units = {
            r'([0-9]+)([^0-9\s]+[Cc])': r'\1 °C',  # Fix Celsius
            r'([0-9]+)([^0-9\s]+[Ff])': r'\1 °F',  # Fix Fahrenheit
            r'([0-9]+)([^0-9\s]+[Kk])': r'\1 K',   # Fix Kelvin
            r'([0-9]+)([^0-9\s]*)[Mm][Ll]': r'\1 ml',  # Fix milliliters
            r'([0-9]+)([^0-9\s]*)[Mm][Gg]': r'\1 mg',  # Fix milligrams
            r'([0-9]+)([^0-9\s]*)[Kk][Gg]': r'\1 kg',  # Fix kilograms
            r'([0-9]+)([^0-9\s]*)[Cc][Mm]': r'\1 cm',  # Fix centimeters
            r'([0-9]+)([^0-9\s]*)[Mm][Mm]': r'\1 mm',  # Fix millimeters
            r'([0-9]+)([^0-9\s]*)[Kk][Mm]': r'\1 km'   # Fix kilometers
        }
        
        for pattern, replacement in units.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _fix_form_text(self, text: str) -> str:
        """
        Fix common errors in form text
        
        Args:
            text: Form text
            
        Returns:
            Corrected text
        """
        # Fix common form field labels
        form_fields = {
            r'\b(?:F|f)irst\s*(?:N|n)ame\b': 'First Name',
            r'\b(?:L|l)ast\s*(?:N|n)ame\b': 'Last Name',
            r'\b(?:M|m)iddle\s*(?:N|n)ame\b': 'Middle Name',
            r'\b(?:F|f)ull\s*(?:N|n)ame\b': 'Full Name',
            r'\b(?:A|a)ddress\b': 'Address',
            r'\b(?:C|c)ity\b': 'City',
            r'\b(?:S|s)tate\b': 'State',
            r'\b(?:Z|z)ip\s*(?:C|c)ode\b': 'Zip Code',
            r'\b(?:P|p)ostal\s*(?:C|c)ode\b': 'Postal Code',
            r'\b(?:C|c)ountry\b': 'Country',
            r'\b(?:E|e)mail\b': 'Email',
            r'\b(?:P|p)hone\b': 'Phone',
            r'\b(?:M|m)obile\b': 'Mobile',
            r'\b(?:D|d)ate\s*(?:O|o)f\s*(?:B|b)irth\b': 'Date of Birth',
            r'\b(?:G|g)ender\b': 'Gender',
            r'\b(?:O|o)ccupation\b': 'Occupation',
            r'\b(?:C|c)ompany\b': 'Company',
            r'\b(?:D|d)epartment\b': 'Department',
            r'\b(?:S|s)ignature\b': 'Signature',
            r'\b(?:D|d)ate\b': 'Date'
        }
        
        for pattern, replacement in form_fields.items():
            text = re.sub(pattern, replacement, text)
        
        # Ensure form field labels are followed by colons
        for field in form_fields.values():
            # Add colon if field exists but doesn't have one
            pattern = f'({field})\\s+([^:\\n]+)'
            replacement = r'\1: \2'
            text = re.sub(pattern, replacement, text)
        
        # Fix check boxes
        text = re.sub(r'\[\s*[xX✓✔]\s*\]', '☑', text)  # Checked box
        text = re.sub(r'\[\s*\]', '☐', text)  # Empty box
        
        # Fix form structure - newlines after each field
        for field in form_fields.values():
            pattern = f'({field}:\\s+[^\\n]+)([^\\n])'
            replacement = r'\1\n\2'
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _fix_table_text(self, text: str) -> str:
        """
        Fix common errors in table text
        
        Args:
            text: Table text
            
        Returns:
            Corrected text
        """
        # Attempt to detect table structure
        lines = text.split('\n')
        
        # Check if this looks like a table with columns
        # Look for delimiter patterns
        if any('|' in line for line in lines) or any('\t' in line for line in lines):
            # Already has delimiters, normalize them
            formatted_lines = []
            
            for line in lines:
                # Replace tabs with pipe delimiters
                line = line.replace('\t', ' | ')
                
                # Normalize pipe delimiters (ensure spaces around them)
                line = re.sub(r'\s*\|\s*', ' | ', line)
                
                # Remove empty columns
                line = re.sub(r'\|\s+\|', '|', line)
                
                # Add to formatted lines
                formatted_lines.append(line)
            
            # Join lines
            table_text = '\n'.join(formatted_lines)
            
            # Try to detect header and add separator
            if len(formatted_lines) > 1:
                if '|' in formatted_lines[0] and '|' in formatted_lines[1]:
                    # Insert separator after header
                    header_parts = formatted_lines[0].split('|')
                    separator_parts = ['-' * len(part.strip()) for part in header_parts]
                    separator_line = '|'.join(separator_parts)
                    
                    formatted_lines.insert(1, separator_line)
                    table_text = '\n'.join(formatted_lines)
            
            return table_text
            
        else:
            # Try to detect columns based on space alignment
            # Check for consistent spacing that might indicate columns
            # Look for words with large gaps between them
            words_positions = []
            
            for line in lines:
                # Find all word positions in the line
                positions = []
                for match in re.finditer(r'\S+', line):
                    positions.append((match.start(), match.end()))
                words_positions.append(positions)
            
            # Need enough lines to detect a pattern
            if len(words_positions) > 2:
                # Try to find column boundaries
                col_starts = {}
                col_ends = {}
                
                for positions in words_positions:
                    for start, end in positions:
                        col_starts[start] = col_starts.get(start, 0) + 1
                        col_ends[end] = col_ends.get(end, 0) + 1
                
                # Find potential column boundaries (frequent positions)
                line_count = len(words_positions)
                threshold = line_count * 0.4  # At least 40% of lines should have a boundary here
                
                potential_cols = sorted([
                    pos for pos, count in col_starts.items() if count >= threshold
                ] + [
                    pos for pos, count in col_ends.items() if count >= threshold
                ])
                
                # Merge close positions
                col_boundaries = []
                curr_boundary = None
                
                for pos in potential_cols:
                    if curr_boundary is None:
                        curr_boundary = pos
                    elif pos - curr_boundary < 5:  # Merge if very close
                        curr_boundary = (curr_boundary + pos) // 2
                    else:
                        col_boundaries.append(curr_boundary)
                        curr_boundary = pos
                
                if curr_boundary is not None:
                    col_boundaries.append(curr_boundary)
                
                # If we have enough column boundaries, format as a table
                if len(col_boundaries) >= 2:
                    formatted_lines = []
                    
                    for line in lines:
                        if not line.strip():
                            formatted_lines.append("")
                            continue
                        
                        # Insert pipe delimiters at column boundaries
                        new_line = ""
                        last_pos = 0
                        
                        for boundary in col_boundaries:
                            if boundary > len(line):
                                continue
                            
                            new_line += line[last_pos:boundary] + " | "
                            last_pos = boundary
                        
                        if last_pos < len(line):
                            new_line += line[last_pos:]
                        
                        # Clean up multiple delimiters
                        new_line = re.sub(r'\|\s+\|', '|', new_line)
                        
                        formatted_lines.append(new_line)
                    
                    # Try to add a separator after header
                    if len(formatted_lines) > 1:
                        header_parts = formatted_lines[0].split('|')
                        separator_parts = ['-' * len(part.strip()) for part in header_parts]
                        separator_line = '|'.join(separator_parts)
                        
                        formatted_lines.insert(1, separator_line)
                    
                    return '\n'.join(formatted_lines)
        
        # If no table structure detected, return original text
        return text
    
    def _apply_general_text_corrections(self, text: str) -> str:
        """
        Apply enhanced general text corrections for OCR output
        
        Args:
            text: OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return ""
        
        # Remove invalid unicode characters
        text = ''.join(c for c in text if ord(c) < 65536)
        
        # Fix quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("''", '"').replace(",,", '"')
        text = text.replace("'", "'").replace("`", "'")
        
        # Fix bullet points and normalize them
        text = re.sub(r'[\*\+\-‣▪•●·](?:\s+|\n)', '• ', text)
        
        # Fix common OCR letter confusions
        text = re.sub(r'(?<=\d)l(?=\d)', '1', text)  # Digit + l + digit → 1
        text = re.sub(r'(?<=\d)I(?=\d)', '1', text)  # Digit + I + digit → 1
        text = re.sub(r'(?<=\d)O(?=\d)', '0', text)  # Digit + O + digit → 0
        text = re.sub(r'(?<=\d)S(?=\d)', '5', text)  # Digit + S + digit → 5
        text = re.sub(r'(?<=\d)Z(?=\d)', '2', text)  # Digit + Z + digit → 2
        text = re.sub(r'(?<=\d)B(?=\d)', '8', text)  # Digit + B + digit → 8
        
        # Fix space issues
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # No space between word and capital letter
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)  # No space between letter and digit
        text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)  # No space between digit and letter
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix spacing after punctuation
        text = re.sub(r'([.!?,:;])([A-Z0-9])', r'\1 \2', text)
        
        # Fix common merged words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Use context to fix common errors
        common_errors = {
            'tbe': 'the',
            'arid': 'and',
            'ofthe': 'of the',
            'forthe': 'for the',
            'tothe': 'to the',
            'inthe': 'in the',
            'fromthe': 'from the',
            'onthe': 'on the',
            'withthe': 'with the',
            'atthe': 'at the',
            'isthe': 'is the',
            'wasthe': 'was the',
            'asthe': 'as the',
            'bythe': 'by the',
            'thatthe': 'that the',
            'butthe': 'but the',
            'andthe': 'and the',
            'Tbis': 'This',
            'ca11': 'call',
            'cornpany': 'company',
            'frorn': 'from',
            'systern': 'system',
            'rnay': 'may',
            'Iine': 'line',
            'tirne': 'time',
            'Iist': 'list',
            'Iike': 'like',
            'sirnple': 'simple',
            'sarne': 'same',
            'frorntbe': 'from the'
        }
        
        for error, correction in common_errors.items():
            text = re.sub(r'\b' + error + r'\b', correction, text)
        
        # Fix newlines - remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphens at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Fix URLs and email addresses
        # Email pattern
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        for email in emails:
            # Common OCR errors in emails
            fixed_email = email.replace(' ', '').replace(',', '.').replace(';', '.')
            text = text.replace(email, fixed_email)
        
        # Fix URL pattern
        url_pattern = r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*\b'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            # Common OCR errors in URLs
            fixed_url = url.replace(' ', '').replace(',', '.').replace(';', '.')
            text = text.replace(url, fixed_url)
        
        return text
    
    def _enhance_text_organization(self, text: str, image_type: ImageType) -> str:
        """
        Enhance text organization based on image type and content
        
        Args:
            text: Text to organize
            image_type: Type of image
            
        Returns:
            Organized text
        """
        # Add appropriate structure based on image type
        if image_type == ImageType.DOCUMENT or image_type == ImageType.BOOK_PAGE:
            # Preserve paragraph structure
            text = self._organize_document_text(text)
        elif image_type == ImageType.FORM:
            # Organize form with proper field formatting
            text = self._organize_form_text(text)
        elif image_type == ImageType.RECEIPT:
            # Organize receipt with sections
            text = self._organize_receipt_text(text)
        elif image_type == ImageType.ID_CARD:
            # Organize ID card fields
            text = self._organize_id_card_text(text)
        elif image_type == ImageType.TABLE:
            # Organize table format
            text = self._organize_table_text(text)
        else:
            # Default organization
            text = self._default_text_organization(text)
        
        return text
    
    def _organize_document_text(self, text: str) -> str:
        """
        Organize document text with proper paragraph structure
        
        Args:
            text: Document text
            
        Returns:
            Organized text
        """
        # Split into lines while preserving paragraph structure
        lines = text.split('\n')
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            # Clean the line
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append("")  # Add empty line to preserve structure
            elif line.startswith('•') or line.startswith('-') or re.match(r'^\d+[\.\)]', line):
                # This is a list item - preserve as standalone line
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            elif re.match(r'^[A-Z][A-Z\s]+:?', line) or re.match(r'^[A-Z][A-Za-z\s]+:', line):
                # This is likely a heading or a form field label (like "NAME:")
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            elif len(line) < 40 and not line.endswith(('.', '?', '!')):
                # Short line that doesn't end with punctuation - might be a heading
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append(line)
            else:
                # Regular text line - add to current paragraph
                # But check if it should start a new paragraph based on content
                if (current_paragraph and 
                    (line[0].isupper() or re.match(r'^[0-9]', line)) and
                    current_paragraph[-1].endswith(('.', '!', '?'))):
                    # This line starts with a capital letter or number and previous line 
                    # ended with punctuation - likely a new paragraph
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
        
        # Join formatted lines, preserving structure
        return '\n'.join(formatted_lines)
    
    def _organize_form_text(self, text: str) -> str:
        """
        Organize form text with field labels and values clearly separated
        
        Args:
            text: Form text
            
        Returns:
            Organized text
        """
        # Split into lines
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Try to identify form field patterns: "Label: Value" or "Label Value"
            label_value_match = re.match(r'^([A-Za-z\s]+):\s*(.+)', line)
            if label_value_match:
                label = label_value_match.group(1).strip()
                value = label_value_match.group(2).strip()
                
                # Format as "Label: Value"
                formatted_lines.append(f"{label}: {value}")
            else:
                # Try to match label without colon
                label_value_match = re.match(r'^([A-Za-z\s]+)\s{2,}(.+)', line)
                if label_value_match:
                    label = label_value_match.group(1).strip()
                    value = label_value_match.group(2).strip()
                    
                    # Format as "Label: Value"
                    formatted_lines.append(f"{label}: {value}")
                else:
                    # No special formatting needed
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _organize_receipt_text(self, text: str) -> str:
        """
        Organize receipt text with clear sections
        
        Args:
            text: Receipt text
            
        Returns:
            Organized text
        """
        # Split into lines
        lines = text.split('\n')
        formatted_lines = []
        
        # Extract header (store name, address, date)
        header_lines = []
        item_lines = []
        total_lines = []
        footer_lines = []
        
        # Track current section
        section = "header"
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Check for transition to items section
            if section == "header" and (
                re.match(r'^ITEM|^DESCRIPTION|^QTY|PRICE|^-+', line, re.IGNORECASE) or
                re.match(r'^={5,}', line)
            ):
                section = "items"
                continue
            
            # Check for transition to totals section
            if section == "items" and (
                re.match(r'^SUBTOTAL|^TAX|^TOTAL|^={5,}', line, re.IGNORECASE)
            ):
                section = "totals"
            
            # Check for transition to footer
            if section == "totals" and (
                re.match(r'^THANK|^RETURN|^EXCHANGE|^POLICY|^RECEIPT', line, re.IGNORECASE)
            ):
                section = "footer"
            
            # Add line to appropriate section
            if section == "header":
                header_lines.append(line)
            elif section == "items":
                item_lines.append(line)
            elif section == "totals":
                total_lines.append(line)
            elif section == "footer":
                footer_lines.append(line)
        
        # Format header - usually store name, address, date
        if header_lines:
            formatted_lines.extend(header_lines)
            formatted_lines.append("")  # Add separator
        
        # Format items section
        if item_lines:
            formatted_lines.append("ITEMS:")
            formatted_lines.extend(["  " + line for line in item_lines])
            formatted_lines.append("")  # Add separator
        
        # Format totals section
        if total_lines:
            formatted_lines.append("TOTALS:")
            formatted_lines.extend(total_lines)
            formatted_lines.append("")  # Add separator
        
        # Format footer section
        if footer_lines:
            formatted_lines.extend(footer_lines)
        
        return '\n'.join(formatted_lines)
    
    def _organize_id_card_text(self, text: str) -> str:
        """
        Organize ID card text with fields clearly labeled
        
        Args:
            text: ID card text
            
        Returns:
            Organized text
        """
        # Split into lines
        lines = text.split('\n')
        formatted_lines = []
        
        # Common ID card fields
        id_fields = [
            'NAME', 'ADDRESS', 'DATE OF BIRTH', 'DOB', 'EXPIRATION DATE', 'SEX', 'GENDER',
            'HEIGHT', 'WEIGHT', 'EYES', 'HAIR', 'DRIVER\'S LICENSE', 'ISSUE DATE',
            'PLACE OF BIRTH', 'NATIONALITY', 'RELIGION', 'MARITAL STATUS', 'BLOOD TYPE',
            'OCCUPATION', 'ID NUMBER', 'SIGNATURE'
        ]
        
        # Extract field values
        field_values = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match field: value pattern
            field_match = None
            for field in id_fields:
                pattern = f'^{field}\\s*:?\\s*(.+)'
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    field_match = (field, match.group(1).strip())
                    break
            
            if field_match:
                field, value = field_match
                field_values[field.upper()] = value
            else:
                # Check if line contains a field
                for field in id_fields:
                    if field.upper() in line.upper():
                        # Extract the value (text after the field)
                        parts = re.split(field, line, flags=re.IGNORECASE)
                        if len(parts) > 1 and parts[1].strip():
                            field_values[field.upper()] = parts[1].strip()
                            break
        
        # Format the output in a standard way
        # First add the name if available
        if 'NAME' in field_values:
            formatted_lines.append(f"NAME: {field_values['NAME']}")
        
        # Add ID number if available
        for field in ['ID NUMBER', 'DRIVER\'S LICENSE']:
            if field in field_values:
                formatted_lines.append(f"{field}: {field_values[field]}")
                break
        
        # Add other fields in a logical order
        field_order = [
            'ADDRESS', 'DATE OF BIRTH', 'DOB', 'PLACE OF BIRTH', 'SEX', 'GENDER',
            'HEIGHT', 'WEIGHT', 'EYES', 'HAIR', 'BLOOD TYPE', 'NATIONALITY',
            'RELIGION', 'MARITAL STATUS', 'OCCUPATION', 'ISSUE DATE',
            'EXPIRATION DATE'
        ]
        
        for field in field_order:
            if field in field_values:
                formatted_lines.append(f"{field}: {field_values[field]}")
        
        # Add any remaining fields
        for field, value in field_values.items():
            if field not in ['NAME', 'ID NUMBER', 'DRIVER\'S LICENSE'] + field_order:
                formatted_lines.append(f"{field}: {value}")
        
        return '\n'.join(formatted_lines)
    
    def _organize_table_text(self, text: str) -> str:
        """
        Organize text from table structure
        
        Args:
            text: Table text
            
        Returns:
            Organized text in table format
        """
        lines = text.split('\n')
        formatted_lines = []
        
        # Handle pipe-delimited tables
        if any('|' in line for line in lines):
            # Already has pipe delimiters, clean up format
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    formatted_lines.append("")
                    continue
                
                # Normalize pipe delimiters (ensure spaces around them)
                line = re.sub(r'\s*\|\s*', ' | ', line.strip())
                
                # Add leading/trailing pipes if missing
                if not line.startswith('|'):
                    line = '| ' + line
                if not line.endswith('|'):
                    line = line + ' |'
                
                formatted_lines.append(line)
                
                # Add separator after header if not present
                if i == 0 and len(lines) > 1:
                    # Check if the next line isn't already a separator
                    if not lines[1].strip().startswith('--') and not lines[1].strip().startswith('=='):
                        # Create separator matching first row format
                        columns = line.count('|') - 1
                        separator = '|' + '|'.join([' --- ' for _ in range(columns)]) + '|'
                        formatted_lines.append(separator)
            
            return '\n'.join(formatted_lines)
        
        # For non-delimited tables, try to identify columns through spacing
        # Find consistent space patterns that might indicate columns
        if len(lines) > 2:
            # Detect potential column boundaries through whitespace patterns
            whitespace_cols = []
            for line in lines[:5]:  # Use first few lines for pattern detection
                if not line.strip():
                    continue
                
                # Find columns of whitespace
                prev_char = ''
                col_start = -1
                for i, char in enumerate(line):
                    if char.isspace() and prev_char not in string.whitespace:
                        col_start = i
                    elif not char.isspace() and prev_char in string.whitespace and col_start >= 0:
                        # End of whitespace column
                        if i - col_start >= 2:  # Minimum 2 spaces for column
                            whitespace_cols.append((col_start, i))
                        col_start = -1
                    prev_char = char
            
            # If we have consistent whitespace columns, convert to table format
            if whitespace_cols:
                # Group nearby column boundaries
                boundaries = []
                for start, end in sorted(whitespace_cols, key=lambda x: x[0]):
                    if not boundaries or start > boundaries[-1] + 3:
                        boundaries.append(start)
                
                # If we have boundaries, format as a table
                if len(boundaries) >= 1:
                    for line in lines:
                        if not line.strip():
                            formatted_lines.append("")
                            continue
                        
                        # Insert pipe delimiters at boundaries
                        new_line = "| "
                        last_pos = 0
                        
                        for boundary in boundaries:
                            if boundary < len(line):
                                new_line += line[last_pos:boundary].strip() + " | "
                                last_pos = boundary
                        
                        # Add remaining text
                        if last_pos < len(line):
                            new_line += line[last_pos:].strip() + " |"
                        
                        formatted_lines.append(new_line)
                    
                    # Add separator after header
                    if len(formatted_lines) > 0:
                        columns = formatted_lines[0].count('|') - 1
                        separator = '|' + '|'.join([' --- ' for _ in range(columns)]) + '|'
                        formatted_lines.insert(1, separator)
                    
                    return '\n'.join(formatted_lines)
        
        # If table structure not detected, just return cleaned text
        return '\n'.join([line.strip() for line in lines])
    
    def _default_text_organization(self, text: str) -> str:
        """
        Default text organization for general text
        
        Args:
            text: General text
            
        Returns:
            Organized text
        """
        # Split text into lines
        lines = text.split('\n')
        
        # Remove excess empty lines (more than 2 consecutive)
        formatted_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if not prev_empty:
                    formatted_lines.append("")
                    prev_empty = True
            else:
                formatted_lines.append(line)
                prev_empty = False
        
        # Join lines
        return '\n'.join(formatted_lines)
    
    def format_text(self, text: str, layout_info: Dict) -> str:
        """
        Format extracted text with enhanced layout information
        
        Args:
            text: Raw OCR text
            layout_info: Layout information
            
        Returns:
            Formatted text
        """
        if not text:
            return ""
        
        # Use layout information to format text appropriately
        document_structure = None
        if layout_info:
            # Determine document structure from layout info
            if layout_info.get("has_table", False):
                document_structure = DocumentStructure.TABLE
            elif layout_info.get("has_form", False):
                document_structure = DocumentStructure.FORM
            elif layout_info.get("columns", 1) > 1:
                document_structure = DocumentStructure.MULTI_COLUMN
            elif layout_info.get("document_type") == "scientific":
                document_structure = DocumentStructure.SCIENTIFIC
            elif "paragraphs" in layout_info and len(layout_info["paragraphs"]) > 1:
                document_structure = DocumentStructure.PARAGRAPHS
            else:
                # Determine structure from text content
                document_structure = self.detect_document_structure(text)
        else:
            # Determine structure from text content
            document_structure = self.detect_document_structure(text)
        
        # Format based on document structure
        if document_structure == DocumentStructure.PLAIN_TEXT:
            formatted_text = self._format_plain_text(text)
        elif document_structure == DocumentStructure.PARAGRAPHS:
            formatted_text = self._format_paragraphs(text, layout_info)
        elif document_structure == DocumentStructure.HEADERS_AND_CONTENT:
            formatted_text = self._format_headers_and_content(text, layout_info)
        elif document_structure == DocumentStructure.BULLET_POINTS:
            formatted_text = self._format_bullet_points(text)
        elif document_structure == DocumentStructure.TABLE:
            formatted_text = self._format_table(text, layout_info)
        elif document_structure == DocumentStructure.FORM:
            formatted_text = self._format_form(text, layout_info)
        elif document_structure == DocumentStructure.MULTI_COLUMN:
            formatted_text = self._format_multi_column(text, layout_info)
        elif document_structure == DocumentStructure.SCIENTIFIC:
            formatted_text = self._format_scientific(text)
        else:  # MIXED or other
            formatted_text = self._default_formatting(text)
        
        # Remove unwanted characters but preserve useful ones
        formatted_text = re.sub(r'[^\w\s.!?,;:()"\'•\-\n]', '', formatted_text)
        
        return formatted_text.strip()
    
    def detect_document_structure(self, text: str) -> DocumentStructure:
        """
        Detect the document structure based on text content
        
        Args:
            text: Text to analyze
            
        Returns:
            Document structure enum
        """
        # Count various features
        bullet_count = len(re.findall(r'(?:^|\n)[•\-*+]', text))
        numbered_list_count = len(re.findall(r'(?:^|\n)\d+[\.\)]', text))
        table_row_count = len(re.findall(r'(?:^|\n)[\w\s]+\|[\w\s]+\|', text))
        form_field_count = len(re.findall(r'(?:^|\n)[\w\s]+:', text))
        header_count = len(re.findall(r'(?:^|\n)[A-Z][A-Z\s]+(?:\n|$)', text))
        paragraph_count = len(re.findall(r'\n\s*\n', text))
        formula_count = len(re.findall(r'[=+\-*/^]|sqrt|sin|cos|tan|log', text))
        
        # Check for multi-column layout (hard to detect from text alone)
        lines = text.split('\n')
        if len(lines) > 10:
            # Check for lines that are unusually short and staggered
            short_line_count = 0
            for line in lines:
                if 5 < len(line.strip()) < 40:
                    short_line_count += 1
            
            if short_line_count > len(lines) * 0.6:
                return DocumentStructure.MULTI_COLUMN
        
        # Determine dominant structure
        if table_row_count > 5:
            return DocumentStructure.TABLE
        elif bullet_count + numbered_list_count > 5:
            return DocumentStructure.BULLET_POINTS
        elif form_field_count > 5:
            return DocumentStructure.FORM
        elif header_count > 2 and paragraph_count > 1:
            return DocumentStructure.HEADERS_AND_CONTENT
        elif paragraph_count > 1:
            return DocumentStructure.PARAGRAPHS
        elif formula_count > 3:
            return DocumentStructure.SCIENTIFIC
        elif len(text.strip()) < 100:
            return DocumentStructure.PLAIN_TEXT
        else:
            return DocumentStructure.MIXED
    
    def _format_plain_text(self, text: str) -> str:
        """
        Format plain text with minimal processing
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text
        """
        # Just clean up spaces and newlines
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(line for line in lines if line)
    
    def _format_paragraphs(self, text: str, layout_info: Dict) -> str:
        """
        Format text with paragraph structure
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted text with paragraphs
        """
        # Use paragraph information from layout if available
        if layout_info and "paragraphs" in layout_info:
            # Sort paragraphs by vertical position
            paragraphs = sorted(layout_info["paragraphs"], key=lambda p: p["bbox"][1])
            
            # Join paragraphs with proper spacing
            return "\n\n".join(p["text"] for p in paragraphs)
        
        # Otherwise, use text-based paragraph detection
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                # Add to current paragraph
                current_paragraph.append(line)
        
        # Add final paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join paragraphs with double newlines
        return '\n\n'.join(paragraphs)
    
    def _format_headers_and_content(self, text: str, layout_info: Dict) -> str:
        """
        Format text with headers and content
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted text with headers and content
        """
        lines = text.split('\n')
        formatted_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                formatted_lines.append("")
                i += 1
                continue
            
            # Check if this line might be a header
            is_header = False
            if re.match(r'^[A-Z][A-Z\s]+', line) or re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}', line):
                # All caps or Title Case with few words - likely a header
                is_header = True
            elif i < len(lines) - 1 and not lines[i+1].strip():
                # Followed by an empty line - could be a header
                is_header = True
            
            if is_header:
                # Add header with proper formatting
                formatted_lines.append("")  # Spacing before header
                formatted_lines.append(line)
                formatted_lines.append("")  # Spacing after header
                i += 1
                
                # Collect content under this header
                content_lines = []
                while i < len(lines) and (not lines[i].strip() or 
                                        not re.match(r'^[A-Z][A-Z\s]+', lines[i].strip())):
                    if lines[i].strip():
                        content_lines.append(lines[i].strip())
                    i += 1
                
                # Format the content as paragraphs
                if content_lines:
                    current_paragraph = []
                    for content_line in content_lines:
                        if not content_line:
                            if current_paragraph:
                                formatted_lines.append(' '.join(current_paragraph))
                                current_paragraph = []
                        else:
                            current_paragraph.append(content_line)
                    
                    if current_paragraph:
                        formatted_lines.append(' '.join(current_paragraph))
                
                # Don't increment i since the while loop already moved to the next header
            else:
                # Regular line
                formatted_lines.append(line)
                i += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_bullet_points(self, text: str) -> str:
        """
        Format text with bullet points
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text with bullet points
        """
        lines = text.split('\n')
        formatted_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                formatted_lines.append("")
                i += 1
                continue
            
            # Check for bullet point or numbered list
            bullet_match = re.match(r'^([•\-*+]|\d+[\.\)])(.+)', line)
            
            if bullet_match:
                # Get bullet/number and content
                bullet = bullet_match.group(1)
                content = bullet_match.group(2).strip()
                
                # Standardize bullets
                if bullet not in ['•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
                    bullet = '•'
                
                # Add formatted bullet point
                formatted_lines.append(f"{bullet} {content}")
                
                # Look for continuation lines (indented)
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r'^([•\-*+]|\d+[\.\)])', lines[i].strip()):
                    formatted_lines.append(f"  {lines[i].strip()}")
                    i += 1
            else:
                # Regular line
                formatted_lines.append(line)
                i += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_table(self, text: str, layout_info: Dict) -> str:
        """
        Format text as a table
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted table text
        """
        lines = text.split('\n')
        table_lines = []
        
        # Find table rows
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check if this might be a table row
            # Tables typically have consistent column alignment or delimiters
            
            # Check for explicit delimiters
            if '|' in line or '\t' in line:
                table_lines.append(line)
                i += 1
                continue
            
            # Check for space-aligned columns (multiple spaces between columns)
            if re.search(r'\S+\s{2,}\S+', line):
                table_lines.append(line)
                i += 1
                continue
            
            # Not a table row
            i += 1
        
        # If no table content found, return original text
        if not table_lines:
            return text
        
        # Format the table
        formatted_table = []
        delimiter = '|' if any('|' in line for line in table_lines) else None
        
        # Try to determine column boundaries for space-delimited tables
        column_boundaries = []
        if not delimiter:
            # Find potential column boundaries from space patterns
            for line in table_lines[:min(5, len(table_lines))]:
                positions = [m.start() for m in re.finditer(r'\s{2,}', line)]
                if positions:
                    column_boundaries.append(positions)
            
            # Find common boundaries
            if column_boundaries:
                # Group close positions
                all_positions = [pos for positions in column_boundaries for pos in positions]
                all_positions.sort()
                
                common_boundaries = []
                current_group = [all_positions[0]]
                
                for pos in all_positions[1:]:
                    if pos - current_group[-1] < 3:
                        current_group.append(pos)
                    else:
                        # Calculate average position for this group
                        common_boundaries.append(sum(current_group) // len(current_group))
                        current_group = [pos]
                
                if current_group:
                    common_boundaries.append(sum(current_group) // len(current_group))
                
                # Use common boundaries to format table
                for line in table_lines:
                    formatted_line = line
                    for boundary in reversed(common_boundaries):
                        if boundary < len(line):
                            formatted_line = formatted_line[:boundary] + ' | ' + formatted_line[boundary:].lstrip()
                    formatted_table.append(formatted_line)
            else:
                # Just use the original table lines
                formatted_table = table_lines
        else:
            # For explicitly delimited tables, just clean up spacing
            for line in table_lines:
                parts = [part.strip() for part in line.split('|')]
                formatted_table.append(' | '.join(parts))
        
        # Add a separator after the header row
        if len(formatted_table) > 1:
            header = formatted_table[0]
            separator = ''
            
            # Create a separator matching the structure of the header
            if '|' in header:
                parts = header.split('|')
                separator = '|'.join('-' * len(part.strip()) for part in parts)
            else:
                separator = '-' * len(header)
            
            formatted_table.insert(1, separator)
        
        # Join the formatted table
        return '\n'.join(formatted_table)
    
    def _format_form(self, text: str, layout_info: Dict) -> str:
        """
        Format text as a form
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted form text
        """
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Check for form field pattern: Label: Value
            field_match = re.match(r'^([A-Za-z\s]+):\s*(.+)', line)
            
            if field_match:
                # Already in the right format
                formatted_lines.append(line)
            else:
                # Check for field pattern without colon
                field_match = re.match(r'^([A-Za-z\s]+)\s{2,}(.+)', line)
                
                if field_match:
                    label = field_match.group(1).strip()
                    value = field_match.group(2).strip()
                    formatted_lines.append(f"{label}: {value}")
                else:
                    # Regular line
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_multi_column(self, text: str, layout_info: Dict) -> str:
        """
        Format text from multi-column layout
        
        Args:
            text: Text to format
            layout_info: Layout information
            
        Returns:
            Formatted multi-column text
        """
        # This is challenging from just OCR text without positional information
        # Try to detect column breaks from text patterns
        
        lines = text.split('\n')
        
        # Check if we have regions from layout info
        if layout_info and "regions" in layout_info:
            # Sort regions by column then by vertical position
            columns = {}
            
            for region in layout_info["regions"]:
                # Get region center x-coordinate
                if isinstance(region, dict) and "bbox" in region:
                    bbox = region["bbox"]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        center_x = bbox[0] + bbox[2] // 2
                    else:
                        continue
                    
                    # Determine which column this belongs to
                    column_idx = center_x // 200  # Simple column bucketing
                    
                    if column_idx not in columns:
                        columns[column_idx] = []
                    
                    columns[column_idx].append(region)
            
            # Sort regions in each column by y-coordinate
            for column_idx in columns:
                columns[column_idx].sort(key=lambda r: r["bbox"][1])
            
            # Reconstruct text by column
            formatted_text = ""
            
            # Process columns from left to right
            for column_idx in sorted(columns.keys()):
                column_text = "\n\n".join([r.get("text", "") for r in columns[column_idx]])
                
                if formatted_text:
                    formatted_text += "\n\n--- Next Column ---\n\n"
                
                formatted_text += column_text
            
            return formatted_text
        
        # Without layout info, try to detect column breaks heuristically
        # This is challenging and error-prone without positional info
        
        # Look for very short lines that may indicate column wrapping
        short_line_threshold = 30
        short_lines = [i for i, line in enumerate(lines) if 0 < len(line.strip()) < short_line_threshold]
        
        # If more than half the lines are short, likely multi-column
        if len(short_lines) > len(lines) * 0.5:
            # Try to recognize column boundaries by line patterns
            # This is a simplistic approach and may not work well
            formatted_lines = []
            formatted_lines.append("NOTE: This text appears to be in multiple columns. " +
                                 "The content below has been reformatted as a single column.")
            formatted_lines.append("")
            
            current_paragraph = []
            for line in lines:
                line = line.strip()
                
                if not line:
                    if current_paragraph:
                        formatted_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                    formatted_lines.append("")
                else:
                    # If line starts with capital letter and previous line was short,
                    # it might be a new paragraph or column break
                    if (current_paragraph and 
                        line[0].isupper() and 
                        len(current_paragraph[-1]) < short_line_threshold):
                        
                        # Check if this might be a continuation by checking for sentence ending
                        if current_paragraph[-1].endswith(('.', '!', '?', ':', ';')):
                            # Likely a new paragraph
                            formatted_lines.append(' '.join(current_paragraph))
                            current_paragraph = [line]
                        else:
                            # Might be continuation or column break
                            # Try to detect by checking for semantic connection
                            if len(current_paragraph[-1].split()) < 4:
                                # Very short line, likely a column break
                                formatted_lines.append(' '.join(current_paragraph))
                                current_paragraph = [line]
                            else:
                                # Probably continuation
                                current_paragraph.append(line)
                    else:
                        current_paragraph.append(line)
            
            # Add final paragraph
            if current_paragraph:
                formatted_lines.append(' '.join(current_paragraph))
            
            return '\n'.join(formatted_lines)
        
        # If not detected as multi-column, format as paragraphs
        return self._format_paragraphs(text, layout_info)
    
    def _format_scientific(self, text: str) -> str:
        """
        Format scientific text with formulas
        
        Args:
            text: Scientific text
            
        Returns:
            Formatted scientific text
        """
        lines = text.split('\n')
        formatted_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                formatted_lines.append("")
                i += 1
                continue
            
            # Check if this line might contain a formula
            formula_indicators = ['=', '+', '-', '*', '/', '^', 'sqrt', 'sin', 'cos', 'tan', 'log']
            is_formula = any(indicator in line for indicator in formula_indicators)
            
            if is_formula:
                # Add formula with proper spacing
                formatted_lines.append("")  # Space before formula
                formatted_lines.append(line)
                formatted_lines.append("")  # Space after formula
            else:
                # Regular text - format as paragraphs
                if i > 0 and formatted_lines and formatted_lines[-1] and not line.startswith(' '):
                    # Continue paragraph
                    formatted_lines[-1] += ' ' + line
                else:
                    # New paragraph or indented line
                    formatted_lines.append(line)
            
            i += 1
        
        return '\n'.join(formatted_lines)
    
    def _default_formatting(self, text: str) -> str:
        """
        Default text formatting for mixed content
        
        Args:
            text: Text to format
            
        Returns:
            Formatted text
        """
        # Split into lines while preserving paragraph structure
        lines = text.split('\n')
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            # Clean the line
            line = line.strip()
            
            if not line:
                # Empty line indicates paragraph break
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []
                formatted_lines.append("")  # Add empty line to preserve structure
            else:
                # Add to current paragraph
                current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
        
        # Join formatted lines
        return '\n'.join(formatted_lines)
    
    def detect_language(self, text: str) -> str:
        """
        Enhanced language detection with better accuracy using rules
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        if not text or len(text) < 20:
            return "unknown"
        
        # Enhanced keywords for language detection
        # Add more common words for better accuracy
        id_keywords = [
            'yang', 'dengan', 'dan', 'untuk', 'dari', 'pada', 'adalah', 'ini', 'itu',
            'dalam', 'tidak', 'akan', 'saya', 'kamu', 'kami', 'mereka', 'bisa', 'oleh',
            'jika', 'telah', 'sudah', 'harus', 'dapat', 'karena', 'kepada', 'maka',
            'tentang', 'setiap', 'seperti', 'juga', 'ada', 'sebuah', 'tersebut',
            'anda', 'sangat', 'kemudian', 'saat', 'selama', 'masih', 'lebih',
            'belum', 'ketika', 'kita', 'baru', 'perlu'
        ]
        
        en_keywords = [
            'the', 'is', 'are', 'and', 'for', 'that', 'have', 'with', 'this', 'from',
            'they', 'will', 'would', 'there', 'their', 'what', 'about', 'which',
            'when', 'one', 'all', 'been', 'but', 'not', 'you', 'your', 'who',
            'more', 'has', 'was', 'were', 'can', 'said', 'out', 'use', 'into',
            'some', 'than', 'other', 'time', 'now', 'only', 'like', 'just'
        ]
        
        # Count keyword occurrences
        text_lower = ' ' + text.lower() + ' '
        id_count = sum(1 for word in id_keywords if f' {word} ' in text_lower)
        en_count = sum(1 for word in en_keywords if f' {word} ' in text_lower)
        
        # Calculate weighted scores
        id_score = id_count / len(id_keywords)
        en_score = en_count / len(en_keywords)
        
        # Add text pattern analysis for better detection
        # Indonesian patterns
        id_patterns = [r'\bakan\s+\w+\b', r'\bsedang\s+\w+\b', r'\btelah\s+\w+\b']
        id_pattern_count = sum(1 for pattern in id_patterns if re.search(pattern, text_lower))
        id_score += id_pattern_count * 0.1
        
        # English patterns
        en_patterns = [r'\bwill\s+\w+\b', r'\bhave\s+\w+\b', r'\bhas\s+\w+\b']
        en_pattern_count = sum(1 for pattern in en_patterns if re.search(pattern, text_lower))
        en_score += en_pattern_count * 0.1
        
        # Make decision based on threshold
        if id_score > 0.15 and id_score > en_score:
            return 'id'
        elif en_score > 0.15:
            return 'en'
        else:
            # Check for other language indicators
            # These are simplified character set checks
            
            # Check for Latin script dominance
            latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            total_chars = sum(1 for c in text if c.isalpha())
            
            if total_chars > 0:
                latin_ratio = latin_chars / total_chars
                
                if latin_ratio > 0.9:
                    # Probably a Latin script language other than English or Indonesian
                    return 'latin'
                elif latin_ratio < 0.3:
                    # Probably a non-Latin script language
                    return 'non-latin'
            
            return 'unknown'
    
    def generate_summary(self, text: str, max_length: int = 200, style: str = "concise") -> str:
        """
        Generate an enhanced extractive summary without relying on AI models
        
        Args:
            text: Input text
            max_length: Maximum summary length
            style: Summary style (concise, detailed, bullets, structured)
            
        Returns:
            Extractive summary
        """
        if not text or len(text) < 100:
            return text[:max_length] if text else ""
        
        # Fall back to enhanced extractive summarization
        try:
            # Approach depends on whether NLTK is available
            if NLTK_AVAILABLE:
                # Use NLTK for better text segmentation and frequency analysis
                return self._generate_extractive_summary_nltk(text, max_length, style)
            else:
                # Use regex-based summarization without NLTK
                return self._generate_extractive_summary_regex(text, max_length, style)
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            
            # Very simple fallback - first few sentences or characters
            sentences = re.split(r'(?<=[.!?])\s+', text)
            summary = sentences[0]
            
            i = 1
            while i < len(sentences) and len(summary) + len(sentences[i]) + 1 <= max_length:
                summary += " " + sentences[i]
                i += 1
            
            return summary
    
    def _generate_extractive_summary_nltk(self, text: str, max_length: int = 200, style: str = "concise") -> str:
        """
        Generate extractive summary using NLTK
        
        Args:
            text: Input text
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            Extractive summary
        """
        # Extract title if available
        title = ""
        first_line = text.split('\n', 1)[0].strip()
        if len(first_line) < 100 and not first_line.endswith(('.', '!', '?')):
            title = first_line
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Use improved TextRank-inspired algorithm for important sentences
        sentence_scores = {}
        
        # Tokenize text into words
        tokens = word_tokenize(text.lower())
        
        # Use POS tagging if available to filter non-important words
        filtered_tokens = []
        try:
            pos_tags = nltk.pos_tag(tokens)
            for word, tag in pos_tags:
                if word not in STOPWORDS and word not in string.punctuation:
                    filtered_tokens.append(word)
        except:
            # Basic filtering if POS tagging fails
            filtered_tokens = [word for word in tokens if word not in STOPWORDS and word not in string.punctuation]
        
        # Calculate word frequencies
        word_freq = FreqDist(filtered_tokens)
        
        # Calculate sentence scores based on word frequencies
        for i, sentence in enumerate(sentences):
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Tokenize sentence
            sent_words = word_tokenize(sentence.lower())
            
            # Filter stopwords and punctuation
            sent_words = [word for word in sent_words if word not in STOPWORDS and word not in string.punctuation]
            
            if not sent_words:
                continue
            
            # Calculate score based on word frequency of important words
            score = sum(word_freq[word] for word in sent_words) / len(sent_words)
            
            # Position bias based on style
            if style == "concise":
                # Strong bias for first and last sentences
                if i < len(sentences) * 0.1:  # First 10%
                    score *= 1.5
                elif i > len(sentences) * 0.9:  # Last 10%
                    score *= 1.2
            else:  # "detailed" or other styles
                # Milder position bias
                if i < len(sentences) * 0.2:  # First 20%
                    score *= 1.25
                elif i > len(sentences) * 0.8:  # Last 20%
                    score *= 1.1
            
            # Extra boost for sentences with key terms
            # Customize key terms based on likely document type
            key_terms = ["summary", "conclusion", "result", "important", "significant", 
                        "key", "main", "primary", "critical", "essential", "crucial"]
            
            # Check if any words match key terms
            if any(word in key_terms for word in sent_words):
                score *= 1.2
            
            sentence_scores[i] = score
        
        # Determine how many sentences to include based on style and max_length
        avg_sent_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 20
        target_sentences = max(1, int(max_length / avg_sent_length))
        
        if style == "detailed":
            target_sentences = min(int(target_sentences * 1.5), len(sentences))
        elif style == "concise":
            target_sentences = max(1, int(target_sentences * 0.7))
        
        # Get top scoring sentences
        top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in top_indices[:target_sentences]]
        
        # Sort indices to preserve original order
        top_indices.sort()
        
        # Extract key sentences
        key_sentences = [sentences[i] for i in top_indices if i < len(sentences)]
        
        # Build summary
        if title:
            # If we have a title, include it
            summary_parts = [title]
            remaining_length = max_length - len(title)
            
            # Add sentences up to remaining length
            for sentence in key_sentences:
                if len(sentence) + 2 <= remaining_length:
                    summary_parts.append(sentence)
                    remaining_length -= len(sentence) + 2
                else:
                    break
            
            summary = title
            if len(summary_parts) > 1:
                summary += ". " + " ".join(summary_parts[1:])
        else:
            # Without title, just join key sentences
            summary = " ".join(key_sentences)
        
        # Format summary based on style
        if style == "bullets":
            return self._format_as_bullet_points(summary)
        elif style == "structured":
            return self._format_as_structured_summary(summary)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _generate_extractive_summary_regex(self, text: str, max_length: int = 200, style: str = "concise") -> str:
        """
        Generate extractive summary using regex (when NLTK is not available)
        
        Args:
            text: Input text
            max_length: Maximum summary length
            style: Summary style
            
        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences based on position and keywords
        sentence_scores = {}
        
        # Simplified stopwords if NLTK is not available
        simple_stopwords = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                          "when", "where", "how", "which", "who", "whom", "this", "that", "these",
                          "those", "then", "just", "so", "than", "such", "both", "through", "about",
                          "for", "is", "of", "while", "during", "to", "from"}
        
        # Build a simple word frequency counter
        word_counts = {}
        for sentence in sentences:
            # Split into words and count frequencies
            words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
            for word in words:
                if word not in simple_stopwords:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Score sentences
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.strip()) < 10:
                continue
            
            # Base score
            score = 0
            
            # Position score - first and last sentences are important
            if i == 0:
                score += 5  # First sentence
            elif i == len(sentences) - 1:
                score += 3  # Last sentence
            elif i < len(sentences) * 0.1:
                score += 2  # Early sentences
            
            # Word importance score
            words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
            if words:
                # Average frequency of non-stopwords
                word_score = sum(word_counts.get(word, 0) for word in words 
                               if word not in simple_stopwords) / len(words)
                score += word_score
            
            # Keyword score
            key_terms = ["summary", "conclusion", "result", "important", "significant", 
                        "key", "main", "primary", "critical", "essential", "crucial"]
            for term in key_terms:
                if term in sentence.lower():
                    score += 3
                    break
            
            sentence_scores[i] = score
        
        # Determine how many sentences to include
        avg_sent_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 20
        target_sentences = max(1, int(max_length / avg_sent_length))
        
        if style == "detailed":
            target_sentences = min(int(target_sentences * 1.5), len(sentences))
        elif style == "concise":
            target_sentences = max(1, int(target_sentences * 0.7))
        
        # Get top scoring sentences
        top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in top_indices[:target_sentences]]
        
        # Sort indices to preserve original order
        top_indices.sort()
        
        # Extract key sentences
        summary = " ".join(sentences[i] for i in top_indices if i < len(sentences))
        
        # Format summary based on style
        if style == "bullets":
            return self._format_as_bullet_points(summary)
        elif style == "structured":
            return self._format_as_structured_summary(summary)
        
        # Truncate if necessary
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _format_as_bullet_points(self, summary: str) -> str:
        """
        Format summary as bullet points
        
        Args:
            summary: Text summary
            
        Returns:
            Bullet point formatted summary
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        # Format as bullet points
        bullet_points = []
        
        for sentence in sentences:
            if sentence.strip():
                # Clean the sentence
                clean_sentence = sentence.strip()
                
                # Ensure it ends with proper punctuation
                if not clean_sentence[-1] in ['.', '!', '?']:
                    clean_sentence += '.'
                
                # Add bullet point
                bullet_points.append(f"• {clean_sentence}")
        
        return '\n'.join(bullet_points)
    
    def _format_as_structured_summary(self, summary: str) -> str:
        """
        Format as structured summary with sections
        
        Args:
            summary: Text summary
            
        Returns:
            Structured summary
        """
        # Extract potential entities for structured summary
        sections = {
            "SUMMARY": summary
        }
        
        # Try to extract key entities
        people = []
        organizations = []
        locations = []
        dates = []
        
        # Look for potential names (capitalized phrases)
        name_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', summary)
        if name_matches:
            people = list(set(name_matches))[:3]  # Top 3 unique names
        
        # Look for potential organizations
        org_patterns = [
            r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)+\s+(?:Inc|Corp|Co|Ltd|LLC|Company|Association|Organization)\b',
            r'\b[A-Z][A-Z]+\b'  # Acronyms
        ]
        org_matches = []
        for pattern in org_patterns:
            org_matches.extend(re.findall(pattern, summary))
        
        if org_matches:
            organizations = list(set(org_matches))[:3]  # Top 3 unique
        
        # Look for potential locations
        loc_patterns = [
            r'\b[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)?\b'  # City, Country format
        ]
        loc_matches = []
        for pattern in loc_patterns:
            loc_matches.extend(re.findall(pattern, summary))
        
        if loc_matches:
            locations = list(set(loc_matches))[:3]  # Top 3 unique
        
        # Look for dates
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',                   # MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b',  # Month Day, Year
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{2,4}\b'  # Day Month Year
        ]
        date_matches = []
        for pattern in date_patterns:
            date_matches.extend(re.findall(pattern, summary))
        
        if date_matches:
            dates = list(set(date_matches))[:2]  # Top 2 unique
        
        # Add sections if not empty
        if people:
            sections["PEOPLE"] = ", ".join(people)
        if organizations:
            sections["ORGANIZATIONS"] = ", ".join(organizations)
        if locations:
            sections["LOCATIONS"] = ", ".join(locations)
        if dates:
            sections["DATES"] = ", ".join(dates)
        
        # Format structured summary
        formatted_summary = []
        
        for section, content in sections.items():
            formatted_summary.append(f"{section}:")
            formatted_summary.append(content)
            formatted_summary.append("")  # Empty line between sections
        
        return "\n".join(formatted_summary).strip()
    
    def extract_key_insights(self, text: str) -> List[str]:
        """
        Extract key insights using rule-based methods
        
        Args:
            text: Input text
            
        Returns:
            List of key insights
        """
        insights = []
        
        # Extract key facts using simple heuristics
        if NLTK_AVAILABLE:
            try:
                # Tokenize sentences
                sentences = sent_tokenize(text)
                
                # Look for sentences with insight markers
                insight_markers = [
                    r'\b(?:key|main|important|significant|critical)\s+(?:point|fact|finding|result|conclusion)\b',
                    r'\b(?:in\s+summary|to\s+summarize|in\s+conclusion|concluding|therefore)\b',
                    r'\b(?:must|should|need\s+to|have\s+to)\b',
                    r'\b(?:increased|decreased|improved|reduced|enhanced|caused)\b'
                ]
                
                fact_sentences = []
                for sentence in sentences:
                    if any(re.search(marker, sentence, re.IGNORECASE) for marker in insight_markers):
                        fact_sentences.append(sentence)
                
                # Take top 3 fact sentences
                for sentence in fact_sentences[:3]:
                    insights.append(sentence)
                
                # If no marker-based insights, use TextRank to find key sentences
                if not fact_sentences and len(sentences) > 3:
                    # Simplified TextRank implementation
                    sentence_scores = {}
                    
                    # Tokenize text into words
                    words = [w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in STOPWORDS]
                    word_freq = FreqDist(words)
                    
                    for i, sentence in enumerate(sentences):
                        if len(sentence) < 15:  # Skip very short sentences
                            continue
                        
                        # Calculate score based on word frequencies
                        words_in_sentence = [w.lower() for w in word_tokenize(sentence) if w.isalnum()]
                        score = sum(word_freq[w] for w in words_in_sentence) / max(1, len(words_in_sentence))
                        
                        # Boost score for sentences at beginning or end
                        if i < len(sentences) * 0.2:  # First 20%
                            score *= 1.25
                        elif i > len(sentences) * 0.8:  # Last 20%
                            score *= 1.1
                        
                        sentence_scores[i] = score
                    
                    # Get top scoring sentences
                    top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
                    top_indices = [idx for idx, _ in top_indices[:2]]  # Take top 2
                    
                    # Add key sentences as insights
                    for idx in sorted(top_indices):
                        if idx < len(sentences):
                            insights.append(sentences[idx])
            
            except Exception as e:
                logger.warning(f"Fact extraction failed: {e}")
        else:
            # Simple regex-based fact extraction without NLTK
            # Split into sentences using regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Define insight marker patterns
            insight_patterns = [
                r'(?:key|main|important|significant|critical).{0,20}(?:point|fact|finding|conclusion)',
                r'(?:in\s+summary|to\s+summarize|in\s+conclusion|concluding|therefore)',
                r'(?:must|should|need to|have to)',
                r'increase|decrease|improve|reduce|enhance|cause'
            ]
            
            # Look for sentences with insight markers
            for sentence in sentences:
                for pattern in insight_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        insights.append(sentence)
                        break
                        
                # Limit to 5 insights
                if len(insights) >= 5:
                    break
            
            # If not enough insight sentences, add first and last sentences
            if len(insights) < 2 and len(sentences) > 2:
                if sentences[0] not in insights:
                    insights.append(sentences[0])  # First sentence
                if sentences[-1] not in insights and sentences[-1] != sentences[0]:
                    insights.append(sentences[-1])  # Last sentence
        
        # Limit to 5 insights
        return insights[:5]