#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information extraction functionality for SmartGlassOCR
Includes structured information extraction from various document types
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

from .model import ImageType

logger = logging.getLogger("SmartGlass-InfoExtraction")

class InformationExtractor:
    """Extract structured information from OCR text"""
    
    def __init__(self, config=None):
        """
        Initialize the information extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def extract_structured_info(self, text: str, image_type: ImageType = None) -> Optional[Dict[str, Any]]:
        """
        Extract structured information using rule-based methods
        
        Args:
            text: Input text
            image_type: Optional image type for specialized extraction
            
        Returns:
            Dictionary of extracted fields or None
        """
        if not text:
            return None
        
        # Choose the appropriate extraction strategy based on image type
        if image_type == ImageType.ID_CARD:
            return self.extract_id_card_info(text)
        elif image_type == ImageType.RECEIPT:
            return self.extract_receipt_info(text)
        elif image_type == ImageType.FORM:
            return self.extract_form_info(text)
        elif image_type == ImageType.TABLE:
            return self.extract_table_info(text)
        
        # Default to generic form extraction
        return self.extract_generic_info(text)
    
    def extract_id_card_info(self, text: str) -> Dict[str, str]:
        """
        Extract information from ID card text
        
        Args:
            text: ID card text
            
        Returns:
            Dictionary of extracted fields
        """
        # Common ID card fields with regex patterns
        field_patterns = {
            'name': r'(?:name|nama)[\s:]+([^\n]+)',
            'date_of_birth': r'(?:date of birth|birth date|birthdate|dob|tanggal lahir)[\s:]+([^\n]+)',
            'gender': r'(?:gender|sex|jenis kelamin)[\s:]+([^\n]+)',
            'address': r'(?:address|alamat)[\s:]+([^\n]+)',
            'id_number': r'(?:id|no|number|nomor)[\s:]+([A-Z0-9\-\s]+)',
            'expiration_date': r'(?:expiration|expiry|exp|berlaku sampai)[\s:]+([^\n]+)',
            'issue_date': r'(?:issue|issued|date of issue|tanggal dikeluarkan)[\s:]+([^\n]+)',
            'nationality': r'(?:nationality|negara|warga negara|citizenship)[\s:]+([^\n]+)',
            'place_of_birth': r'(?:place of birth|birthplace|tempat lahir)[\s:]+([^\n]+)',
            'blood_type': r'(?:blood|blood type|golongan darah)[\s:]+([^\n]+)',
            'marital_status': r'(?:marital status|status perkawinan)[\s:]+([^\n]+)',
            'occupation': r'(?:occupation|job|pekerjaan)[\s:]+([^\n]+)',
            'religion': r'(?:religion|agama)[\s:]+([^\n]+)'
        }
        
        # Extract fields from text
        extracted_info = {}
        text_lower = text.lower()
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Only add if not empty
                if value:
                    extracted_info[field] = value
        
        # Also try a different approach - look for field labels followed by value
        field_labels = {
            'name': ['name', 'nama'],
            'date_of_birth': ['date of birth', 'birth date', 'birthdate', 'dob', 'tanggal lahir'],
            'gender': ['gender', 'sex', 'jenis kelamin'],
            'address': ['address', 'alamat'],
            'id_number': ['id', 'no', 'number', 'nomor', 'nomor kartu'],
            'expiration_date': ['expiration', 'expiry', 'exp', 'berlaku sampai'],
            'issue_date': ['issue', 'issued', 'date of issue', 'tanggal dikeluarkan'],
            'nationality': ['nationality', 'negara', 'warga negara', 'citizenship'],
            'place_of_birth': ['place of birth', 'birthplace', 'tempat lahir'],
            'blood_type': ['blood', 'blood type', 'golongan darah'],
            'marital_status': ['marital status', 'status perkawinan'],
            'occupation': ['occupation', 'job', 'pekerjaan'],
            'religion': ['religion', 'agama']
        }
        
        for field, labels in field_labels.items():
            if field in extracted_info:
                continue  # Already extracted
                
            for label in labels:
                # Try label followed by colon
                pattern = f"\\b{re.escape(label)}\\s*:\\s*([^\\n]+)"
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        extracted_info[field] = value
                        break
                        
                # Try label at the beginning of a line
                pattern = f"^\\s*{re.escape(label)}\\s+([^\\n]+)"
                match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        extracted_info[field] = value
                        break
        
        return extracted_info
    
    def extract_receipt_info(self, text: str) -> Dict[str, Any]:
        """
        Extract information from receipt text
        
        Args:
            text: Receipt text
            
        Returns:
            Dictionary of extracted fields
        """
        receipt_info = {
            'items': []
        }
        
        # Extract merchant/store name - usually found at the top
        lines = text.split('\n')
        if lines and lines[0].strip():
            receipt_info['merchant'] = lines[0].strip()
        
        # Extract date
        date_match = re.search(r'(?:date|tanggal)[\s:]+([0-9/\-\.]+)', text.lower())
        if date_match:
            receipt_info['date'] = date_match.group(1).strip()
        else:
            # Try simple date pattern
            date_pattern = r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b'
            date_match = re.search(date_pattern, text)
            if date_match:
                receipt_info['date'] = date_match.group(1)
        
        # Extract time
        time_match = re.search(r'(?:time|waktu)[\s:]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)', text.lower())
        if time_match:
            receipt_info['time'] = time_match.group(1).strip()
        else:
            # Try simple time pattern
            time_pattern = r'\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b'
            time_match = re.search(time_pattern, text)
            if time_match:
                receipt_info['time'] = time_match.group(1)
        
        # Extract subtotal
        subtotal_match = re.search(r'(?:subtotal|sub[\s-]?total)[\s:]+\$?([0-9\.,]+)', text.lower())
        if subtotal_match:
            receipt_info['subtotal'] = subtotal_match.group(1).strip()
        
        # Extract tax
        tax_match = re.search(r'(?:tax|vat|pajak)[\s:]+\$?([0-9\.,]+)', text.lower())
        if tax_match:
            receipt_info['tax'] = tax_match.group(1).strip()
        
        # Extract total
        total_match = re.search(r'(?:total|amount|jumlah)[\s:]+\$?([0-9\.,]+)', text.lower())
        if total_match:
            receipt_info['total'] = total_match.group(1).strip()
        
        # Extract payment method
        payment_methods = ['cash', 'card', 'credit', 'debit', 'visa', 'mastercard', 'amex', 'american express',
                        'discover', 'tunai', 'kartu', 'kredit', 'debit']
        for method in payment_methods:
            if method.lower() in text.lower():
                receipt_info['payment_method'] = method
                break
        
        # Extract items - this is more complex
        # Look for the items section
        items_section = None
        in_items = False
        items_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check for the start of the items section
            if re.match(r'^(?:items?|description|qty|quantity|item price)', line, re.IGNORECASE):
                in_items = True
                continue
            
            # Check for the end of the items section
            if in_items and re.match(r'^(?:subtotal|total|tax|amount)', line, re.IGNORECASE):
                in_items = False
                continue
            
            # Collect item lines
            if in_items and line:
                items_lines.append(line)
        
        # Parse item lines
        for line in items_lines:
            # Try to match item patterns:
            # 1. ItemName Quantity Price
            # 2. ItemName Price
            # 3. Quantity x ItemName Price
            
            # Pattern 1: ItemName Quantity Price
            match = re.match(r'(.+?)\s+(\d+)\s+\$?([0-9\.,]+)', line)
            if match:
                item_name = match.group(1).strip()
                quantity = match.group(2)
                price = match.group(3)
                receipt_info['items'].append({
                    'name': item_name,
                    'quantity': quantity,
                    'price': price
                })
                continue
            
            # Pattern 2: ItemName Price
            match = re.match(r'(.+?)\s+\$?([0-9\.,]+)', line)
            if match:
                item_name = match.group(1).strip()
                price = match.group(2)
                receipt_info['items'].append({
                    'name': item_name,
                    'quantity': '1',
                    'price': price
                })
                continue
            
            # Pattern 3: Quantity x ItemName Price
            match = re.match(r'(\d+)(?:\s*[xX]\s*)(.+?)\s+\$?([0-9\.,]+)', line)
            if match:
                quantity = match.group(1)
                item_name = match.group(2).strip()
                price = match.group(3)
                receipt_info['items'].append({
                    'name': item_name,
                    'quantity': quantity,
                    'price': price
                })
                continue
            
            # If no pattern matches, just add as item name
            if line:
                receipt_info['items'].append({
                    'name': line,
                    'quantity': '1',
                    'price': '0.00'
                })
        
        return receipt_info
    
    def extract_form_info(self, text: str) -> Dict[str, str]:
        """
        Extract information from form text
        
        Args:
            text: Form text
            
        Returns:
            Dictionary of extracted fields
        """
        # Look for field-value pairs using regex
        # Common pattern: Field: Value or Field - Value
        field_value_pattern = r'([A-Za-z\s]+[A-Za-z])[\s:]+(.+)'
        
        # Extract all field-value pairs
        form_info = {}
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(field_value_pattern, line)
            if match:
                field = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                
                # Only add if value is meaningful
                if value and not re.match(r'^[:\-,.;]*', value):
                    form_info[field] = value
        
        # Try special patterns for common form fields if they weren't found
        field_patterns = {
            'name': r'(?:name|nama)[\s:]+([^\n]+)',
            'email': r'(?:email|e-mail)[\s:]+([^\n]+)',
            'phone': r'(?:phone|telephone|tel|hp|handphone)[\s:]+([^\n]+)',
            'address': r'(?:address|alamat)[\s:]+([^\n]+)',
            'date': r'(?:date|tanggal)[\s:]+([^\n]+)',
            'company': r'(?:company|perusahaan)[\s:]+([^\n]+)',
            'department': r'(?:department|departemen)[\s:]+([^\n]+)'
        }
        
        for field, pattern in field_patterns.items():
            if field not in form_info:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value:
                        form_info[field] = value
        
        return form_info
    
    def extract_table_info(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract information from table text
        
        Args:
            text: Table text
            
        Returns:
            Dictionary with extracted table data
        """
        # Split into lines
        lines = text.split('\n')
        
        # Check if this is a pipe-delimited table
        if any('|' in line for line in lines):
            return self.extract_delimited_table(lines, '|')
        
        # Check if this is a tab-delimited table
        if any('\t' in line for line in lines):
            return self.extract_delimited_table(lines, '\t')
        
        # If no delimiters, try to detect columns by whitespace
        return self.extract_space_delimited_table(lines)
    
    def extract_delimited_table(self, lines: List[str], delimiter: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract data from a delimiter-separated table
        
        Args:
            lines: Text lines
            delimiter: Column delimiter
            
        Returns:
            Dictionary with extracted table data
        """
        table_data = {
            'headers': [],
            'rows': []
        }
        
        # Remove empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return table_data
        
        # Extract headers from the first row
        headers_line = non_empty_lines[0]
        headers = [h.strip() for h in headers_line.split(delimiter)]
        # Remove empty headers
        headers = [h for h in headers if h]
        
        if not headers:
            return table_data
        
        table_data['headers'] = headers
        
        # Start from the second row (skip header and separator rows)
        data_start = 1
        while data_start < len(non_empty_lines):
            if all(c == '-' or c == '=' or c.isspace() for c in non_empty_lines[data_start]):
                # This is a separator row
                data_start += 1
            else:
                break
        
        # Process data rows
        for i in range(data_start, len(non_empty_lines)):
            row = non_empty_lines[i]
            
            # Skip separator rows
            if all(c == '-' or c == '=' or c.isspace() for c in row):
                continue
                
            # Split row by delimiter
            values = [v.strip() for v in row.split(delimiter)]
            
            # Create row data with header mapping
            row_data = {}
            for j, value in enumerate(values):
                if j < len(headers):
                    row_data[headers[j]] = value
            
            if row_data:
                table_data['rows'].append(row_data)
        
        return table_data
    
    def extract_space_delimited_table(self, lines: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract data from a space-delimited table (with column alignment)
        
        Args:
            lines: Text lines
            
        Returns:
            Dictionary with extracted table data
        """
        table_data = {
            'headers': [],
            'rows': []
        }
        
        # Remove empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return table_data
        
        # Try to detect column boundaries from spacing patterns
        # We'll look at word positions in the first few rows
        column_starts = []
        max_analysis_rows = min(5, len(non_empty_lines))
        
        for i in range(max_analysis_rows):
            # Find start positions of words in this line
            positions = [m.start() for m in re.finditer(r'\S+', non_empty_lines[i])]
            
            if i == 0:
                # First row - initialize with positions
                column_starts = positions
            else:
                # Merge with existing positions - keep positions that are close to existing ones
                merged_positions = []
                for pos in positions:
                    # Find closest existing column start
                    closest = min(column_starts, key=lambda x: abs(x - pos))
                    
                    # If close enough, adjust existing; otherwise add new
                    if abs(closest - pos) < 5:
                        # Update the existing position as average
                        idx = column_starts.index(closest)
                        column_starts[idx] = (column_starts[idx] + pos) // 2
                    else:
                        merged_positions.append(pos)
                
                # Add new positions
                column_starts.extend(merged_positions)
                column_starts.sort()
        
        if not column_starts:
            return table_data
        
        # Extract headers from first row
        header_line = non_empty_lines[0]
        headers = []
        
        for i in range(len(column_starts)):
            start = column_starts[i]
            end = column_starts[i+1] if i < len(column_starts) - 1 else len(header_line)
            header = header_line[start:end].strip()
            if header:
                headers.append(header)
        
        if not headers:
            return table_data
        
        table_data['headers'] = headers
        
        # Skip header and separator rows
        data_start = 1
        while data_start < len(non_empty_lines):
            if all(c == '-' or c == '=' or c.isspace() for c in non_empty_lines[data_start]):
                # This is a separator row
                data_start += 1
            else:
                break
        
        # Process data rows
        for i in range(data_start, len(non_empty_lines)):
            row = non_empty_lines[i]
            
            # Skip separator rows
            if all(c == '-' or c == '=' or c.isspace() for c in row):
                continue
            
            # Extract values based on column positions
            values = []
            for j in range(len(column_starts)):
                start = column_starts[j]
                end = column_starts[j+1] if j < len(column_starts) - 1 else len(row)
                
                if start < len(row):
                    value = row[start:end].strip()
                    values.append(value)
                else:
                    values.append("")
            
            # Create row data with header mapping
            row_data = {}
            for j, value in enumerate(values):
                if j < len(headers):
                    row_data[headers[j]] = value
            
            if row_data:
                table_data['rows'].append(row_data)
        
        return table_data
    
    def extract_generic_info(self, text: str) -> Dict[str, str]:
        """
        Extract generic key-value information from any text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted key-value pairs
        """
        # Look for field-value pairs using regex
        # Common pattern: Field: Value or Field - Value
        field_value_pattern = r'([A-Za-z][A-Za-z\s]{2,20})[\s:]+([^\n:]{2,100})'
        
        # Extract all field-value pairs
        info = {}
        
        # Process each line
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            matches = re.finditer(field_value_pattern, line)
            for match in matches:
                field = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                
                # Only add if value is meaningful
                if value and not re.match(r'^[:\-,.;]*', value):
                    info[field] = value
        
        return info

def organize_output(results: dict) -> dict:
    """
    Organize OCR results in a clean, structured format
        
    Args:
        esults: OCR results dictionary
        
    Returns:
    Organized results dictionary
    """
    # If results already has an error status, return it unchanged
    if results.get("status") == "error":
        return results
    
    # Make a copy of the results to avoid modifying the original
    organized = results.copy()
    
    # Ensure required fields exist
    if "text" not in organized:
        organized["text"] = ""
    if "confidence" not in organized:
        organized["confidence"] = 0.0
    if "metadata" not in organized:
        organized["metadata"] = {}
    
    # Clean up metadata section
    if "metadata" in organized:
        # Remove redundant or empty fields
        metadata = organized["metadata"]
        # Ensure all required metadata fields exist
        if "detected_language" not in metadata:
            metadata["detected_language"] = "unknown"
        if "image_type" not in metadata:
            metadata["image_type"] = "unknown"
        if "best_engine" not in metadata:
            metadata["best_engine"] = "unknown"
        # Keep processing time if it exists
        if "processing_time_ms" not in metadata:
            metadata["processing_time_ms"] = 0.0
    
    # Add a timestamp to the organized results
    import time
    from datetime import datetime
    organized["timestamp"] = datetime.now().isoformat()
    
    # Format confidence score for better readability
    organized["confidence_level"] = _format_confidence_level(organized.get("confidence", 0))
    
    # Organize structured information if available
    if "metadata" in organized and "structured_info" in organized["metadata"]:
        structured_info = organized["metadata"]["structured_info"]
        if structured_info:
            # Keep original structured info but add a formatted version
            organized["metadata"]["formatted_info"] = _format_structured_info(structured_info)
    
    return organized

def _format_confidence_level(confidence: float) -> str:
    """
    Format confidence score into a descriptive level
    
    Args:
        confidence: Raw confidence score (0-100)
        
    Returns:
        String description of confidence level
    """
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Good"
    elif confidence >= 40:
        return "Moderate"
    elif confidence >= 20:
        return "Low"
    else:
        return "Very Low"

def _format_structured_info(info: dict) -> dict:
    """
    Format structured information for better readability
    
    Args:
        info: Extracted structured information
        
    Returns:
        Formatted structured information
    """
    if not info:
        return {}
    
    formatted = {}
    
    # Format ID card information
    if "name" in info or "id_number" in info:
        formatted["id_card"] = {k: v for k, v in info.items() if k in [
            "name", "id_number", "date_of_birth", "gender", "address",
            "expiration_date", "issue_date", "nationality"
        ]}
    
    # Format receipt information
    if "merchant" in info or "total" in info:
        receipt_info = {k: v for k, v in info.items() if k in [
            "merchant", "date", "time", "subtotal", "tax", "total"
        ]}
        
        # Format items if available
        if "items" in info and isinstance(info["items"], list):
            receipt_info["items_count"] = len(info["items"])
            receipt_info["items"] = info["items"]
            
        formatted["receipt"] = receipt_info
    
    # If no specific type was detected, include all fields
    if not formatted:
        formatted["general"] = info
        
    return formatted

# Fungsi standalone untuk extract_structured_info
def extract_structured_info(text: str, image_type: Any = None) -> Optional[Dict[str, Any]]:
    """
    Standalone function untuk ekstraksi informasi terstruktur
    Wrapper untuk metode InformationExtractor.extract_structured_info untuk kompatibilitas
    
    Args:
        text: Teks yang akan diekstrak informasinya
        image_type: Tipe gambar (opsional)
        
    Returns:
        Dictionary dari informasi terstruktur atau None
    """
    # Membuat instance InformationExtractor dan memanggil metodenya
    extractor = InformationExtractor()
    return extractor.extract_structured_info(text, image_type)