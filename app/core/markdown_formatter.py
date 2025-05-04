"""
SmartGlass OCR API - Markdown Formatter
Format OCR results as a beautiful Markdown document
"""

import json
import re
from typing import Dict, Any, List
from datetime import datetime

class MarkdownFormatter:
    """Format OCR results as beautiful Markdown documents"""
    
    @staticmethod
    def format_ocr_results(results: Dict[str, Any], filename: str) -> str:
        """
        Convert OCR results to a well-formatted Markdown document
        
        Args:
            results: OCR results dictionary
            filename: Original filename
            
        Returns:
            Markdown formatted string
        """
        md_content = []
        
        # Add frontmatter with metadata
        md_content.append("---")
        md_content.append(f"title: OCR Results for {filename}")
        md_content.append(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append(f"status: {results.get('status', 'unknown')}")
        
        if 'metadata' in results:
            metadata = results['metadata']
            md_content.append(f"language: {metadata.get('detected_language', 'unknown')}")
            md_content.append(f"confidence: {results.get('confidence', 0):.2f}")
            md_content.append(f"image_type: {metadata.get('image_type', 'unknown')}")
            md_content.append(f"engine: {metadata.get('best_engine', 'unknown')}")
            
            # Add signage-specific metadata if available
            if metadata.get('is_outdoor_signage', False):
                md_content.append(f"content_type: {metadata.get('content_type', 'unknown')}")
        
        md_content.append("---")
        md_content.append("")
        
        # Add title and metadata
        md_content.append(f"# OCR Results: {filename}")
        md_content.append(f"*Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md_content.append("")
        
        # Add processing information
        md_content.append("## Processing Information")
        md_content.append("")
        md_content.append("| Property | Value |")
        md_content.append("| -------- | ----- |")
        md_content.append(f"| Status | `{results.get('status', 'Unknown')}` |")
        
        if 'metadata' in results:
            metadata = results['metadata']
            md_content.append(f"| Processing Time | {metadata.get('processing_time_ms', 0):.2f} ms |")
            md_content.append(f"| Detected Language | {metadata.get('detected_language', 'Unknown')} |")
            md_content.append(f"| Image Type | {metadata.get('image_type', 'Unknown')} |")
            md_content.append(f"| OCR Engine | {metadata.get('best_engine', 'Unknown')} |")
            md_content.append(f"| Confidence | {results.get('confidence', 0):.2f}% |")
            
            # Add content type for signage
            if metadata.get('is_outdoor_signage', False):
                md_content.append(f"| Content Type | {metadata.get('content_type', 'Unknown')} |")
        
        md_content.append("")
        
        # Add signage-specific information
        is_signage = 'metadata' in results and results['metadata'].get('is_outdoor_signage', False)
        
        if is_signage:
            md_content.append("## Sign Analysis")
            md_content.append("")
            md_content.append(results['metadata'].get('description', 'This appears to be a sign or banner.'))
            md_content.append("")
        
        # Add summary if available
        if 'summary' in results and results['summary']:
            md_content.append("## Summary")
            md_content.append("")
            md_content.append("> " + results['summary'].replace("\n", "\n> "))
            md_content.append("")
        
        # Add key insights if available
        if 'key_insights' in results and results['key_insights']:
            md_content.append("## Key Insights")
            md_content.append("")
            for insight in results['key_insights']:
                md_content.append(f"* {insight}")
            md_content.append("")
        
        # Add document structure if available
        if 'document_structure' in results:
            md_content.append("## Document Structure")
            md_content.append("")
            md_content.append(f"Detected structure: **{results['document_structure']}**")
            md_content.append("")
        
        # Add extracted text with appropriate formatting based on document type
        if 'text' in results and results['text']:
            md_content.append("## Extracted Text")
            md_content.append("")
            
            # Format text based on document structure or image type
            doc_structure = results.get('document_structure', '')
            image_type = results.get('metadata', {}).get('image_type', '')
            
            # For signage, use code block to preserve formatting
            if is_signage or image_type == 'signage':
                md_content.append("```")
                # Use original_text if available to preserve formatting
                if 'original_text' in results:
                    md_content.append(results['original_text'])
                else:
                    md_content.append(results['text'])
                md_content.append("```")
            elif any(s in doc_structure.lower() or s in image_type.lower() for s in ['table', 'form']):
                # For tables, try to format as Markdown table if possible
                MarkdownFormatter._format_table_text(results['text'], md_content)
            elif any(s in doc_structure.lower() for s in ['code', 'scientific']):
                # For code or scientific notation, use code blocks
                md_content.append("```")
                md_content.append(results['text'])
                md_content.append("```")
            elif 'bullet' in doc_structure.lower():
                # For bullet points, format as Markdown list
                MarkdownFormatter._format_bullet_text(results['text'], md_content)
            else:
                # Regular text - preserve paragraph structure but make it Markdown friendly
                MarkdownFormatter._format_regular_text(results['text'], md_content)
            
            md_content.append("")
        
        # Add structured data if available
        if 'metadata' in results and 'structured_info' in results['metadata'] and results['metadata']['structured_info']:
            structured_info = results['metadata']['structured_info']
            md_content.append("## Structured Information")
            md_content.append("")
            
            # Format based on the type of structured information
            if isinstance(structured_info, dict):
                if 'items' in structured_info:  # Receipt type
                    MarkdownFormatter._format_receipt_info(structured_info, md_content)
                else:  # General key-value pairs
                    MarkdownFormatter._format_key_value_info(structured_info, md_content)
            else:
                # Fallback to JSON
                md_content.append("```json")
                md_content.append(json.dumps(structured_info, indent=2))
                md_content.append("```")
            
            md_content.append("")
        
        # Add image stats if available
        if 'metadata' in results and 'image_stats' in results['metadata']:
            md_content.append("## Image Statistics")
            md_content.append("")
            md_content.append("| Property | Value |")
            md_content.append("| -------- | ----- |")
            
            stats = results['metadata']['image_stats']
            for key, value in stats.items():
                md_content.append(f"| {key.replace('_', ' ').title()} | {value} |")
            
            md_content.append("")
        
        return "\n".join(md_content)
    
    @staticmethod
    def _format_regular_text(text: str, md_content: List[str]):
        """Format regular text preserving paragraph structure"""
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Clean up any markdown syntax characters that might cause issues
                cleaned = MarkdownFormatter._escape_markdown(paragraph.strip())
                md_content.append(cleaned)
                md_content.append("")  # Add empty line between paragraphs
    
    @staticmethod
    def _format_bullet_text(text: str, md_content: List[str]):
        """Format text with bullet points"""
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                md_content.append("")
                continue
            
            # Check if line already has bullet points
            if line.startswith(('•', '-', '*', '+')) or re.match(r'^\d+[\.\)]', line):
                # Convert existing bullet points to Markdown style
                if line.startswith('•'):
                    line = '* ' + line[1:].strip()
                elif re.match(r'^\d+\.', line):
                    # Already in Markdown numbered list format
                    pass
                elif re.match(r'^\d+\)', line):
                    # Convert 1) to 1. for Markdown
                    line = re.sub(r'(\d+)\)', r'\1.', line)
                
                md_content.append(line)
            else:
                # Add as regular paragraph
                md_content.append(line)
    
    @staticmethod
    def _format_table_text(text: str, md_content: List[str]):
        """Format text as a Markdown table if it looks like a table"""
        lines = text.split("\n")
        has_table_structure = False
        
        # Look for table indicators like | or multiple spaces between words
        for line in lines:
            if '|' in line or re.search(r'\S+\s{3,}\S+', line):
                has_table_structure = True
                break
        
        if has_table_structure:
            # Try to convert to Markdown table
            table_lines = []
            header_processed = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if '|' in line:
                    # Already has pipe separator
                    table_lines.append(line)
                    
                    # Add header separator row after first line
                    if not header_processed:
                        # Count columns
                        cols = len(line.split('|'))
                        separator = '|' + '|'.join(['---'] * (cols-1)) + '|'
                        table_lines.append(separator)
                        header_processed = True
                else:
                    # Try to detect columns based on whitespace
                    cols = re.findall(r'\S+(?:\s{2,}|\s*$)', line)
                    if len(cols) > 1:
                        table_line = '| ' + ' | '.join(c.strip() for c in cols) + ' |'
                        table_lines.append(table_line)
                        
                        # Add header separator row after first line
                        if not header_processed:
                            separator = '|' + '|'.join(['---'] * len(cols)) + '|'
                            table_lines.append(separator)
                            header_processed = True
                    else:
                        # Not a table row, add as regular text
                        table_lines.append(line)
            
            md_content.extend(table_lines)
        else:
            # Not a table, format as regular text
            MarkdownFormatter._format_regular_text(text, md_content)
    
    @staticmethod
    def _format_receipt_info(receipt_info: Dict[str, Any], md_content: List[str]):
        """Format receipt structured info"""
        # Add merchant and date if available
        if 'merchant' in receipt_info:
            md_content.append(f"**Merchant**: {receipt_info['merchant']}")
        
        if 'date' in receipt_info:
            md_content.append(f"**Date**: {receipt_info['date']}")
        
        if 'time' in receipt_info:
            md_content.append(f"**Time**: {receipt_info['time']}")
        
        md_content.append("")
        
        # Add items as a table
        if 'items' in receipt_info and receipt_info['items']:
            md_content.append("### Items")
            md_content.append("")
            md_content.append("| Item | Quantity | Price |")
            md_content.append("| ---- | -------- | ----- |")
            
            for item in receipt_info['items']:
                name = item.get('name', '')
                qty = item.get('quantity', '1')
                price = item.get('price', '0.00')
                md_content.append(f"| {name} | {qty} | {price} |")
            
            md_content.append("")
        
        # Add totals
        md_content.append("### Totals")
        md_content.append("")
        
        if 'subtotal' in receipt_info:
            md_content.append(f"**Subtotal**: {receipt_info['subtotal']}")
        
        if 'tax' in receipt_info:
            md_content.append(f"**Tax**: {receipt_info['tax']}")
        
        if 'total' in receipt_info:
            md_content.append(f"**Total**: {receipt_info['total']}")
        
        if 'payment_method' in receipt_info:
            md_content.append(f"**Payment Method**: {receipt_info['payment_method']}")
    
    @staticmethod
    def _format_key_value_info(info: Dict[str, Any], md_content: List[str]):
        """Format general key-value structured info"""
        md_content.append("| Field | Value |")
        md_content.append("| ----- | ----- |")
        
        for key, value in info.items():
            # Format key for display
            display_key = key.replace('_', ' ').title()
            
            # Handle different value types
            if isinstance(value, dict):
                # For nested dict, convert to JSON
                formatted_value = f"View Details\n\n```json\n{json.dumps(value, indent=2)}\n```\n\n"
                md_content.append(f"| {display_key} | {formatted_value} |")
            elif isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    # List of dictionaries - show as expandable
                    formatted_value = f"View {len(value)} items\n\n```json\n{json.dumps(value, indent=2)}\n```\n\n"
                    md_content.append(f"| {display_key} | {formatted_value} |")
                else:
                    # Simple list - format as comma-separated
                    formatted_value = ", ".join(str(item) for item in value)
                    md_content.append(f"| {display_key} | {formatted_value} |")
            else:
                # Simple value
                md_content.append(f"| {display_key} | {value} |")
    
    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape Markdown syntax characters"""
        # Escape characters that have special meaning in Markdown
        for char in ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!']:
            text = text.replace(char, '\\' + char)
        
        return text