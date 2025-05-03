"""
SmartGlass OCR API - Utilities
Helper functions for the API
"""

import os
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any
from flask import current_app
from werkzeug.utils import secure_filename

def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension
    
    Args:
        filename: The filename to check
        
    Returns:
        True if the file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

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
    secure_name = secure_filename(filename)
    base, ext = os.path.splitext(secure_name)
    return f"{base}_{timestamp}_{unique_id}{ext}"

def get_markdown_files() -> List[Dict[str, Any]]:
    """
    Get a list of all markdown files with metadata
    
    Returns:
        List of dictionaries with markdown file information
    """
    md_files = []
    markdown_folder = current_app.config['MARKDOWN_FOLDER']
    
    for file in os.listdir(markdown_folder):
        if file.endswith('.md'):
            file_path = os.path.join(markdown_folder, file)
            stats = os.stat(file_path)
            md_files.append({
                'filename': file,
                'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'size': stats.st_size,
                'url': f"/api/markdown/{file}"
            })
    
    return sorted(md_files, key=lambda x: x['created'], reverse=True)