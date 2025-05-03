import os

class Config:
    """Base configuration"""
    # Application directories
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
    
    # Upload and storage settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(PROJECT_ROOT, 'data', 'uploads'))
    MARKDOWN_FOLDER = os.environ.get('MARKDOWN_FOLDER', os.path.join(PROJECT_ROOT, 'data', 'markdown'))
    
    # File settings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # OCR settings
    OCR_TIMEOUT = 120  # 2 minutes timeout for OCR processing
    DEFAULT_LANGUAGE = 'eng+ind'  # Default OCR language
    DEFAULT_SUMMARY_LENGTH = 200  # Default summary length
    DEFAULT_SUMMARY_STYLE = 'concise'  # Default summary style
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smartglass-ocr-secret')
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'