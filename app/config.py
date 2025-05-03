import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    OCR_TIMEOUT = int(os.environ.get('OCR_TIMEOUT', 120))  # 2 minutes timeout for OCR processing
    DEFAULT_LANGUAGE = os.environ.get('DEFAULT_LANGUAGE', 'eng+ind')  # Default OCR language
    DEFAULT_SUMMARY_LENGTH = int(os.environ.get('DEFAULT_SUMMARY_LENGTH', 200))  # Default summary length
    DEFAULT_SUMMARY_STYLE = os.environ.get('DEFAULT_SUMMARY_STYLE', 'concise')  # Default summary style
    
    # Tesseract configuration
    TESSERACT_PATH = os.environ.get('TESSERACT_PATH', '')
    TESSERACT_DATA_PATH = os.environ.get('TESSERACT_DATA_PATH', '')
    
    # Performance settings
    LIGHTWEIGHT_MODE = os.environ.get('LIGHTWEIGHT_MODE', 'false').lower() == 'true'
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'smartglass-ocr-secret')
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    # Use temporary directories for testing
    import tempfile
    UPLOAD_FOLDER = tempfile.mkdtemp()
    MARKDOWN_FOLDER = tempfile.mkdtemp()

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Additional production settings
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max upload size in production

# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}