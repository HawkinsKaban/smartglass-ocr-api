"""
SmartGlass OCR API - Routes
API endpoints for OCR processing and file management
"""

import os
import time
import logging
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, Response, current_app

from app.core.ocr_processor import OCRProcessor
from app.api.utils import allowed_file, generate_unique_filename, get_markdown_files, convert_numpy_types, clean_response_text

# Configure logging
logger = logging.getLogger("API-Routes")

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize OCR Processor
ocr_processor = OCRProcessor()

# Initialize start time for uptime tracking
start_time = time.time()

# Track active processing tasks
active_tasks = {}

@api_bp.route('/docs')
def docs():
    """API Documentation"""
    documentation = {
        'name': 'SmartGlass OCR API',
        'version': '1.2',  # Updated version
        'description': 'RESTful API for SmartGlassOCR engine with Markdown output',
        'endpoints': [
            {
                'path': '/api/process',
                'method': 'POST',
                'description': 'Process an image or PDF file',
                'parameters': [
                    {'name': 'file', 'in': 'formData', 'required': True, 'type': 'file', 'description': 'File to process'},
                    {'name': 'language', 'in': 'formData', 'required': False, 'type': 'string', 'description': 'OCR language (e.g., "eng", "eng+ind")'},
                    {'name': 'page', 'in': 'formData', 'required': False, 'type': 'integer', 'description': 'Page number for PDF (0-based)'},
                    {'name': 'summary_length', 'in': 'formData', 'required': False, 'type': 'integer', 'description': 'Maximum summary length'},
                    {'name': 'summary_style', 'in': 'formData', 'required': False, 'type': 'string', 'description': 'Summary style (concise, detailed, bullets, structured)'},
                    {'name': 'process_type', 'in': 'formData', 'required': False, 'type': 'string', 'description': 'Processing type (auto, fast, accurate, handwritten, signage)'}
                ],
                'responses': {
                    '200': {
                        'description': 'Successful processing',
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'status': {'type': 'string'},
                                'message': {'type': 'string'},
                                'results': {'type': 'object'},
                                'markdown_file': {'type': 'string'},
                                'markdown_url': {'type': 'string'}
                            }
                        }
                    },
                    '400': {'description': 'Bad request - invalid file or parameters'},
                    '408': {'description': 'Request timeout - processing took too long'},
                    '500': {'description': 'Server error during processing'}
                }
            },
            {
                'path': '/api/markdown',
                'method': 'GET',
                'description': 'List all markdown files',
                'responses': {
                    '200': {
                        'description': 'List of markdown files',
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'files': {'type': 'array', 'items': {'type': 'object'}}
                            }
                        }
                    }
                }
            },
            {
                'path': '/api/markdown/<filename>',
                'method': 'GET',
                'description': 'Get a specific markdown file',
                'parameters': [
                    {'name': 'filename', 'in': 'path', 'required': True, 'type': 'string', 'description': 'Markdown filename'},
                    {'name': 'raw', 'in': 'query', 'required': False, 'type': 'boolean', 'description': 'Set to true to get raw markdown content'}
                ],
                'responses': {
                    '200': {'description': 'Markdown file content'},
                    '404': {'description': 'File not found'}
                }
            },
            {
                'path': '/api/stats',
                'method': 'GET',
                'description': 'Get OCR engine statistics',
                'responses': {
                    '200': {
                        'description': 'OCR engine statistics',
                        'schema': {'type': 'object'}
                    }
                }
            },
            {
                'path': '/api/task_status/<task_id>',
                'method': 'GET',
                'description': 'Check status of a long-running OCR task',
                'parameters': [
                    {'name': 'task_id', 'in': 'path', 'required': True, 'type': 'string', 'description': 'Task ID'}
                ],
                'responses': {
                    '200': {'description': 'Task status and result if complete'},
                    '404': {'description': 'Task not found'}
                }
            }
        ]
    }
    
    return jsonify(documentation)

def process_file_worker(task_id, file_path, original_filename, language, page, summary_length, summary_style, process_type):
    """Background worker to process file with OCR"""
    try:
        # Configure processor based on process_type
        if process_type == 'fast':
            ocr_processor.ocr_engine.config["lightweight_mode"] = True
            ocr_processor.ocr_engine.config["preprocessing_level"] = "minimal"
        elif process_type == 'accurate':
            ocr_processor.ocr_engine.config["lightweight_mode"] = False
            ocr_processor.ocr_engine.config["preprocessing_level"] = "aggressive"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = True
        elif process_type == 'handwritten':
            ocr_processor.ocr_engine.config["preprocessing_level"] = "handwritten"
            ocr_processor.ocr_engine.config["max_image_dimension"] = 1800  # Limit size for handwritten
        elif process_type == 'signage':
            ocr_processor.ocr_engine.config["preprocessing_level"] = "aggressive"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = True
            ocr_processor.ocr_engine.config["adaptive_binarization"] = True
            ocr_processor.ocr_engine.config["edge_enhancement"] = True
            ocr_processor.ocr_engine.config["perspective_correction"] = True
        elif process_type == 'id_card':
            ocr_processor.ocr_engine.config["preprocessing_level"] = "id_card"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = False  # Use only Tesseract for ID cards
            ocr_processor.ocr_engine.config["lightweight_mode"] = True  # Use lightweight mode for speed
            ocr_processor.ocr_engine.config["adaptive_binarization"] = True
            ocr_processor.ocr_engine.config["edge_enhancement"] = True
        else:
            # Auto - reset to defaults
            ocr_processor.ocr_engine.config["lightweight_mode"] = False
            ocr_processor.ocr_engine.config["preprocessing_level"] = "auto"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = True
            
        # Process the file
        result = ocr_processor.process_file(
            file_path=file_path,
            original_filename=original_filename,
            language=language,
            page=page,
            summary_length=summary_length,
            summary_style=summary_style
        )
        
        # Fix: Check if the result is a tuple and extract its values
        if isinstance(result, tuple):
            results, md_filename = result
        else:
            # Handle case where result is not a tuple
            results = result
            md_filename = ""
        
        # Ensure results is a dictionary
        if not isinstance(results, dict):
            results = {
                "status": "error",
                "message": "Invalid result format from OCR processor",
                "metadata": {
                    "processing_time_ms": round((time.time() - active_tasks[task_id]['start_time']) * 1000, 2)
                }
            }
        
        # Convert NumPy types to Python types for JSON serialization
        results = convert_numpy_types(results)
        
        # Ensure we have original_text (in case it's not set by the processor)
        if 'text' in results and 'original_text' not in results:
            results['original_text'] = results['text']
        
        # Clean text and summary in the results
        if 'text' in results:
            results['text'] = clean_response_text(results['text'])
        if 'summary' in results:
            results['summary'] = clean_response_text(results['summary'])
        if 'key_insights' in results and isinstance(results['key_insights'], list):
            results['key_insights'] = [clean_response_text(insight) for insight in results['key_insights']]
        
        # Enhanced formatting for signage/banners
        if process_type == 'signage' or ('metadata' in results and 
                                        isinstance(results['metadata'], dict) and 
                                        results['metadata'].get('is_outdoor_signage')):
            # Format a more descriptive response
            content_type = results.get('metadata', {}).get('content_type', 'unknown').title()
            description = results.get('metadata', {}).get('description', '')
            
            # Add content type and description to response
            results['content_type'] = content_type
            results['description'] = description
        
        # Update task status to complete
        active_tasks[task_id] = {
            'status': 'complete',
            'response': {
                'status': results.get('status', 'success'),
                'message': 'File processed successfully' if results.get('status') != 'error' else results.get('message', 'Processing failed'),
                'results': results,
                'markdown_file': md_filename,
                'markdown_url': f"/api/markdown/{md_filename}"
            }
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error processing task {task_id}: {str(e)}\n{error_trace}")
        
        # Update task status to error
        active_tasks[task_id] = {
            'status': 'error',
            'response': {
                'status': 'error',
                'message': f'Error processing file: {str(e)}',
                'metadata': {'processing_time_ms': time.time() - active_tasks[task_id]['start_time']}
            }
        }

def detect_image_type(file_path):
    """
    Enhanced image type detection with improved accuracy for various document types
    
    Args:
        file_path: Path to the image file
            
    Returns:
        Tuple of (image_type, width, height, is_large_file)
    """
    try:
        import cv2
        import numpy as np
        
        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            return "document", 0, 0, False
            
        # Get dimensions
        height, width = img.shape[:2]
        file_size = os.path.getsize(file_path)
        pixel_count = height * width
        
        # Convert to grayscale for analysis
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Calculate aspect ratio - important for document type detection
        aspect_ratio = width / height
        
        # Enhanced ID card detection - common ID cards have aspect ratios between 1.4 and 1.7
        is_id_card = 1.4 < aspect_ratio < 1.7
        
        # Enhanced ID card detection for Indonesian KTP and other IDs
        if is_id_card:
            # Attempt basic OCR to detect KTP keywords
            try:
                import pytesseract
                # Use PSM 11 (sparse text) for quick keyword detection
                text_sample = pytesseract.image_to_string(gray, config='--psm 11 --oem 1')
                text_lower = text_sample.lower()
                
                # Check for patterns common in Indonesian ID cards
                id_keywords = ["nik", "provinsi", "kabupaten", "kecamatan", "ktp", 
                               "agama", "status perkawinan", "kewarganegaraan",
                               "identity card", "nama", "tempat/tgl lahir", "gol. darah"]
                               
                id_keyword_count = sum(1 for kw in id_keywords if kw in text_lower)
                
                if id_keyword_count >= 2:  # Detected at least 2 keywords
                    return "id_card", width, height, file_size > 5 * 1024 * 1024
            except:
                pass
            
            # Simple check for text regions in ID cards
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_regions = len(contours)
            
            # ID cards typically have multiple text regions
            if text_regions > 8:
                return "id_card", width, height, file_size > 5 * 1024 * 1024
        
        # Enhanced receipt detection - typically tall and narrow with specific content
        if aspect_ratio < 0.7:  # Tall and narrow
            try:
                import pytesseract
                text_sample = pytesseract.image_to_string(gray, config='--psm 6 --oem 1')
                text_lower = text_sample.lower()
                
                # Check for receipt keywords
                receipt_keywords = ["total", "subtotal", "cash", "change", "tax", "amount", 
                                   "item", "qty", "price", "payment", "receipt", "invoice",
                                   "jumlah", "tunai", "kembalian", "pajak", "harga", "kasir",
                                   "pembayaran", "diskon", "tanggal", "waktu"]
                                   
                receipt_keyword_count = sum(1 for kw in receipt_keywords if kw in text_lower)
                
                if receipt_keyword_count >= 2:
                    return "receipt", width, height, file_size > 5 * 1024 * 1024
            except:
                pass
        
        # Calculate edge density using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / (height * width)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Check color variance (useful for signage)
        if len(img.shape) > 2:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv[:,:,1])  # Saturation standard deviation
            has_strong_colors = color_std > 50
        else:
            has_strong_colors = False
        
        # Enhanced handwritten text detection
        # Handwritten text typically has lower edge density and medium contrast
        is_handwritten = False
        try:
            # Calculate texture features for handwriting detection
            # Handwriting typically has specific texture patterns
            from skimage.feature import local_binary_pattern
            
            # Resize for faster processing
            small_gray = cv2.resize(gray, (min(width, 500), int(min(width, 500) * height / width)))
            lbp = local_binary_pattern(small_gray, 8, 1, method='uniform')
            histogram = np.histogram(lbp, bins=10, range=(0, 10))[0] / sum(np.histogram(lbp, bins=10, range=(0, 10))[0])
            
            # Handwritten text has specific LBP histogram patterns
            handwriting_score = histogram[1] + histogram[2] - histogram[8] - histogram[9]
            
            # Additional check with edge coherence
            edge_coherence = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            is_handwritten = handwriting_score > 0.1 and edge_density < 0.06 and 20 < contrast < 70 and edge_coherence < 500
        except:
            # Fallback to simple heuristic if texture analysis fails
            is_handwritten = edge_density < 0.05 and 20 < contrast < 60
        
        if is_handwritten:
            return "handwritten", width, height, file_size > 5 * 1024 * 1024
        
        # Enhanced table detection
        # Tables typically have grid-like structures
        is_table = False
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine to get form structure
            table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Enhance form field boundaries
            cell_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            table_structure = cv2.dilate(table_structure, cell_kernel, iterations=1)
            contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we have enough potential cells and good ratio of horizontal/vertical lines
            horizontal_lines_count = np.count_nonzero(horizontal_lines) / (width * 3)  # Normalized by width
            vertical_lines_count = np.count_nonzero(vertical_lines) / (height * 3)  # Normalized by height
            
            is_table = (len(contours) > 5 and 
                      horizontal_lines_count > 0.05 and 
                      vertical_lines_count > 0.05 and
                      0.2 < horizontal_lines_count / max(vertical_lines_count, 0.001) < 5)
            
            if is_table:
                return "table", width, height, file_size > 5 * 1024 * 1024
        except:
            pass
        
        # Detect if image is a sign/banner
        is_signage = (
            (contrast > 60 or has_strong_colors) and  # Good contrast or strong colors
            (edge_density < 0.08) and  # Not too dense with edges
            (aspect_ratio > 1.5 or aspect_ratio < 0.67)  # Wide/tall
        )
        
        if is_signage:
            return "signage", width, height, file_size > 5 * 1024 * 1024
        
        # Determine general document type for remaining images
        if contrast > 70 and edge_density > 0.04:
            return "document", width, height, file_size > 5 * 1024 * 1024
        elif contrast < 40 and edge_density < 0.03:
            return "low_quality", width, height, file_size > 5 * 1024 * 1024
        else:
            return "natural", width, height, file_size > 5 * 1024 * 1024
            
    except Exception as e:
        logger.warning(f"Error detecting image type: {e}")
        return "document", 0, 0, False

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
        import cv2
        import numpy as np
        
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return file_path
            
        # Resize to manageable dimensions
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

def _convert_pdf_to_image(pdf_path: str, page_num: int = 0):
    """
    Enhanced PDF to image conversion with better quality for OCR
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-based)
        
    Returns:
        Tuple of (path to converted image, total pages)
    """
    if not 'PDF2IMAGE_AVAILABLE' in globals() or not PDF2IMAGE_AVAILABLE:
        logger.error("pdf2image library not available")
        return None, 0
        
    try:
        # Check total pages with PyPDF2 with better error handling
        try:
            from PyPDF2 import PdfReader
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
            
            # Get page dimensions for better conversion quality
            page = pdf.pages[min(page_num, total_pages-1)]
            width, height = page.mediabox.width, page.mediabox.height
            is_landscape = width > height
            
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {e}")
            # Fallback method to get page count
            try:
                import subprocess
                # Use pdfinfo if available
                try:
                    output = subprocess.check_output(['pdfinfo', pdf_path]).decode('utf-8')
                    for line in output.split('\n'):
                        if 'Pages:' in line:
                            total_pages = int(line.split('Pages:')[1].strip())
                            break
                    else:
                        # If Pages: line not found, use pdf2image
                        from pdf2image import convert_from_path
                        total_pages = len(convert_from_path(pdf_path, 72, first_page=1, last_page=5))
                except:
                    # If pdfinfo fails, use pdf2image
                    from pdf2image import convert_from_path
                    total_pages = len(convert_from_path(pdf_path, 72, first_page=1, last_page=5))
            except Exception as e2:
                logger.error(f"Error getting PDF info: {e2}")
                total_pages = 1  # Assume at least 1 page
            
            # Default to unknown orientation
            is_landscape = False
        
        # Validate page number
        if page_num < 0 or page_num >= total_pages:
            logger.error(f"Invalid page number: {page_num}, total pages: {total_pages}")
            return None, total_pages
        
        # Convert with pdf2image using high DPI for better OCR
        try:
            # Higher DPI for better quality, with memory optimization
            # Adjust DPI based on PDF complexity
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                page = doc[min(page_num, doc.page_count-1)]
                
                # Check if PDF has text content already (searchable PDF)
                text = page.get_text()
                has_text = len(text.strip()) > 100
                
                # If PDF already has text content, use higher DPI
                dpi = 600 if has_text else 300
                
                # Adjust DPI based on page size to prevent memory issues
                page_area = page.rect.width * page.rect.height
                if page_area > 1000000:  # Large page
                    dpi = 300
            except:
                # If PyMuPDF fails, use default DPI
                dpi = 300
                has_text = False
            
            # Convert using optimal settings
            from pdf2image import convert_from_path
            pages = convert_from_path(
                pdf_path,
                dpi,
                first_page=page_num + 1,
                last_page=page_num + 1,
                use_cropbox=True,
                transparent=False
            )
            
            if pages:
                # Save as high-quality PNG for better OCR
                image_path = f"{pdf_path}_page_{page_num}.png"
                pages[0].save(image_path, 'PNG')
                
                # Apply image enhancements for better OCR
                try:
                    import cv2
                    img = cv2.imread(image_path)
                    
                    # Apply image preprocessing if needed
                    if not has_text:  # Only enhance if it's not a searchable PDF
                        # Convert to grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Apply adaptive histogram equalization for better contrast
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced = clahe.apply(gray)
                        
                        # Denoise if necessary
                        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
                        
                        # Save enhanced image
                        cv2.imwrite(image_path, denoised)
                except Exception as e:
                    logger.warning(f"Image enhancement failed: {e}")
                
                return image_path, total_pages
        except Exception as e:
            logger.error(f"Error using pdf2image: {e}")
            
            # Try with PyMuPDF as alternative
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                page = doc[min(page_num, doc.page_count-1)]
                
                # Render page to image with high resolution
                zoom = 4.0  # Higher zoom for better quality
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Save image
                image_path = f"{pdf_path}_page_{page_num}.png"
                pix.save(image_path)
                
                return image_path, doc.page_count
            except Exception as e2:
                logger.error(f"Error using PyMuPDF: {e2}")
        
        # Return error
        return None, total_pages
            
    except Exception as e:
        logger.error(f"Error converting PDF: {e}")
        return None, 0

@api_bp.route('/process', methods=['POST'])
def process_file():
    """Process an uploaded file with OCR and generate markdown"""
    # Initialize start_time at the beginning of the function
    start_time = time.time()
    
    # Check if file is included in the request
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'File type not supported'}), 400
    
    try:
        # Save uploaded file
        original_filename = file.filename
        unique_filename = generate_unique_filename(original_filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Get optional parameters
        language = request.form.get('language', current_app.config['DEFAULT_LANGUAGE'])
        page = int(request.form.get('page', 0))
        summary_length = int(request.form.get('summary_length', current_app.config['DEFAULT_SUMMARY_LENGTH']))
        summary_style = request.form.get('summary_style', current_app.config['DEFAULT_SUMMARY_STYLE'])
        process_type = request.form.get('process_type', 'auto').lower()
        
        # Add a specific parameter for handling outdoor signs
        handle_as_signage = request.form.get('handle_as_signage', 'auto').lower()
        
        # Automatically detect image characteristics if process_type is auto
        if process_type == 'auto':
            img_type, width, height, is_large = detect_image_type(file_path)
            
            # Adjust processing parameters based on detected type
            if img_type == "handwritten":
                process_type = "handwritten"
                logger.info(f"Detected handwritten content, switching to handwritten mode")
            elif img_type == "signage" or handle_as_signage == 'true':
                process_type = "signage"
                logger.info(f"Detected signage or banner, switching to signage mode")
            elif img_type == "id_card":
                process_type = "id_card"  # New process type for ID cards
                logger.info(f"Detected ID card, switching to ID card mode")
                
                # For ID cards, preprocess the image before OCR
                optimized_path = preprocess_id_card(file_path)
                if optimized_path != file_path:
                    file_path = optimized_path
                    logger.info(f"Preprocessed ID card image saved to {optimized_path}")
                    
                # Try specialized ID card processing
                try:
                    # Use direct ID card processing method for faster results
                    logger.info("Using specialized ID card processing for KTP")
                    
                    start_time = time.time()
                    results = ocr_processor.ocr_engine.process_id_card(
                        image_path=file_path,
                        language=language
                    )
                    
                    # Add processing time
                    processing_time = time.time() - start_time
                    results['metadata']['processing_time_ms'] = round(processing_time * 1000, 2)
                    
                    # Generate markdown file
                    md_content = ocr_processor.markdown_formatter.format_ocr_results(results, original_filename)
                    md_filename = ocr_processor._save_markdown_file(md_content, original_filename)
                    
                    # Prepare response
                    response = {
                        'status': results.get('status', 'success'),
                        'message': 'File processed successfully',
                        'results': results,
                        'markdown_file': md_filename,
                        'markdown_url': f"/api/markdown/{md_filename}"
                    }
                    
                    return jsonify(response)
                
                except Exception as e:
                    logger.error(f"Error in specialized ID card processing: {str(e)}")
                    # Continue with standard processing if specialized method fails
                
            elif img_type == "table":
                process_type = "accurate"  # Use accurate mode for tables
                logger.info(f"Detected table, switching to accurate mode")
            elif is_large:
                # For large files, use faster processing
                process_type = "fast"
                logger.info(f"Detected large file ({width}x{height}), switching to fast mode")
        
        # Update OCR processor config based on process type
        if process_type == 'signage':
            ocr_processor.ocr_engine.config["preprocessing_level"] = "aggressive"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = True
            ocr_processor.ocr_engine.config["adaptive_binarization"] = True
            ocr_processor.ocr_engine.config["edge_enhancement"] = True
            ocr_processor.ocr_engine.config["perspective_correction"] = True
        elif process_type == 'id_card':  # Add configuration for ID cards
            ocr_processor.ocr_engine.config["preprocessing_level"] = "id_card"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = False  # Use only Tesseract for ID cards
            ocr_processor.ocr_engine.config["adaptive_binarization"] = True
            ocr_processor.ocr_engine.config["edge_enhancement"] = True
            ocr_processor.ocr_engine.config["lightweight_mode"] = True  # Use lightweight mode for speed
        
        # Calculate timeout based on file size and processing type
        file_size = os.path.getsize(file_path)
        default_timeout = int(current_app.config.get('OCR_TIMEOUT', 120))
        
        # Adjust timeout based on file size and process type
        if process_type == 'fast':
            timeout = min(default_timeout, 60)  # Shorter timeout for fast mode
        elif process_type == 'handwritten':
            timeout = max(default_timeout, 300)  # Longer timeout for handwritten
        elif process_type == 'id_card':
            timeout = max(default_timeout, 600)  # Increased timeout to 10 minutes for ID cards
        elif file_size > 5 * 1024 * 1024:  # Files over 5MB
            timeout = max(default_timeout, 240)  # Longer timeout for large files
        else:
            timeout = default_timeout
            
        # Check if the file is a PDF
        if file_path.lower().endswith('.pdf'):
            logger.info(f"Processing PDF file: {file_path}")
            
            # Convert PDF to image first
            image_path, total_pages = _convert_pdf_to_image(file_path, page)
            
            if not image_path:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to convert PDF to image'
                }), 500
                
            # Update file path to use the converted image
            file_path = image_path
            logger.info(f"PDF converted to image: {image_path}")
        
        # Generate task ID for tracking
        import uuid
        task_id = str(uuid.uuid4())
        
        # For immediate processing (small files or fast mode)
        if file_size < 1024 * 1024 or process_type == 'fast':
            try:
                # Process synchronously
                result = ocr_processor.process_file(
                    file_path=file_path,
                    original_filename=original_filename,
                    language=language,
                    page=page,
                    summary_length=summary_length,
                    summary_style=summary_style
                )
                
                # Check if the result is a tuple and extract its values
                if isinstance(result, tuple):
                    results, md_filename = result
                else:
                    # Handle case where result is not a tuple
                    results = result
                    md_filename = ""
                
                # Ensure results is a dictionary
                if not isinstance(results, dict):
                    results = {
                        "status": "error",
                        "message": "Invalid result format from OCR processor",
                        "metadata": {
                            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                        }
                    }
                
                # Convert NumPy types to Python types for JSON serialization
                results = convert_numpy_types(results)
                
                # Ensure we have original_text (in case it's not set by the processor)
                if 'text' in results and 'original_text' not in results:
                    results['original_text'] = results['text']
                
                # Clean text and summary in the results
                if 'text' in results:
                    results['text'] = clean_response_text(results['text'])
                if 'summary' in results:
                    results['summary'] = clean_response_text(results['summary'])
                if 'key_insights' in results and isinstance(results['key_insights'], list):
                    results['key_insights'] = [clean_response_text(insight) for insight in results['key_insights']]
                
                # Enhanced formatting for signage/banners
                if process_type == 'signage' or ('metadata' in results and results['metadata'].get('is_outdoor_signage')):
                    # Format a more descriptive response
                    content_type = results['metadata'].get('content_type', 'unknown').title()
                    description = results['metadata'].get('description', '')
                    
                    # Add content type and description to response
                    results['content_type'] = content_type
                    results['description'] = description
                
                # Prepare response
                response = {
                    'status': results.get('status', 'success'),
                    'message': 'File processed successfully',
                    'results': results,
                    'markdown_file': md_filename,
                    'markdown_url': f"/api/markdown/{md_filename}"
                }
                
                return jsonify(response)
                
            except TimeoutError:
                return jsonify({
                    'status': 'error',
                    'message': 'OCR processing timed out. Try with a smaller image or PDF.',
                    'metadata': {'processing_time_ms': timeout * 1000}
                }), 408
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing file: {str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing file: {str(e)}',
                    'metadata': {
                        'processing_time_ms': round((time.time() - start_time) * 1000, 2)
                    }
                }), 500
        
        # For asynchronous processing (larger files or complex processing)
        else:
            # Create a background task
            active_tasks[task_id] = {
                'status': 'processing',
                'file_path': file_path,
                'original_filename': original_filename,
                'start_time': time.time(),
                'timeout': timeout
            }
            
            # Start background thread for processing
            worker_thread = threading.Thread(
                target=process_file_worker,
                args=(task_id, file_path, original_filename, language, page, 
                     summary_length, summary_style, process_type)
            )
            worker_thread.daemon = True
            worker_thread.start()
            
            # Return task ID for status checking
            return jsonify({
                'status': 'processing',
                'message': 'File processing started in background',
                'task_id': task_id,
                'check_status_url': f"/api/task_status/{task_id}",
                'estimated_time_seconds': timeout
            })
        
    except Exception as e:
        import traceback
        logger.error(f"Error in process_file: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing file: {str(e)}',
            'metadata': {
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }
        }), 500

@api_bp.route('/task_status/<task_id>', methods=['GET'])
def check_task_status(task_id):
    """Check status of a background OCR task"""
    if task_id not in active_tasks:
        return jsonify({
            'status': 'error',
            'message': 'Task not found'
        }), 404
    
    task = active_tasks[task_id]
    
    # Check if task has completed or errored
    if task['status'] in ['complete', 'error']:
        response = task['response']
        
        # Ensure consistent status messages
        if task['status'] == 'error' and 'status' in response and response['status'] != 'error':
            response['status'] = 'error'
        
        # Convert NumPy types to Python types for JSON serialization
        response = convert_numpy_types(response)
        
        # Clean text in results if not already cleaned
        if 'results' in response:
            results = response['results']
            
            # Store original text before cleaning if not already stored
            if 'text' in results and 'original_text' not in results:
                results['original_text'] = results['text']
                
            if 'text' in results:
                results['text'] = clean_response_text(results['text'])
            if 'summary' in results:
                results['summary'] = clean_response_text(results['summary'])
            if 'key_insights' in results and isinstance(results['key_insights'], list):
                results['key_insights'] = [clean_response_text(insight) for insight in results['key_insights']]
        
        # Task is complete, remove from active tasks after a delay
        # Keep task info for a short time in case client checks again
        def cleanup_task():
            time.sleep(60)  # Keep task info for 1 minute
            if task_id in active_tasks:
                del active_tasks[task_id]
                
        cleanup_thread = threading.Thread(target=cleanup_task)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return jsonify(response)
    
    # Check if task has timed out
    if time.time() - task['start_time'] > task['timeout']:
        # Update task status to error
        active_tasks[task_id] = {
            'status': 'error',
            'response': {
                'status': 'error',
                'message': 'OCR processing timed out. Try with a smaller image or PDF.',
                'metadata': {'processing_time_ms': task['timeout'] * 1000}
            }
        }
        
        return jsonify(active_tasks[task_id]['response']), 408
    
    # Task is still processing
    elapsed_time = time.time() - task['start_time']
    return jsonify({
        'status': 'processing',
        'message': 'File is still being processed',
        'elapsed_time_seconds': int(elapsed_time),
        'estimated_remaining_seconds': max(1, int(task['timeout'] - elapsed_time))
    })

@api_bp.route('/markdown', methods=['GET'])
def list_markdown_files():
    """List all markdown files"""
    files = get_markdown_files()
    return jsonify({'files': files})

@api_bp.route('/markdown/<filename>', methods=['GET'])
def get_markdown_file(filename):
    """Get a specific markdown file, either as download or raw content"""
    # Normalize path with os.path.normpath to handle path separators consistently
    file_path = os.path.normpath(os.path.join(current_app.config['MARKDOWN_FOLDER'], filename))
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    
    # Check if raw parameter is set
    raw = request.args.get('raw', 'false').lower() == 'true'
    
    if raw:
        # Return raw markdown content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return Response(content, mimetype='text/markdown')
    else:
        # Return file for download
        return send_file(file_path, as_attachment=True, download_name=filename)

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get OCR engine statistics"""
    stats = ocr_processor.get_statistics()
    
    # Convert NumPy types to Python types for JSON serialization
    stats = convert_numpy_types(stats)
    
    api_stats = {
        'api_version': '1.2',  # Updated version
        'ocr_engine': stats,
        'markdown_files': len(get_markdown_files()),
        'uptime': int(time.time() - start_time),
        'active_tasks': len([t for t in active_tasks.values() if t['status'] == 'processing'])
    }
    
    return jsonify(api_stats)

# Clean up stale tasks periodically
def cleanup_stale_tasks():
    """Remove stale tasks that have timed out"""
    current_time = time.time()
    tasks_to_remove = []
    
    for task_id, task in active_tasks.items():
        if task['status'] == 'processing' and current_time - task['start_time'] > task['timeout'] + 60:
            # Task has timed out and we've waited an extra minute
            tasks_to_remove.append(task_id)
            
    for task_id in tasks_to_remove:
        del active_tasks[task_id]
        
    # Schedule next cleanup
    threading.Timer(300, cleanup_stale_tasks).start()  # Run every 5 minutes

# Start cleanup thread
cleanup_thread = threading.Timer(300, cleanup_stale_tasks)
cleanup_thread.daemon = True
cleanup_thread.start()

# Root endpoint for API
@api_bp.route('/', methods=['GET'])
def api_home():
    """API home endpoint"""
    return jsonify({
        'name': 'SmartGlass OCR API',
        'version': '1.2',  # Updated version
        'documentation': '/api/docs',
        'endpoints': [
            {'path': '/api/process', 'method': 'POST', 'description': 'Process an image or PDF file'},
            {'path': '/api/markdown', 'method': 'GET', 'description': 'List all markdown files'},
            {'path': '/api/markdown/<filename>', 'method': 'GET', 'description': 'Get a specific markdown file'},
            {'path': '/api/stats', 'method': 'GET', 'description': 'Get OCR engine statistics'},
            {'path': '/api/task_status/<task_id>', 'method': 'GET', 'description': 'Check status of a long-running OCR task'}
        ]
    })