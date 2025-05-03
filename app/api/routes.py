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
from app.api.utils import allowed_file, generate_unique_filename, get_markdown_files, convert_numpy_types

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
        'version': '1.1',  # Updated version
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
                    {'name': 'process_type', 'in': 'formData', 'required': False, 'type': 'string', 'description': 'Processing type (auto, fast, accurate, handwritten)'}
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
        else:
            # Auto - reset to defaults
            ocr_processor.ocr_engine.config["lightweight_mode"] = False
            ocr_processor.ocr_engine.config["preprocessing_level"] = "auto"
            ocr_processor.ocr_engine.config["use_all_available_engines"] = True
            
        # Process the file
        results, md_filename = ocr_processor.process_file(
            file_path=file_path,
            original_filename=original_filename,
            language=language,
            page=page,
            summary_length=summary_length,
            summary_style=summary_style
        )
        
        # Convert NumPy types to Python types for JSON serialization
        results = convert_numpy_types(results)
        
        # Update task status to complete
        active_tasks[task_id] = {
            'status': 'complete',
            'response': {
                'status': results.get('status', 'success'),
                'message': 'File processed successfully',
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
    """Detect image type to choose appropriate processing strategy"""
    try:
        import cv2
        import numpy as np
        
        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            return "document", 0, 0, False  # Default for non-image files
            
        # Get dimensions
        height, width = img.shape[:2]
        file_size = os.path.getsize(file_path)
        pixel_count = height * width
        
        # Convert to grayscale for analysis
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Check for handwritten content
        # Simplified handwriting detection:
        # - Calculate edge density using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / (height * width)
        
        # - Calculate contrast
        contrast = np.std(gray)
        
        # - Detect if image is handwritten
        is_handwritten = (edge_density < 0.06 and contrast < 60)
        
        # Calculate expected processing time based on pixel count
        # Handwritten text takes longer to process
        if is_handwritten:
            expected_time = pixel_count / 1000000 * 40  # 40 seconds per million pixels for handwritten
            img_type = "handwritten"
        else:
            expected_time = pixel_count / 1000000 * 20  # 20 seconds per million pixels for normal text
            # Check if it's a document, natural image, or receipt
            if contrast > 70:
                img_type = "document"
            else:
                img_type = "natural"
                
        return img_type, width, height, file_size > 5 * 1024 * 1024  # Is large file?
    except Exception as e:
        logger.warning(f"Error detecting image type: {e}")
        return "document", 0, 0, False

@api_bp.route('/process', methods=['POST'])
def process_file():
    """Process an uploaded file with OCR and generate markdown"""
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
        
        # Automatically detect image characteristics if process_type is auto
        if process_type == 'auto':
            img_type, width, height, is_large = detect_image_type(file_path)
            
            # Adjust processing parameters based on detected type
            if img_type == "handwritten":
                process_type = "handwritten"
                logger.info(f"Detected handwritten content, switching to handwritten mode")
            elif is_large:
                # For large files, use faster processing
                process_type = "fast"
                logger.info(f"Detected large file ({width}x{height}), switching to fast mode")
        
        # Calculate timeout based on file size and processing type
        file_size = os.path.getsize(file_path)
        default_timeout = int(current_app.config.get('OCR_TIMEOUT', 120))
        
        # Adjust timeout based on file size and process type
        if process_type == 'fast':
            timeout = min(default_timeout, 60)  # Shorter timeout for fast mode
        elif process_type == 'handwritten':
            timeout = max(default_timeout, 300)  # Longer timeout for handwritten
        elif file_size > 5 * 1024 * 1024:  # Files over 5MB
            timeout = max(default_timeout, 240)  # Longer timeout for large files
        else:
            timeout = default_timeout
            
        # Check if the file is a PDF and add PyPDF2 if needed
        if file_path.lower().endswith('.pdf'):
            try:
                import PyPDF2
            except ImportError:
                logger.warning("PyPDF2 not installed. Installing...")
                import subprocess
                try:
                    subprocess.check_call(['pip', 'install', 'PyPDF2'])
                except:
                    return jsonify({
                        'status': 'error',
                        'message': 'Missing PDF processing dependencies. Please install PyPDF2.'
                    }), 500
        
        # Generate task ID for tracking
        import uuid
        task_id = str(uuid.uuid4())
        
        # For immediate processing (small files or fast mode)
        if file_size < 1024 * 1024 or process_type == 'fast':
            try:
                # Process synchronously
                results, md_filename = ocr_processor.process_file(
                    file_path=file_path,
                    original_filename=original_filename,
                    language=language,
                    page=page,
                    summary_length=summary_length,
                    summary_style=summary_style
                )
                
                # Convert NumPy types to Python types for JSON serialization
                results = convert_numpy_types(results)
                
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
                    'message': f'Error processing file: {str(e)}'
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
            'message': f'Error processing file: {str(e)}'
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
        
        # Convert NumPy types to Python types for JSON serialization
        response = convert_numpy_types(response)
        
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
        'api_version': '1.1',  # Updated version
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
        'version': '1.1',  # Updated version
        'documentation': '/api/docs',
        'endpoints': [
            {'path': '/api/process', 'method': 'POST', 'description': 'Process an image or PDF file'},
            {'path': '/api/markdown', 'method': 'GET', 'description': 'List all markdown files'},
            {'path': '/api/markdown/<filename>', 'method': 'GET', 'description': 'Get a specific markdown file'},
            {'path': '/api/stats', 'method': 'GET', 'description': 'Get OCR engine statistics'},
            {'path': '/api/task_status/<task_id>', 'method': 'GET', 'description': 'Check status of a long-running OCR task'}
        ]
    })