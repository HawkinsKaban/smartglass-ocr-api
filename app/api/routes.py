"""
SmartGlass OCR API - Routes
API endpoints for OCR processing and file management
"""

import os
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, Response, current_app

from app.core.ocr_processor import OCRProcessor
from app.api.utils import allowed_file, generate_unique_filename, get_markdown_files

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize OCR Processor
ocr_processor = OCRProcessor()

# Initialize start time for uptime tracking
start_time = time.time()

@api_bp.route('/docs')
def docs():
    """API Documentation"""
    documentation = {
        'name': 'SmartGlass OCR API',
        'version': '1.0',
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
                    {'name': 'summary_style', 'in': 'formData', 'required': False, 'type': 'string', 'description': 'Summary style (concise, detailed, bullets, structured)'}
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
                'path': '/api/markdown/',
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
            }
        ]
    }
    
    return jsonify(documentation)

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
        
        # Process the file
        results, md_filename = ocr_processor.process_file(
            file_path=file_path,
            original_filename=original_filename,
            language=language,
            page=page,
            summary_length=summary_length,
            summary_style=summary_style
        )
        
        # Prepare response
        response = {
            'status': 'success',
            'message': 'File processed successfully',
            'results': results,
            'markdown_file': md_filename,
            'markdown_url': f"/api/markdown/{md_filename}"
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error processing file: {str(e)}'
        }), 500

@api_bp.route('/markdown', methods=['GET'])
def list_markdown_files():
    """List all markdown files"""
    files = get_markdown_files()
    return jsonify({'files': files})

@api_bp.route('/markdown/', methods=['GET'])
def get_markdown_file(filename):
    """Get a specific markdown file, either as download or raw content"""
    file_path = os.path.join(current_app.config['MARKDOWN_FOLDER'], filename)
    
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
    api_stats = {
        'api_version': '1.0',
        'ocr_engine': stats,
        'markdown_files': len(get_markdown_files()),
        'uptime': int(time.time() - start_time)
    }
    
    return jsonify(api_stats)

# Root endpoint for API
@api_bp.route('/', methods=['GET'])
def api_home():
    """API home endpoint"""
    return jsonify({
        'name': 'SmartGlass OCR API',
        'version': '1.0',
        'documentation': '/api/docs',
        'endpoints': [
            {'path': '/api/process', 'method': 'POST', 'description': 'Process an image or PDF file'},
            {'path': '/api/markdown', 'method': 'GET', 'description': 'List all markdown files'},
            {'path': '/api/markdown/', 'method': 'GET', 'description': 'Get a specific markdown file'},
            {'path': '/api/stats', 'method': 'GET', 'description': 'Get OCR engine statistics'}
        ]
    })
