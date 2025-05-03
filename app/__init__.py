import os
from flask import Flask

def create_app(test_config=None):
    """Create and configure the Flask application"""
    # Create app instance
    app = Flask(__name__)
    
    # Load default configuration
    from app.config import Config
    app.config.from_object(Config)
    
    # Load test configuration if provided
    if test_config:
        app.config.update(test_config)
    
    # Ensure upload and markdown directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MARKDOWN_FOLDER'], exist_ok=True)
    
    # Register API blueprint
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

def register_error_handlers(app):
    """Register error handlers for the application"""
    from flask import jsonify
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'status': 'error', 'message': 'Resource not found'}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'status': 'error', 'message': 'File too large'}), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500