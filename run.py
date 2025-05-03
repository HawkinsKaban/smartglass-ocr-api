import os
from app import create_app

# Create app instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"Starting SmartGlass OCR API on port {port}")
    print(f"API documentation available at: http://localhost:{port}/api/docs")
    app.run(host='0.0.0.0', port=port, debug=debug)