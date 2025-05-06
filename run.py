import os
import logging
from app import create_app

# Configure clean terminal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Filter out verbose DEBUG messages from certain modules
for module in ['urllib3', 'PIL', 'matplotlib', 'paddle']:
    logging.getLogger(module).setLevel(logging.WARNING)

# Create app instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print("\n" + "=" * 70)
    print(f" SmartGlass OCR API Server Starting ".center(70, "="))
    print("=" * 70)
    print(f"→ Server running on: http://localhost:{port}")
    print(f"→ API documentation: http://localhost:{port}/api/docs")
    print("=" * 70 + "\n")
    
    # Run the app with cleaner output
    app.run(host='0.0.0.0', port=port, debug=debug)