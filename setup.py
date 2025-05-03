#!/usr/bin/env python3
"""
SmartGlass OCR API - Setup Script
This script helps set up the environment for running the SmartGlass OCR API.
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def print_step(step, text):
    """Print a step in the setup process"""
    print(f"[{step}] {text}")

def run_command(command, error_message="Command failed"):
    """Run a shell command and handle errors"""
    try:
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {error_message}")
        print(f"Command '{command}' failed with exit code {e.returncode}")
        return False

def yes_no_prompt(question, default="yes"):
    """Ask a yes/no question and return the answer"""
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"Invalid default answer: '{default}'")
    
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_step("1", "Checking Python version...")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"Error: Python 3.8 or higher is required. You have Python {major}.{minor}")
        sys.exit(1)
    
    print(f"Python version {major}.{minor} is compatible.")
    return True

def setup_virtual_environment():
    """Set up a Python virtual environment"""
    print_step("2", "Setting up virtual environment...")
    
    # Check if virtual environment already exists
    venv_dir = "venv"
    if os.path.exists(venv_dir):
        if yes_no_prompt(f"Virtual environment already exists at '{venv_dir}'. Remove and recreate?", "no"):
            shutil.rmtree(venv_dir)
        else:
            print("Using existing virtual environment.")
            return True
    
    # Create virtual environment
    result = run_command(f"{sys.executable} -m venv {venv_dir}", "Failed to create virtual environment")
    if not result:
        return False
    
    print(f"Virtual environment created at '{venv_dir}'")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_step("3", "Installing Python dependencies...")
    
    # Determine virtual environment activation script
    activate_script = "venv/bin/activate" if platform.system() != "Windows" else "venv\\Scripts\\activate"
    
    # Install required packages
    if platform.system() != "Windows":
        result = run_command(f"source {activate_script} && pip install -r requirements.txt", 
                           "Failed to install dependencies")
    else:
        result = run_command(f"{activate_script} && pip install -r requirements.txt", 
                           "Failed to install dependencies")
    
    if not result:
        return False
    
    print("Dependencies installed successfully.")
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print_step("4", "Checking Tesseract OCR installation...")
    
    tesseract_path = None
    system = platform.system()
    
    if system == "Windows":
        # Try common Windows Tesseract paths
        common_paths = [
            "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
        ]
        for path in common_paths:
            if os.path.exists(path):
                tesseract_path = path
                break
    else:
        # Try common Unix Tesseract paths
        try:
            tesseract_path = subprocess.check_output(["which", "tesseract"]).decode().strip()
        except:
            # Try common locations
            common_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    tesseract_path = path
                    break
    
    if tesseract_path:
        print(f"Tesseract OCR found at: {tesseract_path}")
        
        # Test Tesseract version
        try:
            version = subprocess.check_output([tesseract_path, "--version"]).decode()
            print(f"Tesseract version: {version.splitlines()[0]}")
        except:
            print("Warning: Found Tesseract but couldn't determine version.")
        
        return tesseract_path
    else:
        print("Tesseract OCR not found.")
        print("Please install Tesseract OCR manually:")
        if system == "Windows":
            print("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        elif system == "Darwin":  # macOS
            print("Run: brew install tesseract")
        else:  # Linux
            print("Run: sudo apt-get install tesseract-ocr")
        
        print("\nAfter installation, update the TESSERACT_PATH in the .env file.")
        return None

def setup_data_directories():
    """Set up data directories needed by the application"""
    print_step("5", "Creating data directories...")
    
    directories = ["data/uploads", "data/markdown"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True

def update_env_file(tesseract_path=None):
    """Update the .env file with appropriate values"""
    print_step("6", "Setting up configuration...")
    
    # Check if .env file exists
    env_file = ".env"
    env_example = ".env.example"
    
    # Create .env file from example if it doesn't exist
    if not os.path.exists(env_file):
        if not os.path.exists(env_example):
            with open(env_file, "w") as f:
                f.write("# Security settings\n")
                f.write(f"SECRET_KEY={os.urandom(24).hex()}\n\n")
                f.write("# Storage settings\n")
                f.write("UPLOAD_FOLDER=data/uploads\n")
                f.write("MARKDOWN_FOLDER=data/markdown\n\n")
                f.write("# Server settings\n")
                f.write("DEBUG=true\n")
                f.write("PORT=5000\n\n")
                f.write("# OCR settings\n")
                f.write("OCR_TIMEOUT=120\n")
                f.write("DEFAULT_LANGUAGE=eng+ind\n")
                f.write("DEFAULT_SUMMARY_LENGTH=200\n")
                f.write("DEFAULT_SUMMARY_STYLE=concise\n\n")
                f.write("# Tesseract path settings\n")
                if tesseract_path:
                    f.write(f"TESSERACT_PATH={tesseract_path}\n")
                else:
                    f.write("TESSERACT_PATH=\n")
                f.write("TESSERACT_DATA_PATH=\n\n")
                f.write("# Performance settings\n")
                f.write("LIGHTWEIGHT_MODE=false\n")
        else:
            shutil.copy(env_example, env_file)
            print(f"Created {env_file} from {env_example}")
    
    # Update Tesseract path in .env if found
    if tesseract_path:
        env_content = []
        with open(env_file, "r") as f:
            env_content = f.readlines()
        
        with open(env_file, "w") as f:
            for line in env_content:
                if line.startswith("TESSERACT_PATH="):
                    f.write(f"TESSERACT_PATH={tesseract_path}\n")
                else:
                    f.write(line)
        
        print(f"Updated Tesseract path in {env_file}")
    
    return True

def download_nltk_data():
    """Download NLTK data needed for text processing"""
    print_step("7", "Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("NLTK data downloaded successfully.")
        return True
    except Exception as e:
        print(f"Warning: Failed to download NLTK data: {e}")
        print("You can manually download the data by running:")
        print("python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords')\"")
        return False

def main():
    """Main setup function"""
    print_header("SmartGlass OCR API - Setup")
    
    print("This script will help you set up the SmartGlass OCR API environment.")
    print("It will check dependencies, create directories, and configure the application.")
    print()
    
    if not yes_no_prompt("Do you want to continue?"):
        sys.exit(0)
    
    # Perform setup steps
    check_python_version()
    setup_virtual_environment()
    install_dependencies()
    tesseract_path = check_tesseract()
    setup_data_directories()
    update_env_file(tesseract_path)
    download_nltk_data()
    
    # Print completion message
    print_header("Setup Complete")
    print("The SmartGlass OCR API environment has been set up successfully.")
    print("\nTo run the application:")
    
    if platform.system() != "Windows":
        print("1. Activate the virtual environment: source venv/bin/activate")
    else:
        print("1. Activate the virtual environment: venv\\Scripts\\activate")
    
    print("2. Start the application: python run.py")
    print("\nThe API will be available at: http://localhost:5000")
    print("API documentation at: http://localhost:5000/api/docs")
    
    if not tesseract_path:
        print("\nWarning: Tesseract OCR was not found.")
        print("Please install it and update the TESSERACT_PATH in the .env file.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())