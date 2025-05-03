import os
import pytest
import tempfile
from app import create_app

@pytest.fixture
def client():
    """Create test client"""
    app = create_app({
        'TESTING': True,
        'UPLOAD_FOLDER': tempfile.mkdtemp(),
        'MARKDOWN_FOLDER': tempfile.mkdtemp()
    })
    
    with app.test_client() as client:
        yield client

def test_api_home(client):
    """Test API home endpoint"""
    response = client.get('/api/')
    assert response.status_code == 200
    assert b'SmartGlass OCR API' in response.data

def test_api_docs(client):
    """Test API docs endpoint"""
    response = client.get('/api/docs')
    assert response.status_code == 200
    assert b'endpoints' in response.data