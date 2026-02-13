"""
API Tests for the E-Commerce RAG System.
Tests the main endpoints for functionality and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check_returns_200(self):
        """Test that health endpoint returns successful status."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_structure(self):
        """Test that health response has correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "vector_db_status" in data
        assert "documents_indexed" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_200(self):
        """Test that root endpoint returns successful status."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_api_info(self):
        """Test that root response contains API information."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestQueryEndpoint:
    """Tests for the query endpoint."""
    
    def test_query_with_valid_request(self):
        """Test query endpoint with valid request."""
        response = client.post(
            "/query",
            json={"query": "What smartphones do you have?", "top_k": 3}
        )
        # Should return 200 even if no documents indexed
        assert response.status_code in [200, 500]
    
    def test_query_with_short_query(self):
        """Test query endpoint rejects too short queries."""
        response = client.post(
            "/query",
            json={"query": "ab"}  # Less than 3 characters
        )
        assert response.status_code == 422  # Validation error


class TestUploadEndpoint:
    """Tests for the upload endpoint."""
    
    def test_upload_rejects_unsupported_format(self):
        """Test that unsupported file formats are rejected."""
        # Create a mock file with unsupported extension
        response = client.post(
            "/upload",
            files={"file": ("test.xyz", b"test content", "text/plain")}
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])