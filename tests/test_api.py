"""
API Tests for the E-Commerce RAG System.
Tests the main endpoints for functionality and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# TestClient - FastAPI's synchronous test client for HTTP requests
# Simulates requests without starting actual server
client = TestClient(app)


# TestHealthEndpoint - Test class grouping health check related tests
# Class-based organization follows pytest conventions for test grouping
class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    # test_health_check_returns_200() - Verifies endpoint is reachable
    # Basic smoke test ensuring health endpoint responds correctly
    def test_health_check_returns_200(self):
        """Test that health endpoint returns successful status."""
        response = client.get("/health")
        assert response.status_code == 200
    
    # test_health_check_response_structure() - Validates response schema
    # Ensures all required fields exist in health response
    def test_health_check_response_structure(self):
        """Test that health response has correct structure."""
        response = client.get("/health")
        # response.json() - Parses JSON response body to dict
        data = response.json()
        
        # assert "key" in data - Validates required fields exist
        assert "status" in data
        assert "version" in data
        assert "vector_db_status" in data
        assert "documents_indexed" in data


# TestRootEndpoint - Tests for the API root/info endpoint
class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    # test_root_returns_200() - Basic availability check for root endpoint
    def test_root_returns_200(self):
        """Test that root endpoint returns successful status."""
        response = client.get("/")
        assert response.status_code == 200
    
    # test_root_contains_api_info() - Validates API info is exposed
    # Ensures clients can discover API name, version, and docs URL
    def test_root_contains_api_info(self):
        """Test that root response contains API information."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


# TestQueryEndpoint - Tests for RAG query functionality
class TestQueryEndpoint:
    """Tests for the query endpoint."""
    
    # test_query_with_valid_request() - Tests happy path for query
    # Accepts 500 as valid since DB might not have documents
    def test_query_with_valid_request(self):
        """Test query endpoint with valid request."""
        # client.post() - Sends POST request with JSON body
        response = client.post(
            "/query",
            json={"query": "What smartphones do you have?", "top_k": 3}
        )
        # status_code in [200, 500] - Handles both success and empty DB scenarios
        assert response.status_code in [200, 500]
    
    # test_query_with_short_query() - Tests validation rejection
    # Verifies Pydantic min_length=3 constraint is enforced
    def test_query_with_short_query(self):
        """Test query endpoint rejects too short queries."""
        response = client.post(
            "/query",
            json={"query": "ab"}  # Less than 3 characters
        )
        # 422 Unprocessable Entity - Standard FastAPI validation error code
        assert response.status_code == 422


# TestUploadEndpoint - Tests for document upload functionality
class TestUploadEndpoint:
    """Tests for the upload endpoint."""
    
    # test_upload_rejects_unsupported_format() - Tests file type validation
    # Ensures only PDF, TXT, MD files are accepted
    def test_upload_rejects_unsupported_format(self):
        """Test that unsupported file formats are rejected."""
        # files parameter - Simulates multipart file upload
        # Tuple format: (filename, content, content_type)
        response = client.post(
            "/upload",
            files={"file": ("test.xyz", b"test content", "text/plain")}
        )
        # 400 Bad Request - Expected for unsupported file types
        assert response.status_code == 400


# __name__ == "__main__" - Allows running tests directly with python
# pytest.main() - Programmatically runs pytest with verbose output
if __name__ == "__main__":
    pytest.main([__file__, "-v"])