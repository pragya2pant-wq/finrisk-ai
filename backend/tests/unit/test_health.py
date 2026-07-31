"""
Unit Tests for System Health Endpoint.

Verifies API operational status, metadata accuracy, and author details.
Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi.testclient import TestClient


def test_health_endpoint_status_code(client: TestClient) -> None:
    """
    Verify that the health check endpoint returns HTTP 200 OK.
    """
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint_payload(client: TestClient) -> None:
    """
    Verify that the health check JSON payload contains expected fields and correct authorship metadata.
    """
    response = client.get("/")
    data = response.json()
    
    # Assert expected dictionary keys exist
    assert "status" in data
    assert "platform" in data
    assert "version" in data
    assert "author" in data
    assert "institute" in data
    
    # Assert exact values
    assert data["status"] == "online"
    assert data["author"] == "Pragya Pant"
    assert data["institute"] == "iPEC Solutions"