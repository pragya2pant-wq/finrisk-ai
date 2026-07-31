"""
PyTest Configuration and Global Fixtures.

Provides reusable test clients and shared setup for test suites.
Author: Pragya Pant
Institute: iPEC Solutions
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend root directory is in Python path for test execution
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Fixture providing a simulated FastAPI HTTP test client.
    
    Returns:
        TestClient: Fast, in-memory API test client.
    """
    with TestClient(app) as test_client:
        yield test_client