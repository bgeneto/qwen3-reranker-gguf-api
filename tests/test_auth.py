import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth import verify_token


def test_verify_token_valid():
    # Mock the token
    valid_token = "test-token"
    import os
    os.environ["API_TOKEN"] = valid_token
    
    # Mock the credentials
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=valid_token)
    
    # Should not raise an exception
    verify_token(creds)

def test_verify_token_invalid():
    # Mock the token
    valid_token = "test-token"
    import os
    os.environ["API_TOKEN"] = valid_token
    
    # Mock the credentials
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
    
    with pytest.raises(HTTPException) as exc_info:
        verify_token(creds)
    assert exc_info.value.status_code == 401

def test_verify_token_missing():
    with pytest.raises(HTTPException) as exc_info:
        verify_token(None)
    assert exc_info.value.status_code == 401
