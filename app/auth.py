import os
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

TOKEN = os.getenv("API_TOKEN", "change-me")
bearer_scheme = HTTPBearer(auto_error=False)

def verify_token(cred: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if not cred or cred.credentials != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
