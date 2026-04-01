"""
JWT authentication utilities.

POST /token  → returns access token (HS256, configurable expiry)
get_current_user dependency → validates Bearer token on protected routes
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from api.config import settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# ---------------------------------------------------------------------------
# Fake user store  (replace with DB lookup in production)
# ---------------------------------------------------------------------------
FAKE_USERS_DB: dict[str, dict] = {
    settings.API_USERNAME: {
        "username": settings.API_USERNAME,
        "hashed_password": pwd_context.hash(settings.API_PASSWORD),
    }
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = FAKE_USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = FAKE_USERS_DB.get(token_data.username)
    if user is None:
        raise credentials_exception
    return user
