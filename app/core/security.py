import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import jwt

from app.core.settings import settings

logger = logging.getLogger(__name__)

# Single source of truth for the signing key — settings enforces that it is a
# strong value (or aborts startup in production), so signing here always matches
# verification in app.api.deps.
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash (SHA-256 pre-hashed)."""
    try:
        if isinstance(plain_password, bytes):
            plain_password = plain_password.decode("utf-8")
        # Pre-hash with SHA-256 so bcrypt's 72-byte input limit never truncates.
        hashed_input = hashlib.sha256(plain_password.encode()).hexdigest()
        return bcrypt.checkpw(hashed_input.encode(), hashed_password.encode())
    except Exception as e:  # noqa: BLE001
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using SHA-256 + bcrypt."""
    if isinstance(password, bytes):
        password = password.decode("utf-8")
    hashed_input = hashlib.sha256(password.encode()).hexdigest()
    return bcrypt.hashpw(hashed_input.encode(), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


RESET_TOKEN_EXPIRE_MINUTES = 30


def create_reset_token(user_id: int, current_hashed_password: str, expires_minutes: int = RESET_TOKEN_EXPIRE_MINUTES) -> str:
    """Short-lived, single-purpose token for password reset.

    Includes a fingerprint of the current password hash so the token is
    invalidated automatically the moment it's used (or the password is
    changed some other way) — no separate revocation table needed.
    """
    fp = hashlib.sha256(current_hashed_password.encode()).hexdigest()[:16]
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode = {"sub": str(user_id), "type": "password_reset", "pwd_fp": fp, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_reset_token(token: str) -> Optional[dict]:
    """Return the token's payload if it's a validly-signed, unexpired reset
    token, else None. Caller must still compare `pwd_fp` against the target
    user's current hashed password before trusting it."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:  # noqa: BLE001 (jose raises several JWTError subclasses)
        return None
    if payload.get("type") != "password_reset":
        return None
    return payload


def reset_token_matches(payload: dict, current_hashed_password: str) -> bool:
    fp = hashlib.sha256(current_hashed_password.encode()).hexdigest()[:16]
    return payload.get("pwd_fp") == fp
