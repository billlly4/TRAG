import jwt
from fastapi import HTTPException, status
from jwt import PyJWKClient

from .config import Settings

# Supabase signs with asymmetric keys. RS256 is included because projects
# migrated from the legacy symmetric secret may still present it.
_ALGORITHMS = ["ES256", "RS256"]

_jwk_client: PyJWKClient | None = None


def _jwks(settings: Settings) -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(settings.jwks_url, cache_keys=True)
    return _jwk_client


def verify_token(token: str, settings: Settings) -> dict:
    try:
        signing_key = _jwks(settings).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=_ALGORITHMS,
            audience="authenticated",
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "sub", "aud"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
