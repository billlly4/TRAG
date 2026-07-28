"""Supabase JWT verification.

Verification happens locally against Supabase's JWKS endpoint. The alternative,
calling `supabase.auth.get_user(token)`, costs a network round-trip on every
single request; PyJWKClient caches the signing keys instead.
"""

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
    """Return the token's claims, or raise 401.

    `audience` and `issuer` are validated, not just the signature. A token
    signed by the right key but minted for a different audience is still not a
    valid credential for this API -- omitting that check is the usual mistake
    and makes the whole verification close to worthless.
    """
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
