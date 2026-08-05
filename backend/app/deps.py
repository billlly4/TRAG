from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, ClientOptions, create_client

from .auth import verify_token
from .config import Settings, get_settings
from .schemas import CurrentUser

# auto_error=False so a missing header produces our own 401
_bearer = HTTPBearer(auto_error=False)

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_current_user(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> CurrentUser:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = verify_token(credentials.credentials, settings)
    return CurrentUser(
        id=claims["sub"],
        email=claims.get("email"),
        token=credentials.credentials,
    )


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]


def create_user_client(settings: Settings, token: str) -> Client:
    """A Supabase client scoped to a user's JWT.

    The Authorization header goes in at construction so PostgREST, Storage and
    RPC all run under the caller's identity and RLS decides what they can see.
    (The old `postgrest.auth(token)` approach authenticated table queries only
    -- Storage requests would have gone out as `anon` and been denied.)

    We deliberately do not hold a service_role key -- it bypasses RLS entirely,
    which would move authorisation out of the database and into our request
    handlers. Standalone (not request-scoped) so ingestion background jobs can
    build a client from the JWT they carry.
    """
    options = ClientOptions(
        headers={"Authorization": f"Bearer {token}"},
        auto_refresh_token=False,
        persist_session=False,
    )
    return create_client(
        settings.supabase_url, settings.supabase_publishable_key, options
    )


def get_db(user: CurrentUserDep, settings: SettingsDep) -> Client:
    return create_user_client(settings, user.token)


DbDep = Annotated[Client, Depends(get_db)]
