from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client

from .auth import verify_token
from .config import Settings, get_settings
from .schemas import CurrentUser

# auto_error=False so a missing header produces our own 401 with a
# WWW-Authenticate challenge rather than FastAPI's bare 403.
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


def get_db(user: CurrentUserDep, settings: SettingsDep) -> Client:
    """A Supabase client scoped to the calling user.

    Built with the publishable key and then authenticated with the caller's own
    JWT, so every query runs under their identity and RLS decides what they can
    see. We deliberately do not hold a service_role key -- it bypasses RLS
    entirely, which would move authorisation out of the database and into our
    request handlers.
    """
    client = create_client(settings.supabase_url, settings.supabase_publishable_key)
    client.postgrest.auth(user.token)
    return client


DbDep = Annotated[Client, Depends(get_db)]
