"""
app/routers/auth.py
POST /api/auth/token   — login, returns JWT
POST /api/auth/register — create analyst (admin-only)
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
    require_admin,
    get_current_user,
)
from app.models.db import Analyst, AnalystRole
from app.models.schemas import TokenResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=8, max_length=72)
    role: str = Field(default="analyst", pattern=r"^(analyst|admin)$")
    tenant_id: str = Field(default="default", max_length=64)

    @field_validator("email")
    @classmethod
    def lower_email(cls, v: str) -> str:
        return v.lower().strip()


@router.post("/token", response_model=TokenResponse)
@limiter.limit("10/minute")   # brute-force protection
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    analyst = db.query(Analyst).filter_by(email=form.username.lower()).first()
    if not analyst or not verify_password(form.password, analyst.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({
        "sub": str(analyst.id),
        "role": analyst.role.value,
        "tenant_id": analyst.tenant_id,
    })
    return TokenResponse(access_token=token)


@router.post("/register", status_code=201)
async def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
    _admin: dict = Depends(require_admin),   # only admins can create accounts
):
    if db.query(Analyst).filter_by(email=body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    analyst = Analyst(
        email=body.email,
        hashed_password=hash_password(body.password),
        role=AnalystRole(body.role),
        tenant_id=body.tenant_id,
    )
    db.add(analyst)
    db.commit()
    db.refresh(analyst)
    return {"id": analyst.id, "email": analyst.email, "role": analyst.role.value}


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    """Returns current user info. Requires any valid JWT."""
    return user