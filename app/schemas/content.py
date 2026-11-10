import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.core import config_store


# ===== Homepage (structured JSON blob) =====

class HomepageUpdate(BaseModel):
    content: dict


# ===== Pages =====

class PageListItem(BaseModel):
    slug: str
    title: str

    class Config:
        from_attributes = True


class PublicPageOut(BaseModel):
    slug: str
    title: str
    body: str | None = None
    meta_description: str | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class PageAdminOut(BaseModel):
    id: int
    slug: str
    title: str
    body: str | None = None
    meta_description: str | None = None
    is_published: bool
    sort_order: int
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class PageCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    body: str | None = None
    meta_description: str | None = Field(default=None, max_length=300)
    is_published: bool = True
    sort_order: int = 0

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        v = re.sub(r"[^a-z0-9-]+", "-", v.strip().lower()).strip("-")
        if not v:
            raise ValueError("invalid slug")
        return v


class PageUpdate(BaseModel):
    title: str | None = None
    body: str | None = None
    meta_description: str | None = Field(default=None, max_length=300)
    is_published: bool | None = None
    sort_order: int | None = None


# ===== Public pricing =====

class PublicPackageOut(BaseModel):
    slug: str
    name: str
    price_myr: int
    monthly_token_quota: int
    max_bots: int
    features: list = []
    # ISO currency price_myr is quoted in on this deployment.
    currency: str = Field(default_factory=lambda: (config_store.get("BILLING_CURRENCY") or "MYR").strip().upper())

    class Config:
        from_attributes = True


# ===== Contact form =====

class ContactRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=200)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip()
        if "@" not in v or "." not in v.split("@")[-1] or " " in v:
            raise ValueError("invalid email")
        return v
