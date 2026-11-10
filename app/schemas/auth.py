from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    organization_name: str
    email: EmailStr
    password: str
    full_name: str | None = None
    phone: str | None = None
    referral_code: str | None = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id: int
    organization_id: int
    email: EmailStr
    full_name: str | None = None
    phone: str | None = None
    role: str
    is_platform_admin: bool = False
    is_email_verified: bool = True

    class Config:
        from_attributes = True


class CheckEmailRequest(BaseModel):
    email: EmailStr


class CheckEmailResponse(BaseModel):
    exists: bool


class CheckoutSignupRequest(BaseModel):
    """Single-page checkout signup: no password — one is generated, and the
    user sets their real one via the emailed verify link."""
    organization_name: str
    email: EmailStr
    full_name: str | None = None
    phone: str | None = None
    plan_slug: str
    referral_code: str | None = None


class UpdateProfileRequest(BaseModel):
    full_name: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
