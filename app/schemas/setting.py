from pydantic import BaseModel, model_validator


class SettingsUpdate(BaseModel):
    """Admin submits keys to store. Omit or leave blank to keep the current value."""
    openrouter_api_key: str | None = None
    voyage_api_key: str | None = None
    openrouter_base_url: str | None = None
    default_chat_model: str | None = None
    fonnte_account_token: str | None = None
    field_encryption_key: str | None = None
    embeddings_provider: str | None = None  # "voyage" | "openrouter" | "huggingface"
    embedding_model_openrouter_main: str | None = None
    embedding_model_openrouter_fallback_1: str | None = None
    embedding_model_openrouter_fallback_2: str | None = None
    huggingface_api_key: str | None = None
    embedding_model_huggingface: str | None = None
    evolution_base_url: str | None = None
    evolution_api_key: str | None = None
    whatsapp_default_provider: str | None = None  # "fonnte" | "evolution"
    paypal_client_id: str | None = None
    paypal_client_secret: str | None = None
    paypal_enabled: bool | None = None
    paypal_sandbox: bool | None = None
    billing_currency: str | None = None  # ISO 4217, e.g. "MYR" | "AUD"


class SettingsOut(BaseModel):
    """Never returns raw secrets — only whether they're set + a last-4 hint."""
    openrouter_api_key_set: bool
    openrouter_api_key_hint: str | None = None
    voyage_api_key_set: bool
    voyage_api_key_hint: str | None = None
    openrouter_base_url: str
    default_chat_model: str
    fonnte_account_token_set: bool
    fonnte_account_token_hint: str | None = None
    field_encryption_key_set: bool
    embeddings_provider: str
    embedding_model_openrouter_main: str
    embedding_model_openrouter_fallback_1: str
    embedding_model_openrouter_fallback_2: str
    huggingface_api_key_set: bool
    huggingface_api_key_hint: str | None = None
    embedding_model_huggingface: str
    evolution_base_url: str
    evolution_api_key_set: bool
    evolution_api_key_hint: str | None = None
    whatsapp_default_provider: str
    paypal_client_id: str
    paypal_client_secret_set: bool
    paypal_client_secret_hint: str | None = None
    paypal_enabled: bool
    paypal_sandbox: bool
    billing_currency: str


class ModelOption(BaseModel):
    id: str
    label: str
    provider: str


class FallbackTier(BaseModel):
    """One rung of a package's chat fallback chain, tried in order until one
    answers successfully. Set per-package under Admin -> Packages."""
    label: str
    provider: str  # "self_hosted" | "openrouter"
    base_url: str | None = None  # required for self_hosted; ignored for openrouter
    model: str

    @model_validator(mode="after")
    def _validate(self) -> "FallbackTier":
        if self.provider not in ("self_hosted", "openrouter"):
            raise ValueError("provider must be 'self_hosted' or 'openrouter'")
        if self.provider == "self_hosted" and not (self.base_url or "").strip():
            raise ValueError(f"base_url is required for self-hosted tier '{self.label}'")
        if not self.model.strip():
            raise ValueError(f"model is required for tier '{self.label}'")
        return self


