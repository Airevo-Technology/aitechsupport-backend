from pydantic import BaseModel, Field

from app.core import config_store

from app.schemas.setting import FallbackTier


class PackageOut(BaseModel):
    id: int
    slug: str
    name: str
    price_myr: int
    monthly_token_quota: int
    max_bots: int
    features: list[str] = []
    is_active: bool
    sort_order: int
    # Whether a bot on this plan can connect WhatsApp (Fonnte) at all.
    whatsapp_enabled: bool = True
    # Ordered chat fallback chain for this plan; null -> inherit the
    # platform-wide chain (see models_catalog.chain_for_package).
    fallback_chain: list[FallbackTier] | None = None
    # ISO currency price_myr is quoted in on this deployment (filled at response time).
    currency: str = Field(default_factory=lambda: (config_store.get("BILLING_CURRENCY") or "MYR").strip().upper())

    class Config:
        from_attributes = True


class PackageUpsert(BaseModel):
    slug: str
    name: str
    price_myr: int = 0
    monthly_token_quota: int = 0
    max_bots: int = 1
    features: list[str] = []
    is_active: bool = True
    sort_order: int = 0
    whatsapp_enabled: bool = True
    fallback_chain: list[FallbackTier] | None = None


class SubscriptionOut(BaseModel):
    plan: str
    plan_name: str
    status: str
    tokens_used: int
    tokens_quota: int
    tokens_remaining: int
    max_bots: int
    period_start: str | None = None
    next_billing_date: str | None = None
    # 0-100: % off every paid checkout/renewal from active paid referrals.
    referral_discount_percent: int = 0


class ReferralItem(BaseModel):
    name: str  # masked, e.g. "Ac***"
    joined: str
    plan: str
    counts: bool  # on an active paid plan today -> earns the referrer 10%


class ReferralOut(BaseModel):
    code: str
    link: str
    percent_each: int
    active_referrals: int
    total_referrals: int
    discount_percent: int
    referrals: list[ReferralItem]


class SubscribeRequest(BaseModel):
    package_slug: str
    # "billplz" | "paypal"; omitted -> the first gateway available on this deployment.
    gateway: str | None = None


class PayPalCaptureRequest(BaseModel):
    order_id: str


class PayPalCaptureOut(BaseModel):
    status: str  # activated | already-paid | not-paid


class GatewaysOut(BaseModel):
    currency: str
    billplz: bool
    paypal: bool
    bank_transfer: bool


class CheckoutOut(BaseModel):
    """Result of POST /billing/checkout.

    - Paid plan: `payment_url` is the Billplz page to redirect the browser to;
      the plan activates via webhook after payment.
    - Free plan / downgrade: `subscription` is the already-applied plan and
      `payment_url` is null (no payment needed).
    """
    payment_url: str | None = None
    bill_id: str | None = None  # Billplz bill id, or PayPal order id
    subscription: SubscriptionOut | None = None


class ClientRow(BaseModel):
    organization_id: int
    organization_name: str
    owner_email: str | None = None
    plan: str
    tokens_used: int
    tokens_quota: int
    bots: int
    created_at: str | None = None
    start_date: str | None = None
    next_billing_date: str | None = None


class ClientsPage(BaseModel):
    items: list[ClientRow]
    total: int
    page: int
    page_size: int


class ClientUserRow(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    phone: str | None = None
    role: str
    is_active: bool
    is_email_verified: bool
    last_seen_at: str | None = None
    created_at: str | None = None


class ClientBotRow(BaseModel):
    id: int
    name: str
    is_active: bool
    chat_model: str | None = None
    handoff_mode: str
    whatsapp_number: str | None = None
    knowledge_sources: int
    channels: list[str]
    created_at: str | None = None


class ClientPaymentRow(BaseModel):
    id: int
    plan_slug: str
    amount_cents: int
    currency: str = "MYR"
    method: str
    status: str
    sandbox: bool
    reference_note: str | None = None
    paid_at: str | None = None
    created_at: str | None = None


class ClientDetail(BaseModel):
    client: ClientRow
    subscription_status: str
    messages_used: int
    period_start: str | None = None
    last_reminder_sent: str | None = None
    support_email: str | None = None
    support_whatsapp: str | None = None
    conversations: int
    conversations_needing_human: int
    last_conversation_at: str | None = None
    users: list[ClientUserRow]
    bots: list[ClientBotRow]
    payments: list[ClientPaymentRow]


class SetBillingCycleRequest(BaseModel):
    start_date: str  # ISO date/datetime; next_billing_date = start_date + 30 days


class PaymentRow(BaseModel):
    id: int
    organization_id: int
    organization_name: str | None = None
    plan_slug: str
    amount_cents: int
    method: str
    status: str
    sandbox: bool
    billplz_bill_id: str | None = None
    paypal_order_id: str | None = None
    reference_note: str | None = None
    reported_at: str | None = None
    paid_at: str | None = None
    created_at: str | None = None


class PaymentsSummary(BaseModel):
    total: int
    paid: int
    pending: int
    failed: int
    live_revenue_cents: int  # sum of paid, non-sandbox bills
    currency: str = "MYR"  # this deployment's billing currency


class PaymentsOut(BaseModel):
    summary: PaymentsSummary
    payments: list[PaymentRow]


class BillplzSettingsOut(BaseModel):
    """Billplz gateway config for the admin panel. Secret keys are never returned
    in full — only whether they're set + a last-4 hint. Live and sandbox are
    separate Billplz accounts, each with its own key/collection; `sandbox`
    is just which set is currently active for checkout."""
    enabled: bool
    sandbox: bool
    configured: bool  # the ACTIVE set has an api key + collection

    live_api_key_set: bool
    live_api_key_hint: str | None = None
    live_x_signature_key_set: bool
    live_x_signature_key_hint: str | None = None
    live_collection_id: str

    sandbox_api_key_set: bool
    sandbox_api_key_hint: str | None = None
    sandbox_x_signature_key_set: bool
    sandbox_x_signature_key_hint: str | None = None
    sandbox_collection_id: str


class BillplzSettingsUpdate(BaseModel):
    """Admin submits fields to store. Omit or leave a secret blank to keep it."""
    enabled: bool | None = None
    sandbox: bool | None = None
    live_api_key: str | None = None            # blank -> keep existing
    live_x_signature_key: str | None = None    # blank -> keep existing
    live_collection_id: str | None = None
    sandbox_api_key: str | None = None            # blank -> keep existing
    sandbox_x_signature_key: str | None = None    # blank -> keep existing
    sandbox_collection_id: str | None = None


class BankTransferSettingsOut(BaseModel):
    """Manual bank-transfer fallback config, shown to customers when enabled."""
    enabled: bool
    bank_name: str
    account_name: str
    account_number: str
    qr_url: str | None = None  # presigned inline URL for the QR image, if uploaded
    notify_channel_id: int | None = None
    notify_whatsapp_number: str
    configured: bool


class BankTransferSettingsUpdate(BaseModel):
    enabled: bool | None = None
    bank_name: str | None = None
    account_name: str | None = None
    account_number: str | None = None
    notify_channel_id: int | None = None
    notify_whatsapp_number: str | None = None


class BankTransferReportRequest(BaseModel):
    package_slug: str
    note: str = ""


class BankTransferReportOut(BaseModel):
    payment_id: int
    status: str
