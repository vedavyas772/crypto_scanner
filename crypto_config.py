# =============================================================================
# crypto_config.py — Crypto Pivot Scanner Configuration
#
# Reads secrets from ENVIRONMENT VARIABLES (required for Railway deployment).
# For local Mac testing, falls back to a local .env file if present, so you
# don't have to manually export variables every time you test locally.
#
# On Railway: set these in the Variables tab of your service — no .env file
# needed there, Railway injects them as real environment variables.
#
# Locally: create a file named ".env" (no quotes) in this same folder with:
#   TWILIO_SID=AC55c7d233d419dbb2819cb3c74d27d861
#   TWILIO_TOKEN=your_actual_token
#   TWILIO_TO=whatsapp:+918688342467
#   (etc — see full list below)
# =============================================================================

import os

# ── Load local .env file if present (for Mac testing only) ───────────────────
def _load_dotenv_if_present():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                # Don't override real environment variables (Railway takes priority)
                if key and key not in os.environ:
                    os.environ[key] = value

_load_dotenv_if_present()


def _get_env(key, default=None, required=False):
    val = os.environ.get(key, default)
    if required and (val is None or val == default and default in (None, "")):
        print(f"  ⚠️  WARNING: Environment variable '{key}' is not set.")
    return val


# ── Twilio WhatsApp ───────────────────────────────────────────────────────────
TWILIO_SID   = _get_env("TWILIO_SID",   "your_twilio_sid")
TWILIO_TOKEN = _get_env("TWILIO_TOKEN", "your_twilio_token")
TWILIO_FROM  = _get_env("TWILIO_FROM",  "whatsapp:+14155238886")
TWILIO_TO    = _get_env("TWILIO_TO",    "whatsapp:+91XXXXXXXXXX")

# ── Telegram (activate after 5 days) ─────────────────────────────────────────
TELEGRAM_TOKEN   = _get_env("TELEGRAM_TOKEN",   "paste_your_token_here")
TELEGRAM_CHAT_ID = _get_env("TELEGRAM_CHAT_ID", "paste_your_chat_id_here")

# ── Scanner Parameters (not secrets, safe to hardcode) ────────────────────────
TOP_N_COINS          = int(_get_env("TOP_N_COINS", "50"))
SCAN_INTERVAL_MIN    = 60      # Scan every 60 minutes (1H TF)
EMA_FAST             = 9       # EMA 9
EMA_SLOW             = 26      # EMA 26
RETEST_CANDLES       = 5       # Wait max 5 candles for retest after crossover
PIVOT_TYPE           = "Fibonacci"  # Fibonacci pivot points
PIVOT_ANCHOR         = "Weekly"     # Weekly reset — same as Pine Script on 1H TF
EMA_TOUCH_BUFFER_PCT = 0.001        # 0.1% buffer for wick touching EMA 9

# ── Startup diagnostic — confirms secrets loaded without printing them ────────
if __name__ != "__main__":
    _sid_ok   = TWILIO_SID   != "your_twilio_sid"
    _token_ok = TWILIO_TOKEN != "your_twilio_token"
    _to_ok    = TWILIO_TO    != "whatsapp:+91XXXXXXXXXX"
    print(f"  Config loaded — Twilio SID: {'✅ set' if _sid_ok else '❌ MISSING'}, "
          f"Token: {'✅ set' if _token_ok else '❌ MISSING'}, "
          f"To: {'✅ set' if _to_ok else '❌ MISSING'}")
