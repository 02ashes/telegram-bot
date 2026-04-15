"""Authentication — Telegram WebApp initData validation."""

import hashlib
import hmac
import json
import time
from typing import Optional
from urllib.parse import parse_qs


def validate_webapp_data(init_data: str, bot_token: str) -> Optional[dict]:
    """
    Validate Telegram WebApp initData using HMAC-SHA256.
    Returns parsed user dict if valid, None if invalid.
    See: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
    """
    if not init_data:
        return None

    try:
        parsed = parse_qs(init_data, keep_blank_values=True)
        received_hash = parsed.get("hash", [None])[0]
        if not received_hash:
            return None

        # Build data-check-string (all params except hash, sorted alphabetically)
        data_check_pairs = []
        for key, values in sorted(parsed.items()):
            if key == "hash":
                continue
            data_check_pairs.append(f"{key}={values[0]}")
        data_check_string = "\n".join(data_check_pairs)

        # Compute HMAC
        secret_key = hmac.new(
            b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256
        ).digest()
        computed_hash = hmac.new(
            secret_key, data_check_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        if computed_hash != received_hash:
            return None

        # Check auth_date freshness (reject initData older than 7 days)
        auth_date = int(parsed.get("auth_date", ["0"])[0])
        if auth_date and (time.time() - auth_date > 604800):  # 7 days
            return None

        # Extract user info
        user_data = parsed.get("user", [None])[0]
        if user_data:
            return json.loads(user_data)
        return None

    except Exception:
        return None
