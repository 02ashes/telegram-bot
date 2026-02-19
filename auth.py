"""Authentication — invite codes & user management (JSON file storage)."""

import hashlib
import hmac
import json
import os
import time
from typing import Optional
from urllib.parse import parse_qs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CODES_FILE = os.path.join(DATA_DIR, "codes.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fpath in (CODES_FILE, USERS_FILE):
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump({}, f)


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_json(path: str, data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Invite Codes ──────────────────────────────────────────────


def create_code(code: str) -> bool:
    """Create a one-time invite code. Returns False if code already exists."""
    codes = _load_json(CODES_FILE)
    if code in codes:
        return False
    codes[code] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "used_by": None,
    }
    _save_json(CODES_FILE, codes)
    return True


def validate_and_use_code(code: str, user_id: int, username: str) -> bool:
    """Validate invite code and register user. Returns False if invalid/used."""
    codes = _load_json(CODES_FILE)
    if code not in codes or codes[code]["used_by"] is not None:
        return False

    # Mark code as used
    codes[code]["used_by"] = user_id
    codes[code]["used_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_json(CODES_FILE, codes)

    # Register user
    users = _load_json(USERS_FILE)
    users[str(user_id)] = {
        "username": username,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_used": code,
    }
    _save_json(USERS_FILE, users)
    return True


def list_codes() -> dict:
    return _load_json(CODES_FILE)


def delete_code(code: str) -> bool:
    codes = _load_json(CODES_FILE)
    if code not in codes:
        return False
    del codes[code]
    _save_json(CODES_FILE, codes)
    return True


# ── Users ─────────────────────────────────────────────────────


def is_user_registered(user_id: int) -> bool:
    users = _load_json(USERS_FILE)
    return str(user_id) in users


def remove_user(user_id: int) -> bool:
    users = _load_json(USERS_FILE)
    key = str(user_id)
    if key not in users:
        return False
    del users[key]
    _save_json(USERS_FILE, users)
    return True


def list_users() -> dict:
    return _load_json(USERS_FILE)


# ── Telegram WebApp initData validation ───────────────────────


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

        # Extract user info
        user_data = parsed.get("user", [None])[0]
        if user_data:
            return json.loads(user_data)
        return None

    except Exception:
        return None


# Init data dir on import
_ensure_data_dir()
