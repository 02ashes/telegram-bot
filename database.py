"""
Database module — PostgreSQL via asyncpg.

Tables:
  users        — user profiles, premium status, token balance
  generations  — generation history (prompts, seeds, results)

Usage:
    from database import db
    await db.init()  # call once at startup
    user = await db.get_or_create_user(telegram_id, username)
"""

import asyncpg
import logging
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

# Global connection pool
_pool: asyncpg.Pool | None = None


# ============================================================
# Schema
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    telegram_id     BIGINT UNIQUE NOT NULL,
    username        TEXT DEFAULT '',
    first_name      TEXT DEFAULT '',
    is_premium      BOOLEAN DEFAULT FALSE,
    premium_until   TIMESTAMPTZ,
    tokens          INTEGER DEFAULT 0,
    total_spent     INTEGER DEFAULT 0,
    total_gens      INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS generations (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id),
    telegram_id     BIGINT NOT NULL,
    prompt          TEXT NOT NULL,
    negative        TEXT DEFAULT '',
    mode            TEXT DEFAULT 'generate',
    character       TEXT DEFAULT '',
    seed            BIGINT,
    width           INTEGER,
    height          INTEGER,
    steps           INTEGER,
    cfg             REAL,
    file_id         TEXT DEFAULT '',
    duration_sec    REAL,
    tokens_spent    INTEGER DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_generations_user ON generations(user_id);
CREATE INDEX IF NOT EXISTS idx_generations_telegram ON generations(telegram_id);
CREATE INDEX IF NOT EXISTS idx_users_telegram ON users(telegram_id);
"""


# ============================================================
# Init / Shutdown
# ============================================================
async def init():
    """Initialize database connection pool and create tables."""
    global _pool
    if not config.DATABASE_URL:
        logger.warning("DATABASE_URL not set — database disabled")
        return

    logger.info("Connecting to PostgreSQL...")
    _pool = await asyncpg.create_pool(
        config.DATABASE_URL,
        min_size=2,
        max_size=10,
    )

    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    logger.info("✅ Database ready")


async def close():
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database connection closed")


def is_enabled() -> bool:
    """Check if database is available."""
    return _pool is not None


# ============================================================
# Users
# ============================================================
async def get_or_create_user(
    telegram_id: int,
    username: str = "",
    first_name: str = "",
) -> dict | None:
    """Get existing user or create new one. Returns dict."""
    if not _pool:
        return None

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE telegram_id = $1", telegram_id
        )
        if row:
            # Update username/first_name if changed
            if row["username"] != username or row["first_name"] != first_name:
                await conn.execute(
                    """UPDATE users SET username = $2, first_name = $3,
                       updated_at = NOW() WHERE telegram_id = $1""",
                    telegram_id, username, first_name,
                )
            return dict(row)

        # Create new user
        row = await conn.fetchrow(
            """INSERT INTO users (telegram_id, username, first_name)
               VALUES ($1, $2, $3) RETURNING *""",
            telegram_id, username, first_name,
        )
        logger.info(f"New user: {telegram_id} ({username})")
        return dict(row)


async def get_user(telegram_id: int) -> dict | None:
    """Get user by telegram_id."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE telegram_id = $1", telegram_id
        )
        return dict(row) if row else None


async def is_premium(telegram_id: int) -> bool:
    """Check if user has active premium."""
    user = await get_user(telegram_id)
    if not user:
        return False
    if not user["is_premium"]:
        return False
    if user["premium_until"] and user["premium_until"] < datetime.now(timezone.utc):
        return False
    return True


async def get_balance(telegram_id: int) -> int:
    """Get user's token balance."""
    user = await get_user(telegram_id)
    return user["tokens"] if user else 0


async def add_tokens(telegram_id: int, amount: int) -> int:
    """Add tokens to user's balance. Returns new balance."""
    if not _pool:
        return 0
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE users SET tokens = tokens + $2,
               updated_at = NOW()
               WHERE telegram_id = $1 RETURNING tokens""",
            telegram_id, amount,
        )
        return row["tokens"] if row else 0


async def spend_tokens(telegram_id: int, amount: int = 1) -> bool:
    """Spend tokens. Returns True if successful, False if not enough."""
    if not _pool:
        return True  # No DB = free mode

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE users SET tokens = tokens - $2,
               total_spent = total_spent + $2,
               updated_at = NOW()
               WHERE telegram_id = $1 AND tokens >= $2
               RETURNING tokens""",
            telegram_id, amount,
        )
        return row is not None


async def set_premium(telegram_id: int, days: int = 30) -> None:
    """Grant premium for N days. Extends if already active."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """UPDATE users SET is_premium = TRUE,
               premium_until = GREATEST(COALESCE(premium_until, NOW()), NOW()) + ($2 || ' days')::interval,
               updated_at = NOW()
               WHERE telegram_id = $1""",
            telegram_id, str(days),
        )


async def revoke_premium(telegram_id: int) -> None:
    """Remove premium."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """UPDATE users SET is_premium = FALSE, premium_until = NULL,
               updated_at = NOW() WHERE telegram_id = $1""",
            telegram_id,
        )


async def delete_user(telegram_id: int) -> bool:
    """Delete user and their generations. Returns True if user existed."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        # Delete generations first (FK constraint)
        await conn.execute(
            "DELETE FROM generations WHERE telegram_id = $1", telegram_id
        )
        result = await conn.execute(
            "DELETE FROM users WHERE telegram_id = $1", telegram_id
        )
        return result == "DELETE 1"


async def list_all_users(limit: int = 200) -> list[dict]:
    """List all users with profiles."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT telegram_id, username, first_name, is_premium,
                      premium_until, tokens, total_gens, created_at
               FROM users ORDER BY created_at DESC LIMIT $1""",
            limit,
        )
        return [dict(r) for r in rows]


# ============================================================
# Generations
# ============================================================
async def log_generation(
    telegram_id: int,
    prompt: str,
    mode: str = "generate",
    negative: str = "",
    character: str = "",
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    file_id: str = "",
    duration_sec: float | None = None,
    tokens_spent: int = 1,
) -> None:
    """Log a generation to history."""
    if not _pool:
        return

    async with _pool.acquire() as conn:
        # Get user id
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE telegram_id = $1", telegram_id
        )
        user_id = user["id"] if user else None

        await conn.execute(
            """INSERT INTO generations
               (user_id, telegram_id, prompt, negative, mode, character,
                seed, width, height, steps, cfg, file_id, duration_sec, tokens_spent)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
            user_id, telegram_id, prompt, negative, mode, character,
            seed, width, height, steps, cfg, file_id, duration_sec, tokens_spent,
        )

        # Update total_gens counter
        if user_id:
            await conn.execute(
                "UPDATE users SET total_gens = total_gens + 1 WHERE id = $1",
                user_id,
            )


async def get_history(telegram_id: int, limit: int = 10) -> list[dict]:
    """Get user's generation history."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM generations
               WHERE telegram_id = $1
               ORDER BY created_at DESC LIMIT $2""",
            telegram_id, limit,
        )
        return [dict(r) for r in rows]


async def delete_generation(telegram_id: int, gen_id: int) -> bool:
    """Delete a generation (only own items)."""
    if not _pool:
        return False
    async with _pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM generations WHERE id = $1 AND telegram_id = $2",
            gen_id, telegram_id,
        )
        return result == "DELETE 1"


# ============================================================
# Leaderboards
# ============================================================
async def get_top_users(limit: int = 10) -> list[dict]:
    """Top users by total generations."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT telegram_id, username, first_name, total_gens, is_premium
               FROM users ORDER BY total_gens DESC LIMIT $1""",
            limit,
        )
        return [dict(r) for r in rows]


async def get_top_characters(limit: int = 10) -> list[dict]:
    """Most used characters."""
    if not _pool:
        return []
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT character, COUNT(*) as count
               FROM generations WHERE character != ''
               GROUP BY character ORDER BY count DESC LIMIT $1""",
            limit,
        )
        return [dict(r) for r in rows]


async def get_stats() -> dict:
    """Global stats."""
    if not _pool:
        return {}
    async with _pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        total_gens = await conn.fetchval("SELECT COUNT(*) FROM generations")
        premium_users = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE is_premium = TRUE AND (premium_until IS NULL OR premium_until > NOW())"
        )
        return {
            "total_users": total_users,
            "total_generations": total_gens,
            "premium_users": premium_users,
        }
