"""
Phase 7: Passive feedback logging to SQLite.

Every search response is logged to data/feedback.db.
Faithfulness scores are written async after the response is returned.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "feedback.db"

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS responses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query        TEXT    NOT NULL,
                hyde_query   TEXT,
                verse_ids    TEXT,
                top_verse_id TEXT,
                faith_score  REAL,
                latency_ms   INTEGER,
                created_at   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id  INTEGER NOT NULL REFERENCES responses(id),
                rating       INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                created_at   TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_response ON feedback(response_id);
        """)


def log_response(
    query: str,
    hyde_query: str | None,
    verse_ids: list[str],
    top_verse_id: str | None,
    latency_ms: int,
) -> int:
    """Insert a response record. Returns the row id."""
    with _lock, _conn() as conn:
        cur = conn.execute(
            """INSERT INTO responses
               (query, hyde_query, verse_ids, top_verse_id, latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                query,
                hyde_query,
                json.dumps(verse_ids),
                top_verse_id,
                latency_ms,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid


def update_faith_score(response_id: int, faith_score: float) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE responses SET faith_score = ? WHERE id = ?",
            (faith_score, response_id),
        )


def log_feedback(response_id: int, rating: int) -> None:
    """rating: +1 (helpful) or -1 (not helpful)."""
    if rating not in (1, -1):
        raise ValueError("rating must be +1 or -1")
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO feedback (response_id, rating, created_at) VALUES (?, ?, ?)",
            (response_id, rating, datetime.now(timezone.utc).isoformat()),
        )


def get_metrics(window_days: int = 7) -> dict:
    """Rolling metrics over the last N days."""
    with _conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                          AS total_queries,
                AVG(latency_ms)                   AS avg_latency_ms,
                AVG(faith_score)                  AS avg_faith_score,
                SUM(CASE WHEN faith_score IS NOT NULL THEN 1 ELSE 0 END) AS judged_count
            FROM responses
            WHERE created_at >= datetime('now', ?)
        """, (f"-{window_days} days",)).fetchone()

        feedback_row = conn.execute("""
            SELECT
                SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END) AS thumbs_up,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS thumbs_down
            FROM feedback f
            JOIN responses r ON f.response_id = r.id
            WHERE r.created_at >= datetime('now', ?)
        """, (f"-{window_days} days",)).fetchone()

    return {
        "window_days":    window_days,
        "total_queries":  row["total_queries"] or 0,
        "avg_latency_ms": round(row["avg_latency_ms"] or 0, 1),
        "avg_faith_score": round(row["avg_faith_score"] or 0, 3) if row["avg_faith_score"] else None,
        "judged_count":   row["judged_count"] or 0,
        "thumbs_up":      feedback_row["thumbs_up"] or 0,
        "thumbs_down":    feedback_row["thumbs_down"] or 0,
    }
