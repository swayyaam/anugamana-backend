"""
Passive feedback logging to SQLite.

Every search response is logged. Judge scores are written asynchronously after
the response has already been returned to the user.

Fixed 2026-08-31 (audit E-07):
  * `with sqlite3.connect(...)` commits on exit but does NOT close the
    connection — every search leaked a file descriptor until the process died
    on "too many open files". Now wrapped in contextlib.closing.
  * Foreign keys were never enabled, so feedback rows could reference response
    ids that do not exist. PRAGMA foreign_keys is now on, and log_feedback
    validates the target row before inserting.
  * WAL mode, so the async judge writer and the request path stop blocking
    each other.
"""

import json
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import datetime, timezone

from app.config import FEEDBACK_DB

_lock = threading.Lock()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(FEEDBACK_DB), timeout=10.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        with closing(conn), conn:
            yield conn
    finally:
        pass


def init_db() -> None:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS responses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                query         TEXT    NOT NULL,
                hyde_query    TEXT,
                query_route   TEXT,
                verse_ids     TEXT,
                top_verse_id  TEXT,
                faith_score   REAL,
                grounded      REAL,
                relevant      REAL,
                restraint     REAL,
                latency_ms    INTEGER,
                degraded      TEXT,
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id  INTEGER NOT NULL REFERENCES responses(id),
                rating       INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                created_at   TEXT    NOT NULL,
                UNIQUE(response_id)
            );

            CREATE INDEX IF NOT EXISTS idx_responses_created ON responses(created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_response ON feedback(response_id);
        """)
        _migrate(conn)


def _migrate(conn) -> None:
    """
    Bring a database created by an earlier schema up to date.

    CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so constraints
    and columns added later never arrive without this.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(responses)")}
    for column, ddl in (
        ("query_route", "ALTER TABLE responses ADD COLUMN query_route TEXT"),
        ("grounded", "ALTER TABLE responses ADD COLUMN grounded REAL"),
        ("relevant", "ALTER TABLE responses ADD COLUMN relevant REAL"),
        ("restraint", "ALTER TABLE responses ADD COLUMN restraint REAL"),
        ("degraded", "ALTER TABLE responses ADD COLUMN degraded TEXT"),
    ):
        if column not in existing:
            conn.execute(ddl)

    # One vote per response. Databases created before this constraint existed
    # need the duplicates collapsed before the unique index can be built.
    conn.execute("""
        DELETE FROM feedback
        WHERE id NOT IN (SELECT MAX(id) FROM feedback GROUP BY response_id)
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_response "
        "ON feedback(response_id)"
    )


def log_response(
    query: str,
    hyde_query: str | None,
    verse_ids: list[str],
    top_verse_id: str | None,
    latency_ms: int,
    query_route: str | None = None,
    degraded: list[str] | None = None,
) -> int:
    with _lock, _conn() as conn:
        cur = conn.execute(
            """INSERT INTO responses
               (query, hyde_query, query_route, verse_ids, top_verse_id,
                latency_ms, degraded, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query,
                hyde_query,
                query_route,
                json.dumps(verse_ids),
                top_verse_id,
                latency_ms,
                json.dumps(degraded or []),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid


def update_judge_scores(response_id: int, scores: dict) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            """UPDATE responses
               SET faith_score = ?, grounded = ?, relevant = ?, restraint = ?
               WHERE id = ?""",
            (
                scores.get("score"),
                scores.get("grounded"),
                scores.get("relevant"),
                scores.get("restraint"),
                response_id,
            ),
        )


def response_exists(response_id: int) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM responses WHERE id = ?", (response_id,)
        ).fetchone()
        return row is not None


def log_feedback(response_id: int, rating: int) -> None:
    """rating: +1 (helpful) or -1 (not helpful). One vote per response."""
    if rating not in (1, -1):
        raise ValueError("rating must be +1 or -1")
    if not response_exists(response_id):
        raise ValueError(f"unknown response_id: {response_id}")
    with _lock, _conn() as conn:
        conn.execute(
            """INSERT INTO feedback (response_id, rating, created_at)
               VALUES (?, ?, ?)
               ON CONFLICT(response_id) DO UPDATE SET
                 rating = excluded.rating, created_at = excluded.created_at""",
            (response_id, rating, datetime.now(timezone.utc).isoformat()),
        )


def get_metrics(window_days: int = 7) -> dict:
    with _conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*)       AS total_queries,
                   AVG(latency_ms) AS avg_latency_ms,
                   AVG(faith_score) AS avg_faith_score,
                   AVG(grounded)   AS avg_grounded,
                   AVG(relevant)   AS avg_relevant,
                   AVG(restraint)  AS avg_restraint,
                   SUM(CASE WHEN faith_score IS NOT NULL THEN 1 ELSE 0 END) AS judged_count
            FROM responses
            WHERE created_at >= datetime('now', ?)
        """, (f"-{window_days} days",)).fetchone()

        fb = conn.execute("""
            SELECT SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END) AS thumbs_up,
                   SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS thumbs_down
            FROM feedback f
            JOIN responses r ON f.response_id = r.id
            WHERE r.created_at >= datetime('now', ?)
        """, (f"-{window_days} days",)).fetchone()

    def _round(value, digits=3):
        return round(value, digits) if value is not None else None

    return {
        "window_days":     window_days,
        "total_queries":   row["total_queries"] or 0,
        "avg_latency_ms":  _round(row["avg_latency_ms"], 1) or 0,
        "avg_faith_score": _round(row["avg_faith_score"]),
        "avg_grounded":    _round(row["avg_grounded"]),
        "avg_relevant":    _round(row["avg_relevant"]),
        "avg_restraint":   _round(row["avg_restraint"]),
        "judged_count":    row["judged_count"] or 0,
        "thumbs_up":       fb["thumbs_up"] or 0,
        "thumbs_down":     fb["thumbs_down"] or 0,
        "_note": "Judge scores are model-generated (silver). See docs/JUDGE_VALIDATION.md.",
    }
