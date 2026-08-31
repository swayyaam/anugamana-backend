"""
Feedback persistence — tests for audit E-07.

`with sqlite3.connect(...)` commits but does not close, so every search leaked a
file descriptor. Foreign keys were off, so feedback could reference responses
that never existed, and /metrics was therefore trivially poisonable.
"""

import sqlite3

import pytest

from app.services import feedback_logger


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_logger, "FEEDBACK_DB", tmp_path / "feedback.db")
    feedback_logger.init_db()
    return tmp_path / "feedback.db"


def a_response(**kwargs):
    payload = {
        "query": "I keep failing",
        "hyde_query": None,
        "verse_ids": ["2.47", "3.19"],
        "top_verse_id": "2.47",
        "latency_ms": 1200,
        "query_route": "semantic",
        "degraded": [],
    }
    payload.update(kwargs)
    return feedback_logger.log_response(**payload)


class TestConnectionLifecycle:
    def test_connections_are_closed(self, db):
        """
        The leak regression. sqlite3.Connection exposes no public 'closed' flag,
        so we assert the observable consequence: using the yielded connection
        after the context manager exits raises ProgrammingError.
        """
        captured = []
        with feedback_logger._conn() as conn:
            captured.append(conn)
        with pytest.raises(sqlite3.ProgrammingError):
            captured[0].execute("SELECT 1")

    def test_many_writes_do_not_exhaust_descriptors(self, db):
        for _ in range(250):
            a_response()
        assert feedback_logger.get_metrics()["total_queries"] == 250

    def test_wal_mode_enabled(self, db):
        with feedback_logger._conn() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


class TestForeignKeyIntegrity:
    def test_feedback_on_unknown_response_is_rejected(self, db):
        with pytest.raises(ValueError, match="unknown response_id"):
            feedback_logger.log_feedback(999_999, 1)

    def test_feedback_on_real_response_is_accepted(self, db):
        feedback_logger.log_feedback(a_response(), 1)
        assert feedback_logger.get_metrics()["thumbs_up"] == 1

    def test_rating_must_be_plus_or_minus_one(self, db):
        rid = a_response()
        for bad in (0, 2, -5, 100):
            with pytest.raises(ValueError, match="rating"):
                feedback_logger.log_feedback(rid, bad)

    def test_one_vote_per_response(self, db):
        rid = a_response()
        feedback_logger.log_feedback(rid, 1)
        feedback_logger.log_feedback(rid, -1)   # changes the vote, not a second one
        metrics = feedback_logger.get_metrics()
        assert metrics["thumbs_up"] == 0
        assert metrics["thumbs_down"] == 1


class TestMigration:
    def test_upgrades_a_pre_constraint_database(self, tmp_path, monkeypatch):
        """A database written by the old schema must migrate, not crash."""
        path = tmp_path / "old.db"
        legacy = sqlite3.connect(path)
        legacy.executescript("""
            CREATE TABLE responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL,
                hyde_query TEXT, verse_ids TEXT, top_verse_id TEXT,
                faith_score REAL, latency_ms INTEGER, created_at TEXT NOT NULL);
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT, response_id INTEGER NOT NULL,
                rating INTEGER NOT NULL, created_at TEXT NOT NULL);
            INSERT INTO responses (query, created_at) VALUES ('old', '2026-01-01');
            INSERT INTO feedback (response_id, rating, created_at)
              VALUES (1, 1, '2026-01-01'), (1, -1, '2026-01-02');
        """)
        legacy.commit()
        legacy.close()

        monkeypatch.setattr(feedback_logger, "FEEDBACK_DB", path)
        feedback_logger.init_db()   # must collapse the duplicate and add columns

        with feedback_logger._conn() as conn:
            assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(responses)")}
        assert {"query_route", "grounded", "relevant", "restraint", "degraded"} <= columns

        feedback_logger.log_feedback(1, 1)  # ON CONFLICT path now works


class TestJudgeScores:
    def test_scores_are_recorded_and_averaged(self, db):
        rid = a_response()
        feedback_logger.update_judge_scores(
            rid, {"score": 4.33, "grounded": 4, "relevant": 5, "restraint": 4}
        )
        metrics = feedback_logger.get_metrics()
        assert metrics["judged_count"] == 1
        assert metrics["avg_grounded"] == 4.0

    def test_unjudged_responses_report_none_not_zero(self, db):
        a_response()
        assert feedback_logger.get_metrics()["avg_faith_score"] is None

    def test_metrics_flag_scores_as_model_generated(self, db):
        assert "silver" in feedback_logger.get_metrics()["_note"].lower()
