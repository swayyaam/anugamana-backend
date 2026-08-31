"""
HTTP contract tests.

The previous integration suite targeted a Pinecone architecture the app no longer
imports — all sixteen of its tests failed. These cover the endpoints as shipped.
"""

import pytest


class TestSearchHappyPath:
    def test_returns_200(self, client):
        r = client.post("/search", json={"query": "I keep failing at work"})
        assert r.status_code == 200

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_returns_the_number_of_verses_requested(self, client, top_k):
        """Audit E-02: this was impossible — the last result was always dropped."""
        r = client.post("/search", json={"query": "I keep failing", "top_k": top_k})
        assert len(r.json()["results"]) == top_k

    def test_response_shape(self, client):
        data = client.post("/search", json={"query": "I feel lost"}).json()
        assert set(data) >= {"results", "query_meta"}
        verse = data["results"][0]
        assert set(verse) == {
            "verse_id", "chapter", "verse", "devanagari",
            "sanskrit", "translation", "score", "ai_guidance",
        }

    def test_meta_reports_score_type(self, client):
        meta = client.post("/search", json={"query": "I feel lost"}).json()["query_meta"]
        assert meta["score_type"] in {"cross_encoder", "rrf", "exact", "none"}
        assert meta["status"] == "ok"

    def test_scores_are_in_unit_interval(self, client):
        results = client.post("/search", json={"query": "I feel lost"}).json()["results"]
        assert all(0.0 <= v["score"] <= 1.0 for v in results)

    def test_guidance_is_attached(self, client):
        results = client.post("/search", json={"query": "I feel lost"}).json()["results"]
        assert all(v["ai_guidance"] for v in results)


class TestNonErrorOutcomes:
    """Crisis and off-topic are expected outcomes, not HTTP failures."""

    def test_crisis_returns_200_with_resources(self, client):
        r = client.post("/search", json={"query": "I want to kill myself"})
        assert r.status_code == 200
        data = r.json()
        assert data["query_meta"]["status"] == "crisis"
        assert data["results"] == []
        assert "14416" in data["message"]

    def test_crisis_is_not_logged_as_a_normal_response(self, client):
        client.post("/search", json={"query": "I want to end my life"})
        assert client.get("/metrics").json()["total_queries"] == 0

    def test_off_topic_returns_200(self, client, stub_llms):
        """
        Audit: this used to be 422, which FastAPI also uses for schema
        validation — the frontend could not tell the two apart.
        """
        stub_llms["guardrail_verdict"] = "off_topic"
        r = client.post("/search", json={"query": "how do I center a div"})
        assert r.status_code == 200
        assert r.json()["query_meta"]["status"] == "off_topic"

    def test_no_results_is_reported_explicitly(self, client, stub_retrieval):
        stub_retrieval["verses"] = []
        data = client.post("/search", json={"query": "zzzz"}).json()
        assert data["query_meta"]["status"] == "no_results"
        assert data["message"]


class TestValidation:
    @pytest.mark.parametrize("payload", [
        {"query": "x" * 501},
        {"query": ""},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 99},
        {},
    ])
    def test_bad_input_is_422(self, client, payload):
        assert client.post("/search", json=payload).status_code == 422


class TestFeedback:
    def test_accepts_a_vote_on_a_real_response(self, client):
        rid = client.post("/search", json={"query": "I feel lost"}).json()["query_meta"]["response_id"]
        assert client.post("/feedback", json={"response_id": rid, "rating": 1}).status_code == 200

    def test_rejects_a_vote_on_an_unknown_response(self, client):
        r = client.post("/feedback", json={"response_id": 999999, "rating": 1})
        assert r.status_code == 404

    def test_rejects_an_invalid_rating(self, client):
        rid = client.post("/search", json={"query": "I feel lost"}).json()["query_meta"]["response_id"]
        assert client.post("/feedback", json={"response_id": rid, "rating": 7}).status_code == 400

    def test_vote_is_reflected_in_metrics(self, client):
        rid = client.post("/search", json={"query": "I feel lost"}).json()["query_meta"]["response_id"]
        client.post("/feedback", json={"response_id": rid, "rating": 1})
        assert client.get("/metrics").json()["thumbs_up"] == 1


class TestMetrics:
    def test_shape(self, client):
        data = client.get("/metrics").json()
        assert set(data) >= {
            "window_days", "total_queries", "avg_latency_ms",
            "avg_faith_score", "judged_count", "thumbs_up", "thumbs_down",
        }

    def test_counts_searches(self, client):
        for _ in range(3):
            client.post("/search", json={"query": "I feel lost"})
        assert client.get("/metrics").json()["total_queries"] == 3

    def test_window_is_configurable(self, client):
        assert client.get("/metrics?days=30").json()["window_days"] == 30


class TestHealth:
    def test_root(self, client):
        assert client.get("/").json()["status"] == "online"
