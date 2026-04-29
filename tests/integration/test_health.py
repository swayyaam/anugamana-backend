import pytest
from fastapi.testclient import TestClient


def test_health_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_response_shape(client):
    data = client.get("/").json()
    assert "message" in data
    assert "status" in data


def test_health_online_when_services_ready(client):
    data = client.get("/").json()
    assert data["status"] == "Online"


def test_health_maintenance_when_services_missing(client):
    from unittest.mock import patch
    with patch("app.state.embedder", None), patch("app.state.pc_index", None):
        data = client.get("/").json()
    assert "Maintenance" in data["status"]
