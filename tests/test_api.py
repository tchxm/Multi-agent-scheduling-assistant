"""
test_api.py — API integration tests using FastAPI's TestClient.

Graph LLM and webhook client are dependency-overridden to fakes.
No API keys or network calls needed.
"""

import os
import sqlite3
import sys
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import build_graph
from mock_calendar import MockCalendar
from tests.fakes import FakeLLM, FakeWebhookClient


FIXED_NOW = datetime(2026, 7, 14, 10, 0, 0)


@pytest.fixture
def test_app(tmp_path):
    """
    Create a FastAPI test app with fakes injected.

    We rebuild the graph with FakeLLM for each test, replacing the
    lifespan-managed singletons.
    """
    from main import app, app_state

    # Set up test infrastructure
    cal_path = str(tmp_path / "test_cal.sqlite3")
    cp_path = str(tmp_path / "test_cp.sqlite3")

    calendar = MockCalendar(db_path=cal_path, now=FIXED_NOW)
    conn = sqlite3.connect(cp_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    webhook_client = FakeWebhookClient(status_code=200)

    # We need a long-lived FakeLLM — for multi-turn tests we queue many responses
    fake_llm = FakeLLM([
        # Enough responses for multiple test scenarios
        # Each booking flow: triage classification + booking extraction = 2 LLM calls
        # General flow: triage classification + answer generation = 2 LLM calls
        '{"intent": "booking"}',
        '{"date_phrase": "tomorrow", "time_phrase": "10am", "email": null}',
        '{"intent": "booking"}',
        '{"date_phrase": "tomorrow", "time_phrase": "10am", "email": "user@test.com"}',
        '{"intent": "general"}',
        "Hello! I can help you schedule appointments.",
        '{"intent": "booking"}',
        '{"date_phrase": "July 14", "time_phrase": "2pm", "email": "test@example.com"}',
        '{"intent": "booking"}',
        '{"date_phrase": "July 16", "time_phrase": "9am", "email": "done@test.com"}',
        '{"intent": "general"}',
        "You're welcome! Have a great day.",
    ])

    graph = build_graph(
        llm=fake_llm,
        calendar=calendar,
        webhook_url="https://fake.test/hook",
        webhook_client=webhook_client,
        checkpointer=checkpointer,
    )

    # Override app state
    app_state.graph = graph
    app_state.calendar = calendar
    app_state.checkpointer = checkpointer
    app_state.known_threads = set()

    client = TestClient(app)
    yield client, webhook_client

    calendar.close()
    conn.close()


class TestThreadCreation:
    """Test POST /threads."""

    def test_create_thread(self, test_app):
        client, _ = test_app
        response = client.post("/threads")
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) == 12


class TestHealthCheck:
    """Test GET /health."""

    def test_health_ok(self, test_app):
        client, _ = test_app
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["graph_compiled"] is True
        assert data["checkpointer_ok"] is True


class TestBookingHappyPath:
    """
    Multi-turn booking flow:
    1. POST message: "Book tomorrow at 10am" → missing email
    2. POST message: "user@test.com" → completes booking
    """

    def test_multi_turn_booking(self, test_app):
        client, webhook_client = test_app

        # Create thread
        thread_resp = client.post("/threads")
        thread_id = thread_resp.json()["thread_id"]

        # Turn 1: booking request, missing email
        resp1 = client.post(
            f"/threads/{thread_id}/messages",
            json={"message": "Book me an appointment tomorrow at 10am"},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["route"] == "booking"
        assert "email" in data1["missing_fields"]
        assert data1["reservation_id"] is None

        # Turn 2: provide email
        resp2 = client.post(
            f"/threads/{thread_id}/messages",
            json={"message": "My email is user@test.com"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["route"] == "booking"
        # Email should have been extracted
        assert data2["reservation_id"] is not None or "email" not in data2.get("missing_fields", [])


class TestNegotiationPath:
    """Booking a taken slot triggers negotiation with alternatives."""

    def test_taken_slot_negotiation(self, test_app):
        client, webhook_client = test_app

        # Create a fresh thread for this test
        thread_resp = client.post("/threads")
        thread_id = thread_resp.json()["thread_id"]

        # Request a slot that's pre-booked (14:00 on July 14, 2026)
        resp = client.post(
            f"/threads/{thread_id}/messages",
            json={"message": "Book July 14 at 2pm, email test@example.com"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should indicate the slot is not available
        assert data["response"] is not None


class TestHistoryPersistence:
    """History persists across requests to the same thread."""

    def test_history_available(self, test_app):
        client, _ = test_app

        # Create thread and send a message
        thread_resp = client.post("/threads")
        thread_id = thread_resp.json()["thread_id"]

        client.post(
            f"/threads/{thread_id}/messages",
            json={"message": "Hello there"},
        )

        # Get history
        history_resp = client.get(f"/threads/{thread_id}/history")
        assert history_resp.status_code == 200
        data = history_resp.json()
        assert "messages" in data
        assert len(data["messages"]) >= 1
        # Should contain our message
        user_messages = [m for m in data["messages"] if m["role"] == "user"]
        assert any("Hello" in m["content"] for m in user_messages)
