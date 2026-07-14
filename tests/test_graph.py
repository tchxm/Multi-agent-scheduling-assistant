"""
test_graph.py — Integration tests for the LangGraph scheduling state machine.

Uses FakeLLM with canned responses to simulate each conversation path.
Uses a real MockCalendar (temp DB) and FakeWebhookClient.
No API keys or network calls needed.
"""

import sqlite3
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph, SchedulingState
from mock_calendar import MockCalendar
from tests.fakes import FakeLLM, FakeWebhookClient


# Fixed reference: Tuesday July 14, 2026
FIXED_NOW = datetime(2026, 7, 14, 10, 0, 0)


@pytest.fixture
def calendar(tmp_path):
    """Fresh calendar with known pre-seeded slots."""
    db_path = str(tmp_path / "test_cal.sqlite3")
    cal = MockCalendar(db_path=db_path, now=FIXED_NOW)
    yield cal
    cal.close()


@pytest.fixture
def checkpointer(tmp_path):
    """Fresh SqliteSaver checkpointer using direct connection."""
    db_path = str(tmp_path / "test_checkpoints.sqlite3")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    yield saver
    conn.close()


@pytest.fixture
def webhook_client():
    return FakeWebhookClient(status_code=200)


def _make_input(message_text):
    """Helper to create a full input state dict for graph.invoke."""
    return {
        "messages": [HumanMessage(content=message_text)],
        "route": None,
        "raw_date_phrase": None,
        "raw_time_phrase": None,
        "resolved_date": None,
        "resolved_time": None,
        "email": None,
        "missing_fields": [],
        "availability_checked": False,
        "slot_available": None,
        "proposed_alternatives": [],
        "reservation_id": None,
        "notification_sent": False,
        "final_response": None,
        "reference_time": None,
    }


def _make_graph(llm, calendar, webhook_client, checkpointer=None):
    """Helper to build a compiled graph with injected fakes."""
    return build_graph(
        llm=llm,
        calendar=calendar,
        webhook_url="https://fake-webhook.test/hook",
        webhook_client=webhook_client,
        checkpointer=checkpointer,
    )


class TestGeneralQuestionRouting:
    """(a) General question routes straight to a direct answer."""

    def test_general_question(self, calendar, webhook_client):
        llm = FakeLLM([
            # triage classification
            '{"intent": "general"}',
            # triage answer generation
            "Hello! I'm your scheduling assistant. I can help you book appointments. What would you like to do?",
        ])

        graph = _make_graph(llm, calendar, webhook_client)
        result = graph.invoke(_make_input("Hello, who are you?"))

        assert result["route"] == "general"
        assert result["final_response"] is not None
        assert "scheduling" in result["final_response"].lower() or "hello" in result["final_response"].lower()
        assert result.get("reservation_id") is None


class TestBookingMissingEmail:
    """(b) Booking request missing email results in asking for it."""

    def test_missing_email(self, calendar, webhook_client):
        llm = FakeLLM([
            # triage classification
            '{"intent": "booking"}',
            # booking specialist extraction (date+time found, no email)
            '{"date_phrase": "tomorrow", "time_phrase": "10am", "email": null}',
        ])

        graph = _make_graph(llm, calendar, webhook_client)
        result = graph.invoke(_make_input("Book me an appointment tomorrow at 10am"))

        assert result["route"] == "booking"
        assert "email" in result["missing_fields"]
        assert result["final_response"] is not None
        assert "email" in result["final_response"].lower()
        assert result.get("reservation_id") is None


class TestFullBookingHappyPath:
    """(c) Full booking request with available slot completes through to reservation."""

    def test_happy_path(self, calendar, webhook_client):
        # 09:00 on 2026-07-16 (Thu) should be available (3rd business day, no pre-bookings)
        llm = FakeLLM([
            # triage classification
            '{"intent": "booking"}',
            # booking specialist extraction
            '{"date_phrase": "July 16", "time_phrase": "9am", "email": "test@example.com"}',
        ])

        graph = _make_graph(llm, calendar, webhook_client)
        result = graph.invoke(_make_input("Book July 16 at 9am, email test@example.com"))

        assert result["route"] == "booking"
        assert result["reservation_id"] is not None
        assert len(result["reservation_id"]) == 8
        assert result["notification_sent"] is True
        assert result["missing_fields"] == []
        assert result["final_response"] is not None
        # Webhook should have been called
        assert len(webhook_client.posts) == 1


class TestTakenSlotNegotiation:
    """(d) Booking a taken slot triggers alternatives, no reservation made."""

    def test_taken_slot_suggests_alternatives(self, calendar, webhook_client):
        # 14:00 on 2026-07-14 is pre-booked
        llm = FakeLLM([
            # triage classification
            '{"intent": "booking"}',
            # booking specialist extraction
            '{"date_phrase": "July 14", "time_phrase": "2pm", "email": "test@example.com"}',
        ])

        graph = _make_graph(llm, calendar, webhook_client)
        input_state = _make_input("Book July 14 at 2pm, email test@example.com")
        # Pin reference_time before July 14 so "July 14" resolves to 2026-07-14
        input_state["reference_time"] = "2026-07-13T10:00:00"
        result = graph.invoke(input_state)

        assert result["route"] == "booking"
        assert result["slot_available"] is False
        assert len(result["proposed_alternatives"]) > 0
        assert result.get("reservation_id") is None
        assert result["final_response"] is not None
        assert "alternative" in result["final_response"].lower() or "booked" in result["final_response"].lower()
        # Webhook should NOT have been called
        assert len(webhook_client.posts) == 0


class TestStatePersistence:
    """(e) State persists across two separate graph.invoke() calls with the same thread_id."""

    def test_state_persists_across_invocations(self, calendar, webhook_client, checkpointer):
        llm = FakeLLM([
            # First invoke: triage → booking, extract date+time but no email
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "10am", "email": null}',
            # Second invoke: triage → booking, now extract email
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "10am", "email": "user@test.com"}',
        ])

        graph = _make_graph(llm, calendar, webhook_client, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-persist-001"}}

        # First call: missing email
        result1 = graph.invoke(_make_input("Book tomorrow at 10am"), config=config)

        assert "email" in result1["missing_fields"]

        # Build second input carrying forward messages
        input2 = _make_input("My email is user@test.com")
        input2["messages"] = result1["messages"] + [HumanMessage(content="My email is user@test.com")]

        # Second call: provide email — should see prior fields in state
        result2 = graph.invoke(input2, config=config)

        # The second call should have picked up the date and time from
        # the conversation context (LLM extracts from full conversation)
        # and the email from this message
        assert result2.get("email") == "user@test.com"


class TestOutsideBusinessHours:
    """Booking a slot outside of valid business hours/days gives a policy warning."""

    def test_outside_hours_warning(self, calendar, webhook_client):
        # 19:00 (7 PM) is outside 9 AM - 5 PM business hours
        llm = FakeLLM([
            # triage classification
            '{"intent": "booking"}',
            # booking specialist extraction
            '{"date_phrase": "tomorrow", "time_phrase": "7pm", "email": "test@example.com"}',
        ])

        graph = _make_graph(llm, calendar, webhook_client)
        result = graph.invoke(_make_input("Book tomorrow at 7pm, email test@example.com"))

        assert result["route"] == "booking"
        assert result["slot_available"] is False
        assert result.get("reservation_id") is None
        assert "9:00 AM and 5:00 PM" in result["final_response"]
        assert "Monday through Friday" in result["final_response"]
        # It should clear the resolved details so the next turn starts fresh
        assert result["resolved_date"] is None
        assert result["resolved_time"] is None
        # Webhook should not be called
        assert len(webhook_client.posts) == 0


class TestMultiTurnDivergenceAndStatePersistence:
    """Integration test asserting the exact 6-message sequence from user report."""

    def test_six_message_sequence(self, calendar, webhook_client, checkpointer):
        # We need a long queue of FakeLLM responses matching the sequence
        llm = FakeLLM([
            # 1. "Book next Monday at 10am"
            '{"intent": "booking"}',
            '{"date_phrase": "next Monday", "time_phrase": "10am", "email": null}',
            
            # 2. "Book tomorrow at 4pm"
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "4pm", "email": null}',
            
            # 3. "Book tomorrow at 4PM"
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "4pm", "email": null}',
            
            # 4. "What are your business hours?" (routed as general, bypasses booking extraction)
            '{"intent": "general"}',
            "We are open Monday through Friday, 9:00 AM to 5:00 PM.",
            
            # 5. "Book Friday at 11am"
            '{"intent": "booking"}',
            '{"date_phrase": "Friday", "time_phrase": "11am", "email": null}',
            
            # 6. "test@example.com" (triage is general, overrules to booking continuation)
            '{"intent": "general"}',
            '{"date_phrase": null, "time_phrase": null, "email": "test@example.com"}',
        ])

        graph = _make_graph(llm, calendar, webhook_client, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-six-message-thread"}}

        # Helper to invoke the graph and carry forward the messages channel
        state = _make_input("")
        state["reference_time"] = "2026-07-13T10:00:00"
        
        def run_message(msg_text):
            nonlocal state
            state["messages"].append(HumanMessage(content=msg_text))
            state = graph.invoke(state, config=config)
            # Add assistant response to history
            state["messages"].append(AIMessage(content=state["final_response"]))

        # Step 1: "Book next Monday at 10am"
        run_message("Book next Monday at 10am")
        # Reference is Monday July 13, 2026. Next Monday is July 20.
        assert state["resolved_date"] == "2026-07-20"
        assert state["resolved_time"] == "10:00"
        assert "email" in state["missing_fields"]

        # Step 2: "Book tomorrow at 4pm"
        run_message("Book tomorrow at 4pm")
        # Tomorrow is Tuesday July 14.
        assert state["resolved_date"] == "2026-07-14"
        assert state["resolved_time"] == "16:00"

        # Step 3: "Book tomorrow at 4PM"
        run_message("Book tomorrow at 4PM")
        assert state["resolved_date"] == "2026-07-14"
        assert state["resolved_time"] == "16:00"

        # Step 4: "What are your business hours?"
        run_message("What are your business hours?")
        # Should be routed as general, state date/time should stay unchanged
        assert state["route"] == "general"
        assert "9:00 AM to 5:00 PM" in state["final_response"]

        # Step 5: "Book Friday at 11am"
        run_message("Book Friday at 11am")
        # Should reset and resolve to Friday, July 17
        assert state["route"] == "booking"
        assert state["resolved_date"] == "2026-07-17"
        assert state["resolved_time"] == "11:00"
        assert "email" in state["missing_fields"]

        # Step 6: "test@example.com"
        run_message("test@example.com")
        # Check availability should have run against July 17, 11:00
        # (Friday at 11am should be available since only 14:00/15:00 are pre-seeded taken on business days 1&2)
        assert state["reservation_id"] is not None
        assert state["notification_sent"] is True
        assert "✅" in state["final_response"]
        assert "2026-07-17" in state["final_response"]
        assert "11:00" in state["final_response"]


class TestUnsupportedCancelRequest:
    """Cancel/reschedule requests must never produce a fabricated success message."""

    def test_cancel_returns_unsupported_message(self, calendar, webhook_client):
        """A standalone cancel request gets a deterministic refusal, no LLM involved."""
        llm = FakeLLM([
            # The unsupported-action check fires BEFORE the classifier,
            # so the classifier is never called. No LLM responses are consumed.
        ])

        graph = _make_graph(llm, calendar, webhook_client)
        result = graph.invoke(_make_input("Cancel my appointment"))

        assert result["route"] == "general"
        assert "not" in result["final_response"].lower() or "aren't" in result["final_response"].lower()
        assert "support" in result["final_response"].lower()
        assert result.get("reservation_id") is None
        # Verify the booking flow was never entered
        assert result.get("resolved_date") is None
        assert result.get("resolved_time") is None

    def test_reschedule_returns_unsupported_message(self, calendar, webhook_client):
        """Reschedule requests are also intercepted."""
        llm = FakeLLM([])

        graph = _make_graph(llm, calendar, webhook_client)
        result = graph.invoke(_make_input("I need to reschedule my appointment"))

        assert result["route"] == "general"
        assert "support" in result["final_response"].lower()
        assert result.get("reservation_id") is None

    def test_cancel_mid_booking_still_intercepted(self, calendar, webhook_client):
        """Even if a booking is in progress, a cancel request is intercepted."""
        llm = FakeLLM([
            # Turn 1: normal booking flow
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "3pm", "email": null}',
            # Turn 2: cancel request — unsupported check fires before classifier
            # so no more LLM responses are consumed
        ])

        graph = _make_graph(llm, calendar, webhook_client)

        # Turn 1: start a booking
        result1 = graph.invoke(_make_input("Book tomorrow at 3pm"))
        assert result1["route"] == "booking"
        assert "email" in result1["missing_fields"]

        # Turn 2: cancel mid-flow
        input2 = _make_input("Cancel the appointment")
        input2["messages"] = result1["messages"] + [HumanMessage(content="Cancel the appointment")]
        input2["resolved_date"] = result1["resolved_date"]
        input2["resolved_time"] = result1["resolved_time"]
        result2 = graph.invoke(input2)

        assert result2["route"] == "general"
        assert "support" in result2["final_response"].lower()
        # Must NOT have a fabricated reservation_id
        assert result2.get("reservation_id") is None

    def test_respond_node_refuses_without_reservation_id(self, calendar, webhook_client):
        """respond_node must refuse to confirm if reservation_id is None."""
        from graph import respond_node
        fake_state = {
            "resolved_date": "2026-07-14",
            "resolved_time": "10:00",
            "email": "test@example.com",
            "reservation_id": None,
            "notification_sent": False,
        }
        result = respond_node(fake_state)
        assert "went wrong" in result["final_response"].lower() or "no reservation" in result["final_response"].lower()
        assert "✅" not in result["final_response"]


class TestDraftResetAndStateClearing:
    """Verifies active booking draft reset when user cancels/resets mid-session."""

    def test_cancel_draft_clears_state(self, calendar, webhook_client):
        # Turn 1: normal booking flow starts
        llm = FakeLLM([
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "3pm", "email": null}',
        ])
        graph = _make_graph(llm, calendar, webhook_client)

        result1 = graph.invoke(_make_input("Book tomorrow at 3pm"))
        assert result1["route"] == "booking"
        assert result1["resolved_date"] is not None
        assert result1["resolved_time"] is not None

        # Turn 2: User says "cancel" to reset the booking draft
        # Deterministic _is_reset_request fires, no LLM call needed
        input2 = _make_input("cancel")
        input2["messages"] = result1["messages"] + [HumanMessage(content="cancel")]
        input2["resolved_date"] = result1["resolved_date"]
        input2["resolved_time"] = result1["resolved_time"]
        result2 = graph.invoke(input2)

        assert result2["route"] == "general"
        assert "cleared" in result2["final_response"].lower()
        # Draft values must be completely reset/cleared
        assert result2["resolved_date"] is None
        assert result2["resolved_time"] is None
        assert result2["raw_date_phrase"] is None
        assert result2["raw_time_phrase"] is None
        assert result2["missing_fields"] == []

    def test_unsupported_action_clears_state(self, calendar, webhook_client):
        # Turn 1: normal booking flow starts
        llm = FakeLLM([
            '{"intent": "booking"}',
            '{"date_phrase": "tomorrow", "time_phrase": "3pm", "email": null}',
        ])
        graph = _make_graph(llm, calendar, webhook_client)

        result1 = graph.invoke(_make_input("Book tomorrow at 3pm"))
        assert result1["resolved_date"] is not None

        # Turn 2: User requests unsupported action "cancel appointment at wednesday 9pm"
        # Matches _is_unsupported_action, clears state
        input2 = _make_input("cancel appointment at wednesday 9pm")
        input2["messages"] = result1["messages"] + [HumanMessage(content="cancel appointment at wednesday 9pm")]
        input2["resolved_date"] = result1["resolved_date"]
        input2["resolved_time"] = result1["resolved_time"]
        result2 = graph.invoke(input2)

        assert result2["route"] == "general"
        assert "support" in result2["final_response"].lower()
        # Stale draft state must be completely cleared
        assert result2["resolved_date"] is None
        assert result2["resolved_time"] is None


