"""
graph.py — LangGraph state machine for the Multi-Agent Scheduling Assistant.

Implements a real multi-node graph (not a single mega-prompt):
  triage → booking_specialist → check_availability → reserve → notify → respond

Each node has a narrow, inspectable responsibility. Conditional routing
between nodes is based on state fields (route, missing_fields, slot_available).
Persistence via LangGraph's SqliteSaver checkpointer, keyed by thread_id.
"""

import json
import os
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from date_resolver import resolve_date_phrase, resolve_time_phrase
from mock_calendar import MockCalendar
from notifications import send_booking_notification


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class SchedulingState(TypedDict):
    messages: list              # LangChain message objects (conversation history)
    route: str | None           # "general" | "booking", set by triage
    raw_date_phrase: str | None
    raw_time_phrase: str | None
    resolved_date: str | None   # YYYY-MM-DD from deterministic parser
    resolved_time: str | None   # HH:MM from deterministic parser
    email: str | None
    missing_fields: list        # which of [date, time, email] still needed
    availability_checked: bool
    slot_available: bool | None
    proposed_alternatives: list  # alternate slot suggestions when taken
    reservation_id: str | None
    notification_sent: bool
    final_response: str | None  # what gets shown to the user this turn
    reference_time: str | None  # Base date/time for tests (ISO string)


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------

def get_llm(provider: str | None = None):
    """
    Return a ChatModel based on the LLM_PROVIDER env var.
    Default: Groq (llama-3.3-70b-versatile).
    """
    provider = provider or os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        from langchain_groq import ChatGroq
        # Inject placeholder key if missing to avoid startup validation crash
        api_key = os.getenv("GROQ_API_KEY") or "gsk_placeholder_key_not_set"
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or "sk-ant-placeholder-key-not-set"
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0,
            api_key=api_key,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY") or "sk-placeholder-key-not-set"
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _is_plausible_continuation(message_text: str, missing_fields: list) -> bool:
    """
    Check if the user message is a plausible continuation reply
    answering a missing booking field rather than a new/unrelated question.
    """
    text = message_text.strip().lower()

    # 1. Contains an email
    if "@" in text:
        return True

    # 2. Simple confirmation/agreement words
    confirmations = {"yes", "no", "y", "n", "sure", "ok", "okay", "that works", "perfect", "please"}
    if text in confirmations:
        return True

    # 3. Selection of alternative slot (e.g. "1", "first", "second", "third", "number 2")
    if text.isdigit() and int(text) in (1, 2, 3, 4, 5):
        return True
    if any(word in text for word in ["first", "second", "third", "last", "number"]):
        return True

    # 4. If we are asking for email and the message is short (typical email reply)
    if "email" in missing_fields and len(text.split()) <= 2:
        return True

    # 5. If it's a brief time/date indicator (e.g. "4pm", "tomorrow", "Friday") under 4 words
    if len(text.split()) <= 4:
        indicators = ["am", "pm", ":", "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if any(ind in text for ind in indicators) or any(char.isdigit() for char in text):
            return True

    return False


def _is_unsupported_action(message_text: str) -> bool:
    """
    Detect cancel/reschedule/modify requests that this system does not support.
    Returns True if the message is an unsupported action request.
    """
    text = message_text.strip().lower()
    unsupported_keywords = ["cancel", "reschedule", "modify", "change my appointment",
                            "delete my booking", "remove my appointment", "undo my booking"]
    return any(kw in text for kw in unsupported_keywords)


def _is_reset_request(message_text: str) -> bool:
    """
    Check if the user wants to reset or clear the current booking draft.
    """
    text = message_text.strip().lower()
    reset_keywords = ["cancel", "reset", "clear", "start over", "cancel this", "cancel request", "abort"]
    return text in reset_keywords


def _is_unsupported_action(message_text: str) -> bool:
    """
    Detect cancel/reschedule/modify requests of existing appointments that this system does not support.
    Use specific phrases to avoid false positives on words like 'cancel' in multi-action messages.
    """
    text = message_text.strip().lower()
    unsupported_keywords = [
        "cancel my appointment", "cancel the appointment", "cancel my booking", 
        "delete my booking", "remove my appointment", "undo my booking",
        "reschedule my appointment", "change my booking", "cancel existing booking",
        "cancel appointment at"
    ]
    return any(kw in text for kw in unsupported_keywords)


def triage_node(state: SchedulingState, llm=None) -> dict:
    """
    Classify user intent: general question, booking request, or unsupported actions.
    If general, answer directly. If booking, route to booking specialist.
    Unsupported/cancel requests return a friendly refusal and clear out stale draft state.
    """
    updates = {}
    
    # If the previous reservation was completed, start fresh for this turn
    if state.get("reservation_id"):
        updates.update({
            "raw_date_phrase": None,
            "raw_time_phrase": None,
            "resolved_date": None,
            "resolved_time": None,
            "reservation_id": None,
            "slot_available": None,
            "availability_checked": False,
            "proposed_alternatives": [],
            "notification_sent": False,
            "final_response": None,
        })
        # Merge updates so the rest of the node's local logic operates on the cleaned state
        state = {**state, **updates}

    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        updates.update({"route": "general", "final_response": "Hello! How can I help you today?"})
        return updates

    # 1. Check if the user is asking to reset/cancel the active draft
    if _is_reset_request(last_message.content):
        updates.update({
            "route": "general",
            "raw_date_phrase": None,
            "raw_time_phrase": None,
            "resolved_date": None,
            "resolved_time": None,
            "slot_available": None,
            "availability_checked": False,
            "proposed_alternatives": [],
            "missing_fields": [],
            "final_response": (
                "I've cleared your current scheduling request. Let me know when you'd like to "
                "schedule a new appointment!"
            ),
        })
        return updates

    # 2. Check deterministic unsupported cancellation/modification requests
    if _is_unsupported_action(last_message.content):
        updates.update({
            "route": "general",
            "raw_date_phrase": None,
            "raw_time_phrase": None,
            "resolved_date": None,
            "resolved_time": None,
            "slot_available": None,
            "availability_checked": False,
            "proposed_alternatives": [],
            "missing_fields": [],
            "final_response": (
                "I'm sorry, but cancellations and rescheduling of existing bookings aren't "
                "currently supported through this assistant. Please contact support directly "
                "to cancel or modify an existing booking.\n\n"
                "I can help you **schedule a new appointment** if you'd like!"
            ),
        })
        return updates

    # 3. Call the classifier for all other requests
    classification_prompt = SystemMessage(content="""You are a triage classifier. Analyze the user's message and classify their intent into one of:

1. {"intent": "booking"} — if the user wants to book a new appointment, or is changing/specifying the date/time/email for the appointment they are currently trying to book (e.g. "let's do Friday instead", "change time to 3pm", "reschedule to tomorrow at 4pm").
2. {"intent": "unsupported"} — if the user wants to cancel, delete, or reschedule an existing completed reservation without proposing a new date/time in the same message (e.g. "I want to cancel my appointment", "cancel the reservation", "how do I reschedule?").
3. {"intent": "general"} — for greetings, general questions, chitchat, etc. (e.g. "what are your hours?", "hi").

Respond with EXACTLY one JSON object, nothing else.""")

    response = llm.invoke([classification_prompt, last_message])
    content = response.content.strip()

    try:
        result = json.loads(content)
        intent = result.get("intent", "general")
    except (json.JSONDecodeError, AttributeError):
        # Fallback: check for keywords
        content_lower = content.lower()
        if "booking" in content_lower:
            intent = "booking"
        else:
            intent = "general"

    # Clear active session on unsupported intent
    if intent == "unsupported":
        updates.update({
            "route": "general",
            "raw_date_phrase": None,
            "raw_time_phrase": None,
            "resolved_date": None,
            "resolved_time": None,
            "slot_available": None,
            "availability_checked": False,
            "proposed_alternatives": [],
            "missing_fields": [],
            "final_response": (
                "I'm sorry, but cancellations and rescheduling of existing bookings aren't "
                "currently supported through this assistant. Please contact support directly "
                "to cancel or modify an existing booking.\n\n"
                "I can help you **schedule a new appointment** if you'd like!"
            ),
        })
        return updates

    # Continuation check: If we have an active, incomplete booking session,
    # keep routing to booking ONLY if the message is a plausible continuation reply.
    if intent == "general" and (state.get("resolved_date") or state.get("resolved_time") or state.get("email")):
        if not state.get("reservation_id"):
            missing = state.get("missing_fields", [])
            if _is_plausible_continuation(last_message.content, missing):
                intent = "booking"

    if intent == "general":
        # Generate a direct answer
        answer_prompt = SystemMessage(content="""You are a friendly scheduling assistant. The user asked a general question (not about booking). Give a helpful, concise answer. If they greet you, greet them back and let them know you can help schedule appointments. IMPORTANT: Never claim that an action (booking, cancellation, modification) was completed unless you are explicitly told it was. You can only help schedule NEW appointments.""")
        answer = llm.invoke([answer_prompt] + messages)
        updates.update({
            "route": "general",
            "final_response": answer.content,
        })
        return updates
    else:
        updates.update({"route": "booking"})
        return updates


def booking_specialist_node(state: SchedulingState, llm=None) -> dict:
    """
    Extract booking details (date phrase, time phrase, email) from conversation.
    Then resolve date/time deterministically. If anything's missing, ask the user.
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # Focus extraction strictly on the latest user message to avoid mixing context with history
    extraction_prompt = SystemMessage(content="""You are a booking detail extractor. Analyze the user's latest message to find:
1. date_phrase: any mention of a date (e.g., "tomorrow", "next Monday", "July 20th", "in 3 days"). Extract the raw phrase exactly as the user said it.
2. time_phrase: any mention of a time (e.g., "3pm", "at 15:00", "in the morning at 10"). Extract the raw phrase.
3. email: any email address mentioned.

Respond with EXACTLY one JSON object like:
{"date_phrase": "tomorrow", "time_phrase": "3pm", "email": "user@example.com"}

Use null for any field you cannot find in the message. Only output the JSON, nothing else.""")

    response = llm.invoke([extraction_prompt, last_message])
    content = response.content.strip()

    # Clean up potential markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    try:
        extracted = json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        extracted = {}

    # If the user provided BOTH a date and a time in this message,
    # it's a fresh booking request. Reset the in-progress slot details.
    if extracted.get("date_phrase") and extracted.get("time_phrase"):
        state.update({
            "raw_date_phrase": None,
            "raw_time_phrase": None,
            "resolved_date": None,
            "resolved_time": None,
            "slot_available": None,
            "availability_checked": False,
            "proposed_alternatives": [],
        })

    # Merge new fields with existing state
    raw_date = extracted.get("date_phrase") or state.get("raw_date_phrase")
    raw_time = extracted.get("time_phrase") or state.get("raw_time_phrase")
    email = extracted.get("email") or state.get("email")

    # Deterministic resolution — NO LLM date math
    now = state.get("reference_time")
    if not now:
        now = datetime.now()
    elif isinstance(now, str):
        try:
            now = datetime.fromisoformat(now)
        except ValueError:
            now = datetime.now()
    resolved_date = state.get("resolved_date")
    resolved_time = state.get("resolved_time")

    if raw_date:
        new_resolved = resolve_date_phrase(raw_date, now)
        if new_resolved:
            resolved_date = new_resolved

    if raw_time:
        new_resolved = resolve_time_phrase(raw_time)
        if new_resolved:
            resolved_time = new_resolved

    # Determine what's still missing
    missing = []
    if not resolved_date:
        missing.append("date")
    if not resolved_time:
        missing.append("time")
    if not email:
        missing.append("email")

    update = {
        "raw_date_phrase": raw_date,
        "raw_time_phrase": raw_time,
        "resolved_date": resolved_date,
        "resolved_time": resolved_time,
        "email": email,
        "missing_fields": missing,
    }

    if missing:
        # Build a natural question asking for what's missing
        missing_descriptions = {
            "date": "what date you'd like",
            "time": "what time you'd prefer",
            "email": "your email address for the confirmation",
        }
        missing_parts = [missing_descriptions[f] for f in missing]

        if len(missing_parts) == 1:
            question = f"I'd love to help you book that! Could you tell me {missing_parts[0]}?"
        elif len(missing_parts) == 2:
            question = f"I'd love to help you book that! Could you tell me {missing_parts[0]} and {missing_parts[1]}?"
        else:
            question = f"I'd love to help you book an appointment! I'll need {missing_parts[0]}, {missing_parts[1]}, and {missing_parts[2]}."

        # Add context about what we already have
        known = []
        if resolved_date:
            known.append(f"date: {resolved_date}")
        if resolved_time:
            known.append(f"time: {resolved_time}")
        if email:
            known.append(f"email: {email}")

        if known:
            question += f"\n\n(I already have: {', '.join(known)})"

        update["final_response"] = question

    return update


def check_availability_node(state: SchedulingState, calendar: MockCalendar = None) -> dict:
    """
    Check if the requested slot is available. If not, suggest alternatives.
    """
    date = state["resolved_date"]
    time = state["resolved_time"]
    print(f"[DEBUG check_availability_node] resolved_date={date!r}, resolved_time={time!r}")
    is_valid = calendar.is_valid_slot(date, time)
    print(f"[DEBUG check_availability_node] is_valid_slot={is_valid}")

    # Check if requested slot is within valid business hours and business days
    if not is_valid:
        message = (
            f"Please note that we only book appointments between 9:00 AM and 5:00 PM "
            f"on the hour, Monday through Friday.\n\n"
            f"Could you suggest a different date or time?"
        )
        return {
            "availability_checked": True,
            "slot_available": False,
            "proposed_alternatives": [],
            "final_response": message,
            # Clear resolved fields so the next user reply gets re-extracted
            "resolved_time": None,
            "resolved_date": None,
            "raw_time_phrase": None,
            "raw_date_phrase": None,
        }

    available = calendar.is_slot_available(date, time)

    if available:
        return {
            "availability_checked": True,
            "slot_available": True,
        }
    else:
        # Slot is taken — suggest alternatives (negotiation)
        alternatives = calendar.suggest_alternatives(date, count=3)

        if alternatives:
            alt_formatted = "\n".join(f"  • {alt}" for alt in alternatives)
            message = (
                f"Unfortunately, {date} at {time} is already booked. "
                f"Here are some available alternatives:\n{alt_formatted}\n\n"
                f"Would any of these work for you?"
            )
        else:
            message = (
                f"Unfortunately, {date} at {time} is already booked and "
                f"I couldn't find any nearby alternatives. "
                f"Could you suggest a different date?"
            )

        return {
            "availability_checked": True,
            "slot_available": False,
            "proposed_alternatives": alternatives,
            "final_response": message,
            # Clear resolved fields so the next user reply gets re-extracted
            "resolved_time": None,
            "resolved_date": None,
            "raw_time_phrase": None,
            "raw_date_phrase": None,
        }


def reserve_node(state: SchedulingState, calendar: MockCalendar = None) -> dict:
    """
    Reserve the slot. On race-condition failure, signal to re-check availability.
    """
    date = state["resolved_date"]
    time = state["resolved_time"]
    email = state["email"]

    reservation_id = calendar.reserve_slot(date, time, email)

    if reservation_id:
        return {"reservation_id": reservation_id}
    else:
        # Race condition — someone else took it between check and reserve
        return {
            "reservation_id": None,
            "slot_available": False,
            "availability_checked": False,
        }


def notify_node(
    state: SchedulingState,
    webhook_url: str | None = None,
    webhook_client=None,
) -> dict:
    """
    Send booking confirmation via webhook. Never blocks the booking on failure.
    """
    url = webhook_url or os.getenv("WEBHOOK_URL", "")
    print(f"[DEBUG notify_node] Resolving Webhook URL: {url!r}")
    if not url:
        print("[DEBUG notify_node] Webhook URL is empty or None! Skipping notification.")
        return {"notification_sent": False}

    success = send_booking_notification(
        webhook_url=url,
        email=state["email"],
        details={
            "date": state["resolved_date"],
            "time": state["resolved_time"],
            "reservation_id": state["reservation_id"],
        },
        client=webhook_client,
    )
    print(f"[DEBUG notify_node] send_booking_notification result: {success}")
    return {"notification_sent": success}


def respond_node(state: SchedulingState) -> dict:
    """
    Build the final confirmation message after a successful booking.

    SAFETY GUARD: This node structurally refuses to emit a success message
    unless reservation_id is set — meaning reserve_slot() actually returned
    a valid ID. This makes it impossible for the LLM to hallucinate a
    "confirmed" message without a grounding tool result.
    """
    reservation_id = state.get("reservation_id")

    # Hard guard: if no real reservation was made, refuse to confirm anything.
    if not reservation_id:
        return {
            "final_response": (
                "⚠️ Something went wrong — no reservation was actually created. "
                "Please try booking again."
            )
        }

    date = state["resolved_date"]
    time = state["resolved_time"]
    email = state["email"]
    notification_sent = state.get("notification_sent", False)

    if notification_sent:
        message = (
            f"✅ You're all set! Your appointment is booked for "
            f"**{date}** at **{time}**.\n\n"
            f"📧 A confirmation has been sent to **{email}**.\n"
            f"📋 Your reservation ID is: **{reservation_id}**"
        )
    else:
        message = (
            f"✅ Your appointment is booked for "
            f"**{date}** at **{time}**.\n\n"
            f"⚠️ The confirmation notification didn't go through, "
            f"but your booking is confirmed. Please note your reservation ID: "
            f"**{reservation_id}**"
        )

    return {"final_response": message}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    llm=None,
    calendar: MockCalendar | None = None,
    webhook_url: str | None = None,
    webhook_client=None,
    checkpointer=None,
):
    """
    Build and compile the LangGraph scheduling assistant.

    All dependencies are injectable for testing:
    - llm: ChatModel (or FakeLLM for tests)
    - calendar: MockCalendar instance
    - webhook_url: URL for notifications
    - webhook_client: httpx client (or FakeWebhookClient for tests)
    - checkpointer: LangGraph checkpointer (SqliteSaver or MemorySaver)
    """
    if llm is None:
        llm = get_llm()
    if calendar is None:
        calendar = MockCalendar()

    # Wrap node functions to inject dependencies via closures
    def _triage(state):
        return triage_node(state, llm=llm)

    def _booking_specialist(state):
        return booking_specialist_node(state, llm=llm)

    def _check_availability(state):
        return check_availability_node(state, calendar=calendar)

    def _reserve(state):
        return reserve_node(state, calendar=calendar)

    def _notify(state):
        return notify_node(state, webhook_url=webhook_url, webhook_client=webhook_client)

    def _respond(state):
        return respond_node(state)

    # Build the graph
    graph = StateGraph(SchedulingState)

    # Add nodes
    graph.add_node("triage", _triage)
    graph.add_node("booking_specialist", _booking_specialist)
    graph.add_node("check_availability", _check_availability)
    graph.add_node("reserve", _reserve)
    graph.add_node("notify", _notify)
    graph.add_node("respond", _respond)

    # Set entry point
    graph.set_entry_point("triage")

    # Conditional edge: triage → general (END) or booking_specialist
    def route_after_triage(state):
        if state.get("route") == "booking":
            return "booking_specialist"
        return END  # general question — final_response already set

    graph.add_conditional_edges("triage", route_after_triage, {
        "booking_specialist": "booking_specialist",
        END: END,
    })

    # Conditional edge: booking_specialist → check_availability or END (missing fields)
    def route_after_booking(state):
        missing = state.get("missing_fields", [])
        if missing:
            return END  # Wait for user to provide missing info
        return "check_availability"

    graph.add_conditional_edges("booking_specialist", route_after_booking, {
        "check_availability": "check_availability",
        END: END,
    })

    # Conditional edge: check_availability → reserve (available) or END (negotiate)
    def route_after_availability(state):
        if state.get("slot_available"):
            return "reserve"
        return END  # Negotiation message set, wait for user

    graph.add_conditional_edges("check_availability", route_after_availability, {
        "reserve": "reserve",
        END: END,
    })

    # Conditional edge: reserve → notify (success) or check_availability (race condition)
    def route_after_reserve(state):
        if state.get("reservation_id"):
            return "notify"
        return "check_availability"  # Race condition, re-check

    graph.add_conditional_edges("reserve", route_after_reserve, {
        "notify": "notify",
        "check_availability": "check_availability",
    })

    # notify → respond (always)
    graph.add_edge("notify", "respond")

    # respond → END (always)
    graph.add_edge("respond", END)

    # Compile with checkpointer
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


def get_initial_state() -> SchedulingState:
    """Return a clean initial state for a new thread."""
    return SchedulingState(
        messages=[],
        route=None,
        raw_date_phrase=None,
        raw_time_phrase=None,
        resolved_date=None,
        resolved_time=None,
        email=None,
        missing_fields=[],
        availability_checked=False,
        slot_available=None,
        proposed_alternatives=[],
        reservation_id=None,
        notification_sent=False,
        final_response=None,
        reference_time=None,
    )
