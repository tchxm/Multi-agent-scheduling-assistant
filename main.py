"""
main.py — FastAPI application for the Multi-Agent Scheduling Assistant.

Endpoints:
  POST /threads              → Create a new conversation thread
  POST /threads/{id}/messages → Send a message, invoke the graph, return state
  GET  /threads/{id}/history  → Full message history for a thread
  GET  /health                → Health check

Lifespan-managed singletons for the compiled graph, calendar DB, and
LangGraph checkpointer. CORS enabled for the frontend.
"""

import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from graph import build_graph, get_initial_state
from mock_calendar import MockCalendar

load_dotenv()


# ---------------------------------------------------------------------------
# App state (populated during lifespan)
# ---------------------------------------------------------------------------

class AppState:
    graph = None
    calendar = None
    checkpointer = None
    checkpoint_conn = None
    known_threads: set = set()


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize singletons at startup, clean up on shutdown."""
    # Checkpointer (LangGraph persistence)
    # SqliteSaver.from_conn_string is a context manager — create conn directly
    db_path = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite3")
    app_state.checkpoint_conn = sqlite3.connect(db_path, check_same_thread=False)
    app_state.checkpointer = SqliteSaver(app_state.checkpoint_conn)
    app_state.checkpointer.setup()

    # Mock calendar (separate DB)
    calendar_path = os.getenv("CALENDAR_DB", "reservations.sqlite3")
    app_state.calendar = MockCalendar(db_path=calendar_path)

    # Build the compiled graph
    webhook_url = os.getenv("WEBHOOK_URL", "")
    app_state.graph = build_graph(
        calendar=app_state.calendar,
        webhook_url=webhook_url,
        checkpointer=app_state.checkpointer,
    )

    yield

    # Cleanup
    app_state.calendar.close()
    app_state.checkpoint_conn.close()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Scheduling Assistant",
    description="LangGraph-powered scheduling with deterministic date resolution and slot negotiation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse

# Serve frontend static files
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


@app.get("/")
async def root_redirect():
    """Redirect root path to the frontend chat UI."""
    return RedirectResponse(url="/frontend/")


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ThreadResponse(BaseModel):
    thread_id: str


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str
    route: str | None = None
    resolved_date: str | None = None
    resolved_time: str | None = None
    missing_fields: list[str] = []
    reservation_id: str | None = None
    notification_sent: bool = False


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    graph_compiled: bool
    checkpointer_ok: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/threads", response_model=ThreadResponse)
async def create_thread():
    """Create a new conversation thread."""
    thread_id = uuid.uuid4().hex[:12]
    app_state.known_threads.add(thread_id)
    return ThreadResponse(thread_id=thread_id)


@app.post("/threads/{thread_id}/messages", response_model=MessageResponse)
async def send_message(thread_id: str, request: MessageRequest):
    """Send a user message, invoke the graph, return the assistant's response."""
    app_state.known_threads.add(thread_id)

    config = {"configurable": {"thread_id": thread_id}}

    # Get existing state from checkpointer (if any)
    existing_state = None
    try:
        existing_state = app_state.graph.get_state(config)
    except Exception:
        pass

    # Build input: append the new human message to existing messages
    existing_messages = []
    if existing_state and existing_state.values:
        existing_messages = existing_state.values.get("messages", [])

    new_message = HumanMessage(content=request.message)
    input_state = {
        "messages": existing_messages + [new_message],
        "route": None,
        "final_response": None,
        "availability_checked": False,
        "slot_available": None,
        "proposed_alternatives": [],
        "notification_sent": False,
    }

    # Carry forward previously extracted fields from prior turns
    if existing_state and existing_state.values:
        prev = existing_state.values
        # If the last turn completed a reservation, start fresh for booking slot details
        is_new_booking = prev.get("reservation_id") is not None
        fields_to_carry = ["email", "reference_time"] if is_new_booking else ["raw_date_phrase", "raw_time_phrase", "resolved_date",
                      "resolved_time", "email", "reservation_id", "reference_time"]
        for field in fields_to_carry:
            if field not in input_state:
                input_state[field] = prev.get(field)

    # Invoke the graph
    try:
        result = app_state.graph.invoke(input_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph invocation error: {str(e)}")

    # Extract response
    final_response = result.get("final_response", "I'm sorry, I didn't understand that.")

    # Append the assistant message to the conversation
    assistant_msg = AIMessage(content=final_response)
    result_messages = result.get("messages", [])
    result_messages.append(assistant_msg)

    # Update state with the assistant message included
    try:
        app_state.graph.update_state(config, {"messages": result_messages})
    except Exception:
        pass  # Non-critical — the main state was already saved by invoke

    return MessageResponse(
        response=final_response,
        route=result.get("route"),
        resolved_date=result.get("resolved_date"),
        resolved_time=result.get("resolved_time"),
        missing_fields=result.get("missing_fields", []),
        reservation_id=result.get("reservation_id"),
        notification_sent=result.get("notification_sent", False),
    )


@app.get("/threads/{thread_id}/history", response_model=HistoryResponse)
async def get_history(thread_id: str):
    """Retrieve full message history for a thread."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = app_state.graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Thread not found")

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Thread not found or empty")

    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(HistoryMessage(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            history.append(HistoryMessage(role="assistant", content=msg.content))

    return HistoryResponse(messages=history)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check: confirms graph is compiled and checkpointer is reachable."""
    graph_ok = app_state.graph is not None
    checkpointer_ok = app_state.checkpointer is not None

    return HealthResponse(
        status="ok" if (graph_ok and checkpointer_ok) else "degraded",
        graph_compiled=graph_ok,
        checkpointer_ok=checkpointer_ok,
    )
