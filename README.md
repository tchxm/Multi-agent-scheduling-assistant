# Multi-Agent Scheduling Assistant

A production-quality, multi-agent scheduling assistant built with **LangGraph**, **FastAPI**, and a **vanilla JavaScript** frontend. The system uses a 6-node state machine graph to route conversations through triage, slot extraction, availability checking, reservation, webhook notification, and response generation — each as an independently testable node with explicit state transitions.

> **69 automated tests** · Zero API keys required to test · Full frontend/backend separation · Deterministic date parsing · SQLite persistence

---

## Table of Contents

- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Run](#setup--run)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment (Render)](#deployment-render)
- [Demo Walkthrough](#demo-walkthrough)

---

## Architecture

```
                            User Message
                                 │
                                 ▼
                         ┌──────────────┐
                    ┌────│    triage     │────┐
                    │    │    (LLM)      │    │
                    │    └──────────────┘     │
              booking                    general / unsupported
                    │                        │
                    ▼                        ▼
          ┌──────────────────┐          Direct answer
          │ booking_specialist│          or honest refusal
          │ (LLM + dateparser)│              ──► END
          └────────┬─────────┘
                   │
          missing fields? ──yes──► ask user ──► END
                   │                 (wait for reply)
                   │ no
                   ▼
          ┌──────────────────┐
          │ check_availability│──── invalid slot ──► policy message ──► END
          │     (SQLite)      │
          └────────┬─────────┘
                   │
          slot taken? ──yes──► suggest alternatives ──► END
                   │                (negotiation loop)
                   │ available
                   ▼
          ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
          │   reserve    │ ──► │    notify     │ ──► │   respond    │ ──► END
          │   (SQLite)   │     │  (webhook)   │     │  (message)   │
          └──────┬───────┘     └──────────────┘     └──────────────┘
                 │
          race condition?
                 │
                 ▼
          back to check_availability
```

**Key architectural property:** The `respond` node — the only node that generates a "✅ booked" confirmation — is structurally unreachable unless `reserve_slot()` returned a real database ID. This is enforced by both the graph's conditional edges and a hard guard inside `respond_node` itself, making it impossible for the LLM to hallucinate a successful booking.

---

## Design Decisions

### 1. Deterministic Date Resolution (Not LLM Arithmetic)

LLMs are unreliable at date math — "tomorrow" can drift, month-end rollovers are inconsistent, and relative weekdays like "next Monday" often return wrong results. Instead:

- The LLM's **only job** is to extract the raw phrase (e.g., `"next Friday"`, `"3pm"`).
- `date_resolver.py` resolves it deterministically using a **regex-based weekday handler** (for patterns `dateparser` can't handle) plus `dateparser` for everything else.
- All date arithmetic uses `RELATIVE_BASE` tied to the server's `now`, making tests fully deterministic.

### 2. Real Multi-Node Graph (Not a Single Mega-Prompt)

Each node has one narrow responsibility:

| Node | Responsibility | LLM? |
|------|---------------|------|
| `triage` | Classify intent (booking vs. general vs. unsupported) | ✅ |
| `booking_specialist` | Extract date/time/email from latest message | ✅ |
| `check_availability` | Validate business hours, check slot in SQLite | ❌ |
| `reserve` | Atomically reserve slot with race-condition handling | ❌ |
| `notify` | POST confirmation to webhook | ❌ |
| `respond` | Build final confirmation message (grounded in real data) | ❌ |

Only 2 of 6 nodes use the LLM. The remaining 4 are pure deterministic logic operating on state fields — independently testable, auditable, and reproducible.

### 3. Tool Failures Negotiate (Not Silently Retry or Crash)

When `check_availability` finds a taken slot, it doesn't fail silently. Instead it:
1. Queries for up to 3 alternative available slots nearby.
2. Formats them as a human-readable list.
3. Clears the resolved date/time so the next user reply gets freshly extracted.

This creates a natural **negotiation loop** where the user can pick an alternative or suggest a new time.

### 4. Grounded Responses (No Hallucinated Confirmations)

The system enforces a strict rule: **no action is ever confirmed to the user without a real tool call backing it.**

- `respond_node` refuses to emit a "✅ booked" message unless `reservation_id` is non-null.
- Unsupported actions (cancel/reschedule) are intercepted before the LLM runs, with a deterministic refusal message — zero chance of fabrication.
- The general-answer LLM prompt explicitly prohibits claiming any action was completed.

### 5. Persistence via LangGraph's SqliteSaver

Conversation state is checkpointed after every turn using LangGraph's `SqliteSaver`, keyed by `thread_id`. This provides:
- Free resumability — refresh the page, paste the thread ID, and the full conversation loads.
- No hand-rolled session management code.
- The frontend auto-recovers from stale thread IDs (e.g., after server restart) by detecting the 404 and creating a new thread automatically.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.11+, FastAPI, Uvicorn | REST API with async lifespan management |
| **Orchestration** | LangGraph + SqliteSaver | State machine graph with checkpointed persistence |
| **LLM** | Groq (Llama 3.3 70B) | Default provider; swappable to OpenAI/Anthropic via env var |
| **Date Parsing** | dateparser + custom regex | Deterministic relative-date resolution |
| **Calendar DB** | SQLite (`reservations.sqlite3`) | Slot storage with atomic reservations |
| **Notifications** | httpx POST | Webhook notifications to webhook.site/Pipedream |
| **Frontend** | Vanilla HTML/CSS/JS | Single-file glassmorphic chat UI with debug strips |
| **Tests** | pytest | 69 tests with FakeLLM/FakeWebhookClient — zero network calls |

---

## Project Structure

```
Assignment2/
├── main.py                   # FastAPI app (4 endpoints, lifespan-managed singletons)
├── graph.py                  # LangGraph state machine (6 nodes, conditional edges)
├── date_resolver.py          # Deterministic date/time resolution (regex + dateparser)
├── mock_calendar.py          # SQLite-backed calendar (slots, reservations, alternatives)
├── notifications.py          # Webhook notification sender (dependency-injectable)
├── requirements.txt          # Pinned dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Excludes .env, *.sqlite3, __pycache__, etc.
├── README.md                 # This file
├── frontend/
│   └── index.html            # Vanilla JS chat UI (glassmorphic, debug strips, thread management)
└── tests/
    ├── __init__.py
    ├── fakes.py              # FakeLLM + FakeWebhookClient (test doubles)
    ├── test_date_resolver.py # 25 tests — date/time parsing edge cases
    ├── test_mock_calendar.py # 16 tests — slot management, reservations, alternatives
    ├── test_notifications.py #  7 tests — webhook success/failure/timeout handling
    ├── test_graph.py         # 16 tests — graph routing, state management, safety guards
    └── test_api.py           #  5 tests — HTTP endpoint integration tests
```

---

## Setup & Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```env
GROQ_API_KEY=gsk_your-actual-key
LLM_PROVIDER=groq
WEBHOOK_URL=https://webhook.site/your-unique-url
```

### 3. Run the Server

```bash
uvicorn main:app --reload
```

Visit **http://localhost:8000/** for the chat interface.

### 4. Run Tests (No API Key Needed)

```bash
pytest tests/ -v
```

All 69 tests use `FakeLLM` and `FakeWebhookClient` — **zero network calls**, fully offline.

---

## API Reference

### `POST /threads`

Create a new conversation thread.

**Response:**
```json
{ "thread_id": "a1b2c3d4e5f6" }
```

---

### `POST /threads/{thread_id}/messages`

Send a user message. The graph runs to completion and returns the assistant's response plus full state metadata.

**Request:**
```json
{ "message": "Book tomorrow at 3pm, email me@example.com" }
```

**Response:**
```json
{
  "response": "✅ You're all set! Your appointment is booked for 2026-07-16 at 15:00...",
  "route": "booking",
  "resolved_date": "2026-07-16",
  "resolved_time": "15:00",
  "missing_fields": [],
  "reservation_id": "a1b2c3d4",
  "notification_sent": true
}
```

---

### `GET /threads/{thread_id}/history`

Retrieve the full message history for a thread (for resuming sessions).

**Response:**
```json
{
  "messages": [
    { "role": "user", "content": "Book tomorrow at 3pm" },
    { "role": "assistant", "content": "I'd love to help! Could you tell me your email?" }
  ]
}
```

---

### `GET /health`

Health check. Confirms the graph is compiled and the checkpointer is connected.

**Response:**
```json
{
  "status": "ok",
  "graph_compiled": true,
  "checkpointer_ok": true
}
```

---

## Testing

### Test Architecture

All tests are fully **hermetic** — no API keys, no network calls, no flaky external dependencies:

- **`FakeLLM`** — A deterministic LLM stand-in that returns pre-scripted responses in order. Tests declare exactly what the LLM should "say" at each step.
- **`FakeWebhookClient`** — Records all POST calls and lets tests assert on payloads.
- **In-memory SQLite** — Each test gets a fresh `:memory:` database for the calendar and checkpointer.
- **Fixed `reference_time`** — Date-dependent tests pin `now` to a specific timestamp, making them deterministic regardless of when they run.

### Test Coverage Summary

| Module | Tests | What's Verified |
|--------|------:|-----------------|
| `test_date_resolver.py` | 25 | tomorrow, next Monday/Friday/Sunday, case-insensitive AM/PM, "this Wednesday", "coming Thursday", "in 3 days", specific dates (July 20th), garbage/empty/None inputs, month-end rollover, year-end rollover, next Monday from a Monday, extra whitespace handling |
| `test_mock_calendar.py` | 16 | business day generation, weekend skipping, slot availability checks, valid/invalid slot validation, atomic reservations, race-condition double-booking prevention, alternative slot suggestions, cross-day fallthrough |
| `test_notifications.py` | 7 | successful webhook POST, HTTP 500/404 error handling, connection errors, timeouts, generic exception safety, URL correctness |
| `test_graph.py` | 16 | general question routing, missing-field prompting, full happy-path booking, taken-slot negotiation, cross-invocation state persistence, out-of-hours policy enforcement, 6-message state divergence regression, unsupported cancel/reschedule interception, respond_node safety guard, draft state clearing |
| `test_api.py` | 5 | thread creation, health check, multi-turn HTTP booking flow, negotiation via HTTP, history persistence |
| **Total** | **69** | |

### Running

```bash
# Full verbose run
pytest tests/ -v

# Single module
pytest tests/test_graph.py -v

# Single test
pytest tests/test_graph.py::TestFullBookingHappyPath::test_happy_path -v
```

---

## Deployment (Render)

### Steps

1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add **Environment Variables**:

   | Variable | Value |
   |----------|-------|
   | `GROQ_API_KEY` | Your Groq API key |
   | `LLM_PROVIDER` | `groq` |
   | `WEBHOOK_URL` | Your webhook.site URL |

### Known Limitations

- **Ephemeral disk on free tier:** Render's free tier resets disk on full container restarts, which clears both the reservations DB and the LangGraph checkpoint DB. Conversations and bookings are preserved across idle wake-ups and hot reloads, but not across cold starts.
- **Frontend auto-recovery:** The frontend detects stale thread IDs (from a prior server session) and automatically creates a new thread, preventing the user from being stuck on a dead conversation.

---

## Demo Walkthrough

The following sequence demonstrates all key features in order:

| Step | What to Show | What to Look For |
|------|-------------|-----------------|
| 1 | Type a greeting: *"Hi there!"* | Triage routes to `general`, debug strip shows `route: general` |
| 2 | Ask a general question: *"What are your business hours?"* | Gets a helpful answer without entering the booking flow |
| 3 | Book an appointment: *"Book tomorrow at 3pm"* | Debug strip shows resolved date (actual YYYY-MM-DD, not "tomorrow"), `missing: [email]` |
| 4 | Provide email: *"user@example.com"* | Full booking completes, `reservation_id` appears, `notified: true` |
| 5 | Open [webhook.site](https://webhook.site) | The webhook POST with `{email, date, time, reservation_id}` is visible |
| 6 | Book a taken slot (e.g., *"Book tomorrow at 2pm"* if 14:00 is pre-booked) | `slot_available: false`, alternative slots are suggested |
| 7 | Try cancelling: *"Cancel my appointment"* | Gets an honest refusal: *"cancellations aren't supported"*, state is cleared |
| 8 | Copy the thread ID, refresh the page, paste it in "Resume Thread" | Full conversation history loads from the checkpointer |
| 9 | Run `pytest tests/ -v` in terminal | All 69 tests pass — no API key needed |
| 10 | Open browser DevTools → Network tab | Shows the actual `POST /threads/{id}/messages` calls, proving real frontend/backend separation |
