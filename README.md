# Multi-Agent Scheduling Assistant

A LangGraph-powered multi-agent scheduling assistant with deterministic date
resolution, slot negotiation, and mock webhook notifications.

## Architecture

```
User Message
      │
      ▼
┌─────────────┐
│  triage     │ ── general ──► direct answer ──► END
│  (LLM)      │
└──────┬──────┘
       │ booking
       ▼
┌──────────────────┐
│ booking_specialist│ ── missing fields ──► ask user ──► END
│ (LLM + dateparser)│
└──────┬───────────┘
       │ all fields present
       ▼
┌──────────────────┐
│ check_availability│ ── slot taken ──► suggest alternatives ──► END
│ (SQLite)          │                    (negotiation loop)
└──────┬───────────┘
       │ slot available
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  reserve     │ ──► │   notify     │ ──► │   respond    │ ──► END
│  (SQLite)    │     │  (webhook)   │     │  (message)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       │ race condition
       ▼
  back to check_availability
```

## Four Positioning Decisions

1. **Deterministic date resolution, not LLM arithmetic.** LLMs are unreliable
   at date math ("tomorrow" can drift, month-end/leap-year edge cases are
   inconsistent). The LLM only extracts the raw phrase; `dateparser` resolves
   it against the real server `now`. This is testable and correct by construction.

2. **Real multi-node graph, not a single mega-prompt.** Triage and Booking
   Specialist are separate LangGraph nodes with conditional routing. Routing
   logic and validation logic are independently testable, and the trace of
   which node ran is visible for debugging.

3. **Tool failures negotiate, they don't silently retry or crash.** If
   `check_availability` finds the slot taken, the graph proposes concrete
   alternative slots and loops back for the user's choice.

4. **Persistence via LangGraph's SqliteSaver checkpointer**, keyed by
   `thread_id`. No hand-rolled session logic — resumability comes for free.

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn
- **Orchestration:** LangGraph with SqliteSaver checkpointer
- **LLM:** Groq (llama-3.3-70b-versatile) by default; supports OpenAI/Anthropic via env var
- **Date parsing:** dateparser (deterministic relative-date resolution)
- **Calendar DB:** SQLite (`reservations.sqlite3`)
- **Notifications:** httpx POST to webhook.site/Pipedream
- **Frontend:** Single static HTML/CSS/vanilla JS file
- **Tests:** pytest with FakeLLM and FakeWebhookClient

## File Structure

```
Assignment2/
├── main.py                  # FastAPI app with endpoints
├── graph.py                 # LangGraph state machine (6 nodes)
├── date_resolver.py         # Deterministic date/time resolution
├── mock_calendar.py         # SQLite-backed mock calendar
├── notifications.py         # Webhook notification sender
├── requirements.txt         # Pinned dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── frontend/
│   └── index.html           # Vanilla JS chat interface
└── tests/
    ├── __init__.py
    ├── fakes.py             # FakeLLM + FakeWebhookClient
    ├── test_date_resolver.py
    ├── test_mock_calendar.py
    ├── test_notifications.py
    ├── test_graph.py
    └── test_api.py
```

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual keys:
#   GROQ_API_KEY=gsk_...
#   WEBHOOK_URL=https://webhook.site/your-unique-url
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000/frontend/` for the chat interface.

### 4. Run tests (no API key needed)

```bash
pytest tests/ -v
```

All tests use FakeLLM and FakeWebhookClient — zero network calls.

## Deployment (Render)

1. Create a new Web Service on Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `GROQ_API_KEY` = your Groq key
   - `LLM_PROVIDER` = `groq`
   - `WEBHOOK_URL` = your webhook.site URL

### Known Limitation

Render's free tier has an ephemeral disk, so both the reservations DB and the
LangGraph checkpoint DB reset on a full container restart (not on ordinary
page refreshes or idle wake-ups within a running container).

## Demo Video Script

1. **General question** → triage answers directly, no booking flow triggered
2. **"Book tomorrow at 3pm"** → show the debug strip resolving "tomorrow" to
   a real date before any tool runs
3. **Deliberately trigger a taken slot** → show the negotiation/alternative
   slots offered
4. **Complete a booking** → show webhook.site's inbox receiving the mock
   notification live
5. **Refresh the page, resume the same thread_id** → show history intact
6. **Live `pytest tests/ -v` run** → 100% green, no API key needed
7. **DevTools network tab** → show the actual POST `/threads/{id}/messages`
   call, proving frontend/backend separation

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/threads` | Create a new thread |
| `POST` | `/threads/{thread_id}/messages` | Send a message |
| `GET` | `/threads/{thread_id}/history` | Get message history |
| `GET` | `/health` | Health check |

### Example Response (`POST /threads/{id}/messages`)

```json
{
  "response": "You're booked for 2026-07-14 at 10:00...",
  "route": "booking",
  "resolved_date": "2026-07-14",
  "resolved_time": "10:00",
  "missing_fields": [],
  "reservation_id": "a1b2c3d4",
  "notification_sent": true
}
```
