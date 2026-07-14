"""
notifications.py — Mock notification sender via webhook.

POSTs booking confirmation payloads to a webhook URL (e.g. webhook.site
or Pipedream). Dependency-injectable: accepts an optional httpx client
so tests never hit a real network endpoint.
"""

import httpx


def send_booking_notification(
    webhook_url: str,
    email: str,
    details: dict,
    client: httpx.Client | None = None,
) -> bool:
    """
    Send a booking confirmation notification via webhook POST.

    Args:
        webhook_url: The URL to POST to (e.g. https://webhook.site/xxx).
        email:       The user's email address.
        details:     Dict with booking details (date, time, reservation_id).
        client:      Optional httpx.Client for dependency injection in tests.
                     If None, creates a fresh client for the request.

    Returns:
        True if the POST succeeded (2xx status), False on any failure.
        Never raises — catches all exceptions and returns False.
        A failed notification should be treated as "booking succeeded,
        but confirmation email is delayed."
    """
    payload = {
        "email": email,
        "date": details.get("date"),
        "time": details.get("time"),
        "reservation_id": details.get("reservation_id"),
    }

    try:
        if client is not None:
            response = client.post(webhook_url, json=payload)
        else:
            response = httpx.post(webhook_url, json=payload, timeout=10.0)

        return 200 <= response.status_code < 300
    except Exception:
        # Network error, timeout, DNS failure, etc.
        # Never let a notification failure crash the booking flow.
        return False
