"""
test_notifications.py — Unit tests for the webhook notification sender.

Uses FakeWebhookClient so no real network calls are made.
"""

import pytest

from notifications import send_booking_notification
from tests.fakes import FakeWebhookClient


WEBHOOK_URL = "https://webhook.site/test-endpoint"
TEST_EMAIL = "user@example.com"
TEST_DETAILS = {
    "date": "2026-07-14",
    "time": "10:00",
    "reservation_id": "abc12345",
}


class TestSendBookingNotification:
    """Tests for send_booking_notification."""

    def test_successful_notification(self):
        client = FakeWebhookClient(status_code=200)
        result = send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert result is True
        assert len(client.posts) == 1
        payload = client.posts[0]["json"]
        assert payload["email"] == TEST_EMAIL
        assert payload["date"] == "2026-07-14"
        assert payload["time"] == "10:00"
        assert payload["reservation_id"] == "abc12345"

    def test_server_error_returns_false(self):
        client = FakeWebhookClient(status_code=500)
        result = send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert result is False

    def test_404_returns_false(self):
        client = FakeWebhookClient(status_code=404)
        result = send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert result is False

    def test_connection_error_returns_false(self):
        client = FakeWebhookClient(raise_on_post=ConnectionError("Network unreachable"))
        result = send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert result is False

    def test_timeout_error_returns_false(self):
        client = FakeWebhookClient(raise_on_post=TimeoutError("Request timed out"))
        result = send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert result is False

    def test_never_raises_unhandled_exception(self):
        """Any exception type should be caught, never propagated."""
        client = FakeWebhookClient(raise_on_post=RuntimeError("Unexpected error"))
        # Should not raise
        result = send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert result is False

    def test_correct_url_used(self):
        client = FakeWebhookClient(status_code=200)
        send_booking_notification(WEBHOOK_URL, TEST_EMAIL, TEST_DETAILS, client=client)
        assert client.posts[0]["url"] == WEBHOOK_URL
