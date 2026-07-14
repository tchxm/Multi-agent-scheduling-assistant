"""
test_mock_calendar.py — Unit tests for the SQLite-backed mock calendar.

Uses a temporary in-memory or temp-file DB with a fixed `now` for
deterministic pre-seeding.
"""

import os
import tempfile
from datetime import datetime

import pytest

from mock_calendar import MockCalendar, _next_business_days


# Fixed reference: Tuesday, July 14, 2026
FIXED_NOW = datetime(2026, 7, 14, 10, 0, 0)


@pytest.fixture
def calendar(tmp_path):
    """Create a fresh MockCalendar with a temp DB and fixed now."""
    db_path = str(tmp_path / "test_calendar.sqlite3")
    cal = MockCalendar(db_path=db_path, now=FIXED_NOW)
    yield cal
    cal.close()


class TestNextBusinessDays:
    """Tests for the _next_business_days helper."""

    def test_returns_correct_count(self):
        days = _next_business_days(FIXED_NOW, 5)
        assert len(days) == 5

    def test_skips_weekends(self):
        # July 14, 2026 is Tuesday → 5 business days: Tue-Fri + next Mon
        days = _next_business_days(FIXED_NOW, 5)
        assert days == ["2026-07-14", "2026-07-15", "2026-07-16",
                        "2026-07-17", "2026-07-20"]

    def test_starting_on_saturday(self):
        saturday = datetime(2026, 7, 18, 10, 0, 0)  # July 18, 2026 = Sat
        days = _next_business_days(saturday, 3)
        # Should skip Sat/Sun, start from Mon July 20
        assert days == ["2026-07-20", "2026-07-21", "2026-07-22"]


class TestCheckAvailability:
    """Tests for check_availability."""

    def test_returns_available_slots(self, calendar):
        # July 14 is the first business day — 14:00 and 15:00 should be taken
        available = calendar.check_availability("2026-07-14")
        assert "14:00" not in available
        assert "15:00" not in available
        assert "09:00" in available
        assert "10:00" in available

    def test_all_seeded_business_hours_present(self, calendar):
        # July 16 (third business day) should have all 8 slots available
        available = calendar.check_availability("2026-07-16")
        assert len(available) == 8  # 09:00 through 16:00

    def test_nonexistent_date_returns_empty(self, calendar):
        available = calendar.check_availability("2030-01-01")
        assert available == []


class TestIsValidSlot:
    """Tests for is_valid_slot."""

    def test_valid_weekday_business_hours(self, calendar):
        # Tuesday July 14, 2026 at 9:00 AM (09:00) is valid
        assert calendar.is_valid_slot("2026-07-14", "09:00") is True
        # Tuesday July 14, 2026 at 4:00 PM (16:00) is valid
        assert calendar.is_valid_slot("2026-07-14", "16:00") is True

    def test_invalid_weekend(self, calendar):
        # Saturday July 18, 2026 is invalid (weekend)
        assert calendar.is_valid_slot("2026-07-18", "09:00") is False

    def test_invalid_business_hours(self, calendar):
        # Tuesday July 14, 2026 at 7:00 PM (19:00) is invalid
        assert calendar.is_valid_slot("2026-07-14", "19:00") is False
        # Tuesday July 14, 2026 at 8:00 AM (08:00) is invalid
        assert calendar.is_valid_slot("2026-07-14", "08:00") is False
        # Tuesday July 14, 2026 at 9:30 AM (09:30) is invalid (not on the hour)
        assert calendar.is_valid_slot("2026-07-14", "09:30") is False


class TestIsSlotAvailable:
    """Tests for is_slot_available."""

    def test_available_slot(self, calendar):
        assert calendar.is_slot_available("2026-07-14", "09:00") is True

    def test_taken_slot(self, calendar):
        assert calendar.is_slot_available("2026-07-14", "14:00") is False

    def test_nonexistent_slot(self, calendar):
        assert calendar.is_slot_available("2030-01-01", "09:00") is False


class TestReserveSlot:
    """Tests for reserve_slot."""

    def test_successful_reservation(self, calendar):
        rid = calendar.reserve_slot("2026-07-14", "09:00", "test@example.com")
        assert rid is not None
        assert len(rid) == 8  # short uuid hex

    def test_slot_marked_taken_after_reservation(self, calendar):
        calendar.reserve_slot("2026-07-14", "10:00", "test@example.com")
        assert calendar.is_slot_available("2026-07-14", "10:00") is False

    def test_reserving_taken_slot_returns_none(self, calendar):
        # 14:00 on July 14 is pre-booked
        rid = calendar.reserve_slot("2026-07-14", "14:00", "test@example.com")
        assert rid is None

    def test_double_reservation_returns_none(self, calendar):
        # Reserve once, then try again
        rid1 = calendar.reserve_slot("2026-07-14", "11:00", "user1@example.com")
        rid2 = calendar.reserve_slot("2026-07-14", "11:00", "user2@example.com")
        assert rid1 is not None
        assert rid2 is None


class TestSuggestAlternatives:
    """Tests for suggest_alternatives."""

    def test_returns_alternatives(self, calendar):
        # 14:00 is taken on July 14, so alternatives should be other times
        alts = calendar.suggest_alternatives("2026-07-14", count=3)
        assert len(alts) == 3
        # None of the alternatives should be the taken slots
        for alt in alts:
            assert "14:00" not in alt or "2026-07-14" not in alt

    def test_alternatives_are_available_slots(self, calendar):
        alts = calendar.suggest_alternatives("2026-07-14", count=3)
        for alt in alts:
            parts = alt.split(" ")
            assert len(parts) == 2
            date, time = parts
            assert calendar.is_slot_available(date, time)

    def test_falls_through_to_next_day(self, calendar):
        # Book all remaining slots on July 14
        available = calendar.check_availability("2026-07-14")
        for slot in available:
            calendar.reserve_slot("2026-07-14", slot, "bulk@example.com")

        # Now ask for alternatives on July 14 — should get slots from July 15+
        alts = calendar.suggest_alternatives("2026-07-14", count=2)
        assert len(alts) >= 1
        assert "2026-07-15" in alts[0] or "2026-07-16" in alts[0]
