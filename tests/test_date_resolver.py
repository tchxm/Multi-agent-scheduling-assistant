"""
test_date_resolver.py — Unit tests for deterministic date/time resolution.

All tests use a fixed `now` so results are deterministic regardless of
when the test suite runs.
"""

from datetime import datetime

from date_resolver import resolve_date_phrase, resolve_time_phrase


# Fixed reference: Monday, July 13, 2026
FIXED_NOW = datetime(2026, 7, 13, 10, 0, 0)


class TestResolveDatePhrase:
    """Tests for resolve_date_phrase."""

    def test_tomorrow(self):
        result = resolve_date_phrase("tomorrow", FIXED_NOW)
        assert result == "2026-07-14"

    def test_next_monday(self):
        # July 13, 2026 is Monday. "next Monday" must resolve to Monday July 20.
        result = resolve_date_phrase("next Monday", FIXED_NOW)
        assert result == "2026-07-20"

    def test_next_friday(self):
        # From Monday July 13: next Friday = July 17
        result = resolve_date_phrase("next Friday", FIXED_NOW)
        assert result == "2026-07-17"

    def test_next_sunday_is_7_days_out(self):
        # "next Sunday" from Monday July 13 → Sunday July 19
        result = resolve_date_phrase("next Sunday", FIXED_NOW)
        assert result == "2026-07-19"

    def test_next_weekday_case_insensitive(self):
        # From Monday July 13: next Tuesday = Tuesday July 14
        result = resolve_date_phrase("Next TUESDAY", FIXED_NOW)
        assert result == "2026-07-14"

    def test_this_wednesday(self):
        # "this Wednesday" from Monday July 13 → Wednesday July 15
        result = resolve_date_phrase("this Wednesday", FIXED_NOW)
        assert result == "2026-07-15"

    def test_coming_thursday(self):
        # "coming Thursday" from Monday July 13 → Thursday July 16
        result = resolve_date_phrase("coming Thursday", FIXED_NOW)
        assert result == "2026-07-16"

    def test_in_3_days(self):
        result = resolve_date_phrase("in 3 days", FIXED_NOW)
        assert result == "2026-07-16"

    def test_specific_date(self):
        result = resolve_date_phrase("July 20th", FIXED_NOW)
        assert result is not None
        assert "07-20" in result

    def test_garbage_input_returns_none(self):
        result = resolve_date_phrase("asdfghjkl", FIXED_NOW)
        assert result is None

    def test_empty_string_returns_none(self):
        result = resolve_date_phrase("", FIXED_NOW)
        assert result is None

    def test_none_input_returns_none(self):
        result = resolve_date_phrase(None, FIXED_NOW)
        assert result is None

    def test_month_end_rollover(self):
        # Fixed now at July 31 — "tomorrow" should give August 1
        end_of_july = datetime(2026, 7, 31, 10, 0, 0)
        result = resolve_date_phrase("tomorrow", end_of_july)
        assert result == "2026-08-01"

    def test_year_end_rollover(self):
        # Fixed now at Dec 31 — "tomorrow" should give Jan 1 next year
        end_of_year = datetime(2026, 12, 31, 10, 0, 0)
        result = resolve_date_phrase("tomorrow", end_of_year)
        assert result == "2027-01-01"

    def test_next_monday_from_monday(self):
        # "next Monday" on a Monday should be 7 days out, not today
        a_monday = datetime(2026, 7, 20, 10, 0, 0)  # Monday July 20
        result = resolve_date_phrase("next Monday", a_monday)
        assert result == "2026-07-27"

    def test_next_weekday_with_extra_whitespace(self):
        result = resolve_date_phrase("  next  Monday  ", FIXED_NOW)
        assert result == "2026-07-20"


class TestResolveTimePhrase:
    """Tests for resolve_time_phrase."""

    def test_3pm(self):
        result = resolve_time_phrase("3pm")
        assert result == "15:00"

    def test_cases_for_pm_resolve_identically(self):
        assert resolve_time_phrase("4pm") == "16:00"
        assert resolve_time_phrase("4PM") == "16:00"
        assert resolve_time_phrase("4Pm") == "16:00"

    def test_3_30_pm(self):
        result = resolve_time_phrase("3:30 PM")
        assert result == "15:30"

    def test_24_hour_format(self):
        result = resolve_time_phrase("15:00")
        assert result == "15:00"

    def test_9am(self):
        result = resolve_time_phrase("9am")
        assert result == "09:00"

    def test_noon(self):
        result = resolve_time_phrase("noon")
        assert result == "12:00"

    def test_empty_string_returns_none(self):
        result = resolve_time_phrase("")
        assert result is None

    def test_none_input_returns_none(self):
        result = resolve_time_phrase(None)
        assert result is None

    def test_garbage_returns_none(self):
        result = resolve_time_phrase("asdfghjkl")
        assert result is None
