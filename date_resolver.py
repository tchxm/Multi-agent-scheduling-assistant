"""
date_resolver.py — Deterministic date and time resolution.

The LLM's only job is to extract raw phrases like "tomorrow" or "next Friday at 3pm".
This module resolves those phrases to actual dates/times using a combination of:
  1. A deterministic regex-based handler for relative-weekday patterns that
     dateparser cannot parse (e.g. "next Monday", "this Friday", "coming Wednesday").
  2. dateparser for everything else (e.g. "tomorrow", "in 3 days", "July 20th").

No LLM call touches date arithmetic.
"""

import re
from datetime import datetime, timedelta

import dateparser

# Weekday name → weekday number (Monday=0 .. Sunday=6), matching datetime.weekday()
_WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Regex for "next <weekday>", "this <weekday>", "coming <weekday>",
# "next week <weekday>", "<weekday> next week"
_RELATIVE_WEEKDAY_RE = re.compile(
    r"^(?:(?:next|this|coming)(?:\s+week)?\s+(\w+)"  # "next Monday", "next week Monday"
    r"|(\w+)\s+next\s+week)$",                        # "Monday next week"
    re.IGNORECASE,
)


def _resolve_next_weekday(phrase: str, now: datetime) -> str | None:
    """
    Deterministic handler for 'next <weekday>' and similar patterns.

    dateparser (as of v1.4.1) returns None for "next Monday", "this Tuesday",
    "coming Friday", etc. This function handles them with direct weekday
    arithmetic: it always returns the NEXT occurrence of the named weekday
    that is strictly in the future (at least 1 day from now).

    Returns YYYY-MM-DD or None if the phrase doesn't match the pattern.
    """
    match = _RELATIVE_WEEKDAY_RE.match(phrase.strip())
    if not match:
        return None

    # Either group 1 or group 2 captured the weekday name
    weekday_name = (match.group(1) or match.group(2)).lower()

    if weekday_name not in _WEEKDAY_MAP:
        return None

    target_weekday = _WEEKDAY_MAP[weekday_name]
    current_weekday = now.weekday()

    # Days ahead: always at least 1 day in the future
    days_ahead = (target_weekday - current_weekday) % 7
    if days_ahead == 0:
        days_ahead = 7  # "next Monday" on a Monday means 7 days out

    result = now + timedelta(days=days_ahead)
    return result.strftime("%Y-%m-%d")


def resolve_date_phrase(phrase: str, now: datetime) -> str | None:
    """
    Resolve a relative or absolute date phrase to YYYY-MM-DD.

    First tries a deterministic regex handler for "next <weekday>" patterns
    that dateparser can't parse. Falls through to dateparser (with
    RELATIVE_BASE set to `now`) for everything else.

    Args:
        phrase: Raw date phrase extracted by the LLM, e.g. "tomorrow",
                "next Monday", "July 20th", "in 3 days".
        now:    The current server datetime used as the reference point.

    Returns:
        A YYYY-MM-DD string, or None if the phrase is unparseable.
    """
    if not phrase or not phrase.strip():
        return None

    # Try deterministic weekday handler first
    weekday_result = _resolve_next_weekday(phrase, now)
    if weekday_result is not None:
        return weekday_result

    # Fall through to dateparser for everything else
    try:
        parsed = dateparser.parse(
            phrase.strip(),
            settings={
                "RELATIVE_BASE": now,
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        )
        if parsed is None:
            return None
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None


def resolve_time_phrase(phrase: str) -> str | None:
    """
    Resolve a time phrase to HH:MM (24-hour format).

    Handles formats like "3pm", "3:30 PM", "15:00", "3 in the afternoon".

    Args:
        phrase: Raw time phrase extracted by the LLM.

    Returns:
        An HH:MM string in 24-hour format, or None if unparseable.
    """
    if not phrase or not phrase.strip():
        return None

    try:
        # Pre-process: lowercase it and strip extra spaces to maximize compatibility
        cleaned = phrase.strip().lower()
        parsed = dateparser.parse(
            cleaned,
            languages=["en"],
            settings={
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        )
        if parsed is None:
            return None
        return parsed.strftime("%H:%M")
    except Exception:
        return None
