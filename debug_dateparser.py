"""Debug dateparser behavior with relative weekday phrases."""
import dateparser
from datetime import datetime

now = datetime(2026, 7, 13, 10, 0, 0)  # Sunday July 13, 2026
print(f"Reference now: {now} (Sunday)\n")

settings_combo = [
    {"RELATIVE_BASE": now, "PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
    {"RELATIVE_BASE": now, "PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False, "STRICT_PARSING": False},
    {"RELATIVE_BASE": now, "RETURN_AS_TIMEZONE_AWARE": False},
]

phrases = [
    "next Monday",
    "next monday",
    "next Tuesday",
    "next Friday",
    "Monday",
    "this Monday",
    "coming Monday",
    "next week Monday",
    "on Monday",
]

for i, settings in enumerate(settings_combo):
    print(f"--- Settings combo {i}: {settings} ---")
    for phrase in phrases:
        result = dateparser.parse(phrase, settings=settings)
        print(f"  {phrase!r:25s} => {result}")
    print()

# Also try with explicit languages
print("--- With languages=['en'] ---")
for phrase in phrases:
    result = dateparser.parse(
        phrase,
        languages=["en"],
        settings={"RELATIVE_BASE": now, "PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
    )
    print(f"  {phrase!r:25s} => {result}")
