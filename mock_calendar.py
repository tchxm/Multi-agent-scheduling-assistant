"""
mock_calendar.py — SQLite-backed mock calendar for slot management.

Stores reservations in its own DB file (reservations.sqlite3), separate from
LangGraph's checkpoint DB. Pre-seeds available slots for the next several
business days on first initialization, with a few slots pre-booked so
check_availability has realistic "taken" results for demos.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta


# Business hours: 9am to 5pm, on the hour
BUSINESS_HOURS = [f"{h:02d}:00" for h in range(9, 17)]  # 09:00 .. 16:00


def _next_business_days(start: datetime, count: int) -> list[str]:
    """Return the next `count` business days (Mon-Fri) as YYYY-MM-DD strings."""
    days = []
    current = start
    while len(days) < count:
        # weekday(): 0=Mon .. 4=Fri, 5=Sat, 6=Sun
        if current.weekday() < 5:
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


class MockCalendar:
    """
    SQLite-backed mock calendar.

    On init, creates the slots table and pre-seeds available slots for the
    next 5 business days. The first 2 business days have 14:00 and 15:00
    pre-booked to provide realistic "taken" slots for demo/testing.
    """

    def __init__(self, db_path: str = "reservations.sqlite3", now: datetime | None = None):
        self.db_path = db_path
        self.now = now or datetime.now()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Create schema and seed slots if the table doesn't exist yet."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS slots (
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'available',
                email TEXT,
                reservation_id TEXT,
                PRIMARY KEY (date, time)
            )
        """)
        self.conn.commit()

        # Check if slots are already seeded
        cursor.execute("SELECT COUNT(*) FROM slots")
        if cursor.fetchone()[0] > 0:
            return  # Already seeded

        # Seed slots for the next 5 business days
        business_days = _next_business_days(self.now, 5)

        for day in business_days:
            for time_slot in BUSINESS_HOURS:
                cursor.execute(
                    "INSERT OR IGNORE INTO slots (date, time, status) VALUES (?, ?, 'available')",
                    (day, time_slot),
                )

        # Pre-book 14:00 and 15:00 on the first 2 business days
        for day in business_days[:2]:
            for taken_time in ["14:00", "15:00"]:
                cursor.execute(
                    "UPDATE slots SET status = 'booked', email = 'existing@example.com', "
                    "reservation_id = ? WHERE date = ? AND time = ?",
                    (f"pre-{uuid.uuid4().hex[:8]}", day, taken_time),
                )

        self.conn.commit()

    def check_availability(self, date: str) -> list[str]:
        """
        Return list of available HH:MM slots for the given date.

        Args:
            date: YYYY-MM-DD string.

        Returns:
            List of available time strings, e.g. ["09:00", "10:00", "11:00"].
            Empty list if fully booked, invalid date, or no slots exist for that date.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT time FROM slots WHERE date = ? AND status = 'available' ORDER BY time",
            (date,),
        )
        return [row[0] for row in cursor.fetchall()]

    def is_slot_available(self, date: str, time: str) -> bool:
        """
        Check if a specific slot is available.

        Args:
            date: YYYY-MM-DD string.
            time: HH:MM string.

        Returns:
            True if the slot exists and is available, False otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT status FROM slots WHERE date = ? AND time = ?",
            (date, time),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        return row[0] == "available"

    def is_valid_slot(self, date: str, time: str) -> bool:
        """
        Check if a given date and time represent a valid business hours slot.

        Valid slots are:
          - Monday through Friday
          - 9:00 AM to 5:00 PM on the hour (09:00 to 16:00 inclusive)
        """
        if time not in BUSINESS_HOURS:
            return False

        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                return False
        except ValueError:
            return False

        return True

    def reserve_slot(self, date: str, time: str, email: str) -> str | None:
        """
        Atomically reserve a slot.

        Re-checks availability inside the same transaction before writing,
        to handle race conditions where the slot was taken between a prior
        check and this reserve call.

        Args:
            date:  YYYY-MM-DD string.
            time:  HH:MM string.
            email: User's email address.

        Returns:
            A short reservation_id string on success, or None if the slot
            was already taken.
        """
        cursor = self.conn.cursor()
        try:
            # Re-check inside transaction
            cursor.execute(
                "SELECT status FROM slots WHERE date = ? AND time = ?",
                (date, time),
            )
            row = cursor.fetchone()
            if row is None or row[0] != "available":
                return None

            reservation_id = uuid.uuid4().hex[:8]
            cursor.execute(
                "UPDATE slots SET status = 'booked', email = ?, reservation_id = ? "
                "WHERE date = ? AND time = ? AND status = 'available'",
                (email, reservation_id, date, time),
            )
            if cursor.rowcount == 0:
                # Race condition: someone else got it
                return None

            self.conn.commit()
            return reservation_id
        except Exception:
            self.conn.rollback()
            return None

    def suggest_alternatives(self, date: str, count: int = 3) -> list[str]:
        """
        Suggest alternative available slots.

        Looks for available times on the same day first. If not enough,
        checks subsequent days until `count` alternatives are found or
        no more slots exist.

        Args:
            date:  YYYY-MM-DD string of the originally requested date.
            count: Number of alternatives to suggest.

        Returns:
            List of strings like ["2026-07-14 10:00", "2026-07-14 11:00", "2026-07-15 09:00"].
        """
        alternatives = []
        cursor = self.conn.cursor()

        # First: same day, different times
        cursor.execute(
            "SELECT date, time FROM slots WHERE date = ? AND status = 'available' ORDER BY time",
            (date,),
        )
        for row in cursor.fetchall():
            alternatives.append(f"{row[0]} {row[1]}")
            if len(alternatives) >= count:
                return alternatives

        # Then: subsequent days
        cursor.execute(
            "SELECT date, time FROM slots WHERE date > ? AND status = 'available' ORDER BY date, time",
            (date,),
        )
        for row in cursor.fetchall():
            alternatives.append(f"{row[0]} {row[1]}")
            if len(alternatives) >= count:
                return alternatives

        return alternatives

    def close(self):
        """Close the database connection."""
        self.conn.close()
