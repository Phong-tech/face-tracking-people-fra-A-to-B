"""CSV logging and restoring state from earlier runs.

Only anonymous data is stored: timestamps, counts and temporary tracking
IDs.  No images, no video, nothing that identifies a person.

* ``crossing_events.csv``  one row per confirmed ENTER / EXIT (and RESET)
* ``occupancy_log.csv``    periodic snapshots of the counters

Files are opened in append mode for every write, flushed and fsync'ed, so
nothing is lost if the program is killed, and old data is never overwritten.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.counter import ENTER, EXIT, RESET, OccupancyState

log = logging.getLogger("people_counter.logger")

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OCCUPANCY_HEADER = [
    "timestamp", "current_occupancy", "total_entered",
    "total_exited", "peak_occupancy", "peak_timestamp",
]
EVENTS_HEADER = ["timestamp", "tracking_id", "direction", "current_occupancy"]


def now_string(moment: Optional[datetime] = None) -> str:
    """Local date and time, e.g. ``2026-10-28 14:03:59``."""
    return (moment or datetime.now()).strftime(TIME_FORMAT)


def parse_timestamp(text: str) -> Optional[datetime]:
    """Parse a logged timestamp; returns None if it is malformed."""
    try:
        return datetime.strptime(text.strip(), TIME_FORMAT)
    except (ValueError, AttributeError):
        return None


# ------------------------------------------------------------ reading CSVs
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries.  Missing file -> empty list."""
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return []


@dataclass
class ReplayResult:
    """Totals rebuilt by replaying crossing_events.csv."""

    # Since the most recent RESET (this is what the app restores on start-up).
    entered: int = 0
    exited: int = 0
    current: int = 0
    peak: int = 0
    peak_timestamp: Optional[str] = None
    # Across the whole history, ignoring resets (used by the report).
    lifetime_entered: int = 0
    lifetime_exited: int = 0
    lifetime_peak: int = 0
    lifetime_peak_timestamp: Optional[str] = None
    events_replayed: int = 0


def replay_events(
    rows: list[dict[str, str]],
    timeline: Optional[list[tuple[str, int]]] = None,
) -> ReplayResult:
    """Rebuild totals and the peak by replaying every logged crossing.

    If ``timeline`` is given, ``(timestamp, occupancy)`` pairs are appended
    to it so a chart can be drawn.
    """
    result = ReplayResult()
    state = OccupancyState()

    def close_session() -> None:
        # Fold the session's peak into the lifetime peak (earliest wins ties).
        if state.peak > result.lifetime_peak:
            result.lifetime_peak = state.peak
            result.lifetime_peak_timestamp = state.peak_timestamp

    for row in rows:
        direction = (row.get("direction") or "").strip().upper()
        timestamp = (row.get("timestamp") or "").strip()
        if direction == RESET:
            close_session()
            state.reset()
            if timeline is not None:
                timeline.append((timestamp, 0))
        elif direction in (ENTER, EXIT):
            if state.apply(direction, timestamp):
                result.events_replayed += 1
                if direction == ENTER:
                    result.lifetime_entered += 1
                else:
                    result.lifetime_exited += 1
                if timeline is not None:
                    timeline.append((timestamp, state.current))
        # Rows with any other direction are skipped as corrupt.

    close_session()
    result.entered = state.entered
    result.exited = state.exited
    result.current = state.current
    result.peak = state.peak
    result.peak_timestamp = state.peak_timestamp
    return result


# ------------------------------------------------------------------ writer
class OccupancyLogger:
    """Writes the two CSV files and restores totals from them."""

    def __init__(self, data_dir: Path, interval_seconds: float) -> None:
        self.data_dir = data_dir
        self.occupancy_path = data_dir / "occupancy_log.csv"
        self.events_path = data_dir / "crossing_events.csv"
        self._interval = interval_seconds
        self._last_logged = float("-inf")
        self._error_reported = False

        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.error("Cannot create data folder %s: %s", data_dir, exc)
        self._ensure_header(self.occupancy_path, OCCUPANCY_HEADER)
        self._ensure_header(self.events_path, EVENTS_HEADER)

    # ------------------------------------------------------------ restoring
    def restore_state(self) -> OccupancyState:
        """Rebuild the counters from existing logs (empty state if none)."""
        events = read_csv_rows(self.events_path)
        result = replay_events(events)
        if result.events_replayed or any(
            (r.get("direction") or "").upper() == RESET for r in events
        ):
            return OccupancyState(result.entered, result.exited,
                                  result.peak, result.peak_timestamp)

        # No crossing events: fall back to the last periodic snapshot.
        samples = read_csv_rows(self.occupancy_path)
        if samples:
            last = samples[-1]
            try:
                return OccupancyState(
                    int(last["total_entered"]), int(last["total_exited"]),
                    int(last["peak_occupancy"]), (last.get("peak_timestamp") or None),
                )
            except (KeyError, ValueError):
                log.warning("Last row of %s is unreadable; starting from zero",
                            self.occupancy_path.name)
        return OccupancyState()

    # -------------------------------------------------------------- writing
    def log_crossing(self, timestamp: str, track_id: int, direction: str, current: int) -> None:
        self._append(self.events_path, [timestamp, track_id, direction, current])

    def log_reset(self, timestamp: str) -> None:
        """Record a manual reset so later restarts do not resurrect old totals."""
        self._append(self.events_path, [timestamp, "", RESET, 0])

    def log_occupancy(self, timestamp: str, state: OccupancyState) -> None:
        self._append(self.occupancy_path, [
            timestamp, state.current, state.entered, state.exited,
            state.peak, state.peak_timestamp or "",
        ])

    def maybe_log_occupancy(self, now: float, timestamp: str, state: OccupancyState) -> bool:
        """Write a periodic snapshot if ``interval_seconds`` have passed."""
        if now - self._last_logged < self._interval:
            return False
        self._last_logged = now
        self.log_occupancy(timestamp, state)
        return True

    # ------------------------------------------------------------ internals
    def _ensure_header(self, path: Path, header: list[str]) -> None:
        """Create the file with a header if it is missing or empty; never overwrite."""
        try:
            if path.exists() and path.stat().st_size > 0:
                return
            self._append(path, header)
        except OSError as exc:
            log.error("Cannot prepare %s: %s", path, exc)

    def _append(self, path: Path, row: list) -> None:
        try:
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
                handle.flush()
                os.fsync(handle.fileno())      # survive a crash / power loss
        except OSError as exc:
            # A full disk must not stop people counting; report it once.
            if not self._error_reported:
                log.error("Could not write to %s: %s", path, exc)
                self._error_reported = True
