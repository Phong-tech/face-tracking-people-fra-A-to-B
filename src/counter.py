"""Line-crossing and occupancy logic.

This file contains NO camera or YOLO code.  It only works with numbers
(x/y positions and times), which makes it easy to unit-test and easy to
explain in a report.

How a crossing is decided
-------------------------
Every frame, each tracked person has one reference point (near their
feet).  We measure how far that point is from the counting line:

    offset = point_coordinate - line_position

* offset < -dead_zone  -> the person is clearly on side A  (side = -1)
* offset > +dead_zone  -> the person is clearly on side B  (side = +1)
* otherwise            -> the person is inside the dead zone (side = 0)

A crossing is only counted when the person's last *clear* side is
different from the new *clear* side.  While they are in the dead zone
nothing happens, so standing on or wobbling around the line cannot
produce repeated counts (this is hysteresis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

ENTER = "ENTER"
EXIT = "EXIT"
RESET = "RESET"   # written to the event log when the user resets the counters


@dataclass
class _TrackState:
    """What we remember about one tracking ID while it is on screen."""

    side: Optional[int]   # last clear side (-1 or +1); None until seen clearly
    last_seen: float      # time of last update, in seconds


class LineCrossingDetector:
    """Detects ENTER/EXIT crossings of a straight line for many tracks."""

    def __init__(
        self,
        orientation: str,
        line_position: float,
        dead_zone: float,
        entry_direction: str,
        timeout_seconds: float,
    ) -> None:
        if orientation not in ("horizontal", "vertical"):
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        if dead_zone < 0:
            raise ValueError("dead_zone must not be negative")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.orientation = orientation
        self.line_position = float(line_position)
        self.dead_zone = float(dead_zone)
        self.timeout_seconds = float(timeout_seconds)
        self._entry_positive = True
        self.set_entry_direction(entry_direction)
        self._tracks: dict[int, _TrackState] = {}

    # ------------------------------------------------------------------ setup
    def set_entry_direction(self, entry_direction: str) -> None:
        """Choose which movement direction counts as ENTER.

        "down" / "right" mean moving towards larger pixel coordinates
        (pixel y grows downwards, x grows to the right).
        """
        valid = ("down", "up") if self.orientation == "horizontal" else ("right", "left")
        if entry_direction not in valid:
            raise ValueError(f"entry_direction must be one of {valid}")
        self._entry_positive = entry_direction in ("down", "right")

    @property
    def entry_direction(self) -> str:
        if self.orientation == "horizontal":
            return "down" if self._entry_positive else "up"
        return "right" if self._entry_positive else "left"

    def set_line(self, line_position: float, dead_zone: Optional[float] = None) -> None:
        """Move the line.  Call :meth:`rebaseline` if the move was deliberate."""
        self.line_position = float(line_position)
        if dead_zone is not None:
            self.dead_zone = float(dead_zone)

    def rebaseline(self) -> None:
        """Forget which side everybody was on.

        Needed after the user moves the line, otherwise people who are
        suddenly on a "different" side would be counted by mistake.
        """
        for state in self._tracks.values():
            state.side = None

    # --------------------------------------------------------------- tracking
    def _side_of(self, x: float, y: float) -> int:
        coordinate = y if self.orientation == "horizontal" else x
        offset = coordinate - self.line_position
        if offset < -self.dead_zone:
            return -1
        if offset > self.dead_zone:
            return 1
        return 0

    def update(self, track_id: int, point: tuple[float, float], now: float) -> Optional[str]:
        """Feed one tracked position.  Returns ENTER, EXIT or None."""
        side = self._side_of(point[0], point[1])

        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState(side=None, last_seen=now)
            self._tracks[track_id] = state
        state.last_seen = now

        if side == 0:
            return None                      # inside the dead zone: undecided
        if state.side is None:
            state.side = side                # first clear sighting: just remember it
            return None
        if side == state.side:
            return None                      # still on the same side

        # The person has clearly moved from one side to the other.
        state.side = side                    # remember the new side -> no repeat count
        moved_positive = side > 0
        return ENTER if moved_positive == self._entry_positive else EXIT

    def expire_old_tracks(self, now: float) -> list[int]:
        """Forget IDs not seen for ``timeout_seconds``.  Returns the removed IDs."""
        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.last_seen >= self.timeout_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return expired

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)


class OccupancyState:
    """Running totals: entered, exited, current occupancy and the peak."""

    def __init__(
        self,
        entered: int = 0,
        exited: int = 0,
        peak: int = 0,
        peak_timestamp: Optional[str] = None,
    ) -> None:
        self.entered = max(0, entered)
        # Exits can never exceed entries, otherwise occupancy would be negative.
        self.exited = min(max(0, exited), self.entered)
        self.peak = max(peak, self.current)
        self.peak_timestamp = peak_timestamp

    @property
    def current(self) -> int:
        """current occupancy = total entered - total exited (never below 0)."""
        return self.entered - self.exited

    def apply(self, direction: str, timestamp: str) -> bool:
        """Apply one crossing.  Returns False if it was ignored.

        An EXIT while the room is already counted as empty is ignored: it
        usually means somebody was inside before the program started, and
        it keeps the occupancy from going negative.
        """
        if direction == ENTER:
            self.entered += 1
            if self.current > self.peak:
                self.peak = self.current
                self.peak_timestamp = timestamp
            return True
        if direction == EXIT:
            if self.current <= 0:
                return False
            self.exited += 1
            return True
        raise ValueError(f"Unknown direction: {direction!r}")

    def reset(self) -> None:
        self.entered = 0
        self.exited = 0
        self.peak = 0
        self.peak_timestamp = None
