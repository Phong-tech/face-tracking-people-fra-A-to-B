"""Unit tests for the camera-free logic.  Run with:  python3 -m pytest -v"""

from __future__ import annotations

import pytest

from src.counter import ENTER, EXIT, LineCrossingDetector, OccupancyState
from src.serial_manager import SerialManager, format_count_message, parse_count_message

LINE = 100.0
DEAD_ZONE = 20.0          # dead band = 80 ... 120


def make_detector(entry: str = "down", orientation: str = "horizontal",
                  timeout: float = 3.0) -> LineCrossingDetector:
    return LineCrossingDetector(orientation, LINE, DEAD_ZONE, entry, timeout)


# Points are (x, y).  y=40 is clearly above the line, y=160 clearly below.
ABOVE = (50.0, 40.0)
BELOW = (50.0, 160.0)


# ---------------------------------------------------------- 1. enter
def test_crossing_from_outside_to_inside_is_enter():
    det = make_detector(entry="down")
    assert det.update(1, ABOVE, now=0.0) is None      # first sighting only remembers the side
    assert det.update(1, BELOW, now=0.1) == ENTER


# ----------------------------------------------------------- 2. exit
def test_crossing_from_inside_to_outside_is_exit():
    det = make_detector(entry="down")
    det.update(1, BELOW, now=0.0)
    assert det.update(1, ABOVE, now=0.1) == EXIT


def test_entry_direction_up_reverses_meaning():
    det = make_detector(entry="up")
    det.update(1, BELOW, now=0.0)
    assert det.update(1, ABOVE, now=0.1) == ENTER


def test_vertical_line_with_entry_right():
    det = make_detector(entry="right", orientation="vertical")
    det.update(1, (40.0, 50.0), now=0.0)
    assert det.update(1, (160.0, 50.0), now=0.1) == ENTER
    assert det.update(1, (40.0, 50.0), now=0.2) == EXIT


# ------------------------------------------------ 3. same side
def test_staying_on_the_same_side_counts_nothing():
    det = make_detector()
    results = [det.update(1, (50.0, y), now=i * 0.1)
               for i, y in enumerate([30, 40, 55, 70, 60, 35])]
    assert results == [None] * 6


# --------------------------------------------------- 4. dead zone
def test_staying_inside_dead_zone_counts_nothing():
    det = make_detector()
    det.update(1, ABOVE, now=0.0)
    for i, y in enumerate([85, 100, 115, 119, 100, 82], start=1):
        assert det.update(1, (50.0, y), now=i * 0.1) is None


def test_entering_dead_zone_then_returning_does_not_count():
    det = make_detector()
    det.update(1, ABOVE, now=0.0)
    assert det.update(1, (50.0, 115.0), now=0.1) is None   # deep in the dead zone
    assert det.update(1, (50.0, 60.0), now=0.2) is None    # back where it started


def test_leaving_dead_zone_on_far_side_counts_once():
    det = make_detector()
    det.update(1, ABOVE, now=0.0)
    det.update(1, (50.0, 110.0), now=0.1)                  # dead zone
    assert det.update(1, (50.0, 135.0), now=0.2) == ENTER


# ------------------------------------------ 5. no repeated counts
def test_same_crossing_is_not_counted_twice():
    det = make_detector()
    det.update(1, ABOVE, now=0.0)
    assert det.update(1, BELOW, now=0.1) == ENTER
    # Person keeps walking / standing on the far side:
    for i, y in enumerate([165, 170, 180, 150, 140, 130], start=2):
        assert det.update(1, (50.0, y), now=i * 0.1) is None


def test_jitter_around_the_line_after_crossing_does_not_recount():
    det = make_detector()
    det.update(1, ABOVE, now=0.0)
    assert det.update(1, BELOW, now=0.1) == ENTER
    # Wobbling between "far side" and the dead zone must not count again.
    for i, y in enumerate([118, 125, 119, 130, 100, 121], start=2):
        assert det.update(1, (50.0, y), now=i * 0.1) is None


def test_walking_back_is_a_real_exit():
    det = make_detector()
    det.update(1, ABOVE, now=0.0)
    assert det.update(1, BELOW, now=0.1) == ENTER
    assert det.update(1, ABOVE, now=0.2) == EXIT


def test_rebaseline_after_moving_the_line_prevents_false_count():
    det = make_detector()
    det.update(1, (50.0, 90.0), now=0.0)        # above the line at y=100
    det.set_line(60.0)                          # user drags the line above the person
    det.rebaseline()
    assert det.update(1, (50.0, 90.0), now=0.1) is None   # new baseline, no event


# --------------------------------------- 6. occupancy never negative
def test_exit_at_zero_occupancy_is_ignored():
    state = OccupancyState()
    assert state.apply(EXIT, "t0") is False
    assert state.current == 0 and state.exited == 0

    assert state.apply(ENTER, "t1") is True
    assert state.apply(EXIT, "t2") is True
    assert state.apply(EXIT, "t3") is False
    assert state.current == 0
    assert (state.entered, state.exited) == (1, 1)


def test_occupancy_equals_entered_minus_exited():
    state = OccupancyState()
    for _ in range(4):
        state.apply(ENTER, "t")
    state.apply(EXIT, "t")
    assert state.current == state.entered - state.exited == 3


def test_restoring_inconsistent_totals_cannot_give_negative_occupancy():
    state = OccupancyState(entered=2, exited=9)
    assert state.current == 0


# ------------------------------------------------------ peak value
def test_peak_and_peak_timestamp_are_recorded():
    state = OccupancyState()
    state.apply(ENTER, "10:00:00")
    state.apply(ENTER, "10:05:00")      # peak = 2
    state.apply(EXIT, "10:10:00")
    state.apply(ENTER, "10:15:00")      # back to 2, not a new peak
    assert state.peak == 2
    assert state.peak_timestamp == "10:05:00"
    state.apply(ENTER, "10:20:00")      # peak = 3
    assert (state.peak, state.peak_timestamp) == (3, "10:20:00")


def test_reset_clears_everything():
    state = OccupancyState()
    state.apply(ENTER, "t")
    state.reset()
    assert (state.current, state.entered, state.exited, state.peak) == (0, 0, 0, 0)
    assert state.peak_timestamp is None


# -------------------------------------------- 7. expiring track IDs
def test_old_tracking_ids_are_expired_after_timeout():
    det = make_detector(timeout=3.0)
    det.update(1, ABOVE, now=0.0)
    det.update(2, ABOVE, now=2.0)
    assert det.expire_old_tracks(now=2.5) == []
    assert det.expire_old_tracks(now=3.1) == [1]        # id 1 unseen for > 3 s
    assert det.active_track_count == 1
    assert det.expire_old_tracks(now=6.0) == [2]
    assert det.active_track_count == 0


def test_expired_id_that_returns_starts_fresh_and_is_not_counted():
    det = make_detector(timeout=3.0)
    det.update(1, ABOVE, now=0.0)
    det.expire_old_tracks(now=10.0)
    # Same ID shows up on the other side later: treated as a new track.
    assert det.update(1, BELOW, now=10.1) is None


# ---------------------------------- 8. invalid Arduino serial messages
@pytest.mark.parametrize("message", [
    "", "\n", "COUNT", "COUNT:", "COUNT:\n", "COUNT:abc", "COUNT:-1", "COUNT:1.5",
    "COUNT:12345", "COUNT:5 extra", "COUNT: 5", "count:5", "HELLO:5", "5", ":5",
    "COUNT:５",            # full-width digit: must not be accepted
    "COUNT:²",             # superscript digit: must not be accepted
    "COUNT:5;COUNT:6",
    b"\xff\xfeCOUNT:5", b"COUNT:\x00", None, 42,
])
def test_invalid_serial_messages_are_rejected(message):
    assert parse_count_message(message) is None


@pytest.mark.parametrize("message, expected", [
    ("COUNT:0", 0), ("COUNT:5", 5), ("COUNT:5\n", 5), ("COUNT:12\r\n", 12),
    ("COUNT:9999", 9999), (b"COUNT:7\n", 7),
])
def test_valid_serial_messages_are_parsed(message, expected):
    assert parse_count_message(message) == expected


def test_format_count_message_round_trips_and_rejects_bad_values():
    assert format_count_message(5) == "COUNT:5\n"
    assert parse_count_message(format_count_message(123)) == 123
    for bad in (-1, 10000, 2.5, "5", True):
        with pytest.raises(ValueError):
            format_count_message(bad)


# ------------------------------------------ serial manager behaviour
class FakeSerial:
    def __init__(self, fail_writes: bool = False) -> None:
        self.written: list[bytes] = []
        self.closed = False
        self.in_waiting = 0
        self._fail = fail_writes

    def write(self, data: bytes) -> None:
        if self._fail:
            raise OSError("device disconnected")
        self.written.append(data)

    def read(self, n: int) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


def test_serial_manager_waits_for_arduino_reset_then_sends_and_heartbeats():
    fake = FakeSerial()
    mgr = SerialManager(port="/dev/cu.usbmodemTEST", settle_seconds=2.0,
                        heartbeat_interval=2.0, serial_factory=lambda p, b: fake)
    mgr.update(5, now=0.0)                    # opens the port
    assert mgr.is_connected and fake.written == []
    mgr.update(5, now=1.0)                    # still settling
    assert fake.written == []
    mgr.update(5, now=2.1)                    # first message
    assert fake.written == [b"COUNT:5\n"]
    mgr.update(5, now=2.5)                    # unchanged, no heartbeat yet
    assert len(fake.written) == 1
    mgr.update(5, now=4.2)                    # heartbeat
    assert len(fake.written) == 2
    mgr.update(6, now=4.3)                    # change is sent immediately
    assert fake.written[-1] == b"COUNT:6\n"


def test_serial_write_error_does_not_raise_and_reconnects():
    ports = [FakeSerial(fail_writes=True), FakeSerial()]
    mgr = SerialManager(port="auto", settle_seconds=0.0, reconnect_interval=5.0,
                        serial_factory=lambda p, b: ports.pop(0),
                        port_finder=lambda: ["/dev/cu.usbmodemTEST"])
    mgr.update(3, now=0.0)                    # connect
    mgr.update(3, now=0.1)                    # write fails -> disconnected, no exception
    assert not mgr.is_connected
    mgr.update(3, now=2.0)                    # too early to retry
    assert not mgr.is_connected
    mgr.update(3, now=5.2)                    # reconnects
    assert mgr.is_connected
    mgr.update(3, now=5.3)
    assert mgr.status_text.startswith("connected")


def test_serial_manager_without_arduino_stays_quiet():
    mgr = SerialManager(port="auto", port_finder=lambda: [], serial_factory=None)
    for t in range(0, 20, 5):
        assert mgr.update(2, now=float(t)) == []
    assert not mgr.is_connected


def test_disabled_serial_manager_never_connects():
    mgr = SerialManager(enabled=False)
    mgr.update(1, now=0.0)
    assert mgr.status_text.startswith("disabled")
