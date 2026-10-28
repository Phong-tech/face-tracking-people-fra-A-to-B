"""USB-serial link from the Mac to the Arduino.

Design goals
------------
* NEVER block or crash the camera loop.  Every call is quick, and every
  serial error is caught, logged and turned into "disconnected".
* Reconnect automatically every few seconds.
* Wait a short time after opening the port, because opening it resets
  most Arduino boards.  We do that by *not sending yet* instead of
  sleeping, so the video keeps running.

Protocol (one text line per message)
------------------------------------
    COUNT:<0-9999>\\n
"""

from __future__ import annotations

import glob
import logging
from typing import Any, Callable, Optional, Union

log = logging.getLogger("people_counter.serial")

MAX_COUNT = 9999
MESSAGE_PREFIX = "COUNT:"

# Likely Arduino device names.  usbmodem is what the UNO WiFi Rev2 shows up as on macOS.
PORT_PATTERNS = (
    "/dev/cu.usbmodem*",
    "/dev/cu.usbserial*",
    "/dev/ttyACM*",      # Linux
    "/dev/ttyUSB*",      # Linux
)


# ---------------------------------------------------------------- protocol
def format_count_message(count: int) -> str:
    """Build the line sent to the Arduino, e.g. ``COUNT:5\\n``."""
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_COUNT:
        raise ValueError(f"count must be an integer from 0 to {MAX_COUNT}")
    return f"{MESSAGE_PREFIX}{count}\n"


def parse_count_message(line: Union[str, bytes, None]) -> Optional[int]:
    """Return the count in a valid ``COUNT:n`` line, or None if it is invalid.

    The Arduino sketch applies the same rules, so this function documents
    (and lets us test) exactly what counts as a valid message.
    """
    if line is None:
        return None
    if isinstance(line, bytes):
        try:
            line = line.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(line, str):
        return None

    line = line.rstrip("\r\n")
    if not line.startswith(MESSAGE_PREFIX):
        return None
    digits = line[len(MESSAGE_PREFIX):]
    # isascii() matters: str.isdigit() also accepts digits such as "５" or "²".
    if not (1 <= len(digits) <= 4 and digits.isascii() and digits.isdigit()):
        return None
    return int(digits)


def find_arduino_ports() -> list[str]:
    """Return serial device paths that look like an Arduino, best guess first."""
    ports: list[str] = []
    for pattern in PORT_PATTERNS:
        ports.extend(sorted(glob.glob(pattern)))
    return ports


# ---------------------------------------------------------------- manager
class SerialManager:
    """Keeps (re)connecting to the Arduino and sends COUNT messages."""

    def __init__(
        self,
        port: str = "auto",
        baud_rate: int = 9600,
        enabled: bool = True,
        settle_seconds: float = 2.0,
        heartbeat_interval: float = 2.0,
        reconnect_interval: float = 5.0,
        serial_factory: Optional[Callable[[str, int], Any]] = None,
        port_finder: Callable[[], list[str]] = find_arduino_ports,
    ) -> None:
        self._configured_port = port
        self._baud_rate = baud_rate
        self._enabled = enabled
        self._settle_seconds = settle_seconds
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_interval = reconnect_interval
        self._serial_factory = serial_factory
        self._port_finder = port_finder

        self._serial: Any = None
        self._port_in_use = ""
        self._ready_at = 0.0
        self._next_attempt = 0.0
        self._last_sent_count: Optional[int] = None
        self._last_send_time = 0.0
        self._last_error = ""

    # ------------------------------------------------------------- status
    @property
    def is_connected(self) -> bool:
        return self._serial is not None

    @property
    def status_text(self) -> str:
        """Short text for the on-screen display."""
        if not self._enabled:
            return "disabled (camera-only)"
        if self._serial is not None:
            return f"connected ({self._port_in_use})"
        return "not connected (retrying)"

    # ------------------------------------------------------------ main API
    def update(self, count: int, now: float) -> list[str]:
        """Call once per frame.  Sends COUNT when needed.

        ``now`` is a monotonic clock in seconds.  Returns any text lines the
        Arduino sent back (they are read so its output buffer never fills).
        """
        if not self._enabled:
            return []

        if self._serial is None:
            if now >= self._next_attempt:
                self._try_connect(now)
            return []

        incoming = self._read_incoming()
        if self._serial is None or now < self._ready_at:
            return incoming            # still waiting for the Arduino to reboot

        send_due = (
            count != self._last_sent_count
            or now - self._last_send_time >= self._heartbeat_interval
        )
        if send_due:
            self._send(count, now)
        return incoming

    def close(self) -> None:
        """Close the port.  Safe to call more than once."""
        self._disconnect("closed", now=None)

    # ------------------------------------------------------------ internals
    def _open_port(self, port: str) -> Any:
        if self._serial_factory is not None:
            return self._serial_factory(port, self._baud_rate)
        import serial  # imported here so the rest of the app works without pyserial

        return serial.Serial(port, self._baud_rate, timeout=0, write_timeout=0.5)

    def _choose_port(self) -> Optional[str]:
        if self._configured_port.lower() != "auto":
            return self._configured_port
        ports = self._port_finder()
        return ports[0] if ports else None

    def _try_connect(self, now: float) -> None:
        self._next_attempt = now + self._reconnect_interval
        port = self._choose_port()
        if port is None:
            self._report_error("No Arduino-like serial port found "
                               "(looked for /dev/cu.usbmodem* and /dev/cu.usbserial*)")
            return
        try:
            self._serial = self._open_port(port)
        except ImportError:
            self._report_error("pyserial is not installed; running without Arduino")
            self._enabled = False
            return
        except Exception as exc:  # noqa: BLE001 - serial errors must never stop the camera
            self._serial = None
            self._report_error(f"Could not open {port}: {exc}")
            return

        self._port_in_use = port
        self._ready_at = now + self._settle_seconds   # Arduino reboots when the port opens
        self._last_sent_count = None
        self._last_error = ""
        log.info("Arduino port opened: %s (waiting %.1f s for it to reset)",
                 port, self._settle_seconds)

    def _send(self, count: int, now: float) -> None:
        try:
            message = format_count_message(count)
            self._serial.write(message.encode("ascii"))
            self._last_sent_count = count
            self._last_send_time = now
        except Exception as exc:  # noqa: BLE001
            self._disconnect(f"write failed: {exc}", now)

    def _read_incoming(self) -> list[str]:
        """Drain whatever the Arduino printed, without blocking."""
        lines: list[str] = []
        try:
            waiting = self._serial.in_waiting
            if waiting:
                data = self._serial.read(waiting)
                text = data.decode("ascii", errors="replace")
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        except Exception as exc:  # noqa: BLE001
            self._disconnect(f"read failed: {exc}", None)
        return lines

    def _disconnect(self, reason: str, now: Optional[float]) -> None:
        serial_obj, self._serial = self._serial, None
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:  # noqa: BLE001
                pass
            if reason != "closed":
                self._report_error(f"Arduino connection lost ({reason})")
        if now is not None:
            self._next_attempt = now + self._reconnect_interval

    def _report_error(self, message: str) -> None:
        """Log an error once, not on every retry."""
        if message != self._last_error:
            log.warning("%s", message)
            self._last_error = message
