#!/usr/bin/env python3
"""Anonymous people counter for a MacBook camera + Arduino UNO WiFi Rev2.

What it does
------------
1. Opens the camera and runs YOLO (person class only) with tracking.
2. Gives every person a temporary tracking ID for the current session.
3. Counts a person as ENTER or EXIT when their foot-level reference point
   clearly crosses a virtual line (see src/counter.py).
4. Shows live numbers, logs to CSV and sends ``COUNT:n`` to the Arduino.

Privacy: processing is local, there is no face recognition, and no frame
is ever saved to disk (this file contains no image-writing calls).

Run:  python3 people_counter.py            (see README.md for setup)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import cv2

from src.config import AppConfig, ConfigError, load_config
from src.counter import LineCrossingDetector, OccupancyState
from src.logger import OccupancyLogger, now_string
from src.overlay import (GREEN, GREY, RED, WHITE, YELLOW, draw_banner,
                         draw_counting_line, draw_hud, draw_person,
                         occupancy_colour)
from src.serial_manager import SerialManager

PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "People Counter  (Q/Esc quit | R reset | W/S or A/D move line | F flip)"
RESET_CONFIRM_SECONDS = 5.0
MAX_CAMERA_FAILURES = 30        # consecutive failed reads before reconnecting
MAX_INFERENCE_FAILURES = 10     # consecutive model errors before giving up

log = logging.getLogger("people_counter")


# ----------------------------------------------------------------- errors
class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


class ModelError(RuntimeError):
    """The YOLO model could not be loaded."""


# ------------------------------------------------------------- detections
@dataclass(frozen=True)
class Detection:
    """One tracked person in one frame."""

    track_id: int                                  # temporary, NOT an identity
    box: tuple[float, float, float, float]         # x1, y1, x2, y2 in pixels
    confidence: float


class PersonTracker(Protocol):
    def track(self, frame: Any) -> list[Detection]: ...


class YoloPersonTracker:
    """Ultralytics YOLO restricted to the 'person' class, with persistent IDs."""

    PERSON_CLASS_ID = 0

    def __init__(self, cfg: AppConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError(
                "The 'ultralytics' package is not installed. Run: "
                "python3 -m pip install -r requirements.txt"
            ) from exc
        try:
            # The first run downloads the weights (internet needed once).
            self._model = YOLO(cfg.model_path)
        except Exception as exc:  # noqa: BLE001
            raise ModelError(
                f"Could not load YOLO model '{cfg.model_path}': {exc}\n"
                "Check your internet connection (needed for the first download) "
                "or set model_path in config.json to a local .pt file."
            ) from exc
        self._cfg = cfg

    def track(self, frame: Any) -> list[Detection]:
        kwargs: dict[str, Any] = dict(
            persist=True,                         # keep IDs between frames
            classes=[self.PERSON_CLASS_ID],       # people only
            conf=self._cfg.detection_confidence,
            imgsz=self._cfg.inference_image_size,
            tracker=self._cfg.tracker,
            verbose=False,
        )
        if self._cfg.device != "auto":
            kwargs["device"] = self._cfg.device
        results = self._model.track(frame, **kwargs)
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:     # no confirmed tracks this frame
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        return [
            Detection(int(i), (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c))
            for i, b, c in zip(ids, xyxy, confidences)
        ]


# ---------------------------------------------------------------- camera
def open_camera(index: int, width: int, height: int) -> "cv2.VideoCapture":
    """Open the camera, preferring Apple's AVFoundation backend on macOS."""
    backends: list[tuple[str, int]] = []
    if sys.platform == "darwin":
        backends.append(("AVFoundation", cv2.CAP_AVFOUNDATION))
    backends.append(("default", cv2.CAP_ANY))

    for name, api in backends:
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # The first frames can fail while macOS shows the permission prompt
        # or the sensor warms up, so try for a few seconds.
        for _ in range(40):
            ok, frame = cap.read()
            if ok and frame is not None:
                log.info("Camera %d opened with %s backend (%dx%d)",
                         index, name, frame.shape[1], frame.shape[0])
                return cap
            time.sleep(0.1)
        cap.release()

    raise CameraError(
        f"Could not open camera {index}. On macOS, allow Camera access for your "
        "Terminal/IDE in System Settings > Privacy & Security > Camera, close "
        "other apps using the camera, or try another camera_index in config.json."
    )


def reference_point(box: tuple[float, float, float, float], ratio: float) -> tuple[float, float]:
    """Point used for counting: horizontal centre, near the feet.

    ratio 0.0 = top of the box, 1.0 = bottom.  Using the lower part
    approximates where the person stands on the floor.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * ratio)


# ------------------------------------------------------------------- app
class PeopleCounterApp:
    """Ties camera, tracker, counting logic, logging and Arduino together."""

    def __init__(
        self,
        cfg: AppConfig,
        tracker: PersonTracker,
        serial_manager: SerialManager,
        logger: OccupancyLogger,
        camera: Optional["cv2.VideoCapture"] = None,
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.serial = serial_manager
        self.logger = logger
        self.camera = camera

        self.state: OccupancyState = logger.restore_state()
        self.line_position = cfg.line_position     # fraction; W/S/A/D change it live
        self.detector = LineCrossingDetector(
            cfg.line_orientation, 0.0, 0.0, cfg.entry_direction, cfg.tracking_timeout_seconds)

        self._reset_deadline: Optional[float] = None
        self._fps = 0.0
        self._last_frame_time = time.monotonic()
        self._camera_failures = 0
        self._inference_failures = 0

    # ----------------------------------------------------------- main loop
    def run(self) -> int:
        """Run until the user quits.  Returns a process exit code."""
        log.info("Restored state: occupancy=%d entered=%d exited=%d peak=%d",
                 self.state.current, self.state.entered, self.state.exited, self.state.peak)
        if self.cfg.show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        exit_code = 0
        try:
            if self.camera is None:
                self.camera = open_camera(
                    self.cfg.camera_index, self.cfg.frame_width, self.cfg.frame_height)
            log.info("Counting started. Press Q or Esc in the window to quit.")
            while True:
                frame = self._read_frame()
                if frame is None:
                    continue
                if not self._process_frame(frame):
                    break
        except KeyboardInterrupt:
            log.info("Interrupted from the keyboard.")
        except (CameraError, RuntimeError) as exc:
            log.error("%s", exc)
            exit_code = 1
        finally:
            self._shutdown()
        return exit_code

    # ------------------------------------------------------------- frames
    def _read_frame(self) -> Optional[Any]:
        """Read one frame; recover from short camera glitches."""
        ok, frame = self.camera.read() if self.camera is not None else (False, None)
        if ok and frame is not None:
            self._camera_failures = 0
            return frame

        self._camera_failures += 1
        if self._camera_failures < MAX_CAMERA_FAILURES:
            time.sleep(0.03)
            return None
        # Too many failures in a row: try to reopen the camera once.
        log.warning("Camera stopped delivering frames; trying to reopen it...")
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self._camera_failures = 0
        self.camera = open_camera(
            self.cfg.camera_index, self.cfg.frame_width, self.cfg.frame_height)
        return None

    def _process_frame(self, frame: Any) -> bool:
        """Handle one frame.  Returns False when the user wants to quit."""
        now = time.monotonic()
        height, width = frame.shape[:2]
        self._update_fps(now)

        # Line position in pixels (the config stores it as a fraction).
        span = height if self.cfg.line_orientation == "horizontal" else width
        line_px = int(self.line_position * span)
        dead_zone_px = int(self.cfg.dead_zone_fraction * span)
        self.detector.set_line(line_px, dead_zone_px)

        detections = self._detect(frame)
        points: dict[int, tuple[float, float]] = {}
        for det in detections:
            point = reference_point(det.box, self.cfg.reference_point_ratio)
            points[det.track_id] = point
            direction = self.detector.update(det.track_id, point, now)
            if direction is not None:
                self._handle_crossing(det.track_id, direction)
        self.detector.expire_old_tracks(now)

        # Arduino + periodic logging never block the loop.
        for message in self.serial.update(self.state.current, now):
            log.info("Arduino: %s", message)
        self.logger.maybe_log_occupancy(now, now_string(), self.state)

        if self.cfg.show_window:
            self._draw(frame, detections, points, line_px, dead_zone_px, now)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return False                       # window closed with the mouse
            return self._handle_key(key, now)
        return True

    def _detect(self, frame: Any) -> list[Detection]:
        try:
            detections = self.tracker.track(frame)
            self._inference_failures = 0
            return detections
        except Exception as exc:  # noqa: BLE001
            self._inference_failures += 1
            log.warning("Detection failed on a frame: %s", exc)
            if self._inference_failures >= MAX_INFERENCE_FAILURES:
                raise RuntimeError("Detection keeps failing; stopping.") from exc
            return []

    # ------------------------------------------------------------ counting
    def _handle_crossing(self, track_id: int, direction: str) -> None:
        timestamp = now_string()
        if self.state.apply(direction, timestamp):
            self.logger.log_crossing(timestamp, track_id, direction, self.state.current)
            log.info("%s (track %d) -> occupancy %d | entered %d, exited %d | peak %d",
                     direction, track_id, self.state.current,
                     self.state.entered, self.state.exited, self.state.peak)
        else:
            log.info("Ignored %s (track %d): occupancy is already 0", direction, track_id)

    # ------------------------------------------------------------ keyboard
    def _handle_key(self, key: int, now: float) -> bool:
        """React to a key press.  Returns False to quit."""
        if key == 255:                                  # no key pressed
            if self._reset_deadline is not None and now > self._reset_deadline:
                self._reset_deadline = None             # confirmation timed out
                log.info("Reset not confirmed in time; cancelled.")
            return True

        char = chr(key).lower() if 32 <= key < 127 else ""

        # Reset needs two steps: R, then Y.  Any other key cancels.
        if self._reset_deadline is not None:
            self._reset_deadline = None
            if char == "y":
                self._reset_counters()
                return True
            log.info("Reset cancelled.")
            if key == 27 or char == "q":
                return False
            return True

        if key == 27 or char == "q":
            return False
        if char == "r":
            self._reset_deadline = now + RESET_CONFIRM_SECONDS
            log.info("Reset requested. Press Y within %.0f s to confirm.", RESET_CONFIRM_SECONDS)
        elif char == "f":
            flipped = {"down": "up", "up": "down", "right": "left", "left": "right"}
            self.detector.set_entry_direction(flipped[self.detector.entry_direction])
            self.detector.rebaseline()
            log.info("Entry direction is now '%s'. (Set entry_direction in "
                     "config.json to keep it.)", self.detector.entry_direction)
        elif char in ("w", "s") and self.cfg.line_orientation == "horizontal":
            self._move_line(-self.cfg.line_step if char == "w" else self.cfg.line_step)
        elif char in ("a", "d") and self.cfg.line_orientation == "vertical":
            self._move_line(-self.cfg.line_step if char == "a" else self.cfg.line_step)
        return True

    def _move_line(self, delta: float) -> None:
        self.line_position = min(0.95, max(0.05, self.line_position + delta))
        self.detector.rebaseline()      # avoid false counts caused by the move itself
        log.info("Line position: %.2f (copy into config.json to keep it)", self.line_position)

    def _reset_counters(self) -> None:
        self.state.reset()
        self.logger.log_reset(now_string())
        self.logger.log_occupancy(now_string(), self.state)
        log.info("Counters reset to zero.")

    # ------------------------------------------------------------- drawing
    def _update_fps(self, now: float) -> None:
        elapsed = now - self._last_frame_time
        self._last_frame_time = now
        if elapsed > 0:
            instant = 1.0 / elapsed
            self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant

    def _draw(self, frame: Any, detections: list[Detection],
              points: dict[int, tuple[float, float]], line_px: int,
              dead_zone_px: int, now: float) -> None:
        draw_counting_line(frame, self.cfg.line_orientation, line_px,
                           dead_zone_px, self.detector.entry_direction)
        for det in detections:
            draw_person(frame, det.track_id, det.box, points[det.track_id])

        state, cfg = self.state, self.cfg
        colour = occupancy_colour(state.current, cfg.low_occupancy_threshold,
                                  cfg.medium_occupancy_threshold)
        arduino_colour = GREEN if self.serial.is_connected else GREY
        draw_hud(frame, [
            (f"Occupancy: {state.current} / {cfg.max_capacity}", colour),
            (f"Entered: {state.entered}    Exited: {state.exited}", WHITE),
            (f"Peak: {state.peak}", WHITE),
            (f"Peak time: {state.peak_timestamp or '-'}", WHITE),
            (f"Arduino: {self.serial.status_text}", arduino_colour),
            (f"FPS: {self._fps:.1f}   Line: {cfg.line_orientation} {self.line_position:.2f}", GREY),
        ])
        if state.current >= cfg.max_capacity:
            draw_banner(frame, "MAXIMUM CAPACITY REACHED", RED)
        if self._reset_deadline is not None:
            remaining = max(0.0, self._reset_deadline - now)
            draw_banner(frame, f"Reset counters? Press Y to confirm "
                               f"(any other key cancels) - {remaining:.0f}s", YELLOW)

    # ------------------------------------------------------------ shutdown
    def _shutdown(self) -> None:
        """Release everything, even after an error."""
        try:
            self.logger.log_occupancy(now_string(), self.state)   # final snapshot
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not write final log row: %s", exc)
        if self.camera is not None:
            self.camera.release()
        self.serial.close()
        if self.cfg.show_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        log.info("Stopped. Final occupancy: %d, peak: %d (%s)",
                 self.state.current, self.state.peak, self.state.peak_timestamp or "-")


# ------------------------------------------------------------------ main
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymous people counter")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json",
                        help="path to the settings file (default: config.json)")
    parser.add_argument("--no-arduino", action="store_true",
                        help="run in camera-only mode without the Arduino")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index (overrides config.json)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        log.error("Configuration problem: %s", exc)
        return 2
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.no_arduino:
        cfg.serial_enabled = False

    try:
        tracker = YoloPersonTracker(cfg)
    except ModelError as exc:
        log.error("%s", exc)
        return 3

    serial_manager = SerialManager(
        port=cfg.serial_port, baud_rate=cfg.baud_rate, enabled=cfg.serial_enabled,
        settle_seconds=cfg.serial_settle_seconds,
        heartbeat_interval=cfg.heartbeat_interval_seconds,
        reconnect_interval=cfg.reconnect_interval_seconds)
    data_dir = cfg.resolved_data_dir(PROJECT_ROOT)
    logger = OccupancyLogger(data_dir, cfg.log_interval_seconds)
    log.info("Data folder: %s", data_dir)

    app = PeopleCounterApp(cfg, tracker, serial_manager, logger)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
