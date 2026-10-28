"""Configuration loading and validation for the people counter.

All tunable values live in ``config.json``.  This module turns that file
into a typed :class:`AppConfig` and rejects bad values early with a clear
message, instead of failing later in the middle of a camera loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# Which entry directions make sense for each line orientation.
#   horizontal line -> people move up/down across it
#   vertical line   -> people move left/right across it
VALID_ENTRY_DIRECTIONS = {
    "horizontal": ("down", "up"),
    "vertical": ("right", "left"),
}


class ConfigError(ValueError):
    """Raised when config.json is missing a valid value."""


@dataclass
class AppConfig:
    """Every user-adjustable setting, with safe defaults."""

    # --- Camera -----------------------------------------------------------
    camera_index: int = 0
    frame_width: int = 1280          # requested size; the camera may choose another
    frame_height: int = 720
    show_window: bool = True

    # --- Detection / tracking (YOLO) --------------------------------------
    model_path: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"
    detection_confidence: float = 0.4
    inference_image_size: int = 640
    device: str = "auto"             # "auto", "cpu" or "mps" (Apple Silicon GPU)

    # --- Counting line ----------------------------------------------------
    line_orientation: str = "horizontal"   # "horizontal" or "vertical"
    line_position: float = 0.5             # fraction of frame height/width (0-1)
    entry_direction: str = "down"          # horizontal: down/up, vertical: right/left
    dead_zone_fraction: float = 0.05       # half-width of the dead band, as a fraction
    reference_point_ratio: float = 0.85    # 0 = top of box, 1 = bottom (feet)
    tracking_timeout_seconds: float = 3.0
    line_step: float = 0.02                # how far W/A/S/D move the line

    # --- Arduino / serial -------------------------------------------------
    serial_enabled: bool = True
    serial_port: str = "auto"
    baud_rate: int = 9600
    serial_settle_seconds: float = 2.0     # wait after opening (Arduino resets)
    heartbeat_interval_seconds: float = 2.0
    reconnect_interval_seconds: float = 5.0

    # --- Occupancy thresholds ---------------------------------------------
    low_occupancy_threshold: int = 5       # below this: green
    medium_occupancy_threshold: int = 10   # at/above this: red (blue in between)
    max_capacity: int = 15

    # --- Logging ----------------------------------------------------------
    log_interval_seconds: float = 60.0
    data_dir: str = "data"                 # relative paths are relative to the project

    def resolved_data_dir(self, project_root: Path) -> Path:
        """Return the data folder as an absolute path."""
        path = Path(self.data_dir).expanduser()
        return path if path.is_absolute() else project_root / path


def _is_int(value: Any) -> bool:
    # bool is a subclass of int in Python, but "true" is not a valid camera index.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(cfg: AppConfig) -> None:
    """Raise :class:`ConfigError` if any value is unusable."""

    def need(condition: bool, message: str) -> None:
        if not condition:
            raise ConfigError(message)

    need(cfg.camera_index >= 0, "camera_index must be 0 or higher")
    need(cfg.frame_width >= 160 and cfg.frame_height >= 120,
         "frame_width/frame_height must be at least 160x120")
    need(0.0 < cfg.detection_confidence < 1.0,
         "detection_confidence must be between 0 and 1 (exclusive)")
    need(cfg.inference_image_size >= 32, "inference_image_size must be at least 32")
    need(cfg.device in ("auto", "cpu", "mps") or cfg.device.startswith("cuda"),
         "device must be 'auto', 'cpu', 'mps' or a CUDA device such as 'cuda:0'")

    need(cfg.line_orientation in VALID_ENTRY_DIRECTIONS,
         "line_orientation must be 'horizontal' or 'vertical'")
    need(0.05 <= cfg.line_position <= 0.95,
         "line_position must be between 0.05 and 0.95")
    need(cfg.entry_direction in VALID_ENTRY_DIRECTIONS[cfg.line_orientation],
         f"entry_direction must be one of "
         f"{VALID_ENTRY_DIRECTIONS[cfg.line_orientation]} for a "
         f"{cfg.line_orientation} line")
    need(0.0 <= cfg.dead_zone_fraction <= 0.25,
         "dead_zone_fraction must be between 0 and 0.25")
    need(0.5 <= cfg.reference_point_ratio <= 1.0,
         "reference_point_ratio must be between 0.5 and 1.0")
    need(cfg.tracking_timeout_seconds > 0, "tracking_timeout_seconds must be above 0")
    need(0.001 <= cfg.line_step <= 0.2, "line_step must be between 0.001 and 0.2")

    need(cfg.baud_rate in (300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200),
         "baud_rate must be a standard rate such as 9600")
    need(cfg.serial_port.strip() != "", "serial_port must be 'auto' or a port path")
    need(cfg.serial_settle_seconds >= 0, "serial_settle_seconds must be 0 or higher")
    need(cfg.heartbeat_interval_seconds > 0, "heartbeat_interval_seconds must be above 0")
    need(cfg.reconnect_interval_seconds > 0, "reconnect_interval_seconds must be above 0")

    need(cfg.low_occupancy_threshold >= 1, "low_occupancy_threshold must be at least 1")
    need(cfg.low_occupancy_threshold < cfg.medium_occupancy_threshold,
         "low_occupancy_threshold must be smaller than medium_occupancy_threshold")
    need(cfg.medium_occupancy_threshold <= cfg.max_capacity,
         "medium_occupancy_threshold must not exceed max_capacity")
    need(cfg.max_capacity <= 9999, "max_capacity must be 9999 or lower")
    need(cfg.log_interval_seconds >= 1, "log_interval_seconds must be at least 1")


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    """Build and validate an AppConfig from a plain dictionary."""
    defaults = AppConfig()
    known = {f.name for f in fields(AppConfig)}

    unknown = sorted(set(raw) - known)
    if unknown:
        # A typo such as "dead_zone" would otherwise be silently ignored.
        raise ConfigError(f"Unknown setting(s) in config: {', '.join(unknown)}")

    values: dict[str, Any] = {}
    for name, value in raw.items():
        default = getattr(defaults, name)
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int):
            ok = _is_int(value)
        elif isinstance(default, float):
            ok = _is_number(value)
        else:
            ok = isinstance(value, str)
        if not ok:
            raise ConfigError(
                f"Setting '{name}' has the wrong type "
                f"(expected {type(default).__name__}, got {type(value).__name__})"
            )
        values[name] = float(value) if isinstance(default, float) else value

    cfg = AppConfig(**values)
    validate_config(cfg)
    return cfg


def load_config(path: Path) -> AppConfig:
    """Load config.json.  A missing file means "use the defaults"."""
    if not path.exists():
        return config_from_dict({})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain a JSON object")
    return config_from_dict(raw)
