"""Drawing helpers for the live camera window (OpenCV).

Everything here only draws onto the frame that is shown on screen.
Frames are never written to disk.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# Colours are BGR (OpenCV's order).
GREEN = (80, 200, 60)
BLUE = (230, 140, 40)
RED = (60, 60, 240)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
GREY = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX

_DIRECTION_VECTORS = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}


def occupancy_colour(count: int, low: int, medium: int) -> tuple[int, int, int]:
    """Same colour rule as the Arduino LEDs: green / blue / red."""
    if count < low:
        return GREEN
    if count < medium:
        return BLUE
    return RED


def _put_text(frame: np.ndarray, text: str, org: tuple[int, int],
              scale: float = 0.6, colour: tuple[int, int, int] = WHITE,
              thickness: int = 1) -> None:
    # A dark outline keeps the text readable on any background.
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def _centred_text(frame: np.ndarray, text: str, centre: tuple[int, int],
                  colour: tuple[int, int, int]) -> None:
    (width, height), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    _put_text(frame, text, (centre[0] - width // 2, centre[1] + height // 2),
              scale=0.7, colour=colour, thickness=2)


def draw_counting_line(
    frame: np.ndarray, orientation: str, position_px: int,
    dead_zone_px: int, entry_direction: str,
) -> None:
    """Draw the counting line, its dead zone and ENTER / EXIT arrows."""
    height, width = frame.shape[:2]
    dx, dy = _DIRECTION_VECTORS[entry_direction]
    arrow_half = 45

    if orientation == "horizontal":
        cv2.line(frame, (0, position_px), (width, position_px), YELLOW, 2)
        for offset in (-dead_zone_px, dead_zone_px):          # dead-zone edges
            y = position_px + offset
            for x in range(0, width, 24):
                cv2.line(frame, (x, y), (x + 10, y), GREY, 1)
        anchors = {"ENTER": (int(width * 0.25), position_px),
                   "EXIT": (int(width * 0.75), position_px)}
    else:
        cv2.line(frame, (position_px, 0), (position_px, height), YELLOW, 2)
        for offset in (-dead_zone_px, dead_zone_px):
            x = position_px + offset
            for y in range(0, height, 24):
                cv2.line(frame, (x, y), (x, y + 10), GREY, 1)
        anchors = {"ENTER": (position_px, int(height * 0.25)),
                   "EXIT": (position_px, int(height * 0.75))}

    for label, (ax, ay) in anchors.items():
        sign = 1 if label == "ENTER" else -1       # EXIT points the opposite way
        vx, vy = dx * sign, dy * sign
        start = (ax - vx * arrow_half, ay - vy * arrow_half)
        tip = (ax + vx * arrow_half, ay + vy * arrow_half)
        colour = GREEN if label == "ENTER" else RED
        cv2.arrowedLine(frame, start, tip, colour, 4, tipLength=0.35)
        _centred_text(frame, label, (tip[0] + vx * 40, tip[1] + vy * 20), colour)


def draw_person(
    frame: np.ndarray, track_id: int,
    box: tuple[float, float, float, float], point: tuple[float, float],
) -> None:
    """Bounding box, temporary tracking ID and the point used for counting."""
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)
    _put_text(frame, f"ID {track_id}", (x1, max(18, y1 - 8)), colour=WHITE)
    cv2.circle(frame, (int(point[0]), int(point[1])), 6, YELLOW, -1)


def draw_hud(frame: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
    """Semi-transparent information panel in the top-left corner."""
    line_height = 26
    panel_width = 430
    panel_height = 14 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, (text, colour) in enumerate(lines):
        _put_text(frame, text, (18, 32 + index * line_height), colour=colour)


def draw_banner(frame: np.ndarray, text: str, colour: tuple[int, int, int] = YELLOW) -> None:
    """Large message at the bottom of the frame (e.g. the reset confirmation)."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - 46), (width, height), (0, 0, 0), -1)
    _put_text(frame, text, (16, height - 16), scale=0.7, colour=colour, thickness=2)
