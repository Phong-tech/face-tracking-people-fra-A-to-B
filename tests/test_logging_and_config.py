"""Tests for CSV logging, restoring state, config validation and the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import generate_report
from src.config import ConfigError, config_from_dict, load_config
from src.counter import ENTER, EXIT
from src.logger import OccupancyLogger, read_csv_rows


def test_logger_creates_files_with_headers(tmp_path: Path):
    OccupancyLogger(tmp_path, 60)
    assert (tmp_path / "occupancy_log.csv").read_text().splitlines()[0] == (
        "timestamp,current_occupancy,total_entered,total_exited,peak_occupancy,peak_timestamp")
    assert (tmp_path / "crossing_events.csv").read_text().splitlines()[0] == (
        "timestamp,tracking_id,direction,current_occupancy")


def test_state_is_restored_after_restart_and_logs_are_preserved(tmp_path: Path):
    first = OccupancyLogger(tmp_path, 60)
    first.log_crossing("2026-10-28 10:00:00", 1, ENTER, 1)
    first.log_crossing("2026-10-28 10:01:00", 2, ENTER, 2)
    first.log_crossing("2026-10-28 10:02:00", 1, EXIT, 1)

    second = OccupancyLogger(tmp_path, 60)       # "restart"
    state = second.restore_state()
    assert (state.entered, state.exited, state.current) == (2, 1, 1)
    assert (state.peak, state.peak_timestamp) == (2, "2026-10-28 10:01:00")
    assert len(read_csv_rows(tmp_path / "crossing_events.csv")) == 3   # nothing overwritten


def test_reset_marker_survives_restart(tmp_path: Path):
    log = OccupancyLogger(tmp_path, 60)
    log.log_crossing("2026-10-28 10:00:00", 1, ENTER, 1)
    log.log_reset("2026-10-28 10:05:00")
    assert OccupancyLogger(tmp_path, 60).restore_state().current == 0


def test_periodic_logging_respects_interval(tmp_path: Path):
    log = OccupancyLogger(tmp_path, 60)
    state = log.restore_state()
    assert log.maybe_log_occupancy(0.0, "2026-10-28 10:00:00", state) is True
    assert log.maybe_log_occupancy(30.0, "2026-10-28 10:00:30", state) is False
    assert log.maybe_log_occupancy(61.0, "2026-10-28 10:01:01", state) is True
    assert len(read_csv_rows(tmp_path / "occupancy_log.csv")) == 2


# ---------------------------------------------------------------- config
def test_default_config_file_is_valid():
    cfg = load_config(Path(__file__).resolve().parent.parent / "config.json")
    assert cfg.camera_index == 0 and cfg.max_capacity >= cfg.medium_occupancy_threshold


@pytest.mark.parametrize("bad", [
    {"detection_confidence": 1.5},
    {"line_orientation": "diagonal"},
    {"line_orientation": "horizontal", "entry_direction": "left"},
    {"low_occupancy_threshold": 10, "medium_occupancy_threshold": 5},
    {"medium_occupancy_threshold": 20, "max_capacity": 15},
    {"camera_index": "0"},
    {"camera_index": True},
    {"baud_rate": 1234},
    {"not_a_setting": 1},
    {"log_interval_seconds": 0},
])
def test_invalid_config_values_are_rejected(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_missing_config_file_falls_back_to_defaults(tmp_path: Path):
    assert load_config(tmp_path / "nope.json").camera_index == 0


def test_broken_json_gives_clear_error(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{ not json")
    with pytest.raises(ConfigError):
        load_config(path)


# ---------------------------------------------------------------- report
def test_report_builds_hourly_summary_and_busiest_hour(tmp_path: Path, capsys):
    log = OccupancyLogger(tmp_path, 60)
    log.log_crossing("2026-10-28 09:10:00", 1, ENTER, 1)
    log.log_crossing("2026-10-28 14:10:00", 2, ENTER, 2)
    log.log_crossing("2026-10-28 14:20:00", 3, ENTER, 3)
    log.log_crossing("2026-10-28 15:00:00", 2, EXIT, 2)
    with (tmp_path / "occupancy_log.csv").open("a") as handle:
        handle.write("2026-10-28 09:30:00,1,1,0,1,2026-10-28 09:10:00\n")
        handle.write("2026-10-28 14:30:00,3,3,0,3,2026-10-28 14:20:00\n")
        handle.write("2026-10-28 14:45:00,3,3,0,3,2026-10-28 14:20:00\n")

    assert generate_report.generate_report(tmp_path, chart=False) == 0
    out = capsys.readouterr().out
    assert "Total entries        : 3" in out
    assert "Total exits          : 1" in out
    assert "Highest occupancy    : 3" in out
    assert "2026-10-28 14:20:00" in out
    assert "14:00-14:59" in out.split("Busiest hour")[1].splitlines()[0]

    rows = read_csv_rows(tmp_path / "hourly_summary.csv")
    by_hour = {r["hour"]: r for r in rows}
    assert float(by_hour["14:00-14:59"]["average_occupancy"]) == 3.0
    assert by_hour["14:00-14:59"]["entries"] == "2"
    assert by_hour["15:00-15:59"]["exits"] == "1"


def test_report_with_no_data_does_not_crash(tmp_path: Path):
    assert generate_report.generate_report(tmp_path, chart=False) == 1


def test_report_writes_chart_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    log = OccupancyLogger(tmp_path, 60)
    log.log_crossing("2026-10-28 09:10:00", 1, ENTER, 1)
    log.log_crossing("2026-10-28 09:20:00", 2, ENTER, 2)
    assert generate_report.generate_report(tmp_path, chart=True) == 0
    assert (tmp_path / "occupancy_chart.png").stat().st_size > 1000
