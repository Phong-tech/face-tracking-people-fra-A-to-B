#!/usr/bin/env python3
"""Statistics report for the anonymous people counter.

Reads data/crossing_events.csv and data/occupancy_log.csv, prints a summary,
writes data/hourly_summary.csv and (if matplotlib is installed) a PNG chart.
It never touches the camera and can be run at any time, also while
people_counter.py is running.

Usage:  python3 generate_report.py [--data-dir PATH] [--no-chart]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.counter import ENTER, EXIT
from src.logger import ReplayResult, parse_timestamp, read_csv_rows, replay_events

PROJECT_ROOT = Path(__file__).resolve().parent
HOURLY_HEADER = ["hour", "samples", "average_occupancy", "max_occupancy", "entries", "exits"]


def _hour_label(hour: int) -> str:
    return f"{hour:02d}:00-{hour:02d}:59"


def build_hourly_summary(
    occupancy_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Group the data by hour of day (0-23) and return one row per hour with data.

    * average/max occupancy come from the periodic observations
    * entries/exits are counted from the crossing events
    """
    samples: dict[int, list[int]] = defaultdict(list)
    for row in occupancy_rows:
        moment = parse_timestamp(row.get("timestamp", ""))
        try:
            value = int(row.get("current_occupancy", ""))
        except (TypeError, ValueError):
            continue
        if moment is not None:
            samples[moment.hour].append(value)

    entries: dict[int, int] = defaultdict(int)
    exits: dict[int, int] = defaultdict(int)
    for row in event_rows:
        moment = parse_timestamp(row.get("timestamp", ""))
        direction = (row.get("direction") or "").strip().upper()
        if moment is None:
            continue
        if direction == ENTER:
            entries[moment.hour] += 1
        elif direction == EXIT:
            exits[moment.hour] += 1

    summary: list[dict[str, object]] = []
    for hour in sorted(set(samples) | set(entries) | set(exits)):
        values = samples.get(hour, [])
        summary.append({
            "hour": _hour_label(hour),
            "samples": len(values),
            "average_occupancy": round(sum(values) / len(values), 2) if values else "",
            "max_occupancy": max(values) if values else "",
            "entries": entries.get(hour, 0),
            "exits": exits.get(hour, 0),
        })
    return summary


def find_busiest_hour(summary: list[dict[str, object]]) -> Optional[tuple[str, str]]:
    """Return (hour label, explanation).  Highest average occupancy wins;
    if there are no periodic observations, the hour with most entries is used."""
    with_average = [r for r in summary if r["average_occupancy"] != ""]
    if with_average:
        best = max(with_average, key=lambda r: float(r["average_occupancy"]))
        return str(best["hour"]), f"highest average occupancy ({best['average_occupancy']})"
    with_entries = [r for r in summary if int(r["entries"]) > 0]
    if with_entries:
        best = max(with_entries, key=lambda r: int(r["entries"]))
        return str(best["hour"]), f"most entries ({best['entries']}); no periodic samples available"
    return None


def write_hourly_summary(path: Path, summary: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HOURLY_HEADER)
        writer.writeheader()
        writer.writerows(summary)


def make_chart(path: Path, event_rows: list[dict[str, str]],
               occupancy_rows: list[dict[str, str]], result: ReplayResult) -> bool:
    """Save a PNG of occupancy over time.  Returns False if it cannot be drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")              # no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib is not installed - skipping the chart)")
        return False

    timeline: list[tuple[str, int]] = []
    replay_events(event_rows, timeline)
    step_times, step_values = [], []
    for text, value in timeline:
        moment = parse_timestamp(text)
        if moment is not None:
            step_times.append(moment)
            step_values.append(value)

    sample_times, sample_values = [], []
    for row in occupancy_rows:
        moment = parse_timestamp(row.get("timestamp", ""))
        try:
            value = int(row.get("current_occupancy", ""))
        except (TypeError, ValueError):
            continue
        if moment is not None:
            sample_times.append(moment)
            sample_values.append(value)

    if not step_times and not sample_times:
        print("  (no data to chart yet)")
        return False

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if step_times:
        ax.step(step_times, step_values, where="post", label="After each crossing")
    if sample_times:
        ax.plot(sample_times, sample_values, "o", markersize=3, alpha=0.6,
                label="Periodic observation")
    peak_moment = parse_timestamp(result.lifetime_peak_timestamp or "")
    if peak_moment is not None:
        ax.annotate(f"Peak: {result.lifetime_peak}", (peak_moment, result.lifetime_peak),
                    textcoords="offset points", xytext=(10, 10),
                    arrowprops={"arrowstyle": "->"})
    ax.set_title("Occupancy over time")
    ax.set_xlabel("Local time")
    ax.set_ylabel("People inside")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def generate_report(data_dir: Path, chart: bool = True) -> int:
    events_path = data_dir / "crossing_events.csv"
    occupancy_path = data_dir / "occupancy_log.csv"
    event_rows = read_csv_rows(events_path)
    occupancy_rows = read_csv_rows(occupancy_path)

    if not event_rows and not occupancy_rows:
        print(f"No data found in {data_dir}. Run people_counter.py first.")
        return 1

    result = replay_events(event_rows)
    final_occupancy = result.current
    if not event_rows and occupancy_rows:            # fall back to the last snapshot
        try:
            final_occupancy = int(occupancy_rows[-1]["current_occupancy"])
        except (KeyError, ValueError):
            pass

    summary = build_hourly_summary(occupancy_rows, event_rows)
    summary_path = data_dir / "hourly_summary.csv"
    write_hourly_summary(summary_path, summary)

    print("=" * 56)
    print(" PEOPLE COUNTER REPORT")
    print("=" * 56)
    print(f" Data folder          : {data_dir}")
    print(f" Total entries        : {result.lifetime_entered}")
    print(f" Total exits          : {result.lifetime_exited}")
    print(f" Final occupancy      : {final_occupancy}")
    print(f" Highest occupancy    : {result.lifetime_peak}")
    print(f" Peak occurred at     : {result.lifetime_peak_timestamp or 'n/a'}")
    busiest = find_busiest_hour(summary)
    if busiest:
        print(f" Busiest hour         : {busiest[0]}  [{busiest[1]}]")
    else:
        print(" Busiest hour         : n/a")
    print()
    print(" Average occupancy by hour")
    print("   hour           samples  average  max   entries  exits")
    for row in summary:
        print(f"   {row['hour']:<14} {row['samples']:>7}  {str(row['average_occupancy']):>7}  "
              f"{str(row['max_occupancy']):>3}  {row['entries']:>7}  {row['exits']:>5}")
    if not summary:
        print("   (no data)")
    print()
    print(f" Hourly summary written to {summary_path}")

    if chart:
        chart_path = data_dir / "occupancy_chart.png"
        if make_chart(chart_path, event_rows, occupancy_rows, result):
            print(f" Chart written to {chart_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate occupancy statistics from the CSV logs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                        help="folder containing the CSV files (default: ./data)")
    parser.add_argument("--no-chart", action="store_true", help="skip the PNG chart")
    args = parser.parse_args(argv)
    return generate_report(args.data_dir, chart=not args.no_chart)


if __name__ == "__main__":
    sys.exit(main())
