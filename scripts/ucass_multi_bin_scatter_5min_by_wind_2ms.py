#!/usr/bin/env python3
"""
UCASS 1 / 2 / 6 multi-bin intercomparison — 5-minute bins, wind split at 2 m/s.

Same processing as ucass_multi_bin_scatter_5min.py (complete 300 s triple overlap),
with mean wind speed per interval and two groups on each panel:
  WS < 2 m/s
  WS > 2 m/s
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_utils import format_regression_equation
from ucass_multi_bin_scatter_5min import (
    AGG_INTERVAL,
    BIN_RANGE_LABELS,
    CALIBRATION,
    FIGURE_DPI,
    SCATTER_PAIRS,
    SECONDS_PER_INTERVAL,
    SELECTED_BIN_RANGES,
    UCASS_SOURCES,
    aggregate_to_complete_intervals,
    extract_per_second_sums,
    fit_scatter_stats,
    load_ucass_csv,
    merge_all_ucass,
    resolve_bin_columns,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ucass"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"

WIND_THRESHOLD_MS = 2.0

OUTPUT_FIGURE = OUTPUT_DIR / "UCASS_multi_bin_intercomparison_scatter_5min_by_wind_2ms.png"
OUTPUT_CSV = OUTPUT_DIR / "UCASS_multi_bin_scatter_5min_by_wind_2ms.csv"

WIND_DATA_PATTERN = re.compile(
    r'"?(2026-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"?,\s*'
    r"(\d+),\s*"
    r"([\d.]+),\s*"
    r"([\d.]+),\s*"
    r"(\d+),\s*"
    r"([\d.]+),\s*"
    r"(\d+),\s*"
    r"(\d+)"
)


@dataclass(frozen=True)
class WindGroup:
    key: str
    label: str
    color: str
    mask: pd.Series


def load_wind_rtf(path: Path) -> pd.DataFrame:
    rows: list[tuple[str, ...]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        match = WIND_DATA_PATTERN.search(line.strip().rstrip("\\"))
        if match:
            rows.append(match.groups())
    df = pd.DataFrame(rows, columns=[
        "TIMESTAMP", "RECORD", "AirTC_Avg", "RH", "WindDir", "WS_ms_Avg", "WSDiag", "BP_mbar_Avg",
    ])
    df["Timestamp"] = pd.to_datetime(df["TIMESTAMP"])
    df["WS_ms_Avg"] = pd.to_numeric(df["WS_ms_Avg"], errors="coerce")
    return df.dropna(subset=["Timestamp", "WS_ms_Avg"]).sort_values("Timestamp")


def attach_mean_wind(intervals: pd.DataFrame, per_second: pd.DataFrame) -> pd.DataFrame:
    indexed = per_second.set_index("Timestamp")
    ws_mean = indexed["WS_ms_Avg"].resample(AGG_INTERVAL).mean()
    out = intervals.merge(
        ws_mean.rename("WS_ms_Avg").reset_index(),
        on="Timestamp",
        how="left",
    )
    out["wind_class"] = np.where(
        out["WS_ms_Avg"] < WIND_THRESHOLD_MS,
        "low",
        "high",
    )
    return out


def wind_groups(intervals: pd.DataFrame) -> list[WindGroup]:
    low_mask = intervals["WS_ms_Avg"] < WIND_THRESHOLD_MS
    return [
        WindGroup("low", f"< {WIND_THRESHOLD_MS:.0f} m/s", "#1f77b4", low_mask),
        WindGroup("high", f"> {WIND_THRESHOLD_MS:.0f} m/s", "#d62728", ~low_mask),
    ]


def plot_pair_scatter_by_wind(
    ax: plt.Axes,
    intervals: pd.DataFrame,
    uid_x: int,
    uid_y: int,
    groups: list[WindGroup],
) -> None:
    col_x = f"UCASS{uid_x}_counts"
    col_y = f"UCASS{uid_y}_counts"
    stats_lines: list[str] = []
    axis_max = max(
        float(intervals[col_x].max()) if len(intervals) else 0.0,
        float(intervals[col_y].max()) if len(intervals) else 0.0,
        1.0,
    ) * 1.05

    for group in groups:
        gdf = intervals.loc[group.mask]
        x = gdf[col_x].to_numpy(dtype=float)
        y = gdf[col_y].to_numpy(dtype=float)
        fit = fit_scatter_stats(x, y)

        if fit["n"] == 0:
            stats_lines.append(f"{group.label}: n = 0")
            ax.scatter(
                [], [], s=60, color=group.color, edgecolors="white", linewidths=0.6,
                label=f"{group.label} (n=0)",
            )
            continue

        ax.scatter(
            x, y, s=60, color=group.color, edgecolors="white", linewidths=0.6,
            label=f"{group.label} (n={fit['n']})", zorder=3,
        )

        if len(x) >= 2 and np.isfinite(fit["slope"]) and x.max() > x.min():
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_line, fit["slope"] * x_line + fit["intercept"], "--",
                color=group.color, linewidth=1.2,
            )

        if np.isfinite(fit["r_squared"]):
            reg_eq = format_regression_equation(fit["slope"], fit["intercept"])
            stats_lines.append(
                f"{group.label}:\n{reg_eq}\n"
                f"$R^2$ = {fit['r_squared']:.3f},  r = {fit['r']:.3f},  "
                f"p = {fit['p_value']:.3g},  n = {fit['n']}"
            )
        else:
            stats_lines.append(f"{group.label}: insufficient variation, n = {fit['n']}")

    ax.plot([0, axis_max], [0, axis_max], ":", color="black", linewidth=0.8, label="1:1 line")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_xlabel(f"UCASS {uid_x} counts ({BIN_RANGE_LABELS[uid_x]}, 5 min)")
    ax.set_ylabel(f"UCASS {uid_y} counts ({BIN_RANGE_LABELS[uid_y]}, 5 min)")
    title = "UCASS 6 vs UCASS 2" if (uid_x, uid_y) == (6, 2) else f"UCASS {uid_x} vs UCASS {uid_y}"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(
        0.98, 0.02, "\n\n".join(stats_lines),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="none"),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bin_cols: dict[int, list[str]] = {}
    for uid, ranges in SELECTED_BIN_RANGES.items():
        cal_name, col_low, col_up, col_template = CALIBRATION[uid]
        bin_cols[uid] = resolve_bin_columns(cal_name, col_low, col_up, col_template, ranges)
        print(f"UCASS {uid} ({cal_name}) selected columns: {', '.join(bin_cols[uid])}")

    raw_cache: dict[Path, pd.DataFrame] = {}
    per_second_frames: dict[int, pd.DataFrame] = {}
    for uid, cfg in UCASS_SOURCES.items():
        csv_path = cfg["csv"]
        if csv_path not in raw_cache:
            raw_cache[csv_path] = load_ucass_csv(csv_path, sep=cfg.get("sep", ","))
        per_second_frames[uid] = extract_per_second_sums(
            raw_cache[csv_path], uid, cfg["id_col"], bin_cols[uid],
        )

    per_second = merge_all_ucass(per_second_frames)
    wind = load_wind_rtf(WIND_RTF)
    per_second = pd.merge(
        per_second,
        wind[["Timestamp", "WS_ms_Avg"]],
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )

    intervals, interval_audit = aggregate_to_complete_intervals(per_second, AGG_INTERVAL)
    intervals = attach_mean_wind(intervals, per_second)
    intervals["wind_threshold_ms"] = WIND_THRESHOLD_MS
    intervals.to_csv(OUTPUT_CSV, index=False)

    groups = wind_groups(intervals)
    n_low = int(groups[0].mask.sum())
    n_high = int(groups[1].mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    print("\nUCASS multi-bin intercomparison — 5-minute bins, wind split at 2 m/s")
    print("=" * 60)
    print(f"Per-second rows (instruments + wind): {len(per_second)}")
    print(f"Complete 5-minute intervals         : {len(intervals)}")
    print(f"Wind threshold                      : {WIND_THRESHOLD_MS:.1f} m/s")
    print(f"Intervals < {WIND_THRESHOLD_MS:.0f} m/s                  : {n_low}")
    print(f"Intervals > {WIND_THRESHOLD_MS:.0f} m/s                  : {n_high}")

    for ax, (uid_x, uid_y) in zip(axes, SCATTER_PAIRS):
        plot_pair_scatter_by_wind(ax, intervals, uid_x, uid_y, groups)
        label = "UCASS 6 vs 2" if (uid_x, uid_y) == (6, 2) else f"UCASS {uid_x} vs {uid_y}"
        for group in groups:
            gdf = intervals.loc[group.mask]
            fit = fit_scatter_stats(
                gdf[f"UCASS{uid_x}_counts"].to_numpy(float),
                gdf[f"UCASS{uid_y}_counts"].to_numpy(float),
            )
            if np.isfinite(fit["r_squared"]):
                print(
                    f"{label} [{group.label}]: R²={fit['r_squared']:.6f}, "
                    f"r={fit['r']:.6f}, p={fit['p_value']:.6e}, n={fit['n']}"
                )
            else:
                print(f"{label} [{group.label}]: insufficient variation, n={fit['n']}")

    fig.suptitle(
        f"UCASS intercomparison (5 min, complete overlap, wind split at {WIND_THRESHOLD_MS:.0f} m/s)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure: {OUTPUT_FIGURE}")
    print(f"Saved CSV   : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
