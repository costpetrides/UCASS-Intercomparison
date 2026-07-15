#!/usr/bin/env python3
"""
UCASS 2 vs UCASS 6 intercomparison — fixed non-overlapping bins (10 min and 5 min).

  UCASS 2 (AD002): 4.86–6.04 µm  ->  column b3.1
  UCASS 6 (AA006): 4.66–6.02 µm  ->  column b9

Each non-overlapping interval = one scatter point:
  x = sum of UCASS 2 bin counts in that interval
  y = sum of UCASS 6 bin counts in that interval
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_utils import format_regression_equation

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ucass"

UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"

UCASS2_BIN_LO, UCASS2_BIN_HI = 4.86, 6.04
UCASS6_BIN_LO, UCASS6_BIN_HI = 4.66, 6.02

CALIBRATION = {
    2: ("AD002", 4, 5, "b{bin}.1"),
    6: ("AA006", 2, 3, "b{bin}"),
}

FIGURE_DPI = 300

INTERVAL_CONFIGS = [
    {
        "agg_interval": "10min",
        "label": "10 min",
        "output_png": OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_10min.png",
        "output_csv": OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_10min.csv",
    },
    {
        "agg_interval": "5min",
        "label": "5 min",
        "output_png": OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_5min.png",
        "output_csv": OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_5min.csv",
    },
]


def resolve_bin_column(
    cal_name: str,
    col_low: int,
    col_up: int,
    target_lo: float,
    target_hi: float,
    col_template: str,
) -> tuple[int, str, float, float]:
    raw = pd.read_excel(BINS_XLSX, header=None)
    lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)
    upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)

    for bin_idx, (lo, hi) in enumerate(zip(lower, upper)):
        if abs(lo - target_lo) < 0.01 and abs(hi - target_hi) < 0.01:
            return bin_idx, col_template.format(bin=bin_idx), lo, hi

    raise ValueError(f"No bin in {cal_name} matches {target_lo}–{target_hi} µm")


def load_ucass_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, sep=",", low_memory=False)
    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " + df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce",
    )
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def extract_per_second_pairs(
    df: pd.DataFrame,
    ucass2_col: str,
    ucass6_col: str,
) -> pd.DataFrame:
    pairs = df.loc[
        (df["UCASS_ID"] == 6) & (df["UCASS_ID.1"] == 2),
        ["Timestamp", ucass2_col, ucass6_col],
    ].copy()
    pairs = pairs.rename(columns={
        ucass2_col: "UCASS2_counts",
        ucass6_col: "UCASS6_counts",
    })
    if pairs["Timestamp"].duplicated().any():
        pairs = pairs.groupby("Timestamp", as_index=False)[
            ["UCASS2_counts", "UCASS6_counts"]
        ].sum()
    return pairs.sort_values("Timestamp").reset_index(drop=True)


def aggregate_to_fixed_intervals(pairs: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Sum per-second counts into non-overlapping fixed-width intervals."""
    return (
        pairs.set_index("Timestamp")
        .resample(interval)
        .sum(min_count=1)
        .dropna(subset=["UCASS2_counts", "UCASS6_counts"])
        .reset_index()
    )


def plot_scatter(
    intervals: pd.DataFrame,
    *,
    interval_label: str,
    lo2: float,
    hi2: float,
    lo6: float,
    hi6: float,
    output_png: Path,
    output_csv: Path,
) -> None:
    x = intervals["UCASS2_counts"].to_numpy(dtype=float)
    y = intervals["UCASS6_counts"].to_numpy(dtype=float)

    r, p_value = stats.pearsonr(x, y)
    r_squared = r ** 2
    slope, intercept, _, _, _ = stats.linregress(x, y)

    intervals.to_csv(output_csv, index=False)

    axis_max = max(float(np.max(x)), float(np.max(y)), 1.0) * 1.08

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=60, color="#2ca02c", edgecolors="white", linewidths=0.6, zorder=3)

    if x.max() > x.min():
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_line, slope * x_line + intercept, "--", color="gray",
            linewidth=1.2, label="OLS fit",
        )

    ax.plot([0, axis_max], [0, axis_max], ":", color="black", linewidth=0.8, label="1:1 line")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_xlabel(f"UCASS 2 counts ({lo2:.2f}–{hi2:.2f} µm, {interval_label})")
    ax.set_ylabel(f"UCASS 6 counts ({lo6:.2f}–{hi6:.2f} µm, {interval_label})")
    ax.set_title("UCASS 2 vs UCASS 6", fontsize=11)
    reg_eq = format_regression_equation(slope, intercept)
    ax.text(
        0.98, 0.02,
        f"{reg_eq}\n$R^2$ = {r_squared:.3f},  r = {r:.3f},  n = {len(intervals)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_png, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"\nUCASS 2 vs UCASS 6 — fixed {interval_label} bin scatter")
    print("=" * 55)
    print(f"Aggregation: non-overlapping {interval_label} intervals")
    print(f"Intervals: {len(intervals)}")
    print(f"R²       : {r_squared:.6f}")
    print(f"r        : {r:.6f}")
    print(f"p-value  : {p_value:.6e}")
    print(f"Saved plot: {output_png}")
    print(f"Saved CSV : {output_csv}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, ucass2_col, lo2, hi2 = resolve_bin_column(
        *CALIBRATION[2][:3], UCASS2_BIN_LO, UCASS2_BIN_HI, CALIBRATION[2][3],
    )
    _, ucass6_col, lo6, hi6 = resolve_bin_column(
        *CALIBRATION[6][:3], UCASS6_BIN_LO, UCASS6_BIN_HI, CALIBRATION[6][3],
    )

    df = load_ucass_csv(UCASS362_CSV)
    per_second = extract_per_second_pairs(df, ucass2_col, ucass6_col)

    print("UCASS 2 vs UCASS 6 — selected-bin scatter")
    print("=" * 55)
    print(f"UCASS 2: column {ucass2_col}  ({lo2:.2f}–{hi2:.2f} µm, AD002)")
    print(f"UCASS 6: column {ucass6_col}  ({lo6:.2f}–{hi6:.2f} µm, AA006)")
    print(f"Period   : {per_second['Timestamp'].min()} -> {per_second['Timestamp'].max()} UTC")

    for cfg in INTERVAL_CONFIGS:
        intervals = aggregate_to_fixed_intervals(per_second, cfg["agg_interval"])
        plot_scatter(
            intervals,
            interval_label=cfg["label"],
            lo2=lo2,
            hi2=hi2,
            lo6=lo6,
            hi6=hi6,
            output_png=cfg["output_png"],
            output_csv=cfg["output_csv"],
        )


if __name__ == "__main__":
    main()
