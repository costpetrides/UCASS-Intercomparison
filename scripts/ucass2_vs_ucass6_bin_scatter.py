#!/usr/bin/env python3
"""
UCASS 2 vs UCASS 6 intercomparison — fixed non-overlapping 10-minute bins.

  UCASS 2 (AD002): 4.86–6.04 µm  ->  column b3.1
  UCASS 6 (AA006): 4.66–6.02 µm  ->  column b9

Each non-overlapping 10-minute interval = one scatter point:
  x = sum of UCASS 2 bin counts in that interval
  y = sum of UCASS 6 bin counts in that interval
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

OUTPUT_PNG = OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_10min.png"
OUTPUT_CSV = OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_10min.csv"
FIGURE_DPI = 300
AGG_INTERVAL = "10min"


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
    intervals = aggregate_to_fixed_intervals(per_second, AGG_INTERVAL)

    x = intervals["UCASS2_counts"].to_numpy(dtype=float)
    y = intervals["UCASS6_counts"].to_numpy(dtype=float)

    r, p_value = stats.pearsonr(x, y)
    r_squared = r ** 2
    slope, intercept, _, _, _ = stats.linregress(x, y)

    intervals.to_csv(OUTPUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=60, color="#2ca02c", edgecolors="white", linewidths=0.6, zorder=3)

    if x.max() > x.min():
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "--", color="gray", linewidth=1.2, label="OLS fit")

    ax.plot([0, 25], [0, 25], ":", color="black", linewidth=0.8, label="1:1 line")

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_xlabel("UCASS 2 counts (4.86–6.04 µm, 10 min)")
    ax.set_ylabel("UCASS 6 counts (4.66–6.02 µm, 10 min)")
    ax.set_title("UCASS 2 vs UCASS 6", fontsize=11)
    ax.text(
        0.98, 0.02,
        f"$R^2$ = {r_squared:.3f},  r = {r:.3f},  n = {len(intervals)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI)
    plt.close(fig)

    print("UCASS 2 vs UCASS 6 — fixed 10-minute bin scatter")
    print("=" * 55)
    print(f"UCASS 2: column {ucass2_col}  ({lo2:.2f}–{hi2:.2f} µm, AD002)")
    print(f"UCASS 6: column {ucass6_col}  ({lo6:.2f}–{hi6:.2f} µm, AA006)")
    print("Aggregation: non-overlapping 10-minute intervals")
    print(f"Intervals: {len(intervals)}")
    print(f"Period   : {per_second['Timestamp'].min()} -> {per_second['Timestamp'].max()} UTC")
    print(f"R²       : {r_squared:.6f}")
    print(f"r        : {r:.6f}")
    print(f"p-value  : {p_value:.6e}")
    print(f"\nSaved plot: {OUTPUT_PNG}")
    print(f"Saved CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
