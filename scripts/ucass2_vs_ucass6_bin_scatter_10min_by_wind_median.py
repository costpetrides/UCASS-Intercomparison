#!/usr/bin/env python3
"""
UCASS 2 vs UCASS 6 — 10-minute scatter coloured by median-split mean wind speed.

Complementary to the fixed-threshold analysis in
ucass2_vs_ucass6_bin_scatter_10min_by_wind.py (unchanged).

Groups:
  Low wind  — mean wind < median(WS_ms_Avg)
  High wind — mean wind ≥ median(WS_ms_Avg)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ucass"

UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"

UCASS2_BIN_LO, UCASS2_BIN_HI = 4.86, 6.04
UCASS6_BIN_LO, UCASS6_BIN_HI = 4.66, 6.02

CALIBRATION = {
    2: ("AD002", 4, 5, "b{bin}.1"),
    6: ("AA006", 2, 3, "b{bin}"),
}

OUTPUT_PNG = OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_10min_by_wind_median.png"
OUTPUT_CSV = OUTPUT_DIR / "UCASS2_vs_UCASS6_bin_scatter_10min_by_wind_median.csv"
FIGURE_DPI = 300
AGG_INTERVAL = "10min"

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


def resolve_bin_column(
    cal_name: str,
    col_low: int,
    col_up: int,
    target_lo: float,
    target_hi: float,
    col_template: str,
) -> tuple[str, float, float]:
    raw = pd.read_excel(BINS_XLSX, header=None)
    lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)
    upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)

    for bin_idx, (lo, hi) in enumerate(zip(lower, upper)):
        if abs(lo - target_lo) < 0.01 and abs(hi - target_hi) < 0.01:
            return col_template.format(bin=bin_idx), lo, hi

    raise ValueError(f"No bin in {cal_name} matches {target_lo}–{target_hi} µm")


def load_ucass_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, sep=",", low_memory=False)
    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " + df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce",
    )
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def load_wind_rtf(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        match = WIND_DATA_PATTERN.search(line.strip().rstrip("\\"))
        if match:
            rows.append(match.groups())
    df = pd.DataFrame(rows, columns=[
        "TIMESTAMP", "RECORD", "AirTC_Avg", "RH", "WindDir", "WS_ms_Avg", "WSDiag", "BP_mbar_Avg",
    ])
    df["Timestamp"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df["WS_ms_Avg"] = pd.to_numeric(df["WS_ms_Avg"], errors="coerce")
    return df.dropna(subset=["Timestamp", "WS_ms_Avg"]).sort_values("Timestamp")


def extract_per_second_pairs(df: pd.DataFrame, ucass2_col: str, ucass6_col: str) -> pd.DataFrame:
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


def merge_wind(pairs: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        pairs,
        wind[["Timestamp", "WS_ms_Avg"]],
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )
    return merged.sort_values("Timestamp").reset_index(drop=True)


def aggregate_to_fixed_intervals(merged: pd.DataFrame, interval: str) -> pd.DataFrame:
    return (
        merged.set_index("Timestamp")
        .resample(interval)
        .agg({
            "UCASS2_counts": "sum",
            "UCASS6_counts": "sum",
            "WS_ms_Avg": "mean",
        })
        .dropna()
        .reset_index()
    )


def fit_group_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
    n = len(x)
    if n < 2:
        return {"n": n, "r": np.nan, "r_squared": np.nan, "p_value": np.nan, "slope": np.nan, "intercept": np.nan}

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"n": n, "r": np.nan, "r_squared": np.nan, "p_value": np.nan, "slope": np.nan, "intercept": np.nan}

    r, p_value = stats.pearsonr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return {
        "n": n,
        "r": float(r),
        "r_squared": float(r ** 2),
        "p_value": float(p_value),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def plot_group_fit(ax: plt.Axes, x: np.ndarray, fit: dict[str, float | int | str], color: str) -> None:
    if len(x) < 2 or not np.isfinite(fit["slope"]):
        return
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        fit["slope"] * x_line + fit["intercept"],
        "--",
        color=color,
        linewidth=1.4,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ucass2_col, _, _ = resolve_bin_column(
        *CALIBRATION[2][:3], UCASS2_BIN_LO, UCASS2_BIN_HI, CALIBRATION[2][3],
    )
    ucass6_col, _, _ = resolve_bin_column(
        *CALIBRATION[6][:3], UCASS6_BIN_LO, UCASS6_BIN_HI, CALIBRATION[6][3],
    )

    ucass = load_ucass_csv(UCASS362_CSV)
    per_second = extract_per_second_pairs(ucass, ucass2_col, ucass6_col)
    wind = load_wind_rtf(WIND_RTF)
    merged_second = merge_wind(per_second, wind)
    intervals = aggregate_to_fixed_intervals(merged_second, AGG_INTERVAL)

    median_wind = float(intervals["WS_ms_Avg"].median())
    intervals["median_wind_threshold_ms"] = median_wind
    intervals["wind_class"] = np.where(
        intervals["WS_ms_Avg"] < median_wind,
        "low",
        "high",
    )
    intervals.to_csv(OUTPUT_CSV, index=False)

    groups = [
        WindGroup(
            "low",
            f"Low wind (< median, {median_wind:.3f} m/s)",
            "#1f77b4",
            intervals["WS_ms_Avg"] < median_wind,
        ),
        WindGroup(
            "high",
            f"High wind (≥ median, {median_wind:.3f} m/s)",
            "#d62728",
            intervals["WS_ms_Avg"] >= median_wind,
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    stats_lines: list[str] = [f"Median wind = {median_wind:.3f} m/s"]

    for group in groups:
        subset = intervals.loc[group.mask]
        x = subset["UCASS2_counts"].to_numpy(dtype=float)
        y = subset["UCASS6_counts"].to_numpy(dtype=float)
        fit = fit_group_stats(x, y)

        ax.scatter(
            x,
            y,
            s=70,
            color=group.color,
            edgecolors="white",
            linewidths=0.6,
            label=f"{group.label} (n = {fit['n']})",
            zorder=3,
        )
        plot_group_fit(ax, x, fit, group.color)

        if np.isfinite(fit["r_squared"]):
            stats_lines.append(
                f"{group.label}: $R^2$={fit['r_squared']:.3f}, r={fit['r']:.3f}, "
                f"p={fit['p_value']:.3g}, n={fit['n']}"
            )
        else:
            stats_lines.append(f"{group.label}: insufficient variation, n={fit['n']}")

    ax.plot([0, 25], [0, 25], ":", color="black", linewidth=0.8, label="1:1 line")

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_xlabel("UCASS 2 counts (4.86–6.04 µm, 10 min)")
    ax.set_ylabel("UCASS 6 counts (4.66–6.02 µm, 10 min)")
    ax.set_title("UCASS 2 vs UCASS 6", fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.98,
        0.02,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI)
    plt.close(fig)

    print("UCASS 2 vs UCASS 6 — 10-minute scatter by median wind split")
    print("=" * 60)
    print(f"Intervals total      : {len(intervals)}")
    print(f"Median wind threshold: {median_wind:.6f} m/s")
    for line in stats_lines[1:]:
        print(line)
    print(f"\nSaved plot: {OUTPUT_PNG}")
    print(f"Saved CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
