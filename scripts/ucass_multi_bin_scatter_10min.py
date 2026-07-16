#!/usr/bin/env python3
"""
UCASS 1 / 2 / 6 intercomparison — fixed non-overlapping 10-minute bins.

Per second, sum counts across selected size bins only for each instrument.
Synchronize all three instruments at 1 Hz (inner join on Timestamp).
Per 10-minute interval, sum those per-second totals only when the interval
has complete triple overlap: 600 common seconds spanning the full bin.

Generates pairwise scatter plots:
  UCASS 1 vs UCASS 2
  UCASS 1 vs UCASS 6
  UCASS 2 vs UCASS 6
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

UCASS13_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"

FIGURE_DPI = 300
AGG_INTERVAL = "10min"
SECONDS_PER_INTERVAL = 600
BIN_LENGTH = pd.Timedelta(minutes=10)
ONE_SECOND = pd.Timedelta(seconds=1)

CALIBRATION = {
    1: ("AA001", 0, 1, "b{bin}"),
    2: ("AD002", 4, 5, "b{bin}.1"),
    6: ("AA006", 2, 3, "b{bin}"),
}

SELECTED_BIN_RANGES: dict[int, tuple[tuple[float, float], ...]] = {
    1: (
        (3.90, 4.44),
        (4.44, 4.93),
        (4.93, 5.37),
        (5.37, 5.85),
        (5.85, 6.28),
        (6.28, 6.77),
        (6.77, 7.18),
        (7.18, 7.67),
        (7.67, 8.12),
    ),
    6: (
        (3.90, 4.66),
        (4.66, 6.02),
        (6.02, 7.58),
        (7.58, 9.80),
    ),
    2: (
        (3.98, 4.16),
        (4.16, 4.86),
        (4.86, 6.04),
        (6.04, 7.40),
        (7.40, 9.24),
    ),
}

BIN_RANGE_LABELS = {
    1: "3.90–8.12 µm",
    2: "3.98–9.24 µm",
    6: "3.90–9.80 µm",
}

OUTPUT_FIGURE = OUTPUT_DIR / "UCASS_multi_bin_intercomparison_scatter_10min.png"
OUTPUT_MASTER_CSV = OUTPUT_DIR / "UCASS_multi_bin_scatter_10min.csv"

SCATTER_PAIRS = [(1, 2), (1, 6), (6, 2)]

UCASS_SOURCES = {
    1: {"csv": UCASS13_CSV, "sep": ";", "id_col": "UCASS_ID"},
    2: {"csv": UCASS362_CSV, "sep": ",", "id_col": "UCASS_ID.1"},
    6: {"csv": UCASS362_CSV, "sep": ",", "id_col": "UCASS_ID"},
}


def resolve_bin_columns(
    cal_name: str,
    col_low: int,
    col_up: int,
    col_template: str,
    ranges: tuple[tuple[float, float], ...],
) -> list[str]:
    raw = pd.read_excel(BINS_XLSX, header=None)
    lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)
    upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)

    columns: list[str] = []
    for target_lo, target_hi in ranges:
        matched = False
        for bin_idx, (lo, hi) in enumerate(zip(lower, upper)):
            if abs(lo - target_lo) < 0.01 and abs(hi - target_hi) < 0.01:
                columns.append(col_template.format(bin=bin_idx))
                matched = True
                break
        if not matched:
            raise ValueError(
                f"No bin in {cal_name} matches {target_lo:.2f}–{target_hi:.2f} µm"
            )
    return columns


def load_ucass_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, sep=sep, low_memory=False)
    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " + df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce",
    )
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def extract_per_second_sums(
    df: pd.DataFrame,
    ucass_id: int,
    id_col: str,
    bin_cols: list[str],
) -> pd.DataFrame:
    subset = df.loc[df[id_col] == ucass_id, ["Timestamp", *bin_cols]].copy()
    subset[f"UCASS{ucass_id}_counts"] = subset[bin_cols].sum(axis=1)
    subset = subset[["Timestamp", f"UCASS{ucass_id}_counts"]]

    if subset["Timestamp"].duplicated().any():
        subset = subset.groupby("Timestamp", as_index=False)[f"UCASS{ucass_id}_counts"].sum()

    return subset.sort_values("Timestamp").reset_index(drop=True)


def merge_all_ucass(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for uid in (1, 2, 6):
        frame = frames[uid]
        if merged is None:
            merged = frame
        else:
            merged = pd.merge(
                merged,
                frame,
                on="Timestamp",
                how="inner",
                validate="one_to_one",
            )
    return merged.sort_values("Timestamp").reset_index(drop=True)


def is_complete_interval(per_second: pd.DataFrame, bin_start: pd.Timestamp) -> bool:
    """True when all three instruments share every second in [bin_start, bin_start + 10 min)."""
    bin_end = bin_start + BIN_LENGTH
    timestamps = per_second.loc[
        (per_second["Timestamp"] >= bin_start) & (per_second["Timestamp"] < bin_end),
        "Timestamp",
    ].sort_values()

    if len(timestamps) != SECONDS_PER_INTERVAL:
        return False

    return timestamps.iloc[0] == bin_start and timestamps.iloc[-1] == bin_end - ONE_SECOND


def aggregate_to_complete_intervals(per_second: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_cols = [f"UCASS{uid}_counts" for uid in (1, 2, 6)]
    indexed = per_second.set_index("Timestamp")
    summed = indexed[count_cols].resample(interval).sum(min_count=1)
    n_seconds = indexed.resample(interval).size().rename("n_common_seconds")

    audit_rows: list[dict[str, object]] = []
    keep_bins: list[pd.Timestamp] = []

    for bin_start, n_common in n_seconds.items():
        if n_common == 0:
            continue
        complete = is_complete_interval(per_second, bin_start)
        audit_rows.append({
            "Timestamp": bin_start,
            "n_common_seconds": int(n_common),
            "complete_overlap": complete,
        })
        if complete:
            keep_bins.append(bin_start)

    audit = pd.DataFrame(audit_rows)
    intervals = (
        summed.loc[keep_bins]
        .join(n_seconds.loc[keep_bins])
        .reset_index()
    )
    return intervals, audit


def fit_scatter_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
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


def plot_pair_scatter(
    ax: plt.Axes,
    intervals: pd.DataFrame,
    uid_x: int,
    uid_y: int,
) -> dict[str, float | int]:
    col_x = f"UCASS{uid_x}_counts"
    col_y = f"UCASS{uid_y}_counts"
    x = intervals[col_x].to_numpy(dtype=float)
    y = intervals[col_y].to_numpy(dtype=float)
    fit = fit_scatter_stats(x, y)

    ax.scatter(
        x,
        y,
        s=60,
        color="#2ca02c",
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    if len(x) >= 2 and np.isfinite(fit["slope"]) and x.max() > x.min():
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_line, fit["slope"] * x_line + fit["intercept"], "--", color="gray",
            linewidth=1.2, label="OLS fit",
        )

    axis_max = max(x.max(initial=0), y.max(initial=0), 1.0) * 1.05
    ax.plot([0, axis_max], [0, axis_max], ":", color="black", linewidth=0.8, label="1:1 line")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)

    ax.set_xlabel(f"UCASS {uid_x} counts ({BIN_RANGE_LABELS[uid_x]}, 10 min)")
    ax.set_ylabel(f"UCASS {uid_y} counts ({BIN_RANGE_LABELS[uid_y]}, 10 min)")
    title = "UCASS 6 vs UCASS 2" if (uid_x, uid_y) == (6, 2) else f"UCASS {uid_x} vs UCASS {uid_y}"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)

    if np.isfinite(fit["r_squared"]):
        reg_eq = format_regression_equation(fit["slope"], fit["intercept"])
        stats_text = (
            f"{reg_eq}\n"
            f"$R^2$ = {fit['r_squared']:.3f},  r = {fit['r']:.3f},  "
            f"p = {fit['p_value']:.3g},  n = {fit['n']}"
        )
    else:
        stats_text = f"insufficient variation, n = {fit['n']}"

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    return fit


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
            raw_cache[csv_path],
            uid,
            cfg["id_col"],
            bin_cols[uid],
        )

    per_second = merge_all_ucass(per_second_frames)
    intervals, interval_audit = aggregate_to_complete_intervals(per_second, AGG_INTERVAL)
    intervals.to_csv(OUTPUT_MASTER_CSV, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    n_candidate = len(interval_audit)
    n_excluded = int((~interval_audit["complete_overlap"]).sum())

    print("\nUCASS multi-bin intercomparison — complete 10-minute bins only")
    print("=" * 60)
    print(f"Per-second rows (all instruments): {len(per_second)}")
    print(f"Candidate 10-minute intervals    : {n_candidate}")
    print(f"Excluded (incomplete overlap)    : {n_excluded}")
    print(f"Retained (complete overlap)        : {len(intervals)}")
    print(f"Period                           : {per_second['Timestamp'].min()} -> {per_second['Timestamp'].max()} UTC")

    if n_excluded:
        print("\nExcluded intervals:")
        for row in interval_audit.loc[~interval_audit["complete_overlap"]].itertuples(index=False):
            print(f"  {row.Timestamp}  (n_common_seconds = {row.n_common_seconds})")

    for ax, (uid_x, uid_y) in zip(axes, SCATTER_PAIRS):
        fit = plot_pair_scatter(ax, intervals, uid_x, uid_y)
        label = "UCASS 6 vs 2" if (uid_x, uid_y) == (6, 2) else f"UCASS {uid_x} vs {uid_y}"
        if np.isfinite(fit["r_squared"]):
            print(
                f"{label}: R²={fit['r_squared']:.6f}, "
                f"r={fit['r']:.6f}, p={fit['p_value']:.6e}, n={fit['n']}"
            )
        else:
            print(f"{label}: insufficient variation, n={fit['n']}")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"\nSaved figure   : {OUTPUT_FIGURE}")
    print(f"Saved master CSV: {OUTPUT_MASTER_CSV}")


if __name__ == "__main__":
    main()
