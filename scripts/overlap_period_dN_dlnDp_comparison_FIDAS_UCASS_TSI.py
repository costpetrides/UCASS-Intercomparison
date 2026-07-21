#!/usr/bin/env python3
"""
Overlap-period FIDAS vs UCASS vs TSI dN/dlnDp comparison (standalone analysis).

Uses the exact UCASS reference timestamps where UCASS 1/2/6 and wind overlap.
FIDAS and TSI spectra are included only when their timestamp matches exactly.

UCASS: volume-weighted  sum(counts_i) / sum(V)  (Intercomparison.py method)
FIDAS: volume-weighted  sum(n_i * V) / sum(V)    with V = Flowrate_Lpm * 1 min
TSI:   volume-weighted  sum(Ci * V) / sum(V)    with V = Q * (ts - DTC * td)

Does not modify any existing scripts or outputs.

Dependencies: numpy, pandas, matplotlib, openpyxl
Optional: numbers-parser (if UCASS_size_bins.numbers is used for calibrations)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tsi_mean_psd import load_tsi_spectra
from plot_utils import mask_nonpositive_for_log
from fidas_utils import (
    diameter_from_column,
    filter_to_reference_timestamps,
    geometric_bin_boundaries,
    load_fidas_excel,
    reference_period_bounds,
    sorted_psd_columns,
    ucass_reference_timestamps,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

UCASS27_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_NUMBERS = ROOT / "data" / "ucass" / "UCASS_size_bins.numbers"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"
FIDAS_XLSX = ROOT / "data" / "fidas" / "FIDAS200.txt"

FIDAS_INTEGRATION_MIN = 1.0  # minutes (60 s cadence in export)

OUTPUT_PNG = ROOT / "outputs" / "overlap" / "overlap_period_FIDAS_UCASS_TSI_dN_dlnDp.png"
OUTPUT_TSI_CSV = ROOT / "outputs" / "overlap" / "overlap_period_TSI_dN_dlnDp.csv"
OUTPUT_VALIDATION_CSV = ROOT / "outputs" / "overlap" / "overlap_period_FIDAS_UCASS_TSI_validation.csv"

UCASS_IDS = (1, 2, 6)
UCASS_SOURCES = {
    1: {"csv": UCASS27_CSV, "sep": ";", "id_col": "UCASS_ID", "bin_suffix": ""},
    2: {"csv": UCASS362_CSV, "sep": ";", "id_col": "UCASS_ID.1", "bin_suffix": ".1"},
    6: {"csv": UCASS362_CSV, "sep": ";", "id_col": "UCASS_ID", "bin_suffix": ""},
}
CALIBRATION_MAP = {1: "AA001", 6: "AA006", 2: "AD002"}
SAMPLE_AREA = 5.0e-07
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"
FIGURE_DPI = 300

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

UCASS_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 6: "#2ca02c"}
TSI_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# Reference timestamps (UCASS 1/2/6 + wind exact overlap)
# ---------------------------------------------------------------------------

def select_reference_period(
    master: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    reference = ucass_reference_timestamps(master)
    period_start, period_end = reference_period_bounds(reference)
    return reference, period_start, period_end


# ---------------------------------------------------------------------------
# UCASS pipeline (mirrors overlap_period_dN_dlnDp_comparison.py)
# ---------------------------------------------------------------------------

def load_bins_table(bins_file: Path) -> pd.DataFrame:
    if bins_file.suffix.lower() == ".xlsx":
        return pd.read_excel(bins_file, header=None)
    from numbers_parser import Document
    doc = Document(bins_file)
    table = doc.sheets[0].tables[0]
    return pd.DataFrame([
        [table.cell(r, c).value for c in range(table.num_cols)]
        for r in range(table.num_rows)
    ])


def load_calibrations() -> dict:
    bins_file = BINS_NUMBERS if BINS_NUMBERS.is_file() else BINS_XLSX
    if not bins_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {BINS_NUMBERS} or {BINS_XLSX}")
    raw = load_bins_table(bins_file)
    mapping = {"AA001": (0, 1), "AA006": (2, 3), "AD002": (4, 5), "AD005": (6, 7)}
    calibrations = {}
    for name, (col_low, col_up) in mapping.items():
        lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)
        upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)
        calibrations[name] = {"centre": np.sqrt(lower * upper)}
    return calibrations


def load_ucass_csv(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, sep=sep, low_memory=False)
    if "GPS_Date" not in df.columns:
        alt_sep = "," if sep == ";" else ";"
        df = pd.read_csv(path, skiprows=4, sep=alt_sep, low_memory=False)
    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " + df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce",
    )
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def extract_ucass_counts(df: pd.DataFrame, uid: int, id_col: str, bin_suffix: str) -> pd.DataFrame:
    count_cols = [f"b{i}" for i in range(1, 16)]
    raw_cols = [f"b{i}{bin_suffix}" for i in range(1, 16)]
    sub = df.loc[df[id_col] == uid, ["Timestamp"] + raw_cols].copy()
    sub = sub.rename(columns=dict(zip(raw_cols, count_cols)))
    if sub["Timestamp"].duplicated().any():
        sub = sub.groupby("Timestamp", as_index=False)[count_cols].sum()
    return sub.sort_values("Timestamp").reset_index(drop=True)


def load_wind_rtf(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        m = WIND_DATA_PATTERN.search(line.strip().rstrip("\\"))
        if m:
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=[
        "TIMESTAMP", "RECORD", "AirTC_Avg", "RH", "WindDir", "WS_ms_Avg", "WSDiag", "BP_mbar_Avg",
    ])
    df["Timestamp"] = pd.to_datetime(df["TIMESTAMP"])
    df["WS_ms_Avg"] = pd.to_numeric(df["WS_ms_Avg"], errors="coerce")
    return df.dropna(subset=["Timestamp"])


def build_ucass_master() -> pd.DataFrame:
    df_wind = load_wind_rtf(WIND_RTF)
    raw_cache: dict = {}
    master = None
    for uid in UCASS_IDS:
        cfg = UCASS_SOURCES[uid]
        key = (cfg["csv"], cfg["sep"])
        if key not in raw_cache:
            raw_cache[key] = load_ucass_csv(cfg["csv"], cfg["sep"])
        counts = extract_ucass_counts(
            raw_cache[key], uid, cfg["id_col"], cfg["bin_suffix"]
        )
        renamed = counts.rename(columns={f"b{i}": f"b{i}_id{uid}" for i in range(1, 16)})
        master = renamed if master is None else pd.merge(
            master, renamed, on="Timestamp", how="inner", validate="one_to_one"
        )
    master = pd.merge(
        master,
        df_wind[["Timestamp", "WS_ms_Avg"]],
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )
    master["sample_vol_cm3"] = SAMPLE_AREA * master["WS_ms_Avg"] * 1e6
    return master


def ucass_dln_dp(centres: np.ndarray) -> np.ndarray:
    widths = np.log(centres[1:]) - np.log(centres[:-1])
    return np.append(widths, widths[-1])


def compute_ucass_overlap_psd(
    master: pd.DataFrame,
    ucass_bins: dict[int, np.ndarray],
    reference_timestamps: pd.DatetimeIndex,
) -> tuple[dict, pd.DataFrame]:
    period = filter_to_reference_timestamps(master, reference_timestamps).copy()
    sample_vol = period["sample_vol_cm3"].to_numpy(dtype=float)
    total_vol = sample_vol.sum()

    results = {}
    for uid in UCASS_IDS:
        counts = period[[f"b{i}_id{uid}" for i in range(1, 16)]].to_numpy(dtype=float)
        conc = counts.sum(axis=0) / total_vol
        dln = ucass_dln_dp(ucass_bins[uid])
        dN_dlnDp = conc / dln
        results[uid] = {
            "n_records": len(period),
            "total_sample_vol_cm3": total_vol,
            "total_counts": counts.sum(),
            "Diameter_um": ucass_bins[uid],
            "conc_cm3": conc,
            "dlnDp": dln,
            "dN_dlnDp": dN_dlnDp,
        }
    return results, period


# ---------------------------------------------------------------------------
# FIDAS pipeline
# ---------------------------------------------------------------------------

def load_fidas() -> pd.DataFrame:
    return load_fidas_excel(FIDAS_XLSX)


def compute_fidas_overlap_psd(
    df: pd.DataFrame,
    reference_timestamps: pd.DatetimeIndex,
) -> tuple[dict, pd.DataFrame]:
    psd_cols = sorted_psd_columns(df.columns)
    centres = np.array([diameter_from_column(c) for c in psd_cols])
    lower, upper = geometric_bin_boundaries(centres)
    dln = np.log(upper / lower)

    period = filter_to_reference_timestamps(df, reference_timestamps).copy()
    psd = period[psd_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = psd.notna().all(axis=1)
    period = period.loc[valid_mask]
    psd = psd.loc[valid_mask]

    flow_lpm = pd.to_numeric(period["Flowrate_Lpm"], errors="coerce").to_numpy(float)
    vol_L = flow_lpm * FIDAS_INTEGRATION_MIN
    vol_cm3 = vol_L * 1000.0
    total_vol_L = vol_L.sum()
    total_vol_cm3 = vol_cm3.sum()

    dn = psd.to_numpy(dtype=float)
    cn = pd.to_numeric(period["Cn"], errors="coerce").to_numpy(float)

    conc = (dn * vol_L[:, None]).sum(axis=0) / total_vol_L
    N_total_bins = (dn * vol_L[:, None]).sum(axis=0)
    conc_from_N = N_total_bins / total_vol_L
    cn_vol_weighted = (cn * vol_L).sum() / total_vol_L
    dN_dlnDp = conc / dln

    return {
        "n_records": len(period),
        "total_sample_vol_L": total_vol_L,
        "total_sample_vol_cm3": total_vol_cm3,
        "psd_columns": psd_cols,
        "Diameter_um": centres,
        "conc_cm3": conc,
        "conc_from_N": conc_from_N,
        "dlnDp": dln,
        "dN_dlnDp": dN_dlnDp,
        "cn_vol_weighted": cn_vol_weighted,
        "cn_arithmetic": cn.mean(),
    }, period


# ---------------------------------------------------------------------------
# TSI pipeline (mirrors tsi_mean_dN_dlnDp.py bin geometry)
# ---------------------------------------------------------------------------

def compute_tsi_overlap_psd(
    spectra: pd.DataFrame,
    bins: pd.DataFrame,
    reference_timestamps: pd.DatetimeIndex,
) -> tuple[dict, pd.DataFrame]:
    centres = bins["Diameter_um"].to_numpy(dtype=float)
    lower = bins["Lower_um"].to_numpy(dtype=float)
    upper = bins["Upper_um"].to_numpy(dtype=float)
    dln = np.log(upper / lower)

    period = filter_to_reference_timestamps(spectra, reference_timestamps).copy()
    conc_cols = [f"conc_Bin_{int(b)}" for b in bins["Bin"]]
    conc_data = period[conc_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = conc_data.notna().all(axis=1)
    period = period.loc[valid_mask]
    conc_data = conc_data.loc[valid_mask]

    vol_cm3 = period["sample_vol_cm3"].to_numpy(dtype=float)
    total_vol_cm3 = vol_cm3.sum()

    conc = (conc_data.to_numpy(dtype=float) * vol_cm3[:, None]).sum(axis=0) / total_vol_cm3
    count_cols = [f"Bin {int(b)}" for b in bins["Bin"]]
    counts = period[count_cols].to_numpy(dtype=float)
    conc_from_N = counts.sum(axis=0) / total_vol_cm3
    total_conc = conc.sum()
    dN_dlnDp = conc / dln

    return {
        "n_records": len(period),
        "total_sample_vol_cm3": total_vol_cm3,
        "Diameter_um": centres,
        "conc_cm3": conc,
        "conc_from_N": conc_from_N,
        "dlnDp": dln,
        "dN_dlnDp": dN_dlnDp,
        "total_conc_cm3": total_conc,
    }, period


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validations(ucass: dict, fidas: dict, tsi: dict) -> pd.DataFrame:
    rows = []

    def add_check(
        instrument: str,
        uid_or_na: str,
        check: str,
        expected: float,
        actual: float,
        *,
        atol: float = 1e-9,
        rtol: float = 0.0,
    ):
        abs_diff = abs(actual - expected)
        scale = max(abs(expected), abs(actual), 1e-30)
        rel_diff = abs_diff / scale
        passed = abs_diff <= atol or rel_diff <= rtol
        rows.append({
            "instrument": instrument,
            "UCASS_ID": uid_or_na,
            "check": check,
            "expected": expected,
            "actual": actual,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "pass": passed,
        })

    for uid in UCASS_IDS:
        r = ucass[uid]
        sum_conc = r["conc_cm3"].sum()
        reint = (r["dN_dlnDp"] * r["dlnDp"]).sum()
        add_check("UCASS", str(uid), "sum(conc_i) == sum(dN/dlnDp * dlnDp)", sum_conc, reint)
        add_check(
            "UCASS", str(uid), "sum(conc_i) == total_counts/total_V", sum_conc,
            r["total_counts"] / r["total_sample_vol_cm3"],
        )

    f = fidas
    sum_conc = f["conc_cm3"].sum()
    reint = (f["dN_dlnDp"] * f["dlnDp"]).sum()
    add_check("FIDAS", "—", "sum(conc_i) == sum(dN/dlnDp * dlnDp)", sum_conc, reint)
    add_check(
        "FIDAS", "—", "sum(conc_i) == sum(N_i)/sum(V) (primary total concentration)",
        sum_conc, f["conc_from_N"].sum(),
    )
    add_check(
        "FIDAS", "—", "sum(conc_i) == volume-weighted mean Cn (export cross-check)",
        sum_conc, f["cn_vol_weighted"], rtol=0.01,
    )
    add_check(
        "FIDAS", "—", "conc_i == sum(N_i)/sum(V) per bin (max abs diff)", 0.0,
        np.max(np.abs(f["conc_cm3"] - f["conc_from_N"])),
    )

    t = tsi
    sum_conc = t["conc_cm3"].sum()
    reint = (t["dN_dlnDp"] * t["dlnDp"]).sum()
    add_check("TSI", "—", "sum(conc_i) == sum(dN/dlnDp * dlnDp)", sum_conc, reint)
    add_check(
        "TSI", "—", "sum(conc_i) == sum(N_i)/sum(V) (primary total concentration)",
        sum_conc, t["conc_from_N"].sum(),
    )
    add_check(
        "TSI", "—", "conc_i == sum(N_i)/sum(V) per bin (max abs diff)", 0.0,
        np.max(np.abs(t["conc_cm3"] - t["conc_from_N"])),
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_tsi_csv(tsi: dict) -> None:
    pd.DataFrame({
        "Diameter_um": tsi["Diameter_um"],
        "conc_cm3": tsi["conc_cm3"],
        "dlnDp": tsi["dlnDp"],
        "dN_dlnDp": tsi["dN_dlnDp"],
    }).to_csv(OUTPUT_TSI_CSV, index=False)


def plot_comparison(
    ucass: dict,
    fidas: dict,
    tsi: dict,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(
        fidas["Diameter_um"], mask_nonpositive_for_log(fidas["dN_dlnDp"]),
        "-o", markersize=3, linewidth=1.2, color="#9467bd", label="FIDAS 200",
        zorder=5,
    )
    for uid in UCASS_IDS:
        r = ucass[uid]
        ax.loglog(
            r["Diameter_um"], mask_nonpositive_for_log(r["dN_dlnDp"]),
            "-o", markersize=5, linewidth=1.2,
            color=UCASS_COLORS[uid], label=f"UCASS {uid}",
        )
    ax.loglog(
        tsi["Diameter_um"], mask_nonpositive_for_log(tsi["dN_dlnDp"]),
        "-o", markersize=3, linewidth=1.2, color=TSI_COLOR, label="TSI OPS 3330",
        zorder=5,
    )

    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("dN/dlnDp (# cm$^{-3}$)")
    ax.set_title(
        "Overlap-period mean PSD (dN/dlnDp)\n"
        f"{period_start} – {period_end} UTC"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI)
    plt.close(fig)


def print_report(
    reference_timestamps: pd.DatetimeIndex,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    ucass_period: pd.DataFrame,
    fidas_period: pd.DataFrame,
    tsi_period: pd.DataFrame,
    ucass: dict,
    fidas: dict,
    tsi: dict,
    validation: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("OVERLAP-PERIOD FIDAS vs UCASS vs TSI dN/dlnDp COMPARISON")
    print("=" * 72)
    print(f"UCASS reference timestamps : {len(reference_timestamps)}")
    print(f"Period bounds (inclusive)  : {period_start} -> {period_end} UTC")
    print()
    print("RECORD COUNTS (exact timestamp match to UCASS reference)")
    print(f"  UCASS master rows in period : {len(ucass_period)}")
    print(f"  FIDAS spectra in period     : {fidas['n_records']}")
    print(f"  TSI spectra in period       : {tsi['n_records']}")
    print(f"  FIDAS timestamp range       : {fidas_period['Timestamp'].min()} -> {fidas_period['Timestamp'].max()}")
    print(f"  TSI timestamp range         : {tsi_period['Timestamp'].min()} -> {tsi_period['Timestamp'].max()}")
    print()
    print("SAMPLE VOLUMES")
    print(f"  UCASS total V               : {ucass[1]['total_sample_vol_cm3']:.2f} cm³")
    print(f"  FIDAS total V               : {fidas['total_sample_vol_L']:.2f} L ({fidas['total_sample_vol_cm3']:.0f} cm³)")
    print(f"  TSI total V                 : {tsi['total_sample_vol_cm3']:.0f} cm³")
    print()
    print("INTEGRATED NUMBER CONCENTRATION (sum of bin conc)")
    for uid in UCASS_IDS:
        print(f"  UCASS {uid}  sum(conc_i)     : {ucass[uid]['conc_cm3'].sum():.6f} #/cm³")
    print(f"  FIDAS        sum(conc_i)     : {fidas['conc_cm3'].sum():.6f} #/cm³")
    print(f"  FIDAS        vol-weighted Cn : {fidas['cn_vol_weighted']:.6f} #/cm³")
    print(f"  TSI          sum(conc_i)     : {tsi['conc_cm3'].sum():.6f} #/cm³")
    print()
    print("VALIDATION")
    for _, row in validation.iterrows():
        status = "PASS" if row["pass"] else "FAIL"
        label = f"{row['instrument']} {row['UCASS_ID']}: {row['check']}"
        print(f"  [{status}] {label}")
        if not row["pass"] or row["check"].startswith("conc_i =="):
            print(f"         expected={row['expected']:.6e}, actual={row['actual']:.6e}, diff={row['abs_diff']:.6e}")
    print()
    print("OUTPUT FILES")
    for p in (OUTPUT_PNG, OUTPUT_TSI_CSV, OUTPUT_VALIDATION_CSV):
        print(f"  {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    tsi_spectra, _, tsi_bins = load_tsi_spectra()
    master = build_ucass_master()
    fidas_df = load_fidas()
    reference, period_start, period_end = select_reference_period(master)

    cal = load_calibrations()
    ucass_bins = {uid: cal[CALIBRATION_MAP[uid]]["centre"][1:] for uid in UCASS_IDS}

    ucass_results, ucass_period = compute_ucass_overlap_psd(
        master, ucass_bins, reference,
    )

    fidas_results, fidas_period = compute_fidas_overlap_psd(
        fidas_df, reference,
    )

    tsi_results, tsi_period = compute_tsi_overlap_psd(
        tsi_spectra, tsi_bins, reference,
    )

    if len(ucass_period) == 0:
        raise RuntimeError("No UCASS records at reference timestamps.")
    if fidas_results["n_records"] == 0:
        raise RuntimeError("No FIDAS spectra at UCASS reference timestamps.")
    if tsi_results["n_records"] == 0:
        raise RuntimeError("No TSI spectra at UCASS reference timestamps.")

    validation = run_validations(ucass_results, fidas_results, tsi_results)

    save_tsi_csv(tsi_results)
    validation.to_csv(OUTPUT_VALIDATION_CSV, index=False)
    plot_comparison(ucass_results, fidas_results, tsi_results, period_start, period_end)

    if not validation["pass"].all():
        raise RuntimeError(
            "Validation failed — see overlap_period_FIDAS_UCASS_TSI_validation.csv"
        )
    print_report(
        reference, period_start, period_end,
        ucass_period, fidas_period, tsi_period,
        ucass_results, fidas_results, tsi_results, validation,
    )


if __name__ == "__main__":
    main()
