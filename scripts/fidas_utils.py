"""
Shared helpers for FIDAS 200 Excel exports.

Supports both legacy dN column names (e.g. dN0_1037) and numeric-centre
columns (e.g. 0.103730) used in current exports.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DN_COLUMN_PATTERN = re.compile(r"^dN(\d+)_(\d+)$")
NUMERIC_CENTRE_PATTERN = re.compile(r"^\d+\.\d+$")

MIN_DIAMETER_UM = 0.05
MAX_DIAMETER_UM = 50.0


def is_psd_column(name: str) -> bool:
    if DN_COLUMN_PATTERN.match(name):
        return True
    if NUMERIC_CENTRE_PATTERN.match(name):
        try:
            diameter = float(name)
        except ValueError:
            return False
        return MIN_DIAMETER_UM <= diameter <= MAX_DIAMETER_UM
    return False


def diameter_from_column(name: str) -> float:
    match = DN_COLUMN_PATTERN.match(str(name))
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    if NUMERIC_CENTRE_PATTERN.match(str(name)):
        diameter = float(name)
        if MIN_DIAMETER_UM <= diameter <= MAX_DIAMETER_UM:
            return diameter
    raise ValueError(f"Not a FIDAS PSD column: {name!r}")


def detect_psd_columns(columns: pd.Index) -> list[str]:
    return [str(col) for col in columns if is_psd_column(str(col))]


def sorted_psd_columns(columns: pd.Index) -> list[str]:
    psd_columns = detect_psd_columns(columns)
    return sorted(psd_columns, key=diameter_from_column)


def build_psd_table(psd_columns: list[str]) -> pd.DataFrame:
    records = [
        {"column": col, "Diameter_um": diameter_from_column(col)}
        for col in psd_columns
    ]
    return (
        pd.DataFrame(records)
        .sort_values("Diameter_um", kind="mergesort")
        .reset_index(drop=True)
    )


def normalize_fidas_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Timestamp and Flowrate_Lpm aliases expected by analysis scripts."""
    out = df.copy()
    if "Timestamp" not in out.columns:
        if "TimeStamp" in out.columns:
            out["Timestamp"] = pd.to_datetime(out["TimeStamp"], errors="coerce")
        elif "date" in out.columns and "time" in out.columns:
            out["Timestamp"] = pd.to_datetime(
                out["date"].astype(str) + " " + out["time"].astype(str),
                errors="coerce",
            )
        else:
            raise ValueError(
                "FIDAS export must include TimeStamp or date/time columns."
            )
    if "Flowrate_Lpm" not in out.columns:
        if "flowrate" in out.columns:
            out["Flowrate_Lpm"] = pd.to_numeric(out["flowrate"], errors="coerce")
        else:
            raise ValueError(
                "FIDAS export must include Flowrate_Lpm or flowrate column."
            )
    return out


def load_fidas_export(path: Path, sheet=0) -> pd.DataFrame:
    """Load a FIDAS 200 export (.xlsx or tab-delimited .txt)."""
    if not path.is_file():
        raise FileNotFoundError(f"FIDAS input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError(f"Unsupported FIDAS export format: {path.suffix}")
    return normalize_fidas_columns(df)


def load_fidas_excel(path: Path, sheet=0) -> pd.DataFrame:
    return load_fidas_export(path, sheet=sheet)


def geometric_bin_boundaries(
    centres_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centres = np.asarray(centres_um, dtype=float)
    n = len(centres)
    if n < 2:
        raise ValueError("At least two bin centres are required.")

    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)

    if n == 2:
        mid = np.sqrt(centres[0] * centres[1])
        lower[0] = centres[0] ** 2 / mid
        upper[0] = mid
        lower[1] = mid
        upper[1] = centres[1] ** 2 / mid
        return lower, upper

    edges = np.sqrt(centres[:-1] * centres[1:])
    upper[0] = edges[0]
    lower[0] = centres[0] ** 2 / upper[0]
    for i in range(1, n - 1):
        lower[i] = edges[i - 1]
        upper[i] = edges[i]
    lower[-1] = edges[-1]
    upper[-1] = centres[-1] ** 2 / lower[-1]
    return lower, upper


def resolve_common_overlap_period(
    ucass_master: pd.DataFrame,
    fidas_df: pd.DataFrame,
    tsi_spectra: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive overlap window shared by UCASS, FIDAS, and optionally TSI."""
    starts = [
        ucass_master["Timestamp"].min(),
        fidas_df["Timestamp"].min(),
    ]
    ends = [
        ucass_master["Timestamp"].max(),
        fidas_df["Timestamp"].max(),
    ]
    if tsi_spectra is not None and not tsi_spectra.empty:
        starts.append(tsi_spectra["Timestamp"].min())
        ends.append(tsi_spectra["Timestamp"].max())

    period_start = max(starts)
    period_end = min(ends)
    if period_start > period_end:
        raise RuntimeError(
            "No common overlap period across instruments: "
            f"{period_start} > {period_end}"
        )
    return period_start, period_end


UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"
UCASS_SOURCES = {
    1: {"csv": "UCASS13.csv", "sep": ";", "id_col": "UCASS_ID"},
    2: {"csv": "UCASS62.csv", "sep": ",", "id_col": "UCASS_ID.1"},
    6: {"csv": "UCASS62.csv", "sep": ",", "id_col": "UCASS_ID"},
}


def resolve_ucass_measurement_period(
    root: Path | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive window where UCASS 1, 2, and 6 all have measurements."""
    repo = root or Path(__file__).resolve().parent.parent
    master = None
    raw_cache: dict[tuple[Path, str], pd.DataFrame] = {}

    for uid, cfg in UCASS_SOURCES.items():
        csv_path = repo / "data" / "ucass" / cfg["csv"]
        cache_key = (csv_path, cfg["sep"])
        if cache_key not in raw_cache:
            raw = pd.read_csv(csv_path, skiprows=4, sep=cfg["sep"], low_memory=False)
            raw["Timestamp"] = pd.to_datetime(
                raw["GPS_Date"].astype(str) + " " + raw["GPS_Time[UTC]"].astype(str),
                format=UCASS_DATE_FORMAT,
                errors="coerce",
            )
            raw_cache[cache_key] = raw.dropna(subset=["Timestamp"])

        sub = raw_cache[cache_key].loc[
            raw_cache[cache_key][cfg["id_col"]] == uid, ["Timestamp"]
        ].copy()
        if sub["Timestamp"].duplicated().any():
            sub = sub.groupby("Timestamp", as_index=False).size()[["Timestamp"]]

        master = sub if master is None else pd.merge(
            master, sub, on="Timestamp", how="inner", validate="one_to_one"
        )

    if master is None or master.empty:
        raise RuntimeError("Could not resolve UCASS measurement period.")

    return master["Timestamp"].min(), master["Timestamp"].max()


def filter_fidas_to_ucass_period(
    df: pd.DataFrame,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Keep FIDAS rows within the UCASS 1/2/6 common measurement window."""
    period_start, period_end = resolve_ucass_measurement_period(root)
    filtered = df.loc[
        (df["Timestamp"] >= period_start) & (df["Timestamp"] <= period_end)
    ].copy()
    if filtered.empty:
        raise ValueError(
            "No FIDAS spectra within UCASS measurement period "
            f"{period_start} -> {period_end}"
        )
    return filtered, period_start, period_end
