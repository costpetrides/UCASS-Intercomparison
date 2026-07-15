"""Shared helpers for publication PSD figures."""

from __future__ import annotations

import numpy as np


def mask_nonpositive_for_log(values: np.ndarray) -> np.ndarray:
    """
    Return a copy with non-positive values set to NaN for log-scale plotting.

    Does not modify source data used for CSV export or scientific calculations.
    """
    masked = np.asarray(values, dtype=float).copy()
    masked[masked <= 0] = np.nan
    return masked


def format_regression_equation(slope: float, intercept: float) -> str:
    """Format OLS line as y = ax + b for plot annotations."""
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return "y = n/a"
    if intercept >= 0:
        return f"y = {slope:.3f}x + {intercept:.3f}"
    return f"y = {slope:.3f}x − {abs(intercept):.3f}"
