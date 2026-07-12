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
