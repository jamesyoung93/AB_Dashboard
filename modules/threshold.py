"""Treatment threshold binarization and suggestion utilities.

Used by the Overview tab to convert a continuous treatment column
(e.g., touch count, spend tier, impression count) into a binary
treated / control indicator.

Key functions
-------------
compute_threshold_stats(series, threshold)
    Count treated / control at a given cut point.

suggest_thresholds(series)
    Return 3 candidate thresholds: median, mean, and the KDE trough.

binarize_treatment(series, threshold)
    Return a 0 / 1 Series where values >= threshold are coded as 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelmin
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThresholdStats:
    """Summary statistics for a given treatment threshold.

    Attributes:
        n_treated: Number of observations with value >= threshold.
        n_control: Number of observations with value < threshold.
        pct_treated: Percentage of observations that are treated.
        pct_control: Percentage of observations that are control.
        ratio_str: Human-readable treated:control ratio string,
            e.g. ``'1:2.4'``.
    """

    n_treated: int
    n_control: int
    pct_treated: float
    pct_control: float
    ratio_str: str


@dataclass
class ThresholdSuggestion:
    """A single suggested threshold value with provenance metadata.

    Attributes:
        value: The suggested threshold value.
        label: Short human-readable label shown in the UI, e.g.
            ``'Median (32.0)'``.
        method: Machine-readable method identifier: one of
            ``'median'``, ``'mean'``, or ``'kde_trough'``.
    """

    value: float
    label: str
    method: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_threshold_stats(
    series: pd.Series,
    threshold: float,
) -> ThresholdStats:
    """Compute treated / control counts and percentages at a threshold.

    Observations with ``value >= threshold`` are classified as treated;
    all others as control.  Missing values are excluded from all counts.

    Args:
        series: The continuous treatment column.  Must be numeric.
        threshold: The cut-point value used to binarize treatment.

    Returns:
        A :class:`ThresholdStats` dataclass with n_treated, n_control,
        pct_treated, pct_control, and a formatted ratio string.

    Raises:
        ValueError: If ``series`` contains no non-missing values after
            dropping NaNs.
    """
    clean = series.dropna()
    if len(clean) == 0:
        raise ValueError("series contains no non-missing values.")

    n_total = len(clean)
    n_treated = int((clean >= threshold).sum())
    n_control = n_total - n_treated

    pct_treated = 100.0 * n_treated / n_total
    pct_control = 100.0 * n_control / n_total

    ratio_str = _format_ratio(n_treated, n_control)

    return ThresholdStats(
        n_treated=n_treated,
        n_control=n_control,
        pct_treated=round(pct_treated, 1),
        pct_control=round(pct_control, 1),
        ratio_str=ratio_str,
    )


def suggest_thresholds(series: pd.Series) -> list[ThresholdSuggestion]:
    """Suggest three candidate thresholds for treatment binarization.

    The three suggestions are:

    1. **Median** — splits the sample as close to 50/50 as possible.
    2. **Mean** — relevant when the distribution is right-skewed and the
       analyst thinks "above average engagement" is a meaningful cutoff.
    3. **KDE trough** — the first local minimum of the kernel density
       estimate, which often corresponds to a natural break between a
       zero/near-zero mass and a right-shifted distribution (e.g., a
       spike at 0 touches followed by a spread of 1+ touches).  Falls
       back to the median if no local minimum is found.

    All suggestions are rounded to 4 significant figures for readability.

    Args:
        series: The continuous treatment column.  Must be numeric with
            at least 2 distinct non-missing values.

    Returns:
        List of exactly 3 :class:`ThresholdSuggestion` objects in the
        order [median, mean, kde_trough].

    Raises:
        ValueError: If ``series`` has fewer than 2 distinct non-missing
            values.
    """
    clean = series.dropna()
    unique_vals = clean.nunique()
    if unique_vals < 2:
        raise ValueError(
            f"series must have at least 2 distinct non-missing values; "
            f"found {unique_vals}."
        )

    median_val = float(clean.median())
    mean_val = float(clean.mean())
    kde_val = _find_kde_trough(clean, fallback=median_val)

    suggestions = [
        ThresholdSuggestion(
            value=round(median_val, 4),
            label=f"Median ({median_val:.4g})",
            method="median",
        ),
        ThresholdSuggestion(
            value=round(mean_val, 4),
            label=f"Mean ({mean_val:.4g})",
            method="mean",
        ),
        ThresholdSuggestion(
            value=round(kde_val, 4),
            label=f"KDE trough ({kde_val:.4g})",
            method="kde_trough",
        ),
    ]

    return suggestions


def binarize_treatment(series: pd.Series, threshold: float) -> pd.Series:
    """Binarize a continuous treatment series at a given threshold.

    Values >= threshold are coded 1 (treated); values < threshold are
    coded 0 (control).  Missing values remain NaN.  The returned Series
    preserves the original index.

    Args:
        series: The continuous treatment column.  Must be numeric.
        threshold: The cut-point value.

    Returns:
        A ``pd.Series`` of dtype ``Int8`` (nullable integer to preserve
        NaN) with the same index as ``series``.  Values are 0 or 1.

    Raises:
        TypeError: If ``series`` is not a numeric dtype.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"series must be a numeric dtype; got '{series.dtype}'."
        )

    result = pd.Series(
        np.where(series.isna(), np.nan, (series >= threshold).astype(float)),
        index=series.index,
        name=series.name,
    )
    # Cast to nullable Int8 so that NaN rows stay NaN and non-NaN rows are int
    result = result.where(series.notna(), other=pd.NA).astype("Int8")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_kde_trough(series: pd.Series, fallback: float) -> float:
    """Find the first local minimum of the KDE as a natural break point.

    Uses a Gaussian KDE evaluated on a fine grid (512 points spanning
    the observed range), then applies ``scipy.signal.argrelmin`` with
    ``order=5`` to locate local minima.  Returns the x-value of the
    first local minimum (smallest x among all detected troughs).

    If no local minimum is found — which happens when the KDE is
    monotone or has a single mode with no valley — the ``fallback``
    value is returned and a note is recorded via ``warnings.warn``.

    The KDE bandwidth is selected automatically via Scott's rule
    (scipy default), which works well for unimodal and mildly bimodal
    distributions.

    Args:
        series: Non-NaN numeric values to model.
        fallback: Value to return when no KDE trough is found.

    Returns:
        The x-coordinate of the first local minimum, or ``fallback``.
    """
    import warnings

    values = series.to_numpy(dtype=np.float64)

    # Degenerate check: need enough spread for a meaningful KDE
    if np.std(values) == 0 or len(values) < 10:
        warnings.warn(
            "KDE trough detection skipped (insufficient spread or N<10); "
            "using fallback.",
            stacklevel=3,
        )
        return fallback

    kde = gaussian_kde(values)

    x_min, x_max = values.min(), values.max()
    grid = np.linspace(x_min, x_max, 512)
    density = kde(grid)

    # Locate local minima; order=5 smooths over minor density wiggles
    (trough_indices,) = argrelmin(density, order=5)

    if len(trough_indices) == 0:
        warnings.warn(
            "No KDE local minimum found; using fallback threshold.",
            stacklevel=3,
        )
        return fallback

    # Return the value at the first (leftmost) trough
    first_trough_idx = trough_indices[0]
    return float(grid[first_trough_idx])


def _format_ratio(n_treated: int, n_control: int) -> str:
    """Format a treated:control ratio as a human-readable string.

    Args:
        n_treated: Number of treated observations.
        n_control: Number of control observations.

    Returns:
        A string such as ``'1:2.4'`` (treated normalised to 1) or
        ``'∞:0'`` when control is zero, or ``'0:∞'`` when treated is
        zero.
    """
    if n_treated == 0 and n_control == 0:
        return "0:0"
    if n_treated == 0:
        return "0:inf"
    if n_control == 0:
        return "inf:0"

    ratio = n_control / n_treated
    return f"1:{ratio:.1f}"
