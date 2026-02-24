"""Balance diagnostics module for ACID-Dash.

Computes and visualizes covariate balance between treated and control groups.
Core outputs:
  - Standardized Mean Difference (SMD) tables (raw and adjusted)
  - Love plot: horizontal dot plot of absolute SMD pre/post adjustment
  - Propensity score overlap plot: KDE histograms for treated vs control
  - Covariate distribution plots: overlapping KDE or grouped bar chart

References:
  Austin (2009) Statistics in Medicine 28(25): 3083-3107.
  Stuart (2010) Statistical Science 25(1): 1-21.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # noqa: F401 (available for callers)
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats  # noqa: F401 (available for callers)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMD_PASS_THRESHOLD = 0.10
SMD_CAUTION_THRESHOLD = 0.25

STATUS_PASS = "Pass"
STATUS_CAUTION = "Caution"
STATUS_FAIL = "Fail"

# Colour palette — consistent across all plots
COLOUR_RAW = "#d62728"       # red for unadjusted
COLOUR_ADJUSTED = "#1f77b4"  # blue for adjusted/matched/weighted
COLOUR_TREATED = "#1f77b4"   # blue for treated group
COLOUR_CONTROL = "#ff7f0e"   # orange for control group


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BalanceResult:
    """Container for covariate balance diagnostics.

    Attributes:
        table: DataFrame with columns
            [covariate, mean_treated, mean_control, smd_raw,
             smd_adjusted, status].
            One row per covariate (or per dummy level for categoricals).
        max_smd: Maximum absolute adjusted SMD across all covariates.
        all_pass: True when every covariate has |adjusted SMD| < 0.10.
    """

    table: pd.DataFrame
    max_smd: float
    all_pass: bool
    # Internal: raw-only result when no adjustment has been applied
    _raw_only: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# SMD computation helpers
# ---------------------------------------------------------------------------

def _smd_continuous(
    vals_treated: np.ndarray,
    vals_control: np.ndarray,
) -> float:
    """Compute SMD for a continuous variable.

    Formula: (mean_T - mean_C) / sqrt((var_T + var_C) / 2)

    Uses ddof=1 (sample variance). Returns 0.0 when both groups have
    zero variance (constant columns).

    Args:
        vals_treated: Numeric values for the treated group.
        vals_control: Numeric values for the control group.

    Returns:
        Signed SMD (float). Positive means treated > control.
    """
    mean_t = np.nanmean(vals_treated)
    mean_c = np.nanmean(vals_control)
    var_t = np.nanvar(vals_treated, ddof=1)
    var_c = np.nanvar(vals_control, ddof=1)
    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    if pooled_sd == 0.0:
        return 0.0
    return float((mean_t - mean_c) / pooled_sd)


def _smd_binary(
    p_treated: float,
    p_control: float,
) -> float:
    """Compute SMD for a binary (0/1) variable using the proportion formula.

    Formula: (p_T - p_C) / sqrt((p_T*(1-p_T) + p_C*(1-p_C)) / 2)

    Returns 0.0 when denominator is zero (e.g., one group has p=0 or p=1
    and the other matches).

    Args:
        p_treated: Proportion of 1s in the treated group.
        p_control: Proportion of 1s in the control group.

    Returns:
        Signed SMD (float).
    """
    denom = np.sqrt(
        (p_treated * (1 - p_treated) + p_control * (1 - p_control)) / 2.0
    )
    if denom == 0.0:
        return 0.0
    return float((p_treated - p_control) / denom)


def _is_string_like_dtype(series: pd.Series) -> bool:
    """Return True for categorical, object, or string (StringDtype) series.

    Handles both legacy Pandas object dtype and Pandas 2+ StringDtype without
    relying on the deprecated ``pd.api.types.is_categorical_dtype``.

    Args:
        series: A pandas Series of any dtype.

    Returns:
        True if the series should be treated as categorical/text for SMD
        computation purposes.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype == object:
        return True
    # Pandas 1.0+ StringDtype (e.g., pd.StringDtype())
    if hasattr(pd, "StringDtype") and isinstance(series.dtype, pd.StringDtype):
        return True
    # Pandas 2.0+ ArrowDtype wrapping string
    try:
        import pyarrow as _pa  # noqa: F401
        if hasattr(pd, "ArrowDtype") and isinstance(series.dtype, pd.ArrowDtype):
            if series.dtype.pyarrow_dtype in (
                _pa.string(), _pa.large_string(), _pa.utf8(), _pa.large_utf8()
            ):
                return True
    except ImportError:
        pass
    # Pandas 3.0+ native str dtype (dtype.name == 'str')
    if hasattr(series.dtype, "name") and series.dtype.name in ("str", "string"):
        return True
    return False


def _expand_categorical(
    series: pd.Series,
    name: str,
) -> pd.DataFrame:
    """Dummy-code a categorical Series.

    Each level becomes a binary column named ``name_LEVEL``.
    All levels present in the series are included (no reference-level
    drop — each level's SMD is meaningful on its own).

    Args:
        series: Categorical or object-typed pandas Series.
        name: Base name for the dummy columns.

    Returns:
        DataFrame of 0/1 dummy columns.
    """
    dummies = pd.get_dummies(series, prefix=name, prefix_sep="_", dtype=float)
    return dummies


def _is_binary(series: pd.Series) -> bool:
    """Return True if a numeric series contains only values in {0, 1}.

    Args:
        series: Numeric pandas Series (NaNs ignored).

    Returns:
        True if the unique non-null values are a subset of {0, 1}.
    """
    unique_vals = set(series.dropna().unique())
    return unique_vals <= {0, 1, 0.0, 1.0}


def _compute_smd_for_series(
    series: pd.Series,
    mask_treated: pd.Series,
    mask_control: pd.Series,
    name: str,
) -> list[dict]:
    """Compute SMD row(s) for a single covariate series.

    Handles three cases:
      1. Numeric binary (0/1): binary SMD formula.
      2. Numeric continuous: continuous SMD formula.
      3. Categorical/object: dummy-code, then binary SMD per level.

    Args:
        series: The covariate column (full dataset length).
        mask_treated: Boolean mask selecting treated rows.
        mask_control: Boolean mask selecting control rows.
        name: Column name used for labelling rows.

    Returns:
        List of dicts, each with keys:
        [covariate, mean_treated, mean_control, smd].
    """
    rows: list[dict] = []

    if _is_string_like_dtype(series):
        # Categorical: dummy-code and compute binary SMD per level
        dummies = _expand_categorical(series, name)
        for col in dummies.columns:
            d = dummies[col]
            vals_t = d[mask_treated].values
            vals_c = d[mask_control].values
            p_t = float(np.nanmean(vals_t))
            p_c = float(np.nanmean(vals_c))
            rows.append({
                "covariate": col,
                "mean_treated": p_t,
                "mean_control": p_c,
                "smd": _smd_binary(p_t, p_c),
            })

    elif pd.api.types.is_numeric_dtype(series):
        vals_t = series[mask_treated].values.astype(float)
        vals_c = series[mask_control].values.astype(float)
        mean_t = float(np.nanmean(vals_t))
        mean_c = float(np.nanmean(vals_c))

        if _is_binary(series):
            rows.append({
                "covariate": name,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "smd": _smd_binary(mean_t, mean_c),
            })
        else:
            rows.append({
                "covariate": name,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "smd": _smd_continuous(vals_t, vals_c),
            })
    else:
        # Unknown type: attempt numeric coercion, warn on failure
        warnings.warn(
            f"Column '{name}' has unrecognised dtype {series.dtype}; "
            "skipping balance computation for this covariate.",
            stacklevel=3,
        )

    return rows


def _smd_status(abs_smd: float) -> str:
    """Map an absolute SMD value to a Pass / Caution / Fail status.

    Args:
        abs_smd: Absolute value of the SMD.

    Returns:
        "Pass", "Caution", or "Fail".
    """
    if abs_smd < SMD_PASS_THRESHOLD:
        return STATUS_PASS
    if abs_smd < SMD_CAUTION_THRESHOLD:
        return STATUS_CAUTION
    return STATUS_FAIL


# ---------------------------------------------------------------------------
# Public API: compute_balance
# ---------------------------------------------------------------------------

def compute_balance(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: Sequence[str],
    weights: np.ndarray | None = None,
    matched_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> BalanceResult:
    """Compute covariate balance between treated and control groups.

    Computes raw (unadjusted) SMD for all covariates, and optionally
    an adjusted SMD when ``weights`` or ``matched_indices`` are provided.
    Categorical covariates are dummy-coded and each level gets its own row.

    Args:
        df: Full dataset (all rows).
        treatment_col: Name of the binary treatment column (0/1).
        covariate_cols: List of covariate column names to assess.
        weights: Optional 1-D array of observation weights (length == len(df)).
            When provided, adjusted SMD is computed on the weighted sample.
            Typically IPW weights. Either ``weights`` or ``matched_indices``
            may be provided, not both.
        matched_indices: Optional tuple (treated_idx, control_idx) of integer
            arrays giving the row indices of the matched treated and control
            units. When provided, adjusted SMD is computed on the matched
            subsample.

    Returns:
        BalanceResult with:
          - table: DataFrame [covariate, mean_treated, mean_control,
                               smd_raw, smd_adjusted, status]
          - max_smd: Maximum absolute *adjusted* SMD.
          - all_pass: True when all |adjusted SMD| < 0.10.

    Raises:
        ValueError: If both ``weights`` and ``matched_indices`` are provided,
            or if the treatment column is not binary.
    """
    if weights is not None and matched_indices is not None:
        raise ValueError(
            "Provide either 'weights' or 'matched_indices', not both."
        )

    treatment = df[treatment_col]
    unique_treatment = set(treatment.dropna().unique())
    if not unique_treatment <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(
            f"Treatment column '{treatment_col}' must be binary (0/1). "
            f"Found unique values: {unique_treatment}"
        )

    mask_treated = treatment == 1
    mask_control = treatment == 0

    has_adjustment = (weights is not None) or (matched_indices is not None)

    # -----------------------------------------------------------------------
    # Raw SMD
    # -----------------------------------------------------------------------
    raw_rows: list[dict] = []
    for col in covariate_cols:
        if col not in df.columns:
            warnings.warn(f"Covariate column '{col}' not found in DataFrame; skipping.")
            continue
        raw_rows.extend(
            _compute_smd_for_series(df[col], mask_treated, mask_control, col)
        )

    # Build raw table
    raw_df = pd.DataFrame(raw_rows, columns=["covariate", "mean_treated", "mean_control", "smd"])

    # -----------------------------------------------------------------------
    # Adjusted SMD
    # -----------------------------------------------------------------------
    if has_adjustment:
        adj_rows: list[dict] = []

        if weights is not None:
            # Weighted SMD: treat weights as sampling weights
            w = np.asarray(weights, dtype=float)
            w_treated = w[mask_treated.values]
            w_control = w[mask_control.values]

            for col in covariate_cols:
                if col not in df.columns:
                    continue
                series = df[col]

                if _is_string_like_dtype(series):
                    dummies = _expand_categorical(series, col)
                    for dcol in dummies.columns:
                        d = dummies[dcol].values.astype(float)
                        d_t = d[mask_treated.values]
                        d_c = d[mask_control.values]
                        # Weighted means = weighted proportions for 0/1
                        p_t = float(np.average(d_t, weights=w_treated))
                        p_c = float(np.average(d_c, weights=w_control))
                        adj_rows.append({
                            "covariate": dcol,
                            "smd_adj": _smd_binary(p_t, p_c),
                            "mean_treated_adj": p_t,
                            "mean_control_adj": p_c,
                        })

                elif pd.api.types.is_numeric_dtype(series):
                    v = series.values.astype(float)
                    v_t = v[mask_treated.values]
                    v_c = v[mask_control.values]
                    m_t = float(np.average(v_t, weights=w_treated))
                    m_c = float(np.average(v_c, weights=w_control))

                    if _is_binary(series):
                        adj_rows.append({
                            "covariate": col,
                            "smd_adj": _smd_binary(m_t, m_c),
                            "mean_treated_adj": m_t,
                            "mean_control_adj": m_c,
                        })
                    else:
                        # Weighted variance
                        var_t = float(np.average((v_t - m_t) ** 2, weights=w_treated))
                        var_c = float(np.average((v_c - m_c) ** 2, weights=w_control))
                        pooled_sd = np.sqrt((var_t + var_c) / 2.0)
                        smd = float((m_t - m_c) / pooled_sd) if pooled_sd > 0 else 0.0
                        adj_rows.append({
                            "covariate": col,
                            "smd_adj": smd,
                            "mean_treated_adj": m_t,
                            "mean_control_adj": m_c,
                        })

        elif matched_indices is not None:
            treated_idx, control_idx = matched_indices
            df_matched_treated = df.iloc[treated_idx].reset_index(drop=True)
            df_matched_control = df.iloc[control_idx].reset_index(drop=True)
            # Build a combined matched frame with artificial treatment indicator
            df_matched = pd.concat(
                [
                    df_matched_treated.assign(**{treatment_col: 1}),
                    df_matched_control.assign(**{treatment_col: 0}),
                ],
                ignore_index=True,
            )
            mask_t_m = df_matched[treatment_col] == 1
            mask_c_m = df_matched[treatment_col] == 0

            for col in covariate_cols:
                if col not in df_matched.columns:
                    continue
                series_m = df_matched[col]
                sub_rows = _compute_smd_for_series(series_m, mask_t_m, mask_c_m, col)
                for r in sub_rows:
                    adj_rows.append({
                        "covariate": r["covariate"],
                        "smd_adj": r["smd"],
                        "mean_treated_adj": r["mean_treated"],
                        "mean_control_adj": r["mean_control"],
                    })

        adj_df = pd.DataFrame(adj_rows)

        # Merge raw and adjusted on covariate name
        table = raw_df.merge(adj_df, on="covariate", how="left")
        table = table.rename(columns={
            "smd": "smd_raw",
            "smd_adj": "smd_adjusted",
            "mean_treated_adj": "mean_treated_adj",
            "mean_control_adj": "mean_control_adj",
        })
        table["status"] = table["smd_adjusted"].abs().apply(_smd_status)

    else:
        # No adjustment: smd_adjusted == smd_raw
        table = raw_df.rename(columns={"smd": "smd_raw"})
        table["smd_adjusted"] = table["smd_raw"]
        table["status"] = table["smd_adjusted"].abs().apply(_smd_status)

    # Final column ordering
    final_cols = [
        "covariate", "mean_treated", "mean_control",
        "smd_raw", "smd_adjusted", "status",
    ]
    # Keep only final columns that exist (adjusted means may or may not be present)
    table = table[[c for c in final_cols if c in table.columns]]

    max_smd = float(table["smd_adjusted"].abs().max())
    all_pass = bool((table["smd_adjusted"].abs() < SMD_PASS_THRESHOLD).all())

    return BalanceResult(
        table=table.reset_index(drop=True),
        max_smd=max_smd,
        all_pass=all_pass,
        _raw_only=not has_adjustment,
    )


# ---------------------------------------------------------------------------
# Visualization: Love plot
# ---------------------------------------------------------------------------

def love_plot(
    balance_raw: BalanceResult,
    balance_adjusted: BalanceResult | None = None,
    title: str = "Love Plot: Covariate Balance",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal dot plot showing absolute SMD before and after adjustment.

    Covariates are sorted by raw absolute SMD (descending) on the Y-axis.
    Red dots indicate unadjusted SMD; blue dots indicate adjusted SMD.
    Vertical dashed lines mark the 0.10 (pass) and 0.25 (caution) thresholds.

    Args:
        balance_raw: BalanceResult computed without adjustment.
        balance_adjusted: Optional BalanceResult computed after matching or
            weighting. When None, only raw SMD dots are shown.
        title: Plot title string.
        figsize: Optional (width, height) in inches. Defaults to a height
            proportional to the number of covariates.

    Returns:
        matplotlib Figure object (caller must call plt.show() or savefig()
        as needed; the figure is NOT displayed automatically).
    """
    raw_table = balance_raw.table.copy()
    raw_table["abs_smd_raw"] = raw_table["smd_raw"].abs()

    if balance_adjusted is not None:
        adj_table = balance_adjusted.table[["covariate", "smd_adjusted"]].copy()
        adj_table["abs_smd_adj"] = adj_table["smd_adjusted"].abs()
        merged = raw_table.merge(adj_table, on="covariate", how="left")
    else:
        merged = raw_table.copy()
        merged["abs_smd_adj"] = merged["abs_smd_raw"]

    # Sort by raw absolute SMD descending
    merged = merged.sort_values("abs_smd_raw", ascending=True).reset_index(drop=True)

    n_covariates = len(merged)
    if figsize is None:
        height = max(4.0, n_covariates * 0.35 + 1.5)
        figsize = (8.0, height)

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(n_covariates)
    covariate_labels = merged["covariate"].tolist()

    # Draw connecting line segments (raw -> adjusted) for visual clarity
    if balance_adjusted is not None:
        for i, row in merged.iterrows():
            ax.plot(
                [row["abs_smd_raw"], row["abs_smd_adj"]],
                [i, i],
                color="grey",
                linewidth=0.6,
                alpha=0.5,
                zorder=1,
            )

    # Raw SMD dots
    ax.scatter(
        merged["abs_smd_raw"],
        y_positions,
        color=COLOUR_RAW,
        s=55,
        zorder=3,
        label="Raw (unadjusted)",
        edgecolors="white",
        linewidths=0.4,
    )

    # Adjusted SMD dots
    if balance_adjusted is not None:
        ax.scatter(
            merged["abs_smd_adj"],
            y_positions,
            color=COLOUR_ADJUSTED,
            s=55,
            zorder=3,
            label="Adjusted (matched/weighted)",
            edgecolors="white",
            linewidths=0.4,
        )

    # Threshold lines
    ax.axvline(
        SMD_PASS_THRESHOLD,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.2,
        label=f"|SMD| = {SMD_PASS_THRESHOLD} (Pass threshold)",
        zorder=2,
    )
    ax.axvline(
        SMD_CAUTION_THRESHOLD,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label=f"|SMD| = {SMD_CAUTION_THRESHOLD} (Caution threshold)",
        zorder=2,
    )

    # Shading regions
    ax.axvspan(0, SMD_PASS_THRESHOLD, alpha=0.06, color="#2ca02c", zorder=0)
    ax.axvspan(
        SMD_PASS_THRESHOLD, SMD_CAUTION_THRESHOLD,
        alpha=0.06, color="#ff7f0e", zorder=0,
    )
    ax.axvspan(
        SMD_CAUTION_THRESHOLD,
        max(merged["abs_smd_raw"].max() * 1.1, SMD_CAUTION_THRESHOLD + 0.05),
        alpha=0.06, color="#d62728", zorder=0,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(covariate_labels, fontsize=8)
    ax.set_xlabel("Absolute Standardized Mean Difference (|SMD|)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualization: Propensity score overlap plot
# ---------------------------------------------------------------------------

def propensity_overlap_plot(
    ps_treated: np.ndarray | pd.Series,
    ps_control: np.ndarray | pd.Series,
    title: str = "Propensity Score Overlap",
    figsize: tuple[float, float] = (8.0, 4.5),
    bw_adjust: float = 0.5,
) -> Figure:
    """Overlapping KDE density plots of propensity scores for treated and control.

    Reports the percentage of treated units outside the common support region
    (defined as [min(ps_control), max(ps_control)]).

    Args:
        ps_treated: Propensity scores for treated observations.
        ps_control: Propensity scores for control observations.
        title: Plot title string.
        figsize: Figure size (width, height) in inches.
        bw_adjust: Bandwidth adjustment factor for KDE smoothing
            (passed to seaborn.kdeplot; 1.0 = Scott's rule).

    Returns:
        matplotlib Figure object.
    """
    ps_t = np.asarray(ps_treated, dtype=float)
    ps_c = np.asarray(ps_control, dtype=float)

    # Common support: [min(control), max(control)]
    support_lo = float(ps_c.min())
    support_hi = float(ps_c.max())
    outside_support = np.sum((ps_t < support_lo) | (ps_t > support_hi))
    pct_outside = 100.0 * outside_support / len(ps_t)

    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(
        ps_t,
        ax=ax,
        color=COLOUR_TREATED,
        fill=True,
        alpha=0.35,
        bw_adjust=bw_adjust,
        label=f"Treated (N={len(ps_t):,})",
        linewidth=1.5,
    )
    sns.kdeplot(
        ps_c,
        ax=ax,
        color=COLOUR_CONTROL,
        fill=True,
        alpha=0.35,
        bw_adjust=bw_adjust,
        label=f"Control (N={len(ps_c):,})",
        linewidth=1.5,
    )

    # Common support region shading
    ax.axvspan(
        support_lo, support_hi,
        alpha=0.08, color="grey",
        label=f"Common support [{support_lo:.3f}, {support_hi:.3f}]",
        zorder=0,
    )
    ax.axvline(support_lo, color="grey", linestyle=":", linewidth=1.0, zorder=1)
    ax.axvline(support_hi, color="grey", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xlabel("Propensity Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: % outside common support
    annotation_text = (
        f"{pct_outside:.1f}% of treated units outside\ncommon support"
    )
    ax.text(
        0.97, 0.95,
        annotation_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        color="#d62728" if pct_outside > 5 else "#2ca02c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualization: Covariate distribution plot
# ---------------------------------------------------------------------------

def covariate_distribution_plot(
    series_treated: pd.Series,
    series_control: pd.Series,
    name: str,
    is_categorical: bool = False,
    figsize: tuple[float, float] = (7.0, 4.0),
    bw_adjust: float = 0.5,
) -> Figure:
    """Overlapping distribution plot for a single covariate.

    For continuous variables: overlapping KDE density plots.
    For categorical variables: grouped bar chart of proportions.

    Args:
        series_treated: Covariate values for the treated group.
        series_control: Covariate values for the control group.
        name: Covariate name used in axis labels and title.
        is_categorical: If True, render a grouped bar chart.
            If False (default), render overlapping KDE curves.
            Auto-detected from dtype when False and series is object/category.
        figsize: Figure size (width, height) in inches.
        bw_adjust: KDE bandwidth adjustment (continuous variables only).

    Returns:
        matplotlib Figure object.
    """
    # Auto-detect categorical when dtype suggests it
    if not is_categorical and _is_string_like_dtype(series_treated):
        is_categorical = True

    fig, ax = plt.subplots(figsize=figsize)

    if is_categorical:
        # Grouped bar chart of proportions
        cats_treated = series_treated.value_counts(normalize=True).sort_index()
        cats_control = series_control.value_counts(normalize=True).sort_index()

        all_categories = sorted(
            set(cats_treated.index) | set(cats_control.index),
            key=str,
        )
        x = np.arange(len(all_categories))
        bar_width = 0.38

        vals_t = [cats_treated.get(c, 0.0) for c in all_categories]
        vals_c = [cats_control.get(c, 0.0) for c in all_categories]

        ax.bar(
            x - bar_width / 2,
            vals_t,
            width=bar_width,
            color=COLOUR_TREATED,
            alpha=0.75,
            label=f"Treated (N={len(series_treated):,})",
        )
        ax.bar(
            x + bar_width / 2,
            vals_c,
            width=bar_width,
            color=COLOUR_CONTROL,
            alpha=0.75,
            label=f"Control (N={len(series_control):,})",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(c) for c in all_categories],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_xlabel(name, fontsize=10)

    else:
        # Overlapping KDE for continuous variables
        vals_t = series_treated.dropna().values.astype(float)
        vals_c = series_control.dropna().values.astype(float)

        sns.kdeplot(
            vals_t,
            ax=ax,
            color=COLOUR_TREATED,
            fill=True,
            alpha=0.35,
            bw_adjust=bw_adjust,
            label=f"Treated (N={len(vals_t):,})",
            linewidth=1.5,
        )
        sns.kdeplot(
            vals_c,
            ax=ax,
            color=COLOUR_CONTROL,
            fill=True,
            alpha=0.35,
            bw_adjust=bw_adjust,
            label=f"Control (N={len(vals_c):,})",
            linewidth=1.5,
        )
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)

    ax.set_title(
        f"Distribution of {name}: Treated vs Control",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Utility: summarise balance result as a styled display string
# ---------------------------------------------------------------------------

def balance_summary(result: BalanceResult) -> str:
    """Return a human-readable text summary of a BalanceResult.

    Args:
        result: BalanceResult from compute_balance.

    Returns:
        Multi-line string suitable for console printing or Streamlit st.text.
    """
    n = len(result.table)
    n_pass = (result.table["status"] == STATUS_PASS).sum()
    n_caution = (result.table["status"] == STATUS_CAUTION).sum()
    n_fail = (result.table["status"] == STATUS_FAIL).sum()

    lines = [
        f"Balance Summary ({n} covariate rows)",
        f"  Pass     (|SMD| < {SMD_PASS_THRESHOLD}):    {n_pass}",
        f"  Caution  (|SMD| {SMD_PASS_THRESHOLD}–{SMD_CAUTION_THRESHOLD}): {n_caution}",
        f"  Fail     (|SMD| > {SMD_CAUTION_THRESHOLD}):   {n_fail}",
        f"  Max |adjusted SMD|: {result.max_smd:.4f}",
        f"  All pass: {result.all_pass}",
    ]
    return "\n".join(lines)
