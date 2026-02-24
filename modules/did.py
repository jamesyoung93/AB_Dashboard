"""Difference-in-Differences (DiD) estimation module for ACID-Dash.

Implements:
- Standard 2x2 DiD via OLS (statsmodels): Y ~ treated + post + treated:post + covariates
- ANCOVA form DiD: Y_post - Y_pre ~ treated + covariates
- Event study (dynamic DiD): period dummies interacted with treatment, reference period = -1
- Parallel trends data helper for visualization
- Stub for staggered DiD (Callaway-Sant'Anna via csdid — not yet implemented)

All estimates target the ATT (Average Treatment Effect on the Treated).
Clustered standard errors are supported via statsmodels cov_type='cluster'.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DidResult:
    """Results from a standard or ANCOVA DiD estimation.

    Attributes:
        att: ATT point estimate — coefficient on the Treated*Post interaction
            (standard DiD) or on the Treated indicator (ANCOVA form).
        se: Standard error. Clustered if cluster_col was provided; otherwise
            HC3 heteroskedasticity-robust.
        ci_lower: Lower bound of the 95% confidence interval.
        ci_upper: Upper bound of the 95% confidence interval.
        p_value: Two-sided p-value for the ATT estimate.
        n_treated: Number of unique treated units in the estimation sample.
        n_control: Number of unique control units in the estimation sample.
        model_summary: Full statsmodels OLS/WLS summary as a string.
        method: Estimation method used. One of 'standard_did' or 'ancova'.
    """

    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    model_summary: str
    method: str  # 'standard_did' | 'ancova'


@dataclass
class EventStudyResult:
    """Results from an event study (dynamic DiD) estimation.

    Attributes:
        periods: Relative time periods (integers). Period -1 is omitted
            (reference / normalisation period).
        coefficients: ATT estimate for each period relative to treatment onset.
        standard_errors: Standard errors for each period's coefficient.
        ci_lower: Lower 95% CI bounds per period.
        ci_upper: Upper 95% CI bounds per period.
        n_treated: Number of unique treated units used.
        n_control: Number of unique control units used.
        model_summary: Full statsmodels summary as a string.
    """

    periods: list[int]
    coefficients: list[float]
    standard_errors: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    n_treated: int
    n_control: int
    model_summary: str


# ---------------------------------------------------------------------------
# Helper: post indicator
# ---------------------------------------------------------------------------


def create_post_indicator(
    df: pd.DataFrame,
    time_col: str,
    post_period_start: int | float | str,
) -> pd.Series:
    """Create a binary 0/1 post-period indicator.

    A row is "post" (1) if its value in ``time_col`` is greater than or equal
    to ``post_period_start``. The comparison is numeric when both the column
    values and the threshold are numeric; otherwise string comparison is used.

    Args:
        df: Input DataFrame. Must contain ``time_col``.
        time_col: Name of the column that identifies the time period.
        post_period_start: Threshold value. Periods >= this value are "post".

    Returns:
        pd.Series of dtype int (0 or 1), same index as ``df``.

    Raises:
        KeyError: If ``time_col`` is not a column in ``df``.
        TypeError: If the comparison between the column dtype and the threshold
            cannot be performed (e.g. string column with numeric threshold).
    """
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame.")

    time_vals = df[time_col]

    # Attempt numeric comparison; fall back to string if needed.
    try:
        post = (time_vals >= type(time_vals.iloc[0])(post_period_start)).astype(int)
    except (ValueError, TypeError):
        post = (time_vals.astype(str) >= str(post_period_start)).astype(int)

    return post.rename("_post")


# ---------------------------------------------------------------------------
# Helper: parallel trends data
# ---------------------------------------------------------------------------


def parallel_trends_data(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Compute period-level mean outcome and 95% CI by treatment group.

    Returns a tidy DataFrame suitable for plotting the parallel trends
    assumption. Each row is one (time period, treatment group) combination.

    Args:
        df: Input panel DataFrame. Must contain ``outcome_col``,
            ``treatment_col``, and ``time_col``.
        outcome_col: Name of the outcome variable column.
        treatment_col: Name of the binary (0/1) treatment indicator column.
        time_col: Name of the time period column.

    Returns:
        DataFrame with columns:
            - ``time``: Time period value (same type as ``df[time_col]``).
            - ``group``: 'treated' or 'control'.
            - ``mean``: Mean outcome for the group in that period.
            - ``ci_lower``: Lower bound of the 95% confidence interval on the mean.
            - ``ci_upper``: Upper bound of the 95% confidence interval on the mean.
            - ``n``: Number of observations in the cell.

    Raises:
        KeyError: If any of the required columns are missing from ``df``.
    """
    for col in (outcome_col, treatment_col, time_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    records: list[dict] = []
    group_map = {1: "treated", 0: "control"}

    for treat_val, group_label in group_map.items():
        subset = df[df[treatment_col] == treat_val]
        for period, period_df in subset.groupby(time_col):
            y = period_df[outcome_col].dropna().values
            n = len(y)
            mean_val = float(np.mean(y))

            if n >= 2:
                sem = stats.sem(y)
                t_crit = stats.t.ppf(0.975, df=n - 1)
                ci_lower = mean_val - t_crit * sem
                ci_upper = mean_val + t_crit * sem
            else:
                ci_lower = ci_upper = mean_val

            records.append(
                {
                    "time": period,
                    "group": group_label,
                    "mean": mean_val,
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "n": n,
                }
            )

    return pd.DataFrame(records).sort_values(["group", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal: resolve clustering groups
# ---------------------------------------------------------------------------


def _resolve_cluster_groups(
    df: pd.DataFrame,
    cluster_col: Optional[str],
    entity_col: Optional[str],
    treatment_col: str,
) -> Optional[pd.Series]:
    """Determine which column to use for clustered standard errors.

    Priority: cluster_col > entity_col > treatment_col.

    Args:
        df: Estimation DataFrame (already indexed / filtered).
        cluster_col: Explicit clustering column requested by the caller.
        entity_col: Entity / unit identifier column.
        treatment_col: Treatment indicator column (fallback).

    Returns:
        A pd.Series of cluster membership values, or None if clustering
        should not be applied (single cluster would cause degenerate result).
    """
    col = cluster_col or entity_col or treatment_col

    if col not in df.columns:
        warnings.warn(
            f"Cluster column '{col}' not found in DataFrame. "
            "Falling back to HC3 robust SEs (no clustering).",
            stacklevel=3,
        )
        return None

    groups = df[col]
    n_clusters = groups.nunique()

    if n_clusters < 2:
        warnings.warn(
            f"Only {n_clusters} cluster(s) found in '{col}'. "
            "Clustered SEs require at least 2 clusters. "
            "Falling back to HC3 robust SEs.",
            stacklevel=3,
        )
        return None

    if n_clusters < 10:
        warnings.warn(
            f"Only {n_clusters} clusters found in '{col}'. "
            "Clustered SEs may be unreliable with fewer than 10-20 clusters. "
            "Interpret with caution.",
            stacklevel=3,
        )

    return groups


# ---------------------------------------------------------------------------
# Internal: robust results extraction helpers
# ---------------------------------------------------------------------------


def _param_names(base_fit) -> list[str]:  # noqa: ANN001
    """Return ordered parameter names from a fitted statsmodels OLS model.

    Args:
        base_fit: A fitted ``OLSResults`` object (before calling
            ``get_robustcov_results``).

    Returns:
        List of parameter name strings in the order they appear in
        ``base_fit.params`` / ``base_fit.model.exog_names``.
    """
    return list(base_fit.model.exog_names)


def _extract_param(
    fit,  # noqa: ANN001
    param_name: str,
    names: list[str],
) -> tuple[float, float, float, float, float]:
    """Extract (coef, se, ci_lower, ci_upper, p_value) for a named parameter.

    Works with both labelled (pandas Series) and unlabelled (numpy ndarray)
    ``params`` / ``bse`` / ``pvalues`` / ``conf_int()`` objects returned by
    statsmodels ``get_robustcov_results``.

    Args:
        fit: Robust results object returned by ``get_robustcov_results``.
        param_name: Name of the parameter to extract (must be in ``names``).
        names: Ordered list of parameter names from ``_param_names(base_fit)``.

    Returns:
        Tuple of (coef, se, ci_lower, ci_upper, p_value) as Python floats.

    Raises:
        ValueError: If ``param_name`` is not found in ``names``.
    """
    if param_name not in names:
        raise ValueError(
            f"Parameter '{param_name}' not found in model parameters: {names}"
        )
    idx = names.index(param_name)

    params = fit.params
    bse = fit.bse
    pvalues = fit.pvalues
    ci = fit.conf_int(alpha=0.05)

    # Labelled (pandas Series / DataFrame) or unlabelled (numpy array)
    if hasattr(params, "iloc"):
        coef = float(params.iloc[idx])
        se = float(bse.iloc[idx])
        pval = float(pvalues.iloc[idx])
    else:
        coef = float(params[idx])
        se = float(bse[idx])
        pval = float(pvalues[idx])

    if hasattr(ci, "iloc"):
        ci_lo = float(ci.iloc[idx, 0])
        ci_hi = float(ci.iloc[idx, 1])
    else:
        ci_lo = float(ci[idx, 0])
        ci_hi = float(ci[idx, 1])

    return coef, se, ci_lo, ci_hi, pval


# ---------------------------------------------------------------------------
# Internal: build OLS formula
# ---------------------------------------------------------------------------


def _sanitise_name(name: str) -> str:
    """Wrap a column name in Q() quoting for statsmodels formula API.

    This handles column names that contain spaces, hyphens, or other
    characters that would break patsy formula parsing.

    Args:
        name: Raw column name string.

    Returns:
        A patsy-safe reference: ``Q("name")`` if the name contains
        non-alphanumeric/underscore characters or starts with a digit,
        otherwise the name itself.
    """
    safe = name.replace('"', '\\"')
    needs_quoting = (
        not name.replace("_", "").isalnum()
        or (name[0].isdigit() if name else False)
    )
    return f'Q("{safe}")' if needs_quoting else name


# ---------------------------------------------------------------------------
# Standard 2x2 DiD
# ---------------------------------------------------------------------------


def run_did(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    post_period_start: int | float | str,
    covariate_cols: Optional[list[str]] = None,
    entity_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
) -> DidResult:
    """Estimate ATT via standard two-period Difference-in-Differences (OLS).

    Fits the model::

        Y ~ treated + post + treated:post [+ covariates]

    via OLS using the statsmodels formula API. The coefficient on
    ``treated:post`` is the ATT (Average Treatment Effect on the Treated).

    Standard errors are clustered at the level specified by ``cluster_col``
    (explicit) > ``entity_col`` > ``treatment_col``. If clustering produces
    fewer than 2 unique clusters the function falls back to HC3
    heteroskedasticity-robust SEs with a warning.

    Args:
        df: Panel or repeated cross-section DataFrame. Each row is one
            observation (e.g. customer x week). Must contain all referenced
            columns.
        outcome_col: Name of the continuous outcome variable column.
        treatment_col: Name of the binary (0/1) treatment indicator column.
            Values should be 0 (control) or 1 (treated).
        time_col: Name of the time period column. Used only to create the
            post indicator; the model does not include period dummies beyond
            the single post/pre split.
        post_period_start: Scalar threshold. Observations in periods >=
            this value are labelled ``post = 1``.
        covariate_cols: Optional list of additional covariate column names to
            include in the regression as additive controls.
        entity_col: Optional column identifying the entity (customer, unit)
            for clustering. Used as the cluster level if ``cluster_col`` is
            not provided.
        cluster_col: Optional explicit column to cluster standard errors on.
            Takes precedence over ``entity_col``.

    Returns:
        DidResult dataclass with ATT, SE, 95% CI, p-value, unit counts, full
        model summary, and ``method='standard_did'``.

    Raises:
        KeyError: If any referenced column is not in ``df``.
        ValueError: If ``outcome_col`` has no non-missing values after
            applying the post indicator.

    Example:
        >>> result = run_did(df, 'revenue', 'channel_email', 'week', 11,
        ...                   covariate_cols=['prior_spend', 'tenure_years'],
        ...                   entity_col='customer_id')
        >>> print(f"ATT = {result.att:.2f} (95% CI: {result.ci_lower:.2f}, {result.ci_upper:.2f})")
    """
    # --- Input validation ---
    required = [outcome_col, treatment_col, time_col]
    if covariate_cols:
        required.extend(covariate_cols)
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Work on a copy to avoid mutating caller's DataFrame
    work = df.copy()

    # Create post indicator and interaction term
    work["_post"] = create_post_indicator(work, time_col, post_period_start)
    work["_treated"] = work[treatment_col]
    work["_treated_post"] = work["_treated"] * work["_post"]

    # Drop rows with missing outcome or treatment
    n_before = len(work)
    work = work.dropna(subset=[outcome_col, treatment_col])
    n_dropped = n_before - len(work)
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} rows dropped due to missing values in "
            f"'{outcome_col}' or '{treatment_col}'.",
            stacklevel=2,
        )

    if len(work) == 0:
        raise ValueError(
            f"No observations remain after dropping rows with missing values in "
            f"'{outcome_col}' or '{treatment_col}'."
        )

    # Build formula
    outcome_q = _sanitise_name(outcome_col)
    formula = f"{outcome_q} ~ _treated + _post + _treated_post"

    if covariate_cols:
        cov_terms = " + ".join(_sanitise_name(c) for c in covariate_cols)
        formula = f"{formula} + {cov_terms}"

    # Fit OLS
    model = smf.ols(formula, data=work)
    base_fit = model.fit()

    # Apply clustered or robust SEs
    cluster_groups = _resolve_cluster_groups(
        work, cluster_col, entity_col, treatment_col
    )

    if cluster_groups is not None:
        fit = base_fit.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_groups,
        )
    else:
        fit = base_fit.get_robustcov_results(cov_type="HC3")

    # Extract ATT — coefficient on _treated_post
    names = _param_names(base_fit)
    att, se, ci_lower, ci_upper, p_value = _extract_param(
        fit, "_treated_post", names
    )

    # Count distinct units if entity_col is available
    if entity_col and entity_col in work.columns:
        n_treated = int(work.loc[work[treatment_col] == 1, entity_col].nunique())
        n_control = int(work.loc[work[treatment_col] == 0, entity_col].nunique())
    else:
        n_treated = int((work[treatment_col] == 1).sum())
        n_control = int((work[treatment_col] == 0).sum())

    return DidResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_treated=n_treated,
        n_control=n_control,
        model_summary=fit.summary().as_text(),
        method="standard_did",
    )


# ---------------------------------------------------------------------------
# ANCOVA form DiD
# ---------------------------------------------------------------------------


def run_ancova(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    post_period_start: int | float | str,
    entity_col: str,
    covariate_cols: Optional[list[str]] = None,
    cluster_col: Optional[str] = None,
) -> DidResult:
    """Estimate ATT via ANCOVA-form DiD (change-score regression).

    Computes ``Y_post - Y_pre`` for each entity using pre-period and
    post-period averages, then regresses the change score on the treatment
    indicator and optional covariates::

        (Y_post_avg - Y_pre_avg) ~ treated [+ covariates]

    The coefficient on ``treated`` is the ATT. This form is equivalent to
    the fully-interacted DiD under homogeneous effects and is often more
    efficient when pre-period baseline variation is large.

    Covariates are taken from the pre-period rows (time-invariant covariates
    like firm size and tenure are valid; time-varying covariates are averaged
    over the pre-period).

    Args:
        df: Panel DataFrame with at least two periods per entity (one pre,
            one post). Must contain ``outcome_col``, ``treatment_col``,
            ``time_col``, and ``entity_col``.
        outcome_col: Name of the continuous outcome variable column.
        treatment_col: Name of the binary (0/1) treatment indicator.
        time_col: Name of the time period column.
        post_period_start: Threshold. Periods >= this value are "post".
        entity_col: Column that identifies each entity (e.g. 'customer_id').
            Required for pre/post aggregation.
        covariate_cols: Optional list of additional covariate columns. Values
            are averaged over the pre-period for each entity.
        cluster_col: Column to cluster standard errors on. Defaults to
            ``entity_col`` if not provided.

    Returns:
        DidResult with ATT, SE, 95% CI, p-value, unit counts, model summary,
        and ``method='ancova'``.

    Raises:
        KeyError: If any referenced column is missing from ``df``.
        ValueError: If fewer than 2 entities have both pre and post
            observations, or if no pre-period observations exist.

    Example:
        >>> result = run_ancova(df, 'revenue', 'channel_email', 'week', 11,
        ...                      entity_col='customer_id',
        ...                      covariate_cols=['prior_spend'])
        >>> print(f"ATT (ANCOVA) = {result.att:.2f}")
    """
    required = [outcome_col, treatment_col, time_col, entity_col]
    if covariate_cols:
        required.extend(covariate_cols)
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    work = df.copy()
    work["_post"] = create_post_indicator(work, time_col, post_period_start)

    pre_df = work[work["_post"] == 0]
    post_df = work[work["_post"] == 1]

    if len(pre_df) == 0:
        raise ValueError(
            "No pre-period observations found. "
            f"Check that some rows have '{time_col}' < {post_period_start}."
        )
    if len(post_df) == 0:
        raise ValueError(
            "No post-period observations found. "
            f"Check that some rows have '{time_col}' >= {post_period_start}."
        )

    # Aggregate to entity level
    pre_agg = pre_df.groupby(entity_col).agg(
        _y_pre=(outcome_col, "mean"),
        _treated=(treatment_col, "first"),
    )
    post_agg = post_df.groupby(entity_col).agg(
        _y_post=(outcome_col, "mean"),
    )

    entity_df = pre_agg.join(post_agg, how="inner")
    entity_df["_delta_y"] = entity_df["_y_post"] - entity_df["_y_pre"]

    # Merge covariates (pre-period average)
    if covariate_cols:
        cov_agg = pre_df.groupby(entity_col)[covariate_cols].mean()
        entity_df = entity_df.join(cov_agg, how="left")

    if len(entity_df) < 2:
        raise ValueError(
            "Fewer than 2 entities have both pre- and post-period observations. "
            "ANCOVA requires a balanced or partially balanced panel."
        )

    entity_df = entity_df.reset_index()

    # Build formula
    formula = "_delta_y ~ _treated"
    if covariate_cols:
        cov_terms = " + ".join(_sanitise_name(c) for c in covariate_cols)
        formula = f"{formula} + {cov_terms}"

    model = smf.ols(formula, data=entity_df)
    base_fit = model.fit()

    # Clustering: default to entity-level (one obs per entity now, so no
    # clustering benefit — fall back to HC3 unless explicit cluster_col given)
    cluster_groups = None
    if cluster_col and cluster_col in entity_df.columns:
        cluster_groups = _resolve_cluster_groups(
            entity_df, cluster_col, entity_col=None, treatment_col=treatment_col
        )

    if cluster_groups is not None:
        fit = base_fit.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_groups,
        )
    else:
        fit = base_fit.get_robustcov_results(cov_type="HC3")

    names = _param_names(base_fit)
    att, se, ci_lower, ci_upper, p_value = _extract_param(fit, "_treated", names)

    n_treated = int(entity_df["_treated"].sum())
    n_control = int((entity_df["_treated"] == 0).sum())

    return DidResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_treated=n_treated,
        n_control=n_control,
        model_summary=fit.summary().as_text(),
        method="ancova",
    )


# ---------------------------------------------------------------------------
# Event study (dynamic DiD)
# ---------------------------------------------------------------------------


def run_event_study(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    event_time_col: str,
    covariate_cols: Optional[list[str]] = None,
    cluster_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    reference_period: int = -1,
) -> EventStudyResult:
    """Estimate dynamic treatment effects via an event study (binned DiD).

    Fits the model::

        Y ~ sum_{tau != ref} beta_tau * (treated * 1[event_time == tau])
            + event_time_dummies + treated [+ covariates]

    where ``event_time_col`` encodes time **relative to treatment onset**
    (e.g. -2, -1, 0, 1, 2, ...). The reference period (default: -1, the
    period immediately before treatment) is omitted to normalise the
    pre-treatment coefficients. Pre-treatment coefficients near zero
    provide visual evidence for the parallel trends assumption.

    Args:
        df: Panel DataFrame where each row is one entity x period observation.
            Must include ``outcome_col``, ``treatment_col``, ``time_col``,
            and ``event_time_col``.
        outcome_col: Continuous outcome variable.
        treatment_col: Binary (0/1) treatment indicator (1 = ever-treated unit,
            regardless of the event-time period). Control units should have 0.
        time_col: Calendar time period column (used for sorting output).
        event_time_col: Integer column giving time **relative to treatment
            onset** for each row. For never-treated control units this column
            can be any value (e.g. 0 or NaN); control rows with NaN in
            ``event_time_col`` are assigned a placeholder that keeps them in
            the comparison group.
        covariate_cols: Optional additive covariate controls.
        cluster_col: Column to cluster SEs on. Falls back to ``entity_col``
            then ``treatment_col``.
        entity_col: Entity / unit identifier (used as fallback cluster level).
        reference_period: Event time period to omit as normalisation
            reference. Default is -1 (one period before treatment onset).

    Returns:
        EventStudyResult with parallel lists of periods, coefficients, SEs,
        and 95% CI bounds, plus n_treated, n_control, and model summary.

    Raises:
        KeyError: If any referenced column is missing.
        ValueError: If fewer than 2 non-reference periods are found, or if
            the reference period is not present in the data.

    Example:
        >>> result = run_event_study(df, 'revenue', 'channel_email',
        ...                          'week', 'event_time',
        ...                          entity_col='customer_id')
        >>> for t, beta, se in zip(result.periods, result.coefficients,
        ...                         result.standard_errors):
        ...     print(f"tau={t:+d}: {beta:.2f} ({se:.2f})")
    """
    required = [outcome_col, treatment_col, time_col, event_time_col]
    if covariate_cols:
        required.extend(covariate_cols)
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    work = df.copy()

    # For control units, fill missing event_time with a sentinel that will
    # not collide with any real event period.
    _CONTROL_SENTINEL = "_control_"

    # Convert event_time to string so we can use it as a categorical in patsy.
    # NaN in event_time_col for control rows → use sentinel so they enter
    # the regression in all period cells (absorbed by the treated==0 condition).
    work["_event_time_str"] = work[event_time_col].apply(
        lambda x: str(int(x)) if pd.notna(x) else _CONTROL_SENTINEL
    )

    # Identify periods present (excluding sentinel and reference)
    treated_mask = work[treatment_col] == 1
    event_periods_raw = (
        work.loc[treated_mask, event_time_col].dropna().astype(int).unique()
    )
    event_periods = sorted(event_periods_raw.tolist())

    if reference_period not in event_periods:
        raise ValueError(
            f"Reference period {reference_period} not found among event-time "
            f"values for treated units: {event_periods}. "
            "Adjust reference_period or check event_time_col."
        )

    non_ref_periods = [p for p in event_periods if p != reference_period]
    if len(non_ref_periods) < 1:
        raise ValueError(
            "Only one non-reference period found. "
            "Event study requires at least 2 distinct event-time periods."
        )

    # Build interaction dummies manually to avoid patsy formula complexity.
    # For each non-reference period tau, create: treated * 1[event_time == tau]
    interaction_cols = []
    for tau in non_ref_periods:
        col_name = f"_inter_t{tau:+d}".replace("+", "p").replace("-", "m")
        work[col_name] = (
            (work[treatment_col] == 1)
            & (work[event_time_col].fillna(np.nan).apply(
                lambda x: int(x) == tau if pd.notna(x) else False
            ))
        ).astype(float)
        interaction_cols.append((tau, col_name))

    # Build event-time period dummies for the outcome model (absorb time FE)
    # to control for common time shocks.
    all_periods_str = sorted(
        work.loc[
            work["_event_time_str"] != _CONTROL_SENTINEL, "_event_time_str"
        ].unique(),
        key=lambda s: int(s),
    )

    # The period dummies will be added as separate columns.
    # Reference period dummy is omitted (absorbed into intercept).
    # Use only alphanumeric + underscore names so patsy doesn't misparse
    # the minus sign in negative period numbers (e.g. -2 → m2).
    period_dummy_cols = []
    ref_str = str(reference_period)
    for period_str in all_periods_str:
        if period_str == ref_str:
            continue
        safe_suffix = period_str.replace("-", "m").replace("+", "p")
        dcol = f"_period_d_{safe_suffix}"
        work[dcol] = (work["_event_time_str"] == period_str).astype(float)
        period_dummy_cols.append(dcol)

    # Construct formula using only columns with safe names.
    # All internal column names (_inter_*, _period_d_*) are already safe
    # (alphanumeric + underscore). User-supplied columns are sanitised via
    # _sanitise_name() which wraps unsafe names in Q("...").
    outcome_q = _sanitise_name(outcome_col)
    rhs_terms = [_sanitise_name(treatment_col)]
    rhs_terms += [c for _, c in interaction_cols]
    rhs_terms += period_dummy_cols
    if covariate_cols:
        rhs_terms += [_sanitise_name(c) for c in covariate_cols]

    formula = f"{outcome_q} ~ " + " + ".join(rhs_terms)

    model = smf.ols(formula, data=work)
    base_fit = model.fit()

    # SE adjustment
    cluster_groups = _resolve_cluster_groups(
        work, cluster_col, entity_col, treatment_col
    )
    if cluster_groups is not None:
        fit = base_fit.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_groups,
        )
    else:
        fit = base_fit.get_robustcov_results(cov_type="HC3")

    # Extract coefficients for each interaction term
    periods_out: list[int] = []
    coefs_out: list[float] = []
    ses_out: list[float] = []
    ci_lower_out: list[float] = []
    ci_upper_out: list[float] = []

    names = _param_names(base_fit)

    for tau, col_name in interaction_cols:
        if col_name not in names:
            warnings.warn(
                f"Interaction term '{col_name}' (tau={tau}) was dropped by "
                "statsmodels (possible collinearity). Skipping period.",
                stacklevel=2,
            )
            continue
        beta, se_val, ci_lo, ci_hi, _ = _extract_param(fit, col_name, names)

        periods_out.append(tau)
        coefs_out.append(beta)
        ses_out.append(se_val)
        ci_lower_out.append(ci_lo)
        ci_upper_out.append(ci_hi)

    # Insert reference period as zero (by normalisation)
    ref_idx = sorted(periods_out + [reference_period]).index(reference_period)
    periods_out.insert(ref_idx, reference_period)
    coefs_out.insert(ref_idx, 0.0)
    ses_out.insert(ref_idx, 0.0)
    ci_lower_out.insert(ref_idx, 0.0)
    ci_upper_out.insert(ref_idx, 0.0)

    # Unit counts
    if entity_col and entity_col in work.columns:
        n_treated = int(work.loc[work[treatment_col] == 1, entity_col].nunique())
        n_control = int(work.loc[work[treatment_col] == 0, entity_col].nunique())
    else:
        n_treated = int((work[treatment_col] == 1).sum())
        n_control = int((work[treatment_col] == 0).sum())

    return EventStudyResult(
        periods=periods_out,
        coefficients=coefs_out,
        standard_errors=ses_out,
        ci_lower=ci_lower_out,
        ci_upper=ci_upper_out,
        n_treated=n_treated,
        n_control=n_control,
        model_summary=fit.summary().as_text(),
    )


# ---------------------------------------------------------------------------
# Staggered DiD stub
# ---------------------------------------------------------------------------


def run_staggered_did(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    entity_col: str,
    cohort_col: Optional[str] = None,
    covariate_cols: Optional[list[str]] = None,
) -> None:
    """Stub for Callaway-Sant'Anna staggered Difference-in-Differences.

    This function is not yet implemented. Callaway-Sant'Anna (2021) DiD
    for staggered treatment adoption requires the ``csdid`` Python package,
    which is still maturing. The classical two-way fixed effects (TWFE)
    estimator is biased under staggered adoption due to negative weighting
    of already-treated units as controls for later-treated units
    (Goodman-Bacon 2021).

    When implemented, this function will:
    - Estimate group-time ATTs: ATT(g, t) for each cohort g and period t.
    - Aggregate to an overall ATT avoiding negative weighting.
    - Produce a heterogeneity-robust event study plot.

    Args:
        df: Panel DataFrame with one row per entity x period.
        outcome_col: Continuous outcome variable.
        treatment_col: Binary treatment indicator (0/1). Treatment is
            assumed to be absorbing (once treated, always treated).
        time_col: Calendar time period column.
        entity_col: Entity / unit identifier column.
        cohort_col: Optional column identifying the treatment cohort
            (period of first treatment). If None, it will be inferred
            from the data.
        covariate_cols: Optional pre-treatment covariates for the
            propensity model within the Callaway-Sant'Anna estimator.

    Raises:
        NotImplementedError: Always. Install ``csdid`` (pip install csdid)
            and await implementation in a future version.

    References:
        Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences
        with multiple time periods. Journal of Econometrics, 225(2), 200-230.

        Goodman-Bacon, A. (2021). Difference-in-differences with variation
        in treatment timing. Journal of Econometrics, 225(2), 254-277.
    """
    raise NotImplementedError(
        "Staggered DiD (Callaway-Sant'Anna) is not yet implemented. "
        "This estimator requires the 'csdid' Python package "
        "(pip install csdid). Once the package is installed and its API "
        "is stable, run_staggered_did() will be completed. "
        "In the meantime, use run_did() for a standard 2x2 DiD, "
        "noting that classical TWFE is biased under staggered adoption "
        "(see Goodman-Bacon 2021, Callaway & Sant'Anna 2021)."
    )
