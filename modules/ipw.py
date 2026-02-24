"""Inverse Probability Weighting (IPW) module for ACID-Dash.

Implements:
- Standard IPW (Horvitz-Thompson) for ATE and ATT estimation
- Stabilized weights (normalized by marginal treatment probability)
- Weight trimming at user-specified percentile bounds
- Bootstrap standard errors and confidence intervals

Propensity scores are accepted as a pre-computed numpy array from
``modules/propensity.py``; this module handles only the weighting,
estimation, and uncertainty quantification steps.

References:
    Lunceford, J.K. & Davidian, M. (2004). Stratification and weighting
        via the propensity score in estimation of causal treatment effects.
        Statistics in Medicine, 23(19), 2937-2960.
    Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). Estimation of
        regression coefficients when some regressors are not always observed.
        JASA, 89(427), 846-866.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IpwResult:
    """Results from an IPW analysis.

    Attributes:
        estimand: Target estimand ('ATE' or 'ATT').
        estimate: Point estimate of the treatment effect.
        se: Bootstrap standard error.
        ci_lower: Lower bound of the 95% bootstrap CI.
        ci_upper: Upper bound of the 95% bootstrap CI.
        p_value: Two-sided p-value from t-distribution.
        n_treated: Number of treated units.
        n_control: Number of control units.
        weights_summary: Dict with weight diagnostics (min, max, mean,
            median, pct_trimmed, ess_treated, ess_control).
        method: String identifying the method ('ipw' or 'ipw_stabilized').
    """

    estimand: Literal["ATE", "ATT"]
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_treated: int
    n_control: int
    weights_summary: dict
    method: str


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def compute_ipw_weights(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    estimand: Literal["ATE", "ATT"] = "ATT",
    stabilized: bool = True,
    trim_percentile: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute IPW weights from propensity scores.

    For the ATT estimand (default):
      - Treated units: weight = 1
      - Control units: weight = e(X) / (1 - e(X))

    For the ATE estimand:
      - Treated units: weight = 1 / e(X)
      - Control units: weight = 1 / (1 - e(X))

    Stabilized weights normalize by marginal treatment probability
    to reduce variance without introducing bias.

    Args:
        propensity_scores: P(T=1|X) for each observation, shape (n,).
        treatment: Binary treatment indicator (0/1), shape (n,).
        estimand: Target estimand, 'ATE' or 'ATT'.
        stabilized: If True, apply stabilization (recommended).
        trim_percentile: If provided, clip propensity scores at this
            percentile from both tails before computing weights.
            E.g., 0.01 clips at the 1st and 99th percentiles.

    Returns:
        Tuple of (weights, summary_dict) where weights has shape (n,)
        and summary_dict contains diagnostic statistics.
    """
    ps = np.asarray(propensity_scores, dtype=float).copy()
    t = np.asarray(treatment, dtype=int)

    n = len(ps)
    n_trimmed = 0

    # Trim propensity scores at specified percentiles
    if trim_percentile is not None and trim_percentile > 0:
        lo = float(np.percentile(ps, trim_percentile * 100))
        hi = float(np.percentile(ps, (1 - trim_percentile) * 100))
        mask_trim = (ps < lo) | (ps > hi)
        n_trimmed = int(mask_trim.sum())
        ps = np.clip(ps, lo, hi)

    # Clip to prevent division by zero
    ps = np.clip(ps, 0.001, 0.999)

    weights = np.zeros(n, dtype=float)

    if estimand == "ATE":
        weights[t == 1] = 1.0 / ps[t == 1]
        weights[t == 0] = 1.0 / (1.0 - ps[t == 0])

        if stabilized:
            p_treat = float(t.mean())
            weights[t == 1] *= p_treat
            weights[t == 0] *= (1.0 - p_treat)

    elif estimand == "ATT":
        weights[t == 1] = 1.0
        weights[t == 0] = ps[t == 0] / (1.0 - ps[t == 0])

        if stabilized:
            # Normalize control weights to sum to N_treated
            w_ctrl = weights[t == 0]
            if w_ctrl.sum() > 0:
                weights[t == 0] = w_ctrl * (float(t.sum()) / w_ctrl.sum())

    # Effective sample size: ESS = (sum w)^2 / sum(w^2)
    w_t = weights[t == 1]
    w_c = weights[t == 0]
    ess_t = float(w_t.sum() ** 2 / (w_t**2).sum()) if len(w_t) > 0 else 0.0
    ess_c = float(w_c.sum() ** 2 / (w_c**2).sum()) if len(w_c) > 0 else 0.0

    summary = {
        "min": round(float(weights.min()), 4),
        "max": round(float(weights.max()), 4),
        "mean": round(float(weights.mean()), 4),
        "median": round(float(np.median(weights)), 4),
        "pct_trimmed": round(100.0 * n_trimmed / n, 1) if n > 0 else 0.0,
        "ess_treated": round(ess_t, 1),
        "ess_control": round(ess_c, 1),
    }

    return weights, summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_mean_diff(
    outcome: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted mean difference: E[Y|T=1] - E[Y|T=0]."""
    t = treatment.astype(bool)
    y_t = outcome[t]
    y_c = outcome[~t]
    w_t = weights[t]
    w_c = weights[~t]

    if w_t.sum() == 0 or w_c.sum() == 0:
        return 0.0

    wm_t = float(np.average(y_t, weights=w_t))
    wm_c = float(np.average(y_c, weights=w_c))
    return wm_t - wm_c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ipw(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    propensity_scores: np.ndarray,
    estimand: Literal["ATE", "ATT"] = "ATT",
    stabilized: bool = True,
    trim_percentile: float | None = None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> IpwResult:
    """Run IPW estimation of the treatment effect.

    Computes IPW weights from pre-computed propensity scores, estimates
    the ATE or ATT as a weighted mean difference, and quantifies
    uncertainty via the nonparametric bootstrap.

    Args:
        df: Full dataset with one row per observation.
        outcome_col: Name of the outcome variable column (numeric).
        treatment_col: Name of the binary treatment indicator (0/1).
        propensity_scores: Pre-computed propensity scores for every row
            of ``df``, in the same row order. Shape (len(df),).
        estimand: Target causal estimand. 'ATT' (default) for the
            Average Treatment Effect on the Treated, or 'ATE' for the
            Average Treatment Effect on the population.
        stabilized: If True (default), use stabilized IPW weights.
        trim_percentile: Trim propensity scores at this percentile
            from both tails. E.g., 0.01 trims at the 1st/99th. None
            disables trimming.
        n_bootstrap: Number of bootstrap replicates for SE estimation.
        seed: Random seed for reproducibility.

    Returns:
        IpwResult dataclass with the point estimate, SE, 95% CI,
        p-value, unit counts, weight diagnostics, and method label.

    Raises:
        ValueError: If treatment is not binary or lengths mismatch.
    """
    if len(propensity_scores) != len(df):
        raise ValueError(
            f"propensity_scores length ({len(propensity_scores)}) must "
            f"match df length ({len(df)})."
        )

    outcome = df[outcome_col].values.astype(float)
    treatment = df[treatment_col].values.astype(int)

    unique_vals = set(np.unique(treatment))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"treatment_col '{treatment_col}' must be binary (0/1). "
            f"Found: {unique_vals}"
        )

    n_treated = int((treatment == 1).sum())
    n_control = int((treatment == 0).sum())

    if n_treated == 0:
        raise ValueError("No treated units found.")
    if n_control == 0:
        raise ValueError("No control units found.")

    # Compute weights and point estimate
    weights, weights_summary = compute_ipw_weights(
        propensity_scores, treatment, estimand, stabilized, trim_percentile
    )
    estimate = _weighted_mean_diff(outcome, treatment, weights)

    # Bootstrap SE
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_estimates = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_w, _ = compute_ipw_weights(
            propensity_scores[idx], treatment[idx],
            estimand, stabilized, trim_percentile,
        )
        boot_estimates[b] = _weighted_mean_diff(
            outcome[idx], treatment[idx], boot_w,
        )

    se = float(np.std(boot_estimates, ddof=1))
    ci_lower = float(np.percentile(boot_estimates, 2.5))
    ci_upper = float(np.percentile(boot_estimates, 97.5))

    # p-value from t-distribution
    df_t = max(min(n_treated, n_control) - 1, 1)
    t_stat = estimate / se if se > 0 else np.nan
    if np.isnan(t_stat):
        p_value = float("nan")
    else:
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))

    method = "ipw_stabilized" if stabilized else "ipw"

    return IpwResult(
        estimand=estimand,
        estimate=round(estimate, 6),
        se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_treated=n_treated,
        n_control=n_control,
        weights_summary=weights_summary,
        method=method,
    )
