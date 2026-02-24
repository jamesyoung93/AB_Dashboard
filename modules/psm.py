"""Propensity Score Matching (PSM) module for ACID-Dash.

Implements nearest-neighbor matching (1:1 and 1:k) with optional caliper
constraint, with- or without-replacement semantics, bootstrap standard errors,
and ATT estimation on the matched sample.

Propensity scores are accepted as a pre-computed numpy array from
``modules/propensity.py``; this module handles only the matching, ATT
estimation, and uncertainty quantification steps.

References:
    Stuart (2010). Matching methods for causal inference: a review and a look
        forward. Statistical Science, 25(1), 1-21.
    Austin (2011). An introduction to propensity score methods for reducing the
        effects of confounding in observational studies. Multivariate
        Behavioral Research, 46(3), 399-424.
    Rosenbaum & Rubin (1985). Constructing a control group using multivariate
        matched sampling methods. The American Statistician, 39(1), 33-38.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PsmResult:
    """Results from a PSM analysis.

    Attributes:
        att: Average Treatment Effect on the Treated point estimate.
            Defined as mean(Y_treated - Y_matched_control) for 1:1 matching,
            or mean of distance-weighted control outcomes for 1:k matching.
        se: Bootstrap standard error of the ATT estimate.
        ci_lower: Lower bound of the 95% bootstrap confidence interval.
        ci_upper: Upper bound of the 95% bootstrap confidence interval.
        p_value: Two-sided p-value derived from a t-statistic (ATT / SE).
        n_matched_treated: Number of treated units that received a match.
        n_matched_control: Number of control units used as matches (may be
            less than n_matched_treated for 1:1 without-replacement matching
            when caliper discards pairs; may be greater for 1:k matching).
        n_unmatched: Number of treated units that could NOT be matched (only
            non-zero when caliper is active or the control pool is exhausted).
        matched_indices: DataFrame with columns [treated_idx, control_idx,
            distance] where the index columns are positional row indices into
            the original ``df``.  For 1:k matching each treated unit appears
            k times (one row per matched control), with ``distance`` recording
            the logit-scale Euclidean distance used during matching.
    """

    att: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_matched_treated: int
    n_matched_control: int
    n_unmatched: int
    matched_indices: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MatchingMethod = Literal["nearest_1to1", "nearest_1tok", "caliper"]


def _logit(ps: np.ndarray) -> np.ndarray:
    """Compute logit (log-odds) of propensity scores.

    Args:
        ps: Array of propensity scores in (0, 1).  Values are clipped to
            [0.001, 0.999] before the transform to prevent ±inf.

    Returns:
        Array of logit-transformed propensity scores.
    """
    ps_clipped = np.clip(ps, 0.001, 0.999)
    return np.log(ps_clipped / (1.0 - ps_clipped))


def _match_1to1_without_replacement(
    treated_logit: np.ndarray,
    control_logit: np.ndarray,
    treated_pos: np.ndarray,
    control_pos: np.ndarray,
    caliper: float | None,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[float]]:
    """Greedy 1:1 nearest-neighbor matching without replacement.

    Each treated unit is visited in random order.  The nearest *unmatched*
    control is selected.  If a caliper is supplied, pairs whose logit
    distance exceeds the caliper are discarded and the treated unit is
    recorded as unmatched.

    Args:
        treated_logit: Logit propensity scores for treated units, shape (N_t,).
        control_logit: Logit propensity scores for control units, shape (N_c,).
        treated_pos: Original DataFrame row positions for treated units.
        control_pos: Original DataFrame row positions for control units.
        caliper: Maximum allowable logit distance (None disables caliper).
        rng: Seeded random number generator for shuffling treated order.

    Returns:
        Tuple of (matched_treated_pos, matched_control_pos, distances).
        Positions that failed the caliper are excluded from all three lists.
    """
    n_ctrl = len(control_logit)

    # Visit treated units in random order to avoid systematic bias
    shuffle_idx = rng.permutation(len(treated_logit))

    # NearestNeighbors on all control logits; we will manually exclude
    # already-matched controls by checking a boolean mask.
    available = np.ones(n_ctrl, dtype=bool)

    matched_t: list[int] = []
    matched_c: list[int] = []
    distances: list[float] = []

    for ti in shuffle_idx:
        # Find sorted distances to all *available* controls
        ctrl_indices = np.where(available)[0]
        if len(ctrl_indices) == 0:
            break  # control pool exhausted

        dists = np.abs(treated_logit[ti] - control_logit[ctrl_indices])
        best_local = int(np.argmin(dists))
        best_dist = float(dists[best_local])
        best_ctrl = ctrl_indices[best_local]

        if caliper is not None and best_dist > caliper:
            continue  # treated unit is unmatched

        matched_t.append(int(treated_pos[ti]))
        matched_c.append(int(control_pos[best_ctrl]))
        distances.append(best_dist)
        available[best_ctrl] = False  # mark control as used

    return matched_t, matched_c, distances


def _match_1to1_with_replacement(
    treated_logit: np.ndarray,
    control_logit: np.ndarray,
    treated_pos: np.ndarray,
    control_pos: np.ndarray,
    caliper: float | None,
    nn: NearestNeighbors,
) -> tuple[list[int], list[int], list[float]]:
    """Nearest-neighbor 1:1 matching with replacement via ball_tree.

    Args:
        treated_logit: Logit propensity scores for treated units, shape (N_t,).
        control_logit: Logit propensity scores for control units, shape (N_c,).
        treated_pos: Original DataFrame row positions for treated units.
        control_pos: Original DataFrame row positions for control units.
        caliper: Maximum allowable logit distance (None disables caliper).
        nn: Fitted NearestNeighbors instance (fitted on control_logit).

    Returns:
        Tuple of (matched_treated_pos, matched_control_pos, distances).
    """
    dists, ctrl_local_idxs = nn.kneighbors(
        treated_logit.reshape(-1, 1), n_neighbors=1
    )
    dists = dists[:, 0]
    ctrl_local_idxs = ctrl_local_idxs[:, 0]

    matched_t: list[int] = []
    matched_c: list[int] = []
    distances: list[float] = []

    for ti, (d, ci) in enumerate(zip(dists, ctrl_local_idxs)):
        if caliper is not None and d > caliper:
            continue
        matched_t.append(int(treated_pos[ti]))
        matched_c.append(int(control_pos[ci]))
        distances.append(float(d))

    return matched_t, matched_c, distances


def _match_1tok(
    treated_logit: np.ndarray,
    control_logit: np.ndarray,
    treated_pos: np.ndarray,
    control_pos: np.ndarray,
    k_neighbors: int,
    caliper: float | None,
    with_replacement: bool,
    rng: np.random.Generator,
    nn: NearestNeighbors,
) -> tuple[list[int], list[int], list[float]]:
    """Nearest-neighbor 1:k matching.

    For each treated unit, retrieves up to k nearest controls.  If
    ``with_replacement=False``, this is implemented as repeated 1:1 without-
    replacement passes (which approximates but does not guarantee exactly k
    matches per treated unit when the control pool is small).  With
    replacement, all k neighbors are fetched for every treated unit.

    Rows in the returned lists are parallel: each entry corresponds to one
    (treated, control) pair.  A treated unit will appear up to k times.

    Args:
        treated_logit: Logit propensity scores for treated units.
        control_logit: Logit propensity scores for control units.
        treated_pos: Original DataFrame row positions for treated units.
        control_pos: Original DataFrame row positions for control units.
        k_neighbors: Number of control matches to seek per treated unit.
        caliper: Maximum allowable logit distance (None disables caliper).
        with_replacement: Whether controls can be reused.
        rng: Seeded random number generator (used only when
            ``with_replacement=False``).
        nn: Fitted NearestNeighbors instance (fitted on control_logit).

    Returns:
        Tuple of (matched_treated_pos, matched_control_pos, distances).
        Each list has at most ``k_neighbors * len(treated_pos)`` entries.
    """
    if with_replacement:
        actual_k = min(k_neighbors, len(control_logit))
        dists, ctrl_local_idxs = nn.kneighbors(
            treated_logit.reshape(-1, 1), n_neighbors=actual_k
        )
        matched_t: list[int] = []
        matched_c: list[int] = []
        distances: list[float] = []
        for ti in range(len(treated_logit)):
            for ki in range(actual_k):
                d = float(dists[ti, ki])
                if caliper is not None and d > caliper:
                    break  # distances are sorted ascending; no need to continue
                matched_t.append(int(treated_pos[ti]))
                matched_c.append(int(control_pos[ctrl_local_idxs[ti, ki]]))
                distances.append(d)
        return matched_t, matched_c, distances

    # Without replacement: greedy pass, building up to k pairs per treated unit
    n_ctrl = len(control_logit)
    available = np.ones(n_ctrl, dtype=bool)
    # Dict: treated_pos -> list of (ctrl_pos, dist)
    pairs: dict[int, list[tuple[int, float]]] = {
        int(tp): [] for tp in treated_pos
    }

    for _pass in range(k_neighbors):
        # Re-randomize treated order each pass
        pass_order = rng.permutation(len(treated_logit))
        for ti in pass_order:
            tp = int(treated_pos[ti])
            if len(pairs[tp]) > _pass:
                continue  # already found a match this pass
            ctrl_indices = np.where(available)[0]
            if len(ctrl_indices) == 0:
                break
            dists_arr = np.abs(treated_logit[ti] - control_logit[ctrl_indices])
            best_local = int(np.argmin(dists_arr))
            best_dist = float(dists_arr[best_local])
            best_ctrl = ctrl_indices[best_local]
            if caliper is not None and best_dist > caliper:
                continue
            pairs[tp].append((int(control_pos[best_ctrl]), best_dist))
            available[best_ctrl] = False

    matched_t_out: list[int] = []
    matched_c_out: list[int] = []
    distances_out: list[float] = []
    for tp, pair_list in pairs.items():
        for cp, d in pair_list:
            matched_t_out.append(tp)
            matched_c_out.append(cp)
            distances_out.append(d)

    return matched_t_out, matched_c_out, distances_out


def _compute_att_from_pairs(
    df: pd.DataFrame,
    outcome_col: str,
    matched_indices: pd.DataFrame,
    k_neighbors: int,
) -> float:
    """Compute the ATT from a matched_indices DataFrame.

    For 1:1 matching (k_neighbors=1) or caliper matching, each treated unit
    has exactly one control; ATT = mean(Y_t - Y_c).

    For 1:k matching (k_neighbors > 1), each treated unit may have multiple
    controls whose outcomes are averaged with distance-based weights
    (inverse distance, normalised within each treated unit's match set).
    If all distances in a group are zero, equal weights are used.

    Args:
        df: Original dataset.
        outcome_col: Name of the outcome column.
        matched_indices: DataFrame with columns [treated_idx, control_idx,
            distance] produced by the matching step.
        k_neighbors: k used during matching.  Determines weighting scheme.

    Returns:
        ATT point estimate as a float.
    """
    y = df[outcome_col].values

    if k_neighbors == 1:
        # Simple mean difference
        y_treated = y[matched_indices["treated_idx"].values]
        y_control = y[matched_indices["control_idx"].values]
        return float(np.mean(y_treated - y_control))

    # 1:k case: compute weighted average control outcome per treated unit
    mi = matched_indices.copy()
    mi["y_treated"] = y[mi["treated_idx"].values]
    mi["y_control"] = y[mi["control_idx"].values]

    # Inverse-distance weights (small distance = high weight)
    # Add tiny epsilon to avoid division by zero when distance == 0
    eps = 1e-10
    mi["inv_dist"] = 1.0 / (mi["distance"].values + eps)

    att_contributions = []
    for _, group in mi.groupby("treated_idx", sort=False):
        w = group["inv_dist"].values
        w = w / w.sum()  # normalise
        y_ctrl_weighted = float(np.dot(w, group["y_control"].values))
        y_t = float(group["y_treated"].iloc[0])
        att_contributions.append(y_t - y_ctrl_weighted)

    return float(np.mean(att_contributions))


def _bootstrap_att(
    df: pd.DataFrame,
    outcome_col: str,
    matched_indices: pd.DataFrame,
    k_neighbors: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap the ATT by resampling matched pairs with replacement.

    Pairs are defined at the treated-unit level: for each replicate, a
    sample (with replacement) of the set of unique treated indices is drawn,
    and all rows in ``matched_indices`` belonging to the selected treated units
    are included.  The ATT is then recomputed on this resampled matched set.

    Args:
        df: Original dataset.
        outcome_col: Name of the outcome column.
        matched_indices: DataFrame with columns [treated_idx, control_idx,
            distance] from the primary matching step.
        k_neighbors: k used during matching, forwarded to ATT computation.
        n_bootstrap: Number of bootstrap replicates.
        rng: Seeded random number generator.

    Returns:
        Array of bootstrap ATT estimates, shape (n_bootstrap,).
    """
    unique_treated = matched_indices["treated_idx"].unique()
    n_t = len(unique_treated)
    boot_atts = np.empty(n_bootstrap, dtype=float)

    # Build a dict mapping treated_idx -> slice of rows in matched_indices.
    # We keep matched_indices with its integer RangeIndex so that column
    # names remain intact after grouping.
    groups: dict[int, pd.DataFrame] = {
        int(tid): grp.reset_index(drop=True)
        for tid, grp in matched_indices.groupby("treated_idx", sort=False)
    }

    for b in range(n_bootstrap):
        sampled_treated = rng.choice(unique_treated, size=n_t, replace=True)
        # Gather all rows for the sampled treated units (with repeats)
        frames = [groups[int(t)] for t in sampled_treated]
        boot_mi = pd.concat(frames, ignore_index=True)
        boot_atts[b] = _compute_att_from_pairs(df, outcome_col, boot_mi, k_neighbors)

    return boot_atts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_psm(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    propensity_scores: np.ndarray,
    method: _MatchingMethod = "nearest_1to1",
    caliper: float | None = None,
    k_neighbors: int = 3,
    with_replacement: bool = False,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> PsmResult:
    """Run propensity score matching and estimate the ATT.

    Matching is performed on the logit of the propensity score (the
    "linear propensity score"), which improves discrimination near the tails
    of the propensity distribution.

    Args:
        df: Full dataset with one row per observation.  Must contain
            ``outcome_col`` and ``treatment_col``.
        outcome_col: Name of the outcome variable column (numeric).
        treatment_col: Name of the binary treatment indicator column
            (must be 0/1 or boolean).
        propensity_scores: Pre-computed propensity scores for every row of
            ``df``, in the same row order as ``df``.  Shape (len(df),).
            Produced by ``modules/propensity.py``.
        method: Matching algorithm to use.  One of:

            ``"nearest_1to1"``
                Greedy nearest-neighbor 1:1 matching on logit propensity
                score.  Default.

            ``"nearest_1tok"``
                Nearest-neighbor 1:k matching; returns up to ``k_neighbors``
                controls per treated unit, with distance-weighted ATT.

            ``"caliper"``
                Same as ``"nearest_1to1"`` but discards any pair whose logit
                distance exceeds ``caliper * std(logit_ps)``.  Reports
                unmatched treated units.

        caliper: Caliper width in units of the standard deviation of the
            logit propensity score.  Only used when ``method="caliper"``.
            Default is ``None`` (no caliper); when ``None`` and method is
            ``"caliper"``, a default of 0.2 SD is applied.
        k_neighbors: Number of control matches per treated unit.  Only used
            when ``method="nearest_1tok"``.
        with_replacement: If ``True``, the same control unit can be matched to
            multiple treated units.  If ``False`` (default), each control is
            used at most once (greedy without-replacement).
        n_bootstrap: Number of bootstrap replicates for SE estimation.
        seed: Integer random seed for reproducibility.

    Returns:
        A :class:`PsmResult` dataclass.

    Raises:
        ValueError: If ``treatment_col`` is not binary, if ``propensity_scores``
            has the wrong length, or if ``method`` is not recognised.
        ValueError: If fewer than 2 treated units match (ATT is undefined).

    Notes:
        Performance: ``NearestNeighbors(algorithm='ball_tree')`` on a 1-D
        logit array scales as O(N log N).  Expect < 5 s on 100 K rows.

        For without-replacement 1:k matching the implementation is a greedy
        multi-pass algorithm and may not find exactly k controls for every
        treated unit when the control pool is small relative to k * N_treated.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if len(propensity_scores) != len(df):
        raise ValueError(
            f"propensity_scores length ({len(propensity_scores)}) must match "
            f"df length ({len(df)})."
        )
    if method not in ("nearest_1to1", "nearest_1tok", "caliper"):
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from 'nearest_1to1', 'nearest_1tok', 'caliper'."
        )

    treatment = df[treatment_col].values.astype(int)
    unique_vals = set(np.unique(treatment))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"treatment_col must be binary (0/1).  Found unique values: {unique_vals}."
        )

    # ------------------------------------------------------------------
    # Logit transform
    # ------------------------------------------------------------------
    logit_ps = _logit(np.asarray(propensity_scores, dtype=float))

    treated_mask = treatment == 1
    control_mask = treatment == 0

    # Positional row indices (integer positions, not index labels)
    all_positions = np.arange(len(df))
    treated_pos = all_positions[treated_mask]
    control_pos = all_positions[control_mask]

    n_treated_total = int(treated_mask.sum())
    n_control_total = int(control_mask.sum())

    if n_treated_total == 0:
        raise ValueError("No treated units found.")
    if n_control_total == 0:
        raise ValueError("No control units found.")

    treated_logit = logit_ps[treated_mask]
    control_logit = logit_ps[control_mask]

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Caliper width (in logit units)
    # ------------------------------------------------------------------
    logit_std = float(np.std(logit_ps))
    caliper_abs: float | None = None
    if method == "caliper":
        cal_sd = caliper if caliper is not None else 0.2
        caliper_abs = cal_sd * logit_std
    elif caliper is not None:
        # Allow caller to pass caliper for non-caliper methods; treat as
        # advisory (issue a warning rather than failing).
        warnings.warn(
            "caliper argument is ignored when method is not 'caliper'. "
            "Use method='caliper' to enable caliper matching.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Fit NearestNeighbors on control logits (ball_tree, Euclidean/L2)
    # ------------------------------------------------------------------
    nn = NearestNeighbors(algorithm="ball_tree", metric="euclidean")
    nn.fit(control_logit.reshape(-1, 1))

    # ------------------------------------------------------------------
    # Run matching
    # ------------------------------------------------------------------
    if method == "nearest_1tok":
        matched_t, matched_c, distances = _match_1tok(
            treated_logit=treated_logit,
            control_logit=control_logit,
            treated_pos=treated_pos,
            control_pos=control_pos,
            k_neighbors=k_neighbors,
            caliper=caliper_abs,
            with_replacement=with_replacement,
            rng=rng,
            nn=nn,
        )
        k_eff = k_neighbors
    elif with_replacement:
        # 1:1 with replacement (caliper support included)
        matched_t, matched_c, distances = _match_1to1_with_replacement(
            treated_logit=treated_logit,
            control_logit=control_logit,
            treated_pos=treated_pos,
            control_pos=control_pos,
            caliper=caliper_abs,
            nn=nn,
        )
        k_eff = 1
    else:
        # 1:1 without replacement (also used for caliper)
        matched_t, matched_c, distances = _match_1to1_without_replacement(
            treated_logit=treated_logit,
            control_logit=control_logit,
            treated_pos=treated_pos,
            control_pos=control_pos,
            caliper=caliper_abs,
            rng=rng,
        )
        k_eff = 1

    if len(matched_t) == 0:
        raise ValueError(
            "No matches found. Consider widening the caliper, increasing "
            "the control pool, or checking the propensity score distribution."
        )

    matched_indices = pd.DataFrame(
        {
            "treated_idx": matched_t,
            "control_idx": matched_c,
            "distance": distances,
        }
    )

    # ------------------------------------------------------------------
    # ATT point estimate
    # ------------------------------------------------------------------
    att = _compute_att_from_pairs(df, outcome_col, matched_indices, k_eff)

    # ------------------------------------------------------------------
    # Bootstrap standard error
    # ------------------------------------------------------------------
    boot_atts = _bootstrap_att(
        df=df,
        outcome_col=outcome_col,
        matched_indices=matched_indices,
        k_neighbors=k_eff,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    se = float(np.std(boot_atts, ddof=1))

    # 95% percentile bootstrap CI
    ci_lower = float(np.percentile(boot_atts, 2.5))
    ci_upper = float(np.percentile(boot_atts, 97.5))

    # Two-sided p-value from t-distribution with (n_matched_treated - 1) df
    n_matched_treated = int(matched_indices["treated_idx"].nunique())
    t_stat = att / se if se > 0.0 else np.nan
    if np.isnan(t_stat):
        p_value = float("nan")
        warnings.warn(
            "Bootstrap SE is zero; p-value is undefined. "
            "Check for degenerate outcome or matching.",
            UserWarning,
            stacklevel=2,
        )
    else:
        df_t = max(n_matched_treated - 1, 1)
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))

    # ------------------------------------------------------------------
    # Summary counts
    # ------------------------------------------------------------------
    n_matched_control = int(matched_indices["control_idx"].nunique())
    n_unmatched = n_treated_total - n_matched_treated

    return PsmResult(
        att=round(att, 6),
        se=round(se, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(p_value, 6),
        n_matched_treated=n_matched_treated,
        n_matched_control=n_matched_control,
        n_unmatched=n_unmatched,
        matched_indices=matched_indices,
    )


# ---------------------------------------------------------------------------
# Helper functions for downstream consumers
# ---------------------------------------------------------------------------


def get_matched_data(
    df: pd.DataFrame,
    matched_indices: pd.DataFrame,
    treatment_col: str,
) -> pd.DataFrame:
    """Construct a matched-sample DataFrame from PSM results.

    Returns a tidy DataFrame containing one row per observation in the
    matched sample (treated and control rows interleaved), with an added
    ``match_id`` column that groups each matched pair (or group for 1:k).

    Args:
        df: Original full dataset.
        matched_indices: The ``matched_indices`` field from a
            :class:`PsmResult`.  Must contain columns
            [treated_idx, control_idx, distance].
        treatment_col: Name of the treatment indicator column; preserved
            as-is in the output.

    Returns:
        DataFrame with the same columns as ``df`` plus:

        - ``match_id`` (int): Identifies a matched pair/group.  Treated and
          their matched controls share the same ``match_id``.
        - ``match_role`` (str): ``"treated"`` or ``"control"``.
        - ``match_distance`` (float): Logit-space distance for this pair.
    """
    rows_treated: list[pd.DataFrame] = []
    rows_control: list[pd.DataFrame] = []

    for match_id, row in matched_indices.iterrows():
        t_idx = int(row["treated_idx"])
        c_idx = int(row["control_idx"])
        dist = float(row["distance"])

        t_row = df.iloc[[t_idx]].copy()
        t_row["match_id"] = match_id
        t_row["match_role"] = "treated"
        t_row["match_distance"] = dist

        c_row = df.iloc[[c_idx]].copy()
        c_row["match_id"] = match_id
        c_row["match_role"] = "control"
        c_row["match_distance"] = dist

        rows_treated.append(t_row)
        rows_control.append(c_row)

    matched_df = pd.concat(rows_treated + rows_control, ignore_index=True)
    return matched_df


def match_quality_summary(
    df: pd.DataFrame,
    matched_indices: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list[str],
) -> dict:
    """Compute a covariate balance summary for a matched sample.

    Computes standardised mean differences (SMD) for each numeric covariate
    in the matched sample, alongside aggregate matching quality statistics.

    The SMD for a numeric covariate X is:

        SMD = (mean_X_treated - mean_X_control) / sqrt((sd_t^2 + sd_c^2) / 2)

    For binary covariates (only 0/1 values), the pooled-proportion formula
    is used instead:

        SMD = (p_t - p_c) / sqrt((p_t*(1-p_t) + p_c*(1-p_c)) / 2)

    Args:
        df: Original full dataset.
        matched_indices: The ``matched_indices`` field from a
            :class:`PsmResult`.
        treatment_col: Name of the treatment indicator column.
        covariate_cols: List of covariate column names to include in the
            balance summary.  Non-numeric columns are silently skipped.

    Returns:
        Dictionary with the following keys:

        - ``"n_matched"`` (int): Total observations in the matched sample
          (treated + unique controls).
        - ``"n_matched_treated"`` (int): Unique treated units matched.
        - ``"n_matched_control"`` (int): Unique control units matched.
        - ``"n_unmatched"`` (int): Treated units without a match.
        - ``"mean_distance"`` (float): Mean logit-scale matching distance.
        - ``"max_distance"`` (float): Maximum logit-scale matching distance.
        - ``"covariate_smd"`` (dict[str, float]): Per-covariate SMD in the
          matched sample.  Values closer to 0 indicate better balance.
          Returns ``nan`` for covariates with zero pooled SD.
        - ``"n_covariates_balanced"`` (int): Number of covariates with
          |SMD| < 0.1 (the conventional "good balance" threshold).
        - ``"n_covariates_total"`` (int): Total numeric covariates evaluated.
    """
    n_treated_total = int((df[treatment_col].values.astype(int) == 1).sum())
    n_matched_treated = int(matched_indices["treated_idx"].nunique())
    n_matched_control = int(matched_indices["control_idx"].nunique())
    n_unmatched = n_treated_total - n_matched_treated

    mean_dist = float(matched_indices["distance"].mean())
    max_dist = float(matched_indices["distance"].max())

    # Gather unique treated and control positional indices
    t_indices = matched_indices["treated_idx"].unique()
    c_indices = matched_indices["control_idx"].unique()

    smd_dict: dict[str, float] = {}
    numeric_cols: list[str] = []

    for col in covariate_cols:
        if col not in df.columns:
            continue
        col_data = df[col]
        if not pd.api.types.is_numeric_dtype(col_data):
            continue
        numeric_cols.append(col)

        x_t = col_data.iloc[t_indices].values.astype(float)
        x_c = col_data.iloc[c_indices].values.astype(float)

        # Detect binary covariate
        unique_vals = np.unique(np.concatenate([x_t, x_c]))
        is_binary = set(unique_vals).issubset({0.0, 1.0, np.nan})

        if is_binary:
            p_t = float(np.nanmean(x_t))
            p_c = float(np.nanmean(x_c))
            denom = np.sqrt((p_t * (1.0 - p_t) + p_c * (1.0 - p_c)) / 2.0)
            smd = (p_t - p_c) / denom if denom > 0 else float("nan")
        else:
            mean_t = float(np.nanmean(x_t))
            mean_c = float(np.nanmean(x_c))
            sd_t = float(np.nanstd(x_t, ddof=1))
            sd_c = float(np.nanstd(x_c, ddof=1))
            pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2.0)
            smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else float("nan")

        smd_dict[col] = round(smd, 4)

    n_balanced = sum(
        1 for v in smd_dict.values() if not np.isnan(v) and abs(v) < 0.1
    )

    return {
        "n_matched": n_matched_treated + n_matched_control,
        "n_matched_treated": n_matched_treated,
        "n_matched_control": n_matched_control,
        "n_unmatched": n_unmatched,
        "mean_distance": round(mean_dist, 6),
        "max_distance": round(max_dist, 6),
        "covariate_smd": smd_dict,
        "n_covariates_balanced": n_balanced,
        "n_covariates_total": len(numeric_cols),
    }
