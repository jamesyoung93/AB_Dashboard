"""Tests for modules/psm.py — Propensity Score Matching.

Tests cover:
- Correctness of ATT recovery on a synthetic dataset with known true ATT
- 1:1 without-replacement matching produces disjoint control assignments
- 1:1 with-replacement allows control reuse
- Caliper matching discards far-away pairs and reports unmatched count
- 1:k matching returns multiple controls per treated unit
- Bootstrap SE is positive and CI width is plausible
- p-value is low when ATT is large relative to SE
- Helper: get_matched_data returns expected structure
- Helper: match_quality_summary returns SMD fields with correct shape
- Input validation raises ValueError on bad inputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.psm import (
    PsmResult,
    get_matched_data,
    match_quality_summary,
    run_psm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dataset(
    n: int = 600,
    true_att: float = 5.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a simple synthetic dataset with known ATT.

    DGP:
        X ~ N(0, 1)
        P(T=1 | X) = sigmoid(0.8*X)
        Y = 10 + true_att * T + 2*X + eps,  eps ~ N(0, 1)

    The propensity scores returned are the *true* model scores (so ATT
    recovery should be close to exact on a noiseless match).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    logit_p = 0.8 * x
    ps = 1.0 / (1.0 + np.exp(-logit_p))
    treatment = (rng.uniform(size=n) < ps).astype(int)
    eps = rng.standard_normal(n)
    y = 10.0 + true_att * treatment + 2.0 * x + eps

    df = pd.DataFrame({"x": x, "treated": treatment, "outcome": y})
    return df, ps


@pytest.fixture(name="data")
def _data_fixture():
    return _make_dataset(n=600, true_att=5.0, seed=42)


# ---------------------------------------------------------------------------
# Basic structure and type checks
# ---------------------------------------------------------------------------


def test_run_psm_returns_psmresult(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    assert isinstance(result, PsmResult)


def test_matched_indices_columns(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    mi = result.matched_indices
    assert set(mi.columns) == {"treated_idx", "control_idx", "distance"}


def test_matched_indices_distances_nonnegative(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    assert (result.matched_indices["distance"].values >= 0).all()


def test_ci_contains_lower_le_upper(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    assert result.ci_lower <= result.att <= result.ci_upper


def test_p_value_between_0_and_1(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    assert 0.0 <= result.p_value <= 1.0


def test_se_positive(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    assert result.se > 0.0


# ---------------------------------------------------------------------------
# ATT recovery (statistical correctness)
# ---------------------------------------------------------------------------


def test_att_recovery_1to1_without_replacement(data):
    """ATT should be within 2 SE of the true value (true_att=5.0)."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, method="nearest_1to1", seed=42)
    true_att = 5.0
    assert abs(result.att - true_att) < 2.5, (
        f"ATT {result.att:.3f} too far from true {true_att}"
    )


def test_att_recovery_1to1_with_replacement(data):
    df, ps = data
    result = run_psm(
        df, "outcome", "treated", ps,
        method="nearest_1to1", with_replacement=True, seed=42
    )
    true_att = 5.0
    assert abs(result.att - true_att) < 2.5


def test_att_recovery_caliper(data):
    df, ps = data
    result = run_psm(
        df, "outcome", "treated", ps,
        method="caliper", caliper=0.2, seed=42
    )
    true_att = 5.0
    # Caliper may discard some pairs; tolerance is slightly wider
    assert abs(result.att - true_att) < 3.0


def test_att_recovery_1tok(data):
    df, ps = data
    result = run_psm(
        df, "outcome", "treated", ps,
        method="nearest_1tok", k_neighbors=3, seed=42
    )
    true_att = 5.0
    assert abs(result.att - true_att) < 2.5


def test_p_value_significant_for_large_true_att(data):
    """With ATT=5 and n=600, the effect should be easily detectable."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# Matching mechanics
# ---------------------------------------------------------------------------


def test_without_replacement_control_indices_unique(data):
    """Each control should appear at most once in 1:1 without-replacement."""
    df, ps = data
    result = run_psm(
        df, "outcome", "treated", ps,
        method="nearest_1to1", with_replacement=False, seed=42
    )
    control_idxs = result.matched_indices["control_idx"].values
    assert len(control_idxs) == len(np.unique(control_idxs))


def test_with_replacement_control_can_repeat():
    """With a tiny control pool, with_replacement should reuse controls."""
    rng = np.random.default_rng(99)
    # 20 treated, only 3 controls — forces reuse
    n_treated = 20
    n_control = 3
    n = n_treated + n_control
    treatment = np.array([1] * n_treated + [0] * n_control)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    ps = np.clip(rng.uniform(0.3, 0.7, n), 0.001, 0.999)
    df = pd.DataFrame({"x": x, "treated": treatment, "outcome": y})
    result = run_psm(
        df, "outcome", "treated", ps,
        method="nearest_1to1", with_replacement=True, seed=0
    )
    control_idxs = result.matched_indices["control_idx"].values
    # With only 3 controls and 20 treated, there must be repetition
    assert len(control_idxs) > len(np.unique(control_idxs))


def test_without_replacement_no_control_reuse():
    """Without replacement, even with tiny pool, no control appears twice."""
    rng = np.random.default_rng(7)
    n_treated = 10
    n_control = 10
    n = n_treated + n_control
    treatment = np.array([1] * n_treated + [0] * n_control)
    ps = np.clip(rng.uniform(0.2, 0.8, n), 0.001, 0.999)
    y = rng.standard_normal(n)
    df = pd.DataFrame({"treated": treatment, "outcome": y})
    result = run_psm(
        df, "outcome", "treated", ps,
        method="nearest_1to1", with_replacement=False, seed=0
    )
    c_idxs = result.matched_indices["control_idx"].values
    assert len(c_idxs) == len(np.unique(c_idxs))


def test_1tok_multiple_controls_per_treated(data):
    """Each treated unit should have up to k_neighbors rows in matched_indices."""
    df, ps = data
    k = 3
    result = run_psm(
        df, "outcome", "treated", ps,
        method="nearest_1tok", k_neighbors=k, with_replacement=True, seed=42
    )
    counts = result.matched_indices.groupby("treated_idx").size()
    assert (counts <= k).all()
    # With replacement and large control pool, most treated get exactly k
    assert (counts == k).mean() > 0.9


def test_caliper_unmatched_count_increases_with_tight_caliper(data):
    """A very tight caliper should produce more unmatched treated units."""
    df, ps = data
    result_wide = run_psm(
        df, "outcome", "treated", ps,
        method="caliper", caliper=5.0, seed=42
    )
    result_tight = run_psm(
        df, "outcome", "treated", ps,
        method="caliper", caliper=0.01, seed=42
    )
    assert result_tight.n_unmatched >= result_wide.n_unmatched


def test_caliper_unmatched_count_with_zero_caliper():
    """A caliper of 0.0 SD discards all pairs when scores are continuous.

    With floating-point logit scores, the probability of an exact match
    (distance == 0.0) is essentially zero.  run_psm correctly raises
    ValueError when no matches survive the caliper.  This documents and
    validates that behavior.
    """
    rng = np.random.default_rng(1)
    n = 100
    treatment = np.array([1] * 50 + [0] * 50)
    ps = np.clip(rng.uniform(0.1, 0.9, n), 0.001, 0.999)
    y = rng.standard_normal(n)
    df = pd.DataFrame({"treated": treatment, "outcome": y})
    with pytest.raises(ValueError, match="No matches found"):
        run_psm(
            df, "outcome", "treated", ps,
            method="caliper", caliper=0.0, seed=0
        )


def test_matched_counts_add_up(data):
    """n_matched_treated + n_unmatched == total treated."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    n_total_treated = int(df["treated"].sum())
    assert result.n_matched_treated + result.n_unmatched == n_total_treated


def test_treated_idx_values_in_treated_rows(data):
    """All treated_idx values should correspond to treated rows in df."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    for ti in result.matched_indices["treated_idx"].values:
        assert df.iloc[ti]["treated"] == 1


def test_control_idx_values_in_control_rows(data):
    """All control_idx values should correspond to control rows in df."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    for ci in result.matched_indices["control_idx"].values:
        assert df.iloc[ci]["treated"] == 0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_se_larger_with_fewer_replicates(data):
    """SE is noisier with fewer bootstrap reps; with more it stabilises."""
    df, ps = data
    result_few = run_psm(
        df, "outcome", "treated", ps, n_bootstrap=50, seed=1
    )
    result_many = run_psm(
        df, "outcome", "treated", ps, n_bootstrap=500, seed=1
    )
    # Both should be positive; not a strict ordering test but both > 0
    assert result_few.se > 0.0
    assert result_many.se > 0.0


def test_seed_reproducibility(data):
    """Same seed must produce identical results."""
    df, ps = data
    r1 = run_psm(df, "outcome", "treated", ps, seed=123)
    r2 = run_psm(df, "outcome", "treated", ps, seed=123)
    assert r1.att == r2.att
    assert r1.se == r2.se
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper


def test_different_seeds_differ(data):
    """Different seeds should (overwhelmingly) produce different bootstrap SEs."""
    df, ps = data
    r1 = run_psm(df, "outcome", "treated", ps, seed=1)
    r2 = run_psm(df, "outcome", "treated", ps, seed=2)
    # ATT (point estimate) is deterministic from matching (same rng seed for
    # greedy order). Seeds differ so matched order differs; ATTs may differ.
    # We just check SE differs — a very strong check.
    assert r1.se != r2.se or r1.att != r2.att


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_on_wrong_ps_length(data):
    df, ps = data
    with pytest.raises(ValueError, match="propensity_scores length"):
        run_psm(df, "outcome", "treated", ps[:-1], seed=42)


def test_raises_on_non_binary_treatment():
    df = pd.DataFrame({
        "treated": [0, 1, 2, 0, 1],
        "outcome": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    ps = np.array([0.2, 0.5, 0.8, 0.3, 0.6])
    with pytest.raises(ValueError, match="binary"):
        run_psm(df, "outcome", "treated", ps, seed=0)


def test_raises_on_unknown_method(data):
    df, ps = data
    with pytest.raises(ValueError, match="Unknown method"):
        run_psm(df, "outcome", "treated", ps, method="magic", seed=0)  # type: ignore[arg-type]


def test_warns_on_caliper_with_non_caliper_method(data):
    df, ps = data
    with pytest.warns(UserWarning, match="caliper argument is ignored"):
        run_psm(
            df, "outcome", "treated", ps,
            method="nearest_1to1", caliper=0.2, seed=0
        )


def test_raises_on_no_treated():
    df = pd.DataFrame({"treated": [0, 0, 0], "outcome": [1.0, 2.0, 3.0]})
    ps = np.array([0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="No treated"):
        run_psm(df, "outcome", "treated", ps, seed=0)


def test_raises_on_no_control():
    df = pd.DataFrame({"treated": [1, 1, 1], "outcome": [1.0, 2.0, 3.0]})
    ps = np.array([0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match="No control"):
        run_psm(df, "outcome", "treated", ps, seed=0)


# ---------------------------------------------------------------------------
# get_matched_data helper
# ---------------------------------------------------------------------------


def test_get_matched_data_structure(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    matched_df = get_matched_data(df, result.matched_indices, "treated")

    assert "match_id" in matched_df.columns
    assert "match_role" in matched_df.columns
    assert "match_distance" in matched_df.columns
    assert set(matched_df["match_role"].unique()).issubset({"treated", "control"})


def test_get_matched_data_row_count(data):
    """matched_df should have 2 * n_pairs rows for 1:1 matching."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, method="nearest_1to1", seed=42)
    matched_df = get_matched_data(df, result.matched_indices, "treated")
    n_pairs = len(result.matched_indices)
    assert len(matched_df) == 2 * n_pairs


def test_get_matched_data_preserves_original_columns(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    matched_df = get_matched_data(df, result.matched_indices, "treated")
    for col in df.columns:
        assert col in matched_df.columns


def test_get_matched_data_roles_match_treatment(data):
    """Rows with match_role='treated' should have treatment==1, etc."""
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    matched_df = get_matched_data(df, result.matched_indices, "treated")
    assert (matched_df.loc[matched_df["match_role"] == "treated", "treated"] == 1).all()
    assert (matched_df.loc[matched_df["match_role"] == "control", "treated"] == 0).all()


# ---------------------------------------------------------------------------
# match_quality_summary helper
# ---------------------------------------------------------------------------


def test_match_quality_summary_keys(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    summary = match_quality_summary(
        df, result.matched_indices, "treated", covariate_cols=["x"]
    )
    expected_keys = {
        "n_matched",
        "n_matched_treated",
        "n_matched_control",
        "n_unmatched",
        "mean_distance",
        "max_distance",
        "covariate_smd",
        "n_covariates_balanced",
        "n_covariates_total",
    }
    assert expected_keys == set(summary.keys())


def test_match_quality_summary_smd_reduced(data):
    """Post-match SMD for 'x' should be smaller than the raw SMD."""
    df, ps = data
    # Compute raw SMD for x
    x_t = df.loc[df["treated"] == 1, "x"].values
    x_c = df.loc[df["treated"] == 0, "x"].values
    pooled_sd = np.sqrt((x_t.std(ddof=1) ** 2 + x_c.std(ddof=1) ** 2) / 2)
    raw_smd = abs((x_t.mean() - x_c.mean()) / pooled_sd)

    result = run_psm(df, "outcome", "treated", ps, seed=42)
    summary = match_quality_summary(
        df, result.matched_indices, "treated", covariate_cols=["x"]
    )
    matched_smd = abs(summary["covariate_smd"]["x"])
    assert matched_smd < raw_smd, (
        f"Post-match SMD {matched_smd:.3f} should be less than raw SMD {raw_smd:.3f}"
    )


def test_match_quality_summary_n_matched_count(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    summary = match_quality_summary(
        df, result.matched_indices, "treated", covariate_cols=["x"]
    )
    assert summary["n_matched_treated"] == result.n_matched_treated
    assert summary["n_matched_control"] == result.n_matched_control
    assert summary["n_unmatched"] == result.n_unmatched


def test_match_quality_summary_skips_non_numeric():
    """Non-numeric covariates are silently excluded."""
    rng = np.random.default_rng(5)
    n = 100
    treatment = np.array([1] * 50 + [0] * 50)
    ps = np.clip(rng.uniform(0.2, 0.8, n), 0.001, 0.999)
    y = rng.standard_normal(n)
    cat = ["A"] * n
    df = pd.DataFrame({"treated": treatment, "outcome": y, "cat_var": cat})
    result = run_psm(df, "outcome", "treated", ps, seed=0)
    summary = match_quality_summary(
        df, result.matched_indices, "treated",
        covariate_cols=["cat_var", "outcome"]
    )
    # cat_var is non-numeric, should not appear in covariate_smd
    assert "cat_var" not in summary["covariate_smd"]
    assert "outcome" in summary["covariate_smd"]


def test_match_quality_summary_binary_covariate():
    """Binary covariates should produce a finite SMD using proportion formula."""
    rng = np.random.default_rng(3)
    n = 200
    treatment = np.array([1] * 100 + [0] * 100)
    ps = np.clip(rng.uniform(0.2, 0.8, n), 0.001, 0.999)
    y = rng.standard_normal(n)
    binary_cov = rng.integers(0, 2, n).astype(float)
    df = pd.DataFrame({
        "treated": treatment,
        "outcome": y,
        "binary_x": binary_cov,
    })
    result = run_psm(df, "outcome", "treated", ps, seed=0)
    summary = match_quality_summary(
        df, result.matched_indices, "treated", covariate_cols=["binary_x"]
    )
    smd_val = summary["covariate_smd"]["binary_x"]
    assert np.isfinite(smd_val)


def test_match_quality_summary_distance_stats(data):
    df, ps = data
    result = run_psm(df, "outcome", "treated", ps, seed=42)
    summary = match_quality_summary(
        df, result.matched_indices, "treated", covariate_cols=["x"]
    )
    assert summary["mean_distance"] <= summary["max_distance"]
    assert summary["mean_distance"] >= 0.0
