"""Tests for the IPW module (modules/ipw.py)."""

import numpy as np
import pandas as pd
import pytest

from modules.ipw import IpwResult, compute_ipw_weights, run_ipw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Small dataset with known treatment effect."""
    rng = np.random.default_rng(42)
    n = 200
    treated = np.array([1] * 80 + [0] * 120)
    # Confounded: treated have higher baseline
    x = rng.normal(0, 1, n) + 0.5 * treated
    # Outcome with true ATT = 3.0
    y = 10 + 3.0 * treated + 2.0 * x + rng.normal(0, 2, n)
    return pd.DataFrame({
        "treatment": treated,
        "outcome": y,
        "covariate": x,
    })


@pytest.fixture
def propensity_scores(simple_df: pd.DataFrame) -> np.ndarray:
    """Pre-compute propensity scores for simple_df."""
    from modules.propensity import fit_propensity
    result = fit_propensity(
        simple_df, "treatment", ["covariate"], method="logistic",
    )
    return result.scores


# ---------------------------------------------------------------------------
# compute_ipw_weights tests
# ---------------------------------------------------------------------------


class TestComputeIpwWeights:
    def test_att_weights_treated_are_one(self) -> None:
        ps = np.array([0.3, 0.7, 0.5, 0.2])
        t = np.array([1, 1, 0, 0])
        weights, _ = compute_ipw_weights(ps, t, estimand="ATT", stabilized=False)
        assert weights[0] == 1.0
        assert weights[1] == 1.0

    def test_att_weights_control_formula(self) -> None:
        ps = np.array([0.3, 0.5])
        t = np.array([0, 0])
        weights, _ = compute_ipw_weights(ps, t, estimand="ATT", stabilized=False)
        # Control weight = ps / (1 - ps)
        assert abs(weights[0] - 0.3 / 0.7) < 0.01
        assert abs(weights[1] - 0.5 / 0.5) < 0.01

    def test_ate_weights_formula(self) -> None:
        ps = np.array([0.4, 0.6])
        t = np.array([1, 0])
        weights, _ = compute_ipw_weights(ps, t, estimand="ATE", stabilized=False)
        assert abs(weights[0] - 1.0 / 0.4) < 0.01  # treated: 1/ps
        assert abs(weights[1] - 1.0 / 0.4) < 0.01  # control: 1/(1-ps)

    def test_stabilized_att_normalizes_control(self) -> None:
        rng = np.random.default_rng(99)
        n = 100
        t = np.array([1] * 40 + [0] * 60)
        ps = np.clip(rng.uniform(0.2, 0.8, n), 0.05, 0.95)
        weights, _ = compute_ipw_weights(ps, t, estimand="ATT", stabilized=True)
        # Stabilized: control weights sum to N_treated
        assert abs(weights[t == 0].sum() - 40) < 0.01

    def test_trimming_clips_extreme_ps(self) -> None:
        ps = np.array([0.001, 0.999, 0.5, 0.5])
        t = np.array([1, 0, 1, 0])
        weights_no_trim, s1 = compute_ipw_weights(ps, t, trim_percentile=None)
        weights_trimmed, s2 = compute_ipw_weights(ps, t, trim_percentile=0.1)
        # Trimmed max weight should be smaller
        assert s2["max"] <= s1["max"]
        assert s2["pct_trimmed"] > 0

    def test_summary_has_required_keys(self) -> None:
        ps = np.array([0.3, 0.7])
        t = np.array([1, 0])
        _, summary = compute_ipw_weights(ps, t)
        assert "min" in summary
        assert "max" in summary
        assert "ess_treated" in summary
        assert "ess_control" in summary
        assert "pct_trimmed" in summary

    def test_ess_less_than_or_equal_n(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        t = np.array([1] * 50 + [0] * 50)
        ps = np.clip(rng.uniform(0.2, 0.8, n), 0.05, 0.95)
        _, summary = compute_ipw_weights(ps, t, estimand="ATT")
        assert summary["ess_treated"] <= 50
        assert summary["ess_control"] <= 50


# ---------------------------------------------------------------------------
# run_ipw tests
# ---------------------------------------------------------------------------


class TestRunIpw:
    def test_returns_ipw_result(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            n_bootstrap=100, seed=42,
        )
        assert isinstance(result, IpwResult)

    def test_att_estimate_reasonable(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            estimand="ATT", n_bootstrap=200, seed=42,
        )
        # True ATT = 3.0, should be within 3.0
        assert abs(result.estimate - 3.0) < 3.0, (
            f"IPW ATT {result.estimate:.3f} too far from 3.0"
        )

    def test_ate_estimate_positive(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            estimand="ATE", n_bootstrap=100, seed=42,
        )
        assert result.estimate > 0, "ATE should be positive"

    def test_confidence_interval_contains_point(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            n_bootstrap=200, seed=42,
        )
        assert result.ci_lower <= result.estimate <= result.ci_upper

    def test_se_positive(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            n_bootstrap=100, seed=42,
        )
        assert result.se > 0

    def test_p_value_between_0_and_1(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            n_bootstrap=100, seed=42,
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_unit_counts(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            n_bootstrap=50, seed=42,
        )
        assert result.n_treated == 80
        assert result.n_control == 120

    def test_stabilized_vs_unstabilized(self, simple_df, propensity_scores) -> None:
        res_stab = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            stabilized=True, n_bootstrap=100, seed=42,
        )
        res_unstab = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            stabilized=False, n_bootstrap=100, seed=42,
        )
        assert res_stab.method == "ipw_stabilized"
        assert res_unstab.method == "ipw"
        # Both should give reasonable estimates
        assert abs(res_stab.estimate - 3.0) < 5.0
        assert abs(res_unstab.estimate - 3.0) < 5.0

    def test_trimming_works(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            trim_percentile=0.05, n_bootstrap=50, seed=42,
        )
        assert result.weights_summary["pct_trimmed"] > 0

    def test_length_mismatch_raises(self, simple_df) -> None:
        bad_ps = np.array([0.5, 0.5])  # wrong length
        with pytest.raises(ValueError, match="length"):
            run_ipw(simple_df, "outcome", "treatment", bad_ps)

    def test_non_binary_treatment_raises(self) -> None:
        df = pd.DataFrame({
            "treatment": [0, 1, 2, 3],
            "outcome": [1.0, 2.0, 3.0, 4.0],
        })
        ps = np.array([0.3, 0.5, 0.7, 0.9])
        with pytest.raises(ValueError, match="binary"):
            run_ipw(df, "outcome", "treatment", ps)

    def test_weights_summary_in_result(self, simple_df, propensity_scores) -> None:
        result = run_ipw(
            simple_df, "outcome", "treatment", propensity_scores,
            n_bootstrap=50, seed=42,
        )
        ws = result.weights_summary
        assert isinstance(ws, dict)
        assert ws["ess_treated"] > 0
        assert ws["ess_control"] > 0


# ---------------------------------------------------------------------------
# Integration: IPW on sample data
# ---------------------------------------------------------------------------


class TestIpwSampleData:
    @pytest.fixture(scope="class")
    def omni_df(self) -> pd.DataFrame:
        from pathlib import Path
        data_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
        return pd.read_csv(data_dir / "synthetic_omnichannel.csv")

    @pytest.fixture(scope="class")
    def post_customer_df(self, omni_df: pd.DataFrame) -> pd.DataFrame:
        post = omni_df[omni_df["week"] > 10].copy()
        return post.groupby("customer_id").agg(
            revenue=("revenue", "mean"),
            channel_email=("channel_email", "first"),
            prior_spend=("prior_spend", "first"),
            tenure_years=("tenure_years", "first"),
            company_size=("company_size", "first"),
            industry=("industry", "first"),
            engagement_score=("engagement_score", "first"),
            channel_webinar=("channel_webinar", "mean"),
            sales_rep_touches=("sales_rep_touches", "mean"),
            social_paid_impressions=("social_paid_impressions", "mean"),
        ).reset_index()

    @pytest.fixture(scope="class")
    def propensity_sample(self, post_customer_df: pd.DataFrame):
        from modules.propensity import fit_propensity
        return fit_propensity(
            post_customer_df, "channel_email",
            ["prior_spend", "tenure_years", "company_size",
             "industry", "engagement_score", "channel_webinar",
             "sales_rep_touches"],
            method="logistic",
        )

    def test_ipw_att_positive(self, post_customer_df, propensity_sample) -> None:
        result = run_ipw(
            post_customer_df, "revenue", "channel_email",
            propensity_sample.scores, estimand="ATT",
            n_bootstrap=200, seed=42,
        )
        assert result.estimate > 0, "IPW ATT should be positive"

    def test_ipw_att_reasonable(self, post_customer_df, propensity_sample) -> None:
        result = run_ipw(
            post_customer_df, "revenue", "channel_email",
            propensity_sample.scores, estimand="ATT",
            n_bootstrap=200, seed=42,
        )
        # True ATT = 3.5; IPW can have some bias, so wider tolerance
        assert abs(result.estimate - 3.5) < 5.0, (
            f"IPW ATT {result.estimate:.3f} too far from 3.5"
        )

    def test_ipw_significant(self, post_customer_df, propensity_sample) -> None:
        result = run_ipw(
            post_customer_df, "revenue", "channel_email",
            propensity_sample.scores, estimand="ATT",
            n_bootstrap=500, seed=42,
        )
        assert result.p_value < 0.10, "IPW should detect significant effect"
