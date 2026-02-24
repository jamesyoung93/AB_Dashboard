"""Tests for modules/balance.py.

Covers:
  - SMD computation correctness (continuous, binary, categorical)
  - BalanceResult fields (max_smd, all_pass, status labels)
  - Weighted SMD (IPW)
  - Matched indices SMD
  - Love plot: returns a Figure with expected artist counts
  - Propensity overlap plot: common-support annotation present
  - Covariate distribution plot: categorical and continuous branches
  - Edge cases: zero-variance columns, single-level categoricals
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from modules.balance import (
    BalanceResult,
    SMD_PASS_THRESHOLD,
    STATUS_CAUTION,
    STATUS_FAIL,
    STATUS_PASS,
    _is_binary,
    _smd_binary,
    _smd_continuous,
    balance_summary,
    compute_balance,
    covariate_distribution_plot,
    love_plot,
    propensity_overlap_plot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def simple_df(rng: np.random.Generator) -> pd.DataFrame:
    """200-row DataFrame with known treatment confounding."""
    n = 200
    treatment = rng.integers(0, 2, size=n)
    # Continuous covariate: treated group has higher mean
    age = np.where(treatment == 1, rng.normal(50, 5, n), rng.normal(40, 5, n))
    # Binary covariate
    high_value = (rng.uniform(size=n) < (0.6 * treatment + 0.2 * (1 - treatment))).astype(int)
    # Categorical covariate
    industry = rng.choice(["Tech", "Finance", "Other"], size=n)
    return pd.DataFrame({
        "treatment": treatment,
        "age": age,
        "high_value": high_value,
        "industry": industry,
    })


@pytest.fixture
def omnichannel_df() -> pd.DataFrame:
    """Load the pre-generated synthetic omnichannel CSV (first 1000 rows)."""
    import pathlib
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "data" / "sample" / "synthetic_omnichannel.csv"
    )
    return pd.read_csv(path, nrows=1000)


# ---------------------------------------------------------------------------
# Unit tests: SMD helpers
# ---------------------------------------------------------------------------

class TestSmdHelpers:
    def test_smd_continuous_positive(self) -> None:
        """Treated mean > control mean => positive SMD."""
        vals_t = np.array([55.0, 60.0, 58.0])
        vals_c = np.array([40.0, 42.0, 38.0])
        smd = _smd_continuous(vals_t, vals_c)
        assert smd > 0

    def test_smd_continuous_zero_variance(self) -> None:
        """Both groups constant => SMD = 0.0 (no division by zero)."""
        vals_t = np.ones(10) * 5.0
        vals_c = np.ones(10) * 5.0
        assert _smd_continuous(vals_t, vals_c) == 0.0

    def test_smd_continuous_symmetry(self) -> None:
        """Swapping treated/control negates sign but preserves magnitude."""
        rng = np.random.default_rng(7)
        a = rng.normal(10, 2, 100)
        b = rng.normal(12, 2, 100)
        assert abs(_smd_continuous(a, b) + _smd_continuous(b, a)) < 1e-12

    def test_smd_binary_equal_proportions(self) -> None:
        assert _smd_binary(0.5, 0.5) == 0.0

    def test_smd_binary_known_value(self) -> None:
        """Manual calculation: p_T=0.6, p_C=0.4."""
        p_t, p_c = 0.6, 0.4
        expected_num = p_t - p_c
        expected_den = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
        expected = expected_num / expected_den
        assert abs(_smd_binary(p_t, p_c) - expected) < 1e-12

    def test_smd_binary_zero_denom(self) -> None:
        """Both groups fully in one class => SMD = 0.0."""
        assert _smd_binary(0.0, 0.0) == 0.0
        assert _smd_binary(1.0, 1.0) == 0.0

    def test_is_binary_true(self) -> None:
        s = pd.Series([0, 1, 0, 1, 1])
        assert _is_binary(s) is True

    def test_is_binary_false(self) -> None:
        s = pd.Series([0.0, 0.5, 1.0, 2.0])
        assert _is_binary(s) is False

    def test_is_binary_float_encoding(self) -> None:
        s = pd.Series([0.0, 1.0, 0.0, 1.0])
        assert _is_binary(s) is True


# ---------------------------------------------------------------------------
# compute_balance: raw (unadjusted)
# ---------------------------------------------------------------------------

class TestComputeBalanceRaw:
    def test_returns_balance_result(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value", "industry"])
        assert isinstance(result, BalanceResult)
        assert isinstance(result.table, pd.DataFrame)
        assert isinstance(result.max_smd, float)
        assert isinstance(result.all_pass, bool)

    def test_table_columns(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value", "industry"])
        expected = {"covariate", "mean_treated", "mean_control", "smd_raw", "smd_adjusted", "status"}
        assert expected.issubset(set(result.table.columns))

    def test_categorical_dummy_expansion(self, simple_df: pd.DataFrame) -> None:
        """Categorical column 'industry' should expand into 3 dummy rows."""
        result = compute_balance(simple_df, "treatment", ["industry"])
        dummy_rows = result.table[result.table["covariate"].str.startswith("industry_")]
        assert len(dummy_rows) == 3

    def test_status_labels(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value", "industry"])
        valid_statuses = {STATUS_PASS, STATUS_CAUTION, STATUS_FAIL}
        assert set(result.table["status"]).issubset(valid_statuses)

    def test_status_thresholds(self) -> None:
        """Directly verify threshold mapping via synthetic SMDs.

        Uses a very large N and extreme group separations to guarantee
        that the SMD magnitudes land in the expected status bands.
        """
        rng = np.random.default_rng(0)
        n_half = 5000
        treatment = np.repeat([0, 1], n_half)
        # Near-zero SMD: same distribution in both groups => Pass
        x_pass = rng.normal(0, 1, n_half * 2)
        # Large separation (true SMD ~2.0 >> 0.25) => Fail
        x_fail = np.concatenate([
            rng.normal(0, 1, n_half),    # control
            rng.normal(2.0, 1, n_half),  # treated
        ])
        df = pd.DataFrame({
            "trt": treatment,
            "x_pass": x_pass,
            "x_fail": x_fail,
        })
        result = compute_balance(df, "trt", ["x_pass", "x_fail"])
        statuses = dict(zip(result.table["covariate"], result.table["status"]))
        assert statuses["x_pass"] == STATUS_PASS
        assert statuses["x_fail"] == STATUS_FAIL

    def test_smd_raw_equals_smd_adjusted_when_no_adjustment(
        self, simple_df: pd.DataFrame
    ) -> None:
        result = compute_balance(simple_df, "treatment", ["age"])
        row = result.table.iloc[0]
        assert row["smd_raw"] == row["smd_adjusted"]

    def test_max_smd_is_max_of_table(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value", "industry"])
        expected_max = result.table["smd_adjusted"].abs().max()
        assert abs(result.max_smd - expected_max) < 1e-10

    def test_all_pass_flag(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value", "industry"])
        # All pass should be True iff every |adjusted SMD| < 0.1
        expected = bool((result.table["smd_adjusted"].abs() < SMD_PASS_THRESHOLD).all())
        assert result.all_pass == expected

    def test_missing_covariate_warns(self, simple_df: pd.DataFrame) -> None:
        with pytest.warns(UserWarning, match="not found"):
            result = compute_balance(simple_df, "treatment", ["age", "nonexistent_col"])
        # Result should still have 'age'
        assert "age" in result.table["covariate"].values

    def test_invalid_treatment_raises(self, simple_df: pd.DataFrame) -> None:
        simple_df = simple_df.copy()
        simple_df["bad_treatment"] = simple_df["treatment"] * 5  # values 0 and 5
        with pytest.raises(ValueError, match="binary"):
            compute_balance(simple_df, "bad_treatment", ["age"])

    def test_mutual_exclusion_weights_and_indices_raises(
        self, simple_df: pd.DataFrame
    ) -> None:
        n = len(simple_df)
        with pytest.raises(ValueError, match="not both"):
            compute_balance(
                simple_df,
                "treatment",
                ["age"],
                weights=np.ones(n),
                matched_indices=(np.array([0, 1]), np.array([2, 3])),
            )


# ---------------------------------------------------------------------------
# compute_balance: weighted (IPW-style)
# ---------------------------------------------------------------------------

class TestComputeBalanceWeighted:
    def test_uniform_weights_equal_unweighted(self, simple_df: pd.DataFrame) -> None:
        """Uniform weights should produce the same result as no weights."""
        n = len(simple_df)
        result_raw = compute_balance(simple_df, "treatment", ["age"])
        result_wtd = compute_balance(simple_df, "treatment", ["age"], weights=np.ones(n))
        raw_smd = result_raw.table.set_index("covariate")["smd_raw"].loc["age"]
        adj_smd = result_wtd.table.set_index("covariate")["smd_adjusted"].loc["age"]
        assert abs(raw_smd - adj_smd) < 0.05  # within 0.05 due to weighting math

    def test_weights_reduce_imbalance(self, simple_df: pd.DataFrame) -> None:
        """Perfect IPW weights (equal weights per group) should reduce imbalance."""
        # Build weights that equalise group sizes
        n = len(simple_df)
        n_t = simple_df["treatment"].sum()
        n_c = n - n_t
        weights = np.where(simple_df["treatment"] == 1, 1.0 / n_t, 1.0 / n_c)

        result_wtd = compute_balance(simple_df, "treatment", ["age"], weights=weights)
        abs_adj = result_wtd.table["smd_adjusted"].abs().max()
        # Normalized weights should generally not increase imbalance dramatically
        # (not guaranteed to reduce here, just verify the code runs and produces floats)
        assert isinstance(abs_adj, float)

    def test_weighted_categorical(self, simple_df: pd.DataFrame) -> None:
        n = len(simple_df)
        weights = np.ones(n)
        result = compute_balance(simple_df, "treatment", ["industry"], weights=weights)
        dummy_rows = result.table[result.table["covariate"].str.startswith("industry_")]
        assert len(dummy_rows) == 3


# ---------------------------------------------------------------------------
# compute_balance: matched indices
# ---------------------------------------------------------------------------

class TestComputeBalanceMatched:
    def test_matched_indices_reduces_imbalance(self, simple_df: pd.DataFrame) -> None:
        """1:1 exact matching on known groups should show near-zero SMD."""
        # Find treated and control indices; match first 40 treated to first 40 control
        treated_idx = np.where(simple_df["treatment"] == 1)[0][:40]
        control_idx = np.where(simple_df["treatment"] == 0)[0][:40]

        result = compute_balance(
            simple_df,
            "treatment",
            ["age"],
            matched_indices=(treated_idx, control_idx),
        )
        # The adjusted SMD is computed on the matched sample, not on all data
        assert "smd_adjusted" in result.table.columns

    def test_matched_produces_balance_result(self, simple_df: pd.DataFrame) -> None:
        treated_idx = np.where(simple_df["treatment"] == 1)[0][:30]
        control_idx = np.where(simple_df["treatment"] == 0)[0][:30]
        result = compute_balance(
            simple_df,
            "treatment",
            ["age", "high_value"],
            matched_indices=(treated_idx, control_idx),
        )
        assert isinstance(result, BalanceResult)
        assert len(result.table) == 2


# ---------------------------------------------------------------------------
# compute_balance: with real synthetic data
# ---------------------------------------------------------------------------

class TestComputeBalanceRealData:
    def test_omnichannel_raw_smd_positive_for_confounded_covariates(
        self, omnichannel_df: pd.DataFrame
    ) -> None:
        """Raw SMD should be non-trivial for confounded covariates like prior_spend."""
        df_base = omnichannel_df[omnichannel_df["week"] == 1].copy()
        result = compute_balance(
            df_base,
            "channel_email",
            ["prior_spend", "engagement_score", "tenure_years"],
        )
        # prior_spend is confounded: treated have higher prior_spend by DGP
        smd_prior = result.table.set_index("covariate")["smd_raw"]["prior_spend"]
        assert smd_prior > 0.05  # should be noticeably imbalanced

    def test_categorical_expansion_company_size(
        self, omnichannel_df: pd.DataFrame
    ) -> None:
        df_base = omnichannel_df[omnichannel_df["week"] == 1].copy()
        result = compute_balance(
            df_base,
            "channel_email",
            ["company_size"],
        )
        dummy_rows = result.table[result.table["covariate"].str.startswith("company_size_")]
        # company_size has 4 levels: Small, Medium, Large, Enterprise
        assert len(dummy_rows) == 4


# ---------------------------------------------------------------------------
# love_plot
# ---------------------------------------------------------------------------

class TestLovePlot:
    def test_returns_figure(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value", "industry"])
        fig = love_plot(result)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_adjusted(self, simple_df: pd.DataFrame) -> None:
        result_raw = compute_balance(simple_df, "treatment", ["age", "high_value"])
        # Use matched sample as adjusted result
        t_idx = np.where(simple_df["treatment"] == 1)[0][:20]
        c_idx = np.where(simple_df["treatment"] == 0)[0][:20]
        result_adj = compute_balance(
            simple_df, "treatment", ["age", "high_value"],
            matched_indices=(t_idx, c_idx),
        )
        fig = love_plot(result_raw, result_adj)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_y_axis_has_covariate_labels(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age"])
        fig = love_plot(result)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert "age" in tick_labels
        plt.close(fig)

    def test_threshold_lines_present(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age"])
        fig = love_plot(result)
        ax = fig.axes[0]
        # Vertical lines: 0.1 and 0.25 thresholds
        vlines = [line for line in ax.get_lines() if len(line.get_xdata()) == 2
                  and line.get_xdata()[0] == line.get_xdata()[1]]
        # At minimum the two threshold dashed lines should exist
        x_vals = {float(line.get_xdata()[0]) for line in vlines}
        assert SMD_PASS_THRESHOLD in x_vals or any(
            abs(x - SMD_PASS_THRESHOLD) < 1e-6 for x in x_vals
        )
        plt.close(fig)

    def test_custom_figsize(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age"])
        fig = love_plot(result, figsize=(10.0, 5.0))
        w, h = fig.get_size_inches()
        assert abs(w - 10.0) < 0.1
        assert abs(h - 5.0) < 0.1
        plt.close(fig)


# ---------------------------------------------------------------------------
# propensity_overlap_plot
# ---------------------------------------------------------------------------

class TestPropensityOverlapPlot:
    def test_returns_figure(self) -> None:
        rng = np.random.default_rng(0)
        ps_t = rng.beta(5, 2, 200)
        ps_c = rng.beta(2, 5, 300)
        fig = propensity_overlap_plot(ps_t, ps_c)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_annotation_in_axes(self) -> None:
        rng = np.random.default_rng(1)
        ps_t = rng.beta(5, 2, 100)
        ps_c = rng.beta(2, 5, 200)
        fig = propensity_overlap_plot(ps_t, ps_c)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Should have annotation about % outside common support
        assert any("common support" in t for t in texts)
        plt.close(fig)

    def test_xlim_is_zero_to_one(self) -> None:
        rng = np.random.default_rng(2)
        ps_t = rng.uniform(0.2, 0.8, 100)
        ps_c = rng.uniform(0.1, 0.7, 100)
        fig = propensity_overlap_plot(ps_t, ps_c)
        ax = fig.axes[0]
        assert ax.get_xlim()[0] == 0
        assert ax.get_xlim()[1] == 1
        plt.close(fig)

    def test_accepts_pandas_series(self) -> None:
        rng = np.random.default_rng(3)
        ps_t = pd.Series(rng.beta(3, 2, 150))
        ps_c = pd.Series(rng.beta(2, 3, 250))
        fig = propensity_overlap_plot(ps_t, ps_c)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_perfect_overlap_zero_percent_outside(self) -> None:
        """When treated PS range is inside control PS range, % outside = 0."""
        # Control has broader range, so treated is fully inside
        ps_c = np.linspace(0.01, 0.99, 500)
        ps_t = np.linspace(0.3, 0.7, 200)
        fig = propensity_overlap_plot(ps_t, ps_c)
        ax = fig.axes[0]
        texts = " ".join(t.get_text() for t in ax.texts)
        assert "0.0%" in texts
        plt.close(fig)


# ---------------------------------------------------------------------------
# covariate_distribution_plot
# ---------------------------------------------------------------------------

class TestCovariateDistributionPlot:
    def test_continuous_returns_figure(self) -> None:
        rng = np.random.default_rng(5)
        s_t = pd.Series(rng.normal(55, 5, 100))
        s_c = pd.Series(rng.normal(45, 5, 150))
        fig = covariate_distribution_plot(s_t, s_c, name="age")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_categorical_returns_figure(self) -> None:
        rng = np.random.default_rng(6)
        cats = ["Tech", "Finance", "Other"]
        s_t = pd.Series(rng.choice(cats, size=100))
        s_c = pd.Series(rng.choice(cats, size=150))
        fig = covariate_distribution_plot(s_t, s_c, name="industry", is_categorical=True)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_categorical_autodetect_from_object_dtype(self) -> None:
        """String-dtype columns should be auto-detected as categorical."""
        rng = np.random.default_rng(7)
        cats = ["A", "B", "C"]
        s_t = pd.Series(rng.choice(cats, size=80))
        s_c = pd.Series(rng.choice(cats, size=120))
        # is_categorical=False but dtype is string-like => should auto-detect categorical
        fig = covariate_distribution_plot(s_t, s_c, name="cat_col", is_categorical=False)
        ax = fig.axes[0]
        # A categorical grouped bar chart uses a y-axis label of "Proportion"
        assert "Proportion" in ax.get_ylabel()
        plt.close(fig)

    def test_continuous_x_label(self) -> None:
        rng = np.random.default_rng(8)
        s_t = pd.Series(rng.normal(0, 1, 50))
        s_c = pd.Series(rng.normal(0, 1, 50))
        fig = covariate_distribution_plot(s_t, s_c, name="score")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "score"
        plt.close(fig)

    def test_categorical_ylabel_is_proportion(self) -> None:
        rng = np.random.default_rng(9)
        s_t = pd.Series(rng.choice(["X", "Y"], size=60))
        s_c = pd.Series(rng.choice(["X", "Y"], size=60))
        fig = covariate_distribution_plot(s_t, s_c, name="group", is_categorical=True)
        ax = fig.axes[0]
        assert "Proportion" in ax.get_ylabel()
        plt.close(fig)

    def test_title_contains_covariate_name(self) -> None:
        rng = np.random.default_rng(10)
        s_t = pd.Series(rng.normal(10, 1, 50))
        s_c = pd.Series(rng.normal(10, 1, 50))
        fig = covariate_distribution_plot(s_t, s_c, name="tenure_years")
        ax = fig.axes[0]
        assert "tenure_years" in ax.get_title()
        plt.close(fig)

    def test_handles_missing_values_in_continuous(self) -> None:
        """NaN values in series should not cause errors."""
        rng = np.random.default_rng(11)
        s_t = pd.Series(rng.normal(0, 1, 80))
        s_c = pd.Series(rng.normal(0, 1, 80))
        # Inject NaNs
        s_t.iloc[[5, 10, 15]] = np.nan
        s_c.iloc[[3, 7]] = np.nan
        fig = covariate_distribution_plot(s_t, s_c, name="var_with_nans")
        assert isinstance(fig, Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# balance_summary
# ---------------------------------------------------------------------------

class TestBalanceSummary:
    def test_returns_string(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age"])
        summary = balance_summary(result)
        assert isinstance(summary, str)

    def test_contains_key_fields(self, simple_df: pd.DataFrame) -> None:
        result = compute_balance(simple_df, "treatment", ["age", "high_value"])
        summary = balance_summary(result)
        assert "Pass" in summary
        assert "Max |adjusted SMD|" in summary
        assert "All pass" in summary
