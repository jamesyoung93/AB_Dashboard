"""Tests for modules/propensity.py and modules/threshold.py.

Tests are grouped into two classes:
    TestPropensity  — fit_propensity, PropensityResult, _encode_covariates
    TestThreshold   — compute_threshold_stats, suggest_thresholds,
                      binarize_treatment

All tests use a fixed seed (42) for reproducibility.  Tests are
deliberately implementation-agnostic: they check contracts (output
shapes, value ranges, types) rather than exact numeric values, with
a handful of directional correctness checks on known DGPs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from modules.propensity import (
    PropensityResult,
    _encode_covariates,
    fit_propensity,
)
from modules.threshold import (
    ThresholdStats,
    ThresholdSuggestion,
    binarize_treatment,
    compute_threshold_stats,
    suggest_thresholds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def simple_df(rng: np.random.Generator) -> pd.DataFrame:
    """200-row DataFrame with numeric and categorical covariates."""
    n = 200
    return pd.DataFrame(
        {
            "treated": rng.integers(0, 2, n).astype(int),
            "x_num": rng.normal(0, 1, n),
            "x_cat": rng.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture(scope="module")
def confounded_df(rng: np.random.Generator) -> pd.DataFrame:
    """500-row DataFrame where treatment is strongly predicted by covariates."""
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.5 * x1 + 1.5 * x2
    p = 1 / (1 + np.exp(-logit))
    treated = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"treated": treated, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def exponential_series(rng: np.random.Generator) -> pd.Series:
    """Right-skewed series (exponential) — common for touch counts."""
    return pd.Series(rng.exponential(scale=3.0, size=500))


@pytest.fixture(scope="module")
def bimodal_series() -> pd.Series:
    """Bimodal series: mass at ~0 and a second mode at ~10."""
    rng = np.random.default_rng(7)
    low = rng.normal(0.5, 0.3, 300)
    high = rng.normal(10.0, 1.5, 200)
    return pd.Series(np.concatenate([low, high]))


# ---------------------------------------------------------------------------
# TestPropensity
# ---------------------------------------------------------------------------


class TestPropensity:
    """Tests for modules/propensity.fit_propensity and helpers."""

    # --- Output structure ---

    def test_returns_propensity_result(self, simple_df: pd.DataFrame) -> None:
        res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"])
        assert isinstance(res, PropensityResult)

    def test_logistic_fields(self, simple_df: pd.DataFrame) -> None:
        res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], method="logistic")
        assert res.method == "logistic"
        assert res.scaler is not None  # StandardScaler present for logistic
        assert isinstance(res.auc, float)
        assert isinstance(res.feature_names, list)
        assert len(res.feature_names) > 0

    def test_gbm_fields(self, simple_df: pd.DataFrame) -> None:
        res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], method="gbm")
        assert res.method == "gbm"
        assert res.scaler is None  # No scaler for scale-invariant GBM

    def test_scores_shape_matches_input(self, simple_df: pd.DataFrame) -> None:
        res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"])
        assert res.scores.shape == (len(simple_df),)

    def test_scores_are_valid_probabilities(self, simple_df: pd.DataFrame) -> None:
        res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"])
        assert np.all(res.scores >= 0.0)
        assert np.all(res.scores <= 1.0)
        assert not np.any(np.isnan(res.scores))

    def test_auc_in_valid_range(self, simple_df: pd.DataFrame) -> None:
        for method in ("logistic", "gbm"):
            res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], method=method)
            assert 0.0 <= res.auc <= 1.0, f"AUC out of range for {method}: {res.auc}"

    # --- Categorical encoding ---

    def test_feature_names_after_dummy_encoding(self, simple_df: pd.DataFrame) -> None:
        res = fit_propensity(simple_df, "treated", ["x_num", "x_cat"])
        # x_cat has 3 levels (A, B, C); drop_first leaves 2 dummies
        # Feature list: x_num + x_cat_B + x_cat_C  (or similar names)
        assert "x_num" in res.feature_names
        # At least 2 dummies from x_cat
        cat_dummies = [f for f in res.feature_names if f.startswith("x_cat_")]
        assert len(cat_dummies) == 2

    def test_all_numeric_covariates(self, confounded_df: pd.DataFrame) -> None:
        """No encoding needed; feature names should equal covariate names."""
        res = fit_propensity(confounded_df, "treated", ["x1", "x2"])
        assert set(res.feature_names) == {"x1", "x2"}

    # --- AUC directional correctness ---

    def test_gbm_auc_higher_than_random_on_confounded_data(
        self, confounded_df: pd.DataFrame
    ) -> None:
        """On data where treatment is strongly predicted, AUC should be well above 0.5."""
        res = fit_propensity(confounded_df, "treated", ["x1", "x2"], method="gbm")
        assert res.auc > 0.70, f"Expected AUC > 0.70, got {res.auc:.3f}"

    def test_logistic_recovers_signal_on_confounded_data(
        self, confounded_df: pd.DataFrame
    ) -> None:
        res = fit_propensity(confounded_df, "treated", ["x1", "x2"], method="logistic")
        assert res.auc > 0.65, f"Expected AUC > 0.65, got {res.auc:.3f}"

    # --- Seed reproducibility ---

    def test_same_seed_gives_identical_scores(self, simple_df: pd.DataFrame) -> None:
        res1 = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], seed=99)
        res2 = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], seed=99)
        np.testing.assert_array_equal(res1.scores, res2.scores)

    def test_different_seeds_may_give_different_gbm_scores(
        self, simple_df: pd.DataFrame
    ) -> None:
        """GBM has stochastic subsampling so different seeds should differ."""
        res1 = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], method="gbm", seed=1)
        res2 = fit_propensity(simple_df, "treated", ["x_num", "x_cat"], method="gbm", seed=2)
        # Not guaranteed to differ but extremely likely with subsample=0.8
        # Use a soft check: they shouldn't be bit-identical
        assert not np.array_equal(res1.scores, res2.scores), (
            "Scores from different seeds are identical — seed control may be broken."
        )

    # --- Missing value handling ---

    def test_missing_covariates_imputed_without_error(self, rng: np.random.Generator) -> None:
        n = 100
        df = pd.DataFrame(
            {
                "treated": rng.integers(0, 2, n).astype(int),
                "x_num": rng.normal(0, 1, n),
            }
        )
        # Introduce 10% missing values
        miss_idx = rng.choice(n, size=10, replace=False)
        df.loc[miss_idx, "x_num"] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_propensity(df, "treated", ["x_num"])
        assert res.scores.shape == (n,)
        assert not np.any(np.isnan(res.scores))

    def test_missing_treatment_rows_dropped(self, rng: np.random.Generator) -> None:
        n = 100
        df = pd.DataFrame(
            {
                "treated": rng.integers(0, 2, n).astype(float),
                "x": rng.normal(0, 1, n),
            }
        )
        df.loc[0:4, "treated"] = np.nan  # 5 missing treatment rows

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_propensity(df, "treated", ["x"])
        # Scores should match non-missing treatment rows
        assert res.scores.shape == (95,)

    # --- Error cases ---

    def test_raises_on_empty_covariate_list(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="covariate_cols must contain"):
            fit_propensity(simple_df, "treated", [])

    def test_raises_on_invalid_method(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="method must be"):
            fit_propensity(simple_df, "treated", ["x_num"], method="rf")  # type: ignore[arg-type]

    def test_raises_on_non_binary_treatment(self, rng: np.random.Generator) -> None:
        df = pd.DataFrame(
            {
                "treated": rng.integers(0, 5, 100),  # 5 classes — invalid
                "x": rng.normal(0, 1, 100),
            }
        )
        with pytest.raises(ValueError, match="must be binary"):
            fit_propensity(df, "treated", ["x"])

    def test_raises_on_missing_column(self, simple_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            fit_propensity(simple_df, "treated", ["does_not_exist"])

    # --- _encode_covariates unit tests ---

    def test_encode_covariates_numeric_only(self, rng: np.random.Generator) -> None:
        df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50)})
        X, names = _encode_covariates(df)
        assert X.shape == (50, 2)
        assert names == ["a", "b"]
        assert X.dtype == np.float64

    def test_encode_covariates_categorical(self) -> None:
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0],
                "cat": ["X", "Y", "X", "Z"],
            }
        )
        X, names = _encode_covariates(df)
        # cat has 3 levels -> 2 dummies with drop_first
        assert X.shape[1] == 3  # num + cat_Y + cat_Z (or cat_X + cat_Z depending on order)
        assert "num" in names

    def test_encode_covariates_missing_numeric_imputed_with_mean(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, _ = _encode_covariates(df)
        # Imputed value should be mean of non-missing: (1+2+4+5)/4 = 3.0
        assert X[2, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# TestThreshold
# ---------------------------------------------------------------------------


class TestThreshold:
    """Tests for modules/threshold functions."""

    # --- compute_threshold_stats ---

    def test_threshold_stats_returns_correct_type(
        self, exponential_series: pd.Series
    ) -> None:
        stats = compute_threshold_stats(exponential_series, threshold=3.0)
        assert isinstance(stats, ThresholdStats)

    def test_threshold_stats_counts_sum_to_total(
        self, exponential_series: pd.Series
    ) -> None:
        threshold = float(exponential_series.median())
        stats = compute_threshold_stats(exponential_series, threshold=threshold)
        assert stats.n_treated + stats.n_control == len(exponential_series.dropna())

    def test_threshold_stats_percentages_sum_to_100(
        self, exponential_series: pd.Series
    ) -> None:
        stats = compute_threshold_stats(exponential_series, threshold=1.0)
        assert stats.pct_treated + stats.pct_control == pytest.approx(100.0, abs=0.01)

    def test_threshold_stats_at_minimum_all_treated(
        self, exponential_series: pd.Series
    ) -> None:
        below_min = float(exponential_series.min()) - 1.0
        stats = compute_threshold_stats(exponential_series, threshold=below_min)
        assert stats.n_treated == len(exponential_series.dropna())
        assert stats.n_control == 0

    def test_threshold_stats_above_maximum_all_control(
        self, exponential_series: pd.Series
    ) -> None:
        above_max = float(exponential_series.max()) + 1.0
        stats = compute_threshold_stats(exponential_series, threshold=above_max)
        assert stats.n_treated == 0
        assert stats.n_control == len(exponential_series.dropna())

    def test_threshold_stats_ignores_nan(self) -> None:
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        stats = compute_threshold_stats(s, threshold=3.0)
        assert stats.n_treated + stats.n_control == 4  # 5 - 1 NaN

    def test_threshold_stats_ratio_format(self, exponential_series: pd.Series) -> None:
        stats = compute_threshold_stats(exponential_series, threshold=3.0)
        assert ":" in stats.ratio_str

    def test_threshold_stats_raises_on_empty_series(self) -> None:
        with pytest.raises(ValueError, match="no non-missing values"):
            compute_threshold_stats(pd.Series([np.nan, np.nan]), threshold=0.0)

    # --- suggest_thresholds ---

    def test_suggest_returns_three_suggestions(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        assert len(suggestions) == 3

    def test_suggest_all_are_threshold_suggestion(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        for s in suggestions:
            assert isinstance(s, ThresholdSuggestion)

    def test_suggest_methods_are_correct(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        methods = [s.method for s in suggestions]
        assert methods == ["median", "mean", "kde_trough"]

    def test_suggest_median_value_matches_pandas(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        median_sugg = next(s for s in suggestions if s.method == "median")
        expected = exponential_series.dropna().median()
        assert median_sugg.value == pytest.approx(expected, rel=1e-4)

    def test_suggest_mean_value_matches_pandas(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        mean_sugg = next(s for s in suggestions if s.method == "mean")
        expected = exponential_series.dropna().mean()
        assert mean_sugg.value == pytest.approx(expected, rel=1e-4)

    def test_suggest_kde_trough_in_data_range(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        kde_sugg = next(s for s in suggestions if s.method == "kde_trough")
        lo = float(exponential_series.min())
        hi = float(exponential_series.max())
        assert lo <= kde_sugg.value <= hi

    def test_suggest_kde_trough_finds_valley_in_bimodal(
        self, bimodal_series: pd.Series
    ) -> None:
        """The KDE trough should fall between the two modes (~0.5 and ~10)."""
        suggestions = suggest_thresholds(bimodal_series)
        kde_sugg = next(s for s in suggestions if s.method == "kde_trough")
        # The trough should be somewhere between the two modes
        assert 1.0 < kde_sugg.value < 8.0, (
            f"Expected trough between 1 and 8, got {kde_sugg.value:.3f}"
        )

    def test_suggest_labels_contain_values(
        self, exponential_series: pd.Series
    ) -> None:
        suggestions = suggest_thresholds(exponential_series)
        for s in suggestions:
            # Label should contain a number-like string
            assert any(ch.isdigit() for ch in s.label)

    def test_suggest_raises_on_constant_series(self) -> None:
        with pytest.raises(ValueError, match="at least 2 distinct"):
            suggest_thresholds(pd.Series([5.0] * 100))

    def test_suggest_falls_back_to_median_when_no_trough(self) -> None:
        """Unimodal, tight distribution may yield no local minimum."""
        # A perfectly normal distribution rarely has a KDE local min
        s = pd.Series(np.random.default_rng(0).normal(10, 0.1, 1000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            suggestions = suggest_thresholds(s)
        kde_sugg = next(x for x in suggestions if x.method == "kde_trough")
        # Should still return a numeric value (the fallback)
        assert isinstance(kde_sugg.value, float)
        assert not np.isnan(kde_sugg.value)

    # --- binarize_treatment ---

    def test_binarize_returns_series(self, exponential_series: pd.Series) -> None:
        result = binarize_treatment(exponential_series, threshold=3.0)
        assert isinstance(result, pd.Series)

    def test_binarize_values_are_zero_or_one(
        self, exponential_series: pd.Series
    ) -> None:
        result = binarize_treatment(exponential_series, threshold=3.0)
        non_null = result.dropna()
        assert set(non_null.unique()).issubset({0, 1})

    def test_binarize_preserves_index(self, exponential_series: pd.Series) -> None:
        result = binarize_treatment(exponential_series, threshold=3.0)
        pd.testing.assert_index_equal(result.index, exponential_series.index)

    def test_binarize_nan_preserved(self) -> None:
        s = pd.Series([1.0, np.nan, 5.0, np.nan])
        result = binarize_treatment(s, threshold=3.0)
        assert result.isna().sum() == 2  # two NaN inputs
        assert result.notna().sum() == 2

    def test_binarize_correct_split(self) -> None:
        s = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = binarize_treatment(s, threshold=3.0)
        # 0, 1, 2 -> 0 (control); 3, 4, 5 -> 1 (treated)
        expected = pd.array([0, 0, 0, 1, 1, 1], dtype="Int8")
        pd.testing.assert_extension_array_equal(result.array, expected)

    def test_binarize_threshold_is_inclusive(self) -> None:
        """Exactly at threshold should be coded as treated (>=)."""
        s = pd.Series([2.9, 3.0, 3.1])
        result = binarize_treatment(s, threshold=3.0)
        assert int(result.iloc[0]) == 0  # 2.9 < 3.0
        assert int(result.iloc[1]) == 1  # 3.0 == threshold -> treated
        assert int(result.iloc[2]) == 1  # 3.1 > 3.0

    def test_binarize_dtype_is_int8(self, exponential_series: pd.Series) -> None:
        result = binarize_treatment(exponential_series, threshold=1.0)
        assert result.dtype == pd.Int8Dtype()

    def test_binarize_raises_on_non_numeric(self) -> None:
        s = pd.Series(["a", "b", "c"])
        with pytest.raises(TypeError, match="numeric dtype"):
            binarize_treatment(s, threshold=0.0)

    def test_binarize_consistency_with_compute_stats(
        self, exponential_series: pd.Series
    ) -> None:
        """n_treated from binarize should match compute_threshold_stats."""
        threshold = float(exponential_series.median())
        binary = binarize_treatment(exponential_series, threshold=threshold)
        stats = compute_threshold_stats(exponential_series, threshold=threshold)

        n_treated_binary = int(binary.dropna().sum())
        assert n_treated_binary == stats.n_treated

    def test_suggest_and_binarize_pipeline(
        self, exponential_series: pd.Series
    ) -> None:
        """End-to-end: pick median suggestion, binarize, check balance."""
        suggestions = suggest_thresholds(exponential_series)
        median_sugg = next(s for s in suggestions if s.method == "median")
        binary = binarize_treatment(exponential_series, threshold=median_sugg.value)
        stats = compute_threshold_stats(exponential_series, threshold=median_sugg.value)

        # Median threshold: roughly 50% each side
        assert 30.0 <= stats.pct_treated <= 70.0, (
            f"Median split should be near 50/50, got {stats.pct_treated:.1f}% treated"
        )
        # Binary column should have at most 2 distinct non-null values
        assert binary.dropna().nunique() <= 2
