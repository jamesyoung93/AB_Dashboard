"""Tests for modules/did.py — Difference-in-Differences estimation.

Each test targets one function or one behavioural contract. The synthetic
data generating process has a known true ATT (5.0) so correctness tests
check that the true value falls inside the 95% CI, as recommended in the
project success metrics (CI containment, not p-value).

Seed: 42 (reproducible). All tests should pass in < 10 s on a laptop.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from modules.did import (
    DidResult,
    EventStudyResult,
    create_post_indicator,
    parallel_trends_data,
    run_ancova,
    run_did,
    run_event_study,
    run_staggered_did,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

TRUE_ATT = 5.0
SEED = 42


@pytest.fixture(scope="module")
def panel_df() -> pd.DataFrame:
    """Synthetic 2x2 DiD panel with true ATT = 5.0.

    Periods 1-2 are pre-treatment; periods 3-4 are post-treatment.
    event_time = period - 3 (so period 3 → event_time=0, period 2 → -1).
    100 units: 50 treated, 50 control.
    """
    rng = np.random.default_rng(SEED)
    n_units = 100
    rows = []
    for i in range(n_units):
        treated = int(i < 50)
        for t in range(1, 5):
            post = int(t >= 3)
            y = (
                10
                + 3 * treated
                + 2 * post
                + TRUE_ATT * treated * post
                + rng.normal(0, 2)
            )
            rows.append(
                {
                    "unit": i,
                    "period": t,
                    "treated": treated,
                    "outcome": y,
                    "cov1": rng.normal(0, 1),
                    "event_time": t - 3,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# create_post_indicator
# ---------------------------------------------------------------------------


class TestCreatePostIndicator:
    def test_integer_threshold(self, panel_df: pd.DataFrame) -> None:
        post = create_post_indicator(panel_df, "period", 3)
        assert post.dtype == int or np.issubdtype(post.dtype, np.integer)
        assert set(post.unique()) == {0, 1}
        assert post.sum() == 200  # 100 units x 2 post periods

    def test_pre_periods_are_zero(self, panel_df: pd.DataFrame) -> None:
        post = create_post_indicator(panel_df, "period", 3)
        pre_mask = panel_df["period"] < 3
        assert (post[pre_mask] == 0).all()

    def test_post_periods_are_one(self, panel_df: pd.DataFrame) -> None:
        post = create_post_indicator(panel_df, "period", 3)
        post_mask = panel_df["period"] >= 3
        assert (post[post_mask] == 1).all()

    def test_missing_column_raises(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="MISSING"):
            create_post_indicator(panel_df, "MISSING", 3)

    def test_boundary_inclusive(self, panel_df: pd.DataFrame) -> None:
        """Period == post_period_start should be post=1."""
        post = create_post_indicator(panel_df, "period", 3)
        at_boundary = panel_df[panel_df["period"] == 3].index
        assert (post.loc[at_boundary] == 1).all()


# ---------------------------------------------------------------------------
# parallel_trends_data
# ---------------------------------------------------------------------------


class TestParallelTrendsData:
    def test_output_schema(self, panel_df: pd.DataFrame) -> None:
        result = parallel_trends_data(panel_df, "outcome", "treated", "period")
        assert set(result.columns) == {
            "time",
            "group",
            "mean",
            "ci_lower",
            "ci_upper",
            "n",
        }

    def test_groups_present(self, panel_df: pd.DataFrame) -> None:
        result = parallel_trends_data(panel_df, "outcome", "treated", "period")
        assert set(result["group"].unique()) == {"treated", "control"}

    def test_row_count(self, panel_df: pd.DataFrame) -> None:
        """4 periods x 2 groups = 8 rows."""
        result = parallel_trends_data(panel_df, "outcome", "treated", "period")
        assert len(result) == 8

    def test_ci_ordering(self, panel_df: pd.DataFrame) -> None:
        result = parallel_trends_data(panel_df, "outcome", "treated", "period")
        assert (result["ci_lower"] <= result["mean"]).all()
        assert (result["mean"] <= result["ci_upper"]).all()

    def test_missing_column_raises(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="BAD"):
            parallel_trends_data(panel_df, "BAD", "treated", "period")

    def test_n_column(self, panel_df: pd.DataFrame) -> None:
        result = parallel_trends_data(panel_df, "outcome", "treated", "period")
        # Each cell should have 50 observations (50 treated / 50 control)
        assert (result["n"] == 50).all()


# ---------------------------------------------------------------------------
# run_did
# ---------------------------------------------------------------------------


class TestRunDid:
    def test_returns_dataclass(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert isinstance(result, DidResult)

    def test_method_label(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert result.method == "standard_did"

    def test_att_close_to_truth(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert abs(result.att - TRUE_ATT) < 1.5

    def test_true_att_in_ci(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(
                panel_df,
                "outcome",
                "treated",
                "period",
                3,
                entity_col="unit",
            )
        assert result.ci_lower <= TRUE_ATT <= result.ci_upper, (
            f"True ATT={TRUE_ATT} not in [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_ci_ordering(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert result.ci_lower < result.att < result.ci_upper

    def test_p_value_significant(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert result.p_value < 0.01

    def test_unit_counts(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(
                panel_df,
                "outcome",
                "treated",
                "period",
                3,
                entity_col="unit",
            )
        assert result.n_treated == 50
        assert result.n_control == 50

    def test_with_covariates(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(
                panel_df,
                "outcome",
                "treated",
                "period",
                3,
                covariate_cols=["cov1"],
                entity_col="unit",
            )
        assert result.ci_lower <= TRUE_ATT <= result.ci_upper

    def test_model_summary_string(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert isinstance(result.model_summary, str)
        assert len(result.model_summary) > 100

    def test_missing_column_raises(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="MISSING"):
            run_did(panel_df, "outcome", "MISSING", "period", 3)

    def test_missing_outcome_rows_dropped(self, panel_df: pd.DataFrame) -> None:
        df_miss = panel_df.copy()
        df_miss.loc[df_miss["unit"] == 0, "outcome"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(df_miss, "outcome", "treated", "period", 3)
        # Should still work after dropping NaN rows
        assert abs(result.att - TRUE_ATT) < 1.5

    def test_covariate_with_space_in_name(self, panel_df: pd.DataFrame) -> None:
        """Column names containing spaces must be handled via Q() quoting."""
        df2 = panel_df.copy()
        df2["zip code"] = "10001"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(
                df2,
                "outcome",
                "treated",
                "period",
                3,
                covariate_cols=["zip code"],
            )
        assert abs(result.att - TRUE_ATT) < 1.5

    def test_se_positive(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(panel_df, "outcome", "treated", "period", 3)
        assert result.se > 0

    def test_clustered_se_with_entity_col(self, panel_df: pd.DataFrame) -> None:
        """Clustering on entity_col should produce valid (not NaN) SEs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_did(
                panel_df,
                "outcome",
                "treated",
                "period",
                3,
                entity_col="unit",
            )
        assert np.isfinite(result.se)
        assert np.isfinite(result.att)


# ---------------------------------------------------------------------------
# run_ancova
# ---------------------------------------------------------------------------


class TestRunAncova:
    def test_returns_dataclass(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ancova(
                panel_df, "outcome", "treated", "period", 3, entity_col="unit"
            )
        assert isinstance(result, DidResult)

    def test_method_label(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ancova(
                panel_df, "outcome", "treated", "period", 3, entity_col="unit"
            )
        assert result.method == "ancova"

    def test_att_close_to_truth(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ancova(
                panel_df, "outcome", "treated", "period", 3, entity_col="unit"
            )
        assert abs(result.att - TRUE_ATT) < 1.5

    def test_true_att_in_ci(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ancova(
                panel_df,
                "outcome",
                "treated",
                "period",
                3,
                entity_col="unit",
                covariate_cols=["cov1"],
            )
        assert result.ci_lower <= TRUE_ATT <= result.ci_upper

    def test_unit_counts(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ancova(
                panel_df, "outcome", "treated", "period", 3, entity_col="unit"
            )
        assert result.n_treated == 50
        assert result.n_control == 50

    def test_no_post_raises(self, panel_df: pd.DataFrame) -> None:
        """If post_period_start is beyond all data, no post rows → ValueError."""
        with pytest.raises(ValueError, match="No post-period"):
            run_ancova(
                panel_df,
                "outcome",
                "treated",
                "period",
                999,
                entity_col="unit",
            )

    def test_no_pre_raises(self, panel_df: pd.DataFrame) -> None:
        """If post_period_start is at or below min period, no pre rows → ValueError."""
        with pytest.raises(ValueError, match="No pre-period"):
            run_ancova(
                panel_df,
                "outcome",
                "treated",
                "period",
                0,
                entity_col="unit",
            )

    def test_missing_entity_col_raises(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="MISSING"):
            run_ancova(
                panel_df, "outcome", "treated", "period", 3, entity_col="MISSING"
            )


# ---------------------------------------------------------------------------
# run_event_study
# ---------------------------------------------------------------------------


class TestRunEventStudy:
    def test_returns_dataclass(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        assert isinstance(result, EventStudyResult)

    def test_periods_sorted(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        assert result.periods == sorted(result.periods)

    def test_reference_period_coef_is_zero(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time",
                reference_period=-1
            )
        ref_idx = result.periods.index(-1)
        assert result.coefficients[ref_idx] == 0.0
        assert result.standard_errors[ref_idx] == 0.0

    def test_pre_period_near_zero(self, panel_df: pd.DataFrame) -> None:
        """Pre-treatment coefficient (tau=-2) should be near zero (parallel trends)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        pre_idx = result.periods.index(-2)
        assert abs(result.coefficients[pre_idx]) < 3.0, (
            f"Pre-treatment coef={result.coefficients[pre_idx]:.3f} too far from zero"
        )

    def test_post_period_near_att(self, panel_df: pd.DataFrame) -> None:
        """First post-treatment coefficient (tau=0) should be near TRUE_ATT."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        post_idx = result.periods.index(0)
        assert abs(result.coefficients[post_idx] - TRUE_ATT) < 2.5

    def test_parallel_lists_same_length(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        n = len(result.periods)
        assert len(result.coefficients) == n
        assert len(result.standard_errors) == n
        assert len(result.ci_lower) == n
        assert len(result.ci_upper) == n

    def test_ci_ordering(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        for i, tau in enumerate(result.periods):
            if tau == -1:  # reference period: all zeros
                continue
            assert result.ci_lower[i] <= result.coefficients[i] <= result.ci_upper[i]

    def test_unit_counts(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df,
                "outcome",
                "treated",
                "period",
                "event_time",
                entity_col="unit",
            )
        assert result.n_treated == 50
        assert result.n_control == 50

    def test_bad_reference_period_raises(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Reference period"):
            run_event_study(
                panel_df,
                "outcome",
                "treated",
                "period",
                "event_time",
                reference_period=-99,
            )

    def test_missing_event_time_col_raises(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="MISSING"):
            run_event_study(
                panel_df, "outcome", "treated", "period", "MISSING"
            )

    def test_model_summary_string(self, panel_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_event_study(
                panel_df, "outcome", "treated", "period", "event_time"
            )
        assert isinstance(result.model_summary, str)
        assert len(result.model_summary) > 100


# ---------------------------------------------------------------------------
# run_staggered_did
# ---------------------------------------------------------------------------


class TestRunStaggeredDid:
    def test_raises_not_implemented(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(NotImplementedError):
            run_staggered_did(
                panel_df, "outcome", "treated", "period", "unit"
            )

    def test_error_mentions_csdid(self, panel_df: pd.DataFrame) -> None:
        with pytest.raises(NotImplementedError, match="csdid"):
            run_staggered_did(
                panel_df, "outcome", "treated", "period", "unit"
            )
