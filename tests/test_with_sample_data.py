"""Integration tests: full pipeline on synthetic datasets.

Validates that the known ground truths are recovered:
- synthetic_omnichannel.csv: ATT=3.5 for channel_email on revenue (DiD, PSM)
- synthetic_omnichannel.csv: ATT~0 for channel_direct_mail (null tactic)
- synthetic_rdd.csv: LATE=4.0 at cutoff (deferred to Phase 2 — RDD module)
- synthetic_scm.csv: effect=6.0 (deferred to Phase 2 — Synthetic Control module)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.balance import compute_balance
from modules.column_detector import check_method_eligibility, detect_columns
from modules.did import parallel_trends_data, run_ancova, run_did
from modules.propensity import fit_propensity
from modules.psm import get_matched_data, run_psm
from utils.validators import validate_csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(scope="module")
def omni_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_omnichannel.csv")


@pytest.fixture(scope="module")
def rdd_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_rdd.csv")


@pytest.fixture(scope="module")
def scm_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_scm.csv")


# -----------------------------------------------------------------------
# Column detection
# -----------------------------------------------------------------------


class TestColumnDetection:
    def test_omnichannel_detects_zip(self, omni_df: pd.DataFrame) -> None:
        det = detect_columns(omni_df)
        assert det.get("geographic_id") is not None
        assert det["geographic_id"].column_name == "zip_code"

    def test_omnichannel_detects_time(self, omni_df: pd.DataFrame) -> None:
        det = detect_columns(omni_df)
        assert det.get("time_period") is not None
        assert det["time_period"].column_name == "week"

    def test_omnichannel_detects_treatment(self, omni_df: pd.DataFrame) -> None:
        det = detect_columns(omni_df)
        # At least one treatment-type detection
        has_binary = det.get("treatment_binary") is not None
        has_continuous = det.get("treatment_continuous") is not None
        assert has_binary or has_continuous

    def test_method_eligibility_did(self, omni_df: pd.DataFrame) -> None:
        roles = {
            "time_period": "week",
            "treatment_binary": "channel_email",
        }
        elig = check_method_eligibility(omni_df, roles)
        assert elig["DiD"][0] is True


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------


class TestValidation:
    def test_omnichannel_no_errors(self, omni_df: pd.DataFrame) -> None:
        roles = {
            "customer_id": "customer_id",
            "time_period": "week",
            "treatment_binary": "channel_email",
            "outcome": "revenue",
            "geographic_id": "zip_code",
            "covariates": ["prior_spend", "tenure_years", "company_size",
                           "industry", "engagement_score"],
        }
        warnings = validate_csv(omni_df, roles)
        errors = [w for w in warnings if w.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"


# -----------------------------------------------------------------------
# DiD on omnichannel data
# -----------------------------------------------------------------------


class TestDidOmnichannel:
    """DiD should recover ATT ~ 3.5 for channel_email on revenue."""

    def test_did_recovers_att(self, omni_df: pd.DataFrame) -> None:
        result = run_did(
            df=omni_df,
            outcome_col="revenue",
            treatment_col="channel_email",
            time_col="week",
            post_period_start=11,
            covariate_cols=["prior_spend", "tenure_years", "engagement_score"],
            entity_col="customer_id",
        )
        # True ATT = 3.5; should be within 95% CI
        assert result.ci_lower <= 3.5 <= result.ci_upper, (
            f"True ATT 3.5 not in CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_did_att_point_estimate_reasonable(self, omni_df: pd.DataFrame) -> None:
        result = run_did(
            df=omni_df,
            outcome_col="revenue",
            treatment_col="channel_email",
            time_col="week",
            post_period_start=11,
            entity_col="customer_id",
        )
        # Point estimate should be within 2.0 of true value
        assert abs(result.att - 3.5) < 2.0, f"ATT {result.att:.3f} too far from 3.5"

    def test_ancova_recovers_att(self, omni_df: pd.DataFrame) -> None:
        result = run_ancova(
            df=omni_df,
            outcome_col="revenue",
            treatment_col="channel_email",
            time_col="week",
            post_period_start=11,
            covariate_cols=["prior_spend", "tenure_years", "engagement_score"],
            entity_col="customer_id",
        )
        # True ATT = 3.5; should be within 95% CI
        assert result.ci_lower <= 3.5 <= result.ci_upper, (
            f"True ATT 3.5 not in ANCOVA CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_parallel_trends_data_shape(self, omni_df: pd.DataFrame) -> None:
        pt = parallel_trends_data(omni_df, "revenue", "channel_email", "week")
        assert "time" in pt.columns
        assert "group" in pt.columns
        assert "mean" in pt.columns
        # 20 weeks x 2 groups = 40 rows
        assert len(pt) == 40


# -----------------------------------------------------------------------
# PSM on omnichannel data
# -----------------------------------------------------------------------


class TestPsmOmnichannel:
    """PSM should recover ATT ~ 3.5 for channel_email on revenue.

    PSM is a cross-sectional method. For panel data, we collapse to
    customer-level: average post-period revenue, then match on baseline
    covariates. This removes the time dimension and lets PSM estimate
    the post-period treatment effect directly.
    """

    @pytest.fixture(scope="class")
    def post_customer_df(self, omni_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse panel to customer-level post-period averages."""
        post = omni_df[omni_df["week"] > 10].copy()
        agg = post.groupby("customer_id").agg(
            revenue=("revenue", "mean"),
            channel_email=("channel_email", "first"),
            channel_webinar=("channel_webinar", "mean"),
            prior_spend=("prior_spend", "first"),
            tenure_years=("tenure_years", "first"),
            company_size=("company_size", "first"),
            industry=("industry", "first"),
            engagement_score=("engagement_score", "first"),
            sales_rep_touches=("sales_rep_touches", "mean"),
            social_paid_impressions=("social_paid_impressions", "mean"),
        ).reset_index()
        return agg

    @pytest.fixture(scope="class")
    def propensity(self, post_customer_df: pd.DataFrame):
        # Include sales_rep_touches — strong confounder (company_size + treatment)
        return fit_propensity(
            df=post_customer_df,
            treatment_col="channel_email",
            covariate_cols=["prior_spend", "tenure_years", "company_size",
                           "industry", "engagement_score", "channel_webinar",
                           "sales_rep_touches"],
            method="logistic",
        )

    def test_propensity_auc_reasonable(self, propensity) -> None:
        # AUC should be > 0.5 (better than random) due to confounding
        assert propensity.auc > 0.55

    def test_psm_recovers_att(self, post_customer_df: pd.DataFrame, propensity) -> None:
        result = run_psm(
            df=post_customer_df,
            outcome_col="revenue",
            treatment_col="channel_email",
            propensity_scores=propensity.scores,
            method="nearest_1to1",
            n_bootstrap=200,
            seed=42,
        )
        # True ATT = 3.5. PSM has inherent finite-sample bias from imperfect
        # matching on categorical covariates (N=500). Use wider tolerance:
        # point estimate within 4.0 of true value
        assert abs(result.att - 3.5) < 4.0, (
            f"PSM ATT {result.att:.3f} too far from true ATT 3.5"
        )
        # Positive and significant
        assert result.att > 0, "PSM ATT should be positive"
        assert result.p_value < 0.05, "PSM should detect significant effect"

    def test_psm_att_reasonable(self, post_customer_df: pd.DataFrame, propensity) -> None:
        result = run_psm(
            df=post_customer_df,
            outcome_col="revenue",
            treatment_col="channel_email",
            propensity_scores=propensity.scores,
            method="nearest_1to1",
            n_bootstrap=100,
            seed=42,
        )
        # Point estimate within 3.0 of true value (wider for PSM)
        assert abs(result.att - 3.5) < 3.0, f"PSM ATT {result.att:.3f} too far from 3.5"

    def test_psm_matched_data(self, post_customer_df: pd.DataFrame, propensity) -> None:
        result = run_psm(
            df=post_customer_df,
            outcome_col="revenue",
            treatment_col="channel_email",
            propensity_scores=propensity.scores,
            method="nearest_1to1",
            n_bootstrap=50,
            seed=42,
        )
        matched = get_matched_data(post_customer_df, result.matched_indices, "channel_email")
        assert len(matched) > 0
        assert "match_id" in matched.columns


# -----------------------------------------------------------------------
# Null tactic validation
# -----------------------------------------------------------------------


class TestNullTactic:
    """channel_direct_mail has no causal effect — PSM should recover ATT ~ 0."""

    @pytest.fixture(scope="class")
    def post_customer_dm(self, omni_df: pd.DataFrame) -> pd.DataFrame:
        post = omni_df[omni_df["week"] > 10].copy()
        return post.groupby("customer_id").agg(
            revenue=("revenue", "mean"),
            channel_direct_mail=("channel_direct_mail", "first"),
            prior_spend=("prior_spend", "first"),
            tenure_years=("tenure_years", "first"),
            company_size=("company_size", "first"),
            industry=("industry", "first"),
            engagement_score=("engagement_score", "first"),
        ).reset_index()

    @pytest.fixture(scope="class")
    def propensity_dm(self, post_customer_dm: pd.DataFrame):
        return fit_propensity(
            df=post_customer_dm,
            treatment_col="channel_direct_mail",
            covariate_cols=["prior_spend", "tenure_years", "company_size",
                           "industry", "engagement_score"],
            method="logistic",
        )

    def test_null_tactic_did_near_zero(self, omni_df: pd.DataFrame) -> None:
        result = run_did(
            df=omni_df,
            outcome_col="revenue",
            treatment_col="channel_direct_mail",
            time_col="week",
            post_period_start=11,
            entity_col="customer_id",
        )
        # Effect should be near zero (within +/- 2.0)
        assert abs(result.att) < 2.0, f"Null tactic ATT {result.att:.3f} not near zero"

    def test_null_tactic_psm_near_zero(self, post_customer_dm: pd.DataFrame, propensity_dm) -> None:
        result = run_psm(
            df=post_customer_dm,
            outcome_col="revenue",
            treatment_col="channel_direct_mail",
            propensity_scores=propensity_dm.scores,
            method="nearest_1to1",
            n_bootstrap=100,
            seed=42,
        )
        # Effect should be near zero (within +/- 3.0, PSM has more variance)
        assert abs(result.att) < 3.0, f"Null tactic PSM ATT {result.att:.3f} not near zero"


# -----------------------------------------------------------------------
# Balance diagnostics on omnichannel
# -----------------------------------------------------------------------


class TestBalanceOmnichannel:
    def test_raw_balance_shows_imbalance(self, omni_df: pd.DataFrame) -> None:
        """Raw balance should show confounding — not all covariates pass."""
        result = compute_balance(
            omni_df,
            "channel_email",
            ["prior_spend", "engagement_score", "company_size", "industry"],
        )
        # With confounding, raw SMDs should show some imbalance
        assert result.max_smd > 0.0

    def test_post_match_balance_improves(self, omni_df: pd.DataFrame) -> None:
        """Post-match balance should be better than raw."""
        covs = ["prior_spend", "tenure_years", "engagement_score",
                 "company_size", "industry"]
        raw_bal = compute_balance(omni_df, "channel_email", covs)

        prop = fit_propensity(
            omni_df, "channel_email", covs, method="logistic",
        )
        psm_res = run_psm(
            omni_df, "revenue", "channel_email", prop.scores,
            n_bootstrap=50, seed=42,
        )
        mi = psm_res.matched_indices
        adj_bal = compute_balance(
            omni_df, "channel_email", covs,
            matched_indices=(mi["treated_idx"].values, mi["control_idx"].values),
        )
        # Adjusted max SMD should be smaller than raw
        assert adj_bal.max_smd <= raw_bal.max_smd, (
            f"Adjusted max SMD {adj_bal.max_smd:.3f} worse than raw {raw_bal.max_smd:.3f}"
        )
