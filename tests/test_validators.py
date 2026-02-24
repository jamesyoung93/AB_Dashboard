"""Unit tests for utils/validators.py.

Tests validate the advisory-warning system. Each check function is tested
independently with targeted DataFrames, then validate_csv is tested end-to-end
with the real synthetic datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.validators import (
    ValidationWarning,
    _balance_check,
    _cardinality_check,
    _date_parsing_check,
    _duplicate_check,
    _null_check,
    _type_inference_check,
    _zip_format_check,
    validate_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Minimal clean panel DataFrame matching the proposal schema."""
    rng = np.random.default_rng(42)
    n_customers = 50
    n_weeks = 10
    records = []
    for cid in range(1, n_customers + 1):
        for week in range(1, n_weeks + 1):
            records.append(
                {
                    "customer_id": cid,
                    "week": week,
                    "channel_email": rng.integers(0, 2),
                    "revenue": rng.normal(50, 10),
                    "prior_spend": rng.normal(500, 100),
                    "company_size": rng.choice(["Small", "Medium", "Large"]),
                    "zip_code": rng.integers(10000, 99950),
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def standard_roles() -> dict:
    return {
        "customer_id": "customer_id",
        "time_period": "week",
        "treatment_binary": "channel_email",
        "outcome": "revenue",
        "covariates": ["prior_spend", "company_size"],
        "geographic_id": "zip_code",
    }


# ---------------------------------------------------------------------------
# _null_check
# ---------------------------------------------------------------------------


class TestNullCheck:
    def test_clean_column_produces_no_warnings(self, clean_df: pd.DataFrame) -> None:
        warnings = _null_check(clean_df, "revenue")
        assert warnings == []

    def test_above_threshold_produces_warning(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.loc[:25, "revenue"] = np.nan  # ~5% of 500 rows
        warnings = _null_check(df, "revenue", threshold=0.04)
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert "null" in warnings[0].message.lower()
        assert warnings[0].column == "revenue"

    def test_majority_null_is_error(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df.loc[:300, "revenue"] = np.nan  # >50%
        warnings = _null_check(df, "revenue")
        assert len(warnings) == 1
        assert warnings[0].severity == "error"

    def test_exactly_at_threshold_produces_no_warning(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        # Set exactly 5% (25 of 500) to null -- should NOT trigger (>5% not >=5%)
        df.loc[:24, "revenue"] = np.nan
        warnings = _null_check(df, "revenue", threshold=0.05)
        assert warnings == []


# ---------------------------------------------------------------------------
# _type_inference_check
# ---------------------------------------------------------------------------


class TestTypeInferenceCheck:
    def test_numeric_column_no_warning(self, clean_df: pd.DataFrame) -> None:
        warnings = _type_inference_check(clean_df, "revenue", "outcome")
        assert warnings == []

    def test_currency_string_produces_error(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df["revenue_str"] = "$" + df["revenue"].astype(str)
        warnings = _type_inference_check(df, "revenue_str", "outcome")
        assert len(warnings) == 1
        assert warnings[0].severity == "error"
        assert "numeric" in warnings[0].message.lower()

    def test_non_outcome_role_not_checked(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df["company_str"] = df["company_size"]
        # covariates role does NOT trigger type check in _type_inference_check
        warnings = _type_inference_check(df, "company_str", "covariates")
        assert warnings == []

    def test_numeric_string_parses_fine(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df["revenue_str"] = df["revenue"].astype(str)
        warnings = _type_inference_check(df, "revenue_str", "outcome")
        assert warnings == []


# ---------------------------------------------------------------------------
# _date_parsing_check
# ---------------------------------------------------------------------------


class TestDateParsingCheck:
    def test_integer_period_no_warning(self, clean_df: pd.DataFrame) -> None:
        warnings = _date_parsing_check(clean_df, "week")
        assert warnings == []

    def test_date_string_no_warning(self) -> None:
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-08", "2024-01-15"]})
        warnings = _date_parsing_check(df, "date")
        assert warnings == []

    def test_unparseable_produces_warning(self) -> None:
        df = pd.DataFrame({"period": ["Q1-24", "Q2-24", "garbage", "more-garbage"]})
        warnings = _date_parsing_check(df, "period")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_numeric_string_period_no_warning(self) -> None:
        df = pd.DataFrame({"period": ["1", "2", "3", "4", "5"]})
        warnings = _date_parsing_check(df, "period")
        assert warnings == []


# ---------------------------------------------------------------------------
# _cardinality_check
# ---------------------------------------------------------------------------


class TestCardinalityCheck:
    def test_low_cardinality_no_warning(self, clean_df: pd.DataFrame) -> None:
        # company_size has 3 unique values
        warnings = _cardinality_check(clean_df, "company_size", "covariates")
        assert warnings == []

    def test_high_cardinality_produces_warning(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df["free_text"] = [f"note_{i}" for i in range(len(df))]
        warnings = _cardinality_check(df, "free_text", "covariates")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert "categories" in warnings[0].message.lower()

    def test_numeric_column_not_flagged(self, clean_df: pd.DataFrame) -> None:
        # Numeric columns with many unique values should not trigger cardinality check
        warnings = _cardinality_check(clean_df, "revenue", "covariates")
        assert warnings == []

    def test_non_covariate_role_not_checked(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df["free_text"] = [f"note_{i}" for i in range(len(df))]
        warnings = _cardinality_check(df, "free_text", "outcome")
        assert warnings == []


# ---------------------------------------------------------------------------
# _balance_check
# ---------------------------------------------------------------------------


class TestBalanceCheck:
    def test_balanced_treatment_no_warning(self, clean_df: pd.DataFrame) -> None:
        # channel_email is ~50/50 by construction
        warnings = _balance_check(clean_df, "channel_email")
        assert warnings == []

    def test_very_rare_treated_produces_warning(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        # 3% treated
        df["rare"] = 0
        df.loc[:14, "rare"] = 1
        warnings = _balance_check(df, "rare")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert "treated" in warnings[0].message.lower()

    def test_near_universal_treated_produces_warning(self, clean_df: pd.DataFrame) -> None:
        df = clean_df.copy()
        df["common"] = 1
        df.loc[:10, "common"] = 0  # 97% treated
        warnings = _balance_check(df, "common")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"

    def test_exactly_10pct_boundary_no_warning(self) -> None:
        df = pd.DataFrame({"t": [1] * 10 + [0] * 90})  # exactly 10%
        warnings = _balance_check(df, "t")
        assert warnings == []

    def test_just_below_10pct_produces_warning(self) -> None:
        df = pd.DataFrame({"t": [1] * 9 + [0] * 91})  # 9%
        warnings = _balance_check(df, "t")
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# _duplicate_check
# ---------------------------------------------------------------------------


class TestDuplicateCheck:
    def test_no_duplicates_no_warning(self, clean_df: pd.DataFrame) -> None:
        warnings = _duplicate_check(clean_df, "customer_id", "week")
        assert warnings == []

    def test_duplicates_produce_warning(self, clean_df: pd.DataFrame) -> None:
        df = pd.concat([clean_df, clean_df.head(5)], ignore_index=True)
        warnings = _duplicate_check(df, "customer_id", "week")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.column == "__dataset__"
        assert w.severity == "warning"
        assert "5" in w.message
        assert "duplicate" in w.message.lower()


# ---------------------------------------------------------------------------
# _zip_format_check
# ---------------------------------------------------------------------------


class TestZipFormatCheck:
    def test_valid_integer_zips_no_warning(self) -> None:
        df = pd.DataFrame({"zip_code": [10001, 90210, 60614, 33101, 77002]})
        warnings = _zip_format_check(df, "zip_code")
        assert warnings == []

    def test_valid_string_zips_no_warning(self) -> None:
        df = pd.DataFrame({"zip_code": ["10001", "02134", "60614", "90210"]})
        warnings = _zip_format_check(df, "zip_code")
        assert warnings == []

    def test_non_numeric_string_produces_warning(self) -> None:
        df = pd.DataFrame({"zip_code": ["ABC12", "XYZ99", "10001"]})
        warnings = _zip_format_check(df, "zip_code")
        assert len(warnings) >= 1
        assert any("not 5-digit" in w.message for w in warnings)

    def test_6_digit_zip_produces_warning(self) -> None:
        df = pd.DataFrame({"zip_code": [100010, 900210]})
        warnings = _zip_format_check(df, "zip_code")
        assert len(warnings) >= 1

    def test_out_of_range_zip_produces_info(self) -> None:
        df = pd.DataFrame({"zip_code": [99999]})  # above 99950
        warnings = _zip_format_check(df, "zip_code")
        assert len(warnings) == 1
        assert warnings[0].severity == "info"
        assert "valid US range" in warnings[0].message

    def test_zero_padded_zip_valid(self) -> None:
        # Leading-zero ZIP (New England)
        df = pd.DataFrame({"zip_code": ["02134", "02139", "01001"]})
        warnings = _zip_format_check(df, "zip_code")
        assert warnings == []


# ---------------------------------------------------------------------------
# validate_csv (end-to-end)
# ---------------------------------------------------------------------------


class TestValidateCsv:
    def test_clean_data_no_warnings(
        self, clean_df: pd.DataFrame, standard_roles: dict
    ) -> None:
        warnings = validate_csv(clean_df, standard_roles)
        assert warnings == []

    def test_empty_df_raises(self, standard_roles: dict) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_csv(pd.DataFrame(), standard_roles)

    def test_missing_roles_skipped_gracefully(self, clean_df: pd.DataFrame) -> None:
        # Only specify a subset of roles -- no KeyError
        warnings = validate_csv(clean_df, {"customer_id": "customer_id"})
        assert isinstance(warnings, list)

    def test_null_in_outcome_caught(
        self, clean_df: pd.DataFrame, standard_roles: dict
    ) -> None:
        df = clean_df.copy()
        df.loc[:50, "revenue"] = np.nan
        warnings = validate_csv(df, standard_roles)
        columns_warned = {w.column for w in warnings}
        assert "revenue" in columns_warned

    def test_duplicate_rows_caught(
        self, clean_df: pd.DataFrame, standard_roles: dict
    ) -> None:
        df = pd.concat([clean_df, clean_df.head(3)], ignore_index=True)
        warnings = validate_csv(df, standard_roles)
        dataset_warnings = [w for w in warnings if w.column == "__dataset__"]
        assert len(dataset_warnings) == 1
        assert "duplicate" in dataset_warnings[0].message.lower()

    def test_imbalanced_treatment_caught(
        self, clean_df: pd.DataFrame, standard_roles: dict
    ) -> None:
        df = clean_df.copy()
        df["channel_email"] = 0
        df.loc[:20, "channel_email"] = 1  # ~4% treated
        warnings = validate_csv(df, standard_roles)
        trt_warnings = [w for w in warnings if w.column == "channel_email"]
        assert len(trt_warnings) == 1

    def test_high_cardinality_covariate_caught(
        self, clean_df: pd.DataFrame, standard_roles: dict
    ) -> None:
        df = clean_df.copy()
        df["free_text"] = [f"note_{i}" for i in range(len(df))]
        roles = dict(standard_roles)
        roles["covariates"] = ["prior_spend", "company_size", "free_text"]
        warnings = validate_csv(df, roles)
        card_warnings = [w for w in warnings if "categories" in w.message.lower()]
        assert len(card_warnings) == 1

    def test_none_role_skipped(self, clean_df: pd.DataFrame) -> None:
        roles = {
            "customer_id": "customer_id",
            "time_period": "week",
            "treatment_binary": None,
            "outcome": "revenue",
        }
        warnings = validate_csv(clean_df, roles)
        # No warnings expected; None roles should be silently skipped
        assert isinstance(warnings, list)

    def test_warning_namedtuple_fields(
        self, clean_df: pd.DataFrame, standard_roles: dict
    ) -> None:
        df = clean_df.copy()
        df.loc[:30, "revenue"] = np.nan
        warnings = validate_csv(df, standard_roles)
        assert len(warnings) > 0
        w = warnings[0]
        assert hasattr(w, "column")
        assert hasattr(w, "severity")
        assert hasattr(w, "message")
        assert w.severity in ("info", "warning", "error")

    def test_synthetic_omnichannel_clean(self) -> None:
        """Integration test: real synthetic data should have no warnings."""
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_omnichannel.csv"
        )
        roles = {
            "customer_id": "customer_id",
            "time_period": "week",
            "treatment_binary": "channel_email",
            "outcome": "revenue",
            "covariates": ["prior_spend", "tenure_years", "industry", "company_size"],
            "geographic_id": "zip_code",
        }
        warnings = validate_csv(df, roles)
        assert warnings == [], f"Unexpected warnings on clean data: {warnings}"
