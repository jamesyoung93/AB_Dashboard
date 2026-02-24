"""Unit tests for modules/column_detector.py.

Tests cover each heuristic helper, each role detector, the top-level
detect_columns function, and check_method_eligibility, using targeted
DataFrames and the real synthetic datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.column_detector import (
    ColumnSuggestion,
    _cardinality_ratio,
    _detect_customer_id,
    _detect_geographic_id,
    _detect_outcome,
    _detect_time_period,
    _detect_treatment_binary,
    _detect_treatment_continuous,
    _has_moderate_variance,
    _is_binary_01,
    _is_high_cardinality,
    _is_non_negative_numeric,
    _is_sequential_or_parseable_date,
    _looks_like_zip,
    _name_matches,
    check_method_eligibility,
    detect_columns,
)


# ---------------------------------------------------------------------------
# Helper: small DataFrame builders
# ---------------------------------------------------------------------------


def _make_df(**kwargs: list) -> pd.DataFrame:
    """Build a small DataFrame from keyword-to-list mappings."""
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# _name_matches
# ---------------------------------------------------------------------------


class TestNameMatches:
    def test_exact_match(self) -> None:
        assert _name_matches("zip", ["zip", "postal"]) is True

    def test_substring_match(self) -> None:
        assert _name_matches("zip_code", ["zip"]) is True

    def test_case_insensitive(self) -> None:
        assert _name_matches("ZIP_CODE", ["zip"]) is True

    def test_no_match(self) -> None:
        assert _name_matches("revenue", ["zip", "postal"]) is False


# ---------------------------------------------------------------------------
# _looks_like_zip
# ---------------------------------------------------------------------------


class TestLooksLikeZip:
    def test_valid_integer_zips(self) -> None:
        s = pd.Series([10001, 90210, 60614, 33101, 77002, 2134])
        assert _looks_like_zip(s) is True

    def test_valid_string_zips(self) -> None:
        s = pd.Series(["10001", "02134", "60614", "90210"])
        assert _looks_like_zip(s) is True

    def test_non_zip_integers(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])  # Not 5-digit
        assert _looks_like_zip(s) is False

    def test_mixed_valid_invalid_below_threshold(self) -> None:
        # <90% valid -- should return False
        valid = [10001, 90210, 60614, 33101, 77002]
        invalid = [999999, 123456789, 0]
        s = pd.Series(valid + invalid)
        assert _looks_like_zip(s) is False

    def test_empty_series(self) -> None:
        assert _looks_like_zip(pd.Series([], dtype=float)) is False

    def test_out_of_range_integer(self) -> None:
        s = pd.Series([99999, 99998, 99997, 99996, 99995])  # above 99950
        assert _looks_like_zip(s) is False


# ---------------------------------------------------------------------------
# _is_sequential_or_parseable_date
# ---------------------------------------------------------------------------


class TestIsSequentialOrParseableDate:
    def test_integer_sequence(self) -> None:
        s = pd.Series(list(range(1, 21)))
        assert _is_sequential_or_parseable_date(s) is True

    def test_date_strings(self) -> None:
        s = pd.Series(["2024-01-01", "2024-01-08", "2024-01-15"])
        assert _is_sequential_or_parseable_date(s) is True

    def test_garbage_strings(self) -> None:
        s = pd.Series(["Q1-24", "Q2-24", "spring", "fall"])
        assert _is_sequential_or_parseable_date(s) is False

    def test_single_value_false(self) -> None:
        # Only 1 unique value -- not a meaningful time sequence
        s = pd.Series([1, 1, 1, 1])
        assert _is_sequential_or_parseable_date(s) is False

    def test_numeric_strings(self) -> None:
        s = pd.Series(["1", "2", "3", "4", "5"])
        assert _is_sequential_or_parseable_date(s) is True


# ---------------------------------------------------------------------------
# _is_binary_01
# ---------------------------------------------------------------------------


class TestIsBinary01:
    def test_integer_01(self) -> None:
        s = pd.Series([0, 1, 0, 1, 1])
        assert _is_binary_01(s) is True

    def test_boolean(self) -> None:
        s = pd.Series([True, False, True])
        assert _is_binary_01(s) is True

    def test_float_01(self) -> None:
        s = pd.Series([0.0, 1.0, 0.0])
        assert _is_binary_01(s) is True

    def test_string_01(self) -> None:
        s = pd.Series(["0", "1", "0", "1"])
        assert _is_binary_01(s) is True

    def test_three_values_false(self) -> None:
        s = pd.Series([0, 1, 2])
        assert _is_binary_01(s) is False

    def test_yes_no_false(self) -> None:
        s = pd.Series(["yes", "no"])
        assert _is_binary_01(s) is False


# ---------------------------------------------------------------------------
# _is_high_cardinality
# ---------------------------------------------------------------------------


class TestIsHighCardinality:
    def test_high_cardinality(self) -> None:
        s = pd.Series(range(1000))  # 100% unique
        assert _is_high_cardinality(s, min_ratio=0.5) is True

    def test_low_cardinality(self) -> None:
        s = pd.Series([1, 2, 3] * 100)  # 1% unique
        assert _is_high_cardinality(s, min_ratio=0.5) is False

    def test_empty_series(self) -> None:
        s = pd.Series([], dtype=float)
        assert _is_high_cardinality(s) is False


# ---------------------------------------------------------------------------
# _detect_geographic_id
# ---------------------------------------------------------------------------


class TestDetectGeographicId:
    def test_high_confidence_name_and_value(self) -> None:
        df = _make_df(zip_code=[10001, 90210, 60614, 33101, 77002])
        candidates = {"zip_code"}
        result = _detect_geographic_id(df, candidates)
        assert result is not None
        assert result.confidence == "high"
        assert result.column_name == "zip_code"

    def test_medium_confidence_name_only(self) -> None:
        df = _make_df(zip_code=[1, 2, 3, 4, 5])  # Not real ZIPs
        candidates = {"zip_code"}
        result = _detect_geographic_id(df, candidates)
        assert result is not None
        assert result.confidence == "medium"

    def test_no_candidates(self) -> None:
        df = _make_df(revenue=[1.0, 2.0, 3.0])
        result = _detect_geographic_id(df, {"revenue"})
        assert result is None


# ---------------------------------------------------------------------------
# _detect_customer_id
# ---------------------------------------------------------------------------


class TestDetectCustomerId:
    def test_high_cardinality_customer_id(self) -> None:
        df = _make_df(customer_id=list(range(1, 501)))
        result = _detect_customer_id(df, {"customer_id"})
        assert result is not None
        assert result.column_name == "customer_id"
        assert result.confidence == "medium"

    def test_low_cardinality_gives_low_confidence(self) -> None:
        # Panel: 500 customers x 20 weeks = 10000 rows, but only 500 unique IDs
        cids = list(range(1, 501)) * 20
        df = pd.DataFrame({"customer_id": cids})
        result = _detect_customer_id(df, {"customer_id"})
        assert result is not None
        assert result.confidence == "low"

    def test_account_id_detected(self) -> None:
        df = _make_df(account_id=list(range(1, 201)))
        result = _detect_customer_id(df, {"account_id"})
        assert result is not None


# ---------------------------------------------------------------------------
# _detect_time_period
# ---------------------------------------------------------------------------


class TestDetectTimePeriod:
    def test_week_column_high_confidence(self) -> None:
        df = _make_df(week=list(range(1, 21)))
        result = _detect_time_period(df, {"week"})
        assert result is not None
        assert result.confidence == "high"
        assert result.column_name == "week"

    def test_date_column_high_confidence(self) -> None:
        df = _make_df(date=["2024-01-01", "2024-01-08", "2024-01-15"])
        result = _detect_time_period(df, {"date"})
        assert result is not None
        assert result.confidence == "high"

    def test_no_time_column(self) -> None:
        df = _make_df(revenue=[1.0, 2.0, 3.0])
        result = _detect_time_period(df, {"revenue"})
        assert result is None


# ---------------------------------------------------------------------------
# _detect_treatment_binary
# ---------------------------------------------------------------------------


class TestDetectTreatmentBinary:
    def test_binary_with_treatment_name(self) -> None:
        df = _make_df(channel_email=[0, 1, 0, 1, 1, 0])
        result = _detect_treatment_binary(df, {"channel_email"})
        assert result is not None
        assert result.confidence == "medium"

    def test_binary_without_treatment_name(self) -> None:
        df = _make_df(flag_x=[0, 1, 0, 1])
        result = _detect_treatment_binary(df, {"flag_x"})
        assert result is not None
        # "flag" is in the name hints, so still medium
        assert result.confidence == "medium"

    def test_non_binary_not_detected(self) -> None:
        df = _make_df(multi=[0, 1, 2, 3])
        result = _detect_treatment_binary(df, {"multi"})
        assert result is None

    def test_candidate_only_when_no_name_hint(self) -> None:
        df = _make_df(mystery=[0, 1, 0, 1, 1])
        result = _detect_treatment_binary(df, {"mystery"})
        assert result is not None
        assert result.confidence == "low"


# ---------------------------------------------------------------------------
# _detect_treatment_continuous
# ---------------------------------------------------------------------------


class TestDetectTreatmentContinuous:
    def test_webinar_count_detected_low(self) -> None:
        rng = np.random.default_rng(42)
        vals = rng.lognormal(mean=1.2, sigma=0.8, size=100)
        df = pd.DataFrame({"webinar_count": vals})
        result = _detect_treatment_continuous(df, {"webinar_count"})
        assert result is not None
        assert result.confidence == "low"

    def test_binary_column_not_detected(self) -> None:
        df = _make_df(touch_count=[0, 1, 0, 1])
        result = _detect_treatment_continuous(df, {"touch_count"})
        assert result is None  # binary excluded

    def test_no_name_match_not_detected(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"revenue": rng.exponential(scale=10, size=100)})
        result = _detect_treatment_continuous(df, {"revenue"})
        assert result is None  # "revenue" is not in treatment_continuous hints


# ---------------------------------------------------------------------------
# _detect_outcome
# ---------------------------------------------------------------------------


class TestDetectOutcome:
    def test_revenue_detected(self) -> None:
        df = _make_df(revenue=[10.0, 20.0, 30.0])
        result = _detect_outcome(df, {"revenue"})
        assert result is not None
        assert result.confidence == "medium"

    def test_units_sold_detected(self) -> None:
        df = _make_df(units_sold=[1, 2, 3, 4, 5])
        result = _detect_outcome(df, {"units_sold"})
        assert result is not None

    def test_non_numeric_not_detected(self) -> None:
        df = _make_df(revenue=["high", "low", "medium"])
        result = _detect_outcome(df, {"revenue"})
        assert result is None

    def test_no_match_returns_none(self) -> None:
        df = _make_df(foobar=[1.0, 2.0, 3.0])
        result = _detect_outcome(df, {"foobar"})
        assert result is None


# ---------------------------------------------------------------------------
# detect_columns (integration)
# ---------------------------------------------------------------------------


class TestDetectColumns:
    def test_returns_all_role_keys(self, tmp_path) -> None:
        df = _make_df(
            customer_id=list(range(100)),
            week=list(range(1, 21)) * 5,
            channel_email=[0, 1] * 50,
            revenue=list(np.random.default_rng(0).normal(50, 10, 100)),
            zip_code=[10001] * 100,
        )
        result = detect_columns(df)
        assert set(result.keys()) == {
            "geographic_id",
            "customer_id",
            "time_period",
            "treatment_binary",
            "treatment_continuous",
            "outcome",
            "covariates",
        }

    def test_covariates_is_list(self) -> None:
        df = _make_df(
            customer_id=list(range(100)),
            week=list(range(1, 21)) * 5,
            revenue=list(np.random.default_rng(0).normal(50, 10, 100)),
        )
        result = detect_columns(df)
        assert isinstance(result["covariates"], list)

    def test_column_not_claimed_by_two_roles(self) -> None:
        """No single column should appear in both a single role and covariates."""
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_omnichannel.csv"
        )
        result = detect_columns(df)
        claimed = set()
        for role, suggestion in result.items():
            if role == "covariates":
                for s in suggestion:
                    assert s.column_name not in claimed, (
                        f"Column '{s.column_name}' claimed by covariates and another role."
                    )
                    claimed.add(s.column_name)
            elif suggestion is not None:
                assert suggestion.column_name not in claimed, (
                    f"Column '{suggestion.column_name}' claimed by multiple roles."
                )
                claimed.add(suggestion.column_name)

    def test_synthetic_omnichannel_zip_detected(self) -> None:
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_omnichannel.csv"
        )
        result = detect_columns(df)
        geo = result["geographic_id"]
        assert geo is not None
        assert geo.column_name == "zip_code"
        assert geo.confidence == "high"

    def test_synthetic_omnichannel_week_detected(self) -> None:
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_omnichannel.csv"
        )
        result = detect_columns(df)
        time = result["time_period"]
        assert time is not None
        assert time.column_name == "week"
        assert time.confidence == "high"

    def test_suggestion_has_correct_fields(self) -> None:
        df = _make_df(zip_code=[10001, 90210, 60614])
        result = detect_columns(df)
        geo = result["geographic_id"]
        assert geo is not None
        assert isinstance(geo, ColumnSuggestion)
        assert geo.confidence in ("high", "medium", "low")
        assert len(geo.reason) > 0


# ---------------------------------------------------------------------------
# check_method_eligibility
# ---------------------------------------------------------------------------


class TestCheckMethodEligibility:
    def test_did_eligible_with_multiple_periods(self) -> None:
        df = _make_df(week=list(range(1, 21)) * 10, treated=[0, 1] * 100)
        roles = {"time_period": "week", "treatment_binary": "treated"}
        result = check_method_eligibility(df, roles)
        assert result["DiD"][0] is True
        assert "20" in result["DiD"][1]

    def test_did_ineligible_with_one_period(self) -> None:
        df = _make_df(week=[1] * 100, treated=[0, 1] * 50)
        roles = {"time_period": "week", "treatment_binary": "treated"}
        result = check_method_eligibility(df, roles)
        assert result["DiD"][0] is False
        assert "1 time period" in result["DiD"][1]

    def test_did_ineligible_no_time_column(self) -> None:
        df = _make_df(revenue=[1.0, 2.0])
        result = check_method_eligibility(df, {})
        assert result["DiD"][0] is False

    def test_rdd_eligible_with_continuous_running_var(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"score": rng.uniform(0, 100, 1000)})
        roles = {"treatment_continuous": "score"}
        result = check_method_eligibility(df, roles)
        assert result["RDD"][0] is True

    def test_rdd_ineligible_no_running_variable(self) -> None:
        df = _make_df(revenue=[1.0, 2.0, 3.0])
        result = check_method_eligibility(df, {})
        assert result["RDD"][0] is False
        assert "No continuous running variable" in result["RDD"][1]

    def test_rdd_ineligible_binary_assigned_as_continuous(self) -> None:
        df = _make_df(treated=[0, 1, 0, 1])
        roles = {"treatment_continuous": "treated"}
        result = check_method_eligibility(df, roles)
        assert result["RDD"][0] is False

    def test_scm_eligible_on_scm_synthetic_data(self) -> None:
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_scm.csv"
        )
        roles = {
            "customer_id": "zip_code",
            "time_period": "week",
            "treatment_binary": "treated",
        }
        result = check_method_eligibility(df, roles)
        ok, msg = result["SCM"]
        assert ok is True, f"SCM should be eligible on SCM dataset. Reason: {msg}"
        assert "5 treated units" in msg

    def test_scm_ineligible_too_few_units(self) -> None:
        # Only 2 treated units
        rng = np.random.default_rng(42)
        weeks = list(range(1, 21)) * 10
        units = list(range(1, 11)) * 20
        treated = [1 if u <= 2 else 0 for u in units]
        df = pd.DataFrame({"unit_id": units, "week": weeks, "treated": treated})
        roles = {
            "customer_id": "unit_id",
            "time_period": "week",
            "treatment_binary": "treated",
        }
        result = check_method_eligibility(df, roles)
        assert result["SCM"][0] is False
        assert "2 treated unit" in result["SCM"][1]

    def test_all_methods_returned(self) -> None:
        df = _make_df(week=[1, 2], treated=[0, 1])
        result = check_method_eligibility(df, {"time_period": "week"})
        assert set(result.keys()) == {"DiD", "RDD", "SCM"}

    def test_result_values_are_tuples(self) -> None:
        df = _make_df(week=[1, 2], treated=[0, 1])
        result = check_method_eligibility(df, {"time_period": "week"})
        for method, val in result.items():
            assert isinstance(val, tuple), f"{method} value should be a tuple"
            assert isinstance(val[0], bool)
            assert isinstance(val[1], str)

    def test_synthetic_omnichannel_did_eligible(self) -> None:
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_omnichannel.csv"
        )
        roles = {
            "time_period": "week",
            "treatment_binary": "channel_email",
        }
        result = check_method_eligibility(df, roles)
        assert result["DiD"][0] is True

    def test_synthetic_rdd_rdd_eligible(self) -> None:
        df = pd.read_csv(
            "C:/Users/Admin/research/_incubator/acid-dash/data/sample/synthetic_rdd.csv"
        )
        roles = {
            "treatment_binary": "treated",
            "treatment_continuous": "engagement_score",
        }
        result = check_method_eligibility(df, roles)
        assert result["RDD"][0] is True
