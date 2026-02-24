"""CSV validation utilities for ACID-Dash.

Validates user-uploaded DataFrames against assigned column roles. All checks
produce ValidationWarning namedtuples -- warnings are advisory only. The user
decides whether to proceed; nothing is blocked.
"""

from __future__ import annotations

import re
from collections import namedtuple
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ValidationWarning = namedtuple(
    "ValidationWarning",
    ["column", "severity", "message"],
)
"""A single advisory warning raised during CSV validation.

Attributes:
    column: Column name the warning applies to, or ``"__dataset__"`` for
        dataset-level checks.
    severity: One of ``"info"``, ``"warning"``, or ``"error"`` (still
        advisory -- nothing is blocked, but ``"error"`` severity indicates
        the analysis may produce incorrect results if ignored).
    message: Human-readable description suitable for display in the sidebar.
"""

# Sentinel for dataset-level (non-column-specific) warnings
_DATASET = "__dataset__"

# ZIP validity range (per proposal Section 3)
_ZIP_MIN = 501
_ZIP_MAX = 99950
_ZIP_PATTERN = re.compile(r"^\d{5}$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _null_check(
    df: pd.DataFrame,
    column: str,
    threshold: float = 0.05,
) -> list[ValidationWarning]:
    """Flag columns where null percentage exceeds *threshold*.

    Args:
        df: Full DataFrame.
        column: Column name to inspect.
        threshold: Fraction of nulls above which a warning is raised.
            Defaults to 0.05 (5%).

    Returns:
        List of zero or one ValidationWarning.
    """
    null_pct = df[column].isna().mean()
    if null_pct > threshold:
        pct_str = f"{null_pct:.1%}"
        severity = "error" if null_pct > 0.50 else "warning"
        return [
            ValidationWarning(
                column=column,
                severity=severity,
                message=(
                    f"{pct_str} of values are null. "
                    "Consider imputing (mean/median/mode) or dropping rows "
                    "before analysis."
                ),
            )
        ]
    return []


def _type_inference_check(
    df: pd.DataFrame,
    column: str,
    role: str,
) -> list[ValidationWarning]:
    """Verify that columns assigned to numeric roles actually parse as numeric.

    For roles that expect numeric data (``outcome``, ``treatment_continuous``),
    attempt coercion and report failures.

    Args:
        df: Full DataFrame.
        column: Column name to inspect.
        role: The semantic role assigned to the column (e.g. ``"outcome"``).

    Returns:
        List of zero or one ValidationWarning.
    """
    numeric_roles = {"outcome", "treatment_continuous"}
    if role not in numeric_roles:
        return []

    series = df[column].dropna()
    if pd.api.types.is_numeric_dtype(series):
        return []

    # Attempt coercion to float
    coerced = pd.to_numeric(series, errors="coerce")
    fail_count = coerced.isna().sum()
    if fail_count > 0:
        return [
            ValidationWarning(
                column=column,
                severity="error",
                message=(
                    f"Assigned as '{role}' but {fail_count:,} values cannot "
                    "be parsed as numeric. Check for currency symbols, commas, "
                    "or text entries."
                ),
            )
        ]
    return []


def _date_parsing_check(
    df: pd.DataFrame,
    column: str,
) -> list[ValidationWarning]:
    """Attempt to parse a time-period column as dates or sequential integers.

    Args:
        df: Full DataFrame.
        column: Column name to inspect (already assigned the ``time_period``
            role).

    Returns:
        List of zero or one ValidationWarning.
    """
    series = df[column].dropna()

    # Already integer -- interpret as period index; no warning
    if pd.api.types.is_integer_dtype(series):
        return []

    # Already a datetime type -- fine
    if pd.api.types.is_datetime64_any_dtype(series):
        return []

    # Try parsing as datetime string (pandas 2.x: infer_datetime_format removed)
    try:
        parsed = pd.to_datetime(series, errors="raise")
        if parsed.notna().all():
            return []
    except Exception:
        pass

    # Try numeric coercion (e.g. "1", "2", ...)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return []

    return [
        ValidationWarning(
            column=column,
            severity="warning",
            message=(
                "Could not parse as dates or sequential integers. "
                "Verify the time-period column format "
                "(expected: integers, YYYY-MM-DD strings, or similar)."
            ),
        )
    ]


def _cardinality_check(
    df: pd.DataFrame,
    column: str,
    role: str,
    threshold: int = 50,
) -> list[ValidationWarning]:
    """Warn when a categorical covariate has more unique values than *threshold*.

    High-cardinality columns assigned as covariates risk overfitting the
    propensity model or causing dummy-variable explosion.

    Args:
        df: Full DataFrame.
        column: Column name to inspect.
        role: Semantic role. Only ``"covariates"`` triggers this check.
        threshold: Maximum acceptable number of unique values. Defaults to 50.

    Returns:
        List of zero or one ValidationWarning.
    """
    if role != "covariates":
        return []

    # Only flag object/categorical columns -- numeric covariates are fine
    if pd.api.types.is_numeric_dtype(df[column]):
        return []

    n_unique = df[column].nunique(dropna=True)
    if n_unique > threshold:
        return [
            ValidationWarning(
                column=column,
                severity="warning",
                message=(
                    f"{n_unique} unique categories detected. "
                    "High-cardinality categorical covariates may cause "
                    "instability in propensity models. Consider binning or "
                    "excluding this column."
                ),
            )
        ]
    return []


def _balance_check(
    df: pd.DataFrame,
    treatment_col: str,
) -> list[ValidationWarning]:
    """Warn if treated/control ratio is severely imbalanced.

    Flags if the treated fraction is outside the 10%-90% range. Extreme
    imbalance degrades propensity score overlap and inflates IPW weights.

    Args:
        df: Full DataFrame.
        treatment_col: Column holding the binary treatment indicator.

    Returns:
        List of zero or one ValidationWarning.
    """
    series = df[treatment_col].dropna()

    # Binarise if needed (e.g. True/False or "1"/"0")
    binary = pd.to_numeric(series, errors="coerce")
    if binary.isna().any():
        return []  # Cannot assess -- type check will catch this

    treated_frac = (binary == 1).mean()

    if treated_frac < 0.10:
        return [
            ValidationWarning(
                column=treatment_col,
                severity="warning",
                message=(
                    f"Only {treated_frac:.1%} of rows are treated "
                    f"({int((binary == 1).sum()):,} treated vs "
                    f"{int((binary == 0).sum()):,} control). "
                    "Severe imbalance may produce unreliable estimates. "
                    "Confirm the treatment column is correctly assigned."
                ),
            )
        ]
    if treated_frac > 0.90:
        return [
            ValidationWarning(
                column=treatment_col,
                severity="warning",
                message=(
                    f"{treated_frac:.1%} of rows are treated "
                    f"({int((binary == 1).sum()):,} treated vs "
                    f"{int((binary == 0).sum()):,} control). "
                    "Near-complete treatment exposure leaves very few controls. "
                    "Propensity-based methods may have poor overlap."
                ),
            )
        ]
    return []


def _duplicate_check(
    df: pd.DataFrame,
    customer_id_col: str,
    time_period_col: str,
) -> list[ValidationWarning]:
    """Flag duplicate (customer_id, time_period) rows.

    Duplicate panel rows typically indicate a join error upstream. They inflate
    sample sizes and bias estimates.

    Args:
        df: Full DataFrame.
        customer_id_col: Column holding customer identifiers.
        time_period_col: Column holding time period identifiers.

    Returns:
        List of zero or one ValidationWarning.
    """
    n_dupes = df.duplicated(subset=[customer_id_col, time_period_col]).sum()
    if n_dupes > 0:
        return [
            ValidationWarning(
                column=_DATASET,
                severity="warning",
                message=(
                    f"{n_dupes:,} duplicate (customer_id x time_period) rows "
                    "detected. Duplicates inflate effective sample size and "
                    "may bias estimates. Consider deduplicating before analysis."
                ),
            )
        ]
    return []


def _zip_format_check(
    df: pd.DataFrame,
    column: str,
) -> list[ValidationWarning]:
    """Flag values in a geographic_id column that are not valid 5-digit ZIPs.

    A valid US ZIP is a 5-digit string (or integer) in the range 00501-99950
    (per proposal Section 3).

    Args:
        df: Full DataFrame.
        column: Column name assigned the ``geographic_id`` role.

    Returns:
        List of zero or more ValidationWarnings.
    """
    warnings: list[ValidationWarning] = []
    series = df[column].dropna()

    # Normalise to zero-padded 5-digit strings for uniform validation
    if pd.api.types.is_integer_dtype(series):
        str_series = series.astype(str).str.zfill(5)
    else:
        str_series = series.astype(str).str.strip()

    # Check format: must be exactly 5 digits
    bad_format = str_series[~str_series.str.match(r"^\d{5}$")]
    if len(bad_format) > 0:
        n_bad = len(bad_format)
        examples = ", ".join(bad_format.unique()[:5].tolist())
        warnings.append(
            ValidationWarning(
                column=column,
                severity="warning",
                message=(
                    f"{n_bad:,} values are not 5-digit ZIP codes "
                    f"(examples: {examples}). "
                    "Non-standard ZIPs will be excluded from geographic "
                    "visualization."
                ),
            )
        )
        # Skip range check if format is already broken
        return warnings

    # Check numeric range: 00501-99950
    numeric_zips = str_series.astype(int)
    out_of_range = numeric_zips[
        (numeric_zips < _ZIP_MIN) | (numeric_zips > _ZIP_MAX)
    ]
    if len(out_of_range) > 0:
        n_oor = len(out_of_range)
        examples = ", ".join(str(v).zfill(5) for v in out_of_range.unique()[:5])
        warnings.append(
            ValidationWarning(
                column=column,
                severity="info",
                message=(
                    f"{n_oor:,} ZIP values are outside the valid US range "
                    f"(00501-99950) (examples: {examples}). "
                    "These ZIPs will not match the bundled centroid lookup."
                ),
            )
        )

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_csv(
    df: pd.DataFrame,
    column_roles: dict[str, Any],
) -> list[ValidationWarning]:
    """Validate a user-uploaded DataFrame against its assigned column roles.

    Runs all configured checks and returns a flat list of ValidationWarnings.
    Warnings are advisory -- nothing is blocked. The caller (Streamlit UI)
    decides how to present them and whether to proceed.

    Checks performed:
        - Null percentage (>5% -> warning; >50% -> error)
        - Type inference for numeric roles (outcome, treatment_continuous)
        - Date parsing for time_period column
        - Cardinality warnings for high-cardinality categorical covariates (>50)
        - Treatment balance check (ratio <10:90 or >90:10)
        - Duplicate (customer_id x time_period) row detection
        - ZIP format validation (5-digit, range 00501-99950)

    Args:
        df: The uploaded DataFrame. Must not be empty.
        column_roles: Mapping from role name to column name (str) or list of
            column names (for ``"covariates"``). Recognised role keys::

                {
                    "customer_id":           "<col>",
                    "time_period":           "<col>",
                    "treatment_binary":      "<col>",   # optional
                    "treatment_continuous":  "<col>",   # optional
                    "outcome":               "<col>",
                    "covariates":            ["<col>", ...],
                    "geographic_id":         "<col>",   # optional
                }

            Roles absent from the dict are silently skipped. A role mapped to
            ``None`` or an empty string is also skipped.

    Returns:
        List of ValidationWarning namedtuples, possibly empty if the data
        is clean. Ordered by: dataset-level checks first, then per-column
        checks in role order.

    Raises:
        ValueError: If *df* is empty.

    Example::

        warnings = validate_csv(df, {
            "customer_id":      "customer_id",
            "time_period":      "week",
            "treatment_binary": "channel_email",
            "outcome":          "revenue",
            "covariates":       ["prior_spend", "tenure_years", "industry"],
            "geographic_id":    "zip_code",
        })
        for w in warnings:
            print(f"[{w.severity.upper()}] {w.column}: {w.message}")
    """
    if df.empty:
        raise ValueError("DataFrame is empty; cannot validate.")

    warnings: list[ValidationWarning] = []

    def _resolve(role: str) -> str | None:
        """Return the column name for *role*, or None if unset."""
        val = column_roles.get(role)
        if not val:
            return None
        return str(val)

    def _resolve_list(role: str) -> list[str]:
        """Return the list of column names for a multi-column role."""
        val = column_roles.get(role, [])
        if not val:
            return []
        if isinstance(val, str):
            return [val]
        return [str(c) for c in val]

    # ------------------------------------------------------------------
    # Dataset-level checks first
    # ------------------------------------------------------------------

    customer_col = _resolve("customer_id")
    time_col = _resolve("time_period")

    if customer_col and time_col:
        if customer_col in df.columns and time_col in df.columns:
            warnings.extend(_duplicate_check(df, customer_col, time_col))

    # ------------------------------------------------------------------
    # Per-column checks
    # ------------------------------------------------------------------

    # Single-column roles (ordered for readability in the UI)
    single_roles = [
        ("customer_id", customer_col),
        ("time_period", time_col),
        ("treatment_binary", _resolve("treatment_binary")),
        ("treatment_continuous", _resolve("treatment_continuous")),
        ("outcome", _resolve("outcome")),
        ("geographic_id", _resolve("geographic_id")),
    ]

    for role, col in single_roles:
        if col is None or col not in df.columns:
            continue

        # Null check -- applies to every assigned column
        warnings.extend(_null_check(df, col))

        # Role-specific checks
        if role == "time_period":
            warnings.extend(_date_parsing_check(df, col))

        elif role in ("treatment_continuous", "outcome"):
            warnings.extend(_type_inference_check(df, col, role))

        elif role == "treatment_binary":
            warnings.extend(_balance_check(df, col))

        elif role == "geographic_id":
            warnings.extend(_zip_format_check(df, col))

    # Covariate checks (multi-column)
    for cov_col in _resolve_list("covariates"):
        if cov_col not in df.columns:
            continue
        warnings.extend(_null_check(df, cov_col))
        warnings.extend(_type_inference_check(df, cov_col, "covariates"))
        warnings.extend(_cardinality_check(df, cov_col, "covariates"))

    return warnings
