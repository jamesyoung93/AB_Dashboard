"""Auto-detection heuristics for column role assignment in ACID-Dash.

On CSV upload the app calls :func:`detect_columns` to propose semantic roles
for each column. All suggestions are surfaced in the sidebar dropdowns and
can be manually overridden -- nothing is locked in automatically.

Confidence levels map to UI behaviour as per PROPOSAL.md Section 3:
    - ``"high"``   -- auto-assign; show green checkmark.
    - ``"medium"`` -- auto-suggest; yellow highlight; manual confirm required.
    - ``"low"``    -- do not auto-assign; list as candidate only.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ColumnSuggestion(NamedTuple):
    """A suggested column assignment for a semantic role.

    Attributes:
        column_name: The DataFrame column being suggested.
        confidence: One of ``"high"``, ``"medium"``, or ``"low"``.
        reason: Human-readable explanation shown in the UI tooltip.
    """

    column_name: str
    confidence: str  # "high" | "medium" | "low"
    reason: str


# Convenience alias for the full detection result
DetectionResult = dict[str, "ColumnSuggestion | list[ColumnSuggestion] | None"]
"""Return type of :func:`detect_columns`.

Keys are role names; values are:
    - :class:`ColumnSuggestion` for single-column roles, or
    - ``list[ColumnSuggestion]`` for ``"covariates"``, or
    - ``None`` if no candidate was found.
"""


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_ZIP_MIN = 501
_ZIP_MAX = 99_950

# Name fragments used for heuristic matching (all lower-case)
_NAME_HINTS: dict[str, list[str]] = {
    "zip":                  ["zip", "postal"],
    "customer_id":          ["customer", "account", "client", "user", "id"],
    "time_period":          ["week", "period", "month", "date"],
    "treatment_binary":     ["treat", "flag", "tactic", "campaign", "exposed", "channel"],
    "treatment_continuous": ["touch", "spend", "count", "frequency", "impression"],
    "outcome":              ["revenue", "units", "conversion", "volume", "lift",
                             "outcome", "response", "target", "profit"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _col_lower(col: str) -> str:
    """Return the lower-cased column name."""
    return col.lower()


def _name_matches(col: str, fragments: list[str]) -> bool:
    """Return True if *col* (case-insensitive) contains any of *fragments*."""
    low = _col_lower(col)
    return any(frag in low for frag in fragments)


def _is_numeric(series: pd.Series) -> bool:
    """Return True if *series* has a numeric dtype."""
    return pd.api.types.is_numeric_dtype(series)


def _is_non_negative_numeric(series: pd.Series) -> bool:
    """Return True if *series* is numeric and all non-null values are >= 0."""
    if not _is_numeric(series):
        return False
    return series.dropna().ge(0).all()


def _cardinality_ratio(series: pd.Series) -> float:
    """Return unique-count / non-null-count. Ranges 0 to 1."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    return series.nunique(dropna=True) / len(non_null)


def _is_high_cardinality(series: pd.Series, min_ratio: float = 0.5) -> bool:
    """Return True if the column has high cardinality relative to row count.

    Args:
        series: The column to evaluate.
        min_ratio: Minimum ratio of unique values to non-null rows.
            Defaults to 0.5 (at least 50% distinct values).
    """
    return _cardinality_ratio(series) >= min_ratio


def _looks_like_zip(series: pd.Series) -> bool:
    """Return True if the majority of non-null values look like valid ZIPs.

    A value is a valid ZIP if it represents a 5-digit integer in 00501-99950.

    Args:
        series: Column to inspect (any dtype).
    """
    s = series.dropna()
    if len(s) == 0:
        return False

    if _is_numeric(s):
        # Integer column: check range and that values fit in 5 digits
        try:
            vals = s.astype(int)
            in_range = vals.between(_ZIP_MIN, _ZIP_MAX)
            five_digit = vals.between(1_000, 99_999)
            return bool((in_range & five_digit).mean() >= 0.90)
        except (ValueError, TypeError):
            return False

    # String column: zero-pad then pattern-match
    try:
        str_vals = s.astype(str).str.strip().str.zfill(5)
        pattern_ok = str_vals.str.match(r"^\d{5}$")
        numeric = pd.to_numeric(str_vals, errors="coerce")
        in_range = numeric.between(_ZIP_MIN, _ZIP_MAX)
        return bool((pattern_ok & in_range).mean() >= 0.90)
    except Exception:
        return False


def _is_sequential_or_parseable_date(series: pd.Series) -> bool:
    """Return True if *series* looks like sequential periods or parseable dates.

    Args:
        series: Column to inspect.
    """
    s = series.dropna()
    if len(s) == 0:
        return False

    # Already an integer type with reasonable period count
    if pd.api.types.is_integer_dtype(s):
        n_unique = s.nunique()
        # Reasonable period range: 1 to 10,000
        return bool(1 < n_unique <= 10_000)

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return True

    # Try parsing as datetime string (pandas 2.x: infer_datetime_format removed)
    try:
        pd.to_datetime(s, errors="raise")
        return True
    except Exception:
        pass

    # Try numeric (e.g. week stored as string "1", "2", ...)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        n_unique = numeric.nunique()
        return bool(1 < n_unique <= 10_000)

    return False


def _is_binary_01(series: pd.Series) -> bool:
    """Return True if *series* has exactly two unique values in {0, 1, True, False}.

    Args:
        series: Column to inspect.
    """
    vals = series.dropna()
    unique = set(vals.unique())
    return unique in ({0, 1}, {True, False}, {0.0, 1.0}, {"0", "1"})


def _has_moderate_variance(series: pd.Series) -> bool:
    """Return True if *series* has meaningful variance (coefficient of variation >= 0.1).

    Args:
        series: Numeric column.
    """
    s = series.dropna()
    if len(s) < 2:
        return False
    mean = s.mean()
    std = s.std()
    if mean == 0:
        return std > 0
    return abs(std / mean) >= 0.10


def _is_low_cardinality_categorical(series: pd.Series, max_unique: int = 50) -> bool:
    """Return True if *series* is non-numeric with <= *max_unique* distinct values.

    Args:
        series: Column to evaluate.
        max_unique: Ceiling on unique value count. Defaults to 50.
    """
    if _is_numeric(series):
        return False
    return series.nunique(dropna=True) <= max_unique


# ---------------------------------------------------------------------------
# Role detectors (one per role)
# ---------------------------------------------------------------------------


def _detect_geographic_id(
    df: pd.DataFrame,
    candidates: set[str],
) -> ColumnSuggestion | None:
    """Detect the geographic ID (ZIP code) column.

    High confidence when column name contains 'zip'/'postal' AND values pass
    the 5-digit range check.

    Args:
        df: Full DataFrame.
        candidates: Set of column names not yet claimed by another role.

    Returns:
        ColumnSuggestion or None.
    """
    for col in (c for c in df.columns if c in candidates):
        name_hit = _name_matches(col, _NAME_HINTS["zip"])
        value_hit = _looks_like_zip(df[col])

        if name_hit and value_hit:
            return ColumnSuggestion(
                column_name=col,
                confidence="high",
                reason=(
                    "Column name contains 'zip'/'postal' and >=90% of "
                    "values are valid 5-digit US ZIPs."
                ),
            )
        if name_hit and not value_hit:
            # Name match but values don't look like ZIPs -- flag as medium
            return ColumnSuggestion(
                column_name=col,
                confidence="medium",
                reason=(
                    "Column name suggests a ZIP field but values do not all "
                    "match the 5-digit US format. Confirm before proceeding."
                ),
            )
        if value_hit and not name_hit:
            return ColumnSuggestion(
                column_name=col,
                confidence="medium",
                reason=(
                    "Values look like 5-digit US ZIPs but the column name "
                    "does not contain 'zip'/'postal'. Confirm assignment."
                ),
            )
    return None


def _detect_customer_id(
    df: pd.DataFrame,
    candidates: set[str],
) -> ColumnSuggestion | None:
    """Detect the customer / account identifier column.

    Args:
        df: Full DataFrame.
        candidates: Unclaimed column names.

    Returns:
        ColumnSuggestion or None.
    """
    for col in (c for c in df.columns if c in candidates):
        name_hit = _name_matches(col, _NAME_HINTS["customer_id"])
        high_card = _is_high_cardinality(df[col], min_ratio=0.5)

        if name_hit and high_card:
            return ColumnSuggestion(
                column_name=col,
                confidence="medium",
                reason=(
                    "Column name matches customer/account/client/user/id "
                    "pattern and has high cardinality (>=50% unique values). "
                    "Confirm this is the unique customer identifier."
                ),
            )
        if name_hit:
            return ColumnSuggestion(
                column_name=col,
                confidence="low",
                reason=(
                    "Column name matches customer/account/client/user/id "
                    "pattern but cardinality is lower than expected for a "
                    "unique key. Confirm before assigning."
                ),
            )
    return None


def _detect_time_period(
    df: pd.DataFrame,
    candidates: set[str],
) -> ColumnSuggestion | None:
    """Detect the time-period column.

    Args:
        df: Full DataFrame.
        candidates: Unclaimed column names.

    Returns:
        ColumnSuggestion or None.
    """
    for col in (c for c in df.columns if c in candidates):
        name_hit = _name_matches(col, _NAME_HINTS["time_period"])
        value_hit = _is_sequential_or_parseable_date(df[col])

        if name_hit and value_hit:
            return ColumnSuggestion(
                column_name=col,
                confidence="high",
                reason=(
                    "Column name contains 'week'/'period'/'month'/'date' and "
                    "values are sequential integers or parseable dates."
                ),
            )
        if name_hit:
            return ColumnSuggestion(
                column_name=col,
                confidence="medium",
                reason=(
                    "Column name suggests a time period but values could not "
                    "be confirmed as sequential integers or dates."
                ),
            )
    return None


def _detect_treatment_binary(
    df: pd.DataFrame,
    candidates: set[str],
) -> ColumnSuggestion | None:
    """Detect a binary (0/1) treatment indicator column.

    Uses a two-pass approach: first looks for binary columns whose names
    match treatment patterns (medium confidence), then falls back to any
    binary column (low confidence).  Iterates in sorted order for
    deterministic results across runs.

    Args:
        df: Full DataFrame.
        candidates: Unclaimed column names.

    Returns:
        ColumnSuggestion or None.
    """
    # Pass 1: binary + name match (medium confidence)
    for col in (c for c in df.columns if c in candidates):
        if _is_binary_01(df[col]) and _name_matches(col, _NAME_HINTS["treatment_binary"]):
            return ColumnSuggestion(
                column_name=col,
                confidence="medium",
                reason=(
                    "Exactly 2 unique values {0, 1} and column name matches "
                    "treat/flag/tactic/campaign/exposed/channel pattern."
                ),
            )
    # Pass 2: any binary column (low confidence)
    for col in (c for c in df.columns if c in candidates):
        if _is_binary_01(df[col]):
            return ColumnSuggestion(
                column_name=col,
                confidence="low",
                reason=(
                    "Exactly 2 unique values {0, 1} but column name does not "
                    "match a treatment pattern. Candidate only."
                ),
            )
    return None


def _detect_treatment_continuous(
    df: pd.DataFrame,
    candidates: set[str],
) -> ColumnSuggestion | None:
    """Detect a continuous (non-binary) treatment / exposure column.

    Args:
        df: Full DataFrame.
        candidates: Unclaimed column names.

    Returns:
        ColumnSuggestion or None (always low confidence per proposal).
    """
    for col in (c for c in df.columns if c in candidates):
        name_hit = _name_matches(col, _NAME_HINTS["treatment_continuous"])
        non_neg = _is_non_negative_numeric(df[col])
        moderate_var = _has_moderate_variance(df[col]) if non_neg else False
        # Must not already be binary (that's covered by treatment_binary)
        is_binary = _is_binary_01(df[col])

        if name_hit and non_neg and moderate_var and not is_binary:
            return ColumnSuggestion(
                column_name=col,
                confidence="low",
                reason=(
                    "Column name matches touch/spend/count/frequency/impression "
                    "and values are non-negative numeric with moderate variance. "
                    "Listed as continuous treatment candidate -- confirm before use."
                ),
            )
    return None


def _detect_outcome(
    df: pd.DataFrame,
    candidates: set[str],
) -> ColumnSuggestion | None:
    """Detect the primary outcome column.

    Args:
        df: Full DataFrame.
        candidates: Unclaimed column names.

    Returns:
        ColumnSuggestion or None.
    """
    for col in (c for c in df.columns if c in candidates):
        name_hit = _name_matches(col, _NAME_HINTS["outcome"])
        numeric = _is_numeric(df[col])

        if name_hit and numeric:
            return ColumnSuggestion(
                column_name=col,
                confidence="medium",
                reason=(
                    "Column name matches revenue/units/conversion/score/"
                    "volume/lift/spend pattern and is numeric."
                ),
            )
    return None


def _detect_covariates(
    df: pd.DataFrame,
    candidates: set[str],
) -> list[ColumnSuggestion]:
    """Suggest all remaining unclaimed columns as covariates.

    A covariate candidate is:
        - Any numeric column not assigned to another role, OR
        - Any low-cardinality categorical column (<=50 unique values).

    Args:
        df: Full DataFrame.
        candidates: Unclaimed column names.

    Returns:
        List of ColumnSuggestions (may be empty).
    """
    suggestions: list[ColumnSuggestion] = []
    for col in sorted(candidates):
        series = df[col]
        if _is_numeric(series):
            suggestions.append(
                ColumnSuggestion(
                    column_name=col,
                    confidence="medium",
                    reason="Numeric column not assigned to another role; suggested as covariate.",
                )
            )
        elif _is_low_cardinality_categorical(series):
            suggestions.append(
                ColumnSuggestion(
                    column_name=col,
                    confidence="medium",
                    reason=(
                        f"Categorical column with {series.nunique(dropna=True)} unique values; "
                        "suggested as covariate."
                    ),
                )
            )
    return suggestions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_columns(df: pd.DataFrame) -> DetectionResult:
    """Apply heuristic column role detection to a user-uploaded DataFrame.

    Each column is evaluated for a set of semantic roles using name-based and
    value-based heuristics defined in PROPOSAL.md Section 3. The function
    assigns columns greedily in priority order:

    1. ``geographic_id`` (ZIP)
    2. ``customer_id``
    3. ``time_period``
    4. ``treatment_binary``
    5. ``treatment_continuous``
    6. ``outcome``
    7. ``covariates`` (all remaining eligible columns)

    A column is claimed by at most one single-column role. Covariates receive
    everything left over.

    Args:
        df: The uploaded DataFrame. Must have at least one column.

    Returns:
        Dictionary with the following keys::

            {
                "geographic_id":         ColumnSuggestion | None,
                "customer_id":           ColumnSuggestion | None,
                "time_period":           ColumnSuggestion | None,
                "treatment_binary":      ColumnSuggestion | None,
                "treatment_continuous":  ColumnSuggestion | None,
                "outcome":               ColumnSuggestion | None,
                "covariates":            list[ColumnSuggestion],
            }

    Example::

        import pandas as pd
        from modules.column_detector import detect_columns

        df = pd.read_csv("data/sample/synthetic_omnichannel.csv")
        roles = detect_columns(df)

        for role, suggestion in roles.items():
            if isinstance(suggestion, list):
                for s in suggestion:
                    print(f"  covariate [{s.confidence}]: {s.column_name}")
            elif suggestion:
                print(f"{role} [{suggestion.confidence}]: {suggestion.column_name}")
    """
    remaining: set[str] = set(df.columns)

    def _claim(col: str) -> None:
        remaining.discard(col)

    result: DetectionResult = {
        "geographic_id": None,
        "customer_id": None,
        "time_period": None,
        "treatment_binary": None,
        "treatment_continuous": None,
        "outcome": None,
        "covariates": [],
    }

    # 1 - Geographic ID
    geo = _detect_geographic_id(df, remaining)
    if geo:
        result["geographic_id"] = geo
        _claim(geo.column_name)

    # 2 - Customer ID
    cid = _detect_customer_id(df, remaining)
    if cid:
        result["customer_id"] = cid
        _claim(cid.column_name)

    # 3 - Time period
    time = _detect_time_period(df, remaining)
    if time:
        result["time_period"] = time
        _claim(time.column_name)

    # 4 - Binary treatment
    tbin = _detect_treatment_binary(df, remaining)
    if tbin:
        result["treatment_binary"] = tbin
        _claim(tbin.column_name)

    # 5 - Continuous treatment
    tcont = _detect_treatment_continuous(df, remaining)
    if tcont:
        result["treatment_continuous"] = tcont
        _claim(tcont.column_name)

    # 6 - Outcome
    outcome = _detect_outcome(df, remaining)
    if outcome:
        result["outcome"] = outcome
        _claim(outcome.column_name)

    # 7 - Covariates (everything left that qualifies)
    result["covariates"] = _detect_covariates(df, remaining)

    return result


# ---------------------------------------------------------------------------
# Method eligibility
# ---------------------------------------------------------------------------


def check_method_eligibility(
    df: pd.DataFrame,
    column_roles: dict[str, "str | list[str] | None"],
) -> dict[str, tuple[bool, str]]:
    """Determine which causal inference methods are eligible given the data.

    Checks structural requirements for each method and returns a dict mapping
    method name to ``(eligible: bool, reason: str)``. Ineligible methods are
    greyed out in the UI with the reason shown as a tooltip.

    Methods evaluated:

    - **DiD** (Difference-in-Differences): requires >=2 distinct time periods.
    - **RDD** (Regression Discontinuity): requires a designated continuous
      running variable (i.e. ``treatment_continuous`` role is assigned).
    - **SCM** (Synthetic Control): requires >=5 pre-treatment periods and
      >=3 distinct treated units.

    Args:
        df: The uploaded DataFrame.
        column_roles: Mapping of role name to column name (or list for
            covariates). Expected keys: ``"time_period"``,
            ``"treatment_binary"``, ``"treatment_continuous"``. Missing or
            ``None`` roles are treated as unassigned.

    Returns:
        Dict with keys ``"DiD"``, ``"RDD"``, ``"SCM"``. Each value is a
        ``(eligible, reason)`` tuple::

            {
                "DiD": (True,  "20 distinct time periods detected."),
                "RDD": (False, "No continuous running variable assigned. "
                               "Assign a column to the 'Treatment (continuous)' "
                               "role to enable RDD."),
                "SCM": (True,  "10 pre-treatment periods and 5 treated units detected."),
            }

    Example::

        eligibility = check_method_eligibility(df, column_roles)
        for method, (ok, msg) in eligibility.items():
            status = "eligible" if ok else "ineligible"
            print(f"{method}: {status} -- {msg}")
    """

    def _resolve(role: str) -> str | None:
        val = column_roles.get(role)
        return str(val) if val else None

    time_col = _resolve("time_period")
    tbin_col = _resolve("treatment_binary")
    tcont_col = _resolve("treatment_continuous")

    eligibility: dict[str, tuple[bool, str]] = {}

    # ------------------------------------------------------------------
    # DiD: requires >=2 distinct time periods
    # ------------------------------------------------------------------
    if time_col and time_col in df.columns:
        n_periods = df[time_col].nunique(dropna=True)
        if n_periods >= 2:
            eligibility["DiD"] = (
                True,
                f"{n_periods} distinct time periods detected.",
            )
        else:
            eligibility["DiD"] = (
                False,
                f"Only {n_periods} time period detected. DiD requires at "
                "least 2 periods (pre and post). Assign a time-period column "
                "with >=2 distinct values.",
            )
    else:
        eligibility["DiD"] = (
            False,
            "No time-period column assigned. DiD requires a time column "
            "with >=2 distinct periods.",
        )

    # ------------------------------------------------------------------
    # RDD: requires a designated continuous running variable
    # ------------------------------------------------------------------
    if tcont_col and tcont_col in df.columns:
        series = df[tcont_col].dropna()
        n_unique = series.nunique()
        if n_unique > 2 and _is_numeric(series):
            eligibility["RDD"] = (
                True,
                f"Continuous running variable '{tcont_col}' assigned "
                f"({n_unique} unique values).",
            )
        else:
            eligibility["RDD"] = (
                False,
                f"'{tcont_col}' does not appear to be a continuous variable "
                f"({n_unique} unique values). RDD requires a numeric running "
                "variable with more than 2 distinct values.",
            )
    else:
        eligibility["RDD"] = (
            False,
            "No continuous running variable assigned. Assign a column to "
            "the 'Treatment (continuous)' role to enable RDD.",
        )

    # ------------------------------------------------------------------
    # SCM: requires >=5 pre-treatment periods and >=3 treated units
    # ------------------------------------------------------------------
    scm_ok = True
    scm_reasons: list[str] = []

    # Count pre-treatment periods -- requires both time and binary treatment
    if time_col and tbin_col and time_col in df.columns and tbin_col in df.columns:
        try:
            treat_series = pd.to_numeric(df[tbin_col], errors="coerce")
            time_numeric = pd.to_numeric(df[time_col], errors="coerce")

            if treat_series.dropna().empty or (treat_series == 1).sum() == 0:
                scm_ok = False
                scm_reasons.append("No treated observations found in the treatment column.")
            else:
                # Determine whether the treatment flag is time-varying or
                # unit-level (constant within units across all periods).
                #
                # Time-varying: onset = earliest period where treated==1.
                # Unit-level flag: treatment is constant; estimate onset as
                # the median time period (assumes a reasonable pre/post split
                # exists in the data, as SCM requires it).
                cid_col = _resolve("customer_id")

                if cid_col and cid_col in df.columns:
                    # Check variance of treatment within units
                    unit_treat_var = (
                        df.groupby(cid_col)[tbin_col]
                        .apply(lambda x: pd.to_numeric(x, errors="coerce").std())
                        .fillna(0)
                    )
                    treatment_is_time_varying = bool((unit_treat_var > 0).any())
                else:
                    treatment_is_time_varying = False

                all_periods = time_numeric.dropna().sort_values().unique()
                n_total_periods = len(all_periods)

                if treatment_is_time_varying:
                    # Find earliest period where at least one unit switches on
                    onset = time_numeric[treat_series == 1].min()
                    pre_periods = int((time_numeric < onset).sum() > 0)
                    pre_periods = time_numeric[time_numeric < onset].nunique()
                    pre_label = f"{pre_periods} pre-treatment periods detected (time-varying treatment)."
                else:
                    # Unit-level flag: assume the dataset spans both pre and
                    # post periods; count total periods as a proxy.  SCM will
                    # need the analyst to specify which periods are pre-treatment
                    # in the analysis module itself.
                    pre_periods = n_total_periods // 2
                    pre_label = (
                        f"Treatment indicator appears unit-level (constant within units). "
                        f"Estimated {pre_periods} usable pre-treatment periods "
                        f"(from {n_total_periods} total). Confirm treatment onset in analysis."
                    )

                if pre_periods < 5:
                    scm_ok = False
                    scm_reasons.append(
                        f"Only {pre_periods} pre-treatment period(s) detected "
                        "(>=5 required for Synthetic Control). " + pre_label
                    )
                else:
                    scm_reasons.append(pre_label)

                # Count distinct treated units
                if cid_col and cid_col in df.columns:
                    n_treated_units = df.loc[treat_series == 1, cid_col].nunique()
                else:
                    # No unit ID: use row count as a rough proxy
                    n_treated_units = int((treat_series == 1).sum())

                if n_treated_units < 3:
                    scm_ok = False
                    scm_reasons.append(
                        f"Only {n_treated_units} treated unit(s) detected "
                        "(>=3 required for Synthetic Control)."
                    )
                else:
                    scm_reasons.append(f"{n_treated_units} treated units detected.")

        except Exception:
            scm_ok = False
            scm_reasons.append(
                "Could not parse time or treatment column to assess SCM eligibility."
            )
    else:
        scm_ok = False
        scm_reasons.append(
            "Both a time-period column and a binary treatment column must be "
            "assigned to assess SCM eligibility."
        )

    eligibility["SCM"] = (scm_ok, " ".join(scm_reasons))

    return eligibility
