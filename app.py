"""ACID-Dash: A/B Test Causal Inference Dashboard.

Reorganised for stakeholder consumption:
  Tab 1 - Executive Summary: pre-computed DiD & PSM for every binary
          tactic, top-N positive/negative lift table with assumption
          checks, and an interactive subnational map.
  Tab 2 - Manual Analysis: full parameter controls for power users.
  Tab 3 - Balance Diagnostics: SMD tables, Love plots, overlap.
  Tab 4 - Geographic Explorer: extended map views.

Launch:
    streamlit run app.py --server.address 127.0.0.1 --server.headless true
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modules.balance import (
    compute_balance,
    covariate_distribution_plot,
    love_plot,
    propensity_overlap_plot,
)
from modules.column_detector import check_method_eligibility, detect_columns
from modules.did import (
    DidResult,
    parallel_trends_data,
    run_ancova,
    run_did,
    run_event_study,
)
from modules.geo import zip_outcome_map, zip_treatment_map
from modules.ipw import IpwResult, compute_ipw_weights, run_ipw
from modules.propensity import fit_propensity
from modules.psm import PsmResult, get_matched_data, run_psm
from modules.threshold import (
    binarize_treatment,
    compute_threshold_stats,
    suggest_thresholds,
)
from utils.validators import validate_csv

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ACID-Dash",
    page_icon="\u2697",  # alembic
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Sticky tab bar: keep tabs visible when scrolling
st.markdown(
    """
    <style>
    /* Pin the Streamlit tab bar to the top of the viewport */
    div[data-testid="stTabs"] > div[role="tablist"] {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: var(--background-color, white);
        padding-top: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stTabs"] > div[role="tablist"] {
            background-color: var(--background-color, #0e1117);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_DIR = Path(__file__).parent / "data" / "sample"

# Per-sample-dataset preferred column assignments
SAMPLE_COLUMN_DEFAULTS: dict[str, dict[str, str | list[str] | None]] = {
    "synthetic_omnichannel.csv": {
        "customer_id": "customer_id",
        "time_period": "week",
        "treatment_binary": None,
        "treatment_continuous": "tv_streaming_impressions",
        "outcome": "units_sold",
        "geographic_id": "zip_code",
        "covariates": ["prior_spend", "region"],
    },
    "synthetic_campaigns.csv": {
        "customer_id": "customer_id",
        "time_period": "week",
        "treatment_binary": None,
        "treatment_continuous": None,
        "outcome": "revenue",
        "geographic_id": "zip_code",
        "covariates": [
            "prior_spend", "tenure_years", "engagement_score",
            "company_size", "industry", "region",
        ],
    },
}

TOP_N = 5  # rows per direction in the executive summary table

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "df": None,
    "column_roles": {},
    "validation_warnings": [],
    "method_eligibility": {},
    "propensity_result": None,
    "did_result": None,
    "psm_result": None,
    "ipw_result": None,
    "event_study_result": None,
    "balance_raw": None,
    "balance_adjusted": None,
    "balance_ipw": None,
    "analysis_run": False,
    "threshold_value": None,
    "treatment_binarized": False,
    "_data_source": None,
    # Auto-analysis state
    "auto_results": None,
    "auto_covariates": None,
    "auto_selected_treatment": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Auto-analysis: covariate selection by variance
# ---------------------------------------------------------------------------


def _select_covariates_by_variance(
    df: pd.DataFrame,
    exclude_cols: set[str],
    max_covariates: int = 10,
    min_cv: float = 0.1,
) -> list[str]:
    """Pick the most informative covariates based on coefficient of variation.

    Numeric columns are ranked by CV (std/|mean|). Low-cardinality
    categoricals (<=20 unique values) are always included since they are
    important confounders (region, industry, etc.).
    """
    numeric_candidates: list[tuple[str, float]] = []
    categorical_picks: list[str] = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue

        if pd.api.types.is_numeric_dtype(series):
            unique_vals = set(series.unique())
            # Skip binary cols (they are treatment candidates, not covariates)
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                continue
            mean = series.mean()
            std = series.std()
            cv = abs(std / mean) if mean != 0 else std
            if cv >= min_cv:
                numeric_candidates.append((col, cv))
        elif series.nunique() <= 20:
            categorical_picks.append(col)

    # Sort numeric by CV descending
    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
    numeric_picks = [col for col, _ in numeric_candidates[:max_covariates]]

    return numeric_picks + categorical_picks


# ---------------------------------------------------------------------------
# Auto-analysis: detect treatment candidates
# ---------------------------------------------------------------------------


def _detect_treatment_candidates(
    df: pd.DataFrame,
    exclude_cols: set[str],
) -> list[tuple[str, str, str]]:
    """Return (analysis_col, display_label, source) for each treatment candidate.

    Binary columns (0/1) outside *exclude_cols* are included directly.
    Continuous columns whose names contain channel/tactic/campaign keywords
    are auto-binarized at the median.
    """
    _TREATMENT_KEYWORDS = [
        "channel", "tactic", "campaign", "touch", "impression",
        "exposed", "treat", "flag",
    ]
    candidates: list[tuple[str, str, str]] = []
    seen_cols: set[str] = set()

    for col in df.columns:
        if col in exclude_cols:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Binary 0/1 columns
        if pd.api.types.is_numeric_dtype(series):
            unique = set(series.unique())
            if unique.issubset({0, 1, 0.0, 1.0}) and len(unique) == 2:
                candidates.append((col, col, "binary"))
                seen_cols.add(col)
                continue

        # Continuous columns with treatment-like names -> auto-binarize
        if col not in seen_cols and pd.api.types.is_numeric_dtype(series):
            col_lower = col.lower()
            if any(kw in col_lower for kw in _TREATMENT_KEYWORDS):
                median_val = float(series.median())
                bin_col = f"_auto_{col}_bin"
                df[bin_col] = (df[col] >= median_val).astype(int)
                label = f"{col} (>= {median_val:.1f})"
                candidates.append((bin_col, label, "binarized"))
                seen_cols.add(col)

    return candidates


# ---------------------------------------------------------------------------
# Auto-analysis: assumption checks
# ---------------------------------------------------------------------------


def _check_parallel_trends(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    post_start: int | float,
) -> bool:
    """Return True if pre-period trends are approximately parallel.

    Fits Y ~ treated * time_trend in the pre-period and checks whether
    the interaction coefficient (differential trend) has p > 0.05.
    """
    try:
        import statsmodels.formula.api as smf

        pre = df[df[time_col] < post_start].copy()
        if len(pre) < 20:
            return False
        pre["_t"] = pre[treatment_col].astype(int)
        pre["_time"] = pd.to_numeric(pre[time_col], errors="coerce")
        pre["_t_time"] = pre["_t"] * pre["_time"]
        pre = pre.dropna(subset=[outcome_col, "_t", "_time"])
        if len(pre) < 20:
            return False
        model = smf.ols(f"Q('{outcome_col}') ~ _t + _time + _t_time", data=pre)
        fit = model.fit()
        pval = fit.pvalues.get("_t_time", 0.0)
        return float(pval) > 0.05
    except Exception:
        return False


def _check_balance(balance_result) -> bool:
    """True if all covariates pass SMD < 0.1."""
    if balance_result is None:
        return False
    return balance_result.all_pass


def _check_overlap(auc: float) -> bool:
    """True if propensity model AUC < 0.85 (adequate overlap)."""
    return auc < 0.85


def _check_sample_size(n_treated: int, n_control: int, threshold: int = 30) -> bool:
    return n_treated >= threshold and n_control >= threshold


# ---------------------------------------------------------------------------
# Auto-analysis: result container
# ---------------------------------------------------------------------------


@dataclass
class AutoResult:
    treatment_col: str
    treatment_label: str
    source: str  # 'binary' or 'binarized'
    did_result: DidResult | None = None
    psm_result: PsmResult | None = None
    propensity_auc: float = 0.5
    overall_lift: float = 0.0  # naive mean difference
    parallel_trends_ok: bool = False
    balance_ok: bool = False
    overlap_ok: bool = False
    sample_size_ok: bool = False
    balance_result: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Auto-analysis: main runner
# ---------------------------------------------------------------------------


def _run_auto_analysis(
    df: pd.DataFrame,
    treatments: list[tuple[str, str, str]],
    outcome_col: str,
    time_col: str | None,
    post_start: int | float | None,
    covariates: list[str],
    entity_col: str | None,
    zip_col: str | None,
) -> list[AutoResult]:
    """Run DiD and PSM for every treatment candidate. Returns sorted by lift."""
    results: list[AutoResult] = []

    for treat_col, label, source in treatments:
        if treat_col not in df.columns:
            continue

        ar = AutoResult(
            treatment_col=treat_col,
            treatment_label=label,
            source=source,
        )

        treatment = df[treat_col].astype(int)
        n_treated = int((treatment == 1).sum())
        n_control = int((treatment == 0).sum())
        ar.sample_size_ok = _check_sample_size(n_treated, n_control)

        if n_treated < 5 or n_control < 5:
            results.append(ar)
            continue

        # Naive overall lift
        y = df[outcome_col]
        ar.overall_lift = float(y[treatment == 1].mean() - y[treatment == 0].mean())

        # DiD
        if time_col and post_start is not None:
            try:
                did_res = run_did(
                    df=df,
                    outcome_col=outcome_col,
                    treatment_col=treat_col,
                    time_col=time_col,
                    post_period_start=post_start,
                    covariate_cols=covariates if covariates else None,
                    entity_col=entity_col,
                )
                ar.did_result = did_res
            except Exception:
                pass

            # Parallel trends check
            ar.parallel_trends_ok = _check_parallel_trends(
                df, outcome_col, treat_col, time_col, post_start,
            )

        # PSM
        if covariates:
            try:
                prop = fit_propensity(
                    df=df,
                    treatment_col=treat_col,
                    covariate_cols=covariates,
                    method="logistic",
                )
                ar.propensity_auc = prop.auc
                ar.overlap_ok = _check_overlap(prop.auc)

                psm_res = run_psm(
                    df=df,
                    outcome_col=outcome_col,
                    treatment_col=treat_col,
                    propensity_scores=prop.scores,
                    method="nearest_1to1",
                    n_bootstrap=200,  # fewer for speed
                )
                ar.psm_result = psm_res

                # Post-match balance
                mi = psm_res.matched_indices
                bal = compute_balance(
                    df, treat_col, covariates,
                    matched_indices=(
                        mi["treated_idx"].values,
                        mi["control_idx"].values,
                    ),
                )
                ar.balance_ok = _check_balance(bal)
                ar.balance_result = bal
            except Exception:
                pass

        results.append(ar)

    # Sort by absolute overall lift descending
    results.sort(key=lambda r: abs(r.overall_lift), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Helpers (shared)
# ---------------------------------------------------------------------------


def _reset_results() -> None:
    """Clear all analysis results when data or config changes."""
    for key in [
        "propensity_result", "did_result", "psm_result", "ipw_result",
        "event_study_result", "ancova_result", "balance_raw",
        "balance_adjusted", "balance_ipw", "analysis_run",
        "auto_results", "auto_covariates",
    ]:
        st.session_state[key] = None if key != "analysis_run" else False


def _reset_treatment_continuous() -> None:
    st.session_state.treatment_binarized = False
    st.session_state.threshold_value = None
    if st.session_state.df is not None and "_treatment_binary" in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.drop(columns=["_treatment_binary"])
    _reset_results()


def _get_analysis_treatment_col(
    treatment_binary_col: str | None,
    treatment_continuous_col: str | None,
    df: pd.DataFrame,
) -> str | None:
    if treatment_binary_col:
        return treatment_binary_col
    if treatment_continuous_col:
        if (
            st.session_state.get("treatment_binarized")
            and "_treatment_binary" in df.columns
        ):
            return "_treatment_binary"
    return None


def _assumption_icon(ok: bool) -> str:
    return "Pass" if ok else "Fail"


# ---------------------------------------------------------------------------
# Sidebar: Data source (minimal — collapsed by default)
# ---------------------------------------------------------------------------

st.sidebar.title("ACID-Dash")
st.sidebar.caption("A/B Test Causal Inference Dashboard")

upload_mode = st.sidebar.radio(
    "Data source",
    ["Use sample data", "Upload CSV"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

df: pd.DataFrame | None = None

if upload_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        source_key = f"upload:{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("_data_source") != source_key:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state._data_source = source_key
                st.session_state.treatment_binarized = False
                st.session_state.threshold_value = None
                _reset_results()
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")
else:
    sample_files = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
    if sample_files:
        sample_name = st.sidebar.selectbox(
            "Select sample dataset",
            [f.name for f in sample_files],
        )
        source_key = f"sample:{sample_name}"
        if st.session_state.get("_data_source") != source_key:
            sample_path = SAMPLE_DIR / sample_name
            df = pd.read_csv(sample_path)
            st.session_state.df = df
            st.session_state._data_source = source_key
            st.session_state.treatment_binarized = False
            st.session_state.threshold_value = None
            _reset_results()
    else:
        st.sidebar.warning("No sample data found.")

df = st.session_state.df

if df is None:
    st.title("ACID-Dash")
    st.info(
        "Upload a CSV or select sample data from the sidebar to get started.\n\n"
        "ACID-Dash helps you evaluate the causal impact of engagement tactics "
        "using Difference-in-Differences, Propensity Score Matching, "
        "Inverse Probability Weighting, and more."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: Column Assignment (hidden in expander)
# ---------------------------------------------------------------------------

_INTERNAL_COLS = {"_treatment_binary"}
# Also exclude auto-binarized columns from user-visible list
_auto_cols = {c for c in df.columns if c.startswith("_auto_")}
columns = [c for c in df.columns if c not in _INTERNAL_COLS and c not in _auto_cols]

detections = detect_columns(df[columns])
none_option = ["(none)"]

_source = st.session_state.get("_data_source", "")
_sample_name = _source.split(":", 1)[1] if _source and _source.startswith("sample:") else None
_sample_overrides = SAMPLE_COLUMN_DEFAULTS.get(_sample_name, {}) if _sample_name else {}


def _effective_default(role: str) -> str | None:
    if role in _sample_overrides:
        return _sample_overrides[role]
    det = detections.get(role)
    if det is None:
        return None
    if isinstance(det, list):
        return [s.column_name for s in det]
    return det.column_name


def _default_index(col_name: str | None, options: list[str]) -> int:
    if col_name and col_name in options:
        return options.index(col_name)
    return 0


with st.sidebar.expander("Column Assignment", expanded=False):
    cid_options = none_option + columns
    cid_default_name = _effective_default("customer_id")
    customer_id_col = st.selectbox(
        "Customer ID", cid_options,
        index=_default_index(cid_default_name, cid_options),
        help="Unique customer/entity identifier",
        on_change=_reset_results,
    )
    if customer_id_col == "(none)":
        customer_id_col = None

    time_options = none_option + columns
    time_default_name = _effective_default("time_period")
    time_col = st.selectbox(
        "Time Period", time_options,
        index=_default_index(time_default_name, time_options),
        help="Week, month, or date column",
        on_change=_reset_results,
    )
    if time_col == "(none)":
        time_col = None

    treat_b_options = none_option + columns
    treat_b_default_name = _effective_default("treatment_binary")
    treatment_binary_col = st.selectbox(
        "Treatment (binary)", treat_b_options,
        index=_default_index(treat_b_default_name, treat_b_options),
        help="Binary treatment indicator (0/1)",
        on_change=_reset_results,
    )
    if treatment_binary_col == "(none)":
        treatment_binary_col = None

    treat_c_options = none_option + columns
    treat_c_default_name = _effective_default("treatment_continuous")
    treatment_continuous_col = st.selectbox(
        "Treatment (continuous)", treat_c_options,
        index=_default_index(treat_c_default_name, treat_c_options),
        help="Continuous treatment intensity (binarized via threshold)",
        on_change=_reset_treatment_continuous,
    )
    if treatment_continuous_col == "(none)":
        treatment_continuous_col = None

    outcome_options = none_option + columns
    outcome_default_name = _effective_default("outcome")
    outcome_col = st.selectbox(
        "Outcome Variable", outcome_options,
        index=_default_index(outcome_default_name, outcome_options),
        help="Primary outcome (revenue, units, score, etc.)",
        on_change=_reset_results,
    )
    if outcome_col == "(none)":
        outcome_col = None

    zip_options = none_option + columns
    zip_default_name = _effective_default("geographic_id")
    zip_col = st.selectbox(
        "ZIP Code", zip_options,
        index=_default_index(zip_default_name, zip_options),
        help="5-digit US ZIP code for geographic visualization",
        on_change=_reset_results,
    )
    if zip_col == "(none)":
        zip_col = None

    assigned_cols = {
        c for c in [customer_id_col, time_col, treatment_binary_col,
                    treatment_continuous_col, outcome_col, zip_col]
        if c is not None
    }
    available_covariates = [c for c in columns if c not in assigned_cols]

    _cov_override = _sample_overrides.get("covariates")
    if _cov_override is not None:
        cov_default_names = [c for c in _cov_override if c in available_covariates]
    else:
        cov_defaults_detected = detections.get("covariates", [])
        if isinstance(cov_defaults_detected, list):
            cov_default_names = [
                s.column_name for s in cov_defaults_detected
                if s.column_name in available_covariates
            ]
        else:
            cov_default_names = []

    covariate_cols = st.multiselect(
        "Covariates", available_covariates,
        default=cov_default_names,
        help="Covariates for propensity model and balance diagnostics",
        on_change=_reset_results,
    )

column_roles: dict[str, str | list[str] | None] = {
    "customer_id": customer_id_col,
    "time_period": time_col,
    "treatment_binary": treatment_binary_col,
    "treatment_continuous": treatment_continuous_col,
    "outcome": outcome_col,
    "geographic_id": zip_col,
    "covariates": covariate_cols,
}
st.session_state.column_roles = column_roles

active_treatment_col = treatment_binary_col or treatment_continuous_col
analysis_treatment_col = _get_analysis_treatment_col(
    treatment_binary_col, treatment_continuous_col, df,
)

# ---------------------------------------------------------------------------
# Auto-analysis: load precomputed cache or compute on the fly
# ---------------------------------------------------------------------------

# Map of sample dataset filenames to their precomputed JSON caches
_PRECOMPUTED_CACHE: dict[str, str] = {
    "synthetic_omnichannel.csv": "precomputed_omnichannel.json",
    "synthetic_campaigns.csv": "precomputed_campaigns.json",
}


def _load_precomputed(cache_path: Path, df: pd.DataFrame) -> tuple[list[AutoResult], list[str]]:
    """Load pre-computed auto-analysis results from a JSON file.

    Also recreates the auto-binarized columns in *df* so the subnational
    map can look up treatment assignment per row.
    """
    with open(cache_path) as f:
        cache = json.load(f)

    covariates: list[str] = cache["covariates"]
    results: list[AutoResult] = []

    for r in cache["results"]:
        treat_col = r["treatment_col"]
        label = r["treatment_label"]
        source = r["source"]

        # Recreate binarized column in df if needed
        if source == "binarized" and treat_col not in df.columns:
            # Extract original col name and threshold from label
            # Label format: "col_name (>= threshold)"
            orig_col = label.split(" (>= ")[0]
            threshold_str = label.split("(>= ")[1].rstrip(")")
            if orig_col in df.columns:
                df[treat_col] = (df[orig_col] >= float(threshold_str)).astype(int)

        ar = AutoResult(
            treatment_col=treat_col,
            treatment_label=label,
            source=source,
            overall_lift=r.get("overall_lift", 0.0),
            parallel_trends_ok=r.get("parallel_trends_ok", False),
            balance_ok=r.get("balance_ok", False),
            overlap_ok=r.get("overlap_ok", False),
            sample_size_ok=r.get("sample_size_ok", False),
            propensity_auc=r.get("propensity_auc", 0.5),
        )

        # Reconstruct DidResult
        if "did_att" in r:
            ar.did_result = DidResult(
                att=r["did_att"],
                se=r["did_se"],
                ci_lower=r["did_ci_lower"],
                ci_upper=r["did_ci_upper"],
                p_value=r["did_p_value"],
                n_treated=r["did_n_treated"],
                n_control=r["did_n_control"],
                model_summary="(precomputed — run manual analysis for full summary)",
                method=r.get("did_method", "standard_did"),
            )

        # Reconstruct PsmResult (without matched_indices — not needed for display)
        if "psm_att" in r:
            ar.psm_result = PsmResult(
                att=r["psm_att"],
                se=r["psm_se"],
                ci_lower=r["psm_ci_lower"],
                ci_upper=r["psm_ci_upper"],
                p_value=r["psm_p_value"],
                n_matched_treated=r.get("psm_n_matched_treated", 0),
                n_matched_control=r.get("psm_n_matched_control", 0),
                n_unmatched=r.get("psm_n_unmatched", 0),
                matched_indices=pd.DataFrame(),  # placeholder
            )

        results.append(ar)

    # Sort by absolute lift descending (same order as live computation)
    results.sort(key=lambda x: abs(x.overall_lift), reverse=True)
    return results, covariates


if outcome_col and st.session_state.auto_results is None:
    # Check for precomputed cache first (sample data only)
    _loaded_from_cache = False
    if _sample_name and _sample_name in _PRECOMPUTED_CACHE:
        cache_file = SAMPLE_DIR / _PRECOMPUTED_CACHE[_sample_name]
        if cache_file.exists():
            try:
                auto_results_cached, auto_covariates_cached = _load_precomputed(
                    cache_file, df,
                )
                st.session_state.auto_results = auto_results_cached
                st.session_state.auto_covariates = auto_covariates_cached
                st.session_state.df = df  # persist binarized columns
                _loaded_from_cache = True
            except Exception:
                pass  # fall through to live computation

    if not _loaded_from_cache:
        # Determine structural columns to exclude
        structural_cols = {
            c for c in [customer_id_col, time_col, outcome_col, zip_col]
            if c is not None
        }

        # Select covariates by variance
        auto_covariates = _select_covariates_by_variance(
            df, exclude_cols=structural_cols, max_covariates=10,
        )
        st.session_state.auto_covariates = auto_covariates

        # Detect treatment candidates
        treatment_candidates = _detect_treatment_candidates(
            df, exclude_cols=structural_cols | {outcome_col},
        )

        # Determine post-period start (midpoint of time periods)
        post_start = None
        if time_col and time_col in df.columns:
            time_values = sorted(df[time_col].dropna().unique())
            if len(time_values) >= 2:
                mid_idx = len(time_values) // 2
                post_start = time_values[mid_idx]

        # Run auto-analysis
        if treatment_candidates:
            with st.spinner("Computing causal estimates for all tactics..."):
                auto_results = _run_auto_analysis(
                    df=df,
                    treatments=treatment_candidates,
                    outcome_col=outcome_col,
                    time_col=time_col,
                    post_start=post_start,
                    covariates=auto_covariates,
                    entity_col=customer_id_col,
                    zip_col=zip_col,
                )
                st.session_state.auto_results = auto_results
        else:
            st.session_state.auto_results = []


# ---------------------------------------------------------------------------
# Main panel tabs
# ---------------------------------------------------------------------------

tab_exec, tab_manual, tab_balance, tab_map = st.tabs([
    "Executive Summary",
    "Manual Analysis",
    "Balance Diagnostics",
    "Geographic Explorer",
])


# ===== TAB 1: EXECUTIVE SUMMARY =====
with tab_exec:
    st.header("Executive Summary")

    auto_results: list[AutoResult] = st.session_state.auto_results or []
    auto_covariates = st.session_state.auto_covariates or []

    if not auto_results:
        st.info(
            "No treatment candidates detected. Assign an outcome variable "
            "and ensure the data has binary tactic columns (0/1)."
        )
    else:
        # Build summary DataFrame
        summary_rows = []
        for ar in auto_results:
            did_att = ar.did_result.att if ar.did_result else None
            did_p = ar.did_result.p_value if ar.did_result else None
            psm_att = ar.psm_result.att if ar.psm_result else None
            psm_p = ar.psm_result.p_value if ar.psm_result else None

            summary_rows.append({
                "Treatment": ar.treatment_label,
                "Overall Lift": round(ar.overall_lift, 3),
                "DiD ATT": round(did_att, 3) if did_att is not None else None,
                "DiD p-value": round(did_p, 4) if did_p is not None else None,
                "PSM ATT": round(psm_att, 3) if psm_att is not None else None,
                "PSM p-value": round(psm_p, 4) if psm_p is not None else None,
                "Parallel Trends": _assumption_icon(ar.parallel_trends_ok),
                "Balance": _assumption_icon(ar.balance_ok),
                "Overlap": _assumption_icon(ar.overlap_ok),
                "Sample Size": _assumption_icon(ar.sample_size_ok),
                "_treat_col": ar.treatment_col,
            })

        summary_df = pd.DataFrame(summary_rows)

        # Overall metrics
        n_positive = int((summary_df["Overall Lift"] > 0).sum())
        n_negative = int((summary_df["Overall Lift"] < 0).sum())
        n_sig_did = int(
            summary_df["DiD p-value"].dropna().lt(0.05).sum()
        )
        n_sig_psm = int(
            summary_df["PSM p-value"].dropna().lt(0.05).sum()
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Tactics Analysed", len(summary_df))
        mc2.metric("Positive Lift", n_positive)
        mc3.metric("Sig. DiD (p<0.05)", n_sig_did)
        mc4.metric("Sig. PSM (p<0.05)", n_sig_psm)

        st.caption(
            f"Covariates selected by variance: "
            f"**{', '.join(auto_covariates[:5])}**"
            + (f" + {len(auto_covariates) - 5} more" if len(auto_covariates) > 5 else "")
        )

        # --- Top N Positive Lift ---
        st.subheader(f"Top {TOP_N} Positive Lift")
        top_pos = summary_df[summary_df["Overall Lift"] > 0].nlargest(
            TOP_N, "Overall Lift",
        )
        display_cols = [
            "Treatment", "Overall Lift", "DiD ATT", "DiD p-value",
            "PSM ATT", "PSM p-value",
            "Parallel Trends", "Balance", "Overlap", "Sample Size",
        ]
        if len(top_pos) > 0:
            st.dataframe(
                top_pos[display_cols].style.map(
                    lambda v: "background-color: #d4edda" if v == "Pass" else (
                        "background-color: #f8d7da" if v == "Fail" else ""
                    ),
                    subset=["Parallel Trends", "Balance", "Overlap", "Sample Size"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No tactics with positive lift detected.")

        # --- Top N Negative Lift ---
        st.subheader(f"Top {TOP_N} Negative Lift")
        top_neg = summary_df[summary_df["Overall Lift"] < 0].nsmallest(
            TOP_N, "Overall Lift",
        )
        if len(top_neg) > 0:
            st.dataframe(
                top_neg[display_cols].style.map(
                    lambda v: "background-color: #d4edda" if v == "Pass" else (
                        "background-color: #f8d7da" if v == "Fail" else ""
                    ),
                    subset=["Parallel Trends", "Balance", "Overlap", "Sample Size"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No tactics with negative lift detected.")

        # --- Full table (collapsed) ---
        with st.expander("All Tactics"):
            st.dataframe(
                summary_df[display_cols].style.map(
                    lambda v: "background-color: #d4edda" if v == "Pass" else (
                        "background-color: #f8d7da" if v == "Fail" else ""
                    ),
                    subset=["Parallel Trends", "Balance", "Overlap", "Sample Size"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        # --- Subnational Map ---
        st.subheader("Subnational Lift Map")

        if zip_col and zip_col in df.columns and outcome_col:
            # Treatment selector from available results
            treatment_options = [
                (ar.treatment_label, ar.treatment_col)
                for ar in auto_results
                if ar.treatment_col in df.columns
            ]
            if treatment_options:
                selected_label = st.selectbox(
                    "Select treatment to map",
                    [label for label, _ in treatment_options],
                    key="exec_map_treatment",
                )
                selected_treat_col = dict(treatment_options)[selected_label]

                # Filter to post-period if time data available
                map_df = df
                if time_col and time_col in df.columns:
                    time_values = sorted(df[time_col].dropna().unique())
                    if len(time_values) >= 2:
                        post_start_map = time_values[len(time_values) // 2]
                        map_df = df[df[time_col] >= post_start_map]

                # Compute per-ZIP lift
                lift_rows: list[dict] = []
                for zip_code, grp in map_df.groupby(zip_col):
                    t_vals = grp[grp[selected_treat_col] == 1][outcome_col]
                    c_vals = grp[grp[selected_treat_col] == 0][outcome_col]
                    if len(t_vals) >= 1 and len(c_vals) >= 1:
                        lift_rows.append({
                            zip_col: zip_code,
                            "lift": round(float(t_vals.mean() - c_vals.mean()), 2),
                            "n_treated": len(t_vals),
                            "n_control": len(c_vals),
                            "n_total": len(grp),
                            "mean_treated": round(float(t_vals.mean()), 2),
                            "mean_control": round(float(c_vals.mean()), 2),
                        })

                if lift_rows:
                    lift_df = pd.DataFrame(lift_rows)
                    max_abs = max(
                        abs(float(lift_df["lift"].min())),
                        abs(float(lift_df["lift"].max())),
                        0.01,
                    )

                    fig_map = zip_outcome_map(
                        lift_df, zip_col, "lift",
                        title=f"Subnational Lift: {selected_label} -> {outcome_col}",
                        color_scale="RdBu_r",
                        size_col="n_total",
                        hover_cols=["n_treated", "n_control",
                                    "mean_treated", "mean_control"],
                    )
                    fig_map.update_traces(
                        marker=dict(cmin=-max_abs, cmax=max_abs),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

                    # Summary metrics for selected treatment
                    lc1, lc2, lc3 = st.columns(3)
                    lc1.metric("Mean ZIP Lift", f"{lift_df['lift'].mean():+.2f}")
                    lc2.metric("ZIPs Mapped", f"{len(lift_df)}")
                    pos_pct = float((lift_df["lift"] > 0).mean())
                    lc3.metric("% Positive", f"{pos_pct:.0%}")
                else:
                    st.warning("No ZIPs with both treated and control for this tactic.")
            else:
                st.info("No treatment candidates to map.")
        else:
            st.info(
                "Assign a **ZIP Code** column in the sidebar (Column Assignment) "
                "to see the subnational map."
            )

        # --- Forest plot: all methods for all treatments ---
        st.subheader("Method Comparison")
        forest_rows = []
        for ar in auto_results:
            if ar.did_result:
                forest_rows.append({
                    "label": f"{ar.treatment_label} (DiD)",
                    "att": ar.did_result.att,
                    "ci_lo": ar.did_result.ci_lower,
                    "ci_hi": ar.did_result.ci_upper,
                })
            if ar.psm_result:
                forest_rows.append({
                    "label": f"{ar.treatment_label} (PSM)",
                    "att": ar.psm_result.att,
                    "ci_lo": ar.psm_result.ci_lower,
                    "ci_hi": ar.psm_result.ci_upper,
                })

        if forest_rows:
            fig_forest, ax_forest = plt.subplots(
                figsize=(9, max(3, len(forest_rows) * 0.45)),
            )
            labels = [r["label"] for r in forest_rows]
            atts = [r["att"] for r in forest_rows]
            ci_lows = [r["ci_lo"] for r in forest_rows]
            ci_highs = [r["ci_hi"] for r in forest_rows]
            y_pos = list(range(len(labels)))

            ax_forest.axvline(0, color="grey", linestyle="-", linewidth=0.5)
            ax_forest.errorbar(
                atts, y_pos,
                xerr=[
                    [a - cl for a, cl in zip(atts, ci_lows)],
                    [ch - a for a, ch in zip(atts, ci_highs)],
                ],
                fmt="o", color="#1f77b4", capsize=4, markersize=7,
            )
            ax_forest.set_yticks(y_pos)
            ax_forest.set_yticklabels(labels, fontsize=8)
            ax_forest.set_xlabel("Estimated ATT (95% CI)")
            ax_forest.set_title("Treatment Effects Across Methods")
            ax_forest.invert_yaxis()
            ax_forest.spines["top"].set_visible(False)
            ax_forest.spines["right"].set_visible(False)
            fig_forest.tight_layout()
            st.pyplot(fig_forest)
            plt.close(fig_forest)


# ===== TAB 2: MANUAL ANALYSIS =====
with tab_manual:
    st.header("Manual Analysis")
    st.caption(
        "Configure column assignments in the sidebar, select methods below, "
        "and click Run to perform a custom analysis."
    )

    # Data overview
    if df is not None:
        with st.expander("Data Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Columns", f"{len(columns)}")
            if active_treatment_col and active_treatment_col in df.columns:
                treat_series = df[active_treatment_col]
                if treat_series.nunique() == 2:
                    n_treated = int(treat_series.sum())
                    col3.metric("Treated", f"{n_treated:,}")
                    col4.metric("Control", f"{len(df) - n_treated:,}")
                else:
                    col3.metric("Unique values", f"{treat_series.nunique()}")
            preview_cols = [c for c in df.columns if c not in _INTERNAL_COLS and not c.startswith("_auto_")]
            st.dataframe(df[preview_cols].head(100), use_container_width=True)

    # Treatment threshold (for continuous treatment)
    if treatment_continuous_col and treatment_continuous_col in df.columns:
        st.subheader("Treatment Threshold")
        treat_series = df[treatment_continuous_col]
        suggestions = suggest_thresholds(treat_series)

        smin, smax = float(treat_series.min()), float(treat_series.max())
        default_val = suggestions[0].value if suggestions else (smin + smax) / 2

        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            sug_cols[i].button(
                f"{sug.label}: {sug.value:.2f}",
                key=f"sug_{i}",
                help=sug.method,
                on_click=lambda v=sug.value: st.session_state.update(threshold_value=v),
            )

        threshold = st.slider(
            "Treatment threshold",
            min_value=smin, max_value=smax,
            value=st.session_state.threshold_value or default_val,
            help="Values >= threshold = treated; < threshold = control",
        )
        st.session_state.threshold_value = threshold

        thresh_stats = compute_threshold_stats(treat_series, threshold)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Treated", f"{thresh_stats.n_treated:,} ({thresh_stats.pct_treated:.1f}%)")
        sc2.metric("Control", f"{thresh_stats.n_control:,} ({thresh_stats.pct_control:.1f}%)")
        sc3.metric("Ratio", thresh_stats.ratio_str)

        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        ax_hist.hist(treat_series.dropna(), bins=50, alpha=0.7, color="#1f77b4", edgecolor="white")
        ax_hist.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
        ax_hist.set_xlabel(treatment_continuous_col)
        ax_hist.set_ylabel("Count")
        ax_hist.legend()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        if st.session_state.get("treatment_binarized"):
            st.success(
                f"Treatment binarized at threshold "
                f"{st.session_state.threshold_value:.2f}."
            )
        if st.button("Apply threshold (binarize treatment)"):
            binary_col = binarize_treatment(treat_series, threshold)
            df["_treatment_binary"] = binary_col.fillna(0).astype(int)
            st.session_state.df = df
            st.session_state.treatment_binarized = True
            _reset_results()
            st.rerun()

    # Method selection
    st.subheader("Method Selection")
    eligibility = check_method_eligibility(df, column_roles)
    st.session_state.method_eligibility = eligibility

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        run_did_method = st.checkbox(
            "Difference-in-Differences", value=True,
            disabled=not eligibility.get("DiD", (True, ""))[0],
            help=eligibility.get("DiD", (True, ""))[1] if not eligibility.get("DiD", (True, ""))[0] else "Standard 2x2 DiD",
        )
    with mcol2:
        run_psm_method = st.checkbox(
            "Propensity Score Matching", value=True,
            help="1:1 nearest-neighbor matching",
        )
    with mcol3:
        run_ipw_method = st.checkbox(
            "Inverse Probability Weighting", value=False,
            help="IPW estimation (ATT or ATE)",
        )

    # DiD parameters
    post_period_start = None
    did_cluster_col = None
    did_run_event_study = False
    did_run_ancova = False

    if run_did_method and time_col:
        with st.expander("DiD Parameters"):
            time_values = sorted(df[time_col].dropna().unique())
            if len(time_values) >= 2:
                mid_idx = len(time_values) // 2
                post_period_start = st.selectbox(
                    "Post-period starts at", time_values, index=mid_idx,
                    help="First time period of the post-treatment window",
                )
            else:
                st.warning("Need at least 2 time periods for DiD.")

            did_cluster_col = st.selectbox(
                "Cluster SEs on",
                none_option + [c for c in [customer_id_col, zip_col] if c],
                help="Cluster standard errors at this level",
            )
            if did_cluster_col == "(none)":
                did_cluster_col = None

            did_run_event_study = st.checkbox("Run event study", value=False)
            did_run_ancova = st.checkbox("Run ANCOVA form", value=False)

    # PSM parameters
    psm_propensity_method = "logistic"
    psm_matching_method = "nearest_1to1"
    psm_k = 3
    psm_caliper = None
    psm_replacement = False
    psm_n_bootstrap = 500

    if run_psm_method:
        with st.expander("PSM Parameters"):
            psm_propensity_method = st.selectbox(
                "Propensity model", ["logistic", "gbm"],
                help="Logistic regression (default) or Gradient Boosting",
            )
            psm_matching_method = st.selectbox(
                "Matching method",
                ["nearest_1to1", "nearest_1tok", "caliper"],
            )
            if psm_matching_method == "nearest_1tok":
                psm_k = st.slider("k neighbors", 1, 10, 3)
            if psm_matching_method == "caliper":
                psm_caliper = st.slider(
                    "Caliper (SD of logit PS)", 0.05, 1.0, 0.2, step=0.05,
                )
            psm_replacement = st.checkbox("Match with replacement", value=False)
            psm_n_bootstrap = st.number_input(
                "Bootstrap replicates", 100, 2000, 500, step=100,
            )

    # IPW parameters
    ipw_estimand = "ATT"
    ipw_stabilized = True
    ipw_trim = 0.01
    ipw_propensity_method = "logistic"
    ipw_n_bootstrap = 500

    if run_ipw_method:
        with st.expander("IPW Parameters"):
            ipw_estimand = st.selectbox("Estimand", ["ATT", "ATE"])
            ipw_stabilized = st.checkbox("Stabilized weights", value=True)
            ipw_trim = st.slider(
                "Trim percentile", 0.0, 0.10, 0.01, step=0.005,
            )
            ipw_propensity_method = st.selectbox(
                "Propensity model (IPW)", ["logistic", "gbm"],
                key="ipw_prop_method",
            )
            ipw_n_bootstrap = st.number_input(
                "Bootstrap replicates (IPW)", 100, 2000, 500, step=100,
            )

    # Run button
    can_run = analysis_treatment_col is not None and outcome_col is not None

    if (
        treatment_continuous_col
        and not treatment_binary_col
        and analysis_treatment_col is None
    ):
        st.warning(
            "Continuous treatment selected. Apply a threshold above "
            "to binarize before running analysis."
        )

    run_clicked = st.button(
        "Run Analysis", type="primary",
        disabled=not can_run, use_container_width=True,
    )

    # --- Execute analysis ---
    if run_clicked and can_run:
        st.session_state.analysis_run = True
        _reset_results()
        st.session_state.analysis_run = True
        active_treat = analysis_treatment_col

        if active_treat in df.columns and hasattr(df[active_treat].dtype, "na_value"):
            df[active_treat] = df[active_treat].fillna(0).astype(int)
            st.session_state.df = df

        # Raw balance
        if covariate_cols and active_treat:
            try:
                balance_raw = compute_balance(df, active_treat, covariate_cols)
                st.session_state.balance_raw = balance_raw
            except Exception as e:
                st.error(f"Balance computation failed: {e}")

        # DiD
        if run_did_method and time_col and post_period_start is not None:
            try:
                with st.spinner("Running DiD..."):
                    did_result = run_did(
                        df=df, outcome_col=outcome_col,
                        treatment_col=active_treat, time_col=time_col,
                        post_period_start=post_period_start,
                        covariate_cols=covariate_cols if covariate_cols else None,
                        entity_col=customer_id_col, cluster_col=did_cluster_col,
                    )
                    st.session_state.did_result = did_result

                    if did_run_event_study:
                        df_es = df.copy()
                        time_vals_es = sorted(df_es[time_col].unique())
                        post_idx = time_vals_es.index(post_period_start) if post_period_start in time_vals_es else len(time_vals_es) // 2
                        time_to_event = {t: i - post_idx for i, t in enumerate(time_vals_es)}
                        df_es["_event_time"] = df_es[time_col].map(time_to_event)
                        es_result = run_event_study(
                            df=df_es, outcome_col=outcome_col,
                            treatment_col=active_treat, time_col=time_col,
                            event_time_col="_event_time",
                            covariate_cols=covariate_cols if covariate_cols else None,
                        )
                        st.session_state.event_study_result = es_result

                    if did_run_ancova:
                        ancova_result = run_ancova(
                            df=df, outcome_col=outcome_col,
                            treatment_col=active_treat, time_col=time_col,
                            post_period_start=post_period_start,
                            covariate_cols=covariate_cols if covariate_cols else None,
                            entity_col=customer_id_col,
                        )
                        st.session_state.ancova_result = ancova_result
            except Exception as e:
                st.error(f"DiD failed: {e}")

        # PSM
        if run_psm_method and covariate_cols:
            try:
                with st.spinner("Fitting propensity model..."):
                    prop_result = fit_propensity(
                        df=df, treatment_col=active_treat,
                        covariate_cols=covariate_cols,
                        method=psm_propensity_method,
                    )
                    st.session_state.propensity_result = prop_result

                with st.spinner("Running PSM..."):
                    psm_result = run_psm(
                        df=df, outcome_col=outcome_col,
                        treatment_col=active_treat,
                        propensity_scores=prop_result.scores,
                        method=psm_matching_method,
                        caliper=psm_caliper, k_neighbors=psm_k,
                        with_replacement=psm_replacement,
                        n_bootstrap=psm_n_bootstrap,
                    )
                    st.session_state.psm_result = psm_result

                    if covariate_cols:
                        mi = psm_result.matched_indices
                        balance_adj = compute_balance(
                            df, active_treat, covariate_cols,
                            matched_indices=(
                                mi["treated_idx"].values,
                                mi["control_idx"].values,
                            ),
                        )
                        st.session_state.balance_adjusted = balance_adj
            except Exception as e:
                st.error(f"PSM failed: {e}")

        # IPW
        if run_ipw_method and covariate_cols:
            try:
                with st.spinner("Running IPW..."):
                    if st.session_state.propensity_result is None:
                        prop_result = fit_propensity(
                            df=df, treatment_col=active_treat,
                            covariate_cols=covariate_cols,
                            method=ipw_propensity_method,
                        )
                        st.session_state.propensity_result = prop_result

                    prop_scores = st.session_state.propensity_result.scores
                    ipw_result = run_ipw(
                        df=df, outcome_col=outcome_col,
                        treatment_col=active_treat,
                        propensity_scores=prop_scores,
                        estimand=ipw_estimand, stabilized=ipw_stabilized,
                        trim_percentile=ipw_trim if ipw_trim > 0 else None,
                        n_bootstrap=ipw_n_bootstrap,
                    )
                    st.session_state.ipw_result = ipw_result

                    if covariate_cols:
                        ipw_weights, _ = compute_ipw_weights(
                            prop_scores,
                            df[active_treat].values.astype(int),
                            estimand=ipw_estimand, stabilized=ipw_stabilized,
                            trim_percentile=ipw_trim if ipw_trim > 0 else None,
                        )
                        balance_ipw = compute_balance(
                            df, active_treat, covariate_cols, weights=ipw_weights,
                        )
                        st.session_state.balance_ipw = balance_ipw
            except Exception as e:
                st.error(f"IPW failed: {e}")

    # --- Display manual results ---
    if st.session_state.analysis_run:
        st.divider()
        st.subheader("Results")
        results_collected: list[tuple[str, float, float, float, float]] = []

        did_res: DidResult | None = st.session_state.did_result
        if did_res is not None:
            st.markdown("**Difference-in-Differences**")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("ATT", f"{did_res.att:.3f}")
            rc2.metric("SE", f"{did_res.se:.3f}")
            rc3.metric("95% CI", f"[{did_res.ci_lower:.3f}, {did_res.ci_upper:.3f}]")
            rc4.metric("p-value", f"{did_res.p_value:.4f}")
            results_collected.append(("DiD", did_res.att, did_res.se, did_res.ci_lower, did_res.ci_upper))
            with st.expander("Full Model Summary"):
                st.code(did_res.model_summary, language=None)

        ancova_res = st.session_state.get("ancova_result")
        if ancova_res is not None:
            st.markdown("**ANCOVA (Covariate-Adjusted DiD)**")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("ATT", f"{ancova_res.att:.3f}")
            ac2.metric("SE", f"{ancova_res.se:.3f}")
            ac3.metric("95% CI", f"[{ancova_res.ci_lower:.3f}, {ancova_res.ci_upper:.3f}]")
            ac4.metric("p-value", f"{ancova_res.p_value:.4f}")
            results_collected.append(("ANCOVA", ancova_res.att, ancova_res.se, ancova_res.ci_lower, ancova_res.ci_upper))

        es_res = st.session_state.event_study_result
        if es_res is not None:
            st.markdown("**Event Study**")
            fig_es, ax_es = plt.subplots(figsize=(10, 5))
            ax_es.axhline(0, color="grey", linestyle="-", linewidth=0.5)
            ax_es.axvline(-0.5, color="grey", linestyle=":", alpha=0.5, label="Treatment onset")
            ax_es.errorbar(
                es_res.periods, es_res.coefficients,
                yerr=[
                    [c - cl for c, cl in zip(es_res.coefficients, es_res.ci_lower)],
                    [cu - c for c, cu in zip(es_res.coefficients, es_res.ci_upper)],
                ],
                fmt="o-", color="#1f77b4", capsize=3,
            )
            ax_es.set_xlabel("Periods relative to treatment")
            ax_es.set_ylabel("Estimated effect")
            ax_es.set_title("Event Study: Dynamic Treatment Effects")
            ax_es.legend()
            st.pyplot(fig_es)
            plt.close(fig_es)

        psm_res: PsmResult | None = st.session_state.psm_result
        if psm_res is not None:
            st.markdown("**Propensity Score Matching**")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("ATT", f"{psm_res.att:.3f}")
            pc2.metric("SE", f"{psm_res.se:.3f}")
            pc3.metric("95% CI", f"[{psm_res.ci_lower:.3f}, {psm_res.ci_upper:.3f}]")
            pc4.metric("p-value", f"{psm_res.p_value:.4f}")
            results_collected.append(("PSM", psm_res.att, psm_res.se, psm_res.ci_lower, psm_res.ci_upper))

        ipw_res: IpwResult | None = st.session_state.ipw_result
        if ipw_res is not None:
            st.markdown("**Inverse Probability Weighting**")
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric(ipw_res.estimand, f"{ipw_res.estimate:.3f}")
            ic2.metric("SE", f"{ipw_res.se:.3f}")
            ic3.metric("95% CI", f"[{ipw_res.ci_lower:.3f}, {ipw_res.ci_upper:.3f}]")
            ic4.metric("p-value", f"{ipw_res.p_value:.4f}")
            results_collected.append(("IPW", ipw_res.estimate, ipw_res.se, ipw_res.ci_lower, ipw_res.ci_upper))

        if len(results_collected) >= 1:
            st.markdown("**Method Comparison**")
            fig_forest, ax_forest = plt.subplots(figsize=(8, max(3, len(results_collected) * 0.8)))
            methods = [r[0] for r in results_collected]
            atts = [r[1] for r in results_collected]
            ci_lows = [r[3] for r in results_collected]
            ci_highs = [r[4] for r in results_collected]
            y_pos = list(range(len(methods)))

            ax_forest.axvline(0, color="grey", linestyle="-", linewidth=0.5)
            ax_forest.errorbar(
                atts, y_pos,
                xerr=[
                    [a - cl for a, cl in zip(atts, ci_lows)],
                    [ch - a for a, ch in zip(atts, ci_highs)],
                ],
                fmt="o", color="#1f77b4", capsize=5, markersize=8,
            )
            ax_forest.set_yticks(y_pos)
            ax_forest.set_yticklabels(methods)
            ax_forest.set_xlabel("Estimated ATT (95% CI)")
            ax_forest.set_title("Treatment Effect Estimates Across Methods")
            ax_forest.invert_yaxis()
            fig_forest.tight_layout()
            st.pyplot(fig_forest)
            plt.close(fig_forest)

            summary_data = []
            for name, att, se, ci_lo, ci_hi in results_collected:
                summary_data.append({
                    "Method": name,
                    "Estimate": f"{att:.3f}",
                    "SE": f"{se:.3f}",
                    "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                })
            st.table(pd.DataFrame(summary_data))
    else:
        st.info("Click **Run Analysis** above to generate results.")


# ===== TAB 3: BALANCE DIAGNOSTICS =====
with tab_balance:
    st.header("Covariate Balance Diagnostics")

    if st.session_state.balance_raw is not None:
        bal_raw = st.session_state.balance_raw
        bal_adj = st.session_state.balance_adjusted
        bal_ipw = st.session_state.balance_ipw

        st.subheader("Standardized Mean Differences")
        display_df = bal_raw.table.copy()

        def _status_color(status: str) -> str:
            return {
                "Pass": "background-color: #d4edda",
                "Caution": "background-color: #fff3cd",
                "Fail": "background-color: #f8d7da",
            }.get(status, "")

        if bal_adj is not None:
            adj_map = dict(zip(bal_adj.table["covariate"], bal_adj.table["smd_adjusted"]))
            adj_status_map = dict(zip(bal_adj.table["covariate"], bal_adj.table["status"]))
            display_df["smd_psm"] = display_df["covariate"].map(adj_map)
            display_df["status_psm"] = display_df["covariate"].map(adj_status_map)

        if bal_ipw is not None:
            ipw_map = dict(zip(bal_ipw.table["covariate"], bal_ipw.table["smd_adjusted"]))
            ipw_status_map = dict(zip(bal_ipw.table["covariate"], bal_ipw.table["status"]))
            display_df["smd_ipw"] = display_df["covariate"].map(ipw_map)
            display_df["status_ipw"] = display_df["covariate"].map(ipw_status_map)

        st.dataframe(
            display_df.style.map(_status_color, subset=["status"]),
            use_container_width=True, hide_index=True,
        )

        if bal_adj is not None:
            st.subheader("Love Plot (PSM Adjusted)")
            fig_love = love_plot(bal_raw, bal_adj)
            st.pyplot(fig_love)
            plt.close(fig_love)

        if bal_ipw is not None:
            st.subheader("Love Plot (IPW Weighted)")
            fig_love_ipw = love_plot(bal_raw, bal_ipw, title="Love Plot: IPW-Weighted Balance")
            st.pyplot(fig_love_ipw)
            plt.close(fig_love_ipw)

        prop = st.session_state.propensity_result
        if prop is not None and analysis_treatment_col:
            st.subheader("Propensity Score Overlap")
            treat_mask = df[analysis_treatment_col].astype(bool)
            ps_t = prop.scores[treat_mask.values]
            ps_c = prop.scores[~treat_mask.values]
            fig_overlap = propensity_overlap_plot(ps_t, ps_c)
            st.pyplot(fig_overlap)
            plt.close(fig_overlap)
            st.metric("Propensity Model AUC", f"{prop.auc:.3f}",
                      help="AUC near 0.5 = good overlap; near 1.0 = poor overlap")

        if covariate_cols and analysis_treatment_col:
            st.subheader("Covariate Distributions")
            selected_cov = st.selectbox(
                "Select covariate", covariate_cols, key="cov_dist_select",
            )
            if selected_cov:
                treat_mask_cov = df[analysis_treatment_col].astype(bool)
                s_t = df.loc[treat_mask_cov.values, selected_cov]
                s_c = df.loc[~treat_mask_cov.values, selected_cov]
                is_cat = df[selected_cov].dtype == "object" or df[selected_cov].nunique() < 10
                fig_cov = covariate_distribution_plot(s_t, s_c, selected_cov, is_categorical=is_cat)
                st.pyplot(fig_cov)
                plt.close(fig_cov)
    else:
        st.info("Run a manual analysis (Tab 2) to see balance diagnostics.")


# ===== TAB 4: GEOGRAPHIC EXPLORER =====
with tab_map:
    st.header("Geographic Explorer")

    if zip_col and outcome_col and zip_col in df.columns:
        # Detect binary tactic columns
        _binary_tactic_cols: list[str] = []
        if (
            st.session_state.get("treatment_binarized")
            and "_treatment_binary" in df.columns
            and treatment_continuous_col
        ):
            _binary_tactic_cols.append("_treatment_binary")
        for c in columns:
            if c in {zip_col, outcome_col, time_col, customer_id_col}:
                continue
            vals = set(df[c].dropna().unique())
            if len(vals) == 2 and vals.issubset({0, 1, 0.0, 1.0, True, False}):
                _binary_tactic_cols.append(c)

        _tactic_labels: dict[str, str] = {}
        if "_treatment_binary" in _binary_tactic_cols and treatment_continuous_col:
            threshold_val = st.session_state.get("threshold_value")
            _tactic_labels["_treatment_binary"] = (
                f"{treatment_continuous_col} (binarized >= {threshold_val})"
                if threshold_val is not None
                else f"{treatment_continuous_col} (binarized)"
            )

        map_views = ["Mean Outcome", "Lift by Tactic", "Treatment Geography"]
        map_type = st.radio(
            "Map type", map_views, horizontal=True, label_visibility="collapsed",
        )

        # Post-period filter
        map_df = df
        if time_col and time_col in df.columns:
            time_values_map = sorted(df[time_col].dropna().unique())
            if len(time_values_map) >= 2:
                post_start_map_geo = time_values_map[len(time_values_map) // 2]
                use_post_only = st.checkbox(
                    "Post-period only", value=True,
                    help="Use only post-treatment periods for map calculations",
                )
                if use_post_only:
                    map_df = df[df[time_col] >= post_start_map_geo]
                    st.caption(f"Post-period data: {time_col} >= {post_start_map_geo} ({len(map_df):,} rows)")

        if map_type == "Mean Outcome":
            zip_agg = map_df.groupby(zip_col).agg(
                mean_outcome=(outcome_col, "mean"),
                n_obs=(outcome_col, "count"),
            ).reset_index()

            if analysis_treatment_col and analysis_treatment_col in map_df.columns:
                treat_agg = map_df.groupby(zip_col)[analysis_treatment_col].mean()
                zip_agg = zip_agg.merge(
                    treat_agg.rename("pct_treated").reset_index(), on=zip_col,
                )

            fig_map = zip_outcome_map(
                zip_agg, zip_col, "mean_outcome",
                title=f"Mean {outcome_col} by ZIP Code",
                size_col="n_obs",
                hover_cols=["n_obs"] + (["pct_treated"] if "pct_treated" in zip_agg.columns else []),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        elif map_type == "Lift by Tactic":
            if not _binary_tactic_cols:
                st.info("No binary tactic columns (0/1) found.")
            else:
                tactic_col = st.selectbox(
                    "Select tactic to map", _binary_tactic_cols,
                    format_func=lambda c: _tactic_labels.get(c, c),
                )
                tactic_display = _tactic_labels.get(tactic_col, tactic_col)

                lift_rows: list[dict] = []
                for zip_code, grp in map_df.groupby(zip_col):
                    t_vals = grp[grp[tactic_col] == 1][outcome_col]
                    c_vals = grp[grp[tactic_col] == 0][outcome_col]
                    if len(t_vals) >= 1 and len(c_vals) >= 1:
                        lift_rows.append({
                            zip_col: zip_code,
                            "lift": round(float(t_vals.mean() - c_vals.mean()), 2),
                            "n_treated": len(t_vals),
                            "n_control": len(c_vals),
                            "n_total": len(grp),
                            "mean_treated": round(float(t_vals.mean()), 2),
                            "mean_control": round(float(c_vals.mean()), 2),
                        })

                if not lift_rows:
                    st.warning("No ZIPs with both treated and control observations.")
                else:
                    lift_df = pd.DataFrame(lift_rows)

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    overall_lift = float(lift_df["lift"].mean())
                    mc1.metric("Mean Lift", f"{overall_lift:+.2f}")
                    mc2.metric("Median Lift", f"{lift_df['lift'].median():+.2f}")
                    mc3.metric("ZIPs Mapped", f"{len(lift_df)}")
                    mc4.metric("% Positive Lift", f"{(lift_df['lift'] > 0).mean():.0%}")

                    max_abs = max(abs(float(lift_df["lift"].min())), abs(float(lift_df["lift"].max())), 0.01)
                    fig_map = zip_outcome_map(
                        lift_df, zip_col, "lift",
                        title=f"Lift by ZIP: {tactic_display} -> {outcome_col}",
                        color_scale="RdBu_r", size_col="n_total",
                        hover_cols=["n_treated", "n_control", "mean_treated", "mean_control"],
                    )
                    fig_map.update_traces(marker=dict(cmin=-max_abs, cmax=max_abs))
                    st.plotly_chart(fig_map, use_container_width=True)

                    st.caption(
                        f"Lift = mean({outcome_col} | {tactic_display}=1) - "
                        f"mean({outcome_col} | {tactic_display}=0) within each ZIP. "
                        "This is a **naive** difference-in-means."
                    )

        elif map_type == "Treatment Geography":
            if analysis_treatment_col and analysis_treatment_col in map_df.columns:
                zip_treat = map_df.groupby(zip_col).agg(
                    mean_outcome=(outcome_col, "mean"),
                    pct_treated=(analysis_treatment_col, "mean"),
                ).reset_index()
                zip_treat["_treat_majority"] = (zip_treat["pct_treated"] >= 0.5).astype(int)

                fig_map = zip_treatment_map(
                    zip_treat, zip_col, "_treat_majority", "mean_outcome",
                    title=f"Treatment Geography: Mean {outcome_col}",
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("Assign a binary treatment to see the treatment vs control map.")
    else:
        st.info("Assign a **ZIP Code** column and **Outcome** variable to see the map.")
