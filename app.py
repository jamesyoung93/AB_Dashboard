"""ACID-Dash: A/B Test Causal Inference Dashboard.

Main Streamlit application. Phase 1 implements:
- CSV upload with auto-detected column roles
- Treatment threshold slider for continuous treatments
- Difference-in-Differences (standard 2x2, event study, ANCOVA)
- Propensity Score Matching (1:1, 1:k, caliper)
- Inverse Probability Weighting (IPW, stabilized, ATE/ATT)
- Balance diagnostics (SMD table, Love plot, propensity overlap)
- Geographic visualization (Plotly Scattergeo at ZIP centroids)
- Results tab with per-method output and forest plot

Launch:
    streamlit run app.py --server.address 127.0.0.1 --server.headless true
"""

from __future__ import annotations

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
    initial_sidebar_state="expanded",
)

SAMPLE_DIR = Path(__file__).parent / "data" / "sample"

# Per-sample-dataset preferred column assignments (override auto-detection)
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
}

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
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _reset_results() -> None:
    """Clear all analysis results when data or config changes."""
    for key in [
        "propensity_result", "did_result", "psm_result", "ipw_result",
        "event_study_result", "ancova_result", "balance_raw",
        "balance_adjusted", "balance_ipw", "analysis_run",
    ]:
        st.session_state[key] = None if key != "analysis_run" else False


def _reset_treatment_continuous() -> None:
    """Reset binarization state when continuous treatment column changes."""
    st.session_state.treatment_binarized = False
    st.session_state.threshold_value = None
    # Also remove the binarized column from df if it exists
    if st.session_state.df is not None and "_treatment_binary" in st.session_state.df.columns:
        st.session_state.df = st.session_state.df.drop(columns=["_treatment_binary"])
    _reset_results()


def _get_analysis_treatment_col(
    treatment_binary_col: str | None,
    treatment_continuous_col: str | None,
    df: pd.DataFrame,
) -> str | None:
    """Return the column name suitable for analysis (must be binary).

    Returns the binary treatment column directly if set. If only a
    continuous treatment is set, returns the binarized column only after
    the user has applied a threshold. Returns None otherwise.
    """
    if treatment_binary_col:
        return treatment_binary_col
    if treatment_continuous_col:
        if (
            st.session_state.get("treatment_binarized")
            and "_treatment_binary" in df.columns
        ):
            return "_treatment_binary"
    return None


# ---------------------------------------------------------------------------
# Sidebar: Data upload and column assignment
# ---------------------------------------------------------------------------

st.sidebar.title("ACID-Dash")
st.sidebar.caption("A/B Test Causal Inference Dashboard")

st.sidebar.header("1. Upload Data")

upload_mode = st.sidebar.radio(
    "Data source",
    ["Upload CSV", "Use sample data"],
    index=1,
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
        st.sidebar.warning("No sample data found. Run `tests/generate_sample_data.py` first.")

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
# Sidebar: Column role assignment
# ---------------------------------------------------------------------------

st.sidebar.header("2. Column Assignment")

# Columns visible to the user (exclude internal columns added by ACID-Dash)
_INTERNAL_COLS = {"_treatment_binary"}
columns = [c for c in df.columns if c not in _INTERNAL_COLS]

# Run auto-detection on user columns only
detections = detect_columns(df[columns])
none_option = ["(none)"]

# Apply sample-specific overrides if available
_source = st.session_state.get("_data_source", "")
_sample_name = _source.split(":", 1)[1] if _source and _source.startswith("sample:") else None
_sample_overrides = SAMPLE_COLUMN_DEFAULTS.get(_sample_name, {}) if _sample_name else {}


def _effective_default(role: str) -> str | None:
    """Return the effective default column name for a role.

    Uses sample-specific overrides first, then falls back to auto-detection.
    """
    if role in _sample_overrides:
        return _sample_overrides[role]
    det = detections.get(role)
    if det is None:
        return None
    if isinstance(det, list):
        return [s.column_name for s in det]
    return det.column_name


# Helper for building dropdown defaults
def _default_index(col_name: str | None, options: list[str]) -> int:
    if col_name and col_name in options:
        return options.index(col_name)
    return 0


# Customer ID
cid_options = none_option + columns
cid_default_name = _effective_default("customer_id")
customer_id_col = st.sidebar.selectbox(
    "Customer ID",
    cid_options,
    index=_default_index(cid_default_name, cid_options),
    help="Unique customer/entity identifier",
    on_change=_reset_results,
)
if customer_id_col == "(none)":
    customer_id_col = None

# Time period
time_options = none_option + columns
time_default_name = _effective_default("time_period")
time_col = st.sidebar.selectbox(
    "Time Period",
    time_options,
    index=_default_index(time_default_name, time_options),
    help="Week, month, or date column",
    on_change=_reset_results,
)
if time_col == "(none)":
    time_col = None

# Treatment (binary)
treat_b_options = none_option + columns
treat_b_default_name = _effective_default("treatment_binary")
treatment_binary_col = st.sidebar.selectbox(
    "Treatment (binary)",
    treat_b_options,
    index=_default_index(treat_b_default_name, treat_b_options),
    help="Binary treatment indicator (0/1)",
    on_change=_reset_results,
)
if treatment_binary_col == "(none)":
    treatment_binary_col = None

# Treatment (continuous) — for threshold module
treat_c_options = none_option + columns
treat_c_default_name = _effective_default("treatment_continuous")
treatment_continuous_col = st.sidebar.selectbox(
    "Treatment (continuous)",
    treat_c_options,
    index=_default_index(treat_c_default_name, treat_c_options),
    help="Continuous treatment intensity (will be binarized via threshold)",
    on_change=_reset_treatment_continuous,
)
if treatment_continuous_col == "(none)":
    treatment_continuous_col = None

# Outcome
outcome_options = none_option + columns
outcome_default_name = _effective_default("outcome")
outcome_col = st.sidebar.selectbox(
    "Outcome Variable",
    outcome_options,
    index=_default_index(outcome_default_name, outcome_options),
    help="Primary outcome (revenue, units, score, etc.)",
    on_change=_reset_results,
)
if outcome_col == "(none)":
    outcome_col = None

# ZIP code
zip_options = none_option + columns
zip_default_name = _effective_default("geographic_id")
zip_col = st.sidebar.selectbox(
    "ZIP Code",
    zip_options,
    index=_default_index(zip_default_name, zip_options),
    help="5-digit US ZIP code for geographic visualization",
    on_change=_reset_results,
)
if zip_col == "(none)":
    zip_col = None

# Covariates (multi-select)
assigned_cols = {
    c for c in [customer_id_col, time_col, treatment_binary_col,
                treatment_continuous_col, outcome_col, zip_col]
    if c is not None
}
available_covariates = [c for c in columns if c not in assigned_cols]

# Covariate defaults: sample overrides or auto-detected
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

covariate_cols = st.sidebar.multiselect(
    "Covariates",
    available_covariates,
    default=cov_default_names,
    help="Covariates for propensity model and balance diagnostics",
    on_change=_reset_results,
)

# Build column_roles dict
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

# Determine display treatment column (for overview metrics — may be continuous)
active_treatment_col = treatment_binary_col or treatment_continuous_col

# Determine analysis-ready treatment column (must be binary)
analysis_treatment_col = _get_analysis_treatment_col(
    treatment_binary_col, treatment_continuous_col, df,
)

# ---------------------------------------------------------------------------
# Sidebar: Validation
# ---------------------------------------------------------------------------

if active_treatment_col and outcome_col:
    warnings_list = validate_csv(df, column_roles)
    st.session_state.validation_warnings = warnings_list
    if warnings_list:
        with st.sidebar.expander(f"Validation ({len(warnings_list)} warnings)", expanded=False):
            for w in warnings_list:
                icon = {"info": "\u2139\ufe0f", "warning": "\u26a0\ufe0f", "error": "\U0001f534"}.get(w.severity, "")
                st.markdown(f"{icon} **{w.column}**: {w.message}")

# ---------------------------------------------------------------------------
# Sidebar: Method selection and parameters
# ---------------------------------------------------------------------------

st.sidebar.header("3. Analysis")

# Check method eligibility
eligibility = check_method_eligibility(df, column_roles)
st.session_state.method_eligibility = eligibility

# Method checkboxes
run_did_method = st.sidebar.checkbox(
    "Difference-in-Differences",
    value=True,
    disabled=not eligibility.get("DiD", (True, ""))[0],
    help=eligibility.get("DiD", (True, ""))[1] if not eligibility.get("DiD", (True, ""))[0] else "Standard 2x2 DiD with clustered SEs",
)
run_psm_method = st.sidebar.checkbox(
    "Propensity Score Matching",
    value=True,
    help="1:1 nearest-neighbor matching on logit propensity score",
)
run_ipw_method = st.sidebar.checkbox(
    "Inverse Probability Weighting",
    value=False,
    help="IPW estimation using propensity score weights (ATT or ATE)",
)

# DiD parameters
if run_did_method and time_col:
    with st.sidebar.expander("DiD Parameters"):
        time_values = sorted(df[time_col].dropna().unique())
        if len(time_values) >= 2:
            mid_idx = len(time_values) // 2
            post_period_start = st.selectbox(
                "Post-period starts at",
                time_values,
                index=mid_idx,
                help="First time period of the post-treatment window",
            )
        else:
            post_period_start = None
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
else:
    post_period_start = None
    did_cluster_col = None
    did_run_event_study = False
    did_run_ancova = False

# PSM parameters
if run_psm_method:
    with st.sidebar.expander("PSM Parameters"):
        psm_propensity_method = st.selectbox(
            "Propensity model",
            ["logistic", "gbm"],
            help="Logistic regression (default) or Gradient Boosting",
        )
        psm_matching_method = st.selectbox(
            "Matching method",
            ["nearest_1to1", "nearest_1tok", "caliper"],
            help="1:1 nearest neighbor, 1:k, or caliper matching",
        )
        psm_k = 3
        if psm_matching_method == "nearest_1tok":
            psm_k = st.slider("k neighbors", 1, 10, 3)
        psm_caliper = None
        if psm_matching_method == "caliper":
            psm_caliper = st.slider(
                "Caliper (SD of logit PS)", 0.05, 1.0, 0.2, step=0.05,
            )
        psm_replacement = st.checkbox("Match with replacement", value=False)
        psm_n_bootstrap = st.number_input(
            "Bootstrap replicates", 100, 2000, 500, step=100,
        )
else:
    psm_propensity_method = "logistic"
    psm_matching_method = "nearest_1to1"
    psm_k = 3
    psm_caliper = None
    psm_replacement = False
    psm_n_bootstrap = 500

# IPW parameters
if run_ipw_method:
    with st.sidebar.expander("IPW Parameters"):
        ipw_estimand = st.selectbox(
            "Estimand",
            ["ATT", "ATE"],
            help="ATT = effect on treated; ATE = average effect on population",
        )
        ipw_stabilized = st.checkbox("Stabilized weights", value=True,
                                     help="Reduces variance without introducing bias")
        ipw_trim = st.slider(
            "Trim percentile", 0.0, 0.10, 0.01, step=0.005,
            help="Clip propensity scores at this percentile from both tails (0 = no trimming)",
        )
        ipw_propensity_method = st.selectbox(
            "Propensity model (IPW)",
            ["logistic", "gbm"],
            key="ipw_prop_method",
            help="Logistic regression (default) or Gradient Boosting",
        )
        ipw_n_bootstrap = st.number_input(
            "Bootstrap replicates (IPW)", 100, 2000, 500, step=100,
        )
else:
    ipw_estimand = "ATT"
    ipw_stabilized = True
    ipw_trim = 0.01
    ipw_propensity_method = "logistic"
    ipw_n_bootstrap = 500

# Run button — requires binary treatment (directly or via binarization)
can_run = analysis_treatment_col is not None and outcome_col is not None

# Show warning if continuous treatment not yet binarized
if (
    treatment_continuous_col
    and not treatment_binary_col
    and analysis_treatment_col is None
):
    st.sidebar.warning(
        "Continuous treatment selected. Apply a threshold on the "
        "**Overview** tab to binarize before running analysis."
    )

run_clicked = st.sidebar.button(
    "Run Analysis",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Main panel tabs
# ---------------------------------------------------------------------------

tab_overview, tab_balance, tab_results, tab_map = st.tabs(
    ["Overview", "Balance", "Results", "Map"],
)


# ===== OVERVIEW TAB =====
with tab_overview:
    st.header("Data Overview")

    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{len(df.columns)}")
        if active_treatment_col and active_treatment_col in df.columns:
            treat_series = df[active_treatment_col]
            if treat_series.nunique() == 2:
                n_treated = int(treat_series.sum())
                col3.metric("Treated", f"{n_treated:,}")
                col4.metric("Control", f"{len(df) - n_treated:,}")
            else:
                col3.metric("Unique values", f"{treat_series.nunique()}")
        if time_col:
            st.caption(f"Time periods: {df[time_col].nunique()} unique values in `{time_col}`")

        # Data preview (exclude internal columns)
        with st.expander("Data Preview (first 100 rows)"):
            preview_cols = [c for c in df.columns if c not in _INTERNAL_COLS]
            st.dataframe(df[preview_cols].head(100), use_container_width=True)

        # Treatment threshold module
        if treatment_continuous_col and treatment_continuous_col in df.columns:
            st.subheader("Treatment Threshold")
            treat_series = df[treatment_continuous_col]
            suggestions = suggest_thresholds(treat_series)

            # Threshold slider
            smin, smax = float(treat_series.min()), float(treat_series.max())
            default_val = suggestions[0].value if suggestions else (smin + smax) / 2

            # Show suggestions
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
                min_value=smin,
                max_value=smax,
                value=st.session_state.threshold_value or default_val,
                help="Values >= threshold = treated; < threshold = control",
            )
            st.session_state.threshold_value = threshold

            # Show threshold stats
            thresh_stats = compute_threshold_stats(treat_series, threshold)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Treated", f"{thresh_stats.n_treated:,} ({thresh_stats.pct_treated:.1f}%)")
            sc2.metric("Control", f"{thresh_stats.n_control:,} ({thresh_stats.pct_control:.1f}%)")
            sc3.metric("Ratio", thresh_stats.ratio_str)

            # Histogram with threshold line
            fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
            ax_hist.hist(treat_series.dropna(), bins=50, alpha=0.7, color="#1f77b4", edgecolor="white")
            ax_hist.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
            ax_hist.set_xlabel(treatment_continuous_col)
            ax_hist.set_ylabel("Count")
            ax_hist.legend()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            # Binarize button
            if st.session_state.get("treatment_binarized"):
                st.success(
                    f"Treatment binarized at threshold "
                    f"{st.session_state.threshold_value:.2f}. "
                    "Adjust the slider and re-apply to change."
                )
            if st.button("Apply threshold (binarize treatment)"):
                binary_col = binarize_treatment(treat_series, threshold)
                # Convert nullable Int8 to regular int64 for downstream
                # compatibility with numpy/sklearn (fillna(0) treats
                # missing continuous values as control)
                df["_treatment_binary"] = binary_col.fillna(0).astype(int)
                st.session_state.df = df
                st.session_state.treatment_binarized = True
                _reset_results()
                st.rerun()

        # Parallel trends preview (if DiD is possible with a binary treatment)
        # Use analysis_treatment_col (only set when binary) for parallel trends
        pt_treat_col = analysis_treatment_col or (
            active_treatment_col if active_treatment_col and active_treatment_col in df.columns
            and df[active_treatment_col].nunique() == 2 else None
        )
        if pt_treat_col and time_col and outcome_col and pt_treat_col in df.columns:
            treat_s = df[pt_treat_col]
            if treat_s.nunique() == 2:
                st.subheader("Parallel Trends Preview")
                pt_data = parallel_trends_data(df, outcome_col, pt_treat_col, time_col)
                fig_pt, ax_pt = plt.subplots(figsize=(10, 4))
                for group_name, group_df in pt_data.groupby("group"):
                    gd = group_df.sort_values("time")
                    color = "#1f77b4" if group_name == "treated" else "#ff7f0e"
                    ax_pt.plot(gd["time"], gd["mean"], marker="o", label=group_name, color=color)
                    ax_pt.fill_between(gd["time"], gd["ci_lower"], gd["ci_upper"], alpha=0.15, color=color)
                if post_period_start is not None:
                    ax_pt.axvline(post_period_start, color="grey", linestyle=":", label="Post-period start")
                ax_pt.set_xlabel(time_col)
                ax_pt.set_ylabel(outcome_col)
                ax_pt.set_title("Mean Outcome by Period (Treated vs Control)")
                ax_pt.legend()
                st.pyplot(fig_pt)
                plt.close(fig_pt)


# ===== RUN ANALYSIS =====
if run_clicked and can_run:
    st.session_state.analysis_run = True
    _reset_results()
    st.session_state.analysis_run = True

    # Use the validated binary treatment column
    active_treat = analysis_treatment_col

    # Ensure the treatment column is a clean int64 (guards against nullable
    # Int8 from binarization or other non-standard dtypes)
    if active_treat in df.columns and hasattr(df[active_treat].dtype, "na_value"):
        df[active_treat] = df[active_treat].fillna(0).astype(int)
        st.session_state.df = df

    # Compute raw balance
    if covariate_cols and active_treat:
        try:
            balance_raw = compute_balance(df, active_treat, covariate_cols)
            st.session_state.balance_raw = balance_raw
        except Exception as e:
            st.error(f"Balance computation failed: {e}")

    # --- DiD ---
    if run_did_method and time_col and post_period_start is not None:
        try:
            with st.spinner("Running Difference-in-Differences..."):
                did_result = run_did(
                    df=df,
                    outcome_col=outcome_col,
                    treatment_col=active_treat,
                    time_col=time_col,
                    post_period_start=post_period_start,
                    covariate_cols=covariate_cols if covariate_cols else None,
                    entity_col=customer_id_col,
                    cluster_col=did_cluster_col,
                )
                st.session_state.did_result = did_result

                if did_run_event_study:
                    # Build event time column
                    df_es = df.copy()
                    time_vals = sorted(df_es[time_col].unique())
                    post_idx = time_vals.index(post_period_start) if post_period_start in time_vals else len(time_vals) // 2
                    time_to_event = {t: i - post_idx for i, t in enumerate(time_vals)}
                    df_es["_event_time"] = df_es[time_col].map(time_to_event)
                    es_result = run_event_study(
                        df=df_es,
                        outcome_col=outcome_col,
                        treatment_col=active_treat,
                        time_col=time_col,
                        event_time_col="_event_time",
                        covariate_cols=covariate_cols if covariate_cols else None,
                    )
                    st.session_state.event_study_result = es_result

                if did_run_ancova:
                    ancova_result = run_ancova(
                        df=df,
                        outcome_col=outcome_col,
                        treatment_col=active_treat,
                        time_col=time_col,
                        post_period_start=post_period_start,
                        covariate_cols=covariate_cols if covariate_cols else None,
                        entity_col=customer_id_col,
                    )
                    st.session_state.ancova_result = ancova_result

        except Exception as e:
            st.error(f"DiD failed: {e}")

    # --- PSM ---
    if run_psm_method and covariate_cols:
        try:
            with st.spinner("Fitting propensity model..."):
                prop_result = fit_propensity(
                    df=df,
                    treatment_col=active_treat,
                    covariate_cols=covariate_cols,
                    method=psm_propensity_method,
                )
                st.session_state.propensity_result = prop_result

            with st.spinner("Running Propensity Score Matching..."):
                psm_result = run_psm(
                    df=df,
                    outcome_col=outcome_col,
                    treatment_col=active_treat,
                    propensity_scores=prop_result.scores,
                    method=psm_matching_method,
                    caliper=psm_caliper,
                    k_neighbors=psm_k,
                    with_replacement=psm_replacement,
                    n_bootstrap=psm_n_bootstrap,
                )
                st.session_state.psm_result = psm_result

                # Post-match balance
                if covariate_cols:
                    mi = psm_result.matched_indices
                    matched_idx = (
                        mi["treated_idx"].values,
                        mi["control_idx"].values,
                    )
                    balance_adj = compute_balance(
                        df, active_treat, covariate_cols,
                        matched_indices=matched_idx,
                    )
                    st.session_state.balance_adjusted = balance_adj

        except Exception as e:
            st.error(f"PSM failed: {e}")

    # --- IPW ---
    if run_ipw_method and covariate_cols:
        try:
            with st.spinner("Running Inverse Probability Weighting..."):
                # Fit propensity if not already done by PSM
                if st.session_state.propensity_result is None:
                    prop_result = fit_propensity(
                        df=df,
                        treatment_col=active_treat,
                        covariate_cols=covariate_cols,
                        method=ipw_propensity_method,
                    )
                    st.session_state.propensity_result = prop_result

                prop_scores = st.session_state.propensity_result.scores

                ipw_result = run_ipw(
                    df=df,
                    outcome_col=outcome_col,
                    treatment_col=active_treat,
                    propensity_scores=prop_scores,
                    estimand=ipw_estimand,
                    stabilized=ipw_stabilized,
                    trim_percentile=ipw_trim if ipw_trim > 0 else None,
                    n_bootstrap=ipw_n_bootstrap,
                )
                st.session_state.ipw_result = ipw_result

                # IPW-weighted balance
                if covariate_cols:
                    ipw_weights, _ = compute_ipw_weights(
                        prop_scores,
                        df[active_treat].values.astype(int),
                        estimand=ipw_estimand,
                        stabilized=ipw_stabilized,
                        trim_percentile=ipw_trim if ipw_trim > 0 else None,
                    )
                    balance_ipw = compute_balance(
                        df, active_treat, covariate_cols,
                        weights=ipw_weights,
                    )
                    st.session_state.balance_ipw = balance_ipw

        except Exception as e:
            st.error(f"IPW failed: {e}")


# ===== BALANCE TAB =====
with tab_balance:
    st.header("Covariate Balance Diagnostics")

    if st.session_state.balance_raw is not None:
        bal_raw = st.session_state.balance_raw
        bal_adj = st.session_state.balance_adjusted
        bal_ipw = st.session_state.balance_ipw

        # SMD table
        st.subheader("Standardized Mean Differences")
        display_df = bal_raw.table.copy()

        # Color-code status
        def _status_color(status: str) -> str:
            return {
                "Pass": "background-color: #d4edda",
                "Caution": "background-color: #fff3cd",
                "Fail": "background-color: #f8d7da",
            }.get(status, "")

        if bal_adj is not None:
            # Merge adjusted SMD from PSM
            adj_map = dict(zip(bal_adj.table["covariate"], bal_adj.table["smd_adjusted"]))
            adj_status_map = dict(zip(bal_adj.table["covariate"], bal_adj.table["status"]))
            display_df["smd_psm"] = display_df["covariate"].map(adj_map)
            display_df["status_psm"] = display_df["covariate"].map(adj_status_map)

        if bal_ipw is not None:
            # Merge adjusted SMD from IPW
            ipw_map = dict(zip(bal_ipw.table["covariate"], bal_ipw.table["smd_adjusted"]))
            ipw_status_map = dict(zip(bal_ipw.table["covariate"], bal_ipw.table["status"]))
            display_df["smd_ipw"] = display_df["covariate"].map(ipw_map)
            display_df["status_ipw"] = display_df["covariate"].map(ipw_status_map)

        st.dataframe(
            display_df.style.map(
                _status_color,
                subset=["status"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Love plot (PSM)
        if bal_adj is not None:
            st.subheader("Love Plot (PSM Adjusted)")
            fig_love = love_plot(bal_raw, bal_adj)
            st.pyplot(fig_love)
            plt.close(fig_love)

        # Love plot (IPW)
        if bal_ipw is not None:
            st.subheader("Love Plot (IPW Weighted)")
            fig_love_ipw = love_plot(bal_raw, bal_ipw, title="Love Plot: IPW-Weighted Balance")
            st.pyplot(fig_love_ipw)
            plt.close(fig_love_ipw)

        # Propensity score overlap
        prop = st.session_state.propensity_result
        if prop is not None and analysis_treatment_col:
            st.subheader("Propensity Score Overlap")
            treat_mask = df[analysis_treatment_col].astype(bool)
            ps_t = prop.scores[treat_mask.values]
            ps_c = prop.scores[~treat_mask.values]
            fig_overlap = propensity_overlap_plot(ps_t, ps_c)
            st.pyplot(fig_overlap)
            plt.close(fig_overlap)

            # AUC metric
            st.metric("Propensity Model AUC", f"{prop.auc:.3f}",
                      help="AUC near 0.5 = good overlap; AUC near 1.0 = poor overlap (positivity concern)")

        # Covariate distributions
        if covariate_cols and analysis_treatment_col:
            st.subheader("Covariate Distributions")
            selected_cov = st.selectbox(
                "Select covariate",
                covariate_cols,
                key="cov_dist_select",
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
        st.info("Run analysis to see balance diagnostics.")


# ===== RESULTS TAB =====
with tab_results:
    st.header("Causal Inference Results")

    if not st.session_state.analysis_run:
        st.info("Click 'Run Analysis' in the sidebar to generate results.")
    else:
        results_collected: list[tuple[str, float, float, float, float]] = []

        # DiD results
        did_res: DidResult | None = st.session_state.did_result
        if did_res is not None:
            st.subheader("Difference-in-Differences")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("ATT", f"{did_res.att:.3f}")
            rc2.metric("SE", f"{did_res.se:.3f}")
            rc3.metric("95% CI", f"[{did_res.ci_lower:.3f}, {did_res.ci_upper:.3f}]")
            rc4.metric("p-value", f"{did_res.p_value:.4f}")

            st.caption(f"N treated: {did_res.n_treated:,} | N control: {did_res.n_control:,} | Method: {did_res.method}")

            results_collected.append(("DiD", did_res.att, did_res.se, did_res.ci_lower, did_res.ci_upper))

            with st.expander("Full Model Summary"):
                st.code(did_res.model_summary, language=None)

        # ANCOVA results
        ancova_res = st.session_state.get("ancova_result")
        if ancova_res is not None:
            st.subheader("ANCOVA (Covariate-Adjusted DiD)")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("ATT", f"{ancova_res.att:.3f}")
            ac2.metric("SE", f"{ancova_res.se:.3f}")
            ac3.metric("95% CI", f"[{ancova_res.ci_lower:.3f}, {ancova_res.ci_upper:.3f}]")
            ac4.metric("p-value", f"{ancova_res.p_value:.4f}")
            results_collected.append(("ANCOVA", ancova_res.att, ancova_res.se, ancova_res.ci_lower, ancova_res.ci_upper))

        # Event study plot
        es_res = st.session_state.event_study_result
        if es_res is not None:
            st.subheader("Event Study")
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

        # PSM results
        psm_res: PsmResult | None = st.session_state.psm_result
        if psm_res is not None:
            st.subheader("Propensity Score Matching")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("ATT", f"{psm_res.att:.3f}")
            pc2.metric("SE", f"{psm_res.se:.3f}")
            pc3.metric("95% CI", f"[{psm_res.ci_lower:.3f}, {psm_res.ci_upper:.3f}]")
            pc4.metric("p-value", f"{psm_res.p_value:.4f}")

            st.caption(
                f"Matched: {psm_res.n_matched_treated:,} treated, "
                f"{psm_res.n_matched_control:,} control | "
                f"Unmatched: {psm_res.n_unmatched:,} | "
                f"Method: {psm_matching_method}"
            )

            results_collected.append(("PSM", psm_res.att, psm_res.se, psm_res.ci_lower, psm_res.ci_upper))

        # IPW results
        ipw_res: IpwResult | None = st.session_state.ipw_result
        if ipw_res is not None:
            st.subheader("Inverse Probability Weighting")
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric(ipw_res.estimand, f"{ipw_res.estimate:.3f}")
            ic2.metric("SE", f"{ipw_res.se:.3f}")
            ic3.metric("95% CI", f"[{ipw_res.ci_lower:.3f}, {ipw_res.ci_upper:.3f}]")
            ic4.metric("p-value", f"{ipw_res.p_value:.4f}")

            st.caption(
                f"N treated: {ipw_res.n_treated:,} | "
                f"N control: {ipw_res.n_control:,} | "
                f"Method: {ipw_res.method}"
            )

            with st.expander("Weight Diagnostics"):
                ws = ipw_res.weights_summary
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("ESS (Treated)", f"{ws['ess_treated']:.0f}")
                wc2.metric("ESS (Control)", f"{ws['ess_control']:.0f}")
                wc3.metric("Max Weight", f"{ws['max']:.2f}")
                wc4.metric("% Trimmed", f"{ws['pct_trimmed']:.1f}%")

            results_collected.append(("IPW", ipw_res.estimate, ipw_res.se, ipw_res.ci_lower, ipw_res.ci_upper))

        # Forest plot (comparing methods)
        if len(results_collected) >= 1:
            st.subheader("Method Comparison (Forest Plot)")
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

            # Summary table
            st.subheader("Summary Table")
            summary_data = []
            for name, att, se, ci_lo, ci_hi in results_collected:
                summary_data.append({
                    "Method": name,
                    "Estimate": f"{att:.3f}",
                    "SE": f"{se:.3f}",
                    "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                })
            st.table(pd.DataFrame(summary_data))


# ===== MAP TAB =====
with tab_map:
    st.header("Geographic Visualization")

    if zip_col and outcome_col and zip_col in df.columns:
        # Detect all binary tactic columns (potential treatments to map)
        _binary_tactic_cols: list[str] = []
        # Include binarized continuous treatment if threshold has been applied
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

        # Display labels for the tactic selector (readable name for internal col)
        _tactic_labels: dict[str, str] = {}
        if "_treatment_binary" in _binary_tactic_cols and treatment_continuous_col:
            threshold_val = st.session_state.get("threshold_value")
            _tactic_labels["_treatment_binary"] = (
                f"{treatment_continuous_col} (binarized >= {threshold_val})"
                if threshold_val is not None
                else f"{treatment_continuous_col} (binarized)"
            )

        # Map views
        map_views = ["Mean Outcome", "Lift by Tactic", "Treatment Geography"]
        map_type = st.radio(
            "Map type", map_views, horizontal=True, label_visibility="collapsed",
        )

        # Optional: filter to post-period only (for panel data)
        map_df = df
        if time_col and post_period_start is not None:
            use_post_only = st.checkbox(
                "Post-period only",
                value=True,
                help="Use only post-treatment periods for map calculations",
            )
            if use_post_only:
                map_df = df[df[time_col] >= post_period_start]
                st.caption(
                    f"Showing post-period data only "
                    f"({time_col} >= {post_period_start}, "
                    f"{len(map_df):,} rows)"
                )

        if map_type == "Mean Outcome":
            zip_agg = map_df.groupby(zip_col).agg(
                mean_outcome=(outcome_col, "mean"),
                n_obs=(outcome_col, "count"),
            ).reset_index()

            if analysis_treatment_col and analysis_treatment_col in map_df.columns:
                treat_agg = map_df.groupby(zip_col)[analysis_treatment_col].mean()
                zip_agg = zip_agg.merge(
                    treat_agg.rename("pct_treated").reset_index(),
                    on=zip_col,
                )

            fig_map = zip_outcome_map(
                zip_agg, zip_col, "mean_outcome",
                title=f"Mean {outcome_col} by ZIP Code",
                size_col="n_obs",
                hover_cols=(
                    ["n_obs"]
                    + (["pct_treated"] if "pct_treated" in zip_agg.columns else [])
                ),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        elif map_type == "Lift by Tactic":
            if not _binary_tactic_cols:
                st.info(
                    "No binary tactic columns (0/1) found in the data. "
                    "Assign a binary treatment or binarize a continuous one."
                )
            else:
                tactic_col = st.selectbox(
                    "Select tactic to map",
                    _binary_tactic_cols,
                    format_func=lambda c: _tactic_labels.get(c, c),
                    help=(
                        "Binary tactic column (0/1). Lift = "
                        "mean(outcome | tactic=1) − mean(outcome | tactic=0) "
                        "within each ZIP."
                    ),
                )
                # Display name for titles/captions
                tactic_display = _tactic_labels.get(tactic_col, tactic_col)

                # Compute per-ZIP lift
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
                    st.warning(
                        "No ZIPs have both treated and control observations "
                        "for this tactic."
                    )
                else:
                    lift_df = pd.DataFrame(lift_rows)

                    # Summary metrics
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    overall_lift = float(lift_df["lift"].mean())
                    mc1.metric("Mean Lift", f"{overall_lift:+.2f}")
                    mc2.metric("Median Lift", f"{lift_df['lift'].median():+.2f}")
                    mc3.metric("ZIPs Mapped", f"{len(lift_df)}")
                    pos_pct = float((lift_df["lift"] > 0).mean())
                    mc4.metric("% Positive Lift", f"{pos_pct:.0%}")

                    # Diverging colorscale centered at zero
                    max_abs = max(
                        abs(float(lift_df["lift"].min())),
                        abs(float(lift_df["lift"].max())),
                        0.01,  # avoid zero range
                    )

                    fig_map = zip_outcome_map(
                        lift_df, zip_col, "lift",
                        title=f"Lift by ZIP: {tactic_display} → {outcome_col}",
                        color_scale="RdBu_r",
                        size_col="n_total",
                        hover_cols=[
                            "n_treated", "n_control",
                            "mean_treated", "mean_control",
                        ],
                    )
                    # Center the diverging colorscale at 0
                    fig_map.update_traces(
                        marker=dict(cmin=-max_abs, cmax=max_abs),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

                    st.caption(
                        f"Lift = mean({outcome_col} | {tactic_display}=1) − "
                        f"mean({outcome_col} | {tactic_display}=0) within each ZIP. "
                        "This is a **naive** difference-in-means — useful for "
                        "exploring geographic variation, but the absolute level "
                        "may be biased by confounders. For unbiased causal "
                        "estimates, see the **Results** tab (DiD/PSM/IPW)."
                    )

                    # Lift distribution histogram
                    with st.expander("Lift Distribution"):
                        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
                        ax_hist.hist(
                            lift_df["lift"], bins=30, alpha=0.7,
                            color="#1f77b4", edgecolor="white",
                        )
                        ax_hist.axvline(
                            0, color="grey", linestyle="--", linewidth=1,
                        )
                        ax_hist.axvline(
                            overall_lift, color="red", linestyle="--",
                            linewidth=1.5, label=f"Mean lift = {overall_lift:+.2f}",
                        )
                        ax_hist.set_xlabel(f"Lift ({outcome_col})")
                        ax_hist.set_ylabel("ZIP count")
                        ax_hist.set_title(f"Distribution of {tactic_display} Lift Across ZIPs")
                        ax_hist.legend()
                        st.pyplot(fig_hist)
                        plt.close(fig_hist)

        elif map_type == "Treatment Geography":
            if analysis_treatment_col and analysis_treatment_col in map_df.columns:
                zip_treat = map_df.groupby(zip_col).agg(
                    mean_outcome=(outcome_col, "mean"),
                    pct_treated=(analysis_treatment_col, "mean"),
                ).reset_index()
                zip_treat["_treat_majority"] = (
                    zip_treat["pct_treated"] >= 0.5
                ).astype(int)

                fig_map = zip_treatment_map(
                    zip_treat, zip_col, "_treat_majority", "mean_outcome",
                    title=f"Treatment Geography: Mean {outcome_col}",
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(
                    "Assign a binary treatment column to see the "
                    "treatment vs control map."
                )
    else:
        st.info(
            "Assign a **ZIP Code** column and **Outcome** variable in the sidebar "
            "to see the geographic map."
        )
