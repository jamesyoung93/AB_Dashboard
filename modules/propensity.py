"""Shared propensity score estimation for PSM and IPW.

Fits a propensity model (logistic regression or gradient boosting) to
predict P(treatment=1 | covariates) and returns scored results along
with model metadata for downstream matching and weighting modules.

Both PSM (modules/psm.py) and IPW (modules/ipw.py) import and call
``fit_propensity`` rather than re-implementing score estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PropensityResult:
    """Container for the output of ``fit_propensity``.

    Attributes:
        scores: Array of propensity scores P(T=1|X), one per row of the
            input DataFrame, in the original row order.
        model: The fitted sklearn estimator (LogisticRegression or
            GradientBoostingClassifier).
        method: The method string used ('logistic' or 'gbm').
        auc: Area under the ROC curve for the propensity model on the
            training data.  A rough diagnostic: AUC near 0.5 suggests
            treatment is unpredictable from covariates (good overlap);
            AUC near 1.0 suggests near-perfect separation (poor overlap,
            positivity concerns).
        feature_names: Column names used after dummy-encoding, in the
            order fed to the model.  Useful for inspecting coefficients.
        scaler: Fitted StandardScaler (for 'logistic' only; None for
            'gbm' which is scale-invariant).
    """

    scores: np.ndarray
    model: LogisticRegression | GradientBoostingClassifier
    method: Literal["logistic", "gbm"]
    auc: float
    feature_names: list[str]
    scaler: StandardScaler | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list[str],
    method: Literal["logistic", "gbm"] = "logistic",
    seed: int = 42,
) -> PropensityResult:
    """Estimate propensity scores P(treatment=1 | covariates).

    Handles categorical covariates via ``pd.get_dummies`` with
    ``drop_first=True`` (reference-cell encoding) to avoid perfect
    multicollinearity in the logistic model.  Numeric covariates are
    passed through unchanged (logistic: z-scored; GBM: raw, as tree
    methods are scale-invariant).

    Missing values in covariates are imputed with the column mean
    (numeric) or mode (categorical) before encoding.  A warning is
    raised if any covariate has >5% missing values.

    Args:
        df: Input DataFrame.  Must contain ``treatment_col`` and all
            columns in ``covariate_cols``.  Rows with missing treatment
            values are dropped.
        treatment_col: Name of the binary (0/1) treatment column.
        covariate_cols: List of covariate column names.  May contain
            numeric or categorical columns; categorical columns are
            dummy-encoded automatically.
        method: Propensity model to fit.

            - ``'logistic'``: ``sklearn.linear_model.LogisticRegression``
              with ``solver='lbfgs'``, ``max_iter=200``, ``C=1.0``.
              Covariates are z-scored before fitting.
            - ``'gbm'``: ``sklearn.ensemble.GradientBoostingClassifier``
              with ``n_estimators=100``, ``max_depth=3``,
              ``learning_rate=0.1``.  Scale-invariant; no pre-scaling.
        seed: Random seed for reproducible model initialisation.

    Returns:
        A :class:`PropensityResult` dataclass with the fitted model,
        propensity scores in original row order, AUC, feature names,
        and the fitted scaler (or None for GBM).

    Raises:
        ValueError: If ``treatment_col`` is not binary (0/1 or bool).
        ValueError: If ``covariate_cols`` is empty.
        ValueError: If ``method`` is not one of 'logistic' or 'gbm'.
        KeyError: If any column in ``covariate_cols`` or
            ``treatment_col`` is not present in ``df``.
    """
    if not covariate_cols:
        raise ValueError("covariate_cols must contain at least one column.")
    if method not in ("logistic", "gbm"):
        raise ValueError(f"method must be 'logistic' or 'gbm', got '{method}'.")

    # --- Subset and drop rows with missing treatment ---
    cols_needed = [treatment_col, *covariate_cols]
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    work = df[cols_needed].copy()
    n_before = len(work)
    work = work.dropna(subset=[treatment_col])
    if len(work) < n_before:
        import warnings
        warnings.warn(
            f"Dropped {n_before - len(work)} rows with missing treatment values.",
            stacklevel=2,
        )

    # Validate treatment is binary
    treatment_values = set(work[treatment_col].unique())
    allowed = ({0, 1}, {0.0, 1.0}, {True, False}, {0, 1, True, False})
    if not treatment_values.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(
            f"treatment_col '{treatment_col}' must be binary (0/1 or bool). "
            f"Found unique values: {sorted(treatment_values)}"
        )

    y = work[treatment_col].astype(int).to_numpy()

    # --- Impute and encode covariates ---
    X_encoded, feature_names = _encode_covariates(work[covariate_cols])

    # --- Scale (logistic only) ---
    scaler: StandardScaler | None = None
    X_fit = X_encoded
    if method == "logistic":
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_encoded)

    # --- Fit model ---
    if method == "logistic":
        model: LogisticRegression | GradientBoostingClassifier = LogisticRegression(
            solver="lbfgs",
            max_iter=200,
            C=1.0,
            random_state=seed,
        )
    else:  # 'gbm'
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed,
        )

    model.fit(X_fit, y)

    # --- Score ---
    scores = model.predict_proba(X_fit)[:, 1]

    # --- AUC ---
    auc = float(roc_auc_score(y, scores))

    return PropensityResult(
        scores=scores,
        model=model,
        method=method,
        auc=auc,
        feature_names=feature_names,
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_covariates(
    df_cov: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Impute missing values and dummy-encode categorical columns.

    Args:
        df_cov: DataFrame containing only the covariate columns.

    Returns:
        A tuple of (X_array, feature_names) where X_array is a float64
        numpy array of shape (n_rows, n_features) and feature_names is
        the ordered list of column names after encoding.
    """
    import warnings

    df_out = df_cov.copy()

    for col in df_out.columns:
        pct_missing = df_out[col].isna().mean()
        if pct_missing > 0.05:
            warnings.warn(
                f"Covariate '{col}' has {pct_missing:.1%} missing values; "
                "imputing with mean/mode.",
                stacklevel=3,
            )
        if pct_missing > 0.0:
            if df_out[col].dtype.kind in ("O", "b", "U", "S") or isinstance(
                df_out[col].dtype, pd.CategoricalDtype
            ):
                # Categorical: impute with mode
                mode_val = df_out[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
                df_out[col] = df_out[col].fillna(fill_val)
            else:
                # Numeric: impute with mean
                df_out[col] = df_out[col].fillna(df_out[col].mean())

    # Dummy-encode categorical / object / bool columns
    cat_cols = [
        c
        for c in df_out.columns
        if df_out[c].dtype.kind in ("O", "U", "S")
        or isinstance(df_out[c].dtype, pd.CategoricalDtype)
    ]

    if cat_cols:
        df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=True, dtype=float)

    # Ensure all remaining columns are numeric
    df_out = df_out.astype(float)

    return df_out.to_numpy(dtype=np.float64), list(df_out.columns)
