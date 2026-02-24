# Review: ACID-Dash -- A/B Test Causal Inference Dashboard

**Reviewer**: Research Reviewer Agent (Opus 4.6)
**Date**: 2026-02-22
**Proposal version reviewed**: 2026-02-22 initial submission
**Overall recommendation**: **Go with Conditions**

---

## Scores

| Dimension | Score (1-5) | Notes |
|-----------|-------------|-------|
| **Novelty** | 2.5 | The individual methods are textbook. The integration into a self-service, offline, pharma-specific Streamlit tool is useful engineering but not novel research. Competing open-source libraries (CausalML, EconML, DoWhy) already provide most of this functionality. The ZIP-level geographic effect heterogeneity layer and the "actionable zones" quadrant are the most differentiated features. |
| **Feasibility** | 4 | Core methods are well-understood with battle-tested Python implementations. Dependency stack is realistic. Laptop-scale computation is verified. The main risk is in integration polish, not algorithmic difficulty. |
| **Impact** | 4 | High practical utility for James's day job. Directly addresses a recurring pain point. Portable to any pharma commercial analytics team. Strong newsletter/portfolio content. |
| **Fit with Portfolio** | 3.5 | Good thematic alignment with causal inference focus (bridges CBRE's theoretical work). Does not compete with any existing project. Risk: time allocation away from CBRE and ECOCHEMICAL, which are research-grade projects with publication potential. |

**Composite**: 3.5 -- a solid, low-risk tool project with high practical utility and no significant research novelty.

---

## Section-by-Section Critique

### Section 1: Project Title and Summary

**What is good**: The summary is crisp and well-targeted. The value proposition is clear: make rigorous causal inference accessible to commercial analytics teams without requiring R/Stata/SAS expertise. The constraint specification (8-16 GB RAM, no GPU, no admin rights, no internet after install) is honest and well-defined.

**Issues**:

1. [MINOR] The title "A/B Test Causal Inference Dashboard" is slightly misleading. A/B tests are randomized experiments; the methods here (DiD, PSM, IPW, RDD, Synthetic Control) are for observational/quasi-experimental data. Calling this an "A/B Test" dashboard implies randomized data, which is the opposite of the use case described. The target audience -- people who "understand A/B test logic" -- may actually be confused by receiving DiD results when they expected a simple two-sample comparison. Consider renaming to something like "Promotional Effectiveness Causal Dashboard" or acknowledge this naming tension in the README.

2. [MINOR] The claim that users need "no R, Stata, or SAS expertise" is correct, but the tool still requires users to understand what DiD, PSM, and parallel trends mean. The proposal oscillates between targeting "commercial strategists who may not have training in causal inference methodology" (Section 1) and producing outputs that require methodological literacy to interpret (event study plots, Rosenbaum bounds, McCrary density tests). The target audience definition needs sharpening: is this for methodologists who lack Python skills, or for business users who lack causal inference training? The UI complexity implies the former, but the summary implies the latter.

### Section 2: Business and Scientific Motivation

**What is good**: The three threats to internal validity (selection bias, confounding, time trends) are correctly identified and concisely explained. The ZIP-code tactic evaluation workflow is concrete and realistic. The laptop constraint is well-specified.

**Issues**:

1. [SUGGESTION] The proposal does not cite or position against existing open-source causal inference libraries that already solve much of this problem. Specifically:
   - **CausalML** (Uber, open-source): provides uplift modeling and CATE estimation with meta-learners, including T-learner, S-learner, X-learner. Directly applicable to promotional tactic evaluation.
   - **EconML** (Microsoft Research, open-source): provides DML, causal forests, instrumental variables, and DiD estimators with a scikit-learn-compatible API.
   - **DoWhy** (Microsoft/PyWhy, open-source): provides a structured framework for causal inference with propensity score methods, IV, and sensitivity analysis.
   - **CausalPy** (PyMC Labs): Bayesian causal inference including DiD, synthetic control, and regression discontinuity.

   A hostile reviewer (or a technically savvy colleague) will immediately ask: "Why not just use EconML or DoWhy?" The answer is likely: (a) ACID-Dash provides a GUI rather than a code-first interface, (b) it bundles geographic visualization specific to pharma tactic evaluation, (c) it includes balance diagnostics and sensitivity analysis in an integrated workflow, and (d) it targets the specific CSV-upload workflow of pharma commercial analytics. These are legitimate differentiators but they must be stated explicitly. Without this positioning, the proposal reads as if the author is unaware of the existing ecosystem.

### Section 3: Input Data Specification

**What is good**: The column role assignment via dropdowns (rather than rigid naming conventions) is the correct design choice. The auto-detection heuristics are reasonable. The validation logic is thorough and covers the major data quality issues.

**Issues**:

1. [MINOR] The auto-detection confidence levels (High/Medium/Low) are qualitative but not actionable. What does "Medium" confidence mean for the treatment column? If the heuristic assigns a column with 60% confidence, is it auto-selected with a yellow warning, or left unassigned? The proposal should specify the behavior at each confidence level: auto-assign above what threshold, suggest-but-do-not-assign in what range, and ignore below what threshold.

2. [MINOR] The NPI auto-detection heuristic ("values are 10-digit integers starting with 1") is correct for NPIs but will also match some other ID schemes. More importantly, NPIs starting with 2 are valid for organizational NPIs. The heuristic should accept 10-digit integers starting with 1 or 2.

3. [SUGGESTION] Missing from validation: a check for sufficient temporal coverage for DiD. If the user selects DiD but the data contains only one time period, the method should be greyed out with an explanation. Similarly, if RDD is selected but no continuous running variable is assigned, it should be disabled. Method-eligibility validation based on column assignments would prevent user confusion.

### Section 4: Treatment Assignment Threshold Module

**What is good**: The draggable threshold with live N counter and mini-SMD table is a genuinely useful UX feature. The auto-suggest with median, mean, and Jenks natural break is smart. This section demonstrates good product thinking.

**Issues**:

1. [MINOR] The Jenks natural breaks algorithm requires `jenkspy` or equivalent, which is not listed in the dependency stack (Section 10). Either add it or implement a simpler alternative (e.g., kernel density trough via `scipy.signal.argrelmin` on a KDE, which is already available through scipy/seaborn).

2. [SUGGESTION] The "balance preview" that refreshes on threshold drag could be computationally expensive if it recomputes propensity scores on every slider move. Consider debouncing (only recompute after the slider has been stationary for 500ms) or displaying only raw SMDs without propensity-model-based metrics during drag, with a full recompute on slider release.

### Section 5: Causal Inference Method Stack

This is the heart of the proposal. I will evaluate each method individually.

#### 5A: Difference-in-Differences (DiD)

**What is good**: The standard 2x2 DiD specification is correct. The event study (dynamic DiD) is properly described with period-relative dummies. Clustered standard errors at the customer/ZIP level are appropriate.

**Issues**:

1. [MAJOR] **The proposal only implements classical two-group, two-period DiD and its dynamic extension.** This is adequate for simple pre/post designs but inadequate for the most common pharma analytics scenario: staggered treatment adoption, where different HCPs begin receiving tactics at different times. In staggered settings, the classical two-way fixed effects (TWFE) DiD estimator is biased due to "negative weighting" of already-treated units as controls for later-treated units. This was demonstrated by Goodman-Bacon (2021), de Chaisemartin and D'Haultfoeuille (2020), and Sun and Abraham (2021), and a practical estimator was provided by Callaway and Sant'Anna (2021).

   A Python implementation of Callaway-Sant'Anna exists (`csdid` package, released 2024, available on PyPI). For a causal inference tool targeting pharma promotional analytics -- where campaigns roll out across territories over weeks -- omitting staggered DiD is a significant gap. Any HEOR-trained reviewer will flag this immediately.

   **Recommendation**: Add Callaway-Sant'Anna as an option within the DiD module, selectable when the user indicates staggered treatment timing. At minimum, add a warning in the UI: "If treatment was adopted at different times for different units, classical DiD may be biased. Consider Callaway-Sant'Anna (2021) for staggered adoption designs."

2. [MINOR] The event study description specifies `Y ~ sum_t(beta_t * Treated * 1[t=tau])` but omits the period fixed effects and unit fixed effects that are standard in event study specifications. The full specification should be `Y_it = alpha_i + gamma_t + sum_{tau != -1}(beta_tau * D_i * 1[t - E_i = tau]) + epsilon_it` where alpha_i are unit FE, gamma_t are time FE, D_i is the treatment indicator, and E_i is the event date. The omission of FE notation is not necessarily wrong (they are implied by the `linearmodels.PanelOLS` reference), but the formula as written is incomplete and could mislead implementation.

3. [MINOR] The parallel trends visualization is described as "mean outcome by period for treated vs. control, with shaded 95% CI bands." These CIs should be specified: CIs on the period means (which test whether the means are precisely estimated) or CIs on the difference between treated and control (which test parallel trends). The latter is what matters for parallel trends assessment.

#### 5B: Propensity Score Matching (PSM)

**What is good**: The matching methods (1:1, 1:k, caliper) are standard and correctly described. The balance diagnostics (Love plot, SMD table, propensity overlap) are the right outputs. The Abadie-Imbens SE option is a good touch.

**Issues**:

1. [MAJOR] **The propensity model is limited to logistic regression.** While this is the most common choice, the literature has moved toward more flexible propensity models. Generalized Boosted Models (GBM) for propensity score estimation, as implemented in R's `twang` package, are standard in HEOR and consistently produce better balance than logistic regression when the true propensity surface is non-linear. The proposal should at minimum offer `sklearn.ensemble.GradientBoostingClassifier` as an alternative propensity model, selectable via dropdown. This is a near-zero-cost addition (the interface to `NearestNeighbors` is identical once propensity scores are computed) that substantially strengthens the tool's credibility with HEOR-trained users.

2. [MINOR] The proposal states "Abadie-Imbens SEs or bootstrap SEs (100 replicates, adjustable)." Abadie-Imbens SEs are not trivially available in Python -- there is no standard `statsmodels` or `sklearn` function that computes them. The proposal should either cite the specific implementation plan (manual computation following Abadie and Imbens 2006, 2011) or acknowledge this as a stretch goal and default to bootstrap SEs. At 100 bootstrap replicates, the SE estimate will be noisy; 500-1000 replicates is more standard but may push computation time on large datasets.

3. [MINOR] Matching without replacement is implied but not stated. For 1:1 matching, specify whether matching is with or without replacement, as this affects SE computation and effective sample size.

#### 5C: Inverse Probability Weighting (IPW)

**What is good**: The weight formulas for ATE and ATT are correct. Stabilized weights are correctly specified. The trimming and effective sample size reporting are appropriate.

**Issues**:

1. [MINOR] The proposal does not mention doubly robust estimation (Augmented IPW / AIPW), which combines outcome modeling with propensity weighting to achieve consistency if either the propensity model or the outcome model is correctly specified. AIPW is increasingly standard in HEOR and is recommended by ISPOR guidance. Adding AIPW would require only a weighted outcome regression with the IPW weights applied to a residualized outcome -- perhaps 30-50 additional lines of code.

2. [MINOR] The positivity assumption check is mentioned ("0 < P(T=1|X) < 1 for all X") but the implementation only addresses this via weight trimming. The tool should also report the range of estimated propensity scores and warn if any are below 0.01 or above 0.99, as these indicate near-violations of positivity that trimming alone may not adequately address.

#### 5D: Regression Discontinuity Design (RDD)

**What is good**: The choice of `rdrobust` is correct -- it is the standard implementation. The CCT bandwidth selector is appropriate. The McCrary density test for manipulation is standard.

**Issues**:

1. [MAJOR] **The `rdrobust` Python package is a pure-Python port that has limited maintenance and community adoption compared to the R version.** My search confirms it is available on PyPI as version 1.3.0 with a `py3-none-any` wheel (no compiled extensions), which is good for Windows compatibility. However, the Python port (`rdrobust` on PyPI) has significantly fewer downloads and less testing than the R original. The risk is not installation failure but subtle numerical differences or missing features compared to the R implementation.

   More importantly, the RDD use case in pharma promotional analytics is narrow. The proposal correctly notes it applies "when treatment assignment is based on a continuous running variable with a known cutoff (e.g., decile-based targeting)." In practice, very few pharma promotional campaigns use a strict cutoff-based assignment rule. Decile-based targeting is fuzzy (field reps exercise discretion), and spend thresholds are guidelines, not rules. The proposal should be honest about this: RDD is included for completeness but will be applicable to a minority of use cases. This does not argue for removing it (it is low-cost to include), but the UI should not present it as a co-equal method with DiD and PSM.

   **Recommendation**: No action required beyond expectation-setting in the UI. The "Advanced Methods" gating proposed for Synthetic Control should also apply to RDD, or at minimum RDD should include a tooltip: "Applicable only when treatment assignment follows a strict threshold rule on a continuous variable."

2. [MINOR] The proposal lists "IK as alternative" bandwidth selector. The IK (Imbens-Kalyanaraman 2012) bandwidth is an older method superseded by the CCT (Calonico-Cattaneo-Titiunik 2014) selector. Including IK as an option is fine for robustness checking but the default should clearly be CCT, and the UI should not present IK as an equally valid alternative.

#### 5E: Synthetic Control

**What is good**: The choice of `pysyncon` is appropriate. The Abadie-Diamond-Hainmueller SCM plus penalized SCM is the right method set. The placebo test (in-space) is correctly described. The performance warning and gating behind "Advanced Methods" are sensible.

**Issues**:

1. [MAJOR] **`pysyncon` depends on `cvxpy`, which in turn depends on compiled solver backends.** While `cvxpy` itself installs cleanly on Windows via pip wheels, its default solvers (OSQP, SCS, ECOS) also require compiled extensions. As of 2025-2026, these are available as pre-built wheels on PyPI for common Python versions (3.9-3.12, Windows x64), but the proposal pins `numpy` to `<2.0`. NumPy 2.0 was released in June 2024, and by February 2026, most packages in the scientific Python ecosystem have migrated to NumPy 2.x. Pinning `numpy<2.0` is increasingly problematic:
   - `scipy>=1.14` requires NumPy 2.x
   - `scikit-learn>=1.5` is built against NumPy 2.x
   - `statsmodels>=0.14.2` supports NumPy 2.0
   - Pre-built `cvxpy` wheels from late 2025 onward may assume NumPy 2.x

   **Recommendation**: Remove the `numpy<2.0` pin. Pin to `numpy>=1.24,<3.0` or simply `numpy>=2.0,<3.0` to align with the current ecosystem. Test the full dependency stack against NumPy 2.x before finalizing requirements.txt.

2. [MINOR] The proposal caps the donor pool at 200 units by default. This is reasonable, but the default cap should be documented as adjustable and the user should see the uncapped donor count alongside the capped count: "500 potential donors available; using 200 (adjust in settings)."

3. [MINOR] The MSPE ratio (treated/median placebo) is listed as an output but the inferential logic is not specified. The standard approach is to compute the ratio of post-treatment RMSPE to pre-treatment RMSPE for the treated unit and all placebos, then rank the treated unit's ratio among all placebos. The p-value is the rank divided by the number of placebos + 1. This should be stated explicitly.

#### 5F: Covariate-Adjusted DiD (ANCOVA)

**What is good**: The ANCOVA form (`Y_post - Y_pre ~ Treated + X_covariates`) is correctly specified. This is a useful complement to standard DiD.

**Issues**:

1. [MINOR] The distinction between the ANCOVA form and the fully interacted form (`Y ~ Post + Treated + Post*Treated + X + Post*X`) should be made clearer for the user. These are mathematically distinct estimators with different efficiency properties (McKenzie 2012). The ANCOVA form is generally more efficient when baseline covariates are available. The UI should explain when to use each, or default to ANCOVA and offer the interacted form as an option.

#### 5G: Sensitivity / Robustness Tab

**What is good**: The collection of robustness checks is well-chosen. Rosenbaum bounds, placebo outcome tests, pre-period falsification, leave-one-out sensitivity, and caliper sensitivity are all standard and appropriate.

**Issues**:

1. [MAJOR] **Rosenbaum bounds have no established Python implementation.** The proposal states "manual computation following Rosenbaum (2002), or `sensitivity2x2xk` logic adapted from R's `sensitivitymult`." There is no Python package on PyPI that implements Rosenbaum bounds as of February 2026. Porting the R `rbounds` or `sensitivitymv` package logic is feasible but non-trivial -- it requires implementing the Wilcoxon signed-rank test sensitivity analysis, which involves solving for the critical Gamma value that makes the p-value cross 0.05 across a range of Gamma values. This is perhaps 100-200 lines of custom code with careful numerical implementation. The effort estimate for `sensitivity.py` (200-300 LOC, 3-4 hours) may be tight if Rosenbaum bounds are included. Budget 5-6 hours for this module if Rosenbaum bounds are in scope.

2. [SUGGESTION] The E-value (VanderWeele and Ding 2017) is an increasingly popular alternative to Rosenbaum bounds for sensitivity analysis in HEOR, and is simpler to compute (a closed-form formula based on the point estimate and confidence interval). Consider offering both Rosenbaum bounds and the E-value, or substituting the E-value if Rosenbaum bounds prove too complex to implement correctly in the time budget.

3. [MINOR] The placebo outcome test is well-described but should include a warning: the test is only valid if the placebo outcome is truly unaffected by treatment. If the user selects a pre-period outcome that is correlated with the post-period outcome (e.g., prior Rx volume), a significant result may indicate model misspecification rather than confounding.

### Section 6: Balance Diagnostics Module

**What is good**: The SMD table, Love plot, covariate distribution overlays, and propensity score overlap plot are all standard and correctly specified. The traffic-light thresholds (|SMD| < 0.1 green, 0.1-0.25 yellow, > 0.25 red) follow Austin (2009). The dynamic update on parameter change is a strong UX feature.

**Issues**:

1. [MINOR] The SMD thresholds are for continuous covariates. For binary covariates, the raw difference in proportions (or odds ratio) is sometimes preferred over SMD. The proposal should specify how SMD is computed for categorical covariates: presumably using the formula `(p_treated - p_control) / sqrt((p_treated*(1-p_treated) + p_control*(1-p_control))/2)` for each level of a dummy-coded variable. This should be stated explicitly.

2. [SUGGESTION] Consider adding a variance ratio diagnostic (variance of covariate in treated / variance in control, targeting 1.0) alongside SMD. Rubin (2001) recommends both.

### Section 7: Geographic Visualization Module

**What is good**: The Plotly-primary decision is correct for the stated constraints. The "actionable zones" quadrant (Opportunity/Optimized/Diminishing Returns/Cold) is the most differentiated feature of the entire tool and demonstrates genuine product thinking. The ZIP-to-centroid lookup table approach avoids GDAL/geopandas dependency.

**Issues**:

1. [BLOCKER] **Rendering a choropleth of all ~33,000 US ZIP codes in Plotly will be extremely slow or crash on an 8 GB RAM laptop.** The full ZCTA GeoJSON file for the US is approximately 850 MB uncompressed (1.3 GB as GeoJSON). Even with simplified geometries (tolerance 0.0005), the file is 200-400 MB. Loading this into Plotly's client-side JavaScript renderer will consume several GB of browser memory and take 30-60+ seconds to render, if it renders at all.

   The proposal mentions "pre-aggregate to ZIP level before mapping" and "simplify geometries" as mitigations, but does not address the fundamental problem: the GeoJSON geometry file is too large for client-side rendering on a resource-constrained laptop.

   **Recommended solutions** (pick one):
   - **Option A: Render only the ZIPs present in the user's data, not all 33K.** If the CSV contains 200-500 unique ZIPs, load only those ZIP polygons from a pre-indexed GeoJSON. This requires splitting the national GeoJSON into per-state or per-ZIP files, or using a binary spatial index (e.g., FlatGeobuf format). This is the most practical approach.
   - **Option B: Use Plotly scatter_mapbox with ZIP centroids instead of choropleth polygons.** Plot colored circles at ZIP centroids (from the bundled `zip_latlon.csv`). This avoids the GeoJSON entirely and renders in <2 seconds. The visual result is less polished but perfectly functional for the use case.
   - **Option C: Use a tile-based approach (Folium) for full choropleth and Plotly scatter for the lightweight view.** This contradicts the "Plotly primary" decision but acknowledges the reality that Plotly choropleth at ZIP granularity is impractical offline.

   **My recommendation**: Default to Option B (scatter_mapbox with centroids) for all datasets. Offer Option A (filtered polygon choropleth) as an opt-in for users who need true polygon rendering and have data covering fewer than 500 ZIPs. This avoids the GeoJSON bloat problem entirely for the default case.

2. [MAJOR] **The "Opportunity Zone" extrapolation is scientifically questionable and should carry a strong caveat.** The proposal states that "predicted lift" for opportunity zones uses "the fitted causal model to extrapolate expected effect for untreated/lightly-treated ZIPs." This is causal extrapolation to out-of-sample units -- predicting the treatment effect for units that were never treated, based on a model fitted to treated units. This is a fundamentally different (and much harder) problem than estimating the ATT for treated units. The overlap assumption (Section 5B) requires that untreated units have a positive probability of being treated; for "opportunity zone" ZIPs that received zero tactic investment, this assumption is violated by definition.

   The proposal acknowledges this with a caveat label: "Model-predicted, not observed." This is necessary but insufficient. The UI should also display uncertainty bands that reflect the extrapolation risk -- which will be wide, because the model has no data from these regions. A confidence interval computed from the fitted model will understate the true uncertainty because it does not account for model misspecification in the extrapolated region.

   **Recommendation**: Keep the opportunity zone feature (it is commercially useful), but: (a) default to displaying only observed effects, with opportunity zone predictions behind an explicit toggle, (b) add a prominent disclaimer: "Predicted effects for untreated ZIPs are model extrapolations with high uncertainty. They should inform hypothesis generation, not resource allocation decisions," and (c) display prediction intervals (not confidence intervals) that account for both estimation uncertainty and prediction uncertainty.

3. [MINOR] The proposal plans to bundle `data/zip_latlon.csv` (~33K rows, ~1 MB) for ZIP-to-centroid mapping. This is fine, but the file should also include state FIPS codes and state abbreviations to support per-state filtering in the geographic view. The Census ZCTA centroid file includes these fields; ensure they are preserved.

4. [MINOR] Folium's tile layer requires internet access for initial tile download. The proposal correctly notes this but should also note that Plotly's `scatter_mapbox` (if used instead of `choropleth`) also requires a Mapbox tile layer by default, which requires internet. To be truly offline, use `plotly.graph_objects.Scattergeo` (vector-based, no tile server) rather than `scatter_mapbox`.

### Section 8: User Interface Architecture

**What is good**: The Streamlit decision is correct and well-justified. The layout specification is detailed and demonstrates genuine product thinking. The tab structure (Overview / Balance / Results / Geo / Export) is logical.

**Issues**:

1. [MINOR] The PyInstaller fallback is mentioned but not scoped. Packaging a Streamlit app with PyInstaller is non-trivial: Streamlit has complex runtime dependencies (tornado, protobuf, pyarrow) and the resulting `.exe` is typically 200-500 MB. The proposal should either (a) scope the PyInstaller effort as a specific Phase 3 deliverable with its own time estimate (8-12 hours), or (b) acknowledge it as a post-v1 stretch goal and remove it from the Phase 3 scope.

2. [SUGGESTION] The forest plot comparing methods (Results tab) is an excellent feature that should be emphasized. Showing DiD, PSM, and IPW estimates side by side with CIs on a single plot is the most powerful communication tool in the dashboard. Ensure it is prominently placed and clearly labeled.

### Section 9: Performance Constraints

**What is good**: The bottleneck analysis is realistic and well-calibrated. The mitigation strategies (lazy computation, caching, sampling option, progress indicators) are appropriate.

**Issues**:

1. [MINOR] The Synthetic Control performance estimate ("100 donors x 20 periods: ~10-30 seconds. 500 donors: ~2-5 minutes") should be verified empirically during development. `pysyncon`'s optimization uses `scipy.optimize.minimize` with bounds constraints, not `cvxpy` directly (despite the proposal listing `cvxpy` as a dependency). The actual performance depends on the solver convergence behavior, which varies with data characteristics. The proposal should include a benchmark test in the test suite that times SCM on synthetic data of increasing size.

2. [SUGGESTION] For datasets > 100K rows, the proposal offers random sampling. An alternative that preserves the causal design is stratified sampling by treatment status and key covariates. Random sampling could distort the treated/control ratio and covariate balance. Offer stratified sampling as the default, with random as a fallback.

### Section 10: Dependency Stack

**What is good**: The dependency selections are generally sound. The explicit exclusions (geopandas, causalml, dowhy, folium) are well-reasoned. The Windows compatibility notes are accurate.

**Issues**:

1. [BLOCKER] **The `numpy<2.0` pin is incompatible with the rest of the dependency stack as specified.** As detailed in Section 5E above, `scipy>=1.14` and `scikit-learn>=1.5` are built against NumPy 2.x. Pinning `numpy<2.0` while requiring `scipy>=1.11` and `scikit-learn>=1.3` creates a dependency resolution conflict for any pip install performed after mid-2025, because newer versions of scipy and scikit-learn will pull NumPy 2.x. The resolution matrix is:
   - If you want `numpy<2.0`: pin `scipy<1.14`, `scikit-learn<1.5`, `statsmodels<0.14.2`. This freezes the stack at mid-2024 versions.
   - If you want current versions of scipy/sklearn/statsmodels: require `numpy>=2.0`.

   **Recommendation**: Pin to `numpy>=2.0,<3.0` and verify all other dependencies are NumPy 2.x compatible (they are, as of late 2025).

2. [MAJOR] **Kaleido (Plotly static image export) has a breaking version change.** Kaleido 1.0.0, released mid-2025, is incompatible with Plotly 5.x (the version range specified in the proposal). Kaleido 1.0.0 requires Plotly >= 6.1.1. Additionally, Kaleido 1.0.0 no longer bundles Chromium -- it requires Chrome to be installed on the system. On a corporate laptop where users may not have admin rights to install Chrome, this is a deployment blocker.

   **Recommended solutions**:
   - Pin `kaleido>=0.2,<1.0` to use the legacy version that bundles Chromium. This is compatible with Plotly 5.x.
   - Alternatively, upgrade to `plotly>=6.1` and `kaleido>=1.0`, but verify Chrome availability on target machines.
   - As a fallback, use `matplotlib` for all static image exports (the proposal already lists matplotlib as a dependency) and skip Plotly static export entirely. Interactive Plotly charts can be saved as HTML files without kaleido.

   **My recommendation**: Pin `kaleido>=0.2,<1.0` and note in the README that static Plotly image export requires the legacy kaleido version.

3. [MINOR] Missing dependency: `jenkspy` for Jenks natural breaks in the threshold module (Section 4), unless replaced with a scipy-based alternative.

4. [MINOR] Missing dependency: `csdid` for Callaway-Sant'Anna staggered DiD, if implemented per the recommendation in Section 5A.

5. [MINOR] The `linearmodels>=6.0` version constraint is outdated. The current version is 7.0 (released October 2025). Pin to `>=6.0,<8.0` or update to `>=7.0,<8.0`.

### Section 11: Project Structure

**What is good**: Clean, modular structure. Separation of modules by method is correct. The `utils/` split (validators, exporters) is appropriate.

**Issues**:

1. [SUGGESTION] Consider adding a `modules/common.py` or `modules/propensity.py` that encapsulates the shared propensity score estimation logic used by both PSM and IPW. Currently, the proposal implies both `psm.py` and `ipw.py` independently implement propensity score estimation via LogisticRegression. Extracting this into a shared module reduces duplication and ensures consistent propensity models across methods.

### Section 12: Sample Data Specification

**What is good**: The synthetic DGP is well-specified with a known ground truth (ATT = 3.5). The confounding structure (Rheumatology and higher deciles more likely treated AND higher baseline Rx) is realistic. The parallel trends by construction ensures DiD is valid on this data. The seed=42 for reproducibility is correct.

**Issues**:

1. [MAJOR] **The DGP is insufficient for validating all proposed methods.** Specifically:
   - **RDD**: The DGP has no running variable with a cutoff. The proposal acknowledges this ("a synthetic RDD dataset, to be generated separately") but does not include this dataset in the specification. Without it, RDD cannot be validated during development.
   - **Synthetic Control**: The DGP has 500 NPIs with 40% treated (200 treated). Synthetic Control is designed for few treated units (3-10). The DGP does not include a scenario with, e.g., 5 treated ZIPs out of 200 ZIPs, which is the canonical SCM use case.
   - **Staggered adoption**: All treated units are treated for the same post-period (weeks 11-20). There is no staggered adoption, so the DGP cannot validate Callaway-Sant'Anna DiD.

   **Recommendation**: Generate 3 synthetic datasets, not 1:
   - `synthetic_promo.csv`: The existing DGP for DiD, PSM, IPW validation (keep as-is).
   - `synthetic_rdd.csv`: A dataset with a continuous running variable (e.g., decile score) and a sharp cutoff at decile 7, with a known LATE at the cutoff.
   - `synthetic_scm.csv`: A panel dataset with 200 ZIPs, 20 time periods, and 5 treated ZIPs, with a known treatment effect inserted at period 11.

   This adds perhaps 1-2 hours to the sample data generation step but is essential for validating the full method stack.

2. [MINOR] The DGP includes `tactic_b` (continuous digital impressions, correlated with `tactic_a` at r~0.3) but the DGP does not specify whether `tactic_b` has its own causal effect on the outcome. If `tactic_b` is an active confounder (affects both treatment probability and outcome), it should be included in the outcome equation. If it is irrelevant, it should still be included as a candidate covariate to test whether the tool correctly identifies it as uninformative.

3. [MINOR] The DGP generates `rx_volume` as `beta_0 + beta_1*treated + beta_2*post + beta_3*treated*post + X*gamma + epsilon`. The specific values of beta_0 through beta_3 and gamma are not stated (only beta_3 = 3.5 is given as the ATT). For reproducibility and test calibration, all DGP parameters should be specified.

### Section 13: Risks and Open Questions -- Recommendations for R1-R6

#### R1: Streamlit Viability in Corporate Environment

**Recommendation: Agree with the proposer's leaning. Streamlit primary, PyInstaller as documented fallback, HTML not pursued.**

The proposer is correct that the statistical backbone requires Python, making HTML a non-starter. Streamlit installs via `pip install --user streamlit` without admin rights, and serves on localhost with no external network exposure. The PyInstaller fallback is viable but should be scoped as a post-v1 effort (see Section 8 critique).

One additional consideration not mentioned: Streamlit's default behavior opens a browser tab on `localhost:8501`. On some corporate machines, the firewall may block localhost connections or flag the attempt. The `--server.address 127.0.0.1` flag and `--server.headless true` should be documented as defaults for corporate deployment.

**Severity**: [MINOR] -- no action required beyond documenting the `--server.address` and `--server.headless` flags.

#### R2: Synthetic Control Gating

**Recommendation: Agree with the proposer's leaning. Gate behind "Advanced Methods" with a performance warning.**

The gating is appropriate both for performance (SCM is O(donors^2 * periods) in optimization time) and for appropriateness (SCM requires few treated units, which is a niche use case). The warning text should include the minimum recommended number of pre-treatment periods (at least 5-10 for reliable fit) in addition to the maximum recommended number of treated units.

I also recommend gating RDD behind the same "Advanced Methods" toggle, as RDD is similarly niche in pharma analytics.

**Severity**: [MINOR] -- agree with proposal, add RDD to the gating.

#### R3: Geographic Visualization Placement

**Recommendation: Agree with the proposer's leaning. Separate tab with method-selector dropdown.**

The proposer's reasoning is sound: geographic visualization is about *where*, not *which method*. The Geo tab should include a dropdown to select which method's estimates to map, and a "Compare Methods" option that shows a small multiples grid (one map per method) if multiple methods have been run.

**Severity**: [MINOR] -- no action required.

#### R4: HIPAA / Data Governance

**Recommendation: Agree with the proposer's analysis, with one addition.**

NPI-level promotional data is professional data, not PHI. The proposer's three precautions (local RAM only, no shared server deployment, de-identify PHI before upload) are appropriate. However:

[SUGGESTION] Add a data classification acknowledgment checkbox on the upload screen: "I confirm this data has been reviewed for compliance with UCB data governance policies and does not contain Protected Health Information (PHI)." This costs zero development time (one `st.checkbox` call) and provides a paper trail. It also signals to users that data governance is taken seriously, which builds trust.

#### R5: Panel Data Support

**Recommendation: Agree with the proposer's leaning, with caveats.**

Supporting both pre/post and multi-period data is correct. The event study (dynamic DiD) naturally handles multiple periods. However, the proposal should be explicit about how PSM and IPW handle multi-period data:

- PSM can match on baseline (pre-period) covariates and compare post-period outcomes. If there are multiple post-periods, the analyst must choose: match once and compare outcomes at each post-period separately, or match once and compare the average post-period outcome. The default should be the latter (average post-period) with per-period results available as a drill-down.
- IPW is similarly a cross-sectional estimator applied to panel data. The standard approach is to estimate propensity at baseline and apply weights to the post-period outcome.

These details are not method limitations but they need to be specified in the UI flow so users are not confused about what "multi-period" means for each method.

**Severity**: [MINOR] -- add UI guidance for PSM/IPW multi-period handling.

#### R6: Bayesian A/B Test Integration

**Recommendation: Exclude from v1. Revisit for v2 if stakeholder demand exists.**

The proposer's analysis is honest: Bayesian A/B testing "does *not* address confounding -- it is a comparison of means with Bayesian uncertainty." Including it in a causal inference dashboard creates three problems:

1. **Conceptual confusion**: The entire premise of ACID-Dash is that comparing means is wrong for non-randomized data. Adding a "Bayesian comparison of means" module undermines this message. Users who do not read the caveats will run the Bayesian A/B test, get a P(superiority) of 95%, and declare the tactic effective -- exactly the naive analysis ACID-Dash is designed to prevent.

2. **UI clutter**: Seven causal methods plus a Bayesian comparison is eight options. For a tool targeting "commercial strategists who may not have training in causal inference methodology," this is too many choices. Each additional method increases cognitive load and the probability of misuse.

3. **False legitimacy**: Placing a non-causal method alongside causal methods in the same dashboard implicitly endorses it as a valid analytical approach for this data type. Users will assume that if it is in the tool, it is appropriate.

The proposer's mitigation (label it "Descriptive Bayesian Comparison, not causal") is necessary but insufficient. Labels are read only by careful users; careless users will see "P(superiority) = 97%" and stop reading.

**Recommendation**: Do not include the Bayesian A/B test in v1. If stakeholders specifically request it after using the tool, add it in v2 behind an explicit "I understand this does not address confounding" checkbox. The 100 LOC estimate is probably correct, but the risk of misuse exceeds the communication value.

**Severity**: [MAJOR] -- exclude from v1 scope.

### Section 14: Success Metrics

**What is good**: The correctness metrics (recover ATT within tolerance on synthetic data) are well-defined. The cross-validation against R (`MatchIt`, `fixest`) is an excellent quality gate.

**Issues**:

1. [MINOR] Criterion 1 ("Recovers the true ATT 3.5 +/- 0.5 with p < 0.05") is not a valid success criterion. The p < 0.05 threshold is a function of sample size and effect size, not of implementation correctness. With N=10,000 and a true ATT of 3.5, any correct implementation will achieve p < 0.05. The meaningful criterion is: "Recovers the true ATT within the 95% CI of the estimator." If the true value (3.5) falls outside the 95% CI, something is wrong. If it falls inside but with p > 0.05 (unlikely at this sample size), the implementation is still correct.

2. [MINOR] Criterion 6 ("SMD values match R's `MatchIt::summary()` to 2 decimal places") is excellent but should also include matching `fixest::feols()` coefficient estimates and clustered SEs to 2 decimal places. Standard errors are more sensitive to implementation differences than point estimates.

3. [MINOR] Criterion 15 ("non-technical user can obtain a DiD result with geographic visualization in <5 minutes") is desirable but not testable without user testing. Consider replacing with a concrete proxy: "App loads and displays overview tab within 10 seconds; user can select method, click Run, and view results within 3 additional clicks from the overview tab."

### Section 15: Effort Estimate

**What is good**: The module-level estimates are granular and the per-module LOC ranges are reasonable. The 40-55 hour total with Claude Code assistance is in the right ballpark for the core functionality.

**Issues**:

1. [MAJOR] **The effort estimate is optimistic by approximately 30-40%.** Specific underestimates:

   | Module | Proposal Estimate | My Estimate | Reason |
   |--------|------------------|-------------|--------|
   | `sensitivity.py` | 3-4 hours | 5-7 hours | Rosenbaum bounds require manual implementation from R port; no Python package exists |
   | `geo.py` | 4-5 hours | 6-8 hours | GeoJSON performance problem (Blocker B1) requires significant architectural work |
   | `synth.py` | 3-4 hours | 4-6 hours | `pysyncon` API wrapping is straightforward, but placebo tests with progress bars add complexity |
   | Integration + debugging | 5-8 hours | 10-15 hours | Cross-module state management in Streamlit (session_state, caching invalidation) is consistently underestimated |
   | `exporters.py` | 3-4 hours | 5-7 hours | PDF generation with embedded plots is fiddly; kaleido version issues add debugging time |
   | Tests | 4-6 hours | 6-8 hours | 3 synthetic datasets instead of 1; cross-validation against R adds comparison logic |

   **Revised estimate**: 55-75 hours of development time, 20-25 hours of human review, 7-9 weeks calendar time.

   This does not change the Go recommendation -- a 9-week timeline is still reasonable for evenings/weekends. But the Phase 3 estimate should be extended by 1-2 weeks to accommodate the geographic visualization rework and export debugging.

2. [MINOR] The Phase Plan allocates weeks 1-3 to "Core pipeline (CSV upload, column detection, validation, DiD, PSM, balance diagnostics, basic UI)." This is the right Phase 1 scope. However, getting a functional Streamlit app with CSV upload, column assignment, DiD, PSM, and balance diagnostics in 3 weeks at evening/weekend pace (24-30 hours) is aggressive. The proposer should identify a minimum viable demo for the end of Week 2 (CSV upload + column detection + raw DiD only, no balance diagnostics or PSM) to create an early milestone that reduces schedule risk.

### Hypothesis

**What is good**: The implicit hypothesis is honestly framed. The falsification criterion (domain experts find outputs no more useful than existing analyses) is appropriate for a tool project.

**Issues**:

1. [MINOR] The hypothesis is about comparative utility ("more defensible than current practice"), but the success metrics (Section 14) are about technical correctness (recovering known ATT values). These are different axes. Technical correctness is necessary but not sufficient for "more defensible" -- the outputs must also be understandable and actionable. Consider adding a qualitative success criterion: "At least one commercial analytics colleague reviews the tool output and confirms it provides information they would use in a tactic evaluation."

### Cross-Pollination

**What is good**: The connection to CBRE is genuine (shared causal reasoning intellectual infrastructure). The "opportunity zone" / active learning parallel to ECOCHEMICAL's BoTorch loop is an interesting conceptual link. The newsletter piece ("Your A/B Test Is Lying to You") is strong content.

**Issues**:

1. [MINOR] The proposal states that `validators.py` and `exporters.py` patterns "could be generalized into `commons/`." This should be a concrete commitment for the PR: after ACID-Dash Phase 1 is complete, extract CSV validation utilities and PDF report generation into `commons/` if they are reusable. Do not let this remain aspirational.

---

## Missing Methods, Diagnostics, and Features Standard in HEOR

The following are standard in health economics and outcomes research but absent from the proposal:

1. [MAJOR] **Staggered DiD (Callaway-Sant'Anna 2021)**: As detailed in Section 5A critique. Essential for pharma promotional campaigns with rolling rollouts.

2. [MAJOR] **Doubly Robust Estimation (AIPW)**: Combines outcome modeling with propensity weighting. Increasingly required by ISPOR guidance for observational studies. Low implementation cost (30-50 LOC on top of existing IPW module).

3. [MINOR] **Negative Control Outcomes**: The placebo outcome test (Section 5G) partially addresses this, but a formal negative control outcome framework (Lipsitch, Tchetgen Tchetgen, and Cohen 2010) would strengthen credibility with epidemiologists.

4. [MINOR] **Target Trial Emulation Framework**: The proposal uses traditional DiD/PSM/IPW framing. The HEOR field is moving toward the "target trial emulation" framework (Hernan and Robins 2016) as the organizing principle for observational causal inference. While this does not change the underlying methods, framing the tool's workflow as "define the target trial, then select the estimation method" would resonate with HEOR-trained users.

5. [SUGGESTION] **Covariate Balance Propensity Score (CBPS)**: An alternative to logistic regression for propensity estimation that directly optimizes for covariate balance rather than predictive accuracy. Available in R (`CBPS` package) but would need Python implementation. Nice-to-have, not essential.

6. [SUGGESTION] **ISPOR PALISADE Checklist Integration**: The ISPOR Task Force published a "Good Practices Report" checklist for ML methods in HEOR. While ACID-Dash uses traditional methods (not ML), adapting a subset of this checklist as an auto-generated quality checklist in the PDF report would enhance credibility.

---

## Methodological Errors and Clarifications

1. [MINOR] **Section 5C (IPW)**: The ATT weight formula is stated as `w_i = T_i + (1-T_i) * e_i/(1-e_i)`. This is correct but should be noted as producing weights that sum to the treated sample size for treated units and to a pseudo-count for control units. The stabilized ATT weight should be `w_i = T_i + (1-T_i) * (e_i/(1-e_i)) * (P(T=0)/P(T=1))`, which is different from the ATE stabilization formula. The proposal's stabilization description ("multiply by P(T=1) and P(T=0) respectively") is ambiguous -- specify whether it applies to ATE weights, ATT weights, or both, and give the exact formulas.

2. [MINOR] **Section 5D (RDD)**: The estimand is stated as LATE ("the causal effect of treatment for units right at the threshold"). Strictly, the sharp RDD estimand is the Average Treatment Effect at the cutoff, which equals the LATE only if the RDD is sharp. If the RDD is fuzzy (treatment uptake is not 100% at the cutoff), the estimand is the LATE via a Wald/2SLS estimator. The proposal does not distinguish between sharp and fuzzy RDD. Add a UI element that asks: "Is treatment assigned deterministically by the running variable (sharp RDD) or probabilistically (fuzzy RDD)?"

3. [MINOR] **Section 5A (DiD)**: The SUTVA assumption is listed but not operationalized. In pharma promotional analytics, spillover is common: detailing one HCP in a practice can influence prescribing by other HCPs in the same practice. If the unit of analysis is HCP but multiple HCPs share a practice, SUTVA is violated. The tool should at minimum note this risk and suggest analyzing at the practice or ZIP level to reduce spillover.

---

## Blocking Issues Summary

| # | Issue | Section | Fix |
|---|-------|---------|-----|
| B1 | ZIP-level choropleth with ~33K polygons will crash or hang on 8 GB laptop | 7.1 | Default to scatter plot at ZIP centroids; offer filtered polygon choropleth for <500 ZIPs |
| B2 | `numpy<2.0` pin is incompatible with current scipy/sklearn/statsmodels versions | 10 | Remove `numpy<2.0` pin; pin to `numpy>=2.0,<3.0` |

---

## Major Concerns Summary

| # | Issue | Section | Fix |
|---|-------|---------|-----|
| M1 | Missing staggered DiD (Callaway-Sant'Anna); classical TWFE biased for staggered adoption | 5A | Add `csdid` as optional DiD variant; at minimum add UI warning for staggered designs |
| M2 | No doubly robust estimation (AIPW); increasingly required by ISPOR | 5C | Add AIPW as IPW extension (~30-50 LOC) |
| M3 | Propensity model limited to logistic regression; GBM is standard in HEOR | 5B | Add `GradientBoostingClassifier` as alternative propensity model |
| M4 | Kaleido 1.0.0 incompatible with Plotly 5.x and requires Chrome; corporate deployment risk | 10 | Pin `kaleido>=0.2,<1.0` |
| M5 | Rosenbaum bounds have no Python implementation; effort underestimated | 5G | Budget 5-7 hours; consider E-value as simpler alternative |
| M6 | Sample DGP insufficient for RDD and SCM validation; missing 2 of 3 needed synthetic datasets | 12 | Generate 3 synthetic datasets (promo, RDD, SCM) |
| M7 | Bayesian A/B test creates confusion risk; undermines causal premise of the tool | R6, 13 | Exclude from v1 |
| M8 | Effort estimate optimistic by ~30-40%; integration/debugging underestimated | 15 | Revise to 55-75 hours; extend timeline by 1-2 weeks |
| M9 | No positioning against existing causal inference libraries (CausalML, EconML, DoWhy) | 2 | Add explicit differentiation section |
| M10 | Opportunity zone extrapolation violates overlap assumption; risk of misleading users | 7.3 | Gate behind explicit toggle; add prominent disclaimer; use prediction intervals |

---

## Minor Issues Summary

| # | Issue | Section | Fix |
|---|-------|---------|-----|
| m1 | Tool name implies randomized A/B tests; methods are for observational data | 1 | Acknowledge naming tension in README |
| m2 | Target audience definition unclear (methodologists without Python vs. business users without causal training) | 1 | Sharpen target audience specification |
| m3 | NPI heuristic should accept 10-digit integers starting with 1 or 2 | 3 | Update heuristic |
| m4 | Missing method-eligibility validation (grey out DiD if single period, grey out RDD if no running variable) | 3 | Add method eligibility checks based on column assignments |
| m5 | Missing `jenkspy` dependency for Jenks natural breaks | 4, 10 | Add to requirements or use scipy-based alternative |
| m6 | Event study formula omits unit and time fixed effects notation | 5A | Specify full FE event study equation |
| m7 | Abadie-Imbens SEs not available in standard Python packages | 5B | Acknowledge as stretch goal; default to bootstrap |
| m8 | IPW stabilized weight formula ambiguous for ATT vs ATE | 5C | Specify exact formulas for both |
| m9 | Sharp vs fuzzy RDD not distinguished | 5D | Add UI element for sharp/fuzzy selection |
| m10 | SCM placebo test p-value computation not specified | 5E | Specify rank-based p-value formula |
| m11 | ANCOVA vs interacted DiD distinction not explained for users | 5F | Add UI guidance |
| m12 | SMD computation for categorical covariates not specified | 6.1 | Specify formula |
| m13 | `Scattergeo` needed for fully offline Plotly maps (not `scatter_mapbox`) | 7.4 | Use `Scattergeo` as default for offline |
| m14 | PyInstaller packaging not scoped with time estimate | 8.1 | Either scope at 8-12 hours or defer to post-v1 |
| m15 | `linearmodels` version constraint outdated (6.0 vs current 7.0) | 10 | Update to `>=7.0,<8.0` |
| m16 | Success criterion 1 conflates statistical significance with implementation correctness | 14 | Replace "p < 0.05" with "true ATT within 95% CI" |
| m17 | All DGP parameters (beta_0, beta_1, beta_2, gamma) not specified | 12 | Document all parameter values |
| m18 | SUTVA violation risk from within-practice spillover not noted | 5A | Add UI warning about spillover |

---

## What Is Genuinely Good

1. **The treatment threshold module (Section 4) is excellent UX design.** The draggable histogram slider with live N counter and mini-balance preview is the kind of interactive feature that makes the difference between a useful tool and a script. This demonstrates genuine product thinking that goes beyond "wrap statsmodels in Streamlit."

2. **The "actionable zones" quadrant (Section 7.3) is the most commercially differentiated feature.** Mapping tactic intensity against estimated lift to identify Opportunity/Optimized/Diminishing Returns/Cold zones is exactly the kind of strategic output that commercial analytics teams need. No existing open-source causal inference library provides this. If the extrapolation caveats are properly handled, this feature alone justifies the tool.

3. **The balance diagnostics module (Section 6) is thorough and correctly specified.** The Love plot, SMD table, propensity overlap, and covariate distribution overlays -- all with dynamic refresh on parameter change -- represent the complete standard diagnostic suite. The traffic-light thresholds follow Austin (2009). This module would pass peer review in a HEOR journal.

4. **The forest plot comparing methods (Section 8.2, Results tab) is a powerful communication tool.** Showing DiD, PSM, and IPW estimates side by side with CIs allows non-technical stakeholders to immediately assess whether the treatment effect is robust across methods.

5. **The cross-validation against R criterion (Section 14, Criterion 6) is an excellent quality gate.** Matching `MatchIt::summary()` and `fixest::feols()` output to 2 decimal places ensures implementation correctness against the gold standard.

6. **The dependency stack exclusions are well-reasoned.** Explicitly excluding geopandas (GDAL dependency), causalml (heavy footprint), and dowhy (conceptual overhead) shows awareness of the deployment constraints and a willingness to sacrifice features for reliability.

7. **The phased build plan (Phase 1: DiD + PSM + balance; Phase 2: extended methods + geo; Phase 3: polish + export) is correctly sequenced.** Phase 1 delivers a useful tool; subsequent phases add capabilities incrementally. This is exactly the right structure for a side project.

---

## How a Skeptical Colleague Would Challenge This

1. **"Just use EconML."** The most likely pushback from a technically savvy colleague is: "Why build a new tool when Microsoft's EconML already provides DiD, DML, causal forests, and IV estimators with a scikit-learn API?" The answer must be: ACID-Dash provides a GUI for non-coders, geographic visualization for field force planning, and an integrated diagnostic workflow -- none of which EconML offers. This differentiation must be stated in the proposal.

2. **"The geographic map is the only thing that is not already available elsewhere."** This is largely true. The causal methods, balance diagnostics, and sensitivity analyses are all available in existing Python/R packages. The ZIP-level geographic effect heterogeneity visualization with actionable zones is the unique contribution. The proposal should lean into this as the differentiator rather than framing the causal methods as novel.

3. **"Does this distract from CBRE and ECOCHEMICAL?"** ACID-Dash is a tool project, not a research project. It does not produce publications. At 55-75 hours over 7-9 weeks, it consumes approximately one-third of James's available development time during that period. If CBRE or ECOCHEMICAL are in active development sprints, ACID-Dash should be deprioritized. The proposal should include a portfolio-level scheduling note: "ACID-Dash development should not overlap with CBRE sprint weeks."

4. **"The opportunity zone predictions are snake oil."** Extrapolating causal effects to untreated regions is a fundamentally harder problem than estimating effects for treated regions. Without a strong structural model, these predictions are likely to be dominated by noise. The tool should be honest about this limitation rather than presenting color-coded "opportunity" zones that imply actionable intelligence.

---

## Recommendation

**Go with Conditions.**

ACID-Dash is a well-scoped, low-risk tool project with high practical utility for James's day job. The core methods are well-understood, the dependency stack is realistic, and Phase 1 (DiD + PSM + balance diagnostics) delivers immediate value. The proposal demonstrates excellent product thinking (threshold slider, actionable zones, forest plot, Love plot) that goes beyond a naive "wrap the statistics in a UI."

### Conditions for Proceeding

**Must fix before starting development (Blockers)**:

1. Resolve the geographic visualization architecture (B1). Default to scatter plot at ZIP centroids; polygon choropleth only for <500 ZIPs.
2. Fix the `numpy<2.0` pin (B2). Update to `numpy>=2.0,<3.0`.
3. Pin `kaleido>=0.2,<1.0` to avoid Plotly 5.x / Kaleido 1.0 incompatibility (M4).

**Must fix during Phase 1 (before Phase 2 begins)**:

4. Add explicit differentiation against CausalML, EconML, DoWhy (M9). Two paragraphs in the README.
5. Generate all 3 synthetic datasets (M6): promo, RDD, SCM.
6. Exclude Bayesian A/B test from v1 (M7).
7. Revise effort estimate to 55-75 hours and extend timeline by 1-2 weeks (M8).

**Should fix during Phase 2 (before Phase 3)**:

8. Add staggered DiD warning or Callaway-Sant'Anna implementation (M1).
9. Add GBM as alternative propensity model (M3).
10. Add AIPW as IPW extension (M2).
11. Gate opportunity zone predictions behind toggle with disclaimers (M10).

**Nice-to-have for v2**:

12. Rosenbaum bounds or E-value sensitivity analysis (M5).
13. Bayesian A/B test (if stakeholder demand, with anti-misuse safeguards).
14. PyInstaller packaging.
15. Target trial emulation framing.

### Portfolio Scheduling Note

ACID-Dash development should be scheduled in a window that does not overlap with CBRE sprint weeks. The current CBRE status is "Approved" and ECOCHEMICAL is "Approved with conditions." If either project is in an active development sprint, ACID-Dash takes second priority. The ideal sequencing: complete ACID-Dash Phase 1 before CBRE's next sprint begins, then interleave ACID-Dash Phases 2-3 with CBRE development as time permits.

### Bottom Line

Build it. Phase 1 is achievable in 3-4 weeks and immediately useful at James's day job. The blocking issues are all fixable with straightforward design changes (not fundamental reconception). The tool fills a genuine gap: not in causal inference methods (which exist in many packages) but in the integrated, self-service, offline, geographically-aware workflow that pharma commercial analytics teams actually need. The first time this tool produces a defensible tactic evaluation where a naive comparison would have given a misleading answer, it pays for itself.

---

## Scoring Summary

| Dimension | Score | Weight | Notes |
|-----------|-------|--------|-------|
| Novelty | 2.5/5 | Low | Tool integration, not methodological novelty; geographic features are the differentiator |
| Feasibility | 4/5 | High | Core methods well-understood; dependency stack realistic; laptop-scale verified; effort slightly underestimated |
| Impact | 4/5 | High | Direct professional utility; portable to any pharma analytics team; strong newsletter content |
| Fit with Portfolio | 3.5/5 | Medium | Good causal inference alignment; does not compete; time allocation risk vs CBRE/ECOCHEMICAL |
| Technical Soundness | 3.5/5 | High | Methods correctly described with minor gaps; staggered DiD and AIPW are notable omissions; GeoJSON blocker |
| Risk Assessment | 3.5/5 | Medium | Major risks honestly identified; missing GeoJSON performance risk and ecosystem compatibility risks |
| Literature Grounding | 3/5 | Medium | Core references correct; missing positioning against CausalML/EconML/DoWhy; missing Callaway-Sant'Anna |
| Budget Realism | 4.5/5 | Low | $0 cost; all open-source; no data acquisition; only risk is time investment |

**Overall weighted assessment: 3.6/5 -- a solid, practical tool project that needs minor scope adjustments and dependency fixes before proceeding. Recommended for development with conditions.**

---

## References Cited in This Review

- Austin, P.C. (2009). Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. Statistics in Medicine, 28(25), 3083-3107.
- Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with multiple time periods. Journal of Econometrics, 225(2), 200-230.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. American Economic Review, 110(9), 2964-2996.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. Journal of Econometrics, 225(2), 254-277.
- Hernan, M.A. & Robins, J.M. (2016). Using big data to emulate a target trial when a randomized trial is not available. American Journal of Epidemiology, 183(8), 758-764.
- Lipsitch, M., Tchetgen Tchetgen, E. & Cohen, T. (2010). Negative controls: a tool for detecting confounding and bias in observational studies. Epidemiology, 21(3), 383-388.
- McKenzie, D. (2012). Beyond baseline and follow-up: The case for more T in experiments. Journal of Development Economics, 99(2), 210-221.
- Sun, L. & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics, 225(2), 175-199.
- VanderWeele, T.J. & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
