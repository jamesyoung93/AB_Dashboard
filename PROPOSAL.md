# ACID-Dash: A/B Test Causal Inference Dashboard

**Proposed**: 2026-02-22
**Revised**: 2026-02-22 (R1 -- addressing reviewer feedback)
**Status**: Go with Conditions (R1 conditions resolved)
**Domain**: Causal Inference / Commercial Analytics / Omnichannel Engagement Effectiveness
**Author**: James Young, PhD
**Scope**: Proof-of-concept; local use only; generic demo data (no domain-specific identifiers)

---

## 1. Project Title and Summary

**Title**: ACID-Dash (A/B Test Causal Inference Dashboard)

ACID-Dash is a locally-run, Streamlit-based interactive proof-of-concept dashboard that enables commercial analytics teams to evaluate the causal impact of omnichannel engagement tactics (e.g., email campaigns, webinars, direct mail, sales outreach) on customer outcomes at the ZIP code level. Users upload a CSV containing customer-level or ZIP-aggregated data, assign columns via dropdown menus, select a tactic of interest, and run one or more causal inference methods -- Difference-in-Differences (including staggered adoption), Propensity Score Matching, Inverse Probability Weighting (including doubly robust AIPW), Regression Discontinuity, or Synthetic Control -- to estimate treatment effects with proper uncertainty quantification. The tool outputs publication-quality balance diagnostics, geographic visualizations of effect heterogeneity, sensitivity analyses (E-value, placebo tests), and exportable PDF reports. It runs entirely on a standard corporate laptop (8-16 GB RAM, Windows, no GPU) with no internet dependency after installation.

The target audience is commercial analytics professionals who understand A/B test logic but need a self-service tool for rigorous causal analysis without requiring R, Stata, or SAS expertise. The bundled demo data uses generic customer/omnichannel engagement scenarios with no domain-specific identifiers.

**Positioning**: Unlike code-first libraries (EconML, DoWhy, CausalML) that require Python fluency, ACID-Dash provides a point-and-click GUI with integrated geographic visualization and automated balance diagnostics specifically designed for the CSV-upload workflow of commercial analytics teams. See Section 2.4 for detailed differentiation.

---

## 2. Business and Scientific Motivation

### Why Causal Inference, Not Just Comparison of Means

Commercial analytics routinely evaluates whether an engagement tactic "worked" by comparing mean outcomes (e.g., revenue, conversion rate) between exposed and unexposed customers. This is fundamentally flawed for non-randomized campaigns:

1. **Selection bias**: Customers targeted for high-touch tactics (sales calls, webinars) are pre-selected based on spending potential, firmographics, engagement history, and geography. Comparing treated vs. untreated customers compares high-potential targets to low-potential non-targets, conflating the selection criterion with the treatment effect.

2. **Confounding**: ZIP-level variation in market density, competitor presence, industry mix, and economic conditions drives both tactic deployment and outcomes. Naive comparisons absorb these confounders into the "treatment effect."

3. **Time trends**: Engagement campaigns coincide with product launches, seasonal patterns, and market shifts. Pre-post comparisons without proper controls attribute secular trends to the tactic.

Causal inference methods -- DiD, PSM, IPW, RDD, Synthetic Control -- are designed to address these specific threats to internal validity. They are standard in econometrics and program evaluation but rarely available as self-service tools for commercial analytics teams.

### The ZIP-Code Tactic Evaluation Use Case

The typical workflow:

1. A commercial analytics team has a CSV with one row per customer (or customer-week) containing: customer ID, ZIP code, week/period identifier, one or more tactic indicators (binary or continuous intensity), outcome measures (revenue, units sold, engagement score), and covariates (industry, company size, tenure, prior spend, etc.).

2. Multiple tactics appear in the same dataset (e.g., `channel_email`, `channel_webinar`, `channel_direct_mail`). The analyst selects **one tactic at a time** for evaluation, holding others constant or controlling for them as covariates.

3. The outcome is measured at the customer level but often analyzed or visualized at ZIP granularity for territory planning, resource allocation, and field operations.

4. The analyst needs: (a) a defensible point estimate of the tactic's causal effect, (b) confidence intervals and p-values for leadership presentations, (c) geographic visualization of where the tactic worked best/worst, and (d) a PDF export for the business review.

### Laptop Constraint

This is a proof-of-concept tool that runs on a standard corporate laptop:

- **OS**: Windows 10/11
- **RAM**: 8-16 GB
- **GPU**: None (integrated graphics only)
- **Deployment**: `streamlit run app.py` on localhost; not served to others. POC-level -- results shared via PDF/CSV export, not via hosted app.
- **Network**: No internet dependency after initial `pip install`. No cloud APIs, no external data calls, no telemetry.
- **Python**: Anaconda, Miniconda, or user-level Python install.

### 2.4 Positioning Against Existing Libraries [R1: addresses M9]

A technically savvy colleague will immediately ask: "Why not just use EconML or DoWhy?" The answer:

| Library | What It Does | What ACID-Dash Adds |
|---------|-------------|---------------------|
| **EconML** (Microsoft) | DML, causal forests, DiD, IV estimators with sklearn API | EconML is code-first. No GUI, no geographic visualization, no integrated balance diagnostics, no PDF export. Requires Python fluency. |
| **DoWhy** (PyWhy) | Structured causal inference framework with graph-based identification | DoWhy's API is powerful but complex (DAG specification, refuters). Targets causal inference researchers, not commercial analysts uploading CSVs. |
| **CausalML** (Uber) | Uplift modeling, CATE estimation, meta-learners | Heavy dependency footprint (CatBoost, LightGBM). Targets ML engineers building CATE models, not analysts evaluating individual tactics. |
| **CausalPy** (PyMC Labs) | Bayesian causal inference (DiD, SCM, RDD) | Bayesian-first, requires PyMC/JAX. Targets statisticians, not business analysts. |

**ACID-Dash's differentiators**: (a) point-and-click GUI for non-coders, (b) ZIP-level geographic effect visualization with actionable zone identification, (c) integrated balance diagnostics in the same workflow, (d) treatment threshold slider for continuous treatment binarization, (e) PDF export for business review decks, (f) runs offline on a corporate laptop with zero configuration beyond `pip install`.

---

## 3. Input Data Specification

### Expected CSV Schema

The app does not impose a rigid column naming convention. Instead, users assign semantic roles to columns via dropdown menus in the sidebar. The following roles must be assignable:

| Role | Required? | Expected Type | Example Column Names |
|------|-----------|---------------|---------------------|
| Customer ID | Yes | String or integer | `customer_id`, `account_id`, `client_id`, `user_id` |
| Time period | Yes | Integer, date, or string | `week`, `period`, `month`, `date`, `week_ending` |
| Treatment indicator | Yes | Binary (0/1) or continuous | `channel_email`, `treated`, `campaign_flag`, `touch_count` |
| Outcome variable | Yes | Numeric (continuous) | `revenue`, `units`, `conversion_rate`, `engagement_score` |
| Covariates | Yes (multi-select) | Numeric or categorical | `industry`, `company_size`, `prior_spend`, `tenure`, `segment` |
| Geographic ID | Yes | 5-digit string/integer | `zip`, `zip_code`, `zip5`, `postal_code` |
| Tactic column | Optional | Categorical | `channel`, `tactic_name`, `program_type` |
| Pre-period baseline | Optional | Numeric | `baseline_revenue`, `pre_period_spend` |
| Segment ID | Optional | Categorical | `segment`, `territory`, `region`, `district` |

### Auto-Detection Heuristics

On CSV upload, the app applies heuristic column role detection with confidence scores. All auto-assignments are displayed in the sidebar dropdowns and can be manually overridden:

| Heuristic | Detection Logic | Confidence | Behavior |
|-----------|----------------|------------|----------|
| ZIP code | Column name contains `zip`, `postal`; values are 5-digit integers in range 00501-99950 | High | Auto-assign; show green checkmark |
| Customer ID | Column name contains `customer`, `account`, `client`, `user`, `id`; high cardinality relative to row count | Medium | Auto-suggest; yellow highlight, manual confirm required |
| Week/time | Column name contains `week`, `period`, `month`, `date`; values are sequential integers or parseable dates | High | Auto-assign; show green checkmark |
| Treatment (binary) | Column has exactly 2 unique values {0, 1} or {True, False}; column name contains `treat`, `flag`, `tactic`, `campaign`, `exposed`, `channel` | Medium | Auto-suggest; yellow highlight |
| Treatment (continuous) | Column name contains `touch`, `spend`, `count`, `frequency`, `impression`; values are non-negative numeric with moderate variance | Low | Do not auto-assign; list as candidate |
| Outcome | Column name contains `revenue`, `units`, `conversion`, `score`, `volume`, `lift`, `spend` | Medium | Auto-suggest; yellow highlight |
| Covariate | Remaining numeric or low-cardinality categorical columns not assigned to other roles | Default | Auto-suggest as covariate pool |

[R1: addresses m3 -- removed NPI-specific heuristic; generalized to customer ID patterns]

**Method eligibility validation** [R1: addresses m4]: After column assignment, grey out methods that are ineligible given the data structure:
- DiD: requires >=2 time periods. Greyed out with tooltip if single period detected.
- RDD: requires a user-designated continuous running variable. Greyed out if no continuous columns available.
- Synthetic Control: requires >=5 pre-treatment periods and >=3 treated units. Greyed out with explanation if insufficient.

### Validation Logic

On column assignment, the following validations run automatically with user-visible warnings:

1. **Null check**: Flag columns with >5% missing values; offer imputation options (mean, median, mode, drop) or warn-and-proceed.
2. **Type inference**: Verify numeric columns parse as float; flag string columns mistakenly assigned to numeric roles.
3. **Date parsing**: Attempt `pd.to_datetime()` on time columns; if fails, try integer-as-period interpretation.
4. **Cardinality warnings**: Flag categorical columns with >50 unique values as potential high-cardinality issues (e.g., free-text fields assigned as covariates). Recommend binning or exclusion.
5. **Balance check**: After treatment assignment, warn if treated/control ratio is <10:90 or >90:10.
6. **Duplicate check**: Flag duplicate Customer ID x Time Period rows.
7. **ZIP validation**: Flag non-5-digit values in the geographic ID column.

---

## 4. Treatment Assignment Threshold Module

### Problem

Many treatment columns are continuous rather than binary. In commercial analytics, "treatment" is often operationalized as:
- Number of sales touches in a period: 0, 1, 2, 3, ...
- Marketing spend tier: $0, $500, $2,000, $10,000
- Digital impression count: 0, 5, 20, 100
- Webinar/event attendance: 0 or 1+ events

The user needs to choose a cutoff that binarizes "treated" vs. "control."

### Proposed UI

When the selected treatment column is **continuous** (>2 unique values):

1. **Histogram panel**: Display a histogram of the treatment column values with 30-50 bins.
2. **Draggable threshold line**: A vertical line overlaid on the histogram that the user drags left/right. Default position: the median or the first non-zero value, whichever is more informative.
3. **Live N counter**: Below the histogram, display:
   - `N treated: {count} ({pct}%)`
   - `N control: {count} ({pct}%)`
   - `Ratio: {treated:control}`
4. **Balance preview**: A mini SMD table (top 5 covariates by imbalance) that refreshes on threshold change, giving instant feedback on whether the chosen cutoff creates comparable groups.
5. **Auto-suggest**: Offer 3 suggested cutoffs: median, mean, and first natural break (Jenks or kernel density trough). User can click any suggestion or drag freely.

When the selected treatment column is **already binary** (exactly 2 unique values):
- Threshold slider is hidden.
- N treated/N control displayed directly.
- Balance preview shown.

---

## 5. Causal Inference Method Stack

### A. Difference-in-Differences (DiD)

**What it estimates**: The Average Treatment Effect on the Treated (ATT) -- the change in outcome for treated units relative to what would have happened absent treatment, using the trend in untreated units as the counterfactual.

**Key assumptions**:
1. Parallel trends: Treated and control groups would have followed the same outcome trajectory absent treatment.
2. No anticipation: Treatment does not affect outcomes before the treatment period.
3. SUTVA: No spillover between treated and control units.

**Implementation**:
- **Standard 2x2 DiD**: OLS regression `Y ~ Post + Treated + Post*Treated + Covariates` via `statsmodels.OLS`. The coefficient on `Post*Treated` is the ATT. Clustered standard errors at the customer or ZIP level via `cov_type='cluster'`.
- **Event study (dynamic DiD)**: Interact treatment with period dummies relative to treatment onset: `Y ~ sum_t(beta_t * Treated * 1[t=tau])`. Plot `beta_t` with 95% CIs. Pre-period coefficients should be ~0 if parallel trends holds.
- **Parallel trends visualization**: Line plot of mean outcome by period for treated vs. control, with shaded 95% CI bands. Pre-treatment trends should track visually.

**Output**: ATT point estimate, robust/clustered SE, 95% CI, p-value, parallel trends plot, event study plot.

**Staggered Adoption DiD (Callaway-Sant'Anna)** [R1: addresses M1]:

When treatment is adopted at different times for different units (e.g., an email campaign rolls out across territories over weeks), classical two-way fixed effects (TWFE) DiD is biased due to "negative weighting" of already-treated units as controls for later-treated units (Goodman-Bacon 2021, de Chaisemartin & D'Haultfoeuille 2020). This is the most common scenario in commercial analytics.

- **Implementation**: `csdid` package (Python port of Callaway & Sant'Anna 2021, available on PyPI). Estimates group-time ATTs that avoid negative weighting.
- **UI**: When user indicates staggered treatment timing (checkbox: "Treatment was adopted at different times for different units"), switch from classical TWFE to Callaway-Sant'Anna estimator automatically.
- **Warning**: If staggered adoption is detected (>3 distinct treatment onset periods) but classical DiD is selected, display: "Warning: Treatment appears to be staggered. Classical DiD may be biased. Consider the Callaway-Sant'Anna estimator."
- **Output**: Group-time ATT estimates, aggregated ATT, event study plot with heterogeneity-robust CIs.

**SUTVA note** [R1: addresses m18]: In commercial analytics, spillover is common -- engaging one customer in a business unit can influence behavior of others in the same unit. If the unit of analysis is individual customer but multiple customers share a location or account, SUTVA may be violated. The tool notes this risk and suggests analyzing at the ZIP or account level to reduce spillover.

**Performance**: OLS on 100K rows completes in <2 seconds. Callaway-Sant'Anna is slower (~5-15 seconds on 100K rows) due to group-time estimation.

**Package**: `statsmodels` (OLS with clustered SEs), `linearmodels.PanelOLS` for entity + time FE, `csdid` for staggered adoption.

### B. Propensity Score Matching (PSM)

**What it estimates**: ATT -- comparing outcomes of treated units to outcomes of matched control units with similar propensity scores.

**Key assumptions**:
1. Conditional independence (unconfoundedness): Treatment assignment is independent of potential outcomes conditional on observed covariates.
2. Common support (overlap): Every treated unit has a positive probability of being untreated, and vice versa.
3. Correct propensity model specification.

**Implementation**:
- **Propensity model** [R1: addresses M3]: User selects from two options via dropdown:
  - `sklearn.linear_model.LogisticRegression` (default, simpler, optimizes for prediction of treatment)
  - `sklearn.ensemble.GradientBoostingClassifier` (GBM, recommended when propensity surface is non-linear; standard in HEOR via R's `twang` package). GBM consistently produces better balance than logistic regression when covariate-treatment relationships are non-linear. The interface to `NearestNeighbors` is identical once propensity scores are computed -- this is a near-zero-cost addition.
- **Matching methods** (with or without replacement, user-selectable; default: without replacement) [R1: addresses m8]:
  - Nearest neighbor 1:1 (`sklearn.neighbors.NearestNeighbors` on logit of propensity score, Euclidean or Mahalanobis distance)
  - Nearest neighbor 1:k (k selectable, default k=3, with distance-weighted outcomes)
  - Caliper matching (user-adjustable caliper, default 0.2 SD of logit propensity score; unmatched treated units reported)
- **Balance diagnostics**:
  - Standardized mean differences (SMD) for all covariates, before and after matching
  - Love plot (horizontal dot plot: SMD pre-match in red, SMD post-match in blue, threshold lines at |0.1| and |0.25|)
  - Propensity score overlap plot (overlapping histograms/KDEs for treated and control)
- **ATT estimation**: Mean difference in outcomes between matched treated and control units, with bootstrap SEs (500 replicates by default, adjustable). [R1: addresses m7 -- Abadie-Imbens SEs removed as no standard Python implementation exists; bootstrap is the default. Abadie-Imbens is a v2 stretch goal.]

**Output**: ATT estimate, SE, 95% CI, p-value, matched N treated / N control, number of unmatched units, propensity score overlap plot, Love plot, SMD table.

**Performance**: `NearestNeighbors` with ball_tree algorithm is O(N log N). 100K rows in <5 seconds. No iterative loop matching.

**Package**: `scikit-learn` (LogisticRegression, NearestNeighbors), `scipy.stats` for tests.

### C. Inverse Probability Weighting (IPW)

**What it estimates**: ATE (Average Treatment Effect) and ATT, using propensity-score-derived weights to create a pseudo-population where treatment is independent of covariates.

**Key assumptions**:
1. Conditional independence (same as PSM).
2. Positivity: 0 < P(T=1|X) < 1 for all X.
3. Correct propensity model specification.

**Implementation**:
- **Propensity scores**: Same propensity model as PSM (LogisticRegression or GBM, shared via `modules/propensity.py`).
- **Weight computation** [R1: addresses m8 -- exact formulas for both ATE and ATT]:
  - ATE weights: `w_i = T_i/e_i + (1-T_i)/(1-e_i)`
  - ATE stabilized: `w_i = T_i * P(T=1)/e_i + (1-T_i) * P(T=0)/(1-e_i)`
  - ATT weights: `w_i = T_i + (1-T_i) * e_i/(1-e_i)`
  - ATT stabilized: `w_i = T_i + (1-T_i) * (e_i/(1-e_i)) * (P(T=0)/P(T=1))`
- **Positivity check** [R1: addresses reviewer m11]: Report min/max propensity scores. Warn if any are <0.01 or >0.99 (near-violations of positivity that trimming alone may not address).
- **Trimming**: User-adjustable percentile cutoff (default: trim weights above 99th percentile). Display weight distribution histogram with trimming threshold. Report number of trimmed observations.
- **Weighted outcome regression**: `statsmodels.WLS` with IPW weights. Robust (sandwich) SEs.

**Doubly Robust Estimation (AIPW)** [R1: addresses M2]:

Augmented Inverse Probability Weighting combines outcome modeling with propensity weighting. AIPW is consistent if *either* the propensity model or the outcome model is correctly specified (doubly robust property). Increasingly recommended by ISPOR guidance for observational studies.

- **Implementation**: Fit outcome model (OLS on covariates, separately for treated and control), compute residuals, apply IPW weights to residualized outcomes. ~30-50 additional LOC on top of existing IPW module.
- **Output**: AIPW ATE and ATT estimates with robust SEs, alongside standard IPW estimates for comparison.

**Output**: ATE and ATT estimates (IPW and AIPW), SEs, 95% CIs, p-values, weight distribution histogram, effective sample size (`(sum w)^2 / sum w^2`), number of trimmed observations, positivity diagnostics.

**Performance**: Weight computation is vectorized. <1 second on 100K rows. AIPW adds <1 second for the outcome model.

**Package**: `scikit-learn` (propensity), `statsmodels` (WLS, OLS), `numpy`.

### D. Regression Discontinuity Design (RDD)

**When to use**: When treatment assignment is based on a continuous running variable with a known cutoff (e.g., customers above an engagement score threshold receive a tactic; tier-based targeting where customers above tier 3 are contacted). Gated behind "Advanced Methods" toggle alongside Synthetic Control [R1: addresses reviewer R2 recommendation].

**What it estimates**: For sharp RDD (treatment deterministically assigned at cutoff): Average Treatment Effect at the cutoff. For fuzzy RDD (treatment probability jumps at cutoff but is not 100%): Local Average Treatment Effect (LATE) via Wald/2SLS estimator. [R1: addresses m9 -- sharp vs. fuzzy distinction added. UI asks: "Is treatment assigned deterministically by the running variable?"]

**Key assumptions**:
1. Continuity of potential outcomes at the cutoff (no jump except due to treatment).
2. No precise manipulation of the running variable (units cannot sort themselves to one side of the cutoff).

**Implementation**:
- **Core estimation**: `rdrobust` package (Python port of Calonico, Cattaneo & Titiunik). Local polynomial regression (default: linear) with optimal bandwidth selection (CCT bandwidth as default; IK available as robustness check but not presented as equally valid -- CCT supersedes IK). [R1: addresses reviewer m10]
- **Manual bandwidth override**: Slider for user-selected bandwidth with sensitivity display.
- **McCrary density test**: Histogram of running variable at cutoff to check for manipulation. Visual inspection + formal test (local polynomial density estimation).
- **Visualization**: Scatter plot of outcome vs. running variable with fitted lines on each side of the cutoff, discontinuity highlighted.

**Output**: LATE estimate at cutoff, robust bias-corrected CI, effective N on each side of cutoff, bandwidth used, McCrary density plot, RD plot.

**Performance**: `rdrobust` is fast on moderate datasets. <5 seconds on 100K rows.

**Package**: `rdrobust` (v1.3.0, pip-installable, Windows-compatible).

### E. Synthetic Control

**When to use**: When few treated units exist (e.g., 3-10 treated ZIP codes or territories out of hundreds) and a large donor pool of untreated units is available. Classic use case: evaluating a pilot program launched in a handful of geographies.

**What it estimates**: The treatment effect for treated units by constructing a weighted combination of untreated units that best reproduces the pre-treatment outcome trajectory of the treated unit(s).

**Key assumptions**:
1. No interference between treated and donor units.
2. The treated unit's counterfactual lies in the convex hull of the donor pool.
3. Sufficient pre-treatment periods to fit the synthetic control.

**Implementation**:
- **Core estimation**: `pysyncon` (v1.5.2, actively maintained). Classic Abadie-Diamond-Hainmueller SCM + Penalized SCM (Abadie & L'Hour 2021).
- **Donor pool selection**: User selects which untreated units to include (default: all untreated). Cap at 200 donors by default with user override.
- **Visualizations**:
  - Treated vs. synthetic control time series
  - Gap plot (treated minus synthetic over time)
  - Placebo tests (in-space: re-run SCM treating each donor as if treated; plot distribution of placebo gaps)
- **Progress bar**: SCM optimization can be slow with large donor pools. Display progress bar with estimated time remaining.
- **Warning**: For >200 donors or >50 pre-treatment periods, display a warning about computation time and offer to subsample.

**Output**: Treatment effect time series, gap plot, donor weights table, MSPE ratio (treated/median placebo), placebo test p-value, synthetic vs. actual pre-treatment fit (RMSPE).

**Performance**: Computationally heavier than other methods. 100 donors x 20 periods: ~10-30 seconds. 500 donors: ~2-5 minutes. Gated behind an "Advanced Methods" toggle with a performance warning.

**Package**: `pysyncon` (v1.5.2). Depends on `scipy` and `cvxpy`. `cvxpy` installs cleanly via pip on Windows with pre-built wheels from conda-forge if needed.

### F. Covariate-Adjusted DiD (ANCOVA / DiD + OLS)

**When to use**: When parallel trends is questionable and the analyst wants to control for baseline covariates directly in the DiD specification.

**What it estimates**: ATT conditional on covariates.

**Implementation**:
- OLS regression: `Y_post - Y_pre ~ Treated + X_covariates` (ANCOVA form), or equivalently `Y ~ Post + Treated + Post*Treated + X + Post*X` (fully interacted).
- Clustered SEs at the customer or ZIP level.
- Coefficient table with all covariates displayed.

**Output**: ATT estimate, SE, 95% CI, p-value, covariate coefficient table, R-squared.

**Performance**: OLS. <2 seconds on 100K rows.

**Package**: `statsmodels` (OLS with `cov_type='cluster'`).

### G. Sensitivity / Robustness Tab

A dedicated tab collecting robustness checks that apply across methods:

1. **E-value (primary sensitivity measure)** [R1: addresses M5]: The E-value (VanderWeele & Ding 2017) quantifies the minimum strength of association that an unmeasured confounder would need to have with both the treatment and the outcome to explain away the observed effect. It is a closed-form formula based on the point estimate and confidence interval -- simple to compute, intuitive to interpret, and increasingly standard in observational studies.
   - **Output**: E-value for point estimate and for confidence interval lower bound. Interpretation: "An unmeasured confounder would need to be associated with both the treatment and the outcome by a risk ratio of at least {E-value} to explain away this effect."
   - **Implementation**: ~20 LOC. Formula: `E = RR + sqrt(RR * (RR - 1))` where RR is the risk ratio (or approximated from the standardized effect size).

2. **Rosenbaum bounds (v2 stretch goal)**: Rosenbaum bounds for PSM provide a complementary sensitivity analysis but have no established Python implementation. Deferred to v2. If implemented: compute the range of Gamma at which the treatment effect estimate would become insignificant, following Rosenbaum (2002).

3. **Placebo outcome test**: User selects a column that *should not* be affected by treatment (e.g., a pre-period outcome, or a demographic variable). Run the selected causal method on this placebo outcome. A significant result suggests confounding or model misspecification. **Warning**: The test is only valid if the placebo outcome is truly unaffected by treatment. [R1: addresses reviewer m16]

3. **Pre-period falsification (placebo time)**: For DiD, run the model on pre-treatment periods only (defining an artificial "treatment" at a mid-pre-period point). A significant pre-period "effect" indicates violation of parallel trends.

4. **Leave-one-out sensitivity (for Synthetic Control)**: Re-run SCM dropping each donor unit one at a time. Display range of estimates.

5. **Caliper sensitivity (for PSM)**: Display ATT estimates across a range of caliper values (0.05, 0.1, 0.2, 0.5 SD). Stable estimates across calipers indicate robustness.

---

## 6. Balance Diagnostics Module

A standalone tab (also accessible inline during PSM/IPW workflows) providing:

### 6.1 SMD Table

| Covariate | Mean (Treated) | Mean (Control) | SMD (Raw) | SMD (Matched/Weighted) | Status |
|-----------|---------------|----------------|-----------|----------------------|--------|
| prior_spend | 45.2 | 31.7 | 0.42 | 0.03 | Pass |
| tenure | 8.1 | 7.9 | 0.05 | 0.04 | Pass |
| industry_tech | 0.35 | 0.12 | 0.55 | 0.08 | Pass |

Threshold: |SMD| < 0.1 = "Pass" (green), 0.1-0.25 = "Caution" (yellow), > 0.25 = "Fail" (red).

**SMD computation for categorical covariates** [R1: addresses m12]: For binary dummy variables from categorical covariates, SMD is computed as `(p_treated - p_control) / sqrt((p_treated*(1-p_treated) + p_control*(1-p_control))/2)`. Each level of a dummy-coded variable gets its own SMD row in the table.

### 6.2 Love Plot

Horizontal dot plot. X-axis: absolute SMD. Y-axis: covariates (sorted by raw SMD descending). Two dots per covariate: red (raw) and blue (after matching/weighting). Vertical dashed lines at 0.1 and 0.25.

### 6.3 Covariate Distribution Overlays

For each covariate: overlapping KDE plots (continuous) or grouped bar charts (categorical) for treated vs. control, before and after matching/weighting. Selectable via dropdown.

### 6.4 Propensity Score Overlap

Overlapping histograms (or KDE) of propensity scores for treated and control groups. Shaded region indicates common support. Report % of treated units outside common support.

### 6.5 Dynamic Updates

All diagnostics refresh automatically when:
- The treatment threshold slider moves
- Matching parameters change (caliper, k neighbors)
- IPW trimming percentile changes
- Covariates are added/removed from the propensity model

---

## 7. Geographic Visualization Module

### 7.1 Geographic Effect Display [R1: addresses B1 -- GeoJSON blocker resolved]

**The full US ZCTA GeoJSON (~850 MB uncompressed) will crash or hang on an 8 GB laptop.** The revised approach avoids polygon choropleth as the default:

- **Default view: Scatter at ZIP centroids** (`plotly.graph_objects.Scattergeo`). Plot colored circles at ZIP centroids from the bundled `zip_latlon.csv`. This renders in <2 seconds for any dataset size, requires no GeoJSON file, and works fully offline (no tile server). Circle size encodes sample count; color encodes effect magnitude.
- **Optional: Filtered polygon choropleth** (for datasets with <500 unique ZIPs). If the user's data covers a limited geography (e.g., one state or region), load only those ZIP polygons from a pre-split per-state GeoJSON bundle. This provides true polygon rendering without the memory cost of national-scale GeoJSON.
- **Input**: ZIP code column (5-digit US ZIP)
- **Aggregation**: Effect size or outcome lift computed per ZIP (mean effect for customers within each ZIP, or direct ZIP-level estimate if data is already ZIP-aggregated)
- **Color scale**: Diverging (green = positive lift, red = negative lift, grey = neutral/no data). Midpoint at 0 or user-defined reference.
- **Library**: `plotly.graph_objects.Scattergeo` (vector-based, no tile server, fully offline) [R1: addresses m13 -- Scattergeo instead of scatter_mapbox for offline use]

### 7.2 Overlay Layers

- **Bubble layer**: Tactic intensity (touch count, spend) as bubble size at ZIP centroid. Larger bubble = more tactic investment.
- **Treatment assignment**: Color-coded dots for treated (blue) vs. control (orange) ZIPs.

### 7.3 Actionable Zones

Computed from the intersection of tactic intensity and estimated treatment effect:

| Zone | Tactic Intensity | Estimated Lift | Interpretation | Color |
|------|-----------------|----------------|----------------|-------|
| **Opportunity** | Low | High (predicted) | Underinvested; increase coverage | Gold star overlay |
| **Optimized** | High | High | Tactics working well; maintain | Green |
| **Diminishing returns** | High | Low or negative | Overinvested or wrong tactic | Red hatching |
| **Cold** | Low | Low | Low potential or wrong tactic | Grey |

**Opportunity zone predictions** [R1: addresses M10]: Predicted lift for untreated/lightly-treated ZIPs is model extrapolation to out-of-sample units -- fundamentally harder and less reliable than estimating effects for treated units. The overlap assumption is violated by definition for ZIPs that received zero tactic investment.

- **Default**: Display only observed effects (Optimized / Diminishing Returns / Cold zones). Opportunity zone predictions are **hidden by default**, gated behind an explicit toggle: "Show model-predicted opportunity zones (experimental)."
- **Disclaimer** (displayed when toggle is on): "Predicted effects for untreated ZIPs are model extrapolations with high uncertainty. They should inform hypothesis generation, not resource allocation decisions."
- **Uncertainty**: Display prediction intervals (not confidence intervals) that account for both estimation uncertainty and prediction uncertainty.
- **Caveat label on all predicted points**: "Model-predicted, not observed."

### 7.4 Filters

- By tactic (if tactic column provided)
- By time period
- By treatment assignment status
- By segment/territory/region
- By effect size range (slider)

### 7.5 ZIP-to-Lat/Lon Mapping

Embed a lightweight lookup table (`data/zip_latlon.csv`) mapping 5-digit ZIP codes to latitude/longitude centroids, state FIPS codes, and state abbreviations. Source: US Census Bureau ZCTA (ZIP Code Tabulation Area) centroid file (public domain, ~33K rows, ~1 MB). Bundled with the app -- no runtime API call. State fields support per-state filtering in the geographic view. [R1: addresses reviewer m15]

### 7.6 Library Decision

**Decision: Plotly `Scattergeo` only.** [R1: revised from Plotly choropleth + Folium fallback]

Folium is dropped from scope. Folium's tile layer requires internet for tile download, and its HTML-only export is incompatible with PDF report generation. Plotly's `Scattergeo` provides vector-based mapping that works fully offline, exports to PNG/SVG via kaleido, and integrates natively with Streamlit via `st.plotly_chart`. For the POC use case (localhost only, results shared via export), this is sufficient.

---

## 8. User Interface Architecture

### 8.1 Framework Evaluation

| Criterion | Streamlit | Self-Contained HTML |
|-----------|-----------|-------------------|
| Setup complexity | `pip install streamlit` (no admin) | Zero install (double-click HTML file) |
| Python ecosystem access | Full (statsmodels, sklearn, rdrobust, pysyncon) | None (must reimplement all stats in JS) |
| Interactivity | Native widgets (sliders, dropdowns, file upload) | Custom JS (significant development overhead) |
| State management | `st.session_state` | Manual (localStorage or in-memory) |
| Export capabilities | PDF via `fpdf2`/`reportlab`; CSV via `pandas`; PNG via matplotlib/plotly | Limited (browser print-to-PDF only) |
| Computation | Full Python numerical stack | Browser JS (limited for large datasets) |
| Maintenance | Update via `pip install --upgrade` | Manual file replacement |
| Sharing | `streamlit run app.py` on localhost; shareable via screen share or LAN | Email the HTML file; anyone can open |

**Decision: Streamlit, localhost only, POC scope.** [R1: addresses R1, simplifies deployment]

The causal inference method stack (statsmodels, sklearn, rdrobust, pysyncon) cannot be replicated in client-side JavaScript. Streamlit is the only viable framework.

- `streamlit run app.py` serves on `localhost:8501` with no network exposure.
- Launch flags for corporate environments: `--server.address 127.0.0.1 --server.headless true --server.fileWatcherType none` (avoids port binding issues, OneDrive/watchdog conflicts). [R1: addresses reviewer R1 note]
- Results are shared via PDF/CSV export, not via hosted app. This is a personal/team proof-of-concept tool.
- PyInstaller packaging is deferred to post-v1 if there is demand. [R1: addresses m14]

### 8.2 Layout

```
+-------------------------------+----------------------------------------------+
|         SIDEBAR               |                MAIN PANEL                    |
|                               |                                              |
| [Upload CSV]                  |  [Overview] [Balance] [Results] [Geo] [Export]|
|                               |                                              |
| --- Column Assignment ---     |  Tab content changes based on selection:     |
| Customer ID: [dropdown]       |                                              |
| Time Period: [dropdown]       |  OVERVIEW TAB:                               |
| Treatment:   [dropdown]       |  - Data summary (N, periods, treated/ctrl)   |
| Outcome:     [dropdown]       |  - Outcome distribution histograms           |
| Covariates:  [multi-select]   |  - Threshold slider (if continuous trt)      |
| ZIP Code:    [dropdown]       |  - Parallel trends preview                   |
| Tactic:      [dropdown]       |                                              |
|                               |  BALANCE TAB:                                |
| --- Analysis Config ---       |  - SMD table                                 |
| Tactic Filter: [dropdown]     |  - Love plot                                 |
| Method: [checkbox group]      |  - Distribution overlays                     |
|   [x] DiD                     |  - Propensity overlap                        |
|   [x] PSM                     |                                              |
|   [ ] IPW                     |  RESULTS TAB:                                |
|   [ ] RDD                     |  - Per-method results panels                 |
|   [ ] Synthetic Control       |  - Forest plot comparing methods             |
|   [ ] Covariate-Adj DiD       |  - Sensitivity/robustness subtab             |
|                               |                                              |
| --- Method Parameters ---     |  GEO TAB:                                    |
| (context-sensitive panel       |  - Choropleth of effect by ZIP              |
|  based on selected method)     |  - Bubble overlay (tactic intensity)        |
|                               |  - Opportunity/diminishing return zones      |
| [Run Analysis]                |                                              |
|                               |  EXPORT TAB:                                 |
|                               |  - Download matched sample CSV               |
|                               |  - Download all plots (PNG/SVG zip)          |
|                               |  - Download PDF summary report               |
+-------------------------------+----------------------------------------------+
```

### 8.3 Export Capabilities

| Export | Format | Contents |
|--------|--------|----------|
| Matched/weighted sample | CSV | Full dataset with match indicators, propensity scores, IPW weights, treatment assignment |
| Individual plots | PNG (300 DPI) or SVG | Any displayed plot, downloadable via button |
| All plots archive | ZIP | All generated plots from current analysis session |
| PDF summary report | PDF (via `fpdf2`) | Executive summary: method, sample sizes, ATT/ATE estimates, key plots (balance, results, geo map), interpretation notes |

---

## 9. Performance Constraints and Optimization Plan

**Target**: CSV up to 100K rows, ~20 columns, full analysis pipeline on 8 GB RAM laptop in under 60 seconds.

### Bottleneck Analysis and Mitigations

| Component | Estimated Time (100K rows) | Bottleneck | Mitigation |
|-----------|---------------------------|------------|------------|
| CSV load + validation | 1-3s | I/O | `pd.read_csv` with dtype hints; lazy validation (validate only assigned columns) |
| Propensity model | 1-2s | CPU | `LogisticRegression(solver='lbfgs', max_iter=200)` |
| PSM matching (1:1, 100K) | 2-5s | CPU/Memory | `NearestNeighbors(algorithm='ball_tree')` on logit scores (1D) |
| DiD OLS | <1s | N/A | `statsmodels.OLS` is highly optimized |
| IPW weighting | <1s | N/A | Vectorized numpy operations |
| RDD | 2-5s | CPU | `rdrobust` is compiled |
| Synthetic Control | 10-300s | Optimization | Cap donor pool at 200; progress bar; optional subsampling |
| Geographic rendering | 1-3s | Plotly Scattergeo | Pre-aggregate to ZIP centroids; no GeoJSON needed [R1: B1 resolved] |
| Balance diagnostics | 1-2s | CPU | Vectorized SMD computation |
| PDF export | 2-5s | I/O | Generate plots at screen resolution; embed as images |

### Optimization Strategies

1. **Lazy computation**: Only run the user-selected method(s). Do not pre-compute all methods.
2. **Caching**: `@st.cache_data` on CSV load, propensity model fit, and matching results. Cache invalidates on column reassignment or parameter change.
3. **Sampling option**: For datasets >100K rows, offer optional random sampling (user-selectable N, default 50K) with a warning banner: "Results based on {N}-row sample. Full dataset analysis may be slow."
4. **Progress indicators**: `st.progress()` bar for Synthetic Control and large PSM runs.
5. **Memory management**: Release intermediate DataFrames after matching. Use `float32` instead of `float64` where precision is not critical. Avoid full-dataset copies.
6. **Geographic pre-aggregation**: Aggregate customer-level effects to ZIP centroids before rendering. Scattergeo at centroids, not polygon choropleth. [R1: B1 resolved]
7. **Stratified sampling** [R1: addresses reviewer suggestion]: For large datasets, offer stratified sampling by treatment status and key covariates (default) rather than random sampling, to preserve treated/control ratio and covariate balance.

---

## 10. Dependency Stack

All packages are pip-installable without admin rights, have no system-level dependencies (no GDAL, no R, no compiled C that requires Visual Studio Build Tools beyond what wheels provide), and are stable on Windows.

### Core Dependencies

| Package | Version Constraint | Purpose | Windows Status |
|---------|-------------------|---------|----------------|
| `streamlit` | >=1.30,<2.0 | Application framework | Stable |
| `pandas` | >=2.0,<3.0 | Data manipulation | Stable |
| `numpy` | >=2.0,<3.0 | Numerical computation | Stable [R1: B2 fix -- numpy<2.0 incompatible with current scipy/sklearn] |
| `scipy` | >=1.11,<2.0 | Statistical tests, optimization | Stable |
| `statsmodels` | >=0.14,<1.0 | OLS, WLS, clustered SEs, ANCOVA | Stable |
| `scikit-learn` | >=1.3,<2.0 | LogisticRegression, NearestNeighbors | Stable |
| `plotly` | >=5.18,<6.0 | Interactive plots, choropleth maps | Stable |
| `matplotlib` | >=3.7,<4.0 | Static plots (Love plot, event study) | Stable |
| `seaborn` | >=0.13,<1.0 | KDE overlays, distribution plots | Stable |

### Specialized Dependencies

| Package | Version Constraint | Purpose | Windows Status |
|---------|-------------------|---------|----------------|
| `rdrobust` | >=1.3 | Regression discontinuity estimation | Stable (pure Python + scipy) |
| `pysyncon` | >=1.5 | Synthetic control method | Stable (depends on cvxpy) |
| `cvxpy` | >=1.4 | Convex optimization (pysyncon dependency) | Stable via pip wheels |
| `linearmodels` | >=6.0,<8.0 | Panel OLS with FE (optional, for multi-period DiD) | Stable [R1: m15 -- version updated] |
| `csdid` | >=0.1 | Callaway-Sant'Anna staggered DiD | Stable [R1: M1 -- staggered adoption support] |

### Utility Dependencies

| Package | Version Constraint | Purpose | Windows Status |
|---------|-------------------|---------|----------------|
| `fpdf2` | >=2.7 | PDF report generation | Stable (pure Python) |
| `kaleido` | >=0.2,<1.0 | Plotly static image export (PNG/SVG) | Stable [R1: M4 fix -- kaleido 1.0 incompatible with Plotly 5.x and requires Chrome] |
| `openpyxl` | >=3.1 | Excel export (optional) | Stable |
| `joblib` | >=1.3 | Parallel computation (bootstrap SEs) | Stable |

### Explicitly Excluded

| Package | Reason |
|---------|--------|
| `geopandas` | Depends on GDAL/Fiona; notoriously difficult to install on Windows without conda |
| `causalml` | Heavy dependency footprint (CatBoost, LightGBM); overkill for this use case. See Section 2.4 for positioning. |
| `dowhy` | Powerful but complex API; adds conceptual overhead for the target audience. See Section 2.4 for positioning. |
| `folium` / `streamlit-folium` | Tile maps require internet; dropped from scope in favor of Plotly Scattergeo [R1] |
| `jenkspy` | Natural breaks algorithm; replaced with scipy-based KDE trough detection to avoid extra dependency [R1: m5] |

---

## 11. Project Structure

```
acid_dash/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Pinned dependencies
├── README.md                       # Setup and usage instructions
├── PROPOSAL.md                     # This document
├── LOGBOOK.md                      # Development log
├── .gitignore
│
├── data/
│   ├── zip_latlon.csv              # Bundled ZIP-to-centroid lookup (~33K rows, Census ZCTA + state FIPS)
│   └── sample/
│       ├── synthetic_omnichannel.csv   # Main demo: DiD/PSM/IPW validation [R1: generic, no domain IDs]
│       ├── synthetic_rdd.csv           # RDD validation: engagement score cutoff [R1: M6]
│       └── synthetic_scm.csv           # Synthetic Control validation: few treated ZIPs [R1: M6]
│
├── modules/
│   ├── __init__.py
│   ├── column_detector.py          # Auto-detection heuristics for column roles
│   ├── threshold.py                # Treatment threshold slider + binarization logic
│   ├── propensity.py               # Shared propensity score estimation (LR + GBM) [R1: reviewer suggestion]
│   ├── did.py                      # Difference-in-Differences (standard + event study + staggered + ANCOVA)
│   ├── psm.py                      # Propensity Score Matching (1:1, 1:k, caliper)
│   ├── ipw.py                      # Inverse Probability Weighting (stabilized, trimmed, AIPW)
│   ├── rdd.py                      # Regression Discontinuity Design (rdrobust wrapper)
│   ├── synth.py                    # Synthetic Control (pysyncon wrapper)
│   ├── balance.py                  # Balance diagnostics (SMD, Love plot, overlap)
│   ├── sensitivity.py              # Robustness checks (E-value, placebo tests, caliper sensitivity)
│   └── geo.py                      # Geographic visualization (Plotly Scattergeo, zones)
│
├── utils/
│   ├── __init__.py
│   ├── validators.py               # CSV validation (nulls, types, cardinality, ZIP format)
│   └── exporters.py                # Report generation (PDF, CSV, PNG/SVG zip)
│
└── tests/
    ├── __init__.py
    ├── test_column_detector.py     # Unit tests for auto-detection
    ├── test_did.py                 # DiD correctness tests against known results
    ├── test_psm.py                 # PSM correctness tests
    ├── test_ipw.py                 # IPW correctness tests
    ├── test_balance.py             # Balance diagnostic computation tests
    ├── test_validators.py          # Validation logic tests
    └── test_with_sample_data.py    # Integration test: full pipeline on synthetic data
```

---

## 12. Sample Data Specification [R1: complete rewrite -- generic omnichannel, no domain-specific IDs]

Three synthetic datasets are generated to validate the full method stack. All use generic customer/omnichannel engagement framing with no domain-specific identifiers (no NPI, no HCP, no Rx, no medical terminology). This makes the demo shareable without data governance concerns.

### 12.1 Main Dataset: `synthetic_omnichannel.csv` (DiD / PSM / IPW / AIPW validation)

**Dimensions**: 10,000 rows x 14 columns

| Column | Type | Description | Generation Logic |
|--------|------|-------------|-----------------|
| `customer_id` | int | Unique customer identifier | Sequential integers 1-500 (500 unique customers) |
| `zip_code` | str | 5-digit US ZIP | Sampled from 200 real US ZIP codes (stratified across 4 Census regions) |
| `week` | int | Week number (1-20) | 20 weeks per customer; weeks 1-10 = pre-period, 11-20 = post-period |
| `channel_email` | int (0/1) | Binary treatment: received email campaign | 40% treated, assigned at customer level with confounding on company_size and industry |
| `channel_webinar` | float | Continuous treatment: webinar attendance count | Log-normal(mu=1.2, sigma=0.8), correlated with channel_email (r~0.3); has own causal effect on outcome (beta=0.8 per attendance) |
| `channel_direct_mail` | int (0/1) | Second binary tactic (for multi-tactic demo) | 25% treated, weakly correlated with email (r~0.15); no causal effect (null tactic for placebo test validation) |
| `revenue` | float | **Primary outcome** ($/week) | DGP: `beta_0(50) + beta_1(5)*treated + beta_2(2)*post + beta_3(3.5)*treated*post + X*gamma + epsilon(N(0,8))`; **true ATT = 3.5 $/week** |
| `units_sold` | float | Secondary outcome | Correlated with revenue (r~0.7); true ATT = 1.2 units/week |
| `prior_spend` | float | Pre-period total spend (covariate) | Normal(500, 150), clipped >= 50; confounded with treatment (higher spenders more likely treated) |
| `tenure_years` | float | Years as customer (covariate) | Uniform(0.5, 15) |
| `company_size` | str | Firmographic (covariate) | Categorical: {Small: 30%, Medium: 30%, Large: 25%, Enterprise: 15%}; Enterprise customers 3x more likely to be treated |
| `industry` | str | Customer industry (covariate) | Categorical: {Technology: 25%, Finance: 20%, Healthcare: 20%, Manufacturing: 20%, Other: 15%}; Technology customers 2x more likely to be treated |
| `engagement_score` | float | Pre-period engagement index (covariate; also usable as RDD running variable) | Normal(50, 15), clipped [0, 100]; correlated with treatment (r~0.4) |
| `region` | str | Census region | Categorical: {Northeast: 25%, South: 30%, Midwest: 20%, West: 25%} |

**Data Generating Process (DGP)** [R1: addresses m17 -- all parameters specified]:
- `beta_0 = 50` (baseline revenue)
- `beta_1 = 5` (treated group higher baseline -- confounding)
- `beta_2 = 2` (common time trend)
- `beta_3 = 3.5` (**true ATT for channel_email on revenue**)
- `gamma_company_size = {Small: 0, Medium: 3, Large: 8, Enterprise: 15}` (confounding on company size)
- `gamma_industry = {Technology: 5, Finance: 3, Healthcare: 2, Manufacturing: 0, Other: -2}` (confounding on industry)
- `gamma_prior_spend = 0.02` (per dollar of prior spend)
- `gamma_engagement = 0.1` (per engagement score point)
- `epsilon ~ N(0, 8)` (idiosyncratic noise)
- Confounding: Enterprise and Technology customers are more likely to be treated AND have higher baseline revenue
- Parallel trends: Pre-period trends are parallel by construction (common time trend + group fixed effects)
- `channel_webinar` has own causal effect (beta=0.8 per attendance); useful for testing threshold module
- `channel_direct_mail` has **no causal effect** (null tactic); useful for placebo method validation

**Known ground truths for validation**:
- DiD/PSM/IPW/AIPW should recover ATT ~ 3.5 +/- SE for `channel_email` on `revenue`
- PSM/IPW should recover ~ 0.0 for `channel_direct_mail` (null tactic)
- True ATT falls within 95% CI = implementation is correct [R1: addresses m16]

### 12.2 RDD Dataset: `synthetic_rdd.csv` [R1: addresses M6]

**Dimensions**: 5,000 rows x 6 columns

| Column | Type | Description | Generation Logic |
|--------|------|-------------|-----------------|
| `customer_id` | int | Unique customer ID | 5,000 unique |
| `engagement_score` | float | Running variable | Uniform(20, 80) |
| `treated` | int (0/1) | Sharp RDD: treated if `engagement_score >= 50` | Deterministic cutoff |
| `revenue_post` | float | Outcome | Continuous at cutoff + **true LATE = 4.0** at cutoff; `revenue = 30 + 0.5*score + 4.0*(score >= 50) + N(0, 5)` |
| `company_size` | str | Covariate | Same distribution as main dataset |
| `region` | str | Covariate | Same distribution as main dataset |

**Known ground truth**: LATE at cutoff = 4.0. `rdrobust` should recover this within the 95% CI.

### 12.3 Synthetic Control Dataset: `synthetic_scm.csv` [R1: addresses M6]

**Dimensions**: 4,000 rows (200 ZIPs x 20 weeks) x 5 columns

| Column | Type | Description | Generation Logic |
|--------|------|-------------|-----------------|
| `zip_code` | str | 200 unique ZIP codes | Sampled from real US ZIPs |
| `week` | int | Week 1-20 | Weeks 1-10 = pre-treatment, 11-20 = post-treatment |
| `treated` | int (0/1) | 5 treated ZIPs out of 200 | Fixed at ZIP level |
| `revenue` | float | ZIP-level aggregate outcome | Common factor model: `revenue = mu_i + lambda_t + delta*treated*post + epsilon`; **true treatment effect = 6.0** |
| `market_size` | float | ZIP-level covariate | Log-normal, time-invariant |

**Known ground truth**: Treatment effect = 6.0 for treated ZIPs post-treatment. Synthetic control should recover this within the placebo test distribution.

### Generator Script

`tests/generate_sample_data.py` with seed=42 for all three datasets. Single script, three output files.

---

## 13. Resolved Design Decisions [R1: updated per reviewer recommendations]

The following items were flagged as open questions in the initial proposal. Reviewer recommendations have been accepted and incorporated.

### R1. Streamlit Deployment -- RESOLVED

**Decision**: Streamlit on localhost, POC scope. No HTML fallback. PyInstaller deferred to post-v1. Launch with `--server.address 127.0.0.1 --server.headless true --server.fileWatcherType none`.

### R2. Advanced Methods Gating -- RESOLVED

**Decision**: Both Synthetic Control **and** RDD are gated behind a collapsible "Advanced Methods" section in the sidebar. Default: collapsed/unchecked. Warning text includes minimum data requirements (SCM: >=5 pre-periods, few treated units; RDD: requires continuous running variable with known cutoff).

### R3. Geographic Visualization Placement -- RESOLVED

**Decision**: Separate Geo tab with method-selector dropdown. User selects which method's estimates to map. Optionally: "Compare Methods" small multiples grid if multiple methods have been run.

### R4. Data Governance -- NOT APPLICABLE

This tool operates at generic customer level (not patient level). The demo data contains no domain-specific identifiers (no NPI, no PHI, no medical terminology). The tool runs on localhost, processes data in local RAM only, and does not persist uploads. HIPAA is not relevant. No data classification checkbox needed.

### R5. Panel Data Support -- RESOLVED

**Decision**: Support both pre/post and multi-period data. DiD event study handles arbitrary periods. For PSM and IPW with multi-period data [R1: addresses reviewer R5 note]:
- Match on baseline (pre-period) covariates
- Default: compare average post-period outcome between matched groups
- Drill-down: per-period ATT estimates available as secondary output

### R6. Bayesian A/B Test -- EXCLUDED FROM V1 [R1: addresses M7]

**Decision**: Excluded. The reviewer correctly identified that including a non-causal method in a causal inference dashboard undermines the tool's core message. Users who do not read caveats will run the Bayesian comparison, get P(superiority) = 95%, and declare the tactic effective -- exactly the naive analysis ACID-Dash is designed to prevent. Revisit for v2 only if stakeholders specifically request it, with anti-misuse safeguards.

### Remaining Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | Staggered DiD (`csdid` Python package) less mature than R version | M | M | Fall back to classical TWFE with warning if `csdid` fails; validate against known DGP |
| 2 | `pysyncon` optimization convergence issues on edge cases | L | L | Cap donor pool; timeout with user-visible message |
| 3 | Time allocation away from CBRE/ECOCHEMICAL | M | M | Schedule ACID-Dash in non-sprint windows; Phase 1 before CBRE sprint |
| 4 | Colleagues adopt tool and James becomes maintainer | L | M | Scope as personal POC; no support commitment |

---

## 14. Success Metrics

### Correctness

1. **DiD**: True ATT (3.5) falls within the 95% CI of the DiD estimator on `synthetic_omnichannel.csv`. [R1: addresses m16 -- CI containment, not p-value]
2. **Staggered DiD**: On a staggered variant of the synthetic data, Callaway-Sant'Anna ATT estimate is within 1.0 unit of true value.
3. **PSM**: Recovers ATT within 1.0 unit of true value; post-matching SMDs < 0.1 for all covariates. Null tactic (`channel_direct_mail`) ATT is not significantly different from zero.
4. **IPW/AIPW**: ATE/ATT estimates within 1.0 unit of true value; effective sample size > 50% of nominal. AIPW and IPW estimates agree within SEs.
5. **RDD**: On `synthetic_rdd.csv`, true LATE (4.0) falls within the 95% CI of the `rdrobust` estimate.
6. **Synthetic Control**: On `synthetic_scm.csv`, the treated-vs-synthetic gap matches the known treatment effect (6.0) within the placebo test distribution.
7. **Cross-validation against R**: Balance diagnostics (SMD values) match R's `MatchIt::summary()` output to 2 decimal places on the same data. DiD coefficient estimates and clustered SEs match R's `fixest::feols()` output to 2 decimal places. [R1: addresses reviewer -- SEs are more sensitive than point estimates]

### Performance

7. App loads (CSV upload to overview tab rendered) in <10 seconds for a 50K-row CSV on an 8 GB RAM laptop.
8. Full single-method analysis (DiD or PSM) completes in <30 seconds on 100K rows.
9. Full multi-method analysis (DiD + PSM + IPW) completes in <60 seconds on 100K rows.
10. Synthetic Control on 100 donors x 20 periods completes in <60 seconds.

### Geographic

11. Scattergeo renders correctly for ZIP centroids across all 50 US states + DC. [R1: scatter, not choropleth]
12. Opportunity/diminishing-return zone overlay renders without crashing on 500+ ZIP codes.

### Export

13. PDF summary report includes: method name, sample sizes, ATT/ATE estimate with CI and p-value, at least one balance plot, at least one results plot, and an interpretation sentence.
14. CSV export includes all original columns plus: propensity score, match indicator, IPW weight, treatment assignment (post-threshold).

### User Experience

15. A non-technical user (business analyst with Excel proficiency) can upload a CSV and obtain a DiD result with geographic visualization in <5 minutes without reading documentation.

---

## 15. Effort Estimate

### Module-Level Estimates [R1: revised per M8 -- effort estimate increased ~35%]

| Module | LOC (est.) | Complexity | Dependencies | Dev Time (with Claude Code) |
|--------|-----------|------------|--------------|---------------------------|
| `app.py` (layout + routing) | 350-450 | Medium | streamlit | 4-5 hours |
| `column_detector.py` | 150-200 | Low | pandas | 1-2 hours |
| `threshold.py` | 100-150 | Low | streamlit, numpy | 1 hour |
| `validators.py` | 150-200 | Low | pandas | 1-2 hours |
| `propensity.py` | 100-150 | Low | sklearn | 1-2 hours |
| `did.py` | 350-450 | Medium-High | statsmodels, linearmodels, csdid | 5-6 hours (includes staggered DiD) |
| `psm.py` | 300-400 | Medium-High | sklearn, scipy | 4-5 hours |
| `ipw.py` | 200-300 | Medium | sklearn, statsmodels | 3-4 hours (includes AIPW) |
| `rdd.py` | 150-200 | Medium | rdrobust | 2-3 hours |
| `synth.py` | 250-350 | Medium-High | pysyncon | 4-6 hours (placebo tests + progress bar) |
| `balance.py` | 200-300 | Medium | matplotlib, seaborn, plotly | 3-4 hours |
| `sensitivity.py` | 150-250 | Medium | scipy, statsmodels | 3-4 hours (E-value primary; Rosenbaum deferred) |
| `geo.py` | 250-350 | Medium | plotly | 4-5 hours (Scattergeo, simpler than choropleth) |
| `exporters.py` | 200-300 | Medium | fpdf2, plotly, kaleido | 4-6 hours (PDF with embedded plots is fiddly) |
| `generate_sample_data.py` | 200-300 | Low-Medium | numpy, pandas | 2-3 hours (3 datasets, not 1) |
| Tests (all) | 500-700 | Medium | pytest | 6-8 hours (3 synthetic datasets; R cross-validation) |
| Data (zip_latlon.csv) | N/A | Low | Census download | 1 hour |
| Integration + debugging | N/A | High | All | 10-15 hours (Streamlit session_state, caching, cross-module state) |

### Summary

| Metric | Estimate |
|--------|----------|
| **Total LOC** | ~3,400-5,000 |
| **Total dev time** (Claude Code assisted) | ~55-75 hours |
| **Human review time** | ~20-25 hours |
| **Calendar time** (evenings/weekends, 8-10 hr/week) | ~7-9 weeks |
| **Complexity tier** | Medium |

### Phase Plan

| Phase | Scope | Duration |
|-------|-------|----------|
| **Phase 1**: Core pipeline | CSV upload, column detection, validation, DiD (standard + staggered), PSM (LR + GBM), propensity module, balance diagnostics, basic UI. **Week 2 milestone**: CSV upload + column detection + raw DiD running (minimum viable demo). | Weeks 1-4 |
| **Phase 2**: Extended methods + geo | IPW/AIPW, RDD, Synthetic Control, sensitivity tab (E-value, placebo tests), Scattergeo visualization, actionable zones | Weeks 4-7 |
| **Phase 3**: Polish + export | PDF export, forest plot, styling, performance optimization, documentation, all 3 synthetic dataset validations | Weeks 7-9 |

---

## Hypothesis

ACID-Dash is a tool project, not a hypothesis-driven research project. However, the implicit hypothesis is:

**A self-service causal inference dashboard with automated balance diagnostics, geographic visualization, and sensitivity analysis will produce more defensible engagement tactic evaluations than the current practice of comparing raw mean outcomes between exposed and unexposed customers.** This is falsified if analytics professionals find the tool's outputs no more useful or trustworthy than existing Excel/PowerPoint-based analyses for tactic evaluation decisions. Qualitative success criterion: at least one colleague reviews the tool output and confirms it provides information they would use in a tactic evaluation. [R1: addresses reviewer suggestion on qualitative validation]

---

## Resource Estimate

- **Compute**: Laptop-scale. All computation runs on CPU. No GPU, no HPC, no cloud.
- **Data acquisition**: Zero. User provides their own CSV. Synthetic data is generated.
- **Estimated agent-hours**: ~55-75 hours of Claude Code time across all phases. [R1: revised per M8]
- **Human review time**: ~20-25 hours for code review, UI testing, and stakeholder feedback. [R1: revised per M8]
- **Cost**: $0 (all open-source dependencies; no paid APIs; no data purchases).

---

## Cross-Pollination

### Connection to CBRE

CBRE (Causal Biological Reasoning Engine) develops novel causal graph architectures for biological intervention prediction. ACID-Dash applies established causal inference methods (DiD, PSM, IPW) to commercial analytics data. The conceptual bridge: both projects use causal reasoning to move beyond correlational analysis, but CBRE targets graph-structured biological networks while ACID-Dash targets tabular promotional data. Shared intellectual infrastructure: understanding of potential outcomes framework, d-separation, confounding adjustment.

### Connection to ECOCHEMICAL

ECOCHEMICAL's active learning loop (BoTorch/GPyTorch) optimizes experimental design for co-culture screening. ACID-Dash's geographic "opportunity zone" analysis performs an analogous function: identifying underexplored regions in a feature space (geography x tactic intensity) where the expected return on investment is high. The conceptual parallel could inspire a future common abstraction for "where should we invest next?" analyses across domains.

### Shared Infrastructure

- `commons/` utilities are not directly shared (ACID-Dash is a standalone tool, not a research pipeline), but patterns from ACID-Dash's `validators.py` and `exporters.py` could be generalized into `commons/` if other projects need CSV validation or PDF report generation.

### Newsletter Potential

Strong Foretodata piece: **"Your A/B Test Is Lying to You: Why Commercial Analytics Needs Causal Inference"** -- accessible explanation of why comparing means is wrong for non-randomized engagement campaigns, with ACID-Dash as the solution demo. Bridges James's commercial analytics work with his research lab's causal inference focus.

---

## Go/No-Go Recommendation

**Recommendation: GO WITH CONDITIONS (all conditions resolved in R1).**

### Reasons to Proceed

1. **Direct professional utility**: This tool addresses a real, recurring pain point. Every engagement tactic review involves "did the tactic work?" currently answered with naive comparisons. ACID-Dash provides a rigorous alternative.

2. **Low risk, high floor**: The core methods (DiD, PSM, IPW) are well-understood and have battle-tested Python implementations. The development risk is in UI/UX and integration, not in algorithmic novelty. The worst case is a functional but unpolished tool that still improves on the status quo.

3. **Contained scope**: ~3,400-5,000 LOC, ~55-75 development hours, $0 budget. No wet lab, no HPC, no data acquisition.

4. **Portable value**: Generic demo data with no domain-specific identifiers makes this shareable and demoable. Potential for open-source release or Foretodata content.

5. **Causal inference portfolio synergy**: Developing ACID-Dash deepens James's applied causal inference skills (DiD, PSM, IPW, RDD, Synthetic Control), which directly supports the CBRE project's theoretical causal reasoning work.

### Reasons for Caution

1. **Not a research publication**: This is a tool, not a paper. It earns its place in the portfolio through professional utility, not academic output. Should not crowd out CBRE or ECOCHEMICAL development time.

2. **Maintenance burden**: Scope as personal POC; no support commitment.

### Portfolio Scheduling Note [R1: addresses reviewer recommendation]

ACID-Dash development should be scheduled in a window that does not overlap with CBRE sprint weeks. Ideal sequencing: complete ACID-Dash Phase 1 before CBRE's next sprint begins, then interleave Phases 2-3 as time permits.

### Bottom Line

Build it. Phase 1 (DiD + staggered DiD + PSM + balance + basic UI) is achievable in 3-4 weeks and immediately useful. The extended methods (AIPW, RDD, Synthetic Control, geographic viz) are added incrementally. The tool pays for itself the first time it prevents a flawed tactic evaluation from reaching a business review.

---

## References

1. Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
2. Imbens, G.W. & Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
3. Abadie, A. (2021). Using synthetic controls: feasibility, data requirements, and methodological aspects. *Journal of Economic Literature*, 59(2), 391-425.
4. Calonico, S., Cattaneo, M.D. & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295-2326.
5. Rosenbaum, P.R. (2002). *Observational Studies* (2nd ed.). Springer.
6. Austin, P.C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.
7. Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
8. Facure, M. (2022). *Causal Inference for the Brave and True*. https://matheusfacure.github.io/python-causality-handbook/
9. Stuart, E.A. (2010). Matching methods for causal inference: a review and a look forward. *Statistical Science*, 25(1), 1-21.
10. Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
11. Abadie, A. & L'Hour, J. (2021). A penalized synthetic control estimator for disaggregated data. *Journal of the American Statistical Association*, 116(536), 1817-1834.
12. Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277. [R1: added for staggered DiD]
13. de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. *American Economic Review*, 110(9), 2964-2996. [R1: added for staggered DiD]
14. VanderWeele, T.J. & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274. [R1: added for E-value sensitivity]
15. Austin, P.C. (2009). Balance diagnostics for comparing the distribution of baseline covariates. *Statistics in Medicine*, 28(25), 3083-3107. [R1: added for SMD thresholds]

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| Initial | 2026-02-22 | Original proposal |
| R1 | 2026-02-22 | Addressed all reviewer feedback: 2 blockers (B1 geo viz, B2 numpy pin), 10 major concerns (M1-M10), 18 minor issues. Generalized demo data to omnichannel/customer (no domain-specific IDs). Excluded Bayesian A/B test. Added staggered DiD, AIPW, GBM propensity, E-value sensitivity. Revised effort estimate to 55-75 hours. |

## Reviewer Handoff

```
---
REVIEWER TASK: Evaluate ACID-Dash Proposal
STATUS: Review Complete -- Go with Conditions (all conditions resolved in R1)
REVIEW_PATH: _incubator/acid-dash/REVIEW.md
PROPOSAL_PATH: _incubator/acid-dash/PROPOSAL.md
REVIEWED_BY: research-reviewer agent
DATE: 2026-02-22

NEXT STEP: Proceed to build (Phase 1: Core pipeline)
```
