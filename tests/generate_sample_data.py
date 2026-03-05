"""Generate synthetic datasets for ACID-Dash validation.

Three datasets with known ground truths:
1. synthetic_omnichannel.csv — DiD/PSM/IPW validation (ATT=3.5)
2. synthetic_rdd.csv — RDD validation (LATE=4.0)
3. synthetic_scm.csv — Synthetic Control validation (effect=6.0)

All use generic customer/omnichannel framing. Seed=42 for reproducibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

# Real US ZIP codes sampled across Census regions for realism
SAMPLE_ZIPS = {
    "Northeast": [
        "10001", "10002", "10003", "10010", "10011", "10012", "10013",
        "02101", "02102", "02103", "02108", "02109", "02110", "02111",
        "19101", "19102", "19103", "19104", "19106", "19107", "19109",
        "06101", "06103", "06105", "06106",
        "07101", "07102", "07103", "07104", "07105",
        "08601", "08602", "08608", "08609",
        "03101", "03102", "03103", "03104",
        "05401", "05402", "05403", "05404",
        "02901", "02902", "02903", "02904", "02905",
        "04101", "04102", "04103", "04104",
    ],
    "South": [
        "20001", "20002", "20003", "20004", "20005", "20006", "20007",
        "30301", "30302", "30303", "30305", "30306", "30307", "30308",
        "33101", "33109", "33125", "33126", "33127", "33128", "33129",
        "77001", "77002", "77003", "77004", "77005", "77006", "77007",
        "28201", "28202", "28203", "28204", "28205", "28206", "28207",
        "37201", "37203", "37204", "37206", "37207", "37208", "37209",
        "70112", "70113", "70114", "70115", "70116", "70117", "70118",
        "23219", "23220", "23221", "23222", "23223", "23224", "23225",
        "29401", "29403", "29405", "29407",
    ],
    "Midwest": [
        "60601", "60602", "60603", "60604", "60605", "60606", "60607",
        "48201", "48202", "48203", "48204", "48205", "48206", "48207",
        "44101", "44102", "44103", "44104", "44105", "44106", "44107",
        "55401", "55402", "55403", "55404", "55405", "55406", "55407",
        "63101", "63102", "63103", "63104",
        "46201", "46202", "46203", "46204",
        "53201", "53202", "53203", "53204",
        "43201", "43202", "43203", "43204", "43205",
    ],
    "West": [
        "90001", "90002", "90003", "90004", "90005", "90006", "90007",
        "94101", "94102", "94103", "94104", "94105", "94107", "94108",
        "98101", "98102", "98103", "98104", "98105", "98106", "98107",
        "80201", "80202", "80203", "80204", "80205", "80206", "80207",
        "85001", "85003", "85004", "85006", "85007", "85008", "85009",
        "97201", "97202", "97203", "97204", "97205", "97206", "97207",
        "89101", "89102", "89103", "89104",
        "84101", "84102", "84103", "84104",
        "96801", "96802", "96803", "96804",
    ],
}


def generate_omnichannel(rng: np.random.Generator) -> pd.DataFrame:
    """Generate main omnichannel dataset for DiD/PSM/IPW validation.

    DGP:
        revenue = beta_0(50) + beta_1(0)*treated + beta_2(2)*post
                  + beta_3(3.5)*treated*post + gamma*X
                  + 1.2*tv_national + 0.8*tv_local + 0.2*tv_streaming
                  + 0.003*social_paid + 0.05*social_organic
                  + 0.3*sales_rep + epsilon(N(0,8))
        True ATT for channel_email on revenue = 3.5
        channel_direct_mail has NO causal effect (null tactic)
        channel_webinar has own causal effect (beta=0.8 per attendance)

    Channel structure:
        tv_national:        Independent of treatment (broad reach), +1.2 revenue
        tv_local:           Region-correlated, +0.8 revenue
        tv_streaming:       Engagement/industry-correlated (weak confounder), +0.2/impression
        social_paid:        Company-size/spend-correlated (confounder), +0.003/impression
        social_organic:     Engagement-correlated (confounder), +0.05/point
        sales_rep_touches:  STRONG confounder (company_size + prior_spend), +0.3/touch
    """
    n_customers = 300
    n_weeks = 20
    pre_periods = 10

    # DGP parameters
    beta_0 = 50.0   # baseline revenue
    beta_1 = 0.0    # no direct treatment baseline effect (all confounding via observables)
    beta_2 = 2.0    # common time trend (per post-period)
    beta_3 = 3.5    # TRUE ATT for channel_email
    beta_webinar = 0.8  # causal effect per webinar attendance (time-invariant)
    gamma_company = {"Small": 0.0, "Medium": 3.0, "Large": 8.0, "Enterprise": 15.0}
    gamma_industry = {
        "Technology": 5.0, "Finance": 3.0, "Healthcare": 2.0,
        "Manufacturing": 0.0, "Other": -2.0,
    }
    gamma_prior_spend = 0.02
    gamma_engagement = 0.1
    sigma_epsilon = 8.0

    # Assign ZIP codes (200 unique, stratified by region)
    all_zips = []
    zip_regions = {}
    for region, zips in SAMPLE_ZIPS.items():
        n_select = min(50, len(zips))
        selected = rng.choice(zips, size=n_select, replace=False)
        all_zips.extend(selected)
        for z in selected:
            zip_regions[z] = region
    all_zips = np.array(all_zips)

    # Customer-level attributes (time-invariant)
    company_sizes = rng.choice(
        ["Small", "Medium", "Large", "Enterprise"],
        size=n_customers,
        p=[0.30, 0.30, 0.25, 0.15],
    )
    industries = rng.choice(
        ["Technology", "Finance", "Healthcare", "Manufacturing", "Other"],
        size=n_customers,
        p=[0.25, 0.20, 0.20, 0.20, 0.15],
    )
    prior_spend = np.clip(rng.normal(500, 150, n_customers), 50, None)
    tenure_years = rng.uniform(0.5, 15, n_customers)
    engagement_score = np.clip(rng.normal(50, 15, n_customers), 0, 100)
    customer_zips = rng.choice(all_zips, size=n_customers)
    customer_regions = np.array([zip_regions[z] for z in customer_zips])

    # Treatment assignment with confounding
    # Enterprise customers 3x more likely, Technology 2x more likely
    # Centered to target ~40% overall treatment rate
    logit_treat = (
        -1.2
        + 1.1 * (company_sizes == "Enterprise")
        + 0.7 * (company_sizes == "Large")
        + 0.7 * (industries == "Technology")
        + 0.001 * (prior_spend - 500)
        + 0.015 * (engagement_score - 50)
    )
    p_treat = 1 / (1 + np.exp(-logit_treat))
    channel_email = (rng.uniform(size=n_customers) < p_treat).astype(int)

    # channel_direct_mail: weakly correlated with email, NO causal effect
    p_dm = 0.20 + 0.10 * channel_email
    channel_direct_mail = (rng.uniform(size=n_customers) < p_dm).astype(int)

    # --- New channel variables (customer-level base rates) ---

    # TV national GRP: weekly schedule, same for all customers in a week.
    # Independent of treatment — broad market-level advertising.
    tv_national_weekly = np.clip(rng.normal(100, 20, n_weeks), 40, 200)

    # TV local GRP: region-level base rates (South/West spend more).
    # Time-invariant base + weekly noise per customer.
    tv_local_base = {"Northeast": 70, "South": 100, "Midwest": 60, "West": 90}

    # TV streaming impressions: engagement + industry correlated (weak confounder).
    # Tech/Finance customers see more streaming ads.
    tv_streaming_mu = (
        5.0
        + 0.05 * engagement_score
        + 3.0 * (industries == "Technology").astype(float)
        + 1.5 * (industries == "Finance").astype(float)
    )

    # Social paid impressions: company-size + prior_spend correlated (confounder).
    social_paid_mu = (
        200.0
        + 200.0 * (company_sizes == "Medium").astype(float)
        + 500.0 * (company_sizes == "Large").astype(float)
        + 1000.0 * (company_sizes == "Enterprise").astype(float)
        + 0.5 * prior_spend
    )

    # Social organic engagement score: engagement_score correlated (confounder).
    social_organic_mu = 10.0 + 0.6 * engagement_score

    # Sales rep touches: STRONG confounder through shared causes.
    # Large/Enterprise companies AND high-spenders get more rep attention.
    # Since treatment is also driven by company_size, sales_rep is a
    # confounder (company_size -> sales_rep -> revenue AND
    # company_size -> treatment), NOT a mediator.
    sales_rep_mu = (
        1.0
        + 2.0 * (company_sizes == "Medium").astype(float)
        + 5.0 * (company_sizes == "Large").astype(float)
        + 10.0 * (company_sizes == "Enterprise").astype(float)
        + 0.01 * prior_spend
    )

    # Build panel
    rows = []
    for i in range(n_customers):
        for w in range(1, n_weeks + 1):
            post = int(w > pre_periods)
            treated = channel_email[i]

            # Webinar attendance (continuous, log-normal, independent of email)
            # Independent of treatment so it doesn't confound the email ATT
            webinar_count = max(0.0, rng.lognormal(1.0, 0.8) - 1.0)
            webinar_count = round(webinar_count, 1)

            # Channel exposure variables (per customer-week)
            tv_nat = round(float(tv_national_weekly[w - 1]), 1)
            tv_loc = round(
                max(0.0, tv_local_base[customer_regions[i]] + rng.normal(0, 15)),
                1,
            )
            tv_str = round(
                max(0.0, tv_streaming_mu[i] + rng.normal(0, 2)), 1,
            )
            soc_paid = round(
                max(0.0, social_paid_mu[i] + rng.normal(0, 100)), 0,
            )
            soc_org = round(
                max(0.0, social_organic_mu[i] + rng.normal(0, 5)), 1,
            )
            sales_rep = round(
                max(0.0, sales_rep_mu[i] + rng.normal(0, 1.5)), 1,
            )

            # Revenue DGP (including all channel effects)
            revenue = (
                beta_0
                + beta_1 * treated
                + beta_2 * post
                + beta_3 * treated * post
                + beta_webinar * webinar_count
                + gamma_company[company_sizes[i]]
                + gamma_industry[industries[i]]
                + gamma_prior_spend * prior_spend[i]
                + gamma_engagement * engagement_score[i]
                + 0.012 * tv_nat      # +1.2 at mean GRP ~100
                + 0.01 * tv_loc       # +0.8 at mean GRP ~80
                + 0.2 * tv_str        # +1.6 at mean ~8 impressions
                + 0.003 * soc_paid    # +2.1 at mean ~700 impressions
                + 0.05 * soc_org      # +2.0 at mean ~40 points
                + 0.3 * sales_rep     # +1.8 at mean ~6 touches
                + rng.normal(0, sigma_epsilon)
            )

            # Units sold: correlated with revenue, true ATT=1.2
            units = (
                10
                + 1.2 * treated * post
                + 0.12 * revenue
                + rng.normal(0, 3)
            )

            rows.append({
                "customer_id": i + 1,
                "zip_code": customer_zips[i],
                "week": w,
                "channel_email": treated,
                "channel_webinar": webinar_count,
                "channel_direct_mail": channel_direct_mail[i],
                "tv_national_grp": tv_nat,
                "tv_local_grp": tv_loc,
                "tv_streaming_impressions": tv_str,
                "social_paid_impressions": soc_paid,
                "social_organic_score": soc_org,
                "sales_rep_touches": sales_rep,
                "revenue": round(revenue, 2),
                "units_sold": round(max(0, units), 2),
                "prior_spend": round(prior_spend[i], 2),
                "tenure_years": round(tenure_years[i], 2),
                "company_size": company_sizes[i],
                "industry": industries[i],
                "engagement_score": round(engagement_score[i], 2),
                "region": customer_regions[i],
            })

    return pd.DataFrame(rows)


def generate_rdd(rng: np.random.Generator) -> pd.DataFrame:
    """Generate RDD dataset with sharp cutoff at engagement_score=50.

    DGP:
        revenue = 30 + 0.5*score + 4.0*(score >= 50) + N(0, 5)
        True LATE at cutoff = 4.0
    """
    n = 2000
    engagement_score = rng.uniform(20, 80, n)
    treated = (engagement_score >= 50).astype(int)

    company_sizes = rng.choice(
        ["Small", "Medium", "Large", "Enterprise"],
        size=n,
        p=[0.30, 0.30, 0.25, 0.15],
    )
    regions = rng.choice(
        ["Northeast", "South", "Midwest", "West"],
        size=n,
        p=[0.25, 0.30, 0.20, 0.25],
    )

    revenue_post = (
        30
        + 0.5 * engagement_score
        + 4.0 * treated
        + rng.normal(0, 5, n)
    )

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "engagement_score": np.round(engagement_score, 2),
        "treated": treated,
        "revenue_post": np.round(revenue_post, 2),
        "company_size": company_sizes,
        "region": regions,
    })


def generate_scm(rng: np.random.Generator) -> pd.DataFrame:
    """Generate Synthetic Control dataset: 100 ZIPs, 20 weeks, 5 treated.

    DGP:
        revenue = mu_i + lambda_t + 6.0*treated*post + epsilon
        True treatment effect = 6.0
    """
    n_zips = 100
    n_weeks = 20
    pre_periods = 10
    n_treated = 5
    true_effect = 6.0

    # Select ZIPs
    all_zips_flat = []
    for zips in SAMPLE_ZIPS.values():
        all_zips_flat.extend(zips)
    selected_zips = rng.choice(all_zips_flat, size=n_zips, replace=False)

    # Mark first 5 as treated
    treated_zips = set(selected_zips[:n_treated])

    # ZIP-level fixed effects
    mu = rng.normal(50, 10, n_zips)
    # Time fixed effects (common trend)
    lambda_t = np.linspace(0, 5, n_weeks)
    # Market size (time-invariant covariate)
    market_size = np.round(rng.lognormal(3, 0.5, n_zips), 2)

    rows = []
    for i, z in enumerate(selected_zips):
        is_treated = z in treated_zips
        for w in range(1, n_weeks + 1):
            post = int(w > pre_periods)
            revenue = (
                mu[i]
                + lambda_t[w - 1]
                + true_effect * is_treated * post
                + rng.normal(0, 3)
            )
            rows.append({
                "zip_code": z,
                "week": w,
                "treated": int(is_treated),
                "revenue": round(revenue, 2),
                "market_size": market_size[i],
            })

    return pd.DataFrame(rows)


def generate_campaigns(rng: np.random.Generator) -> pd.DataFrame:
    """Generate multi-campaign dataset with known per-campaign ATTs.

    Simulates a realistic media-mix scenario where 15 campaigns across
    5 channels (TV, Email, Social, Direct Mail, Field Rep) run over a
    20-week panel.  Each campaign has a distinct causal effect on revenue,
    ranging from strongly positive to null to negative.

    DGP:
        revenue = 50 + sum(campaign_att_i * exposed_i * post)
                  + covariates + epsilon(N(0, 8))

    Known ATTs (revenue, post-period only):
        TV:
            tv_30s_brand .............. +0.5  (weak brand lift)
            tv_45s_product ............ +2.5  (strong product focus)
            tv_60s_testimonial ........ +3.5  (strongest — long-form trust)
            tv_15s_reminder ........... +0.2  (negligible digital pre-roll)
        Email:
            email_personalized ........ +2.0  (personalized subject line)
            email_generic ............. +0.3  (generic newsletter, weak)
            email_promo ............... +1.5  (promotional / discount CTA)
        Social:
            social_retargeting ........ +2.5  (high intent, retargeted)
            social_brand_awareness .... +0.5  (broad reach, low intent)
            social_ugc ................ -0.8  (negative — cannibalizes organic)
        Direct Mail:
            dm_catalog ................ +1.2  (full product catalog)
            dm_postcard ............... +0.0  (null — no effect)
            dm_letter ................. +1.8  (personalized letter)
        Field Rep:
            rep_product_demo .......... +4.0  (strongest overall — in-person)
            rep_followup_call ......... +1.0  (phone follow-up)

    Confounding structure:
        - High-value customers (Enterprise/Large, high prior_spend) are
          more likely to receive rep_product_demo, email_personalized,
          and dm_letter (targeting bias).
        - Tech/Finance customers are more likely to see social_retargeting
          and tv_45s_product (industry targeting).
        - Brand campaigns (tv_30s_brand, social_brand_awareness) have
          near-random assignment (low confounding).
        - Promo campaigns target customers with lower engagement_score
          (win-back strategy).
    """
    n_customers = 300
    n_weeks = 20
    pre_periods = 10

    # --- Campaign definitions: (name, true_att, assignment_fn) ---
    # Assignment functions return P(exposed) per customer given their
    # characteristics.  They are called once and the exposure is held
    # constant across all weeks (customer-level assignment, like the
    # original channel_email design).

    beta_0 = 50.0
    sigma_epsilon = 8.0

    gamma_company = {"Small": 0.0, "Medium": 3.0, "Large": 8.0, "Enterprise": 15.0}
    gamma_industry = {
        "Technology": 5.0, "Finance": 3.0, "Healthcare": 2.0,
        "Manufacturing": 0.0, "Other": -2.0,
    }
    gamma_prior_spend = 0.02
    gamma_engagement = 0.1

    # Assign ZIP codes
    all_zips = []
    zip_regions = {}
    for region, zips in SAMPLE_ZIPS.items():
        n_select = min(50, len(zips))
        selected = rng.choice(zips, size=n_select, replace=False)
        all_zips.extend(selected)
        for z in selected:
            zip_regions[z] = region
    all_zips = np.array(all_zips)

    # Customer-level attributes
    company_sizes = rng.choice(
        ["Small", "Medium", "Large", "Enterprise"],
        size=n_customers, p=[0.30, 0.30, 0.25, 0.15],
    )
    industries = rng.choice(
        ["Technology", "Finance", "Healthcare", "Manufacturing", "Other"],
        size=n_customers, p=[0.25, 0.20, 0.20, 0.20, 0.15],
    )
    prior_spend = np.clip(rng.normal(500, 150, n_customers), 50, None)
    tenure_years = rng.uniform(0.5, 15, n_customers)
    engagement_score = np.clip(rng.normal(50, 15, n_customers), 0, 100)
    customer_zips = rng.choice(all_zips, size=n_customers)
    customer_regions = np.array([zip_regions[z] for z in customer_zips])

    # Helper arrays for propensity computation
    is_enterprise = (company_sizes == "Enterprise").astype(float)
    is_large = (company_sizes == "Large").astype(float)
    is_medium = (company_sizes == "Medium").astype(float)
    is_tech = (industries == "Technology").astype(float)
    is_finance = (industries == "Finance").astype(float)
    spend_z = (prior_spend - 500) / 150  # standardized
    engage_z = (engagement_score - 50) / 15

    def _assign(logit: np.ndarray) -> np.ndarray:
        """Bernoulli draw from logistic of *logit*."""
        p = 1 / (1 + np.exp(-logit))
        return (rng.uniform(size=n_customers) < p).astype(int)

    # --- Campaign assignment (customer-level, time-invariant) ---
    campaigns: dict[str, tuple[float, np.ndarray]] = {}

    # TV campaigns
    # 30s brand: broad reach, near-random (low confounding)
    campaigns["tv_30s_brand"] = (
        0.5,
        _assign(-0.4 + 0.1 * spend_z),  # ~40%, near-random
    )
    # 45s product: tech/finance targeted
    campaigns["tv_45s_product"] = (
        2.5,
        _assign(-0.8 + 0.6 * is_tech + 0.4 * is_finance + 0.2 * spend_z),
    )
    # 60s testimonial: targets higher-value customers
    campaigns["tv_60s_testimonial"] = (
        3.5,
        _assign(-1.2 + 0.5 * is_enterprise + 0.4 * is_large + 0.3 * spend_z),
    )
    # 15s reminder: very broad, nearly random
    campaigns["tv_15s_reminder"] = (
        0.2,
        _assign(-0.2 + 0.05 * engage_z),  # ~45%, near-random
    )

    # Email campaigns
    # Personalized: targets high-value, high-engagement
    campaigns["email_personalized"] = (
        2.0,
        _assign(-1.0 + 0.5 * is_enterprise + 0.4 * is_large
                + 0.3 * engage_z + 0.2 * spend_z),
    )
    # Generic: broad newsletter, near-random
    campaigns["email_generic"] = (
        0.3,
        _assign(-0.3 + 0.1 * engage_z),  # ~42%
    )
    # Promo: win-back — targets LOW engagement (reverse confounding)
    campaigns["email_promo"] = (
        1.5,
        _assign(-0.5 - 0.4 * engage_z + 0.2 * spend_z),
    )

    # Social campaigns
    # Retargeting: tech/finance + high engagement
    campaigns["social_retargeting"] = (
        2.5,
        _assign(-1.0 + 0.5 * is_tech + 0.3 * is_finance + 0.3 * engage_z),
    )
    # Brand awareness: broad, near-random
    campaigns["social_brand_awareness"] = (
        0.5,
        _assign(-0.4 + 0.1 * is_tech),  # ~40%
    )
    # UGC: cannibalizes organic — high-engagement customers targeted
    campaigns["social_ugc"] = (
        -0.8,
        _assign(-0.8 + 0.4 * engage_z + 0.2 * is_tech),
    )

    # Direct Mail
    # Catalog: medium targeting
    campaigns["dm_catalog"] = (
        1.2,
        _assign(-0.6 + 0.3 * is_large + 0.2 * is_enterprise + 0.1 * spend_z),
    )
    # Postcard: near-random, null effect
    campaigns["dm_postcard"] = (
        0.0,
        _assign(-0.3 + 0.05 * engage_z),  # ~43%
    )
    # Letter: high-value targeting
    campaigns["dm_letter"] = (
        1.8,
        _assign(-1.3 + 0.6 * is_enterprise + 0.5 * is_large + 0.3 * spend_z),
    )

    # Field Rep
    # Product demo: very strong targeting to Enterprise/Large
    campaigns["rep_product_demo"] = (
        4.0,
        _assign(-2.0 + 1.2 * is_enterprise + 0.8 * is_large + 0.3 * spend_z),
    )
    # Follow-up call: moderate targeting
    campaigns["rep_followup_call"] = (
        1.0,
        _assign(-0.7 + 0.4 * is_enterprise + 0.3 * is_large + 0.2 * engage_z),
    )

    # --- Build panel ---
    rows = []
    for i in range(n_customers):
        for w in range(1, n_weeks + 1):
            post = int(w > pre_periods)

            # Revenue DGP: baseline + covariate effects + campaign effects
            revenue = (
                beta_0
                + gamma_company[company_sizes[i]]
                + gamma_industry[industries[i]]
                + gamma_prior_spend * prior_spend[i]
                + gamma_engagement * engagement_score[i]
                + 2.0 * post  # common time trend
            )

            # Add each campaign's causal effect (only in post-period)
            for cname, (att, assignment) in campaigns.items():
                revenue += att * assignment[i] * post

            # Noise
            revenue += rng.normal(0, sigma_epsilon)

            # Units sold (correlated outcome)
            units = 10 + 0.12 * revenue + rng.normal(0, 3)

            row = {
                "customer_id": i + 1,
                "zip_code": customer_zips[i],
                "week": w,
                "revenue": round(revenue, 2),
                "units_sold": round(max(0, units), 2),
                "prior_spend": round(prior_spend[i], 2),
                "tenure_years": round(tenure_years[i], 2),
                "company_size": company_sizes[i],
                "industry": industries[i],
                "engagement_score": round(engagement_score[i], 2),
                "region": customer_regions[i],
            }

            # Add campaign exposure columns
            for cname, (att, assignment) in campaigns.items():
                row[cname] = assignment[i]

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Generate all four synthetic datasets."""
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic_omnichannel.csv ...")
    df_omni = generate_omnichannel(rng)
    df_omni.to_csv(OUTPUT_DIR / "synthetic_omnichannel.csv", index=False)
    print(f"  {len(df_omni)} rows x {len(df_omni.columns)} cols")
    print(f"  Treated (email): {df_omni['channel_email'].mean():.1%}")
    print(f"  Weeks: {df_omni['week'].nunique()}")

    print("\nGenerating synthetic_campaigns.csv ...")
    df_campaigns = generate_campaigns(rng)
    df_campaigns.to_csv(OUTPUT_DIR / "synthetic_campaigns.csv", index=False)
    n_camp_cols = [c for c in df_campaigns.columns if c.startswith(("tv_", "email_", "social_", "dm_", "rep_"))]
    print(f"  {len(df_campaigns)} rows x {len(df_campaigns.columns)} cols")
    print(f"  Campaigns: {len(n_camp_cols)}")
    for c in n_camp_cols:
        pct = df_campaigns[c].mean()
        print(f"    {c}: {pct:.1%} exposed")

    print("\nGenerating synthetic_rdd.csv ...")
    df_rdd = generate_rdd(rng)
    df_rdd.to_csv(OUTPUT_DIR / "synthetic_rdd.csv", index=False)
    print(f"  {len(df_rdd)} rows x {len(df_rdd.columns)} cols")
    print(f"  Treated: {df_rdd['treated'].mean():.1%}")

    print("\nGenerating synthetic_scm.csv ...")
    df_scm = generate_scm(rng)
    df_scm.to_csv(OUTPUT_DIR / "synthetic_scm.csv", index=False)
    print(f"  {len(df_scm)} rows x {len(df_scm.columns)} cols")
    print(f"  Treated ZIPs: {df_scm.groupby('zip_code')['treated'].first().sum()}")

    print(f"\nAll datasets saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
