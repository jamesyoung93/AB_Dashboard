"""Geographic visualization module for ACID-Dash.

Creates Plotly Scattergeo maps at ZIP code centroids. ZIP centroids
are resolved from a bundled prefix-to-coordinate lookup covering all
US 3-digit ZIP prefixes used in the sample datasets.

For user-uploaded data, the module will attempt to resolve ZIPs via
the optional ``pgeocode`` library if installed, falling back to the
bundled lookup for any unresolved codes.

References:
    PROPOSAL.md Section 7.1: Geographic Visualization
    Review Blocker B1: use Scattergeo, not choropleth
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Bundled ZIP prefix centroids (3-digit prefix -> (lat, lon))
# Covers all prefixes used in sample data + major US metro areas
# ---------------------------------------------------------------------------

_PREFIX_CENTROIDS: dict[str, tuple[float, float]] = {
    # Northeast
    "100": (40.75, -73.99),   # New York City
    "021": (42.36, -71.06),   # Boston
    "191": (39.95, -75.16),   # Philadelphia
    "061": (41.76, -72.68),   # Hartford
    "071": (40.73, -74.17),   # Newark
    "086": (40.22, -74.76),   # Trenton
    "031": (42.99, -71.45),   # Manchester NH
    "054": (44.48, -73.21),   # Burlington VT
    "029": (41.82, -71.41),   # Providence
    "041": (43.66, -70.26),   # Portland ME
    # South
    "200": (38.90, -77.03),   # Washington DC
    "303": (33.75, -84.39),   # Atlanta
    "331": (25.76, -80.19),   # Miami
    "770": (29.76, -95.37),   # Houston
    "282": (35.23, -80.84),   # Charlotte
    "372": (36.16, -86.78),   # Nashville
    "701": (29.95, -90.07),   # New Orleans
    "232": (37.54, -77.44),   # Richmond
    "294": (32.78, -79.93),   # Charleston SC
    # Midwest
    "606": (41.88, -87.63),   # Chicago
    "482": (42.33, -83.05),   # Detroit
    "441": (41.50, -81.69),   # Cleveland
    "554": (44.98, -93.27),   # Minneapolis
    "631": (38.63, -90.20),   # St Louis
    "462": (39.77, -86.16),   # Indianapolis
    "532": (43.04, -87.91),   # Milwaukee
    "432": (39.96, -82.99),   # Columbus
    # West
    "900": (34.05, -118.24),  # Los Angeles
    "941": (37.78, -122.42),  # San Francisco
    "981": (47.61, -122.33),  # Seattle
    "802": (39.74, -104.99),  # Denver
    "850": (33.45, -112.07),  # Phoenix
    "972": (45.52, -122.68),  # Portland OR
    "891": (36.17, -115.14),  # Las Vegas
    "841": (40.76, -111.89),  # Salt Lake City
    "968": (21.31, -157.86),  # Honolulu
}


# ---------------------------------------------------------------------------
# ZIP centroid resolution
# ---------------------------------------------------------------------------


def _resolve_zip_centroid(zip_code: str) -> tuple[float, float] | None:
    """Resolve a single ZIP code to (lat, lon) using the prefix lookup.

    Adds a small deterministic offset based on the last 2 digits
    to spread points within a metro area (~0.002 deg per unit).

    Args:
        zip_code: 5-digit US ZIP code as a string.

    Returns:
        (latitude, longitude) tuple, or None if the prefix is unknown.
    """
    z = str(zip_code).zfill(5)
    prefix = z[:3]
    if prefix not in _PREFIX_CENTROIDS:
        return None

    base_lat, base_lon = _PREFIX_CENTROIDS[prefix]
    # Deterministic spread: last 2 digits offset from center
    try:
        suffix = int(z[3:])
    except ValueError:
        suffix = 0
    offset = (suffix - 5) * 0.003
    return (base_lat + offset, base_lon + offset * 0.8)


def resolve_zip_centroids(
    zip_codes: list[str],
) -> dict[str, tuple[float, float]]:
    """Resolve a list of ZIP codes to (lat, lon) centroids.

    Attempts:
      1. Bundled prefix lookup (always available)
      2. ``pgeocode`` library (optional, for ZIPs not in bundled data)

    Args:
        zip_codes: List of 5-digit ZIP code strings.

    Returns:
        Dict mapping ZIP code string to (latitude, longitude).
        ZIPs that could not be resolved are omitted.
    """
    result: dict[str, tuple[float, float]] = {}
    unresolved: list[str] = []

    # First pass: bundled lookup
    for z in zip_codes:
        z_str = str(z).zfill(5)
        centroid = _resolve_zip_centroid(z_str)
        if centroid is not None:
            result[z_str] = centroid
        else:
            unresolved.append(z_str)

    # Second pass: try pgeocode for unresolved ZIPs
    if unresolved:
        try:
            import pgeocode

            nomi = pgeocode.Nominatim("us")
            geo_df = nomi.query_postal_code(unresolved)
            if isinstance(geo_df, pd.DataFrame):
                for i, z in enumerate(unresolved):
                    lat = geo_df.iloc[i].get("latitude")
                    lon = geo_df.iloc[i].get("longitude")
                    if pd.notna(lat) and pd.notna(lon):
                        result[z] = (float(lat), float(lon))
            elif isinstance(geo_df, pd.Series):
                lat = geo_df.get("latitude")
                lon = geo_df.get("longitude")
                if pd.notna(lat) and pd.notna(lon):
                    result[unresolved[0]] = (float(lat), float(lon))
        except (ImportError, Exception):
            pass

    return result


# ---------------------------------------------------------------------------
# Public API: Scattergeo map
# ---------------------------------------------------------------------------


def zip_outcome_map(
    df: pd.DataFrame,
    zip_col: str,
    value_col: str,
    title: str = "Geographic Distribution",
    color_scale: str = "RdBu_r",
    size_col: str | None = None,
    hover_cols: list[str] | None = None,
) -> go.Figure:
    """Create a Plotly Scattergeo map of ZIP-level values.

    Each ZIP code is plotted as a circle at its centroid. Circle color
    encodes ``value_col`` and circle size encodes ``size_col`` (or a
    fixed size if not provided).

    Args:
        df: DataFrame with one row per ZIP code. Must contain ``zip_col``
            and ``value_col``.
        zip_col: Column containing 5-digit US ZIP code strings.
        value_col: Numeric column to encode as marker color.
        title: Plot title.
        color_scale: Plotly color scale name (default "RdBu_r").
        size_col: Optional numeric column to encode as marker size.
        hover_cols: Optional list of columns to include in hover text.

    Returns:
        Plotly Figure object. Display with ``st.plotly_chart(fig)``.
    """
    # Resolve centroids
    zip_strings = df[zip_col].astype(str).str.zfill(5).tolist()
    centroids = resolve_zip_centroids(list(set(zip_strings)))

    # Build lat/lon columns
    work = df.copy()
    work["_zip_str"] = work[zip_col].astype(str).str.zfill(5)
    work["_lat"] = work["_zip_str"].map(lambda z: centroids.get(z, (None, None))[0])
    work["_lon"] = work["_zip_str"].map(lambda z: centroids.get(z, (None, None))[1])

    # Drop rows with no centroid
    n_before = len(work)
    work = work.dropna(subset=["_lat", "_lon"])
    n_dropped = n_before - len(work)

    if len(work) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No ZIP codes could be geocoded.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16),
        )
        return fig

    # Marker size
    if size_col and size_col in work.columns:
        sizes = work[size_col].fillna(0).values
        # Normalize to 5-25 range
        smin, smax = sizes.min(), sizes.max()
        if smax > smin:
            sizes = 5 + 20 * (sizes - smin) / (smax - smin)
        else:
            sizes = np.full(len(sizes), 10.0)
    else:
        sizes = np.full(len(work), 10.0)

    # Hover text
    hover_parts = [f"ZIP: {z}" for z in work["_zip_str"]]
    hover_parts = [
        h + f"<br>{value_col}: {v:.2f}"
        for h, v in zip(hover_parts, work[value_col].values)
    ]
    if hover_cols:
        for hc in hover_cols:
            if hc in work.columns:
                hover_parts = [
                    h + f"<br>{hc}: {v}"
                    for h, v in zip(hover_parts, work[hc].values)
                ]

    fig = go.Figure(
        go.Scattergeo(
            lat=work["_lat"].values,
            lon=work["_lon"].values,
            mode="markers",
            marker=dict(
                size=sizes,
                color=work[value_col].values,
                colorscale=color_scale,
                colorbar=dict(title=value_col),
                line=dict(width=0.5, color="white"),
                opacity=0.8,
            ),
            text=hover_parts,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showlakes=True,
            lakecolor="rgb(204, 224, 255)",
            subunitcolor="rgb(200, 200, 200)",
            countrycolor="rgb(200, 200, 200)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
    )

    if n_dropped > 0:
        fig.add_annotation(
            text=f"{n_dropped} ZIP(s) could not be geocoded",
            xref="paper", yref="paper",
            x=0.5, y=-0.05, showarrow=False,
            font=dict(size=10, color="grey"),
        )

    return fig


def zip_treatment_map(
    df: pd.DataFrame,
    zip_col: str,
    treatment_col: str,
    outcome_col: str,
    title: str = "Treatment Effect by Geography",
) -> go.Figure:
    """Create a two-color Scattergeo map showing treated vs control ZIPs.

    Treated ZIPs are colored blue, control ZIPs orange. Marker size
    encodes the outcome variable.

    Args:
        df: DataFrame with one row per ZIP. Must contain zip_col,
            treatment_col (binary), and outcome_col (numeric).
        zip_col: ZIP code column name.
        treatment_col: Binary treatment column name.
        outcome_col: Numeric outcome column name.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    zip_strings = df[zip_col].astype(str).str.zfill(5).tolist()
    centroids = resolve_zip_centroids(list(set(zip_strings)))

    work = df.copy()
    work["_zip_str"] = work[zip_col].astype(str).str.zfill(5)
    work["_lat"] = work["_zip_str"].map(lambda z: centroids.get(z, (None, None))[0])
    work["_lon"] = work["_zip_str"].map(lambda z: centroids.get(z, (None, None))[1])
    work = work.dropna(subset=["_lat", "_lon"])

    if len(work) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No ZIP codes could be geocoded.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig

    # Normalize outcome for marker size
    outcome_vals = work[outcome_col].fillna(0).values
    omin, omax = outcome_vals.min(), outcome_vals.max()
    if omax > omin:
        sizes = 5 + 15 * (outcome_vals - omin) / (omax - omin)
    else:
        sizes = np.full(len(outcome_vals), 10.0)

    fig = go.Figure()

    for treat_val, color, label in [(1, "#1f77b4", "Treated"), (0, "#ff7f0e", "Control")]:
        mask = work[treatment_col] == treat_val
        sub = work[mask]
        if len(sub) == 0:
            continue

        hover = [
            f"ZIP: {z}<br>{outcome_col}: {v:.2f}<br>Group: {label}"
            for z, v in zip(sub["_zip_str"].values, sub[outcome_col].values)
        ]

        fig.add_trace(
            go.Scattergeo(
                lat=sub["_lat"].values,
                lon=sub["_lon"].values,
                mode="markers",
                marker=dict(
                    size=sizes[mask.values],
                    color=color,
                    line=dict(width=0.5, color="white"),
                    opacity=0.75,
                ),
                text=hover,
                hoverinfo="text",
                name=label,
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showlakes=True,
            lakecolor="rgb(204, 224, 255)",
            subunitcolor="rgb(200, 200, 200)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
        legend=dict(x=0.01, y=0.99),
    )

    return fig
