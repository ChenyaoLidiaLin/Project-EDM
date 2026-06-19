import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = "data"

st.set_page_config(page_title="Madrid Road Risk", layout="wide")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Spanish connector words that should stay lowercase when title-casing a
# district name (e.g. "puente de vallecas" -> "Puente de Vallecas").
_SMALL_WORDS = {"de", "del", "la", "las", "los", "el", "y"}


def format_district(name: str) -> str:
    """Turn a raw lowercase district code into a readable display label."""
    if pd.isna(name):
        return name
    words = str(name).split()
    out = [
        w if (i > 0 and w in _SMALL_WORDS) else w.capitalize()
        for i, w in enumerate(words)
    ]
    return " ".join(out)


def derive_vehicle_cat(v) -> str:
    """Mirrors engineer_features() in train_model.py: motorcycle dominates
    fall/rollover (62%), truck dominates object impact (18.5%)."""
    if pd.isna(v):
        return "unknown"
    v = str(v)
    if "motocicleta" in v or "ciclomotor" in v: return "motorcycle"
    if "bicicleta" in v or "vmu" in v:          return "bike"
    if "autobus" in v or "autocar" in v:        return "bus"
    if "camion" in v or "furgon" in v or "tractocamion" in v: return "truck"
    if "turismo" in v or "todo terreno" in v:   return "car"
    return "other"


def derive_single_vehicle(n) -> str:
    """Mirrors engineer_features() in train_model.py."""
    if pd.isna(n):
        return "unknown"
    return "single" if n == 1 else "multiple"


@st.cache_data
def add_model_features(_acc: pd.DataFrame) -> pd.DataFrame:
    """Adds the same engineered columns the model was trained on, so the
    simulator-to-map highlight (tab 1) can match against them."""
    df = _acc.copy()
    df["vehicle_cat"]     = df["tipo_vehiculo"].apply(derive_vehicle_cat)
    df["single_vehicle"]  = df["n_vehicles"].apply(derive_single_vehicle)
    df["hour"]            = pd.to_numeric(
        df["hora"].astype(str).str.split(":").str[0], errors="coerce"
    )
    df["day_of_week"]     = df["dia_semana"]
    return df


@st.cache_data
def compute_default_districts(sensors_df: pd.DataFrame, min_n: int = 3, n_each: int = 2):
    """Pick the most attention-grabbing districts: the highest- and lowest-risk
    ones, so the map isn't empty (or overloaded) on first load."""
    agg = (
        sensors_df[sensors_df["n_accidents"] >= min_n]
        .groupby("district")["risk_index"].mean()
        .sort_values()
    )
    if agg.empty:
        return []
    lowest  = agg.head(n_each).index.tolist()
    highest = agg.tail(n_each).index.tolist()
    ordered, seen = [], set()
    for d in highest + lowest:
        if d not in seen:
            ordered.append(d)
            seen.add(d)
    return ordered


@st.cache_data
def load_sensors():
    return pd.read_parquet(f"{DATA_DIR}/sensor_risk.parquet")


@st.cache_data
def load_district_yearly():
    return pd.read_parquet(f"{DATA_DIR}/district_risk_yearly.parquet")


@st.cache_data
def load_accidents():
    return pd.read_parquet(f"{DATA_DIR}/accidents_clean.parquet")


@st.cache_data
def load_travel_risk():
    return pd.read_parquet(f"{DATA_DIR}/travel_risk.parquet")


@st.cache_resource
def load_model():
    pipe       = joblib.load(f"{DATA_DIR}/accident_type_model.joblib")
    importance = pd.read_parquet(f"{DATA_DIR}/feature_importance.parquet")
    with open(f"{DATA_DIR}/model_metrics.json") as f:
        metrics = json.load(f)
    return pipe, importance, metrics


st.title("Madrid Road Risk: accident rates normalized by traffic volume")
st.markdown(
    "Analysis of Madrid traffic accidents (2016-2024) matched with data from the nearest "
    "traffic sensors (flow, occupancy, mean speed). The goal is to find locations where "
    "**more accidents happen than the local traffic volume would predict**, not just the "
    "spots with the highest raw accident count."
)

# Load datasets needed for the KPI bar and the tabs that follow.
acc            = load_accidents()
acc            = add_model_features(acc)
sensors        = load_sensors()
dist_year      = load_district_yearly()
districts_list = sorted(acc["distrito"].dropna().unique())
default_districts = compute_default_districts(sensors)

# ----------------------------------------------------------------------------
# KPI bar — orients the user in a few seconds before they start exploring
# ----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Total accidents logged",
        f"{len(acc):,}",
        help=f"Madrid, {int(acc['year'].min())}–{int(acc['year'].max())}, "
             "deduplicated by case number.",
    )

with kpi2:
    eligible_sensors = sensors[sensors["n_accidents"] >= 5]
    top_sensor = eligible_sensors.loc[eligible_sensors["risk_index"].idxmax()]
    st.metric(
        "Highest-risk sensor",
        f"index {top_sensor['risk_index']:.2f}",
        help=(
            f"Sensor #{int(top_sensor['id_sensor_cercano'])} · "
            f"{format_district(top_sensor['district'])} · "
            f"{int(top_sensor['n_accidents'])} accidents recorded nearby. "
            "See the Normalized risk map tab for the full picture."
        ),
    )

with kpi3:
    latest_year       = int(dist_year["year"].max())
    df_latest         = dist_year[dist_year["year"] == latest_year]
    top_district_row  = df_latest.loc[df_latest["risk_index"].idxmax()]
    st.metric(
        f"Riskiest district ({latest_year})",
        format_district(top_district_row["distrito"]),
        f"index {top_district_row['risk_index']:.2f}",
        help=f"{int(top_district_row['n_accidents'])} accidents recorded in {latest_year}, "
             "normalized by local traffic volume.",
    )

with kpi4:
    st.metric(
        "Traffic sensors monitored",
        f"{len(sensors):,}",
        help="Sensors with at least one accident matched nearby, used to build the risk map.",
    )

st.divider()

# Pre-populate the simulator's widget state so the risk map (tab 1) can read
# the *current* simulator profile even though those widgets are only created
# later, in tab 2's block.
SIM_DEFAULTS = {
    "sim_vehicle_cat": "car",
    "sim_vehicle_count": "Single vehicle",
    "sim_hour":         12,
    "sim_month":        6,
    "sim_weather":      "clear",
    "sim_day_of_week":  "monday",
    "sim_district":     districts_list[0],
}
for _key, _default in SIM_DEFAULTS.items():
    st.session_state.setdefault(_key, _default)

tab1, tab2, tab3, tab4 = st.tabs([
    "Normalized risk map",
    "Risk simulator",
    "When is it riskiest?",
    "Trends over time",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Risk map
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        districts     = sorted(sensors["district"].dropna().unique())
        sel_districts = st.multiselect(
            "District", districts, default=default_districts,
            format_func=format_district,
        )

        min_acc = st.slider("Minimum recorded accidents at sensor", 1, 50, 3)

        df_map = sensors[sensors["n_accidents"] >= min_acc].copy()
        if sel_districts:
            df_map = df_map[df_map["district"].isin(sel_districts)]
        df_map = df_map.dropna(subset=["lat", "lon"])

        if not sel_districts:
            st.caption(
                f"Showing **all {len(districts)} districts**. Use the filter above to focus "
                "on specific ones — by default we pre-select the highest- and lowest-risk "
                "districts for a meaningful first look."
            )

        st.markdown(
            "**Risk index** = accident rate per unit of typical traffic at that location, "
            "divided by the Madrid-wide average rate. "
            "A value of 1 means average risk; above 1, the location has more accidents "
            "than its traffic volume would lead you to expect."
        )

        st.divider()
        st.markdown("**Linked to the Risk simulator tab**")
        highlight_on   = st.checkbox(
            "Highlight sensors matching the simulator's current profile", value=True
        )
        sim_is_weekend = st.session_state["sim_day_of_week"] in ("saturday", "sunday")
        st.caption(
            f"Current profile: **{format_district(st.session_state['sim_district'])}**, "
            f"**{st.session_state['sim_vehicle_cat']}**, "
            f"**{st.session_state['sim_vehicle_count'].lower()}**, "
            f"**{st.session_state['sim_weather']}**, around "
            f"**{st.session_state['sim_hour']}:00** on "
            f"**{st.session_state['sim_day_of_week'].capitalize()}**. "
            "Change it in the **Risk simulator** tab."
        )

        # ── Top / bottom risk metric cards ──────────────────────────────────
        st.divider()
        st.markdown("**Risk extremes (current filters)**")
        if df_map.empty:
            st.caption("No sensors match the current filters.")
        else:
            top_s    = df_map.loc[df_map["risk_index"].idxmax()]
            bottom_s = df_map.loc[df_map["risk_index"].idxmin()]
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(
                    "🔺 Top risk",
                    f"{top_s['risk_index']:.2f}",
                    help=(
                        f"Sensor #{int(top_s['id_sensor_cercano'])} · "
                        f"{format_district(top_s['district'])} · "
                        f"{int(top_s['n_accidents'])} accidents"
                    ),
                )
            with mcol2:
                st.metric(
                    "🔻 Lowest risk",
                    f"{bottom_s['risk_index']:.2f}",
                    help=(
                        f"Sensor #{int(bottom_s['id_sensor_cercano'])} · "
                        f"{format_district(bottom_s['district'])} · "
                        f"{int(bottom_s['n_accidents'])} accidents"
                    ),
                )

    matched_ids = set()
    if highlight_on:
        sim_single = "single" if st.session_state["sim_vehicle_count"] == "Single vehicle" else "multiple"
        match_mask = (
            (acc["distrito"] == st.session_state["sim_district"])
            & (acc["vehicle_cat"] == st.session_state["sim_vehicle_cat"])
            & (acc["single_vehicle"] == sim_single)
            & (acc["weather"] == st.session_state["sim_weather"])
            & (acc["day_of_week"] == st.session_state["sim_day_of_week"])
            & acc["hour"].between(st.session_state["sim_hour"] - 2, st.session_state["sim_hour"] + 2)
            & acc["id_sensor_cercano"].notna()
        )
        matched_ids = set(acc.loc[match_mask, "id_sensor_cercano"].unique())

    with col2:
        # Friendlier hover: use the district name as the closest available
        # proxy for a street address, together with an explicit sensor label.
        df_map = df_map.copy()
        df_map["display_district"] = df_map["district"].apply(format_district)
        df_map["sensor_label"] = (
            "Sensor #" + df_map["id_sensor_cercano"].astype(str)
            + "  —  " + df_map["display_district"]
        )

        fig = px.scatter_map(
            df_map,
            lat="lat", lon="lon",
            color="risk_index",
            size="n_accidents",
            size_max=18,
            color_continuous_scale="RdYlGn_r",
            range_color=[0.3, 3],
            hover_name="sensor_label",
            hover_data={
                "display_district": True,
                "n_accidents": True,
                "exposure": ":.0f",
                "risk_index": ":.2f",
                "lat": False, "lon": False,
                "sensor_label": False,
            },
            labels={
                "display_district":  "District (approx. area)",
                "n_accidents":        "Accidents recorded",
                "exposure":           "Cumul. traffic exposure",
                "risk_index":         "Risk index",
            },
            zoom=10.3,
            center={"lat": 40.43, "lon": -3.70},
            height=600,
        )
        fig.update_layout(map_style="open-street-map",
                          margin=dict(l=0, r=0, t=0, b=0))

        if highlight_on:
            ring_df = df_map[df_map["id_sensor_cercano"].isin(matched_ids)]
            if not ring_df.empty:
                ring = go.Scattermap(
                    lat=ring_df["lat"], lon=ring_df["lon"],
                    mode="markers",
                    marker=dict(size=24, color="black", opacity=0.55),
                    hoverinfo="skip",
                    name="Matches simulator profile",
                    showlegend=True,
                )
                fig.add_trace(ring)
                fig.data = tuple(reversed(fig.data))
                fig.update_layout(legend=dict(yanchor="top", y=0.99, x=0.01))
            else:
                st.caption(
                    "No sensors match this exact profile yet — try a wider hour range "
                    "(±2h is used for matching), or a different weather, day of week, or "
                    "vehicle type in the simulator."
                )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk locations")
    top = (
        df_map.assign(district=lambda d: d["district"].apply(format_district))
        .sort_values("risk_index", ascending=False)
        .head(15)[["district", "id_sensor_cercano", "n_accidents", "exposure", "risk_index"]]
        .rename(columns={"id_sensor_cercano":  "sensor id",
                         "n_accidents":         "accidents",
                         "exposure":            "cumulative exposure",
                         "risk_index":          "risk index"})
    )
    st.dataframe(top, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Risk simulator
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    pipe, importance, metrics = load_model()

    st.subheader("What type of accident is most likely under these conditions?")
    st.markdown(
        "Gradient boosting model trained on accidents from 2016–2022 and evaluated on "
        "2023–2024 (temporal hold-out). Given vehicle type, number of vehicles, time, "
        "weather, and district conditions, it estimates the probability of each accident "
        "type. "
        f"Overall test accuracy: **{metrics['accuracy_test']:.0%}** "
        f"(macro F1: {metrics['macro_f1_test']:.2f}). This is an exploratory model: "
        "the available features explain a significant portion of accident type, but "
        "unobserved factors (driver behaviour, road layout) also play a role."
    )
    st.caption(
        "💡 These settings also drive the highlighted sensors in the "
        "**Normalized risk map** tab, showing where on the map this profile has "
        "historically occurred."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        vehicle_cat = st.selectbox(
            "Vehicle type",
            ["car", "motorcycle", "truck", "bike", "bus", "other"],
            key="sim_vehicle_cat",
        )
        vehicle_count_type = st.radio(
            "Vehicles involved", ["Single vehicle", "Multiple vehicles"],
            key="sim_vehicle_count",
        )
    with c2:
        hour  = st.slider("Hour of day", 0, 23, key="sim_hour")
        month = st.slider("Month", 1, 12, key="sim_month")
        weather = st.selectbox(
            "Weather",
            ["clear", "cloudy", "light rain", "heavy rain", "snowing", "hailing", "unknown"],
            key="sim_weather",
        )
    with c3:
        district = st.selectbox(
            "District", districts_list, key="sim_district", format_func=format_district
        )
        day_of_week = st.selectbox(
            "Day of week",
            ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            key="sim_day_of_week",
        )

    single_vehicle = "single" if vehicle_count_type == "Single vehicle" else "multiple"

    # ── Plausibility check ───────────────────────────────────────────────────
    # Collisions almost never involve a single vehicle (rear-end / side impact
    # requires at least two parties). Flag this combination rather than
    # silently feeding the model an input pattern it has essentially never seen.
    if single_vehicle == "single":
        st.caption(
            "ℹ️ A single-vehicle accident with this profile is most consistent with a "
            "fall/rollover or object-impact type — collisions by definition involve "
            "more than one vehicle, so expect that probability to be very low below."
        )

    X = pd.DataFrame([{
        "hour":           hour,
        "month":          month,
        "single_vehicle": single_vehicle,
        "vehicle_cat":    vehicle_cat,
        "weather":        weather,
        "day_of_week":    day_of_week,
        "distrito":       district,
    }])

    proba    = pipe.predict_proba(X)[0]
    classes  = pipe.named_steps["clf"].classes_
    proba_df = pd.DataFrame({"accident type": classes, "probability": proba})
    proba_df = proba_df.sort_values("probability", ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Estimated probability by accident type**")
        fig_p = px.bar(proba_df, x="probability", y="accident type",
                       orientation="h", range_x=[0, 1])
        fig_p.update_layout(yaxis_title="", xaxis_title="probability",
                             height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_p, use_container_width=True)

    with col_b:
        st.markdown("**Feature importance in the model**")
        fig_i = px.bar(importance, x="importance", y="feature", orientation="h")
        fig_i.update_layout(yaxis_title="", xaxis_title="importance",
                             height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_i, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: When is it riskiest to travel by a given mode?
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    travel_risk = load_travel_risk()

    st.subheader("When is it riskiest to get around by each travel mode?")
    st.markdown(
        "For a chosen district and travel mode, this compares every time-slot / weather "
        "combination against **that mode's own typical rate in that district** — not the "
        "city-wide average. A value above 1 means accidents are more frequent than usual "
        "for that combination; below 1, less frequent. This is the same empirical Bayes "
        "shrinkage used in the risk map, applied within each district/mode pair so that "
        "combinations with very few recorded accidents aren't overstated."
    )

    col1, col2 = st.columns(2)
    with col1:
        t_district = st.selectbox(
            "District", sorted(travel_risk["distrito"].unique()), key="travel_district",
            format_func=format_district,
        )
    with col2:
        modes       = sorted(travel_risk["travel_type"].unique())
        default_idx = modes.index("pedestrian") if "pedestrian" in modes else 0
        t_mode      = st.selectbox("Travel mode", modes, index=default_idx, key="travel_mode")

    df_t = travel_risk[
        (travel_risk["distrito"] == t_district) & (travel_risk["travel_type"] == t_mode)
    ].dropna(subset=["risk_index"])

    if df_t.empty or df_t["n_accidents"].sum() == 0:
        st.info("Not enough recorded accidents for this district / travel mode combination.")
    else:
        time_order    = ["night (0-5h)", "morning (6-11h)", "afternoon (12-18h)", "evening (19-23h)"]
        weather_order = ["clear", "cloudy", "light rain", "heavy rain",
                         "snowing", "hailing", "unknown"]

        pivot = df_t.pivot(index="weather", columns="time_slot", values="risk_index")
        pivot = pivot.reindex(
            index=[w for w in weather_order if w in pivot.index],
            columns=[t for t in time_order   if t in pivot.columns],
        )

        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r", range_color=[0.3, 3],
            text_auto=".2f", aspect="auto",
            labels=dict(x="time slot", y="weather", color="risk index"),
        )
        fig_heat.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Blank cells mean no accidents of this type were recorded under that "
            "combination in the data."
        )

        best = df_t.sort_values("risk_index", ascending=False).iloc[0]
        st.success(
            f"Highest risk for **{t_mode}** in **{format_district(t_district)}**: "
            f"**{best['weather']}** during **{best['time_slot']}** — risk index "
            f"**{best['risk_index']:.2f}** ({int(best['n_accidents'])} accidents recorded)."
        )
        st.caption(
            f"Based on {int(df_t['n_accidents'].sum())} recorded accidents in total for "
            "this selection."
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Trends over time
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Risk index over time by district")
    st.markdown(
        "The same normalized risk index from the map tab, computed year by year for each "
        "district. This shows whether an area has improved or worsened relative to its "
        "own traffic levels over time."
    )

    districts = sorted(dist_year["distrito"].unique())
    default   = ["centro", "salamanca", "puente de vallecas"]
    default   = [d for d in default if d in districts]
    sel       = st.multiselect("Districts to compare", districts, default=default,
                               format_func=format_district)

    if sel:
        df_plot = dist_year[dist_year["distrito"].isin(sel)].copy()
        df_plot["district"] = df_plot["distrito"].apply(format_district)
        fig = px.line(df_plot, x="year", y="risk_index", color="district", markers=True,
                      labels={"year": "year", "risk_index": "risk index",
                               "district": "district"})
        fig.add_hline(y=1, line_dash="dash", line_color="gray",
                      annotation_text="Madrid average")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one district.")

    st.subheader("District ranking by year")
    year_sel = st.slider("Year", int(dist_year["year"].min()),
                         int(dist_year["year"].max()),
                         int(dist_year["year"].max()))
    df_rank = (
        dist_year[dist_year["year"] == year_sel]
        .sort_values("risk_index", ascending=False)
        .copy()
    )
    df_rank["district"] = df_rank["distrito"].apply(format_district)
    fig_rank = px.bar(df_rank, x="district", y="risk_index",
                      labels={"district": "district", "risk_index": "risk index"})
    fig_rank.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_rank.update_layout(height=400)
    st.plotly_chart(fig_rank, use_container_width=True)
