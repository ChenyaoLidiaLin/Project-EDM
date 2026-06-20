import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = "data"

st.set_page_config(page_title="Madrid Road Risk", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_sensors():
    return pd.read_parquet(f"{DATA_DIR}/sensor_risk.parquet")


@st.cache_data
def load_district_yearly():
    return pd.read_parquet(f"{DATA_DIR}/district_risk_yearly.parquet")


@st.cache_data
def load_districts():
    """Load only the district column — avoids reading all of accidents_clean
    just to populate dropdowns."""
    df = pd.read_parquet(f"{DATA_DIR}/accidents_clean.parquet",
                         columns=["distrito"])
    return sorted(df["distrito"].dropna().unique())


@st.cache_data
def load_travel_risk():
    return pd.read_parquet(f"{DATA_DIR}/travel_risk.parquet")


@st.cache_data
def load_acc_for_matching():
    """Load only the columns needed for the simulator → map highlighting.

    Kept separate from load_accidents() so the full parquet isn't read just to
    populate the sensor-matching overlay — column projection keeps memory low
    and Streamlit's cache avoids re-reading on every widget interaction.
    """
    df = pd.read_parquet(
        f"{DATA_DIR}/accidents_clean.parquet",
        columns=["distrito", "weather", "time_slot", "hour",
                 "id_sensor_cercano", "tipo_vehiculo"],
    )

    def _vcat(v):
        if pd.isna(v): return "unknown"
        v = str(v)
        if "motocicleta" in v or "ciclomotor" in v: return "motorcycle"
        if "bicicleta"   in v or "vmu"        in v: return "bike"
        if "autobus"     in v or "autocar"     in v: return "bus"
        if "camion"      in v or "furgon"      in v or "tractocamion" in v: return "truck"
        if "turismo"     in v or "todo terreno" in v: return "car"
        return "other"

    df["vehicle_cat"] = df["tipo_vehiculo"].apply(_vcat)
    return df


@st.cache_resource
def load_model():
    pipe       = joblib.load(f"{DATA_DIR}/accident_type_model.joblib")
    importance = pd.read_parquet(f"{DATA_DIR}/feature_importance.parquet")
    with open(f"{DATA_DIR}/model_metrics.json") as f:
        metrics = json.load(f)
    return pipe, importance, metrics


@st.cache_data
def compute_default_district(sensors_df: pd.DataFrame, min_n: int = 5) -> str:
    """Return the single highest-risk district (mean risk index across its sensors).

    Used as the shared default for both the map multiselect and the simulator
    district dropdown so both tabs open on the same, most illustrative example.
    """
    agg = (
        sensors_df[sensors_df["n_accidents"] >= min_n]
        .groupby("district")["risk_index"].mean()
    )
    if agg.empty:
        return sensors_df["district"].dropna().iloc[0]
    return agg.idxmax()


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.title("Madrid Road Risk: accident rates normalized by traffic volume")
st.markdown(
    "Analysis of Madrid traffic accidents (2016–2024) matched with data from the nearest "
    "traffic sensors (flow, occupancy, mean speed). The goal is to find locations where "
    "**more accidents happen than the local traffic volume would predict**, not just the "
    "spots with the highest raw accident count."
)

sensors        = load_sensors()
dist_year      = load_district_yearly()
districts_list = load_districts()
default_district = compute_default_district(sensors)   # single highest-risk district

# ---------------------------------------------------------------------------
# KPI bar
# ---------------------------------------------------------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    total_acc = pd.read_parquet(f"{DATA_DIR}/accidents_clean.parquet",
                                columns=["num_expediente"]).shape[0]
    st.metric(
        "Total accidents logged",
        f"{total_acc:,}",
        help="Madrid 2016–2024, deduplicated by case number.",
    )

with kpi2:
    eligible  = sensors[sensors["n_accidents"] >= 5]
    top_s     = eligible.loc[eligible["risk_index"].idxmax()]
    st.metric(
        "Highest-risk sensor",
        f"index {top_s['risk_index']:.2f}",
        help=(
            f"Sensor #{int(top_s['id_sensor_cercano'])} · "
            f"{format_district(top_s['district'])} · "
            f"{int(top_s['n_accidents'])} accidents. "
            "See the Normalized risk map tab."
        ),
    )

with kpi3:
    latest_year      = int(dist_year["year"].max())
    top_dist_row     = dist_year[dist_year["year"] == latest_year].loc[
        dist_year[dist_year["year"] == latest_year]["risk_index"].idxmax()
    ]
    st.metric(
        f"Riskiest district ({latest_year})",
        format_district(top_dist_row["distrito"]),
        f"index {top_dist_row['risk_index']:.2f}",
        help=f"{int(top_dist_row['n_accidents'])} accidents in {latest_year}.",
    )

with kpi4:
    st.metric(
        "Traffic sensors monitored",
        f"{len(sensors):,}",
        help="Sensors with at least one matched accident nearby.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Session-state defaults for simulator (used by the map tab for highlighting)
# ---------------------------------------------------------------------------

SIM_DEFAULTS = {
    "sim_hour":         12,
    "sim_month":        6,
    "sim_vehicle":      "car",
    "sim_single":       "multiple",
    "sim_weather":      "clear",
    "sim_dow":          "friday",
    "sim_district":     default_district,
}
for _k, _v in SIM_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


tab1, tab2, tab3, tab4 = st.tabs([
    "Normalized risk map",
    "Risk simulator",
    "When is it riskiest?",
    "Trends over time",
])


# ---------------------------------------------------------------------------
# TAB 1: Risk map
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        all_districts_map = sorted(sensors["district"].dropna().unique())
        sel_districts = st.multiselect(
            "District", all_districts_map, default=[default_district],
            format_func=format_district,
        )
        min_acc = st.slider("Minimum recorded accidents at sensor", 1, 50, 3)

        df_map = sensors[sensors["n_accidents"] >= min_acc].copy()
        if sel_districts:
            df_map = df_map[df_map["district"].isin(sel_districts)]
        df_map = df_map.dropna(subset=["lat", "lon"])

        st.markdown(
            "**Risk index** = accident rate per unit of typical traffic at that location, "
            "divided by the Madrid-wide average rate. "
            "1 = city average. Above 1 → more accidents than traffic alone would explain."
        )

        st.divider()
        st.markdown("**Linked to the Risk simulator tab**")
        highlight_on = st.checkbox(
            "Highlight sensors matching the simulator's current profile", value=True
        )
        st.caption(
            f"Current profile: **{format_district(st.session_state['sim_district'])}**, "
            f"**{st.session_state['sim_weather']}**, "
            f"**{st.session_state['sim_vehicle']}**, "
            f"hour {st.session_state['sim_hour']}h. "
            "Change it in the **Risk simulator** tab."
        )

        st.divider()
        st.markdown("**Risk extremes (current filters)**")
        if df_map.empty:
            st.caption("No sensors match the current filters.")
        else:
            top_s2    = df_map.loc[df_map["risk_index"].idxmax()]
            bottom_s2 = df_map.loc[df_map["risk_index"].idxmin()]
            mc1, mc2  = st.columns(2)
            with mc1:
                st.metric("🔺 Top risk", f"{top_s2['risk_index']:.2f}",
                          help=f"{format_district(top_s2['district'])} · "
                               f"{int(top_s2['n_accidents'])} accidents")
            with mc2:
                st.metric("🔻 Lowest risk", f"{bottom_s2['risk_index']:.2f}",
                          help=f"{format_district(bottom_s2['district'])} · "
                               f"{int(bottom_s2['n_accidents'])} accidents")

    # Build matched sensor IDs from simulator state
    matched_ids = set()
    if highlight_on and not df_map.empty:
        acc_match = load_acc_for_matching()
        hour_tol = 2
        match_mask = (
            (acc_match["distrito"]      == st.session_state["sim_district"])
            & (acc_match["weather"]     == st.session_state["sim_weather"])
            & (acc_match["vehicle_cat"] == st.session_state["sim_vehicle"])
            & acc_match["hour"].between(
                st.session_state["sim_hour"] - hour_tol,
                st.session_state["sim_hour"] + hour_tol,
            )
            & acc_match["id_sensor_cercano"].notna()
        )
        matched_ids = set(acc_match.loc[match_mask, "id_sensor_cercano"].unique())

    with col2:
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
                "n_accidents":       True,
                "exposure":          ":.0f",
                "risk_index":        ":.2f",
                "lat": False, "lon": False, "sensor_label": False,
            },
            labels={
                "display_district": "District (approx. area)",
                "n_accidents":       "Accidents recorded",
                "exposure":          "Cumul. traffic exposure",
                "risk_index":        "Risk index",
            },
            zoom=10.3,
            center={"lat": 40.43, "lon": -3.70},
            height=600,
        )
        fig.update_layout(map_style="open-street-map",
                          margin=dict(l=0, r=0, t=0, b=0))

        if highlight_on and matched_ids:
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
                    "No sensors match this exact profile — try different settings "
                    "in the simulator tab."
                )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk locations")
    top_tbl = (
        df_map.sort_values("risk_index", ascending=False)
        .head(15)[["display_district", "id_sensor_cercano", "n_accidents",
                   "exposure", "risk_index"]]
        .rename(columns={
            "display_district":  "district",
            "id_sensor_cercano": "sensor id",
            "n_accidents":       "accidents",
            "exposure":          "cumulative exposure",
            "risk_index":        "risk index",
        })
    )
    st.dataframe(top_tbl, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2: Risk simulator
# ---------------------------------------------------------------------------
with tab2:
    pipe, importance, metrics = load_model()

    st.subheader("What type of accident is most likely under these conditions?")
    st.markdown(
        "Gradient boosting model trained on 2016–2022 accidents, evaluated on "
        "2023–2024 (temporal hold-out). Given vehicle type, time, weather, and "
        "district, it estimates the probability of each accident type. "
        f"Test accuracy: **{metrics['accuracy_test']:.0%}** "
        f"(macro F1: {metrics['macro_f1_test']:.2f})."
    )
    st.caption(
        "💡 These settings also drive the highlighted sensors in the "
        "**Normalized risk map** tab."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        vehicle = st.selectbox(
            "Vehicle type",
            ["car", "motorcycle", "bike", "truck", "bus", "other", "unknown"],
            key="sim_vehicle",
        )
        single = st.radio(
            "Vehicles involved",
            ["single", "multiple"],
            index=1,
            key="sim_single",
            horizontal=True,
        )
        hour = st.slider("Hour of day", 0, 23, key="sim_hour")
    with c2:
        month = st.slider("Month", 1, 12, key="sim_month")
        weather = st.selectbox(
            "Weather",
            ["clear", "cloudy", "light rain", "heavy rain",
             "snowing", "hailing", "unknown"],
            key="sim_weather",
        )
        dow = st.selectbox(
            "Day of week",
            ["monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday"],
            key="sim_dow",
        )
    with c3:
        district_sim = st.selectbox(
            "District", districts_list,
            key="sim_district",
            format_func=format_district,
        )

    X = pd.DataFrame([{
        "hour":           st.session_state["sim_hour"],
        "month":          st.session_state["sim_month"],
        "single_vehicle": st.session_state["sim_single"],
        "vehicle_cat":    st.session_state["sim_vehicle"],
        "weather":        st.session_state["sim_weather"],
        "day_of_week":    st.session_state["sim_dow"],
        "distrito":       st.session_state["sim_district"],
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
        fig_i = px.bar(importance.sort_values("importance"),
                       x="importance", y="feature", orientation="h")
        fig_i.update_layout(yaxis_title="", xaxis_title="importance",
                             height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_i, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: When is it riskiest?
# ---------------------------------------------------------------------------
with tab3:
    travel_risk = load_travel_risk()

    st.subheader("When is it riskiest to get around by each travel mode?")
    st.markdown(
        "For a chosen district and travel mode, this shows which time-slot / weather "
        "combination produces the highest **accident rate per unit of traffic volume** — "
        "not just the highest raw accident count. "
        "The denominator is the mean traffic flow recorded at nearby sensors during each "
        "time slot (the same exposure proxy used in the risk map), so the index captures "
        "when accidents happen *more than traffic alone would predict*. "
        "For example: fewer cars drive in heavy rain, so a raw count of 3 accidents at "
        "3 am in a storm is far more alarming per unit of flow than 3 accidents on a clear "
        "Friday afternoon. "
        "The same Marshall (1991) empirical Bayes shrinkage is applied within each "
        "district / mode pair, so cells with very few observations are pulled toward 1.0 "
        "rather than overstated. "
        "**Note:** the traffic-flow denominator reflects all vehicles at the sensor, not "
        "only cyclists or pedestrians — there are no mode-specific flow counters in the "
        "dataset. The index is therefore best read as a *relative comparison across time "
        "slots and weather* for the same mode and district, not as an absolute rate."
    )

    col1, col2 = st.columns(2)
    with col1:
        t_district = st.selectbox(
            "District", sorted(travel_risk["distrito"].unique()),
            key="travel_district", format_func=format_district,
        )
    with col2:
        modes       = sorted(travel_risk["travel_type"].unique())
        default_idx = modes.index("pedestrian") if "pedestrian" in modes else 0
        t_mode      = st.selectbox("Travel mode", modes, index=default_idx, key="travel_mode")

    df_t = travel_risk[
        (travel_risk["distrito"] == t_district) &
        (travel_risk["travel_type"] == t_mode)
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
            f"**{best['weather']}** during **{best['time_slot']}** — "
            f"risk index **{best['risk_index']:.2f}** "
            f"({int(best['n_accidents'])} accidents · "
            f"mean traffic exposure {best['exposure']:.0f} veh/h)."
        )
        st.caption(
            f"Based on {int(df_t['n_accidents'].sum())} recorded accidents in total "
            "for this selection. Risk index = accident rate per unit of sensor-measured "
            "traffic flow, divided by the mean rate across all time slots for this "
            "district / mode pair."
        )


# ---------------------------------------------------------------------------
# TAB 4: Trends over time
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Risk index over time by district")
    st.markdown(
        "The same normalized risk index from the map tab, computed year by year for each "
        "district. This shows whether an area has improved or worsened relative to its "
        "own traffic levels over time."
    )

    districts_trend = sorted(dist_year["distrito"].unique())
    default_trend   = ["centro", "salamanca", "puente de vallecas"]
    default_trend   = [d for d in default_trend if d in districts_trend]
    sel = st.multiselect("Districts to compare", districts_trend, default=default_trend,
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
    year_sel = st.slider(
        "Year", int(dist_year["year"].min()),
        int(dist_year["year"].max()), int(dist_year["year"].max()),
    )
    df_rank = (
        dist_year[dist_year["year"] == year_sel]
        .sort_values("risk_index", ascending=False)
        .copy()
    )
    df_rank["district"] = df_rank["distrito"].apply(format_district)
    fig_rank = px.bar(df_rank, x="district", y="risk_index",
                      labels={"district": "district", "risk_index": "risk index"})
    fig_rank.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_rank.update_layout(height=400, xaxis_tickangle=-40)
    st.plotly_chart(fig_rank, use_container_width=True)
