import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = "data"

st.set_page_config(page_title="Madrid Road Risk", layout="wide")


@st.cache_data
def load_sensors():
    return pd.read_parquet(f"{DATA_DIR}/sensor_risk.parquet")


@st.cache_data
def load_district_yearly():
    return pd.read_parquet(f"{DATA_DIR}/district_risk_yearly.parquet")


@st.cache_data
def load_accidents():
    return pd.read_parquet(f"{DATA_DIR}/accidents_clean.parquet")


@st.cache_resource
def load_model():
    pipe     = joblib.load(f"{DATA_DIR}/accident_type_model.joblib")
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

tab1, tab2, tab3 = st.tabs([
    "Normalized risk map",
    "Risk simulator",
    "Trends over time",
])

# TAB 1: Risk map
with tab1:
    sensors = load_sensors()

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filters")
        districts = sorted(sensors["district"].dropna().unique())
        sel_districts = st.multiselect("District", districts, default=[])

        min_acc = st.slider("Minimum recorded accidents at sensor", 1, 50, 3)

        st.markdown(
            "**Risk index** = accident rate per unit of typical traffic at that location, "
            "divided by the Madrid-wide average rate. "
            "A value of 1 means average risk; above 1, the location has more accidents "
            "than its traffic volume would lead you to expect."
        )

    df_map = sensors[sensors["n_accidents"] >= min_acc].copy()
    if sel_districts:
        df_map = df_map[df_map["district"].isin(sel_districts)]
    df_map = df_map.dropna(subset=["lat", "lon"])

    with col2:
        fig = px.scatter_map(
            df_map,
            lat="lat", lon="lon",
            color="risk_index",
            size="n_accidents",
            size_max=18,
            color_continuous_scale="RdYlGn_r",
            range_color=[0.3, 3],
            hover_data={"district": True, "n_accidents": True,
                        "risk_index": ":.2f", "lat": False, "lon": False},
            zoom=10.3,
            center={"lat": 40.43, "lon": -3.70},
            height=600,
        )
        fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk locations")
    top = (
        df_map.sort_values("risk_index", ascending=False)
        .head(15)[["district", "n_accidents", "exposure", "risk_index"]]
        .rename(columns={"n_accidents": "accidents",
                         "exposure": "cumulative exposure",
                         "risk_index": "risk index"})
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

# TAB 2: Risk simulator
with tab2:
    pipe, importance, metrics = load_model()
    acc = load_accidents()

    st.subheader("What type of accident is most likely under these conditions?")
    st.markdown(
        "Random forest model trained on accidents from 2016-2022 and evaluated on "
        "2023-2024 (temporal hold-out). Given a set of traffic, weather, time, and "
        "district conditions, it estimates the probability of each accident type. "
        f"Overall test accuracy: **{metrics['accuracy_test']:.0%}** "
        f"(macro F1: {metrics['macro_f1_test']:.2f}). This is an exploratory model: "
        "traffic context alone explains only part of what drives accident type, but it "
        "lets you compare how relative risk shifts across different scenarios."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        flow    = st.slider("Traffic flow (veh/h)", 0, 6000, 1200, step=50)
        occ     = st.slider("Occupancy (%)", 0.0, 50.0, 5.0, step=0.5)
        speed   = st.slider("Mean speed (km/h)", 0.0, 100.0, 20.0, step=1.0)
    with c2:
        weather   = st.selectbox("Weather",
                                 ["clear", "cloudy", "light rain", "heavy rain",
                                  "snowing", "hailing", "unknown"])
        time_slot = st.selectbox("Time slot",
                                 ["night (0-5h)", "morning (6-11h)",
                                  "afternoon (12-18h)", "evening (19-23h)"])
        month = st.slider("Month", 1, 12, 6)
    with c3:
        district   = st.selectbox("District", sorted(acc["distrito"].dropna().unique()))
        day_type   = st.radio("Day type", ["Weekday", "Weekend / public holiday"])
        is_weekend = "1" if day_type == "Weekend / public holiday" else "0"

    X = pd.DataFrame([{
        "intensidad":          flow,
        "ocupacion":           occ,
        "vmed":                speed,
        "month":               month,
        "weather":             weather,
        "time_slot":           time_slot,
        "is_weekend_holiday":  is_weekend,
        "distrito":            district,
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

# TAB 3: Trends over time
with tab3:
    dist_year = load_district_yearly()

    st.subheader("Risk index over time by district")
    st.markdown(
        "The same normalized risk index from the map tab, computed year by year for each "
        "district. This shows whether an area has improved or worsened relative to its "
        "own traffic levels over time."
    )

    districts = sorted(dist_year["distrito"].unique())
    default = ["centro", "salamanca", "puente de vallecas"]
    default = [d for d in default if d in districts]
    sel = st.multiselect("Districts to compare", districts, default=default)

    if sel:
        df_plot = dist_year[dist_year["distrito"].isin(sel)]
        fig = px.line(df_plot, x="year", y="risk_index", color="distrito", markers=True,
                      labels={"year": "year", "risk_index": "risk index",
                               "distrito": "district"})
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
    )
    fig_rank = px.bar(df_rank, x="distrito", y="risk_index",
                      labels={"distrito": "district", "risk_index": "risk index"})
    fig_rank.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_rank.update_layout(height=400)
    st.plotly_chart(fig_rank, use_container_width=True)
