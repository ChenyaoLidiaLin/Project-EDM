import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = "data"

st.set_page_config(page_title="Riesgo vial de Madrid", layout="wide")


@st.cache_data
def load_sensores():
    return pd.read_parquet(f"{DATA_DIR}/sensores_riesgo.parquet")


@st.cache_data
def load_distrito_anual():
    return pd.read_parquet(f"{DATA_DIR}/distrito_riesgo_anual.parquet")


@st.cache_data
def load_accidentes():
    return pd.read_parquet(f"{DATA_DIR}/accidentes_clean.parquet")


@st.cache_resource
def load_model():
    pipe = joblib.load(f"{DATA_DIR}/modelo_tipo_accidente.joblib")
    importance = pd.read_parquet(f"{DATA_DIR}/feature_importance.parquet")
    with open(f"{DATA_DIR}/metricas_modelo.json") as f:
        metrics = json.load(f)
    return pipe, importance, metrics


st.title("Riesgo vial de Madrid: accidentes normalizados por tráfico")
st.markdown(
    "Análisis de accidentes de tráfico de Madrid (2016-2024) cruzados con datos "
    "de los sensores de tráfico más cercanos. La idea central es identificar "
    "puntos donde ocurren **más accidentes de los que el volumen de tráfico "
    "habitual haría esperar**, no simplemente los puntos con más accidentes."
)

tab1, tab2, tab3 = st.tabs([
    "Mapa de riesgo normalizado",
    "Simulador de riesgo",
    "Evolución temporal",
])

# ---------------------------------------------------------------------------
# TAB 1: Mapa de riesgo
# ---------------------------------------------------------------------------
with tab1:
    sensores = load_sensores()

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Filtros")
        distritos = sorted(sensores["distrito"].dropna().unique())
        sel_distritos = st.multiselect("Distrito", distritos, default=[])

        min_acc = st.slider("Mínimo de accidentes registrados en el sensor",
                             1, 50, 3)

        st.markdown(
            "**Índice de riesgo** = tasa de accidentes por unidad de tráfico "
            "habitual en ese punto, dividida por la tasa media de Madrid. "
            "Un valor de 1 indica riesgo medio; por encima de 1, el punto "
            "tiene más accidentes de los que su tráfico haría esperar."
        )

    df_map = sensores[sensores["n_accidentes"] >= min_acc].copy()
    if sel_distritos:
        df_map = df_map[df_map["distrito"].isin(sel_distritos)]
    df_map = df_map.dropna(subset=["lat", "lon"])

    with col2:
        fig = px.scatter_map(
            df_map,
            lat="lat", lon="lon",
            color="indice_riesgo",
            size="n_accidentes",
            size_max=18,
            color_continuous_scale="RdYlGn_r",
            range_color=[0.3, 3],
            hover_data={"distrito": True, "n_accidentes": True,
                        "indice_riesgo": ":.2f", "lat": False, "lon": False},
            zoom=10.3,
            center={"lat": 40.43, "lon": -3.70},
            height=600,
        )
        fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Puntos con mayor índice de riesgo")
    top = (
        df_map.sort_values("indice_riesgo", ascending=False)
        .head(15)[["distrito", "n_accidentes", "exposicion", "indice_riesgo"]]
        .rename(columns={"n_accidentes": "accidentes",
                          "exposicion": "exposición acumulada",
                          "indice_riesgo": "índice de riesgo"})
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2: Simulador de riesgo
# ---------------------------------------------------------------------------
with tab2:
    pipe, importance, metrics = load_model()
    acc = load_accidentes()

    st.subheader("¿Qué tipo de accidente es más probable en estas condiciones?")
    st.markdown(
        "Modelo de bosque aleatorio entrenado con accidentes de 2016-2022 y "
        "validado con 2023-2024 (validación temporal). Dadas unas condiciones "
        "de tráfico, meteorología, hora y distrito, estima la probabilidad de "
        "cada tipo de accidente. "
        f"Precisión global en el conjunto de test: **{metrics['accuracy_test']:.0%}** "
        f"(F1 macro: {metrics['macro_f1_test']:.2f}). Es un modelo exploratorio: "
        "el contexto de tráfico explica solo una parte del tipo de accidente, "
        "pero permite comparar cómo cambia el riesgo relativo entre escenarios."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        intensidad = st.slider("Intensidad de tráfico (veh/h)", 0, 6000, 1200, step=50)
        ocupacion = st.slider("Ocupación (%)", 0.0, 50.0, 5.0, step=0.5)
        vmed = st.slider("Velocidad media (km/h)", 0.0, 100.0, 20.0, step=1.0)
    with c2:
        meteo = st.selectbox(
            "Meteorología",
            ["despejado", "nublado", "lluvia debil", "lluvia intensa",
             "nevando", "granizando", "se desconoce"],
        )
        bloque = st.selectbox(
            "Franja horaria",
            ["madrugada (0-5h)", "manana (6-11h)", "tarde (12-18h)", "noche (19-23h)"],
        )
        mes = st.slider("Mes", 1, 12, 6)
    with c3:
        distrito = st.selectbox("Distrito", sorted(acc["distrito"].dropna().unique()))
        finde = st.radio("Tipo de día", ["laborable", "fin de semana / festivo"])

    finde_val = "1" if finde == "fin de semana / festivo" else "0"

    X = pd.DataFrame([{
        "intensidad": intensidad,
        "ocupacion": ocupacion,
        "vmed": vmed,
        "mes": mes,
        "estado_meteorologico": meteo,
        "bloque_horario": bloque,
        "es_finde_festivo": finde_val,
        "distrito": distrito,
    }])

    proba = pipe.predict_proba(X)[0]
    classes = pipe.named_steps["clf"].classes_
    proba_df = pd.DataFrame({"tipo_accidente": classes, "probabilidad": proba})
    proba_df = proba_df.sort_values("probabilidad", ascending=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Probabilidad estimada por tipo de accidente**")
        fig_p = px.bar(proba_df, x="probabilidad", y="tipo_accidente",
                       orientation="h", range_x=[0, 1])
        fig_p.update_layout(yaxis_title="", xaxis_title="probabilidad",
                             height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_p, use_container_width=True)

    with col_b:
        st.markdown("**Importancia de las variables en el modelo**")
        fig_i = px.bar(importance, x="importancia", y="variable", orientation="h")
        fig_i.update_layout(yaxis_title="", height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_i, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Evolución temporal
# ---------------------------------------------------------------------------
with tab3:
    dist_year = load_distrito_anual()

    st.subheader("Evolución del índice de riesgo por distrito")
    st.markdown(
        "Mismo índice de riesgo normalizado de la pestaña anterior, calculado "
        "año a año por distrito. Permite ver si una zona ha mejorado o "
        "empeorado con el tiempo en relación con su propio nivel de tráfico."
    )

    distritos = sorted(dist_year["distrito"].unique())
    default = ["centro", "salamanca", "puente de vallecas"]
    default = [d for d in default if d in distritos]
    sel = st.multiselect("Distritos a comparar", distritos, default=default)

    if sel:
        df_plot = dist_year[dist_year["distrito"].isin(sel)]
        fig = px.line(df_plot, x="anio", y="indice_riesgo", color="distrito", markers=True)
        fig.add_hline(y=1, line_dash="dash", line_color="gray",
                       annotation_text="media de Madrid")
        fig.update_layout(height=500, yaxis_title="índice de riesgo", xaxis_title="año")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona al menos un distrito.")

    st.subheader("Ranking de distritos por año")
    anio_sel = st.slider("Año", int(dist_year["anio"].min()), int(dist_year["anio"].max()),
                          int(dist_year["anio"].max()))
    df_rank = (
        dist_year[dist_year["anio"] == anio_sel]
        .sort_values("indice_riesgo", ascending=False)
    )
    fig_rank = px.bar(df_rank, x="distrito", y="indice_riesgo")
    fig_rank.add_hline(y=1, line_dash="dash", line_color="gray")
    fig_rank.update_layout(height=400, xaxis_title="", yaxis_title="índice de riesgo")
    st.plotly_chart(fig_rank, use_container_width=True)
