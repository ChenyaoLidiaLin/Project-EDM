"""
Limpieza y preparación de datos de accidentes de tráfico de Madrid (2016-2024)
con datos de tráfico del sensor más cercano.

Genera:
  data/accidentes_clean.parquet   -> dataset a nivel de accidente (deduplicado por expediente)
  data/sensores_riesgo.parquet    -> índice de riesgo por sensor (para el mapa)
  data/distrito_riesgo_anual.parquet -> índice de riesgo por distrito y año (series temporales)
"""

import csv
import unicodedata
import pandas as pd
import numpy as np
from pyproj import Transformer

RAW_PATH = "/mnt/user-data/uploads/accidentes_con_trafico_final__1_.csv"
OUT_DIR = "/home/claude/project/data"

# ---------------------------------------------------------------------------
# 1. Lectura robusta del CSV (el fichero tiene un sufijo ";;" en cada línea y
#    algunas filas vienen con todo el registro entre comillas y comillas
#    internas duplicadas, propio de una doble escritura CSV).
# ---------------------------------------------------------------------------

def fix_line(line: str) -> str:
    line = line.rstrip("\r\n")
    if line.endswith(";;"):
        line = line[:-2]
    if line.startswith('"') and line.endswith('"'):
        inner = line[1:-1]
        inner = inner.replace('""', '"')
        return inner
    return line


def load_raw() -> pd.DataFrame:
    with open(RAW_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    fixed = [fix_line(lines[0])] + [fix_line(l) for l in lines[1:]]

    reader = csv.reader(fixed)
    header = next(reader)
    rows = [r for r in reader if len(r) == len(header)]
    df = pd.DataFrame(rows, columns=header)
    return df


# ---------------------------------------------------------------------------
# 2. Limpieza de columnas
# ---------------------------------------------------------------------------

def strip_accents(s):
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def assign_bloque_horario(hora):
    h = pd.to_numeric(hora.str.split(":").str[0], errors="coerce")
    bins = [-1, 5, 11, 18, 23]
    labels = ["madrugada (0-5h)", "manana (6-11h)", "tarde (12-18h)", "noche (19-23h)"]
    return pd.cut(h, bins=bins, labels=labels)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"estado_meteorológico": "estado_meteorologico"})

    num_cols = ["coordenada_x_utm", "coordenada_y_utm", "es_festivo",
                 "id_sensor_cercano", "intensidad", "ocupacion", "vmed"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")

    df["fecha"] = pd.to_datetime(df["fecha"], format="%m/%d/%Y", errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month

    text_cols = ["dia_semana", "distrito", "tipo_accidente", "tipo_vehiculo",
                  "sexo", "rango_edad", "estado_meteorologico"]
    for c in text_cols:
        df[c] = (df[c].astype(str).str.strip().str.lower()
                 .replace({"nan": np.nan, "": np.nan}))
        df[c] = df[c].apply(strip_accents)

    df["rango_edad"] = df["rango_edad"].str.replace("anos", "años", regex=False)

    df["bloque_horario"] = assign_bloque_horario(df["hora"])
    df["es_finde_festivo"] = (
        (df["es_festivo"] == 1) | df["dia_semana"].isin(["sabado", "domingo"])
    ).astype("Int64")

    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    mask = df["coordenada_x_utm"].notna() & df["coordenada_y_utm"].notna()
    lon, lat = transformer.transform(
        df.loc[mask, "coordenada_x_utm"].values,
        df.loc[mask, "coordenada_y_utm"].values,
    )
    df.loc[mask, "lon"] = lon
    df.loc[mask, "lat"] = lat

    return df


# ---------------------------------------------------------------------------
# 3. Dataset a nivel de accidente (un registro por expediente)
# ---------------------------------------------------------------------------

def build_accident_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["num_expediente"].notna() & df["fecha"].notna()]

    n_vehiculos = df.groupby("num_expediente")["tipo_vehiculo"].count()
    acc = df.drop_duplicates(subset="num_expediente", keep="first").copy()
    acc["n_vehiculos"] = acc["num_expediente"].map(n_vehiculos)

    def agrupa_tipo(t):
        if pd.isna(t):
            return np.nan
        if "atropello" in t:
            return "atropello"
        if "colision" in t or "alcance" in t:
            return "colision"
        if "caida" in t or "vuelco" in t or "despen" in t:
            return "caida_vuelco"
        if "choque" in t:
            return "choque_objeto"
        return "otros"

    acc["tipo_accidente_grp"] = acc["tipo_accidente"].apply(agrupa_tipo)

    keep = ["fecha", "anio", "mes", "dia_semana", "hora", "bloque_horario",
            "es_finde_festivo", "distrito", "num_expediente", "tipo_accidente",
            "tipo_accidente_grp", "tipo_vehiculo", "n_vehiculos",
            "estado_meteorologico", "lon", "lat",
            "id_sensor_cercano", "intensidad", "ocupacion", "vmed"]
    return acc[keep]


# ---------------------------------------------------------------------------
# 4. Proxy de exposicion al trafico por sensor / bloque horario / tipo de dia
# ---------------------------------------------------------------------------

def build_exposure(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["intensidad"].notna() & df["id_sensor_cercano"].notna()]
    exposure = (
        base.groupby(["id_sensor_cercano", "bloque_horario", "es_finde_festivo"],
                      observed=True)["intensidad"]
        .mean()
        .reset_index()
        .rename(columns={"intensidad": "exposicion"})
    )
    return exposure


# ---------------------------------------------------------------------------
# 5. Indice de riesgo: estimador empirico de Bayes (shrinkage tipo Marshall)
#
#    El dataset solo contiene filas de accidentes (no hay combinaciones
#    sensor/franja sin accidente), por lo que un GLM con offset clasico queda
#    mal calibrado. En su lugar, a cada accidente se le asigna una
#    "exposicion" = intensidad media historica observada en ese sensor, esa
#    franja horaria y tipo de dia (proxy del trafico habitual de ese
#    contexto). La tasa cruda de cada sensor es accidentes / suma de
#    exposiciones. Esa tasa es muy inestable cuando hay pocos accidentes, asi
#    que se aplica un encogimiento empirico de Bayes (Marshall, 1991) hacia
#    la media global, con un peso proporcional a la exposicion acumulada.
# ---------------------------------------------------------------------------

def _attach_exposure_per_accident(acc: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    out = acc.merge(
        exposure, on=["id_sensor_cercano", "bloque_horario", "es_finde_festivo"], how="left"
    )
    return out[out["exposicion"] > 0]


def _empirical_bayes_index(n_acc: np.ndarray, exposicion: np.ndarray):
    """Tasa encogida (empirical Bayes) e indice de riesgo (tasa encogida /
    tasa global).

    E_i = m * exposicion_i es el numero de accidentes esperado en la unidad i
    si tuviera la tasa global m. Se usa como pseudo-conteo de confianza un
    valor k igual a la mediana de E_i: unidades con E_i << k (poca exposicion
    acumulada, estimacion poco fiable) se acercan a la tasa global, y
    unidades con E_i >> k conservan su tasa observada.
    """
    m = n_acc.sum() / exposicion.sum()          # tasa global
    crude = n_acc / exposicion                   # tasa cruda por unidad
    E = m * exposicion                           # accidentes esperados bajo la tasa global
    k = np.median(E)
    w = E / (E + k)
    shrunk = w * crude + (1 - w) * m
    indice = shrunk / m
    return shrunk, indice, m, k


def build_sensor_risk(acc: pd.DataFrame, exposure: pd.DataFrame):
    df = _attach_exposure_per_accident(acc[acc["id_sensor_cercano"].notna()], exposure)

    sensor_data = (
        df.groupby("id_sensor_cercano")
        .agg(n_accidentes=("num_expediente", "count"), exposicion=("exposicion", "sum"))
        .reset_index()
    )

    shrunk, indice, m, k = _empirical_bayes_index(
        sensor_data["n_accidentes"].values, sensor_data["exposicion"].values
    )
    sensor_data["tasa_encogida"] = shrunk
    sensor_data["indice_riesgo"] = indice

    coords = acc.groupby("id_sensor_cercano").agg(
        lon=("lon", "mean"), lat=("lat", "mean"), distrito=("distrito", "first")
    ).reset_index()
    sensor_data = sensor_data.merge(coords, on="id_sensor_cercano", how="left")
    return sensor_data, m, k


# ---------------------------------------------------------------------------
# 6. Evolucion temporal del indice de riesgo por distrito y año
#    (se usan la tasa global m y el pseudo-conteo k de todo el periodo para
#    que los indices de distintos años sean comparables entre si)
# ---------------------------------------------------------------------------

def build_district_yearly(acc: pd.DataFrame, exposure: pd.DataFrame, m: float, k: float) -> pd.DataFrame:
    df = _attach_exposure_per_accident(
        acc[acc["id_sensor_cercano"].notna() & acc["distrito"].notna()], exposure
    )

    out = (
        df.groupby(["distrito", "anio"])
        .agg(n_accidentes=("num_expediente", "count"), exposicion=("exposicion", "sum"))
        .reset_index()
    )

    crude = out["n_accidentes"] / out["exposicion"]
    E = m * out["exposicion"]
    w = E / (E + k)
    out["tasa_encogida"] = w * crude + (1 - w) * m
    out["indice_riesgo"] = out["tasa_encogida"] / m
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Cargando y limpiando datos crudos...")
    raw = load_raw()
    print(f"  filas leidas correctamente: {len(raw)}")

    df = clean(raw)

    print("Construyendo dataset a nivel de accidente...")
    acc = build_accident_level(df)
    print(f"  accidentes (expedientes unicos): {len(acc)}")
    acc.to_parquet(f"{OUT_DIR}/accidentes_clean.parquet", index=False)

    print("Calculando exposicion proxy al trafico...")
    exposure = build_exposure(df)
    print(f"  combinaciones sensor/franja/finde: {len(exposure)}")

    print("Calculando indice de riesgo por sensor (empirical Bayes)...")
    sensor_risk, tasa_global, k = build_sensor_risk(acc, exposure)
    print(f"  tasa global: {tasa_global:.6f}  |  k: {k:.3f}  |  sensores: {len(sensor_risk)}")
    print(sensor_risk["indice_riesgo"].describe())
    sensor_risk.to_parquet(f"{OUT_DIR}/sensores_riesgo.parquet", index=False)

    print("Calculando evolucion temporal por distrito...")
    dist_year = build_district_yearly(acc, exposure, tasa_global, k)
    dist_year.to_parquet(f"{OUT_DIR}/distrito_riesgo_anual.parquet", index=False)
    print(f"  filas distrito-año: {len(dist_year)}")

    print("Listo.")
