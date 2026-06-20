"""
Data cleaning and preparation for Madrid traffic accidents (2016-2024)
matched with traffic data from the nearest sensor.

Outputs:
  ../data/accidents_clean.parquet       one row per accident (deduplicated by case number)
  ../data/sensor_risk.parquet           risk index per sensor (used by the map)
  ../data/district_risk_yearly.parquet  risk index per district and year (time series)
"""

import csv
import unicodedata
import pandas as pd
import numpy as np
from pyproj import Transformer

RAW_PATH = "../data/accidentes_con_trafico_final.csv"
OUT_DIR = "../data"

WEATHER_MAP = {
    "despejado": "clear",
    "nublado": "cloudy",
    "lluvia debil": "light rain",
    "lluvia intensa": "heavy rain",
    "nevando": "snowing",
    "granizando": "hailing",
    "se desconoce": "unknown",
}


# The raw CSV has a trailing ";;" on every line and some rows wrap the entire
# record in outer quotes with internal double-quote escaping (standard CSV
# double-write artifact).
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


def strip_accents(s):
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def assign_time_slot(hora):
    h = pd.to_numeric(hora.str.split(":").str[0], errors="coerce")
    bins   = [-1, 5, 11, 18, 23]
    labels = ["night (0-5h)", "morning (6-11h)", "afternoon (12-18h)", "evening (19-23h)"]
    return pd.cut(h, bins=bins, labels=labels)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"estado_meteorológico": "estado_meteorologico"})

    num_cols = ["coordenada_x_utm", "coordenada_y_utm", "es_festivo",
                "id_sensor_cercano", "intensidad", "ocupacion", "vmed"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")

    df["fecha"] = pd.to_datetime(df["fecha"], format="%m/%d/%Y", errors="coerce")
    df["year"]  = df["fecha"].dt.year
    df["month"] = df["fecha"].dt.month

    # Extract hour as a continuous numeric feature
    df["hour"] = pd.to_numeric(df["hora"].str.split(":").str[0], errors="coerce")

    text_cols = ["dia_semana", "distrito", "tipo_accidente", "tipo_vehiculo",
                 "estado_meteorologico"]
    for c in text_cols:
        df[c] = (df[c].astype(str).str.strip().str.lower()
                 .replace({"nan": np.nan, "": np.nan}))
        df[c] = df[c].apply(strip_accents)

    DAY_MAP = {
        "lunes":     "monday",
        "martes":    "tuesday",
        "miercoles": "wednesday",
        "jueves":    "thursday",
        "viernes":   "friday",
        "sabado":    "saturday",
        "domingo":   "sunday",
    }
    df["day_of_week"] = df["dia_semana"].map(DAY_MAP).fillna("unknown")

    df["weather"]   = df["estado_meteorologico"].map(WEATHER_MAP).fillna("unknown")
    df["time_slot"] = assign_time_slot(df["hora"])

    # vmed=0 and intensidad=0 are sensor artifacts (no reading), not real zeros
    df["vmed"]       = df["vmed"].replace(0, np.nan)
    df["intensidad"] = df["intensidad"].replace(0, np.nan)

    # Convert UTM coordinates (ETRS89 zone 30N) to WGS84 lat/lon
    transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
    mask = df["coordenada_x_utm"].notna() & df["coordenada_y_utm"].notna()
    lon, lat = transformer.transform(
        df.loc[mask, "coordenada_x_utm"].values,
        df.loc[mask, "coordenada_y_utm"].values,
    )
    df.loc[mask, "lon"] = lon
    df.loc[mask, "lat"] = lat

    return df


def build_accident_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["num_expediente"].notna() & df["fecha"].notna()]

    # Count vehicles involved per case before deduplication
    n_vehicles = df.groupby("num_expediente")["tipo_vehiculo"].count()
    acc = df.drop_duplicates(subset="num_expediente", keep="first").copy()
    acc["n_vehicles"] = acc["num_expediente"].map(n_vehicles)

    def group_accident_type(t):
        if pd.isna(t):
            return np.nan
        if "atropello" in t:
            return "pedestrian knockdown"
        if "colision" in t or "alcance" in t:
            return "collision"
        if "caida" in t or "vuelco" in t or "despen" in t:
            return "fall / rollover"
        if "choque" in t:
            return "object impact"
        return "other"

    acc["accident_type"] = acc["tipo_accidente"].apply(group_accident_type)

    keep = ["fecha", "year", "month", "hour", "day_of_week", "hora", "time_slot",
            "distrito", "num_expediente", "tipo_accidente",
            "accident_type", "tipo_vehiculo", "n_vehicles",
            "weather", "lon", "lat",
            "id_sensor_cercano", "intensidad", "ocupacion", "vmed"]
    return acc[keep]


def build_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Traffic exposure proxy: mean flow per sensor, time slot, and day type.

    Since the dataset only contains accident rows (no zero-accident observations),
    we use the mean historical flow recorded at each sensor/slot/day-type as a
    proxy for typical traffic volume in that context.
    """
    base = df[df["intensidad"].notna() & df["id_sensor_cercano"].notna() & df["day_of_week"].notna()]
    exposure = (
        base.groupby(["id_sensor_cercano", "time_slot", "day_of_week"],
                     observed=True)["intensidad"]
        .mean()
        .reset_index()
        .rename(columns={"intensidad": "exposure"})
    )
    return exposure


def _attach_exposure_per_accident(acc: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    out = acc.merge(
        exposure, on=["id_sensor_cercano", "time_slot", "day_of_week"], how="left"
    )
    return out[out["exposure"] > 0]


def _empirical_bayes_index(n_acc: np.ndarray, exposure: np.ndarray):
    """Compute shrunk accident rates and risk indices using empirical Bayes.

    For each unit i, E_i = m * exposure_i is the expected accident count under
    the city-wide rate m. We use k = median(E_i) as the prior weight: units with
    low exposure shrink toward the global rate, while well-observed units keep
    their own rate. Method follows Marshall (1991).
    """
    m      = n_acc.sum() / exposure.sum()   # city-wide accident rate
    crude  = n_acc / exposure               # raw rate per unit
    E      = m * exposure                   # expected accidents under global rate
    k      = np.median(E)                   # prior strength (median expected count)
    w      = E / (E + k)                    # shrinkage weight
    shrunk = w * crude + (1 - w) * m
    index  = shrunk / m                     # risk index (1 = city average)
    return shrunk, index, m, k


def build_sensor_risk(acc: pd.DataFrame, exposure: pd.DataFrame):
    df = _attach_exposure_per_accident(acc[acc["id_sensor_cercano"].notna()], exposure)

    sensor_data = (
        df.groupby("id_sensor_cercano")
        .agg(n_accidents=("num_expediente", "count"), exposure=("exposure", "sum"))
        .reset_index()
    )

    shrunk, index, m, k = _empirical_bayes_index(
        sensor_data["n_accidents"].values, sensor_data["exposure"].values
    )
    sensor_data["shrunk_rate"] = shrunk
    sensor_data["risk_index"]  = index

    coords = acc.groupby("id_sensor_cercano").agg(
        lon=("lon", "mean"), lat=("lat", "mean"), district=("distrito", "first")
    ).reset_index()
    sensor_data = sensor_data.merge(coords, on="id_sensor_cercano", how="left")
    return sensor_data, m, k


def build_district_yearly(acc: pd.DataFrame, exposure: pd.DataFrame, m: float, k: float) -> pd.DataFrame:
    """Year-by-year risk index per district.

    We reuse the global m and k from the full period so that indices remain
    comparable across years.
    """
    df = _attach_exposure_per_accident(
        acc[acc["id_sensor_cercano"].notna() & acc["distrito"].notna() & acc["day_of_week"].notna()], exposure
    )

    out = (
        df.groupby(["distrito", "year"])
        .agg(n_accidents=("num_expediente", "count"), exposure=("exposure", "sum"))
        .reset_index()
    )

    crude  = out["n_accidents"] / out["exposure"]
    E      = m * out["exposure"]
    w      = E / (E + k)
    out["shrunk_rate"] = w * crude + (1 - w) * m
    out["risk_index"]  = out["shrunk_rate"] / m
    return out


def build_travel_risk(acc: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Accident rate per district / travel mode / time slot / weather combination,
    normalized by district-level traffic exposure (same methodology as the sensor
    risk map and district yearly trends).

    WHY exposure matters here
    -------------------------
    The previous version divided accidents by the mean accident count, so the
    denominator was still an accident-derived number.  That tells you "when do
    more accidents happen?", not "when do accidents happen more than traffic
    volume would predict?".  Using the sensor flow as denominator captures the
    second, harder question: e.g. fewer cars drive in heavy rain, so a raw
    count of 3 accidents at 3 am in a storm is far more alarming than 3
    accidents on a clear Friday afternoon.

    HOW the denominator is built
    ----------------------------
    The `exposure` table has one row per (sensor, time_slot, day_of_week) with
    the mean historical intensidad (veh/h) for that combination.  Each sensor
    is mapped to the district it belongs to (using the first district label in
    the accident table).  We then sum over sensors and average over day-of-week
    to get a single exposure figure per (district, time_slot) — representing
    the typical total traffic volume in that part of the city during that part
    of the day.  Weather is *not* used as a grouping key for the denominator:
    we want to capture "accidents per unit of traffic despite the rain", not
    "accidents per unit of rainy-day traffic" (which would cancel out the
    reduced traffic and understate the real risk).

    EMPIRICAL BAYES SHRINKAGE
    -------------------------
    Applied within each (district, travel_type) pair, exactly as in
    _empirical_bayes_index(): each (time_slot, weather) cell is shrunk toward
    the pair's own mean rate.  Cells with very few accidents converge to 1.0;
    well-observed cells keep their own estimated rate.
    """

    # ── 1. Map sensor → district; aggregate exposure to district × time_slot ──
    sensor_to_district = (
        acc.dropna(subset=["id_sensor_cercano", "distrito"])
        .groupby("id_sensor_cercano")["distrito"].first()
    )
    exp = exposure.copy()
    exp["distrito"] = exp["id_sensor_cercano"].map(sensor_to_district)
    # Sum over sensors in the same district; average over day-of-week so that
    # a district with 5 sensors doesn't look 5× busier than one with 1.
    exp_district = (
        exp.dropna(subset=["distrito"])
        .groupby(["distrito", "time_slot", "day_of_week"], observed=True)["exposure"]
        .sum()
        .reset_index()
        .groupby(["distrito", "time_slot"], observed=True)["exposure"]
        .mean()
        .reset_index()
    )

    # ── 2. Classify travel / road-user type ──────────────────────────────────
    def _travel_type(row):
        a = row["accident_type"]
        v = str(row["tipo_vehiculo"]) if pd.notna(row["tipo_vehiculo"]) else ""
        if a == "pedestrian knockdown":
            return "pedestrian"
        if "motocicleta" in v or "ciclomotor" in v:
            return "motorcycle"
        if "bicicleta" in v or "vmu" in v:
            return "bike / e-scooter"
        if "turismo" in v or "todo terreno" in v:
            return "car"
        if "autobus" in v or "autocar" in v:
            return "bus"
        if "camion" in v or "furgon" in v or "tractocamion" in v:
            return "truck / van"
        return None

    acc = acc.copy()
    acc["travel_type"] = acc.apply(_travel_type, axis=1)

    # ── 3. Count accidents per (district × travel_type × time_slot × weather) ─
    grp = (
        acc.dropna(subset=["distrito", "travel_type", "time_slot", "weather"])
        .groupby(["distrito", "travel_type", "time_slot", "weather"], observed=True)
        .size()
        .reset_index(name="n_accidents")
    )

    # ── 4. Attach district-level traffic exposure ────────────────────────────
    # Join on (distrito, time_slot) only — weather is intentionally excluded
    # from the denominator (see docstring above).
    grp = grp.merge(exp_district, on=["distrito", "time_slot"], how="inner")
    grp = grp[grp["exposure"] > 0]

    # ── 5. Empirical Bayes shrinkage within each (district, travel_type) pair ─
    results = []
    for _, sub in grp.groupby(["distrito", "travel_type"], observed=True):
        n = sub["n_accidents"].values.astype(float)
        e = sub["exposure"].values.astype(float)
        rate  = n / e
        m     = rate.mean()            # mean rate for this district × mode pair
        if m <= 0:
            sub = sub.copy()
            sub["risk_index"] = 1.0
            results.append(sub)
            continue
        E = m * e                      # expected count under mean rate
        k = max(float(np.median(E)), 1e-6)   # prior strength (Marshall 1991)
        w = E / (E + k)                # shrinkage weight (0 = shrink fully, 1 = keep raw)
        shrunk = w * rate + (1 - w) * m
        sub = sub.copy()
        sub["risk_index"] = shrunk / m
        results.append(sub)

    return pd.concat(results, ignore_index=True)[
        ["distrito", "travel_type", "time_slot", "weather",
         "n_accidents", "exposure", "risk_index"]
    ]


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading and cleaning raw data...")
    raw = load_raw()
    print(f"  rows parsed successfully: {len(raw)}")

    df = clean(raw)

    print("Building accident-level dataset...")
    acc = build_accident_level(df)
    print(f"  unique accidents (case numbers): {len(acc)}")
    acc.to_parquet(f"{OUT_DIR}/accidents_clean.parquet", index=False)

    print("Computing traffic exposure proxy...")
    exposure = build_exposure(df)
    print(f"  sensor / slot / day-type combinations: {len(exposure)}")

    print("Computing sensor risk index (empirical Bayes)...")
    sensor_risk, global_rate, k = build_sensor_risk(acc, exposure)
    print(f"  global rate: {global_rate:.6f}  |  k: {k:.3f}  |  sensors: {len(sensor_risk)}")
    print(sensor_risk["risk_index"].describe())
    sensor_risk.to_parquet(f"{OUT_DIR}/sensor_risk.parquet", index=False)

    print("Computing yearly district trends...")
    dist_year = build_district_yearly(acc, exposure, global_rate, k)
    dist_year.to_parquet(f"{OUT_DIR}/district_risk_yearly.parquet", index=False)
    print(f"  district-year rows: {len(dist_year)}")

    print("Computing travel risk by mode, time slot and weather...")
    travel_risk = build_travel_risk(acc, exposure)
    travel_risk.to_parquet(f"{OUT_DIR}/travel_risk.parquet", index=False)
    print(f"  travel-risk rows: {len(travel_risk)}")

    print("Done.")
