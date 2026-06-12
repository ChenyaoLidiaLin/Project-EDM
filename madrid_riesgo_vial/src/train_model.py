"""
Modelo predictivo: dado el contexto de trafico, meteorologia, hora y zona,
estima la probabilidad de que ocurra un accidente y el tipo mas probable.

Validacion temporal: entrena con accidentes hasta 2022 y evalua con 2023-2024,
para evitar fugas de informacion entre periodos.

Genera:
  data/modelo_tipo_accidente.joblib
  data/feature_importance.parquet
  data/metricas_modelo.json
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

DATA_DIR = "/home/claude/project/data"

NUM_FEATURES = ["intensidad", "ocupacion", "vmed", "mes"]
CAT_FEATURES = ["estado_meteorologico", "bloque_horario", "es_finde_festivo", "distrito"]
TARGET = "tipo_accidente_grp"


def main():
    acc = pd.read_parquet(f"{DATA_DIR}/accidentes_clean.parquet")

    df = acc.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET]).copy()
    df["es_finde_festivo"] = df["es_finde_festivo"].astype(str)

    train = df[df["anio"] <= 2022]
    test = df[df["anio"] > 2022]

    X_train, y_train = train[NUM_FEATURES + CAT_FEATURES], train[TARGET]
    X_test, y_test = test[NUM_FEATURES + CAT_FEATURES], test[TARGET]

    preprocess = ColumnTransformer([
        ("num", "passthrough", NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])

    print("Entrenando modelo (train: hasta 2022, test: 2023-2024)...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    print("Calculando importancia de variables (permutation importance)...")
    pi = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    importance = pd.DataFrame({
        "variable": NUM_FEATURES + CAT_FEATURES,
        "importancia": pi.importances_mean,
    }).sort_values("importancia", ascending=False)
    print(importance)

    joblib.dump(pipe, f"{DATA_DIR}/modelo_tipo_accidente.joblib")
    importance.to_parquet(f"{DATA_DIR}/feature_importance.parquet", index=False)

    with open(f"{DATA_DIR}/metricas_modelo.json", "w") as f:
        json.dump({
            "accuracy_test": report["accuracy"],
            "macro_f1_test": report["macro avg"]["f1-score"],
            "clases": sorted(y_train.unique().tolist()),
            "n_train": len(train),
            "n_test": len(test),
        }, f, indent=2)

    print("Listo.")


if __name__ == "__main__":
    main()
