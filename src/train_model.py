"""
Predictive model: given traffic conditions, weather, time of day, and vehicle
type, estimates the most likely accident type.

Temporal validation: trains on accidents up to 2022 and evaluates on 2023-2024
to avoid any information leakage between periods.

Outputs:
  ../data/accident_type_model.joblib
  ../data/feature_importance.parquet
  ../data/model_metrics.json
"""

import json
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR = "../data"

NUM_FEATURES = ["n_vehicles", "hour", "month"]
CAT_FEATURES = ["vehicle_cat", "weather", "is_weekend_holiday", "distrito"]
TARGET = "accident_type"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # vehicle_cat: motorcycle dominates fall/rollover (62%),
    # truck dominates object impact (18.5%), car dominates the rest
    def vehicle_cat(v):
        if pd.isna(v): return "unknown"
        v = str(v)
        if "motocicleta" in v or "ciclomotor" in v: return "motorcycle"
        if "bicicleta" in v or "vmu" in v:          return "bike"
        if "autobus" in v or "autocar" in v:        return "bus"
        if "camion" in v or "furgon" in v or "tractocamion" in v: return "truck"
        if "turismo" in v or "todo terreno" in v:   return "car"
        return "other"

    df["vehicle_cat"] = df["tipo_vehiculo"].apply(vehicle_cat)

    return df


def main():
    acc = pd.read_parquet(f"{DATA_DIR}/accidents_clean.parquet")
    acc = engineer_features(acc)

    df = acc.dropna(subset=CAT_FEATURES + [TARGET]).copy()
    df["is_weekend_holiday"] = df["is_weekend_holiday"].astype(str)

    # Temporal split: train on 2016-2022, test on 2023-2024
    train = df[df["year"] <= 2022]
    test  = df[df["year"] > 2022]

    X_train, y_train = train[NUM_FEATURES + CAT_FEATURES], train[TARGET]
    X_test,  y_test  = test[NUM_FEATURES + CAT_FEATURES],  test[TARGET]

    # GradientBoostingClassifier does not support class_weight natively,
    # so we pass balanced sample weights at fit time
    sample_weights = compute_sample_weight("balanced", y_train)

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", GradientBoostingClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42
        )),
    ])

    print("Training model (train: up to 2022, test: 2023-2024)...")
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    print("Computing feature importance (permutation-based)...")
    pi = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )

    FEATURE_LABELS = {
        "n_vehicles":         "vehicles involved",
        "hour":               "hour of day",
        "month":              "month",
        "vehicle_cat":        "vehicle type",
        "weather":            "weather",
        "is_weekend_holiday": "weekend / holiday",
        "distrito":           "district",
    }

    importance = pd.DataFrame({
        "feature":    [FEATURE_LABELS.get(f, f) for f in NUM_FEATURES + CAT_FEATURES],
        "importance": pi.importances_mean,
    }).sort_values("importance", ascending=False)
    print(importance)

    joblib.dump(pipe, f"{DATA_DIR}/accident_type_model.joblib")
    importance.to_parquet(f"{DATA_DIR}/feature_importance.parquet", index=False)

    with open(f"{DATA_DIR}/model_metrics.json", "w") as f:
        json.dump({
            "accuracy_test": report["accuracy"],
            "macro_f1_test": report["macro avg"]["f1-score"],
            "classes":       sorted(y_train.unique().tolist()),
            "n_train":       len(train),
            "n_test":        len(test),
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
