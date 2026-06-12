"""
Predictive model: given traffic conditions, weather, time of day, and district,
estimates the most likely accident type.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

DATA_DIR = "../data"

NUM_FEATURES = ["intensidad", "ocupacion", "vmed", "month"]
CAT_FEATURES = ["weather", "time_slot", "is_weekend_holiday", "distrito"]
TARGET = "accident_type"


def main():
    acc = pd.read_parquet(f"{DATA_DIR}/accidents_clean.parquet")

    df = acc.dropna(subset=NUM_FEATURES + CAT_FEATURES + [TARGET]).copy()
    df["is_weekend_holiday"] = df["is_weekend_holiday"].astype(str)

    # Temporal split: train on 2016-2022, test on 2023-2024
    train = df[df["year"] <= 2022]
    test  = df[df["year"] > 2022]

    X_train, y_train = train[NUM_FEATURES + CAT_FEATURES], train[TARGET]
    X_test,  y_test  = test[NUM_FEATURES + CAT_FEATURES],  test[TARGET]

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

    print("Training model (train: up to 2022, test: 2023-2024)...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    print("Computing feature importance (permutation-based)...")
    pi = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )

    FEATURE_LABELS = {
        "intensidad":          "traffic flow",
        "ocupacion":           "occupancy",
        "vmed":                "mean speed",
        "weather":             "weather",
        "time_slot":           "time slot",
        "is_weekend_holiday":  "weekend / holiday",
        "distrito":            "district",
    }

    importance = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(f, f) for f in NUM_FEATURES + CAT_FEATURES],
        "importance": pi.importances_mean,
    }).sort_values("importance", ascending=False)
    print(importance)

    joblib.dump(pipe, f"{DATA_DIR}/accident_type_model.joblib")
    importance.to_parquet(f"{DATA_DIR}/feature_importance.parquet", index=False)

    with open(f"{DATA_DIR}/model_metrics.json", "w") as f:
        json.dump({
            "accuracy_test":  report["accuracy"],
            "macro_f1_test":  report["macro avg"]["f1-score"],
            "classes":        sorted(y_train.unique().tolist()),
            "n_train":        len(train),
            "n_test":         len(test),
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
