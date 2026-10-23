"""Train one reusable Titanic classification pipeline from Kaggle train.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL
REQUIRED = ["PassengerId", "Survived", *FEATURES]


def json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"train.csv is missing required Kaggle columns: {missing}")
    if df["Survived"].isna().any() or not set(df["Survived"].unique()).issubset({0, 1}):
        raise ValueError("Survived must contain only non-null binary labels 0 and 1.")
    if df["PassengerId"].duplicated().any():
        raise ValueError("PassengerId values must be unique.")

    X = df[FEATURES].copy()
    y = df["Survived"].astype(int)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["PassengerId"], test_size=0.20, random_state=42, stratify=y
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    # The model is fit exactly once. The same persisted Pipeline serves every later task.
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test).astype(int)
    accuracy = float(accuracy_score(y_test, y_pred))

    out = args.output_dir.resolve()
    demo_dir = out / "demo"
    out.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "model.pkl")

    metrics = {
        "dataset": "Kaggle Titanic competition train.csv",
        "source_url": "https://www.kaggle.com/competitions/titanic/data",
        "rows": int(len(df)),
        "features": FEATURES,
        "model": "LogisticRegression inside an imputation/encoding/scaling Pipeline",
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.20, "random_state": 42, "stratified": True},
        "accuracy": accuracy,
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=[0, 1], target_names=["did_not_survive", "survived"], output_dict=True, zero_division=0),
        "training_passengers": [int(v) for v in id_train.tolist()],
        "held_out_passengers": [int(v) for v in id_test.tolist()],
        "versions": {"python": "3.x", "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (out / "artifacts" / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # All demonstration passengers are drawn only from the held-out portion.
    # The three strategies use disjoint passenger records; Survived is never exported.
    held_out = pd.DataFrame({"PassengerId": id_test.to_numpy(), **{c: X_test[c].to_numpy() for c in FEATURES}})
    held_out = held_out.sample(frac=1, random_state=114).reset_index(drop=True)
    selected = held_out.iloc[:31].copy()
    if len(selected) != 31:
        raise ValueError("The held-out split must contain at least 31 records for demonstrations.")
    def features_only(frame):
        return [{k: json_value(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]

    online = features_only(selected.iloc[:1])[0]
    batch = features_only(selected.iloc[1:21])
    streaming = features_only(selected.iloc[21:31])
    (demo_dir / "online_input.json").write_text(json.dumps(online, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(batch).to_csv(demo_dir / "batch_inputs.csv", index=False)
    pd.DataFrame(streaming).to_csv(demo_dir / "streaming_inputs.csv", index=False)
    demo_manifest = {
        "source": "stratified 20% test/hold-out portion of the supplied Kaggle train.csv",
        "target_column_excluded": True,
        "strategies_are_disjoint": True,
        "online_passenger_id": int(online["PassengerId"]),
        "batch_passenger_ids": [int(x["PassengerId"]) for x in batch],
        "streaming_passenger_ids": [int(x["PassengerId"]) for x in streaming],
    }
    (demo_dir / "manifest.json").write_text(json.dumps(demo_manifest, indent=2) + "\n", encoding="utf-8")

    print("TRAINING COMPLETE")
    print(f"Dataset rows: {len(df)} | train: {len(X_train)} | held-out test: {len(X_test)}")
    print(f"Stratified hold-out accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Pipeline saved: {out / 'model.pkl'}")
    print("Held-out demos: online=1, batch=20, streaming=10 (disjoint; target excluded)")
    print("Confusion matrix [true 0/1 rows, predicted 0/1 columns]:")
    print(np.array(metrics["confusion_matrix_labels_0_then_1"]))


if __name__ == "__main__":
    main()
