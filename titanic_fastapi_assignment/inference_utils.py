"""Shared helpers for every inference strategy and the FastAPI application."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Saved pipeline not found at {path}. Run python train_model.py first.")
    return joblib.load(path)


def feature_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create the exact feature schema used at training; ignore identifiers."""
    return pd.DataFrame([{name: record.get(name) for name in FEATURES} for record in records], columns=FEATURES)


def predict_records(model, records: list[dict[str, Any]]) -> list[int]:
    if not records:
        return []
    return [int(value) for value in model.predict(feature_frame(records))]


def prediction_text(value: int) -> str:
    return "Survived" if int(value) == 1 else "Did not survive"
