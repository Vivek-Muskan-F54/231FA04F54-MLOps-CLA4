"""FastAPI model-serving application for the saved Titanic pipeline."""
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from inference_utils import load_model, predict_records, prediction_text

app = FastAPI(
    title="Titanic Survival Model API",
    description="Serving the saved Kaggle Titanic Logistic Regression pipeline.",
    version="1.0.0",
)
model = load_model()


class PassengerDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    PassengerId: Optional[int] = Field(default=None, ge=1, description="Optional Kaggle passenger identifier; not used as a model feature")
    Pclass: int = Field(ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(default=None, gt=0, le=100)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0)
    Embarked: Optional[Literal["C", "Q", "S"]] = None


@app.get("/health", tags=["status"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", tags=["inference"])
def predict(passenger: PassengerDetails):
    record = passenger.model_dump()
    passenger_id = record.pop("PassengerId", None)
    prediction = predict_records(model, [record])[0]
    return {
        "PassengerId": passenger_id,
        "predicted_survival": prediction,
        "prediction": prediction_text(prediction),
    }
