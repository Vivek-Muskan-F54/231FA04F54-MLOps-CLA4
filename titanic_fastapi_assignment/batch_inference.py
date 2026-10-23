"""Predict a 20-record hold-out batch together and save batch_predictions.csv."""
import pandas as pd

from inference_utils import ROOT, load_model, predict_records, prediction_text


def main() -> None:
    records = pd.read_csv(ROOT / "demo" / "batch_inputs.csv").to_dict(orient="records")
    if len(records) != 20:
        raise ValueError(f"Expected exactly 20 batch input records, found {len(records)}")
    predictions = predict_records(load_model(), records)
    result = pd.DataFrame({
        "PassengerId": [int(r["PassengerId"]) for r in records],
        "predicted_survival": predictions,
        "prediction": [prediction_text(v) for v in predictions],
    })
    output = ROOT / "batch_predictions.csv"
    result.to_csv(output, index=False)
    print(f"BATCH INFERENCE — {len(result)} held-out passengers predicted together")
    print(result.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
