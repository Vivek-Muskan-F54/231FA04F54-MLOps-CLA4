"""Simulate streaming inference for ten held-out records at one record/second."""
import time

import pandas as pd

from inference_utils import ROOT, load_model, predict_records, prediction_text


def main() -> None:
    records = pd.read_csv(ROOT / "demo" / "streaming_inputs.csv").to_dict(orient="records")
    if len(records) != 10:
        raise ValueError(f"Expected exactly 10 streaming input records, found {len(records)}")
    model = load_model()
    rows = []
    print("STREAMING INFERENCE — one held-out passenger per second")
    for index, record in enumerate(records, start=1):
        prediction = predict_records(model, [record])[0]
        row = {
            "sequence": index,
            "PassengerId": int(record["PassengerId"]),
            "predicted_survival": prediction,
            "prediction": prediction_text(prediction),
        }
        rows.append(row)
        print(f"[{index:02d}/10] PassengerId={row['PassengerId']} -> {row['predicted_survival']} ({row['prediction']})", flush=True)
        if index < len(records):
            time.sleep(1)
    output = ROOT / "streaming_predictions.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
