"""Run one online-style prediction using a held-out record and model.pkl."""
import json
from pathlib import Path

from inference_utils import ROOT, load_model, predict_records, prediction_text


def main() -> None:
    record = json.loads((ROOT / "demo" / "online_input.json").read_text(encoding="utf-8"))
    prediction = predict_records(load_model(), [record])[0]
    result = {
        "PassengerId": int(record["PassengerId"]),
        "predicted_survival": prediction,
        "prediction": prediction_text(prediction),
    }
    out = ROOT / "artifacts" / "online_prediction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("ONLINE INFERENCE — one held-out passenger")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
