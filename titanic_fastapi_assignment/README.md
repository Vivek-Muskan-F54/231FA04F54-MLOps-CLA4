# Titanic Model Deployment with FastAPI

This project trains one reusable classification pipeline on the Kaggle Titanic competition's labeled `train.csv`, demonstrates online, batch, and simulated streaming inference on held-out passengers, and serves the same saved pipeline through FastAPI.

**Dataset:** [Kaggle Titanic — Data](https://www.kaggle.com/competitions/titanic/data). Download the competition's official `train.csv` after accepting Kaggle's competition rules. Place it at `data/train.csv`. The raw dataset is intentionally excluded from this ZIP/repository; access and redistribution remain subject to Kaggle's competition rules.

## Train once

```bash
python -m pip install -r requirements.txt
python train_model.py
```

Training uses a stratified 80/20 split with `random_state=42`. Missing numeric values are median-imputed and standardized; missing categorical values are mode-imputed and one-hot encoded. A Logistic Regression classifier is fit once inside a scikit-learn `Pipeline`. Its held-out accuracy and split details are saved to `artifacts/metrics.json`; the full reusable pipeline is saved to `model.pkl`.

The training script selects 31 disjoint demonstration passengers exclusively from the held-out split. It exports only their features and passenger identifiers, never their `Survived` labels. The single online record, 20 batch records, and 10 streaming records are distinct. Passenger identifiers are not model features.

## Run the inference demonstrations

```bash
python online_inference.py
python batch_inference.py
python streaming_inference.py
```

The online result is written to `artifacts/online_prediction.json`. Batch predictions are written to `batch_predictions.csv`. Streaming processes one record at a time with a one-second interval and writes `streaming_predictions.csv`. Screenshots of all outputs are in `screenshots/`.

## Start and test the API

In a terminal, start the server:

```bash
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000/docs> for Swagger UI. `GET /health` reports API/model status. `POST /predict` accepts passenger details and returns a binary prediction; Pydantic validates field types and ranges and rejects unknown fields. Example payload:

```json
{
  "PassengerId": 1000,
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

The screenshots show the valid prediction and an invalid `Pclass=4` request (HTTP 422). With the server running, the Python client can be tested in another terminal:

```bash
python client.py
```

## Files

- `train_model.py`: preprocessing, stratified split, one-time training, accuracy evaluation, and held-out demo selection.
- `model.pkl`: complete trained preprocessing/classification pipeline used by every inference mode and the API.
- `online_inference.py`, `batch_inference.py`, `streaming_inference.py`: the three inference demonstrations.
- `app.py`, `client.py`: FastAPI application and Python API client.
- `batch_predictions.csv`, `streaming_predictions.csv`: prediction outputs.
- `report.pdf`: concise observations and measured test accuracy.
- `screenshots/`: output and Swagger test evidence.

## Results

The measured stratified hold-out accuracy, confusion matrix, and classification metrics are recorded in `artifacts/metrics.json` and summarized in `report.pdf`. Every inference route loads the same `model.pkl`; no route retrains a model.

Verification completed: the saved pipeline reloads; the batch output has 20 rows; the streaming output has 10 rows; all 31 demonstration passenger IDs are distinct and held out from training; `/health` returned HTTP 200; a valid `/predict` and `client.py` returned HTTP 200; the invalid `Pclass=4` request returned HTTP 422; and the one-page PDF and six screenshots were opened and checked.
