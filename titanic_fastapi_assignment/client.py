"""Minimal Python client for the FastAPI /predict endpoint."""
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
BASE_URL = os.getenv("TITANIC_API_URL", "http://127.0.0.1:8000")


def main() -> None:
    passenger = json.loads((ROOT / "demo" / "online_input.json").read_text(encoding="utf-8"))
    response = requests.post(f"{BASE_URL}/predict", json=passenger, timeout=15)
    response.raise_for_status()
    print("Python client POST /predict: HTTP", response.status_code)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
