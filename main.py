import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import create_model

BASE_DIR = Path(__file__).resolve().parent


class WineMLP(nn.Module):

    def __init__(self, in_features=13, hidden=32, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


app = FastAPI(title="Şarap Çeşidi Sınıflandırıcı")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

with open(BASE_DIR / "model" / "meta.json", encoding="utf-8") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]
TARGET_LABELS = ["Çeşit A", "Çeşit B", "Çeşit C"]

scaler = joblib.load(BASE_DIR / "model" / "scaler.pkl")

model = WineMLP(in_features=META["n_features"])
model.load_state_dict(torch.load(BASE_DIR / "model" / "wine_mlp.pt", map_location="cpu"))
model.eval()

FEATURE_LABELS = {
    "alcohol": "Alkol Oranı (%)",
    "malic_acid": "Malik Asit (g/L)",
    "ash": "Kül Miktarı (g/L)",
    "alcalinity_of_ash": "Kül Alkalinitesi",
    "magnesium": "Magnezyum (mg/L)",
    "total_phenols": "Toplam Fenol",
    "flavanoids": "Flavonoid",
    "nonflavanoid_phenols": "Flavonoid Olmayan Fenol",
    "proanthocyanins": "Proantosiyanin",
    "color_intensity": "Renk Yoğunluğu",
    "hue": "Ton (Hue)",
    "od280/od315_of_diluted_wines": "OD280/OD315 Oranı",
    "proline": "Prolin (mg/L)",
}

fields = {name: (float, ...) for name in FEATURE_NAMES}
WineFeatures = create_model("WineFeatures", **fields)


@app.get("/")
def index(request: Request):
    features = [
        {
            "key": name,
            "label": FEATURE_LABELS.get(name, name),
            **META["feature_stats"][name],
        }
        for name in FEATURE_NAMES
    ]
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "features": features,
            "accuracy": round(META["test_accuracy"] * 100, 1),
            "n_samples": META["n_samples"],
        },
    )


@app.post("/predict")
def predict(payload: WineFeatures):
    x = np.array([[getattr(payload, name) for name in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)
    x_t = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad(): 
        logits = model(x_t)
        proba = F.softmax(logits, dim=1).numpy()[0]

    pred_idx = int(np.argmax(proba))
    return {
        "prediction": TARGET_LABELS[pred_idx],
        "prediction_index": pred_idx,
        "probabilities": [
            {"label": TARGET_LABELS[i], "probability": round(float(p) * 100, 2)}
            for i, p in enumerate(proba)
        ],
    }


@app.get("/api/random-sample")
def random_sample():
    from sklearn.datasets import load_wine
    data = load_wine()
    idx = np.random.randint(0, len(data.data))
    sample = {name: float(val) for name, val in zip(FEATURE_NAMES, data.data[idx])}
    sample["_true_label"] = TARGET_LABELS[int(data.target[idx])]
    return sample
