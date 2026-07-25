
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)


class WineMLP(nn.Module):
    """13 kimyasal özellik -> 3 üzüm çeşidi sınıflandırıcı."""

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


def main():
    data = load_wine()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = WineMLP(in_features=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 300
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_pred = test_logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, test_pred)
    print(f"\nTest doğruluğu: {acc:.4f}")
    print(classification_report(y_test, test_pred, target_names=[f"class_{i}" for i in range(3)]))

    # --- Kaydet ---
    torch.save(model.state_dict(), "model/wine_mlp.pt")
    joblib.dump(scaler, "model/scaler.pkl")

    feature_stats = {
        name: {
            "min": float(np.min(X[:, i])),
            "max": float(np.max(X[:, i])),
            "mean": float(np.mean(X[:, i])),
            "std": float(np.std(X[:, i])),
        }
        for i, name in enumerate(feature_names)
    }

    meta = {
        "feature_names": feature_names,
        "feature_stats": feature_stats,
        "test_accuracy": acc,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "architecture": "13 -> 32 (ReLU) -> 32 (ReLU) -> 3",
        "framework": "pytorch",
    }
    with open("model/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Kaydedildi -> model/wine_mlp.pt, model/scaler.pkl, model/meta.json")


if __name__ == "__main__":
    main()
