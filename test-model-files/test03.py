from __future__ import annotations
import argparse, os
from typing import List, Tuple
import pandas as pd, numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# ----------- Dataset & Model ----------- #
class FluDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, p: int, h: int):
        self.X = torch.stack([tensor[i - p:i] for i in range(p, tensor.shape[0] - h + 1)])
        self.y = torch.stack([tensor[i:i + h] for i in range(p, tensor.shape[0] - h + 1)])
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SFNN(nn.Module):
    def __init__(self, p, n, hidden, depth, h):
        super().__init__()
        self.h, self.n = h, n
        layers = [nn.Flatten(), nn.Linear(p * n, hidden), nn.BatchNorm1d(hidden),
                  nn.ReLU(), nn.Dropout(0.1)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden, h * n))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).view(-1, self.h, self.n)

# ----------- Training & Forecasting ----------- #
def train(model, tensor, args):
    ds = FluDataset(tensor, args.window, args.horizon)
    tr, vl = random_split(ds, [int(len(ds)*0.9), len(ds)-int(len(ds)*0.9)])
    tl, vl = DataLoader(tr, batch_size=args.batch), DataLoader(vl, batch_size=args.batch)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); mse = nn.MSELoss()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0
        for X, y in tl:
            X, y = X.to(args.device), y.to(args.device)
            opt.zero_grad(); loss = mse(model(X), y); loss.backward(); opt.step()
            tot += loss.item() * X.size(0)
        if ep % 5 == 0:
            model.eval(); vtot = 0
            with torch.no_grad():
                for X, y in vl:
                    vtot += mse(model(X.to(args.device)), y.to(args.device)).item() * X.size(0)
            print(f"Ep {ep:3d}/{args.epochs} train {tot/len(tr):.4f} val {vtot/len(vl):.4f}")

def forecast(model, tensor, args) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(tensor[-args.window:].unsqueeze(0).to(args.device)).cpu().numpy()[0]

# ----------- Main Entry ----------- #
def main():
    ap = argparse.ArgumentParser()
    for k, d in [("window", 8), ("horizon", 4),
                 ("hidden", 512), ("depth", 4), ("batch", 32),
                 ("epochs", 100), ("lr", 1e-4)]:
        ap.add_argument(f"--{k}", type=type(d), default=d)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="sfnn_forecast_flu.csv")
    args = ap.parse_args()

    # Step 1: Load flu count tensor and states
    loaded = torch.load("/Users/jackyan/Desktop/influenza-DL/data/test02_flu_counts.pt")
    if isinstance(loaded, dict):  # best practice: saved as dict {"tensor": ..., "states": ...}
        tensor = loaded["tensor"]
        states = loaded["states"]
    else:  # fallback (if only tensor was saved)
        tensor = loaded
        states = [f"S{i+1}" for i in range(tensor.shape[1])]  # dummy names if unknown

    print(f"Loaded tensor shape: {tensor.shape}, number of states: {len(states)}")

    # Step 2: Train model
    T, N = tensor.shape
    model = SFNN(args.window, N, args.hidden, args.depth, args.horizon).to(args.device)
    train(model, tensor, args)

    # Step 3: Forecast and save
    preds = forecast(model, tensor, args)
    pd.DataFrame(preds, columns=states) \
        .assign(lead_time=lambda d: d.index + 1) \
        .to_csv(args.out, index=False)
    print("Saved forecast →", args.out)

if __name__ == "__main__":
    main()
