from __future__ import annotations
import argparse, os
import pandas as pd, numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# ----------- Dataset ----------- #
class FluDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, p: int, h: int):
        self.X = torch.stack([tensor[i - p:i] for i in range(p, tensor.shape[0] - h + 1)])
        self.y = torch.stack([tensor[i:i + h] for i in range(p, tensor.shape[0] - h + 1)])
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ----------- LSTM Model ----------- #
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, h):
        super().__init__()
        self.h = h
        self.n = output_dim 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim * h)
    def forward(self, x):
        out, _ = self.lstm(x)          # [B, P, H]
        out = out[:, -1, :]            # [B, H]
        out = self.linear(out)         # [B, H * N]
        return out.view(-1, self.h, self.n)  # reshape


# ----------- Train & Forecast ----------- #
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

# ----------- Main ----------- #
def main():
    ap = argparse.ArgumentParser()
    for k, d in [("window", 8), ("horizon", 4), ("hidden", 128),
                 ("depth", 2), ("batch", 32), ("epochs", 100), ("lr", 1e-3)]:
        ap.add_argument(f"--{k}", type=type(d), default=d)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="forecast_lstm.csv")
    args = ap.parse_args()

    # Load preprocessed tensor from test02
    loaded = torch.load("data/test02_flu_counts.pt")
    tensor = loaded["tensor"] if isinstance(loaded, dict) else loaded
    states = loaded["states"] if isinstance(loaded, dict) else [f"S{i+1}" for i in range(tensor.shape[1])]
    print(f"Loaded tensor shape: {tensor.shape}, number of states: {len(states)}")

    # Build & train model
    T, N = tensor.shape
    model = LSTMForecaster(input_dim=N, hidden_dim=args.hidden, num_layers=args.depth,
                           output_dim=N, h=args.horizon).to(args.device)
    train(model, tensor, args)

    # Forecast
    preds = forecast(model, tensor, args)
    pd.DataFrame(preds, columns=states) \
        .assign(lead_time=lambda d: d.index + 1) \
        .to_csv(args.out, index=False)
    print("Saved forecast →", args.out)

if __name__ == "__main__":
    main()
