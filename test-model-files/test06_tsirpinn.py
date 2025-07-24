from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import torch
from torch import nn
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# ---------- TSIR‑PINN components ----------
class BetaNet(nn.Module):
    def __init__(self, hidden=64, depth=2):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(1 if i == 0 else hidden, hidden),
                nn.Tanh()
            ]
        layers.append(nn.Linear(hidden, 1))
        self.body = nn.Sequential(*layers)
        self.softplus = nn.Softplus()  # ensure β(t) > 0

    def forward(self, t):  # t shape: [*, 1]
        return self.softplus(self.body(t))


def tsir_forward(beta_fn, I0, S0, horizon, dt=1.0, N=1.0):
    """β(t) horizon simulation"""
    I, S = [I0], [S0]
    for k in range(horizon):
        t_val = torch.tensor([[1.0 + k / horizon]], device=I0.device)  #
        beta_t = beta_fn(t_val)               # shape [1,1]
        new_inf = dt * beta_t * I[-1] * S[-1] / N
        I_next = I[-1] + new_inf
        S_next = torch.clamp(S[-1] - new_inf, min=0.0)
        I.append(I_next.squeeze(0))
        S.append(S_next.squeeze(0))
    return torch.stack(I[1:])  # return horizon step，shape [h]

def physics_loss(beta_fn, y_hist, N=1.0, dt=1.0):
    """Make I(t) satisify TSIR equation"""
    t_hist = torch.linspace(0, 1, steps=len(y_hist)-1, device=y_hist.device).view(-1,1)
    beta_t = beta_fn(t_hist).squeeze()              # [T-1]
    I_t, I_tp1 = y_hist[:-1], y_hist[1:]
    S_t = torch.clamp(N - I_t.cumsum(0), min=0.0)   # S_t
    rhs = beta_t * I_t * S_t / N * dt
    return torch.mean((I_tp1 - (I_t + rhs))**2)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_phy", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="forecast_tsir.csv")
    args = ap.parse_args()

    # 1. data pre-processing
    loaded = torch.load("data/test02_flu_counts.pt")
    tensor  = loaded["tensor"].float()            # shape [T, N]
    states  = loaded["states"]
    tensor  = tensor.to(args.device)
    T, Nstates = tensor.shape
    print(f"Tensor {T}×{Nstates}")

    # population estimate
    popN = tensor.max(0).values * 10.0 + 1e-6     # shape [N]

    preds_all = []

    # training
    for idx, state in enumerate(states):
        y_hist = tensor[:, idx]                   # [T]
        I0 = y_hist[-1]                  
        S0 = torch.clamp(popN[idx] - I0, min=0.0)

        beta_net = BetaNet(args.hidden, args.depth).to(args.device)
        opt = torch.optim.Adam(beta_net.parameters(), lr=args.lr)

        for ep in range(1, args.epochs+1):
            opt.zero_grad()
            # loss evaluation
            pred_rate = y_hist.clamp(min=1e-2)    
            poisson = torch.mean(pred_rate - y_hist * torch.log(pred_rate))
            phy = physics_loss(beta_net, y_hist, N=popN[idx], dt=1.0)
            loss = poisson + args.lambda_phy * phy
            loss.backward(); opt.step()
            if ep % 100 == 0:
                print(f"[{state}] ep {ep}/{args.epochs} loss {loss.item():.4f}")

        # 3. prediction horizon weeks
        I_pred_future = tsir_forward(beta_net, I0, S0, args.horizon,
                                     dt=1.0, N=popN[idx]).detach().cpu()  # [h]
        preds_all.append(I_pred_future.view(-1))

    # 4. CSV output
    preds_mat = torch.stack(preds_all, dim=1).squeeze().numpy()
    df_out = pd.DataFrame(preds_mat, columns=states)
    df_out["lead_time"] = np.arange(1, args.horizon+1)
    df_out.to_csv(args.out, index=False)
    print("Saved →", args.out)

if __name__ == "__main__":
    main()
