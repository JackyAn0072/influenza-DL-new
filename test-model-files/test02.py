from __future__ import annotations
import argparse, os, tempfile, zipfile, io
from typing import List, Tuple
import pandas as pd, numpy as np, geopandas as gpd, networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import certifi, requests

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# ----------- Config ----------- #
CONTIG_STATES = [
    "AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
]
SHAPEFILE_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
TRUTH_CSV_URL = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-data/master/data-truth/truth_inc-hosp_2023-07-17.csv"

# ----------- Load Hospitalization Data ----------- #
def load_hosp_data(season_start_year: int) -> pd.DataFrame:
    df = pd.read_csv(TRUTH_CSV_URL)
    df = df[df["location"].isin(CONTIG_STATES)]
    df["date"] = pd.to_datetime(df["date"])
    season_start = pd.to_datetime(f"{season_start_year}-10-01")
    season_end = pd.to_datetime(f"{season_start_year+1}-05-01")
    df = df[(df["date"] >= season_start) & (df["date"] <= season_end)]
    df = df.rename(columns={"location": "state", "value": "hosp"})
    df = df.groupby(["state", "date"], as_index=False)["hosp"].mean()
    return df

def weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["state", "date"], as_index=False)["hosp"].mean()

# ----------- Graph and Dataset ----------- #
def build_adjacency() -> Tuple[np.ndarray, List[str]]:
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "states.zip")
    r = requests.get(SHAPEFILE_URL, timeout=60, verify=False)
    with open(zpath, "wb") as f: f.write(r.content)
    zipfile.ZipFile(zpath).extractall(tmp)
    shp = [f for f in os.listdir(tmp) if f.endswith(".shp")][0]
    gdf = gpd.read_file(os.path.join(tmp, shp)).to_crs(2163)
    gdf = gdf[gdf["STUSPS"].isin(CONTIG_STATES)].reset_index(drop=True)
    states = gdf.STUSPS.tolist()
    G = nx.Graph(); G.add_nodes_from(range(len(states)))
    for i, g in enumerate(gdf.geometry):
        for j in range(i+1, len(states)):
            if g.touches(gdf.geometry.iloc[j]):
                G.add_edge(i, j)
    return nx.to_numpy_array(G, nodelist=range(len(states))), states

def simulate_sir(beta=0.25, gamma=0.1, I0=1e-4, steps=104):
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0] = 1 - I0, I0
    for t in range(1, steps):
        S[t] = S[t-1] - beta * S[t-1] * I[t-1]
        I[t] = I[t-1] + beta * S[t-1] * I[t-1] - gamma * I[t-1]
        R[t] = R[t-1] + gamma * I[t-1]
    return I

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

def main():
    ap = argparse.ArgumentParser()
    for k, d in [("season", 2022), ("window", 8), ("horizon", 4),
                 ("hidden", 512), ("depth", 4), ("batch", 32),
                 ("epochs", 100), ("lr", 1e-4)]:
        ap.add_argument(f"--{k}", type=type(d), default=d)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="sfnn_forecast_hosp.csv")
    args = ap.parse_args()

    df = load_hosp_data(args.season)
    weekly = weekly_series(df)
    A, states = build_adjacency()

    weeks = sorted(weekly.date.unique()); w2i = {w: i for i, w in enumerate(weeks)}
    s2i = {s: i for i, s in enumerate(states)}
    T = np.zeros((len(weeks), len(states)), dtype=np.float32)
    for _, r in weekly.iterrows():
        if r["state"] in s2i:
            T[w2i[r["date"]], s2i[r["state"]]] = np.log1p(r["hosp"])
    print(f"Log-Hosp mean {T.mean():.3f}, max {T.max():.3f}")

    sir = simulate_sir(steps=T.shape[0])
    sir_m = np.tile(sir[:, None], (1, T.shape[1]))
    X = np.concatenate([T, sir_m], axis=1)
    tensor = torch.tensor(X, dtype=torch.float32)

    model = SFNN(args.window, X.shape[1], args.hidden, args.depth, args.horizon).to(args.device)
    train(model, tensor, args)

    preds = forecast(model, tensor, args)
    preds = np.expm1(preds[:, :len(states)])
    pd.DataFrame(preds, columns=states) \
      .assign(lead_time=lambda d: d.index + 1) \
      .to_csv(args.out, index=False)
    print("Saved →", args.out)

if __name__ == "__main__": main()
