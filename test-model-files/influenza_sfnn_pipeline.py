# influenza_sfnn_pipeline.py
"""
Spatial‑Feature Neural Network (SFNN) demo for U.S. influenza
hospital‑admission forecasting (FluSight Hub truth).

Version 1.3  – 2025‑07‑15
"""
from __future__ import annotations
import argparse, io, os, pathlib, shutil, tempfile, zipfile
from typing import List, Tuple

import certifi, requests, pandas as pd, geopandas as gpd, networkx as nx
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

# --------------------------------------------------------------------------- #
# SSL fix                                                                     #
# --------------------------------------------------------------------------- #
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
TARGET_CSV = (
    "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/"
    "main/target-data/target-hospital-admissions.csv")
SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip")
CONTIG_STATES: List[str] = [
    "AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
    "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR"
]

# --------------------------------------------------------------------------- #
# Data helpers                                                                #
# --------------------------------------------------------------------------- #
def _https_csv(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=60, verify=certifi.where()); r.raise_for_status()
    except requests.exceptions.SSLError:
        print("[WARN] SSL verify failed – retrying with verify=False")
        r = requests.get(url, timeout=60, verify=False); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def load_truth(start: str, end: str) -> pd.DataFrame:
    df = _https_csv(TARGET_CSV)
    df["date"] = pd.to_datetime(df["date"])
    mask = (
        df["date"].between(start, end)
        & (df["location"].str.len() == 2)
        & df["location"].str.isalpha()           # drop numeric regions '01'‑'10'
    )
    return (df.loc[mask, ["location","date","value"]]
              .rename(columns={"location":"state"})
              .reset_index(drop=True))

def to_epiweek(s: pd.Series) -> pd.Series:
    iso = s.dt.isocalendar(); return iso.year*100 + iso.week

def weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    df["epiweek"] = to_epiweek(df["date"])
    return df.groupby(["state","epiweek"], as_index=False)["value"].sum()

# --------------------------------------------------------------------------- #
# Spatial adjacency (Queen contiguity)                                        #
# --------------------------------------------------------------------------- #
def build_adjacency() -> Tuple[np.ndarray, List[str]]:
    try:
        gdf = gpd.read_file(SHAPEFILE_URL).to_crs(2163)
    except Exception as e:
        if "SSL" in str(e):
            print("[WARN] GDAL SSL failed – downloading shapefile manually …")
            tmp = tempfile.mkdtemp(); z = pathlib.Path(tmp)/"states.zip"
            r = requests.get(SHAPEFILE_URL, timeout=60, verify=False); r.raise_for_status()
            z.write_bytes(r.content); zipfile.ZipFile(z).extractall(tmp)
            shp = next(pathlib.Path(tmp).glob("*.shp"))
            gdf = gpd.read_file(shp).to_crs(2163)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            raise
    gdf = gdf[gdf["STUSPS"].isin(CONTIG_STATES)].reset_index(drop=True)
    states = gdf["STUSPS"].tolist()

    G = nx.Graph()
    G.add_nodes_from(range(len(states)))          # ensure isolated nodes exist
    for i, g1 in enumerate(gdf.geometry):
        for j in range(i+1, len(gdf)):
            if g1.touches(gdf.geometry.iloc[j]):
                G.add_edge(i, j)

    A = nx.to_numpy_array(G, nodelist=range(len(states)))
    return A, states

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
class FluDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, p: int, h: int):
        self.X, self.y = self._window(tensor, p, h)
    @staticmethod
    def _window(t: torch.Tensor, p: int, h: int):
        xs, ys = [], []
        for idx in range(p, t.shape[0]-h+1):
            xs.append(t[idx-p:idx]); ys.append(t[idx:idx+h])
        return torch.stack(xs), torch.stack(ys)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class SFNN(nn.Module):
    def __init__(self, p:int, n:int, hidden:int, depth:int, h:int):
        super().__init__()
        layers=[nn.Flatten(), nn.Linear(p*n, hidden), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden, h*n))
        self.net = nn.Sequential(*layers); self.h,self.n = h,n
    def forward(self,x): return self.net(x).view(-1,self.h,self.n)

# --------------------------------------------------------------------------- #
# Train & forecast                                                            #
# --------------------------------------------------------------------------- #
def train(model, tensor, args):
    ds = FluDataset(tensor, args.window, args.horizon)
    n_tr = int(len(ds)*0.9); tr,vl = random_split(ds,[n_tr,len(ds)-n_tr])
    tl,vl = DataLoader(tr,batch_size=args.batch,shuffle=True), DataLoader(vl,batch_size=args.batch)
    opt, mse = torch.optim.Adam(model.parameters(), lr=args.lr), nn.MSELoss()
    for ep in range(args.epochs):
        model.train(); tot=0
        for X,y in tl:
            X,y = X.to(args.device), y.to(args.device)
            opt.zero_grad(); loss=mse(model(X),y); loss.backward(); opt.step()
            tot += loss.item()*X.size(0)
        if (ep+1)%5==0:
            model.eval(); vtot=0
            with torch.no_grad():
                for X,y in vl:
                    vtot += mse(model(X.to(args.device)), y.to(args.device)).item()*X.size(0)
            print(f"Ep {ep+1:>3}/{args.epochs} train {tot/len(tr):.4f} "
                  f"val {vtot/len(vl):.4f}")

def forecast(model, tensor, args)->np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(tensor[-args.window:].unsqueeze(0).to(args.device)).cpu().numpy()[0]

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: List[str]|None=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2021)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon",type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth",  type=int, default=4)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",    default="sfnn_forecast_state_level.csv")
    args,_ = ap.parse_known_args(argv)

    # 1 – truth data
    start,end = f"{args.season}-10-03", f"{args.season+1}-10-01"
    print("Fetching FluSight truth …")
    weekly = weekly_series(load_truth(start,end))

    # 2 – graph
    A, states = build_adjacency()

    # 3 – tensor
    ew = np.sort(weekly["epiweek"].unique())
    w2r = {w:i for i,w in enumerate(ew)}
    s2c = {s:i for i,s in enumerate(states)}
    T = np.zeros((len(ew), len(states)), dtype=np.float32)
    for _,r in weekly.iterrows():
        if r["state"] in s2c:                       # safety check
            T[w2r[r["epiweek"]], s2c[r["state"]]] = r["value"]
    tensor = torch.tensor(T)

    # 4 – model & train
    model = SFNN(args.window, tensor.shape[1],
                 args.hidden, args.depth, args.horizon).to(args.device)
    train(model, tensor, args)

    # 5 – forecast
    preds = forecast(model, tensor, args)
    pd.DataFrame(preds, columns=states)\
      .assign(lead_time=lambda d:d.index+1)\
      .to_csv(args.out, index=False)
    print("Saved →", args.out)

if __name__ == "__main__":
    main()
