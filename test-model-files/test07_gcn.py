from __future__ import annotations
import argparse, os, tempfile, zipfile, warnings
from typing import List, Tuple

import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import requests, geopandas as gpd, networkx as nx, certifi
warnings.filterwarnings("ignore")               # take care of warnings
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# ──────────────────────────────  1. contiguous‑state  ──────────────────────────────
CONTIG_ST = [
 "AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
 "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
 "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
]
SHAPE_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"

def contig_adj() -> Tuple[np.ndarray, List[str]]:
    tmp = tempfile.mkdtemp(); z = os.path.join(tmp,"us.zip")
    r = requests.get(SHAPE_URL,timeout=60,verify=False); open(z,"wb").write(r.content)
    zipfile.ZipFile(z).extractall(tmp)
    shp = [f for f in os.listdir(tmp) if f.endswith(".shp")][0]
    gdf = gpd.read_file(os.path.join(tmp, shp)).to_crs(2163)
    gdf = gdf[gdf["STUSPS"].isin(CONTIG_ST)].reset_index(drop=True)
    states = gdf.STUSPS.tolist()
    G = nx.Graph(); G.add_nodes_from(range(len(states)))
    for i,g in enumerate(gdf.geometry):
        for j in range(i+1, len(states)):
            if g.touches(gdf.geometry.iloc[j]): G.add_edge(i,j)
    A = nx.to_numpy_array(G,dtype=np.float32)
    A += np.eye(A.shape[0],dtype=np.float32)
    D = np.diag(1/np.sqrt(A.sum(1)));  A_norm = D @ A @ D
    return A_norm, states

def embed_adj(states_all:List[str]) -> torch.Tensor:
    """返回与 tensor 列数相同的 65×65 邻接矩阵（非大陆州用自环）"""
    A_sub, contig = contig_adj()
    N = len(states_all)
    A = torch.eye(N)                              
    idx = {s:i for i,s in enumerate(states_all)}
    for i,s1 in enumerate(contig):
        for j,s2 in enumerate(contig):
            if s1 in idx and s2 in idx:
                A[idx[s1], idx[s2]] = torch.tensor(A_sub[i,j])
    return A

# ──────────────────────────────  2. time series data  ──────────────────────────────
class FluSeq(Dataset):
    def __init__(self, X: torch.Tensor, p:int, h:int):
        self.X = torch.stack([X[i-p:i] for i in range(p, X.shape[0]-h+1)])
        self.y = torch.stack([X[i:i+h] for i in range(p, X.shape[0]-h+1)])
    def __len__(self): return self.X.size(0)
    def __getitem__(self,i): return self.X[i], self.y[i]

# ──────────────────────────────  3. ST‑GCN  ──────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, A: torch.Tensor, in_f:int, out_f:int):
        super().__init__(); self.register_buffer("A",A); self.lin = nn.Linear(in_f,out_f)
    def forward(self,x): return self.lin(self.A @ x)

class STGCN(nn.Module):
    def __init__(self, A:torch.Tensor, p:int, N:int, hidden:int, depth:int, h:int):
        super().__init__(); self.h=h
        gcn = []; din=1
        for _ in range(depth):
            gcn += [GCNLayer(A,din,hidden), nn.ReLU()]; din=hidden
        self.gcn = nn.Sequential(*gcn)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head= nn.Linear(hidden,h)
    def forward(self,x):                         # x:[B,p,N]
        B,p,N = x.shape; x = x.unsqueeze(-1)     # [B,p,N,1]
        feats=[self.gcn(x[:,t]) for t in range(p)]           # p * [B,N,H]
        feats = torch.stack(feats,1).permute(0,2,1,3).reshape(B*N,p,-1) # [B*N,p,H]
        out,_ = self.gru(feats); h_last=out[:,-1]
        pred = self.head(h_last).view(B,N,self.h).permute(0,2,1)        # [B,h,N]
        return pred

# ──────────────────────────────  4. Train / Forecast  ──────────────────────────────
def train(model,tensor,args):
    ds=FluSeq(tensor,args.window,args.horizon)
    tr,vl = random_split(ds,[int(0.9*len(ds)), len(ds)-int(0.9*len(ds))])
    tl,vl = DataLoader(tr,args.batch,shuffle=True), DataLoader(vl,args.batch)
    opt = torch.optim.Adam(model.parameters(),lr=args.lr); mse=nn.MSELoss()
    for ep in range(1,args.epochs+1):
        model.train(); tot=0
        for X,y in tl:
            loss=mse(model(X.to(args.device)), y.to(args.device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*X.size(0)
        if ep%10==0:
            vtot = sum(mse(model(X.to(args.device)),y.to(args.device)).item()*X.size(0) for X,y in vl)
            print(f"Ep {ep:3d}/{args.epochs} train {tot/len(tr):.4f} val {vtot/len(vl):.4f}")

def forecast(model,tensor,args):
    with torch.no_grad():
        return model(tensor[-args.window:].unsqueeze(0).to(args.device)).cpu().numpy()[0]

# ──────────────────────────────  5. CLI  ──────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    for k,d in [("window",8),("horizon",4),("hidden",64),
                ("depth",2),("batch",32),("epochs",100),("lr",1e-3)]:
        ap.add_argument(f"--{k}",type=type(d),default=d)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",default="forecast_gcn.csv")
    args=ap.parse_args()

    loaded = torch.load("data/test02_flu_counts.pt")
    tensor = loaded["tensor"].float(); states=loaded["states"]
    print("Tensor",tensor.shape)

    A_full = embed_adj(states).to(args.device)
    model = STGCN(A_full, args.window, len(states),
                  args.hidden, args.depth, args.horizon).to(args.device)

    train(model, tensor.to(args.device), args)
    preds = forecast(model, tensor.to(args.device), args)
    pd.DataFrame(preds,columns=states).assign(lead_time=lambda d:d.index+1)\
      .to_csv(args.out,index=False)
    print("Saved →",args.out)

if __name__=="__main__":
    main()
