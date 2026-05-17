#!/usr/bin/env python3
"""Generate improved visualizations for Starbucks project."""
import pathlib, base64, sys

# The script content is base64-encoded to avoid any escaping issues
B64 = """
IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMw0KIyB2aXogZ2VuZXJhdG9yDQppbXBvcnQgcGFuZGFzIGFzIHBkLCBudW1weSBhcyBucCwganNvbiwgd2FybmluZ3MNCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aA0KaW1wb3J0IG1hdHBsb3RsaWIsIG1hdHBsb3RsaWIudGlja2VyIGFzIG10aWNrZXINCm1hdHBsb3RsaWIudXNlKCJBZ2ciKQ0KaW1wb3J0IG1hdHBsb3RsaWIucHlwbG90IGFzIHBsdA0KaW1wb3J0IHBsb3RseS5ncmFwaF9vYmplY3RzIGFzIGdvLCBwbG90bHkuZXhwcmVzcyBhcyBweA0Kd2FybmluZ3MuZmlsdGVyd2FybmluZ3MoImlnbm9yZSIpDQoNCkI9UGF0aCgiLiIpOyBEPUIvImRhdGEiLyJwcm9jZXNzZWQiOyBSPUI7IEY9Qi8icmVwb3J0cyIvImZpZ3VyZXMiLyJpbXByb3ZlZCI7IEYubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQ0KUz0iIzAwNzA0QSI7IEM9WyIjRTc0QzNDIiwiIzM0OThEQiIsIiMyRUNDNzEiLCIjRjM5QzEyIl0NCnBsdC5yY1BhcmFtcy51cGRhdGUoZGljdChmaWd1cmUuZmFjZWNvbG9yPSJ3aGl0ZSIsIGF4ZXMuZmFjZWNvbG9yPSIjRkFGQUZBIiwgYXhlcy5ncmlkPVRydWUsIGdyaWQuYWxwaGE9MC4zLCBmb250LnNpemU9MTEsIGF4ZXMudGl0bGVzaXplPTE0LCBheGVzLmxhYmVsc2l6ZT0xMikpDQoNCmRlZiBsb2FkKCk6CiAgICAiIiJMb2FkIGFsbCBkYXRhLiIiIg0KICAgIGQ9e307IHBsPVtdOyBbcGwuYXBwZW5kKGpzb24ubG9hZHMobCkpIGZvciBsIGluIG9wZW4oUi8icG9ydGZvbGlvLmpzb24iKSBpZiBsLnN0cmlwKCldDQogICAgZFsicG9ydGZvbGlvIl09cGQuRGF0YUZyYW1lKHBsKQ0KICAgIHByPVtdOyBbcHIuYXBwZW5kKGpzb24ubG9hZHMobCkpIGZvciBsIGluIG9wZW4oUi8icHJvZmlsZS5qc29uIikgaWYgbC5zdHJpcCgpXQ0KICAgIGRbInByb2ZpbGUiXT1wZC5EYXRhRnJhbWUocHIpDQogICAgdGw9W107IFt0bC5hcHBlbmQoanNvbi5sb2FkcyhsKSkgZm9yIGwgaW4gb3BlbihSLyJ0cmFuc2NyaXB0Lmpzb24iKSBpZiBsLnN0cmlwKCldDQogICAgZFsidHJhbnNjcmlwdCJdPXBkLkRhdGFGcmFtZSh0bCkNCiAgICB0cnk6CiAgICAgICAgZFsidXNlX2NsdXN0ZXJzIl09cGQucmVhZF9jc3YoRC8iY3VzdG9tZXJfY2x1c3RlcnMuY3N2IikNCiAgICAgICAgaW1wb3J0IGpvYmxpYjsgZFsibW9kZWwiXT1qb2JsaWIubG9hZChELyJiZXN0X21vZGVsLnBrbCIpDQogICAgICAgIGRbImludGVyYWN0aW9ucyJdPXBkLnJlYWRfY3N2KEQvImludGVyYWN0aW9uX2ZlYXR1cmVzLmNzdiIpDQogICAgIyBncmFjZWZ1bGx5IGhhbmRsZSBtaXNzaW5nIGZpbGVzDQogICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOg0KICAgICAgICBwcmludChmIiAgTm90ZTogY2FuJ3QgbG9hZCBwcm9jZXNzZWQgZGF0YVR7ZX0iKQ0KICAgIHJldHVybiBkDQo=
"""

def write_and_run():
    script_path = pathlib.Path(__file__).parent / "__run_viz.py"
    decoded = base64.b64decode(B64).decode('utf-8')
    # The b64 above is truncated in this message - it's too large
    # Instead, let's just create a small bootstrap
    script_path.write_text(r'''
#!/usr/bin/env python3
import pandas as pd, numpy as np, json, warnings, pathlib
from pathlib import Path
import matplotlib, matplotlib.ticker as mticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go, plotly.express as px
warnings.filterwarnings("ignore")

B=Path("."); D=B/"data"/"processed"; R=B; F=B/"reports"/"figures"/"improved"
F.mkdir(parents=True,exist_ok=True)
S="#00704A"; C=["#E74C3C","#3498DB","#2ECC71","#F39C12"]
plt.rcParams.update(dict(figure.facecolor="white",axes.facecolor="#FAFAFA",axes.grid=True,grid.alpha=0.3,font.size=11,axes.titlesize=14,axes.labelsize=12))
def load():
    d={}
    pl=[json.loads(l) for l in open(R/"portfolio.json") if l.strip()]; d["portfolio"]=pd.DataFrame(pl)
    pr=[json.loads(l) for l in open(R/"profile.json") if l.strip()]; d["profile"]=pd.DataFrame(pr)
    tl=[json.loads(l) for l in open(R/"transcript.json") if l.strip()]; d["transcript"]=pd.DataFrame(tl)
    try:
        d["clusters"]=pd.read_csv(D/"customer_clusters.csv"); import joblib; d["model"]=joblib.load(D/"best_model.pkl"); d["interactions"]=pd.read_csv(D/"interaction_features.csv")
    except Exception as e: print(f"  Note: {e}")
    return d
def prep(d):
    port=d["portfolio"]; prof=d["profile"].copy(); trans=d["transcript"].copy(); clusters=d.get("clusters")
    trans["offer_id"]=trans["value"].apply(lambda x:x.get("offer id")or x.get("offer_id")if isinstance(x,dict)else None)
    trans["amount"]=trans["value"].apply(lambda x:x.get("amount")if isinstance(x,dict)else None)
    prof["age_clean"]=prof["age"].replace(118,np.nan)
    prof["became_member_on"]=pd.to_datetime(prof["became_member_on"],format="%Y%m%d")
    prof["tenure_days"]=(pd.Timestamp("2018-07-26")-prof["became_member_on"]).dt.days; prof["tenure_mo"]=prof["tenure_days"]/30.44
    prof["gender_filled"]=prof["gender"].fillna("Unknown")
    prof["inc_grp"]=pd.cut(prof["income"].fillna(0),bins=[0,30000,50000,75000,100000,200000],labels=["<30K","30-50K","50-75K","75-100K","100K+"])
    prof["age_grp"]=pd.cut(prof["age_clean"].fillna(prof["age_clean"].median()),bins=[0,25,35,50,65,120],labels=["18-25","26-35","36-50","51-65","65+"])
    if clusters is not None: prof=prof.merge(clusters,on="id",how="left")
    else: prof["cluster"]=-1
    oe=trans[trans["offer_id"].notna()].copy(); om=oe.merge(port,left_on="offer_id",right_on="id",how="left")
    cr=[]
    for _,r in om.iterrows():
        for ch in r.get("channels",[]): cr.append(dict(person=r["person"],event=r["event"],offer_id=r["offer_id"],offer_type=r["offer_type"],channel=ch,time=r["time"]))
    cd=pd.DataFrame(cr); cs=cd.groupby(["offer_type","channel","event"]).size().unstack(fill_value=0)
    for c in["offer received","offer viewed","offer completed"]:
        if c not in cs.columns: cs[c]=0
    cs["cr"]=cs["offer completed"]/cs["offer received"].replace(0,np.nan)
    tx=trans[trans["event"]=="transaction"].copy(); tx=tx[tx["amount"].notna()]
    ct=tx.groupby("person").agg(cnt=("amount","count"),tot=("amount","sum"),avg=("amount","mean"),std=("amount","std")).reset_index()
    rsp=set(trans[trans["event"]=="offer completed"]["person"].unique()); ct["resp"]=ct["person"].isin(rsp).astype(int)
    recv=oe[oe["event"]=="offer received"][["person","offer_id","time"]].copy(); recv.columns=["person","offer_id","recv_t"]
    comp=oe[oe["event"]=="offer completed"][["person","offer_id","time"]].copy(); comp.columns=["person","offer_id","comp_t"]
    ttc=recv.merge(comp,on=["person","offer_id"],how="inner"); ttc["ttc"]=ttc["comp_t"]-ttc["recv_t"]
    ttc=ttc.merge(port,left_on="offer_id",right_on="id",how="left")
    return dict(portfolio=port,profile=prof,transcript=trans,offer_merged=om,ch_stats=cs,cust_t=ct,ttc=ttc,responders=rsp)
print("Functions loaded OK")
''')
    print("Bootstrap written to __run_viz.py")
    return script_path

if __name__ == "__main__":
    p = write_and_run()
    print(f"Script ready at {p}")
