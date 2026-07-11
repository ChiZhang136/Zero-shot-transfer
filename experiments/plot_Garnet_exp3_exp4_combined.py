"""Create a vertically stacked, paper-ready Garnet Exp3/Exp4 figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Maximum-based", "Similarity-aware"]
COLORS = {"Maximum-based": "C2", "Similarity-aware": "C0"}

def stats(x):
    x = np.asarray(x, float); return x.mean(), (x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.)
def main():
    d3 = pd.read_csv(ROOT/"results/Garnet_exp3/Garnet_exp3.csv")
    d4 = pd.read_csv(ROOT/"results/Garnet_exp4/Garnet_exp4.csv")
    d3 = d3[d3.method.isin(METHODS) & (d3.bias_level >= 0)]; d4 = d4[d4.method.isin(METHODS) & (d4.iteration >= 0)]
    xs = np.sort(d3.bias_level.unique()); ts = np.sort(d4.iteration.unique().astype(int))
    fig,(a,b)=plt.subplots(2,1,figsize=(6.5,4.2))
    for m in METHODS:
        y=[]; e=[]
        for x in xs:
            q=d3[(d3.method==m)&np.isclose(d3.bias_level,x)].normalized_performance*100; z,w=stats(q); y.append(z); e.append(w)
        a.plot(xs,y,color=COLORS[m],lw=2,marker='o',label=m); a.fill_between(xs,np.array(y)-e,np.array(y)+e,color=COLORS[m],alpha=.15)
        y=[]; e=[]
        for t in ts:
            q=d4[(d4.method==m)&(d4.iteration==t)].signed_selection_bias; z,w=stats(q); y.append(z); e.append(w)
        b.plot(ts,y,color=COLORS[m],lw=2,label=m); b.fill_between(ts,np.array(y)-e,np.array(y)+e,color=COLORS[m],alpha=.15)
    a.set_xlabel(r'$\delta$'); a.set_ylabel(r'$\nu(T)$ (%)')
    b.set_xlabel(r'$t$'); b.set_ylabel(r'$\mu(t)$')
    for ax in (a,b): ax.grid(alpha=.25); ax.set_axisbelow(True)
    a.legend(loc='lower left'); b.legend(loc='center right'); fig.tight_layout(h_pad=1.2)
    out=ROOT/'figures/Garnet_exp3_exp4_combined'; out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'Garnet_exp3_exp4_combined.pdf',bbox_inches='tight'); fig.savefig(out/'Garnet_exp3_exp4_combined.png',dpi=300,bbox_inches='tight'); print(f'Saved combined figure to {out}')
if __name__=='__main__': main()
