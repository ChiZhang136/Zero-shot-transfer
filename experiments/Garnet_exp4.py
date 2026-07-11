import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.garnet import generate_target_garnet, generate_sources_from_target
from src.algorithms import robust_bellman_optimal_source_radius
from src.utils import similarity_weights, ensure_dir
RESULT_DIR = PROJECT_ROOT / "results" / "Garnet_exp4"
ensure_dir(RESULT_DIR)
NUM_STATES, NUM_ACTIONS, BRANCHING_FACTOR = 30, 4, 3
DISCOUNT = .95
SOURCE_GAMMAS = np.array([.10, .20, .30, .40])
ITERATIONS, SYNC_PERIOD, STEPSIZE = 400, 5, .05
NUM_SEEDS, NUM_NOISE_TRAJECTORIES = 20, 60
DIAGNOSTIC_NOISE_LEVEL = 1.0
METHOD_ORDER = ["Maximum-based", "Similarity-aware"]
def l1(P, P0): return np.sum(np.abs(P-P0[None]), axis=-1)
def anti(rng, shape):
    x=rng.uniform(-1,1,size=(shape[0]//2,*shape[1:]))
    return np.concatenate([x,-x])
def agg(q, method, w):
    return np.max(q,axis=1) if method=="Maximum-based" else np.tensordot(q,w,axes=(1,0))
def run(P, r, radii, method, w, noise, seed):
    B,K,S,A=noise.shape[0],P.shape[0],P.shape[1],P.shape[2]
    q=np.zeros((B,K,S,A)); rows=[]
    for t in range(ITERATIONS):
        new=np.empty_like(q)
        for b in range(B):
            for k in range(K):
                tq=robust_bellman_optimal_source_radius(P=P[k],rewards=r,radius=radii[k],discount=DISCOUNT,Q=q[b,k],q="inf")
                new[b,k]=(1-STEPSIZE)*q[b,k]+STEPSIZE*(tq+DIAGNOSTIC_NOISE_LEVEL*noise[b,t,k])
        q=new
        if (t+1)%SYNC_PERIOD==0:
            ae=agg(q,method,w); ma=ae.mean(0); ml=q.mean(0)
            am=np.max(ml,axis=0) if method=="Maximum-based" else np.tensordot(w,ml,axes=(0,0))
            z=ma-am
            rows.append({"seed":seed,"iteration":t+1,"method":method,"noise_level":DIAGNOSTIC_NOISE_LEVEL,"signed_selection_bias":float(z.mean()),"positive_selection_bias":float(np.maximum(z,0).mean()),"selection_bias_inf_norm":float(np.abs(z).max()),"num_noise_trajectories":B,"selection_bias_definition":"E[Agg(Q_xi)] - Agg(E[Q_xi])","aggregation_is_written_back":True})
            q=np.repeat(ae[:,None],K,axis=1)
    return rows
def main():
    rows=[]; p=tqdm(total=NUM_SEEDS*2,desc="Garnet Exp4 selection-bias",unit="run")
    for seed in range(NUM_SEEDS):
        target=generate_target_garnet(num_states=NUM_STATES,num_actions=NUM_ACTIONS,branching_factor=BRANCHING_FACTOR,reward_range=(0,1),discount=DISCOUNT,seed=1000+seed)
        P,g,rho=generate_sources_from_target(target_mdp=target,source_gammas=SOURCE_GAMMAS,branching_factor=BRANCHING_FACTOR,seed=2000+seed,p_norm=1)
        radii=l1(P,target.transitions); w=similarity_weights(g,eps=1e-6,power=1)
        noise=anti(np.random.default_rng(3000+seed),(NUM_NOISE_TRAJECTORIES,ITERATIONS,len(g),NUM_STATES,NUM_ACTIONS))
        for method in METHOD_ORDER:
            x=run(P,target.rewards,radii,method,None if method=="Maximum-based" else w,noise,seed)
            for row in x: row.update({"actual_source_gammas":str(g.tolist()),"similarity_weights":str(w.tolist()),"penalty_type":"state_action_local","p_norm":"1","robust_q":"inf","sync_period":SYNC_PERIOD,"stepsize":STEPSIZE})
            rows.extend(x); p.update(1)
    p.close(); pd.DataFrame(rows).to_csv(RESULT_DIR/"Garnet_exp4.csv",index=False)
if __name__=="__main__": main()
