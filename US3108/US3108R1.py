"""Bayesian Movement-Line Learning | GPU-capable, fixed spatial order.

G2 prior signal: Z_i,t = Y_t(i) - (1.0)*mean_j Y_t(j).
One-step workflow: select beta_hat_t using completed-transition proper kernel
scores, forecast Y_(t+1), then update A_t conditional on fixed beta_hat_t.
No beta posterior, no use of future observations in parameter selection.

Six change channels per origin and distance shell:
 M+, M-, M0 (fractions); V+, V- (mean change magnitudes);
 V0 (mean absolute starting value for the unchanged group).
The optional feature_resolution is only used to compare maps in the kernel;
forecasting the continuous Gaussian model itself is not rounded.
Movement-Line uses Y_(t+1) = Y_t + K A_t WITHOUT separate observation noise;
baseline is the Gaussian random walk with empirical mean change and standard deviation.
The conditional B replications are unnecessary when A_t fixes the whole map.
RMSE/MAE compare the county-wise mean of D pre-generated full forecast
maps against the observed full Y_(t+1) map (3108 counties); the same D draws
are also used to evaluate the predictive distribution with the kernel score.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.io import mmread
from scipy.special import ndtr

FEATURES=('M_plus','M_minus','M_zero','V_plus','V_minus','V_zero')
FIELD='L01_unemployment_pct'
# G-variant signal is ONLY for the movement-line intensity prior.
G_NAME='G2'
G_LAMBDA=1.0
# Nine alpha choices from -0.004 to 0.004 (inclusive), step 0.001.
# Candidate grids and remaining numerical settings are defined below.
COUNTY_ALPHA=tuple(round(-0.004+i*0.001, 6) for i in range(9))
COUNTY_TAU=tuple(round(0.20+i*0.005, 6) for i in range(21))
SIM_ALPHA=(0.01,0.03,0.08)
SIM_TAU=(0.35,0.65,0.95)
GRID_ALPHA=(0.01,0.02,0.03,0.05,0.08)
GRID_TAU=(0.35,0.65,0.95)


def choose_device(x):
    if x=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.')
    return torch.device('cuda' if (x=='auto' and torch.cuda.is_available()) else ('cpu' if x=='auto' else x))


def rng_for(device,*ints):
    seed=int(np.random.SeedSequence([int(a) for a in ints]).generate_state(1)[0])
    return torch.Generator(device=device.type).manual_seed(seed)


def write_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    os.replace(tmp,p)


def write_csv(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    os.replace(tmp,p)


def file_sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def make_shells_from_W(W,R):
    """Exact shortest-path distance shells, no reliance on precomputed order 2+ files.
    Uses graph reachability by sparse matrix multiplication, removing visited pairs.
    """
    n=W.shape[0]
    I=sparse.eye(n,format='csr',dtype=np.float64)
    shells=[I]
    visited=I.copy()
    frontier=I.copy()
    for r in range(1,R+1):
        frontier=(frontier@W).tocsr()
        frontier.data[:]=1.0
        frontier.eliminate_zeros()
        # exact r-th shortest path by excluding all previously reached pairs
        frontier=frontier-frontier.multiply(visited)
        frontier.eliminate_zeros()
        if frontier.nnz:
            frontier.data[:]=1.0
        shells.append(frontier)
        visited=visited+frontier
        visited.data[:]=1.0
    return shells


def validate_W(W):
    W=W.tocsr().astype('float64');W.eliminate_zeros();W.sum_duplicates()
    if W.shape[0]!=W.shape[1] or (W-W.T).nnz or np.any(W.diagonal()!=0) or np.any(W.data!=1):
        raise ValueError('order_1.mtx must be symmetric, binary, with diagonal zero')
    return W


def load_county(folder,R,expected_n=3108):
    idxfile=folder/'county_index.csv';monthfile=folder/'county_month.csv';wfile=folder/'order_1.mtx'
    for p in (idxfile,monthfile,wfile):
        if not p.is_file():raise FileNotFoundError(f'Missing {p}. Provide --data-dir with source CSVs and order_1.mtx')
    idx=pd.read_csv(idxfile,dtype={'fips':str})
    if not {'fips','matrix_row'}.issubset(idx.columns):raise ValueError('county_index.csv requires fips and matrix_row')
    n=len(idx)
    if expected_n and n!=expected_n:raise ValueError(f'Expected {expected_n} counties, found {n}; will not drop counties')
    if len(set(idx.fips))!=n or idx.matrix_row.astype(int).tolist()!=list(range(1,n+1)):
        raise ValueError('FIPS must be unique; matrix_row must be 1..n and aligned with order_1.mtx')
    raw=pd.read_csv(monthfile,dtype={'county_fips':str,'month':str})
    if not {'county_fips','month',FIELD}.issubset(raw.columns):
        raise ValueError(f'county_month.csv requires county_fips, month, {FIELD}')
    if raw.duplicated(['county_fips','month']).any():raise ValueError('Duplicate county-month rows')
    months=sorted(raw.month.dropna().unique().tolist())
    panel=raw.pivot(index='county_fips',columns='month',values=FIELD).reindex(index=idx.fips.to_list(),columns=months)
    Y=panel.to_numpy('float64').T.copy()
    if Y.shape[0]<3 or not np.isfinite(Y).all():raise ValueError('County panel must be complete and have >=3 observed months')
    if Y.min()<0 or Y.max()>100:raise ValueError('Invalid unemployment rate range [0,100]')
    W=validate_W(mmread(wfile).tocsr())
    if W.shape!=(n,n):raise ValueError('W and panel county dimensions disagree')
    # Optional precomputed shells are validated against the exact graph construction.
    shells=make_shells_from_W(W,R)
    provided=[idxfile,monthfile,wfile]
    for r in range(2,R+1):
        p=folder/f'order_{r}.mtx'
        if p.is_file():
            ext=mmread(p).tocsr().astype('float64');ext.eliminate_zeros()
            if ext.shape!=(n,n) or (ext-shells[r]).nnz:
                raise ValueError(f'{p} disagrees with graph-distance shell from order_1.mtx')
            provided.append(p)
    return idx,months,Y,shells,provided


class MLEngine:
    def __init__(self,shells,R,device,sigma_obs=1.12965,baseline_mean=-0.01081,epsilon=.5,s_star=.5,
                 pair_mode='paired',gpu_block=8,dtype='float64',feature_resolution=0.):
        if sigma_obs<=0 or epsilon<=0 or s_star<=0 or gpu_block<1:raise ValueError('Scale parameters must be positive')
        if pair_mode not in ('paired','full'):raise ValueError('Invalid pair mode')
        if feature_resolution<0:raise ValueError('feature-resolution must be >=0')
        self.R=R;self.n=shells[0].shape[0];self.device=device
        self.g_lambda=G_LAMBDA
        self.dtype=torch.float32 if dtype=='float32' else torch.float64
        self.sigma_obs=float(sigma_obs);self.baseline_mean=float(baseline_mean);self.epsilon=float(epsilon);self.s_star=float(s_star)
        self.pair_mode=pair_mode;self.gpu_block=gpu_block;self.feature_resolution=feature_resolution
        self.ks=2.0**(-np.arange(R+1,dtype='float64'))
        self.shells_cpu=[h.tocsr().astype('float64') for h in shells[:R+1]]
        sizes=np.stack([np.asarray(h.sum(axis=1)).ravel() for h in self.shells_cpu],axis=1)
        denom=(sizes*self.ks[None,:]).sum(1)
        if not np.all(denom>0):raise ValueError('No positive zero-order weights')
        weights=sizes*self.ks[None,:]/(self.n*denom[:,None])
        if not np.isclose(weights.sum(),1):raise RuntimeError('Kernel origin/order weights do not sum to 1')
        self.coord=torch.as_tensor(np.repeat(weights[:,:,None]/6.,6,axis=2).reshape(-1),dtype=self.dtype,device=device)
        self.sizes=torch.as_tensor(sizes,dtype=self.dtype,device=device)
        def tensor_sparse(H):
            a=H.tocoo();inds=np.vstack((a.row,a.col)).astype('int64')
            return torch.sparse_coo_tensor(torch.as_tensor(inds,device=device),
                            torch.as_tensor(a.data,device=device,dtype=self.dtype),a.shape).coalesce()
        # Order zero uses identity; remaining shells may be large, held only as sparse tensors.
        self.H=[None]+[tensor_sparse(x) for x in self.shells_cpu[1:]]
        KK=sum((float(k)*h for k,h in zip(self.ks,self.shells_cpu)),
               sparse.csr_matrix((self.n,self.n),dtype='float64')).tocsr()
        self.K_cpu=KK
        self.diag_KKt=np.asarray(KK.multiply(KK).sum(axis=1)).reshape(-1)
        if self.n<=150:
            self.K=torch.as_tensor(KK.toarray(),device=device,dtype=self.dtype);self.dense=True
        else:
            self.K=tensor_sparse(KK);self.dense=False
        if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False

    def as_tensor(self,x):return torch.as_tensor(x,device=self.device,dtype=self.dtype)

    def multiply_K(self,x):
        shape=x.shape;b=x.reshape(-1,self.n)
        v=b@self.K.T if self.dense else torch.sparse.mm(self.K,b.T).T
        return v.reshape(shape)

    def origin_signal(self,current):
        """Z_t = g(Y_t), using ONLY the complete map observed at time t."""
        return current-self.g_lambda*current.mean()

    def make_maps(self,current,alpha,tau,draws,generator):
        z=float(alpha)*self.origin_signal(current)[None,:]+float(tau)*torch.randn((draws,self.n),device=self.device,dtype=self.dtype,generator=generator)
        u=self.multiply_K(z)
        # No additional Gaussian observation noise in the Movement-Line forecast.
        # All ML uncertainty arises from A_t = alpha*g(Y_t) + tau*xi_t.
        return current[None,:]+u,z,u

    def baseline_maps(self,current,draws,generator):
        return current[None,:]+self.baseline_mean+self.sigma_obs*torch.randn((draws,self.n),device=self.device,dtype=self.dtype,generator=generator)

    def prediction_mean(self,current,alpha):
        return current+float(alpha)*self.multiply_K(self.origin_signal(current))

    def kernel_scale(self,Y,t):
        if t==0:return self.s_star
        return max(self.s_star,float(torch.quantile((Y[t]-Y[t-1]).abs(),0.5).item()))

    def flat_features(self,current,maps,scale):
        """Each feature: [draw,n,order,coord] -> flatten; normalized to [0,1]."""
        maps=maps.reshape(-1,self.n);L=maps.shape[0]
        if self.feature_resolution>0:
            h=self.feature_resolution
            comp=torch.round(maps/h)*h
            initial=torch.round(current/h)*h
        else:
            comp=maps;initial=current
        diff=comp-initial[None,:]
        p=(diff>0).to(self.dtype);m=(diff<0).to(self.dtype);z=(diff==0).to(self.dtype)
        raw=torch.stack((p,m,z,diff.clamp_min(0.),(-diff).clamp_min(0.),initial.abs()[None,:]*z),dim=-1)
        row=raw.permute(0,2,1).reshape(-1,self.n).T.contiguous() # n x (L*6)
        v=[]
        for r in range(self.R+1):
            if r==0:
                values=raw
            else:
                vv=torch.sparse.mm(self.H[r],row).T.reshape(L,6,self.n).permute(0,2,1)
                values=vv/self.sizes[:,r].clamp_min(1)[None,:,None]
            # Normalize magnitude channels; three M fractions remain unchanged.
            out=torch.cat((values[...,:3],values[...,3:]/(scale+values[...,3:])),dim=-1)
            v.append(out)
        return torch.stack(v,dim=2).reshape(L,-1)

    def k_to_obs(self,x,obs):
        d=((x-obs)*self.coord).abs().sum(-1)
        return torch.exp(-d/self.epsilon)

    def kernel_score(self,x,obs):
        L=len(x)
        if L<2:raise ValueError('At least two full-map samples required for proper score')
        first=self.k_to_obs(x,obs).mean()
        if self.pair_mode=='paired':
            # Cyclic adjacent *distinct* independent draws: unbiased E[k(X,X')].
            second=torch.exp(-((x-x.roll(-1,0))*self.coord).abs().sum(-1)/self.epsilon).mean()
        else:
            z=x*self.coord
            total=torch.zeros((),device=self.device,dtype=self.dtype)
            for s in range(0,L,self.gpu_block):
                for j in range(s,L,self.gpu_block):
                    ds=torch.cdist(z[s:s+self.gpu_block].unsqueeze(0),z[j:j+self.gpu_block].unsqueeze(0),p=1.)[0]
                    b=torch.exp(-ds/self.epsilon)
                    total+=torch.triu(b,diagonal=1).sum() if s==j else b.sum()
            second=2*total/(L*(L-1))
        return float((1-2*first+second).item())

    @torch.no_grad()
    def score_forecast(self,current,observed,scale,maps):
        obs=self.flat_features(current,observed[None,:],scale)[0]
        return self.kernel_score(self.flat_features(current,maps,scale),obs)

    @torch.no_grad()
    def score_candidates(self,current,observed,scale,grid,L,seed,t):
        obs=self.flat_features(current,observed[None,:],scale)[0]
        # Common random numbers: every (alpha, tau) candidate for transition t
        # uses exactly the same iid N(0,I) innovations. This reduces random
        # variation in differences between the candidate kernel scores.
        innovation=self.multiply_K(torch.randn(
            (L,self.n),device=self.device,dtype=self.dtype,
            generator=rng_for(self.device,seed,t,11)))
        trend=self.multiply_K(self.origin_signal(current))
        results=[]
        for a,tt in grid:
            maps=current[None,:]+float(a)*trend[None,:]+float(tt)*innovation
            results.append(self.kernel_score(self.flat_features(current,maps,scale),obs))
        return np.asarray(results,dtype='float64')

    @torch.no_grad()
    def update_intensities(self,current,observed,scale,beta,M,B,seed,t,delta=1.):
        a,tt=map(float,beta)
        rng=rng_for(self.device,seed,t,22)
        signal=self.origin_signal(current)
        A=a*signal[None,:]+tt*torch.randn((M,self.n),device=self.device,dtype=self.dtype,generator=rng)
        U=self.multiply_K(A)
        obs=self.flat_features(current,observed[None,:],scale)[0]
        weights=[]
        # Conditional on A_t, the Movement-Line map is deterministic.
        # B conditional draws would be identical, so score each A_t once.
        chunk=max(1,min(M,256))
        for s in range(0,M,chunk):
            u=U[s:s+chunk]
            maps=current[None,:]+u
            sim=self.flat_features(current,maps,scale)
            weights.append(self.k_to_obs(sim,obs))
        avg=torch.cat(weights)
        w=avg/avg.sum()
        if not torch.isfinite(w).all():raise FloatingPointError('Nonfinite kernel posterior weights')
        post=w@A
        prior=a*signal
        return dict(A_prior=prior,A_post=post,U_prior=self.multiply_K(prior),
                    U_post=self.multiply_K(post),pplus=(w[:,None]*(A>delta)).sum(0),
                    pminus=(w[:,None]*(A<-delta)).sum(0),
                    ess=float((w.square().sum().reciprocal()).item()),
                    mean_match=float(avg.mean().item()),weights=w)


def initial_beta(grid):return np.asarray(grid[len(grid)//2],dtype='float64').copy()


def point_metrics(pred,actual):
    d=np.asarray(pred)-np.asarray(actual)
    return dict(rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(np.abs(d))),
                mean_error=float(np.mean(d)))


def direction_counts_from_mean(mean,current,var,step=.1):
    std=np.sqrt(np.asarray(var,dtype=float))
    lower=(np.asarray(current)-step/2-np.asarray(mean))/std
    upper=(np.asarray(current)+step/2-np.asarray(mean))/std
    dn=ndtr(lower);up=1-ndtr(upper);zero=np.clip(1-up-dn,0,1)
    return dict(plus=float(up.sum()),minus=float(dn.sum()),zero=float(zero.sum()))


def observed_counts(start,end,step=.1):
    a=np.round(np.asarray(start)/step).astype('int64')
    b=np.round(np.asarray(end)/step).astype('int64')
    d=b-a
    return dict(plus=int((d>0).sum()),minus=int((d<0).sum()),zero=int((d==0).sum()))

# ============================================================================
# U.S. 48-state county panel: exactly 3,108 regions, R set at file generation.
# Data files: county_index.csv, county_month.csv, order_1.mtx.
# Distinct shortest-path order 2..R shells are reconstructed from order_1;
# precomputed order_r.mtx are checked when present, never assumed available.
# ============================================================================
R_FIXED=1


@torch.no_grad()
def generate_forecasts(engine,current,beta,D,seed,t):
    """At time t, generate forecasts using Y_t and previously selected beta only.

    The county-wise average over the D maps is used for RMSE/MAE;
    all D maps also define the distribution for the joint proper kernel score.
    """
    movement_maps,_,_=engine.make_maps(current,beta[0],beta[1],D,
                                      rng_for(engine.device,seed,t,31))
    baseline_maps=engine.baseline_maps(current,D,
                                      rng_for(engine.device,seed,t,32))
    return movement_maps,baseline_maps


@torch.no_grad()
def eval_forecast(engine,current,observed,scale,movement_maps,baseline_maps):
    """After Y_(t+1) arrives, compare the pre-generated forecasts to truth."""
    sp=engine.score_forecast(current,observed,scale,movement_maps)
    sb=engine.score_forecast(current,observed,scale,baseline_maps)
    actual=observed.cpu().numpy().astype('float64')
    # Compute ONE predicted complete map by averaging D forecasts county-wise.
    # RMSE/MAE compare those n predicted values with the n observed values;
    # they do not use a single random draw or pool D*n separate errors.
    movement_prediction=movement_maps.mean(dim=0,dtype=torch.float64).cpu().numpy()
    baseline_prediction=baseline_maps.mean(dim=0,dtype=torch.float64).cpu().numpy()
    mv=point_metrics(movement_prediction,actual)
    bb=point_metrics(baseline_prediction,actual)
    return sp,sb,mv,bb,movement_prediction,baseline_prediction


def figures(learning,rolling,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    if learning:
        for val,fn in [('selected_alpha_before','01_selected_alpha.png'),('selected_tau_before','02_selected_tau.png')]:
            fig,ax=plt.subplots(figsize=(9,4));ax.plot([r['t'] for r in learning],[r[val] for r in learning]);
            ax.set(xlabel='Transition t',ylabel=val,title=f'R={R_FIXED} predictive-score parameter selection')
            ax.grid(alpha=.3);fig.tight_layout();fig.savefig(folder/fn,dpi=160);plt.close(fig)
    if rolling:
        x=[r['target_month'] for r in rolling]
        for keys,ylabel,fn in [
            (('baseline_score','movement_score'),f'R={R_FIXED} six-feature kernel score','03_scores.png'),
            (('baseline_rmse','movement_rmse'),'D-draw mean-map RMSE','04_rmse.png'),
            (('baseline_mae','movement_mae'),'D-draw mean-map MAE','05_mae.png')]:
            fig,ax=plt.subplots(figsize=(10,4))
            for key in keys:ax.plot(x,[r[key] for r in rolling],marker='.',label=key)
            ax.set(xlabel='Evaluated month',ylabel=ylabel,title=f'R={R_FIXED}: {ylabel}')
            ax.tick_params(axis='x',rotation=75);ax.grid(alpha=.3);ax.legend()
            fig.tight_layout();fig.savefig(folder/fn,dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=f'US3108 R={R_FIXED} GPU, 6-feature prediction + Bayesian latent updating')
    ap.add_argument('--data-dir',type=Path,default=Path('.'))
    ap.add_argument('--output',type=Path,default=Path(f'US3108R{R_FIXED}_GPU'))
    ap.add_argument('--expected-n',type=int,default=3108)
    ap.add_argument('--max-transitions',type=int,default=0,help='0 means all observed transitions')
    ap.add_argument('--eval-months',type=int,default=24)
    ap.add_argument('--history-window',type=int,choices=(0,12,24),default=0,
                    help='0 = all completed transitions; 12 or 24 = most recent completed transitions only')
    ap.add_argument('--L',type=int,default=512);ap.add_argument('--M',type=int,default=128)
    ap.add_argument('--B',type=int,default=4);ap.add_argument('--D',type=int,default=512)
    ap.add_argument('--sigma-obs',type=float,default=1.12965)
    ap.add_argument('--baseline-mean',type=float,default=-0.01081)
    ap.add_argument('--epsilon',type=float,default=.5)
    ap.add_argument('--s-star',type=float,default=.5)
    ap.add_argument('--delta',type=float,default=1.)
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--dtype',choices=('float32','float64'),default='float64',
                    help='float64 double precision by default; float32 available for speed/memory')
    ap.add_argument('--pair-mode',choices=('paired','full'),default='paired',
                    help='Paired distinct draws: unbiased O(L) estimate; full is costly')
    ap.add_argument('--gpu-block',type=int,default=8)
    ap.add_argument('--feature-resolution',type=float,default=.1,
                    help='0.1 = score on data reported to 1 decimal; 0 = no rounding')
    ap.add_argument('--reported-step',type=float,default=.1)
    ap.add_argument('--seed',type=int,default=20261117)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    if args.L<2 or args.D<2 or args.M<1 or args.B<1 or args.eval_months<1 or args.reported_step<=0:
        ap.error('L, D >= 2; M, B, eval-months >=1; reported-step >0')
    data=args.data_dir.resolve()
    idx,months,panel,shells,src=load_county(data,R_FIXED,args.expected_n)
    T=len(months)-1 if args.max_transitions==0 else min(args.max_transitions,len(months)-1)
    if T<2 or args.eval_months>T:ap.error('Need at least two transitions and eval-months <= T')
    device=choose_device(args.device)
    engine=MLEngine(shells,R_FIXED,device,sigma_obs=args.sigma_obs,baseline_mean=args.baseline_mean,epsilon=args.epsilon,
                    s_star=args.s_star,pair_mode=args.pair_mode,gpu_block=args.gpu_block,
                    dtype=args.dtype,feature_resolution=args.feature_resolution)
    Y=engine.as_tensor(panel)
    grid=np.asarray([(a,tt) for a in COUNTY_ALPHA for tt in COUNTY_TAU],dtype='float64')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=dict(version=f'US3108_ML_NO_OBS_NOISE_R{R_FIXED}_EVAL',g_variant=G_NAME,g_lambda=G_LAMBDA,
        g_formula='Z_t(i)=Y_t(i)-g_lambda*mean(Y_t)',
        R=R_FIXED,eval_R=R_FIXED,n=engine.n,T=T,months=len(months),first_month=months[0],last_month=months[T],
        eval_months=args.eval_months,history_window=args.history_window,
        candidate_noise='common_normal_per_transition',L=args.L,M=args.M,B=args.B,D=args.D,
        alpha_grid=COUNTY_ALPHA,tau_grid=COUNTY_TAU,k_weights=engine.ks.tolist(),
        sigma_obs=args.sigma_obs,baseline_mean_change=args.baseline_mean,epsilon=args.epsilon,s_star=args.s_star,
        device=str(device),dtype=args.dtype,pair_mode=args.pair_mode,seed=args.seed,
        delta=args.delta,features=FEATURES,feature_resolution=args.feature_resolution,
        reported_step=args.reported_step,
        time_selection='selected before observing Y_(t+1) using completed transitions',
        history_selection=('all completed transitions' if args.history_window==0 else
                           f'most recent {args.history_window} completed transitions'),
        posterior='Q(A_t|Y_t,Y_(t+1),fixed beta_hat_t)',
        predictive_mean='exact Y_t+alpha_hat_t*K*g(Y_t); no MC noise',
        movement_observation_noise_sd=0.0,baseline_observation_noise_sd=args.sigma_obs,
        movement_predictive_covariance='tau_hat_t^2*K*K.T',
        conditional_ml_draws='deterministic given A_t; B ignored',
        rmse_mae_definition='RMSE/MAE between county-wise mean of D forecast maps generated at t and observed Y_(t+1), across n counties',
        point_metrics_definition='baseline_point/movement_point metrics separately evaluate analytic conditional-mean forecasts',
        files_sha256={p.name:file_sha(p) for p in src})
    settings=out/'00_settings.json';check=out/'00_checkpoint.json'
    if settings.exists():
        if json.loads(settings.read_text(encoding='utf8'))!=json.loads(json.dumps(cfg)):
            ap.error('Output settings differ, use new --output directory')
    else:write_json(settings,cfg)
    if args.resume:
        if not check.is_file():ap.error('--resume requires an existing checkpoint')
        state=json.loads(check.read_text(encoding='utf8'))
        start=int(state['done_transitions']);totals=np.asarray(state['total_scores'],dtype='float64')
        history_scores=[np.asarray(x,dtype='float64') for x in state.get('history_scores',[])]
        if args.history_window and start and not history_scores:
            ap.error('Rolling-window resume requires history_scores in the checkpoint')
        learning=state['learning'];rolling=state['rolling'];last_attribution=state['last_attribution']
    else:
        if check.exists():ap.error('Output already contains checkpoint, use --resume or new --output')
        start=0;totals=np.zeros(len(grid));history_scores=[];learning=[];rolling=[];last_attribution=[]
    print(f'US3108 | {G_NAME} (lambda={G_LAMBDA}) | R={R_FIXED} | eval kernel R={R_FIXED} | n={engine.n} | device={device} | '
          f'J={len(grid)} L/M/B/D={args.L}/{args.M}/{args.B}/{args.D} | '
          f'pair-mode={args.pair_mode} | history-window={args.history_window} | '
          f'CRN=yes | report step={args.reported_step}',flush=True)
    tick=time.time();eval_start=T-args.eval_months
    all_fips=idx.fips.tolist()
    for t in range(start,T):
        curr=Y[t];actual=Y[t+1]
        beta=grid[int(totals.argmin())].copy() if t>0 else initial_beta(grid)
        scale=engine.kernel_scale(Y,t)
        if t>=eval_start:
            # Step 1 (time t): forecast a complete Y_(t+1) for both methods.
            movement_maps,baseline_maps=generate_forecasts(engine,curr,beta,args.D,args.seed,t)
            # Step 2 (after observation): score D draws and compare their county-wise mean map to Y_(t+1).
            sm,sb,mv,bb,movement_draw,baseline_draw=eval_forecast(
                engine,curr,actual,scale,movement_maps,baseline_maps)
            # Analytic conditional-mean forecasts remain separate diagnostics.
            point=engine.prediction_mean(curr,beta[0]).cpu().numpy().astype('float64')
            basepoint=panel[t].copy();baseline_point=basepoint+args.baseline_mean;obs=panel[t+1].copy()
            mv_point=point_metrics(point,obs);bb_point=point_metrics(baseline_point,obs)
            actual_dir=observed_counts(basepoint,obs,args.reported_step)
            v=beta[1]**2*engine.diag_KKt
            expected=direction_counts_from_mean(point,basepoint,v,args.reported_step)
            expected_base=direction_counts_from_mean(baseline_point,basepoint,
                                   np.full(engine.n,engine.sigma_obs**2),args.reported_step)
            rolling.append(dict(t=t,origin_month=months[t],target_month=months[t+1],R=R_FIXED,
                 alpha_selected=float(beta[0]),tau_selected=float(beta[1]),
                 baseline_score=sb,movement_score=sm,score_gain=sb-sm,
                 baseline_rmse=bb['rmse'],movement_rmse=mv['rmse'],
                 baseline_mae=bb['mae'],movement_mae=mv['mae'],
                 baseline_bias=bb['mean_error'],movement_bias=mv['mean_error'],
                 baseline_point_rmse=bb_point['rmse'],movement_point_rmse=mv_point['rmse'],
                 baseline_point_mae=bb_point['mae'],movement_point_mae=mv_point['mae'],
                 baseline_point_bias=bb_point['mean_error'],movement_point_bias=mv_point['mean_error'],
                 observed_plus=actual_dir['plus'],observed_minus=actual_dir['minus'],observed_zero=actual_dir['zero'],
                 predicted_plus=expected['plus'],predicted_minus=expected['minus'],predicted_zero=expected['zero'],
                 baseline_plus=expected_base['plus'],baseline_minus=expected_base['minus'],baseline_zero=expected_base['zero']))
            std=np.sqrt(v)
            prob_minus=ndtr((basepoint-args.reported_step/2-point)/std)
            prob_plus=1-ndtr((basepoint+args.reported_step/2-point)/std)
            prob_zero=np.clip(1-prob_plus-prob_minus,0,1)
            pred_rows=[]
            for i,fips in enumerate(all_fips):
                pred_rows.append(dict(fips=fips,origin_month=months[t],target_month=months[t+1],
                    current=float(basepoint[i]),observed=float(obs[i]),
                    movement_prediction=float(movement_draw[i]),baseline_prediction=float(baseline_draw[i]),
                    movement_error=float(movement_draw[i]-obs[i]),
                    baseline_error=float(baseline_draw[i]-obs[i]),
                    movement_prediction_mean=float(point[i]),baseline_prediction_mean=float(baseline_point[i]),
                    movement_mean_error=float(point[i]-obs[i]),
                    movement_lower_95=float(point[i]-1.96*std[i]),
                    movement_upper_95=float(point[i]+1.96*std[i]),
                    p_increase=float(prob_plus[i]),p_decrease=float(prob_minus[i]),p_unchanged=float(prob_zero[i])))
            write_csv(out/'predictions_by_month'/f'{months[t+1]}_county_predictions.csv',pred_rows)
            write_csv(out/'04_last_evaluated_county_map.csv',pred_rows)
        q=engine.update_intensities(curr,actual,scale,beta,args.M,args.B,args.seed,t,delta=args.delta)
        # Updated A_t uses the same selected beta_hat_t, never refitted on Y_(t+1).
        if t==T-1:
            before=q['A_prior'].cpu().numpy();after=q['A_post'].cpu().numpy()
            u_before=q['U_prior'].cpu().numpy();u_after=q['U_post'].cpu().numpy()
            plus=q['pplus'].cpu().numpy();minus=q['pminus'].cpu().numpy()
            last_attribution=[dict(fips=fips,origin_month=months[t],target_month=months[t+1],
                                  A_prior=float(before[i]),A_posterior=float(after[i]),
                                  U_prior=float(u_before[i]),U_posterior=float(u_after[i]),
                                  prob_A_positive=float(plus[i]),prob_A_negative=float(minus[i]))
                              for i,fips in enumerate(all_fips)]
            write_csv(out/'06_last_transition_origin_attribution.csv',last_attribution)
        new_scores=engine.score_candidates(curr,actual,scale,grid,args.L,args.seed,t)
        totals+=new_scores
        if args.history_window:
            history_scores.append(new_scores)
            if len(history_scores)>args.history_window:
                totals-=history_scores.pop(0)
        used_transitions=min(t+1,args.history_window) if args.history_window else t+1
        new_beta=grid[int(totals.argmin())]
        learning.append(dict(t=t,origin_month=months[t],target_month=months[t+1],
            selected_alpha_before=float(beta[0]),selected_tau_before=float(beta[1]),
            selected_alpha_after=float(new_beta[0]),selected_tau_after=float(new_beta[1]),
            learning_transitions=used_transitions,
            A_ESS=q['ess'],mean_match=q['mean_match'],
            score_best=float(totals.min()/used_transitions)))
        state=dict(done_transitions=t+1,total_scores=totals.tolist(),
                   history_scores=[x.tolist() for x in history_scores],learning=learning,
                   rolling=rolling,last_attribution=last_attribution)
        write_json(check,state)
        write_csv(out/'01_learning_by_transition.csv',learning)
        write_csv(out/'02_rolling_prediction_scores.csv',rolling)
        print(f'R{R_FIXED} {t+1:3d}/{T}: {months[t]}->{months[t+1]} | '
              f'beta_hat=({beta[0]:+.3f},{beta[1]:.2f}) | A_ESS={q["ess"]:.1f} | '
              f'elapsed {time.time()-tick:.1f}s',flush=True)
    final_used=min(T,args.history_window) if args.history_window else T
    write_csv(out/'03_candidate_scores.csv',[
         dict(g=g+1,alpha=float(b[0]),tau=float(b[1]),
              score_sum=float(totals[g]),transitions_used=final_used,
              mean_score=float(totals[g]/final_used)) for g,b in enumerate(grid)])
    # Forecast without reading the target month. When max-transitions truncates
    # the panel for a smoke test, the forecast month may exist in the input but is not used.
    final_beta=grid[int(totals.argmin())]
    curr=panel[T];pred=engine.prediction_mean(Y[T],final_beta[0]).cpu().numpy().astype('float64')
    variance=final_beta[1]**2*engine.diag_KKt
    std=np.sqrt(variance)
    upper=(curr+args.reported_step/2-pred)/std
    lower=(curr-args.reported_step/2-pred)/std
    probs_plus=1-ndtr(upper);probs_minus=ndtr(lower);probs_zero=np.clip(1-probs_plus-probs_minus,0,1)
    next_month=str(pd.Period(months[T],freq='M')+1)
    write_csv(out/'05_next_month_forecast.csv',[
        dict(fips=fips,current_month=months[T],forecast_month=next_month,
             last_observation=float(curr[i]),forecast_mean=float(pred[i]),
             forecast_lower_95=float(pred[i]-1.96*std[i]),
             forecast_upper_95=float(pred[i]+1.96*std[i]),
             p_increase=float(probs_plus[i]),p_decrease=float(probs_minus[i]),
             p_unchanged=float(probs_zero[i])) for i,fips in enumerate(all_fips)])
    keys=('baseline_score','movement_score','score_gain','baseline_rmse','movement_rmse',
          'baseline_mae','movement_mae','baseline_bias','movement_bias',
          'baseline_point_rmse','movement_point_rmse','baseline_point_mae','movement_point_mae',
          'baseline_point_bias','movement_point_bias')
    summary={k:float(np.mean([r[k] for r in rolling])) for k in keys}
    summary.update(score_win_months=int(sum(r['movement_score']<r['baseline_score'] for r in rolling)),
        rmse_win_months=int(sum(r['movement_rmse']<r['baseline_rmse'] for r in rolling)),
        mae_win_months=int(sum(r['movement_mae']<r['baseline_mae'] for r in rolling)))
    write_csv(out/'08_directional_change_counts.csv',[
        {k:r[k] for k in ('target_month','observed_plus','observed_minus','observed_zero',
         'predicted_plus','predicted_minus','predicted_zero','baseline_plus','baseline_minus','baseline_zero')}
         for r in rolling])
    write_json(out/'results.json',dict(settings=cfg,summary=summary,
            final_parameter=final_beta.tolist(),forecast_month=next_month,elapsed_seconds=time.time()-tick))
    if not args.no_plots:figures(learning,rolling,out)
    print(f'COMPLETE | R={R_FIXED} | eval kernel R={R_FIXED} | summary={summary} | output={out}',flush=True)


if __name__=='__main__':
    main()
