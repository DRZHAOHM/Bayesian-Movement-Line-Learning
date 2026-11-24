"""Bayesian Movement-Line Learning | GPU-capable, fixed spatial order.

One-step workflow: select beta_hat_t using completed-transition proper kernel
scores, forecast Y_(t+1), then update A_t conditional on fixed beta_hat_t.
No beta posterior, no use of future observations in parameter selection.

Six change channels per origin and distance shell:
 M+, M-, M0 (fractions); V+, V- (mean change magnitudes);
 V0 (mean absolute starting value for the unchanged group).
The optional feature_resolution is only used to compare maps in the kernel;
forecasting the continuous Gaussian model itself is not rounded.
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
COUNTY_ALPHA=(-0.02,-0.01,0.,0.01,0.02)
COUNTY_TAU=(0.15,0.35,0.65)
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
    def __init__(self,shells,R,device,sigma_obs=.5,epsilon=.5,s_star=.5,
                 pair_mode='paired',gpu_block=8,dtype='float32',feature_resolution=0.):
        if sigma_obs<=0 or epsilon<=0 or s_star<=0 or gpu_block<1:raise ValueError('Scale parameters must be positive')
        if pair_mode not in ('paired','full'):raise ValueError('Invalid pair mode')
        if feature_resolution<0:raise ValueError('feature-resolution must be >=0')
        self.R=R;self.n=shells[0].shape[0];self.device=device
        self.dtype=torch.float32 if dtype=='float32' else torch.float64
        self.sigma_obs=float(sigma_obs);self.epsilon=float(epsilon);self.s_star=float(s_star)
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

    def make_maps(self,current,alpha,tau,draws,generator):
        z=float(alpha)*current[None,:]+float(tau)*torch.randn((draws,self.n),device=self.device,dtype=self.dtype,generator=generator)
        u=self.multiply_K(z)
        eps=self.sigma_obs*torch.randn((draws,self.n),device=self.device,dtype=self.dtype,generator=generator)
        return current[None,:]+u+eps,z,u

    def baseline_maps(self,current,draws,generator):
        return current[None,:]+self.sigma_obs*torch.randn((draws,self.n),device=self.device,dtype=self.dtype,generator=generator)

    def prediction_mean(self,current,alpha):
        return current+float(alpha)*self.multiply_K(current)

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
        results=[]
        for j,(a,tt) in enumerate(grid):
            maps,_,_=self.make_maps(current,a,tt,L,rng_for(self.device,seed,t,j,11))
            results.append(self.kernel_score(self.flat_features(current,maps,scale),obs))
        return np.asarray(results,dtype='float64')

    @torch.no_grad()
    def update_intensities(self,current,observed,scale,beta,M,B,seed,t,delta=1.):
        a,tt=map(float,beta)
        rng=rng_for(self.device,seed,t,22)
        A=a*current[None,:]+tt*torch.randn((M,self.n),device=self.device,dtype=self.dtype,generator=rng)
        U=self.multiply_K(A)
        obs=self.flat_features(current,observed[None,:],scale)[0]
        weights=[]
        # One origin-intensity draw is shared across exactly B conditional maps.
        chunk=max(1,min(M,256//B))
        for s in range(0,M,chunk):
            u=U[s:s+chunk]
            eps=self.sigma_obs*torch.randn((len(u),B,self.n),device=self.device,dtype=self.dtype,generator=rng)
            maps=current[None,None,:]+u[:,None,:]+eps
            sim=self.flat_features(current,maps.reshape(-1,self.n),scale)
            weights.append(self.k_to_obs(sim,obs).reshape(len(u),B).mean(1))
        avg=torch.cat(weights)
        w=avg/avg.sum()
        if not torch.isfinite(w).all():raise FloatingPointError('Nonfinite kernel posterior weights')
        post=w@A
        prior=a*current
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
# R4 simulation | fixed 10x10 rook lattice, 11 maps, nine true parameter pairs.
# ============================================================================
R_FIXED=4


def rook_W(rows,cols):
    r=[];c=[]
    for i in range(rows):
        for j in range(cols):
            p=i*cols+j
            for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
                x,y=i+di,j+dj
                if 0<=x<rows and 0<=y<cols:r.append(p);c.append(x*cols+y)
    return sparse.csr_matrix((np.ones(len(r)),(r,c)),shape=(rows*cols,rows*cols))


def initial_map(rows,cols,device,dtype):
    r,c=torch.meshgrid(torch.arange(rows,device=device,dtype=dtype),
                       torch.arange(cols,device=device,dtype=dtype),indexing='ij')
    rr=(rows-1)/2;cc=(cols-1)/2
    return (4+.14*r+.10*c+2.2*torch.exp(-((r-rr)**2/rows**2+(c-cc)**2/cols**2)*9.)+
            .45*torch.sin(.75*r)*torch.cos(.58*c)).flatten()


@torch.no_grad()
def make_trajectory(engine,alpha_true,tau_true,Y0,seed):
    a_rng=rng_for(engine.device,seed,1);e_rng=rng_for(engine.device,seed,2)
    Ys=[Y0];As=[];Us=[]
    for _ in range(10):
        a=alpha_true*Ys[-1]+tau_true*torch.randn((engine.n,),device=engine.device,dtype=engine.dtype,generator=a_rng)
        u=engine.multiply_K(a)
        eps=engine.sigma_obs*torch.randn((engine.n,),device=engine.device,dtype=engine.dtype,generator=e_rng)
        Ys.append(Ys[-1]+u+eps);As.append(a);Us.append(u)
    return torch.stack(Ys),torch.stack(As),torch.stack(Us)


@torch.no_grad()
def one_rep(engine,grid,args,sid,rep,a_true,tau_true):
    base_seed=int(np.random.SeedSequence([args.seed,sid,rep]).generate_state(1)[0])
    Y,A_true,U_true=make_trajectory(engine,a_true,tau_true,
             initial_map(args.rows,args.cols,engine.device,engine.dtype),base_seed)
    total=np.zeros(len(grid));records=[];recovery=[]
    for t in range(9):
        beta=grid[int(total.argmin())] if t>0 else initial_beta(grid)
        scale=engine.kernel_scale(Y,t)
        # Score selection never sees Y_{t+1} until after the forecast parameters were fixed.
        q=engine.update_intensities(Y[t],Y[t+1],scale,beta,args.M,args.B,base_seed,t)
        a0=np.sqrt(float(torch.mean((q['A_prior']-A_true[t])**2)))
        a1=np.sqrt(float(torch.mean((q['A_post']-A_true[t])**2)))
        u0=np.sqrt(float(torch.mean((q['U_prior']-U_true[t])**2)))
        u1=np.sqrt(float(torch.mean((q['U_post']-U_true[t])**2)))
        records.append(dict(scenario=sid,replicate=rep,t=t,alpha_selected_before=float(beta[0]),
                            tau_selected_before=float(beta[1]),A_prior_RMSE=a0,A_posterior_RMSE=a1,
                            U_prior_RMSE=u0,U_posterior_RMSE=u1,A_ESS=q['ess']))
        if t>=1:recovery.append((a0,a1,u0,u1))
        total+=engine.score_candidates(Y[t],Y[t+1],scale,grid,args.L,base_seed,t)
    beta=grid[int(total.argmin())]
    scale=engine.kernel_scale(Y,9)
    maps,_,_=engine.make_maps(Y[9],beta[0],beta[1],args.D,rng_for(engine.device,base_seed,9,31))
    base=engine.baseline_maps(Y[9],args.D,rng_for(engine.device,base_seed,9,32))
    sm=engine.score_forecast(Y[9],Y[10],scale,maps)
    sb=engine.score_forecast(Y[9],Y[10],scale,base)
    # Exact conditional expectation for the Gaussian additive model (not MC mean).
    predicted=engine.prediction_mean(Y[9],beta[0]).cpu().numpy()
    truth=Y[10].cpu().numpy();curr=Y[9].cpu().numpy()
    me=point_metrics(predicted,truth);mb=point_metrics(curr,truth)
    means=np.mean(np.asarray(recovery),axis=0)
    row=dict(scenario=sid,replicate=rep,alpha_true=a_true,tau_true=tau_true,
        selected_alpha=float(beta[0]),selected_tau=float(beta[1]),
        beta_recovered=int(np.isclose(beta[0],a_true) and np.isclose(beta[1],tau_true)),
        movement_score=sm,baseline_score=sb,score_gain=sb-sm,
        movement_rmse=me['rmse'],baseline_rmse=mb['rmse'],
        movement_mae=me['mae'],baseline_mae=mb['mae'],
        movement_mean_error=me['mean_error'],baseline_mean_error=mb['mean_error'],
        A_prior_rmse=float(means[0]),A_posterior_rmse=float(means[1]),
        U_prior_rmse=float(means[2]),U_posterior_rmse=float(means[3]),
        A_recovery_pct=float(100*(means[0]-means[1])/max(means[0],1e-12)),
        U_recovery_pct=float(100*(means[2]-means[3])/max(means[2],1e-12)))
    return row,records


def plot_results(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'figures';dest.mkdir(exist_ok=True)
    x=sorted(set(r['scenario'] for r in rows))
    for columns,ylabel,title,filename in [
        (('baseline_score','movement_score'),'Held-out joint kernel score','R4 predictive distribution scores','01_predictive_score.png'),
        (('baseline_rmse','movement_rmse'),'Point forecast RMSE','R4 Y10 forecast RMSE','02_point_rmse.png'),
        (('A_recovery_pct','U_recovery_pct'),'Recovery improvement (%)','R4 A and U after Bayesian update','03_latent_recovery.png')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for key in columns:
            yy=[np.mean([r[key] for r in rows if r['scenario']==sid]) for sid in x]
            ax.plot(x,yy,marker='o',label=key)
        ax.set(xlabel='Scenario',ylabel=ylabel,title=title)
        ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(dest/filename,dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description='R=4 GPU simulation, 6-feature kernel, Bayesian A-only updating')
    ap.add_argument('--rows',type=int,default=10);ap.add_argument('--cols',type=int,default=10)
    ap.add_argument('--scenario',type=int,choices=range(1,10),default=None)
    ap.add_argument('--replicates',type=int,default=50)
    ap.add_argument('--L',type=int,default=128);ap.add_argument('--M',type=int,default=128)
    ap.add_argument('--B',type=int,default=4);ap.add_argument('--D',type=int,default=256)
    ap.add_argument('--sigma-obs',type=float,default=.5);ap.add_argument('--epsilon',type=float,default=.5)
    ap.add_argument('--s-star',type=float,default=.5)
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--dtype',choices=('float32','float64'),default='float32')
    ap.add_argument('--pair-mode',choices=('paired','full'),default='paired')
    ap.add_argument('--gpu-block',type=int,default=8)
    ap.add_argument('--feature-resolution',type=float,default=0.,
                    help='0 means continuous simulation features, no rounding')
    ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--output',type=Path,default=Path('R4_Simulation_GPU'))
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--no-plots',action='store_true')
    args=ap.parse_args()
    if args.replicates<1 or args.L<2 or args.D<2 or args.M<1 or args.B<1:ap.error('Need replicates/M/B>=1, L/D>=2')
    device=choose_device(args.device)
    engine=MLEngine(make_shells_from_W(rook_W(args.rows,args.cols),R_FIXED),R_FIXED,device,
        sigma_obs=args.sigma_obs,epsilon=args.epsilon,s_star=args.s_star,
        pair_mode=args.pair_mode,gpu_block=args.gpu_block,dtype=args.dtype,
        feature_resolution=args.feature_resolution)
    grid=np.asarray([(a,tau) for a in GRID_ALPHA for tau in GRID_TAU],dtype=float)
    scenarios=[(a,tau) for a in SIM_ALPHA for tau in SIM_TAU]
    selected=[args.scenario] if args.scenario else list(range(1,10))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=dict(R=4,n=engine.n,scenario_replicates=args.replicates,scenarios=selected,
         sim_alpha=SIM_ALPHA,sim_tau=SIM_TAU,grid_alpha=GRID_ALPHA,grid_tau=GRID_TAU,
         sigma_obs=args.sigma_obs,epsilon=args.epsilon,s_star=args.s_star,
         L=args.L,M=args.M,B=args.B,D=args.D,device=str(device),dtype=args.dtype,
         pair_mode=args.pair_mode,seed=args.seed,features=FEATURES,feature_resolution=args.feature_resolution,
         k_weights=engine.ks.tolist(),fixed_data_generator='Y_(t+1)=Y_t+K A_t+normal noise, 10x10 11 maps')
    settings=out/'00_settings.json';check=out/'00_checkpoint.json'
    if settings.exists():
        if json.loads(settings.read_text(encoding='utf8'))!=json.loads(json.dumps(cfg)):
            ap.error('Existing output settings differ; choose a new --output')
    else:write_json(settings,cfg)
    if args.resume:
        if not check.is_file():ap.error('--resume requires an existing checkpoint')
        st=json.loads(check.read_text(encoding='utf8'))
        rows=st['rows'];track=st['track']
    else:
        if check.exists():ap.error('Existing checkpoint; use --resume or a new --output')
        rows=[];track=[]
    completed={(int(r['scenario']),int(r['replicate'])) for r in rows}
    tick=time.time()
    print(f'R4 SIM | device={device} | J={len(grid)} | 9 scenarios x {args.replicates} | L/M/B/D={args.L}/{args.M}/{args.B}/{args.D}',flush=True)
    for sid in selected:
        alpha,tau=scenarios[sid-1]
        for rep in range(1,args.replicates+1):
            if (sid,rep) in completed:continue
            row,transition=one_rep(engine,grid,args,sid,rep,alpha,tau)
            rows.append(row);track.extend(transition)
            write_json(check,dict(rows=rows,track=track,completed_replications=len(rows)))
            print(f'R4 scenario {sid}/9, replicate {rep}/{args.replicates} | gain={row["score_gain"]:+.5f} | RMSE={row["movement_rmse"]:.4f} | elapsed={time.time()-tick:.1f}s',flush=True)
        write_csv(out/'01_replications.csv',rows)
        write_csv(out/'03_learning_by_transition.csv',track)
    by=[]
    for sid in selected:
        rr=[x for x in rows if x['scenario']==sid]
        rec={'scenario':sid,'alpha_true':rr[0]['alpha_true'],'tau_true':rr[0]['tau_true'],'replicates':len(rr)}
        for key in ('baseline_score','movement_score','score_gain','beta_recovered',
                    'baseline_rmse','movement_rmse','baseline_mae','movement_mae',
                    'A_recovery_pct','U_recovery_pct'):
            rec[key]=float(np.mean([x[key] for x in rr]))
        by.append(rec)
    write_csv(out/'02_by_scenario.csv',by)
    result={key:float(np.mean([r[key] for r in rows])) for key in
            ('baseline_score','movement_score','score_gain','beta_recovered',
             'baseline_rmse','movement_rmse','baseline_mae','movement_mae',
             'A_recovery_pct','U_recovery_pct')}
    write_json(out/'results.json',dict(settings=cfg,overall=result,by_scenario=by,elapsed_seconds=time.time()-tick))
    if not args.no_plots:plot_results(rows,out)
    print('COMPLETE R4 SIM |',result,'| output:',out,flush=True)


if __name__=='__main__':
    main()
