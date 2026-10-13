"""PPO policy for budgeted depth-portfolio allocation."""
import os, json, time, argparse, hashlib, re, itertools
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
try:
    from .dpa_uniform import uniform_counts
except ImportError:  # also support `python dpa/run_dpa_ppo.py`
    from dpa_uniform import uniform_counts

DATA_DIR=Path(os.environ.get('DPA_DATA_DIR', './data'))
ROOT=Path(os.environ.get('DPA_RUN_DIR', './runs'))
DEPTHS=np.array([8,16,24,32],dtype=np.int32)
BUDGETS=[48,80,120,160,200,240,320]
ALLOWED_DEPTH_IDS=(1,2,3)
M_MAX=6; QDIM=256; HIDDEN=128; DEVICE='cuda'

def qhash(s):
    toks=re.findall(r"[a-z]+|\d+|[^\w\s]",(s or '').lower()); grams=['w:'+x for x in toks]+['b:'+a+'_'+b for a,b in zip(toks,toks[1:])]
    v=np.zeros(QDIM,np.float32)
    for g in grams:
        h=int.from_bytes(hashlib.blake2b(g.encode(),digest_size=8).digest(),'little');v[h%QDIM]+=1. if h>>63==0 else -1.
    n=np.linalg.norm(v);return v/(n or 1.)

def selector_pick(S, qi, sel, kind):
    ids=np.flatnonzero(sel)
    if len(ids)==0:return -1
    x=S[:,qi,ids]
    if kind=='borda':
        ranks=np.empty_like(x,dtype=np.float32)
        for k in range(x.shape[0]):
            order=np.argsort(x[k],kind='stable');ranks[k,order]=np.arange(len(ids),dtype=np.float32)
        return int(ids[int(ranks.mean(0).argmax())])
    z=(x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-6)
    return int(ids[int(z.mean(0).argmax())])

def accuracy(S,Y,qi,sel,kind):
    j=selector_pick(S,qi,sel,kind)
    return float(Y[qi,j]) if j>=0 else 0.

def qfeat_for(data): return np.stack([qhash(t) for t in data['texts'].astype(str)])

def state_features(qi,sel,rem,budget,qfeat,S,M, Hdata=None):
    b=len(qi);sf=np.zeros((b,4,M_MAX,2),np.float32)
    hsel=None if Hdata is None else np.zeros((b,20,Hdata.shape[-1]),np.float16)
    counts=sel.reshape(b,4,5).sum(2).astype(np.float32)/5.
    for i in range(b):
        ids=np.flatnonzero(sel[i])
        if not len(ids):continue
        xx=S[:,qi[i],ids].astype(np.float32)
        xx=(xx-xx.mean(1,keepdims=True))/(xx.std(1,keepdims=True)+1e-6)
        for d in range(4):
            local=np.flatnonzero(ids//5==d)
            if not len(local):continue
            vals=xx[:,local]
            sf[i,d,:M,0]=vals.mean(1);sf[i,d,:M,1]=vals.max(1)
        if hsel is not None:hsel[i,ids]=Hdata[qi[i],ids]
    extras=np.column_stack([counts,rem/np.maximum(budget,1),budget/320.]).astype(np.float32)
    return qfeat[qi].astype(np.float32),sf.reshape(b,-1),extras,hsel

class AC(nn.Module):
    def __init__(self,M,use_h=False,hdim=5280):
        super().__init__();self.M=M;self.use_h=use_h
        self.qp=nn.Sequential(nn.Linear(QDIM,96),nn.Tanh())
        self.sp=nn.Sequential(nn.Linear(4*M_MAX*2,96),nn.Tanh())
        self.ep=nn.Sequential(nn.Linear(6,32),nn.Tanh())
        if use_h:
            self.hnorm=nn.LayerNorm(hdim);self.hp=nn.Sequential(nn.Linear(hdim,32),nn.Tanh())
        dim=96+96+32+(128 if use_h else 0)
        self.body=nn.Sequential(nn.Linear(dim,192),nn.Tanh(),nn.Linear(192,128),nn.Tanh())
        self.pi=nn.Linear(128,5);self.v=nn.Linear(128,1)
    def forward(self,q,s,e,h=None):
        x=[self.qp(q),self.sp(s),self.ep(e)]
        if self.use_h:
            ph=self.hp(self.hnorm(h.float()))
            # h is [batch,20,hidden]; pool mean and max within each depth.
            z=[]
            present=(h.abs().sum(-1)>0).float()
            for d in range(4):
                msk=((torch.arange(20,device=h.device)//5==d).view(1,20).float()*present).unsqueeze(-1)
                hd=ph*msk
                z.append(hd.sum(1)/(msk.sum(1).clamp_min(1)))
            x.append(torch.cat(z,dim=1))
        y=self.body(torch.cat(x,dim=1));return self.pi(y),self.v(y).squeeze(-1)

def action_mask(sel,rem):
    cnt=sel.reshape(len(sel),4,5).sum(2)
    m=np.concatenate([(rem[:,None]>=DEPTHS[None,:])&(cnt<5),np.ones((len(sel),1),bool)],1)
    m[:,0]=False  # Exclude d8; this run uses only d16, d24, and d32.
    m[cnt.sum(1)==0,4]=False
    dead=~m[:,:4].any(1);m[dead,4]=True
    return m

def init_env(rng,train_q,nenv):
    qi=rng.choice(train_q,nenv).astype(np.int32);B=rng.choice(np.asarray(BUDGETS),nenv).astype(np.int32)
    return qi,B,B.copy(),np.zeros((nenv,20),bool),np.zeros(nenv,bool),np.zeros(nenv,np.float32)

def train(model,S,Y,qfeat,Hdata,kind,lr,entropy,seed,updates,device):
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed+1000);torch.set_num_threads(4)
    model.to(device);opt=torch.optim.Adam(model.parameters(),lr=lr);train_q=np.arange(len(Y));NENV=24;HORIZON=22;EPOCHS=4;MB=128
    qi,B,rem,sel,done,pot=init_env(rng,train_q,NENV);logs=[];t0=time.time()
    for upd in range(1,updates+1):
        feats=[];masks=[];acts=[];lps=[];vals=[];rews=[];dns=[]
        for t in range(HORIZON):
            qq,ss,ee,hh=state_features(qi,sel,rem,B,qfeat,S,model.M,Hdata)
            tt=[torch.as_tensor(x,device=device) for x in (qq,ss,ee)]
            ht=None if hh is None else torch.as_tensor(hh,device=device)
            with torch.no_grad(): logits,v=model(*tt,ht)
            mask=action_mask(sel,rem);logits=logits.masked_fill(torch.as_tensor(~mask,device=device),-1e9)
            dist=torch.distributions.Categorical(logits=logits);a=dist.sample();lp=dist.log_prob(a)
            aa=a.cpu().numpy();rr=np.zeros(NENV,np.float32);dd=np.zeros(NENV,np.float32)
            feats.append((qq,ss,ee,hh));masks.append(mask.copy());acts.append(aa.copy());lps.append(lp.cpu().numpy());vals.append(v.cpu().numpy())
            for i,ai in enumerate(aa):
                if ai==4:
                    dd[i]=1
                else:
                    pool=np.arange(ai*5,(ai+1)*5);avail=pool[~sel[i,pool]];pick=int(rng.choice(avail));sel[i,pick]=True;rem[i]-=DEPTHS[ai]
                    new=accuracy(S,Y,qi[i],sel[i],kind);rr[i]=new-pot[i];pot[i]=new
                    if not action_mask(sel[i:i+1],rem[i:i+1])[0,:4].any():dd[i]=1
            rews.append(rr);dns.append(dd)
            # Start a fresh training episode immediately after a stop/forced stop.
            for i in np.flatnonzero(dd):
                qi[i]=int(rng.choice(train_q));B[i]=int(rng.choice(BUDGETS));rem[i]=B[i];sel[i]=False;pot[i]=0.
        qq,ss,ee,hh=state_features(qi,sel,rem,B,qfeat,S,model.M,Hdata)
        with torch.no_grad():_,lv=model(torch.as_tensor(qq,device=device),torch.as_tensor(ss,device=device),torch.as_tensor(ee,device=device),None if hh is None else torch.as_tensor(hh,device=device));lv=lv.cpu().numpy()
        R=np.asarray(rews);D=np.asarray(dns);V=np.asarray(vals);A=np.zeros_like(R);gae=np.zeros(NENV,np.float32)
        for t in reversed(range(HORIZON)):
            nv=lv if t==HORIZON-1 else V[t+1];nt=1-D[t];delta=R[t]+nt*nv-V[t];gae=delta+nt*.95*gae;A[t]=gae
        ret=A+V
        flatq=np.concatenate([x[0] for x in feats]);flats=np.concatenate([x[1] for x in feats]);flate=np.concatenate([x[2] for x in feats]);
        flath=None if Hdata is None else np.concatenate([x[3] for x in feats]);flatm=np.concatenate(masks);flata=np.concatenate(acts);flatlp=np.concatenate(lps);flatv=np.concatenate(vals);flatA=A.reshape(-1);flatret=ret.reshape(-1)
        flatA=(flatA-flatA.mean())/(flatA.std()+1e-8);idx=np.arange(len(flatA));stats=[]
        for _ in range(EPOCHS):
            rng.shuffle(idx)
            for st in range(0,len(idx),MB):
                j=idx[st:st+MB];obs=[torch.as_tensor(x[j],device=device) for x in (flatq,flats,flate)];hhb=None if flath is None else torch.as_tensor(flath[j],device=device)
                logits,vn=model(*obs,hhb);mask=torch.as_tensor(flatm[j],device=device);logits=logits.masked_fill(~mask,-1e9);dist=torch.distributions.Categorical(logits=logits);act=torch.as_tensor(flata[j],device=device);newlp=dist.log_prob(act)
                ratio=torch.exp(newlp-torch.as_tensor(flatlp[j],device=device));adv=torch.as_tensor(flatA[j],device=device);pol=torch.minimum(ratio*adv,torch.clamp(ratio,.8,1.2)*adv).mean();vl=F.mse_loss(vn,torch.as_tensor(flatret[j],device=device));ent=dist.entropy().mean();loss=-pol+.5*vl-entropy*ent
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step();stats.append([float(pol.detach()),float(vl.detach()),float(ent.detach())])
        if upd%40==0 or upd==1:
            row={'update':upd,'policy':float(np.mean(stats,0)[0]),'value':float(np.mean(stats,0)[1]),'entropy':float(np.mean(stats,0)[2]),'elapsed_s':round(time.time()-t0,1)};logs.append(row);print(json.dumps(row),flush=True)
    return model,logs

def policy_rollout(model,qids,budget,S,Y,qfeat,Hdata,kind,draws,seed,device,greedy=True):
    qids=np.asarray(qids,dtype=np.int32)
    # Bound h_T observation batches (especially GSM8K test) while preserving
    # per-question Monte Carlo means and deterministic reproducibility.
    if len(qids)>64:
        pieces=[policy_rollout(model,qids[i:i+64],budget,S,Y,qfeat,Hdata,kind,draws,seed+i,device,greedy)
                for i in range(0,len(qids),64)]
        return np.concatenate([x[0] for x in pieces]),np.concatenate([x[1] for x in pieces])
    rng=np.random.default_rng(seed);n=len(qids)*draws;qi=np.repeat(qids,draws).astype(np.int32);B=np.full(n,budget,np.int32);rem=B.copy();sel=np.zeros((n,20),bool);cost=np.zeros(n,np.float32);done=np.zeros(n,bool);result=np.zeros(n,np.float32)
    for _ in range(22):
        ix=np.flatnonzero(~done)
        if not len(ix):break
        qq,ss,ee,hh=state_features(qi[ix],sel[ix],rem[ix],B[ix],qfeat,S,model.M,Hdata)
        with torch.no_grad():logits,_=model(torch.as_tensor(qq,device=device),torch.as_tensor(ss,device=device),torch.as_tensor(ee,device=device),None if hh is None else torch.as_tensor(hh,device=device))
        mask=action_mask(sel[ix],rem[ix]);logits=logits.masked_fill(torch.as_tensor(~mask,device=device),-1e9)
        act=logits.argmax(1).cpu().numpy() if greedy else torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for jj,i in enumerate(ix):
            a=int(act[jj])
            if a==4:result[i]=accuracy(S,Y,qi[i],sel[i],kind);done[i]=True
            else:
                pool=np.arange(a*5,(a+1)*5);avail=pool[~sel[i,pool]];pick=int(rng.choice(avail));sel[i,pick]=True;rem[i]-=DEPTHS[a];cost[i]+=DEPTHS[a]
                if not action_mask(sel[i:i+1],rem[i:i+1])[0,:4].any():result[i]=accuracy(S,Y,qi[i],sel[i],kind);done[i]=True
    for i in np.flatnonzero(~done):result[i]=accuracy(S,Y,qi[i],sel[i],kind)
    return result.reshape(len(qids),draws).mean(1),cost.reshape(len(qids),draws).mean(1)

def uniform_alloc(B):
    # Preserve the four-depth cache layout while disabling d8 as an action.
    return list(uniform_counts(B, depths=(8,16,24,32), max_per_depth=5,
                               active_depths=(16,24,32)))

def config_acc(S,Y,qids,alloc,kind,draws,seed):
    rng=np.random.default_rng(seed);out=np.zeros((len(qids),draws),np.float32)
    for i,q in enumerate(qids):
        for r in range(draws):
            sel=np.zeros(20,bool)
            for d,n in enumerate(alloc):
                if n:sel[d*5+np.sort(rng.choice(5,size=n,replace=False))]=True
            out[i,r]=accuracy(S,Y,q,sel,kind)
    return out.mean(1)

def best_fixed_val(S,Y,qids,B,kind,draws,seed):
    candidates=[]
    for d,c in enumerate(DEPTHS):
        if d not in ALLOWED_DEPTH_IDS: continue
        n=min(5,B//int(c))
        if n:candidates.append((d,n))
    vals=[]
    for d,n in candidates:
        a=[0]*4;a[d]=n;x=config_acc(S,Y,qids,a,kind,draws,seed+B+d)
        vals.append((float(x.mean()),d,n,int(n*DEPTHS[d])))
    return max(vals,key=lambda x:(x[0],-x[3]))

def frontier_auc(rows):
    # Rows are mean (cost, accuracy fraction); integrate the upper envelope to max budget.
    pts=[(0.,0.)]+[(float(x['cost']),float(x['acc'])) for x in rows];pts.sort()
    env=[];best=0.
    for x,y in pts:
        best=max(best,y)
        if not env or x>env[-1][0]:env.append((x,best))
        else:env[-1]=(x,best)
    if env[-1][0]<max(BUDGETS):env.append((float(max(BUDGETS)),env[-1][1]))
    return float(np.trapz([p[1] for p in env],[p[0] for p in env]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['gsm8k','svamp','mbpp','multiarith']);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--updates',type=int,default=160);ap.add_argument('--draws-val',type=int,default=12);ap.add_argument('--draws-test',type=int,default=48);ap.add_argument('--tune',action='store_true');ap.add_argument('--no-test',action='store_true');ap.add_argument('--only-hT',action='store_true');ap.add_argument('--scores-only',action='store_true');ap.add_argument('--zero-qtext',action='store_true');ap.add_argument('--single-lr',type=float,default=2.5e-4);ap.add_argument('--single-entropy',type=float,default=.005);ap.add_argument('--single-seed',type=int,default=31);a=ap.parse_args()
    dev=f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available(): torch.cuda.set_device(a.gpu)
    ds=a.dataset;kind='borda';taskdir=DATA_DIR;out=ROOT/ds;out.mkdir(parents=True,exist_ok=True)
    def load(split):
        z=np.load(taskdir/f'{ds}_{split}.npz',allow_pickle=False);hp=taskdir/f'{ds}_{split}_hT.npy';H=np.load(hp,mmap_mode='r') if hp.exists() else (z['hT'] if 'hT' in z.files else None);return {'qids':z['qids'].astype(str),'texts':z['texts'].astype(str),'Y':z['labels'].astype(np.uint8),'S':z['scores'].astype(np.float32),'H':H}
    tr=load('train');va0=load('validation');te0=load('test');M=tr['S'].shape[0];use_h=(tr['H'] is not None and not a.scores_only)
    # Fit on train, select checkpoints and baselines on validation, evaluate once on test.
    va=va0; te=te0
    qtr=qfeat_for(tr);qva=qfeat_for(va);qte=qfeat_for(te)
    if a.zero_qtext:
        qtr=np.zeros_like(qtr);qva=np.zeros_like(qva);qte=np.zeros_like(qte)
    # Validation-only baseline selection: one fixed depth/count per cap; test reuses it.
    baselines={};fixed_choice={}
    for B in BUDGETS:
        ua=uniform_alloc(B);uval=config_acc(va['S'],va['Y'],np.arange(len(va['Y'])),ua,kind,a.draws_val,10000+B)
        fx=best_fixed_val(va['S'],va['Y'],np.arange(len(va['Y'])),B,kind,a.draws_val,12000)
        baselines[str(B)]={'uniform_alloc':ua,'uniform_cost':int(np.dot(ua,DEPTHS)),'uniform_val':float(uval.mean()),'fixed_depth':int(DEPTHS[fx[1]]),'fixed_n':int(fx[2]),'fixed_cost':fx[3],'fixed_val':fx[0]};fixed_choice[B]=(fx[1],fx[2])
    print('BASELINES',json.dumps(baselines),flush=True)
    trials=[];best=None
    h_opts=[True] if a.only_hT else ([False,True] if tr['H'] is not None else [False])
    configs=[(h,lr,en,seed) for h in h_opts for lr in [1e-4,3e-4] for en in [.001,.01] for seed in [13,29]] if a.tune else [(use_h,a.single_lr,a.single_entropy,a.single_seed)]
    for ci,(trial_h,lr,en,seed) in enumerate(configs):
        trial_H=tr['H'] if trial_h else None; val_H=va['H'] if trial_h else None
        model=AC(M,trial_h,5280 if trial_h else 0);model,log=train(model,tr['S'],tr['Y'],qtr,trial_H,kind,lr,en,seed,a.updates,dev)
        valrows=[]
        for B in BUDGETS:
            p,c=policy_rollout(model,np.arange(len(va['Y'])),B,va['S'],va['Y'],qva,val_H,kind,a.draws_val,seed+300+B,dev)
            valrows.append({'budget':B,'acc':float(p.mean()),'cost':float(c.mean())})
        score=frontier_auc(valrows)
        item={'lr':lr,'entropy':en,'seed':seed,'use_hT':trial_h,'val_frontier_auc':score,'val':valrows,'log':log}
        trials.append(item);torch.save({'state_dict':model.state_dict(),'dataset':ds,'lr':lr,'entropy':en,'seed':seed,'updates':a.updates,'use_hT':trial_h,'selector':kind},out/f'trial_{ci:02d}.pt')
        print('TRIAL',ci,json.dumps({k:item[k] for k in ['lr','entropy','seed','use_hT','val_frontier_auc','val']}),flush=True)
        if best is None or (score,-np.mean([r['cost'] for r in valrows]))>(best[0],-best[1]):best=(score,float(np.mean([r['cost'] for r in valrows])),ci,model,item)
    # Select one checkpoint per budget on validation; test is used only after selection.
    selected_by_budget={};test_eval={};paired_bootstrap={}
    rng=np.random.default_rng(20261011)
    for B in BUDGETS:
        bi=min(range(len(trials)),key=lambda i:(-trials[i]['val'][BUDGETS.index(B)]['acc'],trials[i]['val'][BUDGETS.index(B)]['cost'],i))
        rec=trials[bi]['val'][BUDGETS.index(B)]; blob=torch.load(out/f'trial_{bi:02d}.pt',map_location=dev,weights_only=False)
        model=AC(M,True,5280);model.load_state_dict(blob['state_dict']);model.to(dev);model.eval()
        p,c=policy_rollout(model,np.arange(len(te['Y'])),B,te['S'],te['Y'],qte,te['H'],kind,a.draws_test,30000+B,dev)
        ua=baselines[str(B)]['uniform_alloc'];u=config_acc(te['S'],te['Y'],np.arange(len(te['Y'])),ua,kind,a.draws_test,10000+B)
        d,n=fixed_choice[B];fa=[0]*4;fa[d]=n;f=config_acc(te['S'],te['Y'],np.arange(len(te['Y'])),fa,kind,a.draws_test,12000+B)
        selected_by_budget[str(B)]={'trial_index':bi,'lr':trials[bi]['lr'],'entropy':trials[bi]['entropy'],'seed':trials[bi]['seed'],'validation_accuracy':float(rec['acc']),'validation_cost':float(rec['cost'])}
        test_eval[str(B)]={'ppo':{'accuracy':float(p.mean()),'cost':float(c.mean()),'n_questions':len(p)},'uniform':{'accuracy':float(u.mean()),'cost':int(np.dot(ua,DEPTHS)),'allocation':ua},'best_fixed_selected_on_validation':{'accuracy':float(f.mean()),'cost':int(n*DEPTHS[d]),'depth':int(DEPTHS[d]),'n':int(n)}}
        delta=p-u;ix=rng.integers(0,len(delta),(10000,len(delta)));boot=delta[ix].mean(1)
        paired_bootstrap[str(B)]={'ppo_minus_uniform_pp':100*float(delta.mean()),'ci95_pp':[100*float(np.quantile(boot,.025)),100*float(np.quantile(boot,.975))]}
        print('TEST',B,json.dumps(test_eval[str(B)]),flush=True)
    results={'protocol':{'dataset':ds,'pool':'existing finite candidate pool; d16,d24,d32 actions; up to five candidates per depth; no answer generation','data_split':'PPO fitted on train; checkpoint and baseline selected on validation; test evaluated only after freeze','splits':{'train':len(tr['Y']),'validation':len(va['Y']),'test':len(te['Y'])},'mbpp_valid_questions_only_required':ds=='mbpp','selector':kind,'action':'choose d16, d24, d32, or stop; candidates drawn without replacement within depth','reward':'change in frozen-selector correctness; hard nominal-budget mask','depths':[16,24,32],'repeats_per_allowed_depth':5,'budget_cap':320,'budgets':BUDGETS,'tuning':'8 hT-only hyperparameter/seed trials; checkpoint selected separately per budget using validation accuracy'},'trials':[{k:t[k] for k in ['lr','entropy','seed','use_hT','val_frontier_auc','val']} for t in trials],'selected_by_budget':selected_by_budget,'test':test_eval,'paired_bootstrap_vs_uniform':paired_bootstrap}
    (out/'ppo_tuning_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('SAVED',out/'ppo_tuning_results.json',flush=True)

if __name__=='__main__':main()
