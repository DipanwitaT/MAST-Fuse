# MAST-Fuse temporal architectural ablation: DR-TCN / BiGRU / BiLSTM / CNN / Mean
from __future__ import annotations
import argparse, random, time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from dataset import create_dataloaders as _create_dataloaders
except ImportError:
    from dataset import get_dataloaders as _get_dataloaders
    _create_dataloaders = None
from model import MASTFuse, TemporalTCN, TemporalGRU
try:
    from losses import StableRegressionLoss
except ImportError:
    StableRegressionLoss = None

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / 'results' / 'temporal_architecture_ablation'
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'
METRIC_DIR = RESULTS_DIR / 'metrics'
for d in (CHECKPOINT_DIR, METRIC_DIR): d.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=32; EPOCHS=150; LEARNING_RATE=1e-4; WEIGHT_DECAY=1e-5; GRADIENT_CLIP=1.0
DROPOUT=0.10; FUSION_DIM=128; NUM_HEADS=4; FF_DIM=256; PATIENCE=20; MIN_DELTA=1e-6; NUM_WORKERS=0
SEEDS=[42,123,456,789,1011]
METHODS=['dr_tcn','bigru','bilstm','cnn','mean']
DISPLAY_NAMES={'dr_tcn':'DR-TCN','bigru':'BiGRU','bilstm':'BiLSTM','cnn':'CNN','mean':'Mean'}

class TemporalLSTM(nn.Module):
    def __init__(self, dim=128, dropout=0.10):
        super().__init__(); h=dim//2
        self.lstm=nn.LSTM(dim,h,num_layers=2,batch_first=True,dropout=dropout,bidirectional=True)
        self.projection=nn.Sequential(nn.Linear(2*h,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(dropout))
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        r=x; x,_=self.lstm(x); return self.norm(self.projection(x)+r)

class TemporalCNN(nn.Module):
    def __init__(self, dim=128, dropout=0.10):
        super().__init__()
        self.conv1=nn.Conv1d(dim,dim,3,padding=1); self.bn1=nn.BatchNorm1d(dim)
        self.conv2=nn.Conv1d(dim,dim,5,padding=2); self.bn2=nn.BatchNorm1d(dim)
        self.conv3=nn.Conv1d(dim,dim,3,padding=1); self.bn3=nn.BatchNorm1d(dim)
        self.act=nn.GELU(); self.drop=nn.Dropout(dropout); self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        r=x; x=x.transpose(1,2)
        for conv,bn in ((self.conv1,self.bn1),(self.conv2,self.bn2),(self.conv3,self.bn3)):
            x=self.drop(self.act(bn(conv(x))))
        return self.norm(x.transpose(1,2)+r)

class TemporalMean(nn.Module):
    def forward(self,x): return x.mean(dim=1,keepdim=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def build_refiner(method):
    method=method.lower()
    if method in ('dr_tcn', 'tcn'):
        # Exact proposed temporal module from model.py:
        # Conv1D(k=3,d=1) -> Conv1D(k=3,d=2) -> Conv1D(k=3,d=4),
        # BN + GELU + dropout after each layer, then one residual skip
        # around the complete stack followed by LayerNorm.
        return TemporalTCN(FUSION_DIM,DROPOUT)
    if method in ('bigru', 'gru'):
        return TemporalGRU(FUSION_DIM,DROPOUT)
    if method in ('bilstm', 'lstm'):
        return TemporalLSTM(FUSION_DIM,DROPOUT)
    if method=='cnn': return TemporalCNN(FUSION_DIM,DROPOUT)
    if method=='mean': return TemporalMean()
    raise ValueError(f'Unknown method: {method}')

def build_model(method):
    method = method.lower()
    canonical = {'tcn':'dr_tcn','gru':'bigru','lstm':'bilstm'}.get(method, method)
    # Instantiate the project's MASTFuse without an extra temporal refiner,
    # then inject exactly one selected temporal module.
    model=MASTFuse(weather_dim=17,spectral_dim=3,dynamic_soil_dim=4,static_soil_dim=28,
                   sequence_length=21,fusion_dim=FUSION_DIM,num_heads=NUM_HEADS,ff_dim=FF_DIM,
                   dropout=DROPOUT,use_weather=True,use_spectral=True,use_dynamic_soil=True,
                   use_static_soil=True,temporal_encoder='none',fusion_type='cross_attention')
    model.temporal_refiner=build_refiner(canonical); model.temporal_encoder=canonical
    return model.to(DEVICE)

def criterion(): return StableRegressionLoss(beta=1.0) if StableRegressionLoss else nn.SmoothL1Loss(beta=1.0)

def move(batch):
    """Normalize both dict-style and tuple-style project DataLoader batches."""
    if isinstance(batch, dict):
        return {k:(v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v)
                for k,v in batch.items()}
    if isinstance(batch, (tuple, list)) and len(batch) >= 5:
        weather, spectral, dynamic_soil, static_soil, target = batch[:5]
        return {
            'weather': weather.to(DEVICE, non_blocking=True),
            'spectral': spectral.to(DEVICE, non_blocking=True),
            'dynamic_soil': dynamic_soil.to(DEVICE, non_blocking=True),
            'static_soil': static_soil.to(DEVICE, non_blocking=True),
            'target': target.to(DEVICE, non_blocking=True),
        }
    raise TypeError(f'Unsupported DataLoader batch type: {type(batch)}')

def create_loaders(seed):
    """Use the project's loader factory without changing the experimental split."""
    if _create_dataloaders is not None:
        try:
            return _create_dataloaders(
                batch_size=BATCH_SIZE, seed=seed,
                use_weather=True, use_spectral=True,
                use_dynamic_soil=True, use_static_soil=True,
                num_workers=NUM_WORKERS,
            )
        except TypeError:
            try:
                return _create_dataloaders(
                    batch_size=BATCH_SIZE, seed=seed,
                    use_weather=True, use_spectral=True,
                    use_dynamic_soil=True, use_static_soil=True,
                )
            except TypeError:
                return _create_dataloaders(batch_size=BATCH_SIZE, seed=seed)
    # Current dataset.py exposes get_dataloaders() with fixed project settings.
    return _get_dataloaders()

def forward(model,batch):
    p=model(weather=batch.get('weather'),spectral=batch.get('spectral'),dynamic_soil=batch.get('dynamic_soil'),static_soil=batch.get('static_soil'))
    p=p.squeeze(-1) if p.ndim>1 else p
    if p.ndim!=1 or not torch.isfinite(p).all(): raise RuntimeError(f'Invalid prediction shape/values: {tuple(p.shape)}')
    return p

def train_epoch(model,loader,opt,loss_fn):
    model.train(); total=0.; n=0
    for batch in loader:
        batch=move(batch); y=batch['target'].view(-1); opt.zero_grad(set_to_none=True)
        p=forward(model,batch); loss=loss_fn(p,y)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP)
        if not torch.isfinite(torch.as_tensor(gn)): raise RuntimeError('Non-finite gradient norm')
        opt.step(); bs=y.size(0); total+=float(loss.detach())*bs; n+=bs
    return total/max(n,1)

@torch.no_grad()
def eval_loss(model,loader,loss_fn):
    model.eval(); total=0.; n=0
    for batch in loader:
        batch=move(batch); y=batch['target'].view(-1); p=forward(model,batch); loss=loss_fn(p,y)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite validation loss')
        bs=y.size(0); total+=float(loss)*bs; n+=bs
    return total/max(n,1)

@torch.no_grad()
def predict(model,loader):
    model.eval(); yp=[]; yt=[]
    for batch in loader:
        batch=move(batch); yp.append(forward(model,batch).cpu().numpy()); yt.append(batch['target'].view(-1).cpu().numpy())
    y_pred=np.concatenate(yp); y_true=np.concatenate(yt)
    if not np.isfinite(y_pred).all() or not np.isfinite(y_true).all(): raise RuntimeError('Non-finite test arrays')
    return y_true,y_pred

def metrics(y,yhat):
    pearson=float(np.corrcoef(y,yhat)[0,1]) if np.std(y)>0 and np.std(yhat)>0 else np.nan
    return {'MAE':float(mean_absolute_error(y,yhat)),'RMSE':float(np.sqrt(mean_squared_error(y,yhat))),
            'R2':float(r2_score(y,yhat)),'Pearson':pearson}

def run_one(method,seed):
    print(f'\n=== {method.upper()} | seed {seed} ==='); set_seed(seed)
    tr,va,te=create_loaders(seed)
    model=build_model(method); nparam=sum(p.numel() for p in model.parameters()); loss_fn=criterion()
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=7,min_lr=1e-7)
    best=float('inf'); best_epoch=0; bad=0; history=[]; ckpt=CHECKPOINT_DIR/f'{method}_seed_{seed}_best.pt'; start=time.time()
    for epoch in range(1,EPOCHS+1):
        tl=train_epoch(model,tr,opt,loss_fn); vl=eval_loss(model,va,loss_fn); sched.step(vl); lr=float(opt.param_groups[0]['lr'])
        history.append({'epoch':epoch,'train_loss':tl,'val_loss':vl,'learning_rate':lr})
        print(f'{method.upper()} seed {seed} | {epoch:03d}/{EPOCHS} | train={tl:.6f} val={vl:.6f} lr={lr:.2e}')
        if vl < best-MIN_DELTA:
            best=vl; best_epoch=epoch; bad=0
            torch.save({'epoch':epoch,'seed':seed,'method':method,'val_loss':vl,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict()},ckpt)
        else: bad+=1
        if bad>=PATIENCE: print(f'Early stopping: best epoch {best_epoch}'); break
    state=torch.load(ckpt,map_location=DEVICE); model.load_state_dict(state['model_state_dict'])
    yt,yp=predict(model,te); m=metrics(yt,yp); vy,vp=predict(model,va); vm=metrics(vy,vp)
    pd.DataFrame(history).to_csv(METRIC_DIR/f'{method}_seed_{seed}_history.csv',index=False)
    pd.DataFrame({'y_true':yt,'y_pred':yp,'residual':yp-yt}).to_csv(METRIC_DIR/f'{method}_seed_{seed}_predictions.csv',index=False)
    out={'method':DISPLAY_NAMES.get(method,method),'method_code':method,'seed':seed,'best_epoch':best_epoch,'best_val_loss':best,'test_MAE':m['MAE'],'test_RMSE':m['RMSE'],'test_R2':m['R2'],'test_Pearson':m['Pearson'],'val_MAE':vm['MAE'],'val_RMSE':vm['RMSE'],'val_R2':vm['R2'],'val_Pearson':vm['Pearson'],'total_parameters':nparam,'training_time_seconds':time.time()-start}
    print(f'RESULT {method.upper()} seed {seed}: MAE={m["MAE"]:.6f}, RMSE={m["RMSE"]:.6f}, R2={m["R2"]:.6f}, Pearson={m["Pearson"]:.6f}')
    return out

def summarize(rows):
    df=pd.DataFrame(rows); df.to_csv(RESULTS_DIR/'temporal_ablation_per_seed.csv',index=False)
    out=[]
    for method in METHODS:
        s=df[df.method_code==method]
        if s.empty: continue
        r={'method':DISPLAY_NAMES.get(method,method),'method_code':method,'n_seeds':len(s),'parameters':int(s.total_parameters.iloc[0])}
        for m in ('test_MAE','test_RMSE','test_R2','test_Pearson'):
            r[m+'_mean']=float(s[m].mean()); r[m+'_std']=float(s[m].std(ddof=1))
        out.append(r)
    sm=pd.DataFrame(out); sm.to_csv(RESULTS_DIR/'temporal_ablation_mean_std.csv',index=False)
    tab=sm[['method','parameters']].copy()
    for m,label in [('test_MAE','MAE'),('test_RMSE','RMSE'),('test_R2','R2'),('test_Pearson','Pearson')]:
        tab[label]=sm[m+'_mean'].map(lambda x:f'{x:.4f}')+' $\\pm$ '+sm[m+'_std'].map(lambda x:f'{x:.4f}')
    tab.columns=['Method','Params','MAE','RMSE','R2','Pearson']; tab.to_latex(RESULTS_DIR/'temporal_ablation_table.tex',index=False,escape=False)
    print('\nSUMMARY\n'+sm.to_string(index=False)); return sm

def smoke():
    set_seed(42); x=torch.randn(2,21,128,device=DEVICE)
    for m in METHODS:
        mod=build_refiner(m).to(DEVICE).eval()
        with torch.no_grad(): y=mod(x)
        assert y.ndim==3 and y.shape[0]==2 and y.shape[2]==128 and torch.isfinite(y).all()
        print(f'{DISPLAY_NAMES.get(m,m):<7} input={tuple(x.shape)} output={tuple(y.shape)} params={sum(p.numel() for p in mod.parameters()):,}')
    model=build_model('dr_tcn').eval(); inp={'weather':torch.randn(2,21,17,device=DEVICE),'spectral':torch.randn(2,21,3,device=DEVICE),'dynamic_soil':torch.randn(2,21,4,device=DEVICE),'static_soil':torch.randn(2,21,28,device=DEVICE)}
    with torch.no_grad(): p=model(**inp)
    assert p.shape==(2,) and torch.isfinite(p).all(); print('Full MAST-Fuse smoke test:',tuple(p.shape),'PASS')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=METHODS); ap.add_argument('--all',action='store_true'); ap.add_argument('--smoke_test',action='store_true'); a=ap.parse_args()
    if a.smoke_test: smoke(); return
    methods=[a.method] if a.method else METHODS
    rows=[]
    for method in methods:
        for seed in SEEDS:
            rows.append(run_one(method,seed)); pd.DataFrame(rows).to_csv(RESULTS_DIR/'temporal_ablation_progress.csv',index=False)
    summarize(rows)
if __name__=='__main__': main()
