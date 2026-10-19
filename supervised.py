"""Optional supervised references: exact official-style training schedule."""
from __future__ import annotations
import collections, csv, hashlib, json, os, platform, re, sys, time, warnings
from pathlib import Path
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['WANDB_DISABLED']='true'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
SEED=20261018
ARMS=['SetFit','SetFit_P','P_official_head']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def rows(lang): return read(OLD/f'{lang}_cohort.json')
def splits(): return read(OLD/'splits.json')
def guard():
    for n,h in read(OUT/'inputs.json').items(): assert sha(OLD/n)==h,n
def fit_heads(x,y,v,balanced=False):
 from sklearn.linear_model import LogisticRegression
 models=[]; p=np.zeros((len(v),y.shape[1])); caught=[]
 for j in range(y.shape[1]):
  vals=np.unique(y[:,j])
  if len(vals)==1: m=int(vals[0]); p[:,j]=m
  else:
   m=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',max_iter=1000,tol=1e-4,random_state=SEED) if balanced else LogisticRegression()
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always'); m.fit(x,y[:,j])
   caught += [str(w.message) for w in ww]
   p[:,j]=m.predict_proba(v)[:,1]
  models.append(m)
 return p,models,caught

def train():
 guard()
 import torch,joblib
 from sentence_transformers import SentenceTransformer
 from setfit import SetFitModel,Trainer,TrainingArguments
 from datasets import Dataset,disable_progress_bars
 from sklearn.linear_model import LogisticRegression
 from sklearn.multioutput import MultiOutputClassifier
 from sklearn.preprocessing import StandardScaler
 from transformers import set_seed
 disable_progress_bars(); torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True)
 for sp in splits():
  sid=sp['split_id']; dest=OUT/'fits'/sid
  if (dest/'complete.json').exists():
   done=read(dest/'complete.json')
   for n,h in done['hashes'].items(): assert sha(dest/n)==h
   print('REUSED COMPLETED',sid,flush=True); continue
  assert not (dest/'started.json').exists(),f'Prior incomplete attempt: {sid}; inspect before any retry'
  dest.mkdir(parents=True,exist_ok=True)
  save(dest/'started.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),split_id=sid))
  lang=sp['language']; rr=rows(lang); y=np.array([r['labels'] for r in rr],np.uint8)
  text=read(OUT/f'{lang}_official_inputs.json'); tr=np.array(sp['train']); ev=np.array(sp['eval'])
  set_seed(42); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
  body=SentenceTransformer(str(CHECKPOINT),device='cuda',local_files_only=True)
  model=SetFitModel(model_body=body,model_head=MultiOutputClassifier(LogisticRegression()),multi_target_strategy='multi-output',normalize_embeddings=False)
  args=TrainingArguments(output_dir=str(dest/'training'),num_epochs=10,batch_size=32,num_iterations=20,seed=42,
   report_to='none',save_strategy='no',eval_strategy='no',logging_steps=500,show_progress_bar=False)
  ds=Dataset.from_dict({'text':[text[i] for i in tr],'label':y[tr].tolist()})
  trainer=Trainer(model=model,args=args,train_dataset=ds)
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always'); trainer.train()
  body.eval()
  with torch.inference_mode(): z=body.encode(text,batch_size=32,convert_to_numpy=True,normalize_embeddings=False,show_progress_bar=False)
  # Official head from trainer is reused, not refit.
  p0=np.stack([p[:,1] for p in model.model_head.predict_proba(z[ev])],axis=1)
  direct=np.asarray(model.predict([text[i] for i in ev]))
  assert np.array_equal(direct,p0>=.5)
  f=np.array([r['numeric_features'][4:8] for r in rr]); sc=StandardScaler().fit(f[tr]); fs=sc.transform(f)
  p1,h1,w1=fit_heads(np.hstack([z[tr],fs[tr]]),y[tr],np.hstack([z[ev],fs[ev]]))
  p2,h2,w2=fit_heads(fs[tr],y[tr],fs[ev])
  np.save(dest/'embeddings.npy',z)
  np.savez_compressed(dest/'predictions.npz',row_id=ev,y=y[ev],probability=np.stack([p0,p1,p2],axis=1),arms=np.array(ARMS[:3]))
  body.save_pretrained(str(dest/'encoder'),create_model_card=False)
  joblib.dump(dict(official=model.model_head,position_head=h1,structure_head=h2,scaler=sc,train=tr,eval=ev),dest/'heads.joblib',compress=3)
  save(dest/'training_log.json',trainer.st_trainer.state.log_history)
  save(dest/'complete.json',dict(status='RESULT',split_id=sid,n_train=len(tr),n_eval=len(ev),
   train_row_ids=tr.tolist(),eval_row_ids=ev.tolist(),train_positives=y[tr].sum(0).tolist(),eval_positives=y[ev].sum(0).tolist(),
   steps=trainer.st_trainer.state.global_step,epochs=trainer.st_trainer.state.epoch,pairs=len(trainer.st_trainer.train_dataset),
   pair_dataset_fingerprint=trainer.st_trainer.train_dataset._fingerprint,wall_seconds=time.perf_counter()-start,
   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
   warnings=[str(w.message) for w in ww]+w1+w2,head_params=model.model_head.estimator.get_params(),
   hashes={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))
  print('COMPLETED',sid,'steps',trainer.st_trainer.state.global_step,'seconds',round(time.perf_counter()-start,2),flush=True)
  del trainer,model,body,z; gc.collect(); torch.cuda.empty_cache()
 print('ALL20 SETFIT FITS COMPLETE',flush=True)

def run(work, input_dir, checkpoint):
    global OUT, OLD, CHECKPOINT
    import pyarrow.parquet as pq
    OUT=work/'setfit'; OLD=work/'cohort'; CHECKPOINT=checkpoint
    OUT.mkdir(exist_ok=False)
    pin=read(Path(__file__).parent/'configs/checkpoint.json')
    for f in pin['files']: assert sha(checkpoint/f['file'])==f['sha256'],f['file']
    save(OUT/'inputs.json',{n:sha(OLD/n) for n in ('python_cohort.json','pharo_cohort.json','splits.json')})
    for lang in ('python','pharo'):
        t=pq.read_table(input_dir/f'{lang}_train.parquet',columns=['combo','labels','class','comment_sentence']).to_pylist()
        combos=[]
        for r in rows(lang):
            x=t[r['train_row']]
            assert (x['labels'],x['class'],x['comment_sentence'])==(r['labels'],r['cls'],r['target'])
            combos.append(x['combo'])
        save(OUT/f'{lang}_official_inputs.json',combos)
    train()
