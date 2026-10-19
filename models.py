"""Fixed frozen-encoder and lexical arms; no hyperparameter search."""
from __future__ import annotations
import collections, csv, hashlib, json, os, platform, re, sys, time, warnings
from pathlib import Path
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
LANGS = ('python','pharo')
LABELS = {'python':['Usage','Parameters','DevelopmentNotes','Expand','Summary'],
          'pharo':['Keyimplementationpoints','Example','Responsibilities','Intent','Keymessages','Collaborators']}
ARMS = ['S','SP','P','SP_matched','T','T_exact','T_coarse','T_nospan','T_permuted']
PAIRS = [('SP','S'),('SP_matched','S'),('SP_matched','SP'),('P','S'),('S','T'),
         ('T_exact','T'),('T_coarse','T'),('T_nospan','T'),('T_exact','T_coarse'),
         ('T_exact','T_nospan'),('T_exact','T_permuted')]
SEED, NBOOT = 20261018, 2000
PARAMS = dict(ngram_range=(1,2),min_df=1,max_features=20000,sublinear_tf=True,
              lowercase=True,token_pattern=r'(?u)\b\w+\b',dtype=np.float64)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def canonical(x):
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),
        default=lambda v:v.item() if isinstance(v,np.generic) else str(v)).encode('utf-8')

def norm(s,mode):
    s=s.lower()
    if mode!='raw': s=re.sub(r'[^a-z0-9,.@#&^%!?\s]',' ' if mode=='space' else '',s)
    return ' '.join(s.split())

def permutations(rows):
    """Derange unique positioned targets; no labels or lengths used to pick map."""
    bygroup=collections.defaultdict(list)
    for i,r in enumerate(rows): bygroup[r['group_id']].append(i)
    donor=list(range(len(rows)))
    eligible=np.zeros(len(rows),dtype=bool)
    detail=[]
    for g,ids in sorted(bygroup.items()):
        keys=collections.defaultdict(list)
        for i in ids:
            r=rows[i]
            keys[(r['normalized_target'],*r['original_span'])].append(i)
        kk=sorted(keys)
        target_spans=collections.defaultdict(set)
        span_targets=collections.defaultdict(set)
        for t,a,b in kk:
            target_spans[t].add((a,b)); span_targets[a,b].add(t)
        ok=len(kk)>=2 and all(len(s)==1 for s in target_spans.values()) and all(len(s)==1 for s in span_targets.values())
        if not ok: continue
        # Hash ordering + one-step cyclic rotation guarantees a derangement.
        ordered=sorted(kk,key=lambda k:hashlib.sha256(canonical([SEED,g,k])).hexdigest())
        for j,k in enumerate(ordered):
            d=keys[ordered[(j+1)%len(ordered)]][0]
            for i in keys[k]: donor[i]=d; eligible[i]=True
        detail.append(dict(group_id=g,rows=len(ids),distinct_targets=len(kk)))
    return dict(donor=donor,eligible=eligible.tolist(),groups=detail,
                eligible_rows=int(eligible.sum()),eligible_groups=len(detail))

def counts(y,p):
    yy=y[:,None,:]
    return np.stack([p*yy,p*(1-yy),(1-p)*yy],axis=-1).astype(np.int64)

def calc(c):
    den=2*c[...,0]+c[...,1]+c[...,2]
    f=np.divide(2*c[...,0],den,out=np.zeros_like(den,dtype=float),where=den!=0)
    total=c.sum(axis=-2); d=2*total[...,0]+total[...,1]+total[...,2]
    mi=np.divide(2*total[...,0],d,out=np.zeros_like(d,dtype=float),where=d!=0)
    return f.mean(axis=-1),mi,f

def boot(c,groups):
    ids=sorted(set(groups)); lookup={g:i for i,g in enumerate(ids)}
    gc=np.zeros((len(ids),)+c.shape[1:],dtype=np.int64)
    for i,g in enumerate(groups): gc[lookup[g]]+=c[i]
    w=np.random.default_rng(SEED).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=NBOOT)
    return calc(np.tensordot(w,gc,axes=(1,0)))

def ci(v): return np.quantile(v,[.025,.975]).tolist()

def guard():
    frozen = read(OUT/'input_hashes.json')
    for n,h in frozen.items():
        assert sha(OLD/n) == h, n

def initialize(work):
    global OUT, OLD
    OUT = work/'models'
    OLD = work/'cohort'

def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    save('input_hashes.json', {n:sha(OLD/n) for n in ('python_cohort.json','pharo_cohort.json','splits.json')})
    for lang in LANGS:
        save(f'{lang}_permutation.json', permutations(read(OLD/f'{lang}_cohort.json')))

def embed(checkpoint, device):
    guard()
    assert not (OUT/'embedding_started.json').exists(),'embedding already attempted'
    save('embedding_started.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    import torch
    import sentence_transformers
    import transformers
    from sentence_transformers import SentenceTransformer
    acquisition=read(Path(__file__).parent/'configs/checkpoint.json')
    for f in acquisition['files']:
        assert sha(checkpoint/f['file']) == f['sha256'], f['file']
    start=time.perf_counter(); cpu=time.process_time()
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    model=SentenceTransformer(str(checkpoint),device=device,local_files_only=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    assert model.max_seq_length==128 and model.get_sentence_embedding_dimension()==384
    rows={l:read(OLD/f'{l}_cohort.json') for l in LANGS}
    unique=list(dict.fromkeys(r['base'] for l in LANGS for r in rows[l]))
    tokens=model.tokenizer(unique,add_special_tokens=True,truncation=False)['input_ids']
    if device == 'cuda': torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.inference_mode():
        z=model.encode(unique,batch_size=32,show_progress_bar=True,convert_to_numpy=True,
                       normalize_embeddings=True,precision='float32')
    if device == 'cuda': torch.cuda.synchronize()
    encoding=time.perf_counter()-t0
    assert z.shape==(len(unique),384) and np.isfinite(z).all()
    lookup={s:i for i,s in enumerate(unique)}
    inputs={}; counts={}
    for l in LANGS:
        ix=[lookup[r['base']] for r in rows[l]]
        np.save(OUT/f'{l}_embeddings.npy',z[ix])
        inputs[l]=hashlib.sha256(canonical([r['base'] for r in rows[l]])).hexdigest()
        lens=[len(tokens[i]) for i in ix]
        counts[l]=dict(rows=len(ix),wordpieces_min=min(lens),wordpieces_max=max(lens),
            wordpieces_mean=float(np.mean(lens)),truncated_rows=sum(n>128 for n in lens),
            truncated_row_ids=[j for j,n in enumerate(lens) if n>128])
    save('embedding_complete.json',dict(status='RESULT',wall_seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu,encoding_seconds=encoding,unique_inputs=len(unique),
        total_rows=sum(len(r) for r in rows.values()),input_sha256=inputs,token_diagnostics=counts,
        gpu=torch.cuda.get_device_name() if device=='cuda' else 'CPU',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,torch=torch.__version__,
        sentence_transformers=sentence_transformers.__version__,transformers=transformers.__version__,
        python=platform.python_version(),numpy=np.__version__,encoder_gradients=False,batch_size=32,
        hashes={f'{l}_embeddings.npy':sha(OUT/f'{l}_embeddings.npy') for l in LANGS},
        acquisition_sha256=sha(Path(__file__).parent/'configs/checkpoint.json')))
    print('EMBEDDED',len(unique),'unique inputs in',encoding,'seconds',flush=True)

def fit():
    guard()
    assert not (OUT/'fit_started.json').exists(),'fit already attempted'
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sklearn
    import scipy
    import joblib
    import psutil
    emb=read(OUT/'embedding_complete.json')
    for n,h in emb['hashes'].items(): assert sha(OUT/n)==h
    save('fit_started.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),embedding_manifest_sha256=sha(OUT/'embedding_complete.json')))
    start=time.perf_counter(); cpu=time.process_time()
    logs=[]; geometry=[]; models={}; split_details=[]
    splits=read(OLD/'splits.json')
    for lang in LANGS:
        rows=read(OLD/f'{lang}_cohort.json'); y=np.array([r['labels'] for r in rows],np.uint8)
        z=np.load(OUT/f'{lang}_embeddings.npy').astype(np.float64)
        assert np.allclose(np.linalg.norm(z,axis=1),1,atol=2e-7)
        f=np.array([r['numeric_features'][4:8] for r in rows]); coarse=np.eye(3)[np.minimum(2,(3*f[:,0]).astype(int))]
        perm=read(OUT/f'{lang}_permutation.json'); pf=f[perm['donor']]
        preds={}
        for proto in ('group','project'):
            a=np.full((len(rows),len(ARMS),y.shape[1]),np.nan)
            preds[proto]=a
        for sp in (s for s in splits if s['language']==lang):
            tr=np.array(sp['train']); ev=np.array(sp['eval'])
            t0=time.perf_counter()
            vec=TfidfVectorizer(**PARAMS); xb=vec.fit_transform([rows[i]['base'] for i in tr]); vb=vec.transform([rows[i]['base'] for i in ev])
            vh=hashlib.sha256(canonical(sorted(vec.vocabulary_.items()))+vec.idf_.tobytes()).hexdigest()
            scalers={}; arrays={}
            for key,raw in (('exact',f),('coarse',coarse),('permuted',pf)):
                sc=StandardScaler(); xx=sc.fit_transform(raw[tr]); vv=sc.transform(raw[ev]); scalers[key]=sc; arrays[key]=(xx,vv)
            xx,vv=arrays['exact']; cx,cv=arrays['coarse']; px,pv=arrays['permuted']
            # Original A_position fitted the shared nine-column scaler and then
            # selected columns 4:8. Preserve that floating-point evaluation path.
            legacy_raw=np.array([r['numeric_features'] for r in rows])
            legacy_scaler=StandardScaler()
            legacy_x=legacy_scaler.fit_transform(legacy_raw[tr])
            legacy_v=legacy_scaler.transform(legacy_raw[ev])
            legacy_x=sparse.csr_matrix(legacy_x[:,[4,5,6,7]])
            legacy_v=sparse.csr_matrix(legacy_v[:,[4,5,6,7]])
            scalers['lexical_exact']=legacy_scaler
            rms=float(np.sqrt(np.mean(np.sum(xx**2,axis=1))))
            assert rms>0
            designs={
                'T':(xb,vb),
                'T_exact':(sparse.hstack([xb,legacy_x],format='csr'),sparse.hstack([vb,legacy_v],format='csr')),
                'S':(sparse.csr_matrix(z[tr]),sparse.csr_matrix(z[ev])),
                'SP':(sparse.csr_matrix(np.hstack([z[tr],xx])),sparse.csr_matrix(np.hstack([z[ev],vv]))),
                'P':(sparse.csr_matrix(xx),sparse.csr_matrix(vv)),
                'SP_matched':(sparse.csr_matrix(np.hstack([z[tr],xx/rms])),sparse.csr_matrix(np.hstack([z[ev],vv/rms]))),
                'T_coarse':(sparse.hstack([xb,cx],format='csr'),sparse.hstack([vb,cv],format='csr')),
                'T_nospan':(sparse.hstack([xb,xx[:,[0,2,3]]],format='csr'),sparse.hstack([vb,vv[:,[0,2,3]]],format='csr')),
                'T_permuted':(sparse.hstack([xb,px],format='csr'),sparse.hstack([vb,pv],format='csr'))}
            shared=time.perf_counter()-t0
            geometry.append(dict(split_id=sp['split_id'],position_scaler_mean=scalers['exact'].mean_.tolist(),
                position_scaler_scale=scalers['exact'].scale_.tolist(),matched_position_divisor=rms,
                semantic_train_mean_squared_norm=float(np.mean(np.sum(z[tr]**2,axis=1))),
                position_train_mean_squared_norm=rms*rms,matched_position_train_mean_squared_norm=float(np.mean(np.sum((xx/rms)**2,axis=1))),
                coarse_train_mean_squared_norm=float(np.mean(np.sum(cx**2,axis=1))),
                nospan_train_mean_squared_norm=float(np.mean(np.sum(xx[:,[0,2,3]]**2,axis=1))),
                vocabulary_idf_sha256=vh,shared_feature_seconds=shared))
            split_details.append(dict(split_id=sp['split_id'],language=lang,protocol=sp['protocol'],split=sp['split'],
                n_train=len(tr),n_eval=len(ev),train_positive=y[tr].sum(axis=0).tolist(),eval_positive=y[ev].sum(axis=0).tolist(),
                train_unsupported=[LABELS[lang][j] for j in range(y.shape[1]) if len(np.unique(y[tr,j]))<2],
                eval_zero_support=[LABELS[lang][j] for j in range(y.shape[1]) if y[ev,j].sum()==0]))
            models[sp['split_id']]=dict(vectorizer=vec,scalers=scalers,matched_divisor=rms,classifiers={})
            for arm,(x,v) in designs.items():
                t0=time.perf_counter(); c0=time.process_time(); p=np.zeros((len(ev),y.shape[1])); mm=[]; ww=[]; fitsec=0.; predsec=0.
                for j in range(y.shape[1]):
                    vals=np.unique(y[tr,j])
                    if len(vals)==1:
                        p[:,j]=vals[0]; mm.append(int(vals[0])); continue
                    m=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',max_iter=1000,tol=.0001,random_state=SEED)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always'); tt=time.perf_counter(); m.fit(x,y[tr,j]); fitsec+=time.perf_counter()-tt
                    ww.extend(str(w.message) for w in caught)
                    tt=time.perf_counter(); p[:,j]=m.predict_proba(v)[:,1]; predsec+=time.perf_counter()-tt; mm.append(m)
                assert np.isfinite(p).all()
                preds[sp['protocol']][ev,ARMS.index(arm)]=p
                models[sp['split_id']]['classifiers'][arm]=mm
                logs.append(dict(split_id=sp['split_id'],language=lang,protocol=sp['protocol'],arm=arm,
                    fit_seconds=fitsec,predict_seconds=predsec,wall_seconds=time.perf_counter()-t0,cpu_seconds=time.process_time()-c0,
                    n_train=len(tr),n_eval=len(ev),features=x.shape[1],warnings=ww,lr_fits=sum(not isinstance(m,int) for m in mm),
                    constants=sum(isinstance(m,int) for m in mm),max_n_iter=max([int(m.n_iter_[0]) for m in mm if not isinstance(m,int)]+[0])))
            print('FITTED',sp['split_id'],'nine fixed arms',flush=True)
        for proto,a in preds.items():
            assert np.isfinite(a).all()
            np.savez_compressed(OUT/f'{lang}_{proto}_predictions.npz',y=y,probability=a,prediction=(a>=.5).astype(np.uint8),
                row_id=np.arange(len(rows)),arms=np.array(ARMS),eligible=np.array(perm['eligible']),
                group=np.array([r['group_id'] for r in rows]),project=np.array([r['project'] for r in rows]))
    joblib.dump(models,OUT/'models.joblib',compress=3)
    save('fit_log.json',logs); save('scaling_geometry.json',geometry); save('split_support.json',split_details)
    proc=psutil.Process(); mem=proc.memory_info()
    save('fit_complete.json',dict(status='RESULT',wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
        python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,
        process_peak_working_set_bytes=getattr(mem,'peak_wset',None),lr_fits=sum(r['lr_fits'] for r in logs),
        constants=sum(r['constants'] for r in logs),warnings=sum(len(r['warnings']) for r in logs),
        entries=len(logs),reused_entries=0,model_sha256=sha(OUT/'models.joblib'),
        predictions={f'{l}_{p}_predictions.npz':sha(OUT/f'{l}_{p}_predictions.npz') for l in LANGS for p in ('group','project')}))

def report():
    guard(); start=time.perf_counter(); metrics=[]; perlabel=[]; contrasts=[]; splitmetrics=[]; sensitivity=[]
    for lang in LANGS:
        rows=read(OLD/f'{lang}_cohort.json'); groups=[r['group_id'] for r in rows]
        components=[r['component_id'] for r in rows]
        for proto in ('group','project'):
            d=np.load(OUT/f'{lang}_{proto}_predictions.npz'); c=counts(d['y'],d['prediction']); ma,mi,f=calc(c.sum(axis=0))
            bm,bi,bf=boot(c,groups); cm,cii,cf=boot(c,components)
            for a,arm in enumerate(ARMS):
                metrics.append(dict(language=lang,protocol=proto,arm=arm,macro_f1=ma[a],micro_f1=mi[a],
                    macro_ci95=ci(bm[:,a]),micro_ci95=ci(bi[:,a])))
                for j,label in enumerate(LABELS[lang]):
                    perlabel.append(dict(language=lang,protocol=proto,arm=arm,label=label,label_index=j,
                        support=int(d['y'][:,j].sum()),f1=f[a,j],ci95=ci(bf[:,a,j])))
            for a,b in PAIRS:
                aa,bb=ARMS.index(a),ARMS.index(b)
                contrasts.append(dict(language=lang,protocol=proto,contrast=f'{a}-{b}',delta_macro=ma[aa]-ma[bb],
                    macro_ci95=ci(bm[:,aa]-bm[:,bb]),delta_micro=mi[aa]-mi[bb],micro_ci95=ci(bi[:,aa]-bi[:,bb]),
                    component_macro_ci95=ci(cm[:,aa]-cm[:,bb]),component_micro_ci95=ci(cii[:,aa]-cii[:,bb]),
                    per_label_delta=(f[aa]-f[bb]).tolist(),per_label_ci95=[ci(bf[:,aa,j]-bf[:,bb,j]) for j in range(f.shape[1])]))
            for sp in (s for s in read(OLD/'splits.json') if s['language']==lang and s['protocol']==proto):
                sm,si,sf=calc(c[sp['eval']].sum(axis=0)); ss=read(OUT/'split_support.json')
                support=next(s for s in ss if s['split_id']==sp['split_id'])
                splitmetrics.append(dict(**support,arms={arm:dict(macro_f1=sm[a],micro_f1=si[a],per_label_f1=sf[a].tolist()) for a,arm in enumerate(ARMS)},
                    contrasts={f'{a}-{b}':dict(delta_macro=sm[ARMS.index(a)]-sm[ARMS.index(b)],delta_micro=si[ARMS.index(a)]-si[ARMS.index(b)]) for a,b in PAIRS}))
            mask=d['eligible'].astype(bool); ec=c[mask]; eg=[g for g,m in zip(groups,mask) if m]
            em,ei,ef=calc(ec.sum(axis=0)); ebm,ebi,ebf=boot(ec,eg)
            aa,bb=ARMS.index('T_exact'),ARMS.index('T_permuted')
            sensitivity.append(dict(language=lang,protocol=proto,rows=int(mask.sum()),groups=len(set(eg)),
                exact_macro=em[aa],permuted_macro=em[bb],delta_macro=em[aa]-em[bb],macro_ci95=ci(ebm[:,aa]-ebm[:,bb]),
                exact_micro=ei[aa],permuted_micro=ei[bb],delta_micro=ei[aa]-ei[bb],micro_ci95=ci(ebi[:,aa]-ebi[:,bb])))
    save('metrics.json',metrics); save('per_label.json',perlabel); save('paired_contrasts.json',contrasts)
    save('fold_project_consistency.json',splitmetrics); save('permutation_sensitivity.json',sensitivity)
    save('report_cost.json',dict(wall_seconds=time.perf_counter()-start,bootstrap_draws=NBOOT,bootstrap_refits=0))
    print('SCORED; artifacts saved',OUT,flush=True)

