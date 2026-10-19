"""Optional supervised reports and fixed matched-head control.

Uses the split-specific encoders trained by supervised.py. The matched control
uses the saved verified normalization and block-scaling equations.
"""
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import models as M
from check import compare

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v): p.write_text(json.dumps(v,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n',encoding='utf-8')

def run(work, source=None):
    source=source or work/'setfit'
    out=work/'supervised-report'
    out.mkdir(exist_ok=False)
    splits=read(work/'cohort/splits.json')
    summary=[]; contrasts=[]; controlled=[]; controlled_contrasts=[]
    for lang in M.LANGS:
        rr=read(work/'cohort'/f'{lang}_cohort.json')
        y=np.array([r['labels'] for r in rr],np.uint8)
        groups=[r['group_id'] for r in rr]; comps=[r['component_id'] for r in rr]
        for proto in ('group','project'):
            a=np.full((len(rr),3,y.shape[1]),np.nan)
            b=np.full_like(a,np.nan)
            for sp in (s for s in splits if s['language']==lang and s['protocol']==proto):
                dest=source/'fits'/sp['split_id']; d=np.load(dest/'predictions.npz')
                done=read(dest/'complete.json')
                for name in ('embeddings.npy','predictions.npz','heads.joblib'):
                    assert M.sha(dest/name)==done['hashes'][name],(sp['split_id'],name)
                tr=np.array(sp['train']); ev=np.array(sp['eval'])
                assert np.array_equal(d['row_id'],ev) and np.array_equal(d['y'],y[ev])
                a[ev]=d['probability']
                z=np.load(dest/'embeddings.npy').astype(np.float64)
                lengths=np.linalg.norm(z,axis=1); assert np.all(lengths>0)
                s=z/lengths[:,None]
                raw=np.array([r['numeric_features'][4:8] for r in rr],dtype=float)
                sc=StandardScaler().fit(raw[tr]); p=sc.transform(raw)
                rms_s=np.sqrt(np.mean(np.sum(s[tr]**2,axis=1)))
                rms_p=np.sqrt(np.mean(np.sum(p[tr]**2,axis=1)))
                divisor=float(rms_p/rms_s)
                designs=dict(S=s,P=p,SP=np.hstack([s,p/divisor]))
                assert np.allclose(np.linalg.norm(s,axis=1),1,rtol=0,atol=1e-14)
                assert np.allclose(np.mean(np.sum((p[tr]/divisor)**2,axis=1)),np.mean(np.sum(s[tr]**2,axis=1)),rtol=0,atol=1e-14)
                heads={}
                for ai,(arm,x) in enumerate(designs.items()):
                    heads[arm]=[]
                    for j in range(y.shape[1]):
                        vals=np.unique(y[tr,j])
                        if len(vals)==1: m=int(vals[0]); b[ev,ai,j]=m
                        else:
                            m=LogisticRegression(C=1,solver='liblinear',class_weight='balanced',max_iter=1000,tol=1e-4,random_state=M.SEED,fit_intercept=True,intercept_scaling=1)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always'); m.fit(x[tr],y[tr,j])
                            assert not caught, [str(w.message) for w in caught]
                            b[ev,ai,j]=m.predict_proba(x[ev])[:,1]
                        heads[arm].append(m)
                joblib.dump(heads,out/f"{sp['split_id']}_heads.joblib",compress=3)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.savez_compressed(out/f'{lang}_{proto}.npz',y=y,official=a,controlled=b)
            c=M.counts(y,a>=.5);ma,mi,f=M.calc(c.sum(0));bm,bi,bf=M.boot(c,groups);cm,cmi,cf=M.boot(c,comps)
            for ai,arm in enumerate(('SetFit','SetFit_P','P_official_head')):
                summary.append(dict(language=lang,protocol=proto,arm=arm,rows=len(rr),macro_f1=ma[ai],micro_f1=mi[ai],macro_ci95=M.ci(bm[:,ai]),micro_ci95=M.ci(bi[:,ai]),per_label=[dict(label=label,support=int(y[:,j].sum()),predicted_positive=int((a[:,ai,j]>=.5).sum()),f1=f[ai,j]) for j,label in enumerate(M.LABELS[lang])]))
            contrasts.append(dict(language=lang,protocol=proto,contrast='SetFit_P-SetFit',delta_macro=ma[1]-ma[0],delta_micro=mi[1]-mi[0],macro_ci95=M.ci(bm[:,1]-bm[:,0]),micro_ci95=M.ci(bi[:,1]-bi[:,0]),component_macro_ci95=M.ci(cm[:,1]-cm[:,0]),component_micro_ci95=M.ci(cmi[:,1]-cmi[:,0]),per_label_delta=(f[1]-f[0]).tolist()))
            c=M.counts(y,b>=.5);ma,mi,f=M.calc(c.sum(0));bm,bi,bf=M.boot(c,groups);cm,cmi,cf=M.boot(c,comps)
            for ai,arm in enumerate(('S','P','SP')):
                controlled.append(dict(language=lang,protocol=proto,arm=arm,macro_f1=ma[ai],micro_f1=mi[ai],macro_ci95=M.ci(bm[:,ai]),micro_ci95=M.ci(bi[:,ai]),per_label=[dict(label=label,support=int(y[:,j].sum()),f1=f[ai,j],f1_ci95=M.ci(bf[:,ai,j])) for j,label in enumerate(M.LABELS[lang])]))
            for arm1,arm0 in [('SP','S'),('SP','P'),('P','S')]:
                aa,bb=('S','P','SP').index(arm1),('S','P','SP').index(arm0)
                controlled_contrasts.append(dict(language=lang,protocol=proto,contrast=f'{arm1}-{arm0}',delta_macro=ma[aa]-ma[bb],delta_micro=mi[aa]-mi[bb],macro_ci95=M.ci(bm[:,aa]-bm[:,bb]),micro_ci95=M.ci(bi[:,aa]-bi[:,bb]),component_macro_ci95=M.ci(cm[:,aa]-cm[:,bb]),component_micro_ci95=M.ci(cmi[:,aa]-cmi[:,bb]),per_label=[dict(label=label,delta_f1=f[aa,j]-f[bb,j],f1_ci95=M.ci(bf[:,aa,j]-bf[:,bb,j])) for j,label in enumerate(M.LABELS[lang])]))
    save(out/'metrics.json',summary);save(out/'paired_contrasts.json',contrasts);save(out/'controlled_metrics.json',controlled)
    save(out/'controlled_contrasts.json',controlled_contrasts)
    result=dict(status='RESULT',atol=1e-12,rtol=0,numeric_cells=0,exact_numeric_cells=0,exact_other_cells=0,max_abs_error=0.)
    for name,value in [('setfit_metrics.json',summary),('setfit_contrasts.json',contrasts),('setfit_controlled_metrics.json',controlled),('setfit_controlled_contrasts.json',controlled_contrasts)]:
        compare(value,read(Path(__file__).parent/'expected'/name),name,result)
    save(out/'comparison.json',result)
    print(json.dumps(result,indent=2))
