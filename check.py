"""Compare regenerated scientific outputs with numeric publication references."""
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent
ATOL = 1e-12

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def compare(a, b, path, result):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a: compare(a[k], b[k], f'{path}/{k}', result)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b)): compare(x, y, f'{path}/{k}', result)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        result['numeric_cells'] += 1
        error = abs(a-b)
        result['max_abs_error'] = max(result['max_abs_error'], error)
        result['exact_numeric_cells'] += int(a == b)
        assert error <= ATOL, (path, a, b, error)
    else:
        assert a == b, (path, a, b)
        result['exact_other_cells'] += 1

def run(work):
    result = dict(status='RESULT', atol=ATOL, rtol=0, numeric_cells=0,
                  exact_numeric_cells=0, exact_other_cells=0, max_abs_error=0.)
    for name,digest in read(BASE/'expected/reconstruction_hashes.json')['canonical_json_sha256'].items():
        canonical=json.dumps(read(work/name),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
        assert hashlib.sha256(canonical).hexdigest()==digest,name
    for name in ('metrics.json','per_label.json','paired_contrasts.json','permutation_sensitivity.json'):
        compare(read(work/'models'/name), read(BASE/'expected'/name), name, result)
    # Fit timings are descriptive measurements, not numerical targets.
    for name, folder in [('recovery.json','recovery'),('cohort_summary.json','cohort')]:
        actual = 'matching_summary.json' if folder == 'recovery' else name
        compare(read(work/folder/actual), read(BASE/'expected'/name), name, result)
    compare(read(work/'models/fold_project_consistency.json'),
            read(BASE/'expected/fold_project_consistency.json'), 'split_metrics', result)
    rows = {l:read(work/'cohort'/f'{l}_cohort.json') for l in ('python','pharo')}
    for sp in read(work/'cohort/splits.json'):
        rr = rows[sp['language']]
        for field in ('group_id','component_id','normalized_source','normalized_target'):
            assert not ({rr[i][field] for i in sp['train']} & {rr[i][field] for i in sp['eval']})
        assert set(sp['train']).isdisjoint(sp['purged'])
    for l in rows:
        for protocol in ('group','project'):
            d=np.load(work/'models'/f'{l}_{protocol}_predictions.npz')
            assert np.array_equal(d['prediction'], d['probability'] >= .5)
            for a,arm in enumerate(d['arms']):
                m=next(x for x in read(work/'models/metrics.json') if (x['language'],x['protocol'],x['arm'])==(l,protocol,arm))
                for avg in ('macro','micro'):
                    assert abs(f1_score(d['y'],d['prediction'][:,a],average=avg,zero_division=0)-m[avg+'_f1'])<=ATOL
    figure2=[]
    metrics=read(work/'models/metrics.json')
    contrasts=read(work/'models/paired_contrasts.json')
    for l in ('python','pharo'):
        for protocol in ('group','project'):
            s=next(m for m in metrics if (m['language'],m['protocol'],m['arm'])==(l,protocol,'S'))
            sp=next(m for m in metrics if (m['language'],m['protocol'],m['arm'])==(l,protocol,'SP'))
            c=next(m for m in contrasts if (m['language'],m['protocol'],m['contrast'])==(l,protocol,'SP-S'))
            figure2.append(dict(language=l,protocol=protocol,baseline_macro_f1=s['macro_f1'],augmented_macro_f1=sp['macro_f1'],delta_macro_f1=c['delta_macro'],paired_ci95=c['macro_ci95']))
    figure3=[]
    for l in ('python','pharo'):
        d=np.load(work/'models'/f'{l}_group_predictions.npz')
        from models import LABELS
        for j,label in enumerate(LABELS[l]):
            y=d['y'][:,j]
            for arm in ('S','P','SP'):
                p=d['prediction'][:,list(d['arms']).index(arm),j]
                tp=int((y*p).sum());fp=int(((1-y)*p).sum());fn=int((y*(1-p)).sum());tn=int(((1-y)*(1-p)).sum())
                figure3.append(dict(language=l,label=label,arm=arm,tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
    for name,value in [('fig02_data.json',figure2),('fig03_data.json',figure3)]:
        compare(value,read(BASE/'expected'/name),name,result)
        (work/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    result['all_numeric_exact'] = result['numeric_cells'] == result['exact_numeric_cells']
    (work/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
