"""Exact-match recovery; accepts only explicitly named Train input files."""
from __future__ import annotations
import collections, csv, hashlib, json, os, platform, re, sys, time, warnings
from pathlib import Path
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
def norm(s, mode='raw'):
    s = s.lower()
    if mode == 'legacy':
        # Sensitivity check only: documented period masking and observed numeric/
        # abbreviation changes. Do NOT promote these lossy matches to primary.
        s = s.replace('^','.')
        s = re.sub(r'(?<=\d)\.(?=\d)', '', s)
        s = re.sub(r'\be\.g\.', 'eg', s)
        s = re.sub(r'\bi\.e\.', 'ie', s)
        s = re.sub(r'\betc\.', 'etc', s)
    if mode != 'raw':
        s = re.sub(r'[^a-z0-9,.@#&^%!?\s]', ' ' if mode in ('space','legacy') else '', s)
    return ' '.join(s.split())

def spans(needle, haystack):
    # Exact contiguous span with alphanumeric boundaries; no fuzzy similarity.
    if not needle:
        return []
    return [(m.start(),m.end()) for m in re.finditer(r'(?<![a-z0-9])'+re.escape(needle)+r'(?![a-z0-9])', haystack)]

def match():
    import pyarrow.parquet as pq
    stats = {}
    for lang in ('python','pharo'):
        with (INPUT / f'{lang}_original.csv').open(encoding='utf-8-sig',newline='') as f:
            original = [{k:r.get(k,'') for k in ('Project','Class','class','Comment')} for r in csv.DictReader(f)]
        # Repeated identical CSV records cannot be claimed as separate independent comments.
        groups = {}
        for i,r in enumerate(original):
            cls = r['Class'] or r['class']
            key = (r['Project'],cls,r['Comment'])
            if key not in groups:
                groups[key] = dict(id=len(groups),project=r['Project'],cls=cls,text=r['Comment'],source_rows=[])
            groups[key]['source_rows'].append(i)
        gs = list(groups.values())
        byclass = collections.defaultdict(list)
        for g in gs:
            g['normalized'] = {m:norm(g['text'],m) for m in ('raw','space','delete','legacy')}
            byclass[g['cls']].append(g)
        original_class_names = set(byclass)
        train = pq.read_table(INPUT/f'{lang}_train.parquet',columns=['class','comment_sentence']).to_pylist()
        out = []
        for i,t in enumerate(train):
            candidates = byclass.get(t['class'],[])
            matches = {}
            for g in candidates:
                methods = {}
                for mode in ('raw','space','delete'):
                    ss = spans(norm(t['comment_sentence'],mode),g['normalized'][mode])
                    if ss: methods[mode] = ss
                if methods: matches[g['id']] = methods
            row = dict(train_row=i,cls=t['class'],text=t['comment_sentence'],candidates=matches)
            if len(matches)>1:
                row['status']='ambiguous_comment'
            elif not matches:
                row['status']='class_absent' if not candidates else 'text_unmatched'
            else:
                gid,methods = next(iter(matches.items()))
                mode = next(m for m in ('raw','space','delete') if m in methods)
                row.update(group_id=gid,method=mode,span=methods[mode])
                row['status']='unique' if len(methods[mode])==1 else 'ambiguous_position'
            out.append(row)
            if row['status'] in ('text_unmatched','class_absent'):
                diagnostic = {g['id']:spans(norm(t['comment_sentence'],'legacy'),g['normalized']['legacy']) for g in candidates}
                diagnostic = {k:v for k,v in diagnostic.items() if v}
                row['legacy_diagnostic'] = diagnostic
        unique = [r for r in out if r['status']=='unique']
        gids = {r['group_id'] for r in unique}
        counts = collections.Counter(r['group_id'] for r in unique)
        classes = {(gs[g]['project'],gs[g]['cls']) for g in gids}
        # Distinct target spans, not duplicate Train rows, define this subset.
        distinct_counts = collections.defaultdict(set)
        for r in unique:
            distinct_counts[r['group_id']].add((r['method'],tuple(map(tuple,r['span'])),r['text']))
        multicounts = {g:len(ss) for g,ss in distinct_counts.items() if len(ss)>=2}
        canonical_equal = sum(norm(r['text'],r['method'])==gs[r['group_id']]['normalized'][r['method']] for r in unique)
        legacy_unique = [r for r in out if len(r.get('legacy_diagnostic',{}))==1 and len(next(iter(r['legacy_diagnostic'].values())))==1]
        stats[lang] = dict(train_rows=len(train),original_rows=len(original),distinct_source_records=len(gs),
            original_project_classes=len({(g['project'],g['cls']) for g in gs}),
            status_counts=dict(collections.Counter(r['status'] for r in out)),
            unique_methods=dict(collections.Counter(r['method'] for r in unique)),
            recovered_records=len(gids),recovered_project_classes=len(classes),
            records_with_two_or_more_train_targets=len(multicounts),targets_in_those_records=sum(multicounts.values()),
            unique_rows_in_multi_target_records=sum(r['group_id'] in multicounts for r in unique),
            unique_distinct_class_text_targets=len({(r['cls'],r['text']) for r in unique}),
            equal_whole_record_rows=canonical_equal,
            literal_exact_unique_rows=sum(r['text'] in gs[r['group_id']]['text'] for r in unique),
            legacy_additional_unique_rows=len(legacy_unique),
            legacy_unresolved_ambiguity=sum(bool(r.get('legacy_diagnostic')) for r in out)-len(legacy_unique),
            legacy_new_records=len({next(iter(r['legacy_diagnostic'])) for r in legacy_unique}-gids),
            recovered_projects=dict(collections.Counter(gs[g]['project'] for g in gids)),
            train_distinct_classes=len({r['class'] for r in train}),
            class_names_present=sum(t['class'] in original_class_names for t in train),
            duplicate_source_rows=len(original)-len(gs))
        (ROOT/f'{lang}_matches.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
        (ROOT/f'{lang}_source_groups.json').write_text(json.dumps([{k:v for k,v in g.items() if k!='normalized'} for g in gs],indent=2),encoding='utf-8')
    (ROOT/'matching_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

