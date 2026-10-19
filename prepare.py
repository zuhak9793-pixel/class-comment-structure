"""Reconstructed cohorts and deterministic component-purged splits."""
from __future__ import annotations
import collections, csv, hashlib, json, os, platform, re, sys, time, warnings
from pathlib import Path
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pyarrow.parquet as pq
import recovery as REC
LANGS = ('python', 'pharo')
LABELS = {'python':['Usage','Parameters','DevelopmentNotes','Expand','Summary'],
          'pharo':['Keyimplementationpoints','Example','Responsibilities','Intent','Keymessages','Collaborators']}
SEED = 20261018
def digest(b):
    return hashlib.sha256(b).hexdigest()

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=lambda v: v.item() if isinstance(v, np.generic) else str(v)).encode("utf-8")

def readj(p):
    return json.loads(p.read_text(encoding="utf-8"))

def writej(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def writecsv(p, rows):
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def mapped_norm(text, mode):
    """Reproduce recovery normalization while retaining original character offsets."""
    chars, offsets = [], []
    for i, c in enumerate(text):
        for lc in c.lower():
            if mode != "raw" and not re.match(r"[a-z0-9,.@#&^%!?\s]", lc):
                if mode == "delete":
                    continue
                lc = " "
            chars.append(lc)
            offsets.append(i)
    val, pos = [], []
    for m in re.finditer(r"\S+", "".join(chars)):
        if val:
            val.append(" ")
            pos.append(offsets[m.start()])
        val.extend(chars[m.start():m.end()])
        pos.extend(offsets[m.start():m.end()])
    result = "".join(val)
    assert result == REC.norm(text, mode)
    return result, pos

def tokens(s):
    return s.split()

def clip_context(before, after, budget):
    b, a = tokens(before), tokens(after)
    nb = min(len(b), budget // 2)
    na = min(len(a), budget - nb)
    nb = min(len(b), budget - na)
    return " ".join((b[-nb:] if nb else []) + a[:na])

class Union:
    def __init__(self, ids):
        self.parent = {g: g for g in ids}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)

def support(y):
    return dict(n=len(y), positives=y.sum(axis=0).tolist(), negatives=(len(y)-y.sum(axis=0)).tolist())

def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    summary = readj(SRC / "matching_summary.json")
    stats, all_splits, cohort_hashes = {}, [], {}
    for lang in LANGS:
        # Explicit paths only. No glob, Test download, Test schema, Java input, or Test comparison.
        train = pq.read_table(INPUT / f"{lang}_train.parquet", columns=["class", "comment_sentence", "labels"]).to_pylist()
        matches = readj(SRC / f"{lang}_matches.json")
        sources = readj(SRC / f"{lang}_source_groups.json")
        assert len(train) == len(matches) == summary[lang]["train_rows"]
        with (INPUT/f"{lang}_original.csv").open(encoding="utf-8-sig", newline="") as f:
            originals = list(csv.DictReader(f))
        for g in sources:
            for j in g["source_rows"]:
                orig = originals[j]
                assert (orig["Project"], orig.get("Class") or orig.get("class"), orig["Comment"]) == (g["project"], g["cls"], g["text"])
        rows, excluded = [], collections.Counter()
        for m, t in zip(matches, train):
            assert m["train_row"] == len(rows) + sum(excluded.values())
            assert m["text"] == t["comment_sentence"] and m["cls"] == t["class"]
            assert len(t["labels"]) == len(LABELS[lang]) and set(t["labels"]) <= {0, 1}
            if m["status"] != "unique":
                excluded[m["status"]] += 1
                continue
            assert m["method"] in ("raw", "space", "delete")
            g = sources[m["group_id"]]
            assert g["id"] == m["group_id"] and g["cls"] == t["class"]
            normed, offsets = mapped_norm(g["text"], m["method"])
            span = REC.spans(REC.norm(t["comment_sentence"], m["method"]), normed)
            assert [list(x) for x in span] == m["span"] and len(span) == 1
            s, e = span[0]
            a, b = offsets[s], offsets[e-1]+1
            assert REC.norm(g["text"][a:b], m["method"]) == REC.norm(t["comment_sentence"], m["method"])
            before, after = g["text"][:a], g["text"][b:]
            if not re.search(r"\w", before + " " + after):
                excluded["no_external_word_context"] += 1
                continue
            # Record content, not CSV ordering, defines identity; identical CSV copies collapse.
            gid = digest(canonical([lang, g["project"], g["cls"], g["text"]]))
            base = " ".join(tokens(t["comment_sentence"])[:128] + tokens(t["class"])[:16])
            context_b = clip_context(before, after, 256)
            context_c = " ".join(tokens(before)[-32:] + tokens(after)[:32])
            nt, no = len(tokens(t["comment_sentence"])), len(tokens(g["text"]))
            feature = [np.log1p(nt), np.log1p(no), np.log1p(len(t["comment_sentence"])),
                       np.log1p(len(g["text"])), a/max(1,len(g["text"])),
                       (b-a)/max(1,len(g["text"])), np.log1p(len(tokens(before))),
                       np.log1p(len(tokens(after))), np.log1p(g["text"].count("\n")+1)]
            rows.append(dict(row_id=len(rows), train_row=m["train_row"], language=lang,
                             group_id=gid, source_group_id=g["id"], source_rows=g["source_rows"],
                             project=g["project"], cls=g["cls"], method=m["method"],
                             normalized_span=[s,e], original_span=[a,b], labels=t["labels"],
                             target=t["comment_sentence"], base=base, context_b=context_b,
                             context_c=context_c, source_comment_sha256=digest(g["text"].encode()),
                             normalized_target=REC.norm(t["comment_sentence"],"space"),
                             normalized_source=REC.norm(g["text"],"space"),
                             target_tokens=nt, source_tokens=no, numeric_features=feature,
                             fragment_proxy=nt <= 5 and not bool(re.search(r"[.!?]\s*$", t["comment_sentence"]))))
        bygroup = collections.defaultdict(list)
        bytarget, bysource = collections.defaultdict(list), collections.defaultdict(list)
        for r in rows:
            bygroup[r["group_id"]].append(r)
            bytarget[r["normalized_target"]].append(r)
            bysource[r["normalized_source"]].append(r)
        union = Union(bygroup)
        for rr in list(bytarget.values()) + list(bysource.values()):
            for r in rr[1:]:
                union.join(rr[0]["group_id"], r["group_id"])
        comps = collections.defaultdict(list)
        for r in rows:
            r["component_id"] = union.find(r["group_id"])
            comps[r["component_id"]].append(r)
        # Deterministic sample-count balancing only; no label-based seed/split search.
        loads = [0,0,0]
        comp_fold = {}
        for cid, rr in sorted(comps.items(), key=lambda x: (-len(x[1]), x[0])):
            fold = min(range(3), key=lambda k: (loads[k], k))
            comp_fold[cid] = fold
            loads[fold] += len(rr)
        for r in rows:
            r["fold"] = comp_fold[r["component_id"]]
        duplicates = []
        for text, rr in bytarget.items():
            if len(rr)>1:
                duplicates.append(dict(text_sha256=digest(text.encode()), rows=[r["row_id"] for r in rr],
                                       groups=sorted({r["group_id"] for r in rr}),
                                       label_vectors=sorted({tuple(r["labels"]) for r in rr})))
        overlaps = []
        for gid, rr in bygroup.items():
            for i,r in enumerate(rr):
                a,b = r["original_span"]
                for q in rr[i+1:]:
                    c,d = q["original_span"]
                    shared_target=max(0,min(b,d)-max(a,c))
                    # Raw external-context sets [0,a) U [b,L); their mutual overlap.
                    L = len(sources[r["source_group_id"]]["text"])
                    context_overlap = L-((b-a)+(d-c)-shared_target)
                    overlaps.append(dict(group_id=gid, row_a=r["row_id"],row_b=q["row_id"],
                                         target_overlap_chars=shared_target,
                                         external_context_overlap_chars=context_overlap,
                                         target_a_in_context_b=bool(a<c or b>d),
                                         target_b_in_context_a=bool(c<a or d>b)))
        writej(OUT/f"{lang}_duplicate_targets.json", duplicates)
        writecsv(OUT/f"{lang}_span_overlap.csv", overlaps)
        # Freeze primary group folds and leave-one-project-out sensitivity; purge linked
        # duplicate components from project-training if they appear in evaluation.
        ids = np.arange(len(rows))
        context_lengths = [len(tokens(r["context_b"])) for r in rows]
        for protocol, names in (("group",range(3)), ("project",sorted({r["project"] for r in rows}))):
            for name in names:
                ev = [r["row_id"] for r in rows if (r["fold"]==name if protocol=="group" else r["project"]==name)]
                evcomps = {rows[i]["component_id"] for i in ev}
                evset = set(ev)
                tr = [int(i) for i in ids if i not in evset and rows[i]["component_id"] not in evcomps]
                trset = set(tr)
                purged = [int(i) for i in ids if i not in evset and i not in trset]
                assert not ({rows[i]["group_id"] for i in tr} & {rows[i]["group_id"] for i in ev})
                assert not ({rows[i]["normalized_target"] for i in tr} & {rows[i]["normalized_target"] for i in ev})
                split_id = f"{lang}_{protocol}_{name}"
                ytr, yev = np.array([rows[i]["labels"] for i in tr]), np.array([rows[i]["labels"] for i in ev])
                item = dict(language=lang, protocol=protocol, split=str(name), split_id=split_id,
                            train=tr, eval=ev, purged=purged, train_support=support(ytr), eval_support=support(yev))
                all_splits.append(item)
        writej(OUT/f"{lang}_cohort.json", rows)
        y = np.array([r["labels"] for r in rows])
        stats[lang] = dict(train_population=len(train), targets=len(rows), original_comment_groups=len(bygroup),
                           dependence_components=len(comps), projects=len({r["project"] for r in rows}),
                           excluded=dict(excluded), methods=dict(collections.Counter(r["method"] for r in rows)),
                           project_targets=dict(collections.Counter(r["project"] for r in rows)),
                           project_groups=dict(collections.Counter(rr[0]["project"] for rr in bygroup.values())),
                           labels=LABELS[lang], label_positive_counts=y.sum(axis=0).tolist(),
                           label_positive_groups=[len({r["group_id"] for r in rows if r["labels"][j]}) for j in range(y.shape[1])],
                           label_positive_projects=[len({r["project"] for r in rows if r["labels"][j]}) for j in range(y.shape[1])],
                           cardinality=dict(collections.Counter(map(int,y.sum(axis=1)))),
                           labelsets=dict(collections.Counter("".join(map(str,r["labels"])) for r in rows)),
                           duplicate_normalized_texts=len(duplicates), duplicate_rows=sum(len(d["rows"]) for d in duplicates),
                           duplicate_texts_cross_comment=sum(len(d["groups"])>1 for d in duplicates),
                           conflicting_duplicate_texts=sum(len(d["label_vectors"])>1 for d in duplicates),
                           within_comment_pairs=len(overlaps), overlapping_target_pairs=sum(o["target_overlap_chars"]>0 for o in overlaps),
                           overlapping_context_pairs=sum(o["external_context_overlap_chars"]>0 for o in overlaps),
                           source_duplicate_texts_cross_group=sum(len({r["group_id"] for r in rr})>1 for rr in bysource.values()),
                           target_gt128=sum(r["target_tokens"]>128 for r in rows),
                           full_context_gt256=sum(len(tokens(sources[r["source_group_id"]]["text"][:r["original_span"][0]]))+len(tokens(sources[r["source_group_id"]]["text"][r["original_span"][1]:]))>256 for r in rows))
        cohort_hashes[lang] = digest((OUT/f"{lang}_cohort.json").read_bytes())
    writej(OUT/"cohort_summary.json",stats)
    writej(OUT/"splits.json",all_splits)
