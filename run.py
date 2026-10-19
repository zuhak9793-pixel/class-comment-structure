"""Publication reproduction entry point. All generated files belong in work-dir."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[name] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
BASE = Path(__file__).resolve().parent

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def verify_inputs(folder):
    for item in read(BASE/'configs/data_sources.json'):
        p = folder/item['file']
        assert p.stat().st_size == item['bytes'], item['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['file']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'embed', 'fit', 'report', 'check', 'setfit', 'setfit-report'])
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--input-dir', type=Path)
    p.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args()
    work = a.work_dir.resolve()
    if work == BASE or BASE in work.parents:
        p.error('Place generated research data outside the source repository.')
    if a.input_dir and (work == a.input_dir.resolve() or work in a.input_dir.resolve().parents):
        p.error('The work directory must not contain or equal the input directory.')
    if a.stage in ('prepare', 'setfit'):
        if not a.input_dir: p.error('--input-dir is required')
        verify_inputs(a.input_dir)
    if a.stage in ('embed', 'setfit') and not a.checkpoint_dir:
        p.error('--checkpoint-dir is required; supply the pinned local snapshot')
    if a.stage == 'prepare':
        import recovery
        import prepare
        work.mkdir(parents=True, exist_ok=True)
        recovery.ROOT = work/'recovery'
        recovery.ROOT.mkdir(exist_ok=False)
        recovery.INPUT = a.input_dir
        recovery.match()
        prepare.INPUT = a.input_dir
        prepare.SRC = recovery.ROOT
        prepare.OUT = work/'cohort'
        prepare.prepare()
        import models
        models.initialize(work)
        models.freeze()
        print('Prepared exact Train cohorts and frozen split membership.')
    elif a.stage == 'setfit':
        import supervised
        supervised.run(work, a.input_dir, a.checkpoint_dir)
    elif a.stage == 'setfit-report':
        import optional_reports
        optional_reports.run(work)
    elif a.stage == 'check':
        import check
        check.run(work)
    else:
        import models
        models.initialize(work)
        if a.stage == 'embed': models.embed(a.checkpoint_dir, a.device)
        elif a.stage == 'fit': models.fit()
        else: models.report()

if __name__ == '__main__': main()
