"""Explicit opt-in acquisition of pinned inputs; never uses a dataset split loader."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',choices=['data','checkpoint'])
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    base=Path(__file__).resolve().parent
    dest=a.destination.resolve()
    if dest==base or base in dest.parents: p.error('Destination must be outside the source repository.')
    cfg=json.loads((base/'configs'/('data_sources.json' if a.source=='data' else 'checkpoint.json')).read_text(encoding='utf-8'))
    files=cfg if a.source=='data' else cfg['files']
    for item in files:
        name=item.get('file')
        target=dest/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            data=target.read_bytes()
        else:
            with urllib.request.urlopen(item['url'],timeout=60) as r: data=r.read(item['bytes']+1)
        assert len(data)==item['bytes'],name
        assert hashlib.sha256(data).hexdigest()==item['sha256'],name
        if not target.exists(): target.write_bytes(data)
        print('Verified',name)

if __name__=='__main__': main()
