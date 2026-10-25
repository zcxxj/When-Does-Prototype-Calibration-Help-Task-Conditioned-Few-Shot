#!/usr/bin/env python3
"""Recompute retained historical PCIA, HPR-D, Scan and DGCNN evidence serially."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sac.review_protocol import evaluation_jobs
from sac.campaign import run_dir


def copy_run(source, destination, job):
    src,dst=run_dir(source,job),run_dir(destination,job)
    dst.mkdir(parents=True,exist_ok=True)
    for name in ('best.pt','last.pt','config.json','completed.json'):
        a,b=src/name,dst/name
        if not a.is_file():raise FileNotFoundError(a)
        if b.exists():
            if hashlib.sha256(a.read_bytes()).digest()!=hashlib.sha256(b.read_bytes()).digest():
                raise ValueError(f'Conflicting output checkpoint/config: {b}')
        else:shutil.copy2(a,b)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=ROOT/'attachments/legacy')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--group',choices=('all','pcia','hyperbolic','scan','dgcnn'),default='all')
    p.add_argument('--dataset',choices=('ModelNet40','ShapeNet70','ScanObjectNN'))
    p.add_argument('--fold',type=int)
    p.add_argument('--shot',type=int,choices=(1,5))
    p.add_argument('--seed',type=int,choices=(7,17,27))
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--list',action='store_true')
    a=p.parse_args();a.out=a.out.resolve();a.legacy=a.legacy.resolve();a.data_root=a.data_root.resolve()
    jobs=[]
    if a.group!='dgcnn':
        for job in evaluation_jobs('test'):
            scan=job['dataset']=='ScanObjectNN'
            allowed={'P0','E2','H'} if scan else {'P0','E2','H','F1','F2','UNIT','QONLY','SOFTK'}
            if a.group=='pcia':allowed={'P0','E2'} if not scan else set()
            elif a.group=='hyperbolic':allowed={'H','F1','F2'} if not scan else set()
            elif a.group=='scan' and not scan:allowed=set()
            methods=[m for m in job['methods'] if m in allowed]
            if methods and (a.group=='all' or job['side'] in ('clean','query')):
                jobs.append(dict(job,methods=methods))
    if a.group in ('all','dgcnn'):
        for dataset in ('ModelNet40','ShapeNet70'):
            for fold in range(5):
                for shot in (1,5):
                    jobs.append(dict(dataset=dataset,fold=fold,shot=shot,seed=7,family='D',methods=['D0','D2'],pool='test'))
    for key in ('dataset','fold','shot','seed'):
        if getattr(a,key) is not None:jobs=[j for j in jobs if j[key]==getattr(a,key)]
    if not jobs:p.error('No jobs match the requested filters')
    if a.list:print(json.dumps(jobs,indent=2));return 0
    common=['--data-root',str(a.data_root),'--device',a.device,'--threads',str(a.threads)]
    for job in jobs:
        if job['family']=='D':
            out=a.out/'dgcnn';source=a.legacy/'review1-evidence-v1/formal'
            copy_run(source,out,job)
            freeze=ROOT/'configs/dgcnn_scope.json';target=out/'freeze.json'
            if target.exists() and target.read_bytes()!=freeze.read_bytes():raise ValueError('Conflicting DGCNN freeze')
            if not target.exists():shutil.copy2(freeze,target)
            cmd=[sys.executable,'-m','sac.evaluate',*common,'--out',str(out),'--pool','test']
        else:
            out=a.out/'historical';out.mkdir(parents=True,exist_ok=True)
            freeze=out/'freeze.json'
            frozen=dict(protocol='retained-historical-test-v1',episodes=700,test_previously_viewed=False)
            if freeze.exists() and json.loads(freeze.read_text())!=frozen:raise ValueError('Conflicting historical freeze')
            if not freeze.exists():freeze.write_text(json.dumps(frozen,indent=2)+'\n')
            if job['seed']!=7:copy_run(a.legacy/'sac-review-v1',out,job)
            cmd=[sys.executable,'-m','sac.review_evaluate',*common,'--out',str(out),
                 '--source-out',str(a.legacy/'full-v1'),'--pool','test']
            for field in ('checkpoint','side','severity'):cmd+=['--'+field,str(job[field])]
            if job['operator'] is not None:cmd+=['--operator',job['operator']]
        for field in ('dataset','fold','shot','seed','family'):cmd+=['--'+field,str(job[field])]
        cmd+=['--methods',*job['methods']]
        subprocess.run(cmd,cwd=ROOT,check=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
