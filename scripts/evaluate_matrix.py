#!/usr/bin/env python3
"""Run the frozen scientific matrix sequentially on a user-selected device.

All paths are supplied explicitly. This entry point has no host, SSH, container,
reservation, or scheduling dependency. It never launches training.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sac.empirical_protocol import evaluation_jobs, write_freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=ROOT/'attachments/legacy')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--protocol',choices=('balanced','imbalanced'))
    p.add_argument('--dataset',choices=('ModelNet40','ShapeNet70'))
    p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--shot',type=int,choices=(1,5))
    p.add_argument('--seed',type=int,choices=(7,17,27))
    p.add_argument('--rerun-reused',action='store_true',help='Also recompute the 600 preserved 5-shot seed-7 cells')
    p.add_argument('--list',action='store_true',help='Print matched jobs without executing')
    a=p.parse_args();a.out=a.out.resolve();a.legacy=a.legacy.resolve();a.data_root=a.data_root.resolve()
    jobs=evaluation_jobs(include_reused=a.rerun_reused)
    for field in ('protocol','dataset','fold','shot','seed'):
        value=getattr(a,field)
        if value is not None:jobs=[j for j in jobs if j[field]==value]
    if not jobs:p.error('No jobs match the requested filters')
    if a.list:
        import json
        print(json.dumps(jobs,indent=2));return 0
    write_freeze(a.out)
    for job in jobs:
        checkpoint_root=a.legacy/('full-v1' if job['seed']==7 else 'sac-review-v1')
        command=[sys.executable,'-m','sac.review_evaluate','--empirical',
                 '--data-root',str(a.data_root),'--out',str(a.out),
                 '--source-out',str(a.legacy/'full-v1'),
                 '--manifest-out',str(a.legacy/'sac-review-v1'),
                 '--checkpoint-out',str(checkpoint_root),
                 '--device',a.device,'--threads',str(a.threads)]
        for field in ('protocol','dataset','fold','shot','seed','family','pool','checkpoint','side','severity'):
            command+=['--'+field,str(job[field])]
        if job['operator'] is not None:command+=['--operator',job['operator']]
        command+=['--methods',*job['methods']]
        subprocess.run(command,cwd=ROOT,check=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
