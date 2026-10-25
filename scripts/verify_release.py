#!/usr/bin/env python3
"""Verify independent release hashes, tensor states, and source isolation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--weights',action='store_true')
    parser.add_argument('--expect-single-commit',action='store_true')
    parser.add_argument('--require-complete',action='store_true',help='Require final matrix, weights and independent-rebuild acceptance')
    args=parser.parse_args(); root=args.root.resolve(); errors=[]
    lines=(root/'SHA256SUMS').read_text().splitlines();hashes={}
    for line in lines:
        expected,name=line.split('  ',1); path=root/name
        actual=sha256(path) if path.is_file() else None
        hashes[name]=actual
        if actual!=expected:errors.append('hash: '+name)
    for row in json.loads((root/'assets_manifest.json').read_text())['files']:
        if hashes.get(row['path'])!=row['sha256']:errors.append('asset manifest hash: '+row['path'])
    for path in root.rglob('*'):
        rel=path.relative_to(root)
        if {'.git','__pycache__','.pytest_cache'}&set(rel.parts) or rel.parts[0] in ('outputs','data'):continue
        if path.is_symlink():errors.append('symlink: '+str(rel))
        if not path.is_file():continue
        if str(rel)!='SHA256SUMS' and str(rel) not in hashes:
            errors.append('file absent from delivery checksum manifest: '+str(rel))
        if any(x.startswith('.') for x in rel.parts[:-1]) or 'node_modules' in rel.parts:
            errors.append('development directory: '+str(rel))
        if path.suffix in ('.py','.json','.jsonl','.md','.csv','.tex','.sh'):
            text=path.read_text(errors='replace')
            if re.search(r'/(?:home/[^/\s]+|workspace|mnt/shared_workspace|opt/review1)(?:/|\b)',text):
                errors.append('internal absolute path: '+str(rel))
        if path.suffix in ('.npz','.pt') and path.stat().st_size>5_000_000 and rel.parts[0]!='attachments':
            errors.append('large binary outside attachments: '+str(rel))
    checked=0
    if args.weights:
        sys.path.insert(0,str(root))
        import torch
        from sac.model import EpisodeModel
        torch.set_num_threads(2)
        for path in sorted((root/'attachments').rglob('*.pt')):
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state.get('format')!='sac-inference-v1' or any(k in state for k in ('optimizer','scheduler','rng')):
                errors.append('checkpoint payload: '+str(path.relative_to(root)));continue
            config=state['config'];model=EpisodeModel(config['family'],config['shot'])
            model.load_state_dict(state['model'],strict=True);checked+=1
    commit_count=None
    acceptance=root/'validation/final_acceptance.json'
    final=json.loads(acceptance.read_text()) if acceptance.is_file() else {}
    requirements=dict(matrix_cells=8760,strict_loaded_weights=224,tensors_equal=True,
                      independent_raw_report=True,legacy_reports_rebuilt=True)
    delivery_complete=all(final.get(k)==v for k,v in requirements.items())
    if (args.require_complete or args.expect_single_commit) and not delivery_complete:
        errors.append('final delivery acceptance is absent or incomplete')
    if args.expect_single_commit:
        proc=subprocess.run(['git','-C',str(root),'rev-list','--count','HEAD'],capture_output=True,text=True)
        commit_count=int(proc.stdout.strip()) if proc.returncode==0 else 0
        if commit_count!=1:errors.append('expected exactly one Git commit')
        tracked=subprocess.run(['git','-C',str(root),'ls-files','attachments'],capture_output=True,text=True,check=True)
        if tracked.stdout.strip():errors.append('attachments must not enter Git history')
    print(json.dumps(dict(complete=not errors,delivery_complete=delivery_complete,checked_hashes=len(lines),strict_loaded_weights=checked,
                          commit_count=commit_count,errors=errors),indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
