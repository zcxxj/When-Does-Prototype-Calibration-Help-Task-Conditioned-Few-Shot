#!/usr/bin/env python3
"""Build a portable reuse index with explicit original/exported weight identity."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sac.prepare_empirical_reuse import build_index
from sac.report_calibration_study import sha256


def checkpoint_identity(entry, metadata, checkpoint, mode):
    """Preserve original provenance and choose one explicit pairing namespace."""
    exported=sha256(checkpoint)
    original=metadata.get('original_checkpoint_sha256')
    if not original or len(original)!=64:
        raise ValueError('Sanitized legacy metadata must identify its original checkpoint')
    if metadata.get('exported_checkpoint_sha256')!=exported:
        raise ValueError('Exported checkpoint bytes do not match their metadata')
    if mode not in ('source','exported'):raise ValueError('Unknown checkpoint identity mode')
    return dict(entry,checkpoint_sha256=original if mode=='source' else exported,
                original_checkpoint_sha256=original,exported_checkpoint_sha256=exported,
                model_state_sha256=metadata['model_state_sha256'])


def prepare(legacy,output,mode='source'):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.'+uuid.uuid4().hex+'.tmp')
    try:index=build_index(legacy,temporary)
    finally:temporary.unlink(missing_ok=True)
    entries=[]
    for entry in index['entries']:
        metadata=json.loads((output.parent/entry['metadata']).read_text())
        entries.append(checkpoint_identity(entry,metadata,output.parent/entry['checkpoint_path'],mode))
    index=dict(index,entries=entries,checkpoint_identity=mode,
               identity_description='Original training-file hashes pair delivered historical predictions; exported inference-file hashes pair freshly rerun predictions. Both map to identical tensors.')
    text=json.dumps(index,indent=2)+'\n'
    if output.exists() and output.read_text()!=text:raise ValueError('Conflicting reuse index; choose a fresh destination')
    if not output.exists():output.write_text(text)
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,default=ROOT/'attachments/legacy')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-identity',choices=('source','exported'),default='source')
    a=p.parse_args();index=prepare(a.legacy,a.output,a.checkpoint_identity)
    print(json.dumps({'reused':len(index['entries']),'checkpoint_identity':index['checkpoint_identity']}))

if __name__=='__main__':main()
