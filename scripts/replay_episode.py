#!/usr/bin/env python3
"""Replay a released episode against archived predictions using explicit assets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sac.data import Episode,H5Pool
from sac.model import EpisodeModel
from sac.review_evaluate import predict_methods
from sac.evaluate import damage_episode,predict_d_methods
from sac.train import seed_everything
from sac.report_calibration_study import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','manifest','points','h5','prediction'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--metadata',type=Path)
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--rtol',type=float,default=1e-4)
    p.add_argument('--atol',type=float,default=1e-4)
    p.add_argument('--output',type=Path,help='Optional validation receipt')
    a=p.parse_args()
    if not 0<=a.episode<700:p.error('episode must be in 0..699')
    meta=json.loads((a.metadata or a.prediction.with_suffix('.json')).read_text())
    if meta.get('method') not in ('P0','E2','UNIT','QONLY','SOFTK','F1','F2','H','D0','D2'):
        p.error('unsupported inference method')
    torch.set_num_threads(a.threads);seed_everything(7)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=state['config']
    model=EpisodeModel(config['family'],config['shot']).to(a.device).eval()
    model.load_state_dict(state['model'],strict=True)
    with np.load(a.manifest,allow_pickle=False) as saved:
        episode=Episode(**{k:saved[k][a.episode].copy() for k in
                          ('rows','class_ids','support_indices','query_indices','support_y','query_y')})
    with np.load(a.points,allow_pickle=False) as saved:points=saved['points'].copy()
    import h5py
    with h5py.File(a.h5,'r') as handle:
        data=np.asarray(handle['data'],dtype=np.float32);labels=np.asarray(handle['label']).reshape(-1)
    pool=H5Pool(a.h5,data,labels,np.arange(len(labels)))
    clouds=pool.clouds(episode,points)
    job={k:meta[k] for k in ('dataset','fold','shot','seed','family')}
    side,operator,severity=meta['condition']
    damaged,_=damage_episode(clouds,episode,dataset=job['dataset'],fold=job['fold'],shot=job['shot'],
        pool=meta['pool'],episode_id=a.episode,side=side,operator=operator,severity=severity)
    if config['family']=='D':result=predict_d_methods(model,episode,damaged,(meta['method'],))
    else:result,_=predict_methods(model,episode,damaged,(meta['method'],))
    current=result[meta['method']]
    with np.load(a.prediction,allow_pickle=False) as saved:
        same=np.array_equal(current['pred'],saved['pred'][a.episode])
        errors={key:float(np.max(np.abs(current[key]-saved[key][a.episode]))) for key in ('logits_before','logits_after')}
        close=all(np.allclose(current[k],saved[k][a.episode],rtol=a.rtol,atol=a.atol) for k in errors)
    report=dict(episode=a.episode,method=meta['method'],predictions_identical=bool(same),
                logits_close=bool(close),max_absolute_logit_error=errors,device=a.device,
                device_name=torch.cuda.get_device_name(torch.device(a.device)) if a.device.startswith('cuda') else 'CPU',
                dataset=meta['dataset'],fold=meta['fold'],shot=meta['shot'],seed=meta['seed'],condition=meta['condition'],
                rtol=a.rtol,atol=a.atol,torch_version=torch.__version__,
                input_sha256={k:sha256(getattr(a,k)) for k in ('checkpoint','manifest','points','prediction')})
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return not (same and close)

if __name__=='__main__':raise SystemExit(main())
