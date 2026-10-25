#!/usr/bin/env python3
"""Repeat the fixed four-setting P0/SAC timing protocol on an idle CUDA device."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=ROOT/'attachments/legacy')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--list',action='store_true')
    a=p.parse_args()
    import numpy as np
    import torch
    from sac.benchmark_review_controls import SETTINGS,EPISODES,benchmark_setting,save_report
    from sac.campaign import run_dir
    from sac.data import Episode
    from sac.evaluate import damage_episode
    from sac.model import EpisodeModel
    from sac.train import base_pools,load_checkpoint,seed_everything
    from sac.report_calibration_study import sha256
    if a.list:print(json.dumps(SETTINGS));return
    if not a.device.startswith('cuda') or not torch.cuda.is_available():p.error('CUDA is required for this timing protocol')
    if a.out.resolve().is_relative_to(a.legacy.resolve()):p.error('Output must be separate from the read-only evidence')
    if a.out.exists() and any(a.out.iterdir()):p.error('Use an empty output directory')
    torch.set_num_threads(4);seed_everything(7);device=torch.device(a.device);torch.cuda.set_device(device)
    properties=torch.cuda.get_device_properties(device)
    uuid='GPU-'+str(properties.uuid).removeprefix('GPU-')
    probe=torch.empty(1,device=device);torch.cuda.synchronize(device)
    def idle():
        r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
        owners=[int(pid) for identifier,pid in csv.reader(r.stdout.splitlines(),skipinitialspace=True) if identifier==uuid]
        if len(owners)!=1:raise RuntimeError('Timing requires an idle selected GPU with only this benchmark context')
        return owners[0]
    owner=idle();del probe
    def check_idle():
        if idle()!=owner:raise RuntimeError('GPU context ownership changed during timing')
    report=dict(complete=False,review1=True,methods=['P0','E2'],settings=[],rows=[],
        hardware=dict(name=properties.name,torch=str(torch.__version__),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),numpy=np.__version__,python=platform.python_version(),
            threads=4,capability=[properties.major,properties.minor]),
        timing_protocol=dict(backend='euclidean.anchor_scores',independent_full_forward=True,
            method_order='alternate every episode pair, including warmups',
            end_to_end_includes=['input_transfer','full_model_forward','one_backend'],
            end_to_end_excludes=['disk_io','corruption_generation','cpu_output_copy'],
            backend_only='identical cached device-resident features',
            memory='absolute peaks; backend cache included; reset per call',
            allocator_cache='cleared once before warmups per mode'))
    pool=base_pools(a.data_root,'ModelNet40',0)['val']
    expected=next(r for r in json.loads((ROOT/'configs/data_sha256.json').read_text())['files']
                  if r['path'].endswith('/Modelnet40_FS_train.h5'))
    if sha256(pool.path)!=expected['sha256']:raise ValueError('Official H5 bytes differ from the frozen benchmark data')
    for shot in (1,5):
        job=dict(dataset='ModelNet40',fold=0,shot=shot,seed=7,family='P0')
        run=run_dir(a.legacy/'full-v1',job);checkpoint=run/'best.pt'
        ep_root=a.legacy/'full-v1/episodes/dev/ModelNet40/fold0'/f'shot{shot}'
        config=json.loads((ep_root/'config.json').read_text())
        if any(config.get(k)!=v for k,v in dict(dataset='ModelNet40',fold=0,shot=shot,pool='dev',count=700,episode_seed=200007,episode_stream=21).items()):
            raise ValueError('Benchmark manifest configuration differs')
        with np.load(ep_root/'episodes.npz') as d:
            episodes=[Episode(**{k:d[k][i].copy() for k in ('rows','class_ids','support_indices','query_indices','support_y','query_y')}) for i in range(EPISODES)]
        with np.load(ep_root/'point_indices.npz') as d:indices=d['points'].copy()
        for ep in episodes:
            if (not np.isin(ep.rows,pool.rows).all() or not np.array_equal(pool.labels[ep.query_rows],ep.class_ids[ep.query_y])
                    or not np.array_equal(pool.labels[ep.support_rows],ep.support_y)):
                raise ValueError('Manifest does not match the official development pool')
        model=EpisodeModel('P0',shot).to(device).eval();state=load_checkpoint(checkpoint,model,restore_rng=False)
        if state['config']!=json.loads((run/'config.json').read_text()):raise ValueError('Checkpoint/config mismatch')
        for selected,side,operator,severity in SETTINGS:
            if selected!=shot:continue
            clouds=[];seeds=[]
            for index,ep in enumerate(episodes):
                damaged,identity=damage_episode(pool.clouds(ep,indices),ep,dataset='ModelNet40',fold=0,shot=shot,pool='dev',episode_id=index,side=side,operator=operator,severity=severity)
                clouds.append(damaged);seeds.append(identity.tolist())
            check_idle();rows=benchmark_setting(model,episodes,clouds,methods=('P0','E2'),review1=True);check_idle()
            setting=dict(shot=shot,side=side,operator=operator,severity=severity)
            report['rows'].extend(dict(**setting,**r) for r in rows)
            report['settings'].append(dict(**setting,dataset='ModelNet40',fold=0,seed=7,pool='dev',
                checkpoint_sha256=sha256(checkpoint),checkpoint_epoch=state['epoch'],manifest_sha256=sha256(ep_root/'episodes.npz'),
                point_indices=indices.tolist(),point_counts=[[len(c) for c in episode] for episode in clouds],
                support_indices=[ep.support_indices.tolist() for ep in episodes],query_indices=[ep.query_indices.tolist() for ep in episodes],corruption_seeds=seeds))
        del model,state;torch.cuda.empty_cache()
    check_idle();report['complete']=True
    print(json.dumps({k:str(v) for k,v in save_report(a.out,report).items()}))

if __name__=='__main__':main()
