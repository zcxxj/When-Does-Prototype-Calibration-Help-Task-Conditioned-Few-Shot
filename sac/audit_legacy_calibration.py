"""Recompute historical primary tables and paired intervals from raw predictions."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .report_calibration_study import OPERATORS, bootstrap, digest, sha256, write_csv


def audit(legacy, archived_report, output):
    legacy, archived_report, output = map(Path,(legacy,archived_report,output))
    output.mkdir(parents=True,exist_ok=True)
    conditions=[('clean',None,0)]+[('query',op,s) for op in OPERATORS for s in range(1,6)]
    per_seed, summaries, files, pairing = [], [], [], {}
    stored_summary=list(csv.DictReader((archived_report/'seed_summary.csv').open()))
    stored_effect=list(csv.DictReader((archived_report/'paired_effects.csv').open()))
    main=[]
    def compare(actual,expected,label):
        if not np.allclose(actual,expected,atol=1e-10,rtol=0):
            raise ValueError(f'Raw recomputation mismatch: {label}: {actual} != {expected}')
    for ds,shot in itertools.product(('ModelNet40','ShapeNet70'),(1,5)):
        seeds=[]
        for seed in (7,17,27):
            arrays={}
            for method in ('P0','E2'):
                batches=[]
                for side,op,severity in conditions:
                    folds=[]
                    name='clean' if side=='clean' else f'{side}_{op}_{severity}'
                    for fold in range(5):
                        path=legacy/'sac-review-v1/predictions/test'/ds/f'fold{fold}/shot{shot}/seed{seed}'/method/'best'/(name+'.npz')
                        meta=json.loads(path.with_suffix('.json').read_text())
                        identity=dict(pool='test',dataset=ds,fold=fold,shot=shot,seed=seed,method=method,checkpoint='best',
                                      family='P0',condition=[side,op,severity],episodes=700,smoke=False)
                        if any(meta.get(k)!=v for k,v in identity.items()) or meta.get('smoke') is not False:
                            raise ValueError(f'Legacy metadata mismatch: {path}')
                        with np.load(path,allow_pickle=False) as d:
                            target,pred=d['local_target'],d['pred']
                            if target.shape!=(700,75) or pred.shape!=target.shape or not np.isin(pred,range(5)).all():
                                raise ValueError(f'Incomplete legacy output: {path}')
                            if not np.all(np.stack([(target==c).sum(-1) for c in range(5)],-1)==15):
                                raise ValueError('Historical primary table is balanced only')
                            if not np.array_equal(d['episode_id'],np.arange(700)):
                                raise ValueError('Historical episode order differs')
                            for label in ('local_target','episode_rows','support_rows','query_rows','class_ids','corruption_seeds'):
                                pair=(label,ds,shot,*((side,op,severity) if label=='corruption_seeds' else ()))
                                value=digest(d[label])
                                if pair in pairing and pairing[pair]!=value:
                                    raise ValueError(f'Historical pairing differs: {path}: {label}')
                                pairing[pair]=value
                            accuracy=(pred==target).mean(-1)
                            compare(accuracy,d['episode_accuracy'],'stored episode accuracy')
                            folds.append(accuracy)
                        files.append(dict(path=str(path),sha256=sha256(path)))
                    batches.append(np.stack(folds))
                arrays[method]=np.stack(batches)
            results={}
            for setting in ('clean','robust_query','drop_differential'):
                values={m: a[0] if setting=='clean' else a[1:].reshape(7,5,5,700).mean(1).mean(0) for m,a in arrays.items()}
                if setting=='drop_differential':
                    values={m:arrays[m][0]-v for m,v in values.items()}
                effect=bootstrap(values['E2'],values['P0'])
                row=dict(dataset=ds,shot=shot,seed=seed,setting=setting,comparison='E2-P0',**effect)
                for m in ('P0','E2'):
                    row[m+'_accuracy_pct']=float(100*values[m].mean())
                per_seed.append(row); results[setting]=row
                archived=[r for r in stored_effect if r['dataset']==ds and int(r['shot'])==shot and int(r['seed'])==seed
                          and r['setting']==setting and r['comparison']=='E2-P0']
                if len(archived)!=1:
                    raise ValueError(f'Historical contrast not uniquely identified: {row}')
                compare(row['delta_pp'],float(archived[0]['delta_pp']),'per seed mean')
                compare(row['ci95_pp'],json.loads(archived[0]['ci95_pp']),'conditional interval')
            seeds.append(results)
        for setting in ('clean','robust_query','drop_differential'):
            for identity,value in [('P0','P0_accuracy_pct'),('E2','E2_accuracy_pct'),('E2-P0','delta_pp')]:
                if setting=='drop_differential' and identity!='E2-P0':
                    continue
                vals=[s[setting][value] for s in seeds]
                row=dict(dataset=ds,shot=shot,setting=setting,identity=identity,train_seeds=[7,17,27],
                         values=vals,mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)))
                matches=[r for r in stored_summary if r['dataset']==ds and int(r['shot'])==shot and r['setting']==setting
                         and (r['comparison']==identity if identity=='E2-P0' else r['method']==identity)]
                if len(matches)!=1:
                    raise ValueError(f'Historical summary not uniquely identified: {row}')
                compare(row['mean'],float(matches[0]['mean']),'three seed mean')
                compare(row['sample_sd'],float(matches[0]['sample_sd']),'three seed sample SD')
                summaries.append(row)
        main.append(dict(dataset=ds,shot=shot,clean_delta_pp=float(np.mean([s['clean']['delta_pp'] for s in seeds])),
                         query_delta_pp=float(np.mean([s['robust_query']['delta_pp'] for s in seeds]))))
    write_csv(output/'main_per_seed_recomputed.csv',per_seed)
    write_csv(output/'main_seed_summary_recomputed.csv',summaries)
    result=dict(complete=True,prediction_files=len(files),per_seed_contrasts=len(per_seed),seed_summary_rows=len(summaries),
                tolerance_pp=1e-10,raw_accuracy_recomputed=True,conditional_bootstrap_recomputed=True,
                seed_sample_sd_ddof=1,main=main,
                scope='P0/E2, best, two synthetic datasets, 1/5-shot, seeds 7/17/27, five folds, clean and all query operators/severities')
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'sources.json').write_text(json.dumps(files,indent=2)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',required=True);p.add_argument('--archived-report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv); print(json.dumps(audit(a.legacy,a.archived_report,a.output),indent=2))


if __name__=='__main__':
    main()
