"""Validate and index the 600 authorized balanced legacy cells without rewriting them."""
import argparse
import json
from pathlib import Path

import numpy as np

from .empirical_protocol import prediction_cells, cell_key, freeze_document
from .report_calibration_study import digest, sha256


def build_index(legacy, destination):
    legacy, destination = Path(legacy).resolve(), Path(destination).resolve()
    cells = [c for c in prediction_cells() if c['reused']]
    entries, pairings, cache = [], {}, {}

    def hashed(path):
        path = Path(path)
        if path not in cache:
            cache[path] = sha256(path)
        return cache[path]

    def paired(key, value):
        if key in pairings and pairings[key] != value:
            raise ValueError(f'Legacy pairing mismatch: {key}')
        pairings[key] = value

    def relative(path):
        # The index is relocatable with its sibling legacy directory.
        import os
        return os.path.relpath(path, destination.parent)

    for c in cells:
        ds, fold, shot = c['dataset'], c['fold'], c['shot']
        condition = 'clean' if c['side'] == 'clean' else f"{c['side']}_{c['operator']}_{c['severity']}"
        path = legacy / 'sac-review-v1/predictions/test' / ds / f'fold{fold}/shot{shot}/seed7' / c['method'] / 'best' / (condition + '.npz')
        metadata = path.with_suffix('.json')
        meta = json.loads(metadata.read_text())
        expected = {k:c[k] for k in ('pool','dataset','fold','shot','seed','method','checkpoint')}
        expected.update(family='P0',condition=[c['side'],c['operator'],c['severity']],smoke=False,episodes=700)
        if any(meta.get(k) != v for k,v in expected.items()) or meta.get('smoke') is not False:
            raise ValueError(f'Legacy metadata mismatch: {path}')
        if meta.get('checkpoint_epoch') is None:
            raise ValueError(f'Legacy checkpoint epoch absent: {path}')
        manifest = legacy / 'sac-review-v1/episodes/test' / ds / f'fold0/shot{shot}/episodes.npz'
        points = manifest.with_name('point_indices.npz')
        checkpoint = legacy / 'full-v1/runs' / ds / f'fold{fold}/shot{shot}/seed7/P0/best.pt'
        for label, actual in (('manifest',manifest),('checkpoint_path',checkpoint)):
            # A portable mirror may change the installation prefix only.
            original = meta.get(label, '')
            tail = original.split('/artifacts/',1)[1] if '/artifacts/' in original else original.removeprefix('attachments/legacy/')
            if tail != str(actual.relative_to(legacy)):
                raise ValueError(f'Legacy {label} provenance mismatch: {path}')
        with np.load(manifest, allow_pickle=False) as source, np.load(points,allow_pickle=False) as sample, np.load(path,allow_pickle=False) as saved:
            if sample['points'].shape != (512,) or len(np.unique(sample['points'])) != 512:
                raise ValueError('Invalid point-sampling vector')
            expected_shapes = dict(pred=(700,75),local_target=(700,75),logits_before=(700,75,5),
                                   logits_after=(700,75,5),episode_accuracy=(700,),episode_id=(700,),
                                   corruption_seeds=(700,100),point_counts=(700,100))
            if any(k not in saved or saved[k].shape != v for k,v in expected_shapes.items()):
                raise ValueError(f'Incomplete legacy prediction: {path}')
            if not np.array_equal(saved['episode_id'],np.arange(700)):
                raise ValueError('Legacy episode order differs')
            mapping = dict(episode_rows='rows',support_rows='support_rows',query_rows='query_rows',class_ids='class_ids',local_target='query_y')
            for dest, orig in mapping.items():
                if dest not in saved or not np.array_equal(saved[dest],source[orig]):
                    raise ValueError(f'Legacy manifest input mismatch {dest}: {path}')
            target,pred = saved['local_target'],saved['pred']
            if not np.isin(target,range(5)).all() or not np.isin(pred,range(5)).all():
                raise ValueError('Legacy labels outside five-way range')
            quotas = np.stack([(target == k).sum(-1) for k in range(5)],axis=-1)
            if not np.all(quotas == 15):
                raise ValueError('Legacy reuse is strictly balanced')
            if not np.allclose((pred==target).mean(-1), saved['episode_accuracy'],atol=1e-8,rtol=1e-6):
                raise ValueError('Legacy stored accuracy differs from prediction')
            if (not np.isfinite(saved['logits_before']).all() or not np.isfinite(saved['logits_after']).all() or
                    not np.array_equal(pred,saved['logits_after'].argmax(-1))):
                raise ValueError('Legacy logits/predictions disagree')
            pair = (ds,fold,shot,c['side'],c['operator'],c['severity'])
            for name in ('logits_before','corruption_seeds','point_counts'):
                paired((name,*pair),digest(saved[name]))
            paired(('checkpoint_epoch',ds,fold,shot),meta['checkpoint_epoch'])
            paired(('source_manifest',ds,shot),hashed(manifest))
        identities = dict(protocol='balanced',manifest_sha256=hashed(manifest),source_manifest_sha256=hashed(manifest),
                          point_indices_sha256=hashed(points),checkpoint_sha256=hashed(checkpoint))
        entries.append(dict(cell=c,source='sac-review-v1',prediction=relative(path),metadata=relative(metadata),
                            prediction_sha256=hashed(path),metadata_sha256=hashed(metadata),
                            manifest=relative(manifest),point_indices=relative(points),checkpoint_path=relative(checkpoint),
                            **identities))
    index = dict(version=1,matrix_sha256=freeze_document()['matrix_sha256'],expected_reused=600,
                 entries=entries,verified='metadata,raw predictions,manifest rows,balanced quotas,shared logits,damage seeds,points,checkpoint file identity')
    if len(entries) != 600:
        raise ValueError('Reuse index must contain exactly 600 cells')
    destination.parent.mkdir(parents=True,exist_ok=True)
    encoded = json.dumps(index,indent=2)+'\n'
    if destination.exists() and destination.read_text()!=encoded:
        raise ValueError('Refusing to replace a conflicting reuse index')
    if not destination.exists():
        destination.write_text(encoded)
    return index


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    index=build_index(args.legacy,args.output)
    print(json.dumps(dict(reused=len(index['entries']),matrix_sha256=index['matrix_sha256'])))


if __name__=='__main__':
    main()
