"""Scientific invariants for macro/micro pairing and uncertainty, without models."""
import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sac.report_calibration_study import (
    aggregate, bootstrap, cell_key, diagnostic_rows, load_predictions, score_predictions, _path)


class CalibrationStudyReporting(unittest.TestCase):
    def score(self, quota, before=None, after=None, method='E2'):
        target=np.repeat(np.arange(5),quota)[None]
        before=target.copy() if before is None else before[None]
        after=target.copy() if after is None else after[None]
        logits=np.full((*target.shape,5),-8.)
        np.put_along_axis(logits,before[...,None],0.,axis=-1)
        return score_predictions(target,logits,after,5,method)

    def test_macro_micro_flip_contributions_differ_as_required(self):
        target=np.repeat(np.arange(5),[5,10,15,20,25])
        before=target.copy();before[0]=1
        after=target.copy();after[-1]=0
        s=self.score([5,10,15,20,25],before,after)
        self.assertAlmostEqual(float(s['corrected_micro'][0]-s['damaged_micro'][0]),0.)
        self.assertAlmostEqual(float(s['corrected_macro'][0]-s['damaged_macro'][0]),1/25-1/125)
        for metric in ('macro','micro'):
            np.testing.assert_allclose(s[metric]-s['initial_'+metric],s['corrected_'+metric]-s['damaged_'+metric],atol=1e-14)

    def test_balanced_macro_micro_and_net_equal(self):
        target=np.repeat(np.arange(5),15); before=target.copy();before[7]=4
        after=target.copy();after[27]=3
        s=self.score([15]*5,before,after)
        for name in ('','initial_','corrected_','damaged_'):
            np.testing.assert_allclose(s[name+'macro'],s[name+'micro'],atol=1e-14)

    def test_empty_group_agreement_missing_and_queryonly_falls_back(self):
        s=self.score([15]*5,np.zeros(75,dtype=int),method='QONLY')
        self.assertTrue(np.isnan(s['group_class_accuracy'][0,1:]).all())
        np.testing.assert_array_equal(s['anchor_fraction'][0,1:],1)
        self.assertEqual(s['anchor_fraction'][0,0],0)
        self.assertAlmostEqual(s['group_class_accuracy'][0,0],.2)

    def test_purity_does_not_replace_prototype_class_agreement(self):
        target=np.repeat(np.arange(5),[5,10,15,20,25]); before=np.zeros(75,dtype=int)
        s=self.score([5,10,15,20,25],before)
        self.assertAlmostEqual(s['group_purity'][0,0],25/75)
        self.assertAlmostEqual(s['group_class_accuracy'][0,0],5/75)

    def test_soft_probability_uniform_and_shot_anchor(self):
        target=np.repeat(np.arange(5),15)[None]
        s=score_predictions(target,np.zeros((1,75,5)),target,5,'SOFTK')
        np.testing.assert_allclose(s['soft_entropy'],np.log(5))
        np.testing.assert_allclose(s['soft_max_probability'],.2)
        np.testing.assert_allclose(s['soft_mass'],15)
        np.testing.assert_allclose(s['anchor_fraction'],.25)

    def test_group_and_reliability_bin_contributions_reconstruct_total(self):
        target=np.repeat(np.arange(5),[5,10,15,20,25]);before=target.copy();before[:3]=1;before[20:22]=4
        after=target.copy();after[50:56]=0
        s=self.score([5,10,15,20,25],before,after)
        c=dict(protocol='imbalanced',pool='test',dataset='ModelNet40',fold=0,shot=5,seed=7,method='E2',checkpoint='best',side='clean',operator=None,severity=0)
        rows=diagnostic_rows(c,s)
        ep=next(r for r in rows if r['scope']=='episode')
        self.assertAlmostEqual(ep['group_size_sd'],np.std(s['group_size'],ddof=0))
        for metric,name in itertools.product(('macro','micro'),('corrected','damaged')):
            field=name+'_'+metric+'_pp'
            for scope in ('prototype_class','reliability_bin'):
                self.assertAlmostEqual(sum(r[field] for r in rows if r['scope']==scope),ep[field])

    def test_bootstrap_matches_historical_shared_fold_resampling(self):
        rng=np.random.RandomState(91); a=rng.rand(5,17); b=rng.rand(5,17)
        result=bootstrap(a,b,resamples=2000)
        rng=np.random.RandomState(400007)
        samples=[float((100*(a-b))[:,rng.randint(17,size=17)].mean()) for _ in range(2000)]
        np.testing.assert_allclose(result['ci95_pp'],np.percentile(samples,[2.5,97.5]),atol=1e-12,rtol=0)

    def fixture(self,seeds=(7,17,27),protocols=('balanced','imbalanced')):
        cells=[];matrix={}
        for protocol,seed,fold,method in itertools.product(protocols,seeds,range(5),('P0','E2','QONLY')):
            c=dict(protocol=protocol,pool='test',dataset='ModelNet40',fold=fold,shot=1,seed=seed,method=method,checkpoint='best',side='clean',operator=None,severity=0)
            cells.append(c)
            val=.6+fold*.001 + (0 if method=='P0' else seed*.001 if method=='E2' else -.02)
            if protocol=='imbalanced': val-=.1
            matrix[cell_key(c)]={m:np.full(7,val) for m in ('macro','micro')}
        return cells,matrix

    def test_seed_sample_sd_and_rule_comparisons_for_every_seed(self):
        cells,matrix=self.fixture();t=aggregate(matrix,cells,resamples=20)
        row=next(r for r in t['seed_summary'] if r.get('comparison')=='E2-P0' and r['protocol']=='balanced' and r['metric']=='micro')
        self.assertAlmostEqual(row['mean'],1.7)
        self.assertAlmostEqual(row['sample_sd'],1.)
        self.assertEqual(row['train_seeds'],[7,17,27])
        q=[r for r in t['paired_effects'] if r['comparison']=='QONLY-E2' and r['protocol']=='balanced' and r['metric']=='micro']
        self.assertEqual({r['seed'] for r in q},{7,17,27})
        self.assertTrue(all(r['delta_pp']<0 for r in q))

    def test_missing_fold_omits_seed_and_three_seed_summary(self):
        cells,matrix=self.fixture(protocols=('balanced',));del matrix[next(k for k in matrix if k[3]==4 and k[5]==17 and k[6]=='E2')]
        t=aggregate(matrix,cells,resamples=20)
        self.assertFalse(any(r.get('method')=='E2' for r in t['seed_summary']))
        self.assertFalse(any(r.get('comparison')=='E2-P0' for r in t['seed_summary']))
        self.assertFalse(any(r.get('method')=='E2' and r['seed']==17 for r in t['per_seed']))
        self.assertTrue(any(r.get('method')=='P0' for r in t['seed_summary']))

    def test_protocols_have_distinct_keys_and_never_cross_pair(self):
        cells,matrix=self.fixture();self.assertEqual(len(matrix),len(cells))
        t=aggregate(matrix,cells,resamples=20)
        gains=[r for r in t['seed_summary'] if r.get('comparison')=='E2-P0' and r['metric']=='micro']
        self.assertEqual({r['protocol'] for r in gains},{'balanced','imbalanced'})
        np.testing.assert_allclose([r['mean'] for r in gains],1.7)

    def test_reject_protocol_metadata_mismatch(self):
        cells,_=self.fixture();c=cells[0]
        with tempfile.TemporaryDirectory() as d:
            p=_path(d,c);p.parent.mkdir(parents=True);np.savez(p,pred=np.zeros((700,75)))
            p.with_suffix('.json').write_text(json.dumps(dict(c,protocol='imbalanced')))
            with self.assertRaisesRegex(ValueError,'Metadata'):
                load_predictions(d,[c],partial=True)

    def write_pair_fixture(self, root, cell, override=None, shared=60, support_offset=0):
        # Complete shapes, intentionally artificial values for identity tests.
        c=cell; protocol=c['protocol']; n=700
        quota=[15]*5 if protocol=='balanced' else [5,10,15,20,25]
        target=np.tile(np.repeat(np.arange(5),quota),(n,1))
        logits=np.full((n,75,5),-8.)
        np.put_along_axis(logits,target[...,None],0.,axis=-1)
        query=np.arange(5,80) if protocol=='balanced' else np.r_[np.arange(5,5+shared),np.arange(80,80+75-shared)]
        supports=np.tile(np.arange(5)+support_offset,(n,1))
        queries=np.tile(query,(n,1))
        data=dict(local_target=target,pred=target,logits_before=logits,logits_after=logits,
                  episode_id=np.arange(n),episode_accuracy=np.ones(n),macro_accuracy=np.ones(n),micro_accuracy=np.ones(n),
                  query_quota=np.tile(quota,(n,1)),corruption_seeds=np.full((n,80),-1),
                  support_rows=supports,query_rows=queries,episode_rows=np.concatenate((supports,queries),axis=1),
                  class_ids=np.tile(np.arange(5),(n,1)))
        fingerprints={'scientific.py':'a'*64}
        code=hashlib.sha256(json.dumps(fingerprints,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        meta=dict(c,family='P0',condition=['clean',None,0],episodes=n,smoke=False,
                  manifest_sha256=('a' if protocol=='balanced' else 'b')*64,
                  source_manifest_sha256='c'*64,point_indices_sha256='d'*64,checkpoint_sha256='e'*64,
                  data_sha256='f'*64,scientific_code_sha256=code,scientific_code_files=fingerprints)
        meta.update(override or {})
        path=_path(root,c);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**data);path.with_suffix('.json').write_text(json.dumps(meta))

    def test_cross_protocol_checkpoint_source_and_points_must_match(self):
        base=dict(pool='test',dataset='ModelNet40',fold=0,shot=1,seed=7,method='P0',checkpoint='best',side='clean',operator=None,severity=0)
        cells=[dict(base,protocol=p) for p in ('balanced','imbalanced')]
        for identity in ('checkpoint_sha256','source_manifest_sha256','point_indices_sha256'):
            with self.subTest(identity=identity),tempfile.TemporaryDirectory() as d:
                self.write_pair_fixture(d,cells[0]);self.write_pair_fixture(d,cells[1],{identity:'0'*64})
                with self.assertRaisesRegex(ValueError,'Unpaired input identity'):
                    load_predictions(d,cells,partial=True)

    def test_cross_protocol_support_cannot_change(self):
        base=dict(pool='test',dataset='ModelNet40',fold=0,shot=1,seed=7,method='P0',checkpoint='best',side='clean',operator=None,severity=0)
        cells=[dict(base,protocol=p) for p in ('balanced','imbalanced')]
        with tempfile.TemporaryDirectory() as d:
            self.write_pair_fixture(d,cells[0]);self.write_pair_fixture(d,cells[1],support_offset=100)
            with self.assertRaisesRegex(ValueError,'Unpaired input identity'):
                load_predictions(d,cells,partial=True)

    def test_cross_protocol_query_overlap_is_exactly_sixty(self):
        base=dict(pool='test',dataset='ModelNet40',fold=0,shot=1,seed=7,method='P0',checkpoint='best',side='clean',operator=None,severity=0)
        cells=[dict(base,protocol=p) for p in ('balanced','imbalanced')]
        with tempfile.TemporaryDirectory() as d:
            self.write_pair_fixture(d,cells[0]);self.write_pair_fixture(d,cells[1],shared=61)
            with self.assertRaisesRegex(ValueError,'exactly 60'):
                load_predictions(d,cells,partial=True)
            self.write_pair_fixture(d,cells[1],shared=60)
            _,_,coverage,_=load_predictions(d,cells,partial=True)
            self.assertTrue(coverage['complete'])


if __name__=='__main__': unittest.main()
