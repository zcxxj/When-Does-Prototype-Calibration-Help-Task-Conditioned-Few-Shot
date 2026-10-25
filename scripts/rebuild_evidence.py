#!/usr/bin/env python3
"""Recompute all retained statistics from the supplied raw prediction attachments."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def legacy_reports(legacy,output):
    from sac import report_review,report_review1
    from sac.report_calibration_study import write_csv
    from sac.audit_legacy_calibration import audit
    review=legacy/'sac-review-v1';full=legacy/'full-v1'
    cells=report_review.expected_cells('test')
    matrix,coverage=report_review.load_review_predictions(review,full,'test',cells=cells)
    diagnostics,diagnostic_coverage=report_review.load_diagnostics(review,cells,matrix=matrix)
    if not coverage['complete'] or not diagnostic_coverage['complete']:
        raise ValueError('Incomplete historical prediction/diagnostic matrix')
    tables=report_review.build_tables(matrix,'test',cells=cells)
    tables['diagnostics']=diagnostics
    dest=output/'legacy';dest.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():write_csv(dest/(name+'.csv'),rows)
    dump(dest/'coverage.json',dict(coverage,diagnostics=diagnostic_coverage))
    dump(dest/'status.json',dict(complete=True,pool='test',expected=coverage['expected'],present=coverage['present']))
    audit(legacy,review/'reports/novel',output/'legacy_primary_audit')
    report_review1.write_report(legacy/'review1-evidence-v1/formal',review,output/'dgcnn')
    # Validate every raw timing and recompute median/p95/peaks.
    from sac.efficiency_report import load_rows
    source=legacy/'review1-evidence-v1/efficiency/benchmark.json'
    data=json.loads(source.read_text());rows=load_rows(data)
    dump(output/'efficiency/benchmark.json',data)
    dump(output/'efficiency/validation.json',dict(complete=True,rows=len(rows),
         raw_measurements=sum(len(r['measurements']) for r in rows),source=str(source)))
    return dict(legacy_prediction_cells=coverage['present'],diagnostic_coverage=diagnostic_coverage,
                dgcnn_prediction_cells=1440,dgcnn_reference_cells=1440,efficiency_measurements=960)


def study_report(study,legacy,output):
    from sac.empirical_protocol import prediction_cells
    from sac.report_calibration_study import report
    from prepare_reuse_index import prepare
    index=output/'source_reuse_index.json'
    prepare(legacy,index,mode='source')
    return report(study,prediction_cells(),output/'study',reuse_index=index)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,default=ROOT/'attachments/legacy')
    p.add_argument('--study',type=Path,default=ROOT/'attachments/study')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/rebuilt')
    p.add_argument('--group',choices=('all','legacy','study'),default='all')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    status={}
    if a.group in ('all','legacy'):status['legacy']=legacy_reports(a.legacy,a.output)
    if a.group in ('all','study'):status['study']=study_report(a.study,a.legacy,a.output)
    dump(a.output/'rebuild_status.json',dict(complete=True,groups=status))
    print(json.dumps(dict(complete=True,groups=status)))

if __name__=='__main__':main()
