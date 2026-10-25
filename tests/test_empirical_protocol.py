import copy
import json
import pytest
from sac.empirical_protocol import prediction_cells, evaluation_jobs, cell_key, freeze_document


def test_frozen_matrix_has_exact_budget_and_no_redundant_unit():
    cells=prediction_cells()
    assert len(cells)==8760 and len({cell_key(c) for c in cells})==8760
    assert sum(c['reused'] for c in cells)==600
    assert len(evaluation_jobs())==1920
    assert freeze_document()['frontend_episodes']==1344000
    assert not any(c['shot']==1 and c['method']=='UNIT' for c in cells)
    assert all(c['protocol']=='balanced' and c['shot']==5 and c['seed']==7 for c in cells if c['reused'])
    assert {c['severity'] for c in cells if c['protocol']=='imbalanced' and c['side']=='query'}=={3}
    assert not any(c['protocol']=='imbalanced' and c['side'] in ('support','both') for c in cells)

