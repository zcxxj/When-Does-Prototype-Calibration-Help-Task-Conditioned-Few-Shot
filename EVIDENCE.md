# Evidence and regeneration map

All paths below are relative to the release root. Raw arrays retain their original bytes. JSON records retain scientific identities while filesystem locations are made portable. `assets_manifest.json` records both original and exported file hashes.

| Retained result | Raw evidence | Statistical reconstruction |
|---|---|---|
| Expanded calibration study: balanced/imbalanced, both shots, three seeds | `attachments/study/protocols/*/predictions/test/`; 600 reused cells identified by `attachments/study/reuse_index.json` | `sac.report_calibration_study`: 8,760 required cells, paired episode bootstrap, seed mean/sample SD, raw-logit diagnostics |
| Historical PCIA-E/SAC, HPR-D and PCIA-H/HR comparisons | `attachments/legacy/sac-review-v1/predictions/test/`, matching `diagnostics/test/` | `sac.report_review`: full historical test matrix; `sac.audit_legacy_calibration` independently checks 4,320 PCIA-E/SAC records against archived statistics |
| ScanObjectNN clean comparisons | `attachments/legacy/sac-review-v1/predictions/test/ScanObjectNN/` | The same historical report, restricted to ScanObjectNN, three folds and seed 7; see `protocol_clean_best_last.csv` |
| DGCNN without PCIA, before/after calibration | `attachments/legacy/review1-evidence-v1/formal/predictions/test/`, plus paired PCIA-E/SAC reference arrays | `sac.report_review1`: 1,440 DGCNN and 1,440 reference cells, fixed seed 7; descriptive paired contrasts |
| Latency and peak memory | `attachments/legacy/review1-evidence-v1/efficiency/benchmark.json` | `sac.efficiency_report.load_rows`: recomputes medians, interpolated p95 and maximum allocation/reservation from all 960 measurements |

Run the complete statistical reconstruction without models or datasets:

```sh
python scripts/rebuild_evidence.py --output outputs/rebuilt
```

`--group legacy` or `--group study` selects one independently complete branch. Missing raw cells, mismatched episode identities or invalid diagnostics cause an error; there is no partial-result option. The historical branch writes `legacy/`, `legacy_primary_audit/`, `dgcnn/`, and `efficiency/`; the expanded branch writes `study/`. These output directories contain newly computed CSVs and coverage/status records.

To rebuild the expanded statistics explicitly:

```sh
python scripts/prepare_reuse_index.py --legacy attachments/legacy \
  --output outputs/source_reuse_index.json --checkpoint-identity source
python -m sac.report_calibration_study --out attachments/study \
  --reuse-index outputs/source_reuse_index.json --destination outputs/study
```

Archived predictions use the original training-checkpoint hash as their pairing identity. The inference files supplied here have different serialized bytes but identical tensors. Metadata includes `original_checkpoint_sha256`, `exported_checkpoint_sha256`, and `model_state_sha256`; no original hash is claimed to be the exported file hash. After running fresh inference with the supplied weights, build the reuse index with `--checkpoint-identity exported` so the reused predictions and new records share the same explicit weight identity. The 600 reused raw arrays and their regenerated metadata hashes are checked in both modes.

`scripts/evaluate_matrix.py` and `scripts/evaluate_legacy.py` reproduce inference from datasets and the supplied inference checkpoints. `sac.train` is the separate from-scratch training entry. `scripts/benchmark_efficiency.py` repeats the timing protocol on a user-selected idle CUDA device. Timing measurements depend on hardware; the supplied RTX 4090 benchmark is preserved as the evidence underlying its reported numbers.

## Numerical repetition across devices

`python scripts/audit_numerical_repeat.py` compares the 2,480 newly recorded balanced P0/E2 cells that overlap the earlier archive, checking labels, object rows, and corruption seeds before comparing predictions. Reused 5-shot seed-7 records have zero repeat difference by identity. The delivered audit summarizes 1,378 changed predictions among 130,200,000 queries, with a maximum 0.000381 pp change in a matched three-seed clean/query mean. Changed predictions occur in RTX 5090 versus earlier RTX 4090 evaluations; matching RTX 4090 evaluations in this audit have no changed predictions. These FP32 runs with TF32 disabled are close numerically but are not globally bitwise identical across devices.

Each current rule comparison still uses one matched frontend pass. Historical full-severity tables and fresh rule-comparison tables retain their own records; they are not silently merged. The script writes the full cell comparison and per-seed matched-scope differences to `outputs/numerical-repeat/`. Validation copies are supplied under `validation/numerical_repeat/`.
