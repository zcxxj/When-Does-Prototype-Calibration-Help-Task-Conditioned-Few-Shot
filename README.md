# When Does Prototype Calibration Help Task-Conditioned Few-Shot Point Cloud Classification?

This repository contains inference and from-scratch training code, fixed sampling manifests, and statistical reconstruction tools. `PCIA-E`, `SAC`, `UNIT`, `QONLY`, and `SOFTK` share the same frozen task-conditioned representation. DGCNN and HPR-D comparisons have the narrower scopes recorded with their evidence. Classification is batch transductive: the complete unlabeled query batch participates in inference.

Repository: [zcxxj/When-Does-Prototype-Calibration-Help-Task-Conditioned-Few-Shot](https://github.com/zcxxj/When-Does-Prototype-Calibration-Help-Task-Conditioned-Few-Shot).

Create a Python 3.12 environment and install `requirements.txt`. The validated tensor environment uses PyTorch 2.13.0 with CUDA 13.0; a supported CUDA-capable device is required for the full experiment runtime. No training or data download runs during installation.

```sh
python -m pip install -r requirements.txt
python -m pytest -q tests
```

The complete reproduction package also requires approximately 41.1 GB of raw prediction records and inference weights in `attachments/` under the repository root. These separate attachments are excluded from Git; cloning this repository does not download them, and no attachment download URL is provided here. The complete raw-result reconstruction and supplied-weight evaluation commands below require those attachments. Training from scratch uses the separately acquired datasets.

`assets_manifest.json` associates scientific files with relative paths, exported SHA-256 values, and original-file SHA-256 values. `SHA256SUMS` covers both the repository files and the separate attachments. Checkpoint receipts in `validation/checkpoints/` document exact per-tensor equality to the training originals and strict loading. Inference checkpoints omit optimizer, scheduler, and random-state payloads; they cannot resume an old training trajectory. The training entry point creates complete new training states. After placing the attachments, verify the complete package with:

```sh
python scripts/verify_release.py --weights --require-complete
```

Obtain the official few-shot H5 releases as described in `DATA.md`, then place them under `data/`. Validate filenames and dimensions with:

```sh
python -m sac.campaign prepare --data-root data --out outputs/data-check
```

A single from-scratch training job is:

```sh
python -m sac.train --data-root data --out outputs/training \
  --dataset ModelNet40 --fold 0 --shot 1 --family P0 --seed 7 --device cuda
```

Training uses the fixed 80-epoch maximum, 400 training and 600 validation episodes, Adam learning rate 0.0008, StepLR(5, 0.5), and early stopping patience 30. Training labels are separate from inference query labels. See the frozen experiment configuration for the complete matrix and result interpretation.

Test data were inspected only after all experiments had been completed. The protocol field `test_previously_viewed` records whether test data had been inspected before completion of the original experiments, so its value is `false`, including when reproducing those experiments later.

The experimental validation records in `validation/` are retained from the original release. `validation/scientific_code_identity.json` records both the current code identity and the original source identity after correction of the timeline metadata. No experiments were rerun for this correction.

Recompute statistics from all supplied raw predictions. `EVIDENCE.md` maps each retained comparison to its exact evidence and reconstruction entry:

```sh
python scripts/rebuild_evidence.py --output outputs/rebuilt
```

Episode bootstrap intervals are conditional on a trained checkpoint. They are distinct from the sample standard deviation across the three training seeds. The independent evidence includes negative gains and does not imply a consistently best calibration rule.

Third-party sources and actual licensing evidence are listed in `THIRD_PARTY.md`. This package does not grant a license on behalf of third-party authors or redistribute the raw datasets.

Run the expanded balanced/imbalanced matrix sequentially (1,920 new full-episode frontend jobs) after extracting the attachments:

```sh
python scripts/evaluate_matrix.py --data-root data --out outputs/study --list
python scripts/evaluate_matrix.py --data-root data --out outputs/study --device cuda
```

The default leaves the 600 verified balanced 5-shot seed-7 cells in the supplied legacy archive. `--rerun-reused` recomputes those cells. Filters for protocol, dataset, fold, shot, and seed allow explicit subsets. Existing outputs are validated against protocol and input identities before resumption. Changed inputs or partial/conflicting artifacts cause an error.

For a fresh run using exported inference weights, prepare the matching reuse index and rebuild its statistics:

```sh
python scripts/prepare_reuse_index.py --output outputs/exported_reuse_index.json \
  --checkpoint-identity exported
python -m sac.report_calibration_study --out outputs/study \
  --reuse-index outputs/exported_reuse_index.json --destination outputs/study-report
```

For the delivered original predictions, use the `source` identity mode as shown in `EVIDENCE.md`. Both identities are explicitly linked by verified tensor equality.

The narrower retained historical comparisons have a separate sequential entry:

```sh
python scripts/evaluate_legacy.py --data-root data --out outputs/legacy --group dgcnn --list
python scripts/evaluate_legacy.py --data-root data --out outputs/legacy --group hyperbolic --device cuda
python scripts/evaluate_legacy.py --data-root data --out outputs/legacy --group scan --device cuda
```

These commands use supplied inference weights and generate fresh outputs. `--group pcia` covers the prior three-seed PCIA-E/SAC tables, including the 5-shot full corruption range. No command creates a new training run implicitly. `scripts/replay_episode.py --help` documents a single-episode check against any selected PCIA calibration prediction attachment.

`--group all` also regenerates the historical representative controls and auxiliary conditions required by the complete old report. Rebuild fresh historical reports with:

```sh
python -m sac.report_review --out outputs/legacy/historical \
  --source-out attachments/legacy/full-v1 --pool test
python -m sac.report_review1 --new-out outputs/legacy/dgcnn \
  --reference-out outputs/legacy/historical --out outputs/legacy-factorial
```

The original efficiency table is reconstructed from its 960 recorded timings by `rebuild_evidence.py`. Repeat the same timing protocol on an idle selected CUDA device with:

```sh
python scripts/benchmark_efficiency.py --data-root data --out outputs/efficiency --device cuda
```

Timing values depend on the recorded hardware. This command requires exclusive compute use of that selected GPU during measurement; it does not stop other processes or manage devices.
