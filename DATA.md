# Data acquisition

Use the released H5 files from [PCIA](https://github.com/cgye96/A_Closer_Look_At_3DFSL/tree/8345631cd92987198eabf523eba3ce657597fd37). These files define the few-shot partitions; the original closed-set datasets are not interchangeable replacements.

| Dataset | Official release | Files under `data/` |
| --- | --- | --- |
| ModelNet40-FS | [PCIA ModelNet40-FS](https://drive.google.com/drive/folders/18WTIClNOWMM9s6mjhwhfLBTNzINZLfRf) | `ModelNet40_FS/Modelnet40_FS_train.h5`, `ModelNet40_FS/Modelnet40_FS_test.h5` |
| ShapeNet70-FS | [PCIA ShapeNet70-FS](https://drive.google.com/drive/folders/1DiRlJ7dB7YrGuc90_2NcpNL7MK3eQBZh) | `ShapeNet70_FS/ShapeNet70_FS_train.h5`, `ShapeNet70_FS/ShapeNet70_FS_test.h5` |
| ScanObjectNN-FS | [PCIA ScanObjectNN-FS](https://drive.google.com/drive/folders/1As3Q0-NPDwJn_9xHviftJIQyhrFcHWrh) | `ScanObjectNN_FS/{train,test}_scanobjectnn_PB_T50_RS_fsl_{1,2,3}.h5` |

Preserve filename capitalization, H5 row order, numeric labels, and point arrays. `sac.campaign prepare` checks all ten files and reports all missing files. It never downloads data silently. The dataset owners' terms apply; raw H5 data are not included in this release.

Synthetic training folds separate base classes, and their novel evaluations share the released test pool and episode lists. Scan uses the three released fold files; its validation reuses the training object pool, which is an explicit limitation. Sampling manifests retain object rows and selected point indices. The imbalanced protocol retains each balanced episode's support and exactly 60 of its 75 query objects, then fills quotas without replacement; it is not a pure class-ratio intervention.
