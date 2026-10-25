# Third-party source and licensing record

The following exact upstream Git trees were checked through the official GitHub API. Their audit records are included under `third_party/`. No top-level package license is inferred from public availability.

| Component | Pinned source | Included adaptations | Licensing evidence |
| --- | --- | --- | --- |
| PCIA / DGCNN | [cgye96/A_Closer_Look_At_3DFSL](https://github.com/cgye96/A_Closer_Look_At_3DFSL/tree/8345631cd92987198eabf523eba3ce657597fd37) | `sac/vendor/dgcnn.py`, `sac/vendor/pcia.py`; PCIA interaction in `sac/model.py`; data protocol and augmentation | The recursively enumerated pinned tree has no LICENSE, COPYING, or NOTICE file. Attribution is retained; no third-party redistribution permission is asserted. |
| HPR hyperbolic operations | [Jonathan-UCAS/HPR](https://github.com/Jonathan-UCAS/HPR/tree/169fe6e94e97c940b05950af79aa1130fd65d725) | `sac/vendor/hpr_pmath.py`, `sac/hyperbolic.py` | The pinned tree has no LICENSE, COPYING, or NOTICE file. The source header credits geoopt and Maxim Kochurov; the original header remains intact. |
| ModelNet-C operators | [jiawei-ren/ModelNet-C](https://github.com/jiawei-ren/ModelNet-C/tree/0c82ce0c21165530dcc2c011450767277e92fa96) | `sac/corruptions.py`, `sac/vendor/corrupt_utils_reference.py` | The exact upstream `LICENSE` is preserved in `third_party/ModelNet-C-LICENSE`. |

The PCIA parameter names and architecture are retained. The adapter separates point processing into buckets when corruptions change point counts, restores full-episode features before interaction, and applies exactly one prototype update. The original PCIA implementation remains as an independent parity reference. HPR mathematical functions retain their numerical clamps and copyright attribution. ModelNet-C adaptations use explicit RNG streams and scale point counts to the 512-point input protocol.

Dependencies such as PyTorch, NumPy, SciPy, h5py, matplotlib and pytest are installed separately under their respective licenses; their source code and binary distributions are not bundled. Dataset download links are provided in `DATA.md`; dataset licensing is not changed by this package. The original research code has no newly selected blanket license in this local delivery. Public redistribution of third-party portions remains subject to the corresponding authors' terms.
