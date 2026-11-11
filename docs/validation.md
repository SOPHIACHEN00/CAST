# Initial source snapshot validation

Date: 2026-11-10.

This is a curated source upload. Existing experiment and analysis implementations were copied with UTF-8/LF normalization; no algorithm changes were made.

## Completed checks

- All 81 copied source files matched their original files after encoding/newline normalization. Both original and published SHA-256 values are in `source_manifest.json`.
- All 72 Python source files parsed and compiled without execution.
- 106 local import references were checked, along with campaign provenance file lists and Bash launcher dependencies.
- All four analysis adapters retained their expected paths to the shared `Analysis Code` helpers.
- All four `rq1_*_campaign.py` dry-run previews completed successfully.
- Multi-free-rider, honest-control, and the two merge-script help commands completed successfully.
- Existing regression suite: **136 passed in 38.42 seconds** across the eight included test modules. These cover the three server aggregation rules, single/multiple attackers, task matrices, result validation, and RQ1 summaries using synthetic data.
- Reviewed publish inventory for credential patterns, absolute machine paths, datasets, model weights, generated results, caches, and backups.

Tests ran in the existing Anaconda environment with PyTorch 2.6.0+cpu, NumPy 1.24.3, pandas 1.5.3, SciPy 1.10.1, scikit-learn 1.3.0, and pytest 7.4.0. The dependency files preserve the older source-environment NumPy/pandas/SciPy/PyTorch versions and use the checked scikit-learn version; a fresh install of that set was not tested.

The workspace's bundled Uni2TS declares version 2.0.0. The optional forecast requirements use that declaration rather than the stale 1.2.0 example in the old workspace README. The third-party package itself is not vendored in this snapshot.

## Validation boundaries

- No full dataset training, GPU campaign, Moirai weight download, ARIMA forecast fitting, or IMDB embedding generation was run for this upload.
- The local test environment lacks torchvision, so the complete FLCE import/CLI was not executed. Existing tests exercise actual runner functions and argument definitions through isolated fixtures, without importing the entire dataset stack.
- No real-data RQ1/RQ2 tables were regenerated. Dataset-specific adapters require their original CSV layouts and, where applicable, a matching raw-file hash manifest.

The source snapshot is verified for curation integrity and the checks above; this report does not claim full experimental reproduction.
