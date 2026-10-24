# Reproducing "Isolation Forest" (Liu, Ting & Zhou, ICDM 2008)

This repository reproduces the main experiment of the paper (Table 3: AUC and run time of iForest
compared with LOF) and the sub-sampling experiment (Figure 6).

iForest is run in two ways:

- `iforest.py`: our own implementation written from scratch, following Algorithms 1–3 and Equations 1–2 of the paper.
- `sklearn.ensemble.IsolationForest`: the scikit-learn implementation, used as a cross-check.

## How to run

Tested with Python 3.13 on Linux.

```bash
pip install -r requirements.txt
python run_table3.py        # Table 3, about 6 minutes; writes results/table3.csv and table3.md
python run_subsampling.py   # Figure 6, about 5 minutes; writes results/subsampling.csv and .png
```

The datasets are downloaded automatically into `data/` on first use (see below).

## Parameters

| Parameter | Value | Where |
|---|---|---|
| Number of trees t | 100 | `--trees` (paper default) |
| Sub-sampling size ψ | 256 | `--psi` (paper default) |
| Tree height limit l | ceil(log2 ψ) = 8 | fixed by ψ, as in Algorithm 1 |
| Sampling | without replacement | as in Section 3 |
| Repetitions | 10 (seeds 0–9), reported as mean ± sd | `--runs` |
| LOF neighbours k | 10 | `--lof-k` (paper setting) |
| Training / evaluation data | full dataset for both; labels only for AUC | as in Section 5 |
| Sub-sampling experiment | ψ = 2, 4, …, 32768; t = 100; 5 seeds | `run_subsampling.py` |
| Timing | single thread (`n_jobs=1`), fit + scoring | |
| Mulcross generator | n = 262144, d = 4, contamination 10%, distance factor 2, 2 anomaly clusters, anomaly std 0.3, seed 0 | `data.mulcross` |

## Data

The real datasets are the ODDS versions, as redistributed by ADBench
(https://github.com/Minqi824/ADBench, folder `adbench/datasets/Classical`):
Http, ForestCover, Smtp, Shuttle, Mammography, Annthyroid, Satellite, Pima, Breastw, Ionosphere.
Their sizes match Table 2 of the paper, except Annthyroid (7,200 rows here vs 6,832 in the paper)
and Http (567,498 vs 567,497).

These parts of the paper could not be reproduced:
- **Arrhythmia:** no reachable copy.
- **Mulcross:** the original generator was not available, so `data.py` contains our own Mulcross-style generator. It is an approximation.
- **ORCA and Random Forests:** these baselines are not reproduced, because ORCA has no Python implementation and the paper's unsupervised RF setup needs an n × n proximity matrix.

## Files

- `iforest.py`: from-scratch Isolation Forest
- `data.py`: dataset download and loading, Mulcross generator
- `run_table3.py`: Table 3 experiment
- `run_subsampling.py`: Figure 6 experiment
- `results/`: output tables, log and figure
