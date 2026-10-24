"""Reproduce Table 3 (AUC and run time) of Liu et al. (2008).

iForest is run twice: our from-scratch implementation (iforest.py) and
scikit-learn's IsolationForest. LOF (k = 10) is run on the datasets where the
paper reports LOF. Every method is trained and evaluated on the full dataset
with labels used only for AUC, as in the paper."""
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from data import FILES, load
from iforest import IsolationForestScratch

PAPER = {  # Table 3 of the paper: iForest AUC, LOF AUC, iForest total time (s)
    "Http (KDDCUP99)": (1.00, None, 15.58), "ForestCover": (0.88, None, 16.33),
    "Mulcross": (0.97, None, 12.52), "Smtp (KDDCUP99)": (0.88, None, 2.72),
    "Shuttle": (1.00, 0.55, 3.13), "Mammography": (0.86, 0.67, 0.66),
    "Annthyroid": (0.82, 0.72, 0.51), "Satellite": (0.71, 0.52, 1.63),
    "Pima": (0.67, 0.49, 0.28), "Breastw": (0.99, 0.37, 0.28),
    "Ionosphere": (0.85, 0.89, 0.48),
}


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="repetitions with different seeds")
    ap.add_argument("--trees", type=int, default=100, help="t, number of trees")
    ap.add_argument("--psi", type=int, default=256, help="sub-sampling size")
    ap.add_argument("--lof-k", type=int, default=10, help="LOF neighbourhood size")
    args = ap.parse_args()

    rows = []
    for name in FILES:
        X, y = load(name)
        own_auc, own_time, sk_auc, sk_time = [], [], [], []
        for seed in range(args.runs):
            model, t_fit = timed(lambda: IsolationForestScratch(args.trees, args.psi, seed).fit(X))
            s, t_eval = timed(lambda: model.score(X))
            own_auc.append(roc_auc_score(y, s)); own_time.append(t_fit + t_eval)

            sk = IsolationForest(n_estimators=args.trees, max_samples=args.psi,
                                 random_state=seed, n_jobs=1)
            _, t_fit = timed(lambda: sk.fit(X))
            s, t_eval = timed(lambda: -sk.score_samples(X))
            sk_auc.append(roc_auc_score(y, s)); sk_time.append(t_fit + t_eval)

        lof_auc = lof_time = np.nan
        if PAPER[name][1] is not None:
            lof = LocalOutlierFactor(n_neighbors=args.lof_k, n_jobs=1)
            _, lof_time = timed(lambda: lof.fit(X))
            lof_auc = roc_auc_score(y, -lof.negative_outlier_factor_)

        rows.append({
            "dataset": name, "n": len(X), "d": X.shape[1], "anomaly %": round(100 * y.mean(), 2),
            "paper iForest AUC": PAPER[name][0],
            "own iForest AUC": np.mean(own_auc), "own iForest AUC sd": np.std(own_auc),
            "sklearn iForest AUC": np.mean(sk_auc), "sklearn iForest AUC sd": np.std(sk_auc),
            "paper LOF AUC": PAPER[name][1], "LOF AUC": lof_auc,
            "paper iForest time (s)": PAPER[name][2],
            "own iForest time (s)": np.mean(own_time), "sklearn iForest time (s)": np.mean(sk_time),
            "LOF time (s)": lof_time,
        })
        print(f"{name:18s} paper {PAPER[name][0]:.2f} | own {np.mean(own_auc):.3f}±{np.std(own_auc):.3f}"
              f" | sklearn {np.mean(sk_auc):.3f}±{np.std(sk_auc):.3f} | LOF {lof_auc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/table3.csv", index=False)
    df.round(3).to_markdown("results/table3.md", index=False)


if __name__ == "__main__":
    main()
