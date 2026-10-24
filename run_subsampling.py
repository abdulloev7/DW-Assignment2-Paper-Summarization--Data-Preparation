"""Reproduce Figure 6: AUC and run time of iForest vs sub-sampling size psi
on Http and ForestCover (scikit-learn implementation, t = 100)."""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from data import load

RUNS = 5
PSIS = [2 ** k for k in range(1, 16)]  # 2 ... 32768

rows = []
for name in ["Http (KDDCUP99)", "ForestCover"]:
    X, y = load(name)
    for psi in PSIS:
        aucs, times = [], []
        for seed in range(RUNS):
            t = time.perf_counter()
            m = IsolationForest(n_estimators=100, max_samples=psi, random_state=seed, n_jobs=1).fit(X)
            s = -m.score_samples(X)
            times.append(time.perf_counter() - t)
            aucs.append(roc_auc_score(y, s))
        rows.append({"dataset": name, "psi": psi, "AUC": np.mean(aucs), "AUC sd": np.std(aucs),
                     "time (s)": np.mean(times)})
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/subsampling.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (name, g) in zip(axes, df.groupby("dataset", sort=False)):
    ax.errorbar(g["psi"], g["AUC"], yerr=g["AUC sd"], marker="o", ms=3, label="AUC")
    ax.set_xscale("log", base=2); ax.set_xlabel("sub-sampling size ψ"); ax.set_ylabel("AUC")
    ax.set_title(name)
    ax2 = ax.twinx(); ax2.plot(g["psi"], g["time (s)"], "--", color="grey"); ax2.set_ylabel("time (s)")
fig.tight_layout(); fig.savefig("results/subsampling.png", dpi=200)
