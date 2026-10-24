"""Dataset loading. Real datasets are the ODDS versions redistributed by ADBench
(https://github.com/Minqi824/ADBench); Mulcross is generated locally."""
import os
import urllib.request
import numpy as np
from scipy.stats import chi2

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# name in the paper -> ADBench file, in the order of Table 2/3
FILES = {
    "Http (KDDCUP99)": "16_http.npz",
    "ForestCover": "10_cover.npz",
    "Mulcross": None,
    "Smtp (KDDCUP99)": "34_smtp.npz",
    "Shuttle": "32_shuttle.npz",
    "Mammography": "23_mammography.npz",
    "Annthyroid": "2_annthyroid.npz",
    "Satellite": "30_satellite.npz",
    "Pima": "29_Pima.npz",
    "Breastw": "4_breastw.npz",
    "Ionosphere": "18_Ionosphere.npz",
}


def mulcross(n=262144, d=4, contamination=0.1, distance_factor=2.0, n_clusters=2,
             anomaly_scale=0.3, seed=0):
    """Mulcross-style data (Rocke & Woodruff 1996): one standard normal cluster and
    `n_clusters` small, denser anomaly clusters placed `distance_factor` times the
    97.5% chi-square radius away from the centre in random directions."""
    rng = np.random.default_rng(seed)
    n_out = int(n * contamination)
    X_norm = rng.standard_normal((n - n_out, d))
    radius = distance_factor * np.sqrt(chi2.ppf(0.975, d)) / 2
    parts = []
    for k, size in enumerate(np.array_split(np.arange(n_out), n_clusters)):
        direction = rng.standard_normal(d)
        centre = radius * direction / np.linalg.norm(direction)
        parts.append(centre + anomaly_scale * rng.standard_normal((len(size), d)))
    X = np.vstack([X_norm] + parts)
    y = np.r_[np.zeros(n - n_out), np.ones(n_out)].astype(int)
    return X, y


def load(name):
    if FILES[name] is None:
        return mulcross()
    path = os.path.join(DATA_DIR, FILES[name])
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {FILES[name]} ...")
        urllib.request.urlretrieve(BASE + FILES[name], path)
    d = np.load(path, allow_pickle=False)
    return d["X"].astype(float), d["y"].astype(int)
