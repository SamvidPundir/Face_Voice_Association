"""Figures for the report: ROC, robustness curves, training curves, t-SNE.

  python eval/plots.py --res results --out report/figs
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

COL = {"mean": "#4C72B0", "attn": "#55A868", "transformer": "#C44E52"}

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--out", default="report/figs")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ROC (seed 0) for the three aggregators
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, key, title in [(axes[0], "cross_video__random", "Cross-video positives, random negatives"),
                       (axes[1], "cross_video__gender", "Cross-video positives, gender-matched negatives")]:
    for n in ["mean", "attn", "transformer"]:
        f = os.path.join(a.res, f"{n}_s0_scores.npz")
        if not os.path.exists(f):
            continue
        l, s = np.load(f)[key]
        fpr, tpr, _ = roc_curve(l, s)
        ax.plot(fpr, tpr, label=n, color=COL[n])
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(a.out, "roc.png"), dpi=180); plt.close()

# Robustness curves (mean over seeds)
runs = {}
for f in glob.glob(os.path.join(a.res, "*_s*.json")):
    r = json.load(open(f)); runs.setdefault(r["args"]["name"], []).append(r)


def curve(name, keys):
    R = runs.get(name, [])
    out = []
    for k in keys:
        if k == "clean":
            out.append(np.mean([r["test"]["cross_video/random"]["EER"] for r in R]) * 100)
        else:
            v = [r["robustness"][k]["cross_video/random"] for r in R if k in r.get("robustness", {})]
            out.append(np.mean(v) * 100 if v else np.nan)
    return out


fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
snr_k = ["clean", "audio_snr_20", "audio_snr_10", "audio_snr_5", "audio_snr_0"]
fd_k = ["clean", "frame_drop_25", "frame_drop_50", "frame_drop_75"]
for n in ["mean", "attn", "transformer"]:
    if n in runs:
        axes[0].plot(["clean", "20", "10", "5", "0"], curve(n, snr_k), "o-", label=n, color=COL[n])
        axes[1].plot(["0%", "25%", "50%", "75%"], curve(n, fd_k), "o-", label=n, color=COL[n])
axes[0].set_xlabel("audio SNR (dB)"); axes[0].set_ylabel("EER (%)"); axes[0].legend()
axes[1].set_xlabel("frame drop rate"); axes[1].set_ylabel("EER (%)")
for ax_ in axes: ax_.set_ylim(30, 45); ax_.axhline(50, color="grey", lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(a.out, "robustness.png"), dpi=180); plt.close()

# Training curves (seed 0)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for n in ["mean", "attn", "transformer"]:
    f = os.path.join(a.res, f"{n}_s0.json")
    if os.path.exists(f):
        lg = json.load(open(f))["log"]
        ax[0].plot([x["loss"] for x in lg], label=n, color=COL[n])
        ax[1].plot([x["val_eer"] * 100 for x in lg], label=n, color=COL[n])
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val EER (%)")
ax[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out, "training.png"), dpi=180); plt.close()

# t-SNE of shared space (transformer seed 0), 10 test identities, both modalities
f = os.path.join(a.res, "transformer_s0_scores.npz")
if os.path.exists(f):
    d = np.load(f, allow_pickle=True)
    ids = d["ids"]; uid = np.unique(ids)[:10]
    sel = np.isin(ids, uid)
    X = np.concatenate([d["hv"][sel], d["ha"][sel]])
    lab = np.concatenate([ids[sel], ids[sel]])
    mod = np.array(["face"] * sel.sum() + ["voice"] * sel.sum())
    Z = TSNE(perplexity=20, random_state=0, init="pca").fit_transform(X)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(5.5, 5))
    for j, u in enumerate(uid):
        for m, mk in [("face", "o"), ("voice", "^")]:
            s = (lab == u) & (mod == m)
            plt.scatter(Z[s, 0], Z[s, 1], c=[cmap(j)], marker=mk, s=18, alpha=0.8)
    plt.scatter([], [], c="k", marker="o", label="face"); plt.scatter([], [], c="k", marker="^", label="voice")
    plt.legend(); plt.xticks([]); plt.yticks([]); plt.title("t-SNE of shared space (10 test identities)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "tsne.png"), dpi=180); plt.close()
print("figures written to", a.out)
