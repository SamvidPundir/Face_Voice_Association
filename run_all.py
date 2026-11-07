"""Run every ablation × seeds, then aggregate into results/table.md.

  python run_all.py                  # train + aggregate
  python run_all.py --aggregate      # aggregate only
  python run_all.py --seeds 0        # quick single-seed pass
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

CONFIGS = {
    "mean":            ["--agg", "mean"],
    "attn":            ["--agg", "attn"],
    "transformer":     ["--agg", "transformer"],
    "transformer_T8":  ["--agg", "transformer", "--T", "8"],
    "transformer_T16": ["--agg", "transformer", "--T", "16"],
    "transformer_noid": ["--agg", "transformer", "--id_w", "0"],
    "transformer_sameclip": ["--agg", "transformer", "--p_cross", "0"],
    "mean_sameclip":   ["--agg", "mean", "--p_cross", "0"],
}


def ms(v, pct=True):
    v = np.array(v) * (100 if pct else 1)
    return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def aggregate(out):
    runs = {}
    for f in glob.glob(os.path.join(out, "*_s*.json")):
        r = json.load(open(f))
        runs.setdefault(r["args"]["name"], []).append(r)
    order = [c for c in CONFIGS if c in runs] + [c for c in runs if c not in CONFIGS]
    L = ["# Results (test identities, mean ± std over seeds, %)\n"]
    L.append("## Verification (EER ↓ / AUC ↑)\n")
    L.append("| config | seeds | EER cross-video, rand neg | EER cross-video, gender neg | EER same-clip, rand neg | AUC cross-video, rand neg | F1@val-thr |")
    L.append("|---|---|---|---|---|---|---|")
    for c in order:
        R = runs[c]; g = lambda k, m: [r["test"][k][m] for r in R]
        L.append(f"| {c} | {len(R)} | {ms(g('cross_video/random','EER'))} | {ms(g('cross_video/gender','EER'))} | "
                 f"{ms(g('same_clip/random','EER'))} | {ms(g('cross_video/random','AUC'))} | {ms(g('cross_video/random','F1'))} |")
    L.append("\n## Matching accuracy (↑)\n")
    keys = ["match1:2/v2f/random", "match1:2/v2f/gender", "match1:2/f2v/random", "match1:2/f2v/gender",
            "match1:10/v2f/random", "match1:10/f2v/random"]
    L.append("| config | " + " | ".join(k.replace("match", "") for k in keys) + " |")
    L.append("|---" * (len(keys) + 1) + "|")
    for c in order:
        L.append(f"| {c} | " + " | ".join(ms([r["test"][k] for r in runs[c]]) for k in keys) + " |")
    L.append("\n## Full-gallery retrieval (identity-level Top-K, ↑)\n")
    L.append("| config | V→F top1 | V→F top5 | V→F top10 | F→V top1 | F→V top5 | F→V top10 |")
    L.append("|---|---|---|---|---|---|---|")
    for c in order:
        cells = [ms([r["test"][f"retrieval/{d}"][f"top{k}"] for r in runs[c]]) for d in ("v2f", "f2v") for k in (1, 5, 10)]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("\n## Robustness (EER, cross-video positives, random neg)\n")
    rk = sorted({k for c in order for r in runs[c] for k in r.get("robustness", {})})
    L.append("| config | clean | " + " | ".join(rk) + " |")
    L.append("|---" * (len(rk) + 2) + "|")
    for c in order:
        clean = ms([r["test"]["cross_video/random"]["EER"] for r in runs[c]])
        cells = [ms([r["robustness"][k]["cross_video/random"] for r in runs[c] if k in r.get("robustness", {})] or [np.nan]) for k in rk]
        L.append(f"| {c} | {clean} | " + " | ".join(cells) + " |")
    L.append("\n## Training\n")
    L.append("| config | best val EER | epochs | train time (s) |\n|---|---|---|---|")
    for c in order:
        R = runs[c]
        L.append(f"| {c} | {ms([r['best_val_eer'] for r in R])} | {np.mean([r['epochs_run'] for r in R]):.0f} | {np.mean([r['train_time_s'] for r in R]):.0f} |")
    open(os.path.join(out, "table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default="")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    if not a.aggregate:
        names = a.only.split(",") if a.only else list(CONFIGS)
        for s in a.seeds.split(","):
            for n in names:
                if os.path.exists(os.path.join(a.out, f"{n}_s{s}.json")):
                    continue
                cmd = [sys.executable, "train.py", "--name", n, "--seed", s, "--out", a.out] + CONFIGS[n] + a.extra
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
    aggregate(a.out)
