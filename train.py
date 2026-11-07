"""Train one configuration on cached features, then evaluate on the held-out test identities.

Example:
  python train.py --name transformer --agg transformer --seed 0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from eval.evaluate import evaluate_split
from eval.metrics import fit_platt
from losses import identity_ce, masked_infonce
from models.heads import FaceVoiceModel
from sampler import FeatureStore, PKSampler


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="transformer")
    p.add_argument("--agg", default="transformer", choices=["mean", "attn", "transformer"])
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--id_w", type=float, default=0.2)
    p.add_argument("--p_cross", type=float, default=0.5)
    p.add_argument("--P", type=int, default=128)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--drop", type=float, default=0.3)
    p.add_argument("--in_drop", type=float, default=0.2)
    p.add_argument("--emb", type=int, default=128)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train_ids", type=int, default=0, help="subsample training identities (scaling study)")
    p.add_argument("--h5", default="data/features.h5,data/features_dev.h5")
    p.add_argument("--manifest", default="data/manifest.csv,data/manifest_dev.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--no_robust", action="store_true")
    return p.parse_args(argv)


def subsample_T(store, idx, T):
    """Evenly subsample the cached sequence to T frames (for the T ablation)."""
    full = store.face.shape[1]
    sel = np.linspace(0, full - 1, T).round().astype(int)
    return store.face[idx][:, sel], store.mask[idx][:, sel], sel


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store = FeatureStore(args.h5, args.manifest)
    tr, va, te = store.split("train"), store.split("val"), store.split("test")
    if args.n_train_ids:
        ids = np.unique(store.df["identity"].values[tr])
        keep = np.random.default_rng(1000 + args.seed).choice(ids, size=min(args.n_train_ids, len(ids)), replace=False)
        tr = tr[np.isin(store.df["identity"].values[tr], keep)]
    print(f"train ids {len(np.unique(store.df['identity'].values[tr]))}, clips {len(tr)}", flush=True)
    full_T = store.face.shape[1]
    if args.T != full_T:  # restrict the store to T evenly spaced frames
        sel = np.linspace(0, full_T - 1, args.T).round().astype(int)
        store.face, store.mask = store.face[:, sel], store.mask[:, sel]
        if store.face_blur is not None:
            store.face_blur = store.face_blur[:, sel]

    sampler = PKSampler(store, tr, P=args.P, K=args.K, p_cross=args.p_cross, T=args.T,
                        seed=args.seed, device=dev)
    model = FaceVoiceModel(agg=args.agg, voice_dim=store.audio.shape[-1], layers=args.layers,
                           heads=args.heads, n_ids=len(sampler.ids) if args.id_w > 0 else 0,
                           max_len=args.T, drop=args.drop, in_drop=args.in_drop, emb=args.emb).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(sampler)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.name}_s{args.seed}"
    ckpt = os.path.join(args.out, "ckpt", f"{tag}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best, bad, log = 1.0, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for f, m, a, ids in sampler:
            hv, ha = model(f, m, a)
            loss = masked_infonce(hv, ha, ids, model.logit_scale) + args.id_w * identity_ce(model, hv, ha, ids)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
        vres, _, _ = evaluate_split(model, store, va, device=dev, full=False, seed=1)
        veer = vres["cross_video/random"]["EER"]
        log.append({"epoch": ep, "loss": tot / len(sampler), "val_eer": veer})
        print(f"[{tag}] ep {ep:3d} loss {tot/len(sampler):.4f} val EER {veer:.4f}", flush=True)
        if veer < best - 1e-4:
            best, bad = veer, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    # threshold + Platt calibration on validation identities
    vres, vraw, _ = evaluate_split(model, store, va, device=dev, full=False, seed=1)
    thr = vres["cross_video/random"]["thr"]
    platt = fit_platt(*vraw["cross_video/random"])

    tres, traw, (hv, ha) = evaluate_split(model, store, te, device=dev, threshold=thr, seed=2)
    out = {"args": vars(args), "best_val_eer": best, "epochs_run": len(log), "train_time_s": time.time() - t0,
           "val_threshold": thr, "platt": platt, "test": tres, "log": log}

    if not args.no_robust:
        rob = {}
        q = lambda r: {k: r[k]["EER"] for k in ["cross_video/random", "cross_video/gender"]}
        rob["shuffled_frames"] = q(evaluate_split(model, store, te, device=dev, shuffle=True, full=False, seed=2)[0])
        if store.audio_snr is not None:
            for j, snr in enumerate([20, 10, 5, 0]):
                rob[f"audio_snr_{snr}"] = q(evaluate_split(model, store, te, device=dev,
                                                            audio=store.audio_snr[te, j], full=False, seed=2)[0])
        rng = np.random.default_rng(0)
        for pdrop in [0.25, 0.5, 0.75]:
            m = store.mask[te].copy()
            m &= rng.random(m.shape) >= pdrop
            empty = ~m.any(1)
            m[empty, 0] = True
            rob[f"frame_drop_{int(pdrop*100)}"] = q(evaluate_split(model, store, te, device=dev, mask=m, full=False, seed=2)[0])
        if store.face_blur is not None:
            rob["face_blur"] = q(evaluate_split(model, store, te, device=dev, face=store.face_blur[te], full=False, seed=2)[0])
        out["robustness"] = rob

    with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez(os.path.join(args.out, f"{tag}_scores.npz"),
             **{k.replace("/", "__"): np.stack([l, s]) for k, (l, s) in traw.items()},
             hv=hv, ha=ha, ids=store.df.iloc[te]["identity"].values.astype(str))
    print(f"[{tag}] TEST cross_video/random EER {tres['cross_video/random']['EER']:.4f} "
          f"AUC {tres['cross_video/random']['AUC']:.4f} | 1:2 v2f {tres['match1:2/v2f/random']:.3f}")


if __name__ == "__main__":
    main(get_args())
