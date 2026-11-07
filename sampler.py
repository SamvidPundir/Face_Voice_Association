"""In-memory cached-feature store and P×K batch construction.

Cache layout (data/features.h5, rows aligned with data/manifest.csv):
  face       [N,T,512] float16   ArcFace per frame
  face_mask  [N,T]     bool      valid (detected) frames
  audio      [N,5,192] float32   ECAPA: 0 clean, 1 SNR10, 2 SNR5, 3/4 random 2 s crops
  audio_snr  [N,4,192] float32   ECAPA at SNR 20/10/5/0 dB (test split only)
  face_blur  [N,T,512] float16   ArcFace on Gaussian-blurred frames (test split only)
"""
import os

import h5py
import numpy as np
import pandas as pd
import torch


class FeatureStore:
    def __init__(self, h5_path="data/features.h5", manifest="data/manifest.csv"):
        """h5_path / manifest may be comma-separated lists of aligned caches; they are concatenated."""
        dt = {"identity": object, "video_id": object, "clip_id": object, "gender": object, "split": object}
        dfs, parts = [], {k: [] for k in ["face", "face_mask", "audio", "audio_snr", "face_blur"]}
        for h, m in zip(h5_path.split(","), manifest.split(",")):
            if not os.path.exists(h):
                continue
            dfs.append(pd.read_csv(m, dtype=dt))
            with h5py.File(h, "r") as f:
                for k in parts:
                    parts[k].append(f[k][:])
        self.df = pd.concat(dfs, ignore_index=True)
        self.face = np.concatenate(parts["face"]).astype(np.float32)
        self.mask = np.concatenate(parts["face_mask"]).astype(bool)
        self.audio = np.concatenate(parts["audio"]).astype(np.float32)
        self.audio_snr = np.concatenate(parts["audio_snr"])
        self.face_blur = np.concatenate(parts["face_blur"]).astype(np.float32)
        ok = np.abs(self.audio[:, 0]).sum(1) > 0  # drop clips that failed to decode
        if not ok.all():
            self.df = self.df[ok].reset_index(drop=True)
            self.face, self.mask, self.audio = self.face[ok], self.mask[ok], self.audio[ok]
            if self.audio_snr is not None: self.audio_snr = self.audio_snr[ok]
            if self.face_blur is not None: self.face_blur = self.face_blur[ok]
        # clips with no detected face at all fall back to every frame being valid (centre crop)
        empty = ~self.mask.any(1)
        self.mask[empty] = True

    def split(self, name):
        return np.where(self.df["split"].values == name)[0]


class PKSampler:
    """P identities × K clips per batch. Face of a positive comes from a different video of the
    same identity with probability p_cross (falls back to same clip if none exists)."""

    def __init__(self, store, idx, P=64, K=2, p_cross=0.5, T=24, frame_drop=(0.1, 0.3),
                 seed=0, device="cuda"):
        self.s, self.idx = store, np.asarray(idx)
        self.P, self.K, self.p_cross, self.T = P, K, p_cross, T
        self.frame_drop, self.device = frame_drop, device
        self.rng = np.random.default_rng(seed)
        df = store.df.iloc[self.idx]
        self.ids = sorted(df["identity"].unique())
        self.id2lbl = {k: i for i, k in enumerate(self.ids)}
        self.by_id = {k: self.idx[(df["identity"] == k).values] for k in self.ids}
        self.video = store.df["video_id"].values

    def __len__(self):
        return max(1, len(self.idx) // (self.P * self.K))

    def _face(self, i):
        f, m = self.s.face[i].copy(), self.s.mask[i].copy()
        T = f.shape[0]
        L = self.rng.integers(max(4, T // 2), T + 1)  # random temporal crop
        st = self.rng.integers(0, T - L + 1)
        m2 = np.zeros_like(m)
        m2[st:st + L] = m[st:st + L]
        drop = self.rng.random(T) < self.rng.uniform(*self.frame_drop)
        m3 = m2 & ~drop
        if not m3.any():
            m3 = m2 if m2.any() else m
        return f[: self.T], m3[: self.T]

    def __iter__(self):
        for _ in range(len(self)):
            pids = self.rng.choice(self.ids, size=min(self.P, len(self.ids)), replace=False)
            F, M, A, L = [], [], [], []
            for p in pids:
                pool = self.by_id[p]
                for a in self.rng.choice(pool, size=self.K, replace=len(pool) < self.K):
                    fi = a
                    if self.rng.random() < self.p_cross:
                        other = pool[self.video[pool] != self.video[a]]
                        if len(other):
                            fi = self.rng.choice(other)
                    f, m = self._face(fi)
                    F.append(f); M.append(m)
                    A.append(self.s.audio[a, self.rng.integers(0, self.s.audio.shape[1])])
                    L.append(self.id2lbl[p])
            d = self.device
            yield (torch.from_numpy(np.stack(F)).to(d), torch.from_numpy(np.stack(M)).to(d),
                   torch.from_numpy(np.stack(A)).to(d), torch.tensor(L, device=d))
