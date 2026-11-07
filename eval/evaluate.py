"""Verification / matching / retrieval on a split, with optional perturbed inputs."""
import numpy as np
import torch

from eval.metrics import verification_metrics


@torch.no_grad()
def embed(model, face, mask, audio, device="cuda", bs=512, shuffle=False, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    HV, HA = [], []
    for s in range(0, len(face), bs):
        f = torch.from_numpy(np.ascontiguousarray(face[s:s + bs], dtype=np.float32)).to(device)
        m = torch.from_numpy(np.ascontiguousarray(mask[s:s + bs])).to(device)
        if shuffle:
            perm = torch.from_numpy(np.stack([rng.permutation(f.shape[1]) for _ in range(len(f))])).to(device)
            f = torch.gather(f, 1, perm.unsqueeze(-1).expand_as(f))
            m = torch.gather(m, 1, perm)
        a = torch.from_numpy(np.ascontiguousarray(audio[s:s + bs], dtype=np.float32)).to(device)
        hv, ha = model(f, m, a)
        HV.append(hv.cpu()); HA.append(ha.cpu())
    return torch.cat(HV).numpy(), torch.cat(HA).numpy()


def build_trials(df, n=10000, seed=0):
    """Rows are local indices into df. Returns dict of (voice_idx, face_idx, label) arrays."""
    rng = np.random.default_rng(seed)
    ids = df["identity"].values
    vids = df["video_id"].values
    gen = df["gender"].values
    N = len(df)
    by_id = {k: np.where(ids == k)[0] for k in np.unique(ids)}
    voices = rng.integers(0, N, n)

    def pos_cross(v):
        pool = by_id[ids[v]]
        o = pool[vids[pool] != vids[v]]
        return rng.choice(o) if len(o) else rng.choice(pool[pool != v]) if len(pool) > 1 else v

    def neg(v, gender_matched):
        while True:
            j = rng.integers(0, N)
            if ids[j] != ids[v] and (not gender_matched or gen[j] == gen[v]):
                return j

    out = {}
    for pos_type in ["same_clip", "cross_video"]:
        for neg_type in ["random", "gender"]:
            vv, ff, ll = [], [], []
            for v in voices[: n // 2]:
                vv.append(v); ff.append(v if pos_type == "same_clip" else pos_cross(v)); ll.append(1)
            for v in voices[n // 2:]:
                vv.append(v); ff.append(neg(v, neg_type == "gender")); ll.append(0)
            out[f"{pos_type}/{neg_type}"] = (np.array(vv), np.array(ff), np.array(ll))
    return out, by_id, pos_cross, neg


def matching_acc(hv, ha, df, N=2, gender_matched=False, n_queries=3000, seed=0, direction="v2f"):
    """1:N matching: query in one modality, 1 true (cross-video when possible) + N-1 impostors."""
    rng = np.random.default_rng(seed + N)
    ids, vids, gen = df["identity"].values, df["video_id"].values, df["gender"].values
    L = len(df)
    by_id = {k: np.where(ids == k)[0] for k in np.unique(ids)}
    correct = 0
    for _ in range(n_queries):
        q = rng.integers(0, L)
        pool = by_id[ids[q]]
        o = pool[vids[pool] != vids[q]]
        t = rng.choice(o) if len(o) else q
        cands = [t]
        while len(cands) < N:
            j = rng.integers(0, L)
            if ids[j] != ids[q] and (not gender_matched or gen[j] == gen[q]):
                cands.append(j)
        cands = np.array(cands)
        if direction == "v2f":
            s = hv[cands] @ ha[q]
        else:
            s = ha[cands] @ hv[q]
        correct += int(np.argmax(s) == 0)
    return correct / n_queries


def retrieval_topk(hv, ha, df, ks=(1, 5, 10), seed=0, direction="v2f"):
    """Gallery = one clip per identity in the other modality (from a different video than the
    query when possible). Identity-level Top-K."""
    rng = np.random.default_rng(seed)
    ids, vids = df["identity"].values, df["video_id"].values
    uid = np.unique(ids)
    hits = {k: 0 for k in ks}
    nq = 0
    for q in range(len(df)):
        gal = []
        for u in uid:
            pool = np.where(ids == u)[0]
            if u == ids[q]:
                o = pool[vids[pool] != vids[q]]
                pool = o if len(o) else pool
            gal.append(rng.choice(pool))
        gal = np.array(gal)
        s = (hv[gal] @ ha[q]) if direction == "v2f" else (ha[gal] @ hv[q])
        rank = np.argsort(-s)
        true_pos = int(np.where(uid[rank] == ids[q])[0][0])
        for k in ks:
            hits[k] += int(true_pos < k)
        nq += 1
    return {f"top{k}": hits[k] / nq for k in ks}


def evaluate_split(model, store, idx, device="cuda", face=None, mask=None, audio=None,
                   shuffle=False, threshold=None, full=True, seed=0):
    df = store.df.iloc[idx].reset_index(drop=True)
    face = store.face[idx] if face is None else face
    mask = store.mask[idx] if mask is None else mask
    audio = store.audio[idx, 0] if audio is None else audio
    hv, ha = embed(model, face, mask, audio, device=device, shuffle=shuffle, seed=seed)
    trials, *_ = build_trials(df, n=10000, seed=seed)
    res, raw = {}, {}
    for k, (v, f, l) in trials.items():
        s = np.sum(hv[f] * ha[v], 1)
        res[k] = verification_metrics(l, s, threshold)
        raw[k] = (l, s)
    if full:
        for d in ["v2f", "f2v"]:
            for gm in [False, True]:
                tag = "gender" if gm else "random"
                for N in [2, 10]:
                    res[f"match1:{N}/{d}/{tag}"] = matching_acc(hv, ha, df, N, gm, seed=seed, direction=d)
            res[f"retrieval/{d}"] = retrieval_topk(hv, ha, df, seed=seed, direction=d)
    return res, raw, (hv, ha)
