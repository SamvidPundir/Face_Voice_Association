"""Download VoxCeleb2 test mp4 + meta, pick a capped, video-diverse subset of clips per identity,
make identity-disjoint train/val/test splits, extract those clips, write data/manifest.csv.

  python data/prepare.py --download            # needs HF access to ProgramComputer/voxceleb
  python data/prepare.py --zip path/to/vox2_test_mp4.zip --meta path/to/vox2_meta.csv
"""
import argparse
import os
import re
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--download", action="store_true")
p.add_argument("--zip", default="raw/vox2/vox2_test_mp4.zip")
p.add_argument("--meta", default="raw/vox2/vox2_meta.csv")
p.add_argument("--out", default="data")
p.add_argument("--per_id", type=int, default=40)
p.add_argument("--n_val", type=int, default=15)
p.add_argument("--n_test", type=int, default=25)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

REMOTE = "datasets/ProgramComputer/voxceleb/vox2/vox2_test_mp4.zip"
if a.download:  # meta only; clips are range-read from the remote zip (no 8.9 GB download)
    from huggingface_hub import HfFileSystem, hf_hub_download
    hf_hub_download("ProgramComputer/voxceleb", "vox2/vox2_meta.csv", repo_type="dataset", local_dir="raw")
    _fs = HfFileSystem()
    open_zip = lambda: zipfile.ZipFile(_fs.open(REMOTE, "rb", block_size=1 << 20))
else:
    open_zip = lambda: zipfile.ZipFile(a.zip)

# gender from meta (columns: VoxCeleb2 ID, VGGFace2 ID, Gender, Set; separators vary)
gender = {}
with open(a.meta, encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        parts = [x.strip() for x in re.split(r"[,\t]", line) if x.strip()]
        if parts and parts[0].startswith("id") and len(parts) >= 3:
            gender[parts[0]] = parts[2].lower()[:1]

zf = open_zip()
clips = defaultdict(lambda: defaultdict(list))
for n in zf.namelist():
    m = re.search(r"(id\d+)/([^/]+)/(\d+)\.mp4$", n)
    if m:
        clips[m.group(1)][m.group(2)].append(n)
ids = sorted(clips)
print(f"{len(ids)} identities, {sum(len(c) for v in clips.values() for c in v.values())} clips in zip")

rng = np.random.default_rng(a.seed)
# stratify split by gender so every split has both
by_g = defaultdict(list)
for i in ids:
    by_g[gender.get(i, "u")].append(i)
split = {}
for g, lst in by_g.items():
    lst = list(rng.permutation(lst))
    nt = round(a.n_test * len(lst) / len(ids)); nv = round(a.n_val * len(lst) / len(ids))
    for i in lst[:nt]: split[i] = "test"
    for i in lst[nt:nt + nv]: split[i] = "val"
    for i in lst[nt + nv:]: split[i] = "train"

rows, jobs = [], []
for i in ids:
    vids = {v: list(rng.permutation(c)) for v, c in clips[i].items()}
    order = list(rng.permutation(list(vids)))
    chosen = []
    while len(chosen) < a.per_id and any(vids.values()):  # round-robin over videos
        for v in order:
            if vids[v] and len(chosen) < a.per_id:
                chosen.append((v, vids[v].pop()))
    for v, n in chosen:
        dst = os.path.join(a.out, "clips", i, v, os.path.basename(n))
        if not os.path.exists(dst):
            jobs.append((n, dst))
        rows.append(dict(identity=i, video_id=v, clip_id=os.path.splitext(os.path.basename(n))[0],
                         gender=gender.get(i, "u"), path=dst.replace("\\", "/"), split=split[i]))

import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
_tl = threading.local()


def fetch(job):
    n, dst = job
    if not hasattr(_tl, "z"):
        _tl.z = open_zip()
    for attempt in range(3):
        try:
            data = _tl.z.read(n)
            break
        except Exception:
            _tl.z = open_zip()
    else:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "wb") as o:
        o.write(data)
    os.replace(dst + ".part", dst)
    return len(data)


print(f"fetching {len(jobs)} clips", flush=True)
with ThreadPoolExecutor(16) as ex:
    nbytes = sum(tqdm(ex.map(fetch, jobs), total=len(jobs), mininterval=15))
print(f"fetched {nbytes/1e9:.2f} GB")
rows = [r for r in rows if os.path.exists(r["path"])]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "manifest.csv"), index=False)
print(df.groupby("split").agg(ids=("identity", "nunique"), clips=("path", "count"),
                              videos=("video_id", "nunique")))
print(df.drop_duplicates("identity").groupby(["split", "gender"]).size())
