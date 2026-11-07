"""Add extra TRAINING identities from the VoxCeleb2 dev set (remote range-read of vox2_mp4_1.zip).
Writes data/manifest_dev.csv (split=train). Val/test identities are untouched.

  python data/prepare_dev.py --n_ids 600 --per_id 8
"""
import argparse
import os
import re
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--remote", default="datasets/ProgramComputer/voxceleb/vox2/vox2_mp4_1.zip")
p.add_argument("--meta", default="raw/vox2/vox2_meta.csv")
p.add_argument("--out", default="data")
p.add_argument("--n_ids", type=int, default=600)
p.add_argument("--per_id", type=int, default=8)
p.add_argument("--threads", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--exclude", default="")
p.add_argument("--out_manifest", default="manifest_dev.csv")
a = p.parse_args()

gender = {}
for line in open(a.meta, encoding="utf-8", errors="ignore"):
    parts = [x.strip() for x in re.split(r"[,\t]", line) if x.strip()]
    if parts and parts[0].startswith("id") and len(parts) >= 3:
        gender[parts[0]] = parts[2].lower()[:1]

fs = HfFileSystem()
open_zip = lambda: zipfile.ZipFile(fs.open(a.remote, "rb", block_size=1 << 20))
zf = open_zip()
clips = defaultdict(lambda: defaultdict(list))
for n in zf.namelist():
    m = re.search(r"(id\d+)/([^/]+)/(\d+)\.mp4$", n)
    if m:
        clips[m.group(1)][m.group(2)].append(n)
existing = set(pd.read_csv(os.path.join(a.out, "manifest.csv")).identity)
for m in filter(None, a.exclude.split(",")):
    existing |= set(pd.read_csv(m).identity)
rng = np.random.default_rng(a.seed)
cand = [i for i in sorted(clips) if i not in existing and len(clips[i]) >= 2]
ids = list(rng.choice(cand, size=min(a.n_ids, len(cand)), replace=False))
print(f"{len(cand)} candidate dev ids with >=2 videos; taking {len(ids)}")

rows, jobs = [], []
for i in ids:
    vids = {v: list(rng.permutation(c)) for v, c in clips[i].items()}
    order = list(rng.permutation(list(vids)))
    chosen = []
    while len(chosen) < a.per_id and any(vids.values()):
        for v in order:
            if vids[v] and len(chosen) < a.per_id:
                chosen.append((v, vids[v].pop()))
    for v, n in chosen:
        dst = os.path.join(a.out, "clips_dev", i, v, os.path.basename(n))
        if not os.path.exists(dst):
            jobs.append((n, dst))
        rows.append(dict(identity=i, video_id=v, clip_id=os.path.splitext(os.path.basename(n))[0],
                         gender=gender.get(i, "u"), path=dst.replace("\\", "/"), split="train"))
import struct, zlib
info = {zi.filename: zi for zi in zf.infolist()}
import requests
from huggingface_hub import get_token
_URL = "https://huggingface.co/" + a.remote.replace("datasets/ProgramComputer/voxceleb/", "datasets/ProgramComputer/voxceleb/resolve/main/")
_loc = {"u": None}


def cdn_url(refresh=False):
    if refresh or _loc["u"] is None:
        r = requests.get(_URL, headers={"Authorization": f"Bearer {get_token()}", "Range": "bytes=0-1"},
                         allow_redirects=False, timeout=60)
        _loc["u"] = r.headers["Location"]
    return _loc["u"]


def fetch(job):
    """One HTTP range request per clip using offsets from the central directory parsed once."""
    n, dst = job
    zi = info[n]
    for _ in range(6):
        try:
            # local header (30 B) + name + extra; over-read 1 KB to cover variable extra field
            end = zi.header_offset + 30 + 1024 + zi.compress_size
            r = requests.get(cdn_url(), headers={"Range": f"bytes={zi.header_offset}-{end}"}, timeout=120)
            if r.status_code in (401, 403):
                cdn_url(refresh=True); continue
            blob = r.content
            nl, xl = struct.unpack("<HH", blob[26:30])
            raw = blob[30 + nl + xl: 30 + nl + xl + zi.compress_size]
            data = raw if zi.compress_type == 0 else zlib.decompress(raw, -15)
            assert len(data) == zi.file_size
            break
        except Exception:
            continue
    else:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "wb") as o:
        o.write(data)
    os.replace(dst + ".part", dst)
    return len(data)


print(f"fetching {len(jobs)} clips", flush=True)
with ThreadPoolExecutor(a.threads) as ex:
    nb = sum(tqdm(ex.map(fetch, jobs), total=len(jobs), mininterval=15))
print(f"fetched {nb/1e9:.2f} GB")
df = pd.DataFrame([r for r in rows if os.path.exists(r["path"])])
df.to_csv(os.path.join(a.out, a.out_manifest), index=False)
print(len(df), "clips,", df.identity.nunique(), "ids,", df.drop_duplicates("identity").gender.value_counts().to_dict())
