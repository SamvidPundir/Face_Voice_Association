"""Synthetic cache with weak identity signal, to smoke-test train/eval end-to-end."""
import os
import sys

import h5py
import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "tests/fake"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
n_id, per, T = 60, 20, 24
W1, W2 = rng.normal(size=(512, 16)), rng.normal(size=(192, 16))
rows, F, M, A = [], [], [], []
for i in range(n_id):
    z = rng.normal(size=16); fc, vc = W1 @ z + 0.5 * rng.normal(size=512), W2 @ z + 0.5 * rng.normal(size=192)
    g = "m" if i % 2 else "f"
    for c in range(per):
        rows.append(dict(identity=f"id{i:03d}", video_id=f"v{i}_{c%5}", clip_id=c, gender=g, path="",
                         split="train" if i < 40 else "val" if i < 48 else "test"))
        F.append(fc + 2.0 * rng.normal(size=(T, 512)))
        M.append(rng.random(T) > 0.1)
        A.append(vc + 1.5 * rng.normal(size=(5, 192)))
pd.DataFrame(rows).to_csv(f"{out}/manifest.csv", index=False)
F, A = np.array(F, np.float16), np.array(A, np.float32)
with h5py.File(f"{out}/features.h5", "w") as f:
    f["face"] = F; f["face_mask"] = np.array(M)
    f["audio"] = A; f["audio_snr"] = A[:, :4]; f["face_blur"] = F
print("ok", F.shape, A.shape)
