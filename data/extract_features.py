"""Run frozen encoders once per clip and cache embeddings to data/features.h5 (see sampler.py).

Face:  PyAV decode -> sample FPS frames from the middle window -> SCRFD (det_10g.onnx) 5 kps ->
       ArcFace-template similarity alignment 112x112 -> w600k_r50.onnx 512-d.
Audio: PyAV decode 16 kHz mono -> energy VAD -> SpeechBrain ECAPA 192-d.
"""
import argparse
import os
import sys
import urllib.request
import zipfile

import av
import cv2
import h5py
import numpy as np
import torch  # noqa: F401  (load CUDA DLLs before onnxruntime)
import onnxruntime as ort
import pandas as pd
import torch
from tqdm import tqdm

ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def get_face_models(mdir="models/weights"):
    os.makedirs(mdir, exist_ok=True)
    det, rec = os.path.join(mdir, "det_10g.onnx"), os.path.join(mdir, "w600k_r50.onnx")
    if not (os.path.exists(det) and os.path.exists(rec)):
        zp = os.path.join(mdir, "buffalo_l.zip")
        if not os.path.exists(zp):
            print("downloading buffalo_l ...", flush=True)
            urllib.request.urlretrieve(BUFFALO, zp)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith(("det_10g.onnx", "w600k_r50.onnx")):
                    with z.open(n) as s, open(os.path.join(mdir, os.path.basename(n)), "wb") as o:
                        o.write(s.read())
    ort.set_default_logger_severity(3)
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(det, providers=prov), ort.InferenceSession(rec, providers=prov)


class SCRFD:
    def __init__(self, sess, size=256):
        self.s, self.size = sess, size
        self.inp = sess.get_inputs()[0].name
        self.centers = {}
        for st in (8, 16, 32):
            g = size // st
            yy, xx = np.mgrid[:g, :g]
            c = (np.stack([xx, yy], -1).reshape(-1, 2) * st).astype(np.float32)
            self.centers[st] = np.repeat(c, 2, axis=0)  # 2 anchors per location

    def __call__(self, img_rgb, thr=0.5):
        """Returns (score, kps[5,2]) of the most central confident face, or (0, None)."""
        h, w = img_rgb.shape[:2]
        sc = self.size / max(h, w)
        canvas = np.zeros((self.size, self.size, 3), np.float32)
        canvas[: round(h * sc), : round(w * sc)] = cv2.resize(img_rgb, (round(w * sc), round(h * sc)))
        x = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.s.run(None, {self.inp: x})
        outs = [o[0] if o.ndim == 3 else o for o in outs]
        best, best_kps, best_d = 0.0, None, 1e9
        for k, st in enumerate((8, 16, 32)):
            score = outs[k].reshape(-1)
            kps = outs[k + 6].reshape(-1, 5, 2) * st + self.centers[st][:, None]
            for i in np.where(score > thr)[0]:
                kp = kps[i] / sc
                d = np.linalg.norm(kp.mean(0) - np.array([w / 2, h / 2]))
                if d < best_d:
                    best, best_kps, best_d = float(score[i]), kp, d
        return best, best_kps


def align(img_rgb, kps):
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARC_DST, method=cv2.LMEDS)
    return cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)


def center_crop(img_rgb):
    h, w = img_rgb.shape[:2]
    s = int(min(h, w) * 0.6)
    y0, x0 = (h - s) // 2 + int(0.05 * h), (w - s) // 2
    return cv2.resize(img_rgb[y0:y0 + s, x0:x0 + s], (112, 112))


def read_clip(path, fps=8, T=24, sr=16000):
    c = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    c = av.open(path)
    res = av.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []
    for f in c.decode(audio=0):
        for g in res.resample(f):
            chunks.append(g.to_ndarray().reshape(-1))
    for g in res.resample(None):
        chunks.append(g.to_ndarray().reshape(-1))
    c.close()
    wav = np.concatenate(chunks) if chunks else np.zeros(sr, np.float32)
    src_fps = 25.0
    step = src_fps / fps
    need = int(np.ceil(T * step))
    st = max(0, (len(frames) - need) // 2)
    idx = [min(len(frames) - 1, int(st + k * step)) for k in range(T)]
    return [frames[i] for i in idx], wav.astype(np.float32)


def vad(wav, sr=16000, win=0.03, rel=0.05):
    n = int(sr * win)
    if len(wav) < n * 4:
        return wav
    fr = wav[: len(wav) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    keep = rms > rel * np.percentile(rms, 95)
    out = fr[keep].reshape(-1)
    return out if len(out) > sr else wav


def add_noise(wav, snr, rng):
    noise = rng.normal(size=len(wav)).astype(np.float32)
    p_s, p_n = (wav ** 2).mean() + 1e-10, (noise ** 2).mean()
    return wav + noise * np.sqrt(p_s / (p_n * 10 ** (snr / 10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/features.h5")
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blur_thr", type=float, default=15.0)
    ap.add_argument("--det_stride", type=int, default=3)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    if a.limit:
        df = df.groupby("split").head(a.limit).reset_index(drop=True)
        a.manifest = a.manifest.replace(".csv", f"_limit{a.limit}.csv")
        a.out = a.out.replace(".h5", f"_limit{a.limit}.h5")
        df.to_csv(a.manifest, index=False)
    N, T = len(df), a.T
    det_s, rec_s = get_face_models()
    print("ORT providers:", rec_s.get_providers(), flush=True)
    det, rec_in = SCRFD(det_s), rec_s.get_inputs()[0].name

    from speechbrain.inference.speaker import EncoderClassifier
    try:
        from speechbrain.utils.fetching import LocalStrategy
        kw = {"local_strategy": LocalStrategy.COPY}
    except Exception:
        kw = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="models/weights/ecapa", run_opts={"device": dev}, **kw)

    def ecapa_embed(wavs):
        L = max(len(w) for w in wavs)
        x = torch.zeros(len(wavs), L)
        lens = torch.tensor([len(w) / L for w in wavs])
        for i, w in enumerate(wavs):
            x[i, : len(w)] = torch.from_numpy(w)
        with torch.no_grad():
            return ecapa.encode_batch(x.to(dev), lens.to(dev)).squeeze(1).cpu().numpy()

    def arc(crops):
        x = ((np.stack(crops).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
        return rec_s.run(None, {rec_in: x})[0]

    rng = np.random.default_rng(0)
    with h5py.File(a.out, "w") as h:
        d_face = h.create_dataset("face", (N, T, 512), np.float16)
        d_mask = h.create_dataset("face_mask", (N, T), bool)
        d_aud = h.create_dataset("audio", (N, 5, 192), np.float32)
        d_snr = h.create_dataset("audio_snr", (N, 4, 192), np.float32)
        d_blur = h.create_dataset("face_blur", (N, T, 512), np.float16)
        bad = 0
        for i, r in tqdm(df.iterrows(), total=N, file=sys.stdout, mininterval=10):
            try:
                frames, wav = read_clip(r.path, a.fps, T)
            except Exception as e:
                print("read fail", r.path, e); bad += 1
                continue
            # detect every det_stride-th frame (VoxCeleb2 clips are face tracks); reuse nearest kps
            dk = {j: det(frames[j])[1] for j in range(0, len(frames), a.det_stride)}
            crops, mask = [], []
            for j, fr in enumerate(frames):
                near = sorted(dk, key=lambda q: abs(q - j))
                kps = next((dk[q] for q in near if dk[q] is not None), None)
                if kps is not None:
                    cr = align(fr, kps)
                    ok = dk[near[0]] is not None and \
                        cv2.Laplacian(cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() > a.blur_thr
                else:
                    cr, ok = center_crop(fr), False
                crops.append(cr); mask.append(ok)
            d_face[i] = arc(crops).astype(np.float16)
            d_mask[i] = mask
            test = r.split == "test"
            if test:
                d_blur[i] = arc([cv2.GaussianBlur(c, (0, 0), 2.5) for c in crops]).astype(np.float16)

            w = vad(wav)[: 16000 * 8]
            wavs = [w, add_noise(w, 10, rng), add_noise(w, 5, rng)]
            for _ in range(2):
                if len(w) > 32000:
                    s0 = rng.integers(0, len(w) - 32000); wavs.append(w[s0:s0 + 32000])
                else:
                    wavs.append(w)
            if test:
                wavs += [add_noise(w, snr, rng) for snr in (20, 10, 5, 0)]
            e = ecapa_embed(wavs)
            d_aud[i] = e[:5]
            if test:
                d_snr[i] = e[5:9]
        print(f"done {N} clips, {bad} failed; mean face-mask coverage {d_mask[:].mean():.3f}")


if __name__ == "__main__":
    main()
