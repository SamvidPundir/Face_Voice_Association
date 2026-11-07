"""Gradio demo: upload a face video and a voice clip (any audio/video file) -> P(same person).

  python demo/app.py --run transformer_s0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import av
import cv2
import gradio as gr
import numpy as np
import torch

from data.extract_features import SCRFD, align, center_crop, get_face_models, read_clip, vad
from models.heads import FaceVoiceModel

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="transformer_s0")
ap.add_argument("--res", default="results")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.load(open(os.path.join(a.res, f"{a.run}.json")))
args = meta["args"]
model = FaceVoiceModel(agg=args["agg"], layers=args["layers"], heads=args["heads"], max_len=args["T"],
                       drop=args.get("drop", 0.1), emb=args.get("emb", 128)).to(dev)
sd = torch.load(os.path.join(a.res, "ckpt", f"{a.run}.pt"), map_location=dev)
model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("id_head")}, strict=False)
model.eval()
pa, pb = meta["platt"]
thr = meta["val_threshold"]

det_s, rec_s = get_face_models()
det, rec_in = SCRFD(det_s), rec_s.get_inputs()[0].name
from speechbrain.inference.speaker import EncoderClassifier
try:
    from speechbrain.utils.fetching import LocalStrategy
    kw = {"local_strategy": LocalStrategy.COPY}
except Exception:
    kw = {}
ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir="models/weights/ecapa",
                                       run_opts={"device": dev}, **kw)


def read_audio(path, sr=16000):
    c = av.open(path)
    res = av.AudioResampler(format="flt", layout="mono", rate=sr)
    ch = [g.to_ndarray().reshape(-1) for f in c.decode(audio=0) for g in res.resample(f)]
    c.close()
    return np.concatenate(ch).astype(np.float32)


def face_seq(path, T):
    frames, _ = read_clip(path, fps=8, T=T)
    crops, mask, last, preview = [], [], None, None
    for fr in frames:
        _, kps = det(fr)
        if kps is not None:
            last = kps
        cr = align(fr, kps if kps is not None else last) if (kps is not None or last is not None) else center_crop(fr)
        crops.append(cr); mask.append(kps is not None)
        if preview is None and kps is not None:
            preview = cr
    x = ((np.stack(crops).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
    emb = rec_s.run(None, {rec_in: x})[0]
    m = np.array(mask) if any(mask) else np.ones(len(mask), bool)
    return emb, m, preview if preview is not None else crops[0]


def predict(video, voice):
    if video is None or voice is None:
        return "Upload both a face video and a voice clip.", None
    emb, m, preview = face_seq(video, args["T"])
    w = vad(read_audio(voice))[: 16000 * 8]
    with torch.no_grad():
        za = ecapa.encode_batch(torch.from_numpy(w)[None].to(dev)).squeeze(1)
        hv, ha = model(torch.from_numpy(emb)[None].float().to(dev), torch.from_numpy(m)[None].to(dev), za)
    s = float((hv * ha).sum())
    p = 1 / (1 + np.exp(-(pa * s + pb)))
    verdict = "SAME person" if s >= thr else "DIFFERENT people"
    txt = (f"### P(same person) = {p:.2f}\n\ncosine similarity {s:.3f} (threshold {thr:.3f}) → **{verdict}**\n\n"
           f"faces detected in {m.sum()}/{len(m)} sampled frames")
    return txt, preview


with gr.Blocks(title="Face–Voice Association") as ui:
    gr.Markdown("# Face–Voice Association\nTemporal-Transformer face encoder + ECAPA voice encoder in a shared space.")
    with gr.Row():
        v = gr.Video(label="Face video (.mp4)")
        au = gr.File(label="Voice clip (wav / m4a / mp4)", type="filepath")
    btn = gr.Button("Do they belong to the same person?", variant="primary")
    with gr.Row():
        out = gr.Markdown()
        img = gr.Image(label="aligned face (first detected frame)", height=160)
    btn.click(predict, [v, au], [out, img])

if __name__ == "__main__":
    ui.launch()
