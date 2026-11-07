# Temporal Transformer-Based Cross-Modal Face–Voice Association

Samvid Pundir (230150021) · Piyush Jain (230121040) · Srijan Kumar (230150025)
Indian Institute of Technology Guwahati

We learn a shared embedding space in which a talking-face video and a voice clip of the same person lie close together,
and evaluate it on VoxCeleb2 identities never seen in training.

```
face video ─► SCRFD align ─► ArcFace (frozen) ─► [T×512] ─► Temporal Transformer ─► P_v ─┐
                                                                                          ├─► cosine ─► P(same person)
speech ────► VAD ─────────► ECAPA-TDNN (frozen) ─► [192] ──────────────────────────► P_a ─┘
```

- **Frozen encoders:** SCRFD + ArcFace R50 (InsightFace `buffalo_l`, run through ONNX Runtime) and SpeechBrain ECAPA-TDNN.
  They run once per clip and their outputs are cached, so a training run takes about one minute on a 4 GB GPU.
- **Trained parts:** a temporal aggregator (mean pooling, attention pooling, or a 2-layer Transformer with a [CLS] token)
  and two projection heads into a 128-dimensional shared space.
- **Loss:** symmetric InfoNCE with identity masking (other clips of the same person are not treated as negatives),
  plus an auxiliary identity cross-entropy. Half of the positive pairs take the face and the voice from different videos.

## Results (25 unseen test identities, mean over 3 seeds)

| Model | EER ↓ | AUC ↑ | 1:2 voice→face ↑ | Top-5 of 25 ↑ |
|---|---|---|---|---|
| Mean pooling | 36.9% | 68.3% | 69.5% | 42.4% |
| Attention pooling | 40.4% | 63.7% | 64.9% | 37.4% |
| Temporal Transformer (T=24) | 37.0% | 67.9% | 68.9% | 42.1% |
| **Temporal Transformer (T=8)** | **34.6%** | **70.9%** | **71.9%** | **47.0%** |

Chance is 50% for EER, AUC and 1:2 matching, and 20% for Top-5. `python run_all.py --aggregate` regenerates the full
table, including the ablations and robustness tests, from the result files.

## Setup

Tested on Windows 11 with Python 3.12 and an NVIDIA RTX 3050 (4 GB). Linux works the same; use `venv312/bin/python`.

```bash
py -3.12 -m venv venv312
venv312\Scripts\pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
venv312\Scripts\pip install -r requirements.txt
# speechbrain may replace torch with a CPU build; restore the CUDA build:
venv312\Scripts\pip install --force-reinstall --no-deps torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

`onnxruntime-gpu` is pinned to 1.22 because newer releases require CUDA 13, while this torch build ships CUDA 12.
No system ffmpeg is needed: PyAV decodes the video and audio.

## Data access

The VoxCeleb2 data is read from the Hugging Face mirror `ProgramComputer/voxceleb`, which is gated:

1. Open https://huggingface.co/datasets/ProgramComputer/voxceleb and accept the terms.
2. Run `venv312\Scripts\huggingface-cli login` and paste a read token.

Only the selected clips are range-read from the remote zip archives (about 1.7 GB in total), so the full dataset
(hundreds of GB) is never downloaded.

## Running the pipeline

```bash
# 1. Validation/test identities: VoxCeleb2 test partition, 40 clips per identity, identity-disjoint gender-stratified split
venv312\Scripts\python data/prepare.py --download
# 2. Extra training identities: 600 VoxCeleb2-dev identities, 4 clips each
venv312\Scripts\python data/prepare_dev.py --n_ids 600 --per_id 4
# 3. Run the frozen encoders once and cache the features (about 45 min on an RTX 3050)
venv312\Scripts\python data/extract_features.py
venv312\Scripts\python data/extract_features.py --manifest data/manifest_dev.csv --out data/features_dev.h5
# 4. Train and evaluate every configuration: 8 configs x 3 seeds -> results/*.json and results/table.md
venv312\Scripts\python run_all.py
# 5. Figures (ROC, robustness, training curves, t-SNE) -> report/figs/
venv312\Scripts\python eval/plots.py
# 6. Interactive demo: upload a face video and a voice clip, get P(same person)
venv312\Scripts\python demo/app.py --run transformer_T8_s0
```

Train a single configuration:

```bash
venv312\Scripts\python train.py --name transformer_T8 --agg transformer --T 8 --seed 0
#   --agg {mean,attn,transformer}   --T frames per clip   --id_w identity-loss weight
#   --p_cross fraction of cross-video positives   --no_robust skip the robustness tests
```

Smoke test without any data:

```bash
venv312\Scripts\python tests/make_fake.py tests/fake
venv312\Scripts\python train.py --epochs 5 --h5 tests/fake/features.h5 --manifest tests/fake/manifest.csv --out tests/fake/results
```

## Evaluation protocol

- **Verification:** 10,000 balanced trials. Positives are cross-video (the face and the voice come from different
  videos). Negatives are random or gender-matched. We report EER, AUC, and accuracy/F1 at the threshold fixed on
  validation; Platt scaling gives the calibrated P(same).
- **Matching:** 1:2 and 1:10, both voice→face and face→voice, with random or same-gender impostors.
- **Retrieval:** Top-1/5/10 over a gallery with one clip per test identity.
- **Robustness:** audio SNR 20/10/5/0 dB, 25/50/75% of frames dropped, Gaussian face blur, and shuffled frame order.

## Repository layout

| Path | Purpose |
|---|---|
| `data/prepare.py` | VoxCeleb2-test subset: video-diverse clip selection, identity-disjoint split |
| `data/prepare_dev.py` | extra training identities from VoxCeleb2-dev, fetched with HTTP range requests |
| `data/extract_features.py` | SCRFD alignment + ArcFace, VAD + ECAPA-TDNN, augmented and perturbed views → HDF5 |
| `models/heads.py` | mean, attention and Transformer aggregators; projection heads |
| `losses.py` | identity-masked symmetric InfoNCE; identity cross-entropy |
| `sampler.py` | in-memory feature store; P×K batch sampler with cross-video positives |
| `train.py` | train one configuration, calibrate on validation, evaluate and run robustness tests on test |
| `eval/metrics.py` | EER, AUC, P/R/F1, Platt scaling, bootstrap confidence intervals |
| `eval/evaluate.py` | verification trials, 1:N matching, Top-K retrieval |
| `eval/plots.py` | ROC, robustness, training-curve and t-SNE figures |
| `run_all.py` | ablation grid and the aggregated results table |
| `demo/app.py` | Gradio demo |
| `tests/make_fake.py` | synthetic feature cache for smoke tests |

## Acknowledgements

VoxCeleb (Nagrani et al., 2017; Chung et al., 2018), InsightFace (SCRFD, ArcFace) and SpeechBrain (ECAPA-TDNN).
The dataset is used for research purposes under its terms. No data or model weights are redistributed in this repository.
