"""Trainable heads on top of frozen ArcFace (face) and ECAPA (voice) embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out_dim = d

    def forward(self, x, mask):  # x [B,T,d], mask [B,T] True=valid
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.out_dim = d

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (w * x).sum(1)


class TemporalTransformer(nn.Module):
    def __init__(self, d, max_len=64, layers=2, heads=4, ffn=512, dropout=0.2):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1) + self.pos[:, : T + 1]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        return self.norm(self.enc(x, src_key_padding_mask=pad)[:, 0])


class Projection(nn.Module):
    def __init__(self, din, dout=128, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(hidden, dout))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FaceVoiceModel(nn.Module):
    def __init__(self, agg="transformer", face_dim=512, voice_dim=192, d=256, emb=128,
                 layers=2, heads=4, n_ids=0, max_len=64, drop=0.1, in_drop=0.0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(face_dim, d), nn.LayerNorm(d))
        if agg == "mean":
            self.agg = MeanPool(d)
        elif agg == "attn":
            self.agg = AttnPool(d)
        elif agg == "transformer":
            self.agg = TemporalTransformer(d, max_len=max_len, layers=layers, heads=heads)
        else:
            raise ValueError(agg)
        self.pv = Projection(d, emb, drop=drop)
        self.pa = Projection(voice_dim, emb, drop=drop)
        self.in_drop = nn.Dropout(in_drop)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.id_head = nn.Linear(emb, n_ids) if n_ids > 0 else None

    def embed_face(self, faces, mask):
        x = self.inp(self.in_drop(F.normalize(faces, dim=-1)))
        return self.pv(self.agg(x, mask))

    def embed_voice(self, voice):
        return self.pa(self.in_drop(F.normalize(voice, dim=-1)))

    def forward(self, faces, mask, voice):
        return self.embed_face(faces, mask), self.embed_voice(voice)
