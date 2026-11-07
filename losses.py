import torch
import torch.nn.functional as F


def masked_infonce(hv, ha, ids, logit_scale):
    """Symmetric CLIP-style InfoNCE. Off-diagonal pairs sharing an identity are removed
    from the denominator so they are not treated as negatives."""
    scale = logit_scale.exp().clamp(max=100)
    logits = scale * hv @ ha.t()
    same = ids.unsqueeze(0) == ids.unsqueeze(1)
    eye = torch.eye(len(ids), dtype=torch.bool, device=hv.device)
    logits = logits.masked_fill(same & ~eye, -1e4)
    tgt = torch.arange(len(ids), device=hv.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def identity_ce(model, hv, ha, ids):
    if model.id_head is None:
        return hv.new_zeros(())
    return 0.5 * (F.cross_entropy(model.id_head(hv), ids) + F.cross_entropy(model.id_head(ha), ids))
