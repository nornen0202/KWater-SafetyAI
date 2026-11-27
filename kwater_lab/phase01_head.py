"""Group-fold training and common/class-specific policies for limited multilabel alerts."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .metrics import CODES


class SpatialHead(nn.Module):
    def __init__(self, multiscale):
        super().__init__()
        self.multiscale = multiscale
        self.register_buffer("center", torch.zeros(128))
        self.register_buffer("scale", torch.ones(128))
        self.register_buffer("prior", torch.full((12,), 0.5))
        self.keys = nn.Linear(128, 48)
        self.values = nn.Linear(128, 48)
        self.query = nn.Parameter(torch.randn(12, 48) * 0.12)
        self.readout = nn.Parameter(torch.randn(12, 48) * 0.08)
        self.bias = nn.Parameter(torch.zeros(12))
        self.align = nn.Linear(48, 64)

    def forward(self, x):
        x = x[:, :5] if self.multiscale else x[:, :1]
        x = (x.float() - self.center) / self.scale
        k = self.keys(x)
        attention = torch.softmax(
            torch.einsum("bvtd,cd->bvct", k, self.query) / 48**0.5, dim=-1
        )
        h = torch.einsum("bvct,bvtd->bvcd", attention, self.values(x))
        logits = (h * self.readout).sum(-1) + self.bias
        output = logits[:, 0]
        if self.multiscale:
            detail = (torch.logsumexp(2 * logits[:, 1:], dim=1) - np.log(4)) / 2
            output = 0.5 * output + 0.5 * detail
        return output, h[:, 0]


def fit_head(x, y, evidence, train, multiscale, seed, epochs, *, prior_strength=0.5):
    torch.manual_seed(seed)
    training = torch.as_tensor(train, device=x.device, dtype=torch.long)
    net = SpatialHead(multiscale).to(x.device)
    feat = x[training, :5] if multiscale else x[training, :1]
    flat = feat.float().reshape(-1, 128)
    net.center.copy_(flat.mean(0))
    net.scale.copy_(flat.std(0, unbiased=False).clamp_min(0.015))
    del feat, flat
    net.prior.copy_(y[training].mean(0))
    prior = net.prior.clamp(0.002, 0.998)
    with torch.no_grad():
        net.bias.copy_(torch.logit(prior) - prior_strength * torch.log(prior))
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for chunk in np.array_split(
            rng.permutation(train), max(1, int(np.ceil(len(train) / 64)))
        ):
            idx = torch.as_tensor(chunk, device=x.device)
            gold = y[idx]
            batch = x[idx, :5].clone()
            if x.shape[1] == 6:
                replace = torch.as_tensor(
                    rng.random(len(chunk)) < 0.35, device=x.device
                )
                batch[replace, 0] = x[idx[replace], 5]
            z, h = net(batch)
            p = torch.sigmoid(z + prior_strength * torch.log(prior)).clamp(1e-6, 1 - 1e-6)
            neg = (1 - p + 0.025).clamp(max=1.0)
            smooth = 0.98 * gold + 0.01
            loss = (
                -smooth * torch.log(p)
                - (1 - smooth) * (1 - neg).square() * torch.log(neg)
            ).mean()
            target = evidence[idx]
            mask = target.square().sum(-1) > 0
            cosine = F.cosine_similarity(net.align(h), target, dim=-1)
            loss = loss + 0.025 * ((1 - cosine) * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net.eval()


@torch.inference_mode()
def logits(net, x):
    chunks = []
    for chunk in x.split(64):
        z, _ = net(chunk)
        z = torch.where(net.prior == 0, -100.0, z)
        z = torch.where(net.prior == 1, 100.0, z)
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks)


def probability(z, temp):
    return np.exp(
        -np.logaddexp(0.0, -np.asarray(z, dtype=np.float64) / np.asarray(temp))
    )


def decide(z, policy):
    p = probability(z, policy["temperatures"])
    thresholds = np.asarray(policy["thresholds"])
    ok = (p[:, :11] >= thresholds) & (p[:, 11, None] >= policy["scene_gate"])
    order = np.argsort(
        -np.where(ok, p[:, :11] - np.asarray(policy["offset"]), -np.inf),
        axis=1,
        kind="stable",
    )[:, :3]
    keep = np.zeros_like(ok)
    np.put_along_axis(keep, order, np.take_along_axis(ok, order, axis=1), axis=1)
    return keep


def fixed(threshold=0.20):
    return dict(
        thresholds=[threshold] * 11,
        temperatures=[1.0] * 12,
        scene_gate=0.0,
        offset=[0.0] * 11,
    )


def counts(y, pred):
    hits = (y * pred).sum(0)
    denom = y.sum(0) + pred.sum(0)
    f1 = np.divide(2 * hits, denom, out=np.zeros(11), where=denom != 0)
    macro = float(f1[denom > 0].mean()) if np.any(denom > 0) else 0.0
    return macro, f1


def calibrate(z, y, groups):
    """All inputs are inner-OOF predictions/labels from the outer training partition."""
    labels = np.column_stack([y, y.any(1)])
    grid = (
        0.08,
        0.12,
        0.16,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.70,
        0.80,
        0.90,
    )
    # The uncalibrated fixed policy remains a separate factorial control.
    temp = []
    mapped = np.array(groups)
    support = [len(set(mapped[labels[:, j] > 0])) for j in range(12)]
    for j in range(12):
        if support[j] < 2 or not np.any(labels[:, j] == 0):
            temp.append(1.0)
            continue

        def nll(t):
            q = z[:, j].astype(np.float64) / t
            return float((np.logaddexp(0, q) - labels[:, j] * q).mean())

        temp.append(min((1.0, 0.7, 1.4, 2.0), key=nll))

    def quality(policy):
        pred = decide(z, policy)
        macro, _ = counts(y, pred)
        return macro, -int(pred[~y.any(1)].any(1).sum())

    common = max((dict(fixed(t), temperatures=temp) for t in grid), key=quality)
    common_t = common["thresholds"][0]
    candidate = dict(common, thresholds=list(common["thresholds"]))
    for j in range(11):
        if support[j] < 2:
            continue
        n = int(y[:, j].sum())
        trials = []
        for t in grid:
            values = list(candidate["thresholds"])
            values[j] = (n * t + 20 * common_t) / (n + 20)
            trials.append(dict(candidate, thresholds=values))
        candidate = max([candidate, *trials], key=quality)
    _, f1 = counts(y, decide(z, candidate))
    utility = dict(candidate, offset=(f1 / 2).tolist())
    gated = max(
        (
            dict(utility, scene_gate=g)
            for g in (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
        ),
        key=quality,
    )
    return {
        "fixed": fixed(),
        "common": common,
        "class": candidate,
        "class_utility": utility,
        "class_utility_scene": gated,
    }


def code_sets(ids, predictions):
    return {
        sid: frozenset(c for c, v in zip(CODES, row) if v)
        for sid, row in zip(ids, predictions, strict=True)
    }
