"""
MAST-Fuse / MuSTIPest-V3
INTERNAL COMPONENT + CAPACITY-CONTROLLED ABLATION
==================================================

Purpose
-------
1) Internal MAST-Fuse component ablation:
   - Full MAST-Fuse
   - w/o Cross-Modal Attention
   - w/o Modality Gate
   - w/o Temporal Attention
   - w/o Cross-Modal Attention + Gate
   - w/o Cross-Modal Attention + Temporal Attention
   - w/o Gate + Temporal Attention
   - w/o Cross-Modal Attention + Gate + Temporal Attention

2) Capacity-controlled comparison:
   Compare MAST-Fuse against simpler fusion mechanisms while adding a
   generic trainable capacity adapter to the simpler model so that its
   TOTAL parameter count is as close as possible to the full MAST-Fuse.

The script deliberately does NOT fabricate or reuse previous results.
It trains all requested configurations on the current local dataset.

IMPORTANT
---------
The script imports the user's CURRENT local:
    dataset.py
    model.py
    fusion.py
    encoders.py
    losses.py

Therefore it must be placed in the project root:
    C:\\Users\\dipan\\MAST-Fuse\\

Training protocol:
    seeds       = [42,123,456,789,1011]
    batch       = 32
    epochs      = 150
    lr          = 1e-4
    weight decay= 1e-5
    clip        = 1.0
    dropout     = 0.10
    fusion dim  = 128
    heads       = 4
    FF dim      = 256
    scheduler   = ReduceLROnPlateau(factor=.5, patience=7, min_lr=1e-7)
    early stop  = 20 epochs, min_delta=1e-6

Outputs:
    mastfuse_internal_capacity_results/
        aggregate.csv
        aggregate.tex
        paired_significance.csv
        capacity_matching.csv
        <experiment>/
            seed_42/
            seed_123/
            seed_456/
            seed_789/
            seed_1011/
            summary.json

Use:
    python mastfuse_internal_capacity_ablation.py --smoke-test
    python mastfuse_internal_capacity_ablation.py --group internal
    python mastfuse_internal_capacity_ablation.py --group capacity
    python mastfuse_internal_capacity_ablation.py --group all
    python mastfuse_internal_capacity_ablation.py --group all --resume

A first smoke test is strongly recommended.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel, wilcoxon

from dataset import load_all_data, MuSTIPestDataset, create_dataloader
from model import MASTFuse, RegressionHead
from encoders import MultimodalEncoders
from fusion_ablation_ready import CrossModalFusion
from losses import StableRegressionLoss


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "mastfuse_internal_capacity_results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]

BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

PATIENCE = 20
MIN_DELTA = 1e-6

LR_PATIENCE = 7
LR_FACTOR = 0.5
MIN_LR = 1e-7

FUSION_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.10
SEQ = 21
NUM_WORKERS = 0

TARGET_PARAMS = 2_781_027
PARAM_TOLERANCE = 0.02       # 2 percent
CAPACITY_HIDDEN_MIN = 1
CAPACITY_HIDDEN_MAX = 8192

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODS = ["weather", "spectral", "dynamic_soil", "static_soil"]
DIMS = {
    "weather": 17,
    "spectral": 3,
    "dynamic_soil": 4,
    "static_soil": 28,
}


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

INTERNAL = {
    "full_model": dict(
        name="Full MAST-Fuse",
        use_cross_attention=True,
        use_gating=True,
        #use_residual=True,
        use_temporal_attention=True,
    ),
    "no_cross": dict(
        name="w/o Cross-Modal Attention",
        use_cross_attention=False,
        use_gating=True,
        #use_residual=True,
        use_temporal_attention=True,
    ),
    "no_gate": dict(
        name="w/o Modality Gate",
        use_cross_attention=True,
        use_gating=False,
        #use_residual=True,
        use_temporal_attention=True,
    ),
    "no_temporal": dict(
        name="w/o Temporal Attention",
        use_cross_attention=True,
        use_gating=True,
        #use_residual=True,
        use_temporal_attention=False,
    ),
    "no_cross_gate": dict(
        name="w/o Cross-Modal Attention + Gate",
        use_cross_attention=False,
        use_gating=False,
        #use_residual=True,
        use_temporal_attention=True,
    ),
    "no_cross_temporal": dict(
        name="w/o Cross-Modal Attention + Temporal Attention",
        use_cross_attention=False,
        use_gating=True,
        #use_residual=True,
        use_temporal_attention=False,
    ),
    "no_gate_temporal": dict(
        name="w/o Modality Gate + Temporal Attention",
        use_cross_attention=True,
        use_gating=False,
        #use_residual=True,
        use_temporal_attention=False,
    ),
    "no_cross_gate_temporal": dict(
        name="w/o Cross-Modal Attention + Gate + Temporal Attention",
        use_cross_attention=False,
        use_gating=False,
        #use_residual=True,
        use_temporal_attention=False,
    ),
}


# ============================================================
# GENERIC FUSION BASELINES FOR CAPACITY CONTROL
# ============================================================

class EarlyFusion(nn.Module):
    """Concatenate modality sequences, project to fusion dimension,
    then use temporal attention pooling."""

    def __init__(self, dim=128, nmods=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * nmods, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = TemporalPool(dim, dropout)

    def forward(self, m, return_attention=False):
        z = self.proj(torch.cat(list(m.values()), dim=-1))
        v, tw = self.pool(z)
        mw = torch.full(
            (z.size(0), len(m)),
            1.0 / len(m),
            device=z.device,
            dtype=z.dtype,
        )
        info = {
            "modality_weights": mw,
            "temporal_weights": tw,
            "modality_names": list(m.keys()),
        }
        return (v, z, info) if return_attention else (v, z)


class FeatureFusion(nn.Module):
    """Element-wise mean of modality sequences."""

    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.pool = MeanPool()

    def forward(self, m, return_attention=False):
        z = torch.stack(list(m.values()), dim=0).mean(dim=0)
        v, tw = self.pool(z)
        mw = torch.full(
            (z.size(0), len(m)),
            1.0 / len(m),
            device=z.device,
            dtype=z.dtype,
        )
        info = {
            "modality_weights": mw,
            "temporal_weights": tw,
            "modality_names": list(m.keys()),
        }
        return (v, z, info) if return_attention else (v, z)


class SimpleCrossModalAttention(nn.Module):
    """One shared multi-head attention block over modality tokens
    at every temporal step."""

    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.pool = TemporalPool(dim, dropout)

    def forward(self, m, return_attention=False):
        names = list(m.keys())
        # (B,T,M,D) -> (B*T,M,D)
        x = torch.stack([m[n] for n in names], dim=2)
        B, T, M, D = x.shape
        x = x.reshape(B * T, M, D)
        y, _ = self.mha(x, x, x, need_weights=False)
        y = self.norm(x + self.drop(y))
        z = y.mean(dim=1).reshape(B, T, D)
        v, tw = self.pool(z)
        mw = torch.full(
            (B, M), 1.0 / M, device=z.device, dtype=z.dtype
        )
        info = {
            "modality_weights": mw,
            "temporal_weights": tw,
            "modality_names": names,
        }
        return (v, z, info) if return_attention else (v, z)


class LateFusion(nn.Module):
    """Temporal-pool each modality independently, then average vectors."""

    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.pool = TemporalPool(dim, dropout)

    def forward(self, m, return_attention=False):
        vectors = []
        tws = []
        for x in m.values():
            v, tw = self.pool(x)
            vectors.append(v)
            tws.append(tw)
        v = torch.stack(vectors, dim=0).mean(dim=0)
        z = torch.stack(list(m.values()), dim=0).mean(dim=0)
        tw = torch.stack(tws, dim=0).mean(dim=0)
        mw = torch.full(
            (z.size(0), len(m)),
            1.0 / len(m),
            device=z.device,
            dtype=z.dtype,
        )
        info = {
            "modality_weights": mw,
            "temporal_weights": tw,
            "modality_names": list(m.keys()),
        }
        return (v, z, info) if return_attention else (v, z)


class HybridFusion(nn.Module):
    """Shared cross-modal MHA + learned modality gate + temporal pooling."""

    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.gate = ModalityGate(dim, dropout)
        self.pool = TemporalPool(dim, dropout)

    def forward(self, m, return_attention=False):
        names = list(m.keys())
        x = torch.stack([m[n] for n in names], dim=2)
        B, T, M, D = x.shape
        q = x.reshape(B * T, M, D)
        y, _ = self.att(q, q, q, need_weights=False)
        y = self.norm(q + self.drop(y))
        y = y.reshape(B, T, M, D)
        per_mod = {n: y[:, :, i, :] for i, n in enumerate(names)}
        z, mw = self.gate(per_mod)
        v, tw = self.pool(z)
        info = {
            "modality_weights": mw,
            "temporal_weights": tw,
            "modality_names": names,
        }
        return (v, z, info) if return_attention else (v, z)


class TemporalPool(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        h = max(dim // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(dim, h),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        w = F.softmax(self.score(x).squeeze(-1), dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1), w


class MeanPool(nn.Module):
    def forward(self, x):
        B, T, _ = x.shape
        w = torch.full(
            (B, T), 1.0 / T, device=x.device, dtype=x.dtype
        )
        return x.mean(dim=1), w


class ModalityGate(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        h = max(dim // 2, 1)
        self.score = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, modalities):
        names = list(modalities.keys())
        scores = torch.cat(
            [self.score(modalities[n].mean(dim=1)) for n in names],
            dim=1,
        )
        w = F.softmax(scores, dim=1)
        z = sum(
            modalities[n] * w[:, i].view(-1, 1, 1)
            for i, n in enumerate(names)
        )
        return z, w


CAPACITY_BASELINES = {
    "early_capacity": ("Early Fusion", "early"),
    "feature_capacity": ("Feature Fusion", "feature"),
    "cross_capacity": ("Cross-Modal Attention", "cross"),
    "late_capacity": ("Late Fusion", "late"),
    "hybrid_capacity": ("Hybrid Attention", "hybrid"),
}


# ============================================================
# CAPACITY ADAPTER
# ============================================================

class CapacityAdapter(nn.Module):
    """
    Generic width-controlled residual MLP.

    It is intentionally the SAME generic module for all simple
    fusion baselines. Its hidden width is selected automatically
    to make the total model parameter count close to TARGET_PARAMS.
    """

    def __init__(self, dim=128, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.net(x)


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_data():
    data = load_all_data()
    if not isinstance(data, (tuple, list)) or len(data) != 3:
        raise ValueError("load_all_data() must return train, val, test.")
    return data


def make_loaders(data):
    train, val, test = data

    def make(x, shuffle):
        ds = MuSTIPestDataset(
            x,
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
        )
        return create_dataloader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )

    return make(train, True), make(val, False), make(test, False)


def move_batch(batch):
    return {
        k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def batch_inputs(batch):
    return {
        k: batch[k]
        for k in ["weather", "spectral", "dynamic_soil", "static_soil"]
        if k in batch
    }


def check_batch(batch):
    for name, dim in DIMS.items():
        if name not in batch:
            raise RuntimeError(f"Missing modality: {name}")
        x = batch[name]
        if not torch.isfinite(x).all():
            raise RuntimeError(f"{name} contains NaN/Inf.")
        if x.ndim != 3 or x.shape[1] != SEQ or x.shape[2] != dim:
            raise RuntimeError(
                f"{name}: expected (B,{SEQ},{dim}), got {tuple(x.shape)}"
            )
    if "target" not in batch or not torch.isfinite(batch["target"]).all():
        raise RuntimeError("Invalid target.")


def metrics(y, p):
    e = p - y
    mse = float(np.mean(e ** 2))
    den = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(e ** 2) / den) if den > 0 else np.nan
    pearson = (
        float(np.corrcoef(y, p)[0, 1])
        if np.std(y) > 0 and np.std(p) > 0
        else np.nan
    )
    return {
        "MAE": float(np.mean(np.abs(e))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2,
        "Pearson": pearson,
    }


# ============================================================
# MODEL WRAPPER
# ============================================================

class ExperimentModel(nn.Module):
    """
    Uses the current local MASTFuse trunk.

    For internal ablations, it replaces the current fusion module
    with CrossModalFusion configured through its documented flags.

    For capacity comparisons, it keeps the same MASTFuse encoder/head
    dimensions and substitutes a simpler fusion implementation.
    """

    def __init__(
        self,
        mode: str,
        cfg: Optional[dict] = None,
        simple_name: Optional[str] = None,
        adapter_hidden: Optional[int] = None,
    ):
        super().__init__()

        self.mode = mode
        self.cfg = cfg or {}

        # Construct the authoritative current model.
        self.base = MASTFuse(
            weather_dim=17,
            spectral_dim=3,
            dynamic_soil_dim=4,
            static_soil_dim=28,
            sequence_length=SEQ,
            fusion_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
            fusion_type="cross_attention",
        )

        if mode == "internal":
            # The current fusion.py must expose the four ablation flags.
            sig = inspect.signature(CrossModalFusion.__init__)
            required = {
                "use_cross_attention",
                "use_gating",
                #"use_residual",
                "use_temporal_attention",
            }
            missing = required - set(sig.parameters)
            if missing:
                raise RuntimeError(
                    "Your current fusion.py does not expose the required "
                    f"ablation flags: {sorted(missing)}"
                )

            self.base.fusion = CrossModalFusion(
                fusion_dim=FUSION_DIM,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                dropout=DROPOUT,
                use_weather=True,
                use_spectral=True,
                use_dynamic_soil=True,
                use_static_soil=True,
                use_cross_attention=cfg["use_cross_attention"],
                use_gating=cfg["use_gating"],
                #use_residual=cfg["use_residual"],
                use_temporal_attention=cfg["use_temporal_attention"],
            )

        elif mode == "capacity":
            if simple_name is None:
                raise ValueError("simple_name required.")

            if simple_name == "early":
                self.base.fusion = EarlyFusion(FUSION_DIM, 4, DROPOUT)
            elif simple_name == "feature":
                self.base.fusion = FeatureFusion(FUSION_DIM, DROPOUT)
            elif simple_name == "cross":
                self.base.fusion = SimpleCrossModalAttention(
                    FUSION_DIM, NUM_HEADS, DROPOUT
                )
            elif simple_name == "late":
                self.base.fusion = LateFusion(FUSION_DIM, DROPOUT)
            elif simple_name == "hybrid":
                self.base.fusion = HybridFusion(
                    FUSION_DIM, NUM_HEADS, DROPOUT
                )
            else:
                raise ValueError(simple_name)

            if adapter_hidden is not None and adapter_hidden > 0:
                self.capacity_adapter = CapacityAdapter(
                    FUSION_DIM, adapter_hidden, DROPOUT
                )
            else:
                self.capacity_adapter = nn.Identity()

        else:
            raise ValueError(mode)

    def forward(self, **kwargs):
        if self.mode == "internal":
            return self.base(**kwargs)

        # For simple fusion, manually execute the same current trunk.
        # This preserves the local MASTFuse encoder and regression head.
        # If the local MASTFuse has additional temporal processing before
        # fusion, this remains part of the base model's encoder/trunk only
        # when it is represented in self.base.encoders. We explicitly log
        # the local model structure before training.
        embeddings = self.base.encoders(
            weather=kwargs["weather"],
            spectral=kwargs["spectral"],
            dynamic_soil=kwargs["dynamic_soil"],
            static_soil=kwargs["static_soil"],
        )

        vec, seq = self.base.fusion(embeddings)
        vec = self.capacity_adapter(vec)
        return self.base.regression_head(vec)

    def forward_aux(self, **kwargs):
        if self.mode == "internal":
            return self.base(**kwargs, return_aux=True)

        embeddings = self.base.encoders(
            weather=kwargs["weather"],
            spectral=kwargs["spectral"],
            dynamic_soil=kwargs["dynamic_soil"],
            static_soil=kwargs["static_soil"],
        )
        vec, seq, info = self.base.fusion(
            embeddings, return_attention=True
        )
        vec = self.capacity_adapter(vec)
        pred = self.base.regression_head(vec)
        return {
            "prediction": pred,
            "fused_vector": vec,
            "fused_sequence": seq,
            "attention": info,
        }


# ============================================================
# PARAMETER MATCHING
# ============================================================

def build_capacity_candidate(simple_name: str, hidden: int):
    model = ExperimentModel(
        mode="capacity",
        simple_name=simple_name,
        adapter_hidden=hidden,
    )
    return count_parameters(model)[0]


def find_capacity_hidden(simple_name: str, target=TARGET_PARAMS):
    """
    Search the integer adapter width that minimizes absolute parameter
    difference from the full MAST-Fuse target.
    """
    best_h = None
    best_params = None
    best_diff = float("inf")

    # The parameter count is monotonic in hidden width for this adapter.
    lo, hi = CAPACITY_HIDDEN_MIN, CAPACITY_HIDDEN_MAX

    # Binary search for the nearest transition, then inspect a local window.
    while lo <= hi:
        mid = (lo + hi) // 2
        p = build_capacity_candidate(simple_name, mid)
        if p < target:
            lo = mid + 1
        else:
            hi = mid - 1

    candidates = range(max(1, hi - 8), min(CAPACITY_HIDDEN_MAX, lo + 8) + 1)

    for h in candidates:
        p = build_capacity_candidate(simple_name, h)
        diff = abs(p - target)
        if diff < best_diff:
            best_diff = diff
            best_h = h
            best_params = p

    return best_h, best_params, best_diff / target


def build_capacity_configs():
    rows = []
    configs = {}

    print("\n" + "=" * 90)
    print("CAPACITY MATCHING")
    print("=" * 90)
    print(f"Target full MAST-Fuse parameters: {TARGET_PARAMS:,}")

    for key, (display, simple_name) in CAPACITY_BASELINES.items():
        h, p, rel = find_capacity_hidden(simple_name)
        configs[key] = dict(
            name=f"{display} (capacity-matched)",
            simple_name=simple_name,
            adapter_hidden=h,
            parameters=p,
            relative_error=rel,
        )
        rows.append({
            "experiment": key,
            "strategy": display,
            "adapter_hidden": h,
            "parameters": p,
            "target_parameters": TARGET_PARAMS,
            "absolute_difference": abs(p - TARGET_PARAMS),
            "relative_difference_percent": 100.0 * rel,
        })
        print(
            f"{display:28s} hidden={h:5d} "
            f"params={p:10,d} "
            f"diff={100*rel:7.3f}%"
        )

    with open(OUT / "capacity_matching.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    return configs


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        batch = move_batch(batch)
        check_batch(batch)

        y = batch["target"].reshape(-1)
        optimizer.zero_grad(set_to_none=True)

        pred = model(**batch_inputs(batch))
        loss = loss_fn(pred, y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total += float(loss.item()) * len(y)
        n += len(y)

    return total / max(n, 1)


@torch.no_grad()
def validation_loss(model, loader, loss_fn):
    model.eval()
    total = 0.0
    n = 0

    for batch in loader:
        batch = move_batch(batch)
        check_batch(batch)

        y = batch["target"].reshape(-1)
        pred = model(**batch_inputs(batch))
        loss = loss_fn(pred, y)

        total += float(loss.item()) * len(y)
        n += len(y)

    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []

    for batch in loader:
        batch = move_batch(batch)
        check_batch(batch)
        y = batch["target"].reshape(-1)
        p = model(**batch_inputs(batch))

        ys.append(y.cpu().numpy())
        ps.append(p.reshape(-1).cpu().numpy())

    return np.concatenate(ys), np.concatenate(ps)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return str(x)

    path.write_text(
        json.dumps(obj, indent=2, default=conv),
        encoding="utf-8",
    )


def run_seed(experiment, cfg, seed, data, resume=False):
    seed_dir = OUT / experiment / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = seed_dir / "metrics.json"
    checkpoint = seed_dir / "best_model.pt"

    if resume and metrics_file.exists() and checkpoint.exists():
        print(f"[RESUME] {experiment} seed={seed}")
        return json.loads(metrics_file.read_text(encoding="utf-8"))

    set_seed(seed)

    train_loader, val_loader, test_loader = make_loaders(data)
    if cfg["group"] == "internal":
        model = ExperimentModel(
        mode="internal",
        cfg=cfg,
    ).to(DEVICE)

    elif cfg["group"] == "capacity_reference":
        # Full MAST-Fuse reference: use the authoritative current model
        model = MASTFuse(
        weather_dim=17,
        spectral_dim=3,
        dynamic_soil_dim=4,
        static_soil_dim=28,
        sequence_length=SEQ,
        fusion_dim=FUSION_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,
        fusion_type="cross_attention",
    ).to(DEVICE)

    elif cfg["group"] == "capacity":
        model = ExperimentModel(
        mode="capacity",
        simple_name=cfg["simple_name"],
        adapter_hidden=cfg["adapter_hidden"],
    ).to(DEVICE)

    else:
        raise ValueError(f"Unknown experiment group: {cfg['group']}")
    

    total_params, trainable_params = count_parameters(model)

    print(
        f"\n[{experiment}] seed={seed} "
        f"params={total_params:,} "
        f"device={DEVICE}"
    )

    loss_fn = StableRegressionLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_l = train_one_epoch(
            model, train_loader, optimizer, loss_fn
        )
        val_l = validation_loss(
            model, val_loader, loss_fn
        )
        scheduler.step(val_l)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_l,
            "val_loss": val_l,
            "learning_rate": current_lr,
        })

        print(
            f"{experiment:38s} "
            f"seed={seed:4d} "
            f"epoch={epoch:03d} "
            f"train={train_l:.6f} "
            f"val={val_l:.6f} "
            f"lr={current_lr:.2e}"
        )

        if val_l < best - MIN_DELTA:
            best = float(val_l)
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_loss": best,
                    "config": cfg,
                    "parameters": total_params,
                },
                checkpoint,
            )
        else:
            stale += 1

        if stale >= PATIENCE:
            print(
                f"Early stopping: {experiment}, "
                f"seed={seed}, epoch={epoch}"
            )
            break

    ckpt = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    y_val, p_val = predict(model, val_loader)
    y_test, p_test = predict(model, test_loader)

    val_metrics = metrics(y_val, p_val)
    test_metrics = metrics(y_test, p_test)

    np.save(seed_dir / "val_true.npy", y_val)
    np.save(seed_dir / "val_pred.npy", p_val)
    np.save(seed_dir / "test_true.npy", y_test)
    np.save(seed_dir / "test_pred.npy", p_test)

    elapsed = time.time() - start

    result = {
        "experiment": experiment,
        "name": cfg["name"],
        "group": cfg["group"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "training_time_seconds": elapsed,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "config": cfg,
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_json(metrics_file, result)
    save_json(seed_dir / "history.json", history)

    return result


# ============================================================
# AGGREGATION
# ============================================================

METRICS = ["MAE", "RMSE", "R2", "Pearson"]


def aggregate(results):
    results = sorted(results, key=lambda x: x["seed"])

    if [r["seed"] for r in results] != SEEDS:
        raise RuntimeError("Exactly the five requested seeds are required.")

    out = {
        "parameters_mean": float(np.mean([r["parameters"] for r in results])),
        "parameters_std": float(np.std([r["parameters"] for r in results], ddof=1)),
        "test": {},
        "validation": {},
        "training_time_seconds": {},
        "best_epoch": {},
    }

    for split in ["test", "validation"]:
        for metric in METRICS:
            values = np.array(
                [r[split][metric] for r in results],
                dtype=float,
            )
            out[split][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)),
                "values": values.tolist(),
            }

    times = np.array(
        [r["training_time_seconds"] for r in results],
        dtype=float,
    )
    epochs = np.array(
        [r["best_epoch"] for r in results],
        dtype=float,
    )

    out["training_time_seconds"] = {
        "mean": float(np.mean(times)),
        "std": float(np.std(times, ddof=1)),
        "values": times.tolist(),
    }
    out["best_epoch"] = {
        "mean": float(np.mean(epochs)),
        "std": float(np.std(epochs, ddof=1)),
        "values": epochs.tolist(),
    }

    return out


def run_experiment(key, cfg, data, resume=False):
    results = []

    for seed in SEEDS:
        results.append(
            run_seed(
                key,
                cfg,
                seed,
                data,
                resume=resume,
            )
        )

    agg = aggregate(results)

    summary = {
        "experiment": key,
        "name": cfg["name"],
        "group": cfg["group"],
        "seeds": SEEDS,
        "config": cfg,
        "results": results,
        "aggregate": agg,
    }

    save_json(OUT / key / "summary.json", summary)

    print("\n" + "=" * 90)
    print(cfg["name"])
    print("=" * 90)

    for metric in METRICS:
        a = agg["test"][metric]
        print(
            f"{metric:8s}: "
            f"{a['mean']:.6f} ± {a['std']:.6f}"
        )

    print(
        f"Parameters: "
        f"{agg['parameters_mean']:,.0f} ± "
        f"{agg['parameters_std']:,.0f}"
    )

    return summary


# ============================================================
# PAIRED STATISTICS
# ============================================================

def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full_like(p, np.nan)

    valid = np.isfinite(p)
    if not valid.any():
        return result

    idx = np.where(valid)[0]
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    m = len(ranked)

    q = np.empty(m)
    running = 1.0

    for i in range(m - 1, -1, -1):
        running = min(
            running,
            ranked[i] * m / (i + 1),
        )
        q[i] = running

    restored = np.empty(m)
    restored[order] = q
    result[idx] = restored
    return result


def paired_statistics(summaries):
    base = next(
        s for s in summaries
        if s["experiment"] == "full_model"
    )

    rows = []

    for s in summaries:
        if s["experiment"] == "full_model":
            continue

        for metric in METRICS:
            a = np.array(
                [r["test"][metric] for r in base["results"]],
                dtype=float,
            )
            b = np.array(
                [r["test"][metric] for r in s["results"]],
                dtype=float,
            )

            difference = b - a

            t_stat, p_t = ttest_rel(a, b, nan_policy="omit")

            try:
                w_stat, p_w = wilcoxon(a, b)
            except ValueError:
                w_stat, p_w = np.nan, np.nan

            sd = np.std(difference, ddof=1)
            dz = (
                float(np.mean(difference) / sd)
                if sd > 0 else np.nan
            )

            rows.append({
                "comparison": s["experiment"],
                "name": s["name"],
                "metric": metric,
                "mean_difference_ablation_minus_full": float(
                    np.mean(difference)
                ),
                "dz": dz,
                "t_stat": float(t_stat),
                "p_t": float(p_t),
                "wilcoxon": float(w_stat),
                "p_w": float(p_w),
            })

    qt = benjamini_hochberg([r["p_t"] for r in rows])
    qw = benjamini_hochberg([r["p_w"] for r in rows])

    for i, row in enumerate(rows):
        row["p_t_fdr"] = (
            float(qt[i]) if np.isfinite(qt[i]) else np.nan
        )
        row["p_w_fdr"] = (
            float(qw[i]) if np.isfinite(qw[i]) else np.nan
        )

    path = OUT / "paired_significance.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


# ============================================================
# LATEX TABLES
# ============================================================

def fmt(mean, std):
    return f"{mean:.4f} $\\\\pm$ {std:.4f}"


def write_latex(summaries):
    rows = []

    for s in summaries:
        a = s["aggregate"]["test"]
        rows.append(
            f"{s['name']} & "
            f"{fmt(a['MAE']['mean'], a['MAE']['std'])} & "
            f"{fmt(a['RMSE']['mean'], a['RMSE']['std'])} & "
            f"{fmt(a['R2']['mean'], a['R2']['std'])} & "
            f"{fmt(a['Pearson']['mean'], a['Pearson']['std'])} & "
            f"{s['aggregate']['parameters_mean']:,.0f} \\\\"
        )

    text = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{MAST-Fuse internal component and capacity-controlled ablation results. Values are mean $\pm$ standard deviation over five independent seeds.}",
        r"\label{tab:mastfuse_internal_capacity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Configuration & MAE $\downarrow$ & RMSE $\downarrow$ & $R^2$ $\uparrow$ & Pearson $\uparrow$ & Parameters \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    path = OUT / "aggregate.tex"
    path.write_text(text, encoding="utf-8")
    return path


def write_aggregate_csv(summaries):
    rows = []

    for s in summaries:
        a = s["aggregate"]
        row = {
            "experiment": s["experiment"],
            "name": s["name"],
            "parameters_mean": a["parameters_mean"],
            "parameters_std": a["parameters_std"],
            "training_time_mean_seconds": a["training_time_seconds"]["mean"],
            "training_time_std_seconds": a["training_time_seconds"]["std"],
            "best_epoch_mean": a["best_epoch"]["mean"],
            "best_epoch_std": a["best_epoch"]["std"],
        }

        for split in ["validation", "test"]:
            for metric in METRICS:
                row[f"{split}_{metric}_mean"] = a[split][metric]["mean"]
                row[f"{split}_{metric}_std"] = a[split][metric]["std"]

        rows.append(row)

    path = OUT / "aggregate.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


# ============================================================
# SMOKE TESTS
# ============================================================

def smoke_test():
    print("=" * 90)
    print("MAST-FUSE INTERNAL/CAPACITY ABLATION SMOKE TEST")
    print("=" * 90)
    print("Device:", DEVICE)

    # Verify current MASTFuse construction.
    full = MASTFuse(
        weather_dim=17,
        spectral_dim=3,
        dynamic_soil_dim=4,
        static_soil_dim=28,
        sequence_length=21,
        fusion_dim=128,
        num_heads=4,
        ff_dim=256,
        dropout=0.1,
        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,
        fusion_type="cross_attention",
    ).to(DEVICE)

    p, _ = count_parameters(full)
    print(f"Current local MASTFuse parameters: {p:,}")
    print(f"Expected reference parameter count  : {TARGET_PARAMS:,}")

    if p != TARGET_PARAMS:
        print(
            "WARNING: local model parameter count does not equal "
            "the historical 2,781,027 reference. The script will "
            "continue, but the result must be reported with the "
            "actual local parameter count."
        )

    x = {
        "weather": torch.randn(2, 21, 17, device=DEVICE),
        "spectral": torch.randn(2, 21, 3, device=DEVICE),
        "dynamic_soil": torch.randn(2, 21, 4, device=DEVICE),
        "static_soil": torch.randn(2, 21, 28, device=DEVICE),
    }

    with torch.no_grad():
        y = full(**x)

    assert y.shape == (2,)
    assert torch.isfinite(y).all()
    print("[PASS] Full MAST-Fuse forward:", tuple(y.shape))

    # Check component ablation construction.
    for key, cfg in INTERNAL.items():
        m = ExperimentModel(
            mode="internal",
            cfg=cfg,
        ).to(DEVICE)

        with torch.no_grad():
            y = m(**x)

        assert y.shape == (2,)
        assert torch.isfinite(y).all()

        pp, _ = count_parameters(m)
        print(
            f"[PASS] {key:35s} "
            f"params={pp:,} output={tuple(y.shape)}"
        )

    # Capacity matching can be expensive because it constructs models
    # repeatedly, but this is still useful before a full training run.
    cap = build_capacity_configs()
    for key, cfg in cap.items():
        m = ExperimentModel(
            mode="capacity",
            simple_name=cfg["simple_name"],
            adapter_hidden=cfg["adapter_hidden"],
        ).to(DEVICE)
        with torch.no_grad():
            y = m(**x)
        assert y.shape == (2,)
        assert torch.isfinite(y).all()
        pp, _ = count_parameters(m)
        print(
            f"[PASS] {key:35s} "
            f"params={pp:,} output={tuple(y.shape)}"
        )

    print("=" * 90)
    print("SMOKE TEST PASSED")
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--group",
        choices=["internal", "capacity", "all"],
        default="all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    print("=" * 90)
    print("MAST-Fuse INTERNAL + CAPACITY-CONTROLLED ABLATION")
    print("=" * 90)
    print("Project:", ROOT)
    print("Device :", DEVICE)
    print("Seeds  :", SEEDS)
    print("Epochs :", EPOCHS)
    print("Batch  :", BATCH_SIZE)
    print("=" * 90)

    data = load_data()

    summaries = []

    # --------------------------------------------------------
    # Internal component ablation
    # --------------------------------------------------------
    if args.group in {"internal", "all"}:
        for key, cfg0 in INTERNAL.items():
            cfg = dict(cfg0)
            cfg["group"] = "internal"
            summaries.append(
                run_experiment(
                    key,
                    cfg,
                    data,
                    resume=args.resume,
                )
            )

    # --------------------------------------------------------
    # Capacity-controlled fusion comparison
    # --------------------------------------------------------
    if args.group in {"capacity", "all"}:
        capacity_cfgs = build_capacity_configs()

        # Add the full model as the reference in this group.
        full_cfg = {
            "name": "Full MAST-Fuse (reference)",
            "group": "capacity_reference",
        }

        summaries.append(
            run_experiment(
                "capacity_full_mastfuse",
                full_cfg,
                data,
                resume=args.resume,
            )
        )

        for key, cfg in capacity_cfgs.items():
            cfg = dict(cfg)
            cfg["group"] = "capacity"
            summaries.append(
                run_experiment(
                    key,
                    cfg,
                    data,
                    resume=args.resume,
                )
            )

    if summaries:
        csv_path = write_aggregate_csv(summaries)
        tex_path = write_latex(summaries)

        print("\nAggregate CSV :", csv_path)
        print("LaTeX table   :", tex_path)

        # Paired significance requires the canonical full_model.
        if any(s["experiment"] == "full_model" for s in summaries):
            print("Paired tests  :", paired_statistics(summaries))

    print("\nCompleted.")


if __name__ == "__main__":
    main()
