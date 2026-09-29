#!/usr/bin/env python3
"""
===============================================================================
MuSTIPest-V3 / MAST-Fuse
STRICTLY CONTROLLED FUSION COMPARISON
===============================================================================

Purpose
-------
Compare six fusion strategies while changing ONLY the fusion mechanism:

    1. Early Fusion
    2. Feature Fusion
    3. Cross-Modal Attention
    4. Late Fusion
    5. Hybrid Attention
    6. MAST-Fuse

STRICT CONTROL
--------------
Every method uses exactly the same:
    * train / validation / test split
    * four modality inputs
    * modality-specific encoders from the current project
    * one SHARED DR-TCN temporal refiner (applied to each modality)
    * fusion dimension = 128
    * regression head: 128 -> 128 -> 64 -> 1
    * dropout = 0.10
    * StableRegressionLoss
    * AdamW
    * learning rate = 1e-4
    * weight decay = 1e-5
    * gradient clipping = 1.0
    * ReduceLROnPlateau (factor=0.5, patience=7, min_lr=1e-7)
    * early stopping (patience=20, min_delta=1e-6)
    * batch size = 32
    * 150 epochs maximum
    * five seeds = [42, 123, 456, 789, 1011]

The comparison deliberately does NOT use different modality encoders or
raw-input architectures for the baseline methods. This is essential: a raw
"early fusion" model would change the representation-learning backbone and
would therefore not be a fusion-only ablation.

Fusion definitions
------------------
All four modality representations entering the fusion stage have shape
(B,T,128), after the identical encoder + shared DR-TCN pipeline.

Early Fusion
    Concatenate the four encoded sequences along the feature dimension,
    followed by one 512 -> 128 projection and LayerNorm/GELU.

Feature Fusion
    Element-wise mean of the four common 128-D modality representations.

Cross-Modal Attention
    One shared multi-head self-attention block over modality tokens at each
    time step, followed by mean aggregation over modalities and learned
    temporal attention pooling.

Late Fusion
    Independently temporal-pool each modality to a 128-D vector, then average
    the modality vectors before the COMMON regression head.

Hybrid Attention
    One cross-modal self-attention block, followed by adaptive modality
    gating and learned temporal attention pooling.

MAST-Fuse
    The authoritative current CrossModalFusion implementation imported from
    the project's fusion.py. No baseline is given access to the MAST-Fuse
    components.

Outputs
-------
results/fusion_comparison_controlled/
    raw_results.csv
    mean_std_results.csv
    paper_table.tex
    parameter_comparison.csv
    seed_<seed>/<method>_best.pt
    seed_<seed>/<method>_history.csv
    seed_<seed>/<method>_predictions.csv
    seed_<seed>/<method>_metrics.json
    configuration.json

Usage
-----

1) Sanity check only:
    python fusion_comparison_controlled.py --check-only

2) One seed:
    python fusion_comparison_controlled.py --seed 42

3) Full five-seed experiment:
    python fusion_comparison_controlled.py

4) Force retraining:
    python fusion_comparison_controlled.py --force-rerun

5) If your authoritative current model file has another name:
    python fusion_comparison_controlled.py --model-file "model(8).py"

Notes
-----
* The script expects the current project encoder/fusion/model/loss files.
* If the project exposes the preprocessed .npy files, those are loaded
  directly, avoiding accidental use of an incompatible older dataset.py.
* No numerical results are fabricated by this script. The LaTeX table is
  generated only from successfully completed runs.
===============================================================================
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011]

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
DROPOUT = 0.10
NUM_HEADS = 4
FUSION_DIM = 128
FF_DIM = 256
SEQUENCE_LENGTH = 21
PATIENCE = 20
MIN_DELTA = 1e-6
LR_PATIENCE = 7
LR_FACTOR = 0.5
MIN_LR = 1e-7
NUM_WORKERS = 0

WEATHER_DIM = 17
SPECTRAL_DIM = 3
DYNAMIC_SOIL_DIM = 4
STATIC_SOIL_DIM = 28

METHODS = [
    "early_fusion",
    "feature_fusion",
    "cross_modal_attention",
    "late_fusion",
    "hybrid_attention",
    "mast_fuse",
]

DISPLAY_NAMES = {
    "early_fusion": "Early Fusion",
    "feature_fusion": "Feature Fusion",
    "cross_modal_attention": "Cross-Modal Attention",
    "late_fusion": "Late Fusion",
    "hybrid_attention": "Hybrid Attention",
    "mast_fuse": "MAST-Fuse",
}

MODALITIES = ["weather", "spectral", "dynamic_soil", "static_soil"]

RESULT_DIR = Path("results") / "fusion_comparison_controlled"


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DYNAMIC PROJECT MODULE LOADING
# =============================================================================

def load_module_from_file(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {module_name} file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_project_file(project_root: Path, explicit: Optional[str], candidates: Sequence[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = project_root / p
        return p
    for name in candidates:
        p = project_root / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate the required project file. Tried: "
        + ", ".join(candidates)
    )


def load_project_classes(
    project_root: Path,
    model_file: Optional[str],
    encoders_file: Optional[str],
    fusion_file: Optional[str],
    losses_file: Optional[str],
):
    """Load the authoritative project classes without depending on dataset.py."""

    enc_path = resolve_project_file(
        project_root,
        encoders_file,
        ["encoders(10).py", "encoders(9).py", "encoders(8).py", "encoders.py", "encoders(1).py"],
    )
    fusion_path = resolve_project_file(
        project_root,
        fusion_file,
        ["fusion(2).py", "fusion(8).py", "fusion.py"],
    )
    loss_path = resolve_project_file(
        project_root,
        losses_file,
        ["losses.py"],
    )

    # The current model.py imports modules literally as `encoders`, `fusion`,
    # and therefore these aliases must be installed before loading the model.
    enc_mod = load_module_from_file("encoders", enc_path)
    fusion_mod = load_module_from_file("fusion", fusion_path)

    model_path = resolve_project_file(
        project_root,
        model_file,
        [
            "model(8).py",
            "model(10).py",
            "model(20260913-141947).py",
            "model(20260913-063023).py",
            "model.py",
        ],
    )
    model_mod = load_module_from_file("mast_model_current", model_path)
    loss_mod = load_module_from_file("stable_losses", loss_path)

    required = [
        "MultimodalEncoders",
        "build_temporal_refiner",
        "RegressionHead",
    ]
    missing = [name for name in required if not hasattr(model_mod, name)]
    if missing:
        raise AttributeError(
            f"{model_path} does not expose required classes/functions: {missing}"
        )
    if not hasattr(fusion_mod, "CrossModalFusion"):
        raise AttributeError(f"{fusion_path} does not expose CrossModalFusion")
    if not hasattr(loss_mod, "StableRegressionLoss"):
        raise AttributeError(f"{loss_path} does not expose StableRegressionLoss")

    return {
        "model": model_mod,
        "encoders": enc_mod,
        "fusion": fusion_mod,
        "losses": loss_mod,
        "paths": {
            "model": str(model_path),
            "encoders": str(enc_path),
            "fusion": str(fusion_path),
            "losses": str(loss_path),
        },
    }


# =============================================================================
# DATASET
# =============================================================================

class NPYFusionDataset(Dataset):
    """Four-modality MuSTIPest-V3 dataset backed by split-specific .npy files."""

    def __init__(self, root: Path, split: str):
        self.weather = self._load(root / f"X_weather_{split}.npy")
        self.spectral = self._load(root / f"X_spectral_{split}.npy")
        self.dynamic_soil = self._load(root / f"X_dynamic_soil_{split}.npy")
        self.static_soil = self._load(root / f"X_static_soil_{split}.npy")
        self.target = self._load(root / f"y_{split}.npy").reshape(-1)
        self._validate()

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        return torch.from_numpy(np.load(path, allow_pickle=False)).float()

    def _validate(self) -> None:
        n = len(self.target)
        expected = {
            "weather": (n, SEQUENCE_LENGTH, WEATHER_DIM),
            "spectral": (n, SEQUENCE_LENGTH, SPECTRAL_DIM),
            "dynamic_soil": (n, SEQUENCE_LENGTH, DYNAMIC_SOIL_DIM),
            "static_soil": (n, SEQUENCE_LENGTH, STATIC_SOIL_DIM),
            "target": (n,),
        }
        actual = {
            "weather": tuple(self.weather.shape),
            "spectral": tuple(self.spectral.shape),
            "dynamic_soil": tuple(self.dynamic_soil.shape),
            "static_soil": tuple(self.static_soil.shape),
            "target": tuple(self.target.shape),
        }
        for name in expected:
            if actual[name] != expected[name]:
                raise ValueError(
                    f"{name} shape mismatch: got {actual[name]}, expected {expected[name]}"
                )
            tensor = getattr(self, name)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "weather": self.weather[idx],
            "spectral": self.spectral[idx],
            "dynamic_soil": self.dynamic_soil[idx],
            "static_soil": self.static_soil[idx],
            "target": self.target[idx],
        }


def create_loaders(root: Path, seed: int):
    train_ds = NPYFusionDataset(root, "train")
    val_ds = NPYFusionDataset(root, "val")
    test_ds = NPYFusionDataset(root, "test")

    # Identical batch ordering across all fusion methods for a given seed.
    generator = torch.Generator()
    generator.manual_seed(seed)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# COMMON BACKBONE
# =============================================================================

class SharedBackbone(nn.Module):
    """
    Exactly common to all six methods.

    Modality encoders are the project's current MultimodalEncoders.
    The temporal refiner is ONE shared DR-TCN instance applied to each
    modality, not a separate temporal module per modality.
    """

    def __init__(self, classes):
        super().__init__()
        self.encoders = classes["model"].MultimodalEncoders(
            weather_dim=WEATHER_DIM,
            spectral_dim=SPECTRAL_DIM,
            dynamic_soil_dim=DYNAMIC_SOIL_DIM,
            static_soil_dim=STATIC_SOIL_DIM,
            fusion_dim=FUSION_DIM,
            dropout=DROPOUT,
            max_len=SEQUENCE_LENGTH,
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
        )
        self.temporal_refiner = classes["model"].build_temporal_refiner(
            temporal_encoder="tcn",
            dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
            sequence_length=SEQUENCE_LENGTH,
        )
        self.regression_head = classes["model"].RegressionHead(
            input_dim=FUSION_DIM,
            hidden_dims=(128, 64),
            dropout=DROPOUT,
        )

    def encode(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self.encoders(
            weather=batch["weather"],
            spectral=batch["spectral"],
            dynamic_soil=batch["dynamic_soil"],
            static_soil=batch["static_soil"],
        )
        refined = {name: self.temporal_refiner(x) for name, x in z.items()}
        for name, x in refined.items():
            if x.shape != (batch["target"].shape[0], SEQUENCE_LENGTH, FUSION_DIM):
                raise RuntimeError(
                    f"Temporal output for {name} has invalid shape {tuple(x.shape)}"
                )
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite temporal representation in {name}")
        return refined


# =============================================================================
# COMMON TEMPORAL ATTENTION POOLING
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int = FUSION_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.score(x).squeeze(-1)
        weights = torch.softmax(score, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


# =============================================================================
# FUSION MODULES
# =============================================================================

class EarlyFusion(nn.Module):
    """Latent early fusion: concatenate all encoded modalities, then project."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4 * FUSION_DIM, FUSION_DIM),
            nn.GELU(),
            nn.LayerNorm(FUSION_DIM),
            nn.Dropout(DROPOUT),
        )
        self.temporal_pool = TemporalAttentionPooling()

    def forward(self, z: Dict[str, torch.Tensor]):
        x = torch.cat([z[m] for m in MODALITIES], dim=-1)
        x = self.projection(x)
        pooled, tw = self.temporal_pool(x)
        return pooled, {"temporal_weights": tw}


class FeatureFusion(nn.Module):
    """Parameter-free feature fusion by arithmetic mean in the common latent space."""

    def forward(self, z: Dict[str, torch.Tensor]):
        x = torch.stack([z[m] for m in MODALITIES], dim=0).mean(dim=0)
        pooled = x.mean(dim=1)
        tw = torch.full(
            (x.shape[0], x.shape[1]),
            1.0 / x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )
        return pooled, {"temporal_weights": tw}


class CrossModalAttentionFusion(nn.Module):
    """
    Single shared attention block over the four modality tokens at each time.
    No modality-specific attention blocks are used.
    """

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(FUSION_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(FUSION_DIM, FF_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FF_DIM, FUSION_DIM),
        )
        self.norm2 = nn.LayerNorm(FUSION_DIM)
        self.temporal_pool = TemporalAttentionPooling()

    def forward(self, z: Dict[str, torch.Tensor]):
        # (B,T,M,D) -> (B*T,M,D)
        x = torch.stack([z[m] for m in MODALITIES], dim=2)
        b, t, m, d = x.shape
        tokens = x.reshape(b * t, m, d)
        attn_out, attn_w = self.attn(tokens, tokens, tokens, need_weights=True)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        x = tokens.reshape(b, t, m, d).mean(dim=2)
        pooled, tw = self.temporal_pool(x)
        return pooled, {
            "temporal_weights": tw,
            "cross_modal_attention": attn_w,
        }


class LateFusion(nn.Module):
    """Late feature fusion: independently pool each modality, then average vectors."""

    def __init__(self):
        super().__init__()
        self.pools = nn.ModuleDict({m: TemporalAttentionPooling() for m in MODALITIES})

    def forward(self, z: Dict[str, torch.Tensor]):
        vectors = []
        weights = []
        for m in MODALITIES:
            v, w = self.pools[m](z[m])
            vectors.append(v)
            weights.append(w)
        fused = torch.stack(vectors, dim=0).mean(dim=0)
        return fused, {
            "temporal_weights": torch.stack(weights, dim=1),
        }


class HybridAttentionFusion(nn.Module):
    """Cross-modal attention followed by adaptive modality gating and temporal pooling."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(FUSION_DIM)
        self.gate = nn.Sequential(
            nn.Linear(FUSION_DIM, FUSION_DIM // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_DIM // 2, 1),
        )
        self.temporal_pool = TemporalAttentionPooling()

    def forward(self, z: Dict[str, torch.Tensor]):
        x = torch.stack([z[m] for m in MODALITIES], dim=2)
        b, t, m, d = x.shape
        tokens = x.reshape(b * t, m, d)
        attn_out, attn_w = self.attn(tokens, tokens, tokens, need_weights=True)
        tokens = self.norm(tokens + attn_out)
        attended = tokens.reshape(b, t, m, d)

        logits = self.gate(attended).squeeze(-1)  # (B,T,M)
        weights = torch.softmax(logits, dim=2)
        fused = torch.sum(attended * weights.unsqueeze(-1), dim=2)
        pooled, tw = self.temporal_pool(fused)
        return pooled, {
            "modality_weights": weights,
            "temporal_weights": tw,
            "cross_modal_attention": attn_w,
        }


class MASTFuseAdapter(nn.Module):
    """Adapter around the authoritative current CrossModalFusion."""

    def __init__(self, classes):
        super().__init__()
        self.fusion = classes["fusion"].CrossModalFusion(
            fusion_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
        )

    def forward(self, z: Dict[str, torch.Tensor]):
        try:
            result = self.fusion(z, return_attention=True)
        except TypeError:
            result = self.fusion(z)
        if not isinstance(result, (tuple, list)):
            raise RuntimeError("CrossModalFusion returned an unexpected object")
        fused = result[0]
        info = result[2] if len(result) >= 3 else {}
        return fused, info


# =============================================================================
# CONTROLLED MODEL
# =============================================================================

class ControlledFusionModel(nn.Module):
    """
    Complete model with one and only one variable component: `fusion`.
    """

    def __init__(self, method: str, classes):
        super().__init__()
        self.method = method
        self.backbone = SharedBackbone(classes)

        if method == "early_fusion":
            self.fusion = EarlyFusion()
        elif method == "feature_fusion":
            self.fusion = FeatureFusion()
        elif method == "cross_modal_attention":
            self.fusion = CrossModalAttentionFusion()
        elif method == "late_fusion":
            self.fusion = LateFusion()
        elif method == "hybrid_attention":
            self.fusion = HybridAttentionFusion()
        elif method == "mast_fuse":
            self.fusion = MASTFuseAdapter(classes)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, batch: Mapping[str, torch.Tensor], return_aux: bool = False):
        z = self.backbone.encode(batch)
        fused, info = self.fusion(z)
        pred = self.backbone.regression_head(fused)
        if return_aux:
            return pred, info
        return pred

    def common_state(self):
        return {
            "backbone.encoders": copy.deepcopy(self.backbone.encoders.state_dict()),
            "backbone.temporal_refiner": copy.deepcopy(self.backbone.temporal_refiner.state_dict()),
            "backbone.regression_head": copy.deepcopy(self.backbone.regression_head.state_dict()),
        }

    def load_common_state(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.backbone.encoders.load_state_dict(state["backbone.encoders"])
        self.backbone.temporal_refiner.load_state_dict(state["backbone.temporal_refiner"])
        self.backbone.regression_head.load_state_dict(state["backbone.regression_head"])


# =============================================================================
# CONTROL CHECKS
# =============================================================================

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_component_parameters(model: ControlledFusionModel) -> Dict[str, int]:
    common = (
        count_parameters(model.backbone.encoders)
        + count_parameters(model.backbone.temporal_refiner)
        + count_parameters(model.backbone.regression_head)
    )
    fusion = count_parameters(model.fusion)
    return {
        "common_parameters": common,
        "fusion_parameters": fusion,
        "total_parameters": common + fusion,
    }


def compare_common_parameters(models: Mapping[str, ControlledFusionModel]) -> None:
    ref = models["early_fusion"]
    ref_counts = count_component_parameters(ref)
    for method, model in models.items():
        counts = count_component_parameters(model)
        if counts["common_parameters"] != ref_counts["common_parameters"]:
            raise AssertionError(
                "STRICT CONTROL FAILED: common backbone parameter count differs "
                f"for {method}: {counts['common_parameters']} vs "
                f"{ref_counts['common_parameters']}"
            )


def verify_common_initialization(models: Mapping[str, ControlledFusionModel]) -> None:
    ref = models["early_fusion"]
    for method, model in models.items():
        for component in ["encoders", "temporal_refiner", "regression_head"]:
            a = getattr(ref.backbone, component).state_dict()
            b = getattr(model.backbone, component).state_dict()
            if a.keys() != b.keys():
                raise AssertionError(
                    f"STRICT CONTROL FAILED: state keys differ for {method}/{component}"
                )
            for key in a:
                if not torch.equal(a[key], b[key]):
                    raise AssertionError(
                        f"STRICT CONTROL FAILED: initial weights differ for "
                        f"{method}/{component}/{key}"
                    )


def sanity_check(classes, data_root: Path, device: torch.device) -> None:
    print("\n=== STRICT CONTROL SANITY CHECK ===")
    set_seed(SEEDS[0])
    models = {
        method: ControlledFusionModel(method, classes).to(device)
        for method in METHODS
    }
    compare_common_parameters(models)
    common_state = models["early_fusion"].common_state()
    for method, model in models.items():
        model.load_common_state(common_state)
    verify_common_initialization(models)

    params = []
    for method, model in models.items():
        c = count_component_parameters(model)
        params.append({"method": DISPLAY_NAMES[method], **c})
    df = pd.DataFrame(params)
    print(df.to_string(index=False))

    train_loader, _, _ = create_loaders(data_root, SEEDS[0])
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        for method, model in models.items():
            pred = model(batch)
            if pred.shape != batch["target"].shape:
                raise AssertionError(
                    f"{method}: prediction shape {tuple(pred.shape)} != "
                    f"target shape {tuple(batch['target'].shape)}"
                )
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"{method}: non-finite prediction")
            print(f"  [OK] {DISPLAY_NAMES[method]:24s} output={tuple(pred.shape)}")

    print("\nStrict control checks passed.")
    print("Common encoder + DR-TCN + regression-head weights are identical at initialization.")
    print("Only the fusion module differs.")


# =============================================================================
# LOSS / METRICS
# =============================================================================

def build_loss(classes):
    return classes["losses"].StableRegressionLoss(
        smooth_l1_weight=1.0,
        mse_weight=0.25,
        range_penalty_weight=0.05,
        min_target=0.0,
        max_target=1.0,
        beta=0.1,
    )


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson": pearson,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    ys = []
    ps = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(batch)
            loss = criterion(pred, batch["target"])
            n = batch["target"].numel()
            total_loss += float(loss.item()) * n
            total_n += n
            ys.append(batch["target"].detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    metrics = regression_metrics(y, p)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics, y, p


# =============================================================================
# TRAINING
# =============================================================================

def train_one_seed(
    method: str,
    seed: int,
    classes,
    data_root: Path,
    device: torch.device,
    force_rerun: bool,
):
    seed_dir = RESULT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = seed_dir / f"{method}_best.pt"
    history_file = seed_dir / f"{method}_history.csv"
    pred_file = seed_dir / f"{method}_predictions.csv"
    metric_file = seed_dir / f"{method}_metrics.json"

    if checkpoint.exists() and history_file.exists() and pred_file.exists() and metric_file.exists() and not force_rerun:
        print(f"[SKIP] {DISPLAY_NAMES[method]} seed={seed}: existing results found")
        return json.loads(metric_file.read_text())

    set_seed(seed)

    # Build all methods from the same seed and copy the common initialization
    # from a single reference model. This removes initialization as a source
    # of variation in the common backbone.
    models = {
        m: ControlledFusionModel(m, classes).to(device)
        for m in METHODS
    }
    compare_common_parameters(models)

    common_state = models["early_fusion"].common_state()
    model = models[method]
    model.load_common_state(common_state)
    del models

    train_loader, val_loader, test_loader = create_loaders(data_root, seed)
    criterion = build_loss(classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    best_val = math.inf
    best_epoch = 0
    wait = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = criterion(pred, batch["target"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at method={method}, seed={seed}, epoch={epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            n = batch["target"].numel()
            train_loss_sum += float(loss.item()) * n
            train_n += n

        train_loss = train_loss_sum / max(train_n, 1)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_metrics["MAE"],
            "val_rmse": val_metrics["RMSE"],
            "val_r2": val_metrics["R2"],
            "val_pearson": val_metrics["Pearson"],
            "lr": lr,
            "epoch_seconds": epoch_time,
        })

        improved = val_loss < (best_val - MIN_DELTA)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": method,
                    "seed": seed,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "config": experiment_config(classes),
                },
                checkpoint,
            )
        else:
            wait += 1

        print(
            f"[{DISPLAY_NAMES[method]:24s}] seed={seed} "
            f"epoch={epoch:03d} train={train_loss:.6f} "
            f"val={val_loss:.6f} lr={lr:.2e}"
        )

        if wait >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    training_seconds = time.perf_counter() - training_start

    if not checkpoint.exists():
        raise RuntimeError(f"No checkpoint was saved for {method}, seed={seed}")

    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["state_dict"])
    test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_file, index=False)
    pd.DataFrame({"target": y_true, "prediction": y_pred}).to_csv(pred_file, index=False)

    result = {
        "method": method,
        "method_display": DISPLAY_NAMES[method],
        "seed": seed,
        "parameters": count_parameters(model),
        "common_parameters": count_component_parameters(model)["common_parameters"],
        "fusion_parameters": count_component_parameters(model)["fusion_parameters"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "training_seconds": training_seconds,
        "mean_epoch_seconds": float(history_df["epoch_seconds"].mean()) if len(history_df) else float("nan"),
        "epochs_completed": len(history_df),
        **test_metrics,
    }
    metric_file.write_text(json.dumps(result, indent=2))
    return result


# =============================================================================
# RESULT AGGREGATION / LATEX
# =============================================================================

def aggregate_results(results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(results)
    raw = raw.sort_values(["method", "seed"]).reset_index(drop=True)
    raw.to_csv(RESULT_DIR / "raw_results.csv", index=False)

    rows = []
    for method in METHODS:
        g = raw[raw["method"] == method]
        if g.empty:
            continue
        row = {
            "method": method,
            "Fusion Strategy": DISPLAY_NAMES[method],
            "Parameters": int(round(g["parameters"].mean())),
        }
        for metric in ["MAE", "RMSE", "R2", "Pearson"]:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else 0.0
        row["training_seconds_mean"] = float(g["training_seconds"].mean())
        row["training_seconds_std"] = float(g["training_seconds"].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "mean_std_results.csv", index=False)
    return raw, summary


def latex_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Strictly controlled five-seed comparison of fusion strategies for MuSTIPest-V3. All methods use the identical modality encoders, shared DR-TCN temporal refiner, regression head, training split, optimizer, learning-rate schedule, regularization, gradient clipping, and StableRegressionLoss. Only the fusion mechanism is changed. Results are mean $\pm$ standard deviation over five seeds.}",
        r"\label{tab:fusion_comparison_controlled}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Fusion Strategy & Parameters & MAE $\downarrow$ & RMSE $\downarrow$ & $R^2$ $\uparrow$ & Pearson $\uparrow$ \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        name = row["Fusion Strategy"]
        if name == "MAST-Fuse":
            name = r"\textbf{MAST-Fuse}"
            vals = [
                r"\textbf{" + f"{int(row['Parameters']):,}" + r"}",
                r"\textbf{" + latex_pm(row["MAE_mean"], row["MAE_std"]) + r"}",
                r"\textbf{" + latex_pm(row["RMSE_mean"], row["RMSE_std"]) + r"}",
                r"\textbf{" + latex_pm(row["R2_mean"], row["R2_std"]) + r"}",
                r"\textbf{" + latex_pm(row["Pearson_mean"], row["Pearson_std"]) + r"}",
            ]
        else:
            vals = [
                f"{int(row['Parameters']):,}",
                latex_pm(row["MAE_mean"], row["MAE_std"]),
                latex_pm(row["RMSE_mean"], row["RMSE_std"]),
                latex_pm(row["R2_mean"], row["R2_std"]),
                latex_pm(row["Pearson_mean"], row["Pearson_std"]),
            ]
        lines.append(name + " & " + " & ".join(vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ]
    (RESULT_DIR / "paper_table.tex").write_text("\n".join(lines), encoding="utf-8")


def save_parameter_comparison(summary: pd.DataFrame) -> None:
    cols = ["method", "Fusion Strategy", "Parameters"]
    out = summary[cols].copy()
    out["Common Parameters"] = [
        int(summary.loc[summary["method"] == m, "Parameters"].iloc[0])
        for m in out["method"]
    ]
    out.to_csv(RESULT_DIR / "parameter_comparison.csv", index=False)


# =============================================================================
# CONFIGURATION
# =============================================================================

def experiment_config(classes) -> Dict:
    return {
        "seeds": SEEDS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "dropout": DROPOUT,
        "num_heads": NUM_HEADS,
        "fusion_dim": FUSION_DIM,
        "ff_dim": FF_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "early_stopping_patience": PATIENCE,
        "early_stopping_min_delta": MIN_DELTA,
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "min_lr": MIN_LR,
        "loss": {
            "name": "StableRegressionLoss",
            "smooth_l1_weight": 1.0,
            "mse_weight": 0.25,
            "range_penalty_weight": 0.05,
            "min_target": 0.0,
            "max_target": 1.0,
            "beta": 0.1,
        },
        "temporal_refiner": "DR-TCN",
        "modalities": MODALITIES,
        "dimensions": {
            "weather": WEATHER_DIM,
            "spectral": SPECTRAL_DIM,
            "dynamic_soil": DYNAMIC_SOIL_DIM,
            "static_soil": STATIC_SOIL_DIM,
        },
        "fusion_definitions": {
            "early_fusion": "latent concatenation 4x128 -> 128 + temporal attention pooling",
            "feature_fusion": "element-wise mean of four 128-D sequences + mean temporal pooling",
            "cross_modal_attention": "one shared 4-token multi-head attention block + temporal attention pooling",
            "late_fusion": "independent temporal attention pooling per modality + vector mean",
            "hybrid_attention": "one shared cross-modal attention block + modality gate + temporal attention pooling",
            "mast_fuse": "authoritative CrossModalFusion from fusion.py",
        },
        "project_files": classes["paths"],
    }


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Strictly controlled MuSTIPest-V3 fusion comparison")
    parser.add_argument("--data-root", type=str, default="./preprocessed")
    parser.add_argument("--model-file", type=str, default=None)
    parser.add_argument("--encoders-file", type=str, default=None)
    parser.add_argument("--fusion-file", type=str, default=None)
    parser.add_argument("--losses-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()

    global RESULT_DIR
    if args.output_dir:
        RESULT_DIR = Path(args.output_dir)
        if not RESULT_DIR.is_absolute():
            RESULT_DIR = project_root / RESULT_DIR

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    classes = load_project_classes(
        project_root,
        args.model_file,
        args.encoders_file,
        args.fusion_file,
        args.losses_file,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Project files:")
    for k, v in classes["paths"].items():
        print(f"  {k:8s}: {v}")

    config = experiment_config(classes)
    (RESULT_DIR / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    sanity_check(classes, data_root, device)
    if args.check_only:
        return

    seeds = [args.seed] if args.seed is not None else SEEDS
    results = []
    for seed in seeds:
        for method in METHODS:
            result = train_one_seed(
                method=method,
                seed=seed,
                classes=classes,
                data_root=data_root,
                device=device,
                force_rerun=args.force_rerun,
            )
            results.append(result)

    raw, summary = aggregate_results(results)
    write_latex_table(summary)
    save_parameter_comparison(summary)

    print("\n=== FINAL SUMMARY ===")
    display = summary[[
        "Fusion Strategy", "Parameters",
        "MAE_mean", "MAE_std",
        "RMSE_mean", "RMSE_std",
        "R2_mean", "R2_std",
        "Pearson_mean", "Pearson_std",
    ]].copy()
    print(display.to_string(index=False))
    print(f"\nSaved results to: {RESULT_DIR.resolve()}")
    print(f"LaTeX table:       {(RESULT_DIR / 'paper_table.tex').resolve()}")


if __name__ == "__main__":
    main()
