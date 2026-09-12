#Author - Dipanwita Thakur
import os
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULT_DIR = PROJECT_ROOT / "results" / "fusion_comparison"
MODEL_DIR = RESULT_DIR / "best_models"
PLOT_DIR = RESULT_DIR / "plots"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Dataset dimensions
# -------------------------------------------------------------------------

SEQ_LEN = 21

WEATHER_DIM = 17
SPECTRAL_DIM = 3
DYNAMIC_SOIL_DIM = 4
STATIC_SOIL_DIM = 28

LATENT_DIM = 128


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0

SEEDS = [42]

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    """
    Set all random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET ADAPTER
# =============================================================================

class FusionDataset(Dataset):
    """
    Converts the existing MuSTIPest-V3 split into a PyTorch Dataset.

    Expected fields:

        weather
        spectral
        dynamic_soil
        static_soil
        target
    """

    def __init__(self, split):
        self.weather = self._get(split, "weather")
        self.spectral = self._get(split, "spectral")
        self.dynamic_soil = self._get(split, "dynamic_soil")
        self.static_soil = self._get(split, "static_soil")
        self.target = self._get(split, "target")

        self.weather = self._to_tensor(self.weather)
        self.spectral = self._to_tensor(self.spectral)
        self.dynamic_soil = self._to_tensor(self.dynamic_soil)
        self.static_soil = self._to_tensor(self.static_soil)
        self.target = self._to_tensor(self.target).float().reshape(-1)

        self._validate()

    @staticmethod
    def _get(obj, name):

        if isinstance(obj, dict):
            if name in obj:
                return obj[name]

        if hasattr(obj, name):
            return getattr(obj, name)

        raise AttributeError(
            f"Cannot find '{name}' in dataset split."
        )

    @staticmethod
    def _to_tensor(x):

        if torch.is_tensor(x):
            return x.float()

        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()

        return torch.tensor(x, dtype=torch.float32)

    def _validate(self):

        n = len(self.target)

        if self.weather.shape != (n, SEQ_LEN, WEATHER_DIM):
            raise ValueError(
                f"Weather shape incorrect: {self.weather.shape}; "
                f"expected {(n, SEQ_LEN, WEATHER_DIM)}"
            )

        if self.spectral.shape != (n, SEQ_LEN, SPECTRAL_DIM):
            raise ValueError(
                f"Spectral shape incorrect: {self.spectral.shape}; "
                f"expected {(n, SEQ_LEN, SPECTRAL_DIM)}"
            )

        if self.dynamic_soil.shape != (
            n,
            SEQ_LEN,
            DYNAMIC_SOIL_DIM,
        ):
            raise ValueError(
                f"Dynamic soil shape incorrect: "
                f"{self.dynamic_soil.shape}"
            )

        if self.static_soil.shape != (
            n,
            SEQ_LEN,
            STATIC_SOIL_DIM,
        ):
            raise ValueError(
                f"Static soil shape incorrect: "
                f"{self.static_soil.shape}"
            )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        return {
            "weather": self.weather[idx],
            "spectral": self.spectral[idx],
            "dynamic_soil": self.dynamic_soil[idx],
            "static_soil": self.static_soil[idx],
            "target": self.target[idx],
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset_from_existing_module():
    """
    Uses the existing dataset.py.

    Your dataset.py currently exposes load_all_data().
    """

    import dataset

    if not hasattr(dataset, "load_all_data"):
        raise ImportError(
            "dataset.py does not contain load_all_data()."
        )

    result = dataset.load_all_data()

    print()
    print("=" * 80)
    print("DATASET OBJECT")
    print("=" * 80)

    print("Returned type:", type(result))

    return result


def unpack_data(result):
    """
    Handles common return formats from dataset.py.

    Supported:

        1. dictionary:
            {
                "train": ...,
                "validation": ...,
                "test": ...
            }

        2. tuple/list:
            (train, validation, test)

        3. object containing:
            train
            validation
            test
    """

    if isinstance(result, dict):

        train = result.get("train")

        val = result.get("validation")

        if val is None:
            val = result.get("val")

        test = result.get("test")

        if train is not None and val is not None and test is not None:
            return train, val, test

    if isinstance(result, (tuple, list)):

        if len(result) >= 3:
            return result[0], result[1], result[2]

    if hasattr(result, "train"):

        train = result.train

        val = getattr(
            result,
            "validation",
            getattr(result, "val", None),
        )

        test = getattr(result, "test", None)

        if train is not None and val is not None and test is not None:
            return train, val, test

    raise TypeError(
        "\n"
        "Could not identify train/validation/test splits returned "
        "by dataset.load_all_data().\n\n"
        "Expected one of:\n"
        "  load_all_data() -> (train, validation, test)\n"
        "or\n"
        "  load_all_data() -> {'train': ..., "
        "'validation': ..., 'test': ...}\n"
    )


def create_loaders():

    raw_data = load_dataset_from_existing_module()

    train_raw, val_raw, test_raw = unpack_data(raw_data)

    train_dataset = FusionDataset(train_raw)
    val_dataset = FusionDataset(val_raw)
    test_dataset = FusionDataset(test_raw)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print("=" * 80)
    print("DATALOADERS")
    print("=" * 80)

    print("Train:", len(train_dataset))
    print("Val  :", len(val_dataset))
    print("Test :", len(test_dataset))

    return train_loader, val_loader, test_loader


# =============================================================================
# SHARED MODALITY ENCODER
# =============================================================================

class ModalityEncoder(nn.Module):
    """
    Common modality-specific temporal encoder.

    Input:

        (B, T, D)

    Output:

        (B, T, LATENT_DIM)

    This encoder is deliberately identical across modalities so that
    the fusion comparison isolates the fusion mechanism.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = LATENT_DIM,
        dropout: float = 0.10,
    ):

        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            latent_dim,
        )

        self.conv1 = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            latent_dim,
            latent_dim,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.LayerNorm(latent_dim)

        self.norm2 = nn.LayerNorm(latent_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # B,T,D
        x = self.projection(x)

        residual = x

        # B,D,T
        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = F.gelu(x)

        x = self.conv2(x)

        # B,T,D
        x = x.transpose(1, 2)

        x = self.norm1(x)

        x = x + residual

        x = self.norm2(x)

        x = self.dropout(x)

        return x


# =============================================================================
# COMMON BACKBONE
# =============================================================================

class FourModalEncoders(nn.Module):
    """
    Four independent modality encoders.
    """

    def __init__(self):

        super().__init__()

        self.weather = ModalityEncoder(
            WEATHER_DIM
        )

        self.spectral = ModalityEncoder(
            SPECTRAL_DIM
        )

        self.dynamic_soil = ModalityEncoder(
            DYNAMIC_SOIL_DIM
        )

        self.static_soil = ModalityEncoder(
            STATIC_SOIL_DIM
        )

    def forward(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    ):

        return (
            self.weather(weather),
            self.spectral(spectral),
            self.dynamic_soil(dynamic_soil),
            self.static_soil(static_soil),
        )


# =============================================================================
# TEMPORAL POOLING
# =============================================================================

class TemporalPooling(nn.Module):
    """
    Attention-based temporal pooling.
    """

    def __init__(self, dim=LATENT_DIM):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):

        # x: B,T,D

        scores = self.score(x).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


# =============================================================================
# EARLY FUSION
# =============================================================================

class EarlyFusionModel(nn.Module):
    """
    EARLY FUSION

    Raw modalities are concatenated before the shared encoder.

    [Weather | Spectral | Dynamic Soil | Static Soil]
                           |
                     Shared encoder
                           |
                    Temporal pooling
                           |
                      Regression
    """

    fusion_name = "early_fusion"

    def __init__(self):

        super().__init__()

        total_dim = (
            WEATHER_DIM
            + SPECTRAL_DIM
            + DYNAMIC_SOIL_DIM
            + STATIC_SOIL_DIM
        )

        self.encoder = ModalityEncoder(
            total_dim,
            LATENT_DIM,
        )

        self.temporal_pool = TemporalPooling()

        self.regressor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    ):

        x = torch.cat(
            [
                weather,
                spectral,
                dynamic_soil,
                static_soil,
            ],
            dim=-1,
        )

        z = self.encoder(x)

        fused, temporal_weights = self.temporal_pool(z)

        prediction = self.regressor(fused).squeeze(-1)

        return {
            "prediction": prediction,
            "fused": fused,
            "temporal_weights": temporal_weights,
        }


# =============================================================================
# FEATURE-LEVEL FUSION
# =============================================================================

class FeatureFusionModel(nn.Module):
    """
    FEATURE-LEVEL / INTERMEDIATE FUSION

    Each modality is independently encoded.

              Weather ──────► Encoder ──┐
              Spectral ─────► Encoder ──┤
              Dynamic soil ► Encoder ───┤
              Static soil ─► Encoder ───┤
                                        ▼
                              Concatenation
                                        ▼
                              Fusion projection
                                        ▼
                              Temporal pooling
                                        ▼
                                  Regression
    """

    fusion_name = "feature_fusion"

    def __init__(self):

        super().__init__()

        self.encoders = FourModalEncoders()

        self.fusion = nn.Sequential(
            nn.Linear(
                4 * LATENT_DIM,
                LATENT_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(LATENT_DIM),
            nn.Dropout(0.10),
        )

        self.temporal_pool = TemporalPooling()

        self.regressor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    ):

        z_w, z_s, z_d, z_ss = self.encoders(
            weather,
            spectral,
            dynamic_soil,
            static_soil,
        )

        x = torch.cat(
            [
                z_w,
                z_s,
                z_d,
                z_ss,
            ],
            dim=-1,
        )

        fused_sequence = self.fusion(x)

        fused, temporal_weights = self.temporal_pool(
            fused_sequence
        )

        prediction = self.regressor(
            fused
        ).squeeze(-1)

        return {
            "prediction": prediction,
            "fused": fused,
            "fused_sequence": fused_sequence,
            "temporal_weights": temporal_weights,
        }


# =============================================================================
# CROSS-MODAL ATTENTION
# =============================================================================

class CrossModalAttention(nn.Module):
    """
    Cross-modal multi-head attention.

    The four modality representations are treated as four tokens
    at every temporal position.

    Shape:

        B,T,4,D

    ->

        B*T,4,D

    -> MultiheadAttention

    ->

        B,T,4,D
    """

    def __init__(
        self,
        dim=LATENT_DIM,
        heads=4,
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, modalities):

        # list of B,T,D

        x = torch.stack(
            modalities,
            dim=2,
        )

        B, T, M, D = x.shape

        x = x.reshape(
            B * T,
            M,
            D,
        )

        attended, weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
        )

        x = self.norm1(
            x + attended
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        x = x.reshape(
            B,
            T,
            M,
            D,
        )

        # Average modalities after cross-modal interaction.

        fused = x.mean(dim=2)

        # weights:
        # B*T,M,M

        weights = weights.reshape(
            B,
            T,
            M,
            M,
        )

        return fused, weights


# =============================================================================
# CROSS-MODAL ATTENTION FUSION
# =============================================================================

class CrossModalAttentionModel(nn.Module):
    """
    MAST-Fuse style feature-level cross-modal attention.

    This is the primary proposed fusion model.
    """

    fusion_name = "cross_modal_attention"

    def __init__(self):

        super().__init__()

        self.encoders = FourModalEncoders()

        self.cross_attention = CrossModalAttention(
            LATENT_DIM,
            heads=4,
        )

        self.temporal_pool = TemporalPooling()

        self.regressor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    ):

        modalities = self.encoders(
            weather,
            spectral,
            dynamic_soil,
            static_soil,
        )

        fused_sequence, cross_weights = (
            self.cross_attention(
                list(modalities)
            )
        )

        fused, temporal_weights = self.temporal_pool(
            fused_sequence
        )

        prediction = self.regressor(
            fused
        ).squeeze(-1)

        return {
            "prediction": prediction,
            "fused": fused,
            "fused_sequence": fused_sequence,
            "cross_attention": cross_weights,
            "temporal_weights": temporal_weights,
        }


# =============================================================================
# LATE / DECISION-LEVEL FUSION
# =============================================================================

class SingleModalityPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = TemporalPooling()

        self.head = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, z):

        h, weights = self.pool(z)

        y = self.head(h).squeeze(-1)

        return y, weights


class LateFusionModel(nn.Module):
    """
    LATE / DECISION-LEVEL FUSION

    Each modality independently generates a prediction.

        Weather  -> prediction
        Spectral -> prediction
        Soil     -> prediction
        Static   -> prediction

    The predictions are then learned-weight averaged.

    This is fundamentally different from MAST-Fuse because
    the modalities interact only after prediction.
    """

    fusion_name = "late_fusion"

    def __init__(self):

        super().__init__()

        self.encoders = FourModalEncoders()

        self.predictors = nn.ModuleList(
            [
                SingleModalityPredictor()
                for _ in range(4)
            ]
        )

        self.modality_logits = nn.Parameter(
            torch.zeros(4)
        )

    def forward(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    ):

        modalities = self.encoders(
            weather,
            spectral,
            dynamic_soil,
            static_soil,
        )

        predictions = []

        temporal_weights = []

        for encoder_output, predictor in zip(
            modalities,
            self.predictors,
        ):

            prediction, weights = predictor(
                encoder_output
            )

            predictions.append(prediction)

            temporal_weights.append(weights)

        predictions = torch.stack(
            predictions,
            dim=1,
        )

        modality_weights = torch.softmax(
            self.modality_logits,
            dim=0,
        )

        prediction = (
            predictions
            * modality_weights.unsqueeze(0)
        ).sum(dim=1)

        return {
            "prediction": prediction,
            "individual_predictions": predictions,
            "modality_weights": modality_weights,
            "temporal_weights": torch.stack(
                temporal_weights,
                dim=1,
            ),
        }


# =============================================================================
# HYBRID ATTENTION FUSION
# =============================================================================

class HybridAttentionModel(nn.Module):
    """
    HYBRID FUSION

    Combines:

        1. Modality attention
        2. Cross-modal feature interaction
        3. Temporal attention

    This provides a richer fusion mechanism than simple
    feature concatenation.
    """

    fusion_name = "hybrid_attention"

    def __init__(self):

        super().__init__()

        self.encoders = FourModalEncoders()

        self.modality_score = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.cross_attention = CrossModalAttention(
            LATENT_DIM,
            heads=4,
        )

        self.temporal_pool = TemporalPooling()

        self.regressor = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    ):

        modalities = self.encoders(
            weather,
            spectral,
            dynamic_soil,
            static_soil,
        )

        # -------------------------------------------------------------
        # Modality attention
        # -------------------------------------------------------------

        stacked = torch.stack(
            modalities,
            dim=2,
        )

        # B,T,M,D

        scores = self.modality_score(
            stacked
        ).squeeze(-1)

        modality_weights = torch.softmax(
            scores,
            dim=2,
        )

        weighted_modalities = []

        for i in range(4):

            weighted_modalities.append(
                modalities[i]
                * modality_weights[:, :, i:i+1]
            )

        # -------------------------------------------------------------
        # Cross-modal interaction
        # -------------------------------------------------------------

        fused_sequence, cross_weights = (
            self.cross_attention(
                weighted_modalities
            )
        )

        # -------------------------------------------------------------
        # Temporal attention
        # -------------------------------------------------------------

        fused, temporal_weights = (
            self.temporal_pool(
                fused_sequence
            )
        )

        prediction = self.regressor(
            fused
        ).squeeze(-1)

        return {
            "prediction": prediction,
            "fused": fused,
            "fused_sequence": fused_sequence,
            "modality_weights": modality_weights,
            "cross_attention": cross_weights,
            "temporal_weights": temporal_weights,
        }


# =============================================================================
# MODEL FACTORY
# =============================================================================

MODEL_CLASSES = {

    "early_fusion":
        EarlyFusionModel,

    "feature_fusion":
        FeatureFusionModel,

    "cross_modal_attention":
        CrossModalAttentionModel,

    "late_fusion":
        LateFusionModel,

    "hybrid_attention":
        HybridAttentionModel,
}


def create_model(name):

    if name not in MODEL_CLASSES:

        raise ValueError(
            f"Unknown fusion strategy: {name}"
        )

    return MODEL_CLASSES[name]()


# =============================================================================
# LOSS
# =============================================================================

class StableRegressionLoss(nn.Module):
    """
    Smooth L1 / Huber regression loss.

    More robust than pure MSE against occasional
    synthetic-target outliers.
    """

    def __init__(self, beta=0.10):

        super().__init__()

        self.beta = beta

    def forward(self, prediction, target):

        return F.smooth_l1_loss(
            prediction,
            target,
            beta=self.beta,
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_pred - y_true

    mae = np.mean(
        np.abs(error)
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(mse)

    denominator = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if denominator > 0:

        r2 = 1.0 - (
            np.sum(error ** 2)
            / denominator
        )

    else:

        r2 = np.nan

    # Pearson correlation

    if (
        np.std(y_true) > 1e-12
        and
        np.std(y_pred) > 1e-12
    ):

        correlation = np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]

    else:

        correlation = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson": float(correlation),
    }


# =============================================================================
# FORWARD
# =============================================================================

def forward_batch(
    model,
    batch,
):

    weather = batch["weather"].to(
        DEVICE,
        non_blocking=True,
    )

    spectral = batch["spectral"].to(
        DEVICE,
        non_blocking=True,
    )

    dynamic_soil = batch["dynamic_soil"].to(
        DEVICE,
        non_blocking=True,
    )

    static_soil = batch["static_soil"].to(
        DEVICE,
        non_blocking=True,
    )

    target = batch["target"].to(
        DEVICE,
        non_blocking=True,
    )

    output = model(
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    )

    prediction = output["prediction"]

    return prediction, target


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    total_samples = 0

    for batch in loader:

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction, target = forward_batch(
            model,
            batch,
        )

        loss = criterion(
            prediction,
            target,
        )

        if not torch.isfinite(loss):

            raise FloatingPointError(
                "Non-finite training loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        n = target.size(0)

        total_loss += (
            loss.item() * n
        )

        total_samples += n

    return (
        total_loss
        / max(total_samples, 1)
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    losses = []

    targets = []

    predictions = []

    for batch in loader:

        prediction, target = forward_batch(
            model,
            batch,
        )

        loss = criterion(
            prediction,
            target,
        )

        losses.append(
            loss.item()
            * target.size(0)
        )

        targets.extend(
            target.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        predictions.extend(
            prediction.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    n = len(targets)

    loss = (
        sum(losses)
        / max(n, 1)
    )

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    metrics["Loss"] = float(loss)

    return metrics


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_experiment(
    fusion_name,
    train_loader,
    val_loader,
    test_loader,
    seed,
):

    print()
    print("=" * 80)
    print(
        f"FUSION EXPERIMENT: "
        f"{fusion_name.upper()}"
    )
    print("=" * 80)

    set_seed(seed)

    model = create_model(
        fusion_name
    ).to(DEVICE)

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"Parameters : {total_params:,}"
    )

    print(
        f"Trainable  : {trainable_params:,}"
    )

    criterion = StableRegressionLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    best_val_rmse = float("inf")

    best_epoch = 0

    checkpoint_path = (
        MODEL_DIR
        / f"{fusion_name}_best.pt"
    )

    history = []

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_metrics["RMSE"]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["Loss"],
            "val_mae": val_metrics["MAE"],
            "val_rmse": val_metrics["RMSE"],
            "val_r2": val_metrics["R2"],
            "val_pearson": val_metrics["Pearson"],
            "learning_rate":
                optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        if (
            epoch == 1
            or epoch % 5 == 0
            or val_metrics["RMSE"]
            < best_val_rmse
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train={train_loss:.6f} | "
                f"Val RMSE={val_metrics['RMSE']:.6f} | "
                f"Val MAE={val_metrics['MAE']:.6f} | "
                f"Val R²={val_metrics['R2']:.6f}"
            )

        if (
            val_metrics["RMSE"]
            < best_val_rmse
        ):

            best_val_rmse = (
                val_metrics["RMSE"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "fusion_name":
                        fusion_name,

                    "epoch":
                        epoch,

                    "val_metrics":
                        val_metrics,

                    "seed":
                        seed,

                    "config": {
                        "sequence_length":
                            SEQ_LEN,

                        "weather_dim":
                            WEATHER_DIM,

                        "spectral_dim":
                            SPECTRAL_DIM,

                        "dynamic_soil_dim":
                            DYNAMIC_SOIL_DIM,

                        "static_soil_dim":
                            STATIC_SOIL_DIM,

                        "latent_dim":
                            LATENT_DIM,
                    },
                },
                checkpoint_path,
            )

    # -----------------------------------------------------------------
    # Restore best model
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # -----------------------------------------------------------------
    # Test
    # -----------------------------------------------------------------

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
    )

    result = {
        "fusion":
            fusion_name,

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "test_loss":
            test_metrics["Loss"],

        "MAE":
            test_metrics["MAE"],

        "RMSE":
            test_metrics["RMSE"],

        "R2":
            test_metrics["R2"],

        "Pearson":
            test_metrics["Pearson"],
    }

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        RESULT_DIR
        / f"{fusion_name}_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print()
    print(
        f"BEST EPOCH : {best_epoch}"
    )

    print(
        f"TEST MAE   : {test_metrics['MAE']:.6f}"
    )

    print(
        f"TEST RMSE  : {test_metrics['RMSE']:.6f}"
    )

    print(
        f"TEST R²    : {test_metrics['R2']:.6f}"
    )

    print(
        f"TEST Corr. : {test_metrics['Pearson']:.6f}"
    )

    return result


# =============================================================================
# PARAMETER SUMMARY
# =============================================================================

def create_parameter_summary():

    rows = []

    for name in MODEL_CLASSES:

        set_seed(42)

        model = create_model(
            name
        )

        total, trainable = (
            count_parameters(model)
        )

        rows.append(
            {
                "fusion": name,
                "parameters": total,
                "trainable_parameters":
                    trainable,
            }
        )

    df = pd.DataFrame(rows)

    path = (
        RESULT_DIR
        / "fusion_parameter_comparison.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    print()
    print("=" * 80)
    print("PARAMETER COMPARISON")
    print("=" * 80)

    print(
        df.to_string(
            index=False
        )
    )

    return df


# =============================================================================
# PLOTS
# =============================================================================

def plot_metric(
    results_df,
    metric,
    ylabel,
    filename,
    higher_is_better=False,
):

    df = results_df.copy()

    df = df.sort_values(
        metric,
        ascending=not higher_is_better,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        df["fusion"],
        df[metric],
    )

    plt.ylabel(ylabel)

    plt.xlabel(
        "Fusion strategy"
    )

    plt.title(
        f"Fusion Strategy Comparison — {ylabel}"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / filename,
        dpi=300,
    )

    plt.close()


def create_plots(results_df):

    plot_metric(
        results_df,
        "RMSE",
        "RMSE",
        "fusion_rmse.png",
        higher_is_better=False,
    )

    plot_metric(
        results_df,
        "MAE",
        "MAE",
        "fusion_mae.png",
        higher_is_better=False,
    )

    plot_metric(
        results_df,
        "R2",
        "R²",
        "fusion_r2.png",
        higher_is_better=True,
    )

    # -------------------------------------------------------------
    # Combined normalized comparison
    # -------------------------------------------------------------

    df = results_df.copy()

    # Normalize errors so lower is better.

    for metric in [
        "MAE",
        "RMSE",
    ]:

        min_v = df[metric].min()

        max_v = df[metric].max()

        if max_v > min_v:

            df[
                metric + "_score"
            ] = (
                1
                - (
                    df[metric]
                    - min_v
                )
                / (
                    max_v
                    - min_v
                )
            )

        else:

            df[
                metric + "_score"
            ] = 1.0

    # R2 higher is better.

    min_r2 = df["R2"].min()

    max_r2 = df["R2"].max()

    if max_r2 > min_r2:

        df["R2_score"] = (
            df["R2"] - min_r2
        ) / (
            max_r2 - min_r2
        )

    else:

        df["R2_score"] = 1.0

    df["overall_score"] = (
        df["MAE_score"]
        + df["RMSE_score"]
        + df["R2_score"]
    ) / 3.0

    df = df.sort_values(
        "overall_score",
        ascending=False,
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        df["fusion"],
        df["overall_score"],
    )

    plt.ylabel(
        "Normalized overall score"
    )

    plt.xlabel(
        "Fusion strategy"
    )

    plt.title(
        "Overall Fusion Strategy Comparison"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "fusion_comparison.png",
        dpi=300,
    )

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("MAST-Fuse / MuSTIPest-V3")
    print("FUSION STRATEGY COMPARISON")
    print("=" * 80)

    print(
        f"Device     : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"Batch      : {BATCH_SIZE}"
    )

    print(
        f"Epochs     : {EPOCHS}"
    )

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    print(
        f"Latent dimension : {LATENT_DIM}"
    )

    print(
        f"Output     : {RESULT_DIR}"
    )

    # -----------------------------------------------------------------
    # Parameter comparison before training
    # -----------------------------------------------------------------

    create_parameter_summary()

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    train_loader, val_loader, test_loader = (
        create_loaders()
    )

    # -----------------------------------------------------------------
    # Sanity check
    # -----------------------------------------------------------------

    batch = next(
        iter(train_loader)
    )

    print()
    print("=" * 80)
    print("INPUT SANITY CHECK")
    print("=" * 80)

    for key, value in batch.items():

        print(
            f"{key:15s}: "
            f"{tuple(value.shape)} "
            f"{value.dtype}"
        )

    # -----------------------------------------------------------------
    # Test all model forward passes
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("MODEL FORWARD-PASS TEST")
    print("=" * 80)

    for name in MODEL_CLASSES:

        set_seed(42)

        model = create_model(
            name
        ).to(DEVICE)

        model.eval()

        with torch.no_grad():

            output = model(
                batch["weather"].to(DEVICE),
                batch["spectral"].to(DEVICE),
                batch["dynamic_soil"].to(DEVICE),
                batch["static_soil"].to(DEVICE),
            )

        prediction = output[
            "prediction"
        ]

        print(
            f"{name:25s} "
            f"prediction={tuple(prediction.shape)} "
            f"finite={torch.isfinite(prediction).all().item()}"
        )

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------

    results = []

    for fusion_name in MODEL_CLASSES:

        for seed in SEEDS:

            result = run_experiment(
                fusion_name,
                train_loader,
                val_loader,
                test_loader,
                seed,
            )

            results.append(
                result
            )

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_csv = (
        RESULT_DIR
        / "fusion_comparison_results.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False,
    )

    results_json = (
        RESULT_DIR
        / "fusion_comparison_results.json"
    )

    with open(
        results_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL FUSION COMPARISON")
    print("=" * 80)

    summary_columns = [
        "fusion",
        "parameters",
        "best_epoch",
        "MAE",
        "RMSE",
        "R2",
        "Pearson",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------

    ranking = results_df.sort_values(
        "RMSE",
        ascending=True,
    ).reset_index(
        drop=True
    )

    ranking.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    ranking_path = (
        RESULT_DIR
        / "fusion_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("RANKING BY TEST RMSE")
    print("=" * 80)

    print(
        ranking[
            [
                "Rank",
                "fusion",
                "RMSE",
                "MAE",
                "R2",
            ]
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    create_plots(
        results_df
    )

    # -----------------------------------------------------------------
    # Save experiment configuration
    # -----------------------------------------------------------------

    configuration = {

        "sequence_length":
            SEQ_LEN,

        "weather_dim":
            WEATHER_DIM,

        "spectral_dim":
            SPECTRAL_DIM,

        "dynamic_soil_dim":
            DYNAMIC_SOIL_DIM,

        "static_soil_dim":
            STATIC_SOIL_DIM,

        "latent_dim":
            LATENT_DIM,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRADIENT_CLIP,

        "seeds":
            SEEDS,

        "fusion_strategies":
            list(MODEL_CLASSES.keys()),
    }

    with open(
        RESULT_DIR
        / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            configuration,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("FUSION COMPARISON COMPLETED")
    print("=" * 80)

    print(
        f"Results : {RESULT_DIR}"
    )

    print()
    print("Generated:")
    print(
        "  fusion_comparison_results.csv"
    )
    print(
        "  fusion_comparison_results.json"
    )
    print(
        "  fusion_ranking.csv"
    )
    print(
        "  fusion_parameter_comparison.csv"
    )
    print(
        "  fusion_rmse.png"
    )
    print(
        "  fusion_mae.png"
    )
    print(
        "  fusion_r2.png"
    )
    print(
        "  fusion_comparison.png"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
