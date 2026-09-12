Author - Dipanwita Thakur

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    load_all_data,
    create_dataloaders,
)

from model import MASTFuse

from losses import StableRegressionLoss


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

BATCH_SIZE = 32

EPOCHS = 150

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0

PATIENCE = 20

MIN_DELTA = 1e-5

FUSION_DIM = 128

ATTENTION_HEADS = 4

FF_DIM = 256

DROPOUT = 0.10

NUM_WORKERS = 0

PIN_MEMORY = True

OUTPUT_DIR = Path("results") / "ablations"


# =============================================================================
# CURRENT MAST-FUSE MODALITIES
# =============================================================================

MODALITIES = [
    "weather",
    "spectral",
    "dynamic_soil",
    "static_soil",
]


MODALITY_DISPLAY_NAMES = {
    "weather": "Weather",
    "spectral": "Spectral",
    "dynamic_soil": "Dynamic Soil",
    "static_soil": "Static Soil",
}


MODALITY_DIMS = {
    "weather": 17,
    "spectral": 3,
    "dynamic_soil": 4,
    "static_soil": 28,
}


# =============================================================================
# ARCHITECTURE OPTIONS
# =============================================================================

# These are the architecture names used by the MAST-Fuse model.
#
# If your current model.py supports additional names, add them here.

TEMPORAL_ENCODERS = [
    "tcn",
    "gru",
    "bilstm",
    "transformer",
]


FUSION_TYPES = [
    "cross_attention",
    "gated",
    "concat",
    "mean",
]


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

    # Reproducibility is preferable for an ablation study.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DEVICE
# =============================================================================

def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# DIRECTORY
# =============================================================================

def create_directories() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "checkpoints").mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DIR / "summary").mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# MODALITY COMBINATIONS
# =============================================================================

def generate_modality_combinations() -> List[Dict[str, bool]]:
    """
    Generate all 16 combinations of four modalities.

    Includes:
        0000
        0001
        ...
        1111

    The all-disabled configuration is excluded because the model cannot
    perform meaningful prediction without an input modality.
    """

    combinations = []

    for bits in itertools.product([False, True], repeat=4):

        config = {
            "weather": bits[0],
            "spectral": bits[1],
            "dynamic_soil": bits[2],
            "static_soil": bits[3],
        }

        if not any(bits):
            continue

        combinations.append(config)

    return combinations


# =============================================================================
# MODALITY LABEL
# =============================================================================

def modality_label(
    config: Dict[str, bool],
) -> str:

    enabled = [
        name
        for name in MODALITIES
        if config[name]
    ]

    disabled = [
        name
        for name in MODALITIES
        if not config[name]
    ]

    if len(enabled) == 4:
        return "full_model"

    if len(enabled) == 1:
        return f"only_{enabled[0]}"

    return "_".join(enabled)


# =============================================================================
# MODALITY BIT CODE
# =============================================================================

def modality_code(
    config: Dict[str, bool],
) -> str:

    return "".join(
        "1" if config[name] else "0"
        for name in MODALITIES
    )


# =============================================================================
# ARCHITECTURE LABEL
# =============================================================================

def architecture_label(
    temporal_encoder: str,
    fusion_type: str,
) -> str:

    return (
        f"{temporal_encoder}"
        f"__"
        f"{fusion_type}"
    )


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def build_model(
    modality_config: Dict[str, bool],
    temporal_encoder: str = "tcn",
    fusion_type: str = "cross_attention",
) -> nn.Module:
    """
    Construct MASTFuse using the current project API.

    Current tested dimensions:

        weather       = 17
        spectral      = 3
        dynamic soil  = 4
        static soil   = 28
    """

    model = MASTFuse(
        weather_dim=17,
        spectral_dim=3,
        dynamic_soil_dim=4,
        static_soil_dim=28,

        sequence_length=21,

        fusion_dim=FUSION_DIM,

        num_heads=ATTENTION_HEADS,

        ff_dim=FF_DIM,

        dropout=DROPOUT,

        use_weather=modality_config["weather"],
        use_spectral=modality_config["spectral"],
        use_dynamic_soil=modality_config["dynamic_soil"],
        use_static_soil=modality_config["static_soil"],

        temporal_encoder=temporal_encoder,

        fusion_type=fusion_type,
    )

    return model


# =============================================================================
# MODEL OUTPUT EXTRACTION
# =============================================================================

def extract_prediction(
    output: Any,
) -> torch.Tensor:
    """
    Supports both:

        model(...) -> tensor

    and

        model(...) -> dictionary

    and tuple/list outputs.
    """

    if torch.is_tensor(output):
        prediction = output

    elif isinstance(output, dict):

        possible_keys = [
            "prediction",
            "predictions",
            "output",
            "y_hat",
            "pred",
        ]

        prediction = None

        for key in possible_keys:

            if key in output:
                prediction = output[key]
                break

        if prediction is None:

            raise ValueError(
                "Model returned a dictionary but no prediction "
                f"key was found. Keys: {list(output.keys())}"
            )

    elif isinstance(output, (tuple, list)):

        if len(output) == 0:
            raise ValueError(
                "Model returned an empty tuple/list."
            )

        prediction = output[0]

    else:

        raise TypeError(
            "Unsupported model output type: "
            f"{type(output)}"
        )

    return prediction.reshape(-1)


# =============================================================================
# BATCH TO DEVICE
# =============================================================================

def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:

    result = {}

    for key, value in batch.items():

        if torch.is_tensor(value):

            result[key] = value.to(
                device,
                non_blocking=True,
            )

        else:

            result[key] = value

    return result


# =============================================================================
# MODEL INPUT
# =============================================================================

def prepare_model_inputs(
    batch: Dict[str, torch.Tensor],
    modality_config: Dict[str, bool],
) -> Dict[str, torch.Tensor]:
    """
    Keep only the modalities required by the experiment.

    Static soil can appear either as:

        (B, 21, 28)

    or:

        (B, 28)

    The current model handles the tested formats.
    """

    inputs = {}

    if modality_config["weather"]:

        inputs["weather"] = batch["weather"]

    if modality_config["spectral"]:

        inputs["spectral"] = batch["spectral"]

    if modality_config["dynamic_soil"]:

        inputs["dynamic_soil"] = batch["dynamic_soil"]

    if modality_config["static_soil"]:

        inputs["static_soil"] = batch["static_soil"]

    return inputs


# =============================================================================
# FORWARD
# =============================================================================

def forward_model(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    modality_config: Dict[str, bool],
) -> torch.Tensor:

    inputs = prepare_model_inputs(
        batch,
        modality_config,
    )

    output = model(
        **inputs
    )

    return extract_prediction(output)


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(-1)

    target = np.asarray(
        target,
        dtype=np.float64,
    ).reshape(-1)

    valid = (
        np.isfinite(prediction)
        &
        np.isfinite(target)
    )

    prediction = prediction[valid]

    target = target[valid]

    if len(target) == 0:

        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "n": 0,
        }

    error = prediction - target

    mae = np.mean(
        np.abs(error)
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(mse)

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (target - np.mean(target)) ** 2
    )

    if ss_tot > 1e-12:

        r2 = 1.0 - (
            ss_res / ss_tot
        )

    else:

        r2 = float("nan")

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "n": int(len(target)),
    }


# =============================================================================
# ONE EPOCH — TRAIN
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    modality_config: Dict[str, bool],
    device: torch.device,
) -> float:

    model.train()

    running_loss = 0.0

    total_samples = 0

    for batch in loader:

        batch = move_batch_to_device(
            batch,
            device,
        )

        target = batch["target"].float().reshape(-1)

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = forward_model(
            model,
            batch,
            modality_config,
        )

        loss = criterion(
            prediction,
            target,
        )

        if not torch.isfinite(loss):

            raise FloatingPointError(
                "Non-finite training loss detected."
            )

        loss.backward()

        if GRADIENT_CLIP is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

        optimizer.step()

        batch_size = target.shape[0]

        running_loss += (
            loss.detach().item()
            *
            batch_size
        )

        total_samples += batch_size

    if total_samples == 0:

        return float("nan")

    return (
        running_loss
        /
        total_samples
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    modality_config: Dict[str, bool],
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    all_predictions = []

    all_targets = []

    running_loss = 0.0

    total_samples = 0

    for batch in loader:

        batch = move_batch_to_device(
            batch,
            device,
        )

        target = batch["target"].float().reshape(-1)

        prediction = forward_model(
            model,
            batch,
            modality_config,
        )

        loss = criterion(
            prediction,
            target,
        )

        if not torch.isfinite(loss):

            raise FloatingPointError(
                "Non-finite evaluation loss detected."
            )

        batch_size = target.shape[0]

        running_loss += (
            loss.item()
            *
            batch_size
        )

        total_samples += batch_size

        all_predictions.append(
            prediction.detach()
            .cpu()
            .numpy()
        )

        all_targets.append(
            target.detach()
            .cpu()
            .numpy()
        )

    prediction = np.concatenate(
        all_predictions
    )

    target = np.concatenate(
        all_targets
    )

    metrics = calculate_metrics(
        prediction,
        target,
    )

    metrics["loss"] = (
        running_loss
        /
        max(total_samples, 1)
    )

    return metrics


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    experiment: Dict[str, Any],
) -> None:

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "metrics":
            metrics,

        "experiment":
            experiment,
    }

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_single_experiment(
    experiment_name: str,
    modality_config: Dict[str, bool],
    temporal_encoder: str,
    fusion_type: str,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:

    set_seed(seed)

    print()
    print("-" * 80)
    print(
        f"EXPERIMENT: {experiment_name}"
    )
    print("-" * 80)

    print(
        f"Modalities       : "
        f"{modality_label(modality_config)}"
    )

    print(
        f"Modality code    : "
        f"{modality_code(modality_config)}"
    )

    print(
        f"Temporal encoder : "
        f"{temporal_encoder}"
    )

    print(
        f"Fusion           : "
        f"{fusion_type}"
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = build_model(
        modality_config=modality_config,
        temporal_encoder=temporal_encoder,
        fusion_type=fusion_type,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Parameters       : "
        f"{parameters:,}"
    )

    print(
        f"Trainable        : "
        f"{trainable_parameters:,}"
    )

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    criterion = StableRegressionLoss(
        smooth_l1_weight=1.0,
        mse_weight=0.25,
        range_penalty_weight=0.05,
    ).to(device)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7,
        min_lr=1e-7,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    best_val_loss = float("inf")

    best_epoch = 0

    patience_counter = 0

    best_state = None

    start_time = time.time()

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            modality_config=modality_config,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            modality_config=modality_config,
            device=device,
        )

        val_loss = val_metrics["loss"]

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "learning_rate": current_lr,
        }

        history.append(
            epoch_record
        )

        improved = (
            val_loss
            <
            best_val_loss - MIN_DELTA
        )

        if improved:

            best_val_loss = val_loss

            best_epoch = epoch

            patience_counter = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:

            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f} | "
                f"MAE {val_metrics['mae']:.6f} | "
                f"RMSE {val_metrics['rmse']:.6f} | "
                f"R² {val_metrics['r2']:.6f} | "
                f"LR {current_lr:.2e}"
            )

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch "
                f"{epoch}."
            )

            break

    # -------------------------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # -------------------------------------------------------------------------
    # Final validation/test
    # -------------------------------------------------------------------------

    val_metrics = evaluate(
        model=model,
        loader=loaders["val"],
        criterion=criterion,
        modality_config=modality_config,
        device=device,
    )

    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        modality_config=modality_config,
        device=device,
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    # -------------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------------

    checkpoint_path = (
        OUTPUT_DIR
        /
        "checkpoints"
        /
        f"{experiment_name}_seed{seed}.pt"
    )

    experiment_info = {
        "experiment_name":
            experiment_name,

        "seed":
            seed,

        "modality_config":
            modality_config,

        "temporal_encoder":
            temporal_encoder,

        "fusion_type":
            fusion_type,

        "parameters":
            parameters,

        "trainable_parameters":
            trainable_parameters,
    }

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        best_epoch,
        test_metrics,
        experiment_info,
    )

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    result = {
        "experiment": experiment_name,

        "seed": seed,

        "modality_code":
            modality_code(modality_config),

        "modalities":
            modality_label(modality_config),

        "weather":
            modality_config["weather"],

        "spectral":
            modality_config["spectral"],

        "dynamic_soil":
            modality_config["dynamic_soil"],

        "static_soil":
            modality_config["static_soil"],

        "temporal_encoder":
            temporal_encoder,

        "fusion":
            fusion_type,

        "parameters":
            parameters,

        "trainable_parameters":
            trainable_parameters,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "val_loss":
            val_metrics["loss"],

        "val_mae":
            val_metrics["mae"],

        "val_rmse":
            val_metrics["rmse"],

        "val_r2":
            val_metrics["r2"],

        "test_loss":
            test_metrics["loss"],

        "test_mae":
            test_metrics["mae"],

        "test_rmse":
            test_metrics["rmse"],

        "test_r2":
            test_metrics["r2"],

        "training_time_sec":
            elapsed,

        "checkpoint":
            str(checkpoint_path),
    }

    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    history_path = (
        OUTPUT_DIR
        /
        f"history_{experiment_name}_seed{seed}.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    print()
    print(
        f"BEST EPOCH : {best_epoch}"
    )

    print(
        f"VAL MAE    : {val_metrics['mae']:.6f}"
    )

    print(
        f"VAL RMSE   : {val_metrics['rmse']:.6f}"
    )

    print(
        f"VAL R²     : {val_metrics['r2']:.6f}"
    )

    print(
        f"TEST MAE   : {test_metrics['mae']:.6f}"
    )

    print(
        f"TEST RMSE  : {test_metrics['rmse']:.6f}"
    )

    print(
        f"TEST R²    : {test_metrics['r2']:.6f}"
    )

    return result


# =============================================================================
# CREATE DATA LOADERS
# =============================================================================

def create_loaders(
    data: Any,
    modality_config: Dict[str, bool],
) -> Dict[str, DataLoader]:
    """
    Use the current dataset.py API.

    Based on the current MAST-Fuse dataset implementation:

        create_dataloaders(
            batch_size,
            seed,
            use_weather,
            use_spectral,
            use_dynamic_soil,
            use_static_soil
        )

    If your dataset.py has already loaded data internally, this function
    directly uses create_dataloaders().
    """

    try:

        loaders = create_dataloaders(
            BATCH_SIZE,
            SEED,
            modality_config["weather"],
            modality_config["spectral"],
            modality_config["dynamic_soil"],
            modality_config["static_soil"],
        )

    except TypeError:

        # More explicit keyword-compatible version.
        loaders = create_dataloaders(
            batch_size=BATCH_SIZE,
            seed=SEED,
            use_weather=modality_config["weather"],
            use_spectral=modality_config["spectral"],
            use_dynamic_soil=modality_config["dynamic_soil"],
            use_static_soil=modality_config["static_soil"],
        )

    # -------------------------------------------------------------------------
    # Normalize returned structure
    # -------------------------------------------------------------------------

    if isinstance(loaders, dict):

        # Common names.
        if (
            "train" in loaders
            and "val" in loaders
            and "test" in loaders
        ):

            return {
                "train": loaders["train"],
                "val": loaders["val"],
                "test": loaders["test"],
            }

        if (
            "train_loader" in loaders
            and "val_loader" in loaders
            and "test_loader" in loaders
        ):

            return {
                "train":
                    loaders["train_loader"],

                "val":
                    loaders["val_loader"],

                "test":
                    loaders["test_loader"],
            }

    # Tuple/list format.
    if isinstance(
        loaders,
        (tuple, list)
    ):

        if len(loaders) >= 3:

            return {
                "train": loaders[0],
                "val": loaders[1],
                "test": loaders[2],
            }

    raise TypeError(
        "Could not interpret create_dataloaders() output. "
        f"Received type: {type(loaders)}"
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results: List[Dict[str, Any]],
    filename: str,
) -> None:

    if not results:
        return

    path = (
        OUTPUT_DIR
        /
        filename
    )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        path,
        index=False,
    )

    json_path = path.with_suffix(
        ".json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            default=str,
        )

    print()
    print(
        f"Saved: {path}"
    )


# =============================================================================
# MODALITY ABLATIONS
# =============================================================================

def run_modality_ablations(
    data: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print("MODALITY ABLATION STUDY")
    print("=" * 80)

    print()
    print(
        "Testing all 15 non-empty modality combinations."
    )

    results = []

    combinations = (
        generate_modality_combinations()
    )

    for index, config in enumerate(
        combinations,
        start=1,
    ):

        name = (
            f"M{index:02d}_"
            f"{modality_code(config)}_"
            f"{modality_label(config)}"
        )

        try:

            loaders = create_loaders(
                data,
                config,
            )

            result = run_single_experiment(
                experiment_name=name,
                modality_config=config,
                temporal_encoder="tcn",
                fusion_type="cross_attention",
                loaders=loaders,
                device=device,
                seed=SEED,
            )

            results.append(
                result
            )

        except Exception as exc:

            print()
            print(
                f"FAILED: {name}"
            )

            print(
                f"Reason: {exc}"
            )

            results.append({
                "experiment": name,
                "status": "FAILED",
                "error": str(exc),
                "modalities":
                    modality_label(config),
                "modality_code":
                    modality_code(config),
            })

    save_results(
        results,
        "modality_ablation_results.csv",
    )

    return results


# =============================================================================
# ARCHITECTURE ABLATIONS
# =============================================================================

def run_architecture_ablations(
    data: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print("ARCHITECTURE ABLATION STUDY")
    print("=" * 80)

    print()
    print(
        "Temporal encoders:"
    )

    for encoder in TEMPORAL_ENCODERS:
        print(
            f"  - {encoder}"
        )

    print()
    print(
        "Fusion mechanisms:"
    )

    for fusion in FUSION_TYPES:
        print(
            f"  - {fusion}"
        )

    print()
    print(
        f"Total architecture combinations: "
        f"{len(TEMPORAL_ENCODERS) * len(FUSION_TYPES)}"
    )

    results = []

    full_config = {
        "weather": True,
        "spectral": True,
        "dynamic_soil": True,
        "static_soil": True,
    }

    loaders = create_loaders(
        data,
        full_config,
    )

    index = 1

    for temporal_encoder in TEMPORAL_ENCODERS:

        for fusion_type in FUSION_TYPES:

            name = (
                f"A{index:02d}_"
                f"{temporal_encoder}_"
                f"{fusion_type}"
            )

            try:

                result = run_single_experiment(
                    experiment_name=name,
                    modality_config=full_config,
                    temporal_encoder=temporal_encoder,
                    fusion_type=fusion_type,
                    loaders=loaders,
                    device=device,
                    seed=SEED,
                )

                results.append(
                    result
                )

            except Exception as exc:

                print()
                print(
                    f"FAILED: {name}"
                )

                print(
                    f"Reason: {exc}"
                )

                results.append({
                    "experiment": name,
                    "status": "FAILED",
                    "error": str(exc),
                    "temporal_encoder":
                        temporal_encoder,
                    "fusion":
                        fusion_type,
                })

            index += 1

    save_results(
        results,
        "architecture_ablation_results.csv",
    )

    return results


# =============================================================================
# COMBINED ABLATIONS
# =============================================================================

def run_combined_ablations(
    data: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print("COMBINED MODALITY × ARCHITECTURE ABLATION")
    print("=" * 80)

    modality_combinations = (
        generate_modality_combinations()
    )

    architecture_combinations = list(
        itertools.product(
            TEMPORAL_ENCODERS,
            FUSION_TYPES,
        )
    )

    total = (
        len(modality_combinations)
        *
        len(architecture_combinations)
    )

    print()
    print(
        f"Modality combinations     : "
        f"{len(modality_combinations)}"
    )

    print(
        f"Architecture combinations : "
        f"{len(architecture_combinations)}"
    )

    print(
        f"Total experiments         : "
        f"{total}"
    )

    print()
    print(
        "WARNING: This is the complete factorial study."
    )

    results = []

    index = 1

    for modality_config in modality_combinations:

        loaders = None

        try:

            loaders = create_loaders(
                data,
                modality_config,
            )

        except Exception as exc:

            print(
                f"Could not create loaders for "
                f"{modality_label(modality_config)}: "
                f"{exc}"
            )

            continue

        for (
            temporal_encoder,
            fusion_type,
        ) in architecture_combinations:

            name = (
                f"C{index:03d}_"
                f"{modality_code(modality_config)}_"
                f"{temporal_encoder}_"
                f"{fusion_type}"
            )

            try:

                result = run_single_experiment(
                    experiment_name=name,
                    modality_config=modality_config,
                    temporal_encoder=temporal_encoder,
                    fusion_type=fusion_type,
                    loaders=loaders,
                    device=device,
                    seed=SEED,
                )

                results.append(
                    result
                )

            except Exception as exc:

                print()
                print(
                    f"FAILED: {name}"
                )

                print(
                    f"Reason: {exc}"
                )

                results.append({
                    "experiment": name,
                    "status": "FAILED",
                    "error": str(exc),

                    "modality_code":
                        modality_code(
                            modality_config
                        ),

                    "modalities":
                        modality_label(
                            modality_config
                        ),

                    "temporal_encoder":
                        temporal_encoder,

                    "fusion":
                        fusion_type,
                })

            index += 1

    save_results(
        results,
        "combined_ablation_results.csv",
    )

    return results


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def create_summary_tables() -> None:

    # -------------------------------------------------------------------------
    # Modality summary
    # -------------------------------------------------------------------------

    modality_file = (
        OUTPUT_DIR
        /
        "modality_ablation_results.csv"
    )

    if modality_file.exists():

        df = pd.read_csv(
            modality_file
        )

        successful = df[
            df.get(
                "status",
                pd.Series(
                    ["OK"] * len(df)
                )
            )
            != "FAILED"
        ].copy()

        if not successful.empty:

            columns = [
                c
                for c in [
                    "experiment",
                    "modality_code",
                    "modalities",
                    "weather",
                    "spectral",
                    "dynamic_soil",
                    "static_soil",
                    "test_mae",
                    "test_rmse",
                    "test_r2",
                    "parameters",
                    "best_epoch",
                ]
                if c in successful.columns
            ]

            successful[
                columns
            ].sort_values(
                "test_rmse"
            ).to_csv(
                OUTPUT_DIR
                /
                "summary"
                /
                "modality_summary.csv",
                index=False,
            )

    # -------------------------------------------------------------------------
    # Architecture summary
    # -------------------------------------------------------------------------

    architecture_file = (
        OUTPUT_DIR
        /
        "architecture_ablation_results.csv"
    )

    if architecture_file.exists():

        df = pd.read_csv(
            architecture_file
        )

        successful = df[
            df.get(
                "status",
                pd.Series(
                    ["OK"] * len(df)
                )
            )
            != "FAILED"
        ].copy()

        if not successful.empty:

            columns = [
                c
                for c in [
                    "experiment",
                    "temporal_encoder",
                    "fusion",
                    "test_mae",
                    "test_rmse",
                    "test_r2",
                    "parameters",
                    "best_epoch",
                ]
                if c in successful.columns
            ]

            successful[
                columns
            ].sort_values(
                "test_rmse"
            ).to_csv(
                OUTPUT_DIR
                /
                "summary"
                /
                "architecture_summary.csv",
                index=False,
            )

    # -------------------------------------------------------------------------
    # Combined summary
    # -------------------------------------------------------------------------

    combined_file = (
        OUTPUT_DIR
        /
        "combined_ablation_results.csv"
    )

    if combined_file.exists():

        df = pd.read_csv(
            combined_file
        )

        successful = df[
            df.get(
                "status",
                pd.Series(
                    ["OK"] * len(df)
                )
            )
            != "FAILED"
        ].copy()

        if not successful.empty:

            columns = [
                c
                for c in [
                    "experiment",
                    "modality_code",
                    "modalities",
                    "temporal_encoder",
                    "fusion",
                    "test_mae",
                    "test_rmse",
                    "test_r2",
                    "parameters",
                    "best_epoch",
                ]
                if c in successful.columns
            ]

            successful[
                columns
            ].sort_values(
                "test_rmse"
            ).to_csv(
                OUTPUT_DIR
                /
                "summary"
                /
                "combined_summary.csv",
                index=False,
            )


# =============================================================================
# PRINT FINAL BEST RESULTS
# =============================================================================

def print_best_results() -> None:

    print()
    print("=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    files = [
        (
            "MODALITY",
            "modality_ablation_results.csv",
        ),
        (
            "ARCHITECTURE",
            "architecture_ablation_results.csv",
        ),
        (
            "COMBINED",
            "combined_ablation_results.csv",
        ),
    ]

    for title, filename in files:

        path = (
            OUTPUT_DIR
            /
            filename
        )

        if not path.exists():
            continue

        df = pd.read_csv(
            path
        )

        if "test_rmse" not in df.columns:
            continue

        df = df[
            np.isfinite(
                pd.to_numeric(
                    df["test_rmse"],
                    errors="coerce",
                )
            )
        ]

        if df.empty:
            continue

        best = df.loc[
            df["test_rmse"].idxmin()
        ]

        print()
        print(
            f"{title} BEST MODEL"
        )

        print(
            f"  Experiment : "
            f"{best.get('experiment', '')}"
        )

        if "modalities" in best:

            print(
                f"  Modalities  : "
                f"{best['modalities']}"
            )

        if "temporal_encoder" in best:

            print(
                f"  Temporal    : "
                f"{best['temporal_encoder']}"
            )

        if "fusion" in best:

            print(
                f"  Fusion      : "
                f"{best['fusion']}"
            )

        print(
            f"  Test MAE    : "
            f"{best.get('test_mae', np.nan):.6f}"
        )

        print(
            f"  Test RMSE   : "
            f"{best.get('test_rmse', np.nan):.6f}"
        )

        print(
            f"  Test R²     : "
            f"{best.get('test_r2', np.nan):.6f}"
        )


# =============================================================================
# DATASET SUMMARY
# =============================================================================

def print_dataset_information(
    data: Any,
) -> None:

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    if isinstance(data, dict):

        print(
            "Dataset returned as dictionary."
        )

        for key, value in data.items():

            try:

                print(
                    f"  {key}: "
                    f"{type(value).__name__}"
                )

            except Exception:
                pass

    elif isinstance(
        data,
        (tuple, list),
    ):

        print(
            f"Dataset returned as "
            f"{type(data).__name__} "
            f"with {len(data)} elements."
        )

    else:

        print(
            f"Dataset type: "
            f"{type(data).__name__}"
        )


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "MAST-Fuse modality and "
            "architecture ablation study."
        )
    )

    parser.add_argument(
        "--study",
        type=str,
        default="modality",
        choices=[
            "modality",
            "architecture",
            "combined",
            "all",
        ],
        help=(
            "Ablation study to run."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=(
            "Number of training epochs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Batch size."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=(
            "Learning rate."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=(
            "Early stopping patience."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            "Random seed."
        ),
    )

    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help=(
            "Do not save model checkpoints."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    global EPOCHS
    global BATCH_SIZE
    global LEARNING_RATE
    global PATIENCE
    global SEED

    args = parse_arguments()

    EPOCHS = args.epochs

    BATCH_SIZE = args.batch_size

    LEARNING_RATE = args.lr

    PATIENCE = args.patience

    SEED = args.seed

    set_seed(
        SEED
    )

    create_directories()

    device = get_device()

    print()
    print("=" * 80)
    print("MAST-Fuse / MuSTIPest-V3")
    print("MODALITY + ARCHITECTURE ABLATION STUDY")
    print("=" * 80)

    print(
        f"Device         : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory     : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Epochs         : {EPOCHS}"
    )

    print(
        f"Learning rate  : {LEARNING_RATE}"
    )

    print(
        f"Seed           : {SEED}"
    )

    print(
        f"Study          : {args.study}"
    )

    print(
        f"Output         : "
        f"{OUTPUT_DIR.resolve()}"
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    data = load_all_data()

    print_dataset_information(
        data
    )

    # -------------------------------------------------------------------------
    # Run requested studies
    # -------------------------------------------------------------------------

    if args.study in {
        "modality",
        "all",
    }:

        run_modality_ablations(
            data,
            device,
        )

    if args.study in {
        "architecture",
        "all",
    }:

        run_architecture_ablations(
            data,
            device,
        )

    if args.study in {
        "combined",
        "all",
    }:

        run_combined_ablations(
            data,
            device,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    create_summary_tables()

    print_best_results()

    print()
    print("=" * 80)
    print("ABLATION STUDY COMPLETED")
    print("=" * 80)

    print()
    print(
        "Results directory:"
    )

    print(
        f"  {OUTPUT_DIR.resolve()}"
    )

    print()
    print(
        "Important result files:"
    )

    print(
        "  modality_ablation_results.csv"
    )

    print(
        "  architecture_ablation_results.csv"
    )

    print(
        "  combined_ablation_results.csv"
    )

    print()
    print(
        "Summary files:"
    )

    print(
        "  summary/modality_summary.csv"
    )

    print(
        "  summary/architecture_summary.csv"
    )

    print(
        "  summary/combined_summary.csv"
    )

    print()
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
