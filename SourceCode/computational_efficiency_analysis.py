"""
MAST-Fuse Computational Efficiency and Interpretability Analysis
================================================================

Experiments:
1. Per-sample inference latency:
      - GPU, batch size = 1
      - CPU, batch size = 1
      - approximately 1,000 unseen test samples
      - mean +/- std

2. Peak inference memory:
      - GPU peak allocated memory
      - CPU peak RSS memory when psutil is available
      - model size on disk

3. Training time:
      - time per epoch
      - total training time
      - five-seed mean +/- std

4. Learned interpretability:
      - mean modality gate weights alpha_m
      - mean temporal attention weights beta_t
      - std over test samples

The script is designed to work with the existing:
    dataset.py
    model.py
    fusion.py
    encoders.py

Expected checkpoint:
    best_model.pt

IMPORTANT:
The checkpoint must correspond to the FULL four-modality MAST-Fuse model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from dataset import load_all_data, MuSTIPestDataset, create_dataloader
from model import MASTFuse


# =====================================================================
# Configuration
# =====================================================================

SEEDS = [42, 123, 456, 789, 1011]

BATCH_SIZE = 32
SEQ_LEN = 21
WORKERS = 0

FUSION_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.10

NUM_LATENCY_SAMPLES = 1000
GPU_WARMUP = 50
CPU_WARMUP = 20

# Number of timed repetitions per sample.
# With batch size 1, using one forward pass per sample gives a direct
# per-sample deployment latency estimate.
LATENCY_REPEATS = 1

OUTPUT_DIR = Path("computational_efficiency_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Device
# =====================================================================

CUDA_AVAILABLE = torch.cuda.is_available()

GPU_DEVICE = torch.device("cuda") if CUDA_AVAILABLE else None
CPU_DEVICE = torch.device("cpu")


# =====================================================================
# Dataset
# =====================================================================

def load_test_dataset():
    """
    Load the existing train/validation/test split and construct the
    FULL four-modality test dataset.
    """

    data = load_all_data()

    if not isinstance(data, (tuple, list)) or len(data) != 3:
        raise ValueError(
            "load_all_data() must return "
            "(train, validation, test)"
        )

    train_data, val_data, test_data = data

    test_dataset = MuSTIPestDataset(
        test_data,
        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,
    )

    return test_dataset


def create_single_sample_loader(dataset):
    """
    Batch-size-1 loader for latency experiments.
    """

    return create_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=WORKERS,
    )


def create_training_loader(dataset):
    """
    Standard training loader, included so that the training-time
    experiment can be reproduced if training histories/checkpoints
    are available.
    """

    return create_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=WORKERS,
    )


# =====================================================================
# Model construction
# =====================================================================

def build_model() -> nn.Module:
    """
    Construct the FULL MAST-Fuse model using the paper configuration.

    The exact constructor may differ slightly between model.py versions.
    This function first attempts the expected explicit configuration and
    then falls back to the default constructor.
    """

    try:
        model = MASTFuse(
            weather_dim=17,
            spectral_dim=3,
            dynamic_soil_dim=4,
            static_soil_dim=28,
            fusion_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
            sequence_length=SEQ_LEN,
            temporal_encoder="tcn",
            fusion_type="cross_attention",
            use_weather=True,
            use_spectral=True,
            use_dynamic_soil=True,
            use_static_soil=True,
        )

    except TypeError:
        try:
            model = MASTFuse(
                weather_dim=17,
                spectral_dim=3,
                dynamic_soil_dim=4,
                static_soil_dim=28,
                fusion_dim=FUSION_DIM,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                dropout=DROPOUT,
                max_len=SEQ_LEN,
                use_weather=True,
                use_spectral=True,
                use_dynamic_soil=True,
                use_static_soil=True,
            )

        except TypeError:
            # Final fallback: use model defaults.
            model = MASTFuse()

    return model


# =====================================================================
# Checkpoint loading
# =====================================================================

def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """
    Load a trained MAST-Fuse checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n\n"
            "Please provide the path to the trained full-model "
            "best_model.pt checkpoint."
        )

    model = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # -------------------------------------------------------------
    # Support several common checkpoint formats.
    # -------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Check whether this dictionary itself looks like a
            # state dictionary.
            if all(
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            ):
                state_dict = checkpoint
            else:
                raise KeyError(
                    "Could not find model_state_dict/state_dict "
                    "inside checkpoint."
                )

    else:
        state_dict = checkpoint

    # -------------------------------------------------------------
    # Remove DataParallel prefix if present.
    # -------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        print("\nWARNING: Missing checkpoint parameters:")
        for x in missing:
            print("   ", x)

    if unexpected:
        print("\nWARNING: Unexpected checkpoint parameters:")
        for x in unexpected:
            print("   ", x)

    model.to(device)
    model.eval()

    return model


# =====================================================================
# Parameter count
# =====================================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


# =====================================================================
# Model size
# =====================================================================

def get_model_size(checkpoint_path: str | Path) -> dict:
    """
    Model/checkpoint size on disk.
    """

    checkpoint_path = Path(checkpoint_path)

    size_bytes = checkpoint_path.stat().st_size

    return {
        "checkpoint_size_bytes": int(size_bytes),
        "checkpoint_size_mb_decimal":
            float(size_bytes / (1024 ** 2)),
        "checkpoint_size_mb_si":
            float(size_bytes / 1_000_000),
    }


# =====================================================================
# Batch movement
# =====================================================================

def move_batch(
    batch: dict,
    device: torch.device,
) -> dict:

    result = {}

    for key, value in batch.items():

        if torch.is_tensor(value):
            result[key] = value.to(
                device,
                non_blocking=False,
            )

        else:
            result[key] = value

    return result


# =====================================================================
# Model forward
# =====================================================================

def model_forward(
    model: nn.Module,
    batch: dict,
    return_aux: bool = False,
):
    """
    Forward pass compatible with the existing dataset format.
    """

    x = {}

    for name in [
        "weather",
        "spectral",
        "dynamic_soil",
        "static_soil",
    ]:
        if name in batch:
            x[name] = batch[name]

    # -------------------------------------------------------------
    # Try auxiliary-output mode first when requested.
    # -------------------------------------------------------------

    if return_aux:

        try:
            output = model(
                **x,
                return_aux=True,
            )
            return output

        except TypeError:
            pass

        try:
            output = model(
                **x,
                return_attention=True,
            )
            return output

        except TypeError:
            pass

    # -------------------------------------------------------------
    # Standard forward.
    # -------------------------------------------------------------

    return model(**x)


# =====================================================================
# Extract prediction and interpretability information
# =====================================================================

def extract_prediction(output):
    """
    Extract prediction tensor from different possible MAST-Fuse
    output formats.
    """

    # Tensor
    if torch.is_tensor(output):
        return output

    # Dictionary
    if isinstance(output, dict):

        for key in [
            "prediction",
            "pred",
            "output",
            "y_hat",
        ]:
            if key in output:
                return output[key]

    # Tuple/list
    if isinstance(output, (tuple, list)):

        # In the current MAST-Fuse implementation the first element
        # is the fused vector in some fusion-only configurations, while
        # the complete model returns prediction directly.
        #
        # Find a scalar-like tensor first.
        for item in output:
            if torch.is_tensor(item):
                if item.ndim <= 2:
                    if item.shape[-1] == 1 or item.ndim == 1:
                        return item

        # Fall back to first tensor.
        for item in output:
            if torch.is_tensor(item):
                return item

    raise RuntimeError(
        "Could not extract prediction from model output."
    )


def find_named_tensor(
    obj,
    candidate_names,
):
    """
    Recursively search dictionaries/tuples for attention tensors.
    """

    if isinstance(obj, dict):

        # Exact candidate names first.
        for name in candidate_names:
            if name in obj:
                value = obj[name]
                if torch.is_tensor(value):
                    return value

        # Recursive search.
        for value in obj.values():
            result = find_named_tensor(
                value,
                candidate_names,
            )
            if result is not None:
                return result

    elif isinstance(obj, (tuple, list)):

        for value in obj:
            result = find_named_tensor(
                value,
                candidate_names,
            )

            if result is not None:
                return result

    return None


# =====================================================================
# Attention extraction
# =====================================================================

def extract_attention_weights(output):
    """
    Extract modality and temporal weights.

    Expected conceptual outputs:

        modality_weights:  (B, 4)
        temporal_weights:  (B, 21)

    The function also supports common alternative key names.
    """

    modality_weights = find_named_tensor(
        output,
        [
            "modality_weights",
            "modality_weight",
            "modality_attention",
            "gate_weights",
            "gating_weights",
            "alpha",
            "alphas",
        ],
    )

    temporal_weights = find_named_tensor(
        output,
        [
            "temporal_weights",
            "temporal_weight",
            "temporal_attention",
            "beta",
            "betas",
        ],
    )

    return modality_weights, temporal_weights


# =====================================================================
# Latency measurement
# =====================================================================

@torch.no_grad()
def measure_gpu_latency(
    model: nn.Module,
    loader,
    num_samples: int,
):
    """
    GPU batch-size-1 latency.

    Uses CUDA synchronization and warm-up.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    model.eval()
    device = torch.device("cuda")

    # -------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------

    warmup_count = 0

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        _ = model_forward(
            model,
            batch,
            return_aux=False,
        )

        warmup_count += 1

        if warmup_count >= GPU_WARMUP:
            break

    torch.cuda.synchronize()

    # -------------------------------------------------------------
    # Timed inference
    # -------------------------------------------------------------

    latencies_ms = []

    sample_count = 0

    for batch in loader:

        if sample_count >= num_samples:
            break

        batch = move_batch(
            batch,
            device,
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model_forward(
            model,
            batch,
            return_aux=False,
        )

        torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        sample_count += 1

    latencies_ms = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    return {
        "device": "GPU",
        "batch_size": 1,
        "n_samples": int(len(latencies_ms)),
        "mean_ms_per_sample":
            float(np.mean(latencies_ms)),
        "std_ms_per_sample":
            float(np.std(latencies_ms, ddof=1)),
        "median_ms_per_sample":
            float(np.median(latencies_ms)),
        "min_ms_per_sample":
            float(np.min(latencies_ms)),
        "max_ms_per_sample":
            float(np.max(latencies_ms)),
        "p95_ms_per_sample":
            float(np.percentile(latencies_ms, 95)),
        "throughput_samples_per_second":
            float(
                1000.0 /
                np.mean(latencies_ms)
            ),
    }, latencies_ms


@torch.no_grad()
def measure_cpu_latency(
    model: nn.Module,
    loader,
    num_samples: int,
):
    """
    CPU batch-size-1 latency.
    """

    model.eval()

    device = torch.device("cpu")

    model = model.to(device)

    # -------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------

    warmup_count = 0

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        _ = model_forward(
            model,
            batch,
            return_aux=False,
        )

        warmup_count += 1

        if warmup_count >= CPU_WARMUP:
            break

    # -------------------------------------------------------------
    # Timed inference
    # -------------------------------------------------------------

    latencies_ms = []

    sample_count = 0

    for batch in loader:

        if sample_count >= num_samples:
            break

        batch = move_batch(
            batch,
            device,
        )

        start = time.perf_counter()

        _ = model_forward(
            model,
            batch,
            return_aux=False,
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        sample_count += 1

    latencies_ms = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    return {
        "device": "CPU",
        "batch_size": 1,
        "n_samples": int(len(latencies_ms)),
        "mean_ms_per_sample":
            float(np.mean(latencies_ms)),
        "std_ms_per_sample":
            float(np.std(latencies_ms, ddof=1)),
        "median_ms_per_sample":
            float(np.median(latencies_ms)),
        "min_ms_per_sample":
            float(np.min(latencies_ms)),
        "max_ms_per_sample":
            float(np.max(latencies_ms)),
        "p95_ms_per_sample":
            float(np.percentile(latencies_ms, 95)),
        "throughput_samples_per_second":
            float(
                1000.0 /
                np.mean(latencies_ms)
            ),
    }, latencies_ms


# =====================================================================
# GPU memory measurement
# =====================================================================

@torch.no_grad()
def measure_gpu_memory(
    model: nn.Module,
    loader,
):
    """
    Measure peak GPU memory during batch-size-1 inference.
    """

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # Clean CUDA state
    # -------------------------------------------------------------

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    first_batch = next(iter(loader))

    first_batch = move_batch(
        first_batch,
        device,
    )

    torch.cuda.synchronize()

    _ = model_forward(
        model,
        first_batch,
        return_aux=False,
    )

    torch.cuda.synchronize()

    peak_allocated = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(
            device
        )
    )

    return {
        "peak_gpu_memory_allocated_mb":
            float(
                peak_allocated /
                (1024 ** 2)
            ),
        "peak_gpu_memory_reserved_mb":
            float(
                peak_reserved /
                (1024 ** 2)
            ),
    }


# =====================================================================
# CPU memory measurement
# =====================================================================

def measure_cpu_memory(
    model: nn.Module,
    loader,
):
    """
    Measure process RSS before and after one CPU inference.

    psutil is optional.
    """

    try:
        import psutil
    except ImportError:
        return {
            "cpu_memory_available": False,
            "peak_cpu_memory_mb": None,
        }

    process = psutil.Process(
        os.getpid()
    )

    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    gc.collect()

    before = process.memory_info().rss

    batch = next(iter(loader))

    batch = move_batch(
        batch,
        device,
    )

    with torch.no_grad():
        _ = model_forward(
            model,
            batch,
            return_aux=False,
        )

    after = process.memory_info().rss

    return {
        "cpu_memory_available": True,
        "cpu_rss_before_mb":
            float(before / (1024 ** 2)),
        "cpu_rss_after_mb":
            float(after / (1024 ** 2)),
        "cpu_rss_increment_mb":
            float(
                max(0, after - before)
                / (1024 ** 2)
            ),
    }


# =====================================================================
# Interpretability
# =====================================================================

@torch.no_grad()
def collect_attention_weights(
    model: nn.Module,
    loader,
    device: torch.device,
    max_samples: int | None = None,
):
    """
    Collect modality-gating and temporal-attention weights.

    Returns:
        modality_weights: [N, 4]
        temporal_weights: [N, 21]
    """

    model.eval()
    model.to(device)

    modality_list = []
    temporal_list = []

    n = 0

    for batch in loader:

        if (
            max_samples is not None
            and n >= max_samples
        ):
            break

        batch = move_batch(
            batch,
            device,
        )

        output = model_forward(
            model,
            batch,
            return_aux=True,
        )

        modality_weights, temporal_weights = (
            extract_attention_weights(output)
        )

        if modality_weights is None:
            raise RuntimeError(
                "Could not find modality weights "
                "in the model output. Please inspect "
                "the exact return_aux dictionary of "
                "model.py."
            )

        if temporal_weights is None:
            raise RuntimeError(
                "Could not find temporal weights "
                "in the model output. Please inspect "
                "the exact return_aux dictionary of "
                "model.py."
            )

        modality_weights = (
            modality_weights.detach()
            .float()
            .cpu()
            .numpy()
        )

        temporal_weights = (
            temporal_weights.detach()
            .float()
            .cpu()
            .numpy()
        )

        # ---------------------------------------------------------
        # Expected shape:
        #
        # modality = [B, 4]
        # temporal  = [B, 21]
        # ---------------------------------------------------------

        if modality_weights.ndim == 1:
            modality_weights = (
                modality_weights[None, :]
            )

        if temporal_weights.ndim == 1:
            temporal_weights = (
                temporal_weights[None, :]
            )

        modality_list.append(
            modality_weights
        )

        temporal_list.append(
            temporal_weights
        )

        n += modality_weights.shape[0]

    if not modality_list:
        raise RuntimeError(
            "No attention weights were collected."
        )

    modality_weights = np.concatenate(
        modality_list,
        axis=0,
    )

    temporal_weights = np.concatenate(
        temporal_list,
        axis=0,
    )

    if max_samples is not None:
        modality_weights = (
            modality_weights[:max_samples]
        )

        temporal_weights = (
            temporal_weights[:max_samples]
        )

    return (
        modality_weights,
        temporal_weights,
    )


def summarize_attention(
    modality_weights,
    temporal_weights,
):
    """
    Calculate mean/std attention weights.
    """

    modality_mean = np.mean(
        modality_weights,
        axis=0,
    )

    modality_std = np.std(
        modality_weights,
        axis=0,
        ddof=1,
    )

    temporal_mean = np.mean(
        temporal_weights,
        axis=0,
    )

    temporal_std = np.std(
        temporal_weights,
        axis=0,
        ddof=1,
    )

    return {
        "modality": {
            "mean": modality_mean.tolist(),
            "std": modality_std.tolist(),
            "sum_of_means":
                float(np.sum(modality_mean)),
        },

        "temporal": {
            "mean": temporal_mean.tolist(),
            "std": temporal_std.tolist(),
            "sum_of_means":
                float(np.sum(temporal_mean)),
        },
    }


# =====================================================================
# Save attention CSV files
# =====================================================================

def save_attention_csv(
    modality_weights,
    temporal_weights,
    output_dir,
):
    """
    Save per-sample attention weights.
    """

    output_dir = Path(output_dir)

    # -------------------------------------------------------------
    # Modality
    # -------------------------------------------------------------

    modality_path = (
        output_dir /
        "modality_gate_weights.csv"
    )

    modality_names = [
        "Weather",
        "Spectral",
        "Dynamic Soil",
        "Static Soil",
    ]

    with open(
        modality_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sample",
                *modality_names,
            ]
        )

        for i, row in enumerate(
            modality_weights
        ):

            writer.writerow(
                [
                    i,
                    *[
                        float(x)
                        for x in row
                    ],
                ]
            )

    # -------------------------------------------------------------
    # Temporal
    # -------------------------------------------------------------

    temporal_path = (
        output_dir /
        "temporal_attention_weights.csv"
    )

    temporal_names = [
        f"t_{i + 1}"
        for i in range(
            temporal_weights.shape[1]
        )
    ]

    with open(
        temporal_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "sample",
                *temporal_names,
            ]
        )

        for i, row in enumerate(
            temporal_weights
        ):

            writer.writerow(
                [
                    i,
                    *[
                        float(x)
                        for x in row
                    ],
                ]
            )

    return (
        modality_path,
        temporal_path,
    )


# =====================================================================
# Save summary
# =====================================================================

def save_json(obj, path):
    path = Path(path)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
        )


# =====================================================================
# Training-time analysis
# =====================================================================

def analyse_training_histories(
    result_root,
    seeds=SEEDS,
):
    """
    Read the existing training histories if available.

    Supports:
        history.json

    Expected history formats:

    1.
        [
            [epoch, train_loss, val_loss],
            ...
        ]

    2.
        [
            {
                "epoch": ...,
                "train_loss": ...,
                "val_loss": ...,
                "epoch_time_seconds": ...
            },
            ...
        ]

    If epoch times are absent, the function reports that they cannot
    be reconstructed from the stored history alone.
    """

    result_root = Path(result_root)

    seed_total_times = []
    seed_epoch_times = []

    details = {}

    for seed in seeds:

        # Search recursively for the seed directory.
        candidates = list(
            result_root.rglob(
                f"seed_{seed}/history.json"
            )
        )

        if not candidates:
            details[str(seed)] = {
                "status": "history_not_found"
            }
            continue

        history_path = candidates[0]

        with open(
            history_path,
            "r",
            encoding="utf-8",
        ) as f:
            history = json.load(f)

        epoch_times = []

        for item in history:

            if isinstance(item, dict):

                for key in [
                    "epoch_time_seconds",
                    "epoch_time",
                    "time_seconds",
                ]:

                    if key in item:
                        epoch_times.append(
                            float(item[key])
                        )
                        break

            elif isinstance(item, list):

                # Expected modified format:
                # [epoch, train_loss, val_loss,
                #  epoch_time_seconds]
                if len(item) >= 4:
                    epoch_times.append(
                        float(item[3])
                    )

        # ---------------------------------------------------------
        # Locate metrics.json.
        # ---------------------------------------------------------

        metrics_path = (
            history_path.parent /
            "metrics.json"
        )

        total_time = None

        if metrics_path.exists():

            with open(
                metrics_path,
                "r",
                encoding="utf-8",
            ) as f:
                metrics = json.load(f)

            if (
                "training_time_seconds"
                in metrics
            ):
                total_time = float(
                    metrics[
                        "training_time_seconds"
                    ]
                )

        if total_time is not None:
            seed_total_times.append(
                total_time
            )

        if epoch_times:
            seed_epoch_times.extend(
                epoch_times
            )

        details[str(seed)] = {
            "history_file":
                str(history_path),
            "total_training_time_seconds":
                total_time,
            "number_of_recorded_epochs":
                len(history),
            "number_of_epoch_times":
                len(epoch_times),
        }

    result = {
        "per_seed": details
    }

    if seed_total_times:

        result["total_training_time_seconds"] = {
            "mean":
                float(
                    np.mean(
                        seed_total_times
                    )
                ),
            "std":
                float(
                    np.std(
                        seed_total_times,
                        ddof=1,
                    )
                ),
            "values":
                seed_total_times,
            "mean_minutes":
                float(
                    np.mean(
                        seed_total_times
                    ) / 60.0
                ),
            "std_minutes":
                float(
                    np.std(
                        seed_total_times,
                        ddof=1,
                    ) / 60.0
                ),
        }

    if seed_epoch_times:

        result["epoch_time_seconds"] = {
            "mean":
                float(
                    np.mean(
                        seed_epoch_times
                    )
                ),
            "std":
                float(
                    np.std(
                        seed_epoch_times,
                        ddof=1,
                    )
                ),
            "n_recorded_epochs":
                len(seed_epoch_times),
        }

    else:

        result["epoch_time_seconds"] = {
            "status":
                "Epoch-level timing is not stored "
                "in the current history files."
        }

    return result


# =====================================================================
# Main benchmark
# =====================================================================

def run_benchmark(
    checkpoint_path,
    num_samples=NUM_LATENCY_SAMPLES,
):
    """
    Run all inference, memory and interpretability experiments.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    print("=" * 80)
    print("MAST-Fuse Computational Efficiency Analysis")
    print("=" * 80)

    print("\nCheckpoint:")
    print(checkpoint_path)

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    print("\nLoading unseen test set...")

    test_dataset = load_test_dataset()

    print(
        f"Test samples available: "
        f"{len(test_dataset)}"
    )

    n_samples = min(
        num_samples,
        len(test_dataset),
    )

    # -------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------

    latency_loader = (
        create_single_sample_loader(
            test_dataset
        )
    )

    # -------------------------------------------------------------
    # Model parameters
    # -------------------------------------------------------------

    print("\nLoading model on CPU...")

    cpu_model = load_checkpoint(
        checkpoint_path,
        CPU_DEVICE,
    )

    parameter_info = count_parameters(
        cpu_model
    )

    model_size = get_model_size(
        checkpoint_path
    )

    print(
        "\nParameter count:"
    )

    print(
        f"  Total:     "
        f"{parameter_info['total_parameters']:,}"
    )

    print(
        f"  Trainable: "
        f"{parameter_info['trainable_parameters']:,}"
    )

    print(
        "\nCheckpoint size:"
    )

    print(
        f"  MB: "
        f"{model_size['checkpoint_size_mb_decimal']:.3f}"
    )

    # -------------------------------------------------------------
    # CPU latency
    # -------------------------------------------------------------

    print("\n" + "-" * 80)
    print(
        f"CPU latency: {n_samples} samples, "
        f"batch size 1"
    )
    print("-" * 80)

    cpu_latency, cpu_raw = (
        measure_cpu_latency(
            cpu_model,
            latency_loader,
            n_samples,
        )
    )

    print(
        f"Mean:   "
        f"{cpu_latency['mean_ms_per_sample']:.4f} ms"
    )

    print(
        f"Std:    "
        f"{cpu_latency['std_ms_per_sample']:.4f} ms"
    )

    print(
        f"Median: "
        f"{cpu_latency['median_ms_per_sample']:.4f} ms"
    )

    print(
        f"P95:    "
        f"{cpu_latency['p95_ms_per_sample']:.4f} ms"
    )

    print(
        f"Throughput: "
        f"{cpu_latency['throughput_samples_per_second']:.2f} "
        f"samples/s"
    )

    # -------------------------------------------------------------
    # CPU memory
    # -------------------------------------------------------------

    print("\nCPU memory:")

    cpu_memory = measure_cpu_memory(
        cpu_model,
        latency_loader,
    )

    print(cpu_memory)

    # -------------------------------------------------------------
    # GPU
    # -------------------------------------------------------------

    gpu_latency = None
    gpu_memory = None

    if CUDA_AVAILABLE:

        print("\n" + "-" * 80)
        print(
            f"GPU latency: {n_samples} samples, "
            f"batch size 1"
        )
        print("-" * 80)

        # Reload cleanly on GPU.
        del cpu_model

        gc.collect()

        gpu_model = load_checkpoint(
            checkpoint_path,
            GPU_DEVICE,
        )

        gpu_latency, gpu_raw = (
            measure_gpu_latency(
                gpu_model,
                latency_loader,
                n_samples,
            )
        )

        print(
            f"Mean:   "
            f"{gpu_latency['mean_ms_per_sample']:.4f} ms"
        )

        print(
            f"Std:    "
            f"{gpu_latency['std_ms_per_sample']:.4f} ms"
        )

        print(
            f"Median: "
            f"{gpu_latency['median_ms_per_sample']:.4f} ms"
        )

        print(
            f"P95:    "
            f"{gpu_latency['p95_ms_per_sample']:.4f} ms"
        )

        print(
            f"Throughput: "
            f"{gpu_latency['throughput_samples_per_second']:.2f} "
            f"samples/s"
        )

        # ---------------------------------------------------------
        # GPU memory
        # ---------------------------------------------------------

        print("\nGPU memory:")

        gpu_memory = measure_gpu_memory(
            gpu_model,
            latency_loader,
        )

        print(
            f"Peak allocated: "
            f"{gpu_memory['peak_gpu_memory_allocated_mb']:.3f} MB"
        )

        print(
            f"Peak reserved:  "
            f"{gpu_memory['peak_gpu_memory_reserved_mb']:.3f} MB"
        )

        # ---------------------------------------------------------
        # Attention
        # ---------------------------------------------------------

        print("\n" + "-" * 80)
        print(
            "Collecting modality and temporal attention..."
        )
        print("-" * 80)

        # Use the entire test set for interpretability.
        modality_weights, temporal_weights = (
            collect_attention_weights(
                gpu_model,
                latency_loader,
                GPU_DEVICE,
                max_samples=None,
            )
        )

        attention_summary = (
            summarize_attention(
                modality_weights,
                temporal_weights,
            )
        )

        modality_csv, temporal_csv = (
            save_attention_csv(
                modality_weights,
                temporal_weights,
                OUTPUT_DIR,
            )
        )

        print("\nMean modality weights:")

        modality_names = [
            "Weather",
            "Spectral",
            "Dynamic Soil",
            "Static Soil",
        ]

        for name, mean, std in zip(
            modality_names,
            attention_summary[
                "modality"
            ]["mean"],
            attention_summary[
                "modality"
            ]["std"],
        ):

            print(
                f"  {name:15s}: "
                f"{mean:.6f} ± {std:.6f}"
            )

        print(
            "\nSum of mean modality weights: "
            f"{attention_summary['modality']['sum_of_means']:.6f}"
        )

        print("\nMean temporal weights:")

        for t, (mean, std) in enumerate(
            zip(
                attention_summary[
                    "temporal"
                ]["mean"],
                attention_summary[
                    "temporal"
                ]["std"],
            ),
            start=1,
        ):

            print(
                f"  t={t:02d}: "
                f"{mean:.6f} ± {std:.6f}"
            )

        print(
            "\nSum of mean temporal weights: "
            f"{attention_summary['temporal']['sum_of_means']:.6f}"
        )

        # ---------------------------------------------------------
        # Raw latency files
        # ---------------------------------------------------------

        np.save(
            OUTPUT_DIR /
            "gpu_latency_ms.npy",
            gpu_raw,
        )

        np.save(
            OUTPUT_DIR /
            "cpu_latency_ms.npy",
            cpu_raw,
        )

        del gpu_model

        torch.cuda.empty_cache()

    else:

        print(
            "\nCUDA is not available; "
            "GPU latency/memory measurements skipped."
        )

        attention_summary = {
            "status":
                "Attention extraction skipped because "
                "CUDA is unavailable."
        }

    # -------------------------------------------------------------
    # Save inference summary
    # -------------------------------------------------------------

    inference_summary = {
        "configuration": {
            "sequence_length": SEQ_LEN,
            "fusion_dimension": FUSION_DIM,
            "attention_heads": NUM_HEADS,
            "feed_forward_dimension": FF_DIM,
            "dropout": DROPOUT,
            "latency_batch_size": 1,
            "latency_samples": n_samples,
        },

        "checkpoint": str(
            checkpoint_path
        ),

        "parameters": parameter_info,

        "model_size": model_size,

        "cpu_latency": cpu_latency,

        "cpu_memory": cpu_memory,

        "gpu_latency": gpu_latency,

        "gpu_memory": gpu_memory,

        "attention": attention_summary,
    }

    save_json(
        inference_summary,
        OUTPUT_DIR /
        "computational_efficiency_summary.json",
    )

    print("\n" + "=" * 80)
    print(
        "Inference/efficiency analysis completed."
    )
    print("=" * 80)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    return inference_summary


# =====================================================================
# Command-line interface
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MAST-Fuse computational efficiency "
            "and interpretability analysis."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to the trained full-model "
            "best_model.pt checkpoint."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help=(
            "Number of test samples for latency "
            "measurement. Default: 1000."
        ),
    )

    parser.add_argument(
        "--training-results",
        type=str,
        default=None,
        help=(
            "Root directory containing the existing "
            "five-seed training results."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Run inference/efficiency analysis.
    # -------------------------------------------------------------

    run_benchmark(
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
    )

    # -------------------------------------------------------------
    # Training-time analysis.
    # -------------------------------------------------------------

    if args.training_results is not None:

        print("\n" + "=" * 80)
        print(
            "Five-seed training-time analysis"
        )
        print("=" * 80)

        training_summary = (
            analyse_training_histories(
                args.training_results,
                SEEDS,
            )
        )

        save_json(
            training_summary,
            OUTPUT_DIR /
            "training_time_summary.json",
        )

        print(
            json.dumps(
                training_summary,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()