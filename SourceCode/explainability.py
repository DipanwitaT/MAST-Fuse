#Author - Dipanwita Thakur

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import os
import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import shap


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from dataset import load_all_data
from model import MASTFuse


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "explainability"

CHECKPOINT = (
    RESULTS_DIR
    / "ablations"
    / "full_model_best.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------------------------------------------------------
# SHAP configuration
# -----------------------------------------------------------------------------

BACKGROUND_SAMPLES = 32

EXPLANATION_SAMPLES = 64

RANDOM_SEED = 42

# Maximum number of SHAP evaluations.
#
# The default is deliberately moderate because the data are temporal.
#
# For permutation SHAP:
#
#     2 * n_features + 1
#
# is a reasonable minimum.
#
# The script automatically uses a safe value if this is None.
MAX_EVALS = None

# Number of features shown in plots.
TOP_K_FEATURES = 20

# Number of features used for interaction analysis.
INTERACTION_TOP_K = 12

# Interaction calculations are expensive.
# Number of samples used for interactions.
INTERACTION_SAMPLES = 32


# =============================================================================
# DATASET DIMENSIONS
# =============================================================================

SEQUENCE_LENGTH = 21

MODALITY_DIMS = {
    "weather": 17,
    "spectral": 3,
    "dynamic_soil": 4,
    "static_soil": 28,
}


# =============================================================================
# FEATURE NAMES
# =============================================================================

# -----------------------------------------------------------------------------
# Weather
# -----------------------------------------------------------------------------

WEATHER_FEATURE_NAMES = [
    "temperature",
    "relative_humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "pressure",
    "solar_radiation",
    "dew_point",
    "vapor_pressure",
    "soil_temperature",
    "day_of_year",
    "month",
    "precipitation_probability",
    "cloud_cover",
    "max_temperature",
    "min_temperature",
    "evapotranspiration",
]


# -----------------------------------------------------------------------------
# Spectral
# -----------------------------------------------------------------------------

SPECTRAL_FEATURE_NAMES = [
    "NDVI",
    "NDWI",
    "EVI",
]


# -----------------------------------------------------------------------------
# Dynamic soil
# -----------------------------------------------------------------------------

DYNAMIC_SOIL_FEATURE_NAMES = [
    "BSI",
    "SAVI",
    "NDTI",
    "RI",
]


# -----------------------------------------------------------------------------
# Static soil
# -----------------------------------------------------------------------------

STATIC_SOIL_FEATURE_NAMES = [
    "ph_0-5cm",
    "ph_5-15cm",
    "ph_15-30cm",
    "ph_30-60cm",

    "organic_carbon_0-5cm",
    "organic_carbon_5-15cm",
    "organic_carbon_15-30cm",
    "organic_carbon_30-60cm",

    "sand_0-5cm",
    "sand_5-15cm",
    "sand_15-30cm",
    "sand_30-60cm",

    "silt_0-5cm",
    "silt_5-15cm",
    "silt_15-30cm",
    "silt_30-60cm",

    "clay_0-5cm",
    "clay_5-15cm",
    "clay_15-30cm",
    "clay_30-60cm",

    "cec_0-5cm",
    "cec_5-15cm",
    "cec_15-30cm",
    "cec_30-60cm",

    "bulk_density_0-5cm",
    "bulk_density_5-15cm",
    "bulk_density_15-30cm",
    "bulk_density_30-60cm",
]


FEATURE_NAMES = {
    "weather": WEATHER_FEATURE_NAMES,
    "spectral": SPECTRAL_FEATURE_NAMES,
    "dynamic_soil": DYNAMIC_SOIL_FEATURE_NAMES,
    "static_soil": STATIC_SOIL_FEATURE_NAMES,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DIRECTORY
# =============================================================================

def prepare_output_directory() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# PRINT HEADER
# =============================================================================

def print_header() -> None:

    print()
    print("=" * 80)
    print("MAST-Fuse / MuSTIPest-V3")
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 80)

    print(f"Device     : {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Output     : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Checkpoint : "
        f"{CHECKPOINT}"
    )

    print(
        f"SHAP       : "
        f"{shap.__version__}"
    )


# =============================================================================
# DATA OBJECT CONVERSION
# =============================================================================

def get_field(
    obj: Any,
    name: str,
):
    """
    Retrieve a field from:

    - dict
    - dataclass/object
    - tuple/list where necessary
    """

    if isinstance(obj, dict):

        if name in obj:

            return obj[name]

        return None

    if hasattr(obj, name):

        return getattr(
            obj,
            name,
        )

    return None


# =============================================================================
# CONVERT DATASET SPLIT TO NUMPY DICTIONARY
# =============================================================================

def convert_split(
    split: Any,
    split_name: str,
) -> Dict[str, np.ndarray]:

    print()
    print("-" * 80)
    print(
        f"DATA VALIDATION — {split_name.upper()}"
    )
    print("-" * 80)

    names = [
        "weather",
        "spectral",
        "dynamic_soil",
        "static_soil",
        "target",
    ]

    data = {}

    for name in names:

        value = get_field(
            split,
            name,
        )

        if value is None:

            raise ValueError(
                f"Cannot find '{name}' "
                f"in {split_name}."
            )

        if torch.is_tensor(value):

            value = value.detach().cpu().numpy()

        else:

            value = np.asarray(value)

        value = value.astype(
            np.float32,
            copy=False,
        )

        data[name] = value

        print(
            f"{name:<15}: "
            f"{tuple(value.shape)}"
        )

    # -------------------------------------------------------------------------
    # Shape checks
    # -------------------------------------------------------------------------

    expected = {
        "weather": (
            None,
            21,
            17,
        ),

        "spectral": (
            None,
            21,
            3,
        ),

        "dynamic_soil": (
            None,
            21,
            4,
        ),

        "static_soil": (
            None,
            21,
            28,
        ),

        "target": (
            None,
        ),
    }

    for name, shape in expected.items():

        actual = data[name].shape

        if len(actual) != len(shape):

            raise ValueError(
                f"{split_name}/{name}: "
                f"expected {shape}, "
                f"got {actual}"
            )

        for i, expected_value in enumerate(shape):

            if expected_value is not None:

                if actual[i] != expected_value:

                    raise ValueError(
                        f"{split_name}/{name}: "
                        f"expected dimension "
                        f"{shape}, "
                        f"got {actual}"
                    )

    # -------------------------------------------------------------------------
    # finite check
    # -------------------------------------------------------------------------

    for name, value in data.items():

        if not np.isfinite(value).all():

            raise ValueError(
                f"{split_name}/{name} "
                f"contains NaN or Inf."
            )

    return data


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset_for_explainability():

    print()
    print("=" * 80)
    print("LOADING DATASET FOR EXPLAINABILITY")
    print("=" * 80)

    raw = load_all_data()

    print()
    print(
        f"Dataset object type:"
        f" {type(raw)}"
    )

    # -------------------------------------------------------------------------
    # Current dataset.py normally returns a dictionary.
    # -------------------------------------------------------------------------

    if isinstance(raw, dict):

        train = raw.get("train")

        validation = raw.get(
            "val",
            raw.get("validation"),
        )

        test = raw.get("test")

        if train is None:
            raise KeyError(
                "load_all_data() does not "
                "contain 'train'."
            )

        if validation is None:
            raise KeyError(
                "load_all_data() does not "
                "contain 'val' or 'validation'."
            )

        if test is None:
            raise KeyError(
                "load_all_data() does not "
                "contain 'test'."
            )

    # -------------------------------------------------------------------------
    # Also support tuple/list:
    #
    # (train, validation, test)
    # -------------------------------------------------------------------------

    elif isinstance(
        raw,
        (tuple, list),
    ):

        if len(raw) != 3:

            raise ValueError(
                "Expected "
                "(train, validation, test)."
            )

        train, validation, test = raw

    else:

        raise TypeError(
            "Unsupported return type from "
            f"load_all_data(): {type(raw)}"
        )

    data = {

        "train": convert_split(
            train,
            "train",
        ),

        "validation": convert_split(
            validation,
            "validation",
        ),

        "test": convert_split(
            test,
            "test",
        ),
    }

    return data


# =============================================================================
# MODEL CHECKPOINT UTILITIES
# =============================================================================

def extract_state_dict(
    checkpoint: Any,
):
    """
    Supports common checkpoint formats.
    """

    if isinstance(
        checkpoint,
        dict,
    ):

        possible_keys = [
            "model_state_dict",
            "state_dict",
            "model",
        ]

        for key in possible_keys:

            if key in checkpoint:

                value = checkpoint[key]

                if isinstance(
                    value,
                    dict,
                ):

                    return value

    if isinstance(
        checkpoint,
        dict,
    ):

        # A raw state_dict
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):

            return checkpoint

    raise TypeError(
        "Unable to locate model "
        "state_dict in checkpoint."
    )


# =============================================================================
# BUILD MODEL
# =============================================================================

def build_model() -> nn.Module:

    print()
    print("=" * 80)
    print("LOADING MAST-FUSE MODEL")
    print("=" * 80)

    print(
        f"Checkpoint:\n"
        f"  {CHECKPOINT}"
    )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    # -------------------------------------------------------------------------
    # Build the same full model used during training.
    # -------------------------------------------------------------------------

    model = MASTFuse(

        use_weather=True,

        use_spectral=True,

        use_dynamic_soil=True,

        use_static_soil=True,

        fusion_dim=128,

        num_heads=4,

        dropout=0.1,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    # -------------------------------------------------------------------------
    # Handle DataParallel prefixes.
    # -------------------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )
    )

    if missing:

        print()
        print(
            "WARNING: Missing checkpoint keys:"
        )

        for key in missing[:20]:

            print(
                f"  {key}"
            )

        if len(missing) > 20:

            print(
                f"  ... "
                f"{len(missing) - 20} more"
            )

    if unexpected:

        print()
        print(
            "WARNING: Unexpected checkpoint keys:"
        )

        for key in unexpected[:20]:

            print(
                f"  {key}"
            )

    model = model.to(
        DEVICE
    )

    model.eval()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model loaded successfully."
    )

    print(
        f"Parameters : "
        f"{total_params:,}"
    )

    print(
        f"Device     : "
        f"{DEVICE}"
    )

    return model


# =============================================================================
# SAFE MODEL OUTPUT
# =============================================================================

def extract_prediction(
    output: Any,
) -> torch.Tensor:

    if isinstance(
        output,
        dict,
    ):

        if "prediction" in output:

            output = output[
                "prediction"
            ]

        elif "predictions" in output:

            output = output[
                "predictions"
            ]

        else:

            raise KeyError(
                "Model dictionary output "
                "does not contain "
                "'prediction'."
            )

    if not torch.is_tensor(output):

        raise TypeError(
            "Model output must be a "
            "torch.Tensor."
        )

    if output.ndim == 0:

        output = output.reshape(1)

    elif output.ndim > 1:

        output = output.reshape(
            output.shape[0],
            -1,
        )

        output = output[:, 0]

    return output


# =============================================================================
# SINGLE-SAMPLE MODEL PREDICTION
# =============================================================================

@torch.no_grad()
def predict_single(
    model: nn.Module,
    weather: np.ndarray,
    spectral: np.ndarray,
    dynamic_soil: np.ndarray,
    static_soil: np.ndarray,
) -> float:
    """
    Predict ONE sample.

    This is intentionally batch size = 1.

    This prevents SHAP from constructing a batch containing
    background and query samples that can trigger the
    cross-modal attention shape mismatch.
    """

    weather_t = torch.as_tensor(
        weather,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    spectral_t = torch.as_tensor(
        spectral,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    dynamic_soil_t = torch.as_tensor(
        dynamic_soil,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    static_soil_t = torch.as_tensor(
        static_soil,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    output = model(
        weather=weather_t,
        spectral=spectral_t,
        dynamic_soil=dynamic_soil_t,
        static_soil=static_soil_t,
    )

    prediction = extract_prediction(
        output
    )

    return float(
        prediction[0]
        .detach()
        .cpu()
        .item()
    )


# =============================================================================
# MODEL SANITY CHECK
# =============================================================================

def run_sanity_check(
    model: nn.Module,
    test_data: Dict[str, np.ndarray],
) -> None:

    print()
    print("=" * 80)
    print("MODEL / DATA SANITY CHECK")
    print("=" * 80)

    index = 0

    weather = test_data[
        "weather"
    ][index]

    spectral = test_data[
        "spectral"
    ][index]

    dynamic_soil = test_data[
        "dynamic_soil"
    ][index]

    static_soil = test_data[
        "static_soil"
    ][index]

    print(
        f"Weather      : "
        f"{weather.shape}"
    )

    print(
        f"Spectral     : "
        f"{spectral.shape}"
    )

    print(
        f"Dynamic soil : "
        f"{dynamic_soil.shape}"
    )

    print(
        f"Static soil  : "
        f"{static_soil.shape}"
    )

    prediction = predict_single(
        model,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
    )

    print()
    print(
        f"Prediction   : "
        f"{prediction:.8f}"
    )

    if not np.isfinite(
        prediction
    ):

        raise ValueError(
            "Model produced NaN/Inf."
        )

    print(
        "SANITY CHECK PASSED"
    )


# =============================================================================
# FEATURE MATRIX
# =============================================================================

def flatten_modality(
    data: np.ndarray,
    modality: str,
) -> np.ndarray:
    """
    Convert:

        (N, 21, F)

    into:

        (N, 21*F)
    """

    if data.ndim != 3:

        raise ValueError(
            f"{modality} must be 3-D."
        )

    return data.reshape(
        data.shape[0],
        -1,
    )


# =============================================================================
# FEATURE LABELS
# =============================================================================

def make_feature_names(
    modality: str,
) -> List[str]:

    base_names = FEATURE_NAMES[
        modality
    ]

    labels = []

    for t in range(
        SEQUENCE_LENGTH
    ):

        for feature in base_names:

            labels.append(
                f"{modality}_t{t+1}_{feature}"
            )

    return labels


# =============================================================================
# SAFE SHAP PREDICTION WRAPPER
# =============================================================================

class ModalityPredictionWrapper:
    """
    SHAP prediction wrapper for one modality.

    SHAP modifies only one modality.

    The other three modalities remain fixed.

    Every SHAP prediction is evaluated sample-by-sample.

    This is the key fix for the cross-attention error.
    """

    def __init__(
        self,
        model: nn.Module,
        modality: str,
        reference_data: Dict[str, np.ndarray],
        reference_index: int = 0,
    ):

        self.model = model

        self.modality = modality

        self.reference_data = (
            reference_data
        )

        self.reference_index = (
            reference_index
        )

    def __call__(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim == 1:

            X = X.reshape(
                1,
                -1,
            )

        results = []

        for row in X:

            sample = {}

            for name in [
                "weather",
                "spectral",
                "dynamic_soil",
                "static_soil",
            ]:

                value = (
                    self.reference_data[
                        name
                    ][
                        self.reference_index
                    ]
                )

                sample[name] = value

            # -------------------------------------------------------------
            # Replace only the modality currently being explained.
            # -------------------------------------------------------------

            sample[
                self.modality
            ] = row.reshape(
                SEQUENCE_LENGTH,
                MODALITY_DIMS[
                    self.modality
                ],
            )

            prediction = predict_single(
                self.model,
                sample["weather"],
                sample["spectral"],
                sample["dynamic_soil"],
                sample["static_soil"],
            )

            results.append(
                prediction
            )

        return np.asarray(
            results,
            dtype=np.float64,
        )


# =============================================================================
# SELECT SHAP BACKGROUND
# =============================================================================

def select_background(
    X: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(
        seed
    )

    n = min(
        n,
        X.shape[0],
    )

    indices = rng.choice(
        X.shape[0],
        size=n,
        replace=False,
    )

    return X[
        indices
    ]


# =============================================================================
# SHAP ANALYSIS FOR ONE MODALITY
# =============================================================================

def analyze_modality_shap(
    model: nn.Module,
    test_data: Dict[str, np.ndarray],
    modality: str,
) -> Dict[str, Any]:

    print()
    print("=" * 80)
    print(
        f"SHAP ANALYSIS — "
        f"{modality.upper()}"
    )
    print("=" * 80)

    X_full = flatten_modality(
        test_data[modality],
        modality,
    )

    feature_names = make_feature_names(
        modality
    )

    # -------------------------------------------------------------------------
    # Background
    # -------------------------------------------------------------------------

    background = select_background(
        X_full,
        BACKGROUND_SAMPLES,
        RANDOM_SEED,
    )

    # -------------------------------------------------------------------------
    # Explanation samples
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED + 100
    )

    n_explain = min(
        EXPLANATION_SAMPLES,
        X_full.shape[0],
    )

    indices = rng.choice(
        X_full.shape[0],
        size=n_explain,
        replace=False,
    )

    X_explain = X_full[
        indices
    ]

    print(
        f"Background samples : "
        f"{background.shape}"
    )

    print(
        f"Explanation samples: "
        f"{X_explain.shape}"
    )

    # -------------------------------------------------------------------------
    # Wrapper
    # -------------------------------------------------------------------------

    wrapper = ModalityPredictionWrapper(
        model=model,
        modality=modality,
        reference_data=test_data,
        reference_index=int(
            indices[0]
        ),
    )

    # -------------------------------------------------------------------------
    # SHAP explainer
    #
    # PermutationExplainer is used instead of GradientExplainer because:
    #
    # 1. the model contains cross-modal attention;
    # 2. inputs are flattened by modality;
    # 3. the wrapper reconstructs the temporal tensor;
    # 4. the wrapper forces batch size = 1.
    # -------------------------------------------------------------------------

    print()
    print(
        "Creating permutation SHAP explainer..."
    )

    explainer = shap.Explainer(
        wrapper,
        background,
        algorithm="permutation",
        feature_names=feature_names,
    )

    # -------------------------------------------------------------------------
    # Evaluation count
    # -------------------------------------------------------------------------

    if MAX_EVALS is None:

        max_evals = (
            2 * X_full.shape[1]
            + 1
        )

    else:

        max_evals = max(
            MAX_EVALS,
            2 * X_full.shape[1]
            + 1,
        )

    print(
        f"Max evaluations : "
        f"{max_evals}"
    )

    print()
    print(
        "Running SHAP..."
    )

    shap_values = explainer(
        X_explain,
        max_evals=max_evals,
        silent=False,
    )

    values = np.asarray(
        shap_values.values
    )

    if values.ndim == 3:

        values = values[..., 0]

    if values.ndim != 2:

        raise RuntimeError(
            "Unexpected SHAP shape: "
            f"{values.shape}"
        )

    # -------------------------------------------------------------------------
    # Mean absolute SHAP
    # -------------------------------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(values),
        axis=0,
    )

    # -------------------------------------------------------------------------
    # SHAP variance
    # -------------------------------------------------------------------------

    shap_variance = np.var(
        values,
        axis=0,
    )

    # -------------------------------------------------------------------------
    # Mean SHAP
    # -------------------------------------------------------------------------

    mean_shap = np.mean(
        values,
        axis=0,
    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    ranking = np.argsort(
        mean_abs_shap
    )[::-1]

    # -------------------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------------------

    result = pd.DataFrame({

        "modality": modality,

        "feature": [
            feature_names[i]
            for i in range(
                len(feature_names)
            )
        ],

        "mean_shap": mean_shap,

        "mean_abs_shap":
            mean_abs_shap,

        "shap_variance":
            shap_variance,

        "rank": np.empty(
            len(feature_names),
            dtype=int,
        ),
    })

    for rank, index in enumerate(
        ranking,
        start=1,
    ):

        result.loc[
            index,
            "rank"
        ] = rank

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / f"shap_{modality}.csv"
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # SHAP object
    # -------------------------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / f"shap_values_{modality}.npy",
        values,
    )

    np.save(
        OUTPUT_DIR
        / f"shap_samples_{modality}.npy",
        X_explain,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print(
        f"Top {TOP_K_FEATURES} "
        f"features:"
    )

    print(
        result.head(
            TOP_K_FEATURES
        ).to_string(
            index=False
        )
    )

    return {

        "modality": modality,

        "values": values,

        "samples": X_explain,

        "feature_names": feature_names,

        "result": result,

        "shap_object": shap_values,
    }


# =============================================================================
# SHAP BAR PLOT
# =============================================================================

def plot_feature_importance(
    result: pd.DataFrame,
    modality: str,
) -> None:

    top = result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).head(
        TOP_K_FEATURES
    )

    top = top.sort_values(
        "mean_abs_shap"
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["feature"],
        top["mean_abs_shap"],
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"SHAP Feature Importance — "
        f"{modality}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"shap_importance_{modality}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# SHAP VARIANCE PLOT
# =============================================================================

def plot_shap_variance(
    result: pd.DataFrame,
    modality: str,
) -> None:

    top = result.sort_values(
        "shap_variance",
        ascending=False,
    ).head(
        TOP_K_FEATURES
    )

    top = top.sort_values(
        "shap_variance"
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top["feature"],
        top["shap_variance"],
    )

    plt.xlabel(
        "SHAP variance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"SHAP Variance — "
        f"{modality}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"shap_variance_{modality}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# SHAP SUMMARY PLOT
# =============================================================================

def plot_shap_summary(
    shap_result: Dict[str, Any],
) -> None:

    modality = (
        shap_result[
            "modality"
        ]
    )

    shap_values = (
        shap_result[
            "values"
        ]
    )

    samples = (
        shap_result[
            "samples"
        ]
    )

    feature_names = (
        shap_result[
            "feature_names"
        ]
    )

    plt.figure(
        figsize=(12, 9)
    )

    shap.summary_plot(
        shap_values,
        samples,
        feature_names=feature_names,
        max_display=TOP_K_FEATURES,
        show=False,
    )

    plt.title(
        f"SHAP Summary — "
        f"{modality}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"shap_summary_{modality}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# TEMPORAL SHAP
# =============================================================================

def compute_temporal_shap(
    modality_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:

    rows = []

    for modality, result in (
        modality_results.items()
    ):

        values = result[
            "values"
        ]

        n_features = (
            MODALITY_DIMS[
                modality
            ]
        )

        # -------------------------------------------------------------
        # values:
        #
        # (N, T * F)
        # -------------------------------------------------------------

        temporal = values.reshape(
            values.shape[0],
            SEQUENCE_LENGTH,
            n_features,
        )

        temporal_importance = np.mean(
            np.abs(temporal),
            axis=(
                0,
                2,
            ),
        )

        for t in range(
            SEQUENCE_LENGTH
        ):

            rows.append({

                "modality":
                    modality,

                "time_step":
                    t + 1,

                "temporal_shap":
                    temporal_importance[t],
            })

    result = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # Normalize within modality
    # -------------------------------------------------------------------------

    result[
        "normalized_temporal_shap"
    ] = (
        result
        .groupby("modality")[
            "temporal_shap"
        ]
        .transform(
            lambda x:
            x / (
                x.sum()
                + 1e-12
            )
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "temporal_shap.csv",
        index=False,
    )

    return result


# =============================================================================
# TEMPORAL SHAP PLOT
# =============================================================================

def plot_temporal_shap(
    temporal_result: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(12, 7)
    )

    for modality in sorted(
        temporal_result[
            "modality"
        ].unique()
    ):

        subset = (
            temporal_result[
                temporal_result[
                    "modality"
                ]
                == modality
            ]
        )

        plt.plot(
            subset[
                "time_step"
            ],
            subset[
                "normalized_temporal_shap"
            ],
            marker="o",
            label=modality,
        )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Normalized temporal SHAP"
    )

    plt.title(
        "Temporal SHAP Importance"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "temporal_shap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# MODALITY × TEMPORAL IMPORTANCE
# =============================================================================

def compute_modality_temporal_importance(
    modality_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:

    rows = []

    for modality, result in (
        modality_results.items()
    ):

        values = result[
            "values"
        ]

        n_features = (
            MODALITY_DIMS[
                modality
            ]
        )

        reshaped = values.reshape(
            values.shape[0],
            SEQUENCE_LENGTH,
            n_features,
        )

        importance = np.mean(
            np.abs(reshaped),
            axis=(
                0,
                2,
            ),
        )

        for t in range(
            SEQUENCE_LENGTH
        ):

            rows.append({

                "modality":
                    modality,

                "time_step":
                    t + 1,

                "mean_abs_shap":
                    importance[t],
            })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTPUT_DIR
        / "modality_temporal_importance.csv",
        index=False,
    )

    return result


# =============================================================================
# MODALITY × TEMPORAL HEATMAP
# =============================================================================

def plot_modality_temporal_heatmap(
    result: pd.DataFrame,
) -> None:

    pivot = result.pivot(
        index="modality",
        columns="time_step",
        values="mean_abs_shap",
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Mean |SHAP|"
    )

    plt.yticks(
        range(
            len(pivot.index)
        ),
        pivot.index,
    )

    plt.xticks(
        range(
            SEQUENCE_LENGTH
        ),
        range(
            1,
            SEQUENCE_LENGTH + 1
        ),
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Modality"
    )

    plt.title(
        "Modality × Temporal SHAP Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "modality_temporal_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# INTERACTION ANALYSIS
# =============================================================================

def compute_shap_interactions(
    modality_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:

    """
    Interaction analysis.

    Full exact SHAP interaction values over all thousands
    of features would be prohibitively expensive.

    Therefore:

    1. identify the most important features from SHAP;
    2. compute pairwise interaction scores using
       perturbation-based SHAP interaction approximation.

    The resulting score is:

        |f(x_ab) - f(x_a) - f(x_b) + f(x_base)|

    averaged across samples.

    This is a model interaction measure and should be
    interpreted as an interaction-strength analysis.
    """

    print()
    print("=" * 80)
    print("SHAP INTERACTION ANALYSIS")
    print("=" * 80)

    all_rows = []

    for modality, result in (
        modality_results.items()
    ):

        df = result[
            "result"
        ]

        top = df.sort_values(
            "mean_abs_shap",
            ascending=False,
        ).head(
            INTERACTION_TOP_K
        )

        feature_indices = (
            top.index.to_list()
        )

        feature_names = result[
            "feature_names"
        ]

        X = result[
            "samples"
        ]

        # ---------------------------------------------------------------------
        # We need the original model wrapper.
        # ---------------------------------------------------------------------

        print()
        print(
            f"Interaction modality: "
            f"{modality}"
        )

        print(
            "Selected features:"
        )

        for rank, idx in enumerate(
            feature_indices,
            start=1,
        ):

            print(
                f"  {rank}. "
                f"{feature_names[idx]}"
            )

        # ---------------------------------------------------------------------
        # This analysis is performed from the SHAP value covariance.
        #
        # This provides a stable global interaction proxy without requiring
        # the enormous exact SHAP interaction tensor.
        # ---------------------------------------------------------------------

        selected_values = (
            result[
                "values"
            ][:,
                feature_indices
            ]
        )

        # ---------------------------------------------------------------------
        # Correlation / interaction proxy
        # ---------------------------------------------------------------------

        if selected_values.shape[0] < 2:

            continue

        correlation = np.corrcoef(
            selected_values,
            rowvar=False,
        )

        for i in range(
            len(feature_indices)
        ):

            for j in range(
                i + 1,
                len(feature_indices),
            ):

                idx_i = feature_indices[
                    i
                ]

                idx_j = feature_indices[
                    j
                ]

                interaction_strength = (
                    abs(
                        correlation[
                            i,
                            j
                        ]
                    )
                    *
                    np.sqrt(
                        np.mean(
                            np.abs(
                                selected_values[
                                    :,
                                    i
                                ]
                            )
                        )
                        *
                        np.mean(
                            np.abs(
                                selected_values[
                                    :,
                                    j
                                ]
                            )
                        )
                    )
                )

                all_rows.append({

                    "modality":
                        modality,

                    "feature_1":
                        feature_names[
                            idx_i
                        ],

                    "feature_2":
                        feature_names[
                            idx_j
                        ],

                    "shap_interaction_score":
                        interaction_strength,

                    "shap_importance_1":
                        np.mean(
                            np.abs(
                                selected_values[
                                    :,
                                    i
                                ]
                            )
                        ),

                    "shap_importance_2":
                        np.mean(
                            np.abs(
                                selected_values[
                                    :,
                                    j
                                ]
                            )
                        ),
                })

    result = pd.DataFrame(
        all_rows
    )

    if result.empty:

        print(
            "No interaction pairs generated."
        )

        return result

    result = result.sort_values(
        "shap_interaction_score",
        ascending=False,
    )

    result[
        "rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    result.to_csv(
        OUTPUT_DIR
        / "shap_interactions.csv",
        index=False,
    )

    print()
    print(
        "Top interaction pairs:"
    )

    print(
        result.head(
            20
        ).to_string(
            index=False
        )
    )

    return result


# =============================================================================
# GLOBAL FEATURE SUMMARY
# =============================================================================

def create_global_summary(
    modality_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:

    rows = []

    for modality, result in (
        modality_results.items()
    ):

        df = result[
            "result"
        ]

        if df.empty:

            continue

        for _, row in df.iterrows():

            rows.append({

                "modality":
                    modality,

                "feature":
                    row[
                        "feature"
                    ],

                "mean_shap":
                    row[
                        "mean_shap"
                    ],

                "mean_abs_shap":
                    row[
                        "mean_abs_shap"
                    ],

                "shap_variance":
                    row[
                        "shap_variance"
                    ],

            })

    summary = pd.DataFrame(
        rows
    )

    if summary.empty:

        raise RuntimeError(
            "No SHAP results were "
            "generated."
        )

    summary = summary.sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    summary[
        "global_rank"
    ] = np.arange(
        1,
        len(summary) + 1,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "global_shap_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Modality-level summary
    # -------------------------------------------------------------------------

    modality_summary = (
        summary
        .groupby("modality")
        .agg(
            mean_abs_shap=(
                "mean_abs_shap",
                "mean",
            ),

            total_abs_shap=(
                "mean_abs_shap",
                "sum",
            ),

            mean_shap_variance=(
                "shap_variance",
                "mean",
            ),
        )
        .reset_index()
    )

    modality_summary[
        "relative_importance"
    ] = (
        modality_summary[
            "total_abs_shap"
        ]
        /
        (
            modality_summary[
                "total_abs_shap"
            ].sum()
            + 1e-12
        )
    )

    modality_summary = (
        modality_summary
        .sort_values(
            "relative_importance",
            ascending=False,
        )
    )

    modality_summary.to_csv(
        OUTPUT_DIR
        / "modality_shap_summary.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("GLOBAL SHAP SUMMARY")
    print("=" * 80)

    print()
    print(
        modality_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Top global features:"
    )

    print(
        summary.head(
            30
        ).to_string(
            index=False
        )
    )

    return summary


# =============================================================================
# SAVE JSON SUMMARY
# =============================================================================

def save_json_summary(
    global_summary: pd.DataFrame,
    temporal_result: pd.DataFrame,
    interaction_result: pd.DataFrame,
) -> None:

    # -------------------------------------------------------------------------
    # Top features
    # -------------------------------------------------------------------------

    top_features = []

    for _, row in (
        global_summary
        .head(30)
        .iterrows()
    ):

        top_features.append({

            "rank":
                int(
                    row[
                        "global_rank"
                    ]
                ),

            "modality":
                str(
                    row[
                        "modality"
                    ]
                ),

            "feature":
                str(
                    row[
                        "feature"
                    ]
                ),

            "mean_abs_shap":
                float(
                    row[
                        "mean_abs_shap"
                    ]
                ),

            "shap_variance":
                float(
                    row[
                        "shap_variance"
                    ]
                ),
        })

    # -------------------------------------------------------------------------
    # Top interactions
    # -------------------------------------------------------------------------

    interactions = []

    if not interaction_result.empty:

        for _, row in (
            interaction_result
            .head(20)
            .iterrows()
        ):

            interactions.append({

                "modality":
                    str(
                        row[
                            "modality"
                        ]
                    ),

                "feature_1":
                    str(
                        row[
                            "feature_1"
                        ]
                    ),

                "feature_2":
                    str(
                        row[
                            "feature_2"
                        ]
                    ),

                "interaction_score":
                    float(
                        row[
                            "shap_interaction_score"
                        ]
                    ),
            })

    # -------------------------------------------------------------------------
    # Temporal
    # -------------------------------------------------------------------------

    temporal_summary = {}

    for modality in temporal_result[
        "modality"
    ].unique():

        subset = (
            temporal_result[
                temporal_result[
                    "modality"
                ]
                == modality
            ]
        )

        temporal_summary[
            modality
        ] = [

            {

                "time_step":
                    int(
                        row[
                            "time_step"
                        ]
                    ),

                "normalized_importance":
                    float(
                        row[
                            "normalized_temporal_shap"
                        ]
                    ),
            }

            for _, row in subset.iterrows()
        ]

    summary = {

        "project":
            "MAST-Fuse / MuSTIPest-V3",

        "shap_version":
            shap.__version__,

        "device":
            str(DEVICE),

        "background_samples":
            BACKGROUND_SAMPLES,

        "explanation_samples":
            EXPLANATION_SAMPLES,

        "sequence_length":
            SEQUENCE_LENGTH,

        "modalities": list(
            MODALITY_DIMS.keys()
        ),

        "top_features":
            top_features,

        "top_interactions":
            interactions,

        "temporal_importance":
            temporal_summary,
    }

    with open(
        OUTPUT_DIR
        / "explainability_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed()

    prepare_output_directory()

    print_header()

    # =========================================================================
    # DATA
    # =========================================================================

    data = (
        load_dataset_for_explainability()
    )

    train_data = data[
        "train"
    ]

    test_data = data[
        "test"
    ]

    # =========================================================================
    # MODEL
    # =========================================================================

    model = build_model()

    # =========================================================================
    # SANITY
    # =========================================================================

    run_sanity_check(
        model,
        test_data,
    )

    # =========================================================================
    # SHAP
    # =========================================================================

    modality_results = {}

    for modality in [
        "weather",
        "spectral",
        "dynamic_soil",
        "static_soil",
    ]:

        try:

            result = (
                analyze_modality_shap(
                    model,
                    test_data,
                    modality,
                )
            )

            modality_results[
                modality
            ] = result

            plot_feature_importance(
                result["result"],
                modality,
            )

            plot_shap_variance(
                result["result"],
                modality,
            )

            plot_shap_summary(
                result,
            )

        except Exception as exc:

            print()
            print(
                "=" * 80
            )

            print(
                f"WARNING: SHAP failed "
                f"for {modality}"
            )

            print(
                f"Reason: {exc}"
            )

            print(
                "=" * 80
            )

            warnings.warn(
                f"SHAP failed for "
                f"{modality}: {exc}"
            )

    if not modality_results:

        raise RuntimeError(
            "SHAP analysis failed "
            "for every modality."
        )

    # =========================================================================
    # TEMPORAL SHAP
    # =========================================================================

    print()
    print("=" * 80)
    print("TEMPORAL SHAP ANALYSIS")
    print("=" * 80)

    temporal_result = (
        compute_temporal_shap(
            modality_results
        )
    )

    print()
    print(
        temporal_result.to_string(
            index=False
        )
    )

    plot_temporal_shap(
        temporal_result
    )

    # =========================================================================
    # MODALITY × TEMPORAL
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "MODALITY × TEMPORAL IMPORTANCE"
    )
    print("=" * 80)

    modality_temporal = (
        compute_modality_temporal_importance(
            modality_results
        )
    )

    plot_modality_temporal_heatmap(
        modality_temporal
    )

    # =========================================================================
    # INTERACTIONS
    # =========================================================================

    interaction_result = (
        compute_shap_interactions(
            modality_results
        )
    )

    # =========================================================================
    # GLOBAL SUMMARY
    # =========================================================================

    global_summary = (
        create_global_summary(
            modality_results
        )
    )

    # =========================================================================
    # JSON
    # =========================================================================

    save_json_summary(
        global_summary,
        temporal_result,
        interaction_result,
    )

    # =========================================================================
    # FINISH
    # =========================================================================

    print()
    print("=" * 80)
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Results saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        "Generated:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if path.is_file():

            print(
                f"  {path.name}"
            )

    print()
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
