#Author - Dipanwita Thakur

from pathlib import Path
import torch


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_NAME = "MAST-Fuse"


# =============================================================================
# RANDOM SEEDS
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011]


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# DATA
# =============================================================================

DATA_DIR = Path("./preprocessed")

TARGET_NAME = "aphids_per_plant"


# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

SEQ_DAYS = 7

STEPS_PER_DAY = 3

SEQ_LEN = SEQ_DAYS * STEPS_PER_DAY

# 21 timesteps → next timestep
PREDICTION_HORIZON = 1


# =============================================================================
# DATA SPLIT
# =============================================================================

TRAIN_YEARS = [2021, 2022, 2023]

VAL_YEARS = [2024]

TEST_YEARS = [2025]


# =============================================================================
# MODALITY FEATURES
# =============================================================================

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "cloud_cover",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]


SPECTRAL_FEATURES = [
    "NDVI",
    "NDWI",
    "EVI",
]


DYNAMIC_SOIL_FEATURES = [
    "BSI",
    "SAVI",
    "NDTI",
    "RI",
]


STATIC_SOIL_FEATURES = [
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


# =============================================================================
# INPUT DIMENSIONS
# =============================================================================

WEATHER_DIM = len(WEATHER_FEATURES)
SPECTRAL_DIM = len(SPECTRAL_FEATURES)
DYNAMIC_SOIL_DIM = len(DYNAMIC_SOIL_FEATURES)
STATIC_SOIL_DIM = len(STATIC_SOIL_FEATURES)


# =============================================================================
# MODEL
# =============================================================================

FUSION_DIM = 128

NUM_HEADS = 4

DROPOUT = 0.10


# Temporal encoder
TEMPORAL_ENCODER = "tcn"


# Fusion options:
#   "concat"
#   "attention"
#   "cross_attention"
FUSION_TYPE = "cross_attention"


# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE = 32

EPOCHS = 150

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0


# =============================================================================
# LOSS
# =============================================================================

LOSS_FUNCTION = "huber"

HUBER_DELTA = 1.0


# =============================================================================
# SCHEDULER
# =============================================================================

USE_SCHEDULER = True

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 5

MIN_LR = 1e-7


# =============================================================================
# MODALITY ENABLE/DISABLE
# =============================================================================

USE_WEATHER = True

USE_SPECTRAL = True

USE_DYNAMIC_SOIL = True

USE_STATIC_SOIL = True


# =============================================================================
# NUMERICAL SAFETY
# =============================================================================

CHECK_FINITE = True

USE_AMP = False


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = Path("./outputs")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

RESULTS_DIR = OUTPUT_DIR / "results"

FIGURE_DIR = OUTPUT_DIR / "figures"

LOG_DIR = OUTPUT_DIR / "logs"


# =============================================================================
# DATA FILES
# =============================================================================

WEATHER_TRAIN = DATA_DIR / "X_weather_train.npy"
WEATHER_VAL = DATA_DIR / "X_weather_val.npy"
WEATHER_TEST = DATA_DIR / "X_weather_test.npy"

SPECTRAL_TRAIN = DATA_DIR / "X_spectral_train.npy"
SPECTRAL_VAL = DATA_DIR / "X_spectral_val.npy"
SPECTRAL_TEST = DATA_DIR / "X_spectral_test.npy"

DYNAMIC_SOIL_TRAIN = DATA_DIR / "X_dynamic_soil_train.npy"
DYNAMIC_SOIL_VAL = DATA_DIR / "X_dynamic_soil_val.npy"
DYNAMIC_SOIL_TEST = DATA_DIR / "X_dynamic_soil_test.npy"

STATIC_SOIL_TRAIN = DATA_DIR / "X_static_soil_train.npy"
STATIC_SOIL_VAL = DATA_DIR / "X_static_soil_val.npy"
STATIC_SOIL_TEST = DATA_DIR / "X_static_soil_test.npy"

Y_TRAIN = DATA_DIR / "y_train.npy"
Y_VAL = DATA_DIR / "y_val.npy"
Y_TEST = DATA_DIR / "y_test.npy"


# =============================================================================
# VALIDATION
# =============================================================================

EXPECTED_SHAPES = {
    "weather": (SEQ_LEN, WEATHER_DIM),
    "spectral": (SEQ_LEN, SPECTRAL_DIM),
    "dynamic_soil": (SEQ_LEN, DYNAMIC_SOIL_DIM),
    "static_soil": (SEQ_LEN, STATIC_SOIL_DIM),
}


def print_config():
    """Print experiment configuration."""

    print("\n" + "=" * 80)
    print(PROJECT_NAME)
    print("=" * 80)

    print(f"Device          : {DEVICE}")
    print(f"Sequence        : {SEQ_DAYS} days")
    print(f"Steps/day       : {STEPS_PER_DAY}")
    print(f"Sequence length : {SEQ_LEN}")
    print(f"Prediction      : next timestep")

    print("\nData split")
    print(f"  Train         : {TRAIN_YEARS}")
    print(f"  Validation    : {VAL_YEARS}")
    print(f"  Test          : {TEST_YEARS}")

    print("\nModalities")
    print(f"  Weather       : {WEATHER_DIM}")
    print(f"  Spectral      : {SPECTRAL_DIM}")
    print(f"  Dynamic soil  : {DYNAMIC_SOIL_DIM}")
    print(f"  Static soil   : {STATIC_SOIL_DIM}")

    print("\nModel")
    print(f"  Temporal      : {TEMPORAL_ENCODER}")
    print(f"  Fusion        : {FUSION_TYPE}")
    print(f"  Fusion dim    : {FUSION_DIM}")
    print(f"  Attention heads: {NUM_HEADS}")
    print(f"  Dropout       : {DROPOUT}")

    print("\nTraining")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  LR            : {LEARNING_RATE}")
    print(f"  Weight decay  : {WEIGHT_DECAY}")
    print(f"  Grad clip     : {GRADIENT_CLIP}")

    print("=" * 80)


if __name__ == "__main__":
    print_config()
