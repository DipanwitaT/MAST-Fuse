# Author - Dipanwita Thakur
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Import project configuration
from config import (
    DATA_DIR,
    SEQ_LEN,

    WEATHER_DIM,
    SPECTRAL_DIM,
    DYNAMIC_SOIL_DIM,
    STATIC_SOIL_DIM,

    WEATHER_TRAIN,
    WEATHER_VAL,
    WEATHER_TEST,

    SPECTRAL_TRAIN,
    SPECTRAL_VAL,
    SPECTRAL_TEST,

    DYNAMIC_SOIL_TRAIN,
    DYNAMIC_SOIL_VAL,
    DYNAMIC_SOIL_TEST,

    STATIC_SOIL_TRAIN,
    STATIC_SOIL_VAL,
    STATIC_SOIL_TEST,

    Y_TRAIN,
    Y_VAL,
    Y_TEST,

    BATCH_SIZE,
    SEEDS,
    DEVICE,

    CHECK_FINITE,
)


# =============================================================================
# EXPECTED SHAPES
# =============================================================================

EXPECTED_DIMS = {
    "weather": WEATHER_DIM,
    "spectral": SPECTRAL_DIM,
    "dynamic_soil": DYNAMIC_SOIL_DIM,
    "static_soil": STATIC_SOIL_DIM,
}


EXPECTED_MODALITY_SHAPES = {
    "weather": (SEQ_LEN, WEATHER_DIM),
    "spectral": (SEQ_LEN, SPECTRAL_DIM),
    "dynamic_soil": (SEQ_LEN, DYNAMIC_SOIL_DIM),
    "static_soil": (SEQ_LEN, STATIC_SOIL_DIM),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int = 42):
    """
    Set all relevant random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_npy(path):
    """
    Load a NumPy array from disk.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    np.ndarray
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset file not found:\n  {path.resolve()}"
        )

    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load NumPy file:\n"
            f"  {path.resolve()}\n"
            f"Error: {exc}"
        ) from exc

    return array


def check_array_finite(
    array,
    name,
    allow_nan=False,
):
    """
    Check an array for NaN and Inf values.
    """

    array = np.asarray(array)

    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())

    if nan_count > 0 or inf_count > 0:

        print(
            f"\n[ERROR] Non-finite values in {name}"
        )

        print(f"  Shape : {array.shape}")
        print(f"  NaN   : {nan_count}")
        print(f"  Inf   : {inf_count}")

        if not allow_nan:
            raise ValueError(
                f"{name} contains NaN/Inf values."
            )


def check_array_numeric(array, name):
    """
    Verify that an array is numeric.
    """

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(
            f"{name} is not numeric. "
            f"dtype={array.dtype}"
        )


def check_modality_shape(
    array,
    name,
    expected_shape,
):
    """
    Validate modality shape.

    Expected:
        (N, SEQ_LEN, FEATURES)
    """

    if array.ndim != 3:

        raise ValueError(
            f"\n{name} must be a 3-D tensor.\n"
            f"Expected: (N, {expected_shape[0]}, {expected_shape[1]})\n"
            f"Actual  : {array.shape}"
        )

    if array.shape[1] != expected_shape[0]:

        raise ValueError(
            f"\n{name} has incorrect sequence length.\n"
            f"Expected sequence length: {expected_shape[0]}\n"
            f"Actual: {array.shape[1]}"
        )

    if array.shape[2] != expected_shape[1]:

        raise ValueError(
            f"\n{name} has incorrect feature dimension.\n"
            f"Expected features: {expected_shape[1]}\n"
            f"Actual: {array.shape[2]}"
        )


def check_target_shape(y, name):
    """
    Validate target shape.

    Target must be one-dimensional.
    """

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)

    if y.ndim != 1:

        raise ValueError(
            f"\n{name} must be 1-D.\n"
            f"Expected: (N,)\n"
            f"Actual: {y.shape}"
        )

    return y


def print_array_statistics(array, name):
    """
    Print useful numerical statistics.
    """

    array = np.asarray(array)

    print(f"\n{name}")
    print(f"  Shape : {array.shape}")
    print(f"  Dtype : {array.dtype}")

    if array.size == 0:
        print("  Empty array")
        return

    finite = np.isfinite(array)

    if not finite.all():
        print("  Contains non-finite values.")
        return

    print(f"  Min   : {array.min():.6f}")
    print(f"  Max   : {array.max():.6f}")
    print(f"  Mean  : {array.mean():.6f}")
    print(f"  Std   : {array.std():.6f}")


# =============================================================================
# MODALITY CONTAINER
# =============================================================================

class MultimodalData:
    """
    Container for one train/validation/test split.

    This class keeps all four modalities together and provides
    consistency checks.
    """

    def __init__(
        self,
        weather,
        spectral,
        dynamic_soil,
        static_soil,
        target,
        split_name="unknown",
    ):

        self.split_name = split_name

        self.weather = weather
        self.spectral = spectral
        self.dynamic_soil = dynamic_soil
        self.static_soil = static_soil
        self.target = target

        self.validate()

    @property
    def num_samples(self):
        return len(self.target)

    def validate(self):

        arrays = {
            "weather": self.weather,
            "spectral": self.spectral,
            "dynamic_soil": self.dynamic_soil,
            "static_soil": self.static_soil,
        }

        # ---------------------------------------------------------
        # Validate modality shapes
        # ---------------------------------------------------------

        for name, array in arrays.items():

            expected = EXPECTED_MODALITY_SHAPES[name]

            check_array_numeric(array, name)

            check_modality_shape(
                array,
                name,
                expected,
            )

            if CHECK_FINITE:
                check_array_finite(
                    array,
                    name,
                )

        # ---------------------------------------------------------
        # Validate target
        # ---------------------------------------------------------

        self.target = check_target_shape(
            self.target,
            "target",
        )

        check_array_numeric(
            self.target,
            "target",
        )

        if CHECK_FINITE:
            check_array_finite(
                self.target,
                "target",
            )

        # ---------------------------------------------------------
        # Same number of samples
        # ---------------------------------------------------------

        n = len(self.target)

        for name, array in arrays.items():

            if len(array) != n:

                raise ValueError(
                    f"\nSample count mismatch in {self.split_name}.\n"
                    f"Target : {n}\n"
                    f"{name} : {len(array)}"
                )

    def summary(self):

        print("\n" + "=" * 70)
        print(f"DATA SUMMARY — {self.split_name.upper()}")
        print("=" * 70)

        print(f"Samples        : {self.num_samples}")

        print(
            f"Weather        : "
            f"{self.weather.shape}"
        )

        print(
            f"Spectral       : "
            f"{self.spectral.shape}"
        )

        print(
            f"Dynamic soil   : "
            f"{self.dynamic_soil.shape}"
        )

        print(
            f"Static soil    : "
            f"{self.static_soil.shape}"
        )

        print(
            f"Target         : "
            f"{self.target.shape}"
        )

        print("\nTarget statistics")

        print(
            f"  Min          : "
            f"{self.target.min():.6f}"
        )

        print(
            f"  Max          : "
            f"{self.target.max():.6f}"
        )

        print(
            f"  Mean         : "
            f"{self.target.mean():.6f}"
        )

        print(
            f"  Median       : "
            f"{np.median(self.target):.6f}"
        )

        print(
            f"  Std          : "
            f"{self.target.std():.6f}"
        )

        print("=" * 70)


# =============================================================================
# LOAD SPLIT
# =============================================================================

def load_split(
    weather_path,
    spectral_path,
    dynamic_soil_path,
    static_soil_path,
    target_path,
    split_name,
):
    """
    Load one complete multimodal split.
    """

    print("\n" + "=" * 70)
    print(f"LOADING {split_name.upper()} DATA")
    print("=" * 70)

    weather = load_npy(weather_path)

    spectral = load_npy(spectral_path)

    dynamic_soil = load_npy(dynamic_soil_path)

    static_soil = load_npy(static_soil_path)

    target = load_npy(target_path)

    # ---------------------------------------------------------
    # Target reshape
    # ---------------------------------------------------------

    target = check_target_shape(
        target,
        f"{split_name} target",
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    data = MultimodalData(
        weather=weather,
        spectral=spectral,
        dynamic_soil=dynamic_soil,
        static_soil=static_soil,
        target=target,
        split_name=split_name,
    )

    data.summary()

    return data


# =============================================================================
# DATASET
# =============================================================================

class MuSTIPestDataset(Dataset):
    """
    PyTorch Dataset for MuSTIPest multimodal prediction.

    Each sample contains:

        weather
        spectral
        dynamic_soil
        static_soil
        target

    Modality ablation is supported through the use_* arguments.
    """

    def __init__(
        self,
        data: MultimodalData,
        use_weather=True,
        use_spectral=True,
        use_dynamic_soil=True,
        use_static_soil=True,
    ):

        super().__init__()

        self.weather = torch.from_numpy(
            data.weather.astype(np.float32)
        )

        self.spectral = torch.from_numpy(
            data.spectral.astype(np.float32)
        )

        self.dynamic_soil = torch.from_numpy(
            data.dynamic_soil.astype(np.float32)
        )

        self.static_soil = torch.from_numpy(
            data.static_soil.astype(np.float32)
        )

        self.target = torch.from_numpy(
            data.target.astype(np.float32)
        )

        self.use_weather = use_weather
        self.use_spectral = use_spectral
        self.use_dynamic_soil = use_dynamic_soil
        self.use_static_soil = use_static_soil

        if not any([
            use_weather,
            use_spectral,
            use_dynamic_soil,
            use_static_soil,
        ]):

            raise ValueError(
                "At least one modality must be enabled."
            )

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):

        sample = {}

        if self.use_weather:
            sample["weather"] = self.weather[index]

        if self.use_spectral:
            sample["spectral"] = self.spectral[index]

        if self.use_dynamic_soil:
            sample["dynamic_soil"] = self.dynamic_soil[index]

        if self.use_static_soil:
            sample["static_soil"] = self.static_soil[index]

        sample["target"] = self.target[index]

        return sample


# =============================================================================
# DATALOADER
# =============================================================================

def create_dataloader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=True,
):
    """
    Create PyTorch DataLoader.

    num_workers=0 is intentionally used as the safe Windows default.
    """

    if num_workers < 0:
        raise ValueError(
            "num_workers must be >= 0."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=(
            pin_memory
            and torch.cuda.is_available()
        ),
    )


# =============================================================================
# LOAD ALL DATA
# =============================================================================

def load_all_data():
    """
    Load train, validation and test datasets.
    """

    train = load_split(
        WEATHER_TRAIN,
        SPECTRAL_TRAIN,
        DYNAMIC_SOIL_TRAIN,
        STATIC_SOIL_TRAIN,
        Y_TRAIN,
        "train",
    )

    val = load_split(
        WEATHER_VAL,
        SPECTRAL_VAL,
        DYNAMIC_SOIL_VAL,
        STATIC_SOIL_VAL,
        Y_VAL,
        "validation",
    )

    test = load_split(
        WEATHER_TEST,
        SPECTRAL_TEST,
        DYNAMIC_SOIL_TEST,
        STATIC_SOIL_TEST,
        Y_TEST,
        "test",
    )

    return train, val, test


# =============================================================================
# CREATE DATA LOADERS
# =============================================================================

def create_dataloaders(
    batch_size=BATCH_SIZE,
    seed=42,

    use_weather=True,
    use_spectral=True,
    use_dynamic_soil=True,
    use_static_soil=True,

    num_workers=0,
):
    """
    Create train/validation/test DataLoaders.

    This function is also used by modality ablation experiments.
    """

    set_seed(seed)

    train_data, val_data, test_data = load_all_data()

    train_dataset = MuSTIPestDataset(
        train_data,
        use_weather=use_weather,
        use_spectral=use_spectral,
        use_dynamic_soil=use_dynamic_soil,
        use_static_soil=use_static_soil,
    )

    val_dataset = MuSTIPestDataset(
        val_data,
        use_weather=use_weather,
        use_spectral=use_spectral,
        use_dynamic_soil=use_dynamic_soil,
        use_static_soil=use_static_soil,
    )

    test_dataset = MuSTIPestDataset(
        test_data,
        use_weather=use_weather,
        use_spectral=use_spectral,
        use_dynamic_soil=use_dynamic_soil,
        use_static_soil=use_static_soil,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# =============================================================================
# FIRST BATCH INSPECTION
# =============================================================================

def inspect_batch(
    loader,
    device=None,
):
    """
    Inspect one batch.

    Useful before starting model training.
    """

    if device is None:
        device = DEVICE

    batch = next(iter(loader))

    print("\n" + "=" * 70)
    print("FIRST BATCH INSPECTION")
    print("=" * 70)

    for key, value in batch.items():

        print(
            f"{key:15s} "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype}"
        )

        if CHECK_FINITE:

            if not torch.isfinite(value).all():

                raise ValueError(
                    f"Non-finite values found in "
                    f"batch['{key}']"
                )

    print("\nExpected shapes")

    for key, value in batch.items():

        print(
            f"  {key:15s}: "
            f"{tuple(value.shape)}"
        )

    print("=" * 70)

    return batch


# =============================================================================
# MODALITY ABLATION CONFIGURATIONS
# =============================================================================

def get_modality_ablation_configs():
    """
    Return modality-ablation experiments.

    FULL:
        Weather + Spectral + Dynamic Soil + Static Soil

    Individual modality removal:
        -W
        -S
        -DS
        -SS

    Single-modality models are also provided.
    """

    return {

        # -----------------------------------------------------
        # Full model
        # -----------------------------------------------------

        "full": {
            "use_weather": True,
            "use_spectral": True,
            "use_dynamic_soil": True,
            "use_static_soil": True,
        },

        # -----------------------------------------------------
        # Leave-one-modality-out
        # -----------------------------------------------------

        "without_weather": {
            "use_weather": False,
            "use_spectral": True,
            "use_dynamic_soil": True,
            "use_static_soil": True,
        },

        "without_spectral": {
            "use_weather": True,
            "use_spectral": False,
            "use_dynamic_soil": True,
            "use_static_soil": True,
        },

        "without_dynamic_soil": {
            "use_weather": True,
            "use_spectral": True,
            "use_dynamic_soil": False,
            "use_static_soil": True,
        },

        "without_static_soil": {
            "use_weather": True,
            "use_spectral": True,
            "use_dynamic_soil": True,
            "use_static_soil": False,
        },

        # -----------------------------------------------------
        # Single modality
        # -----------------------------------------------------

        "weather_only": {
            "use_weather": True,
            "use_spectral": False,
            "use_dynamic_soil": False,
            "use_static_soil": False,
        },

        "spectral_only": {
            "use_weather": False,
            "use_spectral": True,
            "use_dynamic_soil": False,
            "use_static_soil": False,
        },

        "dynamic_soil_only": {
            "use_weather": False,
            "use_spectral": False,
            "use_dynamic_soil": True,
            "use_static_soil": False,
        },

        "static_soil_only": {
            "use_weather": False,
            "use_spectral": False,
            "use_dynamic_soil": False,
            "use_static_soil": True,
        },
    }


# =============================================================================
# SAVE DATASET DESCRIPTION
# =============================================================================

def save_dataset_description(
    output_path=None,
):
    """
    Save a machine-readable description of the dataset.
    """

    if output_path is None:
        output_path = DATA_DIR / "dataset_description.json"

    description = {

        "project": "MuSTIPest-V3",

        "temporal": {
            "sequence_days": 7,
            "steps_per_day": 3,
            "sequence_length": 21,
            "prediction_horizon": 1,
        },

        "modalities": {

            "weather": {
                "features": WEATHER_DIM,
                "shape": [
                    "N",
                    SEQ_LEN,
                    WEATHER_DIM,
                ],
            },

            "spectral": {
                "features": SPECTRAL_DIM,
                "variables": [
                    "NDVI",
                    "NDWI",
                    "EVI",
                ],
                "shape": [
                    "N",
                    SEQ_LEN,
                    SPECTRAL_DIM,
                ],
            },

            "dynamic_soil": {
                "features": DYNAMIC_SOIL_DIM,
                "variables": [
                    "BSI",
                    "SAVI",
                    "NDTI",
                    "RI",
                ],
                "shape": [
                    "N",
                    SEQ_LEN,
                    DYNAMIC_SOIL_DIM,
                ],
            },

            "static_soil": {
                "features": STATIC_SOIL_DIM,
                "shape": [
                    "N",
                    SEQ_LEN,
                    STATIC_SOIL_DIM,
                ],
            },
        },

        "target": {
            "name": "aphids_per_plant",
            "shape": [
                "N"
            ],
        },

        "splits": {
            "train": "2021-2023",
            "validation": "2024",
            "test": "2025",
        },
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            description,
            f,
            indent=4,
        )

    print(
        f"\nDataset description saved to:\n"
        f"  {output_path.resolve()}"
    )


# =============================================================================
# MAIN TEST
# =============================================================================

def main():

    print("\n" + "=" * 80)
    print("MuSTIPest-V3 DATASET MODULE")
    print("=" * 80)

    print(f"Data directory : {DATA_DIR.resolve()}")

    print("\nExpected dimensions")

    print(
        f"  Weather       : "
        f"(N, {SEQ_LEN}, {WEATHER_DIM})"
    )

    print(
        f"  Spectral      : "
        f"(N, {SEQ_LEN}, {SPECTRAL_DIM})"
    )

    print(
        f"  Dynamic soil  : "
        f"(N, {SEQ_LEN}, {DYNAMIC_SOIL_DIM})"
    )

    print(
        f"  Static soil   : "
        f"(N, {SEQ_LEN}, {STATIC_SOIL_DIM})"
    )

    print(
        f"  Target        : "
        f"(N,)"
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    train_data, val_data, test_data = load_all_data()

    # ---------------------------------------------------------
    # Create full-model loaders
    # ---------------------------------------------------------

    train_loader = create_dataloader(
        MuSTIPestDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = create_dataloader(
        MuSTIPestDataset(val_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = create_dataloader(
        MuSTIPestDataset(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Inspect
    # ---------------------------------------------------------

    inspect_batch(
        train_loader
    )

    # ---------------------------------------------------------
    # Save description
    # ---------------------------------------------------------

    save_dataset_description()

    print("\nDataset module test completed successfully.")


if __name__ == "__main__":
    main()
