"""
MuSTIPest-V3
============================================================

Preprocessing pipeline for multimodal soybean aphid prediction.

Input:
    multimodal_LV_TARGET_ALL_SITES_2021_2025.csv

Modalities:
    1. Weather
    2. Vegetation spectral indices
    3. Dynamic soil indices
    4. Static soil properties

Target:
    aphids_per_plant

Temporal resolution:
    3 observations/day:
        00:00
        08:00
        16:00

Sequence:
    7 days = 21 timesteps

Prediction:
    next timestep target

IMPORTANT:
    LV latent variables and LV parameters are NOT model inputs.
    They are retained only for auditing/analysis.

Output:
    preprocessed/
        X_weather_train.npy
        X_weather_val.npy
        X_weather_test.npy

        X_spectral_train.npy
        X_spectral_val.npy
        X_spectral_test.npy

        X_dynamic_soil_train.npy
        X_dynamic_soil_val.npy
        X_dynamic_soil_test.npy

        X_static_soil_train.npy
        X_static_soil_val.npy
        X_static_soil_test.npy

        y_train.npy
        y_val.npy
        y_test.npy

        metadata.npz

        preprocessing_summary.txt
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "multimodal_LV_TARGET_ALL_SITES_2021_2025.csv"
)

OUTPUT_DIR = Path("preprocessed")

SEQ_DAYS = 7
STEPS_PER_DAY = 3
SEQ_LEN = SEQ_DAYS * STEPS_PER_DAY

TRAIN_YEARS = [2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]

TARGET_COL = "aphids_per_plant"


# ============================================================
# MODALITIES
# ============================================================

WEATHER_COLUMNS = [
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

SPECTRAL_COLUMNS = [
    "NDVI",
    "NDWI",
    "EVI",
]

DYNAMIC_SOIL_COLUMNS = [
    "BSI",
    "SAVI",
    "NDTI",
    "RI",
]

STATIC_SOIL_COLUMNS = [
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


# ============================================================
# FEATURES USED BY MODEL
# ============================================================

# These variables are deliberately excluded:
#
# aphid_latent_N
# predator_latent_P
# lv_r
# lv_K
# lv_alpha
# lv_beta
# lv_delta
# temperature_suitability
# humidity_suitability
# vegetation_suitability
# soil_suitability
#
# because they are part of the synthetic LV target-generation
# mechanism and would introduce target leakage.

LEAKAGE_COLUMNS = [
    "aphid_latent_N",
    "predator_latent_P",
    "lv_r",
    "lv_K",
    "lv_alpha",
    "lv_beta",
    "lv_delta",
    "temperature_suitability",
    "humidity_suitability",
    "vegetation_suitability",
    "soil_suitability",
]


# ============================================================
# TEMPORAL FEATURES
# ============================================================

TIME_FEATURES = [
    "doy_sin",
    "doy_cos",
    "window_sin",
    "window_cos",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 80)
    print("LOADING MULTIMODAL LV TARGET DATA")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"File   : {INPUT_FILE}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required = (
        ["site_name", "year", "date", "time_local",
         "collection_window", TARGET_COL]
        + WEATHER_COLUMNS
        + SPECTRAL_COLUMNS
        + DYNAMIC_SOIL_COLUMNS
        + STATIC_SOIL_COLUMNS
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# NUMERICAL VALIDATION
# ============================================================

def validate_raw_data(df):

    print("\n" + "=" * 80)
    print("VALIDATING RAW DATA")
    print("=" * 80)

    # --------------------------------------------------------
    # duplicates
    # --------------------------------------------------------

    key = [
        "site_name",
        "date",
        "time_local",
    ]

    duplicates = df.duplicated(key).sum()

    print(f"Temporal duplicates: {duplicates}")

    if duplicates > 0:

        print(
            "WARNING: duplicate temporal observations detected."
        )

        df = (
            df.sort_values(key)
              .groupby(key, as_index=False)
              .mean(numeric_only=True)
        )

        # Restore categorical fields if necessary.
        # Normally your current file should have no duplicates.

    # --------------------------------------------------------
    # missing values
    # --------------------------------------------------------

    numeric_cols = (
        WEATHER_COLUMNS
        + SPECTRAL_COLUMNS
        + DYNAMIC_SOIL_COLUMNS
        + STATIC_SOIL_COLUMNS
        + [TARGET_COL]
    )

    missing = df[numeric_cols].isna().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:

        print("\nMissing values:")

        print(missing)

        raise ValueError(
            "\nInput data contains missing values.\n"
            "Fix the synchronized dataset before preprocessing."
        )

    # --------------------------------------------------------
    # infinite values
    # --------------------------------------------------------

    numeric = df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    inf_count = np.isinf(
        numeric.to_numpy(dtype=np.float64)
    ).sum()

    print(f"NaN count : {numeric.isna().sum().sum()}")
    print(f"Inf count : {inf_count}")

    if inf_count > 0:

        raise ValueError(
            "Infinite values detected in input dataset."
        )

    # --------------------------------------------------------
    # target
    # --------------------------------------------------------

    y = df[TARGET_COL].astype(float)

    print("\nTarget statistics:")
    print(f"  min    : {y.min():.6f}")
    print(f"  p1     : {np.percentile(y, 1):.6f}")
    print(f"  median : {np.median(y):.6f}")
    print(f"  p99    : {np.percentile(y, 99):.6f}")
    print(f"  max    : {y.max():.6f}")
    print(f"  mean   : {y.mean():.6f}")
    print(f"  std    : {y.std():.6f}")
    print(f"  zeros  : {(y == 0).sum()}")

    if (y < 0).any():

        raise ValueError(
            "Negative aphid target values detected."
        )

    return df


# ============================================================
# TEMPORAL ENGINEERING
# ============================================================

def engineer_temporal_features(df):

    print("\n" + "=" * 80)
    print("ENGINEERING TEMPORAL FEATURES")
    print("=" * 80)

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------------
    # day of year
    # --------------------------------------------------------

    df["doy"] = df["date"].dt.dayofyear

    df["doy_sin"] = np.sin(
        2 * np.pi * df["doy"] / 365.25
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["doy"] / 365.25
    )

    # --------------------------------------------------------
    # collection window
    # --------------------------------------------------------

    window_map = {
        "Window_1_00h": 0,
        "Window_2_08h": 1,
        "Window_3_16h": 2,
    }

    df["window_id"] = (
        df["collection_window"]
        .map(window_map)
    )

    if df["window_id"].isna().any():

        unknown = (
            df.loc[
                df["window_id"].isna(),
                "collection_window"
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown collection windows: {unknown}"
        )

    df["window_sin"] = np.sin(
        2 * np.pi * df["window_id"] / 3
    )

    df["window_cos"] = np.cos(
        2 * np.pi * df["window_id"] / 3
    )

    print(
        "Added:",
        ", ".join(TIME_FEATURES)
    )

    return df


# ============================================================
# SORTING
# ============================================================

def sort_temporally(df):

    df = df.copy()

    df["time_order"] = (
        df["time_local"]
        .map({
            "00:00": 0,
            "08:00": 1,
            "16:00": 2,
        })
    )

    if df["time_order"].isna().any():

        print(
            "WARNING: unexpected time values:",
            df.loc[
                df["time_order"].isna(),
                "time_local"
            ].unique()
        )

        # fallback
        df["time_order"] = pd.to_datetime(
            df["time_local"],
            format="%H:%M",
            errors="coerce"
        ).dt.hour

    df = (
        df.sort_values(
            [
                "site_name",
                "date",
                "time_order",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SPLIT BY YEAR
# ============================================================

def split_by_year(df):

    print("\n" + "=" * 80)
    print("CHRONOLOGICAL YEAR SPLIT")
    print("=" * 80)

    train = df[
        df["year"].isin(TRAIN_YEARS)
    ].copy()

    val = df[
        df["year"].isin(VAL_YEARS)
    ].copy()

    test = df[
        df["year"].isin(TEST_YEARS)
    ].copy()

    print(
        f"Train: {TRAIN_YEARS} "
        f"→ {len(train):,} rows"
    )

    print(
        f"Val  : {VAL_YEARS} "
        f"→ {len(val):,} rows"
    )

    print(
        f"Test : {TEST_YEARS} "
        f"→ {len(test):,} rows"
    )

    return train, val, test


# ============================================================
# SCALE DATA
# ============================================================

def fit_scalers(train_df):

    print("\n" + "=" * 80)
    print("FITTING SCALERS ON TRAINING DATA ONLY")
    print("=" * 80)

    scalers = {}

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    scalers["weather"] = RobustScaler()

    scalers["weather"].fit(
        train_df[WEATHER_COLUMNS]
    )

    print(
        f"Weather scaler : RobustScaler "
        f"→ {len(WEATHER_COLUMNS)} features"
    )

    # --------------------------------------------------------
    # Spectral
    # --------------------------------------------------------

    scalers["spectral"] = MinMaxScaler()

    scalers["spectral"].fit(
        train_df[SPECTRAL_COLUMNS]
    )

    print(
        f"Spectral scaler: MinMaxScaler "
        f"→ {len(SPECTRAL_COLUMNS)} features"
    )

    # --------------------------------------------------------
    # Dynamic soil
    # --------------------------------------------------------

    scalers["dynamic_soil"] = RobustScaler()

    scalers["dynamic_soil"].fit(
        train_df[DYNAMIC_SOIL_COLUMNS]
    )

    print(
        f"Dynamic soil  : RobustScaler "
        f"→ {len(DYNAMIC_SOIL_COLUMNS)} features"
    )

    # --------------------------------------------------------
    # Static soil
    # --------------------------------------------------------

    scalers["static_soil"] = RobustScaler()

    scalers["static_soil"].fit(
        train_df[STATIC_SOIL_COLUMNS]
    )

    print(
        f"Static soil   : RobustScaler "
        f"→ {len(STATIC_SOIL_COLUMNS)} features"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    scalers["target"] = MinMaxScaler()

    scalers["target"].fit(
        train_df[[TARGET_COL]]
    )

    print(
        f"Target scaler  : MinMaxScaler "
        f"→ {TARGET_COL} [0,1]"
    )

    return scalers


# ============================================================
# APPLY SCALERS
# ============================================================

def transform_dataframe(df, scalers):

    df = df.copy()

    df[WEATHER_COLUMNS] = (
        scalers["weather"]
        .transform(df[WEATHER_COLUMNS])
    )

    df[SPECTRAL_COLUMNS] = (
        scalers["spectral"]
        .transform(df[SPECTRAL_COLUMNS])
    )

    df[DYNAMIC_SOIL_COLUMNS] = (
        scalers["dynamic_soil"]
        .transform(df[DYNAMIC_SOIL_COLUMNS])
    )

    df[STATIC_SOIL_COLUMNS] = (
        scalers["static_soil"]
        .transform(df[STATIC_SOIL_COLUMNS])
    )

    df[TARGET_COL] = (
        scalers["target"]
        .transform(df[[TARGET_COL]])
        .ravel()
    )

    return df


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(df):

    """
    Create 7-day / 21-timestep sequences.

    Input:
        t-20 ... t

    Target:
        t+1

    Each sample therefore predicts the next observation.

    The sequence must remain within:
        - same site
        - same year

    This prevents sequences from crossing site boundaries
    or annual boundaries.
    """

    X_weather = []
    X_spectral = []
    X_dynamic = []
    X_static = []
    y = []

    groups = df.groupby(
        ["site_name", "year"],
        sort=False
    )

    for (site, year), group in groups:

        group = group.sort_values(
            ["date", "time_order"]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Check temporal continuity
        # ----------------------------------------------------

        timestamps = pd.to_datetime(
            group["date"].astype(str)
            + " "
            + group["time_local"].astype(str)
        )

        dt = timestamps.diff()

        # Expected:
        #
        # 8 hours inside a day
        # 16 hours from 16:00 -> next 00:00
        #
        # A missing observation means we should not create a
        # sequence crossing that gap.

        valid_step = (
            dt.iloc[1:].dt.total_seconds()
            .isin([8 * 3600, 16 * 3600])
        )

        # ----------------------------------------------------
        # Break sequence into continuous segments
        # ----------------------------------------------------

        segment_id = (
            (~valid_step)
            .cumsum()
        )

        # Align because valid_step starts at second row.
        segment_id = pd.Series(
            [0] + segment_id.tolist()
        )

        group["_segment"] = segment_id.values

        # ----------------------------------------------------
        # Process each continuous segment
        # ----------------------------------------------------

        for _, seg in group.groupby(
            "_segment",
            sort=False
        ):

            seg = seg.reset_index(drop=True)

            if len(seg) < SEQ_LEN + 1:
                continue

            weather = (
                seg[WEATHER_COLUMNS]
                .to_numpy(dtype=np.float32)
            )

            spectral = (
                seg[SPECTRAL_COLUMNS]
                .to_numpy(dtype=np.float32)
            )

            dynamic = (
                seg[DYNAMIC_SOIL_COLUMNS]
                .to_numpy(dtype=np.float32)
            )

            static = (
                seg[STATIC_SOIL_COLUMNS]
                .to_numpy(dtype=np.float32)
            )

            target = (
                seg[TARGET_COL]
                .to_numpy(dtype=np.float32)
            )

            # ------------------------------------------------
            # Sliding window
            # ------------------------------------------------

            for i in range(
                SEQ_LEN,
                len(seg)
            ):

                X_weather.append(
                    weather[
                        i - SEQ_LEN:i
                    ]
                )

                X_spectral.append(
                    spectral[
                        i - SEQ_LEN:i
                    ]
                )

                X_dynamic.append(
                    dynamic[
                        i - SEQ_LEN:i
                    ]
                )

                # Static soil is repeated over the temporal
                # window so that the fusion model receives:
                #
                # [21, 28]
                #
                # rather than only [28].

                X_static.append(
                    np.repeat(
                        static[i][None, :],
                        SEQ_LEN,
                        axis=0
                    )
                )

                y.append(
                    target[i]
                )

    X_weather = np.asarray(
        X_weather,
        dtype=np.float32
    )

    X_spectral = np.asarray(
        X_spectral,
        dtype=np.float32
    )

    X_dynamic = np.asarray(
        X_dynamic,
        dtype=np.float32
    )

    X_static = np.asarray(
        X_static,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    return (
        X_weather,
        X_spectral,
        X_dynamic,
        X_static,
        y,
    )


# ============================================================
# NUMERICAL CHECK
# ============================================================

def check_arrays(
    name,
    X_weather,
    X_spectral,
    X_dynamic,
    X_static,
    y,
):

    print("\n" + "=" * 80)
    print(f"{name.upper()} ARRAY CHECK")
    print("=" * 80)

    arrays = {
        "weather": X_weather,
        "spectral": X_spectral,
        "dynamic_soil": X_dynamic,
        "static_soil": X_static,
        "target": y,
    }

    for key, arr in arrays.items():

        nan = np.isnan(arr).sum()
        inf = np.isinf(arr).sum()

        print(
            f"{key:15s} "
            f"shape={str(arr.shape):20s} "
            f"NaN={nan:<8d} "
            f"Inf={inf:<8d} "
            f"min={np.nanmin(arr):.5f} "
            f"max={np.nanmax(arr):.5f}"
        )

        if nan > 0 or inf > 0:

            raise ValueError(
                f"{name}: non-finite values in {key}"
            )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(
    name,
    y
):

    print(f"\n{name}")

    print(
        f"  min    : {np.min(y):.6f}"
    )

    print(
        f"  p1     : {np.percentile(y, 1):.6f}"
    )

    print(
        f"  p25    : {np.percentile(y, 25):.6f}"
    )

    print(
        f"  median : {np.percentile(y, 50):.6f}"
    )

    print(
        f"  p75    : {np.percentile(y, 75):.6f}"
    )

    print(
        f"  p95    : {np.percentile(y, 95):.6f}"
    )

    print(
        f"  p99    : {np.percentile(y, 99):.6f}"
    )

    print(
        f"  max    : {np.max(y):.6f}"
    )

    print(
        f"  mean   : {np.mean(y):.6f}"
    )

    print(
        f"  std    : {np.std(y):.6f}"
    )

    print(
        f"  zeros  : {np.sum(y == 0)}"
    )


# ============================================================
# SAVE
# ============================================================

def save_array(name, array):

    path = OUTPUT_DIR / f"{name}.npy"

    np.save(path, array)

    print(
        f"Saved {path} "
        f"{array.shape}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    scalers,
    train,
    val,
    test
):

    metadata = {

        "input_file": str(INPUT_FILE),

        "target": TARGET_COL,

        "sequence_days": SEQ_DAYS,

        "steps_per_day": STEPS_PER_DAY,

        "sequence_length": SEQ_LEN,

        "prediction_horizon": "next_timestep",

        "train_years": TRAIN_YEARS,

        "validation_years": VAL_YEARS,

        "test_years": TEST_YEARS,

        "weather_features": WEATHER_COLUMNS,

        "spectral_features": SPECTRAL_COLUMNS,

        "dynamic_soil_features": DYNAMIC_SOIL_COLUMNS,

        "static_soil_features": STATIC_SOIL_COLUMNS,

        "excluded_lv_variables": LEAKAGE_COLUMNS,

        "train_rows": len(train),

        "validation_rows": len(val),

        "test_rows": len(test),

        "target_scaler": "MinMaxScaler",

        "weather_scaler": "RobustScaler",

        "spectral_scaler": "MinMaxScaler",

        "dynamic_soil_scaler": "RobustScaler",

        "static_soil_scaler": "RobustScaler",
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # Save sklearn scalers using pickle
    import pickle

    with open(
        OUTPUT_DIR / "scalers.pkl",
        "wb"
    ) as f:

        pickle.dump(
            scalers,
            f
        )


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    train_arrays,
    val_arrays,
    test_arrays
):

    tw, ts, td, tstatic, ty = train_arrays
    vw, vs, vd, vstatic, vy = val_arrays
    ew, es, ed, estatic, ey = test_arrays

    lines = []

    lines.append(
        "MuSTIPest-V3 PREPROCESSING SUMMARY"
    )

    lines.append("=" * 70)

    lines.append(
        f"Sequence: {SEQ_DAYS} days "
        f"({SEQ_LEN} timesteps)"
    )

    lines.append(
        "Prediction: next timestep"
    )

    lines.append(
        f"Train years: {TRAIN_YEARS}"
    )

    lines.append(
        f"Validation years: {VAL_YEARS}"
    )

    lines.append(
        f"Test years: {TEST_YEARS}"
    )

    lines.append("")

    lines.append(
        f"Weather features: {len(WEATHER_COLUMNS)}"
    )

    lines.append(
        f"Spectral features: {len(SPECTRAL_COLUMNS)}"
    )

    lines.append(
        f"Dynamic soil features: "
        f"{len(DYNAMIC_SOIL_COLUMNS)}"
    )

    lines.append(
        f"Static soil features: "
        f"{len(STATIC_SOIL_COLUMNS)}"
    )

    lines.append("")

    lines.append(
        f"Train samples: {len(ty)}"
    )

    lines.append(
        f"Validation samples: {len(vy)}"
    )

    lines.append(
        f"Test samples: {len(ey)}"
    )

    lines.append("")

    lines.append(
        f"Train weather shape: {tw.shape}"
    )

    lines.append(
        f"Train spectral shape: {ts.shape}"
    )

    lines.append(
        f"Train dynamic soil shape: {td.shape}"
    )

    lines.append(
        f"Train static soil shape: {tstatic.shape}"
    )

    lines.append(
        f"Train target shape: {ty.shape}"
    )

    lines.append("")

    lines.append(
        "LV internal variables excluded from model input:"
    )

    for col in LEAKAGE_COLUMNS:
        lines.append(
            f"  - {col}"
        )

    with open(
        OUTPUT_DIR / "preprocessing_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("MuSTIPest-V3")
    print("MULTIMODAL LV TARGET PREPROCESSING")
    print("=" * 80)

    print(
        f"\nInput : {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"\nSequence: "
        f"{SEQ_DAYS} days × "
        f"{STEPS_PER_DAY} observations/day "
        f"= {SEQ_LEN} timesteps"
    )

    # --------------------------------------------------------
    # output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # validate
    # --------------------------------------------------------

    df = validate_raw_data(df)

    # --------------------------------------------------------
    # temporal features
    # --------------------------------------------------------

    df = engineer_temporal_features(df)

    # --------------------------------------------------------
    # sort
    # --------------------------------------------------------

    df = sort_temporally(df)

    # --------------------------------------------------------
    # split
    # --------------------------------------------------------

    train_df, val_df, test_df = split_by_year(df)

    # --------------------------------------------------------
    # scalers
    # --------------------------------------------------------

    scalers = fit_scalers(train_df)

    # --------------------------------------------------------
    # transform
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SCALING")
    print("=" * 80)

    train_scaled = transform_dataframe(
        train_df,
        scalers
    )

    val_scaled = transform_dataframe(
        val_df,
        scalers
    )

    test_scaled = transform_dataframe(
        test_df,
        scalers
    )

    print("Scaling complete.")

    # --------------------------------------------------------
    # sequences
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING TEMPORAL SEQUENCES")
    print("=" * 80)

    print(
        "\nCreating training sequences..."
    )

    train_arrays = create_sequences(
        train_scaled
    )

    print(
        "Creating validation sequences..."
    )

    val_arrays = create_sequences(
        val_scaled
    )

    print(
        "Creating test sequences..."
    )

    test_arrays = create_sequences(
        test_scaled
    )

    # --------------------------------------------------------
    # unpack
    # --------------------------------------------------------

    (
        X_weather_train,
        X_spectral_train,
        X_dynamic_train,
        X_static_train,
        y_train,
    ) = train_arrays

    (
        X_weather_val,
        X_spectral_val,
        X_dynamic_val,
        X_static_val,
        y_val,
    ) = val_arrays

    (
        X_weather_test,
        X_spectral_test,
        X_dynamic_test,
        X_static_test,
        y_test,
    ) = test_arrays

    # --------------------------------------------------------
    # print shapes
    # --------------------------------------------------------

    print("\nGenerated shapes:")

    print(
        f"Weather train       : "
        f"{X_weather_train.shape}"
    )

    print(
        f"Spectral train      : "
        f"{X_spectral_train.shape}"
    )

    print(
        f"Dynamic soil train  : "
        f"{X_dynamic_train.shape}"
    )

    print(
        f"Static soil train   : "
        f"{X_static_train.shape}"
    )

    print(
        f"Target train        : "
        f"{y_train.shape}"
    )

    print(
        f"\nWeather val         : "
        f"{X_weather_val.shape}"
    )

    print(
        f"Spectral val        : "
        f"{X_spectral_val.shape}"
    )

    print(
        f"Dynamic soil val    : "
        f"{X_dynamic_val.shape}"
    )

    print(
        f"Static soil val     : "
        f"{X_static_val.shape}"
    )

    print(
        f"Target val          : "
        f"{y_val.shape}"
    )

    print(
        f"\nWeather test        : "
        f"{X_weather_test.shape}"
    )

    print(
        f"Spectral test       : "
        f"{X_spectral_test.shape}"
    )

    print(
        f"Dynamic soil test   : "
        f"{X_dynamic_test.shape}"
    )

    print(
        f"Static soil test    : "
        f"{X_static_test.shape}"
    )

    print(
        f"Target test         : "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # numerical checks
    # --------------------------------------------------------

    check_arrays(
        "Train",
        *train_arrays
    )

    check_arrays(
        "Validation",
        *val_arrays
    )

    check_arrays(
        "Test",
        *test_arrays
    )

    # --------------------------------------------------------
    # target distributions
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SCALED TARGET DISTRIBUTION")
    print("=" * 80)

    print_target_distribution(
        "Train",
        y_train
    )

    print_target_distribution(
        "Validation",
        y_val
    )

    print_target_distribution(
        "Test",
        y_test
    )

    # --------------------------------------------------------
    # save arrays
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAVING PREPROCESSED DATA")
    print("=" * 80)

    save_array(
        "X_weather_train",
        X_weather_train
    )

    save_array(
        "X_weather_val",
        X_weather_val
    )

    save_array(
        "X_weather_test",
        X_weather_test
    )

    save_array(
        "X_spectral_train",
        X_spectral_train
    )

    save_array(
        "X_spectral_val",
        X_spectral_val
    )

    save_array(
        "X_spectral_test",
        X_spectral_test
    )

    save_array(
        "X_dynamic_soil_train",
        X_dynamic_train
    )

    save_array(
        "X_dynamic_soil_val",
        X_dynamic_val
    )

    save_array(
        "X_dynamic_soil_test",
        X_dynamic_test
    )

    save_array(
        "X_static_soil_train",
        X_static_train
    )

    save_array(
        "X_static_soil_val",
        X_static_val
    )

    save_array(
        "X_static_soil_test",
        X_static_test
    )

    save_array(
        "y_train",
        y_train
    )

    save_array(
        "y_val",
        y_val
    )

    save_array(
        "y_test",
        y_test
    )

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    save_metadata(
        scalers,
        train_df,
        val_df,
        test_df
    )

    write_summary(
        train_arrays,
        val_arrays,
        test_arrays
    )

    # --------------------------------------------------------
    # final
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"\nOutput directory:"
        f"\n  {OUTPUT_DIR.resolve()}"
    )

    print("\nFiles generated:")

    for f in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            f"  {f.name}"
        )

    print("\nReady for:")
    print("  model.py")
    print("  train.py")
    print("  statistics.py")
    print("  explainability.py")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()