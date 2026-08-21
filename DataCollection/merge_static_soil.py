import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

WEATHER_SPECTRAL_FILE = (
    "weather_spectral_ALL_SITES_2021_2025.csv"
)

STATIC_SOIL_FILE = (
    "soil_data/static_soil_ALL_SITES.csv"
)

OUTPUT_FILE = (
    "multimodal_ALL_SITES_2021_2025.csv"
)


# ============================================================
# EXPECTED STATIC SOIL FEATURES
# ============================================================

SOIL_COLUMNS = [
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
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_numeric_finite(df, name):

    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        print(f"{name}: no numerical columns found.")
        return

    nan_count = int(numeric.isna().sum().sum())

    inf_count = int(
        np.isinf(numeric.to_numpy()).sum()
    )

    print(f"{name} NaN count : {nan_count}")
    print(f"{name} Inf count : {inf_count}")

    if nan_count > 0:

        print("\nColumns containing NaN:")

        print(
            numeric.isna()
            .sum()
            .loc[
                lambda x: x > 0
            ]
        )

    if inf_count > 0:

        print("\nWARNING: Infinite values detected.")


# ============================================================
# LOAD WEATHER + SPECTRAL
# ============================================================

def load_weather_spectral():

    print_header(
        "LOADING WEATHER + SPECTRAL DATA"
    )

    path = Path(WEATHER_SPECTRAL_FILE)

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    print(f"File : {path}")
    print(f"Shape: {df.shape}")

    required_columns = [
        "site_name",
        "year",
        "date",
    ]

    for col in required_columns:

        if col not in df.columns:

            raise ValueError(
                f"Required column '{col}' "
                "not found in weather-spectral file."
            )

    df["site_name"] = (
        df["site_name"]
        .astype(str)
        .str.strip()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():

        raise ValueError(
            "Invalid dates found in "
            "weather-spectral dataset."
        )

    # Check temporal uniqueness.
    duplicate_keys = df.duplicated(
        subset=[
            "site_name",
            "date",
            "collection_window",
        ],
        keep=False
    )

    n_duplicates = int(
        duplicate_keys.sum()
    )

    print(
        f"Temporal duplicate rows: "
        f"{n_duplicates}"
    )

    if n_duplicates > 0:

        print(
            "\nWARNING: Weather-spectral "
            "dataset contains duplicate "
            "temporal observations."
        )

        print(
            df.loc[
                duplicate_keys,
                [
                    "site_name",
                    "date",
                    "collection_window",
                ],
            ].head(20)
        )

    check_numeric_finite(
        df,
        "Weather + spectral"
    )

    print(
        "\nSites:"
    )

    print(
        df["site_name"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nYears:"
    )

    print(
        sorted(
            df["year"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    return df


# ============================================================
# LOAD STATIC SOIL
# ============================================================

def load_static_soil():

    print_header(
        "LOADING STATIC SOIL DATA"
    )

    path = Path(STATIC_SOIL_FILE)

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path.resolve()}"
        )

    soil = pd.read_csv(path)

    print(f"File : {path}")
    print(f"Shape: {soil.shape}")

    if "site_name" not in soil.columns:

        raise ValueError(
            "Static soil file must contain "
            "'site_name'."
        )

    soil["site_name"] = (
        soil["site_name"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Check soil columns
    # --------------------------------------------------------

    missing_columns = [
        c
        for c in SOIL_COLUMNS
        if c not in soil.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing static soil columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing_columns
            )
        )

    # --------------------------------------------------------
    # Only retain site + soil measurements
    #
    # We deliberately do NOT merge:
    # latitude
    # longitude
    # data_source
    #
    # because these can conflict with the
    # weather-spectral master.
    # --------------------------------------------------------

    soil = soil[
        ["site_name"] + SOIL_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Check duplicate sites
    # --------------------------------------------------------

    duplicate_sites = (
        soil["site_name"]
        .duplicated(keep=False)
    )

    if duplicate_sites.any():

        print(
            "\nDuplicate soil records detected:"
        )

        print(
            soil.loc[
                duplicate_sites,
                ["site_name"],
            ]
        )

        raise ValueError(
            "Static soil must contain exactly "
            "one record per site."
        )

    # --------------------------------------------------------
    # Convert soil columns to numeric
    # --------------------------------------------------------

    for col in SOIL_COLUMNS:

        soil[col] = pd.to_numeric(
            soil[col],
            errors="coerce"
        )

    check_numeric_finite(
        soil,
        "Static soil"
    )

    # --------------------------------------------------------
    # Check soil completeness
    # --------------------------------------------------------

    missing_soil = (
        soil[SOIL_COLUMNS]
        .isna()
        .sum()
    )

    if (missing_soil > 0).any():

        print(
            "\nMissing soil values:"
        )

        print(
            missing_soil[
                missing_soil > 0
            ]
        )

        raise ValueError(
            "Static soil contains missing "
            "measurement values."
        )

    print(
        f"\nStatic soil sites: "
        f"{soil['site_name'].nunique()}"
    )

    print(
        "Static soil properties: "
        f"{len(SOIL_COLUMNS)}"
    )

    print(
        "\nSites:"
    )

    for site in soil["site_name"]:

        print(
            f"  {site}"
        )

    return soil


# ============================================================
# MERGE
# ============================================================

def merge_data(
    weather_spectral,
    soil,
):

    print_header(
        "MERGING STATIC SOIL"
    )

    # --------------------------------------------------------
    # Site consistency
    # --------------------------------------------------------

    weather_sites = set(
        weather_spectral[
            "site_name"
        ].unique()
    )

    soil_sites = set(
        soil[
            "site_name"
        ].unique()
    )

    print(
        "Weather-spectral sites:"
    )

    for site in sorted(weather_sites):

        print(f"  {site}")

    print(
        "\nStatic soil sites:"
    )

    for site in sorted(soil_sites):

        print(f"  {site}")

    missing_soil_sites = (
        weather_sites - soil_sites
    )

    unused_soil_sites = (
        soil_sites - weather_sites
    )

    if missing_soil_sites:

        raise ValueError(
            "\nStatic soil is missing "
            "for these weather sites:\n"
            + "\n".join(
                f"  - {s}"
                for s in sorted(
                    missing_soil_sites
                )
            )
        )

    if unused_soil_sites:

        print(
            "\nWARNING: Soil contains "
            "sites not present in "
            "weather-spectral data:"
        )

        for site in sorted(
            unused_soil_sites
        ):

            print(
                f"  - {site}"
            )

    # --------------------------------------------------------
    # Merge
    #
    # many weather observations
    #       ↓
    # one static soil record/site
    #
    # Therefore:
    #
    # many_to_one
    # --------------------------------------------------------

    merged = weather_spectral.merge(
        soil,
        on="site_name",
        how="left",
        validate="many_to_one",
    )

    print(
        "\nMerge completed."
    )

    print(
        f"Input rows : "
        f"{len(weather_spectral):,}"
    )

    print(
        f"Output rows: "
        f"{len(merged):,}"
    )

    # --------------------------------------------------------
    # Critical row-count check
    # --------------------------------------------------------

    if len(merged) != len(
        weather_spectral
    ):

        raise ValueError(
            "\nROW COUNT CHANGED DURING "
            "STATIC SOIL MERGE.\n"
            "This indicates an incorrect "
            "soil key or duplicate soil "
            "records."
        )

    return merged


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_dataset(
    merged,
    original,
):

    print_header(
        "FINAL DATASET VALIDATION"
    )

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    print(
        f"Original shape : "
        f"{original.shape}"
    )

    print(
        f"Final shape    : "
        f"{merged.shape}"
    )

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    assert len(merged) == len(
        original
    )

    print(
        "Row count       : PASS"
    )

    # --------------------------------------------------------
    # Site counts
    # --------------------------------------------------------

    print(
        "\nRows per site:"
    )

    print(
        merged[
            "site_name"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Year counts
    # --------------------------------------------------------

    print(
        "\nRows per year:"
    )

    print(
        merged[
            "year"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Check soil completeness
    # --------------------------------------------------------

    soil_nan = (
        merged[
            SOIL_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"\nStatic soil NaN: "
        f"{soil_nan}"
    )

    if soil_nan != 0:

        raise ValueError(
            "Static soil contains NaN "
            "after merging."
        )

    # --------------------------------------------------------
    # Check Inf
    # --------------------------------------------------------

    numeric = (
        merged
        .select_dtypes(
            include=[np.number]
        )
    )

    nan_count = int(
        numeric.isna()
        .sum()
        .sum()
    )

    inf_count = int(
        np.isinf(
            numeric.to_numpy()
        ).sum()
    )

    print(
        f"Numeric NaN     : "
        f"{nan_count}"
    )

    print(
        f"Numeric Inf     : "
        f"{inf_count}"
    )

    if nan_count != 0:

        print(
            "\nColumns with NaN:"
        )

        print(
            numeric.isna()
            .sum()
            .loc[
                lambda x: x > 0
            ]
        )

        raise ValueError(
            "Final dataset contains NaN."
        )

    if inf_count != 0:

        raise ValueError(
            "Final dataset contains Inf."
        )

    # --------------------------------------------------------
    # Check required modalities
    # --------------------------------------------------------

    expected_modalities = [

        # Vegetation
        "NDVI",
        "NDWI",
        "EVI",

        # Dynamic soil
        "BSI",
        "SAVI",
        "NDTI",
        "RI",
    ]

    missing = [
        c
        for c in expected_modalities
        if c not in merged.columns
    ]

    if missing:

        raise ValueError(
            "Missing multimodal columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    print(
        "Vegetation features : PASS"
    )

    print(
        "Dynamic soil        : PASS"
    )

    print(
        "Static soil         : PASS"
    )

    print(
        "\nFINAL VALIDATION: PASS"
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    df,
):

    print_header(
        "SAVING FINAL DATASET"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved:\n"
        f"  {Path(OUTPUT_FILE).resolve()}"
    )

    print(
        f"\nFinal shape: "
        f"{df.shape}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "WEATHER + SPECTRAL + STATIC SOIL"
    )
    print(
        "MULTIMODAL DATASET GENERATION"
    )
    print("=" * 80)

    print(
        "\nInput 1:"
    )

    print(
        f"  {WEATHER_SPECTRAL_FILE}"
    )

    print(
        "\nInput 2:"
    )

    print(
        f"  {STATIC_SOIL_FILE}"
    )

    print(
        "\nOutput:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # 1. Load weather + spectral
    # --------------------------------------------------------

    weather_spectral = (
        load_weather_spectral()
    )

    # --------------------------------------------------------
    # 2. Load static soil
    # --------------------------------------------------------

    soil = load_static_soil()

    # --------------------------------------------------------
    # 3. Merge
    # --------------------------------------------------------

    merged = merge_data(
        weather_spectral,
        soil,
    )

    # --------------------------------------------------------
    # 4. Validate
    # --------------------------------------------------------

    validate_final_dataset(
        merged,
        weather_spectral,
    )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    save_dataset(
        merged
    )

    print()
    print("=" * 80)
    print(
        "MULTIMODAL DATASET READY"
    )
    print("=" * 80)

    print(
        "\nThe final dataset contains:"
    )

    print(
        "  Weather"
    )

    print(
        "  Vegetation: NDVI, NDWI, EVI"
    )

    print(
        "  Dynamic soil: BSI, SAVI, NDTI, RI"
    )

    print(
        "  Static soil: 28 properties"
    )

    print(
        "\nNext step:"
    )

    print(
        "  Generate the LV-based synthetic "
        "aphid target from this dataset."
    )


if __name__ == "__main__":
    main()