"""
===============================================================================
MERGE AND VALIDATE SENTINEL-2 SPECTRAL DATA
===============================================================================

Input:
    spectral_<County>_<Year>.csv

Expected files:
    Jasper 2021-2025
    Polk   2021-2025
    Story  2021-2025

Output:
    spectral_ALL_SITES_2021_2025.csv

Required spectral variables:
    Vegetation:
        NDVI
        NDWI
        EVI

    Dynamic soil:
        BSI
        SAVI
        NDTI
        RI

Important:
    This script DOES NOT generate synthetic/fallback spectral values.
    It only merges and validates the supplied Sentinel-2 files.

Temporal key:
    site_name + year + date + collection_window

===============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = Path("spectral_data")

OUTPUT_FILE = INPUT_DIR / "spectral_ALL_SITES_2021_2025.csv"

COUNTIES = [
    "Jasper_County",
    "Polk_County",
    "Story_County",
]

YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

VEGETATION_FEATURES = [
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

SPECTRAL_FEATURES = (
    VEGETATION_FEATURES +
    DYNAMIC_SOIL_FEATURES
)

KEY_COLUMNS = [
    "site_name",
    "year",
    "date",
    "collection_window",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_column_name(col):
    """
    Normalize column names while preserving useful names.
    """

    col = str(col).strip()

    replacements = {
        "Site": "site_name",
        "site": "site_name",
        "SITE": "site_name",

        "Year": "year",
        "YEAR": "year",

        "Date": "date",
        "DATE": "date",

        "Time": "time_local",
        "time": "time_local",

        "Collection_Window": "collection_window",
        "collectionWindow": "collection_window",
        "window": "collection_window",

        "Ndvi": "NDVI",
        "ndvi": "NDVI",

        "Ndwi": "NDWI",
        "ndwi": "NDWI",

        "Evi": "EVI",
        "evi": "EVI",

        "Bsi": "BSI",
        "bsi": "BSI",

        "Savi": "SAVI",
        "savi": "SAVI",

        "Ndti": "NDTI",
        "ndti": "NDTI",

        "Ri": "RI",
        "ri": "RI",
    }

    return replacements.get(col, col)


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        normalize_column_name(c)
        for c in df.columns
    ]
    return df


# =============================================================================
# FILE VALIDATION
# =============================================================================

def expected_filename(county, year):
    return INPUT_DIR / f"spectral_{county}_{year}.csv"


def check_input_files():
    print_header("CHECKING INPUT FILES")

    missing = []
    found = []

    for county in COUNTIES:
        for year in YEARS:

            path = expected_filename(county, year)

            if path.exists():
                print(f"[OK]     {path.name}")
                found.append(path)
            else:
                print(f"[MISSING] {path.name}")
                missing.append(path)

    print()
    print(f"Expected files : {len(COUNTIES) * len(YEARS)}")
    print(f"Found files    : {len(found)}")
    print(f"Missing files  : {len(missing)}")

    if missing:
        print()
        print("ERROR: Some required files are missing.")
        print()
        print("Missing:")
        for path in missing:
            print(f"    {path.name}")

        raise FileNotFoundError(
            "Cannot create a complete 2021-2025 spectral dataset "
            "because one or more county-year files are missing."
        )

    return found


# =============================================================================
# LOAD ONE FILE
# =============================================================================

def load_one_file(path, county, year):

    print()
    print("-" * 80)
    print(f"Loading: {path.name}")

    df = pd.read_csv(path)

    print(f"Raw shape: {df.shape}")

    df = normalize_columns(df)

    # -------------------------------------------------------------------------
    # SITE NAME
    # -------------------------------------------------------------------------

    if "site_name" not in df.columns:
        df["site_name"] = county

    else:
        # If file contains blank site names, fill from filename.
        df["site_name"] = (
            df["site_name"]
            .astype(str)
            .str.strip()
        )

        df.loc[
            df["site_name"].isin(["", "nan", "None"]),
            "site_name"
        ] = county

    # Force expected county name
    # This prevents accidental names such as "Jasper".
    df["site_name"] = county

    # -------------------------------------------------------------------------
    # YEAR
    # -------------------------------------------------------------------------

    if "year" not in df.columns:
        df["year"] = year
    else:
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce"
        )

        # Fill missing years from filename
        df["year"] = df["year"].fillna(year)

    df["year"] = df["year"].astype(int)

    # Make sure the file actually corresponds to the requested year.
    wrong_years = sorted(
        df.loc[df["year"] != year, "year"].unique()
    )

    if wrong_years:
        raise ValueError(
            f"{path.name} contains unexpected years: "
            f"{wrong_years}. Expected only {year}."
        )

    # -------------------------------------------------------------------------
    # DATE
    # -------------------------------------------------------------------------

    if "date" not in df.columns:

        possible_date_columns = [
            "acquisition_date",
            "datetime",
            "timestamp",
            "time_local",
        ]

        found_date = None

        for candidate in possible_date_columns:
            if candidate in df.columns:
                found_date = candidate
                break

        if found_date is None:
            raise ValueError(
                f"{path.name}: no date column found. "
                f"Available columns:\n{list(df.columns)}"
            )

        df["date"] = df[found_date]

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    ).dt.normalize()

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        print(
            f"[WARNING] {invalid_dates} rows have invalid dates "
            f"and will be removed."
        )

        df = df.dropna(subset=["date"])

    # Verify dates correspond to year
    wrong_date_year = df[
        df["date"].dt.year != year
    ]

    if len(wrong_date_year) > 0:

        print(
            f"[WARNING] {len(wrong_date_year)} rows have "
            f"date/year mismatch."
        )

        # Keep only valid rows
        df = df[
            df["date"].dt.year == year
        ].copy()

    # -------------------------------------------------------------------------
    # COLLECTION WINDOW
    # -------------------------------------------------------------------------

    if "collection_window" not in df.columns:

        print(
            "[INFO] collection_window not found. "
            "Creating one daily record per date."
        )

        df["collection_window"] = 0

    # Normalize window representation
    df["collection_window"] = (
        df["collection_window"]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # REQUIRED SPECTRAL FEATURES
    # -------------------------------------------------------------------------

    missing_features = [
        c for c in SPECTRAL_FEATURES
        if c not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"\n{path.name} is missing required spectral features:\n"
            f"{missing_features}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # NUMERIC CONVERSION
    # -------------------------------------------------------------------------

    for feature in SPECTRAL_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # -------------------------------------------------------------------------
    # REMOVE INF
    # -------------------------------------------------------------------------

    df[SPECTRAL_FEATURES] = (
        df[SPECTRAL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    # -------------------------------------------------------------------------
    # BASIC REPORT
    # -------------------------------------------------------------------------

    print(f"Final shape: {df.shape}")

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        f"Unique dates: {df['date'].nunique()}"
    )

    print(
        f"Rows with NaN spectral values: "
        f"{df[SPECTRAL_FEATURES].isna().any(axis=1).sum()}"
    )

    return df


# =============================================================================
# MERGE ALL FILES
# =============================================================================

def merge_all_files(files):

    print_header("LOADING AND MERGING ALL SPECTRAL FILES")

    all_data = []

    for path in files:

        name = path.stem

        # -------------------------------------------------------------
        # Identify county/year from expected naming convention
        # -------------------------------------------------------------

        matched_county = None
        matched_year = None

        for county in COUNTIES:
            if f"spectral_{county}_" in name:
                matched_county = county
                break

        if matched_county is None:
            raise ValueError(
                f"Cannot determine county from filename: {path.name}"
            )

        for year in YEARS:
            if name.endswith(str(year)):
                matched_year = year
                break

        if matched_year is None:
            raise ValueError(
                f"Cannot determine year from filename: {path.name}"
            )

        df = load_one_file(
            path,
            matched_county,
            matched_year
        )

        all_data.append(df)

    merged = pd.concat(
        all_data,
        ignore_index=True
    )

    print()
    print(f"Merged shape: {merged.shape}")

    return merged


# =============================================================================
# REMOVE EXACT DUPLICATES
# =============================================================================

def remove_exact_duplicates(df):

    print_header("REMOVING EXACT DUPLICATES")

    before = len(df)

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    removed = before - len(df)

    print(f"Rows before : {before:,}")
    print(f"Rows after  : {len(df):,}")
    print(f"Removed     : {removed:,}")

    return df


# =============================================================================
# CHECK TEMPORAL DUPLICATES
# =============================================================================

def check_temporal_duplicates(df):

    print_header("CHECKING TEMPORAL DUPLICATES")

    duplicated = df.duplicated(
        subset=KEY_COLUMNS,
        keep=False
    )

    duplicate_rows = df.loc[
        duplicated,
        KEY_COLUMNS
    ].sort_values(KEY_COLUMNS)

    if len(duplicate_rows) == 0:

        print(
            "[OK] No duplicate temporal keys."
        )

        return df

    print(
        f"[WARNING] {len(duplicate_rows):,} rows "
        f"have duplicate temporal keys."
    )

    print()
    print("Example duplicates:")
    print(
        duplicate_rows
        .head(20)
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We do NOT silently drop these.
    #
    # If there are multiple rows for the same temporal key, they must represent
    # duplicate observations. We aggregate them using mean for spectral values.
    # -------------------------------------------------------------------------

    print()
    print(
        "Aggregating duplicate temporal observations "
        "using the mean of spectral indices..."
    )

    numeric_features = SPECTRAL_FEATURES

    grouped = (
        df.groupby(
            KEY_COLUMNS,
            as_index=False
        )[numeric_features]
        .mean()
    )

    # Restore any useful metadata if present
    for col in ["latitude", "longitude"]:

        if col in df.columns:

            metadata = (
                df.groupby(KEY_COLUMNS)[col]
                .first()
                .reset_index()
            )

            grouped = grouped.merge(
                metadata,
                on=KEY_COLUMNS,
                how="left"
            )

    print(
        f"Rows after temporal aggregation: "
        f"{len(grouped):,}"
    )

    return grouped


# =============================================================================
# CHECK COVERAGE
# =============================================================================

def coverage_report(df):

    print_header("SPECTRAL COVERAGE")

    coverage = (
        df.groupby(
            ["site_name", "year"]
        )
        .agg(
            rows=("date", "size"),
            unique_dates=("date", "nunique"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
        .reset_index()
    )

    print(
        coverage.to_string(index=False)
    )

    print()

    expected_combinations = {
        (county, year)
        for county in COUNTIES
        for year in YEARS
    }

    actual_combinations = {
        (row.site_name, int(row.year))
        for row in coverage.itertuples()
    }

    missing = expected_combinations - actual_combinations

    if missing:

        print("[ERROR] Missing county/year combinations:")

        for county, year in sorted(missing):
            print(
                f"    {county} - {year}"
            )

        raise ValueError(
            "Spectral coverage is incomplete."
        )

    print(
        "[OK] All 15 county-year combinations are present."
    )

    return coverage


# =============================================================================
# CHECK NAN / INF
# =============================================================================

def numerical_quality_report(df):

    print_header("NUMERICAL QUALITY CHECK")

    failed = False

    for feature in SPECTRAL_FEATURES:

        values = df[feature].to_numpy(
            dtype=float
        )

        nan_count = np.isnan(values).sum()
        inf_count = np.isinf(values).sum()

        finite = values[
            np.isfinite(values)
        ]

        if len(finite) > 0:

            min_value = finite.min()
            max_value = finite.max()
            mean_value = finite.mean()
            std_value = finite.std()

        else:

            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan
            std_value = np.nan

        print(
            f"{feature:5s} | "
            f"NaN={nan_count:6d} | "
            f"Inf={inf_count:6d} | "
            f"min={min_value:10.5f} | "
            f"max={max_value:10.5f} | "
            f"mean={mean_value:10.5f} | "
            f"std={std_value:10.5f}"
        )

        if nan_count > 0 or inf_count > 0:
            failed = True

    if failed:

        print()
        print(
            "[WARNING] Missing/non-finite spectral values detected."
        )

    else:

        print()
        print(
            "[OK] No NaN or Inf values in spectral features."
        )


# =============================================================================
# RANGE CHECK
# =============================================================================

def range_check(df):

    print_header("SPECTRAL RANGE CHECK")

    expected_ranges = {

        "NDVI": (-1.0, 1.0),
        "NDWI": (-1.0, 1.0),

        # EVI can theoretically exceed 1 in extreme cases,
        # so a wider physical screening range is used.
        "EVI": (-2.0, 2.0),

        "BSI": (-2.0, 2.0),
        "SAVI": (-2.0, 2.0),
        "NDTI": (-1.0, 1.0),
        "RI": (-5.0, 5.0),
    }

    for feature, (lower, upper) in expected_ranges.items():

        values = df[feature].dropna()

        outside = (
            (values < lower) |
            (values > upper)
        )

        count = outside.sum()

        if count == 0:

            print(
                f"[OK] {feature:5s}: "
                f"all values within [{lower}, {upper}]"
            )

        else:

            print(
                f"[WARNING] {feature:5s}: "
                f"{count:,} values outside "
                f"[{lower}, {upper}]"
            )


# =============================================================================
# CHECK THREE WINDOWS
# =============================================================================

def collection_window_report(df):

    print_header("COLLECTION WINDOW CHECK")

    counts = (
        df.groupby(
            ["site_name", "year", "date"]
        )["collection_window"]
        .nunique()
    )

    print(
        f"Total unique site-date combinations: "
        f"{len(counts):,}"
    )

    print(
        f"Expected windows per day: 3"
    )

    incomplete = counts[counts < 3]

    if len(incomplete) > 0:

        print()
        print(
            f"[WARNING] {len(incomplete):,} "
            f"site-date combinations have fewer than 3 windows."
        )

        print(
            incomplete
            .head(20)
            .to_string()
        )

    else:

        print(
            "[OK] Every site-date contains 3 collection windows."
        )


# =============================================================================
# SORT
# =============================================================================

def sort_dataset(df):

    df = df.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    return df


# =============================================================================
# SAVE
# =============================================================================

def save_dataset(df):

    print_header("SAVING FINAL SPECTRAL DATASET")

    # Put important columns first
    preferred = [
        "site_name",
        "year",
        "date",
        "collection_window",
    ]

    existing_preferred = [
        c for c in preferred
        if c in df.columns
    ]

    remaining = [
        c for c in df.columns
        if c not in existing_preferred
    ]

    # Put spectral variables next
    spectral_existing = [
        c for c in SPECTRAL_FEATURES
        if c in remaining
    ]

    remaining = [
        c for c in remaining
        if c not in spectral_existing
    ]

    final_columns = (
        existing_preferred +
        spectral_existing +
        remaining
    )

    df = df[final_columns]

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved:\n    {OUTPUT_FILE.resolve()}"
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "SPECTRAL DATA MERGING AND VALIDATION"
    )

    print(
        "Target period : 2021–2025"
    )

    print(
        "Sites         : Jasper, Polk, Story"
    )

    print(
        "Vegetation    : NDVI, NDWI, EVI"
    )

    print(
        "Dynamic soil  : BSI, SAVI, NDTI, RI"
    )

    # -------------------------------------------------------------------------
    # 1. Check files
    # -------------------------------------------------------------------------

    files = check_input_files()

    # -------------------------------------------------------------------------
    # 2. Load and merge
    # -------------------------------------------------------------------------

    merged = merge_all_files(files)

    # -------------------------------------------------------------------------
    # 3. Remove exact duplicates
    # -------------------------------------------------------------------------

    merged = remove_exact_duplicates(
        merged
    )

    # -------------------------------------------------------------------------
    # 4. Handle temporal duplicates
    # -------------------------------------------------------------------------

    merged = check_temporal_duplicates(
        merged
    )

    # -------------------------------------------------------------------------
    # 5. Coverage
    # -------------------------------------------------------------------------

    coverage_report(
        merged
    )

    # -------------------------------------------------------------------------
    # 6. Numerical quality
    # -------------------------------------------------------------------------

    numerical_quality_report(
        merged
    )

    # -------------------------------------------------------------------------
    # 7. Physical range check
    # -------------------------------------------------------------------------

    range_check(
        merged
    )

    # -------------------------------------------------------------------------
    # 8. Collection window check
    # -------------------------------------------------------------------------

    collection_window_report(
        merged
    )

    # -------------------------------------------------------------------------
    # 9. Sort
    # -------------------------------------------------------------------------

    merged = sort_dataset(
        merged
    )

    # -------------------------------------------------------------------------
    # 10. Save
    # -------------------------------------------------------------------------

    merged = save_dataset(
        merged
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_header(
        "FINAL DATASET SUMMARY"
    )

    print(
        f"Rows              : {len(merged):,}"
    )

    print(
        f"Columns           : {len(merged.columns)}"
    )

    print(
        f"Sites             : "
        f"{merged['site_name'].nunique()}"
    )

    print(
        f"Years             : "
        f"{sorted(merged['year'].unique().tolist())}"
    )

    print(
        f"Unique dates      : "
        f"{merged['date'].nunique():,}"
    )

    print(
        f"Spectral features : "
        f"{len(SPECTRAL_FEATURES)}"
    )

    print()
    print(
        "Spectral features:"
    )

    for feature in SPECTRAL_FEATURES:
        print(f"    {feature}")

    print()
    print("=" * 80)
    print("MERGING COMPLETE")
    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print()
        print("=" * 80)
        print("ERROR")
        print("=" * 80)

        print(
            str(e)
        )

        print()

        sys.exit(1)