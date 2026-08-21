"""
merge_weather_all_sites.py

Merge the 15 county-year weather files (Jasper/Polk/Story, 2021-2025)
into a clean master file:

    weather_ALL_SITES_2021_2025.csv

The script:
  * verifies all 15 source files exist;
  * uses county/year from the filename as authoritative metadata;
  * normalizes date/time;
  * removes exact duplicates;
  * detects duplicate temporal keys;
  * aggregates genuinely duplicated temporal records by mean for numeric
    measurements and first value for metadata;
  * checks coverage for every county/year;
  * checks NaN/Inf values;
  * saves the clean master dataset.

Run from the directory containing the 15 input CSV files:
    python merge_weather_all_sites.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys


INPUT_DIR = Path("weather_data")
OUTPUT_FILE = INPUT_DIR / "weather_ALL_SITES_2021_2025.csv"

COUNTIES = ["Jasper_County", "Polk_County", "Story_County"]
YEARS = list(range(2021, 2026))


def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def expected_file(county, year):
    return INPUT_DIR / f"weather_{county}_{year}.csv"


def check_files():
    header("CHECKING INPUT WEATHER FILES")

    paths = []

    for county in COUNTIES:
        for year in YEARS:
            path = expected_file(county, year)

            if path.exists():
                print(f"[OK]      {path.name}")
                paths.append(path)
            else:
                print(f"[MISSING] {path.name}")

    if len(paths) != 15:
        raise FileNotFoundError(
            f"Expected 15 county-year files, but found {len(paths)}."
        )

    print(f"\nFound all {len(paths)} required county-year files.")
    return paths


def parse_file_identity(path):
    m = re.fullmatch(
        r"weather_(Jasper_County|Polk_County|Story_County)_(20\d{2})",
        path.stem
    )

    if m is None:
        raise ValueError(
            f"Unexpected weather filename: {path.name}"
        )

    county = m.group(1)
    year = int(m.group(2))

    if year not in YEARS:
        raise ValueError(
            f"Unsupported year {year} in {path.name}"
        )

    return county, year


def load_one(path):
    county, year = parse_file_identity(path)

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"\nLoading {path.name}: {df.shape}")

    # Filename is authoritative for these fields.
    df["site_name"] = county
    df["year"] = year

    # Resolve date.
    if "date" in df.columns:
        date_source = df["date"]
    elif "time_local" in df.columns:
        date_source = df["time_local"]
    else:
        raise ValueError(
            f"{path.name}: neither 'date' nor 'time_local' exists."
        )

    df["date"] = pd.to_datetime(
        date_source, errors="coerce"
    ).dt.normalize()

    bad_dates = df["date"].isna().sum()

    if bad_dates:
        print(f"[WARNING] Removing {bad_dates} rows with invalid dates.")
        df = df.dropna(subset=["date"]).copy()

    # A valid source file must not contain another year.
    wrong_year = df["date"].dt.year != year

    if wrong_year.any():
        n = int(wrong_year.sum())
        print(
            f"[WARNING] Removing {n} rows whose date does not belong "
            f"to {year}."
        )
        df = df.loc[~wrong_year].copy()

    # Normalize collection-window labels if available.
    if "collection_window" in df.columns:
        df["collection_window"] = (
            df["collection_window"].astype(str).str.strip()
        )

    # Convert common weather measurements to numeric when present.
    likely_numeric = [
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
        "u_wind",
        "v_wind",
    ]

    for col in likely_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_sources(paths):
    header("LOADING AND MERGING WEATHER DATA")

    frames = [load_one(path) for path in paths]

    weather = pd.concat(frames, ignore_index=True)

    before = len(weather)
    weather = weather.drop_duplicates().reset_index(drop=True)

    print(f"\nRows after concatenation : {before:,}")
    print(f"Exact duplicate rows removed: {before - len(weather):,}")
    print(f"Rows after exact deduplication: {len(weather):,}")

    return weather


def temporal_key(weather):
    key = ["site_name", "year", "date"]

    if "collection_window" in weather.columns:
        key.append("collection_window")

    return key


def resolve_temporal_duplicates(weather):
    header("CHECKING TEMPORAL DUPLICATES")

    key = temporal_key(weather)

    duplicate_mask = weather.duplicated(key, keep=False)
    n_duplicate = int(duplicate_mask.sum())

    if n_duplicate == 0:
        print("[OK] No duplicate temporal keys.")
        return weather

    print(
        f"[WARNING] {n_duplicate:,} rows share a temporal key."
    )
    print("\nExamples:")
    print(
        weather.loc[duplicate_mask, key]
        .sort_values(key)
        .head(20)
        .to_string(index=False)
    )

    # These are multiple measurements for the same temporal key.
    # Aggregate numeric measurements by mean and metadata by first.
    numeric_cols = weather.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    agg = {}

    for col in weather.columns:
        if col in key:
            continue

        if col in numeric_cols and col != "year":
            agg[col] = "mean"
        else:
            agg[col] = "first"

    weather = (
        weather.groupby(key, as_index=False)
        .agg(agg)
    )

    print(
        f"\nRows after temporal aggregation: {len(weather):,}"
    )

    return weather


def quality_check(weather):
    header("NUMERICAL QUALITY CHECK")

    numeric = weather.select_dtypes(
        include=[np.number]
    )

    inf_count = np.isinf(numeric.to_numpy()).sum()
    nan_count = numeric.isna().sum().sum()

    print(f"Total numeric NaN values : {nan_count:,}")
    print(f"Total numeric Inf values : {inf_count:,}")

    if inf_count:
        print("[INFO] Replacing Inf/-Inf with NaN.")
        weather[numeric.columns] = (
            weather[numeric.columns]
            .replace([np.inf, -np.inf], np.nan)
        )

    missing = weather.isna().sum()
    missing = missing[missing > 0]

    if len(missing):
        print("\nColumns containing missing values:")
        print(missing.to_string())
    else:
        print("[OK] No missing values.")

    return weather


def coverage_check(weather):
    header("WEATHER COVERAGE")

    coverage = (
        weather.groupby(["site_name", "year"])
        .agg(
            rows=("date", "size"),
            unique_dates=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
        .sort_values(["site_name", "year"])
    )

    print(coverage.to_string(index=False))

    expected = {
        (county, year)
        for county in COUNTIES
        for year in YEARS
    }

    actual = set(
        zip(coverage["site_name"], coverage["year"])
    )

    missing = sorted(expected - actual)

    if missing:
        raise ValueError(
            "Missing county/year combinations:\n"
            + "\n".join(
                f"  {county} {year}"
                for county, year in missing
            )
        )

    print(
        "\n[OK] All 15 county-year combinations are present."
    )

    return coverage


def window_check(weather):
    header("TEMPORAL WINDOW CHECK")

    if "collection_window" not in weather.columns:
        print(
            "[INFO] collection_window is not present. "
            "Skipping window-count validation."
        )
        return

    counts = (
        weather.groupby(["site_name", "year", "date"])
        ["collection_window"]
        .nunique()
    )

    print(
        f"Unique site-date combinations: {len(counts):,}"
    )

    print(
        "Window-count distribution:"
    )
    print(
        counts.value_counts()
        .sort_index()
        .to_string()
    )

    incomplete = counts[counts < 3]

    if len(incomplete):
        print(
            f"\n[WARNING] {len(incomplete):,} site-date "
            "combinations have fewer than 3 windows."
        )
    else:
        print(
            "\n[OK] Every site-date has 3 collection windows."
        )


def save(weather):
    header("SAVING CLEAN WEATHER MASTER DATASET")

    key = temporal_key(weather)

    weather = (
        weather
        .sort_values(key)
        .reset_index(drop=True)
    )

    weather.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Output : {OUTPUT_FILE.resolve()}")
    print(f"Rows   : {len(weather):,}")
    print(f"Columns: {len(weather.columns)}")

    return weather


def main():

    header("WEATHER DATA MERGING AND SYNCHRONIZATION")

    print("Period : 2021-2025")
    print("Sites  : Jasper, Polk, Story")
    print("Files  : 15 county-year weather CSV files")

    paths = check_files()

    weather = merge_sources(paths)

    weather = resolve_temporal_duplicates(weather)

    weather = quality_check(weather)

    coverage_check(weather)

    window_check(weather)

    weather = save(weather)

    header("FINAL WEATHER DATASET")

    print(
        weather[
            ["site_name", "year", "date"]
        ].head(10).to_string(index=False)
    )

    print("\nYears:")
    print(sorted(weather["year"].unique()))

    print("\nSites:")
    print(sorted(weather["site_name"].unique()))

    print("\nSUCCESS")
    print(
        "Use weather_ALL_SITES_2021_2025.csv as the "
        "weather master file for multimodal synchronization."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(str(exc))
        sys.exit(1)
