"""
synchronize_weather_spectral.py

Synchronize the Weather and Sentinel-2-derived spectral/dynamic-soil data
onto the Weather 8-hour master grid.

Inputs
------
weather_data/weather_ALL_SITES_2021_2025.csv
spectral_data/spectral_ALL_SITES_2021_2025.csv

Output
------
weather_spectral_ALL_SITES_2021_2025.csv

Design
------
* Weather is the master temporal grid: 3 sites x 5 years x 549 rows/year.
* Spectral data are lower-frequency observations.
* Spectral values are first reduced to one observation per site/year/date.
  The three collection windows in the spectral file contain the same
  satellite-derived values, so they are not treated as independent
  observations.
* For each site and year, spectral variables are linearly interpolated
  over calendar days between the first and last spectral acquisition.
* Values before the first acquisition and after the last acquisition are
  carried to the nearest boundary observation. This avoids NaNs while
  preventing information from one year from entering another year.
* The resulting daily spectral values are then assigned to all three
  weather windows (00h, 08h, 16h).
* No latitude/longitude merge is used; Weather remains the source of the
  geographic coordinates.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

WEATHER_FILE = (
    BASE_DIR / "weather_data" / "weather_ALL_SITES_2021_2025.csv"
)
SPECTRAL_FILE = (
    BASE_DIR / "spectral_data" / "spectral_ALL_SITES_2021_2025.csv"
)

OUTPUT_FILE = BASE_DIR / "weather_spectral_ALL_SITES_2021_2025.csv"


# ---------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------

KEY_COLUMNS = [
    "site_name",
    "year",
    "date",
    "collection_window",
]

SPECTRAL_COLUMNS = [
    "NDVI",
    "NDWI",
    "EVI",
    "BSI",
    "SAVI",
    "NDTI",
    "RI",
]

EXPECTED_WINDOWS = {
    "Window_1_00h",
    "Window_2_08h",
    "Window_3_16h",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_window(x):
    """
    Convert variants such as:
        Window_1_00h
        Window_1_00h00
    into:
        Window_1_00h
    """
    if pd.isna(x):
        return x

    s = str(x).strip()

    replacements = {
        "Window_1_00h00": "Window_1_00h",
        "Window_2_08h00": "Window_2_08h",
        "Window_3_16h00": "Window_3_16h",
    }

    return replacements.get(s, s)


def numeric_inf_count(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return 0
    return int(np.isinf(numeric.to_numpy(dtype=float)).sum())


def report_dataframe(df, name):
    print(f"{name} shape: {df.shape}")
    print(f"{name} NaN count: {int(df.isna().sum().sum())}")
    print(f"{name} Inf count: {numeric_inf_count(df)}")


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_weather():
    section("LOADING WEATHER DATA")

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Weather file not found:\n{WEATHER_FILE}")

    df = pd.read_csv(WEATHER_FILE)

    required = [
        "site_name",
        "year",
        "date",
        "collection_window",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Weather is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["collection_window"] = df["collection_window"].map(normalize_window)

    if df["date"].isna().any():
        raise ValueError("Weather contains invalid dates.")

    if df["year"].isna().any():
        raise ValueError("Weather contains invalid year values.")

    df["year"] = df["year"].astype(int)

    unexpected = set(df["collection_window"].dropna().unique()) - EXPECTED_WINDOWS
    if unexpected:
        warnings.warn(
            f"Unexpected weather collection windows: {sorted(unexpected)}"
        )

    # Check duplicate master keys.
    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    ndup = int(duplicates.sum())

    if ndup:
        dup_examples = (
            df.loc[duplicates, KEY_COLUMNS]
            .sort_values(KEY_COLUMNS)
            .head(20)
        )
        print("Duplicate Weather keys found:")
        print(dup_examples.to_string(index=False))
        raise ValueError(
            "Weather contains duplicate site/year/date/collection_window "
            "keys. Fix the weather master grid before synchronization."
        )

    report_dataframe(df, "Weather")

    print("\nWeather sites:")
    print(sorted(df["site_name"].unique()))

    print("\nWeather years:")
    print(sorted(df["year"].unique()))

    print("\nWeather collection windows:")
    print(sorted(df["collection_window"].dropna().unique()))

    return df


def load_spectral():
    section("LOADING SPECTRAL DATA")

    if not SPECTRAL_FILE.exists():
        raise FileNotFoundError(f"Spectral file not found:\n{SPECTRAL_FILE}")

    df = pd.read_csv(SPECTRAL_FILE)

    required = [
        "site_name",
        "year",
        "date",
        "collection_window",
        *SPECTRAL_COLUMNS,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Spectral is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["collection_window"] = df["collection_window"].map(normalize_window)

    if df["date"].isna().any():
        raise ValueError("Spectral contains invalid dates.")

    if df["year"].isna().any():
        raise ValueError("Spectral contains invalid year values.")

    df["year"] = df["year"].astype(int)

    for col in SPECTRAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    report_dataframe(df, "Spectral")

    # Inf is never a valid spectral value.
    inf_mask = np.isinf(
        df[SPECTRAL_COLUMNS].to_numpy(dtype=float)
    )

    if inf_mask.any():
        raise ValueError(
            "Spectral contains Inf values. Clean the spectral source first."
        )

    print("\nSpectral sites:")
    print(sorted(df["site_name"].unique()))

    print("\nSpectral years:")
    print(sorted(df["year"].unique()))

    return df


# ---------------------------------------------------------------------
# Spectral preparation
# ---------------------------------------------------------------------

def prepare_daily_spectral(spectral):
    """
    Convert the spectral file to one observation per
    site/year/date.

    The source contains three collection-window copies of each
    satellite observation. Since the spectral values are identical
    across those windows, the temporal interpolation is performed
    once per day rather than three times.
    """

    section("PREPARING DAILY SPECTRAL OBSERVATIONS")

    # Check whether different windows actually disagree.
    grouped = (
        spectral
        .groupby(["site_name", "year", "date"], as_index=False)
    )

    rows = []

    for (site, year, date), g in grouped:
        row = {
            "site_name": site,
            "year": int(year),
            "date": date,
        }

        for col in SPECTRAL_COLUMNS:
            vals = g[col].dropna().to_numpy(dtype=float)

            if len(vals) == 0:
                row[col] = np.nan
            else:
                # If multiple window values exist, use their mean.
                # In the supplied data these values are repeated copies.
                row[col] = float(np.mean(vals))

                if len(vals) > 1:
                    spread = float(np.max(vals) - np.min(vals))
                    if spread > 1e-8:
                        warnings.warn(
                            f"{site} {year} {date.date()} has differing "
                            f"{col} values across windows; using mean."
                        )

        rows.append(row)

    daily = pd.DataFrame(rows)

    # A second safety check.
    if daily.duplicated(
        ["site_name", "year", "date"]
    ).any():
        raise ValueError(
            "Daily spectral preparation produced duplicate site/year/date keys."
        )

    daily = daily.sort_values(
        ["site_name", "year", "date"]
    ).reset_index(drop=True)

    print(f"Original spectral rows      : {len(spectral):,}")
    print(f"Daily spectral observations : {len(daily):,}")

    coverage = (
        daily.groupby(["site_name", "year"])
        .agg(
            observations=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    print("\nSpectral coverage:")
    print(coverage.to_string(index=False))

    return daily


# ---------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------

def interpolate_spectral_to_weather_dates(weather, spectral_daily):
    """
    Interpolate spectral variables onto every Weather date.

    Interpolation is strictly within each site/year. This is critical:
    2021 observations cannot influence 2022, etc.

    Boundary handling:
      before first spectral acquisition -> first observed value
      after last spectral acquisition  -> last observed value

    This creates a complete growing-season master grid without
    cross-year leakage.
    """

    section("INTERPOLATING SPECTRAL DATA TO WEATHER DATES")

    weather_dates = (
        weather[
            ["site_name", "year", "date"]
        ]
        .drop_duplicates()
        .sort_values(["site_name", "year", "date"])
        .reset_index(drop=True)
    )

    result_parts = []

    weather_groups = weather_dates.groupby(
        ["site_name", "year"], sort=False
    )

    for (site, year), wg in weather_groups:

        wg = wg.sort_values("date").copy()

        sg = spectral_daily[
            (spectral_daily["site_name"] == site)
            & (spectral_daily["year"] == year)
        ].copy()

        if sg.empty:
            raise ValueError(
                f"No spectral observations exist for {site}, {year}. "
                "Cannot generate a synchronized spectral modality."
            )

        sg = sg.sort_values("date")

        # Use integer day coordinates.
        target_days = (
            wg["date"] - pd.Timestamp("1970-01-01")
        ).dt.total_seconds() / 86400.0

        source_days = (
            sg["date"] - pd.Timestamp("1970-01-01")
        ).dt.total_seconds() / 86400.0

        part = wg.copy()

        for col in SPECTRAL_COLUMNS:

            valid = sg[col].notna()

            if valid.sum() == 0:
                raise ValueError(
                    f"{site} {year}: spectral variable {col} "
                    "contains no valid observations."
                )

            x = source_days[valid].to_numpy(dtype=float)
            y = sg.loc[valid, col].to_numpy(dtype=float)
            xp = target_days.to_numpy(dtype=float)

            if len(x) == 1:
                interpolated = np.full(
                    len(xp),
                    y[0],
                    dtype=float,
                )
            else:
                # np.interp performs linear interpolation internally and
                # carries the endpoint values outside the source range.
                interpolated = np.interp(
                    xp,
                    x,
                    y,
                    left=y[0],
                    right=y[-1],
                )

            part[col] = interpolated

        result_parts.append(part)

    daily_weather_spectral = pd.concat(
        result_parts,
        ignore_index=True,
    )

    print(
        f"Weather unique site/date rows : "
        f"{len(weather_dates):,}"
    )

    print(
        f"Synchronized daily rows       : "
        f"{len(daily_weather_spectral):,}"
    )

    # Coverage report.
    coverage = (
        daily_weather_spectral
        .groupby(["site_name", "year"])
        .size()
        .reset_index(name="weather_dates")
    )

    print("\nSynchronized coverage:")
    print(coverage.to_string(index=False))

    return daily_weather_spectral


# ---------------------------------------------------------------------
# Attach spectral to the 8-hour weather grid
# ---------------------------------------------------------------------

def attach_spectral_to_weather(weather, daily_spectral):
    """
    Merge daily synchronized spectral values into every Weather
    collection window.

    Since spectral values are daily satellite observations and the
    Weather grid has three windows/day, the same daily spectral
    observation is associated with each weather window.
    """

    section("ATTACHING SPECTRAL DATA TO WEATHER 8-HOUR GRID")

    merge_keys = ["site_name", "year", "date"]

    # Only bring spectral columns.
    spectral_for_merge = daily_spectral[
        merge_keys + SPECTRAL_COLUMNS
    ].copy()

    # Safety check.
    if spectral_for_merge.duplicated(merge_keys).any():
        raise ValueError(
            "Synchronized spectral data are not unique on "
            "site/year/date."
        )

    output = weather.merge(
        spectral_for_merge,
        on=merge_keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # Preserve original weather ordering.
    output = output.sort_values(
        ["site_name", "year", "date", "collection_window"]
    ).reset_index(drop=True)

    print(f"Weather rows                 : {len(weather):,}")
    print(f"Output rows                  : {len(output):,}")

    if len(output) != len(weather):
        raise ValueError(
            "Row count changed during spectral attachment. "
            "This indicates an invalid merge."
        )

    return output


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_output(df, weather):
    section("VALIDATING FINAL WEATHER + SPECTRAL DATASET")

    expected_rows = len(weather)

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, got {len(df):,}."
        )

    if df.duplicated(KEY_COLUMNS).any():
        dup = df.loc[
            df.duplicated(KEY_COLUMNS, keep=False),
            KEY_COLUMNS,
        ].head(20)

        print(dup.to_string(index=False))

        raise ValueError(
            "Final dataset contains duplicate weather temporal keys."
        )

    # Validate numerical finiteness.
    numeric = df.select_dtypes(include=[np.number])

    if numeric.isna().any().any():
        bad = numeric.isna().sum()
        bad = bad[bad > 0]
        print("\nNumerical NaNs:")
        print(bad.to_string())

        raise ValueError(
            "Final dataset contains numerical NaNs."
        )

    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError(
            "Final dataset contains Inf values."
        )

    # Spectral-specific validation.
    for col in SPECTRAL_COLUMNS:
        if not np.isfinite(
            df[col].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Spectral column {col} contains NaN/Inf."
            )

    # Check expected windows.
    windows = set(df["collection_window"].dropna().unique())

    unexpected = windows - EXPECTED_WINDOWS
    if unexpected:
        warnings.warn(
            f"Unexpected final collection windows: {sorted(unexpected)}"
        )

    # Site/year row counts.
    coverage = (
        df.groupby(["site_name", "year"])
        .size()
        .reset_index(name="rows")
    )

    print("\nFinal site/year coverage:")
    print(coverage.to_string(index=False))

    print("\nFinal spectral statistics:")
    stats = df[SPECTRAL_COLUMNS].agg(
        ["min", "max", "mean", "std"]
    ).T
    print(stats.to_string())

    print("\nValidation: PASSED")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    section("WEATHER + SPECTRAL TEMPORAL SYNCHRONIZATION")

    print(f"Weather input : {WEATHER_FILE}")
    print(f"Spectral input: {SPECTRAL_FILE}")
    print(f"Output        : {OUTPUT_FILE}")

    print("\nStrategy:")
    print("  1. Weather = master 8-hour temporal grid")
    print("  2. Spectral = lower-frequency observations")
    print("  3. Collapse repeated spectral windows to daily observations")
    print("  4. Interpolate within each site/year only")
    print("  5. Carry nearest boundary observation at season edges")
    print("  6. Attach daily spectral values to all weather windows")
    print("  7. Validate row count, keys, NaN and Inf")

    weather = load_weather()
    spectral = load_spectral()

    spectral_daily = prepare_daily_spectral(spectral)

    synchronized_daily = interpolate_spectral_to_weather_dates(
        weather,
        spectral_daily,
    )

    output = attach_spectral_to_weather(
        weather,
        synchronized_daily,
    )

    validate_output(
        output,
        weather,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.8f",
    )

    section("SAVED")

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows       : {len(output):,}")
    print(f"Columns    : {len(output.columns)}")

    print("\nSpectral columns:")
    for c in SPECTRAL_COLUMNS:
        print(f"  {c}")

    print("\nFirst rows:")
    print(
        output[
            [
                "site_name",
                "year",
                "date",
                "collection_window",
                *SPECTRAL_COLUMNS,
            ]
        ].head(12).to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
