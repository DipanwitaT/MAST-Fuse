"""
=============================================================
 Soybean Yield Prediction — Weather Data Collection
=============================================================
 Source  : Open-Meteo Historical API (free, no API key)
 Interval: Every 8 hours  →  00:00 / 08:00 / 16:00 UTC
 Output  : CSV per coordinate + one merged CSV

 HOW TO USE
 ----------
 1. Install dependencies:
       pip install openmeteo-requests requests-cache retry-requests pandas

 2. Set your coordinates in COORDINATES (add as many as you need):
       COORDINATES = [
           {"name": "Site_A", "lat": 30.1234, "lon": 71.5678},
           {"name": "Site_B", "lat": 31.0000, "lon": 72.0000},
       ]

 3. Set your date range:
       START_DATE = "2021-01-01"
       END_DATE   = "2025-12-31"

 4. Run:
       python weather_collection.py
=============================================================
"""

import pandas as pd
import requests
import requests_cache
from retry_requests import retry
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  USER CONFIGURATION — edit these
# ─────────────────────────────────────────────

COORDINATES = [
    {"name": "Jasper_County", "lat": 41.6932, "lon": -93.0538},  # Iowa
    {"name": "Polk_County",   "lat": 41.6278, "lon": -93.5815},  # Iowa
    {"name": "Story_County",  "lat": 42.0347, "lon": -93.5813},  # Iowa
]

# Soybean growing season: April 1 → September 30, across 2019–2023
DATE_RANGES = [
    
    ("2021-04-01", "2021-09-30"),
    ("2022-04-01", "2022-09-30"),
    ("2023-04-01", "2023-09-30"),
    ("2024-04-01", "2024-09-30"),
    ("2025-04-01", "2025-09-30"),
]

OUTPUT_DIR = "weather_data"  # Folder where CSVs will be saved

# Collection hours (UTC) — every 8 hours
COLLECTION_HOURS = [0, 8, 16]

# ─────────────────────────────────────────────
#  WEATHER VARIABLES to collect
#  (all available at hourly resolution)
# ─────────────────────────────────────────────

HOURLY_VARIABLES = [
    "temperature_2m",            # Air temperature at 2m (°C)
    "relative_humidity_2m",      # Relative humidity at 2m (%)
    "dew_point_2m",              # Dew point at 2m (°C)
    "precipitation",             # Rainfall + snowmelt (mm)
    "rain",                      # Rain only (mm)
    "wind_speed_10m",            # Wind speed at 10m (km/h)
    "wind_direction_10m",        # Wind direction at 10m (°)
    "wind_gusts_10m",            # Wind gusts at 10m (km/h)
    "surface_pressure",          # Atmospheric pressure (hPa)
    "cloud_cover",               # Total cloud cover (%)
    "vapour_pressure_deficit",   # VPD — important for crop stress (kPa)
    "et0_fao_evapotranspiration",# Reference ET (mm) — key for irrigation
    "shortwave_radiation",       # Solar radiation (W/m²)
    "direct_radiation",          # Direct solar radiation (W/m²)
    "diffuse_radiation",         # Diffuse solar radiation (W/m²)
    "soil_temperature_0_to_7cm", # Soil temp 0–7 cm depth (°C)
    "soil_moisture_0_to_7cm",    # Soil moisture 0–7 cm (m³/m³)
]

# ─────────────────────────────────────────────
#  SETUP — caching & retry session
# ─────────────────────────────────────────────

cache_session = requests_cache.CachedSession(".weather_cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)


# ─────────────────────────────────────────────
#  FETCH FUNCTION
# ─────────────────────────────────────────────

def fetch_weather(site: dict, start: str, end: str) -> pd.DataFrame:
    """
    Fetch hourly weather for one coordinate from Open-Meteo,
    then filter to the 3 daily collection times (00:00, 08:00, 16:00 UTC).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        site["lat"],
        "longitude":       site["lon"],
        "start_date":      start,
        "end_date":        end,
        "hourly":          HOURLY_VARIABLES,
        "timezone":        "America/Chicago",  # Iowa — US Central Time
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    print(f"  → Fetching: {site['name']} ({site['lat']}, {site['lon']}) ...")
    response = retry_session.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Build DataFrame from hourly response
    hourly = data.get("hourly", {})
    df = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"])})

    for var in HOURLY_VARIABLES:
        if var in hourly:
            df[var] = hourly[var]
        else:
            df[var] = None

    # ── Filter to 3 collection windows per day ──
    df = df[df["datetime"].dt.hour.isin(COLLECTION_HOURS)].copy()
    df.reset_index(drop=True, inplace=True)

    # Add metadata columns
    df.insert(0, "site_name", site["name"])
    df.insert(1, "latitude",  site["lat"])
    df.insert(2, "longitude", site["lon"])
    df.insert(3, "date",      df["datetime"].dt.date)
    df.insert(4, "time_local", df["datetime"].dt.strftime("%H:%M"))
    df.insert(5, "collection_window", df["datetime"].dt.hour.map({
        0:  "Window_1_00h",
        8:  "Window_2_08h",
        16: "Window_3_16h",
    }))
    df.drop(columns=["datetime"], inplace=True)

    return df


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = []

    print(f"\n{'='*55}")
    print(f"  Weather Data Collection — Soybean Yield Prediction")
    print(f"  Season : April 1 → September 30  |  Years: 2021–2025")
    print(f"  Windows: 00:00 / 08:00 / 16:00  (America/Chicago — Iowa Local Time)")
    print(f"  Sites  : {len(COORDINATES)}")
    print(f"{'='*55}\n")

    for start, end in DATE_RANGES:
        year = start[:4]
        print(f"  ── Year {year} ({start} → {end}) ──")
        for site in COORDINATES:
            try:
                df = fetch_weather(site, start, end)
                df.insert(3, "year", int(year))
                all_frames.append(df)

                site_file = os.path.join(OUTPUT_DIR, f"weather_{site['name']}_{year}.csv")
                df.to_csv(site_file, index=False)
                print(f"     Saved → {site_file}  ({len(df)} records)")

            except Exception as e:
                print(f"     ERROR for {site['name']} {year}: {e}")
        print()

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        merged_file = os.path.join(OUTPUT_DIR, "weather_ALL_SITES_2021_2025.csv")
        merged.to_csv(merged_file, index=False)
        print(f"{'='*55}")
        print(f"  Merged CSV → {merged_file}")
        print(f"  Total records : {len(merged)}")
        print(f"  Columns       : {list(merged.columns)}")
        print(f"{'='*55}\n")
        print("  Records per site per year:")
        for (site_name, year), grp in merged.groupby(["site_name", "year"]):
            print(f"    {site_name:<20} {year}  →  {len(grp):>4} records")
    print("\n  Done.")



if __name__ == "__main__":
    main()
