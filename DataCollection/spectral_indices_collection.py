"""
=============================================================
 Soybean Yield Prediction — Spectral Indices Collection
=============================================================
 Source   : Sentinel-2 SR (Level-2A) via Google Earth Engine
 Indices  : NDVI, NDWI, EVI, BSI, SAVI, NDTI, RI
 Season   : April 1 → September 30  |  Years: 2021–2025
 Interval : 3 windows/day (00:00 / 08:00 / 16:00 local)
            NOTE: Sentinel-2 revisit = ~5 days, so the script
            collects the closest available image to each
            collection window date and flags the actual
            image acquisition time in the output.
 Sites    : Jasper, Polk, Story Counties — Iowa, USA
 Output   : CSV per site-year + merged CSV

 SETUP (one-time)
 ----------------
 1. Install GEE Python API:
       pip install earthengine-api geemap pandas

 2. Authenticate (first time only):
       python -c "import ee; ee.Authenticate()"
       → Opens browser, sign in with your Google account

 3. Initialize with your GEE project:
       Replace GEE_PROJECT below with your project ID
       (create one free at: https://code.earthengine.google.com)

 4. Run:
       python spectral_indices_collection.py
=============================================================
"""

import ee
import pandas as pd
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  USER CONFIGURATION
# ─────────────────────────────────────────────

GEE_PROJECT = "multimodal-pest-prediction"   # ← Replace with your GEE project ID

COORDINATES = [
    {"name": "Jasper_County", "lat": 41.6932, "lon": -93.0538},
    {"name": "Polk_County",   "lat": 41.6278, "lon": -93.5815},
    {"name": "Story_County",  "lat": 42.0347, "lon": -93.5813},
]

DATE_RANGES = [
    ("2021-04-01", "2021-09-30"),
    ("2022-04-01", "2022-09-30"),
    ("2023-04-01", "2023-09-30"),
    ("2024-04-01", "2024-09-30"),
    ("2025-04-01", "2025-09-30"),
]

# Collection windows (local Iowa time labels)
COLLECTION_WINDOWS = ["00:00", "08:00", "16:00"]

# Cloud cover threshold (%) — images above this are excluded
MAX_CLOUD_COVER = 20

# Buffer radius around each point (meters)
BUFFER_METERS = 5000   # 5 km buffer — covers county-level sampling

OUTPUT_DIR = "spectral_data"

# ─────────────────────────────────────────────
#  INITIALIZE GEE
# ─────────────────────────────────────────────

def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print("  GEE initialized successfully.\n")
    except Exception as e:
        print(f"  GEE init failed: {e}")
        print("  Run: python -c \"import ee; ee.Authenticate()\"  first.\n")
        raise


# ─────────────────────────────────────────────
#  SPECTRAL INDEX CALCULATIONS
#  Sentinel-2 SR bands:
#    B2=Blue, B3=Green, B4=Red, B5=RedEdge,
#    B8=NIR, B8A=RedEdge2, B11=SWIR1, B12=SWIR2
# ─────────────────────────────────────────────

def add_spectral_indices(image):
    """Add all 7 spectral index bands to a Sentinel-2 image."""

    B2  = image.select("B2")   # Blue
    B3  = image.select("B3")   # Green
    B4  = image.select("B4")   # Red
    B8  = image.select("B8")   # NIR
    B11 = image.select("B11")  # SWIR1
    B12 = image.select("B12")  # SWIR2

    # ── NDVI: Normalized Difference Vegetation Index ──
    # Measures green vegetation density
    # Range: -1 to 1  |  Healthy crops: 0.4–0.9
    NDVI = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # ── NDWI: Normalized Difference Water Index ──
    # Detects water content in vegetation/soil
    # Range: -1 to 1  |  Water stress < 0
    NDWI = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

    # ── EVI: Enhanced Vegetation Index ──
    # Improved NDVI — reduces soil & atmosphere noise
    # Gain=2.5, C1=6, C2=7.5, L=1
    EVI = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": B8, "RED": B4, "BLUE": B2}
    ).rename("EVI")

    # ── BSI: Bare Soil Index ──
    # Detects exposed/bare soil areas — useful for tillage monitoring
    # BSI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    BSI = image.expression(
        "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        {"SWIR1": B11, "RED": B4, "NIR": B8, "BLUE": B2}
    ).rename("BSI")

    # ── SAVI: Soil Adjusted Vegetation Index ──
    # NDVI adjusted for soil brightness  |  L=0.5 (intermediate cover)
    # SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
    L = 0.5
    SAVI = image.expression(
        "((NIR - RED) / (NIR + RED + L)) * (1 + L)",
        {"NIR": B8, "RED": B4, "L": L}
    ).rename("SAVI")

    # ── NDTI: Normalized Difference Tillage Index ──
    # Monitors crop residue & tillage practices
    # NDTI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    NDTI = image.normalizedDifference(["B11", "B12"]).rename("NDTI")

    # ── RI: Redness Index ──
    # Indicates iron oxide / redness of soil — soil health proxy
    # RI = RED² / (BLUE × GREEN³)
    RI = image.expression(
        "(RED * RED) / (BLUE * GREEN * GREEN * GREEN)",
        {"RED": B4, "BLUE": B2, "GREEN": B3}
    ).rename("RI")

    return image.addBands([NDVI, NDWI, EVI, BSI, SAVI, NDTI, RI])


# ─────────────────────────────────────────────
#  FETCH FUNCTION — one site, one season
# ─────────────────────────────────────────────

def fetch_indices_for_site(site: dict, start: str, end: str) -> pd.DataFrame:
    """
    Fetch spectral indices for one site across a date range.
    Returns a DataFrame with one row per available Sentinel-2 image,
    replicated across 3 daily collection windows.
    """
    point = ee.Geometry.Point([site["lon"], site["lat"]])
    roi   = point.buffer(BUFFER_METERS)

    # Load Sentinel-2 Surface Reflectance collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .map(add_spectral_indices)
    )

    # Get list of images
    image_list = collection.toList(collection.size())
    count = image_list.size().getInfo()

    if count == 0:
        print(f"     No images found for {site['name']} ({start} → {end})")
        return pd.DataFrame()

    print(f"     {site['name']}: {count} Sentinel-2 images found")

    rows = []
    INDEX_BANDS = ["NDVI", "NDWI", "EVI", "BSI", "SAVI", "NDTI", "RI"]

    for i in range(count):
        img = ee.Image(image_list.get(i))

        # Get image acquisition date
        acq_date = img.date().format("YYYY-MM-dd").getInfo()
        acq_time = img.date().format("HH:mm").getInfo()
        cloud_pct = img.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

        # Extract mean index values within the buffer
        stats = img.select(INDEX_BANDS).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=10,        # Sentinel-2 native resolution (10m)
            maxPixels=1e9
        ).getInfo()

        # Replicate across 3 daily collection windows
        for window in COLLECTION_WINDOWS:
            row = {
                "site_name":          site["name"],
                "latitude":           site["lat"],
                "longitude":          site["lon"],
                "date":               acq_date,
                "image_time_utc":     acq_time,
                "collection_window":  f"Window_{COLLECTION_WINDOWS.index(window)+1}_{window.replace(':','h')}",
                "cloud_cover_pct":    round(cloud_pct, 2) if cloud_pct else None,
            }
            for band in INDEX_BANDS:
                val = stats.get(band)
                row[band] = round(val, 6) if val is not None else None
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    init_gee()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = []

    print(f"{'='*60}")
    print(f"  Spectral Indices Collection — Soybean Yield Prediction")
    print(f"  Sensor  : Sentinel-2 SR (10m resolution)")
    print(f"  Indices : NDVI, NDWI, EVI, BSI, SAVI, NDTI, RI")
    print(f"  Season  : April 1 → September 30  |  2021–2025")
    print(f"  Sites   : Jasper, Polk, Story Counties — Iowa")
    print(f"{'='*60}\n")

    for start, end in DATE_RANGES:
        year = start[:4]
        print(f"  ── Year {year} ({start} → {end}) ──")

        for site in COORDINATES:
            try:
                df = fetch_indices_for_site(site, start, end)
                if df.empty:
                    continue

                df.insert(3, "year", int(year))
                all_frames.append(df)

                out_file = os.path.join(OUTPUT_DIR, f"spectral_{site['name']}_{year}.csv")
                df.to_csv(out_file, index=False)
                print(f"     Saved → {out_file}  ({len(df)} rows)\n")

            except Exception as e:
                print(f"     ERROR — {site['name']} {year}: {e}\n")

        print()

    # Merge all
    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        merged_file = os.path.join(OUTPUT_DIR, "spectral_ALL_SITES_2021_2025.csv")
        merged.to_csv(merged_file, index=False)

        print(f"{'='*60}")
        print(f"  Merged CSV → {merged_file}")
        print(f"  Total rows : {len(merged)}")
        print(f"  Columns    : {list(merged.columns)}")
        print(f"{'='*60}\n")
        print("  Image counts per site per year:")
        for (site_name, year), grp in merged.groupby(["site_name", "year"]):
            images = len(grp) // 3   # divide by 3 windows
            print(f"    {site_name:<20} {year}  →  {images:>3} images  ({len(grp)} rows incl. 3 windows)")

    print("\n  Done.")


if __name__ == "__main__":
    main()