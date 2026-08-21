"""
===============================================================
MuSTIPest-V3
LOTKA-VOLTERRA SYNTHETIC APHID TARGET GENERATION
===============================================================

Input:
    multimodal_ALL_SITES_2021_2025.csv

Output:
    multimodal_LV_TARGET_ALL_SITES_2021_2025.csv

Target:
    Synthetic soybean aphid population generated using
    environmentally forced Lotka-Volterra predator-prey dynamics.

Species:
    Prey    : Aphis glycines (Soybean Aphid)
    Predator: Natural enemies (Coccinellidae + Parasitoids)

Dynamics:

    dN/dt = r(T,H,V) * N * (1 - N/K) - alpha*N*P

    dP/dt = beta*N*P - delta*P

Environmental forcing:

    r  <- temperature + humidity + vegetation
    K  <- vegetation + soil moisture + soil properties
    alpha <- predator pressure / environmental suitability

Temporal resolution:
    Weather observations are on an 8-hour grid.
    Therefore dt = 1/3 day.

Important:
    The resulting aphid target is SYNTHETIC / MECHANISTIC.
    It must not be described as measured field aphid counts.
===============================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "multimodal_ALL_SITES_2021_2025.csv"
)

OUTPUT_FILE = (
    "multimodal_LV_TARGET_ALL_SITES_2021_2025.csv"
)


# ------------------------------------------------------------
# Time step
# ------------------------------------------------------------

# Weather observations:
#
# 00:00
# 08:00
# 16:00
#
# 3 observations/day
#
DT = 1.0 / 3.0


# ============================================================
# BIOLOGICAL PARAMETERS
# ============================================================

# Maximum intrinsic aphid growth rate per day.
#
# This is a synthetic-model parameter and should be reported
# as such unless calibrated using experimental literature/data.
R_MAX = 0.32


# Minimum temperature suitability threshold.
T_MIN = 5.0


# Optimum temperature for aphid development.
T_OPT = 25.0


# Upper temperature limit.
T_MAX = 35.0


# Baseline carrying capacity.
K_BASE = 8.0


# Maximum environmentally supported population.
K_MAX = 60.0


# Predation coefficient.
ALPHA_BASE = 0.0045


# Predator conversion efficiency.
BETA_BASE = 0.0020


# Predator natural mortality per day.
DELTA_BASE = 0.035


# Initial aphid population.
INITIAL_N = 0.25


# Initial predator population.
INITIAL_P = 0.60


# Numerical integration substeps.
#
# Smaller substeps make the LV integration more stable.
SUBSTEPS = 8


# Optional deterministic observation transformation.
#
# The mechanistic state is N.
# We report:
#
# aphids_per_plant = N
#
# No random noise is added by default.
ADD_OBSERVATION_NOISE = False

NOISE_STD = 0.05


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "site_name",
    "year",
    "date",
    "time_local",
    "collection_window",

    # Weather
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "vapour_pressure_deficit",
    "soil_moisture_0_to_7cm",

    # Vegetation
    "NDVI",
    "NDWI",
    "EVI",

    # Dynamic soil
    "BSI",
    "SAVI",
    "NDTI",
    "RI",

    # Static soil
    "ph_0-5cm",
    "organic_carbon_0-5cm",
    "cec_0-5cm",
    "bulk_density_0-5cm",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def sigmoid(x):
    """
    Numerically stable sigmoid.
    """

    x = np.clip(x, -40.0, 40.0)

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def minmax_clip(x, xmin, xmax):
    """
    Map x approximately to [0,1].
    """

    if xmax <= xmin:
        return 0.5

    z = (
        np.asarray(x, dtype=float)
        - xmin
    ) / (
        xmax - xmin
    )

    return np.clip(z, 0.0, 1.0)


def safe_numeric(
    df,
    columns,
):
    """
    Convert selected columns to numeric
    and reject non-finite values.
    """

    for col in columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    bad = (
        df[columns]
        .isna()
        .sum()
    )

    bad = bad[bad > 0]

    if len(bad) > 0:

        print(
            "\nMissing values found:"
        )

        print(bad)

        raise ValueError(
            "Required environmental "
            "features contain NaN."
        )

    arr = (
        df[columns]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(arr).all():

        raise ValueError(
            "Required environmental "
            "features contain Inf."
        )

    return df


# ============================================================
# TEMPERATURE RESPONSE
# ============================================================

def temperature_response(
    temperature
):
    """
    Temperature suitability function.

    A triangular response is used:

        T <= Tmin          -> 0
        Tmin < T < Topt    -> increasing
        Topt <= T < Tmax   -> decreasing
        T >= Tmax          -> 0

    Returns:
        [0,1]
    """

    T = float(temperature)

    if T <= T_MIN:
        return 0.0

    if T < T_OPT:

        return (
            (T - T_MIN)
            /
            (T_OPT - T_MIN)
        )

    if T < T_MAX:

        return (
            (T_MAX - T)
            /
            (T_MAX - T_OPT)
        )

    return 0.0


# ============================================================
# HUMIDITY RESPONSE
# ============================================================

def humidity_response(
    humidity
):
    """
    Humidity suitability.

    Peak suitability is obtained around
    moderate-high relative humidity.
    """

    H = float(humidity)

    # Broad response centered around 70%.
    response = np.exp(
        -0.5
        *
        (
            (H - 70.0)
            /
            20.0
        ) ** 2
    )

    return float(
        np.clip(
            response,
            0.0,
            1.0,
        )
    )


# ============================================================
# VEGETATION RESPONSE
# ============================================================

def vegetation_response(
    ndvi,
    evi,
    ndwi,
):
    """
    Composite vegetation suitability.

    NDVI:
        canopy development

    EVI:
        vegetation productivity

    NDWI:
        vegetation water condition
    """

    ndvi_score = minmax_clip(
        ndvi,
        0.0,
        1.0,
    )

    evi_score = minmax_clip(
        evi,
        0.0,
        1.0,
    )

    # NDWI typically ranges approximately
    # between -1 and +1.
    ndwi_score = minmax_clip(
        ndwi,
        -1.0,
        1.0,
    )

    score = (
        0.50 * ndvi_score
        +
        0.30 * evi_score
        +
        0.20 * ndwi_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# SOIL SUITABILITY
# ============================================================

def soil_suitability(
    soil_moisture,
    organic_carbon,
    ph,
):
    """
    Environmental soil suitability.

    This does NOT claim that soil directly determines
    aphid population.

    Instead, it acts as a weak environmental modifier
    of host-plant carrying capacity.
    """

    # Soil moisture.
    #
    # The absolute scale depends on the data source,
    # therefore use a bounded response.
    moisture_score = sigmoid(
        (
            float(soil_moisture)
            - 0.25
        )
        /
        0.08
    )

    # Organic carbon.
    #
    # SoilGrids/SSURGO values in this dataset may use
    # source-specific units. We therefore use a bounded
    # sigmoid rather than assuming a physical percentage.
    oc_score = sigmoid(
        (
            float(organic_carbon)
            - 10.0
        )
        /
        5.0
    )

    # Soybean host suitability around mildly acidic-neutral
    # conditions.
    ph_score = np.exp(
        -0.5
        *
        (
            (
                float(ph)
                - 6.5
            )
            /
            1.0
        ) ** 2
    )

    score = (
        0.50 * moisture_score
        +
        0.25 * oc_score
        +
        0.25 * ph_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# ENVIRONMENTALLY FORCED PARAMETERS
# ============================================================

def compute_environment(
    row
):
    """
    Calculate r, K, alpha and beta for one time step.
    """

    T = float(
        row["temperature_2m"]
    )

    H = float(
        row["relative_humidity_2m"]
    )

    ndvi = float(
        row["NDVI"]
    )

    evi = float(
        row["EVI"]
    )

    ndwi = float(
        row["NDWI"]
    )

    soil_moisture = float(
        row["soil_moisture_0_to_7cm"]
    )

    organic_carbon = float(
        row["organic_carbon_0-5cm"]
    )

    ph = float(
        row["ph_0-5cm"]
    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temp_score = temperature_response(
        T
    )

    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    humidity_score = humidity_response(
        H
    )

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    veg_score = vegetation_response(
        ndvi,
        evi,
        ndwi,
    )

    # --------------------------------------------------------
    # Soil
    # --------------------------------------------------------

    soil_score = soil_suitability(
        soil_moisture,
        organic_carbon,
        ph,
    )

    # --------------------------------------------------------
    # Aphid intrinsic growth
    # --------------------------------------------------------

    # Temperature is the dominant driver.
    #
    # Humidity and vegetation provide secondary
    # environmental modulation.

    growth_modifier = (
        0.60 * temp_score
        +
        0.20 * humidity_score
        +
        0.20 * veg_score
    )

    r = (
        R_MAX
        *
        growth_modifier
    )

    # --------------------------------------------------------
    # Carrying capacity
    # --------------------------------------------------------

    environment_score = (
        0.60 * veg_score
        +
        0.25 * soil_score
        +
        0.15 * humidity_score
    )

    K = (
        K_BASE
        +
        (
            K_MAX - K_BASE
        )
        *
        environment_score
    )

    # --------------------------------------------------------
    # Predation
    # --------------------------------------------------------

    # Predator interaction is allowed to increase under
    # favorable vegetation conditions.
    #
    # This keeps predation coupled to the ecological state
    # without introducing many additional free parameters.

    alpha = (
        ALPHA_BASE
        *
        (
            0.60
            +
            0.40 * veg_score
        )
    )

    # --------------------------------------------------------
    # Predator conversion
    # --------------------------------------------------------

    beta = (
        BETA_BASE
        *
        (
            0.75
            +
            0.25 * veg_score
        )
    )

    # --------------------------------------------------------
    # Predator mortality
    # --------------------------------------------------------

    delta = DELTA_BASE

    return {
        "temperature_suitability":
            temp_score,

        "humidity_suitability":
            humidity_score,

        "vegetation_suitability":
            veg_score,

        "soil_suitability":
            soil_score,

        "r":
            r,

        "K":
            K,

        "alpha":
            alpha,

        "beta":
            beta,

        "delta":
            delta,
    }


# ============================================================
# LOTKA-VOLTERRA DERIVATIVE
# ============================================================

def lv_derivatives(
    N,
    P,
    r,
    K,
    alpha,
    beta,
    delta,
):
    """
    Lotka-Volterra predator-prey equations.

        dN/dt =
            r*N*(1-N/K)
            - alpha*N*P

        dP/dt =
            beta*N*P
            - delta*P
    """

    # Numerical safety.
    N = max(
        float(N),
        0.0,
    )

    P = max(
        float(P),
        0.0,
    )

    K = max(
        float(K),
        1e-6,
    )

    dN = (
        r
        * N
        * (
            1.0
            -
            N / K
        )
        -
        alpha
        * N
        * P
    )

    dP = (
        beta
        * N
        * P
        -
        delta
        * P
    )

    return dN, dP


# ============================================================
# STABLE INTEGRATION
# ============================================================

def integrate_one_step(
    N,
    P,
    env,
    dt=DT,
):
    """
    Positivity-preserving sub-stepped Euler integration.

    For synthetic target generation this is intentionally
    simple and transparent.

    The environment is assumed constant during the
    8-hour interval.
    """

    sub_dt = (
        dt
        /
        SUBSTEPS
    )

    r = env["r"]
    K = env["K"]
    alpha = env["alpha"]
    beta = env["beta"]
    delta = env["delta"]

    for _ in range(
        SUBSTEPS
    ):

        dN, dP = lv_derivatives(
            N,
            P,
            r,
            K,
            alpha,
            beta,
            delta,
        )

        N = (
            N
            +
            sub_dt * dN
        )

        P = (
            P
            +
            sub_dt * dP
        )

        # Enforce ecological positivity.
        N = max(
            N,
            0.0,
        )

        P = max(
            P,
            0.0,
        )

        # Prevent numerical explosion.
        N = min(
            N,
            10.0 * K,
        )

        P = min(
            P,
            1000.0,
        )

    return N, P


# ============================================================
# INITIAL CONDITIONS
# ============================================================

def initialize_population(
    site,
    year,
):
    """
    Deterministic initialization.

    A small site/year effect avoids making every trajectory
    identical while remaining reproducible.
    """

    site_hash = (
        sum(
            ord(c)
            for c in str(site)
        )
        % 17
    )

    year_effect = (
        int(year)
        % 5
    )

    N0 = (
        INITIAL_N
        *
        (
            0.90
            +
            0.01 * site_hash
            +
            0.02 * year_effect
        )
    )

    P0 = (
        INITIAL_P
        *
        (
            0.90
            +
            0.005 * site_hash
        )
    )

    return (
        max(N0, 0.01),
        max(P0, 0.01),
    )


# ============================================================
# GENERATE ONE SITE-YEAR
# ============================================================

def simulate_site_year(
    group
):
    """
    Simulate one site-year chronologically.
    """

    group = group.sort_values(
        [
            "date",
            "time_local",
        ]
    ).copy()

    site = group[
        "site_name"
    ].iloc[0]

    year = int(
        group["year"].iloc[0]
    )

    N, P = initialize_population(
        site,
        year,
    )

    results = []

    for _, row in group.iterrows():

        # ----------------------------------------------
        # Environmental forcing
        # ----------------------------------------------

        env = compute_environment(
            row
        )

        # ----------------------------------------------
        # Store current state
        # ----------------------------------------------

        results.append({

            "aphid_latent_N":
                N,

            "predator_latent_P":
                P,

            "lv_r":
                env["r"],

            "lv_K":
                env["K"],

            "lv_alpha":
                env["alpha"],

            "lv_beta":
                env["beta"],

            "lv_delta":
                env["delta"],

            "temperature_suitability":
                env[
                    "temperature_suitability"
                ],

            "humidity_suitability":
                env[
                    "humidity_suitability"
                ],

            "vegetation_suitability":
                env[
                    "vegetation_suitability"
                ],

            "soil_suitability":
                env[
                    "soil_suitability"
                ],
        })

        # ----------------------------------------------
        # Advance population
        # ----------------------------------------------

        N, P = integrate_one_step(
            N,
            P,
            env,
            DT,
        )

    result_df = pd.DataFrame(
        results,
        index=group.index,
    )

    for col in result_df.columns:

        group[col] = (
            result_df[col]
        )

    return group


# ============================================================
# TARGET POST-PROCESSING
# ============================================================

def create_target(
    df
):
    """
    Create the final aphid target.

    The mechanistic state N is converted to the reported
    synthetic aphid count per plant.

    No arbitrary MinMax scaling is performed here.
    Scaling should happen later in preprocessing.py.

    This is important because the raw target should remain
    interpretable.
    """

    df[
        "aphids_per_plant"
    ] = np.maximum(
        df[
            "aphid_latent_N"
        ].astype(float),
        0.0,
    )

    # --------------------------------------------------------
    # Optional observation noise
    # --------------------------------------------------------

    if ADD_OBSERVATION_NOISE:

        rng = np.random.default_rng(
            42
        )

        noise = rng.normal(
            loc=0.0,
            scale=NOISE_STD,
            size=len(df),
        )

        df[
            "aphids_per_plant"
        ] *= np.exp(
            noise
        )

        df[
            "aphids_per_plant"
        ] = np.maximum(
            df[
                "aphids_per_plant"
            ],
            0.0,
        )

    return df


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_output(
    df
):

    print()
    print("=" * 80)
    print(
        "TARGET VALIDATION"
    )
    print("=" * 80)

    target = (
        df[
            "aphids_per_plant"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print(
        f"Target min  : "
        f"{target.min():.6f}"
    )

    print(
        f"Target p1   : "
        f"{np.percentile(target, 1):.6f}"
    )

    print(
        f"Target p25  : "
        f"{np.percentile(target, 25):.6f}"
    )

    print(
        f"Target median: "
        f"{np.median(target):.6f}"
    )

    print(
        f"Target p75  : "
        f"{np.percentile(target, 75):.6f}"
    )

    print(
        f"Target p95  : "
        f"{np.percentile(target, 95):.6f}"
    )

    print(
        f"Target p99  : "
        f"{np.percentile(target, 99):.6f}"
    )

    print(
        f"Target max  : "
        f"{target.max():.6f}"
    )

    print(
        f"Target mean : "
        f"{target.mean():.6f}"
    )

    print(
        f"Target std  : "
        f"{target.std():.6f}"
    )

    print(
        f"Target zeros: "
        f"{np.sum(target == 0)}"
    )

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------

    if not np.isfinite(
        target
    ).all():

        raise ValueError(
            "Target contains NaN or Inf."
        )

    if np.any(
        target < 0
    ):

        raise ValueError(
            "Negative aphid target detected."
        )

    # --------------------------------------------------------
    # State validation
    # --------------------------------------------------------

    for col in [
        "aphid_latent_N",
        "predator_latent_P",
        "lv_r",
        "lv_K",
        "lv_alpha",
        "lv_beta",
        "lv_delta",
    ]:

        values = df[
            col
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                f"{col} contains NaN/Inf."
            )

    print(
        "\nNumerical validation: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "MuSTIPest-V3"
    )
    print(
        "LOTKA-VOLTERRA SYNTHETIC TARGET GENERATION"
    )
    print("=" * 80)

    print(
        "\nInput:"
    )

    print(
        f"  {INPUT_FILE}"
    )

    print(
        "\nOutput:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print(
        "Model:"
    )

    print(
        "  Prey     : Aphis glycines"
    )

    print(
        "  Predator : Coccinellidae + Parasitoids"
    )

    print(
        "  Time step: 8 hours"
    )

    print(
        "  dt       : 1/3 day"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    path = Path(
        INPUT_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{path.resolve()}"
        )

    df = pd.read_csv(
        path
    )

    print()
    print(
        f"Loaded dataset: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            +
            "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():

        raise ValueError(
            "Invalid date values detected."
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    environmental_columns = [

        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "vapour_pressure_deficit",
        "soil_moisture_0_to_7cm",

        "NDVI",
        "NDWI",
        "EVI",

        "BSI",
        "SAVI",
        "NDTI",
        "RI",

        "ph_0-5cm",
        "organic_carbon_0-5cm",
        "cec_0-5cm",
        "bulk_density_0-5cm",
    ]

    df = safe_numeric(
        df,
        environmental_columns,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "site_name",
            "year",
            "date",
            "time_local",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Check temporal duplicates
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "site_name",
            "date",
            "collection_window",
        ],
        keep=False,
    )

    if duplicates.any():

        print(
            "\nWARNING:"
        )

        print(
            "Temporal duplicate observations:"
            f" {duplicates.sum()}"
        )

        raise ValueError(
            "Input dataset must contain "
            "one observation per site/date/"
            "collection_window."
        )

    # --------------------------------------------------------
    # Simulate each site-year independently
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "RUNNING LV DYNAMICS"
    )
    print("=" * 80)

    groups = []

    grouped = df.groupby(
        [
            "site_name",
            "year",
        ],
        sort=True,
    )

    print(
        f"\nSite-year trajectories: "
        f"{len(grouped)}"
    )

    for (site, year), group in grouped:

        print(
            f"  Simulating "
            f"{site} - {year} "
            f"({len(group)} observations)"
        )

        simulated = simulate_site_year(
            group
        )

        groups.append(
            simulated
        )

    result = pd.concat(
        groups,
        axis=0,
    )

    result = result.sort_values(
        [
            "site_name",
            "year",
            "date",
            "time_local",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Create final target
    # --------------------------------------------------------

    result = create_target(
        result
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_output(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "LV TARGET GENERATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"\nSaved:"
    )

    print(
        f"  {Path(OUTPUT_FILE).resolve()}"
    )

    print(
        f"\nFinal shape:"
        f" {result.shape}"
    )

    print(
        "\nNew target:"
    )

    print(
        "  aphids_per_plant"
    )

    print(
        "\nAdditional LV diagnostics:"
    )

    print(
        "  aphid_latent_N"
    )

    print(
        "  predator_latent_P"
    )

    print(
        "  lv_r"
    )

    print(
        "  lv_K"
    )

    print(
        "  lv_alpha"
    )

    print(
        "  lv_beta"
    )

    print(
        "  lv_delta"
    )

    print(
        "  temperature_suitability"
    )

    print(
        "  humidity_suitability"
    )

    print(
        "  vegetation_suitability"
    )

    print(
        "  soil_suitability"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The aphid target is mechanistically "
        "synthetic and should not be presented "
        "as measured field observations."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()