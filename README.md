# MuSTIPest: Multimodal Pest Dataset Generation with Lotka–Volterra Dynamics

## Overview

This repository provides the data-generation pipeline for **MuSTIPest**, a multimodal dataset designed for machine-learning-based pest population prediction.

The dataset integrates heterogeneous environmental modalities collected across multiple growing seasons and geographic locations:

- Weather observations
- Vegetation spectral indices
- Dynamic soil spectral indices
- Static soil properties
- A synthetic aphid population target generated using a **Lotka–Volterra (LV) ecological dynamical system**

The primary objective is to create a temporally synchronized multimodal dataset suitable for:

- Multimodal deep learning
- Pest population forecasting
- Environmental intelligence
- Agricultural AI
- Time-series regression
- Explainable AI
- Modality and architecture ablation studies

The current dataset covers **2021–2025** and three counties in Iowa, USA:

- Jasper County
- Polk County
- Story County

---

# 1. Dataset Overview

The dataset is constructed from four environmental modalities.

| Modality | Type | Temporal | Features |
|---|---|---|---:|
| Weather | Environmental | Dynamic | 21 |
| Vegetation | Sentinel-2 | Dynamic | 3 |
| Dynamic Soil | Sentinel-2 | Dynamic | 4 |
| Static Soil | Soil database | Static | 28 |
| Aphid target | LV dynamics | Dynamic | 1 |

The final learning problem is formulated as:

\[
X_{t-L+1:t} \rightarrow y_{t+1}
\]

where:

- \(X\) represents the multimodal environmental observations,
- \(L\) is the temporal sequence length,
- \(y_{t+1}\) is the predicted aphid population at the next time step.

---

# 2. Geographic Coverage

The dataset contains observations from three agriculturally significant counties in Iowa.

| Site | Latitude | Longitude |
|---|---:|---:|
| Jasper County | 41.6932 | -93.0538 |
| Polk County | 41.6278 | -93.5815 |
| Story County | 42.0347 | -93.5813 |

These locations are representative of the Midwestern United States, an important soybean-producing region.

---

# 3. Temporal Coverage

The dataset covers five growing seasons:

```text
2021
2022
2023
2024
2025
