# Multimodal Pest Dataset Generation with Lotka–Volterra Dynamics

## Overview

This repository provides the data-generation pipeline for **MAST-Fuse**, a multimodal dataset designed for machine-learning-based pest population prediction.

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

---

# 4. **LV-Based Synthetic Aphid Target Generation**

## Overview

Because consistently sampled field measurements of soybean aphid
abundance were not available across all locations and seasons at the
required 8-hour temporal resolution, MAST-Fuse uses an ecologically
constrained Lotka--Volterra (LV) model to generate the supervised
aphid-density target.

The LV generator is used to construct a controlled and reproducible
simulation benchmark driven by observed environmental variables. It is
not intended to replace field observations or to provide
field-calibrated estimates of soybean aphid abundance.

The predictive MAST-Fuse model does **not** receive the latent LV states
or any intermediate variables produced by the generator. Only the
observed environmental modalities are provided to the predictive model.

---

## 1. Population dynamics

Let:

- `N(t)` = simulated aphid abundance at time `t`
- `P(t)` = effective natural-enemy population/state at time `t`

The coupled dynamics are:

\[
\frac{dN}{dt}
=
r_tN\left(1-\frac{N}{K_t}\right)
-\alpha_tNP,
\]

\[
\frac{dP}{dt}
=
\beta_tNP-\delta P.
\]

The first equation models aphid population growth under environmental
constraints and natural-enemy pressure. The second equation models the
change in the effective natural-enemy state resulting from interaction
with the aphid population and natural-enemy mortality.

The natural-enemy state is an effective simulation variable and should
not be interpreted as a directly measured predator abundance.

---

## 2. Fixed simulation parameters

The generator uses the following fixed parameters:

| Parameter | Value | Description |
|---|---:|---|
| `r_max` | 0.32 day^-1 | Maximum intrinsic aphid growth rate |
| `K_min` | 8 | Minimum carrying capacity |
| `K_max` | 60 | Maximum carrying capacity |
| `alpha_base` | 0.0045 | Baseline aphid--enemy interaction coefficient |
| `beta_base` | 0.0020 | Baseline predator conversion coefficient |
| `delta` | 0.035 day^-1 | Natural-enemy mortality rate |
| `N0_base` | 0.25 | Baseline aphid initial state |
| `P0_base` | 0.60 | Baseline natural-enemy initial state |

These values are fixed simulation parameters. They are not claimed to be
experimentally calibrated field estimates.

The maximum intrinsic growth rate is selected to provide a biologically
plausible scale for the simulated aphid dynamics. The remaining
parameters define the controlled simulation environment.

---

## 3. Environmental forcing

The LV dynamics are driven by observed environmental conditions.

The generator uses:

- air temperature,
- relative humidity,
- NDVI,
- EVI,
- NDWI,
- soil variables.

Environmental suitability functions are normalized to bounded values
between 0 and 1.

### 3.1 Temperature suitability

Temperature suitability is defined using a bounded triangular function:

\[
f_T(T)=
\begin{cases}
0, & T\leq5,\\
\dfrac{T-5}{25-5}, & 5<T<25,\\
\dfrac{35-T}{35-25}, & 25\leq T<35,\\
0, & T\geq35.
\end{cases}
\]

The three temperature points are:

- lower threshold = 5°C,
- optimum = 25°C,
- upper threshold = 35°C.

These values are used as a simplified biologically informed suitability
function. Published observations report a lower developmental threshold
near 5.44°C, favorable population development around 25--27.8°C, and an
upper threshold near 35°C.

The triangular function is therefore a simplified response function and
not a fitted empirical temperature-response curve.

---

## 4. Humidity suitability

Humidity suitability is modeled as a Gaussian-shaped function centered
at 70% relative humidity:

\[
f_H(H)=
\exp\left[
-\frac{1}{2}
\left(\frac{H-70}{20}\right)^2
\right].
\]

The resulting value is clipped to `[0,1]`.

---

## 5. Vegetation suitability

Vegetation suitability combines three spectral indices:

\[
f_V =
0.50f_{\mathrm{NDVI}}
+
0.30f_{\mathrm{EVI}}
+
0.20f_{\mathrm{NDWI}}.
\]

NDVI and EVI are linearly normalized to `[0,1]`.

NDWI is normalized from `[-1,1]` to `[0,1]`.

The resulting vegetation suitability is clipped to `[0,1]`.

---

## 6. Soil suitability

The generator uses a bounded soil-suitability function `f_S` based on
the available soil variables.

The exact implementation in the released generator is the authoritative
definition of `f_S`.

The soil suitability contributes to the environmental carrying capacity
rather than directly being used as a predictive target.

---

## 7. Environmental modulation of the LV parameters

The intrinsic aphid growth rate is:

\[
r_t =
r_{\max}
\left(
0.60f_T(T_t)
+0.20f_H(H_t)
+0.20f_V(t)
\right).
\]

The carrying capacity is:

\[
K_t =
K_{\min}
+
(K_{\max}-K_{\min})
\left(
0.60f_V(t)
+0.25f_S(t)
+0.15f_H(H_t)
\right).
\]

The aphid--enemy interaction coefficient is:

\[
\alpha_t =
\alpha_{\mathrm{base}}
\left(0.60+0.40f_V(t)\right).
\]

The predator conversion coefficient is:

\[
\beta_t =
\beta_{\mathrm{base}}
\left(0.75+0.25f_V(t)\right).
\]

Natural-enemy mortality remains constant:

\[
\delta=0.035~\mathrm{day}^{-1}.
\]

Thus, environmental observations affect both aphid growth and the
population capacity of the system, while vegetation condition also
modulates the interaction terms.

---

## 8. Initial conditions

The baseline simulation states are:

\[
N_0^{base}=0.25,
\qquad
P_0^{base}=0.60.
\]

To introduce deterministic site/year heterogeneity, these baseline
values are modulated using site and year identifiers:

\[
N_0^{(s,y)}
=
0.25
\left(0.90+0.01H_s+0.02Y_y\right),
\]

\[
P_0^{(s)}
=
0.60
\left(0.90+0.005H_s\right).
\]

The site hash and year terms are deterministic. Therefore, rerunning the
generator with the same inputs produces the same initial conditions.

These initialization values are simulation anchors and are not
field-calibrated population measurements.

---

## 9. Temporal integration

Environmental forcing is held constant over each 8-hour observation
interval.

The observation interval is:

\[
\Delta t = \frac{1}{3}\ {\rm day}.
\]

Each interval is divided into eight equal integration steps:

\[
\Delta t_{\mathrm{sub}}
=
\frac{1}{24}\ {\rm day},
\]

corresponding to one hour.

The LV equations are integrated using an explicit Euler method.

After every substep:

- aphid abundance is constrained to be non-negative;
- natural-enemy abundance is constrained to be non-negative.

The generator also applies numerical upper bounds:

- aphid abundance: `10 K_t`;
- natural-enemy state: `1000`.

These upper bounds are numerical safeguards for stable simulation and
are not intended as additional biological constraints.

---

## 10. Observation noise

No random observation noise is added to the generated target:

```text
ADD_OBSERVATION_NOISE = False
