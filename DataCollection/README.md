# MAST-Fuse: Multimodal Pest Dataset Generation with Lotka–Volterra Dynamics

**MAST-Fuse** is a reproducible multimodal dataset-generation pipeline for machine-learning-based pest population prediction. The framework integrates heterogeneous environmental observations with an ecologically constrained **Lotka–Volterra (LV)** dynamical system to generate a synthetic aphid-population target.

The dataset is designed for research in:

* Multimodal deep learning
* Pest population forecasting
* Environmental intelligence
* Agricultural AI
* Time-series regression
* Explainable AI
* Multimodal and architecture ablation studies

> **Important:** The aphid population is a **synthetic simulation target**, generated using an environmentally driven Lotka–Volterra model. It is not a collection of field-measured aphid abundances and should not be interpreted as field-calibrated ground truth.

---

## 1. Overview

MAST-Fuse combines four environmental modalities collected or derived from multiple growing seasons and geographic locations:

| Modality     | Source/Type                 | Temporal Nature | Features |
| ------------ | --------------------------- | --------------: | -------: |
| Weather      | Environmental observations  |         Dynamic |       21 |
| Vegetation   | Sentinel-2 spectral indices |         Dynamic |        3 |
| Dynamic Soil | Sentinel-2 spectral indices |         Dynamic |        4 |
| Static Soil  | Soil database               |          Static |       28 |
| Aphid Target | Lotka–Volterra simulation   |         Dynamic |        1 |

The environmental modalities are temporally synchronized and used to drive an ecological simulation that produces the aphid-population target.

The resulting learning problem is formulated as a next-step forecasting task:

$$
X_{t-L+1:t} \rightarrow y_{t+1},
$$

where:

* \(X\) denotes the multimodal environmental observations,
* \(L\) is the temporal sequence length,
* \(y_{t+1}\) is the predicted aphid population at the next time step.

The predictive model receives **only the observed environmental modalities**. It does not receive the latent LV states, environmental suitability functions, or intermediate variables used by the target generator.

---

# 2. Geographic Coverage

The current dataset covers three counties in Iowa, USA:

| Site          | Latitude | Longitude |
| ------------- | -------: | --------: |
| Jasper County |  41.6932 |  -93.0538 |
| Polk County   |  41.6278 |  -93.5815 |
| Story County  |  42.0347 |  -93.5813 |

These locations represent agricultural areas within the Midwestern United States and provide geographic heterogeneity for multimodal environmental modeling.

---

# 3. Temporal Coverage

The current release covers five growing seasons:

```text
2021
2022
2023
2024
2025
```

Environmental observations are synchronized at an **8-hour temporal resolution**.

The LV simulation uses each 8-hour interval as one environmental forcing interval.

---

# 4. Dataset Architecture

The MAST-Fuse generation pipeline can be summarized as:

```text
                 ┌─────────────────────┐
                 │ Weather Observations│
                 │      21 features    │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Vegetation Indices  │
                 │ NDVI / EVI / NDWI   │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ Dynamic Soil Indices│
                 │      4 features     │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │   Static Soil Data  │
                 │      28 features    │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Environmental Suitability   │
              │                             │
              │ Temperature                 │
              │ Humidity                    │
              │ Vegetation                  │
              │ Soil                        │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Environmentally Modulated   │
              │ Lotka–Volterra Dynamics     │
              │                             │
              │ Aphid state N(t)             │
              │ Natural-enemy state P(t)    │
              └──────────────┬──────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Synthetic Aphid      │
                  │ Population Target    │
                  └──────────────────────┘
```

---

# 5. Lotka–Volterra Target Generation

## 5.1 Motivation

Consistently sampled field measurements of soybean aphid abundance were not available across all locations and seasons at the required 8-hour temporal resolution.

Therefore, MAST-Fuse uses an ecologically constrained Lotka–Volterra model to construct a controlled and reproducible synthetic target.

The purpose of this simulation is to provide a benchmark for studying whether machine-learning models can learn nonlinear relationships between environmental variables and pest population dynamics.

The generated target should **not** be interpreted as:

* field-measured aphid abundance;
* a field-calibrated population estimate;
* a replacement for entomological observations;
* evidence that the simulated population exactly reproduces real soybean aphid dynamics.

---

## 5.2 Population Dynamics

Let:

* \(N(t)\) = simulated aphid abundance,
* \(P(t)\) = effective natural-enemy state.

The coupled dynamics are:

$$
\frac{dN}{dt}
=
r_tN\left(1-\frac{N}{K_t}\right)
-\alpha_tNP,
$$

$$
\frac{dP}{dt}
=
\beta_tNP-\delta P.
$$

The aphid equation models population growth under environmental constraints and natural-enemy pressure.

The second equation models the evolution of an effective natural-enemy state through interaction with the aphid population and natural-enemy mortality.

> The natural-enemy state \(P(t)\) is a latent simulation variable and does not represent directly measured predator abundance.

---

# 6. Simulation Parameters

The generator uses the following fixed parameters:

| Parameter    |       Value | Description                                  |
| ------------ | ----------: | -------------------------------------------- |
| `r_max`      |  0.32 day⁻¹ | Maximum intrinsic aphid growth rate          |
| `K_min`      |           8 | Minimum carrying capacity                    |
| `K_max`      |          60 | Maximum carrying capacity                    |
| `alpha_base` |      0.0045 | Baseline aphid–enemy interaction coefficient |
| `beta_base`  |      0.0020 | Baseline predator conversion coefficient     |
| `delta`      | 0.035 day⁻¹ | Natural-enemy mortality rate                 |
| `N0_base`    |        0.25 | Baseline aphid initial state                 |
| `P0_base`    |        0.60 | Baseline natural-enemy initial state         |

These values are **fixed simulation parameters**. They are not claimed to be experimentally calibrated field estimates.

---

# 7. Environmental Forcing

The LV system is driven by observed environmental conditions, including:

* Air temperature
* Relative humidity
* NDVI
* EVI
* NDWI
* Soil variables

Environmental suitability functions are normalized to bounded values between 0 and 1.

---

## 7.1 Temperature Suitability

Temperature suitability is represented using a bounded triangular response function:

$$
f_T(T)=
\begin{cases}
0, & T \leq 5, \\
\dfrac{T-5}{20}, & 5 < T < 25, \\
\dfrac{35-T}{10}, & 25 \leq T < 35, \\
0, & T \geq 35.
\end{cases}
$$

The response uses:

* Lower threshold: **5°C**
* Optimum: **25°C**
* Upper threshold: **35°C**

The resulting suitability value satisfies \(f_T(T)\in[0,1]\), with maximum suitability at **25°C** and zero suitability at or below **5°C** and at or above **35°C**.

These values provide a simplified, biologically informed suitability function. The function is intentionally simplified and should not be interpreted as a fitted empirical temperature-response model.
---

## 7.2 Humidity Suitability

Humidity suitability is modeled using a Gaussian-shaped function centered at 70% relative humidity:

$$
f_H(H)=
\exp\left[
-\frac{1}{2}
\left(
\frac{H-70}{20}
\right)^2
\right].
$$

The resulting value is clipped to:

$$
f_H\in[0,1].
$$

---

## 7.3 Vegetation Suitability

Vegetation suitability combines NDVI, EVI, and NDWI:

$$
f_V =
0.50f_{\mathrm{NDVI}}
+
0.30f_{\mathrm{EVI}}
+
0.20f_{\mathrm{NDWI}}.
$$

NDVI and EVI are linearly normalized to \([0,1]\).

NDWI is transformed from \([-1,1]\) to \([0,1]\).

The final vegetation suitability is clipped to \([0,1]\).

---

## 7.4 Soil Suitability

The generator computes a bounded soil-suitability function \(f_S\) using the available soil variables.

The **released implementation is the authoritative definition** of the soil-suitability function.

Soil suitability contributes to the environmental carrying capacity and is not itself used as a target.

---

# 8. Environmental Modulation of LV Parameters

The environmental variables dynamically modify the parameters of the LV system.

### 8.1 Intrinsic Growth Rate

$$
r_t =
r_{\max}
\left(
0.60f_T(T_t)
+
0.20f_H(H_t)
+
0.20f_V(t)
\right).
$$

Thus, temperature has the largest contribution to the simulated intrinsic growth rate, followed by humidity and vegetation.

---

### 8.2 Carrying Capacity

$$
K_t =
K_{\min}
+
(K_{\max}-K_{\min})
\left(
0.60f_V(t)
+
0.25f_S(t)
+
0.15f_H(H_t)
\right).
$$

Vegetation, soil, and humidity therefore influence the maximum population capacity of the simulated system.

---

### 8.3 Aphid–Enemy Interaction

The aphid–enemy interaction coefficient is:

$$
\alpha_t =
\alpha_{\mathrm{base}}
\left(
0.60+0.40f_V(t)
\right).
$$

---

### 8.4 Natural-Enemy Conversion

The predator conversion coefficient is:

$$
\beta_t =
\beta_{\mathrm{base}}
\left(
0.75+0.25f_V(t)
\right).
$$

---

### 8.5 Natural-Enemy Mortality

The natural-enemy mortality rate remains constant:

$$
\delta=0.035~\mathrm{day}^{-1}.
$$

Consequently, the environmental observations affect both aphid growth and population capacity, while vegetation additionally modulates the interaction terms.

---

# 9. Initial Conditions

The baseline simulation states are:

$$
N_0^{base}=0.25,
\qquad
P_0^{base}=0.60.
$$

To introduce deterministic site/year heterogeneity, these baseline values are modulated using site and year identifiers:

$$
N_0^{(s,y)}
=
0.25
\left(
0.90+0.01H_s+0.02Y_y
\right),
$$

$$
P_0^{(s)}
=
0.60
\left(
0.90+0.005H_s
\right).
$$

The site hash and year terms are deterministic.

Therefore, running the generator again with the same input data and configuration produces the same initialization values.

These values are simulation anchors and should not be interpreted as field-calibrated initial aphid or predator populations.

---

# 10. Temporal Integration

Environmental forcing is assumed to remain constant during each 8-hour observation interval.

The observation interval is:

$$
\Delta t=\frac{1}{3}\text{ day}.
$$

Each 8-hour interval is divided into eight one-hour integration steps:

$$
\Delta t_{\mathrm{sub}}
=
\frac{1}{24}\text{ day}.
$$

The LV equations are integrated using an explicit Euler scheme.

For each integration substep:

1. Environmental forcing is evaluated.
2. \(r_t\), \(K_t\), \(\alpha_t\), and \(\beta_t\) are computed.
3. The LV differential equations are evaluated.
4. Aphid abundance is updated.
5. Natural-enemy abundance is updated.
6. Negative states are clipped to zero.

---

# 11. Numerical Stability Constraints

The generator applies numerical upper bounds after each integration step.

| State               | Numerical bound |
| ------------------- | --------------: |
| Aphid abundance     |       \(10K_t\) |
| Natural-enemy state |            1000 |

These limits are included as **numerical safeguards** to prevent unstable trajectories.

They are not intended to represent additional biological constraints.

---

# 12. Observation Noise

No random observation noise is added to the generated aphid target.

```text
ADD_OBSERVATION_NOISE = False
```

Consequently, the generated target is deterministic given the environmental inputs, model parameters, initial conditions, and implementation.

This design facilitates:

* Reproducible experiments
* Controlled benchmarking
* Architecture comparison
* Modality ablation
* Sensitivity analysis
* Explainability experiments

---

# 13. Reproducibility

The MAST-Fuse generation process is designed to be deterministic.

Given identical:

* environmental input data,
* geographic identifiers,
* year identifiers,
* simulation parameters,
* preprocessing configuration, and
* software implementation,

the same synthetic target should be generated.

The deterministic initialization mechanism introduces controlled site/year heterogeneity without requiring random target generation.

For reproducible experiments, we recommend recording:

```text
Python version
Package versions
Input-data version
Generator version/commit
Simulation parameters
Preprocessing configuration
```

---

# 14. Data Leakage Considerations

A key design principle of MAST-Fuse is that the variables used internally to generate the target should not be provided to the predictive model unless they are explicitly part of the observed modality set.

The predictive task should follow:

```text
Observed environmental modalities
                │
                ▼
        Temporal sequence
                │
                ▼
        Prediction model
                │
                ▼
       Predicted aphid target
```

The following variables are **latent generator variables** and should not be directly supplied to the predictive model:

* \(N(t)\)
* \(P(t)\)
* \(r_t\)
* \(K_t\)
* \(\alpha_t\)
* \(\beta_t\)
* \(f_T\)
* \(f_H\)
* \(f_V\)
* \(f_S\)

This separation allows the benchmark to evaluate whether a machine-learning model can infer pest dynamics from the environmental observations rather than reproducing the simulator through direct access to its latent states.

---

# 15. Recommended Machine Learning Setup

The dataset can be used for sequence-to-one forecasting:

$$
[X_{t-L+1},\ldots,X_t]\rightarrow y_{t+1}.
$$

Possible model architectures include:

### Recurrent models

* LSTM
* GRU
* BiLSTM

### Temporal convolution models

* Temporal CNN
* TCN

### Transformer-based models

* Temporal Transformer
* Multimodal Transformer
* Cross-modal Attention Transformer

### Multimodal architectures

* Early fusion
* Intermediate fusion
* Late fusion
* Cross-attention fusion
* Gated multimodal fusion

### Explainability

The dataset can additionally support:

* SHAP
* Integrated Gradients
* Attention analysis
* Modality ablation
* Feature importance analysis
* Temporal importance analysis

---

# 16. Suggested Dataset Split

To avoid temporal leakage, experiments should preferably use chronological splits.

For example:

```text
Training:   2021–2023
Validation: 2024
Testing:    2025
```

Alternatively, researchers may perform geographic generalization experiments by holding out one county during training.

For example:

```text
Training:   Jasper + Polk
Testing:    Story
```

The exact split should be reported clearly in experimental results.

---

# 17. Possible Research Tasks

MAST-Fuse can support several experimental settings.

### 17.1 Pest Population Forecasting

Predict the next-step aphid population:

$$
X_{t-L+1:t}\rightarrow N_{t+1}.
$$

### 17.2 Long-Horizon Forecasting

Predict:

$$
N_{t+h}
$$

for \(h>1\).

### 17.3 Multimodal Ablation

Compare:

```text
Weather only
Vegetation only
Soil only
Weather + Vegetation
Weather + Soil
Vegetation + Soil
All modalities
```

### 17.4 Geographic Generalization

Train on selected counties and evaluate on an unseen county.

### 17.5 Temporal Generalization

Train on earlier growing seasons and evaluate on a future season.

### 17.6 Explainable AI

Investigate which environmental variables and modalities contribute most strongly to pest-population predictions.

---

# 18. Evaluation Metrics

For regression-based forecasting, recommended metrics include:

### Mean Absolute Error

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat y_i|.
$$

### Root Mean Squared Error

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
}.
$$

### Coefficient of Determination

$$
R^2 =
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
$$

Depending on the experimental objective, researchers may additionally report:

* MAPE
* SMAPE
* Pearson correlation
* Spearman correlation
* Forecasting horizon error
* Seasonal error
* Site-specific error

---

# 19. Project Structure

A recommended repository organization is:

```text
MAST-Fuse/
│
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── preprocessing/
│   ├── environmental/
│   ├── lv_model/
│   ├── target_generation/
│   └── utils/
│
├── scripts/
│   ├── preprocess_data.py
│   ├── generate_lv_target.py
│   └── build_dataset.py
│
├── configs/
│   └── default.yaml
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── target_analysis.ipynb
│
├── experiments/
│   └── ...
│
└── results/
    └── ...
```

Adapt the structure to the actual files included in the repository.

---

# 20. Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR-USERNAME>/MAST-Fuse.git
cd MAST-Fuse
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Linux/macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If an `environment.yml` file is provided:

```bash
conda env create -f environment.yml
conda activate mast-fuse
```

---

# 21. Dataset Generation

After preparing the environmental input data according to the repository instructions, run:

```bash
python scripts/generate_lv_target.py
```

or, if the complete pipeline is provided:

```bash
python scripts/build_dataset.py
```

The generated dataset should contain synchronized environmental features together with the synthetic aphid-population target.

> The exact commands should be updated to match the executable scripts included in the repository.

---

# 22. Output Format

A typical generated record may contain fields similar to:

```text
timestamp
county
latitude
longitude
year

weather_*
ndvi
evi
ndwi

soil_dynamic_*
soil_static_*

aphid_population
```

The exact column names depend on the released preprocessing pipeline.

Researchers should preserve the distinction between:

**Observed inputs**

```text
Weather
Vegetation
Dynamic soil
Static soil
```

and

**Generated target**

```text
Aphid population
```

---

# 23. Important Interpretation of the Synthetic Target

The LV-generated target has been introduced to provide a controlled learning benchmark in situations where sufficiently dense field observations are unavailable.

Therefore, results obtained on MAST-Fuse should primarily be interpreted as evidence about:

> **A model's ability to learn environmentally driven nonlinear temporal dynamics under the specified simulation mechanism.**

They should not be interpreted directly as evidence of real-world soybean aphid forecasting performance.

Real-world deployment would require validation against independently collected field observations.

---

# 24. Limitations

The current benchmark has several important limitations.

### Synthetic target

The aphid population is generated by a mechanistic simulation rather than direct field observations.

### Simplified ecological dynamics

The LV system represents a simplified interaction between aphids and an effective natural-enemy state.

### Parameter assumptions

The simulation parameters are fixed assumptions and are not claimed to represent universally calibrated biological parameters.

### Simplified environmental responses

Temperature, humidity, vegetation, and soil suitability functions are simplified mathematical representations rather than fully fitted ecological response functions.

### Missing ecological factors

The current formulation does not explicitly model all factors that can affect aphid dynamics, such as:

* precipitation effects,
* wind,
* pesticide application,
* cultivar resistance,
* crop phenology,
* migration,
* dispersal,
* disease,
* detailed predator communities,
* landscape structure.

### Field validation

Independent field observations are required to evaluate how well the simulated target represents real pest population dynamics.

---

# 25. Recommended Validation Strategy

For future development, the following validation strategy is recommended:

```text
MAST-Fuse synthetic benchmark
            │
            ▼
Machine-learning model development
            │
            ▼
Independent field observations
            │
            ▼
Real-world validation
```

A strong future direction is to calibrate or validate the LV parameters using independently collected soybean aphid observations while preserving a separate test set for unbiased evaluation.

---

