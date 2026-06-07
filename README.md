# 🦠 PATHOGEN-SPREAD: Hybrid CatBoost + SEIR Outbreak Forecasting

A **state-level** disease outbreak prediction system combining **CatBoost ML** and
**SEIR epidemiological simulation**. Runs entirely from the terminal with a single
command — no web server required.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Engine** | CatBoost (Gradient Boosting) |
| **Epidemiological Model** | SEIR ODE (SciPy `solve_ivp`) |
| **Database** | SQLite (built-in `sqlite3`) |
| **Visualization** | Matplotlib (dark theme, PNG output) |
| **Dataset** | India COVID-19 (10 states, 2020–2024) |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

That's it. On the first run the script will:

1. Load the CSV dataset into a local SQLite database
2. Train 6 CatBoost models (cases/deaths × 7d/14d/30d)
3. Run the forecast for the configured state
4. Print results to the terminal
5. Save all charts to `outputs/`

### What You'll See in the Terminal

```
==================================================
  PATHOGEN-SPREAD — Outbreak Forecasting
  State: Tamil Nadu
  Date:  2026-06-07
==================================================

[1/5] Loading dataset into database...
[2/5] Checking / Training CatBoost models...
[3/5] Running forecast for Tamil Nadu...
[4/5] Generating charts...
[5/5] Charts saved:
      outputs/chart6_summary.png  ← main dashboard
```

---

## Configuration

Open `run.py` and edit these variables at the top:

```python
STATE_NAME        = "Tamil Nadu"    # which state to forecast
CURRENT_CASES     = 1200            # today's case count
CURRENT_DEATHS    = 18              # today's death count
POPULATION        = 72000000        # state population
TRANSMISSION_RATE = 1.5             # beta (transmission rate)
RECOVERY_RATE     = 0.1             # gamma (recovery rate)
MORTALITY_RATE    = 0.02            # mu (mortality rate)
FORCE_RETRAIN     = True            # set False to skip training on reruns
```

**Available states:** Delhi, Gujarat, Karnataka, Kerala, Maharashtra, Rajasthan,
Tamil Nadu, Telangana, Uttar Pradesh, West Bengal

---

## Output Charts

All charts are saved to the `outputs/` folder with a dark professional theme:

| File | Description |
|------|-------------|
| `chart1_forecast.png` | 30-day cases & deaths forecast curve |
| `chart2_seir.png` | SEIR epidemic curve (S, E, I, R over 180 days) |
| `chart3_features.png` | CatBoost feature importance (horizontal bar chart) |
| `chart4_risk.png` | Semicircular risk gauge (Low / Medium / High) |
| `chart5_metrics.png` | Training metrics — MAE, RMSE, R² per target |
| **`chart6_summary.png`** | **Full dashboard combining all charts** |

---

## Project Structure

```
PATHOGEN-SPREAD/
├── run.py                          # ← Single entry point (python run.py)
├── app/
│   ├── models/
│   │   ├── catboost_model.py       # CatBoost train (6 models) + predict
│   │   ├── seir_model.py           # SEIR ODE simulation (SciPy)
│   │   └── hybrid_model.py         # Engine selector (CatBoost/SEIR/Hybrid)
│   ├── database/
│   │   └── db.py                   # SQLite schema, CSV import, queries
│   ├── utils/
│   │   ├── feature_engineering.py  # Lag features, rolling averages, log1p
│   │   ├── risk_calculator.py      # Risk category + numeric score
│   │   └── explainability.py       # CatBoost feature importance
│   └── data/
│       └── india_covid_catboost_dataset.csv  # Source dataset (18,270 rows)
├── models_saved/                   # Auto-created: trained .cbm model files
├── outputs/                        # Auto-created: chart PNGs
├── requirements.txt
└── README.md
```

---

## Dataset

**File:** `app/data/india_covid_catboost_dataset.csv` — 18,270 rows (1,827 per state × 10 states)

| Column | Type | Description |
|--------|------|-------------|
| `date` | TEXT | Date (YYYY-MM-DD) |
| `state` | TEXT | Indian state name |
| `new_cases` | INT | Daily new confirmed cases |
| `new_deaths` | INT | Daily new deaths |
| `new_vaccinations` | INT | Daily vaccinations administered |
| `tests_performed` | INT | Daily tests conducted |
| `positive_rate` | REAL | Test positivity rate |
| `hospitalized_patients` | INT | Currently hospitalized |
| `icu_patients` | INT | Currently in ICU |
| `mobility_index` | REAL | Population mobility metric |
| `temperature` | REAL | Temperature (°C) |
| `humidity` | REAL | Relative humidity (%) |
| `population_density` | INT | Density per sq km |

---

## Model Architecture

### Hybrid Engine Selection

The system automatically selects the best engine based on available data:

| Historical Rows | Engine | Rationale |
|-----------------|--------|-----------|
| ≥ 60 | **CatBoost** | Sufficient data for ML-based forecasting |
| 30–59 | **Hybrid** | Weighted blend of CatBoost + SEIR |
| < 30 | **SEIR** | Falls back to epidemiological simulation |

### CatBoost — 6 Models per State

| Target | Predicts |
|--------|----------|
| `cases_7d` | Cumulative cases in 7 days |
| `cases_14d` | Cumulative cases in 14 days |
| `cases_30d` | Cumulative cases in 30 days |
| `deaths_7d` | Cumulative deaths in 7 days |
| `deaths_14d` | Cumulative deaths in 14 days |
| `deaths_30d` | Cumulative deaths in 30 days |

- **Transform:** `log1p()` on targets during training, `expm1()` on predictions
- **Validation:** TimeSeriesSplit with 5 folds
- **22 engineered features:** lag values, rolling averages, growth rates, temporal features, and raw epidemiological indicators

**Hyperparameters:**
```python
iterations=300, depth=5, learning_rate=0.05,
l2_leaf_reg=3, min_data_in_leaf=5, loss_function='RMSE'
```

### SEIR Compartmental Model

```
dS/dt = -β·S·I/N          (Susceptible → Exposed)
dE/dt =  β·S·I/N - σ·E    (Exposed → Infected)
dI/dt =  σ·E - γ·I - μ·I  (Infected → Recovered/Deceased)
dR/dt =  γ·I              (Recovered)
dD/dt =  μ·I              (Deceased)
```

- σ = 1/5 (fixed 5-day incubation period)
- Solved with RK45 method over 180 days

### Risk Assessment

```
Risk Score = (growth_rate × 0.6) + (mortality_rate × 100 × 0.4)
```

| Score Range | Category |
|-------------|----------|
| Growth < 5% AND Mortality < 1% | 🟢 **Low** |
| Growth 5–15% OR Mortality 1–3% | 🟡 **Medium** |
| Growth > 15% OR Mortality > 3% | 🔴 **High** |

---

## Evaluation Metrics

Training reports these metrics per target (computed on original scale after `expm1`):

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **R² Score** | Coefficient of determination |
| **Relative MAE%** | MAE / mean(|y|) × 100 |

---

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies
- No GPU required — runs on CPU

---

## License

This project is developed for educational and research purposes.
