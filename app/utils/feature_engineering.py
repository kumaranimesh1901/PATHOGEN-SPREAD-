"""
Feature engineering for the PATHOGEN-SPREAD forecasting system.
Computes lag features, rolling averages, growth rates, time features,
and direct-shift targets for 6-model CatBoost training.
"""

import numpy as np
import pandas as pd

# Module-level constant: features used by all 6 CatBoost models
# Removed noise features: new_vaccinations (corr 0.001), mobility_index,
# temperature, humidity (corr < 0.03), population_density (constant per state)
FEATURE_COLS = [
    "lag_cases_1", "lag_cases_3", "lag_cases_7", "lag_cases_14",
    "lag_deaths_1", "lag_deaths_7",
    "roll7_cases", "roll14_cases", "roll7_deaths",
    "growth_rate", "day_of_week", "month", "week",
    "tests_performed", "positive_rate",
    "hospitalized_patients", "icu_patients",
    # Extended lags and rolling windows
    "lag_cases_21", "lag_cases_30",
    "roll21_cases", "roll30_cases",
    "case_acceleration",
    # High-signal engineered features
    "log_hospitalized", "log_icu", "log_tests",
    "hosp_icu_ratio", "test_positivity_smooth",
    "log_cases_x_tests",
    # Structural break feature
    "year",
    # Death-specific: infection-to-death clinical delay (10–21 days)
    "lag_cases_10",
    "roll14_cases_past",
    "cfr_trend",
]


def engineer_features(df: pd.DataFrame, state_name: str) -> pd.DataFrame:
    """
    Build all engineered features for a single state.

    Steps:
      1. Filter by state
      2. Sort by date
      3. Log-transform targets (log1p)
      4. Lag features on log values (1, 3, 7, 14, 21, 30)
      5. Rolling averages (shift(1) first to prevent leakage)
      6. Growth rate (7-day pct_change)
      6b. Case acceleration, CFR, positivity smoothing
      7. High-signal engineered features (log transforms, ratios, interactions)
      8. Time features (day_of_week, month, ISO week, year)
      9. Log-difference targets for 7d / 14d / 30d horizons
      10. Drop NaN rows, reset index

    Args:
        df: Full DataFrame (may contain multiple states).
        state_name: The state to filter for.

    Returns:
        Engineered DataFrame ready for training or prediction.
    """
    # 1. Filter
    df = df[df["state"] == state_name].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # 2. Sort by date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 3. Log-transform targets
    df["log_cases"] = np.log1p(df["new_cases"].astype(float))
    df["log_deaths"] = np.log1p(df["new_deaths"].astype(float))

    # 4. Lag features on log values
    for lag in [1, 3, 7, 14]:
        df[f"lag_cases_{lag}"] = df["log_cases"].shift(lag)
        df[f"lag_deaths_{lag}"] = df["log_deaths"].shift(lag)

    # 4b. Extended lags (21, 30 days)
    for lag in [21, 30]:
        df[f"lag_cases_{lag}"] = df["log_cases"].shift(lag)
        df[f"lag_deaths_{lag}"] = df["log_deaths"].shift(lag)

    # 4c. Death-specific clinical delay lags (infection-to-death: 10–21 days)
    df["lag_cases_10"] = df["log_cases"].shift(10)

    # 5. Rolling averages (shift(1) first to prevent leakage)
    df["roll7_cases"] = df["log_cases"].shift(1).rolling(7).mean()
    df["roll14_cases"] = df["log_cases"].shift(1).rolling(14).mean()
    df["roll7_deaths"] = df["log_deaths"].shift(1).rolling(7).mean()

    # 5b. Extended rolling averages
    df["roll21_cases"] = df["log_cases"].shift(1).rolling(21).mean()
    df["roll30_cases"] = df["log_cases"].shift(1).rolling(30).mean()

    # 5c. Death-specific: rolling average of cases from 10–24 days ago
    df["roll14_cases_past"] = df["log_cases"].shift(10).rolling(14).mean()

    # 6. Growth rate (7-day percentage change on raw new_cases)
    df["growth_rate"] = df["new_cases"].pct_change(7)
    df["growth_rate"] = df["growth_rate"].replace([np.inf, -np.inf], 0).fillna(0)

    # 6b. Case acceleration (second derivative of new_cases)
    df["case_acceleration"] = df["new_cases"].diff().diff().fillna(0)

    # 6c. CFR trend: 14-day rolling mean of death-to-case ratio (shifted to prevent leakage)
    df["cfr_trend"] = (
        (df["new_deaths"] / (df["new_cases"] + 1))
        .rolling(14).mean()
        .shift(1)
        .fillna(0)
    )

    # 7. High-signal engineered features
    # Log transforms of correlated raw features (correlation > 0.92)
    df["log_hospitalized"] = np.log1p(df["hospitalized_patients"].astype(float))
    df["log_icu"] = np.log1p(df["icu_patients"].astype(float))
    df["log_tests"] = np.log1p(df["tests_performed"].astype(float))

    # ICU-to-hospital ratio (severity signal)
    df["hosp_icu_ratio"] = df["icu_patients"] / (df["hospitalized_patients"] + 1)

    # Fix positive_rate: zeros are reporting artifacts, not true zeros
    pr = df["positive_rate"].copy()
    pr = pr.replace(0, np.nan).ffill()
    df["positive_rate"] = pr.fillna(0)
    df["test_positivity_smooth"] = df["positive_rate"].shift(1).rolling(7).mean()

    # Interaction term: recent case level × testing capacity
    df["log_cases_x_tests"] = df["lag_cases_1"] * df["log_tests"]

    # 8. Time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year

    # 9. Log-difference targets (stationary changes, not absolute levels)
    # Predicting change from current value rather than the future level itself
    df["target_cases_7d"] = df["log_cases"].shift(-7) - df["log_cases"]
    df["target_cases_14d"] = df["log_cases"].shift(-14) - df["log_cases"]
    df["target_cases_30d"] = df["log_cases"].shift(-30) - df["log_cases"]
    df["target_deaths_7d"] = df["log_deaths"].shift(-7) - df["log_deaths"]
    df["target_deaths_14d"] = df["log_deaths"].shift(-14) - df["log_deaths"]
    df["target_deaths_30d"] = df["log_deaths"].shift(-30) - df["log_deaths"]

    # 10. Drop rows with NaN in feature columns, reset index
    required_cols = FEATURE_COLS + [
        "target_cases_7d", "target_cases_14d", "target_cases_30d",
        "target_deaths_7d", "target_deaths_14d", "target_deaths_30d",
    ]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required).reset_index(drop=True)

    return df
