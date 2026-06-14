"""
Transfer-learning fine-tuner for the PATHOGEN-SPREAD platform.
==============================================================
Adapts an existing COVID-trained CatBoost model to a new pathogen using
warm-start (``init_model``).  Designed for the 14–59 day data window
where a full retrain is premature but enough signal exists to shift
the model's learned patterns.

Usage
-----
>>> from app.models.transfer_model import fine_tune_from_covid
>>> metrics = fine_tune_from_covid(df, "Karnataka", "cases_7d",
...     "models_saved/catboost_Karnataka_cases_7d.cbm")
>>> print(metrics)
{"mae": 42.3, "r2_score": 0.81, "model_path": "models_saved/transfer_..."}
"""

import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from app.database.db import get_state_data
from app.utils.feature_engineering import engineer_features, FEATURE_COLS

# Directory shared with the main CatBoost pipeline
MODELS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models_saved",
    )
)

# The 6 targets mirrored from catboost_model.py for convenience
TARGETS = [
    ("target_cases_7d", "cases_7d"),
    ("target_cases_14d", "cases_14d"),
    ("target_cases_30d", "cases_30d"),
    ("target_deaths_7d", "deaths_7d"),
    ("target_deaths_14d", "deaths_14d"),
    ("target_deaths_30d", "deaths_30d"),
]

# Reduced learning rate for fine-tuning — prevents catastrophic forgetting
FINE_TUNE_PARAMS = dict(
    iterations=300,
    depth=5,
    learning_rate=0.008,
    l2_leaf_reg=10,
    min_data_in_leaf=5,
    subsample=0.8,
    colsample_bylevel=0.8,
    loss_function="RMSE",
    early_stopping_rounds=50,
    verbose=False,
)


# ──────────────────────────────────────────────────────────
# Custom exception
# ──────────────────────────────────────────────────────────

class InsufficientDataError(ValueError):
    """Raised when the new-disease dataset is too small for fine-tuning."""
    pass


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _transfer_model_path(disease_name: str, state: str, target: str) -> str:
    """Return the save path for a fine-tuned transfer model."""
    return os.path.join(
        MODELS_DIR,
        f"transfer_{disease_name}_{state}_{target}.cbm",
    )


def _fill_missing_from_covid(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Fill columns that may be absent in new-disease data with COVID
    state-level historical averages.

    Missing columns are added and zero-valued cells in environmental
    columns are back-filled so that ``engineer_features()`` has the
    inputs it expects.
    """
    covid_df = get_state_data(state)
    if covid_df.empty:
        return df

    fill_cols = {
        "mobility_index": covid_df["mobility_index"].mean(),
        "temperature": covid_df["temperature"].mean(),
        "humidity": covid_df["humidity"].mean(),
        "population_density": int(covid_df["population_density"].mode().iloc[0])
        if not covid_df["population_density"].mode().empty
        else 1000,
        "new_vaccinations": 0,
    }

    df = df.copy()
    for col, default in fill_cols.items():
        if col not in df.columns:
            df[col] = default
        else:
            # Replace zeros / NaN with the COVID average
            mask = df[col].isna() | (df[col] == 0)
            df.loc[mask, col] = default

    return df


# ──────────────────────────────────────────────────────────
# Core API
# ──────────────────────────────────────────────────────────

def fine_tune_from_covid(
    new_disease_df: pd.DataFrame,
    state: str,
    target: str,
    base_model_path: str,
) -> dict:
    """Fine-tune a COVID-trained CatBoost model on new-disease data.

    Parameters
    ----------
    new_disease_df : pd.DataFrame
        Raw ingested data for the new disease (at least 14 rows).
        Must contain columns compatible with ``engineer_features()``.
    state : str
        State / region name.
    target : str
        Target short-name — one of ``"cases_7d"``, ``"cases_14d"``,
        ``"cases_30d"``, ``"deaths_7d"``, ``"deaths_14d"``,
        ``"deaths_30d"``.
    base_model_path : str
        Path to the saved COVID CatBoost ``.cbm`` model file.

    Returns
    -------
    dict
        ``{"mae": float, "r2_score": float, "model_path": str}``

    Raises
    ------
    InsufficientDataError
        If ``new_disease_df`` has fewer than 14 rows.
    FileNotFoundError
        If *base_model_path* does not exist.
    """
    if len(new_disease_df) < 14:
        raise InsufficientDataError(
            f"Need at least 14 days of data for fine-tuning. "
            f"Use SEIR cold start instead. (got {len(new_disease_df)} rows)"
        )

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base COVID model not found at '{base_model_path}'"
        )

    # ── Load base model ─────────────────────────────────────
    base_model = CatBoostRegressor()
    base_model.load_model(base_model_path)

    # ── Fill missing environmental columns ──────────────────
    df_filled = _fill_missing_from_covid(new_disease_df, state)

    # ── Engineer features ───────────────────────────────────
    # engineer_features expects a "state" column for filtering
    if "state" not in df_filled.columns:
        df_filled["state"] = state

    df_eng = engineer_features(df_filled, state)
    if len(df_eng) < 14:
        raise InsufficientDataError(
            f"After feature engineering only {len(df_eng)} rows remain. "
            f"Need at least 14. Use SEIR cold start instead."
        )

    # ── Resolve target column name ──────────────────────────
    target_col = f"target_{target}"
    if target_col not in df_eng.columns:
        raise ValueError(
            f"Target column '{target_col}' not found after feature engineering. "
            f"Available: {[c for c in df_eng.columns if c.startswith('target_')]}"
        )

    df_eng = df_eng.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df_eng) < 14:
        raise InsufficientDataError(
            f"Only {len(df_eng)} non-NaN target rows for '{target}'. "
            f"Need at least 14."
        )

    # ── Train / eval split — last 14 days held out ──────────
    split_idx = len(df_eng) - 14
    df_train = df_eng.iloc[:split_idx]
    df_eval = df_eng.iloc[split_idx:]

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[target_col].values

    X_eval = df_eval[FEATURE_COLS]
    y_eval = df_eval[target_col].values

    base_col = "log_cases" if "cases" in target else "log_deaths"

    # ── Fine-tune with warm-start ───────────────────────────
    model = CatBoostRegressor(**FINE_TUNE_PARAMS)
    model.fit(X_train, y_train, init_model=base_model)

    # ── Evaluate on held-out set (original scale) ───────────
    preds_diff = model.predict(X_eval)
    base_vals = df_eval[base_col].values
    preds_orig = np.clip(np.expm1(base_vals + preds_diff), 0, None)
    true_orig = np.clip(np.expm1(base_vals + y_eval), 0, None)

    mae = float(mean_absolute_error(true_orig, preds_orig))
    r2 = float(r2_score(true_orig, preds_orig)) if len(true_orig) > 1 else 0.0

    # ── Retrain on ALL data and save ────────────────────────
    X_all = df_eng[FEATURE_COLS]
    y_all = df_eng[target_col].values
    final_model = CatBoostRegressor(**FINE_TUNE_PARAMS)
    final_model.fit(X_all, y_all, init_model=base_model)

    os.makedirs(MODELS_DIR, exist_ok=True)
    disease_name = new_disease_df.get("disease_name", pd.Series(["unknown"])).iloc[0]
    save_path = _transfer_model_path(disease_name, state, target)
    final_model.save_model(save_path)

    return {
        "mae": round(mae, 4),
        "r2_score": round(r2, 6),
        "model_path": save_path,
    }


def predict_with_transfer(
    disease_name: str,
    state: str,
    df: pd.DataFrame,
) -> dict:
    """Generate forecasts using 6 fine-tuned transfer models.

    Mirrors the interface of
    :func:`app.models.catboost_model.predict_future` but loads
    ``transfer_*`` models instead of ``catboost_*`` models.

    Parameters
    ----------
    disease_name : str
        Disease name used to locate saved transfer models.
    state : str
        State / region name.
    df : pd.DataFrame
        Raw ingested data for the new disease.

    Returns
    -------
    dict
        Forecast dictionary with the same shape as
        ``predict_future()``::

            {
                "cases_7d", "cases_14d", "cases_30d",
                "deaths_7d", "deaths_14d", "deaths_30d",
                "growth_rate", "daily_cases", "daily_deaths",
                "feature_importance",
            }
    """
    # ── Check all 6 transfer models exist ───────────────────
    missing = []
    for _, target_name in TARGETS:
        path = _transfer_model_path(disease_name, state, target_name)
        if not os.path.exists(path):
            missing.append(target_name)
    if missing:
        raise FileNotFoundError(
            f"Missing transfer models for '{disease_name}/{state}': {missing}. "
            f"Run fine_tune_from_covid() first."
        )

    # ── Fill missing columns + engineer features ────────────
    df_filled = _fill_missing_from_covid(df, state)
    if "state" not in df_filled.columns:
        df_filled["state"] = state

    df_eng = engineer_features(df_filled, state)
    if len(df_eng) == 0:
        raise ValueError(
            f"Cannot engineer features for '{disease_name}/{state}'"
        )

    last_row = df_eng.iloc[-1]
    X_pred = pd.DataFrame(
        [last_row[FEATURE_COLS].values], columns=FEATURE_COLS
    )

    current_log_cases = float(last_row["log_cases"])
    current_log_deaths = float(last_row["log_deaths"])

    # ── Predict with each model ─────────────────────────────
    predictions = {}
    models = {}
    for _, target_name in TARGETS:
        model = CatBoostRegressor()
        model.load_model(_transfer_model_path(disease_name, state, target_name))
        models[target_name] = model

        pred_diff = model.predict(X_pred)[0]
        if "cases" in target_name:
            pred_orig = max(0.0, float(np.expm1(current_log_cases + pred_diff)))
        else:
            pred_orig = max(0.0, float(np.expm1(current_log_deaths + pred_diff)))
        predictions[target_name] = pred_orig

    cases_7d = int(round(predictions["cases_7d"]))
    cases_14d = int(round(predictions["cases_14d"]))
    cases_30d = int(round(predictions["cases_30d"]))
    deaths_7d = int(round(predictions["deaths_7d"]))
    deaths_14d = int(round(predictions["deaths_14d"]))
    deaths_30d = int(round(predictions["deaths_30d"]))

    # ── Daily interpolated lists ────────────────────────────
    current_cases = int(last_row.get("new_cases", 0))
    current_deaths = int(last_row.get("new_deaths", 0))

    daily_cases = [int(round(v)) for v in np.linspace(current_cases, cases_30d, 30)]
    daily_deaths = [int(round(v)) for v in np.linspace(current_deaths, deaths_30d, 30)]

    # ── Growth rate ─────────────────────────────────────────
    growth_rate = ((cases_30d - cases_7d) / max(1, cases_7d)) * 100.0

    # ── Feature importance (from cases_7d model) ────────────
    cases7_model = models["cases_7d"]
    feat_names = cases7_model.feature_names_
    feat_scores = cases7_model.get_feature_importance()

    feature_importance = {}
    if feat_names and len(feat_scores) > 0:
        pairs = sorted(
            zip(feat_names, feat_scores), key=lambda x: x[1], reverse=True
        )
        top5 = pairs[:5]
        total = sum(s for _, s in top5)
        if total > 0:
            feature_importance = {
                name: round(score / total * 100, 2)
                for name, score in top5
            }

    return {
        "cases_7d": cases_7d,
        "cases_14d": cases_14d,
        "cases_30d": cases_30d,
        "deaths_7d": deaths_7d,
        "deaths_14d": deaths_14d,
        "deaths_30d": deaths_30d,
        "growth_rate": float(round(growth_rate, 2)),
        "feature_importance": feature_importance,
        "daily_cases": daily_cases,
        "daily_deaths": daily_deaths,
    }
