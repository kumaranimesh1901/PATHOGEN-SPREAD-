"""
CatBoost forecasting model for the PATHOGEN-SPREAD platform.
Trains 6 separate models (cases/deaths × 7d/14d/30d) per state,
using log1p-transformed targets and TimeSeriesSplit cross-validation.
"""
# Optional: pip install optuna

import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from app.database.db import get_state_data
from app.utils.feature_engineering import engineer_features, FEATURE_COLS

# Directory to save models
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_saved")
)

# The 6 targets: (target_column_name, saved_model_basename)
TARGETS = [
    ("target_cases_7d",   "cases_7d"),
    ("target_cases_14d",  "cases_14d"),
    ("target_cases_30d",  "cases_30d"),
    ("target_deaths_7d",  "deaths_7d"),
    ("target_deaths_14d", "deaths_14d"),
    ("target_deaths_30d", "deaths_30d"),
]

# Shared CatBoost hyperparameters
CATBOOST_PARAMS = dict(
    iterations=1000,
    depth=6,
    learning_rate=0.02,
    l2_leaf_reg=7,
    min_data_in_leaf=15,
    subsample=0.8,
    colsample_bylevel=0.8,
    loss_function="RMSE",
    early_stopping_rounds=80,
    verbose=False,
)

# Stronger regularisation for death targets — prevents overfitting to
# high-variance 2021 delta-wave death spikes which are outliers.
CATBOOST_PARAMS_DEATHS = dict(
    iterations=1200,
    depth=5,
    learning_rate=0.015,
    l2_leaf_reg=10,
    min_data_in_leaf=20,
    subsample=0.75,
    colsample_bylevel=0.75,
    loss_function="RMSE",
    early_stopping_rounds=100,
    verbose=False,
)


def _model_path(state_name: str, model_name: str) -> str:
    """Return the full path for a saved model file (per-state)."""
    return os.path.join(MODELS_DIR, f"catboost_{state_name}_{model_name}.cbm")


def train_catboost(state_name: str) -> dict:
    """
    Train 6 CatBoost models for the given state.

    Steps:
      1. Load data from DB
      2. Engineer features (with log1p targets)
      3. For each of 6 targets:
         a. Drop rows where target is NaN
         b. TimeSeriesSplit (5 folds) cross-validation
         c. Collect OOF predictions, compute metrics on original scale
         d. Train final model on ALL data
         e. Save to models_saved/
      4. Return dict of all metrics

    Args:
        state_name: Indian state name (e.g. "Karnataka").

    Returns:
        Dict with per-target metrics (MAE, RMSE, R²).
    """
    df_raw = get_state_data(state_name)
    if len(df_raw) == 0:
        raise ValueError(f"No data found for state '{state_name}'")

    df = engineer_features(df_raw, state_name)
    if len(df) < 60:
        raise ValueError(
            f"Insufficient data for state '{state_name}' after feature engineering "
            f"(need >= 60 rows, got {len(df)})"
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    all_metrics = {}

    for target_col, model_name in TARGETS:
        # Drop rows where this specific target is NaN
        df_target = df.dropna(subset=[target_col]).reset_index(drop=True)
        if len(df_target) < 30:
            print(f"  ⚠ Skipping {model_name}: only {len(df_target)} valid rows")
            continue

        X = df_target[FEATURE_COLS]
        y = df_target[target_col].values  # log-difference scale

        # Sample weights: address variance collapse between pandemic and
        # endemic phases. Higher weight for high-case periods proportional
        # to log magnitude, prevents model from ignoring the flat tail.
        base_col = "log_cases" if "cases" in model_name else "log_deaths"
        weights = 1.0 + df_target[base_col].values

        # TimeSeriesSplit cross-validation (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)
        oof_true = []
        oof_pred = []
        oof_base = []  # current log values for reconstructing original scale

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = weights[train_idx]

            # Use death-specific params for death targets
            params = CATBOOST_PARAMS_DEATHS if target_col.startswith("target_deaths") else CATBOOST_PARAMS
            fold_model = CatBoostRegressor(**params)
            fold_model.fit(X_tr, y_tr, sample_weight=w_tr)

            preds_diff = fold_model.predict(X_val)
            # Reconstruct original scale: expm1(current_log + predicted_diff)
            base_vals = df_target[base_col].values[val_idx]
            preds_orig = np.clip(np.expm1(base_vals + preds_diff), 0, None)
            true_orig = np.clip(np.expm1(base_vals + y_val), 0, None)

            oof_true.extend(true_orig.tolist())
            oof_pred.extend(preds_orig.tolist())

        # Compute metrics on original scale
        oof_true = np.array(oof_true)
        oof_pred = np.array(oof_pred)

        mae = float(mean_absolute_error(oof_true, oof_pred))
        rmse = float(np.sqrt(mean_squared_error(oof_true, oof_pred)))
        r2 = float(r2_score(oof_true, oof_pred)) if len(oof_true) > 1 else 0.0
        mean_y = float(np.mean(np.abs(oof_true))) if float(np.mean(np.abs(oof_true))) > 0 else 1.0
        rel_mae = round(mae / mean_y * 100, 2)
        rel_rmse = round(rmse / mean_y * 100, 2)

        print(
            f"  {model_name:<15s}  MAE={mae:>10.2f}  RMSE={rmse:>10.2f}  "
            f"R²={r2:>8.4f}  RelMAE={rel_mae:.1f}%"
        )

        all_metrics[model_name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 6),
            "relative_mae_pct": rel_mae,
            "relative_rmse_pct": rel_rmse,
        }

        # Train final model on ALL data (with sample weights)
        params = CATBOOST_PARAMS_DEATHS if target_col.startswith("target_deaths") else CATBOOST_PARAMS
        final_model = CatBoostRegressor(**params)
        final_model.fit(X, y, sample_weight=weights)
        final_model.save_model(_model_path(state_name, model_name))

    print(f"✅ All models saved to {MODELS_DIR}")
    return all_metrics


def predict_future(state_name: str, df: pd.DataFrame) -> dict:
    """
    Generate forecasts using the 6 trained CatBoost models.

    Models predict log-differences (change from current level). We
    reconstruct the absolute forecast by adding the predicted diff
    back to the current log value before expm1().

    Steps:
      1. Check all 6 model files exist
      2. Engineer features
      3. Take last row of engineered features
      4. Predict log-diff with each model, reconstruct original scale
      5. Build daily interpolated lists
      6. Compute growth rate and feature importance

    Args:
        state_name: State to forecast for.
        df: Full historical DataFrame for the state.

    Returns:
        Dict with projections, daily lists, and feature importance.
    """
    # 1. Check all 6 model files exist
    missing = [name for _, name in TARGETS if not os.path.exists(_model_path(state_name, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing trained models for '{state_name}': {missing}. "
            f"Run training first: GET /train?state={state_name}"
        )

    # 2. Engineer features
    df_eng = engineer_features(df, state_name)
    if len(df_eng) == 0:
        raise ValueError(f"Cannot engineer features for state '{state_name}'")

    # 3. Take the LAST ROW as a single-row DataFrame
    last_row = df_eng.iloc[-1]
    X_pred = pd.DataFrame([last_row[FEATURE_COLS].values], columns=FEATURE_COLS)

    # Current log-scale values for reconstructing from log-differences
    current_log_cases = float(last_row["log_cases"])
    current_log_deaths = float(last_row["log_deaths"])

    # 4. Predict log-diff with each model, reconstruct to original scale
    predictions = {}
    models = {}
    for target_col, model_name in TARGETS:
        model = CatBoostRegressor()
        model.load_model(_model_path(state_name, model_name))
        models[model_name] = model

        pred_diff = model.predict(X_pred)[0]
        # Reconstruct: add predicted change to current log value
        if "cases" in model_name:
            pred_orig = max(0.0, float(np.expm1(current_log_cases + pred_diff)))
        else:
            pred_orig = max(0.0, float(np.expm1(current_log_deaths + pred_diff)))
        predictions[model_name] = pred_orig

    cases_7d = int(round(predictions["cases_7d"]))
    cases_14d = int(round(predictions["cases_14d"]))
    cases_30d = int(round(predictions["cases_30d"]))
    deaths_7d = int(round(predictions["deaths_7d"]))
    deaths_14d = int(round(predictions["deaths_14d"]))
    deaths_30d = int(round(predictions["deaths_30d"]))

    # 5. Build daily_cases and daily_deaths using linspace
    current_cases = int(last_row.get("new_cases", 0))
    current_deaths = int(last_row.get("new_deaths", 0))

    daily_cases = [
        int(round(v)) for v in np.linspace(current_cases, cases_30d, 30)
    ]
    daily_deaths = [
        int(round(v)) for v in np.linspace(current_deaths, deaths_30d, 30)
    ]

    # 6. Growth rate
    growth_rate = ((cases_30d - cases_7d) / max(1, cases_7d)) * 100.0

    # 7. Feature importance from cases_7d model (top 5)
    cases7_model = models["cases_7d"]
    feat_names = cases7_model.feature_names_
    feat_scores = cases7_model.get_feature_importance()

    if feat_names and len(feat_scores) > 0:
        pairs = sorted(zip(feat_names, feat_scores), key=lambda x: x[1], reverse=True)
        top5 = pairs[:5]
        total = sum(s for _, s in top5)
        if total > 0:
            feature_importance = {
                name: round(score / total * 100, 2) for name, score in top5
            }
        else:
            feature_importance = {name: 0.0 for name, _ in top5}
    else:
        feature_importance = {}

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


def tune_catboost(state_name: str, n_trials: int = 50) -> dict:
    """
    Run Optuna hyperparameter search for the cases_7d target.

    Requires: pip install optuna

    Args:
        state_name: State to tune for.
        n_trials: Number of Optuna trials (default 50).

    Returns:
        Dict of best hyperparameters found.
    """
    import optuna

    # Load and engineer features
    df_raw = get_state_data(state_name)
    if len(df_raw) == 0:
        raise ValueError(f"No data found for state '{state_name}'")

    df = engineer_features(df_raw, state_name)
    if len(df) < 60:
        raise ValueError(
            f"Insufficient data for state '{state_name}' after feature engineering "
            f"(need >= 60 rows, got {len(df)})"
        )

    target_col = "target_cases_7d"
    df_target = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df_target[FEATURE_COLS]
    y = df_target[target_col].values  # log-difference scale
    base_log = df_target["log_cases"].values
    weights = 1.0 + base_log

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 30),
            "loss_function": "RMSE",
            "verbose": False,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        oof_mae_list = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = weights[train_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)

            preds_diff = model.predict(X_val)
            base_vals = base_log[val_idx]
            preds_orig = np.clip(np.expm1(base_vals + preds_diff), 0, None)
            true_orig = np.clip(np.expm1(base_vals + y_val), 0, None)

            fold_mae = float(mean_absolute_error(true_orig, preds_orig))
            oof_mae_list.append(fold_mae)

        return float(np.mean(oof_mae_list))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✅ Best params for {state_name} (cases_7d):")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    print(f"   Best MAE (original scale): {study.best_value:.2f}")

    return study.best_params
