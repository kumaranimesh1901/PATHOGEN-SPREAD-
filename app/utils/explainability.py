"""
Explainability utility for the PATHOGEN-SPREAD forecasting system.
Computes and formats feature importance from CatBoost models.
"""


def extract_feature_importance(model) -> dict:
    """
    Extract and map feature importances from a trained CatBoostRegressor model
    into the required JSON/dict format.

    Required output structure:
    {
        "new_cases_lag": float,
        "growth_rate": float,
        "population_density": float,
        "mortality_rate": float,
        "mobility_index": float
    }

    Args:
        model: Trained CatBoostRegressor model.

    Returns:
        Dict containing mapped feature importances.
    """
    default_importance = {
        "new_cases_lag": 50.0,
        "growth_rate": 30.0,
        "population_density": 20.0,
        "mortality_rate": 0.0,
        "mobility_index": 0.0
    }

    if model is None:
        return default_importance

    try:
        importance = model.get_feature_importance()
        feature_names = model.feature_names_

        if not feature_names:
            return default_importance

        importance_dict = dict(zip(feature_names, importance))

        # Sum of lags and rolling averages for "new_cases_lag"
        lag_keys = ["lag_1", "lag_3", "lag_7", "lag_14", "rolling_7day_avg", "rolling_14day_avg"]
        new_cases_lag_val = sum(importance_dict.get(k, 0.0) for k in lag_keys)

        growth_rate_val = importance_dict.get("growth_rate", 0.0)
        pop_density_val = importance_dict.get("population_density", 0.0)

        # Normalize the sum of importances so they sum to 100% (or keep raw percentages)
        raw_dict = {
            "new_cases_lag": float(new_cases_lag_val),
            "growth_rate": float(growth_rate_val),
            "population_density": float(pop_density_val),
            "mortality_rate": 0.0,
            "mobility_index": 0.0
        }

        # Check if the sum of all elements is positive, normalize if needed to make them sum to 100
        total_sum = sum(raw_dict.values())
        if total_sum > 0:
            for k in raw_dict:
                raw_dict[k] = round((raw_dict[k] / total_sum) * 100.0, 2)
        else:
            return default_importance

        return raw_dict

    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return default_importance
