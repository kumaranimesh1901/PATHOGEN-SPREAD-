"""
Hybrid forecasting coordinator for the PATHOGEN-SPREAD platform.
Selects between CatBoost, SEIR, or a weighted blend based on available data depth.
"""

import numpy as np

from app.database.db import get_state_data, get_row_count
from app.models.catboost_model import predict_future
from app.models.seir_model import run_seir
from app.utils.risk_calculator import calculate_risk


def run_forecast(state_name: str, user_inputs: dict) -> dict:
    """
    Execute the appropriate forecasting engine for a state.

    Engine selection:
      >= 60 rows → CatBoost only
      30-59 rows → Hybrid (weighted CatBoost + SEIR blend)
      < 30 rows  → SEIR only

    Args:
        state_name: Indian state name.
        user_inputs: Dict from the ForecastRequest payload.

    Returns:
        Complete forecast result dict.
    """
    rows = get_row_count(state_name)

    if rows >= 60:
        engine = "catboost"
    elif rows >= 30:
        engine = "hybrid"
    else:
        engine = "seir"

    # Load full DataFrame for CatBoost / hybrid
    df = get_state_data(state_name)

    # ── Defaults ──
    result = {
        "state": state_name,
        "engine_used": engine,
        "cases_7d": 0,
        "cases_14d": 0,
        "cases_30d": 0,
        "deaths_7d": 0,
        "deaths_14d": 0,
        "deaths_30d": 0,
        "peak_day": None,
        "outbreak_duration": None,
        "growth_rate": 0.0,
        "risk_category": "Low",
        "risk_score": 0.0,
        "reproduction_number": None,
        "feature_importance": None,
        "seir_curve": None,
        "daily_cases": [0] * 30,
        "daily_deaths": [0] * 30,
    }

    # ═══════════════════════════════════════════
    # Engine: CatBoost
    # ═══════════════════════════════════════════
    if engine == "catboost":
        cb = predict_future(state_name, df)
        result.update({
            "cases_7d": cb["cases_7d"],
            "cases_14d": cb["cases_14d"],
            "cases_30d": cb["cases_30d"],
            "deaths_7d": cb["deaths_7d"],
            "deaths_14d": cb["deaths_14d"],
            "deaths_30d": cb["deaths_30d"],
            "growth_rate": cb["growth_rate"],
            "feature_importance": cb["feature_importance"],
            "daily_cases": cb["daily_cases"],
            "daily_deaths": cb["daily_deaths"],
        })

    # ═══════════════════════════════════════════
    # Engine: SEIR
    # ═══════════════════════════════════════════
    elif engine == "seir":
        seir_res = run_seir(user_inputs)

        # Compute daily_cases from S curve: daily new infections = S[i] - S[i+1]
        S = seir_res["seir_curve"]["susceptible"]
        daily_cases = [max(0, int(S[i] - S[i + 1])) for i in range(30)]

        # Compute daily_deaths: mu * I[i]
        mu = float(user_inputs.get("mortality_rate", 0.02))
        I_curve = seir_res["seir_curve"]["infected"]
        daily_deaths = [max(0, int(mu * I_curve[i])) for i in range(30)]

        # Growth rate from daily_cases halves
        first_half = sum(daily_cases[:14])
        second_half = sum(daily_cases[14:30])
        seir_growth = ((second_half - first_half) / max(1, first_half)) * 100.0

        result.update({
            "cases_7d": seir_res["cases_7d"],
            "cases_14d": seir_res["cases_14d"],
            "cases_30d": seir_res["cases_30d"],
            "deaths_7d": seir_res["deaths_7d"],
            "deaths_14d": seir_res["deaths_14d"],
            "deaths_30d": seir_res["deaths_30d"],
            "peak_day": seir_res["peak_day"],
            "outbreak_duration": seir_res["outbreak_duration"],
            "growth_rate": round(seir_growth, 2),
            "reproduction_number": seir_res["reproduction_number"],
            "seir_curve": seir_res["seir_curve"],
            "daily_cases": daily_cases,
            "daily_deaths": daily_deaths,
        })

    # ═══════════════════════════════════════════
    # Engine: Hybrid (weighted blend)
    # ═══════════════════════════════════════════
    elif engine == "hybrid":
        cb = predict_future(state_name, df)
        seir_res = run_seir(user_inputs)

        weight_cb = (rows - 30) / 30.0
        weight_seir = 1.0 - weight_cb

        # Blend numeric projections
        for key in ["cases_7d", "cases_14d", "cases_30d",
                     "deaths_7d", "deaths_14d", "deaths_30d"]:
            blended = int(round(
                weight_cb * cb[key] + weight_seir * seir_res[key]
            ))
            result[key] = max(0, blended)

        # SEIR daily values for blending
        S = seir_res["seir_curve"]["susceptible"]
        seir_daily_cases = [max(0, int(S[i] - S[i + 1])) for i in range(30)]
        mu = float(user_inputs.get("mortality_rate", 0.02))
        I_curve = seir_res["seir_curve"]["infected"]
        seir_daily_deaths = [max(0, int(mu * I_curve[i])) for i in range(30)]

        # Blend daily lists
        daily_cases = [
            max(0, int(round(weight_cb * cb["daily_cases"][i] + weight_seir * seir_daily_cases[i])))
            for i in range(30)
        ]
        daily_deaths = [
            max(0, int(round(weight_cb * cb["daily_deaths"][i] + weight_seir * seir_daily_deaths[i])))
            for i in range(30)
        ]

        result.update({
            "growth_rate": round(
                weight_cb * cb["growth_rate"] +
                weight_seir * ((sum(seir_daily_cases[14:30]) - sum(seir_daily_cases[:14]))
                               / max(1, sum(seir_daily_cases[:14])) * 100), 2
            ),
            "peak_day": seir_res["peak_day"],
            "outbreak_duration": seir_res["outbreak_duration"],
            "reproduction_number": seir_res["reproduction_number"],
            "feature_importance": cb["feature_importance"],
            "seir_curve": seir_res["seir_curve"],
            "daily_cases": daily_cases,
            "daily_deaths": daily_deaths,
        })

    # ═══════════════════════════════════════════
    # Risk calculation
    # ═══════════════════════════════════════════
    mortality_rate = float(user_inputs.get("mortality_rate", 0.02))
    risk_category, risk_score = calculate_risk(result["growth_rate"], mortality_rate)
    result["risk_category"] = risk_category
    result["risk_score"] = risk_score

    return result
