"""
Risk calculator utilities for the PATHOGEN-SPREAD forecasting platform.
Computes risk level (Low, Medium, High) and numeric risk score.
"""


def calculate_risk(growth_rate: float, mortality_rate: float) -> tuple:
    """
    Calculate the risk category and risk score.

    Rules:
      - growth_rate < 5% AND mortality_rate < 1%  → "Low"
      - growth_rate 5-15% OR mortality_rate 1-3%  → "Medium"
      - growth_rate > 15% OR mortality_rate > 3%  → "High"
      - risk_score = (growth_rate * 0.6 + mortality_rate * 100 * 0.4) clamped to 0-100

    Args:
        growth_rate: Percentage growth rate of cases (e.g. 8.5 for 8.5%).
        mortality_rate: Mortality rate as a fraction (e.g. 0.02 for 2%).

    Returns:
        Tuple of (risk_category: str, risk_score: float).
    """
    mortality_pct = mortality_rate * 100.0

    # Assign risk category based on rules
    if growth_rate < 5.0 and mortality_pct < 1.0:
        risk_category = "Low"
    elif growth_rate > 15.0 or mortality_pct > 3.0:
        risk_category = "High"
    else:
        risk_category = "Medium"

    # Compute risk score
    risk_score = (growth_rate * 0.6) + (mortality_pct * 40.0)  # mortality_rate * 100 * 0.4 = mortality_pct * 40.0
    risk_score = float(max(0.0, min(100.0, risk_score)))

    return risk_category, risk_score
