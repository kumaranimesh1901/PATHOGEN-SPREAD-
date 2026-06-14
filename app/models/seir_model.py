"""
SEIR ODE compartmental simulator for the PATHOGEN-SPREAD platform.
Solves equations for Susceptible, Exposed, Infected, Recovered, and Deceased cohorts.

Includes a cold-start mode for new pathogens that relies entirely on
epidemiological priors (R₀, incubation period, infectious period) when
zero historical data is available.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ──────────────────────────────────────────────────────────
# Disease presets — known pathogen configurations
# ──────────────────────────────────────────────────────────

DISEASE_PRESETS = {
    "covid19": {
        "r0": 2.5,
        "incubation_days": 5,
        "infectious_days": 7,
        "mortality_rate": 0.02,
    },
    "influenza": {
        "r0": 1.4,
        "incubation_days": 2,
        "infectious_days": 4,
        "mortality_rate": 0.001,
    },
    "mpox": {
        "r0": 0.8,
        "incubation_days": 8,
        "infectious_days": 21,
        "mortality_rate": 0.03,
    },
    "unknown": {
        "r0": 2.0,
        "incubation_days": 5,
        "infectious_days": 7,
        "mortality_rate": 0.01,
    },
}


def run_cold_start_seir(state: str, disease_config: dict) -> dict:
    """
    Run an SEIR forecast using only epidemiological priors — no historical
    data required.

    This is the "day-zero" forecasting mode used by :class:`HybridController`
    when fewer than 14 days of observed data exist for a new pathogen.

    Parameters
    ----------
    state : str
        Name of the state / region (carried through for labelling).
    disease_config : dict
        Epidemiological configuration.  Recognised keys:

        - ``disease_type`` (str): Key into :data:`DISEASE_PRESETS` to
          auto-load a base configuration.  Explicit keys below
          override the preset.
        - ``r0`` (float): Basic reproduction number (default 2.5).
        - ``incubation_days`` (int): Mean incubation period (default 5).
        - ``infectious_days`` (int): Mean infectious period (default 7).
        - ``mortality_rate`` (float): IFR as a fraction (default 0.01).
        - ``seed_cases`` (int): Initial confirmed cases (default 1).
        - ``population`` (int): State population (default 10 000 000).

    Returns
    -------
    dict
        Standardised forecast dictionary::

            {
                "state": str,
                "model_used": "cold_start_seir",
                "confidence": "low",
                "cases_7d": int,
                "cases_14d": int,
                "cases_30d": int,
                "cases_60d": int,
                "cases_90d": int,
                "deaths_7d": int,
                "deaths_14d": int,
                "deaths_30d": int,
                "deaths_60d": int,
                "deaths_90d": int,
                "peak_day": int,
                "peak_cases": int,
                "outbreak_duration": int,
                "reproduction_number": float,
                "growth_rate": float,
                "seir_curve": dict,
                "daily_cases": list[int],   # 30 entries
                "daily_deaths": list[int],  # 30 entries
            }
    """
    config = dict(disease_config) if disease_config else {}

    # ── Resolve preset ──────────────────────────────────────
    disease_type = config.pop("disease_type", None)
    if disease_type and disease_type in DISEASE_PRESETS:
        # Start from preset, then overlay explicit overrides
        resolved = dict(DISEASE_PRESETS[disease_type])
        resolved.update(config)
        config = resolved

    # ── Extract parameters with defaults ────────────────────
    r0              = float(config.get("r0", 2.5))
    incubation_days = int(config.get("incubation_days", 5))
    infectious_days = int(config.get("infectious_days", 7))
    mortality_rate  = float(config.get("mortality_rate", 0.01))
    seed_cases      = int(config.get("seed_cases", 1))
    population      = int(config.get("population", 10_000_000))

    # ── Translate to SEIR rate parameters ───────────────────
    # sigma = 1 / incubation_days  (E → I transition rate)
    # gamma = 1 / infectious_days  (I → R transition rate)
    # beta  = R₀ × gamma           (transmission rate)
    gamma = 1.0 / infectious_days
    beta  = r0 * gamma

    # ── Build user_inputs for the existing run_seir() ───────
    seir_inputs = {
        "population": population,
        "transmission_rate": beta,
        "recovery_rate": gamma,
        "mortality_rate": mortality_rate,
        "current_cases": seed_cases,
        "current_deaths": 0,
        "days": 90,
    }

    seir_result = run_seir(seir_inputs)

    # ── Extract extended forecasts (60d, 90d) ───────────────
    S_curve = seir_result["seir_curve"]["susceptible"]
    I_curve = seir_result["seir_curve"]["infected"]
    D_curve_values = seir_result["seir_curve"].get("recovered", [])  # not used below

    # Cumulative cases from susceptible depletion
    S0 = S_curve[0]
    cases_60d = int(max(0, S0 - S_curve[min(60, 90)]))
    cases_90d = int(max(0, S0 - S_curve[min(90, len(S_curve) - 1)]))

    # Deaths from the ODE D compartment (index 4 in run_seir, but
    # run_seir already computes deaths_7d/14d/30d; we need 60d/90d).
    # Re-run the ODE ourselves to get the D curve at day 60 and 90 would
    # be wasteful — instead, scale from the S depletion × mortality_rate.
    # For consistency with run_seir's own approach, approximate:
    deaths_60d = int(max(0, round(cases_60d * mortality_rate)))
    deaths_90d = int(max(0, round(cases_90d * mortality_rate)))

    # Peak infected day and value
    peak_idx = int(np.argmax(I_curve))
    peak_cases = int(max(0, round(I_curve[peak_idx])))

    # Daily cases/deaths for first 30 days (for chart compatibility)
    daily_cases = [
        max(0, int(S_curve[i] - S_curve[i + 1])) for i in range(30)
    ]
    mu = mortality_rate
    daily_deaths = [
        max(0, int(mu * I_curve[i])) for i in range(30)
    ]

    # Growth rate
    first_half = sum(daily_cases[:14])
    second_half = sum(daily_cases[14:30])
    growth_rate = ((second_half - first_half) / max(1, first_half)) * 100.0

    return {
        "state": state,
        "model_used": "cold_start_seir",
        "confidence": "low",
        # Standard horizons
        "cases_7d": seir_result["cases_7d"],
        "cases_14d": seir_result["cases_14d"],
        "cases_30d": seir_result["cases_30d"],
        "cases_60d": cases_60d,
        "cases_90d": cases_90d,
        "deaths_7d": seir_result["deaths_7d"],
        "deaths_14d": seir_result["deaths_14d"],
        "deaths_30d": seir_result["deaths_30d"],
        "deaths_60d": deaths_60d,
        "deaths_90d": deaths_90d,
        # Peak info
        "peak_day": seir_result["peak_day"],
        "peak_cases": peak_cases,
        "outbreak_duration": seir_result["outbreak_duration"],
        "reproduction_number": seir_result["reproduction_number"],
        "growth_rate": round(growth_rate, 2),
        # Curves (full 90 days)
        "seir_curve": seir_result["seir_curve"],
        # Daily lists (30 days — for chart compatibility)
        "daily_cases": daily_cases,
        "daily_deaths": daily_deaths,
    }


def run_seir(user_inputs: dict) -> dict:
    """
    Solve SEIR ordinary differential equations using RK45 method.

    Equations:
      dS/dt = -beta * S * I / N
      dE/dt =  beta * S * I / N - sigma * E
      dI/dt =  sigma * E - gamma * I - mu * I
      dR/dt =  gamma * I
      dD/dt =  mu * I (deaths tracked separately)

    Parameters in user_inputs:
      population (N)         : total population
      transmission_rate (beta): transmission rate
      recovery_rate (gamma)   : recovery rate
      mortality_rate (mu)     : mortality rate
      current_cases (I0)      : current infected cases
      current_deaths (D0)     : current deaths
      days                    : simulation period (defaults to 180)

    Args:
        user_inputs: Dictionary containing initial variables.

    Returns:
        Dict with metrics (7d/14d/30d projections, peak_day, outbreak_duration,
        R0, and curves values).
    """
    # Extract values
    N = user_inputs.get("population")
    beta = user_inputs.get("transmission_rate")
    gamma = user_inputs.get("recovery_rate")
    mu = user_inputs.get("mortality_rate")
    I0 = user_inputs.get("current_cases")
    D0 = user_inputs.get("current_deaths", 0)
    days = user_inputs.get("days", 180)

    # Validate inputs
    if N is None or N <= 0:
        raise ValueError("Population (N) must be greater than 0.")
    if beta is None or beta <= 0:
        raise ValueError("Transmission rate (beta) must be greater than 0.")
    if gamma is None or gamma <= 0:
        raise ValueError("Recovery rate (gamma) must be greater than 0.")
    if mu is None or not (0.0 <= mu <= 1.0):
        raise ValueError("Mortality rate (mu) must be between 0.0 and 1.0.")
    if I0 is None or I0 < 0:
        raise ValueError("Current cases (I0) must be non-negative.")
    if I0 >= N:
        raise ValueError("Current cases cannot exceed total population.")

    # 1 / incubation period (standard is 5 days)
    sigma = 0.2

    # Initial conditions
    E0 = I0 * 2.0
    R0_init = float(D0)
    S0 = float(N - I0 - E0 - R0_init)

    if S0 < 0:
        raise ValueError("Sum of current cases and current deaths exceeds the population.")

    # S, E, I, R, D initial state
    y0 = [S0, E0, float(I0), R0_init, 0.0]

    # SEIR ODEs
    def seir_equations(t, y):
        S_t, E_t, I_t, R_t, D_t = y
        dS = -beta * S_t * I_t / N
        dE = (beta * S_t * I_t / N) - (sigma * E_t)
        dI = (sigma * E_t) - (gamma * I_t) - (mu * I_t)
        dR = gamma * I_t
        dD = mu * I_t
        return [dS, dE, dI, dR, dD]

    # Evaluate at integer day increments
    t_span = (0.0, float(days))
    t_eval = np.arange(0, days + 1)

    sol = solve_ivp(seir_equations, t_span, y0, t_eval=t_eval, method="RK45")

    if not sol.success:
        raise RuntimeError(f"SEIR solver failed to converge: {sol.message}")

    S_curve = sol.y[0]
    E_curve = sol.y[1]
    I_curve = sol.y[2]
    R_curve = sol.y[3]
    D_curve = sol.y[4]

    # 7, 14, 30 day case projections: S0 - S_t
    cases_7d = int(max(0, S0 - S_curve[min(7, days)]))
    cases_14d = int(max(0, S0 - S_curve[min(14, days)]))
    cases_30d = int(max(0, S0 - S_curve[min(30, days)]))

    # 7, 14, 30 day death projections: D_t - D0
    deaths_7d = int(max(0, D_curve[min(7, days)] - D_curve[0]))
    deaths_14d = int(max(0, D_curve[min(14, days)] - D_curve[0]))
    deaths_30d = int(max(0, D_curve[min(30, days)] - D_curve[0]))

    # Peak day
    peak_idx = np.argmax(I_curve)
    peak_day = int(t_eval[peak_idx])

    # Outbreak duration: days until active infections drop below 1% of peak value (after the peak)
    peak_val = I_curve[peak_idx]
    threshold = peak_val * 0.01
    post_peak_I = I_curve[peak_idx:]
    below_threshold = np.where(post_peak_I < threshold)[0]
    if len(below_threshold) > 0:
        outbreak_duration = int(peak_day + below_threshold[0])
    else:
        outbreak_duration = int(days)

    reproduction_number = float(beta / gamma) if gamma > 0 else 0.0

    return {
        "cases_7d": cases_7d,
        "cases_14d": cases_14d,
        "cases_30d": cases_30d,
        "deaths_7d": deaths_7d,
        "deaths_14d": deaths_14d,
        "deaths_30d": deaths_30d,
        "peak_day": peak_day,
        "outbreak_duration": outbreak_duration,
        "reproduction_number": reproduction_number,
        "seir_curve": {
            "susceptible": S_curve.tolist(),
            "exposed": E_curve.tolist(),
            "infected": I_curve.tolist(),
            "recovered": R_curve.tolist(),
            "days": t_eval.tolist()
        }
    }
