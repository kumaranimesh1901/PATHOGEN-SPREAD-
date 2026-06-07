"""
SEIR ODE compartmental simulator for the PATHOGEN-SPREAD platform.
Solves equations for Susceptible, Exposed, Infected, Recovered, and Deceased cohorts.
"""

import numpy as np
from scipy.integrate import solve_ivp


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
