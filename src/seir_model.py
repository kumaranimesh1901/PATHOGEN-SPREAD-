import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional

class SEIRModel:
    """
    Standard compartmental SEIR (Susceptible, Exposed, Infected, Recovered) model
    fitted using Scipy ODE integration.
    """
    def __init__(self, beta: float = 0.5, sigma: float = 0.2, gamma: float = 0.1):
        # Parameters:
        # beta: contact/transmission rate
        # sigma: incubation rate (1 / incubation period)
        # gamma: recovery rate (1 / infectious period)
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.fitted_params: Dict[str, float] = {}
        
        # States at the end of fitting (used for predictions)
        self.last_state: Optional[np.ndarray] = None
        self.N: float = 1e6 # Population
        self.fit_history: Optional[pd.DataFrame] = None

    @staticmethod
    def _seir_equations(
        y: Tuple[float, float, float, float], t: float, N: float, beta: float, sigma: float, gamma: float
    ) -> List[float]:
        """
        SEIR system of ordinary differential equations.
        """
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def _solve(
        self, t: np.ndarray, init_vals: Tuple[float, float, float, float], N: float, beta: float, sigma: float, gamma: float
    ) -> np.ndarray:
        """
        Solves the SEIR system for a given timeline.
        """
        return odeint(self._seir_equations, init_vals, t, args=(N, beta, sigma, gamma))

    def fit(self, active_cases: np.ndarray, population: int, initial_exposed: Optional[float] = None) -> 'SEIRModel':
        """
        Fits the SEIR model parameters (beta, sigma, gamma) to actual active cases.
        Active cases are defined as: confirmed - recovered - deaths.
        """
        self.N = float(population)
        n_days = len(active_cases)
        t = np.arange(n_days)
        
        # Initial conditions:
        I0 = float(active_cases[0])
        # If not provided, assume exposed is twice the infected count
        E0 = float(initial_exposed) if initial_exposed is not None else I0 * 2.0
        R0 = 0.0
        S0 = self.N - E0 - I0 - R0
        init_vals = (S0, E0, I0, R0)
        
        # Objective function to minimize: Mean Absolute Error (MAE) or Mean Squared Error (MSE)
        def objective(params: List[float]) -> float:
            beta, sigma, gamma = params
            # Solve system
            solution = self._solve(t, init_vals, self.N, beta, sigma, gamma)
            I_pred = solution[:, 2] # Index 2 is I (Infected)
            
            # Penalize negative values or values that deviate heavily
            mse = np.mean((active_cases - I_pred) ** 2)
            return mse

        # Initial guesses and bounds
        # beta: 0.01 to 3.0
        # sigma: 0.05 to 1.0 (incubation of 1 to 20 days)
        # gamma: 0.02 to 1.0 (recovery of 1 to 50 days)
        initial_guess = [self.beta, self.sigma, self.gamma]
        bounds = [(0.01, 3.0), (0.05, 1.0), (0.02, 1.0)]
        
        res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        self.beta, self.sigma, self.gamma = res.x
        self.fitted_params = {
            "beta": float(self.beta),
            "sigma": float(self.sigma),
            "gamma": float(self.gamma),
            "R0": float(self.beta / (self.gamma + 1e-8)) # Basic reproduction number
        }
        
        # Solve with fitted parameters to store history
        fitted_solution = self._solve(t, init_vals, self.N, self.beta, self.sigma, self.gamma)
        self.fit_history = pd.DataFrame(fitted_solution, columns=["S", "E", "I", "R"])
        self.fit_history["active_cases_actual"] = active_cases
        
        # Store last state for future predictions
        self.last_state = fitted_solution[-1]
        
        return self

    def predict(self, days: int) -> np.ndarray:
        """
        Predicts future active infections (I) for the specified number of days
        starting from the end of the fitted period.
        """
        if self.last_state is None:
            raise ValueError("Model must be fitted before making predictions.")
            
        t = np.arange(days + 1) # Including day 0 (which is the last fitted day)
        
        solution = self._solve(t, tuple(self.last_state), self.N, self.beta, self.sigma, self.gamma)
        
        # Return predictions starting from day 1 (exclude day 0, which is the historical end)
        I_pred = solution[1:, 2]
        return I_pred

    def plot_results(self, title: str = "SEIR Outbreak Projection") -> plt.Figure:
        """
        Generates a matplotlib plot showing the fitted curves for S, E, I, R.
        """
        if self.fit_history is None:
            raise ValueError("Model must be fitted before plotting.")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        t = np.arange(len(self.fit_history))
        
        ax.plot(t, self.fit_history["S"], label="Susceptible", color="blue")
        ax.plot(t, self.fit_history["E"], label="Exposed", color="orange")
        ax.plot(t, self.fit_history["I"], label="Infected (Fitted)", color="red", linewidth=2)
        ax.plot(t, self.fit_history["R"], label="Recovered", color="green")
        ax.scatter(t, self.fit_history["active_cases_actual"], label="Actual Active", color="black", s=10, alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel("Days")
        ax.set_ylabel("Population Count")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        return fig
