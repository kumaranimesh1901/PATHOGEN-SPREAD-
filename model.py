import numpy
import torch
import torch.nn as nn
from scipy.integrate import odeint


def run_seir(N=1000000, E0=100, I0=10, beta=0.3, sigma=0.2, gamma=0.1, days=200):
    """Simulate SEIR epidemiological model. N=population, E0=initial exposed, I0=initial infected, beta=transmission rate, sigma=incubation rate, gamma=recovery rate, days=simulation length. Returns numpy array of shape (days, 4) with columns S, E, I, R."""
    try:
        def seir_equations(y, t, beta, sigma, gamma, N):
            """Compute derivatives for the SEIR compartmental model."""
            S, E, I, R = y
            dS = -beta * S * I / N
            dE = beta * S * I / N - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I
            return [dS, dE, dI, dR]

        S0 = N - E0 - I0
        y0 = [S0, E0, I0, 0.0]
        t = numpy.linspace(0, days, days)
        solution = odeint(seir_equations, y0, t, args=(beta, sigma, gamma, N))
        return solution
    except Exception as e:
        print(f"An error occurred in run_seir: {e}")
        return numpy.zeros((days, 4))


class PathogenLSTM(nn.Module):
    """LSTM model for pathogen spread forecasting."""

    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        """Initialize PathogenLSTM with configurable architecture parameters."""
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass. x shape: (batch, seq_len, input_size). Returns shape: (batch, 1)."""
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


if __name__ == "__main__":
    try:
        print("Testing SEIR model...")
        sol = run_seir()
        print("SEIR output shape:", sol.shape)

        print("Testing LSTM model...")
        model = PathogenLSTM()
        print(model)
        dummy = torch.randn(8, 14, 4)
        output = model(dummy)
        print("LSTM output shape:", output.shape)
    except Exception as e:
        print(f"An error occurred: {e}")
