import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple, Optional
from src.seir_model import SEIRModel
from src.lstm_model import LSTMForecaster
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

class HybridPredictor:
    """
    Hybrid outbreak predictor combining the compartmental SEIR model and deep-learning LSTM model.
    """
    def __init__(
        self, 
        seir_weight: float = 0.5, 
        lstm_weight: float = 0.5,
        lookback: int = 14
    ):
        self.seir_weight = seir_weight
        self.lstm_weight = lstm_weight
        self.lookback = lookback
        
        # Models
        self.seir_models: Dict[str, SEIRModel] = {}
        self.lstm_forecaster = LSTMForecaster(lookback=lookback)
        self.preprocessor = DataPreprocessor(lookback=lookback)
        self.feature_cols: List[str] = []
        self.target_col = "confirmed_cases"

    def fit(
        self, 
        df_train: pd.DataFrame, 
        feature_cols: List[str], 
        epochs: int = 40, 
        batch_size: int = 32
    ) -> 'HybridPredictor':
        """
        Fits both the country-specific SEIR models and the global LSTM forecaster.
        """
        self.feature_cols = feature_cols
        
        # 1. Fit country-specific SEIR models
        print("Fitting country-specific SEIR models...")
        for country in df_train["country"].unique():
            country_df = df_train[df_train["country"] == country].sort_values("date")
            population = int(country_df["population"].iloc[0])
            
            # Active cases: confirmed_cases - recovered - deaths
            active_cases = (
                country_df["confirmed_cases"] 
                - country_df["recovered"] 
                - country_df["deaths"]
            ).values
            
            # Fit SEIR
            seir = SEIRModel()
            seir.fit(active_cases, population)
            self.seir_models[country] = seir
            print(f"  Fitted SEIR for {country}: beta={seir.beta:.4f}, gamma={seir.gamma:.4f}, R0={seir.fitted_params['R0']:.2f}")

        # 2. Fit Global LSTM Forecaster
        print("Fitting global LSTM model...")
        # Fit scaler on training data
        scaled_train_df = self.preprocessor.fit_transform(df_train, self.feature_cols)
        
        # Create sequences
        X_train, y_train = self.preprocessor.create_windows(
            scaled_train_df, 
            target_col=self.target_col, 
            feature_cols=self.feature_cols
        )
        
        # Train LSTM
        self.lstm_forecaster.train(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            model_path="models/lstm_model.h5"
        )
        
        return self

    def predict(
        self, 
        df_historical: pd.DataFrame, 
        country: str, 
        days: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates predictions for a specific country for a future number of days.
        Returns:
            seir_preds (unscaled actual count)
            lstm_preds (unscaled actual count)
            hybrid_preds (unscaled actual count)
        """
        country_hist = df_historical[df_historical["country"] == country].sort_values("date")
        
        # --- 1. SEIR Prediction ---
        seir_model = self.seir_models.get(country)
        if seir_model is None:
            # Fallback if country wasn't in training
            seir_preds = np.zeros(days)
        else:
            # Predict active cases
            seir_active = seir_model.predict(days)
            # Reconstruct confirmed cases (approximate using historical recovered ratio,
            # but for consistency we directly predict confirmed cases using cumulative SEIR or active cases)
            # Standard SEIR outputs active cases I(t). Let's return the I(t) predictions.
            # However, for comparing directly, we must ensure they are in the same domain.
            # Let's align all models on predicting "confirmed_cases".
            # For SEIR, cumulative infected is C(t) = E(t) + I(t) + R(t) or I(t) + R(t).
            # Confirmed cases are cumulative.
            # Let's predict new confirmed cases from SEIR:
            # Daily new cases = beta * S * I / N. We can sum this up to get cumulative cases.
            # Let's modify the SEIR output or estimate cumulative cases for comparison.
            # Let's solve SEIR over the forecast period to get the cumulative cases: C = N - S.
            # S is the Susceptible population, so N - S represents the cumulative number of individuals infected!
            # Since S decreases, N - S increases monotonically, which is exactly the cumulative cases!
            # Let's verify this: S(t) is Susceptible, so cumulative cases = N - S(t).
            # This is mathematically beautiful and matches confirmed_cases exactly!
            # Let's compute this for SEIR prediction:
            t = np.arange(days + 1)
            solution = seir_model._solve(
                t, 
                tuple(seir_model.last_state), 
                seir_model.N, 
                seir_model.beta, 
                seir_model.sigma, 
                seir_model.gamma
            )
            S_pred = solution[1:, 0]
            seir_preds = seir_model.N - S_pred
            
        # --- 2. LSTM Prediction ---
        # Get the last lookback days of features for the country
        scaled_hist = self.preprocessor.transform_scaler(country_hist)
        last_lookback_features = scaled_hist[self.feature_cols].values[-self.lookback:]
        
        # Find index of target col in feature list
        target_idx = self.feature_cols.index(self.target_col)
        
        # Predict scaled values recursively
        lstm_scaled_preds = self.lstm_forecaster.predict_recursive(
            initial_sequence=last_lookback_features,
            days=days,
            scaler=self.preprocessor.scaler,
            feature_cols=self.feature_cols,
            target_col_idx=target_idx
        )
        
        # Invert scaling
        # To invert scaling, we reconstruct dummy rows since scaler is fitted on multiple features
        dummy_rows = np.zeros((days, len(self.feature_cols)))
        # Put predicted scaled confirmed cases in the target column
        dummy_rows[:, target_idx] = lstm_scaled_preds
        
        lstm_preds_inverted = self.preprocessor.scaler.inverse_transform(dummy_rows)[:, target_idx]
        
        # Ensure predictions are positive and monotonically non-decreasing (confirmed cases can't drop)
        lstm_preds = np.maximum.accumulate(np.maximum(0, lstm_preds_inverted))
        
        # --- 3. Hybrid Prediction ---
        hybrid_preds = (self.seir_weight * seir_preds) + (self.lstm_weight * lstm_preds)
        
        return seir_preds, lstm_preds, hybrid_preds

    def compare_models(
        self, 
        y_true: np.ndarray, 
        seir_pred: np.ndarray, 
        lstm_pred: np.ndarray, 
        hybrid_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Generates comparison metrics for all models.
        """
        metrics = {}
        for name, pred in [("SEIR", seir_pred), ("LSTM", lstm_pred), ("Hybrid", hybrid_pred)]:
            metrics[name] = self.lstm_forecaster.evaluate(y_true, pred)
        return metrics

    def generate_comparison_chart(
        self, 
        dates: List[str], 
        y_true: np.ndarray, 
        seir_pred: np.ndarray, 
        lstm_pred: np.ndarray, 
        hybrid_pred: np.ndarray,
        country: str
    ) -> go.Figure:
        """
        Creates an interactive Plotly chart comparing model predictions against ground truth.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=y_true, mode='lines+markers', name='Actual Cases',
            line=dict(color='black', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=seir_pred, mode='lines', name='SEIR Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=lstm_pred, mode='lines', name='LSTM Forecast',
            line=dict(color='#1f77b4', dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=hybrid_pred, mode='lines', name='Hybrid Forecast (SEIR+LSTM)',
            line=dict(color='#2ca02c', width=3)
        ))
        
        fig.update_layout(
            title=f"Pathogen Spread Model Comparison - {country}",
            xaxis_title="Date",
            yaxis_title="Confirmed Cases",
            template="plotly_dark",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        
        return fig
