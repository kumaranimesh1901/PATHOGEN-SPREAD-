import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from typing import Dict, Tuple, List, Optional, Union

class LSTMForecaster:
    """
    Deep learning-based time-series forecaster using a multi-layer LSTM architecture.
    """
    def __init__(self, lookback: int = 14, lstm_units: List[int] = [64, 32], dropout_rate: float = 0.2, learning_rate: float = 0.001):
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: Optional[Sequential] = None
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Constructs a multi-layer LSTM model.
        input_shape should be (lookback, num_features).
        """
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1) # True for all but last LSTM layer
            
            if i == 0:
                model.add(LSTM(units, return_sequences=return_seq, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_seq))
                
            model.add(Dropout(self.dropout_rate))
            
        # Output layer for single step regression
        model.add(Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        self.model = model
        return model

    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        epochs: int = 50, 
        batch_size: int = 16, 
        validation_split: float = 0.1, 
        model_path: str = "models/lstm_model.h5"
    ) -> tf.keras.callbacks.History:
        """
        Trains the LSTM model with Early Stopping and Model Checkpointing.
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
            
        # Create directories for saving model if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True, verbose=0)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        # Load best weights
        if os.path.exists(model_path):
            self.load_model(model_path)
            
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for a given 3D feature array X.
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions.")
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def predict_recursive(
        self, 
        initial_sequence: np.ndarray, 
        days: int, 
        scaler, 
        feature_cols: List[str], 
        target_col_idx: int = 0
    ) -> np.ndarray:
        """
        Performs recursive forecasting for multiple days ahead.
        Uses predictions as inputs for subsequent steps.
        
        initial_sequence: array of shape (lookback, num_features)
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions.")
            
        current_seq = initial_sequence.copy()
        predictions = []
        
        for _ in range(days):
            # Reshape sequence for prediction: (1, lookback, num_features)
            pred_input = np.expand_dims(current_seq, axis=0)
            pred_scaled = self.model.predict(pred_input, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            
            # Slide sequence window
            new_row = current_seq[-1].copy()
            # Update target column in feature vector with the new prediction
            new_row[target_col_idx] = pred_scaled
            
            # Roll sequence and insert new row
            current_seq = np.vstack([current_seq[1:], new_row])
            
        return np.array(predictions)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes evaluation metrics: MAE, RMSE, and R2.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Avoid division by zero in MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape)
        }

    def save_model(self, model_path: str) -> None:
        """
        Saves the model to the specified path.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Loads a pre-trained model.
        """
        self.model = load_model(model_path)
