import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Any, Optional
import joblib
import os

class DataPreprocessor:
    """
    Class for handling all preprocessing tasks: missing values, outliers, scaling,
    and sequence windowing for time-series forecasting.
    """
    def __init__(self, lookback: int = 14, target_col: str = "confirmed_cases"):
        self.lookback = lookback
        self.target_col = target_col
        self.scaler = MinMaxScaler()
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values using time-series interpolation and forward/backward filling.
        """
        df_clean = df.copy()
        
        # Sort by country and date to ensure temporal ordering
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.sort_values(by=['country', 'date']).reset_index(drop=True)
        
        # Numeric columns to fill
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Group by country and fill missing values temporally
            df_clean[col] = df_clean.groupby('country')[col].ffill().bfill()
            
        return df_clean

    def detect_and_handle_outliers(
        self, df: pd.DataFrame, col: str, method: str = "iqr", factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Detects outliers in a column and caps them using the IQR or Z-score method.
        """
        df_clean = df.copy()
        
        # Process country-by-country to respect local scales
        for country in df_clean['country'].unique():
            mask = df_clean['country'] == country
            country_data = df_clean.loc[mask, col]
            
            if method == "iqr":
                q1 = country_data.quantile(0.25)
                q3 = country_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (factor * iqr)
                upper_bound = q3 + (factor * iqr)
            elif method == "zscore":
                mean = country_data.mean()
                std = country_data.std()
                lower_bound = mean - (factor * std)
                upper_bound = mean + (factor * std)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
                
            # Cap values at lower and upper bounds
            df_clean.loc[mask, col] = np.clip(country_data, lower_bound, upper_bound)
            
        return df_clean

    def fit_scaler(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        Fits the MinMaxScaler on training features.
        """
        self.feature_cols = feature_cols
        self.scaler.fit(df[feature_cols])
        self.is_fitted = True

    def transform_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies scaling to the dataframe.
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transformation.")
        
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Fits and transforms features in one step.
        """
        self.fit_scaler(df, feature_cols)
        return self.transform_scaler(df)

    def save_scaler(self, filepath: str) -> None:
        """
        Saves the fitted scaler.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        
    def load_scaler(self, filepath: str, feature_cols: List[str]) -> None:
        """
        Loads a pre-fitted scaler.
        """
        self.scaler = joblib.load(filepath)
        self.feature_cols = feature_cols
        self.is_fitted = True

    def create_windows(
        self, df: pd.DataFrame, target_col: str, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates windowed input sequences and targets for training LSTM.
        Ensures windows do not cross between different countries.
        """
        X, y = [], []
        
        for country in df['country'].unique():
            country_df = df[df['country'] == country].sort_values(by='date').reset_index(drop=True)
            
            features = country_df[feature_cols].values
            target = country_df[target_col].values
            
            for i in range(len(country_df) - self.lookback):
                X.append(features[i : i + self.lookback])
                y.append(target[i + self.lookback])
                
        return np.array(X), np.array(y)

    def train_test_split_temporal(
        self, df: pd.DataFrame, train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronologically splits data into train and test sets per country.
        """
        train_dfs = []
        test_dfs = []
        
        for country in df['country'].unique():
            country_df = df[df['country'] == country].sort_values(by='date').reset_index(drop=True)
            split_idx = int(len(country_df) * train_ratio)
            
            train_dfs.append(country_df.iloc[:split_idx])
            test_dfs.append(country_df.iloc[split_idx:])
            
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)
        
        return train_df, test_df
