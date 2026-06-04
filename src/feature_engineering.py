import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    """
    Class for generating time-series features: lags, rolling statistics, and rates of change.
    """
    def __init__(self, target_col: str = "confirmed_cases"):
        self.target_col = target_col

    def add_lags(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Adds lag features for the target column, grouped by country.
        """
        df_feats = df.copy()
        for lag in lags:
            df_feats[f"{self.target_col}_lag_{lag}"] = df_feats.groupby("country")[self.target_col].shift(lag)
        return df_feats

    def add_rolling_stats(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Adds rolling mean and standard deviation for the target column, grouped by country.
        """
        df_feats = df.copy()
        for window in windows:
            # Group by country to avoid mixing temporal features between countries
            df_feats[f"{self.target_col}_roll_mean_{window}"] = (
                df_feats.groupby("country")[self.target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df_feats[f"{self.target_col}_roll_std_{window}"] = (
                df_feats.groupby("country")[self.target_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            )
        return df_feats

    def add_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates growth rate and daily new cases/deaths features.
        """
        df_feats = df.copy()
        
        # Daily change in confirmed cases (daily new cases)
        df_feats["daily_new_cases"] = (
            df_feats.groupby("country")[self.target_col]
            .diff()
            .fillna(0)
            .astype(int)
        )
        
        # Growth factor: daily_new_cases(t) / (daily_new_cases(t-1) + 1)
        df_feats["growth_factor"] = (
            df_feats.groupby("country")["daily_new_cases"]
            .transform(lambda x: x / (x.shift(1) + 1.0))
            .fillna(1.0)
        )
        
        return df_feats

    def transform(self, df: pd.DataFrame, lags: List[int] = [1, 2, 7, 14], windows: List[int] = [7, 14]) -> pd.DataFrame:
        """
        Runs the feature engineering pipeline.
        """
        df_processed = df.copy()
        
        # Add basic time-series components
        df_processed = self.add_lags(df_processed, lags)
        df_processed = self.add_rolling_stats(df_processed, windows)
        df_processed = self.add_growth_rates(df_processed)
        
        # Sort and clean
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        df_processed = df_processed.sort_values(by=['country', 'date']).reset_index(drop=True)
        
        return df_processed
