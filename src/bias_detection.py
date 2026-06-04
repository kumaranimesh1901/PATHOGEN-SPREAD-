import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple

class BiasAuditor:
    """
    Bias Detection Engine using Fairlearn to audit forecasting models
    across demographic groups (countries or density buckets).
    """
    def __init__(self):
        self.metric_frame: Optional[MetricFrame] = None
        self.bias_score: float = 0.0
        self.bias_category: str = "Low Bias"

    def audit(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        sensitive_features: np.ndarray,
        sensitive_name: str = "country"
    ) -> Dict[str, Any]:
        """
        Audits predictions across sensitive groups and computes fairness metrics.
        """
        # Define the regression metrics to compute per group
        metrics = {
            "mae": mean_absolute_error,
            "rmse": lambda y, pred: np.sqrt(mean_squared_error(y, pred)),
            "r2": r2_score,
            "mean_pred": lambda y, pred: float(np.mean(pred))
        }
        
        # Build Fairlearn MetricFrame
        self.metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        group_metrics = self.metric_frame.by_group
        
        # Calculate Demographic Parity Difference (DPD) for regression:
        # Diff in average prediction between highest and lowest group
        mean_preds = group_metrics["mean_pred"]
        dp_difference = float(mean_preds.max() - mean_preds.min())
        
        # Calculate Equalized Odds / Performance Disparity:
        # Diff in error metric (MAE) between highest and lowest group
        maes = group_metrics["mae"]
        mae_difference = float(maes.max() - maes.min())
        
        # Calculate overall Bias Score (0 to 100) based on coefficient of variation of group MAEs
        # Higher variation in error across countries means high unfairness / performance bias
        mean_mae = maes.mean()
        std_mae = maes.std()
        if mean_mae > 0:
            coef_var = (std_mae / mean_mae) * 100.0
            self.bias_score = min(100.0, float(coef_var))
        else:
            self.bias_score = 0.0
            
        # Classify bias level
        if self.bias_score < 15.0:
            self.bias_category = "Low Bias"
        elif self.bias_score <= 30.0:
            self.bias_category = "Medium Bias"
        else:
            self.bias_category = "High Bias"
            
        return {
            "group_metrics": group_metrics.to_dict(orient="index"),
            "demographic_parity_difference": dp_difference,
            "mae_difference": mae_difference,
            "bias_score": self.bias_score,
            "bias_category": self.bias_category,
            "sensitive_feature": sensitive_name
        }

    def generate_fairness_chart(self, audit_results: Dict[str, Any]) -> go.Figure:
        """
        Creates a bar chart using Plotly showing MAE and RMSE across groups.
        """
        group_data = audit_results["group_metrics"]
        groups = list(group_data.keys())
        maes = [group_data[g]["mae"] for g in groups]
        rmses = [group_data[g]["rmse"] for g in groups]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=groups, y=maes, name='Mean Absolute Error (MAE)',
            marker_color='#e74c3c'
        ))
        
        fig.add_trace(go.Bar(
            x=groups, y=rmses, name='Root Mean Squared Error (RMSE)',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title=f"Prediction Error by Group ({audit_results['sensitive_feature'].capitalize()})",
            xaxis_title=audit_results['sensitive_feature'].capitalize(),
            yaxis_title="Error Count",
            barmode='group',
            template="plotly_dark",
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        
        return fig
