import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

class EthicalRiskScorer:
    """
    Ethical Risk Scorer combining outbreak forecasts, density, prediction confidence,
    and healthcare capacities into a unified 0-100 risk scorecard.
    """
    def __init__(
        self, 
        w_severity: float = 0.40, 
        w_density: float = 0.20, 
        w_confidence: float = 0.15, 
        w_capacity: float = 0.25
    ):
        self.w_severity = w_severity
        self.w_density = w_density
        self.w_confidence = w_confidence
        self.w_capacity = w_capacity

    def calculate_score(
        self, 
        predicted_cases: float, 
        population: int, 
        population_density: float, 
        prediction_confidence: float,  # e.g., 0.90 for 90% confidence
        healthcare_capacity: float      # beds per 1000 people
    ) -> Dict[str, Any]:
        """
        Calculates the 0-100 Ethical Outbreak Risk Score.
        """
        # 1. Outbreak Severity (predicted cases relative to population)
        # Normalize: 1% of population infected in a day is an extreme outbreak (score = 100)
        infection_ratio = predicted_cases / max(1, population)
        severity_score = min(100.0, (infection_ratio / 0.01) * 100.0)
        
        # 2. Population Density (people per sq km)
        # Normalize: log scale, 500+ density is very high risk (score = 100)
        density_score = min(100.0, (np.log10(max(1.0, population_density)) / np.log10(500.0)) * 100.0)
        
        # 3. Prediction Confidence (uncertainty penalty)
        # Low confidence = higher risk because we must act conservatively
        confidence_penalty = 100.0 * (1.0 - prediction_confidence)
        
        # 4. Healthcare Capacity Stress (beds per 1000 people)
        # Lower capacity = higher vulnerability risk
        # A capacity of 8.0 beds/1000 is high (score = 0 risk), 0.5 beds/1000 is low (score = 100 risk)
        capacity_score = max(0.0, 100.0 - (healthcare_capacity * 12.5))
        
        # Weighted Risk Score
        raw_score = (
            (self.w_severity * severity_score) +
            (self.w_density * density_score) +
            (self.w_confidence * confidence_penalty) +
            (self.w_capacity * capacity_score)
        )
        
        risk_score = round(min(100.0, max(0.0, raw_score)), 1)
        
        # Determine risk level
        if risk_score <= 30.0:
            level = "Low"
            color = "#2ecc71" # Green
        elif risk_score <= 60.0:
            level = "Medium"
            color = "#f39c12" # Orange
        else:
            level = "High"
            color = "#e74c3c" # Red
            
        return {
            "risk_score": risk_score,
            "risk_level": level,
            "risk_color": color,
            "breakdown": {
                "outbreak_severity": round(severity_score, 1),
                "population_density": round(density_score, 1),
                "confidence_penalty": round(confidence_penalty, 1),
                "healthcare_capacity_vulnerability": round(capacity_score, 1)
            }
        }

    def generate_gauge_chart(self, scorer_results: Dict[str, Any], country: str) -> go.Figure:
        """
        Creates a beautiful interactive Plotly gauge meter for the risk score.
        """
        score = scorer_results["risk_score"]
        level = scorer_results["risk_level"]
        color = scorer_results["risk_color"]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': f"Ethical Outbreak Risk Score - {country}", 'font': {'size': 20}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.15)'},
                    {'range': [30, 60], 'color': 'rgba(243, 156, 18, 0.15)'},
                    {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
