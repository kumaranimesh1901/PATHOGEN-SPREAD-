import os
import argparse
import numpy as np
import pandas as pd


def confidence_meter(predictions: np.ndarray) -> float:
    """Compute model confidence from an array of predictions. Returns 1 - (std/mean) clipped to [0,1]. If mean is 0 returns 0.0."""
    try:
        mean_val = np.mean(predictions)
        if mean_val == 0:
            return 0.0
        confidence = 1.0 - (np.std(predictions) / mean_val)
        return float(np.clip(confidence, 0.0, 1.0))
    except Exception as e:
        print(f"An error occurred in confidence_meter: {e}")
        return 0.0


def ethical_risk_score(prediction: float, confidence: float, region_population: float) -> dict:
    """Compute ethical risk score and public safety alert. prediction=predicted new cases, confidence=model confidence 0-1, region_population=total population of region. Returns dict with risk_level, confidence_adjusted_score, public_alert_message."""
    try:
        proportion = prediction / region_population
        if proportion < 0.001:
            base_risk = 1
        elif proportion < 0.01:
            base_risk = 2
        elif proportion < 0.05:
            base_risk = 3
        else:
            base_risk = 4

        adjusted_score = round(base_risk * confidence, 2)

        if adjusted_score < 1.0:
            risk_level = "LOW"
        elif adjusted_score < 2.0:
            risk_level = "MEDIUM"
        elif adjusted_score < 3.0:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        action = "Immediate action recommended." if risk_level == "CRITICAL" else "Monitor situation closely."
        message = f"Predicted new cases: {int(prediction)}. Model confidence: {confidence*100:.1f}%. Public risk level: {risk_level}. {action}"

        return {
            "risk_level": risk_level,
            "confidence_adjusted_score": adjusted_score,
            "public_alert_message": message,
        }
    except Exception as e:
        print(f"An error occurred in ethical_risk_score: {e}")
        return {
            "risk_level": "UNKNOWN",
            "confidence_adjusted_score": 0.0,
            "public_alert_message": f"Error: {e}",
        }


def run_ethics_report(predictions_csv: str) -> None:
    """Load predictions CSV with columns predicted, actual, confidence, population. Run ethical_risk_score for each row. Print results table. Save to outputs/ethics_report.csv."""
    try:
        os.makedirs("outputs", exist_ok=True)
        df = pd.read_csv(predictions_csv)

        df["risk_level"] = df.apply(
            lambda row: ethical_risk_score(row["predicted"], row["confidence"], row["population"])["risk_level"],
            axis=1,
        )
        df["alert_message"] = df.apply(
            lambda row: ethical_risk_score(row["predicted"], row["confidence"], row["population"])["public_alert_message"],
            axis=1,
        )

        print(df[["predicted", "actual", "confidence", "risk_level"]].to_string())
        df.to_csv("outputs/ethics_report.csv", index=False)
    except Exception as e:
        print(f"An error occurred in run_ethics_report: {e}")


if __name__ == "__main__":
    try:
        print("=== Ethical Risk Score Demo ===")

        ex1 = ethical_risk_score(500, 0.90, 1000000)
        print("Example 1 (prediction=500, confidence=0.90, population=1000000):")
        print(ex1)

        ex2 = ethical_risk_score(25000, 0.75, 1000000)
        print("\nExample 2 (prediction=25000, confidence=0.75, population=1000000):")
        print(ex2)

        ex3 = ethical_risk_score(80000, 0.60, 1000000)
        print("\nExample 3 (prediction=80000, confidence=0.60, population=1000000):")
        print(ex3)
    except Exception as e:
        print(f"An error occurred: {e}")
