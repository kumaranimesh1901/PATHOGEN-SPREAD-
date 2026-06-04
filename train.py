import os
import json
import joblib
import pandas as pd
import numpy as np
from src.utils import generate_synthetic_data
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.seir_model import SEIRModel
from src.lstm_model import LSTMForecaster
from src.hybrid_model import HybridPredictor
from src.explainability import ExplainabilityEngine
from src.bias_detection import BiasAuditor
from src.ethical_risk_score import EthicalRiskScorer
from src.alert_system import PublicSafetyAlert

def run_pipeline():
    print("=" * 60)
    print("STARTING OUTBREAK FORECASTING PIPELINE")
    print("=" * 60)
    
    # Paths
    raw_data_path = "data/raw/covid_data.csv"
    processed_dir = "data/processed"
    models_dir = "models"
    explainability_dir = "models/explainability"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(explainability_dir, exist_ok=True)
    
    # 1. Generate Dataset
    if not os.path.exists(raw_data_path):
        print("Raw dataset not found. Generating synthetic dataset...")
        df_raw = generate_synthetic_data(raw_data_path, num_days=180)
    else:
        print(f"Loading existing raw dataset from {raw_data_path}...")
        df_raw = pd.read_csv(raw_data_path)
        
    # 2. Feature Engineering
    print("Running Feature Engineering...")
    engineer = FeatureEngineer(target_col="confirmed_cases")
    df_engineered = engineer.transform(df_raw)
    
    # Drop rows with NaN values resulting from shift/lags
    df_clean = df_engineered.dropna().reset_index(drop=True)
    df_clean.to_csv(os.path.join(processed_dir, "engineered_features.csv"), index=False)
    print(f"Feature engineering done. Dataset shape: {df_clean.shape}")
    
    # Define features for scaling and model input
    feature_cols = [
        "confirmed_cases", "deaths", "recovered", "population_density", 
        "healthcare_capacity", "mobility_index",
        "confirmed_cases_lag_1", "confirmed_cases_lag_2", "confirmed_cases_lag_7", "confirmed_cases_lag_14",
        "confirmed_cases_roll_mean_7", "confirmed_cases_roll_mean_14",
        "confirmed_cases_roll_std_7", "confirmed_cases_roll_std_14",
        "daily_new_cases", "growth_factor"
    ]
    
    # 3. Train-Test Split (Temporal)
    print("Performing chronological train-test split (80/20)...")
    preprocessor = DataPreprocessor(lookback=14, target_col="confirmed_cases")
    df_train, df_test = preprocessor.train_test_split_temporal(df_clean, train_ratio=0.8)
    
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
    
    # Save split datasets
    df_train.to_csv(os.path.join(processed_dir, "train_split.csv"), index=False)
    df_test.to_csv(os.path.join(processed_dir, "test_split.csv"), index=False)
    
    # 4. Fit Hybrid Model
    # Instantiate HybridPredictor (default weights: 0.5 SEIR, 0.5 LSTM)
    hybrid_predictor = HybridPredictor(seir_weight=0.5, lstm_weight=0.5, lookback=14)
    hybrid_predictor.fit(df_train, feature_cols, epochs=15, batch_size=32)
    
    # Save the scaler
    hybrid_predictor.preprocessor.save_scaler(os.path.join(models_dir, "scaler.joblib"))
    
    # 5. Evaluate and Generate Test Predictions
    print("Evaluating models on test set...")
    countries = df_test["country"].unique()
    
    test_results = []
    comparison_metrics = {}
    
    # We will evaluate prediction on the full test set length per country
    for country in countries:
        country_test = df_test[df_test["country"] == country].sort_values("date")
        test_days = len(country_test)
        
        # We need historical data (the end of the train set) to feed the lookback window
        country_train = df_train[df_train["country"] == country].sort_values("date")
        
        # Predict
        seir_pred, lstm_pred, hybrid_pred = hybrid_predictor.predict(
            df_historical=country_train,
            country=country,
            days=test_days
        )
        
        # True values
        y_true = country_test["confirmed_cases"].values
        dates = country_test["date"].astype(str).tolist()
        
        # Compare
        country_metrics = hybrid_predictor.compare_models(y_true, seir_pred, lstm_pred, hybrid_pred)
        comparison_metrics[country] = country_metrics
        
        # Build predictions dataframe
        for idx in range(test_days):
            test_results.append({
                "date": dates[idx],
                "country": country,
                "actual_cases": int(y_true[idx]),
                "seir_cases": int(seir_pred[idx]),
                "lstm_cases": int(lstm_pred[idx]),
                "hybrid_cases": int(hybrid_pred[idx]),
                "population": country_test["population"].iloc[idx],
                "population_density": country_test["population_density"].iloc[idx],
                "healthcare_capacity": country_test["healthcare_capacity"].iloc[idx],
                "mobility_index": country_test["mobility_index"].iloc[idx]
            })
            
    df_predictions = pd.DataFrame(test_results)
    df_predictions.to_csv(os.path.join(processed_dir, "test_predictions.csv"), index=False)
    print("Test predictions saved.")
    
    # Save comparison metrics report
    with open(os.path.join(processed_dir, "metrics_report.json"), "w") as f:
        json.dump(comparison_metrics, f, indent=4)
    print("Evaluation metrics report saved.")
    
    # 6. SHAP Explainability Engine
    # Fit XAI on scaled train inputs and generate plots
    scaled_train_df = hybrid_predictor.preprocessor.transform_scaler(df_train)
    X_train_seq, y_train_seq = hybrid_predictor.preprocessor.create_windows(
        scaled_train_df, 
        target_col="confirmed_cases", 
        feature_cols=feature_cols
    )
    
    scaled_test_df = hybrid_predictor.preprocessor.transform_scaler(df_test)
    X_test_seq, _ = hybrid_predictor.preprocessor.create_windows(
        scaled_test_df, 
        target_col="confirmed_cases", 
        feature_cols=feature_cols
    )
    
    xai_engine = ExplainabilityEngine()
    xai_engine.fit(X_train_seq, y_train_seq, feature_cols)
    xai_engine.generate_plots(X_test_seq, feature_cols, explainability_dir)
    
    # 7. Bias Detection using Fairlearn
    print("Running Fairlearn Bias Audit...")
    y_true_all = df_predictions["actual_cases"].values
    y_pred_all = df_predictions["hybrid_cases"].values
    sensitive_features = df_predictions["country"].values
    
    auditor = BiasAuditor()
    bias_results = auditor.audit(y_true_all, y_pred_all, sensitive_features, sensitive_name="country")
    
    with open(os.path.join(processed_dir, "bias_report.json"), "w") as f:
        json.dump(bias_results, f, indent=4)
    print(f"Bias Audit complete. Bias Score: {bias_results['bias_score']:.2f} ({bias_results['bias_category']})")
    
    # 8. Ethical Risk Scoring and Safety Alerts
    print("Calculating ethical risk scores and triggering safety alerts...")
    risk_scorer = EthicalRiskScorer()
    alert_engine = PublicSafetyAlert()
    
    alerts_data = {}
    
    # Calculate for the last day of the test prediction (most recent forecast)
    for country in countries:
        country_pred = df_predictions[df_predictions["country"] == country].sort_values("date").iloc[-1]
        
        # Use average R2 score as a representation of confidence
        country_r2 = comparison_metrics[country]["Hybrid"]["R2"]
        confidence = max(0.1, min(0.99, country_r2)) # Clamp between 10% and 99%
        
        # Calculate risk score
        risk_res = risk_scorer.calculate_score(
            predicted_cases=country_pred["hybrid_cases"],
            population=int(country_pred["population"]),
            population_density=float(country_pred["population_density"]),
            prediction_confidence=confidence,
            healthcare_capacity=float(country_pred["healthcare_capacity"])
        )
        
        # Generate alert
        alert_res = alert_engine.generate_alert(
            country=country,
            predicted_cases=country_pred["hybrid_cases"],
            confidence=confidence,
            risk_score=risk_res["risk_score"]
        )
        
        alerts_data[country] = {
            "risk_score": risk_res["risk_score"],
            "risk_level": risk_res["risk_level"],
            "risk_color": risk_res["risk_color"],
            "breakdown": risk_res["breakdown"],
            "threat_level": alert_res["threat_level"],
            "alert_color": alert_res["color"],
            "protocols": alert_res["protocols"],
            "predicted_cases": alert_res["predicted_cases"],
            "confidence": alert_res["confidence"]
        }
        
    with open(os.path.join(processed_dir, "safety_alerts.json"), "w") as f:
        json.dump(alerts_data, f, indent=4)
        
    print("Ethical risk scores and public safety alerts generated.")
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
