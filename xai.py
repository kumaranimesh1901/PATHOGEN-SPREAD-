import os
import argparse
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import joblib
from model import PathogenLSTM


def main(args):
    """Perform XAI analysis on PathogenLSTM using SHAP GradientExplainer."""
    try:
        # Step 1: Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Step 2: Load X_test, take first 100 samples
        X_test = np.load(args.X_test)
        X_test_subset = X_test[:100]

        # Step 3: Load model
        model = PathogenLSTM(input_size=4, hidden_size=128)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.eval()

        # Step 4: Prepare background and test data tensors
        background = torch.tensor(X_test_subset[:50], dtype=torch.float32)
        test_data = torch.tensor(X_test_subset[50:100], dtype=torch.float32)

        # Step 5: Create SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background)

        # Step 6: Compute SHAP values
        shap_values = explainer.shap_values(test_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Step 7: Mean over time axis
        shap_2d = np.mean(np.abs(shap_values), axis=1)

        # Step 8: Feature names
        feature_names = ["new_cases", "new_deaths", "total_cases", "total_deaths"]

        # Step 9: Print mean SHAP values
        mean_shap = shap_2d.mean(axis=0)
        for name, val in zip(feature_names, mean_shap):
            print(f"Feature: {name:<15} | Mean |SHAP|: {val:.6f}")

        # Step 10: Bar plot
        plt.figure(figsize=(8, 5))
        plt.barh(feature_names, mean_shap, color="steelblue")
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Feature Importance — Pathogen Spread Model")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "shap_plot.png"))
        plt.close()

        # Step 11: Print completion
        print("XAI analysis complete. Plot saved.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP-based XAI for PathogenLSTM")
    parser.add_argument("--model", type=str, default="models/lstm_pathogen.pt", help="Path to model checkpoint")
    parser.add_argument("--X_test", type=str, default="data/X_test.npy", help="Path to X_test.npy")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for SHAP plot")
    args = parser.parse_args()
    main(args)
