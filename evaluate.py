import os
import argparse
import json
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import PathogenLSTM


def main(args):
    """Evaluate the trained PathogenLSTM model: compute metrics, plot predictions, save results."""
    try:
        # Step 1: Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Step 2: Load test data
        X_test = np.load(args.X_test)
        y_test = np.load(args.y_test)

        # Step 3: Load model
        model = PathogenLSTM(input_size=4, hidden_size=128)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.eval()

        # Step 4: Run inference
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            preds_tensor = model(X_tensor)
            preds = preds_tensor.numpy().flatten()

        # Step 5: Inverse transform predictions and actuals
        scaler = joblib.load(args.scaler)

        preds_full = np.zeros((len(preds), 4))
        preds_full[:, 0] = preds
        preds_orig = scaler.inverse_transform(preds_full)[:, 0]

        actuals_full = np.zeros((len(y_test), 4))
        actuals_full[:, 0] = y_test
        actuals_orig = scaler.inverse_transform(actuals_full)[:, 0]

        # Step 6: Compute metrics
        mae = mean_absolute_error(actuals_orig, preds_orig)
        rmse = np.sqrt(mean_squared_error(actuals_orig, preds_orig))
        r2 = r2_score(actuals_orig, preds_orig)

        # Step 7: Print metrics
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

        # Step 8: Save metrics to JSON
        metrics_dict = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # Step 9: Plot actual vs predicted
        plt.figure(figsize=(12, 5))
        plt.plot(actuals_orig, color="blue", label="Actual")
        plt.plot(preds_orig, color="red", label="Predicted")
        plt.title("Actual vs Predicted New Cases")
        plt.xlabel("Days")
        plt.ylabel("New Cases")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "prediction_plot.png"))
        plt.close()

        # Step 10: Print completion
        print("Evaluation complete. Results saved to", args.output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PathogenLSTM model")
    parser.add_argument("--X_test", type=str, default="data/X_test.npy", help="Path to X_test.npy")
    parser.add_argument("--y_test", type=str, default="data/y_test.npy", help="Path to y_test.npy")
    parser.add_argument("--model", type=str, default="models/lstm_pathogen.pt", help="Path to model weights")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl", help="Path to scaler pickle")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    main(args)
