import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def main(args):
    """Audit model predictions for bias across continents by comparing per-group MAE to global MAE."""
    try:
        # Step 1: Create output directory
        os.makedirs("outputs", exist_ok=True)

        # Step 2: Load predictions CSV
        df = pd.read_csv(args.predictions)

        # Step 3: Compute global MAE
        global_mae = mean_absolute_error(df["actual"], df["predicted"])

        # Step 4: Print global MAE
        print(f"Global MAE: {global_mae:.4f}")

        # Step 5: Print header
        print(f"{'Continent':<20} {'MAE':>10} {'Status':>10}")
        print("-" * 42)

        # Step 6: Create results list
        results = []

        # Step 7: Per-continent MAE
        for continent in df["continent"].unique():
            df_cont = df[df["continent"] == continent]
            continent_mae = mean_absolute_error(df_cont["actual"], df_cont["predicted"])
            flag = "BIASED" if continent_mae > 1.5 * global_mae else "OK"
            print(f"{str(continent):<20} {continent_mae:>10.4f} {flag:>10}")
            results.append({"continent": continent, "mae": continent_mae, "flag": flag})

        # Step 8: Fairness ratio
        maes = [r["mae"] for r in results]
        if min(maes) > 0:
            fairness_ratio = max(maes) / min(maes)
        else:
            fairness_ratio = float("inf")
        print(f"Fairness Ratio (max/min MAE): {fairness_ratio:.4f}")

        # Step 9: Save report as plain text
        with open("outputs/bias_report.txt", "w") as f:
            f.write(f"Global MAE: {global_mae:.4f}\n\n")
            f.write(f"{'Continent':<20} {'MAE':>10} {'Status':>10}\n")
            f.write("-" * 42 + "\n")
            for r in results:
                f.write(f"{str(r['continent']):<20} {r['mae']:>10.4f} {r['flag']:>10}\n")
            f.write(f"\nFairness Ratio (max/min MAE): {fairness_ratio:.4f}\n")

        # Step 10: Print completion
        print("Bias audit complete. Report saved to outputs/bias_report.txt")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bias audit for pathogen spread predictions")
    parser.add_argument("--predictions", type=str, default="outputs/predictions.csv", help="Path to predictions CSV")
    args = parser.parse_args()
    main(args)
