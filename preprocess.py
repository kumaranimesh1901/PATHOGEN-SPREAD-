import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


def main(args):
    """Preprocess COVID-19 raw data: scale features, create sliding window sequences, and save train/test splits."""
    try:
        # Step 1: Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Step 2: Load CSV, parse date, sort, reset index
        df = pd.read_csv(args.input)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date").reset_index(drop=True)

        # Step 3: Define feature columns
        feature_cols = ["new_cases", "new_deaths", "total_cases", "total_deaths"]

        # Step 4: Drop rows with any null in feature_cols
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        # Step 5: Fit MinMaxScaler, transform, save scaler
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, "models/scaler.pkl")

        # Step 6: Create sliding window sequences
        X = []
        y = []
        seq_len = args.seq_len
        for i in range(len(df) - seq_len):
            X.append(scaled_values[i : i + seq_len])
            y.append(scaled_values[i + seq_len, 0])

        # Step 7: Convert to numpy float32 arrays
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Step 8: Split into train and test
        train_size = int(0.8 * len(X))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        # Get the continents for the test split
        continents = df["continent"].values[seq_len:]
        test_continents = continents[train_size:]

        # Step 9: Save all arrays
        np.save("data/X_train.npy", X_train)
        np.save("data/X_test.npy", X_test)
        np.save("data/y_train.npy", y_train)
        np.save("data/y_test.npy", y_test)
        np.save("data/test_continents.npy", test_continents)

        # Step 10: Print shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape:  {y_test.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess pathogen spread data")
    parser.add_argument("--input", type=str, default="data/covid_raw.csv", help="Path to input raw CSV")
    parser.add_argument("--seq_len", type=int, default=14, help="Sequence length for sliding window")
    args = parser.parse_args()
    main(args)
