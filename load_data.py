import os
import pandas as pd
from datasets import load_dataset


def main():
    """Load COVID-19 dataset from Hugging Face, filter columns, clean rows, and save to CSV."""
    try:
        # Step 1: Create data directory
        os.makedirs("data", exist_ok=True)

        # Step 2: Load dataset from Hugging Face
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("maharshipandya/covid-19-coronavirus-pandemic-dataset")

        # Step 3: Convert the "train" split to pandas DataFrame
        df = pd.DataFrame(dataset["train"])

        # Step 4: Keep only these columns
        keep_cols = ["date", "new_cases", "new_deaths", "total_cases", "total_deaths", "continent"]
        df = df[keep_cols]

        # Step 5: Drop rows where new_cases is null or less than 0
        df = df.dropna(subset=["new_cases"])
        df = df[df["new_cases"] >= 0]

        # Step 6: Reset index
        df = df.reset_index(drop=True)

        # Step 7: Save to data/covid_raw.csv
        df.to_csv("data/covid_raw.csv", index=False)

        # Step 8: Print shape and head
        print("Dataset shape:", df.shape)
        print(df.head())
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
