import os
import pandas as pd
import ssl


def main():
    """Load COVID-19 dataset from Our World in Data (OWID), filter columns, clean rows, and save to CSV."""
    try:
        # Step 1: Create data directory
        os.makedirs("data", exist_ok=True)

        # Bypass SSL verification if running in an environment without configured certs (common on macOS)
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass

        # Step 2: Load dataset from Our World in Data (OWID)
        urls = [
            "https://covid.ourworldindata.org/data/owid-covid-data.csv",
            "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        ]

        df = None
        for i, url in enumerate(urls, 1):
            try:
                print(f"Trying to load dataset (source {i}/{len(urls)}): {url}...")
                df = pd.read_csv(url)
                print("Successfully loaded dataset.")
                break
            except Exception as e:
                print(f"Failed to load from source {i}: {e}")

        if df is None:
            raise RuntimeError(
                "Failed to download the dataset from all available sources. "
                "Please verify your internet connection and DNS settings."
            )

        # Step 3: Keep only these columns
        keep_cols = ["date", "new_cases", "new_deaths", "total_cases", "total_deaths", "continent"]
        df = df[keep_cols]

        # Step 4: Drop rows where new_cases/continent is null or less than 0
        df = df.dropna(subset=["new_cases", "continent"])
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
