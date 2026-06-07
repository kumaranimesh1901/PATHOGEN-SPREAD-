"""
SQLite database module for the PATHOGEN-SPREAD forecasting system.
Schema matches the actual india_covid_catboost_dataset.csv columns exactly.
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd

# Path to SQLite database file
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pathogen.db")
)


def init_db() -> None:
    """
    Create SQLite tables if they do not exist.

    Tables:
      - disease_records: historical data matching the CSV schema exactly
      - forecast_results: stores completed forecast runs
    """
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease_records (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            date                  TEXT,
            state                 TEXT,
            new_cases             INTEGER,
            new_deaths            INTEGER,
            new_vaccinations      INTEGER,
            tests_performed       INTEGER,
            positive_rate         REAL,
            hospitalized_patients INTEGER,
            icu_patients          INTEGER,
            mobility_index        REAL,
            temperature           REAL,
            humidity              REAL,
            population_density    INTEGER
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecast_results (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at          TEXT NOT NULL,
            state               TEXT NOT NULL,
            engine_used         TEXT NOT NULL,
            risk_category       TEXT NOT NULL,
            risk_score          REAL NOT NULL,
            cases_7d            INTEGER NOT NULL,
            cases_14d           INTEGER NOT NULL,
            cases_30d           INTEGER NOT NULL,
            deaths_7d           INTEGER NOT NULL,
            deaths_14d          INTEGER NOT NULL,
            deaths_30d          INTEGER NOT NULL,
            peak_day            INTEGER,
            reproduction_number REAL,
            growth_rate         REAL NOT NULL
        )
        """)

        conn.commit()
    except Exception as e:
        print(f"Database initialization failed: {e}")
        raise
    finally:
        conn.close()


def load_csv_to_db(csv_path: str) -> None:
    """
    Load the raw CSV into disease_records as-is.
    Skips if the table already has rows.
    """
    init_db()

    if not os.path.exists(csv_path):
        print(f"Skipping CSV import: file not found at {csv_path}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM disease_records")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"Database already has {count} records. Skipping import.")
            return

        df = pd.read_csv(csv_path)

        # The 13 columns that exist in the CSV
        csv_cols = [
            "date", "state", "new_cases", "new_deaths", "new_vaccinations",
            "tests_performed", "positive_rate", "hospitalized_patients",
            "icu_patients", "mobility_index", "temperature", "humidity",
            "population_density",
        ]

        # Validate all columns exist
        for col in csv_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from CSV")

        df = df[csv_cols]

        # Cast to native Python types for sqlite3
        df["date"] = df["date"].astype(str)
        df["state"] = df["state"].astype(str)
        df["new_cases"] = df["new_cases"].astype(int)
        df["new_deaths"] = df["new_deaths"].astype(int)
        df["new_vaccinations"] = df["new_vaccinations"].astype(int)
        df["tests_performed"] = df["tests_performed"].astype(int)
        df["positive_rate"] = df["positive_rate"].astype(float)
        df["hospitalized_patients"] = df["hospitalized_patients"].astype(int)
        df["icu_patients"] = df["icu_patients"].astype(int)
        df["mobility_index"] = df["mobility_index"].astype(float)
        df["temperature"] = df["temperature"].astype(float)
        df["humidity"] = df["humidity"].astype(float)
        df["population_density"] = df["population_density"].astype(int)

        placeholders = ", ".join(["?"] * len(csv_cols))
        col_names = ", ".join(csv_cols)
        query = f"INSERT INTO disease_records ({col_names}) VALUES ({placeholders})"

        records = [tuple(row) for row in df.values.tolist()]
        cursor.executemany(query, records)
        conn.commit()
        print(f"Dataset loaded: {len(records)} rows")

    except Exception as e:
        print(f"Failed to load CSV: {e}")
        raise
    finally:
        conn.close()


def get_state_names() -> list:
    """Return sorted list of unique state names in the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT state FROM disease_records ORDER BY state ASC")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Failed to query state names: {e}")
        return []
    finally:
        conn.close()


def get_state_data(state_name: str) -> pd.DataFrame:
    """
    Retrieve all historical rows for a given state, sorted chronologically.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT * FROM disease_records WHERE state = ? ORDER BY date ASC"
        df = pd.read_sql_query(query, conn, params=(state_name,))
        return df
    except Exception as e:
        print(f"Failed to fetch data for state '{state_name}': {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_row_count(state_name: str) -> int:
    """Get the count of rows stored for a specific state."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM disease_records WHERE state = ?", (state_name,)
        )
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Failed to get row count for state '{state_name}': {e}")
        return 0
    finally:
        conn.close()


def save_forecast(result: dict) -> None:
    """Insert a completed forecast into the forecast_results table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO forecast_results (
                created_at, state, engine_used, risk_category, risk_score,
                cases_7d, cases_14d, cases_30d,
                deaths_7d, deaths_14d, deaths_30d,
                peak_day, reproduction_number, growth_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                result.get("state", ""),
                result.get("engine_used", ""),
                result.get("risk_category", ""),
                float(result.get("risk_score", 0.0)),
                int(result.get("cases_7d", 0)),
                int(result.get("cases_14d", 0)),
                int(result.get("cases_30d", 0)),
                int(result.get("deaths_7d", 0)),
                int(result.get("deaths_14d", 0)),
                int(result.get("deaths_30d", 0)),
                result.get("peak_day"),
                result.get("reproduction_number"),
                float(result.get("growth_rate", 0.0)),
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"Failed to save forecast: {e}")
        raise
    finally:
        conn.close()


def get_history(limit: int = 20) -> list:
    """Retrieve the most recent forecast logs."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT state, created_at, engine_used, risk_category, cases_30d
            FROM forecast_results
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "state": r[0],
                "date": r[1],
                "engine": r[2],
                "risk": r[3],
                "cases_30d": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Failed to query history: {e}")
        return []
    finally:
        conn.close()
