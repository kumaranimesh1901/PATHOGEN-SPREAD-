"""
New Disease Ingester
====================
Supports ingesting epidemiological data for arbitrary diseases into
per-disease SQLite tables. Each table mirrors the main ``disease_records``
schema from the PATHOGEN-SPREAD system, with extra columns for
``disease_name`` and ``data_confidence``.

Usage
-----
>>> ing = NewDiseaseIngester("monkeypox", "contact")
>>> ing.ingest_daily("Karnataka", "2024-06-01", 120, 3, hospitalized=18, tests=1100)
>>> df = ing.get_data("Karnataka")
"""

import csv
import os
import re
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

_VALID_PATHOGEN_TYPES = {"respiratory", "vector_borne", "contact", "foodborne"}

_VALID_CONFIDENCE_LEVELS = {"confirmed", "probable", "suspected"}

# Date formats accepted during validation (ISO-8601 date only)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Default DB path — stored alongside main project data
DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "new_diseases.db")
)

# Default epidemiological priors keyed by pathogen type.
# These are rough multipliers used in ``fill_missing_features`` when the
# ingested dataset does not contain certain columns.
_DEFAULT_PRIORS = {
    "respiratory": {
        "test_multiplier": 10,           # tests ≈ new_cases × 10
        "hospitalization_rate": 0.15,    # 15 % of cases hospitalised
        "icu_rate": 0.20,                # 20 % of hospitalised → ICU
        "mobility_index": 60.0,
        "temperature": 25.0,
        "humidity": 50.0,
        "population_density": 1000,
    },
    "vector_borne": {
        "test_multiplier": 8,
        "hospitalization_rate": 0.20,
        "icu_rate": 0.10,
        "mobility_index": 75.0,
        "temperature": 30.0,
        "humidity": 70.0,
        "population_density": 800,
    },
    "contact": {
        "test_multiplier": 6,
        "hospitalization_rate": 0.10,
        "icu_rate": 0.15,
        "mobility_index": 70.0,
        "temperature": 28.0,
        "humidity": 55.0,
        "population_density": 900,
    },
    "foodborne": {
        "test_multiplier": 5,
        "hospitalization_rate": 0.12,
        "icu_rate": 0.08,
        "mobility_index": 80.0,
        "temperature": 27.0,
        "humidity": 60.0,
        "population_density": 1100,
    },
}

# Columns that mirror ``disease_records`` plus the two new ones.
_TABLE_COLUMNS = [
    "date",
    "state",
    "new_cases",
    "new_deaths",
    "new_vaccinations",
    "tests_performed",
    "positive_rate",
    "hospitalized_patients",
    "icu_patients",
    "mobility_index",
    "temperature",
    "humidity",
    "population_density",
    "disease_name",
    "data_confidence",
]


# ──────────────────────────────────────────────────────────
# Helper — safe table-name generation
# ──────────────────────────────────────────────────────────

def _sanitise_table_name(disease_name: str) -> str:
    """Return a safe SQLite table name derived from the disease name."""
    slug = re.sub(r"[^a-z0-9_]", "_", disease_name.lower())
    return f"new_disease_data_{slug}"


# ══════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════

class NewDiseaseIngester:
    """Ingest and query epidemiological data for a new (non-COVID) disease.

    Parameters
    ----------
    disease_name : str
        Human-readable disease identifier (e.g. ``"monkeypox"``).
    pathogen_type : str
        One of ``"respiratory"``, ``"vector_borne"``, ``"contact"``,
        ``"foodborne"``.  Determines default epidemiological priors used
        by :meth:`fill_missing_features`.

    Raises
    ------
    ValueError
        If *pathogen_type* is not one of the four accepted values.
    """

    # ── construction ────────────────────────────────────────

    def __init__(self, disease_name: str, pathogen_type: str) -> None:
        if pathogen_type not in _VALID_PATHOGEN_TYPES:
            raise ValueError(
                f"pathogen_type must be one of {sorted(_VALID_PATHOGEN_TYPES)}, "
                f"got '{pathogen_type}'"
            )

        self.disease_name: str = disease_name
        self.pathogen_type: str = pathogen_type
        self.table_name: str = _sanitise_table_name(disease_name)
        self.priors: dict = _DEFAULT_PRIORS[pathogen_type]

        # Ensure the database and table exist on first use.
        self._init_table()

    # ── public API ──────────────────────────────────────────

    def ingest_daily(
        self,
        state: str,
        date: str,
        new_cases: int,
        new_deaths: int,
        *,
        data_confidence: str = "confirmed",
        **kwargs,
    ) -> None:
        """Append a single day's data to the database.

        Parameters
        ----------
        state : str
            Name of the state / region.
        date : str
            Date string in ``YYYY-MM-DD`` format.
        new_cases : int
            Number of new confirmed cases (must be ≥ 0).
        new_deaths : int
            Number of new deaths (must be ≥ 0, and ≤ *new_cases*).
        data_confidence : str, optional
            One of ``"confirmed"``, ``"probable"``, ``"suspected"``
            (default ``"confirmed"``).
        **kwargs
            Any additional columns from the main schema (e.g.
            ``hospitalized=18``, ``tests=1100``).  Unknown keys are
            silently ignored.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        # ── validation ──
        self._validate_date(date)
        self._validate_non_negative("new_cases", new_cases)
        self._validate_non_negative("new_deaths", new_deaths)

        if new_deaths > new_cases:
            raise ValueError(
                f"new_deaths ({new_deaths}) cannot exceed "
                f"new_cases ({new_cases})"
            )

        if data_confidence not in _VALID_CONFIDENCE_LEVELS:
            raise ValueError(
                f"data_confidence must be one of {sorted(_VALID_CONFIDENCE_LEVELS)}, "
                f"got '{data_confidence}'"
            )

        # Validate any optional numeric kwargs
        for key, value in kwargs.items():
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(
                    f"Optional field '{key}' cannot be negative, got {value}"
                )

        # ── map kwargs → table columns ──
        col_map = {
            "new_vaccinations": kwargs.get("new_vaccinations", kwargs.get("vaccinations", 0)),
            "tests_performed": kwargs.get("tests_performed", kwargs.get("tests", 0)),
            "positive_rate": kwargs.get("positive_rate", 0.0),
            "hospitalized_patients": kwargs.get(
                "hospitalized_patients", kwargs.get("hospitalized", 0)
            ),
            "icu_patients": kwargs.get("icu_patients", kwargs.get("icu", 0)),
            "mobility_index": kwargs.get("mobility_index", 0.0),
            "temperature": kwargs.get("temperature", 0.0),
            "humidity": kwargs.get("humidity", 0.0),
            "population_density": kwargs.get("population_density", 0),
        }

        row = (
            date,
            state,
            int(new_cases),
            int(new_deaths),
            int(col_map["new_vaccinations"]),
            int(col_map["tests_performed"]),
            float(col_map["positive_rate"]),
            int(col_map["hospitalized_patients"]),
            int(col_map["icu_patients"]),
            float(col_map["mobility_index"]),
            float(col_map["temperature"]),
            float(col_map["humidity"]),
            int(col_map["population_density"]),
            self.disease_name,
            data_confidence,
        )

        conn = sqlite3.connect(DB_PATH)
        try:
            placeholders = ", ".join(["?"] * len(_TABLE_COLUMNS))
            col_names = ", ".join(_TABLE_COLUMNS)
            conn.execute(
                f"INSERT INTO {self.table_name} ({col_names}) VALUES ({placeholders})",
                row,
            )
            conn.commit()
        finally:
            conn.close()

    def ingest_bulk(self, csv_path: str) -> int:
        """Read a CSV file and ingest each row via :meth:`ingest_daily`.

        The CSV must have at minimum the columns ``state``, ``date``,
        ``new_cases``, and ``new_deaths``.  Any additional columns whose
        names match accepted kwargs of :meth:`ingest_daily` are passed
        through automatically.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.

        Returns
        -------
        int
            Number of rows successfully ingested.

        Raises
        ------
        FileNotFoundError
            If *csv_path* does not exist.
        ValueError
            If any row fails validation (propagated from
            :meth:`ingest_daily`).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        ingested = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row_num, row in enumerate(reader, start=2):  # row 1 = header
                try:
                    state = row["state"]
                    date = row["date"]
                    new_cases = int(row["new_cases"])
                    new_deaths = int(row["new_deaths"])
                except KeyError as exc:
                    raise ValueError(
                        f"Row {row_num}: missing required column {exc}"
                    ) from exc
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Row {row_num}: invalid numeric value — {exc}"
                    ) from exc

                # Gather optional kwargs
                extra = {}
                optional_fields = {
                    "new_vaccinations", "vaccinations",
                    "tests_performed", "tests",
                    "positive_rate",
                    "hospitalized_patients", "hospitalized",
                    "icu_patients", "icu",
                    "mobility_index",
                    "temperature",
                    "humidity",
                    "population_density",
                    "data_confidence",
                }
                for key in optional_fields:
                    if key in row and row[key] not in (None, ""):
                        try:
                            extra[key] = float(row[key]) if "." in str(row[key]) else int(row[key])
                        except ValueError:
                            extra[key] = row[key]  # e.g. data_confidence is a string

                # Pop data_confidence out so it becomes a keyword arg
                confidence = extra.pop("data_confidence", "confirmed")

                self.ingest_daily(
                    state=state,
                    date=date,
                    new_cases=new_cases,
                    new_deaths=new_deaths,
                    data_confidence=str(confidence),
                    **extra,
                )
                ingested += 1

        return ingested

    def get_data(self, state: str) -> pd.DataFrame:
        """Return all rows for *state*, sorted by date ascending.

        Parameters
        ----------
        state : str
            State / region name to filter on.

        Returns
        -------
        pd.DataFrame
            All columns from the disease table for the requested state.
        """
        conn = sqlite3.connect(DB_PATH)
        try:
            query = (
                f"SELECT * FROM {self.table_name} "
                f"WHERE state = ? ORDER BY date ASC"
            )
            return pd.read_sql_query(query, conn, params=(state,))
        finally:
            conn.close()

    def get_row_count(self, state: str) -> int:
        """Return the number of rows stored for *state*.

        Parameters
        ----------
        state : str
            State / region name to filter on.

        Returns
        -------
        int
        """
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE state = ?",
                (state,),
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing / zero-valued feature columns with sensible defaults.

        The defaults are derived from the ``pathogen_type`` priors set
        during construction.

        Imputation rules
        ~~~~~~~~~~~~~~~~
        - ``tests_performed`` → ``new_cases × test_multiplier``
        - ``hospitalized_patients`` → ``new_cases × hospitalization_rate``
        - ``icu_patients`` → ``hospitalized_patients × icu_rate``
        - ``positive_rate`` → ``new_cases / (tests_performed + 1)``
        - ``new_vaccinations`` → ``0`` (no assumption possible)
        - ``mobility_index`` → pathogen-type default
        - ``temperature`` → pathogen-type default
        - ``humidity`` → pathogen-type default
        - ``population_density`` → pathogen-type default

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to fill.  Modified **in-place** and also returned
            for convenience.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with missing values filled.
        """
        df = df.copy()

        priors = self.priors

        # Ensure columns exist (add with NaN if absent)
        for col in _TABLE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # ── tests_performed ──
        mask = df["tests_performed"].isna() | (df["tests_performed"] == 0)
        df.loc[mask, "tests_performed"] = (
            df.loc[mask, "new_cases"] * priors["test_multiplier"]
        )

        # ── hospitalized_patients ──
        mask = df["hospitalized_patients"].isna() | (df["hospitalized_patients"] == 0)
        df.loc[mask, "hospitalized_patients"] = (
            df.loc[mask, "new_cases"] * priors["hospitalization_rate"]
        ).round().astype(int)

        # ── icu_patients ──
        mask = df["icu_patients"].isna() | (df["icu_patients"] == 0)
        df.loc[mask, "icu_patients"] = (
            df.loc[mask, "hospitalized_patients"] * priors["icu_rate"]
        ).round().astype(int)

        # ── positive_rate ──
        mask = df["positive_rate"].isna() | (df["positive_rate"] == 0)
        df.loc[mask, "positive_rate"] = (
            df.loc[mask, "new_cases"] / (df.loc[mask, "tests_performed"] + 1)
        ).round(4)

        # ── new_vaccinations (no assumption — default to 0) ──
        df["new_vaccinations"] = df["new_vaccinations"].fillna(0).astype(int)

        # ── environmental / demographic defaults ──
        for col, key in [
            ("mobility_index", "mobility_index"),
            ("temperature", "temperature"),
            ("humidity", "humidity"),
            ("population_density", "population_density"),
        ]:
            mask = df[col].isna() | (df[col] == 0)
            df.loc[mask, col] = priors[key]

        return df

    # ── private helpers ─────────────────────────────────────

    def _init_table(self) -> None:
        """Create the per-disease SQLite table if it does not exist."""
        db_dir = os.path.dirname(DB_PATH)
        os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
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
                    population_density    INTEGER,
                    disease_name          TEXT,
                    data_confidence       TEXT DEFAULT 'confirmed'
                )
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _validate_date(date_str: str) -> None:
        """Raise ``ValueError`` if *date_str* is not a valid ``YYYY-MM-DD``."""
        if not _DATE_RE.match(date_str):
            raise ValueError(
                f"Date must be in YYYY-MM-DD format, got '{date_str}'"
            )
        # Also verify it is a real calendar date
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid calendar date '{date_str}': {exc}") from exc

    @staticmethod
    def _validate_non_negative(name: str, value) -> None:
        """Raise ``ValueError`` if *value* is negative."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{name}' must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"'{name}' cannot be negative, got {value}")
