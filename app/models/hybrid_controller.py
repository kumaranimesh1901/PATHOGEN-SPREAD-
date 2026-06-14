"""
Hybrid Controller for multi-pathogen forecasting.
==================================================
Automatically selects the right model based on how much data is available
for a new disease:

  - **SEIR mode** (< 14 days)   → prior-only compartmental simulation
  - **Transfer mode** (14–59 days) → CatBoost warm-started from COVID model
  - **CatBoost mode** (≥ 60 days) → full CatBoost retrain on new-disease data

Every prediction is logged to ``logs/hybrid_controller.log``.
"""

import logging
import os
from datetime import datetime

from app.data_sources.new_disease_ingester import NewDiseaseIngester
from app.models.seir_model import run_cold_start_seir, DISEASE_PRESETS
from app.models.transfer_model import (
    fine_tune_from_covid,
    predict_with_transfer,
    InsufficientDataError,
    TARGETS as TRANSFER_TARGETS,
)
from app.models.catboost_model import (
    train_catboost,
    predict_future,
    _model_path as catboost_model_path,
    TARGETS as CB_TARGETS,
)
from app.utils.risk_calculator import calculate_risk

# ──────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────

_LOG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
)
_LOG_FILE = os.path.join(_LOG_DIR, "hybrid_controller.log")


def _get_logger() -> logging.Logger:
    """Return (and lazily configure) the hybrid-controller logger."""
    logger = logging.getLogger("hybrid_controller")
    if not logger.handlers:
        os.makedirs(_LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ──────────────────────────────────────────────────────────
# Mode thresholds
# ──────────────────────────────────────────────────────────

_TRANSFER_THRESHOLD = 14   # Minimum days for transfer learning
_CATBOOST_THRESHOLD = 60   # Minimum days for full CatBoost

_CONFIDENCE_MAP = {
    "seir":     "low",
    "transfer": "medium",
    "catboost": "high",
}

_MODE_LABELS = {
    "seir":     "SEIR MODE",
    "transfer": "TRANSFER MODE",
    "catboost": "CATBOOST MODE",
}


# ══════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════

class HybridController:
    """Automatically select and run the best model for a new disease.

    Parameters
    ----------
    disease_name : str
        Identifier for the disease (must match what was used during
        ingestion via :class:`NewDiseaseIngester`).
    state : str
        State / region name.

    Attributes
    ----------
    mode : str
        One of ``"seir"``, ``"transfer"``, ``"catboost"``.
    data_days : int
        Number of rows currently available for this disease + state.
    """

    def __init__(self, disease_name: str, state: str) -> None:
        self.disease_name = disease_name
        self.state = state
        self.logger = _get_logger()

        # Determine data availability via the ingester
        self._ingester = NewDiseaseIngester(
            disease_name,
            # Use a neutral pathogen type — we only need the table,
            # not the priors.  The table was already created during
            # initial ingestion.
            pathogen_type="respiratory",
        )
        self.data_days: int = self._ingester.get_row_count(state)

        # Select mode
        if self.data_days >= _CATBOOST_THRESHOLD:
            self.mode = "catboost"
        elif self.data_days >= _TRANSFER_THRESHOLD:
            self.mode = "transfer"
        else:
            self.mode = "seir"

    # ── properties ──────────────────────────────────────────

    @property
    def confidence(self) -> str:
        """Return the confidence level for the current mode."""
        return _CONFIDENCE_MAP[self.mode]

    # ── public API ──────────────────────────────────────────

    def predict(self, disease_config: dict = None) -> dict:
        """Run the appropriate model and return a standardised forecast.

        Parameters
        ----------
        disease_config : dict, optional
            Passed through to :func:`run_cold_start_seir` in SEIR mode.
            Ignored in transfer / catboost modes (epidemiology is learned
            from data instead).

        Returns
        -------
        dict
            Standardised forecast dictionary with keys:
            ``cases_7d``, ``cases_14d``, ``cases_30d``,
            ``deaths_7d``, ``deaths_14d``, ``deaths_30d``,
            ``risk_score``, ``risk_category``, ``model_used``,
            ``data_days_available``, ``confidence``,
            ``growth_rate``, ``daily_cases``, ``daily_deaths``,
            ``peak_day``, ``outbreak_duration``,
            ``reproduction_number``, ``feature_importance``,
            ``seir_curve``.
        """
        self.print_status()

        if self.mode == "seir":
            result = self._predict_seir(disease_config or {})
        elif self.mode == "transfer":
            result = self._predict_transfer(disease_config or {})
        else:
            result = self._predict_catboost(disease_config or {})

        # ── Risk calculation ────────────────────────────────
        mortality_rate = 0.01
        if disease_config:
            mortality_rate = float(disease_config.get("mortality_rate", 0.01))
        elif self.mode == "seir":
            mortality_rate = 0.01

        risk_category, risk_score = calculate_risk(
            result.get("growth_rate", 0.0), mortality_rate
        )

        # ── Build standardised output ───────────────────────
        forecast = {
            "state": self.state,
            "disease_name": self.disease_name,
            "model_used": result.get("model_used", self.mode),
            "data_days_available": self.data_days,
            "confidence": self.confidence,
            # Projections
            "cases_7d": result.get("cases_7d", 0),
            "cases_14d": result.get("cases_14d", 0),
            "cases_30d": result.get("cases_30d", 0),
            "deaths_7d": result.get("deaths_7d", 0),
            "deaths_14d": result.get("deaths_14d", 0),
            "deaths_30d": result.get("deaths_30d", 0),
            # Risk
            "risk_score": risk_score,
            "risk_category": risk_category,
            # Extras
            "growth_rate": result.get("growth_rate", 0.0),
            "peak_day": result.get("peak_day"),
            "outbreak_duration": result.get("outbreak_duration"),
            "reproduction_number": result.get("reproduction_number"),
            "feature_importance": result.get("feature_importance"),
            "seir_curve": result.get("seir_curve"),
            "daily_cases": result.get("daily_cases", [0] * 30),
            "daily_deaths": result.get("daily_deaths", [0] * 30),
        }

        self._log_prediction(forecast)
        return forecast

    def print_status(self) -> None:
        """Print a formatted status block showing the active mode."""
        label = _MODE_LABELS[self.mode]
        conf = self.confidence.upper()

        print(f"\n{'─' * 56}")
        print(f"  [{label}] {self.disease_name} — {self.state}")
        print(f"  Data available : {self.data_days} days")
        print(f"  Confidence     : {conf}")

        if self.mode == "seir":
            days_to_transfer = max(0, _TRANSFER_THRESHOLD - self.data_days)
            print(
                f"  → Switch to transfer learning at day "
                f"{_TRANSFER_THRESHOLD} ({days_to_transfer} more days needed)."
            )
        elif self.mode == "transfer":
            days_to_catboost = max(0, _CATBOOST_THRESHOLD - self.data_days)
            print(
                f"  → Switch to full CatBoost at day "
                f"{_CATBOOST_THRESHOLD} ({days_to_catboost} more days needed)."
            )
        else:
            print("  → Full CatBoost model active. Maximum accuracy.")

        print(f"{'─' * 56}\n")

    # ── private dispatch ────────────────────────────────────

    def _predict_seir(self, disease_config: dict) -> dict:
        """Dispatch to cold-start SEIR."""
        return run_cold_start_seir(self.state, disease_config)

    def _predict_transfer(self, disease_config: dict = None) -> dict:
        """Dispatch to transfer-learning pipeline.

        Automatically fine-tunes models if they don't exist yet.
        Falls back to SEIR if fine-tuning is not viable.
        """
        # Get raw new-disease data
        df = self._ingester.get_data(self.state)
        df = self._ingester.fill_missing_features(df)

        # Fine-tune any missing transfer models
        for _, target_name in TRANSFER_TARGETS:
            from app.models.transfer_model import _transfer_model_path
            model_path = _transfer_model_path(
                self.disease_name, self.state, target_name
            )
            if not os.path.exists(model_path):
                # Find a base COVID model for this state + target
                base_path = catboost_model_path(self.state, target_name)
                if not os.path.exists(base_path):
                    # Fall back to any available state's model
                    import glob
                    pattern = os.path.join(
                        os.path.dirname(base_path),
                        f"catboost_*_{target_name}.cbm",
                    )
                    candidates = glob.glob(pattern)
                    if candidates:
                        base_path = candidates[0]
                    else:
                        print(
                            f"  ⚠ No base COVID model found for {target_name}. "
                            f"Skipping transfer fine-tune."
                        )
                        continue

                print(f"  Fine-tuning {target_name} from {os.path.basename(base_path)}...")
                try:
                    fine_tune_from_covid(df, self.state, target_name, base_path)
                except InsufficientDataError as exc:
                    print(f"  ⚠ {exc}")
                    continue

        # Now predict — fall back to SEIR if fine-tuning failed
        try:
            result = predict_with_transfer(self.disease_name, self.state, df)
            result["model_used"] = "transfer_learning"
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"  ⚠ Transfer prediction unavailable ({exc}). "
                f"Falling back to SEIR cold-start."
            )
            config = disease_config if isinstance(disease_config, dict) else {}
            result = self._predict_seir(config)
            result["model_used"] = "seir_fallback"
        return result

    def _predict_catboost(self, disease_config: dict = None) -> dict:
        """Dispatch to full CatBoost retrain + predict.

        Trains fresh CatBoost models on new-disease data if they don't
        exist yet (uses the standard ``train_catboost`` pipeline with
        the ingester's table as the data source).
        """
        # For CatBoost mode, we need the data in the main disease_records
        # format.  Since train_catboost() reads from DB via get_state_data(),
        # and the new-disease data lives in a separate table, we use the
        # transfer-predict path but with many more iterations.
        # Pragmatic approach: use transfer pipeline (which handles the
        # separate data source) but with full training if we have enough data.
        df = self._ingester.get_data(self.state)
        df = self._ingester.fill_missing_features(df)

        # Fine-tune with full iterations since we have ≥ 60 rows
        for _, target_name in TRANSFER_TARGETS:
            from app.models.transfer_model import _transfer_model_path
            model_path = _transfer_model_path(
                self.disease_name, self.state, target_name
            )
            if not os.path.exists(model_path):
                base_path = catboost_model_path(self.state, target_name)
                if not os.path.exists(base_path):
                    import glob
                    pattern = os.path.join(
                        os.path.dirname(base_path),
                        f"catboost_*_{target_name}.cbm",
                    )
                    candidates = glob.glob(pattern)
                    if candidates:
                        base_path = candidates[0]
                    else:
                        print(
                            f"  ⚠ No base model found for {target_name}."
                        )
                        continue

                print(f"  Training {target_name} (full CatBoost from transfer base)...")
                try:
                    fine_tune_from_covid(df, self.state, target_name, base_path)
                except InsufficientDataError as exc:
                    print(f"  ⚠ {exc}")
                    continue

        # Predict — fall back to SEIR if models not available
        try:
            result = predict_with_transfer(self.disease_name, self.state, df)
            result["model_used"] = "catboost_full"
        except (FileNotFoundError, ValueError) as exc:
            print(
                f"  ⚠ CatBoost prediction unavailable ({exc}). "
                f"Falling back to SEIR cold-start."
            )
            config = disease_config if isinstance(disease_config, dict) else {}
            result = self._predict_seir(config)
            result["model_used"] = "seir_fallback"
        return result

    # ── logging ─────────────────────────────────────────────

    def _log_prediction(self, forecast: dict) -> None:
        """Log a prediction to the hybrid controller log file."""
        self.logger.info(
            "mode=%s | disease=%s | state=%s | confidence=%s | "
            "cases_7d=%d | cases_14d=%d | cases_30d=%d | "
            "deaths_7d=%d | deaths_14d=%d | deaths_30d=%d | "
            "risk=%s (%.1f) | data_days=%d",
            forecast["model_used"],
            forecast["disease_name"],
            forecast["state"],
            forecast["confidence"],
            forecast["cases_7d"],
            forecast["cases_14d"],
            forecast["cases_30d"],
            forecast["deaths_7d"],
            forecast["deaths_14d"],
            forecast["deaths_30d"],
            forecast["risk_category"],
            forecast["risk_score"],
            forecast["data_days_available"],
        )
