"""
PATHOGEN-SPREAD — Terminal-based Outbreak Forecasting
=====================================================
Single-file entry point. Run with:  python run.py

Loads data, trains models, runs forecast, prints results,
and saves all charts to the outputs/ folder.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from datetime import datetime

from app.database.db import init_db, load_csv_to_db, get_state_names, get_row_count
from app.models.catboost_model import train_catboost, predict_future
from app.models.seir_model import run_seir
from app.models.hybrid_model import run_forecast
from app.models.hybrid_controller import HybridController
from app.utils.risk_calculator import calculate_risk


# ═══════════════════════════════════════════════════════════
# USER CONFIGURATION — edit these before running
# ═══════════════════════════════════════════════════════════

STATE_NAME        = "Tamil Nadu"    # which state to forecast
CURRENT_CASES     = 1200            # today's case count
CURRENT_DEATHS    = 18              # today's death count
POPULATION        = 72000000        # state population
TRANSMISSION_RATE = 1.5             # beta
RECOVERY_RATE     = 0.1             # gamma
MORTALITY_RATE    = 0.02            # mu
FORCE_RETRAIN     = True            # set True to retrain even if models exist

CSV_PATH   = "app/data/india_covid_catboost_dataset.csv"
OUTPUT_DIR = "outputs"

# ── New Disease Configuration (set to None to skip) ─────
# To forecast a new disease via HybridController, set these:
NEW_DISEASE_NAME  = None              # e.g. "monkeypox", None to skip
NEW_DISEASE_TYPE  = "unknown"         # preset key: covid19, influenza, mpox, unknown
NEW_DISEASE_STATE = STATE_NAME        # state to forecast for new disease

# ═══════════════════════════════════════════════════════════
# Color palette (used consistently across all charts)
# ═══════════════════════════════════════════════════════════

COLOR_CASES      = "#FF9800"
COLOR_DEATHS     = "#F44336"
COLOR_S          = "#2196F3"
COLOR_E          = "#FF9800"
COLOR_I          = "#F44336"
COLOR_R          = "#4CAF50"
COLOR_BAR        = "#7C4DFF"
COLOR_RISK_LOW   = "#4CAF50"
COLOR_RISK_MED   = "#FF9800"
COLOR_RISK_HIGH  = "#F44336"


# ═══════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _chart_forecast(result, state, path):
    """Chart 1 — 30-Day Forecast: Cases & Deaths."""
    days = list(range(1, 31))
    cases = result["daily_cases"]
    deaths = result["daily_deaths"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(days, cases, color=COLOR_CASES, linewidth=2.2, label="Daily Cases", marker="o", markersize=3)
    ax.plot(days, deaths, color=COLOR_DEATHS, linewidth=2.2, label="Daily Deaths", marker="s", markersize=3)

    # Vertical dashed lines at day 7 and 14
    for d, lbl in [(7, "Day 7"), (14, "Day 14")]:
        ax.axvline(x=d, color="#AAAAAA", linestyle="--", alpha=0.6)
        ax.text(d + 0.3, ax.get_ylim()[1] * 0.92, lbl, color="#AAAAAA", fontsize=9)

    # Annotate key values
    for idx, label in [(6, "7d"), (13, "14d"), (29, "30d")]:
        ax.annotate(f"{cases[idx]:,}", xy=(idx + 1, cases[idx]),
                    textcoords="offset points", xytext=(8, 10),
                    fontsize=8, color=COLOR_CASES,
                    arrowprops=dict(arrowstyle="->", color=COLOR_CASES, lw=0.8))
        ax.annotate(f"{deaths[idx]:,}", xy=(idx + 1, deaths[idx]),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=8, color=COLOR_DEATHS,
                    arrowprops=dict(arrowstyle="->", color=COLOR_DEATHS, lw=0.8))

    ax.set_title(f"30-Day Outbreak Forecast — {state}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_seir(result, state, path):
    """Chart 2 — SEIR Infection Curve."""
    curve = result.get("seir_curve")
    if curve is None:
        return False

    days = curve["days"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(days, curve["susceptible"], color=COLOR_S, linewidth=1.8, label="Susceptible")
    ax.plot(days, curve["exposed"],     color=COLOR_E, linewidth=1.8, label="Exposed")
    ax.plot(days, curve["infected"],    color=COLOR_I, linewidth=1.8, label="Infected")
    ax.plot(days, curve["recovered"],   color=COLOR_R, linewidth=1.8, label="Recovered")

    # Mark peak day
    peak_day = result.get("peak_day")
    if peak_day is not None:
        ax.axvline(x=peak_day, color=COLOR_I, linestyle="--", alpha=0.7)
        ax.text(peak_day + 1, max(curve["infected"]) * 0.95,
                f"Peak Day {peak_day}", color=COLOR_I, fontsize=9)

    ax.set_title(f"SEIR Epidemic Curve — {state}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("Population", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _chart_features(result, path):
    """Chart 3 — Feature Importance horizontal bar chart."""
    fi = result.get("feature_importance")
    if not fi:
        return False

    sorted_fi = sorted(fi.items(), key=lambda x: x[1])
    names = [n for n, _ in sorted_fi]
    scores = [s for _, s in sorted_fi]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))
    bars = ax.barh(names, scores, color=COLOR_BAR, edgecolor="#5C35CC", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", fontsize=9, color="#DDDDDD")

    ax.set_title("CatBoost Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (%)", fontsize=11)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _draw_risk_gauge(ax, risk_score, risk_category):
    """Draw a semicircular risk gauge on the given axes."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw three colored arcs (Low, Medium, High)
    zone_info = [
        (0, 60,  COLOR_RISK_LOW,  "Low"),
        (60, 120, COLOR_RISK_MED,  "Medium"),
        (120, 180, COLOR_RISK_HIGH, "High"),
    ]
    for start_deg, end_deg, color, _ in zone_info:
        theta = np.linspace(np.radians(180 - start_deg), np.radians(180 - end_deg), 100)
        x_outer = np.cos(theta)
        y_outer = np.sin(theta)
        x_inner = 0.7 * np.cos(theta)
        y_inner = 0.7 * np.sin(theta)
        verts = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
        poly = plt.Polygon(verts, closed=True, facecolor=color, alpha=0.7, edgecolor="none")
        ax.add_patch(poly)

    # Draw needle
    needle_angle = np.radians(180 - (risk_score / 100.0) * 180)
    needle_x = 0.85 * np.cos(needle_angle)
    needle_y = 0.85 * np.sin(needle_angle)
    ax.plot([0, needle_x], [0, needle_y], color="white", linewidth=2.5, zorder=5)
    ax.plot(0, 0, "o", color="white", markersize=6, zorder=6)

    # Score text
    ax.text(0, 0.35, f"{risk_score:.1f}", ha="center", va="center",
            fontsize=28, fontweight="bold", color="white")
    ax.text(0, 0.15, risk_category.upper(), ha="center", va="center",
            fontsize=13, fontweight="bold",
            color=COLOR_RISK_LOW if risk_category == "Low"
            else COLOR_RISK_MED if risk_category == "Medium"
            else COLOR_RISK_HIGH)


def _chart_risk(result, path):
    """Chart 4 — Risk Gauge."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_risk_gauge(ax, result["risk_score"], result["risk_category"])
    ax.set_title("Outbreak Risk Assessment", fontsize=14, fontweight="bold",
                 pad=20, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_metrics(metrics, path):
    """Chart 5 — Training Metrics Bar Chart (MAE & RMSE + R² line)."""
    if not metrics:
        return False

    targets = list(metrics.keys())
    mae_vals = [metrics[t]["mae"] for t in targets]
    rmse_vals = [metrics[t]["rmse"] for t in targets]
    r2_vals = [metrics[t].get("r2_score", metrics[t].get("r2", 0)) for t in targets]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))
    bars_mae = ax1.bar(x - width / 2, mae_vals, width, label="MAE", color="#42A5F5", edgecolor="#1E88E5")
    bars_rmse = ax1.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#EF5350", edgecolor="#E53935")

    ax1.set_xlabel("Target", fontsize=11)
    ax1.set_ylabel("Error", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=25, ha="right", fontsize=9)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")

    # Secondary axis for R²
    ax2 = ax1.twinx()
    ax2.plot(x, r2_vals, color="#FFD54F", marker="D", linewidth=2, markersize=7, label="R² Score")
    ax2.set_ylabel("R² Score", fontsize=11, color="#FFD54F")
    ax2.tick_params(axis="y", labelcolor="#FFD54F")
    ax2.legend(loc="upper right", fontsize=10)

    ax1.set_title("Model Training Metrics — MAE & RMSE per Target", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _chart_summary(result, state, metrics, seir_available, features_available, path):
    """Chart 6 — Summary Dashboard (always generated)."""
    today = datetime.now().strftime("%Y-%m-%d")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"PATHOGEN-SPREAD Dashboard — {state} — {today}",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.94, top=0.92, bottom=0.06)

    # ── Top-left: 30-day forecast curve ──
    ax1 = fig.add_subplot(gs[0, 0])
    days = list(range(1, 31))
    ax1.plot(days, result["daily_cases"], color=COLOR_CASES, linewidth=2, label="Cases", marker="o", markersize=2)
    ax1.plot(days, result["daily_deaths"], color=COLOR_DEATHS, linewidth=2, label="Deaths", marker="s", markersize=2)
    ax1.axvline(x=7, color="#AAAAAA", linestyle="--", alpha=0.5)
    ax1.axvline(x=14, color="#AAAAAA", linestyle="--", alpha=0.5)
    ax1.set_title("30-Day Forecast", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Day", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Top-right: Risk gauge ──
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_risk_gauge(ax2, result["risk_score"], result["risk_category"])
    ax2.set_title("Risk Assessment", fontsize=12, fontweight="bold", pad=15)

    # ── Middle-left: SEIR curve or placeholder ──
    ax3 = fig.add_subplot(gs[1, 0])
    curve = result.get("seir_curve")
    if curve is not None and seir_available:
        ax3.plot(curve["days"], curve["susceptible"], color=COLOR_S, linewidth=1.5, label="S")
        ax3.plot(curve["days"], curve["exposed"],     color=COLOR_E, linewidth=1.5, label="E")
        ax3.plot(curve["days"], curve["infected"],    color=COLOR_I, linewidth=1.5, label="I")
        ax3.plot(curve["days"], curve["recovered"],   color=COLOR_R, linewidth=1.5, label="R")
        peak_day = result.get("peak_day")
        if peak_day is not None:
            ax3.axvline(x=peak_day, color=COLOR_I, linestyle="--", alpha=0.6)
        ax3.set_title("SEIR Curve", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Day", fontsize=9)
        ax3.set_ylabel("Population", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)
    else:
        ax3.text(0.5, 0.5, "SEIR Not Used", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=16, color="#888888")
        ax3.set_title("SEIR Curve", fontsize=12, fontweight="bold")
        ax3.axis("off")

    # ── Middle-right: Feature importance or placeholder ──
    ax4 = fig.add_subplot(gs[1, 1])
    fi = result.get("feature_importance")
    if fi and features_available:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1])
        names = [n for n, _ in sorted_fi]
        scores = [s for _, s in sorted_fi]
        ax4.barh(names, scores, color=COLOR_BAR, edgecolor="#5C35CC", linewidth=0.5)
        for i, (name, score) in enumerate(zip(names, scores)):
            ax4.text(score + 0.3, i, f"{score:.1f}%", va="center", fontsize=8, color="#DDDDDD")
        ax4.set_title("Feature Importance", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Importance (%)", fontsize=9)
        ax4.grid(True, alpha=0.2, axis="x")
    else:
        ax4.text(0.5, 0.5, "Not Available", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=16, color="#888888")
        ax4.set_title("Feature Importance", fontsize=12, fontweight="bold")
        ax4.axis("off")

    # ── Bottom row: metric boxes spanning full width ──
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    box_data = [
        ("Cases 7d",  f"{result['cases_7d']:,}",  COLOR_CASES),
        ("Cases 14d", f"{result['cases_14d']:,}",  COLOR_CASES),
        ("Cases 30d", f"{result['cases_30d']:,}",  COLOR_CASES),
        ("Deaths 7d",  f"{result['deaths_7d']:,}",  COLOR_DEATHS),
        ("Deaths 14d", f"{result['deaths_14d']:,}",  COLOR_DEATHS),
        ("Deaths 30d", f"{result['deaths_30d']:,}",  COLOR_DEATHS),
    ]

    for i, (label, value, color) in enumerate(box_data):
        x_pos = (i + 0.5) / len(box_data)
        ax5.text(x_pos, 0.65, value, transform=ax5.transAxes,
                 ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        ax5.text(x_pos, 0.30, label, transform=ax5.transAxes,
                 ha="center", va="center", fontsize=11, color="#CCCCCC")

    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Banner ──
    print("=" * 50)
    print("  PATHOGEN-SPREAD — Outbreak Forecasting")
    print(f"  State: {STATE_NAME}")
    print(f"  Date:  {today}")
    print("=" * 50)

    # ── Step 1: Database setup ──
    print("\n[1/5] Loading dataset into database...")
    init_db()
    load_csv_to_db(CSV_PATH)
    row_count = get_row_count(STATE_NAME)
    print(f"      Rows available for {STATE_NAME}: {row_count}")

    if row_count == 0:
        print(f"\n  ERROR: No data found for state '{STATE_NAME}'.")
        print(f"  Available states: {get_state_names()}")
        sys.exit(1)

    # ── Step 2: Model training ──
    print("\n[2/5] Checking / Training CatBoost models...")

    model_basenames = [
        f"catboost_{STATE_NAME}_cases_7d.cbm",
        f"catboost_{STATE_NAME}_cases_14d.cbm",
        f"catboost_{STATE_NAME}_cases_30d.cbm",
        f"catboost_{STATE_NAME}_deaths_7d.cbm",
        f"catboost_{STATE_NAME}_deaths_14d.cbm",
        f"catboost_{STATE_NAME}_deaths_30d.cbm",
    ]
    model_files = [os.path.join("models_saved", f) for f in model_basenames]
    models_exist = all(os.path.exists(f) for f in model_files)

    metrics = None
    if FORCE_RETRAIN or not models_exist:
        print(f"      Training 6 CatBoost models for {STATE_NAME}...")
        metrics = train_catboost(STATE_NAME)
        print("\n      ── Training Metrics ──")
        for target, m in metrics.items():
            print(f"      {target}:")
            print(f"        MAE  : {m['mae']:.2f}")
            print(f"        RMSE : {m['rmse']:.2f}")
            r2_val = m.get("r2_score", m.get("r2", 0))
            print(f"        R²   : {r2_val:.4f}")
    else:
        print("      Models already trained. Skipping. Set FORCE_RETRAIN=True to retrain.")

    # ── Step 3: Run forecast ──
    print(f"\n[3/5] Running forecast for {STATE_NAME}...")

    user_inputs = {
        "state": STATE_NAME,
        "current_cases": CURRENT_CASES,
        "current_deaths": CURRENT_DEATHS,
        "population": POPULATION,
        "transmission_rate": TRANSMISSION_RATE,
        "recovery_rate": RECOVERY_RATE,
        "mortality_rate": MORTALITY_RATE,
    }

    result = run_forecast(STATE_NAME, user_inputs)

    # ── Print forecast results ──
    print("\n" + "=" * 50)
    print("  FORECAST RESULTS")
    print("=" * 50)
    print(f"  Engine Used    : {result['engine_used'].upper()}")
    print(f"  Risk Category  : {result['risk_category']}")
    print(f"  Risk Score     : {result['risk_score']:.1f} / 100")
    print(f"  Growth Rate    : {result['growth_rate']:.2f}%")
    if result["reproduction_number"]:
        print(f"  R\u2080 (Reprod.)   : {result['reproduction_number']:.2f}")
    print("-" * 50)
    print(f"  Cases  \u2014 7d    : {result['cases_7d']:,}")
    print(f"  Cases  \u2014 14d   : {result['cases_14d']:,}")
    print(f"  Cases  \u2014 30d   : {result['cases_30d']:,}")
    print("-" * 50)
    print(f"  Deaths \u2014 7d    : {result['deaths_7d']:,}")
    print(f"  Deaths \u2014 14d   : {result['deaths_14d']:,}")
    print(f"  Deaths \u2014 30d   : {result['deaths_30d']:,}")
    print("-" * 50)
    if result["peak_day"]:
        print(f"  Peak Day       : Day {result['peak_day']}")
    if result["outbreak_duration"]:
        print(f"  Outbreak Ends  : ~Day {result['outbreak_duration']}")
    print("=" * 50)

    # ── Step 4: Generate charts ──
    print("\n[4/5] Generating charts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use("dark_background")

    # Chart 1 — Forecast
    path1 = os.path.join(OUTPUT_DIR, "chart1_forecast.png")
    _chart_forecast(result, STATE_NAME, path1)

    # Chart 2 — SEIR
    path2 = os.path.join(OUTPUT_DIR, "chart2_seir.png")
    seir_generated = _chart_seir(result, STATE_NAME, path2)

    # Chart 3 — Feature Importance
    path3 = os.path.join(OUTPUT_DIR, "chart3_features.png")
    features_generated = _chart_features(result, path3)

    # Chart 4 — Risk Gauge
    path4 = os.path.join(OUTPUT_DIR, "chart4_risk.png")
    _chart_risk(result, path4)

    # Chart 5 — Training Metrics
    path5 = os.path.join(OUTPUT_DIR, "chart5_metrics.png")
    metrics_generated = False
    if metrics:
        metrics_generated = _chart_metrics(metrics, path5)

    # Chart 6 — Summary Dashboard (always generated)
    path6 = os.path.join(OUTPUT_DIR, "chart6_summary.png")
    _chart_summary(result, STATE_NAME, metrics,
                   seir_available=(seir_generated if result.get("seir_curve") else False),
                   features_available=(features_generated if result.get("feature_importance") else False),
                   path=path6)

    # ── Step 5: Print chart locations ──
    abs_path6 = os.path.abspath(path6)
    print("\n[5/5] Charts saved:")
    print(f"      outputs/chart1_forecast.png")
    print(f"      outputs/chart2_seir.png     {'✓' if seir_generated else '(skipped — SEIR not used)'}")
    print(f"      outputs/chart3_features.png {'✓' if features_generated else '(skipped — CatBoost not used)'}")
    print(f"      outputs/chart4_risk.png")
    print(f"      outputs/chart5_metrics.png  {'✓' if metrics_generated else '(skipped — model not trained this run)'}")
    print(f"      outputs/chart6_summary.png  ← main dashboard")
    print(f"\n  Done. Open the full dashboard at:")
    print(f"  {abs_path6}")

    # ═══════════════════════════════════════════════════════
    # NEW DISEASE FORECAST (optional — gated behind config)
    # ═══════════════════════════════════════════════════════
    if NEW_DISEASE_NAME is not None:
        print("\n" + "=" * 50)
        print(f"  NEW DISEASE FORECAST — {NEW_DISEASE_NAME.upper()}")
        print("=" * 50)

        controller = HybridController(
            disease_name=NEW_DISEASE_NAME,
            state=NEW_DISEASE_STATE,
        )

        disease_config = {
            "disease_type": NEW_DISEASE_TYPE,
            "population": POPULATION,
            "seed_cases": CURRENT_CASES,
        }

        nd_result = controller.predict(disease_config=disease_config)

        print(f"\n  Engine Used    : {nd_result['model_used'].upper()}")
        print(f"  Confidence     : {nd_result['confidence'].upper()}")
        print(f"  Risk Category  : {nd_result['risk_category']}")
        print(f"  Risk Score     : {nd_result['risk_score']:.1f} / 100")
        print(f"  Data Days      : {nd_result['data_days_available']}")
        print("-" * 50)
        print(f"  Cases  — 7d    : {nd_result['cases_7d']:,}")
        print(f"  Cases  — 14d   : {nd_result['cases_14d']:,}")
        print(f"  Cases  — 30d   : {nd_result['cases_30d']:,}")
        print("-" * 50)
        print(f"  Deaths — 7d    : {nd_result['deaths_7d']:,}")
        print(f"  Deaths — 14d   : {nd_result['deaths_14d']:,}")
        print(f"  Deaths — 30d   : {nd_result['deaths_30d']:,}")
        print("=" * 50)


def run_all_states():
    """
    Batch-run forecasts for every state in the database.

    Steps:
      1. Initialise DB and load CSV
      2. Loop over all states
      3. Skip states with < 60 rows
      4. Train models if missing or FORCE_RETRAIN is True
      5. Run forecast, save per-state dashboard PNG
      6. Collect results, export CSV report sorted by risk_score
    """
    today = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("  PATHOGEN-SPREAD — Batch Forecasting (All States)")
    print(f"  Date: {today}")
    print("=" * 60)

    # ── Step 1: Database setup ──
    print("\n[INIT] Loading dataset into database...")
    init_db()
    load_csv_to_db(CSV_PATH)
    states = get_state_names()
    print(f"       Found {len(states)} states: {', '.join(states[:5])}{'...' if len(states) > 5 else ''}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use("dark_background")

    report_rows = []

    for idx, state in enumerate(states, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{len(states)}] {state}")
        print(f"{'─' * 60}")

        # ── Step 2: Check row count ──
        row_count = get_row_count(state)
        if row_count < 60:
            print(f"  ⚠  Skipping {state}: only {row_count} rows (need >= 60)")
            continue

        # ── Step 3: Check / Train models ──
        model_basenames = [
            f"catboost_{state}_cases_7d.cbm",
            f"catboost_{state}_cases_14d.cbm",
            f"catboost_{state}_cases_30d.cbm",
            f"catboost_{state}_deaths_7d.cbm",
            f"catboost_{state}_deaths_14d.cbm",
            f"catboost_{state}_deaths_30d.cbm",
        ]
        model_files = [os.path.join("models_saved", f) for f in model_basenames]
        models_exist = all(os.path.exists(f) for f in model_files)

        if FORCE_RETRAIN or not models_exist:
            print(f"  Training 6 CatBoost models for {state}...")
            try:
                metrics = train_catboost(state)
            except Exception as e:
                print(f"  ✗ Training failed for {state}: {e}")
                continue
        else:
            print(f"  Models already exist. Skipping training.")

        # ── Step 4: Run forecast ──
        user_inputs = {
            "state": state,
            "current_cases": CURRENT_CASES,
            "current_deaths": CURRENT_DEATHS,
            "population": POPULATION,
            "transmission_rate": TRANSMISSION_RATE,
            "recovery_rate": RECOVERY_RATE,
            "mortality_rate": MORTALITY_RATE,
        }

        try:
            result = run_forecast(state, user_inputs)
        except Exception as e:
            print(f"  ✗ Forecast failed for {state}: {e}")
            continue

        print(f"  Risk: {result['risk_category']} ({result['risk_score']:.1f}/100)  |  "
              f"Cases 7d={result['cases_7d']:,}  14d={result['cases_14d']:,}  30d={result['cases_30d']:,}")

        # ── Step 5: Save per-state dashboard PNG ──
        state_slug = state.replace(" ", "_")
        dashboard_path = os.path.join(OUTPUT_DIR, f"dashboard_{state_slug}.png")
        _chart_summary(
            result, state, metrics=None,
            seir_available=bool(result.get("seir_curve")),
            features_available=bool(result.get("feature_importance")),
            path=dashboard_path,
        )
        print(f"  Dashboard saved → {dashboard_path}")

        # ── Step 6: Collect row for report ──
        report_rows.append({
            "state": state,
            "risk_score": result["risk_score"],
            "risk_category": result["risk_category"],
            "cases_7d": result["cases_7d"],
            "cases_14d": result["cases_14d"],
            "cases_30d": result["cases_30d"],
            "deaths_7d": result["deaths_7d"],
            "deaths_14d": result["deaths_14d"],
            "deaths_30d": result["deaths_30d"],
            "growth_rate": result["growth_rate"],
        })

    # ── Step 7: Export CSV report ──
    if report_rows:
        report_df = pd.DataFrame(report_rows)
        report_df = report_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

        report_path = os.path.join(OUTPUT_DIR, f"statewise_report_{today}.csv")
        report_df.to_csv(report_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"  STATEWISE REPORT — {today}")
        print(f"{'=' * 60}")
        print(report_df.to_string(index=False))
        print(f"\n  Report saved → {report_path}")
    else:
        print("\n  ⚠ No states were processed. Report not generated.")

    print(f"\n  Done. {len(report_rows)} states processed.")


if __name__ == "__main__":
    run_all_states()
