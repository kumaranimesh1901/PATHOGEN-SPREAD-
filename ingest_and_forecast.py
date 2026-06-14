#!/usr/bin/env python
"""
PATHOGEN-SPREAD — New Disease Ingestion & Forecasting Utility
============================================================
Allows loading any custom CSV dataset for an arbitrary disease,
performing ingestion into the SQLite database, and running the
hybrid forecasting engine with professional visualizations.

Run with:
    python ingest_and_forecast.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path to support imports if run from app/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_sources.new_disease_ingester import NewDiseaseIngester, _VALID_PATHOGEN_TYPES
from app.models.hybrid_controller import HybridController
from run import (
    _chart_forecast,
    _chart_seir,
    _chart_risk,
    _chart_features,
    _chart_summary,
    OUTPUT_DIR,
)

def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title.upper()}")
    print("=" * 60)

def prompt_input(prompt_text: str, default: str = None) -> str:
    suffix = f" [{default}]: " if default is not None else ": "
    val = input(prompt_text + suffix).strip()
    return val if val else default

def main():
    print_banner("Pathogen-Spread — Custom Disease Ingestion & Forecast")
    
    print("\n[STEP 1] Enter Dataset and Disease Parameters")
    
    # ── 1. Get CSV Path ──
    csv_path = ""
    while not csv_path:
        csv_path = prompt_input("Enter the path to your CSV dataset (e.g. app/data/monkeypox.csv)")
        if not os.path.exists(csv_path):
            print(f"  ❌ File not found at '{csv_path}'. Please check the path and try again.")
            csv_path = ""
            
    # ── 2. Get Disease Name ──
    disease_name = ""
    while not disease_name:
        disease_name = prompt_input("Enter the name of the disease (e.g. monkeypox)").strip().lower()
        if not disease_name:
            print("  ❌ Disease name cannot be empty.")
            
    # ── 3. Get Pathogen Type ──
    print("\nAvailable transmission modes / pathogen types:")
    types_list = sorted(list(_VALID_PATHOGEN_TYPES))
    for idx, t in enumerate(types_list, 1):
        print(f"  {idx}. {t}")
    
    pathogen_type = ""
    while pathogen_type not in _VALID_PATHOGEN_TYPES:
        choice = prompt_input(f"Select pathogen type (1-{len(types_list)} or type it directly)")
        if choice.isdigit() and 1 <= int(choice) <= len(types_list):
            pathogen_type = types_list[int(choice) - 1]
        elif choice in _VALID_PATHOGEN_TYPES:
            pathogen_type = choice
        else:
            print(f"  ❌ Invalid option. Please select one of: {', '.join(types_list)}")

    # ── 4. Get State / Region Name ──
    state_name = ""
    while not state_name:
        state_name = prompt_input("Enter the state/region name (e.g. Tamil Nadu or Karnataka)")
        if not state_name:
            print("  ❌ State/region name cannot be empty.")

    # ── 5. Get Population ──
    population_str = prompt_input("Enter the state population", "10000000")
    try:
        population = int(population_str)
    except ValueError:
        population = 10_000_000
        print(f"  ⚠️ Invalid number. Using default population of {population:,}")

    # ── Ingestion ──
    print_banner(f"Ingesting {disease_name.upper()} dataset")
    print(f"CSV Path      : {csv_path}")
    print(f"Pathogen Type : {pathogen_type}")
    print(f"Target State  : {state_name}")
    print("Ingesting data rows...")
    
    try:
        ingester = NewDiseaseIngester(disease_name=disease_name, pathogen_type=pathogen_type)
        rows_imported = ingester.ingest_bulk(csv_path)
        print(f"✅ Ingestion successful! Loaded {rows_imported} rows into the SQLite table.")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        sys.exit(1)

    # ── Forecasting ──
    print_banner(f"Running Forecast for {disease_name.upper()} in {state_name}")
    
    controller = HybridController(disease_name=disease_name, state=state_name)
    
    # Get last known case count for seed cases
    df_data = ingester.get_data(state_name)
    seed_cases = 1
    if not df_data.empty:
        seed_cases = int(df_data.iloc[-1].get("new_cases", 1))

    disease_config = {
        "disease_type": pathogen_type,
        "population": population,
        "seed_cases": seed_cases,
    }
    
    result = controller.predict(disease_config=disease_config)
    
    # ── Output results to terminal ──
    print("\n" + "=" * 50)
    print(f"  {disease_name.upper()} FORECAST RESULTS — {state_name}")
    print("=" * 50)
    print(f"  Engine Used        : {result['model_used'].upper()}")
    print(f"  Risk Category      : {result['risk_category']}")
    print(f"  Risk Score         : {result['risk_score']:.1f} / 100")
    print(f"  Data Days Available: {result['data_days_available']}")
    print(f"  Confidence         : {result['confidence'].upper()}")
    print("-" * 50)
    print(f"  Cases  — 7d        : {result['cases_7d']:,}")
    print(f"  Cases  — 14d       : {result['cases_14d']:,}")
    print(f"  Cases  — 30d       : {result['cases_30d']:,}")
    print("-" * 50)
    print(f"  Deaths — 7d        : {result['deaths_7d']:,}")
    print(f"  Deaths — 14d       : {result['deaths_14d']:,}")
    print(f"  Deaths — 30d       : {result['deaths_30d']:,}")
    print("=" * 50)

    # ── Generate visual reports ──
    print("\n[STEP 3] Generating visual charts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use("dark_background")

    slug = disease_name.replace(" ", "_")
    state_slug = state_name.replace(" ", "_")
    
    # Generate the forecast chart
    path_forecast = os.path.join(OUTPUT_DIR, f"{slug}_{state_slug}_forecast.png")
    _chart_forecast(result, state_name, path_forecast)
    print(f"      → Saved forecast curve: {path_forecast}")

    # Generate the risk gauge chart
    path_risk = os.path.join(OUTPUT_DIR, f"{slug}_{state_slug}_risk.png")
    _chart_risk(result, path_risk)
    print(f"      → Saved risk gauge:     {path_risk}")

    # Generate SEIR curve (if SEIR was run)
    seir_generated = False
    if result.get("seir_curve"):
        path_seir = os.path.join(OUTPUT_DIR, f"{slug}_{state_slug}_seir.png")
        seir_generated = _chart_seir(result, state_name, path_seir)
        if seir_generated:
            print(f"      → Saved SEIR curves:    {path_seir}")

    # Generate Feature Importance (if ML was run)
    features_generated = False
    if result.get("feature_importance"):
        path_features = os.path.join(OUTPUT_DIR, f"{slug}_{state_slug}_features.png")
        features_generated = _chart_features(result, path_features)
        if features_generated:
            print(f"      → Saved feature importance: {path_features}")

    # Generate full dashboard summary
    path_summary = os.path.join(OUTPUT_DIR, f"{slug}_{state_slug}_dashboard.png")
    _chart_summary(
        result,
        state_name,
        metrics=None,
        seir_available=(seir_generated if result.get("seir_curve") else False),
        features_available=(features_generated if result.get("feature_importance") else False),
        path=path_summary,
    )
    
    print(f"\n✅ All done! The full unified dashboard is saved at:")
    print(f"   {os.path.abspath(path_summary)}")

if __name__ == "__main__":
    main()
