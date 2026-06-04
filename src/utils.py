import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Metadata for synthetic countries
COUNTRY_METADATA: Dict[str, Dict[str, Any]] = {
    "USA": {
        "population": 331000000,
        "population_density": 36,        # people per sq km
        "healthcare_capacity": 2.9,      # beds per 1000 people
        "peak_day": 75,
        "wave_scale": 15000,
        "transmission_coeff": 1.2
    },
    "UK": {
        "population": 67000000,
        "population_density": 275,
        "healthcare_capacity": 2.5,
        "peak_day": 65,
        "wave_scale": 4000,
        "transmission_coeff": 1.4
    },
    "Germany": {
        "population": 83000000,
        "population_density": 240,
        "healthcare_capacity": 8.0,
        "peak_day": 60,
        "wave_scale": 3500,
        "transmission_coeff": 1.1
    },
    "India": {
        "population": 1380000000,
        "population_density": 420,
        "healthcare_capacity": 0.53,
        "peak_day": 105,
        "wave_scale": 45000,
        "transmission_coeff": 1.6
    },
    "Brazil": {
        "population": 212000000,
        "population_density": 25,
        "healthcare_capacity": 2.2,
        "peak_day": 90,
        "wave_scale": 12000,
        "transmission_coeff": 1.3
    }
}

def generate_synthetic_data(output_path: str, num_days: int = 180) -> pd.DataFrame:
    """
    Generates a highly realistic synthetic multi-country COVID-19 dataset with
    epidemiological patterns, lockdown mobility changes, lag-based deaths and recoveries.
    """
    np.random.seed(42)
    start_date = datetime(2020, 3, 1)
    date_list = [start_date + timedelta(days=i) for i in range(num_days)]
    
    records = []
    
    for country, meta in COUNTRY_METADATA.items():
        pop = meta["population"]
        density = meta["population_density"]
        beds = meta["healthcare_capacity"]
        peak = meta["peak_day"]
        scale = meta["wave_scale"]
        trans = meta["transmission_coeff"]
        
        # Initialize running cumulative totals
        running_confirmed = 0
        running_deaths = 0
        running_recovered = 0
        
        # We will generate daily new cases based on a double Gaussian wave to represent two waves
        for day in range(num_days):
            date_str = date_list[day].strftime("%Y-%m-%d")
            
            # Base mobility index simulating lockdowns: decreases when cases are high
            # We construct a mobility index that drops as cases climb, representing social distancing
            # First wave peak
            wave1 = np.exp(-((day - peak) ** 2) / (2 * 20 ** 2))
            # Second wave peak (smaller or larger depending on country)
            wave2 = 0.6 * np.exp(-((day - (peak + 60)) ** 2) / (2 * 15 ** 2))
            
            wave_signal = wave1 + wave2
            
            # Mobility index: starts at 100%, drops to 30-50% during peak outbreaks, then recovers
            mobility = 100.0 - (wave_signal * 50.0) - np.random.normal(0, 2)
            mobility = np.clip(mobility, 20.0, 100.0)
            
            # Calculate daily new cases
            # Daily cases depend on transmission coefficient, mobility index (less mobility = fewer cases), and wave signal
            mobility_factor = mobility / 100.0
            daily_new_cases_base = scale * wave_signal * trans * (0.3 + 0.7 * mobility_factor)
            daily_new_cases = int(max(0, daily_new_cases_base + np.random.normal(0, scale * 0.05)))
            
            running_confirmed += daily_new_cases
            
            # Deaths: 1.5% to 3.0% of cases from ~8 days ago (simulating delay)
            if day > 8:
                past_cases = records[-8][4] if len(records) > 8 else daily_new_cases
                # Death rate increases slightly if healthcare capacity is low
                death_rate = 0.02 * (1.0 + (1.0 / (beds + 0.1)) * 0.1)
                daily_new_deaths = int(max(0, past_cases * death_rate + np.random.normal(0, past_cases * 0.005)))
            else:
                daily_new_deaths = int(daily_new_cases * 0.01)
                
            running_deaths += daily_new_deaths
            
            # Recovered: 95% of cases from ~14 days ago minus deaths
            if day > 14:
                past_cases_14 = records[-14][4] if len(records) > 14 else daily_new_cases
                daily_new_recovered = int(max(0, past_cases_14 * 0.95 - daily_new_deaths))
            else:
                daily_new_recovered = int(daily_new_cases * 0.8)
                
            running_recovered += daily_new_recovered
            
            # Ensure consistency: Recovered + Deaths <= Confirmed
            if running_recovered + running_deaths > running_confirmed:
                running_recovered = max(0, running_confirmed - running_deaths)
            
            records.append([
                date_str,
                country,
                int(running_confirmed),
                int(running_deaths),
                int(running_recovered),
                pop,
                density,
                beds,
                round(mobility, 2),
                daily_new_cases # helper for calculations
            ])
            
    df = pd.DataFrame(records, columns=[
        "date", "country", "confirmed_cases", "deaths", "recovered", 
        "population", "population_density", "healthcare_capacity", "mobility_index", "daily_new_cases"
    ])
    
    # Drop daily_new_cases from final public dataset to match exact requested columns
    df_out = df.drop(columns=["daily_new_cases"])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df_out)} rows at {output_path}")
    return df_out
