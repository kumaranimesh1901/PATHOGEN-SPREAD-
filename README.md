# 🦠 PATHOGEN-SPREAD

**AI-driven pathogen spread prediction using hybrid epidemiological modeling and deep learning, with built-in ethical safeguards.**

This project combines a classical **SEIR compartmental model** with an **LSTM neural network** to forecast COVID-19 case counts. It goes beyond raw prediction by incorporating **Explainable AI (XAI)**, **bias auditing**, and **ethical risk scoring** — ensuring the model is not only accurate but also transparent and fair.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation & Results](#evaluation--results)
- [Ethical AI & Responsible Forecasting](#ethical-ai--responsible-forecasting)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## ✨ Features

| Category | Capability |
|---|---|
| **Data** | Automated ingestion from [Hugging Face Datasets](https://huggingface.co/datasets/maharshipandya/covid-19-coronavirus-pandemic-dataset) |
| **Modeling** | SEIR ODE simulation + LSTM time-series forecasting |
| **Preprocessing** | MinMax scaling, sliding-window sequence generation, 80/20 train-test split |
| **Training** | Configurable LSTM (layers, hidden size, dropout, learning rate, epochs) |
| **Evaluation** | MAE, RMSE, R² metrics with actual-vs-predicted visualizations |
| **Explainability** | SHAP GradientExplainer for per-feature importance |
| **Fairness** | Per-continent bias audit with fairness ratio reporting |
| **Ethics** | Confidence-weighted risk scoring and public safety alert generation |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                           │
│  Hugging Face ──► load_data.py ──► preprocess.py           │
│                     (raw CSV)      (scaled sequences)      │
└──────────────────────────┬─────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │      model.py           │
              │  ┌──────────────────┐   │
              │  │  SEIR ODE Model  │   │
              │  └──────────────────┘   │
              │  ┌──────────────────┐   │
              │  │  PathogenLSTM    │   │
              │  │  (2-layer LSTM)  │   │
              │  └──────────────────┘   │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   evaluate.py         xai.py        ethics.py
   (metrics &       (SHAP feature    (risk scores &
    plots)          importance)     safety alerts)
                                         │
                                         ▼
                                   bias_audit.py
                                 (fairness audit)
```

---

## 📂 Project Structure

```
PATHOGEN-SPREAD-/
├── load_data.py          # Download & clean COVID-19 dataset from Hugging Face
├── preprocess.py         # Scale features, create sliding-window sequences
├── model.py              # SEIR ODE model + PathogenLSTM neural network
├── train.py              # Training loop with configurable hyperparameters
├── evaluate.py           # Compute metrics (MAE, RMSE, R²) and plot results
├── xai.py                # SHAP-based explainability analysis
├── ethics.py             # Ethical risk scoring and public safety alerts
├── bias_audit.py         # Per-continent fairness audit
├── requirements.txt      # Python dependencies
├── data/                 # Raw CSV + preprocessed .npy arrays
│   ├── covid_raw.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── models/               # Saved model weights and scaler
│   ├── lstm_pathogen.pt
│   └── scaler.pkl
└── outputs/              # Evaluation results, plots, and reports
    ├── metrics.json
    ├── prediction_plot.png
    ├── predictions.csv
    ├── shap_plot.png
    └── bias_report.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/PATHOGEN-SPREAD-.git
cd PATHOGEN-SPREAD-

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔧 Usage

Run the full pipeline step-by-step:

### 1. Download Data

```bash
python load_data.py
```

Downloads the COVID-19 dataset from Hugging Face, filters relevant columns (`date`, `new_cases`, `new_deaths`, `total_cases`, `total_deaths`, `continent`), and saves to `data/covid_raw.csv`.

### 2. Preprocess

```bash
python preprocess.py --input data/covid_raw.csv --seq_len 14
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `data/covid_raw.csv` | Path to raw CSV |
| `--seq_len` | `14` | Sliding window length (days) |

Applies MinMax scaling, generates sliding-window sequences, splits 80/20 into train/test, and saves `.npy` arrays.

### 3. Train

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001 --hidden 128
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--hidden` | `128` | LSTM hidden layer size |
| `--device` | `cpu` | Device (`cpu` or `cuda`) |

Saves the trained model to `models/lstm_pathogen.pt`.

### 4. Evaluate

```bash
python evaluate.py
```

Computes MAE, RMSE, and R² on the test set. Generates `outputs/prediction_plot.png` and saves metrics to `outputs/metrics.json`.

### 5. Explainability (XAI)

```bash
python xai.py
```

Runs SHAP GradientExplainer on 50 test samples, computes per-feature importance, and saves `outputs/shap_plot.png`.

### 6. Ethics & Bias

```bash
python ethics.py               # Demo of ethical risk scoring
python bias_audit.py --predictions outputs/predictions.csv
```

- **`ethics.py`** — Computes confidence-weighted risk levels (LOW / MEDIUM / HIGH / CRITICAL) and generates public safety alert messages.
- **`bias_audit.py`** — Compares per-continent MAE against global MAE, flags groups where error exceeds 1.5× the global baseline, and reports a fairness ratio.

---

## 📊 Evaluation & Results

After running the pipeline, check the `outputs/` directory:

| File | Description |
|---|---|
| `metrics.json` | MAE, RMSE, R² scores |
| `prediction_plot.png` | Actual vs. predicted new cases over time |
| `shap_plot.png` | SHAP feature importance bar chart |
| `bias_report.txt` | Per-continent fairness audit results |
| `predictions.csv` | Raw prediction values for further analysis |

---

## 🤖 Ethical AI & Responsible Forecasting

This project embeds responsible AI principles directly into the pipeline:

- **Explainability** — SHAP values reveal which features drive predictions, preventing "black box" decision-making.
- **Bias Auditing** — Automated detection of disproportionate error across geographic regions ensures the model doesn't silently perform worse for underrepresented continents.
- **Risk Communication** — Confidence-adjusted risk scores translate raw predictions into actionable public health guidance, with clear alert levels and recommended actions.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Scientific Computing | NumPy, SciPy |
| Data Processing | Pandas |
| ML Utilities | scikit-learn, Joblib |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Data Source | Hugging Face Datasets |
| Progress Bars | tqdm |

---

## 📄 License

This project is open source. Add a license file to specify terms of use.

---

<p align="center">
  <i>Built with a commitment to transparent, fair, and ethical AI in public health.</i>
</p>
