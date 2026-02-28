# Cybersecurity Attack Detection

Project 1 deliverable for cybersecurity traffic classification using Python machine learning, with an end-to-end workflow from EDA to deployment.

## Project Goal

Build and evaluate a multiclass attack detector for:
- `DDoS`
- `Intrusion`
- `Malware`

The project includes:
- notebook-based analysis and modeling,
- engineered-feature training pipeline,
- model export as a `.pkl` bundle,
- Streamlit web app for manual and CSV-based inference,
- LaTeX final report.

## Repository Structure

```text
GoogleAG/
├── mlcyber.ipynb                    # Main notebook (EDA, feature eng, modeling, evaluation)
├── app.py                           # Streamlit deployment app
├── requirements.txt                 # Python dependencies
├── project_report.tex               # Final LaTeX report (5–10 pages)
├── reconstructed_commit_plan.txt    # Reconstructed timeline commit helper
├── train_model.py / train_hybrid.py / train_legendary.py
├── check_*.py, inspect_*.py         # Analysis and diagnostics helpers
└── *.csv / *.txt                    # EDA exports and metric artifacts
```

## Environment Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Web App

```powershell
python -m streamlit run app.py
```

If you get a missing package error (for example `lightgbm`), re-run:

```powershell
pip install -r requirements.txt
```

## Methodology Summary

### 1) Exploratory Data Analysis
- Distribution checks for target and key fields.
- Numeric and categorical profiling.
- Relationship exploration (protocol, temporal, and traffic patterns).

### 2) Feature Engineering & Selection
- Temporal features (hour/day/weekend/off-hours).
- Port and payload behavior features.
- IP decomposition and transformations.
- Device information extraction (browser/OS/device-type signals).
- One-hot encoding and strict feature-schema alignment for inference.

### 3) Modeling & Evaluation
- LightGBM multiclass training.
- SMOTE-Tomek applied on training split.
- Comparison between:
   - engineered-rule target,
   - original mapped target (`Attack Type` → 3 classes).
- Evaluation with accuracy, macro-F1, and class-wise metrics.

## Results Snapshot

- Engineered-target setup reached strong internal performance after tuning.
- When compared against original mapped labels, scores were much lower and close across models, indicating limited alignment between generated and original labeling semantics.
- This comparison is documented in:
   - notebook analysis,
   - report tables in `project_report.tex`,
   - app-level comparison panels.

## Deployment Overview

The Streamlit app supports:
- single/manual prediction,
- CSV batch scoring,
- downloadable prediction output,
- side-by-side `Predicted` vs `Generated` comparison,
- confidence and class-distribution diagnostics,
- quick preset profiles for attack scenarios.

## Deliverables

- `mlcyber.ipynb` (main technical notebook)
- `app.py` (deployed interface)
- model `.pkl` bundle (generated during notebook workflow)
- `project_report.pdf` (submission report)
- `README.md` (this file)
