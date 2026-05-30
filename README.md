# AI-Driven RF Component Failure Root Cause Analysis & Diagnosis System

An end-to-end machine learning pipeline that automatically diagnoses RF component failures, explains why they happened, and recommends exactly what to fix — ranked by confidence and estimated repair time.

Most predictive maintenance systems stop at "this component will fail." This project goes further: it tells you the root cause and gives you a prioritised repair plan backed by a domain knowledge graph.

---

## What it does

You feed it sensor readings from an RF component. It returns:

- **What failed** — XGBoost + PyTorch 1D-CNN predict the failure type with confidence score
- **Why it failed** — SHAP values identify which sensors drove the prediction and by how much
- **When it appeared** — GradCAM highlights which measurement cycles triggered the CNN
- **What to do** — A knowledge graph traversal produces ranked P1/P2/P3 repair actions with estimated time and confidence

Example output for a degraded engine:

```
Device        : DUT_FD001_sample0
Predicted     : sensor_degradation (99.9% confidence)
Primary cause : Bearing Wear

Repair Actions:
  [P1] Inspect and Replace Bearing Assembly  —  4 hours  —  99.7%
  [P2] Replace Internal Seals                —  6 hours  —  78.7%
  [P3] Re-torque RF Connector                —  15 min   —  77.8%
```

---

## Results

| Model | Metric | Score |
|-------|--------|-------|
| XGBoost CMAPSS (baseline) | Val F1 weighted | 0.9521 |
| XGBoost CMAPSS (after HPO) | Val F1 weighted | 0.9589 |
| XGBoost AI4I 6-class | Val F1 weighted | 0.9477 |
| PyTorch 1D-CNN | Val Accuracy | 0.8666 |
| Ensemble (55% XGB + 45% CNN) | Val F1 weighted | 0.9466 |

Optuna hyperparameter optimisation over 20 trials improved XGBoost Val F1 from 0.9521 to 0.9589 — a gain of +0.0068.

---

## Datasets

Three public datasets are used, each mapped to a unified RF parameter schema:

**NASA CMAPSS — Turbofan Engine Degradation**
- 709 engines across 4 sub-datasets (FD001-FD004)
- 160,359 rows, 14 sensor measurements per cycle
- Binary label: fail if Remaining Useful Life <= 30 cycles
- Download: https://www.kaggle.com/datasets/behrad3d/nasa-cmapss

**UCI SECOM — Semiconductor Manufacturing**
- 1,567 samples, 562 features, 6.6% failure rate
- Extreme class imbalance handled with SMOTE
- Download: https://archive.ics.uci.edu/dataset/179/secom

**Kaggle AI4I 2020 — Predictive Maintenance**
- 10,000 samples, 5 named failure types
- TWF, HDF, PWF, OSF, RNF failure modes
- Download: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

---

## Architecture

The system is built as 6 independent layers plus a Streamlit dashboard:

```
Layer 1  Data Ingestion & Unification
         Load 3 datasets, preprocess, feature engineer, unify to 8 RF params
         Output: 32,198 rows, 4 parquet files

Layer 2  EDA & Visualisation
         48 plots across all datasets — degradation waves, failure distributions, correlations
         Output: outputs/eda/

Layer 3  Dual-Model AI Engine
         XGBoost (tabular) + PyTorch 1D-CNN (temporal sequences) + soft-vote ensemble
         Output: 4 saved models in models/

Layer 4  MLflow + Optuna Experiment Tracking
         Every training run logged with params, metrics, plots, artifacts
         Optuna Bayesian search over 20 trials finds best hyperparameters
         Output: mlruns/ (view at http://localhost:5000)

Layer 5  Explainability — SHAP + GradCAM
         SHAP: which sensors caused failure (waterfall, beeswarm, bar, decision plots)
         GradCAM: which time cycles triggered CNN prediction (heatmap over 30-cycle window)
         Output: outputs/explainability/

Layer 6  Knowledge Graph & Repair Recommender
         NetworkX graph: 42 nodes (16 symptoms + 12 causes + 14 repairs), 53 edges
         SHAP top features traverse graph to ranked P1/P2/P3 repair actions
         Output: diagnosis JSON + diagnosis card PNG + interactive pyvis HTML

Streamlit  5-Page Dashboard
         EDA Explorer, RCA Analyzer, Model Compare, Knowledge Graph, Live Simulator
         Run: streamlit run streamlit_app/app.py
```

---

## Project Structure

```
rf_sentinel/
├── data/
│   ├── raw/                          <- place downloaded datasets here
│   └── processed/                    <- generated parquet files (gitignored)
├── layer1_data_ingestion/            <- config, loaders, preprocessor, feature engineering
├── layer2_eda/                       <- EDA scripts for each dataset
├── layer3_models/                    <- base model, XGBoost, 1D-CNN, ensemble
├── layer4_tracking/                  <- MLflow logger, Optuna HPO
├── layer5_explainability/            <- SHAP explainer, GradCAM
├── layer6_knowledge_graph/           <- build graph, SHAP-to-graph linker, recommender
├── models/                           <- saved trained models (.pkl, .pt, .json)
├── outputs/                          <- all generated plots and reports
├── notebooks/                        <- Jupyter notebooks for layers 1-3
├── streamlit_app/                    <- app.py + 5 pages
└── pyproject.toml
```

---

## Setup

**Requirements:** Python 3.12, Git

```bash
# Clone the repository
git clone https://github.com/OmPatil2806/AI-Driven-RF-Component-Failure-Root-Cause-Analysis-Diagnosis-System.git
cd AI-Driven-RF-Component-Failure-Root-Cause-Analysis-Diagnosis-System

# Create virtual environment
python -m venv rf_env

# Activate (Windows)
rf_env\Scripts\activate

# Activate (Mac/Linux)
source rf_env/bin/activate

# Install dependencies
pip install -e .
```

**Download datasets** and place them in `data/raw/`:
- NASA CMAPSS: train_FD001.txt through train_FD004.txt, test files, RUL files
- UCI SECOM: secom.data, secom_labels.data, secom.names
- Kaggle AI4I: ai4i2020.csv

---

## Running the Project

**Step 1 — Run the full pipeline (first time only)**

```bash
# Layer 1: ingest and unify data
python -m layer1_data_ingestion.pipeline

# Layer 2: generate EDA plots
python -m layer2_eda.eda_runner

# Layer 3: train all models (CNN takes ~3 minutes)
python -c "
from layer1_data_ingestion.loaders import load_cmapss
from layer1_data_ingestion.preprocessor import preprocess_cmapss
from layer1_data_ingestion.config import CMAPSS_USEFUL_SENSORS
from layer3_models.xgb_classifier import RFSentinelXGB
data = load_cmapss('FD001')
p = preprocess_cmapss(data)
m = RFSentinelXGB()
m.build()
m.train(p['X_train'], p['y_train'], p['X_val'], p['y_val'], feature_names=CMAPSS_USEFUL_SENSORS)
m.save()
print('XGBoost done — Val F1:', round(m.val_score, 4))
"

# Layer 4: log models to MLflow + run HPO
python -c "from layer4_tracking.mlflow_logger import log_existing_models; log_existing_models()"
python -c "from layer4_tracking.hpo_optuna import run_xgb_hpo; run_xgb_hpo('FD001', 'binary', n_trials=20)"

# Layer 5: run SHAP and GradCAM analysis
python -c "from layer5_explainability.shap_explainer import run_shap_analysis; run_shap_analysis(n_samples=5)"
python -c "from layer5_explainability.grad_cam import run_gradcam_analysis; run_gradcam_analysis(n_samples=5)"

# Layer 6: build knowledge graph and run full diagnosis
python -m layer6_knowledge_graph.build_graph
python -m layer6_knowledge_graph.recommender
```

**Step 2 — Launch the dashboard (Terminal 1: MLflow, Terminal 2: Streamlit)**

Terminal 1:
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

Terminal 2:
```bash
streamlit run streamlit_app/app.py
# Open http://localhost:8501
```

---

## Streamlit Dashboard

Five pages, each with a distinct purpose:

**EDA Explorer** — Browse all 48 EDA plots across NASA CMAPSS, UCI SECOM, Kaggle AI4I, and the unified RF schema. Select dataset and plot type from dropdowns. Each plot includes an insight explanation.

**RCA Analyzer** — The main diagnostic page. Upload a CSV with 14 sensor columns or use manual input sliders. The app runs the full pipeline: prediction, SHAP explanation, knowledge graph traversal, and ranked repair actions. CSV format: columns s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21.

**Model Compare** — View all model scores side by side, Plotly bar chart, Optuna HPO best parameters, and all saved training plots (confusion matrices, SHAP summaries, training curves).

**Knowledge Graph** — Interactive pyvis graph embedded in the browser. Click and drag nodes. Blue = symptom, Red = cause, Green = repair. Select sensors and weights to query the graph directly for repair recommendations.

**Live Simulator** — Drag sliders for all 14 sensor values. The model predicts instantly, SHAP bar chart updates, and a failure probability gauge responds in real time. Four preset scenarios: Healthy, Early Degradation, Near Failure, Critical Failure.

---

## How SHAP Connects to Repairs

This is the core innovation of the project. SHAP values are not just for visualisation — they drive the repair recommendation engine.

```
Step 1: XGBoost predicts failure with confidence
Step 2: SHAP calculates per-sensor contributions
        s11 = -1.235  (strong failure signal)
        s15 = -0.439  (moderate failure signal)
        s9  = -0.325  (moderate failure signal)

Step 3: Top-3 features map to graph symptom nodes
        s11 -> HPC Pressure Drop
        s15 -> Bypass Ratio Reduction
        s9  -> Core Speed Reduction

Step 4: Graph traversal finds all cause-repair paths
        HPC Pressure Drop -> Bearing Wear (weight 0.85) -> Inspect Bearing (weight 0.95)

Step 5: Final score = SHAP weight x cause weight x repair weight
        1.235 x 0.85 x 0.95 = 0.9973 -> P1 at 99.7% confidence
```

---

## Why Two Models

**XGBoost** is used because it consistently outperforms deep learning on structured tabular sensor data, supports native TreeSHAP for fast exact explanations, trains in under 2 seconds, and handles class imbalance via sample weights. It achieved the highest individual validation F1 of 0.9589 after HPO.

**PyTorch 1D-CNN** is used because XGBoost treats every row independently — it has no concept of time. The CNN receives 30 consecutive measurement cycles as a single input window (shape: batch x 14 sensors x 30 cycles) and learns temporal degradation patterns. It also enables GradCAM, which produces time-step heatmaps that tree models cannot generate.

The ensemble combines both: 55% XGBoost weight for accuracy, 45% CNN weight for temporal knowledge. Soft voting averages class probabilities rather than hard labels, producing more reliable confidence scores.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML models | XGBoost 2.x, PyTorch 2.x |
| Explainability | SHAP, GradCAM |
| Experiment tracking | MLflow, Optuna |
| Knowledge graph | NetworkX, pyvis |
| Dashboard | Streamlit, Plotly |
| Data processing | pandas, numpy, scikit-learn |
| Language | Python 3.12 |

---

## Author

**Om Patil**
GitHub: https://github.com/OmPatil2806

Certificate: Modern Communication Systems — Udemy, Uplatz Training (46.5 hours, May 2026)
