"""
hpo_optuna.py — Bayesian hyperparameter optimisation for RF-Sentinel models.

Uses Optuna's TPE sampler to search XGBoost and 1D-CNN hyperparameter spaces.
Every trial is logged as a nested MLflow run so the full search history is
visible at http://localhost:5000 after running: mlflow ui

Usage
-----
    python -m layer4_tracking.hpo_optuna --model xgb   --trials 20
    python -m layer4_tracking.hpo_optuna --model cnn   --trials 10
    python -m layer4_tracking.hpo_optuna --model all   --trials 20
"""

import numpy as np
import optuna
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from loguru import logger

from layer1_data_ingestion.config import (
    CMAPSS_USEFUL_SENSORS, AI4I_FEATURE_COLS,
)
from layer1_data_ingestion.loaders import load_cmapss, load_ai4i
from layer1_data_ingestion.preprocessor import preprocess_cmapss, preprocess_ai4i
from layer3_models.xgb_classifier import RFSentinelXGB
from layer3_models.cnn1d_model import RFSentinelCNN1D
from layer4_tracking.mlflow_logger import (
    EXPERIMENT_XGB, EXPERIMENT_CNN, setup_mlflow,
)

# ══════════════════════════════════════════════════════════════
# RF-SENTINEL — Optuna Hyperparameter Optimisation
# ══════════════════════════════════════════════════════════════
#
# WHAT IS THIS FILE?
# ──────────────────
# Uses Optuna Bayesian search to find best hyperparameters
# for XGBoost and 1D-CNN automatically.
#
# HOW IT WORKS:
# ─────────────
# Trial 1: Optuna picks random params → trains → scores
# Trial 2: Optuna learns from trial 1 → smarter guess
# Trial 3: Learns from trials 1+2 → even smarter
# ...after N trials → best params found
#
# EVERY TRIAL IS LOGGED TO MLFLOW:
# ─────────────────────────────────
# Open http://localhost:5000 → Training runs
# Sort by val_f1 descending → top row = best params
#
# SEARCH SPACES:
# ──────────────
# XGBoost:
#   n_estimators     : 100 to 500
#   max_depth        : 3 to 8
#   learning_rate    : 0.01 to 0.3 (log scale)
#   subsample        : 0.6 to 1.0
#   colsample_bytree : 0.6 to 1.0
#   reg_alpha        : 0.0 to 2.0
#   reg_lambda       : 0.5 to 5.0
#
# CNN:
#   dropout     : 0.1 to 0.5
#   lr          : 0.0001 to 0.01 (log scale)
#   batch_size  : 32, 64, 128
#   n_epochs    : 15 to 30
# ══════════════════════════════════════════════════════════════

# Suppress per-trial Optuna logs — summary is printed by the study callback
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Function 1: objective_xgb ─────────────────────────────────────────────────

def objective_xgb(
    trial: optuna.Trial,
    X_train, y_train,
    X_val,   y_val,
    feature_names: list,
) -> float:
    """
    Optuna objective for XGBoost — called once per trial.

    Optuna passes a Trial object that suggests hyperparameter values from the
    defined search space. TPE uses previous trial results to bias sampling
    toward regions that yielded higher val_f1.

    Each trial is logged as a nested MLflow run so every combination of params
    and its resulting score is permanently stored and comparable in the UI.

    Parameters
    ----------
    trial        : Optuna Trial object providing suggest_* methods
    X_train, y_train : training features and labels
    X_val,   y_val   : validation features and labels
    feature_names    : column names for SHAP / feature importance

    Returns
    -------
    float  val_f1 — Optuna maximises this value across trials
    """
    # Step 1: Optuna suggests hyperparameters from the defined search space
    params = {
        "n_estimators":     trial.suggest_int(   "n_estimators",     100,  500),
        "max_depth":        trial.suggest_int(   "max_depth",        3,    8),
        "learning_rate":    trial.suggest_float( "learning_rate",    0.01, 0.3,  log=True),
        "subsample":        trial.suggest_float( "subsample",        0.6,  1.0),
        "colsample_bytree": trial.suggest_float( "colsample_bytree", 0.6,  1.0),
        "reg_alpha":        trial.suggest_float( "reg_alpha",        0.0,  2.0),
        "reg_lambda":       trial.suggest_float( "reg_lambda",       0.5,  5.0),
    }

    # Step 2: Train model with these params
    model = RFSentinelXGB()
    model.build(**params)
    metrics = model.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_names,
    )
    val_f1 = metrics["val_score"]

    # Step 3: Log this trial to MLflow as a nested run under the parent HPO run
    with mlflow.start_run(
        run_name=f"xgb_trial_{trial.number}",
        nested=True,
    ):
        mlflow.log_params(params)
        mlflow.log_metric("val_f1",       val_f1)
        mlflow.log_metric("train_f1",     metrics["train_score"])
        mlflow.log_metric("trial_number", trial.number)
        mlflow.set_tag("type", "optuna_trial")

    return val_f1


# ── Function 2: objective_cnn ─────────────────────────────────────────────────

def objective_cnn(
    trial: optuna.Trial,
    train_df,
    val_df,
) -> float:
    """
    Optuna objective for 1D-CNN — called once per trial.

    CNN trials are expensive on CPU (~3 min each), so the default n_trials
    is kept at 10. The epoch count is also part of the search space because
    early-stopping would complicate the sliding-window DataLoader setup.

    Each trial is logged as a nested MLflow run.

    Parameters
    ----------
    trial    : Optuna Trial object
    train_df : training DataFrame (engine-level split, raw sensor rows)
    val_df   : validation DataFrame

    Returns
    -------
    float  val_accuracy — Optuna maximises this value across trials
    """
    # Step 1: Optuna suggests architecture and training params
    params = {
        "dropout":    trial.suggest_float(      "dropout",    0.1, 0.5),
        "lr":         trial.suggest_float(      "lr",         0.0001, 0.01, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    n_epochs = trial.suggest_int("n_epochs", 15, 30)

    # Step 2: Train CNN with suggested params
    model = RFSentinelCNN1D()
    model.build(**params)
    metrics = model.train(
        train_df, None, val_df, None,
        n_epochs=n_epochs,
        batch_size=params["batch_size"],
    )
    val_acc = metrics["val_score"]

    # Step 3: Log trial to MLflow as a nested run
    with mlflow.start_run(
        run_name=f"cnn_trial_{trial.number}",
        nested=True,
    ):
        mlflow.log_params(params)
        mlflow.log_param( "n_epochs",       n_epochs)
        mlflow.log_metric("val_accuracy",   val_acc)
        mlflow.log_metric("train_accuracy", metrics["train_score"])
        mlflow.log_metric("trial_number",   trial.number)
        mlflow.set_tag("type", "optuna_trial")

    return val_acc


# ── Function 3: run_xgb_hpo ───────────────────────────────────────────────────

def run_xgb_hpo(
    dataset: str = "FD001",
    target:  str = "binary",
    n_trials: int = 20,
) -> tuple[optuna.Study, dict, float]:
    """
    Run Optuna hyperparameter search for XGBoost and log all trials to MLflow.

    The parent MLflow run records overall HPO config and best results.
    Each individual trial is a nested child run so they can be filtered and
    compared independently in the MLflow UI.

    After the study completes, the best-params model is retrained on the full
    training set and saved to models/ so it can be loaded immediately.

    Parameters
    ----------
    dataset  : CMAPSS subset ("FD001"–"FD004") for binary, ignored for AI4I
    target   : "binary" for CMAPSS, "multiclass" for AI4I
    n_trials : number of Optuna trials to run

    Returns
    -------
    (study, best_params, best_val_f1)
    """
    # Step 1: Setup MLflow and load/preprocess data
    setup_mlflow()

    if target == "binary":
        data          = load_cmapss(dataset)
        processed     = preprocess_cmapss(data)
        feature_names = CMAPSS_USEFUL_SENSORS
    else:
        data          = load_ai4i()
        processed     = preprocess_ai4i(data, target="multiclass")
        feature_names = AI4I_FEATURE_COLS

    X_train = processed["X_train"]
    y_train = processed["y_train"]
    X_val   = processed["X_val"]
    y_val   = processed["y_val"]

    # Step 2: Create Optuna study — TPE maximises val_f1 across trials
    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgb_{dataset}_{target}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Step 3: Run trials inside an MLflow parent run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(EXPERIMENT_XGB)

    with mlflow.start_run(
        run_name=f"xgb_hpo_{dataset}_{target}_{timestamp}"
    ):
        mlflow.log_param("n_trials",  n_trials)
        mlflow.log_param("dataset",   dataset)
        mlflow.log_param("target",    target)
        mlflow.log_param("optimizer", "optuna_TPE")

        logger.info(
            f"Starting XGBoost HPO — "
            f"{n_trials} trials on {dataset} ({target})"
        )

        study.optimize(
            lambda trial: objective_xgb(
                trial, X_train, y_train, X_val, y_val, feature_names
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        # Log best results to the parent run for quick comparison
        best_params = study.best_params
        best_val_f1 = study.best_value

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.set_tag("type", "hpo_parent_run")

        # Plot optimisation history
        try:
            fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title(f"XGBoost HPO — {dataset} {target}")
            plt.tight_layout()
            mlflow.log_figure(fig1, "hpo_optimization_history.png")
            plt.close()
        except Exception as exc:
            logger.warning(f"HPO history plot skipped: {exc}")

        # Plot parameter importances (requires ≥2 completed trials)
        try:
            fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            mlflow.log_figure(fig2, "hpo_param_importances.png")
            plt.close()
        except Exception:
            pass

    # Step 4: Log best params summary
    logger.success(f"XGBoost HPO complete — best val_f1={best_val_f1:.4f}")
    logger.info("Best params:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    # Step 5: Retrain and save the best model so it's immediately usable
    best_model = RFSentinelXGB()
    best_model.model_name = f"xgb_best_{dataset}_{target}"
    best_model.build(**best_params)
    best_model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    best_model.save()
    logger.success(f"Best model saved: xgb_best_{dataset}_{target}.pkl")

    return study, best_params, best_val_f1


# ── Function 4: run_cnn_hpo ───────────────────────────────────────────────────

def run_cnn_hpo(n_trials: int = 10) -> tuple[optuna.Study, dict, float]:
    """
    Run Optuna hyperparameter search for the 1D-CNN.

    Default n_trials=10 (not 20) because each CNN trial takes ~3 min on CPU.
    10 trials ≈ 30 minutes, which is reasonable for a local run.
    Increase n_trials when running on GPU.

    Parameters
    ----------
    n_trials : number of Optuna trials (default 10 for CPU budget)

    Returns
    -------
    (study, best_params, best_val_acc)
    """
    # Step 1: Setup MLflow and split engines for train/val
    setup_mlflow()
    data      = load_cmapss("FD001")
    train_raw = data["train_raw"]
    all_units = train_raw["unit_id"].unique()
    np.random.seed(42)
    np.random.shuffle(all_units)
    n_train  = int(len(all_units) * 0.8)
    train_df = train_raw[train_raw["unit_id"].isin(all_units[:n_train])]
    val_df   = train_raw[train_raw["unit_id"].isin(all_units[n_train:])]

    # Step 2: Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="cnn_cmapss_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Step 3: Run trials inside an MLflow parent run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(EXPERIMENT_CNN)

    with mlflow.start_run(run_name=f"cnn_hpo_{timestamp}"):
        mlflow.log_param("n_trials",  n_trials)
        mlflow.log_param("dataset",   "NASA_CMAPSS_FD001")
        mlflow.log_param("optimizer", "optuna_TPE")

        logger.info(
            f"Starting CNN HPO — {n_trials} trials "
            f"(this takes ~{n_trials * 3} minutes on CPU)"
        )

        study.optimize(
            lambda trial: objective_cnn(trial, train_df, val_df),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params  = study.best_params
        best_val_acc = study.best_value

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.set_tag("type", "hpo_parent_run")

        try:
            fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title("CNN HPO — CMAPSS FD001")
            plt.tight_layout()
            mlflow.log_figure(fig1, "cnn_hpo_optimization_history.png")
            plt.close()
        except Exception:
            pass

    logger.success(f"CNN HPO complete — best val_acc={best_val_acc:.4f}")
    logger.info("Best CNN params:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    return study, best_params, best_val_acc


# ── Function 5: run_full_hpo ──────────────────────────────────────────────────

def run_full_hpo(
    xgb_trials: int = 20,
    cnn_trials:  int = 10,
) -> dict:
    """
    Run HPO for XGBoost (CMAPSS + AI4I) and CNN in sequence.

    Runs three studies total:
        1. XGBoost binary on CMAPSS FD001
        2. XGBoost multiclass on AI4I 2020
        3. 1D-CNN on CMAPSS FD001

    All results are logged to MLflow and the best XGBoost models are saved
    to models/ automatically.

    Parameters
    ----------
    xgb_trials : trials per XGBoost study (default 20)
    cnn_trials : trials for CNN study (default 10, CPU-budget limited)

    Returns
    -------
    dict  with best_params and best scores for each model
    """
    print("RF-Sentinel — Full Hyperparameter Optimisation")
    print("=" * 55)
    print(f"XGBoost trials : {xgb_trials}")
    print(f"CNN trials     : {cnn_trials} (~{cnn_trials * 3} min)")
    print("All trials logged to MLflow")
    print("=" * 55)

    results = {}

    print("1/3 XGBoost CMAPSS binary HPO...")
    study1, params1, score1 = run_xgb_hpo("FD001", "binary", xgb_trials)
    results["xgb_cmapss"] = {"best_params": params1, "best_val_f1": score1}

    print("2/3 XGBoost AI4I multiclass HPO...")
    study2, params2, score2 = run_xgb_hpo("AI4I", "multiclass", xgb_trials)
    results["xgb_ai4i"] = {"best_params": params2, "best_val_f1": score2}

    print("3/3 CNN HPO (slow on CPU)...")
    study3, params3, score3 = run_cnn_hpo(cnn_trials)
    results["cnn"] = {"best_params": params3, "best_val_acc": score3}

    print()
    print("=" * 55)
    print("HPO COMPLETE — Best Results")
    print("=" * 55)
    print(f"XGBoost CMAPSS best Val F1  : {score1:.4f}")
    print(f"XGBoost AI4I   best Val F1  : {score2:.4f}")
    print(f"CNN            best Val Acc : {score3:.4f}")
    print()
    print("View all trials: http://localhost:5000")
    print("Sort by val_f1 to find best run")
    print("=" * 55)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RF-Sentinel Optuna HPO — optimise XGBoost or 1D-CNN"
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "cnn", "all"],
        default="xgb",
        help="Which model to optimise",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )
    args = parser.parse_args()

    if args.model == "xgb":
        run_xgb_hpo("FD001", "binary", args.trials)
    elif args.model == "cnn":
        run_cnn_hpo(args.trials)
    elif args.model == "all":
        run_full_hpo(args.trials, min(args.trials, 10))
