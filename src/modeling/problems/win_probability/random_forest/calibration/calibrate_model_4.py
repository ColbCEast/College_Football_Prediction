"""
Random Forest Model 4 — Probability Calibration Experiment

Purpose
-------
Evaluate whether post-hoc probability calibration improves the existing
Random Forest Model 4 without modifying the original model.

The original Model 4 remains completely unchanged.

Calibration methods
--------------------
1. Raw Model 4 probabilities
2. Sigmoid / Platt scaling
3. Isotonic regression

Experimental design
-------------------
- Model 4 itself is NOT retrained.
- Existing Model 4 validation predictions are used to fit the calibrators.
- The 2025 test predictions remain untouched during calibration fitting.
- Calibrators learned from 2023–2024 validation predictions are applied
  to the 2025 test probabilities.
- Validation metrics are used for exploratory method selection.
- Test metrics provide the out-of-sample comparison.

Primary probability metrics
---------------------------
- Log Loss
- Brier Score

Secondary calibration metrics
-----------------------------
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)

Discrimination metric
---------------------
- ROC AUC

Outputs
-------
models/win_probability/random_forest/calibration/model_4/
    calibration_comparison.csv
    validation_calibrated_predictions.csv
    test_calibrated_predictions.csv
    validation_calibration_bins.csv
    test_calibration_bins.csv
    calibration_summary.csv
    sigmoid_calibrator.joblib
    isotonic_calibrator.joblib

The original Model 4 directory is never modified.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

MODEL_NUMBER = 4

GAME_ID = "gameId"
SEASON = "season"
TARGET = "win_home_actual"
PROBABILITY = "win_home_probability"
PREDICTION = "win_home_prediction"
SPLIT = "split"

N_CALIBRATION_BINS = 10

# -------------------------------------------------------------------------
# Existing Model 4 artifacts
# -------------------------------------------------------------------------

MODEL_DIR = Path(
    "models/win_probability/random_forest/model_4"
)

VALIDATION_PATH = MODEL_DIR / "validation_predictions.csv"
TEST_PATH = MODEL_DIR / "test_predictions.csv"

# -------------------------------------------------------------------------
# Separate calibration output directory
# -------------------------------------------------------------------------

OUTPUT_DIR = Path(
    "models/win_probability/random_forest/calibration/model_4"
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions():
    """Load existing Model 4 validation and test predictions."""

    print("Loading existing Model 4 predictions...")

    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"  Validation: {validation.shape}")
    print(f"  Test:       {test.shape}")

    return validation, test


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_predictions(validation, test):
    """Validate the prediction datasets before calibration."""

    print("\nValidating prediction datasets...")

    required_columns = {
        GAME_ID,
        SEASON,
        TARGET,
        PROBABILITY,
        PREDICTION,
        SPLIT,
    }

    for name, df in [
        ("validation", validation),
        ("test", test),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name.capitalize()} predictions are missing required "
                f"columns: {sorted(missing)}"
            )

        if df[GAME_ID].duplicated().any():
            raise ValueError(
                f"{name.capitalize()} predictions contain duplicate "
                f"game IDs."
            )

        if df[TARGET].isna().any():
            raise ValueError(
                f"{name.capitalize()} target contains missing values."
            )

        if df[PROBABILITY].isna().any():
            raise ValueError(
                f"{name.capitalize()} probabilities contain missing values."
            )

        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(
                f"{name.capitalize()} target must contain only 0/1 values."
            )

        if not (
            (df[PROBABILITY] >= 0.0)
            & (df[PROBABILITY] <= 1.0)
        ).all():
            raise ValueError(
                f"{name.capitalize()} probabilities must be between 0 and 1."
            )

    # -------------------------------------------------------------------------
    # Validate expected splits
    # -------------------------------------------------------------------------

    validation_splits = set(validation[SPLIT].unique())
    test_splits = set(test[SPLIT].unique())

    if validation_splits != {"validation"}:
        raise ValueError(
            f"Unexpected validation split labels: {validation_splits}"
        )

    if test_splits != {"test"}:
        raise ValueError(
            f"Unexpected test split labels: {test_splits}"
        )

    # -------------------------------------------------------------------------
    # Validate temporal structure
    # -------------------------------------------------------------------------

    validation_seasons = sorted(validation[SEASON].unique())
    test_seasons = sorted(test[SEASON].unique())

    print(f"  Validation seasons: {validation_seasons}")
    print(f"  Test seasons:       {test_seasons}")

    if validation_seasons != [2023, 2024]:
        raise ValueError(
            "Expected validation seasons [2023, 2024], found "
            f"{validation_seasons}."
        )

    if test_seasons != [2025]:
        raise ValueError(
            "Expected test season [2025], found "
            f"{test_seasons}."
        )

    # -------------------------------------------------------------------------
    # Confirm validation and test are disjoint
    # -------------------------------------------------------------------------

    overlap = set(validation[GAME_ID]) & set(test[GAME_ID])

    if overlap:
        raise ValueError(
            f"Validation and test contain {len(overlap)} overlapping "
            "game IDs."
        )

    print("  Dataset validation PASSED.")


# =============================================================================
# CALIBRATION METHODS
# =============================================================================

def fit_sigmoid_calibrator(
    probabilities,
    targets,
):
    """
    Fit Platt/sigmoid calibration.

    Logistic regression is fit to the log-odds of the original model
    probabilities. Clipping prevents infinite values at exactly 0 or 1.
    """

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logit = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    calibrator.fit(
        logit,
        targets,
    )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator,
    probabilities,
):
    """Apply a fitted sigmoid calibrator."""

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logit = np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)

    calibrated = calibrator.predict_proba(logit)[:, 1]

    return np.clip(
        calibrated,
        1e-6,
        1 - 1e-6,
    )


def fit_isotonic_calibrator(
    probabilities,
    targets,
):
    """Fit isotonic regression calibration."""

    calibrator = IsotonicRegression(
        y_min=1e-6,
        y_max=1 - 1e-6,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probabilities,
        targets,
    )

    return calibrator


def apply_isotonic_calibrator(
    calibrator,
    probabilities,
):
    """Apply a fitted isotonic calibrator."""

    calibrated = calibrator.predict(probabilities)

    return np.clip(
        calibrated,
        1e-6,
        1 - 1e-6,
    )


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def calculate_calibration_bins(
    y_true,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
):
    """
    Calculate equal-width calibration bins.

    Each row represents one probability interval and includes:
    - Number of observations
    - Mean predicted probability
    - Observed win rate
    - Absolute calibration error
    - Squared calibration error
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_win_rate": np.nan,
                    "absolute_error": np.nan,
                    "squared_error": np.nan,
                }
            )
            continue

        mean_probability = probabilities[mask].mean()
        observed_rate = y_true[mask].mean()

        absolute_error = abs(
            mean_probability - observed_rate
        )

        squared_error = (
            mean_probability - observed_rate
        ) ** 2

        rows.append(
            {
                "bin": i + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_win_rate": observed_rate,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
            }
        )

    return pd.DataFrame(rows)


def calculate_ece(
    y_true,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
):
    """Calculate Expected Calibration Error."""

    bins = calculate_calibration_bins(
        y_true,
        probabilities,
        n_bins,
    )

    total = bins["count"].sum()

    if total == 0:
        return np.nan

    valid = bins["count"] > 0

    ece = (
        (
            bins.loc[valid, "count"]
            * bins.loc[valid, "absolute_error"]
        ).sum()
        / total
    )

    return ece


def calculate_mce(
    y_true,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
):
    """Calculate Maximum Calibration Error."""

    bins = calculate_calibration_bins(
        y_true,
        probabilities,
        n_bins,
    )

    valid = bins["count"] > 0

    if not valid.any():
        return np.nan

    return bins.loc[
        valid,
        "absolute_error",
    ].max()


# =============================================================================
# MODEL METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    probabilities,
):
    """Calculate probability and classification metrics."""

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "ece": calculate_ece(
            y_true,
            probabilities,
        ),
        "mce": calculate_mce(
            y_true,
            probabilities,
        ),
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_probability_set(
    name,
    y_true,
    probabilities,
):
    """Calculate and print metrics for one probability set."""

    metrics = calculate_metrics(
        y_true,
        probabilities,
    )

    print(f"\n{name}")

    print("-" * 60)

    for metric, value in metrics.items():
        print(f"  {metric:<22}: {value:.6f}")

    return metrics


# =============================================================================
# CALIBRATION PREDICTIONS
# =============================================================================

def create_prediction_output(
    original_df,
    raw_probabilities,
    sigmoid_probabilities,
    isotonic_probabilities,
):
    """Create a standardized calibration prediction dataset."""

    output = original_df[
        [
            GAME_ID,
            SEASON,
            TARGET,
            SPLIT,
        ]
    ].copy()

    output["raw_probability"] = raw_probabilities

    output["sigmoid_probability"] = (
        sigmoid_probabilities
    )

    output["isotonic_probability"] = (
        isotonic_probabilities
    )

    output["raw_prediction"] = (
        raw_probabilities >= 0.50
    ).astype(int)

    output["sigmoid_prediction"] = (
        sigmoid_probabilities >= 0.50
    ).astype(int)

    output["isotonic_prediction"] = (
        isotonic_probabilities >= 0.50
    ).astype(int)

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("RANDOM FOREST MODEL 4 — PROBABILITY CALIBRATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------

    validation, test = load_predictions()

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_predictions(
        validation,
        test,
    )

    # -------------------------------------------------------------------------
    # Extract data
    # -------------------------------------------------------------------------

    y_validation = validation[TARGET].to_numpy()
    p_validation = validation[PROBABILITY].to_numpy()

    y_test = test[TARGET].to_numpy()
    p_test = test[PROBABILITY].to_numpy()

    # -------------------------------------------------------------------------
    # Baseline evaluation
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BASELINE — RAW MODEL 4")
    print("=" * 80)

    validation_raw_metrics = evaluate_probability_set(
        "Validation — Raw Model 4",
        y_validation,
        p_validation,
    )

    test_raw_metrics = evaluate_probability_set(
        "Test — Raw Model 4",
        y_test,
        p_test,
    )

    # -------------------------------------------------------------------------
    # Fit sigmoid calibration
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FITTING SIGMOID CALIBRATION")
    print("=" * 80)

    print(
        "\nCalibration data:"
        "\n  Period: 2023–2024 validation"
        "\n  Rows:   "
        f"{len(validation):,}"
    )

    print(
        "\nThe 2025 test outcomes are NOT used to fit "
        "the calibrator."
    )

    sigmoid_calibrator = fit_sigmoid_calibrator(
        p_validation,
        y_validation,
    )

    validation_sigmoid = apply_sigmoid_calibrator(
        sigmoid_calibrator,
        p_validation,
    )

    test_sigmoid = apply_sigmoid_calibrator(
        sigmoid_calibrator,
        p_test,
    )

    print(
        "\nSigmoid calibration parameters:"
    )

    print(
        f"  Intercept: "
        f"{sigmoid_calibrator.intercept_[0]:.6f}"
    )

    print(
        f"  Coefficient: "
        f"{sigmoid_calibrator.coef_[0, 0]:.6f}"
    )

    # -------------------------------------------------------------------------
    # Fit isotonic calibration
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FITTING ISOTONIC CALIBRATION")
    print("=" * 80)

    isotonic_calibrator = fit_isotonic_calibrator(
        p_validation,
        y_validation,
    )

    validation_isotonic = apply_isotonic_calibrator(
        isotonic_calibrator,
        p_validation,
    )

    test_isotonic = apply_isotonic_calibrator(
        isotonic_calibrator,
        p_test,
    )

    # -------------------------------------------------------------------------
    # Evaluate calibrated models
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION CALIBRATION RESULTS")
    print("=" * 80)

    validation_sigmoid_metrics = evaluate_probability_set(
        "Validation — Sigmoid",
        y_validation,
        validation_sigmoid,
    )

    validation_isotonic_metrics = evaluate_probability_set(
        "Validation — Isotonic",
        y_validation,
        validation_isotonic,
    )

    print("\n" + "=" * 80)
    print("TEST CALIBRATION RESULTS")
    print("=" * 80)

    test_sigmoid_metrics = evaluate_probability_set(
        "Test — Sigmoid",
        y_test,
        test_sigmoid,
    )

    test_isotonic_metrics = evaluate_probability_set(
        "Test — Isotonic",
        y_test,
        test_isotonic,
    )

    # -------------------------------------------------------------------------
    # Build comparison table
    # -------------------------------------------------------------------------

    comparison_rows = [
        {
            "dataset": "validation",
            "method": "raw_model_4",
            **validation_raw_metrics,
        },
        {
            "dataset": "validation",
            "method": "sigmoid",
            **validation_sigmoid_metrics,
        },
        {
            "dataset": "validation",
            "method": "isotonic",
            **validation_isotonic_metrics,
        },
        {
            "dataset": "test",
            "method": "raw_model_4",
            **test_raw_metrics,
        },
        {
            "dataset": "test",
            "method": "sigmoid",
            **test_sigmoid_metrics,
        },
        {
            "dataset": "test",
            "method": "isotonic",
            **test_isotonic_metrics,
        },
    ]

    comparison = pd.DataFrame(
        comparison_rows
    )

    # -------------------------------------------------------------------------
    # Add changes relative to raw Model 4
    # -------------------------------------------------------------------------

    for dataset in ["validation", "test"]:

        raw_row = comparison[
            (comparison["dataset"] == dataset)
            & (comparison["method"] == "raw_model_4")
        ].iloc[0]

        mask = (
            (comparison["dataset"] == dataset)
            & (comparison["method"] != "raw_model_4")
        )

        comparison.loc[
            mask,
            "log_loss_change",
        ] = (
            comparison.loc[mask, "log_loss"]
            - raw_row["log_loss"]
        )

        comparison.loc[
            mask,
            "brier_score_change",
        ] = (
            comparison.loc[mask, "brier_score"]
            - raw_row["brier_score"]
        )

        comparison.loc[
            mask,
            "ece_change",
        ] = (
            comparison.loc[mask, "ece"]
            - raw_row["ece"]
        )

        comparison.loc[
            mask,
            "mce_change",
        ] = (
            comparison.loc[mask, "mce"]
            - raw_row["mce"]
        )

        comparison.loc[
            mask,
            "roc_auc_change",
        ] = (
            comparison.loc[mask, "roc_auc"]
            - raw_row["roc_auc"]
        )

    # Raw model has no change relative to itself.
    raw_mask = comparison["method"] == "raw_model_4"

    comparison.loc[
        raw_mask,
        [
            "log_loss_change",
            "brier_score_change",
            "ece_change",
            "mce_change",
            "roc_auc_change",
        ],
    ] = 0.0

    # -------------------------------------------------------------------------
    # Determine validation winner
    # -------------------------------------------------------------------------

    validation_candidates = comparison[
        comparison["dataset"] == "validation"
    ].copy()

    validation_candidates = validation_candidates.sort_values(
        "log_loss"
    )

    validation_winner = (
        validation_candidates.iloc[0]["method"]
    )

    # -------------------------------------------------------------------------
    # Determine test winners by metric
    # -------------------------------------------------------------------------

    test_candidates = comparison[
        comparison["dataset"] == "test"
    ].copy()

    test_best_log_loss = test_candidates.loc[
        test_candidates["log_loss"].idxmin(),
        "method",
    ]

    test_best_brier = test_candidates.loc[
        test_candidates["brier_score"].idxmin(),
        "method",
    ]

    test_best_ece = test_candidates.loc[
        test_candidates["ece"].idxmin(),
        "method",
    ]

    test_best_mce = test_candidates.loc[
        test_candidates["mce"].idxmin(),
        "method",
    ]

    test_best_auc = test_candidates.loc[
        test_candidates["roc_auc"].idxmax(),
        "method",
    ]

    # -------------------------------------------------------------------------
    # Print summary comparison
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CALIBRATION COMPARISON")
    print("=" * 80)

    display_columns = [
        "dataset",
        "method",
        "log_loss",
        "brier_score",
        "roc_auc",
        "ece",
        "mce",
    ]

    print(
        comparison[
            display_columns
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("CALIBRATION INTERPRETATION")
    print("=" * 80)

    print(
        f"\nValidation log-loss winner: "
        f"{validation_winner}"
    )

    print(
        "\n2025 Test winners:"
    )

    print(
        f"  Log Loss:      {test_best_log_loss}"
    )

    print(
        f"  Brier Score:   {test_best_brier}"
    )

    print(
        f"  ECE:           {test_best_ece}"
    )

    print(
        f"  MCE:           {test_best_mce}"
    )

    print(
        f"  ROC AUC:       {test_best_auc}"
    )

    # -------------------------------------------------------------------------
    # Create calibrated prediction files
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAVING CALIBRATED PREDICTIONS")
    print("=" * 80)

    validation_output = create_prediction_output(
        validation,
        p_validation,
        validation_sigmoid,
        validation_isotonic,
    )

    test_output = create_prediction_output(
        test,
        p_test,
        test_sigmoid,
        test_isotonic,
    )

    validation_output.to_csv(
        OUTPUT_DIR / "validation_calibrated_predictions.csv",
        index=False,
    )

    test_output.to_csv(
        OUTPUT_DIR / "test_calibrated_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Create calibration-bin tables
    # -------------------------------------------------------------------------

    print("\nSaving calibration-bin diagnostics...")

    validation_bin_rows = []

    for method, probabilities in [
        ("raw_model_4", p_validation),
        ("sigmoid", validation_sigmoid),
        ("isotonic", validation_isotonic),
    ]:

        bins = calculate_calibration_bins(
            y_validation,
            probabilities,
        )

        bins.insert(
            0,
            "method",
            method,
        )

        bins.insert(
            0,
            "dataset",
            "validation",
        )

        validation_bin_rows.append(bins)

    validation_bins = pd.concat(
        validation_bin_rows,
        ignore_index=True,
    )

    test_bin_rows = []

    for method, probabilities in [
        ("raw_model_4", p_test),
        ("sigmoid", test_sigmoid),
        ("isotonic", test_isotonic),
    ]:

        bins = calculate_calibration_bins(
            y_test,
            probabilities,
        )

        bins.insert(
            0,
            "method",
            method,
        )

        bins.insert(
            0,
            "dataset",
            "test",
        )

        test_bin_rows.append(bins)

    test_bins = pd.concat(
        test_bin_rows,
        ignore_index=True,
    )

    validation_bins.to_csv(
        OUTPUT_DIR / "validation_calibration_bins.csv",
        index=False,
    )

    test_bins.to_csv(
        OUTPUT_DIR / "test_calibration_bins.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save calibrators
    # -------------------------------------------------------------------------

    print("Saving calibrators...")

    joblib.dump(
        sigmoid_calibrator,
        OUTPUT_DIR / "sigmoid_calibrator.joblib",
    )

    joblib.dump(
        isotonic_calibrator,
        OUTPUT_DIR / "isotonic_calibrator.joblib",
    )

    # -------------------------------------------------------------------------
    # Save calibration summary
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "model": "random_forest_model_4",
                "calibration_methods": (
                    "raw_model_4,sigmoid,isotonic"
                ),
                "calibration_training_period": "2023-2024",
                "test_period": "2025",
                "validation_rows": len(validation),
                "test_rows": len(test),
                "validation_winner_by_log_loss": validation_winner,
                "test_best_log_loss_method": test_best_log_loss,
                "test_best_brier_method": test_best_brier,
                "test_best_ece_method": test_best_ece,
                "test_best_mce_method": test_best_mce,
                "test_best_auc_method": test_best_auc,
                "random_state": RANDOM_STATE,
                "calibration_bins": N_CALIBRATION_BINS,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "calibration_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save comparison
    # -------------------------------------------------------------------------

    comparison.to_csv(
        OUTPUT_DIR / "calibration_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("MODEL 4 CALIBRATION EXPERIMENT COMPLETE")
    print("=" * 80)

    print("\nOriginal Model 4 was not modified.")

    print("\nCalibration artifacts saved to:")
    print(f"  {OUTPUT_DIR.resolve()}")

    print("\nFiles:")
    print("  calibration_comparison.csv")
    print("  validation_calibrated_predictions.csv")
    print("  test_calibrated_predictions.csv")
    print("  validation_calibration_bins.csv")
    print("  test_calibration_bins.csv")
    print("  calibration_summary.csv")
    print("  sigmoid_calibrator.joblib")
    print("  isotonic_calibrator.joblib")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()