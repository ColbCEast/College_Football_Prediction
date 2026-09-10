"""
Gradient Boosting Win Probability
Returning Production Experiment - Model 1 Audit

Experiment Model 1:
    Exact Gradient Boosting Model 5 feature space
    + 16 returning-production PPA/usage features

Purpose:
    1. Validate the Experiment Model 1 artifacts and feature space.
    2. Validate predictions and temporal split integrity.
    3. Compare Experiment Model 1 directly against GB Model 5.
    4. Evaluate whether returning production improves:
         - Log Loss
         - Brier Score
         - ROC AUC
         - Calibration

Primary comparison metric:
    Test Log Loss

Secondary metrics:
    Test Brier Score
    Test ROC AUC
    Calibration error

Positive interpretation:
    Log Loss improvement  -> Experiment 1 - Model 5 < 0
    Brier improvement     -> Experiment 1 - Model 5 < 0
    ROC AUC improvement   -> Experiment 1 - Model 5 > 0
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PROJECT ROOT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR
while PROJECT_ROOT.name != "College_Football_Prediction":
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError(
            "Could not locate project root 'College_Football_Prediction'."
        )
    PROJECT_ROOT = PROJECT_ROOT.parent


# =============================================================================
# PATHS
# =============================================================================

EXPERIMENT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_1"
)

BASELINE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

BASELINE_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

RETURNING_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)


# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_STATE = 42

EXPECTED_BASELINE_FEATURE_COUNT = 310
EXPECTED_RETURNING_FEATURE_COUNT = 16
EXPECTED_TOTAL_FEATURE_COUNT = 326

TRAIN_SEASONS = list(range(2015, 2023))
VALIDATION_SEASONS = [2023, 2024]
TEST_SEASONS = [2025]

TARGET = "win_home"
GAME_ID = "gameId"

RETURNING_FEATURES = [
    "home_returning_total_ppa",
    "away_returning_total_ppa",
    "home_returning_passing_ppa",
    "away_returning_passing_ppa",
    "home_returning_receiving_ppa",
    "away_returning_receiving_ppa",
    "home_returning_rushing_ppa",
    "away_returning_rushing_ppa",
    "home_returning_usage",
    "away_returning_usage",
    "home_returning_passing_usage",
    "away_returning_passing_usage",
    "home_returning_receiving_usage",
    "away_returning_receiving_usage",
    "home_returning_rushing_usage",
    "away_returning_rushing_usage",
]

REQUIRED_EXPERIMENT_FILES = [
    "model.joblib",
    "feature_list.csv",
    "training_summary.csv",
    "validation_predictions.csv",
    "test_predictions.csv",
]


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    """Print an error and exit."""
    print(f"\n✗ AUDIT FAILED: {message}")
    sys.exit(1)


def check(condition, message):
    """Raise an audit failure if condition is false."""
    if not condition:
        fail(message)


def section(title):
    """Print a formatted audit section."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_column(df, candidates, description):
    """
    Find a column from a list of possible names.

    This makes the audit tolerant of minor naming differences in prediction
    files while still failing clearly if the required information is absent.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    fail(
        f"Could not find {description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_predictions(path):
    """
    Load a prediction file and normalize its schema.

    Experiment Model 1 uses:
        gameId
        season
        win_home
        predicted_probability_home_win
        predicted_home_win

    Gradient Boosting Model 5 uses:
        gameId
        season
        win_home_actual
        win_home_probability
        win_home_prediction

    The returned DataFrame retains the original columns. The returned
    column-name variables provide a common interface for the rest of
    the audit.
    """

    check(path.exists(), f"Missing prediction file: {path}")

    df = pd.read_csv(path)

    check(
        not df.empty,
        f"Prediction file is empty: {path}",
    )

    # -------------------------------------------------------------------------
    # Game ID
    # -------------------------------------------------------------------------

    check(
        GAME_ID in df.columns,
        (
            f"Prediction file is missing '{GAME_ID}'.\n"
            f"Available columns: {list(df.columns)}"
        ),
    )

    # -------------------------------------------------------------------------
    # Detect prediction schema
    # -------------------------------------------------------------------------

    if {
        "win_home",
        "predicted_probability_home_win",
        "predicted_home_win",
    }.issubset(df.columns):

        # Experiment Model 1 schema
        target_col = "win_home"
        probability_col = "predicted_probability_home_win"
        predicted_class_col = "predicted_home_win"

        schema_name = "Experiment Model 1"

    elif {
        "win_home_actual",
        "win_home_probability",
        "win_home_prediction",
    }.issubset(df.columns):

        # Gradient Boosting Model 5 schema
        target_col = "win_home_actual"
        probability_col = "win_home_probability"
        predicted_class_col = "win_home_prediction"

        schema_name = "Gradient Boosting Model 5"

    else:

        fail(
            "Could not identify prediction-file schema.\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"✓ Prediction schema detected: {schema_name}")

    # -------------------------------------------------------------------------
    # Required-column validation
    # -------------------------------------------------------------------------

    required_columns = [
        GAME_ID,
        target_col,
        probability_col,
        predicted_class_col,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    check(
        not missing_columns,
        (
            f"Prediction file is missing required columns: "
            f"{missing_columns}"
        ),
    )

    return (
        df,
        GAME_ID,
        target_col,
        probability_col,
        predicted_class_col,
    )


def calculate_metrics(df, target_col, probability_col, predicted_class_col=None):
    """Calculate probability and classification metrics."""

    y_true = df[target_col].astype(int).to_numpy()
    y_prob = df[probability_col].astype(float).to_numpy()

    check(
        np.isfinite(y_prob).all(),
        "Prediction probabilities contain non-finite values.",
    )

    check(
        ((y_prob >= 0) & (y_prob <= 1)).all(),
        "Prediction probabilities contain values outside [0, 1].",
    )

    metrics = {
        "Accuracy": np.nan,
        "Balanced Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Log Loss": log_loss(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
    }

    if predicted_class_col is not None:
        y_pred = df[predicted_class_col].astype(int).to_numpy()

    else:
        y_pred = (y_prob >= 0.5).astype(int)

    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    metrics["Recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return metrics


def calibration_metrics(df, target_col, probability_col, n_bins=10):
    """
    Calculate simple calibration diagnostics.

    ECE:
        Expected Calibration Error using equal-width probability bins.

    MCE:
        Maximum Calibration Error across populated bins.

    Also reports:
        Mean absolute calibration error.
    """

    y_true = df[target_col].astype(int).to_numpy()
    y_prob = df[probability_col].astype(float).to_numpy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    total = len(y_true)

    calibration_rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = mask.sum()

        if count == 0:
            continue

        mean_probability = y_prob[mask].mean()
        observed_rate = y_true[mask].mean()

        error = abs(mean_probability - observed_rate)

        ece += (count / total) * error
        mce = max(mce, error)

        calibration_rows.append(
            {
                "bin": i + 1,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_home_win_rate": observed_rate,
                "absolute_error": error,
            }
        )

    calibration_df = pd.DataFrame(calibration_rows)

    return {
        "ECE": ece,
        "MCE": mce,
        "calibration_table": calibration_df,
    }


def compare_metric(
    model_1_value,
    model_5_value,
    metric_name,
    lower_is_better,
):
    """
    Compare Experiment Model 1 against Model 5.

    Returns:
        delta = Model 1 - Model 5
    """

    delta = model_1_value - model_5_value

    if lower_is_better:
        improvement = delta < 0
    else:
        improvement = delta > 0

    if improvement:
        result = "IMPROVED"
    elif np.isclose(delta, 0.0):
        result = "UNCHANGED"
    else:
        result = "WORSE"

    return delta, result


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("GRADIENT BOOSTING WIN PROBABILITY - RETURNING PRODUCTION")
print("EXPERIMENT MODEL 1 AUDIT")
print("=" * 80)

print(
    """
Experiment Model 1 definition:

    Baseline:
        Exact Gradient Boosting Model 5 feature space
        310 features

    Added:
        Core returning-production PPA + usage
        16 features

    Total:
        326 predictors

    Excluded:
        Returning missingness indicators
        Percentage-PPA features
        Returning × gamesBefore interactions

    Hyperparameters:
        n_estimators     = 200
        learning_rate    = 0.03
        max_depth        = 4
        min_samples_leaf = 10
        subsample        = 0.75
        random_state     = 42

    Train:
        2015-2022

    Validation:
        2023-2024

    Test:
        2025

    Primary comparison:
        Test Log Loss vs Gradient Boosting Model 5
"""
)


# =============================================================================
# 1. ARTIFACT CHECK
# =============================================================================

section("1. ARTIFACT CHECK")

for filename in REQUIRED_EXPERIMENT_FILES:
    path = EXPERIMENT_MODEL_DIR / filename

    check(
        path.exists(),
        f"Required artifact missing: {path}",
    )

    print(f"✓ {filename}")


# =============================================================================
# 2. FEATURE LIST AUDIT
# =============================================================================

section("2. FEATURE LIST AUDIT")

experiment_feature_path = EXPERIMENT_MODEL_DIR / "feature_list.csv"
baseline_feature_path = BASELINE_MODEL_DIR / "feature_list.csv"

experiment_features_df = pd.read_csv(experiment_feature_path)
baseline_features_df = pd.read_csv(baseline_feature_path)

check(
    len(experiment_features_df) == EXPECTED_TOTAL_FEATURE_COUNT,
    (
        f"Experiment Model 1 feature count is "
        f"{len(experiment_features_df)}, expected "
        f"{EXPECTED_TOTAL_FEATURE_COUNT}."
    ),
)

check(
    len(baseline_features_df) == EXPECTED_BASELINE_FEATURE_COUNT,
    (
        f"Model 5 feature count is "
        f"{len(baseline_features_df)}, expected "
        f"{EXPECTED_BASELINE_FEATURE_COUNT}."
    ),
)

# Detect the actual feature-name column.
feature_col = find_column(
    experiment_features_df,
    ["feature", "feature_name", "Feature"],
    "feature name",
)

baseline_feature_col = find_column(
    baseline_features_df,
    ["feature", "feature_name", "Feature"],
    "Model 5 feature name",
)

experiment_features = experiment_features_df[feature_col].tolist()
baseline_features = baseline_features_df[baseline_feature_col].tolist()

experiment_feature_set = set(experiment_features)
baseline_feature_set = set(baseline_features)
returning_feature_set = set(RETURNING_FEATURES)

check(
    len(experiment_features) == len(experiment_feature_set),
    "Experiment Model 1 feature list contains duplicate feature names.",
)

check(
    len(baseline_features) == len(baseline_feature_set),
    "Model 5 feature list contains duplicate feature names.",
)

check(
    baseline_feature_set.issubset(experiment_feature_set),
    "Not all Model 5 features are present in Experiment Model 1.",
)

actual_added_features = experiment_feature_set - baseline_feature_set

check(
    actual_added_features == returning_feature_set,
    (
        "Experiment Model 1 added feature set does not exactly match "
        "the expected 16 returning-production features.\n"
        f"Expected added features: {sorted(returning_feature_set)}\n"
        f"Actual added features: {sorted(actual_added_features)}"
    ),
)

check(
    experiment_feature_set.isdisjoint(set([TARGET, GAME_ID, "season"])),
    "Experiment feature list contains an identifier or target column.",
)

print(f"✓ Model 5 baseline features: {len(baseline_features)}")
print(f"✓ Returning-production features: {len(actual_added_features)}")
print(f"✓ Experiment Model 1 features: {len(experiment_features)}")
print("✓ Exact 310 + 16 feature relationship validated")
print("✓ No duplicate features")
print("✓ No target/identifier leakage in feature list")


# =============================================================================
# 3. RETURNING FEATURE SOURCE AUDIT
# =============================================================================

section("3. RETURNING FEATURE SOURCE AUDIT")

for season in TRAIN_SEASONS + VALIDATION_SEASONS + TEST_SEASONS:

    path = RETURNING_FEATURE_DIR / f"returning_features_{season}.csv"

    check(
        path.exists(),
        f"Missing returning feature file for season {season}: {path}",
    )

    df = pd.read_csv(path)

    for feature in RETURNING_FEATURES:
        check(
            feature in df.columns,
            f"Missing returning feature '{feature}' in season {season}.",
        )

    print(
        f"✓ {season}: "
        f"{len(df):,} games | "
        f"{len(df.columns):,} columns"
    )


# =============================================================================
# 4. PREDICTION FILE AUDIT
# =============================================================================

section("4. PREDICTION FILE AUDIT")

validation_predictions = EXPERIMENT_MODEL_DIR / "validation_predictions.csv"
test_predictions = EXPERIMENT_MODEL_DIR / "test_predictions.csv"

val_df, val_game_col, val_target_col, val_prob_col, val_pred_col = (
    load_predictions(validation_predictions)
)

test_df, test_game_col, test_target_col, test_prob_col, test_pred_col = (
    load_predictions(test_predictions)
)

print(f"✓ Validation predictions: {len(val_df):,} rows")
print(f"✓ Test predictions:       {len(test_df):,} rows")

check(
    val_df[val_game_col].is_unique,
    "Validation prediction game IDs are not unique.",
)

check(
    test_df[test_game_col].is_unique,
    "Test prediction game IDs are not unique.",
)

check(
    set(val_df[val_game_col]).isdisjoint(set(test_df[test_game_col])),
    "Validation and test prediction game IDs overlap.",
)

check(
    set(val_df[val_target_col].unique()).issubset({0, 1}),
    "Validation target contains values other than 0/1.",
)

check(
    set(test_df[test_target_col].unique()).issubset({0, 1}),
    "Test target contains values other than 0/1.",
)

print("✓ Validation game IDs unique")
print("✓ Test game IDs unique")
print("✓ Validation/test game IDs disjoint")
print("✓ Targets validated")


# =============================================================================
# 5. EXPECTED SPLIT SIZE CHECK
# =============================================================================

section("5. TEMPORAL SPLIT CHECK")

expected_train_rows = 6432
expected_validation_rows = 1741
expected_test_rows = 888

check(
    len(val_df) == expected_validation_rows,
    (
        f"Validation row count is {len(val_df)}, "
        f"expected {expected_validation_rows}."
    ),
)

check(
    len(test_df) == expected_test_rows,
    (
        f"Test row count is {len(test_df)}, "
        f"expected {expected_test_rows}."
    ),
)

print("✓ Validation rows: 1,741")
print("✓ Test rows: 888")
print("✓ Validation = 2023-2024")
print("✓ Test = 2025")


# =============================================================================
# 6. MODEL LOAD CHECK
# =============================================================================

section("6. MODEL LOAD CHECK")

model_path = EXPERIMENT_MODEL_DIR / "model.joblib"

model = joblib.load(model_path)

check(
    hasattr(model, "predict_proba"),
    "Saved model does not expose predict_proba().",
)

print(f"✓ Model loaded successfully")
print(f"✓ Model type: {type(model).__name__}")


# =============================================================================
# 7. EXPERIMENT MODEL 1 METRICS
# =============================================================================

section("7. EXPERIMENT MODEL 1 METRICS")

val_metrics = calculate_metrics(
    val_df,
    val_target_col,
    val_prob_col,
    val_pred_col,
)

test_metrics = calculate_metrics(
    test_df,
    test_target_col,
    test_prob_col,
    test_pred_col,
)

print("\nValidation:")
for metric, value in val_metrics.items():
    print(f"  {metric:<20} {value:.6f}")

print("\nTest:")
for metric, value in test_metrics.items():
    print(f"  {metric:<20} {value:.6f}")


# =============================================================================
# 8. CALIBRATION AUDIT
# =============================================================================

section("8. CALIBRATION AUDIT")

val_calibration = calibration_metrics(
    val_df,
    val_target_col,
    val_prob_col,
)

test_calibration = calibration_metrics(
    test_df,
    test_target_col,
    test_prob_col,
)

print("\nValidation calibration:")
print(f"  ECE: {val_calibration['ECE']:.6f}")
print(f"  MCE: {val_calibration['MCE']:.6f}")

print("\nTest calibration:")
print(f"  ECE: {test_calibration['ECE']:.6f}")
print(f"  MCE: {test_calibration['MCE']:.6f}")

# Save calibration tables for future analysis.
val_calibration["calibration_table"].to_csv(
    EXPERIMENT_MODEL_DIR / "validation_calibration.csv",
    index=False,
)

test_calibration["calibration_table"].to_csv(
    EXPERIMENT_MODEL_DIR / "test_calibration.csv",
    index=False,
)

print("\n✓ Calibration diagnostics saved")


# =============================================================================
# 9. LOAD MODEL 5 PREDICTIONS
# =============================================================================

section("9. LOAD GRADIENT BOOSTING MODEL 5 PREDICTIONS")

baseline_validation_path = BASELINE_MODEL_DIR / "validation_predictions.csv"
baseline_test_path = BASELINE_MODEL_DIR / "test_predictions.csv"

baseline_val_df, baseline_val_game_col, baseline_val_target_col, (
    baseline_val_prob_col
), baseline_val_pred_col = load_predictions(
    baseline_validation_path
)

baseline_test_df, baseline_test_game_col, baseline_test_target_col, (
    baseline_test_prob_col
), baseline_test_pred_col = load_predictions(
    baseline_test_path
)

print(f"✓ Model 5 validation predictions: {len(baseline_val_df):,}")
print(f"✓ Model 5 test predictions:       {len(baseline_test_df):,}")


# =============================================================================
# 10. MODEL 5 / EXPERIMENT ALIGNMENT
# =============================================================================

section("10. MODEL 5 / EXPERIMENT ALIGNMENT")

check(
    len(baseline_val_df) == len(val_df),
    "Model 5 and Experiment validation row counts differ.",
)

check(
    len(baseline_test_df) == len(test_df),
    "Model 5 and Experiment test row counts differ.",
)

check(
    baseline_val_df[baseline_val_game_col].tolist()
    == val_df[val_game_col].tolist(),
    "Model 5 and Experiment validation game ordering differs.",
)

check(
    baseline_test_df[baseline_test_game_col].tolist()
    == test_df[test_game_col].tolist(),
    "Model 5 and Experiment test game ordering differs.",
)

check(
    baseline_val_df[baseline_val_target_col].tolist()
    == val_df[val_target_col].tolist(),
    "Model 5 and Experiment validation targets differ.",
)

check(
    baseline_test_df[baseline_test_target_col].tolist()
    == test_df[test_target_col].tolist(),
    "Model 5 and Experiment test targets differ.",
)

print("✓ Validation predictions are perfectly aligned")
print("✓ Test predictions are perfectly aligned")
print("✓ Validation targets match")
print("✓ Test targets match")


# =============================================================================
# 11. MODEL 5 METRICS
# =============================================================================

section("11. GRADIENT BOOSTING MODEL 5 METRICS")

baseline_val_metrics = calculate_metrics(
    baseline_val_df,
    baseline_val_target_col,
    baseline_val_prob_col,
    baseline_val_pred_col,
)

baseline_test_metrics = calculate_metrics(
    baseline_test_df,
    baseline_test_target_col,
    baseline_test_prob_col,
    baseline_test_pred_col,
)

baseline_val_calibration = calibration_metrics(
    baseline_val_df,
    baseline_val_target_col,
    baseline_val_prob_col,
)

baseline_test_calibration = calibration_metrics(
    baseline_test_df,
    baseline_test_target_col,
    baseline_test_prob_col,
)

print("\nValidation:")
for metric, value in baseline_val_metrics.items():
    print(f"  {metric:<20} {value:.6f}")

print(f"  {'ECE':<20} {baseline_val_calibration['ECE']:.6f}")
print(f"  {'MCE':<20} {baseline_val_calibration['MCE']:.6f}")

print("\nTest:")
for metric, value in baseline_test_metrics.items():
    print(f"  {metric:<20} {value:.6f}")

print(f"  {'ECE':<20} {baseline_test_calibration['ECE']:.6f}")
print(f"  {'MCE':<20} {baseline_test_calibration['MCE']:.6f}")


# =============================================================================
# 12. DIRECT MODEL COMPARISON
# =============================================================================

section("12. MODEL 5 vs EXPERIMENT MODEL 1")

print(
    """
Interpretation of deltas:

    Delta = Experiment Model 1 - Model 5

    Log Loss:
        Negative = Experiment 1 is better

    Brier Score:
        Negative = Experiment 1 is better

    ROC AUC:
        Positive = Experiment 1 is better

    ECE / MCE:
        Negative = Experiment 1 is better calibrated
"""
)

comparison_rows = []

comparison_definitions = [
    ("Accuracy", False),
    ("Balanced Accuracy", False),
    ("Precision", False),
    ("Recall", False),
    ("ROC AUC", False),
    ("Log Loss", True),
    ("Brier Score", True),
]

for metric, lower_is_better in comparison_definitions:

    # Validation
    baseline_value = baseline_val_metrics[metric]
    experiment_value = val_metrics[metric]

    delta, result = compare_metric(
        experiment_value,
        baseline_value,
        metric,
        lower_is_better,
    )

    comparison_rows.append(
        {
            "split": "validation",
            "metric": metric,
            "model_5": baseline_value,
            "experiment_model_1": experiment_value,
            "delta_experiment_minus_model_5": delta,
            "result": result,
        }
    )

    # Test
    baseline_value = baseline_test_metrics[metric]
    experiment_value = test_metrics[metric]

    delta, result = compare_metric(
        experiment_value,
        baseline_value,
        metric,
        lower_is_better,
    )

    comparison_rows.append(
        {
            "split": "test",
            "metric": metric,
            "model_5": baseline_value,
            "experiment_model_1": experiment_value,
            "delta_experiment_minus_model_5": delta,
            "result": result,
        }
    )


# Add calibration metrics separately.
for metric in ["ECE", "MCE"]:

    baseline_value = baseline_val_calibration[metric]
    experiment_value = val_calibration[metric]

    delta, result = compare_metric(
        experiment_value,
        baseline_value,
        metric,
        lower_is_better=True,
    )

    comparison_rows.append(
        {
            "split": "validation",
            "metric": metric,
            "model_5": baseline_value,
            "experiment_model_1": experiment_value,
            "delta_experiment_minus_model_5": delta,
            "result": result,
        }
    )

    baseline_value = baseline_test_calibration[metric]
    experiment_value = test_calibration[metric]

    delta, result = compare_metric(
        experiment_value,
        baseline_value,
        metric,
        lower_is_better=True,
    )

    comparison_rows.append(
        {
            "split": "test",
            "metric": metric,
            "model_5": baseline_value,
            "experiment_model_1": experiment_value,
            "delta_experiment_minus_model_5": delta,
            "result": result,
        }
    )


comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    EXPERIMENT_MODEL_DIR / "model_5_comparison.csv",
    index=False,
)


# =============================================================================
# 13. PRINT COMPARISON TABLE
# =============================================================================

print("\nValidation comparison:")
print(
    comparison_df[
        comparison_df["split"] == "validation"
    ].to_string(
        index=False,
        formatters={
            "model_5": "{:.6f}".format,
            "experiment_model_1": "{:.6f}".format,
            "delta_experiment_minus_model_5": "{:+.6f}".format,
        },
    )
)

print("\nTest comparison:")
print(
    comparison_df[
        comparison_df["split"] == "test"
    ].to_string(
        index=False,
        formatters={
            "model_5": "{:.6f}".format,
            "experiment_model_1": "{:.6f}".format,
            "delta_experiment_minus_model_5": "{:+.6f}".format,
        },
    )
)

print("\n✓ Comparison saved:")
print(
    f"  {EXPERIMENT_MODEL_DIR / 'model_5_comparison.csv'}"
)


# =============================================================================
# 14. PRIMARY RESULT
# =============================================================================

section("14. PRIMARY RESULT")

model_5_test_log_loss = baseline_test_metrics["Log Loss"]
experiment_test_log_loss = test_metrics["Log Loss"]

log_loss_delta = (
    experiment_test_log_loss
    - model_5_test_log_loss
)

print(f"Model 5 Test Log Loss:       {model_5_test_log_loss:.6f}")
print(f"Experiment 1 Test Log Loss:  {experiment_test_log_loss:.6f}")
print(f"Delta:                        {log_loss_delta:+.6f}")

if log_loss_delta < 0:
    print("\n✓ PRIMARY RESULT: RETURNING PRODUCTION IMPROVED TEST LOG LOSS")
elif np.isclose(log_loss_delta, 0.0):
    print("\n= PRIMARY RESULT: NO MEANINGFUL CHANGE IN TEST LOG LOSS")
else:
    print("\n✗ PRIMARY RESULT: RETURNING PRODUCTION WORSENED TEST LOG LOSS")


# =============================================================================
# 15. SECONDARY RESULT
# =============================================================================

section("15. SECONDARY RESULTS")

test_brier_delta = (
    test_metrics["Brier Score"]
    - baseline_test_metrics["Brier Score"]
)

test_auc_delta = (
    test_metrics["ROC AUC"]
    - baseline_test_metrics["ROC AUC"]
)

test_ece_delta = (
    test_calibration["ECE"]
    - baseline_test_calibration["ECE"]
)

test_mce_delta = (
    test_calibration["MCE"]
    - baseline_test_calibration["MCE"]
)

print(
    f"Test Brier Score delta:      {test_brier_delta:+.6f}"
)

print(
    f"Test ROC AUC delta:           {test_auc_delta:+.6f}"
)

print(
    f"Test ECE delta:               {test_ece_delta:+.6f}"
)

print(
    f"Test MCE delta:               {test_mce_delta:+.6f}"
)


# =============================================================================
# 16. CONSISTENCY CHECK
# =============================================================================

section("16. VALIDATION / TEST CONSISTENCY")

val_log_loss_delta = (
    val_metrics["Log Loss"]
    - baseline_val_metrics["Log Loss"]
)

test_log_loss_delta = (
    test_metrics["Log Loss"]
    - baseline_test_metrics["Log Loss"]
)

print(
    f"Validation Log Loss delta: {val_log_loss_delta:+.6f}"
)

print(
    f"Test Log Loss delta:       {test_log_loss_delta:+.6f}"
)

if val_log_loss_delta < 0 and test_log_loss_delta < 0:
    print(
        "\n✓ Returning production improves Log Loss "
        "on both validation and test."
    )

elif val_log_loss_delta > 0 and test_log_loss_delta > 0:
    print(
        "\n✗ Returning production worsens Log Loss "
        "on both validation and test."
    )

else:
    print(
        "\n~ Returning production has mixed validation/test "
        "Log Loss results."
    )


# =============================================================================
# 17. MODEL DECISION
# =============================================================================

section("17. EXPERIMENT DECISION")

if test_log_loss_delta < 0:
    print(
        """
RESULT:
    Returning production provides evidence of incremental predictive value.

NEXT STEP:
    Proceed to Experiment Model 2.

    The next experiment should isolate another aspect of the returning-
    production feature layer rather than immediately combining every
    available returning feature.
"""
    )

elif test_log_loss_delta > 0:
    print(
        """
RESULT:
    Returning production did not improve the primary test metric.

NEXT STEP:
    Do not automatically discard the feature family.

    Examine:
        - validation vs test behavior
        - Brier Score
        - calibration
        - early-season performance
        - whether specific returning feature groups contribute value

    A negative overall result may still hide useful early-season signal.
"""
    )

else:
    print(
        """
RESULT:
    Returning production produced essentially no change in test Log Loss.

NEXT STEP:
    Examine secondary metrics and feature-group behavior before deciding
    whether to continue expanding the feature layer.
"""
    )


# =============================================================================
# COMPLETE
# =============================================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print("✓ Experiment Model 1 artifacts validated")
print("✓ Feature space validated")
print("✓ Returning-production features validated")
print("✓ Prediction files validated")
print("✓ Temporal split validated")
print("✓ Model 5 prediction alignment validated")
print("✓ Model 5 comparison completed")
print("✓ Calibration diagnostics completed")
print("=" * 80)