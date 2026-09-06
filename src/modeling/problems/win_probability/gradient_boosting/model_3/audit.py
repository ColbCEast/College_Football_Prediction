"""
Gradient Boosting Win Probability - Model 3 Diagnostic & Stability Audit

Model 3:
- Compact 28-feature model
- Removes Trend and Prior SOS features from Model 2
- Uses only the base win-probability model input datasets
- Fixed temporal split:
    Train:      2015-2022
    Validation: 2023-2024
    Test:       2025

Purpose:
- Validate saved model artifacts
- Validate dataset and feature integrity
- Validate predictions
- Evaluate performance
- Analyze feature importance
- Calculate validation permutation importance
- Analyze feature-family importance
- Evaluate calibration
- Evaluate 2025 test-season performance
- Analyze prediction stability
- Produce audit CSV outputs

This script is diagnostic only.
It does NOT retrain the model or tune hyperparameters.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_3"
)

MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.csv"
TRAINING_SUMMARY_PATH = MODEL_DIR / "training_summary.csv"

VALIDATION_PREDICTIONS_PATH = (
    MODEL_DIR / "validation_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    MODEL_DIR / "test_predictions.csv"
)

AUDIT_DIR = MODEL_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPECTED MODEL 3 FEATURES
# =============================================================================

CORE_STRENGTH_FEATURES = [
    "homePregameElo",
    "awayPregameElo",
    "winPctBefore_home",
    "winPctBefore_away",
    "pointDifferentialBefore_home",
    "pointDifferentialBefore_away",
    "pointDifferentialAvgBefore_home",
    "pointDifferentialAvgBefore_away",
]

RECENT_FORM_FEATURES = [
    "pointDifferentialAvgLast3_home",
    "pointDifferentialAvgLast3_away",
    "pointDifferentialAvgLast5_home",
    "pointDifferentialAvgLast5_away",
    "pointsForAvgLast5_home",
    "pointsForAvgLast5_away",
    "pointsAgainstAvgLast5_home",
    "pointsAgainstAvgLast5_away",
]

OFFENSIVE_FEATURES = [
    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",
    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",
    "yardsPerPassAttemptBefore_home",
    "yardsPerPassAttemptBefore_away",
    "yardsPerRushAttemptBefore_home",
    "yardsPerRushAttemptBefore_away",
]

DEFENSIVE_FEATURES = [
    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",
    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",
]

EXPECTED_FEATURES = (
    CORE_STRENGTH_FEATURES
    + RECENT_FORM_FEATURES
    + OFFENSIVE_FEATURES
    + DEFENSIVE_FEATURES
)

EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURES)

TARGET = "win_home"

IDENTIFIER_COLUMNS = [
    "season",
    "gameId",
    "startDate",
    "seasonType",
]


# =============================================================================
# FEATURES THAT MUST NOT BE PRESENT
# =============================================================================

REMOVED_TREND_FEATURES = [
    "pointsForTrend_home",
    "pointsForTrend_away",
    "pointsAgainstTrend_home",
    "pointsAgainstTrend_away",
    "pointDifferentialTrend_home",
    "pointDifferentialTrend_away",
    "totalYardsTrend_home",
    "totalYardsTrend_away",
    "netPassingYardsTrend_home",
    "netPassingYardsTrend_away",
    "winPctTrend_home",
    "winPctTrend_away",
]

REMOVED_SOS_FEATURES = [
    "priorSOSWinPct_home",
    "priorSOSWinPct_away",
    "priorSOSPointDiff_home",
    "priorSOSPointDiff_away",
]

REMOVED_FEATURES = (
    REMOVED_TREND_FEATURES
    + REMOVED_SOS_FEATURES
)


# =============================================================================
# FEATURE FAMILIES
# =============================================================================

FEATURE_FAMILIES = {
    "Core Strength": CORE_STRENGTH_FEATURES,
    "Recent Form": RECENT_FORM_FEATURES,
    "Offensive Efficiency": OFFENSIVE_FEATURES,
    "Defensive Efficiency": DEFENSIVE_FEATURES,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    """Print a formatted audit section header."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_file_exists(path, label):
    """Validate that a required file exists."""
    exists = path.exists()

    status = "PASS" if exists else "FAIL"

    print(f"{label}: {status}")
    print(f"  {path}")

    return exists


def calculate_performance(y_true, y_prob):
    """Calculate probability and classification metrics."""

    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def calibration_table(y_true, y_prob, n_bins=10):
    """
    Produce a reliability/calibration table using fixed probability bins.
    """

    calibration_df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted_probability": y_prob,
        }
    )

    calibration_df["bin"] = pd.cut(
        calibration_df["predicted_probability"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )

    grouped = (
        calibration_df
        .groupby("bin", observed=False)
        .agg(
            observations=("actual", "size"),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_home_win_rate=("actual", "mean"),
        )
        .reset_index()
    )

    grouped["calibration_error"] = (
        grouped["observed_home_win_rate"]
        - grouped["mean_predicted_probability"]
    )

    grouped["absolute_calibration_error"] = (
        grouped["calibration_error"].abs()
    )

    return grouped


def prediction_stability_table(y_prob, dataset_name):
    """Summarize the distribution of predicted probabilities."""

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    result = {
        "dataset": dataset_name,
        "mean": np.mean(y_prob),
        "std": np.std(y_prob),
        "min": np.min(y_prob),
        "max": np.max(y_prob),
    }

    for percentile in percentiles:
        result[f"p{percentile:02d}"] = np.percentile(
            y_prob,
            percentile,
        )

    return result


# =============================================================================
# START AUDIT
# =============================================================================

print("=" * 78)
print("GRADIENT BOOSTING WIN PROBABILITY - MODEL 3")
print("DIAGNOSTIC & STABILITY AUDIT")
print("=" * 78)

print("\nModel 3 definition:")
print("  28 compact base features")
print("  Trend features: REMOVED")
print("  Prior SOS features: REMOVED")
print("  Enhanced feature merge: NOT USED")
print("  Test set: evaluation only")


# =============================================================================
# 1. FILE VALIDATION
# =============================================================================

print_section("1. REQUIRED FILE VALIDATION")

required_files = {
    "Training data": TRAIN_PATH,
    "Validation data": VALIDATION_PATH,
    "Test data": TEST_PATH,
    "Model": MODEL_PATH,
    "Feature list": FEATURE_LIST_PATH,
    "Training summary": TRAINING_SUMMARY_PATH,
    "Validation predictions": VALIDATION_PREDICTIONS_PATH,
    "Test predictions": TEST_PREDICTIONS_PATH,
}

file_status = {}

for label, path in required_files.items():
    file_status[label] = check_file_exists(path, label)

if not all(file_status.values()):
    raise FileNotFoundError(
        "One or more required Model 3 files are missing."
    )


# =============================================================================
# 2. LOAD DATA
# =============================================================================

print_section("2. LOADING DATA")

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

model = joblib.load(MODEL_PATH)

feature_list_df = pd.read_csv(FEATURE_LIST_PATH)
training_summary_df = pd.read_csv(TRAINING_SUMMARY_PATH)

validation_predictions = pd.read_csv(
    VALIDATION_PREDICTIONS_PATH
)

test_predictions = pd.read_csv(
    TEST_PREDICTIONS_PATH
)

print(f"Training shape:    {train_df.shape}")
print(f"Validation shape:  {validation_df.shape}")
print(f"Test shape:        {test_df.shape}")

print(f"\nModel type: {type(model).__name__}")

print(
    f"Feature list rows: {len(feature_list_df)}"
)

print(
    f"Validation predictions: "
    f"{validation_predictions.shape}"
)

print(
    f"Test predictions: "
    f"{test_predictions.shape}"
)


# =============================================================================
# 3. DATASET VALIDATION
# =============================================================================

print_section("3. DATASET VALIDATION")

expected_shapes = {
    "Training": (6432, 315),
    "Validation": (1741, 315),
    "Test": (888, 315),
}

datasets = {
    "Training": train_df,
    "Validation": validation_df,
    "Test": test_df,
}

dataset_checks = []

for name, df in datasets.items():

    expected_shape = expected_shapes[name]

    shape_pass = df.shape == expected_shape

    target_present = TARGET in df.columns

    target_nulls = (
        df[TARGET].isna().sum()
        if target_present
        else np.nan
    )

    seasons = (
        sorted(df["season"].dropna().unique())
        if "season" in df.columns
        else []
    )

    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Expected: {expected_shape}")
    print(
        f"  Shape check: "
        f"{'PASS' if shape_pass else 'FAIL'}"
    )

    print(
        f"  Target present: "
        f"{'PASS' if target_present else 'FAIL'}"
    )

    print(f"  Target nulls: {target_nulls}")
    print(f"  Seasons: {seasons}")

    dataset_checks.append(
        {
            "dataset": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "expected_rows": expected_shape[0],
            "expected_columns": expected_shape[1],
            "shape_pass": shape_pass,
            "target_present": target_present,
            "target_nulls": target_nulls,
            "seasons": ",".join(map(str, seasons)),
        }
    )

dataset_checks_df = pd.DataFrame(dataset_checks)


# =============================================================================
# 4. FEATURE LIST VALIDATION
# =============================================================================

print_section("4. FEATURE LIST VALIDATION")

if "feature" not in feature_list_df.columns:
    raise ValueError(
        "feature_list.csv must contain a 'feature' column."
    )

saved_features = feature_list_df["feature"].dropna().tolist()

saved_feature_set = set(saved_features)
expected_feature_set = set(EXPECTED_FEATURES)

missing_features = sorted(
    expected_feature_set - saved_feature_set
)

unexpected_features = sorted(
    saved_feature_set - expected_feature_set
)

duplicate_features = (
    feature_list_df["feature"]
    .dropna()
    .duplicated()
    .sum()
)

print(f"Expected feature count: {EXPECTED_FEATURE_COUNT}")
print(f"Saved feature count:    {len(saved_features)}")

print(
    f"Feature count check: "
    f"{'PASS' if len(saved_features) == EXPECTED_FEATURE_COUNT else 'FAIL'}"
)

print(
    f"Feature set check: "
    f"{'PASS' if not missing_features and not unexpected_features else 'FAIL'}"
)

print(
    f"Duplicate features: {duplicate_features}"
)

if missing_features:
    print("\nMissing expected features:")
    for feature in missing_features:
        print(f"  - {feature}")

if unexpected_features:
    print("\nUnexpected features:")
    for feature in unexpected_features:
        print(f"  - {feature}")


# =============================================================================
# 5. REMOVED FEATURE VALIDATION
# =============================================================================

print_section("5. REMOVED FEATURE VALIDATION")

accidental_removed_features = sorted(
    saved_feature_set.intersection(REMOVED_FEATURES)
)

print(
    f"Trend features expected removed: "
    f"{len(REMOVED_TREND_FEATURES)}"
)

print(
    f"Prior SOS features expected removed: "
    f"{len(REMOVED_SOS_FEATURES)}"
)

print(
    f"Accidentally retained removed features: "
    f"{len(accidental_removed_features)}"
)

if accidental_removed_features:
    print("\nFAIL — Removed features found in Model 3:")
    for feature in accidental_removed_features:
        print(f"  - {feature}")
else:
    print("PASS — No Trend or Prior SOS features are present.")


# =============================================================================
# 6. FEATURE PRESENCE IN DATASETS
# =============================================================================

print_section("6. FEATURE PRESENCE IN DATASETS")

feature_presence_rows = []

for dataset_name, df in datasets.items():

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in df.columns
    ]

    present_count = (
        EXPECTED_FEATURE_COUNT - len(missing)
    )

    print(f"\n{dataset_name}:")
    print(
        f"  Expected features: "
        f"{EXPECTED_FEATURE_COUNT}"
    )
    print(
        f"  Present features: "
        f"{present_count}"
    )
    print(
        f"  Missing features: "
        f"{len(missing)}"
    )

    if missing:
        for feature in missing:
            print(f"    - {feature}")

    feature_presence_rows.append(
        {
            "dataset": dataset_name,
            "expected_features": EXPECTED_FEATURE_COUNT,
            "present_features": present_count,
            "missing_features": len(missing),
            "missing_feature_names": ",".join(missing),
        }
    )

feature_presence_df = pd.DataFrame(
    feature_presence_rows
)


# =============================================================================
# 7. MODEL / PIPELINE VALIDATION
# =============================================================================

print_section("7. MODEL / PIPELINE VALIDATION")

print("Model object:")
print(model)

print("\nPipeline steps:")

if hasattr(model, "named_steps"):

    for step_name, step in model.named_steps.items():
        print(
            f"  {step_name}: "
            f"{type(step).__name__}"
        )

    pipeline_names = list(model.named_steps.keys())

    has_imputer = any(
        "imput" in name.lower()
        for name in pipeline_names
    )

    has_gb = any(
        "gradient" in type(step).__name__.lower()
        for step in model.named_steps.values()
    )

else:
    has_imputer = False
    has_gb = False

    print("  WARNING: Model is not a sklearn Pipeline.")

print(
    f"\nMedian imputation pipeline step present: "
    f"{'PASS' if has_imputer else 'FAIL'}"
)

print(
    f"Gradient Boosting classifier present: "
    f"{'PASS' if has_gb else 'FAIL'}"
)


# =============================================================================
# 8. HYPERPARAMETER VALIDATION
# =============================================================================

print_section("8. HYPERPARAMETER VALIDATION")

expected_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "subsample": 1.0,
    "random_state": 42,
}

gb_estimator = None

if hasattr(model, "named_steps"):

    for step in model.named_steps.values():

        if (
            hasattr(step, "n_estimators")
            and hasattr(step, "learning_rate")
        ):
            gb_estimator = step
            break

if gb_estimator is None:

    print(
        "FAIL — Could not locate GradientBoostingClassifier."
    )

else:

    parameter_rows = []

    for parameter, expected_value in expected_params.items():

        actual_value = getattr(
            gb_estimator,
            parameter,
            None,
        )

        passed = actual_value == expected_value

        print(
            f"{parameter}: "
            f"{actual_value} "
            f"(expected {expected_value}) "
            f"{'PASS' if passed else 'FAIL'}"
        )

        parameter_rows.append(
            {
                "parameter": parameter,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "match": passed,
            }
        )

    parameter_df = pd.DataFrame(parameter_rows)


# =============================================================================
# 9. FEATURE MISSINGNESS
# =============================================================================

print_section("9. FEATURE MISSINGNESS")

missingness_rows = []

for feature in EXPECTED_FEATURES:

    row = {
        "feature": feature,
    }

    for dataset_name, df in datasets.items():

        missing_count = df[feature].isna().sum()
        missing_pct = (
            missing_count / len(df) * 100
        )

        row[f"{dataset_name.lower()}_missing_count"] = (
            missing_count
        )

        row[f"{dataset_name.lower()}_missing_pct"] = (
            missing_pct
        )

    missingness_rows.append(row)

missingness_df = pd.DataFrame(
    missingness_rows
)

print(
    missingness_df[
        [
            "feature",
            "training_missing_pct",
            "validation_missing_pct",
            "test_missing_pct",
        ]
    ].to_string(index=False)
)

print(
    f"\nFeatures with training missingness: "
    f"{(missingness_df['training_missing_count'] > 0).sum()}"
)

print(
    f"Total training missing values: "
    f"{missingness_df['training_missing_count'].sum()}"
)

print(
    f"Total validation missing values: "
    f"{missingness_df['validation_missing_count'].sum()}"
)

print(
    f"Total test missing values: "
    f"{missingness_df['test_missing_count'].sum()}"
)


# =============================================================================
# 10. PREDICTION VALIDATION
# =============================================================================

print_section("10. PREDICTION VALIDATION")

prediction_checks = []

for dataset_name, df, prediction_df in [
    (
        "Validation",
        validation_df,
        validation_predictions,
    ),
    (
        "Test",
        test_df,
        test_predictions,
    ),
]:

    print(f"\n{dataset_name} predictions:")

    expected_rows = len(df)
    actual_rows = len(prediction_df)

    print(
        f"  Expected rows: {expected_rows}"
    )

    print(
        f"  Prediction rows: {actual_rows}"
    )

    row_count_pass = (
        expected_rows == actual_rows
    )

    probability_columns = [
        column
        for column in prediction_df.columns
        if "prob" in column.lower()
    ]

    print(
        f"  Probability columns: "
        f"{probability_columns}"
    )

    if not probability_columns:
        raise ValueError(
            f"No probability column found in "
            f"{dataset_name} predictions."
        )

    probability_column = probability_columns[0]

    y_prob = prediction_df[
        probability_column
    ].to_numpy()

    null_count = np.isnan(y_prob).sum()

    invalid_low = (y_prob < 0).sum()
    invalid_high = (y_prob > 1).sum()

    print(f"  Probability column: {probability_column}")
    print(f"  Null probabilities: {null_count}")
    print(f"  < 0: {invalid_low}")
    print(f"  > 1: {invalid_high}")
    print(
        f"  Probability validity: "
        f"{'PASS' if null_count == 0 and invalid_low == 0 and invalid_high == 0 else 'FAIL'}"
    )

    prediction_checks.append(
        {
            "dataset": dataset_name,
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "row_count_pass": row_count_pass,
            "probability_column": probability_column,
            "null_probabilities": null_count,
            "probabilities_below_zero": invalid_low,
            "probabilities_above_one": invalid_high,
        }
    )

validation_probability_column = [
    c
    for c in validation_predictions.columns
    if "prob" in c.lower()
][0]

test_probability_column = [
    c
    for c in test_predictions.columns
    if "prob" in c.lower()
][0]

validation_prob = validation_predictions[
    validation_probability_column
].to_numpy()

test_prob = test_predictions[
    test_probability_column
].to_numpy()

y_validation = validation_df[TARGET].to_numpy()
y_test = test_df[TARGET].to_numpy()


# =============================================================================
# 11. PREDICTION / TARGET ALIGNMENT
# =============================================================================

print_section("11. PREDICTION / TARGET ALIGNMENT")

alignment_rows = []

for dataset_name, df, prediction_df in [
    (
        "Validation",
        validation_df,
        validation_predictions,
    ),
    (
        "Test",
        test_df,
        test_predictions,
    ),
]:

    checks = {}

    for key in ["gameId", "season"]:

        if key in df.columns and key in prediction_df.columns:

            data_values = df[key].to_numpy()
            prediction_values = prediction_df[key].to_numpy()

            matches = np.array_equal(
                data_values,
                prediction_values,
            )

            checks[f"{key}_aligned"] = matches

            print(
                f"{dataset_name} {key} alignment: "
                f"{'PASS' if matches else 'FAIL'}"
            )

        else:

            checks[f"{key}_aligned"] = np.nan

    alignment_rows.append(
        {
            "dataset": dataset_name,
            **checks,
        }
    )

alignment_df = pd.DataFrame(
    alignment_rows
)


# =============================================================================
# 12. PERFORMANCE
# =============================================================================

print_section("12. MODEL PERFORMANCE")

validation_metrics = calculate_performance(
    y_validation,
    validation_prob,
)

test_metrics = calculate_performance(
    y_test,
    test_prob,
)

performance_rows = []

for dataset_name, metrics in [
    ("Validation", validation_metrics),
    ("Test", test_metrics),
]:

    print(f"\n{dataset_name}:")

    print(
        f"  Log Loss:           "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score:        "
        f"{metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC:             "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy:            "
        f"{metrics['accuracy']:.6%}"
    )

    print(
        f"  Balanced Accuracy:  "
        f"{metrics['balanced_accuracy']:.6%}"
    )

    print(
        f"  Precision:           "
        f"{metrics['precision']:.6%}"
    )

    print(
        f"  Recall:              "
        f"{metrics['recall']:.6%}"
    )

    print("\n  Confusion Matrix:")
    print(
        f"    TN: {metrics['true_negatives']}"
        f"    FP: {metrics['false_positives']}"
    )
    print(
        f"    FN: {metrics['false_negatives']}"
        f"    TP: {metrics['true_positives']}"
    )

    performance_rows.append(
        {
            "dataset": dataset_name,
            **metrics,
        }
    )

performance_df = pd.DataFrame(
    performance_rows
)


# =============================================================================
# 13. FEATURE IMPORTANCE
# =============================================================================

print_section("13. IMPURITY FEATURE IMPORTANCE")

if hasattr(model, "named_steps"):

    gb_estimator = None

    for step in model.named_steps.values():

        if hasattr(step, "feature_importances_"):
            gb_estimator = step
            break

if gb_estimator is None:
    raise ValueError(
        "Could not find feature_importances_ "
        "in the saved model."
    )

feature_importances = gb_estimator.feature_importances_

if len(feature_importances) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        "Feature importance count does not match "
        "expected Model 3 feature count."
    )

importance_df = pd.DataFrame(
    {
        "feature": EXPECTED_FEATURES,
        "importance": feature_importances,
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

importance_df["rank"] = (
    np.arange(len(importance_df)) + 1
)

importance_df["importance_pct"] = (
    importance_df["importance"] * 100
)

print(
    importance_df[
        [
            "rank",
            "feature",
            "importance",
            "importance_pct",
        ]
    ].head(20).to_string(index=False)
)


# =============================================================================
# 14. VALIDATION PERMUTATION IMPORTANCE
# =============================================================================

print_section("14. VALIDATION PERMUTATION IMPORTANCE")

print(
    "Calculating permutation importance using "
    "validation Log Loss."
)

X_validation = validation_df[
    EXPECTED_FEATURES
].copy()

y_validation_series = validation_df[TARGET]

permutation = permutation_importance(
    model,
    X_validation,
    y_validation_series,
    scoring="neg_log_loss",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

permutation_df = pd.DataFrame(
    {
        "feature": EXPECTED_FEATURES,
        "mean_log_loss_increase": permutation.importances_mean,
        "std_log_loss_increase": permutation.importances_std,
    }
)

permutation_df = permutation_df.sort_values(
    "mean_log_loss_increase",
    ascending=False,
).reset_index(drop=True)

permutation_df["rank"] = (
    np.arange(len(permutation_df)) + 1
)

print(
    permutation_df[
        [
            "rank",
            "feature",
            "mean_log_loss_increase",
            "std_log_loss_increase",
        ]
    ].head(20).to_string(index=False)
)


# =============================================================================
# 15. FEATURE FAMILY IMPORTANCE
# =============================================================================

print_section("15. FEATURE FAMILY IMPORTANCE")

family_rows = []

for family, features in FEATURE_FAMILIES.items():

    family_importance = importance_df[
        importance_df["feature"].isin(features)
    ]["importance"].sum()

    family_rows.append(
        {
            "feature_family": family,
            "feature_count": len(features),
            "importance": family_importance,
            "importance_pct": family_importance * 100,
        }
    )

family_importance_df = pd.DataFrame(
    family_rows
).sort_values(
    "importance",
    ascending=False,
).reset_index(drop=True)

print(
    family_importance_df.to_string(index=False)
)


# =============================================================================
# 16. CALIBRATION
# =============================================================================

print_section("16. CALIBRATION")

validation_calibration = calibration_table(
    y_validation,
    validation_prob,
    n_bins=10,
)

test_calibration = calibration_table(
    y_test,
    test_prob,
    n_bins=10,
)

validation_calibration.insert(
    0,
    "dataset",
    "Validation",
)

test_calibration.insert(
    0,
    "dataset",
    "Test",
)

calibration_df = pd.concat(
    [
        validation_calibration,
        test_calibration,
    ],
    ignore_index=True,
)

print("\nValidation calibration:")
print(
    validation_calibration.to_string(
        index=False
    )
)

print("\nTest calibration:")
print(
    test_calibration.to_string(
        index=False
    )
)

print(
    "\nValidation mean absolute calibration error: "
    f"{validation_calibration['absolute_calibration_error'].mean():.6f}"
)

print(
    "Test mean absolute calibration error: "
    f"{test_calibration['absolute_calibration_error'].mean():.6f}"
)


# =============================================================================
# 17. TEST SEASON PERFORMANCE
# =============================================================================

print_section("17. TEST SEASON PERFORMANCE")

if "season" in test_df.columns:

    test_season_rows = []

    for season in sorted(
        test_df["season"].dropna().unique()
    ):

        mask = (
            test_df["season"] == season
        ).to_numpy()

        season_y = y_test[mask]
        season_prob = test_prob[mask]

        if len(np.unique(season_y)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                season_y,
                season_prob,
            )

        test_season_rows.append(
            {
                "season": season,
                "games": len(season_y),
                "home_win_rate": season_y.mean(),
                "mean_predicted_probability": season_prob.mean(),
                "log_loss": log_loss(
                    season_y,
                    season_prob,
                ),
                "brier_score": brier_score_loss(
                    season_y,
                    season_prob,
                ),
                "roc_auc": auc,
                "accuracy": accuracy_score(
                    season_y,
                    (season_prob >= 0.5).astype(int),
                ),
            }
        )

    season_performance_df = pd.DataFrame(
        test_season_rows
    )

    print(
        season_performance_df.to_string(
            index=False
        )
    )

else:

    season_performance_df = pd.DataFrame()

    print(
        "WARNING: Season column unavailable."
    )


# =============================================================================
# 18. PREDICTION STABILITY
# =============================================================================

print_section("18. PREDICTION STABILITY")

stability_rows = [
    prediction_stability_table(
        validation_prob,
        "Validation",
    ),
    prediction_stability_table(
        test_prob,
        "Test",
    ),
]

prediction_stability_df = pd.DataFrame(
    stability_rows
)

print(
    prediction_stability_df.to_string(
        index=False
    )
)


# =============================================================================
# 19. PREDICTION CONFIDENCE DISTRIBUTION
# =============================================================================

print_section("19. PREDICTION CONFIDENCE DISTRIBUTION")

confidence_rows = []

for dataset_name, y_prob in [
    ("Validation", validation_prob),
    ("Test", test_prob),
]:

    confidence_rows.append(
        {
            "dataset": dataset_name,
            "probability_lt_0.10": (
                y_prob < 0.10
            ).sum(),
            "probability_0.10_to_0.25": (
                (y_prob >= 0.10)
                & (y_prob < 0.25)
            ).sum(),
            "probability_0.25_to_0.50": (
                (y_prob >= 0.25)
                & (y_prob < 0.50)
            ).sum(),
            "probability_0.50_to_0.75": (
                (y_prob >= 0.50)
                & (y_prob < 0.75)
            ).sum(),
            "probability_0.75_to_0.90": (
                (y_prob >= 0.75)
                & (y_prob < 0.90)
            ).sum(),
            "probability_ge_0.90": (
                y_prob >= 0.90
            ).sum(),
        }
    )

confidence_df = pd.DataFrame(
    confidence_rows
)

print(
    confidence_df.to_string(
        index=False
    )
)


# =============================================================================
# 20. AUDIT SUMMARY
# =============================================================================

print_section("20. AUDIT SUMMARY")

validation_log_loss = validation_metrics[
    "log_loss"
]

test_log_loss = test_metrics[
    "log_loss"
]

validation_brier = validation_metrics[
    "brier_score"
]

test_brier = test_metrics[
    "brier_score"
]

validation_auc = validation_metrics[
    "roc_auc"
]

test_auc = test_metrics[
    "roc_auc"
]

validation_accuracy = validation_metrics[
    "accuracy"
]

test_accuracy = test_metrics[
    "accuracy"
]

audit_summary = pd.DataFrame(
    [
        {
            "model": "Gradient Boosting Model 3",
            "feature_count": EXPECTED_FEATURE_COUNT,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "validation_log_loss": validation_log_loss,
            "test_log_loss": test_log_loss,
            "validation_brier_score": validation_brier,
            "test_brier_score": test_brier,
            "validation_roc_auc": validation_auc,
            "test_roc_auc": test_auc,
            "validation_accuracy": validation_accuracy,
            "test_accuracy": test_accuracy,
            "validation_balanced_accuracy": validation_metrics[
                "balanced_accuracy"
            ],
            "test_balanced_accuracy": test_metrics[
                "balanced_accuracy"
            ],
            "validation_precision": validation_metrics[
                "precision"
            ],
            "test_precision": test_metrics[
                "precision"
            ],
            "validation_recall": validation_metrics[
                "recall"
            ],
            "test_recall": test_metrics[
                "recall"
            ],
            "training_features_with_missing_values": (
                missingness_df[
                    "training_missing_count"
                ] > 0
            ).sum(),
            "training_missing_values": (
                missingness_df[
                    "training_missing_count"
                ].sum()
            ),
            "validation_missing_values": (
                missingness_df[
                    "validation_missing_count"
                ].sum()
            ),
            "test_missing_values": (
                missingness_df[
                    "test_missing_count"
                ].sum()
            ),
            "removed_trend_features": len(
                REMOVED_TREND_FEATURES
            ),
            "removed_sos_features": len(
                REMOVED_SOS_FEATURES
            ),
            "accidental_removed_features_retained": len(
                accidental_removed_features
            ),
        }
    ]
)


# =============================================================================
# 21. SAVE AUDIT OUTPUTS
# =============================================================================

print_section("21. SAVING AUDIT OUTPUTS")

outputs = {
    "audit_summary.csv": audit_summary,
    "feature_importance.csv": importance_df,
    "permutation_importance.csv": permutation_df,
    "feature_importance_by_family.csv": family_importance_df,
    "calibration.csv": calibration_df,
    "season_performance.csv": season_performance_df,
    "prediction_stability.csv": prediction_stability_df,
    "prediction_confidence.csv": confidence_df,
    "missingness.csv": missingness_df,
    "dataset_validation.csv": dataset_checks_df,
    "feature_presence.csv": feature_presence_df,
    "prediction_validation.csv": pd.DataFrame(
        prediction_checks
    ),
    "prediction_alignment.csv": alignment_df,
}

if "parameter_df" in locals():
    outputs["hyperparameters.csv"] = parameter_df

for filename, dataframe in outputs.items():

    output_path = AUDIT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


# =============================================================================
# FINAL STATUS
# =============================================================================

print_section("FINAL AUDIT STATUS")

feature_check_pass = (
    len(missing_features) == 0
    and len(unexpected_features) == 0
    and duplicate_features == 0
)

removed_feature_check_pass = (
    len(accidental_removed_features) == 0
)

dataset_check_pass = all(
    row["shape_pass"]
    and row["target_present"]
    and row["target_nulls"] == 0
    for row in dataset_checks
)

prediction_check_pass = all(
    row["row_count_pass"]
    and row["null_probabilities"] == 0
    and row["probabilities_below_zero"] == 0
    and row["probabilities_above_one"] == 0
    for row in prediction_checks
)

print(
    f"Dataset validation: "
    f"{'PASS' if dataset_check_pass else 'FAIL'}"
)

print(
    f"Feature validation: "
    f"{'PASS' if feature_check_pass else 'FAIL'}"
)

print(
    f"Removed-feature validation: "
    f"{'PASS' if removed_feature_check_pass else 'FAIL'}"
)

print(
    f"Prediction validation: "
    f"{'PASS' if prediction_check_pass else 'FAIL'}"
)

print(
    f"Validation Log Loss: "
    f"{validation_log_loss:.6f}"
)

print(
    f"Test Log Loss: "
    f"{test_log_loss:.6f}"
)

print(
    f"Validation Brier: "
    f"{validation_brier:.6f}"
)

print(
    f"Test Brier: "
    f"{test_brier:.6f}"
)

print(
    f"Validation AUC: "
    f"{validation_auc:.6f}"
)

print(
    f"Test AUC: "
    f"{test_auc:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy:.6%}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.6%}"
)

print("\nAudit complete.")
print(f"Audit directory: {AUDIT_DIR}")