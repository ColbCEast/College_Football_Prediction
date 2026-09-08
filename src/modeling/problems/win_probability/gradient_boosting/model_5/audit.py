"""
GRADIENT BOOSTING WIN PROBABILITY - MODEL 5
DIAGNOSTIC, STABILITY & MODEL 4 COMPARISON AUDIT

Model 5:
- Uses the complete predictive-safe feature space.
- Current predictive-safe feature count: 310.
- Uses the exact hyperparameters selected by Model 4.
- Does NOT perform additional hyperparameter tuning.

Experimental progression:
    Model 3 = 28 features + baseline hyperparameters
    Model 4 = 28 features + temporally tuned hyperparameters
    Model 5 = 310 features + Model 4 hyperparameters

Primary experimental question:
    Does expanding the feature space from 28 to 310 features improve
    predictive performance when hyperparameters are held constant?

This audit:
1. Validates Model 5 artifacts and prediction files.
2. Validates the 310-feature model definition.
3. Checks prediction integrity and probability validity.
4. Checks temporal split integrity.
5. Evaluates Model 5 diagnostics and stability.
6. Compares Model 5 directly against Model 4.
7. Examines probability distribution and calibration.
8. Reports whether the expanded feature space produced a meaningful
   improvement, degradation, or essentially no change.

No model retraining is performed.
"""

from pathlib import Path

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
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

MODEL_4_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_4"
)

MODEL_5_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

FEATURE_CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_feature_classification.csv"
)

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


# =============================================================================
# EXPECTED MODEL DEFINITION
# =============================================================================

EXPECTED_FEATURE_COUNT = 310
MODEL_4_FEATURE_COUNT = 28

EXPECTED_MODEL_4_PARAMETERS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "subsample": 0.75,
    "random_state": 42,
}

EXPECTED_TRAIN_SEASONS = list(range(2015, 2023))
EXPECTED_VALIDATION_SEASONS = [2023, 2024]
EXPECTED_TEST_SEASONS = [2025]

TARGET = "win_home"
ID_COLUMN = "gameId"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    """Print a standardized audit section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_result(label, passed, detail=""):
    """Print a PASS/FAIL audit result."""
    status = "PASSED" if passed else "FAILED"
    suffix = f" - {detail}" if detail else ""
    print(f"{label}: {status}{suffix}")


def safe_metric(metric_function, y_true, y_prob, **kwargs):
    """Calculate a metric safely."""
    try:
        return metric_function(y_true, y_prob, **kwargs)
    except Exception:
        return np.nan


def calculate_metrics(df):
    """Calculate the standard prediction metrics."""
    y_true = df["win_home_actual"]
    y_prob = df["win_home_probability"]
    y_pred = df["win_home_prediction"]

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_prob),
        "Log Loss": log_loss(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
    }


def print_metrics(metrics, title):
    """Print a metric dictionary."""
    print(f"\n{title}")

    for metric, value in metrics.items():
        if pd.isna(value):
            print(f"{metric:<22}: NA")
        else:
            print(f"{metric:<22}: {value:.6f}")


def compare_metrics(model_4_metrics, model_5_metrics):
    """Compare Model 4 and Model 5 metrics."""
    print_header("MODEL 4 VS MODEL 5 PERFORMANCE COMPARISON")

    comparison_rows = []

    for metric in model_4_metrics:
        model_4_value = model_4_metrics[metric]
        model_5_value = model_5_metrics[metric]
        difference = model_5_value - model_4_value

        comparison_rows.append(
            {
                "Metric": metric,
                "Model 4": model_4_value,
                "Model 5": model_5_value,
                "Model 5 - Model 4": difference,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "Model 4": "{:.6f}".format,
                "Model 5": "{:.6f}".format,
                "Model 5 - Model 4": "{:+.6f}".format,
            },
        )
    )

    print("\nInterpretation:")
    print("  Higher is better: Accuracy, Balanced Accuracy, Precision, Recall, ROC AUC")
    print("  Lower is better: Log Loss, Brier Score")


def probability_diagnostics(df, model_name):
    """Print probability distribution diagnostics."""
    print_header(f"{model_name} PROBABILITY DIAGNOSTICS")

    probability = df["win_home_probability"]

    print(f"Mean       : {probability.mean():.6f}")
    print(f"Std Dev    : {probability.std():.6f}")
    print(f"Minimum    : {probability.min():.6f}")
    print(f"25th pct   : {probability.quantile(0.25):.6f}")
    print(f"Median     : {probability.median():.6f}")
    print(f"75th pct   : {probability.quantile(0.75):.6f}")
    print(f"Maximum    : {probability.max():.6f}")

    out_of_bounds = ((probability < 0) | (probability > 1)).sum()

    print_result(
        "Probability bounds",
        out_of_bounds == 0,
        f"{out_of_bounds} probabilities outside [0, 1]",
    )

    missing_probability = probability.isna().sum()

    print_result(
        "Missing probabilities",
        missing_probability == 0,
        f"{missing_probability} missing probabilities",
    )


def calibration_table(df, model_name, n_bins=10):
    """Create and print a simple probability calibration table."""
    print_header(f"{model_name} CALIBRATION DIAGNOSTIC")

    calibration_df = df[
        ["win_home_actual", "win_home_probability"]
    ].copy()

    calibration_df["bin"] = pd.cut(
        calibration_df["win_home_probability"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )

    grouped = (
        calibration_df
        .groupby("bin", observed=False)
        .agg(
            Games=("win_home_actual", "size"),
            Mean_Probability=("win_home_probability", "mean"),
            Actual_Win_Rate=("win_home_actual", "mean"),
        )
        .reset_index()
    )

    grouped["Calibration_Error"] = (
        grouped["Actual_Win_Rate"]
        - grouped["Mean_Probability"]
    )

    print(
        grouped.to_string(
            index=False,
            formatters={
                "Mean_Probability": lambda x: f"{x:.4f}",
                "Actual_Win_Rate": lambda x: f"{x:.4f}",
                "Calibration_Error": lambda x: f"{x:+.4f}",
            },
        )
    )

    populated = grouped[grouped["Games"] > 0]

    if len(populated) > 0:
        weighted_abs_error = (
            (
                populated["Calibration_Error"].abs()
                * populated["Games"]
            ).sum()
            / populated["Games"].sum()
        )

        print(
            f"\nWeighted absolute calibration error: "
            f"{weighted_abs_error:.6f}"
        )

    return grouped


def prediction_integrity_checks(df, split_name):
    """Validate a prediction dataframe."""
    print_header(f"{split_name.upper()} PREDICTION INTEGRITY")

    required_columns = [
        "gameId",
        "season",
        "win_home_actual",
        "win_home_probability",
        "win_home_prediction",
        "split",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    print_result(
        "Required prediction columns",
        len(missing_columns) == 0,
        f"Missing: {missing_columns}" if missing_columns else "All present",
    )

    if missing_columns:
        return False

    print_result(
        "Game IDs unique",
        df["gameId"].is_unique,
        f"{df['gameId'].duplicated().sum()} duplicate IDs",
    )

    print_result(
        "Game IDs nonmissing",
        df["gameId"].notna().all(),
        f"{df['gameId'].isna().sum()} missing IDs",
    )

    print_result(
        "Actual target nonmissing",
        df["win_home_actual"].notna().all(),
        f"{df['win_home_actual'].isna().sum()} missing targets",
    )

    target_values = set(df["win_home_actual"].dropna().unique())

    print_result(
        "Target is binary",
        target_values.issubset({0, 1}),
        f"Observed values: {sorted(target_values)}",
    )

    prediction_values = set(df["win_home_prediction"].dropna().unique())

    print_result(
        "Predictions are binary",
        prediction_values.issubset({0, 1}),
        f"Observed values: {sorted(prediction_values)}",
    )

    probability_valid = (
        df["win_home_probability"].notna().all()
        and df["win_home_probability"].between(0, 1).all()
    )

    print_result(
        "Probabilities valid",
        probability_valid,
        "All probabilities in [0, 1]" if probability_valid else "Invalid values found",
    )

    return True


def split_integrity_check(train_df, validation_df, test_df):
    """Check cross-split game ID and season integrity."""
    print_header("TEMPORAL SPLIT INTEGRITY")

    train_ids = set(train_df["gameId"])
    validation_ids = set(validation_df["gameId"])
    test_ids = set(test_df["gameId"])

    train_validation_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    validation_test_overlap = validation_ids & test_ids

    print_result(
        "Train/validation ID overlap",
        len(train_validation_overlap) == 0,
        f"{len(train_validation_overlap)} overlapping IDs",
    )

    print_result(
        "Train/test ID overlap",
        len(train_test_overlap) == 0,
        f"{len(train_test_overlap)} overlapping IDs",
    )

    print_result(
        "Validation/test ID overlap",
        len(validation_test_overlap) == 0,
        f"{len(validation_test_overlap)} overlapping IDs",
    )

    checks = [
        (
            "Training seasons",
            sorted(train_df["season"].unique()) == EXPECTED_TRAIN_SEASONS,
            sorted(train_df["season"].unique()),
        ),
        (
            "Validation seasons",
            sorted(validation_df["season"].unique()) == EXPECTED_VALIDATION_SEASONS,
            sorted(validation_df["season"].unique()),
        ),
        (
            "Test seasons",
            sorted(test_df["season"].unique()) == EXPECTED_TEST_SEASONS,
            sorted(test_df["season"].unique()),
        ),
    ]

    for label, passed, observed in checks:
        print_result(label, passed, f"Observed: {observed}")


def validate_feature_space():
    """Validate the Model 5 predictive-safe feature space."""
    print_header("MODEL 5 FEATURE-SPACE VALIDATION")

    if not FEATURE_CLASSIFICATION_PATH.exists():
        print_result(
            "Feature classification file",
            False,
            f"Not found: {FEATURE_CLASSIFICATION_PATH}",
        )
        return None

    classification = pd.read_csv(FEATURE_CLASSIFICATION_PATH)

    required_columns = {"column", "predictive_safe"}

    missing = required_columns - set(classification.columns)

    if missing:
        print_result(
            "Feature classification schema",
            False,
            f"Missing columns: {sorted(missing)}",
        )
        return None

    predictive_safe = classification.loc[
        classification["predictive_safe"].astype(bool),
        "column",
    ].tolist()

    print(f"Predictive-safe features found: {len(predictive_safe)}")

    print_result(
        "Feature count",
        len(predictive_safe) == EXPECTED_FEATURE_COUNT,
        f"Expected {EXPECTED_FEATURE_COUNT}, found {len(predictive_safe)}",
    )

    duplicate_features = (
        pd.Series(predictive_safe)
        .loc[pd.Series(predictive_safe).duplicated()]
        .tolist()
    )

    print_result(
        "Duplicate predictive-safe features",
        len(duplicate_features) == 0,
        f"Duplicates: {duplicate_features}",
    )

    leakage_columns = {
        TARGET,
        ID_COLUMN,
        "season",
        "startDate",
        "seasonType",
    }

    leakage_features = sorted(
        set(predictive_safe) & leakage_columns
    )

    print_result(
        "Target/identifier leakage",
        len(leakage_features) == 0,
        f"Leakage columns: {leakage_features}",
    )

    return predictive_safe


def validate_model_5_feature_list(predictive_safe_features):
    """Compare Model 5 saved feature list against metadata."""
    print_header("MODEL 5 SAVED FEATURE LIST")

    feature_path = MODEL_5_DIR / "feature_list.csv"

    if not feature_path.exists():
        print_result(
            "Model 5 feature list",
            False,
            f"Not found: {feature_path}",
        )
        return None

    feature_df = pd.read_csv(feature_path)

    # Handle common possible column names.
    possible_columns = [
        "feature",
        "features",
        "column",
        "feature_name",
    ]

    feature_column = next(
        (column for column in possible_columns if column in feature_df.columns),
        None,
    )

    if feature_column is None:
        print_result(
            "Feature list schema",
            False,
            f"Could not identify feature-name column. Columns: {list(feature_df.columns)}",
        )
        return None

    saved_features = feature_df[feature_column].tolist()

    print(f"Saved Model 5 features: {len(saved_features)}")

    print_result(
        "Saved feature count",
        len(saved_features) == EXPECTED_FEATURE_COUNT,
        f"Expected {EXPECTED_FEATURE_COUNT}, found {len(saved_features)}",
    )

    metadata_set = set(predictive_safe_features)
    saved_set = set(saved_features)

    missing_from_model = sorted(metadata_set - saved_set)
    unexpected_in_model = sorted(saved_set - metadata_set)

    print_result(
        "Metadata/model feature agreement",
        len(missing_from_model) == 0 and len(unexpected_in_model) == 0,
        (
            f"Missing from model: {missing_from_model}; "
            f"Unexpected: {unexpected_in_model}"
        ),
    )

    return saved_features


def dataset_feature_check(predictive_safe_features):
    """Check that the modeling datasets contain the expected feature space."""
    print_header("MODEL INPUT FEATURE-SPACE CHECK")

    datasets = {
        "Training": TRAIN_PATH,
        "Validation": VALIDATION_PATH,
        "Test": TEST_PATH,
    }

    expected = set(predictive_safe_features)

    for name, path in datasets.items():
        if not path.exists():
            print_result(
                f"{name} dataset",
                False,
                f"Not found: {path}",
            )
            continue

        df = pd.read_csv(path)

        available_features = set(df.columns)
        missing = sorted(expected - available_features)

        print_result(
            f"{name} predictive-safe features",
            len(missing) == 0,
            (
                f"All {EXPECTED_FEATURE_COUNT} present"
                if len(missing) == 0
                else f"Missing {len(missing)} features"
            ),
        )

        model_features = [
            column
            for column in predictive_safe_features
            if column in df.columns
        ]

        non_numeric = [
            column
            for column in model_features
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

        print_result(
            f"{name} feature dtypes",
            len(non_numeric) == 0,
            (
                "All 310 predictive-safe features are numeric"
                if len(non_numeric) == 0
                else f"Non-numeric: {non_numeric}"
            ),
        )


def missingness_diagnostics():
    """Report missingness across the three model input datasets."""
    print_header("MODEL 5 MISSINGNESS DIAGNOSTIC")

    datasets = {
        "Training": TRAIN_PATH,
        "Validation": VALIDATION_PATH,
        "Test": TEST_PATH,
    }

    for name, path in datasets.items():
        if not path.exists():
            continue

        df = pd.read_csv(path)

        predictive_features = [
            column
            for column in df.columns
            if column not in {
                TARGET,
                ID_COLUMN,
                "season",
                "startDate",
                "seasonType",
            }
        ]

        missing_counts = df[predictive_features].isna().sum()

        features_with_missing = (missing_counts > 0).sum()
        total_missing = missing_counts.sum()

        print(
            f"{name:<12}: "
            f"{features_with_missing}/"
            f"{len(predictive_features)} features with missing values | "
            f"{total_missing:,} missing cells"
        )

        if features_with_missing > 0:
            print(
                f"  Maximum missing in one feature: "
                f"{missing_counts.max():,}"
            )


def compare_prediction_coverage(model_4_df, model_5_df, split_name):
    """Check that Model 4 and Model 5 predict the same games."""
    print_header(f"MODEL 4 VS MODEL 5 {split_name.upper()} GAME ALIGNMENT")

    ids_4 = set(model_4_df["gameId"])
    ids_5 = set(model_5_df["gameId"])

    only_4 = ids_4 - ids_5
    only_5 = ids_5 - ids_4

    print_result(
        "Same game IDs",
        len(only_4) == 0 and len(only_5) == 0,
        (
            f"Model 4 only: {len(only_4)} | "
            f"Model 5 only: {len(only_5)}"
        ),
    )

    if len(only_4) == 0 and len(only_5) == 0:
        merged = model_4_df[
            ["gameId", "win_home_probability"]
        ].merge(
            model_5_df[
                ["gameId", "win_home_probability"]
            ],
            on="gameId",
            suffixes=("_model4", "_model5"),
        )

        probability_difference = (
            merged["win_home_probability_model5"]
            - merged["win_home_probability_model4"]
        )

        print("\nPrediction probability changes:")
        print(
            f"Mean difference : {probability_difference.mean():+.6f}"
        )
        print(
            f"Mean abs diff   : {probability_difference.abs().mean():.6f}"
        )
        print(
            f"Std difference  : {probability_difference.std():.6f}"
        )
        print(
            f"Min difference  : {probability_difference.min():+.6f}"
        )
        print(
            f"Max difference  : {probability_difference.max():+.6f}"
        )

        correlation = merged[
            [
                "win_home_probability_model4",
                "win_home_probability_model5",
            ]
        ].corr().iloc[0, 1]

        print(
            f"Prediction correlation: {correlation:.6f}"
        )

        return merged

    return None


def temporal_performance(df, model_name):
    """Calculate metrics by season."""
    print_header(f"{model_name} SEASON-BY-SEASON PERFORMANCE")

    rows = []

    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season]

        metrics = calculate_metrics(season_df)

        row = {
            "Season": season,
            "Games": len(season_df),
            **metrics,
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                metric: "{:.6f}".format
                for metric in [
                    "Accuracy",
                    "Balanced Accuracy",
                    "Precision",
                    "Recall",
                    "ROC AUC",
                    "Log Loss",
                    "Brier Score",
                ]
            },
        )
    )

    return result


def evaluate_prediction_direction(model_4_df, model_5_df):
    """Determine how often Model 5 improves/worsens individual probabilities."""
    print_header("MODEL 5 VS MODEL 4 PREDICTION MOVEMENT")

    merged = model_4_df[
        ["gameId", "win_home_actual", "win_home_probability"]
    ].merge(
        model_5_df[
            ["gameId", "win_home_probability"]
        ],
        on="gameId",
        suffixes=("_model4", "_model5"),
    )

    actual = merged["win_home_actual"]

    model_4_prob = merged["win_home_probability_model4"]
    model_5_prob = merged["win_home_probability_model5"]

    # Distance from the actual binary outcome.
    model_4_error = (model_4_prob - actual).abs()
    model_5_error = (model_5_prob - actual).abs()

    improved = (model_5_error < model_4_error).sum()
    worsened = (model_5_error > model_4_error).sum()
    unchanged = (model_5_error == model_4_error).sum()

    total = len(merged)

    print(f"Games where Model 5 moved closer to outcome : {improved:,}")
    print(f"Games where Model 5 moved farther away       : {worsened:,}")
    print(f"Games essentially unchanged                  : {unchanged:,}")

    print(
        f"\nImproved percentage: {improved / total:.2%}"
    )
    print(
        f"Worsened percentage: {worsened / total:.2%}"
    )

    return merged


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main():

    print("=" * 80)
    print("GRADIENT BOOSTING WIN PROBABILITY - MODEL 5")
    print("DIAGNOSTIC, STABILITY & MODEL 4 COMPARISON AUDIT")
    print("=" * 80)

    print("\nModel 5 definition:")
    print("  Feature space: Complete predictive-safe feature space")
    print("  Predictive-safe features: 310")
    print("  Hyperparameters: Exact Model 4 tuned parameters")
    print("  Additional tuning: None")
    print("  Primary experiment: Feature-space expansion")
    print("  Model 4: 28 features")
    print("  Model 5: 310 features")

    # =========================================================================
    # ARTIFACT CHECK
    # =========================================================================

    print_header("ARTIFACT CHECK")

    required_model_4_files = [
        "validation_predictions.csv",
        "test_predictions.csv",
        "feature_list.csv",
        "best_parameters.csv",
    ]

    required_model_5_files = [
        "validation_predictions.csv",
        "test_predictions.csv",
        "feature_list.csv",
        "training_summary.csv",
        "model.joblib",
    ]

    for filename in required_model_4_files:
        path = MODEL_4_DIR / filename
        print_result(
            f"Model 4 {filename}",
            path.exists(),
            str(path),
        )

    for filename in required_model_5_files:
        path = MODEL_5_DIR / filename
        print_result(
            f"Model 5 {filename}",
            path.exists(),
            str(path),
        )

    # =========================================================================
    # LOAD PREDICTIONS
    # =========================================================================

    model_4_validation_path = MODEL_4_DIR / "validation_predictions.csv"
    model_4_test_path = MODEL_4_DIR / "test_predictions.csv"

    model_5_validation_path = MODEL_5_DIR / "validation_predictions.csv"
    model_5_test_path = MODEL_5_DIR / "test_predictions.csv"

    if not all(
        path.exists()
        for path in [
            model_4_validation_path,
            model_4_test_path,
            model_5_validation_path,
            model_5_test_path,
        ]
    ):
        print("\nRequired prediction files are missing.")
        print("Audit cannot continue.")
        return

    model_4_validation = pd.read_csv(model_4_validation_path)
    model_4_test = pd.read_csv(model_4_test_path)

    model_5_validation = pd.read_csv(model_5_validation_path)
    model_5_test = pd.read_csv(model_5_test_path)

    print("\nPrediction files loaded successfully.")

    # =========================================================================
    # PREDICTION INTEGRITY
    # =========================================================================

    prediction_integrity_checks(
        model_5_validation,
        "Model 5 Validation",
    )

    prediction_integrity_checks(
        model_5_test,
        "Model 5 Test",
    )

    # =========================================================================
    # TEMPORAL SPLIT CHECK
    # =========================================================================

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    split_integrity_check(
        train_df,
        validation_df,
        test_df,
    )

    # =========================================================================
    # FEATURE SPACE
    # =========================================================================

    predictive_safe_features = validate_feature_space()

    if predictive_safe_features is not None:
        validate_model_5_feature_list(
            predictive_safe_features
        )

        dataset_feature_check(
            predictive_safe_features
        )

    # =========================================================================
    # MISSINGNESS
    # =========================================================================

    missingness_diagnostics()

    # =========================================================================
    # MODEL 5 METRICS
    # =========================================================================

    model_5_validation_metrics = calculate_metrics(
        model_5_validation
    )

    model_5_test_metrics = calculate_metrics(
        model_5_test
    )

    print_metrics(
        model_5_validation_metrics,
        "MODEL 5 VALIDATION METRICS",
    )

    print_metrics(
        model_5_test_metrics,
        "MODEL 5 TEST METRICS",
    )

    # =========================================================================
    # MODEL 4 METRICS
    # =========================================================================

    model_4_validation_metrics = calculate_metrics(
        model_4_validation
    )

    model_4_test_metrics = calculate_metrics(
        model_4_test
    )

    print_metrics(
        model_4_validation_metrics,
        "MODEL 4 VALIDATION METRICS",
    )

    print_metrics(
        model_4_test_metrics,
        "MODEL 4 TEST METRICS",
    )

    # =========================================================================
    # DIRECT COMPARISON
    # =========================================================================

    print("\n" + "-" * 80)
    print("VALIDATION COMPARISON")
    print("-" * 80)

    compare_metrics(
        model_4_validation_metrics,
        model_5_validation_metrics,
    )

    print("\n" + "-" * 80)
    print("TEST COMPARISON")
    print("-" * 80)

    compare_metrics(
        model_4_test_metrics,
        model_5_test_metrics,
    )

    # =========================================================================
    # GAME ALIGNMENT
    # =========================================================================

    validation_merged = compare_prediction_coverage(
        model_4_validation,
        model_5_validation,
        "Validation",
    )

    test_merged = compare_prediction_coverage(
        model_4_test,
        model_5_test,
        "Test",
    )

    # =========================================================================
    # PROBABILITY DIAGNOSTICS
    # =========================================================================

    probability_diagnostics(
        model_5_validation,
        "MODEL 5 VALIDATION",
    )

    probability_diagnostics(
        model_5_test,
        "MODEL 5 TEST",
    )

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    calibration_table(
        model_5_validation,
        "MODEL 5 VALIDATION",
    )

    calibration_table(
        model_5_test,
        "MODEL 5 TEST",
    )

    calibration_table(
        model_4_validation,
        "MODEL 4 VALIDATION",
    )

    calibration_table(
        model_4_test,
        "MODEL 4 TEST",
    )

    # =========================================================================
    # SEASONAL STABILITY
    # =========================================================================

    model_5_validation_seasonal = temporal_performance(
        model_5_validation,
        "MODEL 5 VALIDATION",
    )

    model_5_test_seasonal = temporal_performance(
        model_5_test,
        "MODEL 5 TEST",
    )

    # =========================================================================
    # PREDICTION MOVEMENT
    # =========================================================================

    if validation_merged is not None:
        evaluate_prediction_direction(
            model_4_validation,
            model_5_validation,
        )

    if test_merged is not None:
        evaluate_prediction_direction(
            model_4_test,
            model_5_test,
        )

    # =========================================================================
    # MODEL 4 HYPERPARAMETER CHECK
    # =========================================================================

    print_header("MODEL 5 HYPERPARAMETER DEFINITION")

    print("Expected Model 4 parameters reused by Model 5:")

    for parameter, value in EXPECTED_MODEL_4_PARAMETERS.items():
        print(f"  {parameter:<18}: {value}")

    print(
        "\nModel 5 should use these parameters exactly. "
        "Model 5 does not perform additional tuning."
    )

    # =========================================================================
    # OVERFITTING / GENERALIZATION DIAGNOSTIC
    # =========================================================================

    print_header("MODEL 5 GENERALIZATION DIAGNOSTIC")

    validation_log_loss = model_5_validation_metrics["Log Loss"]
    test_log_loss = model_5_test_metrics["Log Loss"]

    validation_brier = model_5_validation_metrics["Brier Score"]
    test_brier = model_5_test_metrics["Brier Score"]

    validation_auc = model_5_validation_metrics["ROC AUC"]
    test_auc = model_5_test_metrics["ROC AUC"]

    print(
        f"Validation Log Loss : {validation_log_loss:.6f}"
    )
    print(
        f"Test Log Loss       : {test_log_loss:.6f}"
    )
    print(
        f"Test - Validation   : "
        f"{test_log_loss - validation_log_loss:+.6f}"
    )

    print(
        f"\nValidation Brier    : {validation_brier:.6f}"
    )
    print(
        f"Test Brier          : {test_brier:.6f}"
    )
    print(
        f"Test - Validation   : "
        f"{test_brier - validation_brier:+.6f}"
    )

    print(
        f"\nValidation ROC AUC  : {validation_auc:.6f}"
    )
    print(
        f"Test ROC AUC        : {test_auc:.6f}"
    )
    print(
        f"Test - Validation   : "
        f"{test_auc - validation_auc:+.6f}"
    )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print_header("AUDIT INTERPRETATION")

    validation_log_loss_change = (
        model_5_validation_metrics["Log Loss"]
        - model_4_validation_metrics["Log Loss"]
    )

    test_log_loss_change = (
        model_5_test_metrics["Log Loss"]
        - model_4_test_metrics["Log Loss"]
    )

    validation_brier_change = (
        model_5_validation_metrics["Brier Score"]
        - model_4_validation_metrics["Brier Score"]
    )

    test_brier_change = (
        model_5_test_metrics["Brier Score"]
        - model_4_test_metrics["Brier Score"]
    )

    validation_auc_change = (
        model_5_validation_metrics["ROC AUC"]
        - model_4_validation_metrics["ROC AUC"]
    )

    test_auc_change = (
        model_5_test_metrics["ROC AUC"]
        - model_4_test_metrics["ROC AUC"]
    )

    print("\nFeature-space experiment:")
    print("  Model 4: 28 features")
    print("  Model 5: 310 features")
    print("  Hyperparameters: held constant")
    print()

    print(
        f"Validation Log Loss change : {validation_log_loss_change:+.6f}"
    )
    print(
        f"Test Log Loss change       : {test_log_loss_change:+.6f}"
    )
    print(
        f"Validation Brier change    : {validation_brier_change:+.6f}"
    )
    print(
        f"Test Brier change          : {test_brier_change:+.6f}"
    )
    print(
        f"Validation ROC AUC change  : {validation_auc_change:+.6f}"
    )
    print(
        f"Test ROC AUC change        : {test_auc_change:+.6f}"
    )

    print("\nInterpretation guidance:")

    if test_log_loss_change < 0:
        print(
            "  ✓ Model 5 improved test log loss relative to Model 4."
        )
    elif test_log_loss_change > 0:
        print(
            "  ! Model 5 worsened test log loss relative to Model 4."
        )
    else:
        print(
            "  = Model 5 produced identical test log loss."
        )

    if test_brier_change < 0:
        print(
            "  ✓ Model 5 improved test Brier score relative to Model 4."
        )
    elif test_brier_change > 0:
        print(
            "  ! Model 5 worsened test Brier score relative to Model 4."
        )
    else:
        print(
            "  = Model 5 produced identical test Brier score."
        )

    if test_auc_change > 0:
        print(
            "  ✓ Model 5 improved test ROC AUC relative to Model 4."
        )
    elif test_auc_change < 0:
        print(
            "  ! Model 5 reduced test ROC AUC relative to Model 4."
        )
    else:
        print(
            "  = Model 5 produced identical test ROC AUC."
        )

    print(
        "\nThe primary decision criterion should be probabilistic "
        "performance, especially Log Loss and Brier Score, rather "
        "than classification accuracy alone."
    )

    print(
        "\nIf Model 5 provides a consistent improvement across "
        "validation and test, the expanded feature space is supported."
    )

    print(
        "If performance is similar or worse, the subsequent feature "
        "selection section can investigate whether a smaller subset "
        "of the 310 features provides a better bias/variance tradeoff."
    )

    print("\n" + "=" * 80)
    print("MODEL 5 AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()