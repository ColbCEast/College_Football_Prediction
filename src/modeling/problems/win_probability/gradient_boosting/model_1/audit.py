"""
Gradient Boosting Model 1 — Diagnostic / Stability Audit

Audits the trained Gradient Boosting Model 1 for the win-probability
problem.

Model:
    models/win_probability/gradient_boosting/model_1/model.joblib

Input data:
    data/processed/model_inputs/win_probability/

Audit outputs:
    models/win_probability/gradient_boosting/model_1/audit/

Diagnostics:
    1. Model / dataset validation
    2. Impurity-based feature importance
    3. Permutation feature importance
    4. Feature importance by feature family
    5. Probability calibration
    6. Season-by-season performance
    7. Prediction stability
    8. Overall audit summary

Important:
    - This script does not tune the model.
    - The test set is used only for final diagnostic evaluation.
    - No model fitting occurs during the audit.
    - Permutation importance is calculated on validation data.
    - The saved model is the complete preprocessing + Gradient Boosting
      pipeline.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# =============================================================================
# PATHS
# =============================================================================

# audit.py location:
#
# College_Football_Prediction/
# └── src/
#     └── modeling/
#         └── problems/
#             └── win_probability/
#                 └── gradient_boosting/
#                     └── model_1/
#                         └── audit.py
#
# parents[6] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[6]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_1"
)

AUDIT_DIR = MODEL_DIR / "audit"

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / "model.joblib"

TARGET = "win_home"

EXCLUDED_COLUMNS = [
    "season",
    "gameId",
    "startDate",
    "seasonType",
    TARGET,
]


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# Expected number of predictors for Model 1.
EXPECTED_FEATURE_COUNT = 310

# Number of repeats used for permutation importance.
PERMUTATION_REPEATS = 10

# Calibration bins.
CALIBRATION_BINS = np.arange(
    0.0,
    1.01,
    0.05,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load validation and test datasets.

    The training data is not required for most audit calculations because
    the model has already been trained. It is loaded for structural
    validation and model metadata checks.
    """

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_path = INPUT_DIR / "train.csv"
    validation_path = INPUT_DIR / "validation.csv"
    test_path = INPUT_DIR / "test.csv"

    print(f"Training:   {train_path}")
    print(f"Validation: {validation_path}")
    print(f"Test:       {test_path}")

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print()
    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    return (
        train_df,
        validation_df,
        test_df,
    )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load the complete saved model pipeline."""

    print()
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    print(f"Model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_datasets(
    train_df,
    validation_df,
    test_df,
):
    """Validate dataset structure."""

    print()
    print("=" * 80)
    print("VALIDATING DATASETS")
    print("=" * 80)

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Test": test_df,
    }

    train_columns = list(train_df.columns)

    for name, df in datasets.items():

        if TARGET not in df.columns:
            raise ValueError(
                f"{TARGET!r} missing from {name} dataset."
            )

        if list(df.columns) != train_columns:
            raise ValueError(
                f"{name} dataset columns do not match training."
            )

        if df[TARGET].isna().any():
            raise ValueError(
                f"{name} dataset contains missing target values."
            )

        unique_targets = set(df[TARGET].unique())

        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                f"{name} target contains unexpected values: "
                f"{unique_targets}"
            )

    print("Dataset validation: PASSED")


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def prepare_features(
    validation_df,
    test_df,
):
    """Prepare validation and test feature matrices."""

    feature_columns = [
        column
        for column in validation_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected predictor count. "
            f"Expected {EXPECTED_FEATURE_COUNT}, "
            f"found {len(feature_columns)}."
        )

    X_validation = validation_df[
        feature_columns
    ].copy()

    y_validation = validation_df[
        TARGET
    ].copy()

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET
    ].copy()

    print()
    print("=" * 80)
    print("FEATURE PREPARATION")
    print("=" * 80)

    print(f"Predictors: {len(feature_columns)}")

    print()
    print("Validation shape:")
    print(f"  {X_validation.shape}")

    print()
    print("Test shape:")
    print(f"  {X_test.shape}")

    return (
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    )


# =============================================================================
# MODEL VALIDATION
# =============================================================================

def validate_model(
    model,
    feature_columns,
):
    """
    Validate the structure and configuration of the saved model.
    """

    print()
    print("=" * 80)
    print("VALIDATING MODEL")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Pipeline structure
    # -------------------------------------------------------------------------

    expected_steps = [
        "imputer",
        "classifier",
    ]

    actual_steps = list(
        model.named_steps.keys()
    )

    if actual_steps != expected_steps:
        raise ValueError(
            "Unexpected pipeline structure. "
            f"Expected {expected_steps}, "
            f"found {actual_steps}."
        )

    print("Pipeline structure: PASSED")

    # -------------------------------------------------------------------------
    # Imputer validation
    # -------------------------------------------------------------------------

    imputer = model.named_steps["imputer"]

    if imputer.strategy != "median":
        raise ValueError(
            "Unexpected imputation strategy. "
            f"Expected 'median', found {imputer.strategy!r}."
        )

    if not hasattr(imputer, "statistics_"):
        raise ValueError(
            "The imputer does not contain fitted statistics."
        )

    if len(imputer.statistics_) != len(feature_columns):
        raise ValueError(
            "Number of imputer statistics does not match "
            "number of predictor columns."
        )

    print("Median imputer validation: PASSED")

    # -------------------------------------------------------------------------
    # Classifier validation
    # -------------------------------------------------------------------------

    classifier = model.named_steps["classifier"]

    required_attributes = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "random_state",
        "feature_importances_",
    ]

    for attribute in required_attributes:

        if not hasattr(classifier, attribute):
            raise ValueError(
                f"Gradient Boosting classifier is missing "
                f"expected attribute: {attribute}"
            )

    if len(classifier.feature_importances_) != len(
        feature_columns
    ):
        raise ValueError(
            "Number of feature importances does not match "
            "number of predictor columns."
        )

    print("Gradient Boosting classifier validation: PASSED")

    # -------------------------------------------------------------------------
    # Configuration summary
    # -------------------------------------------------------------------------

    print()
    print("Gradient Boosting configuration:")

    print(
        f"  n_estimators:    "
        f"{classifier.n_estimators}"
    )

    print(
        f"  learning_rate:   "
        f"{classifier.learning_rate}"
    )

    print(
        f"  max_depth:       "
        f"{classifier.max_depth}"
    )

    print(
        f"  min_samples_leaf:"
        f" {classifier.min_samples_leaf}"
    )

    print(
        f"  subsample:       "
        f"{classifier.subsample}"
    )

    print(
        f"  random_state:    "
        f"{classifier.random_state}"
    )

    print()
    print("Model validation: PASSED")


# =============================================================================
# PREDICTIONS
# =============================================================================

def generate_predictions(
    model,
    X_validation,
    X_test,
):
    """Generate probability predictions."""

    print()
    print("=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # -------------------------------------------------------------------------
    # Probability validation
    # -------------------------------------------------------------------------

    for name, probabilities in [
        ("Validation", validation_probabilities),
        ("Test", test_probabilities),
    ]:

        if np.isnan(probabilities).any():
            raise ValueError(
                f"{name} predictions contain NaN values."
            )

        if np.isinf(probabilities).any():
            raise ValueError(
                f"{name} predictions contain infinite values."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                f"{name} predictions contain probabilities "
                "outside [0, 1]."
            )

    print("Validation predictions generated.")
    print("Test predictions generated.")
    print("Probability validation: PASSED")

    return (
        validation_probabilities,
        test_probabilities,
    )


# =============================================================================
# OVERALL PERFORMANCE
# =============================================================================

def calculate_performance(
    name,
    y_true,
    probabilities,
):
    """Calculate overall model performance."""

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "dataset": name,
        "n_games": len(y_true),
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
        "mean_predicted_probability": np.mean(
            probabilities
        ),
        "actual_home_win_rate": np.mean(
            y_true
        ),
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_feature_importance(
    model,
    feature_columns,
):
    """
    Extract impurity-based feature importance from Gradient Boosting.
    """

    print()
    print("=" * 80)
    print("CALCULATING FEATURE IMPORTANCE")
    print("=" * 80)

    classifier = model.named_steps[
        "classifier"
    ]

    importances = classifier.feature_importances_

    if len(importances) != len(feature_columns):
        raise ValueError(
            "Number of feature importances does not match "
            "number of predictor columns."
        )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df[
        "importance_rank"
    ] = np.arange(
        1,
        len(importance_df) + 1,
    )

    importance_df = importance_df[
        [
            "importance_rank",
            "feature",
            "importance",
        ]
    ]

    print()
    print("Top 20 features:")

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    return importance_df


# =============================================================================
# PERMUTATION IMPORTANCE
# =============================================================================

def calculate_permutation_importance(
    model,
    X_validation,
    y_validation,
    feature_columns,
):
    """
    Calculate permutation importance on the validation dataset.

    Scoring uses negative log loss so that larger importance values indicate
    a larger deterioration in log loss when the feature is permuted.
    """

    print()
    print("=" * 80)
    print("CALCULATING PERMUTATION IMPORTANCE")
    print("=" * 80)

    print(
        f"Repeats per feature: {PERMUTATION_REPEATS}"
    )

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="neg_log_loss",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df[
        "importance_rank"
    ] = np.arange(
        1,
        len(importance_df) + 1,
    )

    importance_df = importance_df[
        [
            "importance_rank",
            "feature",
            "importance_mean",
            "importance_std",
        ]
    ]

    print()
    print("Top 20 features:")

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    return importance_df


# =============================================================================
# FEATURE FAMILY CLASSIFICATION
# =============================================================================

def classify_feature(
    feature,
):
    """
    Assign a broad feature family based on the feature name.

    This is intentionally descriptive rather than prescriptive. It allows
    us to understand where the Gradient Boosting model is obtaining
    predictive signal.
    """

    feature_lower = feature.lower()

    # -------------------------------------------------------------------------
    # Elo
    # -------------------------------------------------------------------------

    if "elo" in feature_lower:
        return "Elo"

    # -------------------------------------------------------------------------
    # Trend / rolling features
    # -------------------------------------------------------------------------

    if (
        "last3" in feature_lower
        or "last5" in feature_lower
        or "trend" in feature_lower
    ):
        return "Rolling / Trend"

    # -------------------------------------------------------------------------
    # Passing
    # -------------------------------------------------------------------------

    passing_terms = [
        "passing",
        "passattempt",
        "completion",
        "completions",
        "yardsperpass",
        "interception",
        "sack",
        "qbhur",
    ]

    if any(
        term in feature_lower
        for term in passing_terms
    ):
        return "Passing"

    # -------------------------------------------------------------------------
    # Rushing
    # -------------------------------------------------------------------------

    rushing_terms = [
        "rushing",
        "yardsperrush",
    ]

    if any(
        term in feature_lower
        for term in rushing_terms
    ):
        return "Rushing"

    # -------------------------------------------------------------------------
    # Scoring / results
    # -------------------------------------------------------------------------

    scoring_terms = [
        "pointsfor",
        "pointsagainst",
        "pointdifferential",
        "winpct",
        "winsbefore",
    ]

    if any(
        term in feature_lower
        for term in scoring_terms
    ):
        return "Scoring / Results"

    # -------------------------------------------------------------------------
    # Defense
    # -------------------------------------------------------------------------

    defense_terms = [
        "defensive",
        "tacklesforloss",
        "passesdeflected",
    ]

    if any(
        term in feature_lower
        for term in defense_terms
    ):
        return "Defense"

    # -------------------------------------------------------------------------
    # Turnovers
    # -------------------------------------------------------------------------

    if "turnover" in feature_lower:
        return "Turnovers"

    if "fumbleslost" in feature_lower:
        return "Turnovers"

    # -------------------------------------------------------------------------
    # Situational
    # -------------------------------------------------------------------------

    situational_terms = [
        "thirddown",
        "fourthdown",
        "possession",
    ]

    if any(
        term in feature_lower
        for term in situational_terms
    ):
        return "Situational"

    # -------------------------------------------------------------------------
    # Total offense
    # -------------------------------------------------------------------------

    if "totalyards" in feature_lower:
        return "Total Offense"

    # -------------------------------------------------------------------------
    # Penalties
    # -------------------------------------------------------------------------

    if (
        "penalt" in feature_lower
        or "penalty" in feature_lower
    ):
        return "Penalties"

    # -------------------------------------------------------------------------
    # Advanced metrics
    # -------------------------------------------------------------------------

    advanced_terms = [
        "ppa",
        "successrate",
        "explosiveness",
    ]

    if any(
        term in feature_lower
        for term in advanced_terms
    ):
        return "Advanced Metrics"

    # -------------------------------------------------------------------------
    # Default
    # -------------------------------------------------------------------------

    return "Other"


# =============================================================================
# FEATURE IMPORTANCE BY FAMILY
# =============================================================================

def calculate_family_importance(
    importance_df,
):
    """
    Aggregate Gradient Boosting impurity importance by feature family.
    """

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE BY FAMILY")
    print("=" * 80)

    family_df = importance_df.copy()

    family_df[
        "feature_family"
    ] = family_df[
        "feature"
    ].apply(
        classify_feature
    )

    grouped = (
        family_df
        .groupby("feature_family")
        .agg(
            feature_count=(
                "feature",
                "count",
            ),
            total_importance=(
                "importance",
                "sum",
            ),
            mean_importance=(
                "importance",
                "mean",
            ),
            median_importance=(
                "importance",
                "median",
            ),
        )
        .reset_index()
    )

    grouped[
        "importance_share"
    ] = (
        grouped["total_importance"]
        / grouped["total_importance"].sum()
    )

    grouped = grouped.sort_values(
        "total_importance",
        ascending=False,
    ).reset_index(drop=True)

    print(
        grouped.to_string(
            index=False
        )
    )

    return grouped


# =============================================================================
# CALIBRATION
# =============================================================================

def calculate_calibration(
    name,
    y_true,
    probabilities,
):
    """
    Calculate calibration statistics using fixed probability bins.
    """

    print()
    print("=" * 80)
    print(f"{name.upper()} CALIBRATION")
    print("=" * 80)

    calibration_rows = []

    for lower, upper in zip(
        CALIBRATION_BINS[:-1],
        CALIBRATION_BINS[1:],
    ):

        if upper == 1.0:
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

            calibration_rows.append({
                "dataset": name,
                "bin_lower": lower,
                "bin_upper": upper,
                "n_games": 0,
                "mean_predicted_probability": np.nan,
                "observed_home_win_rate": np.nan,
                "calibration_error": np.nan,
            })

            continue

        mean_probability = np.mean(
            probabilities[mask]
        )

        observed_rate = np.mean(
            y_true[mask]
        )

        calibration_error = (
            observed_rate
            - mean_probability
        )

        calibration_rows.append({
            "dataset": name,
            "bin_lower": lower,
            "bin_upper": upper,
            "n_games": count,
            "mean_predicted_probability": mean_probability,
            "observed_home_win_rate": observed_rate,
            "calibration_error": calibration_error,
        })

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    return calibration_df


# =============================================================================
# SEASON PERFORMANCE
# =============================================================================

def calculate_season_performance(
    model,
    test_df,
):
    """
    Calculate performance for each test season.

    The current test dataset is expected to contain 2025 only, but this
    function is written generally so it remains useful if the split changes.
    """

    print()
    print("=" * 80)
    print("SEASON-BY-SEASON PERFORMANCE")
    print("=" * 80)

    feature_columns = [
        column
        for column in test_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    rows = []

    for season in sorted(
        test_df["season"].unique()
    ):

        season_df = test_df[
            test_df["season"] == season
        ]

        X = season_df[
            feature_columns
        ]

        y = season_df[
            TARGET
        ]

        probabilities = model.predict_proba(
            X
        )[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        rows.append({
            "season": season,
            "rows": len(season_df),
            "auc": roc_auc_score(
                y,
                probabilities,
            ),
            "log_loss": log_loss(
                y,
                probabilities,
            ),
            "brier_score": brier_score_loss(
                y,
                probabilities,
            ),
            "accuracy": accuracy_score(
                y,
                predictions,
            ),
            "home_win_rate": np.mean(y),
            "mean_predicted_home_win_prob": np.mean(
                probabilities
            ),
        })

    season_df = pd.DataFrame(rows)

    print(
        season_df.to_string(
            index=False
        )
    )

    return season_df


# =============================================================================
# PREDICTION STABILITY
# =============================================================================

def calculate_prediction_stability(
    probabilities,
    name,
):
    """
    Summarize the distribution of predicted probabilities.
    """

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    rows = []

    for percentile in percentiles:

        rows.append({
            "dataset": name,
            "statistic": f"p{percentile}",
            "value": np.percentile(
                probabilities,
                percentile,
            ),
        })

    rows.extend([
        {
            "dataset": name,
            "statistic": "mean",
            "value": np.mean(
                probabilities
            ),
        },
        {
            "dataset": name,
            "statistic": "std",
            "value": np.std(
                probabilities
            ),
        },
        {
            "dataset": name,
            "statistic": "min",
            "value": np.min(
                probabilities
            ),
        },
        {
            "dataset": name,
            "statistic": "max",
            "value": np.max(
                probabilities
            ),
        },
    ])

    return pd.DataFrame(rows)


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    audit_summary,
    feature_importance,
    permutation_importance_df,
    family_importance,
    calibration_df,
    season_performance,
    prediction_stability,
):
    """Save all audit outputs."""

    print()
    print("=" * 80)
    print("SAVING AUDIT OUTPUTS")
    print("=" * 80)

    outputs = {
        "audit_summary.csv": audit_summary,
        "feature_importance.csv": feature_importance,
        "permutation_importance.csv": permutation_importance_df,
        "feature_importance_by_family.csv": family_importance,
        "calibration.csv": calibration_df,
        "season_performance.csv": season_performance,
        "prediction_stability.csv": prediction_stability,
    }

    for filename, dataframe in outputs.items():

        path = AUDIT_DIR / filename

        dataframe.to_csv(
            path,
            index=False,
        )

        print(f"Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("GRADIENT BOOSTING MODEL 1 — DIAGNOSTIC / STABILITY AUDIT")
    print("=" * 80)

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Model directory:")
    print(f"  {MODEL_DIR}")

    print()
    print("Audit directory:")
    print(f"  {AUDIT_DIR}")

    # =========================================================================
    # 1. Load data
    # =========================================================================

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    # =========================================================================
    # 2. Load model
    # =========================================================================

    model = load_model()

    # =========================================================================
    # 3. Validate datasets
    # =========================================================================

    validate_datasets(
        train_df,
        validation_df,
        test_df,
    )

    # =========================================================================
    # 4. Prepare features
    # =========================================================================

    (
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        validation_df,
        test_df,
    )

    # =========================================================================
    # 5. Validate model
    # =========================================================================

    validate_model(
        model,
        feature_columns,
    )

    # =========================================================================
    # 6. Generate predictions
    # =========================================================================

    (
        validation_probabilities,
        test_probabilities,
    ) = generate_predictions(
        model,
        X_validation,
        X_test,
    )

    # =========================================================================
    # 7. Overall performance
    # =========================================================================

    validation_performance = calculate_performance(
        "Validation",
        y_validation,
        validation_probabilities,
    )

    test_performance = calculate_performance(
        "Test",
        y_test,
        test_probabilities,
    )

    print()
    print("=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    for performance in [
        validation_performance,
        test_performance,
    ]:

        print()
        print(
            f"{performance['dataset']}:"
        )

        print(
            f"  Log Loss:    "
            f"{performance['log_loss']:.6f}"
        )

        print(
            f"  Brier Score: "
            f"{performance['brier_score']:.6f}"
        )

        print(
            f"  ROC AUC:     "
            f"{performance['roc_auc']:.6f}"
        )

        print(
            f"  Accuracy:    "
            f"{performance['accuracy']:.4%}"
        )

        print(
            f"  Mean Pred:   "
            f"{performance['mean_predicted_probability']:.6f}"
        )

        print(
            f"  Actual Rate: "
            f"{performance['actual_home_win_rate']:.6f}"
        )

    # =========================================================================
    # 8. Feature importance
    # =========================================================================

    feature_importance = calculate_feature_importance(
        model,
        feature_columns,
    )

    # =========================================================================
    # 9. Permutation importance
    # =========================================================================

    permutation_importance_df = (
        calculate_permutation_importance(
            model,
            X_validation,
            y_validation,
            feature_columns,
        )
    )

    # =========================================================================
    # 10. Feature-family importance
    # =========================================================================

    family_importance = calculate_family_importance(
        feature_importance
    )

    # =========================================================================
    # 11. Calibration
    # =========================================================================

    validation_calibration = calculate_calibration(
        "Validation",
        y_validation,
        validation_probabilities,
    )

    test_calibration = calculate_calibration(
        "Test",
        y_test,
        test_probabilities,
    )

    calibration_df = pd.concat(
        [
            validation_calibration,
            test_calibration,
        ],
        ignore_index=True,
    )

    # =========================================================================
    # 12. Season performance
    # =========================================================================

    season_performance = calculate_season_performance(
        model,
        test_df,
    )

    # =========================================================================
    # 13. Prediction stability
    # =========================================================================

    validation_stability = (
        calculate_prediction_stability(
            validation_probabilities,
            "Validation",
        )
    )

    test_stability = (
        calculate_prediction_stability(
            test_probabilities,
            "Test",
        )
    )

    prediction_stability = pd.concat(
        [
            validation_stability,
            test_stability,
        ],
        ignore_index=True,
    )

    # =========================================================================
    # 14. Audit summary
    # =========================================================================

    classifier = model.named_steps[
        "classifier"
    ]

    audit_summary = pd.DataFrame([
        {
            "metric": "predictor_count",
            "value": len(feature_columns),
        },
        {
            "metric": "n_estimators",
            "value": classifier.n_estimators,
        },
        {
            "metric": "learning_rate",
            "value": classifier.learning_rate,
        },
        {
            "metric": "max_depth",
            "value": classifier.max_depth,
        },
        {
            "metric": "min_samples_leaf",
            "value": classifier.min_samples_leaf,
        },
        {
            "metric": "subsample",
            "value": classifier.subsample,
        },
        {
            "metric": "random_state",
            "value": classifier.random_state,
        },
        {
            "metric": "validation_log_loss",
            "value": validation_performance[
                "log_loss"
            ],
        },
        {
            "metric": "test_log_loss",
            "value": test_performance[
                "log_loss"
            ],
        },
        {
            "metric": "validation_brier_score",
            "value": validation_performance[
                "brier_score"
            ],
        },
        {
            "metric": "test_brier_score",
            "value": test_performance[
                "brier_score"
            ],
        },
        {
            "metric": "validation_auc",
            "value": validation_performance[
                "roc_auc"
            ],
        },
        {
            "metric": "test_auc",
            "value": test_performance[
                "roc_auc"
            ],
        },
        {
            "metric": "validation_accuracy",
            "value": validation_performance[
                "accuracy"
            ],
        },
        {
            "metric": "test_accuracy",
            "value": test_performance[
                "accuracy"
            ],
        },
    ])

    # =========================================================================
    # 15. Save outputs
    # =========================================================================

    save_outputs(
        audit_summary,
        feature_importance,
        permutation_importance_df,
        family_importance,
        calibration_df,
        season_performance,
        prediction_stability,
    )

    # =========================================================================
    # 16. Final summary
    # =========================================================================

    print()
    print("=" * 80)
    print("GRADIENT BOOSTING MODEL 1 AUDIT COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Predictors:          "
        f"{len(feature_columns)}"
    )

    print(
        f"Validation Log Loss: "
        f"{validation_performance['log_loss']:.6f}"
    )

    print(
        f"Test Log Loss:       "
        f"{test_performance['log_loss']:.6f}"
    )

    print(
        f"Validation Brier:    "
        f"{validation_performance['brier_score']:.6f}"
    )

    print(
        f"Test Brier:          "
        f"{test_performance['brier_score']:.6f}"
    )

    print(
        f"Validation AUC:      "
        f"{validation_performance['roc_auc']:.6f}"
    )

    print(
        f"Test AUC:            "
        f"{test_performance['roc_auc']:.6f}"
    )

    print()
    print("Audit outputs saved to:")
    print(f"  {AUDIT_DIR}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()