"""
Train and evaluate compact logistic regression models.

Models
------
Model 1:
    Core matchup-strength differences.

Model 2:
    Model 1 + recent-form differences.

Model 3:
    Model 2 + prior strength-of-schedule differences.

Model 4:
    Model 3 + additional matchup dimensions:
        - Turnovers
        - Third-down efficiency
        - Sacks
        - Completion percentage
        - Total yards
        - Possession time
        - Penalty yards

Temporal split
--------------
Training:
    2015-2022

Validation:
    2023-2024

Test:
    2025

The test season is used only once for final evaluation.

Input
-----
data/processed/logistic_matchup_features/
    logistic_matchup_features_{year}.csv

Output
------
data/processed/models/compact_logistic/
    models/
    predictions/
    compact_logistic_comparison.csv

Notes
-----
The matchup feature construction is performed upstream by:

    create_logistic_matchup_features.py

This script therefore does NOT reconstruct the matchup differences.
It consumes the already validated compact feature datasets directly.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TARGET_COLUMN = "win_home"

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------
# IMPORTANT:
# Use the validated compact matchup datasets rather than the enhanced
# 448-column datasets.
# --------------------------------------------------------------------------

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "logistic_regression"
    / "matchup"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "logistic_regression"
    / "compact"
)

MODEL_DIR = OUTPUT_DIR / "models"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

SUMMARY_PATH = (
    OUTPUT_DIR
    / "compact_logistic_comparison.csv"
)


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# --------------------------------------------------------------------------
# MODEL 1 — COMPACT BASELINE
# --------------------------------------------------------------------------

MODEL_1_FEATURES = [
    "winPctDiff",
    "pointDifferentialAvgDiff",
    "pointsForAvgDiff",
    "pointsAgainstAvgDiff",
    "yardsPerPassAttemptDiff",
    "yardsPerRushAttemptDiff",
]


# --------------------------------------------------------------------------
# MODEL 2 — ADD RECENT FORM
# --------------------------------------------------------------------------

MODEL_2_FEATURES = [
    "pointsForTrendDiff",
    "pointsAgainstTrendDiff",
    "pointDifferentialTrendDiff",
    "totalYardsTrendDiff",
    "netPassingYardsTrendDiff",
    "winPctTrendDiff",
]


# --------------------------------------------------------------------------
# MODEL 3 — ADD PRIOR STRENGTH OF SCHEDULE
# --------------------------------------------------------------------------

MODEL_3_FEATURES = [
    "priorSOSWinPctDiff",
    "priorSOSPointDiffDiff",
]


# --------------------------------------------------------------------------
# MODEL 4 — ADD ADDITIONAL GAME-STYLE / EFFICIENCY FEATURES
# --------------------------------------------------------------------------
#
# These features were selected specifically to add dimensions that are not
# already represented strongly by Models 1-3.
#
# Turnovers:
#     Ball security / turnover differential.
#
# Third down:
#     Sustaining drives and offensive efficiency.
#
# Sacks:
#     Pass rush / protection dimension.
#
# Completion percentage:
#     Passing efficiency distinct from yards per attempt.
#
# Total yards:
#     Overall offensive production.
#
# Possession seconds:
#     Offensive tempo / drive sustainability / ability to control possession.
#
# Penalty yards:
#     Discipline / hidden-yardage component.
# --------------------------------------------------------------------------

MODEL_4_FEATURES = [
    "turnoversAvgDiff",
    "thirdDownPctDiff",
    "sacksAvgDiff",
    "completionPctDiff",
    "totalYardsAvgDiff",
    "possessionSecondsAvgDiff",
    "penaltyYardsAvgDiff",
]


# ============================================================================
# NESTED MODEL DEFINITIONS
# ============================================================================

MODEL_FEATURES = {
    "Model 1": (
        MODEL_1_FEATURES
    ),

    "Model 2": (
        MODEL_1_FEATURES
        + MODEL_2_FEATURES
    ),

    "Model 3": (
        MODEL_1_FEATURES
        + MODEL_2_FEATURES
        + MODEL_3_FEATURES
    ),

    "Model 4": (
        MODEL_1_FEATURES
        + MODEL_2_FEATURES
        + MODEL_3_FEATURES
        + MODEL_4_FEATURES
    ),
}


# ============================================================================
# EXPECTED FEATURE COUNTS
# ============================================================================

EXPECTED_FEATURE_COUNTS = {
    "Model 1": 6,
    "Model 2": 12,
    "Model 3": 14,
    "Model 4": 21,
}


# ============================================================================
# OUTPUT / PRINT HELPERS
# ============================================================================

def print_section(title):
    """Print a standardized section header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    """Raise a validation error with a clear message."""

    raise ValueError(
        f"\nVALIDATION FAILED:\n{message}"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_season(year):
    """
    Load one validated compact matchup feature dataset.
    """

    path = (
        INPUT_DIR
        / f"logistic_matchup_features_{year}.csv"
    )

    if not path.exists():

        fail(
            f"Missing matchup feature file for {year}:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    if df.empty:

        fail(
            f"{year}: Input dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:

        fail(
            f"{year}: Target column "
            f"'{TARGET_COLUMN}' is missing."
        )

    return df


def load_all_data():
    """
    Load all seasons and combine them into one dataframe.
    """

    print_section(
        "LOADING COMPACT MATCHUP FEATURE DATA"
    )

    years = (
        TRAIN_YEARS
        + VALIDATION_YEARS
        + TEST_YEARS
    )

    frames = []

    for year in years:

        df = load_season(
            year
        )

        # Avoid repeated DataFrame.insert operations.
        df = df.copy()

        df["model_year"] = year

        frames.append(
            df
        )

        print(
            f"{year}: "
            f"{len(df):,} rows × "
            f"{len(df.columns):,} columns"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print(
        f"Combined data: "
        f"{len(data):,} rows × "
        f"{len(data.columns):,} columns"
    )

    return data


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_model_feature_definitions(
    data
):
    """
    Validate that every expected compact feature exists and that the
    nested Model 1 -> Model 4 structure is correct.
    """

    print_section(
        "VALIDATING MODEL FEATURE DEFINITIONS"
    )

    previous_features = set()

    for model_name, features in MODEL_FEATURES.items():

        expected_count = (
            EXPECTED_FEATURE_COUNTS[
                model_name
            ]
        )

        actual_count = len(
            features
        )

        # --------------------------------------------------------------
        # Count validation
        # --------------------------------------------------------------

        if actual_count != expected_count:

            fail(
                f"{model_name}: expected "
                f"{expected_count} features but "
                f"definition contains {actual_count}."
            )

        # --------------------------------------------------------------
        # Duplicate validation
        # --------------------------------------------------------------

        if len(features) != len(set(features)):

            duplicates = sorted(
                {
                    feature
                    for feature in features
                    if features.count(feature) > 1
                }
            )

            fail(
                f"{model_name}: duplicate features detected: "
                f"{duplicates}"
            )

        # --------------------------------------------------------------
        # Source dataset validation
        # --------------------------------------------------------------

        missing = [
            feature
            for feature in features
            if feature not in data.columns
        ]

        if missing:

            fail(
                f"{model_name}: missing required matchup features:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing
                )
            )

        # --------------------------------------------------------------
        # Nesting validation
        # --------------------------------------------------------------

        current_features = set(
            features
        )

        if previous_features:

            if not previous_features.issubset(
                current_features
            ):

                missing_from_current = sorted(
                    previous_features
                    - current_features
                )

                fail(
                    f"{model_name}: previous model features "
                    "are not fully contained in this model:\n"
                    + "\n".join(
                        f"  - {feature}"
                        for feature in missing_from_current
                    )
                )

        previous_features = current_features

        print(
            f"{model_name}: "
            f"{actual_count} features — VALID"
        )

    # ------------------------------------------------------------------
    # Explicit Model 4 extension validation.
    # ------------------------------------------------------------------

    model_3 = set(
        MODEL_FEATURES["Model 3"]
    )

    model_4 = set(
        MODEL_FEATURES["Model 4"]
    )

    model_4_additions = sorted(
        model_4 - model_3
    )

    expected_model_4_additions = set(
        MODEL_4_FEATURES
    )

    if (
        set(model_4_additions)
        != expected_model_4_additions
    ):

        fail(
            "Model 4 additions do not match the expected seven features."
        )

    print()
    print(
        "Model 4 additions:"
    )

    for feature in MODEL_4_FEATURES:
        print(
            f"  {feature}"
        )

    print(
        "Model nesting: VALID"
    )


# ============================================================================
# TARGET VALIDATION
# ============================================================================

def validate_target(data):
    """
    Validate the binary home-win target.
    """

    print_section(
        "VALIDATING TARGET"
    )

    if data[
        TARGET_COLUMN
    ].isna().any():

        fail(
            "Target contains missing values."
        )

    unique_target = sorted(
        data[
            TARGET_COLUMN
        ].unique()
    )

    if unique_target != [0, 1]:

        fail(
            f"Unexpected target values: "
            f"{unique_target}"
        )

    print(
        f"Target: {TARGET_COLUMN}"
    )

    print(
        f"Unique values: {unique_target}"
    )

    print(
        f"Home wins: "
        f"{int(data[TARGET_COLUMN].sum()):,}"
    )

    print(
        f"Away wins: "
        f"{int((data[TARGET_COLUMN] == 0).sum()):,}"
    )

    print(
        "Target validation: VALID"
    )


# ============================================================================
# IDENTIFIER / DATASET VALIDATION
# ============================================================================

def validate_dataset_integrity(data):
    """
    Validate basic integrity of the combined matchup datasets.
    """

    print_section(
        "VALIDATING DATASET INTEGRITY"
    )

    if "season" not in data.columns:

        fail(
            "Combined data is missing 'season'."
        )

    if "model_year" not in data.columns:

        fail(
            "Combined data is missing 'model_year'."
        )

    # --------------------------------------------------------------
    # Check model year consistency.
    # --------------------------------------------------------------

    mismatched_years = (
        data["season"]
        != data["model_year"]
    )

    if mismatched_years.any():

        count = int(
            mismatched_years.sum()
        )

        fail(
            f"Found {count:,} rows where "
            "'season' does not match 'model_year'."
        )

    # --------------------------------------------------------------
    # Game ID validation if available.
    # --------------------------------------------------------------

    game_id_candidates = [
        "id",
        "gameId",
        "game_id",
    ]

    game_id = None

    for candidate in game_id_candidates:

        if candidate in data.columns:

            game_id = candidate
            break

    if game_id is not None:

        if data[game_id].isna().any():

            fail(
                f"Game ID column '{game_id}' "
                "contains missing values."
            )

        duplicate_count = int(
            data[game_id].duplicated().sum()
        )

        if duplicate_count:

            fail(
                f"Found {duplicate_count:,} duplicate "
                f"game IDs in combined data."
            )

        print(
            f"Game ID column: {game_id}"
        )

        print(
            "Game ID uniqueness: VALID"
        )

    else:

        print(
            "Game ID column: not present "
            "(not required for model training)"
        )

    print(
        "Dataset integrity: VALID"
    )


# ============================================================================
# FEATURE NUMERIC VALIDATION
# ============================================================================

def validate_feature_types(data):
    """
    Confirm all compact matchup features are numeric.
    """

    print_section(
        "VALIDATING FEATURE TYPES"
    )

    all_features = []

    for features in MODEL_FEATURES.values():

        all_features.extend(
            features
        )

    all_features = list(
        dict.fromkeys(all_features)
    )

    for feature in all_features:

        if not pd.api.types.is_numeric_dtype(
            data[feature]
        ):

            fail(
                f"Feature '{feature}' is not numeric."
            )

        numeric_values = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        finite_values = numeric_values[
            numeric_values.notna()
        ]

        if not np.isfinite(
            finite_values.to_numpy()
        ).all():

            fail(
                f"Feature '{feature}' contains "
                "infinite values."
            )

    print(
        f"Validated features: {len(all_features)}"
    )

    print(
        "Numeric / finite validation: VALID"
    )


# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def split_data(data):
    """
    Create the fixed temporal train/validation/test split.
    """

    print_section(
        "TEMPORAL SPLIT"
    )

    train = data[
        data["model_year"].isin(
            TRAIN_YEARS
        )
    ].copy()

    validation = data[
        data["model_year"].isin(
            VALIDATION_YEARS
        )
    ].copy()

    test = data[
        data["model_year"].isin(
            TEST_YEARS
        )
    ].copy()

    print(
        f"Training seasons:   "
        f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"
    )

    print(
        f"Validation seasons: "
        f"{VALIDATION_YEARS}"
    )

    print(
        f"Test seasons:       "
        f"{TEST_YEARS}"
    )

    print()

    print(
        f"Training rows:   {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Test rows:       {len(test):,}"
    )

    if train.empty:

        fail(
            "Training dataset is empty."
        )

    if validation.empty:

        fail(
            "Validation dataset is empty."
        )

    if test.empty:

        fail(
            "Test dataset is empty."
        )

    print(
        "Temporal split: VALID"
    )

    return (
        train,
        validation,
        test,
    )


# ============================================================================
# MODEL PIPELINE
# ============================================================================

def build_pipeline(feature_names):
    """
    Create the standardized logistic regression pipeline.

    Missing values are median-imputed using training data only.
    """

    preprocessing = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                preprocessing,
                feature_names,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    probability,
):
    """
    Calculate classification and probability metrics.
    """

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "auc": roc_auc_score(
            y_true,
            probability,
        ),

        "log_loss": log_loss(
            y_true,
            probability,
        ),

        "brier_score": brier_score_loss(
            y_true,
            probability,
        ),

        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),

        "home_win_rate": float(
            np.mean(y_true)
        ),

        "mean_predicted_home_win_prob": float(
            np.mean(probability)
        ),
    }


# ============================================================================
# COEFFICIENT EXTRACTION
# ============================================================================

def extract_coefficients(
    pipeline,
    feature_names,
):
    """
    Extract logistic regression coefficients.

    Features are standardized, so coefficients represent the change in
    log-odds associated with a one-standard-deviation increase in the
    matchup feature.
    """

    model = pipeline.named_steps[
        "model"
    ]

    coefficients = (
        model.coef_[0]
    )

    rows = []

    for feature, coefficient in zip(
        feature_names,
        coefficients,
    ):

        rows.append(
            {
                "feature": feature,

                "coefficient_per_1sd":
                    float(coefficient),

                "odds_ratio_per_1sd":
                    float(
                        np.exp(coefficient)
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    model_name,
    feature_names,
    train,
    validation,
    test,
):
    """
    Train one compact logistic model and evaluate it.
    """

    print_section(
        f"TRAINING {model_name.upper()}"
    )

    print(
        f"Feature count: {len(feature_names)}"
    )

    print(
        "Features:"
    )

    for feature in feature_names:

        print(
            f"  {feature}"
        )

    # ------------------------------------------------------------------
    # Verify features exist in each split.
    # ------------------------------------------------------------------

    for split_name, df in [
        ("training", train),
        ("validation", validation),
        ("test", test),
    ]:

        missing = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing:

            fail(
                f"{model_name}: missing features in "
                f"{split_name} dataset:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing
                )
            )

    # ------------------------------------------------------------------
    # Feature matrices
    # ------------------------------------------------------------------

    X_train = train[
        feature_names
    ].copy()

    X_validation = validation[
        feature_names
    ].copy()

    X_test = test[
        feature_names
    ].copy()

    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    y_validation = validation[
        TARGET_COLUMN
    ].astype(int)

    y_test = test[
        TARGET_COLUMN
    ].astype(int)

    # ------------------------------------------------------------------
    # Missingness diagnostics
    # ------------------------------------------------------------------

    print()
    print(
        "Training missingness:"
    )

    for feature in feature_names:

        pct = (
            X_train[feature]
            .isna()
            .mean()
            * 100
        )

        print(
            f"  {feature:<35} "
            f"{pct:7.3f}%"
        )

    # ------------------------------------------------------------------
    # Build and fit
    # ------------------------------------------------------------------

    pipeline = build_pipeline(
        feature_names
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    train_probability = (
        pipeline
        .predict_proba(X_train)[:, 1]
    )

    validation_probability = (
        pipeline
        .predict_proba(X_validation)[:, 1]
    )

    test_probability = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    train_metrics = calculate_metrics(
        y_train,
        train_probability,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
    )

    print()
    print(
        "PERFORMANCE"
    )

    metrics_table = pd.DataFrame(
        [
            {
                "dataset": "train",
                **train_metrics,
            },
            {
                "dataset": "validation",
                **validation_metrics,
            },
            {
                "dataset": "test",
                **test_metrics,
            },
        ]
    )

    print(
        metrics_table.to_string(
            index=False,
            formatters={
                "auc":
                    "{:.4f}".format,

                "log_loss":
                    "{:.4f}".format,

                "brier_score":
                    "{:.4f}".format,

                "accuracy":
                    "{:.4f}".format,

                "home_win_rate":
                    "{:.4f}".format,

                "mean_predicted_home_win_prob":
                    "{:.4f}".format,
            },
        )
    )

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    coefficients = extract_coefficients(
        pipeline,
        feature_names,
    )

    print()
    print(
        "COEFFICIENTS"
    )

    print(
        coefficients.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd":
                    "{:.4f}".format,

                "odds_ratio_per_1sd":
                    "{:.4f}".format,
            },
        )
    )

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = (
        MODEL_DIR
        / model_filename
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print()
    print(
        f"Saved model: {model_path}"
    )

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_rows = []

    for split_name, df, probability in [
        (
            "train",
            train,
            train_probability,
        ),
        (
            "validation",
            validation,
            validation_probability,
        ),
        (
            "test",
            test,
            test_probability,
        ),
    ]:

        prediction = (
            probability >= 0.5
        ).astype(int)

        output = pd.DataFrame(
            {
                "model":
                    model_name,

                "dataset":
                    split_name,

                "season":
                    df["model_year"].values,

                "actual_win_home":
                    df[
                        TARGET_COLUMN
                    ].values,

                "predicted_home_win_probability":
                    probability,

                "predicted_win_home":
                    prediction,
            }
        )

        # Preserve game ID when available.

        for game_id_column in [
            "id",
            "gameId",
            "game_id",
        ]:

            if game_id_column in df.columns:

                output[
                    game_id_column
                ] = df[
                    game_id_column
                ].values

                break

        prediction_rows.append(
            output
        )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    prediction_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_predictions.csv"
    )

    prediction_path = (
        PREDICTION_DIR
        / prediction_filename
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"Saved predictions: "
        f"{prediction_path}"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary = {
        "model":
            model_name,

        "feature_count":
            len(feature_names),

        "train_rows":
            len(train),

        "validation_rows":
            len(validation),

        "test_rows":
            len(test),

        "train_auc":
            train_metrics["auc"],

        "validation_auc":
            validation_metrics["auc"],

        "test_auc":
            test_metrics["auc"],

        "train_log_loss":
            train_metrics["log_loss"],

        "validation_log_loss":
            validation_metrics["log_loss"],

        "test_log_loss":
            test_metrics["log_loss"],

        "train_brier_score":
            train_metrics["brier_score"],

        "validation_brier_score":
            validation_metrics["brier_score"],

        "test_brier_score":
            test_metrics["brier_score"],

        "train_accuracy":
            train_metrics["accuracy"],

        "validation_accuracy":
            validation_metrics["accuracy"],

        "test_accuracy":
            test_metrics["accuracy"],
    }

    return (
        summary,
        coefficients,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    print_section(
        "COMPACT LOGISTIC REGRESSION MODELING"
    )

    print(
        f"Project root:      {PROJECT_ROOT}"
    )

    print(
        f"Input directory:   {INPUT_DIR}"
    )

    print(
        f"Output directory:  {OUTPUT_DIR}"
    )

    # ------------------------------------------------------------------
    # Model design
    # ------------------------------------------------------------------

    print()
    print(
        "MODEL DESIGN"
    )
    print(
        "------------"
    )

    for model_name, features in (
        MODEL_FEATURES.items()
    ):

        print(
            f"{model_name}: "
            f"{len(features)} features"
        )

    print()
    print(
        "Model 4 additions:"
    )

    for feature in MODEL_4_FEATURES:

        print(
            f"  + {feature}"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    data = load_all_data()

    # ------------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------------

    validate_dataset_integrity(
        data
    )

    # ------------------------------------------------------------------
    # Validate features
    # ------------------------------------------------------------------

    validate_model_feature_definitions(
        data
    )

    validate_feature_types(
        data
    )

    # ------------------------------------------------------------------
    # Validate target
    # ------------------------------------------------------------------

    validate_target(
        data
    )

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------

    train, validation, test = split_data(
        data
    )

    # ------------------------------------------------------------------
    # Train Models 1-4
    # ------------------------------------------------------------------

    summaries = []

    coefficient_results = {}

    for model_name, feature_names in (
        MODEL_FEATURES.items()
    ):

        summary, coefficients = train_model(
            model_name=model_name,
            feature_names=feature_names,
            train=train,
            validation=validation,
            test=test,
        )

        summaries.append(
            summary
        )

        coefficient_results[
            model_name
        ] = coefficients

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    print_section(
        "MODEL COMPARISON"
    )

    comparison = pd.DataFrame(
        summaries
    )

    comparison = comparison.sort_values(
        "validation_log_loss",
        ascending=True,
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                column:
                    "{:.4f}".format

                for column in comparison.columns

                if column not in [
                    "model",
                    "feature_count",
                    "train_rows",
                    "validation_rows",
                    "test_rows",
                ]
            },
        )
    )

    # ------------------------------------------------------------------
    # Save comparison
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        f"Saved comparison: "
        f"{SUMMARY_PATH}"
    )

    # ------------------------------------------------------------------
    # Validation-based ranking
    # ------------------------------------------------------------------

    best_model = (
        comparison
        .sort_values(
            "validation_log_loss",
            ascending=True,
        )
        .iloc[0]
    )

    print_section(
        "VALIDATION-BASED MODEL RANKING"
    )

    print(
        f"Best validation log loss: "
        f"{best_model['model']}"
    )

    print(
        f"Validation log loss: "
        f"{best_model['validation_log_loss']:.4f}"
    )

    print(
        f"Validation AUC: "
        f"{best_model['validation_auc']:.4f}"
    )

    print()

    # ------------------------------------------------------------------
    # Explicit Model 3 vs Model 4 comparison
    # ------------------------------------------------------------------

    model_3 = comparison[
        comparison["model"] == "Model 3"
    ].iloc[0]

    model_4 = comparison[
        comparison["model"] == "Model 4"
    ].iloc[0]

    validation_log_loss_change = (
        model_4["validation_log_loss"]
        - model_3["validation_log_loss"]
    )

    validation_auc_change = (
        model_4["validation_auc"]
        - model_3["validation_auc"]
    )

    validation_brier_change = (
        model_4["validation_brier_score"]
        - model_3["validation_brier_score"]
    )

    print(
        "MODEL 3 → MODEL 4 VALIDATION CHANGE"
    )

    print(
        f"Validation log loss change: "
        f"{validation_log_loss_change:+.4f}"
    )

    print(
        f"Validation AUC change:      "
        f"{validation_auc_change:+.4f}"
    )

    print(
        f"Validation Brier change:    "
        f"{validation_brier_change:+.4f}"
    )

    print()

    if validation_log_loss_change < 0:

        print(
            "Model 4 improves validation log loss "
            "relative to Model 3."
        )

    elif validation_log_loss_change > 0:

        print(
            "Model 4 worsens validation log loss "
            "relative to Model 3."
        )

    else:

        print(
            "Model 4 has identical validation log loss "
            "to Model 3."
        )

    # ------------------------------------------------------------------
    # Test-set warning
    # ------------------------------------------------------------------

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 2025 test results are reported for "
        "final evaluation only."
    )

    print(
        "The test results should NOT be used to "
        "select between Models 1-4."
    )

    print(
        "Model selection should be based on the "
        "2023-2024 validation results."
    )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    print()
    print_section(
        "COMPACT LOGISTIC REGRESSION MODELING "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()