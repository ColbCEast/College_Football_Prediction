"""
Train Engineered Logistic Regression Model

This script trains a logistic regression model using engineered
home-vs-away matchup features.

Engineering strategy:
    - Convert paired home/away statistics into matchup differences
    - Create home-vs-away Elo difference
    - Preserve selected absolute strength features
    - Remove identifiers and metadata
    - Standardize using training data only
    - Preserve temporal train/validation/test separation

Target:
    win_home
        1 = home team wins
        0 = away team wins
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "modeling"

# Adjust these names if your split script uses different filenames.
TRAIN_PATH = DATA_DIR / "logistic_regression_train.csv"
VALIDATION_PATH = DATA_DIR / "logistic_regression_validation.csv"
TEST_PATH = DATA_DIR / "logistic_regression_test.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "logistic_regression"
MODEL_PATH = MODEL_DIR / "logistic_regression_engineered.joblib"
FEATURE_PATH = MODEL_DIR / "logistic_regression_engineered_features.csv"
METRICS_PATH = MODEL_DIR / "logistic_regression_engineered_metrics.json"


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "win_home"

IDENTIFIER_COLUMNS = [
    "season",
    "gameId",
    "seasonType",
    "startDate",
]

# Features where retaining the absolute home-team value makes conceptual sense.
#
# The matchup difference will still be created for these variables.
PRESERVE_HOME_FEATURES = [
    "homePregameElo",
]

# Features where retaining both absolute home and away values may be useful
# in addition to their matchup difference.
#
# We intentionally keep this list small for the first pass.
PRESERVE_ABSOLUTE_FEATURES = [
    "homePregameElo",
    "awayPregameElo",
]

RANDOM_STATE = 42


# ============================================================================
# LOGGING
# ============================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_split(path, split_name):
    """Load a temporal modeling split."""

    print(f"Loading {split_name}: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find {split_name} file:\n"
            f"  {path}\n\n"
            f"Check the output paths used by split_logistic_regression.py."
        )

    df = pd.read_csv(path)

    print(f"  Shape: {df.shape}")

    return df


# ============================================================================
# COLUMN IDENTIFICATION
# ============================================================================

def remove_identifiers(df):
    """Remove identifiers and metadata from model input."""

    columns_to_remove = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in df.columns
    ]

    columns_to_remove.append(TARGET)

    existing = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    X = df.drop(columns=existing, errors="ignore")

    return X


def canonical_feature_name(column):
    """
    Convert a feature into a canonical statistic name.

    Examples:

        homePointsForAvgBefore_home
            -> pointsForAvgBefore

        awayPointsForAvgBefore_away
            -> pointsForAvgBefore

        pointsForAvgBefore_home
            -> pointsForAvgBefore

        pointsForAvgBefore_away
            -> pointsForAvgBefore

    This allows the script to recognize redundant home/away representations.
    """

    name = column

    # Remove the side suffix.
    if name.endswith("_home"):
        name = name[:-5]
    elif name.endswith("_away"):
        name = name[:-5]

    # Remove leading home/away prefixes.
    if name.startswith("home"):
        name = name[4:]
    elif name.startswith("away"):
        name = name[4:]

    return name


def get_side(column):
    """Determine whether a column represents home or away data."""

    if column.endswith("_home"):
        return "home"

    if column.endswith("_away"):
        return "away"

    return None


# ============================================================================
# DUPLICATE FEATURE HANDLING
# ============================================================================

def remove_exact_duplicate_columns(df):
    """
    Remove columns that contain exactly the same values.

    The feature-building pipeline contains some duplicated representations,
    such as:

        homePointsForAvgBefore_home
        pointsForAvgBefore_home

    If they contain identical values, keeping both gives logistic regression
    redundant predictors without adding information.
    """

    print_header("REMOVING EXACT DUPLICATE FEATURES")

    duplicate_columns = []
    columns = list(df.columns)

    for i, column in enumerate(columns):
        if column in duplicate_columns:
            continue

        for other in columns[i + 1:]:
            if other in duplicate_columns:
                continue

            if df[column].equals(df[other]):
                duplicate_columns.append(other)

    if duplicate_columns:
        print(f"Exact duplicate columns found: {len(duplicate_columns)}")

        for column in duplicate_columns:
            print(f"  Removing: {column}")

        df = df.drop(columns=duplicate_columns)

    else:
        print("No exact duplicate columns found.")

    return df, duplicate_columns


# ============================================================================
# MATCHUP FEATURE ENGINEERING
# ============================================================================

def build_home_away_groups(df):
    """
    Build groups of home/away columns representing the same statistic.

    Returns:

        {
            canonical_name: {
                "home": [columns...],
                "away": [columns...]
            }
        }
    """

    groups = {}

    for column in df.columns:

        side = get_side(column)

        if side is None:
            continue

        canonical = canonical_feature_name(column)

        if canonical not in groups:
            groups[canonical] = {
                "home": [],
                "away": [],
            }

        groups[canonical][side].append(column)

    return groups


def choose_column(columns):
    """
    Choose a representative column from a set of equivalent columns.

    Preference:
        1. Generic feature name
        2. Explicit home/away-prefixed feature

    Example:

        completionPctBefore_home
        homeCompletionPctBefore_home

    -> completionPctBefore_home
    """

    if not columns:
        return None

    generic = [
        column
        for column in columns
        if not column.startswith("home")
        and not column.startswith("away")
    ]

    if generic:
        return sorted(generic)[0]

    return sorted(columns)[0]


def create_matchup_features(X):
    """
    Create home-minus-away matchup features.

    For every statistic represented on both sides:

        matchup_feature = home_value - away_value

    Example:

        homePointsForAvgBefore_home
        awayPointsForAvgBefore_away

    becomes:

        matchup_pointsForAvgBefore

    """

    print_header("ENGINEERING MATCHUP FEATURES")

    X = X.copy()

    groups = build_home_away_groups(X)

    engineered = pd.DataFrame(index=X.index)

    created_features = []
    skipped_features = []

    for canonical, sides in sorted(groups.items()):

        home_column = choose_column(sides["home"])
        away_column = choose_column(sides["away"])

        if home_column is None or away_column is None:
            skipped_features.append(canonical)
            continue

        home_values = pd.to_numeric(
            X[home_column],
            errors="coerce",
        )

        away_values = pd.to_numeric(
            X[away_column],
            errors="coerce",
        )

        # Only create differences for numeric data.
        if not (
            pd.api.types.is_numeric_dtype(home_values)
            and pd.api.types.is_numeric_dtype(away_values)
        ):
            skipped_features.append(canonical)
            continue

        feature_name = f"matchup_{canonical}"

        engineered[feature_name] = home_values - away_values

        created_features.append(
            {
                "feature": feature_name,
                "home_source": home_column,
                "away_source": away_column,
                "transformation": "home_minus_away",
            }
        )

    print(f"Paired statistics found: {len(created_features)}")
    print(f"Matchup features created: {len(created_features)}")
    print(f"Unpaired/skipped statistics: {len(skipped_features)}")

    return engineered, created_features


# ============================================================================
# ABSOLUTE FEATURE ENGINEERING
# ============================================================================

def add_selected_absolute_features(X, engineered):
    """
    Preserve a small number of absolute strength features.

    The first version intentionally only preserves pregame Elo.
    """

    print_header("ADDING SELECTED ABSOLUTE FEATURES")

    added = []

    for column in PRESERVE_ABSOLUTE_FEATURES:

        if column not in X.columns:
            print(f"  Not found: {column}")
            continue

        values = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        engineered[column] = values
        added.append(column)

        print(f"  Added: {column}")

    print(f"Absolute features added: {len(added)}")

    return engineered


# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

def engineer_features(df, split_name):
    """Create the engineered feature matrix for one split."""

    print_header(f"FEATURE ENGINEERING: {split_name.upper()}")

    X = remove_identifiers(df)

    print(f"Original model columns: {X.shape[1]}")

    X, duplicate_columns = remove_exact_duplicate_columns(X)

    engineered, matchup_metadata = create_matchup_features(X)

    engineered = add_selected_absolute_features(
        X,
        engineered,
    )

    print()
    print(f"Original usable columns : {X.shape[1]}")
    print(f"Engineered columns      : {engineered.shape[1]}")

    return engineered, matchup_metadata, duplicate_columns


# ============================================================================
# VALIDATION
# ============================================================================

def validate_target(df, split_name):
    """Validate target column."""

    if TARGET not in df.columns:
        raise ValueError(
            f"{TARGET} not found in {split_name} dataset."
        )

    values = df[TARGET].dropna().unique()

    if not set(values).issubset({0, 1}):
        raise ValueError(
            f"{split_name} target contains values other than 0/1: {values}"
        )

    print(
        f"{split_name} target distribution:\n"
        f"{df[TARGET].value_counts().sort_index().to_string()}"
    )


def validate_feature_alignment(X_train, X_validation, X_test):
    """Ensure all splits have identical feature columns."""

    train_columns = list(X_train.columns)
    validation_columns = list(X_validation.columns)
    test_columns = list(X_test.columns)

    if train_columns != validation_columns:
        raise ValueError(
            "Training and validation engineered features do not align."
        )

    if train_columns != test_columns:
        raise ValueError(
            "Training and test engineered features do not align."
        )

    print(f"Feature alignment verified: {len(train_columns)} features")


# ============================================================================
# MODEL
# ============================================================================

def create_model():
    """
    Create the engineered logistic regression pipeline.

    Imputation and scaling are fitted only on the training data through
    sklearn Pipeline.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X, y, split_name):
    """Evaluate the model on one temporal split."""

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
        "log_loss": log_loss(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
    }

    print_header(f"{split_name.upper()} PERFORMANCE")

    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")

    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    print()
    print("Classification Report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return metrics


# ============================================================================
# COEFFICIENT ANALYSIS
# ============================================================================

def save_feature_coefficients(model, feature_names):
    """Save engineered feature coefficients."""

    logistic_model = model.named_steps["model"]

    coefficients = logistic_model.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "odds_ratio": np.exp(coefficients),
        }
    )

    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    FEATURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficient_df.to_csv(
        FEATURE_PATH,
        index=False,
    )

    print()
    print("TOP 20 ENGINEERED FEATURES")
    print("-" * 70)

    print(
        coefficient_df[
            [
                "feature",
                "coefficient",
                "odds_ratio",
            ]
        ].head(20).to_string(index=False)
    )

    return coefficient_df


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header("TRAINING ENGINEERED LOGISTIC REGRESSION")

    # ----------------------------------------------------------------------
    # Load temporal splits
    # ----------------------------------------------------------------------

    print_header("LOADING TEMPORAL SPLITS")

    train_df = load_split(
        TRAIN_PATH,
        "Training",
    )

    validation_df = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    test_df = load_split(
        TEST_PATH,
        "Test",
    )

    # ----------------------------------------------------------------------
    # Validate targets
    # ----------------------------------------------------------------------

    print_header("VALIDATING TARGETS")

    validate_target(train_df, "Training")
    validate_target(validation_df, "Validation")
    validate_target(test_df, "Test")

    y_train = train_df[TARGET]
    y_validation = validation_df[TARGET]
    y_test = test_df[TARGET]

    # ----------------------------------------------------------------------
    # Engineer features
    # ----------------------------------------------------------------------

    X_train, matchup_metadata, duplicate_train = engineer_features(
        train_df,
        "training",
    )

    X_validation, _, duplicate_validation = engineer_features(
        validation_df,
        "validation",
    )

    X_test, _, duplicate_test = engineer_features(
        test_df,
        "test",
    )

    # ----------------------------------------------------------------------
    # Verify feature consistency
    # ----------------------------------------------------------------------

    print_header("VALIDATING ENGINEERED FEATURES")

    validate_feature_alignment(
        X_train,
        X_validation,
        X_test,
    )

    # ----------------------------------------------------------------------
    # Train model
    # ----------------------------------------------------------------------

    print_header("TRAINING LOGISTIC REGRESSION")

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Model training complete.")

    # ----------------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "Training",
    )

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    # ----------------------------------------------------------------------
    # Save coefficients
    # ----------------------------------------------------------------------

    coefficient_df = save_feature_coefficients(
        model,
        list(X_train.columns),
    )

    # ----------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print(f"Model saved to:")
    print(f"  {MODEL_PATH}")

    # ----------------------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------------------

    metrics = {
        "model": "engineered_logistic_regression",
        "target": TARGET,
        "n_training_rows": len(train_df),
        "n_validation_rows": len(validation_df),
        "n_test_rows": len(test_df),
        "n_engineered_features": X_train.shape[1],
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "engineering": {
            "matchup_features": len(matchup_metadata),
            "exact_duplicate_columns_removed": len(
                set(duplicate_train)
            ),
            "absolute_features": [
                column
                for column in PRESERVE_ABSOLUTE_FEATURES
                if column in X_train.columns
            ],
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"Metrics saved to:")
    print(f"  {METRICS_PATH}")

    # ----------------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------------

    print_header("ENGINEERED LOGISTIC REGRESSION COMPLETE")

    print(f"Training rows     : {len(train_df):,}")
    print(f"Validation rows   : {len(validation_df):,}")
    print(f"Test rows         : {len(test_df):,}")
    print(f"Engineered feats  : {X_train.shape[1]:,}")

    print()
    print("Validation Performance")
    print(
        f"  Accuracy        : {validation_metrics['accuracy']:.4f}"
    )
    print(
        f"  ROC-AUC         : {validation_metrics['roc_auc']:.4f}"
    )
    print(
        f"  Log Loss        : {validation_metrics['log_loss']:.4f}"
    )
    print(
        f"  Brier Score     : {validation_metrics['brier_score']:.4f}"
    )

    print()
    print("Test Performance")
    print(
        f"  Accuracy        : {test_metrics['accuracy']:.4f}"
    )
    print(
        f"  ROC-AUC         : {test_metrics['roc_auc']:.4f}"
    )
    print(
        f"  Log Loss        : {test_metrics['log_loss']:.4f}"
    )
    print(
        f"  Brier Score     : {test_metrics['brier_score']:.4f}"
    )


if __name__ == "__main__":
    main()