"""
Train Random Forest Model 3
============================

Purpose
-------
Train a compact Random Forest model using the same core feature set as
Random Forest Model 2, but WITHOUT the engineered Trend and prior
Strength-of-Schedule (SOS) features.

This creates a controlled A/B experiment:

Model 2:
    Compact base features
    + Trend features
    + Prior SOS features

Model 3:
    Compact base features
    - Trend features
    - Prior SOS features

The purpose is to determine whether the engineered Trend and SOS feature
families improve out-of-sample win-probability prediction.

Feature comparison
------------------
Model 2:
    44 predictors

Model 3:
    28 predictors

The only intended difference between Model 2 and Model 3 is the removal
of the 16 engineered Trend/SOS predictors.

All Random Forest hyperparameters, temporal splits, preprocessing,
evaluation metrics, and random state remain unchanged.

Temporal split
--------------
Training:
    2015-2022

Validation:
    2023-2024

Test:
    2025

Target
------
win_home

Important
---------
- The test set is used only for final evaluation.
- The feature set is fixed before model evaluation.
- No feature selection is performed using validation/test performance.
- Median imputation is fitted using training data only.
- The exact Model 3 feature list is saved.
- Raw Random Forest probabilities are preserved for later calibration.
- Model 3 does NOT use the enhanced logistic feature files.
- Model 3 is intentionally evaluated using the same base model-input
  files as Model 2.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

TARGET_COLUMN = "win_home"
GAME_ID_COLUMN = "gameId"

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]


# ----------------------------------------------------------------------------
# Random Forest configuration
#
# These are intentionally identical to Model 2.
# ----------------------------------------------------------------------------

N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
CLASS_WEIGHT = None
N_JOBS = -1


# ============================================================================
# PROJECT PATHS
# ============================================================================

# train.py is located at:
#
# src/
#   modeling/
#     problems/
#       win_probability/
#         random_forest/
#           model_3/
#             train.py
#
# parents[6] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[6]


MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)


TRAIN_PATH = (
    MODEL_INPUT_DIR
    / "train.csv"
)

VALIDATION_PATH = (
    MODEL_INPUT_DIR
    / "validation.csv"
)

TEST_PATH = (
    MODEL_INPUT_DIR
    / "test.csv"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "random_forest"
    / "model_3"
)


MODEL_OUTPUT_PATH = (
    MODEL_DIR
    / "model.joblib"
)

FEATURE_LIST_OUTPUT_PATH = (
    MODEL_DIR
    / "feature_list.csv"
)

TRAINING_SUMMARY_OUTPUT_PATH = (
    MODEL_DIR
    / "training_summary.csv"
)

VALIDATION_PREDICTIONS_OUTPUT_PATH = (
    MODEL_DIR
    / "validation_predictions.csv"
)

TEST_PREDICTIONS_OUTPUT_PATH = (
    MODEL_DIR
    / "test_predictions.csv"
)


# ============================================================================
# MODEL 3 FEATURE DEFINITIONS
# ============================================================================

"""
Model 3 is intentionally identical to the Model 2 compact feature set
except that Trend and Prior SOS features are removed.

Model 2:
    Core strength:          8
    Recent form:            8
    Trend:                 12
    Prior SOS:              4
    Offensive efficiency:   8
    Defensive efficiency:   4

    Total:                  44

Model 3:
    Core strength:          8
    Recent form:            8
    Offensive efficiency:   8
    Defensive efficiency:   4

    Total:                  28

Therefore:

    Model 2 → Model 3
    44 predictors → 28 predictors

Exactly 16 predictors are removed:

    - 12 Trend features
    - 4 Prior SOS features
"""


# ----------------------------------------------------------------------------
# Core strength
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Recent form
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Offensive efficiency
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Defensive efficiency
# ----------------------------------------------------------------------------

DEFENSIVE_FEATURES = [
    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",

    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",
]


# ----------------------------------------------------------------------------
# Features intentionally excluded from Model 3
#
# These are retained here explicitly so the experiment documents exactly
# what was removed relative to Model 2.
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Final Model 3 feature list
# ----------------------------------------------------------------------------

MODEL_3_FEATURES = (
    CORE_STRENGTH_FEATURES
    + RECENT_FORM_FEATURES
    + OFFENSIVE_FEATURES
    + DEFENSIVE_FEATURES
)


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_split(path, name):
    """
    Load a generic win-probability modeling split.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{name} split does not exist:\n"
            f"  {path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name:<12}: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    return df


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def validate_splits(train, validation, test):

    print_section(
        "VALIDATING MODELING SPLITS"
    )

    required = {
        TARGET_COLUMN,
        GAME_ID_COLUMN,
        "season",
    }

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing)}"
            )

        if df[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"{name} contains missing target values."
            )

        if df[GAME_ID_COLUMN].isna().any():
            raise ValueError(
                f"{name} contains missing game IDs."
            )

        if df[GAME_ID_COLUMN].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate game IDs."
            )

        target_values = sorted(
            df[TARGET_COLUMN]
            .astype(int)
            .unique()
            .tolist()
        )

        if target_values != [0, 1]:
            raise ValueError(
                f"{name} has unexpected target values: "
                f"{target_values}"
            )

    # ------------------------------------------------------------------------
    # Check for game ID overlap
    # ------------------------------------------------------------------------

    train_ids = set(
        train[GAME_ID_COLUMN]
    )

    validation_ids = set(
        validation[GAME_ID_COLUMN]
    )

    test_ids = set(
        test[GAME_ID_COLUMN]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Training and validation game IDs overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Training and test game IDs overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test game IDs overlap."
        )

    # ------------------------------------------------------------------------
    # Validate temporal split
    # ------------------------------------------------------------------------

    train_seasons = sorted(
        train["season"].unique()
    )

    validation_seasons = sorted(
        validation["season"].unique()
    )

    test_seasons = sorted(
        test["season"].unique()
    )

    print()
    print(
        f"Training seasons:   {train_seasons}"
    )

    print(
        f"Validation seasons: {validation_seasons}"
    )

    print(
        f"Test seasons:       {test_seasons}"
    )

    if train_seasons != TRAIN_YEARS:
        raise ValueError(
            f"Unexpected training seasons: "
            f"{train_seasons}"
        )

    if validation_seasons != VALIDATION_YEARS:
        raise ValueError(
            f"Unexpected validation seasons: "
            f"{validation_seasons}"
        )

    if test_seasons != TEST_YEARS:
        raise ValueError(
            f"Unexpected test seasons: "
            f"{test_seasons}"
        )

    print()
    print(
        "Dataset validation passed."
    )


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(train, validation, test):

    print_section(
        "VALIDATING MODEL 3 FEATURES"
    )

    print(
        "Model 2 predictors: 44"
    )

    print(
        f"Model 3 predictors: "
        f"{len(MODEL_3_FEATURES)}"
    )

    print(
        "Removed Trend predictors: "
        f"{len(REMOVED_TREND_FEATURES)}"
    )

    print(
        "Removed SOS predictors: "
        f"{len(REMOVED_SOS_FEATURES)}"
    )

    print(
        "Total removed predictors: "
        f"{len(REMOVED_TREND_FEATURES) + len(REMOVED_SOS_FEATURES)}"
    )

    # ------------------------------------------------------------------------
    # Check for duplicate features
    # ------------------------------------------------------------------------

    if len(MODEL_3_FEATURES) != len(
        set(MODEL_3_FEATURES)
    ):

        duplicates = sorted(
            {
                feature
                for feature in MODEL_3_FEATURES
                if MODEL_3_FEATURES.count(feature) > 1
            }
        )

        raise ValueError(
            "Duplicate Model 3 features:\n"
            + "\n".join(
                f"  {feature}"
                for feature in duplicates
            )
        )

    # ------------------------------------------------------------------------
    # Validate required features
    # ------------------------------------------------------------------------

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = [
            feature
            for feature in MODEL_3_FEATURES
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing Model 3 features:\n"
                + "\n".join(
                    f"  {feature}"
                    for feature in missing
                )
            )

    # ------------------------------------------------------------------------
    # Explicitly verify removed features are not part of Model 3
    # ------------------------------------------------------------------------

    removed_features = (
        REMOVED_TREND_FEATURES
        + REMOVED_SOS_FEATURES
    )

    accidental_inclusions = [
        feature
        for feature in removed_features
        if feature in MODEL_3_FEATURES
    ]

    if accidental_inclusions:

        raise ValueError(
            "Model 3 accidentally includes features that "
            "should have been removed:\n"
            + "\n".join(
                f"  {feature}"
                for feature in accidental_inclusions
            )
        )

    # ------------------------------------------------------------------------
    # Print feature groups
    # ------------------------------------------------------------------------

    print()
    print(
        "Feature validation passed."
    )

    print()
    print(
        "Model 3 feature groups:"
    )

    print(
        f"  Core strength:        "
        f"{len(CORE_STRENGTH_FEATURES)}"
    )

    print(
        f"  Recent form:          "
        f"{len(RECENT_FORM_FEATURES)}"
    )

    print(
        f"  Offensive efficiency: "
        f"{len(OFFENSIVE_FEATURES)}"
    )

    print(
        f"  Defensive efficiency: "
        f"{len(DEFENSIVE_FEATURES)}"
    )

    print()
    print(
        f"Total predictors: "
        f"{len(MODEL_3_FEATURES)}"
    )

    print()
    print(
        "Excluded feature groups:"
    )

    print(
        f"  Trend:                "
        f"{len(REMOVED_TREND_FEATURES)}"
    )

    print(
        f"  Prior SOS:            "
        f"{len(REMOVED_SOS_FEATURES)}"
    )


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def prepare_features(df):

    X = df[
        MODEL_3_FEATURES
    ].copy()

    for column in MODEL_3_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


# ============================================================================
# CREATE MODEL
# ============================================================================

def create_model():

    imputer = SimpleImputer(
        strategy="median"
    )

    random_forest = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                imputer,
            ),
            (
                "model",
                random_forest,
            ),
        ]
    )


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    probability,
):

    prediction = (
        probability >= 0.50
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
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
    }


def print_metrics(
    name,
    y_true,
    probability,
):

    metrics = calculate_metrics(
        y_true,
        probability,
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        prediction,
    )

    print_section(
        f"{name.upper()} PERFORMANCE"
    )

    print(
        f"Log Loss:            "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Brier Score:         "
        f"{metrics['brier_score']:.6f}"
    )

    print(
        f"ROC AUC:             "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.4%}"
    )

    print(
        f"Balanced Accuracy:   "
        f"{metrics['balanced_accuracy']:.4%}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.4%}"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.4%}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        "                 Predicted"
    )

    print(
        "                 Away   Home"
    )

    print(
        f"Actual Away      "
        f"{matrix[0, 0]:5d}  "
        f"{matrix[0, 1]:5d}"
    )

    print(
        f"Actual Home      "
        f"{matrix[1, 0]:5d}  "
        f"{matrix[1, 1]:5d}"
    )

    return metrics


# ============================================================================
# PREDICTION OUTPUT
# ============================================================================

def create_prediction_dataframe(
    df,
    probability,
    split_name,
):

    return pd.DataFrame(
        {
            "gameId": df[
                GAME_ID_COLUMN
            ].values,

            "season": df[
                "season"
            ].values,

            "win_home_actual": df[
                TARGET_COLUMN
            ].astype(int).values,

            "win_home_probability": probability,

            "win_home_prediction": (
                probability >= 0.50
            ).astype(int),

            "split": split_name,
        }
    )


# ============================================================================
# SAVE FEATURE LIST
# ============================================================================

def save_feature_list():

    rows = []

    groups = [
        (
            "Core strength",
            CORE_STRENGTH_FEATURES,
        ),
        (
            "Recent form",
            RECENT_FORM_FEATURES,
        ),
        (
            "Offensive efficiency",
            OFFENSIVE_FEATURES,
        ),
        (
            "Defensive efficiency",
            DEFENSIVE_FEATURES,
        ),
    ]

    rank = 1

    for group, features in groups:

        for feature in features:

            rows.append(
                {
                    "feature_rank": rank,
                    "feature": feature,
                    "feature_group": group,
                }
            )

            rank += 1

    pd.DataFrame(
        rows
    ).to_csv(
        FEATURE_LIST_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved feature list:\n"
        f"  {FEATURE_LIST_OUTPUT_PATH}"
    )


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    print()
    print(
        f"Saved model:\n"
        f"  {MODEL_OUTPUT_PATH}"
    )


# ============================================================================
# SAVE TRAINING SUMMARY
# ============================================================================

def save_training_summary(
    train,
    validation,
    test,
    validation_metrics,
    test_metrics,
):

    summary = pd.DataFrame(
        [
            {
                "model":
                    "random_forest_model_3",

                "feature_count":
                    len(MODEL_3_FEATURES),

                "removed_trend_features":
                    len(REMOVED_TREND_FEATURES),

                "removed_sos_features":
                    len(REMOVED_SOS_FEATURES),

                "train_rows":
                    len(train),

                "validation_rows":
                    len(validation),

                "test_rows":
                    len(test),

                "n_estimators":
                    N_ESTIMATORS,

                "max_depth":
                    MAX_DEPTH,

                "min_samples_split":
                    MIN_SAMPLES_SPLIT,

                "min_samples_leaf":
                    MIN_SAMPLES_LEAF,

                "max_features":
                    MAX_FEATURES,

                "random_state":
                    RANDOM_STATE,

                "validation_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "validation_log_loss":
                    validation_metrics[
                        "log_loss"
                    ],

                "validation_brier_score":
                    validation_metrics[
                        "brier_score"
                    ],

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "test_auc":
                    test_metrics[
                        "roc_auc"
                    ],

                "test_log_loss":
                    test_metrics[
                        "log_loss"
                    ],

                "test_brier_score":
                    test_metrics[
                        "brier_score"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],
            }
        ]
    )

    summary.to_csv(
        TRAINING_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved training summary:\n"
        f"  {TRAINING_SUMMARY_OUTPUT_PATH}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print(
        "RANDOM FOREST MODEL 3 — "
        "TREND/SOS A/B EXPERIMENT"
    )
    print("=" * 80)

    print()
    print(
        "Project root:"
    )

    print(
        f"  {PROJECT_ROOT}"
    )

    print()
    print(
        "Model directory:"
    )

    print(
        f"  {MODEL_DIR}"
    )

    # ========================================================================
    # EXPERIMENT DESCRIPTION
    # ========================================================================

    print_section(
        "EXPERIMENT DESIGN"
    )

    print(
        "Model 2:"
    )

    print(
        "  Compact base features + Trend + Prior SOS"
    )

    print(
        "  Feature count: 44"
    )

    print()
    print(
        "Model 3:"
    )

    print(
        "  Compact base features only"
    )

    print(
        "  Feature count: 28"
    )

    print()
    print(
        "Removed from Model 2:"
    )

    print(
        f"  Trend features:     "
        f"{len(REMOVED_TREND_FEATURES)}"
    )

    print(
        f"  Prior SOS features: "
        f"{len(REMOVED_SOS_FEATURES)}"
    )

    print()
    print(
        "The Random Forest configuration remains unchanged."
    )

    # ========================================================================
    # LOAD MODEL INPUTS
    # ========================================================================

    print_section(
        "LOADING MODEL INPUTS"
    )

    train = load_split(
        TRAIN_PATH,
        "Training",
    )

    validation = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    test = load_split(
        TEST_PATH,
        "Test",
    )

    # ========================================================================
    # VALIDATE SPLITS
    # ========================================================================

    validate_splits(
        train,
        validation,
        test,
    )

    # ========================================================================
    # VALIDATE FEATURE SET
    # ========================================================================

    validate_features(
        train,
        validation,
        test,
    )

    # ========================================================================
    # PREPARE X / Y
    # ========================================================================

    print_section(
        "PREPARING MODEL 3 DATA"
    )

    X_train = prepare_features(
        train
    )

    X_validation = prepare_features(
        validation
    )

    X_test = prepare_features(
        test
    )

    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    y_validation = validation[
        TARGET_COLUMN
    ].astype(int)

    y_test = test[
        TARGET_COLUMN
    ].astype(int)

    print(
        f"Training:   "
        f"{X_train.shape}"
    )

    print(
        f"Validation: "
        f"{X_validation.shape}"
    )

    print(
        f"Test:       "
        f"{X_test.shape}"
    )

    print()
    print(
        "Training target distribution:"
    )

    print(
        y_train
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Training home win rate: "
        f"{y_train.mean():.4f}"
    )

    # ========================================================================
    # MISSINGNESS
    # ========================================================================

    print_section(
        "FEATURE MISSINGNESS"
    )

    missingness = (
        X_train
        .isna()
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    missing_features = (
        missingness[
            missingness > 0
        ]
    )

    if len(missing_features) == 0:

        print(
            "No missing predictor values."
        )

    else:

        print(
            "Training missingness:"
        )

        for feature, percentage in (
            missing_features.items()
        ):

            print(
                f"  {feature:<55}"
                f"{percentage:>7.2f}%"
            )

    print()
    print(
        f"Features with missing values: "
        f"{len(missing_features)}"
    )

    # ========================================================================
    # CREATE MODEL
    # ========================================================================

    print_section(
        "CREATING RANDOM FOREST"
    )

    print(
        f"Trees:             {N_ESTIMATORS}"
    )

    print(
        f"Max depth:         {MAX_DEPTH}"
    )

    print(
        f"Min samples split: {MIN_SAMPLES_SPLIT}"
    )

    print(
        f"Min samples leaf:  {MIN_SAMPLES_LEAF}"
    )

    print(
        f"Max features:      {MAX_FEATURES}"
    )

    print(
        f"Class weight:      {CLASS_WEIGHT}"
    )

    print(
        f"Random state:      {RANDOM_STATE}"
    )

    model = create_model()

    # ========================================================================
    # TRAIN
    # ========================================================================

    print_section(
        "TRAINING MODEL 3"
    )

    print(
        "Fitting Random Forest on "
        "2015-2022 training data..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training completed."
    )

    # ========================================================================
    # PREDICTIONS
    # ========================================================================

    print_section(
        "GENERATING PREDICTIONS"
    )

    validation_probability = (
        model
        .predict_proba(
            X_validation
        )[:, 1]
    )

    test_probability = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Validation predictions generated."
    )

    print(
        "Test predictions generated."
    )

    # ========================================================================
    # EVALUATE
    # ========================================================================

    validation_metrics = print_metrics(
        "Validation",
        y_validation,
        validation_probability,
    )

    test_metrics = print_metrics(
        "Test",
        y_test,
        test_probability,
    )

    # ========================================================================
    # PREDICTION DISTRIBUTION
    # ========================================================================

    print_section(
        "PREDICTION DISTRIBUTION"
    )

    for name, probability in [
        (
            "Validation",
            validation_probability,
        ),
        (
            "Test",
            test_probability,
        ),
    ]:

        print()
        print(name)

        print(
            f"  Mean:   {np.mean(probability):.6f}"
        )

        print(
            f"  Std:    {np.std(probability):.6f}"
        )

        print(
            f"  P01:    {np.percentile(probability, 1):.6f}"
        )

        print(
            f"  P05:    {np.percentile(probability, 5):.6f}"
        )

        print(
            f"  P10:    {np.percentile(probability, 10):.6f}"
        )

        print(
            f"  P25:    {np.percentile(probability, 25):.6f}"
        )

        print(
            f"  P50:    {np.percentile(probability, 50):.6f}"
        )

        print(
            f"  P75:    {np.percentile(probability, 75):.6f}"
        )

        print(
            f"  P90:    {np.percentile(probability, 90):.6f}"
        )

        print(
            f"  P95:    {np.percentile(probability, 95):.6f}"
        )

        print(
            f"  P99:    {np.percentile(probability, 99):.6f}"
        )

        print(
            f"  Min:    {np.min(probability):.6f}"
        )

        print(
            f"  Max:    {np.max(probability):.6f}"
        )

    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_predictions = (
        create_prediction_dataframe(
            validation,
            validation_probability,
            "validation",
        )
    )

    test_predictions = (
        create_prediction_dataframe(
            test,
            test_probability,
            "test",
        )
    )

    validation_predictions.to_csv(
        VALIDATION_PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    test_predictions.to_csv(
        TEST_PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved validation predictions:\n"
        f"  {VALIDATION_PREDICTIONS_OUTPUT_PATH}"
    )

    print(
        f"Saved test predictions:\n"
        f"  {TEST_PREDICTIONS_OUTPUT_PATH}"
    )

    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================

    save_model(
        model
    )

    save_feature_list()

    save_training_summary(
        train,
        validation,
        test,
        validation_metrics,
        test_metrics,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_section(
        "RANDOM FOREST MODEL 3 — TRAINING COMPLETE"
    )

    print()
    print(
        "A/B EXPERIMENT:"
    )

    print(
        "  Model 2: 44 predictors"
    )

    print(
        "  Model 3: 28 predictors"
    )

    print(
        "  Removed: 16 Trend/SOS predictors"
    )

    print()
    print(
        "Validation:"
    )

    print(
        f"  Log Loss:    "
        f"{validation_metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score: "
        f"{validation_metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC:     "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy:    "
        f"{validation_metrics['accuracy']:.4%}"
    )

    print()
    print(
        "Test:"
    )

    print(
        f"  Log Loss:    "
        f"{test_metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score: "
        f"{test_metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC:     "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy:    "
        f"{test_metrics['accuracy']:.4%}"
    )

    print()
    print(
        "Model 3 training complete."
    )

    print()
    print(
        "The test set should not be used to modify the feature set."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the Model 3 diagnostic/stability audit and compare "
        "Model 2 vs Model 3."
    )


if __name__ == "__main__":
    main()