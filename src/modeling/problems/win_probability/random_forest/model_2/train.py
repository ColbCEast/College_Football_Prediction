"""
Train Random Forest Model 2
============================

Purpose
-------
Train a compact Random Forest model using a substantially smaller feature
set than Random Forest Model 1.

Model 1:
    ~310 predictors

Model 2:
    44 deliberately selected predictors

Model 2 adds:
    - Explicit trend features
    - Prior strength-of-schedule features

The engineered trend/SOS features already exist in the enhanced logistic
feature files. They are merged onto the existing generic win-probability
model-input splits by gameId.

This script intentionally does NOT modify the shared model-input files.

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
- The exact Model 2 feature list is saved.
- Raw Random Forest probabilities are preserved for later calibration.
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
# Keep these consistent with Model 1 so this experiment primarily evaluates
# feature reduction and engineered features.
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
#           model_2/
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

ENHANCED_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "logistic_regression"
    / "enhanced"
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
    / "model_2"
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
# MODEL 2 FEATURE DEFINITIONS
# ============================================================================

"""
The feature set is intentionally compact.

There are 60 predictors:

    Core strength:          8
    Recent form:            8
    Trend:                 12
    Prior SOS:              4
    Offensive efficiency:   8
    Defensive efficiency:  4

Total:                     44

NOTE:
The current base feature set contains 44 predictors and the engineered
features add 16 more, resulting in 60 total predictors.

The 16 engineered predictors are:
    - 12 trend features
    - 4 prior SOS features
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
# Explicit trend features
# ----------------------------------------------------------------------------

TREND_FEATURES = [
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


# ----------------------------------------------------------------------------
# Prior strength of schedule
# ----------------------------------------------------------------------------

SOS_FEATURES = [
    "priorSOSWinPct_home",
    "priorSOSWinPct_away",

    "priorSOSPointDiff_home",
    "priorSOSPointDiff_away",
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
# Final feature list
# ----------------------------------------------------------------------------

MODEL_2_FEATURES = (
    CORE_STRENGTH_FEATURES
    + RECENT_FORM_FEATURES
    + TREND_FEATURES
    + SOS_FEATURES
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
    """Load a generic win-probability modeling split."""

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


def load_enhanced_features(year):
    """
    Load the enhanced feature file for a given season.
    """

    path = (
        ENHANCED_FEATURE_DIR
        / f"logistic_features_{year}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Enhanced feature file does not exist for {year}:\n"
            f"  {path}"
        )

    df = pd.read_csv(path)

    if GAME_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Enhanced feature file for {year} does not contain "
            f"'{GAME_ID_COLUMN}':\n"
            f"  {path}"
        )

    return df


# ============================================================================
# MERGE ENHANCED FEATURES
# ============================================================================

def add_enhanced_features(df, split_name):
    """
    Merge the enhanced trend/SOS features onto the generic model-input
    dataframe.

    Only the engineered features needed by Model 2 are merged.

    The merge is performed using gameId.
    """

    print_section(
        f"ADDING ENHANCED FEATURES — {split_name.upper()}"
    )

    seasons = sorted(
        df["season"].unique()
    )

    print(
        f"Seasons found: {seasons}"
    )

    enhanced_frames = []

    for year in seasons:

        enhanced = load_enhanced_features(
            int(year)
        )

        # Keep only game ID and engineered features.
        required_columns = [
            GAME_ID_COLUMN
        ] + TREND_FEATURES + SOS_FEATURES

        missing = [
            column
            for column in required_columns
            if column not in enhanced.columns
        ]

        if missing:
            raise ValueError(
                f"Enhanced feature file for {year} is missing "
                f"required Model 2 engineered features:\n"
                + "\n".join(
                    f"  {column}"
                    for column in missing
                )
            )

        enhanced = enhanced[
            required_columns
        ].copy()

        if enhanced[GAME_ID_COLUMN].duplicated().any():
            duplicate_count = (
                enhanced[GAME_ID_COLUMN]
                .duplicated()
                .sum()
            )

            raise ValueError(
                f"Enhanced feature file for {year} contains "
                f"{duplicate_count} duplicate game IDs."
            )

        enhanced_frames.append(
            enhanced
        )

        print(
            f"  {year}: "
            f"{len(enhanced):,} enhanced rows loaded"
        )

    enhanced_all = pd.concat(
        enhanced_frames,
        ignore_index=True,
    )

    print()
    print(
        f"Total enhanced rows: "
        f"{len(enhanced_all):,}"
    )

    # Ensure no duplicate game IDs across seasons.
    if enhanced_all[GAME_ID_COLUMN].duplicated().any():

        duplicates = (
            enhanced_all[GAME_ID_COLUMN]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Enhanced feature dataset contains "
            f"{duplicates} duplicate game IDs."
        )

    # ------------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------------

    before_rows = len(df)

    merged = df.merge(
        enhanced_all,
        on=GAME_ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = (
        merged["_merge"] != "both"
    ).sum()

    if unmatched > 0:

        missing_ids = (
            merged.loc[
                merged["_merge"] != "both",
                GAME_ID_COLUMN
            ]
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"{unmatched} games in the {split_name} split "
            f"could not be matched to enhanced features.\n"
            f"Example game IDs: {missing_ids}"
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    if len(merged) != before_rows:
        raise ValueError(
            f"Row count changed during enhanced feature merge: "
            f"{before_rows} → {len(merged)}"
        )

    print()
    print(
        "Enhanced features merged successfully."
    )

    print(
        f"  Rows: "
        f"{len(merged):,}"
    )

    print(
        f"  Added features: "
        f"{len(TREND_FEATURES) + len(SOS_FEATURES)}"
    )

    return merged


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
        "VALIDATING MODEL 2 FEATURES"
    )

    print(
        "Model 1 predictors: approximately 310"
    )

    print(
        f"Model 2 predictors: "
        f"{len(MODEL_2_FEATURES)}"
    )

    if len(MODEL_2_FEATURES) != len(
        set(MODEL_2_FEATURES)
    ):
        duplicates = sorted(
            {
                feature
                for feature in MODEL_2_FEATURES
                if MODEL_2_FEATURES.count(feature) > 1
            }
        )

        raise ValueError(
            "Duplicate Model 2 features:\n"
            + "\n".join(
                f"  {feature}"
                for feature in duplicates
            )
        )

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = [
            feature
            for feature in MODEL_2_FEATURES
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing Model 2 features:\n"
                + "\n".join(
                    f"  {feature}"
                    for feature in missing
                )
            )

    print()
    print(
        "Feature validation passed."
    )

    print()
    print(
        "Feature groups:"
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
        f"  Trend:                "
        f"{len(TREND_FEATURES)}"
    )

    print(
        f"  Prior SOS:            "
        f"{len(SOS_FEATURES)}"
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
        f"{len(MODEL_2_FEATURES)}"
    )


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def prepare_features(df):

    X = df[
        MODEL_2_FEATURES
    ].copy()

    for column in MODEL_2_FEATURES:

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
            "Trend",
            TREND_FEATURES,
        ),
        (
            "Prior SOS",
            SOS_FEATURES,
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
                    "random_forest_model_2",

                "feature_count":
                    len(MODEL_2_FEATURES),

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
        "RANDOM FOREST MODEL 2 — "
        "COMPACT FEATURE EXPERIMENT"
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
    # LOAD BASE MODEL INPUTS
    # ========================================================================

    print_section(
        "LOADING BASE MODEL INPUTS"
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
    # VALIDATE BASE SPLITS
    # ========================================================================

    validate_splits(
        train,
        validation,
        test,
    )

    # ========================================================================
    # ADD ENHANCED FEATURES
    # ========================================================================

    train = add_enhanced_features(
        train,
        "Training",
    )

    validation = add_enhanced_features(
        validation,
        "Validation",
    )

    test = add_enhanced_features(
        test,
        "Test",
    )

    # ========================================================================
    # VALIDATE FINAL MODEL 2 FEATURE SET
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
        "PREPARING MODEL 2 DATA"
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
        "TRAINING MODEL 2"
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
        "RANDOM FOREST MODEL 2 — TRAINING COMPLETE"
    )

    print()
    print(
        f"Model 2 feature count: "
        f"{len(MODEL_2_FEATURES)}"
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
        "Model 2 training complete."
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
        "Run the Model 2 diagnostic/stability audit."
    )


if __name__ == "__main__":
    main()