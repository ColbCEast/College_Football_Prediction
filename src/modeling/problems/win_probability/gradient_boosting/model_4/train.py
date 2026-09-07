"""
Train Gradient Boosting Model 4
================================

Purpose
-------
Train a Gradient Boosting win-probability model using the same compact
28-feature feature space as Gradient Boosting Model 3, but select
hyperparameters using expanding-window temporal cross-validation.

Model progression
-----------------
Model 2:
    44 compact features
    Base + Trend + Prior SOS

Model 3:
    28 compact features
    Base features only
    Fixed Gradient Boosting configuration

Model 4:
    28 compact features
    Base features only
    Expanding-window temporal CV for hyperparameter selection

The purpose of Model 4 is to determine whether careful, temporally
appropriate hyperparameter tuning improves upon the fixed Gradient
Boosting configuration used in Model 3.

Feature set
-----------
The exact 28-feature Model 3 feature set is retained.

Core strength:
    8

Recent form:
    8

Offensive efficiency:
    8

Defensive efficiency:
    4

Total:
    28

Temporal modeling structure
---------------------------
Final train/validation/test split:

Training:
    2015-2022

Validation:
    2023-2024

Test:
    2025

Hyperparameter tuning occurs ONLY within the 2015-2022 training period.

Expanding-window temporal CV:

Fold 1:
    Train: 2015-2018
    Validate: 2019

Fold 2:
    Train: 2015-2019
    Validate: 2020

Fold 3:
    Train: 2015-2020
    Validate: 2021

Fold 4:
    Train: 2015-2021
    Validate: 2022

The 2023-2024 validation set is not used for hyperparameter selection.

The 2025 test set is used only for final evaluation.

Primary tuning metric
---------------------
Log Loss

Secondary metrics:
    Brier Score
    ROC AUC
    Accuracy

Hyperparameters tuned
---------------------
    n_estimators
    learning_rate
    max_depth
    min_samples_leaf
    subsample

The search is intentionally compact so that the experiment remains
interpretable and computationally manageable.

Missing values
--------------
Median imputation is performed inside the sklearn Pipeline.

Because the imputer is part of the pipeline, the median is fitted
using only the training portion of each temporal CV fold.

Artifacts saved
----------------
model.joblib
feature_list.csv
training_summary.csv
cv_results.csv
best_parameters.csv
validation_predictions.csv
test_predictions.csv

Important
---------
- The 2023-2024 validation set is not used during hyperparameter tuning.
- The 2025 test set is not used during hyperparameter tuning.
- No feature selection occurs in Model 4.
- The feature set is identical to Model 3.
- Raw Gradient Boosting probabilities are preserved for later calibration.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
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


# ============================================================================
# TEMPORAL CROSS-VALIDATION CONFIGURATION
# ============================================================================

CV_FOLDS = [
    {
        "fold": 1,
        "train_years": list(range(2015, 2019)),
        "validation_years": [2019],
    },
    {
        "fold": 2,
        "train_years": list(range(2015, 2020)),
        "validation_years": [2020],
    },
    {
        "fold": 3,
        "train_years": list(range(2015, 2021)),
        "validation_years": [2021],
    },
    {
        "fold": 4,
        "train_years": list(range(2015, 2022)),
        "validation_years": [2022],
    },
]


# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================
#
# The Model 3 baseline is:
#
#     n_estimators = 500
#     learning_rate = 0.05
#     max_depth = 3
#     min_samples_leaf = 5
#     subsample = 1.0
#
# The search space intentionally surrounds that baseline rather than
# exploring an unnecessarily large space.
# ============================================================================

PARAM_GRID = {
    "n_estimators": [
        200,
        350,
        500,
        750,
    ],
    "learning_rate": [
        0.03,
        0.05,
        0.08,
        0.10,
    ],
    "max_depth": [
        2,
        3,
        4,
    ],
    "min_samples_leaf": [
        3,
        5,
        10,
    ],
    "subsample": [
        0.75,
        1.00,
    ],
}


# ============================================================================
# PROJECT PATHS
# ============================================================================

# train.py is located at:
#
# src/
#   modeling/
#     problems/
#       win_probability/
#         gradient_boosting/
#           model_4/
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


TRAIN_PATH = MODEL_INPUT_DIR / "train.csv"
VALIDATION_PATH = MODEL_INPUT_DIR / "validation.csv"
TEST_PATH = MODEL_INPUT_DIR / "test.csv"


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_4"
)


MODEL_OUTPUT_PATH = MODEL_DIR / "model.joblib"

FEATURE_LIST_OUTPUT_PATH = (
    MODEL_DIR / "feature_list.csv"
)

TRAINING_SUMMARY_OUTPUT_PATH = (
    MODEL_DIR / "training_summary.csv"
)

CV_RESULTS_OUTPUT_PATH = (
    MODEL_DIR / "cv_results.csv"
)

BEST_PARAMETERS_OUTPUT_PATH = (
    MODEL_DIR / "best_parameters.csv"
)

VALIDATION_PREDICTIONS_OUTPUT_PATH = (
    MODEL_DIR / "validation_predictions.csv"
)

TEST_PREDICTIONS_OUTPUT_PATH = (
    MODEL_DIR / "test_predictions.csv"
)


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

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


MODEL_4_FEATURES = (
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

    if not path.exists():
        raise FileNotFoundError(
            f"{name} split does not exist:\n"
            f"  {path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name:<12}: "
        f"{len(df):,} rows × "
        f"{len(df.columns):,} columns"
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

        missing = required - set(df.columns)

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
    # Check game ID overlap
    # ------------------------------------------------------------------------

    train_ids = set(train[GAME_ID_COLUMN])
    validation_ids = set(validation[GAME_ID_COLUMN])
    test_ids = set(test[GAME_ID_COLUMN])

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
    # Validate final temporal split
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
        "VALIDATING MODEL 4 FEATURES"
    )

    print(
        "Model 4 uses the same compact feature set as Model 3."
    )

    print(
        f"Expected predictors: "
        f"{len(MODEL_4_FEATURES)}"
    )

    if len(MODEL_4_FEATURES) != 28:
        raise ValueError(
            "Model 4 feature count is not 28."
        )

    if len(MODEL_4_FEATURES) != len(
        set(MODEL_4_FEATURES)
    ):
        duplicates = sorted(
            {
                feature
                for feature in MODEL_4_FEATURES
                if MODEL_4_FEATURES.count(feature) > 1
            }
        )

        raise ValueError(
            "Duplicate Model 4 features:\n"
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
            for feature in MODEL_4_FEATURES
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} is missing Model 4 features:\n"
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
        f"{len(MODEL_4_FEATURES)}"
    )


# ============================================================================
# TEMPORAL CV VALIDATION
# ============================================================================

def validate_cv_structure(train):

    print_section(
        "VALIDATING TEMPORAL CROSS-VALIDATION"
    )

    available_years = sorted(
        train["season"].unique()
    )

    for fold in CV_FOLDS:

        train_years = fold["train_years"]
        validation_years = fold["validation_years"]

        print()
        print(
            f"Fold {fold['fold']}:"
        )

        print(
            f"  Train:      {train_years}"
        )

        print(
            f"  Validation: {validation_years}"
        )

        if not set(train_years).issubset(
            set(available_years)
        ):
            raise ValueError(
                f"Fold {fold['fold']} training years "
                "are not present in training data."
            )

        if not set(validation_years).issubset(
            set(available_years)
        ):
            raise ValueError(
                f"Fold {fold['fold']} validation years "
                "are not present in training data."
            )

        if max(train_years) >= min(
            validation_years
        ):
            raise ValueError(
                f"Fold {fold['fold']} is not temporally ordered."
            )

    print()
    print(
        "Temporal CV structure validated."
    )


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def prepare_features(df):

    X = df[
        MODEL_4_FEATURES
    ].copy()

    for column in MODEL_4_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


# ============================================================================
# CREATE MODEL
# ============================================================================

def create_model(
    n_estimators,
    learning_rate,
    max_depth,
    min_samples_leaf,
    subsample,
):

    imputer = SimpleImputer(
        strategy="median"
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                imputer,
            ),
            (
                "classifier",
                gradient_boosting,
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
        "log_loss": log_loss(
            y_true,
            probability,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probability,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
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
    }


# ============================================================================
# HYPERPARAMETER COMBINATIONS
# ============================================================================

def generate_parameter_combinations():

    combinations = []

    for n_estimators in PARAM_GRID[
        "n_estimators"
    ]:

        for learning_rate in PARAM_GRID[
            "learning_rate"
        ]:

            for max_depth in PARAM_GRID[
                "max_depth"
            ]:

                for min_samples_leaf in PARAM_GRID[
                    "min_samples_leaf"
                ]:

                    for subsample in PARAM_GRID[
                        "subsample"
                    ]:

                        combinations.append(
                            {
                                "n_estimators":
                                    n_estimators,

                                "learning_rate":
                                    learning_rate,

                                "max_depth":
                                    max_depth,

                                "min_samples_leaf":
                                    min_samples_leaf,

                                "subsample":
                                    subsample,
                            }
                        )

    return combinations


# ============================================================================
# TEMPORAL CROSS-VALIDATION
# ============================================================================

def run_temporal_cv(train):

    print_section(
        "RUNNING TEMPORAL HYPERPARAMETER TUNING"
    )

    parameter_combinations = (
        generate_parameter_combinations()
    )

    print(
        f"Hyperparameter combinations: "
        f"{len(parameter_combinations)}"
    )

    print(
        f"Temporal CV folds: "
        f"{len(CV_FOLDS)}"
    )

    print(
        f"Total model fits: "
        f"{len(parameter_combinations) * len(CV_FOLDS)}"
    )

    print()
    print(
        "Primary tuning metric: Log Loss"
    )

    print(
        "Secondary metrics: Brier Score, ROC AUC"
    )

    results = []

    total_fits = (
        len(parameter_combinations)
        * len(CV_FOLDS)
    )

    completed_fits = 0

    # ------------------------------------------------------------------------
    # Loop through hyperparameter combinations
    # ------------------------------------------------------------------------

    for parameter_index, parameters in enumerate(
        parameter_combinations,
        start=1,
    ):

        fold_metrics = []

        for fold in CV_FOLDS:

            train_mask = train[
                "season"
            ].isin(
                fold["train_years"]
            )

            validation_mask = train[
                "season"
            ].isin(
                fold["validation_years"]
            )

            fold_train = train.loc[
                train_mask
            ]

            fold_validation = train.loc[
                validation_mask
            ]

            X_fold_train = prepare_features(
                fold_train
            )

            X_fold_validation = prepare_features(
                fold_validation
            )

            y_fold_train = fold_train[
                TARGET_COLUMN
            ].astype(int)

            y_fold_validation = (
                fold_validation[
                    TARGET_COLUMN
                ].astype(int)
            )

            model = create_model(
                **parameters
            )

            model.fit(
                X_fold_train,
                y_fold_train,
            )

            probability = (
                model
                .predict_proba(
                    X_fold_validation
                )[:, 1]
            )

            metrics = calculate_metrics(
                y_fold_validation,
                probability,
            )

            fold_metrics.append(
                {
                    "fold": fold["fold"],
                    "train_year_start":
                        min(
                            fold["train_years"]
                        ),
                    "train_year_end":
                        max(
                            fold["train_years"]
                        ),
                    "validation_year_start":
                        min(
                            fold[
                                "validation_years"
                            ]
                        ),
                    "validation_year_end":
                        max(
                            fold[
                                "validation_years"
                            ]
                        ),
                    "train_rows":
                        len(fold_train),
                    "validation_rows":
                        len(fold_validation),
                    **metrics,
                }
            )

            completed_fits += 1

        # --------------------------------------------------------------------
        # Aggregate across folds
        # --------------------------------------------------------------------

        fold_df = pd.DataFrame(
            fold_metrics
        )

        aggregate = {
            **parameters,

            "mean_log_loss":
                fold_df[
                    "log_loss"
                ].mean(),

            "std_log_loss":
                fold_df[
                    "log_loss"
                ].std(ddof=0),

            "mean_brier_score":
                fold_df[
                    "brier_score"
                ].mean(),

            "std_brier_score":
                fold_df[
                    "brier_score"
                ].std(ddof=0),

            "mean_roc_auc":
                fold_df[
                    "roc_auc"
                ].mean(),

            "std_roc_auc":
                fold_df[
                    "roc_auc"
                ].std(ddof=0),

            "mean_accuracy":
                fold_df[
                    "accuracy"
                ].mean(),

            "std_accuracy":
                fold_df[
                    "accuracy"
                ].std(ddof=0),

            "mean_balanced_accuracy":
                fold_df[
                    "balanced_accuracy"
                ].mean(),

            "mean_precision":
                fold_df[
                    "precision"
                ].mean(),

            "mean_recall":
                fold_df[
                    "recall"
                ].mean(),
        }

        # Add individual fold metrics
        for fold_number in range(
            1,
            len(CV_FOLDS) + 1,
        ):

            row = fold_df[
                fold_df["fold"] == fold_number
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            aggregate[
                f"fold_{fold_number}_log_loss"
            ] = row["log_loss"]

            aggregate[
                f"fold_{fold_number}_brier_score"
            ] = row["brier_score"]

            aggregate[
                f"fold_{fold_number}_roc_auc"
            ] = row["roc_auc"]

            aggregate[
                f"fold_{fold_number}_accuracy"
            ] = row["accuracy"]

        results.append(
            aggregate
        )

        # --------------------------------------------------------------------
        # Progress reporting
        # --------------------------------------------------------------------

        if (
            parameter_index == 1
            or parameter_index % 10 == 0
            or parameter_index == len(
                parameter_combinations
            )
        ):

            print(
                f"Completed combination "
                f"{parameter_index:,}/"
                f"{len(parameter_combinations):,} "
                f"| Fits: "
                f"{completed_fits:,}/"
                f"{total_fits:,} "
                f"| Mean Log Loss: "
                f"{aggregate['mean_log_loss']:.6f}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by=[
            "mean_log_loss",
            "mean_brier_score",
            "mean_roc_auc",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    results_df.insert(
        0,
        "rank",
        range(
            1,
            len(results_df) + 1,
        ),
    )

    print()
    print(
        "Temporal hyperparameter tuning completed."
    )

    return results_df


# ============================================================================
# DISPLAY TOP CV RESULTS
# ============================================================================

def print_top_cv_results(results_df):

    print_section(
        "TOP TEMPORAL CV CONFIGURATIONS"
    )

    display_columns = [
        "rank",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "mean_log_loss",
        "std_log_loss",
        "mean_brier_score",
        "mean_roc_auc",
        "mean_accuracy",
    ]

    print(
        results_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================================
# SELECT BEST PARAMETERS
# ============================================================================

def select_best_parameters(results_df):

    best = results_df.iloc[0]

    parameters = {
        "n_estimators":
            int(best["n_estimators"]),

        "learning_rate":
            float(best["learning_rate"]),

        "max_depth":
            int(best["max_depth"]),

        "min_samples_leaf":
            int(best["min_samples_leaf"]),

        "subsample":
            float(best["subsample"]),
    }

    print_section(
        "SELECTED MODEL 4 HYPERPARAMETERS"
    )

    for name, value in parameters.items():

        print(
            f"{name:<20}: {value}"
        )

    print()
    print(
        f"Mean CV Log Loss: "
        f"{best['mean_log_loss']:.6f}"
    )

    print(
        f"Mean CV Brier Score: "
        f"{best['mean_brier_score']:.6f}"
    )

    print(
        f"Mean CV ROC AUC: "
        f"{best['mean_roc_auc']:.6f}"
    )

    return parameters, best


# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================

def train_final_model(
    train,
    parameters,
):

    print_section(
        "TRAINING FINAL MODEL 4"
    )

    X_train = prepare_features(
        train
    )

    y_train = train[
        TARGET_COLUMN
    ].astype(int)

    print(
        "Training final model using "
        "all 2015-2022 training data."
    )

    print()
    print(
        "Selected hyperparameters:"
    )

    for name, value in parameters.items():

        print(
            f"  {name:<18}: {value}"
        )

    model = create_model(
        **parameters
    )

    model.fit(
        X_train,
        y_train,
    )

    print()
    print(
        "Final Model 4 training completed."
    )

    return model


# ============================================================================
# PREDICTION DATAFRAME
# ============================================================================

def create_prediction_dataframe(
    df,
    probability,
    split_name,
):

    return pd.DataFrame(
        {
            "gameId":
                df[
                    GAME_ID_COLUMN
                ].values,

            "season":
                df[
                    "season"
                ].values,

            "win_home_actual":
                df[
                    TARGET_COLUMN
                ].astype(int).values,

            "win_home_probability":
                probability,

            "win_home_prediction":
                (
                    probability >= 0.50
                ).astype(int),

            "split":
                split_name,
        }
    )


# ============================================================================
# PRINT FINAL PERFORMANCE
# ============================================================================

def print_performance(
    name,
    y_true,
    probability,
):

    metrics = calculate_metrics(
        y_true,
        probability,
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
        f"ROC AUC:              "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Accuracy:             "
        f"{metrics['accuracy']:.4%}"
    )

    print(
        f"Balanced Accuracy:    "
        f"{metrics['balanced_accuracy']:.4%}"
    )

    print(
        f"Precision:            "
        f"{metrics['precision']:.4%}"
    )

    print(
        f"Recall:               "
        f"{metrics['recall']:.4%}"
    )

    return metrics


# ============================================================================
# PREDICTION DISTRIBUTION
# ============================================================================

def print_prediction_distribution(
    name,
    probability,
):

    print()
    print(
        f"{name}:"
    )

    print(
        f"  Mean:   "
        f"{np.mean(probability):.6f}"
    )

    print(
        f"  Std:    "
        f"{np.std(probability):.6f}"
    )

    print(
        f"  P01:    "
        f"{np.percentile(probability, 1):.6f}"
    )

    print(
        f"  P05:    "
        f"{np.percentile(probability, 5):.6f}"
    )

    print(
        f"  P10:    "
        f"{np.percentile(probability, 10):.6f}"
    )

    print(
        f"  P25:    "
        f"{np.percentile(probability, 25):.6f}"
    )

    print(
        f"  P50:    "
        f"{np.percentile(probability, 50):.6f}"
    )

    print(
        f"  P75:    "
        f"{np.percentile(probability, 75):.6f}"
    )

    print(
        f"  P90:    "
        f"{np.percentile(probability, 90):.6f}"
    )

    print(
        f"  P95:    "
        f"{np.percentile(probability, 95):.6f}"
    )

    print(
        f"  P99:    "
        f"{np.percentile(probability, 99):.6f}"
    )

    print(
        f"  Min:    "
        f"{np.min(probability):.6f}"
    )

    print(
        f"  Max:    "
        f"{np.max(probability):.6f}"
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
# SAVE BEST PARAMETERS
# ============================================================================

def save_best_parameters(
    parameters,
    best_row,
):

    output = pd.DataFrame(
        [
            {
                **parameters,

                "mean_cv_log_loss":
                    best_row[
                        "mean_log_loss"
                    ],

                "std_cv_log_loss":
                    best_row[
                        "std_log_loss"
                    ],

                "mean_cv_brier_score":
                    best_row[
                        "mean_brier_score"
                    ],

                "mean_cv_roc_auc":
                    best_row[
                        "mean_roc_auc"
                    ],

                "mean_cv_accuracy":
                    best_row[
                        "mean_accuracy"
                    ],

                "random_state":
                    RANDOM_STATE,
            }
        ]
    )

    output.to_csv(
        BEST_PARAMETERS_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved best parameters:\n"
        f"  {BEST_PARAMETERS_OUTPUT_PATH}"
    )


# ============================================================================
# SAVE TRAINING SUMMARY
# ============================================================================

def save_training_summary(
    train,
    validation,
    test,
    parameters,
    best_row,
    validation_metrics,
    test_metrics,
):

    summary = pd.DataFrame(
        [
            {
                "model":
                    "gradient_boosting_model_4",

                "feature_count":
                    len(MODEL_4_FEATURES),

                "train_rows":
                    len(train),

                "validation_rows":
                    len(validation),

                "test_rows":
                    len(test),

                "cv_folds":
                    len(CV_FOLDS),

                "tuning_metric":
                    "log_loss",

                "n_estimators":
                    parameters[
                        "n_estimators"
                    ],

                "learning_rate":
                    parameters[
                        "learning_rate"
                    ],

                "max_depth":
                    parameters[
                        "max_depth"
                    ],

                "min_samples_leaf":
                    parameters[
                        "min_samples_leaf"
                    ],

                "subsample":
                    parameters[
                        "subsample"
                    ],

                "mean_cv_log_loss":
                    best_row[
                        "mean_log_loss"
                    ],

                "std_cv_log_loss":
                    best_row[
                        "std_log_loss"
                    ],

                "mean_cv_brier_score":
                    best_row[
                        "mean_brier_score"
                    ],

                "mean_cv_roc_auc":
                    best_row[
                        "mean_roc_auc"
                    ],

                "validation_log_loss":
                    validation_metrics[
                        "log_loss"
                    ],

                "validation_brier_score":
                    validation_metrics[
                        "brier_score"
                    ],

                "validation_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "test_log_loss":
                    test_metrics[
                        "log_loss"
                    ],

                "test_brier_score":
                    test_metrics[
                        "brier_score"
                    ],

                "test_auc":
                    test_metrics[
                        "roc_auc"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "random_state":
                    RANDOM_STATE,
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
        "GRADIENT BOOSTING MODEL 4 — "
        "TEMPORAL HYPERPARAMETER TUNING"
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
        "Model 3:"
    )

    print(
        "  28 compact features"
    )

    print(
        "  Fixed Gradient Boosting hyperparameters"
    )

    print()
    print(
        "Model 4:"
    )

    print(
        "  28 compact features"
    )

    print(
        "  Expanding-window temporal CV"
    )

    print(
        "  Hyperparameters selected using Log Loss"
    )

    print()
    print(
        "Final train/validation/test split:"
    )

    print(
        "  Training:   2015-2022"
    )

    print(
        "  Validation: 2023-2024"
    )

    print(
        "  Test:       2025"
    )

    print()
    print(
        "Temporal CV:"
    )

    for fold in CV_FOLDS:

        print(
            f"  Fold {fold['fold']}: "
            f"{min(fold['train_years'])}-"
            f"{max(fold['train_years'])}"
            f" → "
            f"{fold['validation_years']}"
        )

    # ========================================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # LOAD DATA
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
    # VALIDATE DATA
    # ========================================================================

    validate_splits(
        train,
        validation,
        test,
    )

    validate_features(
        train,
        validation,
        test,
    )

    validate_cv_structure(
        train
    )

    # ========================================================================
    # MISSINGNESS
    # ========================================================================

    print_section(
        "FEATURE MISSINGNESS"
    )

    X_train = prepare_features(
        train
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

    print(
        f"Total missing values: "
        f"{X_train.isna().sum().sum():,}"
    )

    # ========================================================================
    # TEMPORAL HYPERPARAMETER TUNING
    # ========================================================================

    cv_results = run_temporal_cv(
        train
    )

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    print_top_cv_results(
        cv_results
    )

    # ========================================================================
    # SAVE CV RESULTS
    # ========================================================================

    cv_results.to_csv(
        CV_RESULTS_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved CV results:\n"
        f"  {CV_RESULTS_OUTPUT_PATH}"
    )

    # ========================================================================
    # SELECT PARAMETERS
    # ========================================================================

    parameters, best_row = (
        select_best_parameters(
            cv_results
        )
    )

    save_best_parameters(
        parameters,
        best_row,
    )

    # ========================================================================
    # TRAIN FINAL MODEL
    # ========================================================================

    model = train_final_model(
        train,
        parameters,
    )

    # ========================================================================
    # GENERATE FINAL PREDICTIONS
    # ========================================================================

    print_section(
        "GENERATING FINAL PREDICTIONS"
    )

    X_validation = prepare_features(
        validation
    )

    X_test = prepare_features(
        test
    )

    y_validation = validation[
        TARGET_COLUMN
    ].astype(int)

    y_test = test[
        TARGET_COLUMN
    ].astype(int)

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
    # FINAL EVALUATION
    # ========================================================================

    validation_metrics = print_performance(
        "Validation",
        y_validation,
        validation_probability,
    )

    test_metrics = print_performance(
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

    print_prediction_distribution(
        "Validation",
        validation_probability,
    )

    print_prediction_distribution(
        "Test",
        test_probability,
    )

    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================

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
    # SAVE MODEL
    # ========================================================================

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    print()
    print(
        f"Saved model:\n"
        f"  {MODEL_OUTPUT_PATH}"
    )

    # ========================================================================
    # SAVE FEATURE LIST
    # ========================================================================

    save_feature_list()

    # ========================================================================
    # SAVE TRAINING SUMMARY
    # ========================================================================

    save_training_summary(
        train,
        validation,
        test,
        parameters,
        best_row,
        validation_metrics,
        test_metrics,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_section(
        "GRADIENT BOOSTING MODEL 4 — "
        "TRAINING COMPLETE"
    )

    print()
    print(
        "MODEL DEFINITION:"
    )

    print(
        "  Feature count: 28"
    )

    print(
        "  Feature set: Model 3 compact features"
    )

    print(
        "  Hyperparameter selection: "
        "Expanding-window temporal CV"
    )

    print(
        "  Primary tuning metric: Log Loss"
    )

    print()
    print(
        "SELECTED HYPERPARAMETERS:"
    )

    for name, value in parameters.items():

        print(
            f"  {name:<20}: {value}"
        )

    print()
    print(
        "TEMPORAL CV:"
    )

    print(
        f"  Mean Log Loss: "
        f"{best_row['mean_log_loss']:.6f}"
    )

    print(
        f"  Mean Brier Score: "
        f"{best_row['mean_brier_score']:.6f}"
    )

    print(
        f"  Mean ROC AUC: "
        f"{best_row['mean_roc_auc']:.6f}"
    )

    print()
    print(
        "VALIDATION:"
    )

    print(
        f"  Log Loss: "
        f"{validation_metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score: "
        f"{validation_metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC: "
        f"{validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{validation_metrics['accuracy']:.4%}"
    )

    print()
    print(
        "TEST:"
    )

    print(
        f"  Log Loss: "
        f"{test_metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score: "
        f"{test_metrics['brier_score']:.6f}"
    )

    print(
        f"  ROC AUC: "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{test_metrics['accuracy']:.4%}"
    )

    print()
    print(
        "Model 4 training complete."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the Model 4 diagnostic/stability audit."
    )


if __name__ == "__main__":
    main()