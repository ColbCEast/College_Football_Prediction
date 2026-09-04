"""
Random Forest Win Probability - Model 5

Purpose
-------
Focused Random Forest hyperparameter tuning using expanding-window
temporal cross-validation.

Model 5 is a controlled continuation of Model 4:
    Model 3 = compact 28-feature specification
    Model 4 = broad hyperparameter tuning with standard CV
    Model 5 = focused hyperparameter tuning with temporal CV

The 28-feature specification is held fixed.

Temporal CV
-----------
Fold 1: Train 2015-2018 -> Validate 2019
Fold 2: Train 2015-2019 -> Validate 2020
Fold 3: Train 2015-2020 -> Validate 2021
Fold 4: Train 2015-2021 -> Validate 2022

Official validation set:
    2023-2024

Official test set:
    2025

The official validation and test sets are never used during
hyperparameter selection.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET = "win_home"
GAME_ID = "gameId"
SEASON = "season"

N_ITER = 60

# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = Path(
    "data/processed/model_inputs/win_probability/train.csv"
)

VALIDATION_PATH = Path(
    "data/processed/model_inputs/win_probability/validation.csv"
)

TEST_PATH = Path(
    "data/processed/model_inputs/win_probability/test.csv"
)

OUTPUT_DIR = Path(
    "models/win_probability/random_forest/model_5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# FIXED MODEL 3 / MODEL 4 FEATURE SET
# ============================================================

FEATURE_GROUPS = {
    "core_strength": [
        "homePregameElo",
        "awayPregameElo",
        "winPctBefore_home",
        "winPctBefore_away",
        "pointDifferentialBefore_home",
        "pointDifferentialBefore_away",
        "pointDifferentialAvgBefore_home",
        "pointDifferentialAvgBefore_away",
    ],

    "recent_form": [
        "pointDifferentialAvgLast3_home",
        "pointDifferentialAvgLast3_away",
        "pointDifferentialAvgLast5_home",
        "pointDifferentialAvgLast5_away",
        "pointsForAvgLast5_home",
        "pointsForAvgLast5_away",
        "pointsAgainstAvgLast5_home",
        "pointsAgainstAvgLast5_away",
    ],

    "offensive_efficiency": [
        "home_pregame_offense_successRate",
        "away_pregame_offense_successRate",
        "home_pregame_offense_ppa",
        "away_pregame_offense_ppa",
        "yardsPerPassAttemptBefore_home",
        "yardsPerPassAttemptBefore_away",
        "yardsPerRushAttemptBefore_home",
        "yardsPerRushAttemptBefore_away",
    ],

    "defensive_efficiency": [
        "home_pregame_defense_successRate",
        "away_pregame_defense_successRate",
        "home_pregame_defense_ppa",
        "away_pregame_defense_ppa",
    ],
}


FEATURES = [
    feature
    for group in FEATURE_GROUPS.values()
    for feature in group
]


# ============================================================
# TEMPORAL CV FOLDS
# ============================================================

TEMPORAL_FOLDS = [
    {
        "fold": 1,
        "train_seasons": [2015, 2016, 2017, 2018],
        "validation_seasons": [2019],
    },
    {
        "fold": 2,
        "train_seasons": [2015, 2016, 2017, 2018, 2019],
        "validation_seasons": [2020],
    },
    {
        "fold": 3,
        "train_seasons": [
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
        ],
        "validation_seasons": [2021],
    },
    {
        "fold": 4,
        "train_seasons": [
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
            2021,
        ],
        "validation_seasons": [2022],
    },
]


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [
        500,
        750,
        1000,
    ],

    "model__max_depth": [
        None,
        8,
        12,
        16,
        20,
        30,
    ],

    "model__min_samples_split": [
        2,
        5,
        10,
        20,
        30,
    ],

    "model__min_samples_leaf": [
        5,
        10,
        15,
        20,
        25,
        30,
    ],

    "model__max_features": [
        0.25,
        0.50,
        0.75,
        1.0,
    ],

    "model__bootstrap": [
        True,
        False,
    ],
}


# ============================================================
# HELPERS
# ============================================================

def build_pipeline(params=None):
    """
    Build the Random Forest pipeline.

    Median imputation is fitted independently inside each
    training fold to prevent information leakage.
    """

    model_params = {
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": None,
    }

    if params:
        for key, value in params.items():
            if key.startswith("model__"):
                model_params[key.replace("model__", "")] = value

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(**model_params),
            ),
        ]
    )

    return pipeline


def calculate_metrics(y_true, probabilities):
    """
    Calculate probability and classification metrics.
    """

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "log_loss": log_loss(y_true, probabilities),
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
    }


def validate_columns(df, name):
    """
    Validate required columns and model features.
    """

    required_columns = (
        [GAME_ID, SEASON, TARGET]
        + FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def load_data():
    """
    Load train, validation, and test datasets.
    """

    print("\nLoading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    validate_columns(train_df, "Training data")
    validate_columns(
        validation_df,
        "Validation data",
    )
    validate_columns(
        test_df,
        "Test data",
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


def run_temporal_cv(
    train_df,
    params,
):
    """
    Evaluate one hyperparameter configuration using
    expanding-window temporal cross-validation.

    Each fold independently fits the complete pipeline,
    including median imputation.
    """

    fold_results = []

    for fold in TEMPORAL_FOLDS:

        train_seasons = fold["train_seasons"]
        validation_seasons = fold["validation_seasons"]

        fold_train = train_df[
            train_df[SEASON].isin(train_seasons)
        ].copy()

        fold_validation = train_df[
            train_df[SEASON].isin(validation_seasons)
        ].copy()

        X_train = fold_train[FEATURES]
        y_train = fold_train[TARGET]

        X_validation = fold_validation[FEATURES]
        y_validation = fold_validation[TARGET]

        pipeline = build_pipeline(params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pipeline.fit(
                X_train,
                y_train,
            )

        probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        metrics = calculate_metrics(
            y_validation,
            probabilities,
        )

        metrics.update(
            {
                "fold": fold["fold"],
                "train_start_season": min(
                    train_seasons
                ),
                "train_end_season": max(
                    train_seasons
                ),
                "validation_season": validation_seasons[0],
                "train_rows": len(fold_train),
                "validation_rows": len(
                    fold_validation
                ),
            }
        )

        fold_results.append(metrics)

    fold_df = pd.DataFrame(fold_results)

    return fold_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RANDOM FOREST WIN PROBABILITY - MODEL 5")
    print("FOCUSED HYPERPARAMETER TUNING WITH TEMPORAL CV")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    print("\nDataset shapes:")
    print(f"  Training:   {train_df.shape}")
    print(f"  Validation: {validation_df.shape}")
    print(f"  Test:       {test_df.shape}")

    print("\nFeature configuration:")
    print(f"  Feature count: {len(FEATURES)}")

    print("\nFeatures:")
    for i, feature in enumerate(FEATURES, start=1):
        print(f"  {i:2d}. {feature}")

    # --------------------------------------------------------
    # Verify expected temporal structure
    # --------------------------------------------------------

    print("\nTraining seasons:")
    print(
        sorted(
            train_df[SEASON].dropna().unique()
        )
    )

    print("\nValidation seasons:")
    print(
        sorted(
            validation_df[SEASON].dropna().unique()
        )
    )

    print("\nTest seasons:")
    print(
        sorted(
            test_df[SEASON].dropna().unique()
        )
    )

    # --------------------------------------------------------
    # Display temporal folds
    # --------------------------------------------------------

    print("\nTemporal CV folds:")

    for fold in TEMPORAL_FOLDS:
        print(
            f"  Fold {fold['fold']}: "
            f"{min(fold['train_seasons'])}-"
            f"{max(fold['train_seasons'])}"
            f" -> "
            f"{fold['validation_seasons']}"
        )

    # --------------------------------------------------------
    # Generate hyperparameter configurations
    # --------------------------------------------------------

    print("\nGenerating hyperparameter configurations...")

    parameter_list = list(
        ParameterSampler(
            PARAM_DISTRIBUTIONS,
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"  Configurations: {len(parameter_list)}"
    )

    print(
        f"  Temporal folds: {len(TEMPORAL_FOLDS)}"
    )

    print(
        f"  Total model fits: "
        f"{len(parameter_list) * len(TEMPORAL_FOLDS)}"
    )

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    print("\nStarting temporal hyperparameter search...")
    print("-" * 70)

    all_results = []
    aggregate_results = []

    for iteration, params in enumerate(
        parameter_list,
        start=1,
    ):

        print(
            f"\nConfiguration "
            f"{iteration}/{len(parameter_list)}"
        )

        print(
            "Parameters:"
        )

        for key, value in params.items():
            print(
                f"  {key}: {value}"
            )

        fold_df = run_temporal_cv(
            train_df,
            params,
        )

        # --------------------------------------------
        # Save fold-level results
        # --------------------------------------------

        for _, row in fold_df.iterrows():

            result = {
                "iteration": iteration,
                **params,
                **row.to_dict(),
            }

            all_results.append(result)

        # --------------------------------------------
        # Aggregate across temporal folds
        # --------------------------------------------

        aggregate = {
            "iteration": iteration,
            **params,
        }

        for metric in [
            "log_loss",
            "brier_score",
            "roc_auc",
            "accuracy",
            "balanced_accuracy",
        ]:

            aggregate[
                f"mean_{metric}"
            ] = fold_df[metric].mean()

            aggregate[
                f"std_{metric}"
            ] = fold_df[metric].std(
                ddof=0
            )

            aggregate[
                f"min_{metric}"
            ] = fold_df[metric].min()

            aggregate[
                f"max_{metric}"
            ] = fold_df[metric].max()

        aggregate_results.append(
            aggregate
        )

        print(
            f"Temporal CV Log Loss: "
            f"{aggregate['mean_log_loss']:.6f}"
        )

    # --------------------------------------------------------
    # Create results DataFrames
    # --------------------------------------------------------

    fold_results_df = pd.DataFrame(
        all_results
    )

    aggregate_results_df = pd.DataFrame(
        aggregate_results
    )

    # Rank by temporal CV Log Loss
    aggregate_results_df = (
        aggregate_results_df
        .sort_values(
            "mean_log_loss",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    aggregate_results_df[
        "rank"
    ] = np.arange(
        1,
        len(aggregate_results_df) + 1,
    )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    fold_results_path = (
        OUTPUT_DIR
        / "temporal_cv_fold_results.csv"
    )

    aggregate_results_path = (
        OUTPUT_DIR
        / "tuning_results.csv"
    )

    fold_results_df.to_csv(
        fold_results_path,
        index=False,
    )

    aggregate_results_df.to_csv(
        aggregate_results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Select best configuration
    # --------------------------------------------------------

    best_row = (
        aggregate_results_df
        .iloc[0]
    )

    best_iteration = int(
        best_row["iteration"]
    )

    best_params = parameter_list[
        best_iteration - 1
    ]

    print("\n" + "=" * 70)
    print("BEST TEMPORAL CV CONFIGURATION")
    print("=" * 70)

    print(
        f"\nTemporal CV Log Loss: "
        f"{best_row['mean_log_loss']:.6f}"
    )

    print("\nBest parameters:")

    for key, value in best_params.items():
        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Save best parameters
    # --------------------------------------------------------

    best_params_df = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value in best_params.items()
        ]
    )

    best_params_df.to_csv(
        OUTPUT_DIR
        / "best_params.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Train final model on ALL 2015-2022 data
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    final_pipeline = build_pipeline(
        best_params
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        final_pipeline.fit(
            X_train,
            y_train,
        )

    # --------------------------------------------------------
    # Official validation evaluation
    # --------------------------------------------------------

    print("\nEvaluating official validation set...")

    X_validation = validation_df[
        FEATURES
    ]

    y_validation = validation_df[
        TARGET
    ]

    validation_probabilities = (
        final_pipeline
        .predict_proba(X_validation)[:, 1]
    )

    validation_predictions = (
        validation_probabilities >= 0.5
    ).astype(int)

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    print("\nValidation metrics:")

    for metric, value in validation_metrics.items():
        print(
            f"  {metric}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Official test evaluation
    # --------------------------------------------------------

    print("\nEvaluating official test set...")

    X_test = test_df[FEATURES]

    y_test = test_df[TARGET]

    test_probabilities = (
        final_pipeline
        .predict_proba(X_test)[:, 1]
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    print("\nTest metrics:")

    for metric, value in test_metrics.items():
        print(
            f"  {metric}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR
        / "model.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_path,
    )

    # --------------------------------------------------------
    # Save feature list
    # --------------------------------------------------------

    feature_list_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_group": [
                group
                for group, feature_list
                in FEATURE_GROUPS.items()
                for _ in feature_list
            ],
        }
    )

    feature_list_df.to_csv(
        OUTPUT_DIR
        / "feature_list.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    validation_output = (
        validation_df[
            [GAME_ID, SEASON]
        ].copy()
    )

    validation_output[
        "win_home_actual"
    ] = y_validation.values

    validation_output[
        "win_home_probability"
    ] = validation_probabilities

    validation_output[
        "win_home_prediction"
    ] = validation_predictions

    validation_output[
        "split"
    ] = "validation"

    validation_output.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_output = (
        test_df[
            [GAME_ID, SEASON]
        ].copy()
    )

    test_output[
        "win_home_actual"
    ] = y_test.values

    test_output[
        "win_home_probability"
    ] = test_probabilities

    test_output[
        "win_home_prediction"
    ] = test_predictions

    test_output[
        "split"
    ] = "test"

    test_output.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    rf_model = (
        final_pipeline
        .named_steps["model"]
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                rf_model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df[
        "rank"
    ] = np.arange(
        1,
        len(importance_df) + 1,
    )

    importance_df.to_csv(
        OUTPUT_DIR
        / "feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "model": "random_forest_model_5",
                "feature_count": len(FEATURES),
                "train_rows": len(train_df),
                "validation_rows": len(
                    validation_df
                ),
                "test_rows": len(test_df),
                "train_seasons": ",".join(
                    map(
                        str,
                        sorted(
                            train_df[
                                SEASON
                            ]
                            .unique()
                        ),
                    )
                ),
                "validation_seasons": ",".join(
                    map(
                        str,
                        sorted(
                            validation_df[
                                SEASON
                            ]
                            .unique()
                        ),
                    )
                ),
                "test_seasons": ",".join(
                    map(
                        str,
                        sorted(
                            test_df[
                                SEASON
                            ]
                            .unique()
                        ),
                    )
                ),
                "temporal_cv_folds": len(
                    TEMPORAL_FOLDS
                ),
                "tuning_iterations": N_ITER,
                "internal_temporal_cv_log_loss": (
                    best_row[
                        "mean_log_loss"
                    ]
                ),
                "validation_log_loss": (
                    validation_metrics[
                        "log_loss"
                    ]
                ),
                "validation_brier_score": (
                    validation_metrics[
                        "brier_score"
                    ]
                ),
                "validation_roc_auc": (
                    validation_metrics[
                        "roc_auc"
                    ]
                ),
                "validation_accuracy": (
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_balanced_accuracy": (
                    validation_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "test_log_loss": (
                    test_metrics[
                        "log_loss"
                    ]
                ),
                "test_brier_score": (
                    test_metrics[
                        "brier_score"
                    ]
                ),
                "test_roc_auc": (
                    test_metrics[
                        "roc_auc"
                    ]
                ),
                "test_accuracy": (
                    test_metrics[
                        "accuracy"
                    ]
                ),
                "test_balanced_accuracy": (
                    test_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "random_state": RANDOM_STATE,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "training_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL 5 COMPLETE")
    print("=" * 70)

    print(
        f"\nArtifacts saved to:\n"
        f"  {OUTPUT_DIR}"
    )

    print("\nSaved files:")

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        if path.is_file():
            print(
                f"  {path.name}"
            )


if __name__ == "__main__":
    main()